"""
PolyFlow Stream Manager v2 — Two-Stage Cascaded Pipeline
=========================================================
Stage 1 (always running):  Binary anomaly detection
    PolyGuidedFusion  →  SLDA binary  →  Normal / Anomalous

Stage 2 (triggered when anomaly_score > threshold):
    Same PolyGuidedFusion embedding  →  SLDA 14-class  →  exact event type

Architecture per segment:
  Skeleton  (YOLO11s-pose  → 96-dim)  ──┐
  Polyline  (LK Optical Flow → 64-dim) ──┤→ PolyGuidedFusion → embedding (256-dim)
  Visual    (X3D-XS         → 192-dim) ──┘
                                              ├→ Stage-1 SLDA (binary)   → alert
                                              └→ Stage-2 SLDA (14-class) → event type (if anomalous)

"""

import threading
import queue
import time
import collections
import math
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────────────────────────────
# UCF-Crime class definitions
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "Normal_Videos", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]
NORMAL_IDX: int = CLASS_NAMES.index("Normal_Videos")
BINARY_NAMES: List[str] = ["Normal", "Anomalous"]

# ─────────────────────────────────────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────────────────────────────────────

class PosEnc(nn.Module):
    def __init__(self, d: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class PolyGuidedFusion(nn.Module):
    def __init__(
        self,
        sk_dim: int = 96,
        po_dim: int = 64,
        vis_dim: int = 192,
        d: int = 256,
        heads: int = 4,
        layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sk_proj  = nn.Sequential(nn.Linear(sk_dim,  d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.po_proj  = nn.Sequential(nn.Linear(po_dim,  d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.vis_proj = nn.Sequential(nn.Linear(vis_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))

        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d)
        self.cross_ff   = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))
        self.ff_norm    = nn.LayerNorm(d)

        self.sk_mod  = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.vis_mod = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pos_enc   = PosEnc(d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d // 2, num_classes)
        )

    def forward(self, sk, po, vi) -> Tuple[torch.Tensor, torch.Tensor]:
        B = sk.size(0)
        s = self.sk_proj(sk)
        p = self.po_proj(po)
        v = self.vis_proj(vi)

        attn, _ = self.cross_attn(query=p, key=s, value=s)
        p2 = self.cross_norm(p + attn)
        p2 = self.ff_norm(p2 + self.cross_ff(p2))
        p2 = p2 + self.sk_mod
        v  = v  + self.vis_mod

        cls = self.cls_token.expand(B, -1, -1)
        x = self.pos_enc(torch.cat([cls, p2, v], dim=1))
        z = self.drop(self.encoder(x)[:, 0])
        return self.head(z), z

class StreamingLDA(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, shrinkage: float = 1e-4, device: str = "cuda"):
        super().__init__()
        dev = device if torch.cuda.is_available() else "cpu"
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.shrinkage   = shrinkage
        self.device      = dev

        self.muK   = torch.zeros(num_classes, input_dim).to(dev)
        self.cK    = torch.zeros(num_classes).to(dev)
        self.Sigma = torch.eye(input_dim).to(dev)
        self.num_updates = 0
        self._Lambda: Optional[torch.Tensor] = None

    def _invalidate(self):
        self._Lambda = None

    @property
    def Lambda(self):
        if self._Lambda is None:
            reg = (1 - self.shrinkage) * self.Sigma + self.shrinkage * torch.eye(self.input_dim, device=self.device)
            self._Lambda = torch.linalg.pinv(reg)
        return self._Lambda

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch: int = 1024) -> np.ndarray:
        E   = torch.from_numpy(X).float().to(self.device)
        N   = E.shape[0]
        Lam = self.Lambda
        W   = Lam @ self.muK.T
        c   = 0.5 * (self.muK.T * W).sum(0)
        scores = torch.empty(N, self.num_classes, device=self.device)
        for s in range(0, N, batch):
            e = min(s + batch, N)
            scores[s:e] = E[s:e] @ W - c
        return torch.softmax(scores, dim=1).cpu().numpy().astype(np.float32)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "muK": self.muK.cpu(), "cK": self.cK.cpu(), "Sigma": self.Sigma.cpu(),
            "num_updates": self.num_updates, "input_dim": self.input_dim,
            "num_classes": self.num_classes, "shrinkage": self.shrinkage,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.muK   = ck["muK"].to(self.device)
        self.cK    = ck["cK"].to(self.device)
        self.Sigma = ck["Sigma"].to(self.device)
        self.num_updates = ck["num_updates"]
        self._invalidate()

def load_polyflow_twostage(
    fusion_det_ckpt:   str,
    slda_binary_ckpt:  str,
    fusion_rec_ckpt:   Optional[str] = None,
    slda_14class_ckpt: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[PolyGuidedFusion, StreamingLDA, Optional[StreamingLDA]]:
    dev = device if torch.cuda.is_available() else "cpu"

    ck = torch.load(fusion_det_ckpt, map_location=dev, weights_only=False)
    nc1 = ck["model"]["head.3.weight"].shape[0] if "head.3.weight" in ck["model"] else 2
    det_model = PolyGuidedFusion(num_classes=nc1).to(dev)
    det_model.load_state_dict(ck["model"])
    det_model.eval()
    print(f"[Stage-1] PolyGuidedFusion (binary) loaded epoch={ck.get('epoch','?')} AUC={ck.get('best_auc', '?')}")

    slda_binary = StreamingLDA(256, 2, device=dev)
    slda_binary.load(slda_binary_ckpt)
    print(f"[Stage-1] SLDA binary loaded")

    slda_14class: Optional[StreamingLDA] = None
    if slda_14class_ckpt is not None:
        slda_14class = StreamingLDA(256, 14, device=dev)
        slda_14class.load(slda_14class_ckpt)
        print(f"[Stage-2] SLDA 14-class loaded")

        if fusion_rec_ckpt is not None:
            ck2 = torch.load(fusion_rec_ckpt, map_location=dev, weights_only=False)
            nc2 = ck2["model"]["head.3.weight"].shape[0] if "head.3.weight" in ck2["model"] else 14
            rec_model = PolyGuidedFusion(num_classes=nc2).to(dev)
            rec_model.load_state_dict(ck2["model"])
            rec_model.eval()
            print(f"[Stage-2] PolyGuidedFusion (14-class) loaded epoch={ck2.get('epoch','?')} F1={ck2.get('best_f1','?')}")
            slda_14class._rec_model = rec_model
        else:
            slda_14class._rec_model = None
            print(f"[Stage-2] Reusing Stage-1 backbone for 14-class embeddings")
    else:
        print(f"[Stage-2] Disabled (no 14-class checkpoint provided)")

    return det_model, slda_binary, slda_14class

@dataclass
class Alert:
    timestamp:        float
    camera_id:        str
    is_anomalous:     bool
    anomaly_score:    float
    event_class:      str = "—"
    event_confidence: float = 0.0
    frame_idx:        int   = 0
    stage1_ms:        float = 0.0
    stage2_ms:        float = 0.0
    total_ms:         float = 0.0

@dataclass
class StreamStats:
    camera_id:          str
    fps:                float = 0.0
    stage1_lat_ms:      float = 0.0
    stage2_lat_ms:      float = 0.0
    total_lat_ms:       float = 0.0
    frames_captured:    int   = 0
    segments_processed: int   = 0
    anomalies_detected: int   = 0
    stage2_triggered:   int   = 0
    source_type:        str   = "unknown"
    status:             str   = "idle"

    @property
    def stage2_trigger_rate(self) -> float:
        if self.segments_processed == 0: return 0.0
        return self.stage2_triggered / self.segments_processed

LIMB_PAIRS = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
ANGLE_TRIPLETS = [(5,7,9),(6,8,10),(7,5,11),(8,6,12),(5,11,13),(6,12,14),(11,13,15),(12,14,16),(0,5,6),(11,12,0)]
DIM_LIMB, DIM_CONF, DIM_ANGLE, DIM_BBOX, DIM_COUNT = 32, 17, 10, 4, 1
DIM_STATIC = DIM_LIMB + DIM_CONF + DIM_ANGLE + DIM_BBOX + DIM_COUNT
SKEL_DIM   = DIM_STATIC + DIM_LIMB
POLY_DIM   = 64
X3D_DIM    = 192

LK_PARAMS       = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS  = dict(maxCorners=80, qualityLevel=0.01, minDistance=10, blockSize=7)
MEAN_X3D        = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD_X3D         = np.array([0.225, 0.225, 0.225], dtype=np.float32)

def _angle(a, b, c):
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6: return 0.5
    return float(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)) / np.pi)

def _kpts_to_vec(kpts, box_xyxy, fh, fw, num_persons):
    W, H = float(box_xyxy[2] - box_xyxy[0]), float(box_xyxy[3] - box_xyxy[1])
    if W < 1.0 or H < 1.0: return np.zeros(DIM_STATIC, dtype=np.float32)
    norm = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        norm[i, 0] = (kpts[i, 0] - box_xyxy[0]) / W
        norm[i, 1] = (kpts[i, 1] - box_xyxy[1]) / H
    vec = np.zeros(DIM_STATIC, dtype=np.float32)
    for idx, (i, j) in enumerate(LIMB_PAIRS):
        vec[idx * 2]     = norm[j, 0] - norm[i, 0]
        vec[idx * 2 + 1] = norm[j, 1] - norm[i, 1]
    nv = np.linalg.norm(vec[:DIM_LIMB])
    if nv > 1e-6: vec[:DIM_LIMB] /= nv
    off = DIM_LIMB
    for i in range(17): vec[off + i] = kpts[i, 2] if kpts.shape[1] > 2 else 0.5
    off += DIM_CONF
    for ai, (i, j, k) in enumerate(ANGLE_TRIPLETS): vec[off + ai] = _angle(norm[i], norm[j], norm[k])
    off += DIM_ANGLE
    vec[off]     = (box_xyxy[0] + box_xyxy[2]) / 2.0 / max(fw, 1)
    vec[off + 1] = (box_xyxy[1] + box_xyxy[3]) / 2.0 / max(fh, 1)
    vec[off + 2] = W / max(fw, 1)
    vec[off + 3] = H / max(fh, 1)
    off += DIM_BBOX
    vec[off] = min(num_persons, 5) / 5.0
    return vec

class SkeletonExtractor:
    def __init__(self, model_path="yolo11s-pose.pt", conf=0.25, imgsz=320, device="cuda"):
        from ultralytics import YOLO
        self.yolo   = YOLO(model_path)
        self.conf   = conf
        self.imgsz  = imgsz
        self.device = device
        self.half   = device == "cuda" and torch.cuda.is_available()

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        static = []
        for frame in frames:
            fh, fw = frame.shape[:2]
            results = self.yolo(frame, conf=self.conf, verbose=False, imgsz=self.imgsz, device=self.device, half=self.half)
            r = results[0]
            if r.keypoints is None or len(r.keypoints.data) == 0 or r.boxes is None:
                static.append(np.zeros(DIM_STATIC, dtype=np.float32))
                continue
            areas = ((r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])).cpu().numpy()
            best  = int(np.argmax(areas))
            kpts  = r.keypoints.data[best].cpu().numpy()
            box   = r.boxes.xyxy[best].cpu().numpy()
            static.append(_kpts_to_vec(kpts, box, fh, fw, len(r.boxes)))
        st   = np.stack(static, axis=0)
        limb = st[:, :DIM_LIMB]
        vel  = np.zeros_like(limb)
        vel[1:] = limb[1:] - limb[:-1]
        vel /= np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-6)
        return np.concatenate([st, vel], axis=1)

def _encode_polylines(trajectories, fh, fw):
    feat = np.zeros(POLY_DIM, dtype=np.float32)
    if not trajectories: return feat
    H, W = max(fh, 1), max(fw, 1)
    all_speeds, all_dirs, all_curves, net_disps, loiter_ratios = [], [], [], [], []
    spatial = np.zeros((2, 4), dtype=np.float32)
    for traj in trajectories:
        if len(traj) < 2: continue
        t = np.array(traj, dtype=np.float32)
        diffs = np.diff(t, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)
        all_speeds.extend(speeds.tolist())
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        all_dirs.extend(angles.tolist())
        if len(diffs) > 1:
            ad = (np.diff(angles) + np.pi) % (2 * np.pi) - np.pi
            all_curves.extend(np.abs(ad).tolist())
        net = t[-1] - t[0]
        net_disps.append(np.array([net[0] / W, net[1] / H]))
        mid = t[len(t) // 2]
        gy, gx  = int(np.clip(mid[1] / H * 2, 0, 1)), int(np.clip(mid[0] / W * 4, 0, 3))
        spatial[gy, gx] += 1
        pl, nd = speeds.sum() + 1e-6, np.linalg.norm(net) + 1e-6
        loiter_ratios.append(1.0 - min(nd / pl, 1.0))
    if all_speeds:
        h, _ = np.histogram(all_speeds, bins=16, range=(0, 50))
        feat[0:16] = h / (sum(all_speeds) + 1e-6)
    if all_dirs:
        h, _ = np.histogram(all_dirs, bins=8, range=(-np.pi, np.pi))
        feat[16:24] = h / (len(all_dirs) + 1e-6)
    if all_curves:
        h, _ = np.histogram(all_curves, bins=8, range=(0, np.pi))
        feat[24:32] = h / (len(all_curves) + 1e-6)
    if net_disps:
        nd = np.array(net_disps)
        feat[32], feat[33] = nd[:, 0].mean(), nd[:, 1].mean()
        feat[34], feat[35] = nd[:, 0].std(),  nd[:, 1].std()
        feat[36], feat[37] = nd[:, 0].max(),  nd[:, 1].max()
        feat[38], feat[39] = np.abs(nd[:, 0]).mean(), np.abs(nd[:, 1]).mean()
    feat[40:48] = (spatial / (spatial.sum() + 1e-6)).flatten()
    if all_speeds:
        n = len(all_speeds); bs = max(n // 8, 1)
        for b in range(8):
            sl = all_speeds[b * bs:(b + 1) * bs] if b < 7 else all_speeds[b * bs:]
            feat[48 + b] = float(np.mean(sl)) if sl else 0.0
    feat[56], feat[57] = len(trajectories) / max(80, 1), float(np.mean([len(t) for t in trajectories])) / max(16, 1)
    feat[58] = float(np.std([len(t) for t in trajectories])) if trajectories else 0.0
    h_vals = [len(t) / max(16, 1) for t in trajectories]
    if h_vals:
        ha = np.array(h_vals).clip(1e-9, 1.0)
        feat[59] = float(-np.sum(ha * np.log(ha)) / np.log(len(ha) + 2))
    if loiter_ratios:
        lr = np.array(loiter_ratios)
        feat[60], feat[61], feat[62], feat[63] = lr.mean(), lr.max(), lr.std(), float((lr > 0.7).mean())
    return feat

class PolylineExtractor:
    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames: return np.zeros((1, POLY_DIM), dtype=np.float32)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        fh, fw = prev_gray.shape
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        if p0 is None: p0 = np.zeros((0, 1, 2), dtype=np.float32)
        trajs = [[(float(pt[0, 0]), float(pt[0, 1]))] for pt in p0]
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(p0) > 0:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_idx = np.where(st.flatten() == 1)[0]
                    for ni, oi in enumerate(good_idx):
                        if oi < len(trajs): trajs[oi].append((float(good_new[ni, 0, 0]), float(good_new[ni, 0, 1])))
                    p0 = good_new.reshape(-1, 1, 2)
                else: p0 = np.zeros((0, 1, 2), dtype=np.float32)
            prev_gray = gray
        return _encode_polylines(trajs, fh, fw).reshape(1, -1)

class X3DExtractor:
    def __init__(self, device="cuda", crop_size=160, clip_frames=4):
        from pytorchvideo.models.hub import x3d_xs
        self.device      = device if torch.cuda.is_available() else "cpu"
        self.crop_size   = crop_size
        self.clip_frames = clip_frames

        class _E(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.stages = nn.ModuleList(list(base.blocks[:-1]))
                self.pool   = nn.AdaptiveAvgPool3d(1)
            def forward(self, x):
                for s in self.stages: x = s(x)
                return self.pool(x).flatten(1)
        base = x3d_xs(pretrained=True); base.eval()
        self.model = _E(base).eval().to(self.device)

    def _pre(self, frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = f.shape[:2]; side = min(h, w)
        f = f[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
        f = cv2.resize(f, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        return (f.astype(np.float32) / 255.0 - MEAN_X3D) / STD_X3D

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames: return np.zeros((1, X3D_DIM), dtype=np.float32)
        indices = np.linspace(0, len(frames) - 1, self.clip_frames, dtype=int)
        clip = np.stack([self._pre(frames[i]) for i in indices], axis=0)
        t = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        with torch.no_grad(): feat = self.model(t).cpu().numpy().squeeze()
        return feat.reshape(1, -1)

def resample(feat: np.ndarray, n: int) -> np.ndarray:
    if feat.ndim == 1: feat = feat.reshape(1, -1)
    T, D = feat.shape
    if T == 0: return np.zeros((n, D), dtype=np.float32)
    if T == n: return feat.astype(np.float32)
    if T == 1:
        r = np.repeat(feat, n, axis=0)
        return r + np.random.normal(0, 1e-4, r.shape).astype(np.float32)
    idx   = np.linspace(0, T - 1, n)
    lo    = np.floor(idx).astype(int).clip(0, T - 1)
    hi    = np.ceil(idx).astype(int).clip(0, T - 1)
    alpha = (idx - lo)[:, None]
    return (feat[lo] * (1 - alpha) + feat[hi] * alpha).astype(np.float32)

class FrameBuffer:
    def __init__(self, segment_size: int = 16, max_segments: int = 4):
        self.segment_size = segment_size
        self._frames = collections.deque(maxlen=segment_size * max_segments)
        self._lock   = threading.Lock()
        self._ready  = threading.Event()

    def push(self, frame: np.ndarray):
        with self._lock:
            self._frames.append(frame)
            if len(self._frames) >= self.segment_size: self._ready.set()

    def pop_segment(self, timeout=1.0) -> Optional[List[np.ndarray]]:
        if not self._ready.wait(timeout=timeout): return None
        with self._lock:
            # BUG 3 FIX: Drop stale segments if we're falling behind
            while len(self._frames) >= self.segment_size * 2:
                for _ in range(self.segment_size):
                    if self._frames:
                        self._frames.popleft() # discard stale segment
            
            if len(self._frames) < self.segment_size:
                self._ready.clear()
                return None
            seg = [self._frames.popleft() for _ in range(self.segment_size)]
            if len(self._frames) < self.segment_size: self._ready.clear()
            return seg

class CaptureThread(threading.Thread):
    def __init__(self, camera_id, source, buffer: FrameBuffer, target_fps=25, resize=(320, 240)):
        super().__init__(daemon=True, name=f"capture-{camera_id}")
        self.camera_id   = camera_id
        self.source      = source
        self.buffer      = buffer
        self.target_fps  = target_fps
        self.resize      = resize
        self._stop       = threading.Event()
        self._frames     = 0
        self._t0         = 0.0
        self.source_type = self._detect(source)

    @staticmethod
    def _detect(source) -> str:
        if isinstance(source, int): return "webcam"
        s = str(source).lower()
        if s.startswith("rtsp://") or s.endswith(".m3u8") or s.startswith("http"): return "rtsp"
        return "file"

    def run(self):
        import os
        # BUG 1 FIX: Set low-latency FFmpeg options BEFORE opening
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        )
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # minimize internal buffer
        
        if not cap.isOpened(): return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(src_fps / self.target_fps)))
        self._t0 = time.time()
        fi = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if self.source_type == "file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); fi = 0; continue
                else:
                    # BUG 2 FIX: RTSP dropped - release and retry with backoff
                    cap.release()
                    print(f"[{self.camera_id}] Stream dropped, reconnecting in 3s...")
                    time.sleep(3.0)
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fi = 0
                    continue
            fi += 1
            if fi % interval != 0: continue
            frame = cv2.resize(frame, self.resize)
            self.buffer.push(frame)
            self._frames += 1
        cap.release()

    def stop(self): self._stop.set()
    @property
    def fps(self): return self._frames / max(time.time() - self._t0, 1e-6)
    @property
    def frames_captured(self): return self._frames

class ProcessThread(threading.Thread):
    def __init__(
        self,
        camera_id:         str,
        buffer:            FrameBuffer,
        det_model:         PolyGuidedFusion,
        slda_binary:       StreamingLDA,
        slda_14class:      Optional[StreamingLDA],
        skel_ext:          SkeletonExtractor,
        poly_ext:          PolylineExtractor,
        x3d_ext:           X3DExtractor,
        device:            str,
        alert_queue:       queue.Queue,
        gpu_lock:          threading.Lock,
        anomaly_threshold: float = 0.5,
        seg_num:           int   = 16,
    ):
        super().__init__(daemon=True, name=f"process-{camera_id}")
        self.camera_id         = camera_id
        self.buffer            = buffer
        self.det_model         = det_model
        self.slda_binary       = slda_binary
        self.slda_14class      = slda_14class
        self.skel_ext          = skel_ext
        self.poly_ext          = poly_ext
        self.x3d_ext           = x3d_ext
        self.device            = device if torch.cuda.is_available() else "cpu"
        self.alert_queue       = alert_queue
        self.gpu_lock          = gpu_lock
        self.anomaly_threshold = anomaly_threshold
        self.seg_num           = seg_num
        self.stage2_enabled    = False   # Controlled by PolyFlowStreamManager

        self._stop        = threading.Event()
        self._segments    = 0
        self._anomalies   = 0
        self._s2_triggers = 0
        self._frame_idx   = 0

        self._s1_lat_sum  = 0.0
        self._s2_lat_sum  = 0.0
        self._tot_lat_sum = 0.0

    def run(self):
        while not self._stop.is_set():
            segment = self.buffer.pop_segment(timeout=0.5)
            if segment is None: continue
            t_seg_start = time.perf_counter()
            try:
                alert = self._process_segment(segment, t_seg_start)
                if alert and alert.is_anomalous:
                    try: self.alert_queue.put_nowait(alert)
                    except queue.Full: pass
            except Exception as e: pass
            self._segments  += 1
            self._frame_idx += len(segment)

    def _process_segment(self, frames: List[np.ndarray], t0: float) -> Alert:
        skel_feat = self.skel_ext.extract(frames)
        poly_feat = self.poly_ext.extract(frames)
        sk = resample(skel_feat, self.seg_num)
        po = resample(poly_feat, self.seg_num)

        with self.gpu_lock:
            vi_raw = self.x3d_ext.extract(frames)
            vi = resample(vi_raw, self.seg_num)

            sk_t = torch.from_numpy(sk).float().unsqueeze(0).to(self.device)
            po_t = torch.from_numpy(po).float().unsqueeze(0).to(self.device)
            vi_t = torch.from_numpy(vi).float().unsqueeze(0).to(self.device)

            self.det_model.eval()
            with torch.no_grad(): _, emb = self.det_model(sk_t, po_t, vi_t)

            t_s1 = time.perf_counter()
            bin_probs = self.slda_binary.predict_proba(emb.cpu().numpy())[0]
            stage1_ms = (time.perf_counter() - t_s1) * 1000

            anom_score  = float(bin_probs[1])
            is_anomalous = anom_score > self.anomaly_threshold

            event_class      = "—"
            event_confidence = 0.0
            stage2_ms        = 0.0

            if is_anomalous and self.stage2_enabled and self.slda_14class is not None:
                t_s2 = time.perf_counter()
                rec_model = getattr(self.slda_14class, "_rec_model", None)
                if rec_model is not None:
                    rec_model.eval()
                    with torch.no_grad(): _, emb2 = rec_model(sk_t, po_t, vi_t)
                    s2_emb = emb2.cpu().numpy()
                else:
                    s2_emb = emb.cpu().numpy()

                cls_probs        = self.slda_14class.predict_proba(s2_emb)[0]
                pred_idx         = int(cls_probs.argmax())
                event_class      = CLASS_NAMES[pred_idx]
                event_confidence = float(cls_probs[pred_idx])
                stage2_ms        = (time.perf_counter() - t_s2) * 1000
                self._s2_triggers += 1

        total_ms = (time.perf_counter() - t0) * 1000
        self._s1_lat_sum  += stage1_ms
        self._s2_lat_sum  += stage2_ms
        self._tot_lat_sum += total_ms
        if is_anomalous: self._anomalies += 1

        if is_anomalous:
            # Save snapshot OUTSIDE the GPU lock
            snap_path = f"alert_snapshots/{self.camera_id}_{int(time.time())}.jpg"
            mid_idx = len(frames) // 2
            cv2.imwrite(snap_path, frames[mid_idx])

        return Alert(
            timestamp        = time.time(),
            camera_id        = self.camera_id,
            is_anomalous     = is_anomalous,
            anomaly_score    = anom_score,
            event_class      = event_class,
            event_confidence = event_confidence,
            frame_idx        = self._frame_idx,
            stage1_ms        = stage1_ms,
            stage2_ms        = stage2_ms,
            total_ms         = total_ms,
        )

    def stop(self): self._stop.set()
    @property
    def avg_stage1_ms(self): return self._s1_lat_sum / max(self._segments, 1)
    @property
    def avg_stage2_ms(self): return self._s2_lat_sum / max(self._s2_triggers, 1)
    @property
    def avg_total_ms(self): return self._tot_lat_sum / max(self._segments, 1)
    @property
    def segments_processed(self): return self._segments
    @property
    def anomalies_detected(self): return self._anomalies
    @property
    def stage2_triggered(self): return self._s2_triggers

class PolyFlowStreamManager:
    def __init__(
        self,
        det_model:         PolyGuidedFusion,
        slda_binary:       StreamingLDA,
        slda_14class:      Optional[StreamingLDA] = None,
        device:            str   = "cuda",
        anomaly_threshold: float = 0.5,
        segment_size:      int   = 16,
        seg_num:           int   = 16,
        target_fps:        int   = 25,
        resize:            Tuple[int, int] = (320, 240),
        yolo_model:        str   = "yolo11s-pose.pt",
        yolo_conf:         float = 0.25,
        skel_ext:          Optional[SkeletonExtractor] = None,
        poly_ext:          Optional[PolylineExtractor] = None,
        x3d_ext:           Optional[X3DExtractor] = None,
    ):
        self.det_model         = det_model
        self.slda_binary       = slda_binary
        self.slda_14class      = slda_14class
        self.device            = device if torch.cuda.is_available() else "cpu"
        self.anomaly_threshold = anomaly_threshold
        self.segment_size      = segment_size
        self.seg_num           = seg_num
        self.target_fps        = target_fps
        self.resize            = resize
        self._gpu_lock = threading.Lock()
        # Use pre-loaded extractors if provided, otherwise create new ones
        self._skel_ext = skel_ext if skel_ext is not None else SkeletonExtractor(yolo_model, yolo_conf, device=self.device)
        self._poly_ext = poly_ext if poly_ext is not None else PolylineExtractor()
        self._x3d_ext  = x3d_ext  if x3d_ext  is not None else X3DExtractor(device=self.device)
        self.alerts: queue.Queue = queue.Queue(maxsize=2000)
        self._streams: Dict[str, dict] = {}
        self._running    = False
        self._start_time = 0.0
        self.stage2_enabled = False   # Off by default
        os.makedirs("alert_snapshots", exist_ok=True)

    def add_stream(self, camera_id: str, source) -> None:
        if camera_id in self._streams: return
        buf  = FrameBuffer(self.segment_size, max_segments=4)
        cap  = CaptureThread(camera_id, source, buf, target_fps=self.target_fps, resize=self.resize)
        proc = ProcessThread(
            camera_id, buf, self.det_model, self.slda_binary, self.slda_14class,
            self._skel_ext, self._poly_ext, self._x3d_ext, self.device, self.alerts, self._gpu_lock,
            anomaly_threshold=self.anomaly_threshold, seg_num=self.seg_num,
        )
        self._streams[camera_id] = {"source": source, "source_type": cap.source_type, "buffer": buf, "capture": cap, "process": proc}
        proc.stage2_enabled = self.stage2_enabled
        if self._running: cap.start(); proc.start()

    def remove_stream(self, camera_id: str) -> None:
        s = self._streams.pop(camera_id, None)
        if not s: return
        s["capture"].stop(); s["process"].stop()
        s["capture"].join(timeout=3); s["process"].join(timeout=3)

    def start(self) -> None:
        self._running    = True
        self._start_time = time.time()
        for s in self._streams.values(): s["capture"].start(); s["process"].start()

    def stop(self) -> None:
        self._running = False
        for s in self._streams.values(): s["capture"].stop(); s["process"].stop()
        for s in self._streams.values(): s["capture"].join(timeout=3); s["process"].join(timeout=3)

    def get_alert(self, timeout: float = 1.0) -> Optional[Alert]:
        try: return self.alerts.get(timeout=timeout)
        except queue.Empty: return None

    def get_stats(self) -> Dict[str, StreamStats]:
        return {cid: StreamStats(
            camera_id=cid, fps=s["capture"].fps, stage1_lat_ms=s["process"].avg_stage1_ms,
            stage2_lat_ms=s["process"].avg_stage2_ms, total_lat_ms=s["process"].avg_total_ms,
            frames_captured=s["capture"].frames_captured, segments_processed=s["process"].segments_processed,
            anomalies_detected=s["process"].anomalies_detected, stage2_triggered=s["process"].stage2_triggered,
            source_type=s["source_type"], status="running" if s["capture"].is_alive() else "stopped"
        ) for cid, s in self._streams.items()}
