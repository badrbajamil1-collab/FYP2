import cv2, torch, time
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Optional

# ── Constants ──────────────────────────────────────────────────────────────

LIMB_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
]
ANGLE_TRIPLETS = [
    (5,7,9),(6,8,10),(7,5,11),(8,6,12),(5,11,13),
    (6,12,14),(11,13,15),(12,14,16),(0,5,6),(11,12,0)
]
DIM_LIMB, DIM_CONF, DIM_ANGLE, DIM_BBOX, DIM_COUNT = 32, 17, 10, 4, 1
DIM_STATIC = DIM_LIMB + DIM_CONF + DIM_ANGLE + DIM_BBOX + DIM_COUNT  # 64
SKEL_DIM = DIM_STATIC + DIM_LIMB  # 96
POLY_DIM = 64
X3D_DIM  = 192

MEAN_X3D = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD_X3D  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

LK_PARAMS = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS = dict(maxCorners=80, qualityLevel=0.01, minDistance=10, blockSize=7)

# ── Helpers ────────────────────────────────────────────────────────────────

def resample(feat: np.ndarray, n: int) -> np.ndarray:
    if feat is None: return np.zeros((n, 1), dtype=np.float32)
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

def _safe_kpts(kpts_tensor, idx: int) -> np.ndarray:
    """Safely extract a single person's keypoints as a 2D (17, 3) numpy array.
    Handles both 2D (17, 3) and 3D (N, 17, 3) tensors from YOLO."""
    arr = kpts_tensor.cpu().numpy() if hasattr(kpts_tensor, 'cpu') else np.array(kpts_tensor)
    if arr.ndim == 3:
        return arr[idx]   # (N, 17, 3) — pick person
    elif arr.ndim == 2:
        return arr        # Already (17, 3)
    else:
        return np.zeros((17, 3), dtype=np.float32)

def _angle(a, b, c):
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6: return 0.5
    return float(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1., 1.)) / np.pi)

def _kpts_to_vec(kpts_np, box_xyxy, fh, fw, num_persons):
    W, H = float(box_xyxy[2] - box_xyxy[0]), float(box_xyxy[3] - box_xyxy[1])
    if W < 1. or H < 1.: return np.zeros(DIM_STATIC, dtype=np.float32)
    norm = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        norm[i, 0] = (kpts_np[i, 0] - box_xyxy[0]) / W
        norm[i, 1] = (kpts_np[i, 1] - box_xyxy[1]) / H
    vec = np.zeros(DIM_STATIC, dtype=np.float32)
    for idx, (i, j) in enumerate(LIMB_PAIRS):
        vec[idx*2] = norm[j, 0] - norm[i, 0]
        vec[idx*2+1] = norm[j, 1] - norm[i, 1]
    nv = np.linalg.norm(vec[:DIM_LIMB])
    if nv > 1e-6: vec[:DIM_LIMB] /= nv
    off = DIM_LIMB
    for i in range(17): vec[off+i] = kpts_np[i, 2] if kpts_np.shape[1] > 2 else 0.5
    off += DIM_CONF
    for ai, (i, j, k) in enumerate(ANGLE_TRIPLETS): vec[off+ai] = _angle(norm[i], norm[j], norm[k])
    off += DIM_ANGLE
    vec[off+0] = (box_xyxy[0] + box_xyxy[2]) / 2. / max(fw, 1.)
    vec[off+1] = (box_xyxy[1] + box_xyxy[3]) / 2. / max(fh, 1.)
    vec[off+2] = W / max(fw, 1.)
    vec[off+3] = H / max(fh, 1.)
    off += DIM_BBOX
    vec[off] = min(num_persons, 5) / 5.
    return vec

def _encode_polylines(trajectories, fh, fw):
    feat = np.zeros(POLY_DIM, dtype=np.float32)
    if not trajectories: return feat
    H, W = max(fh, 1), max(fw, 1)
    all_speeds, all_dirs, all_curves, net_disps, loiter_ratios = [], [], [], [], []
    spatial_counts = np.zeros((2, 4), dtype=np.float32)
    for traj in trajectories:
        if len(traj) < 2: continue
        traj = np.array(traj, dtype=np.float32)
        diffs = np.diff(traj, axis=0); speeds = np.linalg.norm(diffs, axis=1)
        all_speeds.extend(speeds.tolist()); angles = np.arctan2(diffs[:, 1], diffs[:, 0]); all_dirs.extend(angles.tolist())
        if len(diffs) > 1:
            ad = (np.diff(angles) + np.pi) % (2 * np.pi) - np.pi
            all_curves.extend(np.abs(ad).tolist())
        net = traj[-1] - traj[0]; net_disps.append(np.array([net[0]/W, net[1]/H]))
        mid = traj[len(traj)//2]; gy, gx = int(np.clip(mid[1]/H*2, 0, 1)), int(np.clip(mid[0]/W*4, 0, 3)); spatial_counts[gy, gx] += 1
        path_len = speeds.sum() + 1e-6; net_dist = np.linalg.norm(net) + 1e-6; loiter_ratios.append(1.0 - min(net_dist / path_len, 1.0))
    if all_speeds:
        h, _ = np.histogram(all_speeds, bins=16, range=(0, 50))
        feat[0:16] = h.astype(np.float32) / (sum(all_speeds) + 1e-6)
    if all_dirs:
        h, _ = np.histogram(all_dirs, bins=8, range=(-np.pi, np.pi))
        feat[16:24] = h.astype(np.float32) / (len(all_dirs) + 1e-6)
    if all_curves:
        h, _ = np.histogram(all_curves, bins=8, range=(0, np.pi))
        feat[24:32] = h.astype(np.float32) / (len(all_curves) + 1e-6)
    if net_disps:
        nd = np.array(net_disps)
        feat[32], feat[33], feat[34], feat[35] = nd[:, 0].mean(), nd[:, 1].mean(), nd[:, 0].std(), nd[:, 1].std()
        feat[36], feat[37], feat[38], feat[39] = nd[:, 0].max(), nd[:, 1].max(), np.abs(nd[:, 0]).mean(), np.abs(nd[:, 1]).mean()
    feat[40:48] = (spatial_counts / (spatial_counts.sum() + 1e-6)).flatten()
    if all_speeds:
        n, bs = len(all_speeds), max(len(all_speeds) // 8, 1)
        for b in range(8):
            sl = all_speeds[b*bs:(b+1)*bs] if b < 7 else all_speeds[b*bs:]
            feat[48+b] = float(np.mean(sl)) if sl else 0.
    feat[56] = float(len(trajectories)) / 80.
    feat[57] = float(np.mean([len(t) for t in trajectories])) / 16. if trajectories else 0
    feat[58] = float(np.std([len(t) for t in trajectories])) if trajectories else 0
    h_vals = [len(t) / 16. for t in trajectories]
    if h_vals:
        h_arr = np.array(h_vals).clip(1e-9, 1.)
        feat[59] = float(-np.sum(h_arr * np.log(h_arr)) / np.log(len(h_arr) + 2))
    if loiter_ratios:
        lr = np.array(loiter_ratios); feat[60], feat[61], feat[62], feat[63] = lr.mean(), lr.max(), lr.std(), (lr > 0.7).mean()
    return feat

# ── Extractors ─────────────────────────────────────────────────────────────

class SkeletonExtractor:
    def __init__(self, model_or_path, conf=0.25, device="cuda"):
        from ultralytics import YOLO
        self.model  = model_or_path if isinstance(model_or_path, YOLO) else YOLO(model_or_path)
        self.conf   = conf
        self.device = device if torch.cuda.is_available() else "cpu"

    def extract(self, frames: List[np.ndarray], batch_size=32, imgsz=320) -> np.ndarray:
        if not frames: return np.zeros((1, SKEL_DIM), dtype=np.float32)
        
        # Optimization: use small imgsz for CPU
        results = self.model(
            frames, verbose=False, conf=self.conf, device=self.device, 
            half=(self.device=="cuda"), batch=batch_size, imgsz=imgsz
        )
        static = []
        for idx, res in enumerate(results):
            fh, fw = frames[idx].shape[:2]
            if res.keypoints is None or res.boxes is None:
                static.append(np.zeros(DIM_STATIC, dtype=np.float32)); continue
            kp_data = res.keypoints.data
            if kp_data is None or kp_data.shape[0] == 0:
                static.append(np.zeros(DIM_STATIC, dtype=np.float32)); continue
            boxes = res.boxes.xyxy.cpu().numpy(); areas = (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])
            best  = int(np.argmax(areas))
            static.append(_kpts_to_vec(_safe_kpts(kp_data, best), boxes[best], fh, fw, len(boxes)))
        st = np.stack(static, axis=0); limb = st[:, :DIM_LIMB]; vel = np.zeros_like(limb)
        vel[1:] = limb[1:] - limb[:-1]; vel /= np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-6)
        return np.concatenate([st, vel], axis=1)

class PolylineExtractor:
    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames: return np.zeros((1, POLY_DIM), dtype=np.float32)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY); fh, fw = prev_gray.shape
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
        if p0 is None: p0 = np.zeros((0, 1, 2), dtype=np.float32)
        trajs = [[(float(pt[0, 0]), float(pt[0, 1]))] for pt in p0]
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if len(p0) > 0:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
                if p1 is not None:
                    good_new = p1[st == 1]; good_idx = np.where(st.flatten() == 1)[0]
                    for ni, oi in enumerate(good_idx):
                        if oi < len(trajs): 
                            trajs[oi].append((float(good_new[ni, 0]), float(good_new[ni, 1])))
                    p0 = good_new.reshape(-1, 1, 2)
                else: p0 = np.zeros((0, 1, 2), dtype=np.float32)
            prev_gray = gray
        return _encode_polylines(trajs, fh, fw).reshape(1, -1)

class X3DExtractor:
    def __init__(self, model_or_none=None, device="cuda", crop_size=160, clip_frames=4):
        self.device, self.crop_size, self.clip_frames = device if torch.cuda.is_available() else "cpu", crop_size, clip_frames
        if model_or_none: self.model = model_or_none
        else:
            from pytorchvideo.models.hub import x3d_xs
            class _E(nn.Module):
                def __init__(self, base):
                    super().__init__(); self.stages = nn.ModuleList(list(base.blocks[:-1])); self.pool = nn.AdaptiveAvgPool3d(1)
                def forward(self, x):
                    for s in self.stages: x = s(x)
                    return self.pool(x).flatten(1)
            self.model = _E(x3d_xs(pretrained=True)).eval().to(self.device)

    def _pre(self, frame):
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w = f.shape[:2]; side = min(h, w)
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

# ── Standalone functions for Full-Video ──────────────────────────────────

def extract_skeleton_full(video_path, yolo_model, frame_interval=16, device="cuda", batch_size=32, imgsz=320):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    frames, fi = [], 0
    while True:
        ret, f = cap.read()
        if not ret: break
        if fi % frame_interval == 0: frames.append(f)
        fi += 1
    cap.release()
    if not frames: return None
    if hasattr(yolo_model, 'extract'):
        # If it's already an extractor (e.g. from stream_manager or manager)
        # Note: the stream_manager SkeletonExtractor might not accept batch_size/imgsz as kwargs to extract()
        # but the polyflow one does. We'll just try to pass them, or fallback.
        try:
            return yolo_model.extract(frames, batch_size=batch_size, imgsz=imgsz)
        except TypeError:
            return yolo_model.extract(frames)
    ext = SkeletonExtractor(yolo_model, device=device)
    return ext.extract(frames, batch_size=batch_size, imgsz=imgsz)

def extract_polylines_full(video_path, frame_interval=16, poly_extractor=None):
    cap = cv2.VideoCapture(str(video_path)); ret, f = cap.read()
    if not ret: cap.release(); return None
    fh, fw = f.shape[:2]; cap.release()
    # Simple chunking for full-video
    cap = cv2.VideoCapture(str(video_path)); segments = []; cur = []; fi = 0
    ext = poly_extractor if poly_extractor is not None else PolylineExtractor()
    while True:
        ret, f = cap.read()
        if not ret: break
        cur.append(f)
        if len(cur) == frame_interval:
            segments.append(ext.extract(cur))
            cur = []
        fi += 1
    cap.release()
    return np.concatenate(segments, axis=0) if segments else None

def extract_x3d_full(video_path, x3d_model, device="cuda", frame_interval=16):
    cap = cv2.VideoCapture(str(video_path)); segments = []; cur = []; fi = 0
    if hasattr(x3d_model, 'extract'):
        ext = x3d_model
    else:
        ext = X3DExtractor(x3d_model, device=device)
    while True:
        ret, f = cap.read()
        if not ret: break
        cur.append(f)
        if len(cur) == frame_interval:
            segments.append(ext.extract(cur))
            cur = []
        fi += 1
    cap.release()
    return np.concatenate(segments, axis=0) if segments else None
