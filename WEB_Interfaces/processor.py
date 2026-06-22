"""
processor.py — Video processing pipeline.

Extracts skeleton (YOLO11s-pose), polyline (LK optical flow), and
X3D-XS visual features from a video file, then runs PolyGuidedFusion +
SLDA for binary anomaly detection.

This is the exact same pipeline as the Colab notebook, adapted for local use.
"""

import time, cv2, math
import numpy as np
import torch

# ── Skeleton constants ────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# Resample helper — match Colab's temporal resampling
# ═══════════════════════════════════════════════════════════════════════════════

def resample(feat, n):
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    T, D = feat.shape
    if T == 0:
        return np.zeros((n, D), dtype=np.float32)
    if T == n:
        return feat.astype(np.float32)
    if T == 1:
        r = np.repeat(feat, n, axis=0)
        return r + np.random.normal(0, 1e-4, r.shape).astype(np.float32)
    idx = np.linspace(0, T - 1, n)
    lo = np.floor(idx).astype(int).clip(0, T - 1)
    hi = np.ceil(idx).astype(int).clip(0, T - 1)
    alpha = (idx - lo)[:, None]
    return (feat[lo] * (1 - alpha) + feat[hi] * alpha).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Skeleton extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _angle(a, b, c):
    ba, bc = a - b, c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.5
    return float(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1., 1.)) / np.pi)


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


def _kpts_to_vec(kpts_np, box_xyxy, fh, fw, num_persons):
    W = float(box_xyxy[2] - box_xyxy[0])
    H = float(box_xyxy[3] - box_xyxy[1])
    if W < 1. or H < 1.:
        return np.zeros(DIM_STATIC, dtype=np.float32)
    norm = np.zeros((17, 2), dtype=np.float32)
    for i in range(17):
        norm[i, 0] = (kpts_np[i, 0] - box_xyxy[0]) / W
        norm[i, 1] = (kpts_np[i, 1] - box_xyxy[1]) / H
    vec = np.zeros(DIM_STATIC, dtype=np.float32)
    for idx, (i, j) in enumerate(LIMB_PAIRS):
        vec[idx*2] = norm[j, 0] - norm[i, 0]
        vec[idx*2+1] = norm[j, 1] - norm[i, 1]
    nv = np.linalg.norm(vec[:DIM_LIMB])
    if nv > 1e-6:
        vec[:DIM_LIMB] /= nv
    off = DIM_LIMB
    for i in range(17):
        vec[off+i] = kpts_np[i, 2] if kpts_np.shape[1] > 2 else 0.5
    off += DIM_CONF
    for ai, (i, j, k) in enumerate(ANGLE_TRIPLETS):
        vec[off+ai] = _angle(norm[i], norm[j], norm[k])
    off += DIM_ANGLE
    vec[off+0] = (box_xyxy[0] + box_xyxy[2]) / 2. / max(fw, 1.)
    vec[off+1] = (box_xyxy[1] + box_xyxy[3]) / 2. / max(fh, 1.)
    vec[off+2] = W / max(fw, 1.)
    vec[off+3] = H / max(fh, 1.)
    off += DIM_BBOX
    vec[off] = min(num_persons, 5) / 5.
    return vec


def extract_skeleton(video_path, yolo_model, frame_interval=16, conf=0.25,
                     batch_size=32, imgsz=640, device='cpu'):
    """Extract 96-dim skeleton features from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    fi = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if fi % frame_interval == 0:
            frames.append(f)
        fi += 1
    cap.release()
    if not frames:
        return None

    half = (device == 'cuda')
    static = []
    for s in range(0, len(frames), batch_size):
        batch = frames[s:s+batch_size]
        results = yolo_model(batch, verbose=False, conf=conf,
                             device=device, half=half, imgsz=imgsz)
        for idx, res in enumerate(results):
            fh, fw = batch[idx].shape[:2]
            if (res.keypoints is None or res.boxes is None):
                static.append(np.zeros(DIM_STATIC, dtype=np.float32))
                continue
            kp_data = res.keypoints.data
            if kp_data is None or kp_data.shape[0] == 0:
                static.append(np.zeros(DIM_STATIC, dtype=np.float32))
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            static.append(_kpts_to_vec(
                _safe_kpts(kp_data, best),
                boxes[best], fh, fw, len(boxes)
            ))

    if not static:
        return None
    st = np.stack(static, axis=0)
    limb = st[:, :DIM_LIMB]
    vel = np.zeros_like(limb)
    vel[1:] = limb[1:] - limb[:-1]
    vel /= np.linalg.norm(vel, axis=1, keepdims=True).clip(min=1e-6)
    return np.concatenate([st, vel], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Polyline extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_polylines(trajectories, frame_h, frame_w):
    feat = np.zeros(POLY_DIM, dtype=np.float32)
    if not trajectories:
        return feat
    H, W = max(frame_h, 1), max(frame_w, 1)
    all_speeds, all_dirs, all_curves, net_disps, loiter_ratios = [], [], [], [], []
    spatial_counts = np.zeros((2, 4), dtype=np.float32)

    for traj in trajectories:
        if len(traj) < 2:
            continue
        traj = np.array(traj, dtype=np.float32)
        diffs = np.diff(traj, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)
        all_speeds.extend(speeds.tolist())
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        all_dirs.extend(angles.tolist())
        if len(diffs) > 1:
            ad = np.diff(angles)
            ad = (ad + np.pi) % (2 * np.pi) - np.pi
            all_curves.extend(np.abs(ad).tolist())
        net = traj[-1] - traj[0]
        net_disps.append(np.array([net[0]/W, net[1]/H]))
        mid = traj[len(traj)//2]
        gy = int(np.clip(mid[1]/H*2, 0, 1))
        gx = int(np.clip(mid[0]/W*4, 0, 3))
        spatial_counts[gy, gx] += 1
        path_len = speeds.sum() + 1e-6
        net_dist = np.linalg.norm(net) + 1e-6
        loiter_ratios.append(1.0 - min(net_dist / path_len, 1.0))

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
        feat[32] = float(nd[:, 0].mean()); feat[33] = float(nd[:, 1].mean())
        feat[34] = float(nd[:, 0].std());  feat[35] = float(nd[:, 1].std())
        feat[36] = float(nd[:, 0].max());  feat[37] = float(nd[:, 1].max())
        feat[38] = float(np.abs(nd[:, 0]).mean())
        feat[39] = float(np.abs(nd[:, 1]).mean())
    feat[40:48] = (spatial_counts / (spatial_counts.sum() + 1e-6)).flatten()
    if all_speeds:
        n = len(all_speeds)
        bs = max(n // 8, 1)
        for b in range(8):
            sl = all_speeds[b*bs:(b+1)*bs] if b < 7 else all_speeds[b*bs:]
            feat[48+b] = float(np.mean(sl)) if sl else 0.
    n_tracks = len(trajectories)
    feat[56] = float(n_tracks) / max(80, 1)
    feat[57] = float(np.mean([len(t) for t in trajectories])) / 16.
    feat[58] = float(np.std([len(t) for t in trajectories])) if trajectories else 0
    h_vals = [len(t) / 16. for t in trajectories]
    if h_vals:
        h_arr = np.array(h_vals).clip(1e-9, 1.)
        feat[59] = float(-np.sum(h_arr * np.log(h_arr)) / np.log(len(h_arr) + 2))
    if loiter_ratios:
        lr = np.array(loiter_ratios)
        feat[60] = float(lr.mean()); feat[61] = float(lr.max())
        feat[62] = float(lr.std());  feat[63] = float((lr > 0.7).mean())
    return feat


def extract_polylines(video_path, frame_interval=16):
    """Extract 64-dim polyline features from a video using Lucas-Kanade flow."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    segment_features = []
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    fh, fw = prev_gray.shape
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)
    if p0 is None:
        p0 = np.zeros((0, 1, 2), dtype=np.float32)
    segment_trajs = [[(pt[0, 0], pt[0, 1])] for pt in p0]
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(p0) > 0:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
            if p1 is not None:
                good_new = p1[st == 1]
                good_idx = np.where(st.flatten() == 1)[0]
                for ni, oi in enumerate(good_idx):
                    if oi < len(segment_trajs):
                        segment_trajs[oi].append(
                            (float(good_new[ni, 0]), float(good_new[ni, 1]))
                        )
                p0 = good_new.reshape(-1, 1, 2)
            else:
                p0 = np.zeros((0, 1, 2), dtype=np.float32)
        if frame_idx > 0 and frame_idx % frame_interval == 0:
            segment_features.append(_encode_polylines(segment_trajs, fh, fw))
            p0 = cv2.goodFeaturesToTrack(gray, mask=None, **FEATURE_PARAMS)
            if p0 is None:
                p0 = np.zeros((0, 1, 2), dtype=np.float32)
            segment_trajs = [[(pt[0, 0], pt[0, 1])] for pt in p0]
        prev_gray = gray
        frame_idx += 1

    if segment_trajs and any(len(t) > 1 for t in segment_trajs):
        segment_features.append(_encode_polylines(segment_trajs, fh, fw))
    cap.release()
    return np.stack(segment_features, axis=0) if segment_features else None


# ═══════════════════════════════════════════════════════════════════════════════
# X3D-XS extraction
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_x3d(frame, cs=160):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    frame = frame[y0:y0+side, x0:x0+side]
    frame = cv2.resize(frame, (cs, cs), interpolation=cv2.INTER_LINEAR)
    return (frame.astype(np.float32) / 255. - MEAN_X3D) / STD_X3D


def extract_x3d(video_path, x3d_extractor, device='cpu',
                frame_interval=16, clip_frames=4, crop_size=160, batch_clips=8):
    """Extract 192-dim X3D-XS visual features from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    sample_idx = set(np.linspace(0, frame_interval - 1, clip_frames, dtype=int))
    all_feats, clip_batch, cur_seg = [], [], []
    fc = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (fc % frame_interval) in sample_idx:
            cur_seg.append(preprocess_x3d(frame, crop_size))
        fc += 1
        if fc % frame_interval == 0:
            if len(cur_seg) == clip_frames:
                clip_batch.append(np.stack(cur_seg, axis=0))
            cur_seg = []
            if len(clip_batch) == batch_clips:
                t = torch.from_numpy(np.stack(clip_batch, 0)).permute(0, 4, 1, 2, 3).to(device)
                with torch.no_grad():
                    all_feats.append(x3d_extractor(t).cpu().numpy())
                clip_batch = []

    if clip_batch:
        t = torch.from_numpy(np.stack(clip_batch, 0)).permute(0, 4, 1, 2, 3).to(device)
        with torch.no_grad():
            all_feats.append(x3d_extractor(t).cpu().numpy())
    cap.release()
    return np.concatenate(all_feats, 0) if all_feats else None


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline — video → prediction
# ═══════════════════════════════════════════════════════════════════════════════

def process_video(video_path, model_manager, seg_num=32):
    """
    Full pipeline: video → skeleton + polyline + x3d → fusion → SLDA → result.

    Returns dict with prediction, score, latencies, or error.
    """
    mm = model_manager
    result = {
        'video': str(video_path),
        'status': 'processing',
        'timings': {},
    }
    total_t0 = time.perf_counter()

    # 1. Skeleton extraction
    t0 = time.perf_counter()
    try:
        device = mm.device
        sk_feat = extract_skeleton(
            video_path, mm.yolo_model,
            frame_interval=16, conf=0.25, batch_size=32, imgsz=640,
            device=device
        )
        if sk_feat is None:
            sk_feat = np.zeros((1, SKEL_DIM), dtype=np.float32)
    except Exception as e:
        sk_feat = np.zeros((1, SKEL_DIM), dtype=np.float32)
        result['skeleton_error'] = str(e)
    result['timings']['skeleton_ms'] = (time.perf_counter() - t0) * 1000

    # 2. Polyline extraction
    t0 = time.perf_counter()
    try:
        po_feat = extract_polylines(video_path, frame_interval=16)
        if po_feat is None:
            po_feat = np.zeros((1, POLY_DIM), dtype=np.float32)
    except Exception as e:
        po_feat = np.zeros((1, POLY_DIM), dtype=np.float32)
        result['polyline_error'] = str(e)
    result['timings']['polyline_ms'] = (time.perf_counter() - t0) * 1000

    # 3. X3D extraction
    t0 = time.perf_counter()
    try:
        x3_feat = extract_x3d(
            video_path, mm.x3d_extractor, device=device,
            frame_interval=16, clip_frames=4, crop_size=160, batch_clips=8
        )
        if x3_feat is None:
            x3_feat = np.zeros((1, X3D_DIM), dtype=np.float32)
    except Exception as e:
        x3_feat = np.zeros((1, X3D_DIM), dtype=np.float32)
        result['x3d_error'] = str(e)
    result['timings']['x3d_ms'] = (time.perf_counter() - t0) * 1000

    # 4. Resample to fixed segments
    sk_t = torch.from_numpy(resample(sk_feat, seg_num)).unsqueeze(0).to(device)
    po_t = torch.from_numpy(resample(po_feat, seg_num)).unsqueeze(0).to(device)
    vi_t = torch.from_numpy(resample(x3_feat, seg_num)).unsqueeze(0).to(device)

    # 5. Fusion + SLDA
    t0 = time.perf_counter()
    mm.fusion_model.eval()
    with torch.no_grad():
        _, emb = mm.fusion_model(sk_t, po_t, vi_t)
    prob = mm.slda.predict_proba(emb.cpu().numpy())[0]
    result['timings']['inference_ms'] = (time.perf_counter() - t0) * 1000

    pred_idx = int(prob.argmax())
    anom_score = float(prob[1])

    result['status'] = 'done'
    result['prediction'] = mm.BINARY_NAMES[pred_idx]
    result['pred_label'] = pred_idx
    result['anom_score'] = anom_score
    result['confidence'] = float(prob[pred_idx])
    result['probs'] = {'Normal': float(prob[0]), 'Anomalous': float(prob[1])}
    result['timings']['total_ms'] = (time.perf_counter() - total_t0) * 1000

    return result


def check_stream_source(source):
    """Check if a video source (file path, RTSP, or camera index) is accessible."""
    try:
        if str(source).isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {'reachable': False, 'error': 'Cannot open source'}
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {'reachable': False, 'error': 'Cannot read frame'}
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            'reachable': True,
            'resolution': f'{w}x{h}',
            'fps': round(fps, 1),
            'total_frames': total if total > 0 else None,
        }
    except Exception as e:
        return {'reachable': False, 'error': str(e)}
