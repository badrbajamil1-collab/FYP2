import time, cv2, torch
import numpy as np
from pathlib import Path
from .features import extract_skeleton_full, extract_polylines_full, extract_x3d_full, resample, SKEL_DIM, POLY_DIM, X3D_DIM

def process_video(video_path, model_manager, seg_num=32, stage2_enabled=False):
    """Full-video inference pipeline."""
    mm = model_manager; device = mm.device
    is_cpu = (device == "cpu")
    result = {'video': str(video_path), 'status': 'processing', 'timings': {}}
    t0_total = time.perf_counter()

    # 1. Extraction
    t0 = time.perf_counter()
    # BUG 3 FIX: Use smaller image size and batch=1 on CPU
    sk_feat = extract_skeleton_full(
        video_path, mm.yolo_model, 
        device=device,
        batch_size=1 if is_cpu else 32,
        imgsz=320 if is_cpu else 640
    )
    if sk_feat is None: sk_feat = np.zeros((1, SKEL_DIM), dtype=np.float32)
    result['timings']['skeleton_ms'] = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    po_feat = extract_polylines_full(video_path, poly_extractor=mm.poly_extractor)
    if po_feat is None: po_feat = np.zeros((1, POLY_DIM), dtype=np.float32)
    result['timings']['polyline_ms'] = (time.perf_counter()-t0)*1000

    t0 = time.perf_counter()
    # X3D is slow on CPU, but for video upload we keep it (user expects waiting)
    vi_feat = extract_x3d_full(video_path, mm.x3d_extractor, device=device)
    if vi_feat is None: vi_feat = np.zeros((1, X3D_DIM), dtype=np.float32)
    result['timings']['x3d_ms'] = (time.perf_counter()-t0)*1000

    # 2. Resample & Predict
    sk_t, po_t, vi_t = [torch.from_numpy(resample(x, seg_num)).unsqueeze(0).to(device) for x in [sk_feat, po_feat, vi_feat]]
    
    t0 = time.perf_counter()
    mm.fusion_model.eval()
    with torch.no_grad(): _, emb = mm.fusion_model(sk_t, po_t, vi_t)
    prob = mm.slda.predict_proba(emb.cpu().numpy())[0]
    result['timings']['inference_ms'] = (time.perf_counter()-t0)*1000

    pred_idx = int(prob.argmax())
    anom_score = float(prob[1])
    prediction = mm.BINARY_NAMES[pred_idx]
    confidence = float(prob[pred_idx])

    # Run Stage 2 Event Recognition if anomalous AND enabled
    if stage2_enabled and pred_idx == 1 and mm.slda_14 is not None:
        t0_rec = time.perf_counter()
        if mm.rec_model is not None:
            mm.rec_model.eval()
            with torch.no_grad(): _, emb_rec = mm.rec_model(sk_t, po_t, vi_t)
            emb_for_slda = emb_rec.cpu().numpy()
        else:
            emb_for_slda = emb.cpu().numpy()
        prob_14 = mm.slda_14.predict_proba(emb_for_slda)[0]
        rec_idx = int(prob_14.argmax())
        prediction = mm.CLASS_NAMES[rec_idx]
        confidence = float(prob_14[rec_idx])
        result['timings']['recognition_ms'] = (time.perf_counter()-t0_rec)*1000

    result.update({
        'status': 'done',
        'prediction': prediction,
        'pred_label': pred_idx,
        'anom_score': anom_score,
        'confidence': confidence,
        'probs': {
            'Normal': float(prob[0]),
            'Anomalous': float(prob[1])
        },
        'timings': {**result['timings'], 'total_ms': (time.perf_counter()-t0_total)*1000}
    })
    return result

def check_stream_source(source):
    """Utility to check if a source is reachable."""
    try:
        if str(source).isdigit(): source = int(source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened(): return {'reachable': False, 'error': 'Cannot open source'}
        ret, frame = cap.read()
        if not ret: cap.release(); return {'reachable': False, 'error': 'Cannot read frame'}
        h, w = frame.shape[:2]; fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return {'reachable': True, 'resolution': f'{w}x{h}', 'fps': round(fps, 1)}
    except Exception as e: return {'reachable': False, 'error': str(e)}
