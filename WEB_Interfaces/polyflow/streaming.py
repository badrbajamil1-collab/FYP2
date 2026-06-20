import collections, threading, queue, time, os, cv2, torch, contextlib
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .models import PolyGuidedFusion, StreamingLDA
from .features import SkeletonExtractor, PolylineExtractor, X3DExtractor, resample

# ── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class Alert:
    timestamp:        float
    camera_id:        str
    is_anomalous:     bool
    anomaly_score:    float
    event_class:      str
    event_confidence: float
    stage1_ms:        float
    stage2_ms:        float
    total_ms:         float
    frame_idx:        int

@dataclass
class StreamStats:
    camera_id:     str
    fps:           float
    stage1_lat_ms: float
    stage2_lat_ms: float
    total_lat_ms:  float
    segments:      int
    anomalies:     int
    s2_triggers:   int

CLASS_NAMES = [
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","RoadAccidents","Robbery","Shooting","Shoplifting",
    "Stealing","Vandalism","Other"
]

# ── Threads ─────────────────────────────────────────────────────────────────

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
            while len(self._frames) >= self.segment_size * 2:
                for _ in range(self.segment_size):
                    if self._frames: self._frames.popleft()
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
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened(): return
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval = max(1, int(round(src_fps / self.target_fps)))
        self._t0 = time.time(); fi = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                if self.source_type == "file": cap.set(cv2.CAP_PROP_POS_FRAMES, 0); fi = 0; continue
                else:
                    cap.release(); time.sleep(3.0)
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    fi = 0; continue
            fi += 1
            if fi % interval != 0: continue
            frame = cv2.resize(frame, self.resize)
            self.buffer.push(frame); self._frames += 1
        cap.release()

    def stop(self): self._stop.set()
    @property
    def fps(self): return self._frames / max(time.time() - self._t0, 1e-6)

class ProcessThread(threading.Thread):
    def __init__(self, camera_id, buffer, det_model, slda_binary, slda_14class, 
                 skel_ext, poly_ext, x3d_ext, device, alert_queue, gpu_lock,
                 anomaly_threshold=0.5, seg_num=16):
        super().__init__(daemon=True, name=f"process-{camera_id}")
        self.camera_id, self.buffer, self.det_model, self.slda_binary, self.slda_14class = camera_id, buffer, det_model, slda_binary, slda_14class
        self.skel_ext, self.poly_ext, self.x3d_ext, self.device, self.alert_queue, self.gpu_lock = skel_ext, poly_ext, x3d_ext, device, alert_queue, gpu_lock
        self.anomaly_threshold, self.seg_num = anomaly_threshold, seg_num
        self._stop = threading.Event(); self._segments = 0; self._anomalies = 0; self._s2_triggers = 0; self._frame_idx = 0
        self._s1_lat_sum = 0.0; self._s2_lat_sum = 0.0; self._tot_lat_sum = 0.0

    def run(self):
        while not self._stop.is_set():
            segment = self.buffer.pop_segment(timeout=0.5)
            if segment is None: continue
            try:
                alert = self._process_segment(segment, time.perf_counter())
                if alert and alert.is_anomalous:
                    try: self.alert_queue.put_nowait(alert)
                    except queue.Full: pass
            except Exception: pass
            self._segments += 1; self._frame_idx += len(segment)

    def _process_segment(self, frames, t0) -> Alert:
        sk = resample(self.skel_ext.extract(frames), self.seg_num)
        po = resample(self.poly_ext.extract(frames), self.seg_num)
        snap_frame = None
        
        # BUG 1 FIX: Skip X3D on CPU - it is too slow for real-time
        if self.device == "cpu":
            vi = np.zeros((self.seg_num, 192), dtype=np.float32)
        else:
            vi = resample(self.x3d_ext.extract(frames), self.seg_num)
            
        with self.gpu_lock:
            sk_t, po_t, vi_t = [torch.from_numpy(x).float().unsqueeze(0).to(self.device) for x in [sk, po, vi]]
            self.det_model.eval()
            with torch.no_grad(): _, emb = self.det_model(sk_t, po_t, vi_t)
            t_s1 = time.perf_counter(); bin_probs = self.slda_binary.predict_proba(emb.cpu().numpy())[0]
            stage1_ms = (time.perf_counter() - t_s1) * 1000
            anom_score = float(bin_probs[1]); is_anomalous = anom_score > self.anomaly_threshold
            event_class, event_confidence, stage2_ms = "—", 0.0, 0.0
            if is_anomalous: snap_frame = frames[len(frames)//2].copy()
            if is_anomalous and self.slda_14class is not None:
                t_s2 = time.perf_counter(); rec_model = getattr(self.slda_14class, "_rec_model", None)
                if rec_model:
                    rec_model.eval()
                    with torch.no_grad(): _, emb2 = rec_model(sk_t, po_t, vi_t)
                    s2_emb = emb2.cpu().numpy()
                else: s2_emb = emb.cpu().numpy()
                cls_probs = self.slda_14class.predict_proba(s2_emb)[0]
                pred_idx = int(cls_probs.argmax()); event_class = CLASS_NAMES[pred_idx]
                event_confidence = float(cls_probs[pred_idx]); stage2_ms = (time.perf_counter() - t_s2) * 1000
                self._s2_triggers += 1
        if snap_frame is not None:
            os.makedirs("alert_snapshots", exist_ok=True)
            cv2.imwrite(f"alert_snapshots/{self.camera_id}_{int(time.time())}.jpg", snap_frame)
        total_ms = (time.perf_counter() - t0) * 1000
        self._s1_lat_sum += stage1_ms; self._s2_lat_sum += stage2_ms; self._tot_lat_sum += total_ms
        if is_anomalous: self._anomalies += 1
        return Alert(time.time(), self.camera_id, is_anomalous, anom_score, event_class, event_confidence, stage1_ms, stage2_ms, total_ms, self._frame_idx)

    @property
    def avg_stage1_ms(self): return self._s1_lat_sum / max(self._segments, 1)
    @property
    def avg_total_ms(self): return self._tot_lat_sum / max(self._segments, 1)

# ── Manager ─────────────────────────────────────────────────────────────────

class PolyFlowStreamManager:
    def __init__(self, det_model, slda_binary, slda_14class=None, device="cuda", anomaly_threshold=0.5, segment_size=16, seg_num=16, target_fps=25, yolo_model="yolo11s-pose.pt"):
        self.det_model, self.slda_binary, self.slda_14class = det_model, slda_binary, slda_14class
        self.device = device if torch.cuda.is_available() else "cpu"
        self.anomaly_threshold, self.segment_size, self.seg_num, self.target_fps = anomaly_threshold, segment_size, seg_num, target_fps
        
        # BUG 2 FIX: On CPU, no shared GPU memory - lock is unnecessary
        self._gpu_lock = threading.Lock() if torch.cuda.is_available() else contextlib.nullcontext()
        
        os.makedirs("alert_snapshots", exist_ok=True)
        self._skel_ext = SkeletonExtractor(yolo_model, device=self.device)
        self._poly_ext = PolylineExtractor(); self._x3d_ext = X3DExtractor(device=self.device)
        self.alerts = queue.Queue(maxsize=2000); self._streams = {}; self._running = False

    def add_stream(self, camera_id, source):
        if camera_id in self._streams: return
        buf = FrameBuffer(self.segment_size); cap = CaptureThread(camera_id, source, buf, target_fps=self.target_fps)
        proc = ProcessThread(camera_id, buf, self.det_model, self.slda_binary, self.slda_14class, self._skel_ext, self._poly_ext, self._x3d_ext, self.device, self.alerts, self._gpu_lock, self.anomaly_threshold, self.seg_num)
        self._streams[camera_id] = {"capture": cap, "process": proc}; 
        if self._running: cap.start(); proc.start()

    def remove_stream(self, camera_id):
        s = self._streams.pop(camera_id, None)
        if s: s["capture"].stop(); s["process"].stop()

    def start(self):
        self._running = True
        for s in self._streams.values(): s["capture"].start(); s["process"].start()

    def stop(self):
        self._running = False
        for s in self._streams.values(): s["capture"].stop(); s["process"].stop()

    def get_alert(self, timeout=1.0):
        try: return self.alerts.get(timeout=timeout)
        except queue.Empty: return None

    def get_stats(self):
        return {cid: StreamStats(cid, s["capture"].fps, s["process"].avg_stage1_ms, 0.0, s["process"].avg_total_ms, 0, 0, 0) for cid, s in self._streams.items()}
