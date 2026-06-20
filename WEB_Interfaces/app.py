"""
PolyFlow Dashboard — Flask + SQLite + Real Model Inference.

Features:
  - Upload video → real PolyGuidedFusion + SLDA inference
  - Add/check/remove live streams
  - Training history & predictions from SQLite
  - Model status page showing which components are loaded

Run:
    py init_db.py        # seed DB (one-time)
    py app.py            # starts on http://127.0.0.1:5000

Model checkpoints (place in ./checkpoints/):
    PolyGuided_AnomalyDetection_best.pt
    binary_slda_detection.pth
"""

import sqlite3, os, json, time, math, threading, torch
# ── CPU Optimization ──────────────────────────────────────────────────────
# Optimize PyTorch for CPU inference if no GPU is present
if not torch.cuda.is_available():
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(max(1, cpu_count - 1))
    torch.set_num_interop_threads(2)
    print(f"[CPU] Optimization active: using {torch.get_num_threads()} threads")

from pathlib import Path
from datetime import datetime
from flask import (Flask, g, render_template, jsonify, request,
                   redirect, url_for)
from werkzeug.utils import secure_filename

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB
BASE_DIR = Path(__file__).parent
DB_PATH  = str(BASE_DIR / "polyflow.db")
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
CKPT_DIR = BASE_DIR / "Model"
CKPT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.mpg', '.mpeg', '.webm'}

_model_manager = None
_model_lock = threading.Lock()

_stream_mgr = None
_stream_mgr_lock = threading.Lock()

_stage2_enabled = False   # Stage 2 (14-class) off by default

def get_model_manager():
    """Lazily load the unified model manager singleton."""
    global _model_manager
    if _model_manager is not None:
        return _model_manager
    with _model_lock:
        if _model_manager is not None:
            return _model_manager
        
        from polyflow.manager import ModelManager
        mm = ModelManager()
        mm.load_all(CKPT_DIR)
        _model_manager = mm
    return _model_manager


def get_stream_manager():
    global _stream_mgr
    if _stream_mgr is not None:
        return _stream_mgr
    with _stream_mgr_lock:
        if _stream_mgr is not None:
            return _stream_mgr
            
        from stream_manager import PolyFlowStreamManager
        
        mm = get_model_manager()
        
        _stream_mgr = PolyFlowStreamManager(
            det_model=mm.fusion_model,
            slda_binary=mm.slda,
            slda_14class=mm.slda_14,
            anomaly_threshold=0.8,
            yolo_model=str(CKPT_DIR / "yolo11s-pose.pt") if (CKPT_DIR / "yolo11s-pose.pt").exists() else "yolo11s-pose.pt",
            skel_ext=mm.yolo_model,
            poly_ext=mm.poly_extractor,
            x3d_ext=mm.x3d_extractor,
        )
        
        _stream_mgr.start()
        
        # Start background alert poller
        def alert_poller():
            with app.app_context():
                while True:
                    try:
                        alert = _stream_mgr.get_alert(timeout=1.0)
                        if alert and alert.is_anomalous:
                            db = open_db()
                            try:
                                db.execute(
                                    "INSERT INTO alerts (camera_id, timestamp, event_type, anom_score, latency_ms, frame_idx, acknowledged) "
                                    "VALUES (?, ?, ?, ?, ?, ?, 0)",
                                    (alert.camera_id, alert.timestamp, alert.event_class, alert.anomaly_score, alert.total_ms, alert.frame_idx)
                                )
                                db.execute("UPDATE streams SET alerts=alerts+1 WHERE camera_id=?", (alert.camera_id,))
                                db.commit()
                            except sqlite3.OperationalError as e:
                                print(f"[DB] alert_poller write failed: {e}")
                                db.rollback()
                            finally:
                                db.close()
                    except Exception as e:
                        print(f"[!] Alert poller thread error: {e}")
                        time.sleep(1.0)
        
        t = threading.Thread(target=alert_poller, daemon=True)
        t.start()

        def wal_checkpoint_worker():
            """Runs a WAL checkpoint every 5 minutes to keep WAL file small."""
            while True:
                time.sleep(300)
                try:
                    db = open_db()
                    db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    db.close()
                    print("[DB] WAL checkpoint completed")
                except Exception as e:
                    print(f"[DB] Checkpoint failed: {e}")

        t_ckpt = threading.Thread(target=wal_checkpoint_worker, daemon=True)
        t_ckpt.start()
        
    return _stream_mgr


# ── Database ───────────────────────────────────────────────────────────────

def open_db(path=None) -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL-compatible settings.
    Use this everywhere instead of sqlite3.connect() directly.
    """
    conn = sqlite3.connect(path or DB_PATH, timeout=5.0)
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def configure_db():
    """Enable WAL mode and performance pragmas once at startup."""
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA busy_timeout=3000")
    db.execute("PRAGMA temp_store=MEMORY")
    db.commit()
    db.close()
    print("[DB] WAL mode configured")


def get_db():
    if "db" not in g:
        g.db = open_db()
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


def ensure_tables():
    """Idempotently create all required tables and indexes."""
    db = open_db()
    db.execute("""
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY, name TEXT, task TEXT, params INTEGER,
            best_epoch INTEGER, best_auc REAL, best_f1 REAL, accuracy REAL,
            precision_ REAL, recall_ REAL, frame_auc REAL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS streams (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id   TEXT UNIQUE NOT NULL,
            source      TEXT NOT NULL,
            status      TEXT DEFAULT 'idle',
            fps         REAL DEFAULT 0,
            latency_ms  REAL DEFAULT 0,
            frames      INTEGER DEFAULT 0,
            segments    INTEGER DEFAULT 0,
            alerts      INTEGER DEFAULT 0,
            added_at    REAL DEFAULT 0,
            updated_at  TEXT DEFAULT (datetime('now'))
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id    TEXT NOT NULL,
            timestamp    REAL NOT NULL,
            event_type   TEXT DEFAULT 'Anomaly',
            anom_score   REAL DEFAULT 0,
            latency_ms   REAL DEFAULT 0,
            frame_idx    INTEGER DEFAULT 0,
            acknowledged INTEGER DEFAULT 0,
            snapshot_path TEXT
        )
    """)
    
    # Indexes
    db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_camera_time ON alerts(camera_id, timestamp DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_unacked ON alerts(acknowledged, anom_score) WHERE acknowledged = 0")
    
    db.commit()
    db.close()


# ── Pages ──────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    db = get_db()
    model = db.execute("SELECT * FROM models WHERE id=1").fetchone()
    if model:
        model = dict(model)
    else:
        model = {'name': 'Not loaded', 'best_auc': 0, 'accuracy': 0,
                 'best_f1': 0, 'frame_auc': 0, 'precision_': 0, 'recall_': 0,
                 'best_epoch': 0, 'params': 0}
    return render_template("dashboard.html", model=model)


@app.route("/training")
def training():
    return render_template("training.html")


@app.route("/predictions")
def predictions_page():
    return render_template("predictions.html")


@app.route("/streams")
def streams_page():
    return render_template("streams.html")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


# ── API endpoints ─────────────────────────────────────────────────────────

@app.route("/api/settings/threshold", methods=["POST"])
def api_set_threshold():
    val = request.json.get("threshold", 0.8)
    mgr = get_stream_manager()
    mgr.anomaly_threshold = float(val)
    # Also update all active threads
    for s in mgr._streams.values():
        s["process"].anomaly_threshold = float(val)
    return jsonify({"ok": True, "threshold": val})


@app.route("/api/settings/stage2", methods=["POST"])
def api_set_stage2():
    """Toggle Stage-2 (14-class event recognition) on/off."""
    global _stage2_enabled
    val = request.json.get("enabled", False)
    _stage2_enabled = bool(val)
    # Update stream manager threads
    mgr = get_stream_manager()
    mgr.stage2_enabled = _stage2_enabled
    for s in mgr._streams.values():
        s["process"].stage2_enabled = _stage2_enabled
    return jsonify({"ok": True, "stage2_enabled": _stage2_enabled})


@app.route("/api/settings/stage2")
def api_get_stage2():
    return jsonify({"stage2_enabled": _stage2_enabled})


@app.route("/api/model")
def api_model():
    db = get_db()
    row = db.execute("SELECT * FROM models WHERE id=1").fetchone()
    if row:
        return jsonify(dict(row))
    return jsonify({})


@app.route("/api/model-status")
def api_model_status():
    """Check which model components are loaded."""
    mm = get_model_manager()
    return jsonify(mm.get_status())


@app.route("/api/training-log")
def api_training_log():
    db = get_db()
    rows = db.execute(
        "SELECT epoch, loss, train_acc, val_f1, val_auc "
        "FROM training_log WHERE model_id=1 ORDER BY epoch"
    ).fetchall()
    return jsonify(rows_to_dicts(rows))


@app.route("/api/predictions")
def api_predictions():
    db = get_db()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 20, type=int)
    search = request.args.get("q", "").strip()
    filter_label = request.args.get("label", "")

    query = "SELECT * FROM predictions WHERE model_id=1"
    params = []
    if search:
        query += " AND stem LIKE ?"
        params.append(f"%{search}%")
    if filter_label in ("0", "1"):
        query += " AND true_label = ?"
        params.append(int(filter_label))
    query += " ORDER BY anom_score DESC"

    total = db.execute(
        query.replace("SELECT *", "SELECT COUNT(*)"), params
    ).fetchone()[0]

    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])
    rows = db.execute(query, params).fetchall()

    return jsonify({
        "data": rows_to_dicts(rows),
        "total": total,
        "page": page,
        "pages": math.ceil(total / per_page),
    })


@app.route("/api/confusion")
def api_confusion():
    db = get_db()
    matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    rows = db.execute(
        "SELECT true_label, pred_label, COUNT(*) as cnt "
        "FROM predictions WHERE model_id=1 "
        "GROUP BY true_label, pred_label"
    ).fetchall()
    for r in rows:
        t, p = r["true_label"], r["pred_label"]
        if t == 1 and p == 1:   matrix["TP"] = r["cnt"]
        elif t == 0 and p == 0: matrix["TN"] = r["cnt"]
        elif t == 0 and p == 1: matrix["FP"] = r["cnt"]
        elif t == 1 and p == 0: matrix["FN"] = r["cnt"]
    return jsonify(matrix)


@app.route("/api/score-distribution")
def api_score_dist():
    db = get_db()
    normals = db.execute(
        "SELECT anom_score FROM predictions WHERE model_id=1 AND true_label=0"
    ).fetchall()
    anomalous = db.execute(
        "SELECT anom_score FROM predictions WHERE model_id=1 AND true_label=1"
    ).fetchall()
    return jsonify({
        "normal": [r["anom_score"] for r in normals],
        "anomalous": [r["anom_score"] for r in anomalous],
    })


# ── Stream management ─────────────────────────────────────────────────────

@app.route("/api/streams")
def api_streams():
    db = get_db()
    rows = db.execute("SELECT * FROM streams ORDER BY camera_id").fetchall()
    
    # Update latest stats from stream_manager
    mgr = get_stream_manager()
    stats = mgr.get_stats()
    
    out = []
    for r in rows:
        d = dict(r)
        cid = d["camera_id"]
        if cid in stats:
            s = stats[cid]
            d["fps"] = s.fps
            d["latency_ms"] = s.total_lat_ms
            d["status"] = s.status
            db.execute("UPDATE streams SET fps=?, latency_ms=?, status=?, frames=? WHERE camera_id=?",
                       (s.fps, s.total_lat_ms, s.status, s.frames_captured, cid))
        out.append(d)
    try:
        db.commit()
    except sqlite3.OperationalError:
        db.rollback()
    
    return jsonify(out)


@app.route("/api/streams/add", methods=["POST"])
def api_add_stream():
    """Add a new stream. JSON body: {camera_id, source}."""
    data = request.get_json(force=True)
    cam_id = data.get("camera_id", "").strip()
    source = data.get("source", "").strip()
    if not cam_id or not source:
        return jsonify({"ok": False, "error": "camera_id and source are required"}), 400

    # Resolve YouTube URLs to direct stream links
    if "youtube.com" in source or "youtu.be" in source:
        try:
            import yt_dlp
            ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                source = info['url']
        except Exception as e:
            return jsonify({"ok": False, "error": f"YouTube extraction failed: {str(e)}"}), 400

    # Check if source is reachable
    from polyflow.inference import check_stream_source
    check = check_stream_source(source)

    db = get_db()
    existing = db.execute("SELECT id FROM streams WHERE camera_id=?", (cam_id,)).fetchone()
    if existing:
        return jsonify({"ok": False, "error": f"Camera '{cam_id}' already exists"}), 400

    status = 'running' if check['reachable'] else 'unreachable'
    fps = check.get('fps', 0) or 0
    db.execute(
        "INSERT INTO streams (camera_id, source, status, fps, latency_ms, frames, alerts, added_at) "
        "VALUES (?, ?, ?, ?, 0, 0, 0, ?)",
        (cam_id, source, status, fps, time.time())
    )
    try:
        db.commit()
    except sqlite3.OperationalError as e:
        db.rollback()
        return jsonify({"ok": False, "error": str(e)}), 503

    if check['reachable']:
        mgr = get_stream_manager()
        mgr.add_stream(cam_id, source)

    return jsonify({
        "ok": True,
        "camera_id": cam_id,
        "status": status,
        "check": check,
    })


@app.route("/api/streams/<camera_id>/check", methods=["POST"])
def api_check_stream(camera_id):
    """Check if a stream source is reachable."""
    db = get_db()
    row = db.execute("SELECT * FROM streams WHERE camera_id=?", (camera_id,)).fetchone()
    if not row:
        return jsonify({"ok": False, "error": "Stream not found"}), 404

    from polyflow.inference import check_stream_source
    check = check_stream_source(row['source'])

    new_status = 'running' if check['reachable'] else 'unreachable'
    db.execute("UPDATE streams SET status=? WHERE camera_id=?", (new_status, camera_id))
    try:
        db.commit()
    except sqlite3.OperationalError as e:
        db.rollback()
        return jsonify({"ok": False, "error": str(e)}), 503

    return jsonify({"ok": True, "check": check, "status": new_status})


@app.route("/api/streams/<camera_id>", methods=["DELETE"])
def api_delete_stream(camera_id):
    db = get_db()
    db.execute("DELETE FROM streams WHERE camera_id=?", (camera_id,))
    try:
        db.commit()
    except sqlite3.OperationalError as e:
        db.rollback()
        return jsonify({"ok": False, "error": str(e)}), 503
    
    mgr = get_stream_manager()
    mgr.remove_stream(camera_id)
    
    return jsonify({"ok": True})


# ── Video upload & processing ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
def api_upload_video():
    """Upload a video file and run real model inference."""
    if 'video' not in request.files:
        return jsonify({"ok": False, "error": "No video file uploaded"}), 400

    file = request.files['video']
    if not file.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"ok": False, "error": f"Invalid file type: {ext}"}), 400

    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / filename
    file.save(str(save_path))

    # Check model readiness
    mm = get_model_manager()
    if not mm.is_ready():
        missing = []
        if not mm.fusion_model: missing.append('PolyGuidedFusion')
        if not mm.slda: missing.append('SLDA')
        if not mm.yolo_model: missing.append('YOLO11s-pose')
        if not mm.x3d_extractor: missing.append('X3D-XS')
        return jsonify({
            "ok": False,
            "error": f"Model not fully loaded. Missing: {', '.join(missing)}",
            "model_status": mm.get_status(),
            "file_saved": str(save_path),
        }), 503

    # Run real inference
    from polyflow.inference import process_video
    try:
        result = process_video(str(save_path), mm, stage2_enabled=_stage2_enabled)

        # Save to DB
        db = get_db()
        stem = Path(filename).stem
        db.execute(
            "INSERT OR REPLACE INTO predictions "
            "(model_id, stem, true_label, pred_label, anom_score) "
            "VALUES (1, ?, -1, ?, ?)",
            (stem, result['pred_label'], result['anom_score'])
        )
        # Add alert if anomalous
        if result['anom_score'] > 0.5:
            db.execute(
                "INSERT INTO alerts "
                "(camera_id, timestamp, event_type, anom_score, latency_ms, "
                "frame_idx, acknowledged) VALUES (?, ?, ?, ?, ?, 0, 0)",
                ('upload', time.time(), result['prediction'],
                 result['anom_score'], result['timings']['total_ms'])
            )
        try:
            db.commit()
        except sqlite3.OperationalError as e:
            db.rollback()
            return jsonify({"ok": False, "error": str(e)}), 503

        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/process-stream", methods=["POST"])
def api_process_stream():
    """Grab latest stats from stream_manager."""
    data = request.get_json(force=True)
    camera_id = data.get("camera_id", "")

    mgr = get_stream_manager()
    stats = mgr.get_stats()
    
    if camera_id not in stats:
        return jsonify({"ok": False, "error": "Stream not running in manager"}), 404
        
    s = stats[camera_id]
    return jsonify({
        "ok": True, 
        "camera_id": camera_id,
        "fps": s.fps,
        "latency_ms": s.total_lat_ms,
        "anom_score": 0.0, # Handled by background alerter now
        "prediction": "Running continuous inference...",
        "timings": {"total_ms": s.total_lat_ms}
    })


# ── Alerts ─────────────────────────────────────────────────────────────────

@app.route("/api/alerts")
def api_alerts():
    db = get_db()
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 15, type=int)
    cam = request.args.get("cam", "")

    query = "SELECT * FROM alerts WHERE 1=1"
    params = []
    if cam:
        query += " AND camera_id = ?"
        params.append(cam)
    query += " ORDER BY timestamp DESC"

    total = db.execute(
        query.replace("SELECT *", "SELECT COUNT(*)"), params
    ).fetchone()[0]

    query += " LIMIT ? OFFSET ?"
    params.extend([per_page, (page - 1) * per_page])
    rows = db.execute(query, params).fetchall()

    return jsonify({
        "data": rows_to_dicts(rows),
        "total": total,
        "page": page,
        "pages": math.ceil(total / per_page),
    })


@app.route("/api/alerts/<int:alert_id>/ack", methods=["POST"])
def ack_alert(alert_id):
    db = get_db()
    try:
        db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
        db.commit()
        return jsonify({"ok": True})
    except sqlite3.OperationalError as e:
        db.rollback()
        return jsonify({"ok": False, "error": str(e)}), 503


@app.route("/api/stats")
def api_stats():
    db = get_db()
    total_vids = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_id=1"
    ).fetchone()[0]
    
    # FIX: Exclude uploaded videos (-1) from accuracy
    correct = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_id=1 AND true_label=pred_label AND true_label != -1"
    ).fetchone()[0]
    labeled = db.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_id=1 AND true_label != -1"
    ).fetchone()[0]
    
    unacked = db.execute(
        "SELECT COUNT(*) FROM alerts WHERE acknowledged=0 AND anom_score > 0.5"
    ).fetchone()[0]
    acked = db.execute(
        "SELECT COUNT(*) FROM alerts WHERE acknowledged=1"
    ).fetchone()[0]
    avg_lat = db.execute(
        "SELECT AVG(latency_ms) FROM alerts"
    ).fetchone()[0] or 0.0
    
    streams_up = db.execute(
        "SELECT COUNT(*) FROM streams WHERE status='running'"
    ).fetchone()[0]
    
    return jsonify({
        "total_videos": total_vids,
        "correct": correct,
        "accuracy": round(correct / max(labeled, 1), 4),
        "unacked_alerts": unacked,
        "acked_alerts": acked,
        "avg_latency_ms": round(avg_lat, 1),
        "streams_online": streams_up,
    })


@app.route("/api/alerts/ack-all", methods=["POST"])
def api_ack_all():
    """Bulk acknowledge alerts for a camera or all."""
    data = request.get_json(force=True) or {}
    cam = data.get("camera_id", "")
    db = get_db()
    try:
        if cam:
            db.execute("UPDATE alerts SET acknowledged=1 WHERE acknowledged=0 AND anom_score > 0.5 AND camera_id=?", (cam,))
        else:
            db.execute("UPDATE alerts SET acknowledged=1 WHERE acknowledged=0 AND anom_score > 0.5")
        db.commit()
        return jsonify({"ok": True})
    except Exception as e:
        db.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Seed data if DB is brand new (MUST happen before configure_db creates empty file)
    if not os.path.exists(DB_PATH):
        print("[!] Database not found. Creating new one...")
        ensure_tables()
    
    # 2. Configure WAL mode
    configure_db()
    
    # 3. Ensure all tables + indexes exist
    ensure_tables()

    print("\n========================================")
    print("  PolyFlow Dashboard")
    print("  http://127.0.0.1:5000")
    print("========================================")
    print(f"  Checkpoints dir: {CKPT_DIR}")
    print(f"  Uploads dir:     {UPLOAD_DIR}")
    print("========================================\n")
    
    app.run(debug=True, port=5000)
