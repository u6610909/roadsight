"""
Vehicle Counter — camera-based traffic counting
===============================================

Upload traffic footage, draw a region over each movement you care about, and get
vehicle counts, class breakdowns and arrival intervals.

    pip install -r requirements.txt
    python app.py
    → http://127.0.0.1:8000

Built for the ABAC adaptive traffic signal study (CSX3010 Senior Project 1), but
nothing here is specific to that junction — any fixed-camera traffic footage
works. This is a standalone application with no dependency on the research
repository it came from.
"""

import base64
import csv
import io
import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

HERE = Path(__file__).resolve().parent
UPLOADS, MODELS, ROIS = HERE / "uploads", HERE / "models", HERE / "rois"
for d in (UPLOADS, MODELS, ROIS):
    d.mkdir(exist_ok=True)

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

# ─────────────────────────────────────────────────────────────────────────────
# Counting parameters
#
# Documented rather than buried, because each is a decision a user may need to
# revisit for their own site.
# ─────────────────────────────────────────────────────────────────────────────

# Detection confidence, taken from the peak of the trained model's F1 curve
# (0.205) rather than a library default. A low threshold is safer than it looks:
# spurious detections rarely form stable tracks, and anything failing the
# displacement test below is discarded anyway.
CONF_THRESHOLD = 0.20

# Counting window in seconds of VIDEO time, never wall-clock. Inference runs
# well below real time on most machines, so a wall-clock window would cover a
# small fraction of the traffic it appears to.
COUNTING_WINDOW_SEC = 7.0

# A tracked vehicle counts only once it has moved this far from where it was
# first seen. Without it, anything parked in view counts as traffic. At the site
# this was built for, taxis queue across three lanes while waiting to be called
# in; they would otherwise have dominated the counts.
MIN_DISPLACEMENT_PX = 60

# Reporting groups by passenger car unit — how much road capacity a vehicle
# consumes. Traffic engineering works in PCU, not raw counts.
#   motorcycle ~0.33   filters to the front, occupies almost no queue space
#   car        ~1.0    includes pickups
#   heavy      ~2.0    buses and larger goods vehicles
# Unrecognised class names fall back to "car".
PCU_GROUP = {
    "motorcycle": "moto", "motorbike": "moto", "bicycle": "moto",
    "car": "car", "truck": "car", "pickup": "car", "van": "car",
    "bus": "heavy", "heavy": "heavy", "special type": "heavy", "lorry": "heavy",
}
PCU_WEIGHT = {"car": 1.0, "moto": 0.33, "heavy": 2.0}

# BGR, for the live preview
DRAW = {"car": (255, 190, 60), "moto": (120, 220, 120), "heavy": (90, 120, 255)}
PREVIEW_W = 900          # preview is downscaled; the model still sees full res
GROUPS = ["car", "moto", "heavy"]

app = Flask(__name__, static_folder=str(HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 ** 3      # 4 GB upload ceiling
JOBS: dict[str, dict] = {}


def safe(name: str) -> str:
    """Filename only, never a path — blocks directory traversal."""
    return secure_filename(Path(str(name)).name)


# ─────────────────────────────────────────────────────────────── pages
@app.route("/")
def hub():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<page>")
def page(page):
    if page not in {"detector.html", "tally.html", "headway.html", "measure.html"}:
        return "Not found", 404
    return send_from_directory(app.static_folder, page)


# ─────────────────────────────────────────────────────────────── library
@app.route("/api/library")
def library():
    rois = []
    for p in sorted(ROIS.glob("*.json")):
        try:
            rois.append({"name": p.name, "lanes": list(json.loads(p.read_text()).keys())})
        except Exception:
            pass
    return jsonify({
        "videos": [{"name": p.name, "mb": round(p.stat().st_size / 1e6)}
                   for p in sorted(UPLOADS.iterdir()) if p.suffix.lower() in VIDEO_EXT],
        "models": [{"name": p.name, "mb": round(p.stat().st_size / 1e6, 1)}
                   for p in sorted(MODELS.glob("*.pt"))],
        "rois": rois,
        "defaults": {"conf": CONF_THRESHOLD, "window": COUNTING_WINDOW_SEC,
                     "displacement": MIN_DISPLACEMENT_PX},
    })


@app.route("/api/upload/<kind>", methods=["POST"])
def upload(kind):
    if kind not in ("video", "model"):
        return jsonify({"error": "unknown upload type"}), 400
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "no file received"}), 400

    name = safe(f.filename)
    ext = Path(name).suffix.lower()
    if kind == "video" and ext not in VIDEO_EXT:
        return jsonify({"error": f"not a video file ({ext or 'no extension'})"}), 400
    if kind == "model" and ext != ".pt":
        return jsonify({"error": "model must be a .pt file"}), 400

    dest = (UPLOADS if kind == "video" else MODELS) / name
    f.save(dest)

    # A truncated or empty file is accepted happily here and then fails much
    # later with an opaque UnpicklingError once a job starts. Catch it now: a
    # real YOLO checkpoint is megabytes, and any video worth counting is larger
    # still.
    size = dest.stat().st_size
    floor = 100_000 if kind == "model" else 10_000
    if size < floor:
        try:
            dest.unlink()
        except OSError:
            pass          # some mounts disallow unlink; rejecting it is enough
        return jsonify({"error": f"that file is only {size} bytes — truncated or "
                                 f"not a real {kind}"}), 400

    return jsonify({"name": name, "mb": round(size / 1e6, 1)})


@app.route("/api/roi/<name>", methods=["GET", "POST"])
def roi(name):
    p = ROIS / safe(name)
    if request.method == "POST":
        if not p.name.endswith(".json"):
            return jsonify({"error": "filename must end in .json"}), 400
        p.write_text(json.dumps(request.json, indent=2))
        return jsonify({"saved": p.name})
    if not p.exists():
        return jsonify({"error": "not found"}), 404
    return jsonify(json.loads(p.read_text()))


# ─────────────────────────────────────────────────────────────── frame
@app.route("/api/frame")
def frame():
    p = UPLOADS / safe(request.args.get("video", ""))
    if not p.exists():
        return jsonify({"error": "video not found"}), 404
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return jsonify({"error": "cannot open this video — unsupported codec?"}), 400
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    at = float(request.args.get("t", 0))
    if at > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(at * fps))
    ok, img = cap.read()
    cap.release()
    if not ok:
        return jsonify({"error": "cannot read a frame at that time"}), 400
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return jsonify({"image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                    "width": int(img.shape[1]), "height": int(img.shape[0]),
                    "fps": round(fps, 2),
                    "duration": round(total / fps, 1) if fps else 0})


def _annotate(img, rois, boxes, names, moved, totals, vid_t, fps):
    """Draw regions, boxes and a running tally, and return a JPEG."""
    f = img.copy()

    for i, (lane, poly) in enumerate(rois.items()):
        col = list(DRAW.values())[i % len(DRAW)]
        ov = f.copy()
        cv2.fillPoly(ov, [poly], tuple(c // 4 for c in col))
        cv2.addWeighted(ov, 0.30, f, 0.70, 0, f)
        cv2.polylines(f, [poly], True, col, 3)
        c = poly.mean(axis=0).astype(int)
        cv2.putText(f, lane, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)

    for box in boxes:
        tid = int(box.id[0]) if box.id is not None else None
        grp = PCU_GROUP.get(str(names.get(int(box.cls[0]), "car")).lower(), "car")
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        counted = tid in moved
        col = DRAW[grp] if counted else (150, 150, 150)
        cv2.rectangle(f, (x1, y1), (x2, y2), col, 2)
        tag = f"{grp} #{tid}" if counted else "waiting"
        cv2.putText(f, tag, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # tally panel
    tot = {g: sum(t[g] for t in totals.values()) for g in GROUPS}
    lines = [f"car   {tot['car']:>4}", f"moto  {tot['moto']:>4}",
             f"heavy {tot['heavy']:>4}",
             f"TOTAL {sum(tot.values()):>4}",
             f"{fps:4.1f} fps   t={vid_t:.0f}s"]
    ov = f.copy()
    cv2.rectangle(ov, (14, 14), (300, 40 + 34 * len(lines)), (18, 18, 22), -1)
    cv2.addWeighted(ov, 0.72, f, 0.28, 0, f)
    for i, ln in enumerate(lines):
        cv2.putText(f, ln, (28, 52 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (235, 235, 245), 2)

    if f.shape[1] > PREVIEW_W:
        k = PREVIEW_W / f.shape[1]
        f = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return buf.tobytes() if ok else None


# ─────────────────────────────────────────────────────────────── counting
def run_job(jid: str, cfg: dict):
    """
    Counting loop. Three decisions make the counts trustworthy:
      · ByteTrack persistent IDs — a vehicle is counted once, not once per frame
      · displacement gate       — parked vehicles are not traffic
      · video-time windows      — processing speed cannot distort the bins
    """
    from ultralytics import YOLO
    J = JOBS[jid]
    try:
        try:
            model = YOLO(str(MODELS / safe(cfg["model"])))
        except Exception as e:
            raise RuntimeError(
                f"could not load '{cfg['model']}' — is it a valid Ultralytics "
                f"checkpoint? ({type(e).__name__})") from e
        names = model.names

        device = cfg.get("device", "auto")
        if device == "auto":
            import torch
            device = ("mps" if torch.backends.mps.is_available()
                      else "0" if torch.cuda.is_available() else "cpu")
        J["device"] = device

        cap = cv2.VideoCapture(str(UPLOADS / safe(cfg["video"])))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        start_s = float(cfg.get("start", 0))
        dur_s = float(cfg.get("duration", 0)) or None
        if start_s:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
        J["total_frames"] = int(dur_s * fps if dur_s else
                                max(1, cap.get(cv2.CAP_PROP_FRAME_COUNT) - start_s * fps))

        gate = float(cfg.get("displacement", MIN_DISPLACEMENT_PX))
        rois = {k: np.array(v, np.int32) for k, v in cfg["rois"].items()}
        counted = {k: set() for k in rois}
        totals = {k: {g: 0 for g in GROUPS} for k in rois}
        origin, moved = {}, set()
        windows, win_start = [], start_s
        win = {k: {g: 0 for g in GROUPS} for k in rois}
        t0 = time.time()
        done = 0

        while not J["stop"]:
            ok, img = cap.read()
            if not ok:
                break
            done += 1
            vid_t = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if dur_s and vid_t - start_s > dur_s:
                break

            res = model.track(img, conf=cfg["conf"], imgsz=cfg["imgsz"], persist=True,
                              tracker="bytetrack.yaml", device=device, verbose=False)[0]

            for box in res.boxes:
                tid = int(box.id[0]) if box.id is not None else None
                if tid is None:
                    continue
                grp = PCU_GROUP.get(str(names.get(int(box.cls[0]), "car")).lower(), "car")
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if tid not in origin:
                    origin[tid] = (cx, cy)
                elif tid not in moved:
                    ox, oy = origin[tid]
                    if np.hypot(cx - ox, cy - oy) >= gate:
                        moved.add(tid)
                if tid not in moved:
                    continue

                for lane, poly in rois.items():
                    if tid in counted[lane]:
                        continue
                    if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                        counted[lane].add(tid)
                        totals[lane][grp] += 1
                        win[lane][grp] += 1

            if vid_t - win_start >= cfg["window"]:
                windows.append({"t": round(win_start, 1),
                                "counts": json.loads(json.dumps(win))})
                win = {k: {g: 0 for g in GROUPS} for k in rois}
                win_start = vid_t

            # Live preview. The annotated frame is stashed as a JPEG for the
            # MJPEG endpoint to serve. Encoding costs a few ms and is skipped
            # entirely when nobody is watching.
            if J.get("watchers", 0) > 0:
                J["jpeg"] = _annotate(img, rois, res.boxes, names, moved,
                                      totals, vid_t, J.get("fps", 0))

            el = time.time() - t0
            J.update(progress=min(1.0, done / J["total_frames"]),
                     video_t=round(vid_t, 1), elapsed=round(el, 1),
                     fps=round(done / el, 2) if el else 0,
                     eta=round((J["total_frames"] - done) / (done / el)) if el and done else None,
                     totals=json.loads(json.dumps(totals)),
                     windows=windows[-40:])

        cap.release()
        J["totals"] = json.loads(json.dumps(totals))
        J["all_windows"] = windows
        J["state"] = "stopped" if J["stop"] else "done"
        if not J["stop"]:
            J["progress"] = 1.0
    except Exception as e:
        J["state"] = "error"
        J["error"] = f"{type(e).__name__}: {e}"


@app.route("/api/run", methods=["POST"])
def start():
    cfg = request.json or {}
    if not cfg.get("rois"):
        return jsonify({"error": "draw at least one region first"}), 400
    if not cfg.get("model"):
        return jsonify({"error": "no model selected — upload a .pt file"}), 400
    if not cfg.get("video"):
        return jsonify({"error": "no video selected"}), 400
    jid = uuid.uuid4().hex[:8]
    JOBS[jid] = {"state": "running", "stop": False, "progress": 0.0,
                 "totals": {}, "windows": [], "cfg": cfg,
                 "watchers": 0, "jpeg": None}
    threading.Thread(target=run_job, args=(jid, cfg), daemon=True).start()
    return jsonify({"job": jid})


@app.route("/api/job/<jid>")
def status(jid):
    J = JOBS.get(jid)
    if not J:
        return jsonify({"error": "no such job"}), 404
    return jsonify({k: v for k, v in J.items()
                    if k not in ("stop", "all_windows", "jpeg")})


@app.route("/api/job/<jid>/stream.mjpg")
def stream(jid):
    """
    Live preview as multipart MJPEG — an <img src> shows it with no JavaScript.

    A watcher count gates the drawing: with nobody looking, run_job skips
    annotation and JPEG encoding entirely, so the preview costs nothing when
    it is not being used.
    """
    J = JOBS.get(jid)
    if not J:
        return "no such job", 404

    def gen():
        J["watchers"] = J.get("watchers", 0) + 1
        try:
            last = None
            while J["state"] == "running":
                buf = J.get("jpeg")
                if buf is not None and buf is not last:
                    last = buf
                    yield (b"--f\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(buf)).encode() +
                           b"\r\n\r\n" + buf + b"\r\n")
                else:
                    time.sleep(0.03)
        finally:
            J["watchers"] = max(0, J.get("watchers", 1) - 1)

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=f",
                    headers={"Cache-Control": "no-store"})


@app.route("/api/job/<jid>/stop", methods=["POST"])
def stop(jid):
    if jid in JOBS:
        JOBS[jid]["stop"] = True
    return jsonify({"ok": True})


@app.route("/api/job/<jid>/export.<fmt>")
def export(jid, fmt):
    J = JOBS.get(jid)
    if not J:
        return "no such job", 404
    cfg, totals = J["cfg"], J.get("totals", {})
    dur = cfg.get("duration") or J.get("video_t", 0)

    if fmt == "json":
        return Response(json.dumps({"config": cfg, "totals": totals,
                                    "windows": J.get("all_windows", [])}, indent=2),
                        mimetype="application/json",
                        headers={"Content-Disposition": f"attachment;filename=counts_{jid}.json"})

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["region", "car", "moto", "heavy", "total", "cars_heavy", "pcu",
                "interval_all_s", "interval_cars_heavy_s", "veh_per_hour"])
    for lane, g in totals.items():
        tot = sum(g.values())
        ch = g["car"] + g["heavy"]
        pcu = sum(g[k] * PCU_WEIGHT[k] for k in GROUPS)
        w.writerow([lane, g["car"], g["moto"], g["heavy"], tot, ch, round(pcu, 1),
                    round(dur / tot, 2) if tot else "",
                    round(dur / ch, 2) if ch else "",
                    round(tot * 3600 / dur) if dur else ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment;filename=counts_{jid}.csv"})


if __name__ == "__main__":
    print("\n  Vehicle Counter  →  http://127.0.0.1:8000\n")
    app.run(host="127.0.0.1", port=8000, threaded=True)
