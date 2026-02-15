# app.py  (UPDATED: PUBLIC + can connect with y_h_A.py via shared JPG files)
# -----------------------------------------------------------------------
# Install:
#   pip install fastapi uvicorn python-multipart ultralytics opencv-python numpy
#
# Run:
#   python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# ✅ Two ways to stream:
# 1) RAW webcam from this API:
#      POST  /stream/start
#      GET   /stream.mjpg
#      POST  /stream/stop
#
# 2) STREAM FROM y_h_A.py (recommended when you run y_h_A separately):
#      y_h_A.py writes:
#         shared_raw.jpg
#         shared_processed.jpg
#      Then API serves:
#         GET /shared/raw.mjpg
#         GET /shared/processed.mjpg
#
# ⚠️ IMPORTANT:
# - If y_h_A.py uses webcam, DO NOT use /stream/start (camera will be busy).
#   Use /shared/raw.mjpg and /shared/processed.mjpg instead.

# python -m uvicorn app:app --reload
# http://localhost:8000/shared/raw.mjpg
# http://localhost:8000/shared/processed.mjpg

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import time
import threading
import os

import cv2
import numpy as np
import heapq
from ultralytics import YOLO


app = FastAPI(title="Demo Backend - Public Raw + Processed + Shared Streams")

# ✅ Allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # change if your Next runs elsewhere
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- OPTIONAL: old auth DB kept but NOT used --------------------
FAKE_USERS = {
    "ansh": {"password": "1234", "name": "Ansh", "role": "student"},
    "admin": {"password": "admin", "name": "Admin", "role": "admin"},
}
TOKENS: Dict[str, str] = {}
NOTIFICATIONS: List[dict] = []


# -------------------- RAW MJPEG STREAM (webcam from API) --------------------
cap = None
cap_lock = threading.Lock()

latest_jpeg: Optional[bytes] = None
jpeg_lock = threading.Lock()

stream_running = False
stream_thread = None


def _camera_worker(camera_index: int = 0):
    """Continuously read webcam frames and keep latest JPEG in memory."""
    global cap, latest_jpeg, stream_running

    with cap_lock:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            stream_running = False
            return

    while stream_running:
        with cap_lock:
            ok, frame = cap.read()

        if not ok or frame is None:
            time.sleep(0.02)
            continue

        cv2.putText(frame, "RAW LIVE", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with jpeg_lock:
                latest_jpeg = jpg.tobytes()

        time.sleep(0.01)

    with cap_lock:
        if cap is not None:
            cap.release()


def _mjpeg_generator():
    boundary = "frame"
    while True:
        with jpeg_lock:
            frame = latest_jpeg

        if frame is None:
            time.sleep(0.02)
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            + frame + b"\r\n"
        )
        time.sleep(0.03)


# -------------------- PROCESSED MJPEG STREAM (YOLO + Heatmap + A*) --------------------
proc_running = False
proc_thread = None

latest_proc_jpeg: Optional[bytes] = None
proc_jpeg_lock = threading.Lock()

GROUND_POINTS: Optional[np.ndarray] = None
H: Optional[np.ndarray] = None
H_INV: Optional[np.ndarray] = None

yolo_model: Optional[YOLO] = None

CONF = 0.4
PERSON_CLASS = 0
warp_width, warp_height = 640, 480
grid_size = 20
alpha = 0.6
sigma = 15
decay = 0.95


def astar(cost_map: np.ndarray, start: tuple, end: tuple) -> List[tuple]:
    rows, cols = cost_map.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0.0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = current[0] + dx, current[1] + dy
            if 0 <= nr < rows and 0 <= nc < cols:
                neighbor = (nr, nc)
                tentative_g = g_score[current] + float(cost_map[neighbor])
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
    return []


def order_points(pts: List[List[int]]) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _ensure_yolo_loaded():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO("yolov5s.pt")


def _processed_worker(camera_index: int = 0):
    global proc_running, latest_proc_jpeg, H, H_INV

    _ensure_yolo_loaded()

    local_cap = cv2.VideoCapture(camera_index)
    if not local_cap.isOpened():
        proc_running = False
        return

    fgbg_local = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=16, detectShadows=False)
    heatmap_acc_local: Optional[np.ndarray] = None

    while proc_running:
        ok, frame = local_cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        h0, w0 = frame.shape[:2]
        if heatmap_acc_local is None:
            heatmap_acc_local = np.zeros((h0, w0), dtype=np.float32)

        if H is None or H_INV is None:
            out_frame = frame.copy()
            cv2.putText(out_frame, "POST /vision/points (4 points)", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ok2, jpg = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok2:
                with proc_jpeg_lock:
                    latest_proc_jpeg = jpg.tobytes()
            time.sleep(0.03)
            continue

        pred = yolo_model.predict(frame, conf=CONF, verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            detections = np.zeros((0, 6), dtype=np.float32)
        else:
            xyxy = pred.boxes.xyxy.cpu().numpy()
            confs = pred.boxes.conf.cpu().numpy()
            clss = pred.boxes.cls.cpu().numpy()
            mask = (clss == PERSON_CLASS)
            detections = np.column_stack([xyxy[mask], confs[mask], clss[mask]])

        fgmask = fgbg_local.apply(frame)
        motion_mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        motion_mask = cv2.medianBlur(motion_mask, 5)

        heatmap_acc_local *= decay
        for *xyxy, _conf, _cls in detections:
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, min(w0 - 1, x1))
            x2 = max(0, min(w0, x2))
            y1 = max(0, min(h0 - 1, y1))
            y2 = max(0, min(h0, y2))
            if x2 > x1 and y2 > y1:
                heatmap_acc_local[y1:y2, x1:x2] += 1.0

        heatmap_acc_local += (motion_mask / 255.0) * 0.5

        blurred = cv2.GaussianBlur(heatmap_acc_local, (0, 0), sigma)
        norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        heatmap_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, alpha, heatmap_color, 1 - alpha, 0)

        warped_heatmap = cv2.warpPerspective(norm, H, (warp_width, warp_height))
        resized_cost = cv2.resize(warped_heatmap, (grid_size, grid_size))
        cost_map = resized_cost.astype(np.float32) + 1.0

        start = (grid_size - 1, 0)
        end = (0, grid_size - 1)
        path = astar(cost_map, start, end)

        pts = []
        for (yy, xx) in path:
            wx = int((xx + 0.5) * warp_width / grid_size)
            wy = int((yy + 0.5) * warp_height / grid_size)
            p = cv2.perspectiveTransform(np.array([[[wx, wy]]], dtype=np.float32), H_INV)[0][0]
            px, py = int(p[0]), int(p[1])
            if 0 <= px < w0 and 0 <= py < h0:
                pts.append((px, py))

        if len(pts) >= 2:
            cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 6)
            cv2.circle(overlay, pts[0], 10, (0, 255, 255), -1)
            cv2.circle(overlay, pts[-1], 10, (255, 0, 255), -1)
        else:
            cv2.putText(overlay, "PATH NOT VISIBLE - change points", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(overlay, f"People: {len(detections)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ok2, jpg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok2:
            with proc_jpeg_lock:
                latest_proc_jpeg = jpg.tobytes()

        time.sleep(0.03)

    local_cap.release()


def _processed_mjpeg_generator():
    boundary = "frame"
    while True:
        with proc_jpeg_lock:
            frame = latest_proc_jpeg

        if frame is None:
            time.sleep(0.02)
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
            + frame + b"\r\n"
        )
        time.sleep(0.03)


# -------------------- SHARED FILE STREAMS (connect with y_h_A.py) --------------------
# y_h_A.py should write these continuously:
SHARED_DIR = r"F:\project\Crowd Management\heat_map\heat_map"
SHARED_RAW_JPG = os.path.join(SHARED_DIR, "shared_raw.jpg")
SHARED_PROCESSED_JPG = os.path.join(SHARED_DIR, "shared_processed.jpg")


def _shared_file_mjpeg_generator(file_path: str):
    boundary = "frame"
    last_mtime = 0.0

    while True:
        if os.path.exists(file_path):
            try:
                mtime = os.path.getmtime(file_path)
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(file_path, "rb") as f:
                        frame = f.read()

                    if frame:
                        yield (
                            b"--" + boundary.encode() + b"\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                            + frame + b"\r\n"
                        )
            except:
                pass

        time.sleep(0.03)


# --- Models ---
class LoginBody(BaseModel):
    username: str
    password: str


class NotificationBody(BaseModel):
    message: str


class PointsBody(BaseModel):
    points: List[List[int]]


# -------------------- Base APIs --------------------
@app.get("/health")
def health():
    return {"ok": True, "ts": int(time.time())}


# (Kept for compatibility; public anyway)
@app.post("/auth/login")
def login(body: LoginBody):
    user = FAKE_USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = f"token_{body.username}_{int(time.time())}"
    TOKENS[token] = body.username
    return {"token": token, "user": {"username": body.username, "name": user["name"], "role": user["role"]}}


@app.get("/notifications")
def get_notifications():
    return {"count": len(NOTIFICATIONS), "items": NOTIFICATIONS}


@app.post("/notifications")
def create_notification(body: NotificationBody):
    notif = {"id": len(NOTIFICATIONS) + 1, "message": body.message, "by": "public", "ts": int(time.time())}
    NOTIFICATIONS.append(notif)
    return {"created": True, "notification": notif}


# -------------------- RAW Stream APIs (PUBLIC) --------------------
@app.post("/stream/start")
def start_stream():
    global stream_running, stream_thread
    if stream_running:
        return {"ok": True, "status": "already_running"}

    stream_running = True
    stream_thread = threading.Thread(target=_camera_worker, args=(0,), daemon=True)
    stream_thread.start()
    return {"ok": True, "status": "started"}


@app.post("/stream/stop")
def stop_stream():
    global stream_running
    stream_running = False
    return {"ok": True, "status": "stopping"}


@app.get("/stream.mjpg")
def stream_mjpg():
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------- PROCESSED Stream APIs (PUBLIC) --------------------
@app.post("/vision/points")
def set_points(body: PointsBody):
    global GROUND_POINTS, H, H_INV

    if len(body.points) != 4:
        raise HTTPException(status_code=400, detail="Need exactly 4 points")

    ordered = order_points(body.points)
    GROUND_POINTS = ordered

    dst_pts = np.array(
        [[0, 0],
         [warp_width - 1, 0],
         [warp_width - 1, warp_height - 1],
         [0, warp_height - 1]], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(GROUND_POINTS, dst_pts)
    H_INV = np.linalg.inv(H)

    return {"ok": True, "ordered_points": GROUND_POINTS.tolist()}


@app.post("/stream/processed/start")
def start_processed_stream():
    global proc_running, proc_thread
    if proc_running:
        return {"ok": True, "status": "already_running"}

    proc_running = True
    proc_thread = threading.Thread(target=_processed_worker, args=(0,), daemon=True)
    proc_thread.start()
    return {"ok": True, "status": "started"}


@app.post("/stream/processed/stop")
def stop_processed_stream():
    global proc_running
    proc_running = False
    return {"ok": True, "status": "stopping"}


@app.get("/stream.processed.mjpg")
def stream_processed_mjpg():
    return StreamingResponse(_processed_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------- SHARED FILE STREAM ENDPOINTS (PUBLIC) --------------------
@app.get("/shared/raw.mjpg")
def shared_raw():
    return StreamingResponse(_shared_file_mjpeg_generator(SHARED_RAW_JPG),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/shared/processed.mjpg")
def shared_processed():
    return StreamingResponse(_shared_file_mjpeg_generator(SHARED_PROCESSED_JPG),
                             media_type="multipart/x-mixed-replace; boundary=frame")