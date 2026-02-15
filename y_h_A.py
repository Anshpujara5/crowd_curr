# python "F:\project\Crowd Management\heat_map\heat_map\y_h_A.py"

import cv2
import numpy as np
import os
from datetime import datetime
import heapq
from ultralytics import YOLO

# ✅ NEW: shared output image path (API will read this)
SHARED_DIR = r"F:\project\Crowd Management\heat_map\heat_map"
os.makedirs(SHARED_DIR, exist_ok=True)
SHARED_PROCESSED_JPG = os.path.join(SHARED_DIR, "shared_processed.jpg")
SHARED_RAW_JPG = os.path.join(SHARED_DIR, "shared_raw.jpg")  # optional


# ---------- A* Pathfinding ----------
def astar(cost_map, start, end):
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


# ---------- Points ordering (robust homography) ----------
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


# ----- Function to let user click 4 points -----
def get_four_points_from_image(image):
    points = []

    def mouse_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"📌 Point {len(points)}: ({x}, {y})")

    clone = image.copy()
    cv2.namedWindow("Select 4 ground points")
    cv2.setMouseCallback("Select 4 ground points", mouse_handler)

    print("\n💁️ Click 4 FLOOR points (a big rectangle) in ANY order.")
    print("✅ Tips: choose wide corners of the walkable floor region.")
    print("🔔 After selecting 4 points, press any key.\n")

    while True:
        temp = clone.copy()
        for p in points:
            cv2.circle(temp, p, 6, (0, 255, 0), -1)
        cv2.imshow("Select 4 ground points", temp)

        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4 and key != 255:
            break

    cv2.destroyWindow("Select 4 ground points")
    return points


# ---------- Select Input Source ----------
while True:
    print("📷 Select input source:")
    print("  1 - Webcam")
    print("  2 - Video file")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        video_path = 0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Webcam not found.")
            continue
        source_info = "webcam"
        break

    if choice == "2":
        video_path = input('📂 Enter video path: ').strip().strip('"')
        if not os.path.exists(video_path):
            print(f"❌ Path does not exist: {video_path}")
            continue
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            continue
        source_info = "video"
        break

    print("❗ Invalid choice. Please enter 1 or 2.")


# ---------- Output Directory ----------
output_dir = r"F:\Crowd Management\data_ansh\output"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"heatmap_output_with_path_{timestamp}.mp4")


# ---------- Load Model ----------
print("🧠 Loading YOLO model...")
model = YOLO("yolov5s.pt")
CONF = 0.4
PERSON_CLASS = 0


# ---------- Video Properties ----------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if (not fps) or (fps != fps):
    fps = 25.0

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError(f"Invalid video resolution: width={width}, height={height}")


# ---------- Video Writer ----------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
if not out.isOpened():
    cap.release()
    raise RuntimeError(f"❌ VideoWriter failed to open: {output_path}")

print("✅ Output will be saved to:", output_path)


# ---------- Background Subtractor ----------
fgbg = cv2.createBackgroundSubtractorMOG2(history=1500, varThreshold=16, detectShadows=False)

# ---------- Heatmap Accumulator ----------
heatmap_acc = np.zeros((height, width), dtype=np.float32)
alpha = 0.6
sigma = 15


# ---------- Preview Frames and Get Ground Points ----------
print("\n📸 Choose a clear frame for selecting 4 floor points.")
print("👉 Controls: Press 'd' skip | 's' select | 'q' quit\n")

while True:
    ret, preview_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        raise RuntimeError("❌ Failed to read from video during preview.")

    temp = preview_frame.copy()
    cv2.putText(temp, "Press 'd' skip | 's' select | 'q' quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Frame Preview", temp)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("d"):
        continue
    if key == ord("s"):
        break
    if key == ord("q"):
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        raise SystemExit("🛑 Exiting selection process.")

cv2.destroyWindow("Frame Preview")

ground_points = get_four_points_from_image(preview_frame)
src_pts = order_points(ground_points)
print("✅ Ordered points (TL,TR,BR,BL):", src_pts.tolist())

dbg = preview_frame.copy()
cv2.polylines(dbg, [src_pts.astype(int).reshape(-1, 1, 2)], True, (0, 255, 255), 3)
cv2.imshow("DEBUG: Selected Quad", dbg)
cv2.waitKey(800)
cv2.destroyWindow("DEBUG: Selected Quad")

if source_info == "video":
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# ---------- Homography Setup ----------
warp_width, warp_height = 640, 480
dst_pts = np.array(
    [[0, 0],
     [warp_width - 1, 0],
     [warp_width - 1, warp_height - 1],
     [0, warp_height - 1]], dtype=np.float32
)

homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
inv_homography = np.linalg.inv(homography_matrix)


# ---------- Frame Loop ----------
frame_index = 0
print("\n🚀 Processing started. Press 'q' to stop.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("🛑 End of video.")
        break

    frame_index += 1

    # YOLO Detection
    pred = model.predict(frame, conf=CONF, verbose=False)[0]
    if pred.boxes is None or len(pred.boxes) == 0:
        detections = np.zeros((0, 6), dtype=np.float32)
    else:
        xyxy = pred.boxes.xyxy.cpu().numpy()
        confs = pred.boxes.conf.cpu().numpy()
        clss = pred.boxes.cls.cpu().numpy()
        mask = (clss == PERSON_CLASS)
        detections = np.column_stack([xyxy[mask], confs[mask], clss[mask]])

    # Motion Detection
    fgmask = fgbg.apply(frame)
    motion_mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
    motion_mask = cv2.medianBlur(motion_mask, 5)

    # Heatmap Update
    heatmap_acc *= 0.95

    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            heatmap_acc[y1:y2, x1:x2] += 1.0

    heatmap_acc += (motion_mask / 255.0) * 0.5

    blurred = cv2.GaussianBlur(heatmap_acc, (0, 0), sigma)
    norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

    heatmap_color = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, alpha, heatmap_color, 1 - alpha, 0)

    # Warped Heatmap and Path Planning
    warped_heatmap = cv2.warpPerspective(norm, homography_matrix, (warp_width, warp_height))

    grid_size = 20
    resized_cost = cv2.resize(warped_heatmap, (grid_size, grid_size))
    cost_map = resized_cost.astype(np.float32) + 1.0

    start = (grid_size - 1, 0)
    end = (0, grid_size - 1)
    path = astar(cost_map, start, end)

    # Draw Path
    pts = []
    for (yy, xx) in path:
        wx = int((xx + 0.5) * warp_width / grid_size)
        wy = int((yy + 0.5) * warp_height / grid_size)

        p = cv2.perspectiveTransform(np.array([[[wx, wy]]], dtype=np.float32), inv_homography)[0][0]
        px, py = int(p[0]), int(p[1])

        if 0 <= px < width and 0 <= py < height:
            pts.append((px, py))

    if len(pts) >= 2:
        cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False, (0, 255, 0), 6)
        cv2.circle(overlay, pts[0], 10, (0, 255, 255), -1)
        cv2.circle(overlay, pts[-1], 10, (255, 0, 255), -1)
    else:
        cv2.putText(overlay, "PATH NOT VISIBLE (bad homography - reselect points)", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show Info
    person_count = len(detections)
    cv2.putText(overlay, f"People: {person_count}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ✅ NEW: Export frames for API
    # processed
    cv2.imwrite(SHARED_PROCESSED_JPG, overlay)
    # raw (optional)
    cv2.imwrite(SHARED_RAW_JPG, frame)

    out.write(overlay)

    cv2.imshow("Output", overlay)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Exit signal received (q pressed).")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Saved: {output_path}")
print(f"✅ Shared processed frame: {SHARED_PROCESSED_JPG}")