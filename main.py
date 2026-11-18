import cv2
import csv
import os
import sys
import time
import threading
from datetime import datetime

# ==========================================
# 1. STUDENT ROSTER (MAP ID NUMBERS TO NAMES)
# ==========================================
STUDENT_ROSTER = {
    "241-02079": {"name": "Joseph D. Faisan", "course": "BSIT", "year": "2nd Year"},
    "241-00432": {"name": "Carl Durango", "course": "BSIT", "year": "3rd Year"},
    "241-00123": {"name": "Juan Dela Cruz", "course": "BSIT", "year": "2nd Year"},
    "241-00456": {"name": "Maria Clara", "course": "BSCS", "year": "3rd Year"}
}

# ==========================================
# 2. SETUP & INITIALIZATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "attendance_log.csv")
STUDENTS_CSV = os.path.join(BASE_DIR, "students.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "attendance_backups")
CSV_HEADER = ["Timestamp", "Student_ID", "Student_Name", "Course", "Year_Level", "Status"]
WINDOW_NAME = "SmartAttend Dashboard"
COOLDOWN_SECONDS = 5

# Load external students.csv if present in the project folder
if os.path.exists(STUDENTS_CSV):
    try:
        with open(STUDENTS_CSV, mode="r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = (row.get("Student_ID") or "").strip()
                if sid:
                    STUDENT_ROSTER[sid] = {
                        "name": (row.get("Name") or "Registered Student").strip(),
                        "course": (row.get("Course") or "N/A").strip(),
                        "year": (row.get("Year_Level") or "N/A").strip()
                    }
    except OSError as e:
        print(f"[WARN] Could not read students.csv: {e}")

ROSTER_LOOKUP = {k.strip().upper(): (k.strip(), v) for k, v in STUDENT_ROSTER.items()}

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print(f"[ERROR] Could not load face cascade: {cascade_path}")
    sys.exit(1)

qr_detector = cv2.QRCodeDetector()
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Gentle contrast booster
csv_lock = threading.Lock()
state_lock = threading.Lock()
stop_event = threading.Event()


def init_csv(reset=False):
    with csv_lock:
        needs_header = reset or not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
        if not needs_header:
            try:
                with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
                    if next(csv.reader(f), []) != CSV_HEADER:
                        needs_header = True
            except (OSError, csv.Error):
                needs_header = True

        # Save a timestamped backup before overwriting existing attendance records
        if needs_header and os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            try:
                with open(CSV_FILE, mode="r", encoding="utf-8") as f:
                    content = f.read()
                if len([line for line in content.splitlines() if line.strip()]) > 1:
                    os.makedirs(BACKUP_DIR, exist_ok=True)
                    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    backup_path = os.path.join(BACKUP_DIR, f"attendance_backup_{stamp}.txt")
                    with open(backup_path, mode="w", encoding="utf-8") as bf:
                        bf.write(content)
                    print(f"[INFO] Backup saved: {backup_path}")
            except OSError as e:
                print(f"[ERROR] Backup failed, reset cancelled: {e}")
                return False

        if needs_header:
            try:
                with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(CSV_HEADER)
            except OSError as e:
                print(f"[ERROR] Could not write CSV header: {e}")
                return False
    return True


init_csv()

# Shared thread state
state = {
    "latest_gray": None,
    "faces": [],
    "bbox": None,
    "last_scanned_id": "",
    "last_scanned_name": "",
    "last_scanned_course": "",
    "last_scanned_year": "",
    "last_log_time": -1e9
}

# ==========================================
# 3. BACKGROUND AI WORKER THREAD (FACE + QR)
# ==========================================
def decode_qr_stable(gray_frame):
    # Stage 1: Direct decode on raw grayscale
    data, bbox, _ = qr_detector.detectAndDecode(gray_frame)
    if data:
        return data, bbox

    # Stage 2: Gentle CLAHE contrast enhancement for dim/uneven lighting
    enhanced = clahe.apply(gray_frame)
    data, bbox, _ = qr_detector.detectAndDecode(enhanced)
    if data:
        return data, bbox

    # Stage 3: Light unsharp mask to sharpen slightly blurry QR edges
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    data, bbox, _ = qr_detector.detectAndDecode(sharpened)
    return data, bbox


def detection_worker():
    prev_bbox = None
    last_bbox_time = 0.0
    last_faces = []
    face_miss_count = 0

    while not stop_event.is_set():
        with state_lock:
            gray_frame = state["latest_gray"]
            state["latest_gray"] = None

        if gray_frame is None:
            time.sleep(0.005)
            continue

        now = time.monotonic()

        # 1. Fast Face Detection on half-scale image with short anti-flicker hold
        h, w = gray_frame.shape[:2]
        small_gray = cv2.resize(gray_frame, (320, 240))
        small_faces = face_cascade.detectMultiScale(
            small_gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40)
        )
        if len(small_faces) > 0:
            sx, sy = w / 320.0, h / 240.0
            last_faces = [(int(x * sx), int(y * sy), int(fw * sx), int(fh * sy)) for (x, y, fw, fh) in small_faces]
            face_miss_count = 0
        else:
            face_miss_count += 1
            if face_miss_count > 3:
                last_faces = []

        # 2. Multi-stage QR Detection + bounding-box smoothing
        data, bbox = decode_qr_stable(gray_frame)
        if bbox is not None and bbox.size == 8:
            pts = bbox.reshape(4, 2).astype("float32")
            if prev_bbox is not None and abs(pts - prev_bbox).max() < 80:
                prev_bbox = (prev_bbox * 0.4) + (pts * 0.6)
            else:
                prev_bbox = pts
            last_bbox_time = now
        elif (now - last_bbox_time) > 0.25:
            prev_bbox = None

        with state_lock:
            state["faces"] = last_faces
            state["bbox"] = prev_bbox

        if data:
            raw_text = "".join(ch for ch in data.strip() if ch.isprintable())
            if not raw_text:
                continue

            if "," in raw_text:
                scanned_id, qr_name = [part.strip() for part in raw_text.split(",", 1)]
            else:
                scanned_id, qr_name = raw_text, ""

            entry = ROSTER_LOOKUP.get(scanned_id.upper())
            if entry:
                scanned_id, info = entry
                scanned_name = info["name"]
                scanned_course = info["course"]
                scanned_year = info["year"]
                status = "Present"
            else:
                scanned_name = qr_name or "Registered Student"
                scanned_course = "N/A"
                scanned_year = "N/A"
                status = "Present (Not in roster)"

            with state_lock:
                should_log = (
                    scanned_id != state["last_scanned_id"]
                    or (now - state["last_log_time"] > COOLDOWN_SECONDS)
                )

            if should_log:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with csv_lock:
                    try:
                        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                timestamp, scanned_id, scanned_name,
                                scanned_course, scanned_year, status
                            ])
                    except OSError as e:
                        print(f"[ERROR] Failed to log attendance: {e}")
                        continue

                with state_lock:
                    state["last_log_time"] = now
                    state["last_scanned_id"] = scanned_id
                    state["last_scanned_name"] = scanned_name
                    state["last_scanned_course"] = scanned_course
                    state["last_scanned_year"] = scanned_year

                print(f"[SUCCESS] QR Read -> ID: {scanned_id} | Name: {scanned_name} ({scanned_course} - {scanned_year}) at {timestamp}")

# ==========================================
# 4. HIGH-FPS WEBCAM LOOP (MAIN THREAD)
# ==========================================
def open_camera(index):
    cap_obj = cv2.VideoCapture(index)
    if cap_obj.isOpened():
        cap_obj.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_obj.set(cv2.CAP_PROP_FPS, 30)
        cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap_obj


def main():
    cam_index = 0
    cap = open_camera(cam_index)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)

    # Start background detection thread
    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    prev_frame_time = 0.0
    smooth_fps = 30.0
    failed_reads = 0

    print("[INFO] SmartAttend Threaded High-FPS Mode Ready. Press 'c' to switch camera, 'r' to reset, 'f' for fullscreen, 'q' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if failed_reads > 40:
                    print("[ERROR] No working camera found. Check connection and restart.")
                    break
                if failed_reads % 10 == 0:
                    cam_index = (cam_index + 1) % 4
                    cap.release()
                    cap = open_camera(cam_index)
                time.sleep(0.05)
                continue
            failed_reads = 0

            current_time = time.monotonic()
            if prev_frame_time > 0 and (current_time - prev_frame_time) > 0:
                instant_fps = 1.0 / (current_time - prev_frame_time)
                smooth_fps = (smooth_fps * 0.85) + (instant_fps * 0.15)
            prev_frame_time = current_time

            # Mirror view + subtle contrast/brightness balance (keeps high FPS)
            frame = cv2.flip(frame, 1)
            frame = cv2.convertScaleAbs(frame, alpha=1.06, beta=5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Hand off latest frame to background thread without blocking video rendering
            with state_lock:
                state["latest_gray"] = gray
                faces = list(state["faces"])
                bbox = state["bbox"]
                last_id = state["last_scanned_id"]
                last_name = state["last_scanned_name"]
                last_course = state["last_scanned_course"]
                last_year = state["last_scanned_year"]
                last_log_time = state["last_log_time"]

            # Scale video to fit resizable window while keeping HUD crisp
            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
            except cv2.error:
                win_w, win_h = 640, 480

            src_h, src_w = frame.shape[:2]
            scale = min(win_w / src_w, win_h / src_h) if (win_w > 1 and win_h > 1) else 1.0
            disp_w, disp_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
            if scale != 1.0:
                frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

            sc = lambda v: max(1, int(round(v * scale)))

            # Draw stabilized QR bounding box if detected
            if bbox is not None:
                pts = (bbox * scale).astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), sc(3))

            # Cooldown timer calculation
            time_since_scan = current_time - last_log_time
            is_recent = (last_id != "") and (time_since_scan <= COOLDOWN_SECONDS)
            time_remaining = max(0, int(COOLDOWN_SECONDS - time_since_scan))

            # Draw Face Bounding Boxes & Persistent Student Name
            for (fx, fy, fw, fh) in faces:
                x, y, w, h = int(fx * scale), int(fy * scale), int(fw * scale), int(fh * scale)
                cx, cy = x + w // 2, y + h // 2

                if is_recent:
                    box_color = (0, 255, 0)
                    label = f"{last_name} ({last_id})"
                    sub_label = f"{last_course} | {last_year}"
                    banner_h = sc(46)
                else:
                    box_color = (0, 255, 255)
                    label = "SCAN QR CODE"
                    sub_label = ""
                    banner_h = sc(28)

                top = max(0, y - banner_h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, sc(2))
                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, sc(15), sc(2))
                cv2.rectangle(frame, (x, top), (x + max(w, sc(250)), top + banner_h), box_color, -1)
                cv2.putText(frame, label, (x + sc(5), top + sc(20)), cv2.FONT_HERSHEY_SIMPLEX, 0.52 * scale, (0, 0, 0), sc(2))
                if sub_label:
                    cv2.putText(frame, sub_label, (x + sc(5), top + sc(39)), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * scale, (0, 0, 0), sc(2))

            # Top Dashboard HUD
            cv2.rectangle(frame, (0, 0), (disp_w, sc(82)), (25, 25, 25), -1)
            cv2.putText(frame, f"SMART ATTEND | CAM {cam_index} | {int(smooth_fps)} FPS", (sc(12), sc(24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), sc(2))

            if is_recent:
                status_text = f"QR SUCCESS! {last_name} ({last_id}) - Wait {time_remaining}s"
                detail_text = f"{last_course} | {last_year}"
                status_color = (0, 255, 0)
            elif last_id != "":
                status_text = f"Last: {last_name} ({last_id}) - Ready for next scan"
                detail_text = f"{last_course} | {last_year}"
                status_color = (255, 255, 0)
            else:
                status_text = "STATUS: Waiting for Student ID QR Code..."
                detail_text = ""
                status_color = (0, 255, 255)

            cv2.putText(frame, status_text, (sc(12), sc(50)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, status_color, sc(2))
            if detail_text:
                cv2.putText(frame, detail_text, (sc(12), sc(72)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, status_color, sc(2))

            # Bottom Controls Bar
            cv2.rectangle(frame, (0, disp_h - sc(25)), (disp_w, disp_h), (25, 25, 25), -1)
            cv2.putText(frame, "[C: Cam | R: Reset | F: Fullscreen | Q: Exit]", (sc(12), disp_h - sc(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48 * scale, (0, 140, 255), sc(2))

            # Pad borders if window aspect ratio differs from camera
            if win_w > 1 and win_h > 1:
                pad_y, pad_x = max(0, win_h - disp_h), max(0, win_w - disp_w)
                if pad_x or pad_y:
                    frame = cv2.copyMakeBorder(
                        frame, pad_y // 2, pad_y - pad_y // 2,
                        pad_x // 2, pad_x - pad_x // 2,
                        cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                cam_index = (cam_index + 1) % 4
                cap.release()
                cap = open_camera(cam_index)
                with state_lock:
                    state["faces"] = []
                    state["bbox"] = None
                print(f"[INFO] Switched to Camera Index {cam_index}")
            elif key == ord("r"):
                if init_csv(reset=True):
                    with state_lock:
                        state["last_scanned_id"] = ""
                        state["last_scanned_name"] = ""
                        state["last_scanned_course"] = ""
                        state["last_scanned_year"] = ""
                        state["last_log_time"] = -1e9
                    print("[INFO] Attendance records RESET!")
            elif key == ord("f"):
                is_full = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN)
            elif key == ord("q"):
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Application closed.")


if __name__ == "__main__":
    main()