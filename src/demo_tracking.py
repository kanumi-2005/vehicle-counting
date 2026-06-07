import cv2
import time
import numpy as np
from collections import defaultdict, deque

from tracking_pipeline import TrackingPipeline
from lane_detector import LaneDetector

# =============================
# CONFIGs
# =============================
WIDTH, HEIGHT = 1280, 720
LINE_Y = 500
VIDEO_PATH = "videos/demo1.mp4"

# =============================
# INIT
# =============================
pipeline = TrackingPipeline("models/best.pt")
lane_detector = LaneDetector(WIDTH, HEIGHT)
cap = cv2.VideoCapture(VIDEO_PATH)

# =============================
# DATA
# =============================
track_history = defaultdict(lambda: deque(maxlen=30))
lane_history = defaultdict(lambda: deque(maxlen=7))
lane_stats = defaultdict(lambda: defaultdict(int))
counted_ids = set()
direction = defaultdict(lambda: None)

prev_time = time.time()

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # =============================
    # FPS
    # =============================
    curr = time.time()
    fps = 1 / max(curr - prev_time, 1e-6)
    prev_time = curr

    # =============================
    # DETECTION + TRACKING
    # =============================
    det = pipeline.detect(frame)
    tracked = pipeline.track(det)

    # =============================
    # DRAW LANES (DYNAMIC PERSPECTIVE)
    # =============================
    frame = lane_detector.draw(frame)

    # =============================
    # COUNT LINE
    # =============================
    cv2.line(frame, (0, LINE_Y), (WIDTH, LINE_Y), (0, 0, 255), 2)

    # =============================
    # SAFETY CHECK
    # =============================
    if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
        cv2.imshow("AI Traffic System (HYBRID LANE)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # =============================
    # PROCESS OBJECTS
    # =============================
    for box, tid, cls_id in zip(
        tracked.xyxy, tracked.tracker_id, tracked.class_id
    ):

        if tid < 0:
            continue

        x1, y1, x2, y2 = map(int, box)

        # Sử dụng điểm tiếp đất đáy-tâm (bottom-center)
        cx = (x1 + x2) // 2
        cy = y2

        class_name = pipeline.model.names[int(cls_id)]

        # =============================
        # DYNAMIC LANE DETECTION (FIXED)
        # =============================
        lane_index = lane_detector.get_lane_index(cx, cy)
        lane = f"Lane {lane_index + 1}"

        # =============================
        # LANE SMOOTHING
        # =============================
        lane_history[tid].append(lane)
        lane = max(
            set(lane_history[tid]), key=lambda x: list(lane_history[tid]).count(x)
        )

        # =============================
        # TRACK HISTORY
        # =============================
        track_history[tid].append((cx, cy))
        pts = list(track_history[tid])

        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (255, 255, 255), 2)

        # =============================
        # DIRECTION
        # =============================
        if len(pts) >= 2:
            prev_y = pts[-2][1]
            curr_y = pts[-1][1]

            if direction[tid] is None:
                direction[tid] = curr_y - prev_y

        # =============================
        # COUNTING
        # =============================
        if len(pts) >= 2:
            prev_y = pts[-2][1]
            curr_y = pts[-1][1]

            if (
                tid not in counted_ids
                and prev_y < LINE_Y <= curr_y
                and direction[tid] is not None
                and direction[tid] > 0
            ):
                lane_stats[lane][class_name] += 1
                counted_ids.add(tid)

        # =============================
        # DRAW BOX & LABELS
        # =============================
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.putText(
            frame,
            f"{class_name} | {lane} | #{tid}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # =============================
    # MODERN HUD DASHBOARD (UPGRADED)
    # =============================
    # 1. Khởi tạo vị trí và tính toán chiều cao động cho bảng điều khiển
    x_start, y_start = 20, 20
    db_w = 400
    
    base_h = 95  # Chiều cao phần tiêu đề + thông số tổng quan
    lane_h = 28  # Chiều cao mỗi dòng Làn đường
    class_h = 20 # Chiều cao mỗi dòng loại xe
    
    total_content_h = base_h + (len(lane_stats) * lane_h) + (sum(len(v) for v in lane_stats.values()) * class_h)
    y_end = min(y_start + total_content_h + 10, HEIGHT - 20)
    x_end = x_start + db_w

    # 2. Tạo hiệu ứng nền kính mờ (Semi-transparent background)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (20, 24, 33), -1)      # Khối nền tối màu xanh rêu/vũ trụ
    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)     # Viền Neon lam mỏng sắc nét
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)                            # Hòa trộn lớp phủ

    # 3. Vẽ Tiêu đề Dashboard
    cv2.putText(frame, "REAL-TIME TRAFFIC AI", (x_start + 15, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.line(frame, (x_start + 15, y_start + 40), (x_end - 15, y_start + 40), (80, 90, 100), 1)

    # 4. Hiển thị thông số chung (FPS & TOTAL) bên cạnh nhau gọn gàng
    cv2.putText(frame, f"FPS: {fps:.1f}", (x_start + 15, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame, f"TOTAL COUNT: {len(counted_ids)}", (x_start + 180, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    cv2.line(frame, (x_start + 15, y_start + 78), (x_end - 15, y_start + 78), (80, 90, 100), 1)

    # 5. Duyệt và hiển thị danh sách dòng dữ liệu động theo từng Làn
    y_pos = y_start + 105
    for lane in sorted(lane_stats.keys()):
        if y_pos + lane_h > y_end: 
            break  # Cơ chế an toàn chống tràn khung hình
        
        total_lane = sum(lane_stats[lane].values())
        
        # Vẽ một ô vuông nhỏ làm Bullet Point cho Làn đường sinh động hơn
        cv2.rectangle(frame, (x_start + 15, y_pos - 11), (x_start + 23, y_pos - 3), (255, 215, 0), -1)
        
        # In tên Làn đường và Tổng số xe trong làn đó
        cv2.putText(frame, f"{lane.upper()}", (x_start + 32, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
        cv2.putText(frame, f"Total: {total_lane}", (x_start + 260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        y_pos += lane_h

        # Duyệt qua các loại xe trong làn đó
        for k, v in lane_stats[lane].items():
            if y_pos + class_h > y_end: 
                break
                
            # Thêm ký tự phân cấp nhánh "> " trực quan
            cv2.putText(frame, f"  > {k.capitalize()}:", (x_start + 35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(frame, f"{v}", (x_start + 180, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_pos += class_h
            
        y_pos += 8  # Tạo khoảng cách nhỏ trước khi sang Làn tiếp theo

    # =============================
    # SHOW
    # =============================
    cv2.imshow("AI Traffic System (HYBRID LANE)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Bấm nút ESC để thoát chương trình
        break

cap.release()
cv2.destroyAllWindows()