import cv2
import time
import argparse
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque

# Import các module pipeline từ hệ thống của bạn
from tracking_pipeline import TrackingPipeline
from lane_detector import LaneDetector

class TrafficSystem:
    def __init__(self, video_path: str, model_path: str, width: int = 1280, height: int = 720):
        self.WIDTH: int = width
        self.HEIGHT: int = height
        
        # Cấu hình đếm xe & Làn đường động
        self.LINE_IN_Y: Optional[int] = None
        self.LINE_OUT_Y: Optional[int] = None
        self.setup_clicks: List[int] = []
        
        # MỚI: Lưu danh sách các vạch chia làn dạng Đoạn thẳng xiên [((x1, y1), (x2, y2)), ...]
        self.lane_lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.temp_point: Optional[Tuple[int, int]] = None  # Điểm thứ nhất khi đang vẽ vạch dọc
        self.mouse_curr: Tuple[int, int] = (0, 0)          # Tọa độ chuột hiện tại để vẽ preview
        
        self.setup_stage: int = 1  # 1: Chọn vạch đếm (Y), 2: Tự vẽ vạch chia làn xiên

        # Khởi tạo Video Capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở hoặc tìm thấy video tại: {video_path}")
            
        self.pipeline = TrackingPipeline(model_path)
        self.lane_detector = LaneDetector(self.WIDTH, self.HEIGHT)
        
        # Cấu trúc lưu trữ dữ liệu tracking chuyên sâu
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.lane_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=7))
        self.lane_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        self.counted_ids: Set[int] = set()
        self.direction: Dict[int, Optional[float]] = defaultdict(lambda: None)
        self.prev_time: float = time.time()

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Hứng sự kiện chuột: Di chuyển để xem trước đường vẽ, Click để chốt điểm"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_curr = (x, y)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            # STAGE 1: Click 2 lần tạo vạch ngang BOUNDARY
            if self.setup_stage == 1:
                if len(self.setup_clicks) == 0:
                    self.LINE_IN_Y = y
                    self.setup_clicks.append(y)
                    print(f"[SETUP STAGE 1] -> Đã đặt vạch IN tại Y = {y}")
                elif len(self.setup_clicks) == 1:
                    self.LINE_OUT_Y = y
                    self.setup_clicks.append(y)
                    print(f"[SETUP STAGE 1] -> Đã đặt vạch OUT tại Y = {y}")
            
            # STAGE 2: Click 2 lần để TỰ VẼ 1 đường phân làn chéo bất kỳ
            elif self.setup_stage == 2:
                if self.temp_point is None:
                    self.temp_point = (x, y)
                    print(f"[SETUP STAGE 2] -> Đã chọn điểm ĐẦU của vạch tại: ({x}, {y})")
                else:
                    self.lane_lines.append((self.temp_point, (x, y)))
                    # Sắp xếp các làn đường từ trái qua phải dựa trên trung điểm X của đoạn thẳng
                    self.lane_lines.sort(key=lambda line: (line[0][0] + line[1][0]) / 2)
                    print(f"[SETUP STAGE 2] -> Đã nối điểm CUỐI tại: ({x}, {y}). Tạo thành công vạch xiên!")
                    self.temp_point = None  # Reset để sẵn sàng vẽ đường tiếp theo

    def interactive_setup(self) -> None:
        """Giao diện đồ họa tương tác đa nhiệm thời gian thực"""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Không thể đọc được frame để cấu hình hệ thống.")
        
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        window_name = "Cau hinh he thong Traffic AI"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "="*60)
        print(" HƯỚNG DẪN TỰ VẼ LÀN ĐƯỜNG (CHỐNG GÓC CAMERA NGHIÊNG):")
        print(" [BƯỚC 1]: Click 2 lần tạo vạch Đếm IN (Xanh) & OUT (Đỏ). Nhấn ENTER.")
        print(" [BƯỚC 2]: TỰ VẼ VẠCH CHIA LÀN:")
        print("           - Click điểm thứ nhất (Đầu đường vạch dọc)")
        print("           - Di chuột (sẽ có đường xem trước) -> Click điểm thứ hai để chốt vạch.")
        print("           - Có thể vẽ bao nhiêu vạch tùy thích. Vẽ xong nhấn ENTER để chạy.")
        print(" [RESET] : Nhấn phím 'R' để xóa làm lại bước hiện tại.")
        print("="*60 + "\n")

        while True:
            setup_frame = frame.copy()
            
            # 1. Vẽ các vạch đếm (Ngang) của Bước 1
            if self.LINE_IN_Y is not None:
                cv2.line(setup_frame, (0, self.LINE_IN_Y), (self.WIDTH, self.LINE_IN_Y), (0, 255, 0), 2)
            if self.LINE_OUT_Y is not None:
                cv2.line(setup_frame, (0, self.LINE_OUT_Y), (self.WIDTH, self.LINE_OUT_Y), (0, 0, 255), 2)
            
            # 2. Vẽ các vạch chia làn đã chốt xong ở Bước 2
            for p1, p2 in self.lane_lines:
                cv2.line(setup_frame, p1, p2, (0, 215, 255), 2, cv2.LINE_AA)
            
            # FIX: Thuật toán tự vẽ nét đứt thủ công cho đường xem trước (Rubber-band)
            if self.setup_stage == 2 and self.temp_point is not None:
                p1 = self.temp_point
                p2 = self.mouse_curr
                # Tính khoảng cách giữa 2 điểm
                dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                if dist > 0:
                    # Chia đoạn thẳng thành các điểm nhỏ cách nhau 10 pixel
                    dash_length = 10
                    points = []
                    for i in range(0, int(dist), dash_length):
                        r = i / dist
                        x = int(p1[0] + (p2[0] - p1[0]) * r)
                        y = int(p1[1] + (p2[1] - p1[1]) * r)
                        points.append((x, y))
                    points.append(p2)
                    
                    # Nối các cặp điểm cách quãng để tạo hiệu ứng nét đứt (Dashed line)
                    for i in range(0, len(points) - 1, 2):
                        cv2.line(setup_frame, points[i], points[i+1], (255, 100, 0), 2, cv2.LINE_AA)

            # Chỉ dẫn Tiếng Việt hiển thị động trên màn hình
            if self.setup_stage == 1:
                cv2.putText(setup_frame, "BUOC 1: Click 2 lan tao vach ngang IN/OUT -> Bam ENTER", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                cv2.putText(setup_frame, "BUOC 2: Click 2 diem de TU VE vach chia lan bat ky -> Bam ENTER de CHAY", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow(window_name, setup_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13: # Phím ENTER
                if self.setup_stage == 1 and len(self.setup_clicks) == 2:
                    self.setup_stage = 2
                    print("[SETUP] -> Chuyển sang STAGE 2: Tự vẽ vạch phân làn.")
                elif self.setup_stage == 2:
                    break  
                    
            elif key in [ord('r'), ord('R')]: # Phím R để xóa vẽ lại
                if self.setup_stage == 1:
                    self.LINE_IN_Y, self.LINE_OUT_Y = None, None
                    self.setup_clicks.clear()
                else:
                    self.lane_lines.clear()
                    self.temp_point = None
                print(f"[SETUP] -> Đã reset cấu hình của STAGE {self.setup_stage}.")
                
            elif key == 27:
                self._safely_release()
                exit()

        # Khởi tạo sẵn danh sách làn trên HUD
        for i in range(len(self.lane_lines) + 1):
            self.lane_stats[f"LANE {i+1}"] = {}

        cv2.destroyWindow(window_name)

    def _get_dynamic_lane(self, cx: int, cy: int) -> str:
        """Thuật toán Hình học giải tích: Xác định xe thuộc làn nào dựa trên vị trí chéo của vạch tự vẽ"""
        if not self.lane_lines:
            return "LANE 1"
        
        # Duyệt qua các đường xiên từ trái sang phải
        for idx, (p1, p2) in enumerate(self.lane_lines):
            x1, y1 = p1
            x2, y2 = p2
            
            # Tìm điểm X giới hạn của đường xiên tại cao độ Y (cy) hiện tại của xe
            if y2 != y1:
                x_border = x1 + (cy - y1) * (x2 - x1) / (y2 - y1)
            else:
                x_border = x1
            
            # Nếu tọa độ X của xe nhỏ hơn điểm biên này -> Xe nằm bên trái vạch -> Thuộc làn hiện tại
            if cx < x_border:
                return f"LANE {idx + 1}"
                
        return f"LANE {len(self.lane_lines) + 1}"

    def _draw_dynamic_infrastructure(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ cơ sở hạ tầng giao thông xiên kèm nhãn tên làn"""
        if self.LINE_IN_Y is None or self.LINE_OUT_Y is None:
            return frame

        y_min = min(self.LINE_IN_Y, self.LINE_OUT_Y)
        
        # Vẽ vạch IN / OUT BOUNDARY
        cv2.line(frame, (0, self.LINE_IN_Y), (self.WIDTH, self.LINE_IN_Y), (0, 255, 0), 2)
        cv2.line(frame, (0, self.LINE_OUT_Y), (self.WIDTH, self.LINE_OUT_Y), (0, 0, 255), 2)
        
        # Vẽ các vạch dọc xiên do người dùng tự vẽ bằng chuột
        for p1, p2 in self.lane_lines:
            cv2.line(frame, p1, p2, (0, 215, 255), 2, cv2.LINE_AA)
            
        # Hiển thị text tên làn động ở khoảng giữa vùng đếm
        # Định vị nhãn dựa trên vị trí trung tâm của các làn đường xiên
        display_y = y_min + 30
        prev_x = 0
        for idx, (p1, p2) in enumerate(self.lane_lines):
            # Tính điểm cắt X của vạch tại dòng chữ hiển thị
            x1, y1 = p1
            x2, y2 = p2
            curr_x = int(x1 + (display_y - y1) * (x2 - x1) / (y2 - y1)) if y2 != y1 else x1
            
            mid_x = (prev_x + curr_x) // 2
            cv2.putText(frame, f"LANE {idx+1}", (mid_x - 30, display_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            prev_x = curr_x
            
        # Làn cuối cùng (bên phải cùng)
        mid_x_last = (prev_x + self.WIDTH) // 2
        cv2.putText(frame, f"LANE {len(self.lane_lines)+1}", (mid_x_last - 30, display_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            
        return frame

    def run(self) -> None:
        """Vòng lặp Core chạy AI Inference, Tracking và xử lý đếm xe thực thời"""
        self.interactive_setup()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[INFO] Đã xử lý xong video stream hoặc hết file video.")
                break

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))

            curr_time = time.time()
            fps = 1 / max(curr_time - self.prev_time, 1e-6)
            self.prev_time = curr_time

            det = self.pipeline.detect(frame)
            tracked = self.pipeline.track(det)

            frame = self._draw_dynamic_infrastructure(frame)

            if tracked.tracker_id is None or len(tracked.tracker_id) == 0:
                self._render_hud(frame, fps)
                cv2.imshow("AI Traffic System (Dynamic Lane Cau Hinh)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            for box, tid, cls_id in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):
                if tid < 0: 
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, y2  
                class_name = self.pipeline.model.names[int(cls_id)]

                # CẢI TIẾN QUAN TRỌNG: Truyền cả cx và cy vào hàm tính toán hình học giải tích đường xiên
                detected_lane = self._get_dynamic_lane(cx, cy)
                self.lane_history[tid].append(detected_lane)
                lane = max(set(self.lane_history[tid]), key=lambda x: list(self.lane_history[tid]).count(x))

                self.track_history[tid].append((cx, cy))
                pts = list(self.track_history[tid])
                move_color = (255, 255, 255) 
                
                if len(pts) >= 2:
                    prev_y, curr_y = pts[-2][1], pts[-1][1]
                    
                    if self.direction[tid] is None and (curr_y - prev_y) != 0:
                        self.direction[tid] = curr_y - prev_y

                    current_dir = self.direction[tid]
                    if current_dir is not None:
                        move_color = (0, 255, 0) if current_dir > 0 else (0, 0, 255)

                    for i in range(1, len(pts)):
                        cv2.line(frame, pts[i - 1], pts[i], move_color, 2)

                    if tid not in self.counted_ids:
                        if f"{class_name} (IN)" not in self.lane_stats[lane]:
                            self.lane_stats[lane][f"{class_name} (IN)"] = 0
                        if f"{class_name} (OUT)" not in self.lane_stats[lane]:
                            self.lane_stats[lane][f"{class_name} (OUT)"] = 0

                        if self.LINE_IN_Y is not None and current_dir is not None and current_dir > 0:
                            if prev_y < self.LINE_IN_Y <= curr_y:
                                self.lane_stats[lane][f"{class_name} (IN)"] += 1
                                self.counted_ids.add(tid)
                        
                        if self.LINE_OUT_Y is not None and current_dir is not None and current_dir < 0:
                            if prev_y >= self.LINE_OUT_Y > curr_y:
                                self.lane_stats[lane][f"{class_name} (OUT)"] += 1
                                self.counted_ids.add(tid)

                cv2.rectangle(frame, (x1, y1), (x2, y2), move_color, 2)
                label_text = f"{class_name.upper()} | {lane} | #{tid}"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, move_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            self._render_hud(frame, fps)
            cv2.imshow("AI Traffic System (Dynamic Lane Cau Hinh)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self._safely_release()

    def _render_hud(self, frame: np.ndarray, fps: float) -> None:
        """Render Dashboard HUD tự động mở rộng theo số lượng làn vừa vẽ"""
        x_start, y_start = 20, 20
        db_w, base_h, lane_h, class_h = 400, 95, 28, 20 
        
        total_content_h = base_h + (len(self.lane_stats) * lane_h) + (sum(len(v) for v in self.lane_stats.values()) * class_h)
        y_end = min(y_start + total_content_h + 10, self.HEIGHT - 20)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + db_w, y_end), (20, 24, 33), -1)      
        cv2.rectangle(overlay, (x_start, y_start), (x_start + db_w, y_end), (0, 255, 255), 1)    
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)                                    

        cv2.putText(frame, "REAL-TIME TRAFFIC AI", (x_start + 15, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.line(frame, (x_start + 15, y_start + 40), (x_start + db_w - 15, y_start + 40), (80, 90, 100), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (x_start + 15, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"TOTAL COUNT: {len(self.counted_ids)}", (x_start + 180, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        
        y_pos = y_start + 105
        for lane in sorted(self.lane_stats.keys()):
            if y_pos + lane_h > y_end: 
                break 
            
            total_lane = sum(self.lane_stats[lane].values())
            cv2.rectangle(frame, (x_start + 15, y_pos - 11), (x_start + 23, y_pos - 3), (255, 215, 0), -1)
            cv2.putText(frame, f"{lane}", (x_start + 32, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
            cv2.putText(frame, f"Total: {total_lane}", (x_start + 260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
            y_pos += lane_h

            for class_flow, count_value in sorted(self.lane_stats[lane].items()):
                if y_pos + class_h > y_end: 
                    break
                cv2.putText(frame, f"  > {class_flow}:", (x_start + 35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(frame, f"{count_value}", (x_start + 240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                y_pos += class_h
            y_pos += 8 

    def _safely_release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Da giai phong video stream.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="He thong AI giam sat giao thong thuc te")
    parser.add_argument("--video", type=str, default="videos/demo1.mp4", help="Duong dan file video")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Duong dan AI model")
    args = parser.parse_args()
    
    app = TrafficSystem(video_path=args.video, model_path=args.model)
    app.run()