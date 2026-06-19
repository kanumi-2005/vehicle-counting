import cv2
import time
import argparse
import numpy as np
import os
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict, deque

# LƯU Ý: Đảm bảo các file/module này nằm cùng thư mục với script này
from tracking_pipeline import TrackingPipeline

class TrafficSystem:
    def __init__(self, video_path: str, model_path: str, width: int = 1280, height: int = 720):
        self.WIDTH: int = width
        self.HEIGHT: int = height
        
        # Lưu danh sách các làn dạng Đa giác: [ [ (x1,y1), (x2,y2),... ], [ Làn 2 ] ]
        self.lane_polygons: List[List[Tuple[int, int]]] = []
        self.current_poly: List[Tuple[int, int]] = [] # Đa giác đang vẽ dở
        self.mouse_curr: Tuple[int, int] = (0, 0)
        
        # Khởi tạo Video Capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Không thể mở hoặc tìm thấy video tại: {video_path}")
            
        # ==========================================
        # SETUP VIDEO WRITER (THÊM MỚI Ở ĐÂY)
        # ==========================================
        os.makedirs("demo", exist_ok=True)
        output_path = os.path.join("demo", "output.mp4")
        
        # Lấy FPS gốc của video để video xuất ra mượt mà tương đương
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0 or np.isnan(original_fps):
            original_fps = 30.0 # Đặt mặc định nếu không đọc được
            
        # Định dạng codec (dùng mp4v cho đuôi .mp4)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.out = cv2.VideoWriter(output_path, fourcc, original_fps, (self.WIDTH, self.HEIGHT))
        print(f"[INFO] Video kết quả sẽ được lưu tại: {output_path}")
        # ==========================================

        self.pipeline = TrackingPipeline(model_path, confidence_threshold=0.3, device="auto", verbose=False, tracker_type="byte", lost_track_buffer=30, frame_rate=30, track_activation_threshold=0.25, minimum_consecutive_frames=3, minimum_iou_threshold=0.1, high_conf_detection_threshold=0.3)
        
        # Thống kê
        self.total_cumulative_count: int = 0  # Tổng số xe từ trước đến nay
        self.counted_ids: Set[int] = set()    # Bộ lưu ID để tính tổng tích lũy
        self.prev_time: float = time.time()

        # Lịch sử di chuyển để vẽ Trajectory & Tính hướng (Vector đuôi dài 30 frames)
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Hứng sự kiện click chuột để chấm các đỉnh của đa giác"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_curr = (x, y)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.current_poly.append((x, y))
            print(f"[SETUP] -> Đã thêm đỉnh thứ {len(self.current_poly)}: ({x}, {y})")

    def interactive_setup(self) -> None:
        """Giao diện vẽ Vùng Làn Đường (Đa giác) trực quan"""
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Không thể đọc được frame để cấu hình hệ thống.")
        
        frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
        window_name = "Cau hinh Vung Lan Duong (ROI Polygons)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n" + "="*60)
        print(" HƯỚNG DẪN VẼ VÙNG LÀN ĐƯỜNG (DẠNG ĐA GIÁC):")
        print(" [BƯỚC 1]: Click chuột liên tiếp để tạo các đỉnh của LÀN 1.")
        print(" [BƯỚC 2]: Nhấn phím 'N' (Next) để đóng vùng LÀN 1 và bắt đầu vẽ LÀN TIẾP THEO.")
        print(" [BƯỚC 3]: Sau khi vẽ xong tất cả các làn, nhấn ENTER để bắt đầu CHẠY.")
        print(" [RESET] : Nhấn phím 'R' để xóa làn đang vẽ dở hoặc xóa hết nếu vẽ lại.")
        print("="*60 + "\n")

        while True:
            setup_frame = frame.copy()
            overlay = setup_frame.copy()
            
            # 1. Vẽ các đa giác làn đường đã chốt xong
            for idx, poly in enumerate(self.lane_polygons):
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], (0, 255, 255)) 
                cv2.polylines(setup_frame, [pts], True, (0, 215, 255), 2)
                
                # Ghi tên làn ở trọng tâm đa giác
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(setup_frame, f"LANE {idx+1}", (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.3, setup_frame, 0.7, 0, setup_frame)
            
            # 2. Vẽ đa giác đang vẽ dở
            if len(self.current_poly) > 0:
                pts_curr = np.array(self.current_poly, dtype=np.int32)
                cv2.polylines(setup_frame, [pts_curr], False, (0, 255, 0), 2)
                cv2.line(setup_frame, self.current_poly[-1], self.mouse_curr, (0, 255, 0), 1, cv2.LINE_AA)
                for pt in self.current_poly:
                    cv2.circle(setup_frame, pt, 4, (0, 0, 255), -1)

            # Chỉ dẫn
            # =========================================================================
            # CẢI TIẾN: VẼ HỘP NỀN HUD ĐỂ CHỮ KHÔNG BỊ CHÌM VÀO VIDEO
            # =========================================================================
            text_overlay = setup_frame.copy()
            # Tạo một hình chữ nhật tối màu làm nền ở góc trái trên
            cv2.rectangle(text_overlay, (15, 15), (650, 155), (20, 24, 33), -1)
            # Vẽ viền mỏng màu vàng chanh cho hộp hướng dẫn thêm chuyên nghiệp
            cv2.rectangle(text_overlay, (15, 15), (650, 155), (0, 255, 255), 1)
            # Trộn nền tối vào frame gốc (độ đậm 75%)
            cv2.addWeighted(text_overlay, 0.75, setup_frame, 0.25, 0, setup_frame)

            # DÒNG 1: TRẠNG THÁI HỆ THỐNG (Chữ màu Vàng)
            status_txt = f"STATUS: Da ve {len(self.lane_polygons)} Lan | Dinh cua lan hien tai: {len(self.current_poly)}"
            cv2.putText(setup_frame, status_txt, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # DÒNG 2: HƯỚNG DẪN THAO TÁC CHUỘT (Chữ màu Trắng)
            cv2.putText(setup_frame, "[CHUOT TRAI] : Click lien tiep de tao cac dinh cua da giac", 
                        (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

            # DÒNG 3: PHÍM CHỐT & CHẠY (Chữ màu Xanh Lá)
            cv2.putText(setup_frame, "[PHIM 'N']   : CHOT lan hien tai | [ENTER]: HOAN THANH & CHAY AI", 
                        (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # DÒNG 4: PHÍM RESET & THOÁT (Chữ màu Cam)
            cv2.putText(setup_frame, "[PHIM 'R']   : XOA lan ve do (hoac xoa het) | [ESC]: THOAT", 
                        (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
            
            # DÒNG 5: THÔNG BÁO ĐỘNG NHẮC NHỞ NGƯỜI DÙNG (Chữ đổi màu theo ngữ cảnh)
            if len(self.current_poly) > 0 and len(self.current_poly) < 3:
                # Nhắc nhở khi đang vẽ nhưng chưa đủ 3 góc
                cv2.putText(setup_frame, "(!) CAN IT NHAT 3 DINH DE CO THE CHOT LAN DUONG", 
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            elif len(self.lane_polygons) == 0:
                # Nhắc nhở nếu chưa vẽ làn nào mà đã đòi chạy
                cv2.putText(setup_frame, "(!) BAN CHUA CO LAN DUONG NAO. HAY VE IT NHAT 1 LAN", 
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 69, 255), 1, cv2.LINE_AA)
            else:
                # Thông báo sẵn sàng
                cv2.putText(setup_frame, "(o) HE THONG DA SAN SANG. AN ENTER DE BAT DAU KICH HOAT AI", 
                            (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            # =========================================================================s

            cv2.imshow(window_name, setup_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key in [ord('n'), ord('N')]:
                if len(self.current_poly) >= 3:
                    self.lane_polygons.append(self.current_poly.copy())
                    print(f"[SETUP] -> Đã lưu LANE {len(self.lane_polygons)}")
                    self.current_poly.clear()
                else:
                    print("[CẢNH BÁO] Vùng đa giác phải có ít nhất 3 đỉnh!")
                    
            elif key == 13: # ENTER
                if len(self.current_poly) >= 3:
                    self.lane_polygons.append(self.current_poly.copy())
                    self.current_poly.clear()
                if len(self.lane_polygons) > 0:
                    break
                else:
                    print("[CẢNH BÁO] Bạn chưa cấu hình vùng làn đường nào!")
                    
            elif key in [ord('r'), ord('R')]:
                if len(self.current_poly) > 0:
                    self.current_poly.clear()
                    print("[SETUP] -> Đã xóa đa giác đang vẽ dở.")
                else:
                    self.lane_polygons.clear()
                    print("[SETUP] -> Đã xóa toàn bộ các làn.")
                    
            elif key == 27: # ESC
                self._safely_release()
                exit()

        cv2.destroyWindow(window_name)

    def run(self) -> None:
        """Vòng lặp AI Core Inference xử lý, đếm và vẽ đường di chuyển"""
        self.interactive_setup()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
            curr_time = time.time()
            fps = 1 / max(curr_time - self.prev_time, 1e-6)
            self.prev_time = curr_time

            # Thấu kính AI nhận diện và tracking
            det = self.pipeline.detect(frame)
            tracked = self.pipeline.track(det)

            # Vẽ nền các vùng đa giác lên frame
            overlay = frame.copy()
            for idx, poly in enumerate(self.lane_polygons):
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.fillPoly(overlay, [pts], (255, 215, 0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Khởi tạo Live Stats
            live_stats: Dict[str, Dict] = {
                f"LANE {i+1}": {"total": 0, "classes": defaultdict(int)} 
                for i in range(len(self.lane_polygons))
            }
            total_live_count = 0 
            
            # Tập hợp các ID có mặt trong frame hiện tại để dọn rác (memory cleanup)
            current_frame_ids = set()

            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                for box, tid, cls_id in zip(tracked.xyxy, tracked.tracker_id, tracked.class_id):
                    if tid < 0: 
                        continue

                    current_frame_ids.add(tid)
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, y2  
                    class_name = self.pipeline.model.names[int(cls_id)]

                    # Cập nhật lịch sử di chuyển
                    self.track_history[tid].append((cx, cy))

                    # KIỂM TRA XE NẰM TRONG LÀN NÀO
                    assigned_lane_idx = None
                    for idx, poly in enumerate(self.lane_polygons):
                        poly_arr = np.array(poly, dtype=np.int32)
                        if cv2.pointPolygonTest(poly_arr, (cx, cy), False) >= 0:
                            assigned_lane_idx = idx
                            break 

                    if assigned_lane_idx is not None:
                        lane_name = f"LANE {assigned_lane_idx + 1}"
                        live_stats[lane_name]["total"] += 1
                        live_stats[lane_name]["classes"][class_name] += 1
                        total_live_count += 1
                        
                        if tid not in self.counted_ids:
                            self.counted_ids.add(tid)
                            self.total_cumulative_count += 1

                        color = (0, 255, 0) # Bbox Xanh lá
                    else:
                        color = (128, 128, 128) # Bbox Xám

                    # Vẽ Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # TÍNH TOÁN VÀ VẼ MOVEMENT LINE (TRAJECTORY)
                    history_pts = list(self.track_history[tid])
                    direction_str = ""
                    
                    if len(history_pts) >= 2:
                        # Vẽ polyline đuôi xe
                        pts_arr = np.array(history_pts, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts_arr], isClosed=False, color=(0, 165, 255), thickness=2)
                        
                        # Xác định hướng dựa trên chênh lệch Y
                        y_diff = history_pts[-1][1] - history_pts[0][1]
                        
                        # Chỉ báo hướng khi xe di chuyển đủ xa (>10 pixels để lọc nhiễu rung lắc)
                        if abs(y_diff) > 10:
                            direction_str = r"[\/] DOWN" if y_diff > 0 else r"[/\] UP"
                            
                        # Vẽ mũi tên định hướng ở đầu đoạn thẳng
                        if len(history_pts) > 5:
                            cv2.arrowedLine(frame, history_pts[-5], history_pts[-1], (255, 0, 255), 2, tipLength=0.5)

                    # Ghi nhãn Label gồm: Tên lớp + ID + Hướng đi
                    label_text = f"{class_name.upper()} #{tid} {direction_str}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

            # TỐI ƯU BỘ NHỚ: Xóa lịch sử tracking của các ID không còn xuất hiện
            lost_ids = list(set(self.track_history.keys()) - current_frame_ids)
            for lost_id in lost_ids:
                del self.track_history[lost_id]

            # Render HUD Dashboard
            self._render_hud_roi(frame, fps, total_live_count, live_stats)
            
            self.out.write(frame)
            
            cv2.imshow("AI Traffic System (Polygon Regions Flow)", frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

        self._safely_release()

    def _render_hud_roi(self, frame: np.ndarray, fps: float, total_live: int, live_stats: Dict) -> None:
        """Render Dashboard HUD động hiển thị báo cáo Live và Tổng Tích Lũy"""
        x_start, y_start = 20, 20
        db_w, base_h, lane_h, class_h = 420, 115, 28, 20 
        
        total_class_rows = sum(len(v["classes"]) for v in live_stats.values())
        total_content_h = base_h + (len(live_stats) * lane_h) + (total_class_rows * class_h)
        y_end = min(y_start + total_content_h + 10, self.HEIGHT - 20)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + db_w, y_end), (20, 24, 33), -1)      
        cv2.rectangle(overlay, (x_start, y_start), (x_start + db_w, y_end), (0, 255, 255), 1)    
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)                                    

        cv2.putText(frame, "REAL-TIME TRAFFIC AI (ROI & TRACKING)", (x_start + 15, y_start + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.line(frame, (x_start + 15, y_start + 40), (x_start + db_w - 15, y_start + 40), (80, 90, 100), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}", (x_start + 15, y_start + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"LIVE TOTAL: {total_live}", (x_start + 15, y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"ALL-TIME TOTAL: {self.total_cumulative_count}", (x_start + 180, y_start + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.line(frame, (x_start + 15, y_start + 102), (x_start + db_w - 15, y_start + 102), (80, 90, 100), 1)

        y_pos = y_start + 122
        for lane in sorted(live_stats.keys()):
            if y_pos + lane_h > y_end: 
                break 
            
            lane_live_total = live_stats[lane]["total"]
            cv2.rectangle(frame, (x_start + 15, y_pos - 11), (x_start + 23, y_pos - 3), (255, 215, 0), -1)
            cv2.putText(frame, f"{lane}", (x_start + 32, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 2)
            cv2.putText(frame, f"Live: {lane_live_total}", (x_start + 280, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)
            y_pos += lane_h

            for class_name, count_value in sorted(live_stats[lane]["classes"].items()):
                if y_pos + class_h > y_end: 
                    break
                if count_value > 0:
                    cv2.putText(frame, f"   > {class_name.upper()}:", (x_start + 35, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
                    cv2.putText(frame, f"{count_value}", (x_start + 280, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    y_pos += class_h
            y_pos += 6 

    def _safely_release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
            
        # Giải phóng stream ghi video (THÊM MỚI Ở ĐÂY)
        if hasattr(self, 'out') and self.out.isOpened():
            self.out.release()
            
        cv2.destroyAllWindows()
        print("[INFO] Da dong luong Stream va giai phong bo nho.")
        print("[INFO] File video da duoc luu tai thu muc 'demo/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="He thong AI giam sat giao thong dang da giac ROI & Trajectory")
    parser.add_argument("--video", type=str, default="video/demo3.mp4", help="Duong dan file video")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Duong dan AI model")
    args = parser.parse_args()
    
    app = TrafficSystem(video_path=args.video, model_path=args.model)
    app.run()