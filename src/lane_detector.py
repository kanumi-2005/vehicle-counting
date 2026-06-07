import cv2
import numpy as np
from collections import deque


class LaneDetector:

    def __init__(self, width=1280, height=720):
        self.W = width
        self.H = height

        # Cao độ xử lý (Từ chân trời xuống sát đáy màn hình)
        self.far_y = 350
        self.near_y = 720

        # NÂNG CẤP 1: Mở rộng rộng rãi đỉnh hình thang ROI để không làm mất vạch biên ở xa
        self.roi_vertices = np.array(
            [[
                (10, self.near_y),  # Dưới - Trái
                (self.W - 10, self.near_y),  # Dưới - Phải
                (900, self.far_y),  # Trên - Phải (Mở rộng từ 770)
                (380, self.far_y),  # Trên - Trái (Mở rộng từ 470)
            ]],
            np.int32,
        )

        # Bộ đệm trung bình động giúp ổn định vùng màu giữa các khung hình
        self.left_history = deque(maxlen=25)
        self.right_history = deque(maxlen=25)

        # Cấu hình lề chạy dự phòng gốc nếu mất dấu vạch hoàn toàn khi khởi động
        self.default_left = [-1.054, 858.88]
        self.default_right = [1.310, 246.80]

    def _preprocess(self, frame):
        """NÂNG CẤP 2: Kết hợp Mặt nạ màu HSV và Canny để lọc bỏ hoàn toàn nhiễu

        từ vỏ xe/con lươn.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lọc vạch sơn màu Trắng (Độ bão hòa thấp, giá trị độ sáng cao)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 45, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # Lọc vạch sơn màu Vàng (Nếu có)
        lower_yellow = np.array([15, 70, 100])
        upper_yellow = np.array([36, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Gộp chung mặt nạ màu sắc vạch đường
        mask_color = cv2.bitwise_or(mask_white, mask_yellow)

        # Trích xuất cạnh bằng Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)  # Giảm ngưỡng để nhạy hơn với lề xa

        # Chỉ giữ lại các đường Canny trùng khớp với màu vạch sơn đường
        lane_edges = cv2.bitwise_and(edges, mask_color)

        # Áp dụng cắt vùng ROI hình thang phối cảnh
        mask_roi = np.zeros_like(lane_edges)
        cv2.fillPoly(mask_roi, self.roi_vertices, 255)

        return cv2.bitwise_and(lane_edges, mask_roi)

    def _get_edges_and_fit(self, edge_img):
        """Thuật toán nới rộng biên độ: Chủ động tìm và ép dải màu dạt ra 

        vạch liền ngoài cùng của lộ giới, loại bỏ hoàn toàn việc co cụm vào giữa.
        """
        MAX_HISTORY = 20
        if not hasattr(self, 'left_history'): self.left_history = []
        if not hasattr(self, 'right_history'): self.right_history = []

        # BƯỚC 1: Cấu hình HoughLinesP thông thoáng hơn để bắt được vạch biên ở xa
        lines = cv2.HoughLinesP(
            edge_img,
            1,
            np.pi / 180,
            threshold=30,        # Hạ ngưỡng một chút để bắt các vạch lề bị mờ/bụi bẩn che
            minLineLength=40,    # Độ dài tối thiểu của vạch
            maxLineGap=100,      # Cho phép nối các đoạn đứt quãng xa hơn
        )

        valid_left_fits = []
        valid_right_fits = []

        # Lấy điểm neo lịch sử để làm tâm ngắm
        if self.left_history:
            hist_avg = np.mean(self.left_history, axis=0)
            target_left_x = hist_avg[0] * self.near_y + hist_avg[1]
        else:
            target_left_x = self.default_left[0] * self.near_y + self.default_left[1]

        if self.right_history:
            hist_avg = np.mean(self.right_history, axis=0)
            target_right_x = hist_avg[0] * self.near_y + hist_avg[1]
        else:
            target_right_x = self.default_right[0] * self.near_y + self.default_right[1]

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2 or y1 == y2:
                    continue

                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                fit = np.polyfit([y1, y2], [x1, x2], 1)
                a, b = fit
                x_at_bottom = a * self.near_y + b
                slope = (y2 - y1) / float(x2 - x1)

                # -----------------------------------------------------------------
                # CẢI TIẾN 1: NỚI LỎNG GÓC NGHIÊNG VÀ NỚI RỘNG KHÔNG GIAN QUÉT (MỞ RỘNG BIÊN)
                # -----------------------------------------------------------------
                # Hạ giới hạn góc dốc xuống 0.2 (hoặc 0.15) để bắt các vạch nghiêng hẳn ra góc màn hình
                if -4.0 < slope < -0.2:  
                    # Nới rộng hành lang an toàn từ 120px lên 350px để thuật toán dám vươn ra rìa ảnh
                    if abs(x_at_bottom - target_left_x) < 350:
                        valid_left_fits.append((fit, x_at_bottom, length))

                elif 0.2 < slope < 4.0:  
                    if abs(x_at_bottom - target_right_x) < 350:
                        valid_right_fits.append((fit, x_at_bottom, length))

        # -------------------------------------------------------------------------
        # CẢI TIẾN 2: CHIẾN LƯỢC CHỌN ĐƯỜNG "NGOÀI CÙNG THOẢ MÃN ĐỘ DÀI"
        # -------------------------------------------------------------------------
        # --- Xử lý bên Trái ---
        if valid_left_fits:
            # Lọc lấy các vạch có độ dài đủ tin cậy ( > 45px) để loại bỏ nhiễu rác ở vỉa hè
            confident_lines = [item for item in valid_left_fits if item[2] > 45]
            if not confident_lines: 
                confident_lines = valid_left_fits # Khôi phục nếu không có đường nào đủ dài
            
            # Trong các đường đủ dài, SẮP XẾP THEO X_AT_BOTTOM TĂNG DẦN để lấy đường ngoài cùng bên trái (X nhỏ nhất)
            confident_lines.sort(key=lambda item: item[1])
            self.left_history.append(confident_lines[0][0])
            
            if len(self.left_history) > MAX_HISTORY:
                self.left_history.popleft()

        # --- Xử lý bên Phải ---
        if valid_right_fits:
            confident_lines = [item for item in valid_right_fits if item[2] > 45]
            if not confident_lines: 
                confident_lines = valid_right_fits
                
            # SẮP XẾP THEO X_AT_BOTTOM GIẢM DẦN để lấy đường ngoài cùng bên phải (X lớn nhất)
            confident_lines.sort(key=lambda item: item[1], reverse=True)
            self.right_history.append(confident_lines[0][0])
            
            if len(self.right_history) > MAX_HISTORY:
                self.right_history.popleft()

        # BƯỚC 3: Xuất kết quả
        final_left = np.mean(self.left_history, axis=0) if self.left_history else self.default_left
        final_right = np.mean(self.right_history, axis=0) if self.right_history else self.default_right

        return final_left, final_right

    def draw(self, frame, alpha=0.25):
        """Tự động phân tách phối cảnh và phủ đều 4 dải màu lên toàn lộ giới."""
        overlay = frame.copy()

        edges = self._preprocess(frame)
        left_fit, right_fit = self._get_edges_and_fit(edges)

        al, bl = left_fit
        ar, br = right_fit

        colors = [
            (255, 100, 100),  # Làn 1: Xanh dương (Biên trái)
            (255, 100, 100),  # Làn 2: Xanh lá
            (100, 230, 255),  # Làn 3: Vàng
            (100, 230, 255),  # Làn 4: Hồng (Biên phải)
        ]

        for i in range(4):
            t1 = i / 4.0
            t2 = (i + 1) / 4.0

            # Tính tọa độ X tại điểm XA (far_y)
            lx_far = al * self.far_y + bl
            rx_far = ar * self.far_y + br
            x_far_a = int(lx_far + (rx_far - lx_far) * t1)
            x_far_b = int(lx_far + (rx_far - lx_far) * t2)

            # Tính tọa độ X tại điểm GẦN (near_y)
            lx_near = al * self.near_y + bl
            rx_near = ar * self.near_y + br
            x_near_a = int(lx_near + (rx_near - lx_near) * t1)
            x_near_b = int(lx_near + (rx_near - lx_near) * t2)

            lane_poly = np.array(
                [
                    [
                        (x_near_a, self.near_y),
                        (x_near_b, self.near_y),
                        (x_far_b, self.far_y),
                        (x_far_a, self.far_y),
                    ]
                ],
                np.int32,
            )

            # cv2.fillPoly(overlay, lane_poly, colors[i])
            # cv2.polylines(overlay, lane_poly, True, (40, 40, 40), 1)

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def get_lane_index(self, cx, cy):
        """Xác định vị trí làn xe động không phụ thuộc góc quay cứng camera."""
        cy = max(self.far_y, min(self.near_y, cy))

        al, bl = (
            np.mean(self.left_history, axis=0)
            if self.left_history
            else self.default_left
        )
        ar, br = (
            np.mean(self.right_history, axis=0)
            if self.right_history
            else self.default_right
        )

        lx = al * cy + bl
        rx = ar * cy + br

        if rx <= lx:
            return 2

        lane_fraction = (cx - lx) / float(rx - lx)
        lane_index = int(lane_fraction * 4)

        return max(0, min(3, lane_index))