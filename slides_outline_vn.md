# Đề cương Slide Thuyết trình: Nhận diện Hành vi Bạo lực từ CCTV

Dưới đây là cấu trúc chi tiết cho các slide thuyết trình của bạn. Bạn có thể sao chép nội dung này vào PowerPoint hoặc Google Slides.

---

## Slide 1: Trang tiêu đề
*   **Tiêu đề chính:** Nhận diện Hành vi Bạo lực từ Camera Giám sát (CCTV)
*   **Tiêu đề phụ:** Sử dụng mạng Nơ-ron Không gian - Thời gian (MobileNetV2 + BiLSTM) kết hợp Cơ chế Chú ý (Temporal Attention)
*   **Người thực hiện:** [Tên của bạn]
*   **Hình ảnh gợi ý:** Ảnh minh họa hệ thống camera giám sát hiện đại với các khung nhận diện.

---

## Slide 2: Đặt vấn đề & Mục tiêu
*   **Vấn đề:**
    *   Sự gia tăng của các hành vi bạo lực tại nơi công cộng.
    *   Hệ thống CCTV hiện nay chủ yếu dùng để xem lại (passive), chưa có khả năng cảnh báo thời gian thực (active).
*   **Mục tiêu:**
    *   Xây dựng mô hình AI nhận diện bạo lực tự động.
    *   Đảm bảo độ chính xác cao và khả năng giải thích (XAI) cho quyết định của máy.

---

## Slide 3: Tổng quan giải pháp kỹ thuật (Architecture)
*   **Kiến trúc cốt lõi:**
    *   **MobileNetV2:** Trích xuất đặc trưng không gian (spatial features) từ từng khung hình. (Ưu điểm: Nhẹ, tốc độ nhanh).
    *   **BiLSTM:** Mô hình hóa mối quan hệ thời gian (temporal dynamics) giữa các khung hình.
    *   **Temporal Attention:** Cơ chế tập trung vào các khoảnh khắc quan trọng nhất trong video.
*   **Input:** Video đầu vào (128x128 pixels, 15 khung hình).

---

## Slide 4: Dữ liệu huấn luyện (Datasets)
*   **Sử dụng tổ hợp 3 bộ dữ liệu lớn:**
    1.  **RWF-2000:** 2,000 video từ thực tế.
    2.  **SCVD:** Dữ liệu camera giám sát thành phố thông minh.
    3.  **Real Life Violence Situations:** Các tình huống bạo lực đa dạng.
*   **Tổng quy mô:** ~6,000 đoạn video.
*   **Kỹ thuật:** GroupShuffleSplit để chống rò rỉ dữ liệu bối cảnh (Scene Leakage).

---

## Slide 5: Quy trình xử lý (Pipeline)
*   **Xử lý video:** Trích xuất 15 khung hình then chốt từ mỗi clip.
*   **Tăng cường dữ liệu (Augmentation):**
    *   Lật ảnh ngẫu nhiên (Horizontal Flip).
    *   Cắt ảnh ngẫu nhiên (Random Crop).
    *   Nhiễu thời gian (Temporal Dropout).
*   *Mục đích:* Giúp mô hình bền bỉ hơn với các góc quay và điều kiện ánh sáng khác nhau.

---

## Slide 6: Phân tích Cơ chế Chú ý (Temporal Attention)
*   **Vai trò:** Không phải khung hình nào trong video cũng chứa hành động bạo lực.
*   **Hoạt động:**
    *   Gán trọng số cho từng khung hình.
    *   Khung hình có sự va chạm/hành động mạnh sẽ có trọng số cao.
    *   Khung hình tĩnh/không có hành động sẽ bị bỏ qua (giảm nhiễu).
*   **Hình ảnh gợi ý:** Biểu đồ cột thể hiện trọng số của 15 khung hình.

---

## Slide 7: Kết quả đạt được
*   **Độ chính xác (Accuracy):** ~90.77%
*   **ROC-AUC:** ~0.95 (Khả năng phân biệt cực tốt).
*   **Violence Recall:** ~96% (Hạn chế tối đa bỏ sót hành vi bạo lực).
*   **Focal Loss:** Giúp mô hình học tốt hơn từ những mẫu "khó" nhận diện.

---

## Slide 8: Trí tuệ nhân tạo có thể giải thích (XAI)
*   **Tại sao cần XAI?** Để con người tin tưởng vào quyết định của AI.
*   **Minh họa:** Hiển thị các khung hình đạt điểm Attention cao nhất.
    *   *Ví dụ:* Khung hình lúc đối tượng vung nắm đấm sẽ được AI đánh dấu là quan trọng nhất.
*   **Hình ảnh gợi ý:** Chèn ảnh từ thư mục `output_results/charts/03_xai_*.png`.

---

## Slide 9: Demo Ứng dụng Web
*   **Tính năng:**
    *   Tải trực tiếp video clip lên trình duyệt.
    *   Xử lý và trả kết quả ngay lập tức (Normal/Violence).
    *   Hiển thị độ tin cậy của mô hình (Confidence score).
*   **Công nghệ:** Flask (Backend) + HTML/CSS hiện đại (Frontend).

---

## Slide 10: Tổng kết & Hướng phát triển
*   **Tổng kết:** Hệ thống hoạt động ổn định, độ chính xác cao, sẵn sàng thử nghiệm.
*   **Hướng phát triển:**
    *   Triển khai trực tiếp trên các thiết bị Edge (Jetson Nano/Raspberry Pi).
    *   Tích hợp cảnh báo qua Telegram/Email cho bộ phận an ninh.
    *   Mở rộng nhận diện thêm các hành vi khác (trộm cắp, hỏa hoạn).

---

## Slide 11: Lời kết
*   **Nội dung:** Cảm ơn thầy cô và các bạn đã lắng nghe.
*   **Thông tin liên hệ:** [Email/Tên của bạn]
