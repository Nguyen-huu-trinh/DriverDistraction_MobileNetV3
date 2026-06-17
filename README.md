# Hệ Thống Nhận Diện Hành Vi Tài Xế Sử Dụng Pipeline YOLOv8 & MobileNetV3 + Mixup (Version Hoàn Thiện)

Kho lưu trữ này chứa mã nguồn hoàn thiện của hệ thống học sâu hiệu năng cao, tối ưu phần cứng để nhận diện chính xác **10 nhóm hành vi của tài xế** (từ $c0$ đến $c9$). Mô hình ứng dụng cấu trúc Pipeline liên kết giữa mạng định vị **YOLOv8 Nano** và mạng phân loại tinh gọn **MobileNetV3 Large**, kết hợp chiến lược chỉnh quy hóa nội suy tuyến tính **Mixup** để tối ưu hóa ranh giới quyết định.

---

## 🎯 Kết Quả Thực Nghiệm Tối Ưu (Final Metrics)

* **Độ chính xác Tập Kiểm Thử (Validation Accuracy):** **`89.25%`**
* **Độ chuẩn xác Trung bình (Average Precision):** **`~91%`**
* **Đặc tính vận hành:** Đạt chuẩn **Real-time** (Thời gian thực), tối ưu hóa luồng xử lý song song trên phần cứng giới hạn (Edge Devices/Nhúng).

---

## 🏗️ Kiến Trúc Hệ Thống (Pipeline Architecture)

Mô hình vận hành theo luồng xử lý tuyến tính khép kín nhằm tối ưu hóa dòng thông tin đầu vào:

[Ảnh Thô] ──> [YOLOv8 Nano] ──> [Cắt Thân Người] ──> [MobileNetV3 Backbone] ──> [Custom Classifier Head] ──> [10 Lớp Hành Vi]


1.  **Tiền xử lý & Khu trú (YOLOv8 Nano):** Tự động phát hiện và cắt sát (crop) vùng thân người tài xế (`person` / Class ID = 0), triệt tiêu hoàn toàn bối cảnh nhiễu không liên quan trong cabin xe.
2.  **Trích xuất đặc trưng (MobileNetV3 Backbone):** Áp dụng cấu trúc *Depthwise Separable Convolution* để giảm tham số, phối hợp cùng khối chú ý *Squeeze-and-Excitation (SE)* để tự động tập trung vào các đặc trưng vi mô (tay, điện thoại, vùng mặt).
3.  **Tầng phân loại tùy biến (Custom Classifier Head):** Cắt bỏ tầng phân loại 1000 lớp mặc định của ImageNet, tái cấu trúc lại theo dạng hình phễu tối ưu:
    * **Global Average Pooling (GAP)** nén dữ liệu về vector 960 chiều.
    * **Tầng ẩn trung gian 1024 neurons** với hàm kích hoạt `Hard-Swish` (tối ưu tính toán song song trên GPU).
    * Bộ lọc **Dropout với áp lực mạnh ($p = 0.6$)** kết hợp tầng tuyến tính đầu ra hạ về **10 neurons**.
