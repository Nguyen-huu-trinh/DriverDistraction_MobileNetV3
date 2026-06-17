import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torchvision.models as models

# 1. Khởi tạo cấu phần mô hình chạy trên CPU cho nhẹ máy demo
device = torch.device("cpu")

# Tải YOLOv8 trực tiếp từ file có sẵn trong thư mục
yolo_model = YOLO("yolov8n.pt")  

# 1. Khởi tạo cấu trúc MobileNetV3-Large gốc
mobilenet = models.mobilenet_v3_large(weights=None)

# 2. Định nghĩa lại tầng classifier chuẩn 1024 neurons giống hệt lúc train trên Kaggle
mobilenet.classifier = nn.Sequential(
    nn.Linear(in_features=960, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.6, inplace=True),
    nn.Linear(in_features=1024, out_features=10, bias=True)
)

# 3. Tiến hành nạp trọng số (Bây giờ kích thước đã khớp hoàn toàn)
mobilenet.load_state_dict(torch.load("best_driver_model_final.pth", map_location=device)) 
mobilenet.to(device)
mobilenet.eval()

# 2. Định nghĩa danh mục nhãn từ c0 đến c9
class_labels = {
    0: "Safe driving (c0)",
    1: "Texting - Right hand (c1)",
    2: "Talking on the phone - Right hand (c2)",
    3: "Texting - Left hand (c3)",
    4: "Talking on the phone - Left hand (c4)",
    5: "Operating the radio (c5)",
    6: "Drinking (c6)",
    7: "Reaching behind (c7)",
    8: "Hair and makeup (c8)",
    9: "Talking to passenger (c9)"
}

# 3. Bộ tiền xử lý ảnh cho MobileNetV3
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Hàm xử lý logic chính
def predict_driver_behavior(input_image):
    if input_image is None:
        return None, "Vui lòng tải ảnh lên!", None

    image_pil = Image.fromarray(input_image.astype('uint8'), 'RGB')
    
    # Bước A: Chạy YOLOv8 để cắt ảnh tài xế
    results = yolo_model(image_pil, verbose=False)
    crop_area = image_pil  
    
    for r in results:
        for box in r.boxes:
            if int(box.cls) == 0:  # Class 0 là Người
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_area = image_pil.crop((x1, y1, x2, y2))  
                break
                
    # Bước B: Đưa qua MobileNetV3 phân loại hành vi
    input_tensor = data_transforms(crop_area).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = mobilenet(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    confidences = {class_labels[i]: float(probabilities[i]) for i in range(10)}
    return np.array(crop_area), confidences

# 5. Thiết kế giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 HỆ THỐNG GIÁM SÁT HÀNH VI TÀI XẾ")
    gr.Markdown("Mô hình phối hợp đa nhiệm: **YOLOv8** (Lọc nhiễu cabin) và **MobileNetV3-Large** (Phân loại).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Tải ảnh tài xế lên tại đây")
            submit_btn = gr.Button("Bắt đầu phân tích hành vi", variant="primary")
            
        with gr.Column():
            cropped_img = gr.Image(label="Vùng không gian được YOLOv8 cô lập (Crop Output)")
            output_labels = gr.Label(num_top_classes=3, label="Dự đoán hành vi (Top 3 chính xác nhất)")
            
    submit_btn.click(
        fn=predict_driver_behavior, 
        inputs=input_img, 
        outputs=[cropped_img, output_labels]
    )

# 6. Kích hoạt Server chạy cục bộ và mở link công khai
demo.launch(share=True)