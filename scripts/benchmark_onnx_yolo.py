import os
import cv2
import numpy as np
import time
import onnxruntime as ort
from ultralytics import YOLO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 128
FRAME_COUNT = 15

def crop_person_from_frames(frames, yolo_model):
    if yolo_model is None or not frames:
        return frames
        
    mid_idx = len(frames) // 2
    img = frames[mid_idx]
    
    t0 = time.time()
    results = yolo_model(img, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_time = time.time() - t0
    
    if len(boxes) == 0:
        return None, yolo_time
        
    x1 = int(np.min(boxes[:, 0]))
    y1 = int(np.min(boxes[:, 1]))
    x2 = int(np.max(boxes[:, 2]))
    y2 = int(np.max(boxes[:, 3]))
    
    h, w = img.shape[:2]
    margin_x = int((x2 - x1) * 0.2)
    margin_y = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    
    cropped_frames = []
    for f in frames:
        cropped_frames.append(f[y1:y2, x1:x2])
        
    return cropped_frames, yolo_time

def test_pipeline(video_path):
    print(f"[*] Đang nạp mô hình...")
    yolo_model = YOLO('yolov8n.pt')
    onnx_path = 'models/CCTV_Violence_Finetuned_int8.onnx'
    ort_session = ort.InferenceSession(onnx_path)
    print(f"[+] Hoàn tất nạp mô hình.")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[-] Không thể mở video: {video_path}")
        return
        
    raw_frames = []
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(total_video_frames // FRAME_COUNT, 1)
    
    for i in range(FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames_window)
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    print(f"[*] Trích xuất {len(raw_frames)} frames từ video.")
    
    cropped, yolo_time = crop_person_from_frames(raw_frames, yolo_model)
    if cropped is None:
        print("[!] Không phát hiện người trong video.")
        return
        
    print(f"[*] YOLOv8 Crop Time: {yolo_time*1000:.2f} ms")
    
    processed_frames = []
    for frame in cropped:
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        processed_frames.append(preprocess_input(frame_resized.astype(np.float32)))
        
    input_data = np.array([processed_frames], dtype=np.float32)
    
    # Warmup ONNX
    _ = ort_session.run(None, {"input": input_data})
    
    # Đo thời gian ONNX
    t0 = time.time()
    prediction = ort_session.run(None, {"input": input_data})[0]
    onnx_time = time.time() - t0
    
    violence_prob = float(prediction[0][1])
    class_name = "Bạo lực" if violence_prob > 0.5 else "Bình thường"
    
    print(f"[*] ONNX Inference Time: {onnx_time*1000:.2f} ms")
    print(f"\n===== KẾT QUẢ PHÂN TÍCH =====")
    print(f"Loại: {class_name}")
    print(f"Độ tin cậy (Bạo lực): {violence_prob*100:.2f}%")
    print(f"Tổng thời gian xử lý: {(yolo_time + onnx_time)*1000:.2f} ms")
    print(f"FPS tương đương: {1.0 / (yolo_time + onnx_time):.1f} FPS")

if __name__ == "__main__":
    video_file = r"D:\violence-detection-cctv\outputs\cctv_violence_evt_12556188.mp4"
    if os.path.exists(video_file):
        test_pipeline(video_file)
    else:
        print(f"[-] File video không tồn tại: {video_file}")
