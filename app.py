import os
import cv2
import numpy as np
import zipfile
import tempfile
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.backend as K
from scipy.ndimage import binary_dilation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Cấu hình các tham số toàn cục
IMG_SIZE = 128
FRAME_COUNT = 15
CLASSES = ['Normal', 'Violence']

# Lưu trữ dữ liệu nhận diện tạm thời
detection_cache = {}


@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                  initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(1,),
                                  initializer='zeros', trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        alpha = K.softmax(K.squeeze(e, axis=-1))
        context = x * K.expand_dims(alpha, axis=-1)
        return K.sum(context, axis=1), alpha

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ReshapeToFrames(Layer):
    """Thay thế Lambda layer: reshape (batch, 15, 128, 128, 3) -> (batch*15, 128, 128, 3)"""
    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size * FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3))

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable()
class ReshapeToSequence(Layer):
    """Thay thế Lambda layer: reshape (batch*15, 1280) -> (batch, 15, 1280)"""
    def call(self, x):
        features = x.shape[-1]
        batch_size = tf.shape(x)[0] // FRAME_COUNT
        return tf.reshape(x, (batch_size, FRAME_COUNT, features))

    def get_config(self):
        return super().get_config()


def build_model_architecture():
    """Tái dựng kiến trúc khớp với model đã huấn luyện (dùng reshape thay Lambda)."""
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    video_input = Input(shape=(FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3))

    x = ReshapeToFrames()(video_input)          # (batch*15, 128, 128, 3)
    x = base_model(x)                           # (batch*15, 4, 4, 1280)
    x = GlobalAveragePooling2D()(x)             # (batch*15, 1280)
    x = ReshapeToSequence()(x)                  # (batch, 15, 1280)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    context_vector, _ = TemporalAttention(name='attention_layer')(x)

    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=video_input, outputs=output)


MODEL_PATH = 'models/CCTV_Violence_Finetuned.keras'
print(f"[*] Đang khởi tạo kiến trúc và nạp trọng số từ {MODEL_PATH}...")
model = build_model_architecture()

# Trích xuất model.weights.h5 từ file .keras (zip) rồi load trực tiếp
with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zf.extract('model.weights.h5', tmp_dir)
        weights_path = os.path.join(tmp_dir, 'model.weights.h5')
        model.load_weights(weights_path)

print("[+] Đã nạp mô hình thành công!")

def extract_frames(video_path):
    """Trích xuất 15 khung hình từ video và tiền xử lý."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(total_frames // FRAME_COUNT, 1)
    
    for i in range(FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames_window)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
        frames.append(preprocess_input(frame.astype(np.float32)))
    
    cap.release()
    
    # Nếu thiếu khung hình thì padding bằng giá trị 0 (đã qua preprocess_input)
    while len(frames) < FRAME_COUNT:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) - 1.0)
        
    return np.array([frames], dtype=np.float32)

def detect_violence_segments(video_path, violence_threshold=0.5, buffer_frames=15):
    """
    Phát hiện các segment bạo lực trong video bằng sliding window.
    Trả về: tuple (violence_frames_array, fps, width, height, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Đọc tất cả frames
    all_frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames_list.append(frame)
    cap.release()
    
    if len(all_frames_list) < FRAME_COUNT:
        return None, fps, width, height, total_frames
    
    # Dùng sliding window để phát hiện bạo lực cho từng frame
    violence_scores = []  # score bạo lực cho mỗi frame
    
    # Stride = 5 frames để tăng tốc độ xử lý
    stride = max(1, len(all_frames_list) // 100)  # tối đa ~100 windows
    
    for start_idx in range(0, len(all_frames_list) - FRAME_COUNT + 1, stride):
        window_frames = []
        for frame_idx in range(start_idx, start_idx + FRAME_COUNT):
            frame = all_frames_list[frame_idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
            frame_processed = preprocess_input(frame_resized.astype(np.float32))
            window_frames.append(frame_processed)
        
        # Dự đoán
        input_data = np.array([window_frames], dtype=np.float32)
        prediction = model.predict(input_data, verbose=0)
        violence_prob = prediction[0][1]  # Xác suất bạo lực
        
        # Map window prediction về tất cả frames trong window
        for offset in range(FRAME_COUNT):
            frame_idx = start_idx + offset
            violence_scores.append(violence_prob)
    
    # Pad lại để có đủ scores cho tất cả frames
    while len(violence_scores) < len(all_frames_list):
        violence_scores.append(0.0)
    
    violence_scores = violence_scores[:len(all_frames_list)]
    
    # Xác định frames bạo lực (> ngưỡng) và dilate để mở rộng vùng
    violence_binary = np.array([1 if score > violence_threshold else 0 for score in violence_scores])
    
    # Dilate để mở rộng vùng bạo lực (thêm buffer frames)
    dilation_size = buffer_frames // 2
    if dilation_size > 0:
        kernel = np.ones(dilation_size)
        violence_binary = binary_dilation(violence_binary, structure=kernel).astype(int)
    
    return violence_binary, fps, width, height, total_frames

def create_violence_segment_video(video_path, violence_binary, fps, width, height, output_path):
    """
    Tạo video chỉ chứa các segment bạo lực.
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    violence_segment_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(violence_binary) and violence_binary[frame_idx]:
            out.write(frame)
            violence_segment_count += 1
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return violence_segment_count > 0, violence_segment_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Không tìm thấy video'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)
    
    try:
        # Trích xuất frames toàn bộ video để dự đoán
        input_data = extract_frames(video_path)
        if input_data is None:
            return jsonify({'error': 'Lỗi khi xử lý video'}), 500
        
        # Dự đoán toàn bộ video
        prediction = model.predict(input_data)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        result = {
            'class': CLASSES[class_idx],
            'confidence': confidence,
            'prediction': prediction[0].tolist(),
            'status': 'success',
            'session_id': None,
            'can_create_segment': False
        }
        
        # Nếu phát hiện bạo lực, chuẩn bị dữ liệu để tạo segment
        if class_idx == 1:  # Violence detected
            violence_binary, fps, width, height, total_frames = detect_violence_segments(
                video_path, 
                violence_threshold=0.5,
                buffer_frames=15
            )
            
            if violence_binary is not None:
                # Tạo session_id để lưu dữ liệu
                session_id = f"session_{int(np.random.random() * 1000000)}"
                
                # Lưu dữ liệu vào cache
                detection_cache[session_id] = {
                    'video_path': video_path,
                    'violence_binary': violence_binary,
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames,
                    'violence_frames': int(np.sum(violence_binary))
                }
                
                result['session_id'] = session_id
                result['can_create_segment'] = True
                result['total_frames'] = int(total_frames)
                result['violence_frames'] = int(np.sum(violence_binary))
                result['violence_percentage'] = float((np.sum(violence_binary) / total_frames * 100) if total_frames > 0 else 0)
        else:
            # Không phát hiện bạo lực, xóa file
            os.remove(video_path)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create-segment', methods=['POST'])
def create_segment():
    """Tạo video segment bạo lực theo yêu cầu của người dùng."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in detection_cache:
            return jsonify({'error': 'Session không tồn tại hoặc hết hạn'}), 400
        
        # Lấy dữ liệu từ cache
        cached_data = detection_cache[session_id]
        video_path = cached_data['video_path']
        violence_binary = cached_data['violence_binary']
        fps = cached_data['fps']
        width = cached_data['width']
        height = cached_data['height']
        total_frames = cached_data['total_frames']
        
        # Tạo video segment
        output_filename = f"violence_segment_{int(np.random.random() * 10000)}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        has_violence, frame_count = create_violence_segment_video(
            video_path, 
            violence_binary, 
            fps, 
            width, 
            height, 
            output_path
        )
        
        result = {
            'status': 'success',
            'violence_segment_video': None,
            'violence_frames': int(frame_count),
            'total_frames': int(total_frames),
            'violence_percentage': float((frame_count / total_frames * 100) if total_frames > 0 else 0)
        }
        
        if has_violence and os.path.exists(output_path):
            result['violence_segment_video'] = f'/download/{output_filename}'
        
        # Xóa file input sau khi tạo video
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Xóa cache
        del detection_cache[session_id]
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Tải xuống file video segment bạo lực."""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='video/mp4', as_attachment=True)
        return jsonify({'error': 'File không tồn tại'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
