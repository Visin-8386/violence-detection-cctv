import os
import cv2
import json
import numpy as np
import zipfile
import tempfile
import threading
from datetime import datetime
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow.keras.backend as K
import base64


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Cấu hình các tham số toàn cục
IMG_SIZE = 128
FRAME_COUNT = 15
CLASSES = ['Normal', 'Violence']
VIOLENCE_THRESHOLD = 0.3
SEGMENT_PRE_SECONDS = 2.0
SEGMENT_POST_SECONDS = 0.0

# Lưu trữ dữ liệu nhận diện tạm thời
detection_cache = {}

# ─── Event Logging System (Heatmap & Analytics) ───────────────────────
EVENTS_FILE = os.path.join('data', 'events.json')
os.makedirs('data', exist_ok=True)
_events_lock = threading.Lock()


def _load_events():
    """Đọc danh sách sự kiện từ file JSON."""
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        with open(EVENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_events(events):
    """Ghi danh sách sự kiện ra file JSON."""
    with open(EVENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(events, f, ensure_ascii=False, indent=2)


def log_violence_event(camera_id, confidence, violence_percentage=0,
                       location='Không xác định'):
    """Ghi nhận một sự kiện bạo lực vào log."""
    event = {
        'id': f"evt_{int(np.random.random() * 1e8)}",
        'camera_id': camera_id,
        'timestamp': datetime.now().isoformat(),
        'confidence': round(confidence, 4),
        'violence_percentage': round(violence_percentage, 2),
        'location': location,
    }
    with _events_lock:
        events = _load_events()
        events.append(event)
        # Giữ tối đa 5000 sự kiện gần nhất
        if len(events) > 5000:
            events = events[-5000:]
        _save_events(events)
    print(f"[EVENT] Logged violence from {camera_id}: {confidence:.1%}")
    return event


# ─── Face Detection cho Privacy Masking ───────────────────────────────
_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(_cascade_path)
print(f"[*] Loaded face cascade from {_cascade_path}")


def blur_faces_in_frame(frame, scale_factor=1.15, min_neighbors=5,
                        blur_strength=99):
    """Phát hiện và làm mờ tất cả khuôn mặt trong một frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 30)
        frame[y:y+h, x:x+w] = blurred
    return frame, len(faces)


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

def build_window_start_indices(total_frames, window_size, stride):
    """Build sliding-window start indices and always include the final window."""
    if total_frames < window_size:
        return []

    last_start = total_frames - window_size
    start_indices = list(range(0, last_start + 1, stride))
    if start_indices[-1] != last_start:
        start_indices.append(last_start)

    return start_indices

def detect_violence_segments(video_path, violence_threshold=VIOLENCE_THRESHOLD):
    """
    Phát hiện các segment bạo lực trong video bằng sliding window + batch predict.
    Trả về: tuple (violence_binary, fps, width, height, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0, 0, 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Đọc và tiền xử lý frames ngay (chỉ lưu bản 128x128 → tiết kiệm ~100x RAM)
    processed_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        frame_processed = preprocess_input(frame_resized.astype(np.float32))
        processed_frames.append(frame_processed)
    cap.release()
    
    actual_total = len(processed_frames)
    if actual_total < FRAME_COUNT:
        return None, fps, width, height, total_frames
    
    # Sliding window: stride = 50% overlap — đủ chính xác mà không quá chậm
    stride = max(FRAME_COUNT // 2, 1)
    window_starts = build_window_start_indices(actual_total, FRAME_COUNT, stride)
    violence_score_sums = np.zeros(actual_total, dtype=np.float32)
    violence_score_counts = np.zeros(actual_total, dtype=np.int32)

    # Batch predict: gom nhiều windows vào 1 lần predict thay vì từng cái
    BATCH_SIZE = 16
    for batch_start in range(0, len(window_starts), BATCH_SIZE):
        batch_indices = window_starts[batch_start:batch_start + BATCH_SIZE]
        batch_input = np.array(
            [processed_frames[s:s + FRAME_COUNT] for s in batch_indices],
            dtype=np.float32
        )
        predictions = model.predict(batch_input, verbose=0)
        
        for i, start_idx in enumerate(batch_indices):
            violence_prob = float(predictions[i][1])
            window_end = start_idx + FRAME_COUNT
            violence_score_sums[start_idx:window_end] += violence_prob
            violence_score_counts[start_idx:window_end] += 1
    
    # Giải phóng RAM
    del processed_frames
    
    # Lấy điểm trung bình trên mỗi frame từ các window chồng lấp.
    violence_scores = np.divide(
        violence_score_sums,
        violence_score_counts,
        out=np.zeros_like(violence_score_sums),
        where=violence_score_counts > 0
    )

    print(f"[DEBUG] detect_violence_segments: windows={len(window_starts)}, stride={stride}, batch_size={BATCH_SIZE}")
    print(f"[DEBUG] total_frames={actual_total}, fps={fps}")
    
    # Xác định frames bạo lực (> ngưỡng) — KHÔNG dilation để giữ đúng vùng phát hiện
    violence_binary = (violence_scores > violence_threshold).astype(int)

    # Log stats
    try:
        intervals = get_violence_intervals(violence_binary)
        print(f"[DEBUG] violence_frame_count={int(np.sum(violence_binary))}")
        print(f"[DEBUG] intervals={intervals}")
    except Exception:
        pass
    
    return violence_binary, fps, width, height, total_frames


def get_violence_intervals(violence_binary):
    """Trả về danh sách (start, end) cho các cụm frame bạo lực liên tiếp."""
    intervals = []
    in_segment = False
    segment_start = 0

    for idx, value in enumerate(violence_binary):
        if value and not in_segment:
            in_segment = True
            segment_start = idx
        elif not value and in_segment:
            intervals.append((segment_start, idx - 1))
            in_segment = False

    if in_segment:
        intervals.append((segment_start, len(violence_binary) - 1))

    return intervals


def expand_violence_windows(violence_binary, fps, pre_seconds=SEGMENT_PRE_SECONDS, post_seconds=SEGMENT_POST_SECONDS):
    """Mở rộng vùng bạo lực để lấy thêm thời gian trước/sau mỗi đoạn."""
    if violence_binary is None or len(violence_binary) == 0:
        return violence_binary

    pre_frames = int(round(fps * pre_seconds)) if fps else 0
    post_frames = int(round(fps * post_seconds)) if fps else 0
    expanded = np.zeros_like(violence_binary, dtype=int)

    in_segment = False
    segment_start = 0

    for idx, is_violence in enumerate(violence_binary):
        if is_violence and not in_segment:
            in_segment = True
            segment_start = idx
        elif not is_violence and in_segment:
            in_segment = False
            segment_end = idx - 1
            start = max(0, segment_start - pre_frames)
            end = min(len(violence_binary) - 1, segment_end + post_frames)
            expanded[start:end + 1] = 1

    if in_segment:
        segment_end = len(violence_binary) - 1
        start = max(0, segment_start - pre_frames)
        end = min(len(violence_binary) - 1, segment_end + post_frames)
        expanded[start:end + 1] = 1

    return expanded

def create_violence_segment_video(video_path, violence_binary, fps, width, height, output_path):
    """
    Tạo video chỉ chứa các segment bạo lực.
    """
    violence_binary = expand_violence_windows(violence_binary, fps)

    cap = cv2.VideoCapture(video_path)
    # Thử codec H.264 (tương thích trình duyệt), fallback sang mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        out.release()
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


def create_violence_segment_videos(video_path, violence_binary, fps, width, height, output_prefix):
    """Tạo riêng từng video cho mỗi cụm bạo lực."""
    if violence_binary is None:
        return []

    raw_intervals = get_violence_intervals(violence_binary)
    if not raw_intervals:
        return []

    # Expand chỉ từ raw binary (KHÔNG qua dilation) → đúng 2s trước
    expanded_binary = expand_violence_windows(violence_binary, fps)

    total_frames = len(violence_binary)
    pre_frames = int(round(fps * SEGMENT_PRE_SECONDS)) if fps else 0
    post_frames = int(round(fps * SEGMENT_POST_SECONDS)) if fps else 0
    intervals = get_violence_intervals(expanded_binary)
    output_files = []
    
    print(f"[DEBUG] create_violence_segment_videos:")
    print(f"  Total frames: {total_frames}, FPS: {fps}")
    print(f"  Pre-frames: {pre_frames} ({SEGMENT_PRE_SECONDS}s), Post-frames: {post_frames} ({SEGMENT_POST_SECONDS}s)")
    print(f"  Raw intervals: {raw_intervals}")
    print(f"  Expanded intervals: {intervals}")

    for index, (start_frame, end_frame) in enumerate(intervals, start=1):
        output_filename = f"{output_prefix}_{index}.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        print(f"  Segment {index}: [{start_frame}, {end_frame}] ({(end_frame-start_frame+1)/fps:.2f}s)")

        cap = cv2.VideoCapture(video_path)
        # Thử codec H.264 (tương thích trình duyệt), fallback sang mp4v
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            out.release()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Seek trực tiếp tới start_frame thay vì đọc từ frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_written = 0
        for _ in range(end_frame - start_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_written += 1

        cap.release()
        out.release()
        output_files.append(output_filename)
        
        print(f"    Output: {output_filename} ({frames_written} frames)")

    return output_files

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ─── Events API (Heatmap & Analytics) ─────────────────────────────────
@app.route('/api/events', methods=['GET'])
def get_events():
    """Trả về danh sách sự kiện bạo lực (filter theo camera, ngày)."""
    camera_id = request.args.get('camera_id')
    date_str = request.args.get('date')  # YYYY-MM-DD
    limit = int(request.args.get('limit', 200))

    events = _load_events()

    if camera_id:
        events = [e for e in events if e['camera_id'] == camera_id]
    if date_str:
        events = [e for e in events if e['timestamp'].startswith(date_str)]

    events = events[-limit:]  # Lấy N sự kiện gần nhất
    return jsonify({'status': 'success', 'events': events, 'total': len(events)})


@app.route('/api/events/stats', methods=['GET'])
def get_event_stats():
    """Trả về thống kê tổng hợp cho dashboard."""
    events = _load_events()
    if not events:
        return jsonify({
            'total_events': 0, 'cameras': {},
            'hourly': [0]*24, 'avg_confidence': 0,
            'recent': []
        })

    # Thống kê theo camera
    cameras = {}
    hourly = [0] * 24
    confidences = []
    for e in events:
        cam = e.get('camera_id', 'unknown')
        cameras[cam] = cameras.get(cam, 0) + 1
        try:
            hour = int(e['timestamp'][11:13])
            hourly[hour] += 1
        except (ValueError, IndexError):
            pass
        confidences.append(e.get('confidence', 0))

    return jsonify({
        'total_events': len(events),
        'cameras': cameras,
        'hourly': hourly,
        'avg_confidence': round(sum(confidences) / len(confidences), 4) if confidences else 0,
        'recent': events[-20:][::-1]  # 20 sự kiện gần nhất, mới nhất trước
    })


@app.route('/api/events/clear', methods=['POST'])
def clear_events():
    """Xóa toàn bộ lịch sử sự kiện."""
    with _events_lock:
        _save_events([])
    return jsonify({'status': 'success', 'message': 'Đã xóa lịch sử'})

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
        violence_prob = float(prediction[0][1])
        class_idx = 1 if violence_prob >= VIOLENCE_THRESHOLD else 0
        confidence = violence_prob if class_idx == 1 else float(prediction[0][0])
        
        result = {
            'class': CLASSES[class_idx],
            'confidence': confidence,
            'prediction': prediction[0].tolist(),
            'violence_probability': violence_prob,
            'threshold': VIOLENCE_THRESHOLD,
            'status': 'success',
            'session_id': None,
            'can_create_segment': False
        }
        
        # Nếu phát hiện bạo lực, chuẩn bị dữ liệu để tạo segment
        if class_idx == 1:  # Violence detected
            violence_binary, fps, width, height, total_frames = detect_violence_segments(
                video_path, 
                violence_threshold=VIOLENCE_THRESHOLD
            )
            
            if violence_binary is not None:
                # Tạo session_id để lưu dữ liệu
                session_id = f"session_{int(np.random.random() * 1000000)}"
                
                # Lưu dữ liệu vào cache (chỉ raw binary, expand khi cắt video)
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
                
                # Tự động ghi event vào hệ thống analytics
                log_violence_event(
                    camera_id=request.form.get('camera_id', 'upload'),
                    confidence=violence_prob,
                    violence_percentage=result['violence_percentage'],
                    location=request.form.get('location', 'Upload')
                )
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
        privacy_mode = data.get('privacy_mode', False)
        
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
        
        # Tạo từng video riêng cho mỗi cụm bạo lực
        output_prefix = f"violence_segment_{int(np.random.random() * 10000)}"
        output_files = create_violence_segment_videos(
            video_path,
            violence_binary,
            fps,
            width,
            height,
            output_prefix,
        )

        total_written_frames = 0
        for output_filename in output_files:
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            if os.path.exists(output_path):
                cap_check = cv2.VideoCapture(output_path)
                total_written_frames += int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_check.release()
        
        # Privacy Masking: làm mờ khuôn mặt nếu bật
        total_faces_blurred = 0
        if privacy_mode and output_files:
            print(f"[PRIVACY] Applying face blur to {len(output_files)} segment(s)...")
            for output_filename in output_files:
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                if not os.path.exists(output_path):
                    continue
                # Đọc video, blur mặt, ghi lại
                cap_p = cv2.VideoCapture(output_path)
                temp_path = output_path + '.tmp.mp4'
                fourcc_p = cv2.VideoWriter_fourcc(*'avc1')
                out_p = cv2.VideoWriter(temp_path, fourcc_p, fps, (width, height))
                if not out_p.isOpened():
                    out_p.release()
                    fourcc_p = cv2.VideoWriter_fourcc(*'mp4v')
                    out_p = cv2.VideoWriter(temp_path, fourcc_p, fps, (width, height))
                
                while True:
                    ret_p, frame_p = cap_p.read()
                    if not ret_p:
                        break
                    frame_p, n_faces = blur_faces_in_frame(frame_p)
                    total_faces_blurred += n_faces
                    out_p.write(frame_p)
                
                cap_p.release()
                out_p.release()
                # Thay file gốc bằng file đã blur
                os.replace(temp_path, output_path)
            print(f"[PRIVACY] Done. Blurred {total_faces_blurred} face instances.")
        
        result = {
            'status': 'success',
            'violence_segment_video': f'/download/{output_files[0]}' if output_files else None,
            'violence_segment_videos': [f'/download/{filename}' for filename in output_files],
            'violence_frames': int(total_written_frames),
            'total_frames': int(total_frames),
            'violence_percentage': float((total_written_frames / total_frames * 100) if total_frames > 0 else 0),
            'privacy_mode': privacy_mode,
            'faces_blurred': total_faces_blurred
        }
        
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

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
    """Nhận diện bạo lực từ stream frames gửi lên từ client."""
    try:
        data = request.get_json()
        if not data or 'frames' not in data:
            return jsonify({'error': 'No frames provided'}), 400
        
        frames_base64 = data['frames']
        if len(frames_base64) < FRAME_COUNT:
            return jsonify({'error': f'Need at least {FRAME_COUNT} frames'}), 400
        
        processed_frames = []
        for b64_str in frames_base64:
            # Giải mã base64 thành ảnh
            try:
                header, encoded = b64_str.split(",", 1)
                data_bytes = base64.b64decode(encoded)
                nparr = np.frombuffer(data_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue

                # Tiền xử lý
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                processed_frames.append(preprocess_input(img_resized.astype(np.float32)))
            except Exception as e:
                print(f"[-] Error processing frame: {e}")
                continue
        
        if len(processed_frames) < FRAME_COUNT:
             # Padding nếu thiếu do lỗi decode
             while len(processed_frames) < FRAME_COUNT:
                processed_frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

        # Lấy 15 frames cuối cùng nếu nhiều hơn
        processed_frames = processed_frames[-FRAME_COUNT:]
        input_data = np.array([processed_frames], dtype=np.float32)
        
        # Dự đoán
        prediction = model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Log event nếu phát hiện bạo lực (cooldown 10s tránh spam)
        if class_idx == 1:
            camera_id = data.get('camera_id', 'webcam')
            now = datetime.now()
            last_key = f"_rt_last_{camera_id}"
            last_time = getattr(predict_realtime, last_key, None)
            if last_time is None or (now - last_time).total_seconds() >= 10:
                setattr(predict_realtime, last_key, now)
                log_violence_event(
                    camera_id=camera_id,
                    confidence=confidence,
                    violence_percentage=100.0,
                    location='Realtime Camera'
                )
        
        return jsonify({
            'class': CLASSES[class_idx],
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"[-] Error in predict_realtime: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
