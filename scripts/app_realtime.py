import os
import cv2
import numpy as np
import zipfile
import tempfile
import tensorflow as tf
import base64
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

app = Flask(__name__)

# Cấu hình các tham số toàn cục
IMG_SIZE = 128
FRAME_COUNT = 15
CLASSES = ['Normal', 'Violence']


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


# Nạp mô hình
MODEL_PATH = 'models/CCTV_Violence_Finetuned.keras'
print(f"[*] Đang nạp mô hình thời gian thực từ {MODEL_PATH}...")
model = build_model_architecture()

with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zf.extract('model.weights.h5', tmp_dir)
        weights_path = os.path.join(tmp_dir, 'model.weights.h5')
        model.load_weights(weights_path)

print("[+] Hệ thống giám sát thời gian thực đã sẵn sàng!")

@app.route('/')
def index():
    return render_template('realtime.html')

@app.route('/predict_realtime', methods=['POST'])
def predict_realtime():
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
            header, encoded = b64_str.split(",", 1)
            data_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(data_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Tiền xử lý
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            processed_frames.append(preprocess_input(img_resized.astype(np.float32)))
        
        # Lấy 15 frames cuối cùng nếu nhiều hơn
        processed_frames = processed_frames[-FRAME_COUNT:]
        input_data = np.array([processed_frames], dtype=np.float32)
        
        # Dự đoán
        prediction = model.predict(input_data, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        return jsonify({
            'class': CLASSES[class_idx],
            'confidence': confidence,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"[-] Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Tắt reloader nếu dùng GPU để tránh nạp model 2 lần
    app.run(debug=True, port=5001, use_reloader=False)
