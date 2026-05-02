import os
import zipfile
import tempfile
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tf2onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

FRAME_COUNT = 15
IMG_SIZE = 128

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
    def call(self, x):
        batch_size = tf.shape(x)[0]
        return tf.reshape(x, (batch_size * FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3))

    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class ReshapeToSequence(Layer):
    def call(self, x):
        features = x.shape[-1]
        batch_size = tf.shape(x)[0] // FRAME_COUNT
        return tf.reshape(x, (batch_size, FRAME_COUNT, features))

    def get_config(self):
        return super().get_config()

def build_model_architecture():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    video_input = Input(shape=(FRAME_COUNT, IMG_SIZE, IMG_SIZE, 3), name="input")

    x = ReshapeToFrames()(video_input)          
    x = base_model(x)                           
    x = GlobalAveragePooling2D()(x)             
    x = ReshapeToSequence()(x)                  

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    context_vector, _ = TemporalAttention(name='attention_layer')(x)

    x = Dense(64, activation='relu')(context_vector)
    x = Dropout(0.5)(x)
    output = Dense(2, activation='softmax')(x)

    return Model(inputs=video_input, outputs=output)

def optimize_keras_to_onnx_int8(keras_model_path, output_dir):
    """
    Chuyển đổi Keras model sang ONNX và Quantize (INT8) để tăng tốc độ Inference.
    """
    print(f"[*] Đang tải mô hình Keras từ: {keras_model_path}")
    model = build_model_architecture()
    with zipfile.ZipFile(keras_model_path, 'r') as zf:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zf.extract('model.weights.h5', tmp_dir)
            weights_path = os.path.join(tmp_dir, 'model.weights.h5')
            model.load_weights(weights_path)
    
    # 1. Chuyển sang ONNX
    onnx_path = os.path.join(output_dir, "model_fp32.onnx")
    print("[*] Đang chuyển đổi sang ONNX (FP32)...")
    spec = (tf.TensorSpec((None, 15, 128, 128, 3), tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_path)
    print(f"[+] Đã lưu ONNX FP32 tại: {onnx_path}")
    
    # 2. Dynamic Quantization (FP32 -> INT8)
    onnx_quant_path = os.path.join(output_dir, "CCTV_Violence_Finetuned_int8.onnx")
    print("[*] Đang thực hiện Dynamic Quantization (INT8)...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=onnx_quant_path,
        weight_type=QuantType.QUInt8
    )
    print(f"[+] Đã lưu ONNX Quantized INT8 tại: {onnx_quant_path}")
    
    # Xóa file FP32 tạm thời
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
        
    print("[+] Tối ưu hóa hoàn tất! Kích thước mô hình đã giảm đáng kể.")

if __name__ == "__main__":
    keras_path = os.path.join("models", "CCTV_Violence_Finetuned.keras")
    out_dir = "models"
    if os.path.exists(keras_path):
        optimize_keras_to_onnx_int8(keras_path, out_dir)
    else:
        print(f"[-] Không tìm thấy file Keras tại {keras_path}")
