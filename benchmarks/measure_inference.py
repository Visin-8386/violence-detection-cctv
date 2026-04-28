"""Measure inference latency for the trained model and write results to reports/inference_benchmark.txt

Usage:
    python benchmarks/measure_inference.py --model models/CCTV_Violence_Finetuned.keras --runs 50 --warmup 5
"""
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

OUTPUT_DIR = 'reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/CCTV_Violence_Finetuned.keras')
    parser.add_argument('--runs', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--out', type=str, default=os.path.join(OUTPUT_DIR, 'inference_benchmark.txt'))
    return parser.parse_args()


def load_model_from_keras_archive(path):
    # Recreate the same architecture as in app.py (to avoid import issues)
    from tensorflow.keras.layers import Layer, Input, GlobalAveragePooling2D, Bidirectional, LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import MobileNetV2
    import tensorflow.keras.backend as K

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

    class ReshapeToFrames(Layer):
        def call(self, x):
            batch_size = tf.shape(x)[0]
            return tf.reshape(x, (batch_size * 15, 128, 128, 3))

        def get_config(self):
            return super().get_config()

    class ReshapeToSequence(Layer):
        def call(self, x):
            features = x.shape[-1]
            batch_size = tf.shape(x)[0] // 15
            return tf.reshape(x, (batch_size, 15, features))

        def get_config(self):
            return super().get_config()

    def build_model_architecture_local():
        base_model = MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))

        video_input = Input(shape=(15, 128, 128, 3))
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

    # Build model and load weights from .keras archive
    try:
        model = build_model_architecture_local()
        import zipfile, tempfile
        with zipfile.ZipFile(path, 'r') as zf:
            with tempfile.TemporaryDirectory() as tmp_dir:
                zf.extract('model.weights.h5', tmp_dir)
                weights_path = os.path.join(tmp_dir, 'model.weights.h5')
                model.load_weights(weights_path)
        return model
    except Exception as e:
        raise RuntimeError(f'Unable to build/load model from {path}: {e}')


def main():
    args = parse_args()

    print(f'Loading model from {args.model}...')
    model = load_model_from_keras_archive(args.model)
    print('Model loaded.')

    # Input shape: (batch, 15, 128, 128, 3) per app.py
    batch = args.batch_size
    input_shape = (batch, 15, 128, 128, 3)

    # Create dummy input and preprocess
    rng = np.random.RandomState(42)
    dummy = rng.randint(0, 255, size=input_shape).astype(np.float32)
    # Apply MobileNetV2 preprocess for each frame
    dummy = preprocess_input(dummy)

    # Warmup
    print(f'Warming up ({args.warmup} runs)...')
    for _ in range(args.warmup):
        _ = model.predict(dummy, verbose=0)

    # Timed runs
    times = []
    print(f'Running benchmark ({args.runs} runs)...')
    for i in range(args.runs):
        t0 = time.perf_counter()
        _ = model.predict(dummy, verbose=0)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if (i + 1) % max(1, args.runs // 10) == 0:
            print(f'  run {i+1}/{args.runs} - last: {times[-1]:.4f}s')

    times = np.array(times)
    stats = {
        'runs': int(args.runs),
        'warmup': int(args.warmup),
        'batch_size': int(args.batch_size),
        'mean_s': float(times.mean()),
        'median_s': float(np.median(times)),
        'min_s': float(times.min()),
        'max_s': float(times.max()),
        'std_s': float(times.std()),
        'p95_s': float(np.percentile(times, 95)),
        'p99_s': float(np.percentile(times, 99)),
    }

    # Environment info
    tf_version = tf.__version__
    devices = tf.config.list_physical_devices()
    device_str = ', '.join([d.device_type for d in devices]) or 'None'

    # Write report
    out_path = args.out
    now = datetime.utcnow().isoformat() + 'Z'
    with open(out_path, 'a', encoding='utf-8') as f:
        f.write('==== Inference Benchmark ====' + '\n')
        f.write(f'timestamp: {now}\n')
        f.write(f'model: {args.model}\n')
        f.write(f'tf_version: {tf_version}\n')
        f.write(f'devices: {device_str}\n')
        for k, v in stats.items():
            f.write(f'{k}: {v}\n')
        f.write('\n')

    print('Benchmark complete. Results written to', out_path)


if __name__ == '__main__':
    main()
