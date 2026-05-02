"""
Benchmark test cho Violence Detection CCTV.
Đo thời gian từng bước: load model, predict nhanh, sliding window, cắt video.
"""
import os
import time
import cv2
import numpy as np

# ── Config ──
VIDEO_PATH = "3.mp4"

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h

def main():
    fps_v, total_frames, w, h = get_video_info(VIDEO_PATH)
    duration = total_frames / fps_v if fps_v else 0
    print("=" * 60)
    print(f"VIDEO: {VIDEO_PATH}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {fps_v:.1f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s")
    print("=" * 60)

    # ── Step 1: Load model ──
    print("\n[1/4] Loading model...")
    t0 = time.perf_counter()

    import tensorflow as tf
    # Suppress TF warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Import từ app.py
    from app import (
        model, extract_frames, detect_violence_segments,
        create_violence_segment_videos, VIOLENCE_THRESHOLD,
        FRAME_COUNT, IMG_SIZE, SEGMENT_PRE_SECONDS
    )
    t_load = time.perf_counter() - t0
    print(f"  → Model loaded: {t_load:.2f}s")

    # ── Step 2: Quick predict (toàn bộ video) ──
    print("\n[2/4] Quick predict (1 lần inference)...")
    t0 = time.perf_counter()
    input_data = extract_frames(VIDEO_PATH)
    t_extract = time.perf_counter() - t0

    t0 = time.perf_counter()
    prediction = model.predict(input_data, verbose=0)
    t_predict = time.perf_counter() - t0

    violence_prob = float(prediction[0][1])
    class_name = "Violence" if violence_prob >= VIOLENCE_THRESHOLD else "Normal"
    print(f"  → Extract frames: {t_extract:.2f}s")
    print(f"  → Predict: {t_predict:.2f}s")
    print(f"  → Result: {class_name} ({violence_prob:.2%})")

    # ── Step 3: Sliding window detection ──
    print("\n[3/4] Sliding window detection (batch predict)...")
    t0 = time.perf_counter()
    violence_binary, fps, width, height, tf_count = detect_violence_segments(
        VIDEO_PATH, violence_threshold=VIOLENCE_THRESHOLD
    )
    t_detect = time.perf_counter() - t0

    if violence_binary is not None:
        violence_count = int(np.sum(violence_binary))
        from app import get_violence_intervals
        intervals = get_violence_intervals(violence_binary)
        print(f"  → Detection time: {t_detect:.2f}s")
        print(f"  → Violence frames: {violence_count}/{len(violence_binary)}")
        print(f"  → Violence segments: {len(intervals)}")
        for i, (s, e) in enumerate(intervals, 1):
            print(f"     Segment {i}: frame {s}-{e} ({(e-s+1)/fps:.2f}s)")
    else:
        print(f"  → Detection time: {t_detect:.2f}s")
        print(f"  → No violence detected (binary is None)")

    # ── Step 4: Create segment videos ──
    t_segment = 0
    segment_files = []
    if violence_binary is not None and int(np.sum(violence_binary)) > 0:
        print("\n[4/4] Creating segment videos...")
        output_prefix = f"benchmark_test_{int(time.time())}"
        t0 = time.perf_counter()
        segment_files = create_violence_segment_videos(
            VIDEO_PATH, violence_binary, fps, width, height, output_prefix
        )
        t_segment = time.perf_counter() - t0
        print(f"  → Segment creation: {t_segment:.2f}s")
        print(f"  → Files created: {len(segment_files)}")
        for f in segment_files:
            fpath = os.path.join("outputs", f)
            if os.path.exists(fpath):
                cap = cv2.VideoCapture(fpath)
                seg_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                seg_dur = seg_frames / fps if fps else 0
                cap.release()
                print(f"     {f}: {seg_frames} frames ({seg_dur:.2f}s)")
    else:
        print("\n[4/4] Skipped (no violence to cut)")

    # ── Summary ──
    t_total = t_extract + t_predict + t_detect + t_segment
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Video duration:       {duration:.1f}s ({total_frames} frames)")
    print(f"  ──────────────────────────────────")
    print(f"  Extract frames:       {t_extract:.2f}s")
    print(f"  Quick predict:        {t_predict:.2f}s")
    print(f"  Sliding window:       {t_detect:.2f}s")
    print(f"  Create segments:      {t_segment:.2f}s")
    print(f"  ──────────────────────────────────")
    print(f"  TOTAL processing:     {t_total:.2f}s")
    print(f"  Speed ratio:          {duration/t_total:.1f}x realtime" if t_total > 0 else "")
    print(f"  Pre-buffer setting:   {SEGMENT_PRE_SECONDS}s")
    print("=" * 60)

    # ── Extrapolation ──
    print("\n📊 ƯỚC TÍNH THEO ĐỘ DÀI VIDEO:")
    for target_dur in [10, 30, 60, 120, 300]:
        ratio = target_dur / duration if duration > 0 else 1
        # Sliding window scales roughly linearly with frame count
        est_detect = t_detect * ratio
        est_extract = t_extract * ratio
        est_total = est_extract + t_predict + est_detect + t_segment * ratio
        print(f"  Video {target_dur:>3}s ({target_dur//60}m{target_dur%60:02d}s): ~{est_total:.1f}s processing")

    # Cleanup benchmark files
    for f in segment_files:
        fpath = os.path.join("outputs", f)
        if os.path.exists(fpath):
            os.remove(fpath)
    print("\n✅ Benchmark files cleaned up.")

if __name__ == "__main__":
    main()
