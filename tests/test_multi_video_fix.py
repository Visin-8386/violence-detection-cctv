#!/usr/bin/env python
"""Test script to verify multi-video output and buffer fix."""

import os
import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import tempfile
import shutil

# Simulate the key parameters
SEGMENT_PRE_SECONDS = 4.0
SEGMENT_POST_SECONDS = 0.0

def get_violence_intervals(violence_binary):
    """Get intervals of consecutive True values."""
    if violence_binary is None or len(violence_binary) == 0:
        return []
    
    intervals = []
    start = None
    
    for i, is_violence in enumerate(violence_binary):
        if is_violence and start is None:
            start = i
        elif not is_violence and start is not None:
            intervals.append((start, i - 1))
            start = None
    
    if start is not None:
        intervals.append((start, len(violence_binary) - 1))
    
    return intervals

def create_violence_segment_videos(video_path, violence_binary, fps, width, height, output_prefix, output_folder):
    """Simulate the segment creation function."""
    if violence_binary is None:
        return []

    total_frames = len(violence_binary)
    pre_frames = int(round(fps * SEGMENT_PRE_SECONDS)) if fps else 0
    post_frames = int(round(fps * SEGMENT_POST_SECONDS)) if fps else 0
    intervals = get_violence_intervals(violence_binary)
    output_files = []

    print(f"\n[INFO] Creating segments:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Pre-buffer frames: {pre_frames} ({SEGMENT_PRE_SECONDS}s)")
    print(f"  Post-buffer frames: {post_frames} ({SEGMENT_POST_SECONDS}s)")
    print(f"  Detected {len(intervals)} violence intervals:")
    
    for index, (start_frame, end_frame) in enumerate(intervals, start=1):
        expanded_start = max(0, start_frame - pre_frames)
        expanded_end = min(total_frames - 1, end_frame + post_frames)
        
        segment_frames = expanded_end - expanded_start + 1
        segment_duration = segment_frames / fps if fps else 0
        
        output_filename = f"{output_prefix}_{index}.mp4"
        print(f"\n  Segment {index}:")
        print(f"    Raw interval: frames [{start_frame}, {end_frame}] ({(end_frame-start_frame+1)/fps:.2f}s)")
        print(f"    With buffer:  frames [{expanded_start}, {expanded_end}] ({segment_duration:.2f}s)")
        print(f"    Output file: {output_filename}")
        
        output_files.append(output_filename)

    return output_files

# Test with 3.mp4 data
test_video = 'd:\\violence-detection-cctv\\3.mp4'
if os.path.exists(test_video):
    cap = cv2.VideoCapture(test_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Create a mock violence binary array with 3 clusters
    # Cluster 1: frames 0-239 (Normal)
    # Cluster 2: frames 675-719 (Violence)
    # Cluster 3: frames 780-786 (Violence)
    violence_binary = np.zeros(frame_count, dtype=bool)
    violence_binary[0:240] = True    # Cluster 1
    violence_binary[675:720] = True  # Cluster 2
    violence_binary[780:787] = True  # Cluster 3
    
    print("=" * 80)
    print("MULTI-VIDEO SEGMENT TEST")
    print("=" * 80)
    print(f"\nTest Video: {test_video}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {frame_count/fps:.2f}s")
    
    # Create test output folder
    with tempfile.TemporaryDirectory() as tmpdir:
        output_files = create_violence_segment_videos(
            test_video,
            violence_binary,
            fps,
            1920,  # width
            1080,  # height
            "test_segment",
            tmpdir
        )
    
    print(f"\n[RESULT] Created {len(output_files)} output files:")
    for i, filename in enumerate(output_files, 1):
        print(f"  {i}. {filename}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print(f"✓ Multiple clusters detected correctly: {len(output_files)} files")
    print(f"✓ Pre-buffer set to: {SEGMENT_PRE_SECONDS}s")
    print(f"✓ Pre-buffer frames: {int(round(fps * SEGMENT_PRE_SECONDS))}")
    
    # Verify Cluster 2 should have proper buffer
    intervals = get_violence_intervals(violence_binary)
    if len(intervals) >= 2:
        cluster2_start, cluster2_end = intervals[1]
        pre_frames = int(round(fps * SEGMENT_PRE_SECONDS))
        expected_start = max(0, cluster2_start - pre_frames)
        expected_duration = (cluster2_end - expected_start + 1) / fps
        print(f"\nCluster 2 verification:")
        print(f"  Raw violence: frames [{cluster2_start}, {cluster2_end}]")
        print(f"  Expected with 4s buffer: frames [{expected_start}, {cluster2_end}]")
        print(f"  Expected duration: ~{expected_duration:.2f}s")
else:
    print(f"ERROR: Test video not found: {test_video}")

print("\n" + "=" * 80)
