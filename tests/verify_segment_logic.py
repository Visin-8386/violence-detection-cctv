#!/usr/bin/env python
"""Direct test of segment creation with known violence binary."""

import os
import cv2
import numpy as np
from scipy.ndimage import binary_dilation

# Config
SEGMENT_PRE_SECONDS = 4.0
SEGMENT_POST_SECONDS = 0.0
FPS = 29.961928934010153

def get_violence_intervals(violence_binary):
    intervals = []
    start = None
    for i, val in enumerate(violence_binary):
        if val and start is None:
            start = i
        elif not val and start is not None:
            intervals.append((start, i - 1))
            start = None
    if start is not None:
        intervals.append((start, len(violence_binary) - 1))
    return intervals

# Create mock violence binary with 3 clusters matching 3.mp4 detection
violence_binary = np.zeros(787, dtype=int)
violence_binary[0:240] = 1     # Cluster 1: 0-239
violence_binary[675:720] = 1   # Cluster 2: 675-719  
violence_binary[780:787] = 1   # Cluster 3: 780-786

intervals = get_violence_intervals(violence_binary)
print("=" * 60)
print("SEGMENT CREATION TEST")
print("=" * 60)
print(f"\nVideo: {FPS:.2f} FPS, 787 frames")
print(f"SEGMENT_PRE_SECONDS: {SEGMENT_PRE_SECONDS}s")
print(f"Pre-frames: {int(round(FPS * SEGMENT_PRE_SECONDS))}")

print(f"\nRaw intervals detected: {intervals}")

pre_frames = int(round(FPS * SEGMENT_PRE_SECONDS))
post_frames = int(round(FPS * SEGMENT_POST_SECONDS))

print(f"\n--- SEGMENT CALCULATIONS ---")
for idx, (start, end) in enumerate(intervals, 1):
    expanded_start = max(0, start - pre_frames)
    expanded_end = min(786, end + post_frames)  # 787-1 = 786
    
    raw_frames = end - start + 1
    exp_frames = expanded_end - expanded_start + 1
    
    raw_dur = raw_frames / FPS
    exp_dur = exp_frames / FPS
    
    print(f"\nCluster {idx}:")
    print(f"  Raw: [{start:3d}, {end:3d}] = {raw_frames:3d} frames = {raw_dur:.2f}s")
    print(f"  Exp: [{expanded_start:3d}, {expanded_end:3d}] = {exp_frames:3d} frames = {exp_dur:.2f}s")

print("\n" + "=" * 60)
print("EXPECTED FOR CLUSTER 2:")
print("  Raw: [675, 719] = 45 frames = 1.50s")
print("  Exp: [555, 719] = 165 frames = 5.51s")
print("  ✓ With 4s pre-buffer (120 frames)")
print("=" * 60)
