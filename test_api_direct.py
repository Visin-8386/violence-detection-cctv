#!/usr/bin/env python
"""Test API directly to verify buffer is being applied."""

import requests
import json
import os
import cv2

API_URL = 'http://localhost:5000'
VIDEO_FILE = 'd:\\violence-detection-cctv\\3.mp4'

# Step 1: Get video info
cap = cv2.VideoCapture(VIDEO_FILE)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("=" * 80)
print("TESTING API - MULTI-VIDEO SEGMENT FIX")
print("=" * 80)
print(f"\nVideo info: {fps:.2f} FPS, {total_frames} frames")

# Step 2: Upload and predict
print("\n[1] Uploading video to /predict...")
with open(VIDEO_FILE, 'rb') as f:
    response = requests.post(f'{API_URL}/predict', files={'video': f})

predict_data = response.json()
session_id = predict_data.get('session_id')

print(f"Status: {predict_data.get('class')}")
print(f"Confidence: {predict_data.get('confidence'):.4f}")
print(f"Session ID: {session_id}")
print(f"Can create segment: {predict_data.get('can_create_segment')}")

# Step 3: Create segments
if session_id:
    print(f"\n[2] Creating segments via /create-segment...")
    response = requests.post(f'{API_URL}/create-segment', 
                            json={'session_id': session_id})
    
    segment_data = response.json()
    print(f"Status: {segment_data.get('status')}")
    
    videos = segment_data.get('violence_segment_videos', [])
    print(f"Videos returned: {len(videos)}")
    
    for i, video_url in enumerate(videos, 1):
        print(f"\n  Segment {i}: {video_url}")
    
    # Step 4: Check created files
    print(f"\n[3] Checking created files in outputs/...")
    outputs_folder = 'd:\\violence-detection-cctv\\outputs'
    
    if os.path.exists(outputs_folder):
        files = os.listdir(outputs_folder)
        print(f"Found {len(files)} files:")
        
        for filename in sorted(files):
            filepath = os.path.join(outputs_folder, filename)
            cap = cv2.VideoCapture(filepath)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            duration = frames / fps if fps else 0
            print(f"  {filename}: {frames} frames ({duration:.2f}s)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print(f"Expected clusters:")
    print(f"  Cluster 1: ~8.0s (0-239 frames) → no pre-buffer (starts at 0)")
    print(f"  Cluster 2: ~5.5s (555-719 frames) → with 4s pre-buffer")
    print(f"  Cluster 3: ~4.2s (660-786 frames) → with 4s pre-buffer")
    
else:
    print("ERROR: No session_id returned!")

print("\n" + "=" * 80)
