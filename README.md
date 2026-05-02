# 🎥 Violence Detection CCTV
## Real-Time Violence Behavior Detection System for Surveillance Cameras

> **🚀 Advanced Deep Learning AI for Real-Time Violence Detection**  
> Architecture: `MobileNetV2` + `BiLSTM` + `Temporal Attention`  
> Accuracy: **~90.77%** | ROC-AUC: **~0.95** | Recall: **~96%**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Latest-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

</div>

---

## 📌 Overview

This is an **advanced AI system** that integrates modern deep learning technologies to automatically and accurately detect violent behavior from CCTV surveillance videos.

The system is designed for:
- ✅ **Security Surveillance** - Automatic detection of violent incidents.
- ✅ **Behavioral Analysis** - Providing detailed data and event logs.
- ✅ **Decision Support** - Assisting authorities in rapid response.
- ✅ **Data Science** - Research and development of AI technologies.

### 🎯 Key Features

| Feature | Description |
|-----------|-------|
| **⚡ Rapid Detection** | Analyzes video sequences in <200ms |
| **🎬 Intelligent Segmentation** | Automatically extracts clips containing violence |
| **📊 Dashboard & Heatmap** | Tracks violence trends over time/camera |
| **🎥 Real-time Monitoring** | Live monitoring via Webcam/CCTV |
| **📥 Auto Incident Export** | Automatically records and saves MP4 clips during incidents |
| **🛡️ Privacy Masking** | Automatic face blurring (GDPR compliance) |
| **🐳 Docker Support** | Rapid, consistent environment deployment |
| **🔍 Explainable AI** | Visualizes model attention weights (XAI) |

---

## 🏗️ Architecture & Technology

### 🧠 Neural Network Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Video Input                            │
│              (Batch, 15 frames, 128×128, RGB)              │
└──────────────────────┬─────────────────────────────────────┘
                       │
        ┌──────────────▼────────────────┐
        │   MobileNetV2 + TimeDistributed   │
        │  (Transfer Learning - ImageNet)   │
        │    → Extract spatial features     │
        │     (B×15, 1280-dim vectors)      │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼────────────────┐
        │   Bidirectional LSTM (64 units)  │
        │   → Model temporal dynamics      │
        │   → (B, 15, 128-dim context)    │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼────────────────┐
        │  Temporal Attention Layer     │
        │  → Focus on key frames        │
        │  → Weight normalization       │
        │  → (B, 128-dim context)       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼────────────────┐
        │   Dense(64, ReLU) + Dropout   │
        │   Dense(2, Softmax)           │
        └──────────────┬───────────────┘
                       │
        ┌──────────────▼────────────────┐
        │   Output: [Normal, Violence]  │
        │   Confidence: 0-100%          │
        └──────────────────────────────┘
```

### 📊 Model Statistics

| Metric | Value |
|--------|---------|
| **Model Size** | 3.1 MB (ONNX INT8) - *Previously 30MB* |
| **Inference Latency** | ~199ms / 15 frames (CPU) |
| **Memory Footprint** | ~150MB |
| **Pipeline Optimization** | YOLOv8 (Person Tracking) + ONNX Runtime |
| **Hardware Acceleration** | Optimized for CPU (No GPU required) |

---

## ⚡ Technical Optimizations (Engineering Excellence)

The system utilizes **Inference Pipeline Engineering** to balance Accuracy, Latency, and Scalability for real-world deployment.

### 1. Model Compression & Acceleration (Quantization)
*   **Technique:** Converted Keras model to **ONNX Runtime** with **Dynamic INT8 Quantization**.
*   **Goal:** Resolving computational bottlenecks on non-GPU systems.
*   **Impact:** 
    *   Inference speed increased by **~350%** (700ms down to <200ms on CPU).
    *   Storage optimized to **3.1MB** (10x reduction from original 30MB).

### 2. Spatial-Temporal POI Tracking (Context Filtering)
*   **Technique:** Integrated **YOLOv8 Nano** as a Pre-filter for **Person-of-Interest (POI)** identification.
*   **Mechanism:** The system uses YOLOv8 to locate humans, then applies **Dynamic Cropping** to feed only relevant regions into the violence model.
*   **Impact:** 
    *   Eliminated **False Positives** caused by environmental factors (trees, shadows, weather).
    *   Improved real-world accuracy by removing irrelevant background noise.

### 3. Buffer-based Event Reconstruction
*   **Technique:** Implemented a **Sliding Frame Buffer** (3s Pre-event & 2s Post-event).
*   **Impact:** Automatically exports incident videos in MP4 format, including **3 seconds of context before the violence occurred**, providing full data for surveillance and investigation.

### 🏆 Performance Comparison (CPU-only Benchmarks)

| Metric | Legacy Pipeline (Keras FP32) | Optimized Pipeline (ONNX INT8 + YOLO) |
|--------|------------------------------|----------------------------------------|
| **Model Size** | ~30.2 MB | **3.1 MB** *(10x smaller)* |
| **Inference Latency** | ~600 - 800 ms | **~199 ms** *(3.5x faster)* |
| **False Positive Rate** | High (environmental noise) | **Near Zero** (via POI Tracking) |
| **Deployment** | Heavy, many dependencies | **Lightweight**, Dockerized |

---

## 📊 Data & Training Results

### 🎬 Datasets Used

The model was trained on **3 real-world datasets** totaling **~6,000 videos**:

| Dataset | Source | Videos | Purpose |
|---------|-------|--------|---------|
| **RWF-2000** | Kaggle Real World (Fight) | 2,000 | Fighting, brawling |
| **Real Life Violence** | Kaggle Community | 2,000 | Street violence |
| **SCVD** | Smart City Violence Dataset | ~2,000 | Public violence |

**Data Processing:**
- 📊 **Split:** 80% Train / 10% Validation / 10% Test
- 🔀 **Scene Leakage Prevention:** GroupShuffleSplit by original scene
- 🎲 **Data Augmentation:** Flip, Crop, Brightness jitter, Temporal dropout

### 📈 Achieved Results

```
┌──────────────────────────────────────────┐
│         EVALUATION METRICS               │
├──────────────────────────────────────────┤
│ Accuracy:         90.77%  ⭐⭐⭐⭐⭐    │
│ ROC-AUC Score:    0.95    ⭐⭐⭐⭐⭐    │
│ Violence Recall:  96%     ⭐⭐⭐⭐⭐    │
│ Precision:        0.96    ⭐⭐⭐⭐⭐    │
│ F1-Score:         0.95    ⭐⭐⭐⭐⭐    │
└──────────────────────────────────────────┘
```

### 🎯 Detailed Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.94 | 0.92 | 0.93 | 250 |
| Violence | 0.96 | 0.96 | 0.96 | 250 |
| **Macro Avg** | **0.95** | **0.94** | **0.95** | 500 |

---

## 🚀 Getting Started

### 📌 Option 1: Run Web Application (RECOMMENDED)

**✨ Modern, easy-to-use web interface**

#### System Requirements
- Python 3.10+
- 500MB RAM minimum
- Flask, TensorFlow, OpenCV installed

#### Step 1: Install Dependencies
```bash
cd violence-detection-cctv
pip install flask tensorflow opencv-python numpy scipy onnxruntime ultralytics
```

#### Step 2: Launch Server
```bash
python app.py
```

Output:
```
[*] Loading YOLOv8 for person detection...
[*] Loading optimized ONNX model from models/CCTV_Violence_Finetuned_int8.onnx...
[+] Loaded ONNX model successfully!
 * Running on http://127.0.0.1:5000
```

#### Step 3: Open Browser
Navigate to: **http://127.0.0.1:5000**

---

### 🐳 Option 2: Run with Docker (Recommended for Enterprise)

Running with Docker avoids installation issues related to library versions or OpenCV across different operating systems.

#### Step 1: Build and Run with Docker Compose
```bash
docker-compose up --build -d
```

#### Step 2: Open Browser
Navigate to: **http://127.0.0.1:5000**

---

## 📁 Project Structure

```
📦 violence-detection-cctv/
│
├── 🌐 WEB APPLICATION (PRODUCTION)
│   ├── app.py                       # Main Flask entry point
│   ├── templates/                   # UI templates (Dashboard, Realtime)
│   └── static/                      # CSS/JS assets
│
├── 📂 CORE MODULES
│   ├── scripts/
│   │   ├── optimize_model.py        # Model optimization script
│   │   └── benchmark_onnx_yolo.py   # Performance testing script
│   └── models/                      # Optimized ONNX & Keras models
│
├── 🧪 TESTING & BENCHMARKS
│   └── tests/
│       ├── benchmark_test.py        # Performance testing
│       └── verify_segment_logic.py  # Logic verification
│
├── 📓 RESEARCH & NOTEBOOKS
│   └── notebooks/
│       ├── v8_chay.ipynb            # Main training notebook
│       └── eval_only.ipynb          # Model evaluation
│
├── 📈 RESULTS & DATA
│   ├── charts/                      # Training & XAI visualizations
│   ├── reports/                     # Metrics & classification reports
│   ├── data/
│   │   └── sample_videos/           # Test videos for demo
│   ├── uploads/                     # Temporary upload storage
│   └── outputs/                     # Generated segment videos
│
└── ⚙️ CONFIGURATION
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt             # Dependencies
    └── .gitignore
```

---

## 🔬 Technical Details (Deep Dive)

### 🎓 Training Pipeline

**Phase 1: Base Model Training** (15 epochs)
```
- Freeze: MobileNetV2 backbone
- Learn: Classification head
- Learning Rate: 1e-4
- Loss: CategoricalFocalCrossentropy (α=0.25, γ=2.0)
- Early Stop: Patience=7
```

**Phase 2: Fine-tuning** (10 epochs)
```
- Unfreeze: Last 30 layers of MobileNetV2
- Learn: Entire model
- Learning Rate: 1e-5 (lower)
- Scheduler: ReduceLROnPlateau
- Focus: Adapt to violence domain
```

### 🎲 Data Augmentation

```python
# Train-time augmentation techniques:
- Random Horizontal Flip (50%)
- Random Crop (85%) + Resize to 128×128
- Brightness Jitter (±0.2)
- Temporal Dropout: Randomly zero 1-2 frames
- Mixup: Blend features between samples
```

### 🛡️ Overfitting Prevention

| Technique | Parameter | Effectiveness |
|----------|--------|---------|
| Dropout | 0.5 | Reduces co-adaptation |
| EarlyStopping | patience=7 | Prevents over-training |
| ReduceLROnPlateau | factor=0.5 | Fine-tunes learning convergence |
| GroupShuffleSplit | By scene | Avoids data leakage |

---

## 🔍 Explainable AI (XAI)

### 📊 Temporal Attention Visualization

The model uses an **Attention Mechanism** to highlight:
- ✅ Which frames the model focused on.
- ✅ Attention weight intensity (0-100%).
- ✅ Reasoning behind the prediction.

This helps security personnel focus on the most critical moments of an incident.

---

## 📋 API Reference

### POST `/predict`
Detects violence in an uploaded video.

**Response (Violence Detected):**
```json
{
  "class": "Violence",
  "confidence": 0.95,
  "status": "success",
  "session_id": "session_789456",
  "violence_percentage": 60.0
}
```

### POST `/predict_realtime`
Receives base64 frames from live camera stream.

---

## 🚨 Troubleshooting

### ❌ Server won't start
**Error:** `Address already in use`
**Solution:** Kill the process on port 5000 using `lsof -i :5000` or change port in `app.py`.

### ❌ Low Performance/FPS
**Problem:** Video analysis is slow.
**Solution:** Ensure you are using the `_int8.onnx` optimized model. Ensure `onnxruntime` is installed correctly.

### ❌ GPU not detected
**Problem:** TensorFlow using CPU.
**Solution:** For Windows, ensure CUDA/cuDNN are installed or use WSL2.

---

## 📊 Project Statistics

```
📊 PROJECT STATISTICS

Code Metrics:
├── Python Lines:           ~2,500+
├── HTML/CSS Lines:         ~1,500+
├── Total Files:            50+
├── Git Commits:            120+

Model Metrics:
├── Total Parameters:       4.2M
├── Optimized Size:         3.1MB
├── Inference Speed:        ~199ms (CPU)
└── Accuracy:               90.77%

Dataset:
├── Training Videos:        4,800
├── Total Dataset Size:     200GB+
└── Classes:                2 (Normal/Violence)
```

---

## 📝 Update Logs

### [Latest] - 2026-05-02
- **✨ Heatmap & Analytics:** Added `/dashboard` with 24h trends.
- **🛡️ Privacy Mode:** Automatic face blurring (GDPR compliance).
- **📹 Real-time Improvements:** Frame-buffer MP4 generation.
- **⚡ AI Optimization:** YOLOv8 POI Tracking & ONNX INT8 Quantization.

---

## 👨‍💻 Author

- **Name:** Le Hoang
- **Email:** le294594@gmail.com
- **LinkedIn:** [Visin-8386](https://www.linkedin.com/in/visin-8386/)
- **GitHub:** [Visin-8386](https://github.com/Visin-8386)

---
*Developed by: **Le Hoang***

</div>
