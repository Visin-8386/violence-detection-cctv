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

This is an **advanced AI system** integrating modern deep learning technologies to detect violent behavior from CCTV surveillance videos automatically and accurately.

The system is designed for:
- ✅ **Security Monitoring** - Automatic violence incident detection.
- ✅ **Behavioral Analysis** - Providing detailed event logs and analytics.
- ✅ **Decision Support** - Assisting authorities in rapid response.
- ✅ **Data Science** - AI research and developmental engineering.

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

## ⚡ Technical Optimizations

The project utilizes **Inference Pipeline Engineering** to balance Accuracy, Latency, and Scalability for real-world deployment.

### 1. Model Compression & Acceleration (Quantization)
*   **Technique:** Converted Keras model to **ONNX Runtime** with **Dynamic INT8 Quantization**.
*   **Goal:** Resolving computational bottlenecks on non-GPU systems.
*   **Impact:** 
    *   Inference speed increased by **~350%** (700ms down to <200ms on CPU).
    *   Storage optimized to **3.1MB** (10x reduction from original 30MB).

### 2. Spatial-Temporal POI Tracking (Background Filtering)
*   **Technique:** Integrated **YOLOv8 Nano** as a Pre-filter for **Person-of-Interest (POI)** identification.
*   **Mechanism:** The system uses YOLOv8 to locate humans, then applies **Dynamic Cropping** to feed only relevant regions into the violence model.
*   **Impact:** 
    *   Eliminated **False Positives** caused by environmental factors (trees, shadows, weather).
    *   Improved real-world accuracy by removing irrelevant background noise.

### 3. Buffer-based Event Reconstruction
*   **Technique:** Implemented a **Sliding Frame Buffer** (3s Pre-event & 2s Post-event).
*   **Impact:** Automatically exports incident videos in MP4 format, including **3 seconds of context before the violence occurred**, providing full data for surveillance and investigation.

---

## 🚀 Getting Started

### 📌 Option 1: Run Local (Quick Test)

#### Step 1: Install Dependencies
```bash
cd violence-detection-cctv
pip install flask tensorflow opencv-python numpy scipy onnxruntime ultralytics
```

#### Step 2: Launch Server
```bash
python app.py
```

#### Step 3: Open Browser
Navigate to: **http://127.0.0.1:5000**

---

### 🐳 Option 2: Run with Docker (Recommended)

```bash
docker-compose up --build -d
```

---

## 🔬 Technical Details (Research & Training)

### 🎬 Training Dataset
The model was trained on **3 real-world datasets** totaling **~6,000 videos**:
- **RWF-2000**: Real World Fight dataset (2,000 videos).
- **Real Life Violence**: Street violence situations (2,000 videos).
- **SCVD**: Smart City Violence Dataset (~2,000 videos).

### 📈 Evaluation Metrics
```
┌──────────────────────────────────────────┐
│         EVALUATION METRICS               │
│ Accuracy:         90.77%  ⭐⭐⭐⭐⭐    │
│ ROC-AUC Score:    0.95    ⭐⭐⭐⭐⭐    │
│ Violence Recall:  96%     ⭐⭐⭐⭐⭐    │
└──────────────────────────────────────────┘
```

---

## 🔍 Explainable AI (XAI)

The system uses a **Temporal Attention Mechanism** to visualize which frames in a sequence triggered the violence alert, helping verify model reasoning.

---

## 📊 Project Statistics

```
📊 PROJECT STATISTICS

Code Metrics:
├── Python Lines:           ~2,500+
├── Total Files:            50+
├── Git Commits:            120+

Model Metrics:
├── Total Parameters:       4.2M
├── Optimized Size:         3.1MB
├── Inference Speed:        ~199ms (CPU)
└── Accuracy:               90.77%
```

---

## 📝 Update Logs (Latest: 2026-05-02)
- **✨ Heatmap & Analytics:** Added `/dashboard` displaying 24h incident trends.
- **🛡️ Privacy Mode:** Integrated automatic face blurring (GDPR).
- **📹 Real-time improvements:** Migrated to server-side MP4 generation.
- **⚡ AI Optimization:** Integrated YOLOv8 POI tracking and ONNX INT8 Quantization.

---

## 👨‍💻 Author

- **Name:** Le Hoang
- **Email:** le294594@gmail.com
- **LinkedIn:** [Visin-8386](https://www.linkedin.com/in/visin-8386/)
- **GitHub:** [Visin-8386](https://github.com/Visin-8386)

---
*Developed by: **Le Hoang***

</div>
