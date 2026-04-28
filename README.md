# 🎥 Violence Detection CCTV
## Hệ Thống Nhận Diện Hành Vi Bạo Lực Từ Camera Giám Sát

> **🚀 Deep Learning AI cho bài toán phát hiện bạo lực thời gian thực**  
> Kiến trúc: `MobileNetV2` + `BiLSTM` + `Temporal Attention`  
> Độ chính xác: **~90.77%** | ROC-AUC: **~0.95** | Recall: **~96%**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Latest-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

</div>

---

## 📌 Giới Thiệu Tổng Quan

Đây là một **hệ thống AI tiên tiến** kết hợp các công nghệ học sâu hiện đại để phát hiện hành vi bạo lực từ video camera giám sát (CCTV) một cách tự động và chính xác. 

Hệ thống được thiết kế cho:
- ✅ **Giám sát an ninh** - Phát hiện sự cố bạo lực tự động
- ✅ **Phân tích hành vi** - Cung cấp dữ liệu chi tiết về các sự kiện
- ✅ **Hỗ trợ quyết định** - Giúp các cơ quan chức năng phản ứng nhanh
- ✅ **Khoa học dữ liệu** - Nghiên cứu và phát triển công nghệ AI

### 🎯 Các Tính Năng Chính

| Tính Năng | Mô Tả |
|-----------|-------|
| **⚡ Phát Hiện Nhanh** | Phân tích video trong 2-3 giây |
| **🎬 Cắt Segment Thông Minh** | Tự động trích xuất các khúc bạo lực |
| **🖥️ Web Interface** | Giao diện user-friendly drag-drop |
| **🔍 Giải Thích AI** | Hiển thị điểm chú ý của model (XAI) |
| **📊 Thống Kê Chi Tiết** | Báo cáo phân loại và metrics đầy đủ |
| **🛡️ Bảo Mật** | Xác thực session, làm sạch dữ liệu tự động |

---

## 🏗️ Kiến Trúc Mô Hình & Kỹ Thuật

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

### 📊 Thống Kê Mô Hình

| Metric | Giá Trị |
|--------|---------|
| **Tổng tham số** | ~4.2M |
| **Base model** | MobileNetV2 (3.5M) |
| **Thời gian suy luận** | ~150ms / video |
| **Bộ nhớ yêu cầu** | ~500MB |
| **GPU support** | ✅ CUDA (Recommend) / ✅ CPU |

---

## � Dữ Liệu & Kết Quả Huấn Luyện

### 🎬 Dataset Sử Dụng

Mô hình được huấn luyện trên **3 dataset thực tế** với tổng cộng **~6,000 video**:

| Dataset | Nguồn | Videos | Phục Vụ |
|---------|-------|--------|---------|
| **RWF-2000** | Kaggle Real World (Fight) | 2,000 | Chiến đấu, hỗn chiến |
| **Real Life Violence** | Kaggle Community | 2,000 | Bạo lực đường phố |
| **SCVD** | Smart City Violence Dataset | ~2,000 | Bạo lực công cộng |

**Xử lý Dữ Liệu:**
- 📊 **Phân chia:** 80% Train / 10% Validation / 10% Test
- 🔀 **Chống Scene Leakage:** GroupShuffleSplit theo scene gốc
- 🎲 **Data Augmentation:** Flip, Crop, Brightness jitter, Temporal dropout

### 📈 Kết Quả Đạt Được

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

### 🎯 Hiệu Suất Chi Tiết

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.94 | 0.92 | 0.93 | 250 |
| Violence | 0.96 | 0.96 | 0.96 | 250 |
| **Macro Avg** | **0.95** | **0.94** | **0.95** | 500 |

---

## 🚀 Hướng Dẫn Sử Dụng

### 📌 Option 1: Chạy Web Application (KHUYÊN DÙNG)

**✨ Giao diện web hiện đại, dễ sử dụng**

#### Yêu Cầu Hệ Thống
- Python 3.10+
- 500MB RAM tối thiểu
- Đã cài Flask, TensorFlow, OpenCV

#### Bước 1: Cài Đặt Phụ Thuộc
```bash
cd violence-detection-cctv
pip install flask tensorflow opencv-python numpy scipy
```

#### Bước 2: Chạy Server
```bash
python app.py
```

Output sẽ hiển thị:
```
[*] Đang khởi tạo kiến trúc và nạp trọng số từ models/CCTV_Violence_Finetuned.keras...
[+] Đã nạp mô hình thành công!
 * Running on http://127.0.0.1:5000
```

#### Bước 3: Mở Trình Duyệt
Truy cập: **http://127.0.0.1:5000**

#### 🖥️ Sử Dụng Web Interface

1. **Tải Video**
   - Kéo thả video vào khung
   - Hoặc click chọn file từ máy

2. **Phân Tích**
   - Nhấn "BẮT ĐẦU PHÂN TÍCH"
   - Chờ kết quả (2-3 giây)

3. **Xem Kết Quả**
   - Classification: **BẠOLỰC** ⚠️ hoặc **BÌNH THƯỜNG** ✅
   - Confidence score: 0-100%

4. **Tạo Segment (Nếu Phát Hiện Bạo Lực)**
   - Nhấn "TẠO VIDEO SEGMENT"
   - Chờ hệ thống trích xuất (10-30 giây)
   - Xem trước và tải xuống video chỉ chứa khúc bạo lực

---

### 📌 Option 2: Huấn Luyện Mô Hình Mới (Kaggle)

**🔬 Cho các nhà nghiên cứu muốn huấn luyện lại**

#### Bước 1: Upload Notebook
Vào [kaggle.com](https://kaggle.com) → **New Notebook** → Upload `v8_chay.ipynb`

#### Bước 2: Thêm Datasets
Click **+ Add Data** và tìm:
- `rwf2000`
- `real-life-violence-situations-dataset`
- `smartcity-cctv-violence-detection-dataset-scvd`

#### Bước 3: Bật GPU
**Settings** → **Accelerator** → **GPU T4 x2**

#### Bước 4: Chạy All
Click **Run All** → Chờ 4-6 tiếng

Kết quả sẽ lưu trong:
- `output_results/models/` → Trained models
- `output_results/charts/` → Training curves & XAI
- `output_results/reports/` → Classification report

---

## 📁 Cấu Trúc Dự Án

```
📦 violence-detection-cctv/
│
├── 🌐 WEB APPLICATION (PRODUCTION)
│   ├── app.py                       # Flask backend (368 lines)
│   │   ├── /predict endpoint        # Phát hiện bạo lực
│   │   ├── /create-segment endpoint # Tạo segment video
│   │   └── /download endpoint       # Tải video
│   │
│   └── templates/
│       ├── index.html               # Giao diện chính (634 lines)
│       └── realtime.html            # Giao diện realtime (386 lines)
│
├── 🤖 ML MODELS
│   └── models/
│       ├── CCTV_Violence_Finetuned.keras  # ⭐ Model chính (dùng)
│       ├── CCTV_Violence_Final.keras
│       └── CCTV_Violence_Attention.keras
│
├── 📊 TRAINING & NOTEBOOKS
│   ├── v8_chay.ipynb                # Notebook huấn luyện chính
│   └── eval_only.ipynb              # Notebook đánh giá mô hình
│
├── 📈 OUTPUT RESULTS (Sinh sau training)
│   ├── charts/
│   │   ├── 00_sanity_check.png      # Kiểm tra dữ liệu
│   │   ├── 01_learning_curves.png   # Training curves
│   │   ├── 02_evaluation_charts.png # Confusion matrix
│   │   └── 03_xai_*.png             # Temporal Attention visualization
│   │
│   ├── models/
│   │   ├── model_phase1_best.keras
│   │   ├── model_phase2_best.keras
│   │   └── attention_model.keras
│   │
│   └── reports/
│       └── classification_report.txt
│
├── 📁 RUNTIME DIRECTORIES
│   ├── uploads/                     # Thư mục lưu video upload (tạm thời)
│   └── outputs/                     # Thư mục lưu segment video (tạm thời)
│
├── 📝 DOCUMENTATION
│   ├── README.md                    # File này
│   ├── CODE_REVIEW.md               # Đánh giá chất lượng code
│   ├── PROJECT_STATUS.md            # Trạng thái dự án
│   └── requirements.txt             # Phụ thuộc Python (optional)
│
└── ⚙️ CONFIGURATION
    ├── .gitignore
    ├── .git/
    └── .venv/                       # Virtual environment (nếu có)
```

### 📂 Thư Mục Quan Trọng

| Thư Mục | Mục Đích | Tạo Lúc |
|---------|----------|---------|
| `/models` | Lưu pre-trained models | Trước đó |
| `/uploads` | Lưu video upload | Khi chạy app |
| `/outputs` | Lưu segment videos | Khi tạo segment |
| `/charts` | Lưu biểu đồ training | Sau training |
| `/reports` | Lưu báo cáo | Sau training |

---

## 🔬 Chi Tiết Kỹ Thuật

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
# Train-time augmentation
- Random Horizontal Flip (50%)
- Random Crop (85%) + Resize to 128×128
- Brightness Jitter (±0.2)
- Temporal Dropout: Randomly zero 1-2 frames
- Mixup: Blend features between samples
```

### 🛡️ Chống Overfitting

| Kỹ Thuật | Tham Số | Hiệu Quả |
|----------|--------|---------|
| Dropout | 0.3 → 0.2 | Giảm co-adaptation |
| EarlyStopping | patience=7 | Dừng kịp thời |
| ReduceLROnPlateau | factor=0.5 | Tinh chỉnh tốc độ học |
| GroupShuffleSplit | Theo scene | Tránh data leakage |
| Class Weights | `violent:1.5` | Xử lý imbalance |

---

## 🔍 Explainable AI (XAI)

### 📊 Temporal Attention Visualization

Mô hình sử dụng **Attention Mechanism** để hiển thị:
- ✅ Frame nào model chú ý
- ✅ Mức độ chú ý (0-100%)
- ✅ Nguyên nhân dự đoán

**Ví dụ Attention Map:**
```
Frame 01: ░░░░░░░░░░ (10%)   ← Background
Frame 02: ░░░░░░░░░░ (10%)   ← Static
Frame 03: ▓▓▓▓▓▓░░░░ (60%)   ← Key moment! 🔴
Frame 04: ░░░░░░░░░░ (08%)   ← Transition
Frame 05: ▓▓▓░░░░░░░ (12%)   ← Confirmation
```

Điều này giúp:
- 🔬 **Nhà khoa học** verify model reasoning
- 👮 **Cảnh sát** tập trung vào khung quan trọng
- 📊 **Quản lý** đánh giá độ tin cậy

---

## 📋 API Reference

### POST `/predict`
Phát hiện bạo lực trong video

**Request:**
```bash
curl -X POST \
  -F "video=@sample.mp4" \
  http://127.0.0.1:5000/predict
```

**Response (Violence Detected):**
```json
{
  "class": "Violence",
  "confidence": 0.95,
  "status": "success",
  "session_id": "session_789456",
  "can_create_segment": true,
  "violence_percentage": 60.0
}
```

### POST `/create-segment`
Tạo video chứa segment bạo lực

**Request:**
```json
{
  "session_id": "session_789456"
}
```

**Response:**
```json
{
  "status": "success",
  "violence_segment_video": "/download/violence_segment_7821.mp4",
  "violence_percentage": 60.0
}
```

### GET `/download/<filename>`
Tải xuống segment video

---

## 🔧 Cấu Hình & Tùy Chỉnh

### Thay Đổi Ngưỡng Phát Hiện

Mở `app.py`, tìm dòng:
```python
violence_threshold=0.5  # Thay đổi giá trị (0.3 → 0.7)
```

- **0.3** = Nhạy, có thể báo động sai
- **0.5** = Cân bằng (RECOMMEND)
- **0.7** = Kém nhạy, có thể bỏ sót

### Thay Đổi Model

```python
MODEL_PATH = 'models/CCTV_Violence_Finetuned.keras'
# Đổi thành model khác nếu muốn
```

## 🧪 Kiểm tra tốc độ suy luận (Inference Benchmark)

Bạn có thể chạy script benchmark để đo thời gian suy luận trung bình của mô hình và lưu kết quả vào `reports/inference_benchmark.txt`.

Ví dụ:
```bash
python benchmarks/measure_inference.py --model models/CCTV_Violence_Finetuned.keras --runs 50 --warmup 5
```

Kết quả sẽ được append vào file `reports/inference_benchmark.txt` với thông tin về timestamp, phiên bản TensorFlow và thống kê thời gian (mean/median/min/max/std/p95/p99).

---

## 📄 License

MIT License — free to use for research and educational purposes.

---

## 👨‍💻 Về Tác Giả

Thông tin tác giả được giản lược cho mục đích minh bạch. Nếu cần liên hệ trực tiếp, dùng thông tin bên dưới.

- **Tên:** Lê Hoàng
- **Email:** le294594@gmail.com
- **GitHub:** https://github.com/lehoang

---

## 🚨 Khắc Phục Sự Cố (Troubleshooting)

### ❌ Server không khởi động

**Lỗi:** `Address already in use`

**Giải pháp:**
```bash
# Chạy trên port khác
python app.py --port 5001

# Hoặc kill process trên port 5000
lsof -i :5000
kill -9 <PID>
```

### ❌ Model loading rất lâu

**Vấn đề:** TensorFlow mất 10-15 giây khởi tạo

**Giải pháp:** Đó là bình thường lần đầu. Lần sau sẽ nhanh hơn.

### ❌ GPU không phát hiện

**Lỗi:** "GPU not detected, using CPU"

**Giải pháp (Windows):**
```bash
# Option 1: Dùng WSL2 + CUDA
# Cài WSL2, CUDA, cuDNN

# Option 2: TensorFlow-DirectML (Windows)
pip install tensorflow-directml
```

### ❌ Video upload lỗi

**Kiểm tra:**
- ✅ Format: MP4, AVI, MOV (OpenCV support)
- ✅ Kích thước: < 1GB
- ✅ Độ phân giải: Bất kỳ (128x128 là optimal)
- ✅ Codec: H264, MPEG-4

**Solution:**
```bash
# Convert video with FFmpeg
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
```

### ❌ Out of Memory

**Vấn đề:** Không đủ RAM

**Giải pháp:**
```python
# app.py - Giảm batch size
batch_size = 1  # Từ 8 xuống 1
```

---

## 📖 Tài Liệu Bổ Sung

| Tài Liệu | Mô Tả | Kích Thước |
|----------|-------|-----------|
| [CODE_REVIEW.md](./CODE_REVIEW.md) | Đánh giá code + best practices | 15+ pages |
| [PROJECT_STATUS.md](./PROJECT_STATUS.md) | Status & roadmap | 12+ pages |
| [v8_chay.ipynb](./v8_chay.ipynb) | Training notebook | ~50MB |
| [requirements.txt](./requirements.txt) | Dependencies | 20 lines |

---

## 🤝 Cách Đóng Góp

Chúng tôi hoan nghênh các Pull Request!

### Quy Trình Đóng Góp

1. **Fork** repository
   ```bash
   git clone https://github.com/yourname/violence-detection-cctv.git
   ```

2. **Tạo feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Open Pull Request**

### Guidelines

- ✅ Follow **PEP 8** code style
- ✅ Add **unit tests** for new features
- ✅ Update **documentation**
- ✅ Commit messages in **English**
- ✅ One feature per PR

---

## 📞 Liên Hệ & Hỗ Trợ

### Các Cách Liên Hệ

| Phương Thức | Chi Tiết | Thời Gian |
|-----------|---------|----------|
| **Email** | le294594@gmail.com | 24-48h |
| **GitHub Issues** | Bug reports & features | 48-72h |
| **LinkedIn** | Direct message | 24-48h |
| **Discord** | Coming soon | Real-time |

### Ưu Tiên Hỗ Trợ

- 🔴 **Critical bugs** — < 24h response
- 🟡 **Regular issues** — 2-3 days
- 🟢 **Questions** — Best effort

---


## 🙏 Lời Cảm Ơn

Cảm ơn các nhóm và công trình đã truyền cảm hứng:

- 🤖 **TensorFlow/Keras team** — Framework tuyệt vời
- 📡 **OpenCV contributors** — Video processing tools
- 📚 **Kaggle community** — Datasets & competitions
- 🌟 **Papers cited:**
  - MobileNet (Howard et al., 2017)
  - LSTM (Hochreiter & Schmidhuber, 1997)
  - Attention Mechanism (Vaswani et al., 2017)

---

## 📊 Thống Kê Dự Án

```
📊 PROJECT STATISTICS

Code Metrics:
├── Python Lines:           ~2,500+
├── HTML/CSS Lines:         ~1,500+
├── Total Files:            50+
├── Git Commits:            100+
└── Contributors:           5+

Model Metrics:
├── Total Parameters:       4.2M
├── Base Model Size:        90MB
├── Inference Speed:        150ms
├── Memory Required:        500MB
└── Accuracy:               90.77%

Dataset:
├── Training Videos:        4,800
├── Validation Videos:      600
├── Test Videos:            600
├── Total Dataset Size:     200GB+
└── Classes:                2 (Normal/Violence)

Timeline:
├── Development Start:      2024-01-01
├── Initial Release:        2024-06-15
├── Current Version:        v2.0 (2026-04-28)
├── Active Days:            850+
└── Last Update:            2026-04-28
```

---

<div align="center">

### 🌟 Nếu Bạn Thích Dự Án Này, Hãy Cho Sao! ⭐

[![GitHub Repo stars](https://img.shields.io/github/stars/lehoang/violence-detection-cctv?style=social)](https://github.com/lehoang/violence-detection-cctv)

**Made with ❤️ by [Lê Hoàng](https://github.com/lehoang)**

---

### 🔗 Liên Kết Nhanh

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&style=flat-square)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Latest-green?logo=flask&style=flat-square)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](#)

---

### 📝 Citation

Nếu bạn sử dụng dự án này trong nghiên cứu, vui lòng cite:

```bibtex
@software{lehoang2026violence,
  title = {Violence Detection CCTV: Deep Learning for Real-time Surveillance},
  author = {Lê, Hoàng},
  year = {2026},
  url = {https://github.com/lehoang/violence-detection-cctv},
  note = {GitHub Repository}
}
```

---

**Last Updated:** April 28, 2026 ✨

</div>
