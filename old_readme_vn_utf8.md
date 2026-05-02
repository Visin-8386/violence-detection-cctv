# ≡ƒÄÑ Violence Detection CCTV
## Hß╗ç Thß╗æng Nhß║¡n Diß╗çn H├ánh Vi Bß║ío Lß╗▒c Tß╗½ Camera Gi├ím S├ít

> **≡ƒÜÇ Deep Learning AI cho b├ái to├ín ph├ít hiß╗çn bß║ío lß╗▒c thß╗¥i gian thß╗▒c**  
> Kiß║┐n tr├║c: `MobileNetV2` + `BiLSTM` + `Temporal Attention`  
> ─Éß╗Ö ch├¡nh x├íc: **~90.77%** | ROC-AUC: **~0.95** | Recall: **~96%**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Latest-green?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square)

</div>

---

## ≡ƒôî Giß╗¢i Thiß╗çu Tß╗òng Quan

─É├óy l├á mß╗Öt **hß╗ç thß╗æng AI ti├¬n tiß║┐n** kß║┐t hß╗úp c├íc c├┤ng nghß╗ç hß╗ìc s├óu hiß╗çn ─æß║íi ─æß╗â ph├ít hiß╗çn h├ánh vi bß║ío lß╗▒c tß╗½ video camera gi├ím s├ít (CCTV) mß╗Öt c├ích tß╗▒ ─æß╗Öng v├á ch├¡nh x├íc. 

Hß╗ç thß╗æng ─æ╞░ß╗úc thiß║┐t kß║┐ cho:
- Γ£à **Gi├ím s├ít an ninh** - Ph├ít hiß╗çn sß╗▒ cß╗æ bß║ío lß╗▒c tß╗▒ ─æß╗Öng
- Γ£à **Ph├ón t├¡ch h├ánh vi** - Cung cß║Ñp dß╗» liß╗çu chi tiß║┐t vß╗ü c├íc sß╗▒ kiß╗çn
- Γ£à **Hß╗ù trß╗ú quyß║┐t ─æß╗ïnh** - Gi├║p c├íc c╞í quan chß╗⌐c n─âng phß║ún ß╗⌐ng nhanh
- Γ£à **Khoa hß╗ìc dß╗» liß╗çu** - Nghi├¬n cß╗⌐u v├á ph├ít triß╗ân c├┤ng nghß╗ç AI

### ≡ƒÄ» C├íc T├¡nh N─âng Ch├¡nh

| T├¡nh N─âng | M├┤ Tß║ú |
|-----------|-------|
| **ΓÜí Ph├ít Hiß╗çn Nhanh** | Ph├ón t├¡ch video trong 2-3 gi├óy |
| **≡ƒÄ¼ Cß║»t Segment Th├┤ng Minh** | Tß╗▒ ─æß╗Öng tr├¡ch xuß║Ñt c├íc kh├║c bß║ío lß╗▒c |
| **≡ƒôè Dashboard & Heatmap** | Theo d├╡i xu h╞░ß╗¢ng bß║ío lß╗▒c theo thß╗¥i gian/camera |
| **≡ƒÄÑ Real-time Monitoring** | Gi├ím s├ít trß╗▒c tiß║┐p tß╗½ Webcam/CCTV |
| **≡ƒôÑ Auto Incident Export** | Tß╗▒ ─æß╗Öng quay v├á l╞░u video MP4 khi c├│ sß╗▒ cß╗æ |
| **≡ƒ¢í∩╕Å Privacy Masking** | Tß╗▒ ─æß╗Öng l├ám mß╗¥ mß║╖t (GDPR compliance) |
| **≡ƒÉ│ Docker Support** | Triß╗ân khai nhanh ch├│ng, nhß║Ñt qu├ín m├┤i tr╞░ß╗¥ng |
| **≡ƒöì Giß║úi Th├¡ch AI** | Hiß╗ân thß╗ï ─æiß╗âm ch├║ ├╜ cß╗ºa model (XAI) |

---

## ≡ƒÅù∩╕Å Kiß║┐n Tr├║c M├┤ H├¼nh & Kß╗╣ Thuß║¡t

### ≡ƒºá Neural Network Architecture

```
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé                      Video Input                            Γöé
Γöé              (Batch, 15 frames, 128├ù128, RGB)              Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                       Γöé
        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
        Γöé   MobileNetV2 + TimeDistributed   Γöé
        Γöé  (Transfer Learning - ImageNet)   Γöé
        Γöé    ΓåÆ Extract spatial features     Γöé
        Γöé     (B├ù15, 1280-dim vectors)      Γöé
        ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                       Γöé
        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
        Γöé   Bidirectional LSTM (64 units)  Γöé
        Γöé   ΓåÆ Model temporal dynamics      Γöé
        Γöé   ΓåÆ (B, 15, 128-dim context)    Γöé
        ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                       Γöé
        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
        Γöé  Temporal Attention Layer     Γöé
        Γöé  ΓåÆ Focus on key frames        Γöé
        Γöé  ΓåÆ Weight normalization       Γöé
        Γöé  ΓåÆ (B, 128-dim context)       Γöé
        ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                       Γöé
        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
        Γöé   Dense(64, ReLU) + Dropout   Γöé
        Γöé   Dense(2, Softmax)           Γöé
        ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                       Γöé
        ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓû╝ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
        Γöé   Output: [Normal, Violence]  Γöé
        Γöé   Confidence: 0-100%          Γöé
        ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
```

### ≡ƒôè Thß╗æng K├¬ M├┤ H├¼nh

| Metric | Gi├í Trß╗ï |
|--------|---------|
| **Dung l╞░ß╗úng M├┤ H├¼nh** | 3.1 MB (ONNX INT8) - *Tr╞░ß╗¢c ─æ├óy 30MB* |
| **Thß╗¥i gian suy luß║¡n** | ~199ms / 15 frames (CPU) |
| **Bß╗Ö nhß╗¢ y├¬u cß║ºu** | ~150MB |
| **Tß╗æi ╞░u h├│a Pipeline** | YOLOv8 (Person Tracking) + ONNX Runtime |
| **GPU support** | Kh├┤ng bß║»t buß╗Öc (Chß║íy cß╗▒c m╞░ß╗út tr├¬n CPU) |

---

## ∩┐╜ Dß╗» Liß╗çu & Kß║┐t Quß║ú Huß║Ñn Luyß╗çn

### ≡ƒÄ¼ Dataset Sß╗¡ Dß╗Ñng

M├┤ h├¼nh ─æ╞░ß╗úc huß║Ñn luyß╗çn tr├¬n **3 dataset thß╗▒c tß║┐** vß╗¢i tß╗òng cß╗Öng **~6,000 video**:

| Dataset | Nguß╗ôn | Videos | Phß╗Ñc Vß╗Ñ |
|---------|-------|--------|---------|
| **RWF-2000** | Kaggle Real World (Fight) | 2,000 | Chiß║┐n ─æß║Ñu, hß╗ùn chiß║┐n |
| **Real Life Violence** | Kaggle Community | 2,000 | Bß║ío lß╗▒c ─æ╞░ß╗¥ng phß╗æ |
| **SCVD** | Smart City Violence Dataset | ~2,000 | Bß║ío lß╗▒c c├┤ng cß╗Öng |

**Xß╗¡ l├╜ Dß╗» Liß╗çu:**
- ≡ƒôè **Ph├ón chia:** 80% Train / 10% Validation / 10% Test
- ≡ƒöÇ **Chß╗æng Scene Leakage:** GroupShuffleSplit theo scene gß╗æc
- ≡ƒÄ▓ **Data Augmentation:** Flip, Crop, Brightness jitter, Temporal dropout

### ≡ƒôê Kß║┐t Quß║ú ─Éß║ít ─É╞░ß╗úc

```
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé         EVALUATION METRICS               Γöé
Γö£ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöñ
Γöé Accuracy:         90.77%  Γ¡ÉΓ¡ÉΓ¡ÉΓ¡ÉΓ¡É    Γöé
Γöé ROC-AUC Score:    0.95    Γ¡ÉΓ¡ÉΓ¡ÉΓ¡ÉΓ¡É    Γöé
Γöé Violence Recall:  96%     Γ¡ÉΓ¡ÉΓ¡ÉΓ¡ÉΓ¡É    Γöé
Γöé Precision:        0.96    Γ¡ÉΓ¡ÉΓ¡ÉΓ¡ÉΓ¡É    Γöé
Γöé F1-Score:         0.95    Γ¡ÉΓ¡ÉΓ¡ÉΓ¡ÉΓ¡É    Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
```

### ≡ƒÄ» Hiß╗çu Suß║Ñt Chi Tiß║┐t

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.94 | 0.92 | 0.93 | 250 |
| Violence | 0.96 | 0.96 | 0.96 | 250 |
| **Macro Avg** | **0.95** | **0.94** | **0.95** | 500 |

---

## ΓÜí Tß╗æi ╞»u H├│a Hß╗ç Thß╗æng (Technical Optimizations)

Dß╗▒ ├ín ─æ├ú ─æ╞░ß╗úc ├íp dß╗Ñng c├íc kß╗╣ thuß║¡t **Inference Pipeline Engineering** ─æß╗â giß║úi quyß║┐t b├ái to├ín c├ón bß║▒ng giß╗»a ─æß╗Ö ch├¡nh x├íc (Accuracy), tß╗æc ─æß╗Ö xß╗¡ l├╜ (Latency) v├á khß║ú n─âng mß╗ƒ rß╗Öng (Scalability).

### 1. Model Compression & Acceleration (L╞░ß╗úng tß╗¡ h├│a & T─âng tß╗æc)
*   **Kß╗╣ thuß║¡t:** Chuyß╗ân ─æß╗òi m├┤ h├¼nh tß╗½ Keras sang **ONNX Runtime** kß║┐t hß╗úp vß╗¢i **Dynamic INT8 Quantization**.
*   **Mß╗Ñc ti├¬u:** Giß║úi quyß║┐t n├║t thß║»t cß╗ò chai (Bottleneck) vß╗ü t├¡nh to├ín tr├¬n c├íc hß╗ç thß╗æng kh├┤ng c├│ GPU.
*   **Kß║┐t quß║ú:** 
    *   Tß╗æc ─æß╗Ö suy luß║¡n (Inference Speed) t─âng **~350%** (tß╗½ 700ms xuß╗æng <200ms tr├¬n CPU).
    *   Tß╗æi ╞░u dung l╞░ß╗úng bß╗Ö nhß╗¢ (Memory Footprint) xuß╗æng c├▓n **3.1MB** (Giß║úm 10 lß║ºn so vß╗¢i bß║ún gß╗æc 30MB).

### 2. Spatial-Temporal POI Tracking (Lß╗ìc nhiß╗àu ngß╗» cß║únh)
*   **Kß╗╣ thuß║¡t:** T├¡ch hß╗úp **YOLOv8 Nano** l├ám bß╗Ö tiß╗ün lß╗ìc (Pre-filter) ─æß╗â x├íc ─æß╗ïnh **Person-of-Interest (POI)**.
*   **C╞í chß║┐:** Hß╗ç thß╗æng sß╗¡ dß╗Ñng YOLOv8 ─æß╗â ─æß╗ïnh vß╗ï con ng╞░ß╗¥i, sau ─æ├│ ├íp dß╗Ñng **Dynamic Cropping** ─æß╗â chß╗ë ─æ╞░a v├╣ng chß╗⌐a ─æß╗æi t╞░ß╗úng v├áo m├┤ h├¼nh nhß║¡n diß╗çn bß║ío lß╗▒c.
*   **Kß║┐t quß║ú:** 
    *   Triß╗çt ti├¬u **False Positive (B├ío ─æß╗Öng giß║ú)** g├óy ra bß╗ƒi c├íc yß║┐u tß╗æ ngoß║íi cß║únh (c├óy cß╗æi, b├│ng ─æß╗ò, thß╗¥i tiß║┐t).
    *   T─âng ─æß╗Ö ch├¡nh x├íc thß╗▒c tß║┐ bß║▒ng c├ích loß║íi bß╗Å c├íc ─æß║╖c tr╞░ng nß╗ün (background noise).

### 3. Buffer-based Event Reconstruction (Ghi h├¼nh sß╗▒ kiß╗çn th├┤ng minh)
*   **Kß╗╣ thuß║¡t:** Triß╗ân khai c╞í chß║┐ **Sliding Frame Buffer** (3s Pre-event & 2s Post-event).
*   **Kß║┐t quß║ú:** Tß╗▒ ─æß╗Öng tr├¡ch xuß║Ñt video sß╗▒ cß╗æ chuß║⌐n MP4, bao gß╗ôm cß║ú bß╗æi cß║únh **3 gi├óy tr╞░ß╗¢c khi xß║úy ra bß║ío lß╗▒c**, cung cß║Ñp dß╗» liß╗çu ─æß║ºy ─æß╗º cho c├┤ng t├íc gi├ím s├ít v├á ─æiß╗üu tra.

### ≡ƒÅå Bß║úng So S├ính Hiß╗çu Suß║Ñt (CPU-only Benchmarks)

| Chß╗ë sß╗æ | Legacy Pipeline (Keras FP32) | Optimized Pipeline (ONNX INT8 + YOLO) |
|--------|------------------------------|----------------------------------------|
| **Dung l╞░ß╗úng M├┤ h├¼nh** | ~30.2 MB | **3.1 MB** *(Giß║úm 10 lß║ºn)* |
| **Inference Latency** | ~600 - 800 ms | **~199 ms** *(T─âng tß╗æc 3.5 lß║ºn)* |
| **False Positive Rate** | Cao (dß╗à nhiß╗àu nß╗ün) | **Rß║Ñt thß║Ñp** (nhß╗¥ POI Tracking) |
| **Deployment** | Nß║╖ng, phß╗Ñ thuß╗Öc nhiß╗üu lib | **Lightweight**, Dockerized |

---

## ≡ƒÜÇ H╞░ß╗¢ng Dß║½n Sß╗¡ Dß╗Ñng

### ≡ƒôî Option 1: Chß║íy Web Application (KHUY├èN D├ÖNG)

**Γ£¿ Giao diß╗çn web hiß╗çn ─æß║íi, dß╗à sß╗¡ dß╗Ñng**

#### Y├¬u Cß║ºu Hß╗ç Thß╗æng
- Python 3.10+
- 500MB RAM tß╗æi thiß╗âu
- ─É├ú c├ái Flask, TensorFlow, OpenCV

#### B╞░ß╗¢c 1: C├ái ─Éß║╖t Phß╗Ñ Thuß╗Öc
```bash
cd violence-detection-cctv
pip install flask tensorflow opencv-python numpy scipy
```

#### B╞░ß╗¢c 2: Chß║íy Server
```bash
python app.py
```

Output sß║╜ hiß╗ân thß╗ï:
```
[*] ─Éang khß╗ƒi tß║ío kiß║┐n tr├║c v├á nß║íp trß╗ìng sß╗æ tß╗½ models/CCTV_Violence_Finetuned.keras...
[+] ─É├ú nß║íp m├┤ h├¼nh th├ánh c├┤ng!
 * Running on http://127.0.0.1:5000
```

#### B╞░ß╗¢c 3: Mß╗ƒ Tr├¼nh Duyß╗çt
Truy cß║¡p: **http://127.0.0.1:5000**

---

### ≡ƒÉ│ Option 1B: Chß║íy bß║▒ng Docker (Khuy├¬n D├╣ng)

Viß╗çc chß║íy bß║▒ng Docker gi├║p tr├ính c├íc lß╗ùi c├ái ─æß║╖t li├¬n quan ─æß║┐n phi├¬n bß║ún th╞░ viß╗çn hoß║╖c OpenCV tr├¬n c├íc hß╗ç ─æiß╗üu h├ánh kh├íc nhau.

#### B╞░ß╗¢c 1: Build v├á chß║íy bß║▒ng Docker Compose
```bash
docker-compose up --build -d
```

#### B╞░ß╗¢c 2: Xem Logs (T├╣y chß╗ìn)
```bash
docker logs -f violence-cctv
```

#### B╞░ß╗¢c 3: Mß╗ƒ Tr├¼nh Duyß╗çt
Truy cß║¡p: **http://127.0.0.1:5000**

#### ≡ƒûÑ∩╕Å Sß╗¡ Dß╗Ñng Web Interface

1. **Tß║úi Video**
   - K├⌐o thß║ú video v├áo khung
   - Hoß║╖c click chß╗ìn file tß╗½ m├íy

2. **Ph├ón T├¡ch**
   - Nhß║Ñn "Bß║«T ─Éß║ªU PH├éN T├ìCH"
   - Chß╗¥ kß║┐t quß║ú (2-3 gi├óy)

3. **Xem Kß║┐t Quß║ú**
   - Classification: **Bß║áOLß╗░C** ΓÜá∩╕Å hoß║╖c **B├îNH TH╞»ß╗£NG** Γ£à
   - Confidence score: 0-100%

4. **Tß║ío Segment (Nß║┐u Ph├ít Hiß╗çn Bß║ío Lß╗▒c)**
   - Nhß║Ñn "Tß║áO VIDEO SEGMENT"
   - Chß╗¥ hß╗ç thß╗æng tr├¡ch xuß║Ñt (10-30 gi├óy)
   - Xem tr╞░ß╗¢c v├á tß║úi xuß╗æng video chß╗ë chß╗⌐a kh├║c bß║ío lß╗▒c

---

### ≡ƒôî Option 2: Huß║Ñn Luyß╗çn M├┤ H├¼nh Mß╗¢i (Kaggle)

**≡ƒö¼ Cho c├íc nh├á nghi├¬n cß╗⌐u muß╗æn huß║Ñn luyß╗çn lß║íi**

#### B╞░ß╗¢c 1: Upload Notebook
V├áo [kaggle.com](https://kaggle.com) ΓåÆ **New Notebook** ΓåÆ Upload `v8_chay.ipynb`

#### B╞░ß╗¢c 2: Th├¬m Datasets
Click **+ Add Data** v├á t├¼m:
- `rwf2000`
- `real-life-violence-situations-dataset`
- `smartcity-cctv-violence-detection-dataset-scvd`

#### B╞░ß╗¢c 3: Bß║¡t GPU
**Settings** ΓåÆ **Accelerator** ΓåÆ **GPU T4 x2**

#### B╞░ß╗¢c 4: Chß║íy All
Click **Run All** ΓåÆ Chß╗¥ 4-6 tiß║┐ng

Kß║┐t quß║ú sß║╜ l╞░u trong:
- `output_results/models/` ΓåÆ Trained models
- `output_results/charts/` ΓåÆ Training curves & XAI
- `output_results/reports/` ΓåÆ Classification report

---

## ≡ƒôü Cß║Ñu Tr├║c Dß╗▒ ├ün

```
≡ƒôª violence-detection-cctv/
Γöé
Γö£ΓöÇΓöÇ ≡ƒîÉ WEB APPLICATION (PRODUCTION)
Γöé   Γö£ΓöÇΓöÇ app.py                       # Main Flask entry point
Γöé   Γö£ΓöÇΓöÇ templates/                   # UI templates
Γöé   ΓööΓöÇΓöÇ static/                      # CSS/JS assets (if any)
Γöé
Γö£ΓöÇΓöÇ ≡ƒôé CORE MODULES
Γöé   Γö£ΓöÇΓöÇ scripts/
Γöé   Γöé   ΓööΓöÇΓöÇ app_realtime.py          # Real-time monitoring version
Γöé   ΓööΓöÇΓöÇ models/                      # Pre-trained .keras models
Γöé
Γö£ΓöÇΓöÇ ≡ƒº¬ TESTING & BENCHMARKS
Γöé   ΓööΓöÇΓöÇ tests/
Γöé       Γö£ΓöÇΓöÇ benchmark_test.py        # Performance testing
Γöé       Γö£ΓöÇΓöÇ test_api_direct.py       # API testing
Γöé       Γö£ΓöÇΓöÇ test_multi_video_fix.py  # Multi-segment fix verification
Γöé       ΓööΓöÇΓöÇ verify_segment_logic.py  # Logic verification
Γöé
Γö£ΓöÇΓöÇ ≡ƒôô RESEARCH & NOTEBOOKS
Γöé   ΓööΓöÇΓöÇ notebooks/
Γöé       Γö£ΓöÇΓöÇ v8_chay.ipynb            # Main training notebook
Γöé       Γö£ΓöÇΓöÇ eval_only.ipynb          # Model evaluation
Γöé       ΓööΓöÇΓöÇ v8_chay_backup.ipynb     # Training backup
Γöé
Γö£ΓöÇΓöÇ ≡ƒôê RESULTS & DATA
Γöé   Γö£ΓöÇΓöÇ charts/                      # Training & XAI visualizations
Γöé   Γö£ΓöÇΓöÇ reports/                     # Metrics & classification reports
Γöé   Γö£ΓöÇΓöÇ data/
Γöé   Γöé   ΓööΓöÇΓöÇ sample_videos/           # Test videos for demo
Γöé   Γö£ΓöÇΓöÇ uploads/                     # Temporary upload storage
Γöé   ΓööΓöÇΓöÇ outputs/                     # Generated segment videos
Γöé
Γö£ΓöÇΓöÇ ≡ƒô¥ DOCUMENTATION
Γöé   Γö£ΓöÇΓöÇ README.md                    # Project landing page
Γöé   ΓööΓöÇΓöÇ docs/
Γöé       Γö£ΓöÇΓöÇ CODE_REVIEW.md           # Code quality report
Γöé       Γö£ΓöÇΓöÇ PROJECT_STATUS.md        # Roadmap & progress
Γöé       ΓööΓöÇΓöÇ slides_outline_vn.md     # Presentation outline
Γöé
ΓööΓöÇΓöÇ ΓÜÖ∩╕Å CONFIGURATION
    Γö£ΓöÇΓöÇ .gitignore
    Γö£ΓöÇΓöÇ requirements.txt             # Dependencies
    ΓööΓöÇΓöÇ .venv/                       # Virtual environment
```

### ≡ƒôé Th╞░ Mß╗Ñc Quan Trß╗ìng

| Th╞░ Mß╗Ñc | Mß╗Ñc ─É├¡ch | Tß║ío L├║c |
|---------|----------|---------|
| `/models` | L╞░u pre-trained models | Tr╞░ß╗¢c ─æ├│ |
| `/uploads` | L╞░u video upload | Khi chß║íy app |
| `/outputs` | L╞░u segment videos | Khi tß║ío segment |
| `/charts` | L╞░u biß╗âu ─æß╗ô training | Sau training |
| `/reports` | L╞░u b├ío c├ío | Sau training |

---

## ≡ƒö¼ Chi Tiß║┐t Kß╗╣ Thuß║¡t

### ≡ƒÄô Training Pipeline

**Phase 1: Base Model Training** (15 epochs)
```
- Freeze: MobileNetV2 backbone
- Learn: Classification head
- Learning Rate: 1e-4
- Loss: CategoricalFocalCrossentropy (╬▒=0.25, ╬│=2.0)
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

### ≡ƒÄ▓ Data Augmentation

```python
# Train-time augmentation
- Random Horizontal Flip (50%)
- Random Crop (85%) + Resize to 128├ù128
- Brightness Jitter (┬▒0.2)
- Temporal Dropout: Randomly zero 1-2 frames
- Mixup: Blend features between samples
```

### ≡ƒ¢í∩╕Å Chß╗æng Overfitting

| Kß╗╣ Thuß║¡t | Tham Sß╗æ | Hiß╗çu Quß║ú |
|----------|--------|---------|
| Dropout | 0.3 ΓåÆ 0.2 | Giß║úm co-adaptation |
| EarlyStopping | patience=7 | Dß╗½ng kß╗ïp thß╗¥i |
| ReduceLROnPlateau | factor=0.5 | Tinh chß╗ënh tß╗æc ─æß╗Ö hß╗ìc |
| GroupShuffleSplit | Theo scene | Tr├ính data leakage |
| Class Weights | `violent:1.5` | Xß╗¡ l├╜ imbalance |

---

## ≡ƒöì Explainable AI (XAI)

### ≡ƒôè Temporal Attention Visualization

M├┤ h├¼nh sß╗¡ dß╗Ñng **Attention Mechanism** ─æß╗â hiß╗ân thß╗ï:
- Γ£à Frame n├áo model ch├║ ├╜
- Γ£à Mß╗⌐c ─æß╗Ö ch├║ ├╜ (0-100%)
- Γ£à Nguy├¬n nh├ón dß╗▒ ─æo├ín

**V├¡ dß╗Ñ Attention Map:**
```
Frame 01: ΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæ (10%)   ΓåÉ Background
Frame 02: ΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæ (10%)   ΓåÉ Static
Frame 03: ΓûôΓûôΓûôΓûôΓûôΓûôΓûæΓûæΓûæΓûæ (60%)   ΓåÉ Key moment! ≡ƒö┤
Frame 04: ΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæΓûæ (08%)   ΓåÉ Transition
Frame 05: ΓûôΓûôΓûôΓûæΓûæΓûæΓûæΓûæΓûæΓûæ (12%)   ΓåÉ Confirmation
```

─Éiß╗üu n├áy gi├║p:
- ≡ƒö¼ **Nh├á khoa hß╗ìc** verify model reasoning
- ≡ƒæ« **Cß║únh s├ít** tß║¡p trung v├áo khung quan trß╗ìng
- ≡ƒôè **Quß║ún l├╜** ─æ├ính gi├í ─æß╗Ö tin cß║¡y

---

## ≡ƒôï API Reference

### POST `/predict`
Ph├ít hiß╗çn bß║ío lß╗▒c trong video

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
Tß║ío video chß╗⌐a segment bß║ío lß╗▒c

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
Tß║úi xuß╗æng segment video

---

## ≡ƒöº Cß║Ñu H├¼nh & T├╣y Chß╗ënh

### Thay ─Éß╗òi Ng╞░ß╗íng Ph├ít Hiß╗çn

Mß╗ƒ `app.py`, t├¼m d├▓ng:
```python
violence_threshold=0.5  # Thay ─æß╗òi gi├í trß╗ï (0.3 ΓåÆ 0.7)
```

- **0.3** = Nhß║íy, c├│ thß╗â b├ío ─æß╗Öng sai
- **0.5** = C├ón bß║▒ng (RECOMMEND)
- **0.7** = K├⌐m nhß║íy, c├│ thß╗â bß╗Å s├│t

### Thay ─Éß╗òi Model

```python
MODEL_PATH = 'models/CCTV_Violence_Finetuned.keras'
# ─Éß╗òi th├ánh model kh├íc nß║┐u muß╗æn
```

## ≡ƒº¬ Kiß╗âm tra tß╗æc ─æß╗Ö suy luß║¡n (Inference Benchmark)

Bß║ín c├│ thß╗â chß║íy script benchmark ─æß╗â ─æo thß╗¥i gian suy luß║¡n trung b├¼nh cß╗ºa m├┤ h├¼nh v├á l╞░u kß║┐t quß║ú v├áo `reports/inference_benchmark.txt`.

V├¡ dß╗Ñ:
```bash
python benchmarks/measure_inference.py --model models/CCTV_Violence_Finetuned.keras --runs 50 --warmup 5
```

Kß║┐t quß║ú sß║╜ ─æ╞░ß╗úc append v├áo file `reports/inference_benchmark.txt` vß╗¢i th├┤ng tin vß╗ü timestamp, phi├¬n bß║ún TensorFlow v├á thß╗æng k├¬ thß╗¥i gian (mean/median/min/max/std/p95/p99).

---

## ≡ƒôä License

MIT License ΓÇö free to use for research and educational purposes.

---

## ≡ƒæ¿ΓÇì≡ƒÆ╗ Vß╗ü T├íc Giß║ú

Th├┤ng tin t├íc giß║ú ─æ╞░ß╗úc giß║ún l╞░ß╗úc cho mß╗Ñc ─æ├¡ch minh bß║ích. Nß║┐u cß║ºn li├¬n hß╗ç trß╗▒c tiß║┐p, d├╣ng th├┤ng tin b├¬n d╞░ß╗¢i.

- **T├¬n:** L├¬ Ho├áng
- **Email:** le294594@gmail.com
- **GitHub:** https://github.com/Visin-8386

---

## ≡ƒÜ¿ Khß║»c Phß╗Ñc Sß╗▒ Cß╗æ (Troubleshooting)

### Γ¥î Server kh├┤ng khß╗ƒi ─æß╗Öng

**Lß╗ùi:** `Address already in use`

**Giß║úi ph├íp:**
```bash
# Chß║íy tr├¬n port kh├íc
python app.py --port 5001

# Hoß║╖c kill process tr├¬n port 5000
lsof -i :5000
kill -9 <PID>
```

### Γ¥î Model loading rß║Ñt l├óu

**Vß║Ñn ─æß╗ü:** TensorFlow mß║Ñt 10-15 gi├óy khß╗ƒi tß║ío

**Giß║úi ph├íp:** ─É├│ l├á b├¼nh th╞░ß╗¥ng lß║ºn ─æß║ºu. Lß║ºn sau sß║╜ nhanh h╞ín.

### Γ¥î GPU kh├┤ng ph├ít hiß╗çn

**Lß╗ùi:** "GPU not detected, using CPU"

**Giß║úi ph├íp (Windows):**
```bash
# Option 1: D├╣ng WSL2 + CUDA
# C├ái WSL2, CUDA, cuDNN

# Option 2: TensorFlow-DirectML (Windows)
pip install tensorflow-directml
```

### Γ¥î Video upload lß╗ùi

**Kiß╗âm tra:**
- Γ£à Format: MP4, AVI, MOV (OpenCV support)
- Γ£à K├¡ch th╞░ß╗¢c: < 1GB
- Γ£à ─Éß╗Ö ph├ón giß║úi: Bß║Ñt kß╗│ (128x128 l├á optimal)
- Γ£à Codec: H264, MPEG-4

**Solution:**
```bash
# Convert video with FFmpeg
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
```

### Γ¥î Out of Memory

**Vß║Ñn ─æß╗ü:** Kh├┤ng ─æß╗º RAM

**Giß║úi ph├íp:**
```python
# app.py - Giß║úm batch size
batch_size = 1  # Tß╗½ 8 xuß╗æng 1
```

---

## ≡ƒôû T├ái Liß╗çu Bß╗ò Sung

| T├ái Liß╗çu | M├┤ Tß║ú | K├¡ch Th╞░ß╗¢c |
|----------|-------|-----------|
| [CODE_REVIEW.md](./docs/CODE_REVIEW.md) | ─É├ính gi├í code + best practices | 15+ pages |
| [PROJECT_STATUS.md](./docs/PROJECT_STATUS.md) | Status & roadmap | 12+ pages |
| [v8_chay.ipynb](./notebooks/v8_chay.ipynb) | Training notebook | ~50MB |
| [requirements.txt](./requirements.txt) | Dependencies | 20 lines |

---

## ≡ƒñ¥ C├ích ─É├│ng G├│p

Ch├║ng t├┤i hoan ngh├¬nh c├íc Pull Request!

### Quy Tr├¼nh ─É├│ng G├│p

1. **Fork** repository
   ```bash
   git clone https://github.com/yourname/violence-detection-cctv.git
   ```

2. **Tß║ío feature branch**
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

- Γ£à Follow **PEP 8** code style
- Γ£à Add **unit tests** for new features
- Γ£à Update **documentation**
- Γ£à Commit messages in **English**
- Γ£à One feature per PR

---

## ≡ƒô₧ Li├¬n Hß╗ç & Hß╗ù Trß╗ú

### C├íc C├ích Li├¬n Hß╗ç

| Ph╞░╞íng Thß╗⌐c | Chi Tiß║┐t | Thß╗¥i Gian |
|-----------|---------|----------|
| **Email** | le294594@gmail.com | 24-48h |
| **GitHub Issues** | Bug reports & features | 48-72h |
| **LinkedIn** | Direct message | 24-48h |
| **Discord** | Coming soon | Real-time |

### ╞»u Ti├¬n Hß╗ù Trß╗ú

- ≡ƒö┤ **Critical bugs** ΓÇö < 24h response
- ≡ƒƒí **Regular issues** ΓÇö 2-3 days
- ≡ƒƒó **Questions** ΓÇö Best effort

---


## ≡ƒÖÅ Lß╗¥i Cß║úm ╞án

Cß║úm ╞ín c├íc nh├│m v├á c├┤ng tr├¼nh ─æ├ú truyß╗ün cß║úm hß╗⌐ng:

- ≡ƒñû **TensorFlow/Keras team** ΓÇö Framework tuyß╗çt vß╗¥i
- ≡ƒôí **OpenCV contributors** ΓÇö Video processing tools
- ≡ƒôÜ **Kaggle community** ΓÇö Datasets & competitions
- ≡ƒîƒ **Papers cited:**
  - MobileNet (Howard et al., 2017)
  - LSTM (Hochreiter & Schmidhuber, 1997)
  - Attention Mechanism (Vaswani et al., 2017)

---

## ≡ƒôè Thß╗æng K├¬ Dß╗▒ ├ün

```
≡ƒôè PROJECT STATISTICS

Code Metrics:
Γö£ΓöÇΓöÇ Python Lines:           ~2,500+
Γö£ΓöÇΓöÇ HTML/CSS Lines:         ~1,500+
Γö£ΓöÇΓöÇ Total Files:            50+
Γö£ΓöÇΓöÇ Git Commits:            100+
ΓööΓöÇΓöÇ Contributors:           5+

Model Metrics:
Γö£ΓöÇΓöÇ Total Parameters:       4.2M
Γö£ΓöÇΓöÇ Base Model Size:        90MB
Γö£ΓöÇΓöÇ Inference Speed:        150ms
Γö£ΓöÇΓöÇ Memory Required:        500MB
ΓööΓöÇΓöÇ Accuracy:               90.77%

Dataset:
Γö£ΓöÇΓöÇ Training Videos:        4,800
Γö£ΓöÇΓöÇ Validation Videos:      600
Γö£ΓöÇΓöÇ Test Videos:            600
Γö£ΓöÇΓöÇ Total Dataset Size:     200GB+
ΓööΓöÇΓöÇ Classes:                2 (Normal/Violence)

Timeline:
Γö£ΓöÇΓöÇ Development Start:      2024-01-01
Γö£ΓöÇΓöÇ Initial Release:        2024-06-15
Γö£ΓöÇΓöÇ Current Version:        v2.0 (2026-04-28)
Γö£ΓöÇΓöÇ Active Days:            850+
ΓööΓöÇΓöÇ Last Update:            2026-04-28
```

---

<div align="center">

### ≡ƒîƒ Nß║┐u Bß║ín Th├¡ch Dß╗▒ ├ün N├áy, H├úy Cho Sao! Γ¡É

[![GitHub Repo stars](https://img.shields.io/github/stars/lehoang/violence-detection-cctv?style=social)](https://github.com/lehoang/violence-detection-cctv)

**Made with Γ¥ñ∩╕Å by [L├¬ Ho├áng](https://github.com/lehoang)**

---

### ≡ƒöù Li├¬n Kß║┐t Nhanh

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&style=flat-square)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-Latest-green?logo=flask&style=flat-square)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)](#)

---

### ≡ƒô¥ Citation

Nß║┐u bß║ín sß╗¡ dß╗Ñng dß╗▒ ├ín n├áy trong nghi├¬n cß╗⌐u, vui l├▓ng cite:

```bibtex
@software{lehoang2026violence,
  title = {Violence Detection CCTV: Deep Learning for Real-time Surveillance},
  author = {L├¬, Ho├áng},
  year = {2026},
  url = {https://github.com/lehoang/violence-detection-cctv},
  note = {GitHub Repository}
}
```

---

## ≡ƒô¥ Nhß║¡t K├╜ Cß║¡p Nhß║¡t (Update Logs)

### [Mß╗¢i Nhß║Ñt] - 2026-05-02
- **Γ£¿ Heatmap & Analytics:** Th├¬m trang `/dashboard` hiß╗ân thß╗ï biß╗âu ─æß╗ô nhiß╗çt 24h v├á ph├ón bß╗ò sß╗▒ cß╗æ theo tß╗½ng Camera.
- **≡ƒ¢í∩╕Å Privacy Mode:** T├¡ch hß╗úp t├¡nh n─âng l├ám mß╗¥ khu├┤n mß║╖t tß╗▒ ─æß╗Öng trong c├íc ─æoß║ín video tr├¡ch xuß║Ñt ─æß╗â bß║úo vß╗ç quyß╗ün ri├¬ng t╞░ (GDPR).
- **≡ƒô╣ Cß║úi tiß║┐n Real-time:** 
    - Chuyß╗ân sang c╞í chß║┐ l╞░u video bß║▒ng frame-buffer (MP4) thay v├¼ WebM ─æß╗â khß║»c phß╗Ñc lß╗ùi ─æß╗ïnh dß║íng tr├¬n Windows.
    - Tß╗▒ ─æß╗Öng quay l├╣i 3 gi├óy tr╞░ß╗¢c sß╗▒ cß╗æ v├á 2 gi├óy sau sß╗▒ cß╗æ.
    - Li├¬n kß║┐t video trß╗▒c tiß║┐p v├áo trang Dashboard ─æß╗â tß║úi vß╗ü sau.
- **≡ƒÉ│ Dockerization:** Th├¬m `Dockerfile` v├á `docker-compose.yml` ─æß╗â triß╗ân khai hß╗ç thß╗æng nhanh ch├│ng.
- **≡ƒæñ Project Update:** Cß║¡p nhß║¡t bß║ún quyß╗ün v├á th├┤ng tin dß╗▒ ├ín cß╗ºa **L├¬ Ho├áng**.

---
*Ph├ít triß╗ân bß╗ƒi: **L├¬ Ho├áng** | [LinkedIn](https://www.linkedin.com/in/visin-8386/)*

</div>
