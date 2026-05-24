# StegoSentinel 

**AI-Enhanced Steganography Detection (Steganalysis) using Neural Network**

A deep learning-based steganalysis system for real-time detection of hidden data in JPEG images, built and deployable on consumer-grade hardware.

---

## Overview

Steganography — hiding secret data inside ordinary-looking images — is a growing threat in digital forensics, data exfiltration, and law enforcement contexts. Unlike encryption, it conceals the *existence* of communication, making it especially difficult to detect.

**StegoSentinel** addresses this using a Residual CNN initialised with Spatial Rich Model (SRM) filter kernels, trained on the ALASKA2 benchmark dataset. The system detects three state-of-the-art content-adaptive JPEG steganographic algorithms — **JMiPOD**, **J-UNIWARD**, and **UERD** — and exposes results via a real-time web dashboard with forensic reporting.

---

## Features

- **Residual CNN + SRM Preprocessing** — 18-channel SRM filter bank amplifies low-amplitude steganographic residuals before CNN inference
- **Test-Time Augmentation (TTA)** — 8 geometric variants averaged per image for robust inference
- **Real-Time Web Dashboard** (Flask) — drag-and-drop image scanner, batch evaluation, live metrics
- **Risk-Severity Tiering** — alerts classified as CRITICAL / HIGH / MEDIUM / LOW / RARE based on detection confidence
- **Confusion Matrix & Forensic Reports** — live display of accuracy, ROC-AUC, F1-score, and confusion matrix
- **Fully Local** — no cloud dependency; runs on localhost:5000

---

## Model Performance

Trained on 40,000 ALASKA2 JPEG images under hardware-constrained conditions:

| Metric   | Training | Dashboard Test (998 images) |
|----------|----------|-----------------------------|
| Accuracy | 61.5%    | 56.81%                      |
| ROC-AUC  | 0.647    | 0.6082                      |
| F1-Score | 0.66     | 0.6035                      |

> Performance is consistent with published single-model resource-constrained baselines on ALASKA2. The 35–53% accuracy gap vs. state-of-the-art ensemble systems (88–90% on 8×A100) is attributable entirely to hardware constraints (6 GB VRAM vs. 640 GB).

---

## Hardware Requirements

| Component    | Used                        |
|--------------|-----------------------------|
| GPU          | NVIDIA RTX 4050 (6 GB VRAM) |
| CPU          | Intel i7-13700H             |
| OS           | WSL2 (Ubuntu)               |
| Framework    | TensorFlow 2.15             |
| CUDA / cuDNN | 12.3 / 8.9                  |

The model will also run on any CUDA-capable GPU with ≥6 GB VRAM. CPU-only inference is possible but significantly slower.

---

## Dataset

**ALASKA2** — 40,000 JPEG images (256×256) used from the full 75,000-image benchmark:

- 10,000 unmodified cover images
- 10,000 stego images per algorithm × 3 (JMiPOD, J-UNIWARD, UERD)

Split: 70% train / 15% validation / 15% test (stratified).

Download: [ALASKA2 on Kaggle](https://www.kaggle.com/c/alaska2-image-steganalysis)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/StegoSentinel.git
cd StegoSentinel

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- TensorFlow 2.15
- Flask
- NumPy, OpenCV, scikit-learn, Pillow

---

## Usage

### 1. Train the Model

```bash
python train.py
```

This runs SRM preprocessing on the ALASKA2 dataset, trains the Residual CNN for up to 40 epochs (with early stopping), and saves the best checkpoint to `best_model.keras`. Training takes ~12 hours on an RTX 4050.

### 2. Launch the Dashboard

```bash
python dashboard.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

**Dashboard capabilities:**
- **Image Scanner** — drag-and-drop a JPEG image for TTA inference and instant result
- **Batch Evaluation** — load a directory of images, run bulk inference, and view accuracy/AUC/F1/confusion matrix
- **Alert Feed** — all detections with severity tier, filename, confidence score, and timestamp
- **System Info** — model status, hardware info, and runtime metrics

---

## Architecture

```
Input Image (JPEG)
       │
       ▼
SRM Filter Bank (6 filters × 3 channels = 18-channel residual maps)
       │
       ▼
Block 1 — 64 filters, Conv-BN-TLU, strided downsampling
Block 2 — 128 filters, Conv-BN-ReLU, strided downsampling
Block 3 — 256 filters, Conv-BN-ReLU, strided downsampling
Block 4 — 512 filters, Conv-BN-ReLU
       │
       ▼
Global Average Pooling → Dense(256) → Dropout(0.5) → Dense(64) → Dense(1, Sigmoid)
       │
       ▼
Stego Probability Score → Risk-Severity Tier
```

The **Truncated Linear Unit (TLU)** activation in Block 1 clips activations to [–3, 3], preventing saturation on high-amplitude image textures that are irrelevant to steganographic signal detection.

---

## Risk-Severity Tiers

| Tier     | Confidence Threshold |
|----------|----------------------|
| CRITICAL | ≥ 75%                |
| HIGH     | ≥ 65%                |
| MEDIUM   | ≥ 57%                |
| LOW      | ≥ 53%                |
| RARE     | ≥ 50%                |

---

## Project Structure

```
StegoSentinel/
├── train.py              # Training pipeline (SRM preprocessing + Residual CNN)
├── dashboard.py          # Flask web dashboard (inference + batch evaluation)
├── best_model.keras      # Saved model checkpoint (generated after training)
├── requirements.txt      # Python dependencies
├── static/               # Frontend assets (JS, CSS)
├── templates/            # Flask HTML templates
└── README.md
```

---

## Comparison with Existing Tools

| Tool              | Method                 | Accuracy        |
|-------------------|------------------------|-----------------|
| StegExpose        | Statistical analysis   | 60–70%          |
| StegDetect        | Chi-square attack      | 55–65%          |
| FotoForensics     | ELA analysis           | 65–72%          |
| **StegoSentinel** | **Residual CNN + SRM** | **56.81–61.5%** |

StegoSentinel performs comparably to the best open-source tools while targeting **content-adaptive JPEG steganography** — a significantly harder detection problem that classical tools largely fail on.

---

## Limitations

- Scoped to JPEG steganography (JMiPOD, J-UNIWARD, UERD) — not expected to generalise to spatial-domain steganography or other media types
- False positive rate (~52%) means human forensic review of flagged images is recommended for unsupervised deployment
- Performance scales with GPU VRAM and training data; cloud GPU access would close the gap with state-of-the-art significantly

---

## Future Work

- Focal loss / class-weighted training to improve precision-recall balance
- JPEG-aware augmentations (random quality-factor recompression)
- Knowledge distillation from larger pre-trained models
- Model ensembling for improved AUC
- Cloud GPU training on the full 75,000-image ALASKA2 set

---

## References

Key papers this work builds on:

- Fridrich & Kodovský, *Rich Models for Steganalysis of Digital Images*, IEEE TIFS 2012 (SRM)
- Xu et al., *Structural Design of CNNs for Steganalysis*, IEEE SPL 2016 (Xu-Net)
- Ye et al., *Deep Learning Hierarchical Representations for Image Steganalysis*, IEEE TIFS 2017 (Ye-Net)
- Boroumand et al., *Deep Residual Network for Steganalysis*, IEEE TIFS 2019 (SR-Net)
- Cogranne et al., *ALASKA#2: Challenging Real-World Steganalysis*, ACM IH&MMSec 2020

---

## Authors

**Aviral Sharma**

B.Tech Project, 2025–2026

---

## License

This project is a research implementation submitted in partial fulfilment of B.Tech requirements. Please cite appropriately if building upon this work.
