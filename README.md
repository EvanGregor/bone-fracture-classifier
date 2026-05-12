---
title: Bone Fracture Classifier
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - medical-imaging
  - bone-fracture
  - efficientnet
  - yolov8
  - ensemble
  - grad-cam
  - x-ray
---

# 🦴 Bone Fracture Classifier

An ensemble deep-learning system for automated bone fracture detection in X-ray images.
It combines **EfficientNet** (classification) and **YOLOv8** (detection) with **Grad-CAM** explainability.

> ⚠️ **Disclaimer:** For research and educational use only. Not a substitute for clinical diagnosis.

---

## 🏗️ Architecture

```
X-ray Input
     │
     ├──► CLAHE Enhancement
     │
     ├──► EfficientNet-B3  ──► Fracture probability  ─┐
     │         └──► Grad-CAM heatmap                   ├──► Weighted Ensemble ──► Final Decision
     └──► YOLOv8           ──► Region detections      ─┘
```

| Component     | Detail                              |
|---------------|-------------------------------------|
| Classifier    | EfficientNet-B3 (timm)              |
| Detector      | YOLOv8 (Ultralytics)                |
| Ensemble      | Weighted score fusion (0.45 / 0.55) |
| Explainability| Grad-CAM on EfficientNet            |
| Preprocessing | CLAHE + border crop                 |

---

## 📊 Datasets

| Dataset       | Usage                                      |
|---------------|--------------------------------------------|
| [FracAtlas](https://github.com/saddam-hussain-lio/FracAtlas) | Fractured X-ray images (train/val/test) |
| [MURA v1.1](https://stanfordmlgroup.github.io/competitions/mura/) | Normal musculoskeletal X-rays |
| [GRAZPEDWRI-DX](https://www.kaggle.com/datasets/jasonroggy/grazpedwri-dx) | Paediatric wrist X-rays (YOLO training) |

---

## 🚀 Running Locally

```bash
# 1. Clone
git clone https://github.com/<your-username>/bone-fracture-classifier
cd bone-fracture-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add model weights
#    Place best_effnet_v4.pth and best.pt inside models/

# 4. Launch
python app.py
```

---

## 📁 Repository Structure

```
bone-fracture-classifier/
├── app.py                    # Gradio application (HF Space entry point)
├── requirements.txt
├── models/
│   ├── best_effnet_v4.pth    # EfficientNet checkpoint (not tracked in git)
│   └── best.pt               # YOLOv8 checkpoint     (not tracked in git)
├── examples/
│   ├── sample_fracture.jpg
│   └── sample_normal.jpg
├── notebooks/
│   ├── MuraFractatlas.ipynb  # EfficientNet training
│   ├── YOLOTraining.ipynb    # YOLOv8 training
│   ├── Ensemble.ipynb        # Ensemble inference
│   ├── EnsembleEval.ipynb    # Evaluation metrics
│   └── GradioDemo.ipynb      # Gradio prototype
└── README.md
```

---

## 📈 Performance

| Metric          | EfficientNet | YOLOv8 | Ensemble |
|-----------------|-------------|--------|----------|
| AUC-ROC         | ~0.92       | —      | **~0.94**|
| Threshold       | 0.6623      | 0.15   | 0.4355   |
| Weight          | 0.45        | 0.55   | —        |

*Evaluated on a balanced held-out set drawn from FracAtlas + MURA.*

---

## 🛠️ Configuration

Key parameters in `app.py` (also in `ENSEMBLE_CONFIG`):

```python
ENSEMBLE_CONFIG = {
    "effnet_threshold" : 0.6623,   # Youden-optimal threshold for EfficientNet
    "yolo_conf"        : 0.15,     # YOLO confidence threshold
    "yolo_iou"         : 0.45,     # YOLO NMS IoU threshold
    "effnet_weight"    : 0.45,     # Ensemble weight for EfficientNet score
    "yolo_weight"      : 0.55,     # Ensemble weight for YOLO score
    "final_threshold"  : 0.4355,   # Decision threshold for ensemble score
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

## ✍️ Author

Built as a bone fracture classification research project using FracAtlas, MURA, and GRAZPEDWRI-DX datasets.
