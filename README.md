# Handwritten Arabic Numeral Recognition

## Overview

This project develops and evaluates a complete machine learning pipeline for classifying handwritten Arabic-Indic numerals (digits 0 through 9). It covers the full workflow from raw image loading and data splitting through classical baselines, a custom convolutional neural network with augmentation, model evaluation, and a live web interface for real-time prediction.

The dataset consists of 9,350 grayscale images at approximately 28x28 pixels, with exactly 935 images per class, sourced from Mendeley Data.

---

## Results

All four models were evaluated on the same held-out test set of 1,403 images drawn from a fixed stratified split (70 / 15 / 15).

| Model | Test Accuracy | Correct / Total |
|---|---|---|
| Raw Pixels + KNN | 68.64% | 963 / 1,403 |
| HOG + SVM | 90.59% | 1,272 / 1,403 |
| CNN (no augmentation) | 98.86% | 1,387 / 1,403 |
| CNN (with augmentation) | 98.86% | 1,387 / 1,403 |

The CNN with augmentation is the final deployed model. It achieves a macro-average F1-score of 0.9886, with perfect precision and recall on digits 4, 7, and 8. The most common confusions are between digit 0 and digit 5 (6 errors), which closely resemble each other in handwritten form.

### Per-class performance (CNN with augmentation)

| Digit | Precision | Recall | F1-score |
|---|---|---|---|
| 0 | 0.993 | 0.957 | 0.975 |
| 1 | 0.993 | 0.986 | 0.989 |
| 2 | 0.965 | 0.986 | 0.975 |
| 3 | 0.986 | 0.971 | 0.978 |
| 4 | 1.000 | 1.000 | 1.000 |
| 5 | 0.959 | 0.993 | 0.975 |
| 6 | 1.000 | 0.993 | 0.996 |
| 7 | 1.000 | 1.000 | 1.000 |
| 8 | 1.000 | 1.000 | 1.000 |
| 9 | 0.993 | 1.000 | 0.996 |

---

## Project Structure

```
Final Project/
├── Handwritten Arabic Numerals (0-9)/   # dataset root (not tracked in git)
│   └── ANGKA ARAB/
│       └── 0/ 1/ ... 9/                 # 935 images per digit
├── src/
│   ├── config.py          # all hyperparameters, paths, and constants
│   ├── utils.py           # seed control, device selection, timer
│   ├── data_pipeline.py   # image loading, stratified split, transforms, DataLoaders
│   ├── models.py          # CNN architecture definition
│   ├── baselines.py       # HOG+SVM and KNN baseline trainers
│   ├── train.py           # training loop with early stopping and LR scheduling
│   ├── evaluate.py        # evaluation metrics, confusion matrices, plots
│   ├── main.py            # full pipeline entry point
│   └── api.py             # FastAPI backend for web inference
├── ui/
│   └── index.html         # browser-based canvas drawing interface
├── outputs/               # generated automatically at runtime
│   ├── models/            # saved model weights (.pth)
│   ├── plots/             # all figures (confusion matrices, learning curves, etc.)
│   └── splits/            # split_indices.json for reproducibility
├── DOCUMENTATION.md       # detailed technical documentation
├── report.pdf             # project report
├── requirements.txt
└── README.md
```

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
cd "Final Project"
python src/main.py
```

This trains all four models end-to-end, saves weights, generates all evaluation plots, and writes a final summary JSON to `outputs/`.

### Optional flags

```bash
python src/main.py --skip-baselines   # train CNNs only, skip HOG+SVM and KNN
python src/main.py --cnn-only         # train the augmented CNN only
python src/main.py --no-aug           # run the no-augmentation ablation only
```

### 3. Start the web interface

```bash
cd "Final Project"
uvicorn src.api:app --reload
```

Then open `ui/index.html` in a browser. Draw a digit on the canvas and submit to receive a live prediction with confidence scores.

---

## Model Architecture

The CNN uses three convolutional blocks followed by global average pooling and a fully connected classifier:

- **Block 1:** Conv2d(1, 32) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- **Block 2:** Conv2d(32, 64) + BatchNorm + ReLU + MaxPool + Dropout(0.25)
- **Block 3:** Conv2d(64, 128) + BatchNorm + ReLU + GlobalAvgPool
- **Classifier:** Linear(128, 10) with Dropout(0.5)

Total parameters: 157,290 (all trainable).

---

## Training Configuration

| Parameter | Value |
|---|---|
| Data split | 70 / 15 / 15 stratified by class |
| Optimizer | Adam, lr = 1e-3, weight decay = 1e-4 |
| Loss function | Cross-entropy with label smoothing (0.1) |
| Batch size | 64 |
| Max epochs | 60 |
| Early stopping patience | 10 epochs |
| LR scheduler | ReduceLROnPlateau, factor = 0.5, patience = 5 |
| Augmentation | Random affine: rotation +/-12 deg, translation +/-8%, zoom 0.92-1.08 |
| Random seed | 42 (fixed across Python, NumPy, PyTorch) |

---

## Reproducibility

The train/val/test split is saved to `outputs/splits/split_indices.json` after the first run. Normalization statistics (mean = 0.9811, std = 0.0763) are computed only from the training set and saved to `outputs/norm_stats.json`. Re-running the pipeline with the same seed reproduces the exact same split and results.

---

## Dataset

Handwritten Arabic Numerals (0-9), Mendeley Data.
9,350 grayscale images, 10 classes, 935 images per class, approximately 28x28 pixels.
License: CC BY-NC 3.0.

---

## Author

Mohammad — 2026
