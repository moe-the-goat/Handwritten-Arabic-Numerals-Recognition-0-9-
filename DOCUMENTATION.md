# Handwritten Arabic Numeral Recognition — Project Documentation

> Complete technical documentation covering dataset analysis, feature engineering, model architecture, training procedures, evaluation methodology, and deployment via a real-time web interface.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Project Structure](#3-project-structure)
4. [Environment & Dependencies](#4-environment--dependencies)
5. [Data Pipeline](#5-data-pipeline)
   - 5.1 [Image Loading & Validation](#51-image-loading--validation)
   - 5.2 [Stratified Train/Val/Test Split](#52-stratified-trainvaltest-split)
   - 5.3 [Normalization Statistics](#53-normalization-statistics)
   - 5.4 [Data Augmentation](#54-data-augmentation)
6. [Baseline Models](#6-baseline-models)
   - 6.1 [HOG + SVM](#61-hog--svm)
   - 6.2 [Raw Pixels + KNN](#62-raw-pixels--knn)
7. [CNN Architecture](#7-cnn-architecture)
   - 7.1 [Network Design](#71-network-design)
   - 7.2 [Design Decisions](#72-design-decisions)
8. [Training Procedure](#8-training-procedure)
   - 8.1 [Optimizer & Scheduler](#81-optimizer--scheduler)
   - 8.2 [Early Stopping](#82-early-stopping)
   - 8.3 [Training with Augmentation](#83-training-with-augmentation)
   - 8.4 [Training without Augmentation (Ablation)](#84-training-without-augmentation-ablation)
9. [Evaluation & Results](#9-evaluation--results)
   - 9.1 [Model Comparison](#91-model-comparison)
   - 9.2 [CNN (aug) Detailed Results](#92-cnn-aug-detailed-results)
   - 9.3 [Confusion Matrix Analysis](#93-confusion-matrix-analysis)
   - 9.4 [Per-Class Precision, Recall & F1](#94-per-class-precision-recall--f1)
   - 9.5 [Learning Curves](#95-learning-curves)
   - 9.6 [Misclassified Sample Analysis](#96-misclassified-sample-analysis)
   - 9.7 [Augmentation Effect](#97-augmentation-effect)
10. [FastAPI Web Interface](#10-fastapi-web-interface)
    - 10.1 [Backend API](#101-backend-api)
    - 10.2 [Canvas Preprocessing Pipeline](#102-canvas-preprocessing-pipeline)
    - 10.3 [Frontend UI](#103-frontend-ui)
11. [Generated Outputs](#11-generated-outputs)
12. [How to Run](#12-how-to-run)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Project Overview

This project implements a complete pipeline for recognizing handwritten Arabic-Indic numerals (٠–٩, digits 0–9) from grayscale images. The pipeline covers every stage from raw data loading to real-time browser-based prediction:

1. **Data loading & stratified splitting** — 9,350 images split 70/15/15
2. **Classical baselines** — HOG+SVM and KNN to establish reference performance
3. **Deep learning** — A 3-block CNN trained with and without data augmentation
4. **Comprehensive evaluation** — Confusion matrices, classification reports, learning curves, misclassified image galleries, and model comparison charts
5. **Augmentation verification** — Statistical and visual proof that augmentation produces meaningful variation
6. **Real-time web demo** — FastAPI backend + HTML5 canvas frontend for live digit prediction

**Final result**: The CNN achieves **98.86% test accuracy** (macro F1 = 0.9886), with digits 4, 7, and 8 reaching perfect 1.000 F1-scores.

---

## 2. Dataset Description

| Property | Value |
|---|---|
| Name | Handwritten Arabic Numerals (0–9) |
| Total images | 9,350 |
| Classes | 10 (digits 0–9) |
| Images per class | 935 (perfectly balanced) |
| Resolution | ~28×28 pixels (grayscale) |
| Format | PNG |
| Background | White (~255) |
| Ink | Thin gray strokes (pixel intensity ~100–160) |

The dataset is stored under `Handwritten Arabic Numerals (0-9)/ANGKA ARAB/{0..9}/`, with each digit class in its own folder. Each image depicts a handwritten Arabic-Indic numeral on a white background with relatively thin, gray pen strokes.

**Key observations about the data:**
- The images have thin strokes with grayish ink, not bold black marks. This is critical for preprocessing in the web interface.
- The ink pixels occupy only about 0.4–4.5% of the total image area.
- The background is nearly pure white, leading to a high mean pixel value (0.981 after normalization to [0,1]).
- All classes are perfectly balanced (935 samples each), so no class weighting is needed.

---

## 3. Project Structure

```
Final Project/
├── src/
│   ├── config.py                  # All hyperparameters and paths
│   ├── utils.py                   # Seed, device, Timer
│   ├── data_pipeline.py           # Loading, splitting, transforms, DataLoaders
│   ├── models.py                  # CNN architecture
│   ├── train.py                   # Training loop with scheduler & early stopping
│   ├── baselines.py               # HOG+SVM and KNN
│   ├── evaluate.py                # Metrics, plots, reporting
│   ├── main.py                    # Full pipeline orchestrator
│   ├── api.py                     # FastAPI backend
│   └── verify_augmentation.py     # Augmentation diagnostic
├── ui/
│   └── index.html                 # Web frontend (canvas + results)
├── outputs/
│   ├── norm_stats.json            # {mean, std}
│   ├── baseline_hog_svm.json      # HOG+SVM results
│   ├── baseline_knn.json          # KNN results
│   ├── cnn_aug_history.json       # Epoch-by-epoch training history (aug)
│   ├── cnn_noaug_history.json     # Epoch-by-epoch training history (no aug)
│   ├── final_summary.json         # All final metrics
│   ├── models/
│   │   ├── cnn_aug_best.pth       # Best CNN weights (with augmentation)
│   │   └── cnn_noaug_best.pth     # Best CNN weights (without augmentation)
│   ├── plots/
│   │   ├── sample_images.png      # Grid of dataset samples
│   │   ├── cm_hog_svm.png         # HOG+SVM confusion matrix
│   │   ├── cm_knn.png             # KNN confusion matrix
│   │   ├── cm_cnn_aug.png         # CNN (aug) confusion matrix
│   │   ├── cm_cnn_noaug.png       # CNN (no aug) confusion matrix
│   │   ├── learning_curves_cnn_aug.png
│   │   ├── learning_curves_cnn_noaug.png
│   │   ├── misclassified_cnn_aug.png
│   │   ├── augmentation_verification.png
│   │   └── model_comparison.png
│   └── splits/
│       └── split_indices.json     # Reproducible split indices
├── Handwritten Arabic Numerals (0-9)/
│   └── ANGKA ARAB/
│       ├── 0/ ... 9/              # Raw dataset images
├── requirements.txt
├── README.md
└── DOCUMENTATION.md               # This file
```

---

## 4. Environment & Dependencies

| Component | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.10.0+cpu |
| torchvision | 0.15+ |
| scikit-learn | 1.2+ |
| scikit-image | 0.20+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| FastAPI | 0.135.1 |
| uvicorn | 0.41.0 |
| Pillow | 9.5+ |
| NumPy | 1.24+ |

Install all dependencies:

```bash
pip install -r requirements.txt
```

For the web demo, also install:

```bash
pip install fastapi uvicorn python-multipart
```

---

## 5. Data Pipeline

The data pipeline (implemented in `src/data_pipeline.py`) handles everything from raw image files to PyTorch DataLoaders.

### 5.1 Image Loading & Validation

`load_dataset()` scans the dataset directory and returns parallel lists of file paths and integer labels (0–9). It validates that every file is a readable image and prints a per-class count summary. The function raises an error if any class has zero images.

`load_images_as_arrays()` reads image files into NumPy arrays:
1. Opens each PNG with Pillow in grayscale mode (`"L"`)
2. Resizes to 28×28 if necessary (using Lanczos resampling)
3. Returns a list of `(28, 28)` uint8 arrays

### 5.2 Stratified Train/Val/Test Split

`stratified_split()` uses scikit-learn's `train_test_split` with `stratify=labels` to ensure each class has proportional representation in every split:

| Split | Ratio | Samples | Per class |
|---|---|---|---|
| Train | 70% | 6,545 | ~654–655 |
| Validation | 15% | 1,402 | ~140–141 |
| Test | 15% | 1,403 | ~140–141 |

The split indices are saved to `outputs/splits/split_indices.json` for reproducibility. The random seed is fixed at **42**.

### 5.3 Normalization Statistics

`compute_norm_stats()` computes the channel-wise mean and standard deviation from the **training set only** (never from validation or test data, to prevent data leakage):

| Statistic | Value |
|---|---|
| Mean | 0.9811 |
| Std | 0.0763 |

The high mean (≈0.98) reflects that the images are mostly white background. Both train and eval transforms normalize using these values: `(pixel - mean) / std`.

### 5.4 Data Augmentation

Data augmentation is applied **only to the training set** during CNN training. The augmentation chain uses `torchvision.transforms`:

| Transform | Parameters | Purpose |
|---|---|---|
| RandomAffine rotation | ±12° | Simulate natural hand rotation |
| RandomAffine translate | (0.08, 0.08) | Handle positional variation |
| RandomAffine scale | (0.92, 1.08) | Handle size variation |
| Fill mode | 0 (black) | Fill exposed edges |

After augmentation, images are converted to tensors and normalized.

**Verification**: The `verify_augmentation.py` script confirms augmentation is working by:
1. Generating 8 augmented versions of the same image and plotting them side-by-side
2. Computing pixel-level difference maps between augmented and original
3. Running 100 augmented copies and measuring mean pixel difference (confirmed > 0.001)
4. Comparing batch outputs from augmented vs. non-augmented DataLoaders

The verification plot is saved as `augmentation_verification.png`.

---

## 6. Baseline Models

Two classical baselines establish reference performance before moving to deep learning.

### 6.1 HOG + SVM

**Feature extraction**: Histogram of Oriented Gradients (HOG) computes gradient orientation histograms in local cells, capturing edge and stroke structure.

| HOG Parameter | Value |
|---|---|
| Orientations | 9 |
| Pixels per cell | (4, 4) |
| Cells per block | (2, 2) |
| Block normalization | L2-Hys |

This produces a 576-dimensional feature vector per 28×28 image.

**Classifier**: Linear SVM with `C = 10.0`, using scikit-learn's `LinearSVC` with squared hinge loss.

**Results**:

| Metric | Value |
|---|---|
| **Test accuracy** | **90.59%** |
| Training time | ~52 seconds |
| Macro precision | 0.9078 |
| Macro recall | 0.9060 |
| Macro F1 | 0.9065 |

**Strengths**: Good overall accuracy for a classical method; fast inference. Digits 8 (F1=0.968) and 4 (F1=0.953) are well-recognized.

**Weaknesses**: Struggles with digit 2 (F1=0.781) and digit 0 (F1=0.851), where stroke-level gradients are ambiguous.

### 6.2 Raw Pixels + KNN

Uses raw flattened pixel values (784 dimensions for 28×28) as features with k-Nearest Neighbors classification.

| Parameter | Value |
|---|---|
| k | 5 |
| Distance metric | Minkowski (p=2, Euclidean) |

**Results**:

| Metric | Value |
|---|---|
| **Test accuracy** | **68.64%** |
| Training time | ~0.006 seconds |
| Macro precision | 0.7797 |
| Macro recall | 0.6866 |
| Macro F1 | 0.7024 |

**Why it's weak**: Raw pixels are highly sensitive to translation, rotation, and stroke thickness. A slight shift of a digit makes the pixel vectors very different. Digit 1 achieves only 0.510 F1-score despite having high recall (0.929) because it gets confused with many other digits (low precision of 0.351).

**Takeaway**: The 22-point gap between KNN (68.6%) and HOG+SVM (90.6%) demonstrates the value of engineered features (HOG) over raw pixels. The further 8-point gap to CNN (98.9%) demonstrates the power of learned features.

---

## 7. CNN Architecture

### 7.1 Network Design

The `ArabicDigitCNN` is a custom 3-block convolutional network with **157,290 trainable parameters**.

```
Input: 1×28×28 (grayscale)

Block 1:
  Conv2d(1 → 32, 3×3, padding=1) → BatchNorm2d → ReLU
  Conv2d(32 → 32, 3×3, padding=1) → BatchNorm2d → ReLU
  MaxPool2d(2×2)
  Dropout2d(0.25)
  → Output: 32×14×14

Block 2:
  Conv2d(32 → 64, 3×3, padding=1) → BatchNorm2d → ReLU
  Conv2d(64 → 64, 3×3, padding=1) → BatchNorm2d → ReLU
  MaxPool2d(2×2)
  Dropout2d(0.25)
  → Output: 64×7×7

Block 3:
  Conv2d(64 → 128, 3×3, padding=1) → BatchNorm2d → ReLU
  AdaptiveAvgPool2d(1×1)
  → Output: 128×1×1

Classification Head:
  Flatten → Linear(128 → 128) → ReLU → Dropout(0.5) → Linear(128 → 10)
  → Output: 10 class logits
```

### 7.2 Design Decisions

1. **Double convolutions per block**: Two Conv2d layers before each pooling step doubles the effective receptive field within each spatial resolution level, allowing the network to learn more complex local patterns.

2. **BatchNorm after every convolution**: Stabilizes training by normalizing intermediate activations, allows higher learning rates, and adds mild regularization.

3. **Global Average Pooling (GAP) in Block 3**: Instead of a large fully-connected layer from a flattened feature map (which would be 128×7×7 = 6,272 dimensions), GAP reduces each channel to a single value. This dramatically reduces parameters and forces each filter to represent a meaningful spatial summary.

4. **Progressive channel widening** (32 → 64 → 128): Deeper layers capture more abstract features and need more channels to represent them. The spatial dimensions shrink (28→14→7→1) while channels grow.

5. **Dropout at two levels**: Dropout2d(0.25) on feature maps after pooling (spatial dropout) and Dropout(0.5) in the FC head prevent co-adaptation of features and reduce overfitting.

6. **Small model size** (~157K params): Appropriate for a relatively small dataset (6,545 training images). A much larger model would overfit; a smaller one would underfit.

---

## 8. Training Procedure

### 8.1 Optimizer & Scheduler

| Component | Configuration |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Loss function | CrossEntropyLoss |
| LR scheduler | ReduceLROnPlateau(patience=5, factor=0.5) |
| Metric monitored | Validation loss |

Adam's adaptive per-parameter learning rates are well-suited for the varying gradient scales across conv and FC layers. The L2 weight decay (1e-4) provides additional regularization.

The scheduler halves the learning rate when validation loss stops improving for 5 consecutive epochs, allowing finer convergence.

### 8.2 Early Stopping

Training stops if validation loss doesn't improve for 10 consecutive epochs. The best model checkpoint (by validation loss) is saved to `outputs/models/cnn_*_best.pth`.

### 8.3 Training with Augmentation

| Detail | Value |
|---|---|
| Epochs (max) | 60 |
| Epochs run | 60 (no early stop) |
| Best epoch | 57 |
| Best val accuracy | 99.00% |
| Final train accuracy | 97.56% |
| Final val accuracy | 98.72% |
| **Test accuracy** | **98.86%** |

Training observations:
- The model converged to 95%+ validation accuracy by epoch 10.
- Train accuracy remains lower than validation accuracy (97.56% vs 98.72%) — a sign that augmentation is making the training task harder (good for generalization).
- No early stopping was triggered, meaning the model continued to find marginal improvements through all 60 epochs.
- Learning rate was reduced multiple times by the scheduler during later epochs.

### 8.4 Training without Augmentation (Ablation)

| Detail | Value |
|---|---|
| Epochs (max) | 60 |
| Epochs run | 41 (early stopped) |
| Best epoch | 39 |
| Best val accuracy | 99.07% |
| Final train accuracy | 99.42% |
| Final val accuracy | 98.93% |
| **Test accuracy** | **98.86%** |

Without augmentation:
- Training was faster to converge (fewer epochs needed).
- Train accuracy reached 99.42% — much higher than the augmented model's 97.56%, showing the training set is easier without augmentation.
- Early stopping triggered at epoch 41, indicating the model started overfitting.
- Despite higher train accuracy, the test accuracy is identical (98.86%), suggesting the **augmented model generalizes just as well while being more robust**.

---

## 9. Evaluation & Results

### 9.1 Model Comparison

| Model | Test Accuracy | Macro F1 | Training Time |
|---|---|---|---|
| Raw + KNN (k=5) | 68.64% | 0.7024 | 0.006 s |
| HOG + SVM (C=10) | 90.59% | 0.9065 | 52.0 s |
| CNN (no aug) | 98.86% | — | 41 epochs |
| **CNN (aug)** | **98.86%** | **0.9886** | 60 epochs |

The model comparison chart is saved as `model_comparison.png`.

### 9.2 CNN (aug) Detailed Results

The primary model achieves **98.86% accuracy** on the 1,403-sample test set, misclassifying only **16 images** out of 1,403.

### 9.3 Confusion Matrix Analysis

The confusion matrix for CNN (aug) on the test set:

|  | Pred 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **True 0** | **134** | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| **True 1** | 0 | **138** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **True 2** | 0 | 0 | **139** | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| **True 3** | 0 | 0 | 4 | **136** | 0 | 0 | 0 | 0 | 0 | 0 |
| **True 4** | 0 | 0 | 0 | 0 | **141** | 0 | 0 | 0 | 0 | 0 |
| **True 5** | 1 | 0 | 0 | 0 | 0 | **139** | 0 | 0 | 0 | 0 |
| **True 6** | 0 | 1 | 0 | 0 | 0 | 0 | **139** | 0 | 0 | 0 |
| **True 7** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **140** | 0 | 0 |
| **True 8** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **140** | 0 |
| **True 9** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **141** |

**Error patterns identified:**
- **0 → 5** (6 errors): The most common confusion. Arabic digits 0 (٠, a dot) and 5 (٥) share similar round shapes.
- **3 → 2** (4 errors): Digits 3 and 2 have similar curved strokes in Arabic script.
- **2 → 3** (2 errors): The reverse of the above confusion.
- **1 → 2, 1 → 9, 5 → 0, 6 → 1**: One error each — isolated edge cases.

**Perfect classes** (zero errors): Digits **4, 7, 8, 9** have zero misclassifications on the test set.

### 9.4 Per-Class Precision, Recall & F1

| Digit | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 | 0.9926 | 0.9571 | 0.9745 | 140 |
| 1 | 0.9928 | 0.9857 | 0.9892 | 140 |
| 2 | 0.9653 | 0.9858 | 0.9754 | 141 |
| 3 | 0.9855 | 0.9714 | 0.9784 | 140 |
| 4 | **1.0000** | **1.0000** | **1.0000** | 141 |
| 5 | 0.9586 | 0.9929 | 0.9754 | 140 |
| 6 | 1.0000 | 0.9929 | 0.9964 | 140 |
| 7 | **1.0000** | **1.0000** | **1.0000** | 140 |
| 8 | **1.0000** | **1.0000** | **1.0000** | 140 |
| 9 | 0.9930 | 1.0000 | 0.9965 | 141 |

| Average | Precision | Recall | F1 |
|---|---|---|---|
| **Macro** | 0.9888 | 0.9886 | 0.9886 |
| **Weighted** | 0.9888 | 0.9886 | 0.9886 |

**Observations:**
- Three digit classes (4, 7, 8) achieve **perfect scores** across all metrics.
- The weakest recall is digit 0 (0.957) due to confusion with digit 5.
- The weakest precision is digit 5 (0.959) because it absorbs false positives from digit 0.
- All digits exceed 0.95 F1-score — uniformly strong performance.

### 9.5 Learning Curves

Learning curves plot training and validation loss/accuracy across epochs:

**CNN with augmentation** (`learning_curves_cnn_aug.png`):
- Smooth convergence over 60 epochs
- Train accuracy steadily climbs to 97.56%
- Validation accuracy plateaus near 98.7–99.0%
- The gap between train and val accuracy reflects the augmentation penalty (augmented training samples are harder)
- No sign of overfitting — validation curves remain stable

**CNN without augmentation** (`learning_curves_cnn_noaug.png`):
- Faster initial convergence
- Train accuracy rises sharply to 99.42%
- Growing gap between train and val accuracy suggests early overfitting
- Early stopping triggers at epoch 41

### 9.6 Misclassified Sample Analysis

The gallery `misclassified_cnn_aug.png` displays up to 16 misclassified test images with their true and predicted labels. These are the model's hardest cases:

- Most errors involve genuinely ambiguous handwriting where the digit is poorly formed
- The 0→5 confusion cases typically show round/circular shapes that resemble either digit
- The 2↔3 confusions come from samples where the curve direction is ambiguous

### 9.7 Augmentation Effect

Both the augmented and non-augmented CNN achieve **identical test accuracy (98.86%)**, but the training dynamics differ significantly:

| Aspect | With Augmentation | Without Augmentation |
|---|---|---|
| Epochs to converge | 60 (no early stop) | 41 (early stopped) |
| Final train accuracy | 97.56% | 99.42% |
| Final val accuracy | 98.72% | 98.93% |
| Test accuracy | 98.86% | 98.86% |
| Risk of overfitting | Low | Higher |

**Interpretation**: Augmentation prevents the model from memorizing the training set (lower train accuracy) while maintaining equivalent test performance. For a larger or noisier dataset, the augmented model would likely generalize better. The augmented model is the preferred choice because it demonstrates more robust learning behavior.

---

## 10. FastAPI Web Interface

A real-time web demo allows users to draw digits on an HTML5 canvas and see CNN predictions instantly.

### 10.1 Backend API

Implemented in `src/api.py` using FastAPI:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves the HTML frontend |
| `/predict` | POST | Accepts a base64-encoded canvas image, returns prediction |
| `/health` | GET | Returns model status and device info |
| `/samples` | GET | Returns one base64-encoded training sample per class |

On startup, the API:
1. Loads `norm_stats.json` for normalization parameters
2. Loads `cnn_aug_best.pth` model weights
3. Sets the model to evaluation mode

### 10.2 Canvas Preprocessing Pipeline

The preprocessing pipeline in `/predict` bridges the gap between mouse-drawn canvas images and the training data distribution. This is critical because:
- Training images have **thin gray strokes** (pixel intensity ~100–160)
- Canvas drawings produce **thick black strokes** by default

The preprocessing steps:
1. **Decode** base64 PNG → PIL image → grayscale
2. **Bounding-box crop**: Find the drawn content, compute tight bounding box with 15% padding
3. **Resize** the cropped region to 20×20 (preserving aspect ratio)
4. **Center** the 20×20 content in a 28×28 white canvas
5. **Conditional Gaussian blur** (σ=0.5) if ink is too bold
6. **Normalize** using training set mean/std
7. **Predict** with softmax probabilities for all 10 classes

The canvas uses gray stroke color (`#777777`) and thin line width (20px at 560px canvas → ~1px at 28px) to match training data characteristics.

### 10.3 Frontend UI

The frontend (`ui/index.html`) features a dark-themed, responsive interface:

- **Drawing canvas**: 300×300px display (backed by 560×560px resolution for smoother strokes)
- **28×28 preview**: Shows the exact preprocessed image sent to the model
- **Reference samples**: Displays one real training example per class so users can match the expected style
- **Probability bars**: Animated bars showing confidence for all 10 digit classes
- **Big digit display**: Prominent animated display of the predicted digit
- **Drawing tips**: Guidance on drawing thin, centered strokes
- **Keyboard shortcuts**: Enter to predict, Escape to clear

**Running the demo:**

```bash
cd "Final Project"
python src/api.py
# Open http://localhost:8001 in a browser
```

---

## 11. Generated Outputs

All outputs are saved under the `outputs/` directory:

### Models
| File | Description |
|---|---|
| `models/cnn_aug_best.pth` | Best CNN weights (augmented training) |
| `models/cnn_noaug_best.pth` | Best CNN weights (no augmentation) |

### Metrics & Data
| File | Description |
|---|---|
| `norm_stats.json` | `{mean: 0.9811, std: 0.0763}` |
| `baseline_hog_svm.json` | Accuracy, per-class report, confusion matrix, timing |
| `baseline_knn.json` | Same as above for KNN |
| `cnn_aug_history.json` | Epoch-by-epoch train/val loss and accuracy |
| `cnn_noaug_history.json` | Same for ablation run |
| `final_summary.json` | All test accuracies, CNN report, confusion matrix |
| `splits/split_indices.json` | Train/val/test indices for reproducibility |

### Plots (10 figures)
| File | Description |
|---|---|
| `plots/sample_images.png` | Grid of dataset samples (6 per class) |
| `plots/cm_hog_svm.png` | HOG+SVM confusion matrix |
| `plots/cm_knn.png` | KNN confusion matrix |
| `plots/cm_cnn_aug.png` | CNN (aug) confusion matrix |
| `plots/cm_cnn_noaug.png` | CNN (no aug) confusion matrix |
| `plots/learning_curves_cnn_aug.png` | Train/val loss & accuracy curves |
| `plots/learning_curves_cnn_noaug.png` | Same for ablation |
| `plots/misclassified_cnn_aug.png` | Gallery of misclassified test images |
| `plots/augmentation_verification.png` | Visual proof augmentation works |
| `plots/model_comparison.png` | Bar chart comparing all models |

---

## 12. How to Run

### Full Pipeline

```bash
cd "Final Project"
python src/main.py
```

This runs everything: data loading → baselines → CNN (aug) → CNN (no aug) → evaluation → plots.

### Options

```bash
python src/main.py --skip-baselines    # Skip HOG+SVM and KNN
python src/main.py --cnn-only          # Only train CNN (skip baselines and ablation)
```

### Augmentation Verification

```bash
python src/verify_augmentation.py
```

### Web Demo

```bash
python src/api.py
# Open http://localhost:8001
```

---

## 13. Configuration Reference

All hyperparameters are centralized in `src/config.py`:

### Paths

| Variable | Value |
|---|---|
| `DATA_ROOT` | `Handwritten Arabic Numerals (0-9)/ANGKA ARAB` |
| `OUTPUT_DIR` | `outputs/` |
| `PLOT_DIR` | `outputs/plots/` |
| `MODEL_DIR` | `outputs/models/` |
| `SPLIT_DIR` | `outputs/splits/` |

### General

| Variable | Value |
|---|---|
| `SEED` | 42 |
| `NUM_CLASSES` | 10 |
| `IMAGE_SIZE` | (28, 28) |
| `CLASS_NAMES` | ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] |

### Data Augmentation

| Key | Value |
|---|---|
| `rotation_range` | 12 |
| `width_shift` | 0.08 |
| `height_shift` | 0.08 |
| `zoom_range` | (0.92, 1.08) |
| `fill_mode` | "constant" |
| `fill_value` | 0 |

### CNN Training

| Key | Value |
|---|---|
| `batch_size` | 64 |
| `epochs` | 60 |
| `lr` | 0.001 |
| `weight_decay` | 0.0001 |
| `scheduler_patience` | 5 |
| `scheduler_factor` | 0.5 |
| `early_stop_patience` | 10 |
| `dropout_conv` | 0.25 |
| `dropout_fc` | 0.5 |

### HOG+SVM

| Key | Value |
|---|---|
| `orientations` | 9 |
| `pixels_per_cell` | (4, 4) |
| `cells_per_block` | (2, 2) |
| `svm_C` | 10.0 |

### KNN

| Key | Value |
|---|---|
| `n_neighbors` | 5 |

---

*Documentation generated for the Handwritten Arabic Numeral Recognition project.*
