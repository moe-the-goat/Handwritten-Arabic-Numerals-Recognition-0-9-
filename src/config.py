# config.py -- Central configuration for the Arabic Numeral Recognition project.
# Every tunable parameter lives here so experiments stay reproducible.

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR  = os.path.join(PROJECT_ROOT, "Handwritten Arabic Numerals (0-9)", "ANGKA ARAB")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR    = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
SPLIT_DIR    = os.path.join(OUTPUT_DIR, "splits")

# Create directories
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, SPLIT_DIR]:
    os.makedirs(d, exist_ok=True)

# Dataset properties  (9350 images total, 935 per digit)
NUM_CLASSES  = 10
IMAGE_SIZE   = 28
CLASS_NAMES  = [str(i) for i in range(10)]
TOTAL_IMAGES = 9350

FOLDER_TO_LABEL = {str(i): i for i in range(10)}

SEED = 42

# Stratified train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Mean and std are computed from the training split at runtime
NORMALIZE_MEAN = None
NORMALIZE_STD  = None

# Augmentation: mild geometric transforms to simulate handwriting variation
AUGMENTATION = dict(
    rotation_range=12,        # ±12°
    width_shift=0.08,
    height_shift=0.08,
    zoom_range=(0.92, 1.08),
    fill_mode="constant",
    fill_value=0,
)

# CNN training hyper-parameters
CNN = dict(
    batch_size=64,
    epochs=60,
    learning_rate=1e-3,
    weight_decay=1e-4,
    lr_scheduler_patience=5,
    lr_scheduler_factor=0.5,
    early_stop_patience=10,
    dropout_conv=0.25,
    dropout_fc=0.5,
)

# Classical baseline hyper-parameters
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
)
SVM_C = 10.0
KNN_K = 5
