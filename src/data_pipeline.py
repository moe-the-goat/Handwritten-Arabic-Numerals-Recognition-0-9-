# data_pipeline.py -- Image loading, train/val/test splitting, and DataLoader creation.

import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    DATASET_DIR, SPLIT_DIR, SEED, IMAGE_SIZE, NUM_CLASSES,
    FOLDER_TO_LABEL, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    AUGMENTATION, CNN,
)
from utils import set_global_seed

# --- Loading raw images from disk ---

def load_dataset():
    """Scan the dataset folder and return (file_paths, labels) arrays."""
    paths, labels = [], []
    for folder_name in sorted(os.listdir(DATASET_DIR)):
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = FOLDER_TO_LABEL.get(folder_name)
        if label is None:
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                paths.append(os.path.join(folder_path, fname))
                labels.append(label)
    print(f"[DATA] Loaded {len(paths)} images across {NUM_CLASSES} classes")
    return np.array(paths), np.array(labels)


def load_images_as_arrays(paths):
    """Read image files into a (N, 28, 28) uint8 numpy array."""
    images = []
    for p in paths:
        img = Image.open(p).convert("L")  # grayscale
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        images.append(np.array(img, dtype=np.uint8))
    return np.stack(images)


# --- Stratified split ---

def stratified_split(paths, labels, save=True):
    """70/15/15 stratified split. Saves indices for reproducibility."""
    set_global_seed(SEED)
    val_test_ratio = VAL_RATIO + TEST_RATIO
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=SEED,
    )
    relative_test = TEST_RATIO / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=SEED,
    )
    splits = {
        "train": (X_train, y_train),
        "val":   (X_val, y_val),
        "test":  (X_test, y_test),
    }
    print(f"[SPLIT] train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    if save:
        # persist file paths so anyone can recreate the exact same split
        split_record = {}
        for key in splits:
            p, l = splits[key]
            split_record[key] = {
                "paths": p.tolist(),
                "labels": l.tolist(),
            }
        out_path = os.path.join(SPLIT_DIR, "split_indices.json")
        with open(out_path, "w") as f:
            json.dump(split_record, f, indent=2)
        print(f"[SPLIT] Saved split indices to {out_path}")

    return splits


# --- Normalization stats from train set ---

def compute_norm_stats(images_uint8):
    """Channel mean & std over uint8 images (N x H x W)."""
    imgs = images_uint8.astype(np.float32) / 255.0
    mean = imgs.mean()
    std  = imgs.std()
    print(f"[NORM] Train mean={mean:.4f}  std={std:.4f}")
    return float(mean), float(std)


# --- PyTorch Dataset ---

class ArabicDigitDataset(Dataset):
    """Wraps in-memory uint8 numpy images for use with a DataLoader."""

    def __init__(self, images_uint8, labels, transform=None):
        self.images = images_uint8
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img, mode="L")
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label


# --- Transforms (augmented vs plain) ---

def get_train_transform(mean, std):
    """Training: random affine augmentation + normalization."""
    aug = AUGMENTATION
    return T.Compose([
        T.RandomAffine(
            degrees=aug["rotation_range"],
            translate=(aug["width_shift"], aug["height_shift"]),
            scale=aug["zoom_range"],
            fill=aug["fill_value"],
        ),
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])


def get_eval_transform(mean, std):
    """Validation/test: just normalize, no augmentation."""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[mean], std=[std]),
    ])


# --- DataLoader factory ---

def build_loaders(splits, mean, std, batch_size=None):
    """Build train/val/test DataLoaders from the split dictionary."""
    bs = batch_size or CNN["batch_size"]

    train_imgs = load_images_as_arrays(splits["train"][0])
    val_imgs   = load_images_as_arrays(splits["val"][0])
    test_imgs  = load_images_as_arrays(splits["test"][0])

    train_ds = ArabicDigitDataset(
        train_imgs, splits["train"][1],
        transform=get_train_transform(mean, std)
    )
    val_ds = ArabicDigitDataset(
        val_imgs, splits["val"][1], transform=get_eval_transform(mean, std)
    )
    test_ds = ArabicDigitDataset(
        test_imgs, splits["test"][1], transform=get_eval_transform(mean, std)
    )

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False,
                              num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# --- Flat arrays for sklearn baselines ---

def get_flat_arrays(splits):
    """
    Return (X_train, y_train, X_val, y_val, X_test, y_test)
    where X is float32 in [0,1], shape (N, 784).
    """
    data = {}
    for key in ["train", "val", "test"]:
        imgs = load_images_as_arrays(splits[key][0])
        X = imgs.astype(np.float32).reshape(len(imgs), -1) / 255.0
        y = splits[key][1]
        data[key] = (X, y)
    return (data["train"][0], data["train"][1],
            data["val"][0],   data["val"][1],
            data["test"][0],  data["test"][1])
