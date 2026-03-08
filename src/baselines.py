# baselines.py -- Classical ML baselines: HOG+SVM and raw-pixel KNN.

import os
import json
import time
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import HOG_PARAMS, SVM_C, KNN_K, IMAGE_SIZE, OUTPUT_DIR, CLASS_NAMES


# --- HOG feature extraction ---

def extract_hog_features(X_flat):
    """Compute HOG descriptors for each flattened 784-d image."""
    feats = []
    for i in range(len(X_flat)):
        img = X_flat[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
        h = hog(img,
                orientations=HOG_PARAMS["orientations"],
                pixels_per_cell=HOG_PARAMS["pixels_per_cell"],
                cells_per_block=HOG_PARAMS["cells_per_block"],
                block_norm="L2-Hys")
        feats.append(h)
    return np.array(feats, dtype=np.float32)


# --- HOG + SVM ---

def train_hog_svm(X_train, y_train, X_test, y_test):
    """Train an RBF-SVM on HOG features, return (accuracy, report, cm, time)."""
    print("[BASELINE] Extracting HOG features (train) ...")
    hog_train = extract_hog_features(X_train)
    print("[BASELINE] Extracting HOG features (test) ...")
    hog_test  = extract_hog_features(X_test)

    print(f"[BASELINE] Training SVM (C={SVM_C}) ...")
    t0 = time.perf_counter()
    clf = SVC(C=SVM_C, kernel="rbf", gamma="scale", random_state=42)
    clf.fit(hog_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(hog_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[BASELINE] HOG + SVM  accuracy = {acc:.4f}  ({elapsed:.1f}s)")
    return acc, report, cm, elapsed


# --- Raw pixels + KNN ---

def train_knn(X_train, y_train, X_test, y_test):
    """Train KNN on raw pixel vectors, return (accuracy, report, cm, time)."""
    print(f"[BASELINE] Training KNN (k={KNN_K}) ...")
    t0 = time.perf_counter()
    clf = KNeighborsClassifier(n_neighbors=KNN_K, n_jobs=-1)
    clf.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[BASELINE] Raw + KNN  accuracy = {acc:.4f}  ({elapsed:.1f}s)")
    return acc, report, cm, elapsed


def save_baseline_results(name, acc, report, cm, elapsed):
    """Persist baseline results to JSON."""
    out = {
        "model": name,
        "accuracy": float(acc),
        "train_time_seconds": float(elapsed),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    path = os.path.join(OUTPUT_DIR, f"baseline_{name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] Baseline results → {path}")
