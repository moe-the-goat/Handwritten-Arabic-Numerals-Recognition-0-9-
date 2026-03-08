# evaluate.py -- Evaluation metrics, confusion matrices, learning curves,
#                misclassified gallery, and model comparison chart.

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non‑interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from config import PLOT_DIR, OUTPUT_DIR, CLASS_NAMES, NUM_CLASSES


# --- Evaluate CNN on test DataLoader ---

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Run inference, return (accuracy, preds, labels, probs)."""
    model.eval()
    preds, labels_all, probs_all = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        prob = torch.softmax(outputs, dim=1).cpu().numpy()
        pred = outputs.argmax(1).cpu().numpy()
        preds.append(pred)
        labels_all.append(labels.numpy())
        probs_all.append(prob)
    preds = np.concatenate(preds)
    labels_all = np.concatenate(labels_all)
    probs_all  = np.concatenate(probs_all)
    acc = accuracy_score(labels_all, preds)
    return acc, preds, labels_all, probs_all


# --- Confusion matrix ---

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", fname="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {title} → {path}")
    return cm


# --- Per-class precision / recall / F1 ---

def print_classification_report(y_true, y_pred, title="Classification Report"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(report_str)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    return report_dict


# --- Learning curves (loss + accuracy vs epoch) ---

def plot_learning_curves(history, fname="learning_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Learning curves → {path}")


# --- Misclassified sample gallery ---

def plot_misclassified(images_uint8, y_true, y_pred, n=16,
                       fname="misclassified_samples.png"):
    """Grid of incorrectly classified images showing true vs predicted labels."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("[PLOT] No misclassified samples!")
        return
    n = min(n, len(wrong))
    idxs = np.random.choice(wrong, n, replace=False)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for i, idx in enumerate(idxs):
        axes[i].imshow(images_uint8[idx], cmap="gray")
        axes[i].set_title(f"T:{y_true[idx]}  P:{y_pred[idx]}", fontsize=10,
                          color="red")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Misclassified Samples", fontsize=14)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Misclassified gallery → {path}")


# --- Model comparison bar chart ---

def plot_comparison(results_dict, fname="model_comparison.png"):
    """Bar chart comparing test accuracy across all models."""
    names = list(results_dict.keys())
    accs  = [results_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"][:len(names)])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Comparison — Test Accuracy", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Comparison chart → {path}")


# --- Sample images grid (for the report) ---

def plot_sample_images(images_uint8, labels, n_per_class=5,
                       fname="sample_images.png"):
    """Show a few examples per class in one figure."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class,
                             figsize=(n_per_class * 1.5, NUM_CLASSES * 1.5))
    for cls in range(NUM_CLASSES):
        idxs = np.where(labels == cls)[0][:n_per_class]
        for j, idx in enumerate(idxs):
            axes[cls, j].imshow(images_uint8[idx], cmap="gray")
            axes[cls, j].axis("off")
            if j == 0:
                axes[cls, j].set_ylabel(str(cls), fontsize=12, rotation=0,
                                        labelpad=20)
    fig.suptitle("Sample Images per Class", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Sample images → {path}")


# --- Save final summary JSON ---

def save_summary(results_dict, cnn_report, cnn_cm):
    """Save a JSON summary of all results."""
    summary = {
        "accuracies": {k: float(v) for k, v in results_dict.items()},
        "cnn_classification_report": cnn_report,
        "cnn_confusion_matrix": cnn_cm.tolist() if hasattr(cnn_cm, "tolist") else cnn_cm,
    }
    path = os.path.join(OUTPUT_DIR, "final_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVE] Final summary → {path}")
