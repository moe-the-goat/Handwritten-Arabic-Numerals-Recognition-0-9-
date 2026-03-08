#!/usr/bin/env python3
# main.py -- Orchestrates the full pipeline: data loading, baseline training,
#            CNN training (with and without augmentation), evaluation, and plotting.
#
# Usage:
#   python src/main.py                   # everything
#   python src/main.py --skip-baselines  # skip HOG+SVM / KNN
#   python src/main.py --cnn-only        # CNN only

import os
import sys
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (SEED, OUTPUT_DIR, PLOT_DIR, MODEL_DIR, SPLIT_DIR,
                    CNN as CNN_CFG, CLASS_NAMES)
from utils import set_global_seed, get_device, Timer
from data_pipeline import (load_dataset, load_images_as_arrays,
                           stratified_split, compute_norm_stats,
                           build_loaders, get_flat_arrays)
from models import ArabicDigitCNN, count_parameters
from train import train_model
from baselines import (train_hog_svm, train_knn, save_baseline_results)
from evaluate import (evaluate_model, plot_confusion_matrix,
                      print_classification_report, plot_learning_curves,
                      plot_misclassified, plot_comparison,
                      plot_sample_images, save_summary)

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Arabic Numeral Recognition Pipeline")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip HOG+SVM and KNN baselines")
    p.add_argument("--cnn-only", action="store_true",
                   help="Only run CNN (skip baselines)")
    p.add_argument("--no-aug", action="store_true",
                   help="Train CNN *without* augmentation (for ablation)")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(SEED)
    device = get_device()

    results = {}  # model_name -> test accuracy

    # ---- Step 1: Load dataset & create stratified split ----
    print("\n" + "=" * 70)
    print("  STEP 1 -- Load dataset & stratified split")
    print("=" * 70)
    paths, labels = load_dataset()
    splits = stratified_split(paths, labels)

    train_images = load_images_as_arrays(splits["train"][0])
    mean, std = compute_norm_stats(train_images)

    norm_path = os.path.join(OUTPUT_DIR, "norm_stats.json")
    with open(norm_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f)
    print(f"[SAVE] Normalization stats -> {norm_path}")

    # Quick visual overview of the dataset
    all_images = load_images_as_arrays(paths)
    plot_sample_images(all_images, labels, n_per_class=6)
    del all_images

    # ---- Step 2: Classical baselines (HOG+SVM, KNN) ----
    if not args.skip_baselines and not args.cnn_only:
        print("\n" + "=" * 70)
        print("  STEP 2 -- Baseline models")
        print("=" * 70)

        X_tr, y_tr, X_val, y_val, X_te, y_te = get_flat_arrays(splits)

        with Timer("HOG + SVM"):
            acc_svm, rep_svm, cm_svm, t_svm = train_hog_svm(X_tr, y_tr, X_te, y_te)
        results["HOG + SVM"] = acc_svm
        save_baseline_results("hog_svm", acc_svm, rep_svm, cm_svm, t_svm)
        _plot_cm_from_array(cm_svm, "HOG + SVM -- Confusion Matrix", "cm_hog_svm.png")

        with Timer("KNN"):
            acc_knn, rep_knn, cm_knn, t_knn = train_knn(X_tr, y_tr, X_te, y_te)
        results["Raw + KNN"] = acc_knn
        save_baseline_results("knn", acc_knn, rep_knn, cm_knn, t_knn)
        _plot_cm_from_array(cm_knn, "Raw Pixels + KNN -- Confusion Matrix", "cm_knn.png")
    else:
        print("\n[SKIP] Baselines skipped by user flag.")

    # ---- Step 3: CNN with data augmentation ----
    print("\n" + "=" * 70)
    print("  STEP 3 -- CNN training (with data augmentation)")
    print("=" * 70)

    train_loader, val_loader, test_loader = build_loaders(splits, mean, std)

    model = ArabicDigitCNN()
    total_p, train_p = count_parameters(model)
    print(f"[MODEL] Parameters: {total_p:,} total, {train_p:,} trainable")

    with Timer("CNN Training"):
        model, history = train_model(model, train_loader, val_loader,
                                     device=device, tag="cnn_aug")

    acc_cnn, preds_cnn, labels_cnn, probs_cnn = evaluate_model(
        model, test_loader, device)
    print(f"\n  *  CNN (aug) Test Accuracy = {acc_cnn:.4f}")
    results["CNN (aug)"] = acc_cnn

    cnn_report = print_classification_report(labels_cnn, preds_cnn,
                                             "CNN (aug) -- Test Set")
    cm_cnn = plot_confusion_matrix(labels_cnn, preds_cnn,
                                   "CNN (aug) -- Confusion Matrix",
                                   "cm_cnn_aug.png")
    plot_learning_curves(history, "learning_curves_cnn_aug.png")

    # Gallery of misclassified images for error analysis
    test_images = load_images_as_arrays(splits["test"][0])
    plot_misclassified(test_images, labels_cnn, preds_cnn, n=16,
                       fname="misclassified_cnn_aug.png")

    # ---- Step 3b: Ablation -- CNN without augmentation ----
    if not args.cnn_only:
        print("\n" + "=" * 70)
        print("  STEP 3b -- Ablation: CNN without augmentation")
        print("=" * 70)

        from data_pipeline import (ArabicDigitDataset, get_eval_transform)
        from torch.utils.data import DataLoader

        tr_imgs = load_images_as_arrays(splits["train"][0])
        val_imgs = load_images_as_arrays(splits["val"][0])
        tst_imgs = load_images_as_arrays(splits["test"][0])

        noaug_transform = get_eval_transform(mean, std)
        tr_ds  = ArabicDigitDataset(tr_imgs,  splits["train"][1], noaug_transform)
        val_ds = ArabicDigitDataset(val_imgs, splits["val"][1],   noaug_transform)
        tst_ds = ArabicDigitDataset(tst_imgs, splits["test"][1],  noaug_transform)

        bs = CNN_CFG["batch_size"]
        tr_ld  = DataLoader(tr_ds,  batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
        val_ld = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
        tst_ld = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

        model_noaug = ArabicDigitCNN()
        with Timer("CNN (no aug) Training"):
            model_noaug, hist_noaug = train_model(
                model_noaug, tr_ld, val_ld, device=device, tag="cnn_noaug")

        acc_noaug, preds_noaug, labels_noaug, _ = evaluate_model(
            model_noaug, tst_ld, device)
        print(f"\n  *  CNN (no aug) Test Accuracy = {acc_noaug:.4f}")
        results["CNN (no aug)"] = acc_noaug

        print_classification_report(labels_noaug, preds_noaug,
                                    "CNN (no aug) -- Test Set")
        plot_confusion_matrix(labels_noaug, preds_noaug,
                              "CNN (no aug) -- Confusion Matrix",
                              "cm_cnn_noaug.png")
        plot_learning_curves(hist_noaug, "learning_curves_cnn_noaug.png")

    # ---- Step 4: Comparison & final summary ----
    print("\n" + "=" * 70)
    print("  STEP 4 -- Comparison & final summary")
    print("=" * 70)

    plot_comparison(results)
    save_summary(results, cnn_report, cm_cnn)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:<20s}  {acc:.4f}  ({acc:.2%})")
    print("=" * 70)
    print("  All outputs saved in:", OUTPUT_DIR)
    print("=" * 70 + "\n")


def _plot_cm_from_array(cm, title, fname):
    """Plot a confusion matrix from an already-computed numpy array."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
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
    print(f"[PLOT] {title} -> {path}")


if __name__ == "__main__":
    main()
