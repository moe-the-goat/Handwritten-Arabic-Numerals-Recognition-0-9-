# train.py -- Training loop with early stopping and learning-rate scheduling.

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import CNN as CFG, MODEL_DIR, OUTPUT_DIR, SEED
from utils import set_global_seed, get_device, Timer


def train_model(model, train_loader, val_loader, device=None, tag="cnn"):
    """Train the model, save best weights, return (model, history_dict)."""
    set_global_seed(SEED)
    if device is None:
        device = get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(),
                     lr=CFG["learning_rate"],
                     weight_decay=CFG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=CFG["lr_scheduler_factor"],
                                  patience=CFG["lr_scheduler_patience"])

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, CFG["epochs"] + 1):
        # --- Training pass ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # --- Validation pass ---
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * imgs.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += imgs.size(0)

        val_loss = running_loss / total
        val_acc  = correct / total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:3d}/{CFG['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.1e}",
              flush=True)

        # --- Check early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save best model & history
    model_path = os.path.join(MODEL_DIR, f"{tag}_best.pth")
    torch.save(best_state, model_path)
    print(f"[SAVE] Best model → {model_path}")

    hist_path = os.path.join(OUTPUT_DIR, f"{tag}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] Training history → {hist_path}")

    # Restore best weights into model
    model.load_state_dict(best_state)
    return model, history
