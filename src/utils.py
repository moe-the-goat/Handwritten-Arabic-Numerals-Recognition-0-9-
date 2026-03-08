# utils.py -- Seed control, device selection, and a handy timer.
import os
import random
import time
import numpy as np
import torch


def set_global_seed(seed: int = 42):
    """Lock every random source so results are fully reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick GPU if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[INFO] Using CPU")
    return dev


class Timer:
    """Context-manager that prints elapsed wall-clock time on exit."""
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"[TIMER] {self.label}: {elapsed:.2f}s")
