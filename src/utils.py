from pathlib import Path
import pickle
import sys
import torch
import numpy as np

def load_plans(plans_file: str | Path) -> dict:
    with open(plans_file, "rb") as f:
        return pickle.load(f)

def log_large_vars(local_vars, threshold_mb=100):
    sizes = []
    for name, obj in local_vars.items():
        if isinstance(obj, np.ndarray):
            mb = obj.nbytes / 1e6
        elif isinstance(obj, torch.Tensor):
            mb = obj.element_size() * obj.nelement() / 1e6
        else:
            continue
        if mb >= threshold_mb:
            sizes.append((mb, name, type(obj).__name__, getattr(obj, 'shape', '?')))
    for mb, name, typ, shape in sorted(sizes, reverse=True):
        print(f"  {name:30s} {mb:8.1f} MB  {typ} {shape}")
