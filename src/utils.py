from pathlib import Path
import pickle
import sys
from typing import Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def load_plans(plans_file) -> dict:
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


def _pad(
    x: torch.Tensor, d_div: int, h_div: int, w_div: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _B, _C, D, H, W = x.shape
    D_pad = (-D) % d_div; H_pad = (-H) % h_div; W_pad = (-W) % w_div
    if D_pad + H_pad + W_pad:
        x = F.pad(x, (0, W_pad, 0, H_pad, 0, D_pad))
    return x, (D_pad, H_pad, W_pad)


def _unpad(arr: np.ndarray, original_dhw: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = original_dhw
    return arr[:, :D, :H, :W]


def _sentinel(arr: Optional[np.ndarray]) -> np.ndarray:
    return arr if arr is not None else np.zeros((0,), dtype=np.float32)


# Suffix appended to NPZ field names for each aggregation method.
# 'sum' uses no suffix (backward-compatible with existing files).
_AGG_FIELD_SUFFIX: dict = {
    "sum":     "",
    "mean":    "_mean",
    "abs_sum": "_abs_sum",
    "abs_avg": "_abs_avg",
}


def _load_npz_fields(path: Path) -> dict:
    """Load all fields from an .npz file into a plain dict. Returns {} if missing."""
    if not path.exists():
        return {}
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def _is_empty(arr: np.ndarray) -> bool:
    return arr.ndim == 1 and arr.shape[0] == 0


# Primary .npz key for each method name (no aggregation suffix)
METHOD_NPZ_KEYS = {
    "saliency":             "saliency",
    "occlusion":            "occlusion",
    "integrated_gradients": "integrated_gradients",
    "gradient_shap":        "gradient_shap",
    "ablation":             "ablation",
    "input_ablation":       "input_ablation",
}


def methods_already_computed(npz_path: Path, methods: Set[str], agg_sfx: str) -> bool:
    """True if every requested method's primary key is already in the .npz."""
    if not npz_path.exists():
        return False
    existing = _load_npz_fields(npz_path)
    return all(METHOD_NPZ_KEYS.get(m, m) + agg_sfx in existing for m in methods)
