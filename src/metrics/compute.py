import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.utils import _is_empty

PICAI_OVERLAP_THRESHOLD = 0.10  # PI-CAI: predicted lesion counts as TP only if
                                # intersection / GT_volume >= 10 %

METHODS = ["saliency", "occlusion", "integrated_gradients"]


def _detection_overlap(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Fraction of GT voxels covered by the prediction mask.

    Returns intersection / GT_volume, or 0.0 when GT is empty.
    Masks must be broadcastable (same shape or both (D,H,W)).
    """
    gt_volume = float(gt_mask.sum())
    if gt_volume == 0:
        return 0.0
    return float((pred_mask & gt_mask).sum()) / gt_volume


def _channel_stats(arr: np.ndarray) -> dict:
    """Compute per-channel attribution statistics for a (3, D, H, W) array."""
    ch_sum  = arr.sum(axis=(1, 2, 3))
    ch_mean = arr.mean(axis=(1, 2, 3))
    ch_max  = arr.max(axis=(1, 2, 3))
    ch_std  = arr.std(axis=(1, 2, 3))
    total   = ch_sum.sum()
    ch_frac = (ch_sum / total).tolist() if total > 0 else [0.0] * len(ch_sum)
    return {
        "ch_sum":      ch_sum.tolist(),
        "ch_mean":     ch_mean.tolist(),
        "ch_max":      ch_max.tolist(),
        "ch_std":      ch_std.tolist(),
        "ch_fraction": ch_frac,
        "dominant_ch": int(np.argmax(ch_mean)),
    }


def _zone_category(pca_in_pz: int, pca_in_tz: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (zone_category, primary_zone) from voxel counts."""
    if pca_in_pz > 0 and pca_in_tz == 0:
        return "pz_only", "pz"
    if pca_in_tz > 0 and pca_in_pz == 0:
        return "tz_only", "tz"
    if pca_in_pz > 0 and pca_in_tz > 0:
        cat = "both_pz" if pca_in_pz >= pca_in_tz else "both_tz"
        return cat, "pz" if pca_in_pz >= pca_in_tz else "tz"
    return "unknown", None


def compute_metrics(xai_dir: Path, model_name: str, metrics_dir: Path) -> List[dict]:
    """
    Read all .npz files in xai_dir/fold_*/, compute per-case metrics from the
    saved label, prediction, and zones arrays, and write sample_data.json.
    Returns the list of per-sample records.
    """
    import json

    model_metrics_dir = metrics_dir / model_name
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_metrics_dir / "sample_data.json"

    npz_files = sorted((xai_dir / model_name).glob("fold_*/*.npz"))
    print(f"\nComputing metrics for {model_name}: {len(npz_files)} npz files")

    records = []
    for npz_path in npz_files:
        try:
            d     = np.load(npz_path, allow_pickle=True)
            label = d["label"]       # (1, D, H, W) or (0,)
            pred  = d["prediction"]  # (1, D, H, W)
            zones = d["zones"]       # (D, H, W) int8 or (0,)

            case_id = str(d["case_id"])
            fold    = int(d["fold"])

            has_pca          = (not _is_empty(label)) and bool(label.sum() > 1)
            predicted_pos    = (not _is_empty(pred)) and bool(pred.max() > 0.5)

            # PI-CAI TP criterion: prediction must overlap ≥10 % of the GT lesion volume.
            overlap = 0.0
            if predicted_pos and has_pca:
                pred_mask_3d = pred[0] > 0.5   # (D, H, W) bool
                gt_mask_3d   = label[0] > 0    # (D, H, W) bool
                overlap      = _detection_overlap(pred_mask_3d, gt_mask_3d)

            if predicted_pos and has_pca and overlap >= PICAI_OVERLAP_THRESHOLD:
                classification = "tp"
            elif predicted_pos and not has_pca:
                classification = "fp"
            elif predicted_pos and has_pca and overlap < PICAI_OVERLAP_THRESHOLD:
                classification = "fp"  # predicted but missed the lesion
            elif not predicted_pos and not has_pca:
                classification = "tn"
            else:
                classification = "fn"

            # Zone stats from saved zones + label
            if not _is_empty(label) and not _is_empty(zones) and zones.ndim == 3:
                lbl_2d   = label[0]          # (D, H, W)
                pca_pz   = int((lbl_2d * (zones == 1)).sum())
                pca_tz   = int((lbl_2d * (zones == 2)).sum())
            else:
                pca_pz = pca_tz = 0

            pred_pz = pred_tz = 0
            if predicted_pos and not _is_empty(zones) and zones.ndim == 3:
                pred_mask = (pred[0] > 0.5)  # (D, H, W)
                pred_pz   = int((pred_mask * (zones == 1)).sum())
                pred_tz   = int((pred_mask * (zones == 2)).sum())

            zone_cat, primary_zone = _zone_category(pca_pz, pca_tz)

            record: dict = {
                "case_id":           case_id,
                "fold":              fold,
                "model":             model_name,
                "classification":    classification,
                "has_pca":           has_pca,
                "predicted_positive": predicted_pos,
                "primary_zone":      primary_zone,
                "zone_category":     zone_cat,
                "pz_voxels":         pca_pz if has_pca else None,
                "tz_voxels":         pca_tz if has_pca else None,
                "pred_pz_voxels":    pred_pz if predicted_pos else None,
                "pred_tz_voxels":    pred_tz if predicted_pos else None,
            }

            # Channel attribution stats (only when XAI maps were computed)
            for method in METHODS:
                arr = d.get(method, np.zeros((0,), dtype=np.float32))
                if not _is_empty(arr) and arr.ndim == 4:
                    record[method] = _channel_stats(np.abs(arr))
                else:
                    record[method] = None

            records.append(record)

        except Exception as exc:
            warnings.warn(f"Failed to load {npz_path}: {exc}")

    print(f"  Loaded {len(records)} samples.")
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved: {cache_path}")
    return records
