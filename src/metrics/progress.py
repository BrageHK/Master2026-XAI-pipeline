import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.metrics.compute import (
    PICAI_OVERLAP_THRESHOLD,
    _channel_stats,
    _detection_overlap,
    _zone_category,
)


def _build_progress_record(
    predicted_pos: bool,
    lbl_crop: Optional[np.ndarray],
    zones_crop: Optional[np.ndarray],
    sal_np: Optional[np.ndarray],
    occ_np: Optional[np.ndarray] = None,
    abl_np: Optional[np.ndarray] = None,
    occ_tz_np: Optional[np.ndarray] = None,
    occ_pz_np: Optional[np.ndarray] = None,
    ig_np: Optional[np.ndarray] = None,
    pred_crop: Optional[np.ndarray] = None,
    pred_cancer_voxels: int = 0,
    pred_max_prob: float = 0.0,
    confidence: float = 0.0,
) -> dict:
    """Build a per-case record for progress.json from already-computed arrays."""
    has_pca = (lbl_crop is not None) and bool(lbl_crop.sum() > 1)

    # PI-CAI TP criterion: prediction must overlap ≥10 % of the GT lesion volume.
    overlap = 0.0
    if predicted_pos and has_pca and pred_crop is not None:
        pred_mask_3d = pred_crop[0] > 0.5   # (D, H, W) bool
        gt_mask_3d   = lbl_crop[0] > 0      # (D, H, W) bool
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

    pca_pz = pca_tz = 0
    if has_pca and zones_crop is not None and zones_crop.ndim == 3:
        lbl_2d = lbl_crop[0]
        pca_pz = int((lbl_2d * (zones_crop == 1)).sum())
        pca_tz = int((lbl_2d * (zones_crop == 2)).sum())

    pred_pz = pred_tz = 0
    if predicted_pos and pred_crop is not None and zones_crop is not None and zones_crop.ndim == 3:
        pred_mask = (pred_crop[0] > 0.5)
        pred_pz = int((pred_mask * (zones_crop == 1)).sum())
        pred_tz = int((pred_mask * (zones_crop == 2)).sum())

    zone_cat, primary_zone = _zone_category(pca_pz, pca_tz)

    sal_frac = None
    if sal_np is not None and sal_np.ndim == 4:
        sal_frac = _channel_stats(np.abs(sal_np))["ch_fraction"]

    occ_frac = None
    if occ_np is not None and occ_np.ndim == 4:
        occ_frac = _channel_stats(np.abs(occ_np))["ch_fraction"]

    # Per-strategy occlusion stats (fraction + mean)
    occ_zero_stats = None
    if occ_np is not None and occ_np.ndim == 4:
        s = _channel_stats(np.abs(occ_np))
        occ_zero_stats = {"ch_fraction": s["ch_fraction"], "ch_mean": s["ch_mean"]}

    occ_zm_stats = None
    if (occ_tz_np is not None and occ_tz_np.ndim == 4
            and occ_pz_np is not None and occ_pz_np.ndim == 4
            and zones_crop is not None and zones_crop.ndim == 3):
        _merged = np.zeros_like(occ_tz_np)
        _merged[:, zones_crop == 2] = occ_tz_np[:, zones_crop == 2]
        _merged[:, zones_crop == 1] = occ_pz_np[:, zones_crop == 1]
        s = _channel_stats(np.abs(_merged))
        occ_zm_stats = {"ch_fraction": s["ch_fraction"], "ch_mean": s["ch_mean"]}

    # Use first available strategy for the legacy occlusion_ch_fraction field
    occ_frac = (occ_zero_stats or occ_zm_stats or {}).get("ch_fraction", occ_frac)

    # AblationCAM produces a single-channel spatial map (1, D, H, W);
    # there is no per-input-channel breakdown, so we store None.
    abl_frac = None  # reserved for future per-channel ablation variants

    ig_frac = None
    if ig_np is not None and ig_np.ndim == 4:
        ig_frac = _channel_stats(np.abs(ig_np))["ch_fraction"]

    return {
        "done":                    True,
        "error":                   None,
        "predicted_pos":           predicted_pos,
        "pred_cancer_voxels":      pred_cancer_voxels,
        "pred_max_prob":           round(float(pred_max_prob), 4),
        "confidence":              round(float(confidence), 4),
        "has_pca":                 has_pca,
        "gt_cancer_voxels":        int(lbl_crop.sum()) if has_pca else 0,
        "classification":          classification,
        "primary_zone":            primary_zone,
        "zone_category":           zone_cat,
        "pz_voxels":               pca_pz if has_pca else None,
        "tz_voxels":               pca_tz if has_pca else None,
        "pred_pz_voxels":          pred_pz if predicted_pos else None,
        "pred_tz_voxels":          pred_tz if predicted_pos else None,
        "saliency_ch_fraction":        sal_frac,
        "occlusion_ch_fraction":       occ_frac,
        "ablation_ch_fraction":        abl_frac,
        "ig_ch_fraction":              ig_frac,
        "occlusion_zero_ch_fraction":  occ_zero_stats["ch_fraction"] if occ_zero_stats else None,
        "occlusion_zero_ch_mean":      occ_zero_stats["ch_mean"]     if occ_zero_stats else None,
        "occlusion_zm_ch_fraction":    occ_zm_stats["ch_fraction"]   if occ_zm_stats else None,
        "occlusion_zm_ch_mean":        occ_zm_stats["ch_mean"]       if occ_zm_stats else None,
    }


def _save_progress(progress: dict, progress_file: Path) -> None:
    """Atomically write progress dict to JSON (safe against mid-write kills)."""
    tmp = progress_file.with_name(".progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, progress_file)
