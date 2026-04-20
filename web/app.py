#!/usr/bin/env python3
"""
PI-CAI XAI Visualization Server.

Serves an interactive website for browsing XAI results from generate_xai_data.py.

Usage:
    uv run python web/app.py
    uv run python web/app.py --port 8080 --host 0.0.0.0
"""

import argparse
import base64
import io
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, abort, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR  = PROJECT_ROOT / "results" / "metrics"
XAI_DIR      = PROJECT_ROOT / "results" / "xai"
ANALYSIS_DIR = PROJECT_ROOT / "results" / "analysis"
MODELS        = ["umamba_mtl", "swin_unetr", "nnunet"]
ZONES_PRED_DIR = XAI_DIR / "zones"   # umamba predicted zones: zones/fold_N/case_id.npz

CHANNEL_NAMES = ["T2W", "ADC", "HBV"]

# ---------------------------------------------------------------------------
# App + dynamic data load
# ---------------------------------------------------------------------------
app = Flask(__name__)

_CACHE_TTL = 30  # seconds between rescans
_cache_time: float = 0.0
_CASE_DATA: Dict[str, List[dict]] = {}
_CASE_INDEX: Dict[str, Dict[str, dict]] = {}


def _is_empty(arr) -> bool:
    return arr is None or (isinstance(arr, np.ndarray) and arr.size == 0)


def _scan_progress_records(model: str) -> List[dict]:
    """Build case records from progress.json files (includes TN/FN without NPZ files)."""
    records = []
    model_dir = XAI_DIR / model
    if not model_dir.exists():
        return records

    for fold_dir in sorted(model_dir.glob("fold_*")):
        progress_path = fold_dir / "progress.json"
        if not progress_path.exists():
            continue
        try:
            fold = int(fold_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        npz_cases = {f.stem for f in fold_dir.glob("*.npz")}

        try:
            with open(progress_path) as f:
                progress = json.load(f)
        except Exception:
            continue

        for case_id, entry in progress.items():
            if not entry.get("done") or entry.get("error"):
                continue

            def _ch_stat(key: str):
                frac = entry.get(f"{key}_ch_fraction")
                if not frac:
                    return None
                result = {"ch_fraction": frac, "dominant_ch": int(np.argmax(frac))}
                mean = entry.get(f"{key}_ch_mean")
                if mean:
                    result["ch_mean"] = mean
                return result

            occ_by_strat = {}
            for _strat, _key in (("zero", "occlusion_zero"), ("zone_median", "occlusion_zm")):
                _s = _ch_stat(_key)
                if _s:
                    occ_by_strat[_strat] = _s

            record: dict = {
                "case_id":            case_id,
                "fold":               fold,
                "model":              model,
                "classification":     entry.get("classification"),
                "has_pca":            entry.get("has_pca", False),
                "predicted_positive": entry.get("predicted_pos", False),
                "primary_zone":       entry.get("primary_zone"),
                "zone_category":      entry.get("zone_category"),
                "pz_voxels":          entry.get("pz_voxels"),
                "tz_voxels":          entry.get("tz_voxels"),
                "pred_pz_voxels":     entry.get("pred_pz_voxels"),
                "pred_tz_voxels":     entry.get("pred_tz_voxels"),
                "pred_cancer_voxels": entry.get("pred_cancer_voxels"),
                "gt_cancer_voxels":   entry.get("gt_cancer_voxels"),
                "confidence":         entry.get("confidence"),
                "pred_max_prob":      entry.get("pred_max_prob"),
                "has_npz":            case_id in npz_cases,
                "saliency":           _ch_stat("saliency"),
                "occlusion":          _ch_stat("occlusion"),
                "ablation":           _ch_stat("ablation"),
                "integrated_gradients": _ch_stat("ig"),
                "occlusion_by_strategy": occ_by_strat,
            }
            records.append(record)

    return records


def _load_all_sample_data() -> Dict[str, List[dict]]:
    """Always load from progress.json (source of truth for all models)."""
    return {model: _scan_progress_records(model) for model in MODELS}


def _get_case_data() -> Dict[str, List[dict]]:
    """Return case data, refreshing from disk at most every _CACHE_TTL seconds."""
    global _cache_time, _CASE_DATA, _CASE_INDEX
    if time.monotonic() - _cache_time > _CACHE_TTL:
        _CASE_DATA  = _load_all_sample_data()
        _CASE_INDEX = {
            model: {r["case_id"]: r for r in records}
            for model, records in _CASE_DATA.items()
        }
        _cache_time = time.monotonic()
    return _CASE_DATA


def _get_case_index() -> Dict[str, Dict[str, dict]]:
    _get_case_data()
    return _CASE_INDEX


# Initialise on startup
_get_case_data()


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _model_stats(model: str) -> dict:
    path = ANALYSIS_DIR / model / "summary.json"
    if path.exists():
        with open(path) as f:
            s = json.load(f)
        def _r(v): return round(v, 3) if v is not None else 0.0
        return dict(
            tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"],
            total=s["total_cases"],
            precision=_r(s["precision"]),
            sensitivity=_r(s["sensitivity"]),
            specificity=_r(s.get("specificity")),
            f1=_r(s["f1"]),
            has_data=s["total_cases"] > 0,
        )
    # Fallback: compute from loaded records
    records = _get_case_data().get(model, [])
    counts  = {c: sum(1 for r in records if r["classification"] == c)
               for c in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total       = tp + fp + fn + tn
    precision   = tp / (tp + fp)          if (tp + fp) > 0     else 0.0
    sensitivity = tp / (tp + fn)          if (tp + fn) > 0     else 0.0
    specificity = tn / (tn + fp)          if (tn + fp) > 0     else 0.0
    f1          = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0  else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, total=total,
                precision=round(precision, 3), sensitivity=round(sensitivity, 3),
                specificity=round(specificity, 3), f1=round(f1, 3),
                has_data=total > 0)


def _available_charts(model: str) -> dict:
    """
    Returns a nested dict describing which chart PNGs are available.
    Structure: {channel_activation: {method: [filter]}, saliency: {class: {zone: [fname]}}, occlusion: ..., ablation: ...}
    """
    base = METRICS_DIR / model
    out: dict = {"channel_activation": {}, "saliency": {}, "occlusion": {}, "ablation": {}}

    # Channel activation charts from ANALYSIS_DIR — 3x3 grid (cls x zone)
    ca_base   = ANALYSIS_DIR / model / "channel_activation"
    cls_keys  = ("tp_fp", "tp", "fp")
    zone_keys = ("all", "pz", "tz")
    for method in ("saliency", "occlusion"):
        method_dir = ca_base / method
        if not method_dir.exists():
            continue
        grid: dict = {}
        for cls_key in cls_keys:
            for zone_key in zone_keys:
                cell_dir = method_dir / cls_key / zone_key
                pie  = (cell_dir / "pie.png").exists()
                dist = (cell_dir / "distribution.png").exists()
                if pie or dist:
                    grid.setdefault(cls_key, {})[zone_key] = {"pie": pie, "distribution": dist}
        if grid:
            out["channel_activation"][method] = grid

    for method in ("saliency", "occlusion", "ablation"):
        method_dir = base / method
        if not method_dir.exists():
            continue
        for cls_dir in sorted(method_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            out[method][cls] = {}
            for zone_dir in sorted(cls_dir.iterdir()):
                if not zone_dir.is_dir():
                    continue
                zone = zone_dir.name
                pngs = sorted(f.name for f in zone_dir.glob("*.png"))
                if pngs:
                    out[method][cls][zone] = pngs
    return out


# ---------------------------------------------------------------------------
# NPZ loading + slice rendering
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _load_npz(model: str, case_id: str) -> tuple:
    """Load a case NPZ, searching fold_0..fold_4. Returns (data_dict, fold_number)."""
    for fold in range(5):
        path = XAI_DIR / model / f"fold_{fold}" / f"{case_id}.npz"
        if path.exists():
            raw = np.load(path, allow_pickle=True)
            return {k: raw[k] for k in raw.files}, fold
    raise FileNotFoundError(f"NPZ not found: {model}/{case_id}")


def _load_zones_pred(case_id: str, fold: int) -> np.ndarray | None:
    """Load umamba predicted zones (orientation-corrected crop) from ZONES_PRED_DIR.

    Returns zones_crop array (D_crop, W, H) int8 in the same orientation as model NPZ
    zones, or None if not available.
    Falls back to searching other folds if the primary fold file is missing.
    """
    path = ZONES_PRED_DIR / f"fold_{fold}" / f"{case_id}.npz"
    if not path.exists():
        for f in range(5):
            alt = ZONES_PRED_DIR / f"fold_{f}" / f"{case_id}.npz"
            if alt.exists():
                path = alt
                break
        else:
            return None
    raw = np.load(path, allow_pickle=True)
    if "zones_crop" in raw.files:
        return raw["zones_crop"]   # (D_crop, W, H) int8
    # Old-format file: only has full zones — return None and let caller fall back
    return None


def _is_sentinel(arr) -> bool:
    return arr is None or (isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size == 0)


# Non-sum aggregation methods: (name, npz_field_suffix).
# 'sum' uses no suffix (backward-compatible); rendered separately in the existing code path.
_AGG_NON_SUM = [("mean", "_mean"), ("abs_sum", "_abs_sum"), ("abs_avg", "_abs_avg")]


def _fix_zones(zones: np.ndarray, model: str) -> np.ndarray:
    """Swap zone labels 1↔2 for nnunet NPZ files that were saved with raw NIfTI
    encoding (1=TZ, 2=PZ) before the remapping fix in generate_xai_data.py."""
    if model != "nnunet" or _is_sentinel(zones):
        return zones
    fixed = zones.copy()
    fixed[zones == 1] = 2
    fixed[zones == 2] = 1
    return fixed


# Zone discrete colormap: 0=bg (dark), 1=PZ (blue), 2=TZ (red)
_ZONE_CMAP = mcolors.ListedColormap(["#000000", "#3a7bd5", "#d93025"])
_ZONE_NORM = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], _ZONE_CMAP.N)


def _arr_to_b64(arr2d: np.ndarray, cmap: str, vmin: float, vmax: float,
                is_zone: bool = False, transparent: bool = False) -> str:
    """Render a (H, W) numpy array as a base64 data-URL PNG (200×200 px).

    When transparent=True, renders an RGBA PNG where the alpha channel encodes
    "how much signal is here" — background/zero pixels are fully transparent so
    the client can composite the overlay on top of the MRI without any black bleed.
    """
    arr2d = np.ascontiguousarray(arr2d)
    if is_zone:
        rgba = _ZONE_CMAP(_ZONE_NORM(arr2d))
        rgba_arr = (rgba * 255).astype(np.uint8)
        if transparent:
            rgba_arr[:, :, 3] = ((arr2d > 0.5) * 255).astype(np.uint8)
    else:
        norm = np.clip((arr2d - vmin) / max(float(vmax - vmin), 1e-9), 0.0, 1.0)
        rgba = matplotlib.colormaps.get_cmap(cmap)(norm)
        rgba_arr = (rgba * 255).astype(np.uint8)
        if transparent:
            rgba_arr[:, :, 3] = (norm * 255).astype(np.uint8)
    if transparent:
        img = Image.fromarray(rgba_arr, "RGBA")
    else:
        img = Image.fromarray(rgba_arr[:, :, :3], "RGB")
    img = img.resize((200, 200), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _arr_to_pil_frame(arr2d: np.ndarray, cmap: str, vmin: float, vmax: float,
                      is_zone: bool = False, transparent: bool = False,
                      bg_arr2d: np.ndarray = None, overlay_alpha: float = 0.6) -> Image.Image:
    """Render a (H, W) array as a PIL RGB Image (200×200 px) for GIF assembly."""
    arr2d = np.ascontiguousarray(arr2d)
    if is_zone:
        rgba = _ZONE_CMAP(_ZONE_NORM(arr2d))
        rgba_arr = (rgba * 255).astype(np.uint8)
    else:
        norm = np.clip((arr2d - vmin) / max(float(vmax - vmin), 1e-9), 0.0, 1.0)
        rgba = matplotlib.colormaps.get_cmap(cmap)(norm)
        rgba_arr = (rgba * 255).astype(np.uint8)
    overlay_img = Image.fromarray(rgba_arr, "RGBA").resize((200, 200), Image.LANCZOS)

    if transparent and bg_arr2d is not None:
        bg_arr2d = np.ascontiguousarray(bg_arr2d)
        bv, bx = float(bg_arr2d.min()), float(bg_arr2d.max())
        bg_norm = np.clip((bg_arr2d - bv) / max(float(bx - bv), 1e-9), 0.0, 1.0)
        bg_rgba = matplotlib.colormaps.get_cmap("gray")(bg_norm)
        bg_rgb = (bg_rgba[:, :, :3] * 255).astype(np.uint8)
        bg_img = Image.fromarray(bg_rgb, "RGB").resize((200, 200), Image.LANCZOS).convert("RGBA")
        r, g, b, a = overlay_img.split()
        overlay_img.putalpha(Image.fromarray((np.array(a) * overlay_alpha).astype(np.uint8)))
        composite = bg_img.copy()
        composite.alpha_composite(overlay_img)
        return composite.convert("RGB")
    return overlay_img.convert("RGB")


def _render_all_slices(arr: np.ndarray, cmap: str, vmin: float, vmax: float,
                        is_zone: bool = False, transparent: bool = False) -> List[str]:
    """Render all depth slices of a (D, H, W) array as base64 data-URL PNGs."""
    return [_arr_to_b64(arr[sl], cmap, vmin, vmax, is_zone, transparent)
            for sl in range(arr.shape[0])]


def _build_case_payload(model: str, case_id: str) -> dict:
    """
    Load the NPZ for (model, case_id), render every depth slice for every panel
    as a base64 PNG, and return a JSON-serialisable dict.

    All rendering happens here — the client only swaps <img> src attributes.
    """
    npz, fold_found = _load_npz(model, case_id)

    image      = npz["image"]       # (3, D, H, W)
    pred       = npz["prediction"]  # (1, D, H, W)
    saliency   = npz.get("saliency",               np.zeros((0,), dtype=np.float32))
    ig         = npz.get("integrated_gradients",   np.zeros((0,), dtype=np.float32))
    occlusion  = npz.get("occlusion",              np.zeros((0,), dtype=np.float32))
    occ_tz     = npz.get("occlusion_tz",           np.zeros((0,), dtype=np.float32))
    occ_pz     = npz.get("occlusion_pz",           np.zeros((0,), dtype=np.float32))
    ablation      = npz.get("ablation",            np.zeros((0,), dtype=np.float32))
    zones_npz     = _fix_zones(npz.get("zones", np.zeros((0,), dtype=np.int8)), model)
    label         = npz.get("label",               np.zeros((0,), dtype=np.float32))
    inp_abl       = npz.get("input_ablation",      np.zeros((0,), dtype=np.float32))
    zm_baseline   = npz.get("zone_median_baseline", np.zeros((0,), dtype=np.float32))

    # Apply 90° CCW (left) rotation for MONAI models (swin_unetr, umamba_mtl)
    if model != "nnunet":
        def _rot_spatial(arr):
            if _is_sentinel(arr) or arr.ndim < 3:
                return arr
            return np.rot90(arr, k=1, axes=(arr.ndim - 2, arr.ndim - 1))
        image       = _rot_spatial(image)
        pred        = _rot_spatial(pred)
        saliency    = _rot_spatial(saliency)
        ig          = _rot_spatial(ig)
        occlusion   = _rot_spatial(occlusion)
        occ_tz      = _rot_spatial(occ_tz)
        occ_pz      = _rot_spatial(occ_pz)
        ablation    = _rot_spatial(ablation)
        zones_npz   = _rot_spatial(zones_npz)
        label       = _rot_spatial(label)
        zm_baseline = _rot_spatial(zm_baseline)

    # Zone display priority: umamba predictions > NPZ zones > error
    zones_error_msg: str | None = None
    zones_pred = _load_zones_pred(case_id, fold_found)
    if model != "nnunet" and zones_pred is not None and zones_pred.ndim == 3:
        zones_pred = np.rot90(zones_pred, k=1, axes=(1, 2))
    if zones_pred is not None and zones_pred.ndim == 3:
        zones = zones_pred    # (D_crop, W, H) orientation-corrected
    elif not _is_sentinel(zones_npz) and zones_npz.ndim == 3:
        zones = zones_npz     # fallback to model-native zones
    else:
        zones = None
        zones_error_msg = "Prostate zone predictions are missing for this case."

    # zones_npz (model-native space) is used for zone-median occlusion merging,
    # which must match the spatial dimensions of occ_tz/occ_pz
    zones_for_occlusion = zones_npz

    n_slices = image.shape[1]
    panels: dict = {}

    # MRI channels (gray, normalise per channel)
    for i, name in enumerate(("t2w", "adc", "hbv")):
        ch = image[i]  # (D, H, W)
        v0, v1 = float(ch.min()), float(ch.max())
        panels[name] = _render_all_slices(ch, "gray", v0, v1)

    # Zone-median baseline image (tiled zone patches used as occlusion baseline)
    if not _is_sentinel(zm_baseline) and zm_baseline.ndim == 4:
        for i, name in enumerate(("t2w", "adc", "hbv")):
            ch = zm_baseline[i]
            v0, v1 = float(ch.min()), float(ch.max())
            panels[f"zm_baseline_{name}"] = _render_all_slices(ch, "gray", v0, v1)

    # Cancer probability (transparent background for client-side compositing)
    prob = pred[0]  # (D, H, W)
    panels["prediction"] = _render_all_slices(prob, "turbo", 0.0, 1.0, transparent=True)

    # Ground-truth label
    if not _is_sentinel(label) and label.ndim == 4:
        panels["label"] = _render_all_slices(label[0], "turbo", 0.0, 1.0, transparent=True)

    # Cross-channel saliency: single global vmax across all 3 channels
    if not _is_sentinel(saliency) and saliency.ndim == 4:
        sal_abs = np.abs(saliency)  # (3, D, H, W)
        sal_vmax = float(np.percentile(sal_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"saliency_{i}"] = _render_all_slices(sal_abs[i], "turbo", 0.0, sal_vmax, transparent=True)

    # Integrated Gradients: same layout as saliency (3 channels, single global vmax)
    if not _is_sentinel(ig) and ig.ndim == 4:
        ig_abs = np.abs(ig)  # (3, D, H, W)
        ig_vmax = float(np.percentile(ig_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"integrated_gradients_{i}"] = _render_all_slices(ig_abs[i], "turbo", 0.0, ig_vmax, transparent=True)

    # Cross-channel occlusion: single global vmax across all 3 channels
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occ_abs = np.abs(occlusion)  # (3, D, H, W)
        occ_vmax = float(np.percentile(occ_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"occlusion_{i}"] = _render_all_slices(occ_abs[i], "turbo", 0.0, occ_vmax, transparent=True)

    # Zone-median occlusion: merge TZ and PZ attribution maps by zone mask
    # Uses zones_for_occlusion (model-native space) which matches occ_tz/occ_pz dimensions.
    occ_merged = None
    if (not _is_sentinel(occ_tz) and occ_tz.ndim == 4
            and not _is_sentinel(occ_pz) and occ_pz.ndim == 4
            and not _is_sentinel(zones_for_occlusion) and zones_for_occlusion.ndim == 3):
        occ_merged = np.zeros_like(occ_tz)
        occ_merged[:, zones_for_occlusion == 2] = occ_tz[:, zones_for_occlusion == 2]
        occ_merged[:, zones_for_occlusion == 1] = occ_pz[:, zones_for_occlusion == 1]
        occ_merged_abs = np.abs(occ_merged)
        occ_zm_vmax = float(np.percentile(occ_merged_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"occlusion_zm_{i}"] = _render_all_slices(
                occ_merged_abs[i], "turbo", 0.0, occ_zm_vmax, transparent=True
            )

    # TZ-masked and PZ-masked: zone attribution in its region, average of both elsewhere
    if (not _is_sentinel(occ_tz) and occ_tz.ndim == 4
            and not _is_sentinel(occ_pz) and occ_pz.ndim == 4
            and not _is_sentinel(zones_for_occlusion) and zones_for_occlusion.ndim == 3):
        occ_avg = (occ_tz + occ_pz) / 2.0

        occ_tz_masked = occ_avg.copy()
        occ_tz_masked[:, zones_for_occlusion == 2] = occ_tz[:, zones_for_occlusion == 2]
        occ_tz_masked_abs = np.abs(occ_tz_masked)
        occ_tz_vmax = float(np.percentile(occ_tz_masked_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"occlusion_tz_masked_{i}"] = _render_all_slices(
                occ_tz_masked_abs[i], "turbo", 0.0, occ_tz_vmax, transparent=True)

        occ_pz_masked = occ_avg.copy()
        occ_pz_masked[:, zones_for_occlusion == 1] = occ_pz[:, zones_for_occlusion == 1]
        occ_pz_masked_abs = np.abs(occ_pz_masked)
        occ_pz_vmax = float(np.percentile(occ_pz_masked_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"occlusion_pz_masked_{i}"] = _render_all_slices(
                occ_pz_masked_abs[i], "turbo", 0.0, occ_pz_vmax, transparent=True)

    # AblationCAM (single-channel spatial map)
    if not _is_sentinel(ablation) and ablation.ndim == 4:
        abl_map = ablation[0]  # (D, H, W)
        v1 = float(np.percentile(abl_map, 99)) or 1e-6
        panels["ablation"] = _render_all_slices(abl_map, "turbo", 0.0, v1, transparent=True)

    # Zones (discrete colormap)
    if zones is not None and zones.ndim == 3:
        panels["zones"] = _render_all_slices(zones.astype(np.float32), "", 0, 2,
                                              is_zone=True, transparent=True)

    # Render panels for non-sum aggregation methods (appended to existing .npz files).
    # For MONAI models the same 90° CCW rotation must be applied.
    def _rot_maybe(arr):
        if model == "nnunet" or _is_sentinel(arr) or arr.ndim < 3:
            return arr
        return np.rot90(arr, k=1, axes=(arr.ndim - 2, arr.ndim - 1))

    available_aggregations = []
    # sum is available if any of its panels were already rendered
    if (any(f"saliency_{i}" in panels for i in range(3)) or "ablation" in panels or
            any(f"occlusion_{i}" in panels for i in range(3)) or
            any(f"integrated_gradients_{i}" in panels for i in range(3))):
        available_aggregations.append("sum")

    for agg_name, sfx in _AGG_NON_SUM:
        sal_agg   = _rot_maybe(npz.get(f"saliency{sfx}",              np.zeros((0,), dtype=np.float32)))
        ig_agg    = _rot_maybe(npz.get(f"integrated_gradients{sfx}",  np.zeros((0,), dtype=np.float32)))
        occ_agg   = _rot_maybe(npz.get(f"occlusion{sfx}",             np.zeros((0,), dtype=np.float32)))
        occ_tz_agg = _rot_maybe(npz.get(f"occlusion_tz{sfx}",         np.zeros((0,), dtype=np.float32)))
        occ_pz_agg = _rot_maybe(npz.get(f"occlusion_pz{sfx}",         np.zeros((0,), dtype=np.float32)))
        abl_agg   = _rot_maybe(npz.get(f"ablation{sfx}",              np.zeros((0,), dtype=np.float32)))

        has_any = ((not _is_sentinel(sal_agg) and sal_agg.ndim == 4) or
                   (not _is_sentinel(ig_agg)  and ig_agg.ndim == 4)  or
                   (not _is_sentinel(occ_agg) and occ_agg.ndim == 4) or
                   (not _is_sentinel(abl_agg) and abl_agg.ndim == 4))
        if not has_any:
            continue
        available_aggregations.append(agg_name)

        if not _is_sentinel(sal_agg) and sal_agg.ndim == 4:
            sal_abs = np.abs(sal_agg)
            sal_vmax = float(np.percentile(sal_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"saliency{sfx}_{i}"] = _render_all_slices(sal_abs[i], "turbo", 0.0, sal_vmax, transparent=True)

        if not _is_sentinel(ig_agg) and ig_agg.ndim == 4:
            ig_agg_abs = np.abs(ig_agg)
            ig_agg_vmax = float(np.percentile(ig_agg_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"integrated_gradients{sfx}_{i}"] = _render_all_slices(ig_agg_abs[i], "turbo", 0.0, ig_agg_vmax, transparent=True)

        if not _is_sentinel(occ_agg) and occ_agg.ndim == 4:
            occ_abs = np.abs(occ_agg)
            occ_vmax = float(np.percentile(occ_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"occlusion{sfx}_{i}"] = _render_all_slices(occ_abs[i], "turbo", 0.0, occ_vmax, transparent=True)

        if (not _is_sentinel(occ_tz_agg) and occ_tz_agg.ndim == 4
                and not _is_sentinel(occ_pz_agg) and occ_pz_agg.ndim == 4
                and not _is_sentinel(zones_for_occlusion) and zones_for_occlusion.ndim == 3):
            _occ_merged = np.zeros_like(occ_tz_agg)
            _occ_merged[:, zones_for_occlusion == 2] = occ_tz_agg[:, zones_for_occlusion == 2]
            _occ_merged[:, zones_for_occlusion == 1] = occ_pz_agg[:, zones_for_occlusion == 1]
            _occ_merged_abs = np.abs(_occ_merged)
            _zm_vmax = float(np.percentile(_occ_merged_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"occlusion_zm{sfx}_{i}"] = _render_all_slices(_occ_merged_abs[i], "turbo", 0.0, _zm_vmax, transparent=True)

            _occ_avg = (occ_tz_agg + occ_pz_agg) / 2.0

            _tz_m = _occ_avg.copy()
            _tz_m[:, zones_for_occlusion == 2] = occ_tz_agg[:, zones_for_occlusion == 2]
            _tz_m_abs = np.abs(_tz_m)
            _tz_vmax = float(np.percentile(_tz_m_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"occlusion_tz_masked{sfx}_{i}"] = _render_all_slices(_tz_m_abs[i], "turbo", 0.0, _tz_vmax, transparent=True)

            _pz_m = _occ_avg.copy()
            _pz_m[:, zones_for_occlusion == 1] = occ_pz_agg[:, zones_for_occlusion == 1]
            _pz_m_abs = np.abs(_pz_m)
            _pz_vmax = float(np.percentile(_pz_m_abs, 99)) or 1e-6
            for i in range(3):
                panels[f"occlusion_pz_masked{sfx}_{i}"] = _render_all_slices(_pz_m_abs[i], "turbo", 0.0, _pz_vmax, transparent=True)

        if not _is_sentinel(abl_agg) and abl_agg.ndim == 4:
            abl_map = abl_agg[0]
            v1 = float(np.percentile(abl_map, 99)) or 1e-6
            panels[f"ablation{sfx}"] = _render_all_slices(abl_map, "turbo", 0.0, v1, transparent=True)

    # Compute per-channel means on-the-fly from NPZ (not stored in progress.json)
    sal_ch_mean = (np.abs(saliency).mean(axis=(1, 2, 3)).tolist()
                   if not _is_sentinel(saliency) and saliency.ndim == 4 else None)
    ig_ch_mean  = (np.abs(ig).mean(axis=(1, 2, 3)).tolist()
                   if not _is_sentinel(ig) and ig.ndim == 4 else None)
    occ_for_stats = (occlusion if not _is_sentinel(occlusion) and occlusion.ndim == 4
                     else occ_merged)
    occ_ch_mean = (np.abs(occ_for_stats).mean(axis=(1, 2, 3)).tolist()
                   if occ_for_stats is not None else None)

    # Per-strategy occlusion stats computed from NPZ arrays
    def _occ_stats(arr: np.ndarray) -> dict:
        abs_arr = np.abs(arr)
        ch_sum = abs_arr.sum(axis=(1, 2, 3))
        total = float(ch_sum.sum())
        return {
            "ch_fraction": (ch_sum / total).tolist() if total > 0 else [0.0, 0.0, 0.0],
            "ch_mean": abs_arr.mean(axis=(1, 2, 3)).tolist(),
        }

    occ_stats_by_strategy: dict = {}
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occ_stats_by_strategy["zero"] = _occ_stats(occlusion)
    if occ_merged is not None:
        occ_stats_by_strategy["zone_median"] = _occ_stats(occ_merged)
    if "occlusion_tz_masked_0" in panels and not _is_sentinel(occ_tz) and occ_tz.ndim == 4 \
            and not _is_sentinel(occ_pz) and occ_pz.ndim == 4 \
            and not _is_sentinel(zones_for_occlusion) and zones_for_occlusion.ndim == 3:
        occ_avg = (occ_tz + occ_pz) / 2.0
        _tz_m = occ_avg.copy(); _tz_m[:, zones_for_occlusion == 2] = occ_tz[:, zones_for_occlusion == 2]
        _pz_m = occ_avg.copy(); _pz_m[:, zones_for_occlusion == 1] = occ_pz[:, zones_for_occlusion == 1]
        occ_stats_by_strategy["tz_masked"] = _occ_stats(_tz_m)
        occ_stats_by_strategy["pz_masked"] = _occ_stats(_pz_m)

    # Detect which occlusion strategies are available for this case
    occlusion_strategies = []
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occlusion_strategies.append("zero")
    if "occlusion_zm_0" in panels:
        occlusion_strategies.append("zone_median")
    if "occlusion_tz_masked_0" in panels:
        occlusion_strategies.append("tz_masked")
    if "occlusion_pz_masked_0" in panels:
        occlusion_strategies.append("pz_masked")

    # Stats: pull from in-memory sample_data (refreshed periodically)
    record = _get_case_index().get(model, {}).get(case_id, {})

    def _enrich(stat, ch_mean):
        if ch_mean is None:
            return stat
        if stat:
            return {**stat, "ch_mean": ch_mean}
        return {"ch_mean": ch_mean}

    # Merge NPZ-computed per-strategy stats over the progress.json base
    merged_occ_by_strat = {**record.get("occlusion_by_strategy", {})}
    for _strat, _npz_stats in occ_stats_by_strategy.items():
        merged_occ_by_strat[_strat] = {**merged_occ_by_strat.get(_strat, {}), **_npz_stats}

    stats = {
        "input_ablation":          inp_abl.tolist() if (not _is_sentinel(inp_abl) and inp_abl.shape == (3,)) else None,
        "saliency":                _enrich(record.get("saliency"), sal_ch_mean),
        "integrated_gradients":    _enrich(record.get("integrated_gradients"), ig_ch_mean),
        "occlusion":               _enrich(record.get("occlusion"), occ_ch_mean),
        "occlusion_by_strategy":   merged_occ_by_strat,
        "pz_voxels":               record.get("pz_voxels"),
        "tz_voxels":               record.get("tz_voxels"),
        "pred_pz_voxels":          record.get("pred_pz_voxels"),
        "pred_tz_voxels":          record.get("pred_tz_voxels"),
        "confidence":              record.get("confidence"),
        "pred_max_prob":           record.get("pred_max_prob"),
        "ablation_ch_fraction":    record.get("ablation_ch_fraction"),
        "ig_ch_fraction":          record.get("ig_ch_fraction"),
    }

    return {"n_slices": n_slices, "panels": panels, "stats": stats, "record": record,
            "occlusion_strategies": occlusion_strategies, "zones_error_msg": zones_error_msg,
            "available_aggregations": available_aggregations}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    stats = {m: _model_stats(m) for m in MODELS}
    return render_template("index.html", models=MODELS, stats=stats)


@app.route("/model/<model>")
def model_view(model: str):
    if model not in MODELS:
        abort(404)
    case_data = _get_case_data()
    records = case_data.get(model, [])
    stats   = _model_stats(model)
    charts  = _available_charts(model)

    # Case IDs predicted positive by every model
    pos_sets = [
        {r["case_id"] for r in case_data.get(m, []) if r.get("predicted_positive")}
        for m in MODELS
    ]
    all_models_pos: set = pos_sets[0].intersection(*pos_sets[1:]) if pos_sets else set()

    # Case IDs classified as TP / FP by every model
    tp_sets = [
        {r["case_id"] for r in case_data.get(m, []) if r.get("classification") == "tp"}
        for m in MODELS
    ]
    fp_sets = [
        {r["case_id"] for r in case_data.get(m, []) if r.get("classification") == "fp"}
        for m in MODELS
    ]
    all_models_tp: set = tp_sets[0].intersection(*tp_sets[1:]) if tp_sets else set()
    all_models_fp: set = fp_sets[0].intersection(*fp_sets[1:]) if fp_sets else set()

    return render_template("model.html", model=model, records=records,
                           stats=stats, charts=charts,
                           all_models_pos=all_models_pos,
                           all_models_tp=all_models_tp,
                           all_models_fp=all_models_fp)


@app.route("/model/<model>/case/<case_id>")
def case_view(model: str, case_id: str):
    if model not in MODELS:
        abort(404)
    case_index = _get_case_index()
    record = case_index.get(model, {}).get(case_id)
    all_model_records = {m: case_index.get(m, {}).get(case_id) for m in MODELS}
    return render_template("case.html", model=model, case_id=case_id,
                           fold=record["fold"] if record else "?",
                           record=record,
                           all_model_records=all_model_records)


# --- API endpoints ----------------------------------------------------------

@app.route("/api/case/<model>/<case_id>.json")
def api_case(model: str, case_id: str):
    """Return all rendered slices + stats for a case. Heavy — cached after first call."""
    if model not in MODELS:
        abort(404)
    try:
        payload = _build_case_payload(model, case_id)
        return jsonify(payload)
    except FileNotFoundError:
        abort(404)


@app.route("/api/cases/<model>.json")
def api_cases(model: str):
    """Return the in-memory sample_data list for a model."""
    if model not in MODELS:
        abort(404)
    return jsonify(_get_case_data().get(model, []))


@app.route("/api/gif/<model>/<case_id>")
def api_gif(model: str, case_id: str):
    """
    Generate and stream an animated GIF for one panel of a case.

    Query params: panel, bg (t2w/adc/hbv/none), alpha (0-1), duration_ms
    """
    if model not in MODELS:
        abort(404)

    panel       = request.args.get("panel", "t2w")
    bg_key      = request.args.get("bg", "t2w")
    alpha       = float(request.args.get("alpha", 0.6))
    duration_ms = int(request.args.get("duration_ms", 120))

    try:
        npz, _fold = _load_npz(model, case_id)
    except FileNotFoundError:
        abort(404)

    image = npz["image"]  # (3, D, H, W)

    def get_panel_data():
        if panel == "t2w":   return image[0], "gray", None, None, False, False
        if panel == "adc":   return image[1], "gray", None, None, False, False
        if panel == "hbv":   return image[2], "gray", None, None, False, False
        if panel == "prediction":
            p = npz.get("prediction", np.zeros((0,), dtype=np.float32))
            return (None,) * 6 if _is_empty(p) else (p[0], "turbo", 0.0, 1.0, False, True)
        if panel.startswith("saliency"):
            s = npz.get("saliency", np.zeros((0,), dtype=np.float32))
            if _is_empty(s) or s.ndim != 4: return (None,) * 6
            sal_abs = np.abs(s)  # (3, D, H, W)
            global_vmax = float(np.percentile(sal_abs, 99)) or 1e-6
            idx = int(panel[-1]) if panel[-1].isdigit() else -1
            arr = sal_abs[idx] if idx >= 0 else sal_abs.mean(axis=0)
            return arr, "turbo", 0.0, global_vmax, False, True
        if panel.startswith("occlusion"):
            o = npz.get("occlusion", np.zeros((0,), dtype=np.float32))
            if _is_empty(o) or o.ndim != 4: return (None,) * 6
            occ_abs = np.abs(o)  # (3, D, H, W)
            global_vmax = float(np.percentile(occ_abs, 99)) or 1e-6
            idx = int(panel[-1]) if panel[-1].isdigit() else -1
            arr = occ_abs[idx] if idx >= 0 else occ_abs.mean(axis=0)
            return arr, "turbo", 0.0, global_vmax, False, True
        if panel == "ablation":
            ab = npz.get("ablation", np.zeros((0,), dtype=np.float32))
            if _is_empty(ab) or ab.ndim != 4: return (None,) * 6
            arr = ab[0]
            return arr, "turbo", 0.0, float(np.percentile(arr, 99)) or 1e-6, False, True
        if panel == "zones":
            z = _fix_zones(npz.get("zones", np.zeros((0,), dtype=np.int8)), model)
            return (None,) * 6 if _is_empty(z) else (z.astype(np.float32), "", 0, 2, True, True)
        return (None,) * 6

    arr, cmap, vmin, vmax, is_zone, is_overlay = get_panel_data()
    if arr is None:
        abort(404)

    if vmin is None: vmin = float(arr.min())
    if vmax is None: vmax = float(np.percentile(arr, 99)) or 1e-6

    # Auto-pick bg from panel name for per-channel panels
    if panel[-1].isdigit() and bg_key not in ("t2w", "adc", "hbv"):
        bg_key = ("t2w", "adc", "hbv")[int(panel[-1])]

    bg_arr = None
    if is_overlay and bg_key in ("t2w", "adc", "hbv"):
        bg_arr = image[{"t2w": 0, "adc": 1, "hbv": 2}[bg_key]]  # (D, H, W)

    n = arr.shape[0]
    frames = [
        _arr_to_pil_frame(
            arr[i], cmap, vmin, vmax,
            is_zone=is_zone, transparent=is_overlay,
            bg_arr2d=bg_arr[i] if bg_arr is not None else None,
            overlay_alpha=alpha,
        )
        for i in range(n)
    ]

    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   loop=0, duration=duration_ms, optimize=False)
    out.seek(0)
    return send_file(out, mimetype="image/gif", download_name=f"{case_id}_{panel}.gif")


@app.route("/api/gif_table/<model>/<case_id>")
def api_gif_table(model: str, case_id: str):
    """
    Generate an animated GIF of the full 3-row × 7-col viewer table.
    Each frame is one depth slice. All XAI panels use cross-channel normalization.

    Query params: alpha (0-1), duration_ms
    """
    if model not in MODELS:
        abort(404)

    alpha       = float(request.args.get("alpha", 0.6))
    duration_ms = int(request.args.get("duration_ms", 120))

    try:
        npz, _fold = _load_npz(model, case_id)
    except FileNotFoundError:
        abort(404)

    image     = npz["image"]                                           # (3, D, H, W)
    pred      = npz.get("prediction", np.zeros((0,), dtype=np.float32))
    saliency  = npz.get("saliency",   np.zeros((0,), dtype=np.float32))
    occlusion = npz.get("occlusion",  np.zeros((0,), dtype=np.float32))
    ablation  = npz.get("ablation",   np.zeros((0,), dtype=np.float32))
    zones     = _fix_zones(npz.get("zones", np.zeros((0,), dtype=np.int8)), model)
    label     = npz.get("label",      np.zeros((0,), dtype=np.float32))

    # Cross-channel normalization (same logic as _build_case_payload)
    sal_abs = occ_abs = None
    sal_vmax = occ_vmax = abl_vmax = None
    if not _is_sentinel(saliency) and saliency.ndim == 4:
        sal_abs  = np.abs(saliency)
        sal_vmax = float(np.percentile(sal_abs, 99)) or 1e-6
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occ_abs  = np.abs(occlusion)
        occ_vmax = float(np.percentile(occ_abs, 99)) or 1e-6
    if not _is_sentinel(ablation) and ablation.ndim == 4:
        abl_vmax = float(np.percentile(ablation[0], 99)) or 1e-6

    n_depth = image.shape[1]

    # Layout constants
    CELL, GAP, L_MAR, T_MAR = 128, 4, 60, 30
    N_ROWS, N_COLS = 3, 7
    COL_LABELS = ["Original", "Zones", "Pred Mask", "Label GT",
                  "Saliency", "Occlusion", "AblationCAM"]
    ROW_LABELS = ["T2W", "ADC", "HBV"]
    total_w = L_MAR + N_COLS * CELL + (N_COLS - 1) * GAP
    total_h = T_MAR + N_ROWS * CELL + (N_ROWS - 1) * GAP

    # Font (best-effort with fallbacks)
    from PIL import ImageFont as _IF
    _font = None
    for _fp in ["/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf"]:
        try:
            _font = _IF.truetype(_fp, 11)
            break
        except Exception:
            pass
    if _font is None:
        _font = _IF.load_default()

    def _cx(ci: int) -> int: return L_MAR + ci * (CELL + GAP)
    def _cy(ri: int) -> int: return T_MAR + ri * (CELL + GAP)

    def _rc(arr2d, cmap, vmin, vmax, is_zone, bg):
        """Render one cell as a CELL×CELL PIL Image."""
        img = _arr_to_pil_frame(arr2d, cmap, vmin, vmax,
                                is_zone=is_zone, transparent=(bg is not None),
                                bg_arr2d=bg, overlay_alpha=alpha)
        return img.resize((CELL, CELL), Image.LANCZOS)

    frames = []
    for sl in range(n_depth):
        frame = Image.new("RGB", (total_w, total_h), (15, 23, 42))
        draw  = ImageDraw.Draw(frame)

        # Column headers
        for ci, lbl in enumerate(COL_LABELS):
            draw.text((_cx(ci) + CELL // 2, T_MAR // 2), lbl,
                      fill=(148, 163, 184), font=_font, anchor="mm")

        for ri in range(N_ROWS):
            bg_ch = image[ri, sl]  # (H, W)
            v0_bg = float(bg_ch.min())
            v1_bg = float(bg_ch.max())

            # Row label
            draw.text((L_MAR // 2, _cy(ri) + CELL // 2), ROW_LABELS[ri],
                      fill=(148, 163, 184), font=_font, anchor="mm")

            for ci in range(N_COLS):
                cell_img = None

                if ci == 0:   # Original MRI
                    cell_img = _rc(bg_ch, "gray", v0_bg, v1_bg, False, None)

                elif ci == 1:  # Zones
                    if not _is_sentinel(zones) and zones.ndim == 3:
                        cell_img = _rc(zones[sl].astype(np.float32), "", 0, 2, True, bg_ch)

                elif ci == 2:  # Prediction
                    if not _is_sentinel(pred) and pred.ndim == 4:
                        cell_img = _rc(pred[0, sl], "turbo", 0.0, 1.0, False, bg_ch)

                elif ci == 3:  # Label GT
                    if not _is_sentinel(label) and label.ndim == 4:
                        cell_img = _rc(label[0, sl], "turbo", 0.0, 1.0, False, bg_ch)

                elif ci == 4:  # Saliency
                    if sal_vmax is not None:
                        cell_img = _rc(sal_abs[ri, sl], "turbo", 0.0, sal_vmax, False, bg_ch)

                elif ci == 5:  # Occlusion
                    if occ_vmax is not None:
                        cell_img = _rc(occ_abs[ri, sl], "turbo", 0.0, occ_vmax, False, bg_ch)

                elif ci == 6:  # AblationCAM (same map for all rows)
                    if abl_vmax is not None:
                        cell_img = _rc(ablation[0, sl], "turbo", 0.0, abl_vmax, False, bg_ch)

                if cell_img is not None:
                    frame.paste(cell_img, (_cx(ci), _cy(ri)))
                else:
                    draw.rectangle([_cx(ci), _cy(ri),
                                    _cx(ci) + CELL - 1, _cy(ri) + CELL - 1],
                                   fill=(17, 24, 39))

        frames.append(frame)

    out = io.BytesIO()
    frames[0].save(out, format="GIF", save_all=True, append_images=frames[1:],
                   loop=0, duration=duration_ms, optimize=False)
    out.seek(0)
    return send_file(out, mimetype="image/gif", download_name=f"{case_id}_table.gif")


@app.route("/api/analysis/<model>/<path:relpath>")
def api_analysis_chart(model: str, relpath: str):
    """Serve a pre-generated analysis PNG chart from ANALYSIS_DIR."""
    if model not in MODELS:
        abort(404)
    path = ANALYSIS_DIR / model / relpath
    if not path.exists() or path.suffix != ".png":
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/api/chart/<model>/<path:relpath>")
def api_chart(model: str, relpath: str):
    """Serve a pre-generated PNG chart."""
    if model not in MODELS:
        abort(404)
    path = METRICS_DIR / model / relpath
    if not path.exists() or path.suffix != ".png":
        abort(404)
    return send_file(path, mimetype="image/png")


@app.route("/api/confusion_matrix/<model>")
def api_confusion_matrix(model: str):
    """Render a confusion matrix PNG dynamically from live stats."""
    if model not in MODELS:
        abort(404)
    s = _model_stats(model)
    tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]

    matrix = np.array([[tp, fp], [fn, tn]], dtype=int)
    labels = [["TP", "FP"], ["FN", "TN"]]

    fig, ax = plt.subplots(figsize=(3.2, 2.8), dpi=180)
    vmax = max(matrix.max(), 1)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            brightness = val / vmax
            color = "white" if brightness > 0.55 else "#1e293b"
            ax.text(j, i, f"{labels[i][j]}\n{val}",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred +", "Pred −"], fontsize=9)
    ax.set_yticklabels(["Actual +", "Actual −"], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PI-CAI XAI Visualization Server")
    parser.add_argument("--host",  default="127.0.0.1", help="Bind host")
    parser.add_argument("--port",  default=5000, type=int, help="Bind port")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    total = sum(len(v) for v in _get_case_data().values())
    print(f"Loaded {total} cases across {len(MODELS)} models (refreshes every {_CACHE_TTL}s)")
    print(f"XAI NPZ directory: {XAI_DIR}")
    print(f"Starting server at http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
