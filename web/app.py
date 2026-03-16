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
MODELS       = ["umamba_mtl", "swin_unetr", "nnunet"]

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
                return {"ch_fraction": frac, "dominant_ch": int(np.argmax(frac))}

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
        return dict(
            tp=s["tp"], fp=s["fp"], fn=s["fn"], tn=s["tn"],
            total=s["total_cases"],
            precision=round(s["precision"], 3),
            sensitivity=round(s["sensitivity"], 3),
            specificity=round(s.get("specificity", 0.0), 3),
            f1=round(s["f1"], 3),
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
    Structure: {summary: [fname], saliency: {class: {zone: [fname]}}, occlusion: ..., ablation: ...}
    """
    base = METRICS_DIR / model
    out: dict = {"summary": [], "saliency": {}, "occlusion": {}, "ablation": {}}

    for fname in ("zone_distribution.png", "overall_channel_activation.png"):
        if (base / "summary" / fname).exists():
            out["summary"].append(fname)

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
def _load_npz(model: str, case_id: str) -> dict:
    """Load a case NPZ, searching fold_0..fold_4. Result is cached."""
    for fold in range(5):
        path = XAI_DIR / model / f"fold_{fold}" / f"{case_id}.npz"
        if path.exists():
            raw = np.load(path, allow_pickle=True)
            return {k: raw[k] for k in raw.files}
    raise FileNotFoundError(f"NPZ not found: {model}/{case_id}")


def _is_sentinel(arr) -> bool:
    return arr is None or (isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.size == 0)


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
    if transparent:
        if is_zone:
            # Discrete zone colormap; alpha = 255 for PZ/TZ, 0 for background
            rgba = _ZONE_CMAP(_ZONE_NORM(arr2d))           # (H, W, 4) float64 [0,1]
            rgba_arr = (rgba * 255).astype(np.uint8)
            rgba_arr[:, :, 3] = ((arr2d > 0.5) * 255).astype(np.uint8)
        else:
            # Continuous colormap: alpha ∝ normalised value so near-zero → transparent
            norm = np.clip((arr2d - vmin) / max(float(vmax - vmin), 1e-9), 0.0, 1.0)
            rgba = plt.cm.get_cmap(cmap)(norm)             # (H, W, 4) float64 [0,1]
            rgba_arr = (rgba * 255).astype(np.uint8)
            rgba_arr[:, :, 3] = (norm * 255).astype(np.uint8)
        img = Image.fromarray(rgba_arr, "RGBA").resize((200, 200), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    # Opaque render via matplotlib (used for MRI channels)
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    if is_zone:
        ax.imshow(arr2d, cmap=_ZONE_CMAP, norm=_ZONE_NORM, interpolation="nearest")
    else:
        ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _arr_to_pil_frame(arr2d: np.ndarray, cmap: str, vmin: float, vmax: float,
                      is_zone: bool = False, transparent: bool = False,
                      bg_arr2d: np.ndarray = None, overlay_alpha: float = 0.6) -> Image.Image:
    """Render a (H, W) array as a PIL RGB Image (200×200 px) for GIF assembly."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.axis("off")
    fig.patch.set_facecolor("black")
    if is_zone:
        ax.imshow(arr2d, cmap=_ZONE_CMAP, norm=_ZONE_NORM, interpolation="nearest")
    else:
        ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor="black", transparent=False)
    plt.close(fig)
    buf.seek(0)
    overlay_img = Image.open(buf).convert("RGBA")

    if transparent and bg_arr2d is not None:
        fig2, ax2 = plt.subplots(figsize=(2, 2), dpi=100)
        ax2.axis("off")
        fig2.patch.set_facecolor("black")
        bv, bx = float(bg_arr2d.min()), float(bg_arr2d.max())
        ax2.imshow(bg_arr2d, cmap="gray", vmin=bv, vmax=bx, interpolation="nearest")
        buf2 = io.BytesIO()
        plt.savefig(buf2, format="png", bbox_inches="tight", pad_inches=0, facecolor="black")
        plt.close(fig2)
        buf2.seek(0)
        bg_img = Image.open(buf2).convert("RGBA")
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
    npz = _load_npz(model, case_id)

    image      = npz["image"]       # (3, D, H, W)
    pred       = npz["prediction"]  # (1, D, H, W)
    saliency   = npz.get("saliency",       np.zeros((0,), dtype=np.float32))
    occlusion  = npz.get("occlusion",      np.zeros((0,), dtype=np.float32))
    ablation   = npz.get("ablation",       np.zeros((0,), dtype=np.float32))
    zones      = _fix_zones(npz.get("zones", np.zeros((0,), dtype=np.int8)), model)
    label      = npz.get("label",          np.zeros((0,), dtype=np.float32))
    inp_abl    = npz.get("input_ablation", np.zeros((0,), dtype=np.float32))

    n_slices = image.shape[1]
    panels: dict = {}

    # MRI channels (gray, normalise per channel)
    for i, name in enumerate(("t2w", "adc", "hbv")):
        ch = image[i]  # (D, H, W)
        v0, v1 = float(ch.min()), float(ch.max())
        panels[name] = _render_all_slices(ch, "gray", v0, v1)

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

    # Cross-channel occlusion: single global vmax across all 3 channels
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occ_abs = np.abs(occlusion)  # (3, D, H, W)
        occ_vmax = float(np.percentile(occ_abs, 99)) or 1e-6
        for i in range(3):
            panels[f"occlusion_{i}"] = _render_all_slices(occ_abs[i], "turbo", 0.0, occ_vmax, transparent=True)

    # AblationCAM (single-channel spatial map)
    if not _is_sentinel(ablation) and ablation.ndim == 4:
        abl_map = ablation[0]  # (D, H, W)
        v1 = float(np.percentile(abl_map, 99)) or 1e-6
        panels["ablation"] = _render_all_slices(abl_map, "turbo", 0.0, v1, transparent=True)

    # Zones (discrete colormap)
    if not _is_sentinel(zones) and zones.ndim == 3:
        panels["zones"] = _render_all_slices(zones.astype(np.float32), "", 0, 2,
                                              is_zone=True, transparent=True)

    # Compute per-channel means on-the-fly from NPZ (not stored in progress.json)
    sal_ch_mean = (np.abs(saliency).mean(axis=(1, 2, 3)).tolist()
                   if not _is_sentinel(saliency) and saliency.ndim == 4 else None)
    occ_ch_mean = (np.abs(occlusion).mean(axis=(1, 2, 3)).tolist()
                   if not _is_sentinel(occlusion) and occlusion.ndim == 4 else None)

    # Stats: pull from in-memory sample_data (refreshed periodically)
    record = _get_case_index().get(model, {}).get(case_id, {})

    def _enrich(stat, ch_mean):
        if ch_mean is None:
            return stat
        if stat:
            return {**stat, "ch_mean": ch_mean}
        return {"ch_mean": ch_mean}

    stats = {
        "input_ablation":      inp_abl.tolist() if (not _is_sentinel(inp_abl) and inp_abl.shape == (3,)) else None,
        "saliency":            _enrich(record.get("saliency"), sal_ch_mean),
        "occlusion":           _enrich(record.get("occlusion"), occ_ch_mean),
        "pz_voxels":           record.get("pz_voxels"),
        "tz_voxels":           record.get("tz_voxels"),
        "pred_pz_voxels":      record.get("pred_pz_voxels"),
        "pred_tz_voxels":      record.get("pred_tz_voxels"),
        "confidence":          record.get("confidence"),
        "pred_max_prob":       record.get("pred_max_prob"),
        "ablation_ch_fraction": record.get("ablation_ch_fraction"),
    }

    return {"n_slices": n_slices, "panels": panels, "stats": stats, "record": record}


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
    records = _get_case_data().get(model, [])
    stats   = _model_stats(model)
    charts  = _available_charts(model)
    return render_template("model.html", model=model, records=records,
                           stats=stats, charts=charts)


@app.route("/model/<model>/case/<case_id>")
def case_view(model: str, case_id: str):
    if model not in MODELS:
        abort(404)
    record = _get_case_index().get(model, {}).get(case_id)
    return render_template("case.html", model=model, case_id=case_id,
                           fold=record["fold"] if record else "?",
                           record=record)


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
        npz = _load_npz(model, case_id)
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
        npz = _load_npz(model, case_id)
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
