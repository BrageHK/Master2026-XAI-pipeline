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
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from flask import Flask, abort, jsonify, render_template, send_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
METRICS_DIR  = PROJECT_ROOT / "results" / "metrics"
XAI_DIR      = PROJECT_ROOT / "results" / "xai"
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
    """Load from sample_data.json if available, else scan NPZ files directly."""
    data: Dict[str, List[dict]] = {}
    for model in MODELS:
        path = METRICS_DIR / model / "sample_data.json"
        if path.exists():
            with open(path) as f:
                data[model] = json.load(f)
        else:
            data[model] = _scan_npz_records(model)
    return data


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
    records = _get_case_data().get(model, [])
    counts  = {c: sum(1 for r in records if r["classification"] == c)
               for c in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    total       = tp + fp + fn + tn
    precision   = tp / (tp + fp)         if (tp + fp) > 0     else 0.0
    sensitivity = tp / (tp + fn)         if (tp + fn) > 0     else 0.0
    specificity = tn / (tn + fp)         if (tn + fp) > 0     else 0.0
    f1          = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) > 0 else 0.0
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

    for fname in ("confusion_matrix.png", "zone_distribution.png",
                  "overall_channel_activation.png"):
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


# Zone discrete colormap: 0=bg (dark), 1=PZ (blue), 2=TZ (red)
_ZONE_CMAP = mcolors.ListedColormap(["#1a1a1a", "#3a7bd5", "#d93025"])
_ZONE_NORM = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5], _ZONE_CMAP.N)


def _arr_to_b64(arr2d: np.ndarray, cmap: str, vmin: float, vmax: float,
                is_zone: bool = False, transparent: bool = False) -> str:
    """Render a (H, W) numpy array as a base64 data-URL PNG (200×200 px)."""
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    ax.axis("off")
    if transparent:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor("black")
    if is_zone:
        ax.imshow(arr2d, cmap=_ZONE_CMAP, norm=_ZONE_NORM, interpolation="nearest")
    else:
        ax.imshow(arr2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor="none" if transparent else "black",
                transparent=transparent)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


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
    zones      = npz.get("zones",          np.zeros((0,), dtype=np.int8))
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
    panels["prediction"] = _render_all_slices(prob, "hot", 0.0, 1.0, transparent=True)

    # Ground-truth label
    if not _is_sentinel(label) and label.ndim == 4:
        panels["label"] = _render_all_slices(label[0], "hot", 0.0, 1.0, transparent=True)

    # Saliency (absolute mean across channels)
    if not _is_sentinel(saliency) and saliency.ndim == 4:
        sal_mean = np.abs(saliency).mean(axis=0)  # (D, H, W)
        v1 = float(np.percentile(sal_mean, 99)) or 1e-6
        panels["saliency"] = _render_all_slices(sal_mean, "viridis", 0.0, v1, transparent=True)

    # Occlusion (absolute mean across channels)
    if not _is_sentinel(occlusion) and occlusion.ndim == 4:
        occ_mean = np.abs(occlusion).mean(axis=0)  # (D, H, W)
        v1 = float(np.percentile(occ_mean, 99)) or 1e-6
        panels["occlusion"] = _render_all_slices(occ_mean, "plasma", 0.0, v1, transparent=True)

    # AblationCAM (single-channel spatial map)
    if not _is_sentinel(ablation) and ablation.ndim == 4:
        abl_map = ablation[0]  # (D, H, W)
        v1 = float(np.percentile(abl_map, 99)) or 1e-6
        panels["ablation"] = _render_all_slices(abl_map, "inferno", 0.0, v1, transparent=True)

    # Zones (discrete colormap)
    if not _is_sentinel(zones) and zones.ndim == 3:
        panels["zones"] = _render_all_slices(zones.astype(np.float32), "", 0, 2,
                                              is_zone=True, transparent=True)

    # Stats: pull from in-memory sample_data (refreshed periodically)
    record = _get_case_index().get(model, {}).get(case_id, {})
    stats = {
        "input_ablation":      inp_abl.tolist() if (not _is_sentinel(inp_abl) and inp_abl.shape == (3,)) else None,
        "saliency":            record.get("saliency"),
        "occlusion":           record.get("occlusion"),
        "pz_voxels":           record.get("pz_voxels"),
        "tz_voxels":           record.get("tz_voxels"),
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


@app.route("/api/chart/<model>/<path:relpath>")
def api_chart(model: str, relpath: str):
    """Serve a pre-generated PNG chart."""
    if model not in MODELS:
        abort(404)
    path = METRICS_DIR / model / relpath
    if not path.exists() or path.suffix != ".png":
        abort(404)
    return send_file(path, mimetype="image/png")


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
