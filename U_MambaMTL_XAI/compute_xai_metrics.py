#!/usr/bin/env python3
"""
Compute XAI metrics for nnUNet, U-MambaMTL, and SwinUNETR.

Reads .npz attribution files from results/xai/{model}/fold_{n}/, determines
TP/FN/FP/TN classification and tumor zone (PZ/TZ) per sample, and generates
channel-wise XAI charts + JSON exports.

Output structure:
    results/metrics/{model}/
    ├── sample_data.json
    ├── summary/
    │   ├── confusion_matrix.png
    │   ├── zone_distribution.png
    │   └── overall_channel_activation.png
    └── {saliency,occlusion}/
        └── {tp,fn,both}/
            └── {pz,tz,pz_dominated,combined}/
                ├── pie.png
                ├── detection.png
                └── distribution.png

Usage:
    python compute_xai_metrics.py --model swin_unetr
    python compute_xai_metrics.py --model nnunet
    python compute_xai_metrics.py --model all
    python compute_xai_metrics.py --model all --no-cache
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_processing
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNEL_NAMES = ["T2W", "ADC", "HBV"]
CHANNEL_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

METHODS = ["saliency", "occlusion"]
CLASS_FILTERS = ["tp", "fn", "both"]
# pz         = PZ-primary (pz_only + both_pz)
# tz         = TZ-primary (tz_only + both_tz)
# pz_dominated = both zones present, PZ has more voxels (both_pz)
# combined   = all zones
ZONE_FILTERS = ["pz", "tz", "pz_dominated", "combined"]

PROJECT_ROOT = Path(__file__).parent.resolve()
JSON_DATALIST_DIR = PROJECT_ROOT / "json_datalists" / "picai"
RESULTS_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_XAI_DIRS = {
    "nnunet": Path(
        "/cluster/home/bragehk/master/picai_nnunet_semi_supervised_gc_algorithm/results/xai"
    ),
    "umamba_mtl": PROJECT_ROOT / "results" / "xai" / "umamba_mtl",
    "swin_unetr": PROJECT_ROOT / "results" / "xai" / "swin_unetr",
}

ZONE_LABELS = {
    "pz": "PZ-primary (incl. mixed)",
    "tz": "TZ-primary (incl. mixed)",
    "pz_dominated": "PZ-dominated (both zones)",
    "combined": "All zones",
}
CLASS_LABELS = {
    "tp": "True Positives",
    "fn": "False Negatives",
    "both": "TP + FN",
}


# ---------------------------------------------------------------------------
# Phase 1 — Zone mapping from JSON datalists
# ---------------------------------------------------------------------------

def build_case_zone_mapping() -> dict:
    """
    Load all fold JSON datalists and return a mapping
    case_id → {pca_path, zones_path, pz_voxels, tz_voxels, primary_zone, zone_category}.

    Zone computation uses nibabel (no MONAI required).
    zone_category values: pz_only | tz_only | both_pz | both_tz | unknown
    primary_zone values:  pz | tz | None
    """
    case_info: dict = {}

    for fold_file in sorted(JSON_DATALIST_DIR.glob("fold_[0-9].json")):
        with open(fold_file) as f:
            fold_data = json.load(f)

        for split in ("training", "validation"):
            for entry in fold_data.get(split, []):
                img_path = entry["image"][0] if isinstance(entry["image"], list) else entry["image"]
                case_id = Path(img_path).name.split("_0000")[0]

                if case_id in case_info:
                    continue

                case_info[case_id] = {
                    "pca_path": entry.get("pca", ""),
                    "zones_path": entry.get("zones", ""),
                    "pz_voxels": None,
                    "tz_voxels": None,
                    "primary_zone": None,
                    "zone_category": None,
                }

    print(f"Found {len(case_info)} unique case_ids in JSON datalists.")

    n_loaded = 0
    n_skipped = 0
    for case_id, info in case_info.items():
        pca_path, zones_path = info["pca_path"], info["zones_path"]

        if not pca_path or not zones_path:
            n_skipped += 1
            continue
        if not Path(pca_path).exists() or not Path(zones_path).exists():
            n_skipped += 1
            continue

        try:
            pca_img = nib.load(pca_path)
            pca_arr = (pca_img.get_fdata() > 0.5).astype(np.float32)

            if pca_arr.sum() == 0:
                n_skipped += 1
                continue

            zone_img = nib.load(zones_path)
            if zone_img.shape != pca_img.shape:
                zone_img = nib_processing.resample_from_to(zone_img, pca_img, order=0)

            zone_arr = zone_img.get_fdata()
            pz_v = int((pca_arr * (zone_arr == 1)).sum())
            tz_v = int((pca_arr * (zone_arr == 2)).sum())

            info["pz_voxels"] = pz_v
            info["tz_voxels"] = tz_v

            if pz_v > 0 and tz_v == 0:
                info["zone_category"] = "pz_only"
                info["primary_zone"] = "pz"
            elif tz_v > 0 and pz_v == 0:
                info["zone_category"] = "tz_only"
                info["primary_zone"] = "tz"
            elif pz_v > 0 and tz_v > 0:
                info["zone_category"] = "both_pz" if pz_v >= tz_v else "both_tz"
                info["primary_zone"] = "pz" if pz_v >= tz_v else "tz"
            else:
                info["zone_category"] = "unknown"
                info["primary_zone"] = None

            n_loaded += 1
        except Exception as exc:
            warnings.warn(f"Zone load failed for {case_id}: {exc}")
            n_skipped += 1

    print(f"Zone info loaded: {n_loaded} with cancer, {n_skipped} skipped/no-cancer.")
    return case_info


# ---------------------------------------------------------------------------
# Phase 2 — Per-sample data from NPZ files
# ---------------------------------------------------------------------------

def _is_empty(arr: np.ndarray) -> bool:
    return arr.ndim == 1 and arr.shape[0] == 0


def _channel_stats(arr: np.ndarray) -> dict:
    ch_sum  = arr.sum(axis=(1, 2, 3))
    ch_mean = arr.mean(axis=(1, 2, 3))
    ch_max  = arr.max(axis=(1, 2, 3))
    ch_std  = arr.std(axis=(1, 2, 3))
    total = ch_sum.sum()
    ch_fraction = (ch_sum / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
    return {
        "ch_sum": ch_sum.tolist(),
        "ch_mean": ch_mean.tolist(),
        "ch_max": ch_max.tolist(),
        "ch_std": ch_std.tolist(),
        "ch_fraction": ch_fraction,
        "dominant_ch": int(np.argmax(ch_mean)),
    }


def load_sample_data(model_name: str, case_zone_map: dict, no_cache: bool = False) -> list:
    """Load per-sample records, using sample_data.json cache if available."""
    output_dir = RESULTS_METRICS_DIR / model_name
    cache_path = output_dir / "sample_data.json"

    if not no_cache and cache_path.exists():
        print(f"  Loading cached sample data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    xai_dir = MODEL_XAI_DIRS[model_name]
    if not xai_dir.exists():
        print(f"  WARNING: XAI directory not found: {xai_dir}")
        return []

    npz_files = sorted(xai_dir.glob("fold_*/*.npz"))
    print(f"  Found {len(npz_files)} .npz files in {xai_dir}")

    records = []
    for npz_path in npz_files:
        try:
            npz = np.load(npz_path, allow_pickle=True)

            label      = npz["label"]
            prediction = npz["prediction"]
            case_id    = str(npz["case_id"])
            fold       = int(npz["fold"])

            if _is_empty(label):
                zi = case_zone_map.get(case_id, {})
                has_pca = zi.get("pz_voxels") is not None and (
                    (zi.get("pz_voxels") or 0) + (zi.get("tz_voxels") or 0) > 0
                )
            else:
                has_pca = bool(label.sum() > 1)

            predicted_positive = (not _is_empty(prediction)) and bool(prediction.max() > 0.5)

            if predicted_positive and has_pca:
                classification = "tp"
            elif predicted_positive and not has_pca:
                classification = "fp"
            elif not predicted_positive and not has_pca:
                classification = "tn"
            else:
                classification = "fn"

            zi = case_zone_map.get(case_id, {})
            record = {
                "case_id":        case_id,
                "fold":           fold,
                "classification": classification,
                "has_pca":        has_pca,
                "predicted_positive": predicted_positive,
                "primary_zone":   zi.get("primary_zone"),
                "zone_category":  zi.get("zone_category"),
                "pz_voxels":      zi.get("pz_voxels"),
                "tz_voxels":      zi.get("tz_voxels"),
            }

            for method in METHODS:
                arr = npz.get(method, np.zeros((0,), dtype=np.float32))
                record[method] = _channel_stats(arr) if (not _is_empty(arr) and arr.ndim == 4) else None

            records.append(record)
        except Exception as exc:
            warnings.warn(f"Failed to load {npz_path}: {exc}")

    print(f"  Loaded {len(records)} samples.")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Cached to {cache_path}")
    return records


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_samples(records: list, method: str, class_filter: str, zone_filter: str) -> list:
    """Filter by classification and zone, keeping only records with XAI data for method."""
    if class_filter == "tp":
        filtered = [r for r in records if r["classification"] == "tp"]
    elif class_filter == "fn":
        filtered = [r for r in records if r["classification"] == "fn"]
    else:
        filtered = [r for r in records if r["classification"] in ("tp", "fn")]

    if zone_filter == "pz":
        filtered = [r for r in filtered if r.get("primary_zone") == "pz"]
    elif zone_filter == "tz":
        filtered = [r for r in filtered if r.get("primary_zone") == "tz"]
    elif zone_filter == "pz_dominated":
        filtered = [r for r in filtered if r.get("zone_category") == "both_pz"]

    return [r for r in filtered if r.get(method) is not None]


# ---------------------------------------------------------------------------
# Phase 3 — Chart generation
# ---------------------------------------------------------------------------

def plot_channel_pie(samples: list, method: str, title: str, output_path: Path) -> None:
    """Pie chart of average channel attribution magnitude share (T2W / ADC / HBV)."""
    if not samples:
        return

    all_abs_sums = np.abs(np.array([r[method]["ch_sum"] for r in samples]))
    avg = all_abs_sums.mean(axis=0)
    total = avg.sum()
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        avg / total,
        labels=CHANNEL_NAMES,
        autopct="%1.1f%%",
        colors=CHANNEL_COLORS,
        startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title(f"{title}\nn={len(samples)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_detection_bar(records: list, zone_filter: str, title: str, output_path: Path) -> None:
    """
    Grouped bar chart: TP / FN per zone group, plus a dedicated FP bar.

    FPs have no cancer zone, so they appear as their own group.
    Secondary axis shows Sensitivity per zone and overall Precision (dashed line).
    """
    cancer_records = [r for r in records if r["has_pca"]]
    fp_records     = [r for r in records if r["classification"] == "fp"]

    if not cancer_records and not fp_records:
        return

    total_fp = len(fp_records)
    total_tp = sum(1 for r in cancer_records if r["classification"] == "tp")

    all_zone_groups = {
        "PZ-primary":    [r for r in cancer_records if r.get("primary_zone") == "pz"],
        "PZ-dominated":  [r for r in cancer_records if r.get("zone_category") == "both_pz"],
        "TZ-primary":    [r for r in cancer_records if r.get("primary_zone") == "tz"],
        "All zones":     cancer_records,
    }

    if zone_filter == "pz":
        zone_groups = {"PZ-primary": all_zone_groups["PZ-primary"],
                       "All zones":  all_zone_groups["All zones"]}
    elif zone_filter == "tz":
        zone_groups = {"TZ-primary": all_zone_groups["TZ-primary"],
                       "All zones":  all_zone_groups["All zones"]}
    elif zone_filter == "pz_dominated":
        zone_groups = {"PZ-dominated": all_zone_groups["PZ-dominated"],
                       "All zones":    all_zone_groups["All zones"]}
    else:
        zone_groups = all_zone_groups

    labels      = list(zone_groups.keys()) + ["FP\n(no zone)"]
    tp_counts   = []
    fn_counts   = []
    fp_counts   = []
    sensitivities = []

    for zone_label in list(zone_groups.keys()):
        grp = zone_groups[zone_label]
        tp  = sum(1 for r in grp if r["classification"] == "tp")
        fn  = sum(1 for r in grp if r["classification"] == "fn")
        tp_counts.append(tp)
        fn_counts.append(fn)
        fp_counts.append(0)
        denom = tp + fn
        sensitivities.append(tp / denom if denom > 0 else 0.0)

    # FP group
    tp_counts.append(0)
    fn_counts.append(0)
    fp_counts.append(total_fp)
    sensitivities.append(float("nan"))

    x = np.arange(len(labels))
    w = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - w,  tp_counts, w, label="TP", color="#2171b5")
    ax1.bar(x,      fn_counts, w, label="FN", color="#cb181d")
    ax1.bar(x + w,  fp_counts, w, label="FP", color="#fd8d3c")

    sens_x = [xi for xi, s in zip(x, sensitivities) if not np.isnan(s)]
    sens_y = [s  for s        in sensitivities       if not np.isnan(s)]
    if sens_x:
        ax2.plot(sens_x, sens_y, "k^--", markersize=8, label="Sensitivity")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    if not np.isnan(precision):
        ax2.axhline(precision, color="purple", linestyle=":", linewidth=1.5,
                    label=f"Precision={precision:.2f}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Case count")
    ax2.set_ylabel("Sensitivity / Precision")
    ax2.set_ylim(0, 1.15)
    ax1.set_title(f"{title}\ncancer={len(cancer_records)}  FP={total_fp}", fontsize=11)

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distribution(samples: list, method: str, title: str, output_path: Path) -> None:
    """Box plot of per-channel mean attribution across the filtered sample subset."""
    if not samples:
        return

    ch_means = np.array([r[method]["ch_mean"] for r in samples])

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(
        [ch_means[:, i] for i in range(3)],
        tick_labels=CHANNEL_NAMES,
        patch_artist=True,
    )
    for patch, color in zip(bp["boxes"], CHANNEL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Mean Attribution")
    ax.set_title(f"{title}\nn={len(samples)}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 4 — Model-level summary charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix_chart(records: list, output_dir: Path) -> None:
    counts = {c: sum(1 for r in records if r["classification"] == c)
              for c in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    if tp + fp + fn + tn == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["TP", "FP", "FN", "TN"], [tp, fp, fn, tn],
                  color=["#2171b5", "#cb181d", "#fd8d3c", "#74c476"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Count")

    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1          = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else float("nan")

    ax.set_title(
        f"Model Performance\n"
        f"Precision={precision:.3f}  Sensitivity={sensitivity:.3f}  "
        f"Specificity={specificity:.3f}  F1={f1:.3f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.3f} Sensitivity={sensitivity:.3f} "
          f"Specificity={specificity:.3f} F1={f1:.3f}")


def plot_zone_distribution(records: list, output_dir: Path) -> None:
    cancer = [r for r in records if r["has_pca"]]
    if not cancer:
        return

    zone_groups = {
        "PZ only":      [r for r in cancer if r.get("zone_category") == "pz_only"],
        "PZ-dominated": [r for r in cancer if r.get("zone_category") == "both_pz"],
        "TZ-dominated": [r for r in cancer if r.get("zone_category") == "both_tz"],
        "TZ only":      [r for r in cancer if r.get("zone_category") == "tz_only"],
        "Unknown":      [r for r in cancer if r.get("primary_zone") is None],
    }

    labels, totals, tps = [], [], []
    for lbl, grp in zone_groups.items():
        if not grp:
            continue
        labels.append(lbl)
        totals.append(len(grp))
        tps.append(sum(1 for r in grp if r["classification"] == "tp"))

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, totals, w, label="Total",        color="#6baed6")
    ax.bar(x + w/2, tps,    w, label="Detected (TP)", color="#2171b5")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Cases")
    ax.set_title(f"Tumor Zone Distribution & Detection (n={len(cancer)} cancer cases)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "zone_distribution.png", dpi=150)
    plt.close(fig)


def plot_overall_channel_activation(records: list, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, METHODS):
        valid = [r for r in records if r.get(method) is not None]
        if not valid:
            ax.set_title(f"{method.capitalize()} — no data")
            continue
        all_means = np.array([r[method]["ch_mean"] for r in valid])
        avg, std  = all_means.mean(axis=0), all_means.std(axis=0)
        bars = ax.bar(CHANNEL_NAMES, avg, yerr=std, capsize=5, color=CHANNEL_COLORS)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_ylabel("Mean Attribution")
        ax.set_title(f"{method.capitalize()} — Avg Activation (n={len(valid)})")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_channel_activation.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main per-model pipeline
# ---------------------------------------------------------------------------

def process_model(model_name: str, case_zone_map: dict, no_cache: bool = False) -> None:
    print(f"\n{'=' * 60}")
    print(f"Processing model: {model_name}")
    print(f"{'=' * 60}")

    model_dir = RESULTS_METRICS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    records = load_sample_data(model_name, case_zone_map, no_cache=no_cache)
    if not records:
        print(f"  No data found for {model_name}, skipping.")
        return

    # --- Summary charts ---
    summary_dir = model_dir / "summary"
    summary_dir.mkdir(exist_ok=True)
    print("\n  Generating summary charts…")
    plot_confusion_matrix_chart(records, summary_dir)
    plot_zone_distribution(records, summary_dir)
    plot_overall_channel_activation(records, summary_dir)

    # --- Per-combination charts ---
    n_charts = 0
    for method in METHODS:
        for class_filter in CLASS_FILTERS:
            for zone_filter in ZONE_FILTERS:
                subset = filter_samples(records, method, class_filter, zone_filter)
                if not subset:
                    continue

                # Nested output directory: {model}/{method}/{class}/{zone}/
                out_dir = model_dir / method / class_filter / zone_filter
                out_dir.mkdir(parents=True, exist_ok=True)

                title_base = (
                    f"{model_name.upper()} | {method.capitalize()} | "
                    f"{CLASS_LABELS[class_filter]} | {ZONE_LABELS[zone_filter]}"
                )

                plot_channel_pie(
                    subset, method,
                    f"{title_base}\nChannel Attribution Share",
                    out_dir / "pie.png",
                )
                plot_detection_bar(
                    records, zone_filter,
                    f"{title_base}\nDetection Rate by Zone",
                    out_dir / "detection.png",
                )
                plot_distribution(
                    subset, method,
                    f"{title_base}\nChannel Attribution Distribution",
                    out_dir / "distribution.png",
                )
                n_charts += 3

    print(f"\n  Generated {n_charts} combination charts.")
    print(f"  All outputs saved to: {model_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute XAI metrics for nnUNet, U-MambaMTL, and SwinUNETR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_XAI_DIRS.keys()) + ["all"],
        help="Model to process (or 'all' for all three).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cached sample_data.json and recompute from NPZ files.",
    )
    args = parser.parse_args()

    models_to_run = list(MODEL_XAI_DIRS.keys()) if args.model == "all" else [args.model]

    print("Building case_id → zone mapping from JSON datalists…")
    case_zone_map = build_case_zone_mapping()

    for model_name in models_to_run:
        process_model(model_name, case_zone_map, no_cache=args.no_cache)

    print("\nAll done.")


if __name__ == "__main__":
    main()
