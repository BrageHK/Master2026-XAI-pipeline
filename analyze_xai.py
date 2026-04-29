#!/usr/bin/env python3
"""
Analyze XAI results from progress.json files saved during generate_xai_data.py runs.

Reads per-case progress.json files across all folds for one or more models and produces:
  - results/analysis/{model}/summary.json     — aggregated TP/FP/TN/FN + metric scores
  - results/analysis/{model}/confusion_matrix.png
  - results/analysis/{model}/zone_distribution.png
  - results/analysis/{model}/channel_activation/{method}/{filter}/pie.png

Channel activation pie filters: overall, tp, fp, pz, tz

Usage:
  python analyze_xai.py --model umamba_mtl
  python analyze_xai.py --model all
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path(__file__).parent.resolve()
XAI_DIR         = PROJECT_ROOT / "results" / "xai"
ANALYSIS_DIR    = PROJECT_ROOT / "results" / "analysis"

ALL_MODELS    = ["umamba_mtl", "swin_unetr", "nnunet"]
METHODS       = ["saliency", "occlusion_zm", "occlusion_zero", "ig", "gs"]
CHANNEL_NAMES = ["T2W", "ADC", "HBV"]
CHANNEL_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

CLS_FILTERS = [
    ("tp_fp", "FP + TP", lambda r: r.get("classification") in ("tp", "fp")),
    ("tp",    "TP",      lambda r: r.get("classification") == "tp"),
    ("fp",    "FP",      lambda r: r.get("classification") == "fp"),
]

ZONE_FILTERS = [
    ("all", "All zones", lambda r: True),
    ("pz",  "PZ",        lambda r: r.get("primary_zone") == "pz"),
    ("tz",  "TZ",        lambda r: r.get("primary_zone") == "tz"),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(model_name: str) -> List[dict]:
    """Load all per-case records from progress.json files across folds."""
    records = []
    model_dir = XAI_DIR / model_name
    if not model_dir.exists():
        print(f"  No XAI output found for {model_name} at {model_dir}")
        return records

    for fold_dir in sorted(model_dir.glob("fold_*")):
        progress_file = fold_dir / "progress.json"
        if not progress_file.exists():
            continue
        with open(progress_file) as f:
            progress = json.load(f)

        fold_num = int(fold_dir.name.split("_")[1])
        for case_id, rec in progress.items():
            if not rec.get("done") or rec.get("error"):
                continue
            records.append({**rec, "case_id": case_id, "fold": fold_num, "model": model_name})

    print(f"  Loaded {len(records)} completed cases for {model_name}")
    return records


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_classification_counts(records: List[dict]) -> Dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for r in records:
        c = r.get("classification")
        if c in counts:
            counts[c] += 1
    return counts


def safe_div(a: float, b: float) -> Optional[float]:
    return round(a / b, 4) if b > 0 else None


def compute_metrics(counts: Dict[str, int]) -> Dict[str, Optional[float]]:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    return {
        "precision":    safe_div(tp, tp + fp),
        "sensitivity":  safe_div(tp, tp + fn),
        "specificity":  safe_div(tn, tn + fp),
        "f1":           safe_div(2 * tp, 2 * tp + fp + fn),
    }


def mean_ch_fraction(records: List[dict], method_key: str) -> Optional[List[float]]:
    """Mean channel fraction across records that have the given method's attribution."""
    fracs = [r[method_key] for r in records if r.get(method_key) is not None]
    if not fracs:
        return None
    arr = np.array(fracs)  # (N, 3)
    return arr.mean(axis=0).tolist()


def zone_distribution(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Count total and TP per zone_category."""
    cats = ["pz_only", "tz_only", "both_pz", "both_tz", "unknown"]
    dist: Dict[str, Dict[str, int]] = {c: {"total": 0, "tp": 0} for c in cats}
    for r in records:
        cat = r.get("zone_category") or "unknown"
        if cat not in dist:
            cat = "unknown"
        dist[cat]["total"] += 1
        if r.get("classification") == "tp":
            dist[cat]["tp"] += 1
    return dist


# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------

def plot_confusion_matrix(counts: Dict[str, int], metrics: dict, out_path: Path) -> None:
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]

    # 2x2 matrix: rows = Actual, cols = Predicted
    # [TP, FN]
    # [FP, TN]
    matrix = np.array([[tp, fn], [fp, tn]])
    cell_colors = np.array([["#2ecc71", "#e67e22"], ["#e74c3c", "#3498db"]])
    row_labels = ["Actual Positive", "Actual Negative"]
    col_labels = ["Predicted Positive", "Predicted Negative"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Confusion Matrix (all folds)", fontsize=14, fontweight="bold", pad=16)

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle([j, 1 - i], 1, 1, color=cell_colors[i, j], alpha=0.75,
                                  transform=ax.transData)
            ax.add_patch(rect)
            label = ["TP", "FN", "FP", "TN"][i * 2 + j]
            ax.text(j + 0.5, 1 - i + 0.6, label,
                    ha="center", va="center", fontsize=11, color="white", fontweight="bold")
            ax.text(j + 0.5, 1 - i + 0.25, str(matrix[i, j]),
                    ha="center", va="center", fontsize=22, fontweight="bold", color="white")

    for j, lbl in enumerate(col_labels):
        ax.text(j + 0.5, 2.12, lbl, ha="center", va="bottom", fontsize=10, fontweight="bold")
    for i, lbl in enumerate(row_labels):
        ax.text(-0.08, 1 - i + 0.5, lbl, ha="right", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.1, 2.3)

    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("Metrics", fontsize=14, fontweight="bold", pad=16)
    metric_labels = ["Precision", "Sensitivity", "Specificity", "F1"]
    metric_keys   = ["precision", "sensitivity", "specificity", "f1"]
    rows = [[lbl, f"{metrics[k]:.3f}" if metrics[k] is not None else "N/A"]
            for lbl, k in zip(metric_labels, metric_keys)]
    table = ax2.table(cellText=rows, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_zone_distribution_chart(dist: Dict[str, Dict[str, int]], out_path: Path) -> None:
    cat_labels = {
        "pz_only": "PZ only",
        "tz_only": "TZ only",
        "both_pz": "Mixed (PZ-dom.)",
        "both_tz": "Mixed (TZ-dom.)",
        "unknown": "Unknown",
    }
    cats   = list(cat_labels.keys())
    totals = [dist[c]["total"] for c in cats]
    tps    = [dist[c]["tp"]    for c in cats]

    x = np.arange(len(cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, totals, width, label="Total", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, tps,    width, label="TP",    color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([cat_labels[c] for c in cats], fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Zone Distribution (all folds)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_channel_distribution(subset: List[dict], method: str, title: str, out_path: Path) -> None:
    """Box plots of channel attribution fractions (T2W / ADC / HBV) for a given subset."""
    frac_key = f"{method}_ch_fraction"
    data = [r for r in subset if r.get(frac_key)]
    n = len(data)

    fig, ax = plt.subplots(figsize=(5, 5))
    for ch_idx, (ch_name, color) in enumerate(zip(CHANNEL_NAMES, CHANNEL_COLORS)):
        values = [r[frac_key][ch_idx] for r in data]
        ax.boxplot(
            values,
            positions=[ch_idx],
            widths=0.5,
            patch_artist=True,
            manage_ticks=False,
            boxprops=dict(facecolor=color, alpha=0.7),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="#555"),
            capprops=dict(color="#555"),
            flierprops=dict(marker=".", color=color, alpha=0.4, markersize=4),
        )

    ax.set_xticks(range(len(CHANNEL_NAMES)))
    ax.set_xticklabels(CHANNEL_NAMES, fontsize=11)
    ax.set_ylabel("Attribution fraction")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\n(n={n})", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_channel_pie(fracs: List[float], title: str, n: int, out_path: Path) -> None:
    fracs_arr = np.array(fracs)
    fracs_arr = np.clip(fracs_arr, 0, None)
    total = fracs_arr.sum()
    if total <= 0:
        return
    fracs_arr = fracs_arr / total

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        fracs_arr,
        labels=CHANNEL_NAMES,
        colors=CHANNEL_COLORS,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title(f"{title}\n(n={n})", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model_name: str, output_dir: Path) -> None:
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {model_name}")

    records = load_records(model_name)
    if not records:
        print("  No records found. Skipping.")
        return

    out_dir = output_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = sorted({r["fold"] for r in records})

    # ---- Classification counts + metrics ----------------------------------
    counts  = compute_classification_counts(records)
    metrics = compute_metrics(counts)
    print(f"  Counts: {counts}")
    print(f"  Metrics: {metrics}")

    # ---- Zone distribution ------------------------------------------------
    dist = zone_distribution(records)

    # ---- summary.json -----------------------------------------------------
    summary = {
        "model":        model_name,
        "folds":        folds,
        "total_cases":  len(records),
        **counts,
        **metrics,
        "zone_distribution": dist,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out_dir / 'summary.json'}")

    # ---- Charts -----------------------------------------------------------
    plot_confusion_matrix(counts, metrics, out_dir / "confusion_matrix.png")
    plot_zone_distribution_chart(dist, out_dir / "zone_distribution.png")

    for method in METHODS:
        frac_key = f"{method}_ch_fraction"
        for cls_key, cls_label, cls_fn in CLS_FILTERS:
            for zone_key, zone_label, zone_fn in ZONE_FILTERS:
                subset = [r for r in records if cls_fn(r) and zone_fn(r)]
                title = f"{cls_label} / {zone_label}"
                out_base = out_dir / "channel_activation" / method / cls_key / zone_key

                plot_channel_distribution(subset, method, title, out_base / "distribution.png")

                fracs = mean_ch_fraction(subset, frac_key)
                if fracs is not None:
                    plot_channel_pie(fracs, title=title, n=len(subset),
                                     out_path=out_base / "pie.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze XAI results from progress.json files.")
    parser.add_argument(
        "--model", nargs="+", default=["all"],
        help="Model(s) to analyze: umamba_mtl, swin_unetr, nnunet, or all",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ANALYSIS_DIR,
        help="Output directory for analysis results (default: results/analysis/)",
    )
    args = parser.parse_args()

    models = ALL_MODELS if "all" in args.model else args.model
    for model_name in models:
        analyze_model(model_name, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
