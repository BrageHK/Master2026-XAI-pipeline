import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List

from src.metrics.compute import METHODS

CLASS_FILTERS = ["tp", "fp", "both"]
ZONE_FILTERS  = ["pz", "tz", "pz_dominated", "combined"]

CHANNEL_NAMES  = ["t2w", "adc", "hbv"]
CHANNEL_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]

ZONE_LABELS = {
    "pz":           "PZ-primary (incl. mixed)",
    "tz":           "TZ-primary (incl. mixed)",
    "pz_dominated": "PZ-dominated (both zones)",
    "combined":     "All zones",
}
CLASS_LABELS = {
    "tp":   "True Positives",
    "fp":   "False Positives",
    "both": "TP + FP",
}


def filter_samples(records: list, method: str, class_filter: str, zone_filter: str) -> list:
    if class_filter == "tp":
        filtered = [r for r in records if r["classification"] == "tp"]
    elif class_filter == "fp":
        filtered = [r for r in records if r["classification"] == "fp"]
    else:
        filtered = [r for r in records if r["classification"] in ("tp", "fp")]

    if zone_filter == "pz":
        filtered = [r for r in filtered if r.get("primary_zone") == "pz"]
    elif zone_filter == "tz":
        filtered = [r for r in filtered if r.get("primary_zone") == "tz"]
    elif zone_filter == "pz_dominated":
        filtered = [r for r in filtered if r.get("zone_category") == "both_pz"]

    return [r for r in filtered if r.get(method) is not None]


def plot_confusion_matrix_chart(records: list, output_dir: Path) -> None:
    counts = {c: sum(1 for r in records if r["classification"] == c)
              for c in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    if tp + fp + fn + tn == 0:
        return

    precision   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1          = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["TP", "FP", "FN", "TN"], [tp, fp, fn, tn],
                  color=["#2171b5", "#cb181d", "#fd8d3c", "#74c476"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Count")
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
    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, totals, w, label="Total",         color="#6baed6")
    ax.bar(x + w/2, tps,    w, label="Detected (TP)", color="#2171b5")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
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


def plot_channel_pie(samples: list, method: str, title: str, output_path: Path) -> None:
    if not samples:
        return
    all_abs = np.abs(np.array([r[method]["ch_sum"] for r in samples]))
    avg     = all_abs.mean(axis=0)
    total   = avg.sum()
    if total == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, autotexts = ax.pie(
        avg / total, labels=CHANNEL_NAMES, autopct="%1.1f%%",
        colors=CHANNEL_COLORS, startangle=90,
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title(f"{title}\nn={len(samples)}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_detection_bar(records: list, zone_filter: str, title: str, output_path: Path) -> None:
    cancer_records = [r for r in records if r["has_pca"]]
    fp_records     = [r for r in records if r["classification"] == "fp"]
    if not cancer_records and not fp_records:
        return

    all_groups = {
        "PZ-primary":   [r for r in cancer_records if r.get("primary_zone") == "pz"],
        "PZ-dominated": [r for r in cancer_records if r.get("zone_category") == "both_pz"],
        "TZ-primary":   [r for r in cancer_records if r.get("primary_zone") == "tz"],
        "All zones":    cancer_records,
    }
    if zone_filter == "pz":
        groups = {"PZ-primary": all_groups["PZ-primary"], "All zones": all_groups["All zones"]}
    elif zone_filter == "tz":
        groups = {"TZ-primary": all_groups["TZ-primary"], "All zones": all_groups["All zones"]}
    elif zone_filter == "pz_dominated":
        groups = {"PZ-dominated": all_groups["PZ-dominated"], "All zones": all_groups["All zones"]}
    else:
        groups = all_groups

    labels = list(groups.keys()) + ["FP\n(no zone)"]
    tp_c, fn_c, fp_c, sens = [], [], [], []
    for grp in groups.values():
        tp = sum(1 for r in grp if r["classification"] == "tp")
        fn = sum(1 for r in grp if r["classification"] == "fn")
        tp_c.append(tp); fn_c.append(fn); fp_c.append(0)
        denom = tp + fn
        sens.append(tp / denom if denom > 0 else 0.0)
    tp_c.append(0); fn_c.append(0); fp_c.append(len(fp_records)); sens.append(float("nan"))

    x, w = np.arange(len(labels)), 0.25
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - w, tp_c, w, label="TP", color="#2171b5")
    ax1.bar(x,     fn_c, w, label="FN", color="#cb181d")
    ax1.bar(x + w, fp_c, w, label="FP", color="#fd8d3c")
    sx = [xi for xi, s in zip(x, sens) if not np.isnan(s)]
    sy = [s  for s       in sens       if not np.isnan(s)]
    if sx:
        ax2.plot(sx, sy, "k^--", markersize=8, label="Sensitivity")
    total_tp = sum(1 for r in cancer_records if r["classification"] == "tp")
    total_fp = len(fp_records)
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float("nan")
    if not np.isnan(prec):
        ax2.axhline(prec, color="purple", linestyle=":", linewidth=1.5,
                    label=f"Precision={prec:.2f}")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Case count"); ax2.set_ylabel("Sensitivity / Precision")
    ax2.set_ylim(0, 1.15)
    ax1.set_title(f"{title}\ncancer={len(cancer_records)}  FP={total_fp}", fontsize=11)
    l1, ll1 = ax1.get_legend_handles_labels()
    l2, ll2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, ll1 + ll2, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_distribution(samples: list, method: str, title: str, output_path: Path) -> None:
    if not samples:
        return
    ch_means = np.array([r[method]["ch_mean"] for r in samples])
    fig, ax  = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([ch_means[:, i] for i in range(3)],
                    tick_labels=CHANNEL_NAMES, patch_artist=True)
    for patch, color in zip(bp["boxes"], CHANNEL_COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Mean Attribution")
    ax.set_title(f"{title}\nn={len(samples)}", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_charts(records: list, model_name: str, metrics_dir: Path) -> None:
    """Generate all metric charts for *model_name* from *records*."""
    model_dir = metrics_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_dir = model_dir / "summary"
    summary_dir.mkdir(exist_ok=True)

    print(f"\n  Generating summary charts for {model_name}…")
    plot_confusion_matrix_chart(records, summary_dir)
    plot_zone_distribution(records, summary_dir)
    plot_overall_channel_activation(records, summary_dir)

    n_charts = 0
    for method in METHODS:
        for class_filter in CLASS_FILTERS:
            for zone_filter in ZONE_FILTERS:
                subset = filter_samples(records, method, class_filter, zone_filter)
                if not subset:
                    continue
                out_dir = model_dir / method / class_filter / zone_filter
                out_dir.mkdir(parents=True, exist_ok=True)
                title_base = (
                    f"{model_name.upper()} | {method.capitalize()} | "
                    f"{CLASS_LABELS[class_filter]} | {ZONE_LABELS[zone_filter]}"
                )
                plot_channel_pie(subset, method,
                                 f"{title_base}\nChannel Attribution Share",
                                 out_dir / "pie.png")
                plot_distribution(subset, method,
                                  f"{title_base}\nChannel Attribution Distribution",
                                  out_dir / "distribution.png")
                n_charts += 2

    print(f"  Generated {n_charts} combination charts → {model_dir}")
