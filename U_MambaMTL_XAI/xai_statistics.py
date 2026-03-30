import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from shared_modules.xai import create_confusion_matrix


parser = argparse.ArgumentParser(description="XAI statistics across all folds for a given model")
parser.add_argument("--model", type=str, default="umamba_mtl", help="Model name (e.g. umamba_mtl, swin_unetr)")
args = parser.parse_args()

model = args.model
print(f"Creating stats for: {model}\n")
MODEL_DIR = Path(f"xai_outputs/{model}")
CHANNEL_NAMES = ["T2W", "ADC", "HBV"]

# Discover all fold directories
fold_dirs = sorted(MODEL_DIR.glob("f[0-9]*"))
if not fold_dirs:
    raise FileNotFoundError(f"No fold directories found in {MODEL_DIR}")

print(f"Found {len(fold_dirs)} fold(s): {[d.name for d in fold_dirs]}")

# Concatenate results from all folds
results = []
for fold_dir in fold_dirs:
    results_path = fold_dir / "results.json"
    if not results_path.exists():
        print(f"  WARNING: {results_path} not found, skipping")
        continue
    with open(results_path) as f:
        fold_results = json.load(f)
    #print(f"  {fold_dir.name}: {len(fold_results)} samples")
    results.extend(fold_results)

print(f"Total samples across all folds: {len(results)}\n")

OUTPUT_DIR = MODEL_DIR
tn, fp, fn, tp = create_confusion_matrix(OUTPUT_DIR, results, model)

print(f"TP: {tp}  FP: {fp}")
print(f"FN: {fn}  TN: {tn}")
print(f"Accuracy:    {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"Sensitivity: {tp / (tp + fn):.4f}")
print(f"Specificity: {tn / (tn + fp):.4f}")
print(f"Precision:   {tp / (tp + fp):.4f}")
print(f"F1 Score:    {2 * tp / (2 * tp + fp + fn):.4f}")

# ---- Tumor zone distribution (PZ vs TZ) ----
pca_samples = [r for r in results if r["has_pca"]]
#for sample in pca_samples:
    #print(sample["pca_voxels_in_pz"], sample["pca_voxels_in_tz"])
pz_only = [r for r in pca_samples if r["pca_voxels_in_pz"] > 0 and r["pca_voxels_in_tz"] == 0]
tz_only = [r for r in pca_samples if r["pca_voxels_in_tz"] > 0 and r["pca_voxels_in_pz"] == 0]
both = [r for r in pca_samples if r["pca_voxels_in_pz"] > 0 and r["pca_voxels_in_tz"] > 0]
majority_pz = [r for r in both if r["pca_voxels_in_pz"] > r["pca_voxels_in_tz"]]
majority_tz = [r for r in both if r["pca_voxels_in_tz"] > r["pca_voxels_in_pz"]]

print(f"\n--- Tumor Zone Distribution (n={len(pca_samples)} PCa cases) ---")
print(f"PZ only:  {len(pz_only):3d} ({len(pz_only)/len(pca_samples)*100:.1f}%)")
print(f"TZ only:  {len(tz_only):3d} ({len(tz_only)/len(pca_samples)*100:.1f}%)")
print(f"Both:     {len(both):3d} ({len(both)/len(pca_samples)*100:.1f}%)")

# Detection rate per zone
tp_pz = [r for r in pz_only if r["classification"] == "tp"]
tp_tz = [r for r in tz_only if r["classification"] == "tp"]
tp_majority_pz = [r for r in majority_pz if r["classification"] == "tp"]
tp_majority_tz = [r for r in majority_tz if r["classification"] == "tp"]
tp_both = [r for r in both if r["classification"] == "tp"]

print(f"\n--- Detection Rate by Zone ---")
if pz_only:
    print(f"PZ only:  {len(tp_pz)}/{len(pz_only)} detected ({len(tp_pz)/len(pz_only)*100:.1f}%)")
if tz_only:
    print(f"TZ only:  {len(tp_tz)}/{len(tz_only)} detected ({len(tp_tz)/len(tz_only)*100:.1f}%)")
if both:
    print(f"Both:     {len(tp_both)}/{len(both)} detected ({len(tp_both)/len(both)*100:.1f}%)")
if majority_pz:
    print(f"Majority_pz:     {len(tp_majority_pz)}/{len(majority_pz)} detected ({len(tp_majority_pz)/len(majority_pz)*100:.1f}%)")
if majority_tz:
    print(f"Majority_tz:     {len(tp_majority_tz)}/{len(majority_tz)} detected ({len(tp_majority_tz)/len(majority_tz)*100:.1f}%)")

# Bar chart
zone_labels = ["PZ only", "TZ only", "Both zones"]
zone_counts = [len(pz_only), len(tz_only), len(both)]
zone_detected = [len(tp_pz), len(tp_tz), len(tp_both)]

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(zone_labels))
w = 0.35
ax.bar(x - w/2, zone_counts, w, label="Total", color="#6baed6")
ax.bar(x + w/2, zone_detected, w, label="Detected (TP)", color="#2171b5")
ax.set_xticks(x)
ax.set_xticklabels(zone_labels)
ax.set_ylabel("Number of cases")
ax.set_title("Tumor Zone Distribution & Detection Rate")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "zone_distribution.png", dpi=150)

# ============================================================
# Channel-wise XAI Statistics (Saliency & Occlusion maps)
# ============================================================
samples_with_maps = [r for r in results if r["maps_dir"] is not None]
print(f"\n{'='*60}")
print(f"Channel-wise XAI Statistics ({len(samples_with_maps)} samples with maps)")
print(f"{'='*60}")

# Accumulators per map type
per_sample_records = []

for r in samples_with_maps:
    sample_dir = Path(r["maps_dir"])
    saliency_path = sample_dir / "saliency_map.pt"
    occlusion_path = sample_dir / "occlusion_map.pt"

    if not saliency_path.exists() or not occlusion_path.exists():
        print(saliency_path)
        print(occlusion_path)
        print("Whoops skipping")
        continue

    sal = torch.load(saliency_path, weights_only=False)[0]  # [3, H, W, D]
    occ = torch.load(occlusion_path, weights_only=False)[0]  # [3, H, W, D]

    record = {
        "case_id": r["case_id"],
        "classification": r["classification"],
    }

    for map_name, m in [("saliency", sal), ("occlusion", occ)]:
        ch_mean = m.mean(dim=(1, 2, 3)).numpy()  # [3]
        ch_max = m.amax(dim=(1, 2, 3)).numpy()    # [3]
        ch_sum = m.sum(dim=(1, 2, 3)).numpy()     # [3]
        ch_std = m.std(dim=(1, 2, 3)).numpy()     # [3]

        total_activation = ch_sum.sum()
        ch_fraction = ch_sum / total_activation if total_activation > 0 else ch_sum * 0

        dominant_ch = int(np.argmax(ch_mean))

        record[f"{map_name}_ch_mean"] = ch_mean
        record[f"{map_name}_ch_max"] = ch_max
        record[f"{map_name}_ch_sum"] = ch_sum
        record[f"{map_name}_ch_std"] = ch_std
        record[f"{map_name}_ch_fraction"] = ch_fraction
        record[f"{map_name}_dominant_ch"] = dominant_ch

    per_sample_records.append(record)

print(f"Loaded maps for {len(per_sample_records)} samples.\n")

# --- Aggregate statistics ---
for map_name in ["saliency", "occlusion"]:
    all_means = np.stack([r[f"{map_name}_ch_mean"] for r in per_sample_records])
    all_maxs = np.stack([r[f"{map_name}_ch_max"] for r in per_sample_records])
    all_stds = np.stack([r[f"{map_name}_ch_std"] for r in per_sample_records])
    all_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in per_sample_records])
    dominant_channels = [r[f"{map_name}_dominant_ch"] for r in per_sample_records]

    print(f"--- {map_name.upper()} Map Channel Statistics ---")
    print(f"{'Channel':<8} {'Avg Activation':>16} {'Avg Max':>12} {'Avg Std':>12} {'Avg Fraction':>14}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"{name:<8} {all_means[:, i].mean():>16.6f} {all_maxs[:, i].mean():>12.4f} "
              f"{all_stds[:, i].mean():>12.6f} {all_fracs[:, i].mean():>13.1%}")

    # Most important channel (by mean activation) across all samples
    dominant_counts = [dominant_channels.count(i) for i in range(3)]
    print(f"\nDominant channel (highest mean activation per sample):")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  {name}: {dominant_counts[i]:3d} / {len(per_sample_records)} "
              f"({dominant_counts[i]/len(per_sample_records)*100:.1f}%)")
    print()

    # Breakdown by classification (TP vs FP)
    for cls in ["tp", "fp", "tn", "fn"]:
        cls_records = [r for r in per_sample_records if r["classification"] == cls]
        if not cls_records:
            continue
        cls_means = np.stack([r[f"{map_name}_ch_mean"] for r in cls_records])
        cls_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in cls_records])
        cls_dominant = [r[f"{map_name}_dominant_ch"] for r in cls_records]

        print(f"  {cls.upper()} samples (n={len(cls_records)}):")
        print(f"  {'Channel':<8} {'Avg Activation':>16} {'Avg Fraction':>14}")
        for i, name in enumerate(CHANNEL_NAMES):
            print(f"  {name:<8} {cls_means[:, i].mean():>16.6f} {cls_fracs[:, i].mean():>13.1%}")
        dom_counts = [cls_dominant.count(i) for i in range(3)]
        print(f"  Dominant: {', '.join(f'{CHANNEL_NAMES[i]}={dom_counts[i]}' for i in range(3))}")
        print()

# ============================================================
# Plots
# ============================================================

# 1) Average activation per channel (saliency vs occlusion side by side)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    all_means = np.stack([r[f"{map_name}_ch_mean"] for r in per_sample_records])
    avg = all_means.mean(axis=0)
    std = all_means.std(axis=0)

    bars = ax.bar(CHANNEL_NAMES, avg, yerr=std, capsize=5, color=["#e74c3c", "#2ecc71", "#3498db"])
    ax.set_ylabel("Mean Activation")
    ax.set_title(f"{map_name.capitalize()} — Avg Activation per Channel")
    ax.bar_label(bars, fmt="%.4f", padding=3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_avg_activation.png", dpi=150)
print("Saved channel_avg_activation.png")

# 2) Dominant channel pie chart
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    dominant = [r[f"{map_name}_dominant_ch"] for r in per_sample_records]
    counts = [dominant.count(i) for i in range(3)]
    ax.pie(counts, labels=CHANNEL_NAMES, autopct="%1.1f%%",
           colors=["#e74c3c", "#2ecc71", "#3498db"], startangle=90)
    ax.set_title(f"{map_name.capitalize()} — Dominant Channel")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_dominance.png", dpi=150)
print("Saved channel_dominance.png")

# 3) Channel fraction distribution (box plot)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, map_name in zip(axes, ["saliency", "occlusion"]):
    all_fracs = np.stack([r[f"{map_name}_ch_fraction"] for r in per_sample_records])
    bp = ax.boxplot([all_fracs[:, i] for i in range(3)], tick_labels=CHANNEL_NAMES, patch_artist=True)
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Fraction of Total Activation")
    ax.set_title(f"{map_name.capitalize()} — Channel Fraction Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "channel_fraction_boxplot.png", dpi=150)
print("Saved channel_fraction_boxplot.png")

# 4) TP vs FP channel comparison
for map_name in ["saliency", "occlusion"]:
    tp_records = [r for r in per_sample_records if r["classification"] == "tp"]
    fp_records = [r for r in per_sample_records if r["classification"] == "fp"]
    if not tp_records or not fp_records:
        continue

    tp_means = np.stack([r[f"{map_name}_ch_mean"] for r in tp_records]).mean(axis=0)
    fp_means = np.stack([r[f"{map_name}_ch_mean"] for r in fp_records]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, tp_means, w, label="TP", color="#2171b5")
    ax.bar(x + w/2, fp_means, w, label="FP", color="#cb181d")
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNEL_NAMES)
    ax.set_ylabel("Mean Activation")
    ax.set_title(f"{map_name.capitalize()} — TP vs FP Channel Activation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"channel_tp_vs_fp_{map_name}.png", dpi=150)
    print(f"Saved channel_tp_vs_fp_{map_name}.png")

# ============================================================
# Zone-based Channel Dominance Analysis
# ============================================================
print(f"\n{'='*60}")
print(f"Zone-based Channel Dominance Analysis")
print(f"{'='*60}")

# Create a mapping from case_id to zone information
case_to_zone = {}
for r in results:
    if r["has_pca"]:
        case_id = r["case_id"]
        pz_voxels = r["pca_voxels_in_pz"]
        tz_voxels = r["pca_voxels_in_tz"]
        
        # Determine zone category
        if pz_voxels > 0 and tz_voxels == 0:
            zone_category = "PZ only"
        elif tz_voxels > 0 and pz_voxels == 0:
            zone_category = "TZ only"
        elif pz_voxels > 0 and tz_voxels > 0:
            if pz_voxels > tz_voxels:
                zone_category = "Both (PZ majority)"
            elif tz_voxels > pz_voxels:
                zone_category = "Both (TZ majority)"
            else:
                zone_category = "Both (equal)"
        else:
            zone_category = "None"
        
        case_to_zone[case_id] = {
            "category": zone_category,
            "pz_voxels": pz_voxels,
            "tz_voxels": tz_voxels,
            "dominant_zone": r["pca_dominant_zone"]
        }

# Add zone information to per_sample_records
zone_categories = ["PZ only", "TZ only", "Both (PZ majority)", "Both (TZ majority)"]
for r in per_sample_records:
    case_id = r["case_id"]
    if case_id in case_to_zone:
        r["zone_category"] = case_to_zone[case_id]["category"]
        r["zone_dominant"] = case_to_zone[case_id]["dominant_zone"]
    else:
        r["zone_category"] = "None"
        r["zone_dominant"] = None

# Analyze channel dominance by zone category
for map_name in ["saliency", "occlusion"]:
    print(f"\n--- {map_name.upper()} Map - Channel Dominance by Tumor Zone ---")
    print(f"{'Zone Category':<20} {'T2W':>12} {'ADC':>12} {'HBV':>12} {'Total':>8}")
    print("-" * 72)
    
    total_by_zone = {}
    zone_data = {}  # Store data for plotting
    
    for zone_cat in zone_categories:
        zone_records = [r for r in per_sample_records if r.get("zone_category") == zone_cat]
        if not zone_records:
            continue
        
        dominant_channels = [r[f"{map_name}_dominant_ch"] for r in zone_records]
        total_by_zone[zone_cat] = len(zone_records)
        n = len(zone_records)
        
        counts = [dominant_channels.count(i) for i in range(3)]
        zone_data[zone_cat] = counts
        pcts = [c / n * 100 for c in counts]
        print(f"{zone_cat:<20} {counts[0]:>4} ({pcts[0]:4.1f}%) {counts[1]:>4} ({pcts[1]:4.1f}%) {counts[2]:>4} ({pcts[2]:4.1f}%) {n:>8}") 
    # Create stacked bar chart for zone-based channel dominance
    if zone_data:
        zone_labels_filtered = list(zone_data.keys())
        t2w_counts = [zone_data[z][0] for z in zone_labels_filtered]
        adc_counts = [zone_data[z][1] for z in zone_labels_filtered]
        hbv_counts = [zone_data[z][2] for z in zone_labels_filtered]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(zone_labels_filtered))
        width = 0.6
        
        # Stacked bars
        ax.bar(x, t2w_counts, width, label='T2W', color='#e74c3c')
        ax.bar(x, adc_counts, width, bottom=t2w_counts, label='ADC', color='#2ecc71')
        ax.bar(x, hbv_counts, width, bottom=np.array(t2w_counts) + np.array(adc_counts), 
               label='HBV', color='#3498db')
        
        ax.set_xticks(x)
        ax.set_xticklabels(zone_labels_filtered, rotation=45, ha='right')
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'{map_name.capitalize()} — Channel Dominance by Tumor Zone')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'channel_dominance_by_zone_{map_name}.png', dpi=150)
        print(f"Saved channel_dominance_by_zone_{map_name}.png")
    
    print()

# 5) Zone-based average activation comparison
for map_name in ["saliency", "occlusion"]:
    zone_filtered_categories = [z for z in zone_categories 
                               if any(r.get("zone_category") == z for r in per_sample_records)]
    
    if not zone_filtered_categories:
        continue
    
    # Calculate average activation per channel for each zone
    zone_activations = []
    for zone_cat in zone_filtered_categories:
        zone_records = [r for r in per_sample_records if r.get("zone_category") == zone_cat]
        if not zone_records:
            continue
        means = np.stack([r[f"{map_name}_ch_mean"] for r in zone_records]).mean(axis=0)
        zone_activations.append(means)
    
    if zone_activations:
        zone_activations = np.array(zone_activations)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(zone_filtered_categories))
        width = 0.25
        
        ax.bar(x - width, zone_activations[:, 0], width, label='T2W', color='#e74c3c', alpha=0.8)
        ax.bar(x, zone_activations[:, 1], width, label='ADC', color='#2ecc71', alpha=0.8)
        ax.bar(x + width, zone_activations[:, 2], width, label='HBV', color='#3498db', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(zone_filtered_categories, rotation=45, ha='right')
        ax.set_ylabel('Mean Activation')
        ax.set_title(f'{map_name.capitalize()} — Average Channel Activation by Tumor Zone')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'channel_activation_by_zone_{map_name}.png', dpi=150)
        print(f"Saved channel_activation_by_zone_{map_name}.png")

# Save per-sample channel stats to JSON
channel_stats_export = []
for r in per_sample_records:
    entry = {
        "case_id": r["case_id"],
        "classification": r["classification"],
        "zone_category": r.get("zone_category", None),
        "zone_dominant": r.get("zone_dominant", None),
    }
    for map_name in ["saliency", "occlusion"]:
        for i, ch_name in enumerate(CHANNEL_NAMES):
            entry[f"{map_name}_{ch_name}_mean"] = float(r[f"{map_name}_ch_mean"][i])
            entry[f"{map_name}_{ch_name}_max"] = float(r[f"{map_name}_ch_max"][i])
            entry[f"{map_name}_{ch_name}_std"] = float(r[f"{map_name}_ch_std"][i])
            entry[f"{map_name}_{ch_name}_fraction"] = float(r[f"{map_name}_ch_fraction"][i])
        entry[f"{map_name}_dominant_channel"] = CHANNEL_NAMES[r[f"{map_name}_dominant_ch"]]
    channel_stats_export.append(entry)

with open(OUTPUT_DIR / "channel_statistics.json", "w") as f:
    json.dump(channel_stats_export, f, indent=2)
print(f"\nSaved per-sample channel statistics to channel_statistics.json")
