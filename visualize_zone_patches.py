#!/usr/bin/env python3
"""
Visualize zone-median patch sampling for a single case.

Shows the prostate image slices with zone overlays and sampled patch locations,
alongside the actual patch content — lets you verify the zone_median occlusion
baseline looks anatomically reasonable.

Usage:
  uv run visualize_zone_patches.py                          # first available .npz
  uv run visualize_zone_patches.py path/to/case.npz
  uv run visualize_zone_patches.py --window 2 12 12 --n-patches 5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
IMAGES_DIR   = PROJECT_ROOT / "images"
XAI_DIR      = PROJECT_ROOT / "results" / "xai"

CHANNEL_NAMES  = ["T2W", "ADC", "HBV"]
CHANNEL_CMAPS  = ["gray", "YlOrRd", "YlGnBu"]
ZONE_COLORS    = {1: (0.2, 0.8, 0.2, 0.35), 2: (0.2, 0.4, 0.9, 0.35)}   # PZ=green TZ=blue
CANCER_COLOR   = (0.9, 0.1, 0.1, 0.55)
PATCH_COLORS   = {2: "#1a6fbf", 1: "#e07b00"}   # TZ=blue PZ=orange
ZONE_NAMES     = {1: "PZ", 2: "TZ"}


# ---------------------------------------------------------------------------
# Patch sampling (mirrors _compute_zone_medians but also returns anchors)
# ---------------------------------------------------------------------------

def sample_zone_patches(
    image: np.ndarray,
    zones: np.ndarray,
    cancer_mask: np.ndarray,
    occ_window_dhw: tuple,
    n_patches: int,
    rng: np.random.Generator,
) -> dict:
    """Return sampled anchors + patches + per-channel medians per zone.

    Returns dict keyed by zone (1=PZ, 2=TZ):
      {
        "anchors": list of (d0, h0, w0),
        "patches": np.ndarray (n, 3, dW, hW, wW),
        "median":  np.ndarray (3,),
      }
    """
    dW, hW, wW = occ_window_dhw
    D, H, W = zones.shape
    result = {}

    for z_val in (2, 1):
        if n_patches == 0 or dW > D or hW > H or wW > W:
            result[z_val] = {"anchors": [], "patches": np.zeros((0, 3, dW, hW, wW)), "median": np.zeros(3)}
            continue

        zone_only = (zones == z_val) & (~cancer_mask)

        d_anc = np.arange(D - dW + 1)
        h_anc = np.arange(H - hW + 1)
        w_anc = np.arange(W - wW + 1)
        dd, hh, ww = np.meshgrid(d_anc, h_anc, w_anc, indexing="ij")
        dc = dd + dW // 2
        hc = hh + hW // 2
        wc = ww + wW // 2
        valid = zone_only[dc, hc, wc]
        anchors_d = dd[valid].ravel()
        anchors_h = hh[valid].ravel()
        anchors_w = ww[valid].ravel()

        if len(anchors_d) == 0:
            zone_all = (zones == z_val)
            valid2 = zone_all[dc, hc, wc]
            anchors_d = dd[valid2].ravel()
            anchors_h = hh[valid2].ravel()
            anchors_w = ww[valid2].ravel()
            print(f"  zone={ZONE_NAMES[z_val]}: no non-cancer anchors, using all zone voxels")

        if len(anchors_d) == 0:
            print(f"  zone={ZONE_NAMES[z_val]}: no valid anchors at all")
            result[z_val] = {"anchors": [], "patches": np.zeros((0, 3, dW, hW, wW)), "median": np.zeros(3)}
            continue

        n_sample = min(n_patches, len(anchors_d))
        if n_sample < n_patches:
            print(f"  zone={ZONE_NAMES[z_val]}: only {len(anchors_d)} anchors, using all")

        pool      = rng.permutation(len(anchors_d))
        used      = np.zeros(len(anchors_d), dtype=bool)
        patch_vol = dW * hW * wW

        anchors = []
        patches = []
        frac_outside_list = []

        for _slot in range(n_sample):
            best_anchor = -1
            best_frac   = 1.0
            tries       = 0

            for pool_i in pool:
                if used[pool_i]:
                    continue
                d0_ = int(anchors_d[pool_i])
                h0_ = int(anchors_h[pool_i])
                w0_ = int(anchors_w[pool_i])
                frac = float(
                    (zones[d0_:d0_ + dW, h0_:h0_ + hW, w0_:w0_ + wW] != z_val).sum()
                ) / patch_vol
                tries += 1
                if frac <= 0.03:
                    best_anchor = pool_i
                    best_frac   = frac
                    break
                if frac < best_frac:
                    best_frac   = frac
                    best_anchor = pool_i
                if tries >= 100:
                    break

            if best_anchor == -1:
                break

            used[best_anchor] = True
            if best_frac > 0.03:
                print(f"  zone={ZONE_NAMES[z_val]} slot {_slot}: "
                      f"no patch ≤3% outside after {tries} tries; "
                      f"best fit={100*(1-best_frac):.1f}% in zone")

            d0_ = int(anchors_d[best_anchor])
            h0_ = int(anchors_h[best_anchor])
            w0_ = int(anchors_w[best_anchor])
            patch = image[:, d0_:d0_ + dW, h0_:h0_ + hW, w0_:w0_ + wW].copy()
            anchors.append((d0_, h0_, w0_))
            patches.append(patch)
            frac_outside_list.append(best_frac)

        if not patches:
            result[z_val] = {"anchors": [], "patches": np.zeros((0, 3, dW, hW, wW)),
                             "patch_sums": np.array([]), "rep_idx": 0,
                             "frac_outside": np.array([])}
            continue

        patch_sums = np.array([p.sum() for p in patches])
        median_sum = np.median(patch_sums)
        rep_idx    = int(np.argmin(np.abs(patch_sums - median_sum)))

        result[z_val] = {
            "anchors":      anchors,
            "patches":      np.stack(patches, axis=0),
            "patch_sums":   patch_sums,
            "rep_idx":      rep_idx,
            "frac_outside": np.array(frac_outside_list),
        }

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_representative_slice(zones: np.ndarray, z_val: int) -> int:
    """Depth index with the most voxels of the given zone."""
    counts = (zones == z_val).sum(axis=(1, 2))
    return int(np.argmax(counts))


def _norm(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] for display."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _zone_rgba(zones_2d: np.ndarray) -> np.ndarray:
    """Build a (H, W, 4) RGBA overlay from a 2-D zone slice."""
    H, W = zones_2d.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    for z_val, color in ZONE_COLORS.items():
        mask = zones_2d == z_val
        rgba[mask] = color
    return rgba


def _cancer_rgba(cancer_2d: np.ndarray) -> np.ndarray:
    """Build a (H, W, 4) RGBA overlay for the cancer prediction mask."""
    H, W = cancer_2d.shape
    rgba = np.zeros((H, W, 4), dtype=float)
    rgba[cancer_2d] = CANCER_COLOR
    return rgba


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_context_slice(
    ax: plt.Axes,
    image: np.ndarray,
    zones: np.ndarray,
    cancer_mask: np.ndarray,
    d_slice: int,
    anchors: list,
    occ_window_dhw: tuple,
    z_val: int,
    title: str = "",
) -> None:
    """Axial overview slice with zone+cancer overlay and patch bboxes."""
    dW, hW, wW = occ_window_dhw
    img_slice = _norm(image[0, d_slice])   # T2W
    ax.imshow(img_slice, cmap="gray", origin="upper")
    ax.imshow(_zone_rgba(zones[d_slice]), origin="upper")
    ax.imshow(_cancer_rgba(cancer_mask[d_slice]), origin="upper")

    color = PATCH_COLORS[z_val]
    for k, (d0, h0, w0) in enumerate(anchors):
        # Only draw boxes whose depth range includes d_slice
        if d0 <= d_slice < d0 + dW:
            rect = mpatches.Rectangle(
                (w0 - 0.5, h0 - 0.5), wW, hW,
                linewidth=1.5, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(w0 + wW / 2, h0 - 1.5, str(k), color=color,
                    fontsize=7, ha="center", va="bottom", fontweight="bold")

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=ZONE_COLORS[2][:3], label="TZ", alpha=0.7),
        mpatches.Patch(color=ZONE_COLORS[1][:3], label="PZ", alpha=0.7),
        mpatches.Patch(color=CANCER_COLOR[:3], label="Cancer pred", alpha=0.7),
        mpatches.Patch(edgecolor=color, facecolor="none", label=f"{ZONE_NAMES[z_val]} patches"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6, framealpha=0.7)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def plot_patch_row(
    axes: list,
    patch: np.ndarray,
    anchor: tuple,
    patch_idx: int,
    occ_window_dhw: tuple,
) -> None:
    """Fill one row of axes with the 3-channel middle-depth slice of a patch."""
    dW, hW, wW = occ_window_dhw
    d_mid = dW // 2
    d0, h0, w0 = anchor
    for ch, ax in enumerate(axes):
        sl = _norm(patch[ch, d_mid])  # (hW, wW)
        ax.imshow(sl, cmap=CHANNEL_CMAPS[ch], origin="upper", vmin=0, vmax=1)
        ax.set_title(f"{CHANNEL_NAMES[ch]}", fontsize=8)
        ax.set_xlabel(f"d={d0+d_mid} h={h0}:{h0+hW} w={w0}:{w0+wW}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])


def make_zone_figure(
    image: np.ndarray,
    zones: np.ndarray,
    cancer_mask: np.ndarray,
    zone_data: dict,
    occ_window_dhw: tuple,
    z_val: int,
    case_id: str,
) -> plt.Figure:
    """One figure per zone showing context overview + patch rows."""
    dW, hW, wW = occ_window_dhw
    anchors    = zone_data["anchors"]
    patches    = zone_data["patches"]     # (n, 3, dW, hW, wW)
    patch_sums = zone_data["patch_sums"]  # (n,)
    rep_idx    = zone_data["rep_idx"]
    n = len(anchors)
    zone_name = ZONE_NAMES[z_val]

    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.text(0.5, 0.5, f"No {zone_name} patches found", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(f"{case_id} — {zone_name} patches", fontsize=11)
        return fig

    d_rep = _pick_representative_slice(zones, z_val)

    # Layout: 1 overview row + n patch rows; 4 columns (overview, T2W, ADC, HBV)
    n_rows = 1 + n
    n_cols = 4

    # Overview column spans full height → use gridspec
    fig = plt.figure(figsize=(n_cols * 2.8, n_rows * 2.5 + 0.8))
    gs = fig.add_gridspec(
        n_rows, n_cols,
        hspace=0.45, wspace=0.25,
        top=0.92, bottom=0.05, left=0.04, right=0.97,
    )

    # Top row: overview slice (left) + representative patch middle slice (right 3 cols)
    rep_patch  = patches[rep_idx]           # (3, dW, hW, wW)
    rep_anchor = anchors[rep_idx]
    rep_sum    = patch_sums[rep_idx]
    median_sum = float(np.median(patch_sums))

    ax_ctx = fig.add_subplot(gs[0, 0])
    plot_context_slice(
        ax_ctx, image, zones, cancer_mask, d_rep,
        anchors, occ_window_dhw, z_val,
        title=f"Slice d={d_rep}  ({zone_name} anchors shown)",
    )

    d_mid = dW // 2
    for ch in range(3):
        ax_sw = fig.add_subplot(gs[0, ch + 1])
        patch_slice = _norm(rep_patch[ch, d_mid])   # (hW, wW) — middle depth of rep patch
        ax_sw.imshow(patch_slice, cmap=CHANNEL_CMAPS[ch], origin="upper", vmin=0, vmax=1)
        ax_sw.set_title(
            f"Rep. patch  {CHANNEL_NAMES[ch]}\n"
            f"sum={rep_sum:.2f}  (median={median_sum:.2f})",
            fontsize=8,
        )
        ax_sw.set_xticks([]); ax_sw.set_yticks([])

    # Patch rows
    for row_i, (anchor, patch) in enumerate(zip(anchors, patches)):
        axes_row = [fig.add_subplot(gs[row_i + 1, col]) for col in range(n_cols)]
        is_rep = (row_i == rep_idx)

        # First cell: zoomed context at the patch's center depth
        d_center = anchor[0] + dW // 2
        h0, w0 = anchor[1], anchor[2]
        pad = max(hW, wW) // 2
        H_img, W_img = image.shape[2], image.shape[3]
        h_lo = max(0, h0 - pad); h_hi = min(H_img, h0 + hW + pad)
        w_lo = max(0, w0 - pad); w_hi = min(W_img, w0 + wW + pad)
        zoom_img = _norm(image[0, d_center, h_lo:h_hi, w_lo:w_hi])
        axes_row[0].imshow(zoom_img, cmap="gray", origin="upper")
        axes_row[0].imshow(
            _zone_rgba(zones[d_center, h_lo:h_hi, w_lo:w_hi]), origin="upper"
        )
        axes_row[0].imshow(
            _cancer_rgba(cancer_mask[d_center, h_lo:h_hi, w_lo:w_hi]), origin="upper"
        )
        # Draw patch bbox; double width + star label if this is the representative
        edge_color = "gold" if is_rep else PATCH_COLORS[z_val]
        lw = 3 if is_rep else 1.5
        rect = mpatches.Rectangle(
            (w0 - w_lo - 0.5, h0 - h_lo - 0.5), wW, hW,
            linewidth=lw, edgecolor=edge_color, facecolor="none",
        )
        axes_row[0].add_patch(rect)
        label = f"★ {row_i} (rep)" if is_rep else str(row_i)
        axes_row[0].set_title(
            f"{label}  d={d_center}  sum={patch_sums[row_i]:.1f}", fontsize=8,
            color="goldenrod" if is_rep else "black",
        )
        axes_row[0].axis("off")

        plot_patch_row(axes_row[1:], patch, anchor, row_i, occ_window_dhw)

    fig.suptitle(
        f"{case_id}  |  {zone_name} patches  (window={dW}×{hW}×{wW}  n={n})",
        fontsize=11, fontweight="bold",
    )
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "npz", nargs="?", type=str, default=None,
        help="Path to a .npz file. Default: first available in results/xai/.",
    )
    parser.add_argument(
        "--window", nargs=3, type=int, default=[2, 12, 12],
        metavar=("D", "H", "W"),
        help="Occlusion window size (D H W). Default: 2 12 12.",
    )
    parser.add_argument(
        "--n-patches", type=int, default=10,
        help="Number of patches to sample per zone. Default: 10.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Find .npz file
    if args.npz:
        npz_path = Path(args.npz)
    else:
        candidates = sorted(XAI_DIR.glob("**/fold_*/*.npz"))
        # Prefer files with zones
        for p in candidates:
            data = np.load(p, allow_pickle=True)
            if "zones" in data and data["zones"].ndim == 3:
                npz_path = p
                break
        else:
            if not candidates:
                print("No .npz files found in results/xai/. Run generate_xai_data.py first.")
                sys.exit(1)
            npz_path = candidates[0]

    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    image   = data["image"]       # (3, D, H, W)
    zones   = data["zones"]       # (D, H, W) int8  or empty sentinel
    pred    = data["prediction"]  # (1, D, H, W)
    case_id = str(data["case_id"])

    if zones.ndim != 3:
        print(f"ERROR: zones array has shape {zones.shape} — no valid zone data in this file.")
        print("Try a different .npz (e.g. from umamba_mtl or nnunet with zone labels).")
        sys.exit(1)

    cancer_mask = (pred[0] > 0.5)   # (D, H, W)
    occ_window_dhw = tuple(args.window)
    rng = np.random.default_rng(args.seed)

    print(f"  Image:        {image.shape}")
    print(f"  Zones:        {zones.shape}  unique={np.unique(zones)}")
    print(f"  Cancer voxels:{cancer_mask.sum()}")
    print(f"  Window:       {occ_window_dhw}   n_patches={args.n_patches}")

    zone_data = sample_zone_patches(
        image, zones, cancer_mask, occ_window_dhw, args.n_patches, rng
    )

    for z_val, zd in zone_data.items():
        n = len(zd["anchors"])
        if n > 0:
            rep = zd["rep_idx"]
            outside_pct = zd["frac_outside"] * 100
            print(f"  {ZONE_NAMES[z_val]}: {n} patches  "
                  f"outside%={outside_pct.round(1)}  "
                  f"representative=patch[{rep}] (sum={zd['patch_sums'][rep]:.2f})")

    # Save figures
    IMAGES_DIR.mkdir(exist_ok=True)
    saved = []
    for z_val in (2, 1):
        fig = make_zone_figure(
            image, zones, cancer_mask,
            zone_data[z_val], occ_window_dhw, z_val,
            case_id,
        )
        out_path = IMAGES_DIR / f"zone_patches_{case_id}_{ZONE_NAMES[z_val]}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"  Saved: {out_path}")

    print(f"\nDone. {len(saved)} figures saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
