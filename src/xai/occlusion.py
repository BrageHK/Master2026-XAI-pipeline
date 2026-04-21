from typing import Optional, Tuple

import numpy as np
import torch


def _compute_zone_baseline_patches(
    image: np.ndarray,
    zones: np.ndarray,
    cancer_mask: np.ndarray,
    occ_window_dhw: Tuple[int, int, int],
    n_patches: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n non-cancerous patches per zone; return the patch with the median sum.

    From the n sampled patches, each patch's total intensity (sum across all
    channels and voxels) is computed.  The patch whose sum is closest to the
    median of those sums is selected as the representative baseline patch.

    Args:
        image: (3, D, H, W) float32 in DHW layout.
        zones: (D, H, W) int8, 0=bg 1=PZ 2=TZ.
        cancer_mask: (D, H, W) bool — predicted positive voxels (excluded).
        occ_window_dhw: (dW, hW, wW) — occlusion window size in DHW dims.
        n_patches: number of candidate patches to sample per zone.
        rng: optional numpy random generator (reproducibility).

    Returns:
        (tz_patch, pz_patch) each shape (3, dW, hW, wW) float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    dW, hW, wW = occ_window_dhw
    D, H, W = zones.shape

    results = []
    for z_val in (2, 1):  # TZ then PZ
        if n_patches == 0 or dW > D or hW > H or wW > W:
            results.append(np.zeros((3, dW, hW, wW), dtype=np.float32))
            continue

        zone_only = (zones == z_val) & (~cancer_mask)

        # Find valid anchor positions: center voxel must be in zone_only
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
            # Retry: use all zone voxels (ignore cancer exclusion)
            zone_all = (zones == z_val)
            valid2 = zone_all[dc, hc, wc]
            anchors_d = dd[valid2].ravel()
            anchors_h = hh[valid2].ravel()
            anchors_w = ww[valid2].ravel()

        if len(anchors_d) == 0:
            # Depth retry: iterate depth windows from most zone-dense to least,
            # using a looser criterion (zone exists anywhere in the window at its
            # H,W center position, rather than requiring zone at the 3-D center).
            zone_all = (zones == z_val)
            d_range = max(1, D - dW + 1)
            depth_scores = np.array([int(zone_all[d:d + dW].sum()) for d in range(d_range)])
            for d_try in np.argsort(depth_scores)[::-1]:
                if depth_scores[d_try] == 0:
                    break
                zone_hw = zone_all[d_try:d_try + dW].any(axis=0)  # (H, W)
                hh3, ww3 = np.meshgrid(
                    np.arange(H - hW + 1), np.arange(W - wW + 1), indexing="ij"
                )
                valid3 = zone_hw[hh3 + hW // 2, ww3 + wW // 2]
                if valid3.any():
                    anchors_d = np.full(int(valid3.sum()), d_try, dtype=np.intp)
                    anchors_h = hh3[valid3].ravel()
                    anchors_w = ww3[valid3].ravel()
                    print(f"    [zone_median] zone={z_val}: depth retry at d={d_try} "
                          f"({depth_scores[d_try]} zone voxels), {len(anchors_d)} anchors")
                    break

        if len(anchors_d) == 0:
            results.append(np.zeros((3, dW, hW, wW), dtype=np.float32))
            continue

        n_sample = min(n_patches, len(anchors_d))
        if n_sample < n_patches:
            print(f"    [zone_median] zone={z_val}: only {len(anchors_d)} valid anchors "
                  f"(requested {n_patches}), using all.")

        # Shuffle the full anchor pool once; each patch slot consumes from it in
        # order so the same anchor is never used twice.
        pool = rng.permutation(len(anchors_d))  # shuffled indices into anchors_d/h/w
        used = np.zeros(len(anchors_d), dtype=bool)
        patch_vol = dW * hW * wW

        patches = []
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
                frac_outside = float(
                    (zones[d0_:d0_ + dW, h0_:h0_ + hW, w0_:w0_ + wW] != z_val).sum()
                ) / patch_vol
                tries += 1

                if frac_outside <= 0.03:
                    best_anchor = pool_i
                    best_frac   = frac_outside
                    break                    # good enough — stop searching

                if frac_outside < best_frac:
                    best_frac   = frac_outside
                    best_anchor = pool_i

                if tries >= 100:
                    break                    # safety cap — use best found so far

            if best_anchor == -1:
                break                        # pool exhausted, no more patches

            used[best_anchor] = True
            if best_frac > 0.03:
                print(f"    [zone_median] zone={z_val} slot {_slot}: "
                      f"no patch ≤3% outside found after {tries} tries; "
                      f"best fit={100*(1-best_frac):.1f}% in zone")

            d0_ = int(anchors_d[best_anchor])
            h0_ = int(anchors_h[best_anchor])
            w0_ = int(anchors_w[best_anchor])
            patches.append(image[:, d0_:d0_ + dW, h0_:h0_ + hW, w0_:w0_ + wW].copy())

        if not patches:
            results.append(np.zeros((3, dW, hW, wW), dtype=np.float32))
            continue

        patch_sums = np.array([p.sum() for p in patches])          # (n_collected,)
        median_sum = np.median(patch_sums)
        rep_idx    = int(np.argmin(np.abs(patch_sums - median_sum)))  # closest to median
        results.append(patches[rep_idx].astype(np.float32))

    tz_patch, pz_patch = results
    return tz_patch, pz_patch


def _build_baseline_tensor(
    zones_spatial: np.ndarray,
    tz_patch: np.ndarray,
    pz_patch: np.ndarray,
    x_shape: Tuple,
    layout: str,
    device: torch.device,
) -> torch.Tensor:
    """Build a baseline tensor by tiling representative zone patches.

    The TZ representative patch is tiled across all TZ voxels; likewise for PZ.
    Background voxels remain 0.  When a sliding occlusion window lands entirely
    within one zone it is replaced by a spatially consistent sample of that
    zone's representative tissue.

    Args:
        zones_spatial: (D, H, W) int8 in DHW coordinate order.
        tz_patch: (3, dW, hW, wW) float32 — representative TZ patch.
        pz_patch: (3, dW, hW, wW) float32 — representative PZ patch.
        x_shape: full tensor shape, (1, 3, H, W, D) or (1, 3, D, H, W).
        layout: "hwd" for MONAI tensors (H, W, D last), "dhw" for nnUNet.
        device: target torch device.

    Returns:
        Baseline tensor of shape x_shape on device.
    """
    _, dW, hW, wW = tz_patch.shape
    D_z, H_z, W_z = zones_spatial.shape

    # Tile each patch to cover the full spatial extent in DHW
    def _tile(patch: np.ndarray, D: int, H: int, W: int) -> np.ndarray:
        # patch: (3, dW, hW, wW) → tiled (3, D, H, W)
        rD = D // dW + 1
        rH = H // hW + 1
        rW = W // wW + 1
        return np.tile(patch, (1, rD, rH, rW))[:, :D, :H, :W]

    tiled_tz_dhw = _tile(tz_patch, D_z, H_z, W_z)  # (3, D, H, W)
    tiled_pz_dhw = _tile(pz_patch, D_z, H_z, W_z)

    if layout == "hwd":
        # Transpose tiled patches and zones from DHW → HWD to match tensor spatial dims
        tiled_tz = tiled_tz_dhw.transpose(0, 2, 3, 1)           # (3, H, W, D)
        tiled_pz = tiled_pz_dhw.transpose(0, 2, 3, 1)
        zones_vol = zones_spatial.transpose(1, 2, 0)             # (H, W, D)
    else:
        tiled_tz  = tiled_tz_dhw                                 # (3, D, H, W)
        tiled_pz  = tiled_pz_dhw
        zones_vol = zones_spatial                                 # (D, H, W)

    spatial = x_shape[2:]
    baseline_np = np.zeros((3,) + spatial, dtype=np.float32)
    for c in range(3):
        baseline_np[c][zones_vol == 2] = tiled_tz[c][zones_vol == 2]
        baseline_np[c][zones_vol == 1] = tiled_pz[c][zones_vol == 1]

    return torch.from_numpy(baseline_np[np.newaxis]).to(device)  # (1, 3, ...)
