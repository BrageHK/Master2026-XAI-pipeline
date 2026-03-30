#!/usr/bin/env python3
"""
Compute and save TZ/PZ zone crops aligned to each existing XAI npz result.

For each .npz in results/xai/fold_*/, this script:
  1. Loads the stored image crop from the npz.
  2. Runs preprocess_case() to get the full preprocessed volume.
  3. Template-matches the stored crop against the full volume to recover the
     exact (d0, d1, h0, w0) crop coordinates — no model/GPU inference needed.
  4. Loads the anatomical zone map (TZ=1, PZ=2) for the case.
  5. Preprocesses the zone to the same spatial grid as preprocess_case().
  6. Crops the zone to match the stored npz image crop exactly.
  7. Saves the cropped zone to results/xai/zone_crops/fold_X/{case_id}.npz

Output npz keys:
    zone_crop    : (D_crop, H_crop, W_crop) int8 — 0=bg, 1=TZ, 2=PZ
    case_id      : str
    d0, d1       : int — depth crop bounds in preprocessed space
    h0, w0       : int — H/W crop origin in preprocessed space
    occ_crop_hw  : int — H and W crop size

Usage:
    uv run python compute_zone_crops.py              # all folds
    uv run python compute_zone_crops.py --fold 0     # single fold
    uv run python compute_zone_crops.py --no-skip    # recompute existing
"""

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------------
# Paths — mirror captum_xai.py exactly (env vars must precede nnunet imports)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

NNUNET_PREPROCESSED = PROJECT_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = PROJECT_ROOT / "results" / "nnUNet"

IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)
ZONES_BASE = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/picai_labels/anatomical_delineations/zonal_pz_tz/AI"
)

XAI_DIR        = PROJECT_ROOT / "results" / "xai"
ZONE_CROPS_DIR = PROJECT_ROOT / "results" / "xai" / "zone_crops"

os.environ["RESULTS_FOLDER"]        = str(NNUNET_RESULTS)
os.environ["nnUNet_raw_data_base"]  = str(PROJECT_ROOT / "nnunet_base")
os.environ["nnUNet_preprocessed"]   = str(NNUNET_PREPROCESSED)

sys.path.insert(0, str(PROJECT_ROOT / "nnUNet"))
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared utilities from captum_xai (after env vars)
from captum_xai import load_plans, preprocess_case_with_props  # noqa: E402
from nnUNet.nnunet.preprocessing.preprocessing import resample_data_or_seg  # noqa: E402


# ---------------------------------------------------------------------------
# Zone loading
# ---------------------------------------------------------------------------
def load_zone_for_case(case_id: str, plans: dict) -> Optional[np.ndarray]:
    """
    Load the anatomical zone map for *case_id* and resample it to the same
    spatial grid as preprocess_case() output.

    Two-step pipeline:
      1. Resample from original T2W space to imagesTr voxel grid via SimpleITK
         (zone and imagesTr share origin/direction, so SetReferenceImage is exact).
      2. Apply the same axis transpose + spacing resample as nnUNet's
         GenericPreprocessor, using nearest-neighbour to preserve 0/1/2 labels.

    Returns (D, H, W) int8 with 0=bg, 1=TZ, 2=PZ, or None if unavailable.
    """
    zone_path = None
    for ver in ("Yuan23", "HeviAI23"):
        p = ZONES_BASE / ver / f"{case_id}.nii.gz"
        if p.exists():
            zone_path = p
            break
    if zone_path is None:
        return None

    ref_path = IMAGES_TR / f"{case_id}_0000.nii.gz"
    if not ref_path.exists():
        return None

    # Step 1: resample zone to imagesTr space
    zone_itk = sitk.ReadImage(str(zone_path))
    ref_itk   = sitk.ReadImage(str(ref_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_itk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    zone_imagestr_itk = resampler.Execute(zone_itk)

    # ITK spacing is (x, y, z); convert to numpy (z, y, x) = (D, H, W) order
    itk_spacing = np.array(zone_imagestr_itk.GetSpacing())[::-1]
    zone_np     = sitk.GetArrayFromImage(zone_imagestr_itk).astype(np.float32)

    # Step 2: apply plans transpose_forward
    tp          = plans["transpose_forward"]
    zone_np     = zone_np.transpose(tp)
    itk_spacing = itk_spacing[list(tp)]

    # Compute target shape (mirrors load_label_for_case logic)
    target_spacing = np.array(plans["plans_per_stage"][0]["current_spacing"])
    new_shape = np.round(
        itk_spacing / target_spacing * np.array(zone_np.shape)
    ).astype(int)

    # Nearest-neighbour resample to preserve discrete labels 0, 1, 2
    zone_resampled = resample_data_or_seg(
        zone_np[np.newaxis],   # (1, D, H, W)
        new_shape,
        is_seg=True,
        axis=None,
        order=0,               # nearest-neighbour — must be 0 for multi-class
        do_separate_z=False,
    )[0]  # (D, H, W)

    return np.round(zone_resampled).astype(np.int8)


# ---------------------------------------------------------------------------
# Crop coordinate recovery via template matching
# ---------------------------------------------------------------------------
def find_crop_coords(
    stored_img: np.ndarray,
    data: np.ndarray,
) -> Tuple[int, int, int, int]:
    """
    Recover (d0, d1, h0, w0) by template-matching the stored image crop
    against the full preprocessed volume. No GPU or model inference required.

    stored_img : (3, D_crop, H_crop, W_crop) float32 — from npz["image"]
    data       : (3, D, H, W) float32 — from preprocess_case()

    Strategy:
      1. Depth-mean of T2W channel → 2-D FFT cross-correlation to find h0, w0.
      2. Mean over H/W at (h0, w0) → 1-D cross-correlation along depth to find d0.
    """
    D_crop, H_crop = stored_img.shape[1], stored_img.shape[2]

    # Step 1: find h0, w0 via 2-D zero-mean cross-correlation on T2W depth-mean.
    # Subtracting the template mean makes the correlation robust to additive
    # intensity offsets between preprocessing runs: sum(ref * tmpl_zm) is
    # unaffected by a constant added to ref because sum(tmpl_zm) == 0.
    tmpl_2d = stored_img[0].mean(axis=0)        # (H_crop, W_crop)
    tmpl_2d = tmpl_2d - tmpl_2d.mean()          # zero-mean template
    ref_2d  = data[0].mean(axis=0)              # (H, W)
    ref_2d  = ref_2d - ref_2d.mean()            # zero-mean reference
    corr2d  = fftconvolve(ref_2d, tmpl_2d[::-1, ::-1], mode="valid")
    h0, w0  = np.unravel_index(int(corr2d.argmax()), corr2d.shape)

    # Step 2: find d0 via 1-D zero-mean cross-correlation of depth profiles at (h0, w0)
    ref_col  = data[0, :, h0:h0 + H_crop, w0:w0 + H_crop].mean(axis=(1, 2))   # (D,)
    tmpl_col = stored_img[0].mean(axis=(1, 2))                                  # (D_crop,)
    ref_col  = ref_col - ref_col.mean()
    tmpl_col = tmpl_col - tmpl_col.mean()
    corr1d   = np.correlate(ref_col, tmpl_col, mode="valid")                   # (D-D_crop+1,)
    d0       = int(corr1d.argmax())
    d1       = d0 + D_crop

    return d0, d1, int(h0), int(w0)


# ---------------------------------------------------------------------------
# Per-fold processing
# ---------------------------------------------------------------------------
def process_fold(fold: int, skip_existing: bool, plans: dict) -> None:
    fold_xai_dir   = XAI_DIR        / f"fold_{fold}"
    fold_zones_dir = ZONE_CROPS_DIR / f"fold_{fold}"

    if not fold_xai_dir.exists():
        print(f"  Fold {fold}: XAI dir not found at {fold_xai_dir} — skipping")
        return

    npz_files = sorted(fold_xai_dir.glob("*.npz"))
    if not npz_files:
        print(f"  Fold {fold}: no npz files found — skipping")
        return

    fold_zones_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Fold {fold} — {len(npz_files)} case(s)")

    ok, skipped, no_zone, errors = 0, 0, 0, 0

    for npz_path in npz_files:
        case_id  = npz_path.stem
        out_path = fold_zones_dir / f"{case_id}.npz"

        if skip_existing and out_path.exists():
            skipped += 1
            continue

        print(f"\n  {case_id}")
        try:
            # Load stored image crop
            stored      = np.load(npz_path, allow_pickle=True)
            img_arr     = stored["image"]        # (3, D_crop, H_crop, W_crop)
            occ_crop_hw = img_arr.shape[2]

            # Load and preprocess zone map
            zone_full = load_zone_for_case(case_id, plans)
            if zone_full is None:
                print(f"    No zone file — skipping")
                no_zone += 1
                continue

            # Preprocess MRI to get the full spatial volume for template matching
            data, prep_props = preprocess_case_with_props(case_id, plans)
            print(f"    Preprocessed shape: {data.shape}  zone shape: {zone_full.shape}")

            # Apply the same crop_to_nonzero bbox that nnUNet applied to the image.
            # crop_bbox is in pre-transpose space; zone_full is post-transpose/post-resample.
            crop_bbox = prep_props.get("crop_bbox")
            if crop_bbox is not None:
                tp = plans["transpose_forward"]
                orig_size = prep_props["original_size_of_raw_data"]  # (D, H, W) pre-transpose
                slices = []
                for j in range(zone_full.ndim):
                    pre_tp_axis = tp[j]
                    bbox_start, bbox_end = crop_bbox[pre_tp_axis]
                    scale = zone_full.shape[j] / orig_size[pre_tp_axis]
                    slices.append(slice(int(round(bbox_start * scale)), int(round(bbox_end * scale))))
                zone_full = zone_full[tuple(slices)]
                print(f"    Zone after bbox crop: {zone_full.shape}")

            # Recover crop coordinates — prefer saved keys (written by updated
            # captum_xai.py), fall back to prostate centroid from the zone map.
            D_crop = img_arr.shape[1]
            if "d0" in stored:
                d0 = int(stored["d0"])
                d1 = int(stored["d1"])
                h0 = int(stored["h0"])
                w0 = int(stored["w0"])
                print(f"    Crop from npz: d={d0}:{d1} ({d1-d0} slices)  h0={h0}  w0={w0}  hw={occ_crop_hw}")
            else:
                # Zone-centroid fallback: centre the crop on the prostate.
                # Cancer sits inside TZ or PZ, so zone centroid ≈ XAI crop centre.
                D, H, W = zone_full.shape
                zone_coords = np.argwhere(zone_full > 0)
                if len(zone_coords) > 0:
                    dc, hc, wc = zone_coords.mean(axis=0).astype(int)
                else:
                    dc, hc, wc = D // 2, H // 2, W // 2
                d0 = int(np.clip(dc - D_crop // 2, 0, D - D_crop))
                d1 = d0 + D_crop
                h0 = int(np.clip(hc - occ_crop_hw // 2, 0, H - occ_crop_hw))
                w0 = int(np.clip(wc - occ_crop_hw // 2, 0, W - occ_crop_hw))
                print(f"    Crop from zone centroid: d={d0}:{d1} ({d1-d0} slices)  h0={h0}  w0={w0}  hw={occ_crop_hw}")

            # Crop zone to match the stored npz image exactly
            print(f"    Zone full: {zone_full.shape}  unique values: {np.unique(zone_full)}")
            zone_crop = zone_full[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
            print(f"    Zone crop: {zone_crop.shape}  unique values: {np.unique(zone_crop)}")

            np.savez_compressed(
                out_path,
                zone_crop   = zone_crop,        # (D_crop, H_crop, W_crop) int8
                case_id     = case_id,
                d0          = d0,
                d1          = d1,
                h0          = h0,
                w0          = w0,
                occ_crop_hw = occ_crop_hw,
            )
            print(f"    Saved: {out_path}")
            ok += 1

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1

    print(
        f"\nFold {fold} done — "
        f"ok={ok}, skipped={skipped}, no_zone={no_zone}, errors={errors}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save zone crops for existing XAI npz results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    fold_group = parser.add_mutually_exclusive_group()
    fold_group.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        metavar="N",
        help="Process only fold N (0–4).",
    )
    fold_group.add_argument(
        "--folds",
        type=str,
        metavar="N,N,...",
        help="Comma-separated list of folds (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Recompute even if zone crop already exists.",
    )
    args = parser.parse_args()

    if args.fold is not None:
        folds = [args.fold]
    elif args.folds is not None:
        folds = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds = list(range(5))

    plans = load_plans()
    print(f"Plans loaded. Target spacing: {plans['plans_per_stage'][0]['current_spacing']}")
    print(f"XAI input:    {XAI_DIR}")
    print(f"Zone output:  {ZONE_CROPS_DIR}")
    print(f"Folds:        {folds}")

    ZONE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        process_fold(fold, skip_existing=not args.no_skip, plans=plans)

    print("\nAll done.")


if __name__ == "__main__":
    main()
