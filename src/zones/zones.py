import gc
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent.parent.resolve()

NNUNET_ROOT = _ROOT / "picai_nnunet"
UMAMBA_ROOT = _ROOT / "U_MambaMTL_XAI"

IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)
ZONES_BASE = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/picai_labels/anatomical_delineations/zonal_pz_tz/AI"
)
DEFAULT_OUTPUT_ZONES = _ROOT / "results" / "xai" / "zones"

sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))
sys.path.insert(0, str(UMAMBA_ROOT))


def _zones_from_monai_batch(
    batch: dict, d0: int, d1: int
) -> Optional[np.ndarray]:
    """
    Extract zones from MONAI DataModule batch.
    batch["zones"]: (1, 3, H, W, D)  — one-hot, channel 1=PZ, channel 2=TZ
    Returns (D_crop, H, W) int8 with 0=bg, 1=PZ, 2=TZ.
    """
    if "zones" not in batch:
        return None
    z = batch["zones"][0].cpu().numpy()  # (3, H, W, D)
    # argmax over channels: 0=bg, 1=PZ, 2=TZ — avoids boundary voxels being
    # lost to background when one-hot values are interpolated below 0.5.
    zones_hwD = np.argmax(z, axis=0).astype(np.int8)  # (H, W, D)
    zones_Dhw = zones_hwD.transpose(2, 0, 1)          # (D, H, W)
    return zones_Dhw[d0:d1]


def _zones_from_nnunet(
    case_id: str,
    plans: dict,
    prep_props: dict,
    d0: int,
    d1: int,
    h0: int,
    w0: int,
    occ_crop_hw: Optional[int],
) -> Optional[np.ndarray]:
    """
    Load the anatomical zone map for *case_id*, apply the same preprocessing
    as the image (transpose + resample + crop_bbox), remap encoding to
    0=bg, 1=PZ, 2=TZ (raw NIfTI has 1=TZ, 2=PZ), then apply the XAI crop.
    Returns (D_crop, H_crop, W_crop) int8.
    """
    import SimpleITK as sitk  # noqa: E402
    from nnUNet.nnunet.preprocessing.preprocessing import resample_data_or_seg  # noqa: E402

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

    # Step 1: resample zone to imagesTr voxel grid
    zone_itk = sitk.ReadImage(str(zone_path))
    ref_itk  = sitk.ReadImage(str(ref_path))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_itk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    zone_itk = resampler.Execute(zone_itk)

    itk_spacing = np.array(zone_itk.GetSpacing())[::-1]   # (D, H, W)
    zone_np     = sitk.GetArrayFromImage(zone_itk).astype(np.float32)

    # Step 2: apply plans transpose
    tp          = plans["transpose_forward"]
    zone_np     = zone_np.transpose(tp)
    itk_spacing = itk_spacing[list(tp)]

    # Step 3: resample to target spacing (nearest-neighbour)
    target_spacing = np.array(plans["plans_per_stage"][0]["current_spacing"])
    new_shape = np.round(
        itk_spacing / target_spacing * np.array(zone_np.shape)
    ).astype(int)
    zone_full = resample_data_or_seg(
        zone_np[np.newaxis], new_shape, is_seg=True, axis=None, order=0, do_separate_z=False
    )[0]
    zone_full = np.round(zone_full).astype(np.int8)

    # Step 4: apply crop_to_nonzero bbox (mirrors nnUNet preprocessing)
    crop_bbox = prep_props.get("crop_bbox")
    if crop_bbox is not None:
        orig_size = prep_props["original_size_of_raw_data"]
        slices = []
        for j in range(zone_full.ndim):
            pre_tp_axis   = tp[j]
            start, end    = crop_bbox[pre_tp_axis]
            scale         = zone_full.shape[j] / orig_size[pre_tp_axis]
            slices.append(slice(int(round(start * scale)), int(round(end * scale))))
        zone_full = zone_full[tuple(slices)]

    # Step 5: remap raw (0=bg, 1=TZ, 2=PZ) → standard (0=bg, 1=PZ, 2=TZ)
    zone_remapped = np.where(
        zone_full == 2, np.int8(1),
        np.where(zone_full == 1, np.int8(2), np.int8(0))
    ).astype(np.int8)

    # Step 6: apply XAI crop
    if occ_crop_hw is not None:
        return zone_remapped[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
    return zone_remapped[d0:d1]


def _gt_depth_crop(batch: dict, D: int) -> Tuple[int, int]:
    """Compute depth crop [d0, d1] from GT zones in a MONAI batch (existing behaviour)."""
    if "zones" in batch:
        z = batch["zones"][0].cpu().numpy()   # (3, H, W, D)
        zone_mask = np.argmax(z, axis=0) > 0  # (H, W, D) — True where any zone
        if zone_mask.any():
            coords = np.argwhere(zone_mask)    # (N, 3): (h, w, d)
            d_min  = int(coords[:, 2].min())
            d_max  = int(coords[:, 2].max())
            return max(0, d_min - 1), min(D, d_max + 2)
    return 0, D


def _load_umamba_zones(case_id: str, fold: int) -> Optional[dict]:
    """Load umamba predicted zones for *case_id* / *fold*.

    Returns a dict with keys:
      zones      (D, H, W) int8  – full zone prediction in MONAI space
      affine     (4, 4) float64  – MONAI RAS affine (may be None for old files)
      d0, d1     int             – depth crop indices (may be None for old files)
      zones_crop (D_crop, W, H) int8 – orientation-corrected cropped zones (may be None)
    Returns None if the file does not exist.
    """
    path = DEFAULT_OUTPUT_ZONES / f"fold_{fold}" / f"{case_id}.npz"
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    return {
        "zones":      raw["zones"],
        "affine":     raw["affine"]     if "affine"     in raw.files else None,
        "d0":         int(raw["d0"])    if "d0"         in raw.files else None,
        "d1":         int(raw["d1"])    if "d1"         in raw.files else None,
        "zones_crop": raw["zones_crop"] if "zones_crop" in raw.files else None,
    }


def _zones_from_umamba_npz(
    case_id: str,
    fold: int,
    plans: dict,
    prep_props: dict,
    d0: int,
    d1: int,
    h0: int,
    w0: int,
    occ_crop_hw: Optional[int],
) -> Optional[np.ndarray]:
    """Map umamba predicted zones (MONAI space) into nnUNet preprocessed space.

    Drop-in replacement for _zones_from_nnunet() when zone_source='umamba_pred'.
    Falls back to _zones_from_nnunet() on any failure.
    Returns (D_crop, H_crop, W_crop) int8 or None.
    """
    import tempfile
    import nibabel as nib
    import SimpleITK as sitk  # noqa: E402
    from nnUNet.nnunet.preprocessing.preprocessing import resample_data_or_seg  # noqa: E402

    zone_data = _load_umamba_zones(case_id, fold)
    if zone_data is None or zone_data["affine"] is None:
        print(f"    [umamba_pred] No zone file/affine for {case_id} — falling back to GT zones.")
        return _zones_from_nnunet(case_id, plans, prep_props, d0, d1, h0, w0, occ_crop_hw)

    ref_path = IMAGES_TR / f"{case_id}_0000.nii.gz"
    if not ref_path.exists():
        return None

    try:
        zones_Dhw  = zone_data["zones"]   # (D=20, H=128, W=128) int8
        affine_4x4 = zone_data["affine"]  # (4, 4) float64 MONAI RAS affine for (H,W,D) space

        # nibabel expects array in (H, W, D) to match the affine (which maps H/W/D → RAS)
        zones_HWD = zones_Dhw.transpose(1, 2, 0).astype(np.float32)  # (H=128, W=128, D=20)
        nii = nib.Nifti1Image(zones_HWD, affine=affine_4x4)

        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            nib.save(nii, tmp.name)
            zone_itk = sitk.ReadImage(tmp.name)

        # Resample to imagesTr voxel grid (same as step 1 in _zones_from_nnunet)
        ref_itk = sitk.ReadImage(str(ref_path))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_itk)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        zone_itk = resampler.Execute(zone_itk)

        # Steps 2–4: same pipeline as _zones_from_nnunet
        itk_spacing = np.array(zone_itk.GetSpacing())[::-1]   # (D, H, W)
        zone_np     = sitk.GetArrayFromImage(zone_itk).astype(np.float32)

        tp          = plans["transpose_forward"]
        zone_np     = zone_np.transpose(tp)
        itk_spacing = itk_spacing[list(tp)]

        target_spacing = np.array(plans["plans_per_stage"][0]["current_spacing"])
        new_shape = np.round(
            itk_spacing / target_spacing * np.array(zone_np.shape)
        ).astype(int)
        zone_full = resample_data_or_seg(
            zone_np[np.newaxis], new_shape, is_seg=True, axis=None, order=0, do_separate_z=False
        )[0]
        zone_full = np.round(zone_full).astype(np.int8)

        crop_bbox = prep_props.get("crop_bbox")
        if crop_bbox is not None:
            orig_size = prep_props["original_size_of_raw_data"]
            slices = []
            for j in range(zone_full.ndim):
                pre_tp_axis = tp[j]
                start, end  = crop_bbox[pre_tp_axis]
                scale       = zone_full.shape[j] / orig_size[pre_tp_axis]
                slices.append(slice(int(round(start * scale)), int(round(end * scale))))
            zone_full = zone_full[tuple(slices)]

        # No label remapping needed — umamba already uses 0=bg, 1=PZ, 2=TZ

        # Step 5: apply XAI crop
        if occ_crop_hw is not None:
            return zone_full[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
        return zone_full[d0:d1]

    except Exception as exc:
        print(f"    [umamba_pred] Resampling failed for {case_id}: {exc} — falling back to GT zones.")
        return _zones_from_nnunet(case_id, plans, prep_props, d0, d1, h0, w0, occ_crop_hw)


def _ensure_umamba_zones(fold: int, device: torch.device) -> None:
    """Forward-pass umamba to generate zone prediction files for any missing cases.

    Only does work when zone files are absent; skips cases that already have files.
    Called automatically before swin/nnunet processing when zone_source='umamba_pred'.
    """
    from shared_modules.data_module import DataModule   # noqa: E402
    from shared_modules.utils import load_config        # noqa: E402
    from src.models.loader import load_model

    zones_fold_dir = DEFAULT_OUTPUT_ZONES / f"fold_{fold}"
    zones_fold_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/umamba_mtl/config.yaml")
    config.data.json_list        = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.gpus                  = [device.index if device.type == "cuda" else 0]
    config.cache_rate            = 0.0
    config.transforms.label_keys = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()

    missing = [
        Path(b["image"].meta["filename_or_obj"][0]).name.split("_0000")[0]
        for b in dl
        if not (zones_fold_dir / f"{Path(b['image'].meta['filename_or_obj'][0]).name.split('_0000')[0]}.npz").exists()
    ]
    if not missing:
        print(f"  [ensure_umamba_zones] All zone files present for fold {fold}.")
        return

    print(f"  [ensure_umamba_zones] Generating zones for {len(missing)} missing cases in fold {fold}.")
    network = load_model("umamba_mtl", fold, device)

    # Re-iterate — a DataModule can only be set up once, recreate it
    dm2 = DataModule(config=config)
    dm2.setup("validation")
    for batch in dm2.val_dataloader():
        fname   = Path(batch["image"].meta["filename_or_obj"][0]).name
        case_id = fname.split("_0000")[0]
        if case_id not in missing:
            continue

        x = batch["image"].to(device)
        with torch.no_grad():
            out = network(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            zone_logits   = out[:, 2:5]                                    # (1, 3, H, W, D)
            zone_pred_hwD = zone_logits.softmax(dim=1).argmax(dim=1)[0]    # (H, W, D)

        zones_pred_Dhw = zone_pred_hwD.cpu().numpy().transpose(2, 0, 1).astype(np.int8)  # (D, H, W)
        monai_affine   = batch["image"].affine.cpu().numpy().astype(np.float64)           # (4, 4)

        # Depth crop from own zone predictions
        zone_present = (zones_pred_Dhw > 0).any(axis=(1, 2))  # (D,)
        if zone_present.any():
            d_idx = np.argwhere(zone_present)[:, 0]
            d0    = max(0, int(d_idx.min()) - 1)
            d1    = min(zones_pred_Dhw.shape[0], int(d_idx.max()) + 2)
        else:
            d0, d1 = 0, zones_pred_Dhw.shape[0]

        zones_crop_wh = zones_pred_Dhw[d0:d1].transpose(0, 2, 1)   # (D_crop, W, H)

        np.savez_compressed(
            zones_fold_dir / f"{case_id}.npz",
            zones      = zones_pred_Dhw,
            affine     = monai_affine,
            d0         = np.int32(d0),
            d1         = np.int32(d1),
            zones_crop = zones_crop_wh,
        )
        del x, out, zone_logits, zone_pred_hwD, zones_pred_Dhw, zones_crop_wh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del network
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [ensure_umamba_zones] Done.")
