#!/usr/bin/env python3
"""
Unified XAI pipeline for all PI-CAI models.

For each model (umamba_mtl, swin_unetr, nnunet) and fold:
  - Loads the validation split cases
  - The model's existing pipeline crops to the prostate region
  - Runs inference; generates XAI attribution maps ONLY when the model predicts positive
  - Saves per-case .npz with image, prediction, label, zones, and attribution maps
  - Computes classification metrics (TP/FP/TN/FN, zone stats, channel activations)
  - Generates charts (confusion matrix, zone distribution, channel activation)

Zone encoding in saved .npz: 0 = background, 1 = PZ, 2 = TZ.

Usage:
  python generate_xai_data.py --models umamba_mtl --fold 0 --methods saliency
  python generate_xai_data.py --models umamba_mtl swin_unetr nnunet --fold 0,1,2,3,4 --methods saliency occlusion
  python generate_xai_data.py --models all --fold 0 --methods all
  python generate_xai_data.py --models swin_unetr --compute-metrics-only
"""

import argparse
import gc
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from memory_profiler import profile
import objgraph
from captum.attr import Occlusion, Saliency
from src.ablation_cam_3d import AblationCAM3D, find_decoder_feature_layers

from src.utils import load_plans as _load_plans_from_file, log_large_vars

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Outputs
DEFAULT_OUTPUT_XAI     = PROJECT_ROOT / "results" / "xai"
DEFAULT_OUTPUT_METRICS = PROJECT_ROOT / "results" / "metrics"
DEFAULT_OUTPUT_ZONES   = PROJECT_ROOT / "results" / "xai" / "zones"

CHANNEL_NAMES = ["t2w", "adc", "hbv"]

# nnUNet paths
NNUNET_ROOT         = PROJECT_ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_ROOT / "results" / "nnUNet"
TASK_NAME           = "Task2203_picai_baseline"
DATASET_DIR         = NNUNET_PREPROCESSED / TASK_NAME
PLANS_FILE          = NNUNET_RESULTS / "plans.pkl"
SPLITS_DIR          = NNUNET_ROOT / "splits"

IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)
LABELS_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/labelsTr"
)
ZONES_BASE = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/picai_labels/anatomical_delineations/zonal_pz_tz/AI"
)

# U-MambaMTL / SwinUNETR paths
UMAMBA_ROOT    = PROJECT_ROOT / "U_MambaMTL_XAI"
CHECKPOINT_DIR = UMAMBA_ROOT / "gc_algorithms" / "base_container" / "models"

# Set nnUNet env vars before any nnunet import
os.environ.setdefault("RESULTS_FOLDER",        str(NNUNET_RESULTS))
os.environ.setdefault("nnUNet_raw_data_base",  str(NNUNET_ROOT / "nnunet_base"))
os.environ.setdefault("nnUNet_preprocessed",   str(NNUNET_PREPROCESSED))

# Add subproject roots to path so internal imports work
sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))
sys.path.insert(0, str(UMAMBA_ROOT))

# ---------------------------------------------------------------------------
# Metrics/charts constants
# ---------------------------------------------------------------------------
METHODS       = ["saliency", "occlusion"]
CLASS_FILTERS = ["tp", "fp", "both"]
ZONE_FILTERS  = ["pz", "tz", "pz_dominated", "combined"]

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


# ===========================================================================
# Model loaders
# ===========================================================================

def _ensure_focal_loss_importable() -> None:
    """
    Patch sys.modules so nnunet's focal-loss import resolves to the project-root file
    (the bundled nnUNet submodule ships a different version that misses FocalLoss).
    """
    module_key = (
        "nnunet.training.network_training.nnUNet_variants"
        ".loss_function.nnUNetTrainerV2_focalLoss"
    )
    if module_key in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        module_key,
        str(NNUNET_ROOT / "nnUNet_addon" / "nnUNetTrainerV2_focalLoss.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


def load_nnunet(fold: int, device: torch.device) -> torch.nn.Module:
    """Load the nnUNet model for *fold* and return it in eval mode."""
    _ensure_focal_loss_importable()

    from nnUNet_addon.nnUNetTrainerV2_Loss_FL_and_CE import (  # noqa: E402
        nnUNetTrainerV2_Loss_FL_and_CE_checkpoints,
    )

    pkl_path   = NNUNET_RESULTS / f"fold_{fold}" / "model_best.model.pkl"
    model_path = NNUNET_RESULTS / f"fold_{fold}" / "model_best.model"

    with open(pkl_path, "rb") as f:
        info = pickle.load(f)

    tmp_out = tempfile.mkdtemp(prefix=f"nnunet_fold{fold}_")

    trainer = nnUNetTrainerV2_Loss_FL_and_CE_checkpoints(
        plans_file=str(PLANS_FILE),
        fold=fold,
        output_folder=tmp_out,
        dataset_directory=str(DATASET_DIR),
        batch_dice=False,
        stage=0,
        unpack_data=True,
        deterministic=False,
        fp16=False,
    )
    trainer.process_plans(info["plans"])
    trainer.initialize(False)

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    trainer.load_checkpoint_ram(checkpoint, False)

    network: torch.nn.Module = trainer.network
    network.eval()
    if hasattr(network, "do_ds"):
        network.do_ds = False

    # CUDA with smoke-test fallback
    if torch.cuda.is_available():
        try:
            network = network.cuda()
            network(torch.zeros(1, 3, 16, 64, 64, device="cuda"))
        except Exception:
            print("  WARNING: CUDA forward pass failed; falling back to CPU.")
            network = network.cpu()
    else:
        network = network.cpu()

    return network


def load_mamba(model_name: str, fold: int, device: torch.device) -> torch.nn.Module:
    """Load a U-MambaMTL or SwinUNETR checkpoint and return the network in eval mode."""
    from shared_modules.utils import load_config  # noqa: E402

    if model_name.lower() == "umamba_mtl":
        from experiments.picai.umamba_mtl.trainer import LitModel
    elif model_name.lower() == "swin_unetr":
        from experiments.picai.swin_unetr.trainer import LitModel
    else:
        raise ValueError(f"Unknown MONAI model: {model_name!r}")

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/{model_name}/config.yaml")
    config.data.json_list = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.gpus           = [device.index if device.type == "cuda" else 0]
    config.cache_rate     = 0.0

    ckpt_path = CHECKPOINT_DIR / model_name / "weights" / f"f{fold}.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    lit = LitModel.load_from_checkpoint(
        str(ckpt_path),
        config=config,
        map_location=device,
        strict=False,
    )
    network = lit.model
    network.eval()
    network.to(device)
    return network


def load_model(model_name: str, fold: int, device: torch.device) -> torch.nn.Module:
    if model_name.lower() == "nnunet":
        return load_nnunet(fold, device)
    return load_mamba(model_name, fold, device)


# ===========================================================================
# nnUNet preprocessing helpers
# ===========================================================================

def load_plans() -> dict:
    return _load_plans_from_file(PLANS_FILE)


def load_splits() -> Dict:
    from picai_baseline.splits.picai import valid_splits
    return valid_splits


def load_splits_nnunet() -> Dict[int, Dict]:
    """Build per-fold val lists for nnunet:
    nnunet's own val cases + cases missing from nnunet splits entirely,
    distributed by picai_baseline fold assignment.
    """
    from picai_baseline.splits.picai import valid_splits

    nnunet_splits = json.load(open(NNUNET_ROOT / "splits.json"))

    all_in_nnunet: set = set()
    for s in nnunet_splits:
        all_in_nnunet.update(s["train"])
        all_in_nnunet.update(s["val"])

    baseline_all: set = set()
    for v in valid_splits.values():
        baseline_all.update(v["subject_list"])
    missing = baseline_all - all_in_nnunet

    result: Dict[int, Dict] = {}
    for fold_idx, (fold_key, fold_data) in enumerate(valid_splits.items()):
        extra = sorted(set(fold_data["subject_list"]) & missing)
        result[fold_idx] = {"subject_list": list(nnunet_splits[fold_idx]["val"]) + extra}
    return result


def _preprocess_nnunet(case_id: str, plans: dict):
    """
    Load co-registered NIfTI files and preprocess with nnUNet's GenericPreprocessor.
    Returns (data: float32 (3, D, H, W), properties: dict).
    """
    from nnUNet.nnunet.preprocessing.preprocessing import GenericPreprocessor  # noqa: E402

    input_files = []
    for ch in range(3):
        fpath = IMAGES_TR / f"{case_id}_{ch:04d}.nii.gz"
        if not fpath.exists():
            raise FileNotFoundError(f"NIfTI not found: {fpath}")
        input_files.append(str(fpath))

    target_spacing = plans["plans_per_stage"][0]["current_spacing"]
    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality=plans["normalization_schemes"],
        use_nonzero_mask=plans["use_mask_for_norm"],
        transpose_forward=plans["transpose_forward"],
        intensityproperties=plans["dataset_properties"]["intensityproperties"],
    )
    data, _seg, properties = preprocessor.preprocess_test_case(input_files, target_spacing)
    return data.astype(np.float32), properties


def _load_label_nnunet(case_id: str, plans: dict, prep_props: dict) -> Optional[np.ndarray]:
    """Load and resample the binary PCA label to match nnUNet preprocessed space."""
    import SimpleITK as sitk  # noqa: E402
    from nnUNet.nnunet.preprocessing.preprocessing import resample_data_or_seg  # noqa: E402

    label_path = LABELS_TR / f"{case_id}.nii.gz"
    if not label_path.exists():
        return None

    label_itk  = sitk.ReadImage(str(label_path))
    itk_spacing = np.array(label_itk.GetSpacing())[::-1]  # (z, y, x) = (D, H, W)
    label_np    = sitk.GetArrayFromImage(label_itk).astype(np.float32)

    tp          = plans["transpose_forward"]
    label_np    = label_np.transpose(tp)
    itk_spacing = itk_spacing[list(tp)]

    target_spacing = np.array(plans["plans_per_stage"][0]["current_spacing"])
    new_shape = np.round(
        itk_spacing / target_spacing * np.array(label_np.shape)
    ).astype(int)

    label_resampled = resample_data_or_seg(
        label_np[np.newaxis], new_shape, is_seg=True, axis=None, order=1, do_separate_z=False
    )[0]

    # Apply the same crop_to_nonzero bbox that nnUNet's preprocessor applies to the image.
    # Without this, label_resampled is in full (uncropped) space while data/zones are in
    # cropped space, causing all crop-coordinate slices to land in the wrong region.
    crop_bbox = prep_props.get("crop_bbox")
    if crop_bbox is not None:
        orig_size = prep_props["original_size_of_raw_data"]
        slices = []
        for j in range(label_resampled.ndim):
            pre_tp_axis = tp[j]
            start, end  = crop_bbox[pre_tp_axis]
            scale       = label_resampled.shape[j] / orig_size[pre_tp_axis]
            slices.append(slice(int(round(start * scale)), int(round(end * scale))))
        label_resampled = label_resampled[tuple(slices)]

    return (label_resampled > 0).astype(np.float32)


def _compute_nnunet_divisors(plans: dict) -> Tuple[int, int, int]:
    pool_kernels = plans["plans_per_stage"][0]["pool_op_kernel_sizes"]
    d_div = h_div = w_div = 1
    for k in pool_kernels:
        d_div *= k[0]; h_div *= k[1]; w_div *= k[2]
    return d_div, h_div, w_div


def _pad(
    x: torch.Tensor, d_div: int, h_div: int, w_div: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _B, _C, D, H, W = x.shape
    D_pad = (-D) % d_div; H_pad = (-H) % h_div; W_pad = (-W) % w_div
    if D_pad + H_pad + W_pad:
        x = F.pad(x, (0, W_pad, 0, H_pad, 0, D_pad))
    return x, (D_pad, H_pad, W_pad)


def _unpad(arr: np.ndarray, original_dhw: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = original_dhw
    return arr[:, :D, :H, :W]


# ===========================================================================
# Zone helpers  (standardised encoding: 0=bg, 1=PZ, 2=TZ)
# ===========================================================================

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
    pz = z[1]; tz = z[2]                # each (H, W, D)
    zones_hwD = (pz > 0.5).astype(np.int8) + 2 * (tz > 0.5).astype(np.int8)
    zones_Dhw = zones_hwD.transpose(2, 0, 1)  # (D, H, W)
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
        zone_mask = (z[1] + z[2]) > 0.5       # (H, W, D)
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


def _ensure_umamba_zones(fold: int, device: torch.device) -> None:
    """Forward-pass umamba to generate zone prediction files for any missing cases.

    Only does work when zone files are absent; skips cases that already have files.
    Called automatically before swin/nnunet processing when zone_source='umamba_pred'.
    """
    from shared_modules.data_module import DataModule   # noqa: E402
    from shared_modules.utils import load_config        # noqa: E402

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


# ===========================================================================
# Captum forward wrappers
# ===========================================================================

def _make_forward_func_sigmoid(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """For MONAI models — sigmoid on logit channel 1, sum over masked voxels → (B,).

    Strips MetaTensor from both fixed_mask (once, at construction) and model output
    (on each call) to prevent MetaTensor metadata accumulation across occlusion steps.
    """
    # Convert once so the closure holds a plain tensor, not a MetaTensor
    #if hasattr(fixed_mask, "as_tensor"):
        #fixed_mask = fixed_mask.as_tensor()

    def _forward(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)
        #if isinstance(out, (list, tuple)):
            #out = out[0]
        #if hasattr(out, "as_tensor"):
            #out = out.as_tensor()
        cancer_prob = torch.sigmoid(out[:, 1])
        return (cancer_prob * fixed_mask).flatten(1).sum(dim=1)

    def agg_segmentation_wrapper(inp):
        out = network(inp)[:, 0:2, ...]  # (B, 2, H, W, D)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "as_tensor"):
            out = out.as_tensor()
        aggregated_logits = (out[:, 1, ...] * fixed_mask).sum(dim=(1, 2, 3))  # (B,)
        return aggregated_logits
    return  agg_segmentation_wrapper # _forward


def _make_forward_func_softmax(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """For nnUNet — apply softmax over channels then take channel 1 as cancer probability."""
    def _forward(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        cancer_prob = torch.softmax(out, dim=1)[:, 1]
        return (cancer_prob * fixed_mask).flatten(1).sum(dim=1)
    return _forward


# ===========================================================================
# Sentinel helper
# ===========================================================================

def _sentinel(arr: Optional[np.ndarray]) -> np.ndarray:
    return arr if arr is not None else np.zeros((0,), dtype=np.float32)


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


PICAI_OVERLAP_THRESHOLD = 0.10  # PI-CAI: predicted lesion counts as TP only if
                                # intersection / GT_volume >= 10 %


def _detection_overlap(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Fraction of GT voxels covered by the prediction mask.

    Returns intersection / GT_volume, or 0.0 when GT is empty.
    Masks must be broadcastable (same shape or both (D,H,W)).
    """
    gt_volume = float(gt_mask.sum())
    if gt_volume == 0:
        return 0.0
    return float((pred_mask & gt_mask).sum()) / gt_volume


def _build_progress_record(
    predicted_pos: bool,
    lbl_crop: Optional[np.ndarray],
    zones_crop: Optional[np.ndarray],
    sal_np: Optional[np.ndarray],
    occ_np: Optional[np.ndarray] = None,
    abl_np: Optional[np.ndarray] = None,
    occ_tz_np: Optional[np.ndarray] = None,
    occ_pz_np: Optional[np.ndarray] = None,
    pred_crop: Optional[np.ndarray] = None,
    pred_cancer_voxels: int = 0,
    pred_max_prob: float = 0.0,
    confidence: float = 0.0,
) -> dict:
    """Build a per-case record for progress.json from already-computed arrays."""
    has_pca = (lbl_crop is not None) and bool(lbl_crop.sum() > 1)

    # PI-CAI TP criterion: prediction must overlap ≥10 % of the GT lesion volume.
    overlap = 0.0
    if predicted_pos and has_pca and pred_crop is not None:
        pred_mask_3d = pred_crop[0] > 0.5   # (D, H, W) bool
        gt_mask_3d   = lbl_crop[0] > 0      # (D, H, W) bool
        overlap      = _detection_overlap(pred_mask_3d, gt_mask_3d)

    if predicted_pos and has_pca and overlap >= PICAI_OVERLAP_THRESHOLD:
        classification = "tp"
    elif predicted_pos and not has_pca:
        classification = "fp"
    elif predicted_pos and has_pca and overlap < PICAI_OVERLAP_THRESHOLD:
        classification = "fp"  # predicted but missed the lesion
    elif not predicted_pos and not has_pca:
        classification = "tn"
    else:
        classification = "fn"

    pca_pz = pca_tz = 0
    if has_pca and zones_crop is not None and zones_crop.ndim == 3:
        lbl_2d = lbl_crop[0]
        pca_pz = int((lbl_2d * (zones_crop == 1)).sum())
        pca_tz = int((lbl_2d * (zones_crop == 2)).sum())

    pred_pz = pred_tz = 0
    if predicted_pos and pred_crop is not None and zones_crop is not None and zones_crop.ndim == 3:
        pred_mask = (pred_crop[0] > 0.5)
        pred_pz = int((pred_mask * (zones_crop == 1)).sum())
        pred_tz = int((pred_mask * (zones_crop == 2)).sum())

    zone_cat, primary_zone = _zone_category(pca_pz, pca_tz)

    sal_frac = None
    if sal_np is not None and sal_np.ndim == 4:
        sal_frac = _channel_stats(np.abs(sal_np))["ch_fraction"]

    occ_frac = None
    if occ_np is not None and occ_np.ndim == 4:
        occ_frac = _channel_stats(np.abs(occ_np))["ch_fraction"]

    # Per-strategy occlusion stats (fraction + mean)
    occ_zero_stats = None
    if occ_np is not None and occ_np.ndim == 4:
        s = _channel_stats(np.abs(occ_np))
        occ_zero_stats = {"ch_fraction": s["ch_fraction"], "ch_mean": s["ch_mean"]}

    occ_zm_stats = None
    if (occ_tz_np is not None and occ_tz_np.ndim == 4
            and occ_pz_np is not None and occ_pz_np.ndim == 4
            and zones_crop is not None and zones_crop.ndim == 3):
        _merged = np.zeros_like(occ_tz_np)
        _merged[:, zones_crop == 2] = occ_tz_np[:, zones_crop == 2]
        _merged[:, zones_crop == 1] = occ_pz_np[:, zones_crop == 1]
        s = _channel_stats(np.abs(_merged))
        occ_zm_stats = {"ch_fraction": s["ch_fraction"], "ch_mean": s["ch_mean"]}

    # Use first available strategy for the legacy occlusion_ch_fraction field
    occ_frac = (occ_zero_stats or occ_zm_stats or {}).get("ch_fraction", occ_frac)

    # AblationCAM produces a single-channel spatial map (1, D, H, W);
    # there is no per-input-channel breakdown, so we store None.
    abl_frac = None  # reserved for future per-channel ablation variants

    return {
        "done":                    True,
        "error":                   None,
        "predicted_pos":           predicted_pos,
        "pred_cancer_voxels":      pred_cancer_voxels,
        "pred_max_prob":           round(float(pred_max_prob), 4),
        "confidence":              round(float(confidence), 4),
        "has_pca":                 has_pca,
        "gt_cancer_voxels":        int(lbl_crop.sum()) if has_pca else 0,
        "classification":          classification,
        "primary_zone":            primary_zone,
        "zone_category":           zone_cat,
        "pz_voxels":               pca_pz if has_pca else None,
        "tz_voxels":               pca_tz if has_pca else None,
        "pred_pz_voxels":          pred_pz if predicted_pos else None,
        "pred_tz_voxels":          pred_tz if predicted_pos else None,
        "saliency_ch_fraction":        sal_frac,
        "occlusion_ch_fraction":       occ_frac,
        "ablation_ch_fraction":        abl_frac,
        "occlusion_zero_ch_fraction":  occ_zero_stats["ch_fraction"] if occ_zero_stats else None,
        "occlusion_zero_ch_mean":      occ_zero_stats["ch_mean"]     if occ_zero_stats else None,
        "occlusion_zm_ch_fraction":    occ_zm_stats["ch_fraction"]   if occ_zm_stats else None,
        "occlusion_zm_ch_mean":        occ_zm_stats["ch_mean"]       if occ_zm_stats else None,
    }


def _save_progress(progress: dict, progress_file: Path) -> None:
    """Atomically write progress dict to JSON (safe against mid-write kills)."""
    tmp = progress_file.with_name(".progress.json.tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, progress_file)


# ===========================================================================
# MONAI per-fold processing
# ===========================================================================

@profile
def process_fold_monai(
    fold: int,
    model_name: str,
    output_dir: Path,
    methods: set,
    skip_existing: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
    zones_only: bool = False,
) -> None:
    from shared_modules.data_module import DataModule   # noqa: E402
    from shared_modules.utils import load_config        # noqa: E402

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    progress_file = fold_dir / "progress.json"
    progress: dict = {}
    if progress_file.exists():
        with open(progress_file) as _f:
            progress = json.load(_f)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}  |  Fold {fold}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_model(model_name, fold, device)
    print(f"Model ready on {device}.")

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/{model_name}/config.yaml")
    config.data.json_list          = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.gpus                    = [device.index if device.type == "cuda" else 0]
    config.cache_rate              = 0.0
    config.transforms.label_keys   = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()
    print(f"Validation samples: {len(dl)}")

    run_saliency       = "saliency"        in methods
    run_occlusion      = "occlusion"       in methods
    run_ablation_cam   = "ablation"        in methods
    run_input_ablation = "input_ablation"  in methods

    processed, skipped, errors = 0, 0, 0
    objgraph.show_growth()  # reset baseline

    for i, batch in enumerate(dl):
        fname   = Path(batch["image"].meta["filename_or_obj"][0]).name
        case_id = fname.split("_0000")[0]
        out_file = fold_dir / f"{case_id}.npz"

        if skip_existing and progress.get(case_id, {}).get("done"):
            skipped += 1
            continue

        print(f"\n  [{i + 1}/{len(dl)}] {case_id}  shape={batch['image'].shape}")

        try:
            x = batch["image"].to(device)  # (1, 3, H, W, D)

            # ---- Forward pass (always) ------------------------------------
            with torch.no_grad():
                out = network(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                cancer_prob   = torch.sigmoid(out[:, 1])   # (1, H, W, D)
                fixed_mask    = (cancer_prob > 0.5)         # (1, H, W, D) bool
                cancer_voxels = int(fixed_mask.sum().item())
                pred_max_prob = float(cancer_prob.max().item())
                predicted_pos = cancer_voxels > 0

            print(f"    Predicted positive: {predicted_pos}  (cancer voxels={cancer_voxels}  max_prob={pred_max_prob:.3f})")

            # ---- umamba_mtl: compute predicted zones (all cases) ------------
            zones_pred_Dhw: Optional[np.ndarray] = None
            if model_name == "umamba_mtl":
                with torch.no_grad():
                    zone_logits   = out[:, 2:5]                                    # (1, 3, H, W, D)
                    zone_pred_hwD = zone_logits.softmax(dim=1).argmax(dim=1)[0]    # (H, W, D)
                zones_pred_Dhw = zone_pred_hwD.cpu().numpy().transpose(2, 0, 1).astype(np.int8)  # (D, H, W)
                del zone_logits, zone_pred_hwD

            # ---- Depth crop coordinates based on prostate zones -----------
            D = x.shape[4]
            UMAMBA_ROI_D = 20  # roi_size[2] for umamba_mtl

            if zone_source == "umamba_pred" and model_name == "umamba_mtl":
                # Use own just-computed zone predictions
                if zones_pred_Dhw is not None and (zones_pred_Dhw > 0).any():
                    d_idx = np.argwhere((zones_pred_Dhw > 0).any(axis=(1, 2)))[:, 0]
                    d0    = max(0, int(d_idx.min()) - 1)
                    d1    = min(D, int(d_idx.max()) + 2)
                else:
                    d0, d1 = 0, D

            elif zone_source == "umamba_pred" and model_name == "swin_unetr":
                # Load umamba zones and map to swin coordinate space.
                # Both models use spacing [0.5,0.5,3.0] and CenterSpatialCrop;
                # swin roi_d=32 vs umamba roi_d=20 → offset=(32-20)//2=6.
                SWIN_ROI_D = 32
                OFFSET     = (SWIN_ROI_D - UMAMBA_ROI_D) // 2  # = 6
                um_data    = _load_umamba_zones(case_id, fold)
                if um_data is not None and (um_data["zones"] > 0).any():
                    um_mask = (um_data["zones"] > 0).any(axis=(1, 2))  # (D_um,)
                    d_idx   = np.argwhere(um_mask)[:, 0]
                    d0_um   = max(0, int(d_idx.min()) - 1)
                    d1_um   = min(UMAMBA_ROI_D, int(d_idx.max()) + 2)
                    d0      = min(max(0, d0_um + OFFSET), D)
                    d1      = min(D, d1_um + OFFSET)
                else:
                    d0, d1 = _gt_depth_crop(batch, D)
            else:
                # GT behaviour — use zones from DataModule batch
                d0, d1 = _gt_depth_crop(batch, D)

            print(f"    Depth crop: d={d0}:{d1} ({d1 - d0} slices, zone_source={zone_source})")

            # ---- umamba_mtl: save zone prediction file (after d0/d1 known) --
            if model_name == "umamba_mtl" and zones_pred_Dhw is not None:
                zones_pred_dir = DEFAULT_OUTPUT_ZONES / f"fold_{fold}"
                zones_pred_dir.mkdir(parents=True, exist_ok=True)
                monai_affine  = batch["image"].affine.cpu().numpy().astype(np.float64)  # (4,4)
                zones_crop_wh = zones_pred_Dhw[d0:d1].transpose(0, 2, 1)               # (D_crop, W, H)
                np.savez_compressed(
                    zones_pred_dir / f"{case_id}.npz",
                    zones      = zones_pred_Dhw,
                    affine     = monai_affine,
                    d0         = np.int32(d0),
                    d1         = np.int32(d1),
                    zones_crop = zones_crop_wh,
                )
                del zones_crop_wh

            if zones_only:
                del zones_pred_Dhw, x, out, cancer_prob, fixed_mask, batch
                gc.collect()
                continue

            del zones_pred_Dhw

            # ---- Always: image, prediction, label, zones crop -------------
            image_np   = batch["image"][0].cpu().numpy()   # (3, H, W, D)
            image_np   = image_np.transpose(0, 3, 1, 2)    # (3, D, H, W)
            image_crop = image_np[:, d0:d1]                # (3, D_crop, H, W)

            pred_np   = cancer_prob[0].cpu().numpy()        # (H, W, D)
            pred_np   = pred_np.transpose(2, 0, 1)          # (D, H, W)
            pred_crop = pred_np[np.newaxis, d0:d1]          # (1, D_crop, H, W)

            if "pca" in batch:
                lbl_np   = batch["pca"][0, 0].cpu().numpy()  # (H, W, D)
                lbl_np   = lbl_np.transpose(2, 0, 1)          # (D, H, W)
                lbl_crop = lbl_np[np.newaxis, d0:d1]          # (1, D_crop, H, W)
            else:
                lbl_crop = None

            if zone_source == "umamba_pred" and model_name == "swin_unetr":
                # Embed umamba zones into swin depth space, then crop to [d0:d1]
                um_data = _load_umamba_zones(case_id, fold)
                if um_data is not None:
                    SWIN_ROI_D   = 32
                    UMAMBA_ROI_D = 20
                    OFFSET       = (SWIN_ROI_D - UMAMBA_ROI_D) // 2
                    zones_swin   = np.zeros((SWIN_ROI_D, 128, 128), dtype=np.int8)
                    zones_swin[OFFSET:OFFSET + UMAMBA_ROI_D] = um_data["zones"]
                    zones_crop = zones_swin[d0:d1]  # (D_crop, H, W)
                    del zones_swin
                else:
                    zones_crop = _zones_from_monai_batch(batch, d0, d1)
            else:
                zones_crop = _zones_from_monai_batch(batch, d0, d1)  # (D_crop, H, W) or None

            # ---- XAI: only when predicted positive -------------------------
            sal_np = occ_np = occ_tz_np = occ_pz_np = occ_ch_np = abl_np = inp_abl = None
            zone_median_baseline_np = None

            if predicted_pos:
                forward_func = _make_forward_func_sigmoid(network, fixed_mask.as_tensor())

                if run_saliency:
                    x_sal = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        sal_attr = Saliency(forward_func).attribute(x_sal, abs=True)
                    sal_np = sal_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                    del x_sal, sal_attr
                    sal_np = sal_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                    sal_np = sal_np[:, d0:d1]                       # (3, D_crop, H, W)

                if run_occlusion:
                    # x is (1, 3, H, W, D); occ_window/stride are specified as (C, D, H, W)
                    # → reorder to (C, H, W, D) to match monai tensor layout
                    occ_window_monai = (occ_window[0], occ_window[2], occ_window[3], occ_window[1])
                    occ_stride_monai = (occ_stride[0], occ_stride[2], occ_stride[3], occ_stride[1])

                    # --- zone_median baseline (TZ + PZ) ---
                    if occ_strategy in ("zone_median", "all") and zones_crop is not None:
                        cancer_mask_dhw = (pred_crop[0] > 0.5)          # (D_crop, H, W)
                        occ_win_dhw = (occ_window[1], occ_window[2], occ_window[3])
                        tz_patch, pz_patch = _compute_zone_baseline_patches(
                            image_crop, zones_crop, cancer_mask_dhw, occ_win_dhw, n_zone_patches
                        )
                        print(f"    [zone_median] TZ patch sum={tz_patch.sum():.3f}  PZ patch sum={pz_patch.sum():.3f}")
                        # Build combined baseline image (DHW) by tiling patches over zones_crop
                        _D, _H, _W = zones_crop.shape
                        _rD = -(-_D // tz_patch.shape[1])
                        _rH = -(-_H // tz_patch.shape[2])
                        _rW = -(-_W // tz_patch.shape[3])
                        _ttz = np.tile(tz_patch, (1, _rD, _rH, _rW))[:, :_D, :_H, :_W]
                        _tpz = np.tile(pz_patch, (1, _rD, _rH, _rW))[:, :_D, :_H, :_W]
                        zone_median_baseline_np = np.zeros((3, _D, _H, _W), dtype=np.float32)
                        zone_median_baseline_np[:, zones_crop == 2] = _ttz[:, zones_crop == 2]
                        zone_median_baseline_np[:, zones_crop == 1] = _tpz[:, zones_crop == 1]
                        del _ttz, _tpz
                        # Reconstruct full-D zone map for baseline (prostate region at [d0:d1])
                        H_full, W_full, D_full = x.shape[2], x.shape[3], x.shape[4]
                        zones_full_Dhw = np.zeros((D_full, H_full, W_full), dtype=np.int8)
                        zones_full_Dhw[d0:d1] = zones_crop
                        zero_patch = np.zeros_like(tz_patch)
                        occ_baseline_tz = _build_baseline_tensor(
                            zones_full_Dhw, tz_patch, zero_patch, x.shape, "hwd", device
                        )
                        occ_baseline_pz = _build_baseline_tensor(
                            zones_full_Dhw, zero_patch, pz_patch, x.shape, "hwd", device
                        )
                        del zones_full_Dhw, zero_patch

                        with torch.no_grad():
                            occ_attr_tz = Occlusion(forward_func).attribute(
                                x.as_tensor(),
                                sliding_window_shapes=occ_window_monai,
                                strides=occ_stride_monai,
                                baselines=occ_baseline_tz,
                                perturbations_per_eval=ppe,
                                show_progress=False,
                            )
                        occ_tz_np = occ_attr_tz.detach().cpu().numpy()[0]   # (3, H, W, D)
                        del occ_attr_tz, occ_baseline_tz
                        print("    [objgraph after occlusion TZ]"); objgraph.show_growth(limit=5)
                        occ_tz_np = occ_tz_np.transpose(0, 3, 1, 2)[:, d0:d1]  # (3, D_crop, H, W)

                        with torch.no_grad():
                            occ_attr_pz = Occlusion(forward_func).attribute(
                                x.as_tensor(),
                                sliding_window_shapes=occ_window_monai,
                                strides=occ_stride_monai,
                                baselines=occ_baseline_pz,
                                perturbations_per_eval=ppe,
                                show_progress=False,
                            )
                        occ_pz_np = occ_attr_pz.detach().cpu().numpy()[0]   # (3, H, W, D)
                        del occ_attr_pz, occ_baseline_pz
                        print("    [objgraph after occlusion PZ]"); objgraph.show_growth(limit=5)
                        occ_pz_np = occ_pz_np.transpose(0, 3, 1, 2)[:, d0:d1]  # (3, D_crop, H, W)
                    elif occ_strategy == "zone_median" and zones_crop is None:
                        print("    [zone_median] WARNING: zones_crop is None — skipping zone_median baseline.")

                    # --- zero baseline ---
                    if occ_strategy in ("zero", "all") or (occ_strategy == "zone_median" and zones_crop is None):
                        with torch.no_grad():
                            occ_attr = Occlusion(forward_func).attribute(
                                x.as_tensor(),
                                sliding_window_shapes=occ_window_monai,
                                strides=occ_stride_monai,
                                baselines=0.0,
                                perturbations_per_eval=ppe,
                                show_progress=False,
                            )
                        occ_np = occ_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                        del occ_attr
                        print("    [objgraph after occlusion zero]"); objgraph.show_growth(limit=5)
                        occ_np = occ_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                        occ_np = occ_np[:, d0:d1]                       # (3, D_crop, H, W)

                    # --- channel_baseline (T2W/ADC=1, HBV=0) ---
                    if occ_strategy in ("channel_baseline", "all"):
                        # MONAI layout: (1, C, H, W, D); channels 0-1 → 1.0, channel 2 → 0.0
                        occ_baseline_ch = torch.ones_like(x.as_tensor())
                        occ_baseline_ch[:, 2] = 0.0
                        with torch.no_grad():
                            occ_attr = Occlusion(forward_func).attribute(
                                x.as_tensor(),
                                sliding_window_shapes=occ_window_monai,
                                strides=occ_stride_monai,
                                baselines=occ_baseline_ch,
                                perturbations_per_eval=ppe,
                                show_progress=False,
                            )
                        occ_ch_np = occ_attr.detach().cpu().numpy()[0]  # (3, H, W, D)
                        del occ_attr, occ_baseline_ch
                        print("    [objgraph after occlusion ch_baseline]"); objgraph.show_growth(limit=5)
                        occ_ch_np = occ_ch_np.transpose(0, 3, 1, 2)     # (3, D, H, W)
                        occ_ch_np = occ_ch_np[:, d0:d1]                  # (3, D_crop, H, W)

                if run_ablation_cam:
                    try:
                        print("    Running 3D AblationCAM…")
                        target_layers = find_decoder_feature_layers(network, n_layers=1)
                        if not target_layers:
                            raise RuntimeError("No suitable Conv3d layers found in model.")
                        target_layer = target_layers[0]
                        print(f"    Target layer: {target_layer.__class__.__name__}"
                              f"(out_channels={target_layer.out_channels})")
                        _mask = fixed_mask[0].float()  # (H, W, D)

                        def _abl_target(output):
                            cancer = torch.sigmoid(output.unsqueeze(0)[:, 1])  # (1, H, W, D)
                            return (cancer * _mask).sum()

                        with AblationCAM3D(network, [target_layer], batch_size=16) as cam:
                            abl_maps = cam(x.detach().clone(), targets=[_abl_target])
                        # abl_maps: (1, H, W, D) — permute to (1, D, H, W) then crop
                        abl_maps_np = abl_maps.transpose(0, 3, 1, 2)  # (1, D, H, W)
                        abl_np = abl_maps_np[:, d0:d1]                 # (1, D_crop, H, W)
                        del cam
                    except Exception as exc:
                        print(f"    AblationCAM failed: {exc}")
                        traceback.print_exc()
                        abl_np = None

                if run_input_ablation:
                    print("    Running Input Ablation…")
                    forward_func_abl = _make_forward_func_sigmoid(network, fixed_mask)
                    with torch.no_grad():
                        orig_score = forward_func_abl(x).item()
                    weights = []
                    for ch in range(x.shape[1]):
                        x_abl   = x.clone()
                        flat    = x[:, ch].reshape(-1)
                        perm    = torch.randperm(flat.numel(), device=flat.device)
                        x_abl[:, ch] = flat[perm].reshape(x[:, ch].shape)
                        with torch.no_grad():
                            abl_score = forward_func_abl(x_abl).item()
                        w = (orig_score - abl_score) / orig_score if orig_score != 0 else 0.0
                        weights.append(w)
                        print(f"      ch {ch}: orig={orig_score:.4f} abl={abl_score:.4f} w={w:.4f}")
                    inp_abl = np.array(weights, dtype=np.float32)
                    del forward_func_abl, x_abl

            # ---- Save .npz only when model predicts positive ---------------
            # MONAI tensors are (C, H, W, D) / (H, W, D) with H=R, W=A after RAS orientation.
            # nnUNet tensors are (C, D, H, W) / (D, H, W) with H=AP, W=LR (from ITK z,y,x order).
            # After the transpose to DHW, MONAI gives (C, D, R, A) while nnUNet gives (C, D, AP, LR).
            # Swapping the last two spatial axes (H↔W) makes MONAI match nnUNet's axis layout.
            def _sw(a, ndim):
                """Swap the last two spatial axes to convert MONAI → nnUNet orientation."""
                if a is None:
                    return None
                return a.transpose(0, 1, 3, 2) if ndim == 4 else a.transpose(0, 2, 1)

            if predicted_pos:
                np.savez_compressed(
                    out_file,
                    saliency       = _sentinel(_sw(sal_np, 4)).astype(np.float32) if sal_np is not None else _sentinel(None),
                    occlusion              = _sentinel(_sw(occ_np, 4)).astype(np.float32) if occ_np is not None else _sentinel(None),
                    occlusion_tz           = _sentinel(_sw(occ_tz_np, 4)).astype(np.float32) if occ_tz_np is not None else _sentinel(None),
                    occlusion_pz           = _sentinel(_sw(occ_pz_np, 4)).astype(np.float32) if occ_pz_np is not None else _sentinel(None),
                    occlusion_ch_baseline  = _sentinel(_sw(occ_ch_np, 4)).astype(np.float32) if occ_ch_np is not None else _sentinel(None),
                    ablation       = _sentinel(_sw(abl_np, 4)).astype(np.float32) if abl_np is not None else _sentinel(None),
                    input_ablation = _sentinel(inp_abl),
                    image          = image_crop.transpose(0, 1, 3, 2).astype(np.float32),
                    prediction     = pred_crop.transpose(0, 1, 3, 2).astype(np.float32),
                    label          = _sentinel(lbl_crop.transpose(0, 1, 3, 2)).astype(np.float32) if lbl_crop is not None else _sentinel(None),
                    zones          = zones_crop.transpose(0, 2, 1).astype(np.int8) if zones_crop is not None else _sentinel(None),
                    zone_median_baseline = _sentinel(_sw(zone_median_baseline_np, 4)).astype(np.float32) if zone_median_baseline_np is not None else _sentinel(None),
                    channels           = np.array(CHANNEL_NAMES),
                    case_id            = case_id,
                    fold               = fold,
                    model              = model_name,
                    occlusion_strategy = np.array(occ_strategy),
                )
                print(f"    Saved: {out_file}")
                processed += 1
                del forward_func

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np,
                occ_np=occ_np, abl_np=abl_np,
                occ_tz_np=occ_tz_np, occ_pz_np=occ_pz_np,
                pred_crop=pred_crop,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=pred_max_prob,  # MONAI uses sigmoid — identical to pred_max_prob
            )
            _save_progress(progress, progress_file)

            # Free large tensors/arrays to prevent RAM accumulation across cases
            # forward_func must be deleted first — its closure holds fixed_mask
            del x, out, cancer_prob, fixed_mask, batch, occ_np, occ_tz_np, occ_pz_np, occ_ch_np
            # Also drop numpy intermediates — .numpy() shares storage with MetaTensor,
            # so these prevent the MetaTensor from being freed until GC runs
            image_np = image_crop = pred_np = pred_crop = None
            lbl_np = lbl_crop = zones_crop = None

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            progress[case_id] = {"done": False, "error": str(exc), "predicted_pos": None}
            _save_progress(progress, progress_file)

        log_large_vars(locals(), threshold_mb=50)
        gc.collect()
        gc.collect()  # second pass frees objects queued by first pass finalizers
        gc.collect()  # third pass to drain finalizer queue
        print(f"    Uncollectable objects after gc: {len(gc.garbage)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"    GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB  reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"    [objgraph after gc case {i}]"); objgraph.show_growth(limit=5)

    summary = {
        "model": model_name, "fold": fold,
        "total_val_cases": len(dl), "processed": processed,
        "skipped_existing": skipped, "errors": errors,
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFold {fold} done — processed={processed} skipped={skipped} errors={errors}")


# ===========================================================================
# nnUNet per-fold processing
# ===========================================================================

def process_fold_nnunet(
    fold: int,
    output_dir: Path,
    methods: set,
    skip_existing: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    occ_crop_hw: Optional[int] = 128,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
) -> None:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    progress_file = fold_dir / "progress.json"
    progress: dict = {}
    if progress_file.exists():
        with open(progress_file) as _f:
            progress = json.load(_f)

    print(f"\n{'=' * 60}")
    print(f"Model: nnunet  |  Fold {fold}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_nnunet(fold, device)
    print(f"Model ready on {device}.")

    plans     = load_plans()
    splits    = load_splits_nnunet()
    if fold not in splits:
        print(f"WARNING: fold {fold} not in valid_splits. Skipping.")
        return
    val_cases: List[str] = splits[fold]["subject_list"]
    print(f"Validation cases: {len(val_cases)}")

    d_div, h_div, w_div = _compute_nnunet_divisors(plans)
    run_saliency       = "saliency"       in methods
    run_occlusion      = "occlusion"      in methods
    run_ablation_cam   = "ablation"       in methods
    run_input_ablation = "input_ablation" in methods

    processed, skipped, errors = 0, 0, 0

    for i, case_id in enumerate(val_cases):
        out_file = fold_dir / f"{case_id}.npz"

        if skip_existing and progress.get(case_id, {}).get("done"):
            skipped += 1
            continue

        print(f"\n  [{i + 1}/{len(val_cases)}] {case_id}")

        try:
            # ---- Preprocess -----------------------------------------------
            data, prep_props = _preprocess_nnunet(case_id, plans)
            print(f"    Preprocessed shape: {data.shape}")
            original_dhw: Tuple[int, int, int] = tuple(data.shape[1:])  # type: ignore

            label_np = _load_label_nnunet(case_id, plans, prep_props)
            if label_np is None:
                print("    Label not found — prediction-only mode.")

            # ---- Pad to divisible sizes -----------------------------------
            x = torch.from_numpy(data[np.newaxis]).to(device)  # (1, 3, D, H, W)
            x, _padding = _pad(x, d_div, h_div, w_div)
            _B, _C, D_pad, H_pad, W_pad = x.shape

            # ---- Forward pass (always) ------------------------------------
            with torch.no_grad():
                out = network(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                cancer_prob   = torch.softmax(out, dim=1)[:, 1]  # (1, D_pad, H_pad, W_pad)
                fixed_mask    = (cancer_prob > 0.5)
                cancer_voxels = int(fixed_mask.sum().item())
                pred_max_prob = float(cancer_prob.max().item())
                confidence    = pred_max_prob  # softmax already applied above
                predicted_pos = cancer_voxels > 0

            print(f"    Predicted positive: {predicted_pos}  (cancer voxels={cancer_voxels}  max_prob={pred_max_prob:.3f})")

            # ---- Load zones (full) to determine crop coordinates ----------
            D_orig = original_dhw[0]
            if zone_source == "umamba_pred":
                zones_full = _zones_from_umamba_npz(
                    case_id, fold, plans, prep_props, 0, D_orig, 0, 0, None
                )
            else:
                zones_full = _zones_from_nnunet(
                    case_id, plans, prep_props, 0, D_orig, 0, 0, None
                )

            # ---- Crop coordinates based on prostate zones -----------------
            if zones_full is not None and zones_full.any():
                coords = np.argwhere(zones_full > 0)   # (N, 3): (d, h, w)
                d_min  = int(coords[:, 0].min())
                d_max  = int(coords[:, 0].max())
                d0     = max(0, d_min - 1)
                d1     = min(D_orig, d_max + 2)
                if occ_crop_hw is not None:
                    _, hc, wc = coords.mean(axis=0).astype(int)
                    h0 = int(np.clip(hc - occ_crop_hw // 2, 0, H_pad - occ_crop_hw))
                    w0 = int(np.clip(wc - occ_crop_hw // 2, 0, W_pad - occ_crop_hw))
                else:
                    h0, w0 = 0, 0
            else:
                d0, d1 = 0, D_orig
                h0, w0 = 0, 0

            print(f"    Depth crop: d={d0}:{d1}  h0={h0}  w0={w0}  zone_source={zone_source}")

            # ---- Always: image, prediction, label, zones ------------------
            prob_full  = _unpad(cancer_prob.cpu().numpy(), original_dhw)  # (1, D, H, W)
            if occ_crop_hw is not None:
                image_crop = data[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                pred_crop  = prob_full[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                if label_np is not None:
                    lbl_crop: Optional[np.ndarray] = label_np[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw][np.newaxis]
                else:
                    lbl_crop = None
                zones_crop = zones_full[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw] if zones_full is not None else None
            else:
                image_crop = data[:, d0:d1]
                pred_crop  = prob_full[:, d0:d1]
                lbl_crop   = label_np[d0:d1][np.newaxis] if label_np is not None else None
                zones_crop = zones_full[d0:d1] if zones_full is not None else None

            # ---- XAI: only when predicted positive ------------------------
            sal_np = occ_np = occ_tz_np = occ_pz_np = occ_ch_np = abl_np = inp_abl = None
            zone_median_baseline_np = None

            if predicted_pos:
                if occ_crop_hw is not None:
                    fixed_mask_crop = fixed_mask[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                    x_crop = x.detach().clone()[:, :, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                else:
                    fixed_mask_crop = fixed_mask
                    x_crop = x.detach().clone()

                fwd_sal = _make_forward_func_softmax(network, fixed_mask)
                fwd_occ = _make_forward_func_softmax(network, fixed_mask_crop)

                if run_saliency:
                    x_sal = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        sal_attr = Saliency(fwd_sal).attribute(x_sal, abs=True)
                    sal_np = _unpad(sal_attr.detach().cpu().numpy()[0], original_dhw)
                    del x_sal, sal_attr
                    if occ_crop_hw is not None:
                        sal_np = sal_np[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                    sal_np = sal_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                if run_occlusion:
                    # --- zone_median baseline (TZ + PZ) ---
                    if occ_strategy in ("zone_median", "all") and zones_full is not None:
                        # Build zones in x_crop coordinate system: (D_pad, H_crop, W_crop)
                        if occ_crop_hw is not None:
                            zones_xcrop = zones_full[:, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                        else:
                            zones_xcrop = zones_full  # (D_orig, H, W)
                        D_orig_z = zones_xcrop.shape[0]
                        D_pad_val = x_crop.shape[2]
                        if D_pad_val > D_orig_z:
                            zones_xcrop_padded = np.pad(
                                zones_xcrop, ((0, D_pad_val - D_orig_z), (0, 0), (0, 0)),
                                mode="constant", constant_values=0,
                            )
                        else:
                            zones_xcrop_padded = zones_xcrop[:D_pad_val]
                        # Compute medians from unpadded region only
                        image_xcrop_np = x_crop[0].cpu().numpy()       # (3, D_pad, H_crop, W_crop)
                        cancer_xcrop   = fixed_mask_crop[0].cpu().numpy()  # (D_pad, H_crop, W_crop)
                        occ_win_dhw = (occ_window[1], occ_window[2], occ_window[3])
                        tz_patch, pz_patch = _compute_zone_baseline_patches(
                            image_xcrop_np[:, :D_orig_z],
                            zones_xcrop_padded[:D_orig_z],
                            cancer_xcrop[:D_orig_z],
                            occ_win_dhw,
                            n_zone_patches,
                        )
                        print(f"    [zone_median] TZ patch sum={tz_patch.sum():.3f}  PZ patch sum={pz_patch.sum():.3f}")
                        # Build combined baseline image (DHW) by tiling patches over zones_crop
                        if zones_crop is not None:
                            _D, _H, _W = zones_crop.shape
                            _rD = -(-_D // tz_patch.shape[1])
                            _rH = -(-_H // tz_patch.shape[2])
                            _rW = -(-_W // tz_patch.shape[3])
                            _ttz = np.tile(tz_patch, (1, _rD, _rH, _rW))[:, :_D, :_H, :_W]
                            _tpz = np.tile(pz_patch, (1, _rD, _rH, _rW))[:, :_D, :_H, :_W]
                            zone_median_baseline_np = np.zeros((3, _D, _H, _W), dtype=np.float32)
                            zone_median_baseline_np[:, zones_crop == 2] = _ttz[:, zones_crop == 2]
                            zone_median_baseline_np[:, zones_crop == 1] = _tpz[:, zones_crop == 1]
                            del _ttz, _tpz
                        zero_patch = np.zeros_like(tz_patch)
                        occ_baseline_tz = _build_baseline_tensor(
                            zones_xcrop_padded, tz_patch, zero_patch, x_crop.shape, "dhw", device
                        )
                        occ_baseline_pz = _build_baseline_tensor(
                            zones_xcrop_padded, zero_patch, pz_patch, x_crop.shape, "dhw", device
                        )
                        del zones_xcrop, zones_xcrop_padded, image_xcrop_np, cancer_xcrop, zero_patch

                        with torch.no_grad():
                            occ_attr_tz = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_tz,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_tz_np = _unpad(occ_attr_tz.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr_tz, occ_baseline_tz
                        occ_tz_np = occ_tz_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                        with torch.no_grad():
                            occ_attr_pz = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_pz,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_pz_np = _unpad(occ_attr_pz.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr_pz, occ_baseline_pz
                        occ_pz_np = occ_pz_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)
                    elif occ_strategy == "zone_median" and zones_full is None:
                        print("    [zone_median] WARNING: zones_full is None — skipping zone_median baseline.")

                    # --- zero baseline ---
                    if occ_strategy in ("zero", "all") or (occ_strategy == "zone_median" and zones_full is None):
                        with torch.no_grad():
                            occ_attr = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=0.0,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_np = _unpad(occ_attr.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr
                        occ_np = occ_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                    # --- channel_baseline (T2W/ADC=1, HBV=0) ---
                    if occ_strategy in ("channel_baseline", "all"):
                        # nnUNet layout: (1, C, D, H, W); channels 0-1 → 1.0, channel 2 → 0.0
                        occ_baseline_ch = torch.ones_like(x_crop)
                        occ_baseline_ch[:, 2] = 0.0
                        with torch.no_grad():
                            occ_attr = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_ch,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_ch_np = _unpad(occ_attr.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr, occ_baseline_ch
                        occ_ch_np = occ_ch_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                if run_ablation_cam:
                    try:
                        print("    Running 3D AblationCAM…")
                        target_layer = network.conv_blocks_context[6][1].blocks[0].conv
                        print(f"    Target layer: {target_layer.__class__.__name__}"
                              f"(out_channels={target_layer.out_channels})")
                        cam = AblationCAM3D(network, [target_layer], batch_size=16)
                        _mask = fixed_mask_crop[0].float()  # (D_pad, H_crop, W_crop)

                        def _abl_target(output):
                            if isinstance(output, (list, tuple)):
                                output = output[0]
                            cancer_prob = output[1]  # already softmaxed by network
                            return (cancer_prob * _mask).sum()

                        abl_maps = cam(x_crop, targets=[_abl_target])  # (1, D_pad, H_crop, W_crop)
                        abl_np = _unpad(abl_maps, (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        abl_np = abl_np[:, d0:d1]  # (1, D_crop, H_crop, W_crop)
                    except Exception as exc:
                        print(f"    AblationCAM failed: {exc}")
                        traceback.print_exc()
                        abl_np = None

                if run_input_ablation:
                    print("    Running Input Ablation…")
                    with torch.no_grad():
                        orig_score = fwd_occ(x_crop).item()
                    weights = []
                    for ch in range(x_crop.shape[1]):
                        x_abl   = x_crop.clone()
                        flat    = x_crop[:, ch].reshape(-1)
                        perm    = torch.randperm(flat.numel(), device=flat.device)
                        x_abl[:, ch] = flat[perm].reshape(x_crop[:, ch].shape)
                        with torch.no_grad():
                            abl_score = fwd_occ(x_abl).item()
                        w = (orig_score - abl_score) / orig_score if orig_score != 0 else 0.0
                        weights.append(w)
                        print(f"      ch {ch}: orig={orig_score:.4f} abl={abl_score:.4f} w={w:.4f}")
                    inp_abl = np.array(weights, dtype=np.float32)

            # ---- Save .npz only when model predicts positive ---------------
            if predicted_pos:
                np.savez_compressed(
                    out_file,
                    saliency       = _sentinel(sal_np).astype(np.float32) if sal_np is not None else _sentinel(None),
                    occlusion              = _sentinel(occ_np).astype(np.float32) if occ_np is not None else _sentinel(None),
                    occlusion_tz           = _sentinel(occ_tz_np).astype(np.float32) if occ_tz_np is not None else _sentinel(None),
                    occlusion_pz           = _sentinel(occ_pz_np).astype(np.float32) if occ_pz_np is not None else _sentinel(None),
                    occlusion_ch_baseline  = _sentinel(occ_ch_np).astype(np.float32) if occ_ch_np is not None else _sentinel(None),
                    ablation       = _sentinel(abl_np).astype(np.float32) if abl_np is not None else _sentinel(None),
                    input_ablation = _sentinel(inp_abl),
                    image          = image_crop.astype(np.float32),
                    prediction     = pred_crop.astype(np.float32),
                    label          = lbl_crop.astype(np.float32) if lbl_crop is not None else _sentinel(None),
                    zones          = zones_crop.astype(np.int8) if zones_crop is not None else _sentinel(None),
                    zone_median_baseline = _sentinel(zone_median_baseline_np).astype(np.float32) if zone_median_baseline_np is not None else _sentinel(None),
                    channels           = np.array(CHANNEL_NAMES),
                    case_id            = case_id,
                    fold               = fold,
                    model              = "nnunet",
                    occlusion_strategy = np.array(occ_strategy),
                )
                print(f"    Saved: {out_file}")
                processed += 1

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np,
                occ_np=occ_np, abl_np=abl_np,
                occ_tz_np=occ_tz_np, occ_pz_np=occ_pz_np,
                pred_crop=pred_crop,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=confidence,
            )
            _save_progress(progress, progress_file)

            # Free large tensors to prevent RAM accumulation across cases
            del x, out, cancer_prob, fixed_mask

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            progress[case_id] = {"done": False, "error": str(exc), "predicted_pos": None}
            _save_progress(progress, progress_file)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "model": "nnunet", "fold": fold,
        "total_val_cases": len(val_cases), "processed": processed,
        "skipped_existing": skipped, "errors": errors,
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFold {fold} done — processed={processed} skipped={skipped} errors={errors}")


# ===========================================================================
# Dispatcher
# ===========================================================================

def process_fold(
    fold: int,
    model_name: str,
    output_dir: Path,
    methods: set,
    skip_existing: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    occ_crop_hw: Optional[int] = 128,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
) -> None:
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "nnunet":
        if zone_source == "umamba_pred":
            _ensure_umamba_zones(fold, device)
        process_fold_nnunet(
            fold, model_output_dir, methods, skip_existing, occ_window, occ_stride, ppe,
            occ_crop_hw, device, occ_strategy, n_zone_patches, zone_source,
        )
    else:
        if model_name == "swin_unetr" and zone_source == "umamba_pred":
            _ensure_umamba_zones(fold, device)
        process_fold_monai(
            fold, model_name, model_output_dir, methods, skip_existing, occ_window, occ_stride, ppe,
            device, occ_strategy, n_zone_patches, zone_source,
        )


# ===========================================================================
# Metrics computation
# ===========================================================================

def _is_empty(arr: np.ndarray) -> bool:
    return arr.ndim == 1 and arr.shape[0] == 0


def _channel_stats(arr: np.ndarray) -> dict:
    """Compute per-channel attribution statistics for a (3, D, H, W) array."""
    ch_sum  = arr.sum(axis=(1, 2, 3))
    ch_mean = arr.mean(axis=(1, 2, 3))
    ch_max  = arr.max(axis=(1, 2, 3))
    ch_std  = arr.std(axis=(1, 2, 3))
    total   = ch_sum.sum()
    ch_frac = (ch_sum / total).tolist() if total > 0 else [0.0] * len(ch_sum)
    return {
        "ch_sum":      ch_sum.tolist(),
        "ch_mean":     ch_mean.tolist(),
        "ch_max":      ch_max.tolist(),
        "ch_std":      ch_std.tolist(),
        "ch_fraction": ch_frac,
        "dominant_ch": int(np.argmax(ch_mean)),
    }


def _zone_category(pca_in_pz: int, pca_in_tz: int) -> Tuple[Optional[str], Optional[str]]:
    """Return (zone_category, primary_zone) from voxel counts."""
    if pca_in_pz > 0 and pca_in_tz == 0:
        return "pz_only", "pz"
    if pca_in_tz > 0 and pca_in_pz == 0:
        return "tz_only", "tz"
    if pca_in_pz > 0 and pca_in_tz > 0:
        cat = "both_pz" if pca_in_pz >= pca_in_tz else "both_tz"
        return cat, "pz" if pca_in_pz >= pca_in_tz else "tz"
    return "unknown", None


def compute_metrics(xai_dir: Path, model_name: str, metrics_dir: Path) -> List[dict]:
    """
    Read all .npz files in xai_dir/fold_*/, compute per-case metrics from the
    saved label, prediction, and zones arrays, and write sample_data.json.
    Returns the list of per-sample records.
    """
    model_metrics_dir = metrics_dir / model_name
    model_metrics_dir.mkdir(parents=True, exist_ok=True)
    cache_path = model_metrics_dir / "sample_data.json"

    npz_files = sorted((xai_dir / model_name).glob("fold_*/*.npz"))
    print(f"\nComputing metrics for {model_name}: {len(npz_files)} npz files")

    records = []
    for npz_path in npz_files:
        try:
            d     = np.load(npz_path, allow_pickle=True)
            label = d["label"]       # (1, D, H, W) or (0,)
            pred  = d["prediction"]  # (1, D, H, W)
            zones = d["zones"]       # (D, H, W) int8 or (0,)

            case_id = str(d["case_id"])
            fold    = int(d["fold"])

            has_pca          = (not _is_empty(label)) and bool(label.sum() > 1)
            predicted_pos    = (not _is_empty(pred)) and bool(pred.max() > 0.5)

            # PI-CAI TP criterion: prediction must overlap ≥10 % of the GT lesion volume.
            overlap = 0.0
            if predicted_pos and has_pca:
                pred_mask_3d = pred[0] > 0.5   # (D, H, W) bool
                gt_mask_3d   = label[0] > 0    # (D, H, W) bool
                overlap      = _detection_overlap(pred_mask_3d, gt_mask_3d)

            if predicted_pos and has_pca and overlap >= PICAI_OVERLAP_THRESHOLD:
                classification = "tp"
            elif predicted_pos and not has_pca:
                classification = "fp"
            elif predicted_pos and has_pca and overlap < PICAI_OVERLAP_THRESHOLD:
                classification = "fp"  # predicted but missed the lesion
            elif not predicted_pos and not has_pca:
                classification = "tn"
            else:
                classification = "fn"

            # Zone stats from saved zones + label
            if not _is_empty(label) and not _is_empty(zones) and zones.ndim == 3:
                lbl_2d   = label[0]          # (D, H, W)
                pca_pz   = int((lbl_2d * (zones == 1)).sum())
                pca_tz   = int((lbl_2d * (zones == 2)).sum())
            else:
                pca_pz = pca_tz = 0

            pred_pz = pred_tz = 0
            if predicted_pos and not _is_empty(zones) and zones.ndim == 3:
                pred_mask = (pred[0] > 0.5)  # (D, H, W)
                pred_pz   = int((pred_mask * (zones == 1)).sum())
                pred_tz   = int((pred_mask * (zones == 2)).sum())

            zone_cat, primary_zone = _zone_category(pca_pz, pca_tz)

            record: dict = {
                "case_id":           case_id,
                "fold":              fold,
                "model":             model_name,
                "classification":    classification,
                "has_pca":           has_pca,
                "predicted_positive": predicted_pos,
                "primary_zone":      primary_zone,
                "zone_category":     zone_cat,
                "pz_voxels":         pca_pz if has_pca else None,
                "tz_voxels":         pca_tz if has_pca else None,
                "pred_pz_voxels":    pred_pz if predicted_pos else None,
                "pred_tz_voxels":    pred_tz if predicted_pos else None,
            }

            # Channel attribution stats (only when XAI maps were computed)
            for method in METHODS:
                arr = d.get(method, np.zeros((0,), dtype=np.float32))
                if not _is_empty(arr) and arr.ndim == 4:
                    record[method] = _channel_stats(np.abs(arr))
                else:
                    record[method] = None

            records.append(record)

        except Exception as exc:
            warnings.warn(f"Failed to load {npz_path}: {exc}")

    print(f"  Loaded {len(records)} samples.")
    with open(cache_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"  Saved: {cache_path}")
    return records


# ===========================================================================
# Chart generation
# ===========================================================================

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


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified XAI pipeline for all PI-CAI models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["umamba_mtl", "swin_unetr", "nnunet", "all"],
        metavar="MODEL",
        help="Model(s) to process. Use 'all' for all three.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0,1,2,3,4",
        metavar="N,N,...",
        help="Comma-separated fold indices to process.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        choices=["saliency", "occlusion", "ablation", "input_ablation", "all"],
        metavar="METHOD",
        help="XAI methods to run. Use 'all' for all methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_XAI),
        help="Root directory for .npz output files.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_METRICS),
        help="Root directory for metrics JSON and chart outputs.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Recompute XAI even if .npz already exists.",
    )
    parser.add_argument(
        "--compute-metrics-only",
        action="store_true",
        help="Skip XAI generation; only recompute metrics/charts from existing .npz files.",
    )
    parser.add_argument(
        "--occlusion-window",
        type=str,
        default="1,2,12,12",
        metavar="C,D,H,W",
        help="Occlusion sliding window shape (channel, depth, height, width).",
    )
    parser.add_argument(
        "--occlusion-stride",
        type=str,
        default="1,1,6,6",
        metavar="C,D,H,W",
        help="Occlusion sliding window stride.",
    )
    parser.add_argument(
        "--perturbations-per-eval",
        type=int,
        default=32,
        help="Number of occlusion perturbations per forward pass.",
    )
    parser.add_argument(
        "--occ-crop-hw",
        type=int,
        default=128,
        metavar="N",
        help=(
            "nnUNet only: crop H/W to N pixels centered on tumor before occlusion. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--occlusion-strategy",
        type=str,
        default="all",
        choices=["zero", "zone_median", "channel_baseline", "all"],
        help=(
            "Baseline strategy for occlusion: 'zero' replaces occluded windows with 0 "
            "(current default); 'zone_median' uses per-channel median values sampled from "
            "non-cancerous patches within each prostate zone (TZ/PZ); "
            "'channel_baseline' uses baseline=1 for T2W and ADC (channels 0-1) and "
            "baseline=0 for HBV (channel 2); 'all' runs every strategy and saves each "
            "result under its own npz key."
        ),
    )
    parser.add_argument(
        "--occlusion-zone-patches",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of random patches to sample per zone when --occlusion-strategy=zone_median. "
            "Each patch has the same size as the occlusion window."
        ),
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="CUDA device index to use (e.g. 0, 1, 2). Defaults to cuda:0 if CUDA is available.",
    )
    parser.add_argument(
        "--zone-source",
        type=str,
        default="umamba_pred",
        choices=["umamba_pred", "gt"],
        help=(
            "Source for prostate zone maps used to determine depth crop. "
            "'umamba_pred' (default): use U-MambaMTL predicted zones from "
            "results/xai/zones/, ensuring consistent depth crop across all models. "
            "Auto-generates zone files for swin/nnunet folds if umamba hasn't run yet. "
            "'gt': use per-model ground-truth zone NIfTI files (original behaviour)."
        ),
    )

    args = parser.parse_args()

    # Resolve models
    models: List[str] = (
        ["umamba_mtl", "swin_unetr", "nnunet"]
        if "all" in args.models
        else args.models
    )

    # Resolve folds
    folds = [int(f.strip()) for f in args.fold.split(",")]

    # Resolve methods
    methods = set(args.methods)
    if "all" in methods:
        methods = {"saliency", "occlusion", "ablation", "input_ablation"}

    # Parse occlusion params
    occ_window: Tuple[int, int, int, int] = tuple(  # type: ignore[assignment]
        int(v) for v in args.occlusion_window.split(",")
    )
    occ_stride: Tuple[int, int, int, int] = tuple(  # type: ignore[assignment]
        int(v) for v in args.occlusion_stride.split(",")
    )
    occ_crop_hw: Optional[int] = args.occ_crop_hw if args.occ_crop_hw > 0 else None

    if args.device is not None:
        device: Optional[torch.device] = torch.device(f"cuda:{args.device}")
    else:
        device = None  # each process_fold will pick cuda:0 or cpu

    output_dir  = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)

    print(f"Models:            {models}")
    print(f"Folds:             {folds}")
    print(f"Methods:           {', '.join(sorted(methods))}")
    print(f"Output XAI dir:    {output_dir}")
    print(f"Output metrics:    {metrics_dir}")
    print(f"Occlusion window:  {occ_window}  stride: {occ_stride}")
    print(f"Occlusion strategy:{args.occlusion_strategy}  zone patches: {args.occlusion_zone_patches}")
    print(f"Zone source:       {args.zone_source}")
    print(f"Device:            {device if device is not None else 'auto'}")

    for model_name in models:
        if not args.compute_metrics_only:
            for fold in folds:
                process_fold(
                    fold=fold,
                    model_name=model_name,
                    output_dir=output_dir,
                    methods=methods,
                    skip_existing=not args.no_skip,
                    occ_window=occ_window,
                    occ_stride=occ_stride,
                    ppe=args.perturbations_per_eval,
                    occ_crop_hw=occ_crop_hw,
                    device=device,
                    occ_strategy=args.occlusion_strategy,
                    n_zone_patches=args.occlusion_zone_patches,
                    zone_source=args.zone_source,
                )

        # Compute metrics and charts from saved .npz files
        records = compute_metrics(output_dir, model_name, metrics_dir)
        if records:
            generate_charts(records, model_name, metrics_dir)
        else:
            print(f"  No records found for {model_name} — skipping charts.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
