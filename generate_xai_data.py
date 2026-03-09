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
from captum.attr import Occlusion, Saliency
from src.ablation_cam_3d import AblationCAM3D, find_decoder_feature_layers

from src.utils import load_plans as _load_plans_from_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Outputs
DEFAULT_OUTPUT_XAI     = PROJECT_ROOT / "results" / "xai"
DEFAULT_OUTPUT_METRICS = PROJECT_ROOT / "results" / "metrics"

CHANNEL_NAMES = ["t2w", "adc", "hbv"]

# nnUNet paths
NNUNET_ROOT         = PROJECT_ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_ROOT / "results" / "nnUNet"
TASK_NAME           = "Task2203_picai_baseline"
DATASET_DIR         = NNUNET_PREPROCESSED / TASK_NAME
PLANS_FILE          = NNUNET_RESULTS / "plans.pkl"
SPLITS_FILE         = NNUNET_ROOT / "splits.json"

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
CLASS_FILTERS = ["tp", "fn", "both"]
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
    "fn":   "False Negatives",
    "both": "TP + FN",
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

    config = load_config(f"experiments/picai/{model_name}/config.yaml")
    config.data.json_list = f"json_datalists/picai/fold_{fold}.json"
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


def load_splits() -> List[Dict]:
    with open(SPLITS_FILE) as f:
        return json.load(f)


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


def _load_label_nnunet(case_id: str, plans: dict) -> Optional[np.ndarray]:
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


# ===========================================================================
# Captum forward wrappers
# ===========================================================================

def _make_forward_func_sigmoid(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """For MONAI models — sigmoid on logit channel 1, sum over masked voxels → (B,)."""
    def _forward(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        cancer_prob = torch.sigmoid(out[:, 1])
        return (cancer_prob * fixed_mask).flatten(1).sum(dim=1)
    return _forward


def _make_forward_func_softmax(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """For nnUNet — softmax on channel 1, sum over masked voxels → (B,)."""
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


def _build_progress_record(
    predicted_pos: bool,
    lbl_crop: Optional[np.ndarray],
    zones_crop: Optional[np.ndarray],
    sal_np: Optional[np.ndarray],
    occ_np: Optional[np.ndarray],
    abl_np: Optional[np.ndarray] = None,
    pred_cancer_voxels: int = 0,
    pred_max_prob: float = 0.0,
    confidence: float = 0.0,
) -> dict:
    """Build a per-case record for progress.json from already-computed arrays."""
    has_pca = (lbl_crop is not None) and bool(lbl_crop.sum() > 1)

    if predicted_pos and has_pca:
        classification = "tp"
    elif predicted_pos and not has_pca:
        classification = "fp"
    elif not predicted_pos and not has_pca:
        classification = "tn"
    else:
        classification = "fn"

    pca_pz = pca_tz = 0
    if has_pca and zones_crop is not None and zones_crop.ndim == 3:
        lbl_2d = lbl_crop[0]
        pca_pz = int((lbl_2d * (zones_crop == 1)).sum())
        pca_tz = int((lbl_2d * (zones_crop == 2)).sum())

    zone_cat, primary_zone = _zone_category(pca_pz, pca_tz)

    sal_frac = None
    if sal_np is not None and sal_np.ndim == 4:
        sal_frac = _channel_stats(np.abs(sal_np))["ch_fraction"]

    occ_frac = None
    if occ_np is not None and occ_np.ndim == 4:
        occ_frac = _channel_stats(np.abs(occ_np))["ch_fraction"]

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
        "saliency_ch_fraction":    sal_frac,
        "occlusion_ch_fraction":   occ_frac,
        "ablation_ch_fraction":    abl_frac,
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

def process_fold_monai(
    fold: int,
    model_name: str,
    output_dir: Path,
    methods: set,
    skip_existing: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_model(model_name, fold, device)
    print(f"Model ready on {device}.")

    config = load_config(f"experiments/picai/{model_name}/config.yaml")
    config.data.json_list          = f"json_datalists/picai/fold_{fold}.json"
    config.gpus                    = [device.index if device.type == "cuda" else 0]
    config.cache_rate              = 0.0
    config.transforms.label_keys  = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()
    print(f"Validation samples: {len(dl)}")

    run_saliency       = "saliency"        in methods
    run_occlusion      = "occlusion"       in methods
    run_ablation_cam   = "ablation"        in methods
    run_input_ablation = "input_ablation"  in methods

    processed, skipped, errors = 0, 0, 0

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

            # ---- Depth crop coordinates (D is axis 3 in MONAI layout) ----
            D = x.shape[4]
            if predicted_pos:
                mask_np = fixed_mask[0].cpu().numpy()  # (H, W, D)
                coords  = np.argwhere(mask_np)          # (N, 3): (h, w, d)
                d_min   = int(coords[:, 2].min())
                d_max   = int(coords[:, 2].max())
                d0      = max(0, d_min - 1)
                d1      = min(D, d_max + 2)
            else:
                d0, d1 = 0, D

            print(f"    Depth crop: d={d0}:{d1} ({d1 - d0} slices)")

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

            zones_crop = _zones_from_monai_batch(batch, d0, d1)  # (D_crop, H, W) or None

            # ---- XAI: only when predicted positive -------------------------
            sal_np = occ_np = abl_np = inp_abl = None

            if predicted_pos:
                forward_func = _make_forward_func_sigmoid(network, fixed_mask)

                if run_saliency:
                    x_sal = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        sal_attr = Saliency(forward_func).attribute(x_sal, abs=True)
                    sal_np = sal_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                    sal_np = sal_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                    sal_np = sal_np[:, d0:d1]                       # (3, D_crop, H, W)

                if run_occlusion:
                    with torch.enable_grad():
                        occ_attr = Occlusion(forward_func).attribute(
                            x.detach().clone(),
                            sliding_window_shapes=occ_window,
                            strides=occ_stride,
                            baselines=0.0,
                            perturbations_per_eval=ppe,
                            show_progress=True,
                        )
                    occ_np = occ_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                    occ_np = occ_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                    occ_np = occ_np[:, d0:d1]                       # (3, D_crop, H, W)

                if run_ablation_cam:
                    try:
                        print("    Running 3D AblationCAM…")
                        target_layers = find_decoder_feature_layers(network, n_layers=1)
                        if not target_layers:
                            raise RuntimeError("No suitable Conv3d layers found in model.")
                        target_layer = target_layers[0]
                        print(f"    Target layer: {target_layer.__class__.__name__}"
                              f"(out_channels={target_layer.out_channels})")
                        cam = AblationCAM3D(network, [target_layer], batch_size=16)
                        _mask = fixed_mask[0].float()  # (H, W, D)

                        def _abl_target(output):
                            cancer = torch.sigmoid(output.unsqueeze(0)[:, 1])  # (1, H, W, D)
                            return (cancer * _mask).sum()

                        abl_maps = cam(x.detach().clone(), targets=[_abl_target])
                        # abl_maps: (1, H, W, D) — permute to (1, D, H, W) then crop
                        abl_maps_np = abl_maps.transpose(0, 3, 1, 2)  # (1, D, H, W)
                        abl_np = abl_maps_np[:, d0:d1]                 # (1, D_crop, H, W)
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

            # ---- Save .npz only when model predicts positive ---------------
            if predicted_pos:
                np.savez_compressed(
                    out_file,
                    saliency       = _sentinel(sal_np).astype(np.float32) if sal_np is not None else _sentinel(None),
                    occlusion      = _sentinel(occ_np).astype(np.float32) if occ_np is not None else _sentinel(None),
                    ablation       = _sentinel(abl_np).astype(np.float32) if abl_np is not None else _sentinel(None),
                    input_ablation = _sentinel(inp_abl),
                    image          = image_crop.astype(np.float32),
                    prediction     = pred_crop.astype(np.float32),
                    label          = _sentinel(lbl_crop).astype(np.float32) if lbl_crop is not None else _sentinel(None),
                    zones          = zones_crop.astype(np.int8) if zones_crop is not None else _sentinel(None),
                    channels       = np.array(CHANNEL_NAMES),
                    case_id        = case_id,
                    fold           = fold,
                    model          = model_name,
                )
                print(f"    Saved: {out_file}")
                processed += 1

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np, occ_np, abl_np,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=pred_max_prob,  # MONAI uses sigmoid — identical to pred_max_prob
            )
            _save_progress(progress, progress_file)

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            progress[case_id] = {"done": False, "error": str(exc), "predicted_pos": None}
            _save_progress(progress, progress_file)

        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

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

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_nnunet(fold, device)
    print(f"Model ready on {device}.")

    plans     = load_plans()
    splits    = load_splits()
    if fold >= len(splits):
        print(f"WARNING: fold {fold} not in splits.json. Skipping.")
        return
    val_cases: List[str] = splits[fold]["val"]
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

            label_np = _load_label_nnunet(case_id, plans)
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
                confidence    = float(torch.sigmoid(out[:, 1]).max().item())
                predicted_pos = cancer_voxels > 0

            print(f"    Predicted positive: {predicted_pos}  (cancer voxels={cancer_voxels}  max_prob={pred_max_prob:.3f})")

            # ---- Crop coordinates -----------------------------------------
            D_orig = original_dhw[0]
            if predicted_pos:
                mask_np = fixed_mask[0].cpu().numpy()  # (D_pad, H_pad, W_pad)
                coords  = np.argwhere(mask_np)          # (N, 3): (d, h, w)
                d_min   = int(coords[:, 0].min())
                d_max   = int(coords[:, 0].max())
                d0      = max(0, d_min - 1)
                d1      = min(D_orig, d_max + 2)

                if occ_crop_hw is not None:
                    _, hc, wc = coords.mean(axis=0).astype(int)
                    h0 = int(np.clip(hc - occ_crop_hw // 2, 0, H_pad - occ_crop_hw))
                    w0 = int(np.clip(wc - occ_crop_hw // 2, 0, W_pad - occ_crop_hw))
                else:
                    h0, w0 = 0, 0
            else:
                d0, d1 = 0, D_orig
                h0, w0 = 0, 0

            print(f"    Depth crop: d={d0}:{d1}  h0={h0}  w0={w0}")

            # ---- Always: image, prediction, label, zones ------------------
            prob_full  = _unpad(cancer_prob.cpu().numpy(), original_dhw)  # (1, D, H, W)
            if occ_crop_hw is not None and predicted_pos:
                image_crop = data[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                pred_crop  = prob_full[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                if label_np is not None:
                    lbl_crop: Optional[np.ndarray] = label_np[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw][np.newaxis]
                else:
                    lbl_crop = None
            else:
                image_crop = data[:, d0:d1]
                pred_crop  = prob_full[:, d0:d1]
                lbl_crop   = label_np[d0:d1][np.newaxis] if label_np is not None else None

            zones_crop = _zones_from_nnunet(
                case_id, plans, prep_props, d0, d1, h0, w0,
                occ_crop_hw if predicted_pos else None,
            )

            # ---- XAI: only when predicted positive ------------------------
            sal_np = occ_np = abl_np = inp_abl = None

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
                    if occ_crop_hw is not None:
                        sal_np = sal_np[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                    sal_np = sal_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                if run_occlusion:
                    with torch.enable_grad():
                        occ_attr = Occlusion(fwd_occ).attribute(
                            x_crop,
                            sliding_window_shapes=occ_window,
                            strides=occ_stride,
                            baselines=0.0,
                            perturbations_per_eval=ppe,
                            show_progress=True,
                        )
                    occ_np = _unpad(occ_attr.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                    occ_np = occ_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

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
                            cancer_prob = torch.softmax(output.unsqueeze(0), dim=1)[0, 1]
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
                    occlusion      = _sentinel(occ_np).astype(np.float32) if occ_np is not None else _sentinel(None),
                    ablation       = _sentinel(abl_np).astype(np.float32) if abl_np is not None else _sentinel(None),
                    input_ablation = _sentinel(inp_abl),
                    image          = image_crop.astype(np.float32),
                    prediction     = pred_crop.astype(np.float32),
                    label          = lbl_crop.astype(np.float32) if lbl_crop is not None else _sentinel(None),
                    zones          = zones_crop.astype(np.int8) if zones_crop is not None else _sentinel(None),
                    channels       = np.array(CHANNEL_NAMES),
                    case_id        = case_id,
                    fold           = fold,
                    model          = "nnunet",
                )
                print(f"    Saved: {out_file}")
                processed += 1

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np, occ_np, abl_np,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=confidence,
            )
            _save_progress(progress, progress_file)

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            progress[case_id] = {"done": False, "error": str(exc), "predicted_pos": None}
            _save_progress(progress, progress_file)

        if torch.cuda.is_available() and (i + 1) % 10 == 0:
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
) -> None:
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "nnunet":
        process_fold_nnunet(
            fold, model_output_dir, methods, skip_existing, occ_window, occ_stride, ppe, occ_crop_hw
        )
    else:
        process_fold_monai(
            fold, model_name, model_output_dir, methods, skip_existing, occ_window, occ_stride, ppe
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

            if predicted_pos and has_pca:
                classification = "tp"
            elif predicted_pos and not has_pca:
                classification = "fp"
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
                plot_detection_bar(records, zone_filter,
                                   f"{title_base}\nDetection Rate by Zone",
                                   out_dir / "detection.png")
                plot_distribution(subset, method,
                                  f"{title_base}\nChannel Attribution Distribution",
                                  out_dir / "distribution.png")
                n_charts += 3

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
        default=2,
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

    output_dir  = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)

    print(f"Models:           {models}")
    print(f"Folds:            {folds}")
    print(f"Methods:          {', '.join(sorted(methods))}")
    print(f"Output XAI dir:   {output_dir}")
    print(f"Output metrics:   {metrics_dir}")
    print(f"Occlusion window: {occ_window}  stride: {occ_stride}")

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
