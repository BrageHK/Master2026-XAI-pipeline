#!/usr/bin/env python3
"""
Captum XAI (eXplainable AI) inference for PI-CAI nnUNet v1.

For each fold (0-4):
  - Loads the fold's model checkpoint (results/nnUNet/fold_X/model_best.model)
  - Reads that fold's validation cases from splits.json
  - Preprocesses raw .mha images using nnUNet's GenericPreprocessor
  - Generates Saliency and Occlusion attribution maps with Captum
  - Saves per-case .npz files to the output directory

Channels: t2w (ch 0, nonCT norm), adc (ch 1, CT norm), hbv (ch 2, nonCT norm)

Usage:
  uv run python captum_xai.py                        # all 5 folds
  uv run python captum_xai.py --fold 0               # single fold
  uv run python captum_xai.py --folds 0,1            # specific folds
  uv run python captum_xai.py --fold 0 --no-skip     # re-process existing
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import traceback
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from ablation_cam_3d import AblationCAM3D


# ---------------------------------------------------------------------------
# Paths — must be set before nnunet imports (nnunet reads env vars at import)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

NNUNET_PREPROCESSED = PROJECT_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS = PROJECT_ROOT / "results" / "nnUNet"
TASK_NAME = "Task2203_picai_baseline"

PLANS_FILE = NNUNET_RESULTS / "plans.pkl"
DATASET_DIR = NNUNET_PREPROCESSED / TASK_NAME

# Co-registered, resampled NIfTI files produced by picai_prep — these are the
# correct inputs for GenericPreprocessor (same voxel grid per modality).
# Naming: {case_id}_0000.nii.gz (t2w), _0001.nii.gz (adc), _0002.nii.gz (hbv)
IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)
LABELS_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0/workdir/nnUNet_raw_data/Task2203_picai_baseline/labelsTr"
)

SPLITS_FILE = PROJECT_ROOT / "splits.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "xai"

# Must be set before any nnunet import
os.environ["RESULTS_FOLDER"] = str(NNUNET_RESULTS)
os.environ["nnUNet_raw_data_base"] = str(PROJECT_ROOT / "nnunet_base")
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)

# Add paths so nnunet and local trainer class are importable
sys.path.insert(0, str(PROJECT_ROOT / "nnUNet"))
sys.path.insert(0, str(PROJECT_ROOT))

# nnunet imports (after env vars)
from nnUNet.nnunet.preprocessing.preprocessing import GenericPreprocessor
from nnUNet.nnunet.preprocessing.preprocessing import resample_data_or_seg
from captum.attr import Occlusion, Saliency  # noqa: E402


# ---------------------------------------------------------------------------
# Plans loading
# ---------------------------------------------------------------------------
def load_plans() -> dict:
    with open(PLANS_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _ensure_focal_loss_importable() -> None:
    """
    The project-root nnUNetTrainerV2_Loss_FL_and_CE.py imports FocalLoss via
    the nnunet module path.  The bundled nnUNet submodule ships a different
    focal-loss file that does not export that class.

    Patch sys.modules so the import resolves to the project-root file.
    """

    module_key = (
        "nnunet.training.network_training.nnUNet_variants"
        ".loss_function.nnUNetTrainerV2_focalLoss"
    )
    if module_key in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        module_key,
        str(PROJECT_ROOT / "nnUNet_addon" / "nnUNetTrainerV2_focalLoss.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]


def load_fold_network(fold: int) -> torch.nn.Module:
    """
    Load the nnUNet network for *fold* and return it in eval mode.

    The checkpoint pkl contains paths from the original training environment;
    we instantiate the trainer with local paths and load only the weights.
    """
    _ensure_focal_loss_importable()

    from nnUNet_addon.nnUNetTrainerV2_Loss_FL_and_CE import (  # noqa: E402
        nnUNetTrainerV2_Loss_FL_and_CE_checkpoints,
    )

    pkl_path = NNUNET_RESULTS / f"fold_{fold}" / "model_best.model.pkl"
    model_path = NNUNET_RESULTS / f"fold_{fold}" / "model_best.model"

    with open(pkl_path, "rb") as f:
        info = pickle.load(f)

    tmp_out = tempfile.mkdtemp(prefix=f"nnunet_fold{fold}_")

    # Instantiate trainer with local paths; original init args (stored in the
    # checkpoint) point to the training environment and cannot be used here.
    # Args follow nnUNetTrainerV2.__init__:
    #   plans_file, fold, output_folder, dataset_directory,
    #   batch_dice, stage, unpack_data, deterministic, fp16
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

    # Supply the plans from the checkpoint so initialize() skips disk reading
    trainer.process_plans(info["plans"])

    # Build network architecture (does NOT need data on disk)
    trainer.initialize(False)

    # Load trained weights
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    trainer.load_checkpoint_ram(checkpoint, False)

    network = trainer.network
    network.eval()
    # Disable deep-supervision heads so forward() returns a single tensor
    if hasattr(network, "do_ds"):
        network.do_ds = False

    # Determine device: prefer CUDA but fall back to CPU if CUDA kernels are
    # incompatible with the installed PyTorch (e.g. P100/sm_60 with PyTorch
    # compiled for sm_70+).
    if torch.cuda.is_available():
        try:
            network = network.cuda()
            # Quick smoke-test to catch sm_XX incompatibility before the real run
            _probe = torch.zeros(1, device="cuda")
            network(_probe.new_zeros(1, 3, 16, 64, 64))
        except Exception:
            print("  WARNING: CUDA forward pass failed; falling back to CPU.")
            network = network.cpu()
    else:
        network = network.cpu()

    return network


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_case(case_id: str, plans: dict) -> np.ndarray:
    """
    Load the co-registered NIfTI files for *case_id* from IMAGES_TR and
    preprocess them with nnUNet's GenericPreprocessor (resample + normalize).

    The NIfTI files were produced by picai_prep and are already aligned to the
    same voxel grid.  Raw .mha files from the PI-CAI dataset have different
    resolutions per modality and cannot be stacked directly.

    Channel order: [0] t2w (_0000, nonCT), [1] adc (_0001, CT), [2] hbv (_0002, nonCT)

    Returns:
        data: float32 ndarray of shape (3, D, H, W)
    """
    input_files: List[str] = []
    for ch in range(3):
        fpath = IMAGES_TR / f"{case_id}_{ch:04d}.nii.gz"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Preprocessed NIfTI not found: {fpath}\n"
                f"Expected co-registered files in {IMAGES_TR}"
            )
        input_files.append(str(fpath))

    target_spacing = plans["plans_per_stage"][0]["current_spacing"]

    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality=plans["normalization_schemes"],
        use_nonzero_mask=plans["use_mask_for_norm"],
        transpose_forward=plans["transpose_forward"],
        intensityproperties=plans["dataset_properties"]["intensityproperties"],
    )

    data, _seg, _properties = preprocessor.preprocess_test_case(
        input_files, target_spacing
    )
    return data.astype(np.float32)  # (3, D, H, W)


def preprocess_case_with_props(case_id: str, plans: dict):
    """Like preprocess_case() but also returns nnUNet's preprocessing properties dict.

    The properties dict contains 'crop_bbox' (the nonzero bounding box applied by
    crop_to_nonzero, in pre-transpose space) and 'original_size_of_raw_data'.

    Returns:
        data       : float32 ndarray (3, D, H, W)
        properties : dict
    """
    input_files: List[str] = []
    for ch in range(3):
        fpath = IMAGES_TR / f"{case_id}_{ch:04d}.nii.gz"
        if not fpath.exists():
            raise FileNotFoundError(
                f"Preprocessed NIfTI not found: {fpath}\n"
                f"Expected co-registered files in {IMAGES_TR}"
            )
        input_files.append(str(fpath))

    target_spacing = plans["plans_per_stage"][0]["current_spacing"]

    preprocessor = GenericPreprocessor(
        normalization_scheme_per_modality=plans["normalization_schemes"],
        use_nonzero_mask=plans["use_mask_for_norm"],
        transpose_forward=plans["transpose_forward"],
        intensityproperties=plans["dataset_properties"]["intensityproperties"],
    )

    data, _seg, properties = preprocessor.preprocess_test_case(
        input_files, target_spacing
    )
    return data.astype(np.float32), properties


def load_label_for_case(case_id: str, plans: dict) -> Optional[np.ndarray]:
    """
    Load the binary segmentation label for *case_id* and resample it to match
    the coordinate space returned by preprocess_case().

    Mirrors the spatial preprocessing steps of GenericPreprocessor
    (transpose + resample to target spacing) but uses nearest-neighbour
    interpolation appropriate for a segmentation mask.

    Returns:
        (D, H, W) float32 binary array (0 = background, 1 = cancer), or None
        if the label file does not exist.
    """
    label_path = LABELS_TR / f"{case_id}.nii.gz"
    if not label_path.exists():
        return None


    label_itk = sitk.ReadImage(str(label_path))

    # ITK spacing is (x, y, z); convert to (z, y, x) = (D, H, W) numpy order
    itk_spacing = np.array(label_itk.GetSpacing())[::-1]  # (z, y, x)
    label_np = sitk.GetArrayFromImage(label_itk).astype(np.float32)  # (D, H, W)

    # Apply the same axis transposition as the image preprocessor
    transpose_forward = plans["transpose_forward"]
    label_np = label_np.transpose(transpose_forward)          # e.g. (D, H, W) → stays or reorders
    itk_spacing = itk_spacing[list(transpose_forward)]        # match axis order

    # Compute target shape from spacing ratio (same logic as GenericPreprocessor)
    target_spacing = np.array(plans["plans_per_stage"][0]["current_spacing"])
    new_shape = np.round(
        itk_spacing / target_spacing * np.array(label_np.shape)
    ).astype(int)

    # Resample with nearest-neighbour (order=0, is_seg=True)
    label_resampled = resample_data_or_seg(
        label_np[np.newaxis],  # (1, D, H, W)
        new_shape,
        is_seg=True,
        axis=None,
        order=1,
        do_separate_z=False,
    )[0]  # (D, H, W)

    return (label_resampled > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Padding helpers (network requires spatial dims divisible by pooling stride)
# ---------------------------------------------------------------------------
def _compute_divisors(plans: dict) -> Tuple[int, int, int]:
    """
    Return (D_div, H_div, W_div) needed so the volume can pass through the
    full UNet encoder/decoder without shape mismatches.
    """
    pool_kernels = plans["plans_per_stage"][0]["pool_op_kernel_sizes"]
    d_div, h_div, w_div = 1, 1, 1
    for k in pool_kernels:
        d_div *= k[0]
        h_div *= k[1]
        w_div *= k[2]
    return d_div, h_div, w_div


def _pad(
    x: torch.Tensor,
    d_div: int,
    h_div: int,
    w_div: int,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Pad (1, C, D, H, W) to be divisible by the given factors."""
    _B, _C, D, H, W = x.shape
    D_pad = (-D) % d_div
    H_pad = (-H) % h_div
    W_pad = (-W) % w_div
    if D_pad + H_pad + W_pad:
        x = F.pad(x, (0, W_pad, 0, H_pad, 0, D_pad))
    return x, (D_pad, H_pad, W_pad)


def _unpad(
    arr: np.ndarray,
    original_dhw: Tuple[int, int, int],
) -> np.ndarray:
    """Remove padding from (C, D', H', W') → (C, D, H, W)."""
    D, H, W = original_dhw
    return arr[:, :D, :H, :W]


# ---------------------------------------------------------------------------
# Captum forward wrapper
# ---------------------------------------------------------------------------
def _make_forward_func(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """
    Return a function (B, C, D, H, W) → (B,) scalar for Captum.

    Sums the softmax cancer probability over the voxels that were predicted
    as cancer on the *original* (unperturbed) input.  The mask is fixed so
    that perturbing the input cannot change which voxels are evaluated —
    otherwise a patch that merely shifts a boundary voxel across the 0.5
    threshold would appear unimportant even though it clearly matters.

    fixed_mask: bool tensor of shape (1, D, H, W), True where original
                cancer_prob > 0.5.  Must match the spatial size of `inp`.
    """

    def forward_func(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)
        # Guard: deep supervision can return a tuple; take the final output
        if isinstance(out, (list, tuple)):
            out = out[0]
        cancer_prob = torch.softmax(out, dim=1)[:, 1]  # (B, D, H, W)
        return (cancer_prob * fixed_mask).flatten(1).sum(dim=1)  # (B,)

    return forward_func


# ---------------------------------------------------------------------------
# XAI generation
# ---------------------------------------------------------------------------
def generate_xai(
    network: torch.nn.Module,
    data: np.ndarray,
    plans: dict,
    occlusion_window: Tuple[int, int, int, int] = (1, 2, 16, 16),
    occlusion_stride: Tuple[int, int, int, int] = (1, 1, 16, 16),
    perturbations_per_eval: int = 1,
    occ_crop_hw: Optional[int] = 128,
    label_np: Optional[np.ndarray] = None,
    run_saliency: bool = False,
    run_occlusion: bool = False,
    run_ablation_cam: bool = False,
    run_input_ablation: bool = False,
) -> Tuple:
    """
    Generate attribution maps for the requested XAI methods.

    Args:
        network:              eval-mode nnUNet network
        data:                 preprocessed (3, D, H, W) float32 array
        plans:                nnUNet plans dict
        occlusion_window:     (C, D, H, W) shape of the occlusion sliding window
        occlusion_stride:     (C, D, H, W) strides for the sliding window
        perturbations_per_eval: batch size for occlusion forward passes
        occ_crop_hw:          if set, tumor-centered crop H and W to this size
                              before running saliency/occlusion.
                              None disables cropping (full volume).
        label_np:             optional (D, H, W) binary segmentation label;
                              will be cropped to match the XAI output region.
        run_saliency:         compute gradient-based Saliency attribution.
        run_occlusion:        compute sliding-window Occlusion attribution.
        run_ablation_cam:     compute 3D AblationCAM (slow — one forward pass
                              per channel in the target layer).
        run_input_ablation:   compute per-input-channel importance weights by
                              zeroing each channel and measuring the score drop,
                              mirroring AblationCAM's scoring formula.

    Returns:
        (saliency, occlusion, ablation, input_ablation_weights, image, prediction, label):
        saliency/occlusion/ablation are (3-or-1, D_crop, H_crop, W_crop) float32.
        input_ablation_weights is a (n_channels,) float32 array of fractional
        importance scores (one weight per input channel), or None if not requested.
        A method's array is None when that method was not requested or failed.
        Returns (None, ..., None) if no cancer voxels detected.
    """
    device = next(network.parameters()).device
    d_div, h_div, w_div = _compute_divisors(plans)
    original_dhw: Tuple[int, int, int] = tuple(data.shape[1:])  # type: ignore[assignment]

    x = torch.from_numpy(data[np.newaxis]).to(device)  # (1, 3, D, H, W)
    x, _padding = _pad(x, d_div, h_div, w_div)

    # ---- Cancer detection probe (no grad) --------------------------------
    with torch.no_grad():
        out = network(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        cancer_prob = torch.softmax(out, dim=1)[:, 1]  # (1, D, H, W)
        # Precompute the mask once from the original input so that perturbing
        # the input cannot change which voxels count toward the score.
        fixed_mask = (cancer_prob > 0.5)  # (1, D, H, W) bool
        cancer_voxels = fixed_mask.sum().item()

    if cancer_voxels == 0:
        print("    No cancer detected (P(cancer)>0.5 sum=0) — skipping XAI.")
        return None, None, None, None, None, None, None, None, None, None, None

    print(f"    Cancer voxels detected: {int(cancer_voxels)} — running XAI.")

    # ---- Shared crop coordinates ------------------------------------------
    mask_np = fixed_mask[0].cpu().numpy()  # (D_pad, H_pad, W_pad) bool
    coords = np.argwhere(mask_np)           # (N, 3): rows are (d, h, w)

    # Depth: cancer slices ± 1 margin, clamped to original volume
    d_min = int(coords[:, 0].min())
    d_max = int(coords[:, 0].max())
    d0 = max(0, d_min - 1)
    d1 = min(original_dhw[0], d_max + 2)  # exclusive upper bound

    # H/W: tumor-centered patch
    if occ_crop_hw is not None:
        _B, _C, _D, H_pad, W_pad = x.shape
        _, hc, wc = coords.mean(axis=0).astype(int)
        h0 = int(np.clip(hc - occ_crop_hw // 2, 0, H_pad - occ_crop_hw))
        w0 = int(np.clip(wc - occ_crop_hw // 2, 0, W_pad - occ_crop_hw))
        print(f"    Crop: d={d0}:{d1} ({d1-d0} slices), h0={h0}, w0={w0}, hw={occ_crop_hw}"
              f" (centroid h={hc}, w={wc})")
        unpad_dhw = (original_dhw[0], occ_crop_hw, occ_crop_hw)
        fixed_mask_crop = fixed_mask[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
    else:
        h0, w0 = 0, 0
        print(f"    Depth crop: d={d0}:{d1} ({d1-d0} slices)")
        unpad_dhw = original_dhw
        fixed_mask_crop = fixed_mask

    # ---- Saliency (single backward pass, full volume then crop) ----------
    sal_np = None
    if run_saliency:
        forward_func_sal = _make_forward_func(network, fixed_mask)
        x_sal = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sal_attr = Saliency(forward_func_sal).attribute(x_sal, abs=True)
        sal_np = _unpad(sal_attr.detach().cpu().numpy()[0], original_dhw)
        if occ_crop_hw is not None:
            sal_np = sal_np[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
        sal_np = sal_np[:, d0:d1]

    # ---- Occlusion (sliding-window perturbation on cropped volume) -------
    # sliding_window_shapes: (C, D, H, W) — shape of one occluded patch
    # strides:               (C, D, H, W) — step between patches
    occ_np = None
    x_occ = x.detach().clone()
    if occ_crop_hw is not None:
        x_occ = x_occ[:, :, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
    if run_occlusion:
        forward_func_occ = _make_forward_func(network, fixed_mask_crop)
        with torch.enable_grad():
            occ_attr = Occlusion(forward_func_occ).attribute(
                x_occ,
                sliding_window_shapes=occlusion_window,
                strides=occlusion_stride,
                baselines=0.0,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=True,
            )
        occ_np = _unpad(occ_attr.detach().cpu().numpy()[0], unpad_dhw)
        occ_np = occ_np[:, d0:d1]

    # ---- AblationCAM (optional, slow — one forward pass per channel) ------
    ablation_np = None
    if run_ablation_cam:
        try:
            print("    Running 3D AblationCAM…")
            target_layer = network.conv_blocks_context[6][1].blocks[0].conv
            print(f"    Target layer: {target_layer.__class__.__name__}(out_channels={target_layer.out_channels})")
            cam = AblationCAM3D(network, [target_layer], batch_size=16)

            # AblationCAM calls target(model_output_item) where each item has
            # shape (2, D, H, W) — the raw segmentation logits for one sample.
            # We compute the same masked cancer-probability score as in
            # _make_forward_func, but without an extra network forward pass.
            _mask = fixed_mask_crop[0].float()  # (D_pad, H_crop, W_crop)

            def _abl_target(output):
                if isinstance(output, (list, tuple)):
                    output = output[0]
                cancer_prob = torch.softmax(output.unsqueeze(0), dim=1)[0, 1]
                return (cancer_prob * _mask).sum()

            abl_maps = cam(x_occ, targets=[_abl_target])  # (1, D_pad, H_crop, W_crop)
            ablation_np = _unpad(abl_maps, unpad_dhw)    # (1, D, H_crop, W_crop)
            ablation_np = ablation_np[:, d0:d1]          # (1, D_crop, H_crop, W_crop)
        except Exception as exc:
            import traceback as _tb
            print(f"    AblationCAM failed: {exc}")
            _tb.print_exc()
            ablation_np = None

    # ---- Input Ablation (one forward pass per input channel) -------------
    # Each input channel is spatially shuffled (permuted) in turn.  Shuffling
    # destroys all spatial structure while preserving the value distribution,
    # so the model sees in-distribution inputs and the result is not confounded
    # by the normalisation scheme (unlike zeroing, which maps z-score channels
    # to "average intensity everywhere" — a misleading and sometimes adversarial
    # baseline that can cause the score to *increase*, yielding negative weights).
    #
    #   weight[c] = (original_score - score_with_channel_c_shuffled) / original_score
    input_abl_weights = None
    if run_input_ablation:
        print("    Running Input Ablation (permutation importance)…")
        forward_func_abl = _make_forward_func(network, fixed_mask_crop)
        with torch.no_grad():
            original_score = forward_func_abl(x_occ).item()

        n_channels = x_occ.shape[1]
        weights = []
        for ch in range(n_channels):
            x_ablated = x_occ.clone()
            flat = x_occ[:, ch].reshape(-1)
            perm = torch.randperm(flat.numel(), device=flat.device)
            x_ablated[:, ch] = flat[perm].reshape(x_occ[:, ch].shape)
            ablated_score = forward_func_abl(x_ablated).item()
            w = (original_score - ablated_score) / original_score if original_score != 0 else 0.0
            weights.append(w)
            print(f"      ch {ch}: original={original_score:.4f}  ablated={ablated_score:.4f}  weight={w:.4f}")

        input_abl_weights = np.array(weights, dtype=np.float32)

    # ---- Input image crop (for visualization alongside attributions) ------
    if occ_crop_hw is not None:
        image_crop = data[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
    else:
        image_crop = data[:, d0:d1]

    # ---- Model prediction (cancer probability, same crop as image) --------
    prob_np = _unpad(cancer_prob.detach().cpu().numpy(), original_dhw)  # (1, D, H, W)
    if occ_crop_hw is not None:
        prob_np = prob_np[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
    pred_crop = prob_np[:, d0:d1]  # (1, D_crop, H_crop, W_crop)

    # ---- Ground-truth label crop (optional) ------------------------------
    if label_np is not None:
        if occ_crop_hw is not None:
            lbl_crop = label_np[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
        else:
            lbl_crop = label_np[d0:d1]
        lbl_crop = lbl_crop[np.newaxis].astype(np.float32)  # (1, D_crop, H_crop, W_crop)
    else:
        lbl_crop = None

    return (
        sal_np.astype(np.float32) if sal_np is not None else None,
        occ_np.astype(np.float32) if occ_np is not None else None,
        ablation_np.astype(np.float32) if ablation_np is not None else None,
        input_abl_weights,  # (n_channels,) float32 or None
        image_crop.astype(np.float32),
        pred_crop.astype(np.float32),
        lbl_crop,
        d0, d1, h0, w0,
    )


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def load_splits() -> List[Dict]:
    with open(SPLITS_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-fold processing
# ---------------------------------------------------------------------------
def process_fold(
    fold: int,
    val_cases: List[str],
    output_dir: Path,
    plans: dict,
    skip_existing: bool = True,
    occlusion_window: Tuple[int, int, int, int] = (3, 7, 128, 128),
    occlusion_stride: Tuple[int, int, int, int] = (3, 7, 128, 128),
    perturbations_per_eval: int = 32,
    occ_crop_hw: Optional[int] = 128,
    run_saliency: bool = False,
    run_occlusion: bool = False,
    run_ablation_cam: bool = False,
    run_input_ablation: bool = False,
) -> None:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fold {fold} — loading model…")
    network = load_fold_network(fold)
    device = next(network.parameters()).device
    print(f"Model ready on {device}. Validation cases: {len(val_cases)}")

    processed, skipped_existing, no_cancer, errors = 0, 0, 0, 0

    for i, case_id in enumerate(val_cases):
        out_file = fold_dir / f"{case_id}.npz"

        if skip_existing and out_file.exists():
            skipped_existing += 1
            continue

        print(f"\n  [{i+1}/{len(val_cases)}] {case_id}")

        try:
            # -- Preprocess --------------------------------------------------
            data = preprocess_case(case_id, plans)
            print(f"    Preprocessed shape: {data.shape}")

            # -- Load label (best-effort) ------------------------------------
            label_np = load_label_for_case(case_id, plans)
            if label_np is None:
                print("    Label not found — prediction-only mode.")

            # -- XAI ---------------------------------------------------------
            sal, occ, abl, inp_abl, img, pred, lbl, d0, d1, h0, w0 = generate_xai(
                network,
                data,
                plans,
                occlusion_window=occlusion_window,
                occlusion_stride=occlusion_stride,
                perturbations_per_eval=perturbations_per_eval,
                occ_crop_hw=occ_crop_hw,
                label_np=label_np,
                run_saliency=run_saliency,
                run_occlusion=run_occlusion,
                run_ablation_cam=run_ablation_cam,
                run_input_ablation=run_input_ablation,
            )

            # img is always computed; None only on the no-cancer early exit.
            if img is None:
                no_cancer += 1
                continue

            # -- Save --------------------------------------------------------
            # Unrun/failed methods are stored as 0-length sentinels so
            # np.savez always writes every key.
            def _sentinel(arr):
                return arr if arr is not None else np.zeros((0,), dtype=np.float32)

            np.savez_compressed(
                out_file,
                saliency=_sentinel(sal),         # (3, D, crop_hw, crop_hw) or sentinel
                occlusion=_sentinel(occ),        # (3, D, crop_hw, crop_hw) or sentinel
                ablation=_sentinel(abl),         # (1, D, crop_hw, crop_hw) or sentinel
                input_ablation=_sentinel(inp_abl),  # (n_channels,) weights or sentinel
                image=img,                       # (3, D, crop_hw, crop_hw) or (3, D, H, W)
                prediction=pred,                 # (1, D, crop_hw, crop_hw) — cancer probability
                label=_sentinel(lbl),
                channels=np.array(["t2w", "adc", "hbv"]),
                case_id=case_id,
                fold=fold,
                d0=np.int32(d0), d1=np.int32(d1),
                h0=np.int32(h0), w0=np.int32(w0),
            )
            print(f"    Saved: {out_file}  shape={img.shape}")
            processed += 1

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1

        # Periodically free GPU memory
        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # Write a simple fold summary
    summary = {
        "fold": fold,
        "total_val_cases": len(val_cases),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "no_cancer_skipped": no_cancer,
        "errors": errors,
    }
    summary_path = fold_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFold {fold} done — processed={processed}, skipped_existing={skipped_existing}, no_cancer={no_cancer}, errors={errors}")
    print(f"Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Captum XAI (Saliency + Occlusion) for PI-CAI nnUNet validation sets",
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
        help="Comma-separated list of folds to process (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Root output directory; fold-level subdirs are created automatically.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process a case even if its .npz output already exists.",
    )
    parser.add_argument(
        "--occlusion-window",
        type=str,
        default="1,2,12,12",
        metavar="C,D,H,W",
        help=(
            "Sliding window shape for Occlusion (channel, depth, height, width). "
            "Larger values = faster but coarser attribution."
        ),
    )
    parser.add_argument(
        "--occlusion-stride",
        type=str,
        default="1,1,6,6",
        metavar="C,D,H,W",
        help="Strides for the Occlusion sliding window.",
    )
    parser.add_argument(
        "--perturbations-per-eval",
        type=int,
        default=1,
        help="Number of occlusion perturbations per forward pass (higher = faster, more VRAM).",
    )
    parser.add_argument(
        "--occ-crop-hw",
        type=int,
        default=128,
        metavar="N",
        help=(
            "Center-crop H and W to N pixels before running occlusion "
            "(reduces forward passes from ceil(H/stride)² to 1). "
            "Set to 0 to disable cropping and use the full volume."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="METHOD",
        required=True,
        choices=["saliency", "occlusion", "ablation", "input_ablation", "all"],
        help=(
            "XAI methods to run: saliency, occlusion, ablation, input_ablation, or all. "
            "Multiple values accepted (e.g. --methods saliency input_ablation). "
            "'ablation' is slow — one forward pass per channel in the target layer. "
            "'input_ablation' zeros each input channel and measures the score drop, "
            "returning a (n_channels,) weight vector of fractional importance."
        ),
    )

    args = parser.parse_args()

    # Determine folds to process
    if args.fold is not None:
        folds = [args.fold]
    elif args.folds is not None:
        folds = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds = list(range(5))

    occ_window: Tuple[int, int, int, int] = tuple(  # type: ignore[assignment]
        int(v) for v in args.occlusion_window.split(",")
    )
    occ_stride: Tuple[int, int, int, int] = tuple(  # type: ignore[assignment]
        int(v) for v in args.occlusion_stride.split(",")
    )

    splits = load_splits()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = load_plans()
    print(f"Plans loaded from: {PLANS_FILE}")
    print(f"Target spacing:    {plans['plans_per_stage'][0]['current_spacing']}")
    print(f"Output directory:  {output_dir}")
    print(f"Folds to process:  {folds}")
    methods = set(args.methods)
    if "all" in methods:
        methods = {"saliency", "occlusion", "ablation", "input_ablation"}
    run_saliency = "saliency" in methods
    run_occlusion = "occlusion" in methods
    run_ablation_cam = "ablation" in methods
    run_input_ablation = "input_ablation" in methods

    print(f"Occlusion window:  {occ_window}  stride: {occ_stride}")
    print(f"Methods:           {', '.join(sorted(methods))}")

    occ_crop_hw: Optional[int] = args.occ_crop_hw if args.occ_crop_hw > 0 else None

    for fold in folds:
        if fold >= len(splits):
            print(f"WARNING: fold {fold} not found in splits.json (only {len(splits)} folds). Skipping.")
            continue
        val_cases: List[str] = splits[fold]["val"]
        process_fold(
            fold=fold,
            val_cases=val_cases,
            output_dir=output_dir,
            plans=plans,
            skip_existing=not args.no_skip,
            occlusion_window=occ_window,
            occlusion_stride=occ_stride,
            perturbations_per_eval=args.perturbations_per_eval,
            occ_crop_hw=occ_crop_hw,
            run_saliency=run_saliency,
            run_occlusion=run_occlusion,
            run_ablation_cam=run_ablation_cam,
            run_input_ablation=run_input_ablation,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
