#!/usr/bin/env python3
"""Fix nnUNet confidence values in progress.json files by re-running inference.

The old generate_xai_data.py stored pred_crop (a spatially cropped region) in the .npz,
so npz['prediction'].max() may not reflect the true maximum probability from the full
inference volume.

This script re-runs nnUNet inference for each case and extracts the correct confidence
from out[:, 1].max() (the network's softmax output).
"""

import importlib.util
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Paths (must match generate_xai_data.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

NNUNET_ROOT = PROJECT_ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_ROOT / "results" / "nnUNet"
TASK_NAME = "Task2203_picai_baseline"
DATASET_DIR = NNUNET_PREPROCESSED / TASK_NAME
PLANS_FILE = NNUNET_RESULTS / "plans.pkl"

IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)

# Set nnUNet env vars before any nnunet import
os.environ.setdefault("RESULTS_FOLDER", str(NNUNET_RESULTS))
os.environ.setdefault("nnUNet_raw_data_base", str(NNUNET_ROOT / "nnunet_base"))
os.environ.setdefault("nnUNet_preprocessed", str(NNUNET_PREPROCESSED))

# Add subproject roots to path so internal imports work
sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))

from src.utils import load_plans as _load_plans_from_file


# ---------------------------------------------------------------------------
# nnUNet helpers (copied from generate_xai_data.py)
# ---------------------------------------------------------------------------


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

    from nnUNet_addon.nnUNetTrainerV2_Loss_FL_and_CE import (
        nnUNetTrainerV2_Loss_FL_and_CE_checkpoints,
    )

    pkl_path = NNUNET_RESULTS / f"fold_{fold}" / "model_best.model.pkl"
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


def load_plans() -> dict:
    return _load_plans_from_file(PLANS_FILE)


def _preprocess_nnunet(case_id: str, plans: dict):
    """
    Load co-registered NIfTI files and preprocess with nnUNet's GenericPreprocessor.
    Returns (data: float32 (3, D, H, W), properties: dict).
    """
    from nnUNet.nnunet.preprocessing.preprocessing import GenericPreprocessor

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


def _compute_nnunet_divisors(plans: dict) -> Tuple[int, int, int]:
    pool_kernels = plans["plans_per_stage"][0]["pool_op_kernel_sizes"]
    d_div = h_div = w_div = 1
    for k in pool_kernels:
        d_div *= k[0]
        h_div *= k[1]
        w_div *= k[2]
    return d_div, h_div, w_div


def _pad(
    x: torch.Tensor, d_div: int, h_div: int, w_div: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _B, _C, D, H, W = x.shape
    D_pad = (-D) % d_div
    H_pad = (-H) % h_div
    W_pad = (-W) % w_div
    if D_pad + H_pad + W_pad:
        x = F.pad(x, (0, W_pad, 0, H_pad, 0, D_pad))
    return x, (D_pad, H_pad, W_pad)


# ---------------------------------------------------------------------------
# Main fix logic
# ---------------------------------------------------------------------------


def fix_fold(
    fold: int,
    base_dir: Path,
    network: torch.nn.Module,
    plans: dict,
    device: torch.device,
) -> tuple[int, int, int]:
    """Fix confidence values for a single fold by re-running inference.

    Returns:
        Tuple of (cases_updated, cases_skipped, cases_errored)
    """
    fold_dir = base_dir / f"fold_{fold}"
    progress_path = fold_dir / "progress.json"

    if not progress_path.exists():
        print(f"  Fold {fold}: progress.json not found, skipping")
        return 0, 0, 0

    with open(progress_path) as f:
        progress = json.load(f)

    d_div, h_div, w_div = _compute_nnunet_divisors(plans)

    updated = 0
    skipped = 0
    errored = 0

    case_ids = list(progress.keys())
    for i, case_id in enumerate(case_ids):
        data = progress[case_id]

        # Skip cases that weren't processed or had errors
        if not data.get("done", False) or data.get("error"):
            skipped += 1
            continue

        print(f"    [{i + 1}/{len(case_ids)}] {case_id}", end=" ")

        try:
            # Preprocess the case
            img_data, _ = _preprocess_nnunet(case_id, plans)

            # Convert to tensor and pad
            x = torch.from_numpy(img_data[np.newaxis]).to(device)  # (1, 3, D, H, W)
            x, _ = _pad(x, d_div, h_div, w_div)

            # Run inference
            with torch.no_grad():
                out = network(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                # Channel 1 is already softmaxed by the network
                cancer_prob = out[:, 1]
                correct_confidence = round(float(cancer_prob.max().item()), 4)

            old_confidence = data.get("confidence", 0.0)
            data["confidence"] = correct_confidence
            data["pred_max_prob"] = correct_confidence

            print(f"old={old_confidence:.4f} -> new={correct_confidence:.4f}")
            updated += 1

        except Exception as exc:
            print(f"ERROR: {exc}")
            errored += 1
            continue

    # Write back with same formatting
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)

    return updated, skipped, errored


def main():
    base_dir = Path("results/xai/nnunet")

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        return

    print("Fixing nnUNet confidence values via re-inference\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    plans = load_plans()

    total_updated = 0
    total_skipped = 0
    total_errored = 0

    for fold in range(5):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold}: Loading model...")

        try:
            network = load_nnunet(fold, device)
            print(f"  Model loaded successfully")
        except Exception as exc:
            print(f"  Failed to load model: {exc}")
            continue

        updated, skipped, errored = fix_fold(fold, base_dir, network, plans, device)
        total_updated += updated
        total_skipped += skipped
        total_errored += errored
        print(f"\n  Fold {fold}: {updated} updated, {skipped} skipped, {errored} errors")

        # Free memory
        del network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Total: {total_updated} updated, {total_skipped} skipped, {total_errored} errors")


if __name__ == "__main__":
    main()
