#!/usr/bin/env python3
"""
Captum XAI (eXplainable AI) inference for U-MambaMTL and SwinUNETR.

For each fold (0-4):
  - Loads the fold's model checkpoint
  - Reads that fold's validation cases via the MONAI DataModule
  - Generates Saliency, Occlusion, AblationCAM, and/or Input Ablation
    attribution maps with Captum / pytorch-grad-cam
  - Saves per-case .npz files matching the format produced by the nnUNet
    captum_xai.py (same keys, same (C, D, H, W) axis order)

Channels: t2w (ch 0), adc (ch 1), hbv (ch 2)

Usage:
  python captum_xai.py --model umamba_mtl --fold 0 --methods saliency
  python captum_xai.py --model swin_unetr --folds 0,1 --methods saliency occlusion
  python captum_xai.py --model umamba_mtl --fold 0 --methods all
  python captum_xai.py --model umamba_mtl --fold 0 --methods saliency --no-skip
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Add the nnUNet XAI directory to sys.path so we can import AblationCAM3D
# and find_decoder_feature_layers, which live alongside captum_xai.py there.
NNUNET_XAI_DIR = PROJECT_ROOT.parent / "picai_nnunet_semi_supervised_gc_algorithm"
sys.path.insert(0, str(NNUNET_XAI_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / "gc_algorithms" / "base_container" / "models"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "xai"

CHANNEL_NAMES = ["t2w", "adc", "hbv"]

from captum.attr import Occlusion, Saliency  # noqa: E402
from shared_modules.data_module import DataModule  # noqa: E402
from shared_modules.utils import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_name: str, fold: int, device: torch.device) -> torch.nn.Module:
    """
    Load and return the eval-mode network for the given model and fold.

    The LitModel wrapper is used only to restore weights; we return the
    underlying nn.Module so Captum can differentiate through it directly.
    """
    if model_name == "umamba_mtl":
        from experiments.picai.umamba_mtl.trainer import LitModel
    elif model_name == "swin_unetr":
        from experiments.picai.swin_unetr.trainer import LitModel
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    config = load_config(f"experiments/picai/{model_name}/config.yaml")
    config.data.json_list = f"json_datalists/picai/fold_{fold}.json"
    config.gpus = [device.index if device.type == "cuda" else 0]
    config.cache_rate = 0.0

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


# ---------------------------------------------------------------------------
# Captum forward wrapper
# ---------------------------------------------------------------------------
def _make_forward_func(network: torch.nn.Module, fixed_mask: torch.Tensor):
    """
    Return a callable (B, C, H, W, D) → (B,) for Captum.

    Applies sigmoid to channel 1 of the network output (the PCA channel for
    both umamba_mtl and swin_unetr) and sums the probability over the voxels
    that were detected as cancerous on the *original* unperturbed input.
    The mask is fixed so perturbing the input cannot shift boundary voxels
    into or out of the evaluation region.

    fixed_mask: bool tensor of shape (1, H, W, D), True where cancer_prob > 0.5.
    """

    def forward_func(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)  # (B, C, H, W, D) — C=5 for umamba_mtl, C=2 for swin_unetr
        cancer_prob = torch.sigmoid(out[:, 1])  # (B, H, W, D)
        return (cancer_prob * fixed_mask).flatten(1).sum(dim=1)  # (B,)

    return forward_func


# ---------------------------------------------------------------------------
# XAI generation
# ---------------------------------------------------------------------------
def generate_xai(
    network: torch.nn.Module,
    batch: dict,
    occlusion_window: Tuple[int, int, int, int] = (1, 16, 16, 2),
    occlusion_stride: Tuple[int, int, int, int] = (1, 8, 8, 1),
    perturbations_per_eval: int = 1,
    run_saliency: bool = False,
    run_occlusion: bool = False,
    run_ablation_cam: bool = False,
    run_input_ablation: bool = False,
) -> Tuple[Optional[np.ndarray], ...]:
    """
    Generate attribution maps for the requested XAI methods.

    The DataModule already handles all preprocessing (resampling, cropping,
    normalisation) so `batch["image"]` arrives ready for the network as
    (1, 3, H, W, D).

    All outputs are permuted from the model's native (C, H, W, D) axis order
    to (C, D, H, W) to match the nnUNet captum_xai.py save format, then
    depth-cropped to the cancer-containing slices ± 1 margin.

    Axis conventions:
      - Model I/O:  (B, C, H, W, D)  — D is last (depth)
      - Saved .npz: (C, D, H, W)     — D is second (matches nnUNet format)

    Occlusion window / stride follow (C, H, W, D) order (matching model I/O).

    Returns:
        (saliency, occlusion, ablation, input_ablation_weights,
         image, prediction, label)
        Each attribution is (C, D_crop, H, W) float32 or None if not requested
        or failed.  input_ablation_weights is (n_channels,) float32 or None.
        Returns all-None 7-tuple if no cancer voxels are detected.
    """
    device = next(network.parameters()).device
    x = batch["image"].to(device)  # (1, 3, H, W, D)

    # ---- Cancer detection probe (no grad) --------------------------------
    with torch.no_grad():
        out = network(x)
        cancer_prob = torch.sigmoid(out[:, 1])  # (1, H, W, D)
        fixed_mask = (cancer_prob > 0.5)         # (1, H, W, D) bool
        cancer_voxels = fixed_mask.sum().item()

    if cancer_voxels == 0:
        print("    No cancer detected (P(cancer)>0.5 sum=0) — skipping XAI.")
        return None, None, None, None, None, None, None

    print(f"    Cancer voxels detected: {int(cancer_voxels)} — running XAI.")

    # ---- Depth crop coordinates ------------------------------------------
    # fixed_mask shape: (1, H, W, D) — D is axis 3
    mask_np = fixed_mask[0].cpu().numpy()   # (H, W, D)
    coords = np.argwhere(mask_np)            # (N, 3): each row is (h, w, d)
    D = x.shape[4]

    d_min = int(coords[:, 2].min())
    d_max = int(coords[:, 2].max())
    d0 = max(0, d_min - 1)
    d1 = min(D, d_max + 2)  # exclusive upper bound
    print(f"    Depth crop: d={d0}:{d1} ({d1 - d0} slices)")

    forward_func = _make_forward_func(network, fixed_mask)

    # ---- Saliency (single backward pass through full volume) -------------
    sal_np = None
    if run_saliency:
        x_sal = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            sal_attr = Saliency(forward_func).attribute(x_sal, abs=True)
        # (1, 3, H, W, D) → (3, H, W, D) → permute → (3, D, H, W) → depth crop
        sal_np = sal_attr.detach().cpu().numpy()[0]  # (3, H, W, D)
        sal_np = sal_np.transpose(0, 3, 1, 2)         # (3, D, H, W)
        sal_np = sal_np[:, d0:d1]                      # (3, D_crop, H, W)

    # ---- Occlusion (sliding-window perturbation) -------------------------
    occ_np = None
    if run_occlusion:
        with torch.enable_grad():
            occ_attr = Occlusion(forward_func).attribute(
                x.detach().clone(),
                sliding_window_shapes=occlusion_window,  # (C, H, W, D)
                strides=occlusion_stride,                 # (C, H, W, D)
                baselines=0.0,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=True,
            )
        occ_np = occ_attr.detach().cpu().numpy()[0]  # (3, H, W, D)
        occ_np = occ_np.transpose(0, 3, 1, 2)         # (3, D, H, W)
        occ_np = occ_np[:, d0:d1]                      # (3, D_crop, H, W)

    # ---- AblationCAM (one forward pass per channel in the target layer) --
    ablation_np = None
    if run_ablation_cam:
        try:
            from ablation_cam_3d import AblationCAM3D, find_decoder_feature_layers

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
            ablation_np = abl_maps_np[:, d0:d1]            # (1, D_crop, H, W)
        except Exception as exc:
            print(f"    AblationCAM failed: {exc}")
            traceback.print_exc()
            ablation_np = None

    # ---- Input Ablation (permutation importance per input channel) -------
    input_abl_weights = None
    if run_input_ablation:
        print("    Running Input Ablation (permutation importance)…")
        forward_func_abl = _make_forward_func(network, fixed_mask)
        with torch.no_grad():
            original_score = forward_func_abl(x).item()

        n_channels = x.shape[1]
        weights = []
        for ch in range(n_channels):
            x_ablated = x.clone()
            flat = x[:, ch].reshape(-1)
            perm = torch.randperm(flat.numel(), device=flat.device)
            x_ablated[:, ch] = flat[perm].reshape(x[:, ch].shape)
            with torch.no_grad():
                ablated_score = forward_func_abl(x_ablated).item()
            w = (original_score - ablated_score) / original_score if original_score != 0 else 0.0
            weights.append(w)
            print(f"      ch {ch}: original={original_score:.4f}  "
                  f"ablated={ablated_score:.4f}  weight={w:.4f}")

        input_abl_weights = np.array(weights, dtype=np.float32)

    # ---- Image crop (for visualisation alongside attributions) -----------
    image_np = batch["image"][0].cpu().numpy()  # (3, H, W, D)
    image_np = image_np.transpose(0, 3, 1, 2)    # (3, D, H, W)
    image_crop = image_np[:, d0:d1]              # (3, D_crop, H, W)

    # ---- Prediction crop (cancer probability) ----------------------------
    pred_np = cancer_prob[0].cpu().numpy()  # (H, W, D)
    pred_np = pred_np.transpose(2, 0, 1)    # (D, H, W)
    pred_crop = pred_np[np.newaxis, d0:d1]  # (1, D_crop, H, W)

    # ---- Label crop (ground-truth PCA mask) ------------------------------
    lbl_np = batch["pca"][0, 0].cpu().numpy()  # (H, W, D)
    lbl_np = lbl_np.transpose(2, 0, 1)          # (D, H, W)
    lbl_crop = lbl_np[np.newaxis, d0:d1]        # (1, D_crop, H, W)

    return (
        sal_np.astype(np.float32) if sal_np is not None else None,
        occ_np.astype(np.float32) if occ_np is not None else None,
        ablation_np.astype(np.float32) if ablation_np is not None else None,
        input_abl_weights,
        image_crop.astype(np.float32),
        pred_crop.astype(np.float32),
        lbl_crop.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Per-fold processing
# ---------------------------------------------------------------------------
def process_fold(
    fold: int,
    model_name: str,
    output_dir: Path,
    skip_existing: bool = True,
    occlusion_window: Tuple[int, int, int, int] = (1, 16, 16, 2),
    occlusion_stride: Tuple[int, int, int, int] = (1, 8, 8, 1),
    perturbations_per_eval: int = 1,
    run_saliency: bool = False,
    run_occlusion: bool = False,
    run_ablation_cam: bool = False,
    run_input_ablation: bool = False,
) -> None:
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Model: {model_name}  |  Fold {fold} — loading model…")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_model(model_name, fold, device)
    print(f"Model ready on {device}.")

    # ---- DataModule -------------------------------------------------------
    if model_name == "umamba_mtl":
        from experiments.picai.umamba_mtl.trainer import LitModel  # noqa: F401
    elif model_name == "swin_unetr":
        from experiments.picai.swin_unetr.trainer import LitModel  # noqa: F401

    config = load_config(f"experiments/picai/{model_name}/config.yaml")
    config.data.json_list = f"json_datalists/picai/fold_{fold}.json"
    config.gpus = [device.index if device.type == "cuda" else 0]
    config.cache_rate = 0.0
    # Ensure all label keys are loaded by the DataModule transforms
    config.transforms.label_keys = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()

    print(f"Validation samples: {len(dl)}")

    processed, skipped_existing, no_cancer, errors = 0, 0, 0, 0

    for i, batch in enumerate(dl):
        # Extract case_id from the first channel's file path
        fname = Path(batch["image"].meta["filename_or_obj"][0]).name
        case_id = fname.split("_0000")[0]  # e.g. "10038_1000038"

        out_file = fold_dir / f"{case_id}.npz"

        if skip_existing and out_file.exists():
            skipped_existing += 1
            continue

        print(f"\n  [{i + 1}/{len(dl)}] {case_id}")
        print(f"    Image shape: {batch['image'].shape}")

        try:
            sal, occ, abl, inp_abl, img, pred, lbl = generate_xai(
                network=network,
                batch=batch,
                occlusion_window=occlusion_window,
                occlusion_stride=occlusion_stride,
                perturbations_per_eval=perturbations_per_eval,
                run_saliency=run_saliency,
                run_occlusion=run_occlusion,
                run_ablation_cam=run_ablation_cam,
                run_input_ablation=run_input_ablation,
            )

            # img is None only on the no-cancer early exit
            if img is None:
                no_cancer += 1
                continue

            def _sentinel(arr):
                return arr if arr is not None else np.zeros((0,), dtype=np.float32)

            np.savez_compressed(
                out_file,
                saliency=_sentinel(sal),           # (3, D_crop, H, W) or (0,)
                occlusion=_sentinel(occ),           # (3, D_crop, H, W) or (0,)
                ablation=_sentinel(abl),            # (1, D_crop, H, W) or (0,)
                input_ablation=_sentinel(inp_abl),  # (3,) weights or (0,)
                image=img,                          # (3, D_crop, H, W)
                prediction=pred,                    # (1, D_crop, H, W) cancer prob
                label=_sentinel(lbl),               # (1, D_crop, H, W) or (0,)
                channels=np.array(CHANNEL_NAMES),
                case_id=case_id,
                fold=fold,
            )
            print(f"    Saved: {out_file}  shape={img.shape}")
            processed += 1

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1

        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()

    summary = {
        "model": model_name,
        "fold": fold,
        "total_val_cases": len(dl),
        "processed": processed,
        "skipped_existing": skipped_existing,
        "no_cancer_skipped": no_cancer,
        "errors": errors,
    }
    summary_path = fold_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFold {fold} done — processed={processed}, "
          f"skipped_existing={skipped_existing}, "
          f"no_cancer={no_cancer}, errors={errors}")
    print(f"Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Captum XAI (Saliency + Occlusion + AblationCAM + "
                    "Input Ablation) for U-MambaMTL and SwinUNETR validation sets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["umamba_mtl", "swin_unetr"],
        help="Model architecture to evaluate.",
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
        help="Root output directory; model/fold-level subdirs are created automatically.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-process a case even if its .npz output already exists.",
    )
    parser.add_argument(
        "--occlusion-window",
        type=str,
        default="1,16,16,2",
        metavar="C,H,W,D",
        help=(
            "Sliding window shape for Occlusion in (channel, H, W, depth) order. "
            "Larger values = faster but coarser attribution."
        ),
    )
    parser.add_argument(
        "--occlusion-stride",
        type=str,
        default="1,8,8,1",
        metavar="C,H,W,D",
        help="Strides for the Occlusion sliding window in (channel, H, W, depth) order.",
    )
    parser.add_argument(
        "--perturbations-per-eval",
        type=int,
        default=1,
        help="Number of occlusion perturbations per forward pass (higher = faster, more VRAM).",
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
            "'ablation' runs 3D AblationCAM — slow, one forward pass per channel. "
            "'input_ablation' shuffles each input channel and measures the score drop, "
            "returning a (n_channels,) weight vector of fractional importance."
        ),
    )

    args = parser.parse_args()

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

    methods = set(args.methods)
    if "all" in methods:
        methods = {"saliency", "occlusion", "ablation", "input_ablation"}

    run_saliency = "saliency" in methods
    run_occlusion = "occlusion" in methods
    run_ablation_cam = "ablation" in methods
    run_input_ablation = "input_ablation" in methods

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:            {args.model}")
    print(f"Folds to process: {folds}")
    print(f"Output directory: {output_dir}")
    print(f"Methods:          {', '.join(sorted(methods))}")
    print(f"Occlusion window: {occ_window}  stride: {occ_stride}  (C, H, W, D)")

    for fold in folds:
        process_fold(
            fold=fold,
            model_name=args.model,
            output_dir=output_dir,
            skip_existing=not args.no_skip,
            occlusion_window=occ_window,
            occlusion_stride=occ_stride,
            perturbations_per_eval=args.perturbations_per_eval,
            run_saliency=run_saliency,
            run_occlusion=run_occlusion,
            run_ablation_cam=run_ablation_cam,
            run_input_ablation=run_input_ablation,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
