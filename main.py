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
  python main.py --models umamba_mtl --fold 0 --methods saliency
  python main.py --models umamba_mtl swin_unetr nnunet --fold 0,1,2,3,4 --methods saliency occlusion
  python main.py --models all --fold 0 --methods all
  python main.py --models swin_unetr --compute-metrics-only
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# Outputs
DEFAULT_OUTPUT_XAI     = PROJECT_ROOT / "results" / "xai"
DEFAULT_OUTPUT_METRICS = PROJECT_ROOT / "results" / "metrics"

# nnUNet paths
NNUNET_ROOT         = PROJECT_ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_ROOT / "results" / "nnUNet"

# U-MambaMTL / SwinUNETR paths
UMAMBA_ROOT = PROJECT_ROOT / "U_MambaMTL_XAI"

# Set nnUNet env vars before any nnunet import
os.environ.setdefault("RESULTS_FOLDER",        str(NNUNET_RESULTS))
os.environ.setdefault("nnUNet_raw_data_base",  str(NNUNET_ROOT / "nnunet_base"))
os.environ.setdefault("nnUNet_preprocessed",   str(NNUNET_PREPROCESSED))

# Add subproject roots to path so internal imports work
sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))
sys.path.insert(0, str(UMAMBA_ROOT))


def process_fold(
    fold: int,
    model_name: str,
    output_dir: Path,
    methods: set,
    overwrite: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    occ_crop_hw: Optional[int] = 128,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
    aggregation: str = "sum",
    max_cases: Optional[int] = None,
    ig_steps: int = 50,
    ig_internal_batch_size: int = 8,
) -> None:
    from src.pipeline.monai_processor import process_fold_monai
    from src.pipeline.nnunet_processor import process_fold_nnunet
    from src.zones.zones import _ensure_umamba_zones

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "nnunet":
        if zone_source == "umamba_pred":
            _ensure_umamba_zones(fold, device)
        process_fold_nnunet(
            fold, model_output_dir, methods, overwrite, occ_window, occ_stride, ppe,
            occ_crop_hw, device, occ_strategy, n_zone_patches, zone_source, aggregation,
            max_cases, ig_steps, ig_internal_batch_size,
        )
    else:
        if model_name == "swin_unetr" and zone_source == "umamba_pred":
            _ensure_umamba_zones(fold, device)
        process_fold_monai(
            fold, model_name, model_output_dir, methods, overwrite, occ_window, occ_stride, ppe,
            device, occ_strategy, n_zone_patches, zone_source, aggregation=aggregation,
            max_cases=max_cases, ig_steps=ig_steps, ig_internal_batch_size=ig_internal_batch_size,
        )


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
        choices=["saliency", "occlusion", "ablation", "input_ablation", "integrated_gradients", "all"],
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
        "--overwrite",
        action="store_true",
        help="Recompute XAI and overwrite .npz even if results already exist.",
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
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Stop each fold after N predicted-positive cases are saved. Useful for quick tests.",
    )
    parser.add_argument(
        "--aggregation",
        nargs="+",
        default=["sum"],
        choices=["sum", "mean", "abs_sum", "abs_avg"],
        metavar="AGG",
        help=(
            "Forward-function aggregation method(s) for XAI gradient computation. "
            "'sum' is the existing default (backward-compatible). "
            "Non-sum methods append new fields (e.g. saliency_mean) to existing .npz files "
            "without overwriting any data. Requires the base sum .npz to exist first. "
            "Multiple values accepted: --aggregation sum mean abs_sum abs_avg"
        ),
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        metavar="N",
        help="Number of interpolation steps for Integrated Gradients (higher = more accurate, slower).",
    )
    parser.add_argument(
        "--ig-internal-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Internal batch size for Integrated Gradients forward passes.",
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
        methods = {"saliency", "occlusion", "ablation", "input_ablation", "integrated_gradients"}

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

    aggregations: List[str] = list(dict.fromkeys(args.aggregation))  # deduplicate, preserve order

    print(f"Models:            {models}")
    print(f"Folds:             {folds}")
    print(f"Methods:           {', '.join(sorted(methods))}")
    print(f"Aggregation(s):    {aggregations}")
    print(f"Output XAI dir:    {output_dir}")
    print(f"Output metrics:    {metrics_dir}")
    print(f"Occlusion window:  {occ_window}  stride: {occ_stride}")
    print(f"Occlusion strategy:{args.occlusion_strategy}  zone patches: {args.occlusion_zone_patches}")
    print(f"Zone source:       {args.zone_source}")
    print(f"IG steps:          {args.ig_steps}  internal batch size: {args.ig_internal_batch_size}")
    print(f"Device:            {device if device is not None else 'auto'}")

    from src.metrics.compute import compute_metrics
    from src.metrics.charts import generate_charts

    for model_name in models:
        if not args.compute_metrics_only:
            for agg in aggregations:
                for fold in folds:
                    process_fold(
                        fold=fold,
                        model_name=model_name,
                        output_dir=output_dir,
                        methods=methods,
                        overwrite=args.overwrite,
                        occ_window=occ_window,
                        occ_stride=occ_stride,
                        ppe=args.perturbations_per_eval,
                        occ_crop_hw=occ_crop_hw,
                        device=device,
                        occ_strategy=args.occlusion_strategy,
                        n_zone_patches=args.occlusion_zone_patches,
                        zone_source=args.zone_source,
                        aggregation=agg,
                        max_cases=args.max_cases,
                        ig_steps=args.ig_steps,
                        ig_internal_batch_size=args.ig_internal_batch_size,
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
