#!/usr/bin/env python3
"""
Inference script for U-MambaMTL on a single PI-CAI image.

Usage:
  python inference.py --fold 0
  python inference.py --fold 0 --case 10007_1000742
  python inference.py --fold 0 --index 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.resolve()
UMAMBA_ROOT  = PROJECT_ROOT / "U_MambaMTL_XAI"
CHECKPOINT_DIR = UMAMBA_ROOT / "gc_algorithms" / "base_container" / "models"

sys.path.insert(0, str(UMAMBA_ROOT))


def load_model(fold: int, device: torch.device) -> torch.nn.Module:
    from shared_modules.utils import load_config
    from experiments.picai.umamba_mtl.trainer import LitModel

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/umamba_mtl/config.yaml")
    config.data.json_list = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.gpus = [device.index if device.type == "cuda" else 0]
    config.cache_rate = 0.0

    ckpt_path = CHECKPOINT_DIR / "umamba_mtl" / "weights" / f"f{fold}.ckpt"
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


def load_single_case(fold: int, case_id: str | None, index: int):
    from shared_modules.data_module import DataModule
    from shared_modules.utils import load_config

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/umamba_mtl/config.yaml")
    config.data.json_list = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.gpus = [0]
    config.cache_rate = 0.0
    config.transforms.label_keys = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()

    if case_id is not None:
        for batch in dl:
            fname = Path(batch["image"].meta["filename_or_obj"][0]).name
            if case_id in fname:
                return batch
        raise ValueError(f"Case '{case_id}' not found in fold {fold} validation set.")

    # Use index
    for i, batch in enumerate(dl):
        if i == index:
            return batch

    raise IndexError(f"Index {index} out of range (fold {fold} has {len(dl)} validation samples).")


def run_inference(network: torch.nn.Module, batch: dict, device: torch.device):
    x = batch["image"].to(device)  # (1, 3, H, W, D)

    # Warm-up (ensures CUDA kernels are compiled before timing)
    if device.type == "cuda":
        with torch.no_grad():
            _ = network(x)
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        out = network(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if isinstance(out, (list, tuple)):
        out = out[0]

    cancer_prob   = torch.sigmoid(out[:, 1])        # (1, H, W, D)
    cancer_voxels = int((cancer_prob > 0.5).sum())
    max_prob      = float(cancer_prob.max())
    predicted_pos = cancer_voxels > 0

    return elapsed, predicted_pos, cancer_voxels, max_prob


def main():
    parser = argparse.ArgumentParser(description="U-MambaMTL inference on a single PI-CAI case.")
    parser.add_argument("--fold",  type=int, default=0, help="Model fold (0-4)")
    parser.add_argument("--case",  type=str, default=None, help="Case ID (e.g. 10007_1000742)")
    parser.add_argument("--index", type=int, default=0, help="Case index in validation set (used if --case is not given)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    print(f"Loading umamba_mtl fold {args.fold}...")
    t0 = time.perf_counter()
    network = load_model(args.fold, device)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    # ---- Load case ----
    print(f"Loading case (fold={args.fold}, case={args.case!r}, index={args.index})...")
    batch = load_single_case(args.fold, args.case, args.index)

    fname   = Path(batch["image"].meta["filename_or_obj"][0]).name
    case_id = fname.split("_0000")[0]
    shape   = tuple(batch["image"].shape)
    print(f"Case: {case_id}  |  Input shape: {shape}")

    # ---- Inference ----
    elapsed, predicted_pos, cancer_voxels, max_prob = run_inference(network, batch, device)

    print()
    print("=" * 50)
    print(f"Case ID         : {case_id}")
    print(f"Inference time  : {elapsed * 1000:.1f} ms")
    print(f"Predicted pos   : {predicted_pos}")
    print(f"Cancer voxels   : {cancer_voxels}")
    print(f"Max cancer prob : {max_prob:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
