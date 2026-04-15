#!/usr/bin/env python3
"""
Re-extract zones from the (fixed) DataModule and update the 'zones' key
in existing u_mamba_mtl XAI .npz files — no model inference needed.

Run:
    uv run update_zones_umamba.py [--folds 0,1,2,3,4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "U_MambaMTL_XAI"))

UMAMBA_ROOT        = ROOT / "U_MambaMTL_XAI"
DEFAULT_OUTPUT_ZONES = ROOT / "results" / "xai" / "zones"
OUTPUT_DIR         = ROOT / "results" / "xai" / "u_mamba_mtl"


def _zones_from_monai_batch(batch: dict, d0: int, d1: int):
    """Extract zones from MONAI batch (fixed version via DataModule)."""
    if "zones" not in batch:
        return None
    z = batch["zones"][0].cpu().numpy()       # (3, H, W, D)
    zones_hwD = np.argmax(z, axis=0).astype(np.int8)   # (H, W, D)
    zones_Dhw = zones_hwD.transpose(2, 0, 1)            # (D, H, W)
    return zones_Dhw[d0:d1]                              # (D_crop, H, W)


def update_fold(fold: int) -> None:
    from shared_modules.data_module import DataModule  # noqa: E402
    from shared_modules.utils import load_config       # noqa: E402

    fold_dir  = OUTPUT_DIR / f"fold_{fold}"
    zones_dir = DEFAULT_OUTPUT_ZONES / f"fold_{fold}"

    if not fold_dir.exists():
        print(f"  [fold {fold}] Output dir not found, skipping.")
        return

    config = load_config(f"U_MambaMTL_XAI/experiments/picai/umamba_mtl/config.yaml")
    config.data.json_list        = str(UMAMBA_ROOT / f"json_datalists/picai/fold_{fold}.json")
    config.cache_rate            = 0.0
    config.transforms.label_keys = ["pca", "prostate_pred", "zones"]

    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()

    updated = 0
    skipped = 0

    for batch in dl:
        fname   = Path(batch["image"].meta["filename_or_obj"][0]).name
        case_id = fname.replace("_0000.nii.gz", "").replace(".nii.gz", "")

        out_file = fold_dir / f"{case_id}.npz"
        if not out_file.exists():
            skipped += 1
            continue

        # d0/d1 come from the stored zone-prediction file (no need to re-run the model)
        zone_pred_file = zones_dir / f"{case_id}.npz"
        if not zone_pred_file.exists():
            print(f"  [fold {fold}] {case_id}: missing zone prediction file, skipping")
            skipped += 1
            continue

        zp    = np.load(zone_pred_file)
        d0    = int(zp["d0"])
        d1    = int(zp["d1"])

        zones_crop = _zones_from_monai_batch(batch, d0, d1)  # (D_crop, H, W)
        if zones_crop is None:
            print(f"  [fold {fold}] {case_id}: no zones in batch, skipping")
            skipped += 1
            continue

        # Apply the MONAI→nnUNet axis swap (H↔W) used when saving
        zones_stored = zones_crop.transpose(0, 2, 1).astype(np.int8)  # (D_crop, W, H)

        # Re-save .npz with only 'zones' replaced; all other arrays unchanged
        old  = np.load(out_file, allow_pickle=True)
        data = dict(old)
        data["zones"] = zones_stored
        np.savez_compressed(out_file, **data)

        print(f"  [fold {fold}] Updated: {case_id}")
        updated += 1

    print(f"  [fold {fold}] Done — {updated} updated, {skipped} skipped.\n")


def main():
    parser = argparse.ArgumentParser(description="Update umamba_mtl zone maps from fixed DataModule.")
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="Comma-separated fold indices to process (default: 0,1,2,3,4)")
    args = parser.parse_args()

    folds = [int(f.strip()) for f in args.folds.split(",")]
    for fold in folds:
        print(f"=== Fold {fold} ===")
        update_fold(fold)


if __name__ == "__main__":
    main()
