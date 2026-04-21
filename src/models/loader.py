import importlib.util
import os
import pickle
import sys
import tempfile
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent.parent.resolve()

NNUNET_ROOT         = _ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_ROOT / "results" / "nnUNet"
TASK_NAME           = "Task2203_picai_baseline"
DATASET_DIR         = NNUNET_PREPROCESSED / TASK_NAME
PLANS_FILE          = NNUNET_RESULTS / "plans.pkl"

UMAMBA_ROOT    = _ROOT / "U_MambaMTL_XAI"
CHECKPOINT_DIR = UMAMBA_ROOT / "gc_algorithms" / "base_container" / "models"

os.environ.setdefault("RESULTS_FOLDER",        str(NNUNET_RESULTS))
os.environ.setdefault("nnUNet_raw_data_base",  str(NNUNET_ROOT / "nnunet_base"))
os.environ.setdefault("nnUNet_preprocessed",   str(NNUNET_PREPROCESSED))

sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))
sys.path.insert(0, str(UMAMBA_ROOT))


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
