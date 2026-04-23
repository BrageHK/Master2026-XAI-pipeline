import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).parent.parent.parent.resolve()

NNUNET_ROOT         = _ROOT / "picai_nnunet"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnunet_base" / "nnUNet_preprocessed"
NNUNET_RESULTS      = NNUNET_ROOT / "results" / "nnUNet"
TASK_NAME           = "Task2203_picai_baseline"
PLANS_FILE          = NNUNET_RESULTS / "plans.pkl"

IMAGES_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/imagesTr"
)
LABELS_TR = Path(
    "/cluster/projects/vc/data/mic/open/Prostate/PI-CAI-V2.0"
    "/workdir/nnUNet_raw_data/Task2203_picai_baseline/labelsTr"
)

sys.path.insert(0, str(NNUNET_ROOT / "nnUNet"))
sys.path.insert(0, str(NNUNET_ROOT))

from src.utils import load_plans as _load_plans_from_file


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
    print(f"Total cases: loaded: ", len(result))
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
