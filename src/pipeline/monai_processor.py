import gc
import json
import traceback
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion, Saliency

from src.models.loader import load_model
from src.utils import (
    _AGG_FIELD_SUFFIX,
    _load_npz_fields,
    _sentinel,
    log_large_vars,
    methods_already_computed,
)
from src.xai.ablation_cam_3d import AblationCAM3D, find_decoder_feature_layers
from src.xai.forward_wrappers import _make_forward_func_sigmoid
from src.xai.occlusion import _build_baseline_tensor, _compute_zone_baseline_patches
from src.zones.zones import (
    DEFAULT_OUTPUT_ZONES,
    _gt_depth_crop,
    _load_umamba_zones,
    _zones_from_monai_batch,
)
from src.metrics.progress import _build_progress_record, _save_progress

CHANNEL_NAMES = ["t2w", "adc", "hbv"]


def process_fold_monai(
    fold: int,
    model_name: str,
    output_dir: Path,
    methods: Set[str],
    overwrite: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
    zones_only: bool = False,
    aggregation: str = "sum",
    max_cases: Optional[int] = None,
    ig_steps: int = 50,
    ig_internal_batch_size: int = 8,
) -> None:
    from shared_modules.data_module import DataModule   # noqa: E402
    from shared_modules.utils import load_config        # noqa: E402
    from src.models.loader import UMAMBA_ROOT

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

    run_saliency       = "saliency"              in methods
    run_occlusion      = "occlusion"             in methods
    run_ablation_cam   = "ablation"              in methods
    run_input_ablation = "input_ablation"        in methods
    run_ig             = "integrated_gradients"  in methods

    processed, skipped, errors = 0, 0, 0

    for i, batch in enumerate(dl):
        fname   = Path(batch["image"].meta["filename_or_obj"][0]).name
        case_id = fname.split("_0000")[0]
        out_file = fold_dir / f"{case_id}.npz"

        agg_sfx = _AGG_FIELD_SUFFIX[aggregation]
        # For non-sum: require existing .npz (base sum data must exist first).
        if aggregation != "sum":
            if not out_file.exists():
                print(f"  [{i + 1}/{len(dl)}] {case_id}: base .npz missing — skipping (run sum first)")
                skipped += 1
                continue
            if not overwrite and methods_already_computed(out_file, methods, agg_sfx):
                skipped += 1
                continue
        elif not overwrite and methods_already_computed(out_file, methods, agg_sfx):
            skipped += 1
            continue

        print(f"\n  [{i + 1}/{len(dl)}] {case_id}  shape={batch['image'].shape}  agg={aggregation}")

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
            sal_np = occ_np = occ_tz_np = occ_pz_np = occ_ch_np = abl_np = inp_abl = ig_np = None
            zone_median_baseline_np = None

            if predicted_pos:
                forward_func = _make_forward_func_sigmoid(network, fixed_mask.as_tensor(),
                                                          aggregation=aggregation)

                if run_saliency:
                    x_sal = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        sal_attr = Saliency(forward_func).attribute(x_sal, abs=True)
                    sal_np = sal_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                    del x_sal, sal_attr
                    sal_np = sal_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                    sal_np = sal_np[:, d0:d1]                       # (3, D_crop, H, W)

                if run_ig:
                    x_ig = x.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        ig_attr = IntegratedGradients(forward_func).attribute(
                            x_ig,
                            baselines=0,
                            n_steps=ig_steps,
                            internal_batch_size=ig_internal_batch_size,
                        )
                    ig_np = ig_attr.detach().cpu().numpy()[0]   # (3, H, W, D)
                    del x_ig, ig_attr
                    ig_np = ig_np.transpose(0, 3, 1, 2)          # (3, D, H, W)
                    ig_np = ig_np[:, d0:d1]                       # (3, D_crop, H, W)

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
                    forward_func_abl = _make_forward_func_sigmoid(network, fixed_mask,
                                                                  aggregation=aggregation)
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
                if aggregation == "sum":
                    base = {} if overwrite else _load_npz_fields(out_file)
                    # Metadata always written (non-destructive — these don't change between runs)
                    new_fields = dict(
                        image              = image_crop.transpose(0, 1, 3, 2).astype(np.float32),
                        prediction         = pred_crop.transpose(0, 1, 3, 2).astype(np.float32),
                        label              = _sentinel(lbl_crop.transpose(0, 1, 3, 2)).astype(np.float32) if lbl_crop is not None else _sentinel(None),
                        zones              = zones_crop.transpose(0, 2, 1).astype(np.int8) if zones_crop is not None else _sentinel(None),
                        channels           = np.array(CHANNEL_NAMES),
                        case_id            = case_id,
                        fold               = fold,
                        model              = model_name,
                        occlusion_strategy = np.array(occ_strategy),
                    )
                    # XAI maps: only write when computed — preserves existing data for unrun methods
                    if sal_np is not None:
                        new_fields["saliency"] = _sw(sal_np, 4).astype(np.float32)
                    if ig_np is not None:
                        new_fields["integrated_gradients"] = _sw(ig_np, 4).astype(np.float32)
                    if occ_np is not None:
                        new_fields["occlusion"] = _sw(occ_np, 4).astype(np.float32)
                    if occ_tz_np is not None:
                        new_fields["occlusion_tz"] = _sw(occ_tz_np, 4).astype(np.float32)
                    if occ_pz_np is not None:
                        new_fields["occlusion_pz"] = _sw(occ_pz_np, 4).astype(np.float32)
                    if occ_ch_np is not None:
                        new_fields["occlusion_ch_baseline"] = _sw(occ_ch_np, 4).astype(np.float32)
                    if abl_np is not None:
                        new_fields["ablation"] = _sw(abl_np, 4).astype(np.float32)
                    if inp_abl is not None:
                        new_fields["input_ablation"] = inp_abl
                    if zone_median_baseline_np is not None:
                        new_fields["zone_median_baseline"] = _sw(zone_median_baseline_np, 4).astype(np.float32)
                    np.savez_compressed(out_file, **{**base, **new_fields})
                else:
                    # Non-sum: append new fields to existing .npz (non-destructive)
                    existing = _load_npz_fields(out_file)
                    new_fields = {
                        f"saliency{agg_sfx}":              _sentinel(_sw(sal_np, 4)).astype(np.float32) if sal_np is not None else _sentinel(None),
                        f"integrated_gradients{agg_sfx}":  _sentinel(_sw(ig_np, 4)).astype(np.float32) if ig_np is not None else _sentinel(None),
                        f"occlusion{agg_sfx}":             _sentinel(_sw(occ_np, 4)).astype(np.float32) if occ_np is not None else _sentinel(None),
                        f"occlusion_tz{agg_sfx}":          _sentinel(_sw(occ_tz_np, 4)).astype(np.float32) if occ_tz_np is not None else _sentinel(None),
                        f"occlusion_pz{agg_sfx}":          _sentinel(_sw(occ_pz_np, 4)).astype(np.float32) if occ_pz_np is not None else _sentinel(None),
                        f"occlusion_ch_baseline{agg_sfx}": _sentinel(_sw(occ_ch_np, 4)).astype(np.float32) if occ_ch_np is not None else _sentinel(None),
                        f"ablation{agg_sfx}":              _sentinel(_sw(abl_np, 4)).astype(np.float32) if abl_np is not None else _sentinel(None),
                    }
                    np.savez_compressed(out_file, **{**existing, **new_fields})
                print(f"    Saved: {out_file}  (agg={aggregation})")
                processed += 1
                del forward_func

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np,
                occ_np=occ_np, abl_np=abl_np,
                occ_tz_np=occ_tz_np, occ_pz_np=occ_pz_np,
                ig_np=ig_np,
                pred_crop=pred_crop,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=pred_max_prob,  # MONAI uses sigmoid — identical to pred_max_prob
            )
            _save_progress(progress, progress_file)

            if max_cases is not None and processed >= max_cases:
                print(f"  Reached max_cases={max_cases} — stopping early.")
                break

            # Free large tensors/arrays to prevent RAM accumulation across cases
            # forward_func must be deleted first — its closure holds fixed_mask
            del x, out, cancer_prob, fixed_mask, batch, occ_np, occ_tz_np, occ_pz_np, occ_ch_np
            # Also drop numpy intermediates — .numpy() shares storage with MetaTensor,
            # so these prevent the MetaTensor from being freed until GC runs
            image_np = image_crop = pred_np = pred_crop = None
            lbl_np = lbl_crop = zones_crop = ig_np = None

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

    summary = {
        "model": model_name, "fold": fold,
        "total_val_cases": len(dl), "processed": processed,
        "skipped_existing": skipped, "errors": errors,
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFold {fold} done — processed={processed} skipped={skipped} errors={errors}")
