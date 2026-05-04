import gc
import json
import traceback
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
from captum.attr import GradientShap, IntegratedGradients, Occlusion, Saliency

from src.models.loader import load_nnunet
from src.models.preprocessing import (
    _compute_nnunet_divisors,
    _load_label_nnunet,
    _preprocess_nnunet,
    load_plans,
    load_splits_nnunet,
)
from src.utils import (
    _AGG_FIELD_SUFFIX,
    _load_npz_fields,
    _pad,
    _sentinel,
    _unpad,
    log_large_vars,
    methods_already_computed,
)
from src.xai.ablation_cam_3d import AblationCAM3D
from src.xai.forward_wrappers import _make_forward_func_softmax
from src.xai.occlusion import _build_baseline_tensor, _compute_zone_baseline_patches
from src.zones.zones import _zones_from_nnunet, _zones_from_umamba_npz
from src.metrics.progress import _build_progress_record, _save_progress

CHANNEL_NAMES = ["t2w", "adc", "hbv"]


def process_fold_nnunet(
    fold: int,
    output_dir: Path,
    methods: Set[str],
    overwrite: bool,
    occ_window: Tuple[int, int, int, int],
    occ_stride: Tuple[int, int, int, int],
    ppe: int,
    occ_crop_hw: int = 128,
    device: Optional[torch.device] = None,
    occ_strategy: str = "zero",
    n_zone_patches: int = 10,
    zone_source: str = "umamba_pred",
    aggregation: str = "sum",
    max_cases: Optional[int] = None,
    ig_steps: int = 50,
    ig_internal_batch_size: int = 8,
    gs_n_samples: int = 50,
    gs_stdevs: float = 0.0,
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

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = load_nnunet(fold, device)
    print(f"Model ready on {device}.")

    plans     = load_plans()
    splits    = load_splits_nnunet()
    if fold not in splits:
        print(f"WARNING: fold {fold} not in valid_splits. Skipping.")
        return
    val_cases: List[str] = splits[fold]["subject_list"]
    print(f"Validation cases: {len(val_cases)}")

    d_div, h_div, w_div = _compute_nnunet_divisors(plans)
    run_saliency        = "saliency"              in methods
    run_occlusion       = "occlusion"             in methods
    run_ablation_cam    = "ablation"              in methods
    run_input_ablation  = "input_ablation"        in methods
    run_ig              = "integrated_gradients"  in methods
    run_gs              = "gradient_shap"         in methods

    processed, skipped, errors = 0, 0, 0

    for i, case_id in enumerate(val_cases):
        out_file = fold_dir / f"{case_id}.npz"
        agg_sfx = _AGG_FIELD_SUFFIX[aggregation]

        # For non-sum: require existing .npz (base sum data must exist first).
        if aggregation != "sum":
            if not out_file.exists():
                print(f"  [{i + 1}/{len(val_cases)}] {case_id}: base .npz missing — skipping (run sum first)")
                skipped += 1
                continue
            if not overwrite and methods_already_computed(out_file, methods, agg_sfx):
                skipped += 1
                continue
        elif not overwrite and methods_already_computed(out_file, methods, agg_sfx):
            skipped += 1
            continue

        print(f"\n  [{i + 1}/{len(val_cases)}] {case_id}  agg={aggregation}")

        try:
            # ---- Preprocess -----------------------------------------------
            data, prep_props = _preprocess_nnunet(case_id, plans)
            print(f"    Preprocessed shape: {data.shape}")
            original_dhw: Tuple[int, int, int] = tuple(data.shape[1:])  # type: ignore

            label_np = _load_label_nnunet(case_id, plans, prep_props)
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
                confidence    = pred_max_prob  # softmax already applied above
                predicted_pos = cancer_voxels > 0

            print(f"    Predicted positive: {predicted_pos}  (cancer voxels={cancer_voxels}  max_prob={pred_max_prob:.3f})")

            # ---- Load zones (full) to determine crop coordinates ----------
            D_orig = original_dhw[0]
            if zone_source == "umamba_pred":
                zones_full = _zones_from_umamba_npz(
                    case_id, fold, plans, prep_props, 0, D_orig, 0, 0, None
                )
            else:
                zones_full = _zones_from_nnunet(
                    case_id, plans, prep_props, 0, D_orig, 0, 0, None
                )

            # ---- Crop coordinates based on prostate zones -----------------
            if zones_full is not None and zones_full.any():
                coords = np.argwhere(zones_full > 0)   # (N, 3): (d, h, w)
                d_min  = int(coords[:, 0].min())
                d_max  = int(coords[:, 0].max())
                d0     = max(0, d_min - 1)
                d1     = min(D_orig, d_max + 2)
                _, hc, wc = coords.mean(axis=0).astype(int)
                h0 = int(np.clip(hc - occ_crop_hw // 2, 0, H_pad - occ_crop_hw))
                w0 = int(np.clip(wc - occ_crop_hw // 2, 0, W_pad - occ_crop_hw))
            else:
                d0, d1 = 0, D_orig
                h0, w0 = 0, 0

            print(f"    Depth crop: d={d0}:{d1}  h0={h0}  w0={w0}  zone_source={zone_source}")

            # ---- Always: image, prediction, label, zones ------------------
            prob_full  = _unpad(cancer_prob.cpu().numpy(), original_dhw)  # (1, D, H, W)
            image_crop = data[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
            pred_crop  = prob_full[:, d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
            lbl_crop: Optional[np.ndarray] = label_np[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw][np.newaxis] if label_np is not None else None
            zones_crop = zones_full[d0:d1, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw] if zones_full is not None else None

            # ---- XAI: only when predicted positive ------------------------
            sal_np = occ_np = occ_tz_np = occ_pz_np = occ_ch_np = abl_np = inp_abl = ig_np = gs_np = None
            zone_median_baseline_np = None

            if predicted_pos:
                fixed_mask_crop = fixed_mask[:, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                x_crop = x.detach().clone()[:, :, :, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]

                fwd_occ = _make_forward_func_softmax(network, fixed_mask_crop, aggregation=aggregation)

                if run_saliency:
                    x_sal = x_crop.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        sal_attr = Saliency(fwd_occ).attribute(x_sal, abs="abs" in aggregation)
                    sal_np = _unpad(sal_attr.detach().cpu().numpy()[0], (original_dhw[0], occ_crop_hw, occ_crop_hw))
                    del x_sal, sal_attr
                    sal_np = sal_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                if run_ig:
                    x_ig = x_crop.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        ig_attr = IntegratedGradients(fwd_occ).attribute(
                            x_ig,
                            baselines=0,
                            n_steps=ig_steps,
                            internal_batch_size=ig_internal_batch_size,
                        )
                    ig_np = _unpad(ig_attr.detach().cpu().numpy()[0], (original_dhw[0], occ_crop_hw, occ_crop_hw))
                    del x_ig, ig_attr
                    ig_np = ig_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                if run_gs:
                    x_gs = x_crop.detach().clone().requires_grad_(True)
                    baselines_gs = torch.zeros_like(x_gs)
                    try:
                        with torch.enable_grad():
                            gs_attr = GradientShap(fwd_occ).attribute(
                                x_gs,
                                baselines=baselines_gs,
                                n_samples=gs_n_samples,
                                stdevs=gs_stdevs,
                            )
                        gs_np = _unpad(gs_attr.detach().cpu().numpy()[0], (original_dhw[0], occ_crop_hw, occ_crop_hw))
                        del x_gs, gs_attr, baselines_gs
                        gs_np = gs_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)
                    except torch.OutOfMemoryError:
                        print(f"    [gs] OOM for {case_id} (shape {list(x_gs.shape)}) — skipping GradientShap.")
                        del x_gs, baselines_gs
                        torch.cuda.empty_cache()

                if run_occlusion:
                    # --- zone_median baseline (TZ + PZ) ---
                    if occ_strategy in ("zone_median", "all") and zones_full is not None:
                        # Build zones in x_crop coordinate system: (D_pad, H_crop, W_crop)
                        zones_xcrop = zones_full[:, h0:h0 + occ_crop_hw, w0:w0 + occ_crop_hw]
                        D_orig_z = zones_xcrop.shape[0]
                        D_pad_val = x_crop.shape[2]
                        if D_pad_val > D_orig_z:
                            zones_xcrop_padded = np.pad(
                                zones_xcrop, ((0, D_pad_val - D_orig_z), (0, 0), (0, 0)),
                                mode="constant", constant_values=0,
                            )
                        else:
                            zones_xcrop_padded = zones_xcrop[:D_pad_val]
                        # Compute medians from unpadded region only
                        image_xcrop_np = x_crop[0].cpu().numpy()       # (3, D_pad, H_crop, W_crop)
                        cancer_xcrop   = fixed_mask_crop[0].cpu().numpy()  # (D_pad, H_crop, W_crop)
                        occ_win_dhw = (occ_window[1], occ_window[2], occ_window[3])
                        tz_patch, pz_patch = _compute_zone_baseline_patches(
                            image_xcrop_np[:, :D_orig_z],
                            zones_xcrop_padded[:D_orig_z],
                            cancer_xcrop[:D_orig_z],
                            occ_win_dhw,
                            n_zone_patches,
                        )
                        print(f"    [zone_median] TZ patch sum={tz_patch.sum():.3f}  PZ patch sum={pz_patch.sum():.3f}")
                        # Build combined baseline image (DHW) by tiling patches over zones_crop
                        if zones_crop is not None:
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
                        zero_patch = np.zeros_like(tz_patch)
                        occ_baseline_tz = _build_baseline_tensor(
                            zones_xcrop_padded, tz_patch, zero_patch, x_crop.shape, "dhw", device
                        )
                        occ_baseline_pz = _build_baseline_tensor(
                            zones_xcrop_padded, zero_patch, pz_patch, x_crop.shape, "dhw", device
                        )
                        del zones_xcrop, zones_xcrop_padded, image_xcrop_np, cancer_xcrop, zero_patch

                        with torch.no_grad():
                            occ_attr_tz = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_tz,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_tz_np = _unpad(occ_attr_tz.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr_tz, occ_baseline_tz
                        occ_tz_np = occ_tz_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                        with torch.no_grad():
                            occ_attr_pz = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_pz,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_pz_np = _unpad(occ_attr_pz.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr_pz, occ_baseline_pz
                        occ_pz_np = occ_pz_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)
                    elif occ_strategy == "zone_median" and zones_full is None:
                        print("    [zone_median] WARNING: zones_full is None — skipping zone_median baseline.")

                    # --- zero baseline ---
                    if occ_strategy in ("zero", "all") or (occ_strategy == "zone_median" and zones_full is None):
                        with torch.no_grad():
                            occ_attr = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=0.0,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_np = _unpad(occ_attr.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr
                        occ_np = occ_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

                    # --- channel_baseline (T2W/ADC=1, HBV=0) ---
                    if occ_strategy in ("channel_baseline", "all"):
                        # nnUNet layout: (1, C, D, H, W); channels 0-1 → 1.0, channel 2 → 0.0
                        occ_baseline_ch = torch.ones_like(x_crop)
                        occ_baseline_ch[:, 2] = 0.0
                        with torch.no_grad():
                            occ_attr = Occlusion(fwd_occ).attribute(
                                x_crop,
                                sliding_window_shapes=occ_window,
                                strides=occ_stride,
                                baselines=occ_baseline_ch,
                                perturbations_per_eval=ppe,
                                show_progress=True,
                            )
                        occ_ch_np = _unpad(occ_attr.detach().cpu().numpy()[0], (original_dhw[0], x_crop.shape[3], x_crop.shape[4]))
                        del occ_attr, occ_baseline_ch
                        occ_ch_np = occ_ch_np[:, d0:d1]  # (3, D_crop, H_crop, W_crop)

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
                            cancer_prob = output[1]  # already softmaxed by network
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
                if aggregation == "sum":
                    base = {} if overwrite else _load_npz_fields(out_file)
                    # Metadata always written (non-destructive — these don't change between runs)
                    new_fields = dict(
                        image              = image_crop.astype(np.float32),
                        prediction         = pred_crop.astype(np.float32),
                        label              = lbl_crop.astype(np.float32) if lbl_crop is not None else _sentinel(None),
                        zones              = zones_crop.astype(np.int8) if zones_crop is not None else _sentinel(None),
                        channels           = np.array(CHANNEL_NAMES),
                        case_id            = case_id,
                        fold               = fold,
                        model              = "nnunet",
                        occlusion_strategy = np.array(occ_strategy),
                    )
                    # XAI maps: only write when computed — preserves existing data for unrun methods
                    if sal_np is not None:
                        new_fields["saliency"] = sal_np.astype(np.float32)
                    if ig_np is not None:
                        new_fields["integrated_gradients"] = ig_np.astype(np.float32)
                    if gs_np is not None:
                        new_fields["gradient_shap"] = gs_np.astype(np.float32)
                    if occ_np is not None:
                        new_fields["occlusion"] = occ_np.astype(np.float32)
                    if occ_tz_np is not None:
                        new_fields["occlusion_tz"] = occ_tz_np.astype(np.float32)
                    if occ_pz_np is not None:
                        new_fields["occlusion_pz"] = occ_pz_np.astype(np.float32)
                    if occ_ch_np is not None:
                        new_fields["occlusion_ch_baseline"] = occ_ch_np.astype(np.float32)
                    if abl_np is not None:
                        new_fields["ablation"] = abl_np.astype(np.float32)
                    if inp_abl is not None:
                        new_fields["input_ablation"] = inp_abl
                    if zone_median_baseline_np is not None:
                        new_fields["zone_median_baseline"] = zone_median_baseline_np.astype(np.float32)
                    np.savez_compressed(out_file, **{**base, **new_fields})
                else:
                    # Non-sum: append new fields to existing .npz (non-destructive)
                    existing = _load_npz_fields(out_file)
                    new_fields = {
                        f"saliency{agg_sfx}":              _sentinel(sal_np).astype(np.float32) if sal_np is not None else _sentinel(None),
                        f"integrated_gradients{agg_sfx}":  _sentinel(ig_np).astype(np.float32) if ig_np is not None else _sentinel(None),
                        f"gradient_shap{agg_sfx}":         _sentinel(gs_np).astype(np.float32) if gs_np is not None else _sentinel(None),
                        f"occlusion{agg_sfx}":             _sentinel(occ_np).astype(np.float32) if occ_np is not None else _sentinel(None),
                        f"occlusion_tz{agg_sfx}":          _sentinel(occ_tz_np).astype(np.float32) if occ_tz_np is not None else _sentinel(None),
                        f"occlusion_pz{agg_sfx}":          _sentinel(occ_pz_np).astype(np.float32) if occ_pz_np is not None else _sentinel(None),
                        f"occlusion_ch_baseline{agg_sfx}": _sentinel(occ_ch_np).astype(np.float32) if occ_ch_np is not None else _sentinel(None),
                        f"ablation{agg_sfx}":              _sentinel(abl_np).astype(np.float32) if abl_np is not None else _sentinel(None),
                    }
                    np.savez_compressed(out_file, **{**existing, **new_fields})
                print(f"    Saved: {out_file}  (agg={aggregation})")
                processed += 1

            # ---- Record progress ------------------------------------------
            progress[case_id] = _build_progress_record(
                predicted_pos, lbl_crop, zones_crop, sal_np,
                occ_np=occ_np, abl_np=abl_np,
                occ_tz_np=occ_tz_np, occ_pz_np=occ_pz_np,
                ig_np=ig_np, gs_np=gs_np,
                pred_crop=pred_crop,
                pred_cancer_voxels=cancer_voxels, pred_max_prob=pred_max_prob,
                confidence=confidence,
            )
            _save_progress(progress, progress_file)

            if max_cases is not None and processed >= max_cases:
                print(f"  Reached max_cases={max_cases} — stopping early.")
                del x, out, cancer_prob, fixed_mask
                break

            # Free large tensors to prevent RAM accumulation across cases
            del x, out, cancer_prob, fixed_mask

        except Exception as exc:
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors += 1
            progress[case_id] = {"done": False, "error": str(exc), "predicted_pos": None}
            _save_progress(progress, progress_file)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "model": "nnunet", "fold": fold,
        "total_val_cases": len(val_cases), "processed": processed,
        "skipped_existing": skipped, "errors": errors,
    }
    with open(fold_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFold {fold} done — processed={processed} skipped={skipped} errors={errors}")
