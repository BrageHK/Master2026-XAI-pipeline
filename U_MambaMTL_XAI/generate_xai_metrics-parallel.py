from shared_modules.data_module import DataModule
from shared_modules.utils import load_config

import torch
import torch.multiprocessing as mp
import json
from tqdm import tqdm
from captum.attr import Occlusion
from captum.attr import Saliency
from pathlib import Path
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0, help="Fold number (0-4)")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 8)")
parser.add_argument("--model", type=str, default="umamba_mtl", help="Model to evaluate metrics")
args = parser.parse_args()
fold = args.fold

if args.model == "umamba_mtl":
    from experiments.picai.umamba_mtl.trainer import LitModel
elif args.model == "swin_unetr":
    from experiments.picai.umamba_mtl.trainer import LitModel

model = args.model

OUTPUT_DIR = Path(f"xai_outputs/{model}/f{fold}/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings:
dataset = "picai"
label_key = "pca"
checkpoint_path = f"/cluster/home/bragehk/U-MambaMTL-XAI/gc_algorithms/base_container/models/{model}/weights/f{fold}.ckpt"

size = 16
sliding_window_shapes = (1, size, size, 2)

stride_size = 8
strides = (1, stride_size, stride_size, 1)

baselines = 0
perturbations_per_eval = 1


def worker_fn(rank, num_gpus, gpu_ids, model):
    gpu_id = gpu_ids[rank]
    print("GPU ID: ", gpu_id)
    torch.cuda.set_device(gpu_id)

    # Each worker loads its own model
    config = load_config(f"experiments/picai/{model}/config.yaml")
    config.data.json_list =  f"json_datalists/picai/fold_{fold}.json"
    config.gpus = [gpu_id]
    config.cache_rate = 0.0
    config.transforms.label_keys = ["pca", "prostate_pred", "zones"]

    model = LitModel.load_from_checkpoint(checkpoint_path, config=config, map_location=f"cuda:{gpu_id}", strict=False)
    model = model.eval()
    model.to(gpu_id)

    def agg_segmentation_wrapper(inp):
        model_out = model(inp)[:, 0:2, ...]  # (B, 2, H, W, D)
        pca_mask = (model_out[:, 1, ...].sigmoid() > 0.5).float()  # (B, H, W, D)
        aggregated_logits = (model_out[:, 1, ...] * pca_mask).sum(dim=(1, 2, 3))  # (B,)
        return aggregated_logits

    occlusion = Occlusion(agg_segmentation_wrapper)
    attribute_fn = Saliency(agg_segmentation_wrapper)

    # Each worker loads its own dataloader
    dm = DataModule(config=config)
    dm.setup("validation")
    dl = dm.val_dataloader()

    # Resume: load existing results and skip already-processed samples
    partial_path = OUTPUT_DIR / f"results_gpu{gpu_id}.json"
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        done_indices = {r["sample_idx"] for r in results}
        print(f"[GPU {gpu_id}] Resuming — {len(results)} samples already done, skipping them.")
    else:
        results = []
        done_indices = set()

    for sample_idx, batch in enumerate(tqdm(dl, desc=f"GPU {gpu_id} (rank {rank})", position=rank)):


        # Round-robin: only process samples assigned to this worker
        if sample_idx % num_gpus != rank:
            continue

        if sample_idx in done_indices:
            continue

        is_pca = batch["pca"].max() > 0
        case_id = batch["image"].meta["filename_or_obj"][0].split("/")[-1]

        x = batch["image"].to(gpu_id)
        print(f"Image shape: {x.shape}")
        logits = model(x)

        confidence = round(torch.sigmoid(logits)[0, 1].max().item() * 100, 2)
        predicted_positive = (torch.sigmoid(logits[:, 1]) > 0.5).any().item()

        if predicted_positive and is_pca:
            classification = "tp"
        elif predicted_positive and not is_pca:
            classification = "fp"
        elif not predicted_positive and not is_pca:
            classification = "tn"
        else:
            classification = "fn"

        pca_in_pz = int((batch["pca"] * batch["zones"][:, 1:2, ...]).sum().item())
        pca_in_tz = int((batch["pca"] * batch["zones"][:, 2:3, ...]).sum().item())
        logits_mask = (torch.sigmoid(logits[:, 1:2]) > 0.5).cpu()
        pred_pca_in_pz = int((logits_mask * batch["zones"][:, 1:2, ...]).sum().item())
        pred_pca_in_tz = int((logits_mask * batch["zones"][:, 2:3, ...]).sum().item())

        result = {
            "sample_idx": sample_idx,
            "case_id": case_id,
            "classification": classification,
            "has_pca": bool(is_pca),
            "predicted_positive": bool(predicted_positive),
            "confidence": confidence,
            "pca_voxels_in_pz": pca_in_pz,
            "pca_voxels_in_tz": pca_in_tz,
            "pred_pca_voxels_in_pz": pred_pca_in_pz,
            "pred_pca_voxels_in_tz": pred_pca_in_tz,
        }
        if pred_pca_in_pz == 0 and pred_pca_in_tz == 0:
            result["pred_pca_dominant_zone"] = None
        elif pred_pca_in_pz > pred_pca_in_tz:
            result["pred_pca_dominant_zone"] = "pz"
        elif pred_pca_in_tz > pred_pca_in_pz:
            result["pred_pca_dominant_zone"] = "tz"
        else:
            result["pred_pca_dominant_zone"] = "both"

        if pca_in_pz == 0 and pca_in_tz == 0:
            result["pca_dominant_zone"] = None
        elif pca_in_pz > pca_in_tz:
            result["pca_dominant_zone"] = "pz"
        elif pca_in_tz > pca_in_pz:
            result["pca_dominant_zone"] = "tz"
        else:
            result["pca_dominant_zone"] = "both"


        if predicted_positive:
            sample_dir = OUTPUT_DIR / f"sample_{sample_idx:04d}_{case_id}"
            sample_dir.mkdir(exist_ok=True)
            result["maps_dir"] = str(sample_dir)

            saliency_map = attribute_fn.attribute(x, abs=True)

            t0 = time.time()
            occlusion_map = occlusion.attribute(
                x,
                sliding_window_shapes=sliding_window_shapes,
                strides=strides,
                baselines=baselines,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=False
            )
            print(f"[GPU {gpu_id}] Occlusion took {time.time() - t0:.2f}s")

            torch.save(saliency_map.cpu(), sample_dir / "saliency_map.pt")
            torch.save(occlusion_map.cpu(), sample_dir / "occlusion_map.pt")

        print(f"[GPU {gpu_id}][{sample_idx}] {case_id}: {classification.upper()} | confidence={confidence}% | pz={pca_in_pz} tz={pca_in_tz} | pred: pz={pred_pca_in_pz} tz={pred_pca_in_tz}")
        results.append(result)

        # Save incrementally per worker
        with open(OUTPUT_DIR / f"results_gpu{gpu_id}.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"[GPU {gpu_id}] Done. Processed {len(results)} samples.")


def merge_results(gpu_ids):
    """Merge per-GPU result files into a single sorted results.json."""
    all_results = []
    for gpu_id in gpu_ids:
        partial_path = OUTPUT_DIR / f"results_gpu{gpu_id}.json"
        if partial_path.exists():
            with open(partial_path) as f:
                all_results.extend(json.load(f))

    # Deduplicate by sample_idx (in case of overlapping partial files)
    seen = set()
    unique_results = []
    for r in all_results:
        if r["sample_idx"] not in seen:
            seen.add(r["sample_idx"])
            unique_results.append(r)

    unique_results.sort(key=lambda r: r["sample_idx"])

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(unique_results, f, indent=2)

    print(f"\nMerged {len(unique_results)} results into {OUTPUT_DIR / 'results.json'}")
    print(f"  TP: {sum(1 for r in unique_results if r['classification'] == 'tp')}")
    print(f"  FP: {sum(1 for r in unique_results if r['classification'] == 'fp')}")
    print(f"  TN: {sum(1 for r in unique_results if r['classification'] == 'tn')}")
    print(f"  FN: {sum(1 for r in unique_results if r['classification'] == 'fn')}")


if __name__ == "__main__":
    num_gpus = len(GPUS)
    print("numb gpus: ", num_gpus)

    if num_gpus == 1:
        # Single GPU — run directly without spawning
        worker_fn(0, 1, GPUS, model)
    else:
        mp.spawn(worker_fn, args=(num_gpus, GPUS, model), nprocs=num_gpus, join=True)

    merge_results(GPUS)
