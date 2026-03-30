from shared_modules.data_module import DataModule
from shared_modules.utils import load_config
from shared_modules.plotting import slice_comparison_multi

from trainer import LitModel
import torch
from tqdm import tqdm
from monai.transforms import ScaleIntensityRangePercentiles
from captum.attr import Occlusion
from captum.attr import Saliency
from captum.metrics import infidelity, sensitivity_max
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayer


class SemanticSegmentationTarget3D:
    """Target for 3D semantic segmentation - computes weighted sum over a mask region."""
    def __init__(self, category, mask, device="cuda"):
        self.category = category
        self.mask = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        if device == "cuda" and torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # model_output shape: (B, C, H, W, D) or (C, H, W, D)
        if model_output.dim() == 5:
            return (model_output[:, self.category, :, :, :] * self.mask).sum()
        else:
            return (model_output[self.category, :, :, :] * self.mask).sum()


class ModelWrapper3D(torch.nn.Module):
    """Wrapper to handle 3D->2D->3D conversion for grad-cam compatibility."""
    def __init__(self, model, slice_dim=-1):
        super().__init__()
        self.model = model
        self.slice_dim = slice_dim
        self._current_slice = None
        self._full_input_shape = None

    def set_slice(self, slice_idx, full_shape):
        self._current_slice = slice_idx
        self._full_input_shape = full_shape

    def forward(self, x):
        # x is 2D: (B, C, H, W) - expand to 3D for model
        if self._current_slice is not None and x.dim() == 4:
            # Create a zero tensor with full 3D shape
            B, C, H, W = x.shape
            D = self._full_input_shape[-1]
            x_3d = torch.zeros(B, C, H, W, D, device=x.device, dtype=x.dtype)
            x_3d[:, :, :, :, self._current_slice] = x
            out = self.model(x_3d)
            # Return only the slice we care about
            return out[:, :, :, :, self._current_slice]
        return self.model(x)


def compute_ablation_cam_3d(
    model,
    input_tensor,
    target_category,
    target_mask=None,
    target_layers=None,
    device="cuda",
    batch_size=16,
    ratio_channels_to_ablate=0.5
):
    """
    Compute AblationCAM for 3D medical segmentation models.

    Since pytorch-grad-cam is 2D-oriented, this function processes
    the volume slice-by-slice and aggregates the results.

    Args:
        model: The segmentation model (LitModel)
        input_tensor: Input tensor of shape (B, C, H, W, D)
        target_category: Class index to explain (e.g., 1 for PCa)
        target_mask: Optional mask to focus attribution (H, W, D) or None for full volume
        target_layers: List of layers to use for CAM. If None, uses encoder last stage
        device: Device to run on
        batch_size: Batch size for ablation perturbations
        ratio_channels_to_ablate: Fraction of channels to ablate per forward pass

    Returns:
        cam_volume: Attribution map of shape (H, W, D)
    """
    model.eval()

    # Get the underlying model (handle LitModel wrapper)
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    # Default target layers: last encoder stage and mamba layer
    if target_layers is None:
        target_layers = [base_model.encoder.stages[-1]]

    B, C, H, W, D = input_tensor.shape

    # Create mask if not provided (use prediction as mask)
    if target_mask is None:
        with torch.no_grad():
            output = model(input_tensor)
            pred = (torch.sigmoid(output[:, target_category]) > 0.5).float()
            target_mask = pred[0].cpu().numpy()

    # Process slice by slice for compatibility
    cam_slices = []

    for slice_idx in tqdm(range(D), desc="Computing AblationCAM per slice"):
        # Extract 2D slice
        slice_2d = input_tensor[:, :, :, :, slice_idx]  # (B, C, H, W)
        mask_2d = target_mask[:, :, slice_idx] if target_mask.ndim == 3 else target_mask

        # Skip slices with no mask content
        if isinstance(mask_2d, np.ndarray):
            mask_sum = mask_2d.sum()
        else:
            mask_sum = mask_2d.sum().item()

        if mask_sum < 1:
            cam_slices.append(np.zeros((H, W), dtype=np.float32))
            continue

        # Create wrapper for this slice
        wrapper = ModelWrapper3D(model, slice_dim=-1)
        wrapper.set_slice(slice_idx, input_tensor.shape)
        wrapper.to(device)

        # Get target layers from wrapper
        if hasattr(wrapper.model, 'model'):
            wrapper_target_layers = [wrapper.model.model.encoder.stages[-1]]
        else:
            wrapper_target_layers = [wrapper.model.encoder.stages[-1]]

        # Create 2D target
        class Target2D:
            def __init__(self, category, mask_2d):
                self.category = category
                self.mask = torch.from_numpy(mask_2d).float().to(device) if isinstance(mask_2d, np.ndarray) else mask_2d.float().to(device)

            def __call__(self, output):
                # output: (B, C, H, W)
                return (output[:, self.category] * self.mask).sum()

        target = Target2D(target_category, mask_2d)

        try:
            with AblationCAM(
                model=wrapper,
                target_layers=wrapper_target_layers,
                batch_size=batch_size,
                ratio_channels_to_ablate=ratio_channels_to_ablate
            ) as cam:
                grayscale_cam = cam(
                    input_tensor=slice_2d,
                    targets=[target]
                )[0]  # (H, W)
            cam_slices.append(grayscale_cam)
        except Exception as e:
            print(f"Warning: AblationCAM failed for slice {slice_idx}: {e}")
            cam_slices.append(np.zeros((H, W), dtype=np.float32))

    # Stack slices into volume
    cam_volume = np.stack(cam_slices, axis=-1)  # (H, W, D)

    return cam_volume


def compute_ablation_cam_3d_direct(
    model,
    input_tensor,
    target_category,
    target_mask=None,
    ablation_size=(16, 16, 4),
    stride=None,
    device="cuda"
):
    """
    Compute AblationCAM directly on 3D volume using spatial ablation.

    This is a direct 3D implementation that ablates spatial regions
    rather than channels, similar to occlusion sensitivity but using
    the ablation principle.

    Args:
        model: The segmentation model
        input_tensor: Input tensor of shape (B, C, H, W, D)
        target_category: Class index to explain
        target_mask: Optional mask to focus attribution
        ablation_size: Size of ablation window (h, w, d)
        stride: Stride for ablation window. If None, uses ablation_size
        device: Device to run on

    Returns:
        cam_volume: Attribution map of shape (H, W, D)
    """
    model.eval()

    if stride is None:
        stride = ablation_size

    B, C, H, W, D = input_tensor.shape
    ah, aw, ad = ablation_size
    sh, sw, sd = stride if isinstance(stride, tuple) else (stride, stride, stride)

    # Get baseline output
    with torch.no_grad():
        baseline_output = model(input_tensor)
        if target_mask is not None:
            mask_tensor = torch.from_numpy(target_mask).float().to(device) if isinstance(target_mask, np.ndarray) else target_mask.float().to(device)
            baseline_score = (torch.sigmoid(baseline_output[:, target_category]) * mask_tensor).sum().item()
        else:
            baseline_score = torch.sigmoid(baseline_output[:, target_category]).sum().item()

    # Initialize attribution map
    attribution = np.zeros((H, W, D), dtype=np.float32)
    counts = np.zeros((H, W, D), dtype=np.float32)

    # Compute number of windows
    n_h = (H - ah) // sh + 1
    n_w = (W - aw) // sw + 1
    n_d = (D - ad) // sd + 1

    total_windows = n_h * n_w * n_d

    with torch.no_grad():
        for i, h_start in enumerate(tqdm(range(0, H - ah + 1, sh), desc="AblationCAM 3D")):
            for w_start in range(0, W - aw + 1, sw):
                for d_start in range(0, D - ad + 1, sd):
                    # Create ablated input (set region to zero or mean)
                    ablated_input = input_tensor.clone()
                    ablated_input[:, :, h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] = 0

                    # Get ablated output
                    ablated_output = model(ablated_input)
                    if target_mask is not None:
                        ablated_score = (torch.sigmoid(ablated_output[:, target_category]) * mask_tensor).sum().item()
                    else:
                        ablated_score = torch.sigmoid(ablated_output[:, target_category]).sum().item()

                    # Attribution = drop in score (positive means important)
                    importance = baseline_score - ablated_score

                    # Add to attribution map
                    attribution[h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] += importance
                    counts[h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1)
    attribution = attribution / counts

    return attribution

def perturb_fn(inputs):
    """
    Perturbation function for infidelity metric.
    Adds Gaussian noise to inputs.
    """
    noise = torch.randn_like(inputs) * 0.1
    return noise, inputs - noise


def compute_xai_metrics(model_fn, inputs, attributions, target, n_perturb_samples=10):
    """
    Compute Infidelity and Sensitivity metrics for attribution maps.

    Args:
        model_fn: The model forward function
        inputs: Input tensor
        attributions: Attribution map from explainability method
        target: Target class index
        n_perturb_samples: Number of perturbation samples for sensitivity

    Returns:
        dict with 'infidelity' and 'sensitivity' scores
    """
    # Compute Infidelity - measures how well attributions explain model predictions
    # Lower is better
    infid_score = infidelity(
        model_fn,
        perturb_fn,
        inputs,
        attributions,
        target=target,
        normalize=True
    )

    # Compute Sensitivity - measures attribution stability under input perturbations
    # Lower is better (more stable)
    sens_score = sensitivity_max(
        lambda x: Saliency(model_fn).attribute(x, target=target, abs=True),
        inputs,
        perturb_radius=0.02,
        n_perturb_samples=n_perturb_samples
    )

    return {
        'infidelity': infid_score.item() if infid_score.numel() == 1 else infid_score.mean().item(),
        'sensitivity': sens_score.item() if sens_score.numel() == 1 else sens_score.mean().item()
    }


# Output folder for saving images
OUTPUT_DIR = Path("xai_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# Settings:
SAVE_PREDS=False
SAVE_PROB_MAPS=False
dataset="picai" 
label_key = "pca"
config = load_config("config.yaml")
gpu = 0
config.gpus = [gpu]
config.cache_rate = 1.0
config.transforms.label_keys = ["pca", "prostate_pred", "zones"]
checkpoint_path = "/cluster/home/bragehk/U-MambaMTL-XAI/gc_algorithms/base_container/models/umamba_mtl/weights/f0.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path, config=config)

# disable randomness, dropout, etc...
model = model.eval()
model.to(gpu)

dm = DataModule(
    config=config,
    debug_index=13
)

print("Setting up dataloader")
dm.setup("debug")
dl = dm.debug_dataloader()


def agg_segmentation_wrapper(inp):
    # Get the logits from the model
    model_out = model(inp)  

    # Find the predicted class for each pixel 
    # (argmax along the channel dimension)
    out_max = model_out.argmax(dim=1, keepdim=True)

    # Create a binary mask with 1 for the predicted class and 0 otherwise
    # selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
    selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)

    # Multiply the logits with the mask and sum across spatial dimensions
    # Sum over x, y, and z
    aggregated_logits = (model_out * selected_inds).sum(dim=(2, 3, 4))  
    # aggregated_logits = (model_out).sum(dim=(2, 3, 4))  

    return aggregated_logits


occlusion = Occlusion(agg_segmentation_wrapper)
attribute_fn = Saliency(agg_segmentation_wrapper)

size = 64

sliding_window_shapes = (1, size, size, 5)  # Occlude one channel at a time
strides = (1, size, size, 5)               # Larger stride = fewer windows = faster
baselines = 0                          # Value to fill the occluded areas (0 is common)
perturbations_per_eval = 8             # Batch this many perturbations per forward pass (adjust based on GPU memory)

tz = 0
pz = 0

logits = None

for batch in tqdm(dl):
    if batch["pca"].max() == 0:
        print("Not any PCa here!")
        import sys
        sys.exit(0)
        continue
    
    print("To gpu")
    x = batch["image"].to(gpu)
    print("Model inference")
    logits = model(x)
    print("Check if False negative")
    if (torch.sigmoid(logits[:, 1]) > 0.5).any().item():
        import time
        t0 = time.time()
        attention_map = attribute_fn.attribute(x, target=1, abs=True)
        print(f"Saliency took {time.time() - t0:.2f}s")
        t0 = time.time()
        occlusion_map = occlusion.attribute(
            x,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=1,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=True
        )
        print(f"Occlusion took {time.time() - t0:.2f}s")

        # Compute XAI evaluation metrics
        print("Computing XAI evaluation metrics...")
        t0 = time.time()

        # Infidelity for Saliency
        saliency_infidelity = infidelity(
            agg_segmentation_wrapper,
            perturb_fn,
            x,
            attention_map,
            target=1,
            normalize=True
        )
        print(f"Saliency Infidelity: {saliency_infidelity.mean().item():.6f}")

        # Infidelity for Occlusion
        occlusion_infidelity = infidelity(
            agg_segmentation_wrapper,
            perturb_fn,
            x,
            occlusion_map,
            target=1,
            normalize=True
        )
        print(f"Occlusion Infidelity: {occlusion_infidelity.mean().item():.6f}")

        # Sensitivity for Saliency
        saliency_sensitivity = sensitivity_max(
            lambda inp: Saliency(agg_segmentation_wrapper).attribute(inp, target=1, abs=True),
            x,
            perturb_radius=0.02,
            n_perturb_samples=5
        )
        print(f"Saliency Sensitivity: {saliency_sensitivity.max().item():.6f}")

        print(f"XAI metrics took {time.time() - t0:.2f}s")

        # Store metrics for later
        xai_metrics = {
            'saliency_infidelity': saliency_infidelity.mean().item(),
            'occlusion_infidelity': occlusion_infidelity.mean().item(),
            'saliency_sensitivity': saliency_sensitivity.max().item(),
        }
        print("True positive")
    else:
        print("False negative..")
        print((torch.sigmoid(logits) > 0.5).int()[0][1][None].to("cpu")[0,1].max().item())
        import sys
        sys.exit(0)

        
    pca_in_pz = (batch["pca"] * batch["zones"] == 1).sum()
    pca_in_tz = (batch["pca"] * batch["zones"] == 2).sum()
    
    break

def normalize_and_clamp(x):
    for i in range(3):
        # print((x[i] > 0).sum())
        # print((x[i] < 0).sum())
        # x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=0.001, max=0.999)
        # x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=-0.999, max=0.999)
        x[i] = x[i] + 0.5 - x[i].mean()
        # if i == 2:
        #     x[i] = x[i] *2
        # x[i] = torch.clamp(x[i] - x[i].mean(), min=0.001, max=0.999)
        # print((x[i] > 0).sum())
        # print((x[i] < 0).sum())
    return x



img = batch["image"][0]
gt = batch["pca"][0]
pred = (torch.sigmoid(logits) > 0.5).int()[0][1][None].to("cpu")
logit = logits[0][1][None].to("cpu")

print(f"gt shape: {gt.shape}, max: {gt.max()}, min: {gt.min()}")
print(f"pred shape: {pred.shape}, max: {pred.max()}, min: {pred.min()}")

activation = attention_map[0].to("cpu")

occlusion = occlusion_map[0].to("cpu")

occ = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(occlusion)
acc = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(activation)
print("Activation map")
print("max", acc.max())
print("min", acc.min())
acc = normalize_and_clamp(acc)
print("Acc After normalization")
print("max", acc.max())
print("min", acc.min())

print("Occlusion map")
print("max", occ.max())
print("min", occ.min())
occ = normalize_and_clamp(occ)
print("occ After normalization")
print("max", occ.max())
print("min", occ.min())

# Find slice with most label pixels (minimum 10 pixels)
min_pixels = 10
slice_pixel_counts = gt[0].sum(dim=(0, 1))  # Sum over H, W for each slice
best_slice_idx = slice_pixel_counts.argmax().item()
if slice_pixel_counts[best_slice_idx] < min_pixels:
    print(f"Warning: Best slice only has {slice_pixel_counts[best_slice_idx]} label pixels")
print(f"Using slice {best_slice_idx} with {slice_pixel_counts[best_slice_idx].item()} label pixels")

print(f"More pca in pz: {pca_in_pz > pca_in_tz}")
print(f"pca in tz: {pca_in_tz}")
print(f"pca in pz: {pca_in_pz}")
case_id = batch["image"].meta["filename_or_obj"][0].split("/")[-1]
print(f"Case ID: {case_id}")
confidence = round(torch.sigmoid(logits)[0,1].max().item() * 100, 2)
print(f"PCa confidence: {confidence}%")

# Save XAI metrics to JSON file
xai_metrics['case_id'] = case_id
xai_metrics['confidence'] = confidence
with open(OUTPUT_DIR / f"{case_id}_xai_metrics.json", "w") as f:
    json.dump(xai_metrics, f, indent=2)
print(f"XAI metrics saved to {OUTPUT_DIR / f'{case_id}_xai_metrics.json'}")

# slice_comparison_multi(image=ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img), labels=[gt,pred, acc, occ, logit ], titles=["Ground Truth","Prediction", "Saliency", "Occlusion", "logits"])
slice_comparison_multi(image=ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img), labels=[gt, pred, acc, occ], titles=["Ground Truth","Prediction", "Saliency", "Occlusion"])
plt.savefig(OUTPUT_DIR / f"{case_id}_slice_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

import sys
sys.exit(0)


print([acc[i].max() for i in range(3)])
print([acc[i].min() for i in range(3)])


# plt.imshow(activation[1,:,:,5], vmin=0, vmax=1)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white","white", "green"]
        )

fig, ax = plt.subplots(1,3)
for i in range(3):
    ax[i].imshow(acc[i,:,:,best_slice_idx], vmin=-.5, vmax=.5, cmap=cmap)
plt.savefig(OUTPUT_DIR / f"{case_id}_saliency_channels.png", dpi=150, bbox_inches="tight")
plt.close()


i = 2
# normalized = torch.clamp(acc[i] + 0.5 - acc[i].mean(), min=0, max=1)
print(acc[i].mean())
print(acc[i].max())
print(acc[i].min())
# (acc[i] + 0.5 - acc[i].mean()).mean()



from captum.attr import visualization as viz

for i in range(3):
    orig_image = img[i,:,:,best_slice_idx][None].permute(1, 2, 0)
    attr = activation[i,:,:,best_slice_idx][None].permute(1, 2, 0)
    fig, _ = viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map", "blended_heat_map"], signs=["all", "absolute_value", "all"], show_colorbar=True, use_pyplot=False)
    fig.savefig(OUTPUT_DIR / f"{case_id}_channel_{i}_attribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

#orig_image = ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img)[:,:,:,7].permute(1, 2, 0)
#attr = occ_neg[:,:,:,7].permute(1, 2, 0)
#viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map","blended_heat_map"], signs=["all", "all", "negative"], show_colorbar=True, cmap="coolwarm")

#for i in range(3):
    #orig_image = img[i,:,:,15][None].permute(1, 2, 0)
    #attr = occlusion[i,:,:,15][None].permute(1, 2, 0)
    #viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map","blended_heat_map"], signs=["all", "all", "negative"], show_colorbar=True)
