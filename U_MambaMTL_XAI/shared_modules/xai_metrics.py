import torch
import math
import numpy as np
from captum.attr import Occlusion, Saliency

FORWARD_BZ = 2  # Small batch size for 3D volumes


def create_agg_segmentation_wrapper(model: torch.nn.Module):
    """
    Create wrapper that aggregates 3D segmentation output to (B, C).

    Selects the predicted class per voxel and sums logits over spatial dims,
    producing (B, num_classes) output suitable for Captum attribution methods.
    """
    def wrapper(inp):
        model_out = model(inp)
        out_max = model_out.argmax(dim=1, keepdim=True)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
        # Use mean instead of sum to keep output in per-voxel logit range,
        # preventing infidelity from scaling with volume size (N^2).
        aggregated_logits = (model_out * selected_inds).mean(dim=(2, 3, 4))
        return aggregated_logits
    return wrapper


def forward_batch(model_fn, inputs: torch.Tensor, batch_size: int = FORWARD_BZ):
    """Run model forward pass in batches for memory efficiency with 3D volumes."""
    n = inputs.shape[0]
    outputs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(n, i + batch_size)
            out = model_fn(inputs[i:end])
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0).numpy()


def get_explanation_pdt(
    image: torch.Tensor,
    model: torch.nn.Module,
    label: int = 0,
    exp: str = "saliency",
):
    """
    Compute explanation attribution map and model prediction for input.

    Args:
        image: Input tensor (B, C, H, W, D)
        model: Segmentation model
        label: Target class index
        exp: Explanation method ("saliency" or "occlusion")

    Returns:
        expl: Attribution map (same shape as image)
        pdtr: Model prediction for target class, shape (B,)
    """
    image = image.requires_grad_(True)
    model.zero_grad()

    agg_wrapper = create_agg_segmentation_wrapper(model)

    # ----------- occlusion parameters -----------
    size = 28
    sliding_window_shapes = (1, size, size, 2)
    strides = (1, size, size, 1)
    baselines = image.mean().item()
    perturbations_per_eval = 1

    out = agg_wrapper(image)
    pdtr = out[:, label]

    if exp.lower() == "saliency":
        saliency_fn = Saliency(agg_wrapper)
        expl = saliency_fn.attribute(image, target=label, abs=True)
    elif exp.lower() == "occlusion":
        occlusion_fn = Occlusion(agg_wrapper)
        expl = occlusion_fn.attribute(
            image,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=label,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=True,
        )
    else:
        raise NotImplementedError(f"Explanation not implemented for: {exp}")

    return expl, pdtr


def sample_eps_Inf_3d(image: torch.Tensor, epsilon: float, N: int) -> torch.Tensor:
    """
    Generate N random perturbations within L-inf ball of radius epsilon.

    Args:
        image: Input tensor (B, C, H, W, D)
        epsilon: L-inf perturbation radius
        N: Number of perturbation samples

    Returns:
        Perturbation tensor of shape (N, C, H, W, D)
    """
    C, H, W, D = image.shape[-4:]
    return torch.empty(
        N, C, H, W, D,
        device=image.device,
        dtype=image.dtype,
    ).uniform_(-epsilon, epsilon)


def get_explanation_sensitivity(
    image: torch.Tensor,
    model: torch.nn.Module,
    expl: torch.Tensor,
    exp: str,
    label: int,
    norm: float,
    sen_r: float = 0.02,
    sen_N: int = 10,
) -> float:
    """
    Compute maximum sensitivity of explanation under input perturbations.

    Sensitivity = max over perturbations of ||attr(x) - attr(x+d)|| / ||attr(x)||

    Lower values indicate more stable/reliable attributions.

    Args:
        image: Input tensor (B, C, H, W, D)
        model: Segmentation model
        expl: Base explanation/attribution map (same shape as image)
        exp: Explanation method name ("saliency" or "occlusion")
        label: Target class label
        norm: L2 norm of base explanation for normalization
        sen_r: Perturbation radius (L-inf)
        sen_N: Number of perturbation samples

    Returns:
        Maximum sensitivity score (float)
    """
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = sample_eps_Inf_3d(image, sen_r, 1).to(image.device)
        X_noisy = image + sample
        expl_eps, _ = get_explanation_pdt(X_noisy, model, label, exp)
        diff_norm = torch.linalg.norm((expl - expl_eps).float()).item()
        max_diff = max(max_diff, diff_norm / norm)
    return max_diff


def get_exp_infidelity(
    image: torch.Tensor,
    model: torch.nn.Module,
    expl: torch.Tensor,
    label: int,
    pdt: float,
    n_samples: int = 100,
    perturb_scale: float = 0.2,
) -> float:
    """
    Compute infidelity score for 3D volume explanations.

    Uses Gaussian perturbation with optimal scaling:
        beta = E[df * (phi . delta)] / E[(phi . delta)^2]
        infidelity = E[(df - beta * phi . delta)^2]

    Lower infidelity indicates the explanation better predicts
    model output changes under perturbation.

    Args:
        image: Input tensor (B, C, H, W, D)
        model: Segmentation model
        expl: Explanation/attribution map (same shape as image)
        label: Target class label
        pdt: Baseline model prediction (scalar)
        n_samples: Number of perturbation samples
        perturb_scale: Standard deviation of Gaussian perturbation

    Returns:
        Infidelity score (float)
    """
    agg_wrapper = create_agg_segmentation_wrapper(model)

    expl_flat = expl.detach().reshape(-1).cpu().numpy().astype(np.float64)
    image_flat = image.detach().cpu().numpy().reshape(-1)
    total = expl_flat.shape[0]

    exp_sums = np.zeros(n_samples)
    pdt_diffs = np.zeros(n_samples)

    for i in range(n_samples):
        # Gaussian perturbation over all voxels
        perturbation = np.random.normal(size=total) * perturb_scale

        # Create perturbed image: x - delta
        perturbed_flat = image_flat - perturbation
        perturbed = torch.tensor(
            perturbed_flat.reshape(image.shape),
            dtype=image.dtype,
            device=image.device,
        )

        # Explanation dot perturbation: phi . delta
        exp_sums[i] = np.sum(perturbation * expl_flat)

        # Model output difference: f(x) - f(x - delta)
        with torch.no_grad():
            out = agg_wrapper(perturbed)
            pdt_perturbed = out[:, label].sum().item()

        pdt_diffs[i] = (pdt - pdt_perturbed)

        del perturbed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Optimal scaling: beta = E[df * (phi.delta)] / E[(phi.delta)^2]
    denom = np.mean(exp_sums ** 2)
    if denom > 0:
        beta = np.mean(pdt_diffs * exp_sums) / denom
    else:
        beta = 1.0
    exp_sums *= beta

    infid = np.mean((pdt_diffs - exp_sums) ** 2)
    return float(infid)


def evaluate_metrics(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    exp: str = "saliency",
    sen_r: float = 0.02,
    sen_N: int = 10,
    infid_n_samples: int = 100,
    infid_perturb_scale: float = 0.5,
    max_batches: int = 5,
):
    """
    Evaluate infidelity and sensitivity metrics over a dataset.

    Args:
        model: Segmentation model
        data_loader: DataLoader yielding dict batches with "image" key
        exp: Explanation method ("saliency" or "occlusion")
        sen_r: L-inf perturbation radius for sensitivity
        sen_N: Number of perturbation samples for sensitivity
        infid_n_samples: Number of perturbation samples for infidelity
        infid_perturb_scale: Gaussian perturbation scale for infidelity
        max_batches: Maximum number of batches to evaluate

    Returns:
        Tuple of (mean_infidelity, mean_sensitivity)
    """
    model.eval()
    infidelities = []
    max_sensitivities = []

    for i, batch in enumerate(data_loader):
        if i >= max_batches:
            break

        X = batch["image"].cuda()
        label = 1  # Target class (e.g., prostate cancer)

        # Compute explanation and baseline prediction
        expl, pdtr = get_explanation_pdt(X, model, label, exp)
        pdt = pdtr.sum().item()

        # L2 norm of explanation for sensitivity normalization
        norm = torch.linalg.norm(expl.float()).item()
        if norm == 0:
            norm = 1.0

        # Compute infidelity
        infid = get_exp_infidelity(
            X, model, expl, label, pdt,
            n_samples=infid_n_samples,
            perturb_scale=infid_perturb_scale,
        )
        infidelities.append(infid)

        # Compute sensitivity
        sensitivity = get_explanation_sensitivity(
            X, model, expl, exp, label, norm,
            sen_r=sen_r, sen_N=sen_N,
        )
        max_sensitivities.append(sensitivity)

    mean_infid = np.mean(infidelities)
    mean_sens = np.mean(max_sensitivities)

    return mean_infid, mean_sens
