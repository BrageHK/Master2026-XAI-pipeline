"""
3D-compatible AblationCAM for volumetric (B, C, D, H, W) data.

pytorch_grad_cam's AblationCAM has three hardcoded 4D tensor assumptions
that break for 5D inputs.  This module fixes them via minimal subclassing:

  AblationLayer3D  — fixes set_next_batch() and __call__()
  AblationCAM3D    — fixes get_cam_weights() batch tensor creation

Everything else (BaseCAM weight broadcasting, scale_cam_image, etc.)
already supports 3D and is unchanged.
"""

import numpy as np
import torch
import tqdm
from typing import Callable, List

from pytorch_grad_cam.ablation_cam import AblationCAM
from pytorch_grad_cam.ablation_layer import AblationLayer
from pytorch_grad_cam.utils.find_layers import replace_layer_recursive


class AblationLayer3D(AblationLayer):
    """AblationLayer that handles any number of spatial dimensions."""

    def set_next_batch(
        self,
        input_batch_index: int,
        activations: torch.Tensor,
        num_channels_to_ablate: int,
    ) -> None:
        """
        Extract one sample from the batch and repeat it num_channels_to_ablate times.

        Parent class assumes activations are 4D (B, C, H, W) and hardcodes
        ``activations[i, :, :, :]``.  Here we index dynamically so it works
        for 5D (B, C, D, H, W) and beyond.
        """
        # activations: (B, C, *spatial)  →  activation: (C, *spatial)
        activation = activations[input_batch_index]
        # Expand to (num_channels_to_ablate, C, *spatial) — no data copy until .contiguous()
        self.activations = (
            activation.clone()
            .unsqueeze(0)
            .expand(num_channels_to_ablate, *activation.shape)
            .contiguous()
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Zero-out (or set very negative) the selected channel for each batch member.

        Parent class uses ``output[i, ch, :]`` which only zeros the first
        spatial dimension.  Using ``output[i, ch]`` ablates all spatial voxels.

        min_val is computed once before the loop so that earlier ablations do
        not compound into later ones (in-place edits would shift torch.min).
        """
        output = self.activations  # (num_ch, C, *spatial)
        min_val = torch.min(output).item()
        ablation_value = 0 if min_val == 0 else min_val - 1e7
        for i in range(output.size(0)):
            output[i, self.indices[i]] = ablation_value
        return output


class AblationCAM3D(AblationCAM):
    """AblationCAM that works with 3D volumetric inputs (B, C, D, H, W)."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        batch_size: int = 32,
        ratio_channels_to_ablate: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            target_layers,
            reshape_transform,
            ablation_layer=AblationLayer3D(),
            batch_size=batch_size,
            ratio_channels_to_ablate=ratio_channels_to_ablate,
        )

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[Callable],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        """
        Identical to AblationCAM.get_cam_weights except the batch tensor is
        created with unsqueeze+expand so it works for both 2D and 3D inputs.

        Parent bug: ``tensor.repeat(B, 1, 1, 1)`` on a 4D tensor (C, D, H, W)
        repeats the *channel* axis B times, giving (C*B, D, H, W) instead of
        the required (B, C, D, H, W).

        Fix: ``tensor.unsqueeze(0).expand(B, *tensor.shape).contiguous()``
        works for any tensor rank.
        """
        handle = target_layer.register_forward_hook(self.save_activation)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            handle.remove()
            original_scores = np.float32(
                [target(output).cpu().item() for target, output in zip(targets, outputs)]
            )

        ablation_layer = self.ablation_layer
        replace_layer_recursive(self.model, target_layer, ablation_layer)

        number_of_channels = activations.shape[1]
        weights = []

        with torch.no_grad():
            for batch_index, (target, tensor) in enumerate(zip(targets, input_tensor)):
                new_scores = []

                # FIX: tensor shape is (C, *spatial); unsqueeze+expand works for any rank
                batch_tensor = (
                    tensor.unsqueeze(0)
                    .expand(self.batch_size, *tensor.shape)
                    .contiguous()
                )

                channels_to_ablate = ablation_layer.activations_to_be_ablated(
                    activations[batch_index, :], self.ratio_channels_to_ablate
                )
                number_channels_to_ablate = len(channels_to_ablate)

                for i in tqdm.tqdm(range(0, number_channels_to_ablate, self.batch_size)):
                    if i + self.batch_size > number_channels_to_ablate:
                        batch_tensor = batch_tensor[: (number_channels_to_ablate - i)]

                    ablation_layer.set_next_batch(
                        input_batch_index=batch_index,
                        activations=self.activations,
                        num_channels_to_ablate=batch_tensor.size(0),
                    )
                    score = [target(o).cpu().item() for o in self.model(batch_tensor)]
                    new_scores.extend(score)
                    ablation_layer.indices = ablation_layer.indices[batch_tensor.size(0):]

                new_scores = self.assemble_ablation_scores(
                    new_scores,
                    original_scores[batch_index],
                    channels_to_ablate,
                    number_of_channels,
                )
                weights.extend(new_scores)

        weights = np.float32(weights)
        weights = weights.reshape(activations.shape[:2])
        original_scores = original_scores[:, None]
        weights = (original_scores - weights) / original_scores

        replace_layer_recursive(self.model, ablation_layer, target_layer)
        return weights


def find_decoder_feature_layers(
    model: torch.nn.Module,
    n_layers: int = 3,
    min_channels: int = 8,
) -> List[torch.nn.Module]:
    """
    Return the last *n_layers* nn.Conv3d feature layers in *model*, skipping
    the final segmentation head(s).

    "Feature" layers are defined as Conv3d whose *out_channels* >= min_channels.
    The segmentation head is a 1×1 conv with out_channels == n_classes (usually 2),
    which will be below min_channels and therefore excluded automatically.

    Layers are returned in forward order (shallowest → deepest), so passing
    them directly to AblationCAM3D produces a multi-layer averaged CAM.

    Args:
        model:       The network to search.
        n_layers:    How many layers to return.
        min_channels: Minimum out_channels to be considered a feature layer.

    Raises:
        RuntimeError: If fewer than one qualifying layer is found.
    """
    feature_convs = [
        m for m in model.modules()
        if isinstance(m, torch.nn.Conv3d) and m.out_channels >= min_channels
    ]
    if not feature_convs:
        raise RuntimeError(
            f"No nn.Conv3d with out_channels >= {min_channels} found in model"
        )
    # Take the last n_layers in forward (definition) order
    return feature_convs[-n_layers:]
