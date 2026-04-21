import torch


def _make_forward_func_sigmoid(network: torch.nn.Module, fixed_mask: torch.Tensor,
                               aggregation: str = "sum"):
    """For MONAI models — raw logit channel 1 aggregated over masked voxels → (B,).

    aggregation: one of 'sum', 'mean', 'abs_sum', 'abs_avg'.
    Uses raw logits (before sigmoid) so abs variants are meaningful.
    """
    def agg_segmentation_wrapper(inp):
        out = network(inp)[:, 0:2, ...]  # (B, 2, H, W, D)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "as_tensor"):
            out = out.as_tensor()
        flat = (out[:, 1, ...] * fixed_mask).flatten(1)  # (B, N)
        if aggregation == "mean":    return flat.mean(dim=1)
        if aggregation == "abs_sum": return flat.abs().sum(dim=1)
        if aggregation == "abs_avg": return flat.abs().mean(dim=1)
        return flat.sum(dim=1)  # default: sum
    return agg_segmentation_wrapper


def _make_forward_func_softmax(network: torch.nn.Module, fixed_mask: torch.Tensor,
                               aggregation: str = "sum"):
    """For nnUNet — softmax channel 1 aggregated over masked voxels → (B,).

    aggregation: one of 'sum', 'mean', 'abs_sum', 'abs_avg'.
    """
    def _forward(inp: torch.Tensor) -> torch.Tensor:
        out = network(inp)
        if isinstance(out, (list, tuple)):
            out = out[0]
        cancer_prob = torch.softmax(out, dim=1)[:, 1]
        flat = (cancer_prob * fixed_mask).flatten(1)
        if aggregation == "mean":    return flat.mean(dim=1)
        if aggregation == "abs_sum": return flat.abs().sum(dim=1)
        if aggregation == "abs_avg": return flat.abs().mean(dim=1)
        return flat.sum(dim=1)  # default: sum
    return _forward
