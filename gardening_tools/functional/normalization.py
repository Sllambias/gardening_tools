import torch


def volume_wise_znorm(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor using volume-wise z-normalization.
    Assumes the input is a 3D tensor (X, Y, Z) or 2D Tensor (X, Y).
    """
    mean = x.mean()
    std = x.std(unbiased=False).clamp_min(1e-8)
    return (x - mean) / std


def ct_normalization(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor using CT-specific normalization.
    Assumes the input is a 3D tensor (X, Y, Z) or 2D Tensor (X, Y).
    """
    x = x.clamp(-1024, 1024) / 1024
    return x
