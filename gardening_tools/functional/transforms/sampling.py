import torch
import torch.nn.functional as F
from typing import Sequence, List


def torch_downsample_label(label: torch.Tensor, factors: Sequence[float]) -> List[torch.Tensor]:
    """
    Nearest-neighbour down-sampling for label volumes stored as **(B, C, X, Y, Z)**.

    Parameters
    ----------
    label : torch.Tensor
        Input tensor of shape *(B, C, X, Y, Z)*. Any floating-, integer-, or
        boolean-dtype tensor is accepted.
    factors : Sequence[float]
        Iterable of scale factors.  Each factor is applied uniformly to the
        three spatial axes (X, Y, Z).  A factor of 1 returns the original
        tensor instance.

    Returns
    -------
    List[torch.Tensor]
        One tensor per factor, each shaped *(B, C, X', Y', Z')* with the same
        dtype as the input.  Spatial extents are truncated to `floor(dim × f)`.
    """
    if label.ndim != 5:
        raise ValueError("Expects (B,C,H,W,D) tensor, got {}".format(label.shape))
    orig_dtype = label.dtype
    outs = []
    for f in factors:
        if f == 1:
            outs.append(label)
            continue
        scaled = F.interpolate(
            label.float(),
            scale_factor=[f, f, f],
            mode="nearest-exact",  # closest to numpys "edge"-mode, with recompute below
            recompute_scale_factor=True,  # this is needed to reproduce numpy version
        )
        restored = scaled.to(orig_dtype)
        outs.append(restored)
    return outs


def torch_resize(
    image: torch.Tensor,
    target_size: tuple[int],
    clip_to_input_range: bool,
) -> torch.Tensor:
    img_min = image.min()
    img_max = image.max()

    image = image.float()

    if image.ndim == 3:
        mode = "bicubic"
    elif image.ndim == 4:
        if len(target_size) == 2:
            mode = "bicubic"
            image = image[:, torch.randint(0, image.shape[1], (1,)).item()]
        else:
            mode = "trilinear"
    else:
        raise ValueError("Image must be 3D or 4D.")

    result = F.interpolate(image.unsqueeze(0), size=tuple(target_size), mode=mode).squeeze(0)

    if clip_to_input_range:
        result = result.clamp(min=img_min.item(), max=img_max.item())

    return result
