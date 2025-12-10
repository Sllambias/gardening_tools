import numpy as np
import torch


def numpy_gibbs_ringing(image, num_sample, axis, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    m = min(0, img_min)
    image += abs(m)
    if len(image.shape) == 3:
        assert axis in [0, 1, 2], "Incorrect or no axis"

        h, w, d = image.shape
        if axis == 0:
            image = image.transpose(0, 2, 1)
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, d, w]))
            image[:, :, 0 : int(np.ceil(w / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(w / 2) + np.ceil(num_sample / 2)) : w] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, d, w]))
            image = image.transpose(0, 2, 1)
        elif axis == 1:
            image = image.transpose(1, 2, 0)
            image = np.fft.fftshift(np.fft.fftn(image, s=[w, d, h]))
            image[:, :, 0 : int(np.ceil(h / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(h / 2) + np.ceil(num_sample / 2)) : h] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[w, d, h]))
            image = image.transpose(2, 0, 1)
        else:
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, w, d]))
            image[:, :, 0 : int(np.ceil(d / 2) - np.ceil(num_sample / 2))] = 0
            image[:, :, int(np.ceil(d / 2) + np.ceil(num_sample / 2)) : d] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, w, d]))
    elif len(image.shape) == 2:
        assert axis in [0, 1], "incorrect or no axis"
        h, w = image.shape
        if axis == 0:
            image = np.fft.fftshift(np.fft.fftn(image, s=[h, w]))
            image[:, 0 : int(np.ceil(w / 2) - np.ceil(num_sample / 2))] = 0
            image[:, int(np.ceil(w / 2) + np.ceil(num_sample / 2)) : w] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[h, w]))
        else:
            image = image.conj().T
            image = np.fft.fftshift(np.fft.fftn(image, s=[w, h]))
            image[:, 0 : int(np.ceil(h / 2) - np.ceil(num_sample / 2))] = 0
            image[:, int(np.ceil(h / 2) + np.ceil(num_sample / 2)) : h] = 0
            image = abs(np.fft.ifftn(np.fft.ifftshift(image), s=[w, h]))
            image = image.conj().T
    image -= abs(m)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image


def torch_gibbs_ringing(
    image: torch.Tensor,
    num_sample: int,
    mode: str = "rect",
    axes: list[int] = None,
    clip_to_input_range: bool = False,
) -> torch.Tensor:
    assert image.ndim in [2, 3], "Only 2D or 3D images supported"
    if mode == "rect":
        assert axes is not None and all(0 <= ax < image.ndim for ax in axes), (
            "Invalid axes for mode 'rect'"
        )

    img_min = image.min()
    img_max = image.max()
    offset = -img_min if img_min < 0 else 0
    image = image + offset

    kspace = torch.fft.fftshift(
        torch.fft.fftn(image, dim=list(range(image.ndim))), dim=list(range(image.ndim))
    )

    shape = image.shape
    center = [s // 2 for s in shape]

    if mode == "rect":
        mask = torch.ones_like(kspace, dtype=torch.bool)
        for axis in axes:
            c = center[axis]
            half = num_sample // 2
            slc = [slice(None)] * image.ndim
            for i in range(shape[axis]):
                if not (c - half <= i < c + half):
                    slc[axis] = slice(i, i + 1)
                    mask[tuple(slc)] = False
        kspace[~mask] = 0
    elif mode == "radial":
        coords = torch.meshgrid(
            [torch.arange(s, device=image.device) - c for s, c in zip(shape, center)],
            indexing="ij",
        )
        dist = torch.sqrt(sum((g.float() ** 2 for g in coords)))
        mask = dist <= num_sample
        kspace[~mask] = 0
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    result = torch.fft.ifftn(
        torch.fft.ifftshift(kspace, dim=list(range(image.ndim))),
        dim=list(range(image.ndim)),
    )
    result = result.abs()

    result = result - offset
    if clip_to_input_range:
        result = torch.clamp(result, min=img_min, max=img_max)

    return result
