import numpy as np
import math as m
import torch
import torch.nn.functional as F
from typing import Optional, Union
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter


def numpy_create_zero_centered_coordinate_matrix(shape):
    if len(shape) == 3:
        mesh = np.array(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            )
        ).astype(float)
    if len(shape) == 2:
        mesh = np.array(
            np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        ).astype(float)

    for d in range(len(shape)):
        mesh[d] -= (mesh.shape[d + 1] - 1) / 2
        assert np.mean(mesh[d]) == 0, "beware: mesh didnt zero-center"
    return mesh


def torch_create_zero_centered_coordinate_matrix(
    shape: tuple[int, ...],
) -> torch.Tensor:
    # Using standard numpy indexing 'ij', 2D: (H, W) = (y, x), 3D: (D, H, W) = (z, y, x)
    mesh = torch.stack(
        torch.meshgrid(
            *[torch.arange(s, dtype=torch.float32) for s in shape], indexing="ij"
        )
    )
    for d, s in enumerate(shape):
        mesh[d] -= (s - 1) / 2.0
    return mesh


def deform_coordinate_matrix(coordinate_matrix, alpha, sigma):
    deforms = np.array(
        [
            gaussian_filter(
                (np.random.random(coordinate_matrix.shape[1:]) * 2 - 1),
                sigma,
                mode="constant",
                cval=0,
            )
            * alpha
            for _ in range(coordinate_matrix.shape[0])
        ]
    )
    coordinate_matrix = deforms + coordinate_matrix
    return coordinate_matrix


def Rx(theta):
    return np.array(
        [[1, 0, 0], [0, m.cos(theta), -m.sin(theta)], [0, m.sin(theta), m.cos(theta)]]
    )


def Ry(theta):
    return np.array(
        [[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]]
    )


def Rz(theta):
    return np.array(
        [[m.cos(theta), -m.sin(theta), 0], [m.sin(theta), m.cos(theta), 0], [0, 0, 1]]
    )


def Rz2D(theta):
    return np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])


def get_max_rotated_size(patch_size):
    if len(patch_size) == 2:
        max_dim = int(np.sqrt(patch_size[0] ** 2 + patch_size[1] ** 2))
        return (max_dim, max_dim)

    max_dim_0 = max(
        int(np.sqrt(patch_size[0] ** 2 + patch_size[1] ** 2)),
        int(np.sqrt(patch_size[0] ** 2 + patch_size[2] ** 2)),
    )

    max_dim_1 = max(
        int(np.sqrt(patch_size[1] ** 2 + patch_size[0] ** 2)),
        int(np.sqrt(patch_size[1] ** 2 + patch_size[2] ** 2)),
    )

    max_dim_2 = max(
        int(np.sqrt(patch_size[2] ** 2 + patch_size[0] ** 2)),
        int(np.sqrt(patch_size[2] ** 2 + patch_size[1] ** 2)),
    )

    return (max_dim_0, max_dim_1, max_dim_2)


def numpy_spatial(
    image,
    patch_size,
    p_deform,
    p_rot,
    p_rot_per_axis,
    p_scale,
    alpha,
    sigma,
    x_rot,
    y_rot,
    z_rot,
    scale_factor,
    clip_to_input_range,
    label: Optional[np.ndarray] = None,
    skip_label: bool = False,
    do_crop: bool = True,
    random_crop: bool = True,
    order: int = 3,
    cval: Optional[Union[str, int, float]] = "min",
):
    if not do_crop:
        patch_size = image.shape[2:]
    if cval == "min":
        cval = float(image.min())
    else:
        cval = cval
    assert isinstance(cval, (int, float)), f"got {cval} of type {type(cval)}"

    coords = numpy_create_zero_centered_coordinate_matrix(patch_size)
    image_canvas = np.zeros(
        (image.shape[0], image.shape[1], *patch_size), dtype=np.float32
    )

    # First we apply deformation to the coordinate matrix
    if np.random.uniform() < p_deform:
        coords = deform_coordinate_matrix(coords, alpha=alpha, sigma=sigma)

    # Then we rotate the coordinate matrix around one or more axes
    if np.random.uniform() < p_rot:
        rot_matrix = np.eye(len(patch_size))
        if len(patch_size) == 2:
            rot_matrix = np.dot(rot_matrix, Rz2D(z_rot))
        else:
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Rx(x_rot))
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Ry(y_rot))
            if np.random.uniform() < p_rot_per_axis:
                rot_matrix = np.dot(rot_matrix, Rz(z_rot))

        coords = (
            np.dot(coords.reshape(len(patch_size), -1).transpose(), rot_matrix)
            .transpose()
            .reshape(coords.shape)
        )

    # And finally scale it
    # Scaling effect is "inverted"
    # i.e. a scale factor of 0.9 will zoom in
    if np.random.uniform() < p_scale:
        coords *= scale_factor

    if random_crop and do_crop:
        for d in range(len(patch_size)):
            crop_center_idx = [
                np.random.randint(
                    int(patch_size[d] / 2),
                    image.shape[d + 2] - int(patch_size[d] / 2) + 1,
                )
            ]
            coords[d] += crop_center_idx
    else:
        # Reversing the zero-centering of the coordinates
        for d in range(len(patch_size)):
            coords[d] += image.shape[d + 2] / 2.0 - 0.5

    # Mapping the images to the distorted coordinates
    for b in range(image.shape[0]):
        for c in range(image.shape[1]):
            img_min = image.min()
            img_max = image.max()

            image_canvas[b, c] = map_coordinates(
                image[b, c].astype(float),
                coords,
                order=order,
                mode="constant",
                cval=cval,
            ).astype(image.dtype)

            if clip_to_input_range:
                image_canvas[b, c] = np.clip(
                    image_canvas[b, c], a_min=img_min, a_max=img_max
                )

    if label is not None and not skip_label:
        label_canvas = np.zeros(
            (label.shape[0], label.shape[1], *patch_size),
            dtype=np.float32,
        )

        # Mapping the labelmentations to the distorted coordinates
        for b in range(label.shape[0]):
            for c in range(label.shape[1]):
                label_canvas[b, c] = map_coordinates(
                    label[b, c], coords, order=0, mode="constant", cval=0.0
                ).astype(label.dtype)
        return image_canvas, label_canvas
    return image_canvas, None


def torch_spatial(
    image: torch.Tensor,
    patch_size: tuple[int],
    p_deform: float,
    p_rot: float,
    p_rot_per_axis: float,
    p_scale: float,
    alpha: float,
    sigma: float,
    x_rot: float,
    y_rot: float,
    z_rot: float,
    scale_factor: float,
    clip_to_input_range: bool,
    label: Optional[torch.Tensor] = None,
    skip_label: bool = False,
    do_crop: bool = True,
    random_crop: bool = True,
    interpolation_mode: str = "bilinear",
    seed: Optional[int] = None,
):
    if seed is not None:
        torch.manual_seed(seed)

    device, dtype, orig_ndim = image.device, image.dtype, image.ndim
    ndim = len(patch_size)

    # Expand dims if needed
    def _prepare(x):
        return (
            x[None, None]
            if orig_ndim == ndim
            else x[None]
            if orig_ndim == ndim + 1
            else x
        )

    image = _prepare(image)
    if label is not None:
        label = _prepare(label)

    if not do_crop:
        patch_size = image.shape[2:]

    coords = torch_create_zero_centered_coordinate_matrix(patch_size).to(device, dtype)
    if torch.rand(1) < p_deform:
        if ndim == 2:
            # Separable 2D blur
            noise = torch.randn(1, 1, *patch_size, device=device, dtype=dtype)
            ksize = 21
            ax = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
            k = torch.exp(-0.5 * (ax / sigma) ** 2)
            k /= k.sum()
            ky = k.view(1, 1, -1, 1)
            kx = k.view(1, 1, 1, -1)
            noise = F.conv2d(noise, ky, padding=(ksize // 2, 0), groups=1)
            noise = F.conv2d(noise, kx, padding=(0, ksize // 2), groups=1)
        else:
            # Separable 3D blur
            noise = torch.randn(1, ndim, *patch_size, device=device, dtype=dtype)
            ksize = 9
            ax = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
            k = torch.exp(-0.5 * (ax / sigma) ** 2)
            k /= k.sum()
            for dim in range(3):
                shape = [1, 1, 1, 1, 1]
                shape[dim + 2] = ksize
                kernel = k.view(*shape).repeat(ndim, 1, 1, 1, 1)
                padding = [ksize // 2 if i == dim else 0 for i in range(3)]
                noise = F.conv3d(noise, kernel, padding=padding, groups=ndim)
        coords += noise[0] * alpha

    if torch.rand(1) < p_rot:
        rot = torch.eye(ndim, device=device, dtype=dtype)
        if ndim == 2:
            rot = rot @ torch.from_numpy(Rz2D(z_rot)).to(device=device, dtype=dtype)
        else:
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Rx(x_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Ry(y_rot)).to(device=device, dtype=dtype)
            if torch.rand(1) < p_rot_per_axis:
                rot = rot @ torch.from_numpy(Rz(z_rot)).to(device=device, dtype=dtype)
        coords = (rot @ coords.view(ndim, -1)).view_as(coords)

    if torch.rand(1) < p_scale:
        coords *= scale_factor

    if random_crop and do_crop:
        for d in range(ndim):
            lo = patch_size[d] // 2
            hi = image.shape[d + 2] - patch_size[d] // 2 + 1
            coords[d] += torch.randint(lo, hi, (1,), device=device)
    else:
        for d in range(ndim):
            coords[d] += image.shape[d + 2] / 2 - 0.5

    for d in range(ndim):
        coords[d] = 2 * coords[d] / (image.shape[d + 2] - 1) - 1

    # Swap axes to (x, y) or (x, y, z) order for grid_sample (torch does not default to numpy indexing here)
    grid = coords.permute(*range(1, ndim + 1), 0)
    grid = torch.stack([grid] * image.shape[0], dim=0)

    if ndim == 2:
        grid = grid[..., [1, 0]]
    elif ndim == 3:
        grid = grid[..., [2, 1, 0]]
    else:
        raise ValueError("Only 2D and 3D supported")
    grid_sample_args = {
        "mode": interpolation_mode,
        "padding_mode": "zeros",
        "align_corners": True,
    }
    image_canvas = F.grid_sample(image, grid, **grid_sample_args)
    if clip_to_input_range:
        image_canvas = torch.clamp(image_canvas, min=image.min(), max=image.max())

    if label is not None and not skip_label:
        label_canvas = F.grid_sample(
            label.float(),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        label_canvas = label_canvas.to(label.dtype)
    else:
        label_canvas = None

    def _restore(x: Optional[torch.Tensor], target_ndim: int) -> Optional[torch.Tensor]:
        if x is None:
            return None
        # For 3D volumes, we want to keep the spatial dimensions
        current_dims = x.ndim
        for i in range(current_dims - target_ndim):
            x = x.squeeze(0)
        return x

    return _restore(image_canvas, orig_ndim), (
        _restore(label_canvas, orig_ndim if label is not None else 0)
        if label_canvas is not None
        else None
    )
