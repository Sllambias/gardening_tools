import numpy as np
import torch


def numpy_bias_field(image, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()

    if len(image.shape) == 3:
        x, y, z = image.shape
        X, Y, Z = np.meshgrid(
            np.linspace(0, x, x, endpoint=False),
            np.linspace(0, y, y, endpoint=False),
            np.linspace(0, z, z, endpoint=False),
            indexing="ij",
        )
        x0 = np.random.randint(0, x)
        y0 = np.random.randint(0, y)
        z0 = np.random.randint(0, z)
        G = 1 - (
            np.power((X - x0), 2) / (x**2)
            + np.power((Y - y0), 2) / (y**2)
            + np.power((Z - z0), 2) / (z**2)
        )
    else:
        x, y = image.shape
        X, Y = np.meshgrid(
            np.linspace(0, x, x, endpoint=False),
            np.linspace(0, y, y, endpoint=False),
            indexing="ij",
        )
        x0 = np.random.randint(0, x)
        y0 = np.random.randint(0, y)
        G = 1 - (np.power((X - x0), 2) / (x**2) + np.power((Y - y0), 2) / (y**2))
    image = np.multiply(G, image)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image


def torch_bias_field(
    image: torch.Tensor, clip_to_input_range: bool = False
) -> torch.Tensor:
    device = image.device
    img_min = image.min()
    img_max = image.max()

    if len(image.shape) == 3:
        assert image.ndim == 3, "Expected [H, W, D] tensor"
        x, y, z = image.shape
        X, Y, Z = torch.meshgrid(
            torch.linspace(0, x - 1, x, device=device),
            torch.linspace(0, y - 1, y, device=device),
            torch.linspace(0, z - 1, z, device=device),
            indexing="ij",
        )
        x0 = torch.randint(0, x, (1,), device=device)
        y0 = torch.randint(0, y, (1,), device=device)
        z0 = torch.randint(0, z, (1,), device=device)
        G = 1 - (
            (X - x0) ** 2 / (x**2) + (Y - y0) ** 2 / (y**2) + (Z - z0) ** 2 / (z**2)
        )
    else:
        assert image.ndim == 2, "Expected [H, W] tensor"

        x, y = image.shape
        X, Y = torch.meshgrid(
            torch.linspace(0, x - 1, x, device=device),
            torch.linspace(0, y - 1, y, device=device),
            indexing="ij",
        )
        x0 = torch.randint(0, x, (1,), device=device)
        y0 = torch.randint(0, y, (1,), device=device)
        G = 1 - ((X - x0) ** 2 / (x**2) + (Y - y0) ** 2 / (y**2))
    image = G * image

    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)

    return image
