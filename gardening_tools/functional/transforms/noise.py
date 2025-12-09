import torch
import numpy as np


def numpy_additive_noise(image, mean, sigma, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    image += np.random.normal(mean, sigma, image.shape)
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image


def numpy_multiplicative_noise(image, mean, sigma, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    gauss = np.random.normal(mean, sigma, image.shape)
    image += image * gauss
    if clip_to_input_range:
        image = np.clip(image, a_min=img_min, a_max=img_max)
    return image


def torch_additive_noise(image, mean, sigma, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    image = image + torch.normal(mean, sigma, image.shape, device=image.device)
    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image


def torch_multiplicative_noise(image, mean, sigma, clip_to_input_range: bool = False):
    img_min = image.min()
    img_max = image.max()
    gauss = torch.normal(mean, sigma, image.shape, device=image.device)
    image = image + image * gauss
    if clip_to_input_range:
        image = torch.clamp(image, min=img_min, max=img_max)
    return image
