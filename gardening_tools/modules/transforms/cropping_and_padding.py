import numpy as np
import torch
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.cropping_and_padding import (
    torch_croppad,
    numpy_croppad,
    fit_image_to_patch_size,
)
from typing import Union, Literal

__all__ = ["Numpy_CropPad", "Torch_CropPad", "Numpy_Pad", "Torch_Pad"]


class Numpy_CropPad(BaseTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        pad_value: Union[Literal["min", "zero", "edge"], int, float] = "min",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"constant_values": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {
                "constant_values": np.zeros(1, dtype=data.dtype),
                "mode": "constant",
            }
        elif isinstance(pad_value, int) or isinstance(pad_value, float):
            pad_kwargs = {"constant_values": pad_value, "mode": "constant"}
        elif pad_value == "edge":
            pad_kwargs = {"mode": "edge"}
        else:
            print("Unrecognized pad value detected.")
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __croppad__(
        self,
        data_dict: np.ndarray,
        image_properties: dict,
        input_shape: np.ndarray,
        p_oversample_foreground: float,
        target_image_shape: list | tuple,
        target_label_shape: list | tuple,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
        else:
            label = None

        image, label = numpy_croppad(
            image=image,
            image_properties=image_properties,
            label=label,
            input_dims=len(input_shape[1:]),
            patch_size=self.patch_size,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(
        self, packed_data_dict=None, image_properties=None, **unpacked_data_dict
    ):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict

        input_shape, target_image_shape, target_label_shape, pad_kwargs = (
            self.get_params(
                data=data_dict[self.data_key],
                pad_value=self.pad_value,
                target_shape=self.patch_size,
            )
        )

        data_dict = self.__croppad__(
            data_dict=data_dict,
            image_properties=image_properties,
            input_shape=input_shape,
            p_oversample_foreground=self.p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return data_dict


class Torch_CropPad(BaseTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        pad_value: Union[Literal["min", "zero", "replicate"], int, float] = "min",
        patch_size: tuple | list = None,
        p_oversample_foreground=0.0,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.pad_value = pad_value
        self.patch_size = patch_size
        self.p_oversample_foreground = p_oversample_foreground

    @staticmethod
    def get_params(data, pad_value, target_shape):
        if pad_value == "min":
            pad_kwargs = {"value": data.min(), "mode": "constant"}
        elif pad_value == "zero":
            pad_kwargs = {"value": torch.zeros(1, dtype=data.dtype), "mode": "constant"}
        elif isinstance(pad_value, int) or isinstance(pad_value, float):
            pad_kwargs = {"value": pad_value, "mode": "constant"}
        elif pad_value == "replicate":
            pad_kwargs = {"mode": "replicate"}
        else:
            print("Unrecognized pad value detected.")
        input_shape = data.shape
        target_image_shape = (input_shape[0], *target_shape)
        target_label_shape = (1, *target_shape)
        return input_shape, target_image_shape, target_label_shape, pad_kwargs

    def __croppad__(
        self,
        data_dict: np.ndarray,
        foreground_locations: dict,
        input_shape: np.ndarray,
        p_oversample_foreground: float,
        target_image_shape: list | tuple,
        target_label_shape: list | tuple,
        **pad_kwargs,
    ):
        image = data_dict[self.data_key]
        if data_dict.get(self.label_key) is not None:
            label = data_dict[self.label_key]
        else:
            label = None

        image, label = torch_croppad(
            image=image,
            input_dims=len(input_shape[1:]),
            patch_size=self.patch_size,
            foreground_locations=foreground_locations,
            label=label,
            p_oversample_foreground=p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )

        data_dict[self.data_key] = image
        if label is not None:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, data_dict):
        input_shape, target_image_shape, target_label_shape, pad_kwargs = (
            self.get_params(
                data=data_dict[self.data_key],
                pad_value=self.pad_value,
                target_shape=self.patch_size,
            )
        )

        data_dict = self.__croppad__(
            data_dict=data_dict,
            foreground_locations=data_dict.get("foreground_locations"),
            input_shape=input_shape,
            p_oversample_foreground=self.p_oversample_foreground,
            target_image_shape=target_image_shape,
            target_label_shape=target_label_shape,
            **pad_kwargs,
        )
        return data_dict


class Numpy_Pad(Numpy_CropPad):
    def __call__(
        self, packed_data_dict=None, image_properties=None, **unpacked_data_dict
    ):
        data_dict = packed_data_dict if packed_data_dict else unpacked_data_dict
        image_shape = data_dict[self.data_key].shape
        original_patch_size = self.patch_size
        self.patch_size = fit_image_to_patch_size(self.patch_size, image_shape[1:])
        result = super().__call__(
            packed_data_dict=packed_data_dict,
            image_properties=image_properties,
            **unpacked_data_dict,
        )
        self.patch_size = (
            original_patch_size  # Reset patch size to original after padding
        )
        return result


class Torch_Pad(Torch_CropPad):
    def __call__(self, data_dict):
        original_patch_size = self.patch_size
        image_shape = data_dict[self.data_key].shape
        patch_size = fit_image_to_patch_size(original_patch_size, image_shape[1:])
        self.patch_size = patch_size
        result = super().__call__(data_dict)
        self.patch_size = original_patch_size
        return result


class Torch_CenterCrop(BaseTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        target_size: tuple | list = None,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.target_size = target_size

    @staticmethod
    def get_params():
        return

    def __centercrop__(self, data_dict):
        image = data_dict[self.data_key]

        if len(self.target_size) == 2:
            h, w = torch.tensor(image.shape[1:])
            h_delta, w_delta = torch.tensor(image.shape[1:]) - torch.tensor(
                self.target_size
            )
            h1, h2 = torch.floor(h_delta / 2).to(int), torch.ceil(h_delta / 2).to(int)
            w1, w2 = torch.floor(w_delta / 2).to(int), torch.ceil(w_delta / 2).to(int)
            slices = (slice(None, None), slice(h1, h - h2), slice(w1, w - w2))

        if len(self.target_size) == 3:
            h, w, d = torch.tensor(image.shape[1:])
            h_delta, w_delta, d_delta = torch.tensor(image.shape[1:]) - torch.tensor(
                self.target_size
            )
            h1, h2 = torch.floor(h_delta / 2).to(int), torch.ceil(h_delta / 2).to(int)
            w1, w2 = torch.floor(w_delta / 2).to(int), torch.ceil(w_delta / 2).to(int)
            d1, d2 = torch.floor(d_delta / 2).to(int), torch.ceil(d_delta / 2).to(int)
            slices = (
                slice(None, None),
                slice(h1, h - h2),
                slice(w1, w - w2),
                slice(d1, d - d2),
            )

        data_dict[self.data_key] = image[slices]
        if data_dict.get(self.label_key) is not None:
            data_dict[self.label_key] = data_dict[self.label_key][slices]

        return data_dict

    def __call__(self, data_dict):
        data_dict = self.__centercrop__(
            data_dict=data_dict,
        )
        return data_dict
