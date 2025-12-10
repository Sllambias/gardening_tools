import numpy as np
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.sampling import torch_resize
from gardening_tools.functional.transforms.sampling import (
    numpy_simulate_lowres,
    torch_simulate_lowres,
)


class Torch_Resize(BaseTransform):
    def __init__(
        self,
        data_key: str = "image",
        target_size: list = [],
        clip_to_input_range: bool = False,
    ):
        self.target_size = target_size
        self.data_key = data_key
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __resize__(self, data_dict):
        data_dict[self.data_key] = torch_resize(
            data_dict[self.data_key],
            target_size=self.target_size,
            clip_to_input_range=self.clip_to_input_range,
        )
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.__resize__(data_dict)
        return data_dict


class Numpy_SimulateLowres(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_per_channel: float = 0.5,
        p_per_axis: float = 0.33,
        zoom_range=(0.5, 1.0),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis
        self.zoom_range = zoom_range
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(zoom_range, shape, p_per_axis):
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape)
        zoom = np.random.uniform(*zoom_range)
        dim = len(shape)
        zoomed_shape = np.round(shape * zoom).astype(int)
        for i in range(dim):
            if np.random.uniform() < p_per_axis:
                shape[i] = zoomed_shape[i]
        return shape

    def __simulatelowres__(self, image, target_shape):
        image = numpy_simulate_lowres(
            image,
            target_shape=target_shape,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                for c in range(data_dict[self.data_key][b].shape[0]):
                    if np.random.uniform() < self.p_per_channel:
                        target_shape = self.get_params(
                            self.zoom_range,
                            data_dict[self.data_key][b, c].shape,
                            self.p_per_axis,
                        )
                        data_dict[self.data_key][b, c] = self.__simulatelowres__(
                            data_dict[self.data_key][b, c], target_shape
                        )
        return data_dict


class Torch_SimulateLowres(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        p_per_axis: float = 0.33,
        zoom_range=(0.5, 1.0),
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.p_per_axis = p_per_axis
        self.zoom_range = zoom_range
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(zoom_range, shape, p_per_axis):
        if isinstance(shape, (list, tuple)):
            shape = np.array(shape)
        zoom = np.random.uniform(*zoom_range)
        dim = len(shape)
        zoomed_shape = np.round(shape * zoom).astype(int)
        for i in range(dim):
            if np.random.uniform() < p_per_axis:
                shape[i] = zoomed_shape[i]
        return shape

    def __simulatelowres__(self, image, target_shape):
        image = torch_simulate_lowres(
            image,
            target_shape=target_shape,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    target_shape = self.get_params(
                        self.zoom_range,
                        data_dict[self.data_key][c].shape,
                        self.p_per_axis,
                    )
                    data_dict[self.data_key][c] = self.__simulatelowres__(
                        data_dict[self.data_key][c], target_shape
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        target_shape = self.get_params(
                            self.zoom_range,
                            data_dict[self.data_key][b, c].shape,
                            self.p_per_axis,
                        )
                        data_dict[self.data_key][b, c] = self.__simulatelowres__(
                            data_dict[self.data_key][b, c], target_shape
                        )

        return data_dict
