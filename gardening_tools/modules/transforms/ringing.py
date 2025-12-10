from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.ringing import (
    numpy_gibbs_ringing,
    torch_gibbs_ringing,
)
import numpy as np


class Numpy_GibbsRinging(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        cut_freq=(96, 129),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.cut_freq = cut_freq
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(cut_freq, axes):
        cut_freq = np.random.randint(*cut_freq)
        axis = np.random.randint(*axes)
        return cut_freq, axis

    def __gibbsRinging__(self, image, num_sample, axis):
        image = numpy_gibbs_ringing(
            image,
            num_sample=num_sample,
            axis=axis,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                    data_dict[self.data_key][b, c] = self.__gibbsRinging__(
                        data_dict[self.data_key][b, c], cut_freq, axis
                    )
        return data_dict


class Torch_GibbsRinging(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        cut_freq=(96, 129),
        axes=(0, 3),
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.cut_freq = cut_freq
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(cut_freq, axes):
        cut_freq = np.random.randint(*cut_freq)
        axis = np.random.randint(*axes)
        return cut_freq, axis

    def __gibbsRinging__(self, image, num_sample, axis):
        image = torch_gibbs_ringing(
            image,
            num_sample=num_sample,
            axes=[axis],
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                    data_dict[self.data_key][c] = self.__gibbsRinging__(
                        data_dict[self.data_key][c], cut_freq, axis
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        cut_freq, axis = self.get_params(self.cut_freq, self.axes)
                        data_dict[self.data_key][b, c] = self.__gibbsRinging__(
                            data_dict[self.data_key][b, c], cut_freq, axis
                        )

        return data_dict
