from gardening_tools.modules.transforms.BaseTransform import BaseTransform
import numpy as np
from gardening_tools.functional.transforms.bias_field import (
    numpy_bias_field,
    torch_bias_field,
)


class Numpy_BiasField(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __biasField__(self, image):
        image = numpy_bias_field(image, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b, c] = self.__biasField__(
                        data_dict[self.data_key][b, c]
                    )
        return data_dict


class Torch_BiasField(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __biasField__(self, image):
        image = torch_bias_field(image, clip_to_input_range=self.clip_to_input_range)
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    data_dict[self.data_key][c] = self.__biasField__(
                        data_dict[self.data_key][c]
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        data_dict[self.data_key][b, c] = self.__biasField__(
                            data_dict[self.data_key][b, c]
                        )
        return data_dict
