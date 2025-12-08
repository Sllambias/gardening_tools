from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.gamma import numpy_gamma, torch_gamma
import numpy as np


class Numpy_Gamma(BaseTransform):
    """
    WRAPPER FOR NNUNET AUGMENT GAMMA: https://github.com/MIC-DKFZ/batchgenerators/blob/8822a08a7dbfa4986db014e6a74b040778164ca6/batchgenerators/augmentations/color_augmentations.py
    """

    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        p_invert_image: float = 0.05,
        gamma_range=(0.5, 2.0),
        per_channel=True,
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image
        self.per_channel = per_channel
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(p_invert_image):
        # No parameters to retrieve
        do_invert = False
        if np.random.uniform() < p_invert_image:
            do_invert = True
        return do_invert

    def __gamma__(self, image, gamma_range, invert_image, per_channel):
        return numpy_gamma(
            image,
            gamma_range,
            invert_image,
            per_channel,
            clip_to_input_range=self.clip_to_input_range,
        )

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            if np.random.uniform() < self.p_per_sample:
                do_invert = self.get_params(self.p_invert_image)
                data_dict[self.data_key][b] = self.__gamma__(
                    data_dict[self.data_key][b],
                    self.gamma_range,
                    do_invert,
                    per_channel=self.per_channel,
                )
        return data_dict


class Torch_Gamma(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_all_channel: float = 0.0,
        p_invert_image: float = 0.05,
        gamma_range=(0.5, 2.0),
        per_channel=True,
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_all_channel = p_all_channel
        self.gamma_range = gamma_range
        self.p_invert_image = p_invert_image
        self.per_channel = per_channel
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(p_invert_image):
        # No parameters to retrieve
        do_invert = False
        if np.random.uniform() < p_invert_image:
            do_invert = True
        return do_invert

    def __gamma__(self, image, gamma_range, invert_image, per_channel):
        return torch_gamma(
            image,
            gamma_range,
            invert_image,
            per_channel,
            clip_to_input_range=self.clip_to_input_range,
        )

    def __call__(self, data_dict):
        if not self.batched:
            if np.random.uniform() < self.p_all_channel:
                do_invert = self.get_params(self.p_invert_image)
                data_dict[self.data_key] = self.__gamma__(
                    data_dict[self.data_key],
                    self.gamma_range,
                    do_invert,
                    per_channel=self.per_channel,
                )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_all_channel:
                    do_invert = self.get_params(self.p_invert_image)
                    data_dict[self.data_key][b] = self.__gamma__(
                        data_dict[self.data_key][b],
                        self.gamma_range,
                        do_invert,
                        per_channel=self.per_channel,
                    )
        return data_dict
