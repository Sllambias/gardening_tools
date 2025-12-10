import numpy as np
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.noise import (
    numpy_additive_noise,
    numpy_multiplicative_noise,
    torch_additive_noise,
    torch_multiplicative_noise,
)
from typing import Tuple


class Numpy_AdditiveNoise(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __additiveNoise__(self, image, mean, sigma):
        image = numpy_additive_noise(
            image=image,
            mean=mean,
            sigma=sigma,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                mean, sigma = self.get_params(self.mean, self.sigma)
                if np.random.uniform() < self.p_per_sample:
                    data_dict[self.data_key][b, c] = self.__additiveNoise__(
                        data_dict[self.data_key][b, c], mean, sigma
                    )
        return data_dict


class Numpy_MultiplicativeNoise(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __multiplicativeNoise__(self, image, mean, sigma):
        image = numpy_multiplicative_noise(
            image=image,
            mean=mean,
            sigma=sigma,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    mean, sigma = self.get_params(self.mean, self.sigma)
                    data_dict[self.data_key][b, c] = self.__multiplicativeNoise__(
                        data_dict[self.data_key][b, c], mean, sigma
                    )
        return data_dict


class Torch_AdditiveNoise(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __additiveNoise__(self, image, mean, sigma):
        image = torch_additive_noise(
            image=image,
            mean=mean,
            sigma=sigma,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    mean, sigma = self.get_params(self.mean, self.sigma)
                    data_dict[self.data_key][c] = self.__additiveNoise__(
                        data_dict[self.data_key][c], mean, sigma
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        mean, sigma = self.get_params(self.mean, self.sigma)
                        data_dict[self.data_key][b, c] = self.__additiveNoise__(
                            data_dict[self.data_key][b, c], mean, sigma
                        )

        return data_dict


class Torch_MultiplicativeNoise(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        mean=(0.0, 0.0),
        sigma=(1e-3, 1e-4),
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.mean = mean
        self.sigma = sigma
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(mean: Tuple[float], sigma: Tuple[float]) -> Tuple[float]:
        mean = float(np.random.uniform(*mean))
        sigma = float(np.random.uniform(*sigma))
        return mean, sigma

    def __multiplicativeNoise__(self, image, mean, sigma):
        image = torch_multiplicative_noise(
            image=image,
            mean=mean,
            sigma=sigma,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    mean, sigma = self.get_params(self.mean, self.sigma)
                    data_dict[self.data_key][c] = self.__multiplicativeNoise__(
                        data_dict[self.data_key][c], mean, sigma
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        mean, sigma = self.get_params(self.mean, self.sigma)
                        data_dict[self.data_key][b, c] = self.__multiplicativeNoise__(
                            data_dict[self.data_key][b, c], mean, sigma
                        )
        return data_dict
