import numpy as np
from gardening_tools.functional.transforms.motion_ghosting import (
    numpy_motion_ghosting,
    torch_motion_ghosting,
)
from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from typing import Tuple


class Numpy_MotionGhosting(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_sample: float = 1.0,
        alpha=(0.85, 0.95),
        num_reps=(2, 5),
        axes=(0, 3),
        clip_to_input_range=False,
    ):
        self.data_key = data_key
        self.p_per_sample = p_per_sample
        self.alpha = alpha
        self.num_reps = num_reps
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params(
        alpha: Tuple[float], num_reps: Tuple[float], axes: Tuple[float]
    ) -> Tuple[float]:
        alpha = np.random.uniform(*alpha)
        num_reps = np.random.randint(*num_reps)
        axis = np.random.randint(*axes)
        return alpha, num_reps, axis

    def __motionGhosting__(self, image, alpha, num_reps, axis):
        image = numpy_motion_ghosting(
            image=image,
            alpha=alpha,
            num_reps=num_reps,
            axis=axis,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        for b in range(data_dict[self.data_key].shape[0]):
            for c in range(data_dict[self.data_key][b].shape[0]):
                if np.random.uniform() < self.p_per_sample:
                    alpha, num_reps, axis = self.get_params(
                        self.alpha, self.num_reps, self.axes
                    )
                    data_dict[self.data_key][b, c] = self.__motionGhosting__(
                        data_dict[self.data_key][b, c], alpha, num_reps, axis
                    )
        return data_dict


class Torch_MotionGhosting(BaseTransform):
    def __init__(
        self,
        data_key="image",
        p_per_channel: float = 0.0,
        alpha=(0.85, 0.95),
        num_reps=(2, 5),
        axes=(0, 3),
        clip_to_input_range=False,
        batched: bool = True,
    ):
        self.data_key = data_key
        self.p_per_channel = p_per_channel
        self.alpha = alpha
        self.num_reps = num_reps
        self.axes = axes
        self.clip_to_input_range = clip_to_input_range
        self.batched = batched

    @staticmethod
    def get_params(
        alpha: Tuple[float], num_reps: Tuple[float], axes: Tuple[float]
    ) -> Tuple[float]:
        alpha = np.random.uniform(*alpha)
        num_reps = np.random.randint(*num_reps)
        axis = np.random.randint(*axes)
        return alpha, num_reps, axis

    def __motionGhosting__(self, image, alpha, num_reps, axis):
        image = torch_motion_ghosting(
            image=image,
            alpha=alpha,
            num_reps=num_reps,
            axis=axis,
            clip_to_input_range=self.clip_to_input_range,
        )
        return image

    def __call__(self, data_dict):
        if not self.batched:
            for c in range(data_dict[self.data_key].shape[0]):
                if np.random.uniform() < self.p_per_channel:
                    alpha, num_reps, axis = self.get_params(
                        self.alpha, self.num_reps, self.axes
                    )
                    data_dict[self.data_key][c] = self.__motionGhosting__(
                        data_dict[self.data_key][c], alpha, num_reps, axis
                    )
        else:
            for b in range(data_dict[self.data_key].shape[0]):
                for c in range(data_dict[self.data_key].shape[1]):
                    if np.random.uniform() < self.p_per_channel:
                        alpha, num_reps, axis = self.get_params(
                            self.alpha, self.num_reps, self.axes
                        )
                        data_dict[self.data_key][b, c] = self.__motionGhosting__(
                            data_dict[self.data_key][b, c], alpha, num_reps, axis
                        )
        return data_dict
