from gardening_tools.functional.transforms.sampling import torch_downsample_label
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_DownsampleSegForDS(BaseTransform):
    def __init__(
        self,
        deep_supervision: bool = False,
        label_key="label",
        factors=(1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 16),
    ):
        self.deep_supervision = deep_supervision
        self.label_key = label_key
        self.factors = factors

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __downsample__(self, label, factors):
        downsampled_labels = torch_downsample_label(label, factors)
        return downsampled_labels

    def __call__(self, data_dict):
        if self.deep_supervision:
            data_dict[self.label_key] = self.__downsample__(
                data_dict[self.label_key], self.factors
            )
        return data_dict
