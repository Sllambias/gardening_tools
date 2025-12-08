import torch
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_CastLabelToDtype(BaseTransform):
    def __init__(self, label_key, dtype):
        self.label_key = label_key
        self.dtype = dtype

    @staticmethod
    def get_params(dtype):
        if dtype == "int":
            return torch.int64
        elif dtype == "float":
            return torch.float32

    def __cast__(self, data_dict, target_dtype):
        data_dict[self.label_key] = data_dict[self.label_key].to(target_dtype)
        return data_dict

    def __call__(self, data_dict):
        dtype = self.get_params(self.dtype)
        data_dict = self.__cast__(data_dict, dtype)
        return data_dict
