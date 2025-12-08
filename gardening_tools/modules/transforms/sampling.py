from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.sampling import torch_resize


class Torch_Resize(BaseTransform):
    def __init__(self, data_key: str = "image", target_size: list = [], clip_to_input_range: bool = False):
        self.target_size = target_size
        self.data_key = data_key
        self.clip_to_input_range = clip_to_input_range

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __resize__(self, data_dict):
        data_dict[self.data_key] = torch_resize(
            data_dict[self.data_key], target_size=self.target_size, clip_to_input_range=self.clip_to_input_range
        )
        return data_dict

    def __call__(self, data_dict):
        data_dict = self.__resize__(data_dict)
        return data_dict
