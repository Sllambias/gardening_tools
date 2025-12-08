from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.transforms.masking import torch_mask_all_channels


class Torch_Mask(BaseTransform):
    def __init__(
        self,
        data_key: str = "image",
        mask_key: str = "mask",
        pixel_value: float = 0.0,
        ratio: float = 0.6,
        token_size: list[int] = [4],
    ):
        self.data_key = data_key
        self.mask_key = mask_key
        self.pixel_value = pixel_value
        self.ratio = ratio
        self.token_size = token_size

    @staticmethod
    def get_params():
        pass

    def __mask__(
        self,
        image,
    ):
        image, mask = torch_mask_all_channels(
            image=image,
            pixel_value=self.pixel_value,
            ratio=self.ratio,
            token_size=self.token_size,
        )
        return image, mask

    def __call__(self, data_dict):
        image, mask = self.__mask__(data_dict[self.data_key])
        data_dict[self.data_key] = image
        data_dict[self.mask_key] = mask
        return data_dict
