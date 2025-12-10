from gardening_tools.modules.transforms.BaseTransform import BaseTransform
from gardening_tools.functional.normalization import volume_wise_znorm, ct_normalization


class Torch_Normalize(BaseTransform):
    def __init__(
        self, normalize: bool = False, data_key: str = "image", fn=volume_wise_znorm
    ):
        self.normalize = normalize
        self.data_key = data_key
        self.fn = fn

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __normalize__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if isinstance(self.fn, list):
                fn = self.fn[c]
            else:
                fn = self.fn
            data_dict[self.data_key][c] = fn(data_dict[self.data_key][c])
        return data_dict

    def __call__(self, data_dict):
        if self.normalize:
            data_dict = self.__normalize__(data_dict)
        return data_dict


class Torch_CT_NormalizeC0(BaseTransform):
    def __init__(
        self,
        normalize: bool = False,
        data_key: str = "image",
        non_ct_fn=volume_wise_znorm,
        ct_fn=ct_normalization,
    ):
        self.normalize = normalize
        self.data_key = data_key
        self.ct_fn = ct_fn
        self.non_ct_fn = non_ct_fn

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __normalize__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            if c == 0:
                fn = self.ct_fn
            else:
                fn = self.non_ct_fn
            data_dict[self.data_key][c] = fn(data_dict[self.data_key][c])
        return data_dict

    def __call__(self, data_dict):
        if self.normalize:
            data_dict = self.__normalize__(data_dict)
        return data_dict
