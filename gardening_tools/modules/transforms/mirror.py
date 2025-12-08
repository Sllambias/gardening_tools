import torch
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_Mirror(BaseTransform):
    def __init__(
        self,
        data_key="image",
        label_key="label",
        p_per_sample: float = 1.0,
        axes=(0, 1, 2),
        p_mirror_per_axis=0.33,
        skip_label=False,
    ):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.p_mirror_per_axis = p_mirror_per_axis
        self.axes = axes
        self.skip_label = skip_label

    @staticmethod
    def get_params():
        pass

    def __mirror__(self, data_dict, axes):
        image = data_dict[self.data_key]
        label = data_dict.get(self.label_key)

        if torch.rand(1).item() < self.p_per_sample:
            for axis in axes:
                if torch.rand(1).item() < self.p_mirror_per_axis:
                    flip_dim = axis + 1  # C=0, so X,Y,Z → dims 1,2,3
                    image = torch.flip(image, [flip_dim])
                    if label is not None and not self.skip_label:
                        label = torch.flip(label, [flip_dim])

        data_dict[self.data_key] = image
        if label is not None and not self.skip_label:
            data_dict[self.label_key] = label
        return data_dict

    def __call__(self, data_dict):
        return self.__mirror__(data_dict, self.axes)
