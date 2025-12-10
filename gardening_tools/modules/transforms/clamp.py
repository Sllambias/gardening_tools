import torch
from gardening_tools.modules.transforms.BaseTransform import BaseTransform


class Torch_Clamp(BaseTransform):
    def __init__(
        self,
        clamp: bool = False,
        data_key: str = "image",
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ):
        """
        Initialize the Torch_Clamp transform.

        Args:
            clamp (bool): Whether to apply clamping
            data_key (str): Key of the data to clamp in the data dictionary
            lower_percentile (float): Lower percentile for clamping when data has negative values (default: 0.01)
            upper_percentile (float): Upper percentile for clamping (default: 0.99)
        """
        self.clamp = clamp
        self.data_key = data_key
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    @staticmethod
    def get_params():
        # No parameters to retrieve
        pass

    def __clamp__(self, data_dict):
        for c in range(data_dict[self.data_key].shape[0]):
            channel_data = data_dict[self.data_key][c]

            if channel_data.ndim == 2:
                flat_data = channel_data[::2, ::2].flatten()
            elif channel_data.ndim == 3:
                flat_data = channel_data[::2, ::2, ::2].flatten()
            else:
                flat_data = channel_data.flatten()

            sorted_data = torch.sort(flat_data)[0]  # Sort only once
            n = sorted_data.numel()

            if self.lower_percentile is not None:
                lower_idx = int(self.lower_percentile * (n - 1))
                lower_bound = sorted_data[lower_idx]
            else:
                lower_bound = None

            if self.upper_percentile is not None:
                upper_idx = int(self.upper_percentile * (n - 1))
                upper_bound = sorted_data[upper_idx]
            else:
                upper_bound = None

            data_dict[self.data_key][c] = torch.clamp(
                channel_data, min=lower_bound, max=upper_bound
            )

        return data_dict

    def __call__(self, data_dict):
        if self.clamp:
            data_dict = self.__clamp__(data_dict)
        return data_dict
