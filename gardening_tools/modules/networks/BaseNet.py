import torch
from abc import abstractmethod
from torch import nn
from gardening_tools.modules.networks.utils import get_steps_for_sliding_window


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # Attributes that need to be defined for each architecture
        self.abbreviation: str = None
        self.dimensions: int = None

    @abstractmethod
    def forward(self):
        """
        implement in individual trainers.
        DO NOT INCLUDE FINAL SOFTMAX/SIGMOID ETC.
        WILL BE HANDLED BY LOSS FUNCTIONS
        """

    def load_state_dict(self, target_state_dict, *args, **kwargs):
        current_state_dict = self.state_dict()
        # filter unnecessary keys
        target_state_dict = {
            k: v
            for k, v in target_state_dict.items()
            if (k in current_state_dict)
            and (current_state_dict[k].shape == target_state_dict[k].shape)
        }
        super().load_state_dict(target_state_dict, *args, **kwargs)

    def sliding_window_predict(
        self,
        data,
        patch_size,
        overlap,
        mirror=False,
        batch_size=1,
    ):
        """batch_size is the number of windows passed through the network at once."""
        if len(patch_size) == 3:
            mode = "3D"
        elif len(patch_size) == 2:
            mode = "2D"
        else:
            raise ValueError(
                f"patch_size must be of length 2 or 3, but got length: {len(patch_size)} from: {patch_size}"
            )

        if mode == "3D":
            predict_fn = self._sliding_window_predict3D
        elif mode == "2D":
            predict_fn = self._sliding_window_predict2D

        pred = predict_fn(data, patch_size, overlap, batch_size)
        if mirror:
            pred += torch.flip(
                predict_fn(torch.flip(data, (2,)), patch_size, overlap, batch_size), (2,)
            )
            pred += torch.flip(
                predict_fn(torch.flip(data, (3,)), patch_size, overlap, batch_size), (3,)
            )
            pred += torch.flip(
                predict_fn(torch.flip(data, (2, 3)), patch_size, overlap, batch_size), (2, 3)
            )
            div = 4
            if mode == "3D":
                pred += torch.flip(
                    predict_fn(torch.flip(data, (4,)), patch_size, overlap, batch_size), (4,)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (2, 4)), patch_size, overlap, batch_size), (2, 4)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (3, 4)), patch_size, overlap, batch_size), (3, 4)
                )
                pred += torch.flip(
                    predict_fn(torch.flip(data, (2, 3, 4)), patch_size, overlap, batch_size),
                    (2, 3, 4),
                )
                div += 4
            pred /= div
        return pred

    def _full_image_predict(self, data):
        """
        Standard prediction used in cases where models predict on full-size images.
        This is opposed to patch-based predictions where we use a sliding window approach to generate
        full size predictions.
        """
        return self.forward(data)

    def predict_windows(self, data, canvas, windows, batch_size):
        """Predict the windows (index tuples into data and canvas) batch_size at a time and add them to canvas."""
        for start in range(0, len(windows), batch_size):
            chunk = windows[start : start + batch_size]
            patches = torch.cat([data[window] for window in chunk])
            out = self.forward(patches)
            for i, window in enumerate(chunk):
                canvas[window] += out[i : i + 1]
        return canvas

    def _sliding_window_predict3D(self, data, patch_size, overlap, batch_size=1):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=data.device,
        )

        x_steps, y_steps, z_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )
        px, py, pz = patch_size

        windows = []
        for xs in x_steps:
            for ys in y_steps:
                for zs in z_steps:
                    windows.append(
                        (slice(None), slice(None), slice(xs, xs + px), slice(ys, ys + py), slice(zs, zs + pz))
                    )
        return self.predict_windows(data, canvas, windows, batch_size)

    def _sliding_window_predict2D(self, data, patch_size, overlap, batch_size=1):
        """
        Sliding window prediction implementation
        """
        canvas = torch.zeros(
            (1, self.num_classes, *data.shape[2:]),
            device=data.device,
        )

        px, py = patch_size

        # If we have 5 dimensions we are working with 3D data, and need to predict each slice.
        if len(data.shape) == 5:
            x_steps, y_steps = get_steps_for_sliding_window(
                data.shape[3:], patch_size, overlap
            )
            windows = []
            for idx in range(data.shape[2]):
                for xs in x_steps:
                    for ys in y_steps:
                        windows.append((slice(None), slice(None), idx, slice(xs, xs + px), slice(ys, ys + py)))
            return self.predict_windows(data, canvas, windows, batch_size)

        # else we proceed with the data as 2D
        x_steps, y_steps = get_steps_for_sliding_window(
            data.shape[2:], patch_size, overlap
        )

        windows = []
        for xs in x_steps:
            for ys in y_steps:
                windows.append((slice(None), slice(None), slice(xs, xs + px), slice(ys, ys + py)))
        return self.predict_windows(data, canvas, windows, batch_size)
