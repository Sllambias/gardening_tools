"""
Cross Entropy and Dice Loss from the paper:
Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al.
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
    Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
"""

import torch
from torch import nn
from gardening_tools.modules.losses.Dice import SoftDiceLoss
from gardening_tools.modules.losses.CE import CE


class DiceCE(nn.Module):
    def __init__(
        self,
        soft_dice_kwargs={},
        ce_kwargs={},
        weight_ce=1,
        weight_dice=1,
        log_dice=False,
        ignore_label=None,
    ):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DiceCE, self).__init__()
        if ignore_label is not None:
            ce_kwargs["reduction"] = "none"
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = CE()

        self.ignore_label = ignore_label

        self.dc = SoftDiceLoss(**soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, "not implemented for one hot encoding"
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = (
            self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        )
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result
