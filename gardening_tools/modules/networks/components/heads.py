import torch
import torch.nn as nn


class ClsRegHead(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op, dropout_rate=0.0):
        super().__init__()
        if issubclass(pool_op, nn.AdaptiveAvgPool3d):
            self.global_pool = pool_op((1, 1, 1))
        elif issubclass(pool_op, nn.AdaptiveAvgPool2d):
            self.global_pool = pool_op((1, 1))
        self.fc = nn.Linear(input_channels, output_channels)
        self.drop = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = x[-1]
        x = self.global_pool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
