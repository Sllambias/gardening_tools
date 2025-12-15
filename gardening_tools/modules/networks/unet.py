import logging
import torch
import torch.nn as nn
from gardening_tools.modules.networks.components.blocks import (
    MultiLayerConvDropoutNormNonlin,
)
from gardening_tools.modules.networks.components.heads import ClsRegHead
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.decoders import UNetDecoder
from gardening_tools.modules.networks.components.encoders import UNetEncoder


class UNet(BaseNet):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encoder: nn.Module = UNetEncoder,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder: nn.Module = UNetDecoder,
        decoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        dimensions: str = "3D",
        starting_filters: int = 32,
        use_skip_connections: bool = True,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.stem_weight_name = None
        if dimensions == "2D":
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            pool_op = nn.MaxPool2d
            upsample_op = torch.nn.ConvTranspose2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            upsample_op = torch.nn.ConvTranspose3d
        else:
            logging.warn("Uuh, dimensions not in ['2D', '3D']")

        self.encoder = encoder(
            basic_block=encoder_basic_block,
            conv_op=conv_op,
            dropout_op=dropout_op,
            input_channels=input_channels,
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
        )
        self.decoder = decoder(
            basic_block=decoder_basic_block,
            conv_op=conv_op,
            dropout_op=dropout_op,
            norm_op=norm_op,
            output_channels=output_channels,
            starting_filters=starting_filters,
            upsample_op=upsample_op,
            use_skip_connections=use_skip_connections,
            deep_supervision=deep_supervision,
        )
        self.num_classes = output_channels

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)

    def forward_with_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, skips[-1]


class UNetCLSREG(BaseNet):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        encoder: nn.Module = UNetEncoder,
        encoder_basic_block=MultiLayerConvDropoutNormNonlin.get_block_constructor(2),
        decoder: nn.Module = ClsRegHead,
        dimensions: str = "3D",
        starting_filters: int = 32,
        deep_supervision: bool = False,
    ):
        super().__init__()
        assert deep_supervision is False, "UNetCLSREG does not support deep supervision"
        self.stem_weight_name = None

        if dimensions == "2D":
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d
            pool_op = nn.MaxPool2d
            clsreg_pool_op = nn.AdaptiveAvgPool2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d
            pool_op = nn.MaxPool3d
            clsreg_pool_op = nn.AdaptiveAvgPool3d
        else:
            logging.warn("Uuh, dimensions not in ['2D', '3D']")

        self.encoder = encoder(
            basic_block=encoder_basic_block,
            conv_op=conv_op,
            dropout_op=dropout_op,
            input_channels=input_channels,
            norm_op=norm_op,
            pool_op=pool_op,
            starting_filters=starting_filters,
        )
        self.decoder = decoder(
            pool_op=clsreg_pool_op,
            input_channels=starting_filters * 16,
            output_channels=output_channels,
            dropout_rate=0.2,
        )
        self.num_classes = output_channels

    def forward(self, x):
        enc = self.encoder(x)
        return self.decoder(enc)
