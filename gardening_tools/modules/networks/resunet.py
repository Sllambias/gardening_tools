import logging
import torch
import torch.nn as nn
from gardening_tools.modules.networks.BaseNet import BaseNet
from gardening_tools.modules.networks.components.blocks import (
    ResidualBlock,
    MultiLayerConvDropoutNormNonlin,
)
from gardening_tools.modules.networks.components.encoders import ResidualUNetEncoder
from gardening_tools.modules.networks.components.decoders import ResidualUNetDecoder
from typing import List, Tuple, Type, Union


class ResidualEncoderUNet(BaseNet):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dimensions: str,
        kernel_size: int,
        stride: int,
        features_per_stage: list,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = True,
        deep_supervision: bool = False,
        encoder_basic_block: Type[ResidualBlock] = ResidualBlock,
        decoder_basic_block: Type[MultiLayerConvDropoutNormNonlin] = MultiLayerConvDropoutNormNonlin,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        use_skip_connections: bool = True,
    ):
        super().__init__()

        if dimensions == "2D":
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            upsample_op = torch.nn.ConvTranspose2d
            pool_op = nn.AvgPool2d
        elif dimensions == "3D":
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
            upsample_op = torch.nn.ConvTranspose3d
            pool_op = nn.AvgPool3d
        else:
            logging.warn("Uuh, dimensions not in ['2D', '3D']")

        self.num_classes = output_channels

        self.stem_weight_name = "encoder.stem.conv1.conv.weight"

        self.encoder = ResidualUNetEncoder(
            input_channels=input_channels,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_size=kernel_size,
            stride=stride,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=encoder_basic_block,
            pool_op=pool_op,
        )
        self.decoder = ResidualUNetDecoder(
            output_channels=output_channels,
            features_per_stage=features_per_stage[::-1],
            n_conv_per_stage=n_conv_per_stage_decoder,
            basic_block=decoder_basic_block,
            deep_supervision=deep_supervision,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "bias": conv_bias,
            },
            stride_for_transpose_conv=stride,
            upsample_op=upsample_op,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            use_skip_connections=use_skip_connections,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def forward_with_features(self, x):
        skips = self.encoder(x)
        output = self.decoder(skips)
        return output, skips[-1]
