import torch
import torch.nn as nn
from gardening_tools.modules.networks.components.blocks import (
    DoubleConvDropoutNormNonlin,
    MultiLayerConvDropoutNormNonlin,
    ResidualBlock,
    StackedResidualBlocks,
)
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import List, Tuple, Type, Union


class UNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        basic_block=DoubleConvDropoutNormNonlin,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        weightInitializer=None,
        pool_op=nn.MaxPool3d,
        starting_filters: int = 64,
    ) -> None:
        super().__init__()

        # Task specific
        self.filters = starting_filters

        # Model parameters
        self.basic_block = basic_block
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.pool_op = pool_op

        self.in_conv = self.basic_block(
            input_channels=input_channels,
            output_channels=self.filters,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool1 = self.pool_op(2)
        self.encoder_conv1 = self.basic_block(
            input_channels=self.filters,
            output_channels=self.filters * 2,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool2 = self.pool_op(2)
        self.encoder_conv2 = self.basic_block(
            input_channels=self.filters * 2,
            output_channels=self.filters * 4,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool3 = self.pool_op(2)
        self.encoder_conv3 = self.basic_block(
            input_channels=self.filters * 4,
            output_channels=self.filters * 8,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.pool4 = self.pool_op(2)
        self.encoder_conv4 = self.basic_block(
            input_channels=self.filters * 8,
            output_channels=self.filters * 16,
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, x):
        x0 = self.in_conv(x)

        x1 = self.pool1(x0)
        x1 = self.encoder_conv1(x1)

        x2 = self.pool2(x1)
        x2 = self.encoder_conv2(x2)

        x3 = self.pool3(x2)
        x3 = self.encoder_conv3(x3)

        x4 = self.pool4(x3)
        x4 = self.encoder_conv4(x4)

        return [x0, x1, x2, x3, x4]


class ResidualUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        pool_op: Union[nn.AvgPool2d, nn.AvgPool3d],
        kernel_size: int,
        stride: int,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        block: Type[ResidualBlock] = ResidualBlock,
    ):
        super().__init__()

        stem_channels = features_per_stage[0]

        self.stem = MultiLayerConvDropoutNormNonlin(
            input_channels=input_channels,
            output_channels=stem_channels,
            num_layers=1,
            conv_op=conv_op,
            conv_kwargs={
                "kernel_size": kernel_size,
                "stride": 1,
                "bias": conv_bias,
            },
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        input_channels = stem_channels

        stages = []
        for s in range(len(features_per_stage)):
            stage = StackedResidualBlocks(
                n_blocks=n_blocks_per_stage[s],
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=features_per_stage[s],
                kernel_size=kernel_size,
                initial_stride=stride if s != 0 else 1,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                block=block,
                pool_op=pool_op,
            )

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret
