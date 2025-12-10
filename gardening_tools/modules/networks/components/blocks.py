import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from typing import List, Tuple, Type, Union

__all__ = [
    "ConvDropoutNormNonlin",
    "ConvDropoutNorm",
    "DoubleConvDropoutNormNonlin",
    "DoubleLayerResBlock",
    "MultiLayerConvDropoutNormNonlin",
    "MultiLayerResBlock",
    "ResidualBlock",
    "StackedResidualBlocks",
]


class ConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        modules = []
        self.conv = self.conv_op(
            input_channels,
            output_channels,
            padding=conv_kwargs["kernel_size"] // 2,
            dilation=1,
            **self.conv_kwargs,
        )
        modules.append(self.conv)

        if self.dropout_op is not None:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
            modules.append(self.dropout)
        if self.norm_op is not None:
            self.norm_op = self.norm_op(output_channels, **self.norm_op_kwargs)
            modules.append(self.norm_op)
        if self.nonlin is not None:
            self.nonlin = self.nonlin(**self.nonlin_kwargs)
            modules.append(self.nonlin)

        self.all_modules = nn.Sequential(*modules)

    def forward(self, x):
        return self.all_modules(x)


class ConvDropoutNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.norm(x)


class DoubleConvDropoutNormNonlin(nn.Module):
    """
    2/3D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )
        self.conv2 = ConvDropoutNormNonlin(
            output_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MultiLayerConvDropoutNormNonlin(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert num_layers >= 1, "Number of layers must be at least 1, got {}".format(
            num_layers
        )
        self.num_layers = num_layers

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        for layer in range(2, num_layers + 1):
            setattr(
                self,
                f"conv{layer}",
                ConvDropoutNormNonlin(
                    output_channels,
                    output_channels,
                    self.conv_op,
                    self.conv_kwargs,
                    self.norm_op,
                    self.norm_op_kwargs,
                    self.dropout_op,
                    self.dropout_op_kwargs,
                    self.nonlin,
                    self.nonlin_kwargs,
                ),
            )

    def forward(self, x):
        x = self.conv1(x)
        for layer in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{layer}")(x)

        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerConvDropoutNormNonlin(
                input_channels, output_channels, num_layers=n_layers, **kwargs
            )

        return _block


class DoubleLayerResBlock(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert conv_kwargs["dilation"] == 1, "Dilation must be 1 for residual blocks"

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        if (conv_kwargs["stride"] != 1) or (input_channels != output_channels):
            self.downsample_skip = nn.Sequential(
                conv_op(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    stride=conv_kwargs["stride"],
                    bias=False,
                ),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.downsample_skip = lambda x: x

        self.conv2 = ConvDropoutNorm(
            output_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        self.final_nonlin = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += self.downsample_skip(residual)
        x = self.final_nonlin(x)

        return x


class MultiLayerResBlock(nn.Module):
    """
    2D Convolutional layers
    Arguments:
    num_in_filters {int} -- number of input filters
    num_out_filters {int} -- number of output filters
    num_layers {int} -- number of conv layers, must be at least 1
    kernel_size {tuple} -- size of the convolving kernel
    stride {tuple} -- stride of the convolution (default: {(1, 1)})
    activation {str} -- activation function (default: {'relu'})
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        num_layers=2,
        conv_op=nn.Conv2d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        norm_op=nn.BatchNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout2d,
        dropout_op_kwargs={"p": 0.5, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
    ):
        super().__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        assert num_layers >= 1, "Number of layers must be at least 1, got {}".format(
            num_layers
        )

        assert conv_kwargs["stride"] == 1, "Stride must be 1 for residual blocks"
        assert conv_kwargs["conv_dilation"] == 1, (
            "Dilation must be 1 for residual blocks"
        )

        self.num_layers = num_layers

        self.conv1 = ConvDropoutNormNonlin(
            input_channels,
            output_channels,
            self.conv_op,
            self.conv_kwargs,
            self.norm_op,
            self.norm_op_kwargs,
            self.dropout_op,
            self.dropout_op_kwargs,
            self.nonlin,
            self.nonlin_kwargs,
        )

        if (conv_kwargs["stride"] != 1) or (input_channels != output_channels):
            self.downsample_skip = nn.Sequential(
                conv_op(
                    input_channels,
                    output_channels,
                    kernel_size=1,
                    padding=0,
                    stride=conv_kwargs["stride"],
                    bias=False,
                ),
                norm_op(output_channels, **norm_op_kwargs),
            )
        else:
            self.downsample_skip = lambda x: x

        for layer in range(2, num_layers + 1):
            if layer < num_layers:
                setattr(
                    self,
                    f"conv{layer}",
                    ConvDropoutNormNonlin(
                        output_channels,
                        output_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                )
            else:
                # Last layer does not have activation, is added after residual
                setattr(
                    self,
                    f"conv{layer}",
                    ConvDropoutNorm(
                        output_channels,
                        output_channels,
                        self.conv_op,
                        self.conv_kwargs,
                        self.norm_op,
                        self.norm_op_kwargs,
                        self.dropout_op,
                        self.dropout_op_kwargs,
                        self.nonlin,
                        self.nonlin_kwargs,
                    ),
                )

        self.final_nonlin = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        for layer in range(2, self.num_layers + 1):
            x = getattr(self, f"conv{layer}")(x)

        x += self.downsample_skip(residual)
        x = self.final_nonlin(x)

        return x

    @staticmethod
    def get_block_constructor(n_layers):
        def _block(input_channels, output_channels, **kwargs):
            return MultiLayerResBlock(
                input_channels, output_channels, num_layers=n_layers, **kwargs
            )

        return _block


class ResidualBlock(nn.Module):
    def __init__(
        self,
        conv_op: Type[_ConvNd],
        pool_op: Union[nn.AvgPool2d, nn.AvgPool3d],
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int,
        conv_bias: bool = True,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        self.conv1 = ConvDropoutNormNonlin(
            input_channels=input_channels,
            output_channels=output_channels,
            conv_op=conv_op,
            conv_kwargs={
                "stride": stride,
                "kernel_size": kernel_size,
                "bias": conv_bias,
            },
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        self.conv2 = ConvDropoutNormNonlin(
            input_channels=output_channels,
            output_channels=output_channels,
            conv_op=conv_op,
            conv_kwargs={"stride": 1, "kernel_size": kernel_size, "bias": conv_bias},
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=None,
            nonlin_kwargs=None,
        )

        self.nonlin2 = nonlin(**nonlin_kwargs) if nonlin is not None else lambda x: x

        has_stride = stride != 1
        requires_projection = input_channels != output_channels

        if has_stride or requires_projection:
            ops = []
            if stride != 1:
                ops.append(pool_op(kernel_size=stride, stride=stride))
            if requires_projection:
                ops.append(
                    ConvDropoutNormNonlin(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        conv_op=conv_op,
                        conv_kwargs={"stride": 1, "kernel_size": 1, "bias": False},
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        dropout_op=None,
                        dropout_op_kwargs=None,
                        nonlin=None,
                        nonlin_kwargs=None,
                    )
                )
            self.skip = nn.Sequential(*ops)
        else:
            self.skip = lambda x: x

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv2(self.conv1(x))
        out += residual
        return self.nonlin2(out)


class StackedResidualBlocks(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[_ConvNd],
        pool_op: Union[nn.AvgPool2d, nn.AvgPool3d],
        input_channels: int,
        output_channels: Union[int, List[int], Tuple[int, ...]],
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        block: Type[ResidualBlock] = ResidualBlock,
    ):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * n_blocks

        blocks = nn.Sequential(
            block(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=output_channels[0],
                kernel_size=kernel_size,
                stride=initial_stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                pool_op=pool_op,
            ),
            *[
                block(
                    conv_op=conv_op,
                    input_channels=output_channels[n - 1],
                    output_channels=output_channels[n],
                    kernel_size=kernel_size,
                    stride=1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    pool_op=pool_op,
                )
                for n in range(1, n_blocks)
            ],
        )

        self.blocks = blocks
        self.output_channels = output_channels[-1]

    def forward(self, x):
        return self.blocks(x)
