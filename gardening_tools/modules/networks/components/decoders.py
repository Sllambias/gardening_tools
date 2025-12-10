import torch
import torch.nn as nn
from gardening_tools.modules.networks.components.blocks import (
    DoubleConvDropoutNormNonlin,
)


class UNetDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        starting_filters: int = 64,
        basic_block=DoubleConvDropoutNormNonlin,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        deep_supervision=False,
        dropout_in_decoder=False,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        upsample_op=torch.nn.ConvTranspose3d,
        use_skip_connections=True,
        weightInitializer=None,
    ) -> None:
        super().__init__()

        # Task specific
        self.num_classes = output_channels
        self.filters = starting_filters

        # Model parameters
        self.basic_block = basic_block
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.deep_supervision = deep_supervision
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.use_skip_connections = use_skip_connections
        self.upsample = upsample_op

        self.upsample1 = self.upsample(
            self.filters * 16, self.filters * 8, kernel_size=2, stride=2
        )
        self.decoder_conv1 = self.basic_block(
            input_channels=self.filters * (16 if self.use_skip_connections else 8),
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

        self.upsample2 = self.upsample(
            self.filters * 8, self.filters * 4, kernel_size=2, stride=2
        )
        self.decoder_conv2 = self.basic_block(
            input_channels=self.filters * (8 if self.use_skip_connections else 4),
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

        self.upsample3 = self.upsample(
            self.filters * 4, self.filters * 2, kernel_size=2, stride=2
        )
        self.decoder_conv3 = self.basic_block(
            input_channels=self.filters * (4 if self.use_skip_connections else 2),
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

        self.upsample4 = self.upsample(
            self.filters * 2, self.filters, kernel_size=2, stride=2
        )
        self.decoder_conv4 = self.basic_block(
            input_channels=self.filters * (2 if self.use_skip_connections else 1),
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

        self.out_conv = self.conv_op(self.filters, self.num_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(
                self.filters * 16, self.num_classes, kernel_size=1
            )
            self.ds_out_conv1 = self.conv_op(
                self.filters * 8, self.num_classes, kernel_size=1
            )
            self.ds_out_conv2 = self.conv_op(
                self.filters * 4, self.num_classes, kernel_size=1
            )
            self.ds_out_conv3 = self.conv_op(
                self.filters * 2, self.num_classes, kernel_size=1
            )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, xs):
        x_enc = xs[4]

        if self.use_skip_connections:
            x5 = torch.cat([self.upsample1(x_enc), xs[3]], dim=1)
            x5 = self.decoder_conv1(x5)

            x6 = torch.cat([self.upsample2(x5), xs[2]], dim=1)
            x6 = self.decoder_conv2(x6)

            x7 = torch.cat([self.upsample3(x6), xs[1]], dim=1)
            x7 = self.decoder_conv3(x7)

            x8 = torch.cat([self.upsample4(x7), xs[0]], dim=1)
            x8 = self.decoder_conv4(x8)
        else:
            x5 = self.decoder_conv1(self.upsample1(x_enc))
            x6 = self.decoder_conv2(self.upsample2(x5))
            x7 = self.decoder_conv3(self.upsample3(x6))
            x8 = self.decoder_conv4(self.upsample4(x7))

        # We only want to do multiple outputs during training, therefore it is only enabled
        # when grad is also enabled because that means we're training. And if for some reason
        # grad is enabled and you're not training, then there's other, bigger problems.
        if self.deep_supervision and torch.is_grad_enabled():
            ds0 = self.ds_out_conv0(xs[4])
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.out_conv(x8)
            return [ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x8)

        return logits


class ResidualUNetDecoder(nn.Module):
    def __init__(
        self,
        output_channels: int = 1,
        features_per_stage: list = [],
        n_conv_per_stage: list = [],
        basic_block=DoubleConvDropoutNormNonlin,
        conv_op=nn.Conv3d,
        conv_kwargs={
            "kernel_size": 3,
            "stride": 1,
            "bias": True,
        },
        deep_supervision=False,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True, "momentum": 0.1},
        dropout_op=nn.Dropout3d,
        dropout_op_kwargs={"p": 0.0, "inplace": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        stride_for_transpose_conv: int = 2,
        upsample_op=torch.nn.ConvTranspose3d,
        use_skip_connections=True,
        weightInitializer=None,
    ) -> None:
        super().__init__()

        # Task specific
        self.num_classes = output_channels
        self.features_per_stage = features_per_stage
        self.n_conv_per_stage = n_conv_per_stage

        # Model parameters
        self.basic_block = basic_block
        self.conv_op = conv_op
        self.conv_kwargs = conv_kwargs
        self.deep_supervision = deep_supervision
        self.norm_op_kwargs = norm_op_kwargs
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.weightInitializer = weightInitializer
        self.use_skip_connections = use_skip_connections
        self.upsample = upsample_op

        self.upsample1 = self.upsample(
            self.features_per_stage[0],
            self.features_per_stage[1],
            kernel_size=stride_for_transpose_conv,
            stride=stride_for_transpose_conv,
        )

        self.decoder_conv1 = self.basic_block(
            input_channels=self.features_per_stage[1]
            * (2 if self.use_skip_connections else 1),
            output_channels=self.features_per_stage[1],
            num_layers=self.n_conv_per_stage[0],
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample2 = self.upsample(
            self.features_per_stage[1],
            self.features_per_stage[2],
            kernel_size=stride_for_transpose_conv,
            stride=stride_for_transpose_conv,
        )
        self.decoder_conv2 = self.basic_block(
            input_channels=self.features_per_stage[2]
            * (2 if self.use_skip_connections else 1),
            output_channels=self.features_per_stage[2],
            num_layers=self.n_conv_per_stage[1],
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample3 = self.upsample(
            self.features_per_stage[2],
            self.features_per_stage[3],
            kernel_size=stride_for_transpose_conv,
            stride=stride_for_transpose_conv,
        )
        self.decoder_conv3 = self.basic_block(
            input_channels=self.features_per_stage[3]
            * (2 if self.use_skip_connections else 1),
            output_channels=self.features_per_stage[3],
            num_layers=self.n_conv_per_stage[2],
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample4 = self.upsample(
            self.features_per_stage[3],
            self.features_per_stage[4],
            kernel_size=stride_for_transpose_conv,
            stride=stride_for_transpose_conv,
        )
        self.decoder_conv4 = self.basic_block(
            input_channels=self.features_per_stage[4]
            * (2 if self.use_skip_connections else 1),
            output_channels=self.features_per_stage[4],
            num_layers=self.n_conv_per_stage[3],
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.upsample5 = self.upsample(
            self.features_per_stage[4],
            self.features_per_stage[5],
            kernel_size=stride_for_transpose_conv,
            stride=stride_for_transpose_conv,
        )

        self.decoder_conv5 = self.basic_block(
            input_channels=self.features_per_stage[5]
            * (2 if self.use_skip_connections else 1),
            output_channels=self.features_per_stage[5],
            num_layers=self.n_conv_per_stage[4],
            conv_op=self.conv_op,
            conv_kwargs=self.conv_kwargs,
            norm_op=self.norm_op,
            norm_op_kwargs=self.norm_op_kwargs,
            dropout_op=self.dropout_op,
            dropout_op_kwargs=self.dropout_op_kwargs,
            nonlin=self.nonlin,
            nonlin_kwargs=self.nonlin_kwargs,
        )

        self.out_conv = self.conv_op(
            self.features_per_stage[5], self.num_classes, kernel_size=1
        )

        if self.deep_supervision:
            self.ds_out_conv0 = self.conv_op(
                self.features_per_stage[0], self.num_classes, kernel_size=1
            )
            self.ds_out_conv1 = self.conv_op(
                self.features_per_stage[1], self.num_classes, kernel_size=1
            )
            self.ds_out_conv2 = self.conv_op(
                self.features_per_stage[2], self.num_classes, kernel_size=1
            )
            self.ds_out_conv3 = self.conv_op(
                self.features_per_stage[3], self.num_classes, kernel_size=1
            )
            self.ds_out_conv4 = self.conv_op(
                self.features_per_stage[4], self.num_classes, kernel_size=1
            )

        if self.weightInitializer is not None:
            print("initializing weights")
            self.apply(self.weightInitializer)

    def forward(self, xs):
        x_enc = xs[5]

        if self.use_skip_connections:
            x5 = torch.cat([self.upsample1(x_enc), xs[4]], dim=1)
            x5 = self.decoder_conv1(x5)

            x6 = torch.cat([self.upsample2(x5), xs[3]], dim=1)
            x6 = self.decoder_conv2(x6)

            x7 = torch.cat([self.upsample3(x6), xs[2]], dim=1)
            x7 = self.decoder_conv3(x7)

            x8 = torch.cat([self.upsample4(x7), xs[1]], dim=1)
            x8 = self.decoder_conv4(x8)

            x9 = torch.cat([self.upsample5(x8), xs[0]], dim=1)
            x9 = self.decoder_conv5(x9)
        else:
            x5 = self.decoder_conv1(self.upsample1(x_enc))
            x6 = self.decoder_conv2(self.upsample2(x5))
            x7 = self.decoder_conv3(self.upsample3(x6))
            x8 = self.decoder_conv4(self.upsample4(x7))
            x9 = self.decoder_conv5(self.upsample5(x8))

        # We only want to do multiple outputs during training, therefore it is only enabled
        # when grad is also enabled because that means we're training. And if for some reason
        # grad is enabled and you're not training, then there's other, bigger problems.
        if self.deep_supervision and torch.is_grad_enabled():
            ds0 = self.ds_out_conv0(xs[4])
            ds1 = self.ds_out_conv1(x5)
            ds2 = self.ds_out_conv2(x6)
            ds3 = self.ds_out_conv3(x7)
            ds4 = self.ds_out_conv4(x8)
            ds5 = self.out_conv(x9)
            return [ds5, ds4, ds3, ds2, ds1, ds0]

        logits = self.out_conv(x9)

        return logits
