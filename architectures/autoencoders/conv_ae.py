import torch
from architectures import common_blocks
from architectures import encoders
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

BOT_MID_CHANNELS = [512, 256, 128, 64, 32, 1]
DOWNSAMPLE_FACTOR = 16


class ConvBottleneck(torch.nn.Module):
    """
    The encoder for the convolutional autoencoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=False):
        """
        :param input_res: the input resolution of the model (typically [256, 512] or [512, 1024]
        :param latent_size: the desired latent size of the bottleneck
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to utilise bottleneck compression
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(ConvBottleneck, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        # self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_channels = 256
        self.bottleneck_mid_channels = bottleneck_channels
        self.latent_size = latent_size
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck

        self.bridge = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.build_bottleneck()
        self.build_bridge()

    def build_bridge(self):
        if self.compress_bottleneck:
            self.bridge.add_module("bridge(downsample1)",
                                   common_blocks.ERFNetDownsampler(in_channels=256,
                                                                 out_channels=512))
            # self.bridge.add_module("bridge(downsample2",
            #                        common_blocks.ERFNetDownsampler(in_channels=450,
            #                                                      out_channels=512))

    def build_bottleneck(self):
        """
        Build the components related to the autoencoder bottleneck
        """
        if self.compress_bottleneck:
            self.bottleneck.add_module("stage(bottleneck_down)",
                                       common_blocks.BottleneckReduction(self.bottleneck_mid_channels))

        if self.flatten_bottleneck:
            self.bottleneck.add_module("stage(flatten)", torch.nn.Flatten())
            self.bottleneck.add_module("stage(linear)",
                                       torch.nn.Linear(self.bottleneck_h * self.bottleneck_w * self.bottleneck_channels,
                                                       self.latent_size))
            self.bottleneck.add_module("stage(1DBatchNorm)", torch.nn.BatchNorm1d(num_features=self.latent_size))
            self.bottleneck.add_module("stage(ReLU)", torch.nn.ReLU())

    def forward(self, x):
        # x = self.bridge(x)

        return self.bottleneck(x)


class ConvDecoder(torch.nn.Module):
    """
    The decoder of the convolutional autoencoder
    """
    def __init__(self,
                 input_res,
                 num_layers,
                 latent_size,
                 use_residual,
                 upsample_factor,
                 input_channels=256,
                 dropout_rate=None,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=None):
        """
        :param input_res: the input resolution of the model (typically [256, 512] or [512, 1024]
        :param num_layers: the desired number of layers for the convolutional autoencoder
        :param latent_size: the desired latent size of the bottleneck
        :param use_residual: whether or not to use residual modules
        :param input_channels: the number of input channels (usually 3 for RGB images)
        :param dropout_rate: the desired dropout rate to use
        :param upsample_factor: the scale to upsample
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to utilise bottleneck compression
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(ConvDecoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.enc_out_channels, self.dec_out_channels, self.dilations = encoders.get_layer_size(num_layers, use_residual)

        self.latent_size = latent_size
        self.in_channels = input_channels
        self.upsample_factor = upsample_factor
        self.dropout = dropout_rate

        self.encoder_length = len(self.enc_out_channels)
        self.encoder_start = 1
        self.decoder_start = int(self.encoder_length + self.encoder_start)
        self.decoder_end = int(self.decoder_start + self.encoder_length - 1)
        self.use_residual = use_residual
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck
        # self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_channels = 256
        self.bot_mid_channels = bottleneck_channels

        self.decoder = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.bridge = torch.nn.Sequential()

        self.unpack_bridge()
        self.unpack_bottleneck()
        self.build_decoder()

    def unpack_bottleneck(self):
        """
        Build the components related to the autoencoder bottleneck
        """
        if self.flatten_bottleneck:
            self.bottleneck.add_module("stage(linear)",
                                       torch.nn.Linear(self.latent_size,
                                                       self.bottleneck_h * self.bottleneck_w * self.bottleneck_channels))
            self.bottleneck.add_module("stage(reshape)", common_blocks.Reshape(
                shape=(-1, self.bottleneck_channels, self.bottleneck_h, self.bottleneck_w)))

        if self.compress_bottleneck:
            self.bottleneck.add_module("stage(bottleneck_up)",
                                       common_blocks.BottleneckReduction(channels=self.bot_mid_channels,
                                                                         upsample=True))

    def unpack_bridge(self):
        """
        Unpack the bridge to match original encoder output EDANet dimensions
        """
        if self.compress_bottleneck:
            self.bridge.add_module("stage(upsample1)",
                                   common_blocks.ConvBlock(in_channels=512,
                                                           out_channels=256,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

            # self.bridge.add_module("stage(upsample2)",
            #                        common_blocks.ConvBlock(in_channels=450,
            #                                                out_channels=256,
            #                                                kernel_size=1,
            #                                                stride=1,
            #                                                padding=0,
            #                                                use_bn=True,
            #                                                activation=True,
            #                                                upsample_factor=self.upsample_factor))

    def build_decoder(self):
        """
        Iteratively construct the decoder
        """
        for j, out_channels in enumerate(self.dec_out_channels, start=self.decoder_start):
            block_dilation = list(reversed(self.dilations))[j - self.decoder_start]

            if self.use_residual and self.in_channels / out_channels == 1:
                unit = common_blocks.ERFNetFactorisedResidualBlock(channels=out_channels,
                                                                   dilation=block_dilation,
                                                                   dropout_rate=self.dropout)

            else:
                if self.in_channels / out_channels == 2 or j == self.decoder_end:
                    upsample = self.upsample_factor
                else:
                    upsample = None

                unit = common_blocks.ConvBlock(in_channels=self.in_channels,
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               dilation=block_dilation,
                                               use_bn=False if j == self.decoder_end else True,
                                               activation=False if j == self.decoder_end else True,
                                               upsample_factor=upsample,
                                               dropout_rate=None if j == self.decoder_end else self.dropout)

            self.in_channels = out_channels
            self.decoder.add_module(f"stage{j}", unit)

        self.decoder.add_module("stage(tanh)", torch.nn.Tanh())

    def forward(self, x):
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        # x = self.bridge(x)
        x = self.decoder(x)
        return x


class ConvAE(torch.nn.Module):
    """
    A convolutional autoencoder comprising of 4, 6, 8, 10 layers. Implementations of residual modules and bottleneck
    compressions also included.
    """
    def __init__(self,
                 input_res,
                 num_layers,
                 latent_size,
                 use_residual,
                 upsample_factor=2,
                 input_channels=3,
                 dropout_rate=None,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=BOT_MID_CHANNELS):
        """
        :param input_res: the input resolution of the model (typically [256, 512] or [512, 1024]
        :param num_layers: the desired number of layers for the convolutional autoencoder
        :param latent_size: the desired latent size of the bottleneck
        :param use_residual: whether or not to use residual modules
        :param input_channels: the number of input channels (usually 3 for RGB images)
        :param dropout_rate: the desired dropout rate to use
        :param upsample_factor: the scale to upsample
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to utilise bottleneck compression
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(ConvAE, self).__init__()

        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck

        self.encoder = encoders.ConvEncoder(input_res=input_res,
                                            num_layers=num_layers,
                                            use_residual=use_residual,
                                            input_channels=input_channels,
                                            dropout_rate=dropout_rate)

        self.bottleneck = ConvBottleneck(input_res=input_res,
                                         latent_size=latent_size,
                                         flatten_bottleneck=flatten_bottleneck,
                                         compress_bottleneck=compress_bottleneck,
                                         bottleneck_channels=bottleneck_channels)

        self.decoder = ConvDecoder(input_res=input_res,
                                   num_layers=num_layers,
                                   latent_size=latent_size,
                                   upsample_factor=upsample_factor,
                                   use_residual=use_residual,
                                   dropout_rate=dropout_rate,
                                   flatten_bottleneck=flatten_bottleneck,
                                   compress_bottleneck=compress_bottleneck,
                                   bottleneck_channels=bottleneck_channels)

    def forward(self, x):
        x = self.encoder(x)

        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)
        x = self.decoder(x)

        return x

