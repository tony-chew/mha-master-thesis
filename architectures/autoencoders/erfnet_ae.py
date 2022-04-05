import torch
from architectures import common_blocks
from architectures import encoders
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

CHANNELS = [16, 64, 128]
UP_CHANNELS = [64, 16, 3]
# DILATIONS = [[1], [1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16]]
# DROPOUT_RATES = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3]]
DILATIONS = [[1], [1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16, 2, 4, 8, 16]]
DROPOUT_RATES = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]

BOT_CHANNELS = [512, 256, 196, 128, 96, 64, 32, 1]
DOWNSAMPLE_FACTOR = 32


class ERFNetBottleneck(torch.nn.Module):
    """
    The bottleneck for the ERFNet fast segmentation network outfitted as an autoencoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=None):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the desired latent size for the bottleneck
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(ERFNetBottleneck, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.latent_size = latent_size
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck
        self.bottleneck_mid_channels = bottleneck_channels

        self.bridge = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.build_bottleneck()
        self.build_bridge()

    def build_bridge(self):
        if self.compress_bottleneck:
            self.bridge.add_module("bridge(downsample1)",
                                   common_blocks.ERFNetDownsampler(in_channels=128,
                                                                   out_channels=256))
            self.bridge.add_module("bridge(downsample2",
                                   common_blocks.ERFNetDownsampler(in_channels=256,
                                                                   out_channels=512))

    def build_bottleneck(self):
        """
        Build the components related to the autoencoder bottleneck
        """
        if self.compress_bottleneck:
            self.bottleneck.add_module("stage(bottleneck_down)",
                                       common_blocks.BottleneckReduction(self.bottleneck_mid_channels))

        if self.flatten_bottleneck:
            self.bottleneck.add_module("stage(flatten)", torch.nn.Flatten())
            self.bottleneck.add_module("stage(linear)", torch.nn.Linear(
                self.bottleneck_h * self.bottleneck_w * self.bottleneck_channels,
                self.latent_size))
            self.bottleneck.add_module("stage(1DBN)", torch.nn.BatchNorm1d(num_features=self.latent_size))
            self.bottleneck.add_module("stage(ReLU)", torch.nn.ReLU())

    def forward(self, x):
        x = self.bridge(x)

        return self.bottleneck(x)


class ERFNetDecoder(torch.nn.Module):
    """
    The decoder for an outfitted autoencoder based on the ERFNet segmentation network
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 up_channels,
                 dilations,
                 dropout_rates,
                 input_channels=128,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 upsample_factor=2,
                 bottleneck_channels=None):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the latent size of the bottleneck
        :param up_channels: the output channels of each downsampling stage of the encoder
        :param dilations: the dilation rates of each layer of the encoder
        :param dropout_rates: the dropout rates of each layer of the encoder
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param upsample_factor: the factor for upsampling back to original dimensions for each layer
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(ERFNetDecoder, self).__init__()

        self.channels = up_channels
        self.dilations = [elem[::-1] for elem in dilations][::-1]
        self.dropout_rates = [elem[::-1] for elem in dropout_rates][::-1]
        self.upsample_factor = upsample_factor

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.latent_size = latent_size
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck
        self.bottleneck_mid_channels = bottleneck_channels
        self.input_channels = input_channels

        self.decoder = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.bridge = torch.nn.Sequential()

        self.unpack_bottleneck()
        self.unpack_bridge()
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
                                       common_blocks.BottleneckReduction(channels=self.bottleneck_mid_channels,
                                                                         upsample=True))

    def unpack_bridge(self):
        """
        Unpack the bridge to match original encoder output LEDNet dimensions
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

            self.bridge.add_module("stage(upsample2)",
                                   common_blocks.ConvBlock(in_channels=256,
                                                           out_channels=128,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

    def build_decoder(self):
        """
        Build the decoder componenents related to LEDNet (reversed from the encoder for this autoencoder)
        """
        for i, out_channels in enumerate(self.channels):
            out_channels = self.channels[i]
            dropout_per_stage = self.dropout_rates[i]
            dilations_per_stage = self.dilations[i]
            stack = torch.nn.Sequential()

            for j, dilation in enumerate(dilations_per_stage):
                dilation_len = len(dilations_per_stage) - 1
                if j == dilation_len:
                    stack.add_module(f"unit {j + 1}", common_blocks.ConvBlock(in_channels=self.input_channels,
                                                                              out_channels=3 if i == 2 else out_channels,
                                                                              kernel_size=3,
                                                                              stride=1,
                                                                              padding=1,
                                                                              use_bn=False if i == 2 else True,
                                                                              activation=False if i == 2 else True,
                                                                              upsample_factor=self.upsample_factor))
                    self.input_channels = out_channels

                else:
                    stack.add_module(f"unit {j + 1}", common_blocks.ERFNetFactorisedResidualBlock(channels=self.input_channels,
                                                                                                  dilation=dilation,
                                                                                                  dropout_rate=dropout_per_stage[j]))

                self.decoder.add_module(f"stage {i + 1}", stack)

        self.decoder.add_module("stage(tanh)", torch.nn.Tanh())

    def forward(self, x):
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        x = self.bridge(x)
        x = self.decoder(x)
        return x


class ERFNetAE(torch.nn.Module):
    """
    ERFNet from 'ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation',
    https://ieeexplore.ieee.org/document/8063438 outfitted as an autoencoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 channels=CHANNELS,
                 up_channels=UP_CHANNELS,
                 dilations=DILATIONS,
                 dropout_rates=DROPOUT_RATES,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=BOT_CHANNELS,
                 **kwargs):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the latent size of the bottleneck
        :param channels: the output channels of each downsampling stage of the encoder
        :param up_channels: the output channels of each upsampling stage of the decoder
        :param dilations: the dilation rates of each layer of the encoder
        :param dropout_rates: the dropout rates of each layer of the encoder
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        """
        super(ERFNetAE, self).__init__()

        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck

        self.encoder = encoders.ERFNetEncoder(channels=channels,
                                              input_res=input_res,
                                              dilations=dilations,
                                              dropout_rates=dropout_rates)

        self.bottleneck = ERFNetBottleneck(input_res=input_res,
                                           latent_size=latent_size,
                                           flatten_bottleneck=flatten_bottleneck,
                                           compress_bottleneck=compress_bottleneck,
                                           bottleneck_channels=bottleneck_channels)

        self.decoder = ERFNetDecoder(input_res=input_res,
                                     latent_size=latent_size,
                                     up_channels=up_channels,
                                     dilations=dilations,
                                     dropout_rates=dropout_rates,
                                     flatten_bottleneck=flatten_bottleneck,
                                     compress_bottleneck=compress_bottleneck,
                                     bottleneck_channels=bottleneck_channels)

    def encoder_bottleneck(self, x):
        x = self.encoder(x)
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        return x

    def forward(self, x):
        x = self.encoder(x)

        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)
        x = self.decoder(x)

        return x