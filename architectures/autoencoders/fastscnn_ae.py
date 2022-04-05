import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

GLOBAL_FEATURE_CHANNELS = [[128, 128, 128], [96, 96, 96], [64, 64, 64]]
LEARNING_TO_DOWNSAMPLE_CHANNELS = [64, 48, 32]

BOT_CHANNELS = [ 128, 96, 64, 32, 1 ]
DOWNSAMPLE_FACTOR = 32


class FastSCNNBottleneck(torch.nn.Module):
    """
    The bottleneck for the FastSCNN fast segmentation network
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=False):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the desired latent size for the bottleneck
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(FastSCNNBottleneck, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.latent_size = latent_size
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_mid_channels = bottleneck_channels
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck

        self.bottleneck = torch.nn.Sequential()
        self.build_bottleneck()

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
            self.bottleneck.add_module("stage(1DBN)", torch.nn.BatchNorm1d(num_features=self.latent_size))
            self.bottleneck.add_module("stage(ReLU)", torch.nn.ReLU())

    def forward(self, x):
        return self.bottleneck(x)


class FastSCNNDecoder(torch.nn.Module):
    """
    The decoder for an outfitted autoencoder based on the FastSCNN segmentation network
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 upsample_factor=2,
                 bottleneck_channels=False):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the desired latent size for the bottleneck
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param upsample_factor: the factor to perform upsampling in the decoder
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(FastSCNNDecoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_mid_channels = bottleneck_channels
        self.latent_size = latent_size
        self.upsample_factor = upsample_factor
        self.compress_bottleneck = compress_bottleneck
        self.flatten_bottleneck = flatten_bottleneck

        self.bottleneck = torch.nn.Sequential()
        self.unpack_bottleneck()
        self.fastscnn_decoder = self.build_decoder()

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

    def build_decoder(self):
        """`
        Build the components related to FastSCNN (reversed from the encoder for this autoencoder)
        """
        stack = torch.nn.Sequential(
            pyramid_pooling.FastPyramidPooling(),
            common_blocks.FastSCNNGlobalFeatureExtractor(
                in_channels=128,
                out_channels=GLOBAL_FEATURE_CHANNELS,
                upsample_factor=self.upsample_factor),

            common_blocks.FastSCNNLearningToDownsample(
                in_channels=64,
                out_channels=LEARNING_TO_DOWNSAMPLE_CHANNELS,
                upsample_factor=self.upsample_factor),

            common_blocks.ConvBlock(in_channels=32,
                                    out_channels=3,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    use_bn=False,
                                    activation=False,
                                    upsample_factor=self.upsample_factor,
                                    dropout_rate=None),
            torch.nn.Tanh())

        return stack

    def forward(self, x):
        if self.flatten_bottleneck or self.compress_bottleneck:
            x = self.bottleneck(x)

        x = self.fastscnn_decoder(x)
        return x


class FastSCNNAE(torch.nn.Module):
    """
    Fast-SCNN from 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502 outfitted as an
    autoencoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=BOT_CHANNELS):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the latent size of the bottleneck
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(FastSCNNAE, self).__init__()

        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck

        self.encoder = encoders.FastSCNNEncoder(input_res=input_res)

        self.bottleneck = FastSCNNBottleneck(input_res=input_res,
                                             latent_size=latent_size,
                                             flatten_bottleneck=flatten_bottleneck,
                                             compress_bottleneck=compress_bottleneck,
                                             bottleneck_channels=bottleneck_channels)

        self.decoder = FastSCNNDecoder(input_res=input_res,
                                       latent_size=latent_size,
                                       flatten_bottleneck=flatten_bottleneck,
                                       compress_bottleneck=compress_bottleneck,
                                       bottleneck_channels=bottleneck_channels)

    def encoder_bottleneck(self, x):
        x, _ = self.encoder(x)
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        return x

    def forward(self, x):
        x, _ = self.encoder(x)

        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)
        x = self.decoder(x)

        return x
