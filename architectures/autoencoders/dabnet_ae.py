import torch
from architectures import common_blocks
from architectures import encoders
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

INIT_BLOCK_CHANNELS = 32
CHANNELS = [35, 131, 259]
DILATIONS = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
UP_CHANNELS = [64, 32]
UP_DILATIONS = [[16, 16, 8, 8, 4, 4], [2, 2, 2]]

BOT_CHANNELS = [ 512, 256, 128, 64, 32, 1 ]
DOWNSAMPLE_FACTOR = 32


class DABNetBottleneck(torch.nn.Module):
    """
    The bottleneck for the DABNet fast segmentation network
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
        super(DABNetBottleneck, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_mid_channels = bottleneck_channels
        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck
        self.latent_size = latent_size

        self.bridge = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.build_bottleneck()
        self.build_bridge()

    def build_bridge(self):
        if self.compress_bottleneck:
            self.bridge.add_module("bridge(downsample1)",
                                   common_blocks.DABNetDownBlock(in_channels=259,
                                                                 out_channels=456))
            self.bridge.add_module("bridge(downsample2",
                                   common_blocks.DABNetDownBlock(in_channels=456,
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


class DABNetDecoder(torch.nn.Module):
    """
    The decoder for an outfitted autoencoder based on the DABNet fast segmentation network
    """
    def __init__(self,
                 input_res,
                 channels,
                 dilations,
                 latent_size,
                 init_block_channels=128,
                 input_channels=259,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 upsample_factor=2,
                 bottleneck_channels=False):
        """
        :param channels: the output channels of each downsampling stage of the encoder
        :param input_res: the input resolution to the model
        :param latent_size: the desired latent size for the bottleneck
        :param dilations: the dilation rates of each layer of the encoder
        :param init_block_channels: the no of channels for the input of the Upstaging process
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param input_channels: the no of input channels to the decoder model (set at 128)
        :param upsample_factor: the factor to perform upsampling in the decoder
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(DABNetDecoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.bottleneck_mid_channels = bottleneck_channels
        self.latent_size = latent_size
        self.channels = channels
        self.dilations = dilations
        self.init_block_channels = init_block_channels
        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck
        self.input_channels = input_channels
        self.upsample_factor = upsample_factor

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
                                       common_blocks.BottleneckReduction(channels=self.bottleneck_mid_channels,
                                                                         upsample=True))

    def unpack_bridge(self):
        """
        Unpack the bridge to match original encoder output EDANet dimensions
        """
        if self.compress_bottleneck:
            self.bridge.add_module("stage(upsample1)",
                                   common_blocks.ConvBlock(in_channels=512,
                                                           out_channels=456,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

            self.bridge.add_module("stage(upsample2)",
                                   common_blocks.ConvBlock(in_channels=456,
                                                           out_channels=259,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

    def build_decoder(self):
        """
        Build the components related to DABNet. Reversed from the encoder with slight modifications
        There is no residual connection between the upsampling and residual modules themselves and no average pooling
        operation. A 1x1 point-wise convolution follows from the bottleneck to get the number of channels to its
        appropriate depth
        """
        self.conv1 = common_blocks.conv1x1(in_channels=self.input_channels,
                                           out_channels=self.init_block_channels)
        self.input_channels = self.init_block_channels

        for i, (out_channels, dilations_i) in enumerate(zip(self.channels, self.dilations)):
            self.decoder.add_module(f"stage{i + 1}", common_blocks.DABNetUpStage(in_channels=self.input_channels,
                                                                                 out_channels=out_channels,
                                                                                 dilations=dilations_i,
                                                                                 upsample_factor=self.upsample_factor))
            self.input_channels = out_channels

        self.decoder.add_module("tail_block", common_blocks.DABNetInitialBlocks(in_channels=3,
                                                                                out_channels=self.input_channels,
                                                                                upsample=True))
        self.decoder.add_module("stage(tanh)", torch.nn.Tanh())

    def forward(self, x):
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        x = self.bridge(x)
        x = self.conv1(x)
        x = self.decoder(x)
        return x


class DABNet_AE(torch.nn.Module):
    """
    DABNet from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation',
    https://arxiv.org/abs/1907.11357 outfitted as an encoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 init_block_channels=INIT_BLOCK_CHANNELS,
                 channels=CHANNELS,
                 dilations=DILATIONS,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=BOT_CHANNELS):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the latent size of the bottleneck
        :param channels: the output channels of each downsampling stage of the encoder
        :param dilations: the dilation rates of each layer of the encoder
        :param init_block_channels: the no of channels for the initial blocks in the encoder and for the pointwise
                                    convolution in the decoder
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(DABNet_AE, self).__init__()

        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck

        self.encoder = encoders.DABNetEncoder(input_res=input_res,
                                              channels=channels,
                                              init_block_channels=init_block_channels,
                                              dilations=dilations)

        self.bottleneck = DABNetBottleneck(input_res=input_res,
                                           latent_size=latent_size,
                                           flatten_bottleneck=flatten_bottleneck,
                                           compress_bottleneck=compress_bottleneck,
                                           bottleneck_channels=bottleneck_channels)

        self.decoder = DABNetDecoder(input_res=input_res,
                                     channels=UP_CHANNELS,
                                     dilations=UP_DILATIONS,
                                     latent_size=latent_size,
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