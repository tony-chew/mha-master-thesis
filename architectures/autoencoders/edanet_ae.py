import torch
from architectures import common_blocks
from architectures import encoders
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

ENC_CHANNELS = [15, 60, 130, 450]
DEC_CHANNELS = [450, 260, 15, 3]
DILATIONS = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8, 16, 16]]

# ENC_CHANNELS = [ 15, 60, 130, 370 ]
# DEC_CHANNELS = [ 370, 260, 15, 3]
# DILATIONS = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8]]

# ENC_CHANNELS = [ 15, 60, 130, 290 ]
# DEC_CHANNELS = [ 290, 260, 15, 3]
# DILATIONS = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4]]


BOT_CHANNELS = [ 512, 256, 128, 64, 32, 1]
DOWNSAMPLE_FACTOR = 8


class EDANetBottleneck(torch.nn.Module):
    """
    The autoencoder bottleneck for the EDANet fast segmentation network
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
        super(EDANetBottleneck, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.latent_size = latent_size
        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck
        self.bottleneck_mid_channels = bottleneck_channels

        self.bridge = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.build_bottleneck()
        self.build_bridge()

    def build_bridge(self):
        if self.compress_bottleneck:
            self.bridge.add_module("bridge(downsample1)",
                                   common_blocks.DABNetDownBlock(in_channels=450,
                                                                 out_channels=490))
            self.bridge.add_module("bridge(downsample2",
                                   common_blocks.DABNetDownBlock(in_channels=490,
                                                                 out_channels=512))
        #     self.bridge.add_module("bridge(downsample1)",
        #                            common_blocks.DABNetDownBlock(in_channels=290,
        #                                                          out_channels=420))
        #     self.bridge.add_module("bridge(downsample2",
        #                            common_blocks.DABNetDownBlock(in_channels=420,
        #                                                          out_channels=512))
            # self.bridge.add_module("bridge(downsample1)",
            #                        common_blocks.DABNetDownBlock(in_channels=370,
            #                                                      out_channels=490))
            # self.bridge.add_module("bridge(downsample2",
            #                        common_blocks.DABNetDownBlock(in_channels=490,
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
            self.bottleneck.add_module("stage(linear)", torch.nn.Linear(
                self.bottleneck_h * self.bottleneck_w * self.bottleneck_channels,
                self.latent_size))
            self.bottleneck.add_module("stage(1DBN)", torch.nn.BatchNorm1d(num_features=self.latent_size))
            self.bottleneck.add_module("stage(ReLU)", torch.nn.ReLU())

    def forward(self, x):
        x = self.bridge(x)

        return self.bottleneck(x)


class EDANetDecoderAE(torch.nn.Module):
    """
    The decoder for an outfitted autoencoder based on the EDANet segmentation network
    """
    def __init__(self,
                 dec_channels,
                 input_res,
                 latent_size,
                 dilations,
                 dropout_rate=0.02,
                 growth_rate=40,
                 input_channels=DEC_CHANNELS[0],
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 upsample_factor=2,
                 bottleneck_channels=None):
        """
        :param dec_channels: the output channels of each stage of the decoder
        :param input_res: the input resolution to the model
        :param latent_size: the desired latent size for the bottleneck
        :param dilations: the dilation rates of each layer of the encoder
        :param dropout_rate: the dropout rates of each layer of the encoder
        :param growth_rate: the rate of growth of the channels from one layer to the next
        :param input_channels: the no of input channels to the decoder model
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param upsample_factor: the factor to perform upsampling in the decoder
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(EDANetDecoderAE, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.bottleneck_h, self.bottleneck_w = common_blocks.get_bottleneck_res(input_res=self.input_res,
                                                                                downsample_factor=DOWNSAMPLE_FACTOR)
        self.bottleneck_channels = bottleneck_channels[-1] if compress_bottleneck else bottleneck_channels[0]
        self.latent_size = latent_size
        self.dec_channels = dec_channels
        self.dilations = [elem[::-1] for elem in dilations][::-1]
        self.dropout_rate = dropout_rate
        self.growth_rate = growth_rate
        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck
        self.bottleneck_mid_channels = bottleneck_channels
        self.in_channels = input_channels
        self.upsample_factor = upsample_factor

        self.decoder = torch.nn.Sequential()
        self.bottleneck = torch.nn.Sequential()
        self.bridge = torch.nn.Sequential()

        self.unpack_bridge()
        self.unpack_bottleneck()
        self.build_decoder()

    def unpack_bottleneck(self):
        """
        Build the components related to the autoencoder bottleneck (unpack the bottleneck)
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
                                                           out_channels=490,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

            self.bridge.add_module("stage(upsample2)",
                                   common_blocks.ConvBlock(in_channels=490,
                                                           out_channels=450,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

    def build_decoder(self):
        """
        Build the componenents related to EDANet (reversed from the encoder for this autoencoder)
        """
        for i, dilations_per_stage in enumerate(self.dilations):
            block_out_channels = self.dec_channels[i]
            stage = torch.nn.Sequential()
            dilation_len = len(dilations_per_stage) - 1

            for j, dilation in enumerate(dilations_per_stage):
                if j == dilation_len:
                    stage.add_module(f"unit {j + 1}", common_blocks.ConvBlock(in_channels=self.in_channels,
                                                                              out_channels=self.dec_channels[i + 1],
                                                                              kernel_size=3,
                                                                              stride=1,
                                                                              padding=1,
                                                                              use_bn=False if i == 2 else True,
                                                                              activation=False if i == 2 else True,
                                                                              upsample_factor=self.upsample_factor))
                    self.in_channels = self.dec_channels[i + 1]

                else:
                    block_out_channels -= self.growth_rate
                    stage.add_module(f"unit {j + 1}", common_blocks.EDANetResidualBlock(in_channels=self.in_channels,
                                                                                        out_channels=block_out_channels,
                                                                                        dilation=dilation,
                                                                                        dropout_rate=self.dropout_rate,
                                                                                        encoder=False))
                    self.in_channels = block_out_channels
            self.decoder.add_module(f"stage {i + 1}", stage)

        self.decoder.add_module("stage(tanh)", torch.nn.Tanh())

    def forward(self, x):
        if self.compress_bottleneck or self.flatten_bottleneck:
            x = self.bottleneck(x)

        x = self.bridge(x)
        x = self.decoder(x)
        return x


class EDANet_AE(torch.nn.Module):
    """
    EDANet from 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation',
    https://arxiv.org/abs/1809.06323v3 outfitted as an autoencoder
    """
    def __init__(self,
                 input_res,
                 latent_size,
                 enc_channels=ENC_CHANNELS,
                 dec_channels=DEC_CHANNELS,
                 dilations=DILATIONS,
                 flatten_bottleneck=False,
                 compress_bottleneck=False,
                 bottleneck_channels=BOT_CHANNELS,
                 **kwargs):
        """
        :param input_res: the input resolution to the model
        :param latent_size: the latent size of the bottleneck
        :param enc_channels: the output channels of each stage of the encoder, subsequently the input for the decoder
        :param dec_channels: the output channels of each stage of the decoder
        :param dilations: the dilation rates of each layer of the encoder
        :param flatten_bottleneck: whether or not to apply flatten/linear operations to the bottleneck
        :param compress_bottleneck: whether or not to apply compression to the bottleneck
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(EDANet_AE, self).__init__()

        self.flatten_bottleneck = flatten_bottleneck
        self.compress_bottleneck = compress_bottleneck

        self.encoder = encoders.EDANetEncoder(channels=enc_channels,
                                              input_res=input_res,
                                              dilations=dilations)

        self.bottleneck = EDANetBottleneck(input_res=input_res,
                                           latent_size=latent_size,
                                           flatten_bottleneck=flatten_bottleneck,
                                           compress_bottleneck=compress_bottleneck,
                                           bottleneck_channels=bottleneck_channels)

        self.decoder = EDANetDecoderAE(dec_channels=dec_channels,
                                     input_res=input_res,
                                     latent_size=latent_size,
                                     dilations=dilations,
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

        # x = self.encoder_bottleneck(x)
        x = self.decoder(x)

        return x