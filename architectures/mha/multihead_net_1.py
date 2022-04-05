import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")

ENC_CHANNELS = [15, 60, 130, 450]
DEC_CHANNELS = [450, 260, 15, 3]
DILATIONS = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8, 16, 16]]

BOT_CHANNELS = [ 512, 256, 128, 64, 32, 1]
DOWNSAMPLE_FACTOR = 32


class AutoencoderHead(torch.nn.Module):
    """
    The decoder for an outfitted autoencoder based on the EDANet segmentation network
    """
    def __init__(self,
                 dec_channels,
                 dilations,
                 growth_rate=40,
                 dropout_rate=0.02,
                 # dropout_rate=0.2,
                 input_channels=DEC_CHANNELS[0],
                 upsample_factor=2,
                 bottleneck_channels=None):
        """
        :param dec_channels: a list of input channels in the decoder
        :param dilations: the dilation rates of each layer of the encoder
        :param growth_rate: the rate of growth of the channels from one layer to the next
        :param dropout_rate: the dropout rates of each layer of the encoder
        :param input_channels: the number of channels in the input
        :param upsample_factor: the factor for upsampling back to original dimensions for each layer
        :param bottleneck_channels: if compression of bottlenecks, provide a list of channels
        """
        super(AutoencoderHead, self).__init__()

        # bottleneck/bridge related parameters
        self.bottleneck_mid_channels = bottleneck_channels

        # AE head related parameters
        self.in_channels = input_channels
        self.dec_channels = dec_channels
        self.dilations = [elem[::-1] for elem in dilations][::-1]
        self.dropout_rate = dropout_rate
        self.growth_rate = growth_rate
        self.upsample_factor = upsample_factor

        # initialise modules
        self.bottleneck_down = torch.nn.Sequential()
        self.bottleneck_up = torch.nn.Sequential()
        self.decoder = torch.nn.Sequential()

        # construct modules
        self.build_bottleneck()
        self.unpack_bottleneck()
        self.build_decoder()

    def build_bottleneck(self):
        # construct bridge (downsample two times)
        self.bottleneck_down.add_module("bridge(downsample1)",
                                   common_blocks.DABNetDownBlock(in_channels=450,
                                                                 out_channels=490))
        self.bottleneck_down.add_module("bridge(downsample2",
                                   common_blocks.DABNetDownBlock(in_channels=490,
                                                                 out_channels=512))

        # construct latent space (reduce channels to one channel, resulting in 16x32x1 (ie 512) latent space)
        self.bottleneck_down.add_module("stage(bottleneck_down)",
                                   common_blocks.BottleneckReduction(self.bottleneck_mid_channels))

    def unpack_bottleneck(self):
        self.bottleneck_up.add_module("stage(bottleneck_up)",
                                   common_blocks.BottleneckReduction(self.bottleneck_mid_channels,
                                                                     True))

        # deconstruct bridge
        self.bottleneck_up.add_module("bridge(upsample1)",
                                   common_blocks.ConvBlock(in_channels=512,
                                                           out_channels=490,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))
        self.bottleneck_up.add_module("bridge(upsample2)",
                                   common_blocks.ConvBlock(in_channels=490,
                                                           out_channels=450,
                                                           kernel_size=1,
                                                           stride=1,
                                                           padding=0,
                                                           use_bn=True,
                                                           activation=True,
                                                           upsample_factor=self.upsample_factor))

    def build_decoder(self):
        # construct the autoencoder decoder head
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
        x = self.bottleneck_down(x)
        x = self.bottleneck_up(x)
        x = self.decoder(x)

        return x


class SegmentationHead(torch.nn.Module):
    """
    The decoder for the EDANet fast segmentation network
    """
    def __init__(self,
                 num_classes,
                 in_channels=ENC_CHANNELS[-1]):
        """
        :param in_channels: no of channels to the input of the decoder
        :param num_classes: no of classes in the output tensor
        """
        super(SegmentationHead, self).__init__()

        # bridge parameters
        self.out_channels = in_channels

        # initialise/build modules
        self.head = common_blocks.conv1x1(in_channels=self.out_channels,
                                          out_channels=num_classes,
                                          use_bn=False,
                                          activation=False)
        self.upscale = pyramid_pooling.InterpolationBlock(scale_factor=8,
                                                          align_corners=True)

    def forward(self, x):
        x = self.head(x)
        x = self.upscale(x)

        return x


class MultiHeadArchitecture(torch.nn.Module):
    """
    The Multi-Head Architecture network: 1st configuration, semantic branch at original fast segmentation location
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 enc_channels=ENC_CHANNELS,
                 dec_channels=DEC_CHANNELS,
                 dilations=DILATIONS,
                 bottleneck_channels=BOT_CHANNELS,
                 **kwargs):
        """
        :param input_res: the input resolution to the model
        :param num_classes: number of classes in the output
        :param enc_channels: a list of input channels in the encoder
        :param dec_channels: a list of input channels in the decoder
        :param dilations: the dilation rates of each layer of the encoder in list form
        :param bottleneck_channels: a list of output channels in the compression of the bottleneck
        :param downsample_factor: the factor to downsample the input image in the bottleneck
        """
        super(MultiHeadArchitecture, self).__init__()

        self.encoder = encoders.EDANetEncoder(channels=enc_channels,
                                              input_res=input_res,
                                              dilations=dilations)

        self.autoencoder_head = AutoencoderHead(dec_channels=dec_channels,
                                                dilations=dilations,
                                                bottleneck_channels=bottleneck_channels)

        self.segmentation_head = SegmentationHead(num_classes=num_classes)

    def encoder_bottleneck(self, x):
        x = self.encoder(x)
        x = self.autoencoder_head.bottleneck_down(x)

        return x

    def forward(self, x):
        # output = self.encoder_bottleneck(x)

        enc_output = self.encoder(x)

        autoencoder_head = self.autoencoder_head(enc_output)
        segmentation_head = self.segmentation_head(enc_output)

        output = (autoencoder_head, segmentation_head)

        return output