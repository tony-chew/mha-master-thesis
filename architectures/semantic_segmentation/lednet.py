import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

CHANNELS = [32, 64, 128]
DILATIONS = [[0, 1, 1, 1], [0, 1, 1], [0, 1, 2, 5, 9, 2, 5, 9, 17]]
DROPOUT_RATES = [0.03, 0.03, 0.3]


class LEDNetDecoder(torch.nn.Module):
    """
    The decoder for the LEDNet fast segmentation network
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        :param in_channels: no of channels to the input of the decoder
        :param out_channels: no of channels to the output of the decoder
        """
        super(LEDNetDecoder, self).__init__()

        self.apn = common_blocks.LEDNetAPNModule(in_channels=in_channels,
                                                 out_channels=out_channels)

        self.upscale = pyramid_pooling.InterpolationBlock(scale_factor=8,
                                                          align_corners=True)

    def forward(self, x):
        x = self.apn(x)
        x = self.upscale(x)

        return x


class LEDNet(torch.nn.Module):
    """
    LEDNet from 'LEDNet: A Lightweight Encoder-Decoder Network for Real-Time Semantic Segmentation',
    https://arxiv.org/abs/1905.02423
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 channels=CHANNELS,
                 dilations=DILATIONS,
                 dropout_rates=DROPOUT_RATES):
        """
        :param input_res: the input resolution to the model
        :param channels: the output channels of each downsampling stage of the encoder
        :param dilations: the dilation rates of each layer of the encoder
        :param dropout_rates: the dropout rates of each layer of the encoder
        :param num_classes: no of output channels
        """
        super(LEDNet, self).__init__()

        self.encoder = encoders.LEDNetEncoder(channels=channels,
                                              input_res=input_res,
                                              dilations=dilations,
                                              dropout_rates=dropout_rates)

        self.decoder = LEDNetDecoder(in_channels=channels[-1],
                                     out_channels=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x