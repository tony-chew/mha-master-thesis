import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

ENC_CHANNELS = [15, 60, 130, 450]
DILATIONS = [[0], [0, 1, 1, 1, 2, 2], [0, 2, 2, 4, 4, 8, 8, 16, 16]]


class EDANetDecoder(torch.nn.Module):
    """
    The decoder for the EDANet fast segmentation network
    """
    def __init__(self,
                 in_channels=ENC_CHANNELS[-1],
                 out_channels=19):
        """
        :param in_channels: no of channels to the input of the decoder
        :param out_channels: no of channels to the output of the decoder
        """
        super(EDANetDecoder, self).__init__()

        self.head = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=out_channels,
                                          use_bn=False,
                                          activation=False)

        self.upscale = pyramid_pooling.InterpolationBlock(scale_factor=8,
                                                          align_corners=True)

    def forward(self, x):
        x = self.head(x)
        x = self.upscale(x)

        return x


class EDANetTrainEncoder(torch.nn.Module):
    """
    The training module for EDANet
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(EDANetTrainEncoder, self).__init__()

        self.head = common_blocks.conv3x3(in_channels=in_channels,
                                          out_channels=out_channels,
                                          padding=1)

    def forward(self, x):
        return self.head(x)


class EDANet(torch.nn.Module):
    """
    EDANet from 'Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation',
    https://arxiv.org/abs/1809.06323v3
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 enc_channels=ENC_CHANNELS,
                 dilations=DILATIONS):
        """
        :param input_res: the input resolution to the model
        :param num_classes: number of classes for then network to categorise
        :param test_mode: the tag to further upsample the output to its full resolution
        :param enc_channels: the output channels of each stage of the encoder, subsequently the input for the decoder
        :param dilations: the dilation rates of each layer of the encoder
        """
        super(EDANet, self).__init__()

        self.encoder = encoders.EDANetEncoder(channels=enc_channels,
                                              input_res=input_res,
                                              dilations=dilations)

        self.decoder = EDANetDecoder(in_channels=enc_channels[-1],
                                     out_channels=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
