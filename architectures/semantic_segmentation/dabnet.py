import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

INIT_BLOCK_CHANNELS = 32
CHANNELS = [35, 131, 259]
DILATIONS = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]


class DABNetDecoder(torch.nn.Module):
    """
    The decoder for the DABNet fast segmentation network
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        :param in_channels: no of channels to the input of the decoder
        :param out_channels: no of channels to the output of the decoder
        """
        super(DABNetDecoder, self).__init__()

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


class DABNet(torch.nn.Module):
    """
    DABNet from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation',
    https://arxiv.org/abs/1907.11357
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 channels=CHANNELS,
                 init_block_channels=INIT_BLOCK_CHANNELS,
                 dilations=DILATIONS):
        """
        :param input_res: the input resolution to the model
        :param num_classes: number of classes for then network to categorise
        :param channels: the output channels of each stage of the encoder, subsequently the input for the decoder
        :param dilations: the dilation rates of each layer of the encoder
        """
        super(DABNet, self).__init__()

        self.encoder = encoders.DABNetEncoder(channels=channels,
                                              input_res=input_res,
                                              init_block_channels=init_block_channels,
                                              dilations=dilations)

        self.decoder = DABNetDecoder(in_channels=channels[-1],
                                     out_channels=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x