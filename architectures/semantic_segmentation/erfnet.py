import torch
from architectures import common_blocks
from architectures import encoders
import collections

CHANNELS = [16, 64, 128]
UP_CHANNELS = [64, 16]
DILATIONS = [[1], [1, 1, 1, 1, 1, 1], [1, 2, 4, 8, 16, 2, 4, 8, 16]]
UP_DILATIONS = [[1, 1, 1], [1, 1, 1]]
DROPOUT_RATES = [[0.0], [0.03, 0.03, 0.03, 0.03, 0.03, 0.03], [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]]


class ERFNetDecoder(torch.nn.Module):
    """
    The decoder for the ERFNet fast segmentation network
    """
    def __init__(self,
                 dilations=UP_DILATIONS,
                 out_channels=UP_CHANNELS,
                 num_classes=19,
                 in_channels=128):
        """
        :param in_channels: no of channels to the input of the decoder
        :param out_channels: no of channels to the output of the decoder
        :param num_classes: number of classes for then network to categorise
        :param dilations: the dilation rate of each layer of the encoder
        """
        super(ERFNetDecoder, self).__init__()

        self.in_channels = in_channels
        self.dilations = dilations
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.decoder = torch.nn.Sequential()
        self.build_decoder()
        self.head = self.build_head()

    def build_decoder(self):
        for i, out_channels in enumerate(self.out_channels):
            dilations_per_stage = self.dilations[i]
            stage = torch.nn.Sequential()

            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    unit = common_blocks.DeConvBlock(in_channels=self.in_channels,
                                                     out_channels=out_channels,
                                                     stride=2,
                                                     kernel_size=3,
                                                     padding=1,
                                                     output_padding=1)
                else:
                    unit = common_blocks.ERFNetFactorisedResidualBlock(channels=self.in_channels,
                                                                       dilation=1,
                                                                       dropout_rate=None)
                stage.add_module(f"unit{j + 1}", unit)
                self.in_channels = out_channels

            self.decoder.add_module(f"stage{i + 1}", stage)

    def build_head(self):
        return torch.nn.ConvTranspose2d(in_channels=self.in_channels,
                                        out_channels=self.num_classes,
                                        kernel_size=2,
                                        stride=2,
                                        padding=0,
                                        output_padding=0)

    def forward(self, x):
        x = self.decoder(x)
        x = self.head(x)

        return x


class ERFNetTrainEncoder(torch.nn.Module):
    """
    The training module for ERFNet
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ERFNetTrainEncoder, self).__init__()

        self.head = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=out_channels,
                                          use_bn=False,
                                          activation=False)

    def forward(self, x):
        return self.head(x)


class ERFNet(torch.nn.Module):
    """
    ERFNet from 'ERFNet: Efficient Residual Factorized ConvNet for Real-Time Semantic Segmentation',
    https://ieeexplore.ieee.org/document/8063438
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 encoder_only=False,
                 channels=CHANNELS,
                 up_channels=UP_CHANNELS,
                 dilations=DILATIONS,
                 up_dilations=UP_DILATIONS,
                 dropout_rates=DROPOUT_RATES):
        """
        :param input_res: the input resolution to the model
        :param num_classes: number of classes to output
        :param channels: the output channels of each downsampling stage of the encoder
        :param up_channels: the output channels of each stage of the decoder
        :param dilations: the dilation rates of each layer of the encoder
        :param up_dilations: the dilation rates of each layer of the decoder
        """
        super(ERFNet, self).__init__()

        self.encoder = encoders.ERFNetEncoder(channels=channels,
                                              input_res=input_res,
                                              dilations=dilations,
                                              dropout_rates=dropout_rates)

        # self.decoder = ERFNetTrainEncoder(in_channels=channels[-1],
        #                                  out_channels=num_classes)

        self.decoder = ERFNetDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x