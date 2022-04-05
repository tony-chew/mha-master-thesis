import torch
from architectures import common_blocks
from architectures import encoders
from architectures import pyramid_pooling
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")


class FastSCNNDecoder(torch.nn.Module):
    """
    The decoder for the FastSCNN fast segmentation network
    """
    def __init__(self,
                 input_res,
                 xhigh_in_channels,
                 xlow_in_channels,
                 feature_out_channels,
                 num_classes):
        """
        :param input_res: the input resolution to the model
        :param xhigh_in_channels: the output number of channels from the feature extraction module
        :param xlow_in_channels: the output number of channels from the learning to downsample module
        :param feature_out_channels: the desired output number of channels from the feature fusion module
        :param num_classes: the number of classes of the output
        """
        super(FastSCNNDecoder, self).__init__()

        self.input_res_forward = input_res
        self.input_res = InputResolution(input_res[0], input_res[1])
        self.fusion_out_size = self.input_res.height // 8, self.input_res.width // 8

        self.feature_fusion = common_blocks.FastSCNNFeatureFusion(xhigh_in_channels=xhigh_in_channels,
                                                                  xlow_in_channels=xlow_in_channels,
                                                                  out_channels=feature_out_channels,
                                                                  xhigh_in_size=self.fusion_out_size)

        self.classifier = common_blocks.FastSCNNClassifier(in_channels=feature_out_channels,
                                                           num_classes=num_classes)

        self.upscale = pyramid_pooling.InterpolationBlock(scale_factor=None,
                                                          out_size=input_res)

    def forward(self, xhigh, xlow):
        x = self.feature_fusion(xhigh, xlow)
        x = self.classifier(x)
        x = self.upscale(x, self.input_res_forward)

        return x


class FastSCNN(torch.nn.Module):
    """
    Fast-SCNN from 'Fast-SCNN: Fast Semantic Segmentation Network,' https://arxiv.org/abs/1902.04502
    """
    def __init__(self,
                 input_res,
                 num_classes,
                 xhigh_in_channels=64,
                 xlow_in_channels=128,
                 feature_out_channels=128):
        """
        :param input_res: the input resolution to the model
        :param xlow_in_channels: no of channels in the low res branch
        :param xhigh_in_channels: no of channels in the high res branch
        :param feature_out_channels: no of channels in the output of the decoder before upsampling
        :param num_classes: no of classes to categorise semantically
        """
        super(FastSCNN, self).__init__()

        self.encoder = encoders.FastSCNNEncoder(input_res=input_res)

        self.decoder = FastSCNNDecoder(input_res=input_res,
                                       xhigh_in_channels=xhigh_in_channels,
                                       xlow_in_channels=xlow_in_channels,
                                       feature_out_channels=feature_out_channels,
                                       num_classes=num_classes)

    def forward(self, x):
        xlow, xhigh = self.encoder(x)

        x = self.decoder(xhigh, xlow)

        return x