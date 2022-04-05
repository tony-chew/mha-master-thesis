import unittest
import pytest
import torch
import torchinfo
import architectures
import utils
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from architectures.autoencoders import fastscnn_ae
from architectures.autoencoders import conv_ae
from architectures.autoencoders import lednet_ae
from architectures.autoencoders import dabnet_ae
from architectures.autoencoders import edanet_ae
from architectures.autoencoders import erfnet_ae

from architectures.semantic_segmentation import lednet
from architectures.semantic_segmentation import edanet
from architectures.semantic_segmentation import fastscnn
from architectures.semantic_segmentation import dabnet
from architectures.semantic_segmentation import erfnet

from architectures.mha import multihead_net_1
from architectures.mha import multihead_net_2
from architectures.mha import multihead_net_3
from architectures.mha import multihead_net_4

latent_size = 512
dropout_rate = 0.2
input_res = [512, 1024]
num_classes = 19
#input_res = [1024, 2048]

# MODEL = conv_ae.ConvAE(input_res=input_res,
#                        num_layers=6,
#                        use_residual=False,
#                        latent_size=latent_size,
#                        dropout_rate=dropout_rate,
#                        flatten_bottleneck=True,
#                        compress_bottleneck=False)

# MODEL = fastscnn_ae.FastSCNNAE(input_res=input_res,
#                                   latent_size=latent_size,
#                                 flatten_bottleneck=False,
#                                   compress_bottleneck=True)
#
# MODEL = erfnet_ae.ERFNetAE(input_res=input_res,
#                            latent_size=latent_size,
#                            flatten_bottleneck=False,
#                            compress_bottleneck=True)

# MODEL = lednet_ae.LEDNetAE(input_res=input_res,
#                             latent_size=latent_size,
#                             flatten_bottleneck=False,
#                             compress_bottleneck=True)
#
# MODEL = dabnet_ae.DABNet_AE(input_res=input_res,
#                             latent_size=latent_size,
#                             flatten_bottleneck=False,
#                             compress_bottleneck=True)
# #
# MODEL = edanet_ae.EDANet_AE(input_res=input_res,
#                             latent_size=latent_size,
#                             flatten_bottleneck=False,
#                             compress_bottleneck=True)

# MODEL = lednet.LEDNet(input_res=input_res,
#                       num_classes=num_classes)
# MODEL = edanet.EDANet(input_res=input_res,
#                       num_classes=num_classes)

# MODEL = fastscnn.FastSCNN(input_res=input_res,
#                           num_classes=num_classes)
# MODEL = dabnet.DABNet(input_res=input_res,
#                        num_classes=num_classes)
# MODEL = erfnet.ERFNet(input_res=input_res,
#                       num_classes=num_classes)


# MODEL = multihead_net_1.MultiHeadArchitecture(input_res=input_res,
#                                               num_classes=num_classes)
# MODEL = multihead_net_2.MultiHeadArchitecture(input_res=input_res,
#                                               num_classes=num_classes)
# MODEL = multihead_net_3.MultiHeadArchitecture(input_res=input_res,
#                                               num_classes=num_classes)
MODEL = multihead_net_4.MultiHeadArchitecture(input_res=input_res,
                                              num_classes=num_classes)

SHAPE = [1, 3, 512, 1024]
# SHAPE = [1, 3, 1024, 2048]

class TestModel(unittest.TestCase):
    """
    Test the given model and ensure its dimensions and internal layers sync up
    """
    def test_model(self):
        input = torch.rand(SHAPE)
        # print(input)
        output = MODEL(input)

        MODEL.eval()
        model_summary = torchinfo.summary(MODEL, input_size=SHAPE)

        #assert output.shape == input.shape
        assert output.shape == [2, 3, 511, 1024]