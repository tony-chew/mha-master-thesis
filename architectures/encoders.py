import torch
from architectures import common_blocks
from architectures import pyramid_pooling
import collections

InputResolution = collections.namedtuple(
    "input_resolution",
    "height width")


class ConvEncoder(torch.nn.Module):
    """
    The encoder for the convolutional autoencoder
    """
    def __init__(self,
                 input_res,
                 num_layers,
                 use_residual,
                 input_channels=3,
                 dropout_rate=None):
        """
        :param input_res: the input resolution of the model (typically [256, 512] or [512, 1024]
        :param num_layers: the desired number of layers for the convolutional autoencoder
        :param use_residual: whether or not to use residual modules
        :param input_channels: the number of input channels (usually 3 for RGB images)
        :param dropout_rate: the desired dropout rate to use
        :param upsample_factor: the scale to upsample
        :param compress_bottleneck: whether or not to utilise bottleneck compression
        """
        super(ConvEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.num_layers = num_layers
        self.enc_out_channels, self.dec_out_channels, self.dilations = get_layer_size(num_layers, use_residual)

        self.in_channels = input_channels
        self.dropout = dropout_rate
        self.initial_dropout = None if dropout_rate is None else 2 * dropout_rate
        self.use_residual = use_residual
        self.encoder_start = 1

        common_blocks.check_model(self.input_res)
        self.encoder = torch.nn.Sequential()
        self.build_encoder()

    def build_encoder(self):
        """
        Iteratively construct the decoder
        """
        for i, out_channels in enumerate(self.enc_out_channels, start=1):
            block_dilation = self.dilations[i - self.encoder_start]

            if self.use_residual:
                if out_channels / self.in_channels == 1:
                    unit = common_blocks.ERFNetFactorisedResidualBlock(channels=out_channels,
                                                                       dilation=block_dilation,
                                                                       dropout_rate=self.dropout)

                else:
                    unit = common_blocks.ERFNetDownsampler(in_channels=self.in_channels,
                                                           out_channels=out_channels)

            else:
                if out_channels / self.in_channels == 1:
                    stride = 1
                else:
                    stride = 2
                unit = common_blocks.ConvBlock(in_channels=self.in_channels,
                                               out_channels=out_channels,
                                               kernel_size=3,
                                               stride=stride,
                                               padding=1,
                                               dilation=block_dilation,
                                               dropout_rate=self.initial_dropout if i == self.encoder_start else\
                                               self.dropout)

            self.in_channels = out_channels
            self.encoder.add_module(f"stage{i}", unit)

    def forward(self, x):
        return self.encoder(x)


class ERFNetEncoder(torch.nn.Module):
    """
    The encoder for the ERFNet fast segmentation network
    """
    def __init__(self,
                 channels,
                 dilations,
                 dropout_rates,
                 input_res,
                 in_channels=3):
        super(ERFNetEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.channels = channels
        self.dilations = dilations
        self.dropout_rates = dropout_rates
        self.input_channels = in_channels
        common_blocks.check_model(self.input_res)

        self.encoder = torch.nn.Sequential()
        self.build_encoder()

    def build_encoder(self):
        """
        Build the encoder components related to ERFNet
        """
        for i, out_channels in enumerate(self.channels):
            dilations_per_stage = self.dilations[i]
            dropout_per_stage = self.dropout_rates[i]
            stage = torch.nn.Sequential()

            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    unit = common_blocks.ERFNetDownsampler(in_channels=self.input_channels,
                                                            out_channels=out_channels)
                else:
                    unit = common_blocks.ERFNetFactorisedResidualBlock(channels=self.input_channels,
                                                                       dilation=dilation,
                                                                       dropout_rate=dropout_per_stage[j])

                stage.add_module(f"unit{j + 1}", unit)
                self.input_channels = out_channels

            self.encoder.add_module(f"stage{i + 1}", stage)

    def forward(self, x):
        return self.encoder(x)


class LEDNetEncoder(torch.nn.Module):
    """
    The encoder for the LEDNet fast segmentation network
    """
    def __init__(self,
                 channels,
                 input_res,
                 dilations,
                 dropout_rates,
                 input_channels=3):
        """
        :param channels: the output channels of each downsampling stage of the encoder
        :param input_res: the input resolution to the model
        :param dilations: the dilation rates of each layer of the encoder
        :param dropout_rates: the dropout rates of each layer of the encoder
        :param input_channels: the no of input channels to the model (3 for image input)
        """
        super(LEDNetEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.channels = channels
        self.dilations = dilations
        self.dropout_rates = dropout_rates
        self.input_channels = input_channels
        common_blocks.check_model(self.input_res)

        self.encoder = torch.nn.Sequential()
        self.build_encoder()

    def build_encoder(self):
        """
        Build the encoder components related to LEDNet
        """
        for i, dilations_per_stage in enumerate(self.dilations):
            out_channels = self.channels[i]
            dropout_rate = self.dropout_rates[i]
            stack = torch.nn.Sequential()

            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stack.add_module(f"unit {j + 1}", common_blocks.ERFNetDownsampler(in_channels=self.input_channels,
                                                                                      out_channels=out_channels))
                    self.input_channels = out_channels

                else:
                    stack.add_module(f"unit {j + 1}", common_blocks.LEDNetResidualBlock(channels=self.input_channels,
                                                                                        dilation=dilation,
                                                                                        dropout_rate=dropout_rate))

                self.encoder.add_module(f"stage {i + 1}", stack)

    def forward(self, x):
        return self.encoder(x)


class EDANetEncoder(torch.nn.Module):
    """
    The encoder for the EDANet fast segmentation network
    """
    def __init__(self,
                 channels,
                 input_res,
                 dilations,
                 growth_rate=40,
                 dropout_rate=0.02,
                 input_channels=3):
        """
        :param channels: the output channels of each downsampling stage of the encoder
        :param input_res: the input resolution to the model
        :param dilations: the dilation rates of each layer of the encoder
        :param growth rate: the desired growth rate of the channel depth from one input to the output
        :param dropout_rate: the dropout rates of each layer of the encoder
        :param input_channels: the no of input channels to the model (3 for image input)
        """
        super(EDANetEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.channels = channels
        self.dilations = dilations
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        self.in_channels = input_channels
        common_blocks.check_model(self.input_res)

        self.encoder = torch.nn.Sequential()
        self.build_encoder()

    def build_encoder(self):
        """
        Build the components related to LEDNet
        """
        for i, dilations_per_stage in enumerate(self.dilations):
            out_channels = self.channels[i]
            stage = torch.nn.Sequential()

            for j, dilation in enumerate(dilations_per_stage):
                if j == 0:
                    stage.add_module(f"unit {j + 1}", common_blocks.DABNetDownBlock(in_channels=self.in_channels,
                                                                                    out_channels=out_channels))

                else:
                    out_channels += self.growth_rate
                    stage.add_module(f"unit {j + 1}", common_blocks.EDANetResidualBlock(in_channels=self.in_channels,
                                                                                        out_channels=out_channels,
                                                                                        dilation=dilation,
                                                                                        dropout_rate=self.dropout_rate))

                self.in_channels = out_channels
            self.encoder.add_module(f"stage {i + 1}", stage)

    def forward(self, x):
        return self.encoder(x)


class DABNetEncoder(torch.nn.Module):
    """
    The encoder for the DABNet fast segmentation network
    """
    def __init__(self,
                 input_res,
                 channels,
                 init_block_channels,
                 dilations,
                 input_channels=3):
        """
        :param channels: the output channels of each downsampling stage of the encoder
        :param input_res: the input resolution to the model
        :param dilations: the dilation rates of each layer of the encoder
        :param init_block_channels: the no of channels for the initial blocks
        :param input_channels: the no of input channels to the model (3 for image input)
        """
        super(DABNetEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.channels = channels
        self.dilations = dilations
        self.init_block_channels = init_block_channels
        self.input_channels = input_channels
        self.encoder = common_blocks.DABNetDualPathSequential(return_two=False,
                                                              first_ordinals=1,
                                                              last_ordinals=0)

        common_blocks.check_model(self.input_res)
        self.build_encoder()

    def build_encoder(self):
        """
        Build the components related to DABNet
        """
        self.encoder.add_module("init_block", common_blocks.DABNetInitialBlocks(in_channels=self.input_channels,
                                                                                out_channels=self.init_block_channels))
        input_channels = self.init_block_channels

        for i, (out_channels, dilations_i) in enumerate(zip(self.channels, self.dilations)):
            self.encoder.add_module(f"stage{i + 1}", common_blocks.DABNetDownStage(pool_channels=self.input_channels,
                                                                                   in_channels=input_channels,
                                                                                   out_channels=out_channels,
                                                                                   dilations=dilations_i))
            input_channels = out_channels

    def forward(self, x):
        return self.encoder(x, x)


class FastSCNNEncoder(torch.nn.Module):
    """
    The encoder for the FastSCNN fast segmentation network
    """
    def __init__(self,
                 input_res):
        """
        :param input_res: the input resolution to the model
        """
        super(FastSCNNEncoder, self).__init__()

        self.input_res = InputResolution(input_res[0], input_res[1])
        self.pool_out_size = (self.input_res.height // 32, self.input_res.width // 32)
        common_blocks.check_model(self.input_res)

        self.learning_to_downsample = common_blocks.FastSCNNLearningToDownsample()
        self.feature_extractor = common_blocks.FastSCNNGlobalFeatureExtractor()
        self.pyramid_pooling = pyramid_pooling.FastPyramidPooling()

    def forward(self, x):
        xhigh = self.learning_to_downsample(x)
        x = self.feature_extractor(xhigh)
        x = self.pyramid_pooling(x)

        return x, xhigh


def get_layer_size(num_layers,
                   use_residual):
    """
    Specifies the number of depth channels of each output layer
    """
    if num_layers == 4:
        enc_out_channels = [32, 64, 128, 256]
        dec_out_channels = [128, 64, 32, 3]
        dilations = [1, 1, 1, 1]

    elif num_layers == 6:
        enc_out_channels = [32, 64, 128, 128, 256, 256]
        dec_out_channels = [256, 128, 128, 64, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 2]
        else:
            dilations = [1, 1, 1, 1, 1, 1]

    elif num_layers == 8:
        enc_out_channels = [32, 64, 128, 128, 128, 256, 256, 256]
        dec_out_channels = [256, 256, 128, 128, 128, 64, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 1, 2, 2]
        else:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1]

    elif num_layers == 10:
        # enc_out_channels = [32, 64, 128, 128, 128, 128, 256, 256, 256, 256]
        # dec_out_channels = [256, 256, 256, 128, 128, 128, 128, 64, 32, 3]
        enc_out_channels = [32, 64, 128, 128, 128, 128, 256, 256, 512, 512]
        dec_out_channels = [512, 256, 256, 128, 128, 128, 128, 64, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    elif num_layers == 12:
        enc_out_channels = [32, 32, 64, 64, 128, 128, 128, 128, 256, 256, 512, 512]
        dec_out_channels = [512, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32, 3]
        # enc_out_channels = [32, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        # dec_out_channels = [512, 512, 256, 256, 256, 128, 128, 128, 128, 64, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    elif num_layers == 14:
        # NOTE: early layers for edge perception, later layers for global context
        enc_out_channels = [32, 32, 64, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        dec_out_channels = [512, 512, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    elif num_layers == 16:
        enc_out_channels = [32, 32, 32, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        dec_out_channels = [512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32, 3]
        if use_residual:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            dilations = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    else:
        raise NotImplementedError(f"{num_layers} layers not implemented for this model")

    return enc_out_channels, dec_out_channels, dilations