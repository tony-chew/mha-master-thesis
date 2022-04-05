import torch
from architectures import pyramid_pooling


def check_model(input_res):
    """
    Check the conditions of the input resolution, ie must be factors of 8 or 16, aspect ratio of 2 and correctly
    specified number of layers and residual layers
    """
    assert ((input_res.height % 16 == 0) and (input_res.width % 16 == 0)), "Input resolution not the " \
                                                                                     "correct factor"
    assert (input_res.width / input_res.height == 2), "Undesired aspect ratio of input resolution"


def get_bottleneck_res(input_res,
                       downsample_factor=8):
    """
    Gets the bottleneck resolution of the convolutional autoencoder
    """
    bottleneck_h = input_res.height / downsample_factor
    bottleneck_w = input_res.width / downsample_factor

    return int(bottleneck_h), int(bottleneck_w)


class Reshape(torch.nn.Module):
    """
    Reshape a tensor to a new shape.
    """

    def __init__(self,
                 shape):
        """
        :param shape: new shape of the tensor
        """
        super(Reshape, self).__init__()

        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ConvBlock(torch.nn.Module):
    """
    Standard convolutional block followed by batch norm, activation and dropout. Order of layers as follows:

    1. upsample if enabled
    2. (n x n) conv2d layer
    3. batch normalisation
    4. activation function (ReLU)
    5. dropout if enabled
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bn=True,
                 activation=True,
                 upsample_factor=None,
                 dropout_rate=None):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel; either an integer or a tuple
        :param stride: stride of the convolutional layer
        :param padding: padding of the convolutional layer
        :param dilation: the space between kernel size to conduct convolution, default 1
        :param use_bn: tag for whether if batch normalisation is used or not
        :param dropout_rate: dropout used at the end of the cell; None = no dropout is applied
        :param upsample_factor: factor to be used for upsampling; None = no upsampling is applied
        """
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_activ = activation
        self.use_dropout = dropout_rate
        self.upsample_factor = upsample_factor

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=groups)

        if self.upsample_factor is not None:
            self.do_upsampling = torch.nn.Upsample(scale_factor=self.upsample_factor)

        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)

        if self.use_activ:
            self.activation = torch.nn.ReLU(inplace=True)

        if self.use_dropout is not None:
            self.dropout = torch.nn.Dropout(p=self.use_dropout)

    def forward(self, x):
        if self.upsample_factor is not None:
            x = self.do_upsampling(x)

        x = self.conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.use_activ:
            x = self.activation(x)

        if self.use_dropout is not None:
            x = self.dropout(x)

        return x


def conv1x1(in_channels,
            out_channels,
            stride=1,
            padding=0,
            use_bn=True,
            activation=True):
    """
    Returns a simple 1x1 pointwise convolution
    """
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=padding,
                     use_bn=use_bn,
                     activation=activation)


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=0,
            dilation=1,
            use_bn=True,
            activation=True,
            upsample_factor=None):
    """
    Returns a 3x3 convolution operation
    """
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     use_bn=use_bn,
                     activation=activation,
                     upsample_factor=upsample_factor)


def conv5x5(in_channels,
            out_channels,
            stride=1,
            padding=2,
            dilation=1,
            use_bn=True,
            activation=True,
            upsample_factor=None):
    """
    Returns a 5x5 convolution operation
    """
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=5,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     use_bn=use_bn,
                     activation=activation,
                     upsample_factor=upsample_factor)


def conv7x7(in_channels,
            out_channels,
            stride=1,
            padding=3,
            dilation=1,
            use_bn=True,
            activation=True,
            upsample_factor=None):
    """
    Returns a 7x7 convolution operation
    """
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=7,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     use_bn=use_bn,
                     activation=activation,
                     upsample_factor=upsample_factor)


class DeConvBlock(torch.nn.Module):
    """
    Standard deconvolutional block followed by batch norm, activation and dropout. Order of layers as follows:

    1. upsample if enabled
    2. (n x n) deconv2d layer
    3. batch normalisation
    4. activation function (ReLU)
    5. dropout if enabled
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 ext_padding=None,
                 output_padding=0,
                 dilation=1,
                 groups=1,
                 use_bn=True,
                 activation=True):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: size of the kernel; either an integer or a tuple
        :param stride: stride of the convolutional layer
        :param padding: padding of the convolutional layer
        :param ext_padding: extra padding value for deconvolution layer
        :param output_padding: extra padding on top of output
        :param dilation: the space between kernel size to conduct convolution, default 1
        :param use_bn: tag for whether if batch normalisation is used or not
        :param dropout_rate: dropout used at the end of the cell; None = no dropout is applied
        """
        super(DeConvBlock, self).__init__()
        self.use_bn = use_bn
        self.use_activ = activation
        self.use_padding = (ext_padding is not None)

        if self.use_padding:
            self.pad = torch.nn.ZeroPad2d(padding=ext_padding)

        self.conv = torch.nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             output_padding=output_padding,
                                             dilation=dilation,
                                             groups=groups)

        if self.use_bn:
            self.bn = torch.nn.BatchNorm2d(num_features=out_channels)

        if self.use_activ:
            self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_padding:
            x = self.pad(x)

        x = self.conv(x)

        if self.use_bn:
            x = self.bn(x)

        if self.use_activ:
            x = self.activation(x)

        return x


class DWSepConvBlock(torch.nn.Module):
    """
    A depthwise seperable convolutional block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 dw_use_bn=True,
                 pw_use_bn=True,
                 dw_activ=True,
                 pw_activ=True,
                 upsample_factor=None):
        """
        :param in_channels: no. of input channels
        :param out_channels: no. of output channels
        :param kernel_size: desired kernel sizing
        :param stride: desired convolutional stride
        :param padding: desired padding of input
        :param dilation: desired dilation rate
        :param dw_use_bn: whether or not to use BN for the DW section
        :param pw_use_bn: whether or not to use BN for the PW section
        :param dw_activ: whether or not to use ReLU for the DW section
        :param pw_activ: whether or not to use ReLU for the PW section
        :param upsample_factor: if decoding, use this upsample factor
        """
        super(DWSepConvBlock, self).__init__()

        self.dw_conv = ConvBlock(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 use_bn=dw_use_bn,
                                 activation=dw_activ,
                                 upsample_factor=upsample_factor)

        self.pw_conv = conv1x1(in_channels=in_channels,
                               out_channels=out_channels,
                               use_bn=pw_use_bn,
                               activation=pw_activ)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)

        return x


def dws_conv3x3(in_channels,
                out_channels,
                stride=1,
                padding=1,
                dilation=1,
                upsample_factor=None):
    """
    A 3x3 version of the depthwise seperable convolutional block
    """
    return DWSepConvBlock(in_channels=in_channels,
                          out_channels=out_channels,
                          stride=stride,
                          kernel_size=3,
                          padding=padding,
                          dilation=dilation,
                          upsample_factor=upsample_factor)


class BottleneckReduction(torch.nn.Module):
    """
    A depth compression module for the bottleneck of an autoencoder comprising of a series of 1x1 standard convolutions
    1x1 standard convolutions do not add extra parameters and serve as a means to compress the depth only
    """
    def __init__(self,
                 channels,
                 upsample=False):
        """
        :param in_channels: the number of input depth channels
        :param out_channels: the number of output depth channels
        """
        super(BottleneckReduction, self).__init__()

        in_channels = channels[0]
        out_channels = channels[-1]
        mid_channels = channels[1:-1]

        if upsample:
            in_channels, out_channels = out_channels, in_channels
            mid_channels = list(reversed(mid_channels))

        self.bottleneck = torch.nn.Sequential()

        for i, mid_channel in enumerate(mid_channels):
            unit = conv1x1(in_channels=in_channels,
                           out_channels=mid_channel,
                           use_bn=False,
                           activation=False)
            self.bottleneck.add_module(f"stage{i+1}", unit)

            in_channels = mid_channel

        self.bottleneck.add_module("lastbottleneck", conv1x1(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             use_bn=True,
                                                             activation=True))

    def forward(self, x):
        return self.bottleneck(x)


class ERFNetDownsampler(torch.nn.Module):
    """
    A downsampling module for the convolutional autoencoder if residual modules are utilised. Based on the downsampling
    block from ENet also taken by ERFNet

    1. A maxpooling operation of stride 2 is conducted
    2. A 3x3 convolutional operation of stride 2 is conducted in conjunction
    3. The outputs of 1 and 2 are concatenated to form a downsampled image shape
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        :param in_channels: the number of input depth channels
        :param out_channels: the number of output depth channels
        """
        super(ERFNetDownsampler, self).__init__()

        self.pool = torch.nn.MaxPool2d(kernel_size=2,
                                       stride=2)
        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels-in_channels,
                                    stride=2,
                                    kernel_size=3,
                                    padding=1,
                                    dilation=1)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.activ = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        y1 = self.pool(x)
        y2 = self.conv(x)

        x = torch.cat((y2, y1), dim=1)
        x = self.bn(x)
        x = self.activ(x)

        return x


class ERFNetAsymConvBlock(torch.nn.Module):
    """
    An asymmetric convolutional block for use in the ERFNet Factorised Convolutional Module. Image resolution does not
    change

    1. Left-wise convolutional block
        - a (kernel_size, 1) filter size, same concept for padding and dilation
        - no batch normalisation conducted at this stage
    2. Right-wise convolutional block
        - a (1, kernel_size) filter size, same concept for padding and dilation
        - batch normalisation conducted at this stage (in effect this is equivalent to (kernel_size, kernel_size)
            convolutional operation, therefore batch norm can be conducted here
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 lw_use_bn=False,
                 lw_use_activ=True,
                 rw_use_activ=True):
        """
        :param channels: the desired depth channels of the block
        :param kernel_size: the size of the filter
        :param stride: the stride for the filter to conduct convolution
        :param padding: padding sizing to the edge of the image
        :param dilation: dilation size for filter to conduct convolution
        :param lw_use_activ: if the left-wise block should conduct a non-activation function
        :param rw_use_activ: if the right-wise block should conduct a non-activation function
        """
        super(ERFNetAsymConvBlock, self).__init__()

        self.lw_conv = ConvBlock(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=(kernel_size, 1),
                                 stride=stride,
                                 padding=(padding, 0),
                                 dilation=(dilation, 1),
                                 use_bn=lw_use_bn,
                                 activation=lw_use_activ)

        self.rw_conv = ConvBlock(in_channels=channels,
                                 out_channels=channels,
                                 kernel_size=(1, kernel_size),
                                 stride=stride,
                                 padding=(0, padding),
                                 dilation=(1, dilation),
                                 activation=rw_use_activ)

    def forward(self, x):
        return self.rw_conv(self.lw_conv(x))


class ERFNetFactorisedResidualBlock(torch.nn.Module):
    """
    A factorised residual block from ERFNet. Named the Non-Bottleneck-1D Module in its paper

    1. 1st factorised convolutional block
        - ReLU activation function used at the output of each convolution
    2. 2nd factorised convolutional block
        - Dilated convolutions utilised if explicitly stated
        - ReLU activation not used at the output of this block
    3. Identity mapping
        - The famous residual identity mapping is conducted (x = x + identity)
    4. ReLU and dropout applied

    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate,
                 kernel_size=3,
                 stride=1):
        """
        :param channels: the desired depth channels of the block
        :param kernel_size: the size of the filter
        :param stride: the stride for the filter to conduct convolution
        :param dilation: dilation size for filter to conduct convolution
        :param dropout_rate: the dropout_rate to use
        """
        super(ERFNetFactorisedResidualBlock, self).__init__()

        padding1 = (kernel_size - 1) // 2
        padding2 = dilation * padding1
        self.use_dropout = dropout_rate

        self.asym_conv1 = ERFNetAsymConvBlock(channels=channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding1)

        self.asym_conv2 = ERFNetAsymConvBlock(channels=channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding2,
                                              dilation=dilation,
                                              rw_use_activ=False)

        if self.use_dropout is not None:
            self.dropout = torch.nn.Dropout(p=self.use_dropout)

        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.asym_conv1(x)
        x = self.asym_conv2(x)

        if self.use_dropout is not None:
            x = self.dropout(x)

        x = x + identity
        x = self.activation(x)

        return x


class FastSCNNLearningToDownsample(torch.nn.Module):
    """
    The Learning to Downsample Block for FastSCNN. If used in the decoder, the process is essentially reversed.
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=[32, 48, 64],
                 upsample_factor=None):
        """
        :param in_channels: the number of input channels (3 for image input)
        :param out_channels: the output number of channels for each of the layers in this block
        :param upsample_factor: tell the decoder to perform upsampling if not None
        """
        super(FastSCNNLearningToDownsample, self).__init__()
        self.upsample_factor = upsample_factor

        self.layer1 = conv3x3(in_channels=in_channels if upsample_factor is None else out_channels[1],
                              out_channels=out_channels[0] if upsample_factor is None else out_channels[2],
                              stride=2 if upsample_factor is None else 1,
                              padding=1,
                              upsample_factor=upsample_factor)

        self.layer2 = dws_conv3x3(in_channels=out_channels[0],
                                  out_channels=out_channels[1],
                                  stride=2 if upsample_factor is None else 1,
                                  upsample_factor=upsample_factor)

        self.layer3 = dws_conv3x3(in_channels=out_channels[1] if  upsample_factor is None else in_channels,
                                  out_channels=out_channels[2] if upsample_factor is None else out_channels[0],
                                  stride=2 if upsample_factor is None else 1,
                                  upsample_factor=upsample_factor)

    def forward(self, x):
        if self.upsample_factor is None:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        else:
            x = self.layer3(x)
            x = self.layer2(x)
            x = self.layer1(x)

        return x


class FastSCNNResidualBlock(torch.nn.Module):
    """
    Resiudal blocks taken for use in FastSCNN - inspired by MobileNetV2
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 exp_factor=6,
                 upsample_factor=None):
        """
        :param in_channels: no of input channels
        :param out_channels: no of output channels
        :param stride: the desired convolutional stride
        :param exp_factor: expansion factor for the mid channels
        :param upsample_factor: for downsampling, take the upsample factor
        """
        super(FastSCNNResidualBlock, self).__init__()

        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channel = in_channels * exp_factor

        self.layer1 = conv1x1(in_channels=in_channels,
                              out_channels=mid_channel)

        self.layer2 = ConvBlock(in_channels=mid_channel,
                                out_channels=mid_channel,
                                stride=stride,
                                kernel_size=3,
                                padding=1,
                                use_bn=True,
                                activation=True,
                                groups=mid_channel,
                                upsample_factor=upsample_factor)

        self.layer3 = conv1x1(in_channels=mid_channel,
                              out_channels=out_channels,
                              activation=False)

    def forward(self, x):
        if self.residual:
            identity = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.residual:
            x = x + identity

        return x


class FastSCNNGlobalFeatureExtractor(torch.nn.Module):
    """
    The Global feature extractor module for the FastSCNN. For the decoder, the process is essentially reversed.
    """
    def __init__(self,
                 in_channels=64,
                 out_channels=[[64, 64, 64], [96, 96, 96], [128, 128, 128]],
                 upsample_factor=None):
        """
        :param in_channels: no of input channels
        :param out_channels: no of output channels
        :param upsample_factor: take the upsample factor for the encoder
        """
        super(FastSCNNGlobalFeatureExtractor, self).__init__()

        self.low_branch = torch.nn.Sequential()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_factor = upsample_factor

        self.build_low_branch()

    def build_low_branch(self):
        for i, channels_per_stage in enumerate(self.out_channels):
            stage = torch.nn.Sequential()

            for j, out_channel in enumerate(channels_per_stage):
                stride = 2 if (j == 0 and i != len(self.out_channels) - 1 and self.upsample_factor is None) else 1

                if j == 2 and i == 1 and self.upsample_factor is not None:
                    stage.add_module(f"unit {j + 1}", ConvBlock(in_channels=self.in_channels,
                                                                out_channels=out_channel,
                                                                kernel_size=3,
                                                                stride=1,
                                                                padding=1,
                                                                use_bn=True,
                                                                activation=True,
                                                                upsample_factor=self.upsample_factor))
                else:
                    stage.add_module(f"unit {j + 1}", FastSCNNResidualBlock(in_channels=self.in_channels,
                                                                            out_channels=out_channel,
                                                                            stride=stride))
                self.in_channels=out_channel

            self.low_branch.add_module(f"stage {i + 1}", stage)

    def forward(self, x):
        return self.low_branch(x)


class FastSCNNFeatureFusion(torch.nn.Module):
    """
    The feature fusion block
    """
    def __init__(self,
                 xhigh_in_channels,
                 xlow_in_channels,
                 out_channels,
                 xhigh_in_size):
        """
        :param xhigh_in_size: no of input channels for the high res branch
        :param xlow_in_channels: no of input channels for the low res branch
        :param out_channels: no of output channels of the fusion block
        :param xhigh_in_size: spatial size of high res branch
        """
        super(FastSCNNFeatureFusion, self).__init__()

        self.xhigh_in_size = xhigh_in_size

        self.upsample = pyramid_pooling.InterpolationBlock(scale_factor=None,
                                                           out_size=xhigh_in_size)

        self.low_dwconv = ConvBlock(in_channels=xlow_in_channels,
                                    out_channels=out_channels,
                                    stride=1,
                                    kernel_size=3,
                                    padding=1,
                                    use_bn=True,
                                    activation=True,
                                    groups=out_channels)

        self.low_pwconv = conv1x1(in_channels=out_channels,
                                  out_channels=out_channels,
                                  activation=False)

        self.high_conv = conv1x1(in_channels=xhigh_in_channels,
                                 out_channels=out_channels,
                                 activation=False)

        self.activ = torch.nn.ReLU(inplace=True)

    def forward(self, xhigh, xlow):
        xhigh_in_size = self.xhigh_in_size if self.xhigh_in_size is not None else x.shape[2:]

        xlow = self.upsample(xlow, xhigh_in_size)
        xlow = self.low_dwconv(xlow)
        xlow = self.low_pwconv(xlow)

        xhigh = self.high_conv(xhigh)

        x = xlow + xhigh
        return self.activ(x)


class FastSCNNClassifier(torch.nn.Module):
    """
    Classifier block for fastscnn
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        """
        :param in_channels: no of input channels
        :param num_classes: no of classes
        """
        super(FastSCNNClassifier, self).__init__()

        self.conv1 = dws_conv3x3(in_channels=in_channels,
                                 out_channels=in_channels)

        self.conv2 = dws_conv3x3(in_channels=in_channels,
                                 out_channels=in_channels)

        self.dropout = torch.nn.Dropout(p=0.1,
                                        inplace=False)

        self.conv3 = conv1x1(in_channels=in_channels,
                             out_channels=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)

        return x


class LEDNetChannelShuffle(torch.nn.Module):
    """
    Channel shuffle layer. This is a wrapper over the same operation. It is designed to save the number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        """
        :param channels: no of channels
        :param groups: no of groups
        """
        super(LEDNetChannelShuffle, self).__init__()

        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.groups = groups

    def channel_shuffle(self, x):
        """
        Channel shuffle operation from 'ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices,'

        :param x: input tensor
        """
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)

        return x

    def forward(self, x):
        return self.channel_shuffle(x)

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


class LEDNetBranch(torch.nn.Module):
    """
    A branch of the SS-nbt module from LEDNet
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate):
        """
        :param channels: the channels to perform forward pass on
        :param dilation: the dilation rate, to be performed only on the 2nd assymetrical convolutional block
        :param dropout_rate: the desired dropout rate
        """
        super(LEDNetBranch, self).__init__()

        self.asym_conv1 = ERFNetAsymConvBlock(channels=channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1)

        self.asym_conv2 = ERFNetAsymConvBlock(channels=channels,
                                              kernel_size=3,
                                              stride=1,
                                              padding=dilation,
                                              dilation=dilation,
                                              rw_use_activ=False)

        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.asym_conv1(x)
        x = self.asym_conv2(x)

        if self.dropout_rate is not None:
            x = self.dropout(x)

        return x


class LEDNetResidualBlock(torch.nn.Module):
    """
    The SS-nbt module from LEDNet
    """
    def __init__(self,
                 channels,
                 dilation,
                 dropout_rate):
        """
        :param channels: the channels to perform forward pass on
        :param dilation: the dilation rate, to be performed only on the 2nd assymetrical convolutional block
        :param dropout_rate: the desired dropout rate
        """
        super(LEDNetResidualBlock, self).__init__()

        mid_channels = channels // 2

        self.left_branch = LEDNetBranch(channels=mid_channels,
                                        dilation=dilation,
                                        dropout_rate=dropout_rate)

        self.right_branch = LEDNetBranch(channels=mid_channels,
                                         dilation=dilation,
                                         dropout_rate=dropout_rate)

        self.activation = torch.nn.ReLU(inplace=True)

        self.shuffle = LEDNetChannelShuffle(channels=channels,
                                            groups=2)

    def forward(self, x):
        identity = x

        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.left_branch(x1)
        x2 = self.right_branch(x2)

        x = torch.cat((x1, x2), dim=1)

        x = x + identity
        x = self.activation(x)
        x = self.shuffle(x)

        return x


class LEDNetPoolingBranch(torch.nn.Module):
    """
    LEDNet Pooling branch (the right-most branch in the architecture figure in the paper)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 down_size):
        """
        :param in_channels: no of channels for the input
        :param out_channels: no of channels for the output
        :param in_size: input image size
        :param down_size: downscaled image size
        """
        super(LEDNetPoolingBranch, self).__init__()

        self.pool = torch.nn.AdaptiveAvgPool2d(output_size=down_size)

        self.conv = conv1x1(in_channels=in_channels,
                            out_channels=out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


class LEDNetAPNModule(torch.nn.Module):
    """
    Attention pyramid network block
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        :param in_channels: no of input channels
        :param out_channels: no of output channels
        :param in_size: input image size
        """
        super(LEDNetAPNModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_out_channels = 1

        self.pool_branch = LEDNetPoolingBranch(in_channels=in_channels,
                                               out_channels=out_channels,
                                               down_size=1)

        self.mid = conv1x1(in_channels=in_channels,
                            out_channels=out_channels)

        self.down_stage_1 = conv7x7(in_channels=in_channels,
                                out_channels=self.att_out_channels,
                                stride=2)

        self.down_stage_2 = conv5x5(in_channels=self.att_out_channels,
                            out_channels=self.att_out_channels,
                            stride=2)

        self.down_stage_3 = torch.nn.Sequential(conv3x3(in_channels=self.att_out_channels,
                                                 out_channels=self.att_out_channels,
                                                 stride=2,
                                                 padding=1),
                                         conv3x3(in_channels=self.att_out_channels,
                                                 out_channels=self.att_out_channels,
                                                 padding=1))

        self.conv_stage_1 = conv7x7(in_channels=self.att_out_channels,
                             out_channels=self.att_out_channels)
        self.conv_stage_2 = conv5x5(in_channels=self.att_out_channels,
                            out_channels=self.att_out_channels)


    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        b1 = self.pool_branch(x)
        b1 = torch.nn.functional.interpolate(b1,
                                             size=(h, w),
                                             mode="bilinear",
                                             align_corners=True)
        mid = self.mid(x)

        x1 = self.down_stage_1(x)
        x2 = self.down_stage_2(x1)
        x3 = self.down_stage_3(x2)
        x3 = torch.nn.functional.interpolate(x3,
                                             size=(h // 4, w // 4),
                                             mode="bilinear",
                                             align_corners=True)

        x2 = self.conv_stage_2(x2)
        x = x2 + x3
        x = torch.nn.functional.interpolate(x,
                                            size=(h // 2, w // 2),
                                            mode="bilinear",
                                            align_corners=True)

        x1 = self.conv_stage_1(x1)
        x = x + x1
        x = torch.nn.functional.interpolate(x,
                                            size=(h, w),
                                            mode="bilinear",
                                            align_corners=True)

        x = x * mid
        x = x + b1

        return x


class DABNetDualPathSequential(torch.nn.Sequential):
    """
    A sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2))):
        """
        :param return_two: whether to return two outputs after execution
        :param first_ordinals: no of first modules with single input/output
        :param last_ordinals: no of final modules with single input/output
        :param dual_path_scheme: scheme of dual path response for a module (function)
        :param dual_path_scheme_ordinal: scheme of dual path response for an ordinal module (function)
        """
        super(DABNetDualPathSequential, self).__init__()

        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def forward(self, x1, x2=None):
        length = len(self._modules.values())

        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)

        if self.return_two:
            return x1, x2
        else:
            return x1


class DABNetInitialBlocks(torch.nn.Module):
    """
    The initial convolutional blocks of DABNet
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 upsample=False):
        """
        :param in_channels: no of input channels
        :param out_channels: no of output channels
        :param upsample: if True, reverse the process
        """
        super(DABNetInitialBlocks, self).__init__()

        self.upsample=upsample
        self.conv1 = conv3x3(in_channels=in_channels if not self.upsample else out_channels,
                             out_channels=out_channels if not self.upsample else in_channels,
                             padding=1,
                             stride=2 if not self.upsample else 1,
                             upsample_factor=None if not self.upsample else 2,
                             use_bn=True if not self.upsample else False,
                             activation=True if not self.upsample else False)

        self.conv2 = conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             padding=1)

        self.conv3 = conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             padding=1)

    def forward(self, x):
        if not self.upsample:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)

        else:
            x = self.conv3(x)
            x = self.conv2(x)
            x = self.conv1(x)

        return x


class DABNetResidualBlock(torch.nn.Module):
    """
    The proposed residual block for the DABNet
    """
    def __init__(self,
                 channels,
                 dilation):
        """
        :param channels: no of channels through the module
        :param dilation: the dilation rate to use
        """
        super(DABNetResidualBlock, self).__init__()

        mid_channels = channels // 2

        self.bn1 = torch.nn.BatchNorm2d(num_features=channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=mid_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_channels=channels,
                             out_channels=mid_channels,
                             padding=1)

        self.branches = pyramid_pooling.Concurrent(stack=True)
        self.branches.add_module("left_branch", ERFNetAsymConvBlock(channels=mid_channels,
                                                                    kernel_size=3,
                                                                    stride=1,
                                                                    padding=1,
                                                                    lw_use_bn=True))

        self.branches.add_module("right_branch", ERFNetAsymConvBlock(channels=mid_channels,
                                                                     kernel_size=3,
                                                                     stride=1,
                                                                     padding=dilation,
                                                                     dilation=dilation,
                                                                     lw_use_bn=True))

        self.conv2 = conv1x1(in_channels=mid_channels,
                             out_channels=channels,
                             use_bn=False,
                             activation=False)

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv1(x)
        x = self.branches(x)
        x = x.sum(dim=1)

        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = x + identity

        return x


class DABNetDownBlock(torch.nn.Module):
    """
    DABNet specific downsample block for the main branch.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        """
        :param in_channels: no of input channels
        :param out_channels: no of output channels
        """
        super(DABNetDownBlock, self).__init__()

        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        self.conv = conv3x3(in_channels=in_channels,
                            out_channels=mid_channels,
                            stride=2,
                            padding=1,
                            use_bn=False,
                            activation=False)

        if self.expand:
            self.pool = torch.nn.MaxPool2d(kernel_size=2,
                                           stride=2)

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = torch.cat((y, z), dim=1)

        y = self.bn(y)
        y = self.relu(y)

        return y


class DABNetBlock(torch.nn.Module):
    """
    The DABNet block consisting of the downsampling operation following the DAB Modules
    The no of channels is split between the downsampling operation and residual modules and concatenated at the end
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations):
        """
        :param in_channels: the no of channels for the downsample operation
        :param out_channels: the no of channels out of the residual bottleneck modules
        :param dilations: a list of dilations
        """
        super(DABNetBlock, self).__init__()

        mid_channels = out_channels // 2

        self.downsample = DABNetDownBlock(in_channels=in_channels,
                                          out_channels=mid_channels)

        self.blocks = torch.nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.blocks.add_module(f"block{i + 1}", DABNetResidualBlock(channels=mid_channels,
                                                                        dilation=dilation))

    def forward(self, x):
        x = self.downsample(x)
        y = self.blocks(x)

        x = torch.cat((y, x), dim=1)

        return x


class DABNetDownStage(torch.nn.Module):
    """
    The DABnet block stage including the 2D average pooling operation from the input
    """
    def __init__(self,
                 pool_channels,
                 in_channels,
                 out_channels,
                 dilations):
        """
        :param pool_channels: no of channels in the original input image (default 3)
        :param in_channels: no of input channels for the block
        :param out_channels: no of output channels for the block
        :param dilations: a list of dilations
        """
        super(DABNetDownStage, self).__init__()

        self.use_unit = len(dilations) > 0

        self.pool_down = torch.nn.AvgPool2d(kernel_size=3,
                                            stride=2,
                                            padding=1)

        if self.use_unit:
            self.unit = DABNetBlock(in_channels=in_channels,
                                    out_channels=out_channels - pool_channels,
                                    dilations=dilations)

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x, pool):
        pool = self.pool_down(pool)

        if self.use_unit:
            x = self.unit(x)

        x = torch.cat((x, pool), dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x, pool


class DABNetUpStage(torch.nn.Module):
    """
    The upsampling process of DABNet for the outfitted autoencoder
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations,
                 upsample_factor):
        """
        :param in_channels: no of input channels for the block
        :param out_channels: no of output channels for the block
        :param dilations: a list of dilations
        :param upsample_factor: the upsampling factor
        """
        super(DABNetUpStage, self).__init__()

        self.blocks = torch.nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.blocks.add_module(f"block{i + 1}", DABNetResidualBlock(channels=in_channels,
                                                                        dilation=dilation))

        self.upsample = ConvBlock(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  upsample_factor=upsample_factor)

    def forward(self, x):
        x = self.blocks(x)
        x = self.upsample(x)

        return x


class EDANetResidualBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 dropout_rate,
                 growth_rate=40,
                 encoder=True):
        super(EDANetResidualBlock, self).__init__()

        self.use_dropout = dropout_rate != 0.0
        self.encoder = encoder
        mid_channels = out_channels - in_channels if encoder else in_channels - out_channels

        self.conv1 = conv1x1(in_channels=in_channels,
                             out_channels=mid_channels)

        if not encoder:
            self.identity = conv1x1(in_channels=in_channels,
                                    out_channels=in_channels - (2 * growth_rate))

        self.conv2 = ERFNetAsymConvBlock(channels=mid_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         dilation=1,
                                         lw_use_bn=False,
                                         lw_use_activ=False,
                                         rw_use_activ=True)

        self.conv3 = ERFNetAsymConvBlock(channels=mid_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=dilation,
                                         dilation=dilation,
                                         lw_use_bn=False,
                                         lw_use_activ=False,
                                         rw_use_activ=True)

        if self.use_dropout:
            self.dropout = torch.nn.Dropout(p=dropout_rate)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        if not self.encoder:
            identity = self.identity(identity)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.use_dropout:
            x = self.dropout(x)

        x = torch.cat((x, identity), dim=1)
        x = self.relu(x)

        return x