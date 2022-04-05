import torch
from architectures import common_blocks

class InterpolationBlock(torch.nn.Module):
    """
    Interpolation upsampling block.
    """
    def __init__(self,
                 scale_factor,
                 out_size=None,
                 mode="bilinear",
                 align_corners=True,
                 up=True):
        """
        :param scale_factor: multiplier for spatial size
        :param out_size: spatial size of the output tensor for the bilinear interpolation operation
        :param mode: algorithm for upsampling
        :param align corners: whether or not to align the corner pixels of the input and output tensors
        :param up: up or downsample
        """
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor
        self.out_size = out_size
        self.mode = mode
        self.align_corners = align_corners
        self.up = up

    def forward(self, x, size=None):
        if (self.mode == "bilinear") or (size is not None):
            out_size = self.calc_out_size(x) if size is None else size
            return torch.nn.functional.interpolate(
                input=x,
                size=out_size,
                mode=self.mode,
                align_corners=self.align_corners)

        else:
            return torch.nn.functional.interpolate(
                input=x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners)

    def calc_out_size(self, x):
        if self.out_size is not None:
            return self.out_size

        if self.up:
            return tuple(s * self.scale_factor for s in x.shape[2:])

        else:
            return tuple(s // self.scale_factor for s in x.shape[2:])

    def __repr__(self):
        s = '{name}(scale_factor={scale_factor}, out_size={out_size}, mode={mode}, align_corners={align_corners}, up={up})' # noqa
        return s.format(
            name=self.__class__.__name__,
            scale_factor=self.scale_factor,
            out_size=self.out_size,
            mode=self.mode,
            align_corners=self.align_corners,
            up=self.up)

    def calc_flops(self, x):
        assert (x.shape[0] == 1)

        if self.mode == "bilinear":
            num_flops = 9 * x.numel()
        else:
            num_flops = 4 * x.numel()
        num_macs = 0
        return num_flops, num_macs


class Concurrent(torch.nn.Sequential):
    """
    A container for concatenation of modules on the base of the sequential container.
    """
    def __init__(self,
                 axis=1,
                 stack=False,
                 merge_type=None):
        """
        :param axis: the axis on which to concatenate the outputs
        :param stack: whether or not to concatenate tensors along a new dimension
        :param merge_type: type of branch merging
        """
        super(Concurrent, self).__init__()
        assert (merge_type is None) or (merge_type in ["cat", "stack", "sum"])
        self.axis = axis

        if merge_type is not None:
            self.merge_type = merge_type
        else:
            self.merge_type = "stack" if stack else "cat"

    def forward(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))

        if self.merge_type == "stack":
            out = torch.stack(tuple(out), dim=self.axis)

        elif self.merge_type == "cat":
            out = torch.cat(tuple(out), dim=self.axis)

        elif self.merge_type == "sum":
            out = torch.stack(tuple(out), dim=self.axis).sum(self.axis)

        else:
            raise NotImplementedError()

        return out


class FastPyramidPooling(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self,
                 in_channels=128,
                 out_channels=128):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        """
        super(FastPyramidPooling, self).__init__()

        inter_channels = int(in_channels / 4)

        self.conv1 = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=inter_channels)

        self.conv2 = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=inter_channels)

        self.conv3 = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=inter_channels)

        self.conv4 = common_blocks.conv1x1(in_channels=in_channels,
                                          out_channels=inter_channels)

        self.out = common_blocks.conv1x1(in_channels=in_channels * 2,
                                          out_channels=out_channels)

    def pool(self, x, size):
        avgpool = torch.nn.AdaptiveAvgPool2d(output_size=size)

        return avgpool(x)

    def upsample(self, x, size):
        return torch.nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)

        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x
