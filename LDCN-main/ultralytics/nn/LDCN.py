import torch
import torch.nn as nn
from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv,DWConvTranspose2d,Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,RTDETRDecoder, Segment,LightConv, RepConv, SpatialAttention)
import math

class LDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, deformable_groups=1):
        super(LDConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = Conv.default_act
        self.min_offset = nn.Parameter(torch.tensor(-5.0))
        self.max_offset = nn.Parameter(torch.tensor(5.0))
        self.reset_parameters()




    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)

        offset = torch.cat((o1, o2), dim=1)
        offset = torch.clamp(offset, self.min_offset, self.max_offset)  # 固定阈值
        mask_to_zero = (mask > self.max_offset) | (mask < self.min_offset)
        mask = mask.masked_fill(mask_to_zero, 999)
        mask = torch.sigmoid(mask)

        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class L_LDConv(nn.Module):   #lighting LDconv
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels

        self.cv1 = LDConv(in_channels=c_, out_channels=c_, kernel_size=3, stride=1, dilation=1, groups=1,
                         deformable_groups=1)
        self.cv2 = LDConv(in_channels=c1, out_channels=c2, kernel_size=k, stride=s, dilation=1, groups=1,
                         deformable_groups=1)
        self.c1 = c1
        self.c2 = c2

    def forward(self, x):
        if self.c1 == self.c2:
            y = torch.chunk(x, chunks=2, dim=1)
            y1 = self.cv1(y[0])
            y2 = torch.cat((y[1], y1), 1)
        else:
            y2 = self.cv2(x)

        b, n, h, w = y2.data.size()
        b_n = b * n // 2
        z = y2.reshape(b_n, 2, h * w)
        z = z.permute(1, 0, 2)
        z = z.reshape(2, -1, n // 2, h, w)

        return torch.cat((z[0], z[1]), 1)




class B_PGBottleneck(nn.Module):  #Balance

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = L_LDConv(c1, c_, k[0], 1)
        self.cv2 = LDConv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class L_PGBottleneck(nn.Module):  #Lighting

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = L_LDConv(c1, c_, k[0], 1)
        self.cv2 = L_LDConv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class N_PGBottleneck(nn.Module):  #Normal

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = LDConv(c_, c2, k[1], 1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))





class LDC2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(N_PGBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))





