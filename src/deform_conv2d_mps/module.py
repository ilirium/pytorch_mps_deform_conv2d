"""DeformConv2d nn.Module — mirrors torchvision.ops.DeformConv2d.

Holds the conv weight/bias and delegates to the functional ``deform_conv2d``.
Offsets (and, for DCNv2, the modulation mask) are produced by the caller,
exactly as in torchvision, so state dicts are interchangeable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .ops import deform_conv2d, _pair


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        kh, kw = self.kernel_size
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels // self.groups
            for d in self.kernel_size:
                fan_in *= d
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        return deform_conv2d(
            input, offset, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=mask)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"padding={self.padding}, dilation={self.dilation}, "
                f"groups={self.groups}, bias={self.bias is not None}")
