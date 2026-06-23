"""Deformable Conv2d (DCNv1/v2) with a native Metal (MPS) kernel for Apple Silicon.

Public API mirrors torchvision:
    from deform_conv2d_mps import deform_conv2d, DeformConv2d
"""

from .ops import deform_conv2d
from .module import DeformConv2d

__all__ = ["deform_conv2d", "DeformConv2d"]
__version__ = "0.0.1"
