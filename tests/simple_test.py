import torch
import deform_conv2d_mps
from deform_conv2d_mps.ops import _load_native

ext = _load_native()
x = torch.arange(5.,device='mps')
print(torch.ops.deform_conv2d_mps.add_one(x))
