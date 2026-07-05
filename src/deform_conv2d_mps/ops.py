"""Functional op: deform_conv2d, API-compatible with torchvision.ops.deform_conv2d.

Behaviour:
  * On MPS, inference (no grad required) runs the native Metal forward
    (Phase 2: verified against the torchvision CPU reference).
  * Training on MPS (any input requires grad) still falls back to
    ``torchvision.ops.deform_conv2d`` until the native backward lands
    (Phase 3/4).
  * Non-MPS devices always use the torchvision fallback.
"""

from __future__ import annotations

import os
import importlib
from typing import Optional, Tuple

import torch
from torch import Tensor

# Set DCN_MPS_FORCE_NATIVE=1 to bypass ALL readiness gating and exercise the
# native path unconditionally (testing/diagnostics; grad-requiring calls will
# hit the NotImplementedError in backward until Phase 3).
_FORCE_NATIVE = os.environ.get("DCN_MPS_FORCE_NATIVE", "0") == "1"

_FORWARD_READY = True    # Phase 2 (2026-07-05): forward verified on-device.
_BACKWARD_READY = False  # flip once Phase 3/4 backward passes gradcheck.
_ext = None


def _shader_source() -> str:
    """Read the .metal source and inline its #include of bilinear.metalh."""
    here = os.path.join(os.path.dirname(__file__), "_C")
    with open(os.path.join(here, "bilinear.metalh"), "r") as f:
        header = f.read()
    with open(os.path.join(here, "deform_conv2d.metal"), "r") as f:
        metal = f.read()
    # newLibraryWithSource: can't resolve local includes; inline the header.
    # Strip `#pragma once` since the inlined header is no longer a separate file
    # (it would warn -Wpragma-once-outside-header).
    header = "\n".join(
        ln for ln in header.splitlines() if ln.strip() != "#pragma once")
    metal = metal.replace('#include "bilinear.metalh"', header)
    return metal


def _load_native():
    """Import the compiled extension and compile the Metal library once."""
    global _ext
    if _ext is not None:
        return _ext
    _ext = importlib.import_module("deform_conv2d_mps._C_ext")
    _ext._compile_library(_shader_source())
    return _ext


def _pair(x) -> Tuple[int, int]:
    if isinstance(x, (tuple, list)):
        return int(x[0]), int(x[1])
    return int(x), int(x)


class _DeformConv2dFunction(torch.autograd.Function):
    """Autograd wrapper around the native MPS op.

    Forward calls the native op. Backward (Phase 3) dispatches the col2im /
    col2im_coord kernels for grad_input/offset/mask and uses torch ops for
    grad_weight/bias. Currently raises until the kernels land.
    """

    @staticmethod
    def forward(ctx, input, weight, offset, mask, bias,
                stride, padding, dilation, groups, deformable_groups):
        ext = _load_native()
        sh, sw = stride
        ph, pw = padding
        dh, dw = dilation
        out = ext_op_forward(ext, input, weight, offset, mask, bias,
                             sh, sw, ph, pw, dh, dw, groups, deformable_groups)
        ctx.save_for_backward(input, weight, offset, mask,
                              bias if bias is not None else torch.empty(0))
        ctx.params = (stride, padding, dilation, groups, deformable_groups)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # TODO(Phase 3): dispatch deformable_col2im + deformable_col2im_coord.
        raise NotImplementedError(
            "Native MPS backward not implemented yet (scaffold). "
            "Use the torchvision fallback for training until Phase 3 lands.")


def ext_op_forward(ext, input, weight, offset, mask, bias,
                   sh, sw, ph, pw, dh, dw, groups, deformable_groups):
    return torch.ops.deform_conv2d_mps.deform_conv2d_forward(
        input, weight, offset,
        mask if mask is not None else torch.empty(0, device=input.device),
        bias, sh, sw, ph, pw, dh, dw, groups, deformable_groups)


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Deformable Conv2d (DCNv2; DCNv1 when ``mask is None``).

    Signature matches ``torchvision.ops.deform_conv2d``. ``deformable_groups``
    is inferred from ``offset`` channels, as torchvision does.
    """
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # Route native only when autograd won't be needed (backward is Phase 3);
    # _FORCE_NATIVE bypasses the gating entirely for testing.
    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad
        for t in (input, weight, offset, mask, bias))
    use_native = input.device.type == "mps" and (
        _FORCE_NATIVE
        or (_FORWARD_READY and (not needs_grad or _BACKWARD_READY)))
    if use_native:
        kh, kw = weight.shape[-2], weight.shape[-1]
        groups = 1  # extend when groups support lands
        deformable_groups = offset.shape[1] // (2 * kh * kw)
        return _DeformConv2dFunction.apply(
            input, weight, offset, mask, bias,
            stride, padding, dilation, groups, deformable_groups)

    # Fallback: torchvision reference (CPU/CUDA, or MPS via CPU round-trip).
    import torchvision.ops as tvops
    return tvops.deform_conv2d(
        input, offset, weight, bias=bias,
        stride=stride, padding=padding, dilation=dilation, mask=mask)
