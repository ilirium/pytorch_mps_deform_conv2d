"""Forward correctness: native MPS output vs torchvision CPU reference.

Parametrised across kernel size, stride, padding, dilation, and v1/v2 (mask).
Skips automatically when MPS is unavailable. Set DCN_MPS_FORCE_NATIVE=1 to run
against the native kernel once Phase 1 is implemented; otherwise the package
falls back to torchvision and the test trivially passes (reference vs itself).
"""

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.ops import deform_conv2d as tv_deform_conv2d  # noqa: E402

from deform_conv2d_mps import deform_conv2d  # noqa: E402

from conftest import requires_mps  # noqa: E402

KERNELS = [(3, 3), (1, 1), (3, 1)]
STRIDES = [1, 2]
PADS = [0, 1]
DILATIONS = [1, 2]
USE_MASK = [False, True]


def _make_inputs(kh, kw, stride, pad, dil, use_mask, device, dtype=torch.float32):
    torch.manual_seed(0)
    N, inC, outC, H, W = 2, 4, 6, 9, 9
    dg = 1
    out_h = (H + 2 * pad - dil * (kh - 1) - 1) // stride + 1
    out_w = (W + 2 * pad - dil * (kw - 1) - 1) // stride + 1
    x = torch.randn(N, inC, H, W, device=device, dtype=dtype)
    w = torch.randn(outC, inC, kh, kw, device=device, dtype=dtype)
    b = torch.randn(outC, device=device, dtype=dtype)
    offset = torch.randn(N, 2 * dg * kh * kw, out_h, out_w, device=device, dtype=dtype)
    mask = None
    if use_mask:
        mask = torch.rand(N, dg * kh * kw, out_h, out_w, device=device, dtype=dtype)
    return x, w, b, offset, mask, (stride, pad, dil)


_CASES = [(*k, s, p, d, m)
          for k in KERNELS for s in STRIDES for p in PADS
          for d in DILATIONS for m in USE_MASK]


@requires_mps()
@pytest.mark.parametrize("kh,kw,stride,pad,dil,use_mask", _CASES)
def test_forward_matches_reference(kh, kw, stride, pad, dil, use_mask):
    x, w, b, offset, mask, (s, p, d) = _make_inputs(
        kh, kw, stride, pad, dil, use_mask, device="mps")

    out_mps = deform_conv2d(x, offset, w, bias=b,
                            stride=s, padding=p, dilation=d, mask=mask).cpu()

    ref = tv_deform_conv2d(
        x.cpu(), offset.cpu(), w.cpu(), bias=b.cpu(),
        stride=s, padding=p, dilation=d,
        mask=mask.cpu() if mask is not None else None)

    torch.testing.assert_close(out_mps, ref, rtol=1e-4, atol=1e-4)
