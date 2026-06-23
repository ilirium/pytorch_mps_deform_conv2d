"""Backward correctness for the native MPS kernels (Phase 3+).

Two layers of checking:
  1. Compare each analytic gradient (input, offset, mask, weight, bias) against
     the torchvision CPU reference grads for the same inputs.
  2. torch.autograd.gradcheck. MPS is fp32-only, so gradcheck runs on a CPU
     fp64 reference; the MPS grads are validated against CPU fp32 with relaxed
     tolerances in the comparison test.

These are skipped until the native backward is implemented (they xfail on the
scaffold's NotImplementedError).
"""

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.ops import deform_conv2d as tv_deform_conv2d  # noqa: E402

from deform_conv2d_mps import deform_conv2d  # noqa: E402

from conftest import requires_mps  # noqa: E402


def _inputs(device, dtype=torch.float32, use_mask=True, requires_grad=True):
    torch.manual_seed(0)
    N, inC, outC, H, W, k = 2, 4, 6, 7, 7, 3
    dg = 1
    out = H - k + 1
    mk = lambda *s: torch.randn(*s, device=device, dtype=dtype, requires_grad=requires_grad)
    x = mk(N, inC, H, W)
    w = mk(outC, inC, k, k)
    b = mk(outC)
    offset = mk(N, 2 * dg * k * k, out, out)
    mask = (torch.rand(N, dg * k * k, out, out, device=device, dtype=dtype)
            .requires_grad_(requires_grad)) if use_mask else None
    return x, w, b, offset, mask


@requires_mps()
@pytest.mark.parametrize("use_mask", [False, True])
def test_backward_matches_reference(use_mask):
    x, w, b, offset, mask = _inputs("mps", use_mask=use_mask)
    xc, wc, bc, oc, mc = (t.detach().cpu().requires_grad_(True) if t is not None else None
                          for t in (x, w, b, offset, mask))

    out = deform_conv2d(x, offset, w, bias=b, mask=mask)
    out.sum().backward()

    ref = tv_deform_conv2d(xc, oc, wc, bias=bc, mask=mc)
    ref.sum().backward()

    torch.testing.assert_close(x.grad.cpu(), xc.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(offset.grad.cpu(), oc.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(w.grad.cpu(), wc.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(b.grad.cpu(), bc.grad, rtol=2e-3, atol=2e-3)
    if use_mask:
        torch.testing.assert_close(mask.grad.cpu(), mc.grad, rtol=2e-3, atol=2e-3)


def test_gradcheck_cpu_reference():
    """gradcheck on CPU fp64 reference — guards the math the MPS port targets."""
    x, w, b, offset, mask = _inputs("cpu", dtype=torch.float64, use_mask=True)
    inputs = (x, offset, w, b)
    assert torch.autograd.gradcheck(
        lambda xx, oo, ww, bb: tv_deform_conv2d(xx, oo, ww, bias=bb, mask=mask),
        inputs, eps=1e-6, atol=1e-4)
