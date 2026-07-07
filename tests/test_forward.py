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


# ---------------------------------------------------------------------------
# Hand-picked extra cases (Phase 2 Step 4): asymmetric/non-square params,
# bias=None, N=1 + odd channels, out-of-bounds offsets, larger spatial size.
# Defaults mirror the matrix above; each case overrides what it targets.
# ---------------------------------------------------------------------------

def _pair(v):
    return (v, v) if isinstance(v, int) else v


def _run_case(N=2, inC=4, outC=6, H=9, W=9, kh=3, kw=3,
              stride=1, pad=1, dil=1, use_mask=True, use_bias=True,
              dg=1, groups=1, offset_scale=1.0):
    torch.manual_seed(0)
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)
    dh, dw = _pair(dil)
    out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    device, dtype = "mps", torch.float32
    x = torch.randn(N, inC, H, W, device=device, dtype=dtype)
    # groups is inferred from weight.size(1), as torchvision does.
    w = torch.randn(outC, inC // groups, kh, kw, device=device, dtype=dtype)
    b = torch.randn(outC, device=device, dtype=dtype) if use_bias else None
    offset = offset_scale * torch.randn(
        N, 2 * dg * kh * kw, out_h, out_w, device=device, dtype=dtype)
    mask = None
    if use_mask:
        mask = torch.rand(N, dg * kh * kw, out_h, out_w,
                          device=device, dtype=dtype)

    out_mps = deform_conv2d(x, offset, w, bias=b,
                            stride=(sh, sw), padding=(ph, pw),
                            dilation=(dh, dw), mask=mask).cpu()
    ref = tv_deform_conv2d(
        x.cpu(), offset.cpu(), w.cpu(),
        bias=b.cpu() if b is not None else None,
        stride=(sh, sw), padding=(ph, pw), dilation=(dh, dw),
        mask=mask.cpu() if mask is not None else None)
    torch.testing.assert_close(out_mps, ref, rtol=1e-4, atol=1e-4)


_EXTRA_CASES = [
    # Non-square / asymmetric — catches h/w swaps the square matrix hides.
    pytest.param(dict(H=8, W=11), id="rect-input"),
    pytest.param(dict(H=8, W=11, stride=(2, 1)), id="asym-stride"),
    pytest.param(dict(H=8, W=11, pad=(0, 2)), id="asym-pad"),
    pytest.param(dict(H=8, W=11, dil=(2, 1)), id="asym-dilation"),
    pytest.param(dict(H=8, W=11, kh=1, kw=3, pad=(0, 1)), id="rect-kernel-1x3"),
    pytest.param(dict(H=8, W=11, kh=1, kw=3, stride=(2, 1), pad=(0, 2),
                      dil=(2, 1)), id="asym-everything"),
    # bias=None — untested through the native path until now.
    pytest.param(dict(use_bias=False), id="no-bias"),
    pytest.param(dict(use_bias=False, use_mask=False), id="no-bias-v1"),
    # Batch/channel edge cases.
    pytest.param(dict(N=1, inC=3, outC=5), id="n1-odd-channels"),
    # Boundary stress: many samples land outside the input (zero region,
    # incl. the h_im > -1 partial-tap edge).
    pytest.param(dict(offset_scale=8.0), id="boundary-stress"),
    pytest.param(dict(offset_scale=8.0, use_mask=False), id="boundary-stress-v1"),
    # Larger spatial smoke case — grid-size/threadgroup edge effects.
    pytest.param(dict(N=2, inC=8, H=33, W=35, stride=2, pad=1),
                 id="large-spatial"),
    # Phase 5 Step 1: deformable_groups > 1. The kernels have indexed by
    # deformable_group_index since Phase 1, but dg > 1 first runs on-device
    # here. Trap: a wrong dg index reads valid memory from the *wrong* group
    # — plausible values, not NaN — so cover cpg=2, cpg=3 (non-power-of-two
    # split), and dg=C (one channel per group), mask on and off, plus an
    # asym stride/pad/dilation combo.
    pytest.param(dict(dg=2), id="dg2"),
    pytest.param(dict(dg=2, use_mask=False), id="dg2-v1"),
    pytest.param(dict(dg=2, inC=6), id="dg2-cpg3"),
    pytest.param(dict(dg=4), id="dg-eq-C"),
    pytest.param(dict(dg=2, H=8, W=11, stride=(2, 1), pad=(0, 2), dil=(2, 1)),
                 id="dg2-asym"),
    # Phase 5 Step 2: groups > 1. The im2col kernel is group-agnostic (fills
    # all C channels); only the host GEMM is grouped — a wrong slice in the
    # (groups, ...) bmm views mixes channels across groups. groups=2 with
    # C=4 and C=6, mask on/off, asym combo, the groups x dg cross term
    # (independent parameters — cover the interaction), and groups=outC
    # (depthwise-flavoured, weight.size(1) == 1).
    pytest.param(dict(groups=2), id="g2"),
    pytest.param(dict(groups=2, use_mask=False), id="g2-v1"),
    pytest.param(dict(groups=2, inC=6), id="g2-c6"),
    pytest.param(dict(groups=2, H=8, W=11, stride=(2, 1), pad=(0, 2),
                      dil=(2, 1)), id="g2-asym"),
    pytest.param(dict(groups=2, dg=2), id="g2-dg2"),
    pytest.param(dict(inC=6, outC=6, groups=6), id="g-eq-outC-depthwise"),
]


@requires_mps()
@pytest.mark.parametrize("case", _EXTRA_CASES)
def test_forward_extra_cases(case):
    _run_case(**case)
