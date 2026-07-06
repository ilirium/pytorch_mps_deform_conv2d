"""Backward correctness for the native MPS kernels (Phase 3+).

Two layers of checking:
  1. Compare each analytic gradient (input, offset, mask, weight, bias)
     against the torchvision CPU reference grads for identical inputs,
     across a kernel/stride/pad/dilation/mask matrix plus hand-picked trap
     cases, with per-grad tolerances.
  2. torch.autograd.gradcheck, in two layers: fp64 on the CPU reference
     (guards the math the port targets), and fp32 through the native MPS
     path with fp32-appropriate knobs, calibrated on a CPU fp32 run of the
     same case so a knob problem shows up on the reference, not as a bogus
     native failure.

The native backward (Phase 3) is gated behind _BACKWARD_READY: run with
DCN_MPS_FORCE_NATIVE=1 (`make test-backward-native`) to exercise it. Without
the flag, grad-requiring calls fall back to torchvision until the flag flips
(Phase 4), so the comparison is reference-vs-itself and passes trivially.
"""

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.ops import deform_conv2d as tv_deform_conv2d  # noqa: E402

from deform_conv2d_mps import deform_conv2d  # noqa: E402

from conftest import requires_mps  # noqa: E402


def _pair(v):
    return (v, v) if isinstance(v, int) else v


# Per-grad tolerances. grad_input goes through an atomic float scatter whose
# summation order varies run to run -> 2e-3. grad_offset / grad_mask are
# deterministic (no atomics) and grad_weight / grad_bias come from ATen
# GEMM / sum -> 1e-4, matching the forward. Loosen only with a measured
# reason: a grad_input diff that is *stable across runs* and localised is an
# index bug, not atomic noise.
_TOL = {
    "input":  dict(rtol=2e-3, atol=2e-3),
    "offset": dict(rtol=1e-4, atol=1e-4),
    "mask":   dict(rtol=1e-4, atol=1e-4),
    "weight": dict(rtol=1e-4, atol=1e-4),
    "bias":   dict(rtol=1e-4, atol=1e-4),
}


def _run_backward_case(N=2, inC=4, outC=6, H=9, W=9, kh=3, kw=3,
                       stride=1, pad=1, dil=1, use_mask=True, use_bias=True,
                       dg=1, offset_scale=1.0, upstream="sum"):
    """Run one case on MPS and on the torchvision CPU reference with
    identical inputs; compare every gradient with per-grad tolerances.

    With use_bias=False, merely running is part of the check: the autograd
    Function must return None (not zeros) for the bias slot, or autograd
    raises on backward.
    """
    torch.manual_seed(0)
    sh, sw = _pair(stride)
    ph, pw = _pair(pad)
    dh, dw = _pair(dil)
    out_h = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    # CPU leaves first, then identical MPS copies (independent of MPS RNG).
    xc = torch.randn(N, inC, H, W, requires_grad=True)
    wc = torch.randn(outC, inC, kh, kw, requires_grad=True)
    bc = torch.randn(outC, requires_grad=True) if use_bias else None
    oc = (offset_scale * torch.randn(N, 2 * dg * kh * kw, out_h, out_w)
          ).requires_grad_(True)
    mc = (torch.rand(N, dg * kh * kw, out_h, out_w).requires_grad_(True)
          if use_mask else None)

    def to_mps(t):
        return (t.detach().to("mps").requires_grad_(True)
                if t is not None else None)

    x, w, b, offset, mask = map(to_mps, (xc, wc, bc, oc, mc))

    out = deform_conv2d(x, offset, w, bias=b, stride=(sh, sw),
                        padding=(ph, pw), dilation=(dh, dw), mask=mask)
    ref = tv_deform_conv2d(xc, oc, wc, bias=bc, stride=(sh, sw),
                           padding=(ph, pw), dilation=(dh, dw), mask=mc)

    if upstream == "sum":
        out.sum().backward()
        ref.sum().backward()
    else:
        # Non-scalar upstream grad: sum() feeds a constant grad_output,
        # which can hide transposition/scaling bugs in the GEMM wiring.
        g = torch.randn_like(ref.detach())
        out.backward(g.to("mps"))
        ref.backward(g)

    pairs = [("input", x, xc), ("offset", offset, oc), ("weight", w, wc)]
    if use_bias:
        pairs.append(("bias", b, bc))
    if use_mask:
        pairs.append(("mask", mask, mc))
    for name, dev_t, cpu_t in pairs:
        torch.testing.assert_close(
            dev_t.grad.cpu(), cpu_t.grad, **_TOL[name],
            msg=lambda m, name=name: f"grad_{name}: {m}")


# ---------------------------------------------------------------------------
# Parametrised matrix — mirrors the Phase 2 forward matrix, but for grads.
# Tensors stay tiny (N=2, C<=6, H=W=9): backward is ~3x forward cost and
# this runs 48x.
# ---------------------------------------------------------------------------

KERNELS = [(3, 3), (1, 1), (1, 3)]
STRIDES = [1, 2]
PADS = [0, 1]
DILATIONS = [1, 2]
USE_MASK = [False, True]

_CASES = [(*k, s, p, d, m)
          for k in KERNELS for s in STRIDES for p in PADS
          for d in DILATIONS for m in USE_MASK]


@requires_mps()
@pytest.mark.parametrize("kh,kw,stride,pad,dil,use_mask", _CASES)
def test_backward_matches_reference(kh, kw, stride, pad, dil, use_mask):
    _run_backward_case(kh=kh, kw=kw, stride=stride, pad=pad, dil=dil,
                       use_mask=use_mask)


# ---------------------------------------------------------------------------
# Non-scalar upstream grad — out.backward(randn_like(out)).
# ---------------------------------------------------------------------------

@requires_mps()
@pytest.mark.parametrize("use_mask", [False, True])
def test_backward_nonscalar_upstream_grad(use_mask):
    _run_backward_case(use_mask=use_mask, upstream="randn")


# ---------------------------------------------------------------------------
# Hand-picked extras, ported from the forward's trap list. The h/w-swap
# traps now live in *three* kernels (im2col, col2im, col2im_coord).
# ---------------------------------------------------------------------------

_EXTRA_CASES = [
    # Non-square input + asymmetric stride/pad/dilation + non-square kernel.
    pytest.param(dict(H=8, W=11, kh=1, kw=3, stride=(2, 1), pad=(0, 2),
                      dil=(2, 1)), id="asym-everything"),
    # bias=None — grad_bias must be None, not zeros (see helper docstring).
    pytest.param(dict(use_bias=False), id="no-bias"),
    pytest.param(dict(use_bias=False, use_mask=False), id="no-bias-v1"),
    # Batch/channel edge cases.
    pytest.param(dict(N=1, inC=3, outC=5), id="n1-odd-channels"),
    # Offsets x8: OOB region — exercises the col2im +-2 window guard and
    # the coord kernel's -2 sentinel.
    pytest.param(dict(offset_scale=8.0), id="boundary-stress"),
    pytest.param(dict(offset_scale=8.0, use_mask=False),
                 id="boundary-stress-v1"),
    # Larger spatial smoke case — grid-size/threadgroup edge effects.
    pytest.param(dict(N=2, inC=8, H=33, W=35, stride=2, pad=1),
                 id="large-spatial"),
    # Visible gap until Phase 5.
    pytest.param(dict(dg=2),
                 marks=pytest.mark.skip(reason="deformable_groups>1 is Phase 5"),
                 id="dg2"),
]


@requires_mps()
@pytest.mark.parametrize("case", _EXTRA_CASES)
def test_backward_extra_cases(case):
    _run_backward_case(**case)


# ---------------------------------------------------------------------------
# Gradcheck
# ---------------------------------------------------------------------------

# fp32 gradcheck knobs (plan Step 3):
# - eps 1e-3: fp32 central differences; the default 1e-6 drowns in rounding.
# - atol/rtol 1e-2: relaxed for fp32 FD noise; the analytic-comparison tests
#   above are the load-bearing check, this is belt-and-suspenders.
# - nondet_tol 1e-3: grad_input's atomic scatter makes repeated backward
#   calls differ; without this, gradcheck's determinism check fails
#   spuriously ("backward is not deterministic" that looks like a bug).
_FP32_GRADCHECK_KW = dict(eps=1e-3, atol=1e-2, rtol=1e-2, nondet_tol=1e-3)

# gradcheck warns that fp32 inputs "will likely fail" — running fp32 is the
# point on MPS (fp64 is unsupported), and the knobs above account for it.
_expected_fp32_warning = pytest.mark.filterwarnings(
    "ignore:Input #.*double precision:UserWarning")


def _gradcheck_inputs(device, use_mask, dtype=torch.float32):
    """Tiny case — gradcheck is O(numel) backward calls.

    Offsets are kept >= 0.1 away from integer grid points (Phase 3 trick):
    bilinear kinks are genuine non-differentiability, not bugs, and eps=1e-3
    perturbations never cross the 0.1 buffer.
    """
    torch.manual_seed(0)
    N, inC, outC, k, H, W = 1, 2, 2, 2, 5, 5
    dg = 1
    out = H - k + 1  # stride 1, pad 0
    mk = lambda *s: torch.randn(*s, device=device, dtype=dtype,
                                requires_grad=True)
    x = mk(N, inC, H, W)
    w = mk(outC, inC, k, k)
    b = mk(outC)
    o = torch.randn(N, 2 * dg * k * k, out, out,
                    device=device, dtype=dtype) * 0.5
    o = (o.floor() + (o - o.floor()).clamp(0.1, 0.9)).requires_grad_(True)
    m = None
    if use_mask:
        m = (torch.rand(N, dg * k * k, out, out, device=device, dtype=dtype)
             * 0.8 + 0.1).requires_grad_(True)
    return x, o, w, b, m


def _run_fp32_gradcheck(op, device, use_mask):
    x, o, w, b, m = _gradcheck_inputs(device, use_mask)
    if use_mask:
        fn = lambda xx, oo, ww, bb, mm: op(xx, oo, ww, bias=bb, mask=mm)
        inputs = (x, o, w, b, m)
    else:
        fn = lambda xx, oo, ww, bb: op(xx, oo, ww, bias=bb)
        inputs = (x, o, w, b)
    assert torch.autograd.gradcheck(fn, inputs, **_FP32_GRADCHECK_KW)


@_expected_fp32_warning
@pytest.mark.parametrize("use_mask", [False, True])
def test_gradcheck_cpu_fp32_calibration(use_mask):
    """Same case/knobs as the native fp32 gradcheck, on torchvision CPU fp32.

    Calibration guard: if the knobs are too tight for fp32 at all, this
    fails too — fix the knobs here before suspecting the native kernels.
    """
    _run_fp32_gradcheck(tv_deform_conv2d, "cpu", use_mask)


@_expected_fp32_warning
@requires_mps()
@pytest.mark.parametrize("use_mask", [False, True])
def test_gradcheck_native_fp32(use_mask):
    """fp32 gradcheck through the native path (DCNv1 no-mask / DCNv2 mask).

    Checks input, offset, weight, bias (+ mask) via central differences.
    Run with DCN_MPS_FORCE_NATIVE=1; unforced it exercises the fallback.
    """
    _run_fp32_gradcheck(deform_conv2d, "mps", use_mask)


def test_gradcheck_cpu_reference():
    """gradcheck on CPU fp64 reference — guards the math the MPS port targets."""
    torch.manual_seed(0)
    N, inC, outC, H, W, k = 2, 4, 6, 7, 7, 3
    dg = 1
    out = H - k + 1
    mk = lambda *s: torch.randn(*s, dtype=torch.float64, requires_grad=True)
    x = mk(N, inC, H, W)
    w = mk(outC, inC, k, k)
    b = mk(outC)
    offset = mk(N, 2 * dg * k * k, out, out)
    mask = torch.rand(N, dg * k * k, out, out,
                      dtype=torch.float64).requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda xx, oo, ww, bb: tv_deform_conv2d(xx, oo, ww, bias=bb, mask=mask),
        (x, offset, w, b), eps=1e-6, atol=1e-4)
