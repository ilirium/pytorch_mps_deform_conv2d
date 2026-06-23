"""
Gradient checks for torchvision.ops.deform_conv2d (CPU and MPS).

Why this exists
---------------
A deformable convolution back-propagates into FOUR inputs:
    input, weight, offset, mask
The offset/mask gradients are the subtle ones: bilinear sampling makes them
nonlinear, so a forward pass can look correct while the backward is wrong.
That only surfaces later as a model that silently won't train. When validating
a custom MPS kernel, the backward pass is the part most worth testing.

Two independent checks
----------------------
  1. CPU-vs-MPS backward agreement
       Run .backward() on both devices with identical inputs and compare the
       grads of every differentiable input. Catches backend-specific bugs.

  2. torch.autograd.gradcheck (CPU, float64)
       Compares the analytic backward against numerical finite differences.
       Confirms the math is correct, independent of any backend. Needs double
       precision, so it runs on CPU only (MPS has no float64).

Run:
    python example02_deform_conv2d_gradcheck.py
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torchvision.ops import deform_conv2d


# ---------------------------------------------------------------------------
# Geometry (kept small so gradcheck's finite differences stay cheap)
# ---------------------------------------------------------------------------
N = 1
IN_CH = 2
OUT_CH = 3
H, W = 5, 5
KH, KW = 3, 3
STRIDE = (1, 1)
PADDING = (1, 1)
DILATION = (1, 1)
OFFSET_GROUPS = 1


def _out_size(size, k, stride, pad, dil):
    return (size + 2 * pad - dil * (k - 1) - 1) // stride + 1


OUT_H = _out_size(H, KH, STRIDE[0], PADDING[0], DILATION[0])
OUT_W = _out_size(W, KW, STRIDE[1], PADDING[1], DILATION[1])


def make_inputs(device, dtype, requires_grad):
    """Create the five tensors deform_conv2d differentiates through.

    input/weight/bias are always leaf tensors with grads; offset and mask are
    differentiable too (that is the whole point of the check).
    """
    g = torch.Generator(device="cpu").manual_seed(0)

    def rnd(*shape, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=dtype) * scale
        return t.to(device).requires_grad_(requires_grad)

    x = rnd(N, IN_CH, H, W)
    weight = rnd(OUT_CH, IN_CH, KH, KW)
    bias = rnd(OUT_CH)
    offset = rnd(N, 2 * OFFSET_GROUPS * KH * KW, OUT_H, OUT_W, scale=0.5)
    mask = rnd(N, OFFSET_GROUPS * KH * KW, OUT_H, OUT_W, scale=0.5)
    return x, weight, bias, offset, mask


def forward(x, weight, bias, offset, mask):
    return deform_conv2d(
        x, offset, weight, bias=bias,
        stride=STRIDE, padding=PADDING, dilation=DILATION, mask=mask,
    )


# ---------------------------------------------------------------------------
# Check 1: CPU vs MPS backward agreement
# ---------------------------------------------------------------------------
def check_cpu_vs_mps():
    print("=== Check 1: CPU vs MPS backward agreement ===")
    if not torch.backends.mps.is_available():
        print("  MPS not available - skipping.\n")
        return

    names = ("input", "weight", "bias", "offset", "mask")

    def run(device):
        tensors = make_inputs(torch.device(device), torch.float32, requires_grad=True)
        out = forward(*tensors)
        out.sum().backward()                       # scalar loss
        return out.detach().cpu(), [t.grad.detach().cpu() for t in tensors]

    try:
        out_cpu, grads_cpu = run("cpu")
        out_mps, grads_mps = run("mps")
    except NotImplementedError as e:
        print(f"  NotImplementedError: {e}\n")
        return

    fwd_diff = (out_cpu - out_mps).abs().max().item()
    print(f"  forward   max|CPU-MPS| = {fwd_diff:.3e}")
    all_ok = fwd_diff < 1e-4
    for name, gc, gm in zip(names, grads_cpu, grads_mps):
        d = (gc - gm).abs().max().item()
        ok = d < 1e-4
        all_ok &= ok
        print(f"  grad {name:<6} max|CPU-MPS| = {d:.3e}  {'ok' if ok else 'MISMATCH'}")
    print(f"  => {'PASS' if all_ok else 'FAIL'}\n")


# ---------------------------------------------------------------------------
# Check 2: numerical gradcheck (CPU, float64)
# ---------------------------------------------------------------------------
def check_gradcheck():
    print("=== Check 2: autograd.gradcheck (CPU, float64) ===")
    x, weight, bias, offset, mask = make_inputs(
        torch.device("cpu"), torch.float64, requires_grad=True
    )

    # gradcheck calls the op many times with perturbed inputs; the tuple order
    # here must match `forward`'s signature.
    ok = torch.autograd.gradcheck(
        forward,
        (x, weight, bias, offset, mask),
        eps=1e-6, atol=1e-4, rtol=1e-3,
        raise_exception=False,
    )
    print(f"  => {'PASS' if ok else 'FAIL'}\n")


def main():
    print(f"torch {torch.__version__}")
    import torchvision
    print(f"torchvision {torchvision.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"output spatial size: {OUT_H}x{OUT_W}\n")

    check_cpu_vs_mps()
    check_gradcheck()


if __name__ == "__main__":
    main()
