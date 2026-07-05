"""Diagnostic isolation ladder for the deformable_im2col Metal kernel (Phase 1, Step 5).

Run:  python tests/diag_im2col.py
Stops at the first failing stage; paste the full output back.

Trick: there is no Python binding that returns the raw column buffer, so we
call the native forward with an identity weight (eye(C*kh*kw) reshaped to
(C*kh*kw, C, kh, kw)). The GEMM then reproduces the columns verbatim:
output.view(C*kh*kw, out_h*out_w) == columns. This validates the kernel's
index math AND the col-layout contract with `weight.view({outC, -1}).mm(...)`
in one shot.

Ladder:
  1. zero offsets, mask=ones  -> columns must equal torch.nn.functional.unfold
  2. constant offset (+0.5)   -> compare vs Python reference bilinear gather
  3. random offsets + mask    -> same reference, plus full forward vs torchvision
"""

import os
import sys

# Conda envs often link two OpenMP runtimes (torch's libomp + MKL's libiomp5);
# importing torch then aborts with "OMP Error #15". The Makefile exports this,
# but set it here too so `python tests/diag_im2col.py` works standalone.
# Must happen BEFORE `import torch`.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    print("SKIP: MPS not available on this machine.")
    sys.exit(0)

from deform_conv2d_mps import ops  # noqa: E402

_ext = ops._load_native()  # compiles the Metal library, registers the ops
_native_forward = torch.ops.deform_conv2d_mps.deform_conv2d_forward

RTOL, ATOL = 1e-4, 1e-5


# ---------------------------------------------------------------------------
# Python reference: bilinear sample with torchvision's boundary behaviour
# (zero outside (-1, H) / (-1, W), corners clamped individually).
# ---------------------------------------------------------------------------
def _bilinear(plane, h, w):
    H, W = plane.shape
    if h <= -1 or h >= H or w <= -1 or w >= W:
        return 0.0
    import math
    h_low, w_low = math.floor(h), math.floor(w)
    h_high, w_high = h_low + 1, w_low + 1
    lh, lw = h - h_low, w - w_low
    hh, hw = 1.0 - lh, 1.0 - lw
    v1 = plane[h_low, w_low].item() if h_low >= 0 and w_low >= 0 else 0.0
    v2 = plane[h_low, w_high].item() if h_low >= 0 and w_high <= W - 1 else 0.0
    v3 = plane[h_high, w_low].item() if h_high <= H - 1 and w_low >= 0 else 0.0
    v4 = plane[h_high, w_high].item() if h_high <= H - 1 and w_high <= W - 1 else 0.0
    return hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4


def ref_im2col(inp, offset, mask, kh, kw, stride, pad, dil):
    """Reference deformable im2col for a single image, dg == 1.

    inp: (C, H, W); offset: (2*kh*kw, oh, ow); mask: (kh*kw, oh, ow) or None.
    Returns (C*kh*kw, oh*ow).
    """
    C, H, W = inp.shape
    sh, sw = stride
    ph, pw = pad
    dh, dw = dil
    oh = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = torch.zeros(C * kh * kw, oh * ow, dtype=inp.dtype)
    for oy in range(oh):
        for ox in range(ow):
            for ki in range(kh):
                for kj in range(kw):
                    k = ki * kw + kj
                    off_h = offset[2 * k, oy, ox].item()
                    off_w = offset[2 * k + 1, oy, ox].item()
                    h = oy * sh - ph + ki * dh + off_h
                    w = ox * sw - pw + kj * dw + off_w
                    m = mask[k, oy, ox].item() if mask is not None else 1.0
                    for c in range(C):
                        cols[c * kh * kw + k, oy * ow + ox] = \
                            _bilinear(inp[c], h, w) * m
    return cols


def native_columns(inp, offset, mask, kh, kw, stride, pad, dil, dg=1):
    """Run the native forward with an identity weight -> raw columns."""
    C = inp.shape[1]
    eye = torch.eye(C * kh * kw, dtype=inp.dtype)
    weight = eye.view(C * kh * kw, C, kh, kw).to("mps")
    mask_arg = mask.to("mps") if mask is not None \
        else torch.empty(0, device="mps")
    out = _native_forward(
        inp.to("mps"), weight, offset.to("mps"), mask_arg, None,
        stride[0], stride[1], pad[0], pad[1], dil[0], dil[1], 1, dg)
    n, ckk, oh, ow = out.shape
    assert n == 1
    return out.view(ckk, oh * ow).cpu()


def check(name, got, want):
    try:
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)
        print(f"PASS  {name}")
        return True
    except AssertionError as e:
        print(f"FAIL  {name}\n{e}")
        return False


def main():
    torch.manual_seed(0)
    ok = True

    # --- Stage 1: zero offsets, mask=ones -> must equal plain unfold ---------
    for (C, H, W, kh, kw, stride, pad, dil) in [
        (1, 3, 3, 2, 2, (1, 1), (0, 0), (1, 1)),        # plan's tiny case
        (2, 5, 5, 3, 3, (1, 1), (1, 1), (1, 1)),        # padding in play
        (3, 8, 7, 3, 2, (2, 1), (1, 0), (1, 2)),        # stride + dilation
    ]:
        oh = (H + 2 * pad[0] - dil[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (W + 2 * pad[1] - dil[1] * (kw - 1) - 1) // stride[1] + 1
        inp = torch.randn(1, C, H, W)
        offset = torch.zeros(1, 2 * kh * kw, oh, ow)
        mask = torch.ones(1, kh * kw, oh, ow)
        got = native_columns(inp, offset, mask, kh, kw, stride, pad, dil)
        want = F.unfold(inp, (kh, kw), dilation=dil, padding=pad,
                        stride=stride)[0]
        ok &= check(f"stage1 unfold parity C={C} {H}x{W} k={kh}x{kw} "
                    f"s={stride} p={pad} d={dil}", got, want)
    if not ok:
        print("\nStage 1 failed -> base index math is wrong; "
              "fix before looking at bilinear/offset logic.")
        sys.exit(1)

    # --- Stage 2: constant +0.5 offset, no mask (DCNv1 path) -----------------
    C, H, W, kh, kw = 1, 4, 4, 2, 2
    stride, pad, dil = (1, 1), (0, 0), (1, 1)
    oh = ow = 3
    inp = torch.randn(1, C, H, W)
    offset = torch.full((1, 2 * kh * kw, oh, ow), 0.5)
    got = native_columns(inp, offset, None, kh, kw, stride, pad, dil)
    want = ref_im2col(inp[0], offset[0], None, kh, kw, stride, pad, dil)
    ok &= check("stage2 constant offset +0.5, use_mask=0", got, want)
    if not ok:
        print("\nStage 2 failed -> bilinear/offset logic (or the DCNv1 "
              "mask-placeholder binding) is wrong.")
        sys.exit(1)

    # --- Stage 3: random offsets + mask ---------------------------------------
    C, H, W, kh, kw = 2, 6, 5, 3, 3
    stride, pad, dil = (1, 1), (1, 1), (1, 1)
    oh, ow = 6, 5
    inp = torch.randn(1, C, H, W)
    offset = torch.randn(1, 2 * kh * kw, oh, ow) * 2.0  # push some taps OOB
    mask = torch.rand(1, kh * kw, oh, ow)
    got = native_columns(inp, offset, mask, kh, kw, stride, pad, dil)
    want = ref_im2col(inp[0], offset[0], mask[0], kh, kw, stride, pad, dil)
    ok &= check("stage3 random offsets + mask", got, want)

    # --- Stage 3b: full pipeline (N=2, real weight + bias) vs torchvision ----
    import torchvision.ops as tvops
    N, C, H, W, outC, kh, kw = 2, 3, 8, 8, 4, 3, 3
    stride, pad, dil = (1, 1), (1, 1), (1, 1)
    oh = ow = 8
    inp = torch.randn(N, C, H, W)
    weight = torch.randn(outC, C, kh, kw)
    offset = torch.randn(N, 2 * kh * kw, oh, ow)
    mask = torch.rand(N, kh * kw, oh, ow)
    bias = torch.randn(outC)
    got = _native_forward(
        inp.to("mps"), weight.to("mps"), offset.to("mps"), mask.to("mps"),
        bias.to("mps"), stride[0], stride[1], pad[0], pad[1],
        dil[0], dil[1], 1, 1).cpu()
    want = tvops.deform_conv2d(inp, offset, weight, bias=bias, stride=stride,
                               padding=pad, dilation=dil, mask=mask)
    ok &= check("stage3b full forward (N=2, bias) vs torchvision CPU",
                got, want)

    if ok:
        print("\nAll stages passed. Next: DCN_MPS_FORCE_NATIVE=1 "
              "pytest tests/test_forward.py (Phase 2).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
