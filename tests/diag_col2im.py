"""Diagnostic isolation ladder for the Phase 3 backward Metal kernels.

Run:  python tests/diag_col2im.py   (or `make diag-backward`)
Stops at the first failing stage; paste the full output back.

Ladder (grows as Phase 3 steps land):
  0. atomic smoke            -> MSL 3.0 atomic_float scatter-add compiles+runs [Step 1]
  1. fold parity             -> deformable_col2im == F.fold (zero offsets)    [Step 2]
  2. numerical grad_input    -> vs CPU autograd through torchvision           [Step 2]
  3. numerical grad_offset/grad_mask -> vs CPU reference grads                [Step 3]
  4. full backward smoke     -> all five grads vs torchvision CPU             [Steps 4-5]

Tolerances: stages touching the atomic scatter (grad_input) use rtol=2e-3 —
float atomic add order varies run to run, so diffs vs CPU are noise-scaled.
A diff that does NOT vary across runs and localises to specific pixels is an
index bug, not atomics.
"""

import os
import sys

# Conda envs often link two OpenMP runtimes (torch's libomp + MKL's libiomp5);
# importing torch then aborts with "OMP Error #15". The Makefile exports this,
# but set it here too so `python tests/diag_col2im.py` works standalone.
# Must happen BEFORE `import torch`.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

if not torch.backends.mps.is_available():
    print("SKIP: MPS not available on this machine.")
    sys.exit(0)

from deform_conv2d_mps import ops  # noqa: E402

_ext = ops._load_native()  # compiles the Metal library, registers the ops
_col2im = torch.ops.deform_conv2d_mps.deformable_col2im


def native_col2im(cols, offset, mask, H, W, kh, kw, stride, pad, dil, dg=1):
    """Single-image col2im on MPS; cols (C*kh*kw, oh*ow) -> grad_input (C, H, W)."""
    mask_arg = mask.to("mps") if mask is not None \
        else torch.empty(0, device="mps")
    return _col2im(cols.to("mps"), offset.to("mps"), mask_arg,
                   H, W, kh, kw, stride[0], stride[1], pad[0], pad[1],
                   dil[0], dil[1], dg).cpu()


def check(name, got, want, rtol=1e-4, atol=1e-5):
    try:
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
        print(f"PASS  {name}")
        return True
    except AssertionError as e:
        print(f"FAIL  {name}\n{e}")
        return False


def main():
    torch.manual_seed(0)
    ok = True

    # --- Stage 0: atomic scatter-add smoke ------------------------------------
    # in[i] = (i % 7) + 1, scattered into out[i % 4]. Integer-valued fp32 sums
    # are order-independent (exact), so a mismatch is a compile/dispatch bug,
    # not atomics noise. Slot sums stay far below 2^24 (fp32-exact).
    n = 100_000
    x = (torch.arange(n) % 7 + 1).float()
    got = torch.ops.deform_conv2d_mps.atomic_smoke(x.to("mps")).cpu()
    want = torch.zeros(4).index_add_(0, torch.arange(n) % 4, x)
    ok &= check(f"stage0 atomic_smoke n={n}", got, want, rtol=0, atol=0)
    if not ok:
        print("\nStage 0 failed -> MSL 3.0 atomic_float is broken; nothing "
              "downstream can work. Check MTLLanguageVersion3_0 in the .mm.")
        sys.exit(1)

    # --- Stage 1: fold parity --------------------------------------------------
    # Zero offsets, no mask: every tap lands on an integer grid point, so the
    # deformable scatter degenerates to exactly torch.nn.functional.fold.
    # Catches index/decomposition/atomic-binding bugs without bilinear math.
    for (C, H, W, kh, kw, stride, pad, dil) in [
        (1, 3, 3, 2, 2, (1, 1), (0, 0), (1, 1)),        # tiny
        (2, 5, 5, 3, 3, (1, 1), (1, 1), (1, 1)),        # padding in play
        (3, 8, 7, 3, 2, (2, 1), (1, 0), (1, 2)),        # stride + dilation
    ]:
        oh = (H + 2 * pad[0] - dil[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (W + 2 * pad[1] - dil[1] * (kw - 1) - 1) // stride[1] + 1
        cols = torch.randn(C * kh * kw, oh * ow)
        offset = torch.zeros(2 * kh * kw, oh, ow)
        got = native_col2im(cols, offset, None, H, W, kh, kw, stride, pad, dil)
        want = F.fold(cols.unsqueeze(0), (H, W), (kh, kw),
                      dilation=dil, padding=pad, stride=stride)[0]
        ok &= check(f"stage1 fold parity C={C} {H}x{W} k={kh}x{kw} "
                    f"s={stride} p={pad} d={dil}", got, want)
    if not ok:
        print("\nStage 1 failed -> col2im index math / scatter target is "
              "wrong; fix before looking at bilinear weights.")
        sys.exit(1)

    # --- Stage 2: numerical grad_input vs CPU autograd ------------------------
    # Tiny case, constant non-integer offset: hand-traceable if an element is
    # off. grad_columns = W^T @ grad_out reproduces what backward (Step 4)
    # will feed the kernel.
    import torchvision.ops as tvops

    def grad_input_case(name, C, outC, H, W, kh, kw, stride, pad, dil,
                        offset, mask):
        oh = (H + 2 * pad[0] - dil[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (W + 2 * pad[1] - dil[1] * (kw - 1) - 1) // stride[1] + 1
        inp = torch.randn(1, C, H, W, requires_grad=True)
        weight = torch.randn(outC, C, kh, kw)
        gout = torch.randn(1, outC, oh, ow)
        out = tvops.deform_conv2d(inp, offset, weight, stride=stride,
                                  padding=pad, dilation=dil, mask=mask)
        out.backward(gout)
        want = inp.grad[0]

        grad_cols = weight.view(outC, -1).t().mm(gout[0].view(outC, oh * ow))
        got = native_col2im(grad_cols, offset[0],
                            mask[0] if mask is not None else None,
                            H, W, kh, kw, stride, pad, dil)
        return check(name, got, want, rtol=2e-3, atol=1e-4)

    C, outC, H, W, kh, kw = 1, 1, 5, 5, 2, 2
    stride, pad, dil = (1, 1), (0, 0), (1, 1)
    oh = ow = 4
    offset = torch.full((1, 2 * kh * kw, oh, ow), 0.3)
    offset[:, 1::2] = -0.6  # non-integer shift on both axes
    ok &= grad_input_case("stage2 grad_input const offset (0.3, -0.6), no mask",
                          C, outC, H, W, kh, kw, stride, pad, dil, offset, None)

    # 2b: random offsets (some OOB) + mask -> exercises the mask multiply and
    # the boundary sentinels in the scatter path.
    C, outC, H, W, kh, kw = 2, 3, 6, 5, 3, 3
    stride, pad, dil = (1, 1), (1, 1), (1, 1)
    oh, ow = 6, 5
    offset = torch.randn(1, 2 * kh * kw, oh, ow) * 2.0
    mask = torch.rand(1, kh * kw, oh, ow)
    ok &= grad_input_case("stage2b grad_input random offsets + mask",
                          C, outC, H, W, kh, kw, stride, pad, dil, offset, mask)
    if not ok:
        print("\nStage 2 failed with stage 1 green -> bilinear scatter weights "
              "(get_gradient_weight) or the mask/offset indexing are wrong.")
        sys.exit(1)

    # Stages 3-4 land with Phase 3 Steps 3-5.

    if ok:
        print("\nAll implemented stages passed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
