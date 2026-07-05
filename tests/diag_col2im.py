"""Diagnostic isolation ladder for the Phase 3 backward Metal kernels.

Run:  python tests/diag_col2im.py   (or `make diag-backward`)
Stops at the first failing stage; paste the full output back.

Ladder (grows as Phase 3 steps land):
  0. atomic smoke            -> MSL 3.0 atomic_float scatter-add compiles+runs [Step 1]
  1. fold parity             -> deformable_col2im == F.fold (zero offsets)    [Step 2]
  2. numerical grad_input    -> vs CPU autograd through torchvision           [Step 2]
  3. numerical grad_offset/grad_mask -> vs CPU reference grads                [Step 3]
  4. full backward smoke     -> all five grads vs torchvision CPU             [Steps 4-5]
"""

import os
import sys

# Conda envs often link two OpenMP runtimes (torch's libomp + MKL's libiomp5);
# importing torch then aborts with "OMP Error #15". The Makefile exports this,
# but set it here too so `python tests/diag_col2im.py` works standalone.
# Must happen BEFORE `import torch`.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

if not torch.backends.mps.is_available():
    print("SKIP: MPS not available on this machine.")
    sys.exit(0)

from deform_conv2d_mps import ops  # noqa: E402

_ext = ops._load_native()  # compiles the Metal library, registers the ops


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

    # Stages 1-4 land with Phase 3 Steps 2-5.

    if ok:
        print("\nAll implemented stages passed.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
