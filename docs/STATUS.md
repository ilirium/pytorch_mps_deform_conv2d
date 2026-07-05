# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-05

**Overall:** Phases 0–2 done — the native forward is verified against the torchvision CPU reference (2026-07-05, `Implementing_Phase2_004_good.txt`) and **enabled for inference on MPS** (`_FORWARD_READY = True`). Training still falls back to torchvision (via `PYTORCH_ENABLE_MPS_FALLBACK` CPU round-trip) until the native backward lands. Next: Phase 3 (backward kernels).

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | ✅ Done (all 6 `make diag` stages pass on-device) |
| 2 | Forward correctness tests vs torchvision | ✅ Done (60 native cases pass; forward enabled for inference) |
| 3 | Native backward (`col2im`, `col2im_coord`) | ⬜ Not started (stubs) |
| 4 | Backward tests + gradcheck | ⬜ Not started |
| 5 | Packaging, perf, groups / half precision | ⬜ Not started |

## What works today

- Build system (`setup.py` + Makefile), packaging, repo layout — complete.
- Metal pipeline validated end to end: `add_one` kernel compiles, dispatches through `torch::mps`, registers under the MPS dispatch key.
- Python API (`deform_conv2d`, `DeformConv2d`) complete and torchvision-compatible.
- **Native forward live for inference on MPS**: `_FORWARD_READY = True` with autograd-safe routing — native only when no input requires grad (or grad mode is off); grad-requiring calls use the torchvision fallback until Phase 3/4. `DCN_MPS_FORCE_NATIVE=1` bypasses the gating for testing.
- **Forward verified vs torchvision CPU reference** (`make test-forward-native`, rtol/atol=1e-4): 48-case matrix (3 kernels × stride × pad × dilation × mask) + hand-picked extras — non-square input with asymmetric stride/pad/dilation and 1×3 kernel, bias=None, N=1 with odd channels, offsets ×8 (out-of-bounds bilinear), 2×8×33×35 s=2 smoke case. No kernel bugs found; the Phase 1 port was correct as-is.
- Full `make diag` isolation ladder still passes.

## What's missing

- `deformable_col2im` and `deformable_col2im_coord` kernels are empty stubs.
- `_DeformConv2dFunction.backward()` raises `NotImplementedError` (`_BACKWARD_READY = False`; training falls back).
- `groups > 1` and `deformable_groups > 1` untested (dg=2 kept visible as a skipped test until Phase 5).

## Phase 1 implementation notes (2026-07-05)

- `deform_conv2d_forward` in `.mm`: validation → column buffer → per-batch `deformable_im2col` dispatch → interleaved ATen `mm` → bias. PSO cache added (`pipeline_for` now memoizes; cleared on library recompile).
- Build verified on-device (2026-07-05): `make build` succeeds. The 34 `-Wc++20-extensions` warnings (see `Implementing_Phase1_001_bug.txt`) were from torch ≥ 2.14 headers under `-std=c++17`; setup.py now uses `-std=c++20`. The duplicate `-rpath` ld warning (conda LDFLAGS + torch BuildExtension) is harmless.
- Running `python tests/diag_im2col.py` directly aborted with OMP Error #15 (see `Implementing_Phase1_002_good.txt`): the `KMP_DUPLICATE_LIB_OK` workaround only lived in the Makefile. The diag scripts and `tests/conftest.py` now set it themselves before `import torch`; `make diag` added as the canonical entry point.
- Threading: the ATen `mm` runs *outside* `dispatch_sync` (ATen MPS ops sync on the same serial queue → deadlock otherwise); each iteration encodes+commits im2col like `add_one`, and the command buffer is re-fetched per iteration.
- `tests/diag_im2col.py` (Step 5 ladder): identity-weight trick recovers the raw column buffer through the forward; stages = unfold parity → constant offset vs Python bilinear reference → random offset+mask → full pipeline vs torchvision CPU.

## Phase 2 implementation notes (2026-07-05)

- Added `make test-forward-native` (Step 1); the plain `test-forward` target compares torchvision to itself and passes trivially — always check the log header shows `DCN_MPS_FORCE_NATIVE=1`.
- First native run of the existing 48-case matrix passed clean (`Implementing_Phase2_001_good.txt`); widened coverage (Step 4) also passed with no kernel changes (`_002_good.txt`).
- `_NATIVE_READY` split into `_FORWARD_READY` / `_BACKWARD_READY`; routing checks `torch.is_grad_enabled()` + per-tensor `requires_grad` so training never hits the unimplemented backward.
- Gotcha found in Step 6 regression (`_003_bug.txt`): `torchvision::_deform_conv2d_backward` has no MPS kernel, so the fallback backward on MPS tensors needs `PYTORCH_ENABLE_MPS_FALLBACK=1`. Now set in `tests/conftest.py` before `import torch` (previously only the example scripts set it — `make test` had silently depended on the shell env). Final green: `_004_good.txt`.

## Next actions

1. Port the two backward kernels (`deformable_col2im`, `deformable_col2im_coord`); requires `atomic<float>` → set `MTLLanguageVersion3_0` in compile options. (Phase 3)
2. Backward tests + gradcheck; flip `_BACKWARD_READY = True` when green. (Phase 4)

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile + conftest (OMP Error #15 workaround); `PYTORCH_ENABLE_MPS_FALLBACK=1` set in conftest + examples (torchvision backward has no MPS kernel).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
