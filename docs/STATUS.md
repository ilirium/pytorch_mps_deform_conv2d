# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-05

**Overall:** Phase 0 done, Phase 1 code-complete (host wiring + diagnostics written; NOT yet built or run on-device). Package is usable now via the torchvision fallback.

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | 🟡 Code-complete, unverified (needs on-device build + `tests/diag_im2col.py`) |
| 2 | Forward correctness tests vs torchvision | ⬜ Not started (tests written, run on fallback) |
| 3 | Native backward (`col2im`, `col2im_coord`) | ⬜ Not started (stubs) |
| 4 | Backward tests + gradcheck | ⬜ Not started |
| 5 | Packaging, perf, groups / half precision | ⬜ Not started |

## What works today

- Build system (`setup.py` + Makefile), packaging, repo layout — complete.
- Metal pipeline validated end to end: `add_one` kernel compiles, dispatches through `torch::mps`, registers under the MPS dispatch key.
- Python API (`deform_conv2d`, `DeformConv2d`) complete and torchvision-compatible; routes to the torchvision fallback (`_NATIVE_READY = False`).
- `deformable_im2col` Metal kernel written (single-group reference port, untested).

## What's missing

- On-device verification of the new forward wiring (written on a non-Mac machine; never compiled or run). Run `make build`, then `python tests/diag_im2col.py`.
- `deformable_col2im` and `deformable_col2im_coord` kernels are empty stubs.
- `_DeformConv2dFunction.backward()` raises `NotImplementedError`.
- `_NATIVE_READY` still `False` in `ops.py` — flip only after Phase 2 passes.

## Phase 1 implementation notes (2026-07-05)

- `deform_conv2d_forward` in `.mm`: validation → column buffer → per-batch `deformable_im2col` dispatch → interleaved ATen `mm` → bias. PSO cache added (`pipeline_for` now memoizes; cleared on library recompile).
- Build verified on-device (2026-07-05): `make build` succeeds. The 34 `-Wc++20-extensions` warnings (see `Implementing_Phase1_001_bug.txt`) were from torch ≥ 2.14 headers under `-std=c++17`; setup.py now uses `-std=c++20`. The duplicate `-rpath` ld warning (conda LDFLAGS + torch BuildExtension) is harmless.
- Running `python tests/diag_im2col.py` directly aborted with OMP Error #15 (see `Implementing_Phase1_002_good.txt`): the `KMP_DUPLICATE_LIB_OK` workaround only lived in the Makefile. The diag scripts and `tests/conftest.py` now set it themselves before `import torch`; `make diag` added as the canonical entry point.
- Threading: the ATen `mm` runs *outside* `dispatch_sync` (ATen MPS ops sync on the same serial queue → deadlock otherwise); each iteration encodes+commits im2col like `add_one`, and the command buffer is re-fetched per iteration.
- `tests/diag_im2col.py` (Step 5 ladder): identity-weight trick recovers the raw column buffer through the forward; stages = unfold parity → constant offset vs Python bilinear reference → random offset+mask → full pipeline vs torchvision CPU.

## Next actions

1. On a Mac: `make build`, then `python tests/diag_im2col.py` — fix index-math bugs it flags. (Phase 1 exit)
2. Run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` against the torchvision CPU reference. (Phase 2)
3. Port the two backward kernels; requires `atomic<float>` → set `MTLLanguageVersion3_0` in compile options. (Phase 3)

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile (OMP Error #15 workaround).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
