# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-05

**Overall:** Phases 0–1 done — the native forward path works and passes the full `make diag` isolation ladder on-device (2026-07-05, see `Implementing_Phase1_003_good.txt`). Next: Phase 2 (forward test suite). The public API still routes through the torchvision fallback until Phase 2 passes (`_NATIVE_READY = False`).

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | ✅ Done (all 6 `make diag` stages pass on-device) |
| 2 | Forward correctness tests vs torchvision | ⬜ Not started (tests written, run on fallback) |
| 3 | Native backward (`col2im`, `col2im_coord`) | ⬜ Not started (stubs) |
| 4 | Backward tests + gradcheck | ⬜ Not started |
| 5 | Packaging, perf, groups / half precision | ⬜ Not started |

## What works today

- Build system (`setup.py` + Makefile), packaging, repo layout — complete.
- Metal pipeline validated end to end: `add_one` kernel compiles, dispatches through `torch::mps`, registers under the MPS dispatch key.
- Python API (`deform_conv2d`, `DeformConv2d`) complete and torchvision-compatible; routes to the torchvision fallback (`_NATIVE_READY = False`).
- **Native forward verified on-device**: `deformable_im2col` + host wiring pass the full diagnostic ladder (`make diag`) — unfold parity, constant/random offsets, mask, and full pipeline (N=2, bias) vs torchvision CPU, all within rtol=1e-4/atol=1e-5.

## What's missing

- Phase 2 run: `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` (broader shape/param coverage than the diag ladder).
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

1. Run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` (or `make test-native`) against the torchvision CPU reference; flip `_NATIVE_READY = True` (forward-only) when green. (Phase 2)
2. Port the two backward kernels; requires `atomic<float>` → set `MTLLanguageVersion3_0` in compile options. (Phase 3)

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile (OMP Error #15 workaround).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
