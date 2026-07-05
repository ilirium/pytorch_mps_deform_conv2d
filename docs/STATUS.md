# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-05

**Overall:** Phase 0 done, Phase 1 half-done. Package is usable now via the torchvision fallback; the native Metal path is not yet wired.

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | 🟡 In progress (~50%) |
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

- Host-side forward dispatch: `deform_conv2d_forward` in `deform_conv2d_mps.mm` is a `TORCH_CHECK(false)` stub — the im2col kernel is never launched.
- `deformable_col2im` and `deformable_col2im_coord` kernels are empty stubs.
- `_DeformConv2dFunction.backward()` raises `NotImplementedError`.

## Next actions

1. Implement forward wiring in `.mm`: column buffer → dispatch `deformable_im2col` per batch element → `weight.view({outC, -1}).mm(columns)` → reshape → bias. (Phase 1)
2. Run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` against the torchvision CPU reference; fix index-math bugs. (Phase 2)
3. Port the two backward kernels; requires `atomic<float>` → set `MTLLanguageVersion3_0` in compile options. (Phase 3)

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile (OMP Error #15 workaround).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
