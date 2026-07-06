# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-06

**Overall:** Phases 0–3 done — the native forward is live for inference (`_FORWARD_READY = True`), and the **native backward is implemented and passing the full diag ladder** (2026-07-06, `Implementing_Phase3_004_good.txt`): all five grads match torchvision CPU autograd end to end. Training still routes to the torchvision fallback (`_BACKWARD_READY = False`) until Phase 4 gradchecks it and flips the flag. Next: Phase 4 (backward correctness tests).

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | ✅ Done (all 6 `make diag` stages pass on-device) |
| 2 | Forward correctness tests vs torchvision | ✅ Done (60 native cases pass; forward enabled for inference) |
| 3 | Native backward (`col2im`, `col2im_coord`) | ✅ Done (all 4 `diag_col2im` stages pass on-device; gated off pending Phase 4) |
| 4 | Backward tests + gradcheck | ⬜ Not started |
| 5 | Packaging, perf, groups / half precision | ⬜ Not started |

## What works today

- Build system (`setup.py` + Makefile), packaging, repo layout — complete.
- Metal pipeline validated end to end: `add_one` kernel compiles, dispatches through `torch::mps`, registers under the MPS dispatch key.
- Python API (`deform_conv2d`, `DeformConv2d`) complete and torchvision-compatible.
- **Native forward live for inference on MPS**: `_FORWARD_READY = True` with autograd-safe routing — native only when no input requires grad (or grad mode is off); grad-requiring calls use the torchvision fallback until Phase 3/4. `DCN_MPS_FORCE_NATIVE=1` bypasses the gating for testing.
- **Forward verified vs torchvision CPU reference** (`make test-forward-native`, rtol/atol=1e-4): 48-case matrix (3 kernels × stride × pad × dilation × mask) + hand-picked extras — non-square input with asymmetric stride/pad/dilation and 1×3 kernel, bias=None, N=1 with odd channels, offsets ×8 (out-of-bounds bilinear), 2×8×33×35 s=2 smoke case. No kernel bugs found; the Phase 1 port was correct as-is.
- **Native backward implemented** (Phase 3): `deformable_col2im` (atomic scatter → grad_input) and `deformable_col2im_coord` (→ grad_offset/grad_mask) Metal kernels, fused `deform_conv2d_backward` host op (grad_weight/grad_bias via ATen GEMM/sum), and `_DeformConv2dFunction.backward()` wired to it. `DCN_MPS_FORCE_NATIVE=1` runs training natively end to end.
- Full `make diag` passes: forward ladder (6 stages) + backward ladder (`tests/diag_col2im.py`, stages 0–4 incl. all-five-grads smoke vs torchvision CPU at rtol 2e-3).

## What's missing

- Backward not yet gradchecked / correctness-tested beyond the diag smoke (`_BACKWARD_READY = False`; training still falls back by default). `tests/test_backward.py` markers still expect `NotImplementedError` xfails — they xpass under the force flag now; fixing them is Phase 4.
- `groups > 1` and `deformable_groups > 1` untested (dg=2 kept visible as a skipped test until Phase 5; dg wiring verified off-device only).

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

## Phase 3 implementation notes (2026-07-06)

- Landed in four verified increments (`Implementing_Phase3_001…004_good.txt`), riskiest first per [PHASE3_PLAN.md](PHASE3_PLAN.md):
  1. `MTLLanguageVersion3_0` + `atomic_smoke` kernel/op — proved `device atomic_float` scatter-add compiles AND runs before anything depended on it; no regressions in existing shaders from the version bump.
  2. `deformable_col2im` + `get_gradient_weight` in `bilinear.metalh`; single-image host op for isolation testing.
  3. `deformable_col2im_coord` (no atomics — one thread per offset element); single-image host op.
  4. Fused `deform_conv2d_backward` (reuses the single-image ops via `select(0, n)` slices; per-image im2col recompute; three interleaved ATen GEMMs per iteration, all outside `dispatch_sync`) + `ops.py` backward + diag stage 4.
- Off-device verification pattern that worked well: numpy transcriptions of each kernel's index math checked against (a) the transpose of the known-good forward (`col2im == A^T`, exact) and (b) fp64 central finite differences of `L = <g, im2col>` for the coord kernel (~1e-10, offsets kept 0.1 from bilinear kinks). Both ports then passed on-device first try.
- Atomic scatter tolerance: grad_input comparisons use rtol 2e-3 (float atomic add order varies run to run); the coord kernel is deterministic → forward-level 1e-4.
- `make diag-backward` runs the backward ladder alone; `make diag` runs both.

## Next actions

1. Phase 4: run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_backward.py`, fix the xfail markers (they xpass now), add per-grad comparisons + fp32 gradcheck vs CPU, confirm a small training loop converges; flip `_BACKWARD_READY = True` when green.
2. Phase 5: groups / deformable_groups > 1, perf, packaging.

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile + conftest (OMP Error #15 workaround); `PYTORCH_ENABLE_MPS_FALLBACK=1` set in conftest + examples (torchvision backward has no MPS kernel).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
