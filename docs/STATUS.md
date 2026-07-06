# Project Status — deform_conv2d for PyTorch MPS

**Last updated:** 2026-07-06

**Overall:** Phases 0–4 done — **native forward AND backward are live by default on MPS** (`_FORWARD_READY = _BACKWARD_READY = True`, flipped 2026-07-06 after `Implementing_Phase4_005_good.txt`). Backward verified on-device: 48-case per-grad matrix + trap extras vs the torchvision CPU reference, fp32 gradcheck through the native path (with a CPU-calibrated twin), fp64 CPU gradcheck, and a training-loop convergence test. `groups > 1` / `deformable_groups > 1` route to the torchvision fallback until Phase 5. Next: Phase 5 (groups/dg > 1, perf, packaging).

## Phase overview

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold & Metal pipeline check (`add_one`) | ✅ Done |
| 1 | Native forward (`im2col` → matmul → bias) | ✅ Done (all 6 `make diag` stages pass on-device) |
| 2 | Forward correctness tests vs torchvision | ✅ Done (60 native cases pass; forward enabled for inference) |
| 3 | Native backward (`col2im`, `col2im_coord`) | ✅ Done (all 4 `diag_col2im` stages pass on-device; gated off pending Phase 4) |
| 4 | Backward tests + gradcheck | ✅ Done (65 forced-native cases incl. gradcheck + training loop; `_BACKWARD_READY = True`) |
| 5 | Packaging, perf, groups / half precision | ⬜ Not started |

## What works today

- Build system (`setup.py` + Makefile), packaging, repo layout — complete.
- Metal pipeline validated end to end: `add_one` kernel compiles, dispatches through `torch::mps`, registers under the MPS dispatch key.
- Python API (`deform_conv2d`, `DeformConv2d`) complete and torchvision-compatible.
- **Native forward and backward live by default on MPS**: inference AND training route natively (`_FORWARD_READY = _BACKWARD_READY = True`). Routing gates on capability too — `groups == 1` and `deformable_groups == 1` (inferred from shapes as torchvision does); anything else falls back to torchvision until Phase 5. `DCN_MPS_FORCE_NATIVE=1` bypasses all gating for testing; `test_grad_call_routes_native` guards the gate.
- **Forward verified vs torchvision CPU reference** (`make test-forward-native`, rtol/atol=1e-4): 48-case matrix (3 kernels × stride × pad × dilation × mask) + hand-picked extras — non-square input with asymmetric stride/pad/dilation and 1×3 kernel, bias=None, N=1 with odd channels, offsets ×8 (out-of-bounds bilinear), 2×8×33×35 s=2 smoke case. No kernel bugs found; the Phase 1 port was correct as-is.
- **Native backward implemented** (Phase 3): `deformable_col2im` (atomic scatter → grad_input) and `deformable_col2im_coord` (→ grad_offset/grad_mask) Metal kernels, fused `deform_conv2d_backward` host op (grad_weight/grad_bias via ATen GEMM/sum), and `_DeformConv2dFunction.backward()` wired to it.
- **Backward verified** (Phase 4, `make test-backward-native`): 48-case per-grad matrix (3 kernels × stride × pad × dilation × mask) + trap extras (asym-everything, bias=None, N=1 odd channels, offsets ×8, 33×35 s=2 smoke) + non-scalar upstream grad, per-grad tolerances (grad_input 2e-3 atomic scatter, rest 1e-4); fp32 gradcheck through the native path (eps 1e-3, tol 1e-2, nondet_tol 1e-3, CPU-fp32-calibrated twin test); fp64 CPU gradcheck; training-loop convergence (teacher–student, Adam, 150 steps, first steps track an identically-initialised CPU run).
- Full `make diag` passes: forward ladder (6 stages) + backward ladder (`tests/diag_col2im.py`, stages 0–4 incl. all-five-grads smoke vs torchvision CPU at rtol 2e-3).
- `example02` now cross-checks the native backward against the CPU reference (it previously targeted torchvision directly).

## What's missing

- `groups > 1` and `deformable_groups > 1` unimplemented in the kernels — routed to the torchvision fallback (dg=2 kept visible as a skipped test until Phase 5; dg wiring verified off-device only).
- Perf untuned; packaging polish (Phase 5).

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

## Phase 4 implementation notes (2026-07-06)

- Landed in six logged increments (`Implementing_Phase4_001…006`), each green first try — no kernel changes were needed; Phase 3's diag ladder had already caught everything:
  1. `make test-backward-native` target + stale docstring cleanup; baseline green (001).
  2. Per-grad matrix + extras + non-scalar upstream grad (`sum()`'s constant grad_output can hide GEMM transposition bugs). The deterministic grads (offset/mask/weight/bias) held the forward-level 1e-4 — no tolerance calibration needed (002).
  3. fp32 gradcheck: the plan's riskiest unknown, passed with the planned knobs on both the CPU-fp32 calibration twin and the native path (003). Offsets kept ≥ 0.1 from integer grid points; expected "not double precision" UserWarnings filtered.
  4. Training-loop convergence, DCNv1 + DCNv2 (004).
  5. **Pre-flip gate fix:** the routing never checked groups/dg — dg=2 *inference* already routed native unverified, and the flip would have added dg=2 training. Now `groups == 1 and deformable_groups == 1` is required for unforced native routing (groups inferred from shapes as torchvision does; was hardcoded 1). Then the flip itself, as its own commit; full unforced regression green (005).
  6. Post-flip follow-ups the regression surfaced: `test_grad_call_routes_native` (grad_fn assert guards the gate); `example02` had been importing torchvision directly, so its "CPU vs MPS" check compared the CPU fallback with itself (max diffs exactly 0.0) — retargeted at the package, real cross-check now (006).

## Next actions

1. Phase 5: groups / deformable_groups > 1 (lift the capability gate in `ops.py` once kernels support them), perf, packaging.

## Environment / constraints

- Tested against nightly: torch==2.14.0.dev20260622, torchvision==0.29.0.dev20260622.
- Must build with `--no-build-isolation` (dispatch-key value baked in at compile time).
- `KMP_DUPLICATE_LIB_OK` set in Makefile + conftest (OMP Error #15 workaround); `PYTORCH_ENABLE_MPS_FALLBACK=1` set in conftest + examples (torchvision backward has no MPS kernel).
- Native build/test only possible on macOS + Apple Silicon.

See [PHASES.md](PHASES.md) for per-phase details.
