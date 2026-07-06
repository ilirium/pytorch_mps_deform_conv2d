# Phase 4 Implementation Plan — Backward Correctness

**Goal:** the native backward's gradients are verified against the torchvision CPU reference and gradcheck, a small training loop converges, and `_BACKWARD_READY` flips to `True` — native becomes the default for training on MPS.

**Scope:** fp32, contiguous NCHW, `groups == 1`, `deformable_groups == 1` (dg=2 stays a skipped placeholder for Phase 5). No kernel changes expected — Phase 3's diag ladder already matches CPU autograd end to end; this phase is about test coverage, gradcheck rigor, and un-gating.

**Exit criterion:** `make test-backward-native` green with widened coverage + gradcheck; training-loop test converges; `_BACKWARD_READY = True` flipped; full **unforced** suite green (`make test`, `make test-native`, `make diag`, `make examples`) — the unforced run now exercises the native backward by default, which is the real test of the flip.

**Workflow reminder:** build/run only on macOS + Apple Silicon. Console output goes to `docs/Implementing_Phase4_NNN_{bug,good}.txt`; one commit message per change.

---

## Step 1 — `test-backward-native` Makefile target + baseline run

Phase 2 lesson: the plain target runs without the force flag, so with `_BACKWARD_READY = False` it compares the torchvision fallback to itself and passes trivially.

- Add `test-backward-native: $(NATIVE_ENV) $(PYTEST) -q tests/test_backward.py` next to `test-forward-native`. Always check the log header shows `DCN_MPS_FORCE_NATIVE=1`.
- Run it as-is for a baseline. Phase 3's diag stage 4 says the existing two cases should already pass; a failure here means a real bug — stop and fix before widening.
- Housekeeping while in the file: the docstring still claims the tests "xfail on the scaffold's NotImplementedError" — no such markers exist anymore. Rewrite it to describe the native path. (STATUS.md carries the same stale note; sync in Step 6.)
- Log: `Implementing_Phase4_001_{bug,good}.txt`.

## Step 2 — Widen per-grad comparison coverage

Mirror the Phase 2 forward matrix, but for grads. In `tests/test_backward.py`:

- **Parametrized matrix:** 3 kernels (3×3, 1×1, 5×5 or 1×3) × 2 strides × 2 pads × 2 dilations × mask on/off, `out.sum().backward()`, each of the five grads vs CPU reference. Keep cases small (N=2, C≤6, H,W≤11) — backward is ~3× forward cost and the matrix runs 48×.
- **Hand-picked extras**, ported from the forward's trap list: non-square input with asymmetric stride/pad/dilation + non-square kernel (h/w-swap traps live in *three* kernels now); `bias=None` (grad_bias must be `None`, not zeros); N=1 with odd channels; offsets ×8 (OOB region — exercises the col2im ±2 window guard and the coord kernel's `-2` sentinel); the 2×8×33×35 s=2 grid smoke case.
- **Non-scalar upstream grad:** at least one case with `out.backward(torch.randn_like(out))` — `sum()` gives a constant grad_output, which can hide transposition bugs in the GEMM wiring.
- **Per-grad tolerances:** grad_input rtol/atol 2e-3 (atomic scatter, order varies run to run); grad_offset/grad_mask are deterministic (no atomics) → tighten toward 1e-4 like the forward, loosening only with a measured reason; grad_weight/grad_bias come from ATen GEMM/sum → 1e-4. A grad_input diff that is *stable across runs* and localised is an index bug, not atomic noise.

## Step 3 — Gradcheck

MPS is fp32-only, so this is a two-layer check:

- **Keep `test_gradcheck_cpu_reference`** (fp64 CPU, torchvision) — it guards the math the port targets and costs nothing.
- **Add fp32 gradcheck through the native path:** `DCN_MPS_FORCE_NATIVE=1`, tiny case (N=1, C=1–2, k=2, H=W=5 — gradcheck is O(numel) backward calls), `torch.autograd.gradcheck` with fp32-appropriate knobs:
  - `eps ≈ 1e-3` (fp32 central differences; the default 1e-6 drowns in rounding),
  - `atol/rtol` relaxed to ~1e-2/1e-2 — calibrate on the CPU fp32 fallback first: run the same gradcheck through torchvision CPU fp32 and use tolerances that pass there with margin. If fp32 gradcheck can't be made meaningful even for the reference, fall back to comparing native vs CPU-fp32 *analytic* grads (already Step 2) plus a targeted FD spot-check of a few elements, and say so in the test docstring,
  - `nondet_tol > 0` (~1e-3) — grad_input's atomic scatter makes repeated backward calls differ; without this gradcheck's determinism check fails spuriously,
  - offsets initialised ≥ 0.1 away from integer grid points (Phase 3 trick) — bilinear kinks are genuine non-differentiability, not bugs.
- Check inputs, weight, bias, offset; include a mask-grad case (DCNv2) and a no-mask case (DCNv1).

## Step 4 — Training-loop convergence test

The grads being pointwise-correct should imply this, but it's cheap insurance against something systematic (e.g. a grad accumulated into the wrong buffer across batches):

- Small net (`DeformConv2d` + offset-predicting conv, as in `example01`), synthetic regression target, Adam, ~100–200 steps, `DCN_MPS_FORCE_NATIVE=1`.
- Assert final loss < ~0.1× initial loss, and optionally that the trajectory tracks a CPU run of the identical net (same seed/init) within loose tolerance for the first steps.
- Keep it fast (< a few seconds); mark `@requires_mps`. This also stress-tests many consecutive command-buffer cycles — the closest thing to a soak test we have.

## Step 5 — Flip `_BACKWARD_READY = True`

The actual exit act, landed as its own change *after* Steps 1–4 are green:

- `ops.py`: `_BACKWARD_READY = True`, update the comment (date + Phase 4), update the routing docstring and any "falls back until Phase 3/4" wording in `ops.py` / `README.md`.
- The gating (`_FORWARD_READY and (not needs_grad or _BACKWARD_READY)`) now routes grad-requiring calls natively — verify with the dispatch diagnostic or a quick assert that a `requires_grad` call no longer hits torchvision.
- **Full unforced regression is the point of this step:** `make test` and `make test-module` now exercise the native backward for the first time without the force flag. Also `make test-native`, `make diag`, `make examples` (`example02` runs gradcheck — confirm it still targets a sensible path/tolerance now that native handles training).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` stays in conftest/examples: reference comparisons still run torchvision's backward on MPS tensors in places, and dg=2 (Phase 5) will still fall back.

## Step 6 — Docs sync

- STATUS.md: phase table (Phase 4 ✅), "Overall" paragraph, "What works today" (native training default), remove the stale xfail-marker note from "What's missing", Phase 4 implementation-notes section, next actions → Phase 5.
- PHASES.md: Phase 4 → ✅ Done with a summary in the established format.
- Commit message per change, as usual.

## Risks / gotchas

- **fp32 gradcheck is the riskiest unknown.** It may be too noisy to pass meaningfully even against the CPU fp32 reference. Calibrate on the reference first (Step 3); don't burn time tuning tolerances against the native path before knowing what the reference itself needs. The analytic-comparison tests are the load-bearing check; gradcheck is belt-and-suspenders.
- **`nondet_tol` is not optional** — gradcheck re-runs backward and compares; atomic float scatter makes runs differ at ~1e-3 relative. Forgetting it produces a "backward is not deterministic" failure that looks like a bug.
- **Bilinear kinks:** offsets landing exactly on integer coordinates make the true gradient discontinuous; gradcheck and FD comparisons legitimately fail there. Keep random offsets clamped ≥ 0.1 from the grid (Phase 3 pattern), and don't "fix" the kernel in response.
- **Flipping the flag widens the blast radius:** every grad-requiring test in the suite silently switches from fallback to native. If something unrelated (test_module, examples) breaks after Step 5, the flip found it — that's the step doing its job. Land the flip as its own commit so it's trivially revertible.
- **`sum()`-only backward tests are weak** — constant grad_output can mask GEMM transposition and scaling bugs; the `randn_like` upstream-grad case in Step 2 exists for this.
- **Runtime creep:** a 48-case backward matrix + gradcheck can get slow on-device. Keep matrix tensors tiny; if `make test-backward-native` exceeds ~1–2 min, trim the matrix rather than the extras (the extras encode the known traps).

## Order of work

1. Step 1 (target + baseline + docstring cleanup) — establishes ground truth cheaply.
2. Step 2 (widened per-grad matrix) — the load-bearing correctness evidence.
3. Step 3 (gradcheck, calibrated on the CPU fp32 reference first).
4. Step 4 (training-loop convergence).
5. Step 5 (flip `_BACKWARD_READY`, full unforced regression) → Step 6 (docs sync).
