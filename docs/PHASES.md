# Phase Details — deform_conv2d for PyTorch MPS

Companion to [STATUS.md](STATUS.md). Design rationale lives in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).

---

## Phase 0 — Scaffold & Metal pipeline check ✅ Done

**Goal:** prove the `.mm` → shader → tensor pipeline works before writing deform logic.

Done:

- Repo layout, `pyproject.toml` / `setup.py` (`BuildExtension`), Makefile with build/test/run targets.
- `add_one` Metal kernel compiles at runtime (`newLibraryWithSource:`), dispatches through `torch::mps::get_command_buffer()` / `get_dispatch_queue()`, and returns correct results.
- Op registration via `TORCH_LIBRARY` + `m.impl(..., MPS)` verified with a dispatch diagnostic (`tests/diag_dispatch.py`).
- Hard-won fixes (see git log): build with `--no-build-isolation` so the compile-time `c10::DispatchKey::MPS` value matches the installed torch (otherwise the kernel silently registers under the wrong key, e.g. `IPU`); `KMP_DUPLICATE_LIB_OK` in the Makefile for OMP Error #15; Metal shader compile fixes (dropped `atomic<float>` from the col2im stub for now).

## Phase 1 — Native forward ✅ Done (2026-07-05)

**Goal:** forward = deformable im2col (Metal) → GEMM (ATen `mm`) → bias. See [PHASE1_PLAN.md](PHASE1_PLAN.md) for the plan this followed.

Done:

- `deformable_im2col` MSL kernel — bilinear gather + modulation mask, one thread per (channel, out_y, out_x). Single-group reference port of torchvision's CUDA kernel; shared bilinear helpers in `bilinear.metalh` (inlined by `ops.py:_shader_source()`).
- `deform_conv2d_forward` host wiring in `deform_conv2d_mps.mm`: validation → reused column buffer `(C*kh*kw, out_h*out_w)` → per-batch dispatch → interleaved `weight.view({outC, -1}).mm(columns)` → bias. No explicit `synchronize()`. PSO cache in `pipeline_for`.
- Threading gotcha (documented in the .mm): ATen MPS ops sync on the same serial dispatch queue, so the GEMM must run *outside* our `dispatch_sync` block; command buffer re-fetched per batch iteration.
- Verified on-device by `make diag` (`tests/diag_im2col.py`, all 6 stages pass, `docs/Implementing_Phase1_003_good.txt`): unfold parity ×3 (incl. stride/pad/dilation), constant offset vs Python bilinear reference, random offsets + mask, full pipeline (N=2, bias) vs torchvision CPU.
- Environment fixes along the way: `-std=c++20` (torch ≥ 2.14 headers), `KMP_DUPLICATE_LIB_OK` set inside diag scripts/conftest so they run without make.

Scope note: `groups == 1` enforced; `deformable_groups ≥ 1` wired but only dg=1 is exercised so far (Phase 5 widens coverage).

## Phase 2 — Forward correctness ✅ Done (2026-07-05)

**Goal:** MPS forward matches `torchvision.ops.deform_conv2d` (CPU reference). See [PHASE2_PLAN.md](PHASE2_PLAN.md) for the plan this followed.

Done:

- `make test-forward-native` target added (the plain `test-forward` compares torchvision to itself; the log header must show `DCN_MPS_FORCE_NATIVE=1`).
- 48-case matrix (3 kernels × 2 strides × 2 pads × 2 dilations × mask on/off) passed natively on the first run — no kernel fixes needed; the expected im2col index bugs never materialised.
- Coverage widened with hand-picked extras (all passing, rtol/atol=1e-4): non-square 8×11 input with asymmetric stride/pad/dilation and 1×3 kernel (h/w-swap traps), bias=None, N=1 + odd channels (inC=3/outC=5), offsets ×8 (out-of-bounds bilinear zero region), 2×8×33×35 s=2 grid/threadgroup smoke case. dg=2 kept as a skipped placeholder (Phase 5).
- `_NATIVE_READY` split into `_FORWARD_READY = True` / `_BACKWARD_READY = False` with autograd-safe routing: native only when no input requires grad (or grad mode off); training falls back until Phase 3/4.
- Regression finding: `torchvision::_deform_conv2d_backward` has no MPS kernel — the fallback backward needs `PYTORCH_ENABLE_MPS_FALLBACK=1`, now set in `tests/conftest.py` (was shell-env dependent before).
- Logs: `Implementing_Phase2_001…004` (final green: `_004_good.txt` — 66 passed + 1 skipped full suite, 60 + 1 forward-native).

## Phase 3 — Native backward ✅ Done (2026-07-06)

**Goal:** full autograd — grad input, offset, mask via Metal; grad weight, bias via torch ops. See [PHASE3_PLAN.md](PHASE3_PLAN.md) for the plan this followed.

Done (logs: `Implementing_Phase3_001…004_good.txt`):

- `MTLLanguageVersion3_0` in `dcn_compile_library()` + `atomic_smoke` kernel/op proving `device atomic_float` + `atomic_fetch_add_explicit` compile and run (landed alone, forward regression clean — the version bump surfaced no issues in existing shaders).
- `deformable_col2im` (→ grad_input): one thread per column element, forward's exact offset/mask index expressions, ±2 window with `|Δ| < 1` guard, atomic scatter via `get_gradient_weight` (new in `bilinear.metalh`). Verified off-device as the exact transpose of the forward gather (numpy, machine precision), then on-device.
- `deformable_col2im_coord` (→ grad_offset, grad_mask): one thread per offset element (`bp_dir = oc % 2`), loops over `channels_per_deformable_group` column rows; mask grad from h-component threads only; fully-OOB sentinel `inv_h = inv_w = -2`. No atomics. Verified off-device vs fp64 finite differences (~1e-10), then on-device.
- Both kernels exposed as single-image ops (also used by the diag ladder); fused `deform_conv2d_backward` reuses them per batch slice, recomputes im2col per image (as the reference does), accumulates grad_weight via interleaved ATen GEMMs (all outside `dispatch_sync` — deadlock rule), `grad_bias = grad_output.sum({0,2,3})`.
- `_DeformConv2dFunction.backward()` implemented (positional grads, `mask is None → None`, empty bias → `None`). `_BACKWARD_READY` **stays False** — flipping it is Phase 4's exit criterion; `DCN_MPS_FORCE_NATIVE=1` exercises native training today.
- `tests/diag_col2im.py` ladder (stages 0–4: atomic smoke → fold parity → numerical grad_input → grad_offset/grad_mask → full five-grad backward smoke vs torchvision CPU, DCNv2 + DCNv1). `make diag-backward` new; folded into `make diag`. Full suite + examples stay green.

## Phase 4 — Backward correctness ✅ Done (2026-07-06)

**Goal:** gradients match reference; `_BACKWARD_READY` flips — native becomes the default for training. See [PHASE4_PLAN.md](PHASE4_PLAN.md) for the plan this followed.

Done (logs: `Implementing_Phase4_001…006`, all green first try — no kernel changes needed):

- `make test-backward-native` target (the plain target compares the fallback to itself while gated; log header must show `DCN_MPS_FORCE_NATIVE=1`); stale xfail docstring rewritten.
- Per-grad comparison vs torchvision CPU: 48-case matrix (3×3/1×1/1×3 × stride × pad × dilation × mask) + trap extras (asym-everything, bias=None → grad must be `None`, N=1 odd channels, offsets ×8 for the col2im ±2 window guard and coord `-2` sentinel, 33×35 s=2 smoke, skipped dg=2 placeholder) + a non-scalar upstream-grad case (constant `sum()` grad_output can hide GEMM transposition bugs). Per-grad tolerances: grad_input 2e-3 (atomic scatter), the rest held 1e-4.
- Gradcheck, three layers: fp64 CPU reference (kept); fp32 through the native path (eps 1e-3, atol/rtol 1e-2, `nondet_tol` 1e-3 for the atomic scatter, offsets ≥ 0.1 from bilinear kinks); a CPU-fp32 calibration twin with identical case/knobs so knob problems fail there, not as bogus native failures. Expected fp32 UserWarnings filtered.
- Training-loop convergence (`tests/test_training.py`): teacher–student, DCNv1 + DCNv2, Adam 150 steps, final loss < 0.1× initial, first 5 steps track an identically-initialised CPU run within 25%.
- Pre-flip gate fix: routing now requires `groups == 1 and deformable_groups == 1` (inferred from shapes as torchvision does; groups was hardcoded 1, and dg=2 inference had silently routed native). Then `_BACKWARD_READY = True` as its own commit; full unforced regression green.
- Post-flip follow-ups: `test_grad_call_routes_native` (grad_fn assert guards the gate); `example02` retargeted from torchvision to the package (its CPU-vs-MPS check had been comparing the CPU fallback with itself — diffs exactly 0.0); diag ladder "Next:" pointers updated.

## Phase 5 — groups/dg > 1, perf baseline, packaging ✅ Done (2026-07-07)

**Goal:** `groups > 1` / `deformable_groups > 1` run natively (capability gate lifted), performance measured and recorded, package release-ready. See [PHASE5_PLAN.md](PHASE5_PLAN.md) for the plan this followed.

Done (logs: `Implementing_Phase5_001…006`, all green first try — no kernel changes needed in the entire phase):

- **dg > 1 on-device** (001): the dg index math from Phase 1 was correct as written — zero kernel changes. Skipped dg=2 placeholders became real forward/backward matrix cases (cpg=2, cpg=3 non-power-of-two, dg=C, mask on/off, asym, non-scalar upstream grad). Diag ladders gained dg stages, incl. the distinct-constant-offset-per-group trick: a wrong `deformable_group_index` reads valid memory from the *wrong* group — plausible values, not NaN — so each group gets a different constant offset.
- **groups > 1 forward** (002): host-side only — `deformable_im2col` fills all C channels regardless of groups, so the flat `mm` became `at::bmm` over zero-copy views (weight `(g, outC/g, C/g·kh·kw)`, columns `(g, C/g·kh·kw, out_hw)`; channel-major buffer → contiguous group row-blocks; `view()` throws rather than copies). One GEMM dispatch per image, outside `dispatch_sync` (same serial-queue rule as `mm`). Tests: groups=2 × {C=4, C=6} × mask × asym, groups=2+dg=2 cross term, groups=outC depthwise.
- **groups > 1 backward** (003): same change shape — `grad_columns = bmm(w_gᵀ, go_g)` viewed flat (col2im/col2im_coord consume the full-C buffer unchanged; groups never reaches them), `grad_weight` accumulated as `(g, outC/g, C/g·kh·kw)` via `bmm(go_g, cols_gᵀ)` — the final view was written group-ready in Phase 3. Verified off-device against a block-diagonal-weight ground truth first. Tests mirror 002 + groups=2 non-scalar upstream grads (grouped-GEMM transposition errors hide under `sum()`) + gradcheck groups=2/dg=2 with CPU-fp32 calibration twins.
- **Capability gate lifted** (004, own commit like the Phase 4 flip): `native_capable` deleted from `ops.py` — readiness flags are the only gate; groups/dg still inferred from shapes and passed through. 158 tests pass identically forced and unforced; `test_grad_call_routes_native_groups2_dg2` inverts Phase 4's gate test.
- **Perf baseline** (005): `benchmarks/bench.py` extended (fwd+bwd with pre-made upstream grad; shapes 8×64×64×64 k3, 2×256×100×152, groups=32; native / MPS-fallback / pure-CPU impls; aligned markdown table output). **Training ~14x vs the MPS fallback path, 15–17x vs pure CPU.** Finding: current torchvision nightlies run the *forward* natively on MPS (fwd ~parity, 0.8–1.2x) but the backward still falls back — training is the package's payoff. Table + methodology in [STATUS.md](STATUS.md).
- **Packaging** (006): version 0.1.0 (+ `__init__.__version__` sync), classifiers, tested pins as *tested-with* notes (nightlies can't be hard-pinned), README supported-range/API statement + performance summary, docs sync.
- Stretch items (threadgroup tuning, per-batch dispatch batching, precompiled `.metallib`, half precision, channels-last) deliberately **not** taken: the phase exits on correctness + measured baseline + packaging; tuning waits for a profiler-identified bottleneck. The recorded baseline is what any such work must beat.

---

**Constraint:** native build/test requires macOS + Apple Silicon (`make test`, `DCN_MPS_FORCE_NATIVE=1`). Cannot be built or run in a Linux environment.
