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

## Phase 4 — Backward correctness ⬜ Not started

**Goal:** gradients match reference.

- `tests/test_backward.py` exists; run natively with `DCN_MPS_FORCE_NATIVE=1`.
- Compare each grad against torchvision CPU reference; gradcheck against CPU fp32 (MPS has no fp64 — pick tolerances accordingly).
- Confirm a small training loop converges.
- Exit criterion: all tests pass → flip `_BACKWARD_READY = True` in `ops.py` (native becomes the default for training too).

## Phase 5 — Packaging & performance ⬜ Not started

- `benchmarks/bench.py`: MPS native vs CPU-fallback timings.
- README install/usage polish, pin tested torch/torchvision versions.
- Optional: precompiled `.metallib`, threadgroup tuning, channels-last, `groups` / `deformable_groups` beyond 1 (forward currently hardcodes `groups = 1` in `ops.py`), half precision.

---

**Constraint:** native build/test requires macOS + Apple Silicon (`make test`, `DCN_MPS_FORCE_NATIVE=1`). Cannot be built or run in a Linux environment.
