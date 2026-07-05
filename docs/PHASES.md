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

## Phase 2 — Forward correctness ⬜ Not started

**Goal:** MPS forward matches `torchvision.ops.deform_conv2d` (CPU reference).

- `tests/test_forward.py` exists but currently exercises the fallback.
- Run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` (or `make test-native`) — Phase 1 has landed.
- Cover: kernel sizes, stride, padding, dilation, with/without mask (DCNv1 + v2). fp32 tolerances.
- Expected pain: index/stride bugs in the im2col port — the known main time sink.
- Exit criterion: tests pass → flip `_NATIVE_READY = True` in `ops.py` (forward-only; backward still falls back).

## Phase 3 — Native backward ⬜ Not started

**Goal:** full autograd — grad input, offset, mask via Metal; grad weight, bias via torch ops.

- `deformable_col2im` (→ grad_input): scatter-add with bilinear weights. Needs `atomic_fetch_add_explicit` on `device atomic_float` → set `opts.languageVersion = MTLLanguageVersion3_0` in `dcn_compile_library()` (currently omitted so the stub compiles).
- `deformable_col2im_coord` (→ grad_offset, grad_mask): port `get_coordinate_weight` spatial derivatives.
- grad_weight = grad_output ⊗ columns (GEMM), grad_bias = sum — plain torch ops.
- Replace the `NotImplementedError` in `_DeformConv2dFunction.backward()` with dispatches of the above.
- Port reference: `torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu`.

## Phase 4 — Backward correctness ⬜ Not started

**Goal:** gradients match reference.

- `tests/test_backward.py` exists; run natively with `DCN_MPS_FORCE_NATIVE=1`.
- Compare each grad against torchvision CPU reference; gradcheck against CPU fp32 (MPS has no fp64 — pick tolerances accordingly).
- Confirm a small training loop converges.
- Exit criterion: all tests pass → `_NATIVE_READY = True` for the full path.

## Phase 5 — Packaging & performance ⬜ Not started

- `benchmarks/bench.py`: MPS native vs CPU-fallback timings.
- README install/usage polish, pin tested torch/torchvision versions.
- Optional: precompiled `.metallib`, threadgroup tuning, channels-last, `groups` / `deformable_groups` beyond 1 (forward currently hardcodes `groups = 1` in `ops.py`), half precision.

---

**Constraint:** native build/test requires macOS + Apple Silicon (`make test`, `DCN_MPS_FORCE_NATIVE=1`). Cannot be built or run in a Linux environment.
