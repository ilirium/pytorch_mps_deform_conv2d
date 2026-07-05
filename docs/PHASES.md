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

## Phase 1 — Native forward 🟡 In progress (~50%)

**Goal:** forward = deformable im2col (Metal) → GEMM (`torch.matmul`) → bias.

Done:

- `deformable_im2col` MSL kernel written in `src/deform_conv2d_mps/_C/deform_conv2d.metal` — bilinear gather + modulation mask, one thread per (channel, out_y, out_x). Single-group reference port of torchvision's CUDA kernel (`groups == deformable_groups == 1`).
- Shared bilinear helpers in `bilinear.metalh` (inlined into the shader source at load time by `ops.py:_shader_source()`).
- Python side ready: `_DeformConv2dFunction.forward()` calls the native op; `DCN_MPS_FORCE_NATIVE=1` bypasses the fallback for testing.

Remaining (the actual work):

- `deform_conv2d_forward` in `deform_conv2d_mps.mm` is a `TORCH_CHECK(false)` stub. Implement:
  1. Allocate column buffer `(C*kh*kw, out_h*out_w)`.
  2. Fill `DeformConvParams`, dispatch `deformable_im2col` per batch element (mirror the `add_one` encoder pattern).
  3. `out_n = weight.view({outC, -1}).mm(columns)` → reshape to `(outC, out_h, out_w)`; stack over batch; add bias.
- Keep dispatch in-stream; avoid unnecessary `synchronize()`.

## Phase 2 — Forward correctness ⬜ Not started

**Goal:** MPS forward matches `torchvision.ops.deform_conv2d` (CPU reference).

- `tests/test_forward.py` exists but currently exercises the fallback.
- Run `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` once Phase 1 lands.
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
