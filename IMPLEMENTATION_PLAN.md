# Deformable Conv2d for PyTorch MPS — Implementation Plan

**Goal:** A `deform_conv2d` implementation that runs natively on Apple Silicon (Metal Performance Shaders) via a custom Metal kernel, exposed as a PyTorch op + module.

**Scope (decided):**
- Variants: **DCNv2 (modulated) with DCNv1 as the `mask=None` special case** — one kernel path, v1 handled by passing a mask of ones.
- **Forward + backward** (full autograd: grad w.r.t. input, offset, mask, weight, bias).
- **API-compatible with `torchvision.ops.deform_conv2d` / `torchvision.ops.DeformConv2d`** — a drop-in MPS replacement, which also gives a free correctness oracle (the torchvision CPU kernel).

---

## 1. Why this is needed

`torchvision::deform_conv2d` ships CPU and CUDA kernels only. On MPS it raises
`NotImplementedError: The operator torchvision::deform_conv2d is not currently implemented for the MPS device`
(tracked in pytorch/vision#7490). The practical fallback today is `.cpu()` round-trips, which kill performance. This project provides the missing Metal kernel.

## 2. Key design insight — only write Metal for the parts that need it

Deformable conv decomposes into:

1. **deformable im2col** — gather, per output location, the input samples at `kernel_h*kernel_w` deformed positions using **bilinear interpolation**, scaled by the modulation mask, into a column buffer. *(needs a custom kernel)*
2. **GEMM** — `weight_reshaped @ columns` → output. *(use `torch.matmul` / `addmm` — already fast on MPS, no custom kernel)*
3. **bias add**. *(plain tensor op)*

Backward mirrors this:
- grad weight → GEMM of grad-output with columns (`torch` op).
- grad input → **deformable col2im** (scatter with bilinear weights). *(custom kernel)*
- grad offset & grad mask → **deformable col2im_coord** (bilinear spatial derivatives). *(custom kernel)*

So you need **three Metal compute kernels** (`deformable_im2col`, `deformable_col2im`, `deformable_col2im_coord`), each a near-direct port of torchvision's CUDA kernels in `torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu`. Everything else reuses ATen/torch ops that already have MPS support. This is the lowest-risk, highest-reuse route.

## 3. Binding / build approach

- **One Objective-C++ `.mm` file** compiled with `torch.utils.cpp_extension` (`-framework Metal -framework Foundation -ObjC++`). Support both `cpp_extension.load(...)` for dev iteration and a `setup.py` (`BuildExtension`) for installable builds.
- **Metal shaders (`.metal` / MSL)**: start by embedding the shader source as a string and compiling at runtime with `newLibraryWithSource:` (simplest, no offline toolchain step); optionally precompile to a `.metallib` later.
- **Dispatch** through PyTorch's own queue/buffer so it stays in-stream with other MPS work:
  `torch::mps::get_command_buffer()`, `torch::mps::get_dispatch_queue()`, encode + `dispatchThreads`, then `torch::mps::commit()` / `synchronize()` as needed.
- **Op registration:** define the schema and impls with `TORCH_LIBRARY` + `m.impl(..., kMPS)`; register the autograd formula via an autograd Function. Simplest first pass: implement forward as the custom op and write backward as a `torch.autograd.Function` in Python that calls the col2im ops. Move autograd into C++ (`torch::autograd::Function`) later if desired.
- Guard everything behind `if torch.backends.mps.is_available()`; keep a CPU fallback so the package imports on any machine.

## 4. Public API (match torchvision)

```python
deform_conv2d(input, offset, weight, bias=None,
              stride=(1,1), padding=(0,0), dilation=(1,1), mask=None) -> Tensor

class DeformConv2d(in_channels, out_channels, kernel_size,
                   stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Match shape conventions exactly: `offset` is `(N, 2*groups*kh*kw, out_h, out_w)`, `mask` is `(N, groups*kh*kw, out_h, out_w)`. Support `groups` and `deformable_groups` like torchvision. v1 = call with `mask=None`.

## 5. Suggested repository layout

```
pytorch_mps_deform_conv2d/
  README.md
  IMPLEMENTATION_PLAN.md          # this file
  pyproject.toml / setup.py        # BuildExtension for the .mm
  src/
    deform_conv2d_mps/
      __init__.py                  # exports deform_conv2d, DeformConv2d
      ops.py                       # torch.library op + autograd.Function
      module.py                    # DeformConv2d nn.Module
      _C/                          # native sources
        deform_conv2d_mps.mm       # ObjC++ host: launch, op registration
        deform_conv2d.metal        # MSL: im2col / col2im / col2im_coord
        bilinear.metalh            # shared bilinear sample/grad helpers
  tests/
    test_forward.py                # vs torchvision CPU reference
    test_backward.py               # gradcheck + vs reference grads
    test_module.py                 # DeformConv2d parity + state_dict load
  benchmarks/
    bench.py                       # MPS vs CPU-fallback timings
```

## 6. Phased execution

**Phase 0 — Scaffold & toolchain (½ day).** Repo layout, `setup.py` with `BuildExtension`, a trivial Metal kernel (e.g. add-one) compiled and dispatched end-to-end to prove the `.mm` → shader → tensor pipeline works on your machine before touching deform logic.

**Phase 1 — Forward.** Port `deformable_im2col` to MSL (bilinear gather + mask). Wire forward = im2col → `torch.matmul` → reshape → bias. Register as an MPS op.

**Phase 2 — Forward correctness.** `tests/test_forward.py`: random inputs, compare MPS output to `torchvision.ops.deform_conv2d` on CPU across kernel sizes, stride, padding, dilation, groups, with/without mask (v1 + v2). Tolerance for fp32.

**Phase 3 — Backward.** Port `deformable_col2im` (grad input) and `deformable_col2im_coord` (grad offset + mask). grad weight/bias via torch ops. Expose through an autograd Function.

**Phase 4 — Backward correctness.** Compare each gradient against torchvision CPU reference grads; run `torch.autograd.gradcheck` (reference on CPU double; MPS validated against CPU fp32 with appropriate tolerances). Confirm a tiny training loop converges.

**Phase 5 — Packaging & perf.** `pip install -e .`, README with install/usage, `benchmarks/bench.py` (MPS vs CPU fallback). Optional: precompiled `.metallib`, threadgroup-size tuning, channels-last, `deformable_groups`, half precision.

## 7. Risks & gotchas

- **fp32-only on MPS**: gradcheck must compare against a CPU fp32 reference, not fp64 — choose tolerances accordingly.
- **Atomics in col2im**: scatter-add into grad input/offset needs `atomic_fetch_add_explicit` on `device atomic_float`; verify availability on your Metal version, else use a deterministic accumulation strategy.
- **Index/stride bugs** are the main time sink — port the CUDA index math carefully and test each kernel in isolation.
- **In-stream sync**: don't over-`synchronize()`; let the command buffer batch. Only sync where correctness/timing requires.
- **macOS/PyTorch version drift**: pin tested versions in README (recent reports of MPS availability breaking on macOS 26 / PyTorch 2.9–2.10 nightlies — test on a stable release first).

## 8. Reference material

- torchvision CUDA kernel to port: `torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu` (and the CPU one as the oracle).
- Issue: pytorch/vision#7490 (`deform_conv2d for mps`).
- Custom MPS kernel via cpp_extension: github.com/smrfeld/pytorch-cpp-metal-tutorial; pytorch/pytorch#81103.
- Apple "Accelerated PyTorch on Mac" (developer.apple.com/metal/pytorch).
