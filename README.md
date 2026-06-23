# pytorch_mps_deform_conv2d

- Repo: https://github.com/ilirium/pytorch_mps_deform_conv2d
- Git: git@github.com:ilirium/pytorch_mps_deform_conv2d.git

Description: This repository provides a PyTorch implementation of deformable convolution for MPS (Metal Performance Shaders)
on Apple Silicon devices. It includes a custom MPS kernel for deformable convolution
and a PyTorch module that integrates with the MPS backend.


torchvision ships CPU + CUDA kernels for `deform_conv2d` but **not MPS**
(pytorch/vision#7490) — on Apple Silicon it raises `NotImplementedError` or
forces slow `.cpu()` round-trips. This package provides the missing kernel,
**API-compatible with `torchvision.ops.deform_conv2d`** (DCNv1 and DCNv2).

> **Status: scaffold.** The project structure, build, Python API, the Phase-0
> Metal pipeline check, and tests are in place. The deformable Metal kernels are
> partially implemented (forward `im2col` is a reference draft; `col2im` /
> `col2im_coord` are stubs). Until the native path passes its tests, the Python
> layer transparently falls back to the torchvision reference, so the package is
> already usable. See `IMPLEMENTATION_PLAN.md` for the full roadmap.

## PyTorch and Torchvision version

Nightly, installed 2026-06023:
- torch==2.14.0.dev20260622
- torchvision==0.29.0.dev20260622

## Install (editable, for development)

```bash
make install            # = pip install -e ".[test]" --no-build-isolation
```

Requires macOS + Apple Silicon, Python ≥ 3.9, PyTorch with MPS, torchvision (for
the reference/fallback and tests). The native extension only builds on macOS;
elsewhere the package installs Python-only and uses the fallback.

> **Build against your installed torch — use `--no-build-isolation`.** The
> extension bakes in the compile-time value of `c10::DispatchKey::MPS`, so it
> must be compiled against the exact torch you run. With pip's default build
> isolation, a *different* torch gets pulled into a temporary overlay and the
> `DispatchKey` enum values may differ — the MPS kernel then silently registers
> under the wrong key (e.g. `IPU`) and calls fail with "not implemented for the
> MPS device". `--no-build-isolation` (and `make build`) compile against the
> installed torch and avoid this. This matters especially on nightly builds.
> Build deps must already be present: `make install-deps`.

## Usage

```python
import torch
from deform_conv2d_mps import deform_conv2d, DeformConv2d

x      = torch.randn(2, 4, 16, 16, device="mps")
weight = torch.randn(6, 4, 3, 3,   device="mps")
offset = torch.randn(2, 2*3*3, 14, 14, device="mps")   # 2*kh*kw channels
mask   = torch.rand(2,   3*3, 14, 14, device="mps")    # DCNv2; omit for DCNv1

y = deform_conv2d(x, offset, weight, mask=mask)        # functional

m = DeformConv2d(4, 6, kernel_size=3, padding=1).to("mps")
y = m(x, offset, mask=mask)                            # module
```

Signatures match torchvision exactly, so existing code and `state_dict`s are
drop-in.

## Layout

```
src/deform_conv2d_mps/
  ops.py        functional deform_conv2d + autograd.Function (+ torchvision fallback)
  module.py     DeformConv2d nn.Module
  _C/
    deform_conv2d_mps.mm   ObjC++ host: op registration, dispatch, library compile
    deform_conv2d.metal    MSL kernels: add_one, im2col, col2im, col2im_coord
    bilinear.metalh        shared bilinear sample / gradient helpers
tests/          forward vs reference, backward + gradcheck, module parity
benchmarks/     MPS vs CPU-fallback timings
IMPLEMENTATION_PLAN.md     phased roadmap and design rationale
```

## Development

```bash
pytest -q                          # forward/module tests run on the fallback now
DCN_MPS_FORCE_NATIVE=1 pytest -q   # exercise the native path once Phase 1 lands
python benchmarks/bench.py
```

To enable the native path after implementing forward, set `_NATIVE_READY = True`
in `src/deform_conv2d_mps/ops.py`.

### Roadmap (see IMPLEMENTATION_PLAN.md)

- **Phase 0** — Metal pipeline check (`add_one`). ✅ scaffolded
- **Phase 1** — native forward (`im2col` → matmul → bias)
- **Phase 2** — forward tests vs torchvision
- **Phase 3** — native backward (`col2im`, `col2im_coord`)
- **Phase 4** — backward tests + gradcheck
- **Phase 5** — packaging, perf tuning, groups / half precision
