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

> **Status: native forward AND backward live.** On MPS, both inference and
> training run the native Metal kernels: forward verified against the
> torchvision CPU reference (Phase 2, 2026-07-05), backward verified via
> per-grad comparison, fp32 gradcheck, and training-loop convergence
> (Phase 4, 2026-07-06). `groups > 1` and `deformable_groups > 1` run
> natively too (Phase 5, 2026-07-07: grouped GEMMs + dg-indexed kernels,
> verified on-device). Non-MPS devices always use the torchvision fallback.
> See `IMPLEMENTATION_PLAN.md` for the full roadmap.

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

## Performance (Phase 5 baseline, 2026-07-07)

Training (forward+backward) runs **~14x faster** than the CPU-fallback path
MPS users otherwise get, and 15–17x faster than pure-CPU torchvision
(8×64×64×64 k3: 57 ms vs 823/975 ms per iter; groups=32 identical; detection-ish
2×256×100×152: 208 ms vs 2836/3140 ms). Forward-only inference is roughly at
parity with current torchvision nightlies, which run the forward natively on
MPS but still fall back to the CPU for the backward. Full table and
methodology: `docs/STATUS.md`; reproduce with `make bench`.

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
pytest -q                          # full suite (native forward + backward on MPS)
make test-forward-native           # forward tests forced onto the native kernel
make test-backward-native          # backward + training tests, forced native
DCN_MPS_FORCE_NATIVE=1 pytest -q   # force the native path (bypass readiness gating)
python benchmarks/bench.py
```

Native routing is gated in `src/deform_conv2d_mps/ops.py`: MPS calls run
native (`_FORWARD_READY` / `_BACKWARD_READY`, both flipped); the Phase 4
groups/dg capability gate was lifted in Phase 5, so `groups > 1` and
`deformable_groups > 1` route native as well. `DCN_MPS_FORCE_NATIVE=1`
bypasses the readiness gating for testing.

### Roadmap (see IMPLEMENTATION_PLAN.md)

- **Phase 0** — Metal pipeline check (`add_one`). ✅
- **Phase 1** — native forward (`im2col` → matmul → bias). ✅
- **Phase 2** — forward tests vs torchvision. ✅
- **Phase 3** — native backward (`col2im`, `col2im_coord`). ✅
- **Phase 4** — backward tests + gradcheck. ✅
- **Phase 5** — packaging, perf tuning, groups / half precision
