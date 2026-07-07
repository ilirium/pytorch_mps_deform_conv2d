"""Benchmark: native MPS deform_conv2d vs torchvision baselines (Phase 5).

Run:  python benchmarks/bench.py   (or `make bench`)

Three implementations per shape, forward and forward+backward:
  * native   — this package on MPS tensors (native Metal kernels)
  * fallback — torchvision.ops.deform_conv2d on MPS tensors, which
    round-trips through the PYTORCH_ENABLE_MPS_FALLBACK path (transfer +
    CPU compute). This is what MPS users get *without* this package, but
    it flatters the native speedup —
  * cpu      — torchvision.ops.deform_conv2d on CPU tensors: the honest
    pure-CPU baseline (no device transfers).

Warmup runs exclude the one-time Metal library compile and PSO cache
misses from the timings; torch.mps.synchronize() brackets each timed
region. Backward timing uses a pre-made upstream grad (out.backward(g))
with all five inputs requiring grad.

The printed markdown table is the Phase 5 perf baseline recorded in
docs/STATUS.md — any stretch perf work must beat it.
"""

import os

# Must precede `import torch`: OpenMP dupe guard (see Makefile) and the MPS
# fallback flag the `fallback` baseline needs (torchvision has no MPS kernel).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time

import torch
import torchvision
from torchvision.ops import deform_conv2d as tv_deform_conv2d

from deform_conv2d_mps import deform_conv2d


def _bench(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/iter


# name, N, C, outC, H, W, k, groups. dg=1, stride=1, pad=1 throughout.
SHAPES = [
    ("8x64x64x64 k3",        8,  64,  64,  64,  64, 3,  1),   # original default
    ("2x256x100x152 k3",     2, 256, 256, 100, 152, 3,  1),   # detection-ish
    ("8x64x64x64 k3 g32",    8,  64,  64,  64,  64, 3, 32),   # grouped
]

PAD = 1


def make(shape, device, requires_grad, dtype=torch.float32):
    """Inputs for one shape on one device. dg=1; groups via weight.size(1)."""
    _, N, C, outC, H, W, k, g = shape
    torch.manual_seed(0)
    out_h = H + 2 * PAD - k + 1  # stride 1
    out_w = W + 2 * PAD - k + 1
    rg = requires_grad
    x = torch.randn(N, C, H, W, device=device, dtype=dtype, requires_grad=rg)
    w = torch.randn(outC, C // g, k, k, device=device, dtype=dtype,
                    requires_grad=rg)
    b = torch.randn(outC, device=device, dtype=dtype, requires_grad=rg)
    offset = torch.randn(N, 2 * k * k, out_h, out_w, device=device,
                         dtype=dtype, requires_grad=rg)
    mask = torch.rand(N, k * k, out_h, out_w, device=device, dtype=dtype,
                      requires_grad=rg)
    gout = torch.randn(N, outC, out_h, out_w, device=device, dtype=dtype)
    return x, w, b, offset, mask, gout


def bench_impl(op, shape, device, mode, iters, warmup):
    """Time one op on one shape/device. mode: 'fwd' or 'fwdbwd'."""
    x, w, b, offset, mask, gout = make(shape, device, mode == "fwdbwd")
    if mode == "fwd":
        fn = lambda: op(x, offset, w, bias=b, padding=PAD, mask=mask)
    else:
        def fn():
            out = op(x, offset, w, bias=b, padding=PAD, mask=mask)
            out.backward(gout)
    return _bench(fn, iters, warmup)


def main():
    mps_ok = torch.backends.mps.is_available()
    print(f"torch {torch.__version__}  torchvision {torchvision.__version__}  "
          f"mps_available={mps_ok}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print("impls: native = this package on MPS | fallback = torchvision on "
          "MPS tensors (CPU round-trip) | cpu = torchvision on CPU tensors\n")

    rows = []
    for shape in SHAPES:
        name = shape[0]
        for mode, iters, warmup in (("fwd", 30, 5), ("fwdbwd", 15, 5)):
            cpu_ms = bench_impl(tv_deform_conv2d, shape, "cpu", mode,
                                iters, warmup)
            nat_ms = fb_ms = float("nan")
            if mps_ok:
                nat_ms = bench_impl(deform_conv2d, shape, "mps", mode,
                                    iters, warmup)
                fb_ms = bench_impl(tv_deform_conv2d, shape, "mps", mode,
                                   iters, warmup)
            rows.append((name, mode, nat_ms, fb_ms, cpu_ms))
            print(f"[{name:>20s}] {mode:6s} native {nat_ms:9.2f}  "
                  f"fallback {fb_ms:9.2f}  cpu {cpu_ms:9.2f}  (ms/iter)")

    # Markdown table for docs/STATUS.md / README — columns padded so the
    # plain-text output is readable as-is; the :--- / ---: markers keep the
    # same alignment when rendered.
    header = ["shape", "pass", "native MPS (ms)", "MPS fallback (ms)",
              "pure CPU (ms)", "native vs fallback", "native vs CPU"]
    align = ["<", "<", ">", ">", ">", ">", ">"]  # text left, numbers right
    body = []
    for name, mode, nat, fb, cpu in rows:
        label = "forward" if mode == "fwd" else "fwd+bwd"
        body.append([name, label, f"{nat:.2f}", f"{fb:.2f}", f"{cpu:.2f}",
                     f"{fb / nat:.1f}x", f"{cpu / nat:.1f}x"])
    widths = [max(len(r[i]) for r in [header] + body)
              for i in range(len(header))]

    def fmt_row(cells):
        return "| " + " | ".join(
            f"{c:{a}{w}}" for c, a, w in zip(cells, align, widths)) + " |"

    sep = "|" + "|".join(
        (":" + "-" * (w + 1)) if a == "<" else ("-" * (w + 1) + ":")
        for a, w in zip(align, widths)) + "|"

    print("\n" + fmt_row(header))
    print(sep)
    for r in body:
        print(fmt_row(r))


if __name__ == "__main__":
    main()
