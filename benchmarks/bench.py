"""Micro-benchmark: native MPS deform_conv2d vs the CPU-fallback round-trip.

Run:  python benchmarks/bench.py
Reports forward (and, once implemented, forward+backward) latency.
"""

import time

import torch

from deform_conv2d_mps import deform_conv2d


def _bench(fn, iters=50, warmup=10):
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


def make(device, dtype=torch.float32):
    N, inC, outC, H, W, k = 8, 64, 64, 64, 64, 3
    dg = 1
    out = H - k + 1 + 2  # padding=1
    x = torch.randn(N, inC, H, W, device=device, dtype=dtype)
    w = torch.randn(outC, inC, k, k, device=device, dtype=dtype)
    b = torch.randn(outC, device=device, dtype=dtype)
    offset = torch.randn(N, 2 * dg * k * k, out, out, device=device, dtype=dtype)
    mask = torch.rand(N, dg * k * k, out, out, device=device, dtype=dtype)
    return x, w, b, offset, mask


def main():
    print(f"torch {torch.__version__}  mps_available={torch.backends.mps.is_available()}")

    x, w, b, offset, mask = make("cpu")
    cpu_ms = _bench(lambda: deform_conv2d(x, offset, w, bias=b, padding=1, mask=mask))
    print(f"CPU forward:        {cpu_ms:8.3f} ms/iter")

    if torch.backends.mps.is_available():
        xm, wm, bm, om, mm = (t.to("mps") for t in (x, w, b, offset, mask))
        mps_ms = _bench(lambda: deform_conv2d(xm, om, wm, bias=bm, padding=1, mask=mm))
        print(f"MPS forward:        {mps_ms:8.3f} ms/iter   (speedup {cpu_ms / mps_ms:.2f}x)")


if __name__ == "__main__":
    main()
