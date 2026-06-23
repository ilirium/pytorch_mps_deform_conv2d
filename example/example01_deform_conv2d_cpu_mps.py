"""
Example: torchvision.ops.deform_conv2d on CPU and MPS (Apple Silicon).

Shows two ways to use deformable convolution and runs each on the CPU and MPS
backends, then compares the results numerically.

  1. Functional API  -> torchvision.ops.deform_conv2d(...)
  2. Module API      -> torchvision.ops.DeformConv2d(...)

Notes on MPS
------------
torchvision's deform_conv2d historically had no native MPS kernel. Depending on
your torch / torchvision build, calling it on an MPS tensor may either:
  - run natively (recent nightlies with an MPS kernel), or
  - raise NotImplementedError.

To let unsupported ops transparently fall back to the CPU, set the env var
*before* importing torch:

    PYTORCH_ENABLE_MPS_FALLBACK=1 python example01_deform_conv2d_cpu_mps.py

This script sets it automatically if it is not already set, so the MPS path
runs either way (natively if available, otherwise via CPU fallback).

Run:
    python example01_deform_conv2d_cpu_mps.py
"""

import os

# Must be set before `import torch` to take effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from torchvision.ops import deform_conv2d, DeformConv2d


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------
# Convolution geometry. deform_conv2d supports DCNv2 (with a modulation mask).
N = 2            # batch size
IN_CH = 4        # input channels
OUT_CH = 6       # output channels
H, W = 10, 12    # input spatial size
KH, KW = 3, 3    # kernel size
STRIDE = (1, 1)
PADDING = (1, 1)
DILATION = (1, 1)
GROUPS = 1           # weight groups
OFFSET_GROUPS = 1    # deformable/offset groups

torch.manual_seed(0)


def output_size(size, k, stride, pad, dil):
    return (size + 2 * pad - dil * (k - 1) - 1) // stride + 1


OUT_H = output_size(H, KH, STRIDE[0], PADDING[0], DILATION[0])
OUT_W = output_size(W, KW, STRIDE[1], PADDING[1], DILATION[1])


def make_inputs():
    """Create input, offset, mask, weight and bias on the CPU.

    Tensor shapes (the important part of the API):
      input  : (N, IN_CH, H, W)
      weight : (OUT_CH, IN_CH // GROUPS, KH, KW)
      offset : (N, 2 * OFFSET_GROUPS * KH * KW, OUT_H, OUT_W)
                 -> the factor 2 is the (y, x) offset per sampling location
      mask   : (N,     OFFSET_GROUPS * KH * KW, OUT_H, OUT_W)   [DCNv2, optional]
                 -> modulation scalar per sampling location, typically in [0, 1]
      bias   : (OUT_CH,)
    """
    x = torch.randn(N, IN_CH, H, W)
    weight = torch.randn(OUT_CH, IN_CH // GROUPS, KH, KW)
    bias = torch.randn(OUT_CH)

    # Small random offsets so sampling stays near the regular grid.
    offset = torch.randn(N, 2 * OFFSET_GROUPS * KH * KW, OUT_H, OUT_W) * 0.5

    # Modulation mask in [0, 1] (sigmoid keeps it in range, as in DCNv2).
    mask = torch.sigmoid(torch.randn(N, OFFSET_GROUPS * KH * KW, OUT_H, OUT_W))

    return x, offset, mask, weight, bias


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
def run_functional(device, x, offset, mask, weight, bias):
    x = x.to(device)
    offset = offset.to(device)
    mask = mask.to(device)
    weight = weight.to(device)
    bias = bias.to(device)

    out = deform_conv2d(
        x,
        offset,
        weight,
        bias=bias,
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        mask=mask,  # omit for plain DCNv1
    )
    return out.cpu()


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------
def run_module(device, x, offset, mask, weight, bias):
    layer = DeformConv2d(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        kernel_size=(KH, KW),
        stride=STRIDE,
        padding=PADDING,
        dilation=DILATION,
        groups=GROUPS,
        bias=True,
    ).to(device)

    # Use the same weights/bias as the functional run for a fair comparison.
    with torch.no_grad():
        layer.weight.copy_(weight.to(device))
        layer.bias.copy_(bias.to(device))

    out = layer(x.to(device), offset.to(device), mask.to(device))
    return out.cpu()


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _sync(device):
    """Block until all queued work on `device` has finished.

    Required for fair GPU timing: MPS dispatches asynchronously, so without a
    sync the timer would only measure kernel-launch overhead, not execution.
    """
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark(device, x, offset, mask, weight, bias, iters=50, warmup=10):
    """Time the functional deform_conv2d on `device`.

    Returns (mean_ms_per_iter, total_s) or None if the op is unsupported.
    Tensors are moved to the device once so we measure compute, not transfers.
    """
    import time

    x = x.to(device)
    offset = offset.to(device)
    mask = mask.to(device)
    weight = weight.to(device)
    bias = bias.to(device)

    def step():
        return deform_conv2d(
            x, offset, weight, bias=bias,
            stride=STRIDE, padding=PADDING, dilation=DILATION, mask=mask,
        )

    try:
        for _ in range(warmup):      # warm up kernels / caches / autotuner
            step()
        _sync(device)

        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        _sync(device)
        total = time.perf_counter() - t0
    except NotImplementedError as e:
        print(f"  [{device.type:>4}] NotImplementedError: {e}")
        return None

    return (total / iters) * 1e3, total


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print(f"torch {torch.__version__}")
    import torchvision
    print(f"torchvision {torchvision.__version__}")

    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    print(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"output spatial size: {OUT_H}x{OUT_W}\n")

    x, offset, mask, weight, bias = make_inputs()

    devices = [torch.device("cpu")]
    if mps_available:
        devices.append(torch.device("mps"))

    results = {}
    for api_name, fn in (("functional", run_functional), ("module", run_module)):
        print(f"=== {api_name} API ===")
        for device in devices:
            try:
                out = fn(device, x, offset, mask, weight, bias)
                results[(api_name, device.type)] = out
                print(f"  [{device.type:>4}] ok   shape={tuple(out.shape)} "
                      f"mean={out.mean().item():+.5f}")
            except NotImplementedError as e:
                print(f"  [{device.type:>4}] NotImplementedError: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"  [{device.type:>4}] {type(e).__name__}: {e}")
        print()

    # Compare CPU vs MPS for each API.
    for api_name in ("functional", "module"):
        cpu = results.get((api_name, "cpu"))
        gpu = results.get((api_name, "mps"))
        if cpu is not None and gpu is not None:
            max_abs = (cpu - gpu).abs().max().item()
            allclose = torch.allclose(cpu, gpu, atol=1e-4, rtol=1e-4)
            print(f"{api_name:>10}: max|CPU-MPS| = {max_abs:.3e}  "
                  f"allclose(atol=1e-4) = {allclose}")
        elif cpu is not None:
            print(f"{api_name:>10}: MPS result unavailable (CPU only).")

    # -- Timing comparison (functional API) ---------------------------------
    print("\n=== timing (functional, lower is better) ===")
    timings = {}
    for device in devices:
        res = benchmark(device, x, offset, mask, weight, bias)
        if res is not None:
            per_iter_ms, total_s = res
            timings[device.type] = per_iter_ms
            print(f"  [{device.type:>4}] {per_iter_ms:8.3f} ms/iter   "
                  f"({total_s * 1e3:7.1f} ms total)")

    if "cpu" in timings and "mps" in timings:
        speedup = timings["cpu"] / timings["mps"]
        faster = "MPS" if speedup > 1 else "CPU"
        print(f"\n  MPS vs CPU: {speedup:.2f}x  ->  {faster} faster"
              "  (note: tiny tensors often favor CPU due to dispatch overhead)")


if __name__ == "__main__":
    main()
