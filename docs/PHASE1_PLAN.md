# Phase 1 Implementation Plan — Native Forward

**Goal:** replace the `TORCH_CHECK(false)` stub in `deform_conv2d_forward` (`src/deform_conv2d_mps/_C/deform_conv2d_mps.mm`) with a working native path: deformable im2col (Metal) → GEMM (ATen `mm`) → bias.

**Scope:** `groups == deformable_groups == 1`, fp32, contiguous NCHW — matching the existing kernel's reference case. Wider support is Phase 5.

**Exit criterion:** `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` passes (that run is Phase 2; this plan covers getting to a first honest attempt at it).

---

## Step 1 — Input validation & shape computation (host)

In `deform_conv2d_forward`:

- `TORCH_CHECK`: all tensors on MPS, `kFloat`, 4-D input/offset/weight; call `.contiguous()` on input, weight, offset, mask.
- Compute `out_h = (H + 2*pad_h - dilation_h*(kh-1) - 1) / stride_h + 1` (same for `out_w`).
- Validate `offset.size(1) == 2*dg*kh*kw`, `offset.size(2,3) == (out_h, out_w)`; if `use_mask`, same for mask with `dg*kh*kw` channels. `dg = deformable_groups`.
- `use_mask = mask.defined() && mask.numel() > 0` → pass a real buffer either way (bind mask buffer only when used; bind offset buffer as a placeholder at the mask index otherwise, so the encoder always has a valid binding).

## Step 2 — Column buffer & params

- `auto columns = at::empty({C * kh * kw, out_h * out_w}, input.options());` — reused across batch iterations (each im2col dispatch fully overwrites it).
- Fill `DeformConvParams` (field order must match the Metal struct exactly): batch=1 per dispatch, channels=C, height/width, kernel/pad/stride/dilation, out_h/out_w, `deformable_groups=dg`, `channels_per_deformable_group = C / dg`, `use_mask`.
- Pass via `setBytes:&params length:sizeof(params) atIndex:4` (small constant block — no buffer allocation needed).

## Step 3 — Per-batch dispatch loop (host)

Mirror the `add_one` encoder pattern:

```objc
id<MTLComputePipelineState> pso = pipeline_for("deformable_im2col");
dispatch_sync(torch::mps::get_dispatch_queue(), ^{
  @autoreleasepool {
    for (int n = 0; n < N; ++n) {
      id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
      [enc setComputePipelineState:pso];
      // buffer 0: input  + n*C*H*W        elements offset
      // buffer 1: offset + n*(2*dg*kh*kw*out_hw)
      // buffer 2: mask   + n*(dg*kh*kw*out_hw)   (or placeholder)
      // buffer 3: columns (offset 0)
      // buffer 4: params via setBytes
      // dispatchThreads: C*out_h*out_w total, tg = min(maxTotalThreadsPerThreadgroup, total)
      [enc endEncoding];
      // ... GEMM for image n happens outside the encoder (Step 4)
    }
    torch::mps::commit();
  }
});
```

Buffer offsets: `[enc setBuffer:getMTLBufferStorage(t) offset:(t.storage_offset() + n*plane_elems) * t.element_size() atIndex:i]`.

Pipeline-state note: build each PSO once and cache it (static map keyed by function name) — `pipeline_for` currently rebuilds per call, fine for `add_one`, wasteful in a batch loop.

## Step 4 — GEMM + bias

Two options; start with (a), it's simpler:

- **(a) Per-image, interleaved:** after each image's im2col, `auto out_n = weight.view({outC, -1}).mm(columns).view({outC, out_h, out_w});` and copy into pre-allocated `output[n]`. ATen mm on MPS enqueues on the same stream, so ordering with our encoder is preserved without explicit sync.
- **(b) Batched:** allocate `columns` as `(N, C*kh*kw, out_hw)`, one dispatch per image writing to its slice (or fold batch into the grid later), then a single `bmm`. Better perf; do only if (a) is measurably slow.

Bias: `if (bias) output += bias->view({1, outC, 1, 1});`

Do **not** call `torch::mps::synchronize()` — let the stream batch; the returned tensor is valid lazily like any MPS op result.

## Step 5 — Kernel-side verification of `deformable_im2col`

The kernel is a written-but-never-run draft; before trusting the full pipeline, verify it in isolation:

1. Add a small pytest-independent script (`tests/diag_im2col.py`): tiny case N=1, C=1, 3×3 input, 2×2 kernel, stride 1, pad 0, **zero offsets, mask=ones** → columns must equal a plain `torch.nn.functional.unfold`. This isolates base index math from bilinear/offset logic.
2. Then non-zero constant offset (e.g. +0.5) → compare against a 10-line NumPy/Python reference bilinear gather.
3. Then random offsets + mask vs the same reference.

Known things to double-check in the kernel (likely bug sites):

- Offset channel layout: torchvision interleaves `(y, x)` per kernel tap — `off_h_idx` uses index `2*(dgi*kh*kw + k)`, `off_w_idx` `+1`. Present code matches; confirm against reference output, not by eye.
- `col_ptr` layout `(c*kh*kw + k, oy*ow + ox)` must match what the GEMM expects from `weight.view({outC, C*kh*kw})`.
- `bilinear_interpolate` boundary behaviour (must return 0 outside `[-1, H]`/`[-1, W]` bounds like torchvision, including the `h_im > -1` edge case).

## Step 6 — Python glue & first full-pipeline run

- No changes needed in `ops.py` for testing (`DCN_MPS_FORCE_NATIVE=1` already routes to native). Keep `_NATIVE_READY = False` until Phase 2 passes.
- Smoke test: the README usage snippet with `DCN_MPS_FORCE_NATIVE=1`, compare against fallback output with `torch.testing.assert_close(..., rtol=1e-4, atol=1e-5)`.
- Then hand off to Phase 2: `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py`.

## Risks / gotchas

- **Storage offsets:** always add `t.storage_offset()` when computing buffer offsets — slicing per batch element in Python-land tensors would break otherwise (we slice by pointer arithmetic instead, but keep the term for safety).
- **Encoder vs ATen interleaving (option a):** each `mm` ends the current command buffer's encoding scope; re-fetch `torch::mps::get_command_buffer()` inside the loop rather than caching it once, since commits may recycle it.
- **Zero-size edge cases:** `N == 0` or `out_hw == 0` → return correctly-shaped empty tensor before dispatching (Metal dispatch of 0 threads is an error path not worth entering).
- **Index math** remains the expected time sink — hence Step 5's isolation ladder before running the real tests.

## Order of work

1. Steps 1–4 (host wiring) — compiles, runs, produces *some* output.
2. Step 5.1 (unfold parity, zero offsets) — validates base indexing.
3. Steps 5.2–5.3 — validates bilinear + offset + mask.
4. Step 6 — full pipeline smoke test, then Phase 2 test suite.

Environment reminder: build/run only on macOS + Apple Silicon (`make build`, `--no-build-isolation`).
