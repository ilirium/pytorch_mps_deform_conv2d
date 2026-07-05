# Phase 3 Implementation Plan — Native Backward

**Goal:** full native autograd on MPS — grad_input, grad_offset, grad_mask via Metal kernels; grad_weight, grad_bias via plain torch ops. `_DeformConv2dFunction.backward()` stops raising `NotImplementedError`.

**Scope:** fp32, contiguous NCHW, `groups == 1`, `deformable_groups == 1` exercised (dg wiring kept, as in forward). Correctness *testing* is Phase 4 — this phase ends with the kernels implemented, wired, and passing an isolation diag ladder, with `_BACKWARD_READY` still `False`.

**Exit criterion:** `make build` green with `MTLLanguageVersion3_0`; new `tests/diag_col2im.py` ladder passes on-device; forward regression (`make diag`, `make test-forward-native`, `make test`) unchanged; `DCN_MPS_FORCE_NATIVE=1` backward runs end-to-end without `NotImplementedError`.

**Port reference:** `torchvision/csrc/ops/cuda/deform_conv2d_kernel.cu` (`deformable_col2im_kernel`, `deformable_col2im_coord_kernel`), same source the forward was ported from.

**Workflow reminder:** build/run only on macOS + Apple Silicon. Console output goes to `docs/Implementing_Phase3_NNN_{bug,good}.txt`; one commit message per change.

---

## Step 1 — Metal 3.0 language version (riskiest first)

`deformable_col2im` needs scatter-add via `atomic_fetch_add_explicit` on `device atomic_float*`, which requires MSL ≥ 3.0 (macOS 13+). Phase 0 dropped `atomic<float>` from the stub precisely because the default language version rejected it.

- In `dcn_compile_library()` (`deform_conv2d_mps.mm`): set `opts.languageVersion = MTLLanguageVersion3_0;` before `newLibraryWithSource:`.
- Add a trivial atomic smoke kernel (or temporarily give the col2im stub an `atomic_float` argument) to prove the compile actually accepts atomics — a version bump alone can pass while atomics still fail.
- Rebuild, then confirm the Phase 2 baseline is untouched: `make smoke && make diag && make test-forward-native`. A new language version can surface new warnings/errors in *existing* shader code; flush those out before writing new kernels.
- Log: `Implementing_Phase3_001_{bug,good}.txt`.

## Step 2 — `deformable_col2im` kernel (→ grad_input)

Scatter the column-buffer gradient back into `grad_input` with the same sampling locations as the forward gather.

- **Helper:** add `get_gradient_weight(h, w, cur_h, cur_w)` to `bilinear.metalh` — the per-corner bilinear weight `(1-|h-cur_h|)·(1-|w-cur_w|)`, ported from the CUDA reference (keep its bounds handling identical).
- **Thread mapping:** one thread per column element, `total = C·kh·kw·out_h·out_w` per batch image (host loop handles batch, as in forward). Decompose `gid` exactly as the reference: `ox`, `oy`, then `kj`, `ki`, then `c`.
- **Body:** recompute `h_im`, `w_im` from offset (identical index math to `deformable_im2col` — reuse the same expressions, especially the `2*(dgi*kh*kw + k)` / `+1` offset interleaving); apply mask to the col value if `use_mask`; then the reference's ±2 window loop around `(floor(h_im), floor(w_im))`, and for each in-bounds neighbor with `|Δ| < 1`:
  `atomic_fetch_add_explicit(&grad_im[...], weight * val, memory_order_relaxed)`.
- **Signature change:** `grad_im` becomes `device atomic_float*` (buffer binding on the host side is unchanged — same `MTLBuffer`, reinterpreted).

## Step 3 — `deformable_col2im_coord` kernel (→ grad_offset, grad_mask)

- **Thread mapping:** one thread per offset element, `total = 2·dg·kh·kw·out_h·out_w` per image (reference decomposition: `ox`, `oy`, offset channel `oc`; `bp_dir = oc % 2`).
- **Body:** loop over the `channels_per_deformable_group` column rows sharing this offset; for each, accumulate
  `val += col[...] · mask · get_coordinate_weight(im_plane, H, W, h_im, w_im, bp_dir)`
  and (DCNv2) `mval += col[...] · bilinear_interpolate(im_plane, H, W, h_im, w_im)`.
  Write `grad_offset[gid] = val`; when `use_mask` and `oc % 2 == 0`, write the corresponding `grad_mask` element (reference does mask grad from the h-component thread only — port that faithfully).
- Follow the reference's out-of-bounds sentinel (`inv_h = inv_w = -2`) so fully-OOB taps contribute zero; verify our `get_coordinate_weight` bounds guard (added in `bilinear.metalh`) matches the reference behaviour for partially-OOB taps in `(-1, 0)`.
- **No atomics needed:** each thread owns a unique `grad_offset` / `grad_mask` element.

## Step 4 — Host wiring: `deform_conv2d_backward` in the `.mm`

One fused op (mirrors torchvision) returning `(grad_input, grad_offset, grad_mask, grad_weight, grad_bias)`:

- **Schema:** `deform_conv2d_backward(Tensor grad_output, Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor? bias, int stride_h, … int deformable_groups) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` in `TORCH_LIBRARY` + `m.impl(..., MPS)`.
- **Validation:** same checks as forward (device/dtype/shape/`groups == 1`); `grad_output` must be `(N, outC, out_h, out_w)` contiguous.
- **Allocation:** `grad_input`, `grad_offset`, `grad_mask`, `grad_weight` = `at::zeros` (both kernels and the weight GEMM *accumulate*); reuse one `columns` and one `grad_columns` buffer `(C·kh·kw, out_h·out_w)` across the batch loop.
- **Per-batch loop** (respect the Phase 1 threading rule — every ATen op *outside* `dispatch_sync`, command buffer re-fetched per use, since ATen MPS ops sync on the same serial queue):
  1. `grad_columns = w2d.t().mm(grad_output[n].view(outC, out_hw))` — ATen, outside the block.
  2. Dispatch `deformable_col2im_coord(grad_columns, input[n], offset[n], mask[n]) → grad_offset[n], grad_mask[n]`.
  3. Dispatch `deformable_col2im(grad_columns, offset[n], mask[n]) → grad_input[n]` (atomic scatter).
  4. Re-run existing `deformable_im2col(input[n], offset[n], mask[n]) → columns` (recompute, as the reference does — cheaper than saving N column buffers).
  5. `grad_weight += grad_output[n].view(outC, out_hw).mm(columns.t())` — ATen, outside the block.
- **After the loop:** `grad_bias = grad_output.sum({0, 2, 3})` (only if bias defined). Reshape `grad_weight` to `(outC, C/groups, kh, kw)`.
- Mask-less case: bind the placeholder buffer at the mask index as the forward does; return an empty `grad_mask`.

## Step 5 — `ops.py`: implement `_DeformConv2dFunction.backward()`

- Retrieve saved tensors + `ctx.params`; call `torch.ops.deform_conv2d_mps.deform_conv2d_backward(...)` with the empty-tensor convention for mask/bias used by the forward.
- Return positionally for `(input, weight, offset, mask, bias, stride, padding, dilation, groups, deformable_groups)`: `(grad_input, grad_weight, grad_offset, grad_mask_or_None, grad_bias_or_None, None, None, None, None, None)`. Map `mask is None` → `None`, undefined bias → `None`.
- Optional polish: consult `ctx.needs_input_grad` to skip work (fine to defer; compute-all is correct).
- **Do not flip `_BACKWARD_READY`** — that is Phase 4's exit criterion. Update the docstring TODO only.

## Step 6 — Diag isolation ladder: `tests/diag_col2im.py`

Modelled on `diag_im2col.py` (small tensors, one failure mode per stage), run via a new `diag-backward` Makefile target (and folded into `make diag`):

1. **Fold parity:** zero offsets, mask=None → `deformable_col2im` must equal `torch.nn.functional.fold` of the same column gradient (the non-deformable scatter is exactly `fold`). Catches index/atomic bugs without any bilinear math.
2. **Numerical grad_input:** tiny case (N=1, C=1, k=2, 5×5), constant non-integer offset → compare native grad_input against CPU autograd through torchvision.
3. **Numerical grad_offset/grad_mask:** same tiny case with random offsets + mask → compare against CPU reference grads element-wise; small enough that a wrong element is hand-traceable.
4. **Full backward smoke:** N=2 multi-channel with bias, `DCN_MPS_FORCE_NATIVE=1`, all five grads vs torchvision CPU (rtol/atol 2e-3 — see gotchas).

## Step 7 — Regression + docs sync

- Full pass: `make test` (must stay green — without the force flag, grad paths still fall back), `make test-forward-native`, `make diag`, `make examples`.
- `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_backward.py` may now *xpass* (it xfails on `NotImplementedError` today) — expected; fixing markers and full correctness/gradcheck is Phase 4.
- Update STATUS.md (phase table, "what's missing", next actions) and PHASES.md (Phase 3 → ✅ with a summary).
- Commit message per change, as usual.

## Risks / gotchas

- **Atomics are non-deterministic:** float scatter-add order varies run to run, so grad_input diffs vs CPU are noise-scaled — use rtol/atol ≈ 2e-3 (matching `test_backward.py`), not the forward's 1e-4. A diff that *doesn't* vary across runs and localises to specific pixels is an index bug, not atomics.
- **`MTLLanguageVersion3_0` may reject existing code** — that's why Step 1 lands alone with a full forward regression before any new kernel.
- **Deadlock rule still applies:** three ATen GEMMs per batch iteration interleave with two kernel dispatches; any ATen call inside `dispatch_sync` hangs. Same pattern as forward, more chances to slip.
- **Offset/mask index interleaving must mirror the forward exactly** (`2*(dgi*kh*kw + k)` / `+1`, `(c*kh*kw + k)·out_hw` col layout). Top suspect for any coord-grad failure; the forward expressions are known-good — copy them, don't re-derive.
- **Zero-size / zero-channel edge cases:** guard 0-thread dispatches like the forward does (`output.numel() == 0`, `C == 0`).
- **Buffer index count:** `deformable_col2im_coord` uses 7 bindings (0–6) — the placeholder-binding trick for the unused mask must cover both kernels.
- **Grad buffers must be zeroed**, not `empty` — both kernels accumulate; `at::zeros` on MPS is itself an ATen op → outside `dispatch_sync`.

## Order of work

1. Step 1 (language version + atomic smoke) — unblocks everything, riskiest unknown.
2. Step 2 (col2im) → diag stage 1–2 for it in isolation.
3. Step 3 (col2im_coord) → diag stage 3.
4. Step 4 (host op) + Step 5 (ops.py) → diag stage 4.
5. Step 6 complete ladder → Step 7 (regression + docs).
