# Phase 2 Implementation Plan — Forward Correctness

**Goal:** `DCN_MPS_FORCE_NATIVE=1 pytest tests/test_forward.py` passes against the torchvision CPU reference, then flip `_NATIVE_READY = True` (forward-only) in `ops.py`.

**Scope:** fp32, contiguous NCHW, `groups == 1`, `deformable_groups == 1` (dg wiring exists; broader dg/groups coverage is Phase 5). No kernel changes expected unless tests expose bugs.

**Exit criterion:** all forward tests pass natively on-device; `_NATIVE_READY = True` with autograd-safe routing (training still falls back until Phase 3/4); full `make test` stays green.

**Workflow reminder:** build/run only on macOS + Apple Silicon. Console output goes to `docs/Implementing_Phase2_NNN_{bug,good}.txt` for analysis; one commit message per change.

---

## Step 1 — Tooling prep

- Add a `test-forward-native` Makefile target: `$(NATIVE_ENV) $(PYTEST) -q tests/test_forward.py` (currently `test-forward` runs the fallback and trivially passes; `test-native` runs the whole suite including backward, which still raises).
- Sanity: `make build && make smoke && make diag` — confirm the Phase 1 baseline still passes before trusting new results.

## Step 2 — First native run

- `make test-forward-native` → 96 parametrised cases (3 kernels × 2 strides × 2 pads × 2 dilations × mask on/off).
- Save the log as `docs/Implementing_Phase2_001_{bug,good}.txt`.
- If everything passes on the first run, jump to Step 4 (widen coverage) — the current matrix is close to the diag ladder, so a clean pass is plausible but not yet convincing.

## Step 3 — Triage loop (on failure)

For each failing case:

1. Reduce to a minimal repro (single parametrisation, N=1 if possible) in a throwaway script modelled on `tests/diag_im2col.py` — the identity-weight trick recovers the raw column buffer through the forward.
2. Localise: does the same config pass the unfold-parity stage (base index math) but fail with offsets (bilinear/offset channel logic) or only with mask (modulation)?
3. Known likely bug sites (from the Phase 1 port):
   - offset channel interleaving — `(y, x)` per kernel tap, `2*(dgi*kh*kw + k)` / `+1`;
   - `col_ptr` layout `(c*kh*kw + k, oy*ow + ox)` vs `weight.view({outC, -1})`;
   - `bilinear_interpolate` boundary behaviour (zero outside `[-1, H]`/`[-1, W]`, incl. the `h_im > -1` edge);
   - stride/pad/dilation sign or order swaps (h vs w) — the diag ladder only exercised a few combinations.
4. Fix in the `.metal` / `.mm`, rebuild, rerun the single case, then the full matrix. Log each iteration (`_002_`, `_003_`, …).

## Step 4 — Widen test coverage

Extend `tests/test_forward.py` beyond the current matrix (all currently fixed at N=2, inC=4, outC=6, H=W=9, dg=1, bias always on):

- **Non-square everything:** H≠W input (e.g. 8×11), asymmetric `stride=(2,1)`, `padding=(0,2)`, `dilation=(2,1)`, kernel `(1,3)` — catches h/w swaps the square matrix hides. Add as a second, hand-picked case list rather than a full cartesian product (keep runtime sane).
- **bias=None** (currently untested through the native path).
- **N=1** and odd channel counts (inC=3, outC=5).
- **Boundary stress:** offsets scaled ×8 so many samples land out of bounds — exercises the zero-outside behaviour directly.
- **Larger spatial case** (e.g. 2×8×33×35, k=3, s=2, p=1) as a single smoke case — catches grid-size/threadgroup edge effects the tiny cases miss.
- Keep dg=1; add a `pytest.param(..., marks=pytest.mark.skip(reason="Phase 5"))` placeholder for dg=2 so the gap stays visible.
- Tolerances: keep rtol=1e-4/atol=1e-4; if boundary-stress cases are flaky at that level, loosen atol for those cases only and note why (fp32 bilinear on MPS vs CPU accumulation order).

Rerun `make test-forward-native` after each addition; save the final green log as `Implementing_Phase2_NNN_good.txt`.

## Step 5 — Flip `_NATIVE_READY` (forward-only, autograd-safe)

Flipping the single flag as-is would send training runs into `_DeformConv2dFunction.backward()` → `NotImplementedError` instead of the fallback. Make the flip safe:

- In `ops.py`, split readiness: `_FORWARD_READY = True`, `_BACKWARD_READY = False` (drop `_NATIVE_READY` or keep as alias).
- Route native only when grads aren't needed:
  `needs_grad = torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in (input, weight, offset, mask, bias))`
  `use_native = mps and (_FORCE_NATIVE or (_FORWARD_READY and (not needs_grad or _BACKWARD_READY)))`
- `_FORCE_NATIVE` keeps its current bypass-everything meaning (needed for Phase 4 testing later).
- Update the module docstring + README note: inference on MPS is native; training falls back until Phase 3/4.

## Step 6 — Regression + docs sync

- Full pass: `make test` (fallback paths incl. test_module), `make test-forward-native`, `make diag`, `make examples` (example01 now exercises the native path via the flipped flag — verify its CPU/MPS comparison still holds).
- Update STATUS.md (phase table, "what works today", next actions) and PHASES.md (Phase 2 → ✅ with a summary of what was covered and any kernel fixes).
- Commit message per change, as usual.

## Risks / gotchas

- **False green:** without `DCN_MPS_FORCE_NATIVE=1` the tests compare torchvision to itself. Step 1's dedicated target exists to make the native run unmistakable; double-check the log header shows the env var.
- **Asymmetric params are genuinely new ground** — the diag ladder and Phase 1 tests used mostly symmetric configs; treat h/w swaps as the top suspect for any new failure.
- **Tolerance flakiness vs real bugs:** a max-abs-diff that scales with tensor magnitude is accumulation-order noise; a diff localised to specific output rows/columns (edges, particular taps) is an index bug. Check the failure pattern before touching tolerances.
- **`make test` after the flip:** test_backward on MPS will now route native for grad-requiring inputs *only if forced* — with Step 5's gating it should still hit the fallback and pass unchanged. If it doesn't, the gating logic is wrong.

## Order of work

1. Step 1 (tooling) → Step 2 (first run) — cheap, tells us where we stand.
2. Step 3 as needed until the existing matrix is green.
3. Step 4 (coverage) — iterate.
4. Step 5 (flag flip) → Step 6 (regression + docs).
