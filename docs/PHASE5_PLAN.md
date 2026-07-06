# Phase 5 Implementation Plan — Groups, Performance, Packaging

**Goal:** `groups > 1` and `deformable_groups > 1` run natively (capability gate in `ops.py` lifted), performance is measured and recorded (native vs CPU fallback, forward and backward), and the package is release-ready (README, version pins).

**Scope:** fp32, contiguous NCHW. Three sub-tracks, in order of risk: (A) dg > 1 — kernels already index by `deformable_group_index`, but only dg=1 has ever run on-device; (B) groups > 1 — the Metal kernels are group-agnostic (im2col/col2im span all C channels), so this is host-side GEMM slicing plus lifting two `TORCH_CHECK`s; (C) benchmarks + packaging. Half precision, channels-last, precompiled `.metallib`, and threadgroup tuning are **stretch items** — only with measurements in hand, each as its own commit, none blocking phase exit.

**Exit criterion:** forward + backward matrices green with dg ∈ {1, 2} and groups ∈ {1, 2} natively (`make test-forward-native`, `make test-backward-native`); the dg=2 skipped placeholder is a real passing test; capability gate lifted and full **unforced** suite green (`make test`, `make test-native`, `make diag`, `make examples`); `benchmarks/bench.py` reports forward and forward+backward numbers, recorded in STATUS.md/README.

**Workflow reminder:** build/run only on macOS + Apple Silicon. Console output goes to `docs/Implementing_Phase5_NNN_{bug,good}.txt`; one commit message per change.

---

## Step 1 — dg > 1 on-device verification (no expected kernel changes)

The dg index math (`deformable_group_index = c / channels_per_deformable_group`, offset/mask indices `deformable_group_index * col_step + k`) is present in all three kernels and was checked off-device in Phase 3, but no dg > 1 case has ever executed on-device. Host ops already accept dg > 1 (divisibility checks in place); only `ops.py` routes it away.

- **Forced-native first** (`DCN_MPS_FORCE_NATIVE=1` bypasses the capability gate — no routing change needed to start testing).
- Convert the skipped dg=2 placeholder into a real test; add dg cases to both matrices: dg=2 with C=4 and C=6 (cpg=2, cpg=3 — non-power-of-two split), dg=C (one channel per group, extreme case), mask on/off, at least one asym stride/pad/dilation combo. Verify all five grads vs torchvision CPU, per-grad tolerances from Phase 4 (grad_input 2e-3, rest 1e-4).
- Extend the diag ladders: a dg=2 stage in `diag_im2col.py` (constant-offset-per-group trick — give each dg a *different* constant offset so a group-index bug shifts the wrong channels) and in `diag_col2im.py` (numpy transposition check re-run with dg=2 — the Phase 3 off-device harness should already parametrize).
- **Trap to watch:** `deformable_groups` reaches the offset/mask *channel count* (`2·dg·kh·kw`), so a wrong `deformable_group_index` reads valid memory from the wrong group — results are plausible-looking, not NaN. The per-group-distinct-offset diag stage exists precisely for this.
- Log: `Implementing_Phase5_001_{bug,good}.txt`.

## Step 2 — groups > 1 forward (host-side only)

`deformable_im2col` fills the column buffer for all C channels regardless of groups; only the GEMM must become grouped. In `deform_conv2d_mps.mm` forward:

- Drop the `TORCH_CHECK(groups == 1, ...)`; validate `outC % groups == 0` and `C % groups == 0` instead (the `w.size(1) * groups == C` check already exists).
- Replace `weight.view({outC, -1}).mm(columns)` with the grouped form: view columns as `(groups, C/groups·kh·kw, out_h·out_w)` — the column buffer is channel-major, so the group slice along rows is contiguous — view weight as `(groups, outC/groups, C/groups·kh·kw)`, then `at::bmm` (or a per-group `mm` loop; bmm preferred, one dispatch). Same deadlock rule as always: GEMM outside `dispatch_sync`.
- Tests: forward matrix additions with groups=2 × {C=4, C=6} × mask on/off × one asym combo; groups=2 **and** dg=2 together (torchvision requires dg to divide C but groups and dg are independent — cover the cross term); groups=outC (depthwise-flavoured). All vs torchvision CPU at 1e-4.
- Log: `Implementing_Phase5_002_{bug,good}.txt`.

## Step 3 — groups > 1 backward (host-side only)

Same shape of change in `deform_conv2d_backward`:

- `grad_col = wᵀ·grad_out` and `grad_weight += grad_out·colᵀ` become the batched/grouped equivalents (bmm over the `(groups, …)` views; `grad_weight` is already viewed as `{outC, C/groups, kh, kw}` at the end — the view was written group-ready).
- `col2im` / `col2im_coord` need **no changes**: they consume the full-C column buffer and only know about dg.
- Lift the backward's `TORCH_CHECK(groups == 1, ...)`.
- Tests: backward per-grad matrix additions mirroring Step 2's cases (incl. groups=2 + dg=2 cross term and a non-scalar upstream grad with groups=2 — the transposition-bug catcher matters *more* with grouped GEMMs, this is where a wrong slice hides).
- fp32 gradcheck: one tiny groups=2 case and one dg=2 case through the native path, Phase 4 knobs (eps 1e-3, tol 1e-2, nondet_tol 1e-3, offsets ≥ 0.1 off-grid). Calibration twin already exists; add matching cases there.
- Log: `Implementing_Phase5_003_{bug,good}.txt`.

## Step 4 — Lift the capability gate (own commit, like the Phase 4 flip)

- `ops.py`: `native_capable = True` for the implemented range (keep the shape-inference lines — the values still get passed to the ops), or delete the gate entirely; update the module docstring, `_FORCE_NATIVE` comment, and README "falls back until Phase 5" wording.
- **Full unforced regression is the point:** `make test`, `make test-native`, `make diag`, `make examples`. Every dg/groups test now routes native without the force flag; `test_grad_call_routes_native` should gain a dg=2 twin (asserting it *now* routes native — inverting Phase 4's gate test).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` stays in conftest/examples: reference comparisons still run torchvision's backward on MPS tensors.
- Log: `Implementing_Phase5_004_{bug,good}.txt`.

## Step 5 — Benchmarks

`benchmarks/bench.py` exists (forward-only, fixed 8×64×64×64 k=3). Extend, don't rewrite:

- Add forward+backward timing (`out.backward(g)` with pre-made `g`, params `requires_grad`), keeping the existing `torch.mps.synchronize()` bracketing.
- Sweep a few shapes: the current default, a small-batch detection-ish case (2×256×100×152), a groups=32 case. Report ms/iter and native-vs-CPU speedup; note that the "CPU" fallback for MPS tensors round-trips through the fallback path — also bench pure-CPU torchvision as the honest baseline.
- Record the table in STATUS.md (and a summary line in README). These numbers are the **baseline** that any stretch perf work must beat — no tuning before this lands.
- Log: `Implementing_Phase5_005_{bug,good}.txt`.

## Step 6 — Packaging polish

- README: install (`--no-build-isolation` rationale), usage, API compatibility statement, supported range (fp32, NCHW; groups/dg now native), benchmark summary, tested pins: torch==2.14.0.dev20260622 / torchvision==0.29.0.dev20260622 (update if bumped during the phase).
- `pyproject.toml`: version bump, classifiers, pins as *tested-with* notes (nightlies can't be hard-pinned sensibly).
- Log + docs sync (STATUS.md phase table + notes, PHASES.md Phase 5 → ✅ in the established format): `Implementing_Phase5_006_{bug,good}.txt`.

## Stretch (only after Step 5 numbers, each own commit, none required)

- **Threadgroup tuning:** currently 1-D `MIN(maxTotalThreadsPerThreadgroup, 256)`-style sizing; try 2-D threadgroups over (out_x, out_y) for im2col locality. Only if bench shows the kernels (not the GEMMs) dominate.
- **Per-batch dispatch batching:** forward/backward re-fetch the command buffer and commit per image; encoding all N im2col dispatches before the GEMMs may help small-H/W-large-N shapes. Careful with the serial-queue deadlock rule.
- **Precompiled `.metallib`** (build-time `metal`/`metallib` toolchain) — removes first-call compile latency; keep the runtime-compile path as fallback.
- **Half precision / channels-last:** substantial kernel + tolerance work; explicitly out of scope unless there's a concrete need.

## Risks / gotchas

- **dg index bugs read valid-but-wrong memory** — no crash, no NaN, just subtly wrong numbers that a loose tolerance could wave through. The distinct-offset-per-group diag stage and cpg=3 (non-power-of-two) cases are the defence; don't skip them because the matrix passes.
- **Grouped GEMM slice/transposition errors hide under `sum()` backward** — same lesson as Phase 4; the non-scalar upstream-grad case with groups=2 is load-bearing, not optional.
- **`bmm` on MPS follows the same serial-queue rule** as `mm` — all ATen calls outside `dispatch_sync`, command buffer re-fetched per iteration. A deadlock here presents as a silent hang, not an error.
- **Views must not copy:** the grouped views of `columns` and `weight` are only free because the buffers are contiguous in the right order. Assert `.is_contiguous()` where the code assumes it; an accidental `.contiguous()` copy per batch element would erase the perf story.
- **Benchmark honesty:** first-call Metal library compile and PSO cache misses must be excluded (warmup already does this); and the CPU-fallback-on-MPS-tensors path measures transfer + CPU compute, which flatters the native speedup — report pure-CPU too.
- **Scope creep via stretch items:** the phase exits on correctness + measured baseline + packaging. Tuning without a profiler-identified bottleneck is how phases stall.

## Order of work

1. Step 1 (dg > 1 on-device, forced) — riskiest unknown, zero code changes expected.
2. Step 2 (groups forward) → Step 3 (groups backward) — host GEMM wiring.
3. Step 4 (lift the gate, unforced regression) — the flip, own commit.
4. Step 5 (bench baseline) → Step 6 (packaging + docs sync).
5. Stretch items only if Step 5 motivates them.
