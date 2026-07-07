# Research plan: torchvision nightly native MPS deform_conv2d forward

Goal: verify the claim that current torchvision nightlies run `deform_conv2d`
forward natively on MPS (only backward falls back to CPU), and collect
authoritative references: PRs, issues, release notes, docs/dashboards, and
direct source-code links.

## Background (local evidence)

Our Phase 5 bench log showed a CPU-fallback `UserWarning` only for the
*backward* op; fallback forward timing ≈ our native forward timing. Hypothesis:
upstream torchvision added an MPS forward kernel for `deform_conv2d`.

## Steps

1. **Anchor the claim locally.** Re-read the bench log warning text (exact op
   name, e.g. `torchvision::deform_conv2d_backward`) and record the installed
   torchvision nightly version string. These give the search keywords and the
   version window.

2. **Search upstream trackers.**
   - torchvision GitHub: PRs and issues mentioning "deform_conv2d" + "MPS"
     (e.g. tracking issue for MPS op coverage, the PR that added the kernel).
   - PyTorch MPS op-coverage tracking issue(s) (pytorch/pytorch#77764 and
     successors) for cross-references.
   - torchvision release notes / changelog for the version window from step 1.

3. **Locate the source code.**
   - torchvision repo `torchvision/csrc/ops/mps/` — expected Metal/MPS kernel
     and registration (`TORCH_LIBRARY_IMPL(torchvision, MPS, ...)`).
   - Confirm forward is registered for MPS and backward is not (explaining the
     backward-only fallback warning).
   - Record permalinks (pinned to a commit or the main branch) for: kernel
     source, Metal shader if separate, and dispatch registration.

4. **Assess "readiness".** Note any caveats stated upstream: dtype limits,
   deterministic behavior, known issues, whether it shipped in a stable release
   yet or is nightly-only.

5. **Verify.** Fetch each collected link to confirm it resolves and says what
   we cite. Then write findings below and update STATUS.md if warranted.

## Findings (verified 2026-07-07)

**The claim is confirmed.** torchvision has a native MPS *forward* kernel for
`deform_conv2d` since PR #9017 (merged 2025-06-20); it first shipped in the
stable **v0.23.0** release (Aug 2025) and is in every nightly since — including
our bench environment (`torchvision 0.29.0.dev20260622`). The *backward* op has
no MPS implementation upstream as of today, which is exactly why our bench log
shows a CPU-fallback warning only for `torchvision::_deform_conv2d_backward`.

### Where it is described

- **Merged PR (forward kernel):** "[MPS] deformable conv2d kernel" —
  https://github.com/pytorch/vision/pull/9017 — by @Isalia20, merged
  2025-06-20. PR body states: "Adds deformable conv2d kernel forward
  implementation. **Backwards will be added in a followup PR.**" (No such
  follow-up has been merged as of 2026-07-07.)
- **Companion PR (test cleanup):** "Undo part of #9017" —
  https://github.com/pytorch/vision/pull/9115 — by @malfet, reverted #9017's
  unrolled opcheck test back to `generate_opcheck_tests` after a Linux CI
  regression.
- **Release notes:** torchvision **v0.23.0**, "Improvements" section:
  "[MPS] Add deformable conv2d kernel support on MPS (#9017, #9115)" —
  https://github.com/pytorch/vision/releases/tag/v0.23.0
- **Feature-request issues referenced by the PR:**
  - https://github.com/pytorch/vision/issues/7490 — "deform_conv2d for mps"
    (Apr 2023, the main tracking issue)
  - https://github.com/pytorch/vision/issues/8966
  - https://github.com/pytorch/pytorch/issues/141287 — the PyTorch MPS
    missing-op request issue that the `NotImplementedError` / fallback warning
    in our logs points at
- **Dashboard:** the community MPS op-coverage tracker links deform_conv2d via
  issue #7490: https://github.com/users/kulinseth/projects/1
- **Not merged (unrelated to nightly behavior):** PR
  https://github.com/pytorch/vision/pull/9026 ("Deform conv2d mps support",
  @goldfishsound) — an independent fwd+bwd attempt, still open/stale since
  Apr 2025.

### Source code (main branch)

- **MPS forward kernel + dispatch registration:**
  https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/mps/deform_conv2d_kernel.mm
  — `deform_conv2d_forward_kernel(...)` does a Metal `deformable_im2col`
  launch followed by grouped `at::bmm`; ends with
  `TORCH_LIBRARY_IMPL(torchvision, MPS, m)` registering **only**
  `torchvision::deform_conv2d` (no `_deform_conv2d_backward`) — the
  forward-only story in one file.
- **Metal shader (`deformable_im2col`, `bilinear_interpolate_deformable`):**
  https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/mps/mps_kernels.h
- **CPU reference (fwd + bwd, what the fallback runs):**
  https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp
- **Python entry point / autograd wrapper:**
  https://github.com/pytorch/vision/blob/main/torchvision/ops/deform_conv.py

### Readiness caveats noted upstream

- Forward-only: training on MPS still requires `PYTORCH_ENABLE_MPS_FALLBACK=1`
  and pays the CPU round-trip for backward — the gap this package fills.
- #9017 was merged with forward tests passing; #9115 immediately adjusted the
  opcheck test scaffolding. No dtype restrictions called out beyond the usual
  MPS float64 exclusion (kernel is templated via
  `scalarToMetalTypeString(input.scalar_type())`).

### Implication for this package

Forward-only inference is at parity with upstream nightlies; the package's
value is native fwd+bwd (training ~14x vs fallback). Watch
pytorch/vision for the promised backward follow-up PR — if it lands, re-run
the bench and revisit STATUS.md.
