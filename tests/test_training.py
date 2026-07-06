"""Training-loop convergence through the native backward (Phase 4 Step 4).

The per-grad comparisons and gradcheck in test_backward.py prove pointwise
correctness; this is cheap insurance against something systematic that only
shows up across optimizer steps (e.g. a grad accumulated into the wrong
buffer across batches). It also stress-tests many consecutive command-buffer
cycles — the closest thing to a soak test we have.

Teacher-student setup: the target is produced by a frozen copy of the same
architecture with different init, so the regression task is exactly
representable and the loss can drop far. Two assertions:
  1. Convergence: final loss < 0.1x initial loss.
  2. The first few MPS steps track a CPU run of the identical net
     (same seed/init/data) within loose tolerance — a systematic grad error
     diverges immediately; fp32 + atomic-scatter noise does not.

Run with DCN_MPS_FORCE_NATIVE=1 (`make test-backward-native`) to exercise
the native backward; unforced it exercises whatever the gating routes to.
"""

import copy

import pytest
import torch
from torch import nn

from deform_conv2d_mps import DeformConv2d

from conftest import requires_mps  # noqa: E402


class _TinyDCNNet(nn.Module):
    """DeformConv2d + offset-(and mask-)predicting convs, as in example01."""

    def __init__(self, ch=6, k=3, use_mask=True):
        super().__init__()
        kk = k * k
        self.offset_conv = nn.Conv2d(ch, 2 * kk, 3, padding=1)
        self.mask_conv = nn.Conv2d(ch, kk, 3, padding=1) if use_mask else None
        self.dcn = DeformConv2d(ch, ch, k, padding=1)
        self.head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = (torch.sigmoid(self.mask_conv(x))
                if self.mask_conv is not None else None)
        return self.head(torch.relu(self.dcn(x, offset, mask)))


def _train(net, x, y, steps, lr=1e-2):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@requires_mps()
@pytest.mark.parametrize("use_mask", [False, True])
def test_training_converges(use_mask):
    # Data + frozen teacher target (representable by the student arch).
    torch.manual_seed(0)
    x = torch.randn(8, 6, 16, 16)
    teacher = _TinyDCNNet(use_mask=use_mask)
    with torch.no_grad():
        y = teacher(x)

    # Student: identical init on CPU and MPS.
    torch.manual_seed(1)
    net_cpu = _TinyDCNNet(use_mask=use_mask)
    net_mps = copy.deepcopy(net_cpu).to("mps")

    losses = _train(net_mps, x.to("mps"), y.to("mps"), steps=150)
    assert losses[-1] < 0.1 * losses[0], (
        f"no convergence: initial {losses[0]:.4f} -> final {losses[-1]:.4f}")

    # First steps must track the CPU trajectory (loose: fp32 + atomics).
    cpu_losses = _train(net_cpu, x, y, steps=5)
    for i, (lc, lm) in enumerate(zip(cpu_losses, losses)):
        assert abs(lc - lm) <= 0.25 * abs(lc) + 1e-3, (
            f"step {i}: CPU loss {lc:.5f} vs MPS loss {lm:.5f} — "
            "systematic gradient error, not noise")
