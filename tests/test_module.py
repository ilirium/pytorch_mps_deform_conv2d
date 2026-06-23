"""DeformConv2d module: shape/parity and state_dict interchange with torchvision."""

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.ops import DeformConv2d as TVDeformConv2d  # noqa: E402

from deform_conv2d_mps import DeformConv2d


def test_output_shape_cpu():
    m = DeformConv2d(4, 6, kernel_size=3, padding=1)
    x = torch.randn(2, 4, 8, 8)
    offset = torch.zeros(2, 2 * 3 * 3, 8, 8)
    out = m(x, offset)
    assert out.shape == (2, 6, 8, 8)


def test_state_dict_interchange():
    """Our module and torchvision's must share weight/bias layout."""
    ours = DeformConv2d(4, 6, kernel_size=3, padding=1)
    tv = TVDeformConv2d(4, 6, kernel_size=3, padding=1)
    tv.load_state_dict(ours.state_dict())  # raises if layouts differ

    x = torch.randn(1, 4, 8, 8)
    offset = torch.randn(1, 2 * 3 * 3, 8, 8)
    torch.testing.assert_close(ours(x, offset), tv(x, offset), rtol=1e-5, atol=1e-5)


def test_dcnv2_mask_path_cpu():
    m = DeformConv2d(4, 6, kernel_size=3, padding=1)
    x = torch.randn(2, 4, 8, 8)
    offset = torch.randn(2, 2 * 3 * 3, 8, 8)
    mask = torch.rand(2, 3 * 3, 8, 8)
    out = m(x, offset, mask=mask)
    assert out.shape == (2, 6, 8, 8)
