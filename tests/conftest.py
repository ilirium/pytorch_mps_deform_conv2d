import os

# OMP Error #15 workaround (duplicate OpenMP runtimes in conda envs);
# must be set before `import torch`, so pytest works without the Makefile.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torchvision::_deform_conv2d_backward has no MPS kernel, so the fallback
# path needs the CPU round-trip for grad-requiring MPS tensors until the
# native backward lands (Phase 3/4). Same setting as the example scripts;
# must be set before `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "mps: requires an available MPS device")


@pytest.fixture(scope="session")
def mps_available():
    return torch.backends.mps.is_available()


def requires_mps():
    return pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available")
