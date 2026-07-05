import os

# OMP Error #15 workaround (duplicate OpenMP runtimes in conda envs);
# must be set before `import torch`, so pytest works without the Makefile.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
