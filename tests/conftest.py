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
