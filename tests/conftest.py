import pytest

from nvml_info import CUDA


@pytest.fixture
def cuda_available():
    """Fixture that checks if CUDA is available."""
    return CUDA.is_available()


@pytest.fixture
def cuda_instance(cuda_available):
    """Fixture that provides a CUDA instance if available."""
    if cuda_available:
        return CUDA()
    return None
