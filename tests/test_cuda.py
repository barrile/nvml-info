import pytest

from nvml_info import CUDA, Device


class TestCUDAAvailability:
    """Test CUDA driver availability checks."""

    def test_is_available_returns_bool(self):
        """CUDA.is_available() should always return a bool, never raise."""
        result = CUDA.is_available()
        assert isinstance(result, bool)


class TestCUDAInit:
    """Test CUDA initialization."""

    def test_cuda_init_without_driver_raises(self, cuda_available):
        """CUDA() should raise RuntimeError if driver unavailable."""
        if cuda_available:
            pytest.skip("Driver available, skipping negative test")

        with pytest.raises(RuntimeError, match="CUDA driver can't be loaded"):
            CUDA()

    def test_cuda_init_with_driver(self, cuda_available):
        """CUDA() should initialize successfully if driver available."""
        if not cuda_available:
            pytest.skip("NVIDIA driver not available")

        cuda = CUDA()
        assert cuda is not None
        assert hasattr(cuda, "devices")
        assert hasattr(cuda, "device_count")

    def test_cuda_device_count_is_int(self, cuda_instance):
        """device_count should be an integer."""
        if cuda_instance is None:
            pytest.skip("NVIDIA driver not available")

        assert isinstance(cuda_instance.device_count, int)
        assert cuda_instance.device_count >= 0

    def test_cuda_devices_list_length_matches_count(self, cuda_instance):
        """devices list length should match device_count."""
        if cuda_instance is None:
            pytest.skip("NVIDIA driver not available")

        assert len(cuda_instance.devices) == cuda_instance.device_count

    def test_cuda_devices_have_required_attributes(self, cuda_instance):
        """Each GPU should have id, uuid, and name."""
        if cuda_instance is None:
            pytest.skip("NVIDIA driver not available")

        for gpu in cuda_instance.devices:
            assert hasattr(gpu, "id")
            assert hasattr(gpu, "uuid")
            assert hasattr(gpu, "name")
            assert isinstance(gpu.id, int)
            assert isinstance(gpu.uuid, str)
            assert isinstance(gpu.name, str)


class TestGPUQueries:
    """Test GPU property queries."""

    def test_gpu_free_memory_is_int(self, cuda_instance):
        """free_memory_mb should return an integer."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.devices[0]
        assert isinstance(gpu.free_memory_mb, int)
        assert gpu.free_memory_mb >= 0

    def test_gpu_total_memory_is_int(self, cuda_instance):
        """total_memory_mb should return an integer."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.devices[0]
        assert isinstance(gpu.total_memory_mb, int)
        assert gpu.total_memory_mb > 0

    def test_gpu_utilization_is_percentage(self, cuda_instance):
        """utilization_rates should return 0-100."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.devices[0]
        util = gpu.utilization_rates
        assert isinstance(util, int)
        assert 0 <= util <= 100

    def test_gpu_free_less_than_total(self, cuda_instance):
        """Free memory should never exceed total."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.devices[0]
        assert gpu.free_memory_mb <= gpu.total_memory_mb


class TestGPUEquality:
    """Test GPU equality and hashing."""

    def test_gpu_equality_by_uuid(self, cuda_instance):
        """GPUs should be equal if UUIDs match."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu1 = cuda_instance.devices[0]
        gpu2 = cuda_instance.devices[0]
        assert gpu1 == gpu2

    def test_gpu_hashable(self, cuda_instance):
        """GPUs should be hashable and usable in sets."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu_set = set(cuda_instance.devices)
        assert len(gpu_set) > 0
        assert cuda_instance.devices[0] in gpu_set

    def test_gpu_str_representation(self, cuda_instance):
        """str(gpu) should return UUID."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.devices[0]
        assert str(gpu) == gpu.uuid


class TestGetAvailableGPU:
    """Test GPU availability checks."""

    def test_get_available_gpu_with_zero_memory(self, cuda_instance):
        """get_available_GPU(0) should always find a GPU if any exist."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.get_available_GPU(min_memory=0)
        assert gpu is not None

    def test_get_available_gpu_insufficient_memory_raises(self, cuda_instance):
        """get_available_GPU() should raise StopIteration if no GPU meets criteria."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        # Request an unreasonably large amount of memory
        with pytest.raises(StopIteration):
            cuda_instance.get_available_GPU(min_memory=9999999)

    def test_get_available_gpu_returns_gpu_object(self, cuda_instance):
        """get_available_GPU() should return a GPU object."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.get_available_GPU(min_memory=0)
        assert hasattr(gpu, "uuid")
        assert hasattr(gpu, "name")
        assert hasattr(gpu, "id")

    def test_get_available_gpu_respects_memory_constraint(self, cuda_instance):
        """Returned GPU should have at least requested free memory."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        min_mem = 1  # 1 MiB - should be reasonable
        gpu = cuda_instance.get_available_GPU(min_memory=min_mem)
        assert gpu.free_memory_mb >= min_mem

    def test_get_available_gpu_respects_utilization_constraint(self, cuda_instance):
        """Returned GPU should have <50% utilization."""
        if cuda_instance is None or cuda_instance.device_count == 0:
            pytest.skip("No GPUs available")

        gpu = cuda_instance.get_available_GPU(min_memory=0)
        assert gpu.utilization_rates < 50


class TestDeviceModel:
    """Test Device Pydantic model."""

    def test_device_creation_defaults(self):
        """Device should have sensible defaults."""
        device = Device(type="GPU")
        assert device.type == "GPU"
        assert device.required_memory == 8000

    def test_device_custom_memory(self):
        """Device should accept custom required_memory."""
        device = Device(type="CPU", required_memory=16000)
        assert device.type == "CPU"
        assert device.required_memory == 16000

    def test_device_zero_memory_allowed(self):
        """Device should accept 0 memory (validation ge=0)."""
        device = Device(type="GPU", required_memory=0)
        assert device.required_memory == 0

    def test_device_negative_memory_rejected(self):
        """Device should reject negative memory."""
        with pytest.raises(ValueError):
            Device(type="GPU", required_memory=-1)

    def test_device_invalid_type_rejected(self):
        """Device should only accept 'CPU' or 'GPU'."""
        with pytest.raises(ValueError):
            Device(type="TPU")  # type: ignore
