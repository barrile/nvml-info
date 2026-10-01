from pynvml import (
    NVMLError,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetName,
    nvmlDeviceGetUUID,
    nvmlInit,
    nvmlShutdown,
)

from ._gpu import _GPU


class CUDA:
    """
    A class to handle CUDA and its usage of GPUs.

    Attributes
    ----------
    devices : list
        List of GPUs on the machine.
    device_count : int
        Number of GPUs on the machine.
    """

    def __init__(self) -> None:
        """
        Initialize the CUDA class.

        Raises
        ------
        RuntimeError
            If CUDA drivers cannot be found.
        """
        if not self.is_available():
            raise RuntimeError("CUDA driver can't be loaded")

        self.devices = self._get_devices()
        self.device_count = len(self.devices)

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if CUDA is available.

        Returns
        -------
        bool
            True if CUDA is available.
        """
        try:
            nvmlInit()
            nvmlDeviceGetCount()
            nvmlShutdown()
            return True
        except NVMLError:
            return False

    def get_available_GPU(self, min_memory: int) -> _GPU:
        """
        Get the first available GPU on the machine.

        Parameters
        ----------
        min_memory
            The minimum amount of memory in MIB that should be available on the GPU.

        Returns
        -------
        _GPU
            First available GPU.

        Raises
        ------
        StopIteration
            If no GPU is available.
        """
        base_cond = (
            lambda gpu: (gpu._get_free_memory()) / (2**20) >= min_memory
            and gpu.utilization_rates < 50
        )

        availables = [gpu for gpu in self.devices if base_cond(gpu)]
        asc = sorted(availables, key=lambda gpu: gpu.utilization_rates)

        try:
            return asc[0]

        except IndexError:
            raise StopIteration("No GPU available.") from None

    def _get_devices(self) -> list[_GPU]:
        nvmlInit()
        deviceCount = nvmlDeviceGetCount()
        devices = []
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            device_name = nvmlDeviceGetName(handle)
            device_uuid = nvmlDeviceGetUUID(handle)
            device_id = i
            devices.append(_GPU(device_id, device_uuid, device_name))
        nvmlShutdown()

        return devices
