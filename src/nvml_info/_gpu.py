from dataclasses import dataclass

from pynvml import (
    nvmlDeviceGetHandleByUUID,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


@dataclass
class _GPU:
    """
    Class to manage a GPU.

    Attributes
    ----------
    id
        ID of the GPU. It varies in function of CUDA_DEVICE_ORDER env variable.
    uuid
        UUID of the GPU (unique identifier).
    name
        Human readable name of the GPU.
    """

    id: int
    uuid: str
    name: str

    def __eq__(self, other):
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)

    def __str__(self):
        return f"{self.uuid}"

    @property
    def utilization_rates(self) -> int:
        """
        Get the utilization rate of the GPU in the last second.

        Returns
        -------
        int
            Utilization rate (per cent).
        """
        return self._get_utilization_rates()

    def _get_free_memory(self) -> int:
        """
        Get the remaining free memory on the GPU.

        Returns
        -------
        int
            Free memory (size in bits).
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByUUID(self.uuid)
        mem = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        return mem.free

    @property
    def free_memory_mb(self) -> int:
        """
        Get the memory currently free on the GPU, used by any process.

        Returns
        -------
        int
            Free memory (size in MiB).
        """
        return self._get_free_memory() // (2**20)

    @property
    def total_memory_mb(self) -> int:
        """
        Get the total memory of the GPU.

        Returns
        -------
        int
            Total memory (size in MiB).
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByUUID(self.uuid)
        mem = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        return mem.total // (2**20)

    @property
    def used_memory_mb(self) -> int:
        """
        Get the memory currently in use on the GPU.

        Returns
        -------
        int
            Used memory (size in MiB).
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByUUID(self.uuid)
        mem = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        return mem.used // (2**20)

    @property
    def memory_utilization(self) -> int:
        """
        Get the GPU memory utilization as a percentage.

        Returns
        -------
        int
            Memory utilization in per cent (0-100).
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByUUID(self.uuid)
        mem = nvmlDeviceGetMemoryInfo(handle)
        nvmlShutdown()

        if mem.total == 0:
            return 0
        return int(round(mem.used / mem.total * 100))

    def _get_utilization_rates(self) -> int:
        """
        Get the utilization rate of the GPU in the last second.

        Returns
        -------
        int
            Utilization rate (per cent).
        """
        nvmlInit()
        handle = nvmlDeviceGetHandleByUUID(self.uuid)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        nvmlShutdown()

        return utilization.gpu
