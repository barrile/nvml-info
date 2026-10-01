"""Pick the best available GPU for a workload.

Demonstrates `cuda.get_available_GPU(min_memory)`, which returns the
least-utilized GPU with at least `min_memory` MiB of free VRAM (and
utilization under 50 %). Raises `StopIteration` if nothing qualifies.
"""

from nvml_info import CUDA

MIN_FREE_MIB = 8_000


def main() -> None:
    cuda = CUDA()
    try:
        gpu = cuda.get_available_GPU(min_memory=MIN_FREE_MIB)
    except StopIteration:
        print(f"No GPU with at least {MIN_FREE_MIB} MiB free.")
        return

    print(f"Selected: {gpu.name}")
    print(f"  uuid:        {gpu.uuid}")
    print(f"  free:        {gpu.free_memory_mb} MiB")
    print(f"  utilization: {gpu.utilization_rates} %")


if __name__ == "__main__":
    main()
