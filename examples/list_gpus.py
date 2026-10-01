"""Enumerate every GPU on the machine and print its details.

Demonstrates `CUDA()`, `cuda.device_count`, and `cuda.devices`, plus
the GPU properties `id`, `uuid`, `name`, `free_memory_mb`,
`total_memory_mb`, and `utilization_rates`.
"""

from nvml_info import CUDA


def main() -> None:
    cuda = CUDA()
    print(f"Found {cuda.device_count} GPU(s)\n")

    for gpu in cuda.devices:
        print(f"[{gpu.id}] {gpu.name}")
        print(f"    uuid:             {gpu.uuid}")
        print(f"    free memory:      {gpu.free_memory_mb} MiB")
        print(f"    total memory:     {gpu.total_memory_mb} MiB")
        print(f"    utilization:      {gpu.utilization_rates} %")
        print()


if __name__ == "__main__":
    main()
