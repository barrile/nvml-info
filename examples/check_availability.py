"""Check whether the NVIDIA driver and NVML are usable.

Demonstrates `CUDA.is_available()`, the safe, non-raising entry point
that should be called before any other operation.
"""

from nvml_info import CUDA


def main() -> None:
    available = CUDA.is_available()
    print(f"CUDA available: {available}")

    if not available:
        print("No NVIDIA driver detected. The other examples will not work.")
        return


if __name__ == "__main__":
    main()
