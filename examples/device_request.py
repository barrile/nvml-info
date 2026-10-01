"""Build a typed device request using the `Device` pydantic model.

Demonstrates `Device`, used to express a desired execution device
along with the VRAM it needs. `Device` is independent of NVML, so
this example also runs on machines without an NVIDIA driver.
"""

from pydantic import ValidationError

from nvml_info import Device


def main() -> None:
    gpu = Device(type="GPU", required_memory=8_000)
    print(gpu)
    print(f"  type:            {gpu.type}")
    print(f"  required_memory: {gpu.required_memory} MiB")

    cpu = Device(type="CPU")
    print(cpu)

    try:
        Device(type="TPU", required_memory=-1)  # invalid type, negative memory
    except ValidationError as e:
        print("\nValidation errors caught as expected:")
        for err in e.errors():
            print(f"  - {err['loc']}: {err['msg']}")


if __name__ == "__main__":
    main()
