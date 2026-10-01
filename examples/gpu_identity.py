"""Show that GPUs are hashable and compared by UUID.

Demonstrates `GPU.__str__`, `GPU.__eq__`, and `GPU.__hash__` so GPUs
can be used in sets, dict keys, and deduplicated by identity even
though two handles may refer to the same physical card.
"""

from nvml_info import CUDA


def main() -> None:
    cuda = CUDA()
    if not cuda.devices:
        print("No GPUs available.")
        return

    a = cuda.devices[0]
    b = cuda.get_available_GPU(min_memory=0)

    print(f"str(a)            = {a!s}")
    print(f"a == b            = {a == b}")
    print(f"hash(a) == hash(b)= {hash(a) == hash(b)}")

    gpus = {a, b}
    print(f"len({{a, b}})        = {len(gpus)}  (set dedupes by UUID)")

    labelled: dict = {a: "primary"}
    print(f"dict[a]           = {labelled[a]!r}")


if __name__ == "__main__":
    main()
