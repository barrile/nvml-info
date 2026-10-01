# nvml-info

Thin, Pythonic wrapper around `pynvml` for discovering and querying NVIDIA GPUs. Provides typed APIs for GPU discovery, memory monitoring, and utilization tracking.

**Requires**: Python 3.12+, NVIDIA driver with NVML support

## Installation

```bash
pip install nvml-info
```

## Quick Start

```python
from nvml_info import CUDA

# Check if NVIDIA GPU is available
if CUDA.is_available():
    cuda = CUDA()
    print(f"Found {cuda.device_count} GPU(s)")
    
    # Find a GPU with at least 8GB free memory and <50% utilization
    gpu = cuda.get_available_GPU(min_memory=8000)
    print(f"Using GPU: {gpu.name}")
    print(f"Free memory: {gpu.free_memory_mb} MiB")
    print(f"Utilization: {gpu.utilization_rates}%")
```

## API

### `CUDA` class

Main entry point for GPU operations.

#### Class Methods

- **`CUDA.is_available() -> bool`**
  - Check if NVIDIA driver is available (does not raise)
  - Safe to call in any environment

#### Instance Methods

- **`__init__()`**
  - Initialize and enumerate all GPUs
  - Raises `RuntimeError` if driver cannot be loaded
  - Populates `devices` (list of GPU objects) and `device_count`

- **`get_available_GPU(min_memory: int) -> GPU`**
  - Find the least-utilized GPU with sufficient free memory
  - `min_memory`: minimum MiB of free VRAM required
  - Returns: GPU object with lowest current utilization
  - Selection criteria: free memory ≥ `min_memory` AND utilization < 50%
  - Raises `StopIteration` if no GPU meets criteria

#### Properties

- **`devices: list[GPU]`** — List of all detected GPUs
- **`device_count: int`** — Total number of GPUs

### `GPU` class

Represents a single NVIDIA GPU.

#### Properties

- **`id: int`** — GPU index (0-based)
- **`uuid: str`** — Unique UUID (NVIDIA driver assigned)
- **`name: str`** — Human-readable name (e.g., "NVIDIA A100")
- **`free_memory_mb: int`** — Current free VRAM in MiB
- **`total_memory_mb: int`** — Total VRAM in MiB
- **`utilization_rates: int`** — Current GPU utilization (0-100%)

#### Methods

- **`__str__()`** — Returns GPU UUID
- **`__eq__`, `__hash__`** — GPU equality/hashing by UUID (hashable, can be used in sets/dicts)

### `Device` class

Pydantic model for device configuration/requests.

```python
from nvml_info import Device

device = Device(type="GPU", required_memory=8000)
```

- `type: Literal["CPU", "GPU"]` — Device type
- `required_memory: int` — Required memory in MiB (default: 8000, min: 0)

## Error Handling

```python
from nvml_info import CUDA

# Safe check (never raises)
if not CUDA.is_available():
    print("NVIDIA driver not available")
    exit(1)

# Will raise if driver unavailable
try:
    cuda = CUDA()
except RuntimeError as e:
    print(f"Failed to initialize CUDA: {e}")

# Will raise if no suitable GPU found
try:
    gpu = cuda.get_available_GPU(min_memory=16000)
except StopIteration:
    print("No GPU with 16GB free memory available")
```

## Examples

### Find GPU with minimum memory

```python
from nvml_info import CUDA

cuda = CUDA()
gpu = cuda.get_available_GPU(min_memory=10000)  # 10GB
print(f"Selected: {gpu.name} (UUID: {gpu.uuid})")
```

### Monitor GPU utilization

```python
from nvml_info import CUDA

cuda = CUDA()
for gpu in cuda.devices:
    print(f"{gpu.name}: {gpu.utilization_rates}% util, {gpu.free_memory_mb}MB free")
```

### Check available capacity

```python
from nvml_info import CUDA

cuda = CUDA()
total_free = sum(gpu.free_memory_mb for gpu in cuda.devices)
print(f"Total free GPU memory: {total_free} MiB")
```

## Dependencies

- **`pydantic`** — Data validation and settings
- **`nvidia-ml-py`** — NVIDIA Management Library bindings (pynvml)

No GPU is required to import; methods will raise if driver is unavailable.

## Testing

Run tests with pytest:

```bash
pytest tests/
pytest tests/ --cov=nvml_info  # with coverage
```

Tests check CUDA availability and skip GPU-specific tests if no driver is found.

## License

MIT

## Contributing

Contributions welcome! Please:
- Write tests for new features
- Follow PEP 8 (black/ruff)
- Add type hints
- Update README for public APIs

## Related

- [pynvml](https://pypi.org/project/nvidia-ml-py/) — Low-level NVIDIA ML bindings
- [gpustat](https://github.com/wookayin/gpustat) — GPU monitoring CLI
- [pytorch](https://pytorch.org/) — Uses similar GPU discovery patterns
