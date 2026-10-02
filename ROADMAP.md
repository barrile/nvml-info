# nvml-info Roadmap

High-level planning for nvml-info features and improvements.

## Current (v0.1.0)

**Core GPU discovery & monitoring**
- ✅ CUDA availability detection
- ✅ GPU enumeration and properties (uuid, name, id)
- ✅ Real-time utilization rates
- ✅ Free/total memory queries
- ✅ GPU selection by available memory + utilization
- ✅ Pydantic Device model for type-safe requests
- ✅ Comprehensive test suite (40+ tests)
- ✅ Full documentation with examples

---

## v0.1.x — Known Issues & Bug Fixes

Found during a code review. Ordered by priority.

**Correctness**
- [ ] `nvmlShutdown()` is not called in a `finally`/context manager. If a query raises, NVML stays initialized and the init ref-count becomes unbalanced (`CUDA.is_available`, `CUDA._get_devices`, and every `_GPU` query). Introduce an NVML init/shutdown context manager.
- [ ] `CUDA.get_available_GPU` raises `StopIteration` when no GPU matches. This is the wrong exception type and becomes `RuntimeError` inside generators. Use `LookupError` or a custom exception.
- [ ] `_GPU.__eq__` assumes `other.uuid` exists and raises `AttributeError` for other types. Return `NotImplemented` for non-`_GPU` objects.
- [ ] `nvmlDeviceGetName`/`nvmlDeviceGetUUID` return `bytes` on older `pynvml` versions, but `_GPU.uuid`/`name` are typed `str`. Decode when needed or pin the minimum `nvidia-ml-py` version.
- [ ] `get_available_GPU` queries utilization several times per GPU (filter, then sort), and the value can change between calls. Query once and reuse the result.

**Design**
- [ ] `_GPU` memory properties (`free_memory_mb`, `total_memory_mb`, `used_memory_mb`, `memory_utilization`) each do a full init → handle lookup → shutdown cycle. Share one helper or a single memory-info call.
- [ ] The 50% utilization threshold in `get_available_GPU` is hard-coded. Make it a parameter.
- [ ] `CUDA.devices` is a snapshot taken at construction and never refreshed. Document this or add a `refresh()` method.
- [ ] `Device` (pydantic model) is exported but not used by `CUDA`. Integrate it with `get_available_GPU` or document its intended use.
- [ ] `get_available_GPU` does not follow PEP 8 naming (rename to `get_available_gpu` and keep a deprecated alias). `deviceCount` in `_get_devices` should be `device_count`.

**Documentation**
- [ ] `_GPU._get_free_memory` docstring says "size in bits", but the value is in bytes.
- [ ] `_GPU.id` docstring says it depends on `CUDA_DEVICE_ORDER`, but the value is the NVML index (PCI bus order), which can differ from CUDA's default ordering.
- [ ] README/ROADMAP claim "40+ tests"; verify and correct.
- [ ] Replace the `yourusername` placeholder URLs in `pyproject.toml`.
- [ ] Update the aspirational roadmap dates, which are now in the past.

**Testing**
- [ ] Most tests skip when no GPU is present, and `pynvml` is never mocked. Add mocked-NVML tests so CI without a GPU still covers the logic, including error paths and the `finally` cleanup.

**Minor**
- [ ] Remove redundant `int(round(...))` in `memory_utilization` (`round` already returns `int`).

---

## v0.2.0 — Performance & Reliability (Q1 2025)

**Optimization & caching**
- [ ] Cache `nvmlInit` state to reduce repeated initialization overhead
- [ ] Batch GPU queries to avoid redundant init/shutdown cycles
- [ ] Memory query result caching (configurable TTL)
- [ ] Performance benchmarks against raw pynvml

**AMD GPU support**
- [ ] ROCm (AMD GPU) detection and querying
- [ ] Unified API: `CUDA()` for NVIDIA, `ROCm()` for AMD, or abstracted `GPUManager()`
- [ ] Tests for both platforms

**Enhanced monitoring**
- [ ] GPU memory trending (history buffer for utilization/memory)
- [ ] Temperature monitoring (where available)
- [ ] Power consumption tracking

**Code quality**
- [ ] Full type hints on `_GPU` class (currently partial)
- [ ] Mypy strict mode compliance
- [ ] Docstring coverage 100%

---

## v0.3.0 — Integration & Extensibility (Q2 2025)

**Metrics & observability**
- [ ] Prometheus metrics export (`/metrics` endpoint)
- [ ] OpenTelemetry instrumentation
- [ ] Structured logging support

**CLI tool**
- [ ] `nvml-info list` — show all GPUs
- [ ] `nvml-info monitor` — real-time GPU stats
- [ ] `nvml-info select` — find GPU matching criteria
- [ ] JSON output for scripting

**Selection algorithms**
- [ ] Load balancing strategies (round-robin, least-utilized, least-fragmented)
- [ ] Multi-GPU packing algorithms
- [ ] Preemption simulation (what if GPU X fails?)

**Async support**
- [ ] Non-blocking GPU queries (async/await)
- [ ] Background monitoring task
- [ ] Event streams (GPU added/removed, threshold alerts)

---

## v0.4.0 — ML Framework Integration (Q3-Q4 2025)

**Framework adapters**
- [ ] PyTorch integration (detect `torch.cuda` devices automatically)
- [ ] TensorFlow/JAX GPU detection
- [ ] vLLM model loading advisor (pick GPU + batch size)

**Kubernetes support**
- [ ] GPU resource requests/limits parsing
- [ ] Node GPU inventory snapshot
- [ ] Pod GPU allocation tracking

**Advanced scheduling**
- [ ] GPU affinity recommendations
- [ ] NUMA-aware selection
- [ ] Fragmentation analysis

---

## Future — Vision

**Community feedback driven**
- [ ] Multi-platform support (Intel Arc, Apple Silicon)
- [ ] GPU cluster management
- [ ] Historical analytics dashboard
- [ ] Integration with Ray/Dask distributed systems
- [ ] Cost estimation (cloud GPU pricing)

---

## Contributing

Want to work on any of these? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Feature requests welcome—[open an issue](https://github.com/barrile/nvml-info/issues)!

---

## Release Timeline

- **v0.2.0**: March 2025 (performance + AMD support)
- **v0.3.0**: June 2025 (CLI + metrics)
- **v0.4.0**: September 2025 (framework integration)

*Dates are aspirational and may shift based on community feedback.*
