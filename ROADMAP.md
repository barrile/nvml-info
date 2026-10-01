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
