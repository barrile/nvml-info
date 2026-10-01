# Contributing to nvml-info

Thanks for your interest in contributing! This document explains how to get started.

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome all backgrounds and experience levels.

## Getting Started

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/nvml-info.git
cd nvml-info
```

### 2. Install with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --extra dev
```

This installs nvml-info + development tools (pytest, black, ruff, mypy) into a project-local `.venv`.

> Don't have uv? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pipx install uv`.

### 4. Create a branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Workflow

### Run Tests

```bash
uv run pytest                          # Run all tests
uv run pytest tests/ -v                # Verbose output
uv run pytest tests/ --cov=nvml_info   # With coverage report
```

Tests gracefully skip GPU-specific tests if no NVIDIA driver is available.

### Format & Lint

```bash
uv run black src/ tests/               # Format code
uv run ruff check src/ tests/          # Lint
uv run mypy src/                       # Type checking
```

Or run all at once:

```bash
uv run black src/ tests/ && uv run ruff check src/ tests/ && uv run mypy src/
```

### Adding Dependencies

```bash
uv add <package>                  # runtime dep
uv add --dev <package>            # dev-only dep (pytest, ruff, etc.)
```

### Before Committing

- [ ] All tests pass: `uv run pytest`
- [ ] Code formatted: `uv run black src/ tests/`
- [ ] No lint issues: `uv run ruff check src/ tests/`
- [ ] Type safe: `uv run mypy src/`
- [ ] Coverage >= 80% for new code

## Submitting Changes

### 1. Commit with clear messages

```bash
git commit -m "Add: GPU memory caching for performance"
```

Use imperative mood: "Add", "Fix", "Refactor", not "Added", "Fixed", etc.

### 2. Push your branch

```bash
git push origin feature/your-feature-name
```

### 3. Create a Pull Request

On GitHub, click "Compare & pull request" and:

- **Title**: Brief description of what changed (< 70 chars)
- **Description**: Explain *why* this change is useful
  - Link related issues: `Closes #42`
  - Describe testing: "Tested on A100 GPU with 80GB memory"
  - Mention breaking changes if any

Example:

```markdown
## Summary
Add GPU memory caching to avoid repeated nvmlInit calls.

## Motivation
GPU queries were initializing NVML on every call, causing 
~50ms overhead per query. This caches the initialization state.

## Testing
- [x] All existing tests pass
- [x] Added benchmark: 10x faster repeated queries
- [x] Tested on NVIDIA A100, RTX 3090

Closes #15
```

## What We're Looking For

### Bug Fixes
- Reproduction steps in the issue
- Test case that fails before the fix
- Explanation of the root cause

### Features
- Check [ROADMAP.md](ROADMAP.md) first—is it planned?
- Open an issue to discuss before large changes
- Add tests (target: >= 80% coverage)
- Update README if user-facing
- Update docstrings

### Documentation
- Clear examples for common use cases
- Updated ROADMAP.md for roadmap changes
- Docstrings in code

### Performance
- Include before/after benchmarks
- Profile with `cProfile` if claiming speedup
- Don't optimize without measurements

## Project Structure

```
nvml-info/
├── src/nvml_info/        # Main package
│   ├── __init__.py       # Public API
│   ├── cuda.py           # CUDA class
│   └── _gpu.py           # GPU class (internal)
├── tests/                # Test suite
│   ├── conftest.py       # Fixtures
│   └── test_cuda.py      # Tests
├── ROADMAP.md            # Feature roadmap
├── pyproject.toml        # Dependencies & config
└── README.md             # User guide
```

## Design Principles

- **Keep it simple**: Don't abstract prematurely
- **Type safety**: All public functions use type hints
- **Test first**: Write tests, then implementation
- **No surprises**: Document non-obvious behavior
- **Minimal deps**: Only `pydantic` + `nvidia-ml-py`
- **NVIDIA-focused**: AMD/other GPUs as optional add-ons

## Common Patterns

### Adding a new GPU property

```python
# In _gpu.py
@property
def power_draw_watts(self) -> int:
    """Get current power draw in watts."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByUUID(self.uuid)
    power = nvmlDeviceGetPowerUsage(handle)
    nvmlShutdown()
    return power // 1000  # Convert milliwatts to watts
```

Then test it:

```python
# In test_cuda.py
def test_gpu_power_draw(cuda_instance):
    if cuda_instance is None:
        pytest.skip("No GPUs available")
    
    gpu = cuda_instance.devices[0]
    power = gpu.power_draw_watts
    assert isinstance(power, int)
    assert 0 <= power <= 500  # Reasonable range
```

### Adding a CUDA class method

```python
# In cuda.py
def get_coolest_gpu(self) -> _GPU:
    """Find GPU with lowest temperature."""
    if not self.devices:
        raise StopIteration("No GPUs available")
    
    return min(self.devices, key=lambda g: g.temperature)
```

Export in `__init__.py`:

```python
# In __init__.py
from .cuda import CUDA
__all__ = ["CUDA", "Device"]
```

## Questions?

- **Usage question**: [Open a discussion](https://github.com/barrile/nvml-info/discussions)
- **Found a bug**: [Open an issue](https://github.com/barrile/nvml-info/issues)
- **Want to discuss a feature**: [Open an issue](https://github.com/barrile/nvml-info/issues) with `discussion` label

## License

By contributing, you agree to license your work under the MIT License (same as the project).

---

Thank you for contributing! 🙏
