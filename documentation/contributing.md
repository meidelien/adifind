Back to [Documentation Index](index.md)

# Contributing

Guidelines for contributing to AdiFind development.

---

## Development Setup

```bash
git clone https://github.com/meidelien/adifind.git
cd adifind
conda env create -f environment.yml
conda activate adifind
pip install -e ".[dev]"   # Installs pytest, black, flake8, mypy, isort
```

---

## Code Style

AdiFind uses these formatting tools, configured in `pyproject.toml`:

| Tool | Configuration |
|:-----|:-------------|
| **black** | Line length 100, Python 3.9+ target |
| **isort** | Profile "black", line length 100 |
| **flake8** | Max line length 100, ignores E203/W503 |
| **mypy** | Python 3.9, strict mode |

### Formatting Commands

```bash
# Format code
black code/ --line-length 100
isort code/ --profile black --line-length 100

# Check style (no changes)
black code/ --check
flake8 code/
mypy code/
```

### Logging

Use the Python `logging` module — not `print()`:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("📊 Processing %d windows", num_windows)
logger.warning("⚠️ Low memory: %.1f GB free", free_gb)
logger.error("❌ Failed to load model: %s", path)
```

Emoji prefixes in log messages are intentional for terminal readability.

---

## Project Structure

All application code lives in `code/` as flat top-level modules:

```
code/
├── main.py                 # Entry point
├── config.py               # Configuration singletons
├── argument_parser.py      # CLI definitions
├── configuration_manager.py
├── models.py               # Detectron2 model loading
├── core_processing.py      # Inference + merging
├── image_processing.py     # Slide I/O
├── visualization.py        # Export + visualization
├── tissue_guided_processing.py
├── tumor_detection.py
├── batch_processing.py
├── system_utils.py
├── ...
├── optimizations/          # Performance optimization modules
│   ├── async_io.py
│   ├── gpu_acceleration.py
│   └── ...
└── test_smoke.py           # Smoke tests
```

> ⚠️ **Important:** Do not create package nesting. All modules are top-level files imported by name.

---

## Running Tests

```bash
cd code
pytest test_smoke.py -v
```

Tests verify:
- All required package imports (PyTorch, Detectron2, OpenSlide, etc.)
- Config singleton loads with correct defaults
- Model paths are not hardcoded to Windows (Docker portability)
- CLI `--help` and `--version` work
- Argument parser is callable
- GPU backend detection and runtime flag behavior

---

## Adding Configuration Options

1. Add the attribute to the `Config` class in `config.py`:

   ```python
   class Config:
       # ... existing options ...
       MY_NEW_OPTION = True  # Description of what it does
   ```

2. If it should be settable via CLI, add an argument in `argument_parser.py`:

   ```python
   parser.add_argument('--my_flag', action='store_true',
                        help='Enable my new feature')
   ```

3. Map the CLI arg to config in `configuration_manager.py`:

   ```python
   def update_config_from_args(args):
       if hasattr(args, 'my_flag') and args.my_flag:
           config.MY_NEW_OPTION = True
   ```

4. Read the config value in your module:

   ```python
   from config import config
   if config.MY_NEW_OPTION:
       # do something
   ```

---

## Adding Optional Dependencies

Follow the try/except pattern with a module-level flag:

```python
try:
    import new_library
    NEW_LIB_AVAILABLE = True
except ImportError:
    NEW_LIB_AVAILABLE = False

# Usage
if NEW_LIB_AVAILABLE:
    result = new_library.fast_function(data)
else:
    result = fallback_function(data)
```

Add the dependency to `pyproject.toml` under the appropriate optional group:

```toml
[project.optional-dependencies]
my_feature = ["new_library>=1.0"]
```

---

## Backward Compatibility

When renaming a simple compatibility alias, add an entry to `_COMPAT_ALIASES` in `config.py`:

```python
_COMPAT_ALIASES = {
    # ... existing aliases ...
    "OLD_OPTION_NAME": config.NEW_OPTION_NAME,
}
```

This ensures old code referencing the previous name continues to work.

For split configuration renames, use computed compatibility behavior on `Config` when one legacy name now maps to multiple canonical runtime flags.

---

## GPU Operations

All GPU operations must follow the **try GPU → fallback CPU** pattern:

```python
try:
    if CUPY_AVAILABLE and config.USE_CUPY:
        result = gpu_operation(data)
    else:
        raise RuntimeError("GPU not available")
except (RuntimeError, MemoryError) as e:
    logger.debug("GPU fallback: %s", e)
    result = cpu_operation(data)
```

Never assume GPU availability. Always provide a CPU fallback.

---

## See Also

- [Architecture](architecture.md) — Module structure and design patterns
- [Configuration](configuration.md) — All existing options
- [CLI Reference](cli-reference.md) — Existing CLI flags

---

Back to [Documentation Index](index.md)
