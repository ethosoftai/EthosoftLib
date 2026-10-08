# EthosoftLib

**EthosoftLib is a multi-project, extensible Python SDK for Ethosoft technologies.**

It is **not** a Mercan-only library. Integrations live in independent namespaces,
with a general-purpose lazy provider registry in `ethosoftlib.core`. New services,
education tools, data utilities, or native runtimes may be added without changing
or depending on Mercan.

## Status

Early SDK foundation (0.1.0). No PyPI release has been published yet. This
repository implements the shared SDK core and the first *optional* integration:
a low-level MercanRuntime C ABI wrapper.

## Install from source

```bash
git clone https://github.com/ethosoftai/EthosoftLib.git
cd EthosoftLib
python -m pip install -e .
```

Nothing native is downloaded or compiled by this package.

## Discover any registered product

```python
from ethosoftlib import providers

for item in providers.list():
    print(item.category, item.name, item.description)

# The generic SDK does not load libmercan on import or discovery.
# Other projects can register in any domain:
providers.register("storage", "example", "my_package:StorageClient")
# The registered provider is imported only when explicitly created.
```

## MercanRuntime integration (optional)

To use Mercan, first compile/install a **shared** native Mercan library from
[MercanRuntime](https://github.com/Ahmet2001/MercanRuntime) using
`MERCAN_BUILD_SHARED=ON`. Installing the Mercan CLI does not necessarily
install `libmercan.so` for Python usage.

```python
from ethosoftlib.mercan import MercanRuntime

with MercanRuntime(library_path="/absolute/path/to/libmercan.so") as runtime:
    print(runtime.version)
    with runtime.load_model("./model.mercan", gpu_layers=0) as model:
        print(model.architecture, model.tokenizer, model.vocab_size)
        print(model.tokenize("Merhaba"))
        # Supply your model's own chat template; no template is assumed:
        print(model.generate("Merhaba", max_new_tokens=32))
```

Alternatively set `ETHOSOFT_MERCAN_LIBRARY=/path/to/libmercan.so` and use
`MercanRuntime()`. The adapter uses `ctypes` and the stable C API; it does
not vendor Mercan's model classes, native tokenizer, or execution graph.
`generate` is a basic greedy helper, not yet a high-level chat API.

```python
from ethosoftlib import providers

runtime = providers.create(
    "inference", "mercan", library_path="/absolute/path/to/libmercan.so"
)
try:
    print(runtime.version)
finally:
    runtime.close()
```

## Project structure

```text
src/ethosoftlib/
  core/          # Generic errors + lazy provider registry
  mercan/        # Optional native runtime bridge (not a dependency of core)
tests/           # Unit tests; no native runtime required for these
docs/            # Design and extension guidelines
```

## Testing

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m build
```

These checks validate SDK behavior, packaging and ctypes declarations.
They do not replace native Mercan model/parity tests.

## Extending EthosoftLib

Each future integration should own its dependency boundary and module.
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
