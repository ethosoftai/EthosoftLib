# EthosoftLib

**EthosoftLib is a modular, multi-project Python SDK**, not a Mercan-only
wrapper. Product modules stay independent and are registered lazily in the
generic `ethosoftlib.core` registry. The root import never loads native engines.

## Mercan: high-level Python API

Install from GitHub with the optional Hugging Face extra (not yet on PyPI):

```bash
pip install 'ethosoftlib[hub] @ git+https://github.com/ethosoftai/EthosoftLib.git'
```

The public Mercan model is distributed as `model.mercan` in
`MercanAI/Mercan-0.8B-SFT`. The Hugging Face Hub downloader caches this file.

```python
from ethosoftlib.mercan import Model

with Model.from_pretrained("MercanAI/Mercan-0.8B-SFT") as model:
    answer = model.chat("Merhaba, nasılsın?", temperature=0.7, max_tokens=256)
    print(answer)
    print(model.chat("Bir önceki sorum neydi?"))
```

A **compatible libmercan shared library** is required to execute this code.
On Linux, build and stage one using `bash scripts/build_mercan_native_linux.sh`,
or set `ETHOSOFT_MERCAN_LIBRARY=/absolute/path/to/libmercan.so`.
EthosoftLib does not automatically download or install executable native code.
See [Mercan Python guide](docs/MERCAN_PYTHON.md) for step-by-step setup,
CPU/CUDA options, ChatML, cache, native packaging and current limitations.

## Provider-neutral SDK

```python
from ethosoftlib import providers

print(providers.list())   # metadata only; no integration dependencies loaded
providers.register("storage", "example", "my_package:StorageClient")
# Inference and storage are unrelated categories.
```

Add other integrations under their own namespaces:
`ethosoftlib.education`, `ethosoftlib.data`, etc. See
[architecture guidelines](docs/ARCHITECTURE.md).

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Python unit tests do not verify real model inference; native end-to-end tests
require a built libmercan and a downloaded SFT model.
