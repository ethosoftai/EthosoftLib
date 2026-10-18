# Original NedoTokenizer in EthosoftLib

The standalone **NDSRF004** tokenizer is independent of Mercan inference.
It delegates tokenization to the original Rust morphology + surface vocabulary
implementation; this is not a Python imitation.

## Python usage

```python
from ethosoftlib.nedo import Tokenizer

tok = Tokenizer()
print(tok.vocab_size)       # 32000
print(tok.vocab_sha256)     # pinned SHA-256 identity
ids = tok.encode("Merhaba, nasılsın?")
assert tok.decode(ids) == "Merhaba, nasılsın?"
```

Also available via `from ethosoftlib.mercan import NedoTokenizer` and the
generic provider registry `providers.create("tokenizer", "nedo")`.

`encode(str|bytes, add_special_tokens=False)` returns NDSRF004 IDs;
`decode(ids)` decodes UTF-8 strictly by default; `decode_bytes(ids)`
preserves invalid UTF-8. BOS=1, EOS=2, PAD=0; adding BOS/EOS is optional.

## Installation/build

A platform-native EthosoftLib wheel includes the original Rust shared library
and its vocabulary/morphology data. With such a wheel, `Tokenizer()` works
without Mercan model loading, CUDA initialization or environment variables.
The Linux CPU/CUDA build script and desktop native workflow both stage it,
together with NedoTokenizer and Zemberek notices and the Apache 2.0 license.

The pure Python wheel contains only the adapter. When building from source,
compile the Rust bridge or use the combined Mercan build:

```bash
bash scripts/build_nedo_native_linux.sh
pip install -e .
python -c "from ethosoftlib.nedo import Tokenizer; print(Tokenizer().encode('Merhaba'))"
```

Alternatively set `ETHOSOFT_NEDO_LIBRARY=/absolute/path/to/libnedo004_ffi.so`
or pass `library_path`. The library contains the exact 32k NDSRF004 vocab,
morphological model, and SHA validation, no Mercan model download required.

The developer script stages a Linux x86_64 shared library for creating a
platform wheel. Audit/repair Linux dependencies for manylinux before PyPI
publication. The desktop build workflow also prepares Windows/macOS payloads;
local Linux validation does not establish that those CI artifacts have passed.

This is a separate optional integration under `ethosoftlib.nedo` so future
products can use NedoTokenizer without MercanRuntime. No mandatory dependency
is added to `ethosoftlib.core`.

## Experimental cross-platform native build matrix

The workflow `.github/workflows/nedo-native-matrix.yml` compiles the **same
original Rust NDSRF004** as a native shared library on Linux x86_64,
Windows x64 and macOS Apple Silicon runners. It stages the library inside
`ethosoftlib.nedo.lib`, creates platform-tagged wheels, and tests a fresh
wheel installation. Workflow success is not equivalent to a PyPI release
or verified installation on all host configurations.
