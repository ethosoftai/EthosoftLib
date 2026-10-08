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

A normal `pip install ethosoftlib` installs the Python adapter but **does
not** include any native library. Compile the original Rust bridge:

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
publication. Windows/macOS binary packages are not provided yet.

This is a separate optional integration under `ethosoftlib.nedo` so future
products can use NedoTokenizer without MercanRuntime. No mandatory dependency
is added to `ethosoftlib.core`.
