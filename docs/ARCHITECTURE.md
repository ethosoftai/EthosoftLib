# EthosoftLib architecture

## Scope

EthosoftLib is a **general-purpose SDK for multiple independent Ethosoft
products**. No single integration (including Mercan) owns the root namespace,
shared errors, dependency list, release cadence, or registry.

```
ethosoftlib
├── core/            domain-neutral provider registry and errors
├── mercan/          optional MercanRuntime C-ABI binding
└── <future>/        independent services, frameworks or products
```

The root package has **zero mandatory runtime dependencies**. Providers register
using the generic `(category, name, import_path)` convention and load lazily.
Provider categories can be `inference`, `education`, `storage`, `data`, or any
independent domain defined by an integration. There is no global base-class that
requires every product to implement `model`, `tokenize`, or `generate`.

## Adding another integration

For example, a future package may live at `ethosoftlib.education`:

```python
from ethosoftlib import providers

providers.register(
    "education", "curriculum",
    "ethosoftlib.education:CurriculumClient",
    description="Ethosoft curriculum APIs",
)
client = providers.create("education", "curriculum", api_key="...")
```

The example is a **convention**, not an implemented curriculum integration.
For externally distributed plugins, third-party packages may call `register`
during their own initialization. Automatic entry-point discovery is not yet
implemented and should require an explicit trust decision before importing code.

## Mercan adapter boundary

```
ethosoftlib.mercan
      |  ctypes / stable C ABI
      v
libmercan.so / libmercan.dylib / mercan.dll
      |
      v
MercanRuntime native backends and plugins
```

The Python package does not embed native Mercan binaries, llama.cpp, CUDA, or
model files. The native library is loaded only when `MercanRuntime()` is
constructed. Set `ETHOSOFT_MERCAN_LIBRARY` to the shared library path or pass
`library_path=...`.

The native ABI used by this first version matches
`Ahmet2001/MercanRuntime/runtime/libmercan/include/mercan.h`.
The local model loader supports native plugin dispatch; EthosoftLib does not
reimplement architecture discovery, tokenizer algorithms, or ggml graph code.

## Lifecycle and compatibility

Create runtime → load model → create context. Close in the inverse order;
context managers are recommended. Multi-turn KV policies, HF downloads, and
chat template formatting are **not** in this initial low-level binding.
`generate()` is a minimal greedy helper for an already formatted prompt.

Before publishing PyPI wheels, build a compatible shared `libmercan` for each
platform and add real-model end-to-end regression testing. Tests without a
native library are SDK/ABI smoke tests only and **do not prove inference parity**.

## Future products

Future Ethosoft projects must have their own subpackages, optional extras,
public APIs, and independent tests. Prefer a stable integration boundary to
imports from the Mercan provider. The `core` registry cannot gain inference-
specific assumptions.

## Independent NedoTokenizer

`ethosoftlib.nedo` wraps the exact Rust NDSRF004 tokenizer via a standalone
shared library. It has no Mercan model dependency. It is lazily registered
as `tokenizer/nedo` in the generic provider registry. Because the native
Rust library is optional, merely importing EthosoftLib loads no native code.
