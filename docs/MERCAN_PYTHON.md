# Mercan Python SDK (optional EthosoftLib integration)

The high-level API is implemented only in `ethosoftlib.mercan`, not in
`ethosoftlib.core` or other product namespaces.

## Setup

```bash
pip install 'ethosoftlib[hub]'
```

The base EthosoftLib install has **no native dependencies**. The `hub` extra
adds `huggingface_hub` to download the published `model.mercan` artifact.
The Mercan C ABI is also required: it is not the same as merely installing
the Mercan CLI. Use a compatible shared libmercan via
`ETHOSOFT_MERCAN_LIBRARY=/path/to/libmercan.so`, or a platform wheel built
with a staged native payload.

```python
from ethosoftlib.mercan import Model

with Model.from_pretrained("MercanAI/Mercan-0.8B-SFT") as model:
    print(model.chat("Merhaba, nasılsın?", temperature=0.7, max_tokens=256))
    print(model.chat("Az önce ne demiştim?"))  # previous turns retained
```

### Configuration

`Model.from_pretrained` accepts a local `.mercan` file, or an HF
`owner/repo` ID; optional `filename`, `revision`, `cache_dir`,
`local_files_only` and `token` parameters forward to the Hugging Face Hub
file downloader. For repeatable builds, pin `revision` to a commit SHA.
There are no automatic downloads of executables or DLLs.

`device='auto'` deliberately starts on CPU, unless the user sets
`ETHOSOFT_MERCAN_GPU_LAYERS`; `device='cuda'` requests all layers on the
GPU, and requires an appropriately built native runtime.
`gpu_layers` overrides offload count.

`chat` accepts `temperature`, `top_k`, `top_p`, `repeat_penalty`,
`repeat_last_n`, `seed`, `max_tokens` and `save_history`.
Use `reset_history()` to clear previous turns.
ChatML role headers mirror Mercan CLI: `sistem`, `kullanici`, `asistan`;
assistant generation stops at EOS, pad, `im_start`, or `im_end`.

Each turn currently recreates the native inference context and replays
the transcript. Thus, chat history works, but across-turn KV cache reuse
and async streaming are *not* implemented.
Context length is bounded; excessive histories raise a useful error.

## Build Linux native library / wheel

```bash
# Linux builder with CMake, C++17 compiler, git and Rust/Cargo installed.
bash scripts/build_mercan_native_linux.sh
python -m pip install build
python -m build --wheel
```

The script checks out MercanRuntime at a fixed known commit, builds a shared
CPU `libmercan.so`, and stages it into `ethosoftlib.mercan.lib`.
A staged wheel is **platform-tagged**, never `py3-none-any`.
This is a developer build pipeline, NOT an audited manylinux PyPI release.
Inspect bundled dependencies using `ldd` and repair for manylinux using
auditwheel before distributing across Linux systems.

A fresh environment running the standard unit tests does not require native
Mercan. The `Mercan native Linux model smoke` GitHub Actions workflow also
builds `libmercan`, downloads the public `MercanAI/Mercan-0.8B-SFT` artifact,
and runs a real 16-token chat when executed successfully.

## Reuse original model-free NedoTokenizer

```python
from ethosoftlib.mercan import NedoTokenizer

tokenizer = NedoTokenizer()  # Rust NDSRF004 bridge, no LLM required
tokens = tokenizer.encode("Merhaba")
print(tokenizer.decode(tokens))
```

The standalone Rust library must be built or installed separately; see
[docs/NEDO_TOKENIZER.md](NEDO_TOKENIZER.md).

## SDK streaming and async (v0.4)

```python
import asyncio
from ethosoftlib.mercan import Model

with Model.from_pretrained("MercanAI/Mercan-0.8B-SFT") as model:
    for fragment in model.stream_chat("Kısa bir hikâye anlat"):
        print(fragment, end="", flush=True)
    print(model.info())
    print(model.benchmark(runs=3, max_tokens=32))

async def demo():
    with Model.from_pretrained("MercanAI/Mercan-0.8B-SFT") as model:
        response = await model.async_chat("Merhaba")
        print(response)
        async for fragment in model.astream_chat("Nasılsın?"):
            print(fragment, end="", flush=True)

asyncio.run(demo())
```

The stream uses an incremental UTF-8 decoder, so Turkish characters split
between native token pieces are emitted intact. Incomplete streams do
not append a partial assistant response to conversation history.
All async generation runs on a worker thread so the event loop is not
blocked by native inference. Do not close a model while a generation is
active; serialize concurrent calls per model.

`model.info()` provides the C ABI runtime version, architecture, tokenizer,
vocabulary size, model path/size, special-token IDs and configured context.
`model.benchmark()` measures wall-clock generation and estimates token rate
by re-tokenizing output; it is not a pure decoder tokens/s measurement.

Low-level `MercanContext.tokens_used` and `tokens_remaining` show
the conservatively accounted native decode budget. `context.reset()`
destroys the old native context and creates a fresh context, clearing all
KV state in a backend-independent manner. **It does not provide cross-turn
KV reuse or prefix sharing.** High-level chat still replays transcript
tokens into a new native context for each turn. Cross-turn KV reuse requires
token-prefix parity and model-backed cache correctness testing.

See [Plugin SDK](PLUGIN_SDK.md) for architecture extensions.

## Desktop-native binary build (CPU, combined Mercan + Nedo)

The automated GitHub Actions workflow in
`.github/workflows/mercan-native-desktop.yml` builds a platform wheel with
**both** the native Mercan C++ inference library and the exact Rust NDSRF004
tokenizer on Linux x86_64, Windows x64, and macOS Apple Silicon. It installs
the completed wheel into a clean Python environment and verifies library
loading, native runtime version, and Turkish Unicode tokenizer round-trip.

Local prerequisites: Rust, C/C++ compiler, Git, CMake and Python 3.10+.
The developer build command is:

```bash
python scripts/build_mercan_native_desktop.py
python -m pip install build
python -m build --wheel
python scripts/verify_mercan_native_wheel.py
```

Native wheels are tagged `py3-none-<platform>` since the underlying Rust/C++
shared libraries use `ctypes` and do not depend on the CPython extension ABI.
These GitHub Actions artifacts are **not** yet PyPI releases. Linux wheels
must be manylinux-audited/repaired before official publication. GPU-enabled
CUDA and Metal builds need separate distribution verification. Device auto
currently remains CPU-first by design.

The Python base remains free of mandatory dependencies and does not compile
or download executable native code during `pip install` for the pure wheel.
