# Mercan Python SDK (optional EthosoftLib integration)

The high-level API is implemented only in `ethosoftlib.mercan`, not in
`ethosoftlib.core` or other product namespaces.

## Setup

```bash
pip install 'ethosoftlib[hub]'
```

The pure Python EthosoftLib wheel has no bundled native payload. Combined native platform
wheels include Mercan and standalone NedoTokenizer. The `hub` extra
adds `huggingface_hub` to download the published `model.mercan` artifact.
The Mercan C ABI is also required: it is not the same as merely installing
the Mercan CLI. Use a compatible shared libmercan via
`ETHOSOFT_MERCAN_LIBRARY=/path/to/libmercan.so`, or a platform wheel built
with a staged native payload.

EthosoftLib 0.8.0 requires MercanRuntime 0.4.0+ with generation, device and session API v1 for
`Model.from_pretrained`, `generate`, `chat` and streaming. Older libraries can still serve the
low-level bindings, but high-level generation reports a clear upgrade error.
There is no automatic fallback to a Python token generation loop.

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

`device='auto'` uses CUDA when the loaded native library supports it and a
usable CUDA device is visible, otherwise CPU. `device='cpu'` forces CPU.
`device='cuda'` requires both a CUDA build (`GGML_CUDA=ON`) and a usable
device; missing support raises an explicit error before model download.
Allocation or model-load failures also propagate; they do not retry on CPU.
`gpu_layers=-1` offloads all supported layers, a positive count requests
partial offload, and `0` forces CPU with `auto`. Combining `cuda` with `0`,
or `cpu` with a nonzero count, is an error. Generic plugin architectures
currently offload the entire model for any nonzero count.
`ETHOSOFT_MERCAN_GPU_LAYERS` applies only to `auto` without explicit
`gpu_layers`; a nonzero value is a strict CUDA request.

```python
from ethosoftlib.mercan import MercanRuntime, Model

with MercanRuntime(library_path="/path/to/libmercan.so") as runtime:
    print(runtime.capabilities())  # cuda_compiled, cuda_available, backends, devices

with Model.from_pretrained("/models/model.mercan", device="auto") as model:
    print(model.info()["device"])     # cpu or cuda, from loaded weight buffers
    print(model.info()["placement"])  # cpu/gpu/mixed and actual device names
```

CUDA offload may retain CPU weights or operations. `placement` describes
weight ownership; it does not claim that every operation runs on the GPU.
An explicit `library_path` or library environment variable is authoritative.
A CPU wheel remains a CPU build even on a machine with an NVIDIA GPU.

`chat` accepts `temperature`, `top_k`, `top_p`, `repeat_penalty`,
`repeat_last_n`, `seed`, `max_tokens` and `save_history`.
Use `reset_history()` to clear previous turns.
ChatML role headers mirror Mercan CLI: `sistem`, `kullanici`, `asistan`;
assistant generation stops at EOS, pad, `im_start`, or `im_end`.

`chat`, `stream_chat` and their async variants retain a native context and reuse
the exact cached token prefix across committed turns. `reset_history()` clears
both history and KV state and releases the retained context. Interrupted or
failed turns preserve Python history and discard their touched cache, so the
next request safely rebuilds it. Context length is bounded; excessive history
plus output budget raises an error without silently deleting messages.
`save_history=False` uses a temporary request and leaves the saved chat/cache
untouched. Stateless `generate` / `stream_generate` also use temporary contexts.

### Independent conversations over one model

```python
with Model.from_pretrained("/models/model.mercan") as model:
    with model.session(system_prompt="Türkçe kısa cevap ver.") as a, model.session() as b:
        print(a.chat("Merhaba"))
        print(a.chat("Az önce ne demiştim?"))
        print(a.info()["kv_cache"])  # reused_tokens, prefill_tokens, context_allocations
        assert b.history == ()
        a.reset_history()
```

Sessions share loaded weights and own separate histories and KV caches. Calls
sharing one model are serialized; close an active stream before issuing another
request on that model. Closing a session releases its cache; closing the parent
model closes all its sessions. A session cannot outlive its parent model.
Backends that cannot safely truncate cached suffixes rebuild the context.
`context_allocations` is a lifetime counter and is not reset by clearing history.

### Application-owned messages (native engine in 0.6.0)

Use `generate` / `stream_generate` when your application owns the transcript,
or when one model serves multiple independent agent roles:

```python
from contextlib import closing
from ethosoftlib.mercan import Model

with Model.from_pretrained("/models/model.mercan") as model:
    messages = [
        {"role": "system", "content": "Türkçe, kısa cevap ver."},
        {"role": "user", "content": "Merhaba"},
    ]
    print(model.generate(messages, temperature=0, max_tokens=32))
    with closing(model.stream_generate(messages, max_tokens=32)) as stream:
        for fragment in stream:
            print(fragment, end="", flush=True)
```

These methods neither read nor modify `model.history` and never insert a
default system prompt. They share the C++ generation engine with `chat` and
the Mercan CLI: chat formatting, prefill, sampling, stop detection and token
decode all execute in `libmercan`. Python only marshals messages and decodes
the streamed UTF-8 bytes. `model.info()['generation_engine']` is `native-v1`.
`stream_generate` accepts the same sampling options as `chat`, plus an optional
`threading.Event` named `stop_event`. Cancellation is checked between native
decode calls. Close partially consumed iterators to release their context.
`role_system`, `role_user`, `role_assistant` and `threads_batch` can be supplied
when loading a model. `plugins` are loaded before model loading; `use_mmap`
and `check_tensors` configure loading.

The native repetition policy follows the CLI: penalize each distinct token
once within the recent generated-output window, exclude prompt tokens, and
exempt stop tokens. This replaces the older Python high-level sampler's
per-occurrence/prompt penalty. The standalone low-level `sampling.py` helper
retains its historical behavior for custom decode experiments; model-level
generation does not call it. Seed values must fit an unsigned 32-bit integer.
Use matching explicit messages/headers/options to compare CLI and SDK output;
their application-level default system prompts are independent.

Native builds exposing ABI queries are checked for ABI version 1 and exact
parameter struct sizes before struct-returning/passing functions are used.
Historical runtime 0.1.2 builds without queries are still supported, but their
struct layout cannot be checked automatically. Other unversioned builds fail
with an actionable compatibility error.

## Build Linux native library / wheel

```bash
# Linux builder with CMake, C++17 compiler, git and Rust/Cargo installed.
bash scripts/build_mercan_native_linux.sh
python -m pip install build
python -m build --wheel
```

The script checks out MercanRuntime at a fixed known commit, builds a shared
CPU `libmercan.so`, builds the original Rust NedoTokenizer cdylib, and stages
both libraries plus Nedo notices in the package. `Tokenizer()` then works
without a model, an environment variable or a separate native-library install.
A staged wheel is **platform-tagged**, never `py3-none-any`.
This is a developer build pipeline, NOT an audited manylinux PyPI release.
Inspect bundled dependencies using `ldd` and repair for manylinux using
auditwheel before distributing across Linux systems.

To use an existing local MercanRuntime checkout without changing its Git
revision or resetting its prepared llama.cpp dependency:

```bash
MERCAN_USE_LOCAL_SOURCE=1 \
MERCAN_SRC=/absolute/path/to/MercanRuntime \
MERCAN_JOBS=1 bash scripts/build_mercan_native_linux.sh
```

Local mode requires an already prepared patched llama.cpp under
`build/_deps/llama-src`, or at `MERCAN_LLAMA_DIR`. It builds in
`build/ethosoftlib-shared` (`MERCAN_SHARED_BUILD_DIR` overrides this) and stages
the resulting `.so` in the Python package. It does not run `prepare_llama.sh`.
`MERCAN_CUDA=ON` requests a CUDA build in a separate
`build/ethosoftlib-shared-cuda` directory and stages that build instead of
the CPU payload. It requires the NVIDIA CUDA Toolkit (`nvcc`, headers and
cuBLAS). Point `CUDACXX` and `CUDAToolkit_ROOT` to a nonstandard toolkit.
`MERCAN_CUDA_ARCHITECTURES` can set CMake's target architectures (for example
`89` for a local Ada GPU); choose the required architectures for distribution.

```bash
MERCAN_CUDA=ON bash scripts/build_mercan_native_linux.sh
# Verify compiled support separately from device availability:
.native-work/MercanRuntime/build/ethosoftlib-shared-cuda/mercan devices
```

CUDA wheels additionally require compatible CUDA runtime/cuBLAS libraries
and an NVIDIA driver on the target system. The build script does not bundle
or install those dependencies. Keep CPU and CUDA wheel outputs separate;
they have the same package name/version and are alternative builds.

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
blocked by native inference. Output queues are bounded, and cancelling
`async_chat` or closing a stream cancels and joins native work before releasing
the request/context. Cancellation cannot interrupt an in-flight decode.
Do not close a model while a generation is
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
CUDA and Metal builds need separate distribution verification. Automatic
selection currently covers CUDA and CPU; other backends remain available
through the low-level native API's explicit offload parameters.

The Python base remains free of mandatory dependencies and does not compile
or download executable native code during `pip install` for the pure wheel.
