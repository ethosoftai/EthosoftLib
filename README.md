# EthosoftLib

**EthosoftLib is a modular, multi-project Python SDK**, not a Mercan-only
wrapper. Product modules stay independent and are registered lazily in the
generic `ethosoftlib.core` registry. The root import never loads native engines.

## Mercan: high-level Python API

For published PyPI releases, install with the optional Hugging Face extra:

```bash
pip install 'ethosoftlib[hub]'
```

For the latest development version, install directly from GitHub:

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
Platform wheels carry prebuilt native libraries. Importing the SDK performs
no compilation or additional executable downloads.
See [Mercan Python guide](docs/MERCAN_PYTHON.md) for step-by-step setup,
CPU/CUDA options, ChatML, cache, native packaging and current limitations.

For agent frameworks, `model.generate(messages, ...)` and
`model.stream_generate(messages, ...)` accept complete system/user/assistant
message lists without inserting a system prompt or changing chat history.
This is the recommended integration boundary for independent agent roles.

EthosoftLib 0.8.0 uses MercanRuntime 0.4.0's generation, device and session API v1.
`device="auto"` selects usable CUDA or CPU; `device="cuda"` fails clearly
if the loaded library lacks CUDA or no CUDA device is available.
`model.info()` reports actual device placement.
The C++ engine owns chat formatting, sampling and the complete generation loop;
CLI and Python share the same engine. Cancelling or closing an incomplete
stream joins native work and releases its touched context before returning.
Successful chat turns retain a native context and reuse its KV token prefix.
`model.session()` creates independent conversations sharing weights;
`reset_history()` clears both history and cache. Failed or interrupted turns
discard their cache. Native CPU/CUDA wheels also bundle standalone NedoTokenizer.

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

## Separation of Powers agent pipeline (`ethosoftlib.sop`)

Yasama (planning), Yurutme (tool execution) and Yargi (answer) roles run as
separate model calls over any mix of libmercan, Ollama, llama.cpp or the
`mercan` CLI. Pure Python; speech output is optional:

```bash
pip install ethosoftlib              # pipeline + CLI
pip install 'ethosoftlib[sop-speech]'  # optional EMA Lightning voice output
ethosoftlib-sop --help               # or: python -m ethosoftlib.sop --help
```

```python
from ethosoftlib.sop import AgentPipeline, YasamaRuntime, YurutmeRuntime, YargiRuntime
from ethosoftlib.sop.runtime import LibMercanBackend, OllamaBackend
```

See [SOP CLI](docs/sop/CLI.md) and [SOP runtime](docs/sop/RUNTIME.md).

## Web search and pages (`ethosoftlib.web`)

Keyless web search, readable page text and downloads using only the standard
library. DuckDuckGo is the default; set `ETHOSOFT_SEARXNG_URL` to use SearXNG.
Hosts on loopback/private networks are refused unless `allow_private=True`.

```python
from ethosoftlib import web

for hit in web.search("Ankara hava durumu", max_results=3):
    print(hit.title, hit.url, hit.snippet)
page = web.open_url("https://tr.wikipedia.org/wiki/Ankara", max_chars=4000)
web.download("https://example.com/rapor.pdf", "indirilenler/")
```

## Local retrieval (`ethosoftlib.rag`)

BM25 search over local text with Turkish-aware terms (Turkish casing,
apostrophe suffixes, five-letter prefix stemming). Every hit carries its file
and starting line. No dependencies; pass an `embedder` for hybrid search.

```python
from ethosoftlib.rag import Index

index = Index()
index.add_directory("docs", pattern="*.md")
for hit in index.search("Mercan modeli nasıl indirilir?", k=3):
    print(hit.source, hit.line, hit.score, hit.text[:80])
index.save("docs.index.json")
```

With a Mercan model, `Index(embedder=model)` also finds chunks that share no
word with the query; BM25 hits still come first (see
[Mercan Python guide](docs/MERCAN_PYTHON.md#embeddings-and-retrieval)).

## Documents and OCR (`ethosoftlib.ocr`)

Reads PDF text layers exactly, page by page, and reports scanned pages instead
of returning them empty. `render_page` produces the PNG an OCR model will take;
no OCR model ships yet. Needs Poppler (`apt install poppler-utils`).

```python
from ethosoftlib import ocr

doc = ocr.read("rapor.pdf", allow_missing=True)
print(doc.text, doc.needs_ocr)        # needs_ocr: scanned page numbers
png = ocr.render_page("rapor.pdf", 2, dpi=300)
```

`rag.Index.add_file` and `add_directory` index PDF text layers too; hits carry
`metadata["page"]`.

## Math tool-use (`ethosoftlib.math`)

Verified math tool-use toolkit: text -> `<arac_cagrisi>bol(48, 2)</arac_cagrisi><arac_sonucu>24</arac_sonucu>`,
a tool-call executor and a calculator. Pure Python, needs only `sympy`:

```bash
pip install 'ethosoftlib[math]'
```

```python
import ethosoftlib.math as em
em.fill_results("<arac_cagrisi>bol(48, 2)</arac_cagrisi>")  # fills <arac_sonucu>24</arac_sonucu>
em.calculate("4 + 5")                                        # '9'
```

See [docs/MATH.md](docs/MATH.md) and the [function list](docs/MATH_PROTOCOL.md).

## Development

```bash
pip install -e '.[dev]'
python -m pytest -q
python -m build
```

Python unit tests do not verify real model inference; native end-to-end tests
require a built libmercan and a downloaded SFT model.

## NedoTokenizer: model-free exact Rust tokenizer

Use the same **NDSRF004** tokenizer as Mercan without loading an LLM:

```python
from ethosoftlib.nedo import Tokenizer

tokenizer = Tokenizer()
ids = tokenizer.encode("Merhaba, nasılsın?")
print(ids)
print(tokenizer.decode(ids))
print(tokenizer.vocab_size, tokenizer.vocab_sha256)
```

The Python adapter is available as `ethosoftlib.nedo` or
`ethosoftlib.mercan.NedoTokenizer`; the generic registry also offers
`providers.create("tokenizer", "nedo")`. A Rust native tokenizer library
is required at runtime. On Linux build a library with
`bash scripts/build_nedo_native_linux.sh`; the script creates the exact
Rust bridge, not an approximate Python reimplementation.
See [NedoTokenizer guide](docs/NEDO_TOKENIZER.md).

## SDK v0.4 developer tools

The Mercan adapter adds synchronous / asynchronous token streaming,
model metadata inspection, wall-time benchmarking and explicit context KV
reset. The generic EthosoftLib core remains dependency-free.

```python
from ethosoftlib.mercan import Model
with Model.from_pretrained("MercanAI/Mercan-0.8B-SFT") as model:
    print(model.info())
    for chunk in model.stream_chat("Merhaba!"):
        print(chunk, end="", flush=True)
    print(model.benchmark(runs=3, max_tokens=32))
```

See [Mercan Python SDK](docs/MERCAN_PYTHON.md) for async methods.

Architecture developers can validate and package signed-by-checksum
Mercan Plugin ABI v1 binaries without executing them:

```bash
python -m ethosoftlib.plugins scaffold myarchitecture ./myarchitecture
python -m ethosoftlib.plugins validate ./myarchitecture/mercan-plugin.json
python -m ethosoftlib.plugins pack ./myarchitecture/mercan-plugin.json ./myarchitecture.zip
```

See [Plugin SDK](docs/PLUGIN_SDK.md). Native Rust NedoTokenizer wheel
jobs now target Linux x86_64, Windows x64 and macOS Apple Silicon;
builds are separate from PyPI publication.

## Platform packaging status (0.4.0)

GitHub Actions provides separately tested combined Mercan+Nedo CPU native
wheels for Linux x86_64, Windows x64 and Apple Silicon macOS. These are
currently CI artifacts, not published PyPI binaries. Run
`python scripts/build_mercan_native_desktop.py` and
`python -m build --wheel` to produce an OS-native developer wheel.
See [Mercan Python SDK](docs/MERCAN_PYTHON.md) and
[Plugin SDK](docs/PLUGIN_SDK.md).

## Publishing to PyPI

`ethosoftlib` releases are prepared using GitHub Actions and PyPI Trusted
Publishing. The release workflow builds and validates a pure-Python package,
without collecting or storing long-lived API tokens.

See [the PyPI release guide](docs/PYPI_RELEASE.md) for first-time account
setup, safe dry runs, release tags and native wheel limitations.
