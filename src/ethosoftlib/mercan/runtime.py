"""Mercan-specific Python convenience layer; EthosoftLib core remains generic."""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path
from threading import RLock
from typing import Sequence

from .native import bind_api, load_library, native_error, MercanError

# Mercan's native backend uses process-global initialization.
_BACKEND_LOCK = RLock()
_BACKEND_USERS: dict[int, int] = {}


class MercanRuntime:
    """A handle to an existing libmercan shared library.

    Use as a context manager. Close dependent models before closing the runtime.
    This adapter intentionally does NOT install, compile or download native code.
    """

    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        self._lib = bind_api(load_library(library_path))
        self._key = int(self._lib._handle)
        self._closed = False
        self._models = 0
        with _BACKEND_LOCK:
            if _BACKEND_USERS.get(self._key, 0) == 0:
                self._lib.mercan_backend_init()
            _BACKEND_USERS[self._key] = _BACKEND_USERS.get(self._key, 0) + 1

    @property
    def version(self) -> str:
        self._check()
        return self._lib.mercan_version().decode("utf-8", "replace")

    def _check(self) -> None:
        if self._closed:
            raise MercanError("MercanRuntime has been closed")

    def load_model(
        self, path: str | os.PathLike[str], *, gpu_layers: int = 0,
        use_mmap: bool = True, check_tensors: bool = False,
    ) -> MercanModel:
        self._check()
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"Mercan model not found: {model_path}")
        p = self._lib.mercan_model_default_params()
        p.n_gpu_layers = gpu_layers
        p.use_mmap = use_mmap
        p.check_tensors = check_tensors
        handle = self._lib.mercan_model_load(os.fsencode(model_path), p)
        if not handle:
            raise native_error(self._lib, f"Could not load {model_path}")
        self._models += 1
        return MercanModel(self, handle)

    def load_plugin(self, library_path: str | os.PathLike[str]) -> bool:
        """Explicitly load a trusted native plugin (returns False if already loaded).

        Mercan retains native plugins for process lifetime. Only load binaries
        whose provenance you trust; native code has the process's privileges.
        """
        self._check()
        if self._models:
            raise MercanError("Load architecture plugins before loading models")
        path = Path(library_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix not in (".so", ".dll", ".dylib"):
            raise ValueError("Expected a native Mercan plugin shared library")
        from .native import _bind
        _bind(self._lib, "mercan_plugin_load_v1", C.c_int, C.c_char_p)
        _bind(self._lib, "mercan_plugin_last_error_v1", C.c_char_p)
        result = self._lib.mercan_plugin_load_v1(os.fsencode(path))
        if result not in (0, 1):
            raw = self._lib.mercan_plugin_last_error_v1()
            message = raw.decode("utf-8", "replace") if raw else "Unknown native plugin error"
            raise MercanError(f"Mercan plugin load failed: {message}")
        return result == 0

    def load_plugin_from_manifest(self, manifest_path: str | os.PathLike[str]) -> bool:
        """Verify ABI and SHA256 for the current platform before explicit loading."""
        import platform
        from ethosoftlib.plugins import read_manifest
        manifest = read_manifest(manifest_path)
        system, machine = platform.system().lower(), platform.machine().lower()
        if system == "linux" and machine in ("x86_64", "amd64"):
            target = "linux_x86_64"
        elif system == "windows" and machine in ("x86_64", "amd64"):
            target = "windows_amd64"
        elif system == "darwin" and machine in ("arm64", "aarch64"):
            target = "macos_arm64"
        elif system == "darwin" and machine in ("x86_64", "amd64"):
            target = "macos_x86_64"
        else:
            raise MercanError(f"No Mercan plugin target for {system}/{machine}")
        if target not in manifest["binaries"]:
            raise MercanError(f"Plugin has no native binary for {target}")
        base = Path(manifest_path).resolve()
        if base.is_dir():
            base /= "mercan-plugin.json"
        return self.load_plugin(base.parent / manifest["binaries"][target])

    def list_plugins(self) -> tuple[tuple[str, str, str], ...]:
        """Read the name/version/path for registered native plugins."""
        self._check()
        from .native import _bind
        _bind(self._lib, "mercan_plugin_count_v1", C.c_size_t)
        for name in ("mercan_plugin_name_v1", "mercan_plugin_version_v1", "mercan_plugin_path_v1"):
            _bind(self._lib, name, C.c_char_p, C.c_size_t)
        result = []
        for index in range(self._lib.mercan_plugin_count_v1()):
            result.append(tuple(
                (getattr(self._lib, name)(index) or b"").decode("utf-8", "replace")
                for name in ("mercan_plugin_name_v1", "mercan_plugin_version_v1", "mercan_plugin_path_v1")
            ))
        return tuple(result)

    def close(self) -> None:
        if self._closed:
            return
        if self._models:
            raise MercanError("Close all MercanModel objects before closing the runtime")
        with _BACKEND_LOCK:
            remaining = _BACKEND_USERS[self._key] - 1
            if remaining == 0:
                self._lib.mercan_backend_free()
                del _BACKEND_USERS[self._key]
            else:
                _BACKEND_USERS[self._key] = remaining
        self._closed = True

    def __enter__(self) -> MercanRuntime:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class MercanModel:
    def __init__(self, runtime: MercanRuntime, handle: int) -> None:
        self._runtime = runtime
        self._handle = handle
        self._contexts = 0

    def _check(self) -> None:
        self._runtime._check()
        if not self._handle:
            raise MercanError("MercanModel has been closed")

    @property
    def architecture(self) -> str:
        self._check()
        return (self._runtime._lib.mercan_model_architecture(self._handle) or b"").decode()

    @property
    def tokenizer(self) -> str:
        self._check()
        return (self._runtime._lib.mercan_model_tokenizer(self._handle) or b"").decode()

    @property
    def vocab_size(self) -> int:
        self._check()
        return int(self._runtime._lib.mercan_vocab_size(self._handle))

    @property
    def eos_token(self) -> int:
        self._check()
        return int(self._runtime._lib.mercan_eos_token(self._handle))

    @property
    def pad_token(self) -> int:
        self._check()
        return int(self._runtime._lib.mercan_pad_token(self._handle))

    def tokenize(
        self, text: str, *, add_special: bool = False, parse_special: bool = True,
    ) -> list[int]:
        self._check()
        raw = text.encode("utf-8")
        if not raw:
            return []
        fn = self._runtime._lib.mercan_tokenize
        n = fn(self._handle, raw, len(raw), add_special, parse_special, None, 0)
        if n == 0:
            raise native_error(self._runtime._lib, "Tokenization failed")
        capacity = abs(n)
        for _ in range(3):
            out = (C.c_int32 * capacity)()
            count = fn(self._handle, raw, len(raw), add_special, parse_special, out, capacity)
            if count >= 0:
                return list(out[:count])
            capacity = -count
        raise native_error(self._runtime._lib, "Token buffer sizing failed")

    def token_to_bytes(self, token: int, *, special: bool = False) -> bytes:
        self._check()
        capacity = 64
        for _ in range(3):
            output = C.create_string_buffer(capacity)
            count = self._runtime._lib.mercan_token_to_piece(
                self._handle, token, output, capacity, special,
            )
            if 0 <= count <= capacity:
                return output.raw[:count]
            capacity = -count if count < 0 else count + 1
        raise native_error(self._runtime._lib, "Token piece buffer sizing failed")

    def context(
        self, *, n_ctx: int | None = None, n_batch: int | None = None,
        threads: int | None = None,
    ) -> MercanContext:
        self._check()
        p = self._runtime._lib.mercan_context_default_params()
        if n_ctx is not None:
            p.n_ctx = n_ctx
        if n_batch is not None:
            p.n_batch = n_batch
        if threads is not None:
            p.n_threads = threads
            p.n_threads_batch = threads
        if not p.n_ctx or not p.n_batch:
            raise ValueError("n_ctx and n_batch must be positive")
        handle = self._runtime._lib.mercan_context_create(self._handle, p)
        if not handle:
            raise native_error(self._runtime._lib, "Could not create Mercan context")
        self._contexts += 1
        return MercanContext(self, handle, int(p.n_batch))

    def generate(
        self, prompt: str, *, max_new_tokens: int = 128,
        n_ctx: int | None = None,
    ) -> str:
        """Basic deterministic greedy generation from an already formatted prompt.

        Callers must supply the correct model chat template where applicable.
        Sampling, chat history management, streaming, and HF downloads are
        intentionally separate future conveniences, not generic SDK primitives.
        """
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        tokens = self.tokenize(prompt, parse_special=True)
        if not tokens:
            raise ValueError("Prompt did not yield any tokens")
        output = bytearray()
        with self.context(n_ctx=n_ctx) as context:
            if len(tokens) + max_new_tokens > context.size:
                raise ValueError("Prompt + max_new_tokens exceeds context size")
            context.decode(tokens)
            for idx in range(max_new_tokens):
                token = context.greedy_token()
                if token == self.eos_token or token == self.pad_token:
                    break
                output.extend(self.token_to_bytes(token))
                if idx + 1 < max_new_tokens:
                    context.decode([token])
        return output.decode("utf-8", "replace")

    def close(self) -> None:
        if not self._handle:
            return
        if self._contexts:
            raise MercanError("Close all MercanContext objects before closing the model")
        self._runtime._lib.mercan_model_free(self._handle)
        self._handle = None
        self._runtime._models -= 1

    def __enter__(self) -> MercanModel:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class MercanContext:
    def __init__(self, model: MercanModel, handle: int, batch_size: int) -> None:
        self._model = model
        self._handle = handle
        self._batch_size = batch_size

    def _check(self) -> None:
        self._model._check()
        if not self._handle:
            raise MercanError("MercanContext has been closed")

    @property
    def size(self) -> int:
        self._check()
        return int(self._model._runtime._lib.mercan_context_size(self._handle))

    def decode(self, tokens: Sequence[int]) -> None:
        self._check()
        if not tokens:
            raise ValueError("decode expects at least one token")
        fn = self._model._runtime._lib.mercan_decode
        for start in range(0, len(tokens), self._batch_size):
            chunk = tokens[start:start + self._batch_size]
            batch = (C.c_int32 * len(chunk))(*chunk)
            code = fn(self._handle, batch, len(chunk))
            if code != 0:
                raise native_error(self._model._runtime._lib, f"mercan_decode returned {code}")

    def greedy_token(self) -> int:
        self._check()
        fn = self._model._runtime._lib.mercan_logits
        logits = fn(self._handle)
        vocab_size = self._model.vocab_size
        if not logits or vocab_size <= 0:
            raise native_error(self._model._runtime._lib, "Logits unavailable")
        return max(range(vocab_size), key=lambda index: logits[index])

    def sample_token(self, *, temperature: float = 0.7,
                     top_k: int = 40, top_p: float = 0.9,
                     repeat_penalty: float = 1.15,
                     recent_tokens: Sequence[int] = (), rng=None) -> int:
        """Sample native logits before the next decode call."""
        self._check()
        from .sampling import sample_token
        logits = self._model._runtime._lib.mercan_logits(self._handle)
        if not logits:
            raise native_error(self._model._runtime._lib, "Logits unavailable")
        return sample_token(logits, self._model.vocab_size, temperature=temperature,
                            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
                            recent_tokens=recent_tokens, rng=rng)

    def close(self) -> None:
        if self._handle:
            self._model._runtime._lib.mercan_context_free(self._handle)
            self._handle = None
            self._model._contexts -= 1

    def __enter__(self) -> MercanContext:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
