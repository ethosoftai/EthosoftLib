"""High-level Mercan model wrapper. Importing it does not load native binaries."""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
import asyncio
import codecs
from pathlib import Path
import os
import math
from concurrent.futures import TimeoutError as FutureTimeout
from threading import Event
from time import perf_counter
from threading import RLock

from .chat import DEFAULT_SYSTEM_PROMPT, format_chat
from .hub import resolve_model_file
from .native import MercanError
from .runtime import MercanRuntime, MercanModel


class Model:
    """High-level, Mercan-specific API backed by a local native libmercan.

    Chat retains its native context and reuses the verified token prefix across
    turns. Independent sessions share weights and own separate KV caches.
    Always close the model when done.
    """

    NATIVE_GENERATION_API_VERSION = 1
    DEVICE_API_VERSION = 1
    NATIVE_SESSION_API_VERSION = 1

    def __init__(
        self, runtime: MercanRuntime, model: MercanModel, model_path: Path, *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        n_ctx: int | None = None, n_batch: int | None = None,
        threads: int | None = None,
        threads_batch: int | None = None,
        role_system: str = "sistem", role_user: str = "kullanici",
        role_assistant: str = "asistan",
    ) -> None:
        self._runtime = runtime
        self._model = model
        self.model_path = model_path
        self.system_prompt = system_prompt
        self._context_options = {
            "n_ctx": n_ctx, "n_batch": n_batch, "threads": threads,
        }
        if threads_batch is not None:
            self._context_options["threads_batch"] = threads_batch
        self._role_options = dict(role_system=role_system, role_user=role_user,
                                  role_assistant=role_assistant)
        format_chat([{"role": "user", "content": ""}], **self._role_options)
        self._history: list[dict[str, str]] = []
        self._lock = RLock()
        self._inference_lock = RLock()
        self._native_session = None
        self._owner = None
        self._sessions: set[ChatSession] = set()
        self._generating = False
        self._closed = False

    @classmethod
    def from_pretrained(
        cls, model_id: str | Path, *,
        device: str = "auto",
        library_path: str | os.PathLike[str] | None = None,
        filename: str = "model.mercan",
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        gpu_layers: int | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        n_ctx: int | None = None,
        n_batch: int | None = None,
        threads: int | None = None,
        threads_batch: int | None = None,
        plugins: Sequence[str | os.PathLike[str]] = (),
        use_mmap: bool = True, check_tensors: bool = False,
        role_system: str = "sistem", role_user: str = "kullanici",
        role_assistant: str = "asistan",
    ) -> Model:
        """Load an existing .mercan file or download it from Hugging Face.

        'auto' selects usable CUDA, otherwise CPU. 'cuda' requires a CUDA
        build and usable device, and never silently falls back to CPU.
        gpu_layers or ETHOSOFT_MERCAN_GPU_LAYERS can override auto offload.
        """
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        runtime = MercanRuntime(library_path=library_path)
        model = None
        try:
            runtime.require_generation_api()
            runtime.require_session_api()
            device, gpu_layers = runtime.select_device(device, gpu_layers)
            path = resolve_model_file(
                model_id, filename=filename, revision=revision, cache_dir=cache_dir,
                local_files_only=local_files_only, token=token,
            )
            for plugin in plugins:
                runtime.load_plugin(plugin)
            model = runtime.load_model(path, device=device, gpu_layers=gpu_layers,
                                       use_mmap=use_mmap, check_tensors=check_tensors)
            return cls(
                runtime, model, path, system_prompt=system_prompt,
                n_ctx=n_ctx, n_batch=n_batch, threads=threads,
                threads_batch=threads_batch, role_system=role_system,
                role_user=role_user, role_assistant=role_assistant,
            )
        except BaseException:
            if model is not None:
                model.close()
            runtime.close()
            raise

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        """Return copies so callers cannot mutate stored turns indirectly."""
        with self._lock:
            return tuple(message.copy() for message in self._history)

    def reset_history(self) -> None:
        with self._inference_lock, self._lock:
            self._check()
            if self._native_session is not None:
                self._native_session.reset()
            self._history.clear()

    def session(self, *, system_prompt: str | None = None) -> ChatSession:
        """Create independent chat history and native KV cache over shared weights.

        Calls sharing one model are serialized. Closing the model closes all
        sessions; closing one session leaves other sessions and weights alive.
        """
        with self._inference_lock, self._lock:
            self._check()
            return ChatSession(self, system_prompt=system_prompt)

    def _check(self) -> None:
        if self._closed:
            raise MercanError("Model is closed")
        if self._owner is not None:
            self._owner._check()

    @contextmanager
    def _generation_slot(self):
        owner = self._owner or self
        if owner._generating:
            raise MercanError("Close the active generation stream before starting another request on this model")
        owner._generating = True
        try:
            yield
        finally:
            owner._generating = False

    def _validate_generation(self, prompt: str, *, temperature: float,
                             max_tokens: int, top_k: int, top_p: float,
                             repeat_penalty: float, repeat_last_n: int) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be >= 1")
        if not isinstance(repeat_last_n, int) or repeat_last_n < 0:
            raise ValueError("repeat_last_n must be >= 0")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not math.isfinite(repeat_penalty) or repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be finite and > 0")

    def stream_chat(
        self, prompt: str, *, temperature: float = 0.7,
        max_tokens: int = 256, top_k: int = 40, top_p: float = 0.9,
        repeat_penalty: float = 1.15, repeat_last_n: int = 64,
        seed: int | None = None, save_history: bool = True,
        _stop_event: Event | None = None,
    ) -> Iterator[str]:
        """Yield UTF-8-safe text fragments as native tokens are generated.

        A partially consumed stream does not alter conversation history.
        Successful turns retain KV cache. Incomplete turns discard their cache.
        Closing the stream releases its request and model lock.
        """
        self._validate_generation(
            prompt, temperature=temperature, max_tokens=max_tokens,
            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )
        with self._inference_lock, self._lock:
            self._check()
            if _stop_event is not None and _stop_event.is_set():
                return
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(self._history)
            messages.append({"role": "user", "content": prompt})
            options = dict(temperature=temperature, max_tokens=max_tokens,
                           top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
                           repeat_last_n=repeat_last_n, seed=seed)
            if not save_history:
                with closing(self.stream_generate(messages, stop_event=_stop_event, **options)) as stream:
                    yield from stream
                return
            if self._native_session is None:
                self._native_session = self._model.chat_session(context_options=self._context_options)
            fragments = []
            with self._generation_slot(), self._native_session.generation(
                messages=messages, role_options=self._role_options, **options,
            ) as request:
                with closing(self._decode_request(request, _stop_event)) as stream:
                    for piece in stream:
                        fragments.append(piece)
                        yield piece
                if not request.cancelled and not (_stop_event is not None and _stop_event.is_set()):
                    request.commit()
                    self._history.extend((
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "".join(fragments)},
                    ))

    @staticmethod
    def _decode_request(request, stop_event=None):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        with closing(request.stream(stop_event)) as stream:
            for raw in stream:
                piece = decoder.decode(raw)
                if piece:
                    yield piece
        if not request.cancelled:
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail

    def generate(self, messages: Sequence[Mapping[str, str]], **kwargs: object) -> str:
        """Generate from caller-owned messages without reading or writing history.

        No default system prompt is inserted. Supply the complete transcript.
        Generation options are the same as stream_generate().
        """
        return "".join(self.stream_generate(messages, **kwargs))

    def stream_generate(
        self, messages: Sequence[Mapping[str, str]], *, temperature: float = 0.7,
        max_tokens: int = 256, top_k: int = 40, top_p: float = 0.9,
        repeat_penalty: float = 1.15, repeat_last_n: int = 64,
        seed: int | None = None, stop_event: Event | None = None,
    ) -> Iterator[str]:
        """Stream the common native chat engine without changing history.

        C++ owns formatting, prefill, sampling, stop handling and decode.
        Closing the iterator cancels and joins the native worker before freeing
        its context. In-flight device operations finish before cancellation.
        """
        self._validate_generation(
            "messages", temperature=temperature, max_tokens=max_tokens,
            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )
        if not messages:
            raise ValueError("At least one message is required")
        with self._inference_lock, self._lock:
            self._check()
            if stop_event is not None and stop_event.is_set():
                return
            with self._generation_slot(), self._model.generation(
                messages=messages, context_options=self._context_options,
                role_options=self._role_options, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n,
                seed=seed,
            ) as request:
                with closing(self._decode_request(request, stop_event)) as stream:
                    yield from stream

    def chat(
        self, prompt: str, *, temperature: float = 0.7,
        max_tokens: int = 256, top_k: int = 40, top_p: float = 0.9,
        repeat_penalty: float = 1.15, repeat_last_n: int = 64,
        seed: int | None = None, save_history: bool = True,
    ) -> str:
        """Return the concatenated stream while maintaining chat history."""
        return "".join(self.stream_chat(
            prompt, temperature=temperature, max_tokens=max_tokens,
            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n, seed=seed, save_history=save_history,
        ))

    async def async_chat(self, prompt: str, **kwargs: object) -> str:
        """Collect the async stream; task cancellation also cancels native work."""
        stream = self.astream_chat(prompt, **kwargs)
        try:
            return "".join([part async for part in stream])
        finally:
            await stream.aclose()

    async def astream_chat(self, prompt: str, **kwargs: object):
        """Yield an async stream without blocking the caller's event loop.

        The sync generator is owned by one worker thread; stopping iteration
        signals cancellation and avoids recording incomplete chat history.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=1)
        stop = Event()

        def emit(item):
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stop.is_set():
                try:
                    future.result(timeout=0.05)
                    return True
                except FutureTimeout:
                    pass
            future.cancel()
            return False

        def worker() -> None:
            try:
                with closing(self.stream_chat(prompt, _stop_event=stop, **kwargs)) as stream:
                    for chunk in stream:
                        if not emit(("chunk", chunk)):
                            break
            except BaseException as exc:
                emit(("error", exc))
            finally:
                emit(("done", None))

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                kind, value = await queue.get()
                if kind == "done":
                    break
                if kind == "error":
                    raise value
                yield str(value)
        finally:
            stop.set()
            await asyncio.shield(task)

    def info(self) -> dict[str, object]:
        """Snapshot of metadata already exposed by the native public API."""
        with self._lock:
            self._check()
            placement = self._model.placement
            cache = None
            if self._native_session is not None:
                cache = ({"busy": True} if self._native_session._requests
                         else self._native_session.stats())
            return {
                "path": str(self.model_path),
                "file_bytes": self.model_path.stat().st_size,
                "runtime_version": self._runtime.version,
                "generation_engine": "native-v1",
                "session_engine": "native-v1",
                "kv_cache": cache,
                "device": placement["device"],
                "placement": placement,
                "architecture": self._model.architecture,
                "tokenizer": self._model.tokenizer,
                "vocab_size": self._model.vocab_size,
                "eos_token_id": self._model.eos_token,
                "pad_token_id": self._model.pad_token,
                "context_options": self._context_options.copy(),
                "history_turns": len(self._history) // 2,
            }

    def benchmark(
        self, prompt: str = "Merhaba, nasılsın?", *,
        runs: int = 3, max_tokens: int = 32, warmup: int = 0,
    ) -> dict[str, float | int | str]:
        """Wall-clock throughput, excluding model download/load time.

        Output-token rate is an estimate based on re-tokenizing generated text.
        """
        if not isinstance(runs, int) or not 1 <= runs <= 100:
            raise ValueError("runs must be in 1..100")
        if not isinstance(warmup, int) or not 0 <= warmup <= 20:
            raise ValueError("warmup must be in 0..20")
        for _ in range(warmup):
            self.chat(prompt, max_tokens=max_tokens, save_history=False)
        durations = []
        total_chars = 0
        approx_tokens = 0
        for _ in range(runs):
            start = perf_counter()
            reply = self.chat(prompt, max_tokens=max_tokens, save_history=False)
            durations.append(perf_counter() - start)
            total_chars += len(reply)
            if reply:
                approx_tokens += len(self._model.tokenize(reply, parse_special=False))
        elapsed = sum(durations)
        return {
            "runs": runs,
            "total_seconds": elapsed,
            "average_seconds": elapsed / runs,
            "characters_per_second": total_chars / elapsed if elapsed else 0.,
            "approx_tokens_per_second": approx_tokens / elapsed if elapsed else 0.,
            "note": "Output token rate is approximate (re-tokenized text); wall time includes prefill.",
        }

    def close(self) -> None:
        with self._inference_lock, self._lock:
            if self._closed:
                return
            if (self._owner or self)._generating:
                raise MercanError("Close the active generation stream before closing this model or session")
            for session in list(self._sessions):
                session.close()
            if self._native_session is not None:
                self._native_session.close()
                self._native_session = None
            if self._owner is None:
                self._model.close()
                self._runtime.close()
            else:
                self._owner._sessions.discard(self)
            self._closed = True

    def __enter__(self) -> Model:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ChatSession(Model):
    """Independent conversation borrowing one Model's weights and runtime."""

    def __init__(self, owner: Model, *, system_prompt: str | None = None):
        with owner._inference_lock, owner._lock:
            owner._check()
            prompt = owner.system_prompt if system_prompt is None else system_prompt
            if not isinstance(prompt, str):
                raise TypeError("system_prompt must be a string")
            super().__init__(owner._runtime, owner._model, owner.model_path,
                             system_prompt=prompt, **owner._context_options, **owner._role_options)
            self._owner = owner._owner or owner
            self._inference_lock = self._owner._inference_lock
            self._owner._sessions.add(self)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise TypeError("Create a ChatSession with model.session()")
