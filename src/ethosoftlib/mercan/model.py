"""High-level Mercan model wrapper. Importing it does not load native binaries."""
from __future__ import annotations

from collections.abc import Iterator
import asyncio
import codecs
from pathlib import Path
import os
import random
from threading import Event
from time import perf_counter
from threading import RLock

from .chat import DEFAULT_SYSTEM_PROMPT, format_chat
from .hub import resolve_model_file
from .native import MercanError
from .runtime import MercanRuntime, MercanModel


class Model:
    """High-level, Mercan-specific API backed by a local native libmercan.

    Each chat call creates a fresh inference context and replays the transcript.
    The transcript is retained between calls, but native KV cache reuse is not
    yet implemented across turns. Always close the model when done.
    """

    def __init__(
        self, runtime: MercanRuntime, model: MercanModel, model_path: Path, *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        n_ctx: int | None = None, n_batch: int | None = None,
        threads: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._model = model
        self.model_path = model_path
        self.system_prompt = system_prompt
        self._context_options = {
            "n_ctx": n_ctx, "n_batch": n_batch, "threads": threads,
        }
        self._history: list[dict[str, str]] = []
        self._lock = RLock()
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
    ) -> Model:
        """Load an existing .mercan file or download it from Hugging Face.

        'auto' is conservative CPU-first unless the caller opts into GPU
        layers through ETHOSOFT_MERCAN_GPU_LAYERS. device='cuda' uses all
        layers (-1); CUDA requires a CUDA-capable native library.
        """
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        if gpu_layers is not None and (not isinstance(gpu_layers, int) or gpu_layers < -1):
            raise ValueError("gpu_layers must be >= -1")
        if gpu_layers is None:
            if device == "cpu":
                gpu_layers = 0
            elif device == "cuda":
                gpu_layers = -1
            else:
                gpu_layers = int(os.environ.get("ETHOSOFT_MERCAN_GPU_LAYERS", "0"))
        if device == "cpu" and gpu_layers != 0:
            raise ValueError("device='cpu' cannot use GPU layers")
        path = resolve_model_file(
            model_id, filename=filename, revision=revision, cache_dir=cache_dir,
            local_files_only=local_files_only, token=token,
        )
        runtime = MercanRuntime(library_path=library_path)
        try:
            model = runtime.load_model(path, gpu_layers=gpu_layers)
        except BaseException:
            runtime.close()
            raise
        return cls(
            runtime, model, path, system_prompt=system_prompt,
            n_ctx=n_ctx, n_batch=n_batch, threads=threads,
        )

    @property
    def history(self) -> tuple[dict[str, str], ...]:
        """Return copies so callers cannot mutate stored turns indirectly."""
        with self._lock:
            return tuple(message.copy() for message in self._history)

    def reset_history(self) -> None:
        with self._lock:
            self._check()
            self._history.clear()

    def _check(self) -> None:
        if self._closed:
            raise MercanError("Model is closed")

    def _validate_generation(self, prompt: str, *, temperature: float,
                             max_tokens: int, top_k: int, top_p: float,
                             repeat_penalty: float, repeat_last_n: int) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be >= 1")
        if not isinstance(repeat_last_n, int) or repeat_last_n < 0:
            raise ValueError("repeat_last_n must be >= 0")
        from .sampling import sample_token
        sample_token([0.0], 1, temperature=temperature, top_k=top_k,
                     top_p=top_p, repeat_penalty=repeat_penalty,
                     rng=random.Random(0))

    def stream_chat(
        self, prompt: str, *, temperature: float = 0.7,
        max_tokens: int = 256, top_k: int = 40, top_p: float = 0.9,
        repeat_penalty: float = 1.15, repeat_last_n: int = 64,
        seed: int | None = None, save_history: bool = True,
        _stop_event: Event | None = None,
    ) -> Iterator[str]:
        """Yield UTF-8-safe text fragments as native tokens are generated.

        A partially consumed stream does not alter conversation history.
        Native context and model lock are released when the stream is closed.
        """
        self._validate_generation(
            prompt, temperature=temperature, max_tokens=max_tokens,
            top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
        )
        with self._lock:
            self._check()
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.extend(self._history)
            messages.append({"role": "user", "content": prompt})
            formatted = format_chat(messages)
            prompt_tokens = self._model.tokenize(formatted, parse_special=True)
            if not prompt_tokens:
                raise MercanError("Prompt produced no tokens")
            stop_ids = {self._model.eos_token, self._model.pad_token}
            for marker in ("<|im_start|>", "<|im_end|>"):
                single = self._model.tokenize(marker, parse_special=True)
                if len(single) == 1:
                    stop_ids.add(single[0])

            rng = random.Random(seed)
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            fragments: list[str] = []
            completed = False
            with self._model.context(**self._context_options) as context:
                if len(prompt_tokens) + max_tokens > context.size:
                    raise ValueError(
                        f"Prompt ({len(prompt_tokens)} tokens) + max_tokens "
                        f"({max_tokens}) exceeds n_ctx={context.size}"
                    )
                context.decode(prompt_tokens)
                recent = list(prompt_tokens[-repeat_last_n:]) if repeat_last_n else []
                for index in range(max_tokens):
                    if _stop_event is not None and _stop_event.is_set():
                        return
                    next_token = context.sample_token(
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        repeat_penalty=repeat_penalty, recent_tokens=recent, rng=rng,
                    )
                    if next_token in stop_ids:
                        break
                    piece = decoder.decode(self._model.token_to_bytes(next_token))
                    if piece:
                        fragments.append(piece)
                        yield piece
                    if repeat_last_n:
                        recent.append(next_token)
                        if len(recent) > repeat_last_n:
                            del recent[0]
                    if index + 1 < max_tokens:
                        context.decode([next_token])
                tail = decoder.decode(b"", final=True)
                if tail:
                    fragments.append(tail)
                    yield tail
                completed = True
            if completed and save_history:
                self._history.extend((
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "".join(fragments)},
                ))

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
        """Run blocking native generation in a worker thread."""
        return await asyncio.to_thread(self.chat, prompt, **kwargs)

    async def astream_chat(self, prompt: str, **kwargs: object):
        """Yield an async stream without blocking the caller's event loop.

        The sync generator is owned by one worker thread; stopping iteration
        signals cancellation and avoids recording incomplete chat history.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        stop = Event()

        def worker() -> None:
            try:
                for chunk in self.stream_chat(prompt, _stop_event=stop, **kwargs):
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, ("chunk", chunk))
            except BaseException as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

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
            return {
                "path": str(self.model_path),
                "file_bytes": self.model_path.stat().st_size,
                "runtime_version": self._runtime.version,
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
        with self._lock:
            if self._closed:
                return
            self._model.close()
            self._runtime.close()
            self._closed = True

    def __enter__(self) -> Model:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
