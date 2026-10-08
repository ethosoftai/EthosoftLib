"""High-level Mercan model wrapper. Importing it does not load native binaries."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import os
import random
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

    def chat(
        self, prompt: str, *, temperature: float = 0.7,
        max_tokens: int = 256, top_k: int = 40, top_p: float = 0.9,
        repeat_penalty: float = 1.15, repeat_last_n: int = 64,
        seed: int | None = None, save_history: bool = True,
    ) -> str:
        """Return one assistant response while preserving conversation history."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be >= 1")
        if not isinstance(repeat_last_n, int) or repeat_last_n < 0:
            raise ValueError("repeat_last_n must be >= 0")
        # Validate sampling arguments before touching native state.
        from .sampling import sample_token
        sample_token([0.0], 1, temperature=temperature, top_k=top_k,
                     top_p=top_p, repeat_penalty=repeat_penalty, rng=random.Random(0))
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
            output = bytearray()
            with self._model.context(**self._context_options) as context:
                if len(prompt_tokens) + max_tokens > context.size:
                    raise ValueError(
                        f"Prompt ({len(prompt_tokens)} tokens) + max_tokens "
                        f"({max_tokens}) exceeds n_ctx={context.size}"
                    )
                context.decode(prompt_tokens)
                recent = list(prompt_tokens[-repeat_last_n:]) if repeat_last_n else []
                for index in range(max_tokens):
                    next_token = context.sample_token(
                        temperature=temperature, top_k=top_k, top_p=top_p,
                        repeat_penalty=repeat_penalty, recent_tokens=recent, rng=rng,
                    )
                    if next_token in stop_ids:
                        break
                    output.extend(self._model.token_to_bytes(next_token))
                    if repeat_last_n:
                        recent.append(next_token)
                        if len(recent) > repeat_last_n:
                            del recent[0]
                    if index + 1 < max_tokens:
                        context.decode([next_token])
            answer = output.decode("utf-8", "replace")
            if save_history:
                self._history.extend((
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ))
            return answer

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
