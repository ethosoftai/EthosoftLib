"""Owned native requests and a bounded callback-to-iterator streaming bridge.

The C++ engine performs the complete prefill/sampling/decode loop. Python only
marshals inputs, forwards output bytes and coordinates cooperative cancellation.
"""
from __future__ import annotations

import ctypes as C
from queue import Empty, Full, Queue
from threading import Event, Thread

from .native import (ChatMessage, GenerationCallback, GenerationParams,
                     MercanError, bind_generation_api, native_error)


class NativeGeneration:
    def __init__(
        self, model, *, messages=None, prompt=None, context_options=None,
        role_options=None, max_tokens=256, temperature=0.7, top_k=40,
        top_p=0.9, repeat_penalty=1.15, repeat_last_n=64, seed=None,
    ):
        model._check()
        self._model = model
        self._lib = model._runtime._lib
        self._handle = None
        self._running = False
        self._started = False
        bind_generation_api(self._lib)
        params = GenerationParams()
        if self._lib.mercan_generation_default_params_v1(C.byref(params), C.sizeof(params)) != 0:
            raise native_error(self._lib, "Could not initialize generation options")
        for name, value in dict(max_tokens=max_tokens, top_k=top_k,
                                repeat_last_n=repeat_last_n).items():
            if not isinstance(value, int) or not 0 <= value <= 2147483647:
                raise ValueError(f"{name} must fit a nonnegative int32")
            setattr(params, name, value)
        params.temperature, params.top_p, params.repeat_penalty = temperature, top_p, repeat_penalty
        if seed is not None:
            if not isinstance(seed, int) or not 0 <= seed <= 4294967295:
                raise ValueError("seed must be in 0..4294967295")
            params.seed, params.use_seed = seed, 1
        cp = model._context_params(**(context_options or {}))
        if messages is not None:
            if not messages:
                raise ValueError("At least one message is required")
            encoded = []
            for message in messages:
                if message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str):
                    raise ValueError("Messages must contain a known role and string content")
                encoded.append((message["role"].encode(), message["content"].encode("utf-8")))
            native_messages = (ChatMessage * len(encoded))(*(
                ChatMessage(role, content, len(content)) for role, content in encoded
            ))
            roles = role_options or {}
            headers = [roles.get(name, default).encode("utf-8") for name, default in
                       (("role_system", "sistem"), ("role_user", "kullanici"),
                        ("role_assistant", "asistan"))]
            handle = self._lib.mercan_chat_begin_v1(
                model._handle, native_messages, len(encoded), *headers, cp, C.byref(params))
        else:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be non-empty")
            raw = prompt.encode("utf-8")
            handle = self._lib.mercan_generate_begin_v1(model._handle, raw, len(raw), cp, C.byref(params))
        if not handle:
            error = native_error(self._lib, "Native generation initialization failed")
            if "exceeds" in str(error):
                raise ValueError(str(error))
            raise error
        self._handle = handle
        model._contexts += 1

    @property
    def cancelled(self):
        return bool(self._handle and self._lib.mercan_generation_finish_reason_v1(self._handle) == 3)

    def cancel(self):
        if self._handle:
            self._lib.mercan_generation_cancel_v1(self._handle)

    def stream(self, stop_event=None):
        if not self._handle or self._started:
            raise MercanError("Native generation is closed or has already run")
        self._started = self._running = True
        output = Queue(maxsize=1)
        abort, finished = Event(), Event()
        errors = []

        def stopping():
            return abort.is_set() or (stop_event is not None and stop_event.is_set())

        @GenerationCallback
        def receive(data, size, _):
            try:
                if not size:
                    return int(not stopping())
                piece = C.string_at(data, size)
                while not stopping():
                    try:
                        output.put(piece, timeout=0.05)
                        return 1
                    except Full:
                        pass
                return 0
            except BaseException as exc:
                errors.append(exc)
                return 0

        def worker():
            try:
                if self._lib.mercan_generation_run_v1(self._handle, receive, None) < 0:
                    # Native last_error is thread-local: copy it on this thread.
                    errors.append(native_error(self._lib, "Native generation failed"))
            except BaseException as exc:
                errors.append(exc)
            finally:
                finished.set()

        thread = Thread(target=worker, name="mercan-generation")
        try:
            if stopping():
                self.cancel()
            thread.start()
            while not finished.is_set() or not output.empty():
                if stopping():
                    break
                try:
                    yield output.get(timeout=0.05)
                except Empty:
                    pass
            if errors:
                raise errors[0]
        finally:
            abort.set()
            self.cancel()
            if thread.ident is not None:
                thread.join()
            self._running = False

    def close(self):
        if self._running:
            raise MercanError("Close the native generation stream before releasing its request")
        if self._handle:
            self._lib.mercan_generation_free_v1(self._handle)
            self._handle = None
            self._model._contexts -= 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
