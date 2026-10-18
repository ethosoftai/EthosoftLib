"""Owned native requests and a bounded callback-to-iterator streaming bridge.

The C++ engine performs the complete prefill/sampling/decode loop. Python only
marshals inputs, forwards output bytes and coordinates cooperative cancellation.
"""
from __future__ import annotations

import ctypes as C
import json
from queue import Empty, Full, Queue
from threading import Event, Thread

from .native import (ChatMessage, GenerationCallback, GenerationParams,
                     MercanError, bind_generation_api, bind_session_api, native_error)


class NativeChatSession:
    def __init__(self, model, *, context_options=None):
        model._check()
        self._model, self._lib = model, model._runtime._lib
        self._requests = 0
        bind_session_api(self._lib)
        cp = model._context_params(**(context_options or {}))
        self._handle = self._lib.mercan_session_create_v1(model._handle, cp)
        if not self._handle:
            raise native_error(self._lib, "Native session creation failed")
        model._contexts += 1

    def _check(self):
        self._model._check()
        if not self._handle:
            raise MercanError("Native chat session is closed")

    def generation(self, **options):
        self._check()
        if self._requests:
            raise MercanError("Session already has an outstanding request")
        return NativeGeneration(self._model, _session=self, **options)

    def stats(self):
        self._check()
        raw = self._lib.mercan_session_stats_json_v1(self._handle)
        if not raw:
            raise native_error(self._lib, "Native session stats failed")
        return json.loads(raw)

    def reset(self):
        self._check()
        if self._lib.mercan_session_reset_v1(self._handle) != 0:
            raise native_error(self._lib, "Native session reset failed")

    def close(self):
        if self._handle:
            if self._lib.mercan_session_free_v1(self._handle) != 0:
                raise native_error(self._lib, "Close the session request before closing its session")
            self._handle = None
            self._model._contexts -= 1

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()


class NativeGeneration:
    def __init__(
        self, model, *, messages=None, prompt=None, context_options=None,
        role_options=None, max_tokens=256, temperature=0.7, top_k=40,
        top_p=0.9, repeat_penalty=1.15, repeat_last_n=64, seed=None, _session=None,
    ):
        model._check()
        self._model = model
        self._lib = model._runtime._lib
        self._handle = None
        self._running = False
        self._started = False
        self._session = _session
        self._consumed = False
        bind_generation_api(self._lib)
        if _session is not None and context_options is not None:
            raise ValueError("Session context options must be configured when the session is created")
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
            if _session is None:
                handle = self._lib.mercan_chat_begin_v1(
                    model._handle, native_messages, len(encoded), *headers, cp, C.byref(params))
            else:
                handle = self._lib.mercan_session_chat_begin_v1(
                    _session._handle, native_messages, len(encoded), *headers, C.byref(params))
        else:
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be non-empty")
            raw = prompt.encode("utf-8")
            if _session is None:
                handle = self._lib.mercan_generate_begin_v1(model._handle, raw, len(raw), cp, C.byref(params))
            else:
                handle = self._lib.mercan_session_generate_begin_v1(_session._handle, raw, len(raw), C.byref(params))
        if not handle:
            error = native_error(self._lib, "Native generation initialization failed")
            if "exceeds" in str(error):
                raise ValueError(str(error))
            raise error
        self._handle = handle
        if _session is None:
            model._contexts += 1
        else:
            _session._requests += 1

    @property
    def cancelled(self):
        return bool(self._handle and self._lib.mercan_generation_finish_reason_v1(self._handle) == 3)

    def commit(self):
        if not self._session or not self._handle or not self._consumed or self.cancelled:
            raise MercanError("Only a fully consumed successful session request can be committed")
        if self._lib.mercan_generation_commit_session_v1(self._handle) != 0:
            raise native_error(self._lib, "Native session commit failed")

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
            self._consumed = not stopping()
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
            if self._session is None:
                self._model._contexts -= 1
            else:
                self._session._requests -= 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
