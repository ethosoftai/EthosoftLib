"""High-level streaming, async, introspection and benchmark contract tests."""
import asyncio
from pathlib import Path
import threading

import pytest

from ethosoftlib.mercan import Model

class FakeContext:
    size = 1024

    def __init__(self, model):
        self.model = model
        self.index = 0
        self.closed = False

        self.decoded = []

    def decode(self, tokens):
        self.decoded.extend(tokens)

    def sample_token(self, **kwargs):
        result = self.model.tokens[self.index]
        self.index += 1
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class FakeRequest:
    cancelled = False

    def __init__(self, model, max_tokens):
        self.model, self.max_tokens = model, max_tokens
        self.closed = False
        self.session = None
        self.committed = False

    def commit(self):
        self.committed = True
        if self.session:
            self.session.cached += 3

    def stream(self, stop_event=None):
        for token in self.model.tokens[:self.max_tokens]:
            if stop_event is not None and stop_event.is_set():
                self.cancelled = True
                return
            if token == 8:
                break
            yield self.model.token_to_bytes(token)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.closed = True
        if self.session:
            self.session._requests -= 1
            if not self.committed:
                self.session.cached = 0


class FakeNativeSession:
    def __init__(self, model):
        self.model = model
        self._requests = 0
        self.cached = 0
        self.closed = False

    def generation(self, **kwargs):
        request = self.model.generation(**kwargs)
        request.session = self
        self._requests += 1
        return request

    def reset(self):
        self.cached = 0

    def stats(self):
        return {"cached_tokens": self.cached}

    def close(self):
        self.closed = True

class FakeModel:
    device = "cpu"
    placement = {"device": "cpu", "placement": "cpu", "devices": [{"name": "CPU", "backend": "CPU"}]}
    eos_token = 9
    pad_token = 10
    architecture = "nedolm"
    tokenizer = "ndsurf004"
    vocab_size = 32000

    def __init__(self, tokens=(1, 2, 8)):
        self.tokens = tokens
        self.contexts = []
        self.sessions = []
        self.closed = False

    def tokenize(self, value, **kwargs):
        if value == "<|im_start|>":
            return [7]
        if value == "<|im_end|>":
            return [8]
        return [4, 5] if not kwargs.get("parse_special") is False else [4]

    def token_to_bytes(self, token):
        return {1: b"\xc4", 2: b"\xb0"}[token]

    def chat_session(self, **options):
        session = FakeNativeSession(self)
        self.sessions.append(session)
        return session

    def context(self, **options):
        ctx = FakeContext(self)
        self.contexts.append(ctx)
        return ctx

    def generation(self, *, max_tokens, **kwargs):
        request = FakeRequest(self, max_tokens)
        self.contexts.append(request)
        return request

    def close(self):
        self.closed = True

class FakeRuntime:
    version = "0.3-test"

    def close(self):
        self.closed = True


def make_model(tmp_path, **kwargs):
    path = tmp_path / "model.mercan"
    path.write_bytes(b"binary model mock")
    return Model(FakeRuntime(), FakeModel(), path, system_prompt="", **kwargs)


def test_unicode_stream_and_history(tmp_path):
    m = make_model(tmp_path)
    assert list(m.stream_chat("Merhaba")) == ["İ"]
    assert m.history[-1] == {"role": "assistant", "content": "İ"}
    assert m._model.contexts[0].closed
    assert m.chat("Nasılsın?") == "İ"
    assert len(m.history) == 4
    assert len(m._model.sessions) == 1


def test_sessions_share_weights_but_own_history_and_cache(tmp_path):
    m = make_model(tmp_path)
    a, b = m.session(), m.session(system_prompt="Başka bir sohbet")
    assert a._model is b._model is m._model
    assert a.chat("Birinci") == "İ"
    assert b.history == m.history == ()
    assert b.chat("İkinci") == "İ"
    assert a._native_session is not b._native_session
    a.close()
    assert not m._model.closed
    assert b.chat("Devam") == "İ"
    m.close()
    assert b._closed and m._model.closed
    assert all(s.closed for s in m._model.sessions)


def test_history_reset_also_resets_cache(tmp_path):
    m = make_model(tmp_path)
    m.chat("Merhaba")
    assert m.info()["kv_cache"]["cached_tokens"] > 0
    m.reset_history()
    assert m.history == ()
    assert m.info()["kv_cache"]["cached_tokens"] == 0


def test_noncommitting_chat_keeps_the_existing_session(tmp_path):
    m = make_model(tmp_path)
    m.chat("Merhaba")
    history, stats = m.history, m.info()["kv_cache"]
    m.chat("Deneme", save_history=False)
    assert m.history == history
    assert m.info()["kv_cache"] == stats


def test_nested_generation_and_close_are_rejected_while_streaming(tmp_path):
    from ethosoftlib.mercan import MercanError
    m = make_model(tmp_path)
    other = m.session()
    stream = m.stream_chat("Merhaba")
    assert next(stream) == "İ"
    with pytest.raises(MercanError, match="active generation stream"):
        other.chat("Devam")
    with pytest.raises(MercanError, match="active generation stream"):
        m.close()
    stream.close()
    assert m.history == ()
    assert m.info()["kv_cache"]["cached_tokens"] == 0
    m.close()


def test_partial_stream_does_not_write_history(tmp_path):
    m = make_model(tmp_path)
    gen = m.stream_chat("Merhaba")
    assert next(gen) == "İ"
    gen.close()
    assert m.history == ()
    assert m._model.contexts[0].closed


def test_info_and_benchmark_do_not_modify_history(tmp_path):
    m = make_model(tmp_path)
    info = m.info()
    assert info["architecture"] == "nedolm"
    assert info["tokenizer"] == "ndsurf004"
    assert info["vocab_size"] == 32000
    assert info["file_bytes"] == len(b"binary model mock")
    result = m.benchmark("Merhaba", runs=2, max_tokens=3)
    assert result["runs"] == 2
    assert result["average_seconds"] >= 0
    assert m.history == ()
    with pytest.raises(ValueError, match="runs"):
        m.benchmark(runs=0)
    with pytest.raises(ValueError, match="warmup"):
        m.benchmark(warmup=21)


def test_async_chat_and_stream(tmp_path):
    async def run():
        m = make_model(tmp_path)
        assert await m.async_chat("Merhaba") == "İ"
        chunks = []
        async for chunk in m.astream_chat("Tekrar"):
            chunks.append(chunk)
        assert chunks == ["İ"]
        assert len(m.history) == 4
        assert all(c.closed for c in m._model.contexts)
    asyncio.run(run())


def test_async_stream_break_releases_context(tmp_path):
    async def run():
        m = make_model(tmp_path)
        gen = m.astream_chat("Merhaba")
        assert await anext(gen) == "İ"
        await gen.aclose()
        assert all(c.closed for c in m._model.contexts)
    asyncio.run(run())
