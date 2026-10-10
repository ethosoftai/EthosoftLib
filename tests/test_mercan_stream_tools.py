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

class FakeModel:
    eos_token = 9
    pad_token = 10
    architecture = "nedolm"
    tokenizer = "ndsurf004"
    vocab_size = 32000

    def __init__(self, tokens=(1, 2, 8)):
        self.tokens = tokens
        self.contexts = []
        self.closed = False

    def tokenize(self, value, **kwargs):
        if value == "<|im_start|>":
            return [7]
        if value == "<|im_end|>":
            return [8]
        return [4, 5] if not kwargs.get("parse_special") is False else [4]

    def token_to_bytes(self, token):
        return {1: b"\xc4", 2: b"\xb0"}[token]

    def context(self, **options):
        ctx = FakeContext(self)
        self.contexts.append(ctx)
        return ctx

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
