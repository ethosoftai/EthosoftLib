"""Embedding API v1 bindings with a fake native ABI, plus an opt-in real run."""
import ctypes as C
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ethosoftlib.mercan import MercanEmbedder, MercanError, MercanModel, Model
from ethosoftlib.mercan.native import ContextParams, bind_embedding_api
from ethosoftlib.rag import Index


class FakeLibrary:
    size = 3

    def __init__(self):
        self.created = []
        self.freed = []
        self.fail = False

    def mercan_context_default_params(self):
        return ContextParams(4096, 512, 8, 8)

    def mercan_embedder_create_v1(self, model, params, pooling):
        self.created.append((params.n_ctx, params.n_threads, pooling))
        return 77

    def mercan_embedder_size_v1(self, handle):
        return self.size

    def mercan_embed_v1(self, handle, raw, length, out, capacity):
        if self.fail:
            return -1
        for i in range(self.size):
            out[i] = float(length + i)
        return self.size

    def mercan_embedder_free_v1(self, handle):
        self.freed.append(handle)

    def mercan_last_error(self):
        return b"native embed error"


def make_model():
    lib = FakeLibrary()
    runtime = SimpleNamespace(_lib=lib, require_embedding_api=lambda: None, _check=lambda: None)
    return MercanModel(runtime, 1), lib


def test_old_runtime_without_embeddings_is_reported():
    with pytest.raises(MercanError, match="Embedding API v1"):
        bind_embedding_api(SimpleNamespace())


def test_native_embedder_lifecycle_and_errors():
    model, lib = make_model()
    with pytest.raises(ValueError, match="pooling"):
        model.embedder(pooling="cls")
    embedder = model.embedder(pooling="last", n_ctx=256, threads=2)
    assert isinstance(embedder, MercanEmbedder)
    assert lib.created == [(256, 2, 1)] and model._contexts == 1
    assert embedder.embed("İç") == [4.0, 5.0, 6.0]  # UTF-8 length is passed
    with pytest.raises(ValueError):
        embedder.embed("  ")
    lib.fail = True
    with pytest.raises(MercanError, match="native embed error"):
        embedder.embed("x")
    embedder.close()
    embedder.close()
    assert lib.freed == [77] and model._contexts == 0
    with pytest.raises(MercanError, match="closed"):
        embedder.embed("x")


class FakeNativeModel:
    def __init__(self):
        self.embedders = []
        self.closed = False

    def embedder(self, *, pooling, threads):
        vectors = {"mean": [3.0, 4.0], "last": [0.0, 2.0]}
        handle = SimpleNamespace(embed=lambda text: list(vectors[pooling]),
                                 closed=False)
        handle.close = lambda: setattr(handle, "closed", True)
        self.embedders.append((pooling, threads, handle))
        return handle

    def close(self):
        self.closed = True


def make_high_level():
    native = FakeNativeModel()
    runtime = SimpleNamespace(close=lambda: None)
    return Model(runtime, native, Path("model.mercan"), threads=4), native


def test_model_embed_normalizes_reuses_contexts_and_closes_them():
    model, native = make_high_level()
    assert model.embed("tek") == [0.6, 0.8]
    assert model.embed(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert model.embed("x", normalize=False) == [3.0, 4.0]
    assert model.embed("x", pooling="last") == [0.0, 1.0]
    session = model.session()
    assert session.embed("oturum") == [0.6, 0.8]
    assert [(p, t) for p, t, _ in native.embedders] == [("mean", 4), ("last", 4)]
    model.close()
    assert native.closed and all(handle.closed for *_, handle in native.embedders)
    with pytest.raises(MercanError):
        model.embed("x")


def test_model_is_a_rag_embedder():
    model, _ = make_high_level()
    index = Index(embedder=model)
    index.add("içerik", "a.txt")
    assert index.search("ilgisiz")[0].source == "a.txt"


requires_native = pytest.mark.skipif(
    not (os.environ.get("ETHOSOFT_MERCAN_LIBRARY") and os.environ.get("ETHOSOFT_MERCAN_TEST_MODEL")),
    reason="set ETHOSOFT_MERCAN_LIBRARY and ETHOSOFT_MERCAN_TEST_MODEL",
)


@requires_native
def test_real_embeddings_are_deterministic_and_topical():
    with Model.from_pretrained(Path(os.environ["ETHOSOFT_MERCAN_TEST_MODEL"]),
                               library_path=os.environ["ETHOSOFT_MERCAN_LIBRARY"]) as model:
        capital, question, code = model.embed([
            "Ankara Türkiye'nin başkentidir.",
            "Türkiye'nin başkenti hangi şehirdir?",
            "sorted() fonksiyonu listeyi sıralı döndürür.",
        ])
        assert math.isclose(sum(x * x for x in capital), 1.0, rel_tol=1e-4)
        assert model.embed("Ankara Türkiye'nin başkentidir.") == pytest.approx(capital, abs=1e-5)
        cosine = lambda a, b: sum(x * y for x, y in zip(a, b))
        assert cosine(capital, question) > cosine(code, question)
        assert model.chat("Merhaba", max_tokens=4)
