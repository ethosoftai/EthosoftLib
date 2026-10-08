"""Pure Python tests of the standalone tokenizer API, no native compiler required."""
import ctypes as C

import pytest

from ethosoftlib.nedo import (
    Tokenizer, NedoTokenizer, NedoTokenizerError, NedoLibraryNotFound,
    EXPECTED_SHA256, NEDO_VOCAB_SIZE,
)


class FakeFunc:
    def __init__(self, call):
        self.call = call
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.call(*args)


class FakeLibrary:
    def __init__(self):
        self.nedo004_vocab_sha256 = FakeFunc(lambda: EXPECTED_SHA256.encode())
        self.nedo004_vocab_size = FakeFunc(lambda: NEDO_VOCAB_SIZE)
        self.nedo004_encode_copy = FakeFunc(self.encode)
        self.nedo004_decode_copy = FakeFunc(self.decode)

    def encode(self, data, n, out, capacity, required):
        tokens = [b + 3 for b in data[:n]]
        C.cast(required, C.POINTER(C.c_size_t)).contents.value = len(tokens)
        if capacity < len(tokens):
            return -4
        for i, value in enumerate(tokens):
            out[i] = value
        return 0

    def decode(self, ids, n, out, capacity, required):
        raw = bytes(t - 3 for t in list(ids[:n]) if t not in (0, 1, 2))
        C.cast(required, C.POINTER(C.c_size_t)).contents.value = len(raw)
        if capacity < len(raw):
            return -4
        if raw:
            C.memmove(out, raw, len(raw))
        return 0


def test_python_adapter_using_mock_library(monkeypatch):
    import ethosoftlib.nedo as nedo
    monkeypatch.setattr(nedo, "_load_library", lambda _: FakeLibrary())
    tok = Tokenizer()
    assert isinstance(tok, NedoTokenizer)
    assert tok.vocab_size == 32_000
    assert tok.vocab_sha256 == EXPECTED_SHA256
    ids = tok.encode("İstanbul ve deniz 😀")
    assert tok.decode(ids) == "İstanbul ve deniz 😀"
    assert tok.tokenize("Hi") == tok.encode("Hi")
    assert tok.encode("A", add_special_tokens=True) == [1, *tok.encode("A"), 2]
    assert tok.decode_bytes([1, *tok.encode(b"abc"), 2]) == b"abc"
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode_bytes(tok.encode(b"\xff")) == b"\xff"


@pytest.mark.parametrize("bad", [["-1"], [-1], [65536], [True], [1.1]])
def test_reject_invalid_ids(monkeypatch, bad):
    import ethosoftlib.nedo as nedo
    monkeypatch.setattr(nedo, "_load_library", lambda _: FakeLibrary())
    with pytest.raises(ValueError):
        Tokenizer().decode(bad)


def test_provider_is_model_free(monkeypatch):
    from ethosoftlib import providers
    import ethosoftlib.nedo as nedo
    monkeypatch.setattr(nedo, "_load_library", lambda _: FakeLibrary())
    assert providers.describe("tokenizer", "nedo").factory == "ethosoftlib.nedo:Tokenizer"
    assert providers.create("tokenizer", "nedo").encode("Merhaba")


def test_bad_vocab_rejected(monkeypatch):
    import ethosoftlib.nedo as nedo
    fake = FakeLibrary()
    fake.nedo004_vocab_sha256 = FakeFunc(lambda: b"invalid")
    monkeypatch.setattr(nedo, "_load_library", lambda _: fake)
    with pytest.raises(NedoTokenizerError, match="Wrong"):
        Tokenizer()


def test_missing_native_library_message(tmp_path):
    with pytest.raises(NedoLibraryNotFound, match="Unable to load"):
        Tokenizer(library_path=tmp_path / "missing-nedo.so")


def test_mercans_optional_alias():
    from ethosoftlib.mercan import NedoTokenizer as Export
    assert Export is Tokenizer


def test_pyproject_version_is_consistent():
    from ethosoftlib import __version__
    assert __version__ == "0.3.0"
