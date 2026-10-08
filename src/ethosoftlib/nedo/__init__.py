"""Standalone exact NDSRF004 tokenizer backed by the original Rust implementation."""
from __future__ import annotations

import ctypes as C
from ctypes.util import find_library
import os
from pathlib import Path
from typing import Iterable

EXPECTED_SHA256 = "72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600"
NEDO_VOCAB_SIZE = 32_000


class NedoTokenizerError(RuntimeError):
    """Original Rust NedoTokenizer C ABI returned an error."""


class NedoLibraryNotFound(NedoTokenizerError):
    """Native NDSRF004 tokenizer shared library is unavailable."""


def _load_library(library_path: str | os.PathLike[str] | None = None) -> C.CDLL:
    explicit = library_path or os.environ.get("ETHOSOFT_NEDO_LIBRARY")
    if explicit:
        candidates = [os.fspath(explicit)]
    else:
        folder = Path(__file__).resolve().parent / "lib"
        candidates = [
            str(folder / n) for n in
            ("libnedo004_ffi.so", "libnedo004_ffi.dylib", "nedo004_ffi.dll")
            if (folder / n).is_file()
        ]
        system = find_library("nedo004_ffi")
        if system:
            candidates.append(system)
    if not candidates:
        raise NedoLibraryNotFound(
            "Standalone NedoTokenizer library not found; build "
            "MercanRuntime/runtime/nedo004-ffi with cargo, then set "
            "ETHOSOFT_NEDO_LIBRARY or pass library_path=."
        )
    failures = []
    for name in candidates:
        try:
            return C.CDLL(str(Path(name).expanduser()) if explicit else name)
        except OSError as exc:
            failures.append(f"{name}: {exc}")
    raise NedoLibraryNotFound("Unable to load native NedoTokenizer: " + "; ".join(failures))


def _bind(lib: C.CDLL) -> C.CDLL:
    try:
        lib.nedo004_vocab_sha256.argtypes = []
        lib.nedo004_vocab_sha256.restype = C.c_char_p
        lib.nedo004_vocab_size.argtypes = []
        lib.nedo004_vocab_size.restype = C.c_uint32
        uint16 = C.POINTER(C.c_uint16)
        size = C.POINTER(C.c_size_t)
        lib.nedo004_encode_copy.argtypes = [
            C.c_char_p, C.c_size_t, uint16, C.c_size_t, size,
        ]
        lib.nedo004_encode_copy.restype = C.c_int32
        lib.nedo004_decode_copy.argtypes = [
            uint16, C.c_size_t, C.c_void_p, C.c_size_t, size,
        ]
        lib.nedo004_decode_copy.restype = C.c_int32
    except AttributeError as exc:
        raise NedoTokenizerError(
            "Native tokenizer is missing new NDSRF004 copy ABI symbols."
        ) from exc
    return lib


class Tokenizer:
    """Model-free exact NDSRF004 tokenizer for Turkish, code, and mixed text.

    The Rust morphological tokenizer and 32k vocabulary are authoritative.
    No alternate Python tokenization algorithm is implemented.
    """
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, library_path: str | os.PathLike[str] | None = None) -> None:
        self._lib = _bind(_load_library(library_path))
        hash_bytes = self._lib.nedo004_vocab_sha256()
        if not hash_bytes:
            raise NedoTokenizerError("NDSRF004 vocabulary did not initialize")
        self._vocab_sha = hash_bytes.decode("ascii")
        if self._vocab_sha != EXPECTED_SHA256:
            raise NedoTokenizerError(f"Wrong NDSRF004 vocab SHA: {self._vocab_sha}")
        if self.vocab_size != NEDO_VOCAB_SIZE:
            raise NedoTokenizerError("Expected an exact 32,000-entry vocabulary")

    @property
    def vocab_sha256(self) -> str:
        return self._vocab_sha

    @property
    def vocab_size(self) -> int:
        return int(self._lib.nedo004_vocab_size())

    def encode(self, text: str | bytes, *, add_special_tokens: bool = False) -> list[int]:
        """Encode into NDSRF004 ids, optionally adding document BOS/EOS."""
        if isinstance(text, str):
            raw = text.encode("utf-8")
        elif isinstance(text, bytes):
            raw = text
        else:
            raise TypeError("encode expects str or bytes")
        needed = C.c_size_t(0)
        fn = self._lib.nedo004_encode_copy
        rc = fn(raw, len(raw), None, 0, C.byref(needed))
        if rc not in (0, -4):
            raise NedoTokenizerError(f"Native encode failed (code {rc})")
        if needed.value:
            out = (C.c_uint16 * needed.value)()
            rc = fn(raw, len(raw), out, needed.value, C.byref(needed))
            if rc != 0:
                raise NedoTokenizerError(f"Native encode failed (code {rc})")
            ids = list(out[:needed.value])
        else:
            ids = []
        if add_special_tokens:
            return [self.bos_token_id, *ids, self.eos_token_id]
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Return exact decoded bytes; never silently replace invalid UTF-8."""
        values = list(ids)
        if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= 65535
               for x in values):
            raise ValueError("Token IDs must be uint16 integers")
        array = (C.c_uint16 * len(values))(*values)
        needed = C.c_size_t(0)
        fn = self._lib.nedo004_decode_copy
        rc = fn(array, len(values), None, 0, C.byref(needed))
        if rc not in (0, -4):
            raise NedoTokenizerError(f"Native decode failed (code {rc})")
        if not needed.value:
            return b""
        output = C.create_string_buffer(needed.value)
        rc = fn(array, len(values), output, needed.value, C.byref(needed))
        if rc != 0:
            raise NedoTokenizerError(f"Native decode failed (code {rc})")
        return output.raw[:needed.value]

    def decode(self, ids: Iterable[int], *, errors: str = "strict") -> str:
        return self.decode_bytes(ids).decode("utf-8", errors=errors)

    def tokenize(self, text: str | bytes) -> list[int]:
        return self.encode(text)


NedoTokenizer = Tokenizer
__all__ = ["Tokenizer", "NedoTokenizer", "NedoTokenizerError",
           "NedoLibraryNotFound", "EXPECTED_SHA256", "NEDO_VOCAB_SIZE"]
