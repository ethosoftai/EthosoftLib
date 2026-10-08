"""ctypes declarations corresponding to MercanRuntime runtime/libmercan/include/mercan.h.

No vendored or bundled Mercan runtime is assumed. The ABI is loaded on demand.
"""
from __future__ import annotations

import ctypes as C
from ctypes.util import find_library
import os
from pathlib import Path

from ethosoftlib.core.errors import EthosoftError


class MercanError(EthosoftError):
    """Failure reported by the native Mercan runtime."""


class MercanLibraryNotFound(MercanError):
    """The native libmercan shared library could not be loaded."""


class ModelParams(C.Structure):
    _fields_ = [
        ("n_gpu_layers", C.c_int32),
        ("use_mmap", C.c_bool),
        ("check_tensors", C.c_bool),
    ]


class ContextParams(C.Structure):
    _fields_ = [
        ("n_ctx", C.c_uint32),
        ("n_batch", C.c_uint32),
        ("n_threads", C.c_int32),
        ("n_threads_batch", C.c_int32),
    ]


def load_library(path: str | os.PathLike[str] | None = None) -> C.CDLL:
    """Find libmercan, without downloading executable code or native binaries."""
    explicit = path or os.environ.get("ETHOSOFT_MERCAN_LIBRARY") or os.environ.get("MERCAN_LIBRARY_PATH")
    library = os.fspath(explicit) if explicit else find_library("mercan")
    if not library:
        raise MercanLibraryNotFound(
            "libmercan not found. Build/install MercanRuntime with "
            "MERCAN_BUILD_SHARED=ON, then pass library_path=... or set "
            "ETHOSOFT_MERCAN_LIBRARY to the absolute shared-library path."
        )
    try:
        return C.CDLL(str(Path(library).expanduser()) if explicit else library)
    except OSError as exc:
        raise MercanLibraryNotFound(f"Could not load libmercan from {library!r}: {exc}") from exc


def _bind(library: C.CDLL, symbol: str, result: object, *args: object) -> None:
    try:
        function = getattr(library, symbol)
    except AttributeError as exc:
        raise MercanError(f"Missing libmercan C ABI symbol: {symbol}") from exc
    function.restype = result
    function.argtypes = list(args)


def bind_api(library: C.CDLL) -> C.CDLL:
    """Specify return types to prevent pointer truncation in 64-bit Python."""
    ptr = C.c_void_p
    token_ptr = C.POINTER(C.c_int32)
    _bind(library, "mercan_version", C.c_char_p)
    _bind(library, "mercan_last_error", C.c_char_p)
    _bind(library, "mercan_backend_init", None)
    _bind(library, "mercan_backend_free", None)
    _bind(library, "mercan_model_default_params", ModelParams)
    _bind(library, "mercan_context_default_params", ContextParams)
    _bind(library, "mercan_model_load", ptr, C.c_char_p, ModelParams)
    _bind(library, "mercan_model_free", None, ptr)
    _bind(library, "mercan_model_architecture", C.c_char_p, ptr)
    _bind(library, "mercan_model_tokenizer", C.c_char_p, ptr)
    _bind(library, "mercan_context_create", ptr, ptr, ContextParams)
    _bind(library, "mercan_context_free", None, ptr)
    _bind(library, "mercan_tokenize", C.c_int32, ptr, C.c_char_p, C.c_size_t,
          C.c_bool, C.c_bool, token_ptr, C.c_int32)
    _bind(library, "mercan_decode", C.c_int32, ptr, token_ptr, C.c_int32)
    _bind(library, "mercan_logits", C.POINTER(C.c_float), ptr)
    _bind(library, "mercan_vocab_size", C.c_int32, ptr)
    _bind(library, "mercan_context_size", C.c_uint32, ptr)
    _bind(library, "mercan_bos_token", C.c_int32, ptr)
    _bind(library, "mercan_eos_token", C.c_int32, ptr)
    _bind(library, "mercan_pad_token", C.c_int32, ptr)
    _bind(library, "mercan_token_to_piece", C.c_int32, ptr, C.c_int32,
          C.c_void_p, C.c_int32, C.c_bool)
    return library


def native_error(library: C.CDLL, fallback: str) -> MercanError:
    raw = library.mercan_last_error()
    return MercanError(raw.decode("utf-8", "replace") if raw else fallback)
