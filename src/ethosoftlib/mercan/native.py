"""ctypes declarations corresponding to MercanRuntime runtime/libmercan/include/mercan.h.

The ABI is loaded on demand. Platform-specific wheels may bundle libmercan.
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


class GenerationParams(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("max_tokens", C.c_int32),
        ("temperature", C.c_float), ("top_k", C.c_int32),
        ("top_p", C.c_float), ("repeat_penalty", C.c_float),
        ("repeat_last_n", C.c_int32), ("seed", C.c_uint32),
        ("use_seed", C.c_uint32),
    ]


class ChatMessage(C.Structure):
    _fields_ = [("role", C.c_char_p), ("content", C.c_char_p),
                ("content_len", C.c_size_t)]


GenerationCallback = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_size_t, C.c_void_p)


def bind_generation_api(library: C.CDLL) -> None:
    """Bind the versioned native generation engine, with no Python fallback."""
    if not hasattr(library, "mercan_generation_api_version"):
        raise MercanError(
            "Native generation API v1 is required. Rebuild MercanRuntime >=0.2.0 "
            "with MERCAN_BUILD_SHARED=ON or install an updated native wheel."
        )
    _bind(library, "mercan_generation_api_version", C.c_uint32)
    _bind(library, "mercan_generation_params_size_v1", C.c_size_t)
    if (library.mercan_generation_api_version() != 1
            or library.mercan_generation_params_size_v1() != C.sizeof(GenerationParams)):
        raise MercanError("Incompatible libmercan generation API/layout")
    ptr = C.c_void_p
    _bind(library, "mercan_generation_default_params_v1", C.c_int,
          C.POINTER(GenerationParams), C.c_size_t)
    _bind(library, "mercan_generate_begin_v1", ptr, ptr, C.c_char_p, C.c_size_t,
          ContextParams, C.POINTER(GenerationParams))
    _bind(library, "mercan_chat_begin_v1", ptr, ptr, C.POINTER(ChatMessage), C.c_size_t,
          C.c_char_p, C.c_char_p, C.c_char_p, ContextParams, C.POINTER(GenerationParams))
    _bind(library, "mercan_generation_run_v1", C.c_int, ptr, GenerationCallback, ptr)
    _bind(library, "mercan_generation_cancel_v1", None, ptr)
    _bind(library, "mercan_generation_finish_reason_v1", C.c_int, ptr)
    _bind(library, "mercan_generation_free_v1", None, ptr)


def bind_session_api(library: C.CDLL) -> None:
    if not hasattr(library, "mercan_session_api_version"):
        raise MercanError("Native session API v1 is required. Rebuild MercanRuntime >=0.4.0 "
                          "or install an updated native wheel.")
    _bind(library, "mercan_session_api_version", C.c_uint32)
    if library.mercan_session_api_version() != 1:
        raise MercanError("Incompatible libmercan session API")
    ptr = C.c_void_p
    _bind(library, "mercan_session_create_v1", ptr, ptr, ContextParams)
    _bind(library, "mercan_session_generate_begin_v1", ptr, ptr, C.c_char_p, C.c_size_t,
          C.POINTER(GenerationParams))
    _bind(library, "mercan_session_chat_begin_v1", ptr, ptr, C.POINTER(ChatMessage), C.c_size_t,
          C.c_char_p, C.c_char_p, C.c_char_p, C.POINTER(GenerationParams))
    _bind(library, "mercan_generation_commit_session_v1", C.c_int, ptr)
    _bind(library, "mercan_session_reset_v1", C.c_int, ptr)
    _bind(library, "mercan_session_stats_json_v1", C.c_char_p, ptr)
    _bind(library, "mercan_session_free_v1", C.c_int, ptr)


def bind_device_api(library: C.CDLL) -> None:
    """Require device discovery and verified model placement API v1."""
    if not hasattr(library, "mercan_device_api_version"):
        raise MercanError(
            "Device API v1 is required. Rebuild MercanRuntime >=0.3.0 "
            "with MERCAN_BUILD_SHARED=ON or install an updated native wheel."
        )
    _bind(library, "mercan_device_api_version", C.c_uint32)
    if library.mercan_device_api_version() != 1:
        raise MercanError("Incompatible libmercan device API")
    _bind(library, "mercan_cuda_available_v1", C.c_bool)
    _bind(library, "mercan_runtime_devices_json_v1", C.c_char_p)
    _bind(library, "mercan_model_load_device_v1", C.c_void_p,
          C.c_char_p, ModelParams, C.c_char_p)
    _bind(library, "mercan_model_device_v1", C.c_char_p, C.c_void_p)
    _bind(library, "mercan_model_device_json_v1", C.c_char_p, C.c_void_p)


def load_library(path: str | os.PathLike[str] | None = None) -> C.CDLL:
    """Load explicit, package-bundled, then system native library."""
    explicit = path or os.environ.get("ETHOSOFT_MERCAN_LIBRARY") or os.environ.get("MERCAN_LIBRARY_PATH")
    if explicit:
        candidates = [os.fspath(explicit)]
    else:
        folder = Path(__file__).parent / "lib"
        candidates = [
            str(folder / name) for name in ("libmercan.so", "libmercan.dylib", "mercan.dll")
            if (folder / name).is_file()
        ]
        system = find_library("mercan")
        if system:
            candidates.append(system)
    if not candidates:
        raise MercanLibraryNotFound(
            "libmercan not found. Build MercanRuntime with MERCAN_BUILD_SHARED=ON, "
            "pass library_path=..., set ETHOSOFT_MERCAN_LIBRARY, or install a "
            "platform-specific native wheel."
        )
    failures = []
    for library in candidates:
        try:
            return C.CDLL(str(Path(library).expanduser()) if explicit else library)
        except OSError as exc:
            failures.append(f"{library}: {exc}")
    raise MercanLibraryNotFound("Could not load libmercan: " + "; ".join(failures))


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
    _validate_abi(library)
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


def _validate_abi(library: C.CDLL) -> None:
    """Check layouts before passing structs by value; retain legacy 0.1.2.

    Legacy builds lack the handshake and cannot have their layout verified.
    New builds must expose the complete ABI 1 handshake.
    """
    names = ("mercan_abi_version", "mercan_model_params_size",
             "mercan_context_params_size")
    present = [hasattr(library, name) for name in names]
    if not any(present):
        if library.mercan_version() != b"0.1.2":
            raise MercanError("libmercan has no ABI queries and is not a supported 0.1.2 build")
        return
    if not all(present):
        raise MercanError("libmercan exposes an incomplete ABI handshake")
    _bind(library, names[0], C.c_uint32)
    for name in names[1:]:
        _bind(library, name, C.c_size_t)
    actual = tuple(getattr(library, name)() for name in names)
    expected = (1, C.sizeof(ModelParams), C.sizeof(ContextParams))
    if actual != expected:
        raise MercanError(
            f"Incompatible libmercan ABI/layout: got {actual}, expected {expected}. "
            "Install matching EthosoftLib and MercanRuntime builds."
        )


def native_error(library: C.CDLL, fallback: str) -> MercanError:
    raw = library.mercan_last_error()
    return MercanError(raw.decode("utf-8", "replace") if raw else fallback)
