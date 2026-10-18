"""Exercise the real callback bridge with a fake native ABI, no model required."""
import ctypes as C
from threading import get_ident
from types import SimpleNamespace

import pytest

from ethosoftlib.mercan.generation import NativeGeneration
from ethosoftlib.mercan.native import ContextParams, GenerationParams, MercanError


class FakeLibrary:
    def __init__(self, error=False):
        self.error = error
        self.finish = 0
        self.cancel = False
        self.freed = False
        self.worker_id = self.error_reader_id = None

    def mercan_generation_default_params_v1(self, params, size):
        C.cast(params, C.POINTER(GenerationParams)).contents.struct_size = size
        return 0

    def mercan_chat_begin_v1(self, model, messages, count, *args):
        self.messages = [(messages[i].role, messages[i].content) for i in range(count)]
        return 123

    def mercan_generate_begin_v1(self, model, prompt, length, *args):
        self.prompt = prompt[:length]
        return 123

    def mercan_generation_run_v1(self, handle, callback, user):
        self.worker_id = get_ident()
        if self.error:
            self.finish = 4
            return -1
        for raw in (b"\xc4", b"\xb0", b"hello", b"world"):
            buffer = C.create_string_buffer(raw)
            if self.cancel or not callback(C.cast(buffer, C.c_void_p), len(raw), user):
                self.finish = 3
                return 1
        self.finish = 2
        return 0

    def mercan_last_error(self):
        self.error_reader_id = get_ident()
        return b"native worker error"

    def mercan_generation_cancel_v1(self, _):
        self.cancel = True

    def mercan_generation_finish_reason_v1(self, _):
        return self.finish

    def mercan_generation_free_v1(self, _):
        self.freed = True


def make_request(monkeypatch, error=False, **options):
    monkeypatch.setattr("ethosoftlib.mercan.generation.bind_generation_api", lambda lib: None)
    lib = FakeLibrary(error)
    model = SimpleNamespace(_runtime=SimpleNamespace(_lib=lib), _handle=1, _contexts=0,
                            _check=lambda: None,
                            _context_params=lambda **kwargs: ContextParams(1024, 32, 2, 2))
    request = NativeGeneration(model, messages=[{"role": "user", "content": "Türkçe"}], **options)
    return request, model, lib


def test_native_bytes_bridge_and_owned_context(monkeypatch):
    request, model, lib = make_request(monkeypatch)
    with request:
        assert model._contexts == 1
        assert b"".join(request.stream()).decode() == "İhelloworld"
        assert lib.messages == [(b"user", "Türkçe".encode())]
        assert not request.cancelled
        with pytest.raises(MercanError, match="already run"):
            list(request.stream())
    assert lib.freed and model._contexts == 0
    request.close()


def test_partial_stream_cancels_joins_and_releases(monkeypatch):
    request, model, lib = make_request(monkeypatch)
    with request:
        stream = request.stream()
        assert next(stream) == b"\xc4"
        with pytest.raises(MercanError, match="Close the native generation stream"):
            request.close()
        stream.close()
        assert not request._running
        assert lib.cancel
    assert lib.freed and model._contexts == 0


def test_worker_errors_are_read_on_the_native_thread(monkeypatch):
    request, model, lib = make_request(monkeypatch, error=True)
    with request, pytest.raises(MercanError, match="native worker error"):
        list(request.stream())
    assert lib.error_reader_id == lib.worker_id != get_ident()
    assert lib.freed and model._contexts == 0


def test_callback_exception_is_relayed_without_crossing_c_abi(monkeypatch):
    request, model, lib = make_request(monkeypatch)
    def fail(*args):
        raise RuntimeError("callback conversion failed")
    monkeypatch.setattr("ethosoftlib.mercan.generation.C.string_at", fail)
    with request, pytest.raises(RuntimeError, match="callback conversion failed"):
        list(request.stream())
    assert lib.freed and model._contexts == 0


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5])
def test_seed_is_validated_before_native_call(monkeypatch, seed):
    with pytest.raises(ValueError, match="seed"):
        make_request(monkeypatch, seed=seed)
