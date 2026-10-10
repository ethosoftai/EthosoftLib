"""Test context token accounting and KV reset without real model binaries."""
from ethosoftlib.mercan.runtime import MercanContext


class DummyLib:
    def __init__(self):
        self.freed = []
        self.created = 0

    def mercan_context_size(self, handle):
        return 20

    def mercan_context_create(self, model, params):
        self.created += 1
        return 100 + self.created

    def mercan_context_free(self, handle):
        self.freed.append(handle)

    def mercan_decode(self, context, ids, length):
        return 0


class DummyRuntime:
    def __init__(self):
        self._lib = DummyLib()

    def _check(self):
        pass


class DummyModel:
    _handle = 20

    def __init__(self):
        self._runtime = DummyRuntime()
        self._contexts = 1

    def _check(self):
        pass


def test_kv_state_usage_and_reset():
    model = DummyModel()
    ctx = MercanContext(model, 100, 2, object())
    assert ctx.tokens_used == 0
    assert ctx.tokens_remaining == 20
    ctx.decode([1, 2, 3, 4, 5])
    assert ctx.tokens_used == 5
    assert ctx.tokens_remaining == 15
    ctx.reset()
    assert ctx.tokens_used == 0
    assert ctx.tokens_remaining == 20
    assert model._runtime._lib.freed == [100]
    ctx.close()
    assert model._runtime._lib.freed == [100, 101]
    assert model._contexts == 0


def test_reset_keeps_old_context_if_allocation_fails():
    model = DummyModel()
    ctx = MercanContext(model, 100, 2, object())
    model._runtime._lib.mercan_context_create = lambda *args: None
    from ethosoftlib.mercan.native import MercanError
    model._runtime._lib.mercan_last_error = lambda: b"native allocation failed"
    try:
        ctx.reset()
        assert False, "reset must raise"
    except MercanError as error:
        assert "allocation failed" in str(error)
    assert ctx._handle == 100
    assert model._runtime._lib.freed == []
    ctx.close()
