from pathlib import Path
import pytest

from ethosoftlib.mercan.native import ModelParams, ContextParams, load_library, MercanLibraryNotFound


def test_ctypes_layout_matches_native_header():
    import ctypes
    assert ctypes.sizeof(ModelParams) == 8
    assert ctypes.sizeof(ContextParams) == 16


def test_explicit_missing_library_is_actionable(tmp_path: Path):
    with pytest.raises(MercanLibraryNotFound, match="Could not load libmercan"):
        load_library(tmp_path / "libmercan-missing.so")


@pytest.mark.parametrize("version,model_size,context_size", [(2, 8, 16), (1, 32, 16), (1, 8, 32)])
def test_reject_incompatible_native_abi(version, model_size, context_size):
    from types import SimpleNamespace
    from ethosoftlib.mercan.native import _validate_abi, MercanError
    library = SimpleNamespace(
        mercan_abi_version=lambda: version,
        mercan_model_params_size=lambda: model_size,
        mercan_context_params_size=lambda: context_size,
    )
    with pytest.raises(MercanError, match="Incompatible"):
        _validate_abi(library)


def test_native_abi_handshake_and_legacy_policy():
    from types import SimpleNamespace
    from ethosoftlib.mercan.native import _validate_abi, MercanError
    _validate_abi(SimpleNamespace(mercan_abi_version=lambda: 1,
                                mercan_model_params_size=lambda: 8,
                                mercan_context_params_size=lambda: 16))
    _validate_abi(SimpleNamespace(mercan_version=lambda: b"0.1.2"))
    with pytest.raises(MercanError, match="no ABI"):
        _validate_abi(SimpleNamespace(mercan_version=lambda: b"9.0"))
    with pytest.raises(MercanError, match="incomplete"):
        _validate_abi(SimpleNamespace(mercan_abi_version=lambda: 1))


def test_generation_requires_new_native_runtime():
    from types import SimpleNamespace
    from ethosoftlib.mercan.native import bind_generation_api, MercanError
    with pytest.raises(MercanError, match="Rebuild MercanRuntime"):
        bind_generation_api(SimpleNamespace())


def test_old_runtime_is_released_before_model_loading(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from ethosoftlib.mercan import Model, MercanError
    runtime = MagicMock()
    runtime.require_generation_api.side_effect = MercanError("upgrade native runtime")
    monkeypatch.setattr("ethosoftlib.mercan.model.MercanRuntime", lambda **kwargs: runtime)
    path = tmp_path / "model.mercan"
    path.write_bytes(b"not loaded")
    with pytest.raises(MercanError, match="upgrade"):
        Model.from_pretrained(path)
    runtime.close.assert_called_once()
    runtime.load_model.assert_not_called()
