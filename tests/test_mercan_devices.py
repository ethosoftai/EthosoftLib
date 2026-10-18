"""Device selection must fail before downloads and never pretend GPU offload."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ethosoftlib.mercan import MercanError, MercanRuntime, Model
from ethosoftlib.mercan.native import bind_device_api


def runtime(compiled=False, available=False):
    obj = object.__new__(MercanRuntime)
    obj._closed = False
    obj.capabilities = lambda: {"cuda_compiled": compiled, "cuda_available": available}
    return obj


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("ETHOSOFT_MERCAN_GPU_LAYERS", raising=False)


@pytest.mark.parametrize("compiled,available,expected", [
    (False, False, ("cpu", 0)), (True, False, ("cpu", 0)),
    (True, True, ("cuda", -1)),
])
def test_auto_selects_only_usable_cuda(compiled, available, expected):
    assert runtime(compiled, available).select_device() == expected


def test_forced_cpu_ignores_available_cuda_and_environment(monkeypatch):
    monkeypatch.setenv("ETHOSOFT_MERCAN_GPU_LAYERS", "-1")
    assert runtime(True, True).select_device("cpu") == ("cpu", 0)
    assert runtime(True, True).select_device("auto", 0) == ("cpu", 0)


@pytest.mark.parametrize("device,layers", [("cuda", None), ("auto", -1), ("auto", 3)])
def test_explicit_gpu_requires_compiled_and_usable_cuda(device, layers):
    with pytest.raises(MercanError, match="built without CUDA"):
        runtime().select_device(device, layers)
    with pytest.raises(MercanError, match="NVIDIA driver"):
        runtime(True, False).select_device(device, layers)
    assert runtime(True, True).select_device(device, layers) == ("cuda", layers if layers is not None else -1)


@pytest.mark.parametrize("device,layers", [
    ("invalid", None), ("cpu", -1), ("cuda", 0), ("auto", -2),
    ("auto", True), ("auto", 1.5), ("auto", 2147483648),
])
def test_invalid_device_options(device, layers):
    with pytest.raises(ValueError):
        runtime(True, True).select_device(device, layers)


def test_environment_is_an_explicit_offload_request(monkeypatch):
    monkeypatch.setenv("ETHOSOFT_MERCAN_GPU_LAYERS", "4")
    with pytest.raises(MercanError, match="built without CUDA"):
        runtime().select_device()
    assert runtime(True, True).select_device() == ("cuda", 4)
    assert runtime(True, True).select_device("auto", 0) == ("cpu", 0)
    monkeypatch.setenv("ETHOSOFT_MERCAN_GPU_LAYERS", "no")
    with pytest.raises(ValueError, match="ETHOSOFT_MERCAN_GPU_LAYERS"):
        runtime().select_device()


def test_device_api_upgrade_message():
    with pytest.raises(MercanError, match="MercanRuntime >=0.3.0"):
        bind_device_api(SimpleNamespace())


def test_cuda_failure_precedes_download_and_releases_runtime(monkeypatch):
    obj = MagicMock()
    obj.select_device.side_effect = MercanError("built without CUDA")
    download = MagicMock()
    monkeypatch.setattr("ethosoftlib.mercan.model.MercanRuntime", lambda **kwargs: obj)
    monkeypatch.setattr("ethosoftlib.mercan.model.resolve_model_file", download)
    with pytest.raises(MercanError, match="built without CUDA"):
        Model.from_pretrained("unused/repo", device="cuda")
    download.assert_not_called()
    obj.load_model.assert_not_called()
    obj.close.assert_called_once()
