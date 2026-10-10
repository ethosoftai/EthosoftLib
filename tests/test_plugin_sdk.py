import hashlib
import json
from pathlib import Path
import zipfile
import pytest

from ethosoftlib.plugins import PluginValidationError, read_manifest, package_plugin, scaffold


def write_plugin(tmp_path: Path, **overrides):
    path = tmp_path / "mercan-plugin.json"
    binary = tmp_path / "build" / "plugin.so"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"test plugin bytes")
    data = {
        "name": "anka_plugin", "version": "1.2.3",
        "abi_version": 1, "architectures": ["anka"],
        "binaries": {"linux_x86_64": "build/plugin.so"},
        "sha256": {"linux_x86_64": hashlib.sha256(binary.read_bytes()).hexdigest()},
    }
    data.update(overrides)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_manifest_check_and_package(tmp_path):
    path = write_plugin(tmp_path)
    assert read_manifest(path)["name"] == "anka_plugin"
    archive = package_plugin(path, tmp_path / "packed" / "anka.zip")
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == ["build/plugin.so", "mercan-plugin.json"]


def test_binary_integrity_failure(tmp_path):
    path = write_plugin(tmp_path)
    (tmp_path / "build" / "plugin.so").write_bytes(b"tampered")
    with pytest.raises(PluginValidationError, match="SHA256"):
        read_manifest(path)


@pytest.mark.parametrize("change", [
    {"abi_version": 2}, {"abi_version": True},
    {"binaries": {"linux_x86_64": "../plugin.so"}},
    {"binaries": {"linux_x86_64": "/tmp/x.dll"}},
    {"binaries": {"unknown_target": "build/plugin.so"}},
    {"architectures": ["anka", "anka"]}, {"version": "latest"},
])
def test_manifest_rejects_invalid_settings(tmp_path, change):
    with pytest.raises(PluginValidationError):
        read_manifest(write_plugin(tmp_path, **change))


def test_scaffold_is_safe_and_explicit(tmp_path):
    target = scaffold("mylm", tmp_path / "mylm")
    cpp = (target / "plugin.cpp").read_text()
    cmake = (target / "CMakeLists.txt").read_text()
    assert "mercan_plugin_entry_v1" in cpp
    assert '"mylm"' in cpp
    assert "return -1" in cpp
    assert ("$" + "{MERCAN_SDK_INCLUDE}") in cmake
    with pytest.raises(FileExistsError):
        scaffold("mylm", target)
    with pytest.raises(PluginValidationError):
        scaffold("../evil", tmp_path / "evil")


def test_plugin_load_requires_explicit_call(tmp_path, monkeypatch):
    from ethosoftlib.mercan.runtime import MercanRuntime
    import ethosoftlib.mercan.native as native
    binary = tmp_path / "libtest.so"
    binary.write_bytes(b"mock")
    class DummyLibrary:
        def mercan_plugin_load_v1(self, path):
            return 0
        def mercan_plugin_last_error_v1(self):
            return b""
    monkeypatch.setattr(native, "_bind", lambda *args: None)
    runtime = object.__new__(MercanRuntime)
    runtime._lib = DummyLibrary()
    runtime._models = 0
    runtime._closed = False
    assert runtime.load_plugin(binary)
    runtime._models = 1
    from ethosoftlib.mercan.native import MercanError
    with pytest.raises(MercanError, match="before loading models"):
        runtime.load_plugin(binary)
