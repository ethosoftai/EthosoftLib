from pathlib import Path
import sys
import types

import pytest

from ethosoftlib.mercan.hub import resolve_model_file


def test_local_model_needs_no_hub(tmp_path):
    path = tmp_path / "my.mercan"
    path.write_bytes(b"fake")
    assert resolve_model_file(path) == path.resolve()
    assert resolve_model_file(str(path)) == path.resolve()


def test_remote_model_download_uses_hf_cache(monkeypatch, tmp_path):
    target = tmp_path / "model.mercan"
    target.write_bytes(b"fake")
    observed = {}

    def fake_hf_hub_download(**kwargs):
        observed.update(kwargs)
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(hf_hub_download=fake_hf_hub_download))
    result = resolve_model_file(
        "MercanAI/Mercan-0.8B-SFT", revision="abc123",
        cache_dir=tmp_path, local_files_only=True,
    )
    assert result == target
    assert observed["repo_id"] == "MercanAI/Mercan-0.8B-SFT"
    assert observed["filename"] == "model.mercan"
    assert observed["revision"] == "abc123"
    assert observed["local_files_only"] is True


def test_remote_colon_filename(monkeypatch, tmp_path):
    target = tmp_path / "weights.mercan"
    target.write_bytes(b"fake")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_: str(target)),
    )
    assert resolve_model_file("owner/repo:weights.mercan") == target


@pytest.mark.parametrize("model_id", ["owner", "owner/repo/sub", "bad name/repo",
                                        "owner/repo:../unsafe.mercan"])
def test_reject_unsafe_model_id(model_id):
    with pytest.raises(ValueError):
        resolve_model_file(model_id)


def test_missing_local_is_not_remote(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_file(tmp_path / "missing.mercan")


def test_public_import_surface_and_core_independence():
    from ethosoftlib import providers
    from ethosoftlib.mercan import Model
    assert callable(Model.from_pretrained)
    assert providers.describe("inference", "mercan").name == "mercan"
