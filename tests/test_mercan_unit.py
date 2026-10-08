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
