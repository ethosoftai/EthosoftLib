"""Stage original Rust NDSRF004 cdylib in an OS-specific EthosoftLib wheel."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
RUST = ROOT / ".native-work" / "MercanRuntime" / "runtime" / "nedo004-ffi" / "target" / "release"
if sys.platform == "win32":
    filename = "nedo004_ffi.dll"
elif sys.platform == "darwin":
    filename = "libnedo004_ffi.dylib"
else:
    filename = "libnedo004_ffi.so"
source = RUST / filename
destination = ROOT / "src" / "ethosoftlib" / "nedo" / "lib" / filename
if not source.is_file():
    raise SystemExit(f"Missing original Rust NedoTokenizer binary: {source}")
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, destination)
os.environ["ETHOSOFT_NEDO_LIBRARY"] = str(destination)
print("Staged", destination, destination.stat().st_size, "bytes")
