"""Verify combined native Mercan+Nedo desktop wheel in a fresh Python environment."""
from __future__ import annotations
import glob
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile

wheels = glob.glob("dist/*.whl")
if len(wheels) != 1 or "none-any" in wheels[0]:
    raise SystemExit(f"Expected one OS-specific wheel: {wheels}")
wheel = Path(wheels[0]).resolve()
with zipfile.ZipFile(wheel) as z:
    members = z.namelist()
    if not any("/mercan/lib/" in item and item.endswith((".so", ".dll", ".dylib")) for item in members):
        raise SystemExit("libmercan not found in combined native wheel")
    if not any("/nedo/lib/" in item and item.endswith((".so", ".dll", ".dylib")) for item in members):
        raise SystemExit("NedoTokenizer not found in combined native wheel")
with tempfile.TemporaryDirectory() as directory:
    env_dir = Path(directory) / "env"
    venv.create(env_dir, with_pip=True)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
    cmd = (
        "from ethosoftlib.mercan import MercanRuntime; "
        "from ethosoftlib.nedo import Tokenizer; "
        "t=Tokenizer(); s='İstanbul Türkçe 😀'; "
        "assert t.decode(t.encode(s)) == s; "
        "with_runtime=MercanRuntime(); "
        "assert with_runtime.version; "
        "print('libmercan:', with_runtime.version, 'Nedo SHA:', t.vocab_sha256); "
        "with_runtime.close()"
    )
    clean = os.environ.copy()
    clean.pop("ETHOSOFT_MERCAN_LIBRARY", None)
    clean.pop("ETHOSOFT_NEDO_LIBRARY", None)
    subprocess.run([str(python), "-c", cmd], cwd=directory, env=clean, check=True)
print("Combined native wheel verified:", wheel)
