"""Cross-OS native wheel validation; install the wheel into a clean venv."""
from __future__ import annotations
import glob
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv
import zipfile

wheels = glob.glob("dist/*.whl")
if len(wheels) != 1:
    raise SystemExit(f"Expected exactly one wheel, got {wheels}")
wheel = wheels[0]
if "none-any" in wheel:
    raise SystemExit("Native wheel cannot have a platform-independent tag")
with zipfile.ZipFile(wheel) as archive:
    members = archive.namelist()
    if not any("/nedo/lib/" in name and name.endswith((".so", ".dylib", ".dll")) for name in members):
        raise SystemExit("Native Nedo binary missing from wheel")
with tempfile.TemporaryDirectory() as work:
    env = Path(work) / "venv"
    venv.create(env, with_pip=True)
    python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(Path(wheel).resolve())], check=True)
    code = (
        "from ethosoftlib.nedo import Tokenizer; "
        "t=Tokenizer(); s='İstanbul şğü ve 😀'; "
        "assert t.decode(t.encode(s)) == s; "
        "assert t.vocab_size == 32000; "
        "print('Native wheel verified', t.vocab_sha256)"
    )
    clean = os.environ.copy()
    clean.pop("ETHOSOFT_NEDO_LIBRARY", None)
    subprocess.run([str(python), "-c", code], cwd=work, env=clean, check=True)
print("Verified:", wheel)
