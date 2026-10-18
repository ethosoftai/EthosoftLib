"""Verify bundled Mercan, NedoTokenizer and optional multi-turn chat in a clean venv."""
from __future__ import annotations
import argparse
import glob
import os
from pathlib import Path
import subprocess
import tempfile
import venv
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wheel", type=Path)
parser.add_argument("--model", type=Path, help="Optional local model for a two-turn KV reuse test")
parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
parser.add_argument("--expect-device", choices=("cpu", "cuda"))
args = parser.parse_args()
wheels = [str(args.wheel)] if args.wheel else glob.glob("dist/*.whl")
if len(wheels) != 1 or "none-any" in wheels[0]:
    raise SystemExit(f"Expected one OS-specific wheel: {wheels}")
wheel = Path(wheels[0]).resolve()
with zipfile.ZipFile(wheel) as archive:
    members = archive.namelist()
    for product in ("mercan", "nedo"):
        if not any(f"/{product}/lib/" in item and item.endswith((".so", ".dll", ".dylib")) for item in members):
            raise SystemExit(f"{product} native library missing from combined wheel")
    for notice in ("NedoTokenizer-LICENSE.txt", "ZEMBEREK-NOTICE.txt", "APACHE-2.0.txt"):
        if not any(item.endswith("/nedo/lib/" + notice) for item in members):
            raise SystemExit(f"Missing bundled notice: {notice}")
with tempfile.TemporaryDirectory() as directory:
    env_dir = Path(directory) / "env"
    venv.create(env_dir, with_pip=True)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True)
    code = r"""
import os
from ethosoftlib.mercan import MercanRuntime, Model
from ethosoftlib.nedo import Tokenizer
tok = Tokenizer()
for text in ("İstanbul Türkçe 😀", "def greet(x):\n    return x + 1\n", ""):
    assert tok.decode(tok.encode(text)) == text
assert tok.decode_bytes(tok.encode(b"\xff\x00\x80")) == b"\xff\x00\x80"
with MercanRuntime() as runtime:
    runtime.require_generation_api()
    runtime.require_session_api()
    assert runtime.capabilities()["api_version"] == 1
    print("libmercan:", runtime.version, "Nedo SHA:", tok.vocab_sha256)
model_path = os.environ.get("ETHOSOFT_WHEEL_TEST_MODEL")
if model_path:
    with Model.from_pretrained(model_path, device=os.environ["ETHOSOFT_WHEEL_TEST_DEVICE"],
                               system_prompt="Türkçe kısa cevap ver.", threads=2, n_ctx=512) as model:
        assert model.chat("Merhaba, nasılsın?", temperature=0, max_tokens=8).strip()
        assert model.chat("Adın ne?", temperature=0, max_tokens=8).strip()
        info = model.info()
        assert info["kv_cache"]["context_allocations"] == 1
        assert info["kv_cache"]["reused_tokens"] > 0
        expected = os.environ.get("ETHOSOFT_WHEEL_EXPECT_DEVICE")
        if expected:
            assert info["device"] == expected, info
        with model.session() as other:
            assert other.history == ()
            assert other.chat("Merhaba", temperature=0, max_tokens=8).strip()
        model.reset_history()
        assert model.info()["kv_cache"]["cached_tokens"] == 0
        print("Two-turn chat:", info["device"], info["kv_cache"])
"""
    clean = os.environ.copy()
    for key in ("ETHOSOFT_MERCAN_LIBRARY", "MERCAN_LIBRARY_PATH", "ETHOSOFT_NEDO_LIBRARY",
                "ETHOSOFT_MERCAN_GPU_LAYERS", "PYTHONPATH"):
        clean.pop(key, None)
    clean["ETHOSOFT_WHEEL_TEST_DEVICE"] = args.device
    if args.model:
        clean["ETHOSOFT_WHEEL_TEST_MODEL"] = str(args.model.expanduser().resolve(strict=True))
    else:
        clean.pop("ETHOSOFT_WHEEL_TEST_MODEL", None)
    if args.expect_device:
        clean["ETHOSOFT_WHEEL_EXPECT_DEVICE"] = args.expect_device
    else:
        clean.pop("ETHOSOFT_WHEEL_EXPECT_DEVICE", None)
    subprocess.run([str(python), "-c", code], cwd=directory, env=clean, check=True)
print("Combined native wheel verified:", wheel)
