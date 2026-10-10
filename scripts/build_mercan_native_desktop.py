"""Cross-platform Mercan CPU shared-library build for native Python wheels.

Pinned MercanRuntime + pinned patched llama.cpp; no third-party executables
are automatically downloaded at import or pip install time.
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / ".native-work"
MERCAN = WORK / "MercanRuntime"
LLAMA = MERCAN / "build" / "_deps" / "llama-src"
MERCAN_COMMIT = os.environ.get(
    "MERCAN_NATIVE_REF", "f8c8414845dcf52ad1344a477c1cc88ade83c595"
)
LLAMA_COMMIT = "e71b80510c848c00175924ecf3c40333ccae8eb5"


def run(*args: object, cwd: Path | None = None) -> None:
    cmd = [str(value) for value in args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


if not (MERCAN / ".git").exists():
    WORK.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "https://github.com/Ahmet2001/MercanRuntime.git", MERCAN)
run("git", "fetch", "origin", MERCAN_COMMIT, cwd=MERCAN)
run("git", "checkout", "--detach", MERCAN_COMMIT, cwd=MERCAN)
if not (LLAMA / ".git").exists():
    LLAMA.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "https://github.com/ggml-org/llama.cpp.git", LLAMA)
run("git", "fetch", "origin", LLAMA_COMMIT, cwd=LLAMA)
run("git", "checkout", "--detach", LLAMA_COMMIT, cwd=LLAMA)
run("git", "reset", "--hard", LLAMA_COMMIT, cwd=LLAMA)
run("git", "clean", "-fdx", cwd=LLAMA)

vendor = LLAMA / "vendor"
vendor.mkdir(exist_ok=True)
shutil.copytree(MERCAN / "runtime" / "nedo004-ffi", vendor / "nedo004-ffi")
shutil.copytree(MERCAN / "runtime" / "NedoTokenizer", vendor / "NedoTokenizer")
run("cargo", "build", "--release", "--manifest-path", vendor / "nedo004-ffi" / "Cargo.toml")
run("git", "apply", MERCAN / "runtime" / "nedolm" / "nedolm-llama.patch", cwd=LLAMA)

source_models = LLAMA / "src" / "models"
for source, target in (
    (MERCAN / "runtime" / "nedolm" / "nedolm.cpp", source_models / "nedolm.cpp"),
    (MERCAN / "runtime" / "libmercan" / "include" / "mercan_tensor.h", source_models / "mercan_tensor.h"),
    (MERCAN / "runtime" / "libmercan" / "include" / "mercan_graph.h", source_models / "mercan_graph.h"),
    (MERCAN / "runtime" / "libmercan" / "include" / "mercan_kv.h", source_models / "mercan_kv.h"),
    (MERCAN / "runtime" / "nedolm" / "mercan_graph_ggml.hpp", source_models / "mercan_graph_ggml.hpp"),
):
    shutil.copy2(source, target)

build = MERCAN / "build" / "sdk-native-wheel"
run(
    "cmake", "-S", MERCAN, "-B", build,
    "-DCMAKE_BUILD_TYPE=Release", "-DMERCAN_LLAMA_DIR=" + str(LLAMA),
    "-DMERCAN_BUILD_SHARED=ON", "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF",
    "-DGGML_METAL=OFF", "-DLLAMA_CURL=OFF",
)
run("cmake", "--build", build, "--target", "libmercan",
    "--config", "Release", "--parallel", os.environ.get("MERCAN_JOBS", "2"))

filename = (
    "mercan.dll" if sys.platform == "win32" else
    "libmercan.dylib" if sys.platform == "darwin" else "libmercan.so"
)
binaries = list(build.rglob(filename))
if len(binaries) != 1:
    raise SystemExit(f"Expected exactly one {filename}, found {binaries}")
destination = ROOT / "src" / "ethosoftlib" / "mercan" / "lib" / filename
destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(binaries[0], destination)
print("Staged Mercan CPU shared library:", destination, destination.stat().st_size)

# The combined desktop SDK wheel carries both the inference runtime and
# standalone NedoTokenizer, avoiding same-tag native wheel conflicts on PyPI.
nedo_name = (
    "nedo004_ffi.dll" if sys.platform == "win32" else
    "libnedo004_ffi.dylib" if sys.platform == "darwin" else "libnedo004_ffi.so"
)
nedo_source = vendor / "nedo004-ffi" / "target" / "release" / nedo_name
if not nedo_source.is_file():
    raise SystemExit(f"Missing exact original Rust tokenizer cdylib: {nedo_source}")
nedo_destination = ROOT / "src" / "ethosoftlib" / "nedo" / "lib" / nedo_name
nedo_destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(nedo_source, nedo_destination)
print("Staged standalone NDSRF004 shared library:", nedo_destination)
