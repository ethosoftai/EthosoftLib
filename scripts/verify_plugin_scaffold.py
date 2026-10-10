"""Compile a scaffolded architecture plugin using the actual Mercan C++ public SDK."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from ethosoftlib.plugins import read_manifest, package_plugin, scaffold

root = Path(__file__).resolve().parent.parent
sdk_include = root / ".native-work" / "MercanRuntime" / "runtime" / "libmercan" / "include"
if not (sdk_include / "mercan_plugin.h").exists():
    raise SystemExit(f"Mercan public SDK header missing: {sdk_include}")

output = root / ".native-work" / "architecture-sample"
source = scaffold("samplearch", output / "src")
build = output / "build"
subprocess.run([
    "cmake", "-S", str(source), "-B", str(build),
    "-DMERCAN_SDK_INCLUDE=" + str(sdk_include),
], check=True)
subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True)
extensions = {".so", ".dll", ".dylib"}
candidates = [
    p for p in build.rglob("*") if p.is_file()
    and p.suffix in extensions and "mercan_arch_samplearch" in p.name
]
if len(candidates) != 1:
    raise SystemExit(f"Expected one compiled plugin library; got {candidates}")
binary = candidates[0]
machine = platform.machine().lower()
if sys.platform == "win32":
    target = "windows_amd64"
elif sys.platform == "darwin":
    target = "macos_arm64" if machine in ("arm64", "aarch64") else "macos_x86_64"
else:
    target = "linux_x86_64"
manifest_data = {
    "name": "samplearch",
    "version": "0.1.0",
    "abi_version": 1,
    "architectures": ["samplearch"],
    "binaries": {target: os.path.relpath(binary, source)},
    "sha256": {target: hashlib.sha256(binary.read_bytes()).hexdigest()},
}
# Build-tree binary may be outside source; stage under source before packaging.
dest = source / ("plugin" + binary.suffix)
dest.write_bytes(binary.read_bytes())
manifest_data["binaries"][target] = dest.name
manifest = source / "mercan-plugin.json"
manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
assert read_manifest(source)["abi_version"] == 1
zip_path = package_plugin(manifest, output / "samplearch.zip")
assert zip_path.is_file()
print("Compiled and packaged the actual Mercan ABI v1 architecture plugin:", zip_path)
