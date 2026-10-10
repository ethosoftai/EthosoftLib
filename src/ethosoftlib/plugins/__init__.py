"""Declarative Mercan Plugin ABI v1 tools. Validation never loads native code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import zipfile

PLUGIN_ABI_VERSION = 1
PLATFORMS = ("linux_x86_64", "windows_amd64", "macos_arm64", "macos_x86_64")
NAME = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class PluginValidationError(ValueError):
    pass


def _binary(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PluginValidationError("Invalid binary path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.suffix not in (".so", ".dll", ".dylib"):
        raise PluginValidationError("Unsafe binary path")
    full = (root / candidate).resolve()
    if not full.is_relative_to(root.resolve()):
        raise PluginValidationError("Binary path escapes plugin directory")
    return full


def read_manifest(source: str | Path, *, verify: bool = True) -> dict:
    """Validate plugin manifest, ABI v1, platform binaries and SHA256 digests."""
    path = Path(source).resolve()
    if path.is_dir():
        path /= "mercan-plugin.json"
    if path.name != "mercan-plugin.json":
        raise PluginValidationError("Manifest must be mercan-plugin.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PluginValidationError(f"Invalid manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise PluginValidationError("Manifest must be an object")
    if type(data.get("abi_version")) is not int or data["abi_version"] != 1:
        raise PluginValidationError("Only Mercan Plugin ABI v1 is supported")
    if not isinstance(data.get("name"), str) or not NAME.fullmatch(data["name"]):
        raise PluginValidationError("Invalid plugin name")
    if not isinstance(data.get("version"), str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:[-+][\w.-]+)?", data["version"]):
        raise PluginValidationError("Version must be semver")
    arch = data.get("architectures")
    if not isinstance(arch, list) or not arch or any(not isinstance(a, str) or not NAME.fullmatch(a) for a in arch) or len(set(arch)) != len(arch):
        raise PluginValidationError("Invalid architecture identifiers")
    binaries, hashes = data.get("binaries"), data.get("sha256")
    if not isinstance(binaries, dict) or not isinstance(hashes, dict) or not binaries:
        raise PluginValidationError("Missing platform binary map or checksums")
    if set(binaries) != set(hashes) or not set(binaries).issubset(PLATFORMS):
        raise PluginValidationError("Missing SHA256 or unsupported platform")
    for platform, relative in binaries.items():
        binary = _binary(path.parent, relative)
        expected = hashes[platform]
        if not isinstance(expected, str) or not re.fullmatch("[0-9a-fA-F]{64}", expected):
            raise PluginValidationError("Invalid SHA256 digest")
        if verify:
            if not binary.is_file():
                raise PluginValidationError(f"Binary missing: {binary}")
            if hashlib.sha256(binary.read_bytes()).hexdigest() != expected.lower():
                raise PluginValidationError(f"Binary SHA256 mismatch: {binary}")
    return data


def package_plugin(source: str | Path, destination: str | Path) -> Path:
    """Archive only verified manifest-listed binaries, never arbitrary files."""
    path = Path(source).resolve()
    if path.is_dir():
        path /= "mercan-plugin.json"
    data = read_manifest(path)
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for platform, relative in sorted(data["binaries"].items()):
            zf.write(_binary(path.parent, relative), arcname=relative)
        zf.write(path, arcname="mercan-plugin.json")
    return target


def scaffold(name: str, directory: str | Path) -> Path:
    """Generate ABI-v1 registration starter, NOT a completed inference graph."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise PluginValidationError("Invalid plugin name")
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    source = """
#include "mercan_plugin.h"
#include <cstdio>
#include <cstring>
static int probe(const mercan_metadata_v1* m) {
    if (!m || !m->get_string) return -1;
    const char* a = m->get_string(m, "general.architecture");
    return a && std::strcmp(a, "ARCHNAME") == 0 ? 100 : 0;
}
static int validate(const mercan_metadata_v1* m, char*, size_t) {
    return probe(m) > 0 ? 0 : -1;
}
static int graph(const mercan_arch_graph_invocation_v1*, char* err, size_t cap) {
    if (err && cap) std::snprintf(err, cap, "implement graph callback");
    return -1; // See MercanRuntime/examples/plugins/anka/plugin.cpp
}
static const mercan_architecture_v1 arch = {
    MERCAN_ARCH_ABI_VERSION, sizeof(mercan_architecture_v1),
    "ARCHNAME", "Custom ARCHNAME", nullptr,
    MERCAN_ARCH_GRAPH_CALLBACK_V1 | MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES |
    MERCAN_ARCH_TENSOR_ABI_V1 | MERCAN_ARCH_KV_ABI_V1,
    probe, validate, graph
};
static int init(const mercan_plugin_host_v1* h, char* err, size_t cap) {
    if (!h || h->abi_version != 1 || h->struct_size < MERCAN_PLUGIN_HOST_V1_BASE_SIZE || !h->register_architecture) {
        if (err && cap) std::snprintf(err, cap, "incompatible Plugin ABI");
        return -1;
    }
    return h->register_architecture(&arch);
}
static const mercan_plugin_v1 plugin = {
    MERCAN_PLUGIN_ABI_VERSION, sizeof(mercan_plugin_v1),
    "ARCHNAME", "0.1.0", 0, init
};
extern "C" MERCAN_PLUGIN_EXPORT const mercan_plugin_v1* mercan_plugin_entry_v1() {
    return &plugin;
}
""".replace("ARCHNAME", name)
    (directory / "plugin.cpp").write_text(source, encoding="utf-8")
    cmake = (
        'cmake_minimum_required(VERSION 3.16)\n'
        'project(mercan_custom LANGUAGES CXX)\n'
        'set(CMAKE_CXX_STANDARD 17)\n'
        'if(NOT DEFINED MERCAN_SDK_INCLUDE)\n'
        '  message(FATAL_ERROR "Set MERCAN_SDK_INCLUDE to MercanRuntime/runtime/libmercan/include")\n'
        'endif()\n'
        'add_library(mercan_arch_' + name + ' SHARED plugin.cpp)\n'
        'target_include_directories(mercan_arch_' + name + ' PRIVATE "' + '$' + '{MERCAN_SDK_INCLUDE}")\n'
    )
    (directory / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    (directory / "README.md").write_text(
        "This scaffold registers Plugin ABI v1 but cannot run inference until "
        "you implement the Graph ABI callback. Refer to "
        "MercanRuntime/examples/plugins/anka/plugin.cpp for a fully working example.\n",
        encoding="utf-8")
    return directory


__all__ = ["PLUGIN_ABI_VERSION", "PluginValidationError", "read_manifest", "package_plugin", "scaffold"]
