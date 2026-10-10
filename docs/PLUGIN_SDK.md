# Mercan Plugin SDK v1: architecture developer kit

This guide applies to the architecture plugin part of EthosoftLib. Other
Ethosoft products do not require Mercan, C++, a GPU or any native binary.

## Architecture plugin workflow

1. Generate a starter project:

   python -m ethosoftlib.plugins scaffold myarch ./myarch

2. Locate the MercanRuntime public SDK headers:

   MercanRuntime/runtime/libmercan/include/

3. Compile the scaffold project:

   cmake -S ./myarch -B ./myarch/build -DMERCAN_SDK_INCLUDE=/path/to/MercanRuntime/runtime/libmercan/include
   cmake --build ./myarch/build --config Release

The starter exports mercan_plugin_entry_v1(), verifies the host ABI and
registers the model architecture name. The build_graph callback explicitly
returns an error until you implement your inference graph. For a working,
separately-built causal transformer with runtime-owned KV cache, use the
Anka reference plugin:
https://github.com/Ahmet2001/MercanRuntime/tree/main/examples/plugins/anka

Available native contracts:
- Mercan Plugin ABI v1: plugin entrypoint and architecture/tokenizer registration
- Mercan Architecture ABI v1: metadata discovery, validation and graph callback
- Mercan Graph / Tensor / KV ABI v1: opaque graph operations, model weights and cache
- NDSRF004 tokenizer standalone Rust ABI: independent of the model loader

## Manifest example

Create myarch/mercan-plugin.json alongside the binary. Include only the
platform targets you have actually built and tested:

    {
      "name": "myarch",
      "version": "0.1.0",
      "abi_version": 1,
      "architectures": ["myarch"],
      "binaries": {"linux_x86_64": "build/libmercan_arch_myarch.so"},
      "sha256": {"linux_x86_64": "<64-char actual lowercase SHA256>"}
    }

The manifest uses a strict, non-executable JSON schema. Supported native
target keys: linux_x86_64, windows_amd64, macos_arm64, macos_x86_64.

Validate locally before running any native plugin:

    python -m ethosoftlib.plugins validate ./myarch/mercan-plugin.json

The validator checks ABI version, architecture identifiers, binary paths
without traversal, SHA-256 and supported OS. It never loads native code.
Package only integrity-checked files:

    python -m ethosoftlib.plugins pack ./myarch/mercan-plugin.json ./myarch.zip

The ZIP contains the manifest and the listed binaries. It is a portable
plugin distribution, not a PyPI wheel. Hashes detect corruption/mismatched
payloads but do NOT establish publisher identity or trust.

## Explicit native plugin loading

After reviewing and trusting the actual native library, call:

    from ethosoftlib.mercan import MercanRuntime

    with MercanRuntime() as runtime:
        loaded = runtime.load_plugin_from_manifest("./myarch/mercan-plugin.json")
        print("Newly loaded:", loaded)
        print("Active plugins:", runtime.list_plugins())
        with runtime.load_model("./my-model.mercan") as model:
            ...

Plugins must be loaded before models. The Mercan runtime keeps loaded
plugin libraries resident for the process lifetime. Loading a plugin is
arbitrary native code execution with the caller's privileges. The SDK
does not auto-install, auto-discover, auto-load, or execute downloaded
plugin archives. Do not load untrusted plugin binaries.

For long-lived applications, pin both the MercanRuntime ABI release and
every plugin's verified binary hash.

## CI coverage

The EthosoftLib cross-platform native matrix builds a generated plugin
against actual MercanRuntime public headers on Linux, Windows and macOS.
It then validates the manifest and assembles a ZIP.
This tests native ABI compilation and packaging, NOT arbitrary third-party
model weights or unimplemented architectures. The Anka runtime regression
in MercanRuntime covers real external transformer graph execution and KV use.
