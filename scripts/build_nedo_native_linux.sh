#!/usr/bin/env bash
# Build *original* NDSRF004 Rust FFI and stage it for an optional Linux wheel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERCAN_REF="${MERCAN_REF:-a4693b6ce17c6c934a352f81b2183169ab903651}"
MERCAN_SRC="${MERCAN_SRC:-$ROOT/.native-work/MercanRuntime}"
if [[ ! -f "$MERCAN_SRC/runtime/nedo004-ffi/Cargo.toml" ]]; then
  mkdir -p "$(dirname "$MERCAN_SRC")"
  git clone https://github.com/Ahmet2001/MercanRuntime.git "$MERCAN_SRC"
fi
git -C "$MERCAN_SRC" fetch origin "$MERCAN_REF"
git -C "$MERCAN_SRC" checkout --detach "$MERCAN_REF"
cargo build --release --manifest-path "$MERCAN_SRC/runtime/nedo004-ffi/Cargo.toml"
LIB="$MERCAN_SRC/runtime/nedo004-ffi/target/release/libnedo004_ffi.so"
test -s "$LIB"
DEST="$ROOT/src/ethosoftlib/nedo/lib"
mkdir -p "$DEST"
cp "$LIB" "$DEST/libnedo004_ffi.so"
echo "Staged original NDSRF004 Rust shared library at $DEST/libnedo004_ffi.so"
