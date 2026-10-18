#!/usr/bin/env bash
# Build *original* NDSRF004 Rust FFI and stage it for an optional Linux wheel.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERCAN_REF="${MERCAN_REF:-357e6e93ddca5d3f2f5186f2b96e6e929e5d9892}"
MERCAN_SRC="${MERCAN_SRC:-$ROOT/.native-work/MercanRuntime}"
if [[ "${MERCAN_USE_LOCAL_SOURCE:-0}" != "1" ]]; then
  if [[ ! -f "$MERCAN_SRC/runtime/nedo004-ffi/Cargo.toml" ]]; then
    mkdir -p "$(dirname "$MERCAN_SRC")"
    git clone https://github.com/Ahmet2001/MercanRuntime.git "$MERCAN_SRC"
  fi
  git -C "$MERCAN_SRC" fetch origin "$MERCAN_REF"
  git -C "$MERCAN_SRC" checkout --detach "$MERCAN_REF"
fi
cargo build --locked --release --lib --jobs "${MERCAN_JOBS:-2}" \
  --manifest-path "$MERCAN_SRC/runtime/nedo004-ffi/Cargo.toml"
LIB="$MERCAN_SRC/runtime/nedo004-ffi/target/release/libnedo004_ffi.so"
test -s "$LIB"
"${MERCAN_PYTHON:-python3}" "$ROOT/scripts/stage_nedo_native.py" \
  --runtime-source "$MERCAN_SRC" --library "$LIB"
