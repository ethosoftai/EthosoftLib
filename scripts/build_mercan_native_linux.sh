#!/usr/bin/env bash
# Build a compatible libmercan.so and stage it for a platform-specific wheel.
# Uses pinned MercanRuntime source; only run on trusted Linux build workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERCAN_REF="${MERCAN_REF:-f8c8414845dcf52ad1344a477c1cc88ade83c595}"
MERCAN_SRC="${MERCAN_SRC:-$ROOT/.native-work/MercanRuntime}"
MERCAN_JOBS="${MERCAN_JOBS:-2}"
if [[ "${MERCAN_USE_LOCAL_SOURCE:-0}" == "1" ]]; then
  # Never check out, reset, clean or prepare an existing developer checkout.
  test -f "$MERCAN_SRC/runtime/libmercan/include/mercan.h"
  LLAMA_SRC="${MERCAN_LLAMA_DIR:-$MERCAN_SRC/build/_deps/llama-src}"
  if [[ ! -f "$LLAMA_SRC/src/models/nedolm.cpp" ]]; then
    echo "Local mode needs an already prepared patched llama.cpp at $LLAMA_SRC" >&2
    exit 1
  fi
else
  if [[ ! -f "$MERCAN_SRC/CMakeLists.txt" ]]; then
    mkdir -p "$(dirname "$MERCAN_SRC")"
    git clone https://github.com/Ahmet2001/MercanRuntime.git "$MERCAN_SRC"
  fi
  git -C "$MERCAN_SRC" checkout --detach "$MERCAN_REF"
  chmod +x "$MERCAN_SRC/scripts/prepare_llama.sh"
  "$MERCAN_SRC/scripts/prepare_llama.sh"
  LLAMA_SRC="$MERCAN_SRC/build/_deps/llama-src"
fi
SHARED_BUILD="${MERCAN_SHARED_BUILD_DIR:-$MERCAN_SRC/build/ethosoftlib-shared}"
cmake -S "$MERCAN_SRC" -B "$SHARED_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMERCAN_LLAMA_DIR="$LLAMA_SRC" \
  -DMERCAN_BUILD_SHARED=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA="${MERCAN_CUDA:-OFF}" \
  -DMERCAN_BUILD_SDK_TESTS=ON
cmake --build "$SHARED_BUILD" --target libmercan -j "$MERCAN_JOBS"
LIB="$SHARED_BUILD/libmercan.so"
test -s "$LIB"
TARGET="$ROOT/src/ethosoftlib/mercan/lib"
mkdir -p "$TARGET"
cp "$LIB" "$TARGET/libmercan.so"
echo "Staged native library at $TARGET/libmercan.so"
echo "Build the platform wheel with: python -m build --wheel"
echo "Audit/relabel external runtime dependencies before public distribution."
