#!/usr/bin/env bash
# Build a compatible libmercan.so and stage it for a platform-specific wheel.
# Uses pinned MercanRuntime source; only run on trusted Linux build workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERCAN_REF="${MERCAN_REF:-600b322f6c1a3a852ac6141ae99c280dcb956dcf}"
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
  git -C "$MERCAN_SRC" fetch origin "$MERCAN_REF"
  git -C "$MERCAN_SRC" checkout --detach "$MERCAN_REF"
  chmod +x "$MERCAN_SRC/scripts/prepare_llama.sh"
  "$MERCAN_SRC/scripts/prepare_llama.sh"
  LLAMA_SRC="$MERCAN_SRC/build/_deps/llama-src"
fi
MERCAN_CUDA="${MERCAN_CUDA:-OFF}"
case "$MERCAN_CUDA" in
  ON|1|TRUE|YES) BUILD_SUFFIX="-cuda" ;;
  OFF|0|FALSE|NO) BUILD_SUFFIX="" ;;
  *) echo "MERCAN_CUDA must be ON or OFF" >&2; exit 1 ;;
esac
SHARED_BUILD="${MERCAN_SHARED_BUILD_DIR:-$MERCAN_SRC/build/ethosoftlib-shared$BUILD_SUFFIX}"
CUDA_OPTIONS=()
if [[ -n "${MERCAN_CUDA_ARCHITECTURES:-}" ]]; then
  CUDA_OPTIONS+=("-DCMAKE_CUDA_ARCHITECTURES=$MERCAN_CUDA_ARCHITECTURES")
fi
cmake -S "$MERCAN_SRC" -B "$SHARED_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMERCAN_LLAMA_DIR="$LLAMA_SRC" \
  -DMERCAN_BUILD_SHARED=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA="$MERCAN_CUDA" \
  -DMERCAN_BUILD_SDK_TESTS=ON "${CUDA_OPTIONS[@]}"
cmake --build "$SHARED_BUILD" --target libmercan mercan -j "$MERCAN_JOBS"
LIB="$SHARED_BUILD/libmercan.so"
test -s "$LIB"
TARGET="$ROOT/src/ethosoftlib/mercan/lib"
mkdir -p "$TARGET"
STAGED_LIBRARY="$(mktemp "$TARGET/.libmercan.XXXXXX")"
cp "$LIB" "$STAGED_LIBRARY"
chmod 755 "$STAGED_LIBRARY"
mv -f "$STAGED_LIBRARY" "$TARGET/libmercan.so"
echo "Staged native library at $TARGET/libmercan.so"
echo "Device capabilities: $SHARED_BUILD/mercan devices"
echo "Build the platform wheel with: python -m build --wheel"
echo "Audit/relabel external runtime dependencies before public distribution."
