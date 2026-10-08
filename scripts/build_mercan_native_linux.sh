#!/usr/bin/env bash
# Build a compatible libmercan.so and stage it for a platform-specific wheel.
# Uses pinned MercanRuntime source; only run on trusted Linux build workers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MERCAN_REF="${MERCAN_REF:-0e0f46e490d8d5d98bb746c899d6d87124151756}"
MERCAN_SRC="${MERCAN_SRC:-$ROOT/.native-work/MercanRuntime}"
MERCAN_JOBS="${MERCAN_JOBS:-2}"
if [[ ! -f "$MERCAN_SRC/CMakeLists.txt" ]]; then
  mkdir -p "$(dirname "$MERCAN_SRC")"
  git clone https://github.com/Ahmet2001/MercanRuntime.git "$MERCAN_SRC"
fi
git -C "$MERCAN_SRC" checkout --detach "$MERCAN_REF"
chmod +x "$MERCAN_SRC/scripts/prepare_llama.sh"
"$MERCAN_SRC/scripts/prepare_llama.sh"
cmake -S "$MERCAN_SRC" -B "$MERCAN_SRC/build/ethosoftlib-shared" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMERCAN_LLAMA_DIR="$MERCAN_SRC/build/_deps/llama-src" \
  -DMERCAN_BUILD_SHARED=ON \
  -DGGML_NATIVE=OFF
cmake --build "$MERCAN_SRC/build/ethosoftlib-shared" --target libmercan -j "$MERCAN_JOBS"
LIB="$MERCAN_SRC/build/ethosoftlib-shared/libmercan.so"
test -s "$LIB"
TARGET="$ROOT/src/ethosoftlib/mercan/lib"
mkdir -p "$TARGET"
cp "$LIB" "$TARGET/libmercan.so"
echo "Staged native library at $TARGET/libmercan.so"
echo "Build the platform wheel with: python -m build --wheel"
echo "Audit/relabel external runtime dependencies before public distribution."
