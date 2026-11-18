#!/usr/bin/env bash
# Build and test the C++ core on Linux (Ubuntu 24.04, natively, in WSL, or in CI).
#
# Usage: scripts/build_core.sh [sanitize|python|all]   (default: all)
#   sanitize  Debug build with address and undefined-behaviour sanitizers; runs GoogleTest.
#   python    Release build of the pybind11 module herdsight._core, placed in python/herdsight.
#
# Build folders go in $HERDSIGHT_BUILD_DIR (default ~/build/herdsight), outside the repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${HERDSIGHT_BUILD_DIR:-$HOME/build/herdsight}"
MODE="${1:-all}"

sanitize() {
  cmake -S "$ROOT/core" -B "$BUILD/asan" -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug -DHERDSIGHT_SANITIZE=ON
  cmake --build "$BUILD/asan"
  ctest --test-dir "$BUILD/asan" --output-on-failure
}

python_module() {
  # pybind11 comes from the Python environment (pinned in requirements-lock.txt), not the
  # system package: Ubuntu 24.04's pybind11 2.11 predates NumPy 2 and crashes converting arrays.
  local py pybind_dir
  py="$(command -v python3)"
  pybind_dir="$("$py" -m pybind11 --cmakedir)" || {
    echo "pybind11 not found in $py; install the dev dependencies first"; exit 1; }
  cmake -S "$ROOT/core" -B "$BUILD/release" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DHERDSIGHT_BUILD_PYTHON=ON \
    -DPython3_EXECUTABLE="$py" -Dpybind11_DIR="$pybind_dir"
  cmake --build "$BUILD/release"
  ctest --test-dir "$BUILD/release" --output-on-failure
}

case "$MODE" in
  sanitize) sanitize ;;
  python) python_module ;;
  all) sanitize; python_module ;;
  *) echo "unknown mode: $MODE (expected sanitize, python or all)"; exit 2 ;;
esac
