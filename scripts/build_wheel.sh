#!/bin/bash
# Build a platform wheel for prlx with all native artifacts bundled.
#
# Prerequisites (build machine):
#   - CUDA toolkit >= 12.0 (nvcc, libcudart)
#   - LLVM 18-20 dev (at least one version)
#   - Rust toolchain (cargo)
#   - Python 3.8+ with: pip install build setuptools wheel
#   - CMake >= 3.20
#
# Usage:
#   bash scripts/build_wheel.sh
#   ls dist/*.whl

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/python/prlx/data"

# Use Ninja if available (faster than Make)
CMAKE_GEN=""
if command -v ninja &>/dev/null; then
    CMAKE_GEN="-G Ninja"
fi

# CUDA architectures: V100(70), T4(75), A100(80), RTX3090(86), RTX4090(89), H100(90)
# SM 90 PTX provides forward compatibility for Blackwell (SM 120) via JIT.
CUDA_ARCHS="70;75;80;86;89;90"

# LLVM versions to build passes for (continuous range)
LLVM_VERSIONS=(20 19 18)

echo "=== PRLX Wheel Builder ==="
echo "Project root: $PROJECT_ROOT"
echo "CUDA architectures: $CUDA_ARCHS"
echo "LLVM versions: ${LLVM_VERSIONS[*]}"
echo ""

# Clean previous artifacts
rm -rf "$DATA_DIR/bin" "$DATA_DIR/lib" "$DATA_DIR/include"
mkdir -p "$DATA_DIR/bin" "$DATA_DIR/lib" "$DATA_DIR/include"

STEP=0
TOTAL=$(( ${#LLVM_VERSIONS[@]} + 3 ))  # LLVM passes + runtime + differ + headers

# ---- Build LLVM passes ----
for VER in "${LLVM_VERSIONS[@]}"; do
    STEP=$((STEP + 1))
    if command -v "llvm-config-${VER}" &>/dev/null || [ -d "/usr/lib/llvm-${VER}/cmake" ]; then
        echo "[$STEP/$TOTAL] Building LLVM pass (LLVM $VER)..."
        cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build-llvm${VER}" \
            $CMAKE_GEN \
            -DCMAKE_BUILD_TYPE=Release \
            -DPRLX_LLVM_VERSION="$VER" \
            -Wno-dev 2>&1 | tail -3
        cmake --build "$PROJECT_ROOT/build-llvm${VER}" --target PrlxPass -j"$(nproc)" 2>&1 | tail -3
        cp "$PROJECT_ROOT/build-llvm${VER}/lib/pass/libPrlxPass.so" \
           "$DATA_DIR/lib/libPrlxPass.llvm${VER}.so"
        echo "  -> libPrlxPass.llvm${VER}.so"
    else
        echo "[$STEP/$TOTAL] LLVM $VER not found, skipping."
    fi
done

# ---- Build CUDA runtime ----
STEP=$((STEP + 1))
echo "[$STEP/$TOTAL] Building CUDA runtime (shared + static + bitcode)..."
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build-runtime" \
    $CMAKE_GEN \
    -DCMAKE_BUILD_TYPE=Release \
    -DPRLX_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -Wno-dev 2>&1 | tail -3
cmake --build "$PROJECT_ROOT/build-runtime" \
    --target prlx_runtime_shared prlx_runtime prlx_runtime_bitcode \
    -j"$(nproc)" 2>&1 | tail -3
cp "$PROJECT_ROOT/build-runtime/lib/runtime/libprlx_runtime_shared.so" "$DATA_DIR/lib/"
cp "$PROJECT_ROOT/build-runtime/lib/runtime/libprlx_runtime.a" "$DATA_DIR/lib/"
cp "$PROJECT_ROOT/build-runtime/lib/runtime/prlx_runtime_nvptx.bc" "$DATA_DIR/lib/"
echo "  -> libprlx_runtime_shared.so, libprlx_runtime.a, prlx_runtime_nvptx.bc"

# ---- Build Rust differ ----
STEP=$((STEP + 1))
echo "[$STEP/$TOTAL] Building Rust differ..."
(cd "$PROJECT_ROOT/differ" && cargo build --release 2>&1 | tail -3)
cp "$PROJECT_ROOT/differ/target/release/prlx-diff" "$DATA_DIR/bin/"
strip "$DATA_DIR/bin/prlx-diff" 2>/dev/null || true
chmod +x "$DATA_DIR/bin/prlx-diff"
echo "  -> prlx-diff ($(du -h "$DATA_DIR/bin/prlx-diff" | cut -f1))"

# ---- Copy headers ----
STEP=$((STEP + 1))
echo "[$STEP/$TOTAL] Copying headers..."
cp "$PROJECT_ROOT/lib/runtime/prlx_runtime.h" "$DATA_DIR/include/"
cp "$PROJECT_ROOT/lib/common/trace_format.h" "$DATA_DIR/include/"
echo "  -> prlx_runtime.h, trace_format.h"

# ---- Build the wheel ----
echo ""
echo "=== Building wheel ==="
rm -rf "$PROJECT_ROOT/dist"
(cd "$PROJECT_ROOT" && python3 -m build --wheel 2>&1 | tail -5)

# ---- Retag for manylinux (optional) ----
if [ "${PRLX_MANYLINUX:-0}" = "1" ]; then
    echo ""
    echo "=== Retagging wheel for manylinux_2_28_x86_64 ==="
    python3 -m wheel tags \
        --platform-tag manylinux_2_28_x86_64 \
        --remove "$PROJECT_ROOT/dist/"*.whl
fi

echo ""
echo "=== Done ==="
ls -lh "$PROJECT_ROOT/dist/"*.whl 2>/dev/null || echo "Warning: No wheel found in dist/"
