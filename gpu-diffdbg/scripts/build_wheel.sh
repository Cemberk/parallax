#!/bin/bash
# Build a platform wheel for prlx with all native artifacts bundled.
#
# Prerequisites (build machine):
#   - CUDA toolkit (nvcc, libcudart)
#   - LLVM 20 dev (llvm-20-dev, clang-20)
#   - LLVM 18 dev (llvm-18-dev, clang-18)  [optional — skipped if missing]
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

echo "=== PRLX Wheel Builder ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Clean previous artifacts
rm -rf "$DATA_DIR/bin" "$DATA_DIR/lib" "$DATA_DIR/include"
mkdir -p "$DATA_DIR/bin" "$DATA_DIR/lib" "$DATA_DIR/include"

# ---- Step 1: Build LLVM pass for LLVM 20 ----
if command -v llvm-config-20 &>/dev/null || [ -d /usr/lib/llvm-20/cmake ]; then
    echo "[1/5] Building LLVM pass (LLVM 20)..."
    cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build-llvm20" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPRLX_LLVM_VERSION=20 \
        -Wno-dev 2>&1 | tail -3
    cmake --build "$PROJECT_ROOT/build-llvm20" --target PrlxPass -j"$(nproc)" 2>&1 | tail -3
    cp "$PROJECT_ROOT/build-llvm20/lib/pass/libPrlxPass.so" \
       "$DATA_DIR/lib/libPrlxPass.llvm20.so"
    echo "  -> libPrlxPass.llvm20.so"
else
    echo "[1/5] LLVM 20 not found, skipping."
fi

# ---- Step 2: Build LLVM pass for LLVM 18 ----
if command -v llvm-config-18 &>/dev/null || [ -d /usr/lib/llvm-18/cmake ]; then
    echo "[2/5] Building LLVM pass (LLVM 18)..."
    cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build-llvm18" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPRLX_LLVM_VERSION=18 \
        -Wno-dev 2>&1 | tail -3
    cmake --build "$PROJECT_ROOT/build-llvm18" --target PrlxPass -j"$(nproc)" 2>&1 | tail -3
    cp "$PROJECT_ROOT/build-llvm18/lib/pass/libPrlxPass.so" \
       "$DATA_DIR/lib/libPrlxPass.llvm18.so"
    echo "  -> libPrlxPass.llvm18.so"
else
    echo "[2/5] LLVM 18 not found, skipping."
fi

# ---- Step 3: Build CUDA runtime ----
echo "[3/5] Building CUDA runtime (shared + static + bitcode)..."
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build-runtime" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPRLX_CUDA_ARCHITECTURES="70;80;90" \
    -Wno-dev 2>&1 | tail -3
cmake --build "$PROJECT_ROOT/build-runtime" \
    --target prlx_runtime_shared prlx_runtime prlx_runtime_bitcode \
    -j"$(nproc)" 2>&1 | tail -3
cp "$PROJECT_ROOT/build-runtime/lib/runtime/libprlx_runtime_shared.so" "$DATA_DIR/lib/"
cp "$PROJECT_ROOT/build-runtime/lib/runtime/libprlx_runtime.a" "$DATA_DIR/lib/"
cp "$PROJECT_ROOT/build-runtime/lib/runtime/prlx_runtime_nvptx.bc" "$DATA_DIR/lib/"
echo "  -> libprlx_runtime_shared.so, libprlx_runtime.a, prlx_runtime_nvptx.bc"

# ---- Step 4: Build Rust differ ----
echo "[4/5] Building Rust differ..."
(cd "$PROJECT_ROOT/differ" && cargo build --release 2>&1 | tail -3)
cp "$PROJECT_ROOT/differ/target/release/prlx-diff" "$DATA_DIR/bin/"
strip "$DATA_DIR/bin/prlx-diff" 2>/dev/null || true
chmod +x "$DATA_DIR/bin/prlx-diff"
echo "  -> prlx-diff ($(du -h "$DATA_DIR/bin/prlx-diff" | cut -f1))"

# ---- Step 5: Copy headers ----
echo "[5/5] Copying headers..."
cp "$PROJECT_ROOT/lib/runtime/prlx_runtime.h" "$DATA_DIR/include/"
cp "$PROJECT_ROOT/lib/common/trace_format.h" "$DATA_DIR/include/"
echo "  -> prlx_runtime.h, trace_format.h"

# ---- Build the wheel ----
echo ""
echo "=== Building wheel ==="
(cd "$PROJECT_ROOT" && python3 -m build --wheel 2>&1 | tail -5)

echo ""
echo "=== Done ==="
ls -lh "$PROJECT_ROOT/dist/"*.whl 2>/dev/null || echo "Warning: No wheel found in dist/"
