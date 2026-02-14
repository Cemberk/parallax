#!/bin/bash
# GPU DiffDbg v0.1 Release Packaging Script
#
# Creates a distributable archive containing all components:
#   - Compiler Plugin (.so)
#   - Runtime Library (.a)
#   - CLI Tool (gddbg-diff)
#   - Driver Script (gddbg)
#   - Headers and examples
#
# Usage:
#   ./package.sh              # Build and package everything
#   ./package.sh --skip-build # Package without rebuilding

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VERSION="0.1.0"
PACKAGE_NAME="gpu-diffdbg-${VERSION}"
BUILD_DIR="${SCRIPT_DIR}/build"
STAGE_DIR="${SCRIPT_DIR}/${PACKAGE_NAME}"
ARCHIVE="${SCRIPT_DIR}/${PACKAGE_NAME}.tar.gz"

echo "========================================="
echo "GPU DiffDbg v${VERSION} Release Packaging"
echo "========================================="
echo ""

# ---- Step 1: Build everything ----

if [ "$1" != "--skip-build" ]; then
    echo ">>> Step 1: Building C++/CUDA components..."

    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | grep -E "(Found|Built|Error|Warning|configured)" || true
    cmake --build . -j$(nproc) 2>&1 | tail -10

    echo ""
    echo ">>> Step 2: Building Rust differ..."

    cd "${SCRIPT_DIR}/differ"
    cargo build --release 2>&1 | grep -v "^$" | tail -5

    echo ""
else
    echo ">>> Skipping build (--skip-build)"
    echo ""
fi

# ---- Step 2: Stage the release ----

echo ">>> Step 3: Staging release files..."

# Clean previous staging
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/lib"
mkdir -p "${STAGE_DIR}/bin"
mkdir -p "${STAGE_DIR}/include"
mkdir -p "${STAGE_DIR}/examples"

# Compiler Plugin (.so)
if [ -f "${BUILD_DIR}/lib/pass/libGpuDiffDbgPass.so" ]; then
    cp "${BUILD_DIR}/lib/pass/libGpuDiffDbgPass.so" "${STAGE_DIR}/lib/"
    echo "  ✓ Compiler plugin: lib/libGpuDiffDbgPass.so"
else
    echo "  ⚠ Compiler plugin not found (LLVM pass not built?)"
fi

# Runtime Library (.a)
if [ -f "${BUILD_DIR}/lib/runtime/libgddbg_runtime.a" ]; then
    cp "${BUILD_DIR}/lib/runtime/libgddbg_runtime.a" "${STAGE_DIR}/lib/"
    echo "  ✓ Runtime library: lib/libgddbg_runtime.a"
else
    echo "  ⚠ Runtime library not found"
fi

# CLI Tool (gddbg-diff)
if [ -f "${SCRIPT_DIR}/differ/target/release/gddbg-diff" ]; then
    cp "${SCRIPT_DIR}/differ/target/release/gddbg-diff" "${STAGE_DIR}/bin/"
    echo "  ✓ CLI tool: bin/gddbg-diff"
else
    echo "  ⚠ CLI tool not found (cargo build failed?)"
fi

# Driver Script
cp "${SCRIPT_DIR}/gddbg" "${STAGE_DIR}/bin/"
chmod +x "${STAGE_DIR}/bin/gddbg"
echo "  ✓ Driver script: bin/gddbg"

# Headers
cp "${SCRIPT_DIR}/lib/runtime/gddbg_runtime.h" "${STAGE_DIR}/include/"
cp "${SCRIPT_DIR}/lib/common/trace_format.h" "${STAGE_DIR}/include/"
echo "  ✓ Headers: include/gddbg_runtime.h, include/trace_format.h"

# Examples
cp "${SCRIPT_DIR}/examples/"*.cu "${STAGE_DIR}/examples/" 2>/dev/null || true
cp "${SCRIPT_DIR}/examples/README.md" "${STAGE_DIR}/examples/" 2>/dev/null || true
cp "${SCRIPT_DIR}/examples/run_all_demos.sh" "${STAGE_DIR}/examples/" 2>/dev/null || true
chmod +x "${STAGE_DIR}/examples/run_all_demos.sh" 2>/dev/null || true
echo "  ✓ Examples: examples/"

# ---- Step 3: Create README for the release ----

cat > "${STAGE_DIR}/README.md" << 'RELEASE_README'
# GPU DiffDbg v0.1.0

A differential debugger for CUDA kernels. Identifies exactly where two GPU
executions diverge - like `rr` for GPUs.

## Quick Start

```bash
# 1. Add to PATH
export PATH=$PATH:$(pwd)/bin

# 2. Compile your kernel with instrumentation
clang++ -fpass-plugin=lib/libGpuDiffDbgPass.so \
    -I include/ \
    -L lib/ -lgddbg_runtime -lcudart \
    -o my_kernel my_kernel.cu

# 3. Run twice, compare
gddbg check ./my_kernel
```

## Components

| File | Description |
|------|-------------|
| `lib/libGpuDiffDbgPass.so` | LLVM compiler plugin (auto-instruments branches) |
| `lib/libgddbg_runtime.a` | CUDA runtime library (records traces) |
| `bin/gddbg-diff` | Differential analysis CLI (Rust) |
| `bin/gddbg` | Workflow driver (Python) |
| `include/` | Headers for manual instrumentation |
| `examples/` | Validation scenarios |

## Selective Instrumentation

### Compile-Time Filter (Zero Overhead)
Only instrument specific functions:
```bash
GDDBG_FILTER="compute_attention,matmul_*" clang++ -fpass-plugin=lib/libGpuDiffDbgPass.so ...
```

### Runtime ROI Toggle
Enable/disable recording from kernel code:
```cuda
__global__ void my_kernel(...) {
    // Only record when something looks wrong
    if (error_metric > threshold) {
        gddbg_enable();
    }

    // ... code under investigation ...

    gddbg_disable();
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GDDBG_TRACE` | `trace.gddbg` | Output trace file path |
| `GDDBG_FILTER` | (none) | Comma-separated function name patterns |
| `GDDBG_SITES` | `gddbg-sites.json` | Site table output path |
| `GDDBG_ENABLED` | `1` | Set to `0` to disable tracing |
| `GDDBG_BUFFER_SIZE` | `4096` | Events per warp buffer |

## Commands

```bash
gddbg diff <trace_a> <trace_b>              # Compare two traces
gddbg diff <trace_a> <trace_b> --map sites.json  # With source locations
gddbg run ./binary                           # Run with tracing
gddbg check ./binary                         # Run twice and auto-diff
```

## Requirements

- CUDA Toolkit 11.0+
- LLVM 18 (for compiler plugin)
- Python 3.8+ (for driver script)
- Linux x86_64
RELEASE_README

echo "  ✓ Release README"

# ---- Step 4: Create archive ----

echo ""
echo ">>> Step 4: Creating archive..."

cd "${SCRIPT_DIR}"
tar czf "${ARCHIVE}" "${PACKAGE_NAME}/"

echo "  ✓ Archive: ${ARCHIVE}"

# Show summary
echo ""
echo "========================================="
echo "Release Summary"
echo "========================================="
echo ""
echo "Package: ${ARCHIVE}"
echo "Size:    $(du -h "${ARCHIVE}" | cut -f1)"
echo ""
echo "Contents:"
find "${STAGE_DIR}" -type f | sort | while read f; do
    size=$(du -h "$f" | cut -f1)
    rel=$(echo "$f" | sed "s|${STAGE_DIR}/||")
    printf "  %-40s %s\n" "$rel" "$size"
done

echo ""
echo "To install:"
echo "  tar xzf ${PACKAGE_NAME}.tar.gz"
echo "  export PATH=\$PATH:\$(pwd)/${PACKAGE_NAME}/bin"
echo ""
echo "========================================="
echo "✅ Packaging complete!"
echo "========================================="

# Cleanup staging directory
rm -rf "${STAGE_DIR}"
