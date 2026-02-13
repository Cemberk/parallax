#!/bin/bash
# Helper script to compile CUDA kernels with GPU DiffDbg instrumentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

# Default values
ARCH="sm_80"
OUTPUT=""
OPTIMIZE="-O2"

usage() {
    echo "Usage: $0 [OPTIONS] <input.cu>"
    echo ""
    echo "Compile CUDA kernels with GPU DiffDbg instrumentation"
    echo ""
    echo "Options:"
    echo "  -a, --arch ARCH    CUDA architecture (default: sm_80)"
    echo "  -o, --output FILE  Output executable name"
    echo "  -O LEVEL           Optimization level (default: -O2)"
    echo "  -g                 Include debug symbols"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Example:"
    echo "  $0 -a sm_89 -o my_kernel my_kernel.cu"
    exit 1
}

# Parse arguments
DEBUG_FLAGS=""
INPUT_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--arch)
            ARCH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT="$2"
            shift 2
            ;;
        -O*)
            OPTIMIZE="$1"
            shift
            ;;
        -g)
            DEBUG_FLAGS="-g -lineinfo"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            else
                echo "Error: Multiple input files not supported"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$INPUT_FILE" ]]; then
    echo "Error: No input file specified"
    usage
fi

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

if [[ -z "$OUTPUT" ]]; then
    OUTPUT="${INPUT_FILE%.cu}"
fi

# Check if build exists
if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Error: Build directory not found. Run 'make' first."
    exit 1
fi

PASS_LIB="$BUILD_DIR/lib/pass/libGpuDiffDbgPass.so"
RUNTIME_LIB="$BUILD_DIR/lib/runtime/libgddbg_runtime.a"

if [[ ! -f "$PASS_LIB" ]]; then
    echo "Warning: LLVM pass not found at $PASS_LIB"
    echo "Building without instrumentation pass..."
    PASS_LIB=""
fi

if [[ ! -f "$RUNTIME_LIB" ]]; then
    echo "Error: Runtime library not found at $RUNTIME_LIB"
    echo "Run 'make runtime' first."
    exit 1
fi

# Compile with Clang
echo "Compiling $INPUT_FILE with GPU DiffDbg instrumentation..."
echo "  Architecture: $ARCH"
echo "  Output: $OUTPUT"
echo "  Optimization: $OPTIMIZE"

CLANG_CMD="clang++ \
    --cuda-gpu-arch=$ARCH \
    $OPTIMIZE \
    $DEBUG_FLAGS \
    -I$PROJECT_ROOT/lib/runtime \
    -I$PROJECT_ROOT/lib/common \
    -L$BUILD_DIR/lib/runtime \
    $INPUT_FILE \
    -lgddbg_runtime \
    -lcudart \
    -L/usr/local/cuda/lib64 \
    -o $OUTPUT"

if [[ -n "$PASS_LIB" ]]; then
    CLANG_CMD="$CLANG_CMD -fplugin=$PASS_LIB"
fi

echo "Running: $CLANG_CMD"
eval $CLANG_CMD

echo "Successfully compiled: $OUTPUT"
echo ""
echo "To run with tracing:"
echo "  GDDBG_TRACE=output.gddbg ./$OUTPUT"
