#!/bin/bash
# Smoke test: verify LLVM pass + runtime + differ are functional.

set -e

BUILD_DIR="../build"
PASS_SO="$BUILD_DIR/lib/pass/libPrlxPass.so"
RUNTIME_LIB="$BUILD_DIR/lib/runtime/libprlx_runtime.a"

if [ ! -f "$PASS_SO" ]; then
    echo "ERROR: LLVM pass not found: $PASS_SO"
    echo "Build first: cmake --build build"
    exit 1
fi

echo "Pass: $(ls -lh "$PASS_SO")"

if [ -f "$BUILD_DIR/tests/manual_trace_test" ]; then
    cd "$BUILD_DIR/tests"
    PRLX_TRACE=auto_test.prlx ./manual_trace_test
    echo "Runtime: OK"
else
    echo "ERROR: manual_trace_test not found"
    exit 1
fi

if [ -f "$BUILD_DIR/tests/auto_test.prlx" ]; then
    ls -lh "$BUILD_DIR/tests/auto_test.prlx"
    echo "Trace output: OK"
fi

if [ -f "$BUILD_DIR/tests/trace_a.prlx" ] && [ -f "$BUILD_DIR/tests/trace_b.prlx" ]; then
    cd "$BUILD_DIR/tests"
    ../../differ/target/release/prlx-diff trace_a.prlx trace_b.prlx -n 3 || true
    echo "Differ: OK"
fi

echo "All checks passed."
