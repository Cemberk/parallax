#!/bin/bash
# Demonstration of Phase 2B enhanced differential features

set -e

cd "$(dirname "$0")/.."

echo "==================================="
echo "Phase 2B Feature Demonstration"
echo "==================================="
echo ""

# Build if needed
echo "1. Building project..."
cargo build --release 2>&1 | grep -E "(Compiling|Finished)" || echo "Already built"
echo ""

# Run tests
echo "2. Running comprehensive test suite..."
cargo test --quiet 2>&1 | grep -E "(test result|running)" || true
echo ""

# Find real trace files
TRACE_DIR="$(find ../build -name "trace_a.prlx" -exec dirname {} \; 2>/dev/null | head -1)"

if [ -z "$TRACE_DIR" ]; then
    echo "Warning: No real trace files found. Run divergence_test first:"
    echo "  cd ../build/tests && ./divergence_test"
    exit 1
fi

echo "3. Testing with real divergence traces from: $TRACE_DIR"
echo ""

# Test 1: Basic diff
echo "=== Test 1: Basic Divergence Detection ==="
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    -n 5
echo ""

# Test 2: With verbose trace info
echo "=== Test 2: Verbose Mode (shows trace headers) ==="
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    --verbose \
    -n 3
echo ""

# Test 3: With value comparison (may be noisy)
echo "=== Test 3: Value Comparison Enabled ==="
echo "(Note: This compares operand values - can be noisy for data-dependent code)"
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    --values \
    -n 10
echo ""

# Test 4: Lookahead window demo
echo "=== Test 4: Different Lookahead Window Sizes ==="
echo "Small window (4 events):"
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    --lookahead 4 \
    -n 3
echo ""

echo "Large window (64 events - better at re-syncing):"
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    --lookahead 64 \
    -n 3
echo ""

# Test 5: Dump mode for debugging
echo "=== Test 5: Dump Mode (inspect raw events) ==="
echo "First 10 events from trace A:"
./target/release/prlx-diff \
    "$TRACE_DIR/trace_a.prlx" \
    "$TRACE_DIR/trace_b.prlx" \
    --dump-a 10
echo ""

echo "==================================="
echo "All Features Demonstrated!"
echo "==================================="
echo ""
echo "Phase 2B Capabilities:"
echo "  ✓ Bounded lookahead for drift detection"
echo "  ✓ Multiple divergence types (Branch, ActiveMask, Value, Path, ExtraEvents)"
echo "  ✓ Parallel warp comparison with rayon"
echo "  ✓ Colored output with detailed explanations"
echo "  ✓ Configurable comparison modes"
echo "  ✓ Site-based grouping"
echo ""
echo "Next: Phase 3 - LLVM Pass Integration!"
