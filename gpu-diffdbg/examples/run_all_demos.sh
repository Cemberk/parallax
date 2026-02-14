#!/bin/bash
# GPU DiffDbg: One-Click Demo Runner
# Runs all three validation scenarios and shows results

set -e

echo "========================================="
echo "GPU DiffDbg: One-Click Demo"
echo "========================================="
echo ""
echo "This script validates GPU DiffDbg on three critical scenarios:"
echo "  1. Loop Desync (bounded lookahead test)"
echo "  2. Occupancy (memory/performance test)"
echo "  3. Shared Memory Hazard (bug detection test)"
echo ""

# Check if we're in the build/examples directory
if [ ! -f "./loop_divergence" ]; then
    echo "Error: Examples not found. Please build first:"
    echo "  cd ../build && cmake --build ."
    exit 1
fi

# Check if gddbg driver exists
if [ ! -f "../../gddbg" ]; then
    echo "Error: gddbg driver not found at ../../gddbg"
    exit 1
fi

echo "========================================="
echo "Scenario 1: Loop Desync"
echo "========================================="
echo "Testing bounded lookahead with different loop counts..."
echo ""

echo ">>> Run A: 10 iterations"
GDDBG_TRACE=loop_a.gddbg ./loop_divergence 10 | tail -5

echo ""
echo ">>> Run B: 11 iterations"
GDDBG_TRACE=loop_b.gddbg ./loop_divergence 11 | tail -5

echo ""
echo ">>> Comparing traces..."
../../gddbg diff loop_a.gddbg loop_b.gddbg -n 3 || true

echo ""
echo "✓ Loop Desync test complete"
echo "  Expected: ExtraEvents divergence, re-sync at post-loop branch"
echo ""
read -p "Press Enter to continue to Scenario 2..."
echo ""

echo "========================================="
echo "Scenario 2: Occupancy Test"
echo "========================================="
echo "Testing memory and performance under load..."
echo ""

echo ">>> Running Medium load (256 blocks × 256 threads = 65K threads)"
GDDBG_TRACE=occ_medium.gddbg ./occupancy_test medium

echo ""
echo ">>> Trace file info:"
ls -lh occ_medium.gddbg

echo ""
echo "✓ Occupancy test complete"
echo "  Expected: No crashes, reasonable execution time, valid trace"
echo ""
read -p "Press Enter to continue to Scenario 3..."
echo ""

echo "========================================="
echo "Scenario 3: Shared Memory Hazard"
echo "========================================="
echo "Testing bug detection via control flow divergence..."
echo ""

echo ">>> Run A: Correct implementation"
GDDBG_TRACE=matmul_correct.gddbg ./matmul_divergence correct | tail -10

echo ""
echo ">>> Run B: Buggy implementation (shared memory indexing bug)"
GDDBG_TRACE=matmul_buggy.gddbg ./matmul_divergence buggy | tail -10

echo ""
echo ">>> Comparing traces..."
../../gddbg diff matmul_correct.gddbg matmul_buggy.gddbg -n 3 || true

echo ""
echo "✓ Shared Memory Hazard test complete"
echo "  Expected: Branch divergence detected in computation loop"
echo ""

echo "========================================="
echo "✅ All Three Scenarios Complete!"
echo "========================================="
echo ""
echo "Validation Summary:"
echo "  [✓] Scenario 1: Loop Desync"
echo "  [✓] Scenario 2: Occupancy Test"
echo "  [✓] Scenario 3: Shared Memory Hazard"
echo ""
echo "Generated trace files:"
ls -lh *.gddbg 2>/dev/null || echo "  (no trace files found)"
echo ""
echo "Next steps:"
echo "  - Review divergence reports above"
echo "  - Check if ExtraEvents divergence was detected (Scenario 1)"
echo "  - Verify no crashes or timeouts (Scenario 2)"
echo "  - Confirm branch divergence caught the bug (Scenario 3)"
echo ""
echo "If all three passed, GPU DiffDbg is production-ready! 🎉"
echo ""
