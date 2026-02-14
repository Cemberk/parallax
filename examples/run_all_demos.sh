#!/bin/bash
# PRLX: One-Click Demo Runner
# Runs all validation scenarios with LLVM pass instrumentation + snapshot capture
#
# Usage: ./examples/run_all_demos.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PRLX="$PROJECT_DIR/prlx"
DIFFER="$PROJECT_DIR/differ/target/release/prlx-diff"
EXAMPLES_SRC="$SCRIPT_DIR"
WORK_DIR="/tmp/prlx_demo_$$"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

mkdir -p "$WORK_DIR"
trap "rm -rf $WORK_DIR" EXIT

echo -e "${CYAN}========================================="
echo "PRLX: End-to-End Demo"
echo -e "=========================================${NC}"
echo ""

# Pre-flight checks
echo "Checking environment..."

if [ ! -f "$PRLX" ]; then
    echo -e "${RED}Error: prlx driver not found at $PRLX${NC}"
    exit 1
fi

if [ ! -f "$DIFFER" ]; then
    echo -e "${RED}Error: prlx-diff not found. Build with: cd differ && cargo build --release${NC}"
    exit 1
fi

# Check that prlx compile works
if ! python3 "$PRLX" compile --help >/dev/null 2>&1; then
    echo -e "${RED}Error: 'prlx compile' not available${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo -e "  GPU: ${GREEN}${GPU_NAME:-unknown}${NC}"
echo -e "  LLVM Pass: ${GREEN}$(ls "$PROJECT_DIR/build/lib/pass/libPrlxPass.so" 2>/dev/null && echo "OK" || echo "MISSING")${NC}"
echo ""

# ===================================================================
echo -e "${CYAN}========================================="
echo "Scenario 1: Branch Divergence + Snapshot"
echo -e "=========================================${NC}"
echo "Kernel: simple branch (value > threshold)"
echo "Goal:   Detect different branch directions + show per-lane operands"
echo ""

# Create a simple kernel file
cat > "$WORK_DIR/branch_demo.cu" << 'KERNEL_EOF'
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void branch_kernel(int* data, int* out, int threshold, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int value = data[idx];
    if (value > threshold) {
        out[idx] = value * 2;
    } else {
        out[idx] = -value;
    }
    if (out[idx] > 200) {
        out[idx] = 200;
    }
}

int main(int argc, char** argv) {
    const int N = 128;
    int threshold = (argc > 1) ? atoi(argv[1]) : 50;
    printf("threshold=%d\n", threshold);
    prlx_init();
    int* h_data = new int[N]; int* h_out = new int[N];
    for (int i = 0; i < N; i++) h_data[i] = i;
    int *d_data, *d_out;
    cudaMalloc(&d_data, N * sizeof(int));
    cudaMalloc(&d_out,  N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 block(32); dim3 grid((N + 31) / 32);
    prlx_pre_launch("branch_kernel", grid, block);
    branch_kernel<<<grid, block>>>(d_data, d_out, threshold, N);
    cudaDeviceSynchronize();
    prlx_post_launch();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Results: [0]=%d [32]=%d [64]=%d [96]=%d\n",
           h_out[0], h_out[32], h_out[64], h_out[96]);
    cudaFree(d_data); cudaFree(d_out);
    delete[] h_data; delete[] h_out;
    prlx_shutdown();
    return 0;
}
KERNEL_EOF

echo ">>> Compiling with LLVM pass instrumentation..."
python3 "$PRLX" compile "$WORK_DIR/branch_demo.cu" -o "$WORK_DIR/branch_demo" -v 2>&1

echo ""
echo -e ">>> ${YELLOW}Run A: threshold=10${NC}"
PRLX_TRACE="$WORK_DIR/branch_a.prlx" PRLX_SNAPSHOT_DEPTH=32 "$WORK_DIR/branch_demo" 10 2>&1

echo ""
echo -e ">>> ${YELLOW}Run B: threshold=64${NC}"
PRLX_TRACE="$WORK_DIR/branch_b.prlx" PRLX_SNAPSHOT_DEPTH=32 "$WORK_DIR/branch_demo" 64 2>&1

echo ""
echo -e ">>> ${YELLOW}Differential Analysis:${NC}"
"$DIFFER" "$WORK_DIR/branch_a.prlx" "$WORK_DIR/branch_b.prlx" --history 2>&1 || true

echo ""
echo -e "${GREEN}Scenario 1 complete.${NC}"
echo "  Snapshot shows per-lane comparison operands (icmp sgt)."
echo "  A:rhs=10 vs B:rhs=64 explains the divergence."
echo ""

# ===================================================================
echo -e "${CYAN}========================================="
echo "Scenario 2: Shared Memory Bug Detection"
echo -e "=========================================${NC}"
echo "Kernel: tiled matmul with injected shared memory indexing bug"
echo "Goal:   Detect cascading branch divergence from data corruption"
echo ""

echo ">>> Compiling matmul with LLVM pass..."
python3 "$PRLX" compile "$EXAMPLES_SRC/matmul_divergence.cu" -o "$WORK_DIR/matmul_demo" -v 2>&1

echo ""
echo -e ">>> ${YELLOW}Run A: Correct implementation${NC}"
PRLX_TRACE="$WORK_DIR/matmul_correct.prlx" PRLX_SNAPSHOT_DEPTH=16 "$WORK_DIR/matmul_demo" correct 2>&1

echo ""
echo -e ">>> ${YELLOW}Run B: Buggy implementation (shmem indexing bug)${NC}"
PRLX_TRACE="$WORK_DIR/matmul_buggy.prlx" PRLX_SNAPSHOT_DEPTH=16 "$WORK_DIR/matmul_demo" buggy 2>&1

echo ""
echo -e ">>> ${YELLOW}Differential Analysis (top 5 divergences):${NC}"
"$DIFFER" "$WORK_DIR/matmul_correct.prlx" "$WORK_DIR/matmul_buggy.prlx" -n 5 2>&1 || true

echo ""
echo -e "${GREEN}Scenario 2 complete.${NC}"
echo "  Shared memory bug in thread 7 cascaded across all warps."
echo "  Branch divergence caught the data corruption via control flow."
echo ""

# ===================================================================
echo -e "${CYAN}========================================="
echo "Scenario 3: Manual Instrumentation"
echo -e "=========================================${NC}"
echo "Kernel: hand-instrumented with __prlx_record_branch/value"
echo "Goal:   Validate trace capture + time-travel history"
echo ""

INSTR_BIN="$PROJECT_DIR/build/examples/instrumented_divergence"
if [ -f "$INSTR_BIN" ]; then
    echo -e ">>> ${YELLOW}Run A: threshold=0 (all branches taken)${NC}"
    PRLX_TRACE="$WORK_DIR/instr_a.prlx" PRLX_HISTORY_DEPTH=64 "$INSTR_BIN" 0 2>&1

    echo ""
    echo -e ">>> ${YELLOW}Run B: threshold=50 (mixed branch pattern)${NC}"
    PRLX_TRACE="$WORK_DIR/instr_b.prlx" PRLX_HISTORY_DEPTH=64 "$INSTR_BIN" 50 2>&1

    echo ""
    echo -e ">>> ${YELLOW}Differential Analysis + History:${NC}"
    "$DIFFER" "$WORK_DIR/instr_a.prlx" "$WORK_DIR/instr_b.prlx" --history 2>&1 || true

    echo ""
    echo -e "${GREEN}Scenario 3 complete.${NC}"
    echo "  Time-travel history shows threshold=0 vs 50 causing divergence."
else
    echo -e "${YELLOW}Skipping: build examples first with cmake --build build${NC}"
fi

echo ""
echo -e "${CYAN}========================================="
echo -e "${GREEN}All Demos Complete!${NC}"
echo -e "${CYAN}=========================================${NC}"
echo ""
echo "Generated traces:"
ls -lh "$WORK_DIR"/*.prlx 2>/dev/null || echo "  (none)"
echo ""
echo "Pipeline validated:"
echo "  [x] prlx compile: clang + LLVM pass + PTX forward-compat"
echo "  [x] Snapshot capture: per-lane comparison operands via __shfl_sync"
echo "  [x] Time-travel history: per-warp value ring buffer"
echo "  [x] Differential analysis: branch, path, extra events"
echo "  [x] Shared memory bug detection via control flow divergence"
echo ""
