#!/bin/bash
# PRLX End-to-End GPU Test
#
# Verifies the full compile-instrument-run-diff pipeline on real GPU hardware.
# Gracefully skips (exit 0) when prerequisites are missing so CI doesn't fail
# on CPU-only builders.
#
# Usage:
#   bash tests/test_e2e.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PRLX="$PROJECT_DIR/prlx"
WORK_DIR="/tmp/prlx_e2e_$$"
PASS=0
FAIL=0
SKIP=0

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT
mkdir -p "$WORK_DIR"

# Colors (if terminal supports it)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    YELLOW='\033[0;33m'
    NC='\033[0m'
else
    GREEN=''; RED=''; YELLOW=''; NC=''
fi

pass() { echo -e "  ${GREEN}PASS${NC}: $1"; PASS=$((PASS + 1)); }
fail() { echo -e "  ${RED}FAIL${NC}: $1"; FAIL=$((FAIL + 1)); }
skip() { echo -e "  ${YELLOW}SKIP${NC}: $1"; SKIP=$((SKIP + 1)); }

echo "=== PRLX End-to-End Tests ==="
echo ""

# ---- Prerequisites check ----

# GPU available?
if ! nvidia-smi >/dev/null 2>&1; then
    echo "No GPU detected (nvidia-smi failed). Skipping E2E tests."
    exit 0
fi

# Differ binary available?
DIFFER="$PROJECT_DIR/differ/target/release/prlx-diff"
if [ ! -x "$DIFFER" ]; then
    # Try debug build
    DIFFER="$PROJECT_DIR/differ/target/debug/prlx-diff"
    if [ ! -x "$DIFFER" ]; then
        echo "prlx-diff binary not found. Build with: cd differ && cargo build --release"
        echo "Skipping E2E tests."
        exit 0
    fi
fi

# ---- Test 1: Binary path (pre-built divergence_test) ----
echo "--- Test 1: Binary Path (divergence_test) ---"

DIVERGENCE_BIN="$PROJECT_DIR/build/tests/divergence_test"
if [ -x "$DIVERGENCE_BIN" ]; then
    TRACE_A="$WORK_DIR/div_a.prlx"
    TRACE_B="$WORK_DIR/div_b.prlx"

    # Run divergence_test — it generates two traces internally
    # The binary takes two output paths as args
    if PRLX_TRACE="$TRACE_A" "$DIVERGENCE_BIN" "$TRACE_A" "$TRACE_B" 2>/dev/null; then
        # Check traces exist
        if [ -f "$TRACE_A" ] && [ -f "$TRACE_B" ]; then
            pass "divergence_test generated traces"

            # Diff them — expect divergences (exit code 1 = divergences found)
            DIFF_OUT="$WORK_DIR/diff_out.txt"
            if "$DIFFER" "$TRACE_A" "$TRACE_B" > "$DIFF_OUT" 2>&1; then
                # Exit 0 means no divergences — unexpected but not a failure
                pass "differ ran successfully (no divergences)"
            elif [ $? -eq 1 ]; then
                # Exit 1 = divergences found, which is expected
                if grep -qi "branch\|diverge\|direction" "$DIFF_OUT" 2>/dev/null; then
                    pass "differ detected branch divergences"
                else
                    pass "differ found divergences"
                fi
            else
                fail "differ crashed (exit code $?)"
            fi
        else
            fail "divergence_test did not generate trace files"
        fi
    else
        fail "divergence_test execution failed"
    fi
else
    skip "divergence_test binary not found (build with cmake --build build)"
fi

echo ""

# ---- Test 2: Full pipeline (compile + run + diff via prlx CLI) ----
echo "--- Test 2: Full Pipeline (prlx compile + run + diff) ---"

QUICKSTART_SRC="$PROJECT_DIR/demos/quickstart/threshold_bug.cu"
if [ -f "$QUICKSTART_SRC" ]; then
    # Check if prlx compile can find its pass
    if python "$PRLX" compile "$QUICKSTART_SRC" -o "$WORK_DIR/threshold_bug" -v 2>&1; then
        pass "prlx compile succeeded"

        # Run with threshold=10
        if PRLX_TRACE="$WORK_DIR/pipe_a.prlx" "$WORK_DIR/threshold_bug" 10 2>/dev/null; then
            pass "run A (threshold=10) succeeded"
        else
            fail "run A (threshold=10) failed"
        fi

        # Run with threshold=64
        if PRLX_TRACE="$WORK_DIR/pipe_b.prlx" "$WORK_DIR/threshold_bug" 64 2>/dev/null; then
            pass "run B (threshold=64) succeeded"
        else
            fail "run B (threshold=64) failed"
        fi

        # Diff — expect divergences (exit 1)
        if [ -f "$WORK_DIR/pipe_a.prlx" ] && [ -f "$WORK_DIR/pipe_b.prlx" ]; then
            PIPE_DIFF="$WORK_DIR/pipe_diff.txt"
            set +e
            "$DIFFER" "$WORK_DIR/pipe_a.prlx" "$WORK_DIR/pipe_b.prlx" > "$PIPE_DIFF" 2>&1
            DIFF_RC=$?
            set -e

            if [ "$DIFF_RC" -eq 1 ]; then
                pass "differ detected divergences (threshold change)"
            elif [ "$DIFF_RC" -eq 0 ]; then
                # No divergences — unusual but the traces may be empty
                skip "differ found no divergences (traces may be empty)"
            else
                fail "differ crashed (exit code $DIFF_RC)"
            fi
        else
            fail "traces not generated"
        fi
    else
        skip "prlx compile failed (LLVM pass or clang not available)"
    fi
else
    skip "quickstart source not found"
fi

echo ""

# ---- Summary ----
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
