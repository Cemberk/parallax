#!/bin/bash
# Helper script to run GPU DiffDbg trace comparison

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

DIFFER="$BUILD_DIR/bin/gddbg-diff"

usage() {
    echo "Usage: $0 [OPTIONS] <trace_a.gddbg> <trace_b.gddbg>"
    echo ""
    echo "Compare two GPU execution traces"
    echo ""
    echo "Options:"
    echo "  -f, --format FORMAT     Output format: terminal (default) or json"
    echo "  -n, --max-div N         Maximum divergences to report (default: 100)"
    echo "  -v, --show-values       Show value differences"
    echo "  -s, --source-root PATH  Path to source root"
    echo "  -h, --help              Show this help"
    echo ""
    echo "Example:"
    echo "  $0 trace_a.gddbg trace_b.gddbg"
    echo "  $0 --format json --max-div 50 trace_a.gddbg trace_b.gddbg > diff.json"
    exit 1
}

# Check if differ exists
if [[ ! -f "$DIFFER" ]]; then
    echo "Error: Differ tool not found at $DIFFER"
    echo "Run 'make differ' first."
    exit 1
fi

# Parse arguments
FORMAT="terminal"
MAX_DIV="100"
SHOW_VALUES=""
SOURCE_ROOT=""
TRACE_A=""
TRACE_B=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -n|--max-div)
            MAX_DIV="$2"
            shift 2
            ;;
        -v|--show-values)
            SHOW_VALUES="--show-value-diffs"
            shift
            ;;
        -s|--source-root)
            SOURCE_ROOT="--source-root $2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            if [[ -z "$TRACE_A" ]]; then
                TRACE_A="$1"
            elif [[ -z "$TRACE_B" ]]; then
                TRACE_B="$1"
            else
                echo "Error: Too many arguments"
                usage
            fi
            shift
            ;;
    esac
done

if [[ -z "$TRACE_A" ]] || [[ -z "$TRACE_B" ]]; then
    echo "Error: Two trace files required"
    usage
fi

if [[ ! -f "$TRACE_A" ]]; then
    echo "Error: Trace A not found: $TRACE_A"
    exit 1
fi

if [[ ! -f "$TRACE_B" ]]; then
    echo "Error: Trace B not found: $TRACE_B"
    exit 1
fi

# Run differ
exec "$DIFFER" \
    --format "$FORMAT" \
    --max-divergences "$MAX_DIV" \
    $SHOW_VALUES \
    $SOURCE_ROOT \
    "$TRACE_A" \
    "$TRACE_B"
