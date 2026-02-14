#!/usr/bin/env python3
"""
Test gpu-diffdbg with a Triton kernel that has divergent control flow.

This kernel implements a conditional accumulation:
- If input value > threshold, multiply by 2 and accumulate
- Otherwise, negate and accumulate
- Uses atomic add for cross-block reduction

Running with different thresholds produces different branch paths,
which the differ can detect.
"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl
import gddbg

# Enable instrumentation BEFORE any kernel is compiled
gddbg.integrate_with_triton()


@triton.jit
def divergent_kernel(
    x_ptr,
    out_ptr,
    count_ptr,       # atomic counter for "taken" branches
    n_elements,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # ---- Divergent control flow ----
    # This generates branch instructions in LLVM IR
    positive = x > threshold
    # tl.where generates a select, but the mask interactions
    # with load/store generate actual branches

    # Path A: scale up
    scaled = tl.where(positive, x * 2.0, -x)

    # Path B: additional processing for large values
    large = x > (threshold * 2.0)
    result = tl.where(large, scaled + 1.0, scaled)

    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

    # Atomic: count how many elements exceeded threshold per block
    block_count = tl.sum(positive.to(tl.int32), axis=0)
    if pid == 0:
        # Only first block writes (generates a branch on pid)
        tl.atomic_add(count_ptr, block_count)


def run_kernel(x, threshold, trace_path):
    """Run the kernel with tracing enabled, return output."""
    n = x.shape[0]
    out = torch.empty_like(x)
    count = torch.zeros(1, dtype=torch.int32, device="cuda")

    os.environ["GDDBG_TRACE"] = trace_path
    os.environ["GDDBG_HISTORY_DEPTH"] = "64"

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    divergent_kernel[grid](x, out, count, n, threshold, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    return out, count.item()


def main():
    print("=" * 60)
    print("GPU DiffDbg - Triton Divergence Test")
    print("=" * 60)

    # Fixed input data
    torch.manual_seed(42)
    n = 2048
    x = torch.randn(n, device="cuda")

    print(f"\nInput: {n} elements, mean={x.mean():.3f}, std={x.std():.3f}")
    print(f"  Values > 0.0: {(x > 0.0).sum().item()}")
    print(f"  Values > 0.5: {(x > 0.5).sum().item()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        trace_a = os.path.join(tmpdir, "trace_a.gddbg")
        trace_b = os.path.join(tmpdir, "trace_b.gddbg")

        # Run A: threshold = 0.0 (about half take each branch)
        print("\n--- Run A: threshold = 0.0 ---")
        out_a, count_a = run_kernel(x, 0.0, trace_a)
        print(f"  Output mean: {out_a.mean():.4f}")
        print(f"  Positive count: {count_a}")
        print(f"  Trace: {trace_a}")

        # Run B: threshold = 0.5 (fewer take the positive branch)
        print("\n--- Run B: threshold = 0.5 ---")
        out_b, count_b = run_kernel(x, 0.5, trace_b)
        print(f"  Output mean: {out_b.mean():.4f}")
        print(f"  Positive count: {count_b}")
        print(f"  Trace: {trace_b}")

        # Read traces with Python reader
        print("\n--- Trace Analysis (Python reader) ---")
        from gddbg import read_trace

        ta = read_trace(trace_a)
        tb = read_trace(trace_b)
        print(f"  Trace A: {ta.total_events} events, {ta.num_warps} warps, history={ta.header.has_history}")
        print(f"  Trace B: {tb.total_events} events, {tb.num_warps} warps, history={tb.header.has_history}")

        # Show per-warp event breakdown
        for label, t in [("A", ta), ("B", tb)]:
            active_warps = sum(1 for w in t.warps() if w.num_events > 0)
            print(f"  Trace {label}: {active_warps} active warps")
            for w in t.warps():
                if w.num_events > 0:
                    types = {}
                    for e in w.events:
                        name = e.event_type_name
                        types[name] = types.get(name, 0) + 1
                    summary = ", ".join(f"{v} {k}" for k, v in types.items())
                    print(f"    Warp {w.warp_idx}: {w.num_events} events ({summary})")
                    # Show first few events
                    for e in w.events[:3]:
                        print(f"      {e.event_type_name:20s} site=0x{e.site_id:08x} "
                              f"dir={e.branch_dir} mask=0x{e.active_mask:08x} val={e.value_a}")

        # Run the Rust differ
        print("\n--- Differential Analysis (Rust differ) ---")
        differ = Path(__file__).parent / "differ" / "target" / "release" / "gddbg-diff"
        if differ.exists():
            cmd = [str(differ), trace_a, trace_b, "--history"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            print(f"  Exit code: {result.returncode} "
                  f"({'DIVERGENCES FOUND' if result.returncode != 0 else 'IDENTICAL'})")
        else:
            print("  gddbg-diff not found, skipping")

    print("\nDone!")


if __name__ == "__main__":
    main()
