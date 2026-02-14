# PRLX
<img width="1000" height="500" alt="prlx" src="https://github.com/user-attachments/assets/b3f2581b-28c1-4846-b3b6-1d63811d582b" />

Differential debugger for CUDA and Triton GPU kernels.

Run your kernel twice with different inputs (or a known-good vs buggy version). PRLX instruments every branch, captures per-warp execution traces, and diffs them — telling you exactly which warp diverged, at which instruction, and what each lane saw:

```
Site 0xfbe6edc1  (branch_kernel:12)  2 warps affected

  Warp 1, event 3: Branch Direction
    A: TAKEN    B: NOT-TAKEN

    Operand Snapshot (icmp sgt):
    Lane       A:lhs    A:rhs       B:lhs    B:rhs
       0          32       10          32       64  <<<
       1          33       10          33       64  <<<
       2          34       10          34       64  <<<
      ...
```

The threshold changed from 10 to 64. Lanes 0–31 compared their value against the threshold; in run A they all passed, in run B they didn't. That's the bug.

## Install

```bash
pip install prlx
```

Needs CUDA 12+ and LLVM 18/19/20 on the host (for `prlx compile` and Triton integration). The differ and Python trace reader have no external deps.

<details>
<summary>From source</summary>

```bash
cmake -B build && cmake --build build
cd differ && cargo build --release && cd ..
pip install -e .
```

Build deps: CMake 3.20+, LLVM/Clang 18–20, CUDA Toolkit, Rust stable.
</details>

## Usage

### CUDA C

```bash
prlx compile kernel.cu -o kernel
PRLX_TRACE=a.prlx PRLX_SNAPSHOT_DEPTH=32 ./kernel --input-a
PRLX_TRACE=b.prlx PRLX_SNAPSHOT_DEPTH=32 ./kernel --input-b
prlx diff a.prlx b.prlx
```

### Triton

```python
import prlx
prlx.enable()  # hooks the Triton compiler — no kernel changes needed

import os, triton

os.environ["PRLX_TRACE"] = "a.prlx"
my_kernel[grid](...)

os.environ["PRLX_TRACE"] = "b.prlx"
my_kernel[grid](...)
```

### Python API

```python
from prlx import read_trace, diff_traces

# Read traces directly
trace = read_trace("a.prlx")
print(trace.header.kernel_name, trace.num_warps, "warps")
for w in trace.warps():
    for ev in w.events:
        if ev.is_branch:
            print(f"  warp {w.warp_idx}: site {ev.site_id:#x} {'T' if ev.branch_taken else 'NT'}")

# Or just run the differ
diff_traces("a.prlx", "b.prlx", history=True)
```

### TUI

```bash
prlx diff a.prlx b.prlx --tui
```

Interactive terminal UI for navigating divergences across warps.

## Environment Variables

| Variable | Default | What it does |
|---|---|---|
| `PRLX_TRACE` | `trace.prlx` | Output path |
| `PRLX_SNAPSHOT_DEPTH` | `0` | Per-lane operand ring buffer size |
| `PRLX_HISTORY_DEPTH` | `0` | Time-travel value ring buffer size |
| `PRLX_SAMPLE_RATE` | `1` | Record 1 in N events |
| `PRLX_COMPRESS` | `0` | zstd compress the trace |
| `PRLX_ENABLED` | `1` | Kill switch |

## How It Works

PRLX has three backends for instrumenting GPU code:

1. **LLVM pass** (`lib/pass/`) — loaded as `-fpass-plugin` during compilation (clang) or injected between Triton's `make_llir` and `make_ptx` stages. Walks the NVPTX IR, inserts calls to `__prlx_record_branch` / `__prlx_record_value` at every branch and comparison. For Triton's branchless single-BB kernels, it detects predicated ops (`icmp` feeding inline PTX asm or `select`).

2. **NVBit tool** (`lib/nvbit_tool/`) — SASS-level binary instrumentation via NVBit. Works on closed-source kernels where you don't have IR access.

3. **Runtime** (`lib/runtime/`) — device-side ring buffers (one per warp) that record events, value history, and per-lane comparison operand snapshots. Host hooks (`prlx_pre_launch` / `prlx_post_launch`) manage allocation and readback.

Traces are written to `.prlx` files (custom binary format, optionally zstd-compressed). The **differ** (`differ/`, Rust) aligns event streams with bounded lookahead, classifies divergences (branch direction, path length, missing events), and can display per-lane operand diffs.

## Layout

```
lib/pass/           LLVM instrumentation pass (libPrlxPass.so)
lib/runtime/        device-side recording + host hooks
lib/nvbit_tool/     NVBit binary instrumentation backend
lib/common/         shared trace format header
differ/             Rust differ + TUI (prlx-diff)
python/prlx/        trace reader, Triton hook, runtime FFI, CLI
examples/           demo kernels (branch, loop, matmul, occupancy)
```

## License

MIT
