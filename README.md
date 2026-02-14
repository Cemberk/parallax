# PRLX

Differential debugger for CUDA/Triton GPU kernels. Records per-warp execution traces, diffs them, tells you exactly **where**, **who**, **what**, and **why** two runs diverged.

## What It Does

Run your kernel twice with different inputs. PRLX captures branch decisions, active masks, comparison operands (per-lane), and value history. The differ pinpoints divergences down to the warp level:

```
Site: 0xfbe6edc1 (2 warps affected)
  Warp 1 @ event 1: Branch Direction
    Trace A: TAKEN
    Trace B: NOT-TAKEN
    Per-Lane Operand Snapshot (icmp sgt):
    Lane       A:lhs       A:rhs       B:lhs       B:rhs
       0          32          10          32          64 <<<
       1          33          10          33          64 <<<
      ...
```

## Installation

```bash
pip install prlx
```

**Requirements:** CUDA Toolkit 12.0+ and LLVM 18, 19, or 20 on your system.

<details>
<summary>Building from source</summary>

```bash
# Build native components
cmake -B build && cmake --build build
cd differ && cargo build --release && cd ..

# Install in development mode
pip install -e .
```

Requires: CMake 3.20+, LLVM/Clang 18-20, CUDA Toolkit 12.0+, Rust (stable).
</details>

## Quick Start

### CUDA C Kernels

```bash
# Compile with automatic instrumentation
./prlx compile my_kernel.cu -o my_kernel

# Capture two traces
PRLX_TRACE=a.prlx PRLX_SNAPSHOT_DEPTH=32 ./my_kernel --good-input
PRLX_TRACE=b.prlx PRLX_SNAPSHOT_DEPTH=32 ./my_kernel --bad-input

# Diff
./prlx diff a.prlx b.prlx --history
```

### Triton Kernels

```python
import prlx
prlx.enable()   # hooks Triton's compilation pipeline

@triton.jit
def my_kernel(...):
    ...  # branches auto-instrumented, no code changes needed

os.environ["PRLX_TRACE"] = "a.prlx"
my_kernel[grid](...)

os.environ["PRLX_TRACE"] = "b.prlx"
my_kernel[grid](...)
```

### Interactive TUI

```bash
./prlx diff a.prlx b.prlx --tui
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PRLX_TRACE` | `trace.prlx` | Output trace path |
| `PRLX_SNAPSHOT_DEPTH` | `0` | Per-lane operand capture (ring size) |
| `PRLX_HISTORY_DEPTH` | `0` | Time-travel value history (ring size) |
| `PRLX_SAMPLE_RATE` | `1` | Record 1-in-N events |
| `PRLX_COMPRESS` | `0` | zstd compression |
| `PRLX_ENABLED` | `1` | Kill switch |

## Project Structure

```
lib/pass/          LLVM pass — instruments NVPTX modules
lib/runtime/       Device-side recording + host pre/post launch hooks
lib/nvbit_tool/    NVBit binary instrumentation (closed-source kernels)
differ/            Rust differ + TUI
python/prlx/       Triton integration + trace reader
prlx               CLI driver (compile, diff, run, check, triton)
examples/          Demo kernels
```

## Runtime Requirements

| Dependency | Needed For | Install |
|---|---|---|
| CUDA Toolkit 12.0+ | Kernel compilation + tracing | `apt install nvidia-cuda-toolkit` or [NVIDIA](https://developer.nvidia.com/cuda-downloads) |
| LLVM 18, 19, or 20 | `prlx compile` and Triton hook | `apt install llvm-20` or via [apt.llvm.org](https://apt.llvm.org) |

The `prlx-diff` differ and Python API work without any external dependencies.

## License

MIT
