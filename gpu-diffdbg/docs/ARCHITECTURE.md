# GPU DiffDbg Architecture

## Overview

GPU DiffDbg is a differential debugging tool for CUDA GPU kernels that enables developers to compare two execution traces and identify exactly where and why they diverge. The system consists of four major components working together:

1. **LLVM Instrumentation Pass** - Modifies GPU code at compile-time
2. **Device Runtime** - Records execution traces on the GPU
3. **Host Runtime** - Manages trace buffers and file I/O
4. **Trace Differ** - Analyzes and compares traces

## Critical Design Decisions

The following design decisions are **non-negotiable** and are based on expert review. Violating any of these will result in a broken or unusable tool.

### Death Valley 1: The SIMT Divergence Trap

**Problem:** CUDA execution is SIMT (Single Instruction, Multiple Thread). When a warp hits a divergent branch, the hardware executes BOTH paths sequentially with different active masks. A warp doesn't go Left OR Right — it goes Left AND Right.

**Solution:** We record the **full 32-bit active lane mask** at every instrumentation point using `__activemask()`. A divergence occurs when:
- The branch direction differs, OR
- The active mask differs (different threads active even with same branch direction)

The active mask IS the divergence signal. We never hash or compress it — the exact mask is needed to detect which threads diverged.

### Death Valley 2: The ABI / Argument Injection Nightmare

**Problem:** Modifying `__global__` kernel function signatures to add a `TraceBuffer*` argument breaks host-side argument marshaling. The host packs arguments into a buffer — if the device expects N+1 args but the host sends N, you get illegal memory access at launch.

**Solution:** We use a **device-side global variable** instead of modifying kernel signatures:

```cuda
__device__ TraceBuffer* g_gddbg_buffer;
```

The host runtime sets this pointer via `cudaMemcpyToSymbol` before each kernel launch. The LLVM pass emits loads from this global variable, not from a function argument. **Zero ABI changes. Zero host-side patching.**

### Death Valley 3: Memory Contention and the Heisenbug Effect

**Problem:** Adding global memory atomics and stores at every branch creates L2 cache pressure that can slow code by >100%. This timing perturbation can hide race conditions you're trying to debug.

**Solutions:**
1. Use **L1 cache bypass** (`st.global.cg` PTX instruction) for trace stores
2. Design trace events to be **shrinkable to 8 bytes** for compact mode
3. Support **configurable sampling** (`GDDBG_SAMPLE_RATE=N`)

## Component Details

### 1. LLVM Instrumentation Pass

**Location:** `lib/pass/GpuDiffDbgPass.cpp`

The pass operates on LLVM IR during Clang's CUDA compilation pipeline, targeting the NVPTX backend.

**What it instruments:**
- Conditional branches (`br i1 %cond, ...`)
- Shared memory stores (address space 3)
- Atomic operations (`atomicrmw`, `cmpxchg`)
- Function entry/exit points

**Key implementation details:**
- Instruments at LLVM IR level (not PTX or SASS) to preserve source-level debug info
- Uses deterministic FNV-1a hashing of `(filename:function:line:column)` for site IDs
- Instruments BasicBlock terminators, not arbitrary instructions (avoids phi node issues)
- Declares runtime functions as external, resolved at link time

**Site ID Generation:**
```cpp
// Deterministic hash: filename:function:line:column:event_type
uint32_t site_id = fnv1a_hash("kernel.cu:compute:47:12:0");
```

Site IDs MUST be stable across compilations. Sequential counters would break when code changes.

### 2. Device Runtime

**Location:** `lib/runtime/gddbg_runtime.cu`

The device runtime provides recording functions called by instrumented code.

**Data structures:**
- `TraceEvent` (16 bytes): site_id, event_type, branch_dir, active_mask, value_a
- `WarpBuffer`: Per-warp ring buffer with atomic write index
- `TraceBuffer`: Global buffer containing all warp buffers

**Recording strategy:**
- Only lane 0 of each warp records events (avoids redundancy)
- Uses `atomicAdd` on per-warp write index
- Stores with `st.global.cg` PTX instruction (L1 bypass)
- Captures `__activemask()` at every event

**Overhead optimization:**
```cuda
__device__ __forceinline__ void __gddbg_store_event(TraceEvent* dst, const TraceEvent& evt) {
    // PTX inline asm for cache-global store (bypass L1)
    asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};" ...);
}
```

### 3. Host Runtime

**Location:** `lib/runtime/gddbg_host.cpp`

The host runtime manages trace buffer lifecycle.

**Workflow:**
1. `gddbg_init()` - Called at program startup (constructor), reads env vars
2. `gddbg_pre_launch()` - Allocates GPU buffer, sets `g_gddbg_buffer` via `cudaMemcpyToSymbol`
3. Kernel executes, records events
4. `gddbg_post_launch()` - Copies buffer to host, writes to file
5. `gddbg_shutdown()` - Cleanup (destructor)

**Critical implementation:**
```cpp
// Set the device global variable to point to the trace buffer
cudaMemcpyToSymbol(g_gddbg_buffer, &d_trace_buffer, sizeof(void*));
```

This is the key technique that avoids kernel signature modification.

### 4. Trace Differ

**Location:** `tools/gddbg-diff/src/`

The differ compares two trace files and reports divergences.

**Algorithm:**
1. Parse both trace files (memory-mapped for efficiency)
2. Validate compatibility (same kernel, same grid/block dims)
3. For each warp, walk event streams in lockstep
4. Detect divergences:
   - Site ID mismatch → different code path
   - Branch direction mismatch → different decision
   - Active mask mismatch → SIMT divergence
   - Value mismatch → same path, different data
5. Attempt resync using LCS-like lookahead (up to 10 events)
6. Report first divergence per warp

**Resync strategy:**
```rust
fn try_resync(events_a: &[TraceEvent], events_b: &[TraceEvent], limit: usize) -> Option<(usize, usize)> {
    // Scan ahead to find next matching site_id pair
    // This handles different loop iteration counts
}
```

## Trace File Format

**Binary format** (sequential, block-major):

```
[File Header (128 bytes)]
  - magic: "GDDBGGPU\0"
  - version, kernel name, grid/block dims
  - total_warp_slots, events_per_warp

[Warp 0 Data]
  [WarpBufferHeader (16 bytes)]
  [TraceEvent array (16 bytes × num_events)]

[Warp 1 Data]
  ...

[Warp N Data]
```

**Why binary?** Fast sequential reading, mmap-friendly, no parsing overhead.

**Why block-major?** Traces are analyzed warp-by-warp, so keeping each warp's data contiguous improves cache locality.

## Execution Flow Example

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Developer compiles kernel with LLVM pass                 │
│    clang++ --cuda-gpu-arch=sm_80 -fplugin=... kernel.cu    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Pass instruments branches, inserts calls to:             │
│    __gddbg_record_branch(site_id, condition, operand_a)    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Binary links against runtime library                     │
│    Constructor: gddbg_init() reads GDDBG_TRACE env var     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Before kernel launch: gddbg_pre_launch()                 │
│    - Allocates GPU buffer                                   │
│    - Sets g_gddbg_buffer via cudaMemcpyToSymbol           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Kernel executes                                          │
│    - Instrumented code calls __gddbg_record_branch()       │
│    - Lane 0 of each warp writes events to warp buffer      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. After kernel completes: gddbg_post_launch()             │
│    - Copies buffer from GPU to host                         │
│    - Writes trace.gddbg file                                │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Developer runs differ:                                   │
│    gddbg-diff trace_a.gddbg trace_b.gddbg                  │
│    → "Divergence at site 0x12345678, warp 3, block (1,0)"  │
└─────────────────────────────────────────────────────────────┘
```

## Performance Characteristics

**Target overhead:** <15% for branch-only tracing

**Overhead sources:**
1. Global memory writes (16 bytes per event per warp)
2. Atomic increment (one per event per warp, but only lane 0)
3. Register pressure (buffer pointer, write index)

**Mitigation strategies:**
- L1 cache bypass (`st.global.cg`) - avoids polluting kernel's working set
- Only lane 0 records - 32× reduction in atomic contention
- Ring buffer - no dynamic allocation, predictable memory access
- Sampling mode - trace every Nth warp for large grids

**Measured overhead (estimated):**
- Branch-heavy code (>10 branches per 100 instructions): 15-20%
- Straight-line code (<2 branches per 100 instructions): <5%
- Shared memory + atomics tracing: 25-40%

## Future Optimizations

1. **Compact event format** (8 bytes): Compress site_id to 16 bits, pack event_type/branch_dir
2. **Compression**: zstd on warp trace data (not headers)
3. **Selective instrumentation**: Pragma-based or function attribute filtering
4. **GPU-side filtering**: Only record divergent branches (compare with reference mask)
5. **Two-level ring buffer**: Per-warp buffer + global overflow buffer

## Known Limitations

1. **No cross-kernel tracking** - Each kernel launch produces a separate trace
2. **No dynamic parallelism support** - Child kernels are not traced
3. **Limited to 4096 events per warp** (default) - Ring buffer can overflow
4. **No thread-level divergence** - Records at warp level only (by design)
5. **NVPTX only** - AMD ROCm/HIP support requires separate pass

## References

- CUDA Programming Guide (SIMT execution model)
- LLVM NVPTX Backend Documentation
- PTX ISA (cache operators: `st.global.cg`)
- "GPUDet: Deterministic GPU Execution" (ASPLOS 2013)
- Meta CUTracer (Feb 2026) - inspiration for binary instrumentation approach
