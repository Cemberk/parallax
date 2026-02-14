#ifndef PRLX_RUNTIME_H
#define PRLX_RUNTIME_H

#include "../common/trace_format.h"
#include <cuda_runtime.h>

// Per-warp buffer structure (in global memory)
// Needs to be visible to both host and device code
typedef struct {
    WarpBufferHeader header;
    TraceEvent events[PRLX_EVENTS_PER_WARP];
} WarpBuffer;

// Global trace buffer structure
// Needs to be visible to both host and device code
typedef struct {
    TraceFileHeader file_header;
    WarpBuffer warps[];  // Flexible array member
} TraceBuffer;

#ifdef __CUDACC__
// Compiling with nvcc - declare both device and host APIs

// Global device pointer to trace buffer (set by host via cudaMemcpyToSymbol)
extern __device__ TraceBuffer* g_prlx_buffer;

// History ring buffer (separate allocation, set when PRLX_HISTORY_DEPTH > 0)
extern __device__ char* g_prlx_history_buffer;
extern __device__ uint32_t g_prlx_history_depth;

// Sampling rate (1 = record all events, N = record 1 out of every N events)
extern __device__ uint32_t __prlx_sample_rate;

// Snapshot (crash dump) ring buffer (set by host when PRLX_SNAPSHOT_DEPTH > 0)
extern __device__ char* g_prlx_snapshot_buffer;
extern __device__ uint32_t g_prlx_snapshot_depth;

// ---- Region of Interest (ROI) Toggle ----
// Users can enable/disable recording from within their kernel code.
// This allows targeting specific code paths or time slices.
//
// Usage:
//   if (threadIdx.x == 0 && error_metric > threshold) {
//       prlx_enable();
//   }
//   ... buggy code ...
//   prlx_disable();
//
extern __device__ volatile int __prlx_recording_enabled;

__device__ __forceinline__ void prlx_enable()  { __prlx_recording_enabled = 1; }
__device__ __forceinline__ void prlx_disable() { __prlx_recording_enabled = 0; }

// Device-side recording functions (called by instrumented code)
extern "C" {
    __device__ void __prlx_record_branch(
        uint32_t site_id,
        uint32_t condition,
        uint32_t operand_a
    );

    __device__ void __prlx_record_shmem_store(
        uint32_t site_id,
        uint32_t address,
        uint32_t value
    );

    __device__ void __prlx_record_atomic(
        uint32_t site_id,
        uint32_t address,
        uint32_t operand,
        uint32_t result
    );

    __device__ void __prlx_record_func(
        uint32_t site_id,
        uint8_t  is_entry,
        uint32_t arg0
    );

    // History (time-travel): record a variable value into per-warp circular buffer
    __device__ void __prlx_record_value(
        uint32_t site_id,
        uint32_t value
    );

    // Snapshot (crash dump): capture per-lane comparison operands via __shfl_sync
    // ALL active lanes must call this (convergent) — lane 0 writes to snapshot ring
    __device__ void __prlx_record_snapshot(
        uint32_t site_id,
        uint32_t lhs,
        uint32_t rhs,
        uint32_t cmp_pred
    );
}

// Utility functions
__device__ __forceinline__ uint32_t __prlx_warp_id();
__device__ __forceinline__ uint32_t __prlx_lane_id();

// Host-side API (also available when compiling with nvcc)
extern "C" {
    void prlx_init(void);
    void prlx_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim);
    void prlx_post_launch(void);
    void prlx_shutdown(void);

    // Session API: capture multiple kernel launches into a directory with manifest
    void prlx_session_begin(const char* name);
    void prlx_session_end(void);
}

#else
// Compiling with regular C++ compiler - host-side only

extern __device__ TraceBuffer* g_prlx_buffer;

#ifdef __cplusplus
extern "C" {
#endif

void prlx_init(void);
void prlx_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim);
void prlx_post_launch(void);
void prlx_shutdown(void);
void prlx_session_begin(const char* name);
void prlx_session_end(void);

#ifdef __cplusplus
}
#endif

#endif // __CUDACC__

#endif // PRLX_RUNTIME_H
