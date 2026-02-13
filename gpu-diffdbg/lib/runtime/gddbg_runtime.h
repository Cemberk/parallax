#ifndef GDDBG_RUNTIME_H
#define GDDBG_RUNTIME_H

#include "../common/trace_format.h"
#include <cuda_runtime.h>

// Per-warp buffer structure (in global memory)
// Needs to be visible to both host and device code
typedef struct {
    WarpBufferHeader header;
    TraceEvent events[GDDBG_EVENTS_PER_WARP];
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
extern __device__ TraceBuffer* g_gddbg_buffer;

// Device-side recording functions (called by instrumented code)
extern "C" {
    __device__ void __gddbg_record_branch(
        uint32_t site_id,
        uint32_t condition,
        uint32_t operand_a
    );

    __device__ void __gddbg_record_shmem_store(
        uint32_t site_id,
        uint32_t address,
        uint32_t value
    );

    __device__ void __gddbg_record_atomic(
        uint32_t site_id,
        uint32_t address,
        uint32_t operand,
        uint32_t result
    );

    __device__ void __gddbg_record_func(
        uint32_t site_id,
        uint8_t  is_entry,
        uint32_t arg0
    );
}

// Utility functions
__device__ __forceinline__ uint32_t __gddbg_warp_id();
__device__ __forceinline__ uint32_t __gddbg_lane_id();

// Host-side API (also available when compiling with nvcc)
extern "C" {
    void gddbg_init(void);
    void gddbg_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim);
    void gddbg_post_launch(void);
    void gddbg_shutdown(void);
}

#else
// Compiling with regular C++ compiler - host-side only

// Forward declaration of device symbol
extern __device__ TraceBuffer* g_gddbg_buffer;

#ifdef __cplusplus
extern "C" {
#endif

void gddbg_init(void);
void gddbg_pre_launch(const char* kernel_name, dim3 gridDim, dim3 blockDim);
void gddbg_post_launch(void);
void gddbg_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif // __CUDACC__

#endif // GDDBG_RUNTIME_H
