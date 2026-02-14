#include "gddbg_runtime.h"
#include <cuda_runtime.h>

// Global device pointer to trace buffer (set by host via cudaMemcpyToSymbol)
// This is the key to avoiding kernel signature modification (Death Valley 2)
__device__ TraceBuffer* g_gddbg_buffer = nullptr;

// Region of Interest (ROI) toggle
// 1 = recording enabled (default), 0 = recording disabled
// Users call gddbg_enable() / gddbg_disable() from kernel code
__device__ volatile int __gddbg_recording_enabled = 1;

// History ring buffer globals (set by host when GDDBG_HISTORY_DEPTH > 0)
__device__ char* g_gddbg_history_buffer = nullptr;
__device__ uint32_t g_gddbg_history_depth = 0;

// Compute linear warp ID within the grid
__device__ __forceinline__ uint32_t __gddbg_warp_id() {
    // Thread ID within block
    uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x
                 + threadIdx.z * blockDim.x * blockDim.y;

    // Warp ID within block (each warp has 32 threads)
    uint32_t warp_in_block = tid / 32;

    // Linear block ID
    uint32_t linear_block = blockIdx.x
                          + blockIdx.y * gridDim.x
                          + blockIdx.z * gridDim.x * gridDim.y;

    // Warps per block
    uint32_t warps_per_block = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;

    return linear_block * warps_per_block + warp_in_block;
}

// Get lane ID within warp (0-31)
__device__ __forceinline__ uint32_t __gddbg_lane_id() {
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// Cache-bypassing store for 16-byte aligned data (Death Valley 3 mitigation)
// Uses st.global.cg (cache at L2 only, bypass L1)
__device__ __forceinline__ void __gddbg_store_16b(void* dst, const uint32_t* src) {
    asm volatile(
        "st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};"
        :
        : "l"(dst),
          "r"(src[0]),
          "r"(src[1]),
          "r"(src[2]),
          "r"(src[3])
        : "memory"
    );
}

// Cache-bypassing store to avoid thrashing L1 (Death Valley 3 mitigation)
// Uses st.global.cg (cache at L2 only, bypass L1)
__device__ __forceinline__ void __gddbg_store_event(TraceEvent* dst, const TraceEvent& evt) {
    __gddbg_store_16b(dst, reinterpret_cast<const uint32_t*>(&evt));
}

// Record a branch event
extern "C" __device__ void __gddbg_record_branch(
    uint32_t site_id,
    uint32_t condition,
    uint32_t operand_a
) {
    // Only lane 0 of each warp records the event to avoid redundant writes
    if (__gddbg_lane_id() != 0) return;

    // Check ROI toggle - skip if user has disabled recording
    if (!__gddbg_recording_enabled) return;

    // Check if tracing is enabled (buffer pointer is non-null)
    TraceBuffer* buf = g_gddbg_buffer;
    if (buf == nullptr) return;

    uint32_t warp = __gddbg_warp_id();

    // Manual offset calculation to avoid sizeof(WarpBuffer) mismatch
    // Each warp buffer = WarpBufferHeader + (GDDBG_EVENTS_PER_WARP * TraceEvent)
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + GDDBG_EVENTS_PER_WARP * sizeof(TraceEvent);
    char* warp_base = ((char*)buf) + sizeof(TraceFileHeader) + warp * warp_buffer_size;
    WarpBufferHeader* header = (WarpBufferHeader*)warp_base;
    TraceEvent* events = (TraceEvent*)(warp_base + sizeof(WarpBufferHeader));

    // Atomically increment write index
    uint32_t idx = atomicAdd(&header->write_idx, 1);

    // Check for buffer overflow
    if (idx >= GDDBG_EVENTS_PER_WARP) {
        atomicAdd(&header->overflow_count, 1);
        return;
    }

    // Get the FULL active lane mask - CRITICAL for SIMT divergence detection
    // Do NOT hash or compress this value (Death Valley 1)
    uint32_t active_mask = __activemask();

    // Build event
    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_BRANCH;
    evt.branch_dir = (uint8_t)(condition & 1);
    evt._reserved = 0;
    evt.active_mask = active_mask;
    evt.value_a = operand_a;

    // Store with cache bypass
    __gddbg_store_event(&events[idx], evt);
}

// Record a shared memory store event
extern "C" __device__ void __gddbg_record_shmem_store(
    uint32_t site_id,
    uint32_t address,
    uint32_t value
) {
    if (__gddbg_lane_id() != 0) return;
    if (!__gddbg_recording_enabled) return;

    TraceBuffer* buf = g_gddbg_buffer;
    if (buf == nullptr) return;

    uint32_t warp = __gddbg_warp_id();

    // Manual offset calculation
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + GDDBG_EVENTS_PER_WARP * sizeof(TraceEvent);
    char* warp_base = ((char*)buf) + sizeof(TraceFileHeader) + warp * warp_buffer_size;
    WarpBufferHeader* header = (WarpBufferHeader*)warp_base;
    TraceEvent* events = (TraceEvent*)(warp_base + sizeof(WarpBufferHeader));

    uint32_t idx = atomicAdd(&header->write_idx, 1);
    if (idx >= GDDBG_EVENTS_PER_WARP) {
        atomicAdd(&header->overflow_count, 1);
        return;
    }

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_SHMEM_STORE;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = value;

    __gddbg_store_event(&events[idx], evt);
}

// Record an atomic operation event
extern "C" __device__ void __gddbg_record_atomic(
    uint32_t site_id,
    uint32_t address,
    uint32_t operand,
    uint32_t result
) {
    if (__gddbg_lane_id() != 0) return;
    if (!__gddbg_recording_enabled) return;

    TraceBuffer* buf = g_gddbg_buffer;
    if (buf == nullptr) return;

    uint32_t warp = __gddbg_warp_id();

    // Manual offset calculation
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + GDDBG_EVENTS_PER_WARP * sizeof(TraceEvent);
    char* warp_base = ((char*)buf) + sizeof(TraceFileHeader) + warp * warp_buffer_size;
    WarpBufferHeader* header = (WarpBufferHeader*)warp_base;
    TraceEvent* events = (TraceEvent*)(warp_base + sizeof(WarpBufferHeader));

    uint32_t idx = atomicAdd(&header->write_idx, 1);
    if (idx >= GDDBG_EVENTS_PER_WARP) {
        atomicAdd(&header->overflow_count, 1);
        return;
    }

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_ATOMIC;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = operand;

    __gddbg_store_event(&events[idx], evt);
}

// Record a function entry/exit event
extern "C" __device__ void __gddbg_record_func(
    uint32_t site_id,
    uint8_t  is_entry,
    uint32_t arg0
) {
    if (__gddbg_lane_id() != 0) return;
    if (!__gddbg_recording_enabled) return;

    TraceBuffer* buf = g_gddbg_buffer;
    if (buf == nullptr) return;

    uint32_t warp = __gddbg_warp_id();

    // Manual offset calculation
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + GDDBG_EVENTS_PER_WARP * sizeof(TraceEvent);
    char* warp_base = ((char*)buf) + sizeof(TraceFileHeader) + warp * warp_buffer_size;
    WarpBufferHeader* header = (WarpBufferHeader*)warp_base;
    TraceEvent* events = (TraceEvent*)(warp_base + sizeof(WarpBufferHeader));

    uint32_t idx = atomicAdd(&header->write_idx, 1);
    if (idx >= GDDBG_EVENTS_PER_WARP) {
        atomicAdd(&header->overflow_count, 1);
        return;
    }

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = is_entry ? EVENT_FUNC_ENTRY : EVENT_FUNC_EXIT;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = arg0;

    __gddbg_store_event(&events[idx], evt);
}

// Record a value into the per-warp circular history ring buffer (time-travel)
extern "C" __device__ void __gddbg_record_value(
    uint32_t site_id,
    uint32_t value
) {
    if (__gddbg_lane_id() != 0) return;
    if (!__gddbg_recording_enabled) return;

    // Check if history is enabled
    char* hist_buf = g_gddbg_history_buffer;
    if (hist_buf == nullptr) return;

    uint32_t depth = g_gddbg_history_depth;
    if (depth == 0) return;

    uint32_t warp = __gddbg_warp_id();

    // Per-warp history ring: [HistoryRingHeader][HistoryEntry * depth]
    size_t ring_size = sizeof(HistoryRingHeader) + depth * sizeof(HistoryEntry);
    char* ring_base = hist_buf + warp * ring_size;
    HistoryRingHeader* ring = (HistoryRingHeader*)ring_base;
    HistoryEntry* entries = (HistoryEntry*)(ring_base + sizeof(HistoryRingHeader));

    // Circular write: atomically increment and wrap
    uint32_t raw_idx = atomicAdd(&ring->write_idx, 1);
    uint32_t slot = raw_idx % depth;

    // Track total writes for ordering (also used to detect wrap)
    uint32_t seq = atomicAdd(&ring->total_writes, 1);

    // Build history entry
    HistoryEntry entry;
    entry.site_id = site_id;
    entry.value = value;
    entry.seq = seq;
    entry._pad = 0;

    // Store with cache bypass
    __gddbg_store_16b(&entries[slot], reinterpret_cast<const uint32_t*>(&entry));
}
