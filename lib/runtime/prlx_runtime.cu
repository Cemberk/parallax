#include "prlx_runtime.h"
#include <cuda_runtime.h>

// Global device pointer to trace buffer (set by host via cudaMemcpyToSymbol)
// This is the key to avoiding kernel signature modification (Death Valley 2)
__device__ TraceBuffer* g_prlx_buffer = nullptr;

// Region of Interest (ROI) toggle
// 1 = recording enabled (default), 0 = recording disabled
// Users call prlx_enable() / prlx_disable() from kernel code
__device__ volatile int __prlx_recording_enabled = 1;

// History ring buffer globals (set by host when PRLX_HISTORY_DEPTH > 0)
__device__ char* g_prlx_history_buffer = nullptr;
__device__ uint32_t g_prlx_history_depth = 0;

// Sampling rate: 1 = record all (default), N = record 1 out of every N events per warp
__device__ uint32_t __prlx_sample_rate = 1;

// Snapshot (crash dump) ring buffer globals (set by host when PRLX_SNAPSHOT_DEPTH > 0)
__device__ char* g_prlx_snapshot_buffer = nullptr;
__device__ uint32_t g_prlx_snapshot_depth = 0;

// Compute linear warp ID within the grid
__device__ __forceinline__ uint32_t __prlx_warp_id() {
    uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x
                 + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t warp_in_block = tid / 32;
    uint32_t linear_block = blockIdx.x
                          + blockIdx.y * gridDim.x
                          + blockIdx.z * gridDim.x * gridDim.y;
    uint32_t warps_per_block = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;

    return linear_block * warps_per_block + warp_in_block;
}

__device__ __forceinline__ uint32_t __prlx_lane_id() {
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// Cache-bypassing store for 16-byte aligned data (Death Valley 3 mitigation)
// Uses st.global.cg (cache at L2 only, bypass L1)
__device__ __forceinline__ void __prlx_store_16b(void* dst, const uint32_t* src) {
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
__device__ __forceinline__ void __prlx_store_event(TraceEvent* dst, const TraceEvent& evt) {
    __prlx_store_16b(dst, reinterpret_cast<const uint32_t*>(&evt));
}

// Claim a trace event slot for the current warp.
// Returns pointer to the event slot, or nullptr if the event should be skipped
// (wrong lane, recording disabled, buffer full, sampling skip, or overflow).
__device__ __forceinline__ TraceEvent* __prlx_claim_slot() {
    if (__prlx_lane_id() != 0) return nullptr;
    if (!__prlx_recording_enabled) return nullptr;

    TraceBuffer* buf = g_prlx_buffer;
    if (buf == nullptr) return nullptr;

    uint32_t warp = __prlx_warp_id();

    // Manual offset calculation to avoid sizeof(WarpBuffer) mismatch
    // Each warp buffer = WarpBufferHeader + (PRLX_EVENTS_PER_WARP * TraceEvent)
    size_t warp_buffer_size = sizeof(WarpBufferHeader) + PRLX_EVENTS_PER_WARP * sizeof(TraceEvent);
    char* warp_base = ((char*)buf) + sizeof(TraceFileHeader) + warp * warp_buffer_size;
    WarpBufferHeader* header = (WarpBufferHeader*)warp_base;
    TraceEvent* events = (TraceEvent*)(warp_base + sizeof(WarpBufferHeader));

    // Sampling: count all events, but only record 1/N
    uint32_t evt_count = atomicAdd(&header->total_event_count, 1);
    if (__prlx_sample_rate > 1 && (evt_count % __prlx_sample_rate) != 0) return nullptr;

    uint32_t idx = atomicAdd(&header->write_idx, 1);
    if (idx >= PRLX_EVENTS_PER_WARP) {
        atomicAdd(&header->overflow_count, 1);
        return nullptr;
    }

    return &events[idx];
}

// Record a branch event
extern "C" __device__ void __prlx_record_branch(
    uint32_t site_id,
    uint32_t condition,
    uint32_t operand_a
) {
    TraceEvent* slot = __prlx_claim_slot();
    if (!slot) return;

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_BRANCH;
    evt.branch_dir = (uint8_t)(condition & 1);
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = operand_a;

    __prlx_store_event(slot, evt);
}

// Record a shared memory store event
extern "C" __device__ void __prlx_record_shmem_store(
    uint32_t site_id,
    uint32_t address,
    uint32_t value
) {
    TraceEvent* slot = __prlx_claim_slot();
    if (!slot) return;

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_SHMEM_STORE;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = value;

    __prlx_store_event(slot, evt);
}

// Record an atomic operation event
extern "C" __device__ void __prlx_record_atomic(
    uint32_t site_id,
    uint32_t address,
    uint32_t operand,
    uint32_t result
) {
    TraceEvent* slot = __prlx_claim_slot();
    if (!slot) return;

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = EVENT_ATOMIC;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = operand;

    __prlx_store_event(slot, evt);
}

// Record a function entry/exit event
extern "C" __device__ void __prlx_record_func(
    uint32_t site_id,
    uint8_t  is_entry,
    uint32_t arg0
) {
    TraceEvent* slot = __prlx_claim_slot();
    if (!slot) return;

    TraceEvent evt;
    evt.site_id = site_id;
    evt.event_type = is_entry ? EVENT_FUNC_ENTRY : EVENT_FUNC_EXIT;
    evt.branch_dir = 0;
    evt._reserved = 0;
    evt.active_mask = __activemask();
    evt.value_a = arg0;

    __prlx_store_event(slot, evt);
}

// Record a value into the per-warp circular history ring buffer (time-travel)
extern "C" __device__ void __prlx_record_value(
    uint32_t site_id,
    uint32_t value
) {
    if (__prlx_lane_id() != 0) return;
    if (!__prlx_recording_enabled) return;

    char* hist_buf = g_prlx_history_buffer;
    if (hist_buf == nullptr) return;

    uint32_t depth = g_prlx_history_depth;
    if (depth == 0) return;

    uint32_t warp = __prlx_warp_id();

    // Per-warp history ring: [HistoryRingHeader][HistoryEntry * depth]
    size_t ring_size = sizeof(HistoryRingHeader) + depth * sizeof(HistoryEntry);
    char* ring_base = hist_buf + warp * ring_size;
    HistoryRingHeader* ring = (HistoryRingHeader*)ring_base;
    HistoryEntry* entries = (HistoryEntry*)(ring_base + sizeof(HistoryRingHeader));

    // Circular write
    uint32_t raw_idx = atomicAdd(&ring->write_idx, 1);
    uint32_t slot = raw_idx % depth;

    // Track total writes for ordering (also used to detect wrap)
    uint32_t seq = atomicAdd(&ring->total_writes, 1);

    HistoryEntry entry;
    entry.site_id = site_id;
    entry.value = value;
    entry.seq = seq;
    entry._pad = 0;

    __prlx_store_16b(&entries[slot], reinterpret_cast<const uint32_t*>(&entry));
}

// Record per-lane comparison operands for crash dump / divergence analysis
// CRITICAL: ALL active lanes must execute this function (convergent).
// The __shfl_sync intrinsic requires all lanes indicated by the mask to participate.
// Only lane 0 writes the actual snapshot entry to global memory.
extern "C" __device__ void __prlx_record_snapshot(
    uint32_t site_id,
    uint32_t lhs,
    uint32_t rhs,
    uint32_t cmp_pred
) {
    char* snap_buf = g_prlx_snapshot_buffer;
    if (snap_buf == nullptr) return;

    uint32_t depth = g_prlx_snapshot_depth;
    if (depth == 0) return;

    if (!__prlx_recording_enabled) return;

    // Respect sample rate to reduce overhead on hot comparisons
    if (__prlx_sample_rate > 1) {
        uint32_t warp = __prlx_warp_id();
        uint32_t ring_size = sizeof(SnapshotRingHeader) + depth * sizeof(SnapshotEntry);
        SnapshotRingHeader* ring = (SnapshotRingHeader*)(snap_buf + warp * ring_size);
        if ((ring->total_writes % __prlx_sample_rate) != 0) return;
    }

    // Get active mask BEFORE any conditional returns
    uint32_t active = __activemask();
    uint32_t lane = __prlx_lane_id();

    // ALL active lanes participate in shuffle to gather operand values to lane 0
    uint32_t gathered_lhs[32];
    uint32_t gathered_rhs[32];
    for (int src = 0; src < 32; src++) {
        uint32_t l = __shfl_sync(active, lhs, src);
        uint32_t r = __shfl_sync(active, rhs, src);
        if (lane == 0) {
            gathered_lhs[src] = ((active >> src) & 1) ? l : 0;
            gathered_rhs[src] = ((active >> src) & 1) ? r : 0;
        }
    }

    // Only lane 0 writes the snapshot entry
    if (lane != 0) return;

    uint32_t warp = __prlx_warp_id();

    // Per-warp snapshot ring: [SnapshotRingHeader][SnapshotEntry * depth]
    size_t entry_size = sizeof(SnapshotEntry);  // 288 bytes
    size_t ring_size = sizeof(SnapshotRingHeader) + depth * entry_size;
    char* ring_base = snap_buf + warp * ring_size;
    SnapshotRingHeader* ring = (SnapshotRingHeader*)ring_base;
    SnapshotEntry* entries = (SnapshotEntry*)(ring_base + sizeof(SnapshotRingHeader));

    // Circular write
    uint32_t raw_idx = atomicAdd(&ring->write_idx, 1);
    uint32_t slot = raw_idx % depth;

    uint32_t seq = atomicAdd(&ring->total_writes, 1);

    SnapshotEntry entry;
    entry.site_id = site_id;
    entry.active_mask = active;
    entry.seq = seq;
    entry.cmp_predicate = cmp_pred;
    for (int i = 0; i < 32; i++) {
        entry.lhs_values[i] = gathered_lhs[i];
        entry.rhs_values[i] = gathered_rhs[i];
    }
    entry._pad[0] = 0;
    entry._pad[1] = 0;
    entry._pad[2] = 0;
    entry._pad[3] = 0;

    // Store using multiple v4.u32 cache-bypassing stores
    // 288 bytes = 72 x u32 = 18 x v4.u32 stores
    const uint32_t* src = reinterpret_cast<const uint32_t*>(&entry);
    uint32_t* dst = reinterpret_cast<uint32_t*>(&entries[slot]);
    for (int i = 0; i < 18; i++) {
        __prlx_store_16b(dst + i * 4, src + i * 4);
    }
}
