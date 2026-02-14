// Device-side injected functions for NVBit instrumentation.
//
// These functions are called by NVBit at instrumented SASS instruction sites.
// They push events through NVBit's channel mechanism to the host receiver thread.
//
// IMPORTANT: These functions must be compiled with:
//   -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions

#include <stdint.h>
#include "utils/channel.hpp"
#include "common.h"

// Channel device-side handle (set by host via cudaMemcpyToSymbol)
extern "C" __device__ __noinline__ ChannelDev* prlx_channel_dev;

// Current grid launch ID (set by host before each kernel launch)
extern "C" __device__ __noinline__ uint64_t prlx_grid_launch_id;

// Guard: set to 0 to disable instrumentation
extern "C" __device__ __noinline__ int prlx_enabled;

// Compute linear warp ID within the grid
__device__ __forceinline__ uint32_t prlx_compute_warp_id() {
    uint32_t tid = threadIdx.x + threadIdx.y * blockDim.x
                 + threadIdx.z * blockDim.x * blockDim.y;
    uint32_t warp_in_block = tid / 32;
    uint32_t linear_block = blockIdx.x
                          + blockIdx.y * gridDim.x
                          + blockIdx.z * gridDim.x * gridDim.y;
    uint32_t warps_per_block = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;
    return linear_block * warps_per_block + warp_in_block;
}

// Get lane ID within warp
__device__ __forceinline__ uint32_t prlx_lane_id() {
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

// Branch instrumentation: called at BRA/BRX/JMP SASS instructions
extern "C" __device__ __noinline__ void prlx_instr_branch(
    int pred,           // Guard predicate (from NVBit)
    int branch_taken,   // Branch direction
    uint32_t sass_pc,   // SASS PC offset (used as site_id)
    uint64_t grid_id    // Grid launch ID (redundant, for validation)
) {
    if (!pred) return;
    if (!prlx_enabled) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = prlx_grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_BRANCH;
    evt.branch_dir = (uint8_t)(branch_taken & 1);
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    ChannelDev* ch = prlx_channel_dev;
    if (ch != nullptr) {
        ch->push(&evt, sizeof(evt));
    }
}

// Shared memory store instrumentation
extern "C" __device__ __noinline__ void prlx_instr_shmem_store(
    int pred,
    uint32_t sass_pc,
    uint32_t addr,
    uint32_t value
) {
    if (!pred) return;
    if (!prlx_enabled) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = prlx_grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_SHMEM_STORE;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = value;
    evt._reserved = 0;

    ChannelDev* ch = prlx_channel_dev;
    if (ch != nullptr) {
        ch->push(&evt, sizeof(evt));
    }
}

// Atomic operation instrumentation
extern "C" __device__ __noinline__ void prlx_instr_atomic(
    int pred,
    uint32_t sass_pc,
    uint32_t addr,
    uint32_t operand
) {
    if (!pred) return;
    if (!prlx_enabled) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = prlx_grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_ATOMIC;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = operand;
    evt._reserved = 0;

    ChannelDev* ch = prlx_channel_dev;
    if (ch != nullptr) {
        ch->push(&evt, sizeof(evt));
    }
}

// Function entry instrumentation (called at first instruction of function)
extern "C" __device__ __noinline__ void prlx_instr_func_entry(
    int pred,
    uint32_t sass_pc
) {
    if (!pred) return;
    if (!prlx_enabled) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = prlx_grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_FUNC_ENTRY;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    ChannelDev* ch = prlx_channel_dev;
    if (ch != nullptr) {
        ch->push(&evt, sizeof(evt));
    }
}

// Function exit instrumentation (called at RET/EXIT instructions)
extern "C" __device__ __noinline__ void prlx_instr_func_exit(
    int pred,
    uint32_t sass_pc
) {
    if (!pred) return;
    if (!prlx_enabled) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = prlx_grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_FUNC_EXIT;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    ChannelDev* ch = prlx_channel_dev;
    if (ch != nullptr) {
        ch->push(&evt, sizeof(evt));
    }
}
