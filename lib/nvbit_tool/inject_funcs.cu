// Device-side injected functions for NVBit instrumentation.
//
// These functions are called by NVBit at instrumented SASS instruction sites.
// They push events through NVBit's channel mechanism to the host receiver thread.
//
// Following NVBit 1.7.7+ pattern: channel_dev and grid_launch_id are passed as
// function arguments (via nvbit_add_call_arg_const_val64 / nvbit_add_call_arg_launch_val64),
// NOT as extern __device__ globals. This avoids cross-TU device linking issues.
//
// IMPORTANT: These functions must be compiled with:
//   -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions

#include <stdint.h>
#include "utils/channel.hpp"
#include "common.h"

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

// Helper: push event to channel
__device__ __forceinline__ void prlx_push_event(
    uint64_t pchannel_dev, prlx_channel_event_t* evt) {
    ChannelDev* ch = (ChannelDev*)pchannel_dev;
    if (ch != nullptr) {
        ch->push(evt, sizeof(prlx_channel_event_t));
    }
}

// Branch instrumentation: called at BRA/BRX/JMP SASS instructions
// Args wired by host: pred, branch_taken, sass_pc, grid_launch_id, pchannel_dev
extern "C" __device__ __noinline__ void prlx_instr_branch(
    int pred,
    int branch_taken,
    uint32_t sass_pc,
    uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {
    if (!pred) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_BRANCH;
    evt.branch_dir = (uint8_t)(branch_taken & 1);
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    prlx_push_event(pchannel_dev, &evt);
}

// Shared memory store instrumentation
extern "C" __device__ __noinline__ void prlx_instr_shmem_store(
    int pred,
    uint32_t sass_pc,
    uint32_t addr,
    uint32_t value,
    uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {
    if (!pred) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_SHMEM_STORE;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = value;
    evt._reserved = 0;

    prlx_push_event(pchannel_dev, &evt);
}

// Atomic operation instrumentation
extern "C" __device__ __noinline__ void prlx_instr_atomic(
    int pred,
    uint32_t sass_pc,
    uint32_t addr,
    uint32_t operand,
    uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {
    if (!pred) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_ATOMIC;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = operand;
    evt._reserved = 0;

    prlx_push_event(pchannel_dev, &evt);
}

// Function entry instrumentation (called at first instruction of function)
extern "C" __device__ __noinline__ void prlx_instr_func_entry(
    int pred,
    uint32_t sass_pc,
    uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {
    if (!pred) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_FUNC_ENTRY;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    prlx_push_event(pchannel_dev, &evt);
}

// Function exit instrumentation (called at RET/EXIT instructions)
extern "C" __device__ __noinline__ void prlx_instr_func_exit(
    int pred,
    uint32_t sass_pc,
    uint64_t grid_launch_id,
    uint64_t pchannel_dev
) {
    if (!pred) return;
    if (prlx_lane_id() != 0) return;

    prlx_channel_event_t evt;
    evt.grid_launch_id = grid_launch_id;
    evt.warp_id = prlx_compute_warp_id();
    evt.site_id = sass_pc;
    evt.event_type = PRLX_EVENT_FUNC_EXIT;
    evt.branch_dir = 0;
    evt._pad = 0;
    evt.active_mask = __activemask();
    evt.value_a = 0;
    evt._reserved = 0;

    prlx_push_event(pchannel_dev, &evt);
}
