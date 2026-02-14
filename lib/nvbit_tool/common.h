#ifndef PRLX_NVBIT_COMMON_H
#define PRLX_NVBIT_COMMON_H

#include <stdint.h>

// Channel event structure: sent from device-side injected functions
// to the host-side receiver thread via NVBit's ChannelDev/ChannelHost.
// Padded to 32 bytes for efficient channel transfer.
typedef struct {
    uint64_t grid_launch_id;  // 8B - identifies which kernel launch
    uint32_t warp_id;         // 4B - linear warp ID within the grid
    uint32_t site_id;         // 4B - SASS PC offset (host converts to hash)
    uint8_t  event_type;      // 1B - EVENT_BRANCH=0, etc.
    uint8_t  branch_dir;      // 1B - branch direction (0=not-taken, 1=taken)
    uint16_t _pad;            // 2B - padding
    uint32_t active_mask;     // 4B - full 32-bit __activemask()
    uint32_t value_a;         // 4B - primary operand value
    uint32_t _reserved;       // 4B - pad to 32 bytes
} prlx_channel_event_t;

// Compile-time size check (C11 _Static_assert)
#ifdef __cplusplus
static_assert(sizeof(prlx_channel_event_t) == 32, "channel event must be 32 bytes");
#else
_Static_assert(sizeof(prlx_channel_event_t) == 32, "channel event must be 32 bytes");
#endif

// Event types (must match trace_format.h)
#define PRLX_EVENT_BRANCH      0
#define PRLX_EVENT_SHMEM_STORE 1
#define PRLX_EVENT_ATOMIC      2
#define PRLX_EVENT_FUNC_ENTRY  3
#define PRLX_EVENT_FUNC_EXIT   4

// Max events per warp (matches default PRLX_EVENTS_PER_WARP from trace_format.h)
#define PRLX_NVBIT_DEFAULT_BUFFER_SIZE 4096

#endif // PRLX_NVBIT_COMMON_H
