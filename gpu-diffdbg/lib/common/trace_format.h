#ifndef GDDBG_TRACE_FORMAT_H
#define GDDBG_TRACE_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Magic number for trace file format
#define GDDBG_MAGIC 0x4744444247504400ULL  // "GDDBGGPU\0"

// Trace format version
#define GDDBG_VERSION 1

// Default configuration
#define GDDBG_EVENTS_PER_WARP 4096

// Event types
#define EVENT_BRANCH      0
#define EVENT_SHMEM_STORE 1
#define EVENT_ATOMIC      2
#define EVENT_FUNC_ENTRY  3
#define EVENT_FUNC_EXIT   4
#define EVENT_SWITCH      5

// Trace event structure (16 bytes - full version)
// V0: Full-size event used during development and detailed debugging
// MUST be 16-byte aligned for v4.u32 PTX stores
typedef struct __attribute__((aligned(16))) {
    uint32_t site_id;       // Deterministic hash of (file:func:line:col), NOT sequential
    uint8_t  event_type;    // BRANCH=0, SHMEM_STORE=1, ATOMIC=2, etc.
    uint8_t  branch_dir;    // For branches: 0=not-taken, 1=taken
    uint16_t _reserved;     // Padding / future use
    uint32_t active_mask;   // FULL 32-bit __activemask() - CRITICAL for SIMT divergence
    uint32_t value_a;       // Primary value (branch condition operand / store value / etc)
} TraceEvent;

// Compact event structure (8 bytes - future optimization)
typedef struct {
    uint16_t site_id;       // 16-bit hash (65K sites per kernel)
    uint8_t  event_type : 4;
    uint8_t  branch_dir : 1;
    uint8_t  _pad : 3;
    uint8_t  _reserved;
    uint32_t active_mask;   // Still full 32-bit - never compressed
} TraceEventCompact;

// Per-warp buffer header
typedef struct {
    uint32_t write_idx;         // Current write position (atomic)
    uint32_t overflow_count;    // Number of dropped events
    uint32_t num_events;        // Actual events written (set during copy-back)
    uint32_t _reserved;
} WarpBufferHeader;

// File header (160 bytes - must be multiple of 16 for alignment)
typedef struct {
    uint64_t magic;             // GDDBG_MAGIC
    uint32_t version;           // GDDBG_VERSION
    uint32_t flags;             // Bit 0: compact format, Bit 1: compressed

    // Kernel identification
    uint64_t kernel_name_hash;
    char     kernel_name[64];

    // Grid configuration
    uint32_t grid_dim[3];
    uint32_t block_dim[3];
    uint32_t num_warps_per_block;
    uint32_t total_warp_slots;
    uint32_t events_per_warp;

    // Metadata
    uint64_t timestamp;
    uint32_t cuda_arch;         // e.g., 80 for SM_80
    uint32_t _reserved[5];      // Padding to make 160 bytes (multiple of 16)
} TraceFileHeader;

// Site table entry
typedef struct {
    uint32_t site_id;
    uint8_t  event_type;
    uint8_t  _reserved[3];
    uint32_t filename_offset;   // Offset into string table
    uint32_t function_name_offset;
    uint32_t line_number;
    uint32_t column_number;
} SiteTableEntry;

#ifdef __cplusplus
}
#endif

#endif // GDDBG_TRACE_FORMAT_H
