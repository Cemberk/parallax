#ifndef PRLX_TRACE_FORMAT_H
#define PRLX_TRACE_FORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Magic number for trace file format
#define PRLX_MAGIC 0x50524C5800000000ULL  // "PRLX\0\0\0\0"

// Trace format version
#define PRLX_VERSION 1

// Default configuration
#define PRLX_EVENTS_PER_WARP 4096

// Header flags
#define PRLX_FLAG_COMPACT  0x1   // Bit 0: compact event format
#define PRLX_FLAG_COMPRESS 0x2   // Bit 1: zstd compressed
#define PRLX_FLAG_HISTORY  0x4   // Bit 2: history ring section appended
#define PRLX_FLAG_SAMPLED  0x8   // Bit 3: sampling was active (sample_rate > 1)
#define PRLX_FLAG_SNAPSHOT 0x10  // Bit 4: snapshot section appended (per-lane operands)

// History (time-travel) defaults
#define PRLX_HISTORY_DEPTH_DEFAULT 64

// Snapshot (crash dump) defaults
#define PRLX_SNAPSHOT_DEPTH_DEFAULT 32

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
    uint32_t total_event_count; // Total events seen (including sampled-out), for sampling diagnostics
} WarpBufferHeader;

// File header (160 bytes - must be multiple of 16 for alignment)
typedef struct {
    uint64_t magic;             // PRLX_MAGIC
    uint32_t version;           // PRLX_VERSION
    uint32_t flags;             // PRLX_FLAG_* bitmask

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

    // History / sampling / reserved fields (20 bytes to reach 160 total)
    // When PRLX_FLAG_HISTORY is set:
    //   history_depth: entries per warp in the history ring buffer
    //   history_section_offset: byte offset from file start to history data
    //     (0 = immediately after warp event buffers)
    uint32_t history_depth;             // [0] History entries per warp (0 = no history)
    uint32_t history_section_offset;    // [1] Byte offset to history section (0 = auto)
    uint32_t sample_rate;              // [2] Sampling rate (1 = record all, N = record 1/N)
    uint32_t snapshot_depth;           // [3] Snapshot entries per warp (0 = no snapshots)
    uint32_t snapshot_section_offset;  // [4] Byte offset to snapshot section (0 = auto)
} TraceFileHeader;

// ---- History (Time-Travel) Structures ----
// Appended after all warp event buffers when PRLX_FLAG_HISTORY is set.
// Layout per warp: [HistoryRingHeader][HistoryEntry * depth]

// Per-warp history ring header (16 bytes)
typedef struct {
    uint32_t write_idx;     // Current write position (wraps via modulo)
    uint32_t depth;         // Ring capacity (same as TraceFileHeader.history_depth)
    uint32_t total_writes;  // Monotonic counter (detects wrap: total_writes > depth)
    uint32_t _reserved;
} HistoryRingHeader;

// Single history entry (16 bytes, aligned for v4.u32 stores)
typedef struct __attribute__((aligned(16))) {
    uint32_t site_id;       // Source location hash
    uint32_t value;         // Captured variable value
    uint32_t seq;           // Monotonic sequence number (for chronological ordering)
    uint32_t _pad;          // Alignment padding
} HistoryEntry;

// ---- Snapshot (Crash Dump) Structures ----
// Appended after history section (or warp event buffers if no history)
// when PRLX_FLAG_SNAPSHOT is set.
// Layout per warp: [SnapshotRingHeader][SnapshotEntry * depth]
// Captures per-lane comparison operands at branch divergence sites.

// Per-warp snapshot ring header (16 bytes)
typedef struct {
    uint32_t write_idx;     // Current write position (wraps via modulo)
    uint32_t depth;         // Ring capacity (same as TraceFileHeader.snapshot_depth)
    uint32_t total_writes;  // Monotonic counter (detects wrap)
    uint32_t _reserved;
} SnapshotRingHeader;

// Per-lane comparison operand snapshot (288 bytes = 18 * 16, aligned for v4.u32)
// Captures both operands of ICmpInst/FCmpInst for all 32 lanes via __shfl_sync
typedef struct __attribute__((aligned(16))) {
    uint32_t site_id;           // 4B - links back to the branch event
    uint32_t active_mask;       // 4B - which lanes were active
    uint32_t seq;               // 4B - monotonic sequence for ordering
    uint32_t cmp_predicate;     // 4B - ICmp/FCmp predicate enum value
    uint32_t lhs_values[32];    // 128B - per-lane LHS operand
    uint32_t rhs_values[32];    // 128B - per-lane RHS operand
    uint32_t _pad[4];           // 16B - pad to 288 = 18 * 16
} SnapshotEntry;

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

#endif // PRLX_TRACE_FORMAT_H
