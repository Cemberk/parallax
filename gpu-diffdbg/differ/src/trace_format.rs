//! Binary trace format definitions matching the C++ runtime structs
//! CRITICAL: These structs must match trace_format.h EXACTLY (including padding)

use bytemuck::{Pod, Zeroable};

/// Magic number for trace file format: "GDDBGGPU\0"
pub const GDDBG_MAGIC: u64 = 0x4744444247504400;

/// Current trace format version
pub const GDDBG_VERSION: u32 = 1;

/// Default number of events per warp
pub const GDDBG_EVENTS_PER_WARP: usize = 4096;

/// Header flags
pub const GDDBG_FLAG_COMPACT: u32 = 0x1;
pub const GDDBG_FLAG_COMPRESS: u32 = 0x2;
pub const GDDBG_FLAG_HISTORY: u32 = 0x4;

/// Default history ring depth (entries per warp)
pub const GDDBG_HISTORY_DEPTH_DEFAULT: usize = 64;

/// Event type constants
pub const EVENT_BRANCH: u8 = 0;
pub const EVENT_SHMEM_STORE: u8 = 1;
pub const EVENT_ATOMIC: u8 = 2;
pub const EVENT_FUNC_ENTRY: u8 = 3;
pub const EVENT_FUNC_EXIT: u8 = 4;
pub const EVENT_SWITCH: u8 = 5;

/// Trace file header (160 bytes - MUST match C++ TraceFileHeader)
///
/// Layout verified with check_layout.c:
/// - sizeof: 160 bytes
/// - alignof: 8 bytes (natural alignment for uint64_t)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TraceFileHeader {
    pub magic: u64,                 // 8 bytes, offset 0
    pub version: u32,               // 4 bytes, offset 8
    pub flags: u32,                 // 4 bytes, offset 12

    // Kernel identification
    pub kernel_name_hash: u64,      // 8 bytes, offset 16
    pub kernel_name: [u8; 64],      // 64 bytes, offset 24

    // Grid configuration
    pub grid_dim: [u32; 3],         // 12 bytes, offset 88
    pub block_dim: [u32; 3],        // 12 bytes, offset 100
    pub num_warps_per_block: u32,   // 4 bytes, offset 112
    pub total_warp_slots: u32,      // 4 bytes, offset 116
    pub events_per_warp: u32,       // 4 bytes, offset 120

    // Metadata (timestamp is uint64_t so needs 8-byte alignment)
    pub _pad: u32,                  // 4 bytes, offset 124 (padding before timestamp)
    pub timestamp: u64,             // 8 bytes, offset 128
    pub cuda_arch: u32,             // 4 bytes, offset 136

    // History / reserved fields (20 bytes to reach 160 total)
    pub history_depth: u32,         // 4 bytes, offset 140 (history entries per warp)
    pub history_section_offset: u32, // 4 bytes, offset 144 (byte offset to history, 0=auto)
    pub _reserved: [u32; 3],        // 12 bytes, offset 148
}

// Compile-time assertion that header is exactly 160 bytes
const _: () = assert!(std::mem::size_of::<TraceFileHeader>() == 160);

/// Trace event structure (16 bytes - MUST match C++ TraceEvent)
///
/// Layout verified with check_layout.c:
/// - sizeof: 16 bytes
/// - alignof: 16 bytes (explicit __attribute__((aligned(16))))
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct TraceEvent {
    pub site_id: u32,       // 4 bytes, offset 0
    pub event_type: u8,     // 1 byte, offset 4
    pub branch_dir: u8,     // 1 byte, offset 5
    pub _reserved: u16,     // 2 bytes, offset 6 (padding)
    pub active_mask: u32,   // 4 bytes, offset 8
    pub value_a: u32,       // 4 bytes, offset 12
}

// Compile-time assertion that event is exactly 16 bytes
const _: () = assert!(std::mem::size_of::<TraceEvent>() == 16);

/// Per-warp buffer header (16 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WarpBufferHeader {
    pub write_idx: u32,         // Current write position (atomic)
    pub overflow_count: u32,    // Number of dropped events
    pub num_events: u32,        // Actual events written
    pub _reserved: u32,         // Padding
}

const _: () = assert!(std::mem::size_of::<WarpBufferHeader>() == 16);

/// Per-warp history ring header (16 bytes - MUST match C++ HistoryRingHeader)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HistoryRingHeader {
    pub write_idx: u32,     // Current write position (wraps via modulo)
    pub depth: u32,         // Ring capacity
    pub total_writes: u32,  // Monotonic counter (total_writes > depth means wrapped)
    pub _reserved: u32,
}

const _: () = assert!(std::mem::size_of::<HistoryRingHeader>() == 16);

/// Single history entry (16 bytes - MUST match C++ HistoryEntry)
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct HistoryEntry {
    pub site_id: u32,       // Source location hash
    pub value: u32,         // Captured variable value
    pub seq: u32,           // Monotonic sequence number
    pub _pad: u32,          // Alignment padding
}

const _: () = assert!(std::mem::size_of::<HistoryEntry>() == 16);

impl TraceFileHeader {
    /// Get the kernel name as a Rust string (strips null padding)
    pub fn kernel_name_str(&self) -> &str {
        let null_pos = self.kernel_name.iter().position(|&b| b == 0).unwrap_or(64);
        std::str::from_utf8(&self.kernel_name[..null_pos]).unwrap_or("<invalid utf8>")
    }

    /// Calculate the expected file size in bytes (event section only)
    pub fn expected_file_size(&self) -> usize {
        let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
            + self.events_per_warp as usize * std::mem::size_of::<TraceEvent>();
        std::mem::size_of::<TraceFileHeader>()
            + self.total_warp_slots as usize * warp_buffer_size
    }

    /// Check if history data is present
    pub fn has_history(&self) -> bool {
        self.flags & GDDBG_FLAG_HISTORY != 0 && self.history_depth > 0
    }

    /// Calculate the byte offset where the history section starts
    pub fn history_offset(&self) -> usize {
        if self.history_section_offset > 0 {
            self.history_section_offset as usize
        } else {
            self.expected_file_size()
        }
    }

    /// Calculate the size of one warp's history ring (header + entries)
    pub fn history_ring_size(&self) -> usize {
        std::mem::size_of::<HistoryRingHeader>()
            + self.history_depth as usize * std::mem::size_of::<HistoryEntry>()
    }

    /// Calculate the expected total file size including history
    pub fn expected_file_size_with_history(&self) -> usize {
        let base = self.expected_file_size();
        if self.has_history() {
            base + self.total_warp_slots as usize * self.history_ring_size()
        } else {
            base
        }
    }
}

impl TraceEvent {
    /// Check if this is a branch event
    pub fn is_branch(&self) -> bool {
        self.event_type == EVENT_BRANCH
    }

    /// Get branch direction as human-readable string
    pub fn branch_direction_str(&self) -> &'static str {
        if self.branch_dir == 0 { "NOT-TAKEN" } else { "TAKEN" }
    }
}
