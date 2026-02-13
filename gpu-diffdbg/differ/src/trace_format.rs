//! Binary trace format definitions matching the C++ runtime structs
//! CRITICAL: These structs must match trace_format.h EXACTLY (including padding)

use bytemuck::{Pod, Zeroable};

/// Magic number for trace file format: "GDDBGGPU\0"
pub const GDDBG_MAGIC: u64 = 0x4744444247504400;

/// Current trace format version
pub const GDDBG_VERSION: u32 = 1;

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
    pub _reserved: [u32; 5],        // 20 bytes, offset 140
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

impl TraceFileHeader {
    /// Get the kernel name as a Rust string (strips null padding)
    pub fn kernel_name_str(&self) -> &str {
        let null_pos = self.kernel_name.iter().position(|&b| b == 0).unwrap_or(64);
        std::str::from_utf8(&self.kernel_name[..null_pos]).unwrap_or("<invalid utf8>")
    }

    /// Calculate the expected file size in bytes
    pub fn expected_file_size(&self) -> usize {
        let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
            + self.events_per_warp as usize * std::mem::size_of::<TraceEvent>();
        std::mem::size_of::<TraceFileHeader>()
            + self.total_warp_slots as usize * warp_buffer_size
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
