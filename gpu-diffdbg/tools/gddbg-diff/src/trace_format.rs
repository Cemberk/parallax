use anyhow::{anyhow, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

// Constants matching C header
const GDDBG_MAGIC: u64 = 0x4744444247504400;  // "GDDBGGPU\0"
const GDDBG_VERSION: u32 = 1;

// Event types
pub const EVENT_BRANCH: u8 = 0;
pub const EVENT_SHMEM_STORE: u8 = 1;
pub const EVENT_ATOMIC: u8 = 2;
pub const EVENT_FUNC_ENTRY: u8 = 3;
pub const EVENT_FUNC_EXIT: u8 = 4;
pub const EVENT_SWITCH: u8 = 5;

// Trace event structure (16 bytes, matching C struct)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceEvent {
    pub site_id: u32,
    pub event_type: u8,
    pub branch_dir: u8,
    pub _reserved: u16,
    pub active_mask: u32,
    pub value_a: u32,
}

// Warp buffer header (16 bytes, matching C struct)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WarpBufferHeader {
    pub write_idx: u32,
    pub overflow_count: u32,
    pub num_events: u32,
    pub _reserved: u32,
}

// File header (128 bytes, matching C struct)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TraceFileHeader {
    pub magic: u64,
    pub version: u32,
    pub flags: u32,
    pub kernel_name_hash: u64,
    pub kernel_name: [u8; 64],
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub num_warps_per_block: u32,
    pub total_warp_slots: u32,
    pub events_per_warp: u32,
    pub timestamp: u64,
    pub cuda_arch: u32,
    pub _reserved: [u32; 3],
}

pub struct WarpTrace<'a> {
    pub block_idx: (u32, u32, u32),
    pub warp_id: u32,
    pub header: WarpBufferHeader,
    pub events: &'a [TraceEvent],
}

pub struct TraceFile {
    _mmap: Mmap,  // Keep mmap alive
    pub header: TraceFileHeader,
    data_offset: usize,
    warp_buffer_size: usize,
}

impl TraceFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse header
        if mmap.len() < std::mem::size_of::<TraceFileHeader>() {
            return Err(anyhow!("File too small to contain header"));
        }

        let header = unsafe {
            std::ptr::read(mmap.as_ptr() as *const TraceFileHeader)
        };

        // Validate magic number
        if header.magic != GDDBG_MAGIC {
            return Err(anyhow!("Invalid magic number: expected 0x{:016X}, got 0x{:016X}",
                               GDDBG_MAGIC, header.magic));
        }

        // Validate version
        if header.version != GDDBG_VERSION {
            return Err(anyhow!("Unsupported version: {}", header.version));
        }

        let data_offset = std::mem::size_of::<TraceFileHeader>();
        let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
            + (header.events_per_warp as usize * std::mem::size_of::<TraceEvent>());

        Ok(Self {
            _mmap: mmap,
            header,
            data_offset,
            warp_buffer_size,
        })
    }

    pub fn get_warp(&self, warp_idx: u32) -> WarpTrace {
        let offset = self.data_offset + (warp_idx as usize * self.warp_buffer_size);

        // SAFETY: We trust the file format and bounds are checked
        unsafe {
            let ptr = self._mmap.as_ptr().add(offset);
            let header = std::ptr::read(ptr as *const WarpBufferHeader);

            let events_ptr = ptr.add(std::mem::size_of::<WarpBufferHeader>()) as *const TraceEvent;
            let num_events = std::cmp::min(header.num_events, self.header.events_per_warp) as usize;
            let events = std::slice::from_raw_parts(events_ptr, num_events);

            let (block_idx, warp_id) = self.warp_idx_to_coords(warp_idx);

            WarpTrace {
                block_idx,
                warp_id,
                header,
                events,
            }
        }
    }

    pub fn warp_idx_to_coords(&self, warp_idx: u32) -> ((u32, u32, u32), u32) {
        let warps_per_block = self.header.num_warps_per_block;
        let linear_block = warp_idx / warps_per_block;
        let warp_in_block = warp_idx % warps_per_block;

        let blocks_xy = self.header.grid_dim[0] * self.header.grid_dim[1];
        let block_z = linear_block / blocks_xy;
        let block_xy = linear_block % blocks_xy;
        let block_y = block_xy / self.header.grid_dim[0];
        let block_x = block_xy % self.header.grid_dim[0];

        ((block_x, block_y, block_z), warp_in_block)
    }

    pub fn get_kernel_name(&self) -> String {
        let name_bytes = &self.header.kernel_name;
        let len = name_bytes.iter().position(|&c| c == 0).unwrap_or(name_bytes.len());
        String::from_utf8_lossy(&name_bytes[..len]).to_string()
    }
}

pub fn validate_compatible(trace_a: &TraceFile, trace_b: &TraceFile) -> Result<()> {
    // Check kernel name hash
    if trace_a.header.kernel_name_hash != trace_b.header.kernel_name_hash {
        return Err(anyhow!(
            "Kernel mismatch: {} vs {}",
            trace_a.get_kernel_name(),
            trace_b.get_kernel_name()
        ));
    }

    // Check grid/block dimensions
    if trace_a.header.grid_dim != trace_b.header.grid_dim {
        return Err(anyhow!(
            "Grid dimension mismatch: ({},{},{}) vs ({},{},{})",
            trace_a.header.grid_dim[0], trace_a.header.grid_dim[1], trace_a.header.grid_dim[2],
            trace_b.header.grid_dim[0], trace_b.header.grid_dim[1], trace_b.header.grid_dim[2]
        ));
    }

    if trace_a.header.block_dim != trace_b.header.block_dim {
        return Err(anyhow!(
            "Block dimension mismatch: ({},{},{}) vs ({},{},{})",
            trace_a.header.block_dim[0], trace_a.header.block_dim[1], trace_a.header.block_dim[2],
            trace_b.header.block_dim[0], trace_b.header.block_dim[1], trace_b.header.block_dim[2]
        ));
    }

    Ok(())
}
