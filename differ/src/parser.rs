//! Zero-copy trace file parser using memory mapping

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{cast_slice, from_bytes, try_cast_slice};
use memmap2::Mmap;
use serde::Deserialize;
use std::fs::File;
use std::path::Path;

use crate::trace_format::{
    HistoryEntry, HistoryRingHeader, SnapshotEntry, SnapshotRingHeader,
    TraceEvent, TraceFileHeader, WarpBufferHeader,
    PRLX_FLAG_COMPRESS, PRLX_FLAG_HISTORY, PRLX_MAGIC, PRLX_VERSION,
};

/// Backing storage for trace data: either memory-mapped or decompressed
enum TraceData {
    Mapped(Mmap),
    Decompressed(Vec<u8>),
}

impl TraceData {
    fn as_bytes(&self) -> &[u8] {
        match self {
            TraceData::Mapped(m) => &m[..],
            TraceData::Decompressed(v) => &v[..],
        }
    }
}

/// Trace file with transparent decompression support
pub struct TraceFile {
    backing: TraceData,
    header: TraceFileHeader,
    data_offset: usize,
    warp_buffer_size: usize,
}

impl TraceFile {
    /// Open and validate a trace file (handles compressed files transparently)
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open trace file: {}", path.display()))?;

        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap trace file: {}", path.display()))?;

        let header_size = std::mem::size_of::<TraceFileHeader>();
        if mmap.len() < header_size {
            bail!(
                "File too small: {} bytes (expected at least {})",
                mmap.len(),
                header_size
            );
        }

        let header: TraceFileHeader = *from_bytes(&mmap[..header_size]);

        if header.magic != PRLX_MAGIC {
            bail!(
                "Invalid magic number: 0x{:016x} (expected 0x{:016x})",
                header.magic,
                PRLX_MAGIC
            );
        }

        if header.version != PRLX_VERSION {
            bail!(
                "Unsupported version: {} (expected {})",
                header.version,
                PRLX_VERSION
            );
        }

        // If compressed, decompress and create a Vec-backed TraceFile
        if header.flags & PRLX_FLAG_COMPRESS != 0 {
            let compressed_payload = &mmap[header_size..];
            let decompressed = zstd::decode_all(compressed_payload)
                .context("Failed to decompress zstd-compressed trace")?;

            // Build full buffer: header + decompressed payload
            let mut full_data = Vec::with_capacity(header_size + decompressed.len());
            full_data.extend_from_slice(&mmap[..header_size]);
            full_data.extend_from_slice(&decompressed);

            Self::from_data(TraceData::Decompressed(full_data), header)
        } else {
            Self::from_data(TraceData::Mapped(mmap), header)
        }
    }

    /// Initialize from backing data with a pre-parsed header
    fn from_data(backing: TraceData, header: TraceFileHeader) -> Result<Self> {
        let data = backing.as_bytes();

        let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
            + header.events_per_warp as usize * std::mem::size_of::<TraceEvent>();
        let expected_size = header.expected_file_size();

        if data.len() < expected_size {
            bail!(
                "Data too small: {} bytes (expected {} bytes for {} warps)",
                data.len(),
                expected_size,
                header.total_warp_slots
            );
        }

        let data_offset = std::mem::size_of::<TraceFileHeader>();

        Ok(Self {
            backing,
            header,
            data_offset,
            warp_buffer_size,
        })
    }

    /// Get the file header
    pub fn header(&self) -> &TraceFileHeader {
        &self.header
    }

    /// Get the raw backing data
    fn data(&self) -> &[u8] {
        self.backing.as_bytes()
    }

    /// Get the warp buffer header and events for a specific warp (zero-copy)
    pub fn get_warp_data(&self, warp_index: usize) -> Result<(&WarpBufferHeader, &[TraceEvent])> {
        if warp_index >= self.header.total_warp_slots as usize {
            bail!(
                "Warp index {} out of range (total warps: {})",
                warp_index,
                self.header.total_warp_slots
            );
        }

        let warp_offset = self.data_offset + warp_index * self.warp_buffer_size;
        let warp_end = warp_offset + self.warp_buffer_size;

        if warp_end > self.data().len() {
            bail!("Warp buffer extends beyond file boundary");
        }

        let warp_data = &self.data()[warp_offset..warp_end];

        let header: &WarpBufferHeader =
            from_bytes(&warp_data[..std::mem::size_of::<WarpBufferHeader>()]);

        let events_data = &warp_data[std::mem::size_of::<WarpBufferHeader>()..];
        let events: &[TraceEvent] = try_cast_slice(events_data)
            .map_err(|e| anyhow!("Failed to cast events data: {}", e))?;

        // Only return the actual recorded events (not the full pre-allocated buffer)
        let num_events = header.num_events.min(self.header.events_per_warp) as usize;
        let events = &events[..num_events];

        Ok((header, events))
    }

    /// Iterator over all warps in the trace
    pub fn warps(&self) -> impl Iterator<Item = (usize, &WarpBufferHeader, &[TraceEvent])> + '_ {
        (0..self.header.total_warp_slots as usize).filter_map(move |warp_idx| {
            self.get_warp_data(warp_idx)
                .ok()
                .map(|(header, events)| (warp_idx, header, events))
        })
    }

    /// Get total number of recorded events across all warps
    pub fn total_events(&self) -> usize {
        self.warps().map(|(_, header, _)| header.num_events as usize).sum()
    }

    /// Get total number of overflow events
    pub fn total_overflows(&self) -> usize {
        self.warps()
            .map(|(_, header, _)| header.overflow_count as usize)
            .sum()
    }

    /// Check if this trace file contains history data
    pub fn has_history(&self) -> bool {
        self.header.has_history()
    }

    /// Get the history ring for a specific warp (zero-copy)
    ///
    /// Returns None if:
    /// - History flag is not set in the header
    /// - File doesn't have enough data for the history section
    pub fn get_warp_history(
        &self,
        warp_index: usize,
    ) -> Result<Option<(&HistoryRingHeader, &[HistoryEntry])>> {
        if !self.header.has_history() {
            return Ok(None);
        }

        if warp_index >= self.header.total_warp_slots as usize {
            bail!(
                "Warp index {} out of range (total warps: {})",
                warp_index,
                self.header.total_warp_slots
            );
        }

        let history_offset = self.header.history_offset();
        let ring_size = self.header.history_ring_size();
        let depth = self.header.history_depth as usize;

        let warp_ring_offset = history_offset + warp_index * ring_size;
        let warp_ring_end = warp_ring_offset + ring_size;

        // Gracefully handle truncated files (backward compat)
        if warp_ring_end > self.data().len() {
            return Ok(None);
        }

        let ring_data = &self.data()[warp_ring_offset..warp_ring_end];

        let ring_header: &HistoryRingHeader =
            from_bytes(&ring_data[..std::mem::size_of::<HistoryRingHeader>()]);

        let entries_data = &ring_data[std::mem::size_of::<HistoryRingHeader>()..];
        let entries: &[HistoryEntry] = try_cast_slice(entries_data)
            .map_err(|e| anyhow!("Failed to cast history entries: {}", e))?;

        let entries = &entries[..depth.min(entries.len())];

        Ok(Some((ring_header, entries)))
    }

    /// Check if this trace file contains snapshot data
    pub fn has_snapshot(&self) -> bool {
        self.header.has_snapshot()
    }

    /// Get the snapshot ring for a specific warp (zero-copy)
    pub fn get_warp_snapshot(
        &self,
        warp_index: usize,
    ) -> Result<Option<(&SnapshotRingHeader, &[SnapshotEntry])>> {
        if !self.header.has_snapshot() {
            return Ok(None);
        }

        if warp_index >= self.header.total_warp_slots as usize {
            bail!(
                "Warp index {} out of range (total warps: {})",
                warp_index,
                self.header.total_warp_slots
            );
        }

        let snap_offset = self.header.snapshot_offset();
        let ring_size = self.header.snapshot_ring_size();
        let depth = self.header.snapshot_depth as usize;

        let warp_ring_offset = snap_offset + warp_index * ring_size;
        let warp_ring_end = warp_ring_offset + ring_size;

        // Gracefully handle truncated files
        if warp_ring_end > self.data().len() {
            return Ok(None);
        }

        let ring_data = &self.data()[warp_ring_offset..warp_ring_end];

        let ring_header: &SnapshotRingHeader =
            from_bytes(&ring_data[..std::mem::size_of::<SnapshotRingHeader>()]);

        let entries_data = &ring_data[std::mem::size_of::<SnapshotRingHeader>()..];
        let entries: &[SnapshotEntry] = try_cast_slice(entries_data)
            .map_err(|e| anyhow!("Failed to cast snapshot entries: {}", e))?;

        let entries = &entries[..depth.min(entries.len())];

        Ok(Some((ring_header, entries)))
    }

    /// Get the most recent snapshot entry matching a specific site_id for a warp
    pub fn get_snapshot_for_site(
        &self,
        warp_index: usize,
        site_id: u32,
    ) -> Result<Option<SnapshotEntry>> {
        match self.get_warp_snapshot(warp_index)? {
            None => Ok(None),
            Some((ring_header, entries)) => {
                let depth = ring_header.depth as usize;
                let total = ring_header.total_writes as usize;

                if total == 0 || depth == 0 {
                    return Ok(None);
                }

                // Find matching entry with highest seq (most recent)
                let valid_count = total.min(depth);
                let mut best: Option<&SnapshotEntry> = None;

                for entry in entries.iter().take(valid_count) {
                    if entry.site_id == site_id {
                        match best {
                            None => best = Some(entry),
                            Some(prev) if entry.seq > prev.seq => best = Some(entry),
                            _ => {}
                        }
                    }
                }

                Ok(best.copied())
            }
        }
    }

    /// Get ordered history entries for a warp, sorted by sequence number (most recent last)
    pub fn get_ordered_history(&self, warp_index: usize) -> Result<Vec<HistoryEntry>> {
        match self.get_warp_history(warp_index)? {
            None => Ok(Vec::new()),
            Some((ring_header, entries)) => {
                let depth = ring_header.depth as usize;
                let total = ring_header.total_writes as usize;

                if total == 0 {
                    return Ok(Vec::new());
                }

                let valid_count = total.min(depth);

                let mut ordered: Vec<HistoryEntry> = if total <= depth {
                    // Ring hasn't wrapped - entries [0..total) are valid
                    entries[..valid_count].to_vec()
                } else {
                    // Ring has wrapped - all entries are valid
                    entries[..depth].to_vec()
                };

                ordered.sort_by_key(|e| e.seq);
                Ok(ordered)
            }
        }
    }
}

/// A single entry in a session manifest
#[derive(Debug, Clone, Deserialize)]
pub struct SessionLaunch {
    pub launch: u32,
    pub kernel: String,
    pub file: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// Session manifest: lists all kernel launches in a session
#[derive(Debug)]
pub struct SessionManifest {
    pub dir: std::path::PathBuf,
    pub launches: Vec<SessionLaunch>,
}

impl SessionManifest {
    /// Load a session manifest from a session directory
    pub fn load<P: AsRef<Path>>(session_dir: P) -> Result<Self> {
        let dir = session_dir.as_ref().to_path_buf();
        let manifest_path = dir.join("session.json");
        let json_str = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read session manifest: {}", manifest_path.display()))?;
        let launches: Vec<SessionLaunch> = serde_json::from_str(&json_str)
            .with_context(|| "Failed to parse session manifest")?;
        Ok(SessionManifest { dir, launches })
    }

    /// Open a trace file for a specific launch
    pub fn open_trace(&self, launch: &SessionLaunch) -> Result<TraceFile> {
        let trace_path = std::path::Path::new(&launch.file);
        // If the path is relative, resolve it against the session directory
        let full_path = if trace_path.is_absolute() {
            trace_path.to_path_buf()
        } else {
            self.dir.join(trace_path)
        };
        TraceFile::open(&full_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_struct_sizes() {
        // Verify our structs match C++ layout
        assert_eq!(std::mem::size_of::<TraceFileHeader>(), 160);
        assert_eq!(std::mem::size_of::<TraceEvent>(), 16);
        assert_eq!(std::mem::size_of::<WarpBufferHeader>(), 16);
        assert_eq!(std::mem::size_of::<HistoryRingHeader>(), 16);
        assert_eq!(std::mem::size_of::<HistoryEntry>(), 16);
        assert_eq!(std::mem::size_of::<SnapshotRingHeader>(), 16);
        assert_eq!(std::mem::size_of::<SnapshotEntry>(), 288);
    }
}
