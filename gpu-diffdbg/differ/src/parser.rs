//! Zero-copy trace file parser using memory mapping

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::{cast_slice, from_bytes, try_cast_slice};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use crate::trace_format::{
    HistoryEntry, HistoryRingHeader, TraceEvent, TraceFileHeader, WarpBufferHeader,
    GDDBG_FLAG_HISTORY, GDDBG_MAGIC, GDDBG_VERSION,
};

/// Memory-mapped trace file with zero-copy access to events
pub struct TraceFile {
    _mmap: Mmap,
    header: TraceFileHeader,
    data_offset: usize,
    warp_buffer_size: usize,
}

impl TraceFile {
    /// Open and validate a trace file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open trace file: {}", path.display()))?;

        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap trace file: {}", path.display()))?;

        Self::from_mmap(mmap)
    }

    /// Parse a memory-mapped trace file
    fn from_mmap(mmap: Mmap) -> Result<Self> {
        // Check minimum file size
        if mmap.len() < std::mem::size_of::<TraceFileHeader>() {
            bail!(
                "File too small: {} bytes (expected at least {})",
                mmap.len(),
                std::mem::size_of::<TraceFileHeader>()
            );
        }

        // Parse header (zero-copy)
        let header: TraceFileHeader = *from_bytes(&mmap[..std::mem::size_of::<TraceFileHeader>()]);

        // Validate magic number
        if header.magic != GDDBG_MAGIC {
            bail!(
                "Invalid magic number: 0x{:016x} (expected 0x{:016x})",
                header.magic,
                GDDBG_MAGIC
            );
        }

        // Validate version
        if header.version != GDDBG_VERSION {
            bail!(
                "Unsupported version: {} (expected {})",
                header.version,
                GDDBG_VERSION
            );
        }

        // Calculate buffer layout
        let warp_buffer_size = std::mem::size_of::<WarpBufferHeader>()
            + header.events_per_warp as usize * std::mem::size_of::<TraceEvent>();
        let expected_size = header.expected_file_size();

        // Validate file size
        if mmap.len() < expected_size {
            bail!(
                "File too small: {} bytes (expected {} bytes for {} warps)",
                mmap.len(),
                expected_size,
                header.total_warp_slots
            );
        }

        let data_offset = std::mem::size_of::<TraceFileHeader>();

        Ok(Self {
            _mmap: mmap,
            header,
            data_offset,
            warp_buffer_size,
        })
    }

    /// Get the file header
    pub fn header(&self) -> &TraceFileHeader {
        &self.header
    }

    /// Get the raw memory-mapped data
    fn data(&self) -> &[u8] {
        &self._mmap
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

        // Parse header
        let header: &WarpBufferHeader =
            from_bytes(&warp_data[..std::mem::size_of::<WarpBufferHeader>()]);

        // Parse events (zero-copy slice cast)
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

        // Parse ring header
        let ring_header: &HistoryRingHeader =
            from_bytes(&ring_data[..std::mem::size_of::<HistoryRingHeader>()]);

        // Parse entries
        let entries_data = &ring_data[std::mem::size_of::<HistoryRingHeader>()..];
        let entries: &[HistoryEntry] = try_cast_slice(entries_data)
            .map_err(|e| anyhow!("Failed to cast history entries: {}", e))?;

        let entries = &entries[..depth.min(entries.len())];

        Ok(Some((ring_header, entries)))
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

                // How many valid entries?
                let valid_count = total.min(depth);

                // Collect valid entries and sort by seq
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
    }
}
