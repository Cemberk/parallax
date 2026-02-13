//! Core differential analysis logic

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::parser::TraceFile;
use crate::trace_format::TraceEvent;

/// Type of divergence detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceKind {
    /// Control flow diverged (different instruction executed)
    ControlFlow,
    /// Branch direction diverged (same site, different direction)
    BranchDirection,
    /// SIMT active mask diverged (different threads active)
    ActiveMask,
    /// Operand value diverged
    OperandValue,
}

/// A detected divergence between two traces
#[derive(Debug, Clone)]
pub struct Divergence {
    pub warp_index: usize,
    pub event_index: usize,
    pub kind: DivergenceKind,
    pub site_id: u32,
    pub event_a: TraceEvent,
    pub event_b: TraceEvent,
}

/// Result of differential analysis
pub struct DiffResult {
    pub divergences: Vec<Divergence>,
    pub total_warps: usize,
    pub total_events_a: usize,
    pub total_events_b: usize,
}

impl DiffResult {
    /// Check if traces are identical
    pub fn is_identical(&self) -> bool {
        self.divergences.is_empty()
    }

    /// Get divergences grouped by site_id
    pub fn divergences_by_site(&self) -> std::collections::HashMap<u32, Vec<&Divergence>> {
        let mut map = std::collections::HashMap::new();
        for div in &self.divergences {
            map.entry(div.site_id).or_insert_with(Vec::new).push(div);
        }
        map
    }
}

/// Compare two trace files and detect divergences
pub fn diff_traces(trace_a: &TraceFile, trace_b: &TraceFile) -> Result<DiffResult> {
    // Validate headers are compatible
    validate_traces(trace_a, trace_b)?;

    let header_a = trace_a.header();
    let header_b = trace_b.header();

    println!("Comparing {} warps...", header_a.total_warp_slots);

    // Parallel comparison of all warps
    let divergences: Vec<Divergence> = (0..header_a.total_warp_slots as usize)
        .into_par_iter()
        .filter_map(|warp_idx| {
            // Get events for this warp from both traces
            let (_, events_a) = trace_a.get_warp_data(warp_idx).ok()?;
            let (_, events_b) = trace_b.get_warp_data(warp_idx).ok()?;

            // Find first divergence in this warp
            find_first_divergence(warp_idx, events_a, events_b)
        })
        .collect();

    Ok(DiffResult {
        divergences,
        total_warps: header_a.total_warp_slots as usize,
        total_events_a: trace_a.total_events(),
        total_events_b: trace_b.total_events(),
    })
}

/// Validate that two traces are compatible for comparison
fn validate_traces(trace_a: &TraceFile, trace_b: &TraceFile) -> Result<()> {
    let header_a = trace_a.header();
    let header_b = trace_b.header();

    // Check kernel hash
    if header_a.kernel_name_hash != header_b.kernel_name_hash {
        bail!(
            "Kernel mismatch: '{}' (0x{:016x}) vs '{}' (0x{:016x})",
            header_a.kernel_name_str(),
            header_a.kernel_name_hash,
            header_b.kernel_name_str(),
            header_b.kernel_name_hash
        );
    }

    // Check grid dimensions
    if header_a.grid_dim != header_b.grid_dim {
        bail!(
            "Grid dimension mismatch: {:?} vs {:?}",
            header_a.grid_dim,
            header_b.grid_dim
        );
    }

    // Check block dimensions
    if header_a.block_dim != header_b.block_dim {
        bail!(
            "Block dimension mismatch: {:?} vs {:?}",
            header_a.block_dim,
            header_b.block_dim
        );
    }

    // Check total warps
    if header_a.total_warp_slots != header_b.total_warp_slots {
        bail!(
            "Total warp mismatch: {} vs {}",
            header_a.total_warp_slots,
            header_b.total_warp_slots
        );
    }

    Ok(())
}

/// Find the first divergence in a single warp's event stream
fn find_first_divergence(
    warp_index: usize,
    events_a: &[TraceEvent],
    events_b: &[TraceEvent],
) -> Option<Divergence> {
    // Lockstep comparison
    let min_len = events_a.len().min(events_b.len());

    for event_idx in 0..min_len {
        let evt_a = events_a[event_idx];
        let evt_b = events_b[event_idx];

        // Check 1: Control flow divergence (different instruction)
        if evt_a.site_id != evt_b.site_id {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::ControlFlow,
                site_id: evt_a.site_id,
                event_a: evt_a,
                event_b: evt_b,
            });
        }

        // Check 2: Branch direction divergence
        if evt_a.event_type == 0 && evt_a.branch_dir != evt_b.branch_dir {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::BranchDirection,
                site_id: evt_a.site_id,
                event_a: evt_a,
                event_b: evt_b,
            });
        }

        // Check 3: SIMT active mask divergence
        if evt_a.active_mask != evt_b.active_mask {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::ActiveMask,
                site_id: evt_a.site_id,
                event_a: evt_a,
                event_b: evt_b,
            });
        }

        // Check 4: Operand value divergence (optional, for debugging)
        if evt_a.value_a != evt_b.value_a {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::OperandValue,
                site_id: evt_a.site_id,
                event_a: evt_a,
                event_b: evt_b,
            });
        }
    }

    // Check if one trace has more events than the other
    if events_a.len() != events_b.len() {
        // One warp executed more events - this is a length divergence
        // Return a synthetic divergence at the point where they differ
        let event_idx = min_len;
        if event_idx < events_a.len() {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::ControlFlow,
                site_id: events_a[event_idx].site_id,
                event_a: events_a[event_idx],
                event_b: TraceEvent {
                    site_id: 0,
                    event_type: 0,
                    branch_dir: 0,
                    _reserved: 0,
                    active_mask: 0,
                    value_a: 0,
                },
            });
        } else {
            return Some(Divergence {
                warp_index,
                event_index: event_idx,
                kind: DivergenceKind::ControlFlow,
                site_id: events_b[event_idx].site_id,
                event_a: TraceEvent {
                    site_id: 0,
                    event_type: 0,
                    branch_dir: 0,
                    _reserved: 0,
                    active_mask: 0,
                    value_a: 0,
                },
                event_b: events_b[event_idx],
            });
        }
    }

    None
}
