use crate::trace_format::*;

#[derive(Debug, Clone)]
pub enum DivergenceKind {
    BranchDirection,      // Same site, different branch taken/not-taken
    ActiveMaskMismatch,   // Same site, same direction, different threads active (SIMT divergence)
    ValueMismatch,        // Same site, same direction, same mask, different operand values
    SequenceMismatch,     // Different site_id at same position (different code path)
    LengthMismatch,       // One trace has more events than the other
}

#[derive(Debug, Clone)]
pub struct Divergence {
    pub site_id: u32,
    pub block_idx: (u32, u32, u32),
    pub warp_id: u32,
    pub event_index: usize,
    pub kind: DivergenceKind,
    pub trace_a_event: TraceEvent,
    pub trace_b_event: TraceEvent,
}

pub struct DiffOptions {
    pub max_divergences: usize,
    pub show_value_diffs: bool,
}

const LOOKAHEAD_LIMIT: usize = 10;

fn try_resync(events_a: &[TraceEvent], events_b: &[TraceEvent], limit: usize) -> Option<(usize, usize)> {
    // Try to find the next matching site_id pair within lookahead window
    // Returns (skip_a, skip_b) if a match is found
    let limit = std::cmp::min(limit, std::cmp::min(events_a.len(), events_b.len()));

    for skip_a in 0..limit {
        for skip_b in 0..limit {
            if skip_a == 0 && skip_b == 0 {
                continue;  // Already known to mismatch
            }
            if skip_a < events_a.len() && skip_b < events_b.len() {
                if events_a[skip_a].site_id == events_b[skip_b].site_id {
                    return Some((skip_a, skip_b));
                }
            }
        }
    }

    None
}

pub fn diff_traces(trace_a: &TraceFile, trace_b: &TraceFile, opts: &DiffOptions) -> Vec<Divergence> {
    let mut divergences = Vec::new();

    // Iterate over all warp slots
    for warp_idx in 0..trace_a.header.total_warp_slots {
        let warp_a = trace_a.get_warp(warp_idx);
        let warp_b = trace_b.get_warp(warp_idx);

        let block_idx = warp_a.block_idx;
        let warp_id = warp_a.warp_id;

        // Check for length mismatch
        if warp_a.events.len() != warp_b.events.len() {
            if !warp_a.events.is_empty() || !warp_b.events.is_empty() {
                divergences.push(Divergence {
                    site_id: 0,
                    block_idx,
                    warp_id,
                    event_index: std::cmp::min(warp_a.events.len(), warp_b.events.len()),
                    kind: DivergenceKind::LengthMismatch,
                    trace_a_event: if !warp_a.events.is_empty() {
                        warp_a.events[warp_a.events.len() - 1]
                    } else {
                        TraceEvent {
                            site_id: 0, event_type: 0, branch_dir: 0,
                            _reserved: 0, active_mask: 0, value_a: 0
                        }
                    },
                    trace_b_event: if !warp_b.events.is_empty() {
                        warp_b.events[warp_b.events.len() - 1]
                    } else {
                        TraceEvent {
                            site_id: 0, event_type: 0, branch_dir: 0,
                            _reserved: 0, active_mask: 0, value_a: 0
                        }
                    },
                });
            }
        }

        let min_len = std::cmp::min(warp_a.events.len(), warp_b.events.len());

        let mut i = 0;
        while i < min_len {
            let ea = &warp_a.events[i];
            let eb = &warp_b.events[i];

            // Check for site_id mismatch (different code path)
            if ea.site_id != eb.site_id {
                // Try to resync
                if let Some((skip_a, skip_b)) = try_resync(&warp_a.events[i..], &warp_b.events[i..], LOOKAHEAD_LIMIT) {
                    divergences.push(Divergence {
                        site_id: ea.site_id,
                        block_idx,
                        warp_id,
                        event_index: i,
                        kind: DivergenceKind::SequenceMismatch,
                        trace_a_event: *ea,
                        trace_b_event: *eb,
                    });
                    i += std::cmp::max(skip_a, skip_b);
                    continue;
                } else {
                    // Permanent divergence - record and stop comparing this warp
                    divergences.push(Divergence {
                        site_id: ea.site_id,
                        block_idx,
                        warp_id,
                        event_index: i,
                        kind: DivergenceKind::SequenceMismatch,
                        trace_a_event: *ea,
                        trace_b_event: *eb,
                    });
                    break;
                }
            }

            // Check for branch direction mismatch
            if ea.event_type == EVENT_BRANCH && ea.branch_dir != eb.branch_dir {
                divergences.push(Divergence {
                    site_id: ea.site_id,
                    block_idx,
                    warp_id,
                    event_index: i,
                    kind: DivergenceKind::BranchDirection,
                    trace_a_event: *ea,
                    trace_b_event: *eb,
                });
                break;  // First divergence for this warp
            }

            // CRITICAL: Check active mask mismatch (Death Valley 1)
            // A warp with mask 0xFFFFFFFF taking a branch is different from
            // a warp with mask 0xFFFF0000 taking the same branch
            if ea.active_mask != eb.active_mask {
                divergences.push(Divergence {
                    site_id: ea.site_id,
                    block_idx,
                    warp_id,
                    event_index: i,
                    kind: DivergenceKind::ActiveMaskMismatch,
                    trace_a_event: *ea,
                    trace_b_event: *eb,
                });
                // Don't break - active mask differences may propagate
            }

            // Check for value mismatch
            if opts.show_value_diffs && ea.value_a != eb.value_a {
                divergences.push(Divergence {
                    site_id: ea.site_id,
                    block_idx,
                    warp_id,
                    event_index: i,
                    kind: DivergenceKind::ValueMismatch,
                    trace_a_event: *ea,
                    trace_b_event: *eb,
                });
                // Don't break - continue to find branch divergence
            }

            i += 1;
        }

        if divergences.len() >= opts.max_divergences {
            break;
        }
    }

    divergences
}
