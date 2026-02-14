//! Pre-computed side-by-side alignment of two trace event streams.
//!
//! Mirrors the bounded-lookahead algorithm from `differ.rs` but produces
//! paired rows suitable for rendering in a split-pane TUI.

use std::collections::HashMap;

use crate::differ::{DiffResult, Divergence, DivergenceKind};
use crate::parser::TraceFile;
use crate::site_map::SiteMap;
use crate::trace_format::{HistoryEntry, TraceEvent};

/// How a single row's events diverge (if at all).
#[derive(Debug, Clone)]
pub enum RowDivergence {
    Branch { dir_a: u8, dir_b: u8 },
    ActiveMask { mask_a: u32, mask_b: u32 },
    Value { val_a: u32, val_b: u32 },
    Path { site_a: u32, site_b: u32 },
    ExtraInA,
    ExtraInB,
}

/// Pre-formatted event data for display (avoids re-formatting per frame).
#[derive(Debug, Clone)]
pub struct EventDisplay {
    pub event_idx: usize,
    pub site_id: u32,
    pub event_type: u8,
    pub branch_dir: u8,
    pub active_mask: u32,
    pub value_a: u32,
    pub source_loc: Option<String>,
}

/// A single row in the side-by-side view.
#[derive(Debug, Clone)]
pub struct AlignedRow {
    pub row_idx: usize,
    pub event_a: Option<EventDisplay>,
    pub event_b: Option<EventDisplay>,
    pub divergences: Vec<RowDivergence>,
}

/// All aligned rows for a single warp.
pub struct WarpView {
    pub warp_idx: u32,
    pub rows: Vec<AlignedRow>,
    /// Row indices that have at least one divergence (for n/N navigation).
    pub divergence_indices: Vec<usize>,
    pub total_events_a: usize,
    pub total_events_b: usize,
}

/// Holds both traces and lazily builds WarpViews on demand.
pub struct AlignedTrace {
    pub trace_a: TraceFile,
    pub trace_b: TraceFile,
    pub site_map: Option<SiteMap>,
    pub diff_result: DiffResult,
    /// Divergences indexed by warp for fast lookup.
    divs_by_warp: HashMap<u32, Vec<Divergence>>,
    /// Lazily populated cache.
    cache: HashMap<u32, WarpView>,
}

fn event_type_str(t: u8) -> &'static str {
    match t {
        0 => "BRN",
        1 => "SHM",
        2 => "ATM",
        3 => "FEN",
        4 => "FEX",
        5 => "SWT",
        _ => "???",
    }
}

fn make_display(
    evt: &TraceEvent,
    idx: usize,
    site_map: Option<&SiteMap>,
) -> EventDisplay {
    let source_loc = site_map
        .and_then(|m| m.get(evt.site_id))
        .map(|loc| loc.format_short());

    EventDisplay {
        event_idx: idx,
        site_id: evt.site_id,
        event_type: evt.event_type,
        branch_dir: evt.branch_dir,
        active_mask: evt.active_mask,
        value_a: evt.value_a,
        source_loc,
    }
}

impl AlignedTrace {
    pub fn new(
        trace_a: TraceFile,
        trace_b: TraceFile,
        diff_result: DiffResult,
        site_map: Option<SiteMap>,
    ) -> Self {
        // Index divergences by warp_idx for O(1) lookup.
        let mut divs_by_warp: HashMap<u32, Vec<Divergence>> = HashMap::new();
        for div in &diff_result.divergences {
            divs_by_warp
                .entry(div.warp_idx)
                .or_default()
                .push(div.clone());
        }

        Self {
            trace_a,
            trace_b,
            site_map,
            diff_result,
            divs_by_warp,
            cache: HashMap::new(),
        }
    }

    pub fn num_warps(&self) -> u32 {
        self.trace_a.header().total_warp_slots
    }

    /// Check if either trace has history data
    pub fn has_history(&self) -> bool {
        self.trace_a.has_history() || self.trace_b.has_history()
    }

    /// Get ordered history entries for a warp from both traces
    pub fn get_history(&self, warp_idx: u32) -> (Vec<HistoryEntry>, Vec<HistoryEntry>) {
        let hist_a = self
            .trace_a
            .get_ordered_history(warp_idx as usize)
            .unwrap_or_default();
        let hist_b = self
            .trace_b
            .get_ordered_history(warp_idx as usize)
            .unwrap_or_default();
        (hist_a, hist_b)
    }

    /// Get (or build) the aligned view for a warp.
    pub fn warp_view(&mut self, warp_idx: u32) -> &WarpView {
        if !self.cache.contains_key(&warp_idx) {
            let view = self.build_warp_view(warp_idx);
            self.cache.insert(warp_idx, view);
        }
        self.cache.get(&warp_idx).unwrap()
    }

    fn build_warp_view(&self, warp_idx: u32) -> WarpView {
        let events_a = self
            .trace_a
            .get_warp_data(warp_idx as usize)
            .map(|(_, e)| e)
            .unwrap_or(&[]);
        let events_b = self
            .trace_b
            .get_warp_data(warp_idx as usize)
            .map(|(_, e)| e)
            .unwrap_or(&[]);

        let warp_divs = self.divs_by_warp.get(&warp_idx);
        let sm = self.site_map.as_ref();

        let mut rows = Vec::new();
        let mut divergence_indices = Vec::new();
        let mut i_a = 0usize;
        let mut i_b = 0usize;
        let lookahead = 32usize;

        while i_a < events_a.len() && i_b < events_b.len() {
            let ea = &events_a[i_a];
            let eb = &events_b[i_b];

            if ea.site_id == eb.site_id {
                // Matched pair -- check for divergences at this position.
                let mut divs = Vec::new();
                if let Some(wd) = warp_divs {
                    for d in wd {
                        if d.event_idx == i_a {
                            match &d.kind {
                                DivergenceKind::Branch { dir_a, dir_b } => {
                                    divs.push(RowDivergence::Branch {
                                        dir_a: *dir_a,
                                        dir_b: *dir_b,
                                    });
                                }
                                DivergenceKind::ActiveMask { mask_a, mask_b } => {
                                    divs.push(RowDivergence::ActiveMask {
                                        mask_a: *mask_a,
                                        mask_b: *mask_b,
                                    });
                                }
                                DivergenceKind::Value { val_a, val_b } => {
                                    divs.push(RowDivergence::Value {
                                        val_a: *val_a,
                                        val_b: *val_b,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                }

                let row_idx = rows.len();
                if !divs.is_empty() {
                    divergence_indices.push(row_idx);
                }
                rows.push(AlignedRow {
                    row_idx,
                    event_a: Some(make_display(ea, i_a, sm)),
                    event_b: Some(make_display(eb, i_b, sm)),
                    divergences: divs,
                });
                i_a += 1;
                i_b += 1;
            } else {
                // Site mismatch -- bounded lookahead to find drift.
                let mut found_in_b = None;
                for k in 1..=lookahead.min(events_b.len().saturating_sub(i_b)) {
                    if i_b + k < events_b.len() && events_b[i_b + k].site_id == ea.site_id {
                        found_in_b = Some(k);
                        break;
                    }
                }
                let mut found_in_a = None;
                for k in 1..=lookahead.min(events_a.len().saturating_sub(i_a)) {
                    if i_a + k < events_a.len() && events_a[i_a + k].site_id == eb.site_id {
                        found_in_a = Some(k);
                        break;
                    }
                }

                match (found_in_a, found_in_b) {
                    (Some(k), None) => {
                        // Extra events in A
                        for j in 0..k {
                            let row_idx = rows.len();
                            divergence_indices.push(row_idx);
                            rows.push(AlignedRow {
                                row_idx,
                                event_a: Some(make_display(&events_a[i_a + j], i_a + j, sm)),
                                event_b: None,
                                divergences: vec![RowDivergence::ExtraInA],
                            });
                        }
                        i_a += k;
                    }
                    (None, Some(k)) => {
                        // Extra events in B
                        for j in 0..k {
                            let row_idx = rows.len();
                            divergence_indices.push(row_idx);
                            rows.push(AlignedRow {
                                row_idx,
                                event_a: None,
                                event_b: Some(make_display(&events_b[i_b + j], i_b + j, sm)),
                                divergences: vec![RowDivergence::ExtraInB],
                            });
                        }
                        i_b += k;
                    }
                    (Some(k_a), Some(k_b)) => {
                        if k_a <= k_b {
                            for j in 0..k_a {
                                let row_idx = rows.len();
                                divergence_indices.push(row_idx);
                                rows.push(AlignedRow {
                                    row_idx,
                                    event_a: Some(make_display(
                                        &events_a[i_a + j],
                                        i_a + j,
                                        sm,
                                    )),
                                    event_b: None,
                                    divergences: vec![RowDivergence::ExtraInA],
                                });
                            }
                            i_a += k_a;
                        } else {
                            for j in 0..k_b {
                                let row_idx = rows.len();
                                divergence_indices.push(row_idx);
                                rows.push(AlignedRow {
                                    row_idx,
                                    event_a: None,
                                    event_b: Some(make_display(
                                        &events_b[i_b + j],
                                        i_b + j,
                                        sm,
                                    )),
                                    divergences: vec![RowDivergence::ExtraInB],
                                });
                            }
                            i_b += k_b;
                        }
                    }
                    (None, None) => {
                        // True path divergence -- emit both sides, stop.
                        let row_idx = rows.len();
                        divergence_indices.push(row_idx);
                        rows.push(AlignedRow {
                            row_idx,
                            event_a: Some(make_display(ea, i_a, sm)),
                            event_b: Some(make_display(eb, i_b, sm)),
                            divergences: vec![RowDivergence::Path {
                                site_a: ea.site_id,
                                site_b: eb.site_id,
                            }],
                        });
                        i_a = events_a.len();
                        i_b = events_b.len();
                        break;
                    }
                }
            }
        }

        // Trailing events
        while i_a < events_a.len() {
            let row_idx = rows.len();
            divergence_indices.push(row_idx);
            rows.push(AlignedRow {
                row_idx,
                event_a: Some(make_display(&events_a[i_a], i_a, sm)),
                event_b: None,
                divergences: vec![RowDivergence::ExtraInA],
            });
            i_a += 1;
        }
        while i_b < events_b.len() {
            let row_idx = rows.len();
            divergence_indices.push(row_idx);
            rows.push(AlignedRow {
                row_idx,
                event_a: None,
                event_b: Some(make_display(&events_b[i_b], i_b, sm)),
                divergences: vec![RowDivergence::ExtraInB],
            });
            i_b += 1;
        }

        WarpView {
            warp_idx,
            rows,
            divergence_indices,
            total_events_a: events_a.len(),
            total_events_b: events_b.len(),
        }
    }
}

/// Get the event type short string.
pub fn event_type_short(t: u8) -> &'static str {
    event_type_str(t)
}
