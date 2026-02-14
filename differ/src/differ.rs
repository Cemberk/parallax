//! Core differential analysis logic

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::parser::TraceFile;
use crate::site_map::SiteRemapper;
use crate::trace_format::TraceEvent;

/// Configuration for differential analysis
#[derive(Debug, Clone)]
pub struct DiffConfig {
    /// Compare value_a fields? (Can be noisy)
    pub compare_values: bool,
    /// Maximum number of divergences to collect (0 = unlimited)
    pub max_divergences: usize,
    /// Lookahead window size for drift detection
    pub lookahead_window: usize,
    /// Skip kernel name check (for comparing different kernel variants)
    pub force: bool,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            compare_values: false,
            max_divergences: 0, // unlimited
            lookahead_window: 32,
            force: false,
        }
    }
}

/// Type of divergence detected
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivergenceKind {
    /// Branch direction diverged (same site, different direction)
    Branch { dir_a: u8, dir_b: u8 },
    /// SIMT active mask diverged (different threads active)
    ActiveMask { mask_a: u32, mask_b: u32 },
    /// Operand value diverged
    Value { val_a: u32, val_b: u32 },
    /// True control flow divergence (different instruction reached)
    Path { site_a: u32, site_b: u32 },
    /// One trace has extra events (drift detected and recovered)
    ExtraEvents { count: usize, in_trace_b: bool },
}

/// Per-lane comparison operand snapshot context for a divergence
#[derive(Debug, Clone)]
pub struct SnapshotContext {
    pub cmp_predicate: u32,
    pub mask_a: u32,
    pub mask_b: u32,
    pub lhs_a: [u32; 32],
    pub rhs_a: [u32; 32],
    pub lhs_b: [u32; 32],
    pub rhs_b: [u32; 32],
}

/// A detected divergence between two traces
#[derive(Debug, Clone)]
pub struct Divergence {
    pub warp_idx: u32,
    pub event_idx: usize,
    pub site_id: u32,
    pub kind: DivergenceKind,
    /// Per-lane operand snapshot (present when snapshot section available)
    pub snapshot: Option<SnapshotContext>,
}

/// Result of differential analysis
pub struct DiffResult {
    pub divergences: Vec<Divergence>,
    pub total_warps: usize,
    pub total_events_a: usize,
    pub total_events_b: usize,
    pub warps_compared: usize,
    pub warps_diverged: usize,
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

    /// Get divergences grouped by kind
    pub fn divergences_by_kind(&self) -> std::collections::HashMap<String, usize> {
        let mut map = std::collections::HashMap::new();
        for div in &self.divergences {
            let kind_str = match &div.kind {
                DivergenceKind::Branch { .. } => "Branch",
                DivergenceKind::ActiveMask { .. } => "ActiveMask",
                DivergenceKind::Value { .. } => "Value",
                DivergenceKind::Path { .. } => "Path",
                DivergenceKind::ExtraEvents { .. } => "ExtraEvents",
            };
            *map.entry(kind_str.to_string()).or_insert(0) += 1;
        }
        map
    }
}

/// Compare two trace files and detect divergences.
/// If a `SiteRemapper` is provided, trace B's site_ids are translated to
/// trace A's address space before comparison, enabling cross-compilation diff.
pub fn diff_traces(
    trace_a: &TraceFile,
    trace_b: &TraceFile,
    config: &DiffConfig,
) -> Result<DiffResult> {
    diff_traces_with_remap(trace_a, trace_b, config, None)
}

/// Compare two trace files with optional site_id remapping
pub fn diff_traces_with_remap(
    trace_a: &TraceFile,
    trace_b: &TraceFile,
    config: &DiffConfig,
    remapper: Option<&SiteRemapper>,
) -> Result<DiffResult> {
    validate_traces(trace_a, trace_b, config.force)?;

    let header_a = trace_a.header();
    let header_b = trace_b.header();

    println!("Comparing {} warps...", header_a.total_warp_slots);

    let all_divergences: Vec<Vec<Divergence>> = (0..header_a.total_warp_slots as usize)
        .into_par_iter()
        .map(|warp_idx| {
            let (_, events_a) = match trace_a.get_warp_data(warp_idx) {
                Ok(data) => data,
                Err(_) => return Vec::new(),
            };
            let (_, events_b) = match trace_b.get_warp_data(warp_idx) {
                Ok(data) => data,
                Err(_) => return Vec::new(),
            };

            let remapped_b;
            let effective_b = if let Some(remap) = remapper {
                remapped_b = events_b
                    .iter()
                    .map(|e| {
                        let mut re = *e;
                        re.site_id = remap.translate(e.site_id);
                        re
                    })
                    .collect::<Vec<_>>();
                &remapped_b[..]
            } else {
                events_b
            };

            diff_single_warp(warp_idx as u32, events_a, effective_b, config)
        })
        .collect();

    let mut divergences: Vec<Divergence> = all_divergences.into_iter().flatten().collect();

    // Enrich branch/mask divergences with snapshot data (per-lane operands)
    let has_snap_a = trace_a.has_snapshot();
    let has_snap_b = trace_b.has_snapshot();
    if has_snap_a || has_snap_b {
        for div in &mut divergences {
            match &div.kind {
                DivergenceKind::Branch { .. } | DivergenceKind::ActiveMask { .. } => {
                    let snap_a = if has_snap_a {
                        trace_a
                            .get_snapshot_for_site(div.warp_idx as usize, div.site_id)
                            .unwrap_or(None)
                    } else {
                        None
                    };
                    let snap_b = if has_snap_b {
                        trace_b
                            .get_snapshot_for_site(div.warp_idx as usize, div.site_id)
                            .unwrap_or(None)
                    } else {
                        None
                    };

                    if snap_a.is_some() || snap_b.is_some() {
                        let pred = snap_a.as_ref().or(snap_b.as_ref())
                            .map(|s| s.cmp_predicate).unwrap_or(0);
                        div.snapshot = Some(SnapshotContext {
                            cmp_predicate: pred,
                            mask_a: snap_a.as_ref().map(|s| s.active_mask).unwrap_or(0),
                            mask_b: snap_b.as_ref().map(|s| s.active_mask).unwrap_or(0),
                            lhs_a: snap_a.as_ref().map(|s| s.lhs_values).unwrap_or([0; 32]),
                            rhs_a: snap_a.as_ref().map(|s| s.rhs_values).unwrap_or([0; 32]),
                            lhs_b: snap_b.as_ref().map(|s| s.lhs_values).unwrap_or([0; 32]),
                            rhs_b: snap_b.as_ref().map(|s| s.rhs_values).unwrap_or([0; 32]),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    let warps_diverged = divergences
        .iter()
        .map(|d| d.warp_idx)
        .collect::<std::collections::HashSet<_>>()
        .len();

    if config.max_divergences > 0 && divergences.len() > config.max_divergences {
        divergences.truncate(config.max_divergences);
    }

    Ok(DiffResult {
        divergences,
        total_warps: header_a.total_warp_slots as usize,
        total_events_a: trace_a.total_events(),
        total_events_b: trace_b.total_events(),
        warps_compared: header_a.total_warp_slots as usize,
        warps_diverged,
    })
}

/// Validate that two traces are compatible for comparison
fn validate_traces(trace_a: &TraceFile, trace_b: &TraceFile, force: bool) -> Result<()> {
    let header_a = trace_a.header();
    let header_b = trace_b.header();

    if header_a.kernel_name_hash != header_b.kernel_name_hash {
        if force {
            eprintln!(
                "Warning: Kernel mismatch (--force): '{}' vs '{}'",
                header_a.kernel_name_str(),
                header_b.kernel_name_str()
            );
        } else {
            bail!(
                "Kernel mismatch: '{}' (0x{:016x}) vs '{}' (0x{:016x})\n\
                 Hint: use --force to compare different kernel variants",
                header_a.kernel_name_str(),
                header_a.kernel_name_hash,
                header_b.kernel_name_str(),
                header_b.kernel_name_hash
            );
        }
    }

    if header_a.grid_dim != header_b.grid_dim {
        bail!(
            "Grid dimension mismatch: {:?} vs {:?}",
            header_a.grid_dim,
            header_b.grid_dim
        );
    }

    if header_a.block_dim != header_b.block_dim {
        bail!(
            "Block dimension mismatch: {:?} vs {:?}",
            header_a.block_dim,
            header_b.block_dim
        );
    }

    if header_a.total_warp_slots != header_b.total_warp_slots {
        bail!(
            "Total warp mismatch: {} vs {}",
            header_a.total_warp_slots,
            header_b.total_warp_slots
        );
    }

    Ok(())
}

/// Compare a single warp's event stream with bounded lookahead
///
/// This implements the "bounded lookahead" algorithm to handle drift:
/// - When site_ids match, compare masks/branches/values
/// - When site_ids differ, look ahead to detect if one trace just has extra events
/// - If no re-sync found, report true path divergence
fn diff_single_warp(
    warp_idx: u32,
    events_a: &[TraceEvent],
    events_b: &[TraceEvent],
    config: &DiffConfig,
) -> Vec<Divergence> {
    let mut divergences = Vec::new();
    let mut i_a = 0;
    let mut i_b = 0;

    while i_a < events_a.len() && i_b < events_b.len() {
        let evt_a = events_a[i_a];
        let evt_b = events_b[i_b];

        // Case 1: site_ids match - compare details
        if evt_a.site_id == evt_b.site_id {
            if evt_a.event_type == 0 && evt_a.branch_dir != evt_b.branch_dir {
                divergences.push(Divergence {
                    warp_idx,
                    event_idx: i_a,
                    site_id: evt_a.site_id,
                    kind: DivergenceKind::Branch {
                        dir_a: evt_a.branch_dir,
                        dir_b: evt_b.branch_dir,
                    },
                    snapshot: None,
                });
            }

            if evt_a.active_mask != evt_b.active_mask {
                divergences.push(Divergence {
                    warp_idx,
                    event_idx: i_a,
                    site_id: evt_a.site_id,
                    kind: DivergenceKind::ActiveMask {
                        mask_a: evt_a.active_mask,
                        mask_b: evt_b.active_mask,
                    },
                    snapshot: None,
                });
            }

            if config.compare_values && evt_a.value_a != evt_b.value_a {
                divergences.push(Divergence {
                    warp_idx,
                    event_idx: i_a,
                    site_id: evt_a.site_id,
                    kind: DivergenceKind::Value {
                        val_a: evt_a.value_a,
                        val_b: evt_b.value_a,
                    },
                    snapshot: None,
                });
            }

            i_a += 1;
            i_b += 1;
        } else {
            // Case 2: site_ids differ - try bounded lookahead to detect drift

            let mut found_in_b = None;
            for k in 1..=config.lookahead_window.min(events_b.len() - i_b) {
                if i_b + k < events_b.len() && events_b[i_b + k].site_id == evt_a.site_id {
                    found_in_b = Some(k);
                    break;
                }
            }

            let mut found_in_a = None;
            for k in 1..=config.lookahead_window.min(events_a.len() - i_a) {
                if i_a + k < events_a.len() && events_a[i_a + k].site_id == evt_b.site_id {
                    found_in_a = Some(k);
                    break;
                }
            }

            match (found_in_a, found_in_b) {
                (Some(k), None) => {
                    // Stream A has extra events
                    divergences.push(Divergence {
                        warp_idx,
                        event_idx: i_a,
                        site_id: evt_a.site_id,
                        kind: DivergenceKind::ExtraEvents {
                            count: k,
                            in_trace_b: false,
                        },
                        snapshot: None,
                    });
                    i_a += k; // Skip extra events in A
                }
                (None, Some(k)) => {
                    // Stream B has extra events
                    divergences.push(Divergence {
                        warp_idx,
                        event_idx: i_a,
                        site_id: evt_b.site_id,
                        kind: DivergenceKind::ExtraEvents {
                            count: k,
                            in_trace_b: true,
                        },
                        snapshot: None,
                    });
                    i_b += k; // Skip extra events in B
                }
                (Some(k_a), Some(k_b)) => {
                    // Both found - choose the shorter skip to minimize disruption
                    if k_a <= k_b {
                        divergences.push(Divergence {
                            warp_idx,
                            event_idx: i_a,
                            site_id: evt_a.site_id,
                            kind: DivergenceKind::ExtraEvents {
                                count: k_a,
                                in_trace_b: false,
                            },
                            snapshot: None,
                        });
                        i_a += k_a;
                    } else {
                        divergences.push(Divergence {
                            warp_idx,
                            event_idx: i_a,
                            site_id: evt_b.site_id,
                            kind: DivergenceKind::ExtraEvents {
                                count: k_b,
                                in_trace_b: true,
                            },
                            snapshot: None,
                        });
                        i_b += k_b;
                    }
                }
                (None, None) => {
                    // True path divergence - cannot re-sync
                    divergences.push(Divergence {
                        warp_idx,
                        event_idx: i_a,
                        site_id: evt_a.site_id,
                        kind: DivergenceKind::Path {
                            site_a: evt_a.site_id,
                            site_b: evt_b.site_id,
                        },
                        snapshot: None,
                    });
                    // Stop comparing this warp - paths have truly diverged
                    break;
                }
            }
        }
    }

    // Handle trailing events
    if i_a < events_a.len() {
        divergences.push(Divergence {
            warp_idx,
            event_idx: i_a,
            site_id: events_a[i_a].site_id,
            kind: DivergenceKind::ExtraEvents {
                count: events_a.len() - i_a,
                in_trace_b: false,
            },
            snapshot: None,
        });
    } else if i_b < events_b.len() {
        divergences.push(Divergence {
            warp_idx,
            event_idx: i_a,
            site_id: events_b[i_b].site_id,
            kind: DivergenceKind::ExtraEvents {
                count: events_b.len() - i_b,
                in_trace_b: true,
            },
            snapshot: None,
        });
    }

    divergences
}

/// Result of a session diff (multiple kernel launches)
pub struct SessionDiffResult {
    pub kernel_results: Vec<(String, u32, Result<DiffResult>)>,
}

/// Compare two session directories by matching kernel launches
pub fn diff_session(
    session_a: &crate::parser::SessionManifest,
    session_b: &crate::parser::SessionManifest,
    config: &DiffConfig,
) -> SessionDiffResult {
    let mut kernel_results = Vec::new();

    for launch_a in &session_a.launches {
        let matching_b = session_b.launches.iter().find(|lb| {
            lb.kernel == launch_a.kernel && lb.launch == launch_a.launch
        });

        if let Some(launch_b) = matching_b {
            let result = (|| -> Result<DiffResult> {
                let trace_a = session_a.open_trace(launch_a)?;
                let trace_b = session_b.open_trace(launch_b)?;
                diff_traces(&trace_a, &trace_b, config)
            })();
            kernel_results.push((launch_a.kernel.clone(), launch_a.launch, result));
        }
    }

    SessionDiffResult { kernel_results }
}
