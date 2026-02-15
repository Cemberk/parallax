//! Automated root-cause classification for divergences.
//!
//! Analyzes divergence patterns and heuristically classifies them into
//! one of five root-cause categories:
//! - RaceCondition — non-deterministic divergence at/near atomic/shmem sites
//! - FloatingPointPrecision — small ULP distance value divergences
//! - AlgorithmicBug — consistent path divergences across warps
//! - MemoryOrdering — active mask divergences correlated with atomic/shmem
//! - LoopBoundDifference — ExtraEvents divergences (different iteration counts)

use std::collections::HashMap;

use crate::differ::{DiffResult, Divergence, DivergenceKind};
use crate::site_map::SiteMap;
use crate::trace_format::{EVENT_ATOMIC, EVENT_SHMEM_STORE, EVENT_GLOBAL_STORE, EVENT_BRANCH};

/// Root-cause categories
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RootCause {
    RaceCondition,
    FloatingPointPrecision,
    AlgorithmicBug,
    MemoryOrdering,
    LoopBoundDifference,
}

impl RootCause {
    pub fn label(&self) -> &'static str {
        match self {
            RootCause::RaceCondition => "Race Condition",
            RootCause::FloatingPointPrecision => "Floating-Point Precision",
            RootCause::AlgorithmicBug => "Algorithmic Bug",
            RootCause::MemoryOrdering => "Memory Ordering",
            RootCause::LoopBoundDifference => "Loop Bound Difference",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            RootCause::RaceCondition =>
                "Non-deterministic divergence at or near atomic/shared-memory sites. \
                 The kernel likely has a data race.",
            RootCause::FloatingPointPrecision =>
                "Value divergences with small ULP distance, possibly at float comparison sites. \
                 Different rounding or FMA contraction may cause branch divergence.",
            RootCause::AlgorithmicBug =>
                "Consistent path divergences across warps, indicating a deterministic \
                 control flow difference (wrong algorithm, off-by-one, etc.).",
            RootCause::MemoryOrdering =>
                "Active mask divergences correlated with atomic or shared memory operations. \
                 Threads may see values in different orders.",
            RootCause::LoopBoundDifference =>
                "One trace has extra events (different iteration counts). \
                 Loop bounds or recursion depth differ between the two executions.",
        }
    }
}

/// Per-site extracted features for classification
#[derive(Debug)]
struct SiteFeatures {
    site_id: u32,
    event_type: Option<u8>,
    num_divergences: usize,
    num_warps_affected: usize,
    total_warps: usize,
    /// Distribution of divergence kinds at this site
    kind_counts: HashMap<String, usize>,
    /// Whether this site is at/near an atomic or shmem operation
    near_atomic_or_shmem: bool,
    /// Whether this site involves an fcmp predicate (from snapshots)
    has_fcmp_predicate: bool,
    /// Average ULP distance for value divergences at this site
    avg_ulp_distance: Option<f64>,
    /// Whether divergences at this site are consistent across warps
    consistent_across_warps: bool,
}

/// Classification result for a single site
#[derive(Debug, Clone)]
pub struct SiteClassification {
    pub site_id: u32,
    pub root_cause: RootCause,
    pub confidence: f64,
    pub scores: Vec<(RootCause, f64)>,
}

/// Overall classification report
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub sites: Vec<SiteClassification>,
    pub summary: HashMap<RootCause, usize>,
}

/// Compute ULP (Unit in the Last Place) distance between two IEEE 754 f32 bit patterns.
pub fn ulp_distance(a_bits: u32, b_bits: u32) -> Option<u64> {
    let fa = f32::from_bits(a_bits);
    let fb = f32::from_bits(b_bits);

    // Handle NaN/Inf
    if fa.is_nan() || fb.is_nan() || fa.is_infinite() || fb.is_infinite() {
        return None;
    }

    // Convert to signed magnitude representation for ULP computation
    fn to_signed_mag(bits: u32) -> i64 {
        if bits & 0x80000000 != 0 {
            // Negative: flip to two's complement-like representation
            -(bits as i64 & 0x7FFFFFFF)
        } else {
            bits as i64
        }
    }

    let a_int = to_signed_mag(a_bits);
    let b_int = to_signed_mag(b_bits);

    Some((a_int - b_int).unsigned_abs())
}

/// Classify divergences in a DiffResult into root-cause categories.
///
/// The algorithm is a two-pass heuristic:
/// 1. Group divergences by site_id and extract per-site features
/// 2. Score each root-cause category per site using weighted heuristics
pub fn classify(
    result: &DiffResult,
    site_map: Option<&SiteMap>,
) -> ClassificationReport {
    if result.divergences.is_empty() {
        return ClassificationReport {
            sites: Vec::new(),
            summary: HashMap::new(),
        };
    }

    // Pass 1: Extract per-site features
    let features = extract_features(result, site_map);

    // Pass 2: Score and classify each site
    let mut sites = Vec::new();
    let mut summary: HashMap<RootCause, usize> = HashMap::new();

    for feat in &features {
        let scores = score_site(feat);
        let total_score: f64 = scores.iter().map(|(_, s)| s).sum();

        if total_score > 0.0 {
            let (top_cause, top_score) = scores
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let confidence = top_score / total_score;

            *summary.entry(top_cause.clone()).or_insert(0) += 1;

            sites.push(SiteClassification {
                site_id: feat.site_id,
                root_cause: top_cause.clone(),
                confidence,
                scores: scores.clone(),
            });
        }
    }

    // Sort by confidence (highest first)
    sites.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    ClassificationReport { sites, summary }
}

/// Pass 1: Extract per-site features from divergences
fn extract_features(
    result: &DiffResult,
    site_map: Option<&SiteMap>,
) -> Vec<SiteFeatures> {
    let by_site = result.divergences_by_site();
    let all_site_ids: Vec<u32> = by_site.keys().copied().collect();

    let mut features = Vec::new();

    for site_id in &all_site_ids {
        let divs = &by_site[site_id];

        // Count divergence kinds
        let mut kind_counts: HashMap<String, usize> = HashMap::new();
        let mut ulp_distances: Vec<u64> = Vec::new();
        let mut has_fcmp = false;

        for div in divs {
            let kind_str = match &div.kind {
                DivergenceKind::Branch { .. } => "Branch",
                DivergenceKind::ActiveMask { .. } => "ActiveMask",
                DivergenceKind::Value { .. } => "Value",
                DivergenceKind::Path { .. } => "Path",
                DivergenceKind::ExtraEvents { .. } => "ExtraEvents",
            };
            *kind_counts.entry(kind_str.to_string()).or_insert(0) += 1;

            // Extract ULP distances for value divergences
            if let DivergenceKind::Value { val_a, val_b } = &div.kind {
                if let Some(d) = ulp_distance(*val_a, *val_b) {
                    ulp_distances.push(d);
                }
            }

            // Check for fcmp predicates in snapshots
            if let Some(ref snap) = div.snapshot {
                if snap.cmp_predicate <= 15 {
                    has_fcmp = true;
                }
            }
        }

        // Count unique warps affected
        let warps_affected: std::collections::HashSet<u32> =
            divs.iter().map(|d| d.warp_idx).collect();
        let num_warps = warps_affected.len();

        // Determine event type from site map
        let event_type = site_map
            .and_then(|m| m.get(*site_id))
            .map(|loc| loc.event_type);

        // Check neighborhood: are there atomic/shmem sites nearby in the divergence list?
        let near_atomic_or_shmem = is_near_atomic_shmem(*site_id, &all_site_ids, site_map);

        // Average ULP distance
        let avg_ulp = if ulp_distances.is_empty() {
            None
        } else {
            Some(ulp_distances.iter().sum::<u64>() as f64 / ulp_distances.len() as f64)
        };

        // Consistency check: are divergences of the same kind across all warps?
        let consistent = if num_warps >= 2 {
            let dominant_kind = kind_counts.iter().max_by_key(|(_, c)| *c);
            if let Some((_, count)) = dominant_kind {
                *count as f64 / divs.len() as f64 > 0.8
            } else {
                false
            }
        } else {
            false
        };

        features.push(SiteFeatures {
            site_id: *site_id,
            event_type,
            num_divergences: divs.len(),
            num_warps_affected: num_warps,
            total_warps: result.total_warps,
            kind_counts,
            near_atomic_or_shmem,
            has_fcmp_predicate: has_fcmp,
            avg_ulp_distance: avg_ulp,
            consistent_across_warps: consistent,
        });
    }

    features
}

/// Neighborhood analysis: check if a site is near atomic/shmem sites
fn is_near_atomic_shmem(
    site_id: u32,
    all_sites: &[u32],
    site_map: Option<&SiteMap>,
) -> bool {
    let site_map = match site_map {
        Some(m) => m,
        None => return false,
    };

    // Check the site itself
    if let Some(loc) = site_map.get(site_id) {
        let et = loc.event_type;
        if et == EVENT_ATOMIC || et == EVENT_SHMEM_STORE || et == EVENT_GLOBAL_STORE {
            return true;
        }
    }

    // Check nearby sites (within ±3 positions in the site list)
    let idx = all_sites.iter().position(|&s| s == site_id);
    if let Some(pos) = idx {
        let start = pos.saturating_sub(3);
        let end = (pos + 4).min(all_sites.len());
        for i in start..end {
            if i == pos { continue; }
            if let Some(loc) = site_map.get(all_sites[i]) {
                let et = loc.event_type;
                if et == EVENT_ATOMIC || et == EVENT_SHMEM_STORE || et == EVENT_GLOBAL_STORE {
                    return true;
                }
            }
        }
    }

    false
}

/// Pass 2: Score each root-cause category for a site
fn score_site(feat: &SiteFeatures) -> Vec<(RootCause, f64)> {
    vec![
        (RootCause::RaceCondition, score_race(feat)),
        (RootCause::FloatingPointPrecision, score_fp_precision(feat)),
        (RootCause::AlgorithmicBug, score_algorithmic(feat)),
        (RootCause::MemoryOrdering, score_memory_ordering(feat)),
        (RootCause::LoopBoundDifference, score_loop_bound(feat)),
    ]
}

fn score_race(feat: &SiteFeatures) -> f64 {
    let mut score = 0.0;

    // At or near atomic site
    if feat.near_atomic_or_shmem {
        score += 3.0;
    }
    if let Some(et) = feat.event_type {
        if et == EVENT_ATOMIC || et == EVENT_SHMEM_STORE {
            score += 3.0;
        }
    }

    // Both branch and value divergences combined
    let has_branch = feat.kind_counts.contains_key("Branch");
    let has_value = feat.kind_counts.contains_key("Value");
    if has_branch && has_value {
        score += 1.5;
    }

    // Low warp coverage (not all warps affected → non-deterministic)
    if feat.total_warps > 0 {
        let coverage = feat.num_warps_affected as f64 / feat.total_warps as f64;
        if coverage < 0.5 {
            score += 1.0;
        }
    }

    // Inconsistent patterns across warps
    if !feat.consistent_across_warps && feat.num_warps_affected >= 2 {
        score += 1.5;
    }

    score
}

fn score_fp_precision(feat: &SiteFeatures) -> f64 {
    let mut score = 0.0;

    // Float comparison predicate
    if feat.has_fcmp_predicate {
        score += 3.0;
    }

    // Small ULP distance
    if let Some(avg_ulp) = feat.avg_ulp_distance {
        if avg_ulp < 100.0 {
            score += 2.5;
        } else if avg_ulp < 10000.0 {
            score += 1.0;
        }
    }

    // Purely value divergences (no path/branch)
    let has_value = *feat.kind_counts.get("Value").unwrap_or(&0) > 0;
    let has_branch = *feat.kind_counts.get("Branch").unwrap_or(&0) > 0;
    let has_path = *feat.kind_counts.get("Path").unwrap_or(&0) > 0;
    if has_value && !has_path {
        score += 1.0;
    }
    // Branch divergence at fcmp site is classic FP precision issue
    if has_branch && feat.has_fcmp_predicate {
        score += 1.5;
    }

    score
}

fn score_algorithmic(feat: &SiteFeatures) -> f64 {
    let mut score = 0.0;

    // Path divergences
    let path_count = *feat.kind_counts.get("Path").unwrap_or(&0);
    if path_count > 0 {
        score += 3.0;
    }

    // Consistent across warps
    if feat.consistent_across_warps {
        score += 2.0;
    }

    // High warp coverage (deterministic bug affects most warps)
    if feat.total_warps > 0 {
        let coverage = feat.num_warps_affected as f64 / feat.total_warps as f64;
        if coverage > 0.7 {
            score += 1.5;
        }
    }

    // Branch divergences at branch sites
    let has_branch = *feat.kind_counts.get("Branch").unwrap_or(&0) > 0;
    if has_branch && feat.consistent_across_warps {
        score += 1.0;
    }

    score
}

fn score_memory_ordering(feat: &SiteFeatures) -> f64 {
    let mut score = 0.0;

    let has_mask = *feat.kind_counts.get("ActiveMask").unwrap_or(&0) > 0;

    // Active mask near shmem
    if has_mask && feat.near_atomic_or_shmem {
        score += 2.5;
    }

    // Active mask near atomic
    if let Some(et) = feat.event_type {
        if has_mask && (et == EVENT_ATOMIC || et == EVENT_SHMEM_STORE) {
            score += 2.5;
        }
    }

    // Pure mask divergence (no branch or path)
    let has_branch = *feat.kind_counts.get("Branch").unwrap_or(&0) > 0;
    let has_path = *feat.kind_counts.get("Path").unwrap_or(&0) > 0;
    if has_mask && !has_branch && !has_path {
        score += 1.5;
    }

    score
}

fn score_loop_bound(feat: &SiteFeatures) -> f64 {
    let mut score = 0.0;

    // ExtraEvents is the primary signal
    let extra_count = *feat.kind_counts.get("ExtraEvents").unwrap_or(&0);
    if extra_count > 0 {
        score += 3.5;
    }

    // Consistent pattern (same extra event pattern across warps)
    if extra_count > 0 && feat.consistent_across_warps {
        score += 1.5;
    }

    // At a branch site
    if let Some(et) = feat.event_type {
        if extra_count > 0 && et == EVENT_BRANCH {
            score += 1.0;
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ulp_distance_zero() {
        assert_eq!(ulp_distance(0, 0), Some(0));
    }

    #[test]
    fn test_ulp_distance_one_ulp() {
        let a = 1.0f32.to_bits();
        let b = (1.0f32 + f32::EPSILON).to_bits();
        assert_eq!(ulp_distance(a, b), Some(1));
    }

    #[test]
    fn test_ulp_distance_nan() {
        let nan_bits = f32::NAN.to_bits();
        assert_eq!(ulp_distance(nan_bits, 0), None);
    }

    #[test]
    fn test_ulp_distance_inf() {
        let inf_bits = f32::INFINITY.to_bits();
        assert_eq!(ulp_distance(inf_bits, 0), None);
    }

    #[test]
    fn test_ulp_distance_symmetric() {
        let a = 1.0f32.to_bits();
        let b = 2.0f32.to_bits();
        assert_eq!(ulp_distance(a, b), ulp_distance(b, a));
    }

    #[test]
    fn test_classify_empty() {
        let result = DiffResult {
            divergences: Vec::new(),
            total_warps: 0,
            total_events_a: 0,
            total_events_b: 0,
            warps_compared: 0,
            warps_diverged: 0,
            cross_gpu_info: None,
        };
        let report = classify(&result, None);
        assert!(report.sites.is_empty());
    }

    #[test]
    fn test_classify_extra_events() {
        let result = DiffResult {
            divergences: vec![
                Divergence {
                    warp_idx: 0,
                    event_idx: 10,
                    site_id: 0x100,
                    kind: DivergenceKind::ExtraEvents { count: 5, in_trace_b: true },
                    snapshot: None,
                },
                Divergence {
                    warp_idx: 1,
                    event_idx: 10,
                    site_id: 0x100,
                    kind: DivergenceKind::ExtraEvents { count: 5, in_trace_b: true },
                    snapshot: None,
                },
            ],
            total_warps: 4,
            total_events_a: 100,
            total_events_b: 110,
            warps_compared: 4,
            warps_diverged: 2,
            cross_gpu_info: None,
        };
        let report = classify(&result, None);
        assert!(!report.sites.is_empty());
        assert_eq!(report.sites[0].root_cause, RootCause::LoopBoundDifference);
    }

    #[test]
    fn test_classify_path_divergence() {
        let result = DiffResult {
            divergences: vec![
                Divergence {
                    warp_idx: 0,
                    event_idx: 5,
                    site_id: 0x200,
                    kind: DivergenceKind::Path { site_a: 0x200, site_b: 0x300 },
                    snapshot: None,
                },
                Divergence {
                    warp_idx: 1,
                    event_idx: 5,
                    site_id: 0x200,
                    kind: DivergenceKind::Path { site_a: 0x200, site_b: 0x300 },
                    snapshot: None,
                },
                Divergence {
                    warp_idx: 2,
                    event_idx: 5,
                    site_id: 0x200,
                    kind: DivergenceKind::Path { site_a: 0x200, site_b: 0x300 },
                    snapshot: None,
                },
            ],
            total_warps: 4,
            total_events_a: 100,
            total_events_b: 100,
            warps_compared: 4,
            warps_diverged: 3,
            cross_gpu_info: None,
        };
        let report = classify(&result, None);
        assert!(!report.sites.is_empty());
        assert_eq!(report.sites[0].root_cause, RootCause::AlgorithmicBug);
    }
}
