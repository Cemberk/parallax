//! JSON output for CI integration (prlx assert)

use serde::Serialize;
use std::collections::HashMap;

use crate::classifier::ClassificationReport;
use crate::differ::{DiffResult, DivergenceKind, SessionDiffResult};
use crate::site_map::SiteMap;

/// Cross-GPU metadata in JSON output
#[derive(Serialize)]
pub struct JsonCrossGpuInfo {
    pub arch_a: String,
    pub arch_b: String,
    pub warp_size_a: u32,
    pub warp_size_b: u32,
    pub grid_a: [u32; 3],
    pub grid_b: [u32; 3],
    pub block_a: [u32; 3],
    pub block_b: [u32; 3],
}

/// Machine-readable diff report for CI pipelines
#[derive(Serialize)]
pub struct JsonDiffReport {
    pub status: String,
    pub total_divergences: usize,
    pub counted_divergences: usize,
    pub warps_compared: usize,
    pub warps_diverged: usize,
    pub total_events_a: usize,
    pub total_events_b: usize,
    pub divergence_breakdown: HashMap<String, usize>,
    pub divergences: Vec<JsonDivergence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<usize>,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_gpu: Option<JsonCrossGpuInfo>,
}

/// Individual divergence in JSON output
#[derive(Serialize)]
pub struct JsonDivergence {
    pub warp_idx: u32,
    pub event_idx: usize,
    pub site_id: String,
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_location: Option<String>,
}

/// Machine-readable session diff report
#[derive(Serialize)]
pub struct JsonSessionReport {
    pub status: String,
    pub kernel_results: Vec<JsonKernelResult>,
    pub unmatched_a: Vec<String>,
    pub unmatched_b: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<usize>,
    pub passed: bool,
}

/// Per-kernel result in session report
#[derive(Serialize)]
pub struct JsonKernelResult {
    pub kernel: String,
    pub launch: u32,
    pub status: String,
    pub divergences: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Machine-readable classification report
#[derive(Serialize)]
pub struct JsonClassificationReport {
    pub sites: Vec<JsonSiteClassification>,
    pub summary: HashMap<String, usize>,
}

/// Per-site classification in JSON output
#[derive(Serialize)]
pub struct JsonSiteClassification {
    pub site_id: String,
    pub root_cause: String,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_location: Option<String>,
}

/// Format a ClassificationReport as JSON
pub fn format_json_classification(
    report: &ClassificationReport,
    site_map: Option<&SiteMap>,
) -> JsonClassificationReport {
    let sites = report.sites.iter().map(|sc| {
        let source_location = site_map
            .and_then(|m| m.get(sc.site_id))
            .map(|loc| loc.format());
        JsonSiteClassification {
            site_id: format!("0x{:08x}", sc.site_id),
            root_cause: sc.root_cause.label().to_string(),
            confidence: sc.confidence,
            source_location,
        }
    }).collect();

    let summary = report.summary.iter()
        .map(|(cause, count)| (cause.label().to_string(), *count))
        .collect();

    JsonClassificationReport { sites, summary }
}

fn kind_str(kind: &DivergenceKind) -> &'static str {
    match kind {
        DivergenceKind::Branch { .. } => "Branch",
        DivergenceKind::ActiveMask { .. } => "ActiveMask",
        DivergenceKind::Value { .. } => "Value",
        DivergenceKind::Path { .. } => "Path",
        DivergenceKind::ExtraEvents { .. } => "ExtraEvents",
    }
}

/// Format a DiffResult as a JSON report for CI consumption.
///
/// `max_allowed`: if Some(n), the report passes if counted divergences <= n.
///                if None, passes only when there are 0 divergences.
/// `ignore_active_mask`: exclude ActiveMask divergences from the count.
pub fn format_json_report(
    result: &DiffResult,
    site_map: Option<&SiteMap>,
    max_allowed: Option<usize>,
    ignore_active_mask: bool,
) -> JsonDiffReport {
    let counted = if ignore_active_mask {
        result
            .divergences
            .iter()
            .filter(|d| !matches!(d.kind, DivergenceKind::ActiveMask { .. }))
            .count()
    } else {
        result.divergences.len()
    };

    let threshold = max_allowed.unwrap_or(0);
    let passed = counted <= threshold;

    let status = if result.is_identical() {
        "identical".to_string()
    } else {
        "diverged".to_string()
    };

    let divergences: Vec<JsonDivergence> = result
        .divergences
        .iter()
        .map(|div| {
            let source_location = site_map
                .and_then(|m| m.get(div.site_id))
                .map(|loc| loc.format());

            JsonDivergence {
                warp_idx: div.warp_idx,
                event_idx: div.event_idx,
                site_id: format!("0x{:08x}", div.site_id),
                kind: kind_str(&div.kind).to_string(),
                source_location,
            }
        })
        .collect();

    let cross_gpu = result.cross_gpu_info.as_ref().map(|cg| JsonCrossGpuInfo {
        arch_a: cg.arch_a.display(),
        arch_b: cg.arch_b.display(),
        warp_size_a: cg.warp_size_a,
        warp_size_b: cg.warp_size_b,
        grid_a: cg.grid_a,
        grid_b: cg.grid_b,
        block_a: cg.block_a,
        block_b: cg.block_b,
    });

    JsonDiffReport {
        status,
        total_divergences: result.divergences.len(),
        counted_divergences: counted,
        warps_compared: result.warps_compared,
        warps_diverged: result.warps_diverged,
        total_events_a: result.total_events_a,
        total_events_b: result.total_events_b,
        divergence_breakdown: result.divergences_by_kind(),
        divergences,
        threshold: Some(threshold),
        passed,
        cross_gpu,
    }
}

/// Format a SessionDiffResult as a JSON report.
pub fn format_json_session_report(
    result: &SessionDiffResult,
    max_allowed: Option<usize>,
) -> JsonSessionReport {
    let threshold = max_allowed.unwrap_or(0);
    let mut total_divergences = 0usize;
    let mut any_error = false;

    let kernel_results: Vec<JsonKernelResult> = result
        .kernel_results
        .iter()
        .map(|(kernel_name, launch_idx, diff_result)| match diff_result {
            Ok(dr) => {
                let n = dr.divergences.len();
                total_divergences += n;
                JsonKernelResult {
                    kernel: kernel_name.clone(),
                    launch: *launch_idx,
                    status: if dr.is_identical() {
                        "identical".to_string()
                    } else {
                        "diverged".to_string()
                    },
                    divergences: n,
                    error: None,
                }
            }
            Err(e) => {
                any_error = true;
                JsonKernelResult {
                    kernel: kernel_name.clone(),
                    launch: *launch_idx,
                    status: "error".to_string(),
                    divergences: 0,
                    error: Some(format!("{}", e)),
                }
            }
        })
        .collect();

    let has_unmatched =
        !result.unmatched_a.is_empty() || !result.unmatched_b.is_empty();
    let passed =
        !any_error && !has_unmatched && total_divergences <= threshold;

    let status = if total_divergences == 0 && !any_error && !has_unmatched {
        "identical".to_string()
    } else {
        "diverged".to_string()
    };

    JsonSessionReport {
        status,
        kernel_results,
        unmatched_a: result.unmatched_a.clone(),
        unmatched_b: result.unmatched_b.clone(),
        threshold: Some(threshold),
        passed,
    }
}
