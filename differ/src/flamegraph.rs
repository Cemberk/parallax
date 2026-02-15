//! Export divergences to Chrome Trace Format (catapult JSON) for visualization.
//!
//! Open the output file in chrome://tracing or https://ui.perfetto.dev

use anyhow::Result;
use serde::Serialize;
use std::path::Path;

use crate::differ::{DiffResult, DivergenceKind};
use crate::parser::TraceFile;
use crate::site_map::SiteMap;

#[derive(Serialize)]
pub struct ChromeTraceOutput {
    #[serde(rename = "traceEvents")]
    pub trace_events: Vec<ChromeTraceEvent>,
}

#[derive(Serialize)]
pub struct ChromeTraceEvent {
    pub name: String,
    pub cat: String,
    pub ph: String,
    pub ts: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    pub pid: u32,
    pub tid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
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

/// Export diff results to Chrome Trace Format JSON.
///
/// Mapping:
/// - pid = CUDA block index
/// - tid = warp index within block
/// - Duration events for each divergence, grouped by site
/// - Counter events for per-site divergence frequency
pub fn export_flamegraph(
    trace_a: &TraceFile,
    _trace_b: &TraceFile,
    result: &DiffResult,
    site_map: Option<&SiteMap>,
    output: &Path,
) -> Result<()> {
    let header = trace_a.header();
    let warps_per_block = header.num_warps_per_block.max(1);
    let mut events = Vec::new();

    // Metadata: kernel name
    events.push(ChromeTraceEvent {
        name: "process_name".to_string(),
        cat: "__metadata".to_string(),
        ph: "M".to_string(),
        ts: 0.0,
        dur: None,
        pid: 0,
        tid: 0,
        args: Some(serde_json::json!({
            "name": String::from_utf8_lossy(
                &header.kernel_name[..header.kernel_name.iter().position(|&b| b == 0).unwrap_or(header.kernel_name.len())]
            ).to_string()
        })),
    });

    // Duration events per divergence, laid out sequentially per warp
    let mut ts_cursor: f64 = 0.0;
    let event_duration: f64 = 10.0; // microseconds per event (synthetic)

    let by_site = result.divergences_by_site();

    for (&site_id, divs) in &by_site {
        let site_name = site_map
            .and_then(|m| m.get(site_id))
            .map(|loc| loc.format_short())
            .unwrap_or_else(|| format!("0x{:08x}", site_id));

        for div in divs {
            let block_id = div.warp_idx / warps_per_block;
            let warp_in_block = div.warp_idx % warps_per_block;
            let ks = kind_str(&div.kind);

            events.push(ChromeTraceEvent {
                name: format!("{} [{}]", site_name, ks),
                cat: ks.to_string(),
                ph: "X".to_string(),
                ts: ts_cursor,
                dur: Some(event_duration),
                pid: block_id,
                tid: warp_in_block,
                args: Some(serde_json::json!({
                    "site_id": format!("0x{:08x}", site_id),
                    "kind": ks,
                    "event_idx": div.event_idx,
                    "warp_idx": div.warp_idx,
                })),
            });
            ts_cursor += event_duration;
        }
    }

    // Counter events: per-site divergence frequency (heatmap)
    let mut counter_ts = ts_cursor + 100.0;
    for (&site_id, divs) in &by_site {
        let site_name = site_map
            .and_then(|m| m.get(site_id))
            .map(|loc| loc.format_short())
            .unwrap_or_else(|| format!("0x{:08x}", site_id));

        events.push(ChromeTraceEvent {
            name: "divergence_count".to_string(),
            cat: "heatmap".to_string(),
            ph: "C".to_string(),
            ts: counter_ts,
            dur: None,
            pid: 0,
            tid: 0,
            args: Some(serde_json::json!({
                site_name: divs.len(),
            })),
        });
        counter_ts += 1.0;
    }

    let output_data = ChromeTraceOutput {
        trace_events: events,
    };
    let json = serde_json::to_string_pretty(&output_data)?;
    std::fs::write(output, json)?;

    Ok(())
}
