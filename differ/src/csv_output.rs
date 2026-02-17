//! CSV export for divergence results.

use std::io::Write;
use std::path::Path;

use anyhow::Result;

use crate::differ::{DiffResult, DivergenceKind};
use crate::site_map::SiteMap;

fn escape_csv(s: &str) -> String {
    if s.contains(',') {
        format!("\"{}\"", s)
    } else {
        s.to_string()
    }
}

/// Export divergences to a CSV file.
pub fn export_csv(result: &DiffResult, site_map: Option<&SiteMap>, path: &Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;

    writeln!(f, "warp_idx,event_idx,site_id,kind,details,source_location")?;

    for div in &result.divergences {
        let (kind_str, details) = match &div.kind {
            DivergenceKind::Branch { dir_a, dir_b } => {
                ("Branch", format!("dir_a={} dir_b={}", dir_a, dir_b))
            }
            DivergenceKind::ActiveMask { mask_a, mask_b } => (
                "ActiveMask",
                format!("mask_a=0x{:08x} mask_b=0x{:08x}", mask_a, mask_b),
            ),
            DivergenceKind::Value { val_a, val_b } => {
                ("Value", format!("val_a={} val_b={}", val_a, val_b))
            }
            DivergenceKind::Path { site_a, site_b } => (
                "Path",
                format!("site_a=0x{:08x} site_b=0x{:08x}", site_a, site_b),
            ),
            DivergenceKind::ExtraEvents { count, in_trace_b } => (
                "ExtraEvents",
                format!(
                    "count={} in_trace={}",
                    count,
                    if *in_trace_b { "B" } else { "A" }
                ),
            ),
        };

        let source_loc = site_map
            .and_then(|m| m.get(div.site_id))
            .map(|loc| loc.format())
            .unwrap_or_default();

        writeln!(
            f,
            "{},{},0x{:08x},{},{},{}",
            div.warp_idx, div.event_idx, div.site_id, kind_str, escape_csv(&details), escape_csv(&source_loc),
        )?;
    }

    Ok(())
}
