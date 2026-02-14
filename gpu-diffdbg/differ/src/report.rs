//! Formatted output for diff results

use colored::*;

use crate::differ::{DiffResult, Divergence, DivergenceKind};
use crate::site_map::SiteMap;

/// Print a summary of the diff result
pub fn print_summary(result: &DiffResult) {
    println!("\n{}", "=== Diff Summary ===".bold());
    println!("Total warps:          {}", result.total_warps);
    println!("Warps compared:       {}", result.warps_compared);
    println!("Warps diverged:       {}", result.warps_diverged);
    println!("Events in trace A:    {}", result.total_events_a);
    println!("Events in trace B:    {}", result.total_events_b);
    println!("Total divergences:    {}", result.divergences.len());
    println!();

    if result.is_identical() {
        println!("{}", "✓ Traces are IDENTICAL".green().bold());
        return;
    }

    println!(
        "{} {}",
        "✗ DIVERGENCE DETECTED:".red().bold(),
        format!("{} warps affected", result.warps_diverged).red()
    );

    // Print breakdown by kind
    let by_kind = result.divergences_by_kind();
    if !by_kind.is_empty() {
        println!("\nDivergence breakdown:");
        for (kind, count) in by_kind.iter() {
            println!("  {}: {}", kind, count);
        }
    }
    println!();
}

/// Print detailed divergence information
pub fn print_divergences(result: &DiffResult, max_shown: usize, site_map: Option<&SiteMap>) {
    if result.is_identical() {
        return;
    }

    println!("{}", "=== Divergences ===".bold());

    // Group by site_id
    let by_site = result.divergences_by_site();
    let mut sites: Vec<_> = by_site.keys().collect();
    sites.sort();

    let mut shown = 0;
    for site_id in sites {
        if shown >= max_shown {
            println!("\n... {} more divergences not shown", result.divergences.len() - shown);
            break;
        }

        let divs = &by_site[site_id];

        // Format site header with source location if available
        let site_str = if let Some(map) = site_map {
            if let Some(loc) = map.get(*site_id) {
                format!(
                    "{} at {} ({})",
                    format!("0x{:08x}", site_id).cyan(),
                    loc.format_short().green().bold(),
                    format!("{} warps affected", divs.len()).yellow()
                )
            } else {
                format!(
                    "{} ({})",
                    format!("0x{:08x}", site_id).cyan(),
                    format!("{} warps affected", divs.len()).yellow()
                )
            }
        } else {
            format!(
                "{} ({})",
                format!("0x{:08x}", site_id).cyan(),
                format!("{} warps affected", divs.len()).yellow()
            )
        };

        println!("\n{} {}", "Site:".bold(), site_str);

        for div in divs.iter().take(3) {
            // Show first 3 warps for this site
            print_divergence(div, site_map);
            shown += 1;
            if shown >= max_shown {
                break;
            }
        }

        if divs.len() > 3 {
            println!("  ... and {} more warps at this site", divs.len() - 3);
        }
    }
}

/// Print a single divergence
fn print_divergence(div: &Divergence, site_map: Option<&SiteMap>) {
    print!("  Warp {} @ event {}: ", div.warp_idx, div.event_idx);

    match &div.kind {
        DivergenceKind::Branch { dir_a, dir_b } => {
            println!("{}", "Branch Direction".yellow());
            let dir_a_str = if *dir_a == 0 { "NOT-TAKEN" } else { "TAKEN" };
            let dir_b_str = if *dir_b == 0 { "NOT-TAKEN" } else { "TAKEN" };
            println!("    Trace A: {}", dir_a_str.cyan());
            println!("    Trace B: {}", dir_b_str.magenta());
            println!(
                "    → Threads at site 0x{:08x} took different branch",
                div.site_id
            );
        }
        DivergenceKind::ActiveMask { mask_a, mask_b } => {
            println!("{}", "Active Mask Mismatch".magenta());
            println!(
                "    Trace A: 0x{:08x} ({} threads active)",
                mask_a,
                mask_a.count_ones()
            );
            println!(
                "    Trace B: 0x{:08x} ({} threads active)",
                mask_b,
                mask_b.count_ones()
            );
            let diff = mask_a ^ mask_b;
            if diff != 0 {
                println!("    → Lanes differ: 0x{:08x}", diff);
                // Show which lanes differ
                let mut differing_lanes = Vec::new();
                for lane in 0..32 {
                    if (diff & (1 << lane)) != 0 {
                        differing_lanes.push(lane);
                    }
                }
                if differing_lanes.len() <= 8 {
                    println!("    → Lanes: {:?}", differing_lanes);
                }
            }
        }
        DivergenceKind::Value { val_a, val_b } => {
            println!("{}", "Value Mismatch".blue());
            println!("    Trace A: {} (0x{:08x})", val_a, val_a);
            println!("    Trace B: {} (0x{:08x})", val_b, val_b);
            println!("    → Different operand values at site 0x{:08x}", div.site_id);
        }
        DivergenceKind::Path { site_a, site_b } => {
            println!("{}", "TRUE PATH DIVERGENCE".red().bold());

            // Show site_a with source location if available
            if let Some(map) = site_map {
                if let Some(loc) = map.get(*site_a) {
                    println!("    Trace A reached: 0x{:08x} ({})", site_a, loc.format_short().green());
                } else {
                    println!("    Trace A reached: 0x{:08x}", site_a);
                }
                if let Some(loc) = map.get(*site_b) {
                    println!("    Trace B reached: 0x{:08x} ({})", site_b, loc.format_short().green());
                } else {
                    println!("    Trace B reached: 0x{:08x}", site_b);
                }
            } else {
                println!("    Trace A reached: 0x{:08x}", site_a);
                println!("    Trace B reached: 0x{:08x}", site_b);
            }

            println!("    → {}", "Control flow has truly diverged - different code paths executed".red());
        }
        DivergenceKind::ExtraEvents { count, in_trace_b } => {
            println!("{}", "Extra Events (Drift)".cyan());
            if *in_trace_b {
                println!("    Trace B has {} extra event(s)", count);
                println!("    → Trace B executed more iterations or deeper recursion");
            } else {
                println!("    Trace A has {} extra event(s)", count);
                println!("    → Trace A executed more iterations or deeper recursion");
            }
        }
    }
}

/// Print trace file header information
pub fn print_trace_info(name: &str, trace: &crate::parser::TraceFile) {
    let header = trace.header();
    println!("\n{}", format!("=== {} ===", name).bold());
    println!("Kernel:     {}", header.kernel_name_str());
    println!("Hash:       0x{:016x}", header.kernel_name_hash);
    println!("Grid:       {:?}", header.grid_dim);
    println!("Block:      {:?}", header.block_dim);
    println!("Warps:      {}", header.total_warp_slots);
    println!("Events:     {}", trace.total_events());
    println!("Overflows:  {}", trace.total_overflows());
    println!("Timestamp:  {}", header.timestamp);
    println!("CUDA Arch:  SM_{}", header.cuda_arch);
}
