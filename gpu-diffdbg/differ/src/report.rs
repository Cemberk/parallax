//! Formatted output for diff results

use colored::*;

use crate::differ::{DiffResult, Divergence, DivergenceKind};

/// Print a summary of the diff result
pub fn print_summary(result: &DiffResult) {
    println!("\n{}", "=== Diff Summary ===".bold());
    println!("Total warps:       {}", result.total_warps);
    println!("Events in trace A: {}", result.total_events_a);
    println!("Events in trace B: {}", result.total_events_b);
    println!();

    if result.is_identical() {
        println!("{}", "✓ Traces are IDENTICAL".green().bold());
        return;
    }

    println!(
        "{} {}",
        "✗ DIVERGENCE DETECTED:".red().bold(),
        format!("{} warps diverged", result.divergences.len()).red()
    );
    println!();
}

/// Print detailed divergence information
pub fn print_divergences(result: &DiffResult, max_shown: usize) {
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
        println!(
            "\n{} {} ({})",
            "Site:".bold(),
            format!("0x{:08x}", site_id).cyan(),
            format!("{} warps affected", divs.len()).yellow()
        );

        for div in divs.iter().take(3) {
            // Show first 3 warps for this site
            print_divergence(div);
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
fn print_divergence(div: &Divergence) {
    let kind_str = match div.kind {
        DivergenceKind::ControlFlow => "Control Flow".red(),
        DivergenceKind::BranchDirection => "Branch Direction".yellow(),
        DivergenceKind::ActiveMask => "Active Mask".magenta(),
        DivergenceKind::OperandValue => "Operand Value".blue(),
    };

    println!(
        "  Warp {} @ event {}: {} divergence",
        div.warp_index, div.event_index, kind_str
    );

    match div.kind {
        DivergenceKind::ControlFlow => {
            println!(
                "    Trace A: site=0x{:08x} type={}",
                div.event_a.site_id, div.event_a.event_type
            );
            println!(
                "    Trace B: site=0x{:08x} type={}",
                div.event_b.site_id, div.event_b.event_type
            );
        }
        DivergenceKind::BranchDirection => {
            println!(
                "    Trace A: {} (value={})",
                div.event_a.branch_direction_str(),
                div.event_a.value_a
            );
            println!(
                "    Trace B: {} (value={})",
                div.event_b.branch_direction_str(),
                div.event_b.value_a
            );
        }
        DivergenceKind::ActiveMask => {
            println!(
                "    Trace A: active_mask=0x{:08x} ({} threads)",
                div.event_a.active_mask,
                div.event_a.active_mask.count_ones()
            );
            println!(
                "    Trace B: active_mask=0x{:08x} ({} threads)",
                div.event_b.active_mask,
                div.event_b.active_mask.count_ones()
            );
        }
        DivergenceKind::OperandValue => {
            println!(
                "    Trace A: value={} (0x{:08x})",
                div.event_a.value_a, div.event_a.value_a
            );
            println!(
                "    Trace B: value={} (0x{:08x})",
                div.event_b.value_a, div.event_b.value_a
            );
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
