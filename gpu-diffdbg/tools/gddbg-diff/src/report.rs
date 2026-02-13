use crate::differ::*;
use crate::trace_format::*;
use anyhow::Result;
use colored::*;

pub fn print_terminal(
    divergences: &[Divergence],
    trace_a: &TraceFile,
    trace_b: &TraceFile,
    _source_root: Option<&std::path::Path>,
) {
    println!("{}", "═".repeat(70).bright_blue());
    println!(" {} — Trace Comparison Report", "GPU DiffDbg".bright_white().bold());
    println!(
        " Trace A: {} (kernel: {}, grid: {}x{}x{})",
        "trace_a.gddbg".bright_cyan(),
        trace_a.get_kernel_name().bright_yellow(),
        trace_a.header.grid_dim[0],
        trace_a.header.grid_dim[1],
        trace_a.header.grid_dim[2]
    );
    println!(
        " Trace B: {} (kernel: {}, grid: {}x{}x{})",
        "trace_b.gddbg".bright_cyan(),
        trace_b.get_kernel_name().bright_yellow(),
        trace_b.header.grid_dim[0],
        trace_b.header.grid_dim[1],
        trace_b.header.grid_dim[2]
    );
    println!("{}", "═".repeat(70).bright_blue());
    println!();

    if divergences.is_empty() {
        println!(
            "{}",
            " ✓ IDENTICAL TRACES - No divergences found".bright_green().bold()
        );
        println!("{}", "═".repeat(70).bright_blue());
        return;
    }

    for (idx, div) in divergences.iter().enumerate() {
        println!(
            " {} {} — {}",
            "DIVERGENCE".bright_red().bold(),
            format!("#{}", idx + 1).bright_white(),
            format_divergence_kind(&div.kind).bright_yellow()
        );
        println!(" {}", "─".repeat(68).bright_black());

        println!(
            " Location:    site_id=0x{:08X}",
            div.site_id
        );
        println!(
            " Warp:        block({},{},{}) warp {}",
            div.block_idx.0, div.block_idx.1, div.block_idx.2, div.warp_id
        );
        println!(" Sequence:    event #{} (of max)", div.event_index);
        println!();

        match div.kind {
            DivergenceKind::BranchDirection => {
                let dir_a = if div.trace_a_event.branch_dir != 0 {
                    "TAKEN".bright_green()
                } else {
                    "NOT TAKEN".bright_red()
                };
                let dir_b = if div.trace_b_event.branch_dir != 0 {
                    "TAKEN".bright_green()
                } else {
                    "NOT TAKEN".bright_red()
                };

                println!(
                    "   Trace A:   branch {}      (value_a=0x{:08X})",
                    dir_a, div.trace_a_event.value_a
                );
                println!(
                    "   Trace B:   branch {}  (value_a=0x{:08X})",
                    dir_b, div.trace_b_event.value_a
                );
            }
            DivergenceKind::ActiveMaskMismatch => {
                println!(
                    "   Trace A:   active_mask=0x{:08X}  ({} threads active)",
                    div.trace_a_event.active_mask,
                    div.trace_a_event.active_mask.count_ones()
                );
                println!(
                    "   Trace B:   active_mask=0x{:08X}  ({} threads active)",
                    div.trace_b_event.active_mask,
                    div.trace_b_event.active_mask.count_ones()
                );
                println!(
                    "   Diff:      0x{:08X}  ({} threads differ)",
                    div.trace_a_event.active_mask ^ div.trace_b_event.active_mask,
                    (div.trace_a_event.active_mask ^ div.trace_b_event.active_mask).count_ones()
                );
            }
            DivergenceKind::ValueMismatch => {
                println!(
                    "   Trace A:   value=0x{:08X}  ({})",
                    div.trace_a_event.value_a,
                    format_value(div.trace_a_event.value_a)
                );
                println!(
                    "   Trace B:   value=0x{:08X}  ({})",
                    div.trace_b_event.value_a,
                    format_value(div.trace_b_event.value_a)
                );
            }
            DivergenceKind::SequenceMismatch => {
                println!(
                    "   Trace A:   site_id=0x{:08X}",
                    div.trace_a_event.site_id
                );
                println!(
                    "   Trace B:   site_id=0x{:08X}",
                    div.trace_b_event.site_id
                );
                println!("   {}", "Different code paths reached".bright_yellow());
            }
            DivergenceKind::LengthMismatch => {
                println!(
                    "   Trace A:   {} events",
                    div.trace_a_event.site_id  // Abusing this field for length
                );
                println!(
                    "   Trace B:   {} events",
                    div.trace_b_event.site_id
                );
            }
        }

        println!();
    }

    println!("{}", "═".repeat(70).bright_blue());
    println!(
        " Summary: {} unique divergence sites",
        divergences.len().to_string().bright_red().bold()
    );
    println!("{}", "═".repeat(70).bright_blue());
}

pub fn print_json(
    divergences: &[Divergence],
    trace_a: &TraceFile,
    trace_b: &TraceFile,
) -> Result<()> {
    use serde_json::json;

    let div_json: Vec<_> = divergences
        .iter()
        .map(|div| {
            json!({
                "site_id": format!("0x{:08X}", div.site_id),
                "block_idx": [div.block_idx.0, div.block_idx.1, div.block_idx.2],
                "warp_id": div.warp_id,
                "event_index": div.event_index,
                "kind": format_divergence_kind(&div.kind),
                "trace_a": {
                    "site_id": format!("0x{:08X}", div.trace_a_event.site_id),
                    "event_type": div.trace_a_event.event_type,
                    "branch_dir": div.trace_a_event.branch_dir,
                    "active_mask": format!("0x{:08X}", div.trace_a_event.active_mask),
                    "value_a": format!("0x{:08X}", div.trace_a_event.value_a),
                },
                "trace_b": {
                    "site_id": format!("0x{:08X}", div.trace_b_event.site_id),
                    "event_type": div.trace_b_event.event_type,
                    "branch_dir": div.trace_b_event.branch_dir,
                    "active_mask": format!("0x{:08X}", div.trace_b_event.active_mask),
                    "value_a": format!("0x{:08X}", div.trace_b_event.value_a),
                },
            })
        })
        .collect();

    let output = json!({
        "trace_a": {
            "kernel_name": trace_a.get_kernel_name(),
            "grid_dim": trace_a.header.grid_dim,
            "block_dim": trace_a.header.block_dim,
        },
        "trace_b": {
            "kernel_name": trace_b.get_kernel_name(),
            "grid_dim": trace_b.header.grid_dim,
            "block_dim": trace_b.header.block_dim,
        },
        "divergences": div_json,
        "summary": {
            "total_divergences": divergences.len(),
        }
    });

    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn format_divergence_kind(kind: &DivergenceKind) -> String {
    match kind {
        DivergenceKind::BranchDirection => "Branch Direction Mismatch".to_string(),
        DivergenceKind::ActiveMaskMismatch => "Active Mask Mismatch (SIMT Divergence)".to_string(),
        DivergenceKind::ValueMismatch => "Value Mismatch".to_string(),
        DivergenceKind::SequenceMismatch => "Sequence Mismatch (Different Code Path)".to_string(),
        DivergenceKind::LengthMismatch => "Trace Length Mismatch".to_string(),
    }
}

fn format_value(val: u32) -> String {
    // Try to interpret as float
    let as_float = f32::from_bits(val);
    if as_float.is_finite() {
        format!("{} or {}", val as i32, as_float)
    } else {
        format!("{}", val as i32)
    }
}
