//! GPU DiffDbg: Differential debugger for CUDA traces
//!
//! Compares two execution traces and identifies the exact point of divergence.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

mod differ;
mod parser;
mod report;
mod trace_format;

use differ::diff_traces;
use parser::TraceFile;
use report::{print_divergences, print_summary, print_trace_info};

#[derive(Parser, Debug)]
#[command(name = "gddbg-diff")]
#[command(about = "GPU Differential Debugger - Compare CUDA execution traces")]
#[command(version)]
struct Args {
    /// First trace file (baseline)
    #[arg(value_name = "TRACE_A")]
    trace_a: PathBuf,

    /// Second trace file (compare against baseline)
    #[arg(value_name = "TRACE_B")]
    trace_b: PathBuf,

    /// Maximum number of divergences to display
    #[arg(short = 'n', long, default_value = "10")]
    max_shown: usize,

    /// Show detailed trace information
    #[arg(short = 'v', long)]
    verbose: bool,

    /// Dump first N events from trace A (debugging)
    #[arg(long)]
    dump_a: Option<usize>,

    /// Dump first N events from trace B (debugging)
    #[arg(long)]
    dump_b: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load trace files
    println!("Loading traces...");
    let trace_a = TraceFile::open(&args.trace_a)
        .with_context(|| format!("Failed to load trace A: {}", args.trace_a.display()))?;

    let trace_b = TraceFile::open(&args.trace_b)
        .with_context(|| format!("Failed to load trace B: {}", args.trace_b.display()))?;

    // Print trace information
    if args.verbose {
        print_trace_info("Trace A", &trace_a);
        print_trace_info("Trace B", &trace_b);
    }

    // Dump mode (debugging)
    if let Some(n) = args.dump_a {
        dump_trace("Trace A", &trace_a, n);
        return Ok(());
    }
    if let Some(n) = args.dump_b {
        dump_trace("Trace B", &trace_b, n);
        return Ok(());
    }

    // Perform differential analysis
    let result = diff_traces(&trace_a, &trace_b)?;

    // Print results
    print_summary(&result);
    print_divergences(&result, args.max_shown);

    // Exit with non-zero if divergences found
    if !result.is_identical() {
        std::process::exit(1);
    }

    Ok(())
}

/// Dump first N events from a trace (for debugging)
fn dump_trace(name: &str, trace: &TraceFile, max_events: usize) {
    println!("\n=== {} ===", name);
    print_trace_info(name, trace);

    println!("\n=== Events (first {} events) ===", max_events);

    let mut shown = 0;
    for (warp_idx, header, events) in trace.warps() {
        if shown >= max_events {
            break;
        }

        println!(
            "\nWarp {} ({} events, {} overflows):",
            warp_idx, header.num_events, header.overflow_count
        );

        for (event_idx, evt) in events.iter().enumerate() {
            if shown >= max_events {
                break;
            }

            println!(
                "  [{}] site=0x{:08x} type={} branch={} active_mask=0x{:08x} value={}",
                event_idx,
                evt.site_id,
                evt.event_type,
                evt.branch_dir,
                evt.active_mask,
                evt.value_a
            );

            shown += 1;
        }
    }
}
