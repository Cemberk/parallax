//! PRLX: Differential debugger for CUDA traces
//!
//! Compares two execution traces and identifies the exact point of divergence.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

mod differ;
mod parser;
mod report;
mod site_map;
mod trace_format;
mod tui;

use differ::{diff_session, diff_traces, diff_traces_with_remap, DiffConfig};
use parser::{SessionManifest, TraceFile};
use report::{print_divergences, print_history_context, print_summary, print_trace_info};
use site_map::{SiteMap, SiteRemapper};

#[derive(Parser, Debug)]
#[command(name = "prlx-diff")]
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

    /// Compare operand values (can be noisy)
    #[arg(long)]
    values: bool,

    /// Maximum number of divergences to collect (0 = unlimited)
    #[arg(short = 'l', long = "limit", default_value = "0")]
    max_divergences: usize,

    /// Lookahead window size for drift detection
    #[arg(long, default_value = "32")]
    lookahead: usize,

    /// Skip kernel name check (for comparing different kernel variants)
    #[arg(long)]
    force: bool,

    /// Site mapping file (prlx-sites.json) for source location information
    #[arg(long = "map")]
    site_map: Option<PathBuf>,

    /// Dump first N events from trace A (debugging)
    #[arg(long)]
    dump_a: Option<usize>,

    /// Dump first N events from trace B (debugging)
    #[arg(long)]
    dump_b: Option<usize>,

    /// Launch interactive TUI viewer
    #[arg(long)]
    tui: bool,

    /// Show value history context around divergences (time-travel)
    #[arg(long)]
    history: bool,

    /// Display snapshot operands as IEEE 754 floats instead of integers
    #[arg(long)]
    float: bool,

    /// Site map for trace A (for cross-compilation remapping)
    #[arg(long = "remap-a")]
    remap_a: Option<PathBuf>,

    /// Site map for trace B (for cross-compilation remapping)
    #[arg(long = "remap-b")]
    remap_b: Option<PathBuf>,

    /// Treat TRACE_A and TRACE_B as session directories (multi-kernel pipeline diff)
    #[arg(long)]
    session: bool,

    /// Continue comparing after path divergence (default: stop at first per warp)
    #[arg(long)]
    continue_after_path: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.session {
        return run_session_diff(&args);
    }

    println!("Loading traces...");
    let trace_a = TraceFile::open(&args.trace_a)
        .with_context(|| format!("Failed to load trace A: {}", args.trace_a.display()))?;

    let trace_b = TraceFile::open(&args.trace_b)
        .with_context(|| format!("Failed to load trace B: {}", args.trace_b.display()))?;

    let site_map = if let Some(map_path) = &args.site_map {
        Some(
            SiteMap::load(map_path)
                .with_context(|| format!("Failed to load site map: {}", map_path.display()))?,
        )
    } else {
        None
    };

    if let Some(ref map) = site_map {
        println!("Loaded {} site mappings\n", map.len());
    }

    if args.verbose {
        print_trace_info("Trace A", &trace_a);
        print_trace_info("Trace B", &trace_b);
    }

    if let Some(n) = args.dump_a {
        dump_trace("Trace A", &trace_a, n);
        return Ok(());
    }
    if let Some(n) = args.dump_b {
        dump_trace("Trace B", &trace_b, n);
        return Ok(());
    }

    let remapper = if let (Some(remap_a_path), Some(remap_b_path)) =
        (&args.remap_a, &args.remap_b)
    {
        let map_a = SiteMap::load(remap_a_path)
            .with_context(|| format!("Failed to load remap-a: {}", remap_a_path.display()))?;
        let map_b = SiteMap::load(remap_b_path)
            .with_context(|| format!("Failed to load remap-b: {}", remap_b_path.display()))?;
        let r = SiteRemapper::build(&map_a, &map_b);
        println!("Site remapper: {} site_ids remapped\n", r.num_remapped());
        Some(r)
    } else {
        None
    };

    let config = DiffConfig {
        compare_values: args.values,
        max_divergences: args.max_divergences,
        lookahead_window: args.lookahead,
        force: args.force,
        continue_after_path: args.continue_after_path,
    };

    let result = diff_traces_with_remap(&trace_a, &trace_b, &config, remapper.as_ref())?;

    if args.tui {
        tui::run_tui(trace_a, trace_b, result, site_map, args.float)?;
        return Ok(());
    }

    print_summary(&result);
    print_divergences(&result, args.max_shown, site_map.as_ref(), args.float);

    if args.history || trace_a.has_history() || trace_b.has_history() {
        print_history_context(&result, &trace_a, &trace_b, args.max_shown, site_map.as_ref());
    }

    if !result.is_identical() {
        std::process::exit(1);
    }

    Ok(())
}

/// Run session-mode diff: compare two session directories
fn run_session_diff(args: &Args) -> Result<()> {
    println!("Loading sessions...");
    let session_a = SessionManifest::load(&args.trace_a)
        .with_context(|| format!("Failed to load session A: {}", args.trace_a.display()))?;
    let session_b = SessionManifest::load(&args.trace_b)
        .with_context(|| format!("Failed to load session B: {}", args.trace_b.display()))?;

    println!(
        "Session A: {} launches, Session B: {} launches\n",
        session_a.launches.len(),
        session_b.launches.len()
    );

    let config = DiffConfig {
        compare_values: args.values,
        max_divergences: args.max_divergences,
        lookahead_window: args.lookahead,
        force: args.force,
        continue_after_path: args.continue_after_path,
    };

    let session_result = diff_session(&session_a, &session_b, &config);

    let mut any_diverged = false;
    for (kernel_name, launch_idx, result) in &session_result.kernel_results {
        println!("--- Kernel: {} (launch {}) ---", kernel_name, launch_idx);
        match result {
            Ok(diff_result) => {
                print_summary(diff_result);
                if !diff_result.is_identical() {
                    any_diverged = true;
                    print_divergences(diff_result, args.max_shown, None, args.float);
                }
            }
            Err(e) => {
                println!("  ERROR: {}", e);
                any_diverged = true;
            }
        }
        println!();
    }

    // Print unmatched launches as warnings
    for name in &session_result.unmatched_a {
        println!("Warning: Unmatched launch in session A (not in B): {}", name);
    }
    for name in &session_result.unmatched_b {
        println!("Warning: Unmatched launch in session B (not in A): {}", name);
    }
    if !session_result.unmatched_a.is_empty() || !session_result.unmatched_b.is_empty() {
        any_diverged = true;
    }

    if any_diverged {
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
