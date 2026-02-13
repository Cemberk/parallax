use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod differ;
mod report;
mod trace_format;

use differ::{diff_traces, DiffOptions};
use trace_format::TraceFile;

#[derive(Parser)]
#[command(name = "gddbg-diff")]
#[command(about = "Compare two GPU kernel execution traces for differential debugging")]
#[command(version)]
struct Cli {
    /// Path to the first trace file (Trace A / "reference")
    trace_a: PathBuf,

    /// Path to the second trace file (Trace B / "test")
    trace_b: PathBuf,

    /// Output format: "terminal" (default), "json"
    #[arg(short, long, default_value = "terminal")]
    format: String,

    /// Maximum number of divergence sites to report
    #[arg(short = 'n', long, default_value = "100")]
    max_divergences: usize,

    /// Show value differences even when branch direction matches
    #[arg(long)]
    show_value_diffs: bool,

    /// Path to source root (for resolving filenames in traces)
    #[arg(long)]
    source_root: Option<PathBuf>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    eprintln!("[gddbg-diff] Loading trace A: {:?}", cli.trace_a);
    let trace_a = TraceFile::open(&cli.trace_a)?;

    eprintln!("[gddbg-diff] Loading trace B: {:?}", cli.trace_b);
    let trace_b = TraceFile::open(&cli.trace_b)?;

    eprintln!("[gddbg-diff] Validating compatibility...");
    trace_format::validate_compatible(&trace_a, &trace_b)?;

    eprintln!("[gddbg-diff] Comparing traces...");
    let divergences = diff_traces(
        &trace_a,
        &trace_b,
        &DiffOptions {
            max_divergences: cli.max_divergences,
            show_value_diffs: cli.show_value_diffs,
        },
    );

    eprintln!("[gddbg-diff] Found {} divergences", divergences.len());
    eprintln!();

    // Output report
    match cli.format.as_str() {
        "terminal" => report::print_terminal(&divergences, &trace_a, &trace_b, cli.source_root.as_deref()),
        "json" => report::print_json(&divergences, &trace_a, &trace_b)?,
        _ => anyhow::bail!("Unknown format: {}", cli.format),
    }

    // Exit code: 0 = identical, 1 = divergences found, 2 = error
    std::process::exit(if divergences.is_empty() { 0 } else { 1 });
}
