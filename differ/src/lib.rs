//! PRLX: Differential debugger for CUDA traces
//!
//! This library provides functionality to parse and compare GPU execution traces.

pub mod classifier;
pub mod csv_output;
pub mod differ;
pub mod flamegraph;
pub mod json_output;
pub mod parser;
pub mod report;
pub mod site_map;
pub mod trace_format;
pub mod tui;
