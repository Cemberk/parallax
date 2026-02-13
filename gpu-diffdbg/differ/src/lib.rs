//! GPU DiffDbg: Differential debugger for CUDA traces
//!
//! This library provides functionality to parse and compare GPU execution traces.

pub mod differ;
pub mod parser;
pub mod report;
pub mod trace_format;
