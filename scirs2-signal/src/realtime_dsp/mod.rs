//! Real-time streaming DSP with bounded-latency guarantees.
//!
//! This module provides:
//!
//! - [`pipeline`] — Composable block-processing pipeline with a latency budget
//!   enforcer.  Built-in stages: [`pipeline::BlockFilter`] (overlap-save FIR),
//!   [`pipeline::BlockCompressor`] (dynamic range), [`pipeline::BlockPitchShifter`]
//!   (linear-interpolation pitch shift), [`pipeline::BlockDelay`] (sample-exact
//!   alignment delay).
//! - [`bounded_latency`] — Latency statistics ([`bounded_latency::LatencyAnalyzer`]),
//!   overrun detection ([`bounded_latency::OverrunDetector`]), and
//!   earliest-deadline-first scheduling ([`bounded_latency::RealtimeScheduler`]).
//! - [`types`] — Shared data types (`RealtimeConfig`, `SignalBlock`, etc.).

pub mod bounded_latency;
pub mod pipeline;
pub mod types;

// Convenience re-exports.
pub use bounded_latency::{LatencyAnalyzer, LatencyGuarantee, OverrunDetector, RealtimeScheduler};
pub use pipeline::{
    BlockCompressor, BlockDelay, BlockFilter, BlockPitchShifter, RealtimePipeline,
    RealtimeProcessor,
};
pub use types::{LatencyBudget, ProcessingResult, ProcessingStats, RealtimeConfig, SignalBlock};
