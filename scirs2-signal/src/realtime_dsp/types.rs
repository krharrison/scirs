//! Core types for real-time streaming signal processing.
//!
//! These types define the building blocks for bounded-latency DSP pipelines.

// ── RealtimeConfig ────────────────────────────────────────────────────────────

/// Configuration for a real-time processing pipeline.
#[derive(Debug, Clone)]
pub struct RealtimeConfig {
    /// Number of samples per processing block.
    pub block_size: usize,
    /// Maximum allowable latency in samples (hard budget).
    pub max_latency_samples: usize,
    /// Sample rate in Hz (used for latency ↔ time conversions).
    pub sample_rate: f64,
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            max_latency_samples: 512,
            sample_rate: 48_000.0,
        }
    }
}

impl RealtimeConfig {
    /// Convert a sample count to nanoseconds.
    pub fn samples_to_ns(&self, samples: usize) -> u64 {
        ((samples as f64 / self.sample_rate) * 1_000_000_000.0) as u64
    }

    /// Convert nanoseconds to the nearest sample count.
    pub fn ns_to_samples(&self, ns: u64) -> usize {
        ((ns as f64 * self.sample_rate) / 1_000_000_000.0).round() as usize
    }
}

// ── ProcessingStats ───────────────────────────────────────────────────────────

/// Accumulated statistics for a running pipeline.
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Total number of blocks processed so far.
    pub blocks_processed: u64,
    /// Worst-case observed processing latency in nanoseconds.
    pub max_latency_ns: u64,
    /// Exponential-moving-average of processing latency in nanoseconds.
    pub avg_latency_ns: f64,
    /// Number of blocks dropped due to deadline overrun.
    pub underruns: u64,
}

impl ProcessingStats {
    /// Update stats with a newly observed latency measurement (in ns).
    pub fn update(&mut self, latency_ns: u64) {
        self.blocks_processed += 1;
        if latency_ns > self.max_latency_ns {
            self.max_latency_ns = latency_ns;
        }
        // EMA with α = 0.1 for smoothing.
        const ALPHA: f64 = 0.1;
        self.avg_latency_ns = ALPHA * latency_ns as f64 + (1.0 - ALPHA) * self.avg_latency_ns;
    }

    /// Record one underrun event.
    pub fn record_underrun(&mut self) {
        self.underruns += 1;
    }
}

// ── SignalBlock ───────────────────────────────────────────────────────────────

/// A timestamped block of audio samples.
#[derive(Debug, Clone)]
pub struct SignalBlock {
    /// Raw samples for this block.
    pub samples: Vec<f64>,
    /// Capture timestamp in nanoseconds (wall-clock).
    pub timestamp_ns: u64,
    /// Monotonically increasing block identifier.
    pub block_id: u64,
}

impl SignalBlock {
    /// Create a new block.
    pub fn new(samples: Vec<f64>, timestamp_ns: u64, block_id: u64) -> Self {
        Self {
            samples,
            timestamp_ns,
            block_id,
        }
    }
}

// ── ProcessingResult ─────────────────────────────────────────────────────────

/// Result produced by processing one `SignalBlock` through the pipeline.
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Processed output samples (same length as input).
    pub output: Vec<f64>,
    /// Algorithmic latency introduced by all pipeline stages, in samples.
    pub latency_samples: usize,
    /// A snapshot of cumulative pipeline statistics after this block.
    pub stats: ProcessingStats,
}

// ── LatencyBudget ─────────────────────────────────────────────────────────────

/// Breakdown of the total latency budget across pipeline stages.
#[derive(Debug, Clone, Default)]
pub struct LatencyBudget {
    /// Samples consumed by DSP computation (filter tails, etc.).
    pub computation: usize,
    /// Samples consumed by I/O buffering (ring buffers, etc.).
    pub buffering: usize,
    /// Total latency (computation + buffering).
    pub total: usize,
}

impl LatencyBudget {
    /// Create a new budget.
    pub fn new(computation: usize, buffering: usize) -> Self {
        Self {
            computation,
            buffering,
            total: computation + buffering,
        }
    }

    /// Return `true` if this budget is within the supplied ceiling.
    pub fn fits_within(&self, max_samples: usize) -> bool {
        self.total <= max_samples
    }
}
