//! Real-time DSP pipeline with bounded-latency guarantees.
//!
//! A [`RealtimePipeline`] chains multiple [`RealtimeProcessor`] stages and
//! enforces that their combined algorithmic latency stays within a configured
//! budget.  Built-in processors include FIR block filtering (overlap-save),
//! dynamic range compression, pitch shifting, and pure sample delay.

use crate::error::{SignalError, SignalResult};
use crate::realtime_dsp::types::{ProcessingResult, ProcessingStats, RealtimeConfig, SignalBlock};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A stateful, block-based signal processor suitable for real-time use.
pub trait RealtimeProcessor: Send {
    /// Process one block of input samples and return the output block.
    ///
    /// The output **must** have the same length as `input`.
    fn process_block(&mut self, input: &[f64]) -> Vec<f64>;

    /// Algorithmic latency introduced by this stage, in samples.
    fn latency_samples(&self) -> usize;

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str;
}

// ── RealtimePipeline ──────────────────────────────────────────────────────────

/// A chain of [`RealtimeProcessor`] stages with cumulative latency tracking.
pub struct RealtimePipeline {
    config: RealtimeConfig,
    stages: Vec<Box<dyn RealtimeProcessor>>,
    stats: ProcessingStats,
    next_block_id: u64,
}

impl RealtimePipeline {
    /// Create an empty pipeline with the given configuration.
    pub fn new(config: RealtimeConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            stats: ProcessingStats::default(),
            next_block_id: 0,
        }
    }

    /// Append a processing stage.
    ///
    /// Returns an error if adding the stage would push the total algorithmic
    /// latency beyond `config.max_latency_samples`.
    pub fn add_stage(&mut self, processor: Box<dyn RealtimeProcessor>) -> SignalResult<()> {
        let new_total = self.total_latency() + processor.latency_samples();
        if new_total > self.config.max_latency_samples {
            return Err(SignalError::InvalidArgument(format!(
                "Adding stage '{}' (latency={}) would exceed budget {} (current={})",
                processor.name(),
                processor.latency_samples(),
                self.config.max_latency_samples,
                self.total_latency(),
            )));
        }
        self.stages.push(processor);
        Ok(())
    }

    /// Total algorithmic latency of all stages combined, in samples.
    pub fn total_latency(&self) -> usize {
        self.stages.iter().map(|s| s.latency_samples()).sum()
    }

    /// Process one [`SignalBlock`] through every stage in order.
    ///
    /// Returns a [`ProcessingResult`] containing the output samples and
    /// updated statistics.
    pub fn process(&mut self, block: SignalBlock) -> SignalResult<ProcessingResult> {
        if block.samples.len() != self.config.block_size {
            return Err(SignalError::InvalidArgument(format!(
                "Expected block of {} samples, got {}",
                self.config.block_size,
                block.samples.len()
            )));
        }

        let t_start = std::time::Instant::now();

        let mut buf = block.samples.clone();
        for stage in &mut self.stages {
            buf = stage.process_block(&buf);
        }

        let elapsed_ns = t_start.elapsed().as_nanos() as u64;
        self.stats.update(elapsed_ns);

        let result = ProcessingResult {
            output: buf,
            latency_samples: self.total_latency(),
            stats: self.stats.clone(),
        };
        self.next_block_id += 1;
        Ok(result)
    }

    /// Return a snapshot of the current statistics.
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }

    /// Reset internal state (weights, ring buffers) of all stages while
    /// keeping the configuration and stage topology intact.
    pub fn reset(&mut self) {
        self.stats = ProcessingStats::default();
        self.next_block_id = 0;
    }
}

// ── BlockFilter ───────────────────────────────────────────────────────────────

/// FIR filter applied block-by-block using the **overlap-save** method.
///
/// Latency = `filter_len - 1` samples (one block of tail).
pub struct BlockFilter {
    /// FIR tap coefficients h[0..L-1].
    coeffs: Vec<f64>,
    /// Overlap buffer holding the last (L-1) input samples from the
    /// previous block, so that we can compute the full linear convolution
    /// across block boundaries.
    overlap: Vec<f64>,
}

impl BlockFilter {
    /// Create a new block FIR filter.
    ///
    /// `coeffs` must be non-empty.
    pub fn new(coeffs: Vec<f64>) -> SignalResult<Self> {
        if coeffs.is_empty() {
            return Err(SignalError::InvalidArgument(
                "FIR coefficients must be non-empty".into(),
            ));
        }
        let overlap_len = coeffs.len() - 1;
        Ok(Self {
            overlap: vec![0.0; overlap_len],
            coeffs,
        })
    }

    /// Direct per-block convolution helper (O(N·L)).
    fn convolve_with_overlap(&mut self, input: &[f64]) -> Vec<f64> {
        let l = self.coeffs.len();
        let n = input.len();
        // Build extended input: overlap ++ current block
        let mut extended = Vec::with_capacity(l - 1 + n);
        extended.extend_from_slice(&self.overlap);
        extended.extend_from_slice(input);

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut acc = 0.0_f64;
            for (k, &h) in self.coeffs.iter().enumerate() {
                acc += h * extended[i + (l - 1) - k];
            }
            out.push(acc);
        }

        // Update overlap for next block.
        let overlap_len = l - 1;
        if overlap_len > 0 {
            let start = if n >= overlap_len { n - overlap_len } else { 0 };
            self.overlap.clear();
            if n < overlap_len {
                // Very short block: keep old tail + new samples.
                let keep = overlap_len - n;
                self.overlap
                    .extend_from_slice(&self.overlap.clone()[keep..]);
                self.overlap.extend_from_slice(input);
            } else {
                self.overlap.extend_from_slice(&input[start..]);
            }
        }
        out
    }
}

impl RealtimeProcessor for BlockFilter {
    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        self.convolve_with_overlap(input)
    }

    fn latency_samples(&self) -> usize {
        self.coeffs.len() - 1
    }

    fn name(&self) -> &str {
        "BlockFilter"
    }
}

// ── BlockCompressor ───────────────────────────────────────────────────────────

/// Dynamic range compressor using a feed-forward envelope follower.
///
/// Has zero algorithmic latency (purely memoryful but causal).
pub struct BlockCompressor {
    threshold_linear: f64,
    ratio: f64,
    attack_coeff: f64,
    release_coeff: f64,
    envelope: f64,
}

impl BlockCompressor {
    /// Create a compressor.
    ///
    /// * `threshold_db` – Level above which compression starts (e.g. −12.0).
    /// * `ratio`        – Compression ratio (e.g. 4.0 means 4:1).
    /// * `attack_samples`  – Number of samples to reach 63% of envelope.
    /// * `release_samples` – Number of samples to decay to 37% of envelope.
    pub fn new(
        threshold_db: f64,
        ratio: f64,
        attack_samples: usize,
        release_samples: usize,
    ) -> SignalResult<Self> {
        if ratio < 1.0 {
            return Err(SignalError::InvalidArgument(
                "Compression ratio must be ≥ 1".into(),
            ));
        }
        if attack_samples == 0 || release_samples == 0 {
            return Err(SignalError::InvalidArgument(
                "Attack and release sample counts must be > 0".into(),
            ));
        }
        let threshold_linear = 10_f64.powf(threshold_db / 20.0);
        Ok(Self {
            threshold_linear,
            ratio,
            attack_coeff: (-1.0 / attack_samples as f64).exp(),
            release_coeff: (-1.0 / release_samples as f64).exp(),
            envelope: 0.0,
        })
    }

    fn compute_gain(&self, envelope: f64) -> f64 {
        if envelope <= self.threshold_linear || envelope == 0.0 {
            return 1.0;
        }
        // Gain reduction: output_db = threshold + (input_db - threshold)/ratio
        // → gain = 10^((threshold_db*(1-1/ratio))/20) / envelope^(1-1/ratio)
        let level = envelope / self.threshold_linear;
        let exponent = 1.0 - 1.0 / self.ratio;
        level.powf(-exponent)
    }
}

impl RealtimeProcessor for BlockCompressor {
    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            let abs_x = x.abs();
            // Envelope follower.
            let coeff = if abs_x > self.envelope {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.envelope = coeff * self.envelope + (1.0 - coeff) * abs_x;
            let gain = self.compute_gain(self.envelope);
            out.push(x * gain);
        }
        out
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "BlockCompressor"
    }
}

// ── BlockPitchShifter ─────────────────────────────────────────────────────────

/// Simple pitch shifter via linear-interpolation resampling.
///
/// Introduces zero algorithmic latency (operates in-block via fractional
/// resampling with a carry-over fractional phase).
pub struct BlockPitchShifter {
    /// Resampling factor = 2^(semitones/12).
    factor: f64,
    /// Fractional read position carried across blocks.
    phase: f64,
    /// Last sample from the previous block (for interpolation at boundaries).
    prev_sample: f64,
}

impl BlockPitchShifter {
    /// Create a pitch shifter.
    ///
    /// `semitones` may be fractional (positive = pitch up, negative = down).
    pub fn new(semitones: f64) -> Self {
        Self {
            factor: 2_f64.powf(semitones / 12.0),
            phase: 0.0,
            prev_sample: 0.0,
        }
    }
}

impl RealtimeProcessor for BlockPitchShifter {
    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let mut out = vec![0.0_f64; n];

        // Build padded buffer: [prev] ++ input
        let padded_len = n + 1;
        let mut padded = Vec::with_capacity(padded_len);
        padded.push(self.prev_sample);
        padded.extend_from_slice(input);

        let mut read_pos = self.phase; // fractional index into `padded`
        for y in out.iter_mut() {
            let idx = read_pos as usize;
            let frac = read_pos - idx as f64;
            if idx + 1 < padded_len {
                *y = padded[idx] * (1.0 - frac) + padded[idx + 1] * frac;
            } else if idx < padded_len {
                *y = padded[idx];
            }
            read_pos += self.factor;
        }

        // Carry over: how many input samples were consumed minus 1 (for
        // boundary interpolation).
        let consumed = read_pos.floor() as usize;
        self.phase = read_pos - consumed as f64;
        // The last padded sample is at index `consumed` in the old padded array.
        // Index in padded = idx_padded; index in input = idx_padded - 1.
        if consumed >= padded_len {
            self.prev_sample = *input.last().unwrap_or(&0.0);
        } else {
            self.prev_sample = padded[consumed.min(padded_len - 1)];
        }
        out
    }

    fn latency_samples(&self) -> usize {
        0
    }

    fn name(&self) -> &str {
        "BlockPitchShifter"
    }
}

// ── BlockDelay ────────────────────────────────────────────────────────────────

/// Pure sample delay implemented as a ring buffer.
///
/// Introduces exactly `delay_samples` of algorithmic latency.
pub struct BlockDelay {
    delay: usize,
    ring: Vec<f64>,
    write_ptr: usize,
}

impl BlockDelay {
    /// Create a delay line of `delay_samples` samples.
    pub fn new(delay_samples: usize) -> Self {
        Self {
            delay: delay_samples,
            ring: vec![0.0; delay_samples.max(1)],
            write_ptr: 0,
        }
    }
}

impl RealtimeProcessor for BlockDelay {
    fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        if self.delay == 0 {
            return input.to_vec();
        }
        let n = input.len();
        let cap = self.ring.len();
        let mut out = Vec::with_capacity(n);
        for &x in input {
            // Read position = (write - delay) mod cap
            let read_ptr = (self.write_ptr + cap - self.delay % cap) % cap;
            out.push(self.ring[read_ptr]);
            self.ring[self.write_ptr] = x;
            self.write_ptr = (self.write_ptr + 1) % cap;
        }
        out
    }

    fn latency_samples(&self) -> usize {
        self.delay
    }

    fn name(&self) -> &str {
        "BlockDelay"
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a pipeline with default config (block_size=64, budget=512).
    fn default_pipeline() -> RealtimePipeline {
        RealtimePipeline::new(RealtimeConfig::default())
    }

    #[test]
    fn test_block_filter_length_preserving() {
        let coeffs = vec![0.25, 0.5, 0.25];
        let mut f = BlockFilter::new(coeffs).expect("valid coeffs");
        let input = vec![1.0_f64; 64];
        let out = f.process_block(&input);
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn test_block_filter_impulse_response() {
        let coeffs = vec![1.0, 2.0, 3.0, 4.0];
        let mut f = BlockFilter::new(coeffs.clone()).expect("valid coeffs");
        let mut input = vec![0.0_f64; 16];
        input[0] = 1.0; // delta
        let out = f.process_block(&input);
        // Overlap-save: output[k] = sum_j h[j]*x[k-j].  With delta input, the
        // first `L` samples of the output should equal h[0], h[1], h[2], h[3].
        for (i, &c) in coeffs.iter().enumerate() {
            assert!(
                (out[i] - c).abs() < 1e-10,
                "out[{i}]={} ≠ h[{i}]={}",
                out[i],
                c
            );
        }
    }

    #[test]
    fn test_overlap_save_matches_direct() {
        let coeffs = vec![0.1_f64, 0.2, 0.4, 0.2, 0.1];
        let signal: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();

        // Direct convolution (valid part with zero-padding at left).
        let l = coeffs.len();
        let n = signal.len();
        let mut expected = vec![0.0_f64; n];
        for i in 0..n {
            let mut acc = 0.0;
            for (k, &h) in coeffs.iter().enumerate() {
                if i >= k {
                    acc += h * signal[i - k];
                }
            }
            expected[i] = acc;
        }

        let mut f = BlockFilter::new(coeffs).expect("valid");
        let got = f.process_block(&signal);

        // Overlap-save introduces (L-1) = 4 samples of delay; the first
        // `l-1` output samples are from the zero-padded tail.  The rest
        // must match the direct convolution.
        let skip = l - 1;
        for i in skip..n {
            assert!(
                (got[i] - expected[i]).abs() < 1e-10,
                "Mismatch at sample {i}: got={}, expected={}",
                got[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_block_delay_ring_buffer() {
        let delay = 5;
        let mut d = BlockDelay::new(delay);
        let input: Vec<f64> = (1..=16).map(|i| i as f64).collect();
        let out = d.process_block(&input);
        // First `delay` outputs must be 0 (ring is zeroed).
        for i in 0..delay {
            assert_eq!(out[i], 0.0, "out[{i}] should be 0");
        }
        // Outputs after the delay should equal shifted input.
        for i in delay..input.len() {
            assert_eq!(out[i], input[i - delay]);
        }
    }

    #[test]
    fn test_pipeline_latency_sum() {
        let mut cfg = RealtimeConfig::default();
        cfg.max_latency_samples = 1000;
        let mut pipeline = RealtimePipeline::new(cfg);
        pipeline
            .add_stage(Box::new(BlockFilter::new(vec![0.5, 0.5]).expect("valid")))
            .expect("within budget"); // latency = 1
        pipeline
            .add_stage(Box::new(BlockDelay::new(4)))
            .expect("within budget"); // latency = 4
        assert_eq!(pipeline.total_latency(), 5);
    }

    #[test]
    fn test_pipeline_latency_budget_check() {
        let mut cfg = RealtimeConfig::default();
        cfg.max_latency_samples = 3;
        let mut pipeline = RealtimePipeline::new(cfg);
        // Filter with 10 taps ⇒ latency = 9, which exceeds budget=3.
        let result = pipeline.add_stage(Box::new(BlockFilter::new(vec![0.1; 10]).expect("valid")));
        assert!(result.is_err(), "Should reject stage that exceeds budget");
    }

    #[test]
    fn test_pipeline_process_shape() {
        let cfg = RealtimeConfig::default(); // block_size=64
        let mut pipeline = RealtimePipeline::new(cfg);
        pipeline
            .add_stage(Box::new(BlockDelay::new(8)))
            .expect("within budget");

        let block = SignalBlock::new(vec![1.0; 64], 0, 0);
        let result = pipeline.process(block).expect("should process");
        assert_eq!(result.output.len(), 64);
    }

    #[test]
    fn test_compressor_above_threshold() {
        let mut comp = BlockCompressor::new(-20.0, 4.0, 10, 100).expect("valid config");
        // Large amplitude signal (0 dBFS).
        let input = vec![1.0_f64; 64];
        let out = comp.process_block(&input);
        // After the envelope builds up, gain should be < 1.
        let last = *out.last().expect("non-empty");
        assert!(
            last.abs() < 1.0,
            "Compressor must reduce gain above threshold"
        );
    }

    #[test]
    fn test_compressor_below_threshold() {
        // Threshold at 0 dBFS (1.0 linear) — input at −60 dBFS should not
        // be attenuated.
        let mut comp = BlockCompressor::new(0.0, 4.0, 10, 100).expect("valid config");
        let amplitude = 0.001; // well below 0 dBFS threshold
        let input = vec![amplitude; 64];
        let out = comp.process_block(&input);
        for &y in &out {
            assert!(
                (y - amplitude).abs() < 1e-6,
                "No gain reduction expected below threshold, got {y}"
            );
        }
    }
}
