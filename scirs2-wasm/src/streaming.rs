//! Streaming Computation for WASM
//!
//! This module provides incremental / streaming computation primitives for
//! environments where data arrives continuously (audio streams, sensor feeds,
//! real-time data pipelines, etc.).
//!
//! All types are `#[wasm_bindgen]` exported so they can be driven from
//! JavaScript.  Internal arithmetic uses f64 accumulators for numerical
//! stability; output is f32 where JavaScript expects `Float32Array`.
//!
//! ## Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | `OnlineStats` | Incremental mean / variance / skewness (Welford) |
//! | `RollingWindow` | Rolling mean / variance / min / max |
//! | `StreamingFFT` | Overlap-add FFT for audio / time-series streams |
//! | `BufferedProcessor` | Generic accumulate → process → emit pipeline |

use crate::error::WasmError;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ============================================================================
// OnlineStats – Welford one-pass incremental statistics
// ============================================================================

/// Incremental mean, variance and skewness via Welford's online algorithm.
///
/// Accepts f64 samples one at a time or in batches.  Suitable for long
/// streams where storing all data is impractical.
///
/// ## Algorithm
///
/// Welford's method maintains three running accumulators (M1, M2, M3)
/// corresponding to the first three central moments.  This avoids
/// catastrophic cancellation in the naïve two-pass formula.
///
/// ## JavaScript usage
///
/// ```javascript
/// const stats = new OnlineStats();
/// stats.update(1.5);
/// stats.update_batch(new Float64Array([2.0, 3.0, 4.0]));
/// console.log(stats.mean(), stats.variance(), stats.skewness());
/// ```
#[wasm_bindgen]
pub struct OnlineStats {
    /// Number of samples seen so far.
    count: u64,
    /// Running first central moment (mean).
    m1: f64,
    /// Running second central moment ∑(x-μ)² (unnormalised variance).
    m2: f64,
    /// Running third central moment ∑(x-μ)³ (unnormalised skewness).
    m3: f64,
    /// Running minimum.
    min: f64,
    /// Running maximum.
    max: f64,
}

#[wasm_bindgen]
impl OnlineStats {
    /// Create a new empty `OnlineStats` accumulator.
    #[wasm_bindgen(constructor)]
    pub fn new() -> OnlineStats {
        OnlineStats {
            count: 0,
            m1: 0.0,
            m2: 0.0,
            m3: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Incorporate a single new observation.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let n = self.count as f64;
        let delta = x - self.m1;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.m1 += delta_n;
        self.m3 += term1 * delta_n2 * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Incorporate a batch of observations.
    pub fn update_batch(&mut self, xs: &[f64]) {
        for &x in xs {
            self.update(x);
        }
    }

    /// Number of samples accumulated.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Sample mean. Returns NaN if no samples.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.m1
        }
    }

    /// Population variance (divides by N).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return if self.count == 0 { f64::NAN } else { 0.0 };
        }
        self.m2 / self.count as f64
    }

    /// Sample variance (divides by N-1).
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            return if self.count == 0 { f64::NAN } else { 0.0 };
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Population standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Fisher's skewness estimate.  Returns NaN if fewer than 3 samples.
    pub fn skewness(&self) -> f64 {
        if self.count < 3 {
            return f64::NAN;
        }
        let n = self.count as f64;
        let variance = self.m2 / n;
        if variance == 0.0 {
            return 0.0;
        }
        (self.m3 / n) / variance.powf(1.5)
    }

    /// Running minimum seen so far. Returns `f64::INFINITY` if no samples.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Running maximum seen so far. Returns `f64::NEG_INFINITY` if no samples.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Reset the accumulator to its initial empty state.
    pub fn reset(&mut self) {
        self.count = 0;
        self.m1 = 0.0;
        self.m2 = 0.0;
        self.m3 = 0.0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    /// Return a JSON snapshot of all statistics.
    pub fn snapshot_json(&self) -> Result<String, JsValue> {
        let obj = serde_json::json!({
            "count": self.count,
            "mean": self.mean(),
            "variance": self.variance(),
            "sample_variance": self.sample_variance(),
            "std_dev": self.std_dev(),
            "skewness": self.skewness(),
            "min": self.min(),
            "max": self.max(),
        });
        serde_json::to_string(&obj)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RollingWindow – fixed-length sliding window statistics
// ============================================================================

/// Statistics over a fixed-size sliding window of f64 samples.
///
/// As new samples are pushed, the oldest sample is automatically evicted.
/// All statistics are O(1) amortised per update.
///
/// ## JavaScript usage
///
/// ```javascript
/// const rw = new RollingWindow(100);
/// sensor_stream.forEach(v => rw.push(v));
/// const stats = rw.stats_json();
/// const arr = rw.window_as_f32(); // Float32Array for charting
/// ```
#[wasm_bindgen]
pub struct RollingWindow {
    /// Ring buffer of samples.
    buffer: Vec<f64>,
    /// Write position (next slot to overwrite).
    head: usize,
    /// Number of valid samples in the buffer.
    filled: usize,
    /// Window capacity.
    capacity: usize,
    /// Running sum (for mean).
    sum: f64,
    /// Running sum of squares (for variance – note: susceptible to drift on
    /// very long streams; recompute periodically if needed).
    sum_sq: f64,
    /// Number of pushes since last full recompute.
    updates_since_recompute: u64,
    /// How often to recompute sum/sum_sq from scratch (avoids drift).
    recompute_interval: u64,
}

#[wasm_bindgen]
impl RollingWindow {
    /// Create a new rolling window of `capacity` elements.
    ///
    /// # Errors
    ///
    /// Returns a JS error if `capacity == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(capacity: usize) -> Result<RollingWindow, JsValue> {
        if capacity == 0 {
            return Err(
                WasmError::InvalidParameter("RollingWindow: capacity must be > 0".to_string())
                    .into(),
            );
        }
        Ok(RollingWindow {
            buffer: vec![0.0; capacity],
            head: 0,
            filled: 0,
            capacity,
            sum: 0.0,
            sum_sq: 0.0,
            updates_since_recompute: 0,
            recompute_interval: capacity as u64 * 10,
        })
    }

    /// Push a new sample. Evicts the oldest if the window is full.
    pub fn push(&mut self, x: f64) {
        if self.filled == self.capacity {
            // Evict the oldest element at `head`.
            let old = self.buffer[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.filled += 1;
        }

        self.buffer[self.head] = x;
        self.head = (self.head + 1) % self.capacity;

        self.sum += x;
        self.sum_sq += x * x;
        self.updates_since_recompute += 1;

        // Periodic full recompute to prevent floating-point drift.
        if self.updates_since_recompute >= self.recompute_interval {
            self.recompute_sums();
        }
    }

    /// Push multiple samples at once.
    pub fn push_batch(&mut self, xs: &[f64]) {
        for &x in xs {
            self.push(x);
        }
    }

    /// Current number of valid samples in the window.
    pub fn len(&self) -> usize {
        self.filled
    }

    /// `true` if the window contains no samples.
    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Window capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Mean of the current window. Returns NaN if empty.
    pub fn mean(&self) -> f64 {
        if self.filled == 0 {
            return f64::NAN;
        }
        self.sum / self.filled as f64
    }

    /// Population variance of the current window.
    pub fn variance(&self) -> f64 {
        if self.filled < 2 {
            return if self.filled == 0 { f64::NAN } else { 0.0 };
        }
        let n = self.filled as f64;
        let mean = self.sum / n;
        (self.sum_sq / n - mean * mean).max(0.0)
    }

    /// Standard deviation of the current window.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Minimum value in the current window. Returns NaN if empty.
    pub fn min(&self) -> f64 {
        if self.filled == 0 {
            return f64::NAN;
        }
        self.current_slice()
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum value in the current window. Returns NaN if empty.
    pub fn max(&self) -> f64 {
        if self.filled == 0 {
            return f64::NAN;
        }
        self.current_slice()
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Return the window contents as a `Vec<f64>` in chronological order.
    pub fn to_vec(&self) -> Vec<f64> {
        self.current_slice().to_vec()
    }

    /// Return the window contents as a `Vec<f32>` (for `Float32Array` in JS).
    pub fn window_as_f32(&self) -> Vec<f32> {
        self.current_slice().iter().map(|&v| v as f32).collect()
    }

    /// Return summary statistics as a JSON string.
    pub fn stats_json(&self) -> Result<String, JsValue> {
        let obj = serde_json::json!({
            "len": self.filled,
            "capacity": self.capacity,
            "mean": self.mean(),
            "variance": self.variance(),
            "std_dev": self.std_dev(),
            "min": self.min(),
            "max": self.max(),
        });
        serde_json::to_string(&obj)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Reset the window, discarding all data.
    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.updates_since_recompute = 0;
        for v in &mut self.buffer {
            *v = 0.0;
        }
    }
}

impl RollingWindow {
    /// Return a contiguous view of the valid window elements in chronological
    /// order (oldest first).
    fn current_slice(&self) -> Vec<f64> {
        if self.filled < self.capacity {
            self.buffer[..self.filled].to_vec()
        } else {
            // Ring buffer is full; oldest element is at `head`.
            let mut out = Vec::with_capacity(self.capacity);
            out.extend_from_slice(&self.buffer[self.head..]);
            out.extend_from_slice(&self.buffer[..self.head]);
            out
        }
    }

    /// Full recompute of sum and sum_sq to correct floating-point drift.
    fn recompute_sums(&mut self) {
        let slice = self.current_slice();
        self.sum = slice.iter().sum();
        self.sum_sq = slice.iter().map(|&x| x * x).sum();
        self.updates_since_recompute = 0;
    }
}

// ============================================================================
// StreamingFFT – overlap-add FFT for continuous audio/time-series
// ============================================================================

/// Streaming (overlap-add) FFT for real-valued time-series.
///
/// New input samples are accumulated in an internal buffer.  When the buffer
/// contains `window_size` samples, an FFT frame is emitted.  The `hop_size`
/// controls how far the window advances between frames:
///
/// - `hop_size == window_size` → no overlap (standard blocked FFT)
/// - `hop_size == window_size / 2` → 50% overlap (typical for audio)
///
/// Each frame output is an interleaved `[re₀, im₀, re₁, im₁, …]` complex
/// buffer of length `window_size * 2` (or magnitude only if `magnitude_only`
/// is set).
///
/// ## JavaScript usage
///
/// ```javascript
/// const sfft = new StreamingFFT(1024, 512, true);
/// sfft.push_samples(mic_buffer);
/// while (sfft.has_frame()) {
///   const frame = sfft.pop_frame();
///   render_spectrogram(frame);
/// }
/// ```
#[wasm_bindgen]
pub struct StreamingFFT {
    window_size: usize,
    hop_size: usize,
    /// Internal accumulation buffer (ring-style).
    input_buf: Vec<f64>,
    /// How many samples are currently waiting in the buffer.
    buf_fill: usize,
    /// Completed frames waiting to be popped.
    output_frames: Vec<Vec<f32>>,
    /// If true, emit magnitude spectrum instead of complex interleaved.
    magnitude_only: bool,
    /// Optional Hann window coefficients (length == window_size).
    window_fn: Vec<f64>,
}

#[wasm_bindgen]
impl StreamingFFT {
    /// Create a new `StreamingFFT`.
    ///
    /// # Arguments
    ///
    /// * `window_size`    – number of samples per FFT frame (must be ≥ 2)
    /// * `hop_size`       – advance between frames (1 ≤ hop_size ≤ window_size)
    /// * `magnitude_only` – if true, emit `|FFT|` instead of complex interleaved
    ///
    /// # Errors
    ///
    /// Returns a JS error if the arguments are out of range.
    #[wasm_bindgen(constructor)]
    pub fn new(
        window_size: usize,
        hop_size: usize,
        magnitude_only: bool,
    ) -> Result<StreamingFFT, JsValue> {
        if window_size < 2 {
            return Err(WasmError::InvalidParameter(
                "StreamingFFT: window_size must be ≥ 2".to_string(),
            )
            .into());
        }
        if hop_size == 0 || hop_size > window_size {
            return Err(WasmError::InvalidParameter(format!(
                "StreamingFFT: hop_size must be in [1, {}], got {}",
                window_size, hop_size
            ))
            .into());
        }

        // Hann window.
        let window_fn: Vec<f64> = (0..window_size)
            .map(|i| {
                0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * i as f64 / (window_size - 1) as f64).cos())
            })
            .collect();

        Ok(StreamingFFT {
            window_size,
            hop_size,
            input_buf: vec![0.0; window_size],
            buf_fill: 0,
            output_frames: Vec::new(),
            magnitude_only,
            window_fn,
        })
    }

    /// Push new real-valued samples into the stream.
    ///
    /// Any complete frames are computed and queued internally.
    pub fn push_samples(&mut self, samples: &[f64]) {
        for &s in samples {
            // Shift buffer and append new sample.
            if self.buf_fill < self.window_size {
                self.input_buf[self.buf_fill] = s;
                self.buf_fill += 1;
            } else {
                // Buffer full: rotate by hop_size.
                let hop = self.hop_size;
                self.input_buf.copy_within(hop.., 0);
                let new_fill = self.window_size - hop;
                self.input_buf[new_fill] = s;
                self.buf_fill = self.window_size; // stays full
            }

            // Emit a frame when we have enough data.
            if self.buf_fill == self.window_size {
                self.emit_frame();
            }
        }
    }

    /// Return `true` if at least one complete frame is ready.
    pub fn has_frame(&self) -> bool {
        !self.output_frames.is_empty()
    }

    /// Remove and return the oldest completed frame.
    ///
    /// Returns an empty `Vec` if no frames are available.
    pub fn pop_frame(&mut self) -> Vec<f32> {
        if self.output_frames.is_empty() {
            Vec::new()
        } else {
            self.output_frames.remove(0)
        }
    }

    /// Return the number of completed frames waiting to be popped.
    pub fn pending_frames(&self) -> usize {
        self.output_frames.len()
    }

    /// Return the configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Return the configured hop size.
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Flush any partial frame (zero-padded to window_size) and emit it.
    ///
    /// Useful at end-of-stream to process remaining samples.
    pub fn flush(&mut self) {
        if self.buf_fill > 0 {
            // Zero-pad the rest.
            for i in self.buf_fill..self.window_size {
                self.input_buf[i] = 0.0;
            }
            self.buf_fill = self.window_size;
            self.emit_frame();
            self.buf_fill = 0;
        }
    }

    /// Reset the stream, discarding buffered data and pending frames.
    pub fn reset(&mut self) {
        self.buf_fill = 0;
        self.output_frames.clear();
        for v in &mut self.input_buf {
            *v = 0.0;
        }
    }
}

impl StreamingFFT {
    /// Compute DFT of the current window and push the frame into the queue.
    fn emit_frame(&mut self) {
        let n = self.window_size;

        // Apply Hann window.
        let windowed: Vec<f64> = self.input_buf[..n]
            .iter()
            .zip(self.window_fn.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Compute DFT using the Cooley-Tukey definition (no external FFT dep
        // here – we need a simple DFT for the streaming module to remain
        // self-contained and WASM-compatible.  For production use the
        // `scirs2-fft` crate is recommended).
        let frame = if self.magnitude_only {
            dft_magnitude(&windowed)
        } else {
            dft_interleaved(&windowed)
        };

        self.output_frames.push(frame);
    }
}

/// Simple O(n²) DFT – returns interleaved [re, im, re, im, …].
///
/// Not intended for large windows (use scirs2-fft for n > 512).
fn dft_interleaved(x: &[f64]) -> Vec<f32> {
    let n = x.len();
    let mut out = vec![0.0_f32; n * 2];
    let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;

    for k in 0..n {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            let angle = two_pi_over_n * k as f64 * j as f64;
            re += xj * angle.cos();
            im -= xj * angle.sin();
        }
        out[k * 2]     = re as f32;
        out[k * 2 + 1] = im as f32;
    }
    out
}

/// Simple O(n²) DFT – returns only the magnitude spectrum (n/2 + 1 bins).
fn dft_magnitude(x: &[f64]) -> Vec<f32> {
    let n = x.len();
    let bins = n / 2 + 1;
    let mut out = vec![0.0_f32; bins];
    let two_pi_over_n = 2.0 * std::f64::consts::PI / n as f64;

    for k in 0..bins {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        for (j, &xj) in x.iter().enumerate() {
            let angle = two_pi_over_n * k as f64 * j as f64;
            re += xj * angle.cos();
            im -= xj * angle.sin();
        }
        out[k] = (re * re + im * im).sqrt() as f32;
    }
    out
}

// ============================================================================
// BufferedProcessor – accumulate → process → emit pipeline
// ============================================================================

/// Configuration for a `BufferedProcessor`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferedProcessorConfig {
    /// Number of input samples to accumulate before triggering processing.
    pub block_size: usize,
    /// Whether to zero-pad the last block if shorter than `block_size`.
    pub pad_last_block: bool,
    /// Output down-sampling factor: emit 1 output per `downsample` inputs.
    pub downsample: usize,
}

/// The result of processing one block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedBlock {
    /// Block index (0-based, increments per emitted block).
    pub block_index: u64,
    /// Number of input samples that produced this block.
    pub input_count: usize,
    /// Summary statistics for this block.
    pub mean: f64,
    /// Standard deviation of this block.
    pub std_dev: f64,
    /// Root-mean-square energy.
    pub rms: f64,
    /// Peak (absolute maximum) value.
    pub peak: f64,
    /// Down-sampled output (length = `block_size / downsample`).
    pub output: Vec<f32>,
}

/// A generic streaming pipeline that accumulates input samples into blocks,
/// processes each block, and emits `ProcessedBlock` results.
///
/// ## JavaScript usage
///
/// ```javascript
/// const proc = new BufferedProcessor(512, 4, true);
/// mic_stream.forEach(buf => {
///   proc.push(buf);
///   while (proc.has_output()) {
///     const blk = proc.pop_output_json();
///     update_vu_meter(JSON.parse(blk).rms);
///   }
/// });
/// ```
#[wasm_bindgen]
pub struct BufferedProcessor {
    config: BufferedProcessorConfig,
    /// Internal accumulation buffer.
    accumulator: Vec<f64>,
    /// Number of valid samples in `accumulator`.
    acc_fill: usize,
    /// Queue of processed blocks waiting to be popped.
    output_queue: Vec<ProcessedBlock>,
    /// Block counter.
    block_index: u64,
}

#[wasm_bindgen]
impl BufferedProcessor {
    /// Create a new `BufferedProcessor`.
    ///
    /// # Arguments
    ///
    /// * `block_size`    – samples per processing block (must be ≥ 1)
    /// * `downsample`    – output down-sampling factor (must be ≥ 1, must divide `block_size`)
    /// * `pad_last_block` – zero-pad incomplete blocks at flush
    ///
    /// # Errors
    ///
    /// Returns a JS error if arguments are invalid.
    #[wasm_bindgen(constructor)]
    pub fn new(
        block_size: usize,
        downsample: usize,
        pad_last_block: bool,
    ) -> Result<BufferedProcessor, JsValue> {
        if block_size == 0 {
            return Err(
                WasmError::InvalidParameter("BufferedProcessor: block_size must be ≥ 1".to_string())
                    .into(),
            );
        }
        if downsample == 0 {
            return Err(
                WasmError::InvalidParameter("BufferedProcessor: downsample must be ≥ 1".to_string())
                    .into(),
            );
        }
        if block_size % downsample != 0 {
            return Err(WasmError::InvalidParameter(format!(
                "BufferedProcessor: block_size {} must be divisible by downsample {}",
                block_size, downsample
            ))
            .into());
        }

        Ok(BufferedProcessor {
            config: BufferedProcessorConfig {
                block_size,
                pad_last_block,
                downsample,
            },
            accumulator: vec![0.0; block_size],
            acc_fill: 0,
            output_queue: Vec::new(),
            block_index: 0,
        })
    }

    /// Push a slice of samples into the pipeline.
    ///
    /// May trigger zero or more block-processing events internally.
    pub fn push(&mut self, samples: &[f64]) {
        for &s in samples {
            self.accumulator[self.acc_fill] = s;
            self.acc_fill += 1;

            if self.acc_fill == self.config.block_size {
                let block = &self.accumulator[..self.config.block_size];
                let processed = Self::process_block(block, self.block_index, self.config.downsample);
                self.output_queue.push(processed);
                self.block_index += 1;
                self.acc_fill = 0;
            }
        }
    }

    /// Flush partial accumulator (zero-padded if `pad_last_block` is true).
    pub fn flush(&mut self) {
        if self.acc_fill == 0 {
            return;
        }
        if !self.config.pad_last_block {
            return;
        }

        // Zero-pad remaining slots.
        for i in self.acc_fill..self.config.block_size {
            self.accumulator[i] = 0.0;
        }
        let block = &self.accumulator[..self.config.block_size];
        let processed = Self::process_block(block, self.block_index, self.config.downsample);
        self.output_queue.push(processed);
        self.block_index += 1;
        self.acc_fill = 0;
    }

    /// Return `true` if at least one processed block is available.
    pub fn has_output(&self) -> bool {
        !self.output_queue.is_empty()
    }

    /// Return the number of processed blocks waiting to be popped.
    pub fn output_count(&self) -> usize {
        self.output_queue.len()
    }

    /// Remove and return the oldest processed block as a JSON string.
    ///
    /// Returns an empty string if the queue is empty.
    pub fn pop_output_json(&mut self) -> Result<String, JsValue> {
        if self.output_queue.is_empty() {
            return Ok(String::new());
        }
        let block = self.output_queue.remove(0);
        serde_json::to_string(&block)
            .map_err(|e| WasmError::SerializationError(e.to_string()).into())
    }

    /// Reset the processor, discarding all buffered and queued data.
    pub fn reset(&mut self) {
        self.acc_fill = 0;
        self.output_queue.clear();
        self.block_index = 0;
        for v in &mut self.accumulator {
            *v = 0.0;
        }
    }

    /// Number of samples currently waiting in the internal accumulator.
    pub fn buffered_samples(&self) -> usize {
        self.acc_fill
    }

    /// Configured block size.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Configured down-sample factor.
    pub fn downsample(&self) -> usize {
        self.config.downsample
    }
}

impl BufferedProcessor {
    /// Process one full block and return a `ProcessedBlock`.
    fn process_block(block: &[f64], block_index: u64, downsample: usize) -> ProcessedBlock {
        let n = block.len() as f64;

        // Mean.
        let sum: f64 = block.iter().sum();
        let mean = sum / n;

        // Variance + RMS + peak.
        let mut var_acc = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut peak = 0.0_f64;

        for &x in block {
            let d = x - mean;
            var_acc += d * d;
            sum_sq += x * x;
            let abs = x.abs();
            if abs > peak {
                peak = abs;
            }
        }

        let std_dev = (var_acc / n).sqrt();
        let rms = (sum_sq / n).sqrt();

        // Down-sample by averaging groups.
        let out_len = block.len() / downsample;
        let mut output = Vec::with_capacity(out_len);
        for chunk in block.chunks_exact(downsample) {
            let avg: f64 = chunk.iter().sum::<f64>() / downsample as f64;
            output.push(avg as f32);
        }

        ProcessedBlock {
            block_index,
            input_count: block.len(),
            mean,
            std_dev,
            rms,
            peak,
            output,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // OnlineStats
    // -----------------------------------------------------------------------

    #[test]
    fn test_online_stats_basic() {
        let mut s = OnlineStats::new();
        assert!(s.mean().is_nan(), "empty mean should be NaN");

        s.update(2.0);
        s.update(4.0);
        s.update(6.0);

        assert!((s.mean() - 4.0).abs() < 1e-12, "mean = {}", s.mean());
        assert_eq!(s.count(), 3);
        assert!((s.min() - 2.0).abs() < 1e-12);
        assert!((s.max() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_online_stats_variance() {
        let mut s = OnlineStats::new();
        for x in [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            s.update(x);
        }
        // Population variance of {2,4,4,4,5,5,7,9} = 4.0
        assert!((s.variance() - 4.0).abs() < 1e-10, "var = {}", s.variance());
    }

    #[test]
    fn test_online_stats_batch() {
        let mut s = OnlineStats::new();
        s.update_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.mean() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_online_stats_reset() {
        let mut s = OnlineStats::new();
        s.update_batch(&[1.0, 2.0, 3.0]);
        s.reset();
        assert_eq!(s.count(), 0);
        assert!(s.mean().is_nan());
    }

    #[test]
    fn test_online_stats_snapshot_json() {
        let mut s = OnlineStats::new();
        s.update_batch(&[1.0, 2.0, 3.0]);
        let json = s.snapshot_json().expect("snapshot ok");
        assert!(json.contains("mean"));
        assert!(json.contains("variance"));
    }

    // -----------------------------------------------------------------------
    // RollingWindow
    // -----------------------------------------------------------------------

    #[test]
    fn test_rolling_window_basic() {
        let mut rw = RollingWindow::new(3).expect("ok");
        rw.push(1.0);
        rw.push(2.0);
        rw.push(3.0);
        assert!((rw.mean() - 2.0).abs() < 1e-12, "mean = {}", rw.mean());
    }

    #[test]
    fn test_rolling_window_eviction() {
        let mut rw = RollingWindow::new(3).expect("ok");
        rw.push(1.0);
        rw.push(2.0);
        rw.push(3.0);
        rw.push(4.0); // evicts 1.0
        assert!((rw.mean() - 3.0).abs() < 1e-10, "mean = {}", rw.mean());
        assert_eq!(rw.len(), 3);
    }

    #[test]
    fn test_rolling_window_min_max() {
        let mut rw = RollingWindow::new(4).expect("ok");
        rw.push_batch(&[5.0, 2.0, 8.0, 3.0]);
        assert!((rw.min() - 2.0).abs() < 1e-12);
        assert!((rw.max() - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_rolling_window_as_f32() {
        let mut rw = RollingWindow::new(2).expect("ok");
        rw.push(1.5);
        rw.push(2.5);
        let f32s = rw.window_as_f32();
        assert_eq!(f32s.len(), 2);
        assert!((f32s[0] - 1.5_f32).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_window_zero_capacity() {
        assert!(RollingWindow::new(0).is_err());
    }

    // -----------------------------------------------------------------------
    // StreamingFFT
    // -----------------------------------------------------------------------

    #[test]
    fn test_streaming_fft_no_frames_until_window_full() {
        let mut sfft = StreamingFFT::new(8, 4, false).expect("ok");
        sfft.push_samples(&[1.0, 2.0, 3.0]); // only 3, not yet 8
        assert!(!sfft.has_frame());
    }

    #[test]
    fn test_streaming_fft_emits_frame() {
        let mut sfft = StreamingFFT::new(8, 8, false).expect("ok");
        let samples: Vec<f64> = (0..8).map(|i| i as f64).collect();
        sfft.push_samples(&samples);
        assert!(sfft.has_frame(), "should have one frame");
        let frame = sfft.pop_frame();
        assert_eq!(frame.len(), 16, "interleaved: 8 complex = 16 floats");
    }

    #[test]
    fn test_streaming_fft_magnitude() {
        let mut sfft = StreamingFFT::new(8, 8, true).expect("ok");
        let samples: Vec<f64> = vec![1.0; 8];
        sfft.push_samples(&samples);
        assert!(sfft.has_frame());
        let frame = sfft.pop_frame();
        // Magnitude: n/2 + 1 = 5 bins
        assert_eq!(frame.len(), 5, "magnitude bins = n/2+1 = 5");
        // DC component should be 8 * 1.0 = 8 (with Hann window scaling, approx).
        assert!(frame[0] > 0.0, "DC bin must be positive");
    }

    #[test]
    fn test_streaming_fft_flush() {
        let mut sfft = StreamingFFT::new(8, 8, false).expect("ok");
        sfft.push_samples(&[1.0, 2.0, 3.0]);
        assert!(!sfft.has_frame());
        sfft.flush();
        assert!(sfft.has_frame(), "flush should emit padded frame");
    }

    #[test]
    fn test_streaming_fft_invalid_args() {
        assert!(StreamingFFT::new(0, 1, false).is_err());
        assert!(StreamingFFT::new(8, 0, false).is_err());
        assert!(StreamingFFT::new(8, 9, false).is_err());
    }

    // -----------------------------------------------------------------------
    // BufferedProcessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffered_processor_basic() {
        let mut proc = BufferedProcessor::new(4, 2, false).expect("ok");
        proc.push(&[1.0, 2.0, 3.0, 4.0]);
        assert!(proc.has_output());
        let json = proc.pop_output_json().expect("pop ok");
        let block: serde_json::Value = serde_json::from_str(&json).expect("parse ok");
        // mean of [1,2,3,4] = 2.5
        let mean = block["mean"].as_f64().expect("mean field");
        assert!((mean - 2.5).abs() < 1e-10);
        // output length = block_size / downsample = 4/2 = 2
        let out_len = block["output"].as_array().expect("output array").len();
        assert_eq!(out_len, 2);
    }

    #[test]
    fn test_buffered_processor_flush_pad() {
        let mut proc = BufferedProcessor::new(4, 1, true).expect("ok");
        proc.push(&[1.0, 2.0]); // partial block
        assert!(!proc.has_output());
        proc.flush();
        assert!(proc.has_output(), "flush should emit padded block");
    }

    #[test]
    fn test_buffered_processor_flush_no_pad() {
        let mut proc = BufferedProcessor::new(4, 1, false).expect("ok");
        proc.push(&[1.0, 2.0]);
        proc.flush(); // no padding, nothing emitted
        assert!(!proc.has_output());
    }

    #[test]
    fn test_buffered_processor_rms() {
        let mut proc = BufferedProcessor::new(4, 1, false).expect("ok");
        // All 1.0 → rms = 1.0
        proc.push(&[1.0, 1.0, 1.0, 1.0]);
        let json = proc.pop_output_json().expect("ok");
        let block: serde_json::Value = serde_json::from_str(&json).expect("ok");
        let rms = block["rms"].as_f64().expect("rms");
        assert!((rms - 1.0).abs() < 1e-10, "rms = {rms}");
    }

    #[test]
    fn test_buffered_processor_invalid_args() {
        assert!(BufferedProcessor::new(0, 1, false).is_err());
        assert!(BufferedProcessor::new(4, 0, false).is_err());
        assert!(BufferedProcessor::new(4, 3, false).is_err()); // 4 % 3 != 0
    }

    #[test]
    fn test_buffered_processor_reset() {
        let mut proc = BufferedProcessor::new(4, 1, false).expect("ok");
        proc.push(&[1.0, 2.0, 3.0, 4.0]);
        proc.reset();
        assert!(!proc.has_output());
        assert_eq!(proc.buffered_samples(), 0);
    }
}
