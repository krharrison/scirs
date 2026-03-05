//! Neural network weight quantization for inference acceleration
//!
//! Provides post-training quantization (PTQ) and quantization-aware training (QAT)
//! helpers operating directly on `Array2<f64>` weight tensors.
//!
//! ## Supported modes
//! - **Int8 symmetric** – scale only, zero-point = 0
//! - **Int8 asymmetric** – scale + integer zero-point
//! - **Int16 symmetric / asymmetric**
//! - **Per-channel** quantization (one scale per output channel / row)
//! - **Percentile calibration** (clamp outliers before computing scale)
//! - **Fake-quantization** for QAT (quantize then immediately dequantize)
//!
//! ## Quick start
//!
//! ```rust
//! use scirs2_neural::training::quantization::{QuantizationConfig, Quantizer, CalibrationMode};
//! use scirs2_core::ndarray::array;
//!
//! let weights = array![[0.1_f64, -0.5, 0.9], [-0.3, 0.7, 0.0]];
//! let config = QuantizationConfig {
//!     bits: 8,
//!     symmetric: true,
//!     per_channel: false,
//!     calibration: CalibrationMode::MinMax,
//! };
//! let mut quantizer = Quantizer::new(config);
//! quantizer.collect_calibration_data(weights.view());
//! let (qw, params) = quantizer.quantize_i8(weights.view()).expect("quantize failed");
//! let recovered = quantizer.dequantize_i8(qw.view(), &params).expect("dequantize failed");
//! assert_eq!(recovered.shape(), weights.shape());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, ToPrimitive};
use std::fmt::Debug;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Calibration strategy used to determine the quantization range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMode {
    /// Use the global minimum and maximum observed value.
    MinMax,
    /// Clip to a percentile of the observed distribution (default 99.9 %).
    Percentile,
    /// Moving-average min/max (exponential smoothing).
    MovingAverage,
}

impl Default for CalibrationMode {
    fn default() -> Self {
        Self::MinMax
    }
}

/// Top-level quantization configuration.
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Target bit-width (4, 8, or 16 are most common).
    pub bits: u8,
    /// If `true` use symmetric quantization (zero-point = 0).
    /// If `false` use asymmetric quantization (integer zero-point).
    pub symmetric: bool,
    /// Compute one scale per output channel (row in a 2-D weight matrix)
    /// instead of a single tensor-wide scale.
    pub per_channel: bool,
    /// Strategy used during calibration data collection.
    pub calibration: CalibrationMode,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            symmetric: true,
            per_channel: false,
            calibration: CalibrationMode::MinMax,
        }
    }
}

/// Per-tensor (or per-channel) quantization parameters produced by calibration.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor(s).  Length == 1 for per-tensor, == #channels for per-channel.
    pub scales: Vec<f64>,
    /// Integer zero-point(s).  All zeros for symmetric mode.
    pub zero_points: Vec<i32>,
    /// Bit-width used when these params were computed.
    pub bits: u8,
    /// Whether these params were computed in symmetric mode.
    pub symmetric: bool,
}

impl QuantParams {
    /// Return the scale for channel `c` (or the single tensor-wide scale if per-tensor).
    pub fn scale_for(&self, c: usize) -> f64 {
        if self.scales.len() == 1 {
            self.scales[0]
        } else {
            self.scales[c]
        }
    }

    /// Return the zero-point for channel `c`.
    pub fn zero_point_for(&self, c: usize) -> i32 {
        if self.zero_points.len() == 1 {
            self.zero_points[0]
        } else {
            self.zero_points[c]
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Calibration accumulator (internal)
// ────────────────────────────────────────────────────────────────────────────

/// Running statistics collected across multiple calibration batches.
#[derive(Debug, Default, Clone)]
struct CalibStats {
    global_min: f64,
    global_max: f64,
    /// Sorted flat samples, populated lazily for percentile mode.
    samples: Vec<f64>,
    count: usize,
    /// EMA state for MovingAverage mode.
    ema_min: f64,
    ema_max: f64,
}

impl CalibStats {
    fn new() -> Self {
        Self {
            global_min: f64::INFINITY,
            global_max: f64::NEG_INFINITY,
            samples: Vec::new(),
            count: 0,
            ema_min: f64::INFINITY,
            ema_max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, values: &[f64], mode: CalibrationMode, ema_decay: f64) {
        let local_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let local_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if local_min < self.global_min {
            self.global_min = local_min;
        }
        if local_max > self.global_max {
            self.global_max = local_max;
        }

        match mode {
            CalibrationMode::Percentile => {
                // Keep a capped sample pool (max 1M values) to avoid OOM.
                const MAX_SAMPLES: usize = 1_000_000;
                if self.samples.len() < MAX_SAMPLES {
                    let remaining = MAX_SAMPLES - self.samples.len();
                    let take = values.len().min(remaining);
                    self.samples.extend_from_slice(&values[..take]);
                }
            }
            CalibrationMode::MovingAverage => {
                if self.count == 0 {
                    self.ema_min = local_min;
                    self.ema_max = local_max;
                } else {
                    self.ema_min = ema_decay * self.ema_min + (1.0 - ema_decay) * local_min;
                    self.ema_max = ema_decay * self.ema_max + (1.0 - ema_decay) * local_max;
                }
            }
            CalibrationMode::MinMax => {}
        }

        self.count += values.len();
    }

    /// Compute the effective (min, max) range after calibration.
    fn effective_range(&mut self, mode: CalibrationMode, percentile: f64) -> (f64, f64) {
        match mode {
            CalibrationMode::MinMax => (self.global_min, self.global_max),
            CalibrationMode::MovingAverage => (self.ema_min, self.ema_max),
            CalibrationMode::Percentile => {
                if self.samples.is_empty() {
                    return (self.global_min, self.global_max);
                }
                self.samples.sort_by(|a, b| {
                    a.partial_cmp(b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let n = self.samples.len();
                let lo_idx =
                    ((1.0 - percentile) * (n as f64 - 1.0)).round() as usize;
                let hi_idx = (percentile * (n as f64 - 1.0)).round() as usize;
                let lo = self.samples[lo_idx.min(n - 1)];
                let hi = self.samples[hi_idx.min(n - 1)];
                (lo, hi)
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Quantizer
// ────────────────────────────────────────────────────────────────────────────

/// Stateful quantizer.
///
/// Collects calibration data across one or more batches, then produces
/// `QuantParams` and performs quantization / dequantization.
#[derive(Debug, Clone)]
pub struct Quantizer {
    /// Configuration controlling bit-width, symmetry, and calibration mode.
    pub config: QuantizationConfig,
    /// Percentile to use when `config.calibration == CalibrationMode::Percentile`.
    /// Default: 0.999.
    pub percentile: f64,
    /// Decay factor for exponential moving average calibration.
    pub ema_decay: f64,
    // Internal per-channel (or per-tensor) stats.
    stats: Vec<CalibStats>,
    // Number of channels observed.  0 = not yet initialized.
    n_channels: usize,
}

impl Quantizer {
    /// Create a new `Quantizer` with the given configuration.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            percentile: 0.999,
            ema_decay: 0.9,
            stats: Vec::new(),
            n_channels: 0,
        }
    }

    /// Override the percentile threshold (must be in (0.5, 1.0]).
    pub fn with_percentile(mut self, p: f64) -> Self {
        self.percentile = p;
        self
    }

    /// Override the EMA decay factor (must be in (0.0, 1.0)).
    pub fn with_ema_decay(mut self, d: f64) -> Self {
        self.ema_decay = d;
        self
    }

    /// Feed a calibration batch.  Call multiple times before `calibrate()`.
    ///
    /// In per-tensor mode the entire matrix is treated as one channel.
    /// In per-channel mode each row is treated as a separate channel.
    pub fn collect_calibration_data<F>(&mut self, data: ArrayView2<F>)
    where
        F: Float + ToPrimitive + Debug,
    {
        let (nrows, ncols) = (data.nrows(), data.ncols());

        if self.config.per_channel {
            if self.n_channels == 0 {
                self.n_channels = nrows;
                self.stats = vec![CalibStats::new(); nrows];
            }
            for r in 0..nrows {
                let row: Vec<f64> = (0..ncols)
                    .map(|c| {
                        data[[r, c]]
                            .to_f64()
                            .unwrap_or(0.0)
                    })
                    .collect();
                self.stats[r].update(&row, self.config.calibration, self.ema_decay);
            }
        } else {
            if self.n_channels == 0 {
                self.n_channels = 1;
                self.stats = vec![CalibStats::new()];
            }
            let flat: Vec<f64> = data
                .iter()
                .map(|v| v.to_f64().unwrap_or(0.0))
                .collect();
            self.stats[0].update(&flat, self.config.calibration, self.ema_decay);
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn bits_range_signed(bits: u8) -> (i32, i32) {
        let qmin = -(1i32 << (bits - 1));
        let qmax = (1i32 << (bits - 1)) - 1;
        (qmin, qmax)
    }

    fn bits_range_unsigned(bits: u8) -> (i32, i32) {
        (0, (1i32 << bits) - 1)
    }

    /// Compute `QuantParams` from collected calibration data.
    ///
    /// Returns an error if `collect_calibration_data` has never been called.
    pub fn calibrate(&mut self) -> Result<QuantParams> {
        if self.n_channels == 0 {
            return Err(NeuralError::InvalidState(
                "No calibration data collected yet; call collect_calibration_data first"
                    .to_string(),
            ));
        }

        let n = self.stats.len();
        let mut scales = Vec::with_capacity(n);
        let mut zero_points = Vec::with_capacity(n);

        for stat in &mut self.stats {
            let (rmin, rmax) =
                stat.effective_range(self.config.calibration, self.percentile);

            let (scale, zp) = if self.config.symmetric {
                let (qmin, qmax) = Self::bits_range_signed(self.config.bits);
                let abs_max = rmin.abs().max(rmax.abs());
                let scale = if abs_max == 0.0 {
                    1.0
                } else {
                    abs_max / (qmax as f64)
                };
                (scale, 0i32)
            } else {
                // Asymmetric: maps [rmin, rmax] → [qmin, qmax]
                let (qmin, qmax) = Self::bits_range_unsigned(self.config.bits);
                let scale = if (rmax - rmin).abs() < 1e-12 {
                    1.0
                } else {
                    (rmax - rmin) / (qmax - qmin) as f64
                };
                let zp = (qmin as f64 - rmin / scale).round() as i32;
                let zp = zp.clamp(qmin, qmax);
                (scale, zp)
            };

            scales.push(scale);
            zero_points.push(zp);
        }

        Ok(QuantParams {
            scales,
            zero_points,
            bits: self.config.bits,
            symmetric: self.config.symmetric,
        })
    }

    // ── Quantize helpers (per-element) ────────────────────────────────────

    #[inline]
    fn quantize_val_i8(v: f64, scale: f64, zp: i32, qmin: i32, qmax: i32) -> i8 {
        let q = (v / scale).round() as i32 + zp;
        q.clamp(qmin, qmax) as i8
    }

    #[inline]
    fn quantize_val_i16(v: f64, scale: f64, zp: i32, qmin: i32, qmax: i32) -> i16 {
        let q = (v / scale).round() as i32 + zp;
        q.clamp(qmin, qmax) as i16
    }

    // ── Public quantize / dequantize ──────────────────────────────────────

    /// Quantize a weight matrix to `i8`.
    ///
    /// Calibration must have been performed (either via `calibrate()` directly, or
    /// by calling `collect_calibration_data` then this method, which will auto-calibrate).
    pub fn quantize_i8<F>(&mut self, weights: ArrayView2<F>) -> Result<(Array2<i8>, QuantParams)>
    where
        F: Float + ToPrimitive + Debug,
    {
        let params = self.calibrate()?;
        let (nrows, ncols) = (weights.nrows(), weights.ncols());

        let (qmin, qmax) = if params.symmetric {
            Self::bits_range_signed(params.bits)
        } else {
            let (lo, hi) = Self::bits_range_unsigned(params.bits);
            (lo, hi)
        };

        let mut out = Array2::<i8>::zeros((nrows, ncols));

        for r in 0..nrows {
            let ch = if self.config.per_channel { r } else { 0 };
            let scale = params.scale_for(ch);
            let zp = params.zero_point_for(ch);
            for c in 0..ncols {
                let v = weights[[r, c]].to_f64().unwrap_or(0.0);
                out[[r, c]] = Self::quantize_val_i8(v, scale, zp, qmin, qmax);
            }
        }

        Ok((out, params))
    }

    /// Quantize a weight matrix to `i16`.
    pub fn quantize_i16<F>(
        &mut self,
        weights: ArrayView2<F>,
    ) -> Result<(Array2<i16>, QuantParams)>
    where
        F: Float + ToPrimitive + Debug,
    {
        let params = self.calibrate()?;
        let (nrows, ncols) = (weights.nrows(), weights.ncols());

        let (qmin, qmax) = if params.symmetric {
            (-(1i32 << 15), (1i32 << 15) - 1)
        } else {
            (0i32, (1i32 << 16) - 1)
        };

        let mut out = Array2::<i16>::zeros((nrows, ncols));

        for r in 0..nrows {
            let ch = if self.config.per_channel { r } else { 0 };
            let scale = params.scale_for(ch);
            let zp = params.zero_point_for(ch);
            for c in 0..ncols {
                let v = weights[[r, c]].to_f64().unwrap_or(0.0);
                out[[r, c]] = Self::quantize_val_i16(v, scale, zp, qmin, qmax);
            }
        }

        Ok((out, params))
    }

    /// Dequantize an `i8` matrix back to `f64`.
    pub fn dequantize_i8(
        &self,
        quantized: ArrayView2<i8>,
        params: &QuantParams,
    ) -> Result<Array2<f64>> {
        let (nrows, ncols) = (quantized.nrows(), quantized.ncols());
        let mut out = Array2::<f64>::zeros((nrows, ncols));

        for r in 0..nrows {
            let ch = if self.config.per_channel { r } else { 0 };
            let scale = params.scale_for(ch);
            let zp = params.zero_point_for(ch);
            for c in 0..ncols {
                let q = quantized[[r, c]] as i32;
                out[[r, c]] = (q - zp) as f64 * scale;
            }
        }

        Ok(out)
    }

    /// Dequantize an `i16` matrix back to `f64`.
    pub fn dequantize_i16(
        &self,
        quantized: ArrayView2<i16>,
        params: &QuantParams,
    ) -> Result<Array2<f64>> {
        let (nrows, ncols) = (quantized.nrows(), quantized.ncols());
        let mut out = Array2::<f64>::zeros((nrows, ncols));

        for r in 0..nrows {
            let ch = if self.config.per_channel { r } else { 0 };
            let scale = params.scale_for(ch);
            let zp = params.zero_point_for(ch);
            for c in 0..ncols {
                let q = quantized[[r, c]] as i32;
                out[[r, c]] = (q - zp) as f64 * scale;
            }
        }

        Ok(out)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Standalone free-function helpers (PTQ)
// ────────────────────────────────────────────────────────────────────────────

/// Quantize a 2-D weight tensor to `i8` using min-max calibration.
///
/// This is a convenience wrapper for one-shot post-training quantization.
pub fn quantize_weights_i8(
    weights: ArrayView2<f64>,
    bits: u8,
    symmetric: bool,
    per_channel: bool,
) -> Result<(Array2<i8>, QuantParams)> {
    let config = QuantizationConfig {
        bits,
        symmetric,
        per_channel,
        calibration: CalibrationMode::MinMax,
    };
    let mut q = Quantizer::new(config);
    q.collect_calibration_data(weights);
    q.quantize_i8(weights)
}

/// Quantize a 2-D weight tensor to `i16` using min-max calibration.
pub fn quantize_weights_i16(
    weights: ArrayView2<f64>,
    bits: u8,
    symmetric: bool,
    per_channel: bool,
) -> Result<(Array2<i16>, QuantParams)> {
    let config = QuantizationConfig {
        bits,
        symmetric,
        per_channel,
        calibration: CalibrationMode::MinMax,
    };
    let mut q = Quantizer::new(config);
    q.collect_calibration_data(weights);
    q.quantize_i16(weights)
}

/// Dequantize a previously quantized `i8` matrix given scale and zero-point.
///
/// This is a low-level helper that does not require a `Quantizer` instance.
pub fn dequantize(
    quantized: ArrayView2<i8>,
    scale: f64,
    zero_point: i32,
) -> Result<Array2<f64>> {
    let (nrows, ncols) = (quantized.nrows(), quantized.ncols());
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for r in 0..nrows {
        for c in 0..ncols {
            let q = quantized[[r, c]] as i32;
            out[[r, c]] = (q - zero_point) as f64 * scale;
        }
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────────
// Quantization-Aware Training (QAT) – fake quantization
// ────────────────────────────────────────────────────────────────────────────

/// Simulates quantization during a forward pass by quantizing then immediately
/// dequantizing (the "straight-through estimator" pattern).
///
/// The struct holds `QuantParams` which are updated by EMA during training.
#[derive(Debug, Clone)]
pub struct QuantizationAwareTraining {
    /// Underlying quantizer (config + calibration stats).
    quantizer: Quantizer,
    /// Cached quantization parameters, updated each forward pass.
    params: Option<QuantParams>,
    /// Number of forward passes performed (used for warmup).
    step: usize,
    /// Number of warmup steps before fake quantization is applied.
    pub warmup_steps: usize,
}

impl QuantizationAwareTraining {
    /// Create a new QAT wrapper with the given config.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            quantizer: Quantizer::new(config),
            params: None,
            step: 0,
            warmup_steps: 0,
        }
    }

    /// Set warmup steps (fake-quantization not applied until step >= warmup_steps).
    pub fn with_warmup(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Apply fake quantization to `weights`.
    ///
    /// During warmup this is a no-op (returns a clone of input).
    /// After warmup, calibration stats are updated then weights are
    /// quantized-and-dequantized in-place (simulating quantization noise).
    pub fn forward(&mut self, weights: ArrayView2<f64>) -> Result<Array2<f64>> {
        self.step += 1;

        if self.step <= self.warmup_steps {
            return Ok(weights.to_owned());
        }

        // Update EMA calibration statistics.
        self.quantizer.collect_calibration_data(weights);

        // Quantize → i8 → dequantize (straight-through estimator in FP).
        let (qi8, params) = self.quantizer.quantize_i8(weights)?;
        let dq = self.quantizer.dequantize_i8(qi8.view(), &params)?;
        self.params = Some(params);
        Ok(dq)
    }

    /// Return a reference to the most recently computed `QuantParams`, if any.
    pub fn current_params(&self) -> Option<&QuantParams> {
        self.params.as_ref()
    }

    /// Return a reference to the inner `Quantizer`.
    pub fn quantizer(&self) -> &Quantizer {
        &self.quantizer
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Post-Training Quantization helpers
// ────────────────────────────────────────────────────────────────────────────

/// Helper bundle for post-training quantization of an entire model's weight list.
#[derive(Debug)]
pub struct PostTrainingQuantizer {
    config: QuantizationConfig,
    /// Percentile for outlier clipping (default 0.999).
    pub percentile: f64,
}

impl PostTrainingQuantizer {
    /// Create a PTQ helper with the given config.
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            percentile: 0.999,
        }
    }

    /// Quantize a list of weight matrices to `i8`.
    ///
    /// Returns a vector of `(Array2<i8>, QuantParams)` in the same order.
    pub fn quantize_all_i8(
        &self,
        weight_matrices: &[ArrayView2<f64>],
    ) -> Result<Vec<(Array2<i8>, QuantParams)>> {
        weight_matrices
            .iter()
            .enumerate()
            .map(|(idx, w)| {
                let mut q = Quantizer::new(self.config.clone());
                q.percentile = self.percentile;
                q.collect_calibration_data(*w);
                q.quantize_i8(*w).map_err(|e| {
                    NeuralError::ComputationError(format!(
                        "PTQ failed for weight matrix {idx}: {e}"
                    ))
                })
            })
            .collect()
    }

    /// Compute quantization error (mean squared error) between original and
    /// dequantized weights for each matrix.
    pub fn quantization_error(
        &self,
        weight_matrices: &[ArrayView2<f64>],
    ) -> Result<Vec<f64>> {
        weight_matrices
            .iter()
            .enumerate()
            .map(|(idx, w)| {
                let mut q = Quantizer::new(self.config.clone());
                q.percentile = self.percentile;
                q.collect_calibration_data(*w);
                let (qi8, params) = q.quantize_i8(*w).map_err(|e| {
                    NeuralError::ComputationError(format!("PTQ error for layer {idx}: {e}"))
                })?;
                let dq = q.dequantize_i8(qi8.view(), &params)?;
                let mse: f64 = w
                    .iter()
                    .zip(dq.iter())
                    .map(|(orig, rec)| (orig - rec).powi(2))
                    .sum::<f64>()
                    / (w.len() as f64);
                Ok(mse)
            })
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sparsity / statistics reporting
// ────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a quantized weight matrix.
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Mean absolute quantization error.
    pub mae: f64,
    /// Root-mean-squared quantization error.
    pub rmse: f64,
    /// Maximum absolute quantization error.
    pub max_error: f64,
    /// Signal-to-quantization-noise ratio (dB).
    pub sqnr_db: f64,
    /// Total number of parameters.
    pub n_params: usize,
    /// Number of unique quantized values (saturation indicator).
    pub n_unique: usize,
}

impl QuantizationStats {
    /// Compute statistics comparing `original` and `recovered` (dequantized) tensors.
    pub fn compute(original: ArrayView2<f64>, recovered: ArrayView2<f64>) -> Result<Self> {
        if original.shape() != recovered.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "original shape {:?} != recovered shape {:?}",
                original.shape(),
                recovered.shape()
            )));
        }

        let n = original.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "Cannot compute quantization stats for empty tensor".to_string(),
            ));
        }

        let mut mae = 0.0f64;
        let mut mse = 0.0f64;
        let mut max_err = 0.0f64;
        let mut signal_power = 0.0f64;

        for (o, r) in original.iter().zip(recovered.iter()) {
            let err = (o - r).abs();
            mae += err;
            mse += err * err;
            if err > max_err {
                max_err = err;
            }
            signal_power += o * o;
        }

        mae /= n as f64;
        mse /= n as f64;
        let rmse = mse.sqrt();

        let noise_power = mse;
        let sqnr_db = if noise_power < 1e-20 {
            f64::INFINITY
        } else {
            10.0 * (signal_power / (n as f64 * noise_power)).log10()
        };

        // Count unique recovered values as a saturation metric.
        let mut sorted_r: Vec<i64> = recovered
            .iter()
            .map(|v| (v * 1e9).round() as i64)
            .collect();
        sorted_r.sort_unstable();
        sorted_r.dedup();
        let n_unique = sorted_r.len();

        Ok(Self {
            mae,
            rmse,
            max_error: max_err,
            sqnr_db,
            n_params: n,
            n_unique,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-channel scale extraction helper
// ────────────────────────────────────────────────────────────────────────────

/// Compute per-channel (per-row) min-max scales for symmetric Int8.
///
/// Returns `(scales, zero_points)` arrays of length `nrows`.
pub fn compute_per_channel_params(
    weights: ArrayView2<f64>,
    bits: u8,
    symmetric: bool,
) -> Result<(Array1<f64>, Array1<i32>)> {
    let nrows = weights.nrows();
    let (_, qmax_signed) = (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1);
    let (qmin_unsigned, qmax_unsigned) = (0i32, (1i32 << bits) - 1);

    let mut scales = Array1::<f64>::zeros(nrows);
    let mut zps = Array1::<i32>::zeros(nrows);

    for r in 0..nrows {
        let row = weights.row(r);
        let rmin = row.iter().cloned().fold(f64::INFINITY, f64::min);
        let rmax = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if symmetric {
            let abs_max = rmin.abs().max(rmax.abs());
            scales[r] = if abs_max == 0.0 {
                1.0
            } else {
                abs_max / qmax_signed as f64
            };
            zps[r] = 0;
        } else {
            let range = rmax - rmin;
            scales[r] = if range.abs() < 1e-12 {
                1.0
            } else {
                range / (qmax_unsigned - qmin_unsigned) as f64
            };
            let zp = (qmin_unsigned as f64 - rmin / scales[r]).round() as i32;
            zps[r] = zp.clamp(qmin_unsigned, qmax_unsigned);
        }
    }

    Ok((scales, zps))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_weights() -> Array2<f64> {
        array![
            [0.1, -0.5, 0.9, -1.0, 0.3],
            [-0.3, 0.7, 0.0, 0.6, -0.8],
            [0.5, 0.5, -0.5, -0.5, 0.1]
        ]
    }

    #[test]
    fn test_symmetric_int8_roundtrip() {
        let w = make_weights();
        let (qi, params) =
            quantize_weights_i8(w.view(), 8, true, false).expect("quantize failed");
        assert_eq!(qi.shape(), w.shape());
        assert_eq!(params.zero_points[0], 0);

        let mut q = Quantizer::new(QuantizationConfig {
            bits: 8,
            symmetric: true,
            per_channel: false,
            calibration: CalibrationMode::MinMax,
        });
        q.collect_calibration_data(w.view());
        let recovered = q.dequantize_i8(qi.view(), &params).expect("dequantize failed");
        let stats =
            QuantizationStats::compute(w.view(), recovered.view()).expect("stats failed");
        // SQNR should be finite and positive for 8-bit quant
        assert!(stats.sqnr_db > 20.0, "sqnr={}", stats.sqnr_db);
    }

    #[test]
    fn test_asymmetric_int8() {
        let w = make_weights();
        let (qi, params) =
            quantize_weights_i8(w.view(), 8, false, false).expect("quantize failed");
        assert_ne!(params.zero_points[0], 0, "asymmetric should have non-zero zp");
        assert_eq!(qi.shape(), w.shape());
    }

    #[test]
    fn test_per_channel_int8() {
        let w = make_weights();
        let (qi, params) =
            quantize_weights_i8(w.view(), 8, true, true).expect("quantize failed");
        assert_eq!(params.scales.len(), 3, "should have one scale per row");
        assert_eq!(qi.shape(), w.shape());
    }

    #[test]
    fn test_int16_roundtrip() {
        let w = make_weights();
        let (qi16, params) =
            quantize_weights_i16(w.view(), 16, true, false).expect("quantize i16 failed");
        assert_eq!(qi16.shape(), w.shape());

        let mut q = Quantizer::new(QuantizationConfig {
            bits: 16,
            symmetric: true,
            per_channel: false,
            calibration: CalibrationMode::MinMax,
        });
        q.collect_calibration_data(w.view());
        let recovered = q.dequantize_i16(qi16.view(), &params).expect("deq i16 failed");
        let stats =
            QuantizationStats::compute(w.view(), recovered.view()).expect("stats failed");
        // Int16 should be very close
        assert!(stats.rmse < 1e-4, "rmse={}", stats.rmse);
    }

    #[test]
    fn test_dequantize_free_function() {
        // Build a simple 2×2 case manually.
        let qi: Array2<i8> = array![[10i8, -10], [5, -5]];
        let scale = 0.1;
        let zp = 0;
        let out = dequantize(qi.view(), scale, zp).expect("dequantize failed");
        assert!((out[[0, 0]] - 1.0).abs() < 1e-9);
        assert!((out[[0, 1]] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_qat_forward_warmup() {
        let mut qat = QuantizationAwareTraining::new(QuantizationConfig::default())
            .with_warmup(2);
        let w = make_weights();
        // During warmup, output should equal input.
        let out1 = qat.forward(w.view()).expect("qat fwd 1");
        assert!((out1[[0, 0]] - w[[0, 0]]).abs() < 1e-12);
        let _out2 = qat.forward(w.view()).expect("qat fwd 2");
        // After warmup, output is fake-quantized.
        let out3 = qat.forward(w.view()).expect("qat fwd 3");
        assert_eq!(out3.shape(), w.shape());
    }

    #[test]
    fn test_ptq_quantize_all() {
        let w1 = make_weights();
        let w2 = make_weights();
        let views: Vec<ArrayView2<f64>> = vec![w1.view(), w2.view()];
        let ptq = PostTrainingQuantizer::new(QuantizationConfig::default());
        let results = ptq.quantize_all_i8(&views).expect("ptq all failed");
        assert_eq!(results.len(), 2);

        let errors = ptq.quantization_error(&views).expect("ptq errors failed");
        assert_eq!(errors.len(), 2);
        for err in &errors {
            assert!(*err < 1.0, "mse too large: {err}");
        }
    }

    #[test]
    fn test_per_channel_params() {
        let w = make_weights();
        let (scales, zps) =
            compute_per_channel_params(w.view(), 8, true).expect("per-channel failed");
        assert_eq!(scales.len(), 3);
        assert_eq!(zps.len(), 3);
        for &s in scales.iter() {
            assert!(s > 0.0, "scale must be positive");
        }
        for &z in zps.iter() {
            assert_eq!(z, 0, "symmetric → zero_point == 0");
        }
    }

    #[test]
    fn test_no_calibration_error() {
        let config = QuantizationConfig::default();
        let mut q = Quantizer::new(config);
        let w = make_weights();
        let result = q.quantize_i8(w.view());
        // Should fail because no calibration data was collected.
        assert!(result.is_err());
    }
}
