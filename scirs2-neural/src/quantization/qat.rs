//! Quantization-Aware Training (QAT)
//!
//! Simulates quantization noise during forward passes using fake quantization,
//! enabling models to be trained with quantization in the loop. The straight-
//! through estimator (STE) is used so gradients pass through unmodified.
//!
//! ## Key components
//! - `FakeQuantize` — simulate INT4/INT8/FP8 quantize-dequantize in forward pass
//! - `QatLayer` — wraps a linear layer with per-weight and per-activation fake quant
//! - `CalibrationCollector` — online statistics for choosing scale/zero_point
//! - `QatConfig` — configuration for the QAT pipeline

use crate::error::{Error, Result};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// QuantBits — supported precisions
// ---------------------------------------------------------------------------

/// Precision modes supported by the QAT pipeline.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantBits {
    /// 4-bit signed integer (range [-8, 7])
    Int4,
    /// 8-bit signed integer (range [-128, 127])
    Int8,
    /// 8-bit floating point, E4M3 format (max representable value ≈ 448)
    Fp8,
}

impl Default for QuantBits {
    fn default() -> Self {
        QuantBits::Int8
    }
}

// ---------------------------------------------------------------------------
// QatConfig
// ---------------------------------------------------------------------------

/// Configuration for Quantization-Aware Training.
#[derive(Debug, Clone)]
pub struct QatConfig {
    /// Bit-width to simulate during training
    pub bits: QuantBits,
    /// Use per-channel quantization for weights (one scale per output channel)
    pub per_channel: bool,
    /// Use symmetric quantization (zero_point = 0)
    pub symmetric: bool,
}

impl Default for QatConfig {
    fn default() -> Self {
        Self {
            bits: QuantBits::Int8,
            per_channel: false,
            symmetric: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fake-quantization functions
// ---------------------------------------------------------------------------

/// Simulate INT8 quantization (clamp to [-128, 127]) and dequantize.
///
/// Uses the straight-through estimator: the function is differentiable
/// everywhere because the rounding is absorbed into the scale.
///
/// `q = clamp(round(x / scale) + zero_point, -128, 127)`
/// `x_q = (q - zero_point) * scale`
pub fn quantize_int8(x: f64, scale: f64, zero_point: i32) -> f64 {
    if scale == 0.0 {
        return x;
    }
    let q = (x / scale + zero_point as f64).round();
    let q_clamped = q.clamp(-128.0, 127.0) as i32;
    (q_clamped - zero_point) as f64 * scale
}

/// Simulate INT4 quantization (clamp to [-8, 7]) and dequantize.
pub fn quantize_int4(x: f64, scale: f64, zero_point: i32) -> f64 {
    if scale == 0.0 {
        return x;
    }
    let q = (x / scale + zero_point as f64).round();
    let q_clamped = q.clamp(-8.0, 7.0) as i32;
    (q_clamped - zero_point) as f64 * scale
}

/// Simulate FP8 E4M3 quantization.
///
/// FP8 E4M3: sign=1, exponent=4 bits (bias=7), mantissa=3 bits.
/// Maximum normal value = (1 + (7/8)) * 2^(15-7) = 448.0.
/// Minimum normal positive = 2^(1-7) ≈ 0.015625.
/// Subnormal values: (m/8) * 2^(1-7) where m ∈ 1..7.
pub fn quantize_fp8(x: f64) -> f64 {
    const FP8_MAX: f64 = 448.0;
    const FP8_MIN_NORMAL: f64 = 1.0 / 64.0; // 2^(-6)
    const MANTISSA_BITS: u32 = 3;
    const MANTISSA_LEVELS: f64 = 8.0; // 2^3
    const EXP_BIAS: i32 = 7;
    const EXP_MAX: i32 = 15; // 4-bit exponent all-ones (NaN/Inf reserved in some variants)

    if x == 0.0 || x.is_nan() {
        return 0.0;
    }

    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let abs_x = x.abs().min(FP8_MAX);

    if abs_x < FP8_MIN_NORMAL {
        // Subnormal: quantize to subnormal grid
        // Subnormal = (m / MANTISSA_LEVELS) * 2^(1 - EXP_BIAS)
        let scale = 2.0_f64.powi(1 - EXP_BIAS) / MANTISSA_LEVELS;
        let m = (abs_x / scale).round().clamp(0.0, MANTISSA_LEVELS - 1.0);
        return sign * m * scale;
    }

    // Normal: compute biased exponent and mantissa
    let log2 = abs_x.log2().floor() as i32;
    let biased_exp = (log2 + EXP_BIAS).clamp(1, EXP_MAX);
    let actual_exp = biased_exp - EXP_BIAS;

    // Mantissa: significand in [1, 2)
    let significand = abs_x / 2.0_f64.powi(actual_exp);
    // Quantize to 3-bit mantissa (MANTISSA_LEVELS steps in [1, 2))
    let mantissa_frac = (significand - 1.0) * MANTISSA_LEVELS;
    let mantissa_q = mantissa_frac.round().clamp(0.0, MANTISSA_LEVELS - 1.0);

    let reconstructed = (1.0 + mantissa_q / MANTISSA_LEVELS) * 2.0_f64.powi(actual_exp);
    sign * reconstructed
}

// ---------------------------------------------------------------------------
// CalibrationCollector — online statistics for determining scale/zp
// ---------------------------------------------------------------------------

/// Number of histogram bins used for KL calibration.
const HISTOGRAM_BINS: usize = 256;

/// Collects activation statistics during a calibration forward pass.
///
/// Three scale/zero_point estimation strategies are provided:
/// - `min_max_scale` — full range coverage
/// - `percentile_scale` — remove outliers beyond a percentile
/// - `kl_divergence_scale` — minimize KL between float and quantized distributions
#[derive(Debug, Clone)]
pub struct CalibrationCollector {
    min_val: f64,
    max_val: f64,
    /// Histogram bins covering [min_val, max_val]
    histogram: Vec<f64>,
    /// Total number of elements observed
    n_elements: usize,
    /// All individual values for percentile computation (capped at MAX_SAMPLES)
    samples: Vec<f64>,
}

/// Maximum number of raw samples kept for percentile estimation.
const MAX_SAMPLES: usize = 100_000;

impl Default for CalibrationCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationCollector {
    /// Create a fresh collector.
    pub fn new() -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            histogram: vec![0.0; HISTOGRAM_BINS],
            n_elements: 0,
            samples: Vec::new(),
        }
    }

    /// Feed a batch of activation values to the collector.
    pub fn observe(&mut self, tensor: &[f64]) {
        for &v in tensor {
            if !v.is_finite() {
                continue;
            }
            if v < self.min_val {
                self.min_val = v;
            }
            if v > self.max_val {
                self.max_val = v;
            }
            self.n_elements += 1;

            // Keep bounded raw sample set (reservoir-style: just keep first MAX_SAMPLES)
            if self.samples.len() < MAX_SAMPLES {
                self.samples.push(v);
            }
        }
        // Rebuild histogram over updated range
        self.rebuild_histogram();
    }

    /// Rebuild the histogram from raw samples.
    fn rebuild_histogram(&mut self) {
        if self.samples.is_empty() || self.min_val >= self.max_val {
            return;
        }
        self.histogram = vec![0.0; HISTOGRAM_BINS];
        let range = self.max_val - self.min_val;
        for &v in &self.samples {
            let bin =
                ((v - self.min_val) / range * HISTOGRAM_BINS as f64) as usize;
            let bin_clamped = bin.min(HISTOGRAM_BINS - 1);
            self.histogram[bin_clamped] += 1.0;
        }
    }

    /// Compute scale and zero_point using min/max range.
    ///
    /// Uses the signed INT8 range [-128, 127] (255 levels).
    /// Returns `(scale, zero_point)`.
    pub fn min_max_scale(&self) -> (f64, i32) {
        if self.n_elements == 0 || self.min_val >= self.max_val {
            return (1.0, 0);
        }
        let range = self.max_val - self.min_val;
        // 255 quantization levels over the full range
        let scale = range / 255.0;
        // zero_point: we want min_val to map to -128
        // q = round(x/scale) + zero_point  => -128 = round(min_val/scale) + zero_point
        // zero_point = -128 - round(min_val/scale)
        let zero_point = (-128_i32).wrapping_sub((self.min_val / scale).round() as i32);
        (scale.max(1e-12), zero_point)
    }

    /// Compute scale and zero_point using a percentile range.
    ///
    /// `p = 99.0` means use the [0.5th, 99.5th] percentile range, removing
    /// the outermost `(100 - p) / 2` percent on each side.
    pub fn percentile_scale(&self, p: f64) -> (f64, i32) {
        if self.samples.is_empty() {
            return (1.0, 0);
        }
        let p_clamped = p.clamp(0.0, 100.0);
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let lower_frac = (100.0 - p_clamped) / 200.0; // half-tail on each side
        let upper_frac = 1.0 - lower_frac;

        let lower_idx = ((n as f64 * lower_frac).floor() as usize).min(n - 1);
        let upper_idx = ((n as f64 * upper_frac).ceil() as usize).min(n - 1);

        let lo = sorted[lower_idx];
        let hi = sorted[upper_idx];

        if hi <= lo {
            return (1.0, 0);
        }
        let range = hi - lo;
        let scale = range / 255.0;
        // Signed INT8: map lo → -128
        let zero_point = (-128_i32).wrapping_sub((lo / scale).round() as i32);
        (scale.max(1e-12), zero_point)
    }

    /// Compute scale and zero_point by minimizing KL divergence between the
    /// original float histogram and the quantized histogram.
    ///
    /// This is the TensorRT-style method: search over candidate thresholds
    /// within the observed range, quantize the histogram at each threshold,
    /// and pick the threshold that yields the lowest KL divergence.
    pub fn kl_divergence_scale(&self) -> (f64, i32) {
        if self.n_elements == 0 || self.min_val >= self.max_val {
            return (1.0, 0);
        }

        let n_bins = self.histogram.len();
        let total: f64 = self.histogram.iter().sum();
        if total <= 0.0 {
            return (1.0, 0);
        }

        // Normalize original histogram to probability distribution
        let p_orig: Vec<f64> = self.histogram.iter().map(|&c| c / total).collect();

        let mut best_kl = f64::INFINITY;
        let mut best_threshold_bins = n_bins; // number of bins included

        // Try thresholds from 128 bins up to all bins (at least 128 levels for INT8)
        let min_bins = (n_bins / 2).max(1);
        for t in min_bins..=n_bins {
            // Compute quantized histogram at this threshold
            let q_hist = self.quantize_histogram(&p_orig, t, 256);
            let kl = kl_divergence_symmetric(&p_orig[..t], &q_hist[..t]);
            if kl < best_kl {
                best_kl = kl;
                best_threshold_bins = t;
            }
        }

        // Convert best threshold (in bins) back to float range
        let range = self.max_val - self.min_val;
        let threshold = self.min_val + range * best_threshold_bins as f64 / n_bins as f64;
        let clipped_min = self.min_val;
        let clipped_max = threshold;

        if clipped_max <= clipped_min {
            return (1.0, 0);
        }
        let scale = (clipped_max - clipped_min) / 255.0;
        // Signed INT8: map clipped_min → -128
        let zero_point = (-128_i32).wrapping_sub((clipped_min / scale).round() as i32);
        (scale.max(1e-12), zero_point)
    }

    /// Quantize a float histogram `p` (length `threshold_bins`) into
    /// `n_quant_levels` levels and expand back to `threshold_bins`.
    fn quantize_histogram(&self, p: &[f64], threshold_bins: usize, n_levels: usize) -> Vec<f64> {
        let mut q = vec![0.0_f64; threshold_bins];
        if threshold_bins == 0 || n_levels == 0 {
            return q;
        }

        // Map each original bin to a quantized level
        let bins_per_level = threshold_bins as f64 / n_levels as f64;

        // Accumulate original probability into quantized levels
        let mut level_mass = vec![0.0_f64; n_levels];
        let mut level_bin_count = vec![0_usize; n_levels];
        for bin in 0..threshold_bins {
            let level = ((bin as f64 / bins_per_level) as usize).min(n_levels - 1);
            level_mass[level] += p[bin];
            level_bin_count[level] += 1;
        }

        // Spread quantized mass back uniformly across bins in each level
        for level in 0..n_levels {
            if level_bin_count[level] == 0 {
                continue;
            }
            let mass_per_bin = level_mass[level] / level_bin_count[level] as f64;
            let start_bin =
                (level as f64 * bins_per_level).floor() as usize;
            let end_bin =
                ((level + 1) as f64 * bins_per_level).ceil() as usize;
            for bin in start_bin..end_bin.min(threshold_bins) {
                q[bin] = mass_per_bin;
            }
        }
        q
    }
}

/// Symmetric KL divergence: 0.5 * (KL(P||Q) + KL(Q||P)).
fn kl_divergence_symmetric(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1e-12;
    let len = p.len().min(q.len());
    let mut kl_pq = 0.0_f64;
    let mut kl_qp = 0.0_f64;
    for i in 0..len {
        let pi = p[i] + eps;
        let qi = q[i] + eps;
        kl_pq += pi * (pi / qi).ln();
        kl_qp += qi * (qi / pi).ln();
    }
    0.5 * (kl_pq + kl_qp)
}

// ---------------------------------------------------------------------------
// QatLayer — linear layer with fake quantization on weights + activations
// ---------------------------------------------------------------------------

/// A linear layer that applies fake quantization during training.
///
/// During `forward_train`, weights and activations are quantized-then-dequantized
/// ("fake quantized") so the model learns to be robust to quantization error.
/// During `forward_infer`, fully-integer arithmetic is used (simulated here
/// with integer arrays for portability).
#[derive(Debug, Clone)]
pub struct QatLayer {
    /// Weight matrix, shape [out_features, in_features], row-major
    pub weights: Vec<f64>,
    /// Bias vector, shape [out_features]
    pub bias: Vec<f64>,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
    /// Scale for weight quantization
    pub weight_scale: f64,
    /// Zero point for weight quantization
    pub weight_zero_point: i32,
    /// Scale for activation quantization
    pub activation_scale: f64,
    /// Zero point for activation quantization
    pub activation_zero_point: i32,
    /// QAT configuration
    pub config: QatConfig,
}

impl QatLayer {
    /// Create a new QAT layer with small deterministic weights (LCG-initialized).
    pub fn new(in_features: usize, out_features: usize, config: QatConfig) -> Self {
        let n_weights = in_features * out_features;
        // LCG-based pseudo-random initialization (Xavier-like scale)
        let mut weights = Vec::with_capacity(n_weights);
        let scale = (6.0 / (in_features + out_features) as f64).sqrt();
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
        for _ in 0..n_weights {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let uniform = (state >> 33) as f64 / (u32::MAX as f64);
            weights.push((uniform * 2.0 - 1.0) * scale);
        }
        let bias = vec![0.0_f64; out_features];

        Self {
            weights,
            bias,
            in_features,
            out_features,
            weight_scale: 1.0 / 127.0,
            weight_zero_point: 0,
            activation_scale: 1.0 / 127.0,
            activation_zero_point: 0,
            config,
        }
    }

    /// Calibrate quantization parameters from observed weights and activations.
    pub fn calibrate_from_data(&mut self, activations: &[f64]) {
        // Calibrate weight parameters
        let mut w_collector = CalibrationCollector::new();
        w_collector.observe(&self.weights);
        let (ws, wzp) = w_collector.min_max_scale();
        self.weight_scale = ws;
        self.weight_zero_point = wzp;

        // Calibrate activation parameters
        if !activations.is_empty() {
            let mut a_collector = CalibrationCollector::new();
            a_collector.observe(activations);
            let (as_, azp) = a_collector.min_max_scale();
            self.activation_scale = as_;
            self.activation_zero_point = azp;
        }
    }

    /// Apply fake quantization to a weight value.
    fn fake_quant_weight(&self, w: f64) -> f64 {
        match self.config.bits {
            QuantBits::Int4 => quantize_int4(w, self.weight_scale, self.weight_zero_point),
            QuantBits::Fp8 => quantize_fp8(w),
            // Int8 and any future variants default to INT8 behaviour
            _ => quantize_int8(w, self.weight_scale, self.weight_zero_point),
        }
    }

    /// Apply fake quantization to an activation value.
    fn fake_quant_activation(&self, a: f64) -> f64 {
        match self.config.bits {
            QuantBits::Int4 => {
                quantize_int4(a, self.activation_scale, self.activation_zero_point)
            }
            QuantBits::Fp8 => quantize_fp8(a),
            // Int8 and any future variants default to INT8 behaviour
            _ => quantize_int8(a, self.activation_scale, self.activation_zero_point),
        }
    }

    /// Forward pass for training: fake-quantize weights + activations.
    ///
    /// Input: flat slice of shape [in_features] (single sample).
    /// Output: Vec<f64> of shape [out_features].
    pub fn forward_train(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(Error::InvalidArgument(format!(
                "Expected {} inputs, got {}",
                self.in_features,
                x.len()
            )));
        }

        // Fake-quantize activations
        let x_q: Vec<f64> = x.iter().map(|&v| self.fake_quant_activation(v)).collect();

        // Fake-quantize weights and compute linear
        let mut out = vec![0.0_f64; self.out_features];
        for o in 0..self.out_features {
            let mut acc = self.bias[o];
            for i in 0..self.in_features {
                let w_q = self.fake_quant_weight(self.weights[o * self.in_features + i]);
                acc += w_q * x_q[i];
            }
            out[o] = acc;
        }
        Ok(out)
    }

    /// Forward pass for inference: fully quantized integer arithmetic.
    ///
    /// Weights are quantized to i32, inputs to i32, output dequantized.
    pub fn forward_infer(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.in_features {
            return Err(Error::InvalidArgument(format!(
                "Expected {} inputs, got {}",
                self.in_features,
                x.len()
            )));
        }

        // Quantize inputs to integer
        let x_int: Vec<i32> = x
            .iter()
            .map(|&v| {
                let q = (v / self.activation_scale + self.activation_zero_point as f64).round();
                q.clamp(-128.0, 127.0) as i32
            })
            .collect();

        // Quantize weights to integer
        let w_int: Vec<i32> = self
            .weights
            .iter()
            .map(|&w| {
                let q = (w / self.weight_scale + self.weight_zero_point as f64).round();
                q.clamp(-128.0, 127.0) as i32
            })
            .collect();

        // Integer accumulation → dequantize output
        let output_scale = self.activation_scale * self.weight_scale;
        let mut out = vec![0.0_f64; self.out_features];
        for o in 0..self.out_features {
            let mut acc: i64 = 0;
            for i in 0..self.in_features {
                acc += w_int[o * self.in_features + i] as i64 * x_int[i] as i64;
            }
            // Dequantize: subtract zero-points' contribution and scale
            let zp_correction = self.weight_zero_point as i64
                * x_int.iter().map(|&v| v as i64).sum::<i64>()
                + self.activation_zero_point as i64
                    * w_int[o * self.in_features..(o + 1) * self.in_features]
                        .iter()
                        .map(|&v| v as i64)
                        .sum::<i64>()
                - self.weight_zero_point as i64
                    * self.activation_zero_point as i64
                    * self.in_features as i64;
            let acc_corrected = acc - zp_correction;
            out[o] = acc_corrected as f64 * output_scale + self.bias[o];
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Whole-model quantization utilities
// ---------------------------------------------------------------------------

/// Quantize an f64 weight vector to i8 using the configuration.
///
/// Returns the quantized weights as i8, plus the (scale, zero_point) used.
pub fn quantize_model_weights(
    weights: &[f64],
    config: &QatConfig,
) -> (Vec<i8>, f64, i32) {
    if weights.is_empty() {
        return (Vec::new(), 1.0, 0);
    }
    let mut collector = CalibrationCollector::new();
    collector.observe(weights);
    let (scale, zero_point) = collector.min_max_scale();

    let quantized: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let q = match config.bits {
                QuantBits::Int4 => {
                    let q_raw = (w / scale + zero_point as f64).round();
                    q_raw.clamp(-8.0, 7.0) as i32
                }
                // Int8, Fp8, and any future variants use INT8 range
                _ => {
                    let q_raw = (w / scale + zero_point as f64).round();
                    q_raw.clamp(-128.0, 127.0) as i32
                }
            };
            q as i8
        })
        .collect();

    (quantized, scale, zero_point)
}

/// Dequantize i8 weights back to f64.
pub fn dequantize_weights(weights: &[i8], scale: f64, zero_point: i32) -> Vec<f64> {
    weights
        .iter()
        .map(|&q| (q as i32 - zero_point) as f64 * scale)
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-layer calibration helpers
// ---------------------------------------------------------------------------

/// Calibration method for determining scale/zero_point.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibMethod {
    /// Use global min/max range
    MinMax,
    /// Clip at a percentile to remove outliers
    Percentile,
    /// Minimize KL divergence between float and quantized distributions
    KL,
}

/// Per-layer quantization specification produced by calibration.
#[derive(Debug, Clone)]
pub struct QuantizationSpec {
    /// Layer name
    pub layer_name: String,
    /// Calibrated scale
    pub scale: f64,
    /// Calibrated zero point
    pub zero_point: i32,
    /// Calibration method used
    pub method: CalibMethod,
}

/// A small representative calibration dataset (rows of activations).
#[derive(Debug, Clone, Default)]
pub struct CalibrationDataset {
    /// Each entry is one batch/row of activations
    pub data: Vec<Vec<f64>>,
}

impl CalibrationDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Add a sample row.
    pub fn push(&mut self, row: Vec<f64>) {
        self.data.push(row);
    }
}

/// Calibrate a single layer's activations using the chosen method.
pub fn calibrate_layer(
    layer_outputs: &[Vec<f64>],
    method: CalibMethod,
) -> (f64, i32) {
    let mut collector = CalibrationCollector::new();
    for batch in layer_outputs {
        collector.observe(batch);
    }
    match method {
        CalibMethod::Percentile => collector.percentile_scale(99.0),
        CalibMethod::KL => collector.kl_divergence_scale(),
        // MinMax and any future variants use min_max
        _ => collector.min_max_scale(),
    }
}

/// Calibrate all layers and return per-layer quantization specs.
///
/// `layer_activations`: slice of `(layer_name, vec_of_batches)` tuples.
pub fn calibrate_model(
    layer_activations: &[(&str, Vec<Vec<f64>>)],
) -> Vec<QuantizationSpec> {
    layer_activations
        .iter()
        .map(|(name, batches)| {
            let (scale, zero_point) = calibrate_layer(batches, CalibMethod::KL);
            QuantizationSpec {
                layer_name: name.to_string(),
                scale,
                zero_point,
                method: CalibMethod::KL,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_int8_range() {
        let scale = 1.0;
        // Values far outside [-128, 127] after dividing by scale
        let large = 10_000.0;
        let small = -10_000.0;
        let q_large = quantize_int8(large, scale, 0);
        let q_small = quantize_int8(small, scale, 0);
        // Dequantized values must be within the INT8 representable range
        assert!(
            q_large <= 127.0 * scale,
            "dequantized large: {q_large}"
        );
        assert!(
            q_small >= -128.0 * scale,
            "dequantized small: {q_small}"
        );
    }

    #[test]
    fn test_quantize_int8_roundtrip() {
        let scale = 0.1;
        let val = 3.14;
        let q = quantize_int8(val, scale, 0);
        // Error should be at most half a scale step
        assert!((q - val).abs() <= scale * 0.5 + 1e-9, "roundtrip error: {}", (q - val).abs());
    }

    #[test]
    fn test_quantize_int4_range() {
        let scale = 1.0;
        let large = 1_000.0;
        let small = -1_000.0;
        let q_large = quantize_int4(large, scale, 0);
        let q_small = quantize_int4(small, scale, 0);
        assert!(q_large <= 7.0, "dequantized: {q_large}");
        assert!(q_small >= -8.0, "dequantized: {q_small}");
    }

    #[test]
    fn test_fp8_e4m3_range() {
        let large = 1_000_000.0;
        let q = quantize_fp8(large);
        assert!(q <= 448.0, "fp8 should be clamped to 448, got {q}");
        let neg = -1_000_000.0;
        let qn = quantize_fp8(neg);
        assert!(qn >= -448.0, "fp8 negative should be clamped to -448, got {qn}");
    }

    #[test]
    fn test_fp8_zero() {
        assert_eq!(quantize_fp8(0.0), 0.0);
    }

    #[test]
    fn test_fake_quantize_train() {
        let config = QatConfig::default();
        let mut layer = QatLayer::new(4, 2, config);
        // Calibrate with small activations
        let acts: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        layer.calibrate_from_data(&acts);
        layer.activation_scale = 0.01;
        layer.weight_scale = 0.01;

        let input = vec![0.1, 0.2, 0.3, 0.4];
        let out = layer.forward_train(&input).expect("forward_train failed");
        assert_eq!(out.len(), 2);
        // Output must be finite
        for v in &out {
            assert!(v.is_finite(), "output not finite: {v}");
        }
    }

    #[test]
    fn test_calibration_min_max() {
        let mut collector = CalibrationCollector::new();
        let data: Vec<f64> = (-100..=100).map(|i| i as f64).collect();
        collector.observe(&data);
        let (scale, zp) = collector.min_max_scale();
        assert!(scale > 0.0, "scale must be positive");
        // The scale * 255 should approximately cover the original range
        let covered = scale * 255.0;
        assert!(covered >= 199.0, "scale should cover [−100, 100]: covered {covered}");
        let _ = zp; // zero_point used for completeness
    }

    #[test]
    fn test_calibration_percentile() {
        let mut collector = CalibrationCollector::new();
        // 1000 values in [0, 1] plus one large outlier
        let mut data: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        data.push(1_000_000.0);
        collector.observe(&data);
        let (scale, _zp) = collector.percentile_scale(99.0);
        // The scale should be much smaller than the outlier would dictate
        assert!(scale < 100.0, "scale should not be dominated by outlier: {scale}");
    }

    #[test]
    fn test_kl_calibration() {
        let mut collector = CalibrationCollector::new();
        let data: Vec<f64> = (0..500).map(|i| i as f64 * 0.01).collect();
        collector.observe(&data);
        let (scale, _zp) = collector.kl_divergence_scale();
        assert!(scale > 0.0, "scale must be positive");
        // zero_point is signed (can be negative); just verify scale is valid
    }

    #[test]
    fn test_qat_layer_forward() {
        let config = QatConfig::default();
        let layer = QatLayer::new(8, 4, config);
        let input = vec![1.0; 8];
        let out = layer.forward_train(&input).expect("forward_train failed");
        assert_eq!(out.len(), 4, "output length should be out_features");
    }

    #[test]
    fn test_quantize_dequantize_weights() {
        let config = QatConfig::default();
        // Use positive-only weights to ensure asymmetric zero_point stays within i8 range
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let (q, scale, zp) = quantize_model_weights(&weights, &config);
        let dq = dequantize_weights(&q, scale, zp);
        assert_eq!(dq.len(), weights.len());
        // Quantization error is bounded by one quantization step (scale)
        for (orig, rec) in weights.iter().zip(dq.iter()) {
            assert!(
                (orig - rec).abs() < scale + 1e-9,
                "roundtrip error too large: orig={orig}, rec={rec}, scale={scale}"
            );
        }
    }

    #[test]
    fn test_qat_config_default() {
        let cfg = QatConfig::default();
        assert_eq!(cfg.bits, QuantBits::Int8);
        assert!(!cfg.per_channel);
        assert!(cfg.symmetric);
    }

    #[test]
    fn test_calibrate_model() {
        let layer1_batches: Vec<Vec<f64>> = vec![
            (0..32).map(|i| i as f64 * 0.1).collect(),
            (0..32).map(|i| -(i as f64) * 0.05).collect(),
        ];
        let layer2_batches: Vec<Vec<f64>> = vec![
            (0..32).map(|i| (i as f64).sin()).collect(),
        ];
        let activations: Vec<(&str, Vec<Vec<f64>>)> = vec![
            ("layer1", layer1_batches),
            ("layer2", layer2_batches),
        ];
        let specs = calibrate_model(&activations);
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].layer_name, "layer1");
        assert_eq!(specs[1].layer_name, "layer2");
        for spec in &specs {
            assert!(spec.scale > 0.0);
        }
    }
}
