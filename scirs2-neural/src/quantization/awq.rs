//! AWQ (Activation-aware Weight Quantization).
//!
//! Implements the AWQ algorithm from Lin et al. 2023:
//! "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration".
//!
//! AWQ uses per-channel activation statistics collected from calibration data to find
//! optimal per-channel scaling factors before INT4 quantization. The key insight is that
//! not all weights are equally important: weights corresponding to high-activation channels
//! are "salient" and should be protected by scaling. By searching for the optimal scale
//! `s_j = norm_j^alpha` where `alpha` minimises quantisation error, AWQ significantly
//! reduces the perplexity of the quantised model compared to naïve INT4 quantisation.
//!
//! # Example
//!
//! ```
//! use scirs2_neural::quantization::awq::{AwqConfig, AwqCalibrator, AwqQuantizer, ActivationStats};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = AwqConfig::default();
//! let weights = Array2::from_elem((8, 8), 0.5_f64);
//! let mut calibrator = AwqCalibrator::new(8, config.clone());
//! let acts = Array2::from_elem((4, 8), 1.0_f64);
//! calibrator.record_activations(&acts);
//! let scales = calibrator.compute_scales();
//! let quantizer = AwqQuantizer::new(config);
//! let stats = calibrator.into_stats();
//! let layer = quantizer.quantize(&weights, &scales).expect("quantise ok");
//! let _out = layer.forward(&Array2::from_elem((2, 8), 1.0)).expect("forward ok");
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for AWQ quantisation.
///
/// Controls the bit-width, group size, and scale-search hyper-parameters.
#[derive(Debug, Clone)]
pub struct AwqConfig {
    /// Number of quantisation bits (typically 4).
    pub bits: u8,
    /// Group size for per-group quantisation (e.g. 128).
    pub group_size: usize,
    /// Number of grid-search steps when finding the optimal per-channel scale.
    /// A larger value gives finer resolution but takes longer to calibrate.
    pub search_steps: usize,
    /// Enable asymmetric quantisation (uses a non-zero zero-point).
    /// When `false` (default) symmetric quantisation is used.
    pub zero_point: bool,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            search_steps: 20,
            zero_point: false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Activation statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Per-channel activation statistics accumulated from calibration data.
///
/// The L2 norm of each input channel is tracked across all calibration batches.
/// After calling [`finalize`](ActivationStats::finalize) the norms are divided by
/// the sample count so they represent the mean per-sample L2 contribution.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Running sum of squared L2 norms per input channel (shape: `[in_channels]`).
    pub channel_norms: Array1<f64>,
    /// Number of individual activation *samples* (rows) seen so far.
    pub num_samples: usize,
    /// `true` once [`finalize`](ActivationStats::finalize) has been called.
    finalized: bool,
}

impl ActivationStats {
    /// Allocate zeroed statistics for `num_channels` input channels.
    pub fn new(num_channels: usize) -> Self {
        Self {
            channel_norms: Array1::zeros(num_channels),
            num_samples: 0,
            finalized: false,
        }
    }

    /// Update running statistics with a new batch of activations.
    ///
    /// `activations` must have shape `[batch, in_channels]`.  The L2 norm of
    /// each column (channel) is accumulated into `channel_norms`.
    pub fn update(&mut self, activations: &Array2<f64>) {
        let n_channels = self.channel_norms.len();
        let n_cols = activations.ncols();
        let cols_to_process = n_channels.min(n_cols);

        for j in 0..cols_to_process {
            let col = activations.column(j);
            let squared_norm: f64 = col.iter().map(|&v| v * v).sum();
            self.channel_norms[j] += squared_norm.sqrt();
        }
        self.num_samples += activations.nrows();
        self.finalized = false;
    }

    /// Normalise accumulated norms by the number of samples.
    ///
    /// Must be called once before the statistics are used for scale computation.
    /// Calling `finalize` a second time is a no-op.
    pub fn finalize(&mut self) {
        if self.finalized || self.num_samples == 0 {
            return;
        }
        let n = self.num_samples as f64;
        self.channel_norms.mapv_inplace(|v| v / n);
        self.finalized = true;
    }

    /// Return the number of input channels tracked.
    pub fn num_channels(&self) -> usize {
        self.channel_norms.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Calibrator
// ──────────────────────────────────────────────────────────────────────────────

/// Helper that records activations and derives per-channel scaling factors.
///
/// Typical usage:
/// 1. Create with [`AwqCalibrator::new`].
/// 2. Feed batches of activations with [`record_activations`](AwqCalibrator::record_activations).
/// 3. Call [`compute_scales`](AwqCalibrator::compute_scales) to obtain the final scales.
/// 4. Pass scales (and weights) to [`AwqQuantizer::quantize`].
pub struct AwqCalibrator {
    stats: ActivationStats,
    config: AwqConfig,
}

impl AwqCalibrator {
    /// Create a new calibrator for `num_channels` input channels.
    pub fn new(num_channels: usize, config: AwqConfig) -> Self {
        Self {
            stats: ActivationStats::new(num_channels),
            config,
        }
    }

    /// Record a batch of activations with shape `[batch, in_channels]`.
    pub fn record_activations(&mut self, activations: &Array2<f64>) {
        self.stats.update(activations);
    }

    /// Compute per-channel scales from the accumulated activation statistics.
    ///
    /// Uses the simple heuristic `s_j = norm_j^{0.5}` (alpha = 0.5 fixed).
    /// For a more flexible search use [`AwqQuantizer::find_optimal_scale`].
    pub fn compute_scales(&self) -> Array1<f64> {
        let mut stats = self.stats.clone();
        stats.finalize();
        let eps = 1e-8_f64;
        stats.channel_norms.mapv(|v| (v + eps).sqrt())
    }

    /// Consume the calibrator and return the finalised [`ActivationStats`].
    pub fn into_stats(mut self) -> ActivationStats {
        self.stats.finalize();
        self.stats
    }

    /// Return a reference to the accumulated stats (not yet finalised).
    pub fn stats(&self) -> &ActivationStats {
        &self.stats
    }

    /// Return a reference to the config.
    pub fn config(&self) -> &AwqConfig {
        &self.config
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Low-level INT4 helpers
// ──────────────────────────────────────────────────────────────────────────────

/// INT4 quantisation range: `[-8, 7]` (signed 4-bit).
const INT4_MIN: i8 = -8;
const INT4_MAX: i8 = 7;

/// Quantise a single `f64` scalar to a 4-bit signed integer.
///
/// - `scale` should be positive; if it is zero or negative the result is 0.
/// - `zero_point` is the offset for asymmetric quantisation (0 for symmetric).
fn quantize_int4(v: f64, scale: f64, zero_point: i8) -> i8 {
    if scale <= 0.0 {
        return 0;
    }
    let q = (v / scale).round() + zero_point as f64;
    q.max(INT4_MIN as f64).min(INT4_MAX as f64) as i8
}

/// Dequantise a 4-bit signed integer back to `f64`.
fn dequantize_int4(q: i8, scale: f64, zero_point: i8) -> f64 {
    (q as f64 - zero_point as f64) * scale
}

/// Pack two 4-bit values into a single `u8`.
///
/// The *lower* nibble holds `lo` (bits 3:0) and the *upper* nibble holds `hi` (bits 7:4).
#[inline]
fn pack_nibbles(lo: i8, hi: i8) -> u8 {
    let lo_u = (lo as u8) & 0x0F;
    let hi_u = ((hi as u8) & 0x0F) << 4;
    lo_u | hi_u
}

/// Unpack the two 4-bit values from a `u8`.
///
/// Returns `(lo, hi)` with sign extension for the 4-bit range.
#[inline]
fn unpack_nibbles(packed: u8) -> (i8, i8) {
    let lo_raw = (packed & 0x0F) as i8;
    let hi_raw = ((packed >> 4) & 0x0F) as i8;
    // Sign-extend from 4 bits
    let lo = if lo_raw >= 8 { lo_raw - 16 } else { lo_raw };
    let hi = if hi_raw >= 8 { hi_raw - 16 } else { hi_raw };
    (lo, hi)
}

// ──────────────────────────────────────────────────────────────────────────────
// Quantised layer
// ──────────────────────────────────────────────────────────────────────────────

/// An AWQ-quantised linear layer.
///
/// Weights are stored in INT4 nibble-packed form.  Per-group scales (and optional
/// zero-points) allow accurate dequantisation at inference time.
#[derive(Debug, Clone)]
pub struct AwqQuantizedLayer {
    /// Nibble-packed weight data.  Two 4-bit weights are stored per byte.
    /// Layout: row-major over `[out_features, in_features]` before packing.
    /// Length = `out_features * ceil(in_features / 2)`.
    pub weight_q: Vec<u8>,
    /// Per-group scales with shape `[out_features, num_groups]`
    /// where `num_groups = ceil(in_features / group_size)`.
    pub scales: Array2<f64>,
    /// Optional per-group zero-points with the same shape as `scales`.
    pub zero_points: Option<Array2<i8>>,
    /// Number of input features (columns of the original weight matrix).
    pub in_features: usize,
    /// Number of output features (rows of the original weight matrix).
    pub out_features: usize,
    /// Configuration used when quantising.
    pub config: AwqConfig,
}

impl AwqQuantizedLayer {
    /// Dequantise the packed INT4 weights back to `f64`.
    ///
    /// Returns a matrix of shape `[out_features, in_features]`.
    pub fn dequantize_weights(&self) -> Array2<f64> {
        let group_size = self.config.group_size;
        let num_groups = (self.in_features + group_size - 1) / group_size;
        let bytes_per_row = (self.in_features + 1) / 2;

        let mut weights = Array2::<f64>::zeros((self.out_features, self.in_features));

        for i in 0..self.out_features {
            for j in 0..self.in_features {
                let byte_idx = i * bytes_per_row + j / 2;
                let packed = if byte_idx < self.weight_q.len() {
                    self.weight_q[byte_idx]
                } else {
                    0u8
                };

                let q_val = if j % 2 == 0 {
                    let (lo, _) = unpack_nibbles(packed);
                    lo
                } else {
                    let (_, hi) = unpack_nibbles(packed);
                    hi
                };

                let group_idx = (j / group_size).min(num_groups - 1);
                let scale = self.scales[[i, group_idx]];
                let zp = self
                    .zero_points
                    .as_ref()
                    .map(|zps| zps[[i, group_idx]])
                    .unwrap_or(0i8);

                weights[[i, j]] = dequantize_int4(q_val, scale, zp);
            }
        }

        weights
    }

    /// Forward pass: dequantise weights then compute `input @ W^T`.
    ///
    /// `input` must have shape `[batch, in_features]`.
    /// Returns a matrix of shape `[batch, out_features]`.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let (batch, n_in) = (input.nrows(), input.ncols());
        if n_in != self.in_features {
            return Err(NeuralError::ShapeMismatch(format!(
                "AWQ forward: input has {} features, layer expects {}",
                n_in, self.in_features
            )));
        }

        let w = self.dequantize_weights(); // [out, in]
        // output = input @ W^T → [batch, out]
        Ok(input.dot(&w.t().to_owned()))
    }

    /// Compression ratio relative to a dense `f64` weight matrix.
    ///
    /// Returns approximately `0.0625` (4 bits / 64 bits).
    pub fn compression_ratio(&self) -> f64 {
        let orig_bytes = self.out_features * self.in_features * std::mem::size_of::<f64>();
        let packed_bytes = self.weight_q.len();
        let scales_bytes = self.scales.len() * std::mem::size_of::<f64>();
        let zp_bytes = self
            .zero_points
            .as_ref()
            .map(|z| z.len())
            .unwrap_or(0);
        let quantised_total = packed_bytes + scales_bytes + zp_bytes;
        if orig_bytes == 0 {
            return 1.0;
        }
        quantised_total as f64 / orig_bytes as f64
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Quantiser
// ──────────────────────────────────────────────────────────────────────────────

/// AWQ weight quantiser.
///
/// Performs scale-search and produces [`AwqQuantizedLayer`] from a dense weight matrix.
pub struct AwqQuantizer {
    config: AwqConfig,
}

impl AwqQuantizer {
    /// Create a new `AwqQuantizer` with the given configuration.
    pub fn new(config: AwqConfig) -> Self {
        Self { config }
    }

    /// Find the optimal per-channel scale that minimises quantisation error.
    ///
    /// The search strategy follows the AWQ paper:
    ///
    /// For each candidate exponent `alpha` in `linspace(0, 1, search_steps)`:
    ///
    /// ```text
    /// s_j = norm_j^alpha
    /// W_scaled[:, j] = W[:, j] / s_j        (scale inputs)
    /// W_q = quantize(W_scaled)
    /// W_dq[:, j] = dequantize(W_q)[:, j] * s_j   (undo scale)
    /// error_alpha = ||W - W_dq||_F
    /// ```
    ///
    /// The `alpha` yielding the smallest Frobenius error is selected.
    ///
    /// `weights` has shape `[out_features, in_features]`.
    /// `activation_stats` must have `in_features` channels.
    pub fn find_optimal_scale(
        &self,
        weights: &Array2<f64>,
        activation_stats: &ActivationStats,
    ) -> Array1<f64> {
        let n_in = weights.ncols();
        let eps = 1e-8_f64;
        let n_channels = activation_stats.num_channels().min(n_in);

        // Use already-finalised norms (or raw if not finalised — safe either way).
        let norms = &activation_stats.channel_norms;

        let mut best_scales = Array1::<f64>::ones(n_in);
        let mut best_error = f64::INFINITY;

        let steps = self.config.search_steps.max(1);

        for step in 0..=steps {
            let alpha = step as f64 / steps as f64;

            // Build candidate per-channel scale: s_j = (norm_j + eps)^alpha
            let mut candidate = Array1::<f64>::ones(n_in);
            for j in 0..n_channels {
                candidate[j] = (norms[j] + eps).powf(alpha).max(eps);
            }

            // Scale weights column-wise: W_scaled[:, j] = W[:, j] / s_j
            let mut w_scaled = weights.clone();
            for j in 0..n_in {
                let s_j = candidate[j];
                let mut col = w_scaled.column_mut(j);
                col.mapv_inplace(|v| v / s_j);
            }

            // Quantise and dequantise the scaled weights using symmetric per-column scale
            let w_dq = self.simulate_quantize_dequantize(&w_scaled);

            // Undo the channel scale to compare with original
            let mut w_reconstructed = w_dq;
            for j in 0..n_in {
                let s_j = candidate[j];
                let mut col = w_reconstructed.column_mut(j);
                col.mapv_inplace(|v| v * s_j);
            }

            // Frobenius error
            let diff = weights - &w_reconstructed;
            let frobenius: f64 = diff.iter().map(|&v| v * v).sum::<f64>().sqrt();

            if frobenius < best_error {
                best_error = frobenius;
                best_scales = candidate;
            }
        }

        best_scales
    }

    /// Simulate quantise-then-dequantise for a weight matrix using the configured bits.
    ///
    /// This uses a naive per-group symmetric scale derived from the weight magnitudes,
    /// which is sufficient for the grid-search loss evaluation inside `find_optimal_scale`.
    fn simulate_quantize_dequantize(&self, weights: &Array2<f64>) -> Array2<f64> {
        let (n_out, n_in) = (weights.nrows(), weights.ncols());
        let group_size = self.config.group_size.max(1);
        let q_max = (1i32 << (self.config.bits as i32 - 1)) - 1;
        let q_max_f = q_max as f64;

        let mut result = Array2::<f64>::zeros((n_out, n_in));

        for i in 0..n_out {
            let mut j = 0usize;
            while j < n_in {
                let end = (j + group_size).min(n_in);
                // Find max absolute value in group
                let abs_max = weights
                    .row(i)
                    .iter()
                    .skip(j)
                    .take(end - j)
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max);

                let scale = if abs_max > 0.0 {
                    abs_max / q_max_f
                } else {
                    1.0
                };

                for jj in j..end {
                    let v = weights[[i, jj]];
                    let q = (v / scale).round().max(-q_max_f - 1.0).min(q_max_f);
                    result[[i, jj]] = q * scale;
                }

                j = end;
            }
        }

        result
    }

    /// Quantise a weight matrix using the provided per-channel scales.
    ///
    /// The `scales` vector has length `in_features`.  Each column `j` of the weight
    /// matrix is divided by `scales[j]` before INT4 quantisation, which protects
    /// high-salience channels by keeping their effective range larger.
    ///
    /// `weights` has shape `[out_features, in_features]`.
    pub fn quantize(
        &self,
        weights: &Array2<f64>,
        scales: &Array1<f64>,
    ) -> Result<AwqQuantizedLayer> {
        let (n_out, n_in) = (weights.nrows(), weights.ncols());
        if scales.len() < n_in {
            return Err(NeuralError::ShapeMismatch(format!(
                "AWQ quantize: scale length {} < in_features {}",
                scales.len(),
                n_in
            )));
        }
        let group_size = self.config.group_size.max(1);
        let num_groups = (n_in + group_size - 1) / group_size;
        let bytes_per_row = (n_in + 1) / 2;
        let eps = 1e-8_f64;

        let q_max = (1i32 << (self.config.bits as i32 - 1)) - 1;
        let q_max_f = q_max as f64;

        let mut weight_q: Vec<u8> = vec![0u8; n_out * bytes_per_row];
        let mut quant_scales = Array2::<f64>::zeros((n_out, num_groups));
        let mut zero_points_arr: Option<Array2<i8>> = if self.config.zero_point {
            Some(Array2::zeros((n_out, num_groups)))
        } else {
            None
        };

        for i in 0..n_out {
            // Pre-scale weights row: w_scaled[j] = w[j] / s[j]
            let mut w_row_scaled: Vec<f64> = (0..n_in)
                .map(|j| weights[[i, j]] / (scales[j] + eps))
                .collect();

            // Quantise group by group
            let mut group_idx = 0usize;
            let mut j = 0usize;
            while j < n_in {
                let end = (j + group_size).min(n_in);
                let group_slice = &w_row_scaled[j..end];

                let abs_max = group_slice
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max);

                let (group_scale, group_zp) = if self.config.zero_point {
                    // Asymmetric: scale = (max - min) / (q_max - q_min)
                    let g_min = group_slice
                        .iter()
                        .cloned()
                        .fold(f64::INFINITY, f64::min);
                    let g_max = group_slice
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let range = g_max - g_min;
                    let s = if range > eps {
                        range / ((q_max_f * 2.0 + 1.0).max(1.0))
                    } else {
                        1.0
                    };
                    let zp_f = -g_min / s - q_max_f - 1.0;
                    let zp = zp_f
                        .round()
                        .max(INT4_MIN as f64)
                        .min(INT4_MAX as f64) as i8;
                    (s, zp)
                } else {
                    // Symmetric
                    let s = if abs_max > eps {
                        abs_max / q_max_f
                    } else {
                        1.0
                    };
                    (s, 0i8)
                };

                quant_scales[[i, group_idx]] = group_scale;
                if let Some(ref mut zps) = zero_points_arr {
                    zps[[i, group_idx]] = group_zp;
                }

                // Pack nibbles
                for jj in j..end {
                    let q = quantize_int4(w_row_scaled[jj], group_scale, group_zp);
                    // Store back to verify packing
                    w_row_scaled[jj] = q as f64;

                    let byte_idx = i * bytes_per_row + jj / 2;
                    if jj % 2 == 0 {
                        // lo nibble — clear lo and set
                        weight_q[byte_idx] = (weight_q[byte_idx] & 0xF0) | ((q as u8) & 0x0F);
                    } else {
                        // hi nibble — clear hi and set
                        weight_q[byte_idx] =
                            (weight_q[byte_idx] & 0x0F) | (((q as u8) & 0x0F) << 4);
                    }
                }

                group_idx += 1;
                j = end;
            }
        }

        Ok(AwqQuantizedLayer {
            weight_q,
            scales: quant_scales,
            zero_points: zero_points_arr,
            in_features: n_in,
            out_features: n_out,
            config: self.config.clone(),
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_weights(out: usize, inp: usize, fill: f64) -> Array2<f64> {
        Array2::from_elem((out, inp), fill)
    }

    fn make_activations(batch: usize, channels: usize, fill: f64) -> Array2<f64> {
        Array2::from_elem((batch, channels), fill)
    }

    // ── tests ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_activation_stats_update() {
        let mut stats = ActivationStats::new(4);
        let acts = make_activations(3, 4, 2.0_f64);
        stats.update(&acts);
        assert_eq!(stats.num_samples, 3);
        // Each column has 3 rows of value 2.0 → L2 norm per row = 2.0, sum = 6.0
        for j in 0..4 {
            assert!(
                stats.channel_norms[j] > 0.0,
                "channel {j} norm should be positive"
            );
        }
    }

    #[test]
    fn test_activation_stats_finalize_normalises() {
        let mut stats = ActivationStats::new(2);
        let acts = make_activations(4, 2, 1.0_f64);
        stats.update(&acts);
        let raw = stats.channel_norms.clone();
        stats.finalize();
        // After finalise, norms are divided by num_samples (4)
        for j in 0..2 {
            let expected = raw[j] / 4.0;
            assert!(
                (stats.channel_norms[j] - expected).abs() < 1e-10,
                "channel {j}: got {}, expected {expected}",
                stats.channel_norms[j]
            );
        }
        // Second call is no-op
        let after_first = stats.channel_norms.clone();
        stats.finalize();
        for j in 0..2 {
            assert!(
                (stats.channel_norms[j] - after_first[j]).abs() < 1e-12,
                "second finalize must be a no-op"
            );
        }
    }

    #[test]
    fn test_find_optimal_scale_identity() {
        // When all activation norms are equal, the optimal scale should be uniform.
        let n_in = 8usize;
        let n_out = 4usize;
        let config = AwqConfig {
            search_steps: 10,
            ..AwqConfig::default()
        };
        let quantizer = AwqQuantizer::new(config);

        let weights = make_weights(n_out, n_in, 0.3);
        let mut stats = ActivationStats::new(n_in);
        // All channels same norm → uniform activations
        let acts = make_activations(8, n_in, 1.0_f64);
        stats.update(&acts);
        stats.finalize();

        let scales = quantizer.find_optimal_scale(&weights, &stats);
        assert_eq!(scales.len(), n_in);

        // All scales should be approximately equal (same activation norm → same alpha result)
        let s0 = scales[0];
        for j in 1..n_in {
            assert!(
                (scales[j] - s0).abs() < 1e-8,
                "scales should be uniform; got {s0} vs {}",
                scales[j]
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let config = AwqConfig::default();
        let n_in = 128usize;
        let n_out = 16usize;

        // Create weights with values spread in [-1, 1]
        let mut weights = Array2::<f64>::zeros((n_out, n_in));
        for i in 0..n_out {
            for j in 0..n_in {
                weights[[i, j]] = ((i * n_in + j) as f64 / (n_out * n_in) as f64) * 2.0 - 1.0;
            }
        }

        let scales = Array1::<f64>::ones(n_in);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");
        let dq = layer.dequantize_weights();

        // Check shape
        assert_eq!(dq.shape(), [n_out, n_in]);

        // Mean relative error should be small (< 10% for INT4)
        let mut total_abs_err = 0.0_f64;
        let mut total_abs_val = 0.0_f64;
        for i in 0..n_out {
            for j in 0..n_in {
                total_abs_err += (weights[[i, j]] - dq[[i, j]]).abs();
                total_abs_val += weights[[i, j]].abs() + 1e-9;
            }
        }
        let rel_err = total_abs_err / total_abs_val;
        assert!(
            rel_err < 0.10,
            "Roundtrip relative error {rel_err:.4} should be < 10%"
        );
    }

    #[test]
    fn test_awq_layer_forward_shape() {
        let config = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let n_in = 8usize;
        let n_out = 6usize;
        let batch = 3usize;

        let weights = make_weights(n_out, n_in, 0.1);
        let scales = Array1::<f64>::ones(n_in);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");

        let input = Array2::from_elem((batch, n_in), 1.0_f64);
        let output = layer.forward(&input).expect("forward ok");
        assert_eq!(output.shape(), [batch, n_out]);
    }

    #[test]
    fn test_compression_ratio_approximately_correct() {
        // For a large enough matrix the weight storage dominates the scale overhead,
        // so the ratio should be well below 1.0 (compressed relative to f64).
        let config = AwqConfig {
            group_size: 128,
            ..AwqConfig::default()
        };
        let weights = make_weights(32, 512, 0.5);
        let scales = Array1::<f64>::ones(512);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");
        let ratio = layer.compression_ratio();
        assert!(
            ratio < 1.0,
            "Compression ratio {ratio:.4} should be < 1.0 (i.e. smaller than f64)"
        );
    }

    #[test]
    fn test_awq_zero_weight_matrix() {
        let config = AwqConfig::default();
        let weights = Array2::<f64>::zeros((4, 8));
        let scales = Array1::<f64>::ones(8);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");
        let dq = layer.dequantize_weights();
        for v in dq.iter() {
            assert!(
                v.abs() < 1e-8,
                "All-zero weights should dequantise to ~0, got {v}"
            );
        }
    }

    #[test]
    fn test_awq_group_size_works() {
        let config = AwqConfig {
            group_size: 64,
            ..AwqConfig::default()
        };
        let n_in = 256usize;
        let n_out = 8usize;
        let mut weights = Array2::<f64>::zeros((n_out, n_in));
        for i in 0..n_out {
            for j in 0..n_in {
                weights[[i, j]] = (j as f64 / n_in as f64) - 0.5;
            }
        }
        let scales = Array1::<f64>::ones(n_in);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");

        // Verify shape of scales: num_groups = 256 / 64 = 4
        assert_eq!(layer.scales.shape(), [n_out, 4]);
        let dq = layer.dequantize_weights();
        assert_eq!(dq.shape(), [n_out, n_in]);
    }

    #[test]
    fn test_awq_config_default() {
        let config = AwqConfig::default();
        assert_eq!(config.bits, 4);
        assert_eq!(config.group_size, 128);
        assert_eq!(config.search_steps, 20);
        assert!(!config.zero_point);
    }

    #[test]
    fn test_awq_calibrator_records_and_computes_scales() {
        let n_in = 8usize;
        let config = AwqConfig::default();
        let mut calibrator = AwqCalibrator::new(n_in, config);
        // First batch: channel 0 has large activations, others small
        let mut acts = Array2::<f64>::zeros((4, n_in));
        for i in 0..4 {
            acts[[i, 0]] = 10.0;
            acts[[i, 1]] = 1.0;
        }
        calibrator.record_activations(&acts);
        let scales = calibrator.compute_scales();
        assert_eq!(scales.len(), n_in);
        // Channel 0 has larger activation norm → larger scale
        assert!(
            scales[0] > scales[1],
            "channel 0 scale ({}) should exceed channel 1 scale ({})",
            scales[0],
            scales[1]
        );
    }

    #[test]
    fn test_awq_forward_wrong_shape_returns_error() {
        let config = AwqConfig {
            group_size: 4,
            ..AwqConfig::default()
        };
        let weights = make_weights(4, 8, 0.1);
        let scales = Array1::<f64>::ones(8);
        let quantizer = AwqQuantizer::new(config);
        let layer = quantizer.quantize(&weights, &scales).expect("quantize ok");
        // Input with wrong feature dimension
        let bad_input = Array2::from_elem((2, 16), 1.0_f64);
        assert!(
            layer.forward(&bad_input).is_err(),
            "forward with mismatched input should return Err"
        );
    }
}
