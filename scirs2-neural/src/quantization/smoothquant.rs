//! SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs.
//!
//! Implements the SmoothQuant algorithm from Xiao et al. 2022:
//! "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models".
//!
//! # Core idea
//!
//! Activation tensors in Transformer layers tend to have *per-channel* outliers
//! that are orders of magnitude larger than the mean, making uniform INT8
//! activation quantisation lossy.  SmoothQuant migrates the quantisation
//! difficulty from activations to weights using a mathematically equivalent
//! transform:
//!
//! ```text
//! Y = X W^T
//!   = (X diag(s)^{-1}) (diag(s) W^T)
//!   = X_smooth W_smooth^T
//! ```
//!
//! where the per-channel smoothing scale is:
//!
//! ```text
//! s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)
//! ```
//!
//! With `alpha = 0.5` the quantisation difficulty is evenly split between
//! activations and weights, making both amenable to INT8 quantisation with
//! minimal accuracy loss.
//!
//! # Example
//!
//! ```
//! use scirs2_neural::quantization::smoothquant::{
//!     SmoothQuantConfig, SmoothQuantTransformer, ActivationMaxStats,
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! let config = SmoothQuantConfig::default();
//! let weights = Array2::from_elem((4, 4), 0.5_f64);
//! let mut act_stats = ActivationMaxStats::new(4);
//! let acts = Array2::from_elem((8, 4), 2.0_f64);
//! act_stats.update(&acts);
//!
//! let transformer = SmoothQuantTransformer::new(config);
//! let layer = transformer.transform(&weights, &act_stats).expect("transform ok");
//! let output = layer.forward(&acts).expect("forward ok");
//! assert_eq!(output.shape(), [8, 4]);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2};

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for SmoothQuant.
///
/// The most important hyper-parameter is `alpha`, which controls how much of the
/// quantisation difficulty is migrated from activations to weights.
///
/// - `alpha = 0.0` → difficulty stays entirely in activations (no benefit).
/// - `alpha = 0.5` → equal split (recommended default, as in the paper).
/// - `alpha = 1.0` → all difficulty migrated to weights.
#[derive(Debug, Clone)]
pub struct SmoothQuantConfig {
    /// Migration strength `alpha ∈ [0, 1]`.
    pub alpha: f64,
    /// Number of bits for weight quantisation (informational; used downstream).
    pub weight_bits: u8,
    /// Number of bits for activation quantisation (informational; used downstream).
    pub activation_bits: u8,
}

impl Default for SmoothQuantConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            weight_bits: 8,
            activation_bits: 8,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Activation maximum statistics
// ──────────────────────────────────────────────────────────────────────────────

/// Per-channel maximum absolute activation value, accumulated from calibration batches.
///
/// Each call to [`update`](ActivationMaxStats::update) element-wise maximises the
/// running `channel_maxabs` array.  This is the statistic required by the SmoothQuant
/// scale formula.
#[derive(Debug, Clone)]
pub struct ActivationMaxStats {
    /// Running per-channel maximum of |x| (shape `[in_channels]`).
    pub channel_maxabs: Array1<f64>,
    /// Total number of activation *rows* (samples) processed.
    pub num_samples: usize,
}

impl ActivationMaxStats {
    /// Allocate zeroed statistics for `num_channels` input channels.
    pub fn new(num_channels: usize) -> Self {
        Self {
            channel_maxabs: Array1::zeros(num_channels),
            num_samples: 0,
        }
    }

    /// Update per-channel running maxima with a new batch.
    ///
    /// `activations` must have shape `[batch, in_channels]`.
    pub fn update(&mut self, activations: &Array2<f64>) {
        let n_cols = activations.ncols().min(self.channel_maxabs.len());
        for j in 0..n_cols {
            for i in 0..activations.nrows() {
                let v = activations[[i, j]].abs();
                if v > self.channel_maxabs[j] {
                    self.channel_maxabs[j] = v;
                }
            }
        }
        self.num_samples += activations.nrows();
    }

    /// Return the number of channels tracked.
    pub fn num_channels(&self) -> usize {
        self.channel_maxabs.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Scale computation
// ──────────────────────────────────────────────────────────────────────────────

/// Compute per-channel smoothing scales according to the SmoothQuant formula.
///
/// For input channel `j`:
///
/// ```text
/// s_j = max|X_j|^alpha / max|W_j|^(1 - alpha)
/// ```
///
/// where `max|W_j|` is the maximum absolute value in column `j` of the weight
/// matrix `weights` (shape `[out_features, in_features]`).
///
/// Values are clamped to `[eps, f64::MAX]` to prevent division by zero.
pub fn compute_smooth_scales(
    activation_stats: &ActivationMaxStats,
    weights: &Array2<f64>,
    config: &SmoothQuantConfig,
) -> Array1<f64> {
    let n_in = weights.ncols();
    let n_channels = activation_stats.num_channels().min(n_in);
    let eps = 1e-8_f64;
    let alpha = config.alpha.max(0.0).min(1.0);

    let mut scales = Array1::<f64>::ones(n_in);

    for j in 0..n_channels {
        // Per-channel max |activation|
        let act_max = activation_stats.channel_maxabs[j].max(eps);

        // Per-column max |weight|
        let weight_max = weights
            .column(j)
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(eps);

        let s = act_max.powf(alpha) / weight_max.powf(1.0 - alpha);
        scales[j] = s.max(eps);
    }

    scales
}

// ──────────────────────────────────────────────────────────────────────────────
// Smooth transform
// ──────────────────────────────────────────────────────────────────────────────

/// Apply the SmoothQuant channel-wise transform to a weight matrix.
///
/// Given per-channel scales `s` (length `in_features`):
///
/// - **Activation scale** returned is `1/s_j` — multiply input activations by this
///   before INT8 quantisation so that `x_smooth = x / s`.
/// - **Smoothed weights** are `W_smooth[:, j] = W[:, j] * s_j`.
///
/// The product `x_smooth @ W_smooth^T = (x/s) @ (W * s)^T = x @ W^T` preserves the
/// linear operator.
///
/// Returns `(activation_scale, smoothed_weights)`.
pub fn apply_smooth_transform(
    weights: &Array2<f64>,
    scales: &Array1<f64>,
) -> (Array1<f64>, Array2<f64>) {
    let (n_out, n_in) = (weights.nrows(), weights.ncols());
    let eps = 1e-8_f64;

    // activation_scale[j] = 1 / s_j  (applied to input at runtime)
    let n_scale = scales.len().min(n_in);
    let mut activation_scale = Array1::<f64>::ones(n_in);
    for j in 0..n_scale {
        activation_scale[j] = 1.0 / (scales[j] + eps);
    }

    // smoothed_weights[:, j] = weights[:, j] * s_j
    let mut smoothed = Array2::<f64>::zeros((n_out, n_in));
    for i in 0..n_out {
        for j in 0..n_in {
            let s_j = if j < n_scale { scales[j] } else { 1.0 };
            smoothed[[i, j]] = weights[[i, j]] * s_j;
        }
    }

    (activation_scale, smoothed)
}

// ──────────────────────────────────────────────────────────────────────────────
// Smoothed layer
// ──────────────────────────────────────────────────────────────────────────────

/// A SmoothQuant-transformed linear layer.
///
/// The layer stores:
/// - `activation_scale`: element-wise scale to apply to input activations at runtime
///   (`x_smooth = x * activation_scale`, column-wise broadcast).
/// - `smoothed_weights`: the weight matrix after column-wise scaling by `s`.
///
/// After the transform both operands have a smoother distribution and are
/// easier to quantise to INT8 with low error.
#[derive(Debug, Clone)]
pub struct SmoothQuantLayer {
    /// Per-channel scale for input activations: `x_smooth_j = x_j * activation_scale_j`.
    /// Shape: `[in_features]`.
    pub activation_scale: Array1<f64>,
    /// Smoothed weight matrix of shape `[out_features, in_features]`.
    pub smoothed_weights: Array2<f64>,
    /// Configuration used when creating this layer.
    pub config: SmoothQuantConfig,
}

impl SmoothQuantLayer {
    /// Verify the mathematical invariant `Y = X W^T ≈ X_smooth W_smooth^T`.
    ///
    /// Computes both `test_input @ original_weights^T` and
    /// `(test_input * activation_scale) @ smoothed_weights^T` and checks that
    /// the Frobenius difference is small relative to the output magnitude.
    pub fn verify_invariant(
        &self,
        original_weights: &Array2<f64>,
        test_input: &Array2<f64>,
    ) -> bool {
        let n_in = original_weights.ncols();
        if test_input.ncols() != n_in {
            return false;
        }

        // Reference output: X @ W^T
        let y_ref = test_input.dot(&original_weights.t().to_owned());

        // Smoothed output: (X * act_scale) @ W_smooth^T
        let mut x_smooth = test_input.clone();
        for j in 0..n_in.min(self.activation_scale.len()) {
            let s = self.activation_scale[j];
            let mut col = x_smooth.column_mut(j);
            col.mapv_inplace(|v| v * s);
        }
        let y_smooth = x_smooth.dot(&self.smoothed_weights.t().to_owned());

        // Frobenius difference
        let diff_frob: f64 = y_ref
            .iter()
            .zip(y_smooth.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let ref_frob: f64 = y_ref.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();

        let tol = (ref_frob * 1e-8).max(1e-8);
        diff_frob < tol
    }

    /// Forward pass: smooth input activations, then compute `X_smooth @ W_smooth^T`.
    ///
    /// `input` must have shape `[batch, in_features]`.
    /// Returns output of shape `[batch, out_features]`.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let n_in = self.smoothed_weights.ncols();
        if input.ncols() != n_in {
            return Err(NeuralError::ShapeMismatch(format!(
                "SmoothQuant forward: input has {} features, layer expects {}",
                input.ncols(),
                n_in
            )));
        }

        // Apply per-channel activation scaling
        let mut x_smooth = input.clone();
        for j in 0..n_in.min(self.activation_scale.len()) {
            let s = self.activation_scale[j];
            let mut col = x_smooth.column_mut(j);
            col.mapv_inplace(|v| v * s);
        }

        // output = x_smooth @ smoothed_weights^T
        Ok(x_smooth.dot(&self.smoothed_weights.t().to_owned()))
    }

    /// Return the number of input features this layer expects.
    pub fn in_features(&self) -> usize {
        self.smoothed_weights.ncols()
    }

    /// Return the number of output features this layer produces.
    pub fn out_features(&self) -> usize {
        self.smoothed_weights.nrows()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Transformer
// ──────────────────────────────────────────────────────────────────────────────

/// Main SmoothQuant transformer.
///
/// Computes smoothing scales from calibration statistics and applies the
/// channel-wise transform to produce a [`SmoothQuantLayer`] ready for
/// downstream INT8 (or other) quantisation.
pub struct SmoothQuantTransformer {
    config: SmoothQuantConfig,
}

impl SmoothQuantTransformer {
    /// Create a new transformer with the given configuration.
    pub fn new(config: SmoothQuantConfig) -> Self {
        Self { config }
    }

    /// Transform `weights` using the provided activation statistics.
    ///
    /// Steps:
    /// 1. Compute per-channel scales `s` via [`compute_smooth_scales`].
    /// 2. Apply the smooth transform via [`apply_smooth_transform`].
    /// 3. Return a [`SmoothQuantLayer`].
    ///
    /// `weights` has shape `[out_features, in_features]`.
    /// `activation_stats` must have at least `in_features` channels.
    pub fn transform(
        &self,
        weights: &Array2<f64>,
        activation_stats: &ActivationMaxStats,
    ) -> Result<SmoothQuantLayer> {
        let n_in = weights.ncols();
        if activation_stats.num_channels() < n_in {
            return Err(NeuralError::ShapeMismatch(format!(
                "SmoothQuant transform: activation stats have {} channels, weights have {} input features",
                activation_stats.num_channels(),
                n_in
            )));
        }

        let scales = compute_smooth_scales(activation_stats, weights, &self.config);
        let (activation_scale, smoothed_weights) = apply_smooth_transform(weights, &scales);

        Ok(SmoothQuantLayer {
            activation_scale,
            smoothed_weights,
            config: self.config.clone(),
        })
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &SmoothQuantConfig {
        &self.config
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
    fn test_smooth_scales_formula() {
        // Manual calculation with alpha = 0.5:
        //   act_max[0] = 4.0, weight_max[0] = 2.0  → s_0 = 4^0.5 / 2^0.5 = 2/√2 = √2
        //   act_max[1] = 1.0, weight_max[1] = 1.0  → s_1 = 1.0
        let config = SmoothQuantConfig {
            alpha: 0.5,
            ..SmoothQuantConfig::default()
        };

        let mut stats = ActivationMaxStats::new(2);
        let mut acts = Array2::<f64>::zeros((2, 2));
        acts[[0, 0]] = 4.0;
        acts[[1, 0]] = 3.0; // max abs for channel 0 → 4.0
        acts[[0, 1]] = 1.0;
        stats.update(&acts);

        // weights col 0: [2.0, 1.0] → max = 2.0; col 1: [1.0, 0.5] → max = 1.0
        let mut weights = Array2::<f64>::zeros((2, 2));
        weights[[0, 0]] = 2.0;
        weights[[1, 0]] = 1.0;
        weights[[0, 1]] = 1.0;
        weights[[1, 1]] = 0.5;

        let scales = compute_smooth_scales(&stats, &weights, &config);

        let expected_s0 = 4.0_f64.powf(0.5) / 2.0_f64.powf(0.5); // √2
        let expected_s1 = 1.0_f64.powf(0.5) / 1.0_f64.powf(0.5); // 1.0
        assert!(
            (scales[0] - expected_s0).abs() < 1e-6,
            "s_0 expected {expected_s0:.6}, got {:.6}",
            scales[0]
        );
        assert!(
            (scales[1] - expected_s1).abs() < 1e-6,
            "s_1 expected {expected_s1:.6}, got {:.6}",
            scales[1]
        );
    }

    #[test]
    fn test_smooth_transform_invariant() {
        // Y = X W^T must equal X_smooth W_smooth^T (up to floating-point precision)
        let n_in = 4usize;
        let n_out = 3usize;
        let batch = 5usize;

        let config = SmoothQuantConfig::default();

        // Non-trivial weights
        let mut weights = Array2::<f64>::zeros((n_out, n_in));
        for i in 0..n_out {
            for j in 0..n_in {
                weights[[i, j]] = (i as f64 + 1.0) * (j as f64 + 1.0) * 0.1;
            }
        }

        // Non-uniform activation stats
        let mut stats = ActivationMaxStats::new(n_in);
        let mut acts = Array2::<f64>::zeros((batch, n_in));
        for i in 0..batch {
            for j in 0..n_in {
                acts[[i, j]] = (j as f64 + 1.0) * 2.0;
            }
        }
        stats.update(&acts);

        let transformer = SmoothQuantTransformer::new(config);
        let layer = transformer.transform(&weights, &stats).expect("transform ok");

        // Verify invariant
        assert!(
            layer.verify_invariant(&weights, &acts),
            "Mathematical invariant Y = X W^T = X_smooth W_smooth^T violated"
        );
    }

    #[test]
    fn test_smoothquant_alpha_zero() {
        // alpha = 0 → s_j = act_max^0 / weight_max^1 = 1 / weight_max
        let config = SmoothQuantConfig {
            alpha: 0.0,
            ..SmoothQuantConfig::default()
        };

        let n_in = 3usize;
        let weights = {
            let mut w = Array2::<f64>::zeros((2, n_in));
            w[[0, 0]] = 4.0;
            w[[1, 0]] = 3.0; // weight_max[0] = 4.0
            w[[0, 1]] = 2.0; // weight_max[1] = 2.0
            w[[0, 2]] = 1.0; // weight_max[2] = 1.0
            w
        };

        let mut stats = ActivationMaxStats::new(n_in);
        let acts = make_activations(2, n_in, 5.0_f64);
        stats.update(&acts);

        let scales = compute_smooth_scales(&stats, &weights, &config);

        // s_0 = act^0 / weight_max^1 = 1 / 4.0 = 0.25
        let expected_s0 = 1.0 / 4.0_f64;
        assert!(
            (scales[0] - expected_s0).abs() < 1e-7,
            "alpha=0, s_0 expected {expected_s0:.6}, got {:.6}",
            scales[0]
        );
    }

    #[test]
    fn test_smoothquant_alpha_one() {
        // alpha = 1 → s_j = act_max / weight_max^0 = act_max
        let config = SmoothQuantConfig {
            alpha: 1.0,
            ..SmoothQuantConfig::default()
        };

        let n_in = 2usize;
        let weights = make_weights(2, n_in, 0.5_f64);
        let mut stats = ActivationMaxStats::new(n_in);
        let mut acts = Array2::<f64>::zeros((1, n_in));
        acts[[0, 0]] = 3.0;
        acts[[0, 1]] = 7.0;
        stats.update(&acts);

        let scales = compute_smooth_scales(&stats, &weights, &config);

        // s_j = act_max_j^1 / weight_max_j^0 = act_max_j / 1 = act_max_j
        assert!(
            (scales[0] - 3.0).abs() < 1e-7,
            "alpha=1, s_0 expected 3.0, got {}",
            scales[0]
        );
        assert!(
            (scales[1] - 7.0).abs() < 1e-7,
            "alpha=1, s_1 expected 7.0, got {}",
            scales[1]
        );
    }

    #[test]
    fn test_activation_stats_update_max() {
        let mut stats = ActivationMaxStats::new(3);
        // First batch: channel 0 max = 5, channel 1 max = 2, channel 2 max = 1
        let mut acts1 = Array2::<f64>::zeros((2, 3));
        acts1[[0, 0]] = 5.0;
        acts1[[0, 1]] = 2.0;
        acts1[[1, 2]] = 1.0;
        stats.update(&acts1);

        // Second batch: channel 0 max = 3 (lower), channel 1 max = 10 (higher)
        let mut acts2 = Array2::<f64>::zeros((1, 3));
        acts2[[0, 0]] = 3.0;
        acts2[[0, 1]] = 10.0;
        stats.update(&acts2);

        assert!(
            (stats.channel_maxabs[0] - 5.0).abs() < 1e-10,
            "channel 0 max should be 5.0, got {}",
            stats.channel_maxabs[0]
        );
        assert!(
            (stats.channel_maxabs[1] - 10.0).abs() < 1e-10,
            "channel 1 max should be 10.0, got {}",
            stats.channel_maxabs[1]
        );
        assert!(
            (stats.channel_maxabs[2] - 1.0).abs() < 1e-10,
            "channel 2 max should be 1.0, got {}",
            stats.channel_maxabs[2]
        );
        assert_eq!(stats.num_samples, 3);
    }

    #[test]
    fn test_smoothquant_layer_forward_shape() {
        let config = SmoothQuantConfig::default();
        let n_in = 6usize;
        let n_out = 4usize;
        let batch = 7usize;

        let weights = make_weights(n_out, n_in, 0.2_f64);
        let mut stats = ActivationMaxStats::new(n_in);
        let acts = make_activations(batch, n_in, 1.0_f64);
        stats.update(&acts);

        let transformer = SmoothQuantTransformer::new(config);
        let layer = transformer.transform(&weights, &stats).expect("transform ok");

        let output = layer.forward(&acts).expect("forward ok");
        assert_eq!(
            output.shape(),
            [batch, n_out],
            "output shape mismatch: got {:?}",
            output.shape()
        );
    }

    #[test]
    fn test_zero_activation_stats() {
        // All-zero activations should not panic; scales should be clamped to eps.
        let config = SmoothQuantConfig::default();
        let n_in = 4usize;
        let weights = make_weights(2, n_in, 1.0_f64);
        let mut stats = ActivationMaxStats::new(n_in);
        let zero_acts = Array2::<f64>::zeros((4, n_in));
        stats.update(&zero_acts); // channel_maxabs stays at 0

        let scales = compute_smooth_scales(&stats, &weights, &config);
        for j in 0..n_in {
            assert!(
                scales[j] > 0.0,
                "scale[{j}] = {} should be positive even for zero activations",
                scales[j]
            );
        }

        // Transform should succeed without error
        let transformer = SmoothQuantTransformer::new(config);
        let result = transformer.transform(&weights, &stats);
        assert!(result.is_ok(), "transform with zero activations should succeed");
    }

    #[test]
    fn test_smoothquant_default_config() {
        let config = SmoothQuantConfig::default();
        assert!(
            (config.alpha - 0.5).abs() < 1e-10,
            "default alpha should be 0.5"
        );
        assert_eq!(config.weight_bits, 8);
        assert_eq!(config.activation_bits, 8);
    }

    #[test]
    fn test_smoothquant_layer_forward_wrong_shape_returns_error() {
        let config = SmoothQuantConfig::default();
        let weights = make_weights(4, 8, 0.1_f64);
        let mut stats = ActivationMaxStats::new(8);
        stats.update(&make_activations(4, 8, 1.0_f64));

        let transformer = SmoothQuantTransformer::new(config);
        let layer = transformer.transform(&weights, &stats).expect("transform ok");

        let bad_input = Array2::from_elem((2, 16), 1.0_f64);
        assert!(
            layer.forward(&bad_input).is_err(),
            "forward with mismatched feature dim should return Err"
        );
    }
}
