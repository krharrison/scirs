//! DoRA: Weight-Decomposed Low-Rank Adaptation.
//!
//! Better accuracy than LoRA at the same parameter budget by separately
//! learning magnitude (`m`) and direction (`W/||W||`) of weight updates.
//!
//! The decomposition follows Liu et al. 2024:
//! `W_eff = m * (W_0 + BA) / ||W_0 + BA||_col`
//!
//! where `m` is a learned per-output-channel magnitude vector, and
//! the direction is normalised to unit column norm before scaling.
//!
//! # References
//!
//! - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", 2024

use scirs2_core::ndarray::{Array1, Array2};

use crate::{NeuralError, Result};

// ──────────────────────────── Config ────────────────────────────────────────

/// Configuration for a DoRA-adapted linear layer.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::dora::DoraConfig;
///
/// let cfg = DoraConfig::default();
/// assert_eq!(cfg.rank, 4);
/// ```
#[derive(Debug, Clone)]
pub struct DoraConfig {
    /// Low-rank dimension `r` (must satisfy `0 < r ≤ min(out, in)`).
    pub rank: usize,
    /// Scaling factor α; effective scale = α/r.
    pub alpha: f64,
    /// Epsilon added to column norms to avoid division by zero.
    pub column_norm_eps: f64,
}

impl Default for DoraConfig {
    fn default() -> Self {
        Self {
            rank: 4,
            alpha: 1.0,
            column_norm_eps: 1e-8,
        }
    }
}

// ──────────────────────────── Layer ─────────────────────────────────────────

/// A DoRA-adapted linear layer.
///
/// Trainable parameters: `lora_a` [r × in], `lora_b` [out × r], `magnitude` [out].
/// Frozen parameters: `weight` [out × in].
///
/// Forward: `y = (m * (W_0 + s·BA) / ||(W_0 + s·BA)||_col) · x^T`
///
/// where `s = α/r` and column norms are computed row-wise (each row of W
/// corresponds to one output channel / column of W^T).
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::dora::{DoraLinear, DoraConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::<f64>::eye(6);
/// let cfg = DoraConfig { rank: 2, ..Default::default() };
/// let layer = DoraLinear::new(weight, &cfg).expect("create DoraLinear");
/// let input = Array2::<f64>::ones((1, 6));
/// let out = layer.forward(&input).expect("forward");
/// assert_eq!(out.shape(), &[1, 6]);
/// ```
pub struct DoraLinear {
    /// Frozen base weight [out × in].
    weight: Array2<f64>,
    /// LoRA A matrix [r × in].
    pub lora_a: Array2<f64>,
    /// LoRA B matrix [out × r].
    pub lora_b: Array2<f64>,
    /// Per-output-channel magnitude scalar [out].
    pub magnitude: Array1<f64>,
    /// Precomputed scaling α/r.
    scaling: f64,
    config: DoraConfig,
}

impl DoraLinear {
    /// Create a new DoRA layer from a frozen weight matrix.
    ///
    /// - `lora_b` initialised to zeros (zero LoRA delta at init).
    /// - `lora_a` initialised to small constant 0.02 (overwrite before training).
    /// - `magnitude` initialised to per-row L2-norm of `weight`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidArgument`] if `rank == 0` or
    /// `rank > min(out_features, in_features)`.
    pub fn new(weight: Array2<f64>, config: &DoraConfig) -> Result<Self> {
        let (out_features, in_features) = (weight.nrows(), weight.ncols());

        if config.rank == 0 {
            return Err(NeuralError::InvalidArgument(
                "DoRA rank must be > 0".to_string(),
            ));
        }
        if config.rank > in_features.min(out_features) {
            return Err(NeuralError::InvalidArgument(format!(
                "DoRA rank {} is invalid for weight [{out_features}×{in_features}]: \
                 rank must be ≤ min(out, in) = {}",
                config.rank,
                in_features.min(out_features)
            )));
        }

        let scaling = config.alpha / config.rank as f64;

        // lora_b = 0 → zero contribution at initialisation.
        let lora_b = Array2::zeros((out_features, config.rank));
        // lora_a initialised with small constant; caller should replace with
        // Kaiming / Gaussian init before training.
        let lora_a = Array2::from_elem((config.rank, in_features), 0.02);

        // Magnitude = per-row L2 norm of base weight (clamped to eps).
        let magnitude = Array1::from_shape_fn(out_features, |i| {
            row_l2_norm(weight.row(i).iter().copied()).max(config.column_norm_eps)
        });

        Ok(Self {
            weight,
            lora_a,
            lora_b,
            magnitude,
            scaling,
            config: config.clone(),
        })
    }

    /// Compute the effective weight after DoRA decomposition.
    ///
    /// `W_eff[i, :] = magnitude[i] * (W_0 + s·B·A)[i, :] / ||(W_0 + s·B·A)[i, :]||`
    pub fn effective_weight(&self) -> Array2<f64> {
        let delta = self.lora_b.dot(&self.lora_a) * self.scaling;
        let adapted = &self.weight + &delta;

        let mut normalized = adapted;
        for i in 0..normalized.nrows() {
            let row_norm =
                row_l2_norm(normalized.row(i).iter().copied()).max(self.config.column_norm_eps);
            let mag = self.magnitude[i];
            let scale = mag / row_norm;
            normalized.row_mut(i).mapv_inplace(|v| v * scale);
        }
        normalized
    }

    /// Forward pass: `y = x · W_eff^T` where `x` is [batch × in].
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `input.ncols() != in_features`.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let in_features = self.weight.ncols();
        if input.ncols() != in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "DoRA expects {} input features, got {}",
                in_features,
                input.ncols()
            )));
        }
        let w = self.effective_weight();
        Ok(input.dot(&w.t()))
    }

    /// Merge DoRA adaptation into the base weight for inference efficiency.
    ///
    /// After merging:
    /// - `weight` contains the full adapted weight (magnitude × direction).
    /// - `lora_a`, `lora_b` zeroed out.
    /// - `magnitude` reset to new column norms.
    pub fn merge_into_base(&mut self) {
        self.weight = self.effective_weight();

        self.lora_a.fill(0.0);
        self.lora_b.fill(0.0);

        // Reset magnitude to reflect the new base weight.
        for i in 0..self.weight.nrows() {
            let norm =
                row_l2_norm(self.weight.row(i).iter().copied()).max(self.config.column_norm_eps);
            self.magnitude[i] = norm;
        }
    }

    /// Number of trainable parameters.
    ///
    /// = `rank * in_features`  (A)
    /// + `out_features * rank` (B)
    /// + `out_features`        (magnitude)
    pub fn n_trainable_params(&self) -> usize {
        self.lora_a.len() + self.lora_b.len() + self.magnitude.len()
    }

    /// Return the dimension of the layer: `(out_features, in_features)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.weight.nrows(), self.weight.ncols())
    }

    /// Reference to the frozen base weight.
    pub fn weight(&self) -> &Array2<f64> {
        &self.weight
    }
}

// ──────────────────────────── helpers ───────────────────────────────────────

/// Compute the L2 norm of an iterator of f64 values.
fn row_l2_norm(iter: impl Iterator<Item = f64>) -> f64 {
    iter.map(|v| v * v).sum::<f64>().sqrt()
}

// ──────────────────────────── tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_layer(out: usize, in_f: usize, rank: usize) -> DoraLinear {
        let w = Array2::from_shape_fn((out, in_f), |(i, j)| (i * in_f + j) as f64 * 0.1);
        DoraLinear::new(
            w,
            &DoraConfig {
                rank,
                ..Default::default()
            },
        )
        .expect("DoraLinear::new failed")
    }

    #[test]
    fn dora_effective_weight_shape() {
        let layer = make_layer(8, 6, 2);
        let w = layer.effective_weight();
        assert_eq!(w.shape(), &[8, 6]);
    }

    #[test]
    fn dora_zero_adapter_identity() {
        // lora_b is zeros → delta = 0 → adapted = W_0.
        // Each row of effective_weight should be (magnitude[i] / ||W_0[i,:]||) * W_0[i,:].
        let w = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j + 1) as f64);
        let cfg = DoraConfig {
            rank: 2,
            ..Default::default()
        };
        let layer = DoraLinear::new(w.clone(), &cfg).expect("new");
        let eff = layer.effective_weight();

        for i in 0..4 {
            let norm_w = row_l2_norm(w.row(i).iter().copied()).max(cfg.column_norm_eps);
            let mag = layer.magnitude[i];
            for j in 0..4 {
                let expected = w[[i, j]] * mag / norm_w;
                assert!(
                    (eff[[i, j]] - expected).abs() < 1e-10,
                    "row {i} col {j}: expected {expected}, got {}",
                    eff[[i, j]]
                );
            }
        }
    }

    #[test]
    fn dora_magnitude_initialized_correctly() {
        let w = Array2::from_shape_fn((3, 4), |(i, j)| (i + j + 1) as f64);
        let cfg = DoraConfig {
            rank: 2,
            ..Default::default()
        };
        let layer = DoraLinear::new(w.clone(), &cfg).expect("new");
        for i in 0..3 {
            let expected = row_l2_norm(w.row(i).iter().copied()).max(cfg.column_norm_eps);
            assert!(
                (layer.magnitude[i] - expected).abs() < 1e-10,
                "magnitude mismatch at row {i}: expected {expected}, got {}",
                layer.magnitude[i]
            );
        }
    }

    #[test]
    fn dora_forward_output_shape() {
        let layer = make_layer(5, 8, 3);
        let input = Array2::from_elem((4, 8), 1.0);
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[4, 5]);
    }

    #[test]
    fn dora_merge_preserves_forward() {
        let w = Array2::from_shape_fn((4, 6), |(i, j)| (i * 6 + j + 1) as f64 * 0.1);
        let cfg = DoraConfig {
            rank: 2,
            ..Default::default()
        };
        let mut layer = DoraLinear::new(w, &cfg).expect("new");

        // Give lora_b non-zero values.
        layer.lora_b = Array2::from_shape_fn((4, 2), |(i, j)| (i as f64 - j as f64) * 0.01);

        let input = Array2::from_shape_fn((3, 6), |(i, j)| (i * 6 + j) as f64 * 0.05 + 0.1);
        let before = layer.forward(&input).expect("before merge");

        layer.merge_into_base();
        let after = layer.forward(&input).expect("after merge");

        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-9, "merge changed output: {a} vs {b}");
        }
    }

    #[test]
    fn dora_invalid_rank_zero() {
        let w = Array2::<f64>::eye(4);
        let cfg = DoraConfig {
            rank: 0,
            ..Default::default()
        };
        assert!(DoraLinear::new(w, &cfg).is_err());
    }

    #[test]
    fn dora_rank_larger_than_dim() {
        let w = Array2::<f64>::eye(4);
        let cfg = DoraConfig {
            rank: 5,
            ..Default::default()
        };
        assert!(DoraLinear::new(w, &cfg).is_err());
    }

    #[test]
    fn dora_n_params_correct() {
        let out = 8;
        let in_f = 6;
        let rank = 3;
        let layer = make_layer(out, in_f, rank);
        // A: rank*in_f, B: out*rank, magnitude: out
        let expected = rank * in_f + out * rank + out;
        assert_eq!(layer.n_trainable_params(), expected);
    }
}
