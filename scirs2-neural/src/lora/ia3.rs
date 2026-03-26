//! IA³: Infused Adapter by Inhibiting and Amplifying Inner Activations.
//!
//! IA³ achieves extreme parameter efficiency by learning a single element-wise
//! scaling vector `l ∈ ℝ^d` per adapted component.  At inference:
//!
//! ```text
//! output = l ⊙ activation
//! ```
//!
//! Only one vector per layer needs to be stored and trained, giving orders-of-
//! magnitude fewer parameters than LoRA at the cost of less expressiveness.
//!
//! # References
//!
//! - Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper
//!   than In-Context Learning", NeurIPS 2022

use scirs2_core::ndarray::{Array1, Array2};

use crate::{NeuralError, Result};

// ──────────────────────────── Config ────────────────────────────────────────

/// Configuration for IA³ adapters.
///
/// Flags control which projection types carry a scaling vector; only the
/// enabled ones will be adapted.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::ia3::Ia3Config;
///
/// let cfg = Ia3Config::default();
/// assert!(cfg.scale_keys);
/// ```
#[derive(Debug, Clone)]
pub struct Ia3Config {
    /// Whether to scale key projections.
    pub scale_keys: bool,
    /// Whether to scale value projections.
    pub scale_values: bool,
    /// Whether to scale feed-forward intermediate activations.
    pub scale_ffn: bool,
}

impl Default for Ia3Config {
    fn default() -> Self {
        Self {
            scale_keys: true,
            scale_values: true,
            scale_ffn: true,
        }
    }
}

// ──────────────────────────── Adapter ───────────────────────────────────────

/// IA³ adapter: a single learnable scaling vector `l ∈ ℝ^dim`.
///
/// Initialised to all-ones (identity transform at init).
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::ia3::{Ia3Adapter, Ia3Config};
/// use scirs2_core::ndarray::Array1;
///
/// let adapter = Ia3Adapter::new(8, Ia3Config::default());
/// let x = Array1::from_elem(8, 2.0_f64);
/// let y = adapter.forward(&x).expect("forward");
/// // scale initialised to ones → y == x
/// assert_eq!(y, x);
/// ```
pub struct Ia3Adapter {
    /// Learnable scaling vector [dim].
    pub scale: Array1<f64>,
    /// Adapter configuration (stored for reference / serialisation).
    config: Ia3Config,
    /// Dimension of the scaling vector.
    dim: usize,
}

impl Ia3Adapter {
    /// Create a new IA³ adapter for a given activation dimension.
    ///
    /// `scale` is initialised to ones so the adapter is a no-op at init.
    pub fn new(dim: usize, config: Ia3Config) -> Self {
        Self {
            scale: Array1::ones(dim),
            config,
            dim,
        }
    }

    /// Apply the scaling to a single activation vector `x ∈ ℝ^dim`.
    ///
    /// `output[i] = scale[i] · x[i]`
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `input.len() != dim`.
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        if input.len() != self.dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "IA³ adapter expects dimension {}, got {}",
                self.dim,
                input.len()
            )));
        }
        Ok(&self.scale * input)
    }

    /// Apply the scaling to a batch of activations `x ∈ ℝ^[batch × dim]`.
    ///
    /// Each row is scaled element-wise: `output[b, i] = scale[i] · x[b, i]`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `input.ncols() != dim`.
    pub fn forward_batch(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        if input.ncols() != self.dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "IA³ adapter expects input width {}, got {}",
                self.dim,
                input.ncols()
            )));
        }
        let mut out = input.clone();
        for mut row in out.rows_mut() {
            for (x, &s) in row.iter_mut().zip(self.scale.iter()) {
                *x *= s;
            }
        }
        Ok(out)
    }

    /// Merge the IA³ scaling into a weight matrix by scaling each **row**.
    ///
    /// This implements `W' = diag(scale) · W`, which is suitable for merging
    /// into key/value projection matrices where each row corresponds to one
    /// output dimension.
    ///
    /// The weight must have `out_features` rows matching `scale.len()`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `weight.nrows() != dim`.
    pub fn merge_into_weight_rows(&self, weight: &Array2<f64>) -> Result<Array2<f64>> {
        if weight.nrows() != self.dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "IA³ merge_into_weight_rows: weight has {} rows, scale has {} elements",
                weight.nrows(),
                self.dim
            )));
        }
        let mut out = weight.clone();
        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            let s = self.scale[i];
            row.mapv_inplace(|v| v * s);
        }
        Ok(out)
    }

    /// Merge the IA³ scaling into a weight matrix by scaling each **column**.
    ///
    /// This implements `W' = W · diag(scale)`, useful for FFN output projections
    /// where the scale is applied to the intermediate-activation dimension.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `weight.ncols() != dim`.
    pub fn merge_into_weight_cols(&self, weight: &Array2<f64>) -> Result<Array2<f64>> {
        if weight.ncols() != self.dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "IA³ merge_into_weight_cols: weight has {} cols, scale has {} elements",
                weight.ncols(),
                self.dim
            )));
        }
        let mut out = weight.clone();
        for mut row in out.rows_mut() {
            for (x, &s) in row.iter_mut().zip(self.scale.iter()) {
                *x *= s;
            }
        }
        Ok(out)
    }

    /// Number of trainable parameters (just the scale vector length).
    pub fn n_params(&self) -> usize {
        self.dim
    }

    /// Dimension this adapter was created for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &Ia3Config {
        &self.config
    }
}

// ──────────────────────────── tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn ia3_forward_shape() {
        let adapter = Ia3Adapter::new(8, Ia3Config::default());
        let x = Array1::from_elem(8, 1.0_f64);
        let y = adapter.forward(&x).expect("forward");
        assert_eq!(y.len(), 8);
    }

    #[test]
    fn ia3_ones_identity() {
        let adapter = Ia3Adapter::new(6, Ia3Config::default());
        // scale = ones → output must equal input
        let x = Array1::from_shape_fn(6, |i| (i + 1) as f64);
        let y = adapter.forward(&x).expect("forward");
        for (a, b) in y.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1e-14, "identity broken: {a} != {b}");
        }
    }

    #[test]
    fn ia3_scaling_correct() {
        let mut adapter = Ia3Adapter::new(4, Ia3Config::default());
        // Set scale[i] = (i+1) as f64
        for i in 0..4 {
            adapter.scale[i] = (i + 1) as f64;
        }
        let x = Array1::ones(4);
        let y = adapter.forward(&x).expect("forward");
        for i in 0..4 {
            let expected = (i + 1) as f64;
            assert!(
                (y[i] - expected).abs() < 1e-14,
                "scale mismatch at {i}: expected {expected}, got {}",
                y[i]
            );
        }
    }

    #[test]
    fn ia3_wrong_dim_returns_error() {
        let adapter = Ia3Adapter::new(4, Ia3Config::default());
        let x = Array1::ones(5); // wrong length
        assert!(adapter.forward(&x).is_err());
    }

    #[test]
    fn ia3_merge_into_weight_rows() {
        let mut adapter = Ia3Adapter::new(3, Ia3Config::default());
        adapter.scale = Array1::from_vec(vec![2.0, 3.0, 4.0]);

        let w = Array2::from_shape_fn((3, 4), |(_, j)| (j + 1) as f64);
        let merged = adapter.merge_into_weight_rows(&w).expect("merge");

        // Row 0 scaled by 2, row 1 by 3, row 2 by 4.
        for j in 0..4 {
            assert!((merged[[0, j]] - 2.0 * w[[0, j]]).abs() < 1e-14);
            assert!((merged[[1, j]] - 3.0 * w[[1, j]]).abs() < 1e-14);
            assert!((merged[[2, j]] - 4.0 * w[[2, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn ia3_batch_forward() {
        let mut adapter = Ia3Adapter::new(4, Ia3Config::default());
        adapter.scale = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let input = Array2::ones((3, 4));
        let out = adapter.forward_batch(&input).expect("batch forward");
        assert_eq!(out.shape(), &[3, 4]);

        // Every row must equal [1, 2, 3, 4].
        for b in 0..3 {
            for i in 0..4 {
                let expected = (i + 1) as f64;
                assert!(
                    (out[[b, i]] - expected).abs() < 1e-14,
                    "batch [{b},{i}]: expected {expected}, got {}",
                    out[[b, i]]
                );
            }
        }
    }

    #[test]
    fn ia3_n_params() {
        let adapter = Ia3Adapter::new(64, Ia3Config::default());
        assert_eq!(adapter.n_params(), 64);
    }
}
