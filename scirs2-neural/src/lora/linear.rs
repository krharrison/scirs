//! LoRA linear layer implementation.
//!
//! Implements the Low-Rank Adaptation technique where weight updates are decomposed
//! as W_out = W_frozen + (alpha/r) * B @ A, dramatically reducing trainable parameters.

use scirs2_core::ndarray::{Array2, Axis};

use super::types::{LoRAConfig, LoRAStats};
use crate::{NeuralError, Result};

/// A simple xorshift64 PRNG for reproducible initialization.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_1234
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a uniform f64 in [-bound, bound).
    fn next_uniform(&mut self, bound: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * bound
    }
}

/// Compute Kaiming uniform bound: sqrt(6 / fan_in) for ReLU gain=sqrt(2).
/// For LoRA A matrix, we use fan_in = in_features.
fn kaiming_uniform_bound(fan_in: usize) -> f64 {
    // gain for ReLU = sqrt(2), std = gain / sqrt(fan_in), bound = sqrt(3) * std
    let gain = std::f64::consts::SQRT_2;
    let std_dev = gain / (fan_in as f64).sqrt();
    std_dev * 3.0_f64.sqrt()
}

/// LoRA linear layer: W_out = W_frozen + (alpha/r) * B @ A
///
/// The original weight matrix is frozen, and only the low-rank matrices A and B
/// are trainable. Since B is initialized to zeros, the initial output matches
/// the original pretrained model exactly.
///
/// # Architecture
///
/// ```text
///                    ┌──────────────────┐
///   input ──────────>│  W_frozen (frozen)│──────> output_frozen
///     │              └──────────────────┘            │
///     │              ┌──────┐  ┌──────┐              │
///     └─────────────>│ A    │─>│ B    │──> scaled ───┘── output
///                    │(rxin)│  │(outxr)│     (alpha/r)
///                    └──────┘  └──────┘
/// ```
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::{LoRALinear, LoRAConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::<f64>::eye(4);
/// let config = LoRAConfig { rank: 2, ..Default::default() };
/// let lora = LoRALinear::new(weight, &config).expect("failed to create LoRA layer");
///
/// let input = Array2::<f64>::ones((1, 4));
/// let output = lora.forward(&input).expect("forward failed");
/// assert_eq!(output.shape(), &[1, 4]);
/// ```
pub struct LoRALinear {
    /// Frozen original weight matrix [out_features x in_features].
    weight: Array2<f64>,
    /// Low-rank A matrix [rank x in_features], initialized with Kaiming uniform.
    lora_a: Array2<f64>,
    /// Low-rank B matrix [out_features x rank], initialized with zeros.
    lora_b: Array2<f64>,
    /// Low-rank dimension.
    rank: usize,
    /// Scaling alpha.
    alpha: f64,
    /// Precomputed scaling factor: alpha / rank.
    scaling: f64,
    /// Whether the LoRA weights have been merged into the frozen weight.
    merged: bool,
}

impl LoRALinear {
    /// Create a new LoRA linear layer from a pretrained weight matrix.
    ///
    /// The A matrix is initialized with Kaiming uniform distribution,
    /// and the B matrix is initialized to zeros so the initial LoRA
    /// contribution is zero.
    ///
    /// # Arguments
    ///
    /// * `weight` - Original pretrained weight matrix [out_features x in_features]
    /// * `config` - LoRA configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the rank exceeds the weight dimensions or config is invalid.
    pub fn new(weight: Array2<f64>, config: &LoRAConfig) -> Result<Self> {
        config.validate()?;

        let (out_features, in_features) = (weight.nrows(), weight.ncols());

        if config.rank > in_features.min(out_features) {
            return Err(NeuralError::InvalidArgument(format!(
                "LoRA rank {} exceeds min(in_features={}, out_features={})",
                config.rank, in_features, out_features
            )));
        }

        let rank = config.rank;
        let scaling = config.scaling();

        // Initialize A with Kaiming uniform
        let bound = kaiming_uniform_bound(in_features);
        let mut rng = Xorshift64::new(config.seed);
        let lora_a = Array2::from_shape_fn((rank, in_features), |_| rng.next_uniform(bound));

        // Initialize B with zeros
        let lora_b = Array2::zeros((out_features, rank));

        Ok(Self {
            weight,
            lora_a,
            lora_b,
            rank,
            alpha: config.alpha,
            scaling,
            merged: false,
        })
    }

    /// Forward pass: x @ W^T + scaling * x @ A^T @ B^T
    ///
    /// When merged, simply computes x @ W^T (where W already includes LoRA).
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [batch_size x in_features]
    ///
    /// # Errors
    ///
    /// Returns an error on dimension mismatch.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let in_features = self.weight.ncols();
        if input.ncols() != in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "Input has {} features but weight expects {}",
                input.ncols(),
                in_features
            )));
        }

        // x @ W^T
        let output = input.dot(&self.weight.t());

        if self.merged {
            // LoRA is already merged into weight
            return Ok(output);
        }

        // LoRA contribution: scaling * x @ A^T @ B^T
        let lora_output = input.dot(&self.lora_a.t());
        let lora_output = lora_output.dot(&self.lora_b.t());
        let lora_scaled = &lora_output * self.scaling;

        Ok(&output + &lora_scaled)
    }

    /// Merge LoRA weights into the frozen weight: W += scaling * B @ A
    ///
    /// After merging, the forward pass only uses the merged weight,
    /// which is more efficient for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if already merged.
    pub fn merge(&mut self) -> Result<()> {
        if self.merged {
            return Err(NeuralError::InvalidState(
                "LoRA weights are already merged".to_string(),
            ));
        }

        let delta = self.lora_b.dot(&self.lora_a) * self.scaling;
        self.weight = &self.weight + &delta;
        self.merged = true;
        Ok(())
    }

    /// Unmerge LoRA weights from the frozen weight: W -= scaling * B @ A
    ///
    /// Restores the original frozen weight for continued training.
    ///
    /// # Errors
    ///
    /// Returns an error if not currently merged.
    pub fn unmerge(&mut self) -> Result<()> {
        if !self.merged {
            return Err(NeuralError::InvalidState(
                "LoRA weights are not merged".to_string(),
            ));
        }

        let delta = self.lora_b.dot(&self.lora_a) * self.scaling;
        self.weight = &self.weight - &delta;
        self.merged = false;
        Ok(())
    }

    /// Get a reference to the LoRA A matrix [rank x in_features].
    pub fn lora_a(&self) -> &Array2<f64> {
        &self.lora_a
    }

    /// Get a reference to the LoRA B matrix [out_features x rank].
    pub fn lora_b(&self) -> &Array2<f64> {
        &self.lora_b
    }

    /// Get the frozen weight matrix reference.
    pub fn weight(&self) -> &Array2<f64> {
        &self.weight
    }

    /// Get the rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the alpha value.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get the scaling factor (alpha / rank).
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Whether the LoRA weights are currently merged.
    pub fn is_merged(&self) -> bool {
        self.merged
    }

    /// Compute parameter statistics.
    pub fn stats(&self) -> LoRAStats {
        let (out_features, in_features) = (self.weight.nrows(), self.weight.ncols());
        let frozen_params = out_features * in_features;
        let trainable_params = self.rank * in_features + out_features * self.rank;
        let total_params = frozen_params + trainable_params;
        let compression_ratio = if total_params > 0 {
            trainable_params as f64 / total_params as f64
        } else {
            0.0
        };

        LoRAStats {
            total_params,
            trainable_params,
            frozen_params,
            compression_ratio,
        }
    }

    /// Set the LoRA A matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape doesn't match [rank x in_features].
    pub fn set_lora_a(&mut self, a: Array2<f64>) -> Result<()> {
        if a.shape() != self.lora_a.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected A shape {:?}, got {:?}",
                self.lora_a.shape(),
                a.shape()
            )));
        }
        self.lora_a = a;
        Ok(())
    }

    /// Set the LoRA B matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape doesn't match [out_features x rank].
    pub fn set_lora_b(&mut self, b: Array2<f64>) -> Result<()> {
        if b.shape() != self.lora_b.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "Expected B shape {:?}, got {:?}",
                self.lora_b.shape(),
                b.shape()
            )));
        }
        self.lora_b = b;
        Ok(())
    }

    /// Compute the LoRA delta: scaling * B @ A
    pub fn lora_delta(&self) -> Array2<f64> {
        self.lora_b.dot(&self.lora_a) * self.scaling
    }

    /// Compute the effective weight: W + scaling * B @ A (or just W if merged).
    pub fn effective_weight(&self) -> Array2<f64> {
        if self.merged {
            self.weight.clone()
        } else {
            &self.weight + &self.lora_delta()
        }
    }

    /// Compute the rank of the LoRA perturbation using SVD-like analysis.
    /// Returns the number of singular values above the given threshold.
    pub fn effective_rank(&self, threshold: f64) -> usize {
        let delta = self.lora_delta();
        // Simple approximation: compute column norms
        let mut non_zero_cols = 0;
        for col_idx in 0..delta.ncols() {
            let col = delta.index_axis(Axis(1), col_idx);
            let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > threshold {
                non_zero_cols += 1;
            }
        }
        non_zero_cols.min(self.rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_lora_linear_creation() {
        let weight = Array2::<f64>::eye(8);
        let config = LoRAConfig {
            rank: 4,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight, &config);
        assert!(lora.is_ok());
        let lora = lora.expect("creation should succeed");
        assert_eq!(lora.rank(), 4);
        assert_eq!(lora.lora_a().shape(), &[4, 8]);
        assert_eq!(lora.lora_b().shape(), &[8, 4]);
    }

    #[test]
    fn test_zero_b_preserves_original_output() {
        let weight = Array2::from_shape_fn((4, 6), |(_i, _j)| 0.5);
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight.clone(), &config).expect("creation failed");

        let input = Array2::from_shape_fn((2, 6), |(i, j)| (i * 6 + j) as f64 * 0.1);
        let lora_output = lora.forward(&input).expect("forward failed");
        let original_output = input.dot(&weight.t());

        // B is zeros, so LoRA output should equal original
        for (a, b) in lora_output.iter().zip(original_output.iter()) {
            assert!((a - b).abs() < 1e-10, "outputs differ: {a} vs {b}");
        }
    }

    #[test]
    fn test_merged_vs_unmerged_same_output() {
        let weight = Array2::from_shape_fn((4, 6), |(i, j)| (i as f64 + j as f64) * 0.1);
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let mut lora = LoRALinear::new(weight, &config).expect("creation failed");

        // Set non-zero B
        let b = Array2::from_shape_fn((4, 2), |(i, j)| (i as f64 - j as f64) * 0.01);
        lora.set_lora_b(b).expect("set_lora_b failed");

        let input = Array2::from_shape_fn((3, 6), |(i, j)| (i * 6 + j) as f64 * 0.05);

        let unmerged_output = lora.forward(&input).expect("forward failed");
        lora.merge().expect("merge failed");
        let merged_output = lora.forward(&input).expect("forward failed");

        for (a, b) in unmerged_output.iter().zip(merged_output.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "merged vs unmerged differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_merge_unmerge_roundtrip() {
        let weight = Array2::from_shape_fn((4, 6), |(i, j)| (i as f64 + j as f64) * 0.1);
        let original_weight = weight.clone();
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let mut lora = LoRALinear::new(weight, &config).expect("creation failed");

        // Set non-zero B
        let b = Array2::from_shape_fn((4, 2), |(i, j)| (i as f64 - j as f64) * 0.01);
        lora.set_lora_b(b).expect("set_lora_b failed");

        lora.merge().expect("merge failed");
        lora.unmerge().expect("unmerge failed");

        for (a, b) in lora.weight().iter().zip(original_weight.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "weight changed after merge+unmerge: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rank1_lora() {
        let weight = Array2::<f64>::zeros((4, 6));
        let config = LoRAConfig {
            rank: 1,
            alpha: 1.0,
            ..Default::default()
        };
        let mut lora = LoRALinear::new(weight, &config).expect("creation failed");

        // Set rank-1 factors
        let a = Array2::from_shape_fn((1, 6), |(_, j)| j as f64);
        let b = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        lora.set_lora_a(a).expect("set_lora_a failed");
        lora.set_lora_b(b).expect("set_lora_b failed");

        let delta = lora.lora_delta();
        // rank-1: delta = scaling * b @ a, should have rank 1
        // Check that all rows are proportional
        let row0 = delta.row(0).to_owned();
        // Row 0 is all zeros (i=0), check rows 1-3 are multiples of row 1
        if row0.iter().all(|x| x.abs() < 1e-10) {
            let row1 = delta.row(1).to_owned();
            for i in 2..4 {
                let row_i = delta.row(i).to_owned();
                let ratio = i as f64;
                for (a, b) in row_i.iter().zip(row1.iter()) {
                    if b.abs() > 1e-10 {
                        assert!(
                            (a / b - ratio).abs() < 1e-10,
                            "not rank-1: row ratio mismatch"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_stats_parameter_counts() {
        let weight = Array2::<f64>::eye(16);
        let config = LoRAConfig {
            rank: 4,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight, &config).expect("creation failed");
        let stats = lora.stats();

        assert_eq!(stats.frozen_params, 16 * 16); // 256
        assert_eq!(stats.trainable_params, 4 * 16 + 16 * 4); // 128
        assert_eq!(stats.total_params, 256 + 128);
        assert!((stats.compression_ratio - 128.0 / 384.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_equals_min_dim() {
        // Edge case: rank = min(in, out)
        let weight = Array2::<f64>::eye(4);
        let config = LoRAConfig {
            rank: 4,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight, &config);
        assert!(lora.is_ok());
    }

    #[test]
    fn test_rank_exceeds_dim_error() {
        let weight = Array2::<f64>::eye(4);
        let config = LoRAConfig {
            rank: 5,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight, &config);
        assert!(lora.is_err());
    }

    #[test]
    fn test_dimension_mismatch_forward() {
        let weight = Array2::<f64>::eye(4);
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let lora = LoRALinear::new(weight, &config).expect("creation failed");

        let bad_input = Array2::<f64>::ones((2, 5)); // wrong features
        assert!(lora.forward(&bad_input).is_err());
    }

    #[test]
    fn test_double_merge_error() {
        let weight = Array2::<f64>::eye(4);
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let mut lora = LoRALinear::new(weight, &config).expect("creation failed");
        lora.merge().expect("first merge failed");
        assert!(lora.merge().is_err());
    }

    #[test]
    fn test_unmerge_without_merge_error() {
        let weight = Array2::<f64>::eye(4);
        let config = LoRAConfig {
            rank: 2,
            ..Default::default()
        };
        let mut lora = LoRALinear::new(weight, &config).expect("creation failed");
        assert!(lora.unmerge().is_err());
    }
}
