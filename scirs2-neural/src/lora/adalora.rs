//! AdaLoRA: Adaptive Budget Allocation via Singular Value Decomposition.
//!
//! Instead of learning two low-rank matrices A and B as in LoRA, AdaLoRA
//! parameterises the weight update as `ΔW = U · diag(s) · V^T`, where U, s, V
//! are optimised jointly.  An exponential-moving-average importance score is
//! tracked per singular triplet, and low-importance triplets are pruned (masked
//! out) to reach a configurable target-rank budget.
//!
//! # References
//!
//! - Zhang et al., "Adaptive Budget Allocation for Parameter-Efficient
//!   Fine-Tuning", ICLR 2023

use scirs2_core::ndarray::{Array1, Array2};

use crate::{NeuralError, Result};

// ──────────────────────────── Config ────────────────────────────────────────

/// Configuration for an AdaLoRA layer.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::adalora::AdaLoraConfig;
///
/// let cfg = AdaLoraConfig::default();
/// assert_eq!(cfg.initial_rank, 12);
/// assert_eq!(cfg.target_rank, 4);
/// ```
#[derive(Debug, Clone)]
pub struct AdaLoraConfig {
    /// Starting (wide) rank used to allocate U, s, V.
    pub initial_rank: usize,
    /// Budget rank that pruning converges towards.
    pub target_rank: usize,
    /// Number of optimisation steps between pruning events.
    pub delta_t: usize,
    /// EMA decay for importance score (ρ₁ in the paper).
    pub beta1: f64,
    /// EMA decay for gradient sensitivity (ρ₂ in the paper, currently unused
    /// in the gradient-magnitude formulation).
    pub beta2: f64,
    /// Scaling factor α; effective scale = α/initial_rank.
    pub alpha: f64,
}

impl Default for AdaLoraConfig {
    fn default() -> Self {
        Self {
            initial_rank: 12,
            target_rank: 4,
            delta_t: 10,
            beta1: 0.85,
            beta2: 0.85,
            alpha: 1.0,
        }
    }
}

// ──────────────────────────── Layer ─────────────────────────────────────────

/// AdaLoRA layer using SVD parameterisation.
///
/// `ΔW = s_scale · U · diag(s ⊙ mask) · V`
///
/// where `s_scale = α / initial_rank`.
///
/// Trainable tensors: `u_mat`, `singular_values`, `v_mat`.
/// Non-trainable tensors: `weight` (frozen), `importance_ema`, `rank_mask`.
///
/// # Example
///
/// ```
/// use scirs2_neural::lora::adalora::{AdaLoraLayer, AdaLoraConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let weight = Array2::<f64>::eye(8);
/// let cfg = AdaLoraConfig { initial_rank: 4, target_rank: 2, ..Default::default() };
/// let layer = AdaLoraLayer::new(weight, &cfg).expect("create AdaLoraLayer");
/// let input = Array2::<f64>::ones((2, 8));
/// let out = layer.forward(&input).expect("forward");
/// assert_eq!(out.shape(), &[2, 8]);
/// ```
pub struct AdaLoraLayer {
    /// Frozen base weight [out × in].
    weight: Array2<f64>,
    /// Left singular matrix U [out × initial_rank].
    pub u_mat: Array2<f64>,
    /// Singular values s [initial_rank].
    pub singular_values: Array1<f64>,
    /// Right singular matrix V [initial_rank × in].
    pub v_mat: Array2<f64>,
    /// Exponential moving average of importance score per rank index.
    importance_ema: Array1<f64>,
    /// Binary mask: 1.0 = active, 0.0 = pruned.
    rank_mask: Array1<f64>,
    /// Precomputed α / initial_rank.
    scaling: f64,
    config: AdaLoraConfig,
}

impl AdaLoraLayer {
    /// Create a new AdaLoRA layer.
    ///
    /// - `u_mat`, `v_mat` initialised to zeros; `singular_values` to small constant 0.01.
    /// - All ranks active at start.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidArgument`] if `initial_rank == 0` or
    /// `target_rank > initial_rank`.
    pub fn new(weight: Array2<f64>, config: &AdaLoraConfig) -> Result<Self> {
        if config.initial_rank == 0 {
            return Err(NeuralError::InvalidArgument(
                "AdaLoRA initial_rank must be > 0".to_string(),
            ));
        }
        if config.target_rank > config.initial_rank {
            return Err(NeuralError::InvalidArgument(format!(
                "AdaLoRA target_rank {} must be ≤ initial_rank {}",
                config.target_rank, config.initial_rank
            )));
        }

        let (out_f, in_f) = (weight.nrows(), weight.ncols());
        let r = config.initial_rank;
        let scaling = config.alpha / r as f64;

        let u_mat = Array2::zeros((out_f, r));
        let singular_values = Array1::from_elem(r, 0.01);
        let v_mat = Array2::zeros((r, in_f));
        let importance_ema = Array1::zeros(r);
        let rank_mask = Array1::ones(r);

        Ok(Self {
            weight,
            u_mat,
            singular_values,
            v_mat,
            importance_ema,
            rank_mask,
            scaling,
            config: config.clone(),
        })
    }

    /// Compute the effective weight.
    ///
    /// `W_eff = W_0 + scaling · U · diag(s ⊙ mask) · V`
    pub fn effective_weight(&self) -> Array2<f64> {
        let (out_f, in_f) = (self.weight.nrows(), self.weight.ncols());
        let mut delta = Array2::<f64>::zeros((out_f, in_f));

        for k in 0..self.config.initial_rank {
            if self.rank_mask[k] < 0.5 {
                continue;
            }
            let s = self.singular_values[k] * self.scaling;
            let u_col = self.u_mat.column(k);
            let v_row = self.v_mat.row(k);
            // Rank-1 outer product: U[:,k] ⊗ V[k,:]
            for i in 0..out_f {
                for j in 0..in_f {
                    delta[[i, j]] += s * u_col[i] * v_row[j];
                }
            }
        }
        &self.weight + &delta
    }

    /// Forward pass: `y = x · W_eff^T`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] on input-shape mismatch.
    pub fn forward(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        let in_features = self.weight.ncols();
        if input.ncols() != in_features {
            return Err(NeuralError::DimensionMismatch(format!(
                "AdaLoRA expects {} input features, got {}",
                in_features,
                input.ncols()
            )));
        }
        Ok(input.dot(&self.effective_weight().t()))
    }

    /// Update the importance EMA using gradient of the singular values.
    ///
    /// `Imp_k ← β₁ · EMA_k + (1 − β₁) · |s_k · ∂L/∂s_k|`
    ///
    /// The gradient `grad_s` must have length `initial_rank`.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::DimensionMismatch`] if `grad_s.len() != initial_rank`.
    pub fn update_importance(&mut self, grad_s: &Array1<f64>) -> Result<()> {
        if grad_s.len() != self.config.initial_rank {
            return Err(NeuralError::DimensionMismatch(format!(
                "update_importance: expected grad_s of length {}, got {}",
                self.config.initial_rank,
                grad_s.len()
            )));
        }
        for k in 0..self.config.initial_rank {
            let sensitivity = (self.singular_values[k] * grad_s[k]).abs();
            self.importance_ema[k] = self.config.beta1 * self.importance_ema[k]
                + (1.0 - self.config.beta1) * sensitivity;
        }
        Ok(())
    }

    /// Prune ranks to the given `budget` by masking out low-importance triplets.
    ///
    /// Ranks are sorted by `importance_ema` in descending order and the top
    /// `budget.min(initial_rank)` are kept active.
    ///
    /// Returns the number of active ranks after pruning.
    pub fn prune_to_budget(&mut self, budget: usize) -> usize {
        let r = self.config.initial_rank;

        // Sort indices by descending importance score.
        let mut indexed: Vec<(usize, f64)> = (0..r).map(|k| (k, self.importance_ema[k])).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let keep = budget.min(r);
        self.rank_mask.fill(0.0);
        for entry in indexed.iter().take(keep) {
            self.rank_mask[entry.0] = 1.0;
        }
        keep
    }

    /// Number of currently active (unmasked) ranks.
    pub fn active_rank(&self) -> usize {
        self.rank_mask.iter().filter(|&&m| m > 0.5).count()
    }

    /// Return the target rank specified in the configuration.
    pub fn target_rank(&self) -> usize {
        self.config.target_rank
    }

    /// Reference to the importance EMA vector (read-only).
    pub fn importance_ema(&self) -> &Array1<f64> {
        &self.importance_ema
    }

    /// Reference to the rank mask vector (read-only).
    pub fn rank_mask(&self) -> &Array1<f64> {
        &self.rank_mask
    }

    /// Reference to the frozen base weight.
    pub fn weight(&self) -> &Array2<f64> {
        &self.weight
    }
}

// ──────────────────────────── tests ─────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_layer(out: usize, in_f: usize) -> AdaLoraLayer {
        let w = Array2::from_shape_fn((out, in_f), |(i, j)| (i * in_f + j) as f64 * 0.1);
        AdaLoraLayer::new(w, &AdaLoraConfig::default()).expect("new")
    }

    #[test]
    fn adalora_effective_weight_shape() {
        let layer = make_layer(8, 6);
        let w = layer.effective_weight();
        assert_eq!(w.shape(), &[8, 6]);
    }

    #[test]
    fn adalora_zero_singular_values() {
        // When singular_values = 0, delta = 0 → effective_weight == weight.
        let w = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j + 1) as f64);
        let mut layer = AdaLoraLayer::new(w.clone(), &AdaLoraConfig::default()).expect("new");
        layer.singular_values.fill(0.0);

        let eff = layer.effective_weight();
        for (a, b) in eff.iter().zip(w.iter()) {
            assert!((a - b).abs() < 1e-12, "delta should be zero");
        }
    }

    #[test]
    fn adalora_prune_to_budget() {
        let mut layer = make_layer(8, 8);
        // Manually set varied importance scores.
        for k in 0..layer.config.initial_rank {
            layer.importance_ema[k] = k as f64;
        }
        let kept = layer.prune_to_budget(4);
        assert_eq!(kept, 4);
        assert_eq!(layer.active_rank(), 4);
    }

    #[test]
    fn adalora_update_importance_updates_ema() {
        let mut layer = make_layer(4, 4);
        let grad_s = Array1::from_elem(layer.config.initial_rank, 1.0);
        // initial EMA = 0, s = 0.01 → update = 0.85*0 + 0.15*|0.01*1| = 0.0015
        layer.update_importance(&grad_s).expect("update_importance");
        for k in 0..layer.config.initial_rank {
            let expected = (1.0 - 0.85) * (0.01_f64 * 1.0).abs();
            assert!(
                (layer.importance_ema[k] - expected).abs() < 1e-12,
                "EMA mismatch at k={k}: got {}, expected {expected}",
                layer.importance_ema[k]
            );
        }
    }

    #[test]
    fn adalora_all_masked_out() {
        let mut layer = make_layer(4, 6);
        let kept = layer.prune_to_budget(0);
        assert_eq!(kept, 0);
        assert_eq!(layer.active_rank(), 0);

        // Forward should still work (delta = 0).
        let input = Array2::from_elem((2, 6), 1.0);
        assert!(layer.forward(&input).is_ok());
    }

    #[test]
    fn adalora_forward_shape() {
        let layer = make_layer(5, 7);
        let input = Array2::from_elem((3, 7), 0.5);
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[3, 5]);
    }

    #[test]
    fn adalora_default_config() {
        let cfg = AdaLoraConfig::default();
        assert_eq!(cfg.initial_rank, 12);
        assert_eq!(cfg.target_rank, 4);
        assert_eq!(cfg.delta_t, 10);
        assert!((cfg.beta1 - 0.85).abs() < 1e-12);
        assert!((cfg.beta2 - 0.85).abs() < 1e-12);
        assert!((cfg.alpha - 1.0).abs() < 1e-12);
    }

    #[test]
    fn adalora_invalid_rank_zero() {
        let w = Array2::<f64>::eye(4);
        let cfg = AdaLoraConfig {
            initial_rank: 0,
            target_rank: 0,
            ..Default::default()
        };
        assert!(AdaLoraLayer::new(w, &cfg).is_err());
    }

    #[test]
    fn adalora_invalid_target_exceeds_initial() {
        let w = Array2::<f64>::eye(4);
        let cfg = AdaLoraConfig {
            initial_rank: 4,
            target_rank: 8,
            ..Default::default()
        };
        assert!(AdaLoraLayer::new(w, &cfg).is_err());
    }
}
