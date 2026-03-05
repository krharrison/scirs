//! Reward Model for Reinforcement Learning from Human Feedback (RLHF)
//!
//! This module provides components for training a reward model from human preference
//! data. The reward model learns to assign scalar scores to sequences such that
//! preferred outputs receive higher scores than rejected ones.
//!
//! # Overview
//!
//! The reward model consists of:
//! - A **base encoder** that maps input representations to hidden states
//! - A **reward head** (linear projection) that maps hidden states to scalar rewards
//!
//! Training uses the **Bradley-Terry** model: for a pair (chosen, rejected), the
//! probability that "chosen" is preferred is `σ(r_chosen - r_rejected)`.
//!
//! # References
//!
//! - Christiano et al., "Deep Reinforcement Learning from Human Preferences", NeurIPS 2017
//! - Stiennon et al., "Learning to summarize with human feedback", NeurIPS 2020
//! - Ziegler et al., "Fine-Tuning Language Models from Human Preferences", arXiv 2019
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::reward_model::{RewardModelConfig, RewardModel};
//! use scirs2_core::ndarray::Array2;
//!
//! let config = RewardModelConfig {
//!     hidden_size: 64,
//!     num_layers: 2,
//!     dropout: 0.1,
//!     ..Default::default()
//! };
//!
//! let model = RewardModel::<f64>::new(config).expect("reward model init");
//!
//! // Compute reward for a batch of hidden states [batch, hidden_size]
//! let hidden = Array2::<f64>::zeros((4, 64));
//! let rewards = model.reward_head_forward(&hidden).expect("reward head ok");
//! assert_eq!(rewards.len(), 4);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the reward model.
#[derive(Debug, Clone)]
pub struct RewardModelConfig {
    /// Dimensionality of the hidden states produced by the base encoder.
    pub hidden_size: usize,
    /// Number of layers in the reward MLP head (minimum 1).
    pub num_layers: usize,
    /// Intermediate MLP hidden size (0 = same as `hidden_size`).
    pub intermediate_size: usize,
    /// Dropout probability applied inside the MLP head.
    pub dropout: f64,
    /// Margin for pairwise ranking loss (Bradley-Terry uses 0.0).
    pub margin: f64,
    /// Label smoothing for the preference loss (range [0, 0.5)).
    pub label_smoothing: f64,
    /// Reduction strategy for the batch loss ("mean" or "sum").
    pub reduction: LossReduction,
}

impl Default for RewardModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_layers: 1,
            intermediate_size: 0,
            dropout: 0.0,
            margin: 0.0,
            label_smoothing: 0.0,
            reduction: LossReduction::Mean,
        }
    }
}

/// How to aggregate scalar losses over a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossReduction {
    /// Average loss over the batch.
    Mean,
    /// Sum loss over the batch.
    Sum,
}

// ============================================================================
// Reward Model
// ============================================================================

/// A reward model that wraps a linear reward head applied to pre-computed
/// hidden states.
///
/// In practice the base encoder (transformer, LSTM, etc.) is handled
/// externally; this struct owns only the **reward head** weights so that it
/// can be composed with any backbone.
///
/// The reward head is a small MLP:
/// ```text
/// h -> LayerNorm -> Linear(hidden, intermediate) -> GELU -> Linear(intermediate, 1)
/// ```
/// When `num_layers == 1` the intermediate linear is omitted.
#[derive(Debug, Clone)]
pub struct RewardModel<F: Float + Debug> {
    /// Configuration.
    pub config: RewardModelConfig,
    /// Weight matrix for each MLP layer: shape `[out, in]`.
    pub weights: Vec<Array2<F>>,
    /// Bias vectors for each MLP layer.
    pub biases: Vec<Array1<F>>,
    /// Layer-norm scale (gamma) for the input normalisation.
    pub ln_weight: Array1<F>,
    /// Layer-norm bias (beta) for the input normalisation.
    pub ln_bias: Array1<F>,
}

impl<F> RewardModel<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    /// Construct a new `RewardModel` with Xavier-initialised weights.
    pub fn new(config: RewardModelConfig) -> Result<Self> {
        if config.hidden_size == 0 {
            return Err(NeuralError::ConfigError(
                "RewardModel: hidden_size must be > 0".to_string(),
            ));
        }

        let intermediate = if config.intermediate_size == 0 {
            config.hidden_size
        } else {
            config.intermediate_size
        };

        let (weights, biases) = Self::init_weights(&config, intermediate)?;

        let ln_weight = Array1::from_elem(config.hidden_size, F::one());
        let ln_bias = Array1::zeros(config.hidden_size);

        Ok(Self {
            config,
            weights,
            biases,
            ln_weight,
            ln_bias,
        })
    }

    /// Initialise MLP weights using Xavier uniform.
    fn init_weights(
        config: &RewardModelConfig,
        intermediate: usize,
    ) -> Result<(Vec<Array2<F>>, Vec<Array1<F>>)> {
        let mut weights: Vec<Array2<F>> = Vec::new();
        let mut biases: Vec<Array1<F>> = Vec::new();

        let hidden = config.hidden_size;

        if config.num_layers == 1 {
            // Single linear: hidden -> 1
            let w = xavier_uniform(hidden, 1)?;
            let b = Array1::zeros(1);
            weights.push(w);
            biases.push(b);
        } else {
            // First layer: hidden -> intermediate
            let w0 = xavier_uniform(hidden, intermediate)?;
            let b0 = Array1::zeros(intermediate);
            weights.push(w0);
            biases.push(b0);

            // Middle layers: intermediate -> intermediate
            for _ in 1..config.num_layers - 1 {
                let wm = xavier_uniform(intermediate, intermediate)?;
                let bm = Array1::zeros(intermediate);
                weights.push(wm);
                biases.push(bm);
            }

            // Final layer: intermediate -> 1
            let wn = xavier_uniform(intermediate, 1)?;
            let bn = Array1::zeros(1);
            weights.push(wn);
            biases.push(bn);
        }

        Ok((weights, biases))
    }

    /// Apply layer normalisation to a 2-D input `[batch, hidden]`.
    fn layer_norm(&self, x: &Array2<F>) -> Result<Array2<F>> {
        let eps = F::from_f64(1e-5).ok_or_else(|| {
            NeuralError::ComputationError("RewardModel: cannot convert eps".to_string())
        })?;

        let mut out = x.clone();
        let batch = x.nrows();

        for i in 0..batch {
            let row = x.slice(s![i, ..]);
            let mean = row.sum() / F::from_usize(x.ncols()).ok_or_else(|| {
                NeuralError::ComputationError("RewardModel: cannot convert ncols".to_string())
            })?;
            let var = row.mapv(|v| (v - mean) * (v - mean)).sum()
                / F::from_usize(x.ncols()).ok_or_else(|| {
                    NeuralError::ComputationError("RewardModel: cannot convert ncols".to_string())
                })?;
            let std_dev = (var + eps).sqrt();
            for j in 0..x.ncols() {
                out[[i, j]] =
                    (x[[i, j]] - mean) / std_dev * self.ln_weight[j] + self.ln_bias[j];
            }
        }
        Ok(out)
    }

    /// GELU activation: `x * Φ(x)` approximated with `tanh`.
    fn gelu(x: F) -> F {
        // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))
        let c1 = F::from_f64(0.7978845608028654).unwrap_or(F::one()); // sqrt(2/π)
        let c2 = F::from_f64(0.044715).unwrap_or(F::zero());
        let half = F::from_f64(0.5).unwrap_or(F::one());
        let one = F::one();
        let inner = c1 * (x + c2 * x * x * x);
        // tanh approximation
        let tanh_val = inner.tanh();
        half * x * (one + tanh_val)
    }

    /// Apply one linear layer: `x @ W^T + b`.
    ///
    /// `x` has shape `[batch, in_features]`, `W` has shape `[out_features, in_features]`.
    fn linear(x: &Array2<F>, w: &Array2<F>, b: &Array1<F>) -> Result<Array2<F>> {
        let batch = x.nrows();
        let out_features = w.nrows();

        if x.ncols() != w.ncols() {
            return Err(NeuralError::DimensionMismatch(format!(
                "linear: x.ncols={} != w.ncols={}",
                x.ncols(),
                w.ncols()
            )));
        }

        let mut out = Array2::zeros((batch, out_features));
        for i in 0..batch {
            for o in 0..out_features {
                let mut acc = b[o];
                for k in 0..x.ncols() {
                    acc += x[[i, k]] * w[[o, k]];
                }
                out[[i, o]] = acc;
            }
        }
        Ok(out)
    }

    /// Forward pass of the reward head.
    ///
    /// Takes pre-computed hidden states of shape `[batch, hidden_size]` and
    /// returns a 1-D reward vector of length `batch`.
    pub fn reward_head_forward(&self, hidden: &Array2<F>) -> Result<Array1<F>> {
        if hidden.ncols() != self.config.hidden_size {
            return Err(NeuralError::DimensionMismatch(format!(
                "reward_head_forward: expected hidden_size={} but got {}",
                self.config.hidden_size,
                hidden.ncols()
            )));
        }
        if hidden.nrows() == 0 {
            return Err(NeuralError::InvalidArgument(
                "reward_head_forward: batch must be > 0".to_string(),
            ));
        }

        // Layer-norm
        let mut x = self.layer_norm(hidden)?;

        // MLP layers
        let n_layers = self.weights.len();
        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            x = Self::linear(&x, w, b)?;
            // Apply GELU for all but the last layer
            if layer_idx < n_layers - 1 {
                x.mapv_inplace(|v| Self::gelu(v));
            }
        }

        // `x` now has shape [batch, 1]; squeeze to [batch]
        let rewards: Array1<F> = x.column(0).to_owned();
        Ok(rewards)
    }

    // -----------------------------------------------------------------------
    // Loss functions
    // -----------------------------------------------------------------------

    /// Compute the Bradley-Terry preference loss.
    ///
    /// Given rewards for chosen and rejected sequences, computes:
    /// ```text
    /// L = -mean( log σ(r_chosen - r_rejected) )
    /// ```
    /// with optional label smoothing.
    ///
    /// # Arguments
    /// - `r_chosen`  – rewards for the preferred sequences, shape `[batch]`
    /// - `r_rejected` – rewards for the dispreferred sequences, shape `[batch]`
    ///
    /// # Returns
    /// Scalar loss.
    pub fn compute_reward_loss(
        &self,
        r_chosen: &Array1<F>,
        r_rejected: &Array1<F>,
    ) -> Result<F> {
        let n = r_chosen.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "compute_reward_loss: empty batch".to_string(),
            ));
        }
        if r_rejected.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "compute_reward_loss: r_chosen len={} != r_rejected len={}",
                n,
                r_rejected.len()
            )));
        }

        let smoothing = F::from_f64(self.config.label_smoothing).ok_or_else(|| {
            NeuralError::ComputationError(
                "compute_reward_loss: cannot convert label_smoothing".to_string(),
            )
        })?;
        // target = 1.0 - label_smoothing
        let target = F::one() - smoothing;

        let mut total_loss = F::zero();
        for i in 0..n {
            let diff = r_chosen[i] - r_rejected[i];
            // log σ(diff) = -softplus(-diff) = -log(1 + exp(-diff))
            let log_sigmoid = log_sigmoid_f(diff)?;
            // label-smoothed cross-entropy: target * log_p + (1-target) * log(1-p)
            // log(1 - σ(diff)) = log σ(-diff)
            let log_sigmoid_neg = log_sigmoid_f(-diff)?;
            let loss_i = -(target * log_sigmoid + (F::one() - target) * log_sigmoid_neg);
            total_loss += loss_i;
        }

        let n_f = F::from_usize(n)
            .ok_or_else(|| NeuralError::ComputationError("cannot convert n".to_string()))?;

        match self.config.reduction {
            LossReduction::Mean => Ok(total_loss / n_f),
            LossReduction::Sum => Ok(total_loss),
        }
    }

    /// Compute a margin-based pairwise ranking loss.
    ///
    /// ```text
    /// L = mean( max(0, margin - (r_chosen - r_rejected)) )
    /// ```
    ///
    /// This is the hinge variant sometimes used when the Bradley-Terry
    /// assumption does not hold (e.g. noisy preferences).
    ///
    /// # Arguments
    /// - `r_chosen`   – rewards for preferred sequences, shape `[batch]`
    /// - `r_rejected` – rewards for dispreferred sequences, shape `[batch]`
    /// - `margin`     – enforced gap between reward scores (use `None` to
    ///                  fall back to `self.config.margin`)
    ///
    /// # Returns
    /// Scalar hinge loss.
    pub fn pairwise_ranking_loss(
        &self,
        r_chosen: &Array1<F>,
        r_rejected: &Array1<F>,
        margin: Option<f64>,
    ) -> Result<F> {
        let n = r_chosen.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "pairwise_ranking_loss: empty batch".to_string(),
            ));
        }
        if r_rejected.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "pairwise_ranking_loss: r_chosen len={} != r_rejected len={}",
                n,
                r_rejected.len()
            )));
        }

        let m_f64 = margin.unwrap_or(self.config.margin);
        let m = F::from_f64(m_f64).ok_or_else(|| {
            NeuralError::ComputationError(
                "pairwise_ranking_loss: cannot convert margin".to_string(),
            )
        })?;

        let mut total = F::zero();
        for i in 0..n {
            let gap = r_chosen[i] - r_rejected[i];
            let hinge = (m - gap).max(F::zero());
            total += hinge;
        }

        let n_f = F::from_usize(n)
            .ok_or_else(|| NeuralError::ComputationError("cannot convert n".to_string()))?;

        match self.config.reduction {
            LossReduction::Mean => Ok(total / n_f),
            LossReduction::Sum => Ok(total),
        }
    }
}

// ============================================================================
// Standalone reward-loss helpers (usable outside RewardModel)
// ============================================================================

/// Compute Bradley-Terry preference loss directly from reward tensors.
///
/// Equivalent to `RewardModel::compute_reward_loss` but does not require a
/// model instance.
///
/// # Arguments
/// - `r_chosen`       – rewards for the preferred sequences
/// - `r_rejected`     – rewards for the dispreferred sequences
/// - `label_smoothing` – label smoothing factor in `[0, 0.5)`
///
/// # Returns
/// Mean Bradley-Terry loss.
pub fn bradley_terry_loss<F>(
    r_chosen: &Array1<F>,
    r_rejected: &Array1<F>,
    label_smoothing: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let n = r_chosen.len();
    if n == 0 {
        return Err(NeuralError::InvalidArgument(
            "bradley_terry_loss: empty batch".to_string(),
        ));
    }
    if r_rejected.len() != n {
        return Err(NeuralError::DimensionMismatch(format!(
            "bradley_terry_loss: length mismatch {} vs {}",
            n,
            r_rejected.len()
        )));
    }

    let smoothing = F::from_f64(label_smoothing).ok_or_else(|| {
        NeuralError::ComputationError("bradley_terry_loss: cannot convert smoothing".to_string())
    })?;
    let target = F::one() - smoothing;

    let mut total = F::zero();
    for i in 0..n {
        let diff = r_chosen[i] - r_rejected[i];
        let log_p = log_sigmoid_f(diff)?;
        let log_one_minus_p = log_sigmoid_f(-diff)?;
        total += -(target * log_p + (F::one() - target) * log_one_minus_p);
    }

    let n_f = F::from_usize(n)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert n".to_string()))?;
    Ok(total / n_f)
}

/// Compute pairwise ranking (hinge) loss directly from reward tensors.
pub fn pairwise_hinge_loss<F>(
    r_chosen: &Array1<F>,
    r_rejected: &Array1<F>,
    margin: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let n = r_chosen.len();
    if n == 0 {
        return Err(NeuralError::InvalidArgument(
            "pairwise_hinge_loss: empty batch".to_string(),
        ));
    }
    if r_rejected.len() != n {
        return Err(NeuralError::DimensionMismatch(format!(
            "pairwise_hinge_loss: length mismatch {} vs {}",
            n,
            r_rejected.len()
        )));
    }

    let m = F::from_f64(margin).ok_or_else(|| {
        NeuralError::ComputationError("pairwise_hinge_loss: cannot convert margin".to_string())
    })?;

    let mut total = F::zero();
    for i in 0..n {
        let gap = r_chosen[i] - r_rejected[i];
        total += (m - gap).max(F::zero());
    }

    let n_f = F::from_usize(n)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert n".to_string()))?;
    Ok(total / n_f)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Numerically stable log-sigmoid: `log σ(x) = -softplus(-x)`.
fn log_sigmoid_f<F: Float + FromPrimitive + Debug>(x: F) -> Result<F> {
    // For x >= 0: -log(1 + exp(-x))
    // For x < 0:   x - log(1 + exp(x))   (equivalent, avoids overflow)
    let zero = F::zero();
    let one = F::one();

    let result = if x >= zero {
        let neg_x = -x;
        let exp_neg_x = neg_x.exp();
        -(one + exp_neg_x).ln()
    } else {
        let exp_x = x.exp();
        x - (one + exp_x).ln()
    };

    Ok(result)
}

/// Xavier uniform initialisation for a weight matrix of shape `[out, in]`.
fn xavier_uniform<F: Float + FromPrimitive + Debug>(
    fan_in: usize,
    fan_out: usize,
) -> Result<Array2<F>> {
    let limit = (6.0_f64 / (fan_in + fan_out) as f64).sqrt();
    let limit_f = F::from_f64(limit).ok_or_else(|| {
        NeuralError::ComputationError("xavier_uniform: cannot convert limit".to_string())
    })?;

    // Use a simple deterministic initialisation based on index to avoid
    // external RNG dependency.  In real training these weights will be
    // updated during the first gradient step.
    let mut w = Array2::zeros((fan_out, fan_in));
    for o in 0..fan_out {
        for i in 0..fan_in {
            // Deterministic pseudo-random via golden-ratio hash
            let hash = ((o * 2654435761) ^ (i * 2246822519)) as f64;
            let scaled = (hash.sin() * 43758.5453123).fract(); // in (-1, 1)
            let val = F::from_f64(scaled * limit).unwrap_or(F::zero());
            // Clamp to [-limit, limit]
            w[[o, i]] = val.min(limit_f).max(-limit_f);
        }
    }
    Ok(w)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_reward_model_forward_shape() {
        let config = RewardModelConfig {
            hidden_size: 16,
            num_layers: 1,
            ..Default::default()
        };
        let model = RewardModel::<f64>::new(config).expect("init");
        let hidden = Array2::<f64>::zeros((4, 16));
        let rewards = model.reward_head_forward(&hidden).expect("forward");
        assert_eq!(rewards.len(), 4);
    }

    #[test]
    fn test_reward_model_multilayer() {
        let config = RewardModelConfig {
            hidden_size: 32,
            num_layers: 3,
            intermediate_size: 16,
            ..Default::default()
        };
        let model = RewardModel::<f64>::new(config).expect("init");
        let hidden = Array2::<f64>::zeros((8, 32));
        let rewards = model.reward_head_forward(&hidden).expect("forward");
        assert_eq!(rewards.len(), 8);
    }

    #[test]
    fn test_bradley_terry_loss_perfect_separation() {
        // When chosen >> rejected, loss should be near zero
        let chosen = Array1::from(vec![10.0_f64; 4]);
        let rejected = Array1::from(vec![-10.0_f64; 4]);
        let loss = bradley_terry_loss(&chosen, &rejected, 0.0).expect("loss");
        assert!(loss < 1e-3, "loss={loss}");
    }

    #[test]
    fn test_bradley_terry_loss_random() {
        let chosen = Array1::from(vec![1.0_f64, 2.0, 3.0, 4.0]);
        let rejected = Array1::from(vec![0.5_f64, 1.5, 2.5, 3.5]);
        let loss = bradley_terry_loss(&chosen, &rejected, 0.0).expect("loss");
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_pairwise_hinge_loss() {
        let chosen = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let rejected = Array1::from(vec![0.0_f64, 0.0, 0.0]);
        let loss = pairwise_hinge_loss(&chosen, &rejected, 0.5).expect("loss");
        // margin=0.5, gaps are 1,2,3 → all > margin → hinge = 0 for all
        assert!((loss - 0.0).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_pairwise_hinge_loss_active() {
        let chosen = Array1::from(vec![0.1_f64, 0.1]);
        let rejected = Array1::from(vec![0.0_f64, 0.0]);
        // gap = 0.1, margin = 1.0 → hinge = 0.9 for each
        let loss = pairwise_hinge_loss(&chosen, &rejected, 1.0).expect("loss");
        assert!((loss - 0.9).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_label_smoothing() {
        let chosen = Array1::from(vec![1.0_f64]);
        let rejected = Array1::from(vec![-1.0_f64]);
        let loss_no_smooth = bradley_terry_loss(&chosen, &rejected, 0.0).expect("l0");
        let loss_smooth = bradley_terry_loss(&chosen, &rejected, 0.1).expect("l1");
        // With label smoothing the loss should be higher (pulls towards 0.5)
        assert!(loss_smooth > loss_no_smooth, "smooth={loss_smooth} vs {loss_no_smooth}");
    }

    #[test]
    fn test_reward_model_compute_reward_loss() {
        let config = RewardModelConfig {
            hidden_size: 8,
            num_layers: 1,
            ..Default::default()
        };
        let model = RewardModel::<f64>::new(config).expect("init");
        let chosen = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let rejected = Array1::from(vec![0.0_f64, 1.0, 2.0]);
        let loss = model.compute_reward_loss(&chosen, &rejected).expect("loss");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_reward_model_pairwise_ranking_loss() {
        let config = RewardModelConfig {
            hidden_size: 8,
            margin: 1.0,
            ..Default::default()
        };
        let model = RewardModel::<f64>::new(config).expect("init");
        let chosen = Array1::from(vec![2.0_f64, 2.0]);
        let rejected = Array1::from(vec![0.0_f64, 0.0]);
        // gap = 2 > margin = 1 → loss = 0
        let loss = model.pairwise_ranking_loss(&chosen, &rejected, None).expect("loss");
        assert!((loss - 0.0).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let chosen = Array1::from(vec![1.0_f64, 2.0]);
        let rejected = Array1::from(vec![0.0_f64]);
        let result = bradley_terry_loss(&chosen, &rejected, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_reward_head_wrong_dim_error() {
        let config = RewardModelConfig {
            hidden_size: 16,
            ..Default::default()
        };
        let model = RewardModel::<f64>::new(config).expect("init");
        let bad_input = Array2::<f64>::zeros((2, 8)); // wrong hidden size
        let result = model.reward_head_forward(&bad_input);
        assert!(result.is_err());
    }
}
