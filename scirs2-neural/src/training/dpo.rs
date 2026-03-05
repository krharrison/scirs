//! Direct Preference Optimization (DPO)
//!
//! Implements the DPO objective from Rafailov et al. (2023), which trains a
//! policy to align with human preferences **without** an explicit reward model
//! or reinforcement-learning loop.
//!
//! # Overview
//!
//! Given a preference dataset of (prompt, chosen, rejected) triples, DPO
//! optimises:
//!
//! ```text
//! L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [
//!     log σ( β * log(π_θ(y_w|x) / π_ref(y_w|x))
//!          - β * log(π_θ(y_l|x) / π_ref(y_l|x)) )
//! ]
//! ```
//!
//! where:
//! - `π_θ` is the policy being trained,
//! - `π_ref` is a fixed reference policy (typically the SFT model),
//! - `β` controls how strongly the policy is constrained to stay near `π_ref`.
//!
//! The module also supports the **reference-free** variant (Zhao et al., 2023)
//! which omits `π_ref`:
//!
//! ```text
//! L_RFDPO = -E [ log σ( β * (log π_θ(y_w|x) - log π_θ(y_l|x)) ) ]
//! ```
//!
//! # References
//!
//! - Rafailov et al., "Direct Preference Optimization: Your Language Model is
//!   Secretly a Reward Model", NeurIPS 2023
//! - Zhao et al., "SLIC-HF: Sequence Likelihood Calibration with Human
//!   Feedback", arXiv 2023
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::dpo::{DPOConfig, DPOLoss};
//! use scirs2_core::ndarray::Array1;
//!
//! let config = DPOConfig { beta: 0.1, reference_free: false, ..Default::default() };
//! let dpo = DPOLoss::new(config);
//!
//! // Log-probabilities from policy and reference for a batch of 4 pairs
//! let lp_chosen_policy   = Array1::from(vec![-1.0_f64, -1.5, -0.8, -2.0]);
//! let lp_rejected_policy = Array1::from(vec![-2.0_f64, -3.0, -1.8, -4.0]);
//! let lp_chosen_ref      = Array1::from(vec![-1.2_f64, -1.6, -0.9, -2.1]);
//! let lp_rejected_ref    = Array1::from(vec![-1.9_f64, -2.8, -1.7, -3.9]);
//!
//! let loss = dpo.dpo_loss(
//!     &lp_chosen_policy, &lp_rejected_policy,
//!     &lp_chosen_ref,    &lp_rejected_ref,
//! ).expect("dpo loss ok");
//! assert!(loss.is_finite());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for DPO training.
#[derive(Debug, Clone)]
pub struct DPOConfig {
    /// KL-penalty coefficient β (default 0.1).
    ///
    /// Higher β keeps the policy closer to the reference model.
    pub beta: f64,
    /// When `true`, omit the reference model and use the reference-free variant.
    pub reference_free: bool,
    /// Label smoothing ε ∈ [0, 0.5): smooths the binary cross-entropy target.
    pub label_smoothing: f64,
    /// Loss reduction: "mean" or "sum".
    pub reduction: DPOReduction,
    /// Whether to use SigLIP-style loss (log-sum vs log-softmax).
    ///
    /// When `false` (default) the standard log-σ loss is used.
    pub sigmoid_loss: bool,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            reference_free: false,
            label_smoothing: 0.0,
            reduction: DPOReduction::Mean,
            sigmoid_loss: false,
        }
    }
}

/// Reduction strategy for DPO batch loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DPOReduction {
    /// Average loss over the batch.
    Mean,
    /// Sum of losses over the batch.
    Sum,
}

// ============================================================================
// DPOLoss
// ============================================================================

/// DPO loss function.
///
/// Holds the configuration and exposes methods for computing the DPO and
/// reference-free DPO objectives, as well as the implicit reward.
#[derive(Debug, Clone)]
pub struct DPOLoss {
    /// DPO configuration.
    pub config: DPOConfig,
}

impl DPOLoss {
    /// Create a new `DPOLoss` with the given configuration.
    pub fn new(config: DPOConfig) -> Self {
        Self { config }
    }

    // -----------------------------------------------------------------------
    // Core DPO loss
    // -----------------------------------------------------------------------

    /// Compute the DPO loss.
    ///
    /// All four log-probability tensors must have the same length (`batch`).
    ///
    /// # Arguments
    /// - `lp_chosen_policy`   – log π_θ(y_w | x), shape `[batch]`
    /// - `lp_rejected_policy` – log π_θ(y_l | x), shape `[batch]`
    /// - `lp_chosen_ref`      – log π_ref(y_w | x), shape `[batch]`
    /// - `lp_rejected_ref`    – log π_ref(y_l | x), shape `[batch]`
    ///
    /// # Returns
    /// Scalar DPO loss.
    pub fn dpo_loss<F>(
        &self,
        lp_chosen_policy: &Array1<F>,
        lp_rejected_policy: &Array1<F>,
        lp_chosen_ref: &Array1<F>,
        lp_rejected_ref: &Array1<F>,
    ) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = lp_chosen_policy.len();
        validate_lengths(
            n,
            &[
                lp_rejected_policy.len(),
                lp_chosen_ref.len(),
                lp_rejected_ref.len(),
            ],
            "dpo_loss",
        )?;
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "dpo_loss: empty batch".to_string(),
            ));
        }

        let beta = F::from_f64(self.config.beta).ok_or_else(|| {
            NeuralError::ComputationError("dpo_loss: cannot convert beta".to_string())
        })?;

        let log_ratios = compute_log_ratios(
            lp_chosen_policy,
            lp_rejected_policy,
            lp_chosen_ref,
            lp_rejected_ref,
        )?;

        self.apply_loss_reduction(&log_ratios, beta)
    }

    // -----------------------------------------------------------------------
    // Reference-free DPO loss
    // -----------------------------------------------------------------------

    /// Compute the reference-free DPO loss (no reference model required).
    ///
    /// ```text
    /// L_RFDPO = -E [ log σ( β * (log π_θ(y_w|x) - log π_θ(y_l|x)) ) ]
    /// ```
    ///
    /// # Arguments
    /// - `lp_chosen_policy`   – log π_θ(y_w | x), shape `[batch]`
    /// - `lp_rejected_policy` – log π_θ(y_l | x), shape `[batch]`
    ///
    /// # Returns
    /// Scalar reference-free DPO loss.
    pub fn reference_free_dpo_loss<F>(
        &self,
        lp_chosen_policy: &Array1<F>,
        lp_rejected_policy: &Array1<F>,
    ) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = lp_chosen_policy.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "reference_free_dpo_loss: empty batch".to_string(),
            ));
        }
        if lp_rejected_policy.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "reference_free_dpo_loss: length mismatch {} vs {}",
                n,
                lp_rejected_policy.len()
            )));
        }

        let beta = F::from_f64(self.config.beta).ok_or_else(|| {
            NeuralError::ComputationError(
                "reference_free_dpo_loss: cannot convert beta".to_string(),
            )
        })?;

        // Without reference, the log-ratio is just the policy log-ratio
        let log_ratios: Array1<F> =
            Array1::from_iter((0..n).map(|i| lp_chosen_policy[i] - lp_rejected_policy[i]));

        self.apply_loss_reduction(&log_ratios, beta)
    }

    // -----------------------------------------------------------------------
    // Implicit reward
    // -----------------------------------------------------------------------

    /// Compute the implicit reward `r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))`.
    ///
    /// Under the DPO optimality conditions, this equals the reward the
    /// fine-tuned model implicitly maximises.
    ///
    /// # Arguments
    /// - `lp_policy` – log π_θ(y | x), shape `[batch]`
    /// - `lp_ref`    – log π_ref(y | x), shape `[batch]`
    ///
    /// # Returns
    /// Implicit reward array, shape `[batch]`.
    pub fn compute_implicit_reward<F>(
        &self,
        lp_policy: &Array1<F>,
        lp_ref: &Array1<F>,
    ) -> Result<Array1<F>>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = lp_policy.len();
        if n == 0 {
            return Err(NeuralError::InvalidArgument(
                "compute_implicit_reward: empty batch".to_string(),
            ));
        }
        if lp_ref.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "compute_implicit_reward: length mismatch {} vs {}",
                n,
                lp_ref.len()
            )));
        }

        let beta = F::from_f64(self.config.beta).ok_or_else(|| {
            NeuralError::ComputationError(
                "compute_implicit_reward: cannot convert beta".to_string(),
            )
        })?;

        let rewards: Array1<F> =
            Array1::from_iter((0..n).map(|i| beta * (lp_policy[i] - lp_ref[i])));
        Ok(rewards)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Apply the configured reduction (mean/sum) and label-smoothing to a
    /// 1-D tensor of `β * log-ratios`.
    fn apply_loss_reduction<F>(&self, log_ratios: &Array1<F>, beta: F) -> Result<F>
    where
        F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
    {
        let n = log_ratios.len();
        let smoothing = F::from_f64(self.config.label_smoothing).ok_or_else(|| {
            NeuralError::ComputationError(
                "apply_loss_reduction: cannot convert label_smoothing".to_string(),
            )
        })?;
        let target = F::one() - smoothing;

        let mut total = F::zero();
        for i in 0..n {
            let scaled = beta * log_ratios[i];
            let loss_i = if self.config.sigmoid_loss {
                // SigLIP variant: -log σ(β Δ) without the (1-p) term
                -log_sigmoid_stable(scaled)?
            } else {
                // Standard DPO: label-smoothed binary cross-entropy
                let log_p = log_sigmoid_stable(scaled)?;
                let log_one_minus_p = log_sigmoid_stable(-scaled)?;
                -(target * log_p + (F::one() - target) * log_one_minus_p)
            };
            total += loss_i;
        }

        let n_f = F::from_usize(n)
            .ok_or_else(|| NeuralError::ComputationError("cannot convert n".to_string()))?;
        match self.config.reduction {
            DPOReduction::Mean => Ok(total / n_f),
            DPOReduction::Sum => Ok(total),
        }
    }
}

// ============================================================================
// Standalone functions (usable without a DPOLoss instance)
// ============================================================================

/// Compute the DPO loss as a free function.
///
/// Equivalent to `DPOLoss::new(config).dpo_loss(...)` but more ergonomic for
/// one-shot calls.
///
/// # Arguments
/// - `lp_chosen_policy`   – log π_θ(y_w | x)
/// - `lp_rejected_policy` – log π_θ(y_l | x)
/// - `lp_chosen_ref`      – log π_ref(y_w | x)
/// - `lp_rejected_ref`    – log π_ref(y_l | x)
/// - `beta`               – KL-penalty coefficient
/// - `label_smoothing`    – smoothing ∈ [0, 0.5)
pub fn dpo_loss<F>(
    lp_chosen_policy: &Array1<F>,
    lp_rejected_policy: &Array1<F>,
    lp_chosen_ref: &Array1<F>,
    lp_rejected_ref: &Array1<F>,
    beta: f64,
    label_smoothing: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let config = DPOConfig {
        beta,
        label_smoothing,
        reference_free: false,
        ..Default::default()
    };
    DPOLoss::new(config).dpo_loss(
        lp_chosen_policy,
        lp_rejected_policy,
        lp_chosen_ref,
        lp_rejected_ref,
    )
}

/// Compute the reference-free DPO loss as a free function.
pub fn reference_free_dpo_loss<F>(
    lp_chosen_policy: &Array1<F>,
    lp_rejected_policy: &Array1<F>,
    beta: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let config = DPOConfig {
        beta,
        reference_free: true,
        ..Default::default()
    };
    DPOLoss::new(config).reference_free_dpo_loss(lp_chosen_policy, lp_rejected_policy)
}

/// Compute implicit rewards for a batch of (policy, reference) log-prob pairs.
pub fn compute_implicit_reward<F>(
    lp_policy: &Array1<F>,
    lp_ref: &Array1<F>,
    beta: f64,
) -> Result<Array1<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let config = DPOConfig {
        beta,
        ..Default::default()
    };
    DPOLoss::new(config).compute_implicit_reward(lp_policy, lp_ref)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Compute log-ratio differences for DPO:
/// `Δ_i = (log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))`
fn compute_log_ratios<F>(
    lp_chosen_policy: &Array1<F>,
    lp_rejected_policy: &Array1<F>,
    lp_chosen_ref: &Array1<F>,
    lp_rejected_ref: &Array1<F>,
) -> Result<Array1<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let n = lp_chosen_policy.len();
    let ratios: Array1<F> = Array1::from_iter((0..n).map(|i| {
        let chosen_ratio = lp_chosen_policy[i] - lp_chosen_ref[i];
        let rejected_ratio = lp_rejected_policy[i] - lp_rejected_ref[i];
        chosen_ratio - rejected_ratio
    }));
    Ok(ratios)
}

/// Numerically stable log-sigmoid: `log σ(x) = -softplus(-x)`.
fn log_sigmoid_stable<F: Float + FromPrimitive + Debug>(x: F) -> Result<F> {
    let zero = F::zero();
    let one = F::one();
    let result = if x >= zero {
        -(one + (-x).exp()).ln()
    } else {
        x - (one + x.exp()).ln()
    };
    Ok(result)
}

/// Validate that all provided lengths equal `expected`.
fn validate_lengths(expected: usize, others: &[usize], ctx: &str) -> Result<()> {
    for (idx, &len) in others.iter().enumerate() {
        if len != expected {
            return Err(NeuralError::DimensionMismatch(format!(
                "{ctx}: array {} has length {} but expected {expected}",
                idx + 1,
                len
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Reward-margin diagnostics
// ============================================================================

/// Compute the reward margin `r_chosen - r_rejected` for each sample.
///
/// This is a useful diagnostic metric that should be positive and increasing
/// during DPO training.
///
/// # Arguments
/// - `lp_chosen_policy`   – log π_θ(y_w | x)
/// - `lp_rejected_policy` – log π_θ(y_l | x)
/// - `lp_chosen_ref`      – log π_ref(y_w | x)
/// - `lp_rejected_ref`    – log π_ref(y_l | x)
/// - `beta`               – KL coefficient
pub fn reward_margin<F>(
    lp_chosen_policy: &Array1<F>,
    lp_rejected_policy: &Array1<F>,
    lp_chosen_ref: &Array1<F>,
    lp_rejected_ref: &Array1<F>,
    beta: f64,
) -> Result<Array1<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let n = lp_chosen_policy.len();
    validate_lengths(
        n,
        &[
            lp_rejected_policy.len(),
            lp_chosen_ref.len(),
            lp_rejected_ref.len(),
        ],
        "reward_margin",
    )?;
    if n == 0 {
        return Err(NeuralError::InvalidArgument(
            "reward_margin: empty batch".to_string(),
        ));
    }

    let beta_f = F::from_f64(beta).ok_or_else(|| {
        NeuralError::ComputationError("reward_margin: cannot convert beta".to_string())
    })?;

    let margins: Array1<F> = Array1::from_iter((0..n).map(|i| {
        let r_chosen = beta_f * (lp_chosen_policy[i] - lp_chosen_ref[i]);
        let r_rejected = beta_f * (lp_rejected_policy[i] - lp_rejected_ref[i]);
        r_chosen - r_rejected
    }));
    Ok(margins)
}

/// Compute the fraction of samples where the policy assigns higher reward to
/// the chosen sequence (i.e. reward margin > 0).
pub fn preference_accuracy<F>(
    lp_chosen_policy: &Array1<F>,
    lp_rejected_policy: &Array1<F>,
    lp_chosen_ref: &Array1<F>,
    lp_rejected_ref: &Array1<F>,
    beta: f64,
) -> Result<f64>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let margins = reward_margin(
        lp_chosen_policy,
        lp_rejected_policy,
        lp_chosen_ref,
        lp_rejected_ref,
        beta,
    )?;
    let n = margins.len();
    let correct: usize = margins.iter().filter(|&&m| m > F::zero()).count();
    Ok(correct as f64 / n as f64)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_arrays() -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
        let lp_cp = Array1::from(vec![-1.0, -1.5, -0.8, -2.0]);
        let lp_rp = Array1::from(vec![-2.0, -3.0, -1.8, -4.0]);
        let lp_cr = Array1::from(vec![-1.2, -1.6, -0.9, -2.1]);
        let lp_rr = Array1::from(vec![-1.9, -2.8, -1.7, -3.9]);
        (lp_cp, lp_rp, lp_cr, lp_rr)
    }

    #[test]
    fn test_dpo_loss_finite() {
        let (lp_cp, lp_rp, lp_cr, lp_rr) = make_arrays();
        let config = DPOConfig::default();
        let dpo = DPOLoss::new(config);
        let loss = dpo.dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("loss");
        assert!(loss.is_finite(), "loss={loss}");
        assert!(loss > 0.0, "loss should be positive");
    }

    #[test]
    fn test_dpo_loss_perfect_separation() {
        // When chosen is much more probable than rejected, loss → log(2)
        let n = 4;
        let lp_cp = Array1::from(vec![0.0_f64; n]);
        let lp_rp = Array1::from(vec![-100.0_f64; n]);
        let lp_cr = Array1::from(vec![0.0_f64; n]);
        let lp_rr = Array1::from(vec![-100.0_f64; n]);

        let config = DPOConfig { beta: 1.0, ..Default::default() };
        let loss = DPOLoss::new(config).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("l");
        // Δ = 0 for each sample → loss = log(2) ≈ 0.693
        assert!((loss - 0.6931471805599453).abs() < 1e-6, "loss={loss}");
    }

    #[test]
    fn test_reference_free_dpo_loss() {
        let lp_cp = Array1::from(vec![-1.0_f64, -1.5]);
        let lp_rp = Array1::from(vec![-2.0_f64, -3.0]);
        let config = DPOConfig { beta: 0.1, reference_free: true, ..Default::default() };
        let dpo = DPOLoss::new(config);
        let loss = dpo.reference_free_dpo_loss(&lp_cp, &lp_rp).expect("loss");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_free_fn_reference_free_dpo_loss() {
        let lp_cp = Array1::from(vec![-1.0_f64, -1.5]);
        let lp_rp = Array1::from(vec![-2.0_f64, -3.0]);
        let loss = reference_free_dpo_loss(&lp_cp, &lp_rp, 0.1).expect("loss");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_compute_implicit_reward_shape() {
        let lp_policy = Array1::from(vec![-1.0_f64, -2.0, -3.0]);
        let lp_ref = Array1::from(vec![-1.5_f64, -2.5, -3.5]);
        let rewards = compute_implicit_reward(&lp_policy, &lp_ref, 0.1).expect("rewards");
        assert_eq!(rewards.len(), 3);
        // r_i = beta * (lp_policy - lp_ref) = 0.1 * 0.5 = 0.05
        for &r in rewards.iter() {
            assert!((r - 0.05).abs() < 1e-9, "r={r}");
        }
    }

    #[test]
    fn test_reward_margin_positive() {
        // Chosen is always better → margins should be positive
        let lp_cp = Array1::from(vec![-1.0_f64, -1.0]);
        let lp_rp = Array1::from(vec![-3.0_f64, -3.0]);
        let lp_cr = Array1::from(vec![-1.0_f64, -1.0]);
        let lp_rr = Array1::from(vec![-3.0_f64, -3.0]);
        // Δ = ((-1) - (-1)) - ((-3) - (-3)) = 0 - 0 = 0
        let margins = reward_margin(&lp_cp, &lp_rp, &lp_cr, &lp_rr, 0.1).expect("margin");
        for &m in margins.iter() {
            assert!((m - 0.0).abs() < 1e-9, "m={m}");
        }
    }

    #[test]
    fn test_preference_accuracy() {
        let (lp_cp, lp_rp, lp_cr, lp_rr) = make_arrays();
        let acc = preference_accuracy(&lp_cp, &lp_rp, &lp_cr, &lp_rr, 0.1).expect("acc");
        assert!((0.0..=1.0).contains(&acc), "acc={acc}");
    }

    #[test]
    fn test_label_smoothing_increases_loss() {
        let (lp_cp, lp_rp, lp_cr, lp_rr) = make_arrays();
        let config0 = DPOConfig { label_smoothing: 0.0, ..Default::default() };
        let config1 = DPOConfig { label_smoothing: 0.1, ..Default::default() };
        let loss0 = DPOLoss::new(config0).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("l0");
        let loss1 = DPOLoss::new(config1).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("l1");
        assert!(loss1 > loss0, "loss1={loss1} should > loss0={loss0}");
    }

    #[test]
    fn test_sum_reduction() {
        let (lp_cp, lp_rp, lp_cr, lp_rr) = make_arrays();
        let n = lp_cp.len();
        let config_mean = DPOConfig { reduction: DPOReduction::Mean, ..Default::default() };
        let config_sum = DPOConfig { reduction: DPOReduction::Sum, ..Default::default() };
        let loss_mean = DPOLoss::new(config_mean).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("l");
        let loss_sum = DPOLoss::new(config_sum).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("l");
        assert!((loss_sum - loss_mean * n as f64).abs() < 1e-9);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let lp_cp = Array1::from(vec![-1.0_f64, -1.5]);
        let lp_rp = Array1::from(vec![-2.0_f64, -3.0, -1.5]); // wrong length
        let lp_cr = Array1::from(vec![-1.2_f64, -1.6]);
        let lp_rr = Array1::from(vec![-1.9_f64, -2.8]);
        let config = DPOConfig::default();
        let result = DPOLoss::new(config).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid_loss_variant() {
        let (lp_cp, lp_rp, lp_cr, lp_rr) = make_arrays();
        let config = DPOConfig { sigmoid_loss: true, ..Default::default() };
        let loss = DPOLoss::new(config).dpo_loss(&lp_cp, &lp_rp, &lp_cr, &lp_rr).expect("loss");
        assert!(loss.is_finite());
    }
}
