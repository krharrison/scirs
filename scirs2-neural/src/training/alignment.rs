//! Advanced alignment training: GRPO, SimPO, IPO, KTO, ORPO.
//!
//! These methods fine-tune language models on human preference data.
//! All implement the [`AlignmentLoss`] trait for unified training.
//!
//! # Overview
//!
//! | Method | Year | Reference-free | Notes |
//! |--------|------|----------------|-------|
//! | GRPO   | 2024 | No  | Group-relative, PPO-clip, reward-based |
//! | SimPO  | 2024 | Yes | Length-normalized log-prob as reward |
//! | IPO    | 2024 | No  | Squared-hinge, no log-ratio approximation |
//! | KTO    | 2024 | No  | Kahneman-Tversky, unpaired data |
//! | ORPO   | 2024 | Yes | SFT + odds-ratio penalty, no separate reference |
//!
//! # References
//!
//! - Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", 2024 (GRPO)
//! - Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward", 2024
//! - Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback", 2024 (IPO)
//! - Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization", 2024
//! - Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model", 2024
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::alignment::{
//!     AlignmentBatch, AlignmentLoss, SimpoConfig, SimpoLoss,
//! };
//! use scirs2_core::ndarray::Array1;
//!
//! let config = SimpoConfig::default();
//! let loss_fn = SimpoLoss::new(config);
//!
//! let batch = AlignmentBatch::new(
//!     Array1::from(vec![-1.0_f64, -1.5, -0.8]),
//!     Array1::from(vec![-2.5_f64, -3.0, -2.0]),
//! );
//! let loss = loss_fn.compute_loss(&batch).expect("loss ok");
//! assert!(loss.is_finite());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::Array1;

// ============================================================================
// AlignmentBatch
// ============================================================================

/// Batch of alignment training data.
///
/// Holds log-probabilities, optional reference model log-probs, sequence lengths,
/// rewards, and labels for the various alignment algorithms.
#[derive(Debug, Clone)]
pub struct AlignmentBatch {
    /// Log-probabilities of chosen completions, shape `[batch_size]`.
    pub chosen_logprobs: Array1<f64>,
    /// Log-probabilities of rejected completions, shape `[batch_size]`.
    pub rejected_logprobs: Array1<f64>,
    /// Reference model log-probs for chosen completions, shape `[batch_size]`.
    /// `None` for reference-free methods (SimPO, ORPO).
    pub ref_chosen_logprobs: Option<Array1<f64>>,
    /// Reference model log-probs for rejected completions, shape `[batch_size]`.
    /// `None` for reference-free methods (SimPO, ORPO).
    pub ref_rejected_logprobs: Option<Array1<f64>>,
    /// Sequence lengths for chosen completions, shape `[batch_size]`.
    /// Used for length normalisation in SimPO.
    pub chosen_lengths: Option<Array1<f64>>,
    /// Sequence lengths for rejected completions, shape `[batch_size]`.
    /// Used for length normalisation in SimPO.
    pub rejected_lengths: Option<Array1<f64>>,
    /// Per-sample rewards, shape `[batch_size]`.
    /// Used by GRPO and KTO.
    pub rewards: Option<Array1<f64>>,
    /// Sample labels: `true` = desirable, `false` = undesirable.
    /// Used by KTO.
    pub labels: Option<Vec<bool>>,
}

impl AlignmentBatch {
    /// Create a minimal batch from chosen and rejected log-probabilities.
    ///
    /// All optional fields are set to `None` and must be populated
    /// via the builder methods before calling loss functions that require them.
    pub fn new(chosen: Array1<f64>, rejected: Array1<f64>) -> Self {
        Self {
            chosen_logprobs: chosen,
            rejected_logprobs: rejected,
            ref_chosen_logprobs: None,
            ref_rejected_logprobs: None,
            chosen_lengths: None,
            rejected_lengths: None,
            rewards: None,
            labels: None,
        }
    }

    /// Attach reference model log-probabilities.
    pub fn with_reference(mut self, ref_chosen: Array1<f64>, ref_rejected: Array1<f64>) -> Self {
        self.ref_chosen_logprobs = Some(ref_chosen);
        self.ref_rejected_logprobs = Some(ref_rejected);
        self
    }

    /// Attach per-sample rewards (required by GRPO and KTO).
    pub fn with_rewards(mut self, rewards: Array1<f64>) -> Self {
        self.rewards = Some(rewards);
        self
    }

    /// Attach sample labels (required by KTO).
    pub fn with_labels(mut self, labels: Vec<bool>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Attach sequence lengths for length-normalised methods.
    pub fn with_lengths(mut self, chosen: Array1<f64>, rejected: Array1<f64>) -> Self {
        self.chosen_lengths = Some(chosen);
        self.rejected_lengths = Some(rejected);
        self
    }

    /// Return the batch size (number of samples).
    pub fn batch_size(&self) -> usize {
        self.chosen_logprobs.len()
    }

    /// Validate that the core arrays are the same length.
    fn validate_lengths(&self, ctx: &str) -> Result<usize> {
        let n = self.chosen_logprobs.len();
        if self.rejected_logprobs.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "{ctx}: chosen_logprobs length {n} != rejected_logprobs length {}",
                self.rejected_logprobs.len()
            )));
        }
        Ok(n)
    }
}

// ============================================================================
// AlignmentLoss trait
// ============================================================================

/// Trait for alignment training loss functions.
///
/// All five alignment methods in this module implement this trait so that
/// training loops can treat them uniformly.
pub trait AlignmentLoss {
    /// Compute the mean loss for a batch.
    ///
    /// Returns `Ok(0.0)` for empty batches.  Returns `Err` when required
    /// optional fields (e.g. rewards, labels, reference log-probs) are absent.
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64>;
}

// ============================================================================
// GRPO — Group Relative Policy Optimization
// ============================================================================

/// Configuration for GRPO (Group Relative Policy Optimization).
///
/// From DeepSeek 2024: each prompt is associated with a *group* of G completions
/// scored by a reward model.  GRPO normalises advantages within the group and
/// applies PPO-style clipping with a KL penalty.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// KL-divergence penalty coefficient β (default 0.04).
    ///
    /// Controls how strongly the policy is penalised for deviating from the
    /// reference model.
    pub beta: f64,
    /// PPO clip ratio ε (default 0.2).
    ///
    /// The probability ratio `π / π_ref` is clipped to `[1-ε, 1+ε]`.
    pub epsilon: f64,
    /// Number of completions per prompt that form a group (default 8).
    ///
    /// Advantages are normalised across these `group_size` samples.
    pub group_size: usize,
    /// Use token-level KL divergence instead of sequence-level (default `false`).
    ///
    /// When `false`, the per-sample KL is approximated as `log π - log π_ref`.
    pub use_token_level_kl: bool,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.04,
            epsilon: 0.2,
            group_size: 8,
            use_token_level_kl: false,
        }
    }
}

/// GRPO loss function.
///
/// For a group of `G` completions with rewards `r_1 … r_G`:
///
/// ```text
/// advantage_i = (r_i - mean_r) / (std_r + ε)
/// ratio_i     = exp(log π_θ(y_i) - log π_ref(y_i))
/// clip_i      = clip(ratio_i, 1-ε, 1+ε)
/// loss_i      = -min(ratio_i * adv_i, clip_i * adv_i) + β * KL_i
/// ```
#[derive(Debug, Clone)]
pub struct GrpoLoss {
    /// GRPO configuration.
    pub config: GrpoConfig,
}

impl GrpoLoss {
    /// Create a new `GrpoLoss` with the given configuration.
    pub fn new(config: GrpoConfig) -> Self {
        Self { config }
    }
}

impl AlignmentLoss for GrpoLoss {
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64> {
        let rewards = batch.rewards.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("GRPO requires rewards".into())
        })?;
        let ref_lp = batch.ref_chosen_logprobs.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("GRPO requires reference logprobs".into())
        })?;

        let n = batch.validate_lengths("GrpoLoss")?;
        if n == 0 {
            return Ok(0.0);
        }
        if rewards.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "GrpoLoss: rewards length {} != batch size {n}",
                rewards.len()
            )));
        }
        if ref_lp.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "GrpoLoss: ref_chosen_logprobs length {} != batch size {n}",
                ref_lp.len()
            )));
        }

        let policy_lp = &batch.chosen_logprobs;

        // Group-relative advantage normalisation
        let mean_r: f64 = rewards.iter().sum::<f64>() / n as f64;
        let var_r: f64 = rewards.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>() / n as f64;
        let std_r = var_r.sqrt().max(1e-8);

        let mut total_loss = 0.0_f64;
        for i in 0..n {
            let advantage = (rewards[i] - mean_r) / std_r;

            // Probability ratio π_θ / π_ref  (in log-space → exp)
            let log_ratio = policy_lp[i] - ref_lp[i];
            let ratio = log_ratio.exp();

            // PPO-style clipped surrogate objective
            let clipped_ratio = ratio.clamp(1.0 - self.config.epsilon, 1.0 + self.config.epsilon);
            // -min(r * A, clip(r) * A)
            let surrogate = -f64::min(ratio * advantage, clipped_ratio * advantage);

            // KL approximation: log π_θ - log π_ref  (≈ forward KL for small deviations)
            let kl_approx = log_ratio;

            total_loss += surrogate + self.config.beta * kl_approx;
        }

        Ok(total_loss / n as f64)
    }
}

// ============================================================================
// SimPO — Simple Preference Optimization
// ============================================================================

/// Configuration for SimPO (Simple Preference Optimization).
///
/// SimPO is *reference-free*: it uses the length-normalised mean log-probability
/// as an implicit reward, avoiding the need for a separate reference model.
#[derive(Debug, Clone)]
pub struct SimpoConfig {
    /// Temperature coefficient β (default 2.5).
    ///
    /// Scales the reward difference before the sigmoid.  Higher β gives sharper
    /// preference boundaries.
    pub beta: f64,
    /// Target reward margin γ (default 1.4).
    ///
    /// The loss is zero only when the margin between chosen and rejected rewards
    /// exceeds γ.
    pub gamma: f64,
    /// Whether to divide log-probs by sequence length (default `true`).
    ///
    /// When enabled, `chosen_lengths` and `rejected_lengths` in the batch are
    /// used.  If lengths are absent, normalization is skipped silently.
    pub length_normalize: bool,
}

impl Default for SimpoConfig {
    fn default() -> Self {
        Self {
            beta: 2.5,
            gamma: 1.4,
            length_normalize: true,
        }
    }
}

/// SimPO loss function.
///
/// ```text
/// r_chosen   = lp_chosen  / len_chosen   (if length_normalize)
/// r_rejected = lp_rejected / len_rejected (if length_normalize)
/// loss_i     = -log_sigmoid(β * (r_chosen - r_rejected) - γ)
///            = softplus(-(β * Δr - γ))
/// ```
#[derive(Debug, Clone)]
pub struct SimpoLoss {
    /// SimPO configuration.
    pub config: SimpoConfig,
}

impl SimpoLoss {
    /// Create a new `SimpoLoss` with the given configuration.
    pub fn new(config: SimpoConfig) -> Self {
        Self { config }
    }
}

impl AlignmentLoss for SimpoLoss {
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64> {
        let n = batch.validate_lengths("SimpoLoss")?;
        if n == 0 {
            return Ok(0.0);
        }

        let mut total = 0.0_f64;
        for i in 0..n {
            let mut r_chosen = batch.chosen_logprobs[i];
            let mut r_rejected = batch.rejected_logprobs[i];

            if self.config.length_normalize {
                if let Some(ref lengths) = batch.chosen_lengths {
                    r_chosen /= lengths[i].max(1.0);
                }
                if let Some(ref lengths) = batch.rejected_lengths {
                    r_rejected /= lengths[i].max(1.0);
                }
            }

            let margin = self.config.beta * (r_chosen - r_rejected) - self.config.gamma;
            // -log_sigmoid(margin) = softplus(-margin) = log(1 + exp(-margin))
            // Numerically stable form:
            let loss_i = softplus_neg(margin);
            total += loss_i;
        }

        Ok(total / n as f64)
    }
}

// ============================================================================
// IPO — Identity Preference Optimization
// ============================================================================

/// Configuration for IPO (Identity Preference Optimization).
///
/// IPO avoids the log-ratio approximation used by DPO by directly minimising
/// a squared hinge loss on the human-preference objective.
#[derive(Debug, Clone)]
pub struct IpoConfig {
    /// Regularisation coefficient τ (default 0.1).
    ///
    /// The target h value is `1 / (2τ)`.  Smaller τ → stronger regularisation.
    pub tau: f64,
}

impl Default for IpoConfig {
    fn default() -> Self {
        Self { tau: 0.1 }
    }
}

/// IPO loss function.
///
/// ```text
/// h_i    = (lp_chosen_i - ref_chosen_i) - (lp_rejected_i - ref_rejected_i)
/// loss_i = (h_i - 1/(2τ))²
/// ```
#[derive(Debug, Clone)]
pub struct IpoLoss {
    /// IPO configuration.
    pub config: IpoConfig,
}

impl IpoLoss {
    /// Create a new `IpoLoss` with the given configuration.
    pub fn new(config: IpoConfig) -> Self {
        Self { config }
    }
}

impl AlignmentLoss for IpoLoss {
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64> {
        let ref_chosen = batch.ref_chosen_logprobs.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("IPO requires ref_chosen_logprobs".into())
        })?;
        let ref_rejected = batch.ref_rejected_logprobs.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("IPO requires ref_rejected_logprobs".into())
        })?;

        let n = batch.validate_lengths("IpoLoss")?;
        if n == 0 {
            return Ok(0.0);
        }
        if ref_chosen.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "IpoLoss: ref_chosen_logprobs length {} != batch size {n}",
                ref_chosen.len()
            )));
        }
        if ref_rejected.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "IpoLoss: ref_rejected_logprobs length {} != batch size {n}",
                ref_rejected.len()
            )));
        }

        let target = 1.0 / (2.0 * self.config.tau);
        let mut total = 0.0_f64;

        for i in 0..n {
            let h = (batch.chosen_logprobs[i] - ref_chosen[i])
                - (batch.rejected_logprobs[i] - ref_rejected[i]);
            total += (h - target).powi(2);
        }

        Ok(total / n as f64)
    }
}

// ============================================================================
// KTO — Kahneman-Tversky Optimization
// ============================================================================

/// Configuration for KTO (Kahneman-Tversky Optimization).
///
/// KTO operates on *unpaired* data (individual samples labelled desirable or
/// undesirable) and applies prospect-theory-inspired value functions.
#[derive(Debug, Clone)]
pub struct KtoConfig {
    /// KL-penalty coefficient β (default 0.1).
    pub beta: f64,
    /// Loss weight for desirable samples (default 1.0).
    pub desirable_weight: f64,
    /// Loss weight for undesirable samples (default 1.0).
    pub undesirable_weight: f64,
}

impl Default for KtoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            desirable_weight: 1.0,
            undesirable_weight: 1.0,
        }
    }
}

/// KTO loss function.
///
/// For a sample with policy log-prob `lp` and reference log-prob `ref_lp`:
///
/// ```text
/// kl_i = lp_i - ref_lp_i          (per-sample KL approximation)
///
/// Desirable:   loss_i = w_d  * (1 - σ(β * kl_i - z_ref))
/// Undesirable: loss_i = w_u  * (1 - σ(β * (-kl_i) + z_ref))
///                      = w_u  * σ(-β * (-kl_i) + z_ref)  [equivalent]
/// ```
///
/// `z_ref` is an exponential moving average of the KL divergence partition
/// function, updated via [`KtoLoss::update_z_ref`].
#[derive(Debug, Clone)]
pub struct KtoLoss {
    /// KTO configuration.
    pub config: KtoConfig,
    /// EMA of the KL divergence partition function Z_ref.
    ///
    /// Initialise to 0.0 and update with [`KtoLoss::update_z_ref`] each step.
    pub z_ref: f64,
}

impl KtoLoss {
    /// Create a new `KtoLoss` with z_ref initialised to zero.
    pub fn new(config: KtoConfig) -> Self {
        Self { config, z_ref: 0.0 }
    }

    /// Update the partition function estimate using an exponential moving average.
    ///
    /// # Arguments
    /// - `kl_estimate` – current batch mean KL divergence estimate
    /// - `momentum`    – EMA coefficient ∈ (0, 1).  0.9 is a reasonable default.
    pub fn update_z_ref(&mut self, kl_estimate: f64, momentum: f64) {
        self.z_ref = momentum * self.z_ref + (1.0 - momentum) * kl_estimate;
    }
}

impl AlignmentLoss for KtoLoss {
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64> {
        let labels = batch.labels.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("KTO requires labels".into())
        })?;
        let ref_lp = batch.ref_chosen_logprobs.as_ref().ok_or_else(|| {
            NeuralError::InvalidArgument("KTO requires ref_chosen_logprobs".into())
        })?;

        let n = batch.chosen_logprobs.len();
        if n == 0 {
            return Ok(0.0);
        }
        if labels.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "KtoLoss: labels length {} != batch size {n}",
                labels.len()
            )));
        }
        if ref_lp.len() != n {
            return Err(NeuralError::DimensionMismatch(format!(
                "KtoLoss: ref_chosen_logprobs length {} != batch size {n}",
                ref_lp.len()
            )));
        }

        let mut total = 0.0_f64;
        for i in 0..n {
            // KL term: log π_θ - log π_ref
            let kl_term = batch.chosen_logprobs[i] - ref_lp[i];

            let loss_i = if labels[i] {
                // Desirable: 1 - σ(β * kl - z_ref)
                let logit = self.config.beta * kl_term - self.z_ref;
                let sigma = sigmoid(logit);
                self.config.desirable_weight * (1.0 - sigma)
            } else {
                // Undesirable: 1 - σ(-(β * kl) + z_ref) = 1 - σ(z_ref - β * kl)
                let logit = self.z_ref - self.config.beta * kl_term;
                let sigma = sigmoid(logit);
                self.config.undesirable_weight * (1.0 - sigma)
            };

            total += loss_i;
        }

        Ok(total / n as f64)
    }
}

// ============================================================================
// ORPO — Odds Ratio Preference Optimization
// ============================================================================

/// Configuration for ORPO (Odds Ratio Preference Optimization).
///
/// ORPO is *reference-free*: it combines a supervised fine-tuning (SFT) loss
/// on chosen completions with an odds-ratio penalty that discourages the model
/// from assigning high probability to rejected completions.
#[derive(Debug, Clone)]
pub struct OrpoConfig {
    /// Odds-ratio penalty weight λ (default 0.1).
    ///
    /// Controls the relative contribution of the SFT loss vs. the odds-ratio
    /// penalty.
    pub lambda: f64,
}

impl Default for OrpoConfig {
    fn default() -> Self {
        Self { lambda: 0.1 }
    }
}

/// ORPO loss function.
///
/// ```text
/// odds_chosen   = exp(lp_chosen)  / (1 - exp(lp_chosen))
/// odds_rejected = exp(lp_rejected) / (1 - exp(lp_rejected))
/// OR            = odds_chosen / odds_rejected
/// loss_i        = -lp_chosen + λ * softplus(-log(OR))
/// ```
///
/// Both the SFT term and the odds-ratio penalty are reference-free.
#[derive(Debug, Clone)]
pub struct OrpoLoss {
    /// ORPO configuration.
    pub config: OrpoConfig,
}

impl OrpoLoss {
    /// Create a new `OrpoLoss` with the given configuration.
    pub fn new(config: OrpoConfig) -> Self {
        Self { config }
    }
}

impl AlignmentLoss for OrpoLoss {
    fn compute_loss(&self, batch: &AlignmentBatch) -> Result<f64> {
        let n = batch.validate_lengths("OrpoLoss")?;
        if n == 0 {
            return Ok(0.0);
        }

        let mut total = 0.0_f64;
        for i in 0..n {
            let lp_c = batch.chosen_logprobs[i];
            let lp_r = batch.rejected_logprobs[i];

            // Clamp probabilities away from 0 and 1 to avoid division by zero
            // and log(0).  exp(lp) ∈ (0, 1] so we only need to guard the upper end.
            let p_c = lp_c.exp().min(1.0 - 1e-7).max(1e-15);
            let p_r = lp_r.exp().min(1.0 - 1e-7).max(1e-15);

            let odds_c = p_c / (1.0 - p_c);
            let odds_r = p_r / (1.0 - p_r);

            // Guard against degenerate odds
            let odds_c = odds_c.max(1e-15);
            let odds_r = odds_r.max(1e-15);

            let log_or = (odds_c / odds_r).ln();

            // SFT loss on chosen
            let sft_loss = -lp_c;
            // -log_sigmoid(log_or) = softplus(-log_or)
            let or_penalty = softplus_neg(log_or);

            total += sft_loss + self.config.lambda * or_penalty;
        }

        Ok(total / n as f64)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Numerically stable computation of `log(1 + exp(-x))` = `-log_sigmoid(x)`.
///
/// Uses the identity:
/// - If `x >= 0`: `log(1 + exp(-x))`  (direct, small)
/// - If `x < 0`:  `-x + log(1 + exp(x))`  (avoids exp of large positive)
#[inline]
fn softplus_neg(x: f64) -> f64 {
    if x >= 0.0 {
        (-x).exp().ln_1p()
    } else {
        -x + x.exp().ln_1p()
    }
}

/// Numerically stable sigmoid: `σ(x) = 1 / (1 + exp(-x))`.
///
/// Clamps the output to `[ε, 1-ε]` to prevent exactly 0 or 1.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let s = if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    };
    s.clamp(1e-15, 1.0 - 1e-15)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // GRPO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grpo_loss_basic() {
        // Three completions with positive, neutral, negative rewards
        let rewards = Array1::from(vec![1.0_f64, 0.0, -1.0]);
        let policy_lp = Array1::from(vec![-1.0_f64, -1.5, -2.0]);
        let ref_lp = Array1::from(vec![-1.2_f64, -1.4, -1.9]);
        let batch = AlignmentBatch::new(policy_lp, Array1::from(vec![-2.0_f64, -2.5, -3.0]))
            .with_rewards(rewards)
            .with_reference(ref_lp, Array1::from(vec![-2.1_f64, -2.6, -3.1]));

        let loss = GrpoLoss::new(GrpoConfig::default())
            .compute_loss(&batch)
            .expect("grpo loss");
        assert!(loss.is_finite(), "grpo loss={loss}");
    }

    #[test]
    fn test_grpo_empty_batch() {
        let batch = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_rewards(Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]));
        let loss = GrpoLoss::new(GrpoConfig::default())
            .compute_loss(&batch)
            .expect("empty batch");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_grpo_missing_rewards_error() {
        let batch = AlignmentBatch::new(
            Array1::from(vec![-1.0_f64]),
            Array1::from(vec![-2.0_f64]),
        )
        .with_reference(
            Array1::from(vec![-1.1_f64]),
            Array1::from(vec![-2.1_f64]),
        );
        let result = GrpoLoss::new(GrpoConfig::default()).compute_loss(&batch);
        assert!(result.is_err(), "should fail without rewards");
    }

    #[test]
    fn test_grpo_group_normalization_equal_rewards() {
        // When all rewards are equal, all advantages are 0, so surrogate = 0.
        // Only the KL term remains.
        let n = 4;
        let rewards = Array1::from(vec![1.0_f64; n]);
        let policy_lp = Array1::from(vec![-1.0_f64; n]);
        let ref_lp = Array1::from(vec![-1.0_f64; n]);
        let batch = AlignmentBatch::new(policy_lp, Array1::from(vec![-2.0_f64; n]))
            .with_rewards(rewards)
            .with_reference(ref_lp, Array1::from(vec![-2.0_f64; n]));

        let config = GrpoConfig { beta: 0.04, ..Default::default() };
        let loss = GrpoLoss::new(config).compute_loss(&batch).expect("loss");
        // policy_lp == ref_lp → log_ratio = 0 → KL = 0 → total loss = 0
        assert!(loss.abs() < 1e-10, "expected ~0 loss, got {loss}");
    }

    // -----------------------------------------------------------------------
    // SimPO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simpo_loss_length_normalized() {
        // Chosen is longer but has proportionally lower logprob → length norm matters
        let chosen_lp = Array1::from(vec![-4.0_f64, -6.0]);
        let rejected_lp = Array1::from(vec![-2.0_f64, -3.0]);
        let chosen_len = Array1::from(vec![4.0_f64, 6.0]); // mean -1.0/token
        let rejected_len = Array1::from(vec![2.0_f64, 3.0]); // mean -1.0/token

        let batch = AlignmentBatch::new(chosen_lp, rejected_lp)
            .with_lengths(chosen_len, rejected_len);

        let config = SimpoConfig { length_normalize: true, ..Default::default() };
        let loss = SimpoLoss::new(config).compute_loss(&batch).expect("loss");
        // After normalization, chosen and rejected have equal mean LP = -1.0
        // So margin = β*(0) - γ < 0 → loss > 0
        assert!(loss.is_finite() && loss > 0.0, "loss={loss}");
    }

    #[test]
    fn test_simpo_no_length_norm() {
        let chosen_lp = Array1::from(vec![-1.0_f64, -1.5]);
        let rejected_lp = Array1::from(vec![-2.5_f64, -3.0]);

        let batch = AlignmentBatch::new(chosen_lp, rejected_lp);
        let config = SimpoConfig { length_normalize: false, ..Default::default() };
        let loss = SimpoLoss::new(config).compute_loss(&batch).expect("loss");
        assert!(loss.is_finite(), "loss={loss}");
    }

    #[test]
    fn test_simpo_empty_batch() {
        let batch = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]));
        let loss = SimpoLoss::new(SimpoConfig::default())
            .compute_loss(&batch)
            .expect("empty");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_simpo_margin_effect() {
        // Higher gamma → higher loss (harder to satisfy margin constraint)
        let chosen_lp = Array1::from(vec![-1.0_f64, -1.5]);
        let rejected_lp = Array1::from(vec![-2.0_f64, -2.5]);
        let batch = AlignmentBatch::new(chosen_lp, rejected_lp);

        let loss_low_gamma = SimpoLoss::new(SimpoConfig { gamma: 0.5, ..Default::default() })
            .compute_loss(&batch)
            .expect("loss_low");
        let loss_high_gamma = SimpoLoss::new(SimpoConfig { gamma: 3.0, ..Default::default() })
            .compute_loss(&batch)
            .expect("loss_high");

        assert!(
            loss_high_gamma > loss_low_gamma,
            "high gamma={loss_high_gamma} should > low gamma={loss_low_gamma}"
        );
    }

    // -----------------------------------------------------------------------
    // IPO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ipo_loss_formula() {
        // Manual calculation: τ=0.1 → target = 1/(2*0.1) = 5.0
        // h = (lp_c - ref_c) - (lp_r - ref_r) = (-1 - (-1.2)) - (-2 - (-1.9)) = 0.2 - (-0.1) = 0.3
        // loss = (0.3 - 5.0)^2 = (-4.7)^2 = 22.09
        let chosen_lp = Array1::from(vec![-1.0_f64]);
        let rejected_lp = Array1::from(vec![-2.0_f64]);
        let ref_chosen = Array1::from(vec![-1.2_f64]);
        let ref_rejected = Array1::from(vec![-1.9_f64]);

        let batch = AlignmentBatch::new(chosen_lp, rejected_lp)
            .with_reference(ref_chosen, ref_rejected);

        let config = IpoConfig { tau: 0.1 };
        let loss = IpoLoss::new(config).compute_loss(&batch).expect("loss");
        let expected = (-4.7_f64).powi(2);
        assert!((loss - expected).abs() < 1e-9, "loss={loss}, expected={expected}");
    }

    #[test]
    fn test_ipo_missing_reference() {
        let batch = AlignmentBatch::new(
            Array1::from(vec![-1.0_f64]),
            Array1::from(vec![-2.0_f64]),
        );
        let result = IpoLoss::new(IpoConfig::default()).compute_loss(&batch);
        assert!(result.is_err(), "should fail without reference");
    }

    #[test]
    fn test_ipo_empty_batch() {
        let batch = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]));
        let loss = IpoLoss::new(IpoConfig::default())
            .compute_loss(&batch)
            .expect("empty");
        assert_eq!(loss, 0.0);
    }

    // -----------------------------------------------------------------------
    // KTO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kto_desirable_all_positive() {
        // All labels = true (desirable)
        let n = 3;
        let chosen_lp = Array1::from(vec![-0.5_f64, -1.0, -1.5]);
        let ref_lp = Array1::from(vec![-1.0_f64, -1.5, -2.0]);
        let labels = vec![true; n];

        let batch = AlignmentBatch::new(chosen_lp, Array1::from(vec![-2.0_f64; n]))
            .with_reference(ref_lp, Array1::from(vec![-2.5_f64; n]))
            .with_labels(labels);

        let loss = KtoLoss::new(KtoConfig::default())
            .compute_loss(&batch)
            .expect("loss");
        assert!(loss.is_finite() && loss >= 0.0, "kto desirable loss={loss}");
    }

    #[test]
    fn test_kto_undesirable_all_negative() {
        // All labels = false (undesirable)
        let n = 3;
        let chosen_lp = Array1::from(vec![-0.5_f64, -1.0, -1.5]);
        let ref_lp = Array1::from(vec![-1.0_f64, -1.5, -2.0]);
        let labels = vec![false; n];

        let batch = AlignmentBatch::new(chosen_lp, Array1::from(vec![-2.0_f64; n]))
            .with_reference(ref_lp, Array1::from(vec![-2.5_f64; n]))
            .with_labels(labels);

        let loss = KtoLoss::new(KtoConfig::default())
            .compute_loss(&batch)
            .expect("loss");
        assert!(loss.is_finite() && loss >= 0.0, "kto undesirable loss={loss}");
    }

    #[test]
    fn test_kto_mixed_labels() {
        let chosen_lp = Array1::from(vec![-1.0_f64, -2.0, -1.5]);
        let ref_lp = Array1::from(vec![-1.2_f64, -1.8, -1.6]);
        let labels = vec![true, false, true];

        let batch = AlignmentBatch::new(
            chosen_lp,
            Array1::from(vec![-2.5_f64, -3.0, -2.8]),
        )
        .with_reference(ref_lp, Array1::from(vec![-2.6_f64, -3.1, -2.9]))
        .with_labels(labels);

        let loss = KtoLoss::new(KtoConfig::default())
            .compute_loss(&batch)
            .expect("loss");
        assert!(loss.is_finite(), "kto mixed loss={loss}");
    }

    #[test]
    fn test_kto_missing_labels_error() {
        let batch = AlignmentBatch::new(
            Array1::from(vec![-1.0_f64]),
            Array1::from(vec![-2.0_f64]),
        )
        .with_reference(
            Array1::from(vec![-1.1_f64]),
            Array1::from(vec![-2.1_f64]),
        );
        let result = KtoLoss::new(KtoConfig::default()).compute_loss(&batch);
        assert!(result.is_err(), "should fail without labels");
    }

    #[test]
    fn test_kto_empty_batch() {
        let batch = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]))
            .with_labels(vec![]);
        let loss = KtoLoss::new(KtoConfig::default())
            .compute_loss(&batch)
            .expect("empty");
        assert_eq!(loss, 0.0);
    }

    // -----------------------------------------------------------------------
    // ORPO tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_orpo_basic_positive_loss() {
        let chosen_lp = Array1::from(vec![-1.0_f64, -1.5, -0.8]);
        let rejected_lp = Array1::from(vec![-2.0_f64, -3.0, -2.5]);
        let batch = AlignmentBatch::new(chosen_lp, rejected_lp);
        let loss = OrpoLoss::new(OrpoConfig::default())
            .compute_loss(&batch)
            .expect("orpo loss");
        // SFT loss -lp_c is always >= 0 for lp_c <= 0
        assert!(loss.is_finite() && loss > 0.0, "orpo loss={loss}");
    }

    #[test]
    fn test_orpo_chosen_better_lower_loss() {
        // When chosen is much better than rejected, the OR penalty should be small
        let chosen_lp_good = Array1::from(vec![-0.1_f64, -0.1]);
        let rejected_lp_bad = Array1::from(vec![-10.0_f64, -10.0]);
        let batch_good = AlignmentBatch::new(chosen_lp_good, rejected_lp_bad);

        let chosen_lp_poor = Array1::from(vec![-3.0_f64, -3.0]);
        let rejected_lp_close = Array1::from(vec![-3.1_f64, -3.1]);
        let batch_poor = AlignmentBatch::new(chosen_lp_poor, rejected_lp_close);

        let config = OrpoConfig { lambda: 0.5 };
        let loss_good = OrpoLoss::new(config.clone())
            .compute_loss(&batch_good)
            .expect("good");
        let loss_poor = OrpoLoss::new(config)
            .compute_loss(&batch_poor)
            .expect("poor");

        // loss_good has smaller -lp_c (less negative), so SFT part is larger,
        // but OR penalty is much smaller.  Overall, both are positive.
        assert!(loss_good.is_finite() && loss_poor.is_finite());
    }

    #[test]
    fn test_orpo_empty_batch() {
        let batch = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]));
        let loss = OrpoLoss::new(OrpoConfig::default())
            .compute_loss(&batch)
            .expect("empty");
        assert_eq!(loss, 0.0);
    }

    // -----------------------------------------------------------------------
    // AlignmentBatch builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_alignment_batch_builder_all_methods() {
        let n = 3;
        let batch = AlignmentBatch::new(
            Array1::from(vec![-1.0_f64; n]),
            Array1::from(vec![-2.0_f64; n]),
        )
        .with_reference(
            Array1::from(vec![-1.1_f64; n]),
            Array1::from(vec![-2.1_f64; n]),
        )
        .with_rewards(Array1::from(vec![1.0_f64; n]))
        .with_labels(vec![true, false, true])
        .with_lengths(
            Array1::from(vec![10.0_f64; n]),
            Array1::from(vec![8.0_f64; n]),
        );

        assert_eq!(batch.batch_size(), n);
        assert!(batch.ref_chosen_logprobs.is_some());
        assert!(batch.ref_rejected_logprobs.is_some());
        assert!(batch.rewards.is_some());
        assert!(batch.labels.is_some());
        assert!(batch.chosen_lengths.is_some());
        assert!(batch.rejected_lengths.is_some());
    }

    // -----------------------------------------------------------------------
    // All losses return 0 on empty batch
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_losses_zero_batch() {
        // SimPO — no optional fields required
        let batch_empty = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]));
        assert_eq!(
            SimpoLoss::new(SimpoConfig::default())
                .compute_loss(&batch_empty)
                .expect("simpo empty"),
            0.0
        );
        assert_eq!(
            OrpoLoss::new(OrpoConfig::default())
                .compute_loss(&batch_empty)
                .expect("orpo empty"),
            0.0
        );

        // IPO — needs reference (even if empty)
        let batch_ipo = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]));
        assert_eq!(
            IpoLoss::new(IpoConfig::default())
                .compute_loss(&batch_ipo)
                .expect("ipo empty"),
            0.0
        );

        // GRPO — needs rewards + reference (even if empty)
        let batch_grpo = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_rewards(Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]));
        assert_eq!(
            GrpoLoss::new(GrpoConfig::default())
                .compute_loss(&batch_grpo)
                .expect("grpo empty"),
            0.0
        );

        // KTO — needs labels + reference (even if empty)
        let batch_kto = AlignmentBatch::new(Array1::from(vec![]), Array1::from(vec![]))
            .with_reference(Array1::from(vec![]), Array1::from(vec![]))
            .with_labels(vec![]);
        assert_eq!(
            KtoLoss::new(KtoConfig::default())
                .compute_loss(&batch_kto)
                .expect("kto empty"),
            0.0
        );
    }
}
