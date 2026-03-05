//! Proximal Policy Optimization (PPO) primitives
//!
//! This module provides the core building blocks for PPO-based RLHF fine-tuning,
//! including rollout buffers, Generalised Advantage Estimation (GAE-λ), the
//! clipped surrogate objective, and the clipped value-function loss.
//!
//! # Overview
//!
//! PPO is an on-policy algorithm that constrains policy updates via a clipped
//! importance-weight ratio:
//!
//! ```text
//! L_CLIP(θ) = E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
//! ```
//!
//! where `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)` and `A_t` is the advantage.
//!
//! The value function is trained with an additional clipped MSE loss:
//!
//! ```text
//! L_VF = 0.5 * E[max((V_θ(s) - V_target)², (clip(V_θ(s), V_old±ε) - V_target)²)]
//! ```
//!
//! An entropy bonus encourages exploration:
//!
//! ```text
//! L = -L_CLIP + c_v L_VF - c_e S[π_θ](s)
//! ```
//!
//! # References
//!
//! - Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
//! - Stiennon et al., "Learning to summarize with human feedback", NeurIPS 2020
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::ppo::{PPOConfig, PPOBuffer, compute_gae};
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! let config = PPOConfig::default();
//! let mut buf = PPOBuffer::<f64>::new(config.clone(), 4, 8);
//!
//! // Push one dummy step
//! let obs = Array1::<f64>::zeros(8);
//! buf.push(obs, 0usize, 1.0, 0.5, -0.2, false).expect("push ok");
//!
//! // Compute advantages
//! let last_value = 0.0_f64;
//! compute_gae(&mut buf, last_value, config.gamma, config.lam).expect("gae ok");
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::Debug;

// ============================================================================
// PPO Configuration
// ============================================================================

/// Hyper-parameters for Proximal Policy Optimisation.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Clipping parameter ε for the policy ratio (default 0.2).
    pub clip_range: f64,
    /// Clipping parameter for the value function (default 0.2, 0.0 = disabled).
    pub clip_range_vf: f64,
    /// Coefficient for the value function loss (default 0.5).
    pub value_coeff: f64,
    /// Coefficient for the entropy bonus (default 0.01).
    pub entropy_coeff: f64,
    /// Discount factor γ (default 0.99).
    pub gamma: f64,
    /// GAE lambda λ (default 0.95).
    pub lam: f64,
    /// Maximum gradient norm for clipping (0.0 = disabled).
    pub max_grad_norm: f64,
    /// Number of PPO epochs (passes over the collected rollout).
    pub n_epochs: usize,
    /// Mini-batch size for each PPO update.
    pub mini_batch_size: usize,
    /// Normalise advantages to zero mean, unit variance.
    pub normalise_advantages: bool,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            clip_range_vf: 0.2,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            gamma: 0.99,
            lam: 0.95,
            max_grad_norm: 0.5,
            n_epochs: 4,
            mini_batch_size: 64,
            normalise_advantages: true,
        }
    }
}

// ============================================================================
// Step-level experience record
// ============================================================================

/// One step of experience collected during a rollout.
#[derive(Debug, Clone)]
pub struct RolloutStep<F: Float + Debug> {
    /// Observation (flattened feature vector).
    pub obs: Array1<F>,
    /// Action taken (as a discrete index; for continuous actions use a separate
    /// `action_vec` field added by the caller).
    pub action: usize,
    /// Observed scalar reward.
    pub reward: F,
    /// Value estimate V(s) from the critic at this step.
    pub value: F,
    /// Log-probability log π_old(a|s) from the behaviour policy.
    pub log_prob: F,
    /// Whether this step terminated the episode.
    pub done: bool,
    /// GAE advantage A_t (filled in by `compute_gae`).
    pub advantage: F,
    /// Return / value target R_t (filled in by `compute_gae`).
    pub returns: F,
}

// ============================================================================
// PPO Rollout Buffer
// ============================================================================

/// Fixed-capacity circular buffer holding rollout experience for PPO.
///
/// After collection, call `compute_gae` to fill `advantage` and `returns`
/// fields, then sample mini-batches for the PPO update.
#[derive(Debug, Clone)]
pub struct PPOBuffer<F: Float + Debug> {
    /// Stored steps (in collection order).
    pub steps: Vec<RolloutStep<F>>,
    /// Maximum capacity.
    pub capacity: usize,
    /// Observation dimension (used for validation).
    pub obs_dim: usize,
    /// PPO configuration snapshot (used by helper methods).
    pub config: PPOConfig,
}

impl<F> PPOBuffer<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive + Clone,
{
    /// Create a new empty buffer with the given capacity and observation dimension.
    pub fn new(config: PPOConfig, capacity: usize, obs_dim: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity),
            capacity,
            obs_dim,
            config,
        }
    }

    /// Return `true` when the buffer has reached its capacity.
    pub fn is_full(&self) -> bool {
        self.steps.len() >= self.capacity
    }

    /// Current number of stored steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Return `true` when the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Append a single experience step to the buffer.
    ///
    /// # Errors
    /// Returns `InvalidArgument` if the buffer is already full or if `obs`
    /// does not match `obs_dim`.
    pub fn push(
        &mut self,
        obs: Array1<F>,
        action: usize,
        reward: f64,
        value: f64,
        log_prob: f64,
        done: bool,
    ) -> Result<()> {
        if self.is_full() {
            return Err(NeuralError::InvalidArgument(
                "PPOBuffer: buffer is full".to_string(),
            ));
        }
        if obs.len() != self.obs_dim {
            return Err(NeuralError::DimensionMismatch(format!(
                "PPOBuffer: obs dim {} != expected {}",
                obs.len(),
                self.obs_dim
            )));
        }

        let reward_f = F::from_f64(reward).ok_or_else(|| {
            NeuralError::ComputationError("PPOBuffer: cannot convert reward".to_string())
        })?;
        let value_f = F::from_f64(value).ok_or_else(|| {
            NeuralError::ComputationError("PPOBuffer: cannot convert value".to_string())
        })?;
        let log_prob_f = F::from_f64(log_prob).ok_or_else(|| {
            NeuralError::ComputationError("PPOBuffer: cannot convert log_prob".to_string())
        })?;

        self.steps.push(RolloutStep {
            obs,
            action,
            reward: reward_f,
            value: value_f,
            log_prob: log_prob_f,
            done,
            advantage: F::zero(),
            returns: F::zero(),
        });
        Ok(())
    }

    /// Reset the buffer, discarding all stored steps.
    pub fn reset(&mut self) {
        self.steps.clear();
    }

    /// Collect observations into a 2-D matrix `[T, obs_dim]`.
    pub fn obs_matrix(&self) -> Result<Array2<F>> {
        let t = self.steps.len();
        if t == 0 {
            return Err(NeuralError::InvalidArgument(
                "obs_matrix: buffer is empty".to_string(),
            ));
        }
        let mut mat = Array2::zeros((t, self.obs_dim));
        for (i, step) in self.steps.iter().enumerate() {
            for j in 0..self.obs_dim {
                mat[[i, j]] = step.obs[j];
            }
        }
        Ok(mat)
    }

    /// Collect advantages into a 1-D array `[T]`.
    pub fn advantages(&self) -> Array1<F> {
        Array1::from_iter(self.steps.iter().map(|s| s.advantage))
    }

    /// Collect returns into a 1-D array `[T]`.
    pub fn returns_array(&self) -> Array1<F> {
        Array1::from_iter(self.steps.iter().map(|s| s.returns))
    }

    /// Collect old log-probabilities into a 1-D array `[T]`.
    pub fn old_log_probs(&self) -> Array1<F> {
        Array1::from_iter(self.steps.iter().map(|s| s.log_prob))
    }

    /// Collect old values (critic estimates) into a 1-D array `[T]`.
    pub fn old_values(&self) -> Array1<F> {
        Array1::from_iter(self.steps.iter().map(|s| s.value))
    }

    /// Normalise advantages to zero mean and unit variance in-place.
    pub fn normalise_advantages(&mut self) -> Result<()> {
        let t = self.steps.len();
        if t < 2 {
            return Ok(());
        }

        let eps = F::from_f64(1e-8).ok_or_else(|| {
            NeuralError::ComputationError("normalise_advantages: cannot convert eps".to_string())
        })?;
        let t_f = F::from_usize(t)
            .ok_or_else(|| NeuralError::ComputationError("cannot convert t".to_string()))?;

        let mut sum = F::zero();
        for s in &self.steps {
            sum += s.advantage;
        }
        let mean = sum / t_f;

        let mut sq_sum = F::zero();
        for s in &self.steps {
            let diff = s.advantage - mean;
            sq_sum += diff * diff;
        }
        let std_dev = (sq_sum / t_f + eps).sqrt();

        for s in &mut self.steps {
            s.advantage = (s.advantage - mean) / std_dev;
        }
        Ok(())
    }
}

// ============================================================================
// Generalised Advantage Estimation
// ============================================================================

/// Compute Generalised Advantage Estimation (GAE-λ) and fill `advantage` and
/// `returns` in each step of `buf`.
///
/// The algorithm is the time-difference version (Schulman et al., 2016):
///
/// ```text
/// δ_t   = r_t + γ V(s_{t+1}) * (1-done) - V(s_t)
/// A_t   = δ_t + γ λ A_{t+1} * (1-done)
/// R_t   = A_t + V(s_t)
/// ```
///
/// # Arguments
/// - `buf`        – rollout buffer (advantage/returns will be written in-place)
/// - `last_value` – V(s_{T+1}), the bootstrap value for the last state
///                  (0 if the episode ended)
/// - `gamma`      – discount factor
/// - `lam`        – GAE lambda
pub fn compute_gae<F>(
    buf: &mut PPOBuffer<F>,
    last_value: f64,
    gamma: f64,
    lam: f64,
) -> Result<()>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive + Clone,
{
    let t = buf.steps.len();
    if t == 0 {
        return Err(NeuralError::InvalidArgument(
            "compute_gae: buffer is empty".to_string(),
        ));
    }

    let gamma_f = F::from_f64(gamma).ok_or_else(|| {
        NeuralError::ComputationError("compute_gae: cannot convert gamma".to_string())
    })?;
    let lam_f = F::from_f64(lam).ok_or_else(|| {
        NeuralError::ComputationError("compute_gae: cannot convert lam".to_string())
    })?;
    let last_val_f = F::from_f64(last_value).ok_or_else(|| {
        NeuralError::ComputationError("compute_gae: cannot convert last_value".to_string())
    })?;

    let mut gae = F::zero();
    // Iterate backwards
    for i in (0..t).rev() {
        let next_non_terminal = if buf.steps[i].done {
            F::zero()
        } else {
            F::one()
        };
        let next_value = if i + 1 < t {
            buf.steps[i + 1].value
        } else {
            last_val_f
        };

        let delta = buf.steps[i].reward
            + gamma_f * next_value * next_non_terminal
            - buf.steps[i].value;
        gae = delta + gamma_f * lam_f * next_non_terminal * gae;

        buf.steps[i].advantage = gae;
        buf.steps[i].returns = gae + buf.steps[i].value;
    }
    Ok(())
}

// ============================================================================
// PPO Clipped Surrogate Objective
// ============================================================================

/// Compute the PPO clipped policy loss.
///
/// ```text
/// L_CLIP = -mean( min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) )
/// ```
///
/// where `r_t = exp(log_prob_new - log_prob_old)`.
///
/// # Arguments
/// - `log_probs_new` – log π_θ(a|s) under the **new** policy, shape `[T]`
/// - `log_probs_old` – log π_θ_old(a|s) from the behaviour policy, shape `[T]`
/// - `advantages`    – advantage estimates, shape `[T]`
/// - `clip_range`    – ε clipping threshold
///
/// # Returns
/// Scalar loss (positive, since we negate the surrogate objective).
pub fn ppo_clip_loss<F>(
    log_probs_new: &Array1<F>,
    log_probs_old: &Array1<F>,
    advantages: &Array1<F>,
    clip_range: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let t = log_probs_new.len();
    if t == 0 {
        return Err(NeuralError::InvalidArgument(
            "ppo_clip_loss: empty arrays".to_string(),
        ));
    }
    if log_probs_old.len() != t || advantages.len() != t {
        return Err(NeuralError::DimensionMismatch(format!(
            "ppo_clip_loss: length mismatch {t} / {} / {}",
            log_probs_old.len(),
            advantages.len()
        )));
    }

    let eps = F::from_f64(clip_range).ok_or_else(|| {
        NeuralError::ComputationError("ppo_clip_loss: cannot convert clip_range".to_string())
    })?;
    let one = F::one();
    let clip_lo = one - eps;
    let clip_hi = one + eps;

    let mut total = F::zero();
    for i in 0..t {
        let log_ratio = log_probs_new[i] - log_probs_old[i];
        let ratio = log_ratio.exp();
        let surr1 = ratio * advantages[i];
        let surr2 = ratio.max(clip_lo).min(clip_hi) * advantages[i];
        total += surr1.min(surr2);
    }

    let t_f = F::from_usize(t)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert t".to_string()))?;
    // Negate for minimisation
    Ok(-total / t_f)
}

// ============================================================================
// Clipped Value Function Loss
// ============================================================================

/// Compute the clipped value function MSE loss.
///
/// ```text
/// V_clipped = clip(V_new, V_old - ε, V_old + ε)
/// L_VF = 0.5 * mean( max( (V_new - R)², (V_clipped - R)² ) )
/// ```
///
/// When `clip_range_vf == 0.0` the unclipped MSE is returned.
///
/// # Arguments
/// - `values_new`    – V_θ(s) from the current critic, shape `[T]`
/// - `values_old`    – V_θ_old(s) from the behaviour critic, shape `[T]`
/// - `returns`       – target returns R_t, shape `[T]`
/// - `clip_range_vf` – ε for value clipping (0.0 = no clipping)
///
/// # Returns
/// Scalar loss.
pub fn value_loss<F>(
    values_new: &Array1<F>,
    values_old: &Array1<F>,
    returns: &Array1<F>,
    clip_range_vf: f64,
) -> Result<F>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let t = values_new.len();
    if t == 0 {
        return Err(NeuralError::InvalidArgument(
            "value_loss: empty arrays".to_string(),
        ));
    }
    if values_old.len() != t || returns.len() != t {
        return Err(NeuralError::DimensionMismatch(format!(
            "value_loss: length mismatch {t} / {} / {}",
            values_old.len(),
            returns.len()
        )));
    }

    let half = F::from_f64(0.5).ok_or_else(|| {
        NeuralError::ComputationError("value_loss: cannot convert 0.5".to_string())
    })?;

    let clip = clip_range_vf != 0.0;
    let eps = if clip {
        F::from_f64(clip_range_vf).ok_or_else(|| {
            NeuralError::ComputationError(
                "value_loss: cannot convert clip_range_vf".to_string(),
            )
        })?
    } else {
        F::zero()
    };

    let mut total = F::zero();
    for i in 0..t {
        let err_unclipped = values_new[i] - returns[i];
        let loss_unclipped = err_unclipped * err_unclipped;

        let loss = if clip {
            let v_clipped = (values_new[i])
                .max(values_old[i] - eps)
                .min(values_old[i] + eps);
            let err_clipped = v_clipped - returns[i];
            let loss_clipped = err_clipped * err_clipped;
            loss_unclipped.max(loss_clipped)
        } else {
            loss_unclipped
        };

        total += loss;
    }

    let t_f = F::from_usize(t)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert t".to_string()))?;
    Ok(half * total / t_f)
}

// ============================================================================
// Combined PPO Loss
// ============================================================================

/// Output of a single PPO loss computation.
#[derive(Debug, Clone)]
pub struct PPOLossOutput<F: Float + Debug> {
    /// Total combined loss.
    pub total_loss: F,
    /// Policy surrogate loss component.
    pub policy_loss: F,
    /// Value function loss component.
    pub vf_loss: F,
    /// Entropy bonus (before coefficient).
    pub entropy: F,
    /// Clipping fraction — fraction of steps where ratio was clipped.
    pub clip_fraction: F,
    /// Mean ratio `π_new / π_old`.
    pub mean_ratio: F,
}

/// Compute the full PPO loss.
///
/// ```text
/// L = L_CLIP + c_v * L_VF - c_e * H[π]
/// ```
///
/// # Arguments
/// - `log_probs_new` – log π_θ(a|s), shape `[T]`
/// - `log_probs_old` – log π_θ_old(a|s), shape `[T]`
/// - `advantages`    – normalised advantage estimates, shape `[T]`
/// - `values_new`    – current critic values, shape `[T]`
/// - `values_old`    – old critic values, shape `[T]`
/// - `returns`       – GAE returns, shape `[T]`
/// - `entropy`       – policy entropy H[π] (scalar, caller supplies)
/// - `config`        – PPO hyper-parameters
pub fn ppo_loss<F>(
    log_probs_new: &Array1<F>,
    log_probs_old: &Array1<F>,
    advantages: &Array1<F>,
    values_new: &Array1<F>,
    values_old: &Array1<F>,
    returns: &Array1<F>,
    entropy: F,
    config: &PPOConfig,
) -> Result<PPOLossOutput<F>>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let policy_loss = ppo_clip_loss(log_probs_new, log_probs_old, advantages, config.clip_range)?;
    let vf_loss = value_loss(values_new, values_old, returns, config.clip_range_vf)?;

    let c_v = F::from_f64(config.value_coeff).ok_or_else(|| {
        NeuralError::ComputationError("ppo_loss: cannot convert value_coeff".to_string())
    })?;
    let c_e = F::from_f64(config.entropy_coeff).ok_or_else(|| {
        NeuralError::ComputationError("ppo_loss: cannot convert entropy_coeff".to_string())
    })?;

    let total = policy_loss + c_v * vf_loss - c_e * entropy;

    // Diagnostics
    let (clip_frac, mean_ratio) = clip_diagnostics(log_probs_new, log_probs_old, config.clip_range)?;

    Ok(PPOLossOutput {
        total_loss: total,
        policy_loss,
        vf_loss,
        entropy,
        clip_fraction: clip_frac,
        mean_ratio,
    })
}

/// Compute the clipping fraction and mean importance-weight ratio.
fn clip_diagnostics<F>(
    log_probs_new: &Array1<F>,
    log_probs_old: &Array1<F>,
    clip_range: f64,
) -> Result<(F, F)>
where
    F: Float + Debug + NumAssign + FromPrimitive + ToPrimitive,
{
    let t = log_probs_new.len();
    let eps = F::from_f64(clip_range).ok_or_else(|| {
        NeuralError::ComputationError("clip_diagnostics: cannot convert clip_range".to_string())
    })?;
    let one = F::one();
    let lo = one - eps;
    let hi = one + eps;

    let mut clipped = 0usize;
    let mut sum_ratio = F::zero();
    for i in 0..t {
        let ratio = (log_probs_new[i] - log_probs_old[i]).exp();
        sum_ratio += ratio;
        if ratio < lo || ratio > hi {
            clipped += 1;
        }
    }
    let t_f = F::from_usize(t)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert t".to_string()))?;
    let clip_frac = F::from_usize(clipped)
        .ok_or_else(|| NeuralError::ComputationError("cannot convert clipped".to_string()))?
        / t_f;
    let mean_ratio = sum_ratio / t_f;
    Ok((clip_frac, mean_ratio))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_buffer_push_and_gae() {
        let config = PPOConfig::default();
        let mut buf = PPOBuffer::<f64>::new(config.clone(), 5, 4);

        for i in 0..5 {
            let obs = Array1::from(vec![i as f64; 4]);
            buf.push(obs, 0, 1.0, 0.5, -0.3, i == 4).expect("push");
        }
        assert!(buf.is_full());
        assert_eq!(buf.len(), 5);

        compute_gae(&mut buf, 0.0, config.gamma, config.lam).expect("gae");
        for step in &buf.steps {
            assert!(step.advantage.is_finite());
            assert!(step.returns.is_finite());
        }
    }

    #[test]
    fn test_gae_terminal_step() {
        let config = PPOConfig::default();
        let mut buf = PPOBuffer::<f64>::new(config.clone(), 3, 2);
        // Episode ends at step 2 (done=true)
        buf.push(Array1::zeros(2), 0, 1.0, 1.0, 0.0, false).expect("push");
        buf.push(Array1::zeros(2), 0, 1.0, 1.0, 0.0, true).expect("push");
        buf.push(Array1::zeros(2), 0, 1.0, 1.0, 0.0, false).expect("push");

        compute_gae(&mut buf, 0.0, 0.99, 0.95).expect("gae");

        // After terminal step, the advantage chain should restart
        assert!(buf.steps[2].advantage.is_finite());
    }

    #[test]
    fn test_ppo_clip_loss_unclipped() {
        // When ratio = 1 (same policy), loss == -mean(advantages)
        let log_probs = Array1::from(vec![0.0_f64, 0.0, 0.0]);
        let advantages = Array1::from(vec![1.0_f64, 2.0, 3.0]);
        let loss = ppo_clip_loss(&log_probs, &log_probs, &advantages, 0.2).expect("loss");
        // -mean([1,2,3]) = -2.0
        assert!((loss - (-2.0)).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_ppo_clip_loss_clipped() {
        // Ratio = exp(large) = large, clipped at 1.2 for positive advantage
        let log_probs_new = Array1::from(vec![5.0_f64]);
        let log_probs_old = Array1::from(vec![0.0_f64]);
        let advantages = Array1::from(vec![1.0_f64]);
        let loss = ppo_clip_loss(&log_probs_new, &log_probs_old, &advantages, 0.2).expect("loss");
        // Clipped surrogate = 1.2 * 1.0 = 1.2, unclipped = exp(5) * 1.0
        // min(unclipped, clipped) = 1.2 → loss = -1.2
        assert!((loss - (-1.2)).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_value_loss_no_clip() {
        let v_new = Array1::from(vec![1.0_f64, 2.0]);
        let v_old = Array1::from(vec![1.5_f64, 2.5]);
        let returns = Array1::from(vec![2.0_f64, 3.0]);
        // errors: (1-2)^2=1, (2-3)^2=1 → 0.5 * mean(1,1) = 0.5
        let loss = value_loss(&v_new, &v_old, &returns, 0.0).expect("loss");
        assert!((loss - 0.5).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_value_loss_clipped() {
        let v_new = Array1::from(vec![5.0_f64]); // far from v_old
        let v_old = Array1::from(vec![1.0_f64]);
        let returns = Array1::from(vec![2.0_f64]);
        // clip(5.0, 0.8, 1.2) = 1.2
        // loss_unclipped = (5-2)^2 = 9
        // loss_clipped   = (1.2-2)^2 = 0.64
        // max(9, 0.64) = 9 → 0.5 * 9 = 4.5
        let loss = value_loss(&v_new, &v_old, &returns, 0.2).expect("loss");
        assert!((loss - 4.5).abs() < 1e-9, "loss={loss}");
    }

    #[test]
    fn test_normalise_advantages() {
        let config = PPOConfig::default();
        let mut buf = PPOBuffer::<f64>::new(config.clone(), 4, 2);
        for _ in 0..4 {
            buf.push(Array1::zeros(2), 0, 1.0, 0.5, 0.0, false).expect("push");
        }
        compute_gae(&mut buf, 0.0, 0.99, 0.95).expect("gae");
        // manually set diverse advantages
        buf.steps[0].advantage = 1.0;
        buf.steps[1].advantage = 2.0;
        buf.steps[2].advantage = 3.0;
        buf.steps[3].advantage = 4.0;
        buf.normalise_advantages().expect("normalise");
        let adv = buf.advantages();
        let mean: f64 = adv.sum() / adv.len() as f64;
        assert!(mean.abs() < 1e-6, "mean={mean}");
    }

    #[test]
    fn test_ppo_loss_combined() {
        let config = PPOConfig::default();
        let t = 8;
        let log_p_new = Array1::from(vec![-1.0_f64; t]);
        let log_p_old = Array1::from(vec![-1.0_f64; t]);
        let adv = Array1::from(vec![1.0_f64; t]);
        let v_new = Array1::from(vec![1.0_f64; t]);
        let v_old = Array1::from(vec![1.0_f64; t]);
        let rets = Array1::from(vec![1.5_f64; t]);
        let entropy = 0.5_f64;

        let out = ppo_loss(&log_p_new, &log_p_old, &adv, &v_new, &v_old, &rets, entropy, &config)
            .expect("ppo_loss");

        assert!(out.total_loss.is_finite());
        assert!(out.policy_loss.is_finite());
        assert!(out.vf_loss.is_finite());
        assert!(out.clip_fraction.is_finite());
    }

    #[test]
    fn test_buffer_overflow_error() {
        let config = PPOConfig::default();
        let mut buf = PPOBuffer::<f64>::new(config, 1, 2);
        buf.push(Array1::zeros(2), 0, 0.0, 0.0, 0.0, false).expect("push");
        let result = buf.push(Array1::zeros(2), 0, 0.0, 0.0, 0.0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_obs_dim_mismatch_error() {
        let config = PPOConfig::default();
        let mut buf = PPOBuffer::<f64>::new(config, 4, 4);
        let wrong_obs = Array1::zeros(3);
        let result = buf.push(wrong_obs, 0, 0.0, 0.0, 0.0, false);
        assert!(result.is_err());
    }
}
