//! Curriculum Learning for Neural Network Training
//!
//! Implements curriculum learning strategies that present training samples in a
//! meaningful order, typically from easy to hard, to improve convergence and
//! final performance.
//!
//! # Strategies
//!
//! - **Baby Step**: Gradually introduce harder examples as training progresses
//! - **One-Pass**: Each difficulty level is visited exactly once
//! - **Self-Paced Learning**: Use per-sample loss as a proxy for difficulty
//! - **Anti-Curriculum**: Hard examples first (can improve robustness)
//!
//! # Scheduling
//!
//! The *competence* function determines what fraction of the dataset is
//! available at a given training progress. Available schedules:
//!
//! - **Linear**: `c(t) = c_0 + (1 - c_0) * t`
//! - **Sqrt**: `c(t) = c_0 + (1 - c_0) * sqrt(t)`
//! - **Step**: `c(t) = c_0 + ceil(t * num_steps) / num_steps * (1 - c_0)`
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::curriculum::{
//!     CurriculumConfig, CurriculumStrategy, CompetenceSchedule,
//!     CurriculumLearner, DifficultyScorer,
//! };
//!
//! // Create a curriculum learner
//! let config = CurriculumConfig::builder()
//!     .strategy(CurriculumStrategy::BabyStep)
//!     .schedule(CompetenceSchedule::Sqrt)
//!     .initial_competence(0.1)
//!     .num_epochs(100)
//!     .build()
//!     .expect("valid config");
//!
//! // Difficulty scores for 10 samples (0.0 = easy, 1.0 = hard)
//! let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0];
//! let mut learner = CurriculumLearner::new(config, &difficulties);
//!
//! // Get indices for epoch 0 (only easiest samples)
//! let indices = learner.get_batch_indices(0);
//! assert!(!indices.is_empty());
//! assert!(indices.len() <= difficulties.len());
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::{self, Debug, Display};

// ============================================================================
// Types
// ============================================================================

/// Strategy for curriculum learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumStrategy {
    /// Baby Step: gradually introduce harder samples as competence grows.
    /// At each epoch, all samples with difficulty <= competence are available.
    BabyStep,
    /// One-Pass: each difficulty bucket is visited exactly once.
    /// As competence increases, new (harder) samples replace easier ones.
    OnePass,
    /// Self-Paced Learning: use per-sample loss as difficulty proxy.
    /// Samples with loss below a threshold are used; the threshold grows.
    SelfPaced,
    /// Anti-Curriculum: hardest examples first.
    /// Reverses the difficulty ordering.
    AntiCurriculum,
}

impl Display for CurriculumStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BabyStep => write!(f, "BabyStep"),
            Self::OnePass => write!(f, "OnePass"),
            Self::SelfPaced => write!(f, "SelfPaced"),
            Self::AntiCurriculum => write!(f, "AntiCurriculum"),
        }
    }
}

/// Schedule for the competence function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompetenceSchedule {
    /// Linear increase: `c(t) = c_0 + (1 - c_0) * t`.
    Linear,
    /// Square-root increase: `c(t) = c_0 + (1 - c_0) * sqrt(t)`.
    /// Reaches high competence faster.
    Sqrt,
    /// Step-wise increase: competence jumps at regular intervals.
    Step {
        /// Number of discrete competence levels.
        num_steps: usize,
    },
}

impl Display for CompetenceSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linear => write!(f, "Linear"),
            Self::Sqrt => write!(f, "Sqrt"),
            Self::Step { num_steps } => write!(f, "Step({num_steps})"),
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for curriculum learning.
#[derive(Debug, Clone)]
pub struct CurriculumConfig {
    /// Curriculum strategy.
    pub strategy: CurriculumStrategy,
    /// Competence schedule.
    pub schedule: CompetenceSchedule,
    /// Initial competence level (fraction of data available at start).
    /// Must be in (0.0, 1.0].
    pub initial_competence: f64,
    /// Total number of epochs for the curriculum.
    /// Used to compute progress t = epoch / num_epochs.
    pub num_epochs: usize,
    /// For Self-Paced Learning: the growth rate of the loss threshold.
    /// The threshold starts at `self_paced_initial_threshold` and grows
    /// by this factor each epoch.
    pub self_paced_growth_rate: f64,
    /// For Self-Paced Learning: the initial loss threshold.
    pub self_paced_initial_threshold: f64,
    /// Whether to shuffle within the selected subset.
    pub shuffle_within_subset: bool,
    /// Minimum number of samples to include (even at lowest competence).
    pub min_samples: usize,
}

impl Default for CurriculumConfig {
    fn default() -> Self {
        Self {
            strategy: CurriculumStrategy::BabyStep,
            schedule: CompetenceSchedule::Linear,
            initial_competence: 0.1,
            num_epochs: 100,
            self_paced_growth_rate: 1.2,
            self_paced_initial_threshold: 0.5,
            shuffle_within_subset: true,
            min_samples: 1,
        }
    }
}

impl CurriculumConfig {
    /// Create a builder.
    pub fn builder() -> CurriculumConfigBuilder {
        CurriculumConfigBuilder::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.initial_competence <= 0.0 || self.initial_competence > 1.0 {
            return Err(NeuralError::InvalidArgument(
                "initial_competence must be in (0.0, 1.0]".to_string(),
            ));
        }
        if self.num_epochs == 0 {
            return Err(NeuralError::InvalidArgument(
                "num_epochs must be > 0".to_string(),
            ));
        }
        if self.self_paced_growth_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "self_paced_growth_rate must be positive".to_string(),
            ));
        }
        if self.self_paced_initial_threshold <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "self_paced_initial_threshold must be positive".to_string(),
            ));
        }
        if self.min_samples == 0 {
            return Err(NeuralError::InvalidArgument(
                "min_samples must be >= 1".to_string(),
            ));
        }
        if let CompetenceSchedule::Step { num_steps } = self.schedule {
            if num_steps == 0 {
                return Err(NeuralError::InvalidArgument(
                    "Step schedule num_steps must be > 0".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Compute the competence at a given epoch.
    ///
    /// Returns a value in [initial_competence, 1.0].
    pub fn competence_at(&self, epoch: usize) -> f64 {
        let t = (epoch as f64 / self.num_epochs as f64).min(1.0);
        let c0 = self.initial_competence;
        let c = match self.schedule {
            CompetenceSchedule::Linear => c0 + (1.0 - c0) * t,
            CompetenceSchedule::Sqrt => c0 + (1.0 - c0) * t.sqrt(),
            CompetenceSchedule::Step { num_steps } => {
                let step = (t * num_steps as f64).ceil() as usize;
                let step = step.min(num_steps);
                c0 + (1.0 - c0) * (step as f64 / num_steps as f64)
            }
        };
        c.min(1.0).max(c0)
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`CurriculumConfig`].
#[derive(Debug, Clone, Default)]
pub struct CurriculumConfigBuilder {
    config: CurriculumConfig,
}

impl CurriculumConfigBuilder {
    /// Set the curriculum strategy.
    pub fn strategy(mut self, s: CurriculumStrategy) -> Self {
        self.config.strategy = s;
        self
    }

    /// Set the competence schedule.
    pub fn schedule(mut self, s: CompetenceSchedule) -> Self {
        self.config.schedule = s;
        self
    }

    /// Set the initial competence level.
    pub fn initial_competence(mut self, c: f64) -> Self {
        self.config.initial_competence = c;
        self
    }

    /// Set the total number of epochs.
    pub fn num_epochs(mut self, n: usize) -> Self {
        self.config.num_epochs = n;
        self
    }

    /// Set the self-paced learning growth rate.
    pub fn self_paced_growth_rate(mut self, r: f64) -> Self {
        self.config.self_paced_growth_rate = r;
        self
    }

    /// Set the self-paced learning initial threshold.
    pub fn self_paced_initial_threshold(mut self, t: f64) -> Self {
        self.config.self_paced_initial_threshold = t;
        self
    }

    /// Set whether to shuffle within the selected subset.
    pub fn shuffle_within_subset(mut self, s: bool) -> Self {
        self.config.shuffle_within_subset = s;
        self
    }

    /// Set the minimum number of samples.
    pub fn min_samples(mut self, n: usize) -> Self {
        self.config.min_samples = n;
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<CurriculumConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ============================================================================
// Difficulty scorer
// ============================================================================

/// Trait for computing difficulty scores for training samples.
///
/// A difficulty score is a value in [0.0, 1.0] where 0.0 is the easiest
/// and 1.0 is the hardest.
pub trait DifficultyScorer: Debug {
    /// Compute difficulty scores for a batch of samples.
    ///
    /// Returns a vector of scores in [0.0, 1.0].
    fn score(&self, sample_indices: &[usize]) -> Vec<f64>;

    /// Name of the scorer.
    fn name(&self) -> &str;
}

/// Loss-based difficulty scorer: uses the per-sample loss from the
/// previous epoch as a difficulty proxy.
#[derive(Debug, Clone)]
pub struct LossBasedScorer {
    /// Per-sample losses from the most recent epoch.
    losses: Vec<f64>,
    /// Maximum loss (for normalization).
    max_loss: f64,
}

impl LossBasedScorer {
    /// Create a new scorer with initial losses.
    pub fn new(losses: &[f64]) -> Self {
        let max_loss = losses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(f64::EPSILON);
        Self {
            losses: losses.to_vec(),
            max_loss,
        }
    }

    /// Update the scorer with new losses (e.g., after an epoch).
    pub fn update(&mut self, losses: &[f64]) {
        self.losses = losses.to_vec();
        self.max_loss = losses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(f64::EPSILON);
    }
}

impl DifficultyScorer for LossBasedScorer {
    fn score(&self, sample_indices: &[usize]) -> Vec<f64> {
        sample_indices
            .iter()
            .map(|&i| {
                if i < self.losses.len() {
                    (self.losses[i] / self.max_loss).clamp(0.0, 1.0)
                } else {
                    0.5 // default for unknown samples
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "LossBasedScorer"
    }
}

/// Static difficulty scorer that uses pre-assigned scores.
#[derive(Debug, Clone)]
pub struct StaticScorer {
    scores: Vec<f64>,
}

impl StaticScorer {
    /// Create a new static scorer with pre-assigned difficulty scores.
    pub fn new(scores: &[f64]) -> Self {
        Self {
            scores: scores.to_vec(),
        }
    }
}

impl DifficultyScorer for StaticScorer {
    fn score(&self, sample_indices: &[usize]) -> Vec<f64> {
        sample_indices
            .iter()
            .map(|&i| {
                if i < self.scores.len() {
                    self.scores[i].clamp(0.0, 1.0)
                } else {
                    0.5
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "StaticScorer"
    }
}

// ============================================================================
// Curriculum Learner
// ============================================================================

/// Manages curriculum-based sample selection during training.
///
/// The learner maintains difficulty scores for all samples and, given the
/// current epoch, returns the indices of samples that should be used.
#[derive(Debug, Clone)]
pub struct CurriculumLearner {
    /// Configuration.
    config: CurriculumConfig,
    /// Difficulty scores for all samples, in [0.0, 1.0].
    difficulties: Vec<f64>,
    /// Sorted indices (by difficulty, ascending).
    sorted_indices: Vec<usize>,
    /// Per-sample losses from the last epoch (for self-paced learning).
    sample_losses: Vec<f64>,
    /// Current self-paced threshold.
    self_paced_threshold: f64,
    /// Number of total samples.
    num_samples: usize,
    /// Statistics: how many samples were selected at each epoch.
    epoch_sample_counts: Vec<usize>,
}

impl CurriculumLearner {
    /// Create a new curriculum learner.
    ///
    /// # Arguments
    /// * `config` — curriculum configuration
    /// * `difficulties` — difficulty scores for all samples (0.0 = easy, 1.0 = hard)
    pub fn new(config: CurriculumConfig, difficulties: &[f64]) -> Self {
        let num_samples = difficulties.len();
        let difficulties: Vec<f64> = difficulties.iter().map(|&d| d.clamp(0.0, 1.0)).collect();

        // Sort indices by difficulty (ascending for curriculum, descending for anti)
        let mut sorted_indices: Vec<usize> = (0..num_samples).collect();
        match config.strategy {
            CurriculumStrategy::AntiCurriculum => {
                sorted_indices.sort_by(|&a, &b| {
                    difficulties[b]
                        .partial_cmp(&difficulties[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            _ => {
                sorted_indices.sort_by(|&a, &b| {
                    difficulties[a]
                        .partial_cmp(&difficulties[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let self_paced_threshold = config.self_paced_initial_threshold;

        Self {
            config,
            difficulties,
            sorted_indices,
            sample_losses: vec![0.0; num_samples],
            self_paced_threshold,
            num_samples,
            epoch_sample_counts: Vec::new(),
        }
    }

    /// Get the sample indices to use for the given epoch.
    pub fn get_batch_indices(&mut self, epoch: usize) -> Vec<usize> {
        let indices = match self.config.strategy {
            CurriculumStrategy::BabyStep => self.baby_step_indices(epoch),
            CurriculumStrategy::OnePass => self.one_pass_indices(epoch),
            CurriculumStrategy::SelfPaced => self.self_paced_indices(epoch),
            CurriculumStrategy::AntiCurriculum => self.baby_step_indices(epoch),
        };

        self.epoch_sample_counts.push(indices.len());
        indices
    }

    /// Baby Step (and Anti-Curriculum): include all samples with rank <= competence * N.
    fn baby_step_indices(&self, epoch: usize) -> Vec<usize> {
        let competence = self.config.competence_at(epoch);
        let num_to_include = ((competence * self.num_samples as f64).ceil() as usize)
            .max(self.config.min_samples)
            .min(self.num_samples);

        self.sorted_indices[..num_to_include].to_vec()
    }

    /// One-Pass: only include the new "band" of samples unlocked at this epoch.
    fn one_pass_indices(&self, epoch: usize) -> Vec<usize> {
        let competence_now = self.config.competence_at(epoch);
        let competence_prev = if epoch > 0 {
            self.config.competence_at(epoch - 1)
        } else {
            0.0
        };

        let start =
            ((competence_prev * self.num_samples as f64).floor() as usize).min(self.num_samples);
        let end = ((competence_now * self.num_samples as f64).ceil() as usize)
            .max(self.config.min_samples)
            .min(self.num_samples);

        if start >= end {
            // If no new samples, include the most recent band again
            let fallback_start = end.saturating_sub(self.config.min_samples);
            return self.sorted_indices[fallback_start..end].to_vec();
        }

        self.sorted_indices[start..end].to_vec()
    }

    /// Self-Paced: include samples whose loss is below the current threshold.
    fn self_paced_indices(&self, _epoch: usize) -> Vec<usize> {
        let threshold = self.self_paced_threshold;
        let mut indices: Vec<usize> = (0..self.num_samples)
            .filter(|&i| self.sample_losses[i] < threshold)
            .collect();

        // Ensure minimum samples
        if indices.len() < self.config.min_samples && self.num_samples > 0 {
            // Add the easiest samples that are missing
            for &idx in &self.sorted_indices {
                if indices.len() >= self.config.min_samples {
                    break;
                }
                if !indices.contains(&idx) {
                    indices.push(idx);
                }
            }
        }

        indices
    }

    /// Update per-sample losses (for self-paced learning).
    ///
    /// Call this after each epoch with the loss for each sample.
    pub fn update_losses(&mut self, losses: &[f64]) {
        let n = losses.len().min(self.num_samples);
        self.sample_losses[..n].copy_from_slice(&losses[..n]);

        // Grow the self-paced threshold
        self.self_paced_threshold *= self.config.self_paced_growth_rate;
    }

    /// Update difficulty scores (e.g., recompute from losses).
    pub fn update_difficulties(&mut self, new_difficulties: &[f64]) {
        let n = new_difficulties.len().min(self.num_samples);
        for (i, &nd) in new_difficulties.iter().enumerate().take(n) {
            self.difficulties[i] = nd.clamp(0.0, 1.0);
        }

        // Re-sort
        match self.config.strategy {
            CurriculumStrategy::AntiCurriculum => {
                self.sorted_indices.sort_by(|&a, &b| {
                    self.difficulties[b]
                        .partial_cmp(&self.difficulties[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            _ => {
                self.sorted_indices.sort_by(|&a, &b| {
                    self.difficulties[a]
                        .partial_cmp(&self.difficulties[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
    }

    /// Get the competence at the given epoch.
    pub fn competence(&self, epoch: usize) -> f64 {
        self.config.competence_at(epoch)
    }

    /// Get the current self-paced threshold.
    pub fn self_paced_threshold(&self) -> f64 {
        self.self_paced_threshold
    }

    /// Get the difficulty scores.
    pub fn difficulties(&self) -> &[f64] {
        &self.difficulties
    }

    /// Get the sample selection counts history.
    pub fn epoch_sample_counts(&self) -> &[usize] {
        &self.epoch_sample_counts
    }

    /// Get the total number of samples.
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Get the strategy name.
    pub fn strategy_name(&self) -> &str {
        match self.config.strategy {
            CurriculumStrategy::BabyStep => "BabyStep",
            CurriculumStrategy::OnePass => "OnePass",
            CurriculumStrategy::SelfPaced => "SelfPaced",
            CurriculumStrategy::AntiCurriculum => "AntiCurriculum",
        }
    }

    /// Generate a summary of the curriculum progression.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Curriculum Learning Summary ===\n");
        out.push_str(&format!("Strategy: {}\n", self.config.strategy));
        out.push_str(&format!("Schedule: {}\n", self.config.schedule));
        out.push_str(&format!(
            "Initial competence: {:.2}\n",
            self.config.initial_competence
        ));
        out.push_str(&format!("Total samples: {}\n", self.num_samples));
        out.push_str(&format!(
            "Epochs trained: {}\n",
            self.epoch_sample_counts.len()
        ));

        if !self.epoch_sample_counts.is_empty() {
            let first = self.epoch_sample_counts[0];
            let last = self.epoch_sample_counts.last().copied().unwrap_or(0);
            out.push_str(&format!("Samples at first epoch: {first}\n"));
            out.push_str(&format!("Samples at last epoch: {last}\n"));
        }
        out
    }
}

// ============================================================================
// Curriculum scheduler (integration helper)
// ============================================================================

/// A curriculum schedule that can be integrated with training loops.
///
/// This provides a simpler interface for controlling which samples are
/// available at each training step.
#[derive(Debug, Clone)]
pub struct CurriculumSchedule {
    /// The competence schedule.
    schedule: CompetenceSchedule,
    /// Initial competence.
    initial_competence: f64,
    /// Total number of epochs.
    num_epochs: usize,
}

impl CurriculumSchedule {
    /// Create a new curriculum schedule.
    pub fn new(
        schedule: CompetenceSchedule,
        initial_competence: f64,
        num_epochs: usize,
    ) -> Result<Self> {
        if initial_competence <= 0.0 || initial_competence > 1.0 {
            return Err(NeuralError::InvalidArgument(
                "initial_competence must be in (0.0, 1.0]".to_string(),
            ));
        }
        if num_epochs == 0 {
            return Err(NeuralError::InvalidArgument(
                "num_epochs must be > 0".to_string(),
            ));
        }
        Ok(Self {
            schedule,
            initial_competence,
            num_epochs,
        })
    }

    /// Get the competence at a given epoch.
    pub fn competence_at(&self, epoch: usize) -> f64 {
        let t = (epoch as f64 / self.num_epochs as f64).min(1.0);
        let c0 = self.initial_competence;
        let c = match self.schedule {
            CompetenceSchedule::Linear => c0 + (1.0 - c0) * t,
            CompetenceSchedule::Sqrt => c0 + (1.0 - c0) * t.sqrt(),
            CompetenceSchedule::Step { num_steps } => {
                let step = (t * num_steps as f64).ceil() as usize;
                let step = step.min(num_steps);
                c0 + (1.0 - c0) * (step as f64 / num_steps as f64)
            }
        };
        c.min(1.0).max(c0)
    }

    /// Get the fraction of data to use at a given epoch.
    pub fn data_fraction(&self, epoch: usize) -> f64 {
        self.competence_at(epoch)
    }

    /// Get the number of samples to use from a dataset of given size.
    pub fn num_samples_at(&self, epoch: usize, total_samples: usize) -> usize {
        let frac = self.data_fraction(epoch);
        ((frac * total_samples as f64).ceil() as usize)
            .max(1)
            .min(total_samples)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = CurriculumConfig::default();
        assert_eq!(config.strategy, CurriculumStrategy::BabyStep);
        assert!((config.initial_competence - 0.1).abs() < 1e-10);
        assert_eq!(config.num_epochs, 100);
    }

    #[test]
    fn test_config_builder() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::OnePass)
            .schedule(CompetenceSchedule::Sqrt)
            .initial_competence(0.2)
            .num_epochs(50)
            .build()
            .expect("valid config");

        assert_eq!(config.strategy, CurriculumStrategy::OnePass);
        assert!((config.initial_competence - 0.2).abs() < 1e-10);
        assert_eq!(config.num_epochs, 50);
    }

    #[test]
    fn test_config_validation_errors() {
        // initial_competence out of range
        assert!(CurriculumConfig::builder()
            .initial_competence(0.0)
            .build()
            .is_err());
        assert!(CurriculumConfig::builder()
            .initial_competence(1.5)
            .build()
            .is_err());
        assert!(CurriculumConfig::builder()
            .initial_competence(-0.1)
            .build()
            .is_err());

        // num_epochs == 0
        assert!(CurriculumConfig::builder().num_epochs(0).build().is_err());

        // self_paced_growth_rate <= 0
        assert!(CurriculumConfig::builder()
            .self_paced_growth_rate(0.0)
            .build()
            .is_err());

        // min_samples == 0
        assert!(CurriculumConfig::builder().min_samples(0).build().is_err());

        // Step schedule with 0 steps
        assert!(CurriculumConfig::builder()
            .schedule(CompetenceSchedule::Step { num_steps: 0 })
            .build()
            .is_err());
    }

    #[test]
    fn test_linear_competence() {
        let config = CurriculumConfig::builder()
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.1)
            .num_epochs(100)
            .build()
            .expect("valid config");

        let c0 = config.competence_at(0);
        let c50 = config.competence_at(50);
        let c100 = config.competence_at(100);

        assert!((c0 - 0.1).abs() < 1e-10);
        assert!((c50 - 0.55).abs() < 1e-10); // 0.1 + 0.9 * 0.5
        assert!((c100 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt_competence() {
        let config = CurriculumConfig::builder()
            .schedule(CompetenceSchedule::Sqrt)
            .initial_competence(0.0001)
            .num_epochs(100)
            .build()
            .expect("valid config");

        let c0 = config.competence_at(0);
        let c25 = config.competence_at(25);
        let c100 = config.competence_at(100);

        assert!(c0 < 0.01);
        assert!(c25 > c0); // sqrt grows fast initially
        assert!((c100 - 1.0).abs() < 1e-6);

        // sqrt(0.25) = 0.5, so c25 ≈ 0.0001 + 0.9999 * 0.5 ≈ 0.5
        assert!((c25 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_step_competence() {
        let config = CurriculumConfig::builder()
            .schedule(CompetenceSchedule::Step { num_steps: 4 })
            .initial_competence(0.2)
            .num_epochs(100)
            .build()
            .expect("valid config");

        // Steps at 25%, 50%, 75%, 100%
        let c0 = config.competence_at(0);
        let c10 = config.competence_at(10); // t=0.1 → ceil(0.4)=1 → 0.2 + 0.8*0.25 = 0.4
        let c50 = config.competence_at(50); // t=0.5 → ceil(2.0)=2 → 0.2 + 0.8*0.5 = 0.6
        let c100 = config.competence_at(100); // t=1.0 → ceil(4.0)=4 → 0.2 + 0.8*1.0 = 1.0

        assert!((c0 - 0.2).abs() < 1e-10);
        // c10: t=0.1, step=ceil(0.1*4)=ceil(0.4)=1, c=0.2+0.8*0.25=0.4
        assert!((c10 - 0.4).abs() < 1e-10);
        assert!((c50 - 0.6).abs() < 1e-10);
        assert!((c100 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_baby_step_basic() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::BabyStep)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.2)
            .num_epochs(10)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        // Epoch 0: competence = 0.2, should include ~2 easiest samples
        let indices0 = learner.get_batch_indices(0);
        assert!(!indices0.is_empty());
        assert!(indices0.len() <= 10);

        // Epoch 10: competence = 1.0, should include all samples
        let indices10 = learner.get_batch_indices(10);
        assert_eq!(indices10.len(), 10);

        // Later epochs should include at least as many samples as earlier epochs
        assert!(indices10.len() >= indices0.len());
    }

    #[test]
    fn test_anti_curriculum() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::AntiCurriculum)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.3)
            .num_epochs(10)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.9, 0.5, 0.3, 0.7];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        // Epoch 0: should include the hardest samples first
        let indices = learner.get_batch_indices(0);
        assert!(!indices.is_empty());

        // The first selected index should be the hardest sample
        // difficulties sorted descending: [1(0.9), 4(0.7), 2(0.5), 3(0.3), 0(0.1)]
        assert_eq!(indices[0], 1); // index 1 has difficulty 0.9
    }

    #[test]
    fn test_one_pass() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::OnePass)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.2)
            .num_epochs(5)
            .min_samples(1)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        // Each epoch should introduce new (harder) samples
        let indices0 = learner.get_batch_indices(0);
        let indices1 = learner.get_batch_indices(1);
        let indices2 = learner.get_batch_indices(2);

        // One-pass should produce non-empty results
        assert!(!indices0.is_empty());
        assert!(!indices1.is_empty());
        assert!(!indices2.is_empty());
    }

    #[test]
    fn test_self_paced_learning() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::SelfPaced)
            .self_paced_initial_threshold(0.5)
            .self_paced_growth_rate(2.0)
            .min_samples(1)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        // Set sample losses
        learner.update_losses(&[0.1, 0.3, 0.8, 1.2, 2.0]);

        // With threshold 0.5, only samples with loss < 0.5 should be included
        let indices = learner.get_batch_indices(0);
        // Samples 0 (loss=0.1) and 1 (loss=0.3) should be included
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));

        // After update_losses, threshold should grow
        learner.update_losses(&[0.1, 0.3, 0.8, 1.2, 2.0]);
        // New threshold = 0.5 * 2.0 = 1.0
        let indices2 = learner.get_batch_indices(1);
        // Now samples with loss < 1.0 should be included: 0(0.1), 1(0.3), 2(0.8)
        assert!(indices2.len() >= indices.len());
    }

    #[test]
    fn test_update_difficulties() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::BabyStep)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.4)
            .num_epochs(10)
            .build()
            .expect("valid config");

        let initial_difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut learner = CurriculumLearner::new(config, &initial_difficulties);

        let indices_before = learner.get_batch_indices(0);

        // Reverse difficulties
        learner.update_difficulties(&[0.9, 0.7, 0.5, 0.3, 0.1]);

        let indices_after = learner.get_batch_indices(0);

        // After reversing, different samples should be selected as "easy"
        assert_eq!(indices_before.len(), indices_after.len());
        // The easiest now is sample 4 (difficulty 0.1)
        assert!(indices_after.contains(&4));
    }

    #[test]
    fn test_competence_monotone() {
        let config = CurriculumConfig::builder()
            .schedule(CompetenceSchedule::Sqrt)
            .initial_competence(0.05)
            .num_epochs(100)
            .build()
            .expect("valid config");

        let mut prev = 0.0;
        for epoch in 0..=100 {
            let c = config.competence_at(epoch);
            assert!(c >= prev, "competence must be monotone increasing");
            assert!(c >= config.initial_competence);
            assert!(c <= 1.0);
            prev = c;
        }
    }

    #[test]
    fn test_min_samples_guarantee() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::BabyStep)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.01) // very low — would give 0 samples for small datasets
            .num_epochs(100)
            .min_samples(3)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        let indices = learner.get_batch_indices(0);
        assert!(indices.len() >= 3);
    }

    #[test]
    fn test_summary_generation() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::BabyStep)
            .schedule(CompetenceSchedule::Linear)
            .initial_competence(0.2)
            .num_epochs(10)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        for epoch in 0..5 {
            learner.get_batch_indices(epoch);
        }

        let summary = learner.summary();
        assert!(summary.contains("Curriculum Learning Summary"));
        assert!(summary.contains("BabyStep"));
        assert!(summary.contains("5")); // samples
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", CurriculumStrategy::BabyStep), "BabyStep");
        assert_eq!(format!("{}", CurriculumStrategy::OnePass), "OnePass");
        assert_eq!(format!("{}", CurriculumStrategy::SelfPaced), "SelfPaced");
        assert_eq!(
            format!("{}", CurriculumStrategy::AntiCurriculum),
            "AntiCurriculum"
        );
    }

    #[test]
    fn test_schedule_display() {
        assert_eq!(format!("{}", CompetenceSchedule::Linear), "Linear");
        assert_eq!(format!("{}", CompetenceSchedule::Sqrt), "Sqrt");
        assert_eq!(
            format!("{}", CompetenceSchedule::Step { num_steps: 5 }),
            "Step(5)"
        );
    }

    #[test]
    fn test_curriculum_schedule_standalone() {
        let schedule =
            CurriculumSchedule::new(CompetenceSchedule::Linear, 0.1, 100).expect("valid");

        assert!((schedule.competence_at(0) - 0.1).abs() < 1e-10);
        assert!((schedule.competence_at(100) - 1.0).abs() < 1e-10);

        let n = schedule.num_samples_at(0, 1000);
        assert!(n >= 1);
        assert!(n <= 1000);

        let n100 = schedule.num_samples_at(100, 1000);
        assert_eq!(n100, 1000);
    }

    #[test]
    fn test_curriculum_schedule_validation() {
        assert!(CurriculumSchedule::new(CompetenceSchedule::Linear, 0.0, 100).is_err());
        assert!(CurriculumSchedule::new(CompetenceSchedule::Linear, 0.5, 0).is_err());
        assert!(CurriculumSchedule::new(CompetenceSchedule::Linear, 1.5, 100).is_err());
    }

    #[test]
    fn test_loss_based_scorer() {
        let losses = vec![0.1, 0.5, 1.0, 0.3, 0.8];
        let scorer = LossBasedScorer::new(&losses);

        let scores = scorer.score(&[0, 1, 2, 3, 4]);
        assert!((scores[0] - 0.1).abs() < 1e-10); // 0.1/1.0
        assert!((scores[2] - 1.0).abs() < 1e-10); // 1.0/1.0

        // Unknown index returns 0.5
        let scores_unknown = scorer.score(&[10]);
        assert!((scores_unknown[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_loss_based_scorer_update() {
        let mut scorer = LossBasedScorer::new(&[1.0, 2.0, 3.0]);
        assert!((scorer.score(&[2])[0] - 1.0).abs() < 1e-10); // 3/3 = 1.0

        scorer.update(&[10.0, 20.0, 30.0]);
        assert!((scorer.score(&[0])[0] - 10.0 / 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_scorer() {
        let scorer = StaticScorer::new(&[0.0, 0.25, 0.5, 0.75, 1.0]);
        let scores = scorer.score(&[0, 2, 4]);
        assert!((scores[0] - 0.0).abs() < 1e-10);
        assert!((scores[1] - 0.5).abs() < 1e-10);
        assert!((scores[2] - 1.0).abs() < 1e-10);
        assert_eq!(scorer.name(), "StaticScorer");
    }

    #[test]
    fn test_learner_accessors() {
        let config = CurriculumConfig::builder()
            .strategy(CurriculumStrategy::SelfPaced)
            .self_paced_initial_threshold(1.0)
            .build()
            .expect("valid config");

        let difficulties = vec![0.1, 0.5, 0.9];
        let learner = CurriculumLearner::new(config, &difficulties);

        assert_eq!(learner.num_samples(), 3);
        assert_eq!(learner.strategy_name(), "SelfPaced");
        assert!((learner.self_paced_threshold() - 1.0).abs() < 1e-10);
        assert_eq!(learner.difficulties().len(), 3);
        assert!(learner.epoch_sample_counts().is_empty());
    }

    #[test]
    fn test_empty_dataset_handling() {
        let config = CurriculumConfig::builder()
            .min_samples(1)
            .build()
            .expect("valid config");

        let difficulties: Vec<f64> = vec![];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        let indices = learner.get_batch_indices(0);
        assert!(indices.is_empty()); // no samples to include
    }

    #[test]
    fn test_single_sample() {
        let config = CurriculumConfig::builder()
            .initial_competence(0.5)
            .min_samples(1)
            .build()
            .expect("valid config");

        let difficulties = vec![0.5];
        let mut learner = CurriculumLearner::new(config, &difficulties);

        let indices = learner.get_batch_indices(0);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_competence_clamp() {
        let config = CurriculumConfig::builder()
            .initial_competence(0.5)
            .num_epochs(10)
            .build()
            .expect("valid config");

        // Beyond num_epochs should still give 1.0
        let c = config.competence_at(200);
        assert!((c - 1.0).abs() < 1e-10);
    }
}
