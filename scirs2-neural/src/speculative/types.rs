//! Types for speculative decoding.
//!
//! This module defines the core data structures used throughout the speculative
//! decoding pipeline: configuration, verification results, decoding statistics,
//! and token probability distributions.

use std::fmt;

/// Configuration for speculative decoding.
///
/// Controls the behavior of the draft-then-verify loop, including how many
/// draft tokens to generate per step, sampling parameters, and whether to
/// adaptively adjust the draft length based on acceptance rates.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per step.
    ///
    /// Higher values amortize target-model calls but risk lower acceptance
    /// rates. Typical range: 2..=8.
    pub draft_length: usize,

    /// Sampling temperature applied to both draft and target distributions.
    ///
    /// Values below 1.0 sharpen the distribution (more greedy);
    /// values above 1.0 flatten it (more random).
    pub temperature: f64,

    /// Top-k filtering: only the `top_k` highest-probability tokens are
    /// considered during sampling. Set to 0 to disable.
    pub top_k: usize,

    /// Maximum number of tokens to generate in total (including the prompt).
    pub max_tokens: usize,

    /// When `true`, the decoder dynamically adjusts `draft_length` based on
    /// the rolling acceptance rate.
    pub adaptive_draft: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_length: 4,
            temperature: 1.0,
            top_k: 50,
            max_tokens: 512,
            adaptive_draft: false,
        }
    }
}

impl fmt::Display for SpeculativeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeculativeConfig(draft_length={}, temperature={:.2}, top_k={}, max_tokens={}, adaptive={})",
            self.draft_length, self.temperature, self.top_k, self.max_tokens, self.adaptive_draft
        )
    }
}

/// Result of verifying a batch of draft tokens against the target model.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Token ids that were accepted by the rejection-sampling step.
    pub accepted_tokens: Vec<usize>,

    /// If a rejection occurred, the 0-based position within the draft where
    /// the first rejection happened. `None` when every draft token was accepted.
    pub rejected_at: Option<usize>,

    /// Fraction of draft tokens that were accepted (0.0..=1.0).
    pub acceptance_rate: f64,
}

impl VerificationResult {
    /// Create a new verification result.
    pub fn new(
        accepted_tokens: Vec<usize>,
        rejected_at: Option<usize>,
        acceptance_rate: f64,
    ) -> Self {
        Self {
            accepted_tokens,
            rejected_at,
            acceptance_rate,
        }
    }

    /// Returns `true` when all draft tokens were accepted.
    pub fn all_accepted(&self) -> bool {
        self.rejected_at.is_none()
    }

    /// Number of accepted tokens.
    pub fn num_accepted(&self) -> usize {
        self.accepted_tokens.len()
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VerificationResult(accepted={}, rejected_at={:?}, rate={:.2})",
            self.accepted_tokens.len(),
            self.rejected_at,
            self.acceptance_rate
        )
    }
}

/// Aggregate statistics collected during a full speculative decoding run.
#[derive(Debug, Clone)]
pub struct DecodingStats {
    /// Total number of tokens produced (final output length minus prompt length).
    pub total_tokens: usize,

    /// Total number of draft tokens proposed across all steps.
    pub draft_tokens: usize,

    /// Total number of draft tokens accepted across all steps.
    pub accepted_tokens: usize,

    /// Wall-clock time for the decoding run, in milliseconds.
    pub wall_time_ms: f64,

    /// Average number of tokens produced per decoding step.
    ///
    /// For pure autoregressive decoding this is 1.0; speculative decoding
    /// aims for values > 1.0.
    pub tokens_per_step: f64,
}

impl DecodingStats {
    /// Create empty statistics.
    pub fn new() -> Self {
        Self {
            total_tokens: 0,
            draft_tokens: 0,
            accepted_tokens: 0,
            wall_time_ms: 0.0,
            tokens_per_step: 0.0,
        }
    }

    /// Overall acceptance rate across the entire decoding run.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_tokens == 0 {
            0.0
        } else {
            self.accepted_tokens as f64 / self.draft_tokens as f64
        }
    }

    /// Tokens generated per millisecond.
    pub fn throughput(&self) -> f64 {
        if self.wall_time_ms <= 0.0 {
            0.0
        } else {
            self.total_tokens as f64 / self.wall_time_ms
        }
    }
}

impl Default for DecodingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DecodingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DecodingStats(total={}, drafted={}, accepted={}, rate={:.2}, tok/step={:.2}, time={:.1}ms)",
            self.total_tokens,
            self.draft_tokens,
            self.accepted_tokens,
            self.acceptance_rate(),
            self.tokens_per_step,
            self.wall_time_ms,
        )
    }
}

/// A probability distribution over a vocabulary of tokens.
///
/// Wraps a dense vector of non-negative values that sum to 1.0 (within
/// floating-point tolerance). The index into the vector is the token id.
#[derive(Debug, Clone)]
pub struct TokenDistribution {
    /// Probability assigned to each token in the vocabulary.
    probs: Vec<f64>,
}

impl TokenDistribution {
    /// Create a distribution from a probability vector.
    ///
    /// Returns `None` if `probs` is empty or contains negative values.
    /// The vector is normalized to sum to 1.0.
    pub fn from_probs(probs: Vec<f64>) -> Option<Self> {
        if probs.is_empty() {
            return None;
        }
        // Check for negative values
        if probs.iter().any(|&p| p < 0.0) {
            return None;
        }
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 {
            return None;
        }
        let normalized: Vec<f64> = probs.iter().map(|&p| p / sum).collect();
        Some(Self { probs: normalized })
    }

    /// Create a uniform distribution over `vocab_size` tokens.
    pub fn uniform(vocab_size: usize) -> Option<Self> {
        if vocab_size == 0 {
            return None;
        }
        let p = 1.0 / vocab_size as f64;
        Some(Self {
            probs: vec![p; vocab_size],
        })
    }

    /// Create a distribution from log-probabilities.
    ///
    /// Applies the softmax transformation: `p_i = exp(logp_i) / sum(exp(logp_j))`.
    /// Uses the log-sum-exp trick for numerical stability.
    pub fn from_log_probs(log_probs: &[f64]) -> Option<Self> {
        if log_probs.is_empty() {
            return None;
        }
        let max_lp = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_lp.is_nan() {
            return None;
        }
        let exps: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            return None;
        }
        let probs: Vec<f64> = exps.iter().map(|&e| e / sum).collect();
        Some(Self { probs })
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.probs.len()
    }

    /// Probability of a given token.
    ///
    /// Returns 0.0 for out-of-range token ids.
    pub fn prob(&self, token_id: usize) -> f64 {
        self.probs.get(token_id).copied().unwrap_or(0.0)
    }

    /// Borrow the raw probability vector.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// Apply temperature scaling, returning a new distribution.
    ///
    /// Temperature < 1.0 sharpens; temperature > 1.0 flattens.
    /// Returns `None` if temperature is non-positive.
    pub fn with_temperature(&self, temperature: f64) -> Option<Self> {
        if temperature <= 0.0 {
            return None;
        }
        if (temperature - 1.0).abs() < 1e-12 {
            return Some(self.clone());
        }
        // Work in log space for stability
        let log_probs: Vec<f64> = self
            .probs
            .iter()
            .map(|&p| {
                if p > 0.0 {
                    p.ln() / temperature
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        Self::from_log_probs(&log_probs)
    }

    /// Apply top-k filtering, zeroing out all but the `k` highest-probability tokens.
    ///
    /// Returns `None` if `k` is 0.
    pub fn with_top_k(&self, k: usize) -> Option<Self> {
        if k == 0 {
            return None;
        }
        if k >= self.probs.len() {
            return Some(self.clone());
        }
        // Find the k-th largest probability
        let mut sorted: Vec<f64> = self.probs.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[k - 1];

        // Keep only tokens with prob >= threshold (may keep slightly more than k if ties)
        let filtered: Vec<f64> = self
            .probs
            .iter()
            .map(|&p| if p >= threshold { p } else { 0.0 })
            .collect();
        Self::from_probs(filtered)
    }

    /// Sample a token from this distribution using the provided random value
    /// `u` in \[0, 1).
    pub fn sample_with_uniform(&self, u: f64) -> usize {
        let u = u.clamp(0.0, 1.0 - f64::EPSILON);
        let mut cumulative = 0.0;
        for (i, &p) in self.probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        // Fallback: return last token (should not happen with proper normalization)
        self.probs.len().saturating_sub(1)
    }

    /// Token id with the highest probability (argmax).
    pub fn argmax(&self) -> usize {
        self.probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

impl fmt::Display for TokenDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let top = self.argmax();
        write!(
            f,
            "TokenDistribution(vocab={}, top_token={}, top_prob={:.4})",
            self.vocab_size(),
            top,
            self.prob(top),
        )
    }
}
