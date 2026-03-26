//! Configuration and result types for closure-based speculative decoding.
//!
//! These types complement the trait-based [`crate::speculative`] module by
//! providing a lighter-weight, closure-driven API.  No external model traits
//! are required; instead the caller passes draft and target closures directly
//! to [`super::rejection_sampling::SpeculativeDecoder::decode`].

/// Configuration for the closure-based speculative decoder.
///
/// # Defaults
///
/// ```rust
/// use scirs2_neural::inference::speculative::SpeculativeConfig;
/// let cfg = SpeculativeConfig::default();
/// assert_eq!(cfg.draft_steps, 4);
/// assert!((cfg.temperature - 1.0).abs() < 1e-12);
/// assert!((cfg.top_p - 0.9).abs() < 1e-12);
/// assert_eq!(cfg.max_tokens, 256);
/// assert!((cfg.acceptance_threshold - 0.0).abs() < 1e-12);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens the draft model proposes per decoding step.
    ///
    /// Higher values amortise target-model overhead but decrease acceptance
    /// rates when the draft diverges from the target.  Typical range: 2–8.
    pub draft_steps: usize,

    /// Sampling temperature applied when drawing from draft/target distributions.
    ///
    /// Values below 1.0 make the distribution sharper (more greedy); values
    /// above 1.0 make it flatter (more diverse).
    pub temperature: f64,

    /// Nucleus (top-p) sampling threshold in (0.0, 1.0].
    ///
    /// Only the smallest set of tokens whose cumulative probability reaches
    /// `top_p` are considered during sampling.  Set to 1.0 to disable.
    pub top_p: f64,

    /// Maximum number of **new** tokens to generate (not counting the prompt).
    pub max_tokens: usize,

    /// Minimum acceptance probability for the rejection-sampling criterion.
    ///
    /// A value of `0.0` corresponds to the exact rejection-sampling algorithm
    /// of Leviathan et al. (2023), which preserves the target distribution.
    /// Positive values introduce a floor that trades distribution fidelity for
    /// higher acceptance rates.
    pub acceptance_threshold: f64,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            draft_steps: 4,
            temperature: 1.0,
            top_p: 0.9,
            max_tokens: 256,
            acceptance_threshold: 0.0,
        }
    }
}

/// Summary statistics for a complete speculative decoding run.
#[derive(Debug, Clone)]
pub struct SpeculativeResult {
    /// Token ids accepted in the final output (not including the initial
    /// prompt tokens).
    pub accepted_tokens: Vec<u32>,

    /// Total number of draft tokens proposed across all steps.
    pub n_draft_tokens: usize,

    /// Total number of draft tokens that were accepted by the verifier.
    pub n_accepted_tokens: usize,

    /// Fraction of drafted tokens that were accepted: `n_accepted / n_draft`.
    /// In the range `[0.0, 1.0]`.
    pub acceptance_rate: f64,

    /// Number of times the target model's closure was invoked.
    pub n_verification_calls: usize,
}

impl SpeculativeResult {
    /// Construct a result, computing `acceptance_rate` automatically.
    pub(crate) fn new(
        accepted_tokens: Vec<u32>,
        n_draft_tokens: usize,
        n_accepted_tokens: usize,
        n_verification_calls: usize,
    ) -> Self {
        let acceptance_rate = if n_draft_tokens == 0 {
            0.0
        } else {
            n_accepted_tokens as f64 / n_draft_tokens as f64
        };
        Self {
            accepted_tokens,
            n_draft_tokens,
            n_accepted_tokens,
            acceptance_rate,
            n_verification_calls,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speculative_config_default() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.draft_steps, 4);
        assert!((cfg.temperature - 1.0).abs() < 1e-12);
        assert!((cfg.top_p - 0.9).abs() < 1e-12);
        assert_eq!(cfg.max_tokens, 256);
        assert!((cfg.acceptance_threshold - 0.0).abs() < 1e-12);
    }

    #[test]
    fn speculative_result_acceptance_rate_zero_drafts() {
        let r = SpeculativeResult::new(vec![], 0, 0, 0);
        assert!((r.acceptance_rate - 0.0).abs() < 1e-12);
    }

    #[test]
    fn speculative_result_acceptance_rate_computed() {
        let r = SpeculativeResult::new(vec![1, 2, 3], 4, 3, 1);
        assert!((r.acceptance_rate - 0.75).abs() < 1e-12);
    }

    #[test]
    fn speculative_result_leq_one() {
        for (drafted, accepted) in [(10, 8), (5, 5), (3, 0)] {
            let r = SpeculativeResult::new(vec![], drafted, accepted, 0);
            assert!(r.acceptance_rate <= 1.0 + 1e-12);
        }
    }
}
