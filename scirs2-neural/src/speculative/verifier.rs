//! Verification of draft tokens against a target model.
//!
//! Implements the rejection-sampling core of speculative decoding: for each
//! draft token, compare the draft model's probability to the target model's
//! probability and accept or reject accordingly. On rejection, resample from
//! the adjusted distribution `max(0, p_target - p_draft)`.

use super::draft::Xorshift64;
use super::types::VerificationResult;

/// Trait for the authoritative target model.
///
/// The target model provides the "ground truth" log-probabilities over the
/// vocabulary given a context. It is typically a large, expensive model whose
/// calls we want to minimize via speculative decoding.
pub trait TargetModel: Send + Sync {
    /// Return log-probabilities over the full vocabulary given a context.
    ///
    /// The returned vector must have length equal to the vocabulary size.
    /// Values are natural logarithms (base e).
    fn log_probs(&self, context: &[usize]) -> Vec<f64>;

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Verifier that performs rejection sampling on draft tokens.
///
/// Given a batch of draft tokens with their draft-model probabilities and
/// the corresponding target-model probabilities, decides which tokens to
/// accept and, on rejection, resamples from the adjusted distribution.
#[derive(Debug, Clone)]
pub struct SpeculativeVerifier {
    /// PRNG for rejection sampling decisions.
    rng: Xorshift64,
}

impl SpeculativeVerifier {
    /// Create a new verifier with the given PRNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Xorshift64::new(seed),
        }
    }

    /// Verify a batch of draft tokens against target probabilities.
    ///
    /// # Algorithm
    ///
    /// For each position `i` in the draft:
    ///
    /// 1. Let `q_i = draft_probs[i]` (draft probability of draft token `i`)
    /// 2. Let `p_i = target_probs[i]` (target probability of draft token `i`)
    /// 3. If `p_i >= q_i`, accept unconditionally.
    /// 4. Else accept with probability `p_i / q_i`.
    /// 5. On rejection, resample from `adjusted_distribution(target_probs_full, draft_probs_full)`
    ///    and stop processing further draft tokens.
    ///
    /// # Arguments
    ///
    /// * `draft_tokens` — token ids proposed by the draft model.
    /// * `draft_probs` — probability the draft model assigned to each draft token.
    /// * `target_probs_per_position` — for each draft position, the full target
    ///   probability distribution over the vocabulary (as a Vec of Vecs).
    /// * `draft_probs_per_position` — for each draft position, the full draft
    ///   probability distribution over the vocabulary (as a Vec of Vecs).
    ///
    /// # Returns
    ///
    /// A [`VerificationResult`] with accepted tokens, rejection position, and
    /// acceptance rate.
    pub fn verify_draft(
        &mut self,
        draft_tokens: &[usize],
        draft_probs: &[f64],
        target_probs_per_position: &[Vec<f64>],
        draft_probs_per_position: &[Vec<f64>],
    ) -> VerificationResult {
        let n = draft_tokens.len();
        if n == 0 {
            return VerificationResult::new(Vec::new(), None, 1.0);
        }

        let mut accepted = Vec::with_capacity(n);

        for i in 0..n {
            let token = draft_tokens[i];
            let q = draft_probs[i]; // draft prob for the chosen token
            let p = target_probs_per_position
                .get(i)
                .and_then(|v| v.get(token))
                .copied()
                .unwrap_or(0.0);

            // Acceptance criterion
            if q > 0.0 && p < q {
                // Accept with probability p/q
                let u = self.rng.next_f64();
                if u >= p / q {
                    // Rejected: resample from adjusted distribution
                    let target_full = target_probs_per_position.get(i);
                    let draft_full = draft_probs_per_position.get(i);
                    if let (Some(target_f), Some(draft_f)) = (target_full, draft_full) {
                        let resampled = self.adjusted_sample(target_f, draft_f);
                        accepted.push(resampled);
                    }

                    let rate = if i == 0 { 0.0 } else { i as f64 / n as f64 };
                    return VerificationResult::new(accepted, Some(i), rate);
                }
            }
            // p >= q or passed probabilistic acceptance
            accepted.push(token);
        }

        VerificationResult::new(accepted, None, 1.0)
    }

    /// Simple verification without full distribution information.
    ///
    /// This is a convenience wrapper when you only have the probabilities for
    /// the specific draft tokens (not the full distributions). On rejection,
    /// no resampling is performed.
    pub fn verify_simple(
        &mut self,
        draft_tokens: &[usize],
        draft_probs: &[f64],
        target_probs_for_tokens: &[f64],
    ) -> VerificationResult {
        let n = draft_tokens.len();
        if n == 0 {
            return VerificationResult::new(Vec::new(), None, 1.0);
        }

        let mut accepted = Vec::with_capacity(n);

        for i in 0..n {
            let q = draft_probs[i];
            let p = target_probs_for_tokens[i];

            if q > 0.0 && p < q {
                let u = self.rng.next_f64();
                if u >= p / q {
                    let rate = i as f64 / n as f64;
                    return VerificationResult::new(accepted, Some(i), rate);
                }
            }

            accepted.push(draft_tokens[i]);
        }

        VerificationResult::new(accepted, None, 1.0)
    }

    /// Sample from the adjusted distribution `max(0, p_target - p_draft)`, normalized.
    ///
    /// This distribution corrects for the bias introduced by the draft model:
    /// tokens where the target assigns more probability than the draft are
    /// up-weighted, while tokens where the draft over-estimates are zeroed out.
    ///
    /// # Arguments
    ///
    /// * `target_probs` — target model probabilities over vocabulary.
    /// * `draft_probs` — draft model probabilities over vocabulary.
    ///
    /// # Returns
    ///
    /// A token id sampled from the adjusted distribution.
    pub fn adjusted_sample(&mut self, target_probs: &[f64], draft_probs: &[f64]) -> usize {
        adjusted_sample_with_rng(target_probs, draft_probs, &mut self.rng)
    }
}

/// Sample from `max(0, p_target - p_draft)` normalized, using the given PRNG.
///
/// If the adjusted distribution sums to zero (which can happen when the draft
/// perfectly matches or dominates the target), falls back to sampling from the
/// target distribution directly.
fn adjusted_sample_with_rng(
    target_probs: &[f64],
    draft_probs: &[f64],
    rng: &mut Xorshift64,
) -> usize {
    let len = target_probs.len().min(draft_probs.len());
    if len == 0 {
        return 0;
    }

    // Compute max(0, p_target - p_draft)
    let adjusted: Vec<f64> = (0..len)
        .map(|i| {
            let diff = target_probs[i] - draft_probs[i];
            if diff > 0.0 {
                diff
            } else {
                0.0
            }
        })
        .collect();

    let sum: f64 = adjusted.iter().sum();

    if sum <= 1e-15 {
        // Fallback: sample from target distribution
        let target_sum: f64 = target_probs.iter().take(len).sum();
        if target_sum <= 0.0 {
            return 0;
        }
        let u = rng.next_f64() * target_sum;
        let mut cumulative = 0.0;
        for (i, &tp) in target_probs.iter().enumerate().take(len) {
            cumulative += tp;
            if u < cumulative {
                return i;
            }
        }
        return len.saturating_sub(1);
    }

    // Normalize and sample
    let u = rng.next_f64() * sum;
    let mut cumulative = 0.0;
    for (i, &adj) in adjusted.iter().enumerate().take(len) {
        cumulative += adj;
        if u < cumulative {
            return i;
        }
    }

    len.saturating_sub(1)
}

/// Compute the adjusted distribution `max(0, p_target - p_draft)`, normalized.
///
/// Useful for testing and inspection.
pub fn compute_adjusted_distribution(target_probs: &[f64], draft_probs: &[f64]) -> Vec<f64> {
    let len = target_probs.len().min(draft_probs.len());
    if len == 0 {
        return Vec::new();
    }

    let adjusted: Vec<f64> = (0..len)
        .map(|i| {
            let diff = target_probs[i] - draft_probs[i];
            if diff > 0.0 {
                diff
            } else {
                0.0
            }
        })
        .collect();

    let sum: f64 = adjusted.iter().sum();
    if sum <= 1e-15 {
        return vec![0.0; len];
    }

    adjusted.iter().map(|&a| a / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock target model with fixed log-probs for testing.
    struct MockTarget {
        /// Fixed probability distribution to return.
        probs: Vec<f64>,
    }

    impl MockTarget {
        fn new(probs: Vec<f64>) -> Self {
            Self { probs }
        }
    }

    impl TargetModel for MockTarget {
        fn log_probs(&self, _context: &[usize]) -> Vec<f64> {
            self.probs.iter().map(|&p| p.ln()).collect()
        }

        fn vocab_size(&self) -> usize {
            self.probs.len()
        }
    }

    #[test]
    fn test_verification_accepts_when_draft_equals_target() {
        let mut verifier = SpeculativeVerifier::new(42);

        // Draft and target have the same distribution
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let draft_tokens = vec![0, 1, 2];
        let draft_probs = vec![0.25, 0.25, 0.25];
        let target_full = vec![probs.clone(), probs.clone(), probs.clone()];
        let draft_full = vec![probs.clone(), probs.clone(), probs.clone()];

        let result = verifier.verify_draft(&draft_tokens, &draft_probs, &target_full, &draft_full);

        // When p == q, always accept
        assert!(result.all_accepted());
        assert_eq!(result.num_accepted(), 3);
        assert!((result.acceptance_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_verification_accepts_when_target_dominates() {
        let mut verifier = SpeculativeVerifier::new(42);

        // Target assigns higher probability to drafted tokens
        let draft_probs_val = vec![0.1, 0.1, 0.1];
        let draft_tokens = vec![0, 1, 2];

        let target_dist = vec![0.5, 0.3, 0.15, 0.05]; // high prob for tokens 0,1,2
        let draft_dist = vec![0.1, 0.1, 0.1, 0.7]; // low prob for tokens 0,1,2

        let target_full = vec![
            target_dist.clone(),
            target_dist.clone(),
            target_dist.clone(),
        ];
        let draft_full = vec![draft_dist.clone(), draft_dist.clone(), draft_dist.clone()];

        let result =
            verifier.verify_draft(&draft_tokens, &draft_probs_val, &target_full, &draft_full);

        // p_target > p_draft for tokens 0, 1, 2, so always accept
        assert!(result.all_accepted());
    }

    #[test]
    fn test_verification_rejects_when_draft_very_different() {
        // Test over many runs that rejection eventually happens
        let mut any_rejected = false;

        for seed in 0..100 {
            let mut verifier = SpeculativeVerifier::new(seed + 1);

            // Draft assigns high probability, target assigns very low
            let draft_tokens = vec![0, 0, 0, 0, 0];
            let draft_probs_val = vec![0.9, 0.9, 0.9, 0.9, 0.9];

            let target_dist = vec![0.01, 0.01, 0.01, 0.97]; // token 0 has very low target prob
            let draft_dist = vec![0.9, 0.03, 0.03, 0.04];

            let target_full = vec![target_dist.clone(); 5];
            let draft_full = vec![draft_dist.clone(); 5];

            let result =
                verifier.verify_draft(&draft_tokens, &draft_probs_val, &target_full, &draft_full);

            if result.rejected_at.is_some() {
                any_rejected = true;
                break;
            }
        }

        assert!(any_rejected, "should reject at least once over 100 trials");
    }

    #[test]
    fn test_simple_verification_no_resample() {
        let mut verifier = SpeculativeVerifier::new(42);

        let draft_tokens = vec![0, 1];
        let draft_probs = vec![0.5, 0.5];
        let target_probs = vec![0.5, 0.5]; // equal -> always accept

        let result = verifier.verify_simple(&draft_tokens, &draft_probs, &target_probs);
        assert!(result.all_accepted());
    }

    #[test]
    fn test_adjusted_distribution() {
        let target = vec![0.5, 0.3, 0.2];
        let draft = vec![0.2, 0.5, 0.3];

        let adjusted = compute_adjusted_distribution(&target, &draft);

        // max(0, 0.5-0.2) = 0.3, max(0, 0.3-0.5) = 0.0, max(0, 0.2-0.3) = 0.0
        // normalized: [1.0, 0.0, 0.0]
        assert!((adjusted[0] - 1.0).abs() < 1e-10);
        assert!(adjusted[1].abs() < 1e-10);
        assert!(adjusted[2].abs() < 1e-10);
    }

    #[test]
    fn test_adjusted_distribution_empty() {
        let result = compute_adjusted_distribution(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_adjusted_sample_falls_back() {
        let mut rng = Xorshift64::new(42);

        // Draft dominates target everywhere -> adjusted is all zeros -> fallback
        let target = vec![0.1, 0.2, 0.7];
        let draft = vec![0.3, 0.3, 0.9];

        // Should fall back to sampling from target
        let token = adjusted_sample_with_rng(&target, &draft, &mut rng);
        assert!(token < 3);
    }

    #[test]
    fn test_verify_empty_draft() {
        let mut verifier = SpeculativeVerifier::new(42);
        let result = verifier.verify_draft(&[], &[], &[], &[]);
        assert!(result.all_accepted());
        assert_eq!(result.num_accepted(), 0);
    }

    #[test]
    fn test_rejection_sampling_distribution_correctness() {
        // Statistical test: the adjusted_sample function should produce
        // tokens weighted toward where target > draft.
        // Target = [0.6, 0.3, 0.1], Draft = [0.2, 0.5, 0.3]
        // Adjusted = max(0, target - draft) = [0.4, 0.0, 0.0] -> normalized [1.0, 0.0, 0.0]
        // So adjusted_sample should always produce token 0.
        let target_probs = vec![0.6, 0.3, 0.1];
        let draft_probs = vec![0.2, 0.5, 0.3];

        let mut counts = [0usize; 3];
        let n_trials = 1_000;

        for trial in 0..n_trials {
            let mut verifier = SpeculativeVerifier::new(trial as u64 + 1);
            let token = verifier.adjusted_sample(&target_probs, &draft_probs);
            if token < 3 {
                counts[token] += 1;
            }
        }

        // Token 0 should get all samples since it's the only one where target > draft
        let total: usize = counts.iter().sum();
        assert!(total > 0, "should have produced some samples");
        let frac_0 = counts[0] as f64 / total as f64;
        assert!(frac_0 > 0.95, "token 0 should dominate: frac={frac_0:.3}");

        // Second test: more balanced adjusted distribution
        // Target = [0.5, 0.3, 0.2], Draft = [0.1, 0.1, 0.1]
        // Adjusted = [0.4, 0.2, 0.1] -> normalized [4/7, 2/7, 1/7]
        let target2 = vec![0.5, 0.3, 0.2];
        let draft2 = vec![0.1, 0.1, 0.1];
        let expected = [4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0];

        let mut counts2 = [0usize; 3];
        let n_trials2 = 10_000;

        // Use a single verifier with a well-seeded PRNG to avoid correlation
        // from small sequential seeds.
        let mut verifier2 = SpeculativeVerifier::new(0xDEAD_BEEF_CAFE_1234);

        for _ in 0..n_trials2 {
            let token = verifier2.adjusted_sample(&target2, &draft2);
            if token < 3 {
                counts2[token] += 1;
            }
        }

        let total2: usize = counts2.iter().sum();
        if total2 > 100 {
            let empirical: Vec<f64> = counts2.iter().map(|&c| c as f64 / total2 as f64).collect();
            for i in 0..3 {
                let diff = (empirical[i] - expected[i]).abs();
                assert!(
                    diff < 0.1,
                    "token {i}: empirical={:.3}, expected={:.3}, diff={:.3}",
                    empirical[i],
                    expected[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_mock_target_model_trait() {
        let target = MockTarget::new(vec![0.5, 0.3, 0.2]);
        assert_eq!(target.vocab_size(), 3);
        let lp = target.log_probs(&[0]);
        assert_eq!(lp.len(), 3);
        // log(0.5) ≈ -0.693
        assert!((lp[0] - 0.5_f64.ln()).abs() < 1e-10);
    }
}
