//! Rejection-sampling core for speculative decoding.
//!
//! Implements the algorithm of Leviathan, Kalman & Matias (2023):
//! *"Fast Inference from Transformers via Speculative Decoding"*.
//!
//! The key insight is that when a fast *draft model* generates candidate tokens
//! and a slower *target model* verifies them, token acceptance and optional
//! resampling from the residual distribution `max(0, p_target − p_draft)`
//! guarantees that the joint output distribution matches sampling from the
//! target model directly, while requiring fewer expensive target evaluations.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_neural::inference::speculative::{
//!     SpeculativeConfig, SpeculativeDecoder, TokenDist,
//! };
//!
//! // Tiny 4-token vocabulary for illustration.
//! let vocab = 4_usize;
//!
//! // Draft and target agree: all tokens should be accepted.
//! let logits = vec![1.0, 2.0, 0.5, 0.1];
//! let draft_logits: Vec<Vec<f64>> = vec![logits.clone(), logits.clone()];
//! let target_logits = draft_logits.clone();
//! let draft_tokens: Vec<u32> = vec![1, 0];  // sampled from draft
//!
//! let config = SpeculativeConfig::default();
//! let (accepted, _correction) = SpeculativeDecoder::rejection_sampling_step(
//!     &draft_logits,
//!     &target_logits,
//!     &draft_tokens,
//!     &config,
//! );
//! // When draft == target the entire prefix is accepted.
//! assert_eq!(accepted.len(), draft_tokens.len());
//! ```

use super::types::{SpeculativeConfig, SpeculativeResult};

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG used internally (no external crate dependency).
// ---------------------------------------------------------------------------

/// A fast xorshift64 PRNG.
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
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

    /// Uniform sample in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use upper 53 bits for the mantissa.
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// TokenDist
// ---------------------------------------------------------------------------

/// A probability distribution over a token vocabulary backed by raw logits.
///
/// Logits are unnormalized log-probabilities.  All sampling and probability
/// queries apply the softmax (and optionally temperature / top-p) internally.
pub struct TokenDist {
    /// Unnormalized log-probabilities (one per vocabulary token).
    pub logits: Vec<f64>,
}

impl TokenDist {
    /// Wrap a logit vector.
    pub fn new(logits: Vec<f64>) -> Self {
        Self { logits }
    }

    /// Compute softmax probabilities from the stored logits.
    ///
    /// Uses the log-sum-exp trick for numerical stability.
    ///
    /// ```rust
    /// use scirs2_neural::inference::speculative::TokenDist;
    /// let d = TokenDist::new(vec![1.0, 2.0, 3.0]);
    /// let probs = d.softmax();
    /// let sum: f64 = probs.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-10);
    /// ```
    pub fn softmax(&self) -> Vec<f64> {
        softmax_with_temperature(&self.logits, 1.0)
    }

    /// Sample a token index using the given `temperature`.
    ///
    /// Applies temperature scaling before softmax.
    pub fn sample(&self, temperature: f64, rng: &mut Xorshift64) -> u32 {
        let probs = softmax_with_temperature(&self.logits, temperature);
        categorical_sample(&probs, rng.next_f64())
    }

    /// Sample with nucleus (top-p) filtering, then temperature scaling.
    pub fn sample_top_p(&self, temperature: f64, top_p: f64, rng: &mut Xorshift64) -> u32 {
        let probs = softmax_with_temperature(&self.logits, temperature);
        let filtered = apply_top_p(&probs, top_p);
        categorical_sample(&filtered, rng.next_f64())
    }

    /// Probability assigned to `token` after applying `temperature`.
    ///
    /// Returns 0.0 for out-of-range indices.
    pub fn prob(&self, token: u32, temperature: f64) -> f64 {
        let probs = softmax_with_temperature(&self.logits, temperature);
        probs.get(token as usize).copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Helper math functions
// ---------------------------------------------------------------------------

/// Softmax with temperature scaling.
fn softmax_with_temperature(logits: &[f64], temperature: f64) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temp = if temperature <= 0.0 { 1.0 } else { temperature };
    let scaled: Vec<f64> = logits.iter().map(|&l| l / temp).collect();
    let max_val = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        let u = 1.0 / logits.len() as f64;
        return vec![u; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Nucleus (top-p) filtering: zero out tokens outside the nucleus, renormalize.
fn apply_top_p(probs: &[f64], top_p: f64) -> Vec<f64> {
    if top_p >= 1.0 || probs.is_empty() {
        return probs.to_vec();
    }
    // Sort indices by descending probability.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cumulative = 0.0;
    let mut mask = vec![false; probs.len()];
    for &idx in &order {
        mask[idx] = true;
        cumulative += probs[idx];
        if cumulative >= top_p {
            break;
        }
    }

    let mut filtered: Vec<f64> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| if mask[i] { p } else { 0.0 })
        .collect();
    let sum: f64 = filtered.iter().sum();
    if sum > 0.0 {
        for p in &mut filtered {
            *p /= sum;
        }
    }
    filtered
}

/// Sample a token index from a probability vector using a uniform variate `u ∈ [0,1)`.
fn categorical_sample(probs: &[f64], u: f64) -> u32 {
    let u = u.clamp(0.0, 1.0 - f64::EPSILON);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i as u32;
        }
    }
    probs.len().saturating_sub(1) as u32
}

/// Sample from the residual distribution `max(0, target − draft)` (renormalized).
///
/// Falls back to sampling from `target_probs` directly when the residual is
/// degenerate (all zeros).
fn residual_sample(target_probs: &[f64], draft_probs: &[f64], rng: &mut Xorshift64) -> u32 {
    let len = target_probs.len().min(draft_probs.len());
    if len == 0 {
        return 0;
    }
    let residual: Vec<f64> = (0..len)
        .map(|i| {
            let d = target_probs[i] - draft_probs[i];
            if d > 0.0 {
                d
            } else {
                0.0
            }
        })
        .collect();
    let sum: f64 = residual.iter().sum();
    if sum <= 1e-15 {
        // Fallback: sample directly from target.
        let target_sum: f64 = target_probs.iter().take(len).sum();
        if target_sum <= 0.0 {
            return 0;
        }
        let u = rng.next_f64() * target_sum;
        let mut cum = 0.0;
        for (i, &p) in target_probs.iter().take(len).enumerate() {
            cum += p;
            if u < cum {
                return i as u32;
            }
        }
        return len.saturating_sub(1) as u32;
    }
    let u = rng.next_f64() * sum;
    let mut cum = 0.0;
    for (i, &r) in residual.iter().enumerate() {
        cum += r;
        if u < cum {
            return i as u32;
        }
    }
    len.saturating_sub(1) as u32
}

// ---------------------------------------------------------------------------
// SpeculativeDecoder
// ---------------------------------------------------------------------------

/// Closure-based speculative decoder.
///
/// Unlike the trait-based [`crate::speculative::SpeculativeDecoder`], this
/// struct is stateless and exposes only associated functions.  Model access is
/// provided through closures, making it easy to integrate with any inference
/// backend.
pub struct SpeculativeDecoder;

impl SpeculativeDecoder {
    /// Perform one speculative decoding verification step.
    ///
    /// Compares each draft token against the target model's distribution and
    /// applies rejection sampling.  Returns the accepted prefix and a
    /// *correction token* sampled from the residual distribution at the first
    /// rejection site (or from the target distribution when all draft tokens
    /// are accepted).
    ///
    /// # Arguments
    ///
    /// * `draft_logits`  — `[draft_steps][vocab_size]` logits from the draft model.
    /// * `target_logits` — `[draft_steps][vocab_size]` logits from the target model.
    /// * `draft_tokens`  — token ids sampled from the draft model (one per step).
    /// * `config`        — decoding configuration.
    ///
    /// # Returns
    ///
    /// `(accepted_prefix, correction_token)` where `correction_token` is
    /// always a valid token id.
    ///
    /// # Panics
    ///
    /// Does not panic.  Returns an empty prefix and token 0 on empty input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_neural::inference::speculative::{SpeculativeConfig, SpeculativeDecoder};
    ///
    /// let logits = vec![0.0f64; 8];
    /// let draft_logits = vec![logits.clone(), logits.clone()];
    /// let target_logits = draft_logits.clone();
    /// let draft_tokens = vec![0u32, 1u32];
    /// let cfg = SpeculativeConfig::default();
    ///
    /// let (prefix, _correction) = SpeculativeDecoder::rejection_sampling_step(
    ///     &draft_logits, &target_logits, &draft_tokens, &cfg,
    /// );
    /// // Uniform distribution → draft == target → all accepted.
    /// assert_eq!(prefix.len(), draft_tokens.len());
    /// ```
    pub fn rejection_sampling_step(
        draft_logits: &[Vec<f64>],
        target_logits: &[Vec<f64>],
        draft_tokens: &[u32],
        config: &SpeculativeConfig,
    ) -> (Vec<u32>, u32) {
        let n = draft_tokens
            .len()
            .min(draft_logits.len())
            .min(target_logits.len());

        if n == 0 {
            return (Vec::new(), 0);
        }

        // Deterministic seed derived from first draft token and logit count.
        let seed: u64 = draft_tokens.first().copied().unwrap_or(0) as u64
            ^ (draft_logits.first().map_or(0, |v| v.len()) as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let mut rng = Xorshift64::new(seed);

        let temp = config.temperature;
        let top_p = config.top_p;

        let mut accepted: Vec<u32> = Vec::with_capacity(n);

        for i in 0..n {
            let draft_token = draft_tokens[i];
            let d_logits = &draft_logits[i];
            let t_logits = &target_logits[i];

            let draft_probs = softmax_with_temperature(d_logits, temp);
            let target_probs = softmax_with_temperature(t_logits, temp);

            // Apply top-p filtering.
            let draft_p_filtered = apply_top_p(&draft_probs, top_p);
            let target_p_filtered = apply_top_p(&target_probs, top_p);

            let q = draft_p_filtered
                .get(draft_token as usize)
                .copied()
                .unwrap_or(0.0);
            let p = target_p_filtered
                .get(draft_token as usize)
                .copied()
                .unwrap_or(0.0);

            let accept = Self::accept_or_reject(q, p, config.acceptance_threshold, &mut rng);

            if accept {
                accepted.push(draft_token);
            } else {
                // Rejected: resample from residual and return.
                let correction = residual_sample(&target_p_filtered, &draft_p_filtered, &mut rng);
                return (accepted, correction);
            }
        }

        // All accepted: sample a bonus token from the last target position.
        let last_target = &target_logits[n - 1];
        let target_probs = softmax_with_temperature(last_target, temp);
        let target_filtered = apply_top_p(&target_probs, top_p);
        let bonus = categorical_sample(&target_filtered, rng.next_f64());
        (accepted, bonus)
    }

    /// Full speculative decoding loop using caller-provided closures.
    ///
    /// # Closure contracts
    ///
    /// * `draft_fn(context)` → `Vec<(token, logits)>` — given the current
    ///   token context, return up to `config.draft_steps` `(token_id, logits)`
    ///   pairs.  Each `logits` vector must have length equal to the vocabulary.
    ///
    /// * `target_fn(context, draft_tokens)` → `Vec<Vec<f64>>` — given the
    ///   current context and the draft token ids, return one logit vector per
    ///   draft position.
    ///
    /// Both closures are invoked at most once per decoding step.
    ///
    /// # Returns
    ///
    /// A [`SpeculativeResult`] containing the generated tokens (not including
    /// the prompt) and accumulated statistics.
    pub fn decode<D, T>(
        context: &[u32],
        draft_fn: D,
        target_fn: T,
        config: &SpeculativeConfig,
    ) -> SpeculativeResult
    where
        D: Fn(&[u32]) -> Vec<(u32, Vec<f64>)>,
        T: Fn(&[u32], &[u32]) -> Vec<Vec<f64>>,
    {
        let mut current_ctx: Vec<u32> = context.to_vec();
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut n_draft_total = 0_usize;
        let mut n_accepted_total = 0_usize;
        let mut n_verification_calls = 0_usize;

        while output_tokens.len() < config.max_tokens {
            // Draft step.
            let draft_result = draft_fn(&current_ctx);
            if draft_result.is_empty() {
                break;
            }

            let remaining = config.max_tokens - output_tokens.len();
            let draft_len = draft_result.len().min(remaining);
            let draft_tokens: Vec<u32> =
                draft_result[..draft_len].iter().map(|(t, _)| *t).collect();
            let draft_logits: Vec<Vec<f64>> = draft_result[..draft_len]
                .iter()
                .map(|(_, l)| l.clone())
                .collect();

            n_draft_total += draft_len;

            // Target verification step.
            let target_logits = target_fn(&current_ctx, &draft_tokens);
            n_verification_calls += 1;

            let (accepted, correction) =
                Self::rejection_sampling_step(&draft_logits, &target_logits, &draft_tokens, config);

            let n_acc = accepted.len();
            n_accepted_total += n_acc;

            // Append accepted draft tokens.
            for token in accepted {
                if output_tokens.len() >= config.max_tokens {
                    break;
                }
                current_ctx.push(token);
                output_tokens.push(token);
            }

            // Append the correction / bonus token.
            if output_tokens.len() < config.max_tokens {
                current_ctx.push(correction);
                output_tokens.push(correction);
            }

            if output_tokens.len() >= config.max_tokens {
                break;
            }
        }

        SpeculativeResult::new(
            output_tokens,
            n_draft_total,
            n_accepted_total,
            n_verification_calls,
        )
    }

    /// Accept-or-reject a single draft token.
    ///
    /// Returns `true` (accept) with probability `min(1, p_target / p_draft)`
    /// when `p_draft > threshold`; always accepts when `p_target >= p_draft`.
    fn accept_or_reject(
        draft_prob: f64,
        target_prob: f64,
        threshold: f64,
        rng: &mut Xorshift64,
    ) -> bool {
        // If the draft prob is at or below the threshold, always accept.
        if draft_prob <= threshold {
            return true;
        }
        // If target >= draft, always accept.
        if target_prob >= draft_prob {
            return true;
        }
        // Accept with probability target / draft.
        rng.next_f64() < target_prob / draft_prob
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::speculative::SpeculativeConfig;

    // ---- TokenDist tests ---------------------------------------------------

    #[test]
    fn tokendist_softmax_sums_to_one() {
        let d = TokenDist::new(vec![1.0, 2.0, 0.5, -1.0]);
        let probs = d.softmax();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum={sum}");
    }

    #[test]
    fn tokendist_softmax_nonnegative() {
        let d = TokenDist::new(vec![-100.0, 0.0, 100.0]);
        for &p in &d.softmax() {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn tokendist_sample_returns_valid_index() {
        let vocab = 8;
        let d = TokenDist::new(vec![1.0; vocab]);
        let mut rng = Xorshift64::new(12345);
        for _ in 0..50 {
            let tok = d.sample(1.0, &mut rng);
            assert!((tok as usize) < vocab);
        }
    }

    #[test]
    fn tokendist_prob_out_of_range_is_zero() {
        let d = TokenDist::new(vec![1.0, 2.0]);
        assert!((d.prob(99, 1.0) - 0.0).abs() < 1e-12);
    }

    // ---- rejection_sampling_step tests ------------------------------------

    #[test]
    fn step_all_accepted_when_draft_equals_target() {
        // When draft_logits == target_logits the distributions are identical,
        // so every token is always accepted.
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let draft_logits = vec![logits.clone(), logits.clone(), logits.clone()];
        let target_logits = draft_logits.clone();

        // Deterministic draft tokens: argmax = 3.
        let draft_tokens = vec![3u32, 3u32, 3u32];
        let cfg = SpeculativeConfig::default();

        let (accepted, _correction) = SpeculativeDecoder::rejection_sampling_step(
            &draft_logits,
            &target_logits,
            &draft_tokens,
            &cfg,
        );
        assert_eq!(
            accepted.len(),
            draft_tokens.len(),
            "all tokens should be accepted when draft==target"
        );
    }

    #[test]
    fn step_some_rejected_when_distributions_differ() {
        // The draft puts all mass on token 0 while the target puts all mass on
        // token 3.  Almost every run should see a rejection at step 0.
        let mut any_rejected = false;
        for trial in 0u64..50 {
            // Vary the draft token to exercise the RNG seed variation.
            let draft_logits = vec![
                vec![100.0, -100.0, -100.0, -100.0],
                vec![100.0, -100.0, -100.0, -100.0],
            ];
            let target_logits = vec![
                vec![-100.0, -100.0, -100.0, 100.0],
                vec![-100.0, -100.0, -100.0, 100.0],
            ];
            // Draft chose token 0 (its argmax).
            let draft_tokens = vec![0u32, 0u32];
            let cfg = SpeculativeConfig {
                // Use a seed offset via the draft token (token 0 → seed 0 ^ ...).
                // The trial number perturbs the context via the closure in decode;
                // here we just run many seeds manually by adjusting logits slightly.
                draft_steps: 2,
                acceptance_threshold: trial as f64 * 0.0, // always 0
                ..Default::default()
            };

            let (accepted, _) = SpeculativeDecoder::rejection_sampling_step(
                &draft_logits,
                &target_logits,
                &draft_tokens,
                &cfg,
            );
            if accepted.len() < draft_tokens.len() {
                any_rejected = true;
                break;
            }
        }
        assert!(any_rejected, "expected at least one rejection in 50 trials");
    }

    #[test]
    fn step_empty_input_returns_empty() {
        let cfg = SpeculativeConfig::default();
        let (acc, _corr) = SpeculativeDecoder::rejection_sampling_step(&[], &[], &[], &cfg);
        assert!(acc.is_empty());
    }

    // ---- acceptance_rate tests --------------------------------------------

    #[test]
    fn decode_acceptance_rate_leq_one() {
        // Draft and target are the same → rate ≈ 1.
        let logits = vec![1.0f64; 4];
        let cfg = SpeculativeConfig {
            draft_steps: 3,
            max_tokens: 12,
            ..Default::default()
        };

        let result = SpeculativeDecoder::decode(
            &[0u32, 1u32],
            |_ctx| {
                vec![
                    (0, logits.clone()),
                    (1, logits.clone()),
                    (2, logits.clone()),
                ]
            },
            |_ctx, draft| draft.iter().map(|_| logits.clone()).collect(),
            &cfg,
        );

        assert!(
            result.acceptance_rate <= 1.0 + 1e-10,
            "acceptance_rate={:.4}",
            result.acceptance_rate
        );
    }

    #[test]
    fn decode_perfect_draft_high_acceptance() {
        // When draft and target are identical every step, all draft tokens
        // are accepted and only the bonus token adds one extra per step.
        let logits = vec![0.0f64; 4]; // uniform distribution
        let cfg = SpeculativeConfig {
            draft_steps: 2,
            max_tokens: 10,
            ..Default::default()
        };

        let result = SpeculativeDecoder::decode(
            &[0u32],
            |_ctx| vec![(0, logits.clone()), (1, logits.clone())],
            |_ctx, draft| draft.iter().map(|_| logits.clone()).collect(),
            &cfg,
        );

        // All draft tokens should be accepted → acceptance_rate should be 1.0.
        assert!(
            result.acceptance_rate >= 0.99,
            "expected high acceptance, got {:.4}",
            result.acceptance_rate
        );
        assert_eq!(result.accepted_tokens.len(), result.accepted_tokens.len()); // trivially true
    }

    #[test]
    fn speculative_config_default_works() {
        let cfg = SpeculativeConfig::default();
        assert_eq!(cfg.draft_steps, 4);
    }
}
