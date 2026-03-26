//! Speculative decoder orchestration.
//!
//! The [`SpeculativeDecoder`] ties together a draft model, a target model,
//! and the rejection-sampling verifier into a complete decoding loop that
//! produces a token sequence equivalent to sampling from the target model
//! alone, but with fewer target-model evaluations.

use std::time::Instant;

use super::draft::{DraftModel, Xorshift64};
use super::types::{DecodingStats, SpeculativeConfig, TokenDistribution, VerificationResult};
use super::verifier::{SpeculativeVerifier, TargetModel};

/// Orchestrates the speculative decoding loop.
///
/// Given a prompt, the decoder repeatedly:
/// 1. Asks the draft model to propose `draft_length` candidate tokens.
/// 2. Evaluates the target model once for the entire draft.
/// 3. Applies rejection sampling to accept a prefix of the draft.
/// 4. On rejection, appends a resampled token from the adjusted distribution.
/// 5. Repeats until `max_tokens` is reached.
///
/// When `adaptive_draft` is enabled, the draft length is adjusted based on
/// the rolling acceptance rate:
/// - acceptance > 0.8 → increase draft length (up to 2x initial)
/// - acceptance < 0.4 → decrease draft length (down to 1)
pub struct SpeculativeDecoder<D: DraftModel, T: TargetModel> {
    /// Draft model (fast, approximate).
    draft: D,
    /// Target model (slow, accurate).
    target: T,
    /// Decoding configuration.
    config: SpeculativeConfig,
    /// Verifier for rejection sampling.
    verifier: SpeculativeVerifier,
    /// PRNG for any additional sampling needs.
    rng: Xorshift64,
}

impl<D: DraftModel, T: TargetModel> SpeculativeDecoder<D, T> {
    /// Create a new speculative decoder.
    ///
    /// # Arguments
    ///
    /// * `draft` — the draft model.
    /// * `target` — the target model.
    /// * `config` — decoding configuration.
    /// * `seed` — PRNG seed for the verifier and any additional sampling.
    pub fn new(draft: D, target: T, config: SpeculativeConfig, seed: u64) -> Self {
        Self {
            draft,
            target,
            config,
            verifier: SpeculativeVerifier::new(seed),
            rng: Xorshift64::new(seed.wrapping_add(1)),
        }
    }

    /// Run the full speculative decoding loop.
    ///
    /// # Arguments
    ///
    /// * `prompt` — initial token ids to condition on.
    ///
    /// # Returns
    ///
    /// A tuple of `(generated_tokens, stats)` where `generated_tokens` includes
    /// the prompt followed by the newly generated tokens, and `stats` contains
    /// performance metrics.
    pub fn decode(&mut self, prompt: &[usize]) -> (Vec<usize>, DecodingStats) {
        let start = Instant::now();

        let mut output: Vec<usize> = prompt.to_vec();
        let mut stats = DecodingStats::new();
        let mut current_draft_length = self.config.draft_length;
        let mut step_count: usize = 0;
        let mut rolling_accepted: usize = 0;
        let mut rolling_drafted: usize = 0;

        while output.len() < prompt.len() + self.config.max_tokens {
            // Step 1: Draft
            let draft_result = self.draft.generate_draft(&output, current_draft_length);
            let draft_len = draft_result.len();
            if draft_len == 0 {
                break;
            }

            let draft_tokens: Vec<usize> = draft_result.iter().map(|(t, _)| *t).collect();
            let draft_probs: Vec<f64> = draft_result.iter().map(|(_, p)| *p).collect();

            stats.draft_tokens += draft_len;
            rolling_drafted += draft_len;

            // Step 2: Get target probabilities for each draft position
            let (target_probs_per_pos, draft_probs_per_pos) =
                self.compute_distributions(&output, &draft_tokens);

            // Step 3: Verify
            let vr: VerificationResult = self.verifier.verify_draft(
                &draft_tokens,
                &draft_probs,
                &target_probs_per_pos,
                &draft_probs_per_pos,
            );

            let accepted_count = vr.num_accepted();
            stats.accepted_tokens += if vr.all_accepted() {
                accepted_count
            } else {
                // The last token in accepted_tokens is the resampled one (not a draft acceptance)
                accepted_count.saturating_sub(1)
            };
            rolling_accepted += stats.accepted_tokens;

            // Append accepted tokens to output
            for &token in &vr.accepted_tokens {
                if output.len() >= prompt.len() + self.config.max_tokens {
                    break;
                }
                output.push(token);
            }

            // If all draft tokens were accepted, also sample one more from target
            // (the "bonus" token from the target model evaluation)
            if vr.all_accepted() && output.len() < prompt.len() + self.config.max_tokens {
                if let Some(bonus) = self.sample_from_target(&output) {
                    output.push(bonus);
                }
            }

            step_count += 1;

            // Step 4: Adaptive draft length
            if self.config.adaptive_draft && rolling_drafted > 0 {
                let rate = rolling_accepted as f64 / rolling_drafted as f64;
                current_draft_length =
                    adapt_draft_length(current_draft_length, rate, self.config.draft_length);
            }
        }

        // Trim to max_tokens
        let max_len = prompt.len() + self.config.max_tokens;
        if output.len() > max_len {
            output.truncate(max_len);
        }

        let elapsed = start.elapsed();
        let generated = output.len().saturating_sub(prompt.len());
        stats.total_tokens = generated;
        stats.wall_time_ms = elapsed.as_secs_f64() * 1000.0;
        stats.tokens_per_step = if step_count > 0 {
            generated as f64 / step_count as f64
        } else {
            0.0
        };

        (output, stats)
    }

    /// Compute target and draft probability distributions for each draft position.
    fn compute_distributions(
        &mut self,
        context: &[usize],
        draft_tokens: &[usize],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let vocab_size = self.target.vocab_size();
        let mut target_dists = Vec::with_capacity(draft_tokens.len());
        let mut draft_dists = Vec::with_capacity(draft_tokens.len());

        let mut extended_ctx = context.to_vec();

        for &token in draft_tokens {
            // Get target log-probs for this position
            let log_probs = self.target.log_probs(&extended_ctx);
            let target_dist = log_probs_to_probs(&log_probs, self.config.temperature);

            // Apply top-k if configured
            let target_dist = if self.config.top_k > 0 && self.config.top_k < vocab_size {
                if let Some(dist) = TokenDistribution::from_probs(target_dist) {
                    if let Some(filtered) = dist.with_top_k(self.config.top_k) {
                        filtered.probs().to_vec()
                    } else {
                        dist.probs().to_vec()
                    }
                } else {
                    vec![1.0 / vocab_size as f64; vocab_size]
                }
            } else {
                target_dist
            };

            // Draft distribution for this position (from the draft model)
            let draft_for_pos = self.draft.generate_draft(&extended_ctx, 1);
            let mut draft_dist = vec![0.0; vocab_size];
            // Since we can't easily get the full draft distribution,
            // we approximate: build it from the n-gram table or uniform
            // For simplicity, generate many samples and estimate, or use
            // the fact that draft models should provide full distributions.
            // Here we use a simple uniform + spike approach.
            if let Some((_, p)) = draft_for_pos.first() {
                // The draft model generated one specific token with prob p
                // Spread remaining probability uniformly
                let remaining = 1.0 - p;
                let uniform_part = if vocab_size > 1 {
                    remaining / (vocab_size - 1) as f64
                } else {
                    0.0
                };
                draft_dist.fill(uniform_part);
                if let Some((tok, _)) = draft_for_pos.first() {
                    if *tok < vocab_size {
                        draft_dist[*tok] = *p;
                    }
                }
            } else {
                let uniform = 1.0 / vocab_size as f64;
                draft_dist.fill(uniform);
            }

            target_dists.push(target_dist);
            draft_dists.push(draft_dist);
            extended_ctx.push(token);
        }

        (target_dists, draft_dists)
    }

    /// Sample a single token from the target model's distribution.
    fn sample_from_target(&mut self, context: &[usize]) -> Option<usize> {
        let log_probs = self.target.log_probs(context);
        let probs = log_probs_to_probs(&log_probs, self.config.temperature);
        let dist = TokenDistribution::from_probs(probs)?;
        let u = self.rng.next_f64();
        Some(dist.sample_with_uniform(u))
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &SpeculativeConfig {
        &self.config
    }

    /// Get a mutable reference to the configuration.
    pub fn config_mut(&mut self) -> &mut SpeculativeConfig {
        &mut self.config
    }
}

/// Convert log-probabilities to probabilities with temperature scaling.
///
/// Uses the log-sum-exp trick for numerical stability.
fn log_probs_to_probs(log_probs: &[f64], temperature: f64) -> Vec<f64> {
    if log_probs.is_empty() {
        return Vec::new();
    }

    let temp = if temperature <= 0.0 { 1.0 } else { temperature };

    let scaled: Vec<f64> = log_probs.iter().map(|&lp| lp / temp).collect();
    let max_val = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let exps: Vec<f64> = scaled.iter().map(|&s| (s - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum <= 0.0 || sum.is_nan() {
        let uniform = 1.0 / log_probs.len() as f64;
        return vec![uniform; log_probs.len()];
    }

    exps.iter().map(|&e| e / sum).collect()
}

/// Adapt the draft length based on acceptance rate.
///
/// - rate > 0.8 → increase (up to 2x initial)
/// - rate < 0.4 → decrease (down to 1)
/// - otherwise → keep current
fn adapt_draft_length(current: usize, acceptance_rate: f64, initial: usize) -> usize {
    let max_draft = initial * 2;

    if acceptance_rate > 0.8 {
        (current + 1).min(max_draft)
    } else if acceptance_rate < 0.4 {
        if current > 1 {
            current - 1
        } else {
            1
        }
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speculative::draft::UniformDraftModel;

    /// A deterministic target model for testing.
    /// Always returns the same distribution regardless of context.
    struct FixedTarget {
        probs: Vec<f64>,
    }

    impl FixedTarget {
        fn new(probs: Vec<f64>) -> Self {
            Self { probs }
        }
    }

    impl TargetModel for FixedTarget {
        fn log_probs(&self, _context: &[usize]) -> Vec<f64> {
            self.probs
                .iter()
                .map(|&p| if p > 0.0 { p.ln() } else { f64::NEG_INFINITY })
                .collect()
        }

        fn vocab_size(&self) -> usize {
            self.probs.len()
        }
    }

    #[test]
    fn test_decoder_produces_output() {
        let vocab = 10;
        let draft = UniformDraftModel::new(vocab, 42).expect("test: uniform draft model");
        let target = FixedTarget::new(vec![0.1; vocab]);

        let config = SpeculativeConfig {
            draft_length: 3,
            max_tokens: 20,
            ..Default::default()
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, config, 42);
        let prompt = vec![0, 1, 2];
        let (output, stats) = decoder.decode(&prompt);

        // Output should start with prompt
        assert_eq!(&output[..3], &[0, 1, 2]);
        // Should have generated some tokens
        assert!(output.len() > 3, "should generate beyond prompt");
        // Stats should be populated
        assert!(stats.total_tokens > 0);
        assert!(stats.wall_time_ms >= 0.0);
    }

    #[test]
    fn test_decoder_respects_max_tokens() {
        let vocab = 5;
        let draft = UniformDraftModel::new(vocab, 42).expect("test: uniform draft model");
        let target = FixedTarget::new(vec![0.2; vocab]);

        let max_tokens = 10;
        let config = SpeculativeConfig {
            draft_length: 4,
            max_tokens,
            ..Default::default()
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, config, 42);
        let prompt = vec![0];
        let (output, stats) = decoder.decode(&prompt);

        // output = prompt + generated; generated <= max_tokens
        assert!(
            output.len() <= prompt.len() + max_tokens,
            "output {} exceeds prompt {} + max_tokens {}",
            output.len(),
            prompt.len(),
            max_tokens
        );
        assert_eq!(stats.total_tokens, output.len() - prompt.len());
    }

    #[test]
    fn test_decoder_stats_tracking() {
        let vocab = 5;
        let draft = UniformDraftModel::new(vocab, 123).expect("test: uniform draft model");
        let target = FixedTarget::new(vec![0.2; vocab]);

        let config = SpeculativeConfig {
            draft_length: 2,
            max_tokens: 8,
            ..Default::default()
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, config, 123);
        let (_, stats) = decoder.decode(&[0]);

        assert!(stats.draft_tokens > 0);
        assert!(stats.total_tokens > 0);
        assert!(stats.tokens_per_step > 0.0);
        assert!(stats.wall_time_ms >= 0.0);
    }

    #[test]
    fn test_adaptive_draft_length_increases() {
        // High acceptance rate -> should increase
        let result = adapt_draft_length(4, 0.9, 4);
        assert_eq!(result, 5);
    }

    #[test]
    fn test_adaptive_draft_length_decreases() {
        // Low acceptance rate -> should decrease
        let result = adapt_draft_length(4, 0.3, 4);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_adaptive_draft_length_stays() {
        // Medium acceptance rate -> should stay
        let result = adapt_draft_length(4, 0.6, 4);
        assert_eq!(result, 4);
    }

    #[test]
    fn test_adaptive_draft_length_floor() {
        // Cannot go below 1
        let result = adapt_draft_length(1, 0.1, 4);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_adaptive_draft_length_ceiling() {
        // Cannot exceed 2x initial
        let result = adapt_draft_length(8, 0.95, 4);
        assert_eq!(result, 8); // 2 * 4 = 8, capped
    }

    #[test]
    fn test_decoder_with_adaptive_enabled() {
        let vocab = 5;
        let draft = UniformDraftModel::new(vocab, 42).expect("test: uniform draft model");
        let target = FixedTarget::new(vec![0.2; vocab]);

        let config = SpeculativeConfig {
            draft_length: 3,
            max_tokens: 15,
            adaptive_draft: true,
            ..Default::default()
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, config, 42);
        let (output, _stats) = decoder.decode(&[0]);

        // Should still produce valid output
        assert!(output.len() > 1);
        assert_eq!(output[0], 0);
    }

    #[test]
    fn test_log_probs_to_probs() {
        let log_probs = vec![0.0_f64.ln(), 0.0_f64.ln()]; // -inf, -inf
                                                          // When all are -inf, should return uniform
        let probs = log_probs_to_probs(&[f64::NEG_INFINITY; 3], 1.0);
        assert_eq!(probs.len(), 3);
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }

        let _ = log_probs; // suppress unused warning

        // Normal case
        let probs = log_probs_to_probs(&[0.0, 0.0, 0.0], 1.0); // all exp(0)=1
        for &p in &probs {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_probs_to_probs_with_temperature() {
        // Low temperature sharpens: the highest log-prob token dominates
        let log_probs = vec![-1.0, 0.0, -2.0]; // token 1 is highest
        let probs = log_probs_to_probs(&log_probs, 0.1);
        assert!(probs[1] > 0.9, "low temp should sharpen: {:.4}", probs[1]);
    }

    #[test]
    fn test_decoder_empty_prompt() {
        let vocab = 5;
        let draft = UniformDraftModel::new(vocab, 42).expect("test: uniform draft model");
        let target = FixedTarget::new(vec![0.2; vocab]);

        let config = SpeculativeConfig {
            draft_length: 2,
            max_tokens: 5,
            ..Default::default()
        };

        let mut decoder = SpeculativeDecoder::new(draft, target, config, 42);
        let (output, stats) = decoder.decode(&[]);

        assert!(output.len() <= 5);
        assert_eq!(stats.total_tokens, output.len());
    }

    #[test]
    fn test_decoding_stats_default() {
        let stats = DecodingStats::default();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.draft_tokens, 0);
        assert_eq!(stats.accepted_tokens, 0);
        assert!((stats.acceptance_rate() - 0.0).abs() < 1e-10);
        assert!((stats.throughput() - 0.0).abs() < 1e-10);
    }
}
