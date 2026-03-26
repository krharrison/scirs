//! Draft models for speculative decoding.
//!
//! A draft model is a fast, lightweight model that generates candidate tokens
//! which are later verified (or rejected) by a slower, more accurate target model.
//! This module provides the [`DraftModel`] trait and two concrete implementations:
//!
//! - [`NGramDraftModel`]: uses an n-gram frequency table built from a corpus.
//! - [`UniformDraftModel`]: baseline that samples uniformly at random.

use std::collections::HashMap;

/// A simple xorshift64 PRNG for internal sampling.
///
/// Avoids depending on external random crates while still producing
/// reasonable pseudo-random sequences for rejection sampling.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new PRNG with the given seed.
    ///
    /// A seed of 0 is replaced by a default non-zero value.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed
            },
        }
    }

    /// Generate the next u64 value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a uniform f64 in \[0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Trait for draft models that propose candidate tokens.
///
/// A draft model takes a context (sequence of token ids) and produces a
/// sequence of `(token_id, probability)` pairs — the tokens it would generate
/// along with the probability it assigned to each.
pub trait DraftModel: Send + Sync {
    /// Generate `length` draft tokens given the current context.
    ///
    /// Returns a vector of `(token_id, probability)` pairs in generation order.
    fn generate_draft(&self, context: &[usize], length: usize) -> Vec<(usize, f64)>;

    /// Vocabulary size assumed by this draft model.
    fn vocab_size(&self) -> usize;
}

/// An n-gram based draft model.
///
/// Builds a frequency table from a training corpus and uses it to predict the
/// next token given the last `n-1` tokens of context. Falls back to a uniform
/// distribution when the context n-gram has not been seen.
#[derive(Debug, Clone)]
pub struct NGramDraftModel {
    /// Maps an n-gram prefix (length n-1) to a distribution over next tokens.
    /// The inner map is token_id → count.
    table: HashMap<Vec<usize>, HashMap<usize, usize>>,
    /// The "n" in n-gram (e.g. 3 for trigrams).
    n: usize,
    /// Vocabulary size.
    vocab: usize,
    /// PRNG seed for fallback sampling.
    seed: u64,
}

impl NGramDraftModel {
    /// Build an n-gram draft model from a corpus.
    ///
    /// # Arguments
    ///
    /// * `corpus` — token id sequence to learn from.
    /// * `n` — n-gram order (minimum 2).
    /// * `vocab_size` — size of the token vocabulary.
    /// * `seed` — PRNG seed for sampling.
    ///
    /// Returns `None` if `n < 2` or `vocab_size == 0` or corpus is too short.
    pub fn new(corpus: &[usize], n: usize, vocab_size: usize, seed: u64) -> Option<Self> {
        if n < 2 || vocab_size == 0 {
            return None;
        }
        let table = build_ngram_table(corpus, n);
        Some(Self {
            table,
            n,
            vocab: vocab_size,
            seed,
        })
    }

    /// Look up the distribution for a given context.
    ///
    /// Returns a probability vector over the vocabulary. If the context n-gram
    /// is not in the table, returns a uniform distribution.
    fn distribution_for_context(&self, context: &[usize]) -> Vec<f64> {
        let prefix_len = self.n - 1;
        let prefix = if context.len() >= prefix_len {
            &context[context.len() - prefix_len..]
        } else {
            context
        };

        if let Some(counts) = self.table.get(prefix) {
            let total: usize = counts.values().sum();
            if total == 0 {
                return vec![1.0 / self.vocab as f64; self.vocab];
            }
            let mut probs = vec![0.0; self.vocab];
            for (&token, &count) in counts {
                if token < self.vocab {
                    probs[token] = count as f64 / total as f64;
                }
            }
            // Add small smoothing to avoid zero probabilities
            let smoothing = 1e-10;
            let sum_before: f64 = probs.iter().sum();
            for p in &mut probs {
                *p += smoothing;
            }
            let sum_after: f64 = probs.iter().sum();
            if sum_after > 0.0 && sum_before >= 0.0 {
                for p in &mut probs {
                    *p /= sum_after;
                }
            }
            probs
        } else {
            vec![1.0 / self.vocab as f64; self.vocab]
        }
    }

    /// Sample a token from a probability vector using the PRNG.
    fn sample_from_probs(probs: &[f64], rng: &mut Xorshift64) -> (usize, f64) {
        let u = rng.next_f64();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return (i, p);
            }
        }
        let last = probs.len().saturating_sub(1);
        (last, probs.get(last).copied().unwrap_or(0.0))
    }
}

impl DraftModel for NGramDraftModel {
    fn generate_draft(&self, context: &[usize], length: usize) -> Vec<(usize, f64)> {
        let mut rng = Xorshift64::new(self.seed);
        let mut tokens = Vec::with_capacity(length);
        let mut ctx: Vec<usize> = context.to_vec();

        for _ in 0..length {
            let probs = self.distribution_for_context(&ctx);
            let (token, prob) = Self::sample_from_probs(&probs, &mut rng);
            tokens.push((token, prob));
            ctx.push(token);
        }

        tokens
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// A baseline draft model that samples uniformly at random.
///
/// Useful as a lower-bound benchmark: the acceptance rate of uniform drafting
/// gives a baseline that any real draft model should exceed.
#[derive(Debug, Clone)]
pub struct UniformDraftModel {
    /// Vocabulary size.
    vocab: usize,
    /// PRNG seed.
    seed: u64,
}

impl UniformDraftModel {
    /// Create a uniform draft model.
    ///
    /// Returns `None` if `vocab_size == 0`.
    pub fn new(vocab_size: usize, seed: u64) -> Option<Self> {
        if vocab_size == 0 {
            return None;
        }
        Some(Self {
            vocab: vocab_size,
            seed,
        })
    }
}

impl DraftModel for UniformDraftModel {
    fn generate_draft(&self, _context: &[usize], length: usize) -> Vec<(usize, f64)> {
        let mut rng = Xorshift64::new(self.seed);
        let p = 1.0 / self.vocab as f64;
        (0..length)
            .map(|_| {
                let token = (rng.next_u64() as usize) % self.vocab;
                (token, p)
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// Build an n-gram frequency table from a corpus of token ids.
///
/// For each (n-1)-gram prefix observed in the corpus, records the count of
/// each token that follows it.
///
/// # Arguments
///
/// * `corpus` — the sequence of token ids.
/// * `n` — n-gram order (e.g. 3 for trigrams). Must be >= 2.
///
/// # Returns
///
/// A map from prefix (Vec of length n-1) to a map of next-token → count.
pub fn build_ngram_table(corpus: &[usize], n: usize) -> HashMap<Vec<usize>, HashMap<usize, usize>> {
    let mut table: HashMap<Vec<usize>, HashMap<usize, usize>> = HashMap::new();

    if n < 2 || corpus.len() < n {
        return table;
    }

    for window in corpus.windows(n) {
        let prefix = window[..n - 1].to_vec();
        let next_token = window[n - 1];
        let entry = table.entry(prefix).or_default();
        *entry.entry(next_token).or_insert(0) += 1;
    }

    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift64_produces_values_in_range() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "value out of range: {v}");
        }
    }

    #[test]
    fn test_xorshift64_zero_seed_uses_default() {
        let mut rng = Xorshift64::new(0);
        let v = rng.next_u64();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_build_ngram_table_basic() {
        // corpus: [0, 1, 2, 0, 1, 3]
        // bigrams: (0,1), (1,2), (2,0), (0,1), (1,3)
        let corpus = vec![0, 1, 2, 0, 1, 3];
        let table = build_ngram_table(&corpus, 2);

        // prefix [0] -> {1: 2}
        let counts_0 = table.get(&vec![0]);
        assert!(counts_0.is_some());
        let counts_0 = counts_0.expect("test: prefix [0] should exist");
        assert_eq!(counts_0.get(&1).copied().unwrap_or(0), 2);

        // prefix [1] -> {2: 1, 3: 1}
        let counts_1 = table.get(&vec![1]);
        assert!(counts_1.is_some());
        let counts_1 = counts_1.expect("test: prefix [1] should exist");
        assert_eq!(counts_1.get(&2).copied().unwrap_or(0), 1);
        assert_eq!(counts_1.get(&3).copied().unwrap_or(0), 1);
    }

    #[test]
    fn test_build_ngram_table_too_short() {
        let corpus = vec![0];
        let table = build_ngram_table(&corpus, 2);
        assert!(table.is_empty());
    }

    #[test]
    fn test_ngram_draft_generates_correct_length() {
        let corpus = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1];
        let model =
            NGramDraftModel::new(&corpus, 2, 4, 123).expect("test: should build ngram model");
        let draft = model.generate_draft(&[0], 5);
        assert_eq!(draft.len(), 5);
    }

    #[test]
    fn test_ngram_draft_probabilities_positive() {
        let corpus = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let model =
            NGramDraftModel::new(&corpus, 2, 4, 42).expect("test: should build ngram model");
        let draft = model.generate_draft(&[0], 3);
        for (_, prob) in &draft {
            assert!(*prob > 0.0, "draft probability should be positive");
        }
    }

    #[test]
    fn test_ngram_draft_from_known_distribution() {
        // Build a corpus where after token 0, token 1 always follows
        let corpus = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let model =
            NGramDraftModel::new(&corpus, 2, 4, 99).expect("test: should build ngram model");
        let draft = model.generate_draft(&[0], 1);
        // Token 1 should have very high probability (close to 1.0 with smoothing)
        let (token, prob) = draft[0];
        assert_eq!(token, 1, "after 0 should always predict 1");
        assert!(prob > 0.9, "probability should be near 1.0, got {prob}");
    }

    #[test]
    fn test_uniform_draft_correct_length() {
        let model = UniformDraftModel::new(100, 42).expect("test: should build uniform model");
        let draft = model.generate_draft(&[0, 1, 2], 7);
        assert_eq!(draft.len(), 7);
    }

    #[test]
    fn test_uniform_draft_probability() {
        let vocab = 50;
        let model = UniformDraftModel::new(vocab, 42).expect("test: should build uniform model");
        let draft = model.generate_draft(&[], 10);
        let expected_p = 1.0 / vocab as f64;
        for (_, prob) in &draft {
            assert!(
                (*prob - expected_p).abs() < 1e-12,
                "uniform prob should be {expected_p}, got {prob}"
            );
        }
    }

    #[test]
    fn test_uniform_draft_tokens_in_range() {
        let vocab = 20;
        let model = UniformDraftModel::new(vocab, 12345).expect("test: should build uniform model");
        let draft = model.generate_draft(&[], 100);
        for (token, _) in &draft {
            assert!(*token < vocab, "token {token} out of vocab range {vocab}");
        }
    }

    #[test]
    fn test_ngram_invalid_params() {
        assert!(NGramDraftModel::new(&[0, 1], 1, 10, 0).is_none()); // n < 2
        assert!(NGramDraftModel::new(&[0, 1], 2, 0, 0).is_none()); // vocab = 0
    }

    #[test]
    fn test_uniform_invalid_params() {
        assert!(UniformDraftModel::new(0, 42).is_none());
    }
}
