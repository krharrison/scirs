//! # Online Latent Dirichlet Allocation
//!
//! Implementation of Online LDA using stochastic variational inference
//! (Hoffman, Blei, Bach 2010). This enables topic modeling on streaming
//! corpora by processing documents in mini-batches.
//!
//! ## Algorithm
//!
//! Online LDA uses stochastic variational inference:
//!
//! 1. **E-step**: For each mini-batch, run mean-field variational inference
//!    to estimate document-topic proportions (gamma) and topic assignments (phi).
//! 2. **M-step**: Update global topic-word parameters (lambda) using a
//!    weighted combination of the old parameters and the sufficient statistics
//!    from the mini-batch.
//!
//! The learning rate follows the schedule: rho_t = (tau + t)^{-kappa}
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_text::evaluation::online_lda::{OnlineLda, OnlineLdaConfig};
//!
//! // Create a simple vocabulary
//! let vocab_size = 10;
//! let config = OnlineLdaConfig {
//!     num_topics: 3,
//!     vocab_size,
//!     alpha: 0.1,
//!     eta: 0.01,
//!     kappa: 0.7,
//!     tau: 64.0,
//!     batch_size: 4,
//!     max_e_step_iter: 50,
//!     e_step_tol: 1e-3,
//!     random_seed: Some(42),
//!     ..Default::default()
//! };
//!
//! let mut lda = OnlineLda::new(config).expect("Operation failed");
//!
//! // Documents represented as word-count vectors (sparse: Vec<(word_id, count)>)
//! let docs = vec![
//!     vec![(0, 3.0), (1, 2.0), (2, 1.0)],
//!     vec![(3, 2.0), (4, 3.0), (5, 1.0)],
//!     vec![(0, 1.0), (1, 3.0), (2, 2.0)],
//!     vec![(3, 3.0), (4, 1.0), (5, 2.0)],
//! ];
//!
//! // Partial fit on a batch
//! lda.partial_fit(&docs).expect("Operation failed");
//!
//! // Get topic-word distributions
//! let topics = lda.get_topics();
//! assert_eq!(topics.len(), 3);
//! assert_eq!(topics[0].len(), vocab_size);
//! ```

use crate::error::{Result, TextError};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// Configuration for Online LDA.
#[derive(Debug, Clone)]
pub struct OnlineLdaConfig {
    /// Number of topics K.
    pub num_topics: usize,
    /// Vocabulary size V.
    pub vocab_size: usize,
    /// Document-topic Dirichlet prior (alpha). Symmetric prior for all topics.
    /// Default: 0.1.
    pub alpha: f64,
    /// Topic-word Dirichlet prior (eta). Symmetric prior for all words.
    /// Default: 0.01.
    pub eta: f64,
    /// Learning rate decay parameter (kappa).
    /// Controls how quickly the learning rate decreases.
    /// Must be in (0.5, 1.0]. Default: 0.7.
    pub kappa: f64,
    /// Learning rate offset (tau).
    /// Delays the effect of early iterations. Default: 64.0.
    pub tau: f64,
    /// Mini-batch size for partial_fit. Default: 64.
    pub batch_size: usize,
    /// Maximum iterations for E-step variational inference. Default: 100.
    pub max_e_step_iter: usize,
    /// Convergence tolerance for E-step. Default: 1e-3.
    pub e_step_tol: f64,
    /// Total number of documents in the corpus (for scaling).
    /// If unknown, set to a reasonable estimate. Default: 1000.
    pub total_docs: usize,
    /// Random seed for reproducibility. Default: None.
    pub random_seed: Option<u64>,
}

impl Default for OnlineLdaConfig {
    fn default() -> Self {
        Self {
            num_topics: 10,
            vocab_size: 1000,
            alpha: 0.1,
            eta: 0.01,
            kappa: 0.7,
            tau: 64.0,
            batch_size: 64,
            max_e_step_iter: 100,
            e_step_tol: 1e-3,
            total_docs: 1000,
            random_seed: None,
        }
    }
}

/// Sparse document representation: list of (word_id, count) pairs.
pub type SparseDoc = Vec<(usize, f64)>;

/// Online LDA model using stochastic variational inference.
///
/// The model maintains global topic-word parameters (lambda) which are
/// updated incrementally as new documents are processed.
pub struct OnlineLda {
    config: OnlineLdaConfig,
    /// Topic-word variational parameters: K x V matrix.
    /// lambda[k][w] is the unnormalized weight of word w in topic k.
    lambda: Vec<Vec<f64>>,
    /// Number of mini-batch updates performed so far.
    update_count: usize,
    /// RNG for initialization.
    rng: StdRng,
}

impl OnlineLda {
    /// Create a new Online LDA model.
    ///
    /// # Errors
    ///
    /// Returns `TextError::InvalidInput` if configuration parameters are invalid.
    pub fn new(config: OnlineLdaConfig) -> Result<Self> {
        if config.num_topics == 0 {
            return Err(TextError::InvalidInput(
                "num_topics must be at least 1".to_string(),
            ));
        }
        if config.vocab_size == 0 {
            return Err(TextError::InvalidInput(
                "vocab_size must be at least 1".to_string(),
            ));
        }
        if config.kappa <= 0.5 || config.kappa > 1.0 {
            return Err(TextError::InvalidInput(format!(
                "kappa must be in (0.5, 1.0], got {}",
                config.kappa
            )));
        }
        if config.alpha <= 0.0 {
            return Err(TextError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }
        if config.eta <= 0.0 {
            return Err(TextError::InvalidInput("eta must be positive".to_string()));
        }

        let mut rng = match config.random_seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                let mut temp_rng = scirs2_core::random::rng();
                StdRng::seed_from_u64(temp_rng.random::<u64>())
            }
        };

        // Initialize lambda with random values (Gamma(100, 0.01) like sklearn)
        let k = config.num_topics;
        let v = config.vocab_size;
        let mut lambda = vec![vec![0.0; v]; k];
        for topic in lambda.iter_mut() {
            for word_val in topic.iter_mut() {
                // Use exponential-like initialization: eta + random perturbation
                let random_val: f64 = rng.random::<f64>() * 0.01 + config.eta;
                *word_val = random_val * (v as f64);
            }
        }

        Ok(Self {
            config,
            lambda,
            update_count: 0,
            rng,
        })
    }

    /// Compute the digamma function (psi) using Stirling's approximation.
    ///
    /// For x >= 6, uses the asymptotic expansion.
    /// For x < 6, uses the recurrence psi(x) = psi(x+1) - 1/x.
    fn digamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }

        let mut result = 0.0;
        let mut val = x;

        // Use recurrence to shift x to a large value
        while val < 6.0 {
            result -= 1.0 / val;
            val += 1.0;
        }

        // Asymptotic expansion for large x
        result += val.ln() - 0.5 / val;
        let val_sq = val * val;
        result -= 1.0 / (12.0 * val_sq);
        result += 1.0 / (120.0 * val_sq * val_sq);
        result -= 1.0 / (252.0 * val_sq * val_sq * val_sq);

        result
    }

    /// Compute the log-sum-exp of a slice for numerical stability.
    fn log_sum_exp(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val.is_infinite() {
            return max_val;
        }
        let sum: f64 = values.iter().map(|v| (v - max_val).exp()).sum();
        max_val + sum.ln()
    }

    /// Run the E-step for a single document.
    ///
    /// Returns the variational document-topic parameters (gamma) and
    /// the sufficient statistics contribution (word-topic counts).
    fn e_step_doc(
        &self,
        doc: &SparseDoc,
        exp_e_log_beta: &[Vec<f64>],
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let k = self.config.num_topics;
        let v = self.config.vocab_size;

        // Initialize gamma uniformly
        let mut gamma = vec![self.config.alpha + (v as f64) / (k as f64); k];
        let mut phi: Vec<Vec<f64>> = Vec::new(); // phi[word_position][topic]

        let doc_words: Vec<usize> = doc.iter().map(|&(w, _)| w).collect();
        let doc_counts: Vec<f64> = doc.iter().map(|&(_, c)| c).collect();

        for _ in 0..self.config.max_e_step_iter {
            let gamma_old = gamma.clone();

            // Compute E[log theta] = digamma(gamma) - digamma(sum(gamma))
            let gamma_sum: f64 = gamma.iter().sum();
            let digamma_sum = Self::digamma(gamma_sum);
            let e_log_theta: Vec<f64> = gamma
                .iter()
                .map(|&g| Self::digamma(g) - digamma_sum)
                .collect();

            // Update phi for each word in the document
            phi.clear();
            let mut new_gamma = vec![self.config.alpha; k];

            for (idx, &word_id) in doc_words.iter().enumerate() {
                if word_id >= v {
                    continue;
                }

                // phi[word][topic] proportional to exp(E[log theta_k] + E[log beta_{k,w}])
                let mut log_phi_word = vec![0.0f64; k];
                for topic in 0..k {
                    log_phi_word[topic] =
                        e_log_theta[topic] + exp_e_log_beta[topic][word_id].ln().max(-700.0);
                }

                // Normalize phi
                let log_norm = Self::log_sum_exp(&log_phi_word);
                let mut phi_word = vec![0.0f64; k];
                for topic in 0..k {
                    phi_word[topic] = (log_phi_word[topic] - log_norm).exp();
                }

                // Accumulate into new_gamma
                let count = doc_counts[idx];
                for topic in 0..k {
                    new_gamma[topic] += count * phi_word[topic];
                }

                phi.push(phi_word);
            }

            gamma = new_gamma;

            // Check convergence
            let change: f64 = gamma
                .iter()
                .zip(gamma_old.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / k as f64;
            if change < self.config.e_step_tol {
                break;
            }
        }

        // Compute sufficient statistics for this document
        let mut sstats = vec![vec![0.0f64; v]; k];
        for (idx, &word_id) in doc_words.iter().enumerate() {
            if word_id >= v || idx >= phi.len() {
                continue;
            }
            let count = doc_counts[idx];
            for topic in 0..k {
                sstats[topic][word_id] += count * phi[idx][topic];
            }
        }

        Ok((gamma, sstats))
    }

    /// Compute exp(E[log beta]) from lambda.
    ///
    /// E[log beta_{k,w}] = digamma(lambda_{k,w}) - digamma(sum_w lambda_{k,w})
    /// We return exp of this for use in phi updates.
    fn compute_exp_e_log_beta(&self) -> Vec<Vec<f64>> {
        let k = self.config.num_topics;
        let v = self.config.vocab_size;
        let mut result = vec![vec![0.0f64; v]; k];

        for topic in 0..k {
            let lambda_sum: f64 = self.lambda[topic].iter().sum();
            let digamma_sum = Self::digamma(lambda_sum);

            for word in 0..v {
                let e_log = Self::digamma(self.lambda[topic][word]) - digamma_sum;
                // Clamp to avoid overflow/underflow
                result[topic][word] = e_log.exp().max(1e-100);
            }
        }

        result
    }

    /// Process a mini-batch of documents (partial fit).
    ///
    /// Performs one iteration of online variational inference:
    /// 1. E-step on the mini-batch
    /// 2. M-step to update global parameters
    ///
    /// # Arguments
    ///
    /// * `documents` - Slice of sparse documents. Each document is a
    ///   `Vec<(word_id, count)>` where word_id is in `[0, vocab_size)`.
    ///
    /// # Errors
    ///
    /// Returns `TextError::InvalidInput` if documents contain invalid word IDs.
    pub fn partial_fit(&mut self, documents: &[SparseDoc]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        // Validate word IDs
        for (doc_idx, doc) in documents.iter().enumerate() {
            for &(word_id, count) in doc {
                if word_id >= self.config.vocab_size {
                    return Err(TextError::InvalidInput(format!(
                        "Word ID {} in document {} exceeds vocab_size {}",
                        word_id, doc_idx, self.config.vocab_size
                    )));
                }
                if count < 0.0 {
                    return Err(TextError::InvalidInput(format!(
                        "Negative count {} for word {} in document {}",
                        count, word_id, doc_idx
                    )));
                }
            }
        }

        let k = self.config.num_topics;
        let v = self.config.vocab_size;
        let exp_e_log_beta = self.compute_exp_e_log_beta();

        // E-step: accumulate sufficient statistics across the batch
        let mut total_sstats = vec![vec![0.0f64; v]; k];

        for doc in documents {
            let (_gamma, sstats) = self.e_step_doc(doc, &exp_e_log_beta)?;
            for topic in 0..k {
                for word in 0..v {
                    total_sstats[topic][word] += sstats[topic][word];
                }
            }
        }

        // M-step: update lambda using learning rate
        self.update_count += 1;
        let rho = (self.config.tau + self.update_count as f64).powf(-self.config.kappa);
        let scale = self.config.total_docs as f64 / documents.len() as f64;

        for topic in 0..k {
            for word in 0..v {
                let new_lambda = self.config.eta + scale * total_sstats[topic][word];
                self.lambda[topic][word] =
                    (1.0 - rho) * self.lambda[topic][word] + rho * new_lambda;
                // Ensure lambda stays positive
                if self.lambda[topic][word] < 1e-100 {
                    self.lambda[topic][word] = 1e-100;
                }
            }
        }

        Ok(())
    }

    /// Transform documents to topic proportions.
    ///
    /// Runs the E-step to infer document-topic distributions without
    /// updating the global parameters.
    ///
    /// # Arguments
    ///
    /// * `documents` - Slice of sparse documents.
    ///
    /// # Returns
    ///
    /// A Vec of topic proportion vectors, one per document.
    /// Each vector sums to approximately 1.0.
    ///
    /// # Errors
    ///
    /// Returns error if documents contain invalid word IDs.
    pub fn transform(&self, documents: &[SparseDoc]) -> Result<Vec<Vec<f64>>> {
        let exp_e_log_beta = self.compute_exp_e_log_beta();
        let mut result = Vec::with_capacity(documents.len());

        for doc in documents {
            let (gamma, _) = self.e_step_doc(doc, &exp_e_log_beta)?;

            // Normalize gamma to get topic proportions
            let gamma_sum: f64 = gamma.iter().sum();
            let proportions: Vec<f64> = if gamma_sum > 0.0 {
                gamma.iter().map(|&g| g / gamma_sum).collect()
            } else {
                vec![1.0 / self.config.num_topics as f64; self.config.num_topics]
            };

            result.push(proportions);
        }

        Ok(result)
    }

    /// Get the normalized topic-word distributions.
    ///
    /// Returns a K x V matrix where entry `[k][w]` is the probability
    /// of word w in topic k. Each row sums to 1.0.
    pub fn get_topics(&self) -> Vec<Vec<f64>> {
        let k = self.config.num_topics;
        let v = self.config.vocab_size;
        let mut topics = vec![vec![0.0f64; v]; k];

        for topic in 0..k {
            let sum: f64 = self.lambda[topic].iter().sum();
            if sum > 0.0 {
                for word in 0..v {
                    topics[topic][word] = self.lambda[topic][word] / sum;
                }
            }
        }

        topics
    }

    /// Get the top N words for each topic.
    ///
    /// Returns a Vec of K vectors, where each inner vector contains
    /// (word_id, probability) pairs sorted by probability descending.
    pub fn get_top_words(&self, n: usize) -> Vec<Vec<(usize, f64)>> {
        let topics = self.get_topics();
        let mut result = Vec::with_capacity(self.config.num_topics);

        for topic_dist in &topics {
            let mut word_probs: Vec<(usize, f64)> = topic_dist
                .iter()
                .enumerate()
                .map(|(w, &p)| (w, p))
                .collect();
            word_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            word_probs.truncate(n);
            result.push(word_probs);
        }

        result
    }

    /// Compute per-word log-likelihood (approximation) for perplexity.
    ///
    /// Perplexity = exp(-log_likelihood / total_words)
    ///
    /// A lower perplexity indicates a better model.
    ///
    /// # Arguments
    ///
    /// * `documents` - Documents to evaluate.
    ///
    /// # Returns
    ///
    /// Tuple of (perplexity, per_word_log_likelihood).
    ///
    /// # Errors
    ///
    /// Returns error if documents are invalid.
    pub fn perplexity(&self, documents: &[SparseDoc]) -> Result<(f64, f64)> {
        if documents.is_empty() {
            return Err(TextError::InvalidInput(
                "Cannot compute perplexity on empty document set".to_string(),
            ));
        }

        let topics = self.get_topics();
        let topic_proportions = self.transform(documents)?;

        let mut total_log_likelihood = 0.0f64;
        let mut total_words = 0.0f64;

        for (doc_idx, doc) in documents.iter().enumerate() {
            let theta = &topic_proportions[doc_idx];

            for &(word_id, count) in doc {
                if word_id >= self.config.vocab_size {
                    continue;
                }

                // p(w|d) = sum_k theta[k] * beta[k][w]
                let mut word_prob = 0.0f64;
                for (topic, &theta_k) in theta.iter().enumerate() {
                    word_prob += theta_k * topics[topic][word_id];
                }

                if word_prob > 0.0 {
                    total_log_likelihood += count * word_prob.ln();
                } else {
                    // Add a small value to avoid log(0)
                    total_log_likelihood += count * 1e-100_f64.ln();
                }
                total_words += count;
            }
        }

        if total_words <= 0.0 {
            return Err(TextError::InvalidInput(
                "Documents contain no words".to_string(),
            ));
        }

        let per_word_ll = total_log_likelihood / total_words;
        let perplexity = (-per_word_ll).exp();

        Ok((perplexity, per_word_ll))
    }

    /// Get the number of updates performed so far.
    pub fn update_count(&self) -> usize {
        self.update_count
    }

    /// Get the current learning rate.
    pub fn current_learning_rate(&self) -> f64 {
        if self.update_count == 0 {
            (self.config.tau + 1.0).powf(-self.config.kappa)
        } else {
            (self.config.tau + self.update_count as f64).powf(-self.config.kappa)
        }
    }

    /// Reset the model, re-initializing lambda with random values.
    ///
    /// # Errors
    ///
    /// Returns error if re-initialization fails.
    pub fn reset(&mut self) -> Result<()> {
        let k = self.config.num_topics;
        let v = self.config.vocab_size;

        for topic in self.lambda.iter_mut() {
            for word_val in topic.iter_mut() {
                let random_val: f64 = self.rng.random::<f64>() * 0.01 + self.config.eta;
                *word_val = random_val * (v as f64);
            }
        }

        self.update_count = 0;
        let _ = k; // used above
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a simple two-topic synthetic corpus.
    /// Topic A: words 0-4 (tech words)
    /// Topic B: words 5-9 (nature words)
    fn create_two_topic_corpus() -> Vec<SparseDoc> {
        let mut docs = Vec::new();

        // Tech-like documents (words 0-4)
        for _ in 0..20 {
            docs.push(vec![(0, 3.0), (1, 2.0), (2, 1.0), (3, 2.0), (4, 1.0)]);
        }
        // Nature-like documents (words 5-9)
        for _ in 0..20 {
            docs.push(vec![(5, 3.0), (6, 2.0), (7, 1.0), (8, 2.0), (9, 1.0)]);
        }

        docs
    }

    #[test]
    fn test_topics_separate_on_synthetic_data() {
        let docs = create_two_topic_corpus();
        let config = OnlineLdaConfig {
            num_topics: 2,
            vocab_size: 10,
            alpha: 0.1,
            eta: 0.01,
            kappa: 0.7,
            tau: 10.0,
            batch_size: 10,
            max_e_step_iter: 50,
            e_step_tol: 1e-4,
            total_docs: 40,
            random_seed: Some(42),
        };

        let mut lda = OnlineLda::new(config).expect("should create");

        // Train for multiple epochs
        for _ in 0..20 {
            for batch_start in (0..docs.len()).step_by(10) {
                let batch_end = (batch_start + 10).min(docs.len());
                lda.partial_fit(&docs[batch_start..batch_end])
                    .expect("should fit");
            }
        }

        // Check that topics separate: one topic should emphasize words 0-4,
        // the other should emphasize words 5-9
        let topics = lda.get_topics();
        assert_eq!(topics.len(), 2);

        let topic0_first_half: f64 = topics[0][0..5].iter().sum();
        let topic0_second_half: f64 = topics[0][5..10].iter().sum();
        let topic1_first_half: f64 = topics[1][0..5].iter().sum();
        let topic1_second_half: f64 = topics[1][5..10].iter().sum();

        // One topic should have higher mass on first half, other on second half
        let separated = (topic0_first_half > topic0_second_half
            && topic1_second_half > topic1_first_half)
            || (topic0_second_half > topic0_first_half && topic1_first_half > topic1_second_half);

        assert!(
            separated,
            "Topics should separate: T0=[{:.3},{:.3}] T1=[{:.3},{:.3}]",
            topic0_first_half, topic0_second_half, topic1_first_half, topic1_second_half
        );
    }

    #[test]
    fn test_perplexity_decreases_over_training() {
        let docs = create_two_topic_corpus();
        let config = OnlineLdaConfig {
            num_topics: 2,
            vocab_size: 10,
            alpha: 0.1,
            eta: 0.01,
            kappa: 0.7,
            tau: 10.0,
            batch_size: 10,
            max_e_step_iter: 50,
            e_step_tol: 1e-4,
            total_docs: 40,
            random_seed: Some(42),
        };

        let mut lda = OnlineLda::new(config).expect("should create");

        // Measure initial perplexity
        let (perp_initial, _) = lda.perplexity(&docs).expect("should compute");

        // Train
        for _ in 0..15 {
            for batch_start in (0..docs.len()).step_by(10) {
                let batch_end = (batch_start + 10).min(docs.len());
                lda.partial_fit(&docs[batch_start..batch_end])
                    .expect("should fit");
            }
        }

        // Measure final perplexity
        let (perp_final, _) = lda.perplexity(&docs).expect("should compute");

        assert!(
            perp_final < perp_initial,
            "Perplexity should decrease: initial={:.2}, final={:.2}",
            perp_initial,
            perp_final
        );
    }

    #[test]
    fn test_transform_returns_proportions() {
        let docs = create_two_topic_corpus();
        let config = OnlineLdaConfig {
            num_topics: 3,
            vocab_size: 10,
            alpha: 0.1,
            eta: 0.01,
            kappa: 0.7,
            tau: 10.0,
            total_docs: 40,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut lda = OnlineLda::new(config).expect("should create");

        // Partial fit
        lda.partial_fit(&docs).expect("should fit");

        // Transform
        let proportions = lda.transform(&docs).expect("should transform");
        assert_eq!(proportions.len(), docs.len());

        for props in &proportions {
            assert_eq!(props.len(), 3);
            let sum: f64 = props.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Proportions should sum to 1.0, got {}",
                sum
            );
            for &p in props {
                assert!(p >= 0.0, "Proportions should be non-negative");
            }
        }
    }

    #[test]
    fn test_get_top_words() {
        let config = OnlineLdaConfig {
            num_topics: 2,
            vocab_size: 10,
            random_seed: Some(42),
            ..Default::default()
        };

        let lda = OnlineLda::new(config).expect("should create");
        let top_words = lda.get_top_words(5);
        assert_eq!(top_words.len(), 2);
        for topic_words in &top_words {
            assert_eq!(topic_words.len(), 5);
            // Check that words are sorted by probability descending
            for i in 1..topic_words.len() {
                assert!(
                    topic_words[i - 1].1 >= topic_words[i].1,
                    "Top words should be sorted by probability"
                );
            }
        }
    }

    #[test]
    fn test_invalid_config() {
        let config = OnlineLdaConfig {
            num_topics: 0,
            ..Default::default()
        };
        assert!(OnlineLda::new(config).is_err());

        let config = OnlineLdaConfig {
            kappa: 0.3,
            ..Default::default()
        };
        assert!(OnlineLda::new(config).is_err());

        let config = OnlineLdaConfig {
            alpha: -1.0,
            ..Default::default()
        };
        assert!(OnlineLda::new(config).is_err());
    }

    #[test]
    fn test_invalid_word_id() {
        let config = OnlineLdaConfig {
            num_topics: 2,
            vocab_size: 5,
            random_seed: Some(42),
            ..Default::default()
        };
        let mut lda = OnlineLda::new(config).expect("should create");

        let docs = vec![vec![(10, 1.0)]]; // word_id 10 > vocab_size 5
        let result = lda.partial_fit(&docs);
        assert!(result.is_err());
    }

    #[test]
    fn test_learning_rate_schedule() {
        let config = OnlineLdaConfig {
            kappa: 0.7,
            tau: 64.0,
            random_seed: Some(42),
            ..Default::default()
        };
        let lda = OnlineLda::new(config).expect("should create");

        let lr = lda.current_learning_rate();
        // rho = (64 + 1)^(-0.7) = 65^(-0.7)
        let expected = 65.0_f64.powf(-0.7);
        assert!(
            (lr - expected).abs() < 1e-10,
            "Learning rate: got {}, expected {}",
            lr,
            expected
        );
    }

    #[test]
    fn test_reset() {
        let docs = create_two_topic_corpus();
        let config = OnlineLdaConfig {
            num_topics: 2,
            vocab_size: 10,
            random_seed: Some(42),
            ..Default::default()
        };
        let mut lda = OnlineLda::new(config).expect("should create");

        lda.partial_fit(&docs).expect("should fit");
        assert!(lda.update_count() > 0);

        lda.reset().expect("should reset");
        assert_eq!(lda.update_count(), 0);
    }
}
