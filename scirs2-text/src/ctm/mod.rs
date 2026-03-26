//! # Correlated Topic Model (CTM)
//!
//! Implements the Correlated Topic Model of Blei & Lafferty (2006), which uses
//! a logistic-normal prior over topic proportions to capture inter-topic
//! correlations that a plain Dirichlet prior cannot represent.
//!
//! ## Overview
//!
//! Unlike LDA, CTM models the document-topic distribution as:
//!
//! ```text
//!   η_d  ~ N(µ, Σ)
//!   θ_d  =  softmax(η_d)
//!   w_dn ~ Multinomial(θ_d, β)
//! ```
//!
//! Inference is performed via mean-field variational EM.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::ctm::{CorrelatedTopicModel, CtmConfig};
//!
//! let config = CtmConfig {
//!     n_topics: 3,
//!     max_iter: 20,
//!     tol: 1e-4,
//!     vocab_size: 10,
//! };
//! let model = CorrelatedTopicModel::new(config);
//!
//! // Documents as word-count vectors (length = vocab_size)
//! let docs: Vec<Vec<f64>> = (0..5)
//!     .map(|i| (0..10).map(|w| ((i * 3 + w) % 4) as f64).collect())
//!     .collect();
//!
//! let result = model.fit(&docs, 10).expect("CTM fit failed");
//! assert_eq!(result.topic_word_matrix.len(), 3);
//! ```

pub mod inference;
pub mod model;

use crate::error::Result;

// ────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ────────────────────────────────────────────────────────────────────────────

pub use model::{log_likelihood, softmax, top_words, topic_correlation_matrix};

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the Correlated Topic Model.
#[derive(Debug, Clone)]
pub struct CtmConfig {
    /// Number of latent topics.
    pub n_topics: usize,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the ELBO.
    pub tol: f64,
    /// Vocabulary size (may be 0; inferred from data if so).
    pub vocab_size: usize,
}

impl Default for CtmConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            max_iter: 100,
            tol: 1e-4,
            vocab_size: 0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Result type
// ────────────────────────────────────────────────────────────────────────────

/// Output of a fitted Correlated Topic Model.
#[derive(Debug, Clone)]
pub struct CtmResult {
    /// Topic-word probability matrix `K × V`. Each row sums to 1.
    pub topic_word_matrix: Vec<Vec<f64>>,
    /// Document-topic probability matrix `D × K`. Each row sums to 1.
    pub doc_topic_matrix: Vec<Vec<f64>>,
    /// Fitted prior mean µ (length K).
    pub mu: Vec<f64>,
    /// Fitted prior covariance Σ (K × K).
    pub sigma: Vec<Vec<f64>>,
    /// Approximate log-likelihood of the corpus under the fitted model.
    pub log_likelihood: f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Model struct
// ────────────────────────────────────────────────────────────────────────────

/// Correlated Topic Model estimator.
///
/// Fit with [`CorrelatedTopicModel::fit`]; the fitted result is returned as
/// a [`CtmResult`] value (the model itself is stateless after construction).
pub struct CorrelatedTopicModel {
    /// Model configuration.
    pub config: CtmConfig,
    /// Fitted result (populated after `fit`).
    fitted: Option<CtmResult>,
}

impl CorrelatedTopicModel {
    /// Construct a new (unfitted) CTM with the given configuration.
    pub fn new(config: CtmConfig) -> Self {
        Self {
            config,
            fitted: None,
        }
    }

    /// Return a reference to the fitted result, if available.
    pub fn fitted_result(&self) -> Option<&CtmResult> {
        self.fitted.as_ref()
    }

    /// Fit the model and store the result internally, also returning it.
    pub fn fit_and_store(
        &mut self,
        doc_counts_list: &[Vec<f64>],
        vocab_size: usize,
    ) -> Result<&CtmResult> {
        let result = self.fit(doc_counts_list, vocab_size)?;
        self.fitted = Some(result);
        Ok(self.fitted.as_ref().expect("just set"))
    }

    /// Return the top-`n` words for each topic given a vocabulary.
    ///
    /// Requires the model to have been fitted (via `fit_and_store`).
    pub fn top_words_from_fitted(&self, vocab: &[String], n: usize) -> Option<Vec<Vec<String>>> {
        self.fitted
            .as_ref()
            .map(|r| top_words(&r.topic_word_matrix, vocab, n))
    }

    /// Compute the inter-topic correlation matrix from the fitted Σ.
    pub fn correlation_matrix_from_fitted(&self) -> Option<Vec<Vec<f64>>> {
        self.fitted
            .as_ref()
            .map(|r| topic_correlation_matrix(&r.sigma))
    }
}

impl Default for CorrelatedTopicModel {
    fn default() -> Self {
        Self::new(CtmConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctm_default_config() {
        let cfg = CtmConfig::default();
        assert_eq!(cfg.n_topics, 10);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tol - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn ctm_model_default() {
        let m = CorrelatedTopicModel::default();
        assert_eq!(m.config.n_topics, 10);
        assert!(m.fitted_result().is_none());
    }

    #[test]
    fn ctm_fit_and_store() {
        let mut model = CorrelatedTopicModel::new(CtmConfig {
            n_topics: 2,
            max_iter: 5,
            tol: 1e-3,
            vocab_size: 4,
        });
        let docs: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|w| ((i + w) % 3) as f64).collect())
            .collect();
        model.fit_and_store(&docs, 4).expect("fit_and_store failed");
        assert!(model.fitted_result().is_some());
    }
}
