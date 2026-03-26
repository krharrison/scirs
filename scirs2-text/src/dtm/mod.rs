//! # Dynamic Topic Model (DTM)
//!
//! Implements the Dynamic Topic Model of Blei & Lafferty (2006), which extends
//! LDA by modelling topic evolution over discrete time slices via a Gaussian
//! state-space model.
//!
//! ## Model
//!
//! ```text
//!   β_{t,k} | β_{t-1,k} ~ N(β_{t-1,k}, σ² I)   (topic word evolution)
//!   θ_d      ~ Dir(α)                              (document-topic)
//!   z_{dn}  ~ Categorical(θ_d)                    (topic assignment)
//!   w_{dn}  ~ Categorical(β_{t,z_{dn}})           (word generation)
//! ```
//!
//! Inference is performed via variational EM with a Kalman smoother on the
//! topic-word parameters.
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::dtm::{DynamicTopicModel, DtmConfig};
//!
//! let config = DtmConfig {
//!     n_topics: 2,
//!     n_time_slices: 3,
//!     max_iter: 5,
//!     sigma_sq: 0.1,
//!     alpha: 0.1,
//! };
//! let model = DynamicTopicModel::new(config);
//!
//! // 3 time slices, each with 4 documents of 6 words each
//! let docs_by_time: Vec<Vec<Vec<f64>>> = (0..3)
//!     .map(|t| {
//!         (0..4)
//!             .map(|d| (0..6).map(|w| ((t + d + w) % 3) as f64).collect())
//!             .collect()
//!     })
//!     .collect();
//!
//! let result = model.fit(&docs_by_time, 6).expect("DTM fit failed");
//! assert_eq!(result.topic_word_trajectories.len(), 2); // K topics
//! ```

pub mod inference;
pub mod model;

use crate::error::Result;

// ────────────────────────────────────────────────────────────────────────────
// Public re-exports
// ────────────────────────────────────────────────────────────────────────────

pub use inference::{kalman_backward, kalman_forward};
pub use model::{top_words_at_time, topic_evolution};

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the Dynamic Topic Model.
#[derive(Debug, Clone)]
pub struct DtmConfig {
    /// Number of latent topics K.
    pub n_topics: usize,
    /// Number of time slices T (may be 0; inferred from data if so).
    pub n_time_slices: usize,
    /// Maximum number of variational EM iterations.
    pub max_iter: usize,
    /// State-transition variance σ² for the Gaussian random walk.
    pub sigma_sq: f64,
    /// Dirichlet concentration parameter α for document-topic prior.
    pub alpha: f64,
}

impl Default for DtmConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            n_time_slices: 0,
            max_iter: 50,
            sigma_sq: 0.5,
            alpha: 0.01,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Result type
// ────────────────────────────────────────────────────────────────────────────

/// Output of a fitted Dynamic Topic Model.
#[derive(Debug, Clone)]
pub struct DtmResult {
    /// Topic-word trajectories `K × T × V`.
    ///
    /// `trajectories[k][t][w]` is the probability of word `w` under topic `k`
    /// at time slice `t`.  Each slice `trajectories[k][t]` sums to 1.
    pub topic_word_trajectories: Vec<Vec<Vec<f64>>>,
    /// Flattened document-topic distribution (all documents across all time
    /// slices concatenated).  Each row sums to 1.
    pub doc_topic_matrix: Vec<Vec<f64>>,
}

// ────────────────────────────────────────────────────────────────────────────
// Model struct
// ────────────────────────────────────────────────────────────────────────────

/// Dynamic Topic Model estimator.
///
/// Fit via [`DynamicTopicModel::fit`]; the result is returned as a [`DtmResult`].
pub struct DynamicTopicModel {
    /// Model configuration.
    pub config: DtmConfig,
    /// Fitted result (populated after `fit_and_store`).
    fitted: Option<DtmResult>,
}

impl DynamicTopicModel {
    /// Construct a new (unfitted) DTM with the given configuration.
    pub fn new(config: DtmConfig) -> Self {
        Self {
            config,
            fitted: None,
        }
    }

    /// Return a reference to the fitted result, if available.
    pub fn fitted_result(&self) -> Option<&DtmResult> {
        self.fitted.as_ref()
    }

    /// Fit the model and store the result internally, also returning it.
    pub fn fit_and_store(
        &mut self,
        docs_by_time: &[Vec<Vec<f64>>],
        vocab_size: usize,
    ) -> Result<&DtmResult> {
        let result = self.fit(docs_by_time, vocab_size)?;
        self.fitted = Some(result);
        Ok(self.fitted.as_ref().expect("just set"))
    }

    /// Return the top-`n` words for each topic at time `t` using the fitted model.
    pub fn top_words_at(&self, t: usize, vocab: &[String], n: usize) -> Option<Vec<Vec<String>>> {
        self.fitted
            .as_ref()
            .map(|r| top_words_at_time(&r.topic_word_trajectories, t, vocab, n))
    }

    /// Return the evolution of word `word_id` in topic `topic_id` over time.
    pub fn word_evolution(&self, topic_id: usize, word_id: usize) -> Option<Vec<f64>> {
        self.fitted
            .as_ref()
            .map(|r| topic_evolution(&r.topic_word_trajectories, topic_id, word_id))
    }
}

impl Default for DynamicTopicModel {
    fn default() -> Self {
        Self::new(DtmConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtm_default_config() {
        let cfg = DtmConfig::default();
        assert_eq!(cfg.n_topics, 10);
        assert_eq!(cfg.max_iter, 50);
        assert!((cfg.sigma_sq - 0.5).abs() < 1e-12);
        assert!((cfg.alpha - 0.01).abs() < 1e-12);
    }

    #[test]
    fn dtm_default_model() {
        let m = DynamicTopicModel::default();
        assert_eq!(m.config.n_topics, 10);
        assert!(m.fitted_result().is_none());
    }

    #[test]
    fn dtm_fit_and_store() {
        let mut model = DynamicTopicModel::new(DtmConfig {
            n_topics: 2,
            n_time_slices: 2,
            max_iter: 3,
            sigma_sq: 0.1,
            alpha: 0.1,
        });
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|t| {
                (0..3)
                    .map(|d| (0..4).map(|w| ((t + d + w) % 3) as f64).collect())
                    .collect()
            })
            .collect();
        model.fit_and_store(&docs_by_time, 4).expect("fit failed");
        assert!(model.fitted_result().is_some());
    }

    #[test]
    fn dtm_top_words_at_after_fit() {
        let mut model = DynamicTopicModel::new(DtmConfig {
            n_topics: 2,
            n_time_slices: 2,
            max_iter: 3,
            sigma_sq: 0.1,
            alpha: 0.1,
        });
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..2)
            .map(|t| {
                (0..3)
                    .map(|d| (0..5).map(|w| ((t + d + w) % 3) as f64).collect())
                    .collect()
            })
            .collect();
        model.fit_and_store(&docs_by_time, 5).expect("fit failed");
        let vocab: Vec<String> = (0..5).map(|i| format!("w{i}")).collect();
        let tw = model.top_words_at(0, &vocab, 3).expect("no fitted result");
        assert_eq!(tw.len(), 2); // K topics
        assert_eq!(tw[0].len(), 3); // n words
    }

    #[test]
    fn dtm_word_evolution_length_equals_t() {
        let mut model = DynamicTopicModel::new(DtmConfig {
            n_topics: 2,
            n_time_slices: 4,
            max_iter: 3,
            sigma_sq: 0.1,
            alpha: 0.1,
        });
        let docs_by_time: Vec<Vec<Vec<f64>>> = (0..4)
            .map(|t| {
                (0..2)
                    .map(|d| (0..5).map(|w| ((t + d + w) % 3) as f64).collect())
                    .collect()
            })
            .collect();
        model.fit_and_store(&docs_by_time, 5).expect("fit failed");
        let ev = model.word_evolution(0, 2).expect("no fitted result");
        assert_eq!(ev.len(), 4);
    }
}
