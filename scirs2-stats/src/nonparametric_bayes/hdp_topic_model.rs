//! High-level wrapper around [`HdpResult`] providing topic-model helpers.
//!
//! [`HdpTopicModel`] bundles a fitted HDP result with an optional vocabulary
//! and exposes convenience methods such as top words per topic and perplexity.

use crate::error::StatsResult;
use crate::nonparametric_bayes::hdp::{hdp_fit, hdp_perplexity, HdpConfig, HdpResult};

/// A fitted HDP topic model with optional vocabulary.
///
/// Construct via [`HdpTopicModel::fit`].
pub struct HdpTopicModel {
    /// The underlying fitted HDP result.
    pub result: HdpResult,
    /// Optional word-index-to-string mapping; empty if not supplied.
    pub vocabulary: Vec<String>,
}

impl HdpTopicModel {
    /// Fit an HDP topic model to a corpus.
    ///
    /// # Parameters
    /// - `documents`: Corpus; each inner `Vec<usize>` is a sequence of word
    ///   indices in `[0, vocab_size)`.
    /// - `vocab_size`: Vocabulary size V.
    /// - `config`: Sampler configuration.
    ///
    /// # Errors
    /// Propagates any error from [`hdp_fit`].
    pub fn fit(
        documents: &[Vec<usize>],
        vocab_size: usize,
        config: HdpConfig,
    ) -> StatsResult<Self> {
        let result = hdp_fit(documents, vocab_size, &config)?;
        Ok(Self {
            result,
            vocabulary: Vec::new(),
        })
    }

    /// Fit an HDP topic model and attach a vocabulary for display.
    ///
    /// # Parameters
    /// - `documents`: Corpus.
    /// - `vocabulary`: String token list (length must equal `vocab_size`).
    /// - `config`: Sampler configuration.
    pub fn fit_with_vocab(
        documents: &[Vec<usize>],
        vocabulary: Vec<String>,
        config: HdpConfig,
    ) -> StatsResult<Self> {
        let vocab_size = vocabulary.len();
        let result = hdp_fit(documents, vocab_size, &config)?;
        Ok(Self { result, vocabulary })
    }

    /// Return the top-`n` word indices and their probabilities for `topic_id`.
    ///
    /// Returns `(word_index, probability)` pairs sorted by decreasing probability.
    /// Returns an empty `Vec` if `topic_id >= K`.
    pub fn top_words(&self, topic_id: usize, n: usize) -> Vec<(usize, f64)> {
        let k = self.result.topic_word_matrix.nrows();
        if topic_id >= k {
            return Vec::new();
        }
        let row = self.result.topic_word_matrix.row(topic_id);
        let vocab_size = row.len();
        let mut pairs: Vec<(usize, f64)> = (0..vocab_size).map(|v| (v, row[v])).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(n);
        pairs
    }

    /// Return the top-`n` word strings (requires `vocabulary` to have been set).
    ///
    /// Falls back to returning indices as strings if vocabulary is empty.
    pub fn top_word_strings(&self, topic_id: usize, n: usize) -> Vec<(String, f64)> {
        self.top_words(topic_id, n)
            .into_iter()
            .map(|(idx, prob)| {
                let word = if idx < self.vocabulary.len() {
                    self.vocabulary[idx].clone()
                } else {
                    idx.to_string()
                };
                (word, prob)
            })
            .collect()
    }

    /// Per-word perplexity on `documents`.
    ///
    /// Uses the training doc-topic distributions for documents that appear
    /// in the training set and a uniform prior for new documents.
    pub fn perplexity(&self, documents: &[Vec<usize>], config: &HdpConfig) -> f64 {
        hdp_perplexity(documents, &self.result, config)
    }

    /// Borrow the document-topic distribution for document `doc_id`.
    ///
    /// Returns an empty slice if `doc_id` is out of range.
    pub fn doc_topic_distribution(&self, doc_id: usize) -> &[f64] {
        let d = self.result.doc_topic_matrix.nrows();
        if doc_id >= d {
            return &[];
        }
        self.result
            .doc_topic_matrix
            .row(doc_id)
            .to_slice()
            .unwrap_or(&[])
    }

    /// Number of topics in the model (truncation level K).
    pub fn n_topics(&self) -> usize {
        self.result.topic_word_matrix.nrows()
    }

    /// Number of topics that received at least one word.
    pub fn n_topics_used(&self) -> usize {
        self.result.n_topics_used
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_corpus() -> Vec<Vec<usize>> {
        vec![
            vec![0, 1, 2, 0, 1],
            vec![3, 4, 5, 3, 4],
            vec![0, 2, 1, 0],
        ]
    }

    #[test]
    fn test_fit_returns_model() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg);
        assert!(model.is_ok());
    }

    #[test]
    fn test_top_words_length() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        let top = model.top_words(0, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn test_top_words_sorted_descending() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 20, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        let top = model.top_words(0, 6);
        for i in 1..top.len() {
            assert!(top[i - 1].1 >= top[i].1, "top_words not sorted at index {i}");
        }
    }

    #[test]
    fn test_top_words_out_of_range_topic() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 5, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        assert!(model.top_words(999, 5).is_empty());
    }

    #[test]
    fn test_doc_topic_distribution_correct_length() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        let dist = model.doc_topic_distribution(0);
        assert_eq!(dist.len(), 4);
    }

    #[test]
    fn test_doc_topic_distribution_sums_to_one() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        for d in 0..small_corpus().len() {
            let s: f64 = model.doc_topic_distribution(d).iter().sum();
            assert!((s - 1.0).abs() < 1e-9, "doc {d} sum = {s}");
        }
    }

    #[test]
    fn test_doc_topic_out_of_range_returns_empty() {
        let cfg = HdpConfig { n_topics: 4, n_iter: 5, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        assert!(model.doc_topic_distribution(999).is_empty());
    }

    #[test]
    fn test_perplexity_is_finite() {
        let corpus = small_corpus();
        let cfg = HdpConfig { n_topics: 4, n_iter: 20, ..Default::default() };
        let model = HdpTopicModel::fit(&corpus, 6, cfg.clone()).expect("fit");
        let ppl = model.perplexity(&corpus, &cfg);
        assert!(ppl.is_finite() && ppl > 0.0, "perplexity = {ppl}");
    }

    #[test]
    fn test_fit_with_vocab_attaches_strings() {
        let vocab: Vec<String> = (0..6).map(|i| format!("word{i}")).collect();
        let cfg = HdpConfig { n_topics: 4, n_iter: 5, ..Default::default() };
        let model = HdpTopicModel::fit_with_vocab(&small_corpus(), vocab, cfg).expect("fit");
        let top = model.top_word_strings(0, 2);
        assert_eq!(top.len(), 2);
        assert!(top[0].0.starts_with("word"), "expected word string, got {}", top[0].0);
    }

    #[test]
    fn test_n_topics_used() {
        let cfg = HdpConfig { n_topics: 10, n_iter: 30, ..Default::default() };
        let model = HdpTopicModel::fit(&small_corpus(), 6, cfg).expect("fit");
        assert!(model.n_topics_used() <= model.n_topics());
    }
}
