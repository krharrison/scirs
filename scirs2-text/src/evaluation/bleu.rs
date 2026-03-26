//! # BLEU Score (Bilingual Evaluation Understudy)
//!
//! Implementation of BLEU score for machine translation evaluation
//! (Papineni et al. 2002). Supports both corpus-level and sentence-level
//! BLEU with multiple smoothing methods.
//!
//! ## Overview
//!
//! BLEU measures how many n-grams in the hypothesis (candidate translation)
//! appear in the reference(s). It combines modified n-gram precision for
//! n=1..4 with a brevity penalty to discourage overly short translations.
//!
//! Formula: BLEU = BP * exp(sum(w_n * log(p_n)))
//!
//! where:
//! - BP = exp(min(0, 1 - ref_len/hyp_len)) is the brevity penalty
//! - p_n is the modified n-gram precision for order n
//! - w_n is the weight for order n (default: uniform 1/max_n)
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_text::evaluation::bleu::{corpus_bleu, sentence_bleu, SmoothingMethod};
//!
//! let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
//! let reference = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
//! let score = sentence_bleu(&hypothesis, &reference, 4, SmoothingMethod::AddEpsilon(0.1))
//!     .expect("Operation failed");
//! assert!(score > 0.0 && score < 1.0);
//! ```

use std::collections::HashMap;

use crate::error::{Result, TextError};

/// Smoothing method for sentence-level BLEU.
///
/// At corpus level, n-gram counts are aggregated and smoothing is typically
/// unnecessary. At sentence level, zero n-gram counts for higher orders are
/// common and smoothing prevents the score from collapsing to zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothingMethod {
    /// No smoothing (standard BLEU). If any n-gram precision is zero,
    /// the overall score is zero.
    None,
    /// Add-epsilon smoothing (Chen & Cherry method 1).
    /// Adds epsilon to both numerator and denominator for n-gram orders
    /// where the count is zero.
    AddEpsilon(f64),
    /// Exponential decay smoothing (Chen & Cherry method 2).
    /// For n-gram orders with zero matches, use 1/(2^k) where k is
    /// the number of consecutive zero-count orders.
    ExponentialDecay,
}

/// Configuration for BLEU score computation.
#[derive(Debug, Clone)]
pub struct BleuConfig {
    /// Maximum n-gram order (default: 4).
    pub max_n: usize,
    /// Weights for each n-gram order. If None, uniform weights (1/max_n) are used.
    pub weights: Option<Vec<f64>>,
    /// Smoothing method for sentence-level BLEU.
    pub smoothing: SmoothingMethod,
}

impl Default for BleuConfig {
    fn default() -> Self {
        Self {
            max_n: 4,
            weights: None,
            smoothing: SmoothingMethod::None,
        }
    }
}

/// Extract n-grams of a given order from a token sequence.
fn extract_ngrams<'a>(tokens: &'a [&str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts: HashMap<Vec<&'a str>, usize> = HashMap::new();
    if tokens.len() >= n {
        for i in 0..=(tokens.len() - n) {
            let ngram = tokens[i..i + n].to_vec();
            *counts.entry(ngram).or_insert(0) += 1;
        }
    }
    counts
}

/// Compute modified n-gram precision for a single hypothesis against
/// multiple references.
///
/// For each n-gram in the hypothesis, its clipped count is
/// min(hyp_count, max_ref_count). The modified precision is
/// sum(clipped_counts) / sum(hyp_counts).
fn modified_precision(hypothesis: &[&str], references: &[Vec<&str>], n: usize) -> (usize, usize) {
    let hyp_ngrams = extract_ngrams(hypothesis, n);

    if hyp_ngrams.is_empty() {
        return (0, 0);
    }

    // For each n-gram, find the maximum count across all references
    let mut max_ref_counts: HashMap<Vec<&str>, usize> = HashMap::new();
    for reference in references {
        let ref_ngrams = extract_ngrams(reference, n);
        for (ngram, count) in &ref_ngrams {
            let entry = max_ref_counts.entry(ngram.clone()).or_insert(0);
            if *count > *entry {
                *entry = *count;
            }
        }
    }

    // Compute clipped counts
    let mut clipped_count = 0usize;
    let mut total_count = 0usize;

    for (ngram, hyp_count) in &hyp_ngrams {
        let max_ref = max_ref_counts.get(ngram).copied().unwrap_or(0);
        clipped_count += (*hyp_count).min(max_ref);
        total_count += *hyp_count;
    }

    (clipped_count, total_count)
}

/// Compute the closest reference length for brevity penalty.
///
/// Among all references, select the one whose length is closest to
/// the hypothesis length. In case of a tie, use the shorter reference.
fn closest_ref_length(hyp_len: usize, references: &[Vec<&str>]) -> usize {
    let mut best_len = 0usize;
    let mut best_diff = usize::MAX;

    for reference in references {
        let ref_len = reference.len();
        let diff = ref_len.abs_diff(hyp_len);
        if diff < best_diff || (diff == best_diff && ref_len < best_len) {
            best_diff = diff;
            best_len = ref_len;
        }
    }

    best_len
}

/// Compute brevity penalty.
///
/// BP = exp(min(0, 1 - ref_len/hyp_len))
fn brevity_penalty(hyp_len: usize, ref_len: usize) -> f64 {
    if hyp_len == 0 {
        return 0.0;
    }
    let ratio = ref_len as f64 / hyp_len as f64;
    if ratio > 1.0 {
        (1.0 - ratio).exp()
    } else {
        1.0
    }
}

/// Compute corpus-level BLEU score.
///
/// Aggregates n-gram counts across all sentence pairs before computing
/// precision. This is the standard way to compute BLEU for evaluation.
///
/// # Arguments
///
/// * `hypotheses` - List of hypothesis sentences, each as a slice of tokens.
/// * `references` - List of reference sets. Each entry is a Vec of reference
///   sentences for the corresponding hypothesis. Each reference is a Vec of tokens.
/// * `max_n` - Maximum n-gram order (typically 4).
///
/// # Returns
///
/// The corpus-level BLEU score in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if inputs are empty or mismatched.
pub fn corpus_bleu(
    hypotheses: &[Vec<&str>],
    references: &[Vec<Vec<&str>>],
    max_n: usize,
) -> Result<f64> {
    if hypotheses.is_empty() {
        return Err(TextError::InvalidInput(
            "Hypotheses list must not be empty".to_string(),
        ));
    }
    if hypotheses.len() != references.len() {
        return Err(TextError::InvalidInput(format!(
            "Number of hypotheses ({}) must match number of reference sets ({})",
            hypotheses.len(),
            references.len()
        )));
    }
    if max_n == 0 {
        return Err(TextError::InvalidInput(
            "max_n must be at least 1".to_string(),
        ));
    }

    // Validate that each reference set is non-empty
    for (i, refs) in references.iter().enumerate() {
        if refs.is_empty() {
            return Err(TextError::InvalidInput(format!(
                "Reference set at index {} must not be empty",
                i
            )));
        }
    }

    let weights: Vec<f64> = vec![1.0 / max_n as f64; max_n];

    // Aggregate counts across the corpus
    let mut total_clipped = vec![0usize; max_n];
    let mut total_count = vec![0usize; max_n];
    let mut total_hyp_len = 0usize;
    let mut total_ref_len = 0usize;

    for (hyp, refs) in hypotheses.iter().zip(references.iter()) {
        total_hyp_len += hyp.len();
        total_ref_len += closest_ref_length(hyp.len(), refs);

        for n in 1..=max_n {
            let (clipped, count) = modified_precision(hyp, refs, n);
            total_clipped[n - 1] += clipped;
            total_count[n - 1] += count;
        }
    }

    // Compute log-averaged precision
    let mut log_avg = 0.0f64;
    for n in 0..max_n {
        if total_count[n] == 0 || total_clipped[n] == 0 {
            // If any n-gram precision is zero, corpus BLEU is zero
            return Ok(0.0);
        }
        let precision = total_clipped[n] as f64 / total_count[n] as f64;
        log_avg += weights[n] * precision.ln();
    }

    let bp = brevity_penalty(total_hyp_len, total_ref_len);
    Ok(bp * log_avg.exp())
}

/// Compute sentence-level BLEU score with optional smoothing.
///
/// # Arguments
///
/// * `hypothesis` - The hypothesis sentence as a slice of tokens.
/// * `references` - One or more reference sentences, each as a Vec of tokens.
/// * `max_n` - Maximum n-gram order (typically 4).
/// * `smoothing` - Smoothing method to handle zero n-gram counts.
///
/// # Returns
///
/// The sentence-level BLEU score in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if inputs are invalid.
pub fn sentence_bleu(
    hypothesis: &[&str],
    references: &[Vec<&str>],
    max_n: usize,
    smoothing: SmoothingMethod,
) -> Result<f64> {
    if references.is_empty() {
        return Err(TextError::InvalidInput(
            "References must not be empty".to_string(),
        ));
    }
    if max_n == 0 {
        return Err(TextError::InvalidInput(
            "max_n must be at least 1".to_string(),
        ));
    }

    if hypothesis.is_empty() {
        return Ok(0.0);
    }

    let weights: Vec<f64> = vec![1.0 / max_n as f64; max_n];
    let ref_len = closest_ref_length(hypothesis.len(), references);
    let bp = brevity_penalty(hypothesis.len(), ref_len);

    let mut log_avg = 0.0f64;
    let mut consecutive_zeros = 0u32;

    for n in 1..=max_n {
        let (clipped, count) = modified_precision(hypothesis, references, n);

        let precision = match smoothing {
            SmoothingMethod::None => {
                if count == 0 || clipped == 0 {
                    return Ok(0.0);
                }
                clipped as f64 / count as f64
            }
            SmoothingMethod::AddEpsilon(eps) => {
                if count == 0 {
                    eps
                } else {
                    (clipped as f64 + eps) / (count as f64 + eps)
                }
            }
            SmoothingMethod::ExponentialDecay => {
                if count == 0 || clipped == 0 {
                    consecutive_zeros += 1;
                    1.0 / 2.0f64.powi(consecutive_zeros as i32)
                } else {
                    consecutive_zeros = 0;
                    clipped as f64 / count as f64
                }
            }
        };

        if precision <= 0.0 {
            return Ok(0.0);
        }
        log_avg += weights[n - 1] * precision.ln();
    }

    Ok(bp * log_avg.exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_translation() {
        let hypothesis = vec!["the", "cat", "is", "on", "the", "mat"];
        let reference = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
        let score = sentence_bleu(&hypothesis, &reference, 4, SmoothingMethod::None)
            .expect("should compute");
        assert!(
            (score - 1.0).abs() < 1e-9,
            "Perfect translation should score 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_no_overlap() {
        let hypothesis = vec!["a", "b", "c", "d"];
        let reference = vec![vec!["e", "f", "g", "h"]];
        let score = sentence_bleu(&hypothesis, &reference, 4, SmoothingMethod::None)
            .expect("should compute");
        assert!(
            score.abs() < 1e-9,
            "No overlap should score 0.0, got {}",
            score
        );
    }

    #[test]
    fn test_brevity_penalty_applied() {
        // Short hypothesis vs longer reference
        let hypothesis = vec!["the", "cat"];
        let reference = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
        let score = sentence_bleu(&hypothesis, &reference, 1, SmoothingMethod::AddEpsilon(0.1))
            .expect("should compute");
        // Unigram precision is high but BP should penalize
        assert!(score < 1.0, "BP should reduce score for short hyp");
        assert!(score > 0.0, "Score should be positive with partial match");
    }

    #[test]
    fn test_multiple_references() {
        let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
        let references = vec![
            vec!["the", "cat", "is", "on", "the", "mat"],
            vec!["the", "cat", "sat", "on", "the", "mat"],
        ];
        let score = sentence_bleu(&hypothesis, &references, 4, SmoothingMethod::None)
            .expect("should compute");
        assert!(
            (score - 1.0).abs() < 1e-9,
            "Should match second reference perfectly, got {}",
            score
        );
    }

    #[test]
    fn test_corpus_bleu_basic() {
        let hypotheses = vec![
            vec!["the", "cat", "is", "on", "the", "mat"],
            vec!["there", "is", "a", "cat", "on", "the", "mat"],
        ];
        let references = vec![
            vec![vec!["the", "cat", "is", "on", "the", "mat"]],
            vec![vec!["there", "is", "a", "cat", "on", "the", "mat"]],
        ];
        let score = corpus_bleu(&hypotheses, &references, 4).expect("should compute");
        assert!(
            (score - 1.0).abs() < 1e-9,
            "Perfect corpus should score 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_corpus_bleu_empty_fails() {
        let result = corpus_bleu(&[], &[], 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_smoothing_exponential_decay() {
        // Short hypothesis where higher-order n-grams may be zero
        let hypothesis = vec!["the", "cat"];
        let reference = vec![vec!["the", "cat", "sat"]];
        let score_none = sentence_bleu(&hypothesis, &reference, 4, SmoothingMethod::None)
            .expect("should compute");
        let score_smooth = sentence_bleu(
            &hypothesis,
            &reference,
            4,
            SmoothingMethod::ExponentialDecay,
        )
        .expect("should compute");
        // With no smoothing, zero 3-gram/4-gram precision kills the score
        assert!(
            score_none.abs() < 1e-9,
            "No smoothing should give 0 with missing n-grams"
        );
        assert!(
            score_smooth > 0.0,
            "Exponential decay smoothing should give positive score"
        );
    }

    #[test]
    fn test_partial_overlap() {
        let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
        let reference = vec![vec!["the", "cat", "is", "on", "the", "mat"]];
        let score = sentence_bleu(&hypothesis, &reference, 4, SmoothingMethod::AddEpsilon(0.1))
            .expect("should compute");
        // Should be between 0 and 1
        assert!(score > 0.0 && score < 1.0, "Partial overlap: got {}", score);
    }
}
