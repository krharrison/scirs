//! # ROUGE Score (Recall-Oriented Understudy for Gisting Evaluation)
//!
//! Implementation of ROUGE metrics for text summarization evaluation.
//! ROUGE measures the overlap between a generated summary and reference
//! summaries, with an emphasis on recall.
//!
//! ## Variants
//!
//! - **ROUGE-N**: N-gram overlap (ROUGE-1 for unigrams, ROUGE-2 for bigrams)
//! - **ROUGE-L**: Longest Common Subsequence based F-measure
//! - **ROUGE-Lsum**: Sentence-level LCS aggregation
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_text::evaluation::rouge::{rouge_n, rouge_l};
//!
//! let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
//! let reference = vec!["the", "cat", "is", "on", "the", "mat"];
//!
//! let r1 = rouge_n(&hypothesis, &reference, 1).expect("Operation failed");
//! println!("ROUGE-1: P={:.3} R={:.3} F1={:.3}", r1.precision, r1.recall, r1.f1);
//!
//! let rl = rouge_l(&hypothesis, &reference).expect("Operation failed");
//! println!("ROUGE-L: P={:.3} R={:.3} F1={:.3}", rl.precision, rl.recall, rl.f1);
//! ```

use std::collections::HashMap;

use crate::error::{Result, TextError};

/// ROUGE score containing precision, recall, and F1 measure.
#[derive(Debug, Clone, Copy)]
pub struct RougeScore {
    /// Precision: proportion of hypothesis n-grams found in reference.
    pub precision: f64,
    /// Recall: proportion of reference n-grams found in hypothesis.
    pub recall: f64,
    /// F1 score: harmonic mean of precision and recall.
    pub f1: f64,
}

impl RougeScore {
    /// Create a new RougeScore from precision and recall, computing F1.
    fn from_precision_recall(precision: f64, recall: f64) -> Self {
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self {
            precision,
            recall,
            f1,
        }
    }
}

/// Extract n-grams from tokens and return counts.
fn ngram_counts<'a>(tokens: &'a [&str], n: usize) -> HashMap<Vec<&'a str>, usize> {
    let mut counts: HashMap<Vec<&'a str>, usize> = HashMap::new();
    if tokens.len() >= n {
        for i in 0..=(tokens.len() - n) {
            let ngram = tokens[i..i + n].to_vec();
            *counts.entry(ngram).or_insert(0) += 1;
        }
    }
    counts
}

/// Compute ROUGE-N score (n-gram overlap).
///
/// Measures the overlap of n-grams between hypothesis and reference.
/// For multiple references, compute against each and take the maximum F1.
///
/// # Arguments
///
/// * `hypothesis` - Generated text as a slice of tokens.
/// * `reference` - Reference text as a slice of tokens.
/// * `n` - N-gram order (1 for unigrams, 2 for bigrams, etc.).
///
/// # Returns
///
/// `RougeScore` with precision, recall, and F1.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if n is zero.
pub fn rouge_n(hypothesis: &[&str], reference: &[&str], n: usize) -> Result<RougeScore> {
    if n == 0 {
        return Err(TextError::InvalidInput(
            "N-gram order must be at least 1".to_string(),
        ));
    }

    let hyp_ngrams = ngram_counts(hypothesis, n);
    let ref_ngrams = ngram_counts(reference, n);

    let hyp_total: usize = hyp_ngrams.values().sum();
    let ref_total: usize = ref_ngrams.values().sum();

    if hyp_total == 0 && ref_total == 0 {
        return Ok(RougeScore::from_precision_recall(1.0, 1.0));
    }
    if hyp_total == 0 || ref_total == 0 {
        return Ok(RougeScore::from_precision_recall(0.0, 0.0));
    }

    // Count overlapping n-grams (clipped by min count)
    let mut overlap = 0usize;
    for (ngram, &hyp_count) in &hyp_ngrams {
        if let Some(&ref_count) = ref_ngrams.get(ngram) {
            overlap += hyp_count.min(ref_count);
        }
    }

    let precision = overlap as f64 / hyp_total as f64;
    let recall = overlap as f64 / ref_total as f64;

    Ok(RougeScore::from_precision_recall(precision, recall))
}

/// Compute ROUGE-N for multiple references and return the best score (by F1).
///
/// # Arguments
///
/// * `hypothesis` - Generated text tokens.
/// * `references` - Slice of reference texts, each as a Vec of tokens.
/// * `n` - N-gram order.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if references is empty or n is zero.
pub fn rouge_n_multi(
    hypothesis: &[&str],
    references: &[Vec<&str>],
    n: usize,
) -> Result<RougeScore> {
    if references.is_empty() {
        return Err(TextError::InvalidInput(
            "References must not be empty".to_string(),
        ));
    }

    let mut best_score = RougeScore::from_precision_recall(0.0, 0.0);
    for reference in references {
        let score = rouge_n(hypothesis, reference, n)?;
        if score.f1 > best_score.f1 {
            best_score = score;
        }
    }
    Ok(best_score)
}

/// Compute the length of the longest common subsequence between two token sequences.
///
/// Uses dynamic programming with O(m*n) time and O(min(m,n)) space.
fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 || n == 0 {
        return 0;
    }

    // Use the shorter sequence for the DP row to save memory
    let (short, long, s_len, l_len) = if m <= n { (a, b, m, n) } else { (b, a, n, m) };

    let mut prev = vec![0usize; s_len + 1];
    let mut curr = vec![0usize; s_len + 1];

    for i in 1..=l_len {
        for j in 1..=s_len {
            if long[i - 1] == short[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        for val in curr.iter_mut() {
            *val = 0;
        }
    }

    // Result is in prev after the final swap
    *prev.iter().max().unwrap_or(&0)
}

/// Compute ROUGE-L score using Longest Common Subsequence.
///
/// ROUGE-L measures the longest common subsequence between hypothesis
/// and reference, producing precision, recall, and F1.
///
/// # Arguments
///
/// * `hypothesis` - Generated text tokens.
/// * `reference` - Reference text tokens.
///
/// # Returns
///
/// `RougeScore` with LCS-based precision, recall, and F1.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if both inputs are empty.
pub fn rouge_l(hypothesis: &[&str], reference: &[&str]) -> Result<RougeScore> {
    let hyp_len = hypothesis.len();
    let ref_len = reference.len();

    if hyp_len == 0 && ref_len == 0 {
        return Ok(RougeScore::from_precision_recall(1.0, 1.0));
    }
    if hyp_len == 0 || ref_len == 0 {
        return Ok(RougeScore::from_precision_recall(0.0, 0.0));
    }

    let lcs_len = lcs_length(hypothesis, reference);
    let precision = lcs_len as f64 / hyp_len as f64;
    let recall = lcs_len as f64 / ref_len as f64;

    Ok(RougeScore::from_precision_recall(precision, recall))
}

/// Compute ROUGE-Lsum: sentence-level LCS aggregation.
///
/// Splits hypothesis and reference into sentences (using a provided
/// delimiter), computes LCS for each sentence pair, and aggregates.
///
/// # Arguments
///
/// * `hypothesis_sentences` - Hypothesis text split into sentences,
///   each sentence as a Vec of tokens.
/// * `reference_sentences` - Reference text split into sentences.
///
/// # Returns
///
/// `RougeScore` with aggregated LCS-based scores.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if inputs are invalid.
pub fn rouge_l_summary(
    hypothesis_sentences: &[Vec<&str>],
    reference_sentences: &[Vec<&str>],
) -> Result<RougeScore> {
    if hypothesis_sentences.is_empty() && reference_sentences.is_empty() {
        return Ok(RougeScore::from_precision_recall(1.0, 1.0));
    }
    if hypothesis_sentences.is_empty() || reference_sentences.is_empty() {
        return Ok(RougeScore::from_precision_recall(0.0, 0.0));
    }

    let mut total_lcs = 0usize;
    let mut total_hyp_len = 0usize;
    let mut total_ref_len = 0usize;

    // For each reference sentence, find the best matching hypothesis sentence
    for ref_sent in reference_sentences {
        let mut best_lcs = 0usize;
        for hyp_sent in hypothesis_sentences {
            let lcs = lcs_length(hyp_sent, ref_sent);
            if lcs > best_lcs {
                best_lcs = lcs;
            }
        }
        total_lcs += best_lcs;
        total_ref_len += ref_sent.len();
    }

    for hyp_sent in hypothesis_sentences {
        total_hyp_len += hyp_sent.len();
    }

    let precision = if total_hyp_len > 0 {
        total_lcs as f64 / total_hyp_len as f64
    } else {
        0.0
    };
    let recall = if total_ref_len > 0 {
        total_lcs as f64 / total_ref_len as f64
    } else {
        0.0
    };

    Ok(RougeScore::from_precision_recall(precision, recall))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rouge_1_known_overlap() {
        let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
        let reference = vec!["the", "cat", "is", "on", "the", "mat"];
        let score = rouge_n(&hypothesis, &reference, 1).expect("should compute");

        // Hypothesis unigrams: the(2), cat(1), sat(1), on(1), mat(1) = 6 total
        // Reference unigrams: the(2), cat(1), is(1), on(1), mat(1) = 6 total
        // Overlap: the(2), cat(1), on(1), mat(1) = 5
        assert!((score.recall - 5.0 / 6.0).abs() < 1e-9);
        assert!((score.precision - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_2_bigrams() {
        let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
        let reference = vec!["the", "cat", "is", "on", "the", "mat"];
        let score = rouge_n(&hypothesis, &reference, 2).expect("should compute");

        // Hyp bigrams: the-cat, cat-sat, sat-on, on-the, the-mat (5)
        // Ref bigrams: the-cat, cat-is, is-on, on-the, the-mat (5)
        // Overlap: the-cat(1), on-the(1), the-mat(1) = 3
        assert!((score.recall - 3.0 / 5.0).abs() < 1e-9);
        assert!((score.precision - 3.0 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_lcs() {
        let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
        let reference = vec!["the", "cat", "is", "on", "the", "mat"];
        let score = rouge_l(&hypothesis, &reference).expect("should compute");

        // LCS: "the", "cat", "on", "the", "mat" = length 5
        assert!((score.recall - 5.0 / 6.0).abs() < 1e-9);
        assert!((score.precision - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_no_overlap() {
        let hypothesis = vec!["a", "b", "c"];
        let reference = vec!["d", "e", "f"];
        let score = rouge_l(&hypothesis, &reference).expect("should compute");
        assert!(score.f1.abs() < 1e-9);
    }

    #[test]
    fn test_rouge_l_perfect() {
        let hypothesis = vec!["the", "cat", "is", "here"];
        let reference = vec!["the", "cat", "is", "here"];
        let score = rouge_l(&hypothesis, &reference).expect("should compute");
        assert!((score.f1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_rouge_n_multi_best() {
        let hypothesis = vec!["the", "cat", "sat"];
        let references = vec![vec!["a", "dog", "ran"], vec!["the", "cat", "sat"]];
        let score = rouge_n_multi(&hypothesis, &references, 1).expect("should compute");
        assert!((score.f1 - 1.0).abs() < 1e-9, "Should match best reference");
    }

    #[test]
    fn test_rouge_lsum() {
        let hyp_sentences = vec![vec!["the", "cat", "sat"], vec!["on", "the", "mat"]];
        let ref_sentences = vec![vec!["the", "cat", "is"], vec!["on", "the", "mat"]];
        let score = rouge_l_summary(&hyp_sentences, &ref_sentences).expect("should compute");
        assert!(score.f1 > 0.5, "Should have decent overlap");
    }

    #[test]
    fn test_rouge_n_zero_order_error() {
        let result = rouge_n(&["a"], &["a"], 0);
        assert!(result.is_err());
    }
}
