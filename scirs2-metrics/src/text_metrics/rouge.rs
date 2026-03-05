//! ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Metrics
//!
//! Computes ROUGE-1, ROUGE-2, and ROUGE-L scores for evaluating text summarization
//! and text generation quality.
//!
//! - **ROUGE-1**: Unigram overlap (measures informativeness)
//! - **ROUGE-2**: Bigram overlap (measures fluency)
//! - **ROUGE-L**: Longest Common Subsequence (measures sentence-level structure)
//!
//! Each metric returns an F1-measure combining precision and recall.
//!
//! # References
//!
//! Lin, C.-Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries."

use crate::error::{MetricsError, Result};
use std::collections::HashMap;

use super::{get_ngrams, tokenize};

/// Result of a ROUGE computation, containing precision, recall, and F1.
#[derive(Debug, Clone, Copy)]
pub struct RougeScore {
    /// Precision: fraction of candidate n-grams found in reference
    pub precision: f64,
    /// Recall: fraction of reference n-grams found in candidate
    pub recall: f64,
    /// F1: harmonic mean of precision and recall
    pub f1: f64,
}

/// Computes ROUGE-1 (unigram overlap) F1 score between reference and candidate.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
///
/// # Returns
///
/// The ROUGE-1 F1 score in [0, 1].
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::rouge::rouge_1;
///
/// let score = rouge_1("the cat sat on the mat", "the cat sat on the mat")
///     .expect("Failed");
/// assert!((score - 1.0).abs() < 1e-6);
/// ```
pub fn rouge_1(reference: &str, candidate: &str) -> Result<f64> {
    let result = rouge_n_detailed(reference, candidate, 1)?;
    Ok(result.f1)
}

/// Computes ROUGE-2 (bigram overlap) F1 score between reference and candidate.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
///
/// # Returns
///
/// The ROUGE-2 F1 score in [0, 1].
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::rouge::rouge_2;
///
/// let score = rouge_2("the cat sat on the mat", "the cat sat on the mat")
///     .expect("Failed");
/// assert!((score - 1.0).abs() < 1e-6);
/// ```
pub fn rouge_2(reference: &str, candidate: &str) -> Result<f64> {
    let result = rouge_n_detailed(reference, candidate, 2)?;
    Ok(result.f1)
}

/// Computes ROUGE-N with full precision/recall/F1 breakdown.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
/// * `n` - The n-gram order.
///
/// # Returns
///
/// A [`RougeScore`] with precision, recall, and F1.
pub fn rouge_n_detailed(reference: &str, candidate: &str, n: usize) -> Result<RougeScore> {
    if n == 0 {
        return Err(MetricsError::InvalidInput("n must be >= 1".to_string()));
    }

    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    rouge_n_from_tokens(&ref_tokens, &cand_tokens, n)
}

/// Computes ROUGE-N from pre-tokenized sequences.
pub fn rouge_n_from_tokens(
    ref_tokens: &[String],
    cand_tokens: &[String],
    n: usize,
) -> Result<RougeScore> {
    if n == 0 {
        return Err(MetricsError::InvalidInput("n must be >= 1".to_string()));
    }

    // Handle edge cases
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return Ok(RougeScore {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(RougeScore {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        });
    }

    let ref_ngrams = get_ngrams(ref_tokens, n);
    let cand_ngrams = get_ngrams(cand_tokens, n);

    if ref_ngrams.is_empty() && cand_ngrams.is_empty() {
        return Ok(RougeScore {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }
    if ref_ngrams.is_empty() || cand_ngrams.is_empty() {
        return Ok(RougeScore {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        });
    }

    // Count frequencies
    let mut ref_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &ref_ngrams {
        *ref_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut cand_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &cand_ngrams {
        *cand_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    // Compute overlap (clipped matching)
    let mut overlap = 0usize;
    for (ng, &c_count) in &cand_counts {
        let r_count = ref_counts.get(ng).copied().unwrap_or(0);
        overlap += c_count.min(r_count);
    }

    let precision = overlap as f64 / cand_ngrams.len() as f64;
    let recall = overlap as f64 / ref_ngrams.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok(RougeScore {
        precision,
        recall,
        f1,
    })
}

/// Computes ROUGE-L (Longest Common Subsequence) F1 score.
///
/// ROUGE-L uses the LCS to measure sentence-level structural similarity.
/// Precision = LCS length / candidate length
/// Recall = LCS length / reference length
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
///
/// # Returns
///
/// The ROUGE-L F1 score in [0, 1].
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::rouge::rouge_l;
///
/// let score = rouge_l("the cat sat on the mat", "the cat sat on the mat")
///     .expect("Failed");
/// assert!((score - 1.0).abs() < 1e-6);
/// ```
pub fn rouge_l(reference: &str, candidate: &str) -> Result<f64> {
    let result = rouge_l_detailed(reference, candidate)?;
    Ok(result.f1)
}

/// Computes ROUGE-L with full precision/recall/F1 breakdown.
pub fn rouge_l_detailed(reference: &str, candidate: &str) -> Result<RougeScore> {
    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    rouge_l_from_tokens(&ref_tokens, &cand_tokens)
}

/// Computes ROUGE-L from pre-tokenized sequences.
pub fn rouge_l_from_tokens(ref_tokens: &[String], cand_tokens: &[String]) -> Result<RougeScore> {
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return Ok(RougeScore {
            precision: 1.0,
            recall: 1.0,
            f1: 1.0,
        });
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(RougeScore {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        });
    }

    let lcs_len = lcs_length(ref_tokens, cand_tokens);

    let precision = lcs_len as f64 / cand_tokens.len() as f64;
    let recall = lcs_len as f64 / ref_tokens.len() as f64;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    Ok(RougeScore {
        precision,
        recall,
        f1,
    })
}

/// Computes the length of the Longest Common Subsequence.
fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();

    // Use O(n) space with two rows
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
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

    *prev.iter().max().unwrap_or(&0)
}

/// Computes corpus-level ROUGE-1 F1, averaged across sentence pairs.
pub fn corpus_rouge_1(references: &[&str], candidates: &[&str]) -> Result<f64> {
    corpus_rouge_n(references, candidates, 1)
}

/// Computes corpus-level ROUGE-2 F1, averaged across sentence pairs.
pub fn corpus_rouge_2(references: &[&str], candidates: &[&str]) -> Result<f64> {
    corpus_rouge_n(references, candidates, 2)
}

/// Computes corpus-level ROUGE-N F1, averaged across sentence pairs.
pub fn corpus_rouge_n(references: &[&str], candidates: &[&str], n: usize) -> Result<f64> {
    if references.len() != candidates.len() {
        return Err(MetricsError::InvalidInput(
            "references and candidates must have the same length".to_string(),
        ));
    }
    if references.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut f1_sum = 0.0;
    for (r, c) in references.iter().zip(candidates.iter()) {
        let score = rouge_n_detailed(r, c, n)?;
        f1_sum += score.f1;
    }

    Ok(f1_sum / references.len() as f64)
}

/// Computes corpus-level ROUGE-L F1, averaged across sentence pairs.
pub fn corpus_rouge_l(references: &[&str], candidates: &[&str]) -> Result<f64> {
    if references.len() != candidates.len() {
        return Err(MetricsError::InvalidInput(
            "references and candidates must have the same length".to_string(),
        ));
    }
    if references.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut f1_sum = 0.0;
    for (r, c) in references.iter().zip(candidates.iter()) {
        let score = rouge_l_detailed(r, c)?;
        f1_sum += score.f1;
    }

    Ok(f1_sum / references.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ROUGE-1 tests ----

    #[test]
    fn test_rouge_1_perfect() {
        let val =
            rouge_1("the cat sat on the mat", "the cat sat on the mat").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_1_no_overlap() {
        let val = rouge_1("hello world", "goodbye moon").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_1_partial() {
        let val =
            rouge_1("the cat sat on the mat", "the cat sits on a mat").expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_rouge_1_empty_both() {
        let val = rouge_1("", "").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_1_empty_reference() {
        let val = rouge_1("", "hello").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    // ---- ROUGE-2 tests ----

    #[test]
    fn test_rouge_2_perfect() {
        let val =
            rouge_2("the cat sat on the mat", "the cat sat on the mat").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_2_no_overlap() {
        let val = rouge_2("hello world foo", "goodbye moon bar").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_2_partial() {
        // "the cat" bigram is shared
        let val =
            rouge_2("the cat sat on the mat", "the cat runs on a mat").expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_rouge_2_single_word() {
        // Single word has no bigrams
        let val = rouge_2("hello", "hello").expect("should succeed");
        // Both have no bigrams -> treat as perfect match
        assert!((val - 1.0).abs() < 1e-6);
    }

    // ---- ROUGE-L tests ----

    #[test]
    fn test_rouge_l_perfect() {
        let val =
            rouge_l("the cat sat on the mat", "the cat sat on the mat").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_no_overlap() {
        let val = rouge_l("hello world", "goodbye moon").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_partial() {
        let val =
            rouge_l("the cat sat on the mat", "the cat runs on a mat").expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_rouge_l_subsequence() {
        // LCS is "a c" (length 2), ref=3, cand=4
        let val = rouge_l("a b c", "a x c y").expect("should succeed");
        // P = 2/4 = 0.5, R = 2/3 = 0.667, F1 = 2*0.5*0.667 / (0.5+0.667) = 0.571
        assert!(val > 0.5 && val < 0.6);
    }

    #[test]
    fn test_rouge_l_empty_both() {
        let val = rouge_l("", "").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    // ---- ROUGE-N detailed tests ----

    #[test]
    fn test_rouge_n_detailed_precision_recall() {
        let score = rouge_n_detailed("a b c d", "a b e f", 1).expect("should succeed");
        // Overlap: "a", "b" -> 2 shared
        // Precision = 2/4 = 0.5, Recall = 2/4 = 0.5, F1 = 0.5
        assert!((score.precision - 0.5).abs() < 1e-6);
        assert!((score.recall - 0.5).abs() < 1e-6);
        assert!((score.f1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_n_detailed_asymmetric() {
        // Reference has 3 tokens, candidate has 5 tokens
        // Overlap: "a", "b", "c" -> 3 shared
        let score = rouge_n_detailed("a b c", "a b c d e", 1).expect("should succeed");
        // Precision = 3/5 = 0.6, Recall = 3/3 = 1.0
        assert!((score.precision - 0.6).abs() < 1e-6);
        assert!((score.recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_n_zero_n() {
        assert!(rouge_n_detailed("hello", "hello", 0).is_err());
    }

    #[test]
    fn test_rouge_n_from_tokens() {
        let ref_t: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let cand_t: Vec<String> = vec!["a", "b", "d"].into_iter().map(String::from).collect();
        let score = rouge_n_from_tokens(&ref_t, &cand_t, 1).expect("should succeed");
        // 2 out of 3 overlap
        assert!((score.f1 - 2.0 / 3.0).abs() < 0.01);
    }

    // ---- Corpus-level tests ----

    #[test]
    fn test_corpus_rouge_1_perfect() {
        let refs = vec!["hello world", "foo bar"];
        let cands = vec!["hello world", "foo bar"];
        let val = corpus_rouge_1(&refs, &cands).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_rouge_2_partial() {
        let refs = vec!["the cat sat on the mat", "a quick brown fox"];
        let cands = vec!["the cat sits on a mat", "a slow brown fox"];
        let val = corpus_rouge_2(&refs, &cands).expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_corpus_rouge_l_perfect() {
        let refs = vec!["hello world"];
        let cands = vec!["hello world"];
        let val = corpus_rouge_l(&refs, &cands).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_rouge_mismatched_len() {
        let refs = vec!["a"];
        let cands = vec!["a", "b"];
        assert!(corpus_rouge_1(&refs, &cands).is_err());
    }

    // ---- LCS tests ----

    #[test]
    fn test_lcs_identical() {
        let a: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let b = a.clone();
        assert_eq!(lcs_length(&a, &b), 3);
    }

    #[test]
    fn test_lcs_no_common() {
        let a: Vec<String> = vec!["a", "b"].into_iter().map(String::from).collect();
        let b: Vec<String> = vec!["c", "d"].into_iter().map(String::from).collect();
        assert_eq!(lcs_length(&a, &b), 0);
    }

    #[test]
    fn test_lcs_partial() {
        let a: Vec<String> = vec!["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .collect();
        let b: Vec<String> = vec!["a", "x", "c", "y"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(lcs_length(&a, &b), 2); // "a", "c"
    }

    #[test]
    fn test_lcs_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec!["a".to_string()];
        assert_eq!(lcs_length(&a, &b), 0);
    }
}
