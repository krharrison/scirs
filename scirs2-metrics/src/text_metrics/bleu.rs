//! BLEU (Bilingual Evaluation Understudy) Score
//!
//! Computes BLEU scores for evaluating machine translation and text generation.
//! Supports 1-gram through 4-gram evaluation with optional smoothing.
//!
//! The BLEU score measures how many n-grams in the candidate translation
//! match n-grams in the reference translation, with a brevity penalty
//! for translations that are too short.
//!
//! # References
//!
//! Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002).
//! "BLEU: a method for automatic evaluation of machine translation."

use crate::error::{MetricsError, Result};
use std::collections::HashMap;

use super::{get_ngrams, tokenize};

/// Computes the BLEU score between a reference and candidate sentence.
///
/// # Arguments
///
/// * `reference` - The reference (ground truth) text.
/// * `candidate` - The candidate (generated) text.
/// * `max_n` - Maximum n-gram order (typically 4). Must be >= 1.
/// * `smoothing` - If true, uses add-1 smoothing for zero-count n-grams
///   (Chen & Cherry, 2014 smoothing method 1).
///
/// # Returns
///
/// The BLEU score in [0, 1].
///
/// # Errors
///
/// Returns error if max_n is 0 or inputs are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::bleu::bleu_score;
///
/// let bleu = bleu_score("the cat sat on the mat", "the cat sat on the mat", 4, false)
///     .expect("Failed");
/// assert!((bleu - 1.0).abs() < 1e-6);
/// ```
pub fn bleu_score(reference: &str, candidate: &str, max_n: usize, smoothing: bool) -> Result<f64> {
    if max_n == 0 {
        return Err(MetricsError::InvalidInput("max_n must be >= 1".to_string()));
    }

    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    if cand_tokens.is_empty() {
        return Ok(0.0);
    }
    if ref_tokens.is_empty() {
        return Ok(0.0);
    }

    bleu_from_tokens(&ref_tokens, &cand_tokens, max_n, smoothing)
}

/// Computes BLEU score from pre-tokenized sequences.
///
/// This is useful when you already have tokens and want to avoid re-tokenizing.
///
/// # Arguments
///
/// * `ref_tokens` - Reference tokens.
/// * `cand_tokens` - Candidate tokens.
/// * `max_n` - Maximum n-gram order.
/// * `smoothing` - Whether to use add-1 smoothing.
///
/// # Returns
///
/// The BLEU score in [0, 1].
pub fn bleu_from_tokens(
    ref_tokens: &[String],
    cand_tokens: &[String],
    max_n: usize,
    smoothing: bool,
) -> Result<f64> {
    if max_n == 0 {
        return Err(MetricsError::InvalidInput("max_n must be >= 1".to_string()));
    }

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return Ok(0.0);
    }

    let mut log_precision_sum = 0.0;
    let mut weight_count = 0;

    for n in 1..=max_n {
        let ref_ngrams = get_ngrams(ref_tokens, n);
        let cand_ngrams = get_ngrams(cand_tokens, n);

        if cand_ngrams.is_empty() {
            if smoothing {
                // With smoothing, treat missing n-grams as a very small precision
                log_precision_sum += (1.0_f64 / (cand_tokens.len() as f64 + 1.0)).ln();
                weight_count += 1;
            } else {
                // Without smoothing, if any n-gram precision is 0, BLEU is 0
                return Ok(0.0);
            }
            continue;
        }

        // Count reference n-gram frequencies (clipped counting)
        let mut ref_counts: HashMap<Vec<String>, usize> = HashMap::new();
        for ng in &ref_ngrams {
            *ref_counts.entry(ng.clone()).or_insert(0) += 1;
        }

        // Count candidate n-gram matches (clipped to reference counts)
        let mut cand_counts: HashMap<Vec<String>, usize> = HashMap::new();
        for ng in &cand_ngrams {
            *cand_counts.entry(ng.clone()).or_insert(0) += 1;
        }

        let mut clipped_matches = 0usize;
        for (ng, &cand_count) in &cand_counts {
            let ref_count = ref_counts.get(ng).copied().unwrap_or(0);
            clipped_matches += cand_count.min(ref_count);
        }

        let precision = if smoothing && clipped_matches == 0 {
            1.0 / (cand_ngrams.len() as f64 + 1.0)
        } else if clipped_matches == 0 {
            return Ok(0.0);
        } else {
            clipped_matches as f64 / cand_ngrams.len() as f64
        };

        log_precision_sum += precision.ln();
        weight_count += 1;
    }

    if weight_count == 0 {
        return Ok(0.0);
    }

    // Geometric mean of precisions (uniform weights)
    let log_avg = log_precision_sum / weight_count as f64;

    // Brevity penalty
    let bp = brevity_penalty(ref_tokens.len(), cand_tokens.len());

    Ok(bp * log_avg.exp())
}

/// Computes BLEU score for a specific n-gram order only (e.g., BLEU-1, BLEU-2).
///
/// Unlike `bleu_score`, this only uses a single n-gram order.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
/// * `n` - The n-gram order to use.
///
/// # Returns
///
/// The n-gram precision with brevity penalty in [0, 1].
pub fn bleu_n(reference: &str, candidate: &str, n: usize) -> Result<f64> {
    if n == 0 {
        return Err(MetricsError::InvalidInput("n must be >= 1".to_string()));
    }

    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    if cand_tokens.is_empty() || ref_tokens.is_empty() {
        return Ok(0.0);
    }

    let ref_ngrams = get_ngrams(&ref_tokens, n);
    let cand_ngrams = get_ngrams(&cand_tokens, n);

    if cand_ngrams.is_empty() {
        return Ok(0.0);
    }

    // Clipped counting
    let mut ref_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &ref_ngrams {
        *ref_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut cand_counts: HashMap<Vec<String>, usize> = HashMap::new();
    for ng in &cand_ngrams {
        *cand_counts.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut clipped_matches = 0usize;
    for (ng, &cand_count) in &cand_counts {
        let ref_count = ref_counts.get(ng).copied().unwrap_or(0);
        clipped_matches += cand_count.min(ref_count);
    }

    let precision = clipped_matches as f64 / cand_ngrams.len() as f64;
    let bp = brevity_penalty(ref_tokens.len(), cand_tokens.len());

    Ok(bp * precision)
}

/// Computes corpus-level BLEU score across multiple sentence pairs.
///
/// # Arguments
///
/// * `references` - Slice of reference texts.
/// * `candidates` - Slice of candidate texts (must be same length as references).
/// * `max_n` - Maximum n-gram order.
/// * `smoothing` - Whether to use smoothing.
///
/// # Returns
///
/// The corpus-level BLEU score in [0, 1].
pub fn corpus_bleu(
    references: &[&str],
    candidates: &[&str],
    max_n: usize,
    smoothing: bool,
) -> Result<f64> {
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
    if max_n == 0 {
        return Err(MetricsError::InvalidInput("max_n must be >= 1".to_string()));
    }

    // Tokenize all sentences
    let ref_token_lists: Vec<Vec<String>> = references.iter().map(|r| tokenize(r)).collect();
    let cand_token_lists: Vec<Vec<String>> = candidates.iter().map(|c| tokenize(c)).collect();

    // Compute corpus-level n-gram statistics
    let mut total_ref_len = 0usize;
    let mut total_cand_len = 0usize;

    let mut ngram_matches: Vec<usize> = vec![0; max_n];
    let mut ngram_totals: Vec<usize> = vec![0; max_n];

    for (ref_tok, cand_tok) in ref_token_lists.iter().zip(cand_token_lists.iter()) {
        total_ref_len += ref_tok.len();
        total_cand_len += cand_tok.len();

        for n in 1..=max_n {
            let ref_ngs = get_ngrams(ref_tok, n);
            let cand_ngs = get_ngrams(cand_tok, n);

            let mut ref_counts: HashMap<Vec<String>, usize> = HashMap::new();
            for ng in &ref_ngs {
                *ref_counts.entry(ng.clone()).or_insert(0) += 1;
            }

            let mut cand_counts: HashMap<Vec<String>, usize> = HashMap::new();
            for ng in &cand_ngs {
                *cand_counts.entry(ng.clone()).or_insert(0) += 1;
            }

            let mut clipped = 0usize;
            for (ng, &c_count) in &cand_counts {
                let r_count = ref_counts.get(ng).copied().unwrap_or(0);
                clipped += c_count.min(r_count);
            }

            ngram_matches[n - 1] += clipped;
            ngram_totals[n - 1] += cand_ngs.len();
        }
    }

    // Compute geometric mean of precisions
    let mut log_precision_sum = 0.0;
    let mut valid_n = 0;

    for n in 0..max_n {
        if ngram_totals[n] == 0 {
            if smoothing {
                log_precision_sum += (1.0_f64 / (total_cand_len as f64 + 1.0)).ln();
                valid_n += 1;
            } else {
                return Ok(0.0);
            }
            continue;
        }

        let precision = if smoothing && ngram_matches[n] == 0 {
            1.0 / (ngram_totals[n] as f64 + 1.0)
        } else if ngram_matches[n] == 0 {
            return Ok(0.0);
        } else {
            ngram_matches[n] as f64 / ngram_totals[n] as f64
        };

        log_precision_sum += precision.ln();
        valid_n += 1;
    }

    if valid_n == 0 {
        return Ok(0.0);
    }

    let log_avg = log_precision_sum / valid_n as f64;
    let bp = brevity_penalty(total_ref_len, total_cand_len);

    Ok(bp * log_avg.exp())
}

/// Computes the brevity penalty used in BLEU.
fn brevity_penalty(ref_len: usize, cand_len: usize) -> f64 {
    if cand_len == 0 {
        return 0.0;
    }
    if cand_len >= ref_len {
        1.0
    } else {
        (1.0 - ref_len as f64 / cand_len as f64).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- bleu_score tests ----

    #[test]
    fn test_bleu_perfect_match() {
        let val = bleu_score("the cat sat on the mat", "the cat sat on the mat", 4, false)
            .expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_completely_different() {
        let val = bleu_score("hello world", "goodbye moon", 4, false).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_partial_match() {
        let val = bleu_score("the cat sat on the mat", "the cat sits on a mat", 4, true)
            .expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_bleu_empty_candidate() {
        let val = bleu_score("hello world", "", 4, false).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_empty_reference() {
        let val = bleu_score("", "hello world", 4, false).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_max_n_zero() {
        assert!(bleu_score("hello", "hello", 0, false).is_err());
    }

    #[test]
    fn test_bleu_with_smoothing() {
        // Smoothing should give nonzero score even when some n-gram precisions are zero
        let val = bleu_score("a b c d e f", "a b x y z f", 4, true).expect("should succeed");
        assert!(val > 0.0);
    }

    // ---- bleu_n tests ----

    #[test]
    fn test_bleu_1_perfect() {
        let val = bleu_n("the cat sat", "the cat sat", 1).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_1_partial() {
        let val =
            bleu_n("the cat sat on the mat", "the cat sat on a rug", 1).expect("should succeed");
        // 4 out of 6 unigrams match, with brevity penalty = 1.0 (same length)
        assert!(val > 0.5);
    }

    #[test]
    fn test_bleu_2_perfect() {
        let val = bleu_n("the cat sat", "the cat sat", 2).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_4_short_sentence() {
        // Sentence shorter than 4 tokens, no 4-grams possible
        let val = bleu_n("hi there", "hi there", 4).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_bleu_n_zero() {
        assert!(bleu_n("hello", "hello", 0).is_err());
    }

    // ---- corpus_bleu tests ----

    #[test]
    fn test_corpus_bleu_perfect() {
        let refs = vec!["the cat sat on the big mat", "a brown dog ran very fast"];
        let cands = vec!["the cat sat on the big mat", "a brown dog ran very fast"];
        let val = corpus_bleu(&refs, &cands, 4, false).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_bleu_mismatched_lengths() {
        let refs = vec!["hello"];
        let cands = vec!["hello", "world"];
        assert!(corpus_bleu(&refs, &cands, 4, false).is_err());
    }

    #[test]
    fn test_corpus_bleu_partial() {
        let refs = vec!["the quick brown fox", "a lazy dog sleeps"];
        let cands = vec!["the slow brown fox", "a lazy cat sleeps"];
        let val = corpus_bleu(&refs, &cands, 4, true).expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_corpus_bleu_empty() {
        let refs: Vec<&str> = vec![];
        let cands: Vec<&str> = vec![];
        assert!(corpus_bleu(&refs, &cands, 4, false).is_err());
    }

    // ---- brevity_penalty tests ----

    #[test]
    fn test_brevity_penalty_longer_candidate() {
        let bp = brevity_penalty(5, 10);
        assert!((bp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brevity_penalty_shorter_candidate() {
        let bp = brevity_penalty(10, 5);
        assert!(bp < 1.0);
        assert!(bp > 0.0);
    }

    #[test]
    fn test_brevity_penalty_equal_length() {
        let bp = brevity_penalty(5, 5);
        assert!((bp - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brevity_penalty_zero_candidate() {
        let bp = brevity_penalty(5, 0);
        assert!((bp - 0.0).abs() < 1e-10);
    }
}
