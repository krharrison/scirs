//! Error Rate Metrics for Speech Recognition and OCR
//!
//! Computes Word Error Rate (WER) and Character Error Rate (CER)
//! for evaluating speech recognition, OCR, and text normalization systems.
//!
//! Both metrics are based on the edit distance (Levenshtein distance) between
//! the reference and hypothesis sequences. Lower values indicate better performance.
//!
//! # Formulas
//!
//! - WER = (S + D + I) / N  where S=substitutions, D=deletions, I=insertions, N=reference length
//! - CER = (S + D + I) / N  (same formula but at character level)
//!
//! Note: Values can exceed 1.0 when insertions exceed the reference length.

use crate::error::{MetricsError, Result};

use super::tokenize;

/// Computes Word Error Rate (WER) between reference and hypothesis.
///
/// WER is the standard metric for evaluating automatic speech recognition (ASR)
/// systems. It measures the minimum number of word-level edit operations
/// (insertions, deletions, substitutions) needed to transform the hypothesis
/// into the reference, normalized by the reference length.
///
/// # Arguments
///
/// * `reference` - The reference (ground truth) text.
/// * `hypothesis` - The hypothesis (predicted) text.
///
/// # Returns
///
/// The WER as a non-negative float. 0.0 means perfect match.
/// Values > 1.0 are possible when many insertions occur.
///
/// # Errors
///
/// Returns error if reference is empty (would cause division by zero).
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::error_rate::word_error_rate;
///
/// let wer = word_error_rate("the cat sat on the mat", "the cat sat on the mat")
///     .expect("Failed");
/// assert!((wer - 0.0).abs() < 1e-6);
///
/// let wer = word_error_rate("the cat sat", "the dog sat")
///     .expect("Failed");
/// assert!((wer - 1.0 / 3.0).abs() < 1e-6); // 1 substitution out of 3 words
/// ```
pub fn word_error_rate(reference: &str, hypothesis: &str) -> Result<f64> {
    let ref_tokens = tokenize(reference);
    let hyp_tokens = tokenize(hypothesis);

    if ref_tokens.is_empty() {
        if hyp_tokens.is_empty() {
            return Ok(0.0);
        }
        return Err(MetricsError::InvalidInput(
            "Reference is empty but hypothesis is not; cannot compute WER".to_string(),
        ));
    }

    let edit_dist = levenshtein_distance(&ref_tokens, &hyp_tokens);
    Ok(edit_dist as f64 / ref_tokens.len() as f64)
}

/// Computes Word Error Rate from pre-tokenized sequences.
pub fn word_error_rate_from_tokens(ref_tokens: &[String], hyp_tokens: &[String]) -> Result<f64> {
    if ref_tokens.is_empty() {
        if hyp_tokens.is_empty() {
            return Ok(0.0);
        }
        return Err(MetricsError::InvalidInput(
            "Reference is empty but hypothesis is not; cannot compute WER".to_string(),
        ));
    }

    let edit_dist = levenshtein_distance(ref_tokens, hyp_tokens);
    Ok(edit_dist as f64 / ref_tokens.len() as f64)
}

/// Computes Character Error Rate (CER) between reference and hypothesis.
///
/// CER is similar to WER but operates at the character level. It is commonly
/// used for evaluating OCR systems and character-level language models.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `hypothesis` - The hypothesis text.
///
/// # Returns
///
/// The CER as a non-negative float. 0.0 means perfect match.
///
/// # Errors
///
/// Returns error if reference is empty but hypothesis is not.
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::error_rate::character_error_rate;
///
/// let cer = character_error_rate("hello", "hello").expect("Failed");
/// assert!((cer - 0.0).abs() < 1e-6);
///
/// let cer = character_error_rate("cat", "car").expect("Failed");
/// assert!((cer - 1.0 / 3.0).abs() < 1e-6); // 1 substitution out of 3 chars
/// ```
pub fn character_error_rate(reference: &str, hypothesis: &str) -> Result<f64> {
    let ref_chars: Vec<char> = reference.chars().collect();
    let hyp_chars: Vec<char> = hypothesis.chars().collect();

    if ref_chars.is_empty() {
        if hyp_chars.is_empty() {
            return Ok(0.0);
        }
        return Err(MetricsError::InvalidInput(
            "Reference is empty but hypothesis is not; cannot compute CER".to_string(),
        ));
    }

    let edit_dist = levenshtein_distance_chars(&ref_chars, &hyp_chars);
    Ok(edit_dist as f64 / ref_chars.len() as f64)
}

/// Computes WER with detailed edit operation breakdown.
///
/// # Returns
///
/// A tuple of (wer, substitutions, deletions, insertions).
pub fn word_error_rate_detailed(reference: &str, hypothesis: &str) -> Result<WerDetail> {
    let ref_tokens = tokenize(reference);
    let hyp_tokens = tokenize(hypothesis);

    if ref_tokens.is_empty() {
        if hyp_tokens.is_empty() {
            return Ok(WerDetail {
                wer: 0.0,
                substitutions: 0,
                deletions: 0,
                insertions: 0,
                reference_length: 0,
                hypothesis_length: 0,
            });
        }
        return Err(MetricsError::InvalidInput(
            "Reference is empty but hypothesis is not".to_string(),
        ));
    }

    let (subs, dels, ins) = edit_operations(&ref_tokens, &hyp_tokens);
    let total_errors = subs + dels + ins;

    Ok(WerDetail {
        wer: total_errors as f64 / ref_tokens.len() as f64,
        substitutions: subs,
        deletions: dels,
        insertions: ins,
        reference_length: ref_tokens.len(),
        hypothesis_length: hyp_tokens.len(),
    })
}

/// Detailed WER result with edit operation breakdown.
#[derive(Debug, Clone)]
pub struct WerDetail {
    /// The word error rate
    pub wer: f64,
    /// Number of substitutions
    pub substitutions: usize,
    /// Number of deletions (words in reference not in hypothesis)
    pub deletions: usize,
    /// Number of insertions (words in hypothesis not in reference)
    pub insertions: usize,
    /// Length of the reference in words
    pub reference_length: usize,
    /// Length of the hypothesis in words
    pub hypothesis_length: usize,
}

/// Computes corpus-level WER across multiple utterances.
///
/// Uses micro-averaging: total edit distance / total reference length.
pub fn corpus_word_error_rate(references: &[&str], hypotheses: &[&str]) -> Result<f64> {
    if references.len() != hypotheses.len() {
        return Err(MetricsError::InvalidInput(
            "references and hypotheses must have the same length".to_string(),
        ));
    }
    if references.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut total_edit_dist = 0usize;
    let mut total_ref_len = 0usize;

    for (r, h) in references.iter().zip(hypotheses.iter()) {
        let ref_tokens = tokenize(r);
        let hyp_tokens = tokenize(h);
        total_edit_dist += levenshtein_distance(&ref_tokens, &hyp_tokens);
        total_ref_len += ref_tokens.len();
    }

    if total_ref_len == 0 {
        return Err(MetricsError::InvalidInput(
            "Total reference length is zero".to_string(),
        ));
    }

    Ok(total_edit_dist as f64 / total_ref_len as f64)
}

/// Computes corpus-level CER across multiple texts.
pub fn corpus_character_error_rate(references: &[&str], hypotheses: &[&str]) -> Result<f64> {
    if references.len() != hypotheses.len() {
        return Err(MetricsError::InvalidInput(
            "references and hypotheses must have the same length".to_string(),
        ));
    }
    if references.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut total_edit_dist = 0usize;
    let mut total_ref_len = 0usize;

    for (r, h) in references.iter().zip(hypotheses.iter()) {
        let ref_chars: Vec<char> = r.chars().collect();
        let hyp_chars: Vec<char> = h.chars().collect();
        total_edit_dist += levenshtein_distance_chars(&ref_chars, &hyp_chars);
        total_ref_len += ref_chars.len();
    }

    if total_ref_len == 0 {
        return Err(MetricsError::InvalidInput(
            "Total reference length is zero".to_string(),
        ));
    }

    Ok(total_edit_dist as f64 / total_ref_len as f64)
}

/// Computes the Levenshtein distance between two string sequences.
///
/// Uses dynamic programming with O(n) space.
fn levenshtein_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)          // deletion
                .min(curr[j - 1] + 1)        // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Levenshtein distance for character sequences.
fn levenshtein_distance_chars(a: &[char], b: &[char]) -> usize {
    levenshtein_distance(a, b)
}

/// Computes the number of substitutions, deletions, and insertions
/// using the Levenshtein DP backtrace.
fn edit_operations<T: PartialEq>(a: &[T], b: &[T]) -> (usize, usize, usize) {
    let m = a.len();
    let n = b.len();

    // Build full DP table for backtrace
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    // Backtrace to count operations
    let mut subs = 0usize;
    let mut dels = 0usize;
    let mut ins = 0usize;

    let mut i = m;
    let mut j = n;

    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                if cost == 1 {
                    subs += 1;
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            dels += 1;
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            ins += 1;
            j -= 1;
        } else {
            // Should not reach here with valid DP table
            break;
        }
    }

    (subs, dels, ins)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- WER tests ----

    #[test]
    fn test_wer_perfect() {
        let val = word_error_rate("the cat sat on the mat", "the cat sat on the mat")
            .expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_substitution() {
        let val = word_error_rate("the cat sat", "the dog sat").expect("should succeed");
        // 1 substitution out of 3 words
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_all_wrong() {
        let val = word_error_rate("a b c", "d e f").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_insertion() {
        let val = word_error_rate("a b", "a b c").expect("should succeed");
        // 1 insertion out of 2 reference words
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wer_deletion() {
        let val = word_error_rate("a b c", "a c").expect("should succeed");
        // 1 deletion out of 3 reference words
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_empty_both() {
        let val = word_error_rate("", "").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_empty_reference_nonempty_hyp() {
        assert!(word_error_rate("", "hello").is_err());
    }

    #[test]
    fn test_wer_from_tokens() {
        let ref_t: Vec<String> = vec!["a", "b", "c"].into_iter().map(String::from).collect();
        let hyp_t: Vec<String> = vec!["a", "d", "c"].into_iter().map(String::from).collect();
        let val = word_error_rate_from_tokens(&ref_t, &hyp_t).expect("should succeed");
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }

    // ---- CER tests ----

    #[test]
    fn test_cer_perfect() {
        let val = character_error_rate("hello", "hello").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_one_substitution() {
        let val = character_error_rate("cat", "car").expect("should succeed");
        // 1 substitution out of 3 chars
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_insertion() {
        let val = character_error_rate("ab", "abc").expect("should succeed");
        assert!((val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cer_deletion() {
        let val = character_error_rate("abc", "ac").expect("should succeed");
        assert!((val - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_completely_different() {
        let val = character_error_rate("abc", "xyz").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_empty_both() {
        let val = character_error_rate("", "").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_empty_reference() {
        assert!(character_error_rate("", "hello").is_err());
    }

    // ---- Detailed WER tests ----

    #[test]
    fn test_wer_detailed_perfect() {
        let detail =
            word_error_rate_detailed("the cat sat", "the cat sat").expect("should succeed");
        assert_eq!(detail.substitutions, 0);
        assert_eq!(detail.deletions, 0);
        assert_eq!(detail.insertions, 0);
        assert!((detail.wer - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_detailed_substitution() {
        let detail =
            word_error_rate_detailed("the cat sat", "the dog sat").expect("should succeed");
        assert_eq!(detail.substitutions, 1);
        assert_eq!(detail.deletions, 0);
        assert_eq!(detail.insertions, 0);
    }

    #[test]
    fn test_wer_detailed_deletion() {
        let detail = word_error_rate_detailed("a b c", "a c").expect("should succeed");
        assert_eq!(detail.deletions, 1);
        assert_eq!(detail.substitutions, 0);
        assert_eq!(detail.insertions, 0);
    }

    #[test]
    fn test_wer_detailed_insertion() {
        let detail = word_error_rate_detailed("a b", "a b c").expect("should succeed");
        assert_eq!(detail.insertions, 1);
        assert_eq!(detail.substitutions, 0);
        assert_eq!(detail.deletions, 0);
    }

    // ---- Corpus WER/CER tests ----

    #[test]
    fn test_corpus_wer_perfect() {
        let refs = vec!["the cat sat", "a dog ran"];
        let hyps = vec!["the cat sat", "a dog ran"];
        let val = corpus_word_error_rate(&refs, &hyps).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_wer_mismatched() {
        let refs = vec!["a"];
        let hyps = vec!["a", "b"];
        assert!(corpus_word_error_rate(&refs, &hyps).is_err());
    }

    #[test]
    fn test_corpus_wer_partial() {
        let refs = vec!["the cat sat", "hello world"];
        let hyps = vec!["the dog sat", "goodbye world"];
        let val = corpus_word_error_rate(&refs, &hyps).expect("should succeed");
        // 2 substitutions out of 5 total reference words
        assert!((val - 2.0 / 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_cer_perfect() {
        let refs = vec!["hello", "world"];
        let hyps = vec!["hello", "world"];
        let val = corpus_character_error_rate(&refs, &hyps).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_cer_partial() {
        let refs = vec!["cat", "dog"];
        let hyps = vec!["car", "dog"];
        let val = corpus_character_error_rate(&refs, &hyps).expect("should succeed");
        // 1 substitution out of 6 total reference chars
        assert!((val - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_corpus_cer_empty() {
        let refs: Vec<&str> = vec![];
        let hyps: Vec<&str> = vec![];
        assert!(corpus_character_error_rate(&refs, &hyps).is_err());
    }

    // ---- Levenshtein tests ----

    #[test]
    fn test_levenshtein_identical() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "b", "c"];
        assert_eq!(levenshtein_distance(&a, &b), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        let a: Vec<&str> = vec![];
        let b = vec!["a", "b"];
        assert_eq!(levenshtein_distance(&a, &b), 2);
    }

    #[test]
    fn test_levenshtein_one_sub() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "x", "c"];
        assert_eq!(levenshtein_distance(&a, &b), 1);
    }

    #[test]
    fn test_levenshtein_mixed_ops() {
        let a = vec!["a", "b", "c", "d"];
        let b = vec!["a", "c", "d", "e"];
        // delete "b", insert "e" -> 2
        assert_eq!(levenshtein_distance(&a, &b), 2);
    }
}
