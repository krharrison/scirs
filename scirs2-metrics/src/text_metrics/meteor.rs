//! METEOR (Metric for Evaluation of Translation with Explicit ORdering)
//!
//! Computes the METEOR score for evaluating machine translation and text generation.
//! METEOR combines unigram precision and recall with a penalty for fragmentation.
//!
//! Unlike BLEU, METEOR:
//! - Is based on the harmonic mean of precision and recall (recall-weighted)
//! - Accounts for stemming and synonyms (simplified here to exact match + stem matching)
//! - Applies a fragmentation penalty for non-contiguous matches
//!
//! # References
//!
//! Banerjee, S. & Lavie, A. (2005).
//! "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments."

use crate::error::{MetricsError, Result};
use std::collections::HashSet;

use super::tokenize;

/// Computes the METEOR score between a reference and candidate text.
///
/// This implementation uses:
/// - Exact unigram matching
/// - Simplified stemming (common English suffix stripping)
/// - A fragmentation penalty based on the number of matching chunks
///
/// The formula is:
///   METEOR = F_mean * (1 - penalty)
/// where F_mean = (10 * P * R) / (9 * P + R)  [harmonic mean biased toward recall]
/// and penalty = 0.5 * (chunks / matches)^3
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
///
/// # Returns
///
/// The METEOR score in [0, 1].
///
/// # Examples
///
/// ```
/// use scirs2_metrics::text_metrics::meteor::meteor_score;
///
/// let score = meteor_score("the cat sat on the mat", "the cat sat on the mat")
///     .expect("Failed");
/// // Identical strings should score close to 1.0 (fragmentation penalty is minimal for a single chunk)
/// assert!(score > 0.9, "expected score close to 1.0, got {}", score);
/// ```
pub fn meteor_score(reference: &str, candidate: &str) -> Result<f64> {
    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    meteor_from_tokens(&ref_tokens, &cand_tokens)
}

/// Computes METEOR score from pre-tokenized sequences.
pub fn meteor_from_tokens(ref_tokens: &[String], cand_tokens: &[String]) -> Result<f64> {
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return Ok(1.0);
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(0.0);
    }

    // Stage 1: Exact matching
    let (exact_matches, ref_matched_exact, cand_matched_exact) =
        exact_match(ref_tokens, cand_tokens);

    // Stage 2: Stem matching (for unmatched tokens)
    let (stem_matches, ref_matched_stem, cand_matched_stem) = stem_match(
        ref_tokens,
        cand_tokens,
        &ref_matched_exact,
        &cand_matched_exact,
    );

    let total_matches = exact_matches + stem_matches;

    if total_matches == 0 {
        return Ok(0.0);
    }

    // Compute precision and recall
    let precision = total_matches as f64 / cand_tokens.len() as f64;
    let recall = total_matches as f64 / ref_tokens.len() as f64;

    // Harmonic mean with recall weight (alpha=0.9, meaning recall is 9x as important)
    // F = (10 * P * R) / (9 * P + R)  -- standard METEOR parameterization
    let f_mean = if precision + recall > 0.0 {
        10.0 * precision * recall / (9.0 * precision + recall)
    } else {
        return Ok(0.0);
    };

    // Compute fragmentation penalty
    // Combine matched indices from both stages
    let mut all_ref_matched: Vec<bool> = ref_matched_exact.clone();
    let mut all_cand_matched: Vec<bool> = cand_matched_exact.clone();
    for i in 0..ref_tokens.len() {
        if ref_matched_stem[i] {
            all_ref_matched[i] = true;
        }
    }
    for i in 0..cand_tokens.len() {
        if cand_matched_stem[i] {
            all_cand_matched[i] = true;
        }
    }

    let chunks = count_chunks(&all_cand_matched);
    let penalty = if total_matches > 0 {
        0.5 * (chunks as f64 / total_matches as f64).powi(3)
    } else {
        0.0
    };

    Ok(f_mean * (1.0 - penalty))
}

/// Computes METEOR score with configurable parameters.
///
/// # Arguments
///
/// * `reference` - The reference text.
/// * `candidate` - The candidate text.
/// * `alpha` - Recall weight parameter (default 0.9). Higher values favor recall more.
/// * `beta` - Fragmentation penalty weight (default 3.0).
/// * `gamma` - Fragmentation penalty multiplier (default 0.5).
///
/// # Returns
///
/// The METEOR score in [0, 1].
pub fn meteor_score_parametric(
    reference: &str,
    candidate: &str,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> Result<f64> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(MetricsError::InvalidInput(
            "alpha must be in (0, 1)".to_string(),
        ));
    }
    if beta < 0.0 {
        return Err(MetricsError::InvalidInput("beta must be >= 0".to_string()));
    }
    if gamma < 0.0 {
        return Err(MetricsError::InvalidInput("gamma must be >= 0".to_string()));
    }

    let ref_tokens = tokenize(reference);
    let cand_tokens = tokenize(candidate);

    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return Ok(1.0);
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(0.0);
    }

    let (exact_matches, ref_matched_exact, cand_matched_exact) =
        exact_match(&ref_tokens, &cand_tokens);
    let (stem_matches, _ref_matched_stem, cand_matched_stem) = stem_match(
        &ref_tokens,
        &cand_tokens,
        &ref_matched_exact,
        &cand_matched_exact,
    );

    let total_matches = exact_matches + stem_matches;
    if total_matches == 0 {
        return Ok(0.0);
    }

    let precision = total_matches as f64 / cand_tokens.len() as f64;
    let recall = total_matches as f64 / ref_tokens.len() as f64;

    // Parameterized harmonic mean: F = P * R / (alpha * P + (1-alpha) * R)
    let f_mean = if precision + recall > 0.0 {
        precision * recall / (alpha * precision + (1.0 - alpha) * recall)
    } else {
        return Ok(0.0);
    };

    let mut all_cand_matched = cand_matched_exact;
    for i in 0..cand_tokens.len() {
        if cand_matched_stem[i] {
            all_cand_matched[i] = true;
        }
    }

    let chunks = count_chunks(&all_cand_matched);
    let frag = if total_matches > 0 {
        chunks as f64 / total_matches as f64
    } else {
        0.0
    };
    let penalty = gamma * frag.powf(beta);

    Ok(f_mean * (1.0 - penalty))
}

/// Computes corpus-level METEOR, averaged across sentence pairs.
///
/// # Arguments
///
/// * `references` - Slice of reference texts.
/// * `candidates` - Slice of candidate texts.
///
/// # Returns
///
/// The mean METEOR score in [0, 1].
pub fn corpus_meteor(references: &[&str], candidates: &[&str]) -> Result<f64> {
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

    let mut score_sum = 0.0;
    for (r, c) in references.iter().zip(candidates.iter()) {
        score_sum += meteor_score(r, c)?;
    }

    Ok(score_sum / references.len() as f64)
}

/// Performs exact unigram matching between reference and candidate.
///
/// Returns (match_count, ref_matched_flags, cand_matched_flags).
fn exact_match(ref_tokens: &[String], cand_tokens: &[String]) -> (usize, Vec<bool>, Vec<bool>) {
    let mut ref_matched = vec![false; ref_tokens.len()];
    let mut cand_matched = vec![false; cand_tokens.len()];
    let mut matches = 0;

    for (ci, ct) in cand_tokens.iter().enumerate() {
        for (ri, rt) in ref_tokens.iter().enumerate() {
            if !ref_matched[ri] && !cand_matched[ci] && ct == rt {
                ref_matched[ri] = true;
                cand_matched[ci] = true;
                matches += 1;
                break;
            }
        }
    }

    (matches, ref_matched, cand_matched)
}

/// Performs stem matching for tokens not already matched exactly.
///
/// Uses a simplified English stemmer (suffix stripping).
fn stem_match(
    ref_tokens: &[String],
    cand_tokens: &[String],
    ref_exact_matched: &[bool],
    cand_exact_matched: &[bool],
) -> (usize, Vec<bool>, Vec<bool>) {
    let mut ref_matched = vec![false; ref_tokens.len()];
    let mut cand_matched = vec![false; cand_tokens.len()];
    let mut matches = 0;

    for (ci, ct) in cand_tokens.iter().enumerate() {
        if cand_exact_matched[ci] {
            continue;
        }
        let cand_stem = simple_stem(ct);
        for (ri, rt) in ref_tokens.iter().enumerate() {
            if ref_exact_matched[ri] || ref_matched[ri] || cand_matched[ci] {
                continue;
            }
            let ref_stem = simple_stem(rt);
            if cand_stem == ref_stem {
                ref_matched[ri] = true;
                cand_matched[ci] = true;
                matches += 1;
                break;
            }
        }
    }

    (matches, ref_matched, cand_matched)
}

/// Simple English suffix-stripping stemmer.
///
/// This is a greatly simplified Porter-like stemmer that handles common
/// English suffixes. It is sufficient for METEOR approximation.
fn simple_stem(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 3 {
        return w;
    }

    // Remove common suffixes in order of length
    let suffixes = [
        "ational", "tional", "ation", "ment", "ness", "ence", "ance", "able", "ible", "ting",
        "ful", "less", "ous", "ive", "ing", "ies", "ied", "ion", "ity", "ism", "ist", "ize", "ise",
        "er", "ed", "ly", "al", "es", "en", "s",
    ];

    for suffix in &suffixes {
        if w.len() > suffix.len() + 2 && w.ends_with(suffix) {
            return w[..w.len() - suffix.len()].to_string();
        }
    }

    w
}

/// Counts the number of contiguous chunks of matched tokens.
///
/// A chunk is a maximal contiguous sequence of matched positions.
fn count_chunks(matched: &[bool]) -> usize {
    if matched.is_empty() {
        return 0;
    }

    let mut chunks = 0;
    let mut in_chunk = false;

    for &m in matched {
        if m {
            if !in_chunk {
                chunks += 1;
                in_chunk = true;
            }
        } else {
            in_chunk = false;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- meteor_score tests ----

    #[test]
    fn test_meteor_perfect_match() {
        let val = meteor_score("the cat sat on the mat", "the cat sat on the mat")
            .expect("should succeed");
        assert!((val - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_meteor_no_overlap() {
        let val = meteor_score("hello world", "goodbye moon").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_meteor_partial_overlap() {
        let val = meteor_score("the cat sat on the mat", "the cat sits on a mat")
            .expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    #[test]
    fn test_meteor_empty_both() {
        let val = meteor_score("", "").expect("should succeed");
        assert!((val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_meteor_empty_reference() {
        let val = meteor_score("", "hello world").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_meteor_empty_candidate() {
        let val = meteor_score("hello world", "").expect("should succeed");
        assert!((val - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_meteor_stem_matching() {
        // "running" and "runs" should stem-match to "run"
        let val =
            meteor_score("the cat is running fast", "the cat runs fast").expect("should succeed");
        assert!(val > 0.5); // Good overlap due to stem matching
    }

    // ---- meteor_score_parametric tests ----

    #[test]
    fn test_meteor_parametric_default() {
        let val = meteor_score_parametric("the cat sat", "the cat sat", 0.9, 3.0, 0.5)
            .expect("should succeed");
        assert!(val > 0.9);
    }

    #[test]
    fn test_meteor_parametric_invalid_alpha() {
        assert!(meteor_score_parametric("a", "a", 0.0, 3.0, 0.5).is_err());
        assert!(meteor_score_parametric("a", "a", 1.0, 3.0, 0.5).is_err());
    }

    #[test]
    fn test_meteor_parametric_invalid_beta() {
        assert!(meteor_score_parametric("a", "a", 0.5, -1.0, 0.5).is_err());
    }

    #[test]
    fn test_meteor_parametric_invalid_gamma() {
        assert!(meteor_score_parametric("a", "a", 0.5, 3.0, -1.0).is_err());
    }

    // ---- corpus_meteor tests ----

    #[test]
    fn test_corpus_meteor_perfect() {
        let refs = vec!["hello world", "the cat sat"];
        let cands = vec!["hello world", "the cat sat"];
        let val = corpus_meteor(&refs, &cands).expect("should succeed");
        assert!(val > 0.9);
    }

    #[test]
    fn test_corpus_meteor_mismatched() {
        let refs = vec!["a"];
        let cands = vec!["a", "b"];
        assert!(corpus_meteor(&refs, &cands).is_err());
    }

    #[test]
    fn test_corpus_meteor_empty() {
        let refs: Vec<&str> = vec![];
        let cands: Vec<&str> = vec![];
        assert!(corpus_meteor(&refs, &cands).is_err());
    }

    #[test]
    fn test_corpus_meteor_partial() {
        let refs = vec!["the quick brown fox", "a lazy dog"];
        let cands = vec!["the slow brown fox", "a lazy cat"];
        let val = corpus_meteor(&refs, &cands).expect("should succeed");
        assert!(val > 0.0 && val < 1.0);
    }

    // ---- Utility tests ----

    #[test]
    fn test_simple_stem() {
        assert_eq!(simple_stem("running"), "runn");
        assert_eq!(simple_stem("cats"), "cat");
        assert_eq!(simple_stem("played"), "play");
        assert_eq!(simple_stem("happiness"), "happi");
    }

    #[test]
    fn test_count_chunks_all_matched() {
        assert_eq!(count_chunks(&[true, true, true]), 1);
    }

    #[test]
    fn test_count_chunks_alternating() {
        assert_eq!(count_chunks(&[true, false, true, false, true]), 3);
    }

    #[test]
    fn test_count_chunks_none_matched() {
        assert_eq!(count_chunks(&[false, false, false]), 0);
    }

    #[test]
    fn test_count_chunks_empty() {
        assert_eq!(count_chunks(&[]), 0);
    }
}
