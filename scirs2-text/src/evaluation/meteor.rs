//! # METEOR Score (Metric for Evaluation of Translation with Explicit ORdering)
//!
//! Implementation of the METEOR metric (Banerjee & Lavie 2005) for machine
//! translation evaluation. METEOR aligns hypothesis and reference words in
//! multiple stages, then computes a score based on unigram precision/recall
//! and a fragmentation penalty.
//!
//! ## Alignment Stages
//!
//! 1. **Exact match**: Case-insensitive exact word matching.
//! 2. **Stem match**: Simple suffix-stripping stemmer (Porter-like).
//! 3. **Approximate match**: Edit-distance-based matching for near-synonyms.
//!
//! ## Score Formula
//!
//! ```text
//! F_mean = (P * R) / (alpha * P + (1 - alpha) * R)
//! penalty = gamma * (chunks / matches) ^ beta
//! METEOR = F_mean * (1 - penalty)
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_text::evaluation::meteor::{meteor_score, MeteorConfig};
//!
//! let hypothesis = vec!["the", "cat", "sat", "on", "the", "mat"];
//! let reference = vec!["the", "cat", "is", "on", "the", "mat"];
//! let config = MeteorConfig::default();
//! let score = meteor_score(&hypothesis, &reference, &config).expect("Operation failed");
//! println!("METEOR: {:.4}", score.score);
//! ```

use crate::error::{Result, TextError};

/// METEOR score result with component details.
#[derive(Debug, Clone)]
pub struct MeteorScore {
    /// Final METEOR score.
    pub score: f64,
    /// Unigram precision.
    pub precision: f64,
    /// Unigram recall.
    pub recall: f64,
    /// F-mean (harmonic mean of P and R with alpha weighting).
    pub f_mean: f64,
    /// Fragmentation penalty.
    pub penalty: f64,
    /// Number of alignment chunks.
    pub chunks: usize,
    /// Number of matched unigrams.
    pub matches: usize,
}

/// Configuration parameters for METEOR scoring.
#[derive(Debug, Clone)]
pub struct MeteorConfig {
    /// Alpha parameter for F-mean weighting.
    /// Higher alpha gives more weight to recall.
    /// Default: 0.9 (recall-oriented).
    pub alpha: f64,
    /// Beta parameter for fragmentation penalty exponent.
    /// Default: 3.0.
    pub beta: f64,
    /// Gamma parameter for fragmentation penalty weight.
    /// Default: 0.5.
    pub gamma: f64,
    /// Whether to use stemming for matching (stage 2).
    /// Default: true.
    pub use_stemming: bool,
    /// Whether to use approximate matching via edit distance (stage 3).
    /// Default: true.
    pub use_approximate: bool,
    /// Maximum edit distance ratio for approximate matching.
    /// A pair matches if edit_distance / max_len <= threshold.
    /// Default: 0.4.
    pub approximate_threshold: f64,
}

impl Default for MeteorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.9,
            beta: 3.0,
            gamma: 0.5,
            use_stemming: true,
            use_approximate: true,
            approximate_threshold: 0.4,
        }
    }
}

/// Simple Porter-like stemmer that strips common English suffixes.
///
/// This is intentionally simplified -- a production system would use a
/// full Porter or Snowball stemmer. This strips common suffixes to enable
/// matching between inflected forms.
fn simple_stem(word: &str) -> String {
    let w = word.to_lowercase();
    let len = w.len();

    if len <= 3 {
        return w;
    }

    // Order matters: try longer suffixes first
    let suffixes = [
        "ational", "tional", "ences", "ances", "ments", "ously", "ively", "ation", "ness", "ment",
        "able", "ible", "ting", "ally", "ence", "ance", "ings", "ized", "ling", "ful", "ous",
        "ive", "ize", "ing", "ies", "ied", "ion", "ers", "est", "ess", "ism", "ist", "ity", "ble",
        "ful", "ous", "ent", "ant", "ary", "ery", "ory", "al", "ly", "er", "ed", "en", "es", "ty",
    ];

    for suffix in &suffixes {
        if w.ends_with(suffix) && len - suffix.len() >= 3 {
            return w[..len - suffix.len()].to_string();
        }
    }

    // Handle trailing 's' for plurals (but not too aggressively)
    if w.ends_with('s') && !w.ends_with("ss") && len >= 4 {
        return w[..len - 1].to_string();
    }

    w
}

/// Compute Levenshtein edit distance between two strings.
fn edit_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Alignment between a hypothesis token and a reference token.
#[derive(Debug, Clone)]
struct Alignment {
    /// Index in hypothesis.
    hyp_idx: usize,
    /// Index in reference.
    ref_idx: usize,
}

/// Build alignment between hypothesis and reference tokens.
///
/// Performs multi-stage greedy alignment:
/// 1. Exact match (case-insensitive)
/// 2. Stem match
/// 3. Approximate match (edit distance)
///
/// At each stage, unmatched tokens are considered for the next stage.
/// Among multiple possible matches, prefer the one that minimizes
/// crossing alignments (keeps order).
fn build_alignment(
    hypothesis: &[&str],
    reference: &[&str],
    config: &MeteorConfig,
) -> Vec<Alignment> {
    let hyp_lower: Vec<String> = hypothesis.iter().map(|w| w.to_lowercase()).collect();
    let ref_lower: Vec<String> = reference.iter().map(|w| w.to_lowercase()).collect();

    let mut hyp_matched = vec![false; hypothesis.len()];
    let mut ref_matched = vec![false; reference.len()];
    let mut alignments: Vec<Alignment> = Vec::new();

    // Stage 1: Exact match (case-insensitive)
    stage_match(
        &hyp_lower,
        &ref_lower,
        &mut hyp_matched,
        &mut ref_matched,
        &mut alignments,
        |h, r| h == r,
    );

    // Stage 2: Stem match
    if config.use_stemming {
        let hyp_stems: Vec<String> = hyp_lower.iter().map(|w| simple_stem(w)).collect();
        let ref_stems: Vec<String> = ref_lower.iter().map(|w| simple_stem(w)).collect();

        stage_match(
            &hyp_stems,
            &ref_stems,
            &mut hyp_matched,
            &mut ref_matched,
            &mut alignments,
            |h, r| h == r,
        );
    }

    // Stage 3: Approximate match (edit distance)
    if config.use_approximate {
        let threshold = config.approximate_threshold;
        stage_match(
            &hyp_lower,
            &ref_lower,
            &mut hyp_matched,
            &mut ref_matched,
            &mut alignments,
            |h, r| {
                let max_len = h.len().max(r.len());
                if max_len == 0 {
                    return true;
                }
                let dist = edit_distance(h, r);
                (dist as f64 / max_len as f64) <= threshold
            },
        );
    }

    alignments
}

/// Perform a single stage of greedy alignment.
///
/// For each unmatched hypothesis token, find the best unmatched reference token
/// that satisfies the match predicate. Among multiple matches, prefer the one
/// closest in position to reduce crossing.
fn stage_match<F>(
    hyp_forms: &[String],
    ref_forms: &[String],
    hyp_matched: &mut [bool],
    ref_matched: &mut [bool],
    alignments: &mut Vec<Alignment>,
    matches: F,
) where
    F: Fn(&str, &str) -> bool,
{
    for (h_idx, h_form) in hyp_forms.iter().enumerate() {
        if hyp_matched[h_idx] {
            continue;
        }

        let mut best_r_idx: Option<usize> = None;
        let mut best_dist = usize::MAX;

        for (r_idx, r_form) in ref_forms.iter().enumerate() {
            if ref_matched[r_idx] {
                continue;
            }
            if matches(h_form, r_form) {
                let dist = h_idx.abs_diff(r_idx);
                if dist < best_dist {
                    best_dist = dist;
                    best_r_idx = Some(r_idx);
                }
            }
        }

        if let Some(r_idx) = best_r_idx {
            hyp_matched[h_idx] = true;
            ref_matched[r_idx] = true;
            alignments.push(Alignment {
                hyp_idx: h_idx,
                ref_idx: r_idx,
            });
        }
    }
}

/// Count the number of chunks (contiguous aligned regions) in the alignment.
///
/// Alignments are sorted by hypothesis index, then we count how many
/// times the reference index is non-adjacent (i.e., not ref_prev + 1).
fn count_chunks(alignments: &[Alignment]) -> usize {
    if alignments.is_empty() {
        return 0;
    }

    let mut sorted = alignments.to_vec();
    sorted.sort_by_key(|a| a.hyp_idx);

    let mut chunks = 1usize;
    for i in 1..sorted.len() {
        // A new chunk starts if either the hypothesis or reference index
        // is not contiguous with the previous alignment
        let hyp_contiguous = sorted[i].hyp_idx == sorted[i - 1].hyp_idx + 1;
        let ref_contiguous = sorted[i].ref_idx == sorted[i - 1].ref_idx + 1;
        if !hyp_contiguous || !ref_contiguous {
            chunks += 1;
        }
    }

    chunks
}

/// Compute METEOR score for a hypothesis-reference pair.
///
/// # Arguments
///
/// * `hypothesis` - Generated text as a slice of tokens.
/// * `reference` - Reference text as a slice of tokens.
/// * `config` - METEOR configuration parameters.
///
/// # Returns
///
/// `MeteorScore` containing the final score and component details.
///
/// # Errors
///
/// Returns `TextError::InvalidInput` if alpha is outside (0, 1).
pub fn meteor_score(
    hypothesis: &[&str],
    reference: &[&str],
    config: &MeteorConfig,
) -> Result<MeteorScore> {
    if config.alpha <= 0.0 || config.alpha >= 1.0 {
        return Err(TextError::InvalidInput(format!(
            "Alpha must be in (0, 1), got {}",
            config.alpha
        )));
    }

    let hyp_len = hypothesis.len();
    let ref_len = reference.len();

    // Handle empty inputs
    if hyp_len == 0 && ref_len == 0 {
        return Ok(MeteorScore {
            score: 1.0,
            precision: 1.0,
            recall: 1.0,
            f_mean: 1.0,
            penalty: 0.0,
            chunks: 0,
            matches: 0,
        });
    }
    if hyp_len == 0 || ref_len == 0 {
        return Ok(MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            f_mean: 0.0,
            penalty: 0.0,
            chunks: 0,
            matches: 0,
        });
    }

    // Build alignment
    let alignments = build_alignment(hypothesis, reference, config);
    let matches = alignments.len();

    if matches == 0 {
        return Ok(MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            f_mean: 0.0,
            penalty: 0.0,
            chunks: 0,
            matches: 0,
        });
    }

    // Unigram precision and recall
    let precision = matches as f64 / hyp_len as f64;
    let recall = matches as f64 / ref_len as f64;

    // Weighted harmonic mean
    let alpha = config.alpha;
    let f_mean = (precision * recall) / (alpha * precision + (1.0 - alpha) * recall);

    // Fragmentation penalty
    let chunks = count_chunks(&alignments);
    let frag = chunks as f64 / matches as f64;
    let penalty = config.gamma * frag.powf(config.beta);

    // Clamp penalty to [0, 1]
    let penalty = penalty.clamp(0.0, 1.0);

    let score = f_mean * (1.0 - penalty);

    Ok(MeteorScore {
        score,
        precision,
        recall,
        f_mean,
        penalty,
        chunks,
        matches,
    })
}

/// Compute METEOR score with multiple references, returning the best score.
///
/// # Arguments
///
/// * `hypothesis` - Generated text tokens.
/// * `references` - Slice of reference texts.
/// * `config` - METEOR configuration.
///
/// # Errors
///
/// Returns error if references is empty or config is invalid.
pub fn meteor_score_multi(
    hypothesis: &[&str],
    references: &[Vec<&str>],
    config: &MeteorConfig,
) -> Result<MeteorScore> {
    if references.is_empty() {
        return Err(TextError::InvalidInput(
            "References must not be empty".to_string(),
        ));
    }

    let mut best: Option<MeteorScore> = None;
    for reference in references {
        let score = meteor_score(hypothesis, reference, config)?;
        if best.is_none() || score.score > best.as_ref().map_or(0.0, |b| b.score) {
            best = Some(score);
        }
    }

    // Safe because we checked references is not empty above
    best.ok_or_else(|| TextError::InvalidInput("No references provided".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_match_score() {
        let hypothesis = vec!["the", "cat", "is", "on", "the", "mat"];
        let reference = vec!["the", "cat", "is", "on", "the", "mat"];
        let config = MeteorConfig::default();
        let result = meteor_score(&hypothesis, &reference, &config).expect("should compute");

        // Perfect match: P=R=1.0, 1 chunk, penalty is small
        assert!(
            (result.precision - 1.0).abs() < 1e-9,
            "Precision should be 1.0"
        );
        assert!((result.recall - 1.0).abs() < 1e-9, "Recall should be 1.0");
        // 6 matches, 1 chunk => penalty = gamma * (1/6)^beta
        assert!(result.score > 0.9, "Perfect match should score high");
    }

    #[test]
    fn test_no_match_score() {
        let hypothesis = vec!["a", "b", "c"];
        let reference = vec!["x", "y", "z"];
        let config = MeteorConfig {
            use_approximate: false,
            ..Default::default()
        };
        let result = meteor_score(&hypothesis, &reference, &config).expect("should compute");
        assert!(result.score.abs() < 1e-9, "No match should score 0.0");
    }

    #[test]
    fn test_partial_match_with_stemming() {
        let hypothesis = vec!["the", "cats", "sitting", "on", "the", "mats"];
        let reference = vec!["the", "cat", "sat", "on", "the", "mat"];
        let config = MeteorConfig {
            use_stemming: true,
            use_approximate: false,
            ..Default::default()
        };
        let result = meteor_score(&hypothesis, &reference, &config).expect("should compute");

        // "the"(2), "on"(1) match exactly = 3
        // "cats"->"cat" and "cat"->"cat" match by stem, "mats"->"mat" and "mat"->"mat" match by stem
        // "sitting"->"sitt" vs "sat"->"sat" likely don't match
        assert!(
            result.matches >= 3,
            "Should have at least exact matches: got {}",
            result.matches
        );
        assert!(
            result.score > 0.0,
            "Partial match should give positive score"
        );
    }

    #[test]
    fn test_fragmentation_penalty() {
        // Aligned but out of order => multiple chunks => higher penalty
        let hypothesis = vec!["mat", "the", "on", "sat", "cat", "the"];
        let reference = vec!["the", "cat", "sat", "on", "the", "mat"];
        let config = MeteorConfig {
            use_stemming: false,
            use_approximate: false,
            ..Default::default()
        };
        let result = meteor_score(&hypothesis, &reference, &config).expect("should compute");

        // All words match but order is scrambled => many chunks
        assert!(result.chunks > 1, "Scrambled order should produce chunks");
        assert!(
            result.penalty > 0.0,
            "Should have fragmentation penalty: {}",
            result.penalty
        );
    }

    #[test]
    fn test_approximate_matching() {
        let hypothesis = vec!["colour", "neighbours"];
        let reference = vec!["color", "neighbors"];
        let config = MeteorConfig {
            use_stemming: false,
            use_approximate: true,
            approximate_threshold: 0.4,
            ..Default::default()
        };
        let result = meteor_score(&hypothesis, &reference, &config).expect("should compute");

        // "colour" vs "color": edit distance 1, max_len 6, ratio 0.167 < 0.4
        // "neighbours" vs "neighbors": edit distance 1, max_len 10, ratio 0.1 < 0.4
        assert_eq!(result.matches, 2, "Both should match approximately");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_invalid_alpha() {
        let result = meteor_score(
            &["a"],
            &["a"],
            &MeteorConfig {
                alpha: 0.0,
                ..Default::default()
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_reference() {
        let hypothesis = vec!["the", "cat", "sat"];
        let references = vec![vec!["a", "dog", "ran"], vec!["the", "cat", "sat"]];
        let config = MeteorConfig::default();
        let result = meteor_score_multi(&hypothesis, &references, &config).expect("should compute");
        assert!(
            result.score > 0.8,
            "Should match second reference well: {}",
            result.score
        );
    }

    #[test]
    fn test_simple_stem() {
        assert_eq!(simple_stem("running"), "runn");
        assert_eq!(simple_stem("cats"), "cat");
        assert_eq!(simple_stem("happiness"), "happi");
        // Short words should be returned as-is
        assert_eq!(simple_stem("the"), "the");
    }
}
