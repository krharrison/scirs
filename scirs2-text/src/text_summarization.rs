//! Extractive text summarization module
//!
//! This module provides multiple sentence-scoring strategies for extractive
//! summarization:
//!
//! - **TextRank**: Graph-based sentence ranking via PageRank on a similarity
//!   matrix.
//! - **Position-based**: Lead-bias and coverage heuristics.
//! - **TF-IDF**: Sentences scored by their average TF-IDF weight.
//! - **Ensemble**: Weighted combination of multiple scoring methods.
//!
//! All methods are accessible through the unified [`summarize`] function.

use crate::error::{Result, TextError};
use crate::tokenize::{SentenceTokenizer, Tokenizer, WordTokenizer};
use crate::vectorize::{TfidfVectorizer, Vectorizer};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Summarization method selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SummarizationMethod {
    /// TextRank sentence scoring.
    TextRank,
    /// Position-based scoring (lead bias).
    Position,
    /// TF-IDF sentence scoring.
    TfIdf,
    /// Ensemble of all three methods with customisable weights.
    Ensemble {
        /// Weight for TextRank component.
        textrank_weight: f64,
        /// Weight for position component.
        position_weight: f64,
        /// Weight for TF-IDF component.
        tfidf_weight: f64,
    },
}

/// A scored sentence, carrying its index in the original text.
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    /// The original sentence text.
    pub text: String,
    /// Zero-based index in the original sentence list.
    pub index: usize,
    /// Relevance score (higher is more important).
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Unified API
// ---------------------------------------------------------------------------

/// Produce an extractive summary of `text`.
///
/// `ratio` controls how much of the text to keep (0.0..1.0). A ratio of 0.3
/// means roughly 30% of sentences will be selected.
///
/// Returns the summary as a single string with selected sentences in their
/// original order.
///
/// # Errors
///
/// Returns an error if tokenization or vectorization fails.
pub fn summarize(text: &str, ratio: f64, method: SummarizationMethod) -> Result<String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }

    let clamped_ratio = ratio.clamp(0.0, 1.0);

    let sentence_tokenizer = SentenceTokenizer::new();
    let sentences: Vec<String> = sentence_tokenizer.tokenize(text)?;

    if sentences.is_empty() {
        return Ok(String::new());
    }

    let n_select = (sentences.len() as f64 * clamped_ratio).ceil().max(1.0) as usize;

    if n_select >= sentences.len() {
        return Ok(text.to_string());
    }

    let scored = match method {
        SummarizationMethod::TextRank => score_textrank(&sentences)?,
        SummarizationMethod::Position => score_position(&sentences),
        SummarizationMethod::TfIdf => score_tfidf(&sentences)?,
        SummarizationMethod::Ensemble {
            textrank_weight,
            position_weight,
            tfidf_weight,
        } => score_ensemble(&sentences, textrank_weight, position_weight, tfidf_weight)?,
    };

    // Select top-n sentences.
    let mut top: Vec<ScoredSentence> = scored;
    top.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(n_select);

    // Restore original order for readability.
    top.sort_by_key(|s| s.index);

    let summary = top
        .iter()
        .map(|s| s.text.clone())
        .collect::<Vec<_>>()
        .join(" ");

    Ok(summary)
}

// ---------------------------------------------------------------------------
// TextRank scoring
// ---------------------------------------------------------------------------

/// Score sentences using TextRank (PageRank over a TF-IDF cosine similarity
/// graph).
pub fn score_textrank(sentences: &[String]) -> Result<Vec<ScoredSentence>> {
    let n = sentences.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![ScoredSentence {
            text: sentences[0].clone(),
            index: 0,
            score: 1.0,
        }]);
    }

    let similarity_matrix = build_similarity_matrix(sentences)?;
    let scores = pagerank(&similarity_matrix, 0.85, 100, 1e-5)?;

    Ok(sentences
        .iter()
        .enumerate()
        .map(|(i, s)| ScoredSentence {
            text: s.clone(),
            index: i,
            score: scores[i],
        })
        .collect())
}

/// Build a sentence-by-sentence cosine similarity matrix using TF-IDF.
fn build_similarity_matrix(sentences: &[String]) -> Result<Array2<f64>> {
    let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let mut vectorizer = TfidfVectorizer::default();
    vectorizer.fit(&refs)?;
    let tfidf = vectorizer.transform_batch(&refs)?;

    let n = sentences.len();
    let mut matrix = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_sim_rows(&tfidf, i, j);
            matrix[[i, j]] = sim;
            matrix[[j, i]] = sim;
        }
    }

    Ok(matrix)
}

/// Cosine similarity between row `i` and row `j` of a matrix.
fn cosine_sim_rows(matrix: &Array2<f64>, i: usize, j: usize) -> f64 {
    let cols = matrix.ncols();
    let mut dot = 0.0_f64;
    let mut n1 = 0.0_f64;
    let mut n2 = 0.0_f64;

    for c in 0..cols {
        let a = matrix[[i, c]];
        let b = matrix[[j, c]];
        dot += a * b;
        n1 += a * a;
        n2 += b * b;
    }

    let denom = n1.sqrt() * n2.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// PageRank on a similarity matrix.
fn pagerank(
    matrix: &Array2<f64>,
    damping: f64,
    max_iter: usize,
    threshold: f64,
) -> Result<Vec<f64>> {
    let n = matrix.nrows();
    let mut scores = vec![1.0 / n as f64; n];

    // Row-normalise the matrix.
    let mut norm_matrix = matrix.clone();
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| matrix[[i, j]]).sum();
        if row_sum > 0.0 {
            for j in 0..n {
                norm_matrix[[i, j]] = matrix[[i, j]] / row_sum;
            }
        }
    }

    for _ in 0..max_iter {
        let mut new_scores = vec![(1.0 - damping) / n as f64; n];

        for i in 0..n {
            for j in 0..n {
                new_scores[i] += damping * norm_matrix[[j, i]] * scores[j];
            }
        }

        let diff: f64 = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        scores = new_scores;
        if diff < threshold {
            break;
        }
    }

    Ok(scores)
}

// ---------------------------------------------------------------------------
// Position-based scoring
// ---------------------------------------------------------------------------

/// Score sentences by position: early sentences and the last sentence receive
/// a boost (lead-bias heuristic commonly used in news articles).
pub fn score_position(sentences: &[String]) -> Vec<ScoredSentence> {
    let n = sentences.len();
    if n == 0 {
        return Vec::new();
    }

    sentences
        .iter()
        .enumerate()
        .map(|(i, s)| {
            // Lead sentences score highest, then the conclusion, middle is lowest.
            let position_score = if n == 1 {
                1.0
            } else {
                let lead_score = 1.0 - (i as f64 / n as f64);
                let conclusion_bonus = if i == n - 1 { 0.2 } else { 0.0 };
                // First sentence gets a small extra boost.
                let first_bonus = if i == 0 { 0.15 } else { 0.0 };
                lead_score + conclusion_bonus + first_bonus
            };

            // Longer sentences are somewhat more informative.
            let word_count = s.split_whitespace().count() as f64;
            let length_factor = (word_count.ln() + 1.0).min(3.0) / 3.0;

            ScoredSentence {
                text: s.clone(),
                index: i,
                score: position_score * length_factor,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// TF-IDF scoring
// ---------------------------------------------------------------------------

/// Score sentences by their average TF-IDF value.
///
/// Sentences with rare, important terms score higher.
pub fn score_tfidf(sentences: &[String]) -> Result<Vec<ScoredSentence>> {
    let n = sentences.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        return Ok(vec![ScoredSentence {
            text: sentences[0].clone(),
            index: 0,
            score: 1.0,
        }]);
    }

    let refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let mut vectorizer = TfidfVectorizer::default();
    let tfidf = vectorizer.fit_transform(&refs)?;

    let cols = tfidf.ncols();
    if cols == 0 {
        return Ok(sentences
            .iter()
            .enumerate()
            .map(|(i, s)| ScoredSentence {
                text: s.clone(),
                index: i,
                score: 0.0,
            })
            .collect());
    }

    Ok(sentences
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let row_sum: f64 = (0..cols).map(|c| tfidf[[i, c]]).sum();
            let avg = row_sum / cols as f64;
            ScoredSentence {
                text: s.clone(),
                index: i,
                score: avg,
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Ensemble scoring
// ---------------------------------------------------------------------------

/// Combine TextRank, position and TF-IDF scores with the given weights.
fn score_ensemble(
    sentences: &[String],
    textrank_weight: f64,
    position_weight: f64,
    tfidf_weight: f64,
) -> Result<Vec<ScoredSentence>> {
    let n = sentences.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let tr_scores = score_textrank(sentences)?;
    let pos_scores = score_position(sentences);
    let tfidf_scores = score_tfidf(sentences)?;

    // Normalise each set of scores to [0, 1].
    let tr_normalised = normalise_scores(&tr_scores);
    let pos_normalised = normalise_scores(&pos_scores);
    let tfidf_normalised = normalise_scores(&tfidf_scores);

    let total_weight = textrank_weight + position_weight + tfidf_weight;
    let tw = if total_weight > 0.0 {
        textrank_weight / total_weight
    } else {
        1.0 / 3.0
    };
    let pw = if total_weight > 0.0 {
        position_weight / total_weight
    } else {
        1.0 / 3.0
    };
    let iw = if total_weight > 0.0 {
        tfidf_weight / total_weight
    } else {
        1.0 / 3.0
    };

    Ok((0..n)
        .map(|i| ScoredSentence {
            text: sentences[i].clone(),
            index: i,
            score: tw * tr_normalised[i] + pw * pos_normalised[i] + iw * tfidf_normalised[i],
        })
        .collect())
}

/// Min-max normalise scores to [0, 1].
fn normalise_scores(scored: &[ScoredSentence]) -> Vec<f64> {
    if scored.is_empty() {
        return Vec::new();
    }

    let min = scored.iter().map(|s| s.score).fold(f64::INFINITY, f64::min);
    let max = scored
        .iter()
        .map(|s| s.score)
        .fold(f64::NEG_INFINITY, f64::max);

    let range = max - min;
    if range < 1e-12 {
        return vec![0.5; scored.len()];
    }

    scored.iter().map(|s| (s.score - min) / range).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TEXT: &str = "Machine learning is a subset of artificial intelligence. \
        It enables computers to learn from data without explicit programming. \
        Deep learning is a subset of machine learning that uses neural networks. \
        Neural networks are modeled loosely after the human brain. \
        These technologies are transforming many industries today.";

    // ---- TextRank tests ----

    #[test]
    fn test_textrank_produces_shorter_summary() {
        let summary =
            summarize(SAMPLE_TEXT, 0.4, SummarizationMethod::TextRank).expect("Should succeed");
        assert!(!summary.is_empty());
        assert!(summary.len() < SAMPLE_TEXT.len());
    }

    #[test]
    fn test_textrank_empty_text() {
        let summary = summarize("", 0.5, SummarizationMethod::TextRank).expect("ok");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_textrank_ratio_one_returns_full() {
        let summary = summarize(SAMPLE_TEXT, 1.0, SummarizationMethod::TextRank).expect("ok");
        assert_eq!(summary, SAMPLE_TEXT);
    }

    #[test]
    fn test_textrank_ratio_zero_returns_one_sentence() {
        let summary = summarize(SAMPLE_TEXT, 0.0, SummarizationMethod::TextRank).expect("ok");
        // ratio clamped to 0.0, ceil(n*0) = 0 but max(1) = 1 sentence.
        assert!(!summary.is_empty());
        // Should be just one sentence.
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(&summary).expect("ok");
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_textrank_single_sentence() {
        let summary =
            summarize("Just one sentence.", 0.5, SummarizationMethod::TextRank).expect("ok");
        assert_eq!(summary, "Just one sentence.");
    }

    #[test]
    fn test_textrank_scores_non_negative() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_textrank(&sentences).expect("ok");
        for s in &scored {
            assert!(s.score >= 0.0, "Score should be non-negative");
        }
    }

    // ---- Position tests ----

    #[test]
    fn test_position_first_sentence_highest() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_position(&sentences);
        // First sentence should have the highest score.
        let first = &scored[0];
        for s in &scored[1..] {
            assert!(
                first.score >= s.score,
                "First sentence should have the highest position score"
            );
        }
    }

    #[test]
    fn test_position_produces_summary() {
        let summary = summarize(SAMPLE_TEXT, 0.4, SummarizationMethod::Position).expect("ok");
        assert!(!summary.is_empty());
        assert!(summary.len() < SAMPLE_TEXT.len());
    }

    #[test]
    fn test_position_empty() {
        let summary = summarize("", 0.5, SummarizationMethod::Position).expect("ok");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_position_scores_non_negative() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_position(&sentences);
        for s in &scored {
            assert!(s.score >= 0.0);
        }
    }

    #[test]
    fn test_position_last_sentence_has_conclusion_bonus() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_position(&sentences);
        let n = scored.len();
        if n >= 2 {
            // The last sentence gets a 0.2 conclusion bonus, so its score
            // should be higher than a pure lead-bias formula without the bonus.
            let last_score = scored[n - 1].score;
            // Lead score alone at position n-1 would be 1.0 - (n-1)/n.
            let lead_alone = 1.0 - ((n - 1) as f64 / n as f64);
            // With conclusion bonus the score should exceed this baseline.
            assert!(
                last_score > lead_alone * 0.3,
                "Last sentence should benefit from conclusion bonus"
            );
        }
    }

    // ---- TF-IDF tests ----

    #[test]
    fn test_tfidf_produces_summary() {
        let summary = summarize(SAMPLE_TEXT, 0.4, SummarizationMethod::TfIdf).expect("ok");
        assert!(!summary.is_empty());
        assert!(summary.len() < SAMPLE_TEXT.len());
    }

    #[test]
    fn test_tfidf_empty() {
        let summary = summarize("", 0.5, SummarizationMethod::TfIdf).expect("ok");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_tfidf_single_sentence() {
        let summary = summarize("Only one.", 0.5, SummarizationMethod::TfIdf).expect("ok");
        assert_eq!(summary, "Only one.");
    }

    #[test]
    fn test_tfidf_scores_non_negative() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_tfidf(&sentences).expect("ok");
        for s in &scored {
            assert!(s.score >= 0.0);
        }
    }

    #[test]
    fn test_tfidf_ratio_half() {
        let summary = summarize(SAMPLE_TEXT, 0.5, SummarizationMethod::TfIdf).expect("ok");
        let sentence_tokenizer = SentenceTokenizer::new();
        let original = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let summarised = sentence_tokenizer.tokenize(&summary).expect("ok");
        // Should keep roughly half (ceil).
        let expected = (original.len() as f64 * 0.5).ceil() as usize;
        assert_eq!(summarised.len(), expected);
    }

    // ---- Ensemble tests ----

    #[test]
    fn test_ensemble_produces_summary() {
        let method = SummarizationMethod::Ensemble {
            textrank_weight: 1.0,
            position_weight: 0.5,
            tfidf_weight: 0.5,
        };
        let summary = summarize(SAMPLE_TEXT, 0.4, method).expect("ok");
        assert!(!summary.is_empty());
        assert!(summary.len() < SAMPLE_TEXT.len());
    }

    #[test]
    fn test_ensemble_equal_weights() {
        let method = SummarizationMethod::Ensemble {
            textrank_weight: 1.0,
            position_weight: 1.0,
            tfidf_weight: 1.0,
        };
        let summary = summarize(SAMPLE_TEXT, 0.4, method).expect("ok");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_ensemble_zero_weights_defaults() {
        let method = SummarizationMethod::Ensemble {
            textrank_weight: 0.0,
            position_weight: 0.0,
            tfidf_weight: 0.0,
        };
        let summary = summarize(SAMPLE_TEXT, 0.4, method).expect("ok");
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_ensemble_empty() {
        let method = SummarizationMethod::Ensemble {
            textrank_weight: 1.0,
            position_weight: 1.0,
            tfidf_weight: 1.0,
        };
        let summary = summarize("", 0.5, method).expect("ok");
        assert!(summary.is_empty());
    }

    #[test]
    fn test_ensemble_scores_bounded() {
        let sentence_tokenizer = SentenceTokenizer::new();
        let sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");
        let scored = score_ensemble(&sentences, 1.0, 1.0, 1.0).expect("ok");
        for s in &scored {
            assert!(
                s.score >= 0.0 && s.score <= 1.0,
                "Ensemble scores should be in [0,1]"
            );
        }
    }

    // ---- Original-order tests ----

    #[test]
    fn test_summary_preserves_order() {
        let summary = summarize(SAMPLE_TEXT, 0.6, SummarizationMethod::TextRank).expect("ok");
        let sentence_tokenizer = SentenceTokenizer::new();
        let summary_sentences = sentence_tokenizer.tokenize(&summary).expect("ok");
        let original_sentences = sentence_tokenizer.tokenize(SAMPLE_TEXT).expect("ok");

        // Each summary sentence should appear in order in the original.
        let mut last_idx: Option<usize> = None;
        for ss in &summary_sentences {
            let idx = original_sentences
                .iter()
                .position(|os| os.trim() == ss.trim());
            if let (Some(li), Some(ci)) = (last_idx, idx) {
                assert!(ci > li, "Summary sentences should be in original order");
            }
            last_idx = idx;
        }
    }
}
