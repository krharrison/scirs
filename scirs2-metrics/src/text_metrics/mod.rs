//! Text / NLP Evaluation Metrics
//!
//! Standalone functions for evaluating text generation, machine translation,
//! and speech recognition systems.
//!
//! # Sub-modules
//!
//! - [`bleu`] -- BLEU score (1 through 4-gram)
//! - [`rouge`] -- ROUGE-1, ROUGE-2, ROUGE-L
//! - [`meteor`] -- METEOR score
//! - [`error_rate`] -- Character Error Rate (CER) and Word Error Rate (WER)
//!
//! # Quick Example
//!
//! ```
//! use scirs2_metrics::text_metrics::bleu::bleu_score;
//! use scirs2_metrics::text_metrics::rouge::{rouge_1, rouge_2, rouge_l};
//! use scirs2_metrics::text_metrics::meteor::meteor_score;
//! use scirs2_metrics::text_metrics::error_rate::{word_error_rate, character_error_rate};
//!
//! let reference = "the cat sat on the mat";
//! let candidate = "the cat sits on a mat";
//!
//! let bleu = bleu_score(reference, candidate, 4, true).expect("Failed");
//! let r1 = rouge_1(reference, candidate).expect("Failed");
//! let r2 = rouge_2(reference, candidate).expect("Failed");
//! let rl = rouge_l(reference, candidate).expect("Failed");
//! let met = meteor_score(reference, candidate).expect("Failed");
//! let wer = word_error_rate(reference, candidate).expect("Failed");
//! let cer = character_error_rate(reference, candidate).expect("Failed");
//! ```

pub mod bleu;
pub mod error_rate;
pub mod meteor;
pub mod rouge;

/// Simple whitespace tokenizer that lowercases and strips punctuation.
///
/// This is shared across sub-modules.
pub(crate) fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// Extract n-grams from a token sequence.
pub(crate) fn get_ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n || n == 0 {
        return vec![];
    }
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].to_vec())
        .collect()
}
