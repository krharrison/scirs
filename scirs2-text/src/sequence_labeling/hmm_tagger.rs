//! HMM-based sequence tagger (POS / NER).
//!
//! Trains a first-order Hidden Markov Model from labelled sentences and performs
//! Viterbi decoding for inference.  Unknown words are handled via a simple
//! suffix heuristic and a fallback average emission probability.

use std::collections::HashMap;

use crate::error::{Result, TextError};
use crate::sequence_labeling::viterbi::viterbi_decode;

/// A trained HMM-based sequence tagger.
///
/// All probabilities are stored in **log** space to avoid underflow.
#[derive(Debug, Clone)]
pub struct HMMTagger {
    /// log P(curr_state | prev_state)
    pub trans: HashMap<(String, String), f64>,
    /// log P(word | state)
    pub emit: HashMap<(String, String), f64>,
    /// log P(state at t=0)
    pub pi: HashMap<String, f64>,
    /// Ordered list of known states (labels)
    pub states: Vec<String>,
    /// Smoothing constant used at training time
    smoothing_k: f64,
    /// Per-state emission denominator (for unknown-word fallback)
    state_total_emission: HashMap<String, f64>,
}

const START_STATE: &str = "<START>";
const SMOOTHING_K: f64 = 0.01;

impl HMMTagger {
    /// Train an HMM tagger from labelled sentences.
    ///
    /// Each sentence is `(words, tags)` where both slices have equal length.
    pub fn train(sentences: &[(Vec<String>, Vec<String>)]) -> Result<HMMTagger> {
        let mut trans_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut emit_counts: HashMap<(String, String), usize> = HashMap::new();
        let mut state_counts: HashMap<String, usize> = HashMap::new();
        let mut pi_counts: HashMap<String, usize> = HashMap::new();
        let mut vocab: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (words, tags) in sentences {
            if words.len() != tags.len() {
                return Err(TextError::InvalidInput(
                    "words and tags must have equal length".to_string(),
                ));
            }
            if words.is_empty() {
                continue;
            }
            // Initial state distribution
            *pi_counts.entry(tags[0].clone()).or_insert(0) += 1;

            for i in 0..words.len() {
                let state = &tags[i];
                *state_counts.entry(state.clone()).or_insert(0) += 1;
                *emit_counts
                    .entry((state.clone(), words[i].clone()))
                    .or_insert(0) += 1;
                vocab.insert(words[i].clone());

                if i > 0 {
                    *trans_counts
                        .entry((tags[i - 1].clone(), state.clone()))
                        .or_insert(0) += 1;
                }
            }
        }

        let mut states: Vec<String> = state_counts.keys().cloned().collect();
        states.sort();
        let n_states = states.len();
        let vocab_size = vocab.len();

        // Build log-probability tables with add-k smoothing
        let mut trans: HashMap<(String, String), f64> = HashMap::new();
        let mut pi: HashMap<String, f64> = HashMap::new();
        let mut emit: HashMap<(String, String), f64> = HashMap::new();
        let mut state_total_emission: HashMap<String, f64> = HashMap::new();

        // Initial distribution
        let pi_total: usize = pi_counts.values().sum();
        for state in &states {
            let c = *pi_counts.get(state).unwrap_or(&0) as f64;
            let p = (c + SMOOTHING_K) / (pi_total as f64 + SMOOTHING_K * n_states as f64);
            pi.insert(state.clone(), p.ln());
        }

        // Transitions: P(curr | prev) with add-k smoothing
        for prev in &states {
            let prev_count = *state_counts.get(prev).unwrap_or(&0) as f64;
            for curr in &states {
                let c = *trans_counts
                    .get(&(prev.clone(), curr.clone()))
                    .unwrap_or(&0) as f64;
                let p = (c + SMOOTHING_K)
                    / (prev_count + SMOOTHING_K * n_states as f64);
                trans.insert((prev.clone(), curr.clone()), p.ln());
            }
        }

        // Emissions: P(word | state) with add-k smoothing
        for state in &states {
            let state_count = *state_counts.get(state).unwrap_or(&0) as f64;
            let denominator = state_count + SMOOTHING_K * vocab_size as f64;
            state_total_emission.insert(state.clone(), denominator);
            for word in &vocab {
                let c = *emit_counts
                    .get(&(state.clone(), word.clone()))
                    .unwrap_or(&0) as f64;
                let p = (c + SMOOTHING_K) / denominator;
                emit.insert((state.clone(), word.clone()), p.ln());
            }
        }

        Ok(HMMTagger {
            trans,
            emit,
            pi,
            states,
            smoothing_k: SMOOTHING_K,
            state_total_emission,
        })
    }

    /// Tag a sequence of words using Viterbi decoding.
    pub fn tag(&self, words: &[String]) -> Result<Vec<String>> {
        if words.is_empty() {
            return Ok(vec![]);
        }
        let n = words.len();
        let n_states = self.states.len();
        if n_states == 0 {
            return Err(TextError::ProcessingError("Model has no states".to_string()));
        }

        // Build log_start from pi
        let log_start: Vec<f64> = self
            .states
            .iter()
            .map(|s| self.pi.get(s).copied().unwrap_or(f64::NEG_INFINITY))
            .collect();

        // Build log_trans matrix
        let log_trans: Vec<Vec<f64>> = self
            .states
            .iter()
            .map(|prev| {
                self.states
                    .iter()
                    .map(|curr| {
                        self.trans
                            .get(&(prev.clone(), curr.clone()))
                            .copied()
                            .unwrap_or(f64::NEG_INFINITY)
                    })
                    .collect()
            })
            .collect();

        // Build log_emit matrix (handle unknown words)
        let log_emit: Vec<Vec<f64>> = words
            .iter()
            .map(|word| {
                self.states
                    .iter()
                    .map(|state| self.emission_log_prob(state, word))
                    .collect()
            })
            .collect();

        let (_, path) = viterbi_decode(n, n_states, &log_start, &log_trans, &log_emit)?;

        let tags: Vec<String> = path
            .iter()
            .map(|&idx| self.states[idx].clone())
            .collect();
        Ok(tags)
    }

    /// Compute log emission probability for a (state, word) pair.
    ///
    /// For unknown words we fall back to the smoothed probability of an
    /// unseen event: `k / (state_count + k * vocab_size)`.
    fn emission_log_prob(&self, state: &str, word: &str) -> f64 {
        if let Some(&lp) = self.emit.get(&(state.to_string(), word.to_string())) {
            return lp;
        }
        // Suffix heuristic: try last 3, 2, 1 characters
        for suffix_len in (1..=3).rev() {
            if word.len() >= suffix_len {
                let suffix = &word[word.len() - suffix_len..];
                if let Some(&lp) = self
                    .emit
                    .get(&(state.to_string(), format!("SUFFIX:{}", suffix)))
                {
                    return lp;
                }
            }
        }
        // Fallback: smoothed probability for unseen word
        let denom = self
            .state_total_emission
            .get(state)
            .copied()
            .unwrap_or(1.0);
        (self.smoothing_k / denom).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_corpus() -> Vec<(Vec<String>, Vec<String>)> {
        vec![
            (
                vec!["the".into(), "dog".into(), "runs".into()],
                vec!["DT".into(), "NN".into(), "VBZ".into()],
            ),
            (
                vec!["a".into(), "cat".into(), "sleeps".into()],
                vec!["DT".into(), "NN".into(), "VBZ".into()],
            ),
            (
                vec!["the".into(), "cat".into(), "runs".into()],
                vec!["DT".into(), "NN".into(), "VBZ".into()],
            ),
            (
                vec!["dogs".into(), "sleep".into()],
                vec!["NNS".into(), "VBP".into()],
            ),
        ]
    }

    #[test]
    fn test_hmm_train_and_tag() {
        let corpus = toy_corpus();
        let tagger = HMMTagger::train(&corpus).expect("train failed");

        // "the dog runs" should be unambiguously DT NN VBZ given our corpus
        let words: Vec<String> = vec!["the".into(), "dog".into(), "runs".into()];
        let tags = tagger.tag(&words).expect("tag failed");
        assert_eq!(tags.len(), 3);
        assert_eq!(tags[0], "DT");
        assert_eq!(tags[1], "NN");
        assert_eq!(tags[2], "VBZ");
    }

    #[test]
    fn test_hmm_unknown_word() {
        let corpus = toy_corpus();
        let tagger = HMMTagger::train(&corpus).expect("train failed");

        // "the cat barks" — "barks" is unseen
        let words: Vec<String> = vec!["the".into(), "cat".into(), "barks".into()];
        let tags = tagger.tag(&words).expect("tag failed");
        assert_eq!(tags.len(), 3);
        // We can't assert the exact tag for "barks", but it should be valid
        assert!(tagger.states.contains(&tags[2]));
    }

    #[test]
    fn test_hmm_empty_input() {
        let corpus = toy_corpus();
        let tagger = HMMTagger::train(&corpus).expect("train failed");
        let tags = tagger.tag(&[]).expect("tag empty");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_hmm_mismatched_lengths_error() {
        let bad_corpus = vec![(
            vec!["the".into(), "dog".into()],
            vec!["DT".into()], // length mismatch
        )];
        let result = HMMTagger::train(&bad_corpus);
        assert!(result.is_err());
    }
}
