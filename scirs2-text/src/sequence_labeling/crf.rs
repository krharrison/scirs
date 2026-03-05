//! Linear-chain Conditional Random Field (CRF) for sequence labeling.
//!
//! Implements:
//! - Feature functions (word, transition, bigram)
//! - Forward-backward algorithm for computing the partition function and
//!   marginal probabilities
//! - Viterbi decoding for inference
//! - SGD training with gradient computed from forward-backward marginals

use std::collections::{HashMap, HashSet};

use crate::error::{Result, TextError};

/// A single CRF feature with its associated label and weight.
#[derive(Debug, Clone)]
pub enum CRFFeature {
    /// Emission feature: a word-label pair
    WordFeature {
        /// The word token that fires this feature.
        word: String,
        /// The label assigned when this feature is active.
        label: String,
        /// Learned weight for this feature.
        weight: f64,
    },
    /// Transition feature: a label-to-label bigram
    TransitionFeature {
        /// Label at the previous position.
        prev_label: String,
        /// Label at the current position.
        curr_label: String,
        /// Learned weight for this feature.
        weight: f64,
    },
    /// Bigram context feature: previous word + current word + label
    BigramFeature {
        /// Current word token.
        word: String,
        /// Previous word token.
        prev_word: String,
        /// Label assigned when this feature is active.
        label: String,
        /// Learned weight for this feature.
        weight: f64,
    },
}

impl CRFFeature {
    /// Return the feature key used to index into the weight map.
    pub fn key(&self) -> String {
        match self {
            CRFFeature::WordFeature { word, label, .. } => {
                format!("WORD:{}:LABEL:{}", word, label)
            }
            CRFFeature::TransitionFeature {
                prev_label,
                curr_label,
                ..
            } => {
                format!("TRANS:{}:{}", prev_label, curr_label)
            }
            CRFFeature::BigramFeature {
                word,
                prev_word,
                label,
                ..
            } => {
                format!("BIGRAM:{}:{}:LABEL:{}", prev_word, word, label)
            }
        }
    }
}

/// A linear-chain CRF model.
#[derive(Debug, Clone)]
pub struct LinearChainCRF {
    /// Learned feature weights indexed by feature key.
    pub feature_weights: HashMap<String, f64>,
    /// Ordered list of labels.
    pub labels: Vec<String>,
}

impl LinearChainCRF {
    /// Construct a new (untrained) CRF with the given label set.
    pub fn new(labels: Vec<String>) -> Self {
        LinearChainCRF {
            feature_weights: HashMap::new(),
            labels,
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Collect all active features for a given position, previous label, and current label.
    fn active_features(
        &self,
        features: &[Vec<String>],
        t: usize,
        prev_label: Option<&str>,
        curr_label: &str,
    ) -> Vec<String> {
        let mut keys = Vec::new();
        for f in &features[t] {
            // word-label feature
            keys.push(format!("WORD:{}:LABEL:{}", f, curr_label));
            // bigram context feature
            if t > 0 {
                for prev_f in &features[t - 1] {
                    keys.push(format!("BIGRAM:{}:{}:LABEL:{}", prev_f, f, curr_label));
                }
            }
        }
        // transition feature
        if let Some(prev) = prev_label {
            keys.push(format!("TRANS:{}:{}", prev, curr_label));
        }
        keys
    }

    /// Compute the score (un-normalised log-potential) for a given transition.
    fn score(&self, feature_keys: &[String]) -> f64 {
        feature_keys
            .iter()
            .map(|k| self.feature_weights.get(k).copied().unwrap_or(0.0))
            .sum()
    }

    /// Compute per-position, per-label score matrix (log-potentials).
    ///
    /// Returns `scores[t][curr][prev]` where prev=-1 means "no previous label".
    fn compute_scores(
        &self,
        features: &[Vec<String>],
    ) -> Vec<Vec<Vec<f64>>> {
        let t_len = features.len();
        let n_labels = self.labels.len();
        let mut scores = vec![vec![vec![0.0f64; n_labels + 1]; n_labels]; t_len];

        for t in 0..t_len {
            for (j, curr_label) in self.labels.iter().enumerate() {
                // prev = n_labels means "no previous" (t == 0)
                let keys_no_prev =
                    self.active_features(features, t, None, curr_label);
                scores[t][j][n_labels] = self.score(&keys_no_prev);

                for (i, prev_label) in self.labels.iter().enumerate() {
                    let keys =
                        self.active_features(features, t, Some(prev_label), curr_label);
                    scores[t][j][i] = self.score(&keys);
                }
            }
        }
        scores
    }

    // -----------------------------------------------------------------------
    // Forward algorithm
    // -----------------------------------------------------------------------

    /// Compute forward variables and log partition function.
    ///
    /// Returns `(log_Z, alpha)` where `alpha[t][j]` is the log forward
    /// probability of being in label `j` at time `t`.
    pub fn forward_algorithm(
        &self,
        features: &[Vec<String>],
    ) -> Result<(f64, Vec<Vec<f64>>)> {
        let t_len = features.len();
        if t_len == 0 {
            return Ok((0.0, vec![]));
        }
        let n_labels = self.labels.len();
        if n_labels == 0 {
            return Err(TextError::InvalidInput("CRF has no labels".to_string()));
        }

        let scores = self.compute_scores(features);
        let mut alpha = vec![vec![f64::NEG_INFINITY; n_labels]; t_len];

        // t = 0: no previous label
        for j in 0..n_labels {
            alpha[0][j] = scores[0][j][n_labels]; // "no-prev" slot
        }

        // t > 0
        for t in 1..t_len {
            for j in 0..n_labels {
                let log_sum = log_sum_exp(
                    (0..n_labels)
                        .map(|i| alpha[t - 1][i] + scores[t][j][i])
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                alpha[t][j] = log_sum;
            }
        }

        // log Z = log sum_j alpha[T-1][j]
        let log_z = log_sum_exp(&alpha[t_len - 1]);
        Ok((log_z, alpha))
    }

    // -----------------------------------------------------------------------
    // Backward algorithm
    // -----------------------------------------------------------------------

    /// Compute backward variables.
    ///
    /// Returns `beta[t][i]` = log probability of the partial sequence from
    /// `t+1` to the end, starting from state `i` at time `t`.
    pub fn backward_algorithm(&self, features: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        let t_len = features.len();
        if t_len == 0 {
            return Ok(vec![]);
        }
        let n_labels = self.labels.len();
        if n_labels == 0 {
            return Err(TextError::InvalidInput("CRF has no labels".to_string()));
        }

        let scores = self.compute_scores(features);
        let mut beta = vec![vec![f64::NEG_INFINITY; n_labels]; t_len];

        // t = T-1: terminal condition
        for i in 0..n_labels {
            beta[t_len - 1][i] = 0.0; // log(1)
        }

        // t < T-1 (going backwards)
        for t in (0..t_len - 1).rev() {
            for i in 0..n_labels {
                let log_sum = log_sum_exp(
                    (0..n_labels)
                        .map(|j| beta[t + 1][j] + scores[t + 1][j][i])
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                beta[t][i] = log_sum;
            }
        }

        Ok(beta)
    }

    // -----------------------------------------------------------------------
    // Viterbi decoding
    // -----------------------------------------------------------------------

    /// Viterbi decoding: return the most probable label sequence.
    pub fn viterbi(&self, features: &[Vec<String>]) -> Result<Vec<String>> {
        let t_len = features.len();
        if t_len == 0 {
            return Ok(vec![]);
        }
        let n_labels = self.labels.len();
        if n_labels == 0 {
            return Err(TextError::InvalidInput("CRF has no labels".to_string()));
        }

        let scores = self.compute_scores(features);
        let mut dp = vec![vec![f64::NEG_INFINITY; n_labels]; t_len];
        let mut bp = vec![vec![0usize; n_labels]; t_len];

        // Initialise
        for j in 0..n_labels {
            dp[0][j] = scores[0][j][n_labels];
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..n_labels {
                let mut best_score = f64::NEG_INFINITY;
                let mut best_prev = 0usize;
                for i in 0..n_labels {
                    let s = dp[t - 1][i] + scores[t][j][i];
                    if s > best_score {
                        best_score = s;
                        best_prev = i;
                    }
                }
                dp[t][j] = best_score;
                bp[t][j] = best_prev;
            }
        }

        // Find best final state
        let mut best_final = 0usize;
        let mut best_final_score = f64::NEG_INFINITY;
        for j in 0..n_labels {
            if dp[t_len - 1][j] > best_final_score {
                best_final_score = dp[t_len - 1][j];
                best_final = j;
            }
        }

        // Back-trace
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (1..t_len).rev() {
            path[t - 1] = bp[t][path[t]];
        }

        Ok(path.iter().map(|&i| self.labels[i].clone()).collect())
    }

    // -----------------------------------------------------------------------
    // Training
    // -----------------------------------------------------------------------

    /// Train the CRF with SGD.
    ///
    /// # Arguments
    /// * `sequences` – labelled training sequences: `(features, labels)`
    ///   where `features[t]` is the list of active feature strings at position `t`.
    /// * `n_iter`    – number of passes over the training data
    /// * `lr`        – learning rate
    pub fn fit(
        sequences: &[(Vec<Vec<String>>, Vec<String>)],
        n_iter: usize,
        lr: f64,
    ) -> Result<LinearChainCRF> {
        // Collect label set
        let mut label_set: HashSet<String> = HashSet::new();
        for (_, labels) in sequences {
            for l in labels {
                label_set.insert(l.clone());
            }
        }
        let mut label_list: Vec<String> = label_set.into_iter().collect();
        label_list.sort();

        let mut crf = LinearChainCRF::new(label_list);

        for _iter in 0..n_iter {
            for (features, true_labels) in sequences {
                if features.is_empty() {
                    continue;
                }
                if features.len() != true_labels.len() {
                    return Err(TextError::InvalidInput(
                        "features and labels must have equal length".to_string(),
                    ));
                }
                // Compute gradient: empirical - expected
                let mut gradient: HashMap<String, f64> = HashMap::new();

                // --- Empirical counts from true labels ---
                let t_len = features.len();
                for t in 0..t_len {
                    let curr = &true_labels[t];
                    let prev = if t > 0 {
                        Some(true_labels[t - 1].as_str())
                    } else {
                        None
                    };
                    let active = crf.active_features(features, t, prev, curr);
                    for k in active {
                        *gradient.entry(k).or_insert(0.0) += 1.0;
                    }
                }

                // --- Expected counts via forward-backward ---
                let (log_z, alpha) = crf.forward_algorithm(features)?;
                let beta = crf.backward_algorithm(features)?;
                let scores = crf.compute_scores(features);
                let n_labels = crf.labels.len();

                // Unigram marginals P(y_t = j | x)
                for t in 0..t_len {
                    for j in 0..n_labels {
                        let log_marg = alpha[t][j] + beta[t][j] - log_z;
                        let marg = log_marg.exp();
                        // Collect features for this (t, None→j) position
                        let active =
                            crf.active_features(features, t, None, &crf.labels[j].clone());
                        for k in active {
                            *gradient.entry(k).or_insert(0.0) -= marg;
                        }
                    }
                }

                // Pairwise marginals P(y_{t-1}=i, y_t=j | x)
                for t in 1..t_len {
                    for i in 0..n_labels {
                        for j in 0..n_labels {
                            let log_marg =
                                alpha[t - 1][i] + scores[t][j][i] + beta[t][j] - log_z;
                            let marg = log_marg.exp();
                            let prev_label = crf.labels[i].clone();
                            let curr_label = crf.labels[j].clone();
                            let active = crf.active_features(
                                features,
                                t,
                                Some(&prev_label),
                                &curr_label,
                            );
                            for k in active {
                                *gradient.entry(k).or_insert(0.0) -= marg;
                            }
                        }
                    }
                }

                // SGD update
                for (k, g) in &gradient {
                    let w = crf.feature_weights.entry(k.clone()).or_insert(0.0);
                    *w += lr * g;
                }
            }
        }

        Ok(crf)
    }
}

/// Numerically stable log-sum-exp over a slice.
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_sequences() -> Vec<(Vec<Vec<String>>, Vec<String>)> {
        vec![
            (
                vec![
                    vec!["the".to_string()],
                    vec!["dog".to_string()],
                    vec!["runs".to_string()],
                ],
                vec!["DT".to_string(), "NN".to_string(), "VBZ".to_string()],
            ),
            (
                vec![
                    vec!["a".to_string()],
                    vec!["cat".to_string()],
                    vec!["sleeps".to_string()],
                ],
                vec!["DT".to_string(), "NN".to_string(), "VBZ".to_string()],
            ),
            (
                vec![
                    vec!["dogs".to_string()],
                    vec!["sleep".to_string()],
                ],
                vec!["NNS".to_string(), "VBP".to_string()],
            ),
        ]
    }

    #[test]
    fn test_forward_algorithm_shape() {
        let seqs = toy_sequences();
        let crf = LinearChainCRF::fit(&seqs, 1, 0.1).expect("fit failed");
        let (log_z, alpha) = crf
            .forward_algorithm(&seqs[0].0)
            .expect("forward failed");
        assert_eq!(alpha.len(), seqs[0].0.len());
        assert!(log_z.is_finite());
    }

    #[test]
    fn test_backward_algorithm_shape() {
        let seqs = toy_sequences();
        let crf = LinearChainCRF::fit(&seqs, 1, 0.1).expect("fit failed");
        let beta = crf
            .backward_algorithm(&seqs[0].0)
            .expect("backward failed");
        assert_eq!(beta.len(), seqs[0].0.len());
    }

    #[test]
    fn test_viterbi_length() {
        let seqs = toy_sequences();
        let crf = LinearChainCRF::fit(&seqs, 3, 0.05).expect("fit failed");
        let pred = crf.viterbi(&seqs[0].0).expect("viterbi failed");
        assert_eq!(pred.len(), seqs[0].0.len());
    }

    #[test]
    fn test_viterbi_labels_valid() {
        let seqs = toy_sequences();
        let crf = LinearChainCRF::fit(&seqs, 3, 0.05).expect("fit failed");
        for (features, _) in &seqs {
            let pred = crf.viterbi(features).expect("viterbi failed");
            for tag in &pred {
                assert!(crf.labels.contains(tag), "Unknown tag: {}", tag);
            }
        }
    }

    #[test]
    fn test_training_convergence() {
        // After many iterations the CRF should correctly tag training examples
        let seqs = toy_sequences();
        let crf = LinearChainCRF::fit(&seqs, 30, 0.1).expect("fit failed");
        // Check the simplest sequence: "dogs sleep" → NNS VBP
        let pred = crf.viterbi(&seqs[2].0).expect("viterbi failed");
        assert_eq!(pred, vec!["NNS", "VBP"]);
    }

    #[test]
    fn test_log_sum_exp_stability() {
        let v = vec![1000.0, 1001.0, 1002.0];
        let result = log_sum_exp(&v);
        // log(e^1000 + e^1001 + e^1002) ≈ 1002 + log(1 + e^-1 + e^-2)
        let expected = 1002.0 + (1.0_f64 + (-1.0_f64).exp() + (-2.0_f64).exp()).ln();
        assert!((result - expected).abs() < 1e-6, "got {}", result);
    }

    #[test]
    fn test_crf_empty_sequence() {
        let crf = LinearChainCRF::new(vec!["A".into(), "B".into()]);
        let pred = crf.viterbi(&[]).expect("viterbi empty");
        assert!(pred.is_empty());
    }

    #[test]
    fn test_crf_feature_keys() {
        let wf = CRFFeature::WordFeature {
            word: "hello".into(),
            label: "NN".into(),
            weight: 1.0,
        };
        assert_eq!(wf.key(), "WORD:hello:LABEL:NN");

        let tf = CRFFeature::TransitionFeature {
            prev_label: "DT".into(),
            curr_label: "NN".into(),
            weight: 0.5,
        };
        assert_eq!(tf.key(), "TRANS:DT:NN");

        let bf = CRFFeature::BigramFeature {
            word: "dog".into(),
            prev_word: "the".into(),
            label: "NN".into(),
            weight: 0.3,
        };
        assert_eq!(bf.key(), "BIGRAM:the:dog:LABEL:NN");
    }
}
