//! Advanced sequence labeling: full HMM with Baum-Welch EM, CRF tagger with
//! averaged perceptron, and IOB2 span utilities.

use std::collections::HashMap;

use crate::error::{Result, TextError};

// ─────────────────────────────────────────────────────────────────────────────
// HiddenMarkovModel
// ─────────────────────────────────────────────────────────────────────────────

/// A discrete Hidden Markov Model whose probabilities are stored as log-probabilities
/// to avoid numeric underflow.
///
/// Supports:
/// - Construction from explicit matrices.
/// - Viterbi decoding.
/// - Forward algorithm (log-likelihood).
/// - Baum-Welch EM training on unlabelled observation sequences.
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of distinct observation symbols.
    pub n_obs: usize,
    /// Log initial probabilities: `initial[s]` = log P(state_0 = s).
    pub initial: Vec<f64>,
    /// Log transition matrix: `transition[i][j]` = log P(state_t = j | state_{t-1} = i).
    pub transition: Vec<Vec<f64>>,
    /// Log emission matrix: `emission[s][o]` = log P(obs = o | state = s).
    pub emission: Vec<Vec<f64>>,
}

impl HiddenMarkovModel {
    /// Create an HMM with uniform (random-ish) initialisation.
    pub fn new(n_states: usize, n_obs: usize) -> Self {
        let log_pi = (1.0 / n_states as f64).ln();
        let log_t = (1.0 / n_states as f64).ln();
        let log_e = (1.0 / n_obs as f64).ln();

        HiddenMarkovModel {
            n_states,
            n_obs,
            initial: vec![log_pi; n_states],
            transition: vec![vec![log_t; n_states]; n_states],
            emission: vec![vec![log_e; n_obs]; n_states],
        }
    }

    // ── Viterbi decoding ────────────────────────────────────────────────

    /// Find the most likely state sequence for `observations` using the Viterbi
    /// algorithm.
    pub fn decode(&self, observations: &[usize]) -> Result<Vec<usize>> {
        let t = observations.len();
        if t == 0 {
            return Ok(vec![]);
        }
        self.validate_obs(observations)?;

        let s = self.n_states;
        // delta[t][s] = log P(best path to state s at time t, obs[0..=t])
        let mut delta = vec![vec![f64::NEG_INFINITY; s]; t];
        let mut psi: Vec<Vec<usize>> = vec![vec![0; s]; t];

        // Initialise
        for state in 0..s {
            delta[0][state] = self.initial[state] + self.emission[state][observations[0]];
        }

        // Recursion
        for time in 1..t {
            let obs = observations[time];
            for curr in 0..s {
                let em = self.emission[curr][obs];
                let (best_val, best_prev) = (0..s)
                    .map(|prev| (delta[time - 1][prev] + self.transition[prev][curr] + em, prev))
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((f64::NEG_INFINITY, 0));
                delta[time][curr] = best_val;
                psi[time][curr] = best_prev;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t];
        path[t - 1] = (0..s)
            .max_by(|&a, &b| {
                delta[t - 1][a]
                    .partial_cmp(&delta[t - 1][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        for time in (0..t - 1).rev() {
            path[time] = psi[time + 1][path[time + 1]];
        }
        Ok(path)
    }

    // ── Forward algorithm ───────────────────────────────────────────────

    /// Compute log P(observations | model) using the forward algorithm with
    /// log-sum-exp for numerical stability.
    pub fn log_likelihood(&self, observations: &[usize]) -> Result<f64> {
        if observations.is_empty() {
            return Ok(0.0);
        }
        self.validate_obs(observations)?;
        let (alpha, _) = self.forward(observations);
        let last = &alpha[observations.len() - 1];
        Ok(log_sum_exp(last))
    }

    // ── Baum-Welch EM ───────────────────────────────────────────────────

    /// Train the HMM using Baum-Welch EM on multiple observation sequences.
    ///
    /// # Arguments
    /// * `sequences` – collection of observation sequences (each a `Vec<usize>`
    ///   with symbol indices in `0..n_obs`).
    /// * `max_iter`  – maximum EM iterations.
    /// * `tol`       – convergence threshold on log-likelihood improvement.
    pub fn fit(
        &mut self,
        sequences: &[Vec<usize>],
        max_iter: usize,
        tol: f64,
    ) -> Result<()> {
        if sequences.is_empty() {
            return Ok(());
        }
        for seq in sequences {
            self.validate_obs(seq)?;
        }

        let s = self.n_states;
        let o = self.n_obs;
        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // Accumulate sufficient statistics across all sequences
            let mut log_gamma_sum = vec![f64::NEG_INFINITY; s]; // sum over t of gamma[t][state]
            let mut log_xi_sum = vec![vec![f64::NEG_INFINITY; s]; s]; // transition counts
            let mut log_emit_count = vec![vec![f64::NEG_INFINITY; o]; s]; // emission counts
            let mut log_gamma_t0 = vec![f64::NEG_INFINITY; s]; // initial counts

            let mut total_ll = 0.0;

            for seq in sequences {
                let t = seq.len();
                if t == 0 {
                    continue;
                }
                let (alpha, log_scale) = self.forward(seq);
                let beta = self.backward(seq, &log_scale);

                let ll = log_sum_exp(&alpha[t - 1]);
                total_ll += ll;

                // gamma[t][s] = log P(state_t = s | seq)
                let gamma: Vec<Vec<f64>> = (0..t)
                    .map(|ti| {
                        let g: Vec<f64> = (0..s)
                            .map(|si| alpha[ti][si] + beta[ti][si])
                            .collect();
                        let norm = log_sum_exp(&g);
                        g.iter().map(|&v| v - norm).collect()
                    })
                    .collect();

                // Accumulate initial state statistics
                for si in 0..s {
                    log_gamma_t0[si] = log_add(log_gamma_t0[si], gamma[0][si]);
                }

                // Accumulate gamma (denominator for transition and emission)
                for ti in 0..t {
                    for si in 0..s {
                        log_gamma_sum[si] = log_add(log_gamma_sum[si], gamma[ti][si]);
                    }
                }

                // Accumulate emission counts
                for ti in 0..t {
                    let obs = seq[ti];
                    for si in 0..s {
                        log_emit_count[si][obs] =
                            log_add(log_emit_count[si][obs], gamma[ti][si]);
                    }
                }

                // Accumulate xi (transition statistics) for t=0..t-2
                if t < 2 {
                    continue;
                }
                for ti in 0..t - 1 {
                    let obs_next = seq[ti + 1];
                    // xi_unnorm[i][j] = alpha[ti][i] + A[i][j] + B[j][obs_{t+1}] + beta[ti+1][j]
                    let mut xi_unnorm = vec![vec![f64::NEG_INFINITY; s]; s];
                    for i in 0..s {
                        for j in 0..s {
                            xi_unnorm[i][j] = alpha[ti][i]
                                + self.transition[i][j]
                                + self.emission[j][obs_next]
                                + beta[ti + 1][j];
                        }
                    }
                    // Normalise xi
                    let xi_total = log_sum_exp_2d(&xi_unnorm);
                    for i in 0..s {
                        for j in 0..s {
                            let xi_norm = xi_unnorm[i][j] - xi_total;
                            log_xi_sum[i][j] = log_add(log_xi_sum[i][j], xi_norm);
                        }
                    }
                }
            }

            // M-step: update parameters
            // Initial probabilities
            let pi_norm = log_sum_exp(&log_gamma_t0);
            for si in 0..s {
                self.initial[si] = log_gamma_t0[si] - pi_norm;
            }

            // Transition matrix
            for i in 0..s {
                let row_norm = log_sum_exp(&log_xi_sum[i]);
                for j in 0..s {
                    self.transition[i][j] = log_xi_sum[i][j] - row_norm;
                }
            }

            // Emission matrix
            for si in 0..s {
                // Denominator: sum over t of gamma[t][si]  (= log_gamma_sum[si])
                let denom = log_gamma_sum[si];
                for oi in 0..o {
                    self.emission[si][oi] = log_emit_count[si][oi] - denom;
                }
            }

            // Check convergence
            let improvement = total_ll - prev_ll;
            if improvement.abs() < tol && _iter > 0 {
                break;
            }
            prev_ll = total_ll;
        }
        Ok(())
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn validate_obs(&self, observations: &[usize]) -> Result<()> {
        for &o in observations {
            if o >= self.n_obs {
                return Err(TextError::InvalidInput(format!(
                    "observation {} out of range [0, {})",
                    o, self.n_obs
                )));
            }
        }
        Ok(())
    }

    /// Forward algorithm returning log-scaled alpha and per-step log-scalers.
    fn forward(&self, observations: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let t = observations.len();
        let s = self.n_states;
        let mut alpha = vec![vec![f64::NEG_INFINITY; s]; t];
        let mut log_scale = vec![0.0f64; t];

        for si in 0..s {
            alpha[0][si] = self.initial[si] + self.emission[si][observations[0]];
        }
        log_scale[0] = log_sum_exp(&alpha[0]);
        for si in 0..s {
            alpha[0][si] -= log_scale[0];
        }

        for time in 1..t {
            let obs = observations[time];
            for curr in 0..s {
                let terms: Vec<f64> = (0..s)
                    .map(|prev| alpha[time - 1][prev] + self.transition[prev][curr])
                    .collect();
                alpha[time][curr] = log_sum_exp(&terms) + self.emission[curr][obs];
            }
            log_scale[time] = log_sum_exp(&alpha[time]);
            for si in 0..s {
                alpha[time][si] -= log_scale[time];
            }
        }
        (alpha, log_scale)
    }

    /// Backward algorithm using the same log-scaling as forward.
    fn backward(&self, observations: &[usize], log_scale: &[f64]) -> Vec<Vec<f64>> {
        let t = observations.len();
        let s = self.n_states;
        let mut beta = vec![vec![0.0f64; s]; t]; // log(1) = 0

        // Normalise the last time step
        for si in 0..s {
            beta[t - 1][si] -= log_scale[t - 1];
        }

        for time in (0..t - 1).rev() {
            let obs_next = observations[time + 1];
            for prev in 0..s {
                let terms: Vec<f64> = (0..s)
                    .map(|curr| {
                        self.transition[prev][curr]
                            + self.emission[curr][obs_next]
                            + beta[time + 1][curr]
                    })
                    .collect();
                beta[time][prev] = log_sum_exp(&terms) - log_scale[time];
            }
        }
        beta
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CrfTagger — averaged perceptron linear CRF
// ─────────────────────────────────────────────────────────────────────────────

/// A linear CRF-style tagger trained with the **averaged perceptron**
/// algorithm.
///
/// Features include: word identity, lower-case, capitalization, prefixes,
/// suffixes, digit/punctuation flags, and the previous tag.
#[derive(Debug, Clone)]
pub struct CrfTagger {
    n_tags: usize,
    tag_to_id: HashMap<String, usize>,
    id_to_tag: Vec<String>,
    /// Current weights indexed by feature string → per-tag weights.
    weights: HashMap<String, Vec<f64>>,
    /// Accumulated weights for averaging.
    acc_weights: HashMap<String, Vec<f64>>,
    /// Number of weight updates (used for averaging).
    n_updates: usize,
}

impl CrfTagger {
    /// Construct a new tagger for the given tag set.
    pub fn new(tags: Vec<String>) -> Self {
        let n_tags = tags.len();
        let tag_to_id: HashMap<String, usize> = tags
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        CrfTagger {
            n_tags,
            tag_to_id,
            id_to_tag: tags,
            weights: HashMap::new(),
            acc_weights: HashMap::new(),
            n_updates: 0,
        }
    }

    // ── Feature extraction ─────────────────────────────────────────────

    /// Extract feature strings for `tokens[pos]` in context.
    pub fn extract_features(&self, tokens: &[String], pos: usize) -> Vec<String> {
        let word = &tokens[pos];
        let lower = word.to_lowercase();
        let mut feats = Vec::with_capacity(24);

        feats.push(format!("WORD={}", word));
        feats.push(format!("LOWER={}", lower));
        feats.push(format!("IS_UPPER={}", word.chars().all(|c| c.is_uppercase())));
        feats.push(format!(
            "IS_TITLE={}",
            word.chars().next().map_or(false, char::is_uppercase)
                && word.chars().skip(1).all(|c| c.is_lowercase())
        ));
        feats.push(format!(
            "HAS_DIGIT={}",
            word.chars().any(|c| c.is_ascii_digit())
        ));
        feats.push(format!(
            "HAS_HYPHEN={}",
            word.contains('-')
        ));
        feats.push(format!(
            "IS_ALPHA={}",
            word.chars().all(|c| c.is_alphabetic())
        ));

        // Prefix / suffix features
        let chars: Vec<char> = word.chars().collect();
        for n in 1..=4.min(chars.len()) {
            let pfx: String = chars[..n].iter().collect();
            feats.push(format!("PRE{}={}", n, pfx));
        }
        for n in 1..=4.min(chars.len()) {
            let sfx: String = chars[chars.len() - n..].iter().collect();
            feats.push(format!("SUF{}={}", n, sfx));
        }

        // Context words
        if pos > 0 {
            feats.push(format!("PREV={}", tokens[pos - 1].to_lowercase()));
        } else {
            feats.push("PREV=BOS".to_string());
        }
        if pos + 1 < tokens.len() {
            feats.push(format!("NEXT={}", tokens[pos + 1].to_lowercase()));
        } else {
            feats.push("NEXT=EOS".to_string());
        }
        if pos > 1 {
            feats.push(format!("PREV2={}", tokens[pos - 2].to_lowercase()));
        }
        if pos + 2 < tokens.len() {
            feats.push(format!("NEXT2={}", tokens[pos + 2].to_lowercase()));
        }

        // Bigram context
        if pos > 0 {
            feats.push(format!(
                "BIGRAM_PREV={}_{}",
                tokens[pos - 1].to_lowercase(),
                lower
            ));
        }
        if pos + 1 < tokens.len() {
            feats.push(format!(
                "BIGRAM_NEXT={}_{}",
                lower,
                tokens[pos + 1].to_lowercase()
            ));
        }

        feats
    }

    // ── Score computation ──────────────────────────────────────────────

    fn score(&self, features: &[String], tag_id: usize) -> f64 {
        let mut s = 0.0;
        for feat in features {
            if let Some(w) = self.weights.get(feat) {
                if let Some(&wt) = w.get(tag_id) {
                    s += wt;
                }
            }
        }
        s
    }

    fn scores_all(&self, features: &[String]) -> Vec<f64> {
        (0..self.n_tags)
            .map(|t| self.score(features, t))
            .collect()
    }

    // ── Viterbi prediction ─────────────────────────────────────────────

    /// Predict tag sequence using greedy Viterbi decoding.
    pub fn predict(&self, tokens: &[String]) -> Vec<String> {
        if tokens.is_empty() {
            return vec![];
        }
        let n = tokens.len();
        let s = self.n_tags;

        // dp[i][j] = best score to be in tag j at position i
        let mut dp = vec![vec![f64::NEG_INFINITY; s]; n];
        let mut back = vec![vec![0usize; s]; n];

        // Initialise with transition from BOS
        let feats_0 = self.extract_features(tokens, 0);
        let scores_0 = self.scores_all(&feats_0);
        for ti in 0..s {
            dp[0][ti] = scores_0[ti];
        }

        // Transition weights (prefix TRANS:prev:curr)
        for pos in 1..n {
            let feats = self.extract_features(tokens, pos);
            let emit_scores = self.scores_all(&feats);
            for curr in 0..s {
                let curr_tag = &self.id_to_tag[curr];
                let best = (0..s)
                    .map(|prev| {
                        let prev_tag = &self.id_to_tag[prev];
                        let trans_key = format!("TRANS={}_{}", prev_tag, curr_tag);
                        let trans_score = self
                            .weights
                            .get(&trans_key)
                            .and_then(|w| w.get(curr))
                            .copied()
                            .unwrap_or(0.0);
                        (dp[pos - 1][prev] + trans_score + emit_scores[curr], prev)
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((f64::NEG_INFINITY, 0));
                dp[pos][curr] = best.0;
                back[pos][curr] = best.1;
            }
        }

        // Backtrack
        let mut path = vec![0usize; n];
        path[n - 1] = (0..s)
            .max_by(|&a, &b| {
                dp[n - 1][a]
                    .partial_cmp(&dp[n - 1][b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        for pos in (0..n - 1).rev() {
            path[pos] = back[pos + 1][path[pos + 1]];
        }

        path.iter().map(|&ti| self.id_to_tag[ti].clone()).collect()
    }

    // ── Averaged perceptron training ───────────────────────────────────

    /// Update weights for a single token (online perceptron update).
    fn update_weights(&mut self, features: &[String], pred_tag: usize, gold_tag: usize) {
        if pred_tag == gold_tag {
            return;
        }
        self.n_updates += 1;
        for feat in features {
            let w = self
                .weights
                .entry(feat.clone())
                .or_insert_with(|| vec![0.0; self.n_tags]);
            let aw = self
                .acc_weights
                .entry(feat.clone())
                .or_insert_with(|| vec![0.0; self.n_tags]);

            w[gold_tag] += 1.0;
            w[pred_tag] -= 1.0;
            let c = self.n_updates as f64;
            aw[gold_tag] += c;
            aw[pred_tag] -= c;
        }

        // Transition feature
        let gold_tag_str = &self.id_to_tag[gold_tag];
        let pred_tag_str = &self.id_to_tag[pred_tag];
        let gold_trans = format!("TRANS=BOS_{}", gold_tag_str);
        let pred_trans = format!("TRANS=BOS_{}", pred_tag_str);

        for trans_key in [gold_trans, pred_trans] {
            self.weights
                .entry(trans_key.clone())
                .or_insert_with(|| vec![0.0; self.n_tags]);
            self.acc_weights
                .entry(trans_key)
                .or_insert_with(|| vec![0.0; self.n_tags]);
        }
    }

    /// Train with averaged perceptron on labelled sequences.
    ///
    /// `sequences` is a slice of `(tokens, tags)` pairs.
    pub fn train(&mut self, sequences: &[(Vec<String>, Vec<String>)], n_epochs: usize) {
        for _epoch in 0..n_epochs {
            for (tokens, gold_tags) in sequences {
                if tokens.len() != gold_tags.len() {
                    continue;
                }
                let predicted = self.predict(tokens);
                for (pos, (pred_str, gold_str)) in
                    predicted.iter().zip(gold_tags.iter()).enumerate()
                {
                    let pred_id = self.tag_to_id.get(pred_str.as_str()).copied().unwrap_or(0);
                    let gold_id = match self.tag_to_id.get(gold_str.as_str()) {
                        Some(&id) => id,
                        None => continue,
                    };
                    if pred_id != gold_id {
                        let feats = self.extract_features(tokens, pos);
                        self.update_weights(&feats, pred_id, gold_id);
                    }
                }
            }
        }

        // Average weights
        let c = self.n_updates as f64;
        if c > 0.0 {
            for (feat, w) in &mut self.weights {
                if let Some(aw) = self.acc_weights.get(feat) {
                    for (wi, awi) in w.iter_mut().zip(aw.iter()) {
                        *wi -= awi / c;
                    }
                }
            }
        }
    }

    /// Evaluate entity-level F1 on a test set using exact span matching.
    pub fn evaluate(&self, test_set: &[(Vec<String>, Vec<String>)]) -> f64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (tokens, gold_tags) in test_set {
            let pred_tags = self.predict(tokens);
            let gold_spans = iob2_to_spans(tokens, gold_tags);
            let pred_spans = iob2_to_spans(tokens, &pred_tags);

            let gold_set: std::collections::HashSet<_> = gold_spans.iter().cloned().collect();
            let pred_set: std::collections::HashSet<_> = pred_spans.iter().cloned().collect();

            tp += gold_set.intersection(&pred_set).count();
            fp += pred_set.difference(&gold_set).count();
            fn_ += gold_set.difference(&pred_set).count();
        }

        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        };
        let recall = if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        };

        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IOB2 utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert IOB2 tag sequence to named spans.
///
/// Returns `Vec<(entity_type, start_token_idx, end_token_idx_exclusive)>`.
///
/// Example tags: `["O", "B-PER", "I-PER", "O"]` → `[("PER", 1, 3)]`.
pub fn iob2_to_spans(
    _tokens: &[String],
    tags: &[String],
) -> Vec<(String, usize, usize)> {
    let mut spans = Vec::new();
    let mut current: Option<(String, usize)> = None;

    for (i, tag) in tags.iter().enumerate() {
        if tag == "O" {
            if let Some((entity, start)) = current.take() {
                spans.push((entity, start, i));
            }
        } else if let Some(stripped) = tag.strip_prefix("B-") {
            if let Some((entity, start)) = current.take() {
                spans.push((entity, start, i));
            }
            current = Some((stripped.to_string(), i));
        } else if let Some(stripped) = tag.strip_prefix("I-") {
            match &current {
                Some((entity, _)) if entity == stripped => {
                    // Continue current span
                }
                _ => {
                    // Malformed IOB2 — treat I- without matching B- as a new span
                    if let Some((entity, start)) = current.take() {
                        spans.push((entity, start, i));
                    }
                    current = Some((stripped.to_string(), i));
                }
            }
        }
    }

    if let Some((entity, start)) = current {
        spans.push((entity, start, tags.len()));
    }

    spans
}

/// Convert named spans back to IOB2 tag sequence of length `n`.
///
/// `spans` is a slice of `(entity_type, start_inclusive, end_exclusive)`.
/// Positions not covered by any span receive tag `"O"`.
pub fn spans_to_iob2(
    _tokens: &[String],
    spans: &[(String, usize, usize)],
    n: usize,
) -> Vec<String> {
    let mut tags = vec!["O".to_string(); n];
    for (entity, start, end) in spans {
        for i in *start..*end {
            if i >= n {
                break;
            }
            if i == *start {
                tags[i] = format!("B-{}", entity);
            } else {
                tags[i] = format!("I-{}", entity);
            }
        }
    }
    tags
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Log-sum-exp of a slice, stable against overflow/underflow.
fn log_sum_exp(vals: &[f64]) -> f64 {
    let max_val = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = vals
        .iter()
        .map(|&v| (v - max_val).exp())
        .sum();
    max_val + sum.ln()
}

/// Log-add of two log-space values: log(exp(a) + exp(b)).
#[inline]
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a >= b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}

/// Log-sum-exp over all elements of a 2-D slice.
fn log_sum_exp_2d(mat: &[Vec<f64>]) -> f64 {
    let all: Vec<f64> = mat.iter().flat_map(|row| row.iter().copied()).collect();
    log_sum_exp(&all)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HMM tests ───────────────────────────────────────────────────────

    #[test]
    fn test_hmm_decode_length() {
        let mut hmm = HiddenMarkovModel::new(2, 3);
        let obs = vec![0, 1, 2, 0];
        let path = hmm.decode(&obs).expect("decode failed");
        assert_eq!(path.len(), obs.len());
        for &s in &path {
            assert!(s < 2);
        }
    }

    #[test]
    fn test_hmm_empty_decode() {
        let hmm = HiddenMarkovModel::new(2, 2);
        let path = hmm.decode(&[]).expect("decode empty");
        assert!(path.is_empty());
    }

    #[test]
    fn test_hmm_log_likelihood_finite() {
        let hmm = HiddenMarkovModel::new(2, 3);
        let obs = vec![0, 1, 2];
        let ll = hmm.log_likelihood(&obs).expect("ll failed");
        assert!(ll.is_finite(), "log-likelihood should be finite, got {}", ll);
    }

    #[test]
    fn test_hmm_fit_runs_without_error() {
        let mut hmm = HiddenMarkovModel::new(2, 3);
        let seqs = vec![vec![0usize, 1, 2], vec![1, 0, 2], vec![2, 2, 0]];
        hmm.fit(&seqs, 5, 1e-4).expect("fit failed");
        // After training the parameters should still be valid log-probabilities
        let ll = hmm.log_likelihood(&[0, 1]).expect("ll after fit");
        assert!(ll.is_finite());
    }

    #[test]
    fn test_hmm_invalid_observation() {
        let hmm = HiddenMarkovModel::new(2, 3);
        let result = hmm.decode(&[0, 5]); // 5 >= n_obs
        assert!(result.is_err());
    }

    // ── CrfTagger tests ─────────────────────────────────────────────────

    fn simple_ner_data() -> Vec<(Vec<String>, Vec<String>)> {
        vec![
            (
                vec!["John".into(), "lives".into(), "in".into(), "Paris".into()],
                vec!["B-PER".into(), "O".into(), "O".into(), "B-LOC".into()],
            ),
            (
                vec!["Alice".into(), "works".into(), "at".into(), "Google".into()],
                vec!["B-PER".into(), "O".into(), "O".into(), "B-ORG".into()],
            ),
            (
                vec!["Bob".into(), "visited".into(), "London".into()],
                vec!["B-PER".into(), "O".into(), "B-LOC".into()],
            ),
        ]
    }

    #[test]
    fn test_crf_train_predict_length() {
        let tags = vec!["O".to_string(), "B-PER".to_string(), "I-PER".to_string(),
                        "B-LOC".to_string(), "B-ORG".to_string()];
        let mut crf = CrfTagger::new(tags);
        let data = simple_ner_data();
        crf.train(&data, 5);
        let tokens: Vec<String> = vec!["John".into(), "travels".into(), "to".into(), "Paris".into()];
        let preds = crf.predict(&tokens);
        assert_eq!(preds.len(), tokens.len());
    }

    #[test]
    fn test_crf_predict_empty() {
        let crf = CrfTagger::new(vec!["O".to_string(), "B-PER".to_string()]);
        let preds = crf.predict(&[]);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_crf_evaluate_returns_f1() {
        let tags = vec!["O".to_string(), "B-PER".to_string(), "I-PER".to_string(),
                        "B-LOC".to_string(), "B-ORG".to_string()];
        let mut crf = CrfTagger::new(tags);
        let data = simple_ner_data();
        crf.train(&data, 10);
        let f1 = crf.evaluate(&data);
        assert!((0.0..=1.0).contains(&f1), "F1 should be in [0,1], got {}", f1);
    }

    #[test]
    fn test_extract_features_non_empty() {
        let crf = CrfTagger::new(vec!["O".to_string()]);
        let tokens: Vec<String> = vec!["Hello".into(), "World".into()];
        let feats = crf.extract_features(&tokens, 0);
        assert!(!feats.is_empty());
        assert!(feats.iter().any(|f| f.starts_with("WORD=")));
        assert!(feats.iter().any(|f| f.starts_with("SUF")));
    }

    // ── IOB2 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_iob2_to_spans_basic() {
        let tokens: Vec<String> = vec!["O".into(), "B".into(), "I".into(), "N".into()];
        let tags: Vec<String> = vec!["O".into(), "B-PER".into(), "I-PER".into(), "O".into()];
        let spans = iob2_to_spans(&tokens, &tags);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].0, "PER");
        assert_eq!(spans[0].1, 1);
        assert_eq!(spans[0].2, 3);
    }

    #[test]
    fn test_iob2_roundtrip() {
        let tokens: Vec<String> = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let tags: Vec<String> = vec!["B-PER".into(), "I-PER".into(), "O".into(), "B-LOC".into()];
        let spans = iob2_to_spans(&tokens, &tags);
        let recovered = spans_to_iob2(&tokens, &spans, tokens.len());
        assert_eq!(recovered, tags);
    }

    #[test]
    fn test_spans_to_iob2_all_o() {
        let tokens: Vec<String> = vec!["x".into(), "y".into()];
        let tags = spans_to_iob2(&tokens, &[], 2);
        assert_eq!(tags, vec!["O".to_string(), "O".to_string()]);
    }
}
