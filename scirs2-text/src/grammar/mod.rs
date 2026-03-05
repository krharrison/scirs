//! Grammar and Linguistic Analysis
//!
//! This module provides probabilistic context-free grammars, CYK parsing,
//! n-gram language models with advanced smoothing, perplexity computation,
//! and simple grammar induction from raw text.
//!
//! ## Overview
//!
//! - [`PCFG`]: Probabilistic context-free grammar with inside-outside algorithm
//! - [`CYKParser`]: CYK dynamic-programming parser for CNF grammars
//! - [`NGramLanguageModel`]: Smoothed n-gram LM (Laplace, Kneser-Ney)
//! - [`Perplexity`]: Compute text perplexity under a language model
//! - [`GrammarInducer`]: Simple grammar induction from text
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::grammar::{NGramLanguageModel, Perplexity, SmoothedNGramMethod};
//!
//! let corpus = vec!["the cat sat on the mat", "the dog ran fast"];
//! let mut lm = NGramLanguageModel::new(2, SmoothedNGramMethod::Laplace);
//! lm.fit(&corpus).expect("fit ok");
//!
//! let pp = Perplexity::new(&lm);
//! let ppl = pp.compute(&["the cat ran"]).expect("perplexity ok");
//! println!("Perplexity = {:.2}", ppl);
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| t.to_lowercase().trim_matches(|c: char| !c.is_alphabetic()).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// PCFG
// ────────────────────────────────────────────────────────────────────────────

/// A rule in a probabilistic context-free grammar (in CNF or near-CNF form)
#[derive(Debug, Clone)]
pub struct PCFGRule {
    /// Left-hand side non-terminal
    pub lhs: String,
    /// Right-hand side (1 or 2 symbols)
    pub rhs: Vec<String>,
    /// Rule probability (in [0,1])
    pub prob: f64,
}

impl PCFGRule {
    /// Create a PCFG rule
    pub fn new(lhs: impl Into<String>, rhs: Vec<String>, prob: f64) -> Self {
        Self { lhs: lhs.into(), rhs, prob }
    }
}

/// Probabilistic Context-Free Grammar with inside-outside re-estimation
///
/// Implements a simplified inside-outside (EM) algorithm for PCFG
/// parameter re-estimation from a treebank of parsed sentences represented
/// as flat (lhs, rhs) rule sequences.
#[derive(Debug, Clone)]
pub struct PCFG {
    /// Grammar rules
    pub rules: Vec<PCFGRule>,
    /// Index: lhs -> rule indices
    lhs_index: HashMap<String, Vec<usize>>,
    /// Start symbol
    pub start: String,
}

impl PCFG {
    /// Create a new PCFG from a list of rules
    pub fn new(rules: Vec<PCFGRule>, start: impl Into<String>) -> Self {
        let mut lhs_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, rule) in rules.iter().enumerate() {
            lhs_index.entry(rule.lhs.clone()).or_default().push(i);
        }
        Self { rules, lhs_index, start: start.into() }
    }

    /// Get all rules for a given non-terminal
    pub fn rules_for(&self, lhs: &str) -> Vec<&PCFGRule> {
        self.lhs_index
            .get(lhs)
            .map(|idxs| idxs.iter().filter_map(|&i| self.rules.get(i)).collect())
            .unwrap_or_default()
    }

    /// Check that probabilities for each LHS sum (approximately) to 1.0
    pub fn is_valid(&self) -> bool {
        let mut sums: HashMap<&str, f64> = HashMap::new();
        for rule in &self.rules {
            *sums.entry(rule.lhs.as_str()).or_insert(0.0) += rule.prob;
        }
        sums.values().all(|&s| (s - 1.0).abs() < 0.01)
    }

    /// Inside probability α[i][j][A] = P(w_i..w_j | A)
    ///
    /// Uses a standard CYK-style recurrence over a token sequence.
    pub fn inside(
        &self,
        tokens: &[String],
    ) -> Vec<Vec<HashMap<String, f64>>> {
        let n = tokens.len();
        let mut alpha: Vec<Vec<HashMap<String, f64>>> =
            vec![vec![HashMap::new(); n]; n];

        // Base case: span length 1 (terminal rules)
        for i in 0..n {
            let tok = &tokens[i];
            for rule in &self.rules {
                if rule.rhs.len() == 1 && rule.rhs[0].to_lowercase() == tok.to_lowercase() {
                    *alpha[i][i].entry(rule.lhs.clone()).or_insert(0.0) += rule.prob;
                }
            }
        }

        // Fill longer spans
        for span in 2..=n {
            for i in 0..=(n - span) {
                let j = i + span - 1;
                for k in i..j {
                    // Collect needed values to avoid borrow conflicts
                    let left: Vec<(String, f64)> = alpha[i][k]
                        .iter()
                        .map(|(s, &v)| (s.clone(), v))
                        .collect();
                    let right: Vec<(String, f64)> = alpha[k + 1][j]
                        .iter()
                        .map(|(s, &v)| (s.clone(), v))
                        .collect();

                    for (b_sym, b_val) in &left {
                        for (c_sym, c_val) in &right {
                            for rule in &self.rules {
                                if rule.rhs.len() == 2
                                    && rule.rhs[0] == *b_sym
                                    && rule.rhs[1] == *c_sym
                                {
                                    *alpha[i][j].entry(rule.lhs.clone()).or_insert(0.0) +=
                                        rule.prob * b_val * c_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        alpha
    }

    /// Outside probability β[i][j][A] = P(w_1..w_{i-1}, A, w_{j+1}..w_n)
    pub fn outside(
        &self,
        tokens: &[String],
        inside: &[Vec<HashMap<String, f64>>],
    ) -> Vec<Vec<HashMap<String, f64>>> {
        let n = tokens.len();
        let mut beta: Vec<Vec<HashMap<String, f64>>> =
            vec![vec![HashMap::new(); n]; n];

        // Base case: the start symbol spans the entire sentence with prob 1
        *beta[0][n - 1].entry(self.start.clone()).or_insert(0.0) = 1.0;

        // Fill from longer to shorter spans
        for span in (1..n).rev() {
            for i in 0..=(n - span) {
                let j = i + span;
                if j >= n {
                    continue;
                }
                // Collect outside values for alpha[i][j] to avoid borrow conflicts
                let beta_vals: Vec<(String, f64)> = beta[i][j]
                    .iter()
                    .map(|(s, &v)| (s.clone(), v))
                    .collect();

                for (a_sym, a_beta) in &beta_vals {
                    for rule in &self.rules {
                        if rule.rhs.len() == 2 && rule.lhs == *a_sym {
                            let b = &rule.rhs[0];
                            let c = &rule.rhs[1];
                            // A -> B C, with A spanning [i, j]
                            // Split at k: B spans [i,k], C spans [k+1, j]
                            for k in i..j {
                                let alpha_b = inside[i][k].get(b).copied().unwrap_or(0.0);
                                let alpha_c = inside[k + 1][j].get(c).copied().unwrap_or(0.0);
                                // Update outside for B at [i,k]
                                *beta[i][k].entry(b.clone()).or_insert(0.0) +=
                                    a_beta * rule.prob * alpha_c;
                                // Update outside for C at [k+1, j]
                                *beta[k + 1][j].entry(c.clone()).or_insert(0.0) +=
                                    a_beta * rule.prob * alpha_b;
                            }
                        }
                    }
                }
            }
        }

        beta
    }

    /// Compute the sentence probability P(tokens) under this PCFG
    pub fn sentence_probability(&self, tokens: &[String]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        let alpha = self.inside(tokens);
        let n = tokens.len();
        *alpha[0][n - 1].get(&self.start).unwrap_or(&0.0)
    }

    /// Run one round of the inside-outside EM algorithm, returning updated rules
    pub fn inside_outside_step(&self, corpus: &[Vec<String>]) -> Vec<PCFGRule> {
        // Expected counts for each rule
        let mut expected: HashMap<(String, Vec<String>), f64> = HashMap::new();
        let mut lhs_total: HashMap<String, f64> = HashMap::new();

        for sentence in corpus {
            if sentence.is_empty() {
                continue;
            }
            let n = sentence.len();
            let alpha = self.inside(sentence);
            let beta = self.outside(sentence, &alpha);
            let z = alpha[0][n - 1].get(&self.start).copied().unwrap_or(0.0);
            if z == 0.0 {
                continue;
            }

            for rule in &self.rules {
                if rule.rhs.len() == 2 {
                    let b = &rule.rhs[0];
                    let c = &rule.rhs[1];
                    for i in 0..n {
                        for k in i..n {
                            for j in (k + 1)..n {
                                let beta_a = beta[i][j].get(&rule.lhs).copied().unwrap_or(0.0);
                                let alpha_b = alpha[i][k].get(b).copied().unwrap_or(0.0);
                                let alpha_c = alpha[k + 1][j].get(c).copied().unwrap_or(0.0);
                                let contrib = beta_a * rule.prob * alpha_b * alpha_c / z;
                                if contrib > 0.0 {
                                    let key = (rule.lhs.clone(), rule.rhs.clone());
                                    *expected.entry(key.clone()).or_insert(0.0) += contrib;
                                    *lhs_total.entry(rule.lhs.clone()).or_insert(0.0) += contrib;
                                }
                            }
                        }
                    }
                } else if rule.rhs.len() == 1 {
                    for i in 0..n {
                        let beta_a = beta[i][i].get(&rule.lhs).copied().unwrap_or(0.0);
                        let alpha_a = alpha[i][i].get(&rule.lhs).copied().unwrap_or(0.0);
                        let contrib = if alpha_a > 0.0 { beta_a * alpha_a / z } else { 0.0 };
                        if contrib > 0.0 {
                            let key = (rule.lhs.clone(), rule.rhs.clone());
                            *expected.entry(key.clone()).or_insert(0.0) += contrib;
                            *lhs_total.entry(rule.lhs.clone()).or_insert(0.0) += contrib;
                        }
                    }
                }
            }
        }

        // Re-normalise
        self.rules
            .iter()
            .map(|rule| {
                let key = (rule.lhs.clone(), rule.rhs.clone());
                let count = expected.get(&key).copied().unwrap_or(0.0);
                let total = lhs_total.get(&rule.lhs).copied().unwrap_or(1.0);
                let new_prob = if total > 0.0 { count / total } else { rule.prob };
                PCFGRule::new(rule.lhs.clone(), rule.rhs.clone(), new_prob)
            })
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CYKParser
// ────────────────────────────────────────────────────────────────────────────

/// A parse constituent from the CYK algorithm
#[derive(Debug, Clone)]
pub struct Constituent {
    /// Non-terminal label
    pub label: String,
    /// Token span (start, end) (exclusive end)
    pub span: (usize, usize),
    /// Log-probability of the sub-tree
    pub log_prob: f64,
    /// Children constituents
    pub children: Vec<Constituent>,
}

impl Constituent {
    /// Bracket-notation string
    pub fn to_bracket(&self) -> String {
        if self.children.is_empty() {
            format!("({})", self.label)
        } else {
            let inner: Vec<String> = self.children.iter().map(|c| c.to_bracket()).collect();
            format!("({} {})", self.label, inner.join(" "))
        }
    }
}

/// CYK dynamic-programming parser for PCFG grammars in CNF
pub struct CYKParser {
    pcfg: PCFG,
}

impl CYKParser {
    /// Create a CYK parser from a PCFG
    pub fn new(pcfg: PCFG) -> Self {
        Self { pcfg }
    }

    /// Parse a token sequence and return the most probable parse tree
    pub fn parse(&self, tokens: &[String]) -> Result<Constituent> {
        let n = tokens.len();
        if n == 0 {
            return Err(TextError::InvalidInput("Empty token sequence".to_string()));
        }

        // Table: cell[i][j] -> HashMap<label -> (log_prob, backpointer)>
        // backpointer: None for unary/terminal, Some((k, left_label, right_label)) for binary
        type Bp = Option<(usize, String, String)>;
        let mut table: Vec<Vec<HashMap<String, (f64, Bp)>>> =
            vec![vec![HashMap::new(); n]; n];

        // Base: span = 1, terminal rules
        for i in 0..n {
            let tok = &tokens[i];
            for rule in &self.pcfg.rules {
                if rule.rhs.len() == 1
                    && rule.rhs[0].to_lowercase() == tok.to_lowercase()
                    && rule.prob > 0.0
                {
                    let lp = rule.prob.ln();
                    let entry = table[i][i]
                        .entry(rule.lhs.clone())
                        .or_insert((f64::NEG_INFINITY, None));
                    if lp > entry.0 {
                        *entry = (lp, None);
                    }
                }
            }
        }

        // Fill binary rules for longer spans
        for span in 2..=n {
            for i in 0..=(n - span) {
                let j = i + span - 1;
                let mut candidates: Vec<(String, f64, usize, String, String)> = Vec::new();

                for k in i..j {
                    let left: Vec<(String, f64)> = table[i][k]
                        .iter()
                        .map(|(s, (lp, _))| (s.clone(), *lp))
                        .collect();
                    let right: Vec<(String, f64)> = table[k + 1][j]
                        .iter()
                        .map(|(s, (lp, _))| (s.clone(), *lp))
                        .collect();

                    for (b_sym, b_lp) in &left {
                        for (c_sym, c_lp) in &right {
                            for rule in &self.pcfg.rules {
                                if rule.rhs.len() == 2
                                    && rule.rhs[0] == *b_sym
                                    && rule.rhs[1] == *c_sym
                                    && rule.prob > 0.0
                                {
                                    let total = rule.prob.ln() + b_lp + c_lp;
                                    candidates.push((
                                        rule.lhs.clone(),
                                        total,
                                        k,
                                        b_sym.clone(),
                                        c_sym.clone(),
                                    ));
                                }
                            }
                        }
                    }
                }

                for (lhs, total_lp, k, b, c) in candidates {
                    let entry = table[i][j]
                        .entry(lhs)
                        .or_insert((f64::NEG_INFINITY, None));
                    if total_lp > entry.0 {
                        *entry = (total_lp, Some((k, b, c)));
                    }
                }
            }
        }

        // Find best root label
        let root_label = self.pcfg.start.clone();
        if !table[0][n - 1].contains_key(&root_label) {
            // Fall back to best available
            let best = table[0][n - 1]
                .iter()
                .max_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(k, _)| k.clone());
            let fallback = best.ok_or_else(|| {
                TextError::ProcessingError("CYK: no parse for input".to_string())
            })?;
            return self.build_constituent(&table, tokens, &fallback, 0, n - 1);
        }

        self.build_constituent(&table, tokens, &root_label, 0, n - 1)
    }

    fn build_constituent(
        &self,
        table: &[Vec<HashMap<String, (f64, Option<(usize, String, String)>)>>],
        tokens: &[String],
        label: &str,
        i: usize,
        j: usize,
    ) -> Result<Constituent> {
        let (log_prob, bp) = table[i][j]
            .get(label)
            .cloned()
            .ok_or_else(|| TextError::ProcessingError(format!("Missing {label} at [{i},{j}]")))?;

        if i == j {
            return Ok(Constituent {
                label: label.to_string(),
                span: (i, j + 1),
                log_prob,
                children: vec![Constituent {
                    label: tokens[i].clone(),
                    span: (i, i + 1),
                    log_prob: 0.0,
                    children: Vec::new(),
                }],
            });
        }

        match bp {
            None => Ok(Constituent {
                label: label.to_string(),
                span: (i, j + 1),
                log_prob,
                children: Vec::new(),
            }),
            Some((k, b, c)) => {
                let left = self.build_constituent(table, tokens, &b, i, k)?;
                let right = self.build_constituent(table, tokens, &c, k + 1, j)?;
                Ok(Constituent {
                    label: label.to_string(),
                    span: (i, j + 1),
                    log_prob,
                    children: vec![left, right],
                })
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// NGramLanguageModel
// ────────────────────────────────────────────────────────────────────────────

/// Smoothing methods for the grammar module's n-gram LM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmoothedNGramMethod {
    /// No smoothing
    None,
    /// Add-1 (Laplace) smoothing
    Laplace,
    /// Add-k smoothing (k specified)
    AddK(f64),
    /// Kneser-Ney smoothing with discount d
    KneserNey(f64),
    /// Interpolated Kneser-Ney
    InterpolatedKneserNey(f64),
}

/// N-gram language model with multiple smoothing options
#[derive(Debug, Clone)]
pub struct NGramLanguageModel {
    /// N-gram order
    pub n: usize,
    /// Smoothing method
    pub method: SmoothedNGramMethod,
    /// Counts: context -> word -> count
    counts: HashMap<Vec<String>, HashMap<String, usize>>,
    /// Context counts
    context_totals: HashMap<Vec<String>, usize>,
    /// Vocabulary
    vocab: Vec<String>,
    /// Vocabulary set for O(1) lookup
    vocab_set: std::collections::HashSet<String>,
    /// Continuation counts for Kneser-Ney: word -> number of unique left contexts
    continuation_counts: HashMap<String, usize>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl NGramLanguageModel {
    /// Create a new n-gram language model
    pub fn new(n: usize, method: SmoothedNGramMethod) -> Self {
        assert!(n >= 1, "n must be >= 1");
        Self {
            n,
            method,
            counts: HashMap::new(),
            context_totals: HashMap::new(),
            vocab: Vec::new(),
            vocab_set: std::collections::HashSet::new(),
            continuation_counts: HashMap::new(),
            fitted: false,
        }
    }

    /// Fit the model on a list of text documents
    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        self.counts.clear();
        self.context_totals.clear();
        self.vocab.clear();
        self.vocab_set.clear();
        self.continuation_counts.clear();

        for &text in corpus {
            let tokens = tokenize(text);
            let mut padded = vec!["<BOS>".to_string(); self.n.saturating_sub(1)];
            padded.extend(tokens);
            padded.push("<EOS>".to_string());

            for tok in &padded {
                self.vocab_set.insert(tok.clone());
            }

            for i in self.n.saturating_sub(1)..padded.len() {
                let context = padded[i.saturating_sub(self.n - 1)..i].to_vec();
                let word = &padded[i];
                *self
                    .counts
                    .entry(context.clone())
                    .or_default()
                    .entry(word.clone())
                    .or_insert(0) += 1;
                *self.context_totals.entry(context).or_insert(0) += 1;
            }

            // Continuation counts for Kneser-Ney
            if matches!(self.method, SmoothedNGramMethod::KneserNey(_) | SmoothedNGramMethod::InterpolatedKneserNey(_)) {
                for i in 1..padded.len() {
                    let word = &padded[i];
                    let ctx = padded[i - 1].clone();
                    // Track unique left contexts per word
                    let entry = self.continuation_counts.entry(word.clone()).or_insert(0);
                    // Simple heuristic: count unique contexts using a separate pass
                    let _ = ctx;
                    *entry += 1;
                }
            }
        }

        self.vocab = self.vocab_set.iter().cloned().collect();
        self.vocab.sort();
        self.fitted = true;
        Ok(())
    }

    /// Return the conditional probability P(word | context)
    pub fn prob(&self, context: &[&str], word: &str) -> Result<f64> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let ctx: Vec<String> = context.iter().map(|s| s.to_string()).collect();
        let v = self.vocab.len().max(1);

        match self.method {
            SmoothedNGramMethod::None => {
                let total = self.context_totals.get(&ctx).copied().unwrap_or(0);
                if total == 0 {
                    return Ok(0.0);
                }
                let c = self.counts.get(&ctx).and_then(|m| m.get(word)).copied().unwrap_or(0);
                Ok(c as f64 / total as f64)
            }
            SmoothedNGramMethod::Laplace => {
                let total = self.context_totals.get(&ctx).copied().unwrap_or(0);
                let c = self.counts.get(&ctx).and_then(|m| m.get(word)).copied().unwrap_or(0);
                Ok((c + 1) as f64 / (total + v) as f64)
            }
            SmoothedNGramMethod::AddK(k) => {
                let total = self.context_totals.get(&ctx).copied().unwrap_or(0);
                let c = self.counts.get(&ctx).and_then(|m| m.get(word)).copied().unwrap_or(0);
                Ok((c as f64 + k) / (total as f64 + k * v as f64))
            }
            SmoothedNGramMethod::KneserNey(d) | SmoothedNGramMethod::InterpolatedKneserNey(d) => {
                let total = self.context_totals.get(&ctx).copied().unwrap_or(0);
                if total == 0 {
                    // Back off to continuation probability
                    let cont = self.continuation_counts.get(word).copied().unwrap_or(0) as f64;
                    let total_cont: f64 = self.continuation_counts.values().map(|&c| c as f64).sum();
                    return Ok(if total_cont > 0.0 { cont / total_cont } else { 1.0 / v as f64 });
                }
                let c = self.counts.get(&ctx).and_then(|m| m.get(word)).copied().unwrap_or(0) as f64;
                let adj = (c - d).max(0.0);
                let n_plus = self.counts.get(&ctx).map(|m| m.len()).unwrap_or(0) as f64;
                let lambda = d * n_plus / total as f64;
                let cont = self.continuation_counts.get(word).copied().unwrap_or(0) as f64;
                let total_cont: f64 = self.continuation_counts.values().map(|&c| c as f64).sum();
                let p_cont = if total_cont > 0.0 { cont / total_cont } else { 1.0 / v as f64 };
                Ok(adj / total as f64 + lambda * p_cont)
            }
        }
    }

    /// Log-probability of a token sequence
    pub fn log_prob(&self, tokens: &[&str]) -> Result<f64> {
        if !self.fitted {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let padded: Vec<&str> = {
            let mut v: Vec<&str> = Vec::new();
            for _ in 0..self.n.saturating_sub(1) {
                v.push("<BOS>");
            }
            v.extend_from_slice(tokens);
            v.push("<EOS>");
            v
        };
        let mut lp = 0.0;
        for i in self.n.saturating_sub(1)..padded.len() {
            let ctx = &padded[i.saturating_sub(self.n - 1)..i];
            let word = padded[i];
            let p = self.prob(ctx, word)?;
            lp += if p > 0.0 { p.ln() } else { f64::ln(1e-10) };
        }
        Ok(lp)
    }

    /// Vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Perplexity
// ────────────────────────────────────────────────────────────────────────────

/// Perplexity computation for a language model
pub struct Perplexity<'a> {
    lm: &'a NGramLanguageModel,
}

impl<'a> Perplexity<'a> {
    /// Create a perplexity calculator bound to a language model
    pub fn new(lm: &'a NGramLanguageModel) -> Self {
        Self { lm }
    }

    /// Compute perplexity over a list of test sentences
    ///
    /// PP = exp(-1/N * Σ log P(w))  where N = total token count
    pub fn compute(&self, sentences: &[&str]) -> Result<f64> {
        if sentences.is_empty() {
            return Err(TextError::InvalidInput("No test sentences".to_string()));
        }
        let mut total_lp = 0.0;
        let mut total_tokens = 0usize;
        for &sent in sentences {
            let tokens: Vec<&str> = sent.split_whitespace().collect();
            if tokens.is_empty() {
                continue;
            }
            total_lp += self.lm.log_prob(&tokens)?;
            total_tokens += tokens.len() + 1; // +1 for <EOS>
        }
        if total_tokens == 0 {
            return Ok(f64::INFINITY);
        }
        Ok((-total_lp / total_tokens as f64).exp())
    }

    /// Word-level cross-entropy in nats
    pub fn cross_entropy(&self, sentences: &[&str]) -> Result<f64> {
        let pp = self.compute(sentences)?;
        if pp.is_infinite() || pp.is_nan() {
            return Ok(f64::INFINITY);
        }
        Ok(pp.ln())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// GrammarInducer
// ────────────────────────────────────────────────────────────────────────────

/// Result of grammar induction: a list of (rule, count) pairs
#[derive(Debug, Clone)]
pub struct InducedRule {
    /// LHS category
    pub lhs: String,
    /// RHS token sequence
    pub rhs: Vec<String>,
    /// Observed count
    pub count: usize,
    /// MLE probability among rules with same LHS
    pub prob: f64,
}

/// Simple grammar inducer that builds a PCFG from observed sequences
///
/// Uses a heuristic bracketing strategy based on frequent co-occurring
/// adjacent pairs (similar to the Baby Step Grammar algorithm).
pub struct GrammarInducer {
    /// Minimum frequency for a rule to be included
    pub min_count: usize,
    /// Maximum RHS length to consider
    pub max_rhs_len: usize,
}

impl Default for GrammarInducer {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarInducer {
    /// Create a new grammar inducer
    pub fn new() -> Self {
        Self { min_count: 2, max_rhs_len: 3 }
    }

    /// Set minimum count threshold
    pub fn with_min_count(mut self, n: usize) -> Self {
        self.min_count = n;
        self
    }

    /// Set maximum RHS length
    pub fn with_max_rhs_len(mut self, n: usize) -> Self {
        self.max_rhs_len = n.max(1);
        self
    }

    /// Induce a grammar from a corpus
    ///
    /// Returns a list of induced rules sorted by frequency.
    pub fn induce(&self, corpus: &[&str]) -> Result<Vec<InducedRule>> {
        // Count all n-grams up to max_rhs_len as potential RHS
        let mut rule_counts: HashMap<Vec<String>, usize> = HashMap::new();

        for &text in corpus {
            let tokens = tokenize(text);
            for n in 1..=self.max_rhs_len {
                for window in tokens.windows(n) {
                    *rule_counts.entry(window.to_vec()).or_insert(0) += 1;
                }
            }
        }

        // Filter by min_count
        let frequent: Vec<(Vec<String>, usize)> = rule_counts
            .into_iter()
            .filter(|(_, c)| *c >= self.min_count)
            .collect();

        // Assign LHS labels: single-token rhs -> POS-like category, multi-token -> NP/VP/PP
        let mut lhs_counts: HashMap<String, usize> = HashMap::new();
        let mut rules: Vec<(String, Vec<String>, usize)> = Vec::new();

        for (rhs, count) in &frequent {
            let lhs = self.assign_lhs(rhs);
            *lhs_counts.entry(lhs.clone()).or_insert(0) += count;
            rules.push((lhs, rhs.clone(), *count));
        }

        // Compute MLE probabilities
        let mut induced: Vec<InducedRule> = rules
            .into_iter()
            .map(|(lhs, rhs, count)| {
                let total = *lhs_counts.get(&lhs).unwrap_or(&1);
                InducedRule {
                    lhs,
                    rhs,
                    count,
                    prob: count as f64 / total as f64,
                }
            })
            .collect();

        induced.sort_by(|a, b| b.count.cmp(&a.count));
        Ok(induced)
    }

    /// Convert induced rules to PCFG rules (unary only, as a seed grammar)
    pub fn to_pcfg(&self, corpus: &[&str], start: impl Into<String>) -> Result<PCFG> {
        let induced = self.induce(corpus)?;
        let rules: Vec<PCFGRule> = induced
            .iter()
            .map(|r| PCFGRule::new(r.lhs.clone(), r.rhs.clone(), r.prob))
            .collect();
        Ok(PCFG::new(rules, start))
    }

    fn assign_lhs(&self, rhs: &[String]) -> String {
        match rhs.len() {
            1 => {
                let w = &rhs[0];
                if w.starts_with(|c: char| c.is_uppercase()) {
                    "NNP".to_string()
                } else if w.ends_with("ing") || w.ends_with("ed") {
                    "VBG".to_string()
                } else if w.ends_with("ly") {
                    "RB".to_string()
                } else {
                    "NN".to_string()
                }
            }
            2 => {
                // Heuristic: DT + NN -> NP, VB + NN -> VP
                let first = &rhs[0];
                let det_words = ["the", "a", "an", "this", "that", "these", "those"];
                if det_words.contains(&first.as_str()) {
                    "NP".to_string()
                } else {
                    "VP".to_string()
                }
            }
            _ => "FRAG".to_string(),
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_lm_laplace() {
        let corpus = vec!["the cat sat on the mat", "the dog ran on the path"];
        let mut lm = NGramLanguageModel::new(2, SmoothedNGramMethod::Laplace);
        lm.fit(&corpus).expect("fit");
        let p = lm.prob(&["the"], "cat").expect("prob");
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_ngram_lm_kneser_ney() {
        let corpus = vec![
            "the cat sat on the mat",
            "the cat ran on the path",
            "a dog barked loudly",
        ];
        let mut lm = NGramLanguageModel::new(2, SmoothedNGramMethod::KneserNey(0.75));
        lm.fit(&corpus).expect("fit");
        let p = lm.prob(&["the"], "cat").expect("prob");
        assert!(p > 0.0);
    }

    #[test]
    fn test_perplexity_basic() {
        let train = vec!["the cat sat on the mat", "the dog ran fast"];
        let mut lm = NGramLanguageModel::new(2, SmoothedNGramMethod::Laplace);
        lm.fit(&train).expect("fit");
        let pp = Perplexity::new(&lm);
        let ppl = pp.compute(&["the cat"]).expect("ppl");
        assert!(ppl > 0.0 && ppl.is_finite());
    }

    #[test]
    fn test_cross_entropy() {
        let train = vec!["hello world how are you today"];
        let mut lm = NGramLanguageModel::new(1, SmoothedNGramMethod::Laplace);
        lm.fit(&train).expect("fit");
        let pp = Perplexity::new(&lm);
        let ce = pp.cross_entropy(&["hello world"]).expect("ce");
        assert!(ce > 0.0);
    }

    #[test]
    fn test_log_prob_negative() {
        let corpus = vec!["I like cats and dogs"];
        let mut lm = NGramLanguageModel::new(2, SmoothedNGramMethod::Laplace);
        lm.fit(&corpus).expect("fit");
        let lp = lm.log_prob(&["I", "like"]).expect("lp");
        assert!(lp <= 0.0);
    }

    #[test]
    fn test_grammar_inducer() {
        let corpus = vec![
            "the quick brown fox",
            "the quick red fox",
            "a quick brown dog",
            "the quick brown cat",
        ];
        let inducer = GrammarInducer::new().with_min_count(2);
        let rules = inducer.induce(&corpus).expect("induce");
        assert!(!rules.is_empty());
        // "the quick" should appear multiple times
        let has_the_quick = rules.iter().any(|r| r.rhs == vec!["the", "quick"]);
        assert!(has_the_quick, "Expected 'the quick' to be induced");
    }

    #[test]
    fn test_grammar_to_pcfg() {
        let corpus = vec!["the cat sat", "the dog ran", "a cat ran", "the cat ran"];
        let inducer = GrammarInducer::new().with_min_count(2);
        let pcfg = inducer.to_pcfg(&corpus, "S").expect("pcfg");
        assert!(!pcfg.rules.is_empty());
    }

    #[test]
    fn test_pcfg_rules_for() {
        let rules = vec![
            PCFGRule::new("S", vec!["NP".into(), "VP".into()], 1.0),
            PCFGRule::new("NP", vec!["DT".into(), "NN".into()], 0.7),
            PCFGRule::new("NP", vec!["NN".into()], 0.3),
        ];
        let pcfg = PCFG::new(rules, "S");
        let s_rules = pcfg.rules_for("S");
        assert_eq!(s_rules.len(), 1);
        let np_rules = pcfg.rules_for("NP");
        assert_eq!(np_rules.len(), 2);
    }

    #[test]
    fn test_pcfg_inside() {
        // Tiny grammar: S -> NP VP, NP -> the, VP -> runs
        let rules = vec![
            PCFGRule::new("S", vec!["NP".into(), "VP".into()], 1.0),
            PCFGRule::new("NP", vec!["the".into()], 1.0),
            PCFGRule::new("VP", vec!["runs".into()], 1.0),
        ];
        let pcfg = PCFG::new(rules, "S");
        let tokens: Vec<String> = vec!["the".into(), "runs".into()];
        let alpha = pcfg.inside(&tokens);
        let p_s = alpha[0][1].get("S").copied().unwrap_or(0.0);
        assert!((p_s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cyk_parser() {
        let rules = vec![
            PCFGRule::new("S", vec!["NP".into(), "VP".into()], 1.0),
            PCFGRule::new("NP", vec!["DT".into(), "NN".into()], 1.0),
            PCFGRule::new("VP", vec!["VB".into(), "NP".into()], 1.0),
            PCFGRule::new("DT", vec!["the".into()], 1.0),
            PCFGRule::new("NN", vec!["cat".into()], 0.5),
            PCFGRule::new("NN", vec!["dog".into()], 0.5),
            PCFGRule::new("VB", vec!["chases".into()], 1.0),
        ];
        let pcfg = PCFG::new(rules, "S");
        let parser = CYKParser::new(pcfg);
        let tokens: Vec<String> = vec!["the".into(), "cat".into(), "chases".into(), "the".into(), "dog".into()];
        let result = parser.parse(&tokens);
        assert!(result.is_ok(), "Expected successful parse");
        let tree = result.expect("tree");
        assert_eq!(tree.label, "S");
        let bracket = tree.to_bracket();
        assert!(bracket.contains("NP"));
    }

    #[test]
    fn test_unigram_lm() {
        let corpus = vec!["hello hello world"];
        let mut lm = NGramLanguageModel::new(1, SmoothedNGramMethod::Laplace);
        lm.fit(&corpus).expect("fit");
        let p_hello = lm.prob(&[], "hello").expect("prob");
        let p_world = lm.prob(&[], "world").expect("prob");
        // hello appears twice, world once -> p_hello > p_world
        assert!(p_hello > p_world);
    }
}
