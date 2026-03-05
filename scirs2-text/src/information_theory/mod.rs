//! Information-Theoretic Text Analysis
//!
//! This module provides information-theoretic measures for text analysis,
//! including entropy computation, mutual information, pointwise MI,
//! KL/JS divergences, information content, and minimum description length.
//!
//! ## Overview
//!
//! - [`TextEntropy`]: Character/word/n-gram entropy computation
//! - [`MutualInformation`]: Word pair MI for collocation detection
//! - [`PointwiseMI`]: PMI and PPMI for word co-occurrence matrices
//! - [`KLDivergence`]: KL divergence between document distributions
//! - [`JensenShannon`]: JS divergence for document comparison
//! - [`InformationContent`]: IC measures for words (Resnik-style)
//! - [`MinimumDescriptionLength`]: MDL for text segmentation
//!
//! ## Example
//!
//! ```rust
//! use scirs2_text::information_theory::{TextEntropy, MutualInformation, PointwiseMI};
//!
//! // Character entropy
//! let entropy = TextEntropy::new();
//! let h = entropy.character_entropy("hello world").expect("entropy ok");
//! println!("Char entropy = {:.4}", h);
//!
//! // PMI for word pairs
//! let corpus = vec!["new york city", "new city hall", "york city"];
//! let mut pmi = PointwiseMI::new(2);
//! pmi.fit(&corpus).expect("fit ok");
//! let pmi_val = pmi.pmi("new", "york").expect("pmi ok");
//! println!("PMI(new, york) = {:.4}", pmi_val);
//! ```

use crate::error::{Result, TextError};
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Helper: tokenise a string into lowercase alphabetic tokens
// ────────────────────────────────────────────────────────────────────────────

fn simple_tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Build a probability distribution over an iterator of keys
fn freq_to_prob<I: Iterator<Item = K>, K: std::hash::Hash + Eq>(
    iter: I,
) -> HashMap<K, f64> {
    let mut counts: HashMap<K, usize> = HashMap::new();
    let mut total = 0usize;
    for k in iter {
        *counts.entry(k).or_insert(0) += 1;
        total += 1;
    }
    if total == 0 {
        return HashMap::new();
    }
    counts.into_iter().map(|(k, c)| (k, c as f64 / total as f64)).collect()
}

/// Compute Shannon entropy H = -Σ p log₂ p
fn shannon_entropy(probs: impl Iterator<Item = f64>) -> f64 {
    probs.fold(0.0_f64, |acc, p| {
        if p > 0.0 {
            acc - p * p.log2()
        } else {
            acc
        }
    })
}

// ────────────────────────────────────────────────────────────────────────────
// TextEntropy
// ────────────────────────────────────────────────────────────────────────────

/// Text entropy measurements at character, word, and n-gram levels
#[derive(Debug, Clone, Default)]
pub struct TextEntropy {
    /// Include Unicode code-point analysis
    pub unicode_mode: bool,
}

impl TextEntropy {
    /// Create a new TextEntropy analyser
    pub fn new() -> Self {
        Self { unicode_mode: false }
    }

    /// Enable Unicode mode (entropy over code points rather than ASCII bytes)
    pub fn with_unicode(mut self, v: bool) -> Self {
        self.unicode_mode = v;
        self
    }

    /// Compute Shannon entropy over character distribution in bits
    pub fn character_entropy(&self, text: &str) -> Result<f64> {
        if text.is_empty() {
            return Err(TextError::InvalidInput("Empty text for entropy".to_string()));
        }
        let probs = freq_to_prob(text.chars());
        Ok(shannon_entropy(probs.into_values()))
    }

    /// Compute Shannon entropy over word unigrams in bits
    pub fn word_entropy(&self, text: &str) -> Result<f64> {
        let tokens = simple_tokenize(text);
        if tokens.is_empty() {
            return Err(TextError::InvalidInput("No words found".to_string()));
        }
        let probs = freq_to_prob(tokens.into_iter());
        Ok(shannon_entropy(probs.into_values()))
    }

    /// Compute conditional entropy H(w_t | w_{t-1}) using bigrams
    pub fn bigram_conditional_entropy(&self, text: &str) -> Result<f64> {
        let tokens = simple_tokenize(text);
        if tokens.len() < 2 {
            return Err(TextError::InvalidInput(
                "Need at least 2 tokens for bigram entropy".to_string(),
            ));
        }
        // Count unigrams and bigrams
        let mut uni: HashMap<String, usize> = HashMap::new();
        let mut bi: HashMap<(String, String), usize> = HashMap::new();
        let total = tokens.len() - 1;
        for i in 0..total {
            *uni.entry(tokens[i].clone()).or_insert(0) += 1;
            *bi.entry((tokens[i].clone(), tokens[i + 1].clone())).or_insert(0) += 1;
        }
        *uni.entry(tokens[total].clone()).or_insert(0) += 1;
        let uni_total: usize = uni.values().sum();
        let bi_total = total;
        if bi_total == 0 {
            return Ok(0.0);
        }
        // H(W_t | W_{t-1}) = -Σ P(w,w') log P(w'|w)
        let mut h = 0.0;
        for ((w, w2), &bc) in &bi {
            let p_bigram = bc as f64 / bi_total as f64;
            let p_unigram = *uni.get(w).unwrap_or(&1) as f64 / uni_total as f64;
            let p_cond = bc as f64 / (*uni.get(w).unwrap_or(&1)) as f64;
            if p_cond > 0.0 {
                h -= p_bigram * p_cond.log2();
            }
            let _ = (p_unigram, w2);
        }
        Ok(h)
    }

    /// Compute n-gram entropy over sliding windows of length `n`
    pub fn ngram_entropy(&self, text: &str, n: usize) -> Result<f64> {
        if n == 0 {
            return Err(TextError::InvalidInput("n must be at least 1".to_string()));
        }
        let tokens = simple_tokenize(text);
        if tokens.len() < n {
            return Err(TextError::InvalidInput(format!(
                "Text has fewer than {} tokens",
                n
            )));
        }
        let ngrams: Vec<Vec<String>> = tokens.windows(n).map(|w| w.to_vec()).collect();
        let probs = freq_to_prob(ngrams.into_iter());
        Ok(shannon_entropy(probs.into_values()))
    }

    /// Compute byte-level entropy (useful for compression ratio estimation)
    pub fn byte_entropy(&self, text: &str) -> Result<f64> {
        if text.is_empty() {
            return Err(TextError::InvalidInput("Empty text".to_string()));
        }
        let probs = freq_to_prob(text.bytes());
        Ok(shannon_entropy(probs.into_values()))
    }

    /// Compression ratio estimate: H(text) / log2(256)
    /// Values near 1.0 indicate highly compressed (random-like) text.
    pub fn estimated_compression_ratio(&self, text: &str) -> Result<f64> {
        let h = self.byte_entropy(text)?;
        Ok(h / 8.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MutualInformation
// ────────────────────────────────────────────────────────────────────────────

/// Mutual information between word pairs, used for collocation detection
#[derive(Debug, Clone)]
pub struct MutualInformation {
    /// Window size for co-occurrence counting
    window_size: usize,
    /// Word unigram counts
    word_counts: HashMap<String, usize>,
    /// Word pair co-occurrence counts (within window)
    pair_counts: HashMap<(String, String), usize>,
    /// Total number of word tokens
    total_tokens: usize,
    /// Total number of co-occurrence observations
    total_pairs: usize,
}

impl MutualInformation {
    /// Create a new MutualInformation model
    ///
    /// # Arguments
    ///
    /// * `window_size` – half-window; words within `window_size` positions
    ///   on either side of the target are considered co-occurrences.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(1),
            word_counts: HashMap::new(),
            pair_counts: HashMap::new(),
            total_tokens: 0,
            total_pairs: 0,
        }
    }

    /// Fit the model on a corpus of text slices
    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        self.word_counts.clear();
        self.pair_counts.clear();
        self.total_tokens = 0;
        self.total_pairs = 0;

        for text in corpus {
            let tokens = simple_tokenize(text);
            self.total_tokens += tokens.len();
            for tok in &tokens {
                *self.word_counts.entry(tok.clone()).or_insert(0) += 1;
            }
            for i in 0..tokens.len() {
                let w1 = &tokens[i];
                let end = (i + 1 + self.window_size).min(tokens.len());
                for j in (i + 1)..end {
                    let w2 = &tokens[j];
                    // Canonical order
                    let pair = if w1 <= w2 {
                        (w1.clone(), w2.clone())
                    } else {
                        (w2.clone(), w1.clone())
                    };
                    *self.pair_counts.entry(pair).or_insert(0) += 1;
                    self.total_pairs += 1;
                }
            }
        }

        if self.total_tokens == 0 {
            return Err(TextError::InvalidInput("Empty corpus".to_string()));
        }
        Ok(())
    }

    /// Compute mutual information MI(w1; w2) in bits
    ///
    /// Uses: MI = Σ P(x,y) log(P(x,y) / (P(x)P(y)))
    /// with four-fold contingency table.
    pub fn mi(&self, w1: &str, w2: &str) -> Result<f64> {
        if self.total_tokens == 0 {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let n = self.total_tokens as f64;
        let c1 = *self.word_counts.get(w1).unwrap_or(&0) as f64;
        let c2 = *self.word_counts.get(w2).unwrap_or(&0) as f64;
        let key = if w1 <= w2 {
            (w1.to_string(), w2.to_string())
        } else {
            (w2.to_string(), w1.to_string())
        };
        let c12 = *self.pair_counts.get(&key).unwrap_or(&0) as f64;

        if c1 == 0.0 || c2 == 0.0 {
            return Ok(0.0);
        }

        // Approximate MI via the "positive" cell of the 2×2 contingency table
        let p11 = c12 / n;
        let p10 = (c1 - c12).max(0.0) / n;
        let p01 = (c2 - c12).max(0.0) / n;
        let p00 = (n - c1 - c2 + c12).max(0.0) / n;
        let p1_ = c1 / n;
        let p0_ = 1.0 - p1_;
        let p_1 = c2 / n;
        let p_0 = 1.0 - p_1;

        let mut mi = 0.0_f64;
        if p11 > 0.0 && p1_ > 0.0 && p_1 > 0.0 {
            mi += p11 * (p11 / (p1_ * p_1)).log2();
        }
        if p10 > 0.0 && p1_ > 0.0 && p_0 > 0.0 {
            mi += p10 * (p10 / (p1_ * p_0)).log2();
        }
        if p01 > 0.0 && p0_ > 0.0 && p_1 > 0.0 {
            mi += p01 * (p01 / (p0_ * p_1)).log2();
        }
        if p00 > 0.0 && p0_ > 0.0 && p_0 > 0.0 {
            mi += p00 * (p00 / (p0_ * p_0)).log2();
        }

        Ok(mi)
    }

    /// Return the top `k` collocations by MI score
    pub fn top_collocations(&self, k: usize) -> Result<Vec<((String, String), f64)>> {
        if self.total_tokens == 0 {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let mut scored: Vec<((String, String), f64)> = self
            .pair_counts
            .keys()
            .filter_map(|pair| {
                self.mi(&pair.0, &pair.1).ok().map(|s| (pair.clone(), s))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored.into_iter().take(k).collect())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// PointwiseMI – PMI and PPMI
// ────────────────────────────────────────────────────────────────────────────

/// Pointwise Mutual Information (PMI) and Positive PMI (PPMI) for
/// word co-occurrence matrices
#[derive(Debug, Clone)]
pub struct PointwiseMI {
    /// Context window radius
    window: usize,
    /// Word-context co-occurrence counts
    cooc: HashMap<(String, String), usize>,
    /// Word marginal counts (left)
    word_counts: HashMap<String, usize>,
    /// Context marginal counts (right)
    context_counts: HashMap<String, usize>,
    /// Total number of co-occurrence events
    total: usize,
}

impl PointwiseMI {
    /// Create a new PointwiseMI model
    pub fn new(window: usize) -> Self {
        Self {
            window: window.max(1),
            cooc: HashMap::new(),
            word_counts: HashMap::new(),
            context_counts: HashMap::new(),
            total: 0,
        }
    }

    /// Fit on a corpus
    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        self.cooc.clear();
        self.word_counts.clear();
        self.context_counts.clear();
        self.total = 0;

        for text in corpus {
            let tokens = simple_tokenize(text);
            for i in 0..tokens.len() {
                let w = tokens[i].clone();
                let start = if i >= self.window { i - self.window } else { 0 };
                let end = (i + self.window + 1).min(tokens.len());
                for j in start..end {
                    if i == j {
                        continue;
                    }
                    let ctx = tokens[j].clone();
                    *self.cooc.entry((w.clone(), ctx.clone())).or_insert(0) += 1;
                    *self.word_counts.entry(w.clone()).or_insert(0) += 1;
                    *self.context_counts.entry(ctx).or_insert(0) += 1;
                    self.total += 1;
                }
            }
        }

        if self.total == 0 {
            return Err(TextError::InvalidInput("Empty corpus".to_string()));
        }
        Ok(())
    }

    /// Compute PMI(w, c) = log2( P(w,c) / (P(w) * P(c)) )
    pub fn pmi(&self, word: &str, context: &str) -> Result<f64> {
        if self.total == 0 {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let n = self.total as f64;
        let p_wc = *self.cooc.get(&(word.to_string(), context.to_string())).unwrap_or(&0) as f64 / n;
        let p_w = *self.word_counts.get(word).unwrap_or(&0) as f64 / n;
        let p_c = *self.context_counts.get(context).unwrap_or(&0) as f64 / n;

        if p_wc == 0.0 || p_w == 0.0 || p_c == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        Ok((p_wc / (p_w * p_c)).log2())
    }

    /// Compute PPMI(w, c) = max(0, PMI(w, c))
    pub fn ppmi(&self, word: &str, context: &str) -> Result<f64> {
        Ok(self.pmi(word, context)?.max(0.0))
    }

    /// Build a PPMI matrix for the top `vocab_size` most frequent words
    ///
    /// Returns (words, matrix) where matrix[i][j] = PPMI(words[i], words[j])
    pub fn ppmi_matrix(&self, vocab_size: usize) -> Result<(Vec<String>, Vec<Vec<f64>>)> {
        if self.total == 0 {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        // Select top-k words by frequency
        let mut word_freq: Vec<(String, usize)> = self.word_counts.iter().map(|(w, &c)| (w.clone(), c)).collect();
        word_freq.sort_by(|a, b| b.1.cmp(&a.1));
        let words: Vec<String> = word_freq.into_iter().take(vocab_size).map(|(w, _)| w).collect();

        let k = words.len();
        let mut matrix = vec![vec![0.0_f64; k]; k];
        for (i, w) in words.iter().enumerate() {
            for (j, c) in words.iter().enumerate() {
                if i != j {
                    matrix[i][j] = self.ppmi(w, c).unwrap_or(0.0);
                }
            }
        }
        Ok((words, matrix))
    }

    /// Number of unique word types seen
    pub fn vocab_size(&self) -> usize {
        self.word_counts.len()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// KLDivergence
// ────────────────────────────────────────────────────────────────────────────

/// KL divergence between document word distributions
pub struct KLDivergence {
    /// Smoothing epsilon to avoid log(0)
    pub epsilon: f64,
}

impl Default for KLDivergence {
    fn default() -> Self {
        Self::new()
    }
}

impl KLDivergence {
    /// Create with default epsilon = 1e-10
    pub fn new() -> Self {
        Self { epsilon: 1e-10 }
    }

    /// Create with custom smoothing epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Compute KL(P || Q) from raw text
    ///
    /// The distributions P and Q are word unigram distributions derived from
    /// `text_p` and `text_q` respectively.  Laplace-style `epsilon` smoothing
    /// is applied to avoid undefined log(0).
    pub fn from_texts(&self, text_p: &str, text_q: &str) -> Result<f64> {
        let p = self.text_to_distribution(text_p)?;
        let q = self.text_to_distribution(text_q)?;
        self.kl_divergence(&p, &q)
    }

    /// Compute KL(P || Q) from pre-computed probability distributions
    pub fn kl_divergence(
        &self,
        p: &HashMap<String, f64>,
        q: &HashMap<String, f64>,
    ) -> Result<f64> {
        // Collect all keys
        let mut keys: std::collections::HashSet<String> = std::collections::HashSet::new();
        keys.extend(p.keys().cloned());
        keys.extend(q.keys().cloned());

        let mut kl = 0.0_f64;
        for key in &keys {
            let p_val = p.get(key).copied().unwrap_or(0.0) + self.epsilon;
            let q_val = q.get(key).copied().unwrap_or(0.0) + self.epsilon;
            kl += p_val * (p_val / q_val).ln();
        }
        Ok(kl)
    }

    /// Convert text into a normalised word distribution
    pub fn text_to_distribution(&self, text: &str) -> Result<HashMap<String, f64>> {
        let tokens = simple_tokenize(text);
        if tokens.is_empty() {
            return Err(TextError::InvalidInput("Empty text for distribution".to_string()));
        }
        let dist = freq_to_prob(tokens.into_iter());
        Ok(dist)
    }

    /// Symmetrised KL: (KL(P||Q) + KL(Q||P)) / 2
    pub fn symmetric_kl(
        &self,
        p: &HashMap<String, f64>,
        q: &HashMap<String, f64>,
    ) -> Result<f64> {
        let kl_pq = self.kl_divergence(p, q)?;
        let kl_qp = self.kl_divergence(q, p)?;
        Ok((kl_pq + kl_qp) / 2.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// JensenShannon
// ────────────────────────────────────────────────────────────────────────────

/// Jensen-Shannon divergence for document comparison
pub struct JensenShannon {
    kl: KLDivergence,
}

impl Default for JensenShannon {
    fn default() -> Self {
        Self::new()
    }
}

impl JensenShannon {
    /// Create a new JensenShannon calculator
    pub fn new() -> Self {
        Self { kl: KLDivergence::new() }
    }

    /// Compute JSD(P, Q) in nats
    ///
    /// JSD(P || Q) = (KL(P || M) + KL(Q || M)) / 2  where M = (P+Q)/2
    pub fn from_texts(&self, text_p: &str, text_q: &str) -> Result<f64> {
        let p = self.kl.text_to_distribution(text_p)?;
        let q = self.kl.text_to_distribution(text_q)?;
        self.jsd(&p, &q)
    }

    /// Compute JSD from pre-computed distributions
    pub fn jsd(
        &self,
        p: &HashMap<String, f64>,
        q: &HashMap<String, f64>,
    ) -> Result<f64> {
        // M = mixture distribution (P + Q) / 2
        let mut m: HashMap<String, f64> = HashMap::new();
        for (k, v) in p {
            *m.entry(k.clone()).or_insert(0.0) += v / 2.0;
        }
        for (k, v) in q {
            *m.entry(k.clone()).or_insert(0.0) += v / 2.0;
        }

        let kl_pm = self.kl.kl_divergence(p, &m)?;
        let kl_qm = self.kl.kl_divergence(q, &m)?;
        Ok((kl_pm + kl_qm) / 2.0)
    }

    /// JSD-based similarity score in [0, 1]  (1 = identical)
    pub fn similarity(&self, text_p: &str, text_q: &str) -> Result<f64> {
        let jsd = self.from_texts(text_p, text_q)?;
        // JSD ∈ [0, ln 2] (in nats). Normalise to [0,1] then invert.
        let max_jsd = std::f64::consts::LN_2;
        Ok(1.0 - (jsd / max_jsd).clamp(0.0, 1.0))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// InformationContent
// ────────────────────────────────────────────────────────────────────────────

/// Information Content measures for words
///
/// Based on Resnik (1995): IC(c) = -log P(c)
/// Requires a corpus to estimate word probabilities.
#[derive(Debug, Clone)]
pub struct InformationContent {
    /// Log-probability for each word (base-e)
    log_probs: HashMap<String, f64>,
    /// Vocabulary size
    vocab_size: usize,
    /// Smoothing factor
    smoothing: f64,
}

impl InformationContent {
    /// Create a new InformationContent model
    pub fn new() -> Self {
        Self {
            log_probs: HashMap::new(),
            vocab_size: 0,
            smoothing: 1.0,
        }
    }

    /// Set the Laplace smoothing count
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Fit on a corpus
    pub fn fit(&mut self, corpus: &[&str]) -> Result<()> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for text in corpus {
            for tok in simple_tokenize(text) {
                *counts.entry(tok).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return Err(TextError::InvalidInput("Empty corpus".to_string()));
        }
        self.vocab_size = counts.len();
        let n = total as f64;
        let v = self.vocab_size as f64;
        let s = self.smoothing;
        self.log_probs = counts
            .iter()
            .map(|(w, &c)| {
                let p = (c as f64 + s) / (n + s * v);
                (w.clone(), p.ln())
            })
            .collect();
        Ok(())
    }

    /// Information content of a word: IC(w) = -log P(w)
    pub fn ic(&self, word: &str) -> Result<f64> {
        if self.vocab_size == 0 {
            return Err(TextError::ModelNotFitted("Model not fitted".to_string()));
        }
        let lp = self.log_probs.get(word).copied().unwrap_or_else(|| {
            // Unseen word: assign smoothed probability
            let n_approx = self.vocab_size as f64;
            (self.smoothing / (n_approx + self.smoothing * n_approx)).ln()
        });
        Ok(-lp)
    }

    /// Resnik similarity between two words:
    /// sim_resnik(w1, w2) = IC(w1) + IC(w2) - 2 * IC(lcs)
    ///
    /// Without a WordNet hierarchy, we approximate the LCS information content
    /// by taking the minimum IC of the two words (conservative lower bound).
    pub fn resnik_similarity(&self, w1: &str, w2: &str) -> Result<f64> {
        let ic1 = self.ic(w1)?;
        let ic2 = self.ic(w2)?;
        // Approximate IC(LCS) = min(IC(w1), IC(w2)) (without ontology)
        let ic_lcs = ic1.min(ic2);
        Ok(ic_lcs)
    }

    /// Lin similarity: sim_lin = 2 * IC(lcs) / (IC(w1) + IC(w2))
    pub fn lin_similarity(&self, w1: &str, w2: &str) -> Result<f64> {
        let ic1 = self.ic(w1)?;
        let ic2 = self.ic(w2)?;
        if ic1 + ic2 == 0.0 {
            return Ok(0.0);
        }
        let ic_lcs = ic1.min(ic2);
        Ok((2.0 * ic_lcs) / (ic1 + ic2))
    }

    /// Return the most informative (rarest) words in the model
    pub fn most_informative(&self, k: usize) -> Vec<(String, f64)> {
        let mut pairs: Vec<(String, f64)> = self
            .log_probs
            .iter()
            .map(|(w, &lp)| (w.clone(), -lp))
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.into_iter().take(k).collect()
    }
}

impl Default for InformationContent {
    fn default() -> Self {
        Self::new()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MinimumDescriptionLength
// ────────────────────────────────────────────────────────────────────────────

/// Minimum Description Length for text segmentation
///
/// Uses the MDL criterion to find optimal segmentation boundaries:
/// the segmentation that minimises L(grammar) + L(data|grammar).
#[derive(Debug, Clone)]
pub struct MinimumDescriptionLength {
    /// Minimum segment length in tokens
    pub min_segment: usize,
    /// Maximum number of segments to consider
    pub max_segments: usize,
}

impl Default for MinimumDescriptionLength {
    fn default() -> Self {
        Self::new()
    }
}

impl MinimumDescriptionLength {
    /// Create a new MDL segmenter
    pub fn new() -> Self {
        Self { min_segment: 1, max_segments: 20 }
    }

    /// Set minimum segment size in tokens
    pub fn with_min_segment(mut self, n: usize) -> Self {
        self.min_segment = n.max(1);
        self
    }

    /// Set maximum number of segments
    pub fn with_max_segments(mut self, n: usize) -> Self {
        self.max_segments = n.max(2);
        self
    }

    /// Compute the MDL cost of a segment (list of tokens)
    ///
    /// Uses the negative log-likelihood of the segment under a uniform
    /// word distribution to approximate description length.
    fn segment_cost(tokens: &[String]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }
        let probs = freq_to_prob(tokens.iter().cloned());
        // H in bits
        let h = shannon_entropy(probs.values().copied());
        // Description length ≈ n * H
        tokens.len() as f64 * h
    }

    /// Segment text into semantically coherent parts using dynamic programming
    ///
    /// Returns a vector of (start, end) token index pairs defining segments
    /// over the tokenised text.
    pub fn segment(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let tokens = simple_tokenize(text);
        let n = tokens.len();
        if n == 0 {
            return Err(TextError::InvalidInput("Empty text for segmentation".to_string()));
        }

        // dp[i] = (best MDL cost to segment tokens[0..i], last split point)
        let mut dp = vec![(f64::INFINITY, 0usize); n + 1];
        dp[0] = (0.0, 0);

        for i in 1..=n {
            for j in (0..i).rev() {
                if i - j < self.min_segment {
                    continue;
                }
                let seg = &tokens[j..i];
                // Grammar cost: log2(number of segments) approximation
                let grammar_cost = (self.max_segments as f64).log2();
                let data_cost = Self::segment_cost(seg);
                let total = dp[j].0 + grammar_cost + data_cost;
                if total < dp[i].0 {
                    dp[i] = (total, j);
                }
            }
        }

        // Reconstruct segments
        let mut segments = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let prev = dp[pos].1;
            segments.push((prev, pos));
            pos = prev;
        }
        segments.reverse();
        Ok(segments)
    }

    /// Compute the MDL score for a given text (lower is better/more compressible)
    pub fn mdl_score(&self, text: &str) -> Result<f64> {
        let segments = self.segment(text)?;
        let tokens = simple_tokenize(text);
        let mut total = 0.0;
        for (start, end) in segments {
            total += Self::segment_cost(&tokens[start..end]);
        }
        Ok(total)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_entropy_uniform() {
        // "abcd" has uniform character distribution -> H = log2(4) = 2.0
        let te = TextEntropy::new();
        let h = te.character_entropy("abcd").expect("ok");
        assert!((h - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_character_entropy_skewed() {
        // "aaaa" has H = 0
        let te = TextEntropy::new();
        let h = te.character_entropy("aaaa").expect("ok");
        assert!(h < 1e-9);
    }

    #[test]
    fn test_word_entropy() {
        let te = TextEntropy::new();
        let h = te.word_entropy("the cat sat on the mat").expect("ok");
        assert!(h > 0.0);
    }

    #[test]
    fn test_ngram_entropy() {
        let te = TextEntropy::new();
        let h = te.ngram_entropy("the quick brown fox jumps", 2).expect("ok");
        assert!(h >= 0.0);
    }

    #[test]
    fn test_mutual_information_fit_and_mi() {
        let corpus = vec!["new york city", "new york times", "city hall new"];
        let mut mi = MutualInformation::new(2);
        mi.fit(&corpus).expect("fit ok");
        let score = mi.mi("new", "york").expect("mi ok");
        assert!(score >= 0.0, "MI should be non-negative for this corpus");
    }

    #[test]
    fn test_top_collocations() {
        let corpus = vec!["new york city", "new york times", "new york"];
        let mut mi = MutualInformation::new(2);
        mi.fit(&corpus).expect("fit");
        let top = mi.top_collocations(3).expect("top");
        assert!(!top.is_empty());
    }

    #[test]
    fn test_pmi_ppmi() {
        let corpus = vec!["the cat sat", "the dog sat", "the cat ran"];
        let mut pmi = PointwiseMI::new(2);
        pmi.fit(&corpus).expect("fit");
        let ppmi = pmi.ppmi("cat", "sat").expect("ppmi");
        assert!(ppmi >= 0.0);
    }

    #[test]
    fn test_ppmi_matrix() {
        let corpus = vec!["the cat sat", "the dog sat"];
        let mut pmi = PointwiseMI::new(2);
        pmi.fit(&corpus).expect("fit");
        let (words, matrix) = pmi.ppmi_matrix(4).expect("matrix");
        assert!(!words.is_empty());
        assert_eq!(matrix.len(), words.len());
    }

    #[test]
    fn test_kl_divergence_identical() {
        let kl = KLDivergence::new();
        let text = "the cat sat on the mat";
        let score = kl.from_texts(text, text).expect("kl");
        // KL(P || P) should be ~0 (small due to epsilon)
        assert!(score < 0.01);
    }

    #[test]
    fn test_kl_divergence_different() {
        let kl = KLDivergence::new();
        let t1 = "cat dog bird fish";
        let t2 = "algebra calculus matrix vector";
        let score = kl.from_texts(t1, t2).expect("kl");
        assert!(score > 0.0);
    }

    #[test]
    fn test_jsd_identical() {
        let js = JensenShannon::new();
        let t = "the quick brown fox";
        let jsd = js.from_texts(t, t).expect("jsd");
        assert!(jsd < 0.01);
    }

    #[test]
    fn test_jsd_similarity() {
        let js = JensenShannon::new();
        let sim = js.similarity("hello world", "hello world foo").expect("sim");
        assert!(sim > 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_information_content_fit() {
        let corpus = vec!["the cat sat on the mat", "the dog ran fast"];
        let mut ic = InformationContent::new();
        ic.fit(&corpus).expect("fit");
        let ic_the = ic.ic("the").expect("ic");
        let ic_rare = ic.ic("sat").expect("ic rare");
        // "the" appears more often -> lower IC
        assert!(ic_the < ic_rare, "Frequent words should have lower IC");
    }

    #[test]
    fn test_resnik_lin_similarity() {
        let corpus = vec!["cat dog animal pet fish bird animal cat cat cat"];
        let mut ic = InformationContent::new();
        ic.fit(&corpus).expect("fit");
        let sim = ic.lin_similarity("cat", "dog").expect("lin");
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_mdl_segmentation() {
        let mdl = MinimumDescriptionLength::new().with_min_segment(2);
        let text = "the quick brown fox jumps over the lazy dog";
        let segments = mdl.segment(text).expect("segment");
        assert!(!segments.is_empty());
        // Segments should cover the full text
        let first_start = segments.first().map(|s| s.0).unwrap_or(0);
        let last_end = segments.last().map(|s| s.1).unwrap_or(0);
        let tokens = simple_tokenize(text);
        assert_eq!(first_start, 0);
        assert_eq!(last_end, tokens.len());
    }

    #[test]
    fn test_bigram_conditional_entropy() {
        let te = TextEntropy::new();
        let h = te.bigram_conditional_entropy("the cat sat on the mat the cat ran").expect("ok");
        assert!(h >= 0.0);
    }
}
