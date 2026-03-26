//! Hierarchical Dirichlet Process (HDP)
//!
//! Implements the Chinese Restaurant Franchise (CRF) Gibbs sampler for HDP
//! mixture models (Teh et al. 2006).  Each group (document) has its own
//! restaurant; dishes are shared globally via a top-level DP.
//!
//! # Algorithm
//! A collapsed Gibbs sampler with the following per-sweep steps:
//! 1. For each word `w` in document `d`, remove its current assignment,
//!    draw a new topic from the conditional posterior, and re-insert.
//! 2. Re-sample the global stick weights `β` via auxiliary CRF table counts
//!    followed by a symmetric Dirichlet draw.
//! 3. Accumulate per-sweep log-likelihoods.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Public enums / types
// ---------------------------------------------------------------------------

/// Inference algorithm selection for HDP.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdpInference {
    /// Collapsed Gibbs sampler via the Chinese Restaurant Franchise.
    Gibbs,
    /// Mean-field Variational EM (not yet implemented; reserved).
    VariationalEM,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the HDP collapsed Gibbs sampler.
#[derive(Debug, Clone)]
pub struct HdpConfig {
    /// Global (top-level) concentration parameter γ > 0.
    pub gamma: f64,
    /// Per-group (document-level) concentration parameter α > 0.
    pub alpha: f64,
    /// Truncation level K (maximum number of topics).
    pub n_topics: usize,
    /// Number of Gibbs sweeps.
    pub n_iter: usize,
    /// Dirichlet smoothing on word likelihoods (η).
    pub eta: f64,
    /// Seed for the internal LCG PRNG.
    pub seed: u64,
}

impl Default for HdpConfig {
    fn default() -> Self {
        Self {
            gamma: 1.0,
            alpha: 1.0,
            n_topics: 20,
            n_iter: 100,
            eta: 0.01,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// State and result types
// ---------------------------------------------------------------------------

/// Running Gibbs state.
#[derive(Debug, Clone)]
pub struct HdpState {
    /// Global topic usage counts (length K).
    pub topic_counts: Vec<usize>,
    /// Per-document topic counts (D × K).
    pub doc_topic_counts: Vec<Vec<usize>>,
    /// Per-document word-to-topic assignments.
    pub assignments: Vec<Vec<usize>>,
    /// Global topic proportions β (length K+1; last entry = "new-topic" mass).
    pub beta: Vec<f64>,
}

/// Output of [`hdp_fit`].
#[derive(Debug, Clone)]
pub struct HdpResult {
    /// Normalised topic–word matrix of shape \[K × V\].
    pub topic_word_matrix: Array2<f64>,
    /// Normalised document–topic matrix of shape \[D × K\].
    pub doc_topic_matrix: Array2<f64>,
    /// Per-sweep log-likelihoods.
    pub log_likelihoods: Vec<f64>,
    /// Number of topics that received at least one word.
    pub n_topics_used: usize,
    /// Final internal Gibbs state (useful for resuming / inspection).
    pub state: HdpState,
}

// ---------------------------------------------------------------------------
// LCG PRNG helpers (no rand crate)
// ---------------------------------------------------------------------------

/// Minimal LCG state used throughout this module.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Splitmix64 warm-up to avoid seed=0 degeneracy.
        let mut s = seed ^ 0x9e37_79b9_7f4a_7c15_u64;
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        s ^= s >> 31;
        Self { state: s }
    }

    /// Returns a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // LCG parameters from Knuth (64-bit).
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Categorical sample: draw an index from an unnormalised weight slice.
    fn categorical(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let u = self.next_f64() * total;
        let mut cum = 0.0_f64;
        for (i, &w) in weights.iter().enumerate() {
            cum += w;
            if u < cum {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Draw a sample from Gamma(a, 1) using Marsaglia–Tsang (a ≥ 1) or
    /// Ahrens–Dieter boost (a < 1).
    fn gamma(&mut self, a: f64) -> f64 {
        if a < 1.0 {
            // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
            let u = self.next_f64();
            return self.gamma(a + 1.0) * u.powf(1.0 / a);
        }
        // Marsaglia–Tsang
        let d = a - 1.0 / 3.0;
        let c = 1.0 / (3.0 * d.sqrt());
        loop {
            let (x, v) = self.normal_and_v(c, d);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_f64();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
    }

    /// Box–Muller normal sample; returns (z, v) where v = (1+c*z)^3.
    fn normal_and_v(&mut self, c: f64, d: f64) -> (f64, f64) {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 <= 0.0 {
                continue;
            }
            let z = ((-2.0 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            let v_inner = 1.0 + c * z;
            if v_inner > 0.0 {
                let v = v_inner * v_inner * v_inner;
                return (z, v);
            }
            // If v_inner <= 0 we just store d as a sentinel so the caller can
            // reject on v <= 0.
            let _ = d;
        }
    }

    /// Sample from Dirichlet(alpha * ones_K) via Gamma draws.
    /// Returns a vector of length `k` summing to 1.
    fn dirichlet_symmetric(&mut self, k: usize, alpha: f64) -> Vec<f64> {
        let mut samples: Vec<f64> = (0..k).map(|_| self.gamma(alpha)).collect();
        let total: f64 = samples.iter().sum();
        if total > 0.0 {
            for s in &mut samples {
                *s /= total;
            }
        } else {
            // Degenerate: place all mass on first component.
            samples[0] = 1.0;
        }
        samples
    }

    /// Sample from Dirichlet with given concentration vector.
    fn dirichlet(&mut self, alphas: &[f64]) -> Vec<f64> {
        let mut samples: Vec<f64> = alphas.iter().map(|&a| self.gamma(a.max(1e-10))).collect();
        let total: f64 = samples.iter().sum();
        if total > 0.0 {
            for s in &mut samples {
                *s /= total;
            }
        } else {
            samples[0] = 1.0;
        }
        samples
    }
}

// ---------------------------------------------------------------------------
// Core fitting routine
// ---------------------------------------------------------------------------

/// Fit an HDP to a document corpus via collapsed Gibbs sampling.
///
/// # Parameters
/// - `documents`: Each inner `Vec<usize>` is a sequence of word indices in
///   `[0, vocab_size)`.
/// - `vocab_size`: Vocabulary size V.
/// - `config`: Sampler configuration.
///
/// # Errors
/// Returns [`StatsError::InvalidArgument`] if `vocab_size == 0`, if any word
/// index is out of range, or if `documents` is empty.
pub fn hdp_fit(
    documents: &[Vec<usize>],
    vocab_size: usize,
    config: &HdpConfig,
) -> StatsResult<HdpResult> {
    // --- Input validation ---------------------------------------------------
    if vocab_size == 0 {
        return Err(StatsError::InvalidArgument(
            "hdp_fit: vocab_size must be > 0".to_string(),
        ));
    }
    if documents.is_empty() {
        return Err(StatsError::InvalidArgument(
            "hdp_fit: documents must not be empty".to_string(),
        ));
    }
    if config.n_topics == 0 {
        return Err(StatsError::InvalidArgument(
            "hdp_fit: n_topics must be > 0".to_string(),
        ));
    }

    // Validate all word indices.
    for (d, doc) in documents.iter().enumerate() {
        for &w in doc {
            if w >= vocab_size {
                return Err(StatsError::InvalidArgument(format!(
                    "hdp_fit: word index {w} in document {d} >= vocab_size {vocab_size}"
                )));
            }
        }
    }

    let k = config.n_topics;
    let d_count = documents.len();
    let eta = config.eta;
    let alpha = config.alpha;
    let gamma = config.gamma;

    let mut rng = Lcg::new(config.seed);

    // --- Initialisation: assign all words to topic 0 -----------------------
    let mut assignments: Vec<Vec<usize>> = documents
        .iter()
        .map(|doc| vec![0usize; doc.len()])
        .collect();

    // topic-word counts: [K × V]
    let mut tw: Vec<Vec<usize>> = vec![vec![0usize; vocab_size]; k];
    // per-topic total word counts: [K]
    let mut topic_totals: Vec<usize> = vec![0usize; k];
    // doc-topic counts: [D × K]
    let mut dt: Vec<Vec<usize>> = vec![vec![0usize; k]; d_count];

    for (d, doc) in documents.iter().enumerate() {
        for &w in doc {
            tw[0][w] += 1;
            topic_totals[0] += 1;
            dt[d][0] += 1;
        }
    }

    // Global topic proportions β: initialise uniform.
    let mut beta: Vec<f64> = vec![1.0 / k as f64; k];

    let mut log_likelihoods: Vec<f64> = Vec::with_capacity(config.n_iter);
    let mut probs: Vec<f64> = vec![0.0; k];

    // --- Gibbs sweeps -------------------------------------------------------
    for _iter in 0..config.n_iter {
        // ---- Word sampling step --------------------------------------------
        for d in 0..d_count {
            let doc_len = documents[d].len();
            for n in 0..doc_len {
                let w = documents[d][n];
                let old_k = assignments[d][n];

                // Remove word from its current topic.
                tw[old_k][w] -= 1;
                topic_totals[old_k] -= 1;
                dt[d][old_k] -= 1;

                // Compute posterior probabilities for each topic k.
                let vocab_f = vocab_size as f64;
                for ki in 0..k {
                    let n_dk = dt[d][ki] as f64;
                    let n_kw = tw[ki][w] as f64;
                    let n_k_total = topic_totals[ki] as f64;
                    // p(z=ki | ...) ∝ (n_{dk} + α·β_ki) * (n_{kw} + η) / (n_k + V·η)
                    let likelihood = (n_kw + eta) / (n_k_total + vocab_f * eta);
                    let prior = n_dk + alpha * beta[ki];
                    probs[ki] = prior * likelihood;
                }

                // Sample new topic.
                let new_k = rng.categorical(&probs[..k]);

                // Re-insert word into new topic.
                assignments[d][n] = new_k;
                tw[new_k][w] += 1;
                topic_totals[new_k] += 1;
                dt[d][new_k] += 1;
            }
        }

        // ---- Re-sample global β via auxiliary CRF table counts -------------
        // For each document d and topic k, sample m_{dk} = number of tables
        // in document d serving dish k.  Each table count is drawn from
        // a Chinese Restaurant Process with n_{dk} customers and base
        // probability proportional to β_k.
        //
        // Simplified (exact) CRF table-count sampler:
        // P(m_{dk} = m | n_{dk}, α·β_k) is a Stirling-number-based distribution.
        // We use the sequential addition sampler (Antoniak 1974):
        // for j = 1..n_{dk}: add a new table with prob α·β_k / (j-1 + α·β_k).
        let mut m_counts: Vec<usize> = vec![0usize; k];
        for d in 0..d_count {
            for ki in 0..k {
                let n_dk = dt[d][ki];
                if n_dk == 0 {
                    continue;
                }
                let abk = alpha * beta[ki];
                let mut tables = 1usize; // at least one table if n_dk > 0
                for j in 1..n_dk {
                    // probability of opening a new table for the (j+1)-th customer
                    let p_new = abk / (j as f64 + abk);
                    if rng.next_f64() < p_new {
                        tables += 1;
                    }
                }
                m_counts[ki] += tables;
            }
        }

        // Sample β from Dirichlet([m_1, ..., m_K, γ]).
        let mut dir_params: Vec<f64> = m_counts.iter().map(|&m| m as f64 + gamma / k as f64).collect();
        // Ensure all params are positive.
        for p in &mut dir_params {
            if *p < 1e-10 {
                *p = 1e-10;
            }
        }
        let new_beta = rng.dirichlet(&dir_params);
        // Drop the last element (new-topic mass is re-normalised into existing topics
        // at the truncation level K for this finite approximation).
        beta.copy_from_slice(&new_beta[..k]);

        // ---- Log-likelihood -------------------------------------------------
        let ll = compute_log_likelihood(documents, &tw, &topic_totals, &dt, &beta, k, vocab_size, eta, alpha);
        log_likelihoods.push(ll);
    }

    // --- Build result matrices -----------------------------------------------
    let mut topic_word_matrix = Array2::<f64>::zeros((k, vocab_size));
    for ki in 0..k {
        let row_sum: f64 = (0..vocab_size).map(|v| tw[ki][v] as f64 + eta).sum();
        for v in 0..vocab_size {
            topic_word_matrix[[ki, v]] = (tw[ki][v] as f64 + eta) / row_sum;
        }
    }

    let mut doc_topic_matrix = Array2::<f64>::zeros((d_count, k));
    for d in 0..d_count {
        let row_sum: f64 = (0..k).map(|ki| dt[d][ki] as f64 + alpha * beta[ki]).sum();
        for ki in 0..k {
            doc_topic_matrix[[d, ki]] = (dt[d][ki] as f64 + alpha * beta[ki]) / row_sum;
        }
    }

    let n_topics_used = topic_totals.iter().filter(|&&c| c > 0).count();

    let state = HdpState {
        topic_counts: topic_totals,
        doc_topic_counts: dt,
        assignments,
        beta,
    };

    Ok(HdpResult {
        topic_word_matrix,
        doc_topic_matrix,
        log_likelihoods,
        n_topics_used,
        state,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the per-document log marginal likelihood of current assignments.
fn compute_log_likelihood(
    documents: &[Vec<usize>],
    tw: &[Vec<usize>],
    topic_totals: &[usize],
    dt: &[Vec<usize>],
    beta: &[f64],
    k: usize,
    vocab_size: usize,
    eta: f64,
    alpha: f64,
) -> f64 {
    let vocab_f = vocab_size as f64;
    let mut ll = 0.0_f64;

    for (d, doc) in documents.iter().enumerate() {
        for &w in doc {
            // Predictive probability: sum_k p(z=k|doc) * p(w|z=k)
            let mut p_w = 0.0_f64;
            // Document-level topic distribution (normalised counts).
            let doc_total: f64 = doc.len() as f64;
            for ki in 0..k {
                let theta_dk = (dt[d][ki] as f64 + alpha * beta[ki])
                    / (doc_total + alpha);
                let phi_kw = (tw[ki][w] as f64 + eta)
                    / (topic_totals[ki] as f64 + vocab_f * eta);
                p_w += theta_dk * phi_kw;
            }
            ll += (p_w.max(1e-300)).ln();
        }
    }
    ll
}

// ---------------------------------------------------------------------------
// Public helper: compute perplexity on held-out documents
// ---------------------------------------------------------------------------

/// Compute per-word perplexity of `documents` given a fitted HDP result.
///
/// Perplexity = exp(-log-likelihood / total_words).
pub fn hdp_perplexity(documents: &[Vec<usize>], result: &HdpResult, config: &HdpConfig) -> f64 {
    let k = result.topic_word_matrix.nrows();
    let vocab_size = result.topic_word_matrix.ncols();
    let eta = config.eta;
    let alpha = config.alpha;
    let beta = &result.state.beta;
    let tw = &result.state;
    let total_words: usize = documents.iter().map(|d| d.len()).sum();
    if total_words == 0 {
        return 1.0;
    }

    let mut ll = 0.0_f64;
    let vocab_f = vocab_size as f64;

    for (d_idx, doc) in documents.iter().enumerate() {
        if doc.is_empty() {
            continue;
        }
        // Use doc-topic distribution from training if available.
        let has_doc = d_idx < result.doc_topic_matrix.nrows();
        let doc_len_f = doc.len() as f64;
        for &w in doc {
            if w >= vocab_size {
                continue;
            }
            let mut p_w = 0.0_f64;
            for ki in 0..k {
                let theta = if has_doc {
                    result.doc_topic_matrix[[d_idx, ki]]
                } else {
                    // Uniform prior for unseen docs.
                    1.0 / k as f64
                };
                let phi = result.topic_word_matrix[[ki, w]];
                p_w += theta * phi;
            }
            let _ = (doc_len_f, alpha, beta, eta, vocab_f, tw);
            ll += (p_w.max(1e-300)).ln();
        }
    }

    (-ll / total_words as f64).exp()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_documents() -> Vec<Vec<usize>> {
        // Two clearly distinct documents: topic A uses words 0-2, topic B uses 3-5
        let doc_a: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0];
        let doc_b: Vec<usize> = vec![3, 4, 5, 3, 4, 5, 3, 4, 5, 3];
        vec![doc_a, doc_b]
    }

    #[test]
    fn test_default_config() {
        let cfg = HdpConfig::default();
        assert!((cfg.gamma - 1.0).abs() < 1e-10);
        assert!((cfg.alpha - 1.0).abs() < 1e-10);
        assert_eq!(cfg.n_topics, 20);
        assert_eq!(cfg.n_iter, 100);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_basic_fit_no_panic() {
        let docs = simple_documents();
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let result = hdp_fit(&docs, 6, &cfg);
        assert!(result.is_ok(), "hdp_fit should succeed: {result:?}");
    }

    #[test]
    fn test_doc_topic_matrix_rows_sum_to_one() {
        let docs = simple_documents();
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let result = hdp_fit(&docs, 6, &cfg).expect("fit failed");
        for d in 0..result.doc_topic_matrix.nrows() {
            let row_sum: f64 = result.doc_topic_matrix.row(d).sum();
            assert!((row_sum - 1.0).abs() < 1e-9, "doc {d} row sum = {row_sum}");
        }
    }

    #[test]
    fn test_topic_word_matrix_rows_sum_to_one() {
        let docs = simple_documents();
        let cfg = HdpConfig { n_topics: 4, n_iter: 10, ..Default::default() };
        let result = hdp_fit(&docs, 6, &cfg).expect("fit failed");
        for k in 0..result.topic_word_matrix.nrows() {
            let row_sum: f64 = result.topic_word_matrix.row(k).sum();
            assert!((row_sum - 1.0).abs() < 1e-9, "topic {k} row sum = {row_sum}");
        }
    }

    #[test]
    fn test_n_topics_used_leq_config() {
        let docs = simple_documents();
        let cfg = HdpConfig { n_topics: 10, n_iter: 20, ..Default::default() };
        let result = hdp_fit(&docs, 6, &cfg).expect("fit failed");
        assert!(result.n_topics_used <= cfg.n_topics);
    }

    #[test]
    fn test_vocab_size_zero_returns_error() {
        let docs = vec![vec![0usize]];
        let cfg = HdpConfig::default();
        assert!(hdp_fit(&docs, 0, &cfg).is_err());
    }

    #[test]
    fn test_empty_documents_returns_error() {
        let cfg = HdpConfig::default();
        let empty: Vec<Vec<usize>> = vec![];
        assert!(hdp_fit(&empty, 5, &cfg).is_err());
    }

    #[test]
    fn test_empty_single_document_handled() {
        // A corpus containing one empty document and one non-empty one.
        let docs = vec![vec![], vec![0usize, 1, 2]];
        let cfg = HdpConfig { n_topics: 3, n_iter: 5, ..Default::default() };
        // Should not panic.
        let result = hdp_fit(&docs, 3, &cfg);
        assert!(result.is_ok());
    }

    #[test]
    fn test_assignments_length_matches_documents() {
        let docs = vec![
            vec![0usize, 1, 2],
            vec![3usize, 4],
            vec![0usize],
        ];
        let cfg = HdpConfig { n_topics: 3, n_iter: 5, ..Default::default() };
        let result = hdp_fit(&docs, 5, &cfg).expect("fit failed");
        for (d, doc) in docs.iter().enumerate() {
            assert_eq!(
                result.state.assignments[d].len(),
                doc.len(),
                "document {d} assignment length mismatch"
            );
        }
    }

    #[test]
    fn test_out_of_range_word_returns_error() {
        let docs = vec![vec![0usize, 99]]; // word 99 >= vocab_size 5
        let cfg = HdpConfig::default();
        assert!(hdp_fit(&docs, 5, &cfg).is_err());
    }

    #[test]
    fn test_log_likelihood_vector_has_correct_length() {
        let docs = simple_documents();
        let n_iter = 15;
        let cfg = HdpConfig { n_topics: 4, n_iter, ..Default::default() };
        let result = hdp_fit(&docs, 6, &cfg).expect("fit failed");
        assert_eq!(result.log_likelihoods.len(), n_iter);
    }

    #[test]
    fn test_large_vocab_small_corpus_no_panic() {
        let docs = vec![vec![0usize, 1, 2], vec![100, 200, 300]];
        let cfg = HdpConfig { n_topics: 5, n_iter: 5, ..Default::default() };
        assert!(hdp_fit(&docs, 1000, &cfg).is_ok());
    }

    #[test]
    fn test_two_distinct_documents_get_different_dominant_topics() {
        // With enough iterations and clearly distinct word sets, the two documents
        // should be dominated by different topics.
        let docs = simple_documents();
        let cfg = HdpConfig {
            n_topics: 4,
            n_iter: 200,
            seed: 7,
            ..Default::default()
        };
        let result = hdp_fit(&docs, 6, &cfg).expect("fit failed");
        let top_0 = argmax(result.doc_topic_matrix.row(0).as_slice().expect("slice failed"));
        let top_1 = argmax(result.doc_topic_matrix.row(1).as_slice().expect("slice failed"));
        assert_ne!(
            top_0, top_1,
            "documents with disjoint vocabulary should prefer different topics"
        );
    }

    #[test]
    fn test_eta_sensitivity() {
        let docs = simple_documents();
        // Two runs with different eta should produce slightly different topic-word matrices.
        let cfg_low = HdpConfig { n_topics: 4, n_iter: 20, eta: 0.001, ..Default::default() };
        let cfg_high = HdpConfig { n_topics: 4, n_iter: 20, eta: 1.0, ..Default::default() };
        let r_low = hdp_fit(&docs, 6, &cfg_low).expect("low eta fit");
        let r_high = hdp_fit(&docs, 6, &cfg_high).expect("high eta fit");
        // High-eta matrix should be closer to uniform (entropy > low-eta entropy).
        let entropy = |m: &Array2<f64>| -> f64 {
            let mut e = 0.0_f64;
            for r in m.rows() {
                for &p in r.iter() {
                    if p > 0.0 {
                        e -= p * p.ln();
                    }
                }
            }
            e
        };
        assert!(
            entropy(&r_high.topic_word_matrix) >= entropy(&r_low.topic_word_matrix),
            "higher eta should increase topic entropy"
        );
    }

    fn argmax(slice: &[f64]) -> usize {
        slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).expect("NaN in argmax"))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}
