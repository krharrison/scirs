//! Probabilistic Latent Semantic Analysis (pLSA)
//!
//! Implements the pLSA model of Hofmann (1999).  Given a co-occurrence matrix
//! `X[d, w]` counting word `w` in document `d`, pLSA fits a latent topic model:
//!
//! ```text
//! P(w, d) = sum_z P(z | d) P(w | z)
//! ```
//!
//! where `z` ranges over `n_topics` latent topics.
//!
//! ## EM Algorithm
//!
//! **E-step** — posterior topic responsibility:
//! ```text
//! Q(z | d, w) = P(z | d) P(w | z) / sum_{z'} P(z' | d) P(w | z')
//! ```
//!
//! **M-step** — update topic-word and document-topic distributions:
//! ```text
//! P(w | z) ∝ sum_d X[d,w] Q(z | d, w)
//! P(z | d) ∝ sum_w X[d,w] Q(z | d, w)
//! ```
//!
//! ## References
//!
//! - Hofmann, T. (1999). Probabilistic latent semantic indexing.
//!   *Proceedings of SIGIR*, pp. 50–57.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::random::{Rng, RngExt};

// ─── constants ────────────────────────────────────────────────────────────────
const EPS: f64 = 1e-10;

// ─────────────────────────────────────────────────────────────────────────────
// PLSAModel
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted pLSA model.
///
/// After fitting, the model stores:
/// - `p_z_d`: `P(z | d)`, shape `(n_docs, n_topics)`.
/// - `p_w_z`: `P(w | z)`, shape `(n_topics, n_words)`.
#[derive(Debug, Clone)]
pub struct PLSAModel {
    /// Document-topic matrix `P(z | d)`, shape `(n_docs, n_topics)`.
    pub p_z_d: Array2<f64>,
    /// Topic-word matrix `P(w | z)`, shape `(n_topics, n_words)`.
    pub p_w_z: Array2<f64>,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
}

impl PLSAModel {
    /// Return the top-`n` word indices for each topic.
    ///
    /// Returns a vector of length `n_topics`, where each element is a vector
    /// of `n` word indices sorted by descending `P(w | z)`.
    pub fn topic_words(&self, top_n: usize) -> Vec<Vec<usize>> {
        let (n_topics, n_words) = (self.p_w_z.nrows(), self.p_w_z.ncols());
        let top_n = top_n.min(n_words);
        let mut result = Vec::with_capacity(n_topics);
        for z in 0..n_topics {
            let mut indexed: Vec<(usize, f64)> = (0..n_words)
                .map(|w| (w, self.p_w_z[[z, w]]))
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            result.push(indexed.into_iter().take(top_n).map(|(w, _)| w).collect());
        }
        result
    }

    /// Return the topic distribution for each document.
    ///
    /// Returns `P(z | d)`, a reference to the `(n_docs, n_topics)` matrix.
    pub fn document_topics(&self) -> &Array2<f64> {
        &self.p_z_d
    }

    /// Compute perplexity on held-out co-occurrence data.
    ///
    /// `x_test` has shape `(n_docs_test, n_words)`.
    ///
    /// Perplexity is defined as `exp(-log_likelihood / N)` where `N` is the
    /// total number of word tokens.
    ///
    /// For documents not seen during training this uses `P(w, d)` marginalised
    /// over the training document-topic mixture — which means it uses the
    /// average topic-word distribution as a background model.  This is an
    /// approximation; for exact inference on new documents use `infer_topics`.
    pub fn perplexity(&self, x_test: &Array2<f64>) -> Result<f64> {
        // Approximate: use marginalised P(w) = (1/Z) sum_z P(w | z) as background
        let p_w = self.marginal_p_w();
        let n_tokens: f64 = x_test.iter().sum();
        if n_tokens == 0.0 {
            return Err(TransformError::InvalidInput("Empty test matrix".into()));
        }
        let mut ll = 0.0;
        let (nd, nw) = (x_test.nrows(), x_test.ncols());
        if nw != self.p_w_z.ncols() {
            return Err(TransformError::DimensionMismatch(
                "Vocabulary size mismatch".into(),
            ));
        }
        for d in 0..nd {
            for w in 0..nw {
                let cnt = x_test[[d, w]];
                if cnt > 0.0 {
                    ll += cnt * (p_w[w] + EPS).ln();
                }
            }
        }
        Ok((-ll / n_tokens).exp())
    }

    /// Infer topic distribution `P(z | d)` for new documents `x_new`.
    ///
    /// Runs EM with `P(w | z)` fixed, updating only `P(z | d_new)`.
    pub fn infer_topics(&self, x_new: &Array2<f64>, max_iter: usize) -> Result<Array2<f64>> {
        let (nd, nw) = (x_new.nrows(), x_new.ncols());
        let n_topics = self.p_w_z.nrows();
        if nw != self.p_w_z.ncols() {
            return Err(TransformError::DimensionMismatch("Vocab size mismatch".into()));
        }
        let mut rng = scirs2_core::random::rng();
        // Initialise P(z | d) uniformly
        let mut p_z_d = Array2::<f64>::zeros((nd, n_topics));
        for d in 0..nd {
            for z in 0..n_topics {
                p_z_d[[d, z]] = rng.gen_range(0.5..1.5);
            }
            let row_sum: f64 = (0..n_topics).map(|z| p_z_d[[d, z]]).sum();
            for z in 0..n_topics {
                p_z_d[[d, z]] /= row_sum + EPS;
            }
        }

        // EM: keep p_w_z fixed, update p_z_d
        let mut q = Array2::<f64>::zeros((n_topics, nw));
        for _ in 0..max_iter {
            // E-step per document
            let mut new_pzd = Array2::<f64>::zeros((nd, n_topics));
            for d in 0..nd {
                for w in 0..nw {
                    let cnt = x_new[[d, w]];
                    if cnt == 0.0 {
                        continue;
                    }
                    let mut denom = 0.0;
                    for z in 0..n_topics {
                        q[[z, w]] = p_z_d[[d, z]] * self.p_w_z[[z, w]];
                        denom += q[[z, w]];
                    }
                    if denom > EPS {
                        for z in 0..n_topics {
                            new_pzd[[d, z]] += cnt * q[[z, w]] / denom;
                        }
                    }
                }
                // Normalise row
                let row_sum: f64 = (0..n_topics).map(|z| new_pzd[[d, z]]).sum();
                for z in 0..n_topics {
                    p_z_d[[d, z]] = new_pzd[[d, z]] / (row_sum + EPS);
                }
            }
        }
        Ok(p_z_d)
    }

    /// Compute marginalised word probability `P(w) = mean_d P(w | d)`.
    fn marginal_p_w(&self) -> Array1<f64> {
        let nw = self.p_w_z.ncols();
        let n_topics = self.p_w_z.nrows();
        let mut p_w = Array1::<f64>::zeros(nw);
        for z in 0..n_topics {
            for w in 0..nw {
                p_w[w] += self.p_w_z[[z, w]] / n_topics as f64;
            }
        }
        p_w
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config and fit_em
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for pLSA EM fitting.
#[derive(Debug, Clone)]
pub struct PLSAConfig {
    /// Number of latent topics.
    pub n_topics: usize,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change.
    pub tol: f64,
    /// Random seed.
    pub seed: u64,
}

impl Default for PLSAConfig {
    fn default() -> Self {
        Self {
            n_topics: 10,
            max_iter: 200,
            tol: 1e-4,
            seed: 42,
        }
    }
}

/// Fit a pLSA model to co-occurrence matrix `X` (shape `n_docs × n_words`).
///
/// `X[d, w]` is the count (or TF weight) of word `w` in document `d`.
///
/// # Errors
///
/// Returns an error if `X` contains negative values, has zero rows or columns,
/// or if `n_topics` is 0.
pub fn fit_em<S>(x: &ArrayBase<S, Ix2>, config: &PLSAConfig) -> Result<PLSAModel>
where
    S: Data<Elem = f64>,
{
    let x = x.to_owned();
    let (nd, nw) = (x.nrows(), x.ncols());
    let nz = config.n_topics;

    if nd == 0 || nw == 0 {
        return Err(TransformError::InvalidInput(
            "Input matrix must have at least one row and one column".into(),
        ));
    }
    if nz == 0 {
        return Err(TransformError::InvalidInput("n_topics must be > 0".into()));
    }
    if x.iter().any(|&v| v < 0.0) {
        return Err(TransformError::InvalidInput(
            "pLSA requires non-negative co-occurrence counts".into(),
        ));
    }

    let mut rng = scirs2_core::random::rng();

    // ── Initialise P(z|d) and P(w|z) with random Dirichlet-like values ────────
    let mut p_z_d = Array2::<f64>::zeros((nd, nz));
    for d in 0..nd {
        let mut row_sum = 0.0;
        for z in 0..nz {
            let v: f64 = rng.gen_range(0.5..1.5);
            p_z_d[[d, z]] = v;
            row_sum += v;
        }
        for z in 0..nz {
            p_z_d[[d, z]] /= row_sum;
        }
    }

    let mut p_w_z = Array2::<f64>::zeros((nz, nw));
    for z in 0..nz {
        let mut row_sum = 0.0;
        for w in 0..nw {
            let v: f64 = rng.gen_range(0.5..1.5);
            p_w_z[[z, w]] = v;
            row_sum += v;
        }
        for w in 0..nw {
            p_w_z[[z, w]] /= row_sum;
        }
    }

    // ── Pre-compute nonzero positions to speed up the inner loop ──────────────
    let nonzero: Vec<(usize, usize, f64)> = {
        let mut v = Vec::new();
        for d in 0..nd {
            for w in 0..nw {
                let cnt = x[[d, w]];
                if cnt > 0.0 {
                    v.push((d, w, cnt));
                }
            }
        }
        v
    };

    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_iter = 0usize;
    let mut q_buf = vec![0.0f64; nz]; // reusable buffer for E-step

    for iter in 0..config.max_iter {
        // ── M-step accumulators ───────────────────────────────────────────────
        let mut new_p_z_d = Array2::<f64>::zeros((nd, nz));
        let mut new_p_w_z = Array2::<f64>::zeros((nz, nw));
        let mut ll = 0.0f64;

        // ── Combined E+M step (one pass over nonzeros) ────────────────────────
        for &(d, w, cnt) in &nonzero {
            // E-step: compute Q(z | d, w) ∝ P(z|d) P(w|z)
            let mut denom = 0.0;
            for z in 0..nz {
                let qv = p_z_d[[d, z]] * p_w_z[[z, w]];
                q_buf[z] = qv;
                denom += qv;
            }
            if denom < EPS {
                continue;
            }
            ll += cnt * (denom + EPS).ln();
            // M-step accumulation
            for z in 0..nz {
                let weighted = cnt * q_buf[z] / denom;
                new_p_z_d[[d, z]] += weighted;
                new_p_w_z[[z, w]] += weighted;
            }
        }

        // ── Normalise ─────────────────────────────────────────────────────────
        for d in 0..nd {
            let row_sum: f64 = (0..nz).map(|z| new_p_z_d[[d, z]]).sum();
            for z in 0..nz {
                p_z_d[[d, z]] = new_p_z_d[[d, z]] / (row_sum + EPS);
            }
        }
        for z in 0..nz {
            let row_sum: f64 = (0..nw).map(|w| new_p_w_z[[z, w]]).sum();
            for w in 0..nw {
                p_w_z[[z, w]] = new_p_w_z[[z, w]] / (row_sum + EPS);
            }
        }

        // ── Convergence check ─────────────────────────────────────────────────
        let delta = (ll - prev_ll).abs();
        final_iter = iter + 1;
        if iter > 0 && delta < config.tol {
            prev_ll = ll;
            break;
        }
        prev_ll = ll;
    }

    Ok(PLSAModel {
        p_z_d,
        p_w_z,
        n_iter: final_iter,
        log_likelihood: prev_ll,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_corpus(n_docs: usize, n_words: usize, n_topics: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        // Each document is dominated by one topic
        let mut x = Array2::<f64>::zeros((n_docs, n_words));
        let words_per_topic = n_words / n_topics;
        for d in 0..n_docs {
            let topic = d % n_topics;
            let start = topic * words_per_topic;
            let end = (start + words_per_topic).min(n_words);
            for w in start..end {
                x[[d, w]] = rng.gen_range(1.0..10.0);
            }
        }
        x
    }

    #[test]
    fn test_plsa_basic() {
        let x = make_corpus(20, 12, 3);
        let config = PLSAConfig {
            n_topics: 3,
            max_iter: 50,
            tol: 1e-3,
            seed: 0,
        };
        let model = fit_em(&x, &config).expect("pLSA fit failed");
        assert_eq!(model.p_z_d.shape(), &[20, 3]);
        assert_eq!(model.p_w_z.shape(), &[3, 12]);
        // All probabilities in [0, 1]
        assert!(model.p_z_d.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-9));
        assert!(model.p_w_z.iter().all(|&v| v >= 0.0 && v <= 1.0 + 1e-9));
        // Rows of P(z|d) sum to 1
        for d in 0..20 {
            let s: f64 = (0..3).map(|z| model.p_z_d[[d, z]]).sum();
            assert!((s - 1.0).abs() < 1e-6, "P(z|d) row {d} does not sum to 1: {s}");
        }
    }

    #[test]
    fn test_plsa_topic_words() {
        let x = make_corpus(15, 9, 3);
        let config = PLSAConfig {
            n_topics: 3,
            max_iter: 30,
            tol: 1e-3,
            seed: 1,
        };
        let model = fit_em(&x, &config).expect("fit failed");
        let top_words = model.topic_words(3);
        assert_eq!(top_words.len(), 3);
        for tw in &top_words {
            assert_eq!(tw.len(), 3);
        }
    }

    #[test]
    fn test_plsa_perplexity() {
        let x = make_corpus(20, 10, 2);
        let config = PLSAConfig {
            n_topics: 2,
            max_iter: 50,
            tol: 1e-3,
            seed: 2,
        };
        let model = fit_em(&x, &config).expect("fit failed");
        let ppl = model.perplexity(&x).expect("perplexity failed");
        assert!(ppl > 0.0 && ppl.is_finite(), "Perplexity={ppl}");
    }

    #[test]
    fn test_plsa_infer_topics() {
        let x_train = make_corpus(20, 10, 2);
        let config = PLSAConfig {
            n_topics: 2,
            max_iter: 40,
            tol: 1e-3,
            seed: 3,
        };
        let model = fit_em(&x_train, &config).expect("fit failed");
        let x_new = make_corpus(5, 10, 2);
        let new_topics = model.infer_topics(&x_new, 20).expect("infer failed");
        assert_eq!(new_topics.shape(), &[5, 2]);
        for d in 0..5 {
            let s: f64 = (0..2).map(|z| new_topics[[d, z]]).sum();
            assert!((s - 1.0).abs() < 1e-5, "Inferred P(z|d) row {d} sum={s}");
        }
    }

    #[test]
    fn test_plsa_negative_input_error() {
        let mut x = make_corpus(5, 4, 1);
        x[[0, 0]] = -1.0;
        let config = PLSAConfig::default();
        assert!(fit_em(&x, &config).is_err());
    }

    #[test]
    fn test_plsa_zero_topics_error() {
        let x = make_corpus(5, 4, 1);
        let config = PLSAConfig {
            n_topics: 0,
            ..Default::default()
        };
        assert!(fit_em(&x, &config).is_err());
    }
}
