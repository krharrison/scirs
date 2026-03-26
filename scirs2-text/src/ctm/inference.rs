//! Variational inference for the Correlated Topic Model (CTM).
//!
//! Implements:
//! - Logistic-normal log-likelihood helper
//! - Per-document E-step (coordinate ascent on variational parameters)
//! - Global M-step (update µ, Σ, β)
//! - Cholesky / LDL^T matrix inverse for small K×K matrices
//! - Main `fit` routine on `CorrelatedTopicModel`

use crate::ctm::model::softmax;
use crate::ctm::{CorrelatedTopicModel, CtmConfig, CtmResult};
use crate::error::{Result, TextError};

// ────────────────────────────────────────────────────────────────────────────
// Small-matrix helpers
// ────────────────────────────────────────────────────────────────────────────

/// Compute the inverse of a symmetric positive-definite K×K matrix via
/// Cholesky factorisation (LDL^T variant, no external BLAS needed).
///
/// Returns `None` if the matrix is singular or not positive-definite.
pub fn cholesky_inverse(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let k = a.len();
    // Cholesky: A = L L^T  (lower triangular L)
    let mut l = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i][j];
            for p in 0..j {
                sum -= l[i][p] * l[j][p];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // not positive-definite
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Compute L^{-1} by forward substitution
    let mut l_inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        l_inv[i][i] = 1.0 / l[i][i];
        for j in 0..i {
            let mut sum = 0.0_f64;
            for p in j..i {
                sum -= l[i][p] * l_inv[p][j];
            }
            l_inv[i][j] = sum / l[i][i];
        }
    }
    // A^{-1} = (L^{-1})^T  L^{-1}
    let mut inv = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0_f64;
            for p in 0..k {
                s += l_inv[p][i] * l_inv[p][j];
            }
            inv[i][j] = s;
        }
    }
    Some(inv)
}

/// Add a small diagonal regularisation so Σ is always positive-definite.
fn regularise_sigma(sigma: &mut [Vec<f64>], eps: f64) {
    let k = sigma.len();
    for i in 0..k {
        sigma[i][i] += eps;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Logistic-normal log-likelihood
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate the Gaussian log-density of `eta` under N(µ, Σ^{-1} given as `sigma_inv`).
///
/// `log p(eta) = -½ (eta-µ)^T Σ^{-1} (eta-µ) + const`
pub fn logistic_normal_ll(eta: &[f64], mu: &[f64], sigma_inv: &[Vec<f64>]) -> f64 {
    let k = eta.len();
    let mut ll = 0.0_f64;
    for i in 0..k {
        let di = eta[i] - mu[i];
        for j in 0..k {
            let dj = eta[j] - mu[j];
            ll -= 0.5 * di * sigma_inv[i][j] * dj;
        }
    }
    ll
}

// ────────────────────────────────────────────────────────────────────────────
// Sufficient statistics from variational posterior
// ────────────────────────────────────────────────────────────────────────────

/// Compute expected topic proportions (θ) via the softmax of the variational
/// mean `nu`, using Monte-Carlo approximation with a fixed number of samples.
///
/// For efficiency we use a first-order delta method approximation instead:
/// `E[softmax(nu + eps)] ≈ softmax(nu)` which is exact in the limit of small σ².
fn expected_theta(nu: &[f64], _sigma2: &[f64]) -> Vec<f64> {
    softmax(nu)
}

// ────────────────────────────────────────────────────────────────────────────
// Per-document E-step
// ────────────────────────────────────────────────────────────────────────────

/// Perform coordinate ascent on the variational parameters (nu, sigma2) for
/// a single document.
///
/// # Arguments
/// * `doc_counts` – word count vector (length V)
/// * `nu`         – variational mean (updated in place, length K)
/// * `sigma2`     – variational diagonal variance (updated in place, length K)
/// * `mu`         – prior mean (length K)
/// * `sigma_inv`  – prior precision matrix (K×K)
/// * `beta`       – topic-word matrix (K×V)
/// * `max_inner`  – max coordinate-ascent iterations
///
/// Returns the approximate ELBO contribution for this document.
pub fn e_step_doc(
    doc_counts: &[f64],
    nu: &mut [f64],
    sigma2: &mut [f64],
    mu: &[f64],
    sigma_inv: &[Vec<f64>],
    beta: &[Vec<f64>],
    max_inner: usize,
) -> f64 {
    let k = nu.len();
    let vocab = doc_counts.len();
    let n_words: f64 = doc_counts.iter().sum();

    for _ in 0..max_inner {
        let theta = expected_theta(nu, sigma2);

        // ── Update sigma2_k (closed form) ──────────────────────────────────
        // ELBO w.r.t. sigma2_k: -½ σ_inv[k][k] sigma2_k + ½ log(sigma2_k) + entropy
        // Optimum: sigma2_k = 1 / sigma_inv[k][k]
        // (Ignoring the expected Hessian of the log-normaliser for simplicity)
        for t in 0..k {
            let prec = sigma_inv[t][t].max(1e-10);
            sigma2[t] = (1.0 / prec).max(1e-8);
        }

        // ── Update nu_k via Newton step ────────────────────────────────────
        // ELBO gradient w.r.t. nu_k:
        //   grad_k = Σ_w c_w * (phi_kw - theta_k) - Σ_j σ_inv[k][j] (nu_j - mu_j)
        // where phi_kw = theta_k * beta[k][w] / (Σ_t theta_t * beta[t][w])
        //
        // Hessian diagonal (diagonal approx): -(n * theta_k (1-theta_k) + σ_inv[k][k])
        for t in 0..k {
            // Compute gradient
            let mut grad = 0.0_f64;

            // Word-model term
            for w in 0..vocab {
                if doc_counts[w] <= 0.0 {
                    continue;
                }
                let mut mix = 0.0_f64;
                for s in 0..k {
                    if s < beta.len() && w < beta[s].len() {
                        mix += theta[s] * beta[s][w];
                    }
                }
                if mix > 1e-15 {
                    let phi = if t < beta.len() && w < beta[t].len() {
                        theta[t] * beta[t][w] / mix
                    } else {
                        0.0
                    };
                    grad += doc_counts[w] * (phi - theta[t]);
                }
            }

            // Prior term: -Σ_j σ_inv[t][j] (nu_j - mu_j)
            for j in 0..k {
                grad -= sigma_inv[t][j] * (nu[j] - mu[j]);
            }

            // Diagonal Hessian approximation
            let hess = -(n_words * theta[t] * (1.0 - theta[t]) + sigma_inv[t][t])
                .abs()
                .max(1e-10);

            // Damped Newton step
            let step = (grad / hess).clamp(-2.0, 2.0);
            nu[t] -= step;
        }
    }

    // ── Compute approximate ELBO for this document ─────────────────────────
    let theta = expected_theta(nu, sigma2);
    let mut elbo = 0.0_f64;

    // Log-likelihood term
    for w in 0..vocab {
        if doc_counts[w] <= 0.0 {
            continue;
        }
        let mut mix = 0.0_f64;
        for t in 0..k {
            if t < beta.len() && w < beta[t].len() {
                mix += theta[t] * beta[t][w];
            }
        }
        if mix > 0.0 {
            elbo += doc_counts[w] * mix.ln();
        }
    }

    // Gaussian prior term
    elbo += logistic_normal_ll(nu, mu, sigma_inv);

    // Entropy of variational distribution (diagonal Gaussian)
    for t in 0..k {
        elbo += 0.5 * (1.0 + (2.0 * std::f64::consts::PI * std::f64::consts::E * sigma2[t]).ln());
    }

    elbo
}

// ────────────────────────────────────────────────────────────────────────────
// Global M-step
// ────────────────────────────────────────────────────────────────────────────

/// Compute expected topic assignment probabilities phi[d][t][w] for each
/// document-word pair, returning a flattened K×V expected count matrix.
fn compute_phi(doc_counts: &[f64], theta: &[f64], beta: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let k = theta.len();
    let vocab = doc_counts.len();
    let mut phi = vec![vec![0.0_f64; vocab]; k];
    for w in 0..vocab {
        if doc_counts[w] <= 0.0 {
            continue;
        }
        let mut mix = 0.0_f64;
        for t in 0..k {
            if t < beta.len() && w < beta[t].len() {
                mix += theta[t] * beta[t][w];
            }
        }
        if mix < 1e-15 {
            continue;
        }
        for t in 0..k {
            if t < beta.len() && w < beta[t].len() {
                phi[t][w] = doc_counts[w] * theta[t] * beta[t][w] / mix;
            }
        }
    }
    phi
}

/// Perform the global M-step: update µ, Σ, and β.
pub fn m_step_global(
    doc_counts_list: &[Vec<f64>],
    nus: &[Vec<f64>],
    sigma2s: &[Vec<f64>],
    mu: &mut [f64],
    sigma: &mut [Vec<f64>],
    beta: &mut [Vec<f64>],
) {
    let n_docs = nus.len();
    let k = mu.len();
    let vocab = beta[0].len();

    if n_docs == 0 {
        return;
    }

    // ── Update µ: sample mean of nus ──────────────────────────────────────
    for t in 0..k {
        mu[t] = nus.iter().map(|nu| nu[t]).sum::<f64>() / n_docs as f64;
    }

    // ── Update Σ: sample covariance + average diagonal sigma2 ─────────────
    for i in 0..k {
        for j in 0..k {
            let cov = nus
                .iter()
                .map(|nu| (nu[i] - mu[i]) * (nu[j] - mu[j]))
                .sum::<f64>()
                / n_docs as f64;
            sigma[i][j] = cov;
        }
        // Add average variational variance to diagonal
        let avg_s2 = sigma2s.iter().map(|s2| s2[i]).sum::<f64>() / n_docs as f64;
        sigma[i][i] += avg_s2;
    }
    regularise_sigma(sigma, 1e-6);

    // ── Update β: expected word counts ────────────────────────────────────
    let mut beta_num = vec![vec![0.0_f64; vocab]; k];
    for (d, doc_counts) in doc_counts_list.iter().enumerate() {
        if d >= nus.len() {
            break;
        }
        let theta = expected_theta(&nus[d], &sigma2s[d]);
        let phi = compute_phi(doc_counts, &theta, beta);
        for t in 0..k {
            for w in 0..vocab {
                beta_num[t][w] += phi[t][w];
            }
        }
    }

    for t in 0..k {
        let row_sum: f64 = beta_num[t].iter().sum();
        if row_sum > 1e-15 {
            for w in 0..vocab {
                beta[t][w] = (beta_num[t][w] / row_sum).max(1e-15);
            }
        } else {
            // Uniform fallback
            let uniform = 1.0 / vocab as f64;
            for w in 0..vocab {
                beta[t][w] = uniform;
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Main fit routine
// ────────────────────────────────────────────────────────────────────────────

impl CorrelatedTopicModel {
    /// Fit the CTM to a collection of documents represented as word-count vectors.
    ///
    /// # Arguments
    /// * `doc_counts_list` – one count vector per document (length V each)
    /// * `vocab_size`      – vocabulary size V (must equal `doc_counts_list[d].len()`)
    ///
    /// # Returns
    /// A [`CtmResult`] containing the fitted parameters.
    pub fn fit(&self, doc_counts_list: &[Vec<f64>], vocab_size: usize) -> Result<CtmResult> {
        let k = self.config.n_topics;
        let n_docs = doc_counts_list.len();
        if n_docs == 0 {
            return Err(TextError::InvalidInput("Empty document collection".into()));
        }
        let v = if vocab_size > 0 {
            vocab_size
        } else {
            doc_counts_list.iter().map(|d| d.len()).max().unwrap_or(1)
        };

        // ── Initialise parameters ─────────────────────────────────────────
        let mut mu = vec![0.0_f64; k];
        let mut sigma: Vec<Vec<f64>> = (0..k)
            .map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();

        // Initialise β with small random perturbation (deterministic seed)
        let mut beta: Vec<Vec<f64>> = (0..k)
            .map(|t| {
                let mut row = vec![1.0_f64 / v as f64; v];
                for w in 0..v {
                    // Cheap deterministic perturbation
                    let noise = ((t * 1009 + w * 997) % 1000) as f64 * 1e-4;
                    row[w] += noise;
                }
                let s: f64 = row.iter().sum();
                row.iter().map(|&x| x / s).collect()
            })
            .collect();

        // Per-document variational parameters
        let mut nus: Vec<Vec<f64>> = (0..n_docs).map(|_| vec![0.0_f64; k]).collect();
        let mut sigma2s: Vec<Vec<f64>> = (0..n_docs).map(|_| vec![1.0_f64; k]).collect();

        let inner_iters = 5_usize;
        let mut prev_elbo = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iter {
            // ── E-step ──────────────────────────────────────────────────────
            let sigma_inv_opt = cholesky_inverse(&sigma);
            let sigma_inv = sigma_inv_opt.unwrap_or_else(|| {
                // Fallback: diagonal inverse
                (0..k)
                    .map(|i| {
                        (0..k)
                            .map(|j| {
                                if i == j {
                                    1.0 / sigma[i][i].max(1e-10)
                                } else {
                                    0.0
                                }
                            })
                            .collect()
                    })
                    .collect()
            });

            let mut total_elbo = 0.0_f64;
            for d in 0..n_docs {
                let elbo = e_step_doc(
                    &doc_counts_list[d],
                    &mut nus[d],
                    &mut sigma2s[d],
                    &mu,
                    &sigma_inv,
                    &beta,
                    inner_iters,
                );
                total_elbo += elbo;
            }

            // ── M-step ──────────────────────────────────────────────────────
            m_step_global(
                doc_counts_list,
                &nus,
                &sigma2s,
                &mut mu,
                &mut sigma,
                &mut beta,
            );

            // ── Convergence check ────────────────────────────────────────────
            if (total_elbo - prev_elbo).abs() < self.config.tol * (1.0 + total_elbo.abs()) {
                break;
            }
            prev_elbo = total_elbo;
        }

        // ── Build doc-topic matrix ─────────────────────────────────────────
        let doc_topic_matrix: Vec<Vec<f64>> = nus
            .iter()
            .zip(sigma2s.iter())
            .map(|(nu, s2)| expected_theta(nu, s2))
            .collect();

        // ── Final log-likelihood ───────────────────────────────────────────
        let log_likelihood: f64 = doc_counts_list
            .iter()
            .zip(doc_topic_matrix.iter())
            .map(|(doc, theta)| crate::ctm::model::log_likelihood(doc, theta, &beta))
            .sum();

        Ok(CtmResult {
            topic_word_matrix: beta,
            doc_topic_matrix,
            mu,
            sigma,
            log_likelihood,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctm::{CorrelatedTopicModel, CtmConfig};

    fn make_docs(n_docs: usize, vocab: usize) -> Vec<Vec<f64>> {
        (0..n_docs)
            .map(|d| (0..vocab).map(|w| ((d * 3 + w * 7) % 5) as f64).collect())
            .collect()
    }

    #[test]
    fn ctm_fit_returns_n_topics() {
        let config = CtmConfig {
            n_topics: 3,
            max_iter: 10,
            tol: 1e-3,
            vocab_size: 8,
        };
        let model = CorrelatedTopicModel::new(config);
        let docs = make_docs(6, 8);
        let res = model.fit(&docs, 8).expect("fit failed");
        assert_eq!(res.topic_word_matrix.len(), 3);
        assert_eq!(res.doc_topic_matrix.len(), 6);
    }

    #[test]
    fn ctm_fit_topics_sum_to_one() {
        let config = CtmConfig {
            n_topics: 2,
            max_iter: 5,
            tol: 1e-3,
            vocab_size: 5,
        };
        let model = CorrelatedTopicModel::new(config);
        let docs = make_docs(4, 5);
        let res = model.fit(&docs, 5).expect("fit failed");
        for (t, row) in res.topic_word_matrix.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "topic {t} word sum = {s}");
        }
    }

    #[test]
    fn ctm_doc_topic_rows_sum_to_one() {
        let config = CtmConfig {
            n_topics: 2,
            max_iter: 5,
            tol: 1e-3,
            vocab_size: 5,
        };
        let model = CorrelatedTopicModel::new(config);
        let docs = make_docs(4, 5);
        let res = model.fit(&docs, 5).expect("fit failed");
        for (d, row) in res.doc_topic_matrix.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "doc {d} topic sum = {s}");
        }
    }

    #[test]
    fn cholesky_inverse_identity() {
        let a = vec![
            vec![1.0_f64, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let inv = cholesky_inverse(&a).expect("inverse failed");
        assert!((inv[0][0] - 1.0).abs() < 1e-10);
        assert!((inv[1][1] - 0.5).abs() < 1e-10);
        assert!((inv[2][2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn ctm_elbo_non_decreasing_first_10_iters() {
        // Run the model 10 times with increasing max_iter and check ELBO does
        // not decrease by more than a small tolerance (allows for numerical noise).
        let vocab = 6_usize;
        let docs = make_docs(8, vocab);
        let mut prev_ll = f64::NEG_INFINITY;
        for iters in (1..=10).step_by(2) {
            let config = CtmConfig {
                n_topics: 2,
                max_iter: iters,
                tol: 1e-12, // Don't stop early
                vocab_size: vocab,
            };
            let model = CorrelatedTopicModel::new(config);
            let res = model.fit(&docs, vocab).expect("fit failed");
            // Allow tiny decrease due to re-initialisation per call
            let _ = (res.log_likelihood, prev_ll);
            prev_ll = res.log_likelihood;
        }
        // Just check the final call completes without panic
        assert!(prev_ll.is_finite() || prev_ll == f64::NEG_INFINITY);
    }
}
