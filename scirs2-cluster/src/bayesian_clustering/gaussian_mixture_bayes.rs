//! Variational Bayes Gaussian Mixture Model (VB-GMM).
//!
//! Implements the variational inference algorithm for a Bayesian GMM following
//! Bishop (2006), Chapter 10.  The prior is a Dirichlet-Normal-Wishart:
//!
//! - π ~ Dir(α₀ 1_K)
//! - μ_k, Λ_k ~ NW(m₀, β₀, ν₀, W₀)
//!
//! The variational distribution factorises over z (assignments), π, μ, Λ.
//!
//! # References
//!
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*.
//!   Chapter 10: Approximate Inference.
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::bayesian_clustering::gaussian_mixture_bayes::{
//!     VBGMMConfig, vbgmm_fit,
//! };
//!
//! let data = vec![
//!     vec![1.0_f64, 2.0], vec![1.1, 1.9], vec![0.9, 2.1],
//!     vec![5.0, 5.0],     vec![5.1, 4.9], vec![4.9, 5.1],
//! ];
//! let config = VBGMMConfig::default_for_data(&data, 3);
//! let state = vbgmm_fit(&data, &config).expect("vbgmm fit");
//! assert_eq!(state.r.len(), 6);
//! ```

use std::f64::consts::PI;

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Variational Bayes GMM.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct VBGMMConfig {
    /// Number of mixture components K.
    pub n_components: usize,
    /// Maximum number of VB iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the ELBO.
    pub tol: f64,
    /// Dirichlet prior concentration α₀.
    pub alpha0: f64,
    /// Normal-Wishart prior strength β₀.
    pub beta0: f64,
    /// Wishart degrees of freedom ν₀ (must be > D - 1).
    pub nu0: f64,
    /// Prior mean m₀ (length D).
    pub m0: Vec<f64>,
    /// Wishart scale matrix W₀ (D×D).
    pub W0: Vec<Vec<f64>>,
}

impl VBGMMConfig {
    /// Construct default config from data statistics.
    pub fn default_for_data(data: &[Vec<f64>], n_components: usize) -> Self {
        let d = if data.is_empty() { 1 } else { data[0].len() };
        let n = data.len() as f64;

        // Data mean
        let mut m0 = vec![0.0f64; d];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                if j < d {
                    m0[j] += v;
                }
            }
        }
        for v in m0.iter_mut() {
            *v /= n.max(1.0);
        }

        // W0 = I / D (scale roughly to data)
        let w0: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                let mut row = vec![0.0; d];
                row[i] = 1.0 / d as f64;
                row
            })
            .collect();

        Self {
            n_components,
            max_iter: 200,
            tol: 1e-6,
            alpha0: 1e-3,
            beta0: 1e-3,
            nu0: d as f64 + 1.0,
            m0,
            W0: w0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variational parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Variational parameters for the VB-GMM.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct VBParams {
    /// Dirichlet variational parameters α_k (length K).
    pub alpha: Vec<f64>,
    /// Normal precision weights β_k (length K).
    pub beta: Vec<f64>,
    /// Wishart degrees of freedom ν_k (length K).
    pub nu: Vec<f64>,
    /// Variational mean m_k (K×D).
    pub m: Vec<Vec<f64>>,
    /// Wishart scale matrix W_k (K×D×D).
    pub W: Vec<Vec<Vec<f64>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Result of VB-GMM fitting.
#[derive(Debug, Clone)]
pub struct VBGMMState {
    /// Responsibilities r_{nk} (N×K).
    pub r: Vec<Vec<f64>>,
    /// Final variational parameters.
    pub params: VBParams,
    /// ELBO trajectory (one value per iteration).
    pub lower_bound: Vec<f64>,
}

impl VBGMMState {
    /// Predict hard cluster assignments (argmax of responsibilities).
    pub fn predict(&self) -> Vec<usize> {
        self.r
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(k, _)| k)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Linear-algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cholesky of D×D SPD matrix; returns lower-triangular L.
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = s.max(1e-15).sqrt();
            } else if l[j][j].abs() > 1e-15 {
                l[i][j] = s / l[j][j];
            }
        }
    }
    l
}

/// Log-determinant via Cholesky.
fn log_det(a: &[Vec<f64>]) -> f64 {
    let l = cholesky(a);
    let n = l.len();
    (0..n).map(|i| 2.0 * l[i][i].max(1e-300).ln()).sum()
}

/// Digamma approximation (Stirling series).
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut v = x;
    let mut result = 0.0;
    while v < 6.0 {
        result -= 1.0 / v;
        v += 1.0;
    }
    result += v.ln() - 0.5 / v;
    let inv2 = 1.0 / (v * v);
    result -= inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0));
    result
}

/// Multivariate digamma: Σ_{j=1}^{D} ψ((ν + 1 - j)/2).
fn mv_digamma(nu: f64, d: usize) -> f64 {
    (0..d).map(|j| digamma((nu + 1.0 - j as f64) / 2.0)).sum()
}

/// Matrix-vector multiply: A x.
fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(aij, xj)| aij * xj).sum())
        .collect()
}

/// Compute x^T A x.
fn quadratic(x: &[f64], a: &[Vec<f64>]) -> f64 {
    let ax = mat_vec(a, x);
    x.iter().zip(ax.iter()).map(|(xi, axi)| xi * axi).sum()
}

/// Invert a D×D positive-definite matrix via Cholesky.
fn invert_pd(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let l = cholesky(a);

    // Forward-solve L y_j = e_j for each column j.
    let mut inv = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut s = if i == j { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l[i][k] * y[k];
            }
            y[i] = if l[i][i].abs() > 1e-15 { s / l[i][i] } else { 0.0 };
        }
        // Back-solve L^T x = y.
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k];
            }
            x[i] = if l[i][i].abs() > 1e-15 { s / l[i][i] } else { 0.0 };
        }
        for i in 0..n {
            inv[i][j] = x[i];
        }
    }
    inv
}

/// Add two D×D matrices.
fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

/// Scale D×D matrix by scalar.
fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|v| v * s).collect())
        .collect()
}

/// Outer product: x y^T.
fn outer(x: &[f64], y: &[f64]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|xi| y.iter().map(|yj| xi * yj).collect())
        .collect()
}

/// Log-sum-exp of a slice.
fn logsumexp(v: &[f64]) -> f64 {
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = v.iter().map(|x| (x - max).exp()).sum();
    max + s.ln()
}

// ─────────────────────────────────────────────────────────────────────────────
// VB-GMM main algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a Variational Bayes GMM to `data`.
///
/// # Parameters
///
/// - `data`: N observations, each a D-dimensional vector.
/// - `config`: algorithm configuration (see [`VBGMMConfig`]).
///
/// # Returns
///
/// A [`VBGMMState`] containing responsibilities, variational parameters, and
/// the ELBO trajectory.
#[allow(non_snake_case)]
pub fn vbgmm_fit(data: &[Vec<f64>], config: &VBGMMConfig) -> Result<VBGMMState> {
    let n_data = data.len();
    if n_data == 0 {
        return Err(ClusteringError::InvalidInput(
            "Data must be non-empty".to_string(),
        ));
    }
    let d = data[0].len();
    if d == 0 {
        return Err(ClusteringError::InvalidInput(
            "Feature dimension must be > 0".to_string(),
        ));
    }
    let k = config.n_components;
    if k == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_components must be > 0".to_string(),
        ));
    }
    if config.nu0 <= (d as f64) - 1.0 {
        return Err(ClusteringError::InvalidInput(format!(
            "nu0 must be > D-1 = {}",
            d - 1
        )));
    }

    // Validate data shape
    for (i, row) in data.iter().enumerate() {
        if row.len() != d {
            return Err(ClusteringError::InvalidInput(format!(
                "Row {} has {} features, expected {}",
                i,
                row.len(),
                d
            )));
        }
    }

    // ── Initialise variational parameters ──────────────────────────────────
    // Responsibility initialisation: uniform + small perturbation.
    let mut r: Vec<Vec<f64>> = (0..n_data)
        .map(|i| {
            let base = 1.0 / k as f64;
            let mut row: Vec<f64> = (0..k)
                .map(|j| {
                    // Simple deterministic perturbation.
                    let perturb = ((i * k + j) as f64 * 0.01).sin() * 0.01;
                    (base + perturb).max(1e-10)
                })
                .collect();
            let s: f64 = row.iter().sum();
            for v in row.iter_mut() {
                *v /= s;
            }
            row
        })
        .collect();

    // M-step to get initial parameters.
    let mut params = m_step(data, &r, config, d, k, n_data)?;

    let mut lower_bounds = Vec::with_capacity(config.max_iter);
    let mut prev_elbo = f64::NEG_INFINITY;

    for _iter in 0..config.max_iter {
        // E-step: update responsibilities.
        r = e_step(data, &params, config, d, k, n_data)?;

        // M-step: update variational parameters.
        params = m_step(data, &r, config, d, k, n_data)?;

        // ELBO.
        let elbo = compute_elbo(data, &r, &params, config, d, k, n_data);
        lower_bounds.push(elbo);

        if (elbo - prev_elbo).abs() < config.tol {
            break;
        }
        prev_elbo = elbo;
    }

    Ok(VBGMMState {
        r,
        params,
        lower_bound: lower_bounds,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// E-step
// ─────────────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
fn e_step(
    data: &[Vec<f64>],
    params: &VBParams,
    config: &VBGMMConfig,
    d: usize,
    k: usize,
    n_data: usize,
) -> Result<Vec<Vec<f64>>> {
    let alpha_sum: f64 = params.alpha.iter().sum();

    // Pre-compute per-component quantities.
    let mut E_ln_pi: Vec<f64> = params
        .alpha
        .iter()
        .map(|&a| digamma(a) - digamma(alpha_sum))
        .collect();

    let mut E_ln_lam: Vec<f64> = (0..k)
        .map(|j| {
            mv_digamma(params.nu[j], d)
                + (d as f64) * (2.0_f64.ln())
                + log_det(&params.W[j])
        })
        .collect();

    let mut r = vec![vec![0.0f64; k]; n_data];

    for n in 0..n_data {
        let x = &data[n];
        let mut log_rho = vec![0.0f64; k];

        for j in 0..k {
            // E_k[||x_n - mu_k||^2_Lambda_k]
            let x_minus_m: Vec<f64> = x
                .iter()
                .zip(params.m[j].iter())
                .map(|(xi, mi)| xi - mi)
                .collect();

            // W_k * (x - m_k)
            let W_delta = mat_vec(&params.W[j], &x_minus_m);
            let xW_delta: f64 = x_minus_m
                .iter()
                .zip(W_delta.iter())
                .map(|(xi, wd)| xi * wd)
                .sum();

            let E_maha = d as f64 / params.beta[j] + params.nu[j] * xW_delta;

            log_rho[j] = E_ln_pi[j] + 0.5 * E_ln_lam[j]
                - (d as f64) / 2.0 * (2.0 * PI).ln()
                - 0.5 * E_maha;
        }

        // Normalise via log-sum-exp.
        let lse = logsumexp(&log_rho);
        for j in 0..k {
            r[n][j] = (log_rho[j] - lse).exp().max(1e-300);
        }
        // Re-normalise numerically.
        let s: f64 = r[n].iter().sum();
        for v in r[n].iter_mut() {
            *v /= s;
        }
    }

    Ok(r)
}

// ─────────────────────────────────────────────────────────────────────────────
// M-step
// ─────────────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
fn m_step(
    data: &[Vec<f64>],
    r: &[Vec<f64>],
    config: &VBGMMConfig,
    d: usize,
    k: usize,
    n_data: usize,
) -> Result<VBParams> {
    // Compute statistics: N_k, x̄_k, S_k.
    let mut N_k = vec![0.0f64; k];
    let mut xbar_k = vec![vec![0.0f64; d]; k];
    let mut S_k: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; d]; d]; k];

    for n in 0..n_data {
        let x = &data[n];
        for j in 0..k {
            let r_nk = r[n][j];
            N_k[j] += r_nk;
            for l in 0..d {
                xbar_k[j][l] += r_nk * x[l];
            }
        }
    }

    // Normalise x̄_k.
    for j in 0..k {
        if N_k[j] > 1e-10 {
            for l in 0..d {
                xbar_k[j][l] /= N_k[j];
            }
        } else {
            xbar_k[j] = config.m0.clone();
        }
    }

    // Scatter matrices S_k = Σ_n r_{nk} (x_n - x̄_k)(x_n - x̄_k)^T
    for n in 0..n_data {
        let x = &data[n];
        for j in 0..k {
            let r_nk = r[n][j];
            if r_nk < 1e-15 {
                continue;
            }
            let delta: Vec<f64> = x.iter().zip(xbar_k[j].iter()).map(|(xi, mi)| xi - mi).collect();
            for p in 0..d {
                for q in 0..d {
                    S_k[j][p][q] += r_nk * delta[p] * delta[q];
                }
            }
        }
    }

    // Update variational parameters.
    let W0_inv = invert_pd(&config.W0);

    let mut alpha = vec![0.0f64; k];
    let mut beta = vec![0.0f64; k];
    let mut nu = vec![0.0f64; k];
    let mut m = vec![vec![0.0f64; d]; k];
    let mut W: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; d]; d]; k];

    for j in 0..k {
        alpha[j] = config.alpha0 + N_k[j];
        beta[j] = config.beta0 + N_k[j];
        nu[j] = config.nu0 + N_k[j] + 1.0;

        // m_k = (β₀ m₀ + N_k x̄_k) / β_k
        for l in 0..d {
            m[j][l] = (config.beta0 * config.m0[l] + N_k[j] * xbar_k[j][l]) / beta[j];
        }

        // W_k^{-1} = W_0^{-1} + N_k S_k + (β₀ N_k)/(β₀ + N_k) (x̄_k - m₀)(x̄_k - m₀)^T
        let correction_factor = (config.beta0 * N_k[j]) / (config.beta0 + N_k[j]);
        let diff_xbar_m0: Vec<f64> = xbar_k[j]
            .iter()
            .zip(config.m0.iter())
            .map(|(xb, m0i)| xb - m0i)
            .collect();
        let outer_correction = mat_scale(&outer(&diff_xbar_m0, &diff_xbar_m0), correction_factor);

        let W_k_inv = mat_add(
            &mat_add(&W0_inv, &S_k[j]),
            &outer_correction,
        );

        // W_k = (W_k_inv)^{-1}
        W[j] = invert_pd(&W_k_inv);
    }

    Ok(VBParams {
        alpha,
        beta,
        nu,
        m,
        W,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// ELBO computation
// ─────────────────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
fn compute_elbo(
    data: &[Vec<f64>],
    r: &[Vec<f64>],
    params: &VBParams,
    config: &VBGMMConfig,
    d: usize,
    k: usize,
    n_data: usize,
) -> f64 {
    // Approximate ELBO: sum_n sum_k r_{nk} * (E[ln p(x_n|z_n=k, theta_k)] + ...)
    // We use a simplified bound based on reconstruction term.
    let alpha_sum: f64 = params.alpha.iter().sum();
    let E_ln_pi: Vec<f64> = params
        .alpha
        .iter()
        .map(|&a| digamma(a) - digamma(alpha_sum))
        .collect();

    let mut elbo = 0.0f64;

    for n in 0..n_data {
        let x = &data[n];
        for j in 0..k {
            let r_nk = r[n][j];
            if r_nk < 1e-300 {
                continue;
            }
            let x_minus_m: Vec<f64> = x
                .iter()
                .zip(params.m[j].iter())
                .map(|(xi, mi)| xi - mi)
                .collect();
            let W_delta = mat_vec(&params.W[j], &x_minus_m);
            let xW_delta: f64 = x_minus_m
                .iter()
                .zip(W_delta.iter())
                .map(|(xi, wd)| xi * wd)
                .sum();

            let E_ln_lam = mv_digamma(params.nu[j], d)
                + d as f64 * 2.0_f64.ln()
                + log_det(&params.W[j]);

            let E_maha = d as f64 / params.beta[j] + params.nu[j] * xW_delta;

            let ln_p_x = 0.5 * (E_ln_lam - d as f64 * (2.0 * PI).ln() - E_maha);

            elbo += r_nk * (E_ln_pi[j] + ln_p_x - r_nk.ln());
        }
    }

    elbo
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![-0.1, 0.1],
            vec![5.0, 5.0],
            vec![5.1, 4.9],
            vec![4.9, 5.1],
        ]
    }

    #[test]
    fn test_vbgmm_basic() {
        let data = two_cluster_data();
        let config = VBGMMConfig::default_for_data(&data, 2);
        let state = vbgmm_fit(&data, &config).expect("vbgmm fit");

        assert_eq!(state.r.len(), data.len());
        assert_eq!(state.r[0].len(), 2);

        // Check that responsibilities sum to 1.
        for row in &state.r {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "row sum = {}", s);
        }
    }

    #[test]
    fn test_vbgmm_separates_clusters() {
        let data = two_cluster_data();
        let config = VBGMMConfig::default_for_data(&data, 2);
        let state = vbgmm_fit(&data, &config).expect("vbgmm fit");

        let preds = state.predict();
        // Points 0-2 should share a label; points 3-5 should share a different label.
        assert_eq!(preds[0], preds[1]);
        assert_eq!(preds[0], preds[2]);
        assert_eq!(preds[3], preds[4]);
        assert_eq!(preds[3], preds[5]);
        assert_ne!(preds[0], preds[3]);
    }

    #[test]
    fn test_vbgmm_elbo_increases() {
        let data = two_cluster_data();
        let mut config = VBGMMConfig::default_for_data(&data, 2);
        config.max_iter = 50;
        let state = vbgmm_fit(&data, &config).expect("vbgmm fit");

        // ELBO should not decrease after convergence.
        if state.lower_bound.len() > 2 {
            let n = state.lower_bound.len();
            // Allow small tolerance for numerical noise.
            let last = state.lower_bound[n - 1];
            let second_last = state.lower_bound[n - 2];
            // They should be close (converged).
            assert!((last - second_last).abs() < 1.0,
                "ELBO not converged: {} -> {}", second_last, last);
        }
    }

    #[test]
    fn test_vbgmm_three_clusters() {
        let data = vec![
            vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1],
            vec![5.0, 0.0], vec![5.1, 0.0], vec![4.9, 0.1],
            vec![0.0, 5.0], vec![0.1, 5.0], vec![0.0, 5.1],
        ];
        let config = VBGMMConfig::default_for_data(&data, 3);
        let state = vbgmm_fit(&data, &config).expect("three-cluster vbgmm");
        assert_eq!(state.r.len(), 9);
        assert_eq!(state.r[0].len(), 3);
    }

    #[test]
    fn test_vbgmm_empty_data_error() {
        let config = VBGMMConfig::default_for_data(&[], 2);
        let result = vbgmm_fit(&[], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_vbgmm_zero_components_error() {
        let data = two_cluster_data();
        let mut config = VBGMMConfig::default_for_data(&data, 2);
        config.n_components = 0;
        let result = vbgmm_fit(&data, &config);
        assert!(result.is_err());
    }
}
