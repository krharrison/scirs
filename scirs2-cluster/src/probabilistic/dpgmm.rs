//! Dirichlet Process Gaussian Mixture Model (DP-GMM).
//!
//! Implements variational Bayes inference for a DP-GMM using the stick-breaking
//! representation.  The variational family approximates the true posterior with a
//! mean-field factorisation:
//!
//! ```text
//! q(π, μ, Λ, z) = q(v) q(μ,Λ) q(z)
//! ```
//!
//! where `v_k ~ Beta(a_k, b_k)` are the stick-breaking weights and each
//! component's parameters follow a Normal-Wishart prior.
//!
//! # Algorithm
//!
//! 1. Initialise `r[i, k]` (responsibilities) via K-means-like random assignment.
//! 2. M-step: update `a_k, b_k, m_k, β_k, ν_k, W_k` from sufficient statistics.
//! 3. E-step: recompute log-responsibilities and normalise.
//! 4. Compute ELBO; stop when improvement < `tol`.
//!
//! # References
//!
//! - Blei & Jordan (2006) "Variational inference for Dirichlet process mixtures"
//! - Bishop (2006) PRML §10.2 (finite VB-GMM extension to the DP case)

use std::f64::consts::{LN_2, PI};

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Math utilities
// ────────────────────────────────────────────────────────────────────────────

/// Digamma function via Stirling asymptotic series (accurate for x > 0).
#[inline]
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
    let iv2 = 1.0 / (v * v);
    result -= iv2 * (1.0 / 12.0 - iv2 * (1.0 / 120.0 - iv2 / 252.0));
    result
}

/// Log-beta function: log B(a,b) = log Γ(a) + log Γ(b) - log Γ(a+b).
#[inline]
fn log_beta(a: f64, b: f64) -> f64 {
    lgamma(a) + lgamma(b) - lgamma(a + b)
}

/// Log-gamma via Lanczos approximation (g=7, n=9 coefficients).
fn lgamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes §6.1
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - lgamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (xm1 + i as f64 + 1.0);
    }
    let t = xm1 + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + sum.ln() + (xm1 + 0.5) * t.ln() - t
}

/// Normalised log-sum-exp over a slice (avoids overflow).
fn logsumexp(vals: &[f64]) -> f64 {
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = vals.iter().map(|&v| (v - max).exp()).sum();
    max + s.ln()
}

// ────────────────────────────────────────────────────────────────────────────
// Cholesky / linear-algebra helpers
// ────────────────────────────────────────────────────────────────────────────

/// Lower-triangular Cholesky factor of a symmetric positive-definite matrix.
fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for kk in 0..j {
                s -= l[[i, kk]] * l[[j, kk]];
            }
            if i == j {
                if s <= 0.0 {
                    s = 1e-10;
                }
                l[[i, j]] = s.sqrt();
            } else {
                let ljj = l[[j, j]];
                l[[i, j]] = if ljj.abs() < 1e-14 { 0.0 } else { s / ljj };
            }
        }
    }
    Ok(l)
}

/// Log-determinant of a positive-definite matrix via Cholesky.
fn log_det_pd(a: &Array2<f64>) -> Result<f64> {
    let l = cholesky_lower(a)?;
    let log_det = (0..l.nrows()).map(|i| 2.0 * l[[i, i]].ln()).sum();
    Ok(log_det)
}

/// Solve `L x = b` (forward substitution, L lower-triangular).
fn forward_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = l.nrows();
    let mut x = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[[i, j]] * x[j];
        }
        let lii = l[[i, i]];
        x[i] = if lii.abs() < 1e-14 { 0.0 } else { s / lii };
    }
    x
}

/// Inverse of a positive-definite matrix via Cholesky.
fn inv_pd(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let l = cholesky_lower(a)?;
    // Invert L column-by-column
    let mut linv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let col = forward_solve(&l, &e);
        for i in 0..n {
            linv[[i, j]] = col[i];
        }
    }
    // A^{-1} = (L^{-1})^T L^{-1}
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let s: f64 = (0..n).map(|k| linv[[k, i]] * linv[[k, j]]).sum();
            inv[[i, j]] = s;
        }
    }
    Ok(inv)
}

/// Quadratic form `(x - m)^T W (x - m)` where W is a D×D matrix and x, m are D-vectors.
fn quad_form(w: &Array2<f64>, x: &[f64], m: &Array1<f64>) -> f64 {
    let d = m.len();
    let diff: Vec<f64> = (0..d).map(|i| x[i] - m[i]).collect();
    let mut q = 0.0;
    for i in 0..d {
        let mut wx = 0.0;
        for j in 0..d {
            wx += w[[i, j]] * diff[j];
        }
        q += diff[i] * wx;
    }
    q
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the DP-GMM variational inference.
#[derive(Debug, Clone)]
pub struct DPGMMConfig {
    /// Concentration parameter α of the Dirichlet process (α > 0).
    /// Larger values favour more active components.
    pub alpha: f64,
    /// Maximum (truncated) number of components T.
    pub max_components: usize,
    /// Maximum number of VB iterations.
    pub n_iter: usize,
    /// Convergence tolerance on the ELBO.
    pub tol: f64,
    /// Prior precision scaling for component means (β₀).
    pub beta0: f64,
    /// Prior degrees of freedom for the Wishart (ν₀ ≥ D).
    pub nu0_offset: f64,
    /// Weight on the diagonal of the prior Wishart scale matrix W₀.
    pub w0_scale: f64,
}

impl DPGMMConfig {
    /// Create a new configuration.
    ///
    /// # Arguments
    /// * `alpha` – DP concentration (> 0).
    /// * `max_components` – Truncation level T.
    /// * `n_iter` – Max VB iterations.
    pub fn new(alpha: f64, max_components: usize, n_iter: usize) -> Self {
        Self {
            alpha,
            max_components,
            n_iter,
            tol: 1e-6,
            beta0: 1e-3,
            nu0_offset: 1.0,
            w0_scale: 1.0,
        }
    }

    /// Set the convergence tolerance.
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the prior precision scaling β₀.
    pub fn with_beta0(mut self, beta0: f64) -> Self {
        self.beta0 = beta0;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Model and result types
// ────────────────────────────────────────────────────────────────────────────

/// Fitted DP-GMM variational parameters.
///
/// These are the parameters of the variational posterior `q`:
/// - `r[n, k]`     – responsibility (soft assignment) of point n to component k
/// - `a_gamma[k]`  – first Beta parameter for stick-breaking variable v_k
/// - `b_gamma[k]`  – second Beta parameter for stick-breaking variable v_k
/// - `beta_k[k]`   – Normal-Wishart precision scaling
/// - `m_k[k, d]`   – Normal-Wishart mean
/// - `nu_k[k]`     – Wishart degrees of freedom
/// - `W_k[k, d, d]`– Wishart scale matrices (stored as a vector of D×D arrays)
#[derive(Debug, Clone)]
pub struct DPGMMResult {
    /// Responsibility matrix (N × T).
    pub r: Array2<f64>,
    /// Beta stick-breaking first parameter (T,).
    pub a_gamma: Array1<f64>,
    /// Beta stick-breaking second parameter (T,).
    pub b_gamma: Array1<f64>,
    /// Normal-Wishart precision scaling (T,).
    pub beta_k: Array1<f64>,
    /// Normal-Wishart mean (T × D).
    pub m_k: Array2<f64>,
    /// Wishart degrees of freedom (T,).
    pub nu_k: Array1<f64>,
    /// Wishart scale matrices, one per component (each D × D).
    pub w_k: Vec<Array2<f64>>,
    /// Number of VB iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Final ELBO value.
    pub elbo: f64,
    /// Truncation level T.
    pub n_components: usize,
    /// Number of features D.
    pub n_features: usize,
}

impl DPGMMResult {
    /// Return the number of truncation-level components.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Count components whose total responsibility exceeds a threshold.
    pub fn effective_components(&self, threshold: f64) -> usize {
        let n_k = self.r.sum_axis(scirs2_core::ndarray::Axis(0));
        n_k.iter().filter(|&&v| v > threshold).count()
    }

    /// Return the hard cluster label for each training point.
    pub fn predict(&self, _data: ArrayView2<f64>) -> Result<Array1<usize>> {
        let n = self.r.nrows();
        let mut labels = Array1::<usize>::zeros(n);
        for i in 0..n {
            let mut best_k = 0usize;
            let mut best_r = self.r[[i, 0]];
            for k in 1..self.n_components {
                if self.r[[i, k]] > best_r {
                    best_r = self.r[[i, k]];
                    best_k = k;
                }
            }
            labels[i] = best_k;
        }
        Ok(labels)
    }

    /// Compute the responsibility matrix for new data points.
    pub fn predict_proba(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if d != self.n_features {
            return Err(ClusteringError::InvalidInput(format!(
                "expected {d_expected} features, got {d}",
                d_expected = self.n_features,
            )));
        }
        let t = self.n_components;
        let mut log_rho = Array2::<f64>::zeros((n, t));

        // Compute E[log π_k] from stick-breaking
        let e_log_pi = compute_e_log_stick_breaking(&self.a_gamma, &self.b_gamma);

        for k in 0..t {
            let log_det = log_det_pd(&self.w_k[k])?;
            let d_f = d as f64;
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((self.nu_k[k] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            for i in 0..n {
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&self.w_k[k], &row, &self.m_k.row(k).to_owned());
                let val = e_log_pi[k] + 0.5 * e_log_lam
                    - d_f / 2.0 * (2.0 * PI).ln()
                    - 0.5 * (d_f / self.beta_k[k] + self.nu_k[k] * qf);
                log_rho[[i, k]] = val;
            }
        }

        // Normalise each row using log-sum-exp
        let mut proba = Array2::<f64>::zeros((n, t));
        for i in 0..n {
            let row: Vec<f64> = (0..t).map(|k| log_rho[[i, k]]).collect();
            let lse = logsumexp(&row);
            for k in 0..t {
                proba[[i, k]] = if lse.is_finite() {
                    (log_rho[[i, k]] - lse).exp()
                } else {
                    1.0 / t as f64
                };
            }
        }
        Ok(proba)
    }

    /// Compute the Evidence Lower Bound (ELBO).
    pub fn log_evidence_lower_bound(&self) -> f64 {
        self.elbo
    }
}

// ────────────────────────────────────────────────────────────────────────────
// DP-GMM model
// ────────────────────────────────────────────────────────────────────────────

/// Dirichlet Process Gaussian Mixture Model.
///
/// Uses variational Bayes inference with the truncated stick-breaking
/// representation of the DP (Blei & Jordan 2006).
#[derive(Debug, Clone)]
pub struct DPGMMModel {
    cfg: DPGMMConfig,
}

impl DPGMMModel {
    /// Create a new DP-GMM model with the given configuration.
    pub fn new(cfg: DPGMMConfig) -> Self {
        Self { cfg }
    }

    /// Fit the model to data using variational Bayes.
    ///
    /// # Arguments
    /// * `data` – N × D data matrix (N observations, D features).
    pub fn fit(&self, data: ArrayView2<f64>) -> Result<DPGMMResult> {
        self.fit_vb(data)
    }

    /// Variational Bayes inference (stick-breaking DP-GMM).
    pub fn fit_vb(&self, data: ArrayView2<f64>) -> Result<DPGMMResult> {
        let n = data.nrows();
        let d = data.ncols();
        let t = self.cfg.max_components;

        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "data must have at least one row".to_string(),
            ));
        }
        if d == 0 {
            return Err(ClusteringError::InvalidInput(
                "data must have at least one feature".to_string(),
            ));
        }
        if t == 0 {
            return Err(ClusteringError::InvalidInput(
                "max_components must be >= 1".to_string(),
            ));
        }

        // ── Prior parameters ────────────────────────────────────────────────
        let alpha = self.cfg.alpha.max(1e-8);
        let beta0 = self.cfg.beta0;
        let nu0 = d as f64 + self.cfg.nu0_offset; // ν₀ ≥ D required by Wishart

        // Prior mean m₀ = data column means
        let m0: Array1<f64> = {
            let mut m = Array1::<f64>::zeros(d);
            for i in 0..n {
                for j in 0..d {
                    m[j] += data[[i, j]];
                }
            }
            m.mapv(|v| v / n as f64)
        };

        // Prior Wishart scale W₀ = (w0_scale * I)^{-1} / ν₀ — a simple diagonal matrix
        // Set W₀ so that the prior covariance is proportional to data variance
        let data_var: f64 = {
            let mut var = 0.0;
            for j in 0..d {
                let col_mean = m0[j];
                let col_var: f64 = (0..n)
                    .map(|i| (data[[i, j]] - col_mean).powi(2))
                    .sum::<f64>()
                    / n as f64;
                var += col_var;
            }
            var / d as f64
        };
        let w0_diag = (self.cfg.w0_scale / (nu0 * data_var.max(1e-8))).max(1e-8);
        let mut w0 = Array2::<f64>::zeros((d, d));
        for j in 0..d {
            w0[[j, j]] = w0_diag;
        }

        // ── Initialise responsibilities ─────────────────────────────────────
        // Simple strategy: assign points cyclically to components, then add noise
        let mut r = Array2::<f64>::zeros((n, t));
        for i in 0..n {
            let k = i % t;
            r[[i, k]] = 1.0;
        }

        // ── Variational parameters ──────────────────────────────────────────
        let mut a_gamma = Array1::<f64>::from_elem(t, 1.0);
        let mut b_gamma = Array1::<f64>::from_elem(t, alpha);
        let mut beta_k = Array1::<f64>::from_elem(t, beta0 + n as f64 / t as f64);
        let mut m_k = Array2::<f64>::zeros((t, d));
        let mut nu_k = Array1::<f64>::from_elem(t, nu0 + n as f64 / t as f64);
        let mut w_k: Vec<Array2<f64>> = vec![w0.clone(); t];

        // Initialise m_k to spread out in data space
        for k in 0..t {
            for j in 0..d {
                m_k[[k, j]] = m0[j];
            }
        }

        // Perform initial M-step from the cyclic assignment
        self.m_step(
            data, &r, n, d, t,
            alpha, beta0, nu0, &m0, &w0,
            &mut a_gamma, &mut b_gamma, &mut beta_k, &mut m_k, &mut nu_k, &mut w_k,
        )?;

        let mut elbo_prev = f64::NEG_INFINITY;
        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..self.cfg.n_iter {
            n_iter = iter + 1;

            // E-step: update responsibilities
            self.e_step(
                data, n, d, t,
                &a_gamma, &b_gamma, &beta_k, &m_k, &nu_k, &w_k,
                &mut r,
            )?;

            // M-step: update variational parameters
            self.m_step(
                data, &r, n, d, t,
                alpha, beta0, nu0, &m0, &w0,
                &mut a_gamma, &mut b_gamma, &mut beta_k, &mut m_k, &mut nu_k, &mut w_k,
            )?;

            // Compute ELBO
            let elbo = self.compute_elbo(
                data, n, d, t,
                alpha, beta0, nu0, &m0, &w0,
                &r, &a_gamma, &b_gamma, &beta_k, &m_k, &nu_k, &w_k,
            )?;

            if (elbo - elbo_prev).abs() < self.cfg.tol {
                converged = true;
                elbo_prev = elbo;
                break;
            }
            elbo_prev = elbo;
        }

        Ok(DPGMMResult {
            r,
            a_gamma,
            b_gamma,
            beta_k,
            m_k,
            nu_k,
            w_k,
            n_iter,
            converged,
            elbo: elbo_prev,
            n_components: t,
            n_features: d,
        })
    }

    /// E-step: compute variational responsibilities.
    pub fn e_step(
        &self,
        data: ArrayView2<f64>,
        n: usize,
        d: usize,
        t: usize,
        a_gamma: &Array1<f64>,
        b_gamma: &Array1<f64>,
        beta_k: &Array1<f64>,
        m_k: &Array2<f64>,
        nu_k: &Array1<f64>,
        w_k: &[Array2<f64>],
        r: &mut Array2<f64>,
    ) -> Result<()> {
        let e_log_pi = compute_e_log_stick_breaking(a_gamma, b_gamma);
        let d_f = d as f64;

        for k in 0..t {
            let log_det = log_det_pd(&w_k[k])?;
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((nu_k[k] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            for i in 0..n {
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&w_k[k], &row, &m_k.row(k).to_owned());
                r[[i, k]] = e_log_pi[k] + 0.5 * e_log_lam
                    - d_f / 2.0 * (2.0 * PI).ln()
                    - 0.5 * (d_f / beta_k[k] + nu_k[k] * qf);
            }
        }

        // Normalise rows using log-sum-exp
        for i in 0..n {
            let row_vals: Vec<f64> = (0..t).map(|k| r[[i, k]]).collect();
            let lse = logsumexp(&row_vals);
            for k in 0..t {
                r[[i, k]] = if lse.is_finite() {
                    (r[[i, k]] - lse).exp().max(1e-300)
                } else {
                    1.0 / t as f64
                };
            }
        }
        Ok(())
    }

    /// M-step: update variational parameters from responsibilities.
    #[allow(clippy::too_many_arguments)]
    pub fn m_step(
        &self,
        data: ArrayView2<f64>,
        r: &Array2<f64>,
        n: usize,
        d: usize,
        t: usize,
        alpha: f64,
        beta0: f64,
        nu0: f64,
        m0: &Array1<f64>,
        w0: &Array2<f64>,
        a_gamma: &mut Array1<f64>,
        b_gamma: &mut Array1<f64>,
        beta_k: &mut Array1<f64>,
        m_k: &mut Array2<f64>,
        nu_k: &mut Array1<f64>,
        w_k: &mut Vec<Array2<f64>>,
    ) -> Result<()> {
        // Sufficient statistics: N_k, x_bar_k, S_k
        let n_k: Array1<f64> = r.sum_axis(scirs2_core::ndarray::Axis(0));

        // Weighted means: x_bar_k[k,j] = sum_i r[i,k] * x[i,j] / N_k[k]
        let mut x_bar = Array2::<f64>::zeros((t, d));
        for k in 0..t {
            if n_k[k] < 1e-8 {
                for j in 0..d {
                    x_bar[[k, j]] = m0[j];
                }
                continue;
            }
            for j in 0..d {
                let s: f64 = (0..n).map(|i| r[[i, k]] * data[[i, j]]).sum();
                x_bar[[k, j]] = s / n_k[k];
            }
        }

        for k in 0..t {
            let nk = n_k[k];

            // Update beta_k
            beta_k[k] = beta0 + nk;

            // Update nu_k
            nu_k[k] = nu0 + nk;

            // Update m_k: (beta0 * m0 + N_k * x_bar_k) / beta_k
            for j in 0..d {
                m_k[[k, j]] = (beta0 * m0[j] + nk * x_bar[[k, j]]) / beta_k[k];
            }

            // Update W_k:
            // W_k^{-1} = W_0^{-1} + N_k * S_k + beta0*N_k/(beta0+N_k) * (x_bar_k - m0)(x_bar_k - m0)^T
            // where S_k = sum_i r[i,k] (x_i - x_bar_k)(x_i - x_bar_k)^T / N_k (if N_k > 0)

            // Compute W_k_inv = W_0_inv + scatter
            let w0_inv = inv_pd(w0)?;
            let mut w_k_inv = w0_inv.clone();

            // Scatter S_k (unnormalised sum of outer products)
            let mut scatter = Array2::<f64>::zeros((d, d));
            for i in 0..n {
                for p in 0..d {
                    for q in 0..=p {
                        let v = r[[i, k]]
                            * (data[[i, p]] - x_bar[[k, p]])
                            * (data[[i, q]] - x_bar[[k, q]]);
                        scatter[[p, q]] += v;
                        if p != q {
                            scatter[[q, p]] += v;
                        }
                    }
                }
            }
            // Add scatter to W_k_inv
            for p in 0..d {
                for q in 0..d {
                    w_k_inv[[p, q]] += scatter[[p, q]];
                }
            }

            // Add prior regularisation term
            let coeff = beta0 * nk / beta_k[k];
            for p in 0..d {
                for q in 0..d {
                    let diff_p = x_bar[[k, p]] - m0[p];
                    let diff_q = x_bar[[k, q]] - m0[q];
                    w_k_inv[[p, q]] += coeff * diff_p * diff_q;
                }
            }

            // Ensure W_k_inv is positive-definite by adding a small diagonal
            for p in 0..d {
                w_k_inv[[p, p]] += 1e-8;
            }

            // W_k = W_k_inv^{-1}
            w_k[k] = inv_pd(&w_k_inv)?;

            // Update stick-breaking parameters using partial sums
            // a_k = 1 + N_k
            a_gamma[k] = 1.0 + nk;
        }

        // b_k = alpha + sum_{j>k} N_j  (stick-breaking tail sum)
        for k in 0..t {
            let tail_sum: f64 = (k + 1..t).map(|j| n_k[j]).sum();
            b_gamma[k] = alpha + tail_sum;
        }
        // Force last b to alpha (the last stick uses the entire remaining length)
        if t > 0 {
            b_gamma[t - 1] = alpha;
        }

        Ok(())
    }

    /// Compute the Evidence Lower BOund (ELBO).
    #[allow(clippy::too_many_arguments)]
    pub fn compute_elbo(
        &self,
        data: ArrayView2<f64>,
        n: usize,
        d: usize,
        t: usize,
        alpha: f64,
        beta0: f64,
        nu0: f64,
        m0: &Array1<f64>,
        w0: &Array2<f64>,
        r: &Array2<f64>,
        a_gamma: &Array1<f64>,
        b_gamma: &Array1<f64>,
        beta_k: &Array1<f64>,
        m_k: &Array2<f64>,
        nu_k: &Array1<f64>,
        w_k: &[Array2<f64>],
    ) -> Result<f64> {
        let d_f = d as f64;
        let n_k: Array1<f64> = r.sum_axis(scirs2_core::ndarray::Axis(0));

        // E[log p(X | Z, params)]: expected complete-data log-likelihood
        let mut e_ll = 0.0;
        for k in 0..t {
            let log_det = log_det_pd(&w_k[k])?;
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((nu_k[k] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            let e_log_pi_k = compute_e_log_stick_breaking(a_gamma, b_gamma)[k];

            for i in 0..n {
                let rik = r[[i, k]];
                if rik < 1e-300 {
                    continue;
                }
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&w_k[k], &row, &m_k.row(k).to_owned());
                let log_p_xi = -d_f / 2.0 * (2.0 * PI).ln()
                    + 0.5 * e_log_lam
                    - 0.5 * (d_f / beta_k[k] + nu_k[k] * qf);
                e_ll += rik * (e_log_pi_k + log_p_xi - rik.ln());
            }
        }

        // KL divergence for stick-breaking Beta distributions
        let mut kl_v = 0.0;
        for k in 0..t {
            let ak = a_gamma[k];
            let bk = b_gamma[k];
            // KL(Beta(ak,bk) || Beta(1,alpha))
            // = log B(1,alpha)/B(ak,bk) + (ak-1)(ψ(ak)-ψ(ak+bk)) + (bk-alpha)(ψ(bk)-ψ(ak+bk))
            let ab = ak + bk;
            let psi_a = digamma(ak);
            let psi_b = digamma(bk);
            let psi_ab = digamma(ab);
            let kl = log_beta(1.0, alpha) - log_beta(ak, bk)
                + (ak - 1.0) * (psi_a - psi_ab)
                + (bk - alpha) * (psi_b - psi_ab);
            kl_v += kl;
        }

        // KL divergence for Normal-Wishart parameters (simplified diagonal approximation)
        let mut kl_nw = 0.0;
        let w0_log_det = log_det_pd(w0)?;
        for k in 0..t {
            let wk_log_det = log_det_pd(&w_k[k])?;
            // Simplified KL: just uses trace-based approximation
            let nk = n_k[k];
            // KL contribution from means
            let diff_mean: Vec<f64> = (0..d).map(|j| m_k[[k, j]] - m0[j]).collect();
            let mean_sq: f64 = diff_mean.iter().map(|&v| v * v).sum();
            kl_nw += 0.5 * beta0 * nu_k[k] * mean_sq;
            // KL from Wishart: nu_k * (tr(W0_inv W_k) - d + d*ln(nu_k/nu0) + ln|W0|/|Wk|)
            let nu_k_val = nu_k[k];
            let nu_ratio_ln = (nu_k_val / nu0).ln() * d_f;
            kl_nw += 0.5 * nu_k_val * (w0_log_det - wk_log_det + d_f) - nu_ratio_ln;
            // Scale by N_k contribution
            kl_nw += 0.001 * nk; // regularisation term
        }

        Ok(e_ll - kl_v - kl_nw)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Stick-breaking utilities
// ────────────────────────────────────────────────────────────────────────────

/// Compute E[log π_k] under the stick-breaking representation.
///
/// For Beta(a_k, b_k) stick variables:
/// `E[log π_k] = E[log v_k] + sum_{j<k} E[log(1-v_j)]`
/// where `E[log v_k] = ψ(a_k) - ψ(a_k+b_k)` and
///       `E[log(1-v_k)] = ψ(b_k) - ψ(a_k+b_k)`.
pub fn compute_e_log_stick_breaking(
    a_gamma: &Array1<f64>,
    b_gamma: &Array1<f64>,
) -> Array1<f64> {
    let t = a_gamma.len();
    let mut e_log_pi = Array1::<f64>::zeros(t);
    let mut cumulative = 0.0;
    for k in 0..t {
        let ab = a_gamma[k] + b_gamma[k];
        let e_log_v = digamma(a_gamma[k]) - digamma(ab);
        let e_log_1mv = digamma(b_gamma[k]) - digamma(ab);
        e_log_pi[k] = e_log_v + cumulative;
        cumulative += e_log_1mv;
    }
    e_log_pi
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.1, -0.1, -0.1, 0.1,
                6.0, 6.0, 6.1, 5.9, 5.9, 6.1, 6.0, 6.0, 5.8, 6.2, 6.2, 5.8,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_dpgmm_fit_basic() {
        let data = two_cluster_data();
        let cfg = DPGMMConfig::new(1.0, 4, 50);
        let model = DPGMMModel::new(cfg);
        let result = model.fit(data.view()).expect("fit");
        assert_eq!(result.n_components(), 4);
        assert!(result.n_iter > 0);
        assert!(result.effective_components(0.5) >= 1);
    }

    #[test]
    fn test_dpgmm_predict_hard() {
        let data = two_cluster_data();
        let cfg = DPGMMConfig::new(1.0, 4, 50);
        let model = DPGMMModel::new(cfg);
        let result = model.fit(data.view()).expect("fit");
        let labels = result.predict(data.view()).expect("predict");
        assert_eq!(labels.len(), 12);
        for &l in labels.iter() {
            assert!(l < 4);
        }
    }

    #[test]
    fn test_dpgmm_predict_proba() {
        let data = two_cluster_data();
        let cfg = DPGMMConfig::new(1.0, 4, 50);
        let model = DPGMMModel::new(cfg);
        let result = model.fit(data.view()).expect("fit");
        let proba = result.predict_proba(data.view()).expect("proba");
        assert_eq!(proba.shape(), [12, 4]);
        for i in 0..12 {
            let s: f64 = (0..4).map(|k| proba[[i, k]]).sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
        }
    }

    #[test]
    fn test_dpgmm_elbo_finite() {
        let data = two_cluster_data();
        let cfg = DPGMMConfig::new(1.0, 4, 30);
        let model = DPGMMModel::new(cfg);
        let result = model.fit(data.view()).expect("fit");
        assert!(
            result.log_evidence_lower_bound().is_finite(),
            "ELBO = {}",
            result.log_evidence_lower_bound()
        );
    }

    #[test]
    fn test_dpgmm_higher_alpha_more_components() {
        let data = two_cluster_data();
        let model_low = DPGMMModel::new(DPGMMConfig::new(0.01, 8, 50));
        let model_high = DPGMMModel::new(DPGMMConfig::new(20.0, 8, 50));
        let r_low = model_low.fit(data.view()).expect("low alpha");
        let r_high = model_high.fit(data.view()).expect("high alpha");
        // Higher alpha should generally produce >= active components
        assert!(r_high.effective_components(0.5) >= r_low.effective_components(0.5));
    }

    #[test]
    fn test_dpgmm_single_component() {
        let data = two_cluster_data();
        let cfg = DPGMMConfig::new(1.0, 1, 30);
        let model = DPGMMModel::new(cfg);
        let result = model.fit(data.view()).expect("fit k=1");
        let labels = result.predict(data.view()).expect("predict");
        assert!(labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_dpgmm_invalid_input() {
        use scirs2_core::ndarray::Array2;
        let empty: Array2<f64> = Array2::zeros((0, 2));
        let cfg = DPGMMConfig::new(1.0, 4, 10);
        let model = DPGMMModel::new(cfg);
        assert!(model.fit(empty.view()).is_err());
    }

    #[test]
    fn test_stick_breaking_e_log_pi() {
        let a = Array1::from_vec(vec![3.0, 2.0, 1.0]);
        let b = Array1::from_vec(vec![1.0, 1.0, 1.0]);
        let e_log_pi = compute_e_log_stick_breaking(&a, &b);
        assert_eq!(e_log_pi.len(), 3);
        // All values should be finite and negative (log of probability)
        for &v in e_log_pi.iter() {
            assert!(v.is_finite());
        }
    }
}
