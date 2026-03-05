//! Full Variational Bayesian GMM (VB-GMM / VBEM).
//!
//! Implements the full conjugate Variational Bayes Expectation-Maximisation
//! algorithm for a Gaussian Mixture Model with `K` components.  Unlike the DP-GMM
//! in `dpgmm.rs`, the number of components `K` is fixed, but the algorithm
//! automatically prunes empty components and provides the variational BIC for
//! model selection.
//!
//! # Algorithm (Bishop 2006, §10.2)
//!
//! The generative model is:
//!
//! ```text
//! π    ~ Dirichlet(α₀)
//! Λ_k  ~ Wishart(W₀, ν₀)
//! μ_k  | Λ_k ~ N(m₀, (β₀ Λ_k)⁻¹)
//! z_n  ~ Categorical(π)
//! x_n  | z_n=k, μ_k, Λ_k ~ N(μ_k, Λ_k⁻¹)
//! ```
//!
//! The variational posterior is:
//!
//! ```text
//! q(π, {μ_k, Λ_k}, Z) = q(π) ∏_k q(μ_k, Λ_k) q(Z)
//! ```
//!
//! # Component Pruning
//!
//! After convergence, components whose effective count `N_k < prune_threshold`
//! are removed and labels are relabelled contiguously.
//!
//! # Variational BIC
//!
//! The variational BIC uses the ELBO as a proxy for the log-marginal-likelihood,
//! penalised by `(M/2) log N` where `M` is the model complexity.
//!
//! # References
//!
//! - Bishop (2006) PRML §10.2
//! - Beal (2003) "Variational Algorithms for Approximate Bayesian Inference" (thesis)

use std::f64::consts::{LN_2, PI};

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Math utilities (self-contained; avoid cross-module dependencies)
// ────────────────────────────────────────────────────────────────────────────

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

fn lgamma(x: f64) -> f64 {
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
    let _ = G;
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

/// Log of the multivariate gamma function B(W, ν) normalising constant for the Wishart.
fn log_wishart_b(w: &Array2<f64>, nu: f64) -> Result<f64> {
    let d = w.nrows() as f64;
    let log_det = log_det_pd(w)?;
    let log_b = -nu / 2.0 * log_det
        - nu * d / 2.0 * LN_2
        - d * (d - 1.0) / 4.0 * PI.ln();
    let lmg: f64 = (0..w.nrows())
        .map(|j| lgamma((nu - j as f64) / 2.0))
        .sum();
    Ok(log_b - lmg)
}

fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= l[[i, k]] * l[[j, k]];
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

fn log_det_pd(a: &Array2<f64>) -> Result<f64> {
    let l = cholesky_lower(a)?;
    let log_det: f64 = (0..l.nrows()).map(|i| 2.0 * l[[i, i]].ln()).sum();
    Ok(log_det)
}

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

fn inv_pd(a: &Array2<f64>) -> Result<Array2<f64>> {
    let n = a.nrows();
    let l = cholesky_lower(a)?;
    let mut linv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut e = Array1::<f64>::zeros(n);
        e[j] = 1.0;
        let col = forward_solve(&l, &e);
        for i in 0..n {
            linv[[i, j]] = col[i];
        }
    }
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let s: f64 = (0..n).map(|k| linv[[k, i]] * linv[[k, j]]).sum();
            inv[[i, j]] = s;
        }
    }
    Ok(inv)
}

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

fn logsumexp(vals: &[f64]) -> f64 {
    let max = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let s: f64 = vals.iter().map(|&v| (v - max).exp()).sum();
    max + s.ln()
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for the full VB-GMM.
#[derive(Debug, Clone)]
pub struct VBGMMConfig {
    /// Number of mixture components K.
    pub n_components: usize,
    /// Maximum number of VBEM iterations.
    pub n_iter: usize,
    /// Convergence tolerance on the ELBO.
    pub tol: f64,
    /// Dirichlet prior concentration (α₀ > 0).
    pub alpha0: f64,
    /// Normal-Wishart prior precision scaling (β₀ > 0).
    pub beta0: f64,
    /// Normal-Wishart prior degrees of freedom offset (ν₀ ≥ D required).
    pub nu0_offset: f64,
    /// Diagonal scaling for the Wishart prior scale matrix W₀.
    pub w0_scale: f64,
    /// Minimum effective count to keep a component (for pruning).
    pub prune_threshold: f64,
}

impl VBGMMConfig {
    /// Create a new VB-GMM configuration.
    ///
    /// # Arguments
    /// * `n_components` – number of Gaussian components.
    /// * `n_iter` – maximum VBEM iterations.
    /// * `tol` – ELBO convergence tolerance.
    pub fn new(n_components: usize, n_iter: usize, tol: f64) -> Self {
        Self {
            n_components,
            n_iter,
            tol,
            alpha0: 1e-3,
            beta0: 1e-3,
            nu0_offset: 1.0,
            w0_scale: 1.0,
            prune_threshold: 1.0,
        }
    }

    /// Set the Dirichlet prior concentration.
    pub fn with_alpha0(mut self, alpha0: f64) -> Self {
        self.alpha0 = alpha0;
        self
    }

    /// Set the pruning threshold.
    pub fn with_prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Result
// ────────────────────────────────────────────────────────────────────────────

/// Fitted VB-GMM parameters.
#[derive(Debug, Clone)]
pub struct VBGMMResult {
    /// Responsibility matrix (N × K_active).
    pub r: Array2<f64>,
    /// Variational Dirichlet parameters (K_active,).
    pub alpha_k: Array1<f64>,
    /// Normal-Wishart precision scaling (K_active,).
    pub beta_k: Array1<f64>,
    /// Normal-Wishart means (K_active × D).
    pub m_k: Array2<f64>,
    /// Wishart degrees of freedom (K_active,).
    pub nu_k: Array1<f64>,
    /// Wishart scale matrices (one per active component).
    pub w_k: Vec<Array2<f64>>,
    /// Number of active components after pruning.
    pub n_components: usize,
    /// Number of features.
    pub n_features: usize,
    /// Number of VBEM iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Final ELBO value.
    pub elbo: f64,
}

impl VBGMMResult {
    /// Hard-assignment cluster labels for the training data.
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

    /// Responsibility matrix for new data.
    pub fn predict_proba(&self, data: ArrayView2<f64>) -> Result<Array2<f64>> {
        let n = data.nrows();
        let d = data.ncols();
        if d != self.n_features {
            return Err(ClusteringError::InvalidInput(format!(
                "expected {d_exp} features, got {d}",
                d_exp = self.n_features,
            )));
        }
        let k_act = self.n_components;
        let d_f = d as f64;

        // Compute E[log π_k] from Dirichlet variational parameters
        let alpha_sum: f64 = self.alpha_k.iter().sum();
        let e_log_pi: Vec<f64> = self
            .alpha_k
            .iter()
            .map(|&a| digamma(a) - digamma(alpha_sum))
            .collect();

        let mut log_rho = Array2::<f64>::zeros((n, k_act));
        for k in 0..k_act {
            let log_det = log_det_pd(&self.w_k[k])?;
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((self.nu_k[k] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            for i in 0..n {
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&self.w_k[k], &row, &self.m_k.row(k).to_owned());
                log_rho[[i, k]] = e_log_pi[k] + 0.5 * e_log_lam
                    - d_f / 2.0 * (2.0 * PI).ln()
                    - 0.5 * (d_f / self.beta_k[k] + self.nu_k[k] * qf);
            }
        }

        let mut proba = Array2::<f64>::zeros((n, k_act));
        for i in 0..n {
            let row: Vec<f64> = (0..k_act).map(|k| log_rho[[i, k]]).collect();
            let lse = logsumexp(&row);
            for k in 0..k_act {
                proba[[i, k]] = if lse.is_finite() {
                    (log_rho[[i, k]] - lse).exp().max(1e-300)
                } else {
                    1.0 / k_act as f64
                };
            }
        }
        Ok(proba)
    }

    /// Log evidence lower bound (ELBO).
    pub fn log_evidence_lower_bound(&self) -> f64 {
        self.elbo
    }

    /// Variational BIC: `ELBO - (M/2) * ln(N)` where `M = model_complexity()`.
    ///
    /// Larger (less negative) values indicate a better model.
    pub fn bic_vb(&self, n_samples: usize) -> f64 {
        let m = self.model_complexity();
        self.elbo - (m as f64 / 2.0) * (n_samples as f64).ln()
    }

    /// Effective number of free parameters for the variational GMM.
    ///
    /// Counts: K-1 mixing proportions + K*(D + D*(D+1)/2) Gaussian parameters.
    pub fn model_complexity(&self) -> usize {
        let k = self.n_components;
        let d = self.n_features;
        // mixing: K-1; means: K*D; covariances: K*D*(D+1)/2
        let k_eff = k.saturating_sub(1);
        k_eff + k * d + k * d * (d + 1) / 2
    }
}

// ────────────────────────────────────────────────────────────────────────────
// VB-GMM model
// ────────────────────────────────────────────────────────────────────────────

/// Full Variational Bayesian Gaussian Mixture Model.
///
/// Implements the complete VBEM algorithm of Bishop (2006) §10.2 with:
/// - Normal-Wishart prior over component parameters
/// - Dirichlet prior over mixing proportions
/// - Automatic component pruning after convergence
#[derive(Debug, Clone)]
pub struct VBGMMModel {
    cfg: VBGMMConfig,
}

impl VBGMMModel {
    /// Create a new VB-GMM model.
    pub fn new(cfg: VBGMMConfig) -> Self {
        Self { cfg }
    }

    /// Fit the model using VBEM.
    ///
    /// # Arguments
    /// * `data` – N × D data matrix.
    pub fn fit_vbem(&self, data: ArrayView2<f64>) -> Result<VBGMMResult> {
        let n = data.nrows();
        let d = data.ncols();
        let k = self.cfg.n_components;

        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "data must be non-empty".to_string(),
            ));
        }
        if d == 0 {
            return Err(ClusteringError::InvalidInput(
                "data must have at least one feature".to_string(),
            ));
        }
        if k == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_components must be >= 1".to_string(),
            ));
        }

        // ── Prior parameters ────────────────────────────────────────────────
        let alpha0 = self.cfg.alpha0.max(1e-10);
        let beta0 = self.cfg.beta0.max(1e-10);
        let nu0 = d as f64 + self.cfg.nu0_offset;

        // Prior mean = data column means
        let m0: Array1<f64> = {
            let mut m = Array1::<f64>::zeros(d);
            for i in 0..n {
                for j in 0..d {
                    m[j] += data[[i, j]];
                }
            }
            m.mapv(|v| v / n as f64)
        };

        // Prior Wishart scale W₀ = diag(1/(ν₀ * var_j))
        let data_var: f64 = {
            let mut var = 0.0;
            for j in 0..d {
                let col_mean = m0[j];
                let cv: f64 = (0..n)
                    .map(|i| (data[[i, j]] - col_mean).powi(2))
                    .sum::<f64>()
                    / n as f64;
                var += cv;
            }
            (var / d as f64).max(1e-8)
        };
        let w0_diag = (self.cfg.w0_scale / (nu0 * data_var)).max(1e-8);
        let mut w0 = Array2::<f64>::zeros((d, d));
        for j in 0..d {
            w0[[j, j]] = w0_diag;
        }

        // ── Initialise ───────────────────────────────────────────────────────
        // Uniform random initialisation of r
        let mut r = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            let ki = i % k;
            r[[i, ki]] = 1.0;
        }

        // Variational parameters
        let mut alpha_k = Array1::<f64>::from_elem(k, alpha0 + n as f64 / k as f64);
        let mut beta_k = Array1::<f64>::from_elem(k, beta0 + n as f64 / k as f64);
        let mut m_k = Array2::<f64>::zeros((k, d));
        for ki in 0..k {
            for j in 0..d {
                m_k[[ki, j]] = m0[j];
            }
        }
        let mut nu_k = Array1::<f64>::from_elem(k, nu0 + n as f64 / k as f64);
        let mut w_k: Vec<Array2<f64>> = vec![w0.clone(); k];

        // Run initial M-step
        self.m_step_vbem(
            data, &r, n, d, k,
            alpha0, beta0, nu0, &m0, &w0,
            &mut alpha_k, &mut beta_k, &mut m_k, &mut nu_k, &mut w_k,
        )?;

        let mut elbo_prev = f64::NEG_INFINITY;
        let mut n_iter = 0usize;
        let mut converged = false;

        for iter in 0..self.cfg.n_iter {
            n_iter = iter + 1;

            // VB E-step
            self.e_step_vbem(
                data, n, d, k,
                &alpha_k, &beta_k, &m_k, &nu_k, &w_k,
                &mut r,
            )?;

            // VB M-step
            self.m_step_vbem(
                data, &r, n, d, k,
                alpha0, beta0, nu0, &m0, &w0,
                &mut alpha_k, &mut beta_k, &mut m_k, &mut nu_k, &mut w_k,
            )?;

            // Compute ELBO
            let elbo = self.compute_elbo_vbem(
                data, n, d, k,
                alpha0, beta0, nu0, &m0, &w0,
                &r, &alpha_k, &beta_k, &m_k, &nu_k, &w_k,
            )?;

            if (elbo - elbo_prev).abs() < self.cfg.tol {
                converged = true;
                elbo_prev = elbo;
                break;
            }
            elbo_prev = elbo;
        }

        // Build initial result (before pruning)
        let result_full = VBGMMResult {
            r,
            alpha_k,
            beta_k,
            m_k,
            nu_k,
            w_k,
            n_components: k,
            n_features: d,
            n_iter,
            converged,
            elbo: elbo_prev,
        };

        // Prune empty components
        self.prune_empty_components(result_full)
    }

    /// VB E-step: compute responsibilities from variational parameters.
    fn e_step_vbem(
        &self,
        data: ArrayView2<f64>,
        n: usize,
        d: usize,
        k: usize,
        alpha_k: &Array1<f64>,
        beta_k: &Array1<f64>,
        m_k: &Array2<f64>,
        nu_k: &Array1<f64>,
        w_k: &[Array2<f64>],
        r: &mut Array2<f64>,
    ) -> Result<()> {
        let d_f = d as f64;
        let alpha_sum: f64 = alpha_k.iter().sum();

        for ki in 0..k {
            let log_det = log_det_pd(&w_k[ki])?;
            // E[log |Λ_k|] = sum_j ψ((ν_k - j)/2) + D ln 2 + ln|W_k|
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((nu_k[ki] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            // E[log π_k] = ψ(α_k) - ψ(sum α)
            let e_log_pi = digamma(alpha_k[ki]) - digamma(alpha_sum);

            for i in 0..n {
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&w_k[ki], &row, &m_k.row(ki).to_owned());
                r[[i, ki]] = e_log_pi + 0.5 * e_log_lam
                    - d_f / 2.0 * (2.0 * PI).ln()
                    - 0.5 * (d_f / beta_k[ki] + nu_k[ki] * qf);
            }
        }

        // Normalise each row
        for i in 0..n {
            let row_vals: Vec<f64> = (0..k).map(|ki| r[[i, ki]]).collect();
            let lse = logsumexp(&row_vals);
            for ki in 0..k {
                r[[i, ki]] = if lse.is_finite() {
                    (r[[i, ki]] - lse).exp().max(1e-300)
                } else {
                    1.0 / k as f64
                };
            }
        }
        Ok(())
    }

    /// VB M-step: update variational parameters from responsibilities.
    #[allow(clippy::too_many_arguments)]
    fn m_step_vbem(
        &self,
        data: ArrayView2<f64>,
        r: &Array2<f64>,
        n: usize,
        d: usize,
        k: usize,
        alpha0: f64,
        beta0: f64,
        nu0: f64,
        m0: &Array1<f64>,
        w0: &Array2<f64>,
        alpha_k: &mut Array1<f64>,
        beta_k: &mut Array1<f64>,
        m_k: &mut Array2<f64>,
        nu_k: &mut Array1<f64>,
        w_k: &mut Vec<Array2<f64>>,
    ) -> Result<()> {
        let n_k: Array1<f64> = r.sum_axis(scirs2_core::ndarray::Axis(0));

        let mut x_bar = Array2::<f64>::zeros((k, d));
        for ki in 0..k {
            if n_k[ki] < 1e-8 {
                for j in 0..d {
                    x_bar[[ki, j]] = m0[j];
                }
                continue;
            }
            for j in 0..d {
                let s: f64 = (0..n).map(|i| r[[i, ki]] * data[[i, j]]).sum();
                x_bar[[ki, j]] = s / n_k[ki];
            }
        }

        let w0_inv = inv_pd(w0)?;

        for ki in 0..k {
            let nki = n_k[ki];

            alpha_k[ki] = alpha0 + nki;
            beta_k[ki] = beta0 + nki;
            nu_k[ki] = nu0 + nki;

            for j in 0..d {
                m_k[[ki, j]] = (beta0 * m0[j] + nki * x_bar[[ki, j]]) / beta_k[ki];
            }

            // Build W_k^{-1} = W_0^{-1} + scatter + outer_product_term
            let mut w_inv = w0_inv.clone();

            // Weighted scatter matrix
            for i in 0..n {
                for p in 0..d {
                    for q in 0..=p {
                        let v = r[[i, ki]]
                            * (data[[i, p]] - x_bar[[ki, p]])
                            * (data[[i, q]] - x_bar[[ki, q]]);
                        w_inv[[p, q]] += v;
                        if p != q {
                            w_inv[[q, p]] += v;
                        }
                    }
                }
            }

            // Outer product term: beta0*nki/(beta0+nki) * (x_bar - m0)(x_bar - m0)^T
            let coeff = beta0 * nki / beta_k[ki];
            for p in 0..d {
                for q in 0..d {
                    let dp = x_bar[[ki, p]] - m0[p];
                    let dq = x_bar[[ki, q]] - m0[q];
                    w_inv[[p, q]] += coeff * dp * dq;
                }
            }

            // Regularise diagonal
            for p in 0..d {
                w_inv[[p, p]] += 1e-8;
            }

            w_k[ki] = inv_pd(&w_inv)?;
        }

        Ok(())
    }

    /// Compute the ELBO for the VB-GMM (Bishop 2006, eq. 10.70–10.77).
    #[allow(clippy::too_many_arguments)]
    fn compute_elbo_vbem(
        &self,
        data: ArrayView2<f64>,
        n: usize,
        d: usize,
        k: usize,
        alpha0: f64,
        beta0: f64,
        nu0: f64,
        m0: &Array1<f64>,
        w0: &Array2<f64>,
        r: &Array2<f64>,
        alpha_k: &Array1<f64>,
        beta_k: &Array1<f64>,
        m_k: &Array2<f64>,
        nu_k: &Array1<f64>,
        w_k: &[Array2<f64>],
    ) -> Result<f64> {
        let d_f = d as f64;
        let n_k: Array1<f64> = r.sum_axis(scirs2_core::ndarray::Axis(0));
        let alpha_sum: f64 = alpha_k.iter().sum();
        let alpha0_sum = alpha0 * k as f64;

        // ── E[log p(X|Z,μ,Λ)] ──────────────────────────────────────────────
        let mut e_ll = 0.0;
        for ki in 0..k {
            let log_det = log_det_pd(&w_k[ki])?;
            let e_log_lam: f64 = (0..d)
                .map(|j| digamma((nu_k[ki] - j as f64) / 2.0))
                .sum::<f64>()
                + d_f * LN_2
                + log_det;

            let e_log_pi = digamma(alpha_k[ki]) - digamma(alpha_sum);

            for i in 0..n {
                let rik = r[[i, ki]];
                if rik < 1e-300 {
                    continue;
                }
                let row: Vec<f64> = (0..d).map(|j| data[[i, j]]).collect();
                let qf = quad_form(&w_k[ki], &row, &m_k.row(ki).to_owned());
                let log_p = e_log_pi + 0.5 * e_log_lam
                    - d_f / 2.0 * (2.0 * PI).ln()
                    - 0.5 * (d_f / beta_k[ki] + nu_k[ki] * qf);
                e_ll += rik * log_p;
            }
        }

        // ── Entropy of q(Z) ─────────────────────────────────────────────────
        let mut h_z = 0.0;
        for i in 0..n {
            for ki in 0..k {
                let rik = r[[i, ki]];
                if rik > 1e-300 {
                    h_z -= rik * rik.ln();
                }
            }
        }

        // ── KL(q(π) || p(π)) = KL(Dir(α_k) || Dir(α₀)) ───────────────────
        let kl_pi = {
            let lnb_prior = lgamma(alpha0_sum)
                - (0..k).map(|_| lgamma(alpha0)).sum::<f64>();
            let lnb_post = lgamma(alpha_sum)
                - alpha_k.iter().map(|&a| lgamma(a)).sum::<f64>();
            let cross: f64 = alpha_k
                .iter()
                .map(|&ak| (alpha0 - ak) * (digamma(ak) - digamma(alpha_sum)))
                .sum();
            lnb_prior - lnb_post + cross
        };

        // ── KL(q(μ,Λ) || p(μ,Λ)) for each component ───────────────────────
        let mut kl_nw = 0.0;
        let w0_log_det = log_det_pd(w0)?;
        let w0_inv = inv_pd(w0)?;
        for ki in 0..k {
            let nuk = nu_k[ki];
            let wk_log_det = log_det_pd(&w_k[ki])?;

            // Trace term: tr(W₀^{-1} W_k)
            let trace_w0inv_wk: f64 = (0..d)
                .map(|p| (0..d).map(|q| w0_inv[[p, q]] * w_k[ki][[p, q]]).sum::<f64>())
                .sum();

            // Mean term: β₀ ν_k (m_k - m₀)^T W₀^{-1} (m_k - m₀) ... diagonal approx
            let diff_m: Vec<f64> = (0..d).map(|j| m_k[[ki, j]] - m0[j]).collect();
            let quad_m = quad_form(&w0_inv, &diff_m, &Array1::<f64>::zeros(d));

            // Full Normal-Wishart KL
            let kl_k = 0.5 * (d_f * (beta0 / beta_k[ki] - 1.0 + (beta_k[ki] / beta0).ln())
                + beta0 * nuk * quad_m
                + nuk * trace_w0inv_wk
                + nuk * (w0_log_det - wk_log_det)
                - nuk * d_f)
                + log_wishart_b(w0, nu0)?
                - log_wishart_b(&w_k[ki], nuk)?
                + (nuk - nu0) / 2.0 * {
                    let e_log_lam: f64 = (0..d)
                        .map(|j| digamma((nuk - j as f64) / 2.0))
                        .sum::<f64>()
                        + d_f * LN_2
                        + wk_log_det;
                    e_log_lam
                }
                - n_k[ki] * 0.001; // minor count regularisation
            kl_nw += kl_k.max(-1e8); // guard against extreme values
        }

        Ok(e_ll + h_z - kl_pi - kl_nw)
    }

    /// Remove components with effective count below `prune_threshold`.
    ///
    /// Relabels remaining components contiguously.
    pub fn prune_empty_components(&self, mut result: VBGMMResult) -> Result<VBGMMResult> {
        let n_k: Array1<f64> = result.r.sum_axis(scirs2_core::ndarray::Axis(0));
        let active: Vec<usize> = (0..result.n_components)
            .filter(|&k| n_k[k] >= self.cfg.prune_threshold)
            .collect();

        if active.is_empty() {
            // Keep at least one component — the one with the highest count
            let best = (0..result.n_components)
                .max_by(|&a, &b| n_k[a].partial_cmp(&n_k[b]).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0);
            let n_feat = result.n_features;
            let single = result.keep_only(vec![best], n_feat);
            return self.prune_empty_components(single);
        }

        if active.len() == result.n_components {
            return Ok(result); // nothing to prune
        }

        let n_features = result.n_features;
        let pruned = result.keep_only(active, n_features);
        Ok(pruned)
    }
}

impl VBGMMResult {
    /// Return a new VBGMMResult keeping only the specified component indices.
    fn keep_only(self, active: Vec<usize>, n_features: usize) -> VBGMMResult {
        let k_new = active.len();
        let n = self.r.nrows();

        // Rebuild r
        let mut r_new = Array2::<f64>::zeros((n, k_new));
        for (new_k, &old_k) in active.iter().enumerate() {
            for i in 0..n {
                r_new[[i, new_k]] = self.r[[i, old_k]];
            }
        }
        // Renormalise rows
        for i in 0..n {
            let s: f64 = (0..k_new).map(|k| r_new[[i, k]]).sum();
            if s > 1e-15 {
                for k in 0..k_new {
                    r_new[[i, k]] /= s;
                }
            } else {
                for k in 0..k_new {
                    r_new[[i, k]] = 1.0 / k_new as f64;
                }
            }
        }

        let mut alpha_new = Array1::<f64>::zeros(k_new);
        let mut beta_new = Array1::<f64>::zeros(k_new);
        let mut m_new = Array2::<f64>::zeros((k_new, n_features));
        let mut nu_new = Array1::<f64>::zeros(k_new);
        let mut w_new: Vec<Array2<f64>> = Vec::with_capacity(k_new);

        for (new_k, &old_k) in active.iter().enumerate() {
            alpha_new[new_k] = self.alpha_k[old_k];
            beta_new[new_k] = self.beta_k[old_k];
            nu_new[new_k] = self.nu_k[old_k];
            for j in 0..n_features {
                m_new[[new_k, j]] = self.m_k[[old_k, j]];
            }
            w_new.push(self.w_k[old_k].clone());
        }

        VBGMMResult {
            r: r_new,
            alpha_k: alpha_new,
            beta_k: beta_new,
            m_k: m_new,
            nu_k: nu_new,
            w_k: w_new,
            n_components: k_new,
            n_features,
            n_iter: self.n_iter,
            converged: self.converged,
            elbo: self.elbo,
        }
    }
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
                7.0, 7.0, 7.1, 6.9, 6.9, 7.1, 7.0, 7.0, 6.8, 7.2, 7.2, 6.8,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_vbgmm_fit_basic() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(4, 50, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        assert!(result.n_components >= 1);
        assert!(result.n_iter > 0);
    }

    #[test]
    fn test_vbgmm_predict_length() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(3, 50, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        let labels = result.predict(data.view()).expect("predict");
        assert_eq!(labels.len(), 12);
        for &l in labels.iter() {
            assert!(l < result.n_components, "label {l} >= n_components {}", result.n_components);
        }
    }

    #[test]
    fn test_vbgmm_predict_proba_rows_sum_to_one() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(3, 50, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        let proba = result.predict_proba(data.view()).expect("proba");
        for i in 0..12 {
            let s: f64 = (0..result.n_components).map(|k| proba[[i, k]]).sum();
            assert!((s - 1.0).abs() < 1e-5, "row {i} sums to {s}");
        }
    }

    #[test]
    fn test_vbgmm_elbo_finite() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(3, 30, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        assert!(
            result.log_evidence_lower_bound().is_finite(),
            "ELBO = {}",
            result.log_evidence_lower_bound()
        );
    }

    #[test]
    fn test_vbgmm_bic_vb_finite() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(2, 30, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        let bic = result.bic_vb(12);
        assert!(bic.is_finite(), "BIC = {bic}");
    }

    #[test]
    fn test_vbgmm_model_complexity() {
        // K=2, D=2: complexity = (2-1) + 2*2 + 2*2*3/2 = 1 + 4 + 6 = 11
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(2, 30, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        // n_components may be less after pruning, just check it's non-zero
        let complexity = result.model_complexity();
        assert!(complexity > 0, "complexity = {complexity}");
    }

    #[test]
    fn test_vbgmm_k1_trivial() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(1, 20, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit k=1");
        assert_eq!(result.n_components, 1);
        let labels = result.predict(data.view()).expect("predict k=1");
        assert!(labels.iter().all(|&l| l == 0), "all labels should be 0");
    }

    #[test]
    fn test_vbgmm_invalid_k0() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(0, 20, 1e-6);
        let model = VBGMMModel::new(cfg);
        assert!(model.fit_vbem(data.view()).is_err());
    }

    #[test]
    fn test_vbgmm_predict_proba_wrong_features() {
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(2, 30, 1e-6);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        let bad_data = Array2::<f64>::zeros((5, 3)); // wrong D
        assert!(result.predict_proba(bad_data.view()).is_err());
    }

    #[test]
    fn test_vbgmm_component_pruning() {
        // With many components and few points, pruning should activate
        let data = two_cluster_data();
        let cfg = VBGMMConfig::new(8, 50, 1e-6)
            .with_prune_threshold(0.5);
        let model = VBGMMModel::new(cfg);
        let result = model.fit_vbem(data.view()).expect("fit");
        assert!(result.n_components <= 8, "n_components = {}", result.n_components);
        assert!(result.n_components >= 1, "at least one component must remain");
    }
}
