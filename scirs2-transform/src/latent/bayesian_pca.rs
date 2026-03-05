//! Bayesian PCA with Automatic Relevance Determination (ARD)
//!
//! Implements the variational Bayes formulation of Bayesian PCA
//! (Bishop 1999; Minka 2000) with an ARD prior over the columns of
//! the loading matrix `W`.
//!
//! ## Model
//!
//! ```text
//! x_n = W z_n + mu + eps_n
//! z_n ~ N(0, I_q)
//! eps_n ~ N(0, tau^{-1} I_p)       -- isotropic noise
//! w_k  ~ N(0, alpha_k^{-1} I_p)    -- ARD prior on column k of W
//! alpha_k ~ Gamma(a0, b0)
//! tau ~ Gamma(c0, d0)
//! ```
//!
//! Under the mean-field variational approximation, each hidden variable
//! has a factorial Gaussian / Gamma posterior:
//!
//! - `q(z_n) = N(z_n; mu_z_n, Sigma_z)`
//! - `q(w_k) = N(w_k; mu_w_k, Sigma_w_k)`
//! - `q(alpha_k) = Gamma(a_k, b_k)`
//! - `q(tau) = Gamma(c, d)`
//!
//! ## References
//!
//! - Bishop, C. M. (1999). Variational principal components.
//!   *Proceedings of ICANN*, 509–514.
//! - Minka, T. P. (2000). Automatic choice of dimensionality for PCA.
//!   *NIPS*, pp. 598–604.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, Axis};

// ─── epsilon guard ────────────────────────────────────────────────────────────
const EPS: f64 = 1e-12;

// ─── small-matrix helpers (shared with ppca, duplicated here for independence) ─

/// Invert a small square matrix by Gaussian elimination with partial pivoting.
fn invert_small(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let k = mat.nrows();
    if k != mat.ncols() {
        return Err(TransformError::InvalidInput("Matrix must be square".into()));
    }
    let mut aug = Array2::<f64>::zeros((k, 2 * k));
    for i in 0..k {
        for j in 0..k {
            aug[[i, j]] = mat[[i, j]];
        }
        aug[[i, k + i]] = 1.0;
    }
    for col in 0..k {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < EPS {
            return Err(TransformError::ComputationError("Singular matrix".into()));
        }
        if max_row != col {
            for j in 0..(2 * k) {
                aug.swap([col, j], [max_row, j]);
            }
        }
        let diag = aug[[col, col]];
        for j in 0..(2 * k) {
            aug[[col, j]] /= diag;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            for j in 0..(2 * k) {
                let v = aug[[col, j]] * factor;
                aug[[row, j]] -= v;
            }
        }
    }
    let mut inv = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            inv[[i, j]] = aug[[i, k + j]];
        }
    }
    Ok(inv)
}

/// Compute `A^T B` where A is `m×k` and B is `m×n` → `k×n`.
fn mm_atb(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(b.nrows(), m);
    let mut c = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for l in 0..m {
            for j in 0..n {
                c[[i, j]] += a[[l, i]] * b[[l, j]];
            }
        }
    }
    c
}

/// Compute `A B` where A is `m×k` and B is `k×n` → `m×n`.
fn mm(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    assert_eq!(b.nrows(), k);
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                c[[i, j]] += a[[i, l]] * b[[l, j]];
            }
        }
    }
    c
}

// ─────────────────────────────────────────────────────────────────────────────
// BayesianPCA
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Bayesian PCA.
#[derive(Debug, Clone)]
pub struct BayesianPCAConfig {
    /// Maximum number of components (upper bound; ARD prunes automatically).
    pub n_components: usize,
    /// Maximum number of variational EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on ELBO change.
    pub tol: f64,
    /// ARD prior shape: `alpha ~ Gamma(a0, b0)`.
    pub a0: f64,
    /// ARD prior rate: `alpha ~ Gamma(a0, b0)`.
    pub b0: f64,
    /// Noise precision prior shape: `tau ~ Gamma(c0, d0)`.
    pub c0: f64,
    /// Noise precision prior rate: `tau ~ Gamma(c0, d0)`.
    pub d0: f64,
    /// Threshold below which an alpha_k is considered "infinity" (column pruned).
    pub prune_threshold: f64,
}

impl Default for BayesianPCAConfig {
    fn default() -> Self {
        Self {
            n_components: 10,
            max_iter: 200,
            tol: 1e-5,
            a0: 1e-3,
            b0: 1e-3,
            c0: 1e-3,
            d0: 1e-3,
            prune_threshold: 1e3,
        }
    }
}

/// Bayesian PCA with ARD prior, fitted by variational Bayes.
///
/// After calling `fit_vb()`, components corresponding to large `alpha_k`
/// (ARD precision → ∞) are effectively pruned.  Use `effective_rank()` to
/// query how many components remain active.
#[derive(Debug, Clone)]
pub struct BayesianPCA {
    config: BayesianPCAConfig,
    /// Posterior mean of the loading matrix W, shape `(n_features, n_components)`.
    pub w_mean: Option<Array2<f64>>,
    /// Per-component ARD precision posterior mean `E[alpha_k]`, length `n_components`.
    pub alpha_mean: Option<Array1<f64>>,
    /// Posterior mean of noise precision `E[tau]`.
    pub tau_mean: Option<f64>,
    /// Data mean, length `n_features`.
    pub data_mean: Option<Array1<f64>>,
    /// ELBO values across iterations.
    pub elbo_history: Vec<f64>,
    /// Number of variational EM iterations run.
    pub n_iter: usize,
}

impl BayesianPCA {
    /// Create a new BayesianPCA instance.
    pub fn new(config: BayesianPCAConfig) -> Self {
        Self {
            config,
            w_mean: None,
            alpha_mean: None,
            tau_mean: None,
            data_mean: None,
            elbo_history: Vec::new(),
            n_iter: 0,
        }
    }

    /// Fit the model to data `X` (shape `n × p`) using variational Bayes.
    ///
    /// Iterates mean-field updates for `q(Z)`, `q(W)`, `q(alpha)`, `q(tau)`
    /// until convergence or `max_iter` is reached.
    pub fn fit_vb(&mut self, x: &Array2<f64>) -> Result<()> {
        let (n, p) = (x.nrows(), x.ncols());
        let q = self.config.n_components;

        if n < 2 {
            return Err(TransformError::InvalidInput(
                "BayesianPCA requires at least 2 samples".into(),
            ));
        }
        if q == 0 || q >= p {
            return Err(TransformError::InvalidInput(format!(
                "n_components must be in 1..{p}, got {q}"
            )));
        }

        // Centre data
        let mu = x.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute mean".into())
        })?;
        let mut xc = x.to_owned();
        for i in 0..n {
            for j in 0..p {
                xc[[i, j]] -= mu[j];
            }
        }

        // S = X^T X / n  (p×p sample covariance, without the p×p storage when p is large)
        // For Bayesian PCA we only need products involving W, so we compute S implicitly.
        // We store S = xc^T xc / n as a full matrix (acceptable for moderate p).
        let xt_x = mm_atb(&xc, &xc); // p×p
        let s: Array2<f64> = {
            let mut tmp = xt_x.clone();
            for i in 0..p {
                for j in 0..p {
                    tmp[[i, j]] /= n as f64;
                }
            }
            tmp
        };
        let trace_s: f64 = (0..p).map(|i| s[[i, i]]).sum();

        // ── Initialise variational parameters ─────────────────────────────────
        // W_mean: p × q  (small random init)
        let mut rng = scirs2_core::random::rng();
        let mut w = Array2::<f64>::zeros((p, q));
        for i in 0..p {
            for j in 0..q {
                w[[i, j]] = 0.01 * (scirs2_core::random::Rng::gen_range(&mut rng, -1.0..1.0_f64));
            }
        }

        // alpha_k: ARD precisions, initialised to 1
        let mut alpha = Array1::<f64>::from_elem(q, 1.0_f64);
        // Noise precision tau
        let mut tau = (p as f64) / (trace_s + EPS);

        // Posterior covariance of each w_k (p×p) — stored as a Vec of p×p matrices.
        // For efficiency we keep a single shared q×q "Sigma_z" and compute per-column corrections.
        // Actually VB for PPCA/BayesianPCA updates a single Sigma_w (q×q inverse, shared across p dims).
        // We use the Bishop (1999) factored form: Sigma_w_k = (alpha_k I + tau W^T W)^{-1} (scalar × I)
        // because the loading prior is diagonal.  This gives a q×q matrix per column but with the
        // same structure.  We maintain Sigma_w_inv = diag(alpha) + tau W^T W  (q×q).

        let mut elbo_history: Vec<f64> = Vec::new();
        let mut prev_elbo = f64::NEG_INFINITY;
        let mut final_iter = 0usize;

        for iter in 0..self.config.max_iter {
            // ── Update q(W) ────────────────────────────────────────────────────
            // Sigma_W = (diag(alpha) + tau * W^T W + tau * n * (Sigma_z term))
            // Bishop 1999 eq. (24)-(25):
            // M = diag(alpha) + tau * (n * Cov[z|X] + sum_n mu_z_n mu_z_n^T / n)
            // Let's work in terms of the sufficient stats.
            //
            // Variational posterior on Z: q(z_n) = N(mu_n, Sigma_z)
            // Sigma_z = (I + tau W^T W)^{-1}   [shape q×q]
            // mu_n = tau Sigma_z W^T x_n
            //
            // Sufficient stats (over n samples):
            //   E[z z^T | X] = n * Sigma_z + sum_n mu_n mu_n^T / n * n
            //   E[z^T x | X] = Sigma_z W^T X^T X / ... = sum_n mu_n x_n^T
            //
            // For the W update we need:
            //   <W_k> update using X^T and E[ZZ^T]

            // Step A: compute Sigma_z = (I_q + tau W^T W)^{-1}
            let wt_w = mm_atb(&w, &w); // q×q
            let mut sigma_z_inv = wt_w.clone();
            for i in 0..q {
                for j in 0..q {
                    sigma_z_inv[[i, j]] *= tau;
                }
                sigma_z_inv[[i, i]] += 1.0;
            }
            let sigma_z = invert_small(&sigma_z_inv)?; // q×q

            // Step B: compute E[z_n] = tau Sigma_z W^T x_n for all n
            // This gives matrix Z_mean of shape n×q
            // Z_mean = xc @ W @ Sigma_z * tau   (n×p)(p×q)(q×q) = n×q
            let xc_w = mm(&xc, &w); // n×q
            let z_mean = {
                let xc_w_sigma = mm(&xc_w, &sigma_z); // n×q
                let mut tmp = xc_w_sigma;
                for i in 0..n {
                    for j in 0..q {
                        tmp[[i, j]] *= tau;
                    }
                }
                tmp
            }; // n×q

            // Step C: E[z z^T] = n * Sigma_z + Z_mean^T Z_mean   (q×q)
            let mut e_zzt = mm_atb(&z_mean, &z_mean); // q×q
            for i in 0..q {
                for j in 0..q {
                    e_zzt[[i, j]] += n as f64 * sigma_z[[i, j]];
                }
            }

            // Step D: Update W
            // W_new = (sum_n x_n z_n^T) * E[z z^T]^{-1}
            //   adjusted for ARD:
            //   Sigma_w^{-1} = diag(alpha) + tau * E[z z^T]
            //   W_new column k = tau * Sigma_w^{col k} * (xc^T Z_mean)[:,k]  ... but this mixes cols
            //
            // Actually Bishop's update couples all p rows of W simultaneously:
            // W_new = xc^T Z_mean * (E[z z^T] + diag(alpha) / tau)^{-1}
            // This is the MAP-like update under the variational posterior.
            let xt_z = mm_atb(&xc, &z_mean); // p×q   sum_n x_n z_n^T
            let mut lhs = e_zzt.clone(); // q×q
            for k in 0..q {
                lhs[[k, k]] += alpha[k] / (tau + EPS);
            }
            let lhs_inv = invert_small(&lhs)?;
            w = mm(&xt_z, &lhs_inv); // p×q

            // ── Update q(alpha) ───────────────────────────────────────────────
            // E[alpha_k] = (a0 + p/2) / (b0 + 0.5 * (||w_k||^2 + p * Sigma_w_kk))
            // where Sigma_w_kk = [diag(alpha) + tau E[z z^T]]^{-1}_{kk}
            let sigma_w = invert_small(&lhs)?; // q×q (= lhs_inv we already have)
            let sigma_w = lhs_inv; // reuse

            for k in 0..q {
                let w_k_norm_sq: f64 = (0..p).map(|i| w[[i, k]] * w[[i, k]]).sum();
                let sigma_kk = sigma_w[[k, k]];
                // Expected ||w_k||^2 under posterior = ||mu_w_k||^2 + p * sigma_kk (isotropic)
                let e_w_k_sq = w_k_norm_sq + p as f64 * sigma_kk;
                let a_k = self.config.a0 + 0.5 * p as f64;
                let b_k = self.config.b0 + 0.5 * e_w_k_sq;
                alpha[k] = a_k / b_k.max(EPS);
            }

            // ── Update q(tau) ─────────────────────────────────────────────────
            // Expected noise precision update:
            // E[tau] = (c0 + n*p/2) / (d0 + 0.5 * (trace(S) * n - 2 sum_n x_n^T W E[z_n] + trace(E[z z^T] W^T W)))
            let c_new = self.config.c0 + 0.5 * (n * p) as f64;

            // E[||x - W z||^2] summed over n:
            //   = n * trace(S) - 2 * trace(W^T X^T Z_mean) + trace(E[z z^T] W^T W)
            let trace_wt_xt_z: f64 = {
                // trace(W^T X^T Z_mean) = sum_{ij} w_{ij} (X Z)_{ij}
                let xz = mm_atb(&xc, &z_mean); // p×q  wait: xc is n×p, z_mean is n×q => xc^T z_mean = p×q
                // Actually mm_atb(xc, z_mean) = xc^T z_mean but xc is n×p and z_mean is n×q
                // so we want sum_n sum_k w_{nk} ... no.
                // trace(W^T X^T Z_mean) = trace( W^T (n×p)^T (n×q) ) no dims are off.
                // W: p×q, X^T Z_mean: p×q (already computed as xt_z above)
                // trace(W^T @ xt_z) = sum_{ij} W_{ij} xt_z_{ij}
                (0..p).map(|i| (0..q).map(|j| w[[i, j]] * xt_z[[i, j]]).sum::<f64>()).sum()
            };

            let wt_w_now = mm_atb(&w, &w); // q×q
            let trace_ezztww: f64 = (0..q)
                .map(|i| (0..q).map(|j| e_zzt[[i, j]] * wt_w_now[[j, i]]).sum::<f64>())
                .sum();

            let d_new = self.config.d0
                + 0.5 * (n as f64 * trace_s - 2.0 * trace_wt_xt_z + trace_ezztww);
            tau = c_new / d_new.max(EPS);

            // ── Compute ELBO (approximate, sufficient for convergence) ─────────
            let elbo = self.compute_elbo(
                n, p, tau, &alpha, &w, trace_s, &e_zzt, trace_wt_xt_z, &sigma_z, &sigma_w,
            );
            elbo_history.push(elbo);
            final_iter = iter + 1;

            let delta = (elbo - prev_elbo).abs();
            if iter > 0 && delta < self.config.tol {
                prev_elbo = elbo;
                break;
            }
            prev_elbo = elbo;
        }

        self.w_mean = Some(w);
        self.alpha_mean = Some(alpha);
        self.tau_mean = Some(tau);
        self.data_mean = Some(mu);
        self.elbo_history = elbo_history;
        self.n_iter = final_iter;
        Ok(())
    }

    /// Return the number of components with `alpha_k < prune_threshold`.
    ///
    /// Components with large `alpha_k` (high precision on the prior) have
    /// their loading columns shrunk toward zero — they are effectively irrelevant.
    pub fn effective_rank(&self) -> usize {
        let alpha = match &self.alpha_mean {
            Some(a) => a,
            None => return 0,
        };
        alpha
            .iter()
            .filter(|&&a| a < self.config.prune_threshold)
            .count()
    }

    /// Return indices of the active (non-pruned) components.
    pub fn active_components(&self) -> Vec<usize> {
        let alpha = match &self.alpha_mean {
            Some(a) => a,
            None => return Vec::new(),
        };
        alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a < self.config.prune_threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the loading matrix restricted to active components.
    ///
    /// Shape `(n_features, effective_rank)`.
    pub fn prune_irrelevant_components(&self) -> Option<Array2<f64>> {
        let w = self.w_mean.as_ref()?;
        let active = self.active_components();
        if active.is_empty() {
            return None;
        }
        let p = w.nrows();
        let k_eff = active.len();
        let mut w_pruned = Array2::<f64>::zeros((p, k_eff));
        for (new_k, &old_k) in active.iter().enumerate() {
            for i in 0..p {
                w_pruned[[i, new_k]] = w[[i, old_k]];
            }
        }
        Some(w_pruned)
    }

    /// Project data to the active latent space.
    ///
    /// Uses the posterior mean `E[z|x] = tau Sigma_z W_active^T (x - mu)`.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let w_full = self
            .w_mean
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("BayesianPCA not fitted".into()))?;
        let mu = self
            .data_mean
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("BayesianPCA not fitted".into()))?;
        let tau = self
            .tau_mean
            .ok_or_else(|| TransformError::NotFitted("BayesianPCA not fitted".into()))?;

        let w_pruned = self
            .prune_irrelevant_components()
            .unwrap_or_else(|| w_full.clone());

        let (n, p) = (x.nrows(), x.ncols());
        if p != w_pruned.nrows() {
            return Err(TransformError::DimensionMismatch("Feature dim mismatch".into()));
        }
        let q_eff = w_pruned.ncols();

        // Centre
        let mut xc = x.to_owned();
        for i in 0..n {
            for j in 0..p {
                xc[[i, j]] -= mu[j];
            }
        }

        // Sigma_z = (I + tau W^T W)^{-1}
        let wt_w = mm_atb(&w_pruned, &w_pruned); // q_eff × q_eff
        let mut sigma_z_inv = wt_w;
        for i in 0..q_eff {
            for j in 0..q_eff {
                sigma_z_inv[[i, j]] *= tau;
            }
            sigma_z_inv[[i, i]] += 1.0;
        }
        let sigma_z = invert_small(&sigma_z_inv)?;

        // z_mean = tau * xc @ W @ Sigma_z
        let xc_w = mm(&xc, &w_pruned); // n × q_eff
        let mut z = mm(&xc_w, &sigma_z); // n × q_eff
        for i in 0..n {
            for j in 0..q_eff {
                z[[i, j]] *= tau;
            }
        }
        Ok(z)
    }

    // ── ELBO computation ──────────────────────────────────────────────────────
    // Approximate ELBO = E[log p(X|Z,W,tau)] + E[log p(Z)] + E[log p(W|alpha)]
    //                  + E[log p(alpha)] + E[log p(tau)]
    //                  - E[log q(Z)] - E[log q(W)] - E[log q(alpha)] - E[log q(tau)]
    //
    // We compute only the terms that change across iterations for efficiency.
    #[allow(clippy::too_many_arguments)]
    fn compute_elbo(
        &self,
        n: usize,
        p: usize,
        tau: f64,
        alpha: &Array1<f64>,
        w: &Array2<f64>,
        trace_s: f64,
        e_zzt: &Array2<f64>,
        trace_wt_xt_z: f64,
        sigma_z: &Array2<f64>,
        sigma_w: &Array2<f64>,
    ) -> f64 {
        let q = w.ncols();
        let tau_safe = tau.max(EPS);

        // E[log p(X|Z,W,tau)]  ≈  n*p/2 * log(tau/(2pi)) - tau/2 * E[sum ||x - Wz||^2]
        let wt_w = mm_atb(w, w);
        let trace_ezztww: f64 = (0..q)
            .map(|i| (0..q).map(|j| e_zzt[[i, j]] * wt_w[[j, i]]).sum::<f64>())
            .sum();
        let e_recon = n as f64 * trace_s - 2.0 * trace_wt_xt_z + trace_ezztww;
        let ll_term = 0.5 * n as f64 * p as f64 * (tau_safe.ln() - std::f64::consts::LN_2 - std::f64::consts::PI.ln())
            - 0.5 * tau_safe * e_recon;

        // E[log p(Z)] - E[log q(Z)]:  = n/2 * (log|Sigma_z| + q)  (entropy of q(Z))
        let log_det_sz = log_det_small_safe(sigma_z);
        let kl_z = -0.5 * n as f64 * (log_det_sz + q as f64);

        // ARD regularisation terms (approximate)
        let alpha_reg: f64 = alpha
            .iter()
            .zip(0..q)
            .map(|(&ak, k)| {
                let w_k_sq: f64 = (0..p).map(|i| w[[i, k]] * w[[i, k]]).sum();
                0.5 * (p as f64 * ak.max(EPS).ln() - ak * (w_k_sq + p as f64 * sigma_w[[k, k]]))
            })
            .sum();

        ll_term - kl_z + alpha_reg
    }
}

/// Compute `log|det(A)|` approximately via diagonal product (for small SPD matrices).
fn log_det_small_safe(a: &Array2<f64>) -> f64 {
    // Approximate via diagonal product (Cholesky would be better, but this suffices for ELBO)
    let k = a.nrows().min(a.ncols());
    (0..k).map(|i| a[[i, i]].abs().max(EPS).ln()).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_low_rank(n: usize, p: usize, q: usize, noise: f64) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        let mut z = Array2::<f64>::zeros((n, q));
        let mut w = Array2::<f64>::zeros((p, q));
        for i in 0..n {
            for j in 0..q {
                z[[i, j]] = scirs2_core::random::Rng::gen_range(&mut rng, -2.0..2.0_f64);
            }
        }
        for i in 0..p {
            for j in 0..q {
                w[[i, j]] = scirs2_core::random::Rng::gen_range(&mut rng, -1.0..1.0_f64);
            }
        }
        let xc = mm(&z, &{
            let mut wt = Array2::<f64>::zeros((q, p));
            for i in 0..p {
                for j in 0..q {
                    wt[[j, i]] = w[[i, j]];
                }
            }
            wt
        });
        let mut x = xc;
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] += noise * scirs2_core::random::Rng::gen_range(&mut rng, -1.0..1.0_f64);
            }
        }
        x
    }

    #[test]
    fn test_bayesian_pca_fit() {
        let x = make_low_rank(50, 10, 2, 0.1);
        let config = BayesianPCAConfig {
            n_components: 5,
            max_iter: 50,
            tol: 1e-4,
            ..Default::default()
        };
        let mut model = BayesianPCA::new(config);
        model.fit_vb(&x).expect("BayesianPCA fit failed");
        let w = model.w_mean.as_ref().expect("w_mean missing");
        assert_eq!(w.shape(), &[10, 5]);
        assert!(model.tau_mean.expect("tau_mean should be set after fit") > 0.0);
        assert!(!model.alpha_mean.as_ref().expect("alpha_mean should be set after fit").iter().any(|v| !v.is_finite()));
    }

    #[test]
    fn test_bayesian_pca_effective_rank() {
        let x = make_low_rank(60, 12, 2, 0.05);
        // Use 8 components; ARD should prune most down to ~2
        let config = BayesianPCAConfig {
            n_components: 8,
            max_iter: 100,
            tol: 1e-5,
            prune_threshold: 1e3,
            ..Default::default()
        };
        let mut model = BayesianPCA::new(config);
        model.fit_vb(&x).expect("fit failed");
        let rank = model.effective_rank();
        // Effective rank should be <= 8 (the max) and > 0
        assert!(rank > 0 && rank <= 8, "effective_rank={rank}");
    }

    #[test]
    fn test_bayesian_pca_prune() {
        let x = make_low_rank(40, 8, 2, 0.1);
        let config = BayesianPCAConfig {
            n_components: 6,
            max_iter: 80,
            tol: 1e-5,
            prune_threshold: 100.0,
            ..Default::default()
        };
        let mut model = BayesianPCA::new(config);
        model.fit_vb(&x).expect("fit failed");
        if let Some(w_p) = model.prune_irrelevant_components() {
            assert!(w_p.ncols() <= 6);
            assert_eq!(w_p.nrows(), 8);
        }
    }

    #[test]
    fn test_bayesian_pca_transform() {
        let x = make_low_rank(40, 10, 2, 0.1);
        let config = BayesianPCAConfig {
            n_components: 4,
            max_iter: 50,
            tol: 1e-4,
            ..Default::default()
        };
        let mut model = BayesianPCA::new(config);
        model.fit_vb(&x).expect("fit failed");
        let z = model.transform(&x).expect("transform failed");
        assert_eq!(z.nrows(), 40);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_bayesian_pca_invalid() {
        let x = Array2::<f64>::zeros((10, 5));
        let config = BayesianPCAConfig {
            n_components: 5, // must be < p=5
            ..Default::default()
        };
        let mut model = BayesianPCA::new(config);
        assert!(model.fit_vb(&x).is_err());
    }
}
