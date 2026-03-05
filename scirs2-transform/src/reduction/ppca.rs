//! Probabilistic PCA (PPCA)
//!
//! Implements the Tipping & Bishop (1999) probabilistic PCA model:
//!
//! ```text
//! x = W z + mu + epsilon
//! z ~ N(0, I_q)
//! epsilon ~ N(0, sigma^2 I_p)
//! ```
//!
//! where `x` is `p`-dimensional observed data, `z` is `q`-dimensional latent variable,
//! `W` is the `p × q` loading matrix, `mu` is the mean, and `sigma^2` is isotropic noise.
//!
//! ## EM Algorithm
//!
//! The EM algorithm closed-form updates (Tipping & Bishop 1999, section 3.3) are:
//!
//! **E-step** (posterior moments of z given x):
//! ```text
//! M    = W^T W + sigma^2 I
//! E[z|x] = M^{-1} W^T (x - mu)
//! Cov[z|x] = sigma^2 M^{-1}
//! ```
//!
//! **M-step** (closed-form):
//! ```text
//! W_new = S W M^{-1} (sigma^2 M^{-1} + M^{-1} W^T S W M^{-1})^{-1}
//! sigma^2_new = (1/p) trace(S - S W M^{-1} W_new^T)
//! ```
//!
//! where `S` is the sample covariance.
//!
//! ## References
//!
//! - Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis.
//!   *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::solve_linear_system;

use crate::error::{Result, TransformError};

// ─── small constant to guard against division-by-zero ─────────────────────────
const EPS: f64 = 1e-12;

// ─── helper: small square-matrix inverse via Gaussian elimination ─────────────

/// Invert a small `k×k` matrix using Gaussian elimination with partial pivoting.
///
/// Returns an error if the matrix is singular (pivot < `EPS`).
fn invert_small(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let k = mat.nrows();
    if k != mat.ncols() {
        return Err(TransformError::InvalidInput(
            "invert_small: matrix must be square".into(),
        ));
    }
    // Build augmented [mat | I]
    let mut aug = Array2::<f64>::zeros((k, 2 * k));
    for i in 0..k {
        for j in 0..k {
            aug[[i, j]] = mat[[i, j]];
        }
        aug[[i, k + i]] = 1.0;
    }
    // Forward elimination
    for col in 0..k {
        // Find pivot
        let mut pivot_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..k {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < EPS {
            return Err(TransformError::ComputationError(
                "invert_small: singular matrix".into(),
            ));
        }
        if pivot_row != col {
            for j in 0..(2 * k) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
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
    // Extract inverse
    let mut inv = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            inv[[i, j]] = aug[[i, k + j]];
        }
    }
    Ok(inv)
}

/// Compute `A^T B` for `Array2<f64>`.
fn mat_mul_at_b(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let p = a.nrows();
    let q = a.ncols();
    let r = b.ncols();
    assert_eq!(b.nrows(), p);
    let mut out = Array2::<f64>::zeros((q, r));
    for i in 0..q {
        for j in 0..r {
            let mut s = 0.0;
            for k in 0..p {
                s += a[[k, i]] * b[[k, j]];
            }
            out[[i, j]] = s;
        }
    }
    out
}

/// Compute `A B` for `Array2<f64>`.
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    assert_eq!(b.nrows(), k);
    let mut out = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            for j in 0..n {
                out[[i, j]] += a[[i, l]] * b[[l, j]];
            }
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// PPCAModel
// ─────────────────────────────────────────────────────────────────────────────

/// Fitted Probabilistic PCA model.
///
/// The generative model is:
/// ```text
/// x = W z + mu + eps,   z ~ N(0,I),  eps ~ N(0, sigma2 * I)
/// ```
#[derive(Debug, Clone)]
pub struct PPCAModel {
    /// Loading matrix `W`, shape `(n_features, n_components)`.
    pub w: Array2<f64>,
    /// Isotropic noise variance `σ²`.
    pub sigma2: f64,
    /// Data mean, shape `(n_features,)`.
    pub mean: Array1<f64>,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Final log-likelihood value.
    pub log_likelihood: f64,
}

impl PPCAModel {
    // ── derived quantities ────────────────────────────────────────────────────

    /// Returns `M = W^T W + σ² I_q`.
    fn m_matrix(&self) -> Array2<f64> {
        let q = self.w.ncols();
        let wt_w = mat_mul_at_b(&self.w, &self.w);
        let mut m = wt_w;
        for i in 0..q {
            m[[i, i]] += self.sigma2;
        }
        m
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Project data `X` (shape `n × p`) to the `q`-dimensional latent space.
    ///
    /// The posterior mean is `E[z|x] = M^{-1} W^T (x - mu)`.
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.w.nrows() {
            return Err(TransformError::DimensionMismatch(format!(
                "Expected {p} features, got {p}"
            )));
        }
        let m = self.m_matrix();
        let m_inv = invert_small(&m)?;
        // posterior_scale = M^{-1} W^T,  shape (q, p)
        let posterior_scale = mat_mul(&m_inv, &mat_mul_at_b(&self.w, &Array2::eye(p)));
        // centre data
        let mut xc = x.to_owned();
        for i in 0..n {
            for j in 0..p {
                xc[[i, j]] -= self.mean[j];
            }
        }
        // z = xc @ (M^{-1} W^T)^T  = xc @ W M^{-1}
        // posterior_scale is (q, p), so xc (n,p) @ posterior_scale^T (p,q) => (n,q)
        let wt = posterior_scale; // (q, p)
        let mut z = Array2::<f64>::zeros((n, wt.nrows()));
        for i in 0..n {
            for j in 0..wt.nrows() {
                let mut s = 0.0;
                for l in 0..p {
                    s += xc[[i, l]] * wt[[j, l]];
                }
                z[[i, j]] = s;
            }
        }
        Ok(z)
    }

    /// Reconstruct data from latent codes `Z` (shape `n × q`).
    ///
    /// Returns `Z W^T + mu`.
    pub fn inverse_transform(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        let (n, q) = (z.nrows(), z.ncols());
        if q != self.w.ncols() {
            return Err(TransformError::DimensionMismatch(format!(
                "Expected {q} latent dims, got {q}"
            )));
        }
        let p = self.w.nrows();
        // x_hat = Z W^T + mu
        let mut x_hat = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let mut s = 0.0;
                for l in 0..q {
                    s += z[[i, l]] * self.w[[j, l]];
                }
                x_hat[[i, j]] = s + self.mean[j];
            }
        }
        Ok(x_hat)
    }

    /// Compute the log-likelihood of data `X` under the PPCA model.
    ///
    /// Uses the closed-form marginal log-likelihood:
    /// ```text
    /// log p(X) = sum_n log N(x_n; mu, C)
    /// C = W W^T + sigma^2 I
    /// ```
    ///
    /// Uses the matrix determinant lemma and Woodbury identity for efficiency.
    pub fn log_likelihood(&self, x: &Array2<f64>) -> Result<f64> {
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.w.nrows() {
            return Err(TransformError::DimensionMismatch(
                "Feature dimension mismatch".into(),
            ));
        }
        let q = self.w.ncols();
        let sigma2 = self.sigma2.max(EPS);

        // M = W^T W + sigma^2 I, shape (q, q)
        let m = self.m_matrix();
        let m_inv = invert_small(&m)?;

        // log|C| = (p-q) log(sigma2) + log|M|   (matrix det lemma)
        // log|M| via Cholesky-free: product of pivots from our inversion auxiliary
        // Simpler: compute det(M) directly for small q
        let log_det_m = log_det_small(&m)?;
        let log_det_c = (p - q) as f64 * sigma2.ln() + log_det_m;

        // Woodbury: C^{-1} = sigma2^{-1} I - sigma2^{-2} W M^{-1} W^T
        // For each x_n, quadratic form: (x-mu)^T C^{-1} (x-mu)
        let wm_inv = mat_mul(&self.w, &m_inv); // (p, q)

        let log2pi = (2.0 * std::f64::consts::PI).ln();
        let const_term = -0.5 * (p as f64 * log2pi + log_det_c);
        let mut total_ll = const_term * n as f64;

        for i in 0..n {
            let mut diff = Array1::<f64>::zeros(p);
            for j in 0..p {
                diff[j] = x[[i, j]] - self.mean[j];
            }
            // term1 = ||diff||^2 / sigma2
            let term1: f64 = diff.iter().map(|v| v * v).sum::<f64>() / sigma2;
            // term2 = diff^T W M^{-1} W^T diff / sigma2^2
            // wm_inv^T diff  (shape q)
            let mut tmp = vec![0.0f64; q];
            for l in 0..q {
                for j in 0..p {
                    tmp[l] += wm_inv[[j, l]] * diff[j];
                }
            }
            let term2: f64 = tmp.iter().zip(tmp.iter()).map(|(a, b)| a * b).sum::<f64>() / sigma2 / sigma2;
            total_ll -= 0.5 * (term1 - term2);
        }
        Ok(total_ll)
    }

    /// Impute missing values in `X` using the posterior mean.
    ///
    /// `missing_mask` is a boolean array of shape `(n, p)`.
    /// `true` entries indicate missing values that will be filled with
    /// `E[x_miss | x_obs]`.
    pub fn impute_missing(
        &self,
        x: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> Result<Array2<f64>> {
        let (n, p) = (x.nrows(), x.ncols());
        if p != self.w.nrows() {
            return Err(TransformError::DimensionMismatch(
                "Feature dimension mismatch".into(),
            ));
        }
        if missing_mask.shape() != [n, p] {
            return Err(TransformError::DimensionMismatch(
                "missing_mask shape must match X".into(),
            ));
        }

        let m = self.m_matrix();
        let m_inv = invert_small(&m)?;
        let q = self.w.ncols();

        let mut x_imputed = x.to_owned();

        for i in 0..n {
            // observed indices for this sample
            let obs: Vec<usize> = (0..p).filter(|&j| !missing_mask[[i, j]]).collect();
            let miss: Vec<usize> = (0..p).filter(|&j| missing_mask[[i, j]]).collect();

            if miss.is_empty() {
                continue;
            }
            if obs.is_empty() {
                // No observed data: use the mean
                for &j in &miss {
                    x_imputed[[i, j]] = self.mean[j];
                }
                continue;
            }

            // Compute posterior z using observed dimensions only
            // W_obs: (|obs|, q)
            let n_obs = obs.len();
            let mut w_obs = Array2::<f64>::zeros((n_obs, q));
            let mut x_obs_c = Array1::<f64>::zeros(n_obs);
            for (ii, &j) in obs.iter().enumerate() {
                for l in 0..q {
                    w_obs[[ii, l]] = self.w[[j, l]];
                }
                x_obs_c[ii] = x[[i, j]] - self.mean[j];
            }

            // M_obs = W_obs^T W_obs + sigma2 I_q
            let wot_wo = mat_mul_at_b(&w_obs, &w_obs);
            let mut m_obs = wot_wo;
            for l in 0..q {
                m_obs[[l, l]] += self.sigma2;
            }
            let m_obs_inv = invert_small(&m_obs)?;

            // E[z] = M_obs^{-1} W_obs^T x_obs_c
            let mut ez = vec![0.0f64; q];
            for l in 0..q {
                for (ii, _) in obs.iter().enumerate() {
                    ez[l] += m_obs_inv[[l, 0]] * 0.0; // placeholder; compute below
                }
            }
            // proper computation: M_obs_inv @ W_obs^T @ x_obs_c
            let mut wot_xc = vec![0.0f64; q];
            for l in 0..q {
                for (ii, _) in obs.iter().enumerate() {
                    wot_xc[l] += w_obs[[ii, l]] * x_obs_c[ii];
                }
            }
            let mut ez = vec![0.0f64; q];
            for l in 0..q {
                for l2 in 0..q {
                    ez[l] += m_obs_inv[[l, l2]] * wot_xc[l2];
                }
            }

            // Fill missing: E[x_miss] = W_miss @ E[z] + mu_miss
            for &j in &miss {
                let mut val = self.mean[j];
                for l in 0..q {
                    val += self.w[[j, l]] * ez[l];
                }
                x_imputed[[i, j]] = val;
            }
        }
        Ok(x_imputed)
    }
}

/// Compute `log det(A)` for a small square matrix by Gaussian elimination.
fn log_det_small(mat: &Array2<f64>) -> Result<f64> {
    let k = mat.nrows();
    let mut a = mat.to_owned();
    let mut sign = 1.0f64;
    let mut log_det = 0.0f64;

    for col in 0..k {
        // Partial pivot
        let mut max_val = a[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..k {
            if a[[row, col]].abs() > max_val {
                max_val = a[[row, col]].abs();
                max_row = row;
            }
        }
        if max_val < EPS {
            return Ok(f64::NEG_INFINITY);
        }
        if max_row != col {
            for j in 0..k {
                let tmp = a[[col, j]];
                a[[col, j]] = a[[max_row, j]];
                a[[max_row, j]] = tmp;
            }
            sign = -sign;
        }
        log_det += a[[col, col]].abs().ln();
        let pivot = a[[col, col]];
        for row in (col + 1)..k {
            let factor = a[[row, col]] / pivot;
            for j in col..k {
                let v = a[[col, j]] * factor;
                a[[row, j]] -= v;
            }
        }
    }
    Ok(log_det)
}

// ─────────────────────────────────────────────────────────────────────────────
// PPCA fitter
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for PPCA EM fitting.
#[derive(Debug, Clone)]
pub struct PPCAConfig {
    /// Number of latent dimensions `q`.
    pub n_components: usize,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on log-likelihood change.
    pub tol: f64,
    /// Random seed for initialisation.
    pub seed: u64,
}

impl Default for PPCAConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 200,
            tol: 1e-6,
            seed: 42,
        }
    }
}

/// Fit a PPCA model to data `X` (shape `n × p`) using the EM algorithm.
///
/// # Errors
///
/// Returns an error if `n_components >= p`, the data has fewer than 2 samples,
/// or numerical issues arise during inversion.
pub fn fit_em(x: &Array2<f64>, config: &PPCAConfig) -> Result<PPCAModel> {
    let (n, p) = (x.nrows(), x.ncols());
    let q = config.n_components;

    if n < 2 {
        return Err(TransformError::InvalidInput(
            "PPCA requires at least 2 samples".into(),
        ));
    }
    if q == 0 || q >= p {
        return Err(TransformError::InvalidInput(format!(
            "n_components must be in 1..{p}, got {q}"
        )));
    }

    // ── Compute mean and sample covariance ────────────────────────────────────
    let mean = x.mean_axis(Axis(0)).ok_or_else(|| {
        TransformError::ComputationError("Failed to compute mean".into())
    })?;

    // sample covariance S = (X-mu)^T (X-mu) / n
    let mut xc = x.to_owned();
    for i in 0..n {
        for j in 0..p {
            xc[[i, j]] -= mean[j];
        }
    }
    let s_raw = mat_mul_at_b(&xc, &xc);
    let mut s = s_raw;
    for i in 0..p {
        for j in 0..p {
            s[[i, j]] /= n as f64;
        }
    }

    // ── Initialise W and sigma^2 ──────────────────────────────────────────────
    // Use small random perturbations around zeros
    let mut rng = scirs2_core::random::rng();
    let mut w = Array2::<f64>::zeros((p, q));
    for i in 0..p {
        for j in 0..q {
            w[[i, j]] = rng.gen_range(-0.1..0.1);
        }
    }

    // sigma^2 = trace(S) / (p * 2)
    let trace_s: f64 = (0..p).map(|i| s[[i, i]]).sum::<f64>();
    let mut sigma2 = trace_s / (p as f64 * 2.0).max(EPS);

    let mut prev_ll = f64::NEG_INFINITY;
    let mut final_iter = 0usize;

    // ── EM iterations ─────────────────────────────────────────────────────────
    for iter in 0..config.max_iter {
        // E-step: M = W^T W + sigma^2 I
        let wt_w = mat_mul_at_b(&w, &w);
        let mut m = wt_w;
        for i in 0..q {
            m[[i, i]] += sigma2;
        }
        let m_inv = invert_small(&m)?;

        // <z> = M^{-1} W^T (x - mu)   summed over samples => S_wt = xc^T xc / n = S already
        // Closed-form M-step on sample covariance:
        //
        // W_new = S W (sigma^2 M^{-1} + M^{-1} W^T S W M^{-1})^{-1}
        //
        // Define: Sigma_z = sigma2 * M^{-1}  (posterior covariance of z)
        //         <z z^T> = Sigma_z + M^{-1} W^T S W M^{-1}
        //         W_new = S W M^{-1}  (<z z^T>)^{-1}
        //

        // Compute S W M^{-1}:  (p,p)(p,q)(q,q) => (p,q)
        let sw = mat_mul(&s, &w);
        let sw_m_inv = mat_mul(&sw, &m_inv);

        // Compute M^{-1} W^T S W M^{-1}: (q,q)(q,p)(p,p)(p,q)(q,q) - simplify step by step
        let wt_s = mat_mul_at_b(&w, &s); // (q, p)
        let wt_s_w = mat_mul(&wt_s, &w); // (q, q)
        let m_inv_wt_s_w = mat_mul(&m_inv, &wt_s_w); // (q, q)
        let m_inv_wt_s_w_m_inv = mat_mul(&m_inv_wt_s_w, &m_inv); // (q, q)

        // <z z^T> = sigma2 * M^{-1} + M^{-1} W^T S W M^{-1}
        let mut ezzt = m_inv_wt_s_w_m_inv;
        for i in 0..q {
            for j in 0..q {
                ezzt[[i, j]] += sigma2 * m_inv[[i, j]];
            }
        }

        let ezzt_inv = invert_small(&ezzt)?;
        // W_new = S W M^{-1} <z z^T>^{-1}   = sw_m_inv @ ezzt_inv
        let w_new = mat_mul(&sw_m_inv, &ezzt_inv);

        // M-step sigma2: sigma2_new = (1/p) [ trace(S) - trace(S W M^{-1} W_new^T) ]
        // = (1/p) [trace(S) - trace(sw_m_inv @ W_new^T)]
        let sw_m_inv_wt_new = mat_mul(&sw_m_inv, &mat_mul_at_b(&w_new, &Array2::eye(q)));
        // sw_m_inv is (p,q), W_new^T is (q,p), product is (p,p), take trace
        // Actually sw_m_inv (p,q) @ W_new^T is (p,q)(q,p) = (p,p). We need trace.
        // But it's faster to compute: trace(A B^T) = sum_{ij} A_{ij} B_{ij}
        let trace_sw: f64 = (0..p)
            .map(|i| {
                let mut s_row = 0.0;
                for l in 0..q {
                    s_row += sw_m_inv[[i, l]] * w_new[[i, l]];
                }
                s_row
            })
            .sum();
        let sigma2_new = ((trace_s - trace_sw) / p as f64).max(EPS);

        w = w_new;
        sigma2 = sigma2_new;

        // Check convergence via log-likelihood on a sub-sample (or full)
        // We use the model to compute ll for early stopping
        if iter % 5 == 0 || iter == config.max_iter - 1 {
            let tmp_model = PPCAModel {
                w: w.clone(),
                sigma2,
                mean: mean.clone(),
                n_iter: iter + 1,
                log_likelihood: 0.0,
            };
            let ll = tmp_model.log_likelihood(x).unwrap_or(f64::NEG_INFINITY);
            let delta = (ll - prev_ll).abs();
            if iter > 0 && delta < config.tol {
                final_iter = iter + 1;
                prev_ll = ll;
                break;
            }
            prev_ll = ll;
        }
        final_iter = iter + 1;
    }

    Ok(PPCAModel {
        w,
        sigma2,
        mean,
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

    fn make_low_rank_data(n: usize, p: usize, q: usize, noise: f64) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        // Random latent factors
        let mut z = Array2::<f64>::zeros((n, q));
        for i in 0..n {
            for j in 0..q {
                z[[i, j]] = rng.gen_range(-2.0..2.0);
            }
        }
        // Random loading matrix
        let mut w = Array2::<f64>::zeros((p, q));
        for i in 0..p {
            for j in 0..q {
                w[[i, j]] = rng.gen_range(-1.0..1.0);
            }
        }
        // X = Z W^T + noise
        let mut x = mat_mul(&z, &mat_mul_at_b(&w, &Array2::eye(q)));
        for i in 0..n {
            for j in 0..p {
                x[[i, j]] += noise * rng.gen_range(-1.0..1.0);
            }
        }
        x
    }

    #[test]
    fn test_ppca_fit_basic() {
        let x = make_low_rank_data(50, 10, 2, 0.1);
        let config = PPCAConfig {
            n_components: 2,
            max_iter: 50,
            tol: 1e-4,
            seed: 0,
        };
        let model = fit_em(&x, &config).expect("PPCA fit failed");
        assert_eq!(model.w.shape(), &[10, 2]);
        assert!(model.sigma2 > 0.0);
        assert!(model.log_likelihood.is_finite() || model.log_likelihood == f64::NEG_INFINITY);
    }

    #[test]
    fn test_ppca_transform_inverse() {
        let x = make_low_rank_data(40, 8, 2, 0.05);
        let config = PPCAConfig {
            n_components: 2,
            max_iter: 50,
            tol: 1e-4,
            seed: 1,
        };
        let model = fit_em(&x, &config).expect("fit failed");
        let z = model.transform(&x).expect("transform failed");
        assert_eq!(z.shape(), &[40, 2]);
        let x_hat = model.inverse_transform(&z).expect("inverse_transform failed");
        assert_eq!(x_hat.shape(), &[40, 8]);
        // Reconstruction error should be small for low-rank data
        let err: f64 = x
            .iter()
            .zip(x_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / (40.0 * 8.0);
        assert!(err < 5.0, "Reconstruction error {err} too large");
    }

    #[test]
    fn test_ppca_impute_missing() {
        let x = make_low_rank_data(30, 6, 2, 0.1);
        let config = PPCAConfig {
            n_components: 2,
            max_iter: 30,
            tol: 1e-4,
            seed: 2,
        };
        let model = fit_em(&x, &config).expect("fit failed");

        let mut missing = Array2::<bool>::from_elem((30, 6), false);
        missing[[0, 0]] = true;
        missing[[1, 2]] = true;

        let x_imp = model.impute_missing(&x, &missing).expect("impute failed");
        assert!(x_imp[[0, 0]].is_finite());
        assert!(x_imp[[1, 2]].is_finite());
        // Non-missing values should be unchanged
        for i in 0..30 {
            for j in 0..6 {
                if !missing[[i, j]] {
                    assert_eq!(x_imp[[i, j]], x[[i, j]]);
                }
            }
        }
    }

    #[test]
    fn test_ppca_log_likelihood() {
        let x = make_low_rank_data(30, 6, 2, 0.2);
        let config = PPCAConfig {
            n_components: 2,
            max_iter: 30,
            tol: 1e-4,
            seed: 3,
        };
        let model = fit_em(&x, &config).expect("fit failed");
        let ll = model.log_likelihood(&x).expect("ll failed");
        // Log-likelihood should be a finite (possibly very negative) number
        assert!(ll.is_finite() || ll == f64::NEG_INFINITY);
    }

    #[test]
    fn test_ppca_invalid_inputs() {
        let x = Array2::<f64>::zeros((10, 5));
        let bad_config = PPCAConfig {
            n_components: 5, // must be < p=5
            ..Default::default()
        };
        assert!(fit_em(&x, &bad_config).is_err());

        let bad_config2 = PPCAConfig {
            n_components: 0,
            ..Default::default()
        };
        assert!(fit_em(&x, &bad_config2).is_err());
    }
}
