//! BEKK-GARCH (Baba-Engle-Kraft-Kroner) multivariate GARCH model
//!
//! The BEKK parameterisation guarantees positive-definite conditional covariance
//! matrices by construction. The general BEKK(1,1,K) model is:
//!
//! ```text
//! H_t = C C^T + A^T ε_{t-1} ε_{t-1}^T A + B^T H_{t-1} B
//! ```
//!
//! where:
//! - C is a lower-triangular matrix (ensures positive definiteness)
//! - A, B are full (or restricted) parameter matrices
//! - ε_t is the k-dimensional innovation vector
//!
//! # Variants
//!
//! | Variant | A, B constraints | Parameters |
//! |---------|-----------------|------------|
//! | Full BEKK | Full k×k matrices | k(k+1)/2 + 2k² |
//! | Diagonal BEKK | A, B diagonal | k(k+1)/2 + 2k |
//! | Scalar BEKK | A = aI, B = bI | k(k+1)/2 + 2 |
//!
//! # References
//! - Engle, R. F. & Kroner, K. F. (1995). Multivariate Simultaneous Generalized
//!   ARCH. *Econometric Theory*, 11(1), 122–150.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::bekk_garch::{BEKKConfig, BEKKVariant, fit_bekk};
//!
//! let r1: Vec<f64> = vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.005, 0.031, -0.013, 0.009,
//!     0.002, -0.027, 0.016, -0.007, 0.013, 0.004,
//! ];
//! let r2: Vec<f64> = vec![
//!     0.005, -0.01, 0.008, -0.004, 0.006, 0.015, -0.009, 0.012,
//!     -0.002, 0.004, 0.018, -0.007, 0.011, -0.003, 0.009, -0.012,
//!     0.014, 0.001, -0.008, 0.010, -0.003, 0.020, -0.006, 0.005,
//!     0.001, -0.015, 0.009, -0.004, 0.008, 0.002,
//! ];
//! let returns = vec![r1, r2];
//! let config = BEKKConfig {
//!     variant: BEKKVariant::Scalar,
//!     ..BEKKConfig::default()
//! };
//! let result = fit_bekk(&returns, &config).expect("BEKK should fit");
//! // All covariance matrices are positive definite by construction
//! ```

use crate::error::{Result, TimeSeriesError};

// ============================================================
// Configuration
// ============================================================

/// BEKK model variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BEKKVariant {
    /// Full BEKK: A, B are unrestricted k×k matrices
    Full,
    /// Diagonal BEKK: A, B are diagonal matrices
    Diagonal,
    /// Scalar BEKK: A = aI, B = bI
    Scalar,
}

/// Configuration for BEKK-GARCH estimation.
#[derive(Debug, Clone)]
pub struct BEKKConfig {
    /// BEKK variant (Full, Diagonal, or Scalar)
    pub variant: BEKKVariant,
    /// Maximum iterations for optimisation
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for BEKKConfig {
    fn default() -> Self {
        Self {
            variant: BEKKVariant::Diagonal,
            max_iter: 2000,
            tol: 1e-6,
        }
    }
}

// ============================================================
// Result types
// ============================================================

/// Result of BEKK-GARCH estimation.
#[derive(Debug, Clone)]
pub struct BEKKResult {
    /// Lower-triangular matrix C (k×k, row-major)
    pub c_matrix: Vec<f64>,
    /// Matrix A (k×k, row-major) — for scalar/diagonal, stored as full matrix
    pub a_matrix: Vec<f64>,
    /// Matrix B (k×k, row-major) — for scalar/diagonal, stored as full matrix
    pub b_matrix: Vec<f64>,
    /// BEKK variant used
    pub variant: BEKKVariant,
    /// Conditional covariance matrices over time: Vec of k×k row-major
    pub covariances: Vec<Vec<f64>>,
    /// Log-likelihood value
    pub log_likelihood: f64,
    /// Number of series
    pub k: usize,
    /// Number of time observations
    pub n_obs: usize,
}

/// BEKK covariance forecast.
#[derive(Debug, Clone)]
pub struct BEKKForecast {
    /// Forecast step
    pub step: usize,
    /// Forecasted covariance matrix (k×k, row-major)
    pub cov_matrix: Vec<f64>,
}

// ============================================================
// Matrix operations (small dense k×k)
// ============================================================

/// Matrix multiply: C = A * B (k×k).
fn mat_mul(a: &[f64], b: &[f64], k: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += a[i * k + p] * b[p * k + j];
            }
            c[i * k + j] = sum;
        }
    }
    c
}

/// Matrix transpose (k×k).
fn mat_transpose(a: &[f64], k: usize) -> Vec<f64> {
    let mut at = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            at[j * k + i] = a[i * k + j];
        }
    }
    at
}

/// Outer product: ε ε^T.
fn outer_product(eps: &[f64], k: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            out[i * k + j] = eps[i] * eps[j];
        }
    }
    out
}

/// Matrix addition: C = A + B.
fn mat_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// Cholesky log-determinant of a k×k SPD matrix.
fn cholesky_logdet(a: &[f64], k: usize) -> Result<f64> {
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            if i == j {
                let diag = a[i * k + i] - sum;
                if diag <= 1e-15 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "BEKK: covariance matrix not positive definite".to_string(),
                    ));
                }
                l[i * k + j] = diag.sqrt();
            } else {
                l[i * k + j] = (a[i * k + j] - sum) / l[j * k + j];
            }
        }
    }
    let mut log_det = 0.0_f64;
    for i in 0..k {
        log_det += 2.0 * l[i * k + i].ln();
    }
    Ok(log_det)
}

/// Solve Ax = b via Cholesky for k×k SPD A.
fn cholesky_solve(a: &[f64], b: &[f64], k: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0_f64;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            if i == j {
                let diag = a[i * k + i] - sum;
                if diag <= 1e-15 {
                    return Err(TimeSeriesError::NumericalInstability(
                        "BEKK solve: not positive definite".to_string(),
                    ));
                }
                l[i * k + j] = diag.sqrt();
            } else {
                l[i * k + j] = (a[i * k + j] - sum) / l[j * k + j];
            }
        }
    }

    // Forward: Ly = b
    let mut y = vec![0.0_f64; k];
    for i in 0..k {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i * k + j] * y[j];
        }
        y[i] = sum / l[i * k + i];
    }

    // Backward: L^T x = y
    let mut x = vec![0.0_f64; k];
    for i in (0..k).rev() {
        let mut sum = y[i];
        for j in (i + 1)..k {
            sum -= l[j * k + i] * x[j];
        }
        x[i] = sum / l[i * k + i];
    }

    Ok(x)
}

/// Check if a k×k matrix is positive definite (via Cholesky attempt).
fn is_positive_definite(a: &[f64], k: usize) -> bool {
    cholesky_logdet(a, k).is_ok()
}

// ============================================================
// BEKK recursion and log-likelihood
// ============================================================

/// Compute BEKK conditional covariance series and log-likelihood.
///
/// H_t = CC^T + A^T ε_{t-1}ε_{t-1}^T A + B^T H_{t-1} B
fn bekk_recursion(
    returns: &[Vec<f64>],
    c: &[f64],
    a: &[f64],
    b: &[f64],
    k: usize,
    n: usize,
) -> Result<(f64, Vec<Vec<f64>>)> {
    let at = mat_transpose(a, k);
    let bt = mat_transpose(b, k);

    // CC^T (intercept component)
    let ct = mat_transpose(c, k);
    let cc_t = mat_mul(c, &ct, k);

    // Initial H_0 = sample covariance
    let mut sample_cov = vec![0.0_f64; k * k];
    for t in 0..n {
        let eps: Vec<f64> = (0..k).map(|i| returns[i][t]).collect();
        let outer = outer_product(&eps, k);
        for idx in 0..(k * k) {
            sample_cov[idx] += outer[idx];
        }
    }
    for val in sample_cov.iter_mut() {
        *val /= n as f64;
    }

    let mut covariances = Vec::with_capacity(n);
    let mut h_prev = sample_cov;
    covariances.push(h_prev.clone());

    let log2pi_k = (k as f64) * (2.0 * std::f64::consts::PI).ln();
    let mut ll = 0.0_f64;

    for t in 1..n {
        let eps_prev: Vec<f64> = (0..k).map(|i| returns[i][t - 1]).collect();
        let eps_outer = outer_product(&eps_prev, k);

        // A^T * (ε_{t-1}ε_{t-1}^T) * A
        let temp1 = mat_mul(&at, &eps_outer, k);
        let arch_term = mat_mul(&temp1, a, k);

        // B^T * H_{t-1} * B
        let temp2 = mat_mul(&bt, &h_prev, k);
        let garch_term = mat_mul(&temp2, b, k);

        // H_t = CC^T + A^T εε^T A + B^T H B
        let h_t = mat_add(&cc_t, &mat_add(&arch_term, &garch_term));

        // Log-likelihood: -0.5 * [k*log(2π) + log|H_t| + ε_t^T H_t^{-1} ε_t]
        let eps_t: Vec<f64> = (0..k).map(|i| returns[i][t]).collect();

        match cholesky_logdet(&h_t, k) {
            Ok(log_det) => {
                if let Ok(h_inv_eps) = cholesky_solve(&h_t, &eps_t, k) {
                    let quad: f64 = eps_t
                        .iter()
                        .zip(h_inv_eps.iter())
                        .map(|(&e, &v)| e * v)
                        .sum();
                    ll -= 0.5 * (log2pi_k + log_det + quad);
                } else {
                    ll -= 1e10; // penalty for numerical issues
                }
            }
            Err(_) => {
                ll -= 1e10; // penalty
            }
        }

        covariances.push(h_t.clone());
        h_prev = h_t;
    }

    Ok((ll, covariances))
}

// ============================================================
// Parameter mapping
// ============================================================

/// Build C, A, B matrices from a raw parameter vector for Scalar BEKK.
fn params_to_matrices_scalar(raw: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // C: lower triangular with k(k+1)/2 free params
    let n_c = k * (k + 1) / 2;
    let c = lower_tri_from_params(&raw[..n_c], k);

    // a, b: scalar multipliers
    let a_val = sigmoid(raw[n_c]) * 0.5; // bounded to (0, 0.5)
    let b_val = sigmoid(raw[n_c + 1]) * 0.95; // bounded to (0, 0.95)

    // Ensure a² + b² < 1 for stationarity
    let sum_sq = a_val * a_val + b_val * b_val;
    let (a_adj, b_adj) = if sum_sq >= 0.99 {
        let scale = (0.95 / sum_sq).sqrt();
        (a_val * scale, b_val * scale)
    } else {
        (a_val, b_val)
    };

    let mut a_mat = vec![0.0_f64; k * k];
    let mut b_mat = vec![0.0_f64; k * k];
    for i in 0..k {
        a_mat[i * k + i] = a_adj;
        b_mat[i * k + i] = b_adj;
    }

    (c, a_mat, b_mat)
}

/// Build C, A, B matrices from a raw parameter vector for Diagonal BEKK.
fn params_to_matrices_diagonal(raw: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_c = k * (k + 1) / 2;
    let c = lower_tri_from_params(&raw[..n_c], k);

    let mut a_mat = vec![0.0_f64; k * k];
    let mut b_mat = vec![0.0_f64; k * k];

    for i in 0..k {
        let a_i = sigmoid(raw[n_c + i]) * 0.5;
        let b_i = sigmoid(raw[n_c + k + i]) * 0.95;

        // Stationarity: a_i² + b_i² < 1 per element
        let sum_sq = a_i * a_i + b_i * b_i;
        if sum_sq >= 0.99 {
            let scale = (0.95 / sum_sq).sqrt();
            a_mat[i * k + i] = a_i * scale;
            b_mat[i * k + i] = b_i * scale;
        } else {
            a_mat[i * k + i] = a_i;
            b_mat[i * k + i] = b_i;
        }
    }

    (c, a_mat, b_mat)
}

/// Build C, A, B matrices from a raw parameter vector for Full BEKK.
fn params_to_matrices_full(raw: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_c = k * (k + 1) / 2;
    let c = lower_tri_from_params(&raw[..n_c], k);

    let n_a = k * k;
    let mut a_mat = vec![0.0_f64; k * k];
    for i in 0..n_a {
        a_mat[i] = raw[n_c + i] * 0.3; // scale down
    }

    let mut b_mat = vec![0.0_f64; k * k];
    for i in 0..n_a {
        b_mat[i] = raw[n_c + n_a + i] * 0.3;
    }

    // Spectral norm check: ensure the recursion is stable
    // Simple heuristic: scale if Frobenius norms are too large
    let a_frob: f64 = a_mat.iter().map(|&v| v * v).sum::<f64>().sqrt();
    let b_frob: f64 = b_mat.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if a_frob + b_frob > 1.5 {
        let scale = 1.2 / (a_frob + b_frob);
        for v in a_mat.iter_mut() {
            *v *= scale;
        }
        for v in b_mat.iter_mut() {
            *v *= scale;
        }
    }

    (c, a_mat, b_mat)
}

/// Build a lower-triangular matrix from parameters.
fn lower_tri_from_params(params: &[f64], k: usize) -> Vec<f64> {
    let mut mat = vec![0.0_f64; k * k];
    let mut idx = 0;
    for i in 0..k {
        for j in 0..=i {
            if i == j {
                // Diagonal: must be positive (use softplus)
                mat[i * k + j] = softplus(params[idx]);
            } else {
                mat[i * k + j] = params[idx] * 0.01; // scale off-diagonal
            }
            idx += 1;
        }
    }
    mat
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        1e-8
    } else {
        (1.0 + x.exp()).ln().max(1e-8)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================
// Optimisation
// ============================================================

/// Number of free parameters for each BEKK variant.
fn n_params(k: usize, variant: BEKKVariant) -> usize {
    let n_c = k * (k + 1) / 2;
    match variant {
        BEKKVariant::Scalar => n_c + 2,
        BEKKVariant::Diagonal => n_c + 2 * k,
        BEKKVariant::Full => n_c + 2 * k * k,
    }
}

/// Nelder-Mead optimisation for BEKK parameters.
fn nelder_mead_bekk<F>(f: F, x0: Vec<f64>, max_iter: usize, tol: f64) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for i in 0..n {
        let mut vertex = x0.clone();
        vertex[i] += if vertex[i].abs() > 1e-5 {
            0.05 * vertex[i].abs()
        } else {
            0.00025
        };
        simplex.push(vertex);
    }
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _iter in 0..max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let spread: f64 = order
            .iter()
            .map(|&i| (fvals[i] - fvals[best]).abs())
            .fold(0.0_f64, f64::max);
        if spread < tol {
            break;
        }

        let mut centroid = vec![0.0_f64; n];
        for &i in order.iter().take(n) {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + 1.0 * (centroid[j] - simplex[worst][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (reflected[j] - centroid[j]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            let contracted: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 0.5 * (simplex[worst][j] - centroid[j]))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                let best_vertex = simplex[best].clone();
                for i in 0..=n {
                    if i != best {
                        for j in 0..n {
                            simplex[i][j] = best_vertex[j] + 0.5 * (simplex[i][j] - best_vertex[j]);
                        }
                        fvals[i] = f(&simplex[i]);
                    }
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = order[0];
    (simplex[best].clone(), fvals[best])
}

// ============================================================
// Public API
// ============================================================

/// Fit a BEKK-GARCH model to multivariate return data.
///
/// # Arguments
/// * `returns` — Vector of return series, one per asset. All must have the same length.
/// * `config` — BEKK configuration.
///
/// # Returns
/// A `BEKKResult` with estimated matrices C, A, B, conditional covariances, and log-likelihood.
pub fn fit_bekk(returns: &[Vec<f64>], config: &BEKKConfig) -> Result<BEKKResult> {
    let k = returns.len();
    if k < 1 {
        return Err(TimeSeriesError::InvalidInput(
            "BEKK-GARCH requires at least 1 series".to_string(),
        ));
    }
    let n = returns[0].len();
    for (i, series) in returns.iter().enumerate() {
        if series.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: series.len(),
            });
        }
        if series.len() < 20 {
            return Err(TimeSeriesError::InsufficientData {
                message: format!("BEKK-GARCH: series {} too short", i),
                required: 20,
                actual: series.len(),
            });
        }
    }

    let np = n_params(k, config.variant);
    let variant = config.variant;

    // Compute sample covariance for initialisation
    let mut sample_cov = vec![0.0_f64; k * k];
    for t in 0..n {
        for i in 0..k {
            for j in 0..k {
                sample_cov[i * k + j] += returns[i][t] * returns[j][t];
            }
        }
    }
    for val in sample_cov.iter_mut() {
        *val /= n as f64;
    }

    // Initial parameter guess
    let mut x0 = vec![0.0_f64; np];
    // C params: initialise from Cholesky of sample_cov * 0.1
    let n_c = k * (k + 1) / 2;
    {
        let scaled_cov: Vec<f64> = sample_cov.iter().map(|&v| v * 0.1).collect();
        // Simple Cholesky for initialisation
        let mut l = vec![0.0_f64; k * k];
        for i in 0..k {
            for j in 0..=i {
                let mut sum = 0.0_f64;
                for p in 0..j {
                    sum += l[i * k + p] * l[j * k + p];
                }
                if i == j {
                    let diag = (scaled_cov[i * k + i] - sum).max(1e-8);
                    l[i * k + j] = diag.sqrt();
                } else {
                    let denom = l[j * k + j];
                    l[i * k + j] = if denom.abs() > 1e-15 {
                        (scaled_cov[i * k + j] - sum) / denom
                    } else {
                        0.0
                    };
                }
            }
        }

        let mut idx = 0;
        for i in 0..k {
            for j in 0..=i {
                if i == j {
                    // Inverse softplus
                    let v = l[i * k + j].max(1e-4);
                    x0[idx] = (v.exp() - 1.0).ln().max(-5.0);
                } else {
                    x0[idx] = l[i * k + j] / 0.01; // inverse of the 0.01 scaling
                }
                idx += 1;
            }
        }
    }

    // A, B initial values
    match variant {
        BEKKVariant::Scalar => {
            // inverse sigmoid of 0.15 and 0.85
            x0[n_c] = (0.15_f64 / 0.85).ln();
            x0[n_c + 1] = (0.85_f64 / 0.15).ln();
        }
        BEKKVariant::Diagonal => {
            for i in 0..k {
                x0[n_c + i] = (0.15_f64 / 0.85).ln();
                x0[n_c + k + i] = (0.85_f64 / 0.15).ln();
            }
        }
        BEKKVariant::Full => {
            // Identity-like initialisation
            for i in 0..k {
                x0[n_c + i * k + i] = 1.0; // A ≈ 0.3 * I after scaling
                x0[n_c + k * k + i * k + i] = 2.5; // B ≈ 0.3 * 2.5 = 0.75
            }
        }
    }

    let returns_clone: Vec<Vec<f64>> = returns.to_vec();

    let obj = move |raw: &[f64]| -> f64 {
        let (c, a, b) = match variant {
            BEKKVariant::Scalar => params_to_matrices_scalar(raw, k),
            BEKKVariant::Diagonal => params_to_matrices_diagonal(raw, k),
            BEKKVariant::Full => params_to_matrices_full(raw, k),
        };
        match bekk_recursion(&returns_clone, &c, &a, &b, k, n) {
            Ok((ll, _)) => -ll,
            Err(_) => f64::INFINITY,
        }
    };

    let (best_raw, _neg_ll) = nelder_mead_bekk(obj, x0, config.max_iter, config.tol);

    let (c, a, b) = match config.variant {
        BEKKVariant::Scalar => params_to_matrices_scalar(&best_raw, k),
        BEKKVariant::Diagonal => params_to_matrices_diagonal(&best_raw, k),
        BEKKVariant::Full => params_to_matrices_full(&best_raw, k),
    };

    let (ll, covariances) = bekk_recursion(returns, &c, &a, &b, k, n)?;

    Ok(BEKKResult {
        c_matrix: c,
        a_matrix: a,
        b_matrix: b,
        variant: config.variant,
        covariances,
        log_likelihood: ll,
        k,
        n_obs: n,
    })
}

/// Forecast covariance matrices h steps ahead from a fitted BEKK model.
///
/// Uses the BEKK recursion with E[ε_t ε_t^T] = H_t for future periods.
pub fn bekk_forecast(result: &BEKKResult, steps: usize) -> Result<Vec<BEKKForecast>> {
    if steps == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "steps".to_string(),
            message: "Forecast horizon must be >= 1".to_string(),
        });
    }

    let k = result.k;
    let at = mat_transpose(&result.a_matrix, k);
    let bt = mat_transpose(&result.b_matrix, k);
    let ct = mat_transpose(&result.c_matrix, k);
    let cc_t = mat_mul(&result.c_matrix, &ct, k);

    let h_last = result
        .covariances
        .last()
        .ok_or_else(|| TimeSeriesError::InvalidInput("No covariance data".to_string()))?
        .clone();

    let mut forecasts = Vec::with_capacity(steps);
    let mut h_prev = h_last;

    for step in 0..steps {
        // E[ε_t ε_t^T | F_{t-1}] = H_{t-1} for future periods
        // So H_{t+1} = CC^T + A^T H_t A + B^T H_t B
        let temp1 = mat_mul(&at, &h_prev, k);
        let arch_term = mat_mul(&temp1, &result.a_matrix, k);

        let temp2 = mat_mul(&bt, &h_prev, k);
        let garch_term = mat_mul(&temp2, &result.b_matrix, k);

        let h_next = mat_add(&cc_t, &mat_add(&arch_term, &garch_term));

        forecasts.push(BEKKForecast {
            step: step + 1,
            cov_matrix: h_next.clone(),
        });

        h_prev = h_next;
    }

    Ok(forecasts)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_returns(n: usize) -> Vec<Vec<f64>> {
        let mut r1 = Vec::with_capacity(n);
        let mut r2 = Vec::with_capacity(n);
        for i in 0..n {
            let z1 = 0.01 * ((i as f64 * 1.37 + 0.5).sin());
            let z2 = 0.008 * ((i as f64 * 2.71 + 1.3).sin());
            let vol = if i % 7 < 3 { 2.0 } else { 1.0 };
            r1.push(z1 * vol);
            r2.push(z2 * vol + 0.3 * z1 * vol);
        }
        vec![r1, r2]
    }

    #[test]
    fn test_bekk_positive_definite_covariances() {
        let returns = make_returns(60);
        let config = BEKKConfig {
            variant: BEKKVariant::Scalar,
            max_iter: 1000,
            tol: 1e-5,
        };
        let result = fit_bekk(&returns, &config).expect("BEKK should fit");

        // All covariance matrices must be positive definite
        for (t, cov) in result.covariances.iter().enumerate() {
            assert!(
                is_positive_definite(cov, result.k),
                "Covariance at t={} must be positive definite",
                t
            );
        }
    }

    #[test]
    fn test_bekk_scalar_single_series() {
        // Scalar BEKK on a single series should behave like univariate GARCH
        let returns = vec![vec![
            0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009, -0.003, 0.007, 0.025, -0.014,
            0.008, -0.006, 0.011, -0.019, 0.022, 0.003, -0.011, 0.017, -0.005, 0.031, -0.013,
            0.009, 0.002, -0.027, 0.016, -0.007, 0.013, 0.004,
        ]];
        let config = BEKKConfig {
            variant: BEKKVariant::Scalar,
            max_iter: 1500,
            tol: 1e-6,
        };
        let result = fit_bekk(&returns, &config).expect("BEKK should fit single series");

        assert_eq!(result.k, 1);
        assert!(result.log_likelihood.is_finite());

        // All 1×1 covariances (i.e., variances) should be positive
        for cov in &result.covariances {
            assert!(cov[0] > 0.0, "Variance must be positive");
        }
    }

    #[test]
    fn test_bekk_diagonal_variant() {
        let returns = make_returns(60);
        let config = BEKKConfig {
            variant: BEKKVariant::Diagonal,
            max_iter: 1500,
            tol: 1e-5,
        };
        let result = fit_bekk(&returns, &config).expect("Diagonal BEKK should fit");

        assert_eq!(result.k, 2);
        assert!(result.log_likelihood.is_finite());

        // A and B should be diagonal
        for i in 0..result.k {
            for j in 0..result.k {
                if i != j {
                    assert!(
                        result.a_matrix[i * result.k + j].abs() < 1e-10,
                        "A must be diagonal"
                    );
                    assert!(
                        result.b_matrix[i * result.k + j].abs() < 1e-10,
                        "B must be diagonal"
                    );
                }
            }
        }
    }

    #[test]
    fn test_bekk_forecast() {
        let returns = make_returns(60);
        let config = BEKKConfig {
            variant: BEKKVariant::Scalar,
            max_iter: 1000,
            tol: 1e-5,
        };
        let result = fit_bekk(&returns, &config).expect("BEKK should fit");

        let forecasts = bekk_forecast(&result, 5).expect("Should forecast");
        assert_eq!(forecasts.len(), 5);

        for fc in &forecasts {
            // Forecasted covariance should be positive definite
            assert!(
                is_positive_definite(&fc.cov_matrix, result.k),
                "Forecasted covariance must be positive definite at step {}",
                fc.step
            );
        }
    }

    #[test]
    fn test_bekk_dimension_mismatch() {
        let returns = vec![vec![0.01, -0.02, 0.015], vec![0.005, -0.01]];
        let config = BEKKConfig::default();
        assert!(fit_bekk(&returns, &config).is_err());
    }

    #[test]
    fn test_bekk_too_short() {
        let returns = vec![vec![0.01; 5], vec![0.02; 5]];
        let config = BEKKConfig::default();
        assert!(fit_bekk(&returns, &config).is_err());
    }

    #[test]
    fn test_bekk_covariance_symmetry() {
        let returns = make_returns(60);
        let config = BEKKConfig {
            variant: BEKKVariant::Scalar,
            max_iter: 1000,
            tol: 1e-5,
        };
        let result = fit_bekk(&returns, &config).expect("BEKK should fit");

        for cov in &result.covariances {
            for i in 0..result.k {
                for j in 0..result.k {
                    assert!(
                        (cov[i * result.k + j] - cov[j * result.k + i]).abs() < 1e-10,
                        "Covariance matrix must be symmetric"
                    );
                }
            }
        }
    }
}
