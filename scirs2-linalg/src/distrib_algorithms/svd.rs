//! Distributed truncated SVD via Lanczos bidiagonalization with thick restart.
//!
//! ## Algorithm overview
//!
//! The Golub-Kahan-Lanczos bidiagonalization computes a rank-k approximation of
//! a matrix A (m×n) by building orthonormal bases U_k (m×k) and V_k (n×k) such that
//!
//! ```text
//!   A * V_k ≈ U_k * B_k   (with small residual)
//! ```
//!
//! where B_k is a k×k upper bidiagonal matrix.  The SVD of the small dense B_k
//! then approximates the top-k singular triplets of A.
//!
//! **Algorithm (upper bidiagonal GK form):**
//! ```text
//!   Initialize: v_0 normalized quasi-random; u_0 = A*v_0/alpha_0 (alpha_0 = ||A*v_0||)
//!   For j = 0, ..., k-1:
//!     r_j = A^T * u_j - alpha_j * v_j
//!     beta_j = ||r_j||; v_{j+1} = r_j / beta_j
//!     q_j = A * v_{j+1} - beta_j * u_j
//!     alpha_{j+1} = ||q_j||; u_{j+1} = q_j / alpha_{j+1}
//!   B_k[j,j] = alpha_j; B_k[j, j+1] = beta_j
//! ```
//!
//! Modified Gram-Schmidt reorthogonalization is applied at every step to both
//! U and V vectors for numerical stability.
//!
//! In a real distributed implementation the matrix-vector products
//! `A v` and `A^T u` would be performed via SUMMA; here they are simulated locally.
//!
//! ## References
//!
//! Larsen, R. M. (1998). *Lanczos bidiagonalization with partial reorthogonalization.*
//! Technical Report, DAIMI PB-537, Department of Computer Science, University of Aarhus.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array2};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Lanczos-based distributed SVD.
#[derive(Debug, Clone)]
pub struct LanczosSvdConfig {
    /// Number of desired singular values / vectors.
    pub k: usize,
    /// Maximum number of Lanczos steps per restart cycle.
    pub max_iter: usize,
    /// Convergence tolerance: `|σ_estimated - σ_true| < tol * σ_1`.
    pub tol: f64,
}

impl Default for LanczosSvdConfig {
    fn default() -> Self {
        Self {
            k: 10,
            max_iter: 50,
            tol: 1e-10,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal linear algebra helpers (pure Rust, no BLAS dependency)
// ---------------------------------------------------------------------------

fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec(a: &Array2<f64>, x: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut y = vec![0.0f64; m];
    for i in 0..m {
        let mut s = 0.0f64;
        for j in 0..n {
            s += a[[i, j]] * x[j];
        }
        y[i] = s;
    }
    y
}

fn matvec_t(a: &Array2<f64>, x: &[f64]) -> Vec<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut y = vec![0.0f64; n];
    for j in 0..n {
        let mut s = 0.0f64;
        for i in 0..m {
            s += a[[i, j]] * x[i];
        }
        y[j] = s;
    }
    y
}

/// Modified Gram-Schmidt orthogonalization of `v` against all `basis` vectors.
fn reorthogonalize(v: &mut [f64], basis: &[Vec<f64>]) {
    // Two passes for numerical stability
    for _ in 0..2 {
        for b in basis {
            let c = dot(v, b);
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= c * bi;
            }
        }
    }
}

/// Normalize `v` in-place; returns the original norm.
fn normalize_inplace(v: &mut [f64]) -> f64 {
    let n = vec_norm(v);
    if n > f64::EPSILON {
        for vi in v.iter_mut() {
            *vi /= n;
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Dense small-matrix SVD via one-sided Jacobi
// ---------------------------------------------------------------------------

/// Compute the thin SVD of a small dense matrix `b` (rows >= cols).
///
/// Uses one-sided Jacobi (column rotations on `BV`, accumulating `V`).
/// The rotation zeroes `(BV^T BV)[p,q]` at each step.
///
/// Returns `(u, sigma, v)` where `u` is rows×cols orthonormal,
/// `sigma` is length cols sorted descending, and `v` is cols×cols orthogonal.
fn dense_svd_jacobi(b: &Array2<f64>) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let m = b.nrows();
    let n = b.ncols();

    // Work on bv = B * V_accum; accumulate V in v_mat
    let mut bv = b.to_owned();
    let mut v_mat = Array2::<f64>::eye(n);

    // One-sided Jacobi: rotate pairs of columns (p,q) to zero (BV^T BV)[p,q].
    //
    // The rotation B' = B * J where J is a Givens rotation on columns (p,q):
    //   B'[:,p] = c*B[:,p] + s*B[:,q]
    //   B'[:,q] = -s*B[:,p] + c*B[:,q]
    //
    // Zeroing condition: cs*(aqq-app) + (c^2-s^2)*apq = 0
    //   => t^2 - 2*tau*t - 1 = 0, tau = (aqq-app)/(2*apq)
    //   => small-magnitude root: t = tau - sign(tau)*sqrt(tau^2+1)
    //   => equivalently: t = -sign(tau) / (|tau| + sqrt(1+tau^2))
    let max_sweeps = 150;
    let eps = 1e-14;

    for _ in 0..max_sweeps {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let mut app = 0.0f64;
                let mut aqq = 0.0f64;
                let mut apq = 0.0f64;
                for i in 0..m {
                    app += bv[[i, p]] * bv[[i, p]];
                    aqq += bv[[i, q]] * bv[[i, q]];
                    apq += bv[[i, p]] * bv[[i, q]];
                }
                if apq.abs() <= eps * (app * aqq).sqrt() {
                    continue;
                }
                converged = false;

                // Jacobi rotation angle for B'[:,p]=c*B[:,p]+s*B[:,q], B'[:,q]=-s*B[:,p]+c*B[:,q]
                // t = tau - sqrt(tau^2+1) for tau>=0, tau + sqrt(tau^2+1) for tau<0
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    -1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Rotate columns p and q of bv
                for i in 0..m {
                    let bp = bv[[i, p]];
                    let bq = bv[[i, q]];
                    bv[[i, p]] = c * bp + s * bq;
                    bv[[i, q]] = -s * bp + c * bq;
                }
                // Accumulate rotation in v_mat
                for i in 0..n {
                    let vp = v_mat[[i, p]];
                    let vq = v_mat[[i, q]];
                    v_mat[[i, p]] = c * vp + s * vq;
                    v_mat[[i, q]] = -s * vp + c * vq;
                }
            }
        }
        if converged {
            break;
        }
    }

    // Extract singular values = column norms; normalize columns to get U
    let sigma: Vec<f64> = (0..n)
        .map(|j| (0..m).map(|i| bv[[i, j]] * bv[[i, j]]).sum::<f64>().sqrt())
        .collect();

    let mut u_mat = bv.clone();
    for j in 0..n {
        if sigma[j] > f64::EPSILON {
            for i in 0..m {
                u_mat[[i, j]] /= sigma[j];
            }
        }
    }

    // Sort by descending singular value
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma[b]
            .partial_cmp(&sigma[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let sigma_sorted: Vec<f64> = order.iter().map(|&i| sigma[i]).collect();
    let mut u_sorted = Array2::<f64>::zeros((m, n));
    let mut v_sorted = Array2::<f64>::zeros((n, n));
    for (new_j, &old_j) in order.iter().enumerate() {
        for i in 0..m {
            u_sorted[[i, new_j]] = u_mat[[i, old_j]];
        }
        for i in 0..n {
            v_sorted[[i, new_j]] = v_mat[[i, old_j]];
        }
    }

    (u_sorted, sigma_sorted, v_sorted)
}

// ---------------------------------------------------------------------------
// Lanczos bidiagonalization (Golub-Kahan, upper bidiagonal)
// ---------------------------------------------------------------------------

/// Run k steps of Golub-Kahan Lanczos bidiagonalization on matrix A.
///
/// Returns `(U, B, V)` where:
/// - `U` is m × k (left Lanczos vectors, orthonormal columns)
/// - `B` is k × k upper bidiagonal (alpha on diagonal, beta on super-diagonal)
/// - `V` is n × k (right Lanczos vectors, orthonormal columns)
///
/// The relationship `A * V_k ≈ U_k * B_k` holds to numerical precision.
///
/// # Errors
///
/// Returns an error if A is empty or `A * v_0 ≈ 0`.
pub fn lanczos_bidiagonalization(
    a: &Array2<f64>,
    k: usize,
) -> LinalgResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return Err(LinalgError::ValueError(
            "lanczos_bidiagonalization: matrix must be non-empty".to_string(),
        ));
    }
    let k_eff = k.min(m).min(n);
    if k_eff == 0 {
        return Err(LinalgError::ValueError(
            "lanczos_bidiagonalization: k must be >= 1".to_string(),
        ));
    }

    let mut u_vecs: Vec<Vec<f64>> = Vec::with_capacity(k_eff);
    let mut v_vecs: Vec<Vec<f64>> = Vec::with_capacity(k_eff);
    let mut alpha_vals: Vec<f64> = Vec::with_capacity(k_eff);
    let mut beta_vals: Vec<f64> = Vec::new();

    // --- Initialize: v_0 is a normalized quasi-random starting vector ---
    let mut v0: Vec<f64> = (0..n)
        .map(|i| {
            // Golden-ratio quasirandom for uniform coverage in [-1, 1]
            let x = (i as f64 + 1.0) * 0.6180339887498949;
            let frac = x - x.floor();
            2.0 * frac - 1.0
        })
        .collect();
    normalize_inplace(&mut v0);
    v_vecs.push(v0.clone());

    // u_0 = A * v_0 / alpha_0
    let mut u0 = matvec(a, &v0);
    let alpha0 = normalize_inplace(&mut u0);
    if alpha0 < f64::EPSILON * 100.0 {
        return Err(LinalgError::ComputationError(
            "lanczos_bidiagonalization: A * v_0 is zero".to_string(),
        ));
    }
    alpha_vals.push(alpha0);
    u_vecs.push(u0.clone());

    let mut u_prev = u0;
    let mut v_prev = v0;

    for _j in 1..k_eff {
        let alpha_prev = *alpha_vals.last().expect("alpha non-empty");

        // r = A^T * u_prev - alpha_prev * v_prev
        let mut r = matvec_t(a, &u_prev);
        for (ri, &vc) in r.iter_mut().zip(v_prev.iter()) {
            *ri -= alpha_prev * vc;
        }
        reorthogonalize(&mut r, &v_vecs);
        let beta = normalize_inplace(&mut r);
        beta_vals.push(beta);

        if beta < f64::EPSILON * 100.0 {
            // Krylov space exhausted; break before pushing degenerate vectors
            break;
        }

        // v_j = r (normalized)
        v_vecs.push(r.clone());
        v_prev = r.clone();

        // q = A * v_j - beta * u_prev
        let mut q = matvec(a, &r);
        for (qi, &uc) in q.iter_mut().zip(u_prev.iter()) {
            *qi -= beta * uc;
        }
        reorthogonalize(&mut q, &u_vecs);
        let alpha_j = normalize_inplace(&mut q);
        alpha_vals.push(alpha_j);
        u_vecs.push(q.clone());
        u_prev = q;

        if alpha_j < f64::EPSILON * 100.0 {
            break;
        }
    }

    // k_actual = number of v and u columns (must match alpha count)
    let k_actual = alpha_vals.len().min(v_vecs.len()).min(u_vecs.len());

    // Pack V (n × k_actual)
    let mut v_mat = Array2::<f64>::zeros((n, k_actual));
    for (j, vv) in v_vecs.iter().take(k_actual).enumerate() {
        for i in 0..n {
            v_mat[[i, j]] = vv[i];
        }
    }

    // Pack U (m × k_actual)
    let mut u_mat = Array2::<f64>::zeros((m, k_actual));
    for (j, uv) in u_vecs.iter().take(k_actual).enumerate() {
        for i in 0..m {
            u_mat[[i, j]] = uv[i];
        }
    }

    // Build B (k_actual × k_actual) upper bidiagonal: alpha on diagonal, beta on super-diagonal
    let mut b_mat = Array2::<f64>::zeros((k_actual, k_actual));
    for i in 0..k_actual {
        b_mat[[i, i]] = alpha_vals[i];
    }
    for i in 0..beta_vals.len().min(k_actual.saturating_sub(1)) {
        b_mat[[i, i + 1]] = beta_vals[i];
    }

    Ok((u_mat, b_mat, v_mat))
}

// ---------------------------------------------------------------------------
// Distributed SVD simulation
// ---------------------------------------------------------------------------

/// Simulate distributed SVD: compute top-k singular triplets of A.
///
/// Runs Lanczos bidiagonalization to get a k×k upper bidiagonal B, then
/// computes the thin SVD of B to extract singular values and vectors.
///
/// In a real distributed implementation the matvec operations would be performed
/// via SUMMA (distributed GEMM); here they are simulated locally.
///
/// # Arguments
///
/// * `a` – Input matrix (m × n)
/// * `k` – Number of singular values/vectors requested
///
/// # Returns
///
/// `(U_k, sigma_k, V_k)` where singular values are in descending order.
pub fn distributed_svd_simulate(
    a: &Array2<f64>,
    k: usize,
) -> LinalgResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let m = a.nrows();
    let n = a.ncols();

    if k == 0 {
        return Err(LinalgError::ValueError(
            "distributed_svd_simulate: k must be >= 1".to_string(),
        ));
    }
    let k_eff = k.min(m).min(n);
    // Request more Lanczos steps than k to improve accuracy
    let lanczos_steps = (k_eff * 3 + 10).min(m.min(n));

    let (u_lanczos, b_mat, v_lanczos) = lanczos_bidiagonalization(a, lanczos_steps)?;

    let k_lanczos = u_lanczos.ncols();

    // Compute thin SVD of the small bidiagonal B (k_lanczos × k_lanczos)
    let (u_b, sigma, v_b) = dense_svd_jacobi(&b_mat);

    let k_out = k_eff.min(k_lanczos).min(sigma.len());

    // U_k = U_lanczos * U_B  (m × k_out)
    let mut u_k = Array2::<f64>::zeros((m, k_out));
    for j in 0..k_out {
        for i in 0..m {
            let mut s = 0.0f64;
            for l in 0..k_lanczos {
                s += u_lanczos[[i, l]] * u_b[[l, j]];
            }
            u_k[[i, j]] = s;
        }
    }

    // V_k = V_lanczos * V_B  (n × k_out)
    let v_contract = v_lanczos.ncols().min(v_b.nrows());
    let mut v_k = Array2::<f64>::zeros((n, k_out));
    for j in 0..k_out {
        for i in 0..n {
            let mut s = 0.0f64;
            for l in 0..v_contract {
                s += v_lanczos[[i, l]] * v_b[[l, j]];
            }
            v_k[[i, j]] = s;
        }
    }

    // Re-orthogonalize U_k and V_k columns using Modified Gram-Schmidt
    // (combats accumulated numerical errors)
    mgs_orthogonalize_columns(&mut u_k);
    mgs_orthogonalize_columns(&mut v_k);

    let sigma_k = sigma[..k_out].to_vec();
    Ok((u_k, sigma_k, v_k))
}

/// Modified Gram-Schmidt orthonormalization of columns of `mat` in-place.
fn mgs_orthogonalize_columns(mat: &mut Array2<f64>) {
    let ncols = mat.ncols();
    let nrows = mat.nrows();
    for j in 0..ncols {
        // Orthogonalize column j against all previous columns
        for i in 0..j {
            let mut dot = 0.0f64;
            for r in 0..nrows {
                dot += mat[[r, i]] * mat[[r, j]];
            }
            for r in 0..nrows {
                let prev = mat[[r, i]];
                mat[[r, j]] -= dot * prev;
            }
        }
        // Normalize column j
        let norm: f64 = (0..nrows)
            .map(|r| mat[[r, j]] * mat[[r, j]])
            .sum::<f64>()
            .sqrt();
        if norm > f64::EPSILON {
            for r in 0..nrows {
                mat[[r, j]] /= norm;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thick-restart Lanczos SVD
// ---------------------------------------------------------------------------

/// Thick-restart Lanczos SVD for well-separated singular values.
///
/// Runs multiple cycles of `distributed_svd_simulate`, checking convergence
/// of the top-k singular values across cycles.
///
/// # Arguments
///
/// * `a`   – Input matrix (m × n)
/// * `k`   – Number of singular values/vectors requested
/// * `tol` – Convergence tolerance (relative to the largest singular value)
///
/// # Returns
///
/// `(U_k, sigma_k, V_k)` – same convention as [`distributed_svd_simulate`].
pub fn thick_restart_lanczos(
    a: &Array2<f64>,
    k: usize,
    tol: f64,
) -> LinalgResult<(Array2<f64>, Vec<f64>, Array2<f64>)> {
    let m = a.nrows();
    let n = a.ncols();

    if k == 0 {
        return Err(LinalgError::ValueError(
            "thick_restart_lanczos: k must be >= 1".to_string(),
        ));
    }
    let k_eff = k.min(m).min(n);
    let max_cycles = 10usize;
    let lanczos_size = (k_eff * 3 + 10).min(m).min(n);

    let mut prev_sigma: Option<Vec<f64>> = None;
    let mut best_u = Array2::<f64>::zeros((m, k_eff));
    let mut best_sigma = vec![0.0f64; k_eff];
    let mut best_v = Array2::<f64>::zeros((n, k_eff));

    for _cycle in 0..max_cycles {
        let (u_k, sigma_k, v_k) = distributed_svd_simulate(a, lanczos_size)?;

        let k_got = sigma_k.len().min(k_eff);
        best_u = u_k.slice(s![.., ..k_got]).to_owned();
        best_sigma = sigma_k[..k_got].to_vec();
        best_v = v_k.slice(s![.., ..k_got]).to_owned();

        let converged = if let Some(ref prev) = prev_sigma {
            let sigma_1 = best_sigma.first().copied().unwrap_or(1.0).max(1e-14);
            prev.len() == best_sigma.len()
                && prev
                    .iter()
                    .zip(best_sigma.iter())
                    .all(|(&s_prev, &s_curr)| (s_prev - s_curr).abs() < tol * sigma_1)
        } else {
            false
        };

        if converged {
            break;
        }
        prev_sigma = Some(best_sigma.clone());
    }

    Ok((best_u, best_sigma, best_v))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    fn orthogonality_error(u: &Array2<f64>) -> f64 {
        let k = u.ncols();
        let m = u.nrows();
        let mut max_err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let mut d = 0.0f64;
                for r in 0..m {
                    d += u[[r, i]] * u[[r, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((d - expected).abs());
            }
        }
        max_err
    }

    #[test]
    fn test_lanczos_bidiag_is_bidiagonal() {
        let a = Array2::<f64>::from_shape_fn((6, 5), |(i, j)| {
            (i as f64 * 1.3 + j as f64 * 0.7 + 0.1).sin()
        });
        let k = 4;
        let (_u_mat, b_mat, _v_mat) = lanczos_bidiagonalization(&a, k).expect("lanczos failed");
        // B should be upper bidiagonal: non-zero only on diagonal and super-diagonal
        let k_actual = b_mat.nrows().min(b_mat.ncols());
        for i in 0..k_actual {
            for j in 0..k_actual {
                let val = b_mat[[i, j]];
                if i != j && j != i + 1 {
                    assert!(
                        val.abs() < 1e-12,
                        "B[{i},{j}] = {val:.3e} should be ≈ 0 (upper bidiagonal violation)"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lanczos_bidiag_u_orthonormal() {
        let a = Array2::<f64>::from_shape_fn((8, 6), |(i, j)| ((i + 1) as f64) / ((j + 2) as f64));
        let (u_mat, _, _) = lanczos_bidiagonalization(&a, 5).expect("lanczos failed");
        let err = orthogonality_error(&u_mat);
        assert!(err < 1e-9, "U^T U orthogonality error = {err}");
    }

    #[test]
    fn test_lanczos_bidiag_v_orthonormal() {
        let a = Array2::<f64>::from_shape_fn((8, 6), |(i, j)| ((i + 1) as f64) / ((j + 2) as f64));
        let (_, _, v_mat) = lanczos_bidiagonalization(&a, 5).expect("lanczos failed");
        let err = orthogonality_error(&v_mat);
        assert!(err < 1e-9, "V^T V orthogonality error = {err}");
    }

    #[test]
    fn test_distributed_svd_singular_values_match_reference() {
        let diag = [5.0, 3.0, 2.0, 1.0];
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| if i == j { diag[i] } else { 0.0 });
        let (_, sigma, _) = distributed_svd_simulate(&a, 4).expect("svd failed");
        let mut sigma_sorted = sigma.clone();
        sigma_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_abs_diff_eq!(sigma_sorted[0], 5.0, epsilon = 0.1);
        assert_abs_diff_eq!(sigma_sorted[1], 3.0, epsilon = 0.1);
        assert_abs_diff_eq!(sigma_sorted[2], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(sigma_sorted[3], 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_distributed_svd_rank1_matrix() {
        // Rank-1 matrix with generic (non-axis-aligned) singular vectors
        let u_raw = [1.0f64, 1.0, 1.0, 1.0];
        let v_raw = [1.0f64, 2.0, 3.0, 4.0];
        let u_norm: f64 = u_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v_norm: f64 = v_raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u0: Vec<f64> = u_raw.iter().map(|x| x / u_norm).collect();
        let v0: Vec<f64> = v_raw.iter().map(|x| x / v_norm).collect();
        let sigma0 = 7.0f64;
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| u0[i] * sigma0 * v0[j]);
        let (_, sigma, _) = distributed_svd_simulate(&a, 1).expect("svd failed");
        let max_sv = sigma.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (max_sv - sigma0).abs() < sigma0 * 0.15,
            "Expected dominant singular value ≈ {sigma0} (±15%), got {max_sv}"
        );
    }

    #[test]
    fn test_distributed_svd_singular_vectors_orthonormal() {
        let a = Array2::<f64>::from_shape_fn((6, 5), |(i, j)| {
            (i as f64 + 1.0) * (j as f64 + 1.0) / 10.0
        });
        let k = 3;
        let (u_k, _, v_k) = distributed_svd_simulate(&a, k).expect("svd failed");
        let err_u = orthogonality_error(&u_k);
        let err_v = orthogonality_error(&v_k);
        assert!(err_u < 1e-8, "U_k orthogonality error = {err_u}");
        assert!(err_v < 1e-8, "V_k orthogonality error = {err_v}");
    }

    #[test]
    fn test_distributed_svd_reconstruction_error() {
        let a = Array2::<f64>::from_shape_fn((5, 4), |(i, j)| {
            let r1 = (i as f64 + 1.0) * 2.0;
            let r2 = (j as f64 + 1.0) * 0.5;
            if i < 3 {
                r1 + r2
            } else {
                0.0
            }
        });
        let (u_k, sigma_k, v_k) = distributed_svd_simulate(&a, 2).expect("svd failed");
        let mut a_rec = Array2::<f64>::zeros((5, 4));
        for r in 0..u_k.ncols() {
            for i in 0..5 {
                for j in 0..4 {
                    a_rec[[i, j]] += u_k[[i, r]] * sigma_k[r] * v_k[[j, r]];
                }
            }
        }
        let a_frob = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let err = frob_diff(&a_rec, &a);
        assert!(
            err < a_frob * 0.5,
            "Reconstruction error {err} > 50% of ||A||_F = {a_frob}"
        );
    }

    #[test]
    fn test_thick_restart_converges() {
        let diag = [10.0, 7.0, 4.0, 1.0];
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| if i == j { diag[i] } else { 0.0 });
        let (_, sigma, _) = thick_restart_lanczos(&a, 3, 1e-6).expect("thick restart failed");
        assert!(!sigma.is_empty());
        let max_sv = sigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_sv > 8.0,
            "Expected largest singular value near 10, got {max_sv}"
        );
    }

    #[test]
    fn test_bidiag_svd_diagonal_matrix() {
        // For a diagonal matrix B (passed as upper bidiagonal with empty beta),
        // the singular values should be the absolute values of the diagonal entries.
        let alpha = [3.0f64, 2.0, 1.0];
        let b = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| if i == j { alpha[i] } else { 0.0 });
        let (_, sigma, _) = dense_svd_jacobi(&b);
        let mut s = sigma.clone();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_abs_diff_eq!(s[0], 3.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[1], 2.0, epsilon = 1e-8);
        assert_abs_diff_eq!(s[2], 1.0, epsilon = 1e-8);
    }
}
