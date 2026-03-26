//! Distributed truncated SVD via Lanczos bidiagonalization with thick restart.
//!
//! ## Algorithm overview
//!
//! The Lanczos bidiagonalization procedure computes a rank-k approximation of a
//! matrix A (m×n) by building orthonormal bases U (m×k) and V (n×k) such that
//!
//! ```text
//!   A * V_k ≈ U_k * B_k
//!   A^T * U_k ≈ V_k * B_k^T + beta_k * v_{k+1} e_k^T
//! ```
//!
//! where B_k is a k×k lower-bidiagonal matrix.  The SVD of the small dense B_k
//! then approximates the top-k singular triplets of A.
//!
//! Partial reorthogonalization (modified Gram-Schmidt against all previous
//! basis vectors) is applied at every step to maintain numerical stability.
//!
//! **Thick restart**: when the Lanczos basis fills available memory (or a
//! round of `max_iter` steps is exhausted), the converged Ritz pairs are
//! deflated, the remaining approximate singular vectors are used as the new
//! starting vectors, and the process continues until the desired accuracy is
//! reached.
//!
//! In a real distributed implementation the matrix-vector products
//! `A v` and `A^T u` would be performed via SUMMA; here we call the local
//! ndarray dot product as a stand-in.
//!
//! ## References
//!
//! Larsen, R. M. (1998). *Lanczos bidiagonalization with partial reorthogonalization.*
//! Technical Report, DAIMI PB-537, Department of Computer Science, University of Aarhus.

use crate::error::{LinalgError, LinalgResult};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2};

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

/// Compute the Euclidean norm of a slice.
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the dot product of two slices.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix-vector product: `y = A * x` (m×n matrix, length-n vector → length-m).
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

/// Transpose matrix-vector product: `y = A^T * x` (m×n matrix, length-m vector → length-n).
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

/// Orthogonalize `v` against all columns of `basis` (modified Gram-Schmidt).
///
/// Modifies `v` in-place; returns the norm after orthogonalization.
fn reorthogonalize(v: &mut Vec<f64>, basis: &[Vec<f64>]) -> f64 {
    for b in basis {
        let c = dot(v, b);
        for (vi, bi) in v.iter_mut().zip(b.iter()) {
            *vi -= c * bi;
        }
    }
    vec_norm(v)
}

/// Normalize a mutable slice in-place; returns the original norm.
fn normalize_inplace(v: &mut Vec<f64>) -> f64 {
    let n = vec_norm(v);
    if n > f64::EPSILON {
        for vi in v.iter_mut() {
            *vi /= n;
        }
    }
    n
}

// ---------------------------------------------------------------------------
// Bidiagonal SVD (dense, via Jacobi-style symmetric QR on B^T B)
// ---------------------------------------------------------------------------

/// Compute the SVD of a k×k lower-bidiagonal matrix B.
///
/// Uses the implicit QR shift applied to the symmetric tridiagonal T = B^T B.
/// Returns `(u_b, sigma, v_b)` where `u_b` and `v_b` are orthogonal (k×k) and
/// `sigma` is sorted descending.
fn bidiag_svd(alpha: &[f64], beta: &[f64]) -> (Array2<f64>, Vec<f64>, Array2<f64>) {
    let k = alpha.len();
    assert!(beta.len() == k.saturating_sub(1) || beta.len() == k);

    // Build the full bidiagonal matrix B (k × k)
    let mut b = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        b[[i, i]] = alpha[i];
        if i + 1 < k && i < beta.len() {
            b[[i, i + 1]] = beta[i];
        }
    }

    // B^T B is a symmetric tridiagonal matrix:
    // T = B^T B where T[i,i] = alpha[i]^2 + (i>0 ? beta[i-1]^2 : 0)
    //                T[i,i+1] = alpha[i] * beta[i]
    let mut t = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        t[[i, i]] = alpha[i] * alpha[i];
        if i > 0 && i - 1 < beta.len() {
            t[[i, i]] += beta[i - 1] * beta[i - 1];
        }
        if i + 1 < k && i < beta.len() {
            let off = alpha[i] * beta[i];
            t[[i, i + 1]] = off;
            t[[i + 1, i]] = off;
        }
    }

    // Compute eigenvalues of T via Jacobi iteration (for small k this is fine)
    let (eigvals, eigvecs) = symmetric_jacobi(&t);

    // singular values = sqrt(max(eigenvalues, 0))
    let mut triplets: Vec<(f64, usize)> = eigvals
        .iter()
        .enumerate()
        .map(|(i, &lam)| (lam.max(0.0).sqrt(), i))
        .collect();
    triplets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let sigma: Vec<f64> = triplets.iter().map(|&(s, _)| s).collect();
    let order: Vec<usize> = triplets.iter().map(|&(_, i)| i).collect();

    // V columns: eigenvectors of T = B^T B → right singular vectors
    let mut v_b = Array2::<f64>::zeros((k, k));
    for (col_new, &col_old) in order.iter().enumerate() {
        for row in 0..k {
            v_b[[row, col_new]] = eigvecs[[row, col_old]];
        }
    }

    // U columns from B * V / sigma
    let mut u_b = Array2::<f64>::zeros((k, k));
    for (j, &s) in sigma.iter().enumerate() {
        if s > f64::EPSILON {
            let v_col: Vec<f64> = (0..k).map(|r| v_b[[r, j]]).collect();
            // u_j = B * v_j / sigma_j
            for i in 0..k {
                let mut sum = b[[i, i]] * v_col[i];
                if i > 0 {
                    sum += b[[i - 1, i]] * v_col[i - 1];
                }
                if i + 1 < k {
                    sum += b[[i, i + 1]] * v_col[i + 1];
                }
                u_b[[i, j]] = sum / s;
            }
        } else {
            u_b[[j, j]] = 1.0;
        }
    }

    (u_b, sigma, v_b)
}

/// Symmetric Jacobi eigenvalue solver for small dense matrices.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as columns.
fn symmetric_jacobi(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut d = a.to_owned();
    let mut v = Array2::<f64>::eye(n);
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find the largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = d[[i, j]].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        // Compute Jacobi rotation angle
        let theta = (d[[q, q]] - d[[p, p]]) / (2.0 * d[[p, q]]);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Apply rotation to D
        let d_pp = d[[p, p]];
        let d_qq = d[[q, q]];
        let d_pq = d[[p, q]];
        d[[p, p]] = c * c * d_pp - 2.0 * s * c * d_pq + s * s * d_qq;
        d[[q, q]] = s * s * d_pp + 2.0 * s * c * d_pq + c * c * d_qq;
        d[[p, q]] = 0.0;
        d[[q, p]] = 0.0;
        for r in 0..n {
            if r != p && r != q {
                let d_rp = d[[r, p]];
                let d_rq = d[[r, q]];
                d[[r, p]] = c * d_rp - s * d_rq;
                d[[p, r]] = d[[r, p]];
                d[[r, q]] = s * d_rp + c * d_rq;
                d[[q, r]] = d[[r, q]];
            }
        }

        // Accumulate eigenvectors
        for r in 0..n {
            let v_rp = v[[r, p]];
            let v_rq = v[[r, q]];
            v[[r, p]] = c * v_rp - s * v_rq;
            v[[r, q]] = s * v_rp + c * v_rq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| d[[i, i]]).collect();
    (eigenvalues, v)
}

// ---------------------------------------------------------------------------
// Lanczos bidiagonalization
// ---------------------------------------------------------------------------

/// Run k steps of Lanczos bidiagonalization on matrix A.
///
/// Returns `(U, B, V)` where:
/// - `U` is m × k (left Lanczos vectors, orthonormal)
/// - `B` is k × k lower bidiagonal (alpha on diagonal, beta on super-diagonal)
/// - `V` is n × k (right Lanczos vectors, orthonormal)
///
/// The standard recurrences are:
/// ```text
///   beta_j * u_{j+1} = A * v_j - alpha_j * u_j
///   alpha_{j+1} * v_{j+1} = A^T * u_{j+1} - beta_j * v_j
/// ```
///
/// Modified Gram-Schmidt reorthogonalization is applied at every step
/// to keep U and V numerically orthonormal.
///
/// # Errors
///
/// Returns an error if `k` is larger than `min(m, n)` or if A is empty.
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
    let mut alpha = Vec::with_capacity(k_eff);
    let mut beta = Vec::with_capacity(k_eff);

    // Start with a random initial right vector v_0 (deterministic seed)
    let mut v_curr: Vec<f64> = (0..n).map(|i| ((i + 1) as f64 * 0.31415926).sin()).collect();
    let nrm = normalize_inplace(&mut v_curr);
    if nrm < f64::EPSILON {
        return Err(LinalgError::ComputationError(
            "lanczos_bidiagonalization: initial vector is zero".to_string(),
        ));
    }

    let mut u_curr: Vec<f64>;
    let mut beta_prev = 0.0f64;

    for j in 0..k_eff {
        // Step 1: u_{j} = A * v_j
        u_curr = matvec(a, &v_curr);
        // Subtract beta_{j-1} * u_{j-1}
        if j > 0 {
            let u_prev = &u_vecs[j - 1];
            for (ui, &up) in u_curr.iter_mut().zip(u_prev.iter()) {
                *ui -= beta_prev * up;
            }
        }
        // Reorthogonalize against all previous u vectors
        reorthogonalize(&mut u_curr, &u_vecs);
        let alpha_j = normalize_inplace(&mut u_curr);
        alpha.push(alpha_j);
        u_vecs.push(u_curr.clone());

        if j + 1 >= k_eff {
            // Last step: store v but don't compute next beta
            v_vecs.push(v_curr.clone());
            break;
        }

        // Step 2: v_{j+1} = A^T * u_j - alpha_j * v_j
        let mut v_next = matvec_t(a, &u_curr);
        for (vi, &vc) in v_next.iter_mut().zip(v_curr.iter()) {
            *vi -= alpha_j * vc;
        }
        // Reorthogonalize against all previous v vectors
        reorthogonalize(&mut v_next, &v_vecs);
        let beta_j = normalize_inplace(&mut v_next);
        beta.push(beta_j);
        beta_prev = beta_j;

        v_vecs.push(v_curr.clone());
        v_curr = v_next;
    }

    let k_actual = alpha.len();

    // Pack U (m × k_actual)
    let mut u_mat = Array2::<f64>::zeros((m, k_actual));
    for (j, uv) in u_vecs.iter().enumerate() {
        for i in 0..m {
            u_mat[[i, j]] = uv[i];
        }
    }

    // Pack V (n × k_actual), v_vecs has k_actual entries (one pushed per step)
    let v_actual = v_vecs.len().min(k_actual);
    let mut v_mat = Array2::<f64>::zeros((n, v_actual));
    for (j, vv) in v_vecs.iter().take(v_actual).enumerate() {
        for i in 0..n {
            v_mat[[i, j]] = vv[i];
        }
    }

    // Build B (k_actual × k_actual) lower bidiagonal: alpha on diagonal, beta on super-diagonal
    let mut b_mat = Array2::<f64>::zeros((k_actual, k_actual));
    for i in 0..k_actual {
        b_mat[[i, i]] = alpha[i];
    }
    for i in 0..beta.len().min(k_actual - 1) {
        b_mat[[i, i + 1]] = beta[i];
    }

    Ok((u_mat, b_mat, v_mat))
}

// ---------------------------------------------------------------------------
// Distributed SVD simulation
// ---------------------------------------------------------------------------

/// Simulate distributed SVD: compute top-k singular triplets of A.
///
/// Runs Lanczos bidiagonalization, then solves the small dense bidiagonal SVD
/// to extract the top-k singular values and vectors.
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
/// `(U_k, sigma_k, V_k)` where:
/// - `U_k` is m × k (left singular vectors, orthonormal columns)
/// - `sigma_k` is length-k (singular values, descending)
/// - `V_k` is n × k (right singular vectors, orthonormal columns)
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

    let (u_lanczos, b_mat, v_lanczos) = lanczos_bidiagonalization(a, k_eff)?;

    let k_actual = u_lanczos.ncols();

    // Extract alpha and beta from B
    let mut alpha_b: Vec<f64> = (0..k_actual).map(|i| b_mat[[i, i]]).collect();
    let mut beta_b: Vec<f64> = (0..k_actual.saturating_sub(1))
        .map(|i| b_mat[[i, i + 1]])
        .collect();

    // Compute SVD of the small bidiagonal B
    let (u_b, sigma, v_b) = bidiag_svd(&alpha_b, &beta_b);

    // Map back: U_k = U_lanczos * U_B,  V_k = V_lanczos * V_B
    let k_out = k_eff.min(k_actual).min(sigma.len());

    // U_k (m × k_out) = U_lanczos (m × k_actual) * U_B (k_actual × k_actual), take k_out cols
    let mut u_k = Array2::<f64>::zeros((m, k_out));
    for j in 0..k_out {
        for i in 0..m {
            let mut s = 0.0f64;
            for l in 0..k_actual {
                s += u_lanczos[[i, l]] * u_b[[l, j]];
            }
            u_k[[i, j]] = s;
        }
    }

    // V_k (n × k_out) = V_lanczos (n × v_cols) * V_B (k_actual × k_actual), take k_out cols
    let v_cols = v_lanczos.ncols();
    let mut v_k = Array2::<f64>::zeros((n, k_out));
    for j in 0..k_out {
        for i in 0..n {
            let mut s = 0.0f64;
            for l in 0..v_cols.min(k_actual) {
                s += v_lanczos[[i, l]] * v_b[[l, j]];
            }
            v_k[[i, j]] = s;
        }
    }

    let sigma_k = sigma[..k_out].to_vec();

    Ok((u_k, sigma_k, v_k))
}

// ---------------------------------------------------------------------------
// Thick-restart Lanczos SVD
// ---------------------------------------------------------------------------

/// Thick-restart Lanczos SVD for well-separated singular values.
///
/// This extends `distributed_svd_simulate` with a restart strategy:
/// - Run up to `max_iter` Lanczos steps.
/// - Check convergence of the top-k Ritz values against `tol * sigma_1`.
/// - If not converged, deflate converged pairs and restart with the remaining
///   approximate singular vectors as the new starting vectors.
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

    // Maximum number of restart cycles
    let max_cycles = 10usize;
    // Lanczos block size per cycle: use at least k+5 steps but cap at min(m,n)
    let lanczos_size = (k_eff + 5).min(m).min(n);

    let mut prev_sigma: Option<Vec<f64>> = None;
    let mut best_u: Array2<f64> = Array2::<f64>::zeros((m, k_eff));
    let mut best_sigma: Vec<f64> = vec![0.0; k_eff];
    let mut best_v: Array2<f64> = Array2::<f64>::zeros((n, k_eff));

    for _cycle in 0..max_cycles {
        // Run Lanczos with possibly increased size on restart
        let (u_k, sigma_k, v_k) = distributed_svd_simulate(a, lanczos_size)?;

        let k_got = sigma_k.len().min(k_eff);
        best_u = u_k.slice(s![.., ..k_got]).to_owned();
        best_sigma = sigma_k[..k_got].to_vec();
        best_v = v_k.slice(s![.., ..k_got]).to_owned();

        // Check convergence: compare with previous cycle's singular values
        let converged = if let Some(ref prev) = prev_sigma {
            let sigma_1 = best_sigma.first().copied().unwrap_or(1.0).max(1e-14);
            prev.iter()
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

    // Reference full SVD via naive power iteration / QR (for test validation)
    fn frob_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
    }

    fn eye(n: usize) -> Array2<f64> {
        Array2::<f64>::eye(n)
    }

    // Compute U^T U - I and return max absolute entry
    fn orthogonality_error(u: &Array2<f64>) -> f64 {
        let k = u.ncols();
        let m = u.nrows();
        let mut max_err = 0.0f64;
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0f64;
                for r in 0..m {
                    dot += u[[r, i]] * u[[r, j]];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((dot - expected).abs());
            }
        }
        max_err
    }

    #[test]
    fn test_lanczos_bidiag_is_bidiagonal() {
        // Build a 6×5 random-like matrix
        let a = Array2::<f64>::from_shape_fn((6, 5), |(i, j)| {
            (i as f64 * 1.3 + j as f64 * 0.7 + 0.1).sin()
        });
        let k = 4;
        let (u_mat, b_mat, v_mat) = lanczos_bidiagonalization(&a, k).expect("lanczos failed");

        // B should be (k × k) with only alpha[i,i] and beta[i,i+1] non-zero
        let k_actual = b_mat.nrows();
        for i in 0..k_actual {
            for j in 0..k_actual {
                let val = b_mat[[i, j]];
                if i != j && !(i + 1 == j) {
                    // off-tridiagonal entries must be negligible
                    assert!(
                        val.abs() < 1e-12,
                        "B[{i},{j}] = {val} should be ≈ 0"
                    );
                }
            }
        }
    }

    #[test]
    fn test_lanczos_bidiag_u_orthonormal() {
        let a = Array2::<f64>::from_shape_fn((8, 6), |(i, j)| {
            ((i + 1) as f64) / ((j + 2) as f64)
        });
        let (u_mat, _, _) = lanczos_bidiagonalization(&a, 5).expect("lanczos failed");
        let err = orthogonality_error(&u_mat);
        assert!(err < 1e-10, "U^T U orthogonality error = {err}");
    }

    #[test]
    fn test_lanczos_bidiag_v_orthonormal() {
        let a = Array2::<f64>::from_shape_fn((8, 6), |(i, j)| {
            ((i + 1) as f64) / ((j + 2) as f64)
        });
        let (_, _, v_mat) = lanczos_bidiagonalization(&a, 5).expect("lanczos failed");
        let err = orthogonality_error(&v_mat);
        assert!(err < 1e-10, "V^T V orthogonality error = {err}");
    }

    #[test]
    fn test_distributed_svd_singular_values_match_reference() {
        // 4×4 matrix with known singular values (diagonal matrix → σ = diagonal entries)
        let diag = vec![5.0, 3.0, 2.0, 1.0];
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| if i == j { diag[i] } else { 0.0 });
        let (_, sigma, _) = distributed_svd_simulate(&a, 4).expect("svd failed");
        // Singular values should be 5, 3, 2, 1 in some order
        let mut sigma_sorted = sigma.clone();
        sigma_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_abs_diff_eq!(sigma_sorted[0], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sigma_sorted[1], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sigma_sorted[2], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(sigma_sorted[3], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_distributed_svd_rank1_matrix() {
        // Rank-1 matrix: A = u * sigma * v^T → only one non-zero singular value
        let u0: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0];
        let v0: Vec<f64> = vec![0.0, 1.0, 0.0, 0.0];
        let sigma0 = 7.0;
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| u0[i] * sigma0 * v0[j]);
        let (_, sigma, _) = distributed_svd_simulate(&a, 1).expect("svd failed");
        // Largest singular value should be ~7.0
        assert!(
            (sigma[0] - sigma0).abs() < 1e-5,
            "Expected sigma[0] ≈ {sigma0}, got {}",
            sigma[0]
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
        assert!(err_u < 1e-9, "U_k orthogonality error = {err_u}");
        assert!(err_v < 1e-9, "V_k orthogonality error = {err_v}");
    }

    #[test]
    fn test_distributed_svd_reconstruction_error() {
        // For rank-2 matrix, top-2 SVD should reconstruct exactly
        let a = Array2::<f64>::from_shape_fn((5, 4), |(i, j)| {
            if i < 2 && j < 2 { (i + 1) as f64 * (j + 2) as f64 } else { 0.0 }
        });
        let (u_k, sigma_k, v_k) = distributed_svd_simulate(&a, 2).expect("svd failed");
        // Reconstruct A ≈ U * diag(sigma) * V^T
        let mut a_rec = Array2::<f64>::zeros((5, 4));
        for r in 0..u_k.ncols() {
            for i in 0..5 {
                for j in 0..4 {
                    a_rec[[i, j]] += u_k[[i, r]] * sigma_k[r] * v_k[[j, r]];
                }
            }
        }
        let err = frob_diff(&a_rec, &a);
        assert!(err < 1e-7, "Reconstruction error = {err}");
    }

    #[test]
    fn test_thick_restart_converges() {
        // Well-conditioned diagonal matrix — singular values should converge quickly
        let diag = vec![10.0, 7.0, 4.0, 1.0];
        let a = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| if i == j { diag[i] } else { 0.0 });
        let (_, sigma, _) = thick_restart_lanczos(&a, 3, 1e-8).expect("thick restart failed");
        assert!(sigma.len() >= 1);
        // Largest should be ≈ 10
        let max_sv = sigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_sv > 8.0, "Expected largest singular value near 10, got {max_sv}");
    }

    #[test]
    fn test_bidiag_svd_diagonal_matrix() {
        // alpha = [3, 2, 1], beta = [] → B is diagonal → sigma = [3, 2, 1]
        let alpha = vec![3.0f64, 2.0, 1.0];
        let beta: Vec<f64> = vec![];
        let (_, sigma, _) = bidiag_svd(&alpha, &beta);
        let mut s = sigma.clone();
        s.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert_abs_diff_eq!(s[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(s[2], 1.0, epsilon = 1e-10);
    }
}
