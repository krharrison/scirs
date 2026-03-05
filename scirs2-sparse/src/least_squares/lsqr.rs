//! LSQR — Paige & Saunders (1982) iterative least-squares solver
//!
//! Solves
//!   min ||Ax - b||_2          (consistent or over-determined)
//! or the damped variant
//!   min ||Ax - b||_2^2 + damp^2 * ||x||_2^2
//!
//! using Lanczos bidiagonalization with Givens QR updates.
//!
//! # References
//!
//! - Paige, C. C. & Saunders, M. A. (1982). LSQR: An algorithm for sparse
//!   linear equations and sparse least squares. *ACM Trans. Math. Softw.*
//!   8(1), 43–71.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

/// Configuration for the LSQR solver.
#[derive(Debug, Clone)]
pub struct LSQRConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stopping tolerance: residual convergence (`||A^T r|| / (||A|| ||r||)` ≤ atol).
    pub atol: f64,
    /// Stopping tolerance: solution convergence (`||r|| / ||b||` ≤ btol).
    pub btol: f64,
    /// Condition number limit — stop if estimated cond(A) ≥ conlim.
    pub conlim: f64,
    /// Damping parameter λ for regularised problem `min ||Ax-b||^2 + λ^2 ||x||^2`.
    pub damp: f64,
}

impl Default for LSQRConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            conlim: 1e8,
            damp: 0.0,
        }
    }
}

/// Result returned by the LSQR solver.
#[derive(Debug, Clone)]
pub struct LSQRResult {
    /// The computed solution vector (length n).
    pub x: Vec<f64>,
    /// Norm of the final residual `||b - Ax||`.
    pub r_norm: f64,
    /// Norm of the solution `||x||`.
    pub x_norm: f64,
    /// Estimate of `||A||_F`.
    pub a_norm: f64,
    /// Estimate of `cond(A)`.
    pub a_cond: f64,
    /// Norm of `A^T r` (used as a stopping criterion).
    pub ar_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// Convergence flag (0 = not converged, 1–7 = various stop conditions).
    pub flag: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the stable 2-norm of a slice.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Scale a slice in-place by `s`.
#[inline]
fn scale_vec(v: &mut [f64], s: f64) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// y += alpha * x  (saxpy).
#[inline]
fn saxpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/// LSQR iterative solver using arbitrary matrix-vector product closures.
///
/// # Arguments
///
/// * `matvec`  – Closure that computes `A * x` for a given input `x` (length n → m).
/// * `rmatvec` – Closure that computes `A^T * y` for a given input `y` (length m → n).
/// * `b`       – Right-hand side vector (length m).
/// * `m`       – Number of rows of A.
/// * `n`       – Number of columns of A.
/// * `config`  – Solver configuration.
///
/// # Returns
///
/// An `LSQRResult` on success, or a `SparseError` on failure.
pub fn lsqr<F, G>(
    matvec: F,
    rmatvec: G,
    b: &[f64],
    m: usize,
    n: usize,
    config: &LSQRConfig,
) -> SparseResult<LSQRResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if b.len() != m {
        return Err(SparseError::DimensionMismatch {
            expected: m,
            found: b.len(),
        });
    }

    let damp = config.damp;
    let atol = config.atol;
    let btol = config.btol;
    let conlim = config.conlim;
    let max_iter = config.max_iter.max(1);

    // --- Initialise ---
    let mut x = vec![0.0f64; n];

    // u = b / beta1
    let beta1 = vec_norm(b);
    if beta1 == 0.0 {
        return Ok(LSQRResult {
            x,
            r_norm: 0.0,
            x_norm: 0.0,
            a_norm: 0.0,
            a_cond: 1.0,
            ar_norm: 0.0,
            iters: 0,
            flag: 1,
        });
    }
    let mut u: Vec<f64> = b.iter().map(|&bi| bi / beta1).collect();
    #[allow(unused_assignments)]
    let mut beta = beta1;

    // v = A^T u / alpha1
    let mut v = rmatvec(&u);
    let alpha1 = vec_norm(&v);
    if alpha1 == 0.0 {
        return Ok(LSQRResult {
            x,
            r_norm: beta1,
            x_norm: 0.0,
            a_norm: 0.0,
            a_cond: 1.0,
            ar_norm: 0.0,
            iters: 0,
            flag: 2,
        });
    }
    let mut alpha = alpha1;
    scale_vec(&mut v, 1.0 / alpha);

    // Bidiagonalisation state
    let mut w = v.clone();

    // QR factors on bidiagonal
    let mut phibar = beta1;
    let mut rhobar = alpha1;
    let mut cs2 = -1.0f64;
    let mut sn2 = 0.0f64;
    let mut z = 0.0f64;

    // Running norms / condition estimates
    let mut a_norm_sq = 0.0f64;
    let mut d_norm_sq = 0.0f64;
    let mut x_norm = 0.0f64;
    let mut _xx_norm_sq = 0.0f64;
    #[allow(unused_assignments)]
    let mut res1 = beta1 * beta1;
    let mut r_norm = beta1;
    let mut ar_norm = alpha1 * beta1;
    let mut a_norm = 0.0f64;
    let mut a_cond = 1.0f64;

    let mut iters = 0usize;
    let mut flag = 0usize;

    for iter in 0..max_iter {
        iters = iter + 1;

        // --- Bidiagonalisation step ---
        // u = A v - alpha * u
        let mut u_new = matvec(&v);
        saxpy(&mut u_new, -alpha, &u);
        beta = vec_norm(&u_new);

        if beta > 0.0 {
            scale_vec(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        // Accumulate ||A||^2 = sum of alpha^2 + beta^2
        a_norm_sq += alpha * alpha + beta * beta + damp * damp;

        // v = A^T u - beta * v
        let mut v_new = rmatvec(&u);
        saxpy(&mut v_new, -beta, &v);
        alpha = vec_norm(&v_new);
        if alpha > 0.0 {
            scale_vec(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // --- QR step on (alpha, beta, damp) ---
        // Handle damping: construct an extended bidiagonal system.
        let (rho, cs, sn);
        if damp == 0.0 {
            rho = rhobar.hypot(beta);
            cs = rhobar / rho;
            sn = beta / rho;
        } else {
            // First rotation to zero out the damping row.
            let rho_bar_d = rhobar.hypot(damp);
            let cs_bar = rhobar / rho_bar_d;
            let sn_bar = damp / rho_bar_d;
            let phi_bar_new = cs_bar * phibar;
            let gamma = sn_bar * phibar;
            rho = rho_bar_d.hypot(beta);
            cs = rho_bar_d / rho;
            sn = beta / rho;
            phibar = cs * phi_bar_new;
            // Apply second rotation to z estimate
            let z_new = (gamma.hypot(phi_bar_new)) / rho;
            z = z_new;
        };

        let theta = sn * alpha;
        let rhobar_new = -cs * alpha;
        let phi = cs * phibar;
        phibar = sn * phibar;

        rhobar = rhobar_new;

        // --- Update x and w ---
        let phi_rho = phi / rho;
        saxpy(&mut x, phi_rho, &w);

        let theta_rho = theta / rho;
        let mut w_new = v.clone();
        saxpy(&mut w_new, -theta_rho, &w);
        w = w_new;

        // --- Running estimates ---
        // x_norm
        _xx_norm_sq += (phi / rho) * (phi / rho) * d_norm_sq;
        d_norm_sq += 1.0 / (rho * rho);
        x_norm = vec_norm(&x);

        a_norm = a_norm_sq.sqrt();
        a_cond = if a_norm > 0.0 {
            a_norm * d_norm_sq.sqrt()
        } else {
            1.0
        };

        // Residual norm update
        let phi_bar_2 = phibar * phibar;
        res1 = phi_bar_2;
        r_norm = res1.sqrt();
        ar_norm = phibar * alpha * cs.abs();

        // --- Stopping tests ---
        let eps = 1e-15_f64;
        let b_norm = beta1;

        // Test 1: ||A^T r|| / (||A|| ||r||) ≤ atol
        let test1 = r_norm / b_norm;
        let test2 = if a_norm > 0.0 && r_norm > 0.0 {
            ar_norm / (a_norm * r_norm)
        } else {
            0.0
        };

        if test2 <= atol {
            flag = 1;
            break;
        }
        if test1 <= btol {
            flag = 2;
            break;
        }
        if a_cond >= conlim {
            flag = 3;
            break;
        }
        if r_norm <= eps * (a_norm * x_norm + b_norm) {
            flag = 4;
            break;
        }
        if iters >= max_iter {
            flag = 7;
            break;
        }
        // Suppress unused variable warnings for z and cs2/sn2
        let _ = z;
        let _ = cs2;
        let _ = sn2;
        cs2 = cs;
        sn2 = sn;
    }

    // Suppress final unused warnings
    let _ = cs2;
    let _ = sn2;

    Ok(LSQRResult {
        x,
        r_norm,
        x_norm,
        a_norm,
        a_cond,
        ar_norm,
        iters,
        flag,
    })
}

/// Convenience wrapper that uses a `CsrMatrix<f64>` directly.
///
/// # Arguments
///
/// * `a`      – Sparse CSR coefficient matrix (m × n).
/// * `b`      – Right-hand side vector (length m).
/// * `config` – Solver configuration.
pub fn lsqr_sparse<F>(
    a: &CsrMatrix<F>,
    b: &[f64],
    config: &LSQRConfig,
) -> SparseResult<LSQRResult>
where
    F: Float + NumAssign + SparseElement + Debug + Sum + Into<f64> + Copy,
{
    let m = a.rows();
    let n = a.cols();

    // Build owned CSR arrays for the closures (avoid lifetime issues).
    let indptr = a.indptr.clone();
    let indices = a.indices.clone();
    let data_f64: Vec<f64> = a.data.iter().map(|&v| v.into()).collect();

    let indptr2 = indptr.clone();
    let indices2 = indices.clone();
    let data_f64_2 = data_f64.clone();

    let matvec = move |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; m];
        for i in 0..m {
            let mut acc = 0.0f64;
            for pos in indptr[i]..indptr[i + 1] {
                acc += data_f64[pos] * x[indices[pos]];
            }
            y[i] = acc;
        }
        y
    };

    let rmatvec = move |y: &[f64]| -> Vec<f64> {
        let mut x = vec![0.0f64; n];
        for i in 0..m {
            for pos in indptr2[i]..indptr2[i + 1] {
                x[indices2[pos]] += data_f64_2[pos] * y[i];
            }
        }
        x
    };

    lsqr(matvec, rmatvec, b, m, n, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected ~{b}, got {a} (diff {})",
            (a - b).abs()
        );
    }

    // Build a simple 3×3 tridiagonal system: A x = b, exact solution x = [1,1,1].
    fn build_square_system() -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, Vec<f64>) {
        let rows = vec![0usize, 0, 1, 1, 1, 2, 2];
        let cols = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b_vec = vec![3.0, 2.0, 3.0]; // A * [1,1,1]
        let (n, m) = (3, 3);

        // Convert to CSR
        let mut indptr = vec![0usize; m + 1];
        for &r in &rows {
            indptr[r + 1] += 1;
        }
        for i in 0..m {
            indptr[i + 1] += indptr[i];
        }
        let b_clone = b_vec.clone();
        (data, b_vec, rows.iter().map(|_| 0.0f64).collect(), n, m, b_clone)
    }

    /// Build a small matvec pair for a known matrix.
    fn make_matvec(
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        m: usize,
        n: usize,
    ) -> (
        impl Fn(&[f64]) -> Vec<f64>,
        impl Fn(&[f64]) -> Vec<f64>,
    ) {
        let indptr2 = indptr.clone();
        let indices2 = indices.clone();
        let data2 = data.clone();

        let mv = move |x: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; m];
            for i in 0..m {
                for pos in indptr[i]..indptr[i + 1] {
                    y[i] += data[pos] * x[indices[pos]];
                }
            }
            y
        };

        let rmv = move |y: &[f64]| -> Vec<f64> {
            let mut x = vec![0.0; n];
            for i in 0..m {
                for pos in indptr2[i]..indptr2[i + 1] {
                    x[indices2[pos]] += data2[pos] * y[i];
                }
            }
            x
        };

        (mv, rmv)
    }

    #[test]
    fn test_lsqr_square_system() {
        // 3×3 tridiagonal system Ax = b, exact solution x = [1,1,1]
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0]; // A * [1,1,1]

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSQRConfig {
            max_iter: 500,
            atol: 1e-10,
            btol: 1e-10,
            ..Default::default()
        };
        let result = lsqr(mv, rmv, &b, m, n, &config).expect("LSQR failed");

        assert!(result.iters > 0, "should need at least one iteration");
        assert!(result.r_norm < 1e-6, "residual should be small: {}", result.r_norm);
        assert_close(result.x[0], 1.0, 1e-5);
        assert_close(result.x[1], 1.0, 1e-5);
        assert_close(result.x[2], 1.0, 1e-5);
    }

    #[test]
    fn test_lsqr_zero_rhs() {
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 1, 2, 3];
        let indices = vec![0usize, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let b = vec![0.0f64; m];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSQRConfig::default();
        let result = lsqr(mv, rmv, &b, m, n, &config).expect("LSQR zero rhs failed");

        assert_eq!(result.r_norm, 0.0);
        assert!(result.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lsqr_overdetermined() {
        // 4×2 overdetermined system: A = [[1,0],[0,1],[1,1],[2,0]], b = [1,1,2,2]
        // Least-squares solution should be close to [1, 1]
        let m = 4usize;
        let n = 2usize;
        let indptr = vec![0usize, 1, 2, 4, 5];
        let indices = vec![0usize, 1, 0, 1, 0];
        let data = vec![1.0f64, 1.0, 1.0, 1.0, 2.0];
        let b = vec![1.0f64, 1.0, 2.0, 2.0];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSQRConfig {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            ..Default::default()
        };
        let result = lsqr(mv, rmv, &b, m, n, &config).expect("LSQR overdetermined failed");

        // Solution should satisfy normal equations approximately.
        assert!(result.x.len() == n);
        assert_close(result.x[0], 1.0, 1e-4);
        assert_close(result.x[1], 1.0, 1e-4);
    }

    #[test]
    fn test_lsqr_damped() {
        // Square system with damping — solution should be regularised (smaller norm).
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0];

        let (mv_undamped, rmv_undamped) = make_matvec(
            indptr.clone(), indices.clone(), data.clone(), m, n,
        );
        let (mv_damped, rmv_damped) = make_matvec(indptr, indices, data, m, n);

        let undamped = lsqr(
            mv_undamped, rmv_undamped, &b, m, n,
            &LSQRConfig { max_iter: 500, ..Default::default() },
        )
        .expect("undamped");

        let damped = lsqr(
            mv_damped, rmv_damped, &b, m, n,
            &LSQRConfig { max_iter: 500, damp: 1.0, ..Default::default() },
        )
        .expect("damped");

        let norm_undamped: f64 = undamped.x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_damped: f64 = damped.x.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            norm_damped <= norm_undamped + 1e-6,
            "damping should reduce or not increase solution norm: undamped={norm_undamped}, damped={norm_damped}"
        );
    }

    #[test]
    fn test_lsqr_sparse_wrapper() {
        let rows_vec = vec![0usize, 0, 1, 1, 1, 2, 2];
        let cols_vec = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0];

        let a = CsrMatrix::new(data, rows_vec, cols_vec, (3, 3))
            .expect("CsrMatrix construction failed");
        let config = LSQRConfig {
            max_iter: 500,
            atol: 1e-10,
            btol: 1e-10,
            ..Default::default()
        };
        let result = lsqr_sparse(&a, &b, &config).expect("lsqr_sparse failed");

        assert!(result.r_norm < 1e-5, "r_norm = {}", result.r_norm);
        assert_close(result.x[0], 1.0, 1e-4);
    }
}
