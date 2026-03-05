//! CGLS — Conjugate Gradient Least Squares
//!
//! Equivalent to running the Conjugate Gradient method on the normal equations
//!   A^T A x = A^T b
//! but implemented in a numerically stable form that avoids explicitly forming
//! the normal-equations matrix.
//!
//! Solves the (possibly over- or under-determined) least-squares problem
//!   min_{x} ||Ax - b||_2
//!
//! # References
//!
//! - Björck, Å. (1996). *Numerical Methods for Least Squares Problems*. SIAM.
//! - Hestenes, M. R. & Stiefel, E. (1952). Methods of conjugate gradients for
//!   solving linear systems. *J. Res. Natl. Bur. Stand.* 49(6), 409–436.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

/// Result returned by the CGLS solver.
#[derive(Debug, Clone)]
pub struct CGLSResult {
    /// The computed solution vector (length n).
    pub x: Vec<f64>,
    /// Number of iterations performed.
    pub iters: usize,
    /// Final normal-residual norm `||A^T r||` / `||A^T b||`.
    pub rel_norm: f64,
    /// Whether the solver converged within the specified tolerance.
    pub converged: bool,
}

/// Compute 2-norm of a slice.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

/// y += alpha * x.
#[inline]
fn saxpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ---------------------------------------------------------------------------
// Core CGLS algorithm
// ---------------------------------------------------------------------------

/// CGLS iterative solver using arbitrary matrix-vector product closures.
///
/// # Arguments
///
/// * `matvec`   – Closure computing `A * x` (length n → m).
/// * `rmatvec`  – Closure computing `A^T * y` (length m → n).
/// * `b`        – Right-hand side vector (length m).
/// * `m`        – Number of rows.
/// * `n`        – Number of columns.
/// * `max_iter` – Maximum number of iterations.
/// * `tol`      – Convergence tolerance on `||A^T r||` / `||A^T b||`.
///
/// # Returns
///
/// A `CGLSResult` on success, or a `SparseError`.
pub fn cgls<F, G>(
    matvec: F,
    rmatvec: G,
    b: &[f64],
    m: usize,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> SparseResult<CGLSResult>
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

    let max_iter = max_iter.max(1);

    // Initialise: x0 = 0, r = b - A*x0 = b
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec(); // r = b - A*x  (initially b)

    // s = A^T r
    let mut s = rmatvec(&r);
    let mut p = s.clone();

    let s_norm_sq = dot(&s, &s);
    if s_norm_sq == 0.0 {
        // A^T b = 0  ⟹  x = 0 is the least-norm solution.
        return Ok(CGLSResult {
            x,
            iters: 0,
            rel_norm: 0.0,
            converged: true,
        });
    }

    // Reference norm for relative stopping criterion.
    let atb_norm = s_norm_sq.sqrt();

    let mut gamma = s_norm_sq; // ||s||^2 = ||A^T r||^2

    let mut iters = 0usize;
    let mut converged = false;

    for iter in 0..max_iter {
        iters = iter + 1;

        // q = A p
        let q = matvec(&p);

        // alpha = gamma / ||q||^2
        let q_norm_sq = dot(&q, &q);
        if q_norm_sq == 0.0 {
            // A p = 0  ⟹  p is in the null space; algorithm stalls.
            break;
        }
        let alpha = gamma / q_norm_sq;

        // x = x + alpha * p
        saxpy(&mut x, alpha, &p);

        // r = r - alpha * q
        saxpy(&mut r, -alpha, &q);

        // s_new = A^T r
        let s_new = rmatvec(&r);

        let gamma_new = dot(&s_new, &s_new);

        // Relative stopping criterion
        let rel = gamma_new.sqrt() / atb_norm;
        if rel <= tol {
            // Final update: s = s_new  (not needed further)
            converged = true;
            break;
        }

        // beta = gamma_new / gamma
        let beta = gamma_new / gamma;

        // p = s_new + beta * p
        let mut p_new = s_new.clone();
        saxpy(&mut p_new, beta, &p);
        p = p_new;

        s = s_new;
        gamma = gamma_new;

        let _ = &s; // suppress unused
    }

    let rel_norm = gamma.sqrt() / atb_norm;

    Ok(CGLSResult {
        x,
        iters,
        rel_norm,
        converged,
    })
}

/// Convenience wrapper using a `CsrMatrix<f64>`.
pub fn cgls_sparse<F>(
    a: &CsrMatrix<F>,
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> SparseResult<CGLSResult>
where
    F: Float + NumAssign + SparseElement + Debug + Sum + Into<f64> + Copy,
{
    let m = a.rows();
    let n = a.cols();

    let indptr = a.indptr.clone();
    let indices = a.indices.clone();
    let data_f64: Vec<f64> = a.data.iter().map(|&v| v.into()).collect();

    let indptr2 = indptr.clone();
    let indices2 = indices.clone();
    let data_f64_2 = data_f64.clone();

    let matvec = move |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; m];
        for i in 0..m {
            for pos in indptr[i]..indptr[i + 1] {
                y[i] += data_f64[pos] * x[indices[pos]];
            }
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

    cgls(matvec, rmatvec, b, m, n, max_iter, tol)
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
    fn test_cgls_square_spd() {
        // 3×3 tridiagonal SPD system, exact solution x = [1,1,1].
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0]; // A * [1,1,1]

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let result = cgls(mv, rmv, &b, m, n, 500, 1e-10).expect("CGLS failed");

        assert!(result.converged, "CGLS should converge");
        assert_close(result.x[0], 1.0, 1e-4);
        assert_close(result.x[1], 1.0, 1e-4);
        assert_close(result.x[2], 1.0, 1e-4);
    }

    #[test]
    fn test_cgls_zero_rhs() {
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 1, 2, 3];
        let indices = vec![0usize, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let b = vec![0.0f64; m];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let result = cgls(mv, rmv, &b, m, n, 100, 1e-10).expect("CGLS zero rhs failed");

        assert!(result.converged);
        assert!(result.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cgls_overdetermined() {
        let m = 4usize;
        let n = 2usize;
        let indptr = vec![0usize, 1, 2, 4, 5];
        let indices = vec![0usize, 1, 0, 1, 0];
        let data = vec![1.0f64, 1.0, 1.0, 1.0, 2.0];
        let b = vec![1.0f64, 1.0, 2.0, 2.0];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let result = cgls(mv, rmv, &b, m, n, 1000, 1e-8).expect("CGLS overdetermined failed");

        assert_close(result.x[0], 1.0, 1e-4);
        assert_close(result.x[1], 1.0, 1e-4);
    }

    #[test]
    fn test_cgls_sparse_wrapper() {
        let rows_v = vec![0usize, 0, 1, 1, 1, 2, 2];
        let cols_v = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0];

        let a = CsrMatrix::new(data, rows_v, cols_v, (3, 3)).expect("CsrMatrix failed");
        let result = cgls_sparse(&a, &b, 500, 1e-10).expect("cgls_sparse failed");

        assert!(result.converged);
        assert_close(result.x[0], 1.0, 1e-4);
    }
}
