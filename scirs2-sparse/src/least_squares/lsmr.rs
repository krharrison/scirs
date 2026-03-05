//! LSMR — Fong & Saunders (2011) iterative least-squares solver
//!
//! Solves
//!   min ||Ax - b||_2          (consistent or over-determined)
//! or the damped variant
//!   min ||Ax - b||_2^2 + λ^2 ||x||_2^2
//!
//! using Golub-Kahan bidiagonalization with a MINRES-like update for x.
//! LSMR generally gives a smoother monotonic decrease of `||A^T r||` compared
//! to LSQR, making it preferable for ill-conditioned problems.
//!
//! # References
//!
//! - Fong, D. C.-L. & Saunders, M. A. (2011). LSMR: An iterative algorithm
//!   for sparse least-squares problems. *SIAM J. Sci. Comput.* 33(5), 2950–2971.

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;

/// Configuration for the LSMR solver.
#[derive(Debug, Clone)]
pub struct LSMRConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Stopping tolerance for `||A^T r|| / (||A|| ||r||)`.
    pub atol: f64,
    /// Stopping tolerance for `||r|| / ||b||`.
    pub btol: f64,
    /// Condition number limit.
    pub conlim: f64,
    /// Damping parameter λ for the regularised problem.
    pub damp: f64,
    /// Regularisation parameter (added to damping, kept separate for clarity).
    pub lambda: f64,
}

impl Default for LSMRConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            conlim: 1e8,
            damp: 0.0,
            lambda: 0.0,
        }
    }
}

/// Result returned by the LSMR solver.
#[derive(Debug, Clone)]
pub struct LSMRResult {
    /// The computed solution vector (length n).
    pub x: Vec<f64>,
    /// Norm of the final residual `||b - Ax||`.
    pub r_norm: f64,
    /// Norm of the solution `||x||`.
    pub x_norm: f64,
    /// Norm of the normal residual `||A^T r||`.
    pub n_norm: f64,
    /// Number of iterations performed.
    pub iters: usize,
    /// Convergence flag (0 = not converged, 1–7 = various stop conditions).
    pub flag: usize,
}

// ---------------------------------------------------------------------------
// Stable Givens rotation (sym_ortho)
// ---------------------------------------------------------------------------

/// Compute stable Givens rotation (c, s, r) such that
///   [ c   s ] [a]   [r]
///   [-s   c ] [b] = [0]
#[inline]
fn sym_ortho(a: f64, b: f64) -> (f64, f64, f64) {
    if b == 0.0 {
        let c = if a >= 0.0 { 1.0 } else { -1.0 };
        (c, 0.0, a.abs())
    } else if a == 0.0 {
        let s = if b >= 0.0 { 1.0 } else { -1.0 };
        (0.0, s, b.abs())
    } else if b.abs() > a.abs() {
        let tau = a / b;
        let s_sign = if b >= 0.0 { 1.0 } else { -1.0 };
        let s = s_sign / (1.0 + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c_sign = if a >= 0.0 { 1.0 } else { -1.0 };
        let c = c_sign / (1.0 + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

/// Compute 2-norm of a slice.
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

/// LSMR iterative solver using arbitrary matrix-vector product closures.
///
/// Implements the Fong & Saunders (2011) LSMR algorithm based on
/// Golub-Kahan bidiagonalization with MINRES-like recurrence for x.
///
/// # Arguments
///
/// * `matvec`  – Closure computing `A * x` (length n → m).
/// * `rmatvec` – Closure computing `A^T * y` (length m → n).
/// * `b`       – Right-hand side vector (length m).
/// * `m`       – Number of rows.
/// * `n`       – Number of columns.
/// * `config`  – Solver configuration.
pub fn lsmr<F, G>(
    matvec: F,
    rmatvec: G,
    b: &[f64],
    m: usize,
    n: usize,
    config: &LSMRConfig,
) -> SparseResult<LSMRResult>
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

    let damp = config.damp + config.lambda;
    let atol = config.atol;
    let btol = config.btol;
    let conlim = config.conlim;
    let max_iter = config.max_iter.max(1);

    // --- Initialise Golub-Kahan bidiagonalisation ---
    let beta1 = vec_norm(b);
    if beta1 == 0.0 {
        return Ok(LSMRResult {
            x: vec![0.0; n],
            r_norm: 0.0,
            x_norm: 0.0,
            n_norm: 0.0,
            iters: 0,
            flag: 1,
        });
    }

    let mut u: Vec<f64> = b.iter().map(|&bi| bi / beta1).collect();
    #[allow(unused_assignments)]
    let mut beta = 0.0f64;  // overwritten in bidiagonalisation loop

    let mut v = rmatvec(&u);
    let mut alpha = vec_norm(&v);
    if alpha == 0.0 {
        return Ok(LSMRResult {
            x: vec![0.0; n],
            r_norm: beta1,
            x_norm: 0.0,
            n_norm: 0.0,
            iters: 0,
            flag: 2,
        });
    }
    scale_vec(&mut v, 1.0 / alpha);

    // --- LSMR recurrence state ---
    let mut x = vec![0.0f64; n];
    let mut h = v.clone();
    let mut h_bar = vec![0.0f64; n];

    // Givens rotation state
    let mut alpha_bar = alpha;
    let mut zeta_bar = alpha * beta1;
    let mut rho = 1.0f64;
    let mut rho_bar = 1.0f64;
    let mut c_bar = 1.0f64;
    let mut s_bar = 0.0f64;

    // Running norm estimates
    let mut a_norm_sq = alpha * alpha;
    let mut max_rho_bar = 0.0f64;
    let mut min_rho_bar = f64::MAX;

    let mut iters = 0usize;
    let mut flag = 0usize;

    for iter in 0..max_iter {
        iters = iter + 1;

        // --- Bidiagonalisation step ---
        let mut u_new = matvec(&v);
        saxpy(&mut u_new, -alpha, &u);
        beta = vec_norm(&u_new);
        if beta > 0.0 {
            scale_vec(&mut u_new, 1.0 / beta);
        }
        u = u_new;

        let mut v_new = rmatvec(&u);
        saxpy(&mut v_new, -beta, &v);
        alpha = vec_norm(&v_new);
        if alpha > 0.0 {
            scale_vec(&mut v_new, 1.0 / alpha);
        }
        v = v_new;

        // --- Construct and apply rotations ---

        // Rotation to handle damping (if any)
        let (c_hat, s_hat, alpha_hat) = sym_ortho(alpha_bar, damp);

        // Rotation P_{k-1}
        let rho_old = rho;
        let (c, s, rho_new) = sym_ortho(alpha_hat, beta);
        rho = rho_new;

        let theta_new = s * alpha;
        alpha_bar = c * alpha;

        // Rotation \bar{P}_k (Fong & Saunders eq. (9.5))
        let theta_bar = s_bar * rho;
        let rho_bar_old = rho_bar;
        let (c_bar_new, s_bar_new, rho_bar_new) = sym_ortho(c_bar * rho, theta_new);
        rho_bar = rho_bar_new;
        c_bar = c_bar_new;
        s_bar = s_bar_new;

        // Update h, h_bar, x
        let zeta = c_hat * zeta_bar;
        let zeta_bar_new = -s_hat * zeta_bar;
        zeta_bar = zeta_bar_new;

        // h_bar_k = h_k - (theta_bar * rho / (rho_old * rho_bar_old)) * h_bar_{k-1}
        let factor = theta_bar * rho / (rho_old * rho_bar_old.max(1e-300));
        {
            let mut h_bar_new = h.clone();
            saxpy(&mut h_bar_new, -factor, &h_bar);
            h_bar = h_bar_new;
        }

        // x_k = x_{k-1} + (zeta / (rho * rho_bar)) * h_bar_k
        let x_step = zeta / (rho * rho_bar.max(1e-300));
        saxpy(&mut x, x_step, &h_bar);

        // h_{k+1} = v_{k+1} - (theta_new / rho) * h_k
        {
            let theta_rho = theta_new / rho.max(1e-300);
            let mut h_new = v.clone();
            saxpy(&mut h_new, -theta_rho, &h);
            h = h_new;
        }

        // --- Running estimates ---
        a_norm_sq += alpha * alpha + beta * beta + damp * damp;
        max_rho_bar = max_rho_bar.max(rho_bar_old.abs());
        if iter > 0 {
            min_rho_bar = min_rho_bar.min(rho_bar_old.abs());
        }
        let a_norm = a_norm_sq.sqrt();
        let _a_cond = if min_rho_bar < f64::MAX && min_rho_bar > 0.0 {
            max_rho_bar / min_rho_bar
        } else {
            1.0
        };

        let x_norm_est = vec_norm(&x);
        let r_norm = zeta_bar.abs();
        let ar_norm = zeta_bar.abs() * alpha_bar.abs();

        // --- Stopping tests ---
        let b_norm = beta1;
        let test1 = r_norm / b_norm.max(1e-300);
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
        if _a_cond >= conlim {
            flag = 3;
            break;
        }
        if r_norm <= 1e-15 * (a_norm * x_norm_est + b_norm) {
            flag = 4;
            break;
        }
    }

    let r_norm = zeta_bar.abs();
    let n_norm_final = zeta_bar.abs() * alpha_bar.abs();
    let x_norm = vec_norm(&x);

    Ok(LSMRResult {
        x,
        r_norm,
        x_norm,
        n_norm: n_norm_final,
        iters,
        flag,
    })
}

/// Convenience wrapper using a `CsrMatrix<f64>` directly.
pub fn lsmr_sparse<F>(
    a: &CsrMatrix<F>,
    b: &[f64],
    config: &LSMRConfig,
) -> SparseResult<LSMRResult>
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

    lsmr(matvec, rmatvec, b, m, n, config)
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
    fn test_lsmr_square_system() {
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0]; // A * [1,1,1]

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSMRConfig {
            max_iter: 500,
            atol: 1e-10,
            btol: 1e-10,
            ..Default::default()
        };
        let result = lsmr(mv, rmv, &b, m, n, &config).expect("LSMR failed");

        assert!(result.iters > 0);
        assert!(result.r_norm < 1e-5, "r_norm = {}", result.r_norm);
        assert_close(result.x[0], 1.0, 1e-4);
        assert_close(result.x[1], 1.0, 1e-4);
        assert_close(result.x[2], 1.0, 1e-4);
    }

    #[test]
    fn test_lsmr_zero_rhs() {
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 1, 2, 3];
        let indices = vec![0usize, 1, 2];
        let data = vec![2.0, 3.0, 4.0];
        let b = vec![0.0f64; m];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSMRConfig::default();
        let result = lsmr(mv, rmv, &b, m, n, &config).expect("LSMR zero rhs failed");

        assert_eq!(result.r_norm, 0.0);
        assert!(result.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lsmr_overdetermined() {
        let m = 4usize;
        let n = 2usize;
        let indptr = vec![0usize, 1, 2, 4, 5];
        let indices = vec![0usize, 1, 0, 1, 0];
        let data = vec![1.0f64, 1.0, 1.0, 1.0, 2.0];
        let b = vec![1.0f64, 1.0, 2.0, 2.0];

        let (mv, rmv) = make_matvec(indptr, indices, data, m, n);
        let config = LSMRConfig {
            max_iter: 1000,
            atol: 1e-8,
            btol: 1e-8,
            ..Default::default()
        };
        let result = lsmr(mv, rmv, &b, m, n, &config).expect("LSMR overdetermined failed");

        assert_eq!(result.x.len(), n);
        assert_close(result.x[0], 1.0, 1e-3);
        assert_close(result.x[1], 1.0, 1e-3);
    }

    #[test]
    fn test_lsmr_damped() {
        let m = 3usize;
        let n = 3usize;
        let indptr = vec![0usize, 2, 5, 7];
        let indices = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0];

        let (mv_u, rmv_u) = make_matvec(indptr.clone(), indices.clone(), data.clone(), m, n);
        let (mv_d, rmv_d) = make_matvec(indptr, indices, data, m, n);

        let undamped = lsmr(mv_u, rmv_u, &b, m, n, &LSMRConfig { max_iter: 500, ..Default::default() })
            .expect("undamped");
        let damped = lsmr(mv_d, rmv_d, &b, m, n, &LSMRConfig { max_iter: 500, damp: 1.0, ..Default::default() })
            .expect("damped");

        let norm_u: f64 = undamped.x.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_d: f64 = damped.x.iter().map(|v| v * v).sum::<f64>().sqrt();

        assert!(
            norm_d <= norm_u + 1e-6,
            "damping should reduce solution norm: undamped={norm_u}, damped={norm_d}"
        );
    }

    #[test]
    fn test_lsmr_sparse_wrapper() {
        let rows_v = vec![0usize, 0, 1, 1, 1, 2, 2];
        let cols_v = vec![0usize, 1, 0, 1, 2, 1, 2];
        let data = vec![4.0f64, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];
        let b = vec![3.0f64, 2.0, 3.0];

        let a = CsrMatrix::new(data, rows_v, cols_v, (3, 3))
            .expect("CsrMatrix failed");
        let config = LSMRConfig {
            max_iter: 500,
            atol: 1e-10,
            btol: 1e-10,
            ..Default::default()
        };
        let result = lsmr_sparse(&a, &b, &config).expect("lsmr_sparse failed");

        assert!(result.r_norm < 1e-4, "r_norm = {}", result.r_norm);
    }
}
