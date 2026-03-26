//! GPU-ready iterative sparse linear solvers.
//!
//! Implements Conjugate Gradient (CG) for symmetric positive definite systems
//! and BiCGSTAB for general non-symmetric systems.  Both use a BLAS-friendly
//! memory layout and an optional Jacobi (diagonal) preconditioner.  The
//! CPU computation paths mirror what GPU compute shaders would execute.

use crate::error::{SparseError, SparseResult};
use crate::gpu::construction::GpuCsrMatrix;
use crate::gpu::spmv::GpuSpMvBackend;

// ============================================================
// Configuration and result types
// ============================================================

/// GPU compute backend used by the solvers.
///
/// Mirrors [`GpuSpMvBackend`] but is kept independent so that solver
/// configuration and SpMV configuration can evolve separately.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuSolverBackend {
    /// CPU simulation (always available).
    #[default]
    Cpu,
    /// WebGPU via wgpu (feature-gated, not yet wired).
    WebGpu,
}

impl From<GpuSolverBackend> for GpuSpMvBackend {
    fn from(b: GpuSolverBackend) -> Self {
        match b {
            GpuSolverBackend::Cpu => GpuSpMvBackend::Cpu,
            GpuSolverBackend::WebGpu => GpuSpMvBackend::WebGpu,
        }
    }
}

/// Configuration for GPU iterative solvers.
#[derive(Debug, Clone)]
pub struct GpuSolverConfig {
    /// Maximum number of iterations (default 1000).
    pub max_iter: usize,
    /// Convergence tolerance on the residual 2-norm (default 1e-8).
    pub tol: f64,
    /// Whether to apply a Jacobi (diagonal) preconditioner (default true).
    pub precond: bool,
    /// Backend to use for matrix-vector products inside the solver.
    pub backend: GpuSolverBackend,
}

impl Default for GpuSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            precond: true,
            backend: GpuSolverBackend::Cpu,
        }
    }
}

/// Result returned by iterative solvers.
#[derive(Debug, Clone)]
pub struct SolverResult {
    /// Approximate solution vector.
    pub x: Vec<f64>,
    /// Euclidean norm of the final residual `b - A*x`.
    pub residual_norm: f64,
    /// Number of matrix-vector products performed.
    pub n_iter: usize,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
}

// ============================================================
// Helper: dense vector operations
// ============================================================

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// `z = alpha * x + beta * y`
#[inline]
fn axpby(alpha: f64, x: &[f64], beta: f64, y: &[f64], z: &mut [f64]) {
    for ((zi, &xi), &yi) in z.iter_mut().zip(x.iter()).zip(y.iter()) {
        *zi = alpha * xi + beta * yi;
    }
}

// ============================================================
// Jacobi preconditioner
// ============================================================

/// Extract diagonal of A (fallback to 1.0 for zero diagonals).
fn jacobi_diag(matrix: &GpuCsrMatrix) -> Vec<f64> {
    let n = matrix.n_rows;
    let mut diag = vec![1.0_f64; n];
    for row in 0..n {
        let start = matrix.row_ptr[row];
        let end = matrix.row_ptr[row + 1];
        for k in start..end {
            if matrix.col_idx[k] == row {
                let d = matrix.values[k];
                if d.abs() > f64::EPSILON {
                    diag[row] = d;
                }
            }
        }
    }
    diag
}

/// Apply the Jacobi preconditioner: `z[i] = r[i] / diag[i]`.
fn apply_jacobi(diag: &[f64], r: &[f64], z: &mut [f64]) {
    for ((zi, &ri), &di) in z.iter_mut().zip(r.iter()).zip(diag.iter()) {
        *zi = ri / di;
    }
}

// ============================================================
// Conjugate Gradient
// ============================================================

/// Solve the symmetric positive definite system `A * x = b` using the
/// Preconditioned Conjugate Gradient method.
///
/// An optional initial guess `x0` can be supplied.  When `config.precond`
/// is true a Jacobi (diagonal) preconditioner is used.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] when matrix and vector sizes are
/// incompatible.
pub fn cg_csr(
    matrix: &GpuCsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    config: &GpuSolverConfig,
) -> SparseResult<SolverResult> {
    let n = matrix.n_rows;
    if matrix.n_cols != n {
        return Err(SparseError::ComputationError(
            "CG requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }
    if let Some(x) = x0 {
        if x.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: x.len(),
            });
        }
    }

    let diag = if config.precond {
        jacobi_diag(matrix)
    } else {
        vec![1.0; n]
    };

    // x = x0 or zeros
    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![0.0; n],
    };

    // r = b - A*x
    let ax = matrix.spmv(&x)?;
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, &ai)| bi - ai).collect();

    // z = M^{-1} r
    let mut z = vec![0.0; n];
    apply_jacobi(&diag, &r, &mut z);

    // p = z
    let mut p = z.clone();

    let mut rz = dot(&r, &z);
    let b_norm = norm2(b);
    let tol_abs = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        config.tol
    };

    let mut iter = 0usize;
    let mut converged = false;

    while iter < config.max_iter {
        let ap = matrix.spmv(&p)?;
        let pap = dot(&p, &ap);
        if pap.abs() < f64::MIN_POSITIVE {
            break; // breakdown
        }
        let alpha = rz / pap;

        // x += alpha * p
        axpy(alpha, &p, &mut x);

        // r -= alpha * Ap
        axpy(-alpha, &ap, &mut r);

        let r_norm = norm2(&r);
        iter += 1;

        if r_norm <= tol_abs {
            converged = true;
            break;
        }

        // z = M^{-1} r
        apply_jacobi(&diag, &r, &mut z);

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        // p = z + beta * p
        let p_old = p.clone();
        axpby(1.0, &z, beta, &p_old, &mut p);
    }

    let residual = matrix.spmv(&x)?;
    let res_norm = norm2(
        &b.iter()
            .zip(residual.iter())
            .map(|(bi, &ri)| bi - ri)
            .collect::<Vec<_>>(),
    );

    Ok(SolverResult {
        x,
        residual_norm: res_norm,
        n_iter: iter,
        converged,
    })
}

// ============================================================
// BiCGSTAB
// ============================================================

/// Solve a general (possibly non-symmetric) system `A * x = b` using
/// Bi-Conjugate Gradient Stabilised (BiCGSTAB).
///
/// An optional initial guess `x0` can be supplied.  When `config.precond`
/// is true a Jacobi (diagonal) right-preconditioner is used.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] when matrix and vector sizes are
/// incompatible.
pub fn bicgstab_csr(
    matrix: &GpuCsrMatrix,
    b: &[f64],
    x0: Option<&[f64]>,
    config: &GpuSolverConfig,
) -> SparseResult<SolverResult> {
    let n = matrix.n_rows;
    if matrix.n_cols != n {
        return Err(SparseError::ComputationError(
            "BiCGSTAB requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let diag = if config.precond {
        jacobi_diag(matrix)
    } else {
        vec![1.0; n]
    };

    let mut x = match x0 {
        Some(x0) => x0.to_vec(),
        None => vec![0.0; n],
    };

    // r0 = b - A*x
    let ax0 = matrix.spmv(&x)?;
    let mut r: Vec<f64> = b.iter().zip(ax0.iter()).map(|(bi, &ai)| bi - ai).collect();

    // r_hat = r (shadow residual, frozen)
    let r_hat = r.clone();

    let b_norm = norm2(b);
    let tol_abs = if b_norm > 0.0 {
        config.tol * b_norm
    } else {
        config.tol
    };

    // p = r
    let mut p = r.clone();

    let mut rho = dot(&r_hat, &r);
    // omega is always set before use within the loop; the initial value is a
    // standard BiCGSTAB initialization but never read before reassignment.
    #[allow(unused_assignments)]
    let mut omega = 1.0_f64;

    // Preconditioner scratch buffers
    let mut p_hat = vec![0.0; n];
    let mut s_hat = vec![0.0; n];

    let mut iter = 0usize;
    let mut converged = false;

    while iter < config.max_iter {
        // p_hat = M^{-1} p
        apply_jacobi(&diag, &p, &mut p_hat);

        let v = matrix.spmv(&p_hat)?;
        let rtv = dot(&r_hat, &v);
        if rtv.abs() < f64::MIN_POSITIVE {
            break; // breakdown
        }
        let alpha = rho / rtv;

        // s = r - alpha * v
        let mut s: Vec<f64> = r
            .iter()
            .zip(v.iter())
            .map(|(&ri, &vi)| ri - alpha * vi)
            .collect();

        let s_norm = norm2(&s);
        if s_norm <= tol_abs {
            axpy(alpha, &p_hat, &mut x);
            iter += 1;
            converged = true;
            break;
        }

        // s_hat = M^{-1} s
        apply_jacobi(&diag, &s, &mut s_hat);

        let t = matrix.spmv(&s_hat)?;
        let tt = dot(&t, &t);
        omega = if tt > f64::MIN_POSITIVE {
            dot(&t, &s) / tt
        } else {
            break;
        };

        // x += alpha * p_hat + omega * s_hat
        axpy(alpha, &p_hat, &mut x);
        axpy(omega, &s_hat, &mut x);

        // r = s - omega * t
        for ((ri, &si), &ti) in r.iter_mut().zip(s.iter()).zip(t.iter()) {
            *ri = si - omega * ti;
        }

        let r_norm = norm2(&r);
        iter += 1;
        if r_norm <= tol_abs {
            converged = true;
            break;
        }

        // rho_new = (r_hat, r)
        let rho_new = dot(&r_hat, &r);
        if rho_new.abs() < f64::MIN_POSITIVE {
            break;
        }
        let beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;

        // p = r + beta * (p - omega * v)
        for ((pi, &ri), &vi) in p.iter_mut().zip(r.iter()).zip(v.iter()) {
            *pi = ri + beta * (*pi - omega * vi);
        }
    }

    let residual = matrix.spmv(&x)?;
    let res_norm = norm2(
        &b.iter()
            .zip(residual.iter())
            .map(|(bi, &ri)| bi - ri)
            .collect::<Vec<_>>(),
    );

    Ok(SolverResult {
        x,
        residual_norm: res_norm,
        n_iter: iter,
        converged,
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::construction::{GpuCooMatrix, GpuCsrMatrix};

    /// Build an n×n tridiagonal SPD matrix: 4 on diagonal, -1 on off-diagonals.
    fn tridiag_spd(n: usize) -> GpuCsrMatrix {
        let mut coo = GpuCooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 4.0);
            if i > 0 {
                coo.push(i, i - 1, -1.0);
                coo.push(i - 1, i, -1.0);
            }
        }
        coo.to_csr()
    }

    #[test]
    fn test_cg_spd_system() {
        let n = 5;
        let mat = tridiag_spd(n);
        let x_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = mat.spmv(&x_true).expect("spmv failed");

        let config = GpuSolverConfig::default();
        let result = cg_csr(&mat, &b, None, &config).expect("CG failed");
        assert!(result.converged, "CG did not converge");
        assert!(result.residual_norm < 1e-6);
        for (xi, &xt) in result.x.iter().zip(x_true.iter()) {
            assert!((xi - xt).abs() < 1e-6, "x[i]={xi} expected {xt}");
        }
    }

    #[test]
    fn test_bicgstab_general() {
        // Non-symmetric 4×4 system
        let mut coo = GpuCooMatrix::new(4, 4);
        coo.push(0, 0, 4.0);
        coo.push(0, 1, 1.0);
        coo.push(1, 0, 2.0);
        coo.push(1, 1, 5.0);
        coo.push(1, 2, 1.0);
        coo.push(2, 1, 2.0);
        coo.push(2, 2, 6.0);
        coo.push(2, 3, 1.0);
        coo.push(3, 2, 2.0);
        coo.push(3, 3, 7.0);
        let mat = coo.to_csr();

        let x_true = vec![1.0, 2.0, 3.0, 4.0];
        let b = mat.spmv(&x_true).expect("spmv failed");

        let config = GpuSolverConfig::default();
        let result = bicgstab_csr(&mat, &b, None, &config).expect("BiCGSTAB failed");
        assert!(result.converged, "BiCGSTAB did not converge");
        assert!(result.residual_norm < 1e-6);
    }

    #[test]
    fn test_cg_with_precond() {
        // Jacobi preconditioner should reduce iteration count versus no-precond
        let n = 10;
        let mat = tridiag_spd(n);
        let b = vec![1.0; n];

        let config_precond = GpuSolverConfig {
            precond: true,
            ..Default::default()
        };
        let config_nopc = GpuSolverConfig {
            precond: false,
            ..Default::default()
        };

        let r_precond = cg_csr(&mat, &b, None, &config_precond).expect("CG failed");
        let r_nopc = cg_csr(&mat, &b, None, &config_nopc).expect("CG failed");

        assert!(r_precond.converged);
        assert!(r_nopc.converged);
        // Both should converge; with Jacobi precond typically ≤ without
        assert!(r_precond.n_iter <= r_nopc.n_iter + 5); // allow small slack
    }

    #[test]
    fn test_cg_with_initial_guess() {
        let n = 5;
        let mat = tridiag_spd(n);
        let x_true = vec![1.0; n];
        let b = mat.spmv(&x_true).expect("spmv failed");

        // Good initial guess should reduce iterations
        let x0 = vec![0.9; n];
        let config = GpuSolverConfig::default();
        let result = cg_csr(&mat, &b, Some(&x0), &config).expect("CG failed");
        assert!(result.converged);
    }

    #[test]
    fn test_solver_dimension_mismatch() {
        let n = 3;
        let mat = tridiag_spd(n);
        let b_wrong = vec![1.0; n + 1];
        let config = GpuSolverConfig::default();
        assert!(cg_csr(&mat, &b_wrong, None, &config).is_err());
        assert!(bicgstab_csr(&mat, &b_wrong, None, &config).is_err());
    }
}
