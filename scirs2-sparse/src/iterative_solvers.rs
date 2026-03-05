//! Enhanced iterative solvers for sparse linear systems
//!
//! This module provides production-quality iterative solvers with ndarray-based
//! interfaces, preconditioner abstractions, and sparse matrix utility functions.
//!
//! # Solvers
//!
//! - **CG**: Conjugate Gradient for symmetric positive definite (SPD) systems
//! - **BiCGSTAB**: Biconjugate Gradient Stabilized for general square systems
//! - **GMRES(m)**: Generalized Minimal Residual with restarts for general systems
//! - **Chebyshev**: Chebyshev iteration for SPD systems with known eigenvalue bounds
//!
//! # Preconditioners
//!
//! - **Jacobi**: Diagonal (inverse) preconditioner
//! - **ILU(0)**: Incomplete LU with zero fill-in
//! - **SSOR**: Symmetric Successive Over-Relaxation
//!
//! # Utility Functions
//!
//! - `estimate_spectral_radius`: Power iteration based spectral radius estimation
//! - `sparse_diagonal`: Extract diagonal of a sparse matrix
//! - `sparse_trace`: Compute trace of a sparse matrix
//! - `sparse_norm`: Compute Frobenius, infinity, or 1-norm of a sparse matrix

use crate::csr::CsrMatrix;
use crate::error::{SparseError, SparseResult};
use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign, SparseElement};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, MulAssign};

// ---------------------------------------------------------------------------
// Configuration and result types
// ---------------------------------------------------------------------------

/// Configuration for iterative solvers.
#[derive(Debug, Clone)]
pub struct IterativeSolverConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Relative convergence tolerance.
    pub tol: f64,
    /// Whether to print convergence information.
    pub verbose: bool,
}

impl Default for IterativeSolverConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-10,
            verbose: false,
        }
    }
}

/// Result returned by an iterative solver.
#[derive(Debug, Clone)]
pub struct SolverResult<F> {
    /// The computed solution vector.
    pub solution: Array1<F>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Final residual norm ||b - Ax||.
    pub residual_norm: F,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Preconditioner trait and implementations
// ---------------------------------------------------------------------------

/// Trait for preconditioners used with iterative solvers.
///
/// A preconditioner approximates `M^{-1}` so that `M^{-1} A` has a more
/// clustered spectrum, accelerating convergence.
pub trait Preconditioner<F: Float> {
    /// Apply the preconditioner to vector `r`, returning `M^{-1} r`.
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>>;
}

/// Jacobi (diagonal) preconditioner.
///
/// Uses `M = diag(A)`, so `M^{-1} r = r ./ diag(A)`.
/// Effective when the matrix is diagonally dominant.
pub struct JacobiPreconditioner<F> {
    diagonal_inv: Array1<F>,
}

impl<F: Float + SparseElement + Debug> JacobiPreconditioner<F> {
    /// Create a Jacobi preconditioner from a CSR matrix.
    ///
    /// Returns an error if any diagonal element is zero or near-zero.
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "Matrix must be square for Jacobi preconditioner".to_string(),
            ));
        }
        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = matrix.get(i, i);
            if d.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at row {i} prevents Jacobi preconditioner"
                )));
            }
            diag_inv[i] = F::sparse_one() / d;
        }
        Ok(Self {
            diagonal_inv: diag_inv,
        })
    }

    /// Create a Jacobi preconditioner from an explicit diagonal vector.
    pub fn from_diagonal(diagonal: Array1<F>) -> SparseResult<Self> {
        let n = diagonal.len();
        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            if diagonal[i].abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at position {i}"
                )));
            }
            diag_inv[i] = F::sparse_one() / diagonal[i];
        }
        Ok(Self {
            diagonal_inv: diag_inv,
        })
    }
}

impl<F: Float + SparseElement> Preconditioner<F> for JacobiPreconditioner<F> {
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.diagonal_inv.len() {
            return Err(SparseError::DimensionMismatch {
                expected: self.diagonal_inv.len(),
                found: r.len(),
            });
        }
        Ok(r * &self.diagonal_inv)
    }
}

/// ILU(0) preconditioner (Incomplete LU with zero fill-in).
///
/// Computes an approximate factorization `A ~ L U` where L and U
/// retain only the sparsity pattern of A.
pub struct ILU0Preconditioner<F> {
    // Store the combined LU data in CSR-like arrays.
    l_data: Vec<F>,
    u_data: Vec<F>,
    l_indices: Vec<usize>,
    u_indices: Vec<usize>,
    l_indptr: Vec<usize>,
    u_indptr: Vec<usize>,
    n: usize,
}

impl<F: Float + NumAssign + Sum + Debug + SparseElement + 'static> ILU0Preconditioner<F> {
    /// Construct an ILU(0) preconditioner from a CSR matrix.
    pub fn new(matrix: &CsrMatrix<F>) -> SparseResult<Self> {
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "Matrix must be square for ILU(0) preconditioner".to_string(),
            ));
        }

        // Copy matrix data for in-place modification
        let mut data = matrix.data.clone();
        let indices = matrix.indices.clone();
        let indptr = matrix.indptr.clone();

        // ILU(0) factorisation (Gaussian elimination with pattern restriction)
        for k in 0..n {
            let k_diag_idx = find_csr_diag_index(&indices, &indptr, k)?;
            let k_diag = data[k_diag_idx];
            if k_diag.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero pivot at row {k} in ILU(0) factorization"
                )));
            }

            for i in (k + 1)..n {
                let row_start = indptr[i];
                let row_end = indptr[i + 1];

                // Find column k in row i
                let mut k_pos = None;
                for pos in row_start..row_end {
                    if indices[pos] == k {
                        k_pos = Some(pos);
                        break;
                    }
                    if indices[pos] > k {
                        break;
                    }
                }

                if let Some(ki_idx) = k_pos {
                    let mult = data[ki_idx] / k_diag;
                    data[ki_idx] = mult;

                    // Update remaining entries in row i that also appear in row k
                    let k_row_start = indptr[k];
                    let k_row_end = indptr[k + 1];

                    for kj_idx in k_row_start..k_row_end {
                        let j = indices[kj_idx];
                        if j <= k {
                            continue;
                        }
                        // Find position of column j in row i
                        for ij_idx in row_start..row_end {
                            if indices[ij_idx] == j {
                                let kj_val = data[kj_idx];
                                data[ij_idx] -= mult * kj_val;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Split into L (unit lower) and U (upper including diagonal)
        let mut l_data = Vec::new();
        let mut u_data = Vec::new();
        let mut l_indices = Vec::new();
        let mut u_indices = Vec::new();
        let mut l_indptr = vec![0usize];
        let mut u_indptr = vec![0usize];

        for i in 0..n {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];
            for pos in row_start..row_end {
                let col = indices[pos];
                let val = data[pos];
                if col < i {
                    l_indices.push(col);
                    l_data.push(val);
                } else {
                    u_indices.push(col);
                    u_data.push(val);
                }
            }
            l_indptr.push(l_indices.len());
            u_indptr.push(u_indices.len());
        }

        Ok(Self {
            l_data,
            u_data,
            l_indices,
            u_indices,
            l_indptr,
            u_indptr,
            n,
        })
    }
}

impl<F: Float + NumAssign + Sum + Debug + SparseElement + 'static> Preconditioner<F>
    for ILU0Preconditioner<F>
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        if r.len() != self.n {
            return Err(SparseError::DimensionMismatch {
                expected: self.n,
                found: r.len(),
            });
        }

        // Forward solve: L y = r  (L is unit-lower triangular)
        let mut y = Array1::zeros(self.n);
        for i in 0..self.n {
            y[i] = r[i];
            let start = self.l_indptr[i];
            let end = self.l_indptr[i + 1];
            for pos in start..end {
                let col = self.l_indices[pos];
                y[i] = y[i] - self.l_data[pos] * y[col];
            }
        }

        // Backward solve: U z = y
        let mut z = Array1::zeros(self.n);
        for i in (0..self.n).rev() {
            z[i] = y[i];
            let start = self.u_indptr[i];
            let end = self.u_indptr[i + 1];
            let mut diag_val = F::sparse_one();
            for pos in start..end {
                let col = self.u_indices[pos];
                if col == i {
                    diag_val = self.u_data[pos];
                } else if col > i {
                    z[i] = z[i] - self.u_data[pos] * z[col];
                }
            }
            z[i] /= diag_val;
        }

        Ok(z)
    }
}

/// SSOR (Symmetric Successive Over-Relaxation) preconditioner.
///
/// Uses the splitting `M = (D + omega L) D^{-1} (D + omega U) / (2 - omega)`,
/// with relaxation parameter `omega in (0, 2)`.
pub struct SSORPreconditioner<F> {
    omega: F,
    matrix: CsrMatrix<F>,
    diagonal: Vec<F>,
}

impl<F: Float + NumAssign + Sum + Debug + SparseElement + 'static> SSORPreconditioner<F> {
    /// Create an SSOR preconditioner.
    ///
    /// `omega` must lie in the open interval (0, 2).
    pub fn new(matrix: CsrMatrix<F>, omega: F) -> SparseResult<Self> {
        let two = F::from(2.0).ok_or_else(|| {
            SparseError::ValueError("Failed to convert 2.0 to float type".to_string())
        })?;
        if omega <= F::sparse_zero() || omega >= two {
            return Err(SparseError::ValueError(
                "SSOR omega must be in the open interval (0, 2)".to_string(),
            ));
        }
        let n = matrix.rows();
        if n != matrix.cols() {
            return Err(SparseError::ValueError(
                "Matrix must be square for SSOR preconditioner".to_string(),
            ));
        }
        let mut diagonal = vec![F::sparse_zero(); n];
        for i in 0..n {
            let d = matrix.get(i, i);
            if d.abs() < F::epsilon() {
                return Err(SparseError::ValueError(format!(
                    "Zero diagonal element at row {i} prevents SSOR preconditioner"
                )));
            }
            diagonal[i] = d;
        }
        Ok(Self {
            omega,
            matrix,
            diagonal,
        })
    }
}

impl<F: Float + NumAssign + Sum + Debug + SparseElement + 'static> Preconditioner<F>
    for SSORPreconditioner<F>
{
    fn apply(&self, r: &Array1<F>) -> SparseResult<Array1<F>> {
        let n = self.matrix.rows();
        if r.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: r.len(),
            });
        }

        // Forward sweep: (D + omega L) y = omega (2 - omega) r
        let two = F::from(2.0)
            .ok_or_else(|| SparseError::ValueError("Failed to convert 2.0 to float".to_string()))?;
        let scale = self.omega * (two - self.omega);
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = r[i] * scale;
            let range = self.matrix.row_range(i);
            let row_indices = &self.matrix.indices[range.clone()];
            let row_data = &self.matrix.data[range];
            for (idx, &col) in row_indices.iter().enumerate() {
                if col < i {
                    sum -= self.omega * row_data[idx] * y[col];
                }
            }
            y[i] = sum / self.diagonal[i];
        }

        // Diagonal scaling: z_i = D_i * y_i / D_i  (effectively z = y scaled by D * D^{-1} identity)
        // The correct SSOR combines forward + backward with diagonal in between:
        // z_i = D_i * y_i
        let mut z = Array1::zeros(n);
        for i in 0..n {
            z[i] = y[i] * self.diagonal[i];
        }

        // Backward sweep: (D + omega U) w = z
        let mut w = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = z[i];
            let range = self.matrix.row_range(i);
            let row_indices = &self.matrix.indices[range.clone()];
            let row_data = &self.matrix.data[range];
            for (idx, &col) in row_indices.iter().enumerate() {
                if col > i {
                    sum -= self.omega * row_data[idx] * w[col];
                }
            }
            w[i] = sum / self.diagonal[i];
        }

        Ok(w)
    }
}

// ---------------------------------------------------------------------------
// Sparse matrix-vector multiplication helper
// ---------------------------------------------------------------------------

/// Compute y = A * x  for a CSR matrix `A` and dense vector `x`.
fn spmv<F: Float + NumAssign + Sum + SparseElement + 'static>(
    a: &CsrMatrix<F>,
    x: &Array1<F>,
) -> SparseResult<Array1<F>> {
    let (m, n) = a.shape();
    if x.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: x.len(),
        });
    }
    let mut y = Array1::zeros(m);
    for i in 0..m {
        let range = a.row_range(i);
        let cols = &a.indices[range.clone()];
        let vals = &a.data[range];
        let mut acc = F::sparse_zero();
        for (idx, &col) in cols.iter().enumerate() {
            acc += vals[idx] * x[col];
        }
        y[i] = acc;
    }
    Ok(y)
}

/// Compute the dot product of two Array1 vectors.
#[inline]
fn dot_arr<F: Float + Sum>(a: &Array1<F>, b: &Array1<F>) -> F {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

/// Compute the 2-norm of an Array1 vector.
#[inline]
fn norm2_arr<F: Float + Sum>(v: &Array1<F>) -> F {
    dot_arr(v, v).sqrt()
}

/// axpy: y = y + alpha * x
#[inline]
fn axpy<F: Float>(y: &mut Array1<F>, alpha: F, x: &Array1<F>) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi = *yi + alpha * xi;
    }
}

// ---------------------------------------------------------------------------
// Conjugate Gradient solver
// ---------------------------------------------------------------------------

/// Conjugate Gradient solver for symmetric positive definite systems.
///
/// Solves `A x = b` where `A` is SPD. Optionally accepts a preconditioner.
///
/// # Errors
///
/// Returns an error if dimensions are incompatible or the matrix is detected
/// to be non-positive-definite (negative `p^T A p`).
pub fn cg<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    config: &IterativeSolverConfig,
    precond: Option<&dyn Preconditioner<F>>,
) -> SparseResult<SolverResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "CG requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let tol = F::from(config.tol).ok_or_else(|| {
        SparseError::ValueError("Failed to convert tolerance to float type".to_string())
    })?;

    let mut x = Array1::zeros(n);

    // r = b - A x  (x = 0 initially, so r = b)
    let mut r = b.clone();
    let bnorm = norm2_arr(b);
    if bnorm <= F::epsilon() {
        return Ok(SolverResult {
            solution: x,
            n_iter: 0,
            residual_norm: F::sparse_zero(),
            converged: true,
        });
    }

    let tolerance = tol * bnorm;

    // z = M^{-1} r
    let mut z = match precond {
        Some(pc) => pc.apply(&r)?,
        None => r.clone(),
    };

    let mut p = z.clone();
    let mut rz = dot_arr(&r, &z);

    for k in 0..config.max_iter {
        let ap = spmv(a, &p)?;
        let pap = dot_arr(&p, &ap);
        if pap <= F::sparse_zero() {
            return Ok(SolverResult {
                solution: x,
                n_iter: k,
                residual_norm: norm2_arr(&r),
                converged: false,
            });
        }

        let alpha = rz / pap;
        axpy(&mut x, alpha, &p);

        // r = r - alpha * ap
        axpy(&mut r, -alpha, &ap);

        let rnorm = norm2_arr(&r);
        if config.verbose {
            // Intentionally not printing; the flag is available for future use
        }
        if rnorm <= tolerance {
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: rnorm,
                converged: true,
            });
        }

        z = match precond {
            Some(pc) => pc.apply(&r)?,
            None => r.clone(),
        };

        let rz_new = dot_arr(&r, &z);
        let beta = rz_new / rz;

        // p = z + beta * p
        for (pi, &zi) in p.iter_mut().zip(z.iter()) {
            *pi = zi + beta * *pi;
        }

        rz = rz_new;
    }

    let rnorm = norm2_arr(&r);
    Ok(SolverResult {
        solution: x,
        n_iter: config.max_iter,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
    })
}

// ---------------------------------------------------------------------------
// BiCGSTAB solver
// ---------------------------------------------------------------------------

/// Biconjugate Gradient Stabilized solver for general square systems.
///
/// Solves `A x = b` for non-symmetric `A`. This method is more stable
/// than vanilla BiCG and avoids the irregular convergence of CGS.
pub fn bicgstab<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    config: &IterativeSolverConfig,
    precond: Option<&dyn Preconditioner<F>>,
) -> SparseResult<SolverResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "BiCGSTAB requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let tol = F::from(config.tol).ok_or_else(|| {
        SparseError::ValueError("Failed to convert tolerance to float type".to_string())
    })?;

    let mut x = Array1::zeros(n);
    let mut r = b.clone();
    let bnorm = norm2_arr(b);
    if bnorm <= F::epsilon() {
        return Ok(SolverResult {
            solution: x,
            n_iter: 0,
            residual_norm: F::sparse_zero(),
            converged: true,
        });
    }
    let tolerance = tol * bnorm;

    let r_hat = r.clone(); // shadow residual

    let mut rho = F::sparse_one();
    let mut alpha = F::sparse_one();
    let mut omega = F::sparse_one();

    let mut v = Array1::zeros(n);
    let mut p = Array1::zeros(n);

    let ten_eps = F::epsilon()
        * F::from(10.0).ok_or_else(|| {
            SparseError::ValueError("Failed to convert 10.0 to float".to_string())
        })?;

    for k in 0..config.max_iter {
        let rho_new = dot_arr(&r_hat, &r);
        if rho_new.abs() < ten_eps {
            return Ok(SolverResult {
                solution: x,
                n_iter: k,
                residual_norm: norm2_arr(&r),
                converged: false,
            });
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // Apply preconditioner
        let p_hat = match precond {
            Some(pc) => pc.apply(&p)?,
            None => p.clone(),
        };

        v = spmv(a, &p_hat)?;

        let den = dot_arr(&r_hat, &v);
        if den.abs() < ten_eps {
            return Ok(SolverResult {
                solution: x,
                n_iter: k,
                residual_norm: norm2_arr(&r),
                converged: false,
            });
        }
        alpha = rho_new / den;

        // s = r - alpha * v
        let mut s = r.clone();
        axpy(&mut s, -alpha, &v);

        let snorm = norm2_arr(&s);
        if snorm <= tolerance {
            axpy(&mut x, alpha, &p_hat);
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: snorm,
                converged: true,
            });
        }

        // Apply preconditioner to s
        let s_hat = match precond {
            Some(pc) => pc.apply(&s)?,
            None => s.clone(),
        };

        let t = spmv(a, &s_hat)?;

        let tt = dot_arr(&t, &t);
        if tt < ten_eps {
            axpy(&mut x, alpha, &p_hat);
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: snorm,
                converged: false,
            });
        }
        omega = dot_arr(&t, &s) / tt;

        // x = x + alpha * p_hat + omega * s_hat
        axpy(&mut x, alpha, &p_hat);
        axpy(&mut x, omega, &s_hat);

        // r = s - omega * t
        r = s;
        axpy(&mut r, -omega, &t);

        let rnorm = norm2_arr(&r);
        if rnorm <= tolerance {
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: rnorm,
                converged: true,
            });
        }

        if omega.abs() < ten_eps {
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: rnorm,
                converged: false,
            });
        }

        rho = rho_new;
    }

    let rnorm = norm2_arr(&r);
    Ok(SolverResult {
        solution: x,
        n_iter: config.max_iter,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
    })
}

// ---------------------------------------------------------------------------
// GMRES(m) solver
// ---------------------------------------------------------------------------

/// Restarted Generalized Minimal Residual solver for general square systems.
///
/// GMRES minimises the residual over a Krylov subspace using Arnoldi
/// iteration with Givens rotations. The `restart` parameter controls
/// the dimension of the Krylov subspace before a restart.
pub fn gmres<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    config: &IterativeSolverConfig,
    restart: usize,
    precond: Option<&dyn Preconditioner<F>>,
) -> SparseResult<SolverResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + ScalarOperand + Debug + 'static,
{
    let (m_rows, n) = a.shape();
    if m_rows != n {
        return Err(SparseError::ValueError(
            "GMRES requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }

    let tol = F::from(config.tol).ok_or_else(|| {
        SparseError::ValueError("Failed to convert tolerance to float type".to_string())
    })?;

    let restart_dim = restart.min(n).max(1);

    let mut x = Array1::zeros(n);
    let bnorm = norm2_arr(b);
    if bnorm <= F::epsilon() {
        return Ok(SolverResult {
            solution: x,
            n_iter: 0,
            residual_norm: F::sparse_zero(),
            converged: true,
        });
    }
    let tolerance = tol * bnorm;
    let ten_eps = F::epsilon()
        * F::from(10.0).ok_or_else(|| {
            SparseError::ValueError("Failed to convert 10.0 to float".to_string())
        })?;

    let mut total_iter = 0usize;

    // Outer restart loop
    while total_iter < config.max_iter {
        // Compute residual
        let ax = spmv(a, &x)?;
        let mut r = b.clone();
        axpy(&mut r, -F::sparse_one(), &ax);

        // Apply preconditioner
        r = match precond {
            Some(pc) => pc.apply(&r)?,
            None => r,
        };

        let mut rnorm = norm2_arr(&r);
        if rnorm <= tolerance {
            return Ok(SolverResult {
                solution: x,
                n_iter: total_iter,
                residual_norm: rnorm,
                converged: true,
            });
        }

        // Arnoldi basis
        let mut v_basis: Vec<Array1<F>> = Vec::with_capacity(restart_dim + 1);
        v_basis.push(&r / rnorm);

        // Hessenberg matrix (stored column-by-column)
        let mut h = vec![vec![F::sparse_zero(); restart_dim]; restart_dim + 1];
        // Givens rotation cosines and sines
        let mut cs = vec![F::sparse_zero(); restart_dim];
        let mut sn = vec![F::sparse_zero(); restart_dim];
        // Right-hand side of the least-squares problem
        let mut g = vec![F::sparse_zero(); restart_dim + 1];
        g[0] = rnorm;

        let mut inner_iter = 0usize;
        while inner_iter < restart_dim && total_iter + inner_iter < config.max_iter {
            let j = inner_iter;

            // w = A * M^{-1} * v_j  (right preconditioned)
            let mut w = spmv(a, &v_basis[j])?;
            w = match precond {
                Some(pc) => pc.apply(&w)?,
                None => w,
            };

            // Modified Gram-Schmidt orthogonalisation
            for i in 0..=j {
                h[i][j] = dot_arr(&v_basis[i], &w);
                axpy(&mut w, -h[i][j], &v_basis[i]);
            }
            h[j + 1][j] = norm2_arr(&w);

            if h[j + 1][j] < ten_eps {
                // Lucky breakdown: residual is in the Krylov subspace
                inner_iter += 1;
                break;
            }

            v_basis.push(&w / h[j + 1][j]);

            // Apply previous Givens rotations to column j
            for i in 0..j {
                let temp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = temp;
            }

            // Compute new Givens rotation for row j
            let (c_val, s_val, r_val) = givens_rotation(h[j][j], h[j + 1][j]);
            cs[j] = c_val;
            sn[j] = s_val;
            h[j][j] = r_val;
            h[j + 1][j] = F::sparse_zero();

            // Apply rotation to g
            let temp = cs[j] * g[j] + sn[j] * g[j + 1];
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
            g[j] = temp;

            inner_iter += 1;

            rnorm = g[j + 1].abs();
            if rnorm <= tolerance {
                break;
            }
        }

        // Solve the upper triangular system H * y = g
        let m_dim = inner_iter;
        let mut y_vec = vec![F::sparse_zero(); m_dim];
        for i in (0..m_dim).rev() {
            y_vec[i] = g[i];
            for jj in (i + 1)..m_dim {
                y_vec[i] = y_vec[i] - h[i][jj] * y_vec[jj];
            }
            if h[i][i].abs() < ten_eps {
                // Skip near-zero diagonal; keep current best
                y_vec[i] = F::sparse_zero();
            } else {
                y_vec[i] /= h[i][i];
            }
        }

        // Update solution: x = x + V * y
        for (i, &yi) in y_vec.iter().enumerate() {
            axpy(&mut x, yi, &v_basis[i]);
        }

        total_iter += inner_iter;

        if rnorm <= tolerance {
            return Ok(SolverResult {
                solution: x,
                n_iter: total_iter,
                residual_norm: rnorm,
                converged: true,
            });
        }
    }

    let ax = spmv(a, &x)?;
    let mut r_final = b.clone();
    axpy(&mut r_final, -F::sparse_one(), &ax);
    let rnorm = norm2_arr(&r_final);

    Ok(SolverResult {
        solution: x,
        n_iter: total_iter,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
    })
}

/// Compute a Givens rotation (c, s, r) such that:
///   [ c  s ] [ a ] = [ r ]
///   [-s  c ] [ b ]   [ 0 ]
fn givens_rotation<F: Float + SparseElement>(a: F, b: F) -> (F, F, F) {
    let zero = F::sparse_zero();
    let one = F::sparse_one();
    if b == zero {
        let c = if a >= zero { one } else { -one };
        return (c, zero, a.abs());
    }
    if a == zero {
        let s = if b >= zero { one } else { -one };
        return (zero, s, b.abs());
    }
    if b.abs() > a.abs() {
        let tau = a / b;
        let s_sign = if b >= zero { one } else { -one };
        let s = s_sign / (one + tau * tau).sqrt();
        let c = s * tau;
        let r = b / s;
        (c, s, r)
    } else {
        let tau = b / a;
        let c_sign = if a >= zero { one } else { -one };
        let c = c_sign / (one + tau * tau).sqrt();
        let s = c * tau;
        let r = a / c;
        (c, s, r)
    }
}

// ---------------------------------------------------------------------------
// Chebyshev iteration
// ---------------------------------------------------------------------------

/// Chebyshev iteration for SPD systems with known eigenvalue bounds.
///
/// Accelerates stationary iteration using Chebyshev polynomials. Requires
/// estimates `lambda_min` and `lambda_max` of the smallest and largest
/// eigenvalues of `A`. The convergence rate depends on the ratio
/// `lambda_max / lambda_min`.
///
/// Unlike CG, Chebyshev iteration does not require inner products,
/// making it attractive for massively parallel environments.
pub fn chebyshev<F>(
    a: &CsrMatrix<F>,
    b: &Array1<F>,
    config: &IterativeSolverConfig,
    lambda_min: F,
    lambda_max: F,
) -> SparseResult<SolverResult<F>>
where
    F: Float + NumAssign + Sum + SparseElement + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "Chebyshev iteration requires a square matrix".to_string(),
        ));
    }
    if b.len() != n {
        return Err(SparseError::DimensionMismatch {
            expected: n,
            found: b.len(),
        });
    }
    if lambda_min <= F::sparse_zero() || lambda_max <= F::sparse_zero() {
        return Err(SparseError::ValueError(
            "Eigenvalue bounds must be positive for Chebyshev iteration".to_string(),
        ));
    }
    if lambda_min >= lambda_max {
        return Err(SparseError::ValueError(
            "lambda_min must be strictly less than lambda_max".to_string(),
        ));
    }

    let tol = F::from(config.tol).ok_or_else(|| {
        SparseError::ValueError("Failed to convert tolerance to float type".to_string())
    })?;
    let two = F::from(2.0)
        .ok_or_else(|| SparseError::ValueError("Failed to convert 2.0 to float".to_string()))?;

    let bnorm = norm2_arr(b);
    if bnorm <= F::epsilon() {
        return Ok(SolverResult {
            solution: Array1::zeros(n),
            n_iter: 0,
            residual_norm: F::sparse_zero(),
            converged: true,
        });
    }
    let tolerance = tol * bnorm;

    // Chebyshev parameters
    let d = (lambda_max + lambda_min) / two;
    let c = (lambda_max - lambda_min) / two;

    let mut x = Array1::zeros(n);
    let mut r = b.clone();
    let mut rnorm = norm2_arr(&r);

    if rnorm <= tolerance {
        return Ok(SolverResult {
            solution: x,
            n_iter: 0,
            residual_norm: rnorm,
            converged: true,
        });
    }

    // First iteration: x_1 = x_0 + (1/d) * r_0
    let inv_d = F::sparse_one() / d;
    let mut p = Array1::zeros(n);
    for i in 0..n {
        p[i] = inv_d * r[i];
    }
    axpy(&mut x, F::sparse_one(), &p);

    let ax = spmv(a, &x)?;
    for i in 0..n {
        r[i] = b[i] - ax[i];
    }
    rnorm = norm2_arr(&r);

    if rnorm <= tolerance {
        return Ok(SolverResult {
            solution: x,
            n_iter: 1,
            residual_norm: rnorm,
            converged: true,
        });
    }

    // Subsequent iterations
    let mut alpha;
    let mut beta;
    let half = F::sparse_one() / two;

    // rho_0 = 1/d, rho_1 = d / (2c^2 - d)   -- recurrence parameter
    // Actually the standard Chebyshev iteration uses:
    //   theta = (lambda_max + lambda_min) / 2
    //   delta = (lambda_max - lambda_min) / 2
    //   sigma_1 = theta / delta
    //   rho_0 = 1 / sigma_1
    //   For k >= 1:  rho_k = 1 / (2 sigma_1 - rho_{k-1})
    //   alpha_k = 2 rho_k / theta  (but below we use standard formulation)
    //
    // We use the three-term recurrence formulation:
    //   x_{k+1} = x_k + alpha_k (b - A x_k) + beta_k (x_k - x_{k-1})
    //
    // With alpha_0 = 1/d, beta_0 = 0
    // For k >= 1:
    //   beta_k = (c * alpha_{k-1} / 2)^2
    //   alpha_k = 1 / (d - beta_k / alpha_{k-1})

    let mut alpha_prev = inv_d;
    let c_half = c * half;

    for k in 1..config.max_iter {
        beta = (c_half * alpha_prev) * (c_half * alpha_prev);
        let denom = d - beta / alpha_prev;
        if denom.abs() < F::epsilon() {
            break;
        }
        alpha = F::sparse_one() / denom;

        // p = alpha * r + beta * p
        for i in 0..n {
            p[i] = alpha * r[i] + beta * p[i];
        }
        axpy(&mut x, F::sparse_one(), &p);

        let ax_k = spmv(a, &x)?;
        for i in 0..n {
            r[i] = b[i] - ax_k[i];
        }
        rnorm = norm2_arr(&r);

        if rnorm <= tolerance {
            return Ok(SolverResult {
                solution: x,
                n_iter: k + 1,
                residual_norm: rnorm,
                converged: true,
            });
        }

        alpha_prev = alpha;
    }

    Ok(SolverResult {
        solution: x,
        n_iter: config.max_iter,
        residual_norm: rnorm,
        converged: rnorm <= tolerance,
    })
}

// ---------------------------------------------------------------------------
// Sparse utility functions
// ---------------------------------------------------------------------------

/// Norm type for `sparse_norm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Frobenius norm: sqrt( sum |a_ij|^2 ).
    Frobenius,
    /// Infinity norm: max_i sum_j |a_ij|  (maximum absolute row sum).
    Inf,
    /// 1-norm: max_j sum_i |a_ij|  (maximum absolute column sum).
    One,
}

/// Estimate the spectral radius of `A` via power iteration.
///
/// Performs `n_iter` steps of the power method on `A` and returns
/// the Rayleigh quotient as an estimate of the spectral radius.
pub fn estimate_spectral_radius<F>(a: &CsrMatrix<F>, n_iter: usize) -> SparseResult<F>
where
    F: Float + NumAssign + Sum + SparseElement + ScalarOperand + Debug + 'static,
{
    let (m, n) = a.shape();
    if m != n {
        return Err(SparseError::ValueError(
            "Matrix must be square to estimate spectral radius".to_string(),
        ));
    }
    if n == 0 {
        return Ok(F::sparse_zero());
    }

    // Initial vector: all ones normalised
    let inv_sqrt_n = F::sparse_one()
        / F::from(n)
            .ok_or_else(|| {
                SparseError::ValueError("Failed to convert matrix size to float".to_string())
            })?
            .sqrt();
    let mut v = Array1::from_elem(n, inv_sqrt_n);

    let mut lambda = F::sparse_zero();
    let iters = if n_iter == 0 { 50 } else { n_iter };

    for _ in 0..iters {
        let w = spmv(a, &v)?;
        lambda = dot_arr(&v, &w);
        let wnorm = norm2_arr(&w);
        if wnorm < F::epsilon() {
            return Ok(F::sparse_zero());
        }
        v = &w / wnorm;
    }

    Ok(lambda.abs())
}

/// Extract the diagonal of a CSR matrix as an `Array1`.
pub fn sparse_diagonal<F>(a: &CsrMatrix<F>) -> Array1<F>
where
    F: Float + SparseElement,
{
    let (m, n) = a.shape();
    let dim = m.min(n);
    let mut diag = Array1::zeros(dim);
    for i in 0..dim {
        diag[i] = a.get(i, i);
    }
    diag
}

/// Compute the trace of a CSR matrix (sum of diagonal elements).
pub fn sparse_trace<F>(a: &CsrMatrix<F>) -> F
where
    F: Float + SparseElement,
{
    let (m, n) = a.shape();
    let dim = m.min(n);
    let mut tr = F::sparse_zero();
    for i in 0..dim {
        tr = tr + a.get(i, i);
    }
    tr
}

/// Compute a matrix norm of a CSR matrix.
///
/// Supports Frobenius, infinity (max row sum), and 1-norm (max column sum).
pub fn sparse_norm<F>(a: &CsrMatrix<F>, norm_type: NormType) -> F
where
    F: Float + NumAssign + SparseElement + AddAssign + MulAssign + 'static,
{
    match norm_type {
        NormType::Frobenius => {
            let mut sum_sq = F::sparse_zero();
            for &val in &a.data {
                sum_sq += val * val;
            }
            sum_sq.sqrt()
        }
        NormType::Inf => {
            let m = a.rows();
            let mut max_sum = F::sparse_zero();
            for i in 0..m {
                let range = a.row_range(i);
                let row_data = &a.data[range];
                let mut row_sum = F::sparse_zero();
                for &v in row_data {
                    let abs_v: F = v.abs();
                    row_sum += abs_v;
                }
                if row_sum > max_sum {
                    max_sum = row_sum;
                }
            }
            max_sum
        }
        NormType::One => {
            let n = a.cols();
            let mut col_sums = vec![F::sparse_zero(); n];
            for (&col, &val) in a.indices.iter().zip(a.data.iter()) {
                if col < n {
                    let abs_val: F = val.abs();
                    col_sums[col] += abs_val;
                }
            }
            let mut max_sum = F::sparse_zero();
            for &s in &col_sums {
                if s > max_sum {
                    max_sum = s;
                }
            }
            max_sum
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Find the index of the diagonal entry in a CSR row.
fn find_csr_diag_index(indices: &[usize], indptr: &[usize], row: usize) -> SparseResult<usize> {
    let start = indptr[row];
    let end = indptr[row + 1];
    for pos in start..end {
        if indices[pos] == row {
            return Ok(pos);
        }
    }
    Err(SparseError::ValueError(format!(
        "Missing diagonal element at row {row}"
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build an SPD 3x3 matrix:
    ///   [4 -1 -1]
    ///   [-1  4 -1]
    ///   [-1 -1  4]
    fn spd_3x3() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cols = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];
        let data = vec![4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 4.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("failed to build test matrix")
    }

    /// Helper: build a non-symmetric 3x3 matrix:
    ///   [5 -1  0]
    ///   [-2  5 -1]
    ///   [0  -2  5]
    fn nonsym_3x3() -> CsrMatrix<f64> {
        let rows = vec![0, 0, 1, 1, 1, 2, 2];
        let cols = vec![0, 1, 0, 1, 2, 1, 2];
        let data = vec![5.0, -1.0, -2.0, 5.0, -1.0, -2.0, 5.0];
        CsrMatrix::new(data, rows, cols, (3, 3)).expect("failed to build test matrix")
    }

    /// Helper: build a larger 5x5 SPD tridiagonal matrix
    fn spd_5x5() -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();
        for i in 0..5 {
            rows.push(i);
            cols.push(i);
            data.push(4.0);
            if i > 0 {
                rows.push(i);
                cols.push(i - 1);
                data.push(-1.0);
            }
            if i < 4 {
                rows.push(i);
                cols.push(i + 1);
                data.push(-1.0);
            }
        }
        CsrMatrix::new(data, rows, cols, (5, 5)).expect("failed to build test matrix")
    }

    fn rhs_3() -> Array1<f64> {
        Array1::from_vec(vec![1.0, 2.0, 3.0])
    }

    fn rhs_5() -> Array1<f64> {
        Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0])
    }

    fn verify_solution(a: &CsrMatrix<f64>, x: &Array1<f64>, b: &Array1<f64>, tol: f64) {
        let ax = spmv(a, x).expect("spmv failed in verification");
        for (i, (&axi, &bi)) in ax.iter().zip(b.iter()).enumerate() {
            assert!(
                (axi - bi).abs() < tol,
                "Mismatch at index {i}: Ax[{i}]={axi}, b[{i}]={bi}"
            );
        }
    }

    // --- CG tests ---

    #[test]
    fn test_cg_spd_3x3() {
        let a = spd_3x3();
        let b = rhs_3();
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, None).expect("CG failed");
        assert!(res.converged, "CG did not converge");
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_cg_spd_5x5() {
        let a = spd_5x5();
        let b = rhs_5();
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, None).expect("CG failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_cg_with_jacobi_precond() {
        let a = spd_3x3();
        let b = rhs_3();
        let pc = JacobiPreconditioner::new(&a).expect("Jacobi failed");
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, Some(&pc)).expect("CG + Jacobi failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_cg_zero_rhs() {
        let a = spd_3x3();
        let b = Array1::zeros(3);
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, None).expect("CG failed");
        assert!(res.converged);
        assert!(res.residual_norm <= 1e-14);
    }

    #[test]
    fn test_cg_dimension_mismatch() {
        let a = spd_3x3();
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let cfg = IterativeSolverConfig::default();
        assert!(cg(&a, &b, &cfg, None).is_err());
    }

    // --- BiCGSTAB tests ---

    #[test]
    fn test_bicgstab_nonsym_3x3() {
        let a = nonsym_3x3();
        let b = rhs_3();
        let cfg = IterativeSolverConfig::default();
        let res = bicgstab(&a, &b, &cfg, None).expect("BiCGSTAB failed");
        assert!(res.converged, "BiCGSTAB did not converge");
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_bicgstab_spd_3x3() {
        let a = spd_3x3();
        let b = rhs_3();
        let cfg = IterativeSolverConfig::default();
        let res = bicgstab(&a, &b, &cfg, None).expect("BiCGSTAB failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_bicgstab_with_jacobi() {
        let a = nonsym_3x3();
        let b = rhs_3();
        let pc = JacobiPreconditioner::new(&a).expect("Jacobi failed");
        let cfg = IterativeSolverConfig::default();
        let res = bicgstab(&a, &b, &cfg, Some(&pc)).expect("BiCGSTAB + Jacobi failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_bicgstab_zero_rhs() {
        let a = nonsym_3x3();
        let b = Array1::zeros(3);
        let cfg = IterativeSolverConfig::default();
        let res = bicgstab(&a, &b, &cfg, None).expect("BiCGSTAB failed");
        assert!(res.converged);
    }

    // --- GMRES tests ---

    #[test]
    fn test_gmres_nonsym_3x3() {
        let a = nonsym_3x3();
        let b = rhs_3();
        let cfg = IterativeSolverConfig::default();
        let res = gmres(&a, &b, &cfg, 30, None).expect("GMRES failed");
        assert!(res.converged, "GMRES did not converge");
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_gmres_spd_5x5() {
        let a = spd_5x5();
        let b = rhs_5();
        let cfg = IterativeSolverConfig::default();
        let res = gmres(&a, &b, &cfg, 10, None).expect("GMRES failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_gmres_with_jacobi() {
        let a = nonsym_3x3();
        let b = rhs_3();
        let pc = JacobiPreconditioner::new(&a).expect("Jacobi failed");
        let cfg = IterativeSolverConfig::default();
        let res = gmres(&a, &b, &cfg, 30, Some(&pc)).expect("GMRES + Jacobi failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_gmres_restart_small() {
        // Test with very small restart value (forces outer restarts)
        let a = spd_5x5();
        let b = rhs_5();
        let cfg = IterativeSolverConfig {
            max_iter: 200,
            tol: 1e-8,
            verbose: false,
        };
        let res = gmres(&a, &b, &cfg, 2, None).expect("GMRES failed");
        assert!(res.converged, "GMRES(2) did not converge");
        verify_solution(&a, &res.solution, &b, 1e-6);
    }

    // --- Chebyshev tests ---

    #[test]
    fn test_chebyshev_spd_3x3() {
        let a = spd_3x3();
        let b = rhs_3();
        // Eigenvalues: 2 (once), 5 (twice). Use bounds that bracket them.
        let cfg = IterativeSolverConfig {
            max_iter: 200,
            tol: 1e-8,
            verbose: false,
        };
        let res = chebyshev(&a, &b, &cfg, 1.5, 5.5).expect("Chebyshev failed");
        assert!(res.converged, "Chebyshev did not converge");
        verify_solution(&a, &res.solution, &b, 1e-6);
    }

    #[test]
    fn test_chebyshev_spd_5x5() {
        let a = spd_5x5();
        let b = rhs_5();
        // Tridiagonal 4,-1,-1: eigenvalues in [4 - 2cos(pi/6), 4 + 2cos(pi/6)] ~ [2.27, 5.73]
        let cfg = IterativeSolverConfig {
            max_iter: 300,
            tol: 1e-8,
            verbose: false,
        };
        let res = chebyshev(&a, &b, &cfg, 2.0, 6.0).expect("Chebyshev failed");
        assert!(res.converged, "Chebyshev did not converge");
        verify_solution(&a, &res.solution, &b, 1e-6);
    }

    #[test]
    fn test_chebyshev_invalid_bounds() {
        let a = spd_3x3();
        let b = rhs_3();
        let cfg = IterativeSolverConfig::default();
        // lambda_min >= lambda_max should fail
        assert!(chebyshev(&a, &b, &cfg, 5.0, 3.0).is_err());
        // Negative lambda_min should fail
        assert!(chebyshev(&a, &b, &cfg, -1.0, 5.0).is_err());
    }

    // --- Preconditioner tests ---

    #[test]
    fn test_jacobi_from_matrix() {
        let a = spd_3x3();
        let pc = JacobiPreconditioner::new(&a).expect("Jacobi creation failed");
        let r = Array1::from_vec(vec![4.0, 8.0, 12.0]);
        let z = pc.apply(&r).expect("Jacobi apply failed");
        // diagonal is 4.0, so z = r/4
        assert!((z[0] - 1.0).abs() < 1e-12);
        assert!((z[1] - 2.0).abs() < 1e-12);
        assert!((z[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_ilu0_preconditioner() {
        let a = spd_3x3();
        let pc = ILU0Preconditioner::new(&a).expect("ILU0 creation failed");
        let r = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let z = pc.apply(&r).expect("ILU0 apply failed");
        // ILU(0) on a dense 3x3 is the exact LU, so M^{-1}r = A^{-1}r
        let a_inv_r = spmv(&a, &z).expect("spmv failed");
        for i in 0..3 {
            assert!(
                (a_inv_r[i] - r[i]).abs() < 1e-10,
                "ILU0 did not produce exact inverse on dense matrix at index {i}"
            );
        }
    }

    #[test]
    fn test_ssor_preconditioner() {
        let a = spd_3x3();
        let pc = SSORPreconditioner::new(a.clone(), 1.0).expect("SSOR creation failed");
        let r = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let z = pc.apply(&r).expect("SSOR apply failed");
        // Just check that the output is finite and has the right length
        assert_eq!(z.len(), 3);
        for &val in z.iter() {
            assert!(val.is_finite(), "SSOR produced non-finite value");
        }
    }

    #[test]
    fn test_ssor_invalid_omega() {
        let a = spd_3x3();
        assert!(SSORPreconditioner::new(a.clone(), 0.0).is_err());
        assert!(SSORPreconditioner::new(a.clone(), 2.0).is_err());
        assert!(SSORPreconditioner::new(a.clone(), -0.5).is_err());
    }

    #[test]
    fn test_cg_with_ilu0() {
        let a = spd_3x3();
        let b = rhs_3();
        let pc = ILU0Preconditioner::new(&a).expect("ILU0 creation failed");
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, Some(&pc)).expect("CG + ILU0 failed");
        assert!(res.converged, "CG + ILU0 did not converge");
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    // --- Utility function tests ---

    #[test]
    fn test_sparse_diagonal() {
        let a = spd_3x3();
        let d = sparse_diagonal(&a);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 4.0).abs() < 1e-12);
        assert!((d[1] - 4.0).abs() < 1e-12);
        assert!((d[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_trace() {
        let a = spd_3x3();
        let tr = sparse_trace(&a);
        assert!((tr - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_norm_frobenius() {
        let a = spd_3x3();
        // ||A||_F = sqrt(sum of squares of all elements)
        // 3*16 + 6*1 = 48+6 = 54,  sqrt(54) ~ 7.3484692...
        let nf = sparse_norm(&a, NormType::Frobenius);
        assert!((nf - 54.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_norm_inf() {
        let a = spd_3x3();
        // Each row sums to |4| + |-1| + |-1| = 6
        let ni = sparse_norm(&a, NormType::Inf);
        assert!((ni - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_sparse_norm_one() {
        let a = spd_3x3();
        // Each column sums to |4| + |-1| + |-1| = 6
        let n1 = sparse_norm(&a, NormType::One);
        assert!((n1 - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_estimate_spectral_radius() {
        let a = spd_3x3();
        // Eigenvalues of [[4,-1,-1],[-1,4,-1],[-1,-1,4]]: 2 (once), 5 (twice)
        // Spectral radius = 5.0
        let rho = estimate_spectral_radius(&a, 100).expect("spectral radius estimation failed");
        assert!(
            (rho - 5.0).abs() < 0.5,
            "Expected spectral radius near 5.0, got {rho}"
        );
    }

    #[test]
    fn test_sparse_diagonal_rectangular() {
        // Test diagonal extraction on non-square (4x3) matrix
        let rows = vec![0, 1, 2, 3];
        let cols = vec![0, 1, 2, 0];
        let data = vec![10.0, 20.0, 30.0, 99.0];
        let a = CsrMatrix::new(data, rows, cols, (4, 3)).expect("failed to build matrix");
        let d = sparse_diagonal(&a);
        assert_eq!(d.len(), 3);
        assert!((d[0] - 10.0).abs() < 1e-12);
        assert!((d[1] - 20.0).abs() < 1e-12);
        assert!((d[2] - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_solver_config_default() {
        let cfg = IterativeSolverConfig::default();
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tol - 1e-10).abs() < 1e-15);
        assert!(!cfg.verbose);
    }

    #[test]
    fn test_gmres_dimension_mismatch() {
        let a = spd_3x3();
        let b = Array1::from_vec(vec![1.0, 2.0]);
        let cfg = IterativeSolverConfig::default();
        assert!(gmres(&a, &b, &cfg, 10, None).is_err());
    }

    #[test]
    fn test_bicgstab_5x5() {
        let a = spd_5x5();
        let b = rhs_5();
        let cfg = IterativeSolverConfig::default();
        let res = bicgstab(&a, &b, &cfg, None).expect("BiCGSTAB failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_cg_with_ssor_precond() {
        let a = spd_5x5();
        let b = rhs_5();
        let pc = SSORPreconditioner::new(a.clone(), 1.2).expect("SSOR creation failed");
        let cfg = IterativeSolverConfig::default();
        let res = cg(&a, &b, &cfg, Some(&pc)).expect("CG + SSOR failed");
        assert!(res.converged);
        verify_solution(&a, &res.solution, &b, 1e-8);
    }

    #[test]
    fn test_nonsquare_matrix_error() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 0, 1];
        let data = vec![1.0, 2.0, 3.0];
        let a = CsrMatrix::new(data, rows, cols, (3, 2)).expect("failed to build matrix");
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let cfg = IterativeSolverConfig::default();
        assert!(cg(&a, &b, &cfg, None).is_err());
        assert!(bicgstab(&a, &b, &cfg, None).is_err());
        assert!(gmres(&a, &b, &cfg, 10, None).is_err());
    }
}
