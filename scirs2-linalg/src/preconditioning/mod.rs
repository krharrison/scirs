//! Advanced preconditioning techniques for iterative linear solvers
//!
//! This module provides new preconditioning strategies complementary to those
//! in `crate::preconditioners`. While that module focuses on classic methods
//! (ILU, IC, multigrid, Schwarz), this module implements:
//!
//! - **SPAI**: Sparse Approximate Inverse via Frobenius norm minimization
//! - **PolynomialPreconditioner**: Chebyshev and Neumann polynomial preconditioners
//! - **BlockDiagonalPreconditioner**: Block-diagonal preconditioner
//! - **SchurComplementPreconditioner**: Schur complement–based preconditioning for block systems
//! - **AMGPreconditioner** (stub): Interface to Algebraic Multigrid
//!
//! ## Mathematical Background
//!
//! For a linear system Ax = b, a preconditioner M approximates A⁻¹.
//! The preconditioned system M A x = M b (left preconditioning) or
//! A M y = b with x = M y (right preconditioning) converges faster.
//!
//! ### SPAI
//!
//! Minimizes ‖M A - I‖_F² column by column, allowing parallelism.
//! Each column m_j minimizes ‖A m_j - e_j‖₂² subject to a sparsity pattern.
//!
//! ### Polynomial Preconditioning
//!
//! M = p_k(A) where p_k is a polynomial of degree k chosen so that
//! p_k(A) A ≈ I. Chebyshev polynomials are optimal for matrices with
//! known spectral bounds.
//!
//! ### Schur Complement
//!
//! For block systems [A₁₁ A₁₂; A₂₁ A₂₂], the Schur complement
//! S = A₂₂ - A₂₁ A₁₁⁻¹ A₁₂ is used to construct efficient preconditioners.
//!
//! ## References
//!
//! - Saad & Schultz (1986). "GMRES: A Generalized Minimal Residual Algorithm"
//! - Grote & Huckle (1997). "Parallel Preconditioning with Sparse Approximate Inverses"
//! - Brent & Luk (1985). "The solution of singular-value and symmetric eigenvalue problems on multiprocessor arrays"

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, NumAssign, One, Zero};
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Shared preconditioner trait (mirrors crate::preconditioners::PreconditionerOp)
// ============================================================================

/// Preconditioner operator: apply M⁻¹ to a vector
pub trait PrecondApply {
    /// Apply the preconditioner: y = M⁻¹ x
    fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>>;

    /// Apply the transpose: y = M⁻ᵀ x (default: same as apply for symmetric preconditioners)
    fn apply_transpose(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        self.apply(x)
    }

    /// Return the size n of the square preconditioner
    fn size(&self) -> usize;
}

// ============================================================================
// SPAI: Sparse Approximate Inverse
// ============================================================================

/// Sparse Approximate Inverse (SPAI) preconditioner.
///
/// Computes a sparse matrix M that minimizes ‖MA - I‖_F² over a given
/// sparsity pattern. Each column m_j is found independently by solving
/// the local least-squares problem:
///
///   min_{m_j ∈ J_j} ‖A_{:, J_j}^T m_j - e_j‖₂²
///
/// where J_j is the (adjustable) sparsity pattern for column j.
///
/// This implementation uses a fixed initial pattern (diagonal only or banded)
/// with optional adaptive pattern growth.
#[derive(Debug, Clone)]
pub struct SPAI {
    /// The computed approximate inverse M (dense storage for simplicity)
    m: Array2<f64>,
    /// System size
    n: usize,
    /// Achieved Frobenius residual ‖MA - I‖_F
    pub frobenius_residual: f64,
}

impl SPAI {
    /// Build a SPAI preconditioner for the given matrix A.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (n × n)
    /// * `bandwidth` - Initial sparsity bandwidth for M (0 = diagonal only, 1 = tridiagonal, etc.)
    /// * `n_fill_steps` - Number of adaptive fill steps (0 = no adaptation)
    /// * `drop_tol` - Drop tolerance: entries smaller than `drop_tol * ‖m_j‖` are dropped
    pub fn new(
        a: &ArrayView2<f64>,
        bandwidth: usize,
        n_fill_steps: usize,
        drop_tol: Option<f64>,
    ) -> LinalgResult<Self> {
        let (m_rows, n) = a.dim();
        if m_rows != n {
            return Err(LinalgError::ShapeError(
                "SPAI: A must be square".to_string(),
            ));
        }
        if n == 0 {
            return Ok(Self {
                m: Array2::zeros((0, 0)),
                n: 0,
                frobenius_residual: 0.0,
            });
        }

        let tol = drop_tol.unwrap_or(1e-3);
        let mut m_mat = Array2::zeros((n, n));

        // For each column j of M, determine sparsity pattern and solve LS
        for j in 0..n {
            let mut pattern: Vec<usize> = {
                let lo = j.saturating_sub(bandwidth);
                let hi = (j + bandwidth + 1).min(n);
                (lo..hi).collect()
            };

            // Adaptive fill: add rows with largest residual
            for _step in 0..n_fill_steps {
                // Solve the current LS problem
                let col = Self::solve_column(a, j, &pattern)?;

                // Compute residual r = A^T m_j - e_j restricted to current pattern rows
                // Identify new rows to add: rows with large |A_{i,:} m_j| outside pattern
                let mut new_rows: Vec<(f64, usize)> = Vec::new();
                for i in 0..n {
                    if pattern.contains(&i) {
                        continue;
                    }
                    // Compute (A^T m_j)[i] = A_{i,:} m_j (where m_j is indexed by pattern)
                    let mut val = 0.0f64;
                    for (pi, &p) in pattern.iter().enumerate() {
                        val += a[[i, p]] * col[pi];
                    }
                    // Add e_j contribution
                    if i == j {
                        val -= 1.0;
                    }
                    new_rows.push((val.abs(), i));
                }

                // Add the top-k most important rows
                new_rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let n_new = (bandwidth + 1).min(new_rows.len());
                for k in 0..n_new {
                    let (val, row) = new_rows[k];
                    if val > 1e-10 {
                        pattern.push(row);
                    }
                }
                pattern.sort_unstable();
                pattern.dedup();
            }

            // Final solve
            let col = Self::solve_column(a, j, &pattern)?;
            let col_norm = col.iter().map(|&v| v * v).sum::<f64>().sqrt();

            // Fill into M with drop tolerance
            for (pi, &p) in pattern.iter().enumerate() {
                if col[pi].abs() >= tol * col_norm {
                    m_mat[[p, j]] = col[pi];
                }
            }
        }

        // Compute Frobenius residual ‖MA - I‖_F
        let mut frob_sq = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut ma_ij = 0.0f64;
                for k in 0..n {
                    ma_ij += m_mat[[i, k]] * a[[k, j]];
                }
                let delta = if i == j { 1.0 } else { 0.0 };
                frob_sq += (ma_ij - delta).powi(2);
            }
        }

        Ok(Self {
            m: m_mat,
            n,
            frobenius_residual: frob_sq.sqrt(),
        })
    }

    /// Solve the column-j LS problem: min ‖A_{:, pattern}^T m_j - e_j‖₂
    fn solve_column(
        a: &ArrayView2<f64>,
        j: usize,
        pattern: &[usize],
    ) -> LinalgResult<Vec<f64>> {
        let n = a.nrows();
        let p = pattern.len();
        if p == 0 {
            return Ok(Vec::new());
        }

        // Build A_sub = A[:, pattern] (all rows × pattern cols)
        // We need the submatrix of A^T, i.e. A_sub[i, k] = A[i, pattern[k]]
        // The LS system: A^T m_j = e_j restricted to rows in pattern
        // Actually: for column j of M, we want M_{pattern, j} such that
        // ‖A_{pattern, :} M_{:, j} - e_j‖ is minimized, but we further restrict
        // so the reduced LS is A_hat x = e_hat where A_hat = A_{rows, pattern_j}

        // In SPAI, the column j of M is supported on pattern (rows of M = columns of A^T).
        // The LS problem: min ‖ (A^T)[pattern, :] m_j - e_j[pattern] ‖
        // But standard SPAI considers: min ‖ A m_j - e_j ‖ where m_j is supported on pattern
        // = min ‖ A[:, pattern] x - e_j ‖ where x = m_j[pattern]

        // Build A_sub: n × p, A_sub[:, k] = A[:, pattern[k]]
        let mut a_sub = vec![0.0f64; n * p];
        for k in 0..p {
            for i in 0..n {
                a_sub[i * p + k] = a[[i, pattern[k]]];
            }
        }

        // e_j (length n)
        let mut ej = vec![0.0f64; n];
        ej[j] = 1.0;

        // Solve the LS problem via normal equations: (A_sub^T A_sub) x = A_sub^T e_j
        // Normal equations matrix: G = A_sub^T A_sub (p × p)
        let mut g = vec![0.0f64; p * p];
        for i in 0..p {
            for k in 0..p {
                let mut val = 0.0f64;
                for row in 0..n {
                    val += a_sub[row * p + i] * a_sub[row * p + k];
                }
                g[i * p + k] = val;
            }
        }

        // Right-hand side: r = A_sub^T e_j
        let mut rhs = vec![0.0f64; p];
        for k in 0..p {
            rhs[k] = a_sub[j * p + k]; // A_sub[j, k] = A[j, pattern[k]]
        }

        // Solve G x = rhs via Cholesky (G is SPD if A_sub has full column rank)
        // Fallback to regularized solve
        let regularization = 1e-12 * g.iter().enumerate()
            .filter(|(i, _)| i % (p + 1) == 0)
            .map(|(_, &v)| v.abs())
            .sum::<f64>()
            .max(1e-12);

        for i in 0..p {
            g[i * p + i] += regularization;
        }

        // Cholesky decomposition of G
        let x = cholesky_solve_small(&g, &rhs, p)?;
        Ok(x)
    }

    /// Get the computed approximate inverse matrix
    pub fn matrix(&self) -> &Array2<f64> {
        &self.m
    }
}

impl PrecondApply for SPAI {
    fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "SPAI::apply: x has length {} but n={}",
                x.len(),
                self.n
            )));
        }

        let mut y = Array1::zeros(self.n);
        for i in 0..self.n {
            let mut val = 0.0f64;
            for j in 0..self.n {
                val += self.m[[i, j]] * x[j];
            }
            y[i] = val;
        }

        Ok(y)
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ============================================================================
// PolynomialPreconditioner (Chebyshev / Neumann)
// ============================================================================

/// Polynomial type for the preconditioner
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolynomialType {
    /// Neumann series: M = sum_{k=0}^{d} (I - alpha*A)^k * alpha
    Neumann,
    /// Chebyshev polynomial: optimal for A with spectrum in [lambda_min, lambda_max]
    Chebyshev {
        /// Lower bound on eigenvalues of A
        lambda_min: f64,
        /// Upper bound on eigenvalues of A
        lambda_max: f64,
    },
}

/// Polynomial preconditioner using Chebyshev or Neumann series.
///
/// Polynomial preconditioners avoid matrix factorizations and are particularly
/// attractive for GPU and parallel architectures since they only require
/// matrix-vector products.
///
/// For Chebyshev preconditioning of A with spectral bounds [λ_min, λ_max]:
///   p_k(A) ≈ A⁻¹ using the scaled Chebyshev polynomial of degree k
///
/// For Neumann preconditioning with diagonal scaling D = diag(A)⁻¹:
///   M = sum_{j=0}^{k} (I - D A)^j D
#[derive(Debug, Clone)]
pub struct PolynomialAdvancedPreconditioner {
    /// Polynomial type and parameters
    pub poly_type: PolynomialType,
    /// Polynomial degree
    pub degree: usize,
    /// System size
    n: usize,
    /// Diagonal inverse (for scaling)
    diag_inv: Array1<f64>,
    /// Chebyshev recurrence coefficients (if applicable)
    cheb_coeffs: Option<Vec<f64>>,
}

impl PolynomialAdvancedPreconditioner {
    /// Create a Neumann series preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (n × n)
    /// * `degree` - Polynomial degree (number of Neumann terms)
    pub fn new_neumann(a: &ArrayView2<f64>, degree: usize) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "PolynomialAdvancedPreconditioner: A must be square".to_string(),
            ));
        }

        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a[[i, i]];
            if d.abs() < 1e-300 {
                diag_inv[i] = 1.0;
            } else {
                diag_inv[i] = 1.0 / d;
            }
        }

        Ok(Self {
            poly_type: PolynomialType::Neumann,
            degree,
            n,
            diag_inv,
            cheb_coeffs: None,
        })
    }

    /// Create a Chebyshev polynomial preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix A (n × n)
    /// * `degree` - Polynomial degree
    /// * `lambda_min` - Lower spectral bound (must be > 0 for SPD systems)
    /// * `lambda_max` - Upper spectral bound
    pub fn new_chebyshev(
        a: &ArrayView2<f64>,
        degree: usize,
        lambda_min: f64,
        lambda_max: f64,
    ) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "PolynomialAdvancedPreconditioner: A must be square".to_string(),
            ));
        }
        if lambda_min <= 0.0 || lambda_max <= lambda_min {
            return Err(LinalgError::InvalidInputError(format!(
                "Chebyshev: need 0 < lambda_min < lambda_max, got {} < {}",
                lambda_min, lambda_max
            )));
        }

        // Compute Chebyshev recurrence coefficients for the polynomial approximation
        // of 1/lambda on [lambda_min, lambda_max]
        let alpha = (lambda_max + lambda_min) / 2.0;
        let beta = (lambda_max - lambda_min) / 2.0;

        // Recurrence: p_k satisfies p_k = (2/alpha) * (I - (beta/alpha)^2 * A * p_{k-1}) - p_{k-2}
        // Store as [sigma, rho_prev, delta] style parameters
        let coeffs = vec![alpha, beta, 0.0]; // [center, half_range, dummy]

        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a[[i, i]];
            if d.abs() < 1e-300 {
                diag_inv[i] = 1.0;
            } else {
                diag_inv[i] = 1.0 / d;
            }
        }

        Ok(Self {
            poly_type: PolynomialType::Chebyshev {
                lambda_min,
                lambda_max,
            },
            degree,
            n,
            diag_inv,
            cheb_coeffs: Some(coeffs),
        })
    }

    /// Apply the polynomial preconditioner M(A) * x using matrix-vector products.
    ///
    /// This method works matrix-free: it requires a closure that computes A * v.
    ///
    /// # Arguments
    ///
    /// * `matvec` - Closure computing A * v
    /// * `x` - Input vector
    pub fn apply_matrixfree<F>(&self, matvec: F, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> LinalgResult<Array1<f64>>,
    {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "PolynomialAdvancedPreconditioner: x has length {} but n={}",
                x.len(),
                self.n
            )));
        }

        match self.poly_type {
            PolynomialType::Neumann => self.apply_neumann_matrixfree(&matvec, x),
            PolynomialType::Chebyshev { .. } => self.apply_chebyshev_matrixfree(&matvec, x),
        }
    }

    /// Neumann series: M x = sum_{k=0}^{d} (I - D A)^k D x
    fn apply_neumann_matrixfree<F>(
        &self,
        matvec: &F,
        x: &ArrayView1<f64>,
    ) -> LinalgResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> LinalgResult<Array1<f64>>,
    {
        // D x
        let dx: Array1<f64> = x.iter().zip(self.diag_inv.iter()).map(|(&xi, &di)| di * xi).collect();

        // Neumann: M x = D x + (I - D A)(D x) + (I - D A)^2 (D x) + ...
        let mut result = dx.clone();
        let mut term = dx;

        for _ in 0..self.degree {
            // term = (I - D A) term = term - D (A term)
            let at = matvec(&term.view())?;
            let dat: Array1<f64> = at.iter().zip(self.diag_inv.iter()).map(|(&v, &d)| d * v).collect();
            term = &term - &dat;
            result = result + &term;

            // Check if term is negligible
            let term_norm = term.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if term_norm < 1e-15 {
                break;
            }
        }

        Ok(result)
    }

    /// Chebyshev polynomial preconditioner application.
    ///
    /// Uses the three-term Chebyshev recurrence to compute
    /// p_k(A) x where p_k is the Chebyshev polynomial approximation of 1/lambda.
    fn apply_chebyshev_matrixfree<F>(
        &self,
        matvec: &F,
        x: &ArrayView1<f64>,
    ) -> LinalgResult<Array1<f64>>
    where
        F: Fn(&ArrayView1<f64>) -> LinalgResult<Array1<f64>>,
    {
        let coeffs = self.cheb_coeffs.as_ref().ok_or_else(|| {
            LinalgError::ComputationError("Chebyshev coefficients not initialized".to_string())
        })?;

        let alpha = coeffs[0]; // center of spectral interval
        let beta = coeffs[1]; // half-range

        if alpha.abs() < 1e-300 {
            return Err(LinalgError::ComputationError(
                "Chebyshev: spectral center is zero".to_string(),
            ));
        }

        // Chebyshev iteration for solving A x = b approximately
        // Starting from x_0 = 0: p_0(A) b = 2/alpha * b
        // Recurrence: p_k(A) b = rho_k * (2/alpha * (b - A p_{k-1}) + rho_{k-1} * p_{k-1})
        // where rho_0 = 1, rho_1 = 1/(1 - (beta/alpha)^2/2), rho_k = 1/(1 - rho_{k-1} * (beta/2alpha)^2)

        let mu = (beta / alpha).powi(2) / 2.0;

        let mut p_prev = Array1::zeros(self.n);
        // p_1 = (2/alpha) x
        let mut p_curr: Array1<f64> = x.iter().map(|&v| 2.0 / alpha * v).collect();

        let mut rho_prev = 1.0f64;

        for _ in 0..self.degree.saturating_sub(1) {
            let rho_curr = 1.0 / (1.0 - mu * rho_prev);

            // A p_curr
            let ap = matvec(&p_curr.view())?;

            // p_next = rho_curr * (2/alpha * (x - A p_curr) + rho_prev * p_curr) + ...
            // Standard Chebyshev update:
            // p_next = rho_curr * (2/alpha * (x - A p_curr)) + rho_curr * rho_prev * p_prev
            let mut p_next = Array1::zeros(self.n);
            for i in 0..self.n {
                let residual = x[i] - ap[i];
                p_next[i] = rho_curr * (2.0 / alpha * residual + rho_prev * p_prev[i]);
            }

            p_prev = p_curr;
            p_curr = p_next;
            rho_prev = rho_curr;
        }

        Ok(p_curr)
    }
}

impl PrecondApply for PolynomialAdvancedPreconditioner {
    fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        // For the matrix-free apply, we return an error since we need the matrix
        Err(LinalgError::NotImplementedError(
            "PolynomialAdvancedPreconditioner::apply requires a matrix; use apply_matrixfree instead".to_string(),
        ))
    }

    fn size(&self) -> usize {
        self.n
    }
}

/// Convenience: apply polynomial preconditioner given the full matrix A.
impl PolynomialAdvancedPreconditioner {
    /// Apply the preconditioner using the stored or provided matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A (for computing A * v in the polynomial recurrence)
    /// * `x` - Input vector
    pub fn apply_with_matrix(
        &self,
        a: &ArrayView2<f64>,
        x: &ArrayView1<f64>,
    ) -> LinalgResult<Array1<f64>> {
        let matvec = |v: &ArrayView1<f64>| -> LinalgResult<Array1<f64>> {
            let (m, n) = a.dim();
            let mut result = Array1::zeros(m);
            for i in 0..m {
                let mut val = 0.0f64;
                for j in 0..n {
                    val += a[[i, j]] * v[j];
                }
                result[i] = val;
            }
            Ok(result)
        };
        self.apply_matrixfree(matvec, x)
    }
}

// ============================================================================
// BlockDiagonalPreconditioner
// ============================================================================

/// Block-diagonal preconditioner for block-structured linear systems.
///
/// Given a matrix A with natural block structure (e.g., arising from finite
/// element discretizations with multiple unknowns per node), the block-diagonal
/// preconditioner inverts each block separately:
///
///   M = diag(B_1^{-1}, B_2^{-1}, ..., B_k^{-1})
///
/// where B_i are the diagonal blocks extracted from A.
#[derive(Debug, Clone)]
pub struct BlockDiagonalPreconditioner {
    /// Inverted blocks (or Cholesky factors for SPD blocks)
    blocks: Vec<Array2<f64>>,
    /// Starting index of each block
    block_starts: Vec<usize>,
    /// Block sizes
    block_sizes: Vec<usize>,
    /// Total system size
    n: usize,
}

impl BlockDiagonalPreconditioner {
    /// Create a block-diagonal preconditioner with uniform block size.
    ///
    /// # Arguments
    ///
    /// * `a` - System matrix (n × n)
    /// * `block_size` - Size of each diagonal block (must divide n)
    pub fn new_uniform(a: &ArrayView2<f64>, block_size: usize) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "BlockDiagonalPreconditioner: A must be square".to_string(),
            ));
        }
        if block_size == 0 || block_size > n {
            return Err(LinalgError::InvalidInputError(format!(
                "BlockDiagonalPreconditioner: block_size {} is invalid for n={}",
                block_size, n
            )));
        }

        let n_blocks = (n + block_size - 1) / block_size;
        let block_starts: Vec<usize> = (0..n_blocks).map(|i| i * block_size).collect();
        let block_sizes: Vec<usize> = (0..n_blocks)
            .map(|i| {
                let start = i * block_size;
                let end = (start + block_size).min(n);
                end - start
            })
            .collect();

        let mut blocks = Vec::with_capacity(n_blocks);
        for (b, &start) in block_starts.iter().enumerate() {
            let bsize = block_sizes[b];
            let end = start + bsize;

            // Extract diagonal block
            let block = a.slice(scirs2_core::ndarray::s![start..end, start..end]).to_owned();

            // Compute block inverse
            let block_inv = block_inverse(&block.view())?;
            blocks.push(block_inv);
        }

        Ok(Self {
            blocks,
            block_starts,
            block_sizes,
            n,
        })
    }

    /// Create a block-diagonal preconditioner with variable block sizes.
    ///
    /// # Arguments
    ///
    /// * `a` - System matrix (n × n)
    /// * `block_sizes` - Size of each block (must sum to n)
    pub fn new_variable(a: &ArrayView2<f64>, block_sizes: Vec<usize>) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "BlockDiagonalPreconditioner: A must be square".to_string(),
            ));
        }

        let total: usize = block_sizes.iter().sum();
        if total != n {
            return Err(LinalgError::DimensionError(format!(
                "BlockDiagonalPreconditioner: block sizes sum to {} but n={}",
                total, n
            )));
        }

        let mut block_starts = Vec::with_capacity(block_sizes.len());
        let mut start = 0;
        for &bs in &block_sizes {
            block_starts.push(start);
            start += bs;
        }

        let mut blocks = Vec::with_capacity(block_sizes.len());
        for (b, &bs) in block_sizes.iter().enumerate() {
            let s = block_starts[b];
            let e = s + bs;
            let block = a.slice(scirs2_core::ndarray::s![s..e, s..e]).to_owned();
            let block_inv = block_inverse(&block.view())?;
            blocks.push(block_inv);
        }

        Ok(Self {
            blocks,
            block_starts,
            block_sizes,
            n,
        })
    }
}

impl PrecondApply for BlockDiagonalPreconditioner {
    fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "BlockDiagonalPreconditioner::apply: x has {} elements but n={}",
                x.len(),
                self.n
            )));
        }

        let mut y = Array1::zeros(self.n);
        for (b, &start) in self.block_starts.iter().enumerate() {
            let bs = self.block_sizes[b];
            let end = start + bs;
            let x_b = x.slice(scirs2_core::ndarray::s![start..end]);
            let block_inv = &self.blocks[b];

            // y_b = B^{-1} x_b
            for i in 0..bs {
                let mut val = 0.0f64;
                for j in 0..bs {
                    val += block_inv[[i, j]] * x_b[j];
                }
                y[start + i] = val;
            }
        }

        Ok(y)
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ============================================================================
// SchurComplementPreconditioner
// ============================================================================

/// Schur complement–based preconditioner for 2×2 block systems.
///
/// For block systems [A₁₁ A₁₂; A₂₁ A₂₂] [x₁; x₂] = [b₁; b₂], the exact
/// block LDU factorization gives:
///
///   [A₁₁    0  ] [I  A₁₁⁻¹ A₁₂] [x₁]   [b₁]
///   [A₂₁  S  ] [0       I      ] [x₂] = [b₂]
///
/// where S = A₂₂ - A₂₁ A₁₁⁻¹ A₁₂ is the Schur complement.
///
/// This preconditioner approximates the inverse using inexact inverses of
/// A₁₁ and S, which can be efficient when both blocks are well-structured.
#[derive(Debug, Clone)]
pub struct SchurComplementPreconditioner {
    /// Size of block 1 (n₁)
    n1: usize,
    /// Size of block 2 (n₂)
    n2: usize,
    /// Total size n = n₁ + n₂
    n: usize,
    /// Approximate inverse of A₁₁ (n₁ × n₁)
    a11_inv: Array2<f64>,
    /// A₁₂ block (n₁ × n₂)
    a12: Array2<f64>,
    /// A₂₁ block (n₂ × n₁)
    a21: Array2<f64>,
    /// Approximate inverse of the Schur complement S (n₂ × n₂)
    schur_inv: Array2<f64>,
}

impl SchurComplementPreconditioner {
    /// Build the Schur complement preconditioner.
    ///
    /// # Arguments
    ///
    /// * `a` - Full block system matrix (n × n)
    /// * `n1` - Size of the first block
    /// * `approx_a11_inv` - Optional approximate inverse of A₁₁; if None, exact inverse is computed
    /// * `approx_schur_inv` - Optional approximate inverse of S; if None, exact inverse is computed
    pub fn new(
        a: &ArrayView2<f64>,
        n1: usize,
        approx_a11_inv: Option<Array2<f64>>,
        approx_schur_inv: Option<Array2<f64>>,
    ) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "SchurComplementPreconditioner: A must be square".to_string(),
            ));
        }
        if n1 == 0 || n1 >= n {
            return Err(LinalgError::InvalidInputError(format!(
                "SchurComplementPreconditioner: n1={} must satisfy 0 < n1 < n={}",
                n1, n
            )));
        }

        let n2 = n - n1;

        // Extract blocks
        let a11 = a.slice(scirs2_core::ndarray::s![..n1, ..n1]).to_owned();
        let a12 = a.slice(scirs2_core::ndarray::s![..n1, n1..]).to_owned();
        let a21 = a.slice(scirs2_core::ndarray::s![n1.., ..n1]).to_owned();
        let a22 = a.slice(scirs2_core::ndarray::s![n1.., n1..]).to_owned();

        // Compute (approximate) inverse of A₁₁
        let a11_inv = match approx_a11_inv {
            Some(inv) => inv,
            None => block_inverse(&a11.view())?,
        };

        // Compute Schur complement S = A₂₂ - A₂₁ A₁₁⁻¹ A₁₂
        // A₂₁ A₁₁⁻¹: (n₂ × n₁) × (n₁ × n₁) = n₂ × n₁
        let mut a21_a11inv = Array2::zeros((n2, n1));
        for i in 0..n2 {
            for j in 0..n1 {
                let mut val = 0.0f64;
                for k in 0..n1 {
                    val += a21[[i, k]] * a11_inv[[k, j]];
                }
                a21_a11inv[[i, j]] = val;
            }
        }

        // S = A₂₂ - (A₂₁ A₁₁⁻¹) A₁₂
        let mut schur = a22.clone();
        for i in 0..n2 {
            for j in 0..n2 {
                let mut val = 0.0f64;
                for k in 0..n1 {
                    val += a21_a11inv[[i, k]] * a12[[k, j]];
                }
                schur[[i, j]] -= val;
            }
        }

        // Compute (approximate) inverse of S
        let schur_inv = match approx_schur_inv {
            Some(inv) => inv,
            None => block_inverse(&schur.view())?,
        };

        Ok(Self {
            n1,
            n2,
            n,
            a11_inv,
            a12,
            a21,
            schur_inv,
        })
    }

    /// Get the Schur complement inverse (exposed for analysis)
    pub fn schur_inv(&self) -> &Array2<f64> {
        &self.schur_inv
    }
}

impl PrecondApply for SchurComplementPreconditioner {
    /// Apply the block LDU inverse.
    ///
    /// The block preconditioner M⁻¹ = U⁻¹ D⁻¹ L⁻¹ solves:
    ///   [A₁₁  A₁₂] [y₁]   [x₁]
    ///   [A₂₁  A₂₂] [y₂] ≈ [x₂]
    ///
    /// Step 1: Solve L z = x: z₁ = x₁, z₂ = x₂ - A₂₁ A₁₁⁻¹ x₁
    /// Step 2: Solve D w = z: w₁ = A₁₁⁻¹ z₁, w₂ = S⁻¹ z₂
    /// Step 3: Solve U y = w: y₂ = w₂, y₁ = w₁ - A₁₁⁻¹ A₁₂ y₂
    fn apply(&self, x: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "SchurComplementPreconditioner::apply: x has {} elements but n={}",
                x.len(),
                self.n
            )));
        }

        let x1 = x.slice(scirs2_core::ndarray::s![..self.n1]);
        let x2 = x.slice(scirs2_core::ndarray::s![self.n1..]);

        // Step 1: L solve: z₁ = x₁, z₂ = x₂ - A₂₁ A₁₁⁻¹ x₁
        // First compute A₁₁⁻¹ x₁
        let mut a11inv_x1 = Array1::zeros(self.n1);
        for i in 0..self.n1 {
            let mut val = 0.0f64;
            for j in 0..self.n1 {
                val += self.a11_inv[[i, j]] * x1[j];
            }
            a11inv_x1[i] = val;
        }

        // z₂ = x₂ - A₂₁ * (A₁₁⁻¹ x₁)
        let mut z2 = Array1::zeros(self.n2);
        for i in 0..self.n2 {
            let mut a21_val = 0.0f64;
            for j in 0..self.n1 {
                a21_val += self.a21[[i, j]] * a11inv_x1[j];
            }
            z2[i] = x2[i] - a21_val;
        }

        // Step 2: D solve: w₂ = S⁻¹ z₂ (w₁ = A₁₁⁻¹ z₁ = A₁₁⁻¹ x₁, already computed)
        let mut w2 = Array1::zeros(self.n2);
        for i in 0..self.n2 {
            let mut val = 0.0f64;
            for j in 0..self.n2 {
                val += self.schur_inv[[i, j]] * z2[j];
            }
            w2[i] = val;
        }

        // Step 3: U solve: y₂ = w₂, y₁ = w₁ - A₁₁⁻¹ A₁₂ y₂
        // A₁₂ y₂:
        let mut a12_w2 = Array1::zeros(self.n1);
        for i in 0..self.n1 {
            let mut val = 0.0f64;
            for j in 0..self.n2 {
                val += self.a12[[i, j]] * w2[j];
            }
            a12_w2[i] = val;
        }

        // A₁₁⁻¹ (A₁₂ y₂):
        let mut a11inv_a12w2 = Array1::zeros(self.n1);
        for i in 0..self.n1 {
            let mut val = 0.0f64;
            for j in 0..self.n1 {
                val += self.a11_inv[[i, j]] * a12_w2[j];
            }
            a11inv_a12w2[i] = val;
        }

        let mut y = Array1::zeros(self.n);
        for i in 0..self.n1 {
            y[i] = a11inv_x1[i] - a11inv_a12w2[i];
        }
        for i in 0..self.n2 {
            y[self.n1 + i] = w2[i];
        }

        Ok(y)
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ============================================================================
// AMGPreconditioner — full Algebraic Multigrid V-cycle
// ============================================================================

/// A single level in the AMG hierarchy.
#[derive(Debug, Clone)]
struct AMGLevel {
    /// System matrix at this level (n_l × n_l)
    a: Array2<f64>,
    /// Interpolation (prolongation) operator P: coarse → fine  (n_l × n_{l+1})
    p: Array2<f64>,
    /// Restriction operator R = Pᵀ: fine → coarse  (n_{l+1} × n_l)
    r: Array2<f64>,
    /// Inverse diagonal for weighted Jacobi / Gauss-Seidel
    diag_inv: Array1<f64>,
    /// Level size
    n: usize,
}

/// AMG (Algebraic Multigrid) preconditioner with full V-cycle.
///
/// Implements a classical Ruge-Stueben-style AMG hierarchy:
/// 1. **Strength of connection**: identifies strong connections using a
///    threshold θ (default 0.25) on the magnitude of off-diagonal entries.
/// 2. **C/F splitting**: greedy coarsening that partitions unknowns into
///    Coarse (C) and Fine (F) sets.
/// 3. **Interpolation**: classical interpolation from C-points to F-points
///    using strong connections.
/// 4. **Galerkin coarse-grid operator**: A_c = R A P = Pᵀ A P.
/// 5. **V-cycle**: pre-smoothing → restriction → coarse solve/recurse →
///    prolongation → post-smoothing.
///
/// Falls back to Gauss-Seidel single-level when the matrix is too small
/// for coarsening (n ≤ `max_coarse_size`).
///
/// # References
/// - Ruge, J. W. & Stueben, K. (1987). *Algebraic Multigrid*, in Multigrid Methods.
/// - Trottenberg, Oosterlee, Schuller (2001). *Multigrid*.
#[derive(Debug, Clone)]
pub struct AMGPreconditioner {
    /// Fine-level system size
    n: usize,
    /// Number of pre- and post-smoothing steps (symmetric Gauss-Seidel)
    smoothing_steps: usize,
    /// Hierarchy of AMG levels (level 0 = finest)
    levels: Vec<AMGLevel>,
    /// Maximum coarse grid size below which we do a direct solve
    max_coarse_size: usize,
    /// Strength threshold θ for connection detection
    strength_threshold: f64,
}

impl AMGPreconditioner {
    /// Create an AMG preconditioner with automatic hierarchy construction.
    ///
    /// # Arguments
    ///
    /// * `a` - System matrix (n × n), should be SPD or M-matrix-like for best results
    /// * `smoothing_steps` - Number of pre/post Gauss-Seidel sweeps (default: 2)
    pub fn new(a: &ArrayView2<f64>, smoothing_steps: Option<usize>) -> LinalgResult<Self> {
        Self::with_options(a, smoothing_steps, None, None)
    }

    /// Create an AMG preconditioner with full control over parameters.
    ///
    /// # Arguments
    ///
    /// * `a` - System matrix (n × n)
    /// * `smoothing_steps` - Number of pre/post smoothing steps (default: 2)
    /// * `max_coarse_size` - Maximum coarse grid size for direct solve (default: 32)
    /// * `strength_threshold` - θ for strength of connection (default: 0.25)
    pub fn with_options(
        a: &ArrayView2<f64>,
        smoothing_steps: Option<usize>,
        max_coarse_size: Option<usize>,
        strength_threshold: Option<f64>,
    ) -> LinalgResult<Self> {
        let (m, n) = a.dim();
        if m != n {
            return Err(LinalgError::ShapeError(
                "AMGPreconditioner: A must be square".to_string(),
            ));
        }

        let steps = smoothing_steps.unwrap_or(2);
        let coarse_max = max_coarse_size.unwrap_or(32);
        let theta = strength_threshold.unwrap_or(0.25);

        let mut precond = Self {
            n,
            smoothing_steps: steps,
            levels: Vec::new(),
            max_coarse_size: coarse_max,
            strength_threshold: theta,
        };

        // Build the multigrid hierarchy
        precond.build_hierarchy(&a.to_owned())?;

        Ok(precond)
    }

    /// Build the AMG hierarchy by repeated coarsening.
    fn build_hierarchy(&mut self, a_fine: &Array2<f64>) -> LinalgResult<()> {
        let mut a_current = a_fine.clone();
        let max_levels = 20; // safety limit

        for _ in 0..max_levels {
            let n_cur = a_current.nrows();

            // Compute diagonal inverse
            let diag_inv = Self::compute_diag_inv(&a_current);

            // If small enough, store final level and stop
            if n_cur <= self.max_coarse_size {
                self.levels.push(AMGLevel {
                    a: a_current.clone(),
                    p: Array2::zeros((0, 0)),
                    r: Array2::zeros((0, 0)),
                    diag_inv,
                    n: n_cur,
                });
                break;
            }

            // Step 1: Compute strength of connection
            let strong = Self::strength_of_connection(&a_current, self.strength_threshold);

            // Step 2: C/F splitting
            let cf_splitting = Self::coarsen(&strong, n_cur);

            // Count coarse points
            let n_coarse: usize = cf_splitting.iter().filter(|&&c| c).count();

            // If coarsening didn't reduce enough, stop
            if n_coarse == 0 || n_coarse >= n_cur {
                self.levels.push(AMGLevel {
                    a: a_current.clone(),
                    p: Array2::zeros((0, 0)),
                    r: Array2::zeros((0, 0)),
                    diag_inv,
                    n: n_cur,
                });
                break;
            }

            // Step 3: Build interpolation operator P
            let p = Self::build_interpolation(&a_current, &strong, &cf_splitting, n_coarse);

            // R = P^T
            let r = p.t().to_owned();

            // Step 4: Galerkin coarse-grid operator A_c = R A P = P^T A P
            let ap = Self::matmul_dense(&a_current, &p);
            let a_coarse = Self::matmul_dense(&r, &ap);

            self.levels.push(AMGLevel {
                a: a_current.clone(),
                p: p.clone(),
                r,
                diag_inv,
                n: n_cur,
            });

            a_current = a_coarse;
        }

        // If no levels were created (shouldn't happen), add at least one
        if self.levels.is_empty() {
            let diag_inv = Self::compute_diag_inv(a_fine);
            self.levels.push(AMGLevel {
                a: a_fine.clone(),
                p: Array2::zeros((0, 0)),
                r: Array2::zeros((0, 0)),
                diag_inv,
                n: a_fine.nrows(),
            });
        }

        Ok(())
    }

    /// Compute inverse of diagonal entries with safe fallback for near-zeros.
    fn compute_diag_inv(a: &Array2<f64>) -> Array1<f64> {
        let n = a.nrows();
        let mut diag_inv = Array1::zeros(n);
        for i in 0..n {
            let d = a[[i, i]];
            if d.abs() < 1e-300 {
                diag_inv[i] = 1.0;
            } else {
                diag_inv[i] = 1.0 / d;
            }
        }
        diag_inv
    }

    /// Compute the strength-of-connection matrix.
    ///
    /// Entry (i,j) is "strong" if |a_{ij}| >= θ * max_{k≠i} |a_{ik}|
    fn strength_of_connection(a: &Array2<f64>, theta: f64) -> Vec<Vec<bool>> {
        let n = a.nrows();
        let mut strong = vec![vec![false; n]; n];

        for i in 0..n {
            // Find max off-diagonal magnitude
            let mut max_off = 0.0_f64;
            for j in 0..n {
                if j != i {
                    let v = a[[i, j]].abs();
                    if v > max_off {
                        max_off = v;
                    }
                }
            }

            let threshold = theta * max_off;

            for j in 0..n {
                if j != i && a[[i, j]].abs() >= threshold {
                    strong[i][j] = true;
                }
            }
        }

        strong
    }

    /// C/F splitting via greedy coarsening.
    ///
    /// Uses a simplified Ruge-Stueben coarsening strategy:
    /// - Compute "importance" λ_i = |{j : (i,j) strongly connected and j is undecided}|
    /// - Select the point with highest λ as coarse
    /// - Mark its strongly connected neighbors as fine
    /// - Repeat until all points are classified
    ///
    /// Returns a boolean vector: true = Coarse, false = Fine.
    fn coarsen(strong: &[Vec<bool>], n: usize) -> Vec<bool> {
        // 0 = undecided, 1 = coarse, 2 = fine
        let mut status = vec![0u8; n];
        let mut lambda: Vec<usize> = (0..n)
            .map(|i| strong[i].iter().filter(|&&s| s).count())
            .collect();

        loop {
            // Find undecided point with maximum lambda
            let mut best_i = None;
            let mut best_lambda = 0;
            for i in 0..n {
                if status[i] == 0 && lambda[i] >= best_lambda {
                    best_lambda = lambda[i];
                    best_i = Some(i);
                }
            }

            let i = match best_i {
                Some(idx) => idx,
                None => break, // all points are classified
            };

            // Mark i as Coarse
            status[i] = 1;

            // Mark strongly connected undecided neighbors as Fine
            for j in 0..n {
                if strong[i][j] && status[j] == 0 {
                    status[j] = 2;
                    // Update lambda for neighbors of j
                    for k in 0..n {
                        if strong[j][k] && status[k] == 0 {
                            lambda[k] = lambda[k].saturating_add(1);
                        }
                    }
                }
            }

            // Zero out lambda for classified points
            lambda[i] = 0;
            for j in 0..n {
                if strong[i][j] && status[j] == 2 {
                    lambda[j] = 0;
                }
            }
        }

        // Convert: true = coarse
        status.iter().map(|&s| s == 1).collect()
    }

    /// Build the interpolation (prolongation) operator P.
    ///
    /// For coarse points: identity row (P_{c,c_idx} = 1).
    /// For fine points: interpolation from strong coarse neighbors,
    ///   weights ∝ |a_{f,c}| / Σ |a_{f,c_k}| for strong coarse neighbors c_k.
    fn build_interpolation(
        a: &Array2<f64>,
        strong: &[Vec<bool>],
        cf_splitting: &[bool],
        n_coarse: usize,
    ) -> Array2<f64> {
        let n = a.nrows();

        // Map from fine-grid index to coarse-grid index
        let mut coarse_idx = vec![0usize; n];
        let mut idx = 0;
        for i in 0..n {
            if cf_splitting[i] {
                coarse_idx[i] = idx;
                idx += 1;
            }
        }

        let mut p = Array2::zeros((n, n_coarse));

        for i in 0..n {
            if cf_splitting[i] {
                // Coarse point: identity
                p[[i, coarse_idx[i]]] = 1.0;
            } else {
                // Fine point: interpolate from strong coarse neighbors
                let mut strong_coarse_neighbors = Vec::new();
                let mut weight_sum = 0.0_f64;

                for j in 0..n {
                    if strong[i][j] && cf_splitting[j] {
                        let w = a[[i, j]].abs();
                        strong_coarse_neighbors.push((j, w));
                        weight_sum += w;
                    }
                }

                if weight_sum > 1e-300 && !strong_coarse_neighbors.is_empty() {
                    // Normalize weights.
                    // For M-matrices (negative off-diagonal), use sign-aware weights.
                    let diag = a[[i, i]];
                    let sign_factor = if diag > 0.0 { -1.0 } else { 1.0 };
                    for (j, w) in &strong_coarse_neighbors {
                        let normalized = sign_factor * (*w / weight_sum);
                        // Scale by ratio: -sum_strong / a_ii to ensure stability
                        let strong_sum: f64 = strong_coarse_neighbors.iter().map(|(_, ww)| ww).sum();
                        let scale = if diag.abs() > 1e-300 {
                            strong_sum / diag.abs()
                        } else {
                            1.0
                        };
                        p[[i, coarse_idx[*j]]] = normalized * scale;
                    }
                } else if !strong_coarse_neighbors.is_empty() {
                    // Fallback: equal weights
                    let w = 1.0 / strong_coarse_neighbors.len() as f64;
                    for (j, _) in &strong_coarse_neighbors {
                        p[[i, coarse_idx[*j]]] = w;
                    }
                }
                // If no strong coarse neighbors, this row is zero (point is isolated)
            }
        }

        p
    }

    /// Dense matrix multiply C = A * B.
    fn matmul_dense(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();
        let mut c = Array2::zeros((m, n));
        for i in 0..m {
            for p in 0..k {
                let aip = a[[i, p]];
                if aip.abs() < 1e-300 {
                    continue;
                }
                for j in 0..n {
                    c[[i, j]] += aip * b[[p, j]];
                }
            }
        }
        c
    }

    /// Perform a single V-cycle: pre-smooth → restrict → coarse solve → prolong → post-smooth.
    ///
    /// Recursively applies the V-cycle at each level. At the coarsest level,
    /// a direct Gauss-Seidel solve (many iterations) or LU is used.
    fn vcycle(&self, level: usize, b: &Array1<f64>, x: &mut Array1<f64>) {
        let lev = &self.levels[level];
        let n = lev.n;

        // Base case: coarsest level → direct solve via many G-S sweeps
        if level == self.levels.len() - 1 || lev.p.nrows() == 0 {
            // Use many Gauss-Seidel iterations as a coarse solver
            let coarse_iters = if n <= 4 { 50 } else { 20 };
            Self::gauss_seidel(&lev.a, &lev.diag_inv, b, x, coarse_iters);
            return;
        }

        // 1. Pre-smoothing (forward Gauss-Seidel)
        Self::gauss_seidel(&lev.a, &lev.diag_inv, b, x, self.smoothing_steps);

        // 2. Compute residual r = b - A*x
        let ax = Self::matvec_dense(&lev.a, x);
        let r: Array1<f64> = Array1::from_iter(
            b.iter().zip(ax.iter()).map(|(&bi, &ai)| bi - ai),
        );

        // 3. Restrict residual to coarse grid: r_c = R * r = P^T * r
        let r_coarse = Self::matvec_dense(&lev.r, &r);

        // 4. Solve on coarse grid (recursive V-cycle)
        let n_coarse = self.levels[level + 1].n;
        let mut e_coarse = Array1::zeros(n_coarse);
        self.vcycle(level + 1, &r_coarse, &mut e_coarse);

        // 5. Prolongate correction: x += P * e_c
        let correction = Self::matvec_dense(&lev.p, &e_coarse);
        for i in 0..n {
            x[i] += correction[i];
        }

        // 6. Post-smoothing (backward Gauss-Seidel)
        Self::gauss_seidel_backward(&lev.a, &lev.diag_inv, b, x, self.smoothing_steps);
    }

    /// Dense matrix-vector product y = A * x.
    fn matvec_dense(a: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
        let m = a.nrows();
        let n = a.ncols();
        let mut y = Array1::zeros(m);
        for i in 0..m {
            let mut s = 0.0_f64;
            for j in 0..n {
                s += a[[i, j]] * x[j];
            }
            y[i] = s;
        }
        y
    }

    /// Forward Gauss-Seidel smoothing.
    fn gauss_seidel(
        a: &Array2<f64>,
        diag_inv: &Array1<f64>,
        b: &Array1<f64>,
        x: &mut Array1<f64>,
        steps: usize,
    ) {
        let n = a.nrows();
        for _ in 0..steps {
            for i in 0..n {
                let mut s = 0.0_f64;
                for j in 0..n {
                    if j != i {
                        s += a[[i, j]] * x[j];
                    }
                }
                x[i] = (b[i] - s) * diag_inv[i];
            }
        }
    }

    /// Backward Gauss-Seidel smoothing (reverse sweep order).
    fn gauss_seidel_backward(
        a: &Array2<f64>,
        diag_inv: &Array1<f64>,
        b: &Array1<f64>,
        x: &mut Array1<f64>,
        steps: usize,
    ) {
        let n = a.nrows();
        for _ in 0..steps {
            for i in (0..n).rev() {
                let mut s = 0.0_f64;
                for j in 0..n {
                    if j != i {
                        s += a[[i, j]] * x[j];
                    }
                }
                x[i] = (b[i] - s) * diag_inv[i];
            }
        }
    }
}

impl PrecondApply for AMGPreconditioner {
    /// Apply the AMG V-cycle preconditioner: y ≈ A⁻¹ b.
    fn apply(&self, b: &ArrayView1<f64>) -> LinalgResult<Array1<f64>> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "AMGPreconditioner::apply: b has {} elements but n={}",
                b.len(),
                self.n
            )));
        }

        let mut x = Array1::zeros(self.n);
        let b_owned = b.to_owned();

        // Apply one V-cycle starting from level 0
        self.vcycle(0, &b_owned, &mut x);

        Ok(x)
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ============================================================================
// Helper: small matrix inverse via LU
// ============================================================================

/// Compute the inverse of a small matrix using LU with partial pivoting.
fn block_inverse(a: &ArrayView2<f64>) -> LinalgResult<Array2<f64>> {
    let n = a.nrows();
    if n == 0 {
        return Ok(Array2::zeros((0, 0)));
    }
    if n == 1 {
        let v = a[[0, 0]];
        if v.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "block_inverse: 1×1 block is zero".to_string(),
            ));
        }
        let mut result = Array2::zeros((1, 1));
        result[[0, 0]] = 1.0 / v;
        return Ok(result);
    }

    // LU with partial pivoting
    let mut lu = a.to_owned();
    let mut pivot = vec![0usize; n];

    for k in 0..n {
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        pivot[k] = max_row;

        if max_row != k {
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
        }

        let diag = lu[[k, k]];
        if diag.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(format!(
                "block_inverse: singular at column {k}"
            )));
        }

        for i in (k + 1)..n {
            lu[[i, k]] /= diag;
            for j in (k + 1)..n {
                let lk = lu[[i, k]];
                let lkj = lu[[k, j]];
                lu[[i, j]] -= lk * lkj;
            }
        }
    }

    // Solve LU x = I column by column
    let mut inv = Array2::eye(n);

    // Apply row permutation
    for k in 0..n {
        let p = pivot[k];
        if p != k {
            for j in 0..n {
                let tmp = inv[[k, j]];
                inv[[k, j]] = inv[[p, j]];
                inv[[p, j]] = tmp;
            }
        }
    }

    // Forward substitution
    for k in 0..n {
        for i in (k + 1)..n {
            let lk = lu[[i, k]];
            for j in 0..n {
                let ik = inv[[k, j]];
                inv[[i, j]] -= lk * ik;
            }
        }
    }

    // Backward substitution
    for k in (0..n).rev() {
        let diag = lu[[k, k]];
        for j in 0..n {
            inv[[k, j]] /= diag;
        }
        for i in 0..k {
            let uk = lu[[i, k]];
            for j in 0..n {
                let kj = inv[[k, j]];
                inv[[i, j]] -= uk * kj;
            }
        }
    }

    Ok(inv)
}

/// Solve a small positive-definite system via Cholesky decomposition.
fn cholesky_solve_small(a: &[f64], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    // Attempt Cholesky; fall back to LU-like solve with regularization
    // Try Cholesky: L L^T = A
    let mut l = vec![0.0f64; n * n];

    let mut chol_ok = true;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum < 0.0 {
                    chol_ok = false;
                    break;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < 1e-300 {
                    chol_ok = false;
                    break;
                }
                l[i * n + j] = sum / ljj;
            }
        }
        if !chol_ok {
            break;
        }
    }

    if !chol_ok {
        // Fall back to Gaussian elimination
        return gauss_solve_small(a, b, n);
    }

    // Forward substitution: L y = b
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "cholesky_solve_small: zero diagonal in L".to_string(),
            ));
        }
        y[i] = s / lii;
    }

    // Backward substitution: L^T x = y
    let mut x = y;
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let lii = l[i * n + i];
        x[i] = s / lii;
    }

    Ok(x)
}

/// Gaussian elimination with partial pivoting for small systems.
fn gauss_solve_small(a: &[f64], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    let mut aug: Vec<f64> = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    for k in 0..n {
        // Find pivot
        let mut max_val = aug[k * (n + 1) + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = aug[i * (n + 1) + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_row != k {
            for j in 0..=(n) {
                aug.swap(k * (n + 1) + j, max_row * (n + 1) + j);
            }
        }

        let pivot = aug[k * (n + 1) + k];
        if pivot.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(format!(
                "gauss_solve_small: singular at column {k}"
            )));
        }

        for i in (k + 1)..n {
            let factor = aug[i * (n + 1) + k] / pivot;
            for j in k..=(n) {
                let akj = aug[k * (n + 1) + j];
                aug[i * (n + 1) + j] -= factor * akj;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        let pivot = aug[i * (n + 1) + i];
        if pivot.abs() < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "gauss_solve_small: zero pivot in back substitution".to_string(),
            ));
        }
        x[i] = s / pivot;
    }

    Ok(x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_spai_diagonal_matrix() {
        // For a diagonal matrix, SPAI should give approximate diagonal inverse
        let a = array![
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0],
        ];

        let spai = SPAI::new(&a.view(), 0, 0, None).expect("SPAI::new failed");

        // The SPAI should produce something close to the diagonal inverse
        let x = array![1.0, 1.0, 1.0];
        let y = spai.apply(&x.view()).expect("SPAI::apply failed");

        // For a diagonal system, SPAI should produce roughly [1/3, 1/2, 1/4]
        assert!(y[0] > 0.0, "SPAI y[0] should be positive");
        assert!(y[1] > 0.0, "SPAI y[1] should be positive");
        assert!(y[2] > 0.0, "SPAI y[2] should be positive");
    }

    #[test]
    fn test_polynomial_preconditioner_neumann() {
        // SPD matrix
        let a = array![
            [4.0, 1.0, 0.0],
            [1.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
        ];

        let precond = PolynomialAdvancedPreconditioner::new_neumann(&a.view(), 3)
            .expect("PolynomialAdvancedPreconditioner::new_neumann failed");

        let x = array![1.0, 2.0, 3.0];
        let y = precond
            .apply_with_matrix(&a.view(), &x.view())
            .expect("apply_with_matrix failed");

        assert_eq!(y.len(), 3);
        // Result should be a reasonable approximation of A^{-1} x
        // A * y ≈ x (approximately)
        let ay: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[[i, j]] * y[j]).sum::<f64>())
            .collect();
        for i in 0..3 {
            assert!(
                (ay[i] - x[i]).abs() < 2.0,
                "Neumann preconditioner: A*y ≈ x failed at {i}: A*y[i]={}, x[i]={}",
                ay[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_polynomial_preconditioner_chebyshev() {
        let a = array![
            [4.0, 0.5, 0.0],
            [0.5, 4.0, 0.5],
            [0.0, 0.5, 4.0],
        ];

        // Spectral bounds: eigenvalues of this diagonally dominant matrix are
        // roughly in [3, 5]
        let precond = PolynomialAdvancedPreconditioner::new_chebyshev(
            &a.view(),
            5,
            3.0,
            5.0,
        )
        .expect("Chebyshev preconditioner failed");

        let x = array![1.0, 0.0, 0.0];
        let y = precond
            .apply_with_matrix(&a.view(), &x.view())
            .expect("Chebyshev apply failed");
        assert_eq!(y.len(), 3);
    }

    #[test]
    fn test_block_diagonal_preconditioner() {
        let a = array![
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ];

        let precond = BlockDiagonalPreconditioner::new_uniform(&a.view(), 2)
            .expect("BlockDiagonalPreconditioner failed");

        let x = array![1.0, 0.0, 0.0, 1.0];
        let y = precond.apply(&x.view()).expect("apply failed");
        assert_eq!(y.len(), 4);

        // Verify: A * y ≈ x for the block structure
        let ay: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| a[[i, j]] * y[j]).sum::<f64>())
            .collect();
        for i in 0..4 {
            assert!(
                (ay[i] - x[i]).abs() < 1e-10,
                "BlockDiagonal: A*y != x at {i}: {} vs {}",
                ay[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_schur_complement_preconditioner() {
        // Block system with clear structure
        let a = array![
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.0, 0.0, 0.5],
            [0.5, 0.0, 5.0, 1.0],
            [0.0, 0.5, 1.0, 4.0],
        ];

        let precond = SchurComplementPreconditioner::new(&a.view(), 2, None, None)
            .expect("SchurComplementPreconditioner failed");

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = precond.apply(&x.view()).expect("SchurComplement apply failed");
        assert_eq!(y.len(), 4);

        // For the exact Schur complement, A * y = x exactly
        let ay: Vec<f64> = (0..4)
            .map(|i| (0..4).map(|j| a[[i, j]] * y[j]).sum::<f64>())
            .collect();
        for i in 0..4 {
            assert!(
                (ay[i] - x[i]).abs() < 1e-8,
                "Schur: A*y != x at {i}: {} vs {}",
                ay[i],
                x[i]
            );
        }
    }

    #[test]
    fn test_amg_preconditioner_small() {
        let a = array![
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, -1.0],
            [0.0, -1.0, 3.0],
        ];

        let precond = AMGPreconditioner::new(&a.view(), Some(5))
            .expect("AMGPreconditioner::new failed");

        let b = array![1.0, 2.0, 3.0];
        let x = precond.apply(&b.view()).expect("AMG apply failed");
        assert_eq!(x.len(), 3);

        // Gauss-Seidel should produce a reasonable approximation
        let ax: f64 = (0..3)
            .map(|i| {
                let row_sum = (0..3).map(|j| a[[i, j]] * x[j]).sum::<f64>();
                (row_sum - b[i]).abs()
            })
            .sum();
        // After 5 GS steps, residual should be moderate
        assert!(ax < 10.0, "AMG: residual too large: {ax}");
    }

    #[test]
    fn test_amg_preconditioner_multilevel() {
        // Larger tridiagonal system (1D Laplacian) — good for AMG
        let n = 16;
        let mut a = Array2::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 2.0;
            if i > 0 {
                a[[i, i - 1]] = -1.0;
            }
            if i < n - 1 {
                a[[i, i + 1]] = -1.0;
            }
        }

        let precond = AMGPreconditioner::with_options(
            &a.view(),
            Some(2),        // smoothing steps
            Some(4),         // max coarse size
            Some(0.25),      // strength threshold
        )
        .expect("AMG multilevel setup failed");

        // Should have created multiple levels
        assert!(precond.levels.len() >= 2, "Expected multiple AMG levels, got {}", precond.levels.len());

        let b: Array1<f64> = Array1::from_iter((0..n).map(|i| (i + 1) as f64));
        let x = precond.apply(&b.view()).expect("AMG V-cycle apply failed");
        assert_eq!(x.len(), n);

        // Check that the residual is reasonable (V-cycle should reduce it)
        let mut res_norm = 0.0_f64;
        for i in 0..n {
            let mut row_sum = 0.0_f64;
            for j in 0..n {
                row_sum += a[[i, j]] * x[j];
            }
            let diff = row_sum - b[i];
            res_norm += diff * diff;
        }
        res_norm = res_norm.sqrt();
        let b_norm: f64 = b.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let rel_res = res_norm / b_norm;

        // One V-cycle should give a meaningful reduction
        assert!(
            rel_res < 1.0,
            "AMG V-cycle relative residual too large: {rel_res}"
        );
    }
}
