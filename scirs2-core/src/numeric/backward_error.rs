//! Backward error analysis tools for numerical algorithms.
//!
//! This module implements rigorous backward error analysis following the framework
//! established by Higham's "Accuracy and Stability of Numerical Algorithms" (SIAM, 2002).
//!
//! ## Background
//!
//! Backward error analysis asks: "For what perturbed input data does my computed result
//! exactly satisfy the problem?" A small backward error means the algorithm is numerically
//! stable even when the problem itself may be ill-conditioned.
//!
//! For a linear system `Ax = b`, we distinguish:
//!
//! - **Normwise backward error**: measures the relative size of the smallest perturbation
//!   `(δA, δb)` such that `(A + δA) x̃ = b + δb`, using matrix norms.
//!   Formula: `ω = ||r|| / (||A|| ||x̃|| + ||b||)` where `r = b - Ax̃`.
//!
//! - **Componentwise backward error**: Skeel's measure, uses element-wise absolute values
//!   and asks per-component: `ω_i = |r_i| / (|A| |x̃| + |b|)_i`.
//!   This is more informative when matrix entries vary wildly in magnitude.
//!
//! - **Perturbation bound**: given a change `δA` in the matrix, how much can the solution
//!   change? `||δx|| / ||x|| ≤ κ(A) * (||δA|| / ||A|| + ||δb|| / ||b||)`.
//!
//! ## References
//!
//! - Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
//! - Skeel, R. D. (1979). "Scaling for Numerical Stability in Gaussian Elimination."
//!   *J. ACM*, 26(3), 494–526.
//! - LAPACK Working Note 165 (Demmel et al., 2006).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// Norm helpers (no external BLAS dependency — pure Rust)
// ---------------------------------------------------------------------------

/// Compute the 1-norm of a matrix: `max_j Σ_i |a_{ij}|`.
///
/// The 1-norm equals the maximum absolute column sum.
#[inline]
fn matrix_norm_1(a: &ArrayView2<f64>) -> f64 {
    let ncols = a.ncols();
    let mut max_col = 0.0_f64;
    for j in 0..ncols {
        let col_sum: f64 = a.column(j).iter().map(|&x| x.abs()).sum();
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// Compute the ∞-norm of a matrix: `max_i Σ_j |a_{ij}|`.
///
/// The ∞-norm equals the maximum absolute row sum.
#[inline]
fn matrix_norm_inf(a: &ArrayView2<f64>) -> f64 {
    let nrows = a.nrows();
    let mut max_row = 0.0_f64;
    for i in 0..nrows {
        let row_sum: f64 = a.row(i).iter().map(|&x| x.abs()).sum();
        if row_sum > max_row {
            max_row = row_sum;
        }
    }
    max_row
}

/// Compute the Frobenius norm of a matrix: `sqrt(Σ_{ij} a_{ij}^2)`.
#[inline]
fn matrix_norm_frob(a: &ArrayView2<f64>) -> f64 {
    let sum_sq: f64 = a.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

/// Compute the vector 2-norm: `sqrt(Σ_i x_i^2)`.
///
/// Uses a two-pass scaled approach to avoid intermediate overflow/underflow.
#[inline]
fn vector_norm_2(x: &ArrayView1<f64>) -> f64 {
    // Two-pass: find max abs, then scale to avoid over/underflow
    let max_abs = x.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
    if max_abs == 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = x.iter().map(|&v| (v / max_abs).powi(2)).sum();
    max_abs * sum_sq.sqrt()
}

/// Compute the vector ∞-norm: `max_i |x_i|`.
#[inline]
fn vector_norm_inf(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max)
}

/// Compute the vector 1-norm: `Σ_i |x_i|`.
#[inline]
fn vector_norm_1(x: &ArrayView1<f64>) -> f64 {
    x.iter().map(|&v| v.abs()).sum()
}

// ---------------------------------------------------------------------------
// Condition number functions
// ---------------------------------------------------------------------------

/// Compute the 1-norm condition number of a square matrix.
///
/// The condition number `κ_1(A) = ||A||_1 * ||A^{-1}||_1` measures how
/// sensitive the solution of `Ax = b` is to perturbations in `A` or `b`
/// under the 1-norm (maximum absolute column sum).
///
/// This implementation uses a pure-Rust LU factorisation (Doolittle) to
/// compute `||A^{-1}||_1`.  For singular matrices the function returns
/// `f64::INFINITY`.
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if `A` is not square.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::condition_number_1norm;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
/// let kappa = condition_number_1norm(&a.view()).expect("should succeed");
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn condition_number_1norm(a: &ArrayView2<f64>) -> CoreResult<f64> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CoreError::ShapeError(ErrorContext::new(
            "condition_number_1norm requires a square matrix",
        )));
    }
    let norm_a = matrix_norm_1(a);
    if norm_a == 0.0 {
        return Ok(f64::INFINITY);
    }
    match invert_matrix(a) {
        Some(inv) => Ok(norm_a * matrix_norm_1(&inv.view())),
        None => Ok(f64::INFINITY),
    }
}

/// Compute the ∞-norm condition number of a square matrix.
///
/// `κ_∞(A) = ||A||_∞ * ||A^{-1}||_∞`.
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if `A` is not square.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::condition_number_inf;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 1.0]];
/// let kappa = condition_number_inf(&a.view()).expect("should succeed");
/// assert!((kappa - 3.0).abs() < 1e-10);
/// ```
pub fn condition_number_inf(a: &ArrayView2<f64>) -> CoreResult<f64> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CoreError::ShapeError(ErrorContext::new(
            "condition_number_inf requires a square matrix",
        )));
    }
    let norm_a = matrix_norm_inf(a);
    if norm_a == 0.0 {
        return Ok(f64::INFINITY);
    }
    match invert_matrix(a) {
        Some(inv) => Ok(norm_a * matrix_norm_inf(&inv.view())),
        None => Ok(f64::INFINITY),
    }
}

/// Compute the Frobenius-norm condition number of a square matrix.
///
/// `κ_F(A) = ||A||_F * ||A^{-1}||_F`.
///
/// Note: the Frobenius norm condition number is not a proper induced matrix
/// norm condition number, but it is often used in practice because it is
/// easy to compute and bounds the sensitivity of linear systems.
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if `A` is not square.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::condition_number_frob;
///
/// let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
/// let kappa = condition_number_frob(&a.view()).expect("should succeed");
/// // For 2*I: ||A||_F = 2*sqrt(2), ||A^{-1}||_F = (1/2)*sqrt(2), product = 2
/// assert!((kappa - 2.0).abs() < 1e-10);
/// ```
pub fn condition_number_frob(a: &ArrayView2<f64>) -> CoreResult<f64> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CoreError::ShapeError(ErrorContext::new(
            "condition_number_frob requires a square matrix",
        )));
    }
    let norm_a = matrix_norm_frob(a);
    if norm_a == 0.0 {
        return Ok(f64::INFINITY);
    }
    match invert_matrix(a) {
        Some(inv) => Ok(norm_a * matrix_norm_frob(&inv.view())),
        None => Ok(f64::INFINITY),
    }
}

// ---------------------------------------------------------------------------
// Backward error measures for Ax = b
// ---------------------------------------------------------------------------

/// Standard normwise backward error for a linear system `Ax = b`.
///
/// Given a computed solution `x̃`, the **residual** is `r = b - A x̃`.
/// The normwise backward error is:
///
/// ```text
/// η(x̃) = ||r||_2 / (||A||_F * ||x̃||_2 + ||b||_2)
/// ```
///
/// A value near machine epsilon indicates the algorithm is backward-stable.
///
/// # Arguments
///
/// * `a`  — coefficient matrix `A` (m × n)
/// * `b`  — right-hand side vector (m)
/// * `x`  — computed solution vector (n)
///
/// # Errors
///
/// Returns `CoreError::ShapeError` when dimensions are inconsistent.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::backward_error_linear;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![3.0_f64, 4.0];
/// let x = array![3.0_f64, 4.0]; // exact solution
/// let err = backward_error_linear(&a.view(), &b.view(), &x.view()).expect("should succeed");
/// assert!(err < 1e-14);
/// ```
pub fn backward_error_linear(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    x: &ArrayView1<f64>,
) -> CoreResult<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    if b.len() != m {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "backward_error_linear: b length {blen} != a rows {m}",
            blen = b.len()
        ))));
    }
    if x.len() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "backward_error_linear: x length {xlen} != a cols {n}",
            xlen = x.len()
        ))));
    }

    // residual r = b - A*x
    let ax = matvec(a, x);
    let r: Array1<f64> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));

    let norm_r = vector_norm_2(&r.view());
    let norm_a = matrix_norm_frob(a);
    let norm_x = vector_norm_2(x);
    let norm_b = vector_norm_2(b);

    let denom = norm_a * norm_x + norm_b;
    if denom == 0.0 {
        if norm_r == 0.0 {
            return Ok(0.0);
        }
        return Ok(f64::INFINITY);
    }
    Ok(norm_r / denom)
}

/// LAPACK-style normwise backward error for `Ax = b`.
///
/// This is the standard measure used internally by LAPACK's `*GERFS` (iterative
/// refinement) routines.  It uses the ∞-norm:
///
/// ```text
/// η(x̃) = ||r||_∞ / (||A||_∞ * ||x̃||_∞ + ||b||_∞)
/// ```
///
/// Equivalent to [`backward_error_linear`] but with ∞-norms, which is the
/// LAPACK convention.
///
/// # Errors
///
/// Returns `CoreError::ShapeError` when dimensions are inconsistent.
pub fn normwise_backward_error(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    x: &ArrayView1<f64>,
) -> CoreResult<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    if b.len() != m {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "normwise_backward_error: b length {blen} != a rows {m}",
            blen = b.len()
        ))));
    }
    if x.len() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "normwise_backward_error: x length {xlen} != a cols {n}",
            xlen = x.len()
        ))));
    }

    let ax = matvec(a, x);
    let r: Array1<f64> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));

    let norm_r = vector_norm_inf(&r.view());
    let norm_a = matrix_norm_inf(a);
    let norm_x = vector_norm_inf(x);
    let norm_b = vector_norm_inf(b);

    let denom = norm_a * norm_x + norm_b;
    if denom == 0.0 {
        if norm_r == 0.0 {
            return Ok(0.0);
        }
        return Ok(f64::INFINITY);
    }
    Ok(norm_r / denom)
}

/// Skeel's componentwise backward error for `Ax = b`.
///
/// The **componentwise** backward error (Skeel 1979) uses absolute values
/// element-by-element.  It answers: for what smallest `δA`, `δb` with
/// `|δA_{ij}| ≤ ω |A_{ij}|` and `|δb_i| ≤ ω |b_i|` is `x̃` the exact solution?
///
/// ```text
/// ω(x̃) = max_i  |r_i| / (|A| |x̃| + |b|)_i
/// ```
///
/// where `|A|` denotes the matrix of absolute values.
///
/// This measure is far more informative than normwise measures when the matrix
/// entries span many orders of magnitude (e.g., diagonal scaling problems).
///
/// # Errors
///
/// Returns `CoreError::ShapeError` when dimensions are inconsistent.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::componentwise_backward_error;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![5.0_f64, 10.0];
/// let x = array![1.0_f64, 3.0]; // exact: A*x = [2+3, 1+9] = [5, 10]
/// let err = componentwise_backward_error(&a.view(), &b.view(), &x.view()).expect("should succeed");
/// assert!(err < 1e-14);
/// ```
pub fn componentwise_backward_error(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    x: &ArrayView1<f64>,
) -> CoreResult<f64> {
    let (m, n) = (a.nrows(), a.ncols());
    if b.len() != m {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "componentwise_backward_error: b length {blen} != a rows {m}",
            blen = b.len()
        ))));
    }
    if x.len() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "componentwise_backward_error: x length {xlen} != a cols {n}",
            xlen = x.len()
        ))));
    }

    // r = b - A*x
    let ax = matvec(a, x);
    let r: Array1<f64> = Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));

    // (|A| |x̃| + |b|)_i
    let abs_a: Array2<f64> = a.mapv(f64::abs);
    let abs_x: Array1<f64> = x.mapv(f64::abs);
    let denom_vec = matvec(&abs_a.view(), &abs_x.view());

    let mut max_ratio = 0.0_f64;
    for i in 0..m {
        let denom_i = denom_vec[i] + b[i].abs();
        if denom_i == 0.0 {
            if r[i].abs() > 0.0 {
                return Ok(f64::INFINITY);
            }
            // 0/0 component — skip
        } else {
            let ratio = r[i].abs() / denom_i;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
        }
    }
    Ok(max_ratio)
}

/// First-order perturbation bound for the linear system `Ax = b`.
///
/// Given a perturbation `δA` to the matrix, estimates the relative change
/// in the solution using:
///
/// ```text
/// ||δx|| / ||x|| ≤ κ_1(A) * (||δA||_1 / ||A||_1 + ||δb||_1 / ||b||_1)
/// ```
///
/// where `κ_1(A) = ||A||_1 ||A^{-1}||_1` is the 1-norm condition number.
///
/// If `b` is the zero vector, the `δb` term is omitted from the bound.
///
/// # Errors
///
/// Returns `CoreError::ShapeError` if dimensions are inconsistent or if `A`
/// is not square.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::perturbation_bound;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
/// let da = array![[0.0_f64, 0.0], [0.0, 0.0]];
/// let b = array![4.0_f64, 1.0];
/// let db = array![0.0_f64, 0.0];
/// let bound = perturbation_bound(&a.view(), &da.view(), &b.view(), &db.view()).expect("should succeed");
/// assert!(bound < 1e-14);
/// ```
pub fn perturbation_bound(
    a: &ArrayView2<f64>,
    delta_a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    delta_b: &ArrayView1<f64>,
) -> CoreResult<f64> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(CoreError::ShapeError(ErrorContext::new(
            "perturbation_bound: A must be square",
        )));
    }
    if delta_a.nrows() != n || delta_a.ncols() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(
            "perturbation_bound: δA must have same shape as A",
        )));
    }
    if b.len() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "perturbation_bound: b length {blen} != n={n}",
            blen = b.len()
        ))));
    }
    if delta_b.len() != n {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "perturbation_bound: δb length {dblen} != n={n}",
            dblen = delta_b.len()
        ))));
    }

    let kappa = condition_number_1norm(a)?;
    let norm_a = matrix_norm_1(a);
    let norm_da = matrix_norm_1(delta_a);
    let norm_b = vector_norm_1(b);
    let norm_db = vector_norm_1(delta_b);

    if norm_a == 0.0 {
        return Ok(f64::INFINITY);
    }

    let rel_da = norm_da / norm_a;
    let rel_db = if norm_b > 0.0 {
        norm_db / norm_b
    } else {
        0.0
    };

    Ok(kappa * (rel_da + rel_db))
}

// ---------------------------------------------------------------------------
// BackwardErrorAnalysis builder struct
// ---------------------------------------------------------------------------

/// A configurable analyzer that collects multiple backward-error metrics for
/// a linear system `Ax = b` with computed solution `x̃`.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_core::numeric::backward_error::BackwardErrorAnalysis;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![5.0_f64, 10.0];
/// let x = array![1.0_f64, 3.0];
///
/// let report = BackwardErrorAnalysis::new()
///     .analyze(&a.view(), &b.view(), &x.view())
///     .expect("should succeed");
///
/// assert!(report.normwise_backward_error < 1e-14);
/// assert!(report.componentwise_backward_error < 1e-14);
/// ```
#[derive(Debug, Clone, Default)]
pub struct BackwardErrorAnalysis {
    /// Whether to compute the LAPACK-style ∞-norm backward error (default: true).
    pub compute_normwise: bool,
    /// Whether to compute Skeel's componentwise backward error (default: true).
    pub compute_componentwise: bool,
    /// Whether to compute the 1-norm condition number (default: true).
    pub compute_condition_number: bool,
}

/// Report produced by [`BackwardErrorAnalysis::analyze`].
#[derive(Debug, Clone)]
pub struct BackwardErrorReport {
    /// Standard Frobenius-norm based backward error `||r||_2 / (||A||_F ||x̃||_2 + ||b||_2)`.
    pub normwise_backward_error: f64,
    /// LAPACK ∞-norm backward error `||r||_∞ / (||A||_∞ ||x̃||_∞ + ||b||_∞)`.
    pub lapack_normwise_error: f64,
    /// Skeel's componentwise backward error.
    pub componentwise_backward_error: f64,
    /// 1-norm condition number `κ_1(A)` (only if `A` is square; else `NaN`).
    pub condition_number_1norm: f64,
    /// ∞-norm condition number `κ_∞(A)` (only if `A` is square; else `NaN`).
    pub condition_number_inf: f64,
    /// Frobenius-norm condition number `κ_F(A)` (only if `A` is square; else `NaN`).
    pub condition_number_frob: f64,
    /// 2-norm of the residual `r = b - Ax̃`.
    pub residual_norm: f64,
    /// Number of rows in `A`.
    pub nrows: usize,
    /// Number of columns in `A`.
    pub ncols: usize,
}

impl BackwardErrorAnalysis {
    /// Create a new analysis context with all metrics enabled.
    #[must_use]
    pub fn new() -> Self {
        Self {
            compute_normwise: true,
            compute_componentwise: true,
            compute_condition_number: true,
        }
    }

    /// Disable computation of condition numbers (saves time for large matrices).
    #[must_use]
    pub fn without_condition_numbers(mut self) -> Self {
        self.compute_condition_number = false;
        self
    }

    /// Run the full backward error analysis.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ShapeError` if matrix/vector dimensions are inconsistent.
    pub fn analyze(
        &self,
        a: &ArrayView2<f64>,
        b: &ArrayView1<f64>,
        x: &ArrayView1<f64>,
    ) -> CoreResult<BackwardErrorReport> {
        let (m, n) = (a.nrows(), a.ncols());
        if b.len() != m {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "BackwardErrorAnalysis::analyze: b length {blen} != a rows {m}",
                blen = b.len()
            ))));
        }
        if x.len() != n {
            return Err(CoreError::ShapeError(ErrorContext::new(format!(
                "BackwardErrorAnalysis::analyze: x length {xlen} != a cols {n}",
                xlen = x.len()
            ))));
        }

        let ax = matvec(a, x);
        let r: Array1<f64> =
            Array1::from_iter(b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi));
        let residual_norm = vector_norm_2(&r.view());

        let normwise_be = backward_error_linear(a, b, x)?;
        let lapack_be = normwise_backward_error(a, b, x)?;
        let comp_be = componentwise_backward_error(a, b, x)?;

        let (kappa_1, kappa_inf, kappa_f) = if self.compute_condition_number && m == n {
            let k1 = condition_number_1norm(a).unwrap_or(f64::INFINITY);
            let kinf = condition_number_inf(a).unwrap_or(f64::INFINITY);
            let kf = condition_number_frob(a).unwrap_or(f64::INFINITY);
            (k1, kinf, kf)
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };

        Ok(BackwardErrorReport {
            normwise_backward_error: normwise_be,
            lapack_normwise_error: lapack_be,
            componentwise_backward_error: comp_be,
            condition_number_1norm: kappa_1,
            condition_number_inf: kappa_inf,
            condition_number_frob: kappa_f,
            residual_norm,
            nrows: m,
            ncols: n,
        })
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Matrix–vector product `A * x`.
fn matvec(a: &ArrayView2<f64>, x: &ArrayView1<f64>) -> Array1<f64> {
    let m = a.nrows();
    let n = a.ncols();
    let mut result = Array1::<f64>::zeros(m);
    for i in 0..m {
        let mut s = 0.0_f64;
        for j in 0..n {
            s += a[[i, j]] * x[j];
        }
        result[i] = s;
    }
    result
}

/// Attempt to invert an `n×n` matrix via partial-pivoting Gaussian elimination.
///
/// Returns `None` if the matrix is (numerically) singular.
fn invert_matrix(a: &ArrayView2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    // Build augmented matrix [A | I]
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let v = aug[[row, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < f64::EPSILON * 1e6 {
            return None; // Singular
        }
        // Swap rows
        if pivot_row != col {
            for k in 0..(2 * n) {
                let tmp = aug[[col, k]];
                aug[[col, k]] = aug[[pivot_row, k]];
                aug[[pivot_row, k]] = tmp;
            }
        }
        let diag = aug[[col, col]];
        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / diag;
            for k in col..(2 * n) {
                let v = aug[[row, k]] - factor * aug[[col, k]];
                aug[[row, k]] = v;
            }
        }
    }

    // Back substitution
    for col in (0..n).rev() {
        let diag = aug[[col, col]];
        if diag.abs() < f64::EPSILON * 1e6 {
            return None;
        }
        for k in 0..(2 * n) {
            aug[[col, k]] /= diag;
        }
        for row in 0..col {
            let factor = aug[[row, col]];
            for k in 0..(2 * n) {
                let v = aug[[row, k]] - factor * aug[[col, k]];
                aug[[row, k]] = v;
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matrix_norms() {
        // A = [[1, 2], [3, 4]]
        // 1-norm = max(|col0|, |col1|) = max(4, 6) = 6
        // inf-norm = max(|row0|, |row1|) = max(3, 7) = 7
        // frob = sqrt(1+4+9+16) = sqrt(30)
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        assert!((matrix_norm_1(&a.view()) - 6.0).abs() < 1e-12);
        assert!((matrix_norm_inf(&a.view()) - 7.0).abs() < 1e-12);
        assert!((matrix_norm_frob(&a.view()) - 30.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_vector_norms() {
        let x = array![3.0_f64, 4.0];
        assert!((vector_norm_2(&x.view()) - 5.0).abs() < 1e-12);
        assert!((vector_norm_inf(&x.view()) - 4.0).abs() < 1e-12);
        assert!((vector_norm_1(&x.view()) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_condition_number_diagonal() {
        // Diagonal matrix diag(1, 4) — 1-norm kappa = 4
        let a = array![[1.0_f64, 0.0], [0.0, 4.0]];
        let k1 = condition_number_1norm(&a.view()).expect("should succeed");
        assert!((k1 - 4.0).abs() < 1e-10, "k1={k1}");
        let kinf = condition_number_inf(&a.view()).expect("should succeed");
        assert!((kinf - 4.0).abs() < 1e-10, "kinf={kinf}");
    }

    #[test]
    fn test_condition_number_identity() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let k1 = condition_number_1norm(&a.view()).expect("should succeed");
        assert!((k1 - 1.0).abs() < 1e-10);
        let kf = condition_number_frob(&a.view()).expect("should succeed");
        // ||I||_F = sqrt(2), ||I^{-1}||_F = sqrt(2), product = 2
        assert!((kf - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_error_exact_solution() {
        // If x is the exact solution, backward error should be near machine epsilon
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![5.0_f64, 10.0]; // A * [1, 3] = [2+3, 1+9] = [5, 10]
        let x = array![1.0_f64, 3.0];
        let err = backward_error_linear(&a.view(), &b.view(), &x.view()).expect("should succeed");
        assert!(err < 1e-14, "Expected near-zero backward error, got {err}");
    }

    #[test]
    fn test_normwise_backward_error_exact() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![5.0_f64, 10.0];
        let x = array![1.0_f64, 3.0];
        let err = normwise_backward_error(&a.view(), &b.view(), &x.view()).expect("should succeed");
        assert!(err < 1e-14, "Expected near-zero LAPACK backward error, got {err}");
    }

    #[test]
    fn test_componentwise_backward_error_exact() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![5.0_f64, 10.0];
        let x = array![1.0_f64, 3.0];
        let err = componentwise_backward_error(&a.view(), &b.view(), &x.view()).expect("should succeed");
        assert!(err < 1e-14, "Expected near-zero componentwise backward error, got {err}");
    }

    #[test]
    fn test_backward_error_perturbed_solution() {
        let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let b = array![2.0_f64, 2.0];
        // Exact solution: x = [1, 1]; perturb slightly
        let x = array![1.0_f64 + 1e-5, 1.0_f64 - 1e-5];
        let err = backward_error_linear(&a.view(), &b.view(), &x.view()).expect("should succeed");
        // Should be small but non-trivial
        assert!(err > 0.0);
        assert!(err < 1e-3);
    }

    #[test]
    fn test_perturbation_bound_zero_perturbation() {
        let a = array![[4.0_f64, 0.0], [0.0, 1.0]];
        let da = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let b = array![4.0_f64, 1.0];
        let db = array![0.0_f64, 0.0];
        let bound = perturbation_bound(&a.view(), &da.view(), &b.view(), &db.view()).expect("should succeed");
        assert!(bound < 1e-14);
    }

    #[test]
    fn test_full_analysis() {
        let a = array![[3.0_f64, 1.0], [1.0, 2.0]];
        let b = array![7.0_f64, 4.0]; // solution: [2, 1] → 3*2+1=7, 1*2+2=4
        let x = array![2.0_f64, 1.0];
        let report = BackwardErrorAnalysis::new()
            .analyze(&a.view(), &b.view(), &x.view())
            .expect("should succeed");
        assert!(report.normwise_backward_error < 1e-13);
        assert!(report.componentwise_backward_error < 1e-13);
        assert!(report.condition_number_1norm >= 1.0);
        assert!(report.residual_norm < 1e-13);
    }

    #[test]
    fn test_shape_errors() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let b = array![1.0_f64, 2.0, 3.0]; // wrong length
        let x = array![1.0_f64, 0.0];
        assert!(backward_error_linear(&a.view(), &b.view(), &x.view()).is_err());

        // Non-square for condition number
        let a_rect = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(condition_number_1norm(&a_rect.view()).is_err());
    }

    #[test]
    fn test_singular_matrix_condition_number() {
        let a = array![[1.0_f64, 2.0], [2.0, 4.0]]; // rank-1 singular
        let k = condition_number_1norm(&a.view()).expect("should succeed");
        assert!(k.is_infinite(), "Expected infinity for singular matrix, got {k}");
    }

    #[test]
    fn test_matvec_correctness() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let x = array![1.0_f64, 1.0];
        let ax = matvec(&a.view(), &x.view());
        assert!((ax[0] - 3.0).abs() < 1e-12);
        assert!((ax[1] - 7.0).abs() < 1e-12);
    }
}
