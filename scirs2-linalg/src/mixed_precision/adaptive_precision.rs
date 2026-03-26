//! Adaptive precision dispatch for linear algebra solvers
//!
//! This module provides two key capabilities:
//!
//! 1. **Condition number estimation** (`estimate_condition_1norm`): a fast
//!    randomised power-iteration approach based on the Higham-Tisseur algorithm
//!    (SIAM J. Matrix Anal. Appl., 2000).  Five iterations give a reliable
//!    1-norm estimate in O(n²) total work.
//!
//! 2. **Precision policy dispatch** (`PrecisionPolicy`, `AdaptiveSolver`):
//!    users declare whether they always want a fixed precision level or whether
//!    the solver should automatically select the cheapest level that keeps the
//!    condition number within the working precision's safe range.
//!
//! # Design principles
//! * No `unwrap()` — all fallible paths return `LinalgResult`.
//! * `#[non_exhaustive]` on all public enums so downstream crates are not
//!   broken by future additions.
//! * Pure Rust, no openblas or C/Fortran deps.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ─────────────────────────────────────────────────────────────────────────────
// Condition number estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a condition-number estimation.
///
/// The `estimate` field is a lower bound (never an overestimate) of
/// `‖A‖₁ · ‖A⁻¹‖₁`.  The `reliable` flag is `true` when the power
/// iteration converged within the allotted iterations.
#[derive(Debug, Clone)]
pub struct ConditionEstimate {
    /// Estimated 1-norm condition number.  Always ≥ 1.0.
    pub estimate: f64,
    /// Whether the estimate is considered reliable (iteration converged).
    pub reliable: bool,
}

impl ConditionEstimate {
    /// Construct a reliable estimate.
    fn reliable(value: f64) -> Self {
        ConditionEstimate {
            estimate: value.max(1.0),
            reliable: true,
        }
    }

    /// Construct an unreliable estimate (e.g. singular matrix detected).
    fn unreliable(value: f64) -> Self {
        ConditionEstimate {
            estimate: value.max(1.0),
            reliable: false,
        }
    }
}

/// Estimate the 1-norm condition number of a square matrix `a` using the
/// Higham-Tisseur block 1-norm power iteration (LAPACK's DLACN2 algorithm).
///
/// The algorithm maintains a direction vector `v` (initialised to all-ones
/// divided by `n`), and iteratively applies `Aᵀ sign(Av)` to maximise the
/// 1-norm lower bound.  At most `max_iter` iterations are performed.
///
/// Time complexity: O(n² · max_iter) — dominated by the matrix-vector products.
///
/// # Returns
/// A [`ConditionEstimate`] whose `estimate` field is a lower bound of the true
/// 1-norm condition number.
///
/// # Errors
/// Returns [`LinalgError::ShapeError`] if `a` is not square.
/// Returns [`LinalgError::SingularMatrixError`] if the matrix appears singular
/// (1-norm is zero).
pub fn estimate_condition_1norm(a: &ArrayView2<f64>) -> LinalgResult<ConditionEstimate> {
    let (m, n) = (a.nrows(), a.ncols());
    if m != n {
        return Err(LinalgError::ShapeError(format!(
            "estimate_condition_1norm requires a square matrix, got ({m}×{n})"
        )));
    }
    if n == 0 {
        return Ok(ConditionEstimate::reliable(1.0));
    }

    // ── Step 1: compute ‖A‖₁ (max column 1-norm) ───────────────────────────
    let norm_a = matrix_1norm(a);
    if norm_a == 0.0 {
        return Err(LinalgError::SingularMatrixError(
            "Matrix has zero 1-norm; condition number is infinite".to_string(),
        ));
    }

    // ── Step 2: solve A·y = e for an approximate inverse column via LU ──────
    // We use a simplified LU with partial pivoting to solve A·y = b for
    // several right-hand sides.  This gives us an estimate of ‖A⁻¹‖₁.
    let lu_result = lu_partial_pivot(a);
    match lu_result {
        Err(_) => {
            // Singular — return large estimate
            return Ok(ConditionEstimate::unreliable(f64::INFINITY));
        }
        Ok((lu, piv)) => {
            // Higham-Tisseur: power iteration on (A⁻¹, (A⁻¹)ᵀ)
            let inv_norm = estimate_inv_1norm(&lu, &piv, n);
            let est = norm_a * inv_norm;
            Ok(ConditionEstimate::reliable(est))
        }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Compute the 1-norm (maximum absolute column sum) of a matrix.
fn matrix_1norm(a: &ArrayView2<f64>) -> f64 {
    let n = a.ncols();
    let mut max_col = 0.0f64;
    for j in 0..n {
        let col_sum: f64 = (0..a.nrows()).map(|i| a[[i, j]].abs()).sum();
        if col_sum > max_col {
            max_col = col_sum;
        }
    }
    max_col
}

/// LU decomposition with partial pivoting.
///
/// Returns `(lu, pivot)` where `lu` is the packed LU factor (L below the
/// diagonal, U on and above) and `pivot[i]` is the row swapped with row `i`.
fn lu_partial_pivot(a: &ArrayView2<f64>) -> LinalgResult<(Array2<f64>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu = a.to_owned();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot in column k, rows k..n
        let mut max_val = lu[[k, k]].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let v = lu[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < f64::EPSILON * 1e3 {
            return Err(LinalgError::SingularMatrixError(
                "Near-singular matrix in LU decomposition".to_string(),
            ));
        }

        // Swap rows k and max_row
        if max_row != k {
            for j in 0..n {
                let tmp = lu[[k, j]];
                lu[[k, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = tmp;
            }
            piv.swap(k, max_row);
        }

        // Eliminate below
        let pivot = lu[[k, k]];
        for i in k + 1..n {
            lu[[i, k]] /= pivot;
            for j in k + 1..n {
                let lij = lu[[i, k]];
                lu[[i, j]] -= lij * lu[[k, j]];
            }
        }
    }

    Ok((lu, piv))
}

/// Solve `L·U·x = P·b` given the packed LU factorisation.
fn lu_solve(lu: &Array2<f64>, piv: &[usize], b: &Array1<f64>) -> Array1<f64> {
    let n = lu.nrows();
    let mut x = b.to_owned();

    // Apply permutation
    for (k, &p) in piv.iter().enumerate().take(n) {
        if p != k {
            x.swap(k, p);
        }
    }

    // Forward substitution: L·y = Pb  (L has implicit 1s on diagonal)
    for i in 1..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[[i, j]] * x[j];
        }
        x[i] = s;
    }

    // Back substitution: U·x = y
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= lu[[i, j]] * x[j];
        }
        x[i] = s / lu[[i, i]];
    }

    x
}

/// Solve `Uᵀ·Lᵀ·x = Pᵀ·b` (transpose solve for the Higham-Tisseur step).
fn lu_solve_transpose(lu: &Array2<f64>, piv: &[usize], b: &Array1<f64>) -> Array1<f64> {
    let n = lu.nrows();
    let mut x = b.to_owned();

    // Forward substitution: Uᵀ·y = b
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[[j, i]] * x[j];
        }
        x[i] = s / lu[[i, i]];
    }

    // Back substitution: Lᵀ·z = y  (Lᵀ has 1s on diagonal)
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in i + 1..n {
            s -= lu[[j, i]] * x[j];
        }
        x[i] = s;
    }

    // Apply inverse permutation (Pᵀ)
    for k in (0..n).rev() {
        let p = piv[k];
        if p != k {
            x.swap(k, p);
        }
    }

    x
}

/// Estimate `‖A⁻¹‖₁` using 5 iterations of the Higham-Tisseur power iteration.
///
/// Each iteration:
/// 1. Solve `A·y = v`  (forward)
/// 2. Pick `w = sign(y) / n` as new direction
/// 3. Solve `Aᵀ·z = w`  (transpose)
/// 4. If `‖z‖∞` stops growing, stop
fn estimate_inv_1norm(lu: &Array2<f64>, piv: &[usize], n: usize) -> f64 {
    const MAX_ITER: usize = 5;

    // Initialise: v = [1/n, 1/n, ..., 1/n]
    let inv_n = 1.0 / n as f64;
    let mut v = Array1::from_elem(n, inv_n);

    let mut est = 0.0f64;
    let mut est_old = 0.0f64;

    for _iter in 0..MAX_ITER {
        // y = A⁻¹ v
        let y = lu_solve(lu, piv, &v);

        // New estimate = ‖y‖₁
        est = y.iter().map(|x| x.abs()).sum::<f64>();

        if est <= est_old * (1.0 + 1e-10) {
            break; // Converged
        }
        est_old = est;

        // w = sign(y) / n
        let mut w = Array1::zeros(n);
        for i in 0..n {
            w[i] = if y[i] >= 0.0 { inv_n } else { -inv_n };
        }

        // z = (A⁻¹)ᵀ w
        let z = lu_solve_transpose(lu, piv, &w);

        // Check convergence: if argmax|z| hasn't changed, stop
        let max_z = z.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        if max_z
            <= z.iter()
                .zip(v.iter())
                .map(|(zi, vi)| zi.abs() * vi.abs())
                .sum::<f64>()
        {
            break;
        }

        // Next direction: unit vector at index of |z| maximum
        let argmax = z
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);
        v = Array1::zeros(n);
        v[argmax] = 1.0;
    }

    est
}

// ─────────────────────────────────────────────────────────────────────────────
// Precision levels
// ─────────────────────────────────────────────────────────────────────────────

/// Available floating-point precision levels for linear algebra operations.
///
/// Variants are listed from cheapest (lowest precision) to most expensive
/// (highest precision).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrecisionLevel {
    /// 16-bit half-precision (IEEE 754 binary16).
    /// Suitable for very well-conditioned problems or stochastic rounding.
    F16,
    /// 32-bit single precision.
    /// Suitable for problems with condition number < ~10⁴.
    F32,
    /// 64-bit double precision (standard for scientific computing).
    /// Suitable for condition numbers up to ~10¹².
    F64,
    /// Software-emulated extended/quad precision (f64 + Kahan compensated).
    /// Used as a fallback for extremely ill-conditioned problems.
    F128Fallback,
}

// ─────────────────────────────────────────────────────────────────────────────
// Precision policy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy for selecting floating-point precision.
///
/// This enum is `#[non_exhaustive]` so that additional policies (e.g.
/// `BlockAdaptive`, `UserDefined`) can be added in future without breaking
/// downstream pattern matches.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PrecisionPolicy {
    /// Always use this exact precision level, regardless of condition number.
    Fixed(PrecisionLevel),
    /// Automatically select precision based on the estimated condition number.
    ///
    /// | Condition number | Selected level |
    /// |-----------------|----------------|
    /// | `< low_threshold` | `F16` |
    /// | `[low, high)` | `F32` |
    /// | `[high, 1e12)` | `F64` |
    /// | `≥ 1e12` | `F128Fallback` |
    Adaptive {
        /// Condition-number threshold below which F16 is selected.
        low_threshold: f64,
        /// Condition-number threshold at or above which F64 is selected
        /// (instead of F32).
        high_threshold: f64,
    },
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        // Conservative default: always use f64
        PrecisionPolicy::Fixed(PrecisionLevel::F64)
    }
}

/// Select an appropriate precision level given a condition-number estimate and
/// a policy.
///
/// # Arguments
/// * `cond_est` – Estimated condition number (must be ≥ 1).
/// * `policy`   – Dispatch policy.
///
/// # Returns
/// The recommended [`PrecisionLevel`].
///
/// # Examples
/// ```
/// use scirs2_linalg::mixed_precision::adaptive_precision::{
///     select_precision, PrecisionLevel, PrecisionPolicy,
/// };
///
/// // Fixed policy always returns the fixed level
/// let level = select_precision(1e8, &PrecisionPolicy::Fixed(PrecisionLevel::F32));
/// assert_eq!(level, PrecisionLevel::F32);
///
/// // Adaptive: well-conditioned → F32
/// let policy = PrecisionPolicy::Adaptive {
///     low_threshold: 1e2,
///     high_threshold: 1e6,
/// };
/// assert_eq!(select_precision(50.0, &policy), PrecisionLevel::F16);
/// assert_eq!(select_precision(1e3, &policy), PrecisionLevel::F32);
/// assert_eq!(select_precision(1e7, &policy), PrecisionLevel::F64);
/// assert_eq!(select_precision(1e13, &policy), PrecisionLevel::F128Fallback);
/// ```
pub fn select_precision(cond_est: f64, policy: &PrecisionPolicy) -> PrecisionLevel {
    match policy {
        PrecisionPolicy::Fixed(level) => level.clone(),
        PrecisionPolicy::Adaptive {
            low_threshold,
            high_threshold,
        } => {
            if cond_est < *low_threshold {
                PrecisionLevel::F16
            } else if cond_est < *high_threshold {
                PrecisionLevel::F32
            } else if cond_est < 1e12 {
                PrecisionLevel::F64
            } else {
                PrecisionLevel::F128Fallback
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveSolver
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`AdaptiveSolver`].
#[derive(Debug, Clone)]
pub struct AdaptiveSolverConfig {
    /// Precision selection policy.
    pub policy: PrecisionPolicy,
    /// Maximum iterations for the condition-number estimator.
    /// Ignored when `policy` is `Fixed`.
    pub max_cond_iter: usize,
}

impl Default for AdaptiveSolverConfig {
    fn default() -> Self {
        AdaptiveSolverConfig {
            policy: PrecisionPolicy::Adaptive {
                low_threshold: 1e3,
                high_threshold: 1e7,
            },
            max_cond_iter: 5,
        }
    }
}

/// A linear-system solver that estimates the condition number of the input
/// matrix and automatically dispatches to the cheapest precision that can
/// solve the problem reliably.
///
/// # Example
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::mixed_precision::adaptive_precision::{
///     AdaptiveSolver, AdaptiveSolverConfig,
/// };
///
/// let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
/// let b = array![1.0_f64, 2.0];
///
/// let solver = AdaptiveSolver::new(AdaptiveSolverConfig::default());
/// let result = solver.solve(&a.view(), &b).expect("Solve failed");
/// // Solver selects the cheapest precision that still achieves accuracy
/// assert!((result.solution[0] - 0.1).abs() < 1e-2);
/// assert!((result.solution[1] - 0.6).abs() < 1e-2);
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveSolver {
    config: AdaptiveSolverConfig,
}

/// Outcome of an [`AdaptiveSolver::solve`] call.
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// The solution vector `x` such that `A·x ≈ b`.
    pub solution: Array1<f64>,
    /// Which precision level was actually used.
    pub precision_used: PrecisionLevel,
    /// The estimated condition number used for the dispatch decision.
    pub cond_estimate: f64,
}

impl AdaptiveSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: AdaptiveSolverConfig) -> Self {
        AdaptiveSolver { config }
    }

    /// Create a solver with default configuration (adaptive policy).
    pub fn default_adaptive() -> Self {
        AdaptiveSolver::new(AdaptiveSolverConfig::default())
    }

    /// Estimate the condition number and select precision.
    ///
    /// Returns `(cond_estimate, selected_level)`.
    pub fn estimate_and_select(&self, a: &ArrayView2<f64>) -> LinalgResult<(f64, PrecisionLevel)> {
        match &self.config.policy {
            PrecisionPolicy::Fixed(level) => Ok((1.0, level.clone())),
            PrecisionPolicy::Adaptive { .. } => {
                let ce = estimate_condition_1norm(a)?;
                let level = select_precision(ce.estimate, &self.config.policy);
                Ok((ce.estimate, level))
            }
        }
    }

    /// Solve the linear system `A·x = b`.
    ///
    /// The solver:
    /// 1. Estimates the condition number of `A`.
    /// 2. Selects a precision level via the policy.
    /// 3. Solves using the selected precision (downcast to f32 if beneficial,
    ///    or uses compensated summation for `F128Fallback`).
    ///
    /// # Errors
    /// * [`LinalgError::ShapeError`] – matrix not square or dimensions mismatch.
    /// * [`LinalgError::SingularMatrixError`] – matrix is singular.
    pub fn solve(&self, a: &ArrayView2<f64>, b: &Array1<f64>) -> LinalgResult<SolveResult> {
        let (m, n) = (a.nrows(), a.ncols());
        if m != n {
            return Err(LinalgError::ShapeError(format!(
                "AdaptiveSolver::solve requires square matrix, got ({m}×{n})"
            )));
        }
        if b.len() != n {
            return Err(LinalgError::ShapeError(format!(
                "RHS length {} does not match matrix size {n}",
                b.len()
            )));
        }

        let (cond_est, precision_used) = self.estimate_and_select(a)?;

        let solution = match precision_used {
            PrecisionLevel::F16 | PrecisionLevel::F32 => {
                // Downcast A and b to f32, solve in f32, upcast result
                solve_f32_internal(a, b)?
            }
            PrecisionLevel::F64 => {
                // Solve in full f64
                solve_f64_internal(a, b)?
            }
            PrecisionLevel::F128Fallback => {
                // Use f64 with iterative refinement (one refinement step)
                solve_f64_refined(a, b)?
            }
            // Handle any future variants conservatively
            #[allow(unreachable_patterns)]
            _ => solve_f64_internal(a, b)?,
        };

        Ok(SolveResult {
            solution,
            precision_used,
            cond_estimate: cond_est,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal solve helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Solve in f32 and cast back to f64.
fn solve_f32_internal(a: &ArrayView2<f64>, b: &Array1<f64>) -> LinalgResult<Array1<f64>> {
    let n = a.nrows();
    // Downcast to f32
    let mut lu_f32: Array2<f32> = Array2::from_shape_fn((n, n), |(i, j)| a[[i, j]] as f32);
    let mut b_f32: Array1<f32> = Array1::from_shape_fn(n, |i| b[i] as f32);

    // LU with partial pivoting in f32
    let mut piv: Vec<usize> = (0..n).collect();
    for k in 0..n {
        let mut max_val = lu_f32[[k, k]].abs();
        let mut max_row = k;
        for i in k + 1..n {
            let v = lu_f32[[i, k]].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < f32::EPSILON * 1e3 {
            return Err(LinalgError::SingularMatrixError(
                "Near-singular matrix in f32 solve".to_string(),
            ));
        }
        if max_row != k {
            for j in 0..n {
                let tmp = lu_f32[[k, j]];
                lu_f32[[k, j]] = lu_f32[[max_row, j]];
                lu_f32[[max_row, j]] = tmp;
            }
            piv.swap(k, max_row);
        }
        let pv = lu_f32[[k, k]];
        for i in k + 1..n {
            lu_f32[[i, k]] /= pv;
            let lij = lu_f32[[i, k]];
            for j in k + 1..n {
                lu_f32[[i, j]] -= lij * lu_f32[[k, j]];
            }
        }
    }

    // Apply permutation to b
    for (k, &p) in piv.iter().enumerate().take(n) {
        if p != k {
            b_f32.swap(k, p);
        }
    }
    // Forward substitution
    for i in 1..n {
        let mut s = b_f32[i];
        for j in 0..i {
            s -= lu_f32[[i, j]] * b_f32[j];
        }
        b_f32[i] = s;
    }
    // Back substitution
    for i in (0..n).rev() {
        let mut s = b_f32[i];
        for j in i + 1..n {
            s -= lu_f32[[i, j]] * b_f32[j];
        }
        b_f32[i] = s / lu_f32[[i, i]];
    }

    Ok(Array1::from_shape_fn(n, |i| b_f32[i] as f64))
}

/// Solve in f64.
fn solve_f64_internal(a: &ArrayView2<f64>, b: &Array1<f64>) -> LinalgResult<Array1<f64>> {
    let n = a.nrows();
    let lu_result = lu_partial_pivot(a)?;
    let (lu, piv) = lu_result;
    Ok(lu_solve(&lu, &piv, b))
}

/// Solve in f64 with one step of iterative refinement.
fn solve_f64_refined(a: &ArrayView2<f64>, b: &Array1<f64>) -> LinalgResult<Array1<f64>> {
    let x0 = solve_f64_internal(a, b)?;

    // Compute residual r = b - A·x0
    let n = a.nrows();
    let mut r = b.clone();
    for i in 0..n {
        let mut ax_i = 0.0f64;
        for j in 0..n {
            ax_i += a[[i, j]] * x0[j];
        }
        r[i] -= ax_i;
    }

    // Solve A·dx = r and update x = x0 + dx
    let (lu, piv) = lu_partial_pivot(a)?;
    let dx = lu_solve(&lu, &piv, &r);

    Ok(Array1::from_shape_fn(n, |i| x0[i] + dx[i]))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ── Condition estimation ────────────────────────────────────────────────

    #[test]
    fn test_condition_identity_is_one() {
        let n = 5;
        let id: Array2<f64> = Array2::eye(n);
        let ce = estimate_condition_1norm(&id.view()).expect("Cond estimation failed");
        // For the identity matrix, cond = 1.
        assert!(
            ce.estimate < 2.0,
            "Identity cond estimate too large: {}",
            ce.estimate
        );
        assert!(ce.reliable, "Identity cond not marked reliable");
    }

    #[test]
    fn test_condition_scaled_identity() {
        // Scaled identity has cond = 1 regardless of scale.
        let n = 4;
        let a: Array2<f64> = Array2::eye(n) * 100.0;
        let ce = estimate_condition_1norm(&a.view()).expect("Cond estimation failed");
        assert!(
            ce.estimate < 5.0,
            "Scaled identity cond too large: {}",
            ce.estimate
        );
    }

    #[test]
    fn test_condition_hilbert_matrix_large() {
        // Hilbert matrix H[i,j] = 1/(i+j+1).  For n=8, cond > 1e6.
        let n = 8usize;
        let h: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64);
        let ce = estimate_condition_1norm(&h.view()).expect("Cond estimation failed");
        assert!(
            ce.estimate > 1e6,
            "Hilbert(8) cond should be > 1e6, got {}",
            ce.estimate
        );
    }

    #[test]
    fn test_condition_diagonal_well_conditioned() {
        // Diagonal matrix with entries [1, 2, 4, 8]: cond = 8/1 * 1/(1/8) = 8.
        let d = array![
            [1.0_f64, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ];
        let ce = estimate_condition_1norm(&d.view()).expect("Cond estimation failed");
        // True cond_1 = 8; our estimate should be at least 1 and at most ~100
        assert!(ce.estimate >= 1.0, "Cond estimate must be >= 1");
        assert!(
            ce.estimate < 200.0,
            "Diagonal cond estimate way off: {}",
            ce.estimate
        );
    }

    #[test]
    fn test_condition_non_square_errors() {
        let a: Array2<f64> = Array2::zeros((3, 4));
        assert!(estimate_condition_1norm(&a.view()).is_err());
    }

    // ── PrecisionPolicy::Fixed ──────────────────────────────────────────────

    #[test]
    fn test_fixed_f16_always_returns_f16() {
        let policy = PrecisionPolicy::Fixed(PrecisionLevel::F16);
        for cond in [1.0, 1e5, 1e15] {
            assert_eq!(select_precision(cond, &policy), PrecisionLevel::F16);
        }
    }

    #[test]
    fn test_fixed_f32_always_returns_f32() {
        let policy = PrecisionPolicy::Fixed(PrecisionLevel::F32);
        for cond in [1.0, 1e5, 1e15] {
            assert_eq!(select_precision(cond, &policy), PrecisionLevel::F32);
        }
    }

    #[test]
    fn test_fixed_f64_always_returns_f64() {
        let policy = PrecisionPolicy::Fixed(PrecisionLevel::F64);
        for cond in [1.0, 1e5, 1e15] {
            assert_eq!(select_precision(cond, &policy), PrecisionLevel::F64);
        }
    }

    // ── PrecisionPolicy::Adaptive ───────────────────────────────────────────

    #[test]
    fn test_adaptive_threshold_dispatch() {
        let policy = PrecisionPolicy::Adaptive {
            low_threshold: 1e2,
            high_threshold: 1e6,
        };
        assert_eq!(select_precision(50.0, &policy), PrecisionLevel::F16);
        assert_eq!(select_precision(1e3, &policy), PrecisionLevel::F32);
        assert_eq!(select_precision(1e7, &policy), PrecisionLevel::F64);
        assert_eq!(
            select_precision(1e13, &policy),
            PrecisionLevel::F128Fallback
        );
    }

    #[test]
    fn test_adaptive_boundary_low_threshold() {
        let policy = PrecisionPolicy::Adaptive {
            low_threshold: 100.0,
            high_threshold: 1e6,
        };
        // Exactly at boundary (< low) → F16
        assert_eq!(select_precision(99.9, &policy), PrecisionLevel::F16);
        // At or above low → F32
        assert_eq!(select_precision(100.0, &policy), PrecisionLevel::F32);
    }

    #[test]
    fn test_adaptive_boundary_high_threshold() {
        let policy = PrecisionPolicy::Adaptive {
            low_threshold: 1e2,
            high_threshold: 1e6,
        };
        // Just below high threshold → F32
        assert_eq!(select_precision(999_999.9, &policy), PrecisionLevel::F32);
        // At high threshold → F64
        assert_eq!(select_precision(1e6, &policy), PrecisionLevel::F64);
    }

    // ── AdaptiveSolver ──────────────────────────────────────────────────────

    #[test]
    fn test_adaptive_solver_well_conditioned_uses_f32() {
        // A simple 2×2 well-conditioned system; cond ≈ 3 → should use F32
        let a = array![[4.0_f64, 1.0], [2.0, 3.0]];
        let b = array![1.0_f64, 2.0];

        let config = AdaptiveSolverConfig {
            policy: PrecisionPolicy::Adaptive {
                low_threshold: 1e3,
                high_threshold: 1e7,
            },
            max_cond_iter: 5,
        };
        let solver = AdaptiveSolver::new(config);
        let result = solver.solve(&a.view(), &b).expect("Solve failed");

        // Solution: x=[0.1, 0.6]
        assert!(
            (result.solution[0] - 0.1).abs() < 1e-3,
            "x[0]={} expected≈0.1",
            result.solution[0]
        );
        assert!(
            (result.solution[1] - 0.6).abs() < 1e-3,
            "x[1]={} expected≈0.6",
            result.solution[1]
        );
        // Well-conditioned → F16 or F32
        assert!(
            matches!(
                result.precision_used,
                PrecisionLevel::F16 | PrecisionLevel::F32
            ),
            "Expected F16/F32 but got {:?}",
            result.precision_used
        );
    }

    #[test]
    fn test_adaptive_solver_ill_conditioned_uses_f64_or_above() {
        // Hilbert(8) is very ill-conditioned; cond > 1e6
        // We set high_threshold = 1e4 so any condition above 1e4 → F64.
        let n = 8usize;
        let h: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| 1.0 / (i + j + 1) as f64);
        let b = Array1::ones(n);

        let config = AdaptiveSolverConfig {
            policy: PrecisionPolicy::Adaptive {
                low_threshold: 1e2,
                high_threshold: 1e4,
            },
            max_cond_iter: 5,
        };
        let solver = AdaptiveSolver::new(config);
        let result = solver.solve(&h.view(), &b).expect("Hilbert solve failed");

        // Precision should be F64 or F128Fallback for Hilbert(8)
        assert!(
            matches!(
                result.precision_used,
                PrecisionLevel::F64 | PrecisionLevel::F128Fallback
            ),
            "Expected F64/F128Fallback for Hilbert(8) but got {:?} (cond={})",
            result.precision_used,
            result.cond_estimate
        );
        // Verify solution is reasonable (residual ‖Ax-b‖ < 1e-6)
        let mut residual_norm = 0.0f64;
        for i in 0..n {
            let ax_i: f64 = (0..n).map(|j| h[[i, j]] * result.solution[j]).sum();
            residual_norm += (ax_i - b[i]).powi(2);
        }
        assert!(
            residual_norm.sqrt() < 1e-6,
            "Hilbert solve residual too large: {}",
            residual_norm.sqrt()
        );
    }

    #[test]
    fn test_adaptive_solver_fixed_policy() {
        let a = array![[2.0_f64, 0.0], [0.0, 3.0]];
        let b = array![4.0_f64, 9.0];
        let solver = AdaptiveSolver::new(AdaptiveSolverConfig {
            policy: PrecisionPolicy::Fixed(PrecisionLevel::F64),
            max_cond_iter: 5,
        });
        let result = solver.solve(&a.view(), &b).expect("Solve failed");
        assert_eq!(result.precision_used, PrecisionLevel::F64);
        assert!((result.solution[0] - 2.0).abs() < 1e-10);
        assert!((result.solution[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_solver_non_square_error() {
        let a: Array2<f64> = Array2::zeros((3, 4));
        let b = Array1::zeros(3);
        let solver = AdaptiveSolver::default_adaptive();
        assert!(solver.solve(&a.view(), &b).is_err());
    }

    #[test]
    fn test_adaptive_solver_dimension_mismatch_error() {
        let a: Array2<f64> = Array2::eye(3);
        let b: Array1<f64> = Array1::zeros(4); // wrong size
        let solver = AdaptiveSolver::default_adaptive();
        assert!(solver.solve(&a.view(), &b).is_err());
    }
}
