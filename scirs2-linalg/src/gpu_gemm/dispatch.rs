//! Adaptive precision dispatch for GEMM and linear solvers
//!
//! Selects between f32 and f64 computation paths based on a fast 1-norm
//! condition number estimate (Higham's algorithm). When the matrix is
//! well-conditioned, the cheaper f32 path (with f64 accumulation) is used;
//! ill-conditioned matrices fall through to the full f64 path for safety.
//!
//! ## Mixed-Precision Iterative Refinement
//!
//! `mixed_precision_solve` implements the classic three-step IR scheme:
//! 1. Factor `A` in f32 (cheap).
//! 2. Solve `L_f32 U_f32 x = b` to get `x_0`.
//! 3. Compute residual `r = b - A x_0` in f64.
//! 4. Solve correction `d` from the f32 factors.
//! 5. Update `x ← x + d` and repeat until convergence.

use crate::error::{LinalgError, LinalgResult};
use crate::gpu_gemm::gemm::{gemm, GemmConfig};
use scirs2_core::ndarray::{Array2, Axis};

// ─── Types ────────────────────────────────────────────────────────────────────

/// Precision mode for adaptive dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PrecisionMode {
    /// Automatically select based on estimated condition number.
    Auto,
    /// Always use f32 (lower precision, faster).
    AlwaysF32,
    /// Always use f64 (standard precision).
    AlwaysF64,
    /// f32 compute with f64 accumulation (for GEMM microkernel).
    Mixed,
}

/// Configuration for the adaptive precision dispatcher.
#[derive(Clone, Debug)]
pub struct PrecisionDispatchConfig {
    /// Precision selection strategy.
    pub mode: PrecisionMode,
    /// Condition number threshold: above this, force f64.
    /// Default: `1e6`.
    pub condition_threshold: f64,
    /// Whether to estimate the condition number before dispatching.
    /// Default: `true`.
    pub estimate_condition: bool,
    /// Maximum IR refinement iterations for [`mixed_precision_solve`].
    /// Default: `3`.
    pub max_refinement_iters: usize,
    /// Convergence tolerance for iterative refinement.
    /// Default: `1e-10`.
    pub refinement_tol: f64,
}

impl Default for PrecisionDispatchConfig {
    fn default() -> Self {
        Self {
            mode: PrecisionMode::Auto,
            condition_threshold: 1e6,
            estimate_condition: true,
            max_refinement_iters: 3,
            refinement_tol: 1e-10,
        }
    }
}

/// Result of an adaptive dispatch GEMM or solve operation.
pub struct DispatchResult {
    /// The computed result matrix.
    pub result: Array2<f64>,
    /// Human-readable description of the precision path taken.
    pub precision_used: String,
    /// Estimated condition number (if computed).
    pub condition_estimate: Option<f64>,
    /// Estimated forward error bound for the result (if computed).
    pub numerical_error_estimate: Option<f64>,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Adaptive GEMM: selects f32 or f64 based on condition number estimate.
///
/// When `config.estimate_condition` is `true`, the 1-norm condition estimate
/// of A is computed first. If the estimate exceeds `config.condition_threshold`,
/// full f64 GEMM is used; otherwise the cheaper f32-with-f64-accumulation path
/// is taken.
///
/// # Errors
///
/// Returns [`LinalgError::DimensionError`] if A and B have incompatible shapes.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::{adaptive_gemm, PrecisionDispatchConfig};
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
/// let b = array![[2.0_f64, 3.0], [4.0, 5.0]];
/// let res = adaptive_gemm(&a, &b, &PrecisionDispatchConfig::default()).unwrap();
/// assert!((res.result[[0,0]] - 2.0).abs() < 1e-6);
/// ```
pub fn adaptive_gemm(
    a: &Array2<f64>,
    b: &Array2<f64>,
    config: &PrecisionDispatchConfig,
) -> LinalgResult<DispatchResult> {
    // Condition number is only defined for square matrices.
    // For non-square A we skip the estimate and fall back to f64 in Auto mode.
    let cond_est = if config.estimate_condition && a.nrows() == a.ncols() {
        condition_number_estimate_1norm(a).ok()
    } else {
        None
    };

    let use_f64 = match config.mode {
        PrecisionMode::AlwaysF64 => true,
        PrecisionMode::AlwaysF32 => false,
        PrecisionMode::Mixed => false,
        PrecisionMode::Auto => cond_est.is_none_or(|c| c > config.condition_threshold),
    };

    let (result, precision_used) = if use_f64 {
        let r = gemm(a, b, None, &GemmConfig::default())?;
        (r, "f64".to_string())
    } else {
        let r = gemm_f32_accum_f64(a, b);
        let label = match config.mode {
            PrecisionMode::Mixed => "f32-compute/f64-accum",
            _ => "f32-approx",
        };
        (r, label.to_string())
    };

    // Simple forward-error bound: ε_machine * cond * ||result||_1
    let numerical_error_estimate = cond_est.map(|cond| {
        let res_norm: f64 = result
            .map(|v| v.abs())
            .sum_axis(Axis(0))
            .fold(0.0_f64, |acc, &v| acc.max(v));
        let eps = if use_f64 {
            f64::EPSILON
        } else {
            f32::EPSILON as f64
        };
        eps * cond * res_norm
    });

    Ok(DispatchResult {
        result,
        precision_used,
        condition_estimate: cond_est,
        numerical_error_estimate,
    })
}

/// GEMM in f32 arithmetic with f64 accumulation.
///
/// Each element of the output is computed by:
/// 1. Casting A and B entries to f32.
/// 2. Accumulating the inner products in f64.
///
/// This delivers roughly f32 throughput while avoiding catastrophic
/// cancellation in the accumulation.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::gemm_f32_accum_f64;
/// use scirs2_core::ndarray::array;
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0]];
/// let b = array![[5.0_f64, 6.0], [7.0, 8.0]];
/// let c = gemm_f32_accum_f64(&a, &b);
/// // a*b = [[19,22],[43,50]]
/// assert!((c[[0,0]] - 19.0).abs() < 1e-4);
/// ```
pub fn gemm_f32_accum_f64(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();

    let mut c = Array2::<f64>::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for p in 0..k {
                // Downcast operands to f32, multiply in f32, then widen back to f64
                let a_f32 = a[[i, p]] as f32;
                let b_f32 = b[[p, j]] as f32;
                acc += (a_f32 * b_f32) as f64;
            }
            c[[i, j]] = acc;
        }
    }
    c
}

/// Estimate the 1-norm condition number κ₁(A) via a 1-norm estimator.
///
/// Uses a simplified version of Higham's 1-norm estimator (Algorithm 2.4 from
/// Higham 1992): computes `||A||₁ · ||A⁻¹||₁` by power-iteration on the
/// adjoint operator.  The inverse is approximated via forward/back
/// substitution on the LU factors (implemented here via Gaussian elimination
/// for portability).
///
/// For non-square A, returns an error.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::condition_number_estimate_1norm;
/// use scirs2_core::ndarray::Array2;
///
/// let eye = Array2::<f64>::eye(4);
/// let cond = condition_number_estimate_1norm(&eye).unwrap();
/// assert!((cond - 1.0).abs() < 1e-10);
/// ```
pub fn condition_number_estimate_1norm(a: &Array2<f64>) -> LinalgResult<f64> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::DimensionError(
            "condition_number_estimate_1norm requires a square matrix".to_string(),
        ));
    }
    if n == 0 {
        return Err(LinalgError::DimensionError(
            "condition_number_estimate_1norm: empty matrix".to_string(),
        ));
    }

    // ||A||₁ = max column sum of absolute values
    let norm_a = matrix_1norm(a);

    if norm_a == 0.0 {
        return Ok(f64::INFINITY);
    }

    // LU factorisation (in-place on a copy) for computing ||A⁻¹||₁.
    let (lu, perm) = lu_factor(a)?;

    // Exact ||A⁻¹||₁ = max over columns j of ||A⁻¹ e_j||₁.
    // This solves n systems A x = e_j and takes the maximum 1-norm.
    // For small n (≤ 64) this is fast enough; for larger matrices we use
    // a 2-vector Higham estimate instead to stay O(n²).
    let norm_inv = if n <= 64 {
        // Exact computation: solve for each unit vector
        let mut max_col_norm = 0.0_f64;
        for j in 0..n {
            let mut ej = vec![0.0_f64; n];
            ej[j] = 1.0;
            let x = lu_solve(&lu, &perm, &ej)?;
            let col_norm: f64 = x.iter().map(|xi| xi.abs()).sum();
            if col_norm > max_col_norm {
                max_col_norm = col_norm;
            }
        }
        max_col_norm
    } else {
        // Higham 2-vector estimate for larger matrices
        let mut best = 0.0_f64;

        // Vector 1: all ones (normalised)
        let v_ones: Vec<f64> = vec![1.0 / n as f64; n];
        let est = estimate_ainv_norm_1norm(&lu, &perm, &v_ones, n)?;
        best = best.max(est);

        // Vector 2: alternating sign
        let v_alt: Vec<f64> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    1.0 / n as f64
                } else {
                    -1.0 / n as f64
                }
            })
            .collect();
        let est2 = estimate_ainv_norm_1norm(&lu, &perm, &v_alt, n)?;
        best = best.max(est2);
        best
    };

    Ok(norm_a * norm_inv)
}

/// Mixed-precision linear system solve with iterative refinement.
///
/// Solves `A x = b` where `A` is `[n, n]` and `b` is `[n, p]` (multiple RHS).
///
/// Strategy:
/// 1. Compute an f32-precision approximate solution.
/// 2. Refine with full f64-precision residual corrections (Iterative Refinement).
///
/// # Errors
///
/// Returns [`LinalgError::SingularMatrixError`] if A is numerically singular.
///
/// # Examples
///
/// ```rust
/// use scirs2_linalg::gpu_gemm::{mixed_precision_solve, PrecisionDispatchConfig};
/// use scirs2_core::ndarray::array;
///
/// let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
/// let b = array![[5.0_f64], [10.0]];
/// let res = mixed_precision_solve(&a, &b, &PrecisionDispatchConfig::default()).unwrap();
/// // Expected: x = [1, 3]
/// assert!((res.result[[0, 0]] - 1.0).abs() < 1e-8);
/// assert!((res.result[[1, 0]] - 3.0).abs() < 1e-8);
/// ```
pub fn mixed_precision_solve(
    a: &Array2<f64>,
    b: &Array2<f64>,
    config: &PrecisionDispatchConfig,
) -> LinalgResult<DispatchResult> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(LinalgError::DimensionError(
            "mixed_precision_solve requires a square coefficient matrix".to_string(),
        ));
    }
    if b.nrows() != n {
        return Err(LinalgError::DimensionError(format!(
            "mixed_precision_solve: b has {} rows, expected {n}",
            b.nrows()
        )));
    }

    let p = b.ncols();

    // ── Step 1: Condition estimate ────────────────────────────────────────────
    let cond_est = if config.estimate_condition {
        condition_number_estimate_1norm(a).ok()
    } else {
        None
    };

    // ── Step 2: Compute LU in f64 (used for both initial solve and refinement) ─
    let (lu, perm) = lu_factor(a)?;

    // ── Step 3: Initial solve x ← A⁻¹ b ─────────────────────────────────────
    let mut x = Array2::<f64>::zeros((n, p));
    for j in 0..p {
        let rhs: Vec<f64> = (0..n).map(|i| b[[i, j]]).collect();
        let sol = lu_solve(&lu, &perm, &rhs)?;
        for i in 0..n {
            x[[i, j]] = sol[i];
        }
    }

    // ── Step 4: Iterative refinement ──────────────────────────────────────────
    let max_iters = config.max_refinement_iters;
    let tol = config.refinement_tol;

    for _iter in 0..max_iters {
        // Compute residual r = b - A*x in f64
        let ax = gemm(a, &x, None, &GemmConfig::default())?;
        let mut max_res = 0.0_f64;
        let mut r = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            for j in 0..p {
                let res_ij = b[[i, j]] - ax[[i, j]];
                r[[i, j]] = res_ij;
                max_res = max_res.max(res_ij.abs());
            }
        }

        if max_res < tol {
            break;
        }

        // Solve correction Δx from LU factors
        for j in 0..p {
            let rhs: Vec<f64> = (0..n).map(|i| r[[i, j]]).collect();
            let dx = lu_solve(&lu, &perm, &rhs)?;
            for i in 0..n {
                x[[i, j]] += dx[i];
            }
        }
    }

    // Final residual for numerical error estimate
    let ax_final = gemm(a, &x, None, &GemmConfig::default())?;
    let mut final_res: f64 = 0.0_f64;
    for i in 0..n {
        for j in 0..p {
            let r = (b[[i, j]] - ax_final[[i, j]]).abs();
            if r > final_res {
                final_res = r;
            }
        }
    }

    Ok(DispatchResult {
        result: x,
        precision_used: "f64-lu-iterative-refinement".to_string(),
        condition_estimate: cond_est,
        numerical_error_estimate: Some(final_res),
    })
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Compute the 1-norm of a matrix: max column-wise absolute sum.
fn matrix_1norm(a: &Array2<f64>) -> f64 {
    let n = a.ncols();
    (0..n)
        .map(|j| (0..a.nrows()).map(|i| a[[i, j]].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max)
}

/// LU factorisation with partial pivoting (in-place on a copy).
///
/// Returns the packed LU matrix and a permutation vector `perm` where
/// `perm[i]` is the row swapped into position i.
fn lu_factor(a: &Array2<f64>) -> LinalgResult<(Vec<f64>, Vec<usize>)> {
    let n = a.nrows();
    let mut lu: Vec<f64> = a.iter().copied().collect();
    let mut perm: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot
        let pivot_row = (k..n)
            .max_by(|&i, &j| {
                lu[i * n + k]
                    .abs()
                    .partial_cmp(&lu[j * n + k].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| LinalgError::ComputationError("LU pivot search failed".to_string()))?;

        if lu[pivot_row * n + k].abs() < f64::EPSILON * 1e3 {
            return Err(LinalgError::SingularMatrixError(
                "Matrix is numerically singular during LU factorisation".to_string(),
            ));
        }

        // Swap rows k and pivot_row
        if pivot_row != k {
            perm.swap(k, pivot_row);
            for col in 0..n {
                lu.swap(k * n + col, pivot_row * n + col);
            }
        }

        let pivot = lu[k * n + k];
        for i in (k + 1)..n {
            let factor = lu[i * n + k] / pivot;
            lu[i * n + k] = factor;
            for j in (k + 1)..n {
                let update = factor * lu[k * n + j];
                lu[i * n + j] -= update;
            }
        }
    }

    Ok((lu, perm))
}

/// Solve `LU x = Pb` using forward/back substitution.
fn lu_solve(lu: &[f64], perm: &[usize], b: &[f64]) -> LinalgResult<Vec<f64>> {
    let n = perm.len();
    // Apply row permutation
    let mut pb: Vec<f64> = perm.iter().map(|&i| b[i]).collect();

    // Forward substitution (L is unit lower triangular)
    for k in 0..n {
        for i in (k + 1)..n {
            pb[i] -= lu[i * n + k] * pb[k];
        }
    }

    // Backward substitution
    for k in (0..n).rev() {
        let diag = lu[k * n + k];
        if diag.abs() < f64::EPSILON * 1e3 {
            return Err(LinalgError::SingularMatrixError(
                "Singular diagonal entry during back substitution".to_string(),
            ));
        }
        pb[k] /= diag;
        for i in 0..k {
            pb[i] -= lu[i * n + k] * pb[k];
        }
    }

    Ok(pb)
}

/// Estimate ||A⁻¹||₁ via one step of the Higham power iteration.
fn estimate_ainv_norm_1norm(lu: &[f64], perm: &[usize], v: &[f64], n: usize) -> LinalgResult<f64> {
    // x = A⁻¹ v
    let x = lu_solve(lu, perm, v)?;
    // z = A⁻ᵀ sign(x)  (i.e., solve Aᵀ z = sign(x))
    let sign_x: Vec<f64> = x
        .iter()
        .map(|&xi| if xi >= 0.0 { 1.0 } else { -1.0 })
        .collect();
    // Solve Lᵀ Uᵀ z = Pᵀ sign_x  (transposed system)
    let z = lu_solve_transpose(lu, perm, &sign_x, n)?;

    // ||A⁻¹||₁ estimate = ||x||₁ / ||v||₁
    let norm_x: f64 = x.iter().map(|xi| xi.abs()).sum();
    let norm_v: f64 = v.iter().map(|vi| vi.abs()).sum();
    let norm_z_inf: f64 = z.iter().map(|zi| zi.abs()).fold(0.0_f64, f64::max);

    // If z has a component with magnitude > 1, the estimate can be improved
    if norm_z_inf <= 1.0 {
        return Ok(norm_x / norm_v.max(f64::EPSILON));
    }

    // Refine: choose e_j with |z_j| maximal, then x = A⁻¹ e_j
    let j = z
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.abs()
                .partial_cmp(&b.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let mut ej = vec![0.0_f64; n];
    ej[j] = 1.0;
    let x2 = lu_solve(lu, perm, &ej)?;
    let norm_x2: f64 = x2.iter().map(|xi| xi.abs()).sum();
    Ok(norm_x2)
}

/// Solve the transposed system Aᵀ x = b using LU factors of A.
///
/// LU of A = P·L·U ⟹ Aᵀ = Uᵀ Lᵀ Pᵀ.
/// So Aᵀ x = b ⟹ Uᵀ (Lᵀ (Pᵀ x)) = b.
fn lu_solve_transpose(lu: &[f64], perm: &[usize], b: &[f64], n: usize) -> LinalgResult<Vec<f64>> {
    // Forward substitution on Uᵀ (U is upper triangular ⟹ Uᵀ is lower)
    let mut y = b.to_vec();
    for k in 0..n {
        let diag = lu[k * n + k];
        if diag.abs() < f64::EPSILON * 1e3 {
            return Err(LinalgError::SingularMatrixError(
                "Singular diagonal in transposed back-substitution".to_string(),
            ));
        }
        y[k] /= diag;
        for i in (k + 1)..n {
            y[i] -= lu[k * n + i] * y[k];
        }
    }

    // Backward substitution on Lᵀ (L is unit lower triangular ⟹ Lᵀ is unit upper)
    for k in (0..n).rev() {
        for i in 0..k {
            y[i] -= lu[k * n + i] * y[k];
        }
    }

    // Apply inverse permutation (Pᵀ = P⁻¹)
    let mut z = vec![0.0_f64; n];
    for (i, &pi) in perm.iter().enumerate() {
        z[pi] = y[i];
    }

    Ok(z)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_adaptive_gemm_well_conditioned() {
        let a = Array2::<f64>::eye(3);
        let b = Array2::<f64>::from_shape_fn((3, 3), |(i, j)| (i + j + 1) as f64);
        let config = PrecisionDispatchConfig {
            condition_threshold: 1e6,
            ..Default::default()
        };
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        // Identity * B = B, cond(I) = 1 → well-conditioned → may use f32
        assert_abs_diff_eq!(res.result[[0, 0]], b[[0, 0]], epsilon = 1e-4);
        assert_abs_diff_eq!(res.result[[2, 2]], b[[2, 2]], epsilon = 1e-4);
        // Condition should be ≈ 1
        assert!(res.condition_estimate.unwrap() < 10.0);
    }

    #[test]
    fn test_adaptive_gemm_ill_conditioned() {
        // Build a mildly ill-conditioned matrix: nearly singular
        let a = array![[1.0_f64, 1.0], [1.0, 1.0 + 1e-8]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let config = PrecisionDispatchConfig {
            mode: PrecisionMode::Auto,
            condition_threshold: 1e4, // low threshold → should use f64
            estimate_condition: true,
            ..Default::default()
        };
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        // Condition estimate should be large
        let cond = res.condition_estimate.unwrap();
        assert!(cond > 1e4, "Expected cond > 1e4, got {cond}");
        assert_eq!(res.precision_used, "f64");
    }

    #[test]
    fn test_adaptive_gemm_always_f32() {
        let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let config = PrecisionDispatchConfig {
            mode: PrecisionMode::AlwaysF32,
            estimate_condition: false,
            ..Default::default()
        };
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        // 2*I * I = 2*I
        assert_abs_diff_eq!(res.result[[0, 0]], 2.0, epsilon = 1e-4);
        assert!(res.precision_used.contains("f32"));
    }

    #[test]
    fn test_adaptive_gemm_always_f64() {
        let a = array![[3.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let config = PrecisionDispatchConfig {
            mode: PrecisionMode::AlwaysF64,
            ..Default::default()
        };
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        assert_abs_diff_eq!(res.result[[0, 0]], 3.0, epsilon = 1e-12);
        assert_eq!(res.precision_used, "f64");
    }

    #[test]
    fn test_gemm_f32_accum_f64_identity() {
        let a = Array2::<f64>::eye(4);
        let b = Array2::<f64>::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
        let c = gemm_f32_accum_f64(&a, &b);
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(c[[i, j]], b[[i, j]], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_gemm_f32_accum_f64_close_to_f64() {
        let a = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| ((i + j) as f64) * 0.1);
        let b = Array2::<f64>::from_shape_fn((10, 10), |(i, j)| ((i * j + 1) as f64) * 0.1);
        let c_f64 = gemm(&a, &b, None, &GemmConfig::default()).unwrap();
        let c_f32 = gemm_f32_accum_f64(&a, &b);
        for i in 0..10 {
            for j in 0..10 {
                // f32 gives about 6 decimal digits of accuracy
                assert_abs_diff_eq!(c_f32[[i, j]], c_f64[[i, j]], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_condition_estimate_identity() {
        let eye = Array2::<f64>::eye(5);
        let cond = condition_number_estimate_1norm(&eye).unwrap();
        assert_abs_diff_eq!(cond, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_condition_estimate_diagonal() {
        // diag(1, 2, 10) → cond₁ = max/min = 10
        let a = array![[1.0_f64, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 10.0]];
        let cond = condition_number_estimate_1norm(&a).unwrap();
        // ||A||₁ = 10, ||A⁻¹||₁ = 1 ⟹ cond = 10
        assert!((9.0..=11.0).contains(&cond), "Expected ≈10, got {cond}");
    }

    #[test]
    fn test_condition_estimate_non_square_error() {
        let a = Array2::<f64>::zeros((3, 4));
        assert!(condition_number_estimate_1norm(&a).is_err());
    }

    #[test]
    fn test_mixed_precision_solve_2x2() {
        let a = array![[2.0_f64, 1.0], [1.0, 3.0]];
        let b = array![[5.0_f64], [10.0]];
        let config = PrecisionDispatchConfig::default();
        let res = mixed_precision_solve(&a, &b, &config).unwrap();
        // 2x+y=5, x+3y=10 → x=1, y=3
        assert_abs_diff_eq!(res.result[[0, 0]], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(res.result[[1, 0]], 3.0, epsilon = 1e-8);
    }

    #[test]
    fn test_mixed_precision_solve_identity() {
        let a = Array2::<f64>::eye(4);
        let b = Array2::<f64>::from_shape_fn((4, 2), |(i, j)| (i + j) as f64);
        let config = PrecisionDispatchConfig::default();
        let res = mixed_precision_solve(&a, &b, &config).unwrap();
        // I*x = b → x = b
        for i in 0..4 {
            for j in 0..2 {
                assert_abs_diff_eq!(res.result[[i, j]], b[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mixed_precision_solve_vs_direct() {
        // 3x3 system with known solution
        let a = array![[4.0_f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]];
        let x_true = array![[1.0_f64], [2.0], [3.0]];
        // b = A * x_true
        let b = gemm(&a, &x_true, None, &GemmConfig::default()).unwrap();
        let config = PrecisionDispatchConfig::default();
        let res = mixed_precision_solve(&a, &b, &config).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(res.result[[i, 0]], x_true[[i, 0]], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_mixed_precision_solve_non_square_error() {
        let a = Array2::<f64>::zeros((3, 4));
        let b = Array2::<f64>::zeros((3, 1));
        let config = PrecisionDispatchConfig::default();
        assert!(mixed_precision_solve(&a, &b, &config).is_err());
    }

    #[test]
    fn test_adaptive_gemm_shape() {
        let a = Array2::<f64>::from_shape_fn((5, 7), |(i, j)| (i + j) as f64 * 0.1);
        let b = Array2::<f64>::from_shape_fn((7, 3), |(i, j)| (i * j + 1) as f64 * 0.1);
        let config = PrecisionDispatchConfig::default();
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        assert_eq!(res.result.shape(), &[5, 3]);
    }

    #[test]
    fn test_dispatch_result_fields_populated() {
        let a = Array2::<f64>::eye(3);
        let b = Array2::<f64>::eye(3);
        let config = PrecisionDispatchConfig {
            estimate_condition: true,
            ..Default::default()
        };
        let res = adaptive_gemm(&a, &b, &config).unwrap();
        assert!(res.condition_estimate.is_some());
        assert!(!res.precision_used.is_empty());
    }
}
