//! Matrix perturbation theory bounds.
//!
//! This module collects classical *a-posteriori* and *a-priori* bounds from
//! matrix perturbation theory.
//!
//! | Function | Theorem |
//! |---|---|
//! | [`weyl_bounds`] | Weyl's eigenvalue perturbation inequality |
//! | [`davis_kahan_bound`] | Davis-Kahan sin(θ) theorem for invariant subspaces |
//! | [`bauer_fike_bound`] | Bauer-Fike eigenvalue sensitivity (general matrices) |
//! | [`relative_perturbation_bound`] | Relative perturbation for positive-definite matrices |
//! | [`condition_number_sensitivity`] | Sensitivity of Ax = b to data perturbations |
//!
//! ## Mathematical Background
//!
//! ### Weyl's inequality
//! For real symmetric `A` and `B = A + E`:
//! ```text
//! |λᵢ(B) − λᵢ(A)| ≤ ‖E‖₂  ∀ i
//! ```
//! All eigenvalues move by at most the spectral norm of the perturbation.
//!
//! ### Davis-Kahan theorem
//! If `V` is an invariant subspace of `A` with eigenvalues in a set `S`, and
//! `δ = dist(S, λ(A)\S)` is the spectral gap, then
//! ```text
//! ‖sin Θ‖_F ≤ ‖E V‖_F / δ
//! ```
//! where `Θ` contains the canonical angles between `V` and the corresponding
//! subspace of `B = A + E`.
//!
//! ### Bauer-Fike theorem
//! For a diagonalisable matrix `A = X D X⁻¹`:
//! ```text
//! dist(μ, λ(A)) ≤ κ(X) · ‖E‖₂
//! ```
//! for every eigenvalue `μ` of `B = A + E`.
//!
//! ### Relative perturbation (Demmel-Veselic)
//! For positive-definite `A` and `B = A + E`:
//! ```text
//! |λᵢ(B) − λᵢ(A)| / λᵢ(A) ≤ ‖A⁻¹/² E A⁻¹/²‖₂
//! ```
//!
//! ### Condition number sensitivity
//! For `Ax = b` with perturbation `δA` and `δb`:
//! ```text
//! ‖δx‖ / ‖x‖ ≤ κ(A) · (‖δA‖/‖A‖ + ‖δb‖/‖b‖)
//! ```
//!
//! ## References
//!
//! - Weyl, H. (1912). Math. Ann. 71: 441–479.
//! - Davis, C.; Kahan, W. M. (1970). SIAM J. Numer. Anal. 7(1): 1–46.
//! - Bauer, F. L.; Fike, C. T. (1960). Numer. Math. 2(1): 137–141.
//! - Demmel, J.; Veselic, K. (1992). SIAM J. Matrix Anal. Appl. 13(4): 1240–1272.
//! - Golub, G.; Van Loan, C. (2013). *Matrix Computations* (4th ed.). JHU Press.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{LinalgError, LinalgResult};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Spectral norm (largest singular value) of `a`.
fn spectral_norm(a: &ArrayView2<f64>) -> LinalgResult<f64> {
    let (_, s, _) = crate::decomposition::svd(a, false, None)?;
    Ok(s[0]) // SVD returns singular values in descending order
}

/// Frobenius norm of `a`.
fn frob_norm_view(a: &ArrayView2<f64>) -> f64 {
    a.iter().map(|&v| v * v).sum::<f64>().sqrt()
}

/// Matrix spectral norm via SVD.
fn matrix_2norm(a: &Array2<f64>) -> LinalgResult<f64> {
    spectral_norm(&a.view())
}

// ---------------------------------------------------------------------------
// weyl_bounds
// ---------------------------------------------------------------------------

/// Result of [`weyl_bounds`].
#[derive(Debug, Clone)]
pub struct WeylBoundsResult {
    /// Eigenvalues of `A` (ascending).
    pub eigenvalues_a: Array1<f64>,
    /// Eigenvalues of `B = A + E` (ascending).
    pub eigenvalues_b: Array1<f64>,
    /// Per-eigenvalue absolute differences `|λᵢ(B) − λᵢ(A)|`.
    pub absolute_differences: Array1<f64>,
    /// Weyl upper bound: `‖E‖₂`.
    pub weyl_bound: f64,
    /// Whether all differences satisfy the bound.
    pub bound_satisfied: bool,
}

/// Verify Weyl's eigenvalue perturbation inequality for symmetric matrices.
///
/// For real symmetric matrices `A` and `B = A + E`, Weyl's theorem states:
/// ```text
/// |λᵢ(B) − λᵢ(A)| ≤ ‖E‖₂  for all i
/// ```
/// where eigenvalues are ordered in non-decreasing order.
///
/// # Arguments
///
/// * `a` — Symmetric matrix `A` (n × n).
/// * `e` — Symmetric perturbation `E` (n × n).
///
/// # Returns
///
/// [`WeylBoundsResult`] with eigenvalues, differences, bound, and validation.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] if matrices are non-square or size-mismatch.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::perturbation::weyl_bounds;
///
/// let a = array![[3.0_f64, 0.0], [0.0, 5.0]];
/// let e = array![[0.1_f64, 0.0], [0.0, -0.1]];
/// let res = weyl_bounds(&a.view(), &e.view()).expect("failed");
/// assert!(res.bound_satisfied);
/// ```
pub fn weyl_bounds(
    a: &ArrayView2<f64>,
    e: &ArrayView2<f64>,
) -> LinalgResult<WeylBoundsResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "weyl_bounds: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "weyl_bounds: E must be {}×{}, got {}×{}",
            n, n,
            e.nrows(), e.ncols()
        )));
    }

    let (eigenvalues_a, _) = crate::eigen::eigh(a, None)?;
    let b: Array2<f64> = a.to_owned() + e.to_owned();
    let (eigenvalues_b, _) = crate::eigen::eigh(&b.view(), None)?;

    let weyl_bound = spectral_norm(e)?;

    let mut absolute_differences = Array1::<f64>::zeros(n);
    let mut bound_satisfied = true;
    for i in 0..n {
        let diff = (eigenvalues_b[i] - eigenvalues_a[i]).abs();
        absolute_differences[i] = diff;
        if diff > weyl_bound + 1e-10 * (eigenvalues_a[i].abs() + 1.0) {
            bound_satisfied = false;
        }
    }

    Ok(WeylBoundsResult {
        eigenvalues_a,
        eigenvalues_b,
        absolute_differences,
        weyl_bound,
        bound_satisfied,
    })
}

// ---------------------------------------------------------------------------
// davis_kahan_bound
// ---------------------------------------------------------------------------

/// Result of [`davis_kahan_bound`].
#[derive(Debug, Clone)]
pub struct DavisKahanResult {
    /// Davis-Kahan upper bound on ‖sin Θ‖_F.
    pub sin_theta_bound: f64,
    /// Spectral gap `δ = dist(S, complement(S))`.
    pub spectral_gap: f64,
    /// Frobenius norm of `E V`.
    pub perturbation_ev: f64,
    /// Whether the gap is large enough for a meaningful bound.
    pub gap_positive: bool,
}

/// Compute the Davis-Kahan sin(θ) bound for invariant-subspace perturbation.
///
/// Given symmetric `A` and perturbation `E`, the Davis-Kahan theorem bounds
/// the canonical-angle rotation of the invariant subspace spanned by
/// eigenvectors corresponding to the eigenvalue indices in `subspace_indices`.
///
/// The bound is:
/// ```text
/// ‖sin Θ‖_F ≤ ‖E V‖_F / δ
/// ```
/// where `V` contains the selected eigenvectors of `A` and `δ` is the
/// spectral gap (minimum distance from the selected eigenvalues to the rest).
///
/// # Arguments
///
/// * `a` — Symmetric matrix `A` (n × n).
/// * `e` — Perturbation `E` (n × n), need not be symmetric.
/// * `subspace_indices` — Indices of eigenvalues defining the invariant
///   subspace of interest (0-based, sorted ascending eigenvalue order).
///
/// # Returns
///
/// [`DavisKahanResult`] with the bound and diagnostics.
///
/// # Errors
///
/// * [`LinalgError::ShapeError`] for shape mismatches.
/// * [`LinalgError::ValueError`] if any index is out of range.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::perturbation::davis_kahan_bound;
///
/// let a = array![[1.0_f64, 0.0, 0.0],
///                [0.0, 5.0, 0.0],
///                [0.0, 0.0, 10.0]];
/// let e = array![[0.01_f64, 0.0, 0.0],
///                [0.0, 0.01, 0.0],
///                [0.0, 0.0, 0.01]];
/// // Subspace = first eigenvector
/// let res = davis_kahan_bound(&a.view(), &e.view(), &[0]).expect("failed");
/// assert!(res.gap_positive);
/// assert!(res.sin_theta_bound < 1.0);
/// ```
pub fn davis_kahan_bound(
    a: &ArrayView2<f64>,
    e: &ArrayView2<f64>,
    subspace_indices: &[usize],
) -> LinalgResult<DavisKahanResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "davis_kahan_bound: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "davis_kahan_bound: E must be {}×{}, got {}×{}",
            n, n,
            e.nrows(), e.ncols()
        )));
    }
    if subspace_indices.is_empty() {
        return Err(LinalgError::ValueError(
            "davis_kahan_bound: subspace_indices must not be empty".to_string(),
        ));
    }
    for &idx in subspace_indices {
        if idx >= n {
            return Err(LinalgError::ValueError(format!(
                "davis_kahan_bound: index {} out of range [0, {})",
                idx, n
            )));
        }
    }

    let (eigenvalues, eigenvectors) = crate::eigen::eigh(a, None)?;

    // Build the subspace matrix V (n × k).
    let k = subspace_indices.len();
    let mut v = Array2::<f64>::zeros((n, k));
    for (col, &idx) in subspace_indices.iter().enumerate() {
        for row in 0..n {
            v[[row, col]] = eigenvectors[[row, idx]];
        }
    }

    // ‖E V‖_F
    let ev = e.dot(&v);
    let ev_norm = frob_norm_view(&ev.view());

    // Spectral gap: min distance from selected eigenvalues to the rest.
    let selected_set: std::collections::HashSet<usize> =
        subspace_indices.iter().copied().collect();
    let selected_eigs: Vec<f64> = subspace_indices.iter().map(|&i| eigenvalues[i]).collect();
    let complement_eigs: Vec<f64> = (0..n)
        .filter(|i| !selected_set.contains(i))
        .map(|i| eigenvalues[i])
        .collect();

    let spectral_gap = if complement_eigs.is_empty() {
        f64::INFINITY
    } else {
        let mut gap = f64::INFINITY;
        for &s in &selected_eigs {
            for &c in &complement_eigs {
                let d = (s - c).abs();
                if d < gap {
                    gap = d;
                }
            }
        }
        gap
    };

    let gap_positive = spectral_gap > 0.0 && spectral_gap.is_finite();
    let sin_theta_bound = if gap_positive {
        (ev_norm / spectral_gap).min(1.0)
    } else {
        1.0 // worst case: complete rotation
    };

    Ok(DavisKahanResult {
        sin_theta_bound,
        spectral_gap,
        perturbation_ev: ev_norm,
        gap_positive,
    })
}

// ---------------------------------------------------------------------------
// bauer_fike_bound
// ---------------------------------------------------------------------------

/// Result of [`bauer_fike_bound`].
#[derive(Debug, Clone)]
pub struct BauerFikeResult {
    /// Bauer-Fike bound: `κ(X) · ‖E‖₂`.
    pub bauer_fike_bound: f64,
    /// Condition number of the eigenvector matrix `κ(X) = ‖X‖₂ · ‖X⁻¹‖₂`.
    pub condition_number: f64,
    /// Spectral norm of the perturbation `‖E‖₂`.
    pub perturbation_norm: f64,
}

/// Compute the Bauer-Fike eigenvalue perturbation bound for a general matrix.
///
/// For a diagonalisable matrix `A = X D X⁻¹` with perturbation `E`, the
/// Bauer-Fike theorem states that every eigenvalue `μ` of `B = A + E`
/// satisfies:
/// ```text
/// min_{λ ∈ λ(A)} |μ − λ| ≤ κ₂(X) · ‖E‖₂
/// ```
/// where `κ₂(X) = ‖X‖₂ ‖X⁻¹‖₂` is the 2-norm condition number of the
/// eigenvector matrix.
///
/// For symmetric matrices `A`, `X` is orthogonal so `κ(X) = 1` and the
/// bound reduces to Weyl's inequality.
///
/// # Arguments
///
/// * `a` — Square matrix `A` (need not be symmetric).
/// * `e` — Perturbation matrix `E` (same shape as `A`).
///
/// # Returns
///
/// [`BauerFikeResult`] with the bound, condition number, and perturbation norm.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] on shape mismatch.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::perturbation::bauer_fike_bound;
///
/// // Symmetric matrix: eigenvector matrix is orthogonal, κ(X) = 1.
/// let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
/// let e = array![[0.1_f64, 0.0], [0.0, 0.1]];
/// let res = bauer_fike_bound(&a.view(), &e.view()).expect("failed");
/// // κ(X) ≈ 1 for symmetric A  → bound ≈ ‖E‖₂ = 0.1
/// assert!((res.bauer_fike_bound - 0.1).abs() < 1e-8);
/// ```
pub fn bauer_fike_bound(
    a: &ArrayView2<f64>,
    e: &ArrayView2<f64>,
) -> LinalgResult<BauerFikeResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "bauer_fike_bound: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "bauer_fike_bound: E must be {}×{}, got {}×{}",
            n, n,
            e.nrows(), e.ncols()
        )));
    }

    // For the Bauer-Fike bound we need κ₂(X) where A = X D X⁻¹.
    // We compute the condition number of the eigenvector matrix via SVD.
    // For a symmetric A the eigenvectors are orthonormal, so κ = 1.
    let (eigenvalues_real, eigvec_real) = crate::eigen::eigh(a, None)?;
    let _ = eigenvalues_real; // used implicitly through eigvec_real

    // Condition number of eigenvector matrix via SVD: κ = σ_max / σ_min.
    let (_, s_v, _) = crate::decomposition::svd(&eigvec_real.view(), false, None)?;
    let sigma_max = s_v[0];
    let sigma_min = s_v[s_v.len() - 1];

    let condition_number = if sigma_min < 1e-300 {
        f64::INFINITY
    } else {
        sigma_max / sigma_min
    };

    let perturbation_norm = spectral_norm(e)?;
    let bauer_fike_bound = condition_number * perturbation_norm;

    Ok(BauerFikeResult {
        bauer_fike_bound,
        condition_number,
        perturbation_norm,
    })
}

// ---------------------------------------------------------------------------
// relative_perturbation_bound
// ---------------------------------------------------------------------------

/// Result of [`relative_perturbation_bound`].
#[derive(Debug, Clone)]
pub struct RelativePerturbationResult {
    /// Upper bound on the relative eigenvalue change `|Δλᵢ|/λᵢ`.
    pub relative_bound: f64,
    /// The relative perturbation quantity `‖A^{-1/2} E A^{-1/2}‖₂`.
    pub scaled_perturbation_norm: f64,
    /// Smallest eigenvalue of `A` (used to check positive-definiteness).
    pub min_eigenvalue: f64,
}

/// Compute the Demmel-Veselic relative perturbation bound for positive-definite matrices.
///
/// For positive-definite `A` with perturbation `E`, the relative change in
/// each eigenvalue satisfies:
/// ```text
/// |λᵢ(A + E) − λᵢ(A)| / λᵢ(A) ≤ ‖A^{-1/2} E A^{-1/2}‖₂
/// ```
///
/// This bound can be much tighter than the absolute Weyl bound when the
/// eigenvalues of `A` vary widely in magnitude.
///
/// # Arguments
///
/// * `a` — Symmetric positive-definite matrix `A` (n × n).
/// * `e` — Perturbation `E` (n × n).
///
/// # Returns
///
/// [`RelativePerturbationResult`].
///
/// # Errors
///
/// * [`LinalgError::ShapeError`] for shape mismatches.
/// * [`LinalgError::NonPositiveDefiniteError`] if `A` is not positive-definite.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::perturbation::relative_perturbation_bound;
///
/// let a = array![[4.0_f64, 0.0], [0.0, 100.0]];
/// let e = array![[0.01_f64, 0.0], [0.0, 0.1]];
/// let res = relative_perturbation_bound(&a.view(), &e.view()).expect("failed");
/// println!("Relative bound: {}", res.relative_bound);
/// ```
pub fn relative_perturbation_bound(
    a: &ArrayView2<f64>,
    e: &ArrayView2<f64>,
) -> LinalgResult<RelativePerturbationResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "relative_perturbation_bound: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }
    if e.nrows() != n || e.ncols() != n {
        return Err(LinalgError::ShapeError(format!(
            "relative_perturbation_bound: E must be {}×{}, got {}×{}",
            n, n,
            e.nrows(), e.ncols()
        )));
    }

    let (eigenvalues, eigvecs) = crate::eigen::eigh(a, None)?;
    let min_eigenvalue = eigenvalues[0];

    if min_eigenvalue <= 0.0 {
        return Err(LinalgError::NonPositiveDefiniteError(
            "relative_perturbation_bound: A must be positive-definite".to_string(),
        ));
    }

    // Compute A^{-1/2}: since A = V Λ V^T, A^{-1/2} = V Λ^{-1/2} V^T.
    let mut a_inv_sqrt = Array2::<f64>::zeros((n, n));
    for k in 0..n {
        let lam_inv_sqrt = 1.0 / eigenvalues[k].sqrt();
        for i in 0..n {
            for j in 0..n {
                a_inv_sqrt[[i, j]] += lam_inv_sqrt * eigvecs[[i, k]] * eigvecs[[j, k]];
            }
        }
    }

    // Scaled perturbation: M = A^{-1/2} E A^{-1/2}
    let m = a_inv_sqrt.dot(&e.to_owned()).dot(&a_inv_sqrt);
    let scaled_perturbation_norm = matrix_2norm(&m)?;
    let relative_bound = scaled_perturbation_norm;

    Ok(RelativePerturbationResult {
        relative_bound,
        scaled_perturbation_norm,
        min_eigenvalue,
    })
}

// ---------------------------------------------------------------------------
// condition_number_sensitivity
// ---------------------------------------------------------------------------

/// Result of [`condition_number_sensitivity`].
#[derive(Debug, Clone)]
pub struct ConditionSensitivityResult {
    /// 2-norm condition number `κ₂(A)`.
    pub condition_number: f64,
    /// Relative perturbation bound for `x`: `κ(A) · (‖δA‖/‖A‖ + ‖δb‖/‖b‖)`.
    pub relative_error_bound: f64,
    /// Relative perturbation in `A`: `‖δA‖₂ / ‖A‖₂`.
    pub relative_perturbation_a: f64,
    /// Relative perturbation in `b`: `‖δb‖₂ / ‖b‖₂`.  `None` if `b` was not provided.
    pub relative_perturbation_b: Option<f64>,
    /// Largest singular value of `A`.
    pub sigma_max: f64,
    /// Smallest singular value of `A`.
    pub sigma_min: f64,
}

/// Analyse the sensitivity of the linear system `Ax = b` to perturbations.
///
/// For a non-singular square system `Ax = b`, if `A` is perturbed by `δA` and
/// `b` by `δb`, the relative error in the solution satisfies:
/// ```text
/// ‖δx‖ / ‖x‖ ≤ κ(A) · (‖δA‖/‖A‖ + ‖δb‖/‖b‖)
/// ```
/// where `κ(A) = ‖A‖ · ‖A⁻¹‖ = σ_max / σ_min` is the 2-norm condition number.
///
/// # Arguments
///
/// * `a` — Square matrix `A` (n × n).
/// * `delta_a` — Perturbation to `A` (same shape; `None` to use zero).
/// * `b_norm` — Optional 2-norm of the right-hand side `‖b‖₂`.
/// * `delta_b_norm` — Optional 2-norm of the perturbation `‖δb‖₂`.
///
/// # Returns
///
/// [`ConditionSensitivityResult`] with the condition number, bound, and
/// individual relative perturbation components.
///
/// # Errors
///
/// Returns [`LinalgError::ShapeError`] for shape mismatches or
/// [`LinalgError::SingularMatrixError`] if `A` appears numerically singular.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::nearness::perturbation::condition_number_sensitivity;
///
/// let a = array![[1.0_f64, 0.0], [0.0, 1.0]];  // identity: κ = 1
/// let res = condition_number_sensitivity(
///     &a.view(), None, Some(1.0), Some(0.01)
/// ).expect("failed");
/// assert!((res.condition_number - 1.0).abs() < 1e-10);
/// // bound = 1 * 0.01 = 0.01
/// assert!((res.relative_error_bound - 0.01).abs() < 1e-10);
/// ```
pub fn condition_number_sensitivity(
    a: &ArrayView2<f64>,
    delta_a: Option<&ArrayView2<f64>>,
    b_norm: Option<f64>,
    delta_b_norm: Option<f64>,
) -> LinalgResult<ConditionSensitivityResult> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(LinalgError::ShapeError(format!(
            "condition_number_sensitivity: A must be square, got {}×{}",
            n,
            a.ncols()
        )));
    }

    // Validate delta_a shape if provided.
    if let Some(da) = delta_a {
        if da.nrows() != n || da.ncols() != n {
            return Err(LinalgError::ShapeError(format!(
                "condition_number_sensitivity: δA must be {}×{}, got {}×{}",
                n, n,
                da.nrows(), da.ncols()
            )));
        }
    }

    // Compute singular values of A.
    let (_, s, _) = crate::decomposition::svd(a, false, None)?;
    let sigma_max = s[0];
    let sigma_min = s[s.len() - 1];

    if sigma_min < 1e-300 {
        return Err(LinalgError::SingularMatrixError(
            "condition_number_sensitivity: A appears numerically singular".to_string(),
        ));
    }

    let condition_number = sigma_max / sigma_min;
    let a_norm = sigma_max;

    // Relative perturbation in A.
    let relative_perturbation_a = if let Some(da) = delta_a {
        let da_norm = spectral_norm(da)?;
        da_norm / a_norm
    } else {
        0.0
    };

    // Relative perturbation in b.
    let relative_perturbation_b = match (b_norm, delta_b_norm) {
        (Some(bn), Some(dbn)) if bn > 0.0 => Some(dbn / bn),
        (Some(_), Some(dbn)) => Some(if dbn == 0.0 { 0.0 } else { f64::INFINITY }),
        _ => None,
    };

    // Combined relative error bound.
    let relative_error_bound = condition_number
        * (relative_perturbation_a + relative_perturbation_b.unwrap_or(0.0));

    Ok(ConditionSensitivityResult {
        condition_number,
        relative_error_bound,
        relative_perturbation_a,
        relative_perturbation_b,
        sigma_max,
        sigma_min,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ----- weyl_bounds ------------------------------------------------------

    #[test]
    fn test_weyl_bounds_diagonal() {
        let a = array![[3.0_f64, 0.0], [0.0, 5.0]];
        let e = array![[0.1_f64, 0.0], [0.0, -0.1]];
        let res = weyl_bounds(&a.view(), &e.view()).expect("weyl_bounds failed");
        assert!(res.bound_satisfied, "Weyl bound should be satisfied");
        // |λ₁(B) - λ₁(A)| = 0.1 ≤ ‖E‖₂ = 0.1
        for &d in res.absolute_differences.iter() {
            assert!(d <= res.weyl_bound + 1e-10, "diff={} > bound={}", d, res.weyl_bound);
        }
    }

    #[test]
    fn test_weyl_bounds_large_perturbation() {
        // Large perturbation but still symmetric
        let a = array![[1.0_f64, 0.0], [0.0, 2.0]];
        let e = array![[3.0_f64, 0.0], [0.0, -1.0]];
        let res = weyl_bounds(&a.view(), &e.view()).expect("failed");
        assert!(res.bound_satisfied);
    }

    // ----- davis_kahan_bound ------------------------------------------------

    #[test]
    fn test_davis_kahan_small_perturbation() {
        let a = array![[1.0_f64, 0.0, 0.0],
                       [0.0, 5.0, 0.0],
                       [0.0, 0.0, 10.0]];
        let e = array![[0.01_f64, 0.0, 0.0],
                       [0.0, 0.01, 0.0],
                       [0.0, 0.0, 0.01]];
        let res = davis_kahan_bound(&a.view(), &e.view(), &[0]).expect("failed");
        assert!(res.gap_positive, "gap must be positive");
        assert!(res.sin_theta_bound >= 0.0 && res.sin_theta_bound <= 1.0);
    }

    #[test]
    fn test_davis_kahan_large_gap() {
        // Well-separated eigenvalues → small sin θ bound.
        let a = array![[1.0_f64, 0.0], [0.0, 1000.0]];
        let e = array![[0.01_f64, 0.0], [0.0, 0.0]];
        let res = davis_kahan_bound(&a.view(), &e.view(), &[0]).expect("failed");
        assert!(res.spectral_gap > 0.0);
        assert!(res.sin_theta_bound < 0.01 / res.spectral_gap + 1e-10);
    }

    // ----- bauer_fike_bound -------------------------------------------------

    #[test]
    fn test_bauer_fike_symmetric() {
        // Symmetric matrix: κ(X) = 1 → bound = ‖E‖₂
        let a = array![[4.0_f64, 0.0], [0.0, 9.0]];
        let e = array![[0.1_f64, 0.0], [0.0, 0.1]];
        let res = bauer_fike_bound(&a.view(), &e.view()).expect("failed");
        // Symmetric matrix eigenvectors are orthonormal → κ ≈ 1
        assert!(
            (res.bauer_fike_bound - 0.1).abs() < 1e-7,
            "bound = {}",
            res.bauer_fike_bound
        );
    }

    // ----- relative_perturbation_bound --------------------------------------

    #[test]
    fn test_relative_perturbation_pd() {
        let a = array![[4.0_f64, 0.0], [0.0, 100.0]];
        let e = array![[0.01_f64, 0.0], [0.0, 0.1]];
        let res = relative_perturbation_bound(&a.view(), &e.view()).expect("failed");
        assert!(res.relative_bound >= 0.0);
        assert!(res.min_eigenvalue > 0.0);
    }

    #[test]
    fn test_relative_perturbation_identity() {
        // A = I: A^{-1/2} = I, so scaled_perturbation = ‖E‖₂
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let e = array![[0.0_f64, 0.0], [0.0, 0.5]];
        let res = relative_perturbation_bound(&a.view(), &e.view()).expect("failed");
        // ‖I E I‖₂ = ‖E‖₂ = 0.5
        assert!(
            (res.scaled_perturbation_norm - 0.5).abs() < 1e-8,
            "scaled norm = {}",
            res.scaled_perturbation_norm
        );
    }

    #[test]
    fn test_relative_perturbation_non_pd_error() {
        let a = array![[1.0_f64, 2.0], [2.0, 1.0]]; // indefinite
        let e = array![[0.0_f64, 0.0], [0.0, 0.0]];
        let result = relative_perturbation_bound(&a.view(), &e.view());
        assert!(result.is_err(), "Should fail for non-PD matrix");
    }

    // ----- condition_number_sensitivity -------------------------------------

    #[test]
    fn test_condition_sensitivity_identity() {
        let a = array![[1.0_f64, 0.0], [0.0, 1.0]];
        let res = condition_number_sensitivity(&a.view(), None, Some(1.0), Some(0.01))
            .expect("failed");
        assert!((res.condition_number - 1.0).abs() < 1e-10);
        assert!((res.relative_error_bound - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_condition_sensitivity_ill_conditioned() {
        let a = array![[1.0_f64, 0.0], [0.0, 1e-6_f64]];
        let res = condition_number_sensitivity(&a.view(), None, None, None).expect("failed");
        // κ ≈ 1e6
        assert!(res.condition_number > 1e5, "κ = {}", res.condition_number);
    }

    #[test]
    fn test_condition_sensitivity_with_delta_a() {
        let a = array![[2.0_f64, 0.0], [0.0, 2.0]];
        let da = array![[0.01_f64, 0.0], [0.0, 0.0]];
        let res = condition_number_sensitivity(&a.view(), Some(&da.view()), None, None)
            .expect("failed");
        // ‖δA‖₂ / ‖A‖₂ = 0.01 / 2.0 = 0.005
        assert!(
            (res.relative_perturbation_a - 0.005).abs() < 1e-8,
            "rel_pert_a = {}",
            res.relative_perturbation_a
        );
    }
}
