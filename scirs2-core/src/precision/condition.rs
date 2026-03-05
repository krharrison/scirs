//! Condition number analysis and numerical stability diagnostics.
//!
//! The condition number of a problem measures how much the output changes in
//! response to a small change in the input.  A large condition number means
//! the problem is ill-conditioned: small errors in input (e.g. rounding) can
//! produce large errors in output.
//!
//! This module operates on dense matrices stored in row-major order (`Vec<f64>`
//! with shape `(rows, cols)`).  All indices are 0-based.
//!
//! # References
//!
//! - Golub & Van Loan, "Matrix Computations", 4th ed., 2013.
//! - Higham, N.J., "Accuracy and Stability of Numerical Algorithms", 2nd ed., 2002.

use crate::error::{CoreError, ErrorContext};

// ---------------------------------------------------------------------------
// Condition number of a matrix (via SVD power iteration estimate)
// ---------------------------------------------------------------------------

/// Estimate the 2-norm condition number of a square matrix `A`.
///
/// The condition number κ₂(A) = σ_max(A) / σ_min(A) where σ_max and σ_min
/// are the largest and smallest singular values of A.
///
/// This implementation uses a simplified power-iteration approach to estimate
/// σ_max and σ_min without a full SVD decomposition, making it suitable for
/// moderate-sized dense matrices.
///
/// Returns `Err` if `a` does not have `n*n` elements or `n == 0`.
/// Returns `f64::INFINITY` if the matrix is (numerically) singular.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::condition_number_linear_system;
/// // Identity matrix has condition number 1.
/// let identity = vec![1.0_f64, 0.0, 0.0, 1.0];
/// let kappa = condition_number_linear_system(&identity, 2).expect("should succeed");
/// assert!((kappa - 1.0).abs() < 1e-10);
/// ```
pub fn condition_number_linear_system(a: &[f64], n: usize) -> Result<f64, CoreError> {
    validate_square(a, n)?;

    // Use Frobenius norm of A and A^{-1} as a quick estimate.
    // For a 1x1 matrix, the condition number is trivial.
    if n == 1 {
        let v = a[0];
        if v == 0.0 {
            return Ok(f64::INFINITY);
        }
        return Ok(1.0); // |a| / |a^{-1}| = |a| * (1/|a|) = 1
    }

    // Estimate σ_max via power iteration on A^T A.
    let sigma_max = power_iteration_sigma_max(a, n, 200)?;

    // Estimate σ_min via inverse power iteration on A^T A (if A is invertible).
    let sigma_min = inverse_power_iteration_sigma_min(a, n, 200)?;

    if sigma_min < f64::EPSILON * sigma_max {
        return Ok(f64::INFINITY);
    }
    Ok(sigma_max / sigma_min)
}

// ---------------------------------------------------------------------------
// Forward error bound
// ---------------------------------------------------------------------------

/// Compute a forward error bound for the solution of Ax = b.
///
/// Given the condition number κ₂(A) and the relative error δ in b,
/// the forward error bound on x is:
///
///   ‖δx‖ / ‖x‖  ≤  κ(A) · ‖δb‖ / ‖b‖
///
/// This function returns that upper bound given:
/// - `a`: the n×n matrix A
/// - `n`: dimension
/// - `rel_error_b`: relative perturbation in b (‖δb‖ / ‖b‖)
///
/// Returns `Err` if inputs are invalid.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::forward_error_bound;
/// let a = vec![2.0_f64, 1.0, 1.0, 3.0];
/// let bound = forward_error_bound(&a, 2, 1e-10).expect("should succeed");
/// assert!(bound.is_finite());
/// ```
pub fn forward_error_bound(a: &[f64], n: usize, rel_error_b: f64) -> Result<f64, CoreError> {
    if rel_error_b < 0.0 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "forward_error_bound: rel_error_b must be non-negative".to_string(),
        )));
    }
    let kappa = condition_number_linear_system(a, n)?;
    Ok(kappa * rel_error_b)
}

// ---------------------------------------------------------------------------
// Backward error bound (residual-based)
// ---------------------------------------------------------------------------

/// Compute the backward error of an approximate solution `x_approx` of `Ax = b`.
///
/// The (normwise) backward error is:
///
///   η = ‖b - A·x_approx‖ / (‖A‖ · ‖x_approx‖ + ‖b‖)
///
/// which measures how small a perturbation to (A, b) makes `x_approx` an
/// exact solution.
///
/// Returns `Err` if dimensions are inconsistent.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::backward_error_bound;
/// // Exact solution: A=[1,0;0,1], b=[1,2], x=[1,2].
/// let a = vec![1.0_f64, 0.0, 0.0, 1.0];
/// let b = vec![1.0_f64, 2.0];
/// let x = vec![1.0_f64, 2.0];
/// let eta = backward_error_bound(&a, 2, &b, &x).expect("should succeed");
/// assert!(eta < 1e-14);
/// ```
pub fn backward_error_bound(
    a: &[f64],
    n: usize,
    b: &[f64],
    x_approx: &[f64],
) -> Result<f64, CoreError> {
    validate_square(a, n)?;
    if b.len() != n {
        return Err(CoreError::InvalidInput(ErrorContext::new(format!(
            "backward_error_bound: b has length {}, expected {n}",
            b.len()
        ))));
    }
    if x_approx.len() != n {
        return Err(CoreError::InvalidInput(ErrorContext::new(format!(
            "backward_error_bound: x_approx has length {}, expected {n}",
            x_approx.len()
        ))));
    }

    // Compute residual r = b - A * x_approx.
    let r = compute_residual(a, n, b, x_approx);

    let norm_r = l2_norm(&r);
    let norm_a = frobenius_norm(a);
    let norm_x = l2_norm(x_approx);
    let norm_b = l2_norm(b);

    let denominator = norm_a * norm_x + norm_b;
    if denominator < f64::EPSILON {
        if norm_r < f64::EPSILON {
            return Ok(0.0);
        }
        return Ok(f64::INFINITY);
    }
    Ok(norm_r / denominator)
}

// ---------------------------------------------------------------------------
// Mixed forward-backward error
// ---------------------------------------------------------------------------

/// Compute the mixed forward-backward error.
///
/// This combines the forward error (how large is ‖δx‖/‖x‖) and the backward
/// error (how small a perturbation explains the residual).  The mixed error is:
///
///   ε_mixed = backward_error × condition_number
///
/// which is an upper bound on the forward error that accounts for both
/// the quality of the approximate solution and the conditioning of the problem.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::mixed_error_bound;
/// let a = vec![1.0_f64, 0.0, 0.0, 1.0];
/// let b = vec![1.0_f64, 1.0];
/// let x = vec![1.0_f64, 1.0];
/// let me = mixed_error_bound(&a, 2, &b, &x).expect("should succeed");
/// assert!(me < 1e-12);
/// ```
pub fn mixed_error_bound(
    a: &[f64],
    n: usize,
    b: &[f64],
    x_approx: &[f64],
) -> Result<f64, CoreError> {
    let eta = backward_error_bound(a, n, b, x_approx)?;
    let kappa = condition_number_linear_system(a, n)?;
    Ok(eta * kappa)
}

// ---------------------------------------------------------------------------
// Significant decimal digits
// ---------------------------------------------------------------------------

/// Estimate the number of significant decimal digits in a computed result.
///
/// Given the relative error `eps` of the computation (e.g. from the forward
/// error bound), the number of significant digits is approximately:
///
///   d = -log10(eps)
///
/// Returns 0 if `eps >= 1.0` (total loss of significance), and caps at 15
/// (the maximum for f64).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::significant_digits;
/// let d = significant_digits(1e-10);
/// assert_eq!(d, 10);
/// ```
pub fn significant_digits(relative_error: f64) -> u32 {
    if relative_error <= 0.0 {
        return 15; // effectively exact
    }
    if relative_error >= 1.0 {
        return 0;
    }
    let d = (-relative_error.log10()).floor() as i32;
    d.clamp(0, 15) as u32
}

// ---------------------------------------------------------------------------
// Catastrophic cancellation check
// ---------------------------------------------------------------------------

/// Detect catastrophic cancellation when computing `a - b`.
///
/// Catastrophic cancellation occurs when `a` and `b` are nearly equal in
/// magnitude but opposite in sign (or very close positive values), causing
/// the result to have many fewer significant bits than the inputs.
///
/// Returns the estimated number of bits lost due to cancellation.  A value
/// of 0 means no significant cancellation; a value near 52 means total loss
/// of significance (the result has essentially no correct bits).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::condition::catastrophic_cancellation_check;
/// // 1.000000001 - 1.0: the result has ~9 decimal digits ≈ 30 bits,
/// // so ~22 bits are lost from the original 52.
/// let bits_lost = catastrophic_cancellation_check(1.000_000_001_f64, 1.0_f64);
/// assert!(bits_lost > 0);
/// ```
pub fn catastrophic_cancellation_check(a: f64, b: f64) -> u32 {
    if a.is_nan() || b.is_nan() || a.is_infinite() || b.is_infinite() {
        return 0;
    }
    let result = a - b;
    if result == 0.0 {
        // Total cancellation if inputs were non-zero.
        if a != 0.0 {
            return 52;
        }
        return 0;
    }
    if a == 0.0 && b == 0.0 {
        return 0;
    }
    // Relative magnitude of the result compared to the operands.
    let max_operand = a.abs().max(b.abs());
    if max_operand == 0.0 {
        return 0;
    }
    let ratio = result.abs() / max_operand;
    if ratio >= 1.0 {
        return 0;
    }
    // Bits lost ≈ -log2(ratio)
    let bits_lost = (-ratio.log2()).floor() as i32;
    bits_lost.clamp(0, 52) as u32
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn validate_square(a: &[f64], n: usize) -> Result<(), CoreError> {
    if n == 0 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "condition: matrix dimension is 0".to_string(),
        )));
    }
    if a.len() != n * n {
        return Err(CoreError::InvalidInput(ErrorContext::new(format!(
            "condition: expected {}×{} = {} elements, got {}",
            n,
            n,
            n * n,
            a.len()
        ))));
    }
    Ok(())
}

/// Frobenius norm of an n×n matrix.
fn frobenius_norm(a: &[f64]) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// L2 (Euclidean) norm of a vector.
fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix-vector product y = A * x (row-major).
fn matvec(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            y[i] += a[i * n + j] * x[j];
        }
    }
    y
}

/// Transpose matrix-vector product y = A^T * x.
fn matvec_t(a: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        for j in 0..n {
            y[j] += a[i * n + j] * x[i];
        }
    }
    y
}

/// Compute residual r = b - A * x.
fn compute_residual(a: &[f64], n: usize, b: &[f64], x: &[f64]) -> Vec<f64> {
    let ax = matvec(a, n, x);
    b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect()
}

/// Normalize a vector in-place.  Returns the norm, or 0 if the vector is zero.
fn normalize_inplace(v: &mut Vec<f64>) -> f64 {
    let norm = l2_norm(v);
    if norm > 0.0 {
        for vi in v.iter_mut() {
            *vi /= norm;
        }
    }
    norm
}

/// Estimate σ_max = √(λ_max(A^T A)) via power iteration.
fn power_iteration_sigma_max(
    a: &[f64],
    n: usize,
    max_iter: usize,
) -> Result<f64, CoreError> {
    // Initialise with a random-ish vector (use deterministic seed: 1/√n pattern).
    let mut v: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).recip()).collect();
    normalize_inplace(&mut v);

    let mut sigma = 0.0_f64;
    for _ in 0..max_iter {
        // w = A^T A v
        let av = matvec(a, n, &v);
        let mut w = matvec_t(a, n, &av);
        let norm = normalize_inplace(&mut w);
        if (norm - sigma).abs() < f64::EPSILON * norm.max(1.0) {
            sigma = norm;
            break;
        }
        sigma = norm;
        v = w;
    }
    Ok(sigma.sqrt())
}

/// Estimate σ_min via inverse power iteration (if A is not singular).
///
/// We iterate on (A^T A)^{-1} by solving the system via Gaussian elimination
/// with partial pivoting at each step.  This is expensive (O(n³) per iteration)
/// but exact within floating-point arithmetic.
fn inverse_power_iteration_sigma_min(
    a: &[f64],
    n: usize,
    max_iter: usize,
) -> Result<f64, CoreError> {
    // Form B = A^T A explicitly.
    let b = ata(a, n);

    // Check if B is nearly singular to avoid infinite loop.
    let frob_b = frobenius_norm(&b);
    if frob_b < f64::EPSILON {
        return Ok(0.0);
    }

    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

    let mut mu = 0.0_f64;
    for _ in 0..max_iter {
        // Solve B w = v via Gaussian elimination.
        let w = match gaussian_elim(&b, n, &v) {
            Some(w) => w,
            None => return Ok(0.0), // singular
        };
        let mut w = w;
        let norm = normalize_inplace(&mut w);
        if norm < f64::EPSILON {
            return Ok(0.0);
        }
        let lambda_inv = norm;
        let lambda = lambda_inv.recip();
        if (lambda - mu).abs() < f64::EPSILON * mu.max(1.0) {
            mu = lambda;
            break;
        }
        mu = lambda;
        v = w;
    }
    // σ_min = √(λ_min(A^T A)) = √mu
    Ok(mu.max(0.0).sqrt())
}

/// Compute A^T A.
fn ata(a: &[f64], n: usize) -> Vec<f64> {
    let mut b = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f64;
            for k in 0..n {
                s += a[k * n + i] * a[k * n + j];
            }
            b[i * n + j] = s;
        }
    }
    b
}

/// Solve the n×n system Ax = b via Gaussian elimination with partial pivoting.
/// Returns None if the matrix is (numerically) singular.
fn gaussian_elim(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    // Build augmented matrix [A | b].
    let mut mat: Vec<f64> = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            mat[i * (n + 1) + j] = a[i * n + j];
        }
        mat[i * (n + 1) + n] = b[i];
    }

    let m = n + 1; // row stride

    for col in 0..n {
        // Find pivot.
        let mut max_val = mat[col * m + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = mat[row * m + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < f64::EPSILON {
            return None; // singular
        }
        // Swap rows.
        if max_row != col {
            for j in 0..m {
                mat.swap(col * m + j, max_row * m + j);
            }
        }
        let pivot = mat[col * m + col];
        for row in (col + 1)..n {
            let factor = mat[row * m + col] / pivot;
            for j in col..m {
                let v = mat[col * m + j] * factor;
                mat[row * m + j] -= v;
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut s = mat[i * m + n];
        for j in (i + 1)..n {
            s -= mat[i * m + j] * x[j];
        }
        let diag = mat[i * m + i];
        if diag.abs() < f64::EPSILON {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_identity() {
        let id = vec![1.0_f64, 0.0, 0.0, 1.0];
        let kappa = condition_number_linear_system(&id, 2).expect("valid");
        assert!((kappa - 1.0).abs() < 1e-8, "kappa={kappa}");
    }

    #[test]
    fn condition_diagonal() {
        // Diagonal matrix [2, 0; 0, 8] has σ_max=8, σ_min=2, κ=4.
        let a = vec![2.0_f64, 0.0, 0.0, 8.0];
        let kappa = condition_number_linear_system(&a, 2).expect("valid");
        assert!((kappa - 4.0).abs() < 1e-6, "kappa={kappa}");
    }

    #[test]
    fn condition_singular_is_inf() {
        let a = vec![1.0_f64, 1.0, 1.0, 1.0]; // rank 1
        let kappa = condition_number_linear_system(&a, 2).expect("valid");
        assert!(kappa.is_infinite() || kappa > 1e14, "kappa={kappa}");
    }

    #[test]
    fn condition_wrong_size() {
        assert!(condition_number_linear_system(&[1.0, 2.0], 2).is_err());
    }

    #[test]
    fn backward_error_exact_solution() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = vec![3.0_f64, 4.0];
        let x = vec![3.0_f64, 4.0];
        let eta = backward_error_bound(&a, 2, &b, &x).expect("valid");
        assert!(eta < 1e-14, "eta={eta}");
    }

    #[test]
    fn significant_digits_exact() {
        assert_eq!(significant_digits(0.0), 15);
    }

    #[test]
    fn significant_digits_total_loss() {
        assert_eq!(significant_digits(1.5), 0);
    }

    #[test]
    fn significant_digits_middle() {
        assert_eq!(significant_digits(1e-6), 6);
    }

    #[test]
    fn cancellation_check_identical() {
        // a - a: total cancellation if a != 0.
        let bits = catastrophic_cancellation_check(1.0, 1.0);
        assert_eq!(bits, 52);
    }

    #[test]
    fn cancellation_check_none() {
        // 2.0 - 1.0 = 1.0: no cancellation (result same magnitude as inputs).
        let bits = catastrophic_cancellation_check(2.0, 1.0);
        assert_eq!(bits, 0);
    }

    #[test]
    fn forward_error_bound_positive() {
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let fwd = forward_error_bound(&a, 2, 1e-10).expect("valid");
        assert!(fwd >= 0.0);
        assert!(fwd.is_finite() || fwd == f64::INFINITY);
    }
}
