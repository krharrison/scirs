//! # Array Validation Utilities
//!
//! Comprehensive validation functions for ndarray arrays, covering finiteness,
//! sign constraints, matrix properties (symmetry, orthogonality, positive-definiteness,
//! stochasticity), shape matching, and diagnostic summaries.
//!
//! All functions return `CoreResult` and avoid panics.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use ::ndarray::{Array2, ArrayBase, ArrayView2, Axis, Dimension};
use num_traits::Float;
use std::fmt::{Debug, Display};

// ---------------------------------------------------------------------------
// Element-wise assertions
// ---------------------------------------------------------------------------

/// Assert that every element of the array is finite (not NaN, not Inf).
pub fn assert_finite<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if !val.is_finite() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains non-finite value {val} at index {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

/// Assert that every element is strictly positive (> 0).
pub fn assert_positive<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if val.partial_cmp(&F::zero()) != Some(std::cmp::Ordering::Greater) {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains non-positive value {val} at index {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

/// Assert that every element is non-negative (>= 0).
pub fn assert_non_negative<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if val < F::zero() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains negative value {val} at index {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Matrix property assertions
// ---------------------------------------------------------------------------

/// Assert that a 2-D matrix is symmetric within a given tolerance.
///
/// Checks |A\[i,j\] - A\[j,i\]| <= tolerance for all i,j.
pub fn assert_symmetric<F>(matrix: &ArrayView2<F>, name: &str, tolerance: F) -> CoreResult<()>
where
    F: Float + Display,
{
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{name} is not square ({} x {}), cannot be symmetric",
                shape[0], shape[1]
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    let n = shape[0];
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (matrix[[i, j]] - matrix[[j, i]]).abs();
            if diff > tolerance {
                return Err(CoreError::ValueError(
                    ErrorContext::new(format!(
                        "{name} is not symmetric: |A[{i},{j}] - A[{j},{i}]| = {diff} > {tolerance}"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }
    Ok(())
}

/// Assert that a 2-D matrix is orthogonal within tolerance.
///
/// Checks that A^T * A is close to the identity matrix.
pub fn assert_orthogonal<F>(matrix: &ArrayView2<F>, name: &str, tolerance: F) -> CoreResult<()>
where
    F: Float + Display + std::ops::AddAssign + Debug,
{
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{name} is not square ({} x {}), cannot check orthogonality",
                shape[0], shape[1]
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    let n = shape[0];

    // Compute A^T * A element-by-element without external BLAS
    for i in 0..n {
        for j in 0..n {
            let mut dot = F::zero();
            for k in 0..n {
                dot += matrix[[k, i]] * matrix[[k, j]];
            }
            let expected = if i == j { F::one() } else { F::zero() };
            let diff = (dot - expected).abs();
            if diff > tolerance {
                return Err(CoreError::ValueError(
                    ErrorContext::new(format!(
                        "{name} is not orthogonal: (A^T A)[{i},{j}] = {dot}, expected {expected} (diff={diff})"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }
    Ok(())
}

/// Assert that a 2-D symmetric matrix is positive definite.
///
/// Uses attempted Cholesky decomposition (pure Rust, no BLAS) to check.
pub fn assert_positive_definite<F>(matrix: &ArrayView2<F>, name: &str) -> CoreResult<()>
where
    F: Float + Display + Debug,
{
    let shape = matrix.shape();
    if shape[0] != shape[1] {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{name} is not square ({} x {}), cannot check positive definiteness",
                shape[0], shape[1]
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    let n = shape[0];
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[[i, j]];
            for k in 0..j {
                sum = sum - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= F::zero() {
                    return Err(CoreError::ValueError(
                        ErrorContext::new(format!(
                            "{name} is not positive definite: Cholesky failed at diagonal element [{i},{i}] with value {sum}"
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                if l[[j, j]].is_zero() {
                    return Err(CoreError::ValueError(
                        ErrorContext::new(format!(
                            "{name} is not positive definite: zero diagonal in Cholesky at [{j},{j}]"
                        ))
                        .with_location(ErrorLocation::new(file!(), line!())),
                    ));
                }
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Ok(())
}

/// Assert that a 2-D matrix is (row-)stochastic: each row sums to 1 within tolerance,
/// and all elements are non-negative.
pub fn assert_stochastic<F>(matrix: &ArrayView2<F>, name: &str, tolerance: F) -> CoreResult<()>
where
    F: Float + Display + std::iter::Sum,
{
    let shape = matrix.shape();
    // Check non-negative
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            if matrix[[i, j]] < F::zero() {
                return Err(CoreError::ValueError(
                    ErrorContext::new(format!(
                        "{name} has negative entry {val} at [{i},{j}]; not stochastic",
                        val = matrix[[i, j]]
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }
    // Check row sums
    for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
        let row_sum: F = row.iter().copied().sum();
        let diff = (row_sum - F::one()).abs();
        if diff > tolerance {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} row {i} sums to {row_sum}, not 1.0 (diff={diff})"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shape assertions
// ---------------------------------------------------------------------------

/// Assert that the array has the exact expected shape.
pub fn assert_shape<S, D>(array: &ArrayBase<S, D>, expected: &[usize], name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data,
    D: Dimension,
{
    let actual = array.shape();
    if actual != expected {
        return Err(CoreError::ShapeError(
            ErrorContext::new(format!(
                "{name} shape mismatch: expected {expected:?}, got {actual:?}"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// ArrayStats & diagnose_array
// ---------------------------------------------------------------------------

/// Summary statistics for an array, useful for diagnostics.
#[derive(Debug, Clone)]
pub struct ArrayStats<F: Float> {
    /// Number of elements.
    pub count: usize,
    /// Minimum value.
    pub min: F,
    /// Maximum value.
    pub max: F,
    /// Arithmetic mean.
    pub mean: F,
    /// Standard deviation (population).
    pub std_dev: F,
    /// Whether any element is NaN.
    pub has_nan: bool,
    /// Whether any element is infinite.
    pub has_inf: bool,
    /// Number of zero elements.
    pub zero_count: usize,
    /// Number of negative elements.
    pub negative_count: usize,
}

impl<F: Float + Display> std::fmt::Display for ArrayStats<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArrayStats(n={}, min={}, max={}, mean={}, std={}, nan={}, inf={}, zeros={}, neg={})",
            self.count,
            self.min,
            self.max,
            self.mean,
            self.std_dev,
            self.has_nan,
            self.has_inf,
            self.zero_count,
            self.negative_count,
        )
    }
}

/// Compute summary statistics for a flat array.
///
/// Handles NaN and Inf gracefully: they are counted but excluded from
/// min/max/mean/std calculations (which use only finite elements).
pub fn compute_array_stats<S, D, F>(array: &ArrayBase<S, D>) -> CoreResult<ArrayStats<F>>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    let count = array.len();
    if count == 0 {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Cannot compute stats on empty array",
        )));
    }

    let mut has_nan = false;
    let mut has_inf = false;
    let mut zero_count: usize = 0;
    let mut negative_count: usize = 0;
    let mut min_val = F::infinity();
    let mut max_val = F::neg_infinity();
    let mut sum = F::zero();
    let mut finite_count: usize = 0;

    for &val in array.iter() {
        if val.is_nan() {
            has_nan = true;
            continue;
        }
        if val.is_infinite() {
            has_inf = true;
            continue;
        }
        if val.is_zero() {
            zero_count += 1;
        }
        if val < F::zero() {
            negative_count += 1;
        }
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
        sum = sum + val;
        finite_count += 1;
    }

    let (mean, std_dev) = if finite_count > 0 {
        let n = num_traits::cast::<usize, F>(finite_count).unwrap_or(F::one());
        let mean = sum / n;
        // Second pass for variance
        let mut var_sum = F::zero();
        for &val in array.iter() {
            if val.is_finite() {
                let diff = val - mean;
                var_sum = var_sum + diff * diff;
            }
        }
        let variance = var_sum / n;
        (mean, variance.sqrt())
    } else {
        (F::nan(), F::nan())
    };

    // If no finite elements, set min/max to NaN
    if finite_count == 0 {
        min_val = F::nan();
        max_val = F::nan();
    }

    Ok(ArrayStats {
        count,
        min: min_val,
        max: max_val,
        mean,
        std_dev,
        has_nan,
        has_inf,
        zero_count,
        negative_count,
    })
}

/// Comprehensive array health check, returning a human-readable diagnostic string.
///
/// Reports shape, stats, and any issues found (NaN, Inf, negative values, etc.).
pub fn diagnose_array<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> String
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    let shape = array.shape();
    let mut report = format!("=== Diagnostics for '{name}' ===\n");
    report.push_str(&format!("  Shape: {shape:?}\n"));
    report.push_str(&format!("  Total elements: {}\n", array.len()));

    match compute_array_stats(array) {
        Ok(stats) => {
            report.push_str(&format!("  Min: {}\n", stats.min));
            report.push_str(&format!("  Max: {}\n", stats.max));
            report.push_str(&format!("  Mean: {}\n", stats.mean));
            report.push_str(&format!("  Std Dev: {}\n", stats.std_dev));
            report.push_str(&format!("  Has NaN: {}\n", stats.has_nan));
            report.push_str(&format!("  Has Inf: {}\n", stats.has_inf));
            report.push_str(&format!("  Zero count: {}\n", stats.zero_count));
            report.push_str(&format!("  Negative count: {}\n", stats.negative_count));

            // Issue summary
            let mut issues = Vec::new();
            if stats.has_nan {
                issues.push("contains NaN values");
            }
            if stats.has_inf {
                issues.push("contains Inf values");
            }
            if stats.count > 0 && stats.zero_count == stats.count {
                issues.push("all elements are zero");
            }

            if issues.is_empty() {
                report.push_str("  Issues: none\n");
            } else {
                report.push_str(&format!("  Issues: {}\n", issues.join(", ")));
            }
        }
        Err(e) => {
            report.push_str(&format!("  Stats error: {e}\n"));
        }
    }
    report
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::{array, Array1, Array2};

    // -- assert_finite --

    #[test]
    fn test_assert_finite_ok() {
        let a = array![1.0, 2.0, 3.0];
        assert!(assert_finite(&a, "a").is_ok());
    }

    #[test]
    fn test_assert_finite_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        assert!(assert_finite(&a, "a").is_err());
    }

    #[test]
    fn test_assert_finite_inf() {
        let a = array![1.0, f64::INFINITY, 3.0];
        assert!(assert_finite(&a, "a").is_err());
    }

    // -- assert_positive --

    #[test]
    fn test_assert_positive_ok() {
        let a = array![0.1, 1.0, 100.0];
        assert!(assert_positive(&a, "a").is_ok());
    }

    #[test]
    fn test_assert_positive_zero() {
        let a = array![0.0, 1.0];
        assert!(assert_positive(&a, "a").is_err());
    }

    #[test]
    fn test_assert_positive_neg() {
        let a = array![1.0, -0.5];
        assert!(assert_positive(&a, "a").is_err());
    }

    // -- assert_non_negative --

    #[test]
    fn test_assert_non_negative_ok() {
        let a = array![0.0, 1.0, 100.0];
        assert!(assert_non_negative(&a, "a").is_ok());
    }

    #[test]
    fn test_assert_non_negative_neg() {
        let a = array![0.0, -0.001];
        assert!(assert_non_negative(&a, "a").is_err());
    }

    // -- assert_symmetric --

    #[test]
    fn test_assert_symmetric_ok() {
        let m = array![[1.0, 2.0, 3.0], [2.0, 5.0, 6.0], [3.0, 6.0, 9.0]];
        assert!(assert_symmetric(&m.view(), "m", 1e-12).is_ok());
    }

    #[test]
    fn test_assert_symmetric_fail() {
        let m = array![[1.0, 2.0], [3.0, 4.0]]; // not symmetric
        assert!(assert_symmetric(&m.view(), "m", 1e-12).is_err());
    }

    #[test]
    fn test_assert_symmetric_non_square() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(assert_symmetric(&m.view(), "m", 1e-12).is_err());
    }

    // -- assert_orthogonal --

    #[test]
    fn test_assert_orthogonal_identity() {
        let m = Array2::<f64>::eye(3);
        assert!(assert_orthogonal(&m.view(), "I", 1e-10).is_ok());
    }

    #[test]
    fn test_assert_orthogonal_fail() {
        let m = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(assert_orthogonal(&m.view(), "m", 1e-10).is_err());
    }

    // -- assert_positive_definite --

    #[test]
    fn test_assert_positive_definite_ok() {
        // 2x2 positive definite
        let m = array![[4.0, 2.0], [2.0, 3.0]];
        assert!(assert_positive_definite(&m.view(), "m").is_ok());
    }

    #[test]
    fn test_assert_positive_definite_fail() {
        // Not positive definite (negative eigenvalue)
        let m = array![[1.0, 5.0], [5.0, 1.0]];
        assert!(assert_positive_definite(&m.view(), "m").is_err());
    }

    #[test]
    fn test_assert_positive_definite_3x3() {
        let m = array![
            [4.0, 12.0, -16.0],
            [12.0, 37.0, -43.0],
            [-16.0, -43.0, 98.0]
        ];
        assert!(assert_positive_definite(&m.view(), "m").is_ok());
    }

    // -- assert_stochastic --

    #[test]
    fn test_assert_stochastic_ok() {
        let m = array![[0.2, 0.3, 0.5], [0.1, 0.8, 0.1]];
        assert!(assert_stochastic(&m.view(), "m", 1e-10).is_ok());
    }

    #[test]
    fn test_assert_stochastic_bad_sum() {
        let m = array![[0.2, 0.3, 0.4], [0.1, 0.8, 0.1]]; // row 0 sums to 0.9
        assert!(assert_stochastic(&m.view(), "m", 1e-10).is_err());
    }

    #[test]
    fn test_assert_stochastic_negative() {
        let m = array![[0.5, 0.5], [-0.1, 1.1]];
        assert!(assert_stochastic(&m.view(), "m", 1e-10).is_err());
    }

    // -- assert_shape --

    #[test]
    fn test_assert_shape_ok() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(assert_shape(&a, &[2, 2], "a").is_ok());
    }

    #[test]
    fn test_assert_shape_mismatch() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(assert_shape(&a, &[2, 3], "a").is_err());
    }

    // -- compute_array_stats --

    #[test]
    fn test_array_stats_basic() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_array_stats(&a).expect("should succeed");
        assert_eq!(stats.count, 5);
        assert!((stats.min - 1.0).abs() < 1e-12);
        assert!((stats.max - 5.0).abs() < 1e-12);
        assert!((stats.mean - 3.0).abs() < 1e-12);
        assert!(!stats.has_nan);
        assert!(!stats.has_inf);
        assert_eq!(stats.zero_count, 0);
        assert_eq!(stats.negative_count, 0);
    }

    #[test]
    fn test_array_stats_with_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        let stats = compute_array_stats(&a).expect("should succeed");
        assert!(stats.has_nan);
        assert!(!stats.has_inf);
        // finite elements: 1.0, 3.0
        assert!((stats.min - 1.0).abs() < 1e-12);
        assert!((stats.max - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_array_stats_with_inf() {
        let a = array![1.0, f64::INFINITY, -1.0];
        let stats = compute_array_stats(&a).expect("should succeed");
        assert!(stats.has_inf);
        assert_eq!(stats.negative_count, 1);
    }

    #[test]
    fn test_array_stats_empty() {
        let a: Array1<f64> = Array1::from_vec(vec![]);
        assert!(compute_array_stats(&a).is_err());
    }

    #[test]
    fn test_array_stats_display() {
        let a = array![1.0, 2.0, 3.0];
        let stats = compute_array_stats(&a).expect("should succeed");
        let display = format!("{stats}");
        assert!(display.contains("ArrayStats"));
        assert!(display.contains("n=3"));
    }

    // -- diagnose_array --

    #[test]
    fn test_diagnose_array_clean() {
        let a = array![1.0, 2.0, 3.0];
        let report = diagnose_array(&a, "test_array");
        assert!(report.contains("test_array"));
        assert!(report.contains("Issues: none"));
    }

    #[test]
    fn test_diagnose_array_with_nan() {
        let a = array![1.0, f64::NAN, 3.0];
        let report = diagnose_array(&a, "nan_array");
        assert!(report.contains("contains NaN"));
    }

    #[test]
    fn test_diagnose_array_all_zeros() {
        let a = array![0.0, 0.0, 0.0];
        let report = diagnose_array(&a, "zero_array");
        assert!(report.contains("all elements are zero"));
    }

    // -- assert_orthogonal with rotation matrix --

    #[test]
    fn test_assert_orthogonal_rotation() {
        let theta: f64 = std::f64::consts::PI / 4.0;
        let c = theta.cos();
        let s = theta.sin();
        let m = array![[c, -s], [s, c]];
        assert!(assert_orthogonal(&m.view(), "rot", 1e-10).is_ok());
    }

    // -- assert_positive_definite non-square --

    #[test]
    fn test_assert_positive_definite_non_square() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert!(assert_positive_definite(&m.view(), "m").is_err());
    }

    // -- stochastic identity rows --

    #[test]
    fn test_assert_stochastic_identity() {
        let m = Array2::<f64>::eye(3);
        assert!(assert_stochastic(&m.view(), "I", 1e-10).is_ok());
    }
}
