//! Array assertion functions
//!
//! This module extends the low-level array checks provided by
//! [`crate::validation::array_checks`] with higher-level, composable
//! assertion functions that cover common scientific computing invariants.
//!
//! All functions avoid `unwrap()`, return `CoreResult`, and work on any
//! ndarray `ArrayBase` via generic bounds.
//!
//! ## Available assertions
//!
//! | Function | Description |
//! |---|---|
//! | [`assert_finite`] | All elements are finite (re-exported) |
//! | [`assert_non_negative`] | All elements ≥ 0 (re-exported) |
//! | [`assert_probability`] | All elements in \[0, 1\] |
//! | [`assert_monotone`] | Monotone increasing or decreasing |
//! | [`assert_symmetric`] | Matrix symmetry (re-exported) |
//! | [`assert_bounded`] | All elements in \[lo, hi\] |
//! | [`assert_no_nan`] | No NaN values present |
//! | [`assert_no_inf`] | No infinite values present |
//! | [`assert_sum_to`] | Sum of all elements equals target within tolerance |
//! | [`assert_unique`] | All elements are distinct (within tolerance) |

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use ::ndarray::{ArrayBase, ArrayView1, Axis, Dimension};
use num_traits::Float;
use std::fmt::Display;

// Re-export the primitives from the sibling module so callers only need one import.
pub use super::array_checks::{
    assert_finite, assert_non_negative, assert_orthogonal, assert_positive,
    assert_positive_definite, assert_shape, assert_stochastic, assert_symmetric,
};

// ---------------------------------------------------------------------------
// assert_probability
// ---------------------------------------------------------------------------

/// Assert that every element of the array lies in \[0, 1\] (inclusive).
///
/// Useful for checking probability vectors, softmax outputs, Bernoulli
/// parameters, etc.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any element is `< 0` or `> 1`.
/// Non-finite elements (NaN, ±Inf) are also rejected.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use scirs2_core::validation::assertions::assert_probability;
///
/// assert!(assert_probability(&array![0.0, 0.5, 1.0], "probs").is_ok());
/// assert!(assert_probability(&array![0.0, 1.1], "probs").is_err());
/// assert!(assert_probability(&array![f64::NAN], "probs").is_err());
/// ```
pub fn assert_probability<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if !val.is_finite() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains non-finite value {val} at index {idx:?}; \
                     all elements must be finite probabilities in [0, 1]"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
        if val < F::zero() || val > F::one() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains value {val} at index {idx:?} outside [0, 1]"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Monotonicity direction
// ---------------------------------------------------------------------------

/// Direction of monotonicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monotonicity {
    /// Each element is ≥ the previous one (non-decreasing).
    NonDecreasing,
    /// Each element is > the previous one (strictly increasing).
    StrictlyIncreasing,
    /// Each element is ≤ the previous one (non-increasing).
    NonIncreasing,
    /// Each element is < the previous one (strictly decreasing).
    StrictlyDecreasing,
}

// ---------------------------------------------------------------------------
// assert_monotone
// ---------------------------------------------------------------------------

/// Assert that the 1-D view is monotone according to the specified direction.
///
/// Works on any `ArrayView1<F>` where `F: Float + Display`.
/// Arrays of length 0 or 1 trivially satisfy any monotonicity condition.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if the sequence violates the required
/// monotonicity.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use scirs2_core::validation::assertions::{assert_monotone, Monotonicity};
///
/// let asc = array![1.0_f64, 2.0, 3.0];
/// assert!(assert_monotone(&asc.view(), "x", Monotonicity::NonDecreasing).is_ok());
/// assert!(assert_monotone(&asc.view(), "x", Monotonicity::StrictlyIncreasing).is_ok());
///
/// let desc = array![3.0_f64, 2.0, 1.0];
/// assert!(assert_monotone(&desc.view(), "x", Monotonicity::StrictlyDecreasing).is_ok());
///
/// let flat = array![1.0_f64, 1.0, 1.0];
/// assert!(assert_monotone(&flat.view(), "x", Monotonicity::NonDecreasing).is_ok());
/// assert!(assert_monotone(&flat.view(), "x", Monotonicity::StrictlyIncreasing).is_err());
/// ```
pub fn assert_monotone<F>(
    array: &ArrayView1<F>,
    name: &str,
    direction: Monotonicity,
) -> CoreResult<()>
where
    F: Float + Display,
{
    if array.len() <= 1 {
        return Ok(());
    }
    let elements: Vec<F> = array.iter().copied().collect();

    for window in elements.windows(2) {
        let (prev, next) = (window[0], window[1]);

        let ok = match direction {
            Monotonicity::NonDecreasing => next >= prev,
            Monotonicity::StrictlyIncreasing => next > prev,
            Monotonicity::NonIncreasing => next <= prev,
            Monotonicity::StrictlyDecreasing => next < prev,
        };

        if !ok {
            let direction_label = match direction {
                Monotonicity::NonDecreasing => "non-decreasing",
                Monotonicity::StrictlyIncreasing => "strictly increasing",
                Monotonicity::NonIncreasing => "non-increasing",
                Monotonicity::StrictlyDecreasing => "strictly decreasing",
            };
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} is not {direction_label}: found {prev} followed by {next}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_monotone_along_axis (N-D generalisation)
// ---------------------------------------------------------------------------

/// Assert that an N-dimensional array is monotone along a specified axis.
///
/// The check is applied independently to each 1-D slice along `axis`.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any slice violates the monotonicity.
pub fn assert_monotone_along_axis<S, D, F>(
    array: &ArrayBase<S, D>,
    axis: usize,
    name: &str,
    direction: Monotonicity,
) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    let ndim = array.ndim();
    if axis >= ndim {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{name}: axis {axis} is out of range for array with {ndim} dimensions"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    for (slice_idx, lane) in array.lanes(Axis(axis)).into_iter().enumerate() {
        let lane_owned = lane.to_owned();
        assert_monotone(&lane_owned.view(), &format!("{name}[lane {slice_idx}]"), direction)
            .map_err(|e| {
                CoreError::ValueError(
                    ErrorContext::new(format!(
                        "{name}: monotonicity violation in lane {slice_idx} along axis {axis}: {e}"
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                )
            })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_bounded
// ---------------------------------------------------------------------------

/// Assert that every element lies in \[lo, hi\] (inclusive).
///
/// A stricter alternative to separate min/max checks that reports both bounds
/// in a single error message.
///
/// # Errors
///
/// Returns `CoreError::ValueError` on the first element outside \[lo, hi\].
pub fn assert_bounded<S, D, F>(
    array: &ArrayBase<S, D>,
    lo: F,
    hi: F,
    name: &str,
) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    if lo > hi {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{name}: invalid bound specification: lo ({lo}) > hi ({hi})"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    for (idx, &val) in array.indexed_iter() {
        if !val.is_finite() || val < lo || val > hi {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains value {val} at {idx:?} outside [{lo}, {hi}]"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_no_nan
// ---------------------------------------------------------------------------

/// Assert that the array contains no NaN values.
///
/// Unlike [`assert_finite`], this allows infinite values.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any element is NaN.
pub fn assert_no_nan<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if val.is_nan() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains NaN at index {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_no_inf
// ---------------------------------------------------------------------------

/// Assert that the array contains no infinite values.
///
/// Unlike [`assert_finite`], this allows NaN values.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any element is ±Inf.
pub fn assert_no_inf<S, D, F>(array: &ArrayBase<S, D>, name: &str) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    for (idx, &val) in array.indexed_iter() {
        if val.is_infinite() {
            return Err(CoreError::ValueError(
                ErrorContext::new(format!(
                    "{name} contains infinite value {val} at index {idx:?}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_sum_to
// ---------------------------------------------------------------------------

/// Assert that the sum of all elements equals `target` within `tolerance`.
///
/// Useful for checking probability distributions, partition-of-unity
/// constraints, etc.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if `|sum - target| > tolerance`.
pub fn assert_sum_to<S, D, F>(
    array: &ArrayBase<S, D>,
    target: F,
    tolerance: F,
    name: &str,
) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display + std::iter::Sum<F>,
{
    let sum: F = array.iter().copied().sum();
    let diff = (sum - target).abs();
    if diff > tolerance {
        return Err(CoreError::ValueError(
            ErrorContext::new(format!(
                "{name} elements sum to {sum}, expected {target} (|diff|={diff} > tolerance={tolerance})"
            ))
            .with_location(ErrorLocation::new(file!(), line!())),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// assert_unique
// ---------------------------------------------------------------------------

/// Assert that all elements are distinct (no two elements differ by less than
/// `tolerance` in absolute value).
///
/// Elements are considered equal if `|a - b| < tolerance`.  For exact integer
/// comparisons pass `tolerance = 0.0`.
///
/// Complexity: O(n²). For large arrays consider sorting first.
///
/// # Errors
///
/// Returns `CoreError::ValueError` if any two elements are within `tolerance`
/// of each other.
pub fn assert_unique<S, D, F>(
    array: &ArrayBase<S, D>,
    tolerance: F,
    name: &str,
) -> CoreResult<()>
where
    S: ::ndarray::Data<Elem = F>,
    D: Dimension,
    F: Float + Display,
{
    let values: Vec<F> = array.iter().copied().collect();
    for i in 0..values.len() {
        for j in (i + 1)..values.len() {
            let diff = (values[i] - values[j]).abs();
            if diff <= tolerance {
                return Err(CoreError::ValueError(
                    ErrorContext::new(format!(
                        "{name} has duplicate values: elements at flat indices {i} ({}) \
                         and {j} ({}) differ by {diff} ≤ tolerance {tolerance}",
                        values[i], values[j]
                    ))
                    .with_location(ErrorLocation::new(file!(), line!())),
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::{array, Array1, Array2};

    // -- assert_probability --

    #[test]
    fn test_probability_ok() {
        let a = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];
        assert!(assert_probability(&a, "p").is_ok());
    }

    #[test]
    fn test_probability_below_zero() {
        let a = array![-0.01_f64, 0.5];
        assert!(assert_probability(&a, "p").is_err());
    }

    #[test]
    fn test_probability_above_one() {
        let a = array![0.5_f64, 1.01];
        assert!(assert_probability(&a, "p").is_err());
    }

    #[test]
    fn test_probability_nan() {
        let a = array![0.5_f64, f64::NAN];
        assert!(assert_probability(&a, "p").is_err());
    }

    #[test]
    fn test_probability_inf() {
        let a = array![0.5_f64, f64::INFINITY];
        assert!(assert_probability(&a, "p").is_err());
    }

    // -- assert_monotone --

    #[test]
    fn test_monotone_non_decreasing_ok() {
        let a = array![1.0_f64, 1.0, 2.0, 3.0];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::NonDecreasing).is_ok());
    }

    #[test]
    fn test_monotone_strictly_increasing_ok() {
        let a = array![1.0_f64, 2.0, 3.0];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyIncreasing).is_ok());
    }

    #[test]
    fn test_monotone_strictly_increasing_fail_flat() {
        let a = array![1.0_f64, 1.0, 2.0];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyIncreasing).is_err());
    }

    #[test]
    fn test_monotone_non_increasing_ok() {
        let a = array![3.0_f64, 2.0, 2.0, 1.0];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::NonIncreasing).is_ok());
    }

    #[test]
    fn test_monotone_strictly_decreasing_ok() {
        let a = array![3.0_f64, 2.0, 1.0];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyDecreasing).is_ok());
    }

    #[test]
    fn test_monotone_strictly_decreasing_fail() {
        let a = array![3.0_f64, 2.0, 2.5];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyDecreasing).is_err());
    }

    #[test]
    fn test_monotone_empty() {
        let a: Array1<f64> = Array1::from_vec(vec![]);
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyIncreasing).is_ok());
    }

    #[test]
    fn test_monotone_single() {
        let a = array![42.0_f64];
        assert!(assert_monotone(&a.view(), "a", Monotonicity::StrictlyIncreasing).is_ok());
    }

    // -- assert_monotone_along_axis --

    #[test]
    fn test_monotone_along_axis_0_ok() {
        // Each column is non-decreasing along axis 0
        let m = array![[1.0_f64, 10.0], [2.0, 20.0], [3.0, 30.0]];
        assert!(
            assert_monotone_along_axis(&m, 0, "m", Monotonicity::NonDecreasing).is_ok()
        );
    }

    #[test]
    fn test_monotone_along_axis_0_fail() {
        let m = array![[1.0_f64, 10.0], [2.0, 5.0], [3.0, 30.0]];
        assert!(
            assert_monotone_along_axis(&m, 0, "m", Monotonicity::StrictlyIncreasing).is_err()
        );
    }

    #[test]
    fn test_monotone_along_axis_out_of_range() {
        let a = array![1.0_f64, 2.0, 3.0];
        assert!(
            assert_monotone_along_axis(&a, 5, "a", Monotonicity::NonDecreasing).is_err()
        );
    }

    // -- assert_bounded --

    #[test]
    fn test_bounded_ok() {
        let a = array![0.0_f64, 0.5, 1.0];
        assert!(assert_bounded(&a, 0.0, 1.0, "a").is_ok());
    }

    #[test]
    fn test_bounded_fail() {
        let a = array![0.0_f64, 1.5];
        assert!(assert_bounded(&a, 0.0, 1.0, "a").is_err());
    }

    #[test]
    fn test_bounded_nan() {
        let a = array![0.5_f64, f64::NAN];
        assert!(assert_bounded(&a, 0.0, 1.0, "a").is_err());
    }

    #[test]
    fn test_bounded_invalid_bounds() {
        let a = array![0.5_f64];
        assert!(assert_bounded(&a, 1.0, 0.0, "a").is_err()); // lo > hi
    }

    // -- assert_no_nan --

    #[test]
    fn test_no_nan_ok() {
        let a = array![1.0_f64, f64::INFINITY, -1.0];
        assert!(assert_no_nan(&a, "a").is_ok()); // inf is ok for no_nan
    }

    #[test]
    fn test_no_nan_fail() {
        let a = array![1.0_f64, f64::NAN];
        assert!(assert_no_nan(&a, "a").is_err());
    }

    // -- assert_no_inf --

    #[test]
    fn test_no_inf_ok() {
        let a = array![1.0_f64, f64::NAN, -1.0];
        assert!(assert_no_inf(&a, "a").is_ok()); // nan is ok for no_inf
    }

    #[test]
    fn test_no_inf_fail_pos() {
        let a = array![1.0_f64, f64::INFINITY];
        assert!(assert_no_inf(&a, "a").is_err());
    }

    #[test]
    fn test_no_inf_fail_neg() {
        let a = array![1.0_f64, f64::NEG_INFINITY];
        assert!(assert_no_inf(&a, "a").is_err());
    }

    // -- assert_sum_to --

    #[test]
    fn test_sum_to_ok() {
        let a = array![0.25_f64, 0.25, 0.25, 0.25];
        assert!(assert_sum_to(&a, 1.0, 1e-10, "a").is_ok());
    }

    #[test]
    fn test_sum_to_fail() {
        let a = array![0.3_f64, 0.3, 0.3];
        assert!(assert_sum_to(&a, 1.0, 1e-10, "a").is_err());
    }

    // -- assert_unique --

    #[test]
    fn test_unique_ok() {
        let a = array![1.0_f64, 2.0, 3.0, 4.0];
        assert!(assert_unique(&a, 1e-10, "a").is_ok());
    }

    #[test]
    fn test_unique_fail_exact() {
        let a = array![1.0_f64, 2.0, 1.0];
        assert!(assert_unique(&a, 0.0, "a").is_err());
    }

    #[test]
    fn test_unique_fail_within_tolerance() {
        let a = array![1.0_f64, 1.0000000001, 2.0];
        assert!(assert_unique(&a, 1e-8, "a").is_err());
    }

    // -- re-exported checks --

    #[test]
    fn test_re_exported_finite() {
        let a = array![1.0_f64, 2.0];
        assert!(assert_finite(&a, "a").is_ok());
    }

    #[test]
    fn test_re_exported_symmetric() {
        let m = Array2::<f64>::eye(3);
        assert!(assert_symmetric(&m.view(), "I", 1e-12).is_ok());
    }
}
