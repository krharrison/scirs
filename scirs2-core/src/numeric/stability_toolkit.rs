//! # Numeric Stability Toolkit
//!
//! Extended numeric stability utilities that complement the core `stability` module.
//!
//! This module provides:
//! - **Error helpers**: `relative_error`, `absolute_error`
//! - **Compensated summation**: convenience wrapper around Kahan accumulation
//! - **Stable activations**: `softmax_array`, `sigmoid_array` (ndarray-based)
//! - **Condition estimation**: `condition_number_1d` for 1-D ratio analysis
//! - **Numerical differentiation**: `numerical_gradient` with forward/backward/central modes
//! - **Gradient checking**: `check_gradient` to compare analytical vs numerical gradients

use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{Array1, ArrayView1};
use num_traits::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

// ---------------------------------------------------------------------------
// Error measurement helpers
// ---------------------------------------------------------------------------

/// Absolute error between two values: |a - b|.
pub fn absolute_error<T: Float>(a: T, b: T) -> T {
    (a - b).abs()
}

/// Relative error between `computed` and `reference`: |computed - reference| / |reference|.
///
/// Returns `T::infinity()` when `reference` is zero and `computed` is non-zero.
/// Returns `T::zero()` when both are zero.
pub fn relative_error<T: Float>(computed: T, reference: T) -> T {
    let diff = (computed - reference).abs();
    let denom = reference.abs();
    if denom.is_zero() {
        if diff.is_zero() {
            T::zero()
        } else {
            T::infinity()
        }
    } else {
        diff / denom
    }
}

/// Element-wise relative errors between two arrays.
/// Returns `Err` if the arrays have different lengths.
pub fn relative_errors<T: Float + Display>(
    computed: &ArrayView1<T>,
    reference: &ArrayView1<T>,
) -> CoreResult<Array1<T>> {
    if computed.len() != reference.len() {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "Array length mismatch: computed has {} elements, reference has {}",
            computed.len(),
            reference.len()
        ))));
    }
    let out: Vec<T> = computed
        .iter()
        .zip(reference.iter())
        .map(|(&c, &r)| relative_error(c, r))
        .collect();
    Ok(Array1::from_vec(out))
}

/// Maximum relative error across all elements.
pub fn max_relative_error<T: Float + Display>(
    computed: &ArrayView1<T>,
    reference: &ArrayView1<T>,
) -> CoreResult<T> {
    let errs = relative_errors(computed, reference)?;
    Ok(errs
        .iter()
        .copied()
        .fold(T::zero(), |acc, e| if e > acc { e } else { acc }))
}

// ---------------------------------------------------------------------------
// Compensated summation (convenience)
// ---------------------------------------------------------------------------

/// Compute a compensated (Kahan) sum of a slice.
///
/// This is a convenience wrapper that calls Kahan summation from the
/// `stability` module internals.
pub fn compensated_sum<T: Float>(values: &[T]) -> T {
    let mut sum = T::zero();
    let mut compensation = T::zero();
    for &val in values {
        let y = val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Compute a compensated (Neumaier) sum of an ndarray view.
///
/// Uses Neumaier's improvement over Kahan summation: when the addend
/// is larger in magnitude than the running sum the compensation tracks
/// the smaller value, giving correct results even for inputs like
/// `[1e20, 1.0, -1e20]`.
pub fn compensated_sum_array<T: Float>(values: &ArrayView1<T>) -> T {
    if values.is_empty() {
        return T::zero();
    }
    let mut sum = values[0];
    let mut compensation = T::zero();
    for &val in values.iter().skip(1) {
        let t = sum + val;
        if sum.abs() >= val.abs() {
            compensation = compensation + ((sum - t) + val);
        } else {
            compensation = compensation + ((val - t) + sum);
        }
        sum = t;
    }
    sum + compensation
}

// ---------------------------------------------------------------------------
// Pairwise summation (array-friendly)
// ---------------------------------------------------------------------------

/// Pairwise summation for an ndarray view.
///
/// Recursively splits the array and sums halves, achieving O(log n) error growth.
pub fn pairwise_sum_array<T: Float>(values: &ArrayView1<T>) -> T {
    const THRESHOLD: usize = 128;
    let n = values.len();
    match n {
        0 => T::zero(),
        1 => values[0],
        _ if n <= THRESHOLD => compensated_sum_array(values),
        _ => {
            let mid = n / 2;
            let left = values.slice(ndarray::s![..mid]);
            let right = values.slice(ndarray::s![mid..]);
            pairwise_sum_array(&left) + pairwise_sum_array(&right)
        }
    }
}

// ---------------------------------------------------------------------------
// Stable softmax / sigmoid for arrays
// ---------------------------------------------------------------------------

/// Numerically stable softmax for an ndarray 1-D array.
///
/// Subtracts the maximum before exponentiation to prevent overflow.
pub fn softmax_array<T: Float>(values: &ArrayView1<T>) -> Array1<T> {
    if values.is_empty() {
        return Array1::from_vec(vec![]);
    }
    let max_val = values
        .iter()
        .copied()
        .fold(T::neg_infinity(), |a, b| a.max(b));

    let exp_vals: Vec<T> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: T = exp_vals.iter().copied().fold(T::zero(), |a, b| a + b);
    Array1::from_vec(exp_vals.into_iter().map(|e| e / sum).collect())
}

/// Numerically stable sigmoid for an ndarray 1-D array.
pub fn sigmoid_array<T: Float>(values: &ArrayView1<T>) -> Array1<T> {
    let out: Vec<T> = values
        .iter()
        .map(|&x| {
            if x >= T::zero() {
                let exp_neg = (-x).exp();
                T::one() / (T::one() + exp_neg)
            } else {
                let exp_x = x.exp();
                exp_x / (T::one() + exp_x)
            }
        })
        .collect();
    Array1::from_vec(out)
}

/// Numerically stable log-sum-exp for an ndarray 1-D array.
pub fn log_sum_exp_array<T: Float>(values: &ArrayView1<T>) -> T {
    if values.is_empty() {
        return T::neg_infinity();
    }
    let max_val = values
        .iter()
        .copied()
        .fold(T::neg_infinity(), |a, b| a.max(b));
    if max_val.is_infinite() && max_val < T::zero() {
        return max_val;
    }
    let sum: T = values
        .iter()
        .map(|&v| (v - max_val).exp())
        .fold(T::zero(), |a, b| a + b);
    max_val + sum.ln()
}

// ---------------------------------------------------------------------------
// Condition number estimation for 1-D (ratio of max/min absolute values)
// ---------------------------------------------------------------------------

/// Estimate a "condition number" for a 1-D array as max(|x|) / min_nonzero(|x|).
///
/// This gives insight into the dynamic range and potential for cancellation.
/// Returns `Err` if the array is empty or all zeros.
pub fn condition_number_1d<T: Float + Display>(values: &ArrayView1<T>) -> CoreResult<T> {
    if values.is_empty() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "Cannot compute condition number of empty array",
        )));
    }
    let mut max_abs = T::zero();
    let mut min_abs = T::infinity();
    for &v in values.iter() {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
        if a > T::zero() && a < min_abs {
            min_abs = a;
        }
    }
    if max_abs.is_zero() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "All elements are zero; condition number is undefined",
        )));
    }
    if min_abs.is_infinite() {
        return Err(CoreError::ValueError(ErrorContext::new(
            "No non-zero elements found for condition number",
        )));
    }
    Ok(max_abs / min_abs)
}

// ---------------------------------------------------------------------------
// Numerical differentiation
// ---------------------------------------------------------------------------

/// Mode of finite difference approximation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifferenceMode {
    /// f'(x) ~ (f(x+h) - f(x)) / h
    Forward,
    /// f'(x) ~ (f(x) - f(x-h)) / h
    Backward,
    /// f'(x) ~ (f(x+h) - f(x-h)) / (2h)  -- O(h^2) accuracy
    Central,
}

/// Compute the numerical gradient of a scalar function at point `x`.
///
/// * `f` -- function from R^n -> R, taking a slice and returning a scalar.
/// * `x` -- the point at which to evaluate the gradient.
/// * `h` -- step size (e.g. 1e-5).
/// * `mode` -- finite difference mode.
///
/// Returns an `Array1<T>` of the same length as `x`.
pub fn numerical_gradient<T, F>(f: &F, x: &[T], h: T, mode: DifferenceMode) -> CoreResult<Array1<T>>
where
    T: Float + FromPrimitive + Debug,
    F: Fn(&[T]) -> T,
{
    let n = x.len();
    let two = T::from_f64(2.0).ok_or_else(|| {
        CoreError::TypeError(ErrorContext::new("Failed to convert 2.0 to target type"))
    })?;

    let mut grad = Array1::zeros(n);
    let mut x_perturbed = x.to_vec();

    for i in 0..n {
        let original = x_perturbed[i];

        match mode {
            DifferenceMode::Forward => {
                x_perturbed[i] = original + h;
                let f_plus = f(&x_perturbed);
                x_perturbed[i] = original;
                let f_0 = f(&x_perturbed);
                grad[i] = (f_plus - f_0) / h;
            }
            DifferenceMode::Backward => {
                x_perturbed[i] = original;
                let f_0 = f(&x_perturbed);
                x_perturbed[i] = original - h;
                let f_minus = f(&x_perturbed);
                grad[i] = (f_0 - f_minus) / h;
            }
            DifferenceMode::Central => {
                x_perturbed[i] = original + h;
                let f_plus = f(&x_perturbed);
                x_perturbed[i] = original - h;
                let f_minus = f(&x_perturbed);
                grad[i] = (f_plus - f_minus) / (two * h);
            }
        }

        // Restore
        x_perturbed[i] = original;
    }

    Ok(grad)
}

// ---------------------------------------------------------------------------
// Gradient checking
// ---------------------------------------------------------------------------

/// Result of a gradient check.
#[derive(Debug, Clone)]
pub struct GradientCheckResult<T: Float> {
    /// Element-wise relative errors between analytical and numerical gradients.
    pub relative_errors: Array1<T>,
    /// Maximum relative error.
    pub max_relative_error: T,
    /// Mean relative error.
    pub mean_relative_error: T,
    /// Whether the check passed (max_relative_error < tolerance).
    pub passed: bool,
}

impl<T: Float + Display> std::fmt::Display for GradientCheckResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradientCheck(passed={}, max_rel_err={}, mean_rel_err={})",
            self.passed, self.max_relative_error, self.mean_relative_error,
        )
    }
}

/// Check an analytical gradient against a numerical gradient.
///
/// * `f` -- scalar function R^n -> R.
/// * `analytical_grad` -- the gradient your code computes at `x`.
/// * `x` -- the point at which the gradient was computed.
/// * `h` -- finite difference step size (e.g. 1e-5).
/// * `tolerance` -- maximum allowed relative error per component.
///
/// Returns a `GradientCheckResult` with element-wise details.
pub fn check_gradient<T, F>(
    f: &F,
    analytical_grad: &ArrayView1<T>,
    x: &[T],
    h: T,
    tolerance: T,
) -> CoreResult<GradientCheckResult<T>>
where
    T: Float + FromPrimitive + Debug + Display,
    F: Fn(&[T]) -> T,
{
    if analytical_grad.len() != x.len() {
        return Err(CoreError::ShapeError(ErrorContext::new(format!(
            "Analytical gradient length {} does not match input dimension {}",
            analytical_grad.len(),
            x.len()
        ))));
    }

    let numerical = numerical_gradient(f, x, h, DifferenceMode::Central)?;
    let rel_errs = relative_errors(&analytical_grad, &numerical.view())?;
    let max_err = rel_errs
        .iter()
        .copied()
        .fold(T::zero(), |a, b| if b > a { b } else { a });

    let n_f = T::from_usize(rel_errs.len().max(1)).unwrap_or(T::one());
    let sum_err = rel_errs.iter().copied().fold(T::zero(), |a, b| a + b);
    let mean_err = sum_err / n_f;

    Ok(GradientCheckResult {
        relative_errors: rel_errs,
        max_relative_error: max_err,
        mean_relative_error: mean_err,
        passed: max_err < tolerance,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    #[test]
    fn test_absolute_error() {
        assert!((absolute_error(3.0_f64, 3.0) - 0.0).abs() < 1e-15);
        assert!((absolute_error(3.5_f64, 3.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_relative_error_basic() {
        assert!((relative_error(1.01_f64, 1.0) - 0.01).abs() < 1e-10);
        assert!((relative_error(0.0_f64, 0.0) - 0.0).abs() < 1e-15);
        assert!(relative_error(1.0_f64, 0.0).is_infinite());
    }

    #[test]
    fn test_relative_errors_array() {
        let computed = array![1.01, 2.02, 3.03];
        let reference = array![1.0, 2.0, 3.0];
        let errs = relative_errors(&computed.view(), &reference.view()).expect("should succeed");
        assert_eq!(errs.len(), 3);
        for &e in errs.iter() {
            assert!(e < 0.02);
        }
    }

    #[test]
    fn test_relative_errors_mismatch() {
        let a = array![1.0, 2.0];
        let b = array![1.0];
        assert!(relative_errors(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_compensated_sum_accuracy() {
        // Many small values that lose precision with naive sum
        let values: Vec<f64> = (0..10_000).map(|_| 0.01).collect();
        let result = compensated_sum(&values);
        assert!((result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_compensated_sum_array_view() {
        let arr = array![1e20, 1.0, -1e20];
        let result = compensated_sum_array(&arr.view());
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pairwise_sum_array() {
        let arr: Array1<f64> = Array1::from_vec((0..500).map(|i| 0.1 + 0.001 * i as f64).collect());
        let pw = pairwise_sum_array(&arr.view());
        let naive: f64 = arr.iter().sum();
        assert!((pw - naive).abs() < 1e-8);
    }

    #[test]
    fn test_softmax_array() {
        let vals = array![1000.0_f64, 1000.0, 1000.0];
        let sm = softmax_array(&vals.view());
        for &p in sm.iter() {
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
        let total: f64 = sm.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_empty() {
        let vals: Array1<f64> = Array1::from_vec(vec![]);
        let sm = softmax_array(&vals.view());
        assert!(sm.is_empty());
    }

    #[test]
    fn test_sigmoid_array() {
        let vals = array![0.0_f64, 100.0, -100.0];
        let sig = sigmoid_array(&vals.view());
        assert!((sig[0] - 0.5).abs() < 1e-10);
        assert!((sig[1] - 1.0).abs() < 1e-10);
        assert!(sig[2] < 1e-30);
    }

    #[test]
    fn test_log_sum_exp_array() {
        let vals = array![1000.0_f64, 1000.0, 1000.0];
        let lse = log_sum_exp_array(&vals.view());
        let expected = 1000.0 + 3.0_f64.ln();
        assert!((lse - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_array_empty() {
        let vals: Array1<f64> = Array1::from_vec(vec![]);
        let lse = log_sum_exp_array(&vals.view());
        assert!(lse.is_infinite() && lse < 0.0);
    }

    #[test]
    fn test_condition_number_1d() {
        let vals = array![1.0_f64, 10.0, 100.0];
        let cn = condition_number_1d(&vals.view()).expect("should succeed");
        assert!((cn - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_number_1d_all_zeros() {
        let vals = array![0.0_f64, 0.0];
        assert!(condition_number_1d(&vals.view()).is_err());
    }

    #[test]
    fn test_condition_number_1d_empty() {
        let vals: Array1<f64> = Array1::from_vec(vec![]);
        assert!(condition_number_1d(&vals.view()).is_err());
    }

    #[test]
    fn test_numerical_gradient_forward() {
        // f(x) = x0^2 + x1^2, grad = [2*x0, 2*x1]
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = [3.0, 4.0];
        let grad =
            numerical_gradient(&f, &x, 1e-7, DifferenceMode::Forward).expect("should succeed");
        assert!((grad[0] - 6.0).abs() < 1e-4);
        assert!((grad[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_numerical_gradient_backward() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = [3.0, 4.0];
        let grad =
            numerical_gradient(&f, &x, 1e-7, DifferenceMode::Backward).expect("should succeed");
        assert!((grad[0] - 6.0).abs() < 1e-4);
        assert!((grad[1] - 8.0).abs() < 1e-4);
    }

    #[test]
    fn test_numerical_gradient_central() {
        let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let x = [3.0, 4.0];
        let grad =
            numerical_gradient(&f, &x, 1e-5, DifferenceMode::Central).expect("should succeed");
        // Central difference should be more accurate
        assert!((grad[0] - 6.0).abs() < 1e-8);
        assert!((grad[1] - 8.0).abs() < 1e-8);
    }

    #[test]
    fn test_numerical_gradient_sin() {
        // f(x) = sin(x0), grad = [cos(x0)]
        let f = |x: &[f64]| x[0].sin();
        let x = [std::f64::consts::PI / 4.0];
        let grad =
            numerical_gradient(&f, &x, 1e-7, DifferenceMode::Central).expect("should succeed");
        let expected = (std::f64::consts::PI / 4.0).cos();
        assert!((grad[0] - expected).abs() < 1e-8);
    }

    #[test]
    fn test_check_gradient_passes() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let x = [3.0, 4.0];
        let analytical = array![6.0, 16.0]; // [2*x0, 4*x1]
        let result =
            check_gradient(&f, &analytical.view(), &x, 1e-5, 1e-4).expect("should succeed");
        assert!(result.passed, "gradient check should pass");
        assert!(result.max_relative_error < 1e-4);
    }

    #[test]
    fn test_check_gradient_fails() {
        let f = |x: &[f64]| x[0] * x[0] + 2.0 * x[1] * x[1];
        let x = [3.0, 4.0];
        let bad_analytical = array![100.0, 200.0]; // wrong gradient
        let result =
            check_gradient(&f, &bad_analytical.view(), &x, 1e-5, 1e-4).expect("should succeed");
        assert!(
            !result.passed,
            "gradient check should fail with wrong gradient"
        );
    }

    #[test]
    fn test_check_gradient_dimension_mismatch() {
        let f = |x: &[f64]| x[0];
        let x = [1.0, 2.0];
        let analytical = array![1.0]; // wrong dimension
        assert!(check_gradient(&f, &analytical.view(), &x, 1e-5, 1e-4).is_err());
    }

    #[test]
    fn test_max_relative_error() {
        let a = array![1.1_f64, 2.2, 3.3];
        let b = array![1.0, 2.0, 3.0];
        let mre = max_relative_error(&a.view(), &b.view()).expect("should succeed");
        assert!(mre > 0.09 && mre < 0.11);
    }

    #[test]
    fn test_compensated_sum_empty() {
        let empty: Vec<f64> = vec![];
        assert!((compensated_sum(&empty) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_gradient_check_display() {
        let f = |x: &[f64]| x[0] * x[0];
        let x = [2.0];
        let analytical = array![4.0];
        let result =
            check_gradient(&f, &analytical.view(), &x, 1e-5, 1e-4).expect("should succeed");
        let display = format!("{result}");
        assert!(display.contains("GradientCheck"));
    }
}
