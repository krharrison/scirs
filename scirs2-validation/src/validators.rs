//! Generic validation traits and property-test helpers.
//!
//! This module defines the [`ValidatableDistribution`] trait that any stats library
//! can implement, plus a suite of validation functions that check implementations
//! against reference values and verify general mathematical properties.

use crate::reference_values::DistributionReference;

/// Trait for distributions that can be validated against reference values.
///
/// Implementors provide the core functions (pdf, cdf, mean, variance)
/// that the validation framework will test.
pub trait ValidatableDistribution {
    /// Probability density (or mass) function evaluated at `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function evaluated at `x`.
    fn cdf(&self, x: f64) -> f64;

    /// Analytical mean of the distribution.
    fn mean(&self) -> f64;

    /// Analytical variance of the distribution.
    fn variance(&self) -> f64;

    /// Percent-point (quantile) function, if available.
    /// Returns `None` if the implementation does not support PPF.
    fn ppf(&self, _p: f64) -> Option<f64> {
        None
    }
}

/// Outcome of validating a single distribution against its reference.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Name of the distribution that was validated
    pub distribution: String,
    /// PDF errors: `(x, expected, actual)` for points that exceeded tolerance
    pub pdf_errors: Vec<(f64, f64, f64)>,
    /// CDF errors: `(x, expected, actual)` for points that exceeded tolerance
    pub cdf_errors: Vec<(f64, f64, f64)>,
    /// PPF errors: `(p, expected, actual)` for points that exceeded tolerance
    pub ppf_errors: Vec<(f64, f64, f64)>,
    /// Absolute error of the mean (NaN if reference mean is NaN)
    pub mean_error: f64,
    /// Absolute error of the variance (NaN if reference variance is NaN or infinite)
    pub variance_error: f64,
    /// Whether all checks passed within tolerance
    pub passed: bool,
}

impl core::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        write!(f, "[{}] {}", status, self.distribution)?;
        if !self.pdf_errors.is_empty() {
            write!(f, " ({} PDF errors)", self.pdf_errors.len())?;
        }
        if !self.cdf_errors.is_empty() {
            write!(f, " ({} CDF errors)", self.cdf_errors.len())?;
        }
        if !self.ppf_errors.is_empty() {
            write!(f, " ({} PPF errors)", self.ppf_errors.len())?;
        }
        if self.mean_error.is_finite() && self.mean_error > 0.0 {
            write!(f, " (mean err: {:.3e})", self.mean_error)?;
        }
        if self.variance_error.is_finite() && self.variance_error > 0.0 {
            write!(f, " (var err: {:.3e})", self.variance_error)?;
        }
        Ok(())
    }
}

/// Validate a distribution implementation against a set of reference values.
///
/// Compares the distribution's `pdf`, `cdf`, `mean`, and `variance` against
/// the reference data. Points where the absolute error exceeds the tolerance
/// are recorded in the result.
///
/// # Arguments
///
/// * `dist` - The distribution implementation to validate
/// * `reference` - Reference values to compare against
/// * `pdf_tol` - Maximum allowed absolute error for PDF values
/// * `cdf_tol` - Maximum allowed absolute error for CDF values
///
/// # Returns
///
/// A [`ValidationResult`] summarizing all discrepancies found.
pub fn validate_distribution<D: ValidatableDistribution>(
    dist: &D,
    reference: &DistributionReference,
    pdf_tol: f64,
    cdf_tol: f64,
) -> ValidationResult {
    let mut pdf_errors = Vec::new();
    let mut cdf_errors = Vec::new();
    let mut ppf_errors = Vec::new();

    // Check PDF values
    for &(x, expected) in reference.pdf_points {
        let actual = dist.pdf(x);
        if (actual - expected).abs() > pdf_tol {
            pdf_errors.push((x, expected, actual));
        }
    }

    // Check CDF values
    for &(x, expected) in reference.cdf_points {
        let actual = dist.cdf(x);
        if (actual - expected).abs() > cdf_tol {
            cdf_errors.push((x, expected, actual));
        }
    }

    // Check PPF values (using CDF tolerance)
    for &(p, expected) in reference.ppf_points {
        if let Some(actual) = dist.ppf(p) {
            if (actual - expected).abs() > cdf_tol {
                ppf_errors.push((p, expected, actual));
            }
        }
    }

    // Check moments
    let mean_error = if reference.mean.is_nan() {
        // If reference mean is NaN, implementation mean should also be NaN
        if dist.mean().is_nan() {
            0.0
        } else {
            f64::NAN
        }
    } else {
        (dist.mean() - reference.mean).abs()
    };

    let variance_error = if reference.variance.is_nan() || reference.variance.is_infinite() {
        // For undefined/infinite variance, just check consistency
        let dist_var = dist.variance();
        if (reference.variance.is_nan() && dist_var.is_nan())
            || (reference.variance.is_infinite() && dist_var.is_infinite())
        {
            0.0
        } else {
            f64::NAN
        }
    } else {
        (dist.variance() - reference.variance).abs()
    };

    let mean_ok = mean_error.is_nan()
        || mean_error <= pdf_tol
        || (reference.mean.is_nan() && dist.mean().is_nan());
    let var_ok = variance_error.is_nan()
        || variance_error <= pdf_tol
        || (reference.variance.is_infinite() && dist.variance().is_infinite());

    let passed = pdf_errors.is_empty()
        && cdf_errors.is_empty()
        && ppf_errors.is_empty()
        && mean_ok
        && var_ok;

    ValidationResult {
        distribution: reference.name.to_string(),
        pdf_errors,
        cdf_errors,
        ppf_errors,
        mean_error,
        variance_error,
        passed,
    }
}

/// Validate that a PDF integrates to approximately 1 using the trapezoidal rule.
///
/// This is a numerical property test: for any valid continuous PDF, the integral
/// over the support should equal 1. The trapezoidal rule is used for simplicity.
///
/// # Arguments
///
/// * `dist` - The distribution to test
/// * `lower` - Lower bound of integration
/// * `upper` - Upper bound of integration
/// * `n_points` - Number of quadrature points (more = more accurate)
/// * `tolerance` - Maximum allowed deviation from 1.0
///
/// # Returns
///
/// `true` if the numerical integral is within `tolerance` of 1.0.
pub fn validate_pdf_integral<D: ValidatableDistribution>(
    dist: &D,
    lower: f64,
    upper: f64,
    n_points: usize,
    tolerance: f64,
) -> bool {
    if n_points < 2 {
        return false;
    }

    let h = (upper - lower) / (n_points as f64 - 1.0);
    let mut integral = 0.0;

    // Trapezoidal rule
    for i in 0..n_points {
        let x = lower + (i as f64) * h;
        let fx = dist.pdf(x);
        let weight = if i == 0 || i == n_points - 1 {
            0.5
        } else {
            1.0
        };
        integral += weight * fx;
    }
    integral *= h;

    (integral - 1.0).abs() < tolerance
}

/// Validate that a CDF is monotone non-decreasing.
///
/// Checks that `cdf(points[i]) <= cdf(points[i+1])` for all consecutive pairs
/// in the given sorted slice of x-values.
///
/// # Arguments
///
/// * `dist` - The distribution to test
/// * `points` - Sorted slice of x-values to evaluate
///
/// # Returns
///
/// `true` if the CDF is non-decreasing at all given points.
pub fn validate_cdf_monotone<D: ValidatableDistribution>(dist: &D, points: &[f64]) -> bool {
    if points.len() < 2 {
        return true;
    }

    let mut prev = dist.cdf(points[0]);
    for &x in &points[1..] {
        let current = dist.cdf(x);
        // Allow tiny floating-point regressions
        if current < prev - 1e-15 {
            return false;
        }
        prev = current;
    }
    true
}

/// Validate that PPF is the inverse of CDF (round-trip property).
///
/// For each probability `p` in `probabilities`, computes `x = ppf(p)` then
/// checks that `cdf(x) ~ p` within `tolerance`.
///
/// # Arguments
///
/// * `cdf_fn` - CDF function
/// * `ppf_fn` - PPF (quantile) function; returns `None` if undefined for that `p`
/// * `probabilities` - Probabilities to test (should be in (0, 1))
/// * `tolerance` - Maximum allowed round-trip error
///
/// # Returns
///
/// `true` if all round-trips are within tolerance.
pub fn validate_ppf_roundtrip(
    cdf_fn: &dyn Fn(f64) -> f64,
    ppf_fn: &dyn Fn(f64) -> Option<f64>,
    probabilities: &[f64],
    tolerance: f64,
) -> bool {
    for &p in probabilities {
        if let Some(x) = ppf_fn(p) {
            let roundtrip = cdf_fn(x);
            if (roundtrip - p).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

/// Validate CDF boundary conditions: CDF approaches 0 at lower tail, 1 at upper tail.
///
/// # Arguments
///
/// * `dist` - The distribution to test
/// * `lower_x` - A very small x-value in the lower tail
/// * `upper_x` - A very large x-value in the upper tail
/// * `tolerance` - How close to 0/1 the CDF should be
///
/// # Returns
///
/// `true` if CDF(lower_x) < tolerance and CDF(upper_x) > 1 - tolerance.
pub fn validate_cdf_bounds<D: ValidatableDistribution>(
    dist: &D,
    lower_x: f64,
    upper_x: f64,
    tolerance: f64,
) -> bool {
    let cdf_low = dist.cdf(lower_x);
    let cdf_high = dist.cdf(upper_x);
    cdf_low < tolerance && cdf_high > 1.0 - tolerance
}

/// Validate that PDF is non-negative everywhere in the given point set.
///
/// # Arguments
///
/// * `dist` - The distribution to test
/// * `points` - Slice of x-values to evaluate
///
/// # Returns
///
/// `true` if PDF is non-negative at all given points.
pub fn validate_pdf_nonnegative<D: ValidatableDistribution>(dist: &D, points: &[f64]) -> bool {
    for &x in points {
        let fx = dist.pdf(x);
        if fx < -1e-15 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference_values::EXPONENTIAL_1;

    /// A mock "perfect" distribution that returns the exact reference values.
    struct PerfectExponential;

    impl ValidatableDistribution for PerfectExponential {
        fn pdf(&self, x: f64) -> f64 {
            if x < 0.0 {
                0.0
            } else {
                (-x).exp()
            }
        }
        fn cdf(&self, x: f64) -> f64 {
            if x < 0.0 {
                0.0
            } else {
                1.0 - (-x).exp()
            }
        }
        fn mean(&self) -> f64 {
            1.0
        }
        fn variance(&self) -> f64 {
            1.0
        }
        fn ppf(&self, p: f64) -> Option<f64> {
            if p <= 0.0 || p >= 1.0 {
                None
            } else {
                Some(-(1.0 - p).ln())
            }
        }
    }

    /// A deliberately bad distribution for testing failure detection.
    struct BadDistribution;

    impl ValidatableDistribution for BadDistribution {
        fn pdf(&self, _x: f64) -> f64 {
            42.0
        }
        fn cdf(&self, _x: f64) -> f64 {
            0.99
        }
        fn mean(&self) -> f64 {
            -100.0
        }
        fn variance(&self) -> f64 {
            -5.0
        }
    }

    #[test]
    fn test_validate_mock_perfect_distribution() {
        let dist = PerfectExponential;
        let result = validate_distribution(&dist, &EXPONENTIAL_1, 1e-12, 1e-12);
        assert!(result.passed, "Perfect exponential should pass: {}", result);
        assert!(result.pdf_errors.is_empty());
        assert!(result.cdf_errors.is_empty());
    }

    #[test]
    fn test_validate_mock_bad_distribution() {
        let dist = BadDistribution;
        let result = validate_distribution(&dist, &EXPONENTIAL_1, 1e-9, 1e-9);
        assert!(!result.passed, "Bad distribution should fail validation");
        assert!(!result.pdf_errors.is_empty());
        assert!(!result.cdf_errors.is_empty());
    }

    #[test]
    fn test_pdf_integral_trapezoidal_accuracy() {
        let dist = PerfectExponential;
        // Integrate from 0 to 30 with 10000 points; exp(-30) ~ 1e-13
        assert!(validate_pdf_integral(&dist, 0.0, 30.0, 10000, 1e-4));
    }

    #[test]
    fn test_pdf_integral_too_few_points() {
        let dist = PerfectExponential;
        // n_points < 2 should return false
        assert!(!validate_pdf_integral(&dist, 0.0, 30.0, 1, 1e-4));
    }

    #[test]
    fn test_cdf_monotone_valid() {
        let dist = PerfectExponential;
        let points: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        assert!(validate_cdf_monotone(&dist, &points));
    }

    #[test]
    fn test_cdf_monotone_invalid() {
        // A distribution with non-monotone CDF
        struct NonMonotoneCdf;
        impl ValidatableDistribution for NonMonotoneCdf {
            fn pdf(&self, _x: f64) -> f64 {
                0.0
            }
            fn cdf(&self, x: f64) -> f64 {
                // Deliberately non-monotone
                if x > 1.0 && x < 2.0 {
                    0.3
                } else {
                    0.5
                }
            }
            fn mean(&self) -> f64 {
                0.0
            }
            fn variance(&self) -> f64 {
                1.0
            }
        }
        let dist = NonMonotoneCdf;
        let points = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
        assert!(!validate_cdf_monotone(&dist, &points));
    }

    #[test]
    fn test_ppf_roundtrip_valid() {
        let dist = PerfectExponential;
        let cdf_fn = |x: f64| dist.cdf(x);
        let ppf_fn = |p: f64| dist.ppf(p);
        let probs = vec![0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
        assert!(validate_ppf_roundtrip(&cdf_fn, &ppf_fn, &probs, 1e-10));
    }

    #[test]
    fn test_cdf_bounds() {
        let dist = PerfectExponential;
        assert!(validate_cdf_bounds(&dist, -10.0, 30.0, 1e-6));
    }

    #[test]
    fn test_pdf_nonnegative() {
        let dist = PerfectExponential;
        let points: Vec<f64> = (-50..100).map(|i| i as f64 * 0.1).collect();
        assert!(validate_pdf_nonnegative(&dist, &points));
    }

    #[test]
    fn test_validation_result_display() {
        let result = ValidationResult {
            distribution: "Test(1,2)".to_string(),
            pdf_errors: vec![(0.0, 1.0, 0.5)],
            cdf_errors: vec![],
            ppf_errors: vec![],
            mean_error: 0.001,
            variance_error: 0.0,
            passed: false,
        };
        let display = format!("{}", result);
        assert!(display.contains("FAIL"));
        assert!(display.contains("Test(1,2)"));
        assert!(display.contains("1 PDF errors"));
    }

    #[test]
    fn test_cdf_monotone_empty_and_single() {
        let dist = PerfectExponential;
        // Empty or single point should return true (vacuously)
        assert!(validate_cdf_monotone(&dist, &[]));
        assert!(validate_cdf_monotone(&dist, &[1.0]));
    }
}
