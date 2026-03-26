//! Distribution reference validation framework
//!
//! This module provides a systematic harness for testing statistical distributions
//! against known reference values derived from SciPy documentation.
//!
//! # Overview
//!
//! The validation framework consists of:
//! - [`DistributionRefPoint`]: a single (x, pdf, cdf, ppf) reference triplet
//! - [`DistributionValidation`]: a named collection of reference points with tolerances
//! - Helper functions [`check_pdf`], [`check_cdf`], [`check_ppf`] that compare computed
//!   values against references and report failures with contextual messages
//!
//! # Usage
//!
//! ```rust
//! use scirs2_stats::distributions::validation::{check_pdf, check_cdf};
//! let ok = check_pdf(0.398942, 0.398942280401433, 1e-6, "Normal(0,1)", 0.0);
//! assert!(ok);
//! ```

/// A single reference data point for a distribution.
///
/// Each field is `Option<f64>` so that only the values that are meaningful
/// (or can be verified) need to be populated.
#[derive(Debug, Clone)]
pub struct DistributionRefPoint {
    /// The x-value (or probability p for PPF)
    pub x: f64,
    /// Expected PDF value at x (continuous distributions only)
    pub pdf: Option<f64>,
    /// Expected CDF value at x
    pub cdf: Option<f64>,
    /// Expected PPF/quantile value when x is treated as probability p
    pub ppf: Option<f64>,
}

/// A named collection of reference points and tolerances for one distribution configuration.
#[derive(Debug, Clone)]
pub struct DistributionValidation {
    /// Human-readable name such as `"Normal(0,1)"` or `"Gamma(2,1)"`
    pub name: &'static str,
    /// Reference points to validate against
    pub points: &'static [DistributionRefPoint],
    /// Absolute tolerance for comparisons
    pub abs_tol: f64,
    /// Relative tolerance for comparisons (used when abs_tol check fails)
    pub rel_tol: f64,
}

/// Check that an actual PDF value matches the expected reference within tolerance.
///
/// Returns `true` if the absolute error is within `abs_tol`.  When it does not,
/// the function also prints a diagnostic line (using `eprintln!`) so that test
/// output contains useful context even when no panic occurs.
///
/// # Arguments
///
/// * `actual`    - Value returned by the distribution's `pdf()` method
/// * `expected`  - SciPy-derived reference value
/// * `abs_tol`   - Maximum permitted absolute error
/// * `dist_name` - Short description used in the diagnostic message
/// * `x`         - The x-argument at which the PDF was evaluated
pub fn check_pdf(actual: f64, expected: f64, abs_tol: f64, dist_name: &str, x: f64) -> bool {
    let err = (actual - expected).abs();
    if err <= abs_tol {
        return true;
    }
    eprintln!(
        "[FAIL] {dist_name} pdf({x}): actual={actual:.15}, expected={expected:.15}, err={err:.3e}, tol={abs_tol:.3e}"
    );
    false
}

/// Check that an actual CDF value matches the expected reference within tolerance.
///
/// Returns `true` if the absolute error is within `abs_tol`.
///
/// # Arguments
///
/// * `actual`    - Value returned by the distribution's `cdf()` method
/// * `expected`  - SciPy-derived reference value
/// * `abs_tol`   - Maximum permitted absolute error
/// * `dist_name` - Short description used in the diagnostic message
/// * `x`         - The x-argument at which the CDF was evaluated
pub fn check_cdf(actual: f64, expected: f64, abs_tol: f64, dist_name: &str, x: f64) -> bool {
    let err = (actual - expected).abs();
    if err <= abs_tol {
        return true;
    }
    eprintln!(
        "[FAIL] {dist_name} cdf({x}): actual={actual:.15}, expected={expected:.15}, err={err:.3e}, tol={abs_tol:.3e}"
    );
    false
}

/// Check that a PPF (quantile) value matches the expected reference within tolerance.
///
/// Returns `true` if the absolute error is within `abs_tol`.
///
/// # Arguments
///
/// * `actual`    - Value returned by the distribution's `ppf()` method
/// * `expected`  - SciPy-derived reference value
/// * `abs_tol`   - Maximum permitted absolute error
/// * `dist_name` - Short description used in the diagnostic message
/// * `p`         - The probability argument at which the PPF was evaluated
pub fn check_ppf(actual: f64, expected: f64, abs_tol: f64, dist_name: &str, p: f64) -> bool {
    let err = (actual - expected).abs();
    if err <= abs_tol {
        return true;
    }
    eprintln!(
        "[FAIL] {dist_name} ppf({p}): actual={actual:.15}, expected={expected:.15}, err={err:.3e}, tol={abs_tol:.3e}"
    );
    false
}
