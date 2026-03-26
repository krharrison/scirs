//! Numerical validation framework for scientific computing libraries.
//!
//! Provides composable validators for distribution functions, numerical algorithms,
//! and mathematical functions. Designed for cross-ecosystem reuse by NumRS2,
//! SkleaRS, and other COOLJAPAN crates.
//!
//! ## Overview
//!
//! The framework centers on the `NumericalValidator` trait, which every concrete
//! validator implements. Callers supply a *computed* function and a *reference*
//! function; the harness sweeps over the input grid, records per-point
//! `ValidationOutcome`s, and aggregates them into a `NumericalValidationReport`.
//!
//! Three ready-made validators are provided:
//!
//! - `ComparisonValidator` — general-purpose point comparison
//! - `MonotonicityChecker` — asserts strict/weak monotonic ordering
//! - `BoundaryChecker` — validates domain-boundary values
//! - `SymmetryChecker` — asserts even or odd function symmetry
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::validation::numerical_validation::{
//!     ComparisonValidator, NumericalValidator, NumericalValidationConfig,
//! };
//!
//! let validator = ComparisonValidator::new("sin_approximation");
//! let config = NumericalValidationConfig::default();
//! let inputs: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
//! let report = validator.run_validation(
//!     &|x| x.sin(),          // computed
//!     &|x| x.sin(),          // reference (identity)
//!     &inputs,
//!     &config,
//! );
//! assert!(report.all_passed());
//! ```

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// ValidationOutcome
// ---------------------------------------------------------------------------

/// Result of a single validation check at one input point.
#[derive(Debug, Clone)]
pub struct ValidationOutcome {
    /// Whether the validation passed.
    pub passed: bool,
    /// Input value that was tested.
    pub input: f64,
    /// Expected (reference) value.
    pub expected: f64,
    /// Computed value.
    pub computed: f64,
    /// Absolute error |computed − expected|.
    pub abs_error: f64,
    /// Relative error |computed − expected| / |expected|, or `None` when
    /// |expected| ≤ `f64::EPSILON`.
    pub rel_error: Option<f64>,
}

impl ValidationOutcome {
    /// Create a [`ValidationOutcome`] by comparing `computed` against `expected`
    /// using the supplied tolerances.
    ///
    /// Passing criterion (either condition is sufficient):
    /// - `abs_error ≤ tol_abs`
    /// - `rel_error ≤ tol_rel` (when the relative error is defined)
    pub fn new(input: f64, expected: f64, computed: f64, tol_rel: f64, tol_abs: f64) -> Self {
        let abs_error = (computed - expected).abs();
        let rel_error = if expected.abs() > f64::EPSILON {
            Some(abs_error / expected.abs())
        } else {
            None
        };
        let passed = abs_error <= tol_abs || rel_error.is_some_and(|r| r <= tol_rel);
        Self {
            passed,
            input,
            expected,
            computed,
            abs_error,
            rel_error,
        }
    }
}

impl fmt::Display for ValidationOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rel = self
            .rel_error
            .map_or_else(|| "N/A".to_string(), |r| format!("{r:.3e}"));
        write!(
            f,
            "x={:.6e} | expected={:.6e} | computed={:.6e} | abs_err={:.3e} | rel_err={} | {}",
            self.input,
            self.expected,
            self.computed,
            self.abs_error,
            rel,
            if self.passed { "PASS" } else { "FAIL" }
        )
    }
}

// ---------------------------------------------------------------------------
// NumericalValidationReport
// ---------------------------------------------------------------------------

/// Summary report produced by a [`NumericalValidator`] after processing all
/// test points.
#[derive(Debug, Clone)]
pub struct NumericalValidationReport {
    /// Name of the function or method under test.
    pub function_name: String,
    /// Total number of checks performed.
    pub total_checks: usize,
    /// Number of checks that passed.
    pub passed_checks: usize,
    /// All failing [`ValidationOutcome`]s (empty when all checks pass).
    pub failures: Vec<ValidationOutcome>,
    /// Wall-clock time in nanoseconds for the entire validation run.
    pub timing_ns: u64,
    /// Maximum absolute error observed across all test points.
    pub max_abs_error: f64,
    /// Maximum relative error observed (0.0 when none are defined).
    pub max_rel_error: f64,
}

impl NumericalValidationReport {
    /// Pass rate in [0, 1].  Returns 1.0 when `total_checks == 0`.
    pub fn pass_rate(&self) -> f64 {
        if self.total_checks == 0 {
            1.0
        } else {
            self.passed_checks as f64 / self.total_checks as f64
        }
    }

    /// Returns `true` when there are no failures.
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Render the report as a Markdown section with a summary table and, when
    /// failures exist, a detail table listing each failed check.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!(
            "## Validation Report: `{}`\n\n",
            self.function_name
        ));
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Total checks | {} |\n", self.total_checks));
        md.push_str(&format!("| Passed checks | {} |\n", self.passed_checks));
        md.push_str(&format!(
            "| Pass rate | {:.1}% |\n",
            self.pass_rate() * 100.0
        ));
        md.push_str(&format!(
            "| Max absolute error | {:.3e} |\n",
            self.max_abs_error
        ));
        md.push_str(&format!(
            "| Max relative error | {:.3e} |\n",
            self.max_rel_error
        ));
        md.push_str(&format!("| Timing | {} ns |\n", self.timing_ns));
        md.push_str(&format!(
            "| Status | {} |\n\n",
            if self.all_passed() {
                "✓ PASS"
            } else {
                "✗ FAIL"
            }
        ));

        if !self.failures.is_empty() {
            md.push_str("### Failures\n\n");
            md.push_str("| Input | Expected | Computed | Abs Error | Rel Error |\n");
            md.push_str("|-------|----------|----------|-----------|-----------|\n");
            for f in &self.failures {
                let rel = f
                    .rel_error
                    .map_or_else(|| "N/A".to_string(), |r| format!("{r:.3e}"));
                md.push_str(&format!(
                    "| {:.6e} | {:.6e} | {:.6e} | {:.3e} | {} |\n",
                    f.input, f.expected, f.computed, f.abs_error, rel
                ));
            }
        }

        md
    }
}

impl fmt::Display for NumericalValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}/{} passed ({:.1}%), max_abs={:.3e}, max_rel={:.3e}",
            self.function_name,
            self.passed_checks,
            self.total_checks,
            self.pass_rate() * 100.0,
            self.max_abs_error,
            self.max_rel_error,
        )
    }
}

// ---------------------------------------------------------------------------
// NumericalValidationConfig
// ---------------------------------------------------------------------------

/// Configuration knobs shared by all validators.
#[derive(Debug, Clone)]
pub struct NumericalValidationConfig {
    /// Relative tolerance for the pass/fail decision.
    pub relative_tolerance: f64,
    /// Absolute tolerance for the pass/fail decision.
    pub absolute_tolerance: f64,
    /// Number of test points to sample when generating grids internally.
    pub num_test_points: usize,
    /// Random seed for reproducible sampling.
    pub seed: u64,
}

impl Default for NumericalValidationConfig {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-10,
            num_test_points: 100,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// NumericalValidator trait
// ---------------------------------------------------------------------------

/// Core trait for numerical validators.
///
/// Implementors compare a *computed* function against a *reference* function
/// over a supplied grid of inputs and produce a [`NumericalValidationReport`].
pub trait NumericalValidator: Send + Sync {
    /// Human-readable name for this validator instance.
    fn name(&self) -> &str;

    /// Validate a single (input, computed, reference) triple.
    fn validate(
        &self,
        input: f64,
        computed: f64,
        reference: f64,
        config: &NumericalValidationConfig,
    ) -> ValidationOutcome;

    /// Sweep over `inputs`, calling `computed_fn` and `reference_fn` at each
    /// point, and aggregate the results into a [`NumericalValidationReport`].
    fn run_validation(
        &self,
        computed_fn: &dyn Fn(f64) -> f64,
        reference_fn: &dyn Fn(f64) -> f64,
        inputs: &[f64],
        config: &NumericalValidationConfig,
    ) -> NumericalValidationReport;
}

// ---------------------------------------------------------------------------
// ComparisonValidator
// ---------------------------------------------------------------------------

/// General-purpose point-wise comparison validator.
///
/// Applies the tolerances from [`NumericalValidationConfig`] at every input,
/// accumulates statistics, and returns a [`NumericalValidationReport`].
#[derive(Debug, Clone)]
pub struct ComparisonValidator {
    name: String,
}

impl ComparisonValidator {
    /// Create a new [`ComparisonValidator`] with the given display name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl NumericalValidator for ComparisonValidator {
    fn name(&self) -> &str {
        &self.name
    }

    fn validate(
        &self,
        input: f64,
        computed: f64,
        reference: f64,
        config: &NumericalValidationConfig,
    ) -> ValidationOutcome {
        ValidationOutcome::new(
            input,
            reference,
            computed,
            config.relative_tolerance,
            config.absolute_tolerance,
        )
    }

    fn run_validation(
        &self,
        computed_fn: &dyn Fn(f64) -> f64,
        reference_fn: &dyn Fn(f64) -> f64,
        inputs: &[f64],
        config: &NumericalValidationConfig,
    ) -> NumericalValidationReport {
        let start = Instant::now();
        let mut outcomes: Vec<ValidationOutcome> = Vec::with_capacity(inputs.len());

        for &x in inputs {
            let computed = computed_fn(x);
            let reference = reference_fn(x);
            outcomes.push(self.validate(x, computed, reference, config));
        }

        let elapsed_ns = start.elapsed().as_nanos() as u64;
        let total = outcomes.len();
        let passed = outcomes.iter().filter(|o| o.passed).count();
        let failures: Vec<ValidationOutcome> =
            outcomes.iter().filter(|o| !o.passed).cloned().collect();

        let max_abs_error = outcomes.iter().map(|o| o.abs_error).fold(0.0_f64, f64::max);
        let max_rel_error = outcomes
            .iter()
            .filter_map(|o| o.rel_error)
            .fold(0.0_f64, f64::max);

        NumericalValidationReport {
            function_name: self.name.clone(),
            total_checks: total,
            passed_checks: passed,
            failures,
            timing_ns: elapsed_ns,
            max_abs_error,
            max_rel_error,
        }
    }
}

// ---------------------------------------------------------------------------
// MonotonicityChecker
// ---------------------------------------------------------------------------

/// Checks that a function is monotonically increasing or decreasing over a
/// sorted grid of inputs.
///
/// A violation is recorded when `f(x[i+1]) < f(x[i]) - tolerance` (increasing
/// mode) or `f(x[i+1]) > f(x[i]) + tolerance` (decreasing mode).
#[derive(Debug, Clone)]
pub struct MonotonicityChecker {
    /// `true` → assert increasing; `false` → assert decreasing.
    pub increasing: bool,
    /// Tolerance for strict/weak monotonicity.  Values within this margin of
    /// their predecessor are not considered violations.
    pub tolerance: f64,
}

impl MonotonicityChecker {
    /// Build a checker that asserts `f` is monotonically increasing.
    pub fn new_increasing(tolerance: f64) -> Self {
        Self {
            increasing: true,
            tolerance,
        }
    }

    /// Build a checker that asserts `f` is monotonically decreasing.
    pub fn new_decreasing(tolerance: f64) -> Self {
        Self {
            increasing: false,
            tolerance,
        }
    }

    /// Returns `true` when `f` is monotone over all consecutive pairs in
    /// `inputs`.  `inputs` need not be pre-sorted; consecutive pairs are
    /// evaluated in the order given.
    pub fn check(&self, f: &dyn Fn(f64) -> f64, inputs: &[f64]) -> bool {
        if inputs.len() < 2 {
            return true;
        }
        let values: Vec<f64> = inputs.iter().map(|&x| f(x)).collect();
        for window in values.windows(2) {
            let (prev, next) = (window[0], window[1]);
            if self.increasing {
                // next should be >= prev − tol
                if next < prev - self.tolerance {
                    return false;
                }
            } else {
                // next should be <= prev + tol
                if next > prev + self.tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Like [`Self::check`], but additionally returns the index of the first
    /// violation (if any).
    pub fn check_with_first_violation(
        &self,
        f: &dyn Fn(f64) -> f64,
        inputs: &[f64],
    ) -> (bool, Option<usize>) {
        if inputs.len() < 2 {
            return (true, None);
        }
        let values: Vec<f64> = inputs.iter().map(|&x| f(x)).collect();
        for (i, window) in values.windows(2).enumerate() {
            let (prev, next) = (window[0], window[1]);
            let violated = if self.increasing {
                next < prev - self.tolerance
            } else {
                next > prev + self.tolerance
            };
            if violated {
                return (false, Some(i));
            }
        }
        (true, None)
    }
}

// ---------------------------------------------------------------------------
// BoundaryChecker
// ---------------------------------------------------------------------------

/// Validates function values at domain boundaries.
///
/// Useful for CDFs (`F(−∞) == 0`, `F(+∞) == 1`), PDFs (`f(0) == 0` for
/// half-bounded domains), and similar mathematical requirements.
#[derive(Debug, Clone)]
pub struct BoundaryChecker {
    /// Display name used in error messages.
    pub name: String,
}

impl BoundaryChecker {
    /// Create a new [`BoundaryChecker`] with the given display name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Check that `f(lower)` matches `expected_lower` and `f(upper)` matches
    /// `expected_upper` within `tolerance`.
    ///
    /// Returns `(lower_ok, upper_ok)`.
    pub fn check_bounds(
        &self,
        f: &dyn Fn(f64) -> f64,
        lower: f64,
        upper: f64,
        expected_lower: f64,
        expected_upper: f64,
        tolerance: f64,
    ) -> (bool, bool) {
        let lower_ok = (f(lower) - expected_lower).abs() <= tolerance;
        let upper_ok = (f(upper) - expected_upper).abs() <= tolerance;
        (lower_ok, upper_ok)
    }

    /// Check a single boundary point.
    pub fn check_point(
        &self,
        f: &dyn Fn(f64) -> f64,
        point: f64,
        expected: f64,
        tolerance: f64,
    ) -> bool {
        (f(point) - expected).abs() <= tolerance
    }

    /// Check a collection of (point, expected) pairs.  Returns the number of
    /// points that satisfied the tolerance.
    pub fn check_multiple(
        &self,
        f: &dyn Fn(f64) -> f64,
        points: &[(f64, f64)],
        tolerance: f64,
    ) -> usize {
        points
            .iter()
            .filter(|&&(x, expected)| (f(x) - expected).abs() <= tolerance)
            .count()
    }
}

// ---------------------------------------------------------------------------
// SymmetryChecker
// ---------------------------------------------------------------------------

/// Validates even or odd function symmetry: `f(−x) == f(x)` (even) or
/// `f(−x) == −f(x)` (odd).
///
/// Only positive inputs need be passed; the checker evaluates both `x` and
/// `−x` internally.
#[derive(Debug, Clone)]
pub struct SymmetryChecker {
    /// `true` → even symmetry `f(−x) == f(x)`;
    /// `false` → odd symmetry `f(−x) == −f(x)`.
    pub even: bool,
    /// Absolute tolerance for the symmetry assertion.
    pub tolerance: f64,
}

impl SymmetryChecker {
    /// Build a checker for even-symmetric functions.
    pub fn new_even(tolerance: f64) -> Self {
        Self {
            even: true,
            tolerance,
        }
    }

    /// Build a checker for odd-symmetric functions.
    pub fn new_odd(tolerance: f64) -> Self {
        Self {
            even: false,
            tolerance,
        }
    }

    /// Returns `true` when `f` satisfies the configured symmetry at every
    /// input in `inputs`.
    ///
    /// `inputs` may contain positive, negative, or zero values.  For each `x`
    /// the checker evaluates both `f(x)` and `f(−x)`.
    pub fn check(&self, f: &dyn Fn(f64) -> f64, inputs: &[f64]) -> bool {
        for &x in inputs {
            let fx = f(x);
            let fnx = f(-x);
            let expected = if self.even { fx } else { -fx };
            if (fnx - expected).abs() > self.tolerance {
                return false;
            }
        }
        true
    }

    /// Like [`Self::check`] but returns the index of the first violation.
    pub fn check_with_first_violation(
        &self,
        f: &dyn Fn(f64) -> f64,
        inputs: &[f64],
    ) -> (bool, Option<usize>) {
        for (i, &x) in inputs.iter().enumerate() {
            let fx = f(x);
            let fnx = f(-x);
            let expected = if self.even { fx } else { -fx };
            if (fnx - expected).abs() > self.tolerance {
                return (false, Some(i));
            }
        }
        (true, None)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // -----------------------------------------------------------------------
    // ValidationOutcome
    // -----------------------------------------------------------------------

    #[test]
    fn test_outcome_pass_within_abs_tolerance() {
        let outcome = ValidationOutcome::new(1.0, 2.0, 2.0 + 1e-11, 1e-6, 1e-10);
        assert!(outcome.passed);
        assert!(outcome.abs_error < 1e-10);
    }

    #[test]
    fn test_outcome_pass_within_rel_tolerance() {
        let outcome = ValidationOutcome::new(1.0, 1000.0, 1000.0 * (1.0 + 5e-7), 1e-6, 1e-10);
        assert!(outcome.passed);
    }

    #[test]
    fn test_outcome_fail() {
        let outcome = ValidationOutcome::new(1.0, 1.0, 2.0, 1e-6, 1e-10);
        assert!(!outcome.passed);
        assert!((outcome.abs_error - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_outcome_rel_error_undefined_near_zero() {
        let outcome = ValidationOutcome::new(0.0, 0.0, 1e-15, 1e-6, 1e-10);
        // expected is 0.0, so rel_error should be None
        assert!(outcome.rel_error.is_none());
    }

    #[test]
    fn test_outcome_display() {
        let outcome = ValidationOutcome::new(0.5, 0.5, 0.5, 1e-6, 1e-10);
        let s = format!("{outcome}");
        assert!(s.contains("PASS"));
    }

    // -----------------------------------------------------------------------
    // NumericalValidationReport
    // -----------------------------------------------------------------------

    #[test]
    fn test_report_pass_rate_empty() {
        let report = NumericalValidationReport {
            function_name: "empty".to_string(),
            total_checks: 0,
            passed_checks: 0,
            failures: vec![],
            timing_ns: 0,
            max_abs_error: 0.0,
            max_rel_error: 0.0,
        };
        assert!((report.pass_rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_report_pass_rate_partial() {
        let report = NumericalValidationReport {
            function_name: "partial".to_string(),
            total_checks: 4,
            passed_checks: 3,
            failures: vec![ValidationOutcome::new(0.0, 1.0, 2.0, 1e-6, 1e-10)],
            timing_ns: 1000,
            max_abs_error: 1.0,
            max_rel_error: 1.0,
        };
        assert!((report.pass_rate() - 0.75).abs() < 1e-12);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_markdown_format() {
        let report = NumericalValidationReport {
            function_name: "cosine".to_string(),
            total_checks: 10,
            passed_checks: 10,
            failures: vec![],
            timing_ns: 12345,
            max_abs_error: 1e-12,
            max_rel_error: 1e-11,
        };
        let md = report.to_markdown();
        assert!(md.contains("## Validation Report: `cosine`"));
        assert!(md.contains("PASS"));
        assert!(md.contains("100.0%"));
    }

    #[test]
    fn test_report_markdown_includes_failure_table() {
        let report = NumericalValidationReport {
            function_name: "broken_fn".to_string(),
            total_checks: 2,
            passed_checks: 1,
            failures: vec![ValidationOutcome::new(
                std::f64::consts::PI,
                0.0,
                1.0,
                1e-6,
                1e-10,
            )],
            timing_ns: 500,
            max_abs_error: 1.0,
            max_rel_error: 0.0,
        };
        let md = report.to_markdown();
        assert!(md.contains("### Failures"));
        assert!(md.contains("FAIL"));
    }

    // -----------------------------------------------------------------------
    // ComparisonValidator
    // -----------------------------------------------------------------------

    #[test]
    fn test_comparison_validator_identity() {
        let v = ComparisonValidator::new("identity");
        let config = NumericalValidationConfig::default();
        let inputs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1 - 1.0).collect();
        let report = v.run_validation(&|x| x, &|x| x, &inputs, &config);
        assert!(report.all_passed(), "Identity should pass: {report}");
        assert_eq!(report.total_checks, inputs.len());
    }

    #[test]
    fn test_comparison_validator_sin_reference() {
        let v = ComparisonValidator::new("sin_vs_itself");
        let config = NumericalValidationConfig::default();
        let inputs: Vec<f64> = (0..50).map(|i| i as f64 * PI / 50.0).collect();
        let report = v.run_validation(&|x| x.sin(), &|x| x.sin(), &inputs, &config);
        assert!(report.all_passed());
    }

    #[test]
    fn test_comparison_validator_detects_errors() {
        let v = ComparisonValidator::new("wrong_fn");
        let config = NumericalValidationConfig {
            relative_tolerance: 1e-3,
            absolute_tolerance: 1e-3,
            ..Default::default()
        };
        let inputs = vec![1.0, 2.0, 3.0];
        // computed deviates by 1.0, which is >> tolerance
        let report = v.run_validation(&|x| x + 1.0, &|x| x, &inputs, &config);
        assert!(!report.all_passed());
        assert_eq!(report.failures.len(), 3);
    }

    #[test]
    fn test_comparison_validator_name() {
        let v = ComparisonValidator::new("my_validator");
        assert_eq!(v.name(), "my_validator");
    }

    // -----------------------------------------------------------------------
    // MonotonicityChecker
    // -----------------------------------------------------------------------

    #[test]
    fn test_monotonicity_increasing_ok() {
        let checker = MonotonicityChecker::new_increasing(1e-12);
        let inputs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        assert!(checker.check(&|x: f64| x.powi(2), &inputs));
    }

    #[test]
    fn test_monotonicity_decreasing_ok() {
        let checker = MonotonicityChecker::new_decreasing(1e-12);
        let inputs: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();
        assert!(checker.check(&|x: f64| -x, &inputs));
    }

    #[test]
    fn test_monotonicity_violation_detected() {
        let checker = MonotonicityChecker::new_increasing(1e-12);
        // x.sin() is not monotone on [0, 2π]
        let inputs: Vec<f64> = (0..=100).map(|i| i as f64 * 2.0 * PI / 100.0).collect();
        let (ok, idx) = checker.check_with_first_violation(&|x: f64| x.sin(), &inputs);
        assert!(!ok);
        assert!(idx.is_some());
    }

    #[test]
    fn test_monotonicity_single_point_always_passes() {
        let checker = MonotonicityChecker::new_increasing(0.0);
        assert!(checker.check(&|x| x, &[1.0]));
    }

    #[test]
    fn test_monotonicity_cdf_like() {
        // A logistic CDF is strictly increasing.
        let checker = MonotonicityChecker::new_increasing(1e-12);
        let logistic = |x: f64| 1.0 / (1.0 + (-x).exp());
        let inputs: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
        assert!(checker.check(&logistic, &inputs));
    }

    // -----------------------------------------------------------------------
    // BoundaryChecker
    // -----------------------------------------------------------------------

    #[test]
    fn test_boundary_check_bounds_pass() {
        let bc = BoundaryChecker::new("logistic_cdf");
        let logistic = |x: f64| 1.0 / (1.0 + (-x).exp());
        let (lo_ok, hi_ok) = bc.check_bounds(&logistic, -100.0, 100.0, 0.0, 1.0, 1e-6);
        assert!(lo_ok);
        assert!(hi_ok);
    }

    #[test]
    fn test_boundary_check_point() {
        let bc = BoundaryChecker::new("identity");
        assert!(bc.check_point(&|x| x, 0.0, 0.0, 1e-12));
        assert!(!bc.check_point(&|x| x + 1.0, 0.0, 0.0, 0.5));
    }

    #[test]
    fn test_boundary_check_multiple() {
        let bc = BoundaryChecker::new("quadratic");
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0), (3.0, 9.0)];
        let ok = bc.check_multiple(&|x| x * x, &points, 1e-12);
        assert_eq!(ok, 4);
    }

    #[test]
    fn test_boundary_fail() {
        let bc = BoundaryChecker::new("bad");
        let (lo_ok, hi_ok) = bc.check_bounds(&|_| 0.5, -1.0, 1.0, 0.0, 1.0, 1e-6);
        assert!(!lo_ok);
        assert!(!hi_ok);
    }

    // -----------------------------------------------------------------------
    // SymmetryChecker
    // -----------------------------------------------------------------------

    #[test]
    fn test_symmetry_even_cosine() {
        let sc = SymmetryChecker::new_even(1e-12);
        let inputs: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        assert!(sc.check(&|x: f64| x.cos(), &inputs));
    }

    #[test]
    fn test_symmetry_odd_sine() {
        let sc = SymmetryChecker::new_odd(1e-12);
        let inputs: Vec<f64> = (1..=20).map(|i| i as f64 * 0.1).collect();
        assert!(sc.check(&|x: f64| x.sin(), &inputs));
    }

    #[test]
    fn test_symmetry_even_violation() {
        let sc = SymmetryChecker::new_even(1e-12);
        // x.sin() is odd, not even → should fail
        let inputs = vec![0.5, 1.0];
        assert!(!sc.check(&|x: f64| x.sin(), &inputs));
    }

    #[test]
    fn test_symmetry_odd_violation() {
        let sc = SymmetryChecker::new_odd(1e-12);
        // x.cos() is even, not odd → should fail
        let inputs = vec![0.5, 1.0];
        assert!(!sc.check(&|x: f64| x.cos(), &inputs));
    }

    #[test]
    fn test_symmetry_with_first_violation() {
        let sc = SymmetryChecker::new_even(1e-12);
        let (ok, idx) = sc.check_with_first_violation(&|x: f64| x.sin(), &[0.5, 1.0]);
        assert!(!ok);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn test_symmetry_empty_input() {
        let sc = SymmetryChecker::new_even(1e-12);
        assert!(sc.check(&|x| x, &[]));
    }

    // -----------------------------------------------------------------------
    // NumericalValidationConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_default_values() {
        let cfg = NumericalValidationConfig::default();
        assert_eq!(cfg.relative_tolerance, 1e-6);
        assert_eq!(cfg.absolute_tolerance, 1e-10);
        assert_eq!(cfg.num_test_points, 100);
        assert_eq!(cfg.seed, 42);
    }
}
