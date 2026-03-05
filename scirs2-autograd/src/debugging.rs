//! Gradient debugging utilities for computation graphs
//!
//! This module provides three independent facilities:
//!
//! 1. [`GradientChecker`] — numerical gradient verification via finite differences
//! 2. [`NanDetector`]     — detect NaN / Inf values in tensors
//! 3. [`GradientMonitor`] — track gradient statistics across training steps
//!
//! # Examples
//!
//! ## Gradient checking
//!
//! ```rust
//! use scirs2_autograd::debugging::GradientChecker;
//!
//! // f(x) = x^2, analytic gradient = 2x
//! let x = vec![1.0_f64, 2.0, 3.0];
//! let analytic = vec![2.0, 4.0, 6.0];
//!
//! let result = GradientChecker::check_gradients(
//!     |v: &[f64]| v.iter().map(|xi| xi * xi).sum::<f64>(),
//!     &x,
//!     &analytic,
//!     1e-5,
//!     1e-4,
//! );
//! assert!(result.passed, "gradient check failed: {:?}", result);
//! ```
//!
//! ## NaN detection
//!
//! ```rust
//! use scirs2_autograd::debugging::NanDetector;
//!
//! let clean = vec![1.0_f64, 2.0, 3.0];
//! assert!(NanDetector::check_slice(&clean, "layer1").is_ok());
//!
//! let dirty = vec![1.0_f64, f64::NAN, 3.0];
//! assert!(NanDetector::check_slice(&dirty, "layer2").is_err());
//! ```
//!
//! ## Gradient monitoring
//!
//! ```rust
//! use scirs2_autograd::debugging::GradientMonitor;
//!
//! let mut monitor = GradientMonitor::new();
//! monitor.record_gradients("layer1", &[0.1, 0.2, 0.3]);
//! monitor.record_gradients("layer2", &[100.0, 200.0]);
//!
//! let vanishing = monitor.detect_vanishing_gradients(0.01);
//! let exploding = monitor.detect_exploding_gradients(10.0);
//! assert!(exploding.contains(&"layer2".to_string()));
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// GradientChecker
// ─────────────────────────────────────────────────────────────────────────────

/// Result of a gradient check.
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Maximum absolute error across all components
    pub max_error: f64,
    /// Mean absolute error across all components
    pub mean_error: f64,
    /// True if all components are within `tol`
    pub passed: bool,
    /// Indices where the error exceeded the tolerance
    pub failures: Vec<usize>,
}

impl std::fmt::Display for GradCheckResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradCheckResult {{ max_error: {:.2e}, mean_error: {:.2e}, passed: {}, failures: {} }}",
            self.max_error,
            self.mean_error,
            self.passed,
            self.failures.len()
        )
    }
}

/// Numerical gradient verifier using symmetric finite differences.
///
/// For each component `i` of the input vector `x`, the numerical gradient is:
///
/// ```text
/// g_i = (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)
/// ```
///
/// where `e_i` is the `i`-th standard basis vector.
pub struct GradientChecker;

impl GradientChecker {
    /// Verify analytic gradients against numerical finite-difference estimates.
    ///
    /// # Arguments
    /// * `f`             - Scalar function `f: &[f64] -> f64`
    /// * `x`             - Point at which to check gradients
    /// * `analytic_grads`- Analytic gradient values at `x`
    /// * `eps`           - Finite-difference step size (e.g. `1e-5`)
    /// * `tol`           - Acceptable absolute error per component (e.g. `1e-4`)
    ///
    /// # Panics
    /// Does not panic — all errors are returned in `GradCheckResult`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_autograd::debugging::GradientChecker;
    ///
    /// let x = vec![1.0_f64, 2.0, -1.0];
    /// // f(x) = sum(x_i^3), gradient = 3*x_i^2
    /// let analytic = vec![3.0, 12.0, 3.0];
    /// let result = GradientChecker::check_gradients(
    ///     |v: &[f64]| v.iter().map(|xi| xi.powi(3)).sum::<f64>(),
    ///     &x,
    ///     &analytic,
    ///     1e-5,
    ///     1e-3,
    /// );
    /// assert!(result.passed, "{}", result);
    /// ```
    pub fn check_gradients<Func>(
        f: Func,
        x: &[f64],
        analytic_grads: &[f64],
        eps: f64,
        tol: f64,
    ) -> GradCheckResult
    where
        Func: Fn(&[f64]) -> f64,
    {
        assert_eq!(
            x.len(),
            analytic_grads.len(),
            "x and analytic_grads must have the same length"
        );

        let n = x.len();
        let mut max_error = 0.0_f64;
        let mut sum_error = 0.0_f64;
        let mut failures = Vec::new();

        for i in 0..n {
            let numerical = Self::finite_diff(&f, x, i, eps);
            let error = (numerical - analytic_grads[i]).abs();
            sum_error += error;
            if error > max_error {
                max_error = error;
            }
            if error > tol {
                failures.push(i);
            }
        }

        let mean_error = if n > 0 { sum_error / n as f64 } else { 0.0 };
        let passed = failures.is_empty();

        GradCheckResult {
            max_error,
            mean_error,
            passed,
            failures,
        }
    }

    /// Compute the symmetric finite-difference gradient for component `i`.
    fn finite_diff<Func>(f: &Func, x: &[f64], i: usize, eps: f64) -> f64
    where
        Func: Fn(&[f64]) -> f64,
    {
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] += eps;
        x_minus[i] -= eps;
        (f(&x_plus) - f(&x_minus)) / (2.0 * eps)
    }

    /// Check gradients for a function that takes a matrix (stored row-major).
    ///
    /// This is a convenience wrapper around [`check_gradients`] that flattens
    /// the matrix input and works element-by-element.
    ///
    /// # Arguments
    /// * `f`             - Function taking a flat (row-major) slice representing
    ///                     the matrix of shape `(rows, cols)`
    /// * `x`             - Flat matrix values at which to check
    /// * `analytic_grads`- Flat analytic gradient matrix
    /// * `eps`           - Step size
    /// * `tol`           - Tolerance per element
    pub fn check_matrix_gradients<Func>(
        f: Func,
        x: &[f64],
        analytic_grads: &[f64],
        eps: f64,
        tol: f64,
    ) -> GradCheckResult
    where
        Func: Fn(&[f64]) -> f64,
    {
        Self::check_gradients(f, x, analytic_grads, eps, tol)
    }

    /// Compute the full Jacobian of a vector-to-vector function via finite differences.
    ///
    /// Returns a `(m, n)` matrix stored row-major where `m = f(x).len()` and
    /// `n = x.len()`.
    ///
    /// # Arguments
    /// * `f`   - Vector-to-vector function
    /// * `x`   - Input vector
    /// * `eps` - Finite-difference step size
    pub fn numerical_jacobian<Func>(f: &Func, x: &[f64], eps: f64) -> Vec<Vec<f64>>
    where
        Func: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        let m = f(x).len();
        let mut jac = vec![vec![0.0_f64; n]; m];

        for i in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let fplus = f(&x_plus);
            let fminus = f(&x_minus);
            for j in 0..m {
                let fplus_j = fplus.get(j).copied().unwrap_or(0.0);
                let fminus_j = fminus.get(j).copied().unwrap_or(0.0);
                jac[j][i] = (fplus_j - fminus_j) / (2.0 * eps);
            }
        }

        jac
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NanDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Error produced by [`NanDetector`] when NaN or Inf values are found.
#[derive(Debug, Clone)]
pub struct NanError {
    /// Name of the layer or tensor where the issue was found
    pub layer_name: String,
    /// Number of NaN values
    pub nan_count: usize,
    /// Number of +/-Inf values
    pub inf_count: usize,
}

impl std::fmt::Display for NanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NanError in '{}': {} NaN(s), {} Inf(s)",
            self.layer_name, self.nan_count, self.inf_count
        )
    }
}

impl std::error::Error for NanError {}

/// Detector for NaN and Inf values in forward/backward passes.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::debugging::NanDetector;
///
/// let clean = vec![0.0_f64, 1.0, -1.0];
/// NanDetector::check_slice(&clean, "fc1.weight").expect("should be clean");
///
/// let bad = vec![1.0_f64, f64::INFINITY, 3.0];
/// let err = NanDetector::check_slice(&bad, "fc1.bias").unwrap_err();
/// assert_eq!(err.inf_count, 1);
/// ```
pub struct NanDetector;

impl NanDetector {
    /// Check a flat slice of `f64` values for NaN or Inf.
    ///
    /// Returns `Ok(())` when no anomalies are found, or `Err(NanError)` with
    /// counts of NaN and Inf values.
    pub fn check_slice(values: &[f64], name: &str) -> Result<(), NanError> {
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        for &v in values {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            }
        }
        if nan_count > 0 || inf_count > 0 {
            Err(NanError {
                layer_name: name.to_string(),
                nan_count,
                inf_count,
            })
        } else {
            Ok(())
        }
    }

    /// Check a flat slice of `f32` values for NaN or Inf.
    pub fn check_slice_f32(values: &[f32], name: &str) -> Result<(), NanError> {
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        for &v in values {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            }
        }
        if nan_count > 0 || inf_count > 0 {
            Err(NanError {
                layer_name: name.to_string(),
                nan_count,
                inf_count,
            })
        } else {
            Ok(())
        }
    }

    /// Check a 2-D matrix (stored row-major) for NaN or Inf.
    ///
    /// # Arguments
    /// * `matrix` - Row-major matrix data
    /// * `name`   - Descriptive name for error reporting
    pub fn check_matrix(matrix: &[Vec<f64>], name: &str) -> Result<(), NanError> {
        let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        Self::check_slice(&flat, name)
    }

    /// Returns true if the slice contains any NaN value.
    pub fn has_nan(values: &[f64]) -> bool {
        values.iter().any(|v| v.is_nan())
    }

    /// Returns true if the slice contains any Inf value.
    pub fn has_inf(values: &[f64]) -> bool {
        values.iter().any(|v| v.is_infinite())
    }

    /// Count NaN and Inf values, returning `(nan_count, inf_count)`.
    pub fn count_anomalies(values: &[f64]) -> (usize, usize) {
        values.iter().fold((0, 0), |(nans, infs), &v| {
            if v.is_nan() {
                (nans + 1, infs)
            } else if v.is_infinite() {
                (nans, infs + 1)
            } else {
                (nans, infs)
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GradientMonitor
// ─────────────────────────────────────────────────────────────────────────────

/// Per-layer gradient statistics accumulated over one or more recording calls.
#[derive(Debug, Clone)]
struct LayerGradStats {
    /// Running sum of L2 norms (one per `record_gradients` call)
    norm_sum: f64,
    /// Running sum of squared L2 norms
    norm_sq_sum: f64,
    /// Maximum L2 norm observed
    norm_max: f64,
    /// Number of recording calls
    call_count: usize,
}

impl LayerGradStats {
    fn new() -> Self {
        Self {
            norm_sum: 0.0,
            norm_sq_sum: 0.0,
            norm_max: 0.0,
            call_count: 0,
        }
    }

    fn update(&mut self, norm: f64) {
        self.norm_sum += norm;
        self.norm_sq_sum += norm * norm;
        if norm > self.norm_max {
            self.norm_max = norm;
        }
        self.call_count += 1;
    }

    fn mean_norm(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.norm_sum / self.call_count as f64
        }
    }

    fn max_norm(&self) -> f64 {
        self.norm_max
    }
}

/// Gradient statistics monitor for detecting vanishing and exploding gradients.
///
/// Records gradient L2 norms for each named layer across multiple training steps
/// and provides diagnostic reports.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::debugging::GradientMonitor;
///
/// let mut monitor = GradientMonitor::new();
///
/// // Simulate a training step
/// monitor.record_gradients("encoder.w1", &[0.001, 0.0005, 0.002]);
/// monitor.record_gradients("encoder.w2", &[0.0001, 0.0001]);
/// monitor.record_gradients("decoder.w1", &[50.0, 100.0, 75.0]);
///
/// // Check for issues
/// let vanishing = monitor.detect_vanishing_gradients(0.01);
/// assert!(!vanishing.is_empty());
///
/// let exploding = monitor.detect_exploding_gradients(10.0);
/// assert!(exploding.contains(&"decoder.w1".to_string()));
///
/// // Full report
/// let report = monitor.report();
/// assert_eq!(report.len(), 3);
/// ```
#[derive(Debug, Default)]
pub struct GradientMonitor {
    stats: HashMap<String, LayerGradStats>,
    /// Insertion order for deterministic report output
    layer_order: Vec<String>,
}

impl GradientMonitor {
    /// Create a new, empty gradient monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record gradients for a named layer.
    ///
    /// Computes the L2 norm of `grads` and updates running statistics for
    /// `layer_name`.
    ///
    /// # Arguments
    /// * `layer_name` - A descriptive name for the layer (e.g. `"fc1.weight"`)
    /// * `grads`      - Flat gradient values (may be the gradient of any tensor)
    pub fn record_gradients(&mut self, layer_name: &str, grads: &[f64]) {
        let norm = l2_norm(grads);
        let entry = self
            .stats
            .entry(layer_name.to_string())
            .or_insert_with(LayerGradStats::new);
        if entry.call_count == 0 {
            self.layer_order.push(layer_name.to_string());
        }
        entry.update(norm);
    }

    /// Record gradients from `f32` values (converted to f64 internally).
    pub fn record_gradients_f32(&mut self, layer_name: &str, grads: &[f32]) {
        let f64_grads: Vec<f64> = grads.iter().map(|&g| g as f64).collect();
        self.record_gradients(layer_name, &f64_grads);
    }

    /// Return a summary of gradient statistics per layer.
    ///
    /// Each entry is `(layer_name, mean_norm, max_norm)`.
    pub fn report(&self) -> Vec<(String, f64, f64)> {
        self.layer_order
            .iter()
            .filter_map(|name| {
                self.stats.get(name).map(|s| {
                    (name.clone(), s.mean_norm(), s.max_norm())
                })
            })
            .collect()
    }

    /// Return layers whose mean gradient norm is below `threshold`.
    ///
    /// This indicates potential vanishing gradients that may cause slow or
    /// stalled training.
    ///
    /// # Arguments
    /// * `threshold` - Norm threshold below which a layer is considered vanishing
    pub fn detect_vanishing_gradients(&self, threshold: f64) -> Vec<String> {
        self.layer_order
            .iter()
            .filter(|name| {
                self.stats
                    .get(*name)
                    .map(|s| s.mean_norm() < threshold)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Return layers whose max gradient norm exceeds `threshold`.
    ///
    /// This indicates potential exploding gradients that may destabilise
    /// training.
    ///
    /// # Arguments
    /// * `threshold` - Norm threshold above which a layer is considered exploding
    pub fn detect_exploding_gradients(&self, threshold: f64) -> Vec<String> {
        self.layer_order
            .iter()
            .filter(|name| {
                self.stats
                    .get(*name)
                    .map(|s| s.max_norm() > threshold)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.stats.clear();
        self.layer_order.clear();
    }

    /// Return the number of layers being tracked.
    pub fn layer_count(&self) -> usize {
        self.stats.len()
    }

    /// Check if the monitor has any recorded data.
    pub fn is_empty(&self) -> bool {
        self.stats.is_empty()
    }

    /// Return the mean gradient norm for a specific layer.
    ///
    /// Returns `None` if the layer has not been recorded.
    pub fn mean_norm_for(&self, layer_name: &str) -> Option<f64> {
        self.stats.get(layer_name).map(|s| s.mean_norm())
    }

    /// Return the maximum gradient norm for a specific layer.
    ///
    /// Returns `None` if the layer has not been recorded.
    pub fn max_norm_for(&self, layer_name: &str) -> Option<f64> {
        self.stats.get(layer_name).map(|s| s.max_norm())
    }

    /// Format a human-readable report string.
    ///
    /// # Example
    ///
    /// ```rust
    /// use scirs2_autograd::debugging::GradientMonitor;
    /// let mut m = GradientMonitor::new();
    /// m.record_gradients("l1", &[0.1, 0.2]);
    /// let txt = m.format_report();
    /// assert!(txt.contains("l1"));
    /// ```
    pub fn format_report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let sep = "─".repeat(60);
        let _ = writeln!(out, "{sep}");
        let _ = writeln!(
            out,
            " {:<30} {:>12} {:>12}",
            "Layer", "Mean Norm", "Max Norm"
        );
        let _ = writeln!(out, "{sep}");
        for (name, mean, max) in self.report() {
            let _ = writeln!(out, " {:<30} {:>12.4e} {:>12.4e}", name, mean, max);
        }
        let _ = writeln!(out, "{sep}");
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GradientChecker ───────────────────────────────────────────────────────

    #[test]
    fn test_check_gradients_quadratic_passes() {
        // f(x) = sum(x_i^2), grad = 2*x
        let x = vec![1.0_f64, 2.0, 3.0];
        let analytic = vec![2.0, 4.0, 6.0];
        let result = GradientChecker::check_gradients(
            |v| v.iter().map(|xi| xi * xi).sum::<f64>(),
            &x,
            &analytic,
            1e-5,
            1e-4,
        );
        assert!(result.passed, "Expected pass: {result}");
        assert!(result.max_error < 1e-4, "max_error too large: {}", result.max_error);
        assert!(result.failures.is_empty());
    }

    #[test]
    fn test_check_gradients_wrong_grad_fails() {
        let x = vec![1.0_f64, 2.0, 3.0];
        // Deliberately wrong analytic gradients
        let analytic = vec![999.0, 999.0, 999.0];
        let result = GradientChecker::check_gradients(
            |v| v.iter().map(|xi| xi * xi).sum::<f64>(),
            &x,
            &analytic,
            1e-5,
            1e-4,
        );
        assert!(!result.passed, "Expected failure");
        assert_eq!(result.failures.len(), 3);
    }

    #[test]
    fn test_check_gradients_cubic() {
        // f(x) = sum(x_i^3), grad = 3*x_i^2
        let x = vec![1.0_f64, -1.0, 2.0];
        let analytic: Vec<f64> = x.iter().map(|&xi| 3.0 * xi * xi).collect();
        let result = GradientChecker::check_gradients(
            |v| v.iter().map(|xi| xi.powi(3)).sum::<f64>(),
            &x,
            &analytic,
            1e-5,
            1e-3,
        );
        assert!(result.passed, "{result}");
    }

    #[test]
    fn test_numerical_jacobian_identity() {
        // f(x) = x, J = I
        let x = vec![1.0_f64, 2.0, 3.0];
        let jac = GradientChecker::numerical_jacobian(
            &|v: &[f64]| v.to_vec(),
            &x,
            1e-5,
        );
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (jac[i][j] - expected).abs() < 1e-4,
                    "J[{i}][{j}] = {} ≠ {expected}",
                    jac[i][j]
                );
            }
        }
    }

    #[test]
    fn test_numerical_jacobian_affine() {
        // f(x) = [x0 + 2*x1, 3*x0], J = [[1, 2], [3, 0]]
        let x = vec![0.5_f64, 1.5];
        let jac = GradientChecker::numerical_jacobian(
            &|v: &[f64]| vec![v[0] + 2.0 * v[1], 3.0 * v[0]],
            &x,
            1e-5,
        );
        assert!((jac[0][0] - 1.0).abs() < 1e-4, "J[0][0]");
        assert!((jac[0][1] - 2.0).abs() < 1e-4, "J[0][1]");
        assert!((jac[1][0] - 3.0).abs() < 1e-4, "J[1][0]");
        assert!((jac[1][1] - 0.0).abs() < 1e-4, "J[1][1]");
    }

    // ── NanDetector ───────────────────────────────────────────────────────────

    #[test]
    fn test_check_slice_clean() {
        let v = vec![0.0_f64, 1.0, -1.0, 1e10, -1e10];
        NanDetector::check_slice(&v, "clean").expect("should be clean");
    }

    #[test]
    fn test_check_slice_nan() {
        let v = vec![1.0_f64, f64::NAN, 3.0];
        let err = NanDetector::check_slice(&v, "layer").unwrap_err();
        assert_eq!(err.nan_count, 1);
        assert_eq!(err.inf_count, 0);
        assert_eq!(err.layer_name, "layer");
    }

    #[test]
    fn test_check_slice_inf() {
        let v = vec![1.0_f64, f64::INFINITY, -f64::INFINITY];
        let err = NanDetector::check_slice(&v, "layer").unwrap_err();
        assert_eq!(err.nan_count, 0);
        assert_eq!(err.inf_count, 2);
    }

    #[test]
    fn test_check_slice_both() {
        let v = vec![f64::NAN, f64::INFINITY, 1.0, f64::NAN];
        let err = NanDetector::check_slice(&v, "l").unwrap_err();
        assert_eq!(err.nan_count, 2);
        assert_eq!(err.inf_count, 1);
    }

    #[test]
    fn test_check_slice_f32_nan() {
        let v = vec![1.0_f32, f32::NAN];
        let err = NanDetector::check_slice_f32(&v, "layer_f32").unwrap_err();
        assert_eq!(err.nan_count, 1);
    }

    #[test]
    fn test_has_nan_true() {
        assert!(NanDetector::has_nan(&[1.0, f64::NAN]));
    }

    #[test]
    fn test_has_nan_false() {
        assert!(!NanDetector::has_nan(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_has_inf_true() {
        assert!(NanDetector::has_inf(&[1.0, f64::INFINITY]));
    }

    #[test]
    fn test_count_anomalies() {
        let (nans, infs) = NanDetector::count_anomalies(&[
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NAN,
            -f64::INFINITY,
        ]);
        assert_eq!(nans, 2);
        assert_eq!(infs, 2);
    }

    #[test]
    fn test_check_matrix_clean() {
        let m = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        NanDetector::check_matrix(&m, "mat").expect("should be clean");
    }

    #[test]
    fn test_check_matrix_nan() {
        let m = vec![vec![1.0, f64::NAN], vec![3.0, 4.0]];
        let err = NanDetector::check_matrix(&m, "mat").unwrap_err();
        assert_eq!(err.nan_count, 1);
    }

    #[test]
    fn test_nan_error_display() {
        let e = NanError {
            layer_name: "fc1".to_string(),
            nan_count: 2,
            inf_count: 1,
        };
        let s = e.to_string();
        assert!(s.contains("fc1"));
        assert!(s.contains('2'));
        assert!(s.contains('1'));
    }

    // ── GradientMonitor ───────────────────────────────────────────────────────

    #[test]
    fn test_monitor_record_and_report() {
        let mut m = GradientMonitor::new();
        m.record_gradients("l1", &[1.0, 0.0, 0.0]); // norm = 1
        m.record_gradients("l2", &[3.0, 4.0]);       // norm = 5
        let report = m.report();
        assert_eq!(report.len(), 2);
        let names: Vec<&str> = report.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(names, &["l1", "l2"]);
    }

    #[test]
    fn test_monitor_mean_norm() {
        let mut m = GradientMonitor::new();
        m.record_gradients("l1", &[3.0, 4.0]); // norm = 5
        m.record_gradients("l1", &[0.0, 0.0]); // norm = 0
        // mean = (5 + 0) / 2 = 2.5
        let mean = m.mean_norm_for("l1").expect("should exist");
        assert!((mean - 2.5).abs() < 1e-10, "mean = {mean}");
    }

    #[test]
    fn test_monitor_max_norm() {
        let mut m = GradientMonitor::new();
        m.record_gradients("l1", &[1.0]);
        m.record_gradients("l1", &[10.0]);
        m.record_gradients("l1", &[5.0]);
        let max = m.max_norm_for("l1").expect("should exist");
        assert!((max - 10.0).abs() < 1e-10, "max = {max}");
    }

    #[test]
    fn test_detect_vanishing() {
        let mut m = GradientMonitor::new();
        m.record_gradients("big", &[1.0, 1.0]);     // norm ≈ 1.41
        m.record_gradients("small", &[0.0001, 0.0]); // norm ≈ 1e-4
        let v = m.detect_vanishing_gradients(0.01);
        assert!(v.contains(&"small".to_string()));
        assert!(!v.contains(&"big".to_string()));
    }

    #[test]
    fn test_detect_exploding() {
        let mut m = GradientMonitor::new();
        m.record_gradients("normal", &[1.0, 1.0]);      // norm ≈ 1.41
        m.record_gradients("exploding", &[100.0, 0.0]); // norm = 100
        let e = m.detect_exploding_gradients(10.0);
        assert!(e.contains(&"exploding".to_string()));
        assert!(!e.contains(&"normal".to_string()));
    }

    #[test]
    fn test_monitor_reset() {
        let mut m = GradientMonitor::new();
        m.record_gradients("l1", &[1.0, 2.0]);
        assert!(!m.is_empty());
        m.reset();
        assert!(m.is_empty());
        assert_eq!(m.layer_count(), 0);
    }

    #[test]
    fn test_monitor_format_report() {
        let mut m = GradientMonitor::new();
        m.record_gradients("layer_a", &[0.1, 0.2]);
        let text = m.format_report();
        assert!(text.contains("layer_a"));
        assert!(text.contains("Mean Norm"));
    }

    #[test]
    fn test_monitor_record_f32() {
        let mut m = GradientMonitor::new();
        m.record_gradients_f32("l1", &[3.0_f32, 4.0_f32]); // norm = 5
        let max = m.max_norm_for("l1").expect("should exist");
        assert!((max - 5.0).abs() < 1e-5, "max = {max}");
    }

    #[test]
    fn test_monitor_nonexistent_layer() {
        let m = GradientMonitor::new();
        assert!(m.mean_norm_for("does_not_exist").is_none());
        assert!(m.max_norm_for("does_not_exist").is_none());
    }

    #[test]
    fn test_grad_check_result_display() {
        let r = GradCheckResult {
            max_error: 1.23e-5,
            mean_error: 4.56e-6,
            passed: true,
            failures: vec![],
        };
        let s = r.to_string();
        assert!(s.contains("passed: true"));
    }
}
