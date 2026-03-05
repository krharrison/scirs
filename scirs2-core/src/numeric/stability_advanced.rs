//! # Numerical Stability Toolkit — Advanced Extensions
//!
//! This module extends the core stability toolkit with higher-accuracy algorithms
//! that are essential for numerically sensitive computations.
//!
//! ## Algorithms Provided
//!
//! | Function/Type            | Algorithm                                     |
//! |--------------------------|-----------------------------------------------|
//! | [`compensated_dot`]      | Ogita-Rump-Oishi compensated dot product      |
//! | [`accurate_sum`]         | Neumaier (improved Kahan) summation           |
//! | [`RunningStats`]         | Welford online mean/variance/skewness/kurtosis|
//! | [`poly_eval_horner`]     | Horner's method for polynomial evaluation     |
//! | [`log_sum_exp`]          | Numerically stable log-sum-exp                |
//! | [`softmax_stable`]       | Numerically stable softmax                    |
//!
//! ## References
//!
//! - Ogita, T., Rump, S. M., & Oishi, S. (2005). *Accurate sum and dot product*.
//!   SIAM Journal on Scientific Computing, 26(6), 1955–1988.
//! - Neumaier, A. (1974). Rundungsfehleranalyse einiger Verfahren zur Summen-
//!   accumulation. ZAMM, 54(1), 39–51.
//! - Welford, B. P. (1962). Note on a method for calculating corrected sums of
//!   squares and products. Technometrics, 4(3), 419–420.

use crate::error::{CoreError, CoreResult, ErrorContext};

// ---------------------------------------------------------------------------
// Compensated dot product (Ogita-Rump-Oishi)
// ---------------------------------------------------------------------------

/// Compute `twosum(a, b)`: returns `(s, e)` where `s = fl(a + b)` and `e` is
/// the exact rounding error, i.e. `a + b = s + e` exactly.
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

/// Compute `twoprodfma(a, b)`: returns `(p, e)` where `p = fl(a * b)` and
/// `p + e = a * b` exactly (using fused multiply-add).
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Use fma to get the exact rounding error
    let e = a.mul_add(b, -p);
    (p, e)
}

/// Ogita-Rump-Oishi compensated dot product.
///
/// Computes `Σ a[i] * b[i]` with approximately twice the working precision
/// (≈ 2 ulp relative error instead of the standard `n` ulp bound).
///
/// # Arguments
///
/// * `a` – First slice of `f64` values.
/// * `b` – Second slice of `f64` values (must have the same length as `a`).
///
/// # Errors
///
/// Returns `Err` if `a.len() != b.len()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::compensated_dot;
///
/// let a = [1.0_f64, 2.0, 3.0];
/// let b = [4.0_f64, 5.0, 6.0];
/// let result = compensated_dot(&a, &b).expect("should succeed");
/// assert!((result - 32.0).abs() < 1e-14);
/// ```
pub fn compensated_dot(a: &[f64], b: &[f64]) -> CoreResult<f64> {
    if a.len() != b.len() {
        return Err(CoreError::InvalidArgument(
            ErrorContext::new(format!(
                "compensated_dot: slice lengths must match: a.len()={} b.len()={}",
                a.len(),
                b.len()
            )),
        ));
    }

    let mut s = 0.0_f64;
    let mut c = 0.0_f64; // error accumulator

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let (p, e) = two_prod(ai, bi);
        let (s_new, f) = two_sum(s, p);
        c += e + f;
        s = s_new;
    }

    Ok(s + c)
}

// ---------------------------------------------------------------------------
// Accurate summation (Neumaier)
// ---------------------------------------------------------------------------

/// Neumaier summation (improved Kahan).
///
/// Achieves `O(1)` relative error independent of `n`, even when intermediate
/// partial sums are much larger than the individual terms (a case where
/// standard Kahan summation degrades).
///
/// # Arguments
///
/// * `values` – Slice of values to sum.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::accurate_sum;
///
/// let v = [1.0_f64, 1e15, -1e15, 0.5];
/// let s = accurate_sum(&v);
/// assert!((s - 1.5).abs() < 1e-10);
/// ```
pub fn accurate_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64; // compensation

    for &x in values {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            c += (sum - t) + x;
        } else {
            c += (x - t) + sum;
        }
        sum = t;
    }

    sum + c
}

// ---------------------------------------------------------------------------
// Running statistics (Welford's online algorithm)
// ---------------------------------------------------------------------------

/// Online statistics accumulator using Welford's algorithm.
///
/// Tracks mean, variance, skewness, and kurtosis in a single numerically
/// stable pass over the data. All moments are updated incrementally, so
/// this is suitable for streaming data.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::RunningStats;
///
/// let mut stats = RunningStats::new();
/// for x in [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     stats.push(x);
/// }
/// let mean = stats.mean().expect("should succeed");
/// let var  = stats.variance().expect("should succeed");
/// assert!((mean - 5.0).abs() < 1e-12);
/// assert!((var - 4.0).abs() < 1e-10);  // population variance
/// ```
#[derive(Debug, Clone)]
pub struct RunningStats {
    n: u64,
    mean: f64,
    /// Second central moment accumulator (M2 in Welford notation)
    m2: f64,
    /// Third central moment accumulator
    m3: f64,
    /// Fourth central moment accumulator
    m4: f64,
}

impl RunningStats {
    /// Create a new, empty `RunningStats` accumulator.
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    /// Add a new observation `x` to the running statistics.
    ///
    /// Time complexity: `O(1)` per call.
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// Current sample count.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Running mean, or `None` if no values have been added.
    pub fn mean(&self) -> Option<f64> {
        if self.n == 0 {
            None
        } else {
            Some(self.mean)
        }
    }

    /// Population variance `σ²`, or `None` if fewer than 1 value is present.
    ///
    /// Uses `n` in the denominator (population estimator).
    pub fn variance(&self) -> Option<f64> {
        if self.n < 1 {
            None
        } else {
            Some(self.m2 / self.n as f64)
        }
    }

    /// Sample variance `s²`, or `None` if fewer than 2 values are present.
    ///
    /// Uses `n - 1` in the denominator (Bessel-corrected / unbiased estimator).
    pub fn variance_sample(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / (self.n - 1) as f64)
        }
    }

    /// Population standard deviation, or `None` if fewer than 1 value is present.
    pub fn std_dev(&self) -> Option<f64> {
        self.variance().map(f64::sqrt)
    }

    /// Sample standard deviation, or `None` if fewer than 2 values are present.
    pub fn std_dev_sample(&self) -> Option<f64> {
        self.variance_sample().map(f64::sqrt)
    }

    /// Skewness (standardised third central moment), or `None` if fewer than
    /// 2 values are present or if the variance is zero.
    ///
    /// Returns the population skewness estimator (no Bessel correction).
    pub fn skewness(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let var = self.m2 / self.n as f64;
        if var == 0.0 {
            return None;
        }
        let n = self.n as f64;
        Some((self.m3 / n) / var.powf(1.5))
    }

    /// Excess kurtosis (standardised fourth central moment minus 3), or `None`
    /// if fewer than 2 values are present or if the variance is zero.
    ///
    /// A normal distribution has excess kurtosis of 0.
    pub fn kurtosis(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let var = self.m2 / self.n as f64;
        if var == 0.0 {
            return None;
        }
        let n = self.n as f64;
        Some((self.m4 / n) / (var * var) - 3.0)
    }

    /// Merge another `RunningStats` accumulator into this one.
    ///
    /// Allows parallel accumulation: compute partial statistics on separate
    /// threads, then merge the results.
    pub fn merge(&mut self, other: &RunningStats) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let combined_n = (self.n + other.n) as f64;
        let self_n = self.n as f64;
        let other_n = other.n as f64;
        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta * delta2;
        let delta4 = delta2 * delta2;

        let new_m4 = self.m4
            + other.m4
            + delta4 * self_n * other_n * (self_n * self_n - self_n * other_n + other_n * other_n)
                / (combined_n * combined_n * combined_n)
            + 6.0 * delta2 * (self_n * self_n * other.m2 + other_n * other_n * self.m2)
                / (combined_n * combined_n)
            + 4.0 * delta3 * (self_n * other.m3 - other_n * self.m3) / combined_n;

        let new_m3 = self.m3
            + other.m3
            + delta3 * self_n * other_n * (self_n - other_n) / (combined_n * combined_n)
            + 3.0 * delta * (self_n * other.m2 - other_n * self.m2) / combined_n;

        let new_m2 = self.m2 + other.m2 + delta2 * self_n * other_n / combined_n;
        let new_mean = (self_n * self.mean + other_n * other.mean) / combined_n;

        self.n += other.n;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
    }
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Horner's method for polynomial evaluation
// ---------------------------------------------------------------------------

/// Evaluate a polynomial at `x` using Horner's method.
///
/// Interprets `coeffs` in **descending** order of degree:
/// `coeffs[0]*x^(n-1) + coeffs[1]*x^(n-2) + ... + coeffs[n-1]`.
///
/// This requires only `n-1` multiplications and `n-1` additions (vs. `2n`
/// operations for naive evaluation), and has superior numerical stability.
///
/// # Arguments
///
/// * `coeffs` – Polynomial coefficients in **descending** degree order.
/// * `x`      – Point at which to evaluate.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::poly_eval_horner;
///
/// // p(x) = 2x^2 + 3x + 1, evaluated at x=2 => 2*4 + 3*2 + 1 = 15
/// let result = poly_eval_horner(&[2.0_f64, 3.0, 1.0], 2.0);
/// assert!((result - 15.0).abs() < 1e-14);
/// ```
pub fn poly_eval_horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0_f64, |acc, &c| acc * x + c)
}

// ---------------------------------------------------------------------------
// Numerically stable log-sum-exp
// ---------------------------------------------------------------------------

/// Compute `log(Σ exp(xᵢ))` in a numerically stable manner.
///
/// Uses the standard max-shift trick: `log Σ exp(xᵢ) = m + log Σ exp(xᵢ - m)`
/// where `m = max(xᵢ)`. This prevents both overflow and underflow.
///
/// Returns `f64::NEG_INFINITY` for an empty slice (consistent with the
/// convention that log of an empty sum = log(0) = -∞).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::log_sum_exp;
///
/// let v = [1000.0_f64, 1001.0, 1002.0];
/// let lse = log_sum_exp(&v);
/// // Should not overflow
/// assert!(lse.is_finite());
/// // Approximately 1002 + log(1 + e^-1 + e^-2)
/// assert!((lse - 1002.407606).abs() < 1e-4);
/// ```
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if max.is_infinite() {
        return max;
    }

    let sum_exp: f64 = values.iter().map(|&x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

// ---------------------------------------------------------------------------
// Numerically stable softmax
// ---------------------------------------------------------------------------

/// Compute the softmax of `values` in a numerically stable manner.
///
/// Uses the max-shift trick to prevent overflow: `softmax(x)ᵢ = exp(xᵢ - m) / Σ exp(xⱼ - m)`
/// where `m = max(xⱼ)`.
///
/// Returns an empty `Vec` for an empty input.
///
/// # Arguments
///
/// * `values` – Input logits.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::stability_advanced::softmax_stable;
///
/// let v = [1000.0_f64, 1001.0, 1002.0];
/// let s = softmax_stable(&v);
/// // Sum must be 1
/// let total: f64 = s.iter().sum();
/// assert!((total - 1.0).abs() < 1e-14);
/// // Last element must be the largest
/// assert!(s[2] > s[1] && s[1] > s[0]);
/// ```
pub fn softmax_stable(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = values.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();

    if sum == 0.0 {
        // Degenerate case: all values are -infinity
        let n = values.len();
        return vec![1.0 / n as f64; n];
    }

    exps.into_iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- compensated_dot ---

    #[test]
    fn test_compensated_dot_basic() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let result = compensated_dot(&a, &b).expect("should succeed");
        assert!((result - 32.0).abs() < 1e-14);
    }

    #[test]
    fn test_compensated_dot_mismatched_lengths() {
        let a = [1.0_f64, 2.0];
        let b = [3.0_f64];
        assert!(compensated_dot(&a, &b).is_err());
    }

    #[test]
    fn test_compensated_dot_empty() {
        let result = compensated_dot(&[], &[]).expect("empty dot should succeed");
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_compensated_dot_cancellation() {
        // Classic ill-conditioned case: nearly cancelling values
        let a: Vec<f64> = vec![1e8, -1e8 + 1.0];
        let b: Vec<f64> = vec![1.0, 1.0];
        let result = compensated_dot(&a, &b).expect("should succeed");
        // Exact answer: 1e8 + (-1e8 + 1) = 1
        assert!((result - 1.0).abs() < 1e-6);
    }

    // --- accurate_sum ---

    #[test]
    fn test_accurate_sum_basic() {
        let v = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!((accurate_sum(&v) - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_accurate_sum_cancellation() {
        let v = [1.0_f64, 1e15, -1e15, 0.5];
        let s = accurate_sum(&v);
        assert!((s - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_accurate_sum_empty() {
        assert_eq!(accurate_sum(&[]), 0.0);
    }

    // --- RunningStats ---

    #[test]
    fn test_running_stats_mean_variance() {
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.push(x);
        }
        let mean = stats.mean().expect("mean should exist");
        let var = stats.variance().expect("variance should exist");
        assert!((mean - 5.0).abs() < 1e-12, "mean={mean}");
        assert!((var - 4.0).abs() < 1e-10, "pop var={var}");
    }

    #[test]
    fn test_running_stats_empty() {
        let stats = RunningStats::new();
        assert!(stats.mean().is_none());
        assert!(stats.variance().is_none());
        assert!(stats.skewness().is_none());
        assert!(stats.kurtosis().is_none());
    }

    #[test]
    fn test_running_stats_sample_variance() {
        let mut stats = RunningStats::new();
        for &x in &[2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.push(x);
        }
        // Sample variance = M2 / (n-1) = 32 / 7 ≈ 4.571
        let s_var = stats.variance_sample().expect("sample variance should exist");
        assert!((s_var - 32.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_stats_skewness_symmetric() {
        let mut stats = RunningStats::new();
        // Symmetric distribution should have skewness ≈ 0
        for &x in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
            stats.push(x);
        }
        let skew = stats.skewness().expect("skewness should exist");
        assert!(skew.abs() < 1e-10, "skewness={skew}");
    }

    #[test]
    fn test_running_stats_kurtosis_normal() {
        // For a normal-like distribution the excess kurtosis should be ≈ 0
        let mut stats = RunningStats::new();
        // Use a reasonably symmetric dataset
        let data = [-2.0_f64, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 2.0];
        for &x in &data {
            stats.push(x);
        }
        // Just check it computes without panic and returns a finite value
        let kurt = stats.kurtosis().expect("kurtosis should exist");
        assert!(kurt.is_finite(), "kurtosis must be finite, got {kurt}");
    }

    #[test]
    fn test_running_stats_merge() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut full = RunningStats::new();
        for &x in &data {
            full.push(x);
        }

        let mut left = RunningStats::new();
        let mut right = RunningStats::new();
        for &x in &data[..4] {
            left.push(x);
        }
        for &x in &data[4..] {
            right.push(x);
        }
        left.merge(&right);

        let m_full = full.mean().expect("full mean");
        let m_merged = left.mean().expect("merged mean");
        assert!((m_full - m_merged).abs() < 1e-12);

        let v_full = full.variance().expect("full var");
        let v_merged = left.variance().expect("merged var");
        assert!((v_full - v_merged).abs() < 1e-10);
    }

    // --- poly_eval_horner ---

    #[test]
    fn test_poly_eval_horner_quadratic() {
        // p(x) = 2x^2 + 3x + 1 at x=2 => 8 + 6 + 1 = 15
        let result = poly_eval_horner(&[2.0_f64, 3.0, 1.0], 2.0);
        assert!((result - 15.0).abs() < 1e-14);
    }

    #[test]
    fn test_poly_eval_horner_constant() {
        let result = poly_eval_horner(&[7.0_f64], 42.0);
        assert!((result - 7.0).abs() < 1e-14);
    }

    #[test]
    fn test_poly_eval_horner_empty() {
        // Empty coefficient list => 0
        let result = poly_eval_horner(&[], 5.0);
        assert_eq!(result, 0.0);
    }

    // --- log_sum_exp ---

    #[test]
    fn test_log_sum_exp_stability() {
        let v = [1000.0_f64, 1001.0, 1002.0];
        let lse = log_sum_exp(&v);
        assert!(lse.is_finite(), "must not overflow");
        assert!((lse - 1002.407606).abs() < 1e-4);
    }

    #[test]
    fn test_log_sum_exp_empty() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sum_exp_single() {
        let v = [5.0_f64];
        assert!((log_sum_exp(&v) - 5.0).abs() < 1e-14);
    }

    // --- softmax_stable ---

    #[test]
    fn test_softmax_stable_sums_to_one() {
        let v = [1.0_f64, 2.0, 3.0];
        let s = softmax_stable(&v);
        let total: f64 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_softmax_stable_large_values() {
        let v = [1000.0_f64, 1001.0, 1002.0];
        let s = softmax_stable(&v);
        let total: f64 = s.iter().sum();
        assert!((total - 1.0).abs() < 1e-14, "softmax sum={total}");
        // Last element largest
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn test_softmax_stable_empty() {
        let s = softmax_stable(&[]);
        assert!(s.is_empty());
    }
}
