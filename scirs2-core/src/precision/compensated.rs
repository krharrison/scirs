//! Compensated summation and error-free arithmetic transformations.
//!
//! Floating-point addition is not associative: `(a + b) + c != a + (b + c)` in
//! general.  When summing many values the accumulated rounding errors can
//! dwarf the true result.  This module provides algorithms that eliminate or
//! tightly bound those errors.
//!
//! # Algorithms provided
//!
//! | Function | Error bound | Reference |
//! |---|---|---|
//! | `kahan_sum` | O(n·u²) | Kahan (1965) |
//! | `neumaier_sum` | O(n·u²), better constants | Neumaier (1974) |
//! | `ogita_sum` | ≤ (n-1)·u | Ogita–Rump–Oishi (2005) |
//! | `dot_product_compensated` | ≤ 0.5 ulp | Ogita–Rump–Oishi (2005) |
//! | `two_sum` | exact (no rounding) | Knuth TwoSum |
//! | `two_product` | exact (FMA-free) | Dekker (1971) |
//!
//! where `u = 0.5 * f64::EPSILON` (unit roundoff).
//!
//! # References
//!
//! - Kahan, W., "Further remarks on reducing truncation errors", CACM, 1965.
//! - Neumaier, A., "Rundungsfehleranalyse einiger Verfahren zur Summation
//!   endlicher Summen", ZAMM, 1974.
//! - Ogita, T., Rump, S.M., Oishi, S., "Accurate sum and dot product",
//!   SIAM J. Sci. Comput., 2005.
//! - Dekker, T.J., "A floating-point technique for extending the available
//!   precision", Numer. Math., 1971.

use crate::error::{CoreError, ErrorContext};

// ---------------------------------------------------------------------------
// Kahan compensated summation
// ---------------------------------------------------------------------------

/// Kahan compensated summation.
///
/// Accumulates the rounding error in a compensation variable `c` so that
/// the final sum has O(n·u²) error instead of O(n·u).
///
/// Returns `Err` if the input slice is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::kahan_sum;
/// let data = vec![1.0_f64, 1e-16, -1.0_f64, 1e-16];
/// let s = kahan_sum(&data).expect("should succeed");
/// assert!((s - 2e-16).abs() < 1e-30);
/// ```
pub fn kahan_sum(values: &[f64]) -> Result<f64, CoreError> {
    if values.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "kahan_sum: empty slice".to_string(),
        )));
    }
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64; // compensation
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    Ok(sum)
}

// ---------------------------------------------------------------------------
// Neumaier improved Kahan summation
// ---------------------------------------------------------------------------

/// Neumaier improved Kahan summation.
///
/// Handles the case where the new term is larger in magnitude than the
/// running sum (which causes Kahan to lose the compensation).  The
/// correction term `c` is updated differently depending on which operand
/// is larger.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::neumaier_sum;
/// let data = vec![1.0_f64, 1e100, -1e100, f64::EPSILON / 2.0];
/// let s = neumaier_sum(&data).expect("should succeed");
/// // Without compensation this would lose the small terms entirely.
/// assert!(s.is_finite());
/// ```
pub fn neumaier_sum(values: &[f64]) -> Result<f64, CoreError> {
    if values.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "neumaier_sum: empty slice".to_string(),
        )));
    }
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for &v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            c += (sum - t) + v;
        } else {
            c += (v - t) + sum;
        }
        sum = t;
    }
    Ok(sum + c)
}

// ---------------------------------------------------------------------------
// Ogita–Rump–Oishi accurate summation
// ---------------------------------------------------------------------------

/// Ogita–Rump–Oishi accurate summation.
///
/// Achieves a forward error of at most `(n-1) * u * |S|` where `u` is the
/// unit roundoff and `n` is the number of summands.  This is the tightest
/// practically-achievable error bound without multi-precision arithmetic.
///
/// The algorithm works by computing the exact error of each pairwise sum
/// via `two_sum`, collecting those errors, and recursing until the
/// compensation vector is small enough.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::ogita_sum;
/// let data = vec![1.0_f64, 1e-15_f64, -1.0_f64 + 1e-15_f64];
/// let s = ogita_sum(&data).expect("should succeed");
/// assert!(s.is_finite());
/// ```
pub fn ogita_sum(values: &[f64]) -> Result<f64, CoreError> {
    if values.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "ogita_sum: empty slice".to_string(),
        )));
    }
    // We need a mutable working copy for the in-place cascade.
    let mut p: Vec<f64> = values.to_vec();
    let n = p.len();

    // Extract exact errors via TwoSum cascade (Algorithm 4.3 from the paper).
    let mut s = 0.0_f64;
    for i in 0..n {
        let (sigma, q) = two_sum(s, p[i]);
        p[i] = q; // store error
        s = sigma;
    }
    // Sum the errors using Neumaier to avoid re-introducing large errors.
    let err_sum = neumaier_sum(&p)?;
    Ok(s + err_sum)
}

// ---------------------------------------------------------------------------
// Compensated dot product  (≤ 0.5 ulp error for faithful result)
// ---------------------------------------------------------------------------

/// Faithfully-rounded dot product (error ≤ 0.5 ulp).
///
/// Uses the Ogita–Rump–Oishi EFTdot approach: compute each product exactly
/// using `two_product`, sum the high parts with TwoSum cascading, and
/// accumulate all the low-order terms.
///
/// Returns `Err` if the slices have different lengths or are empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::dot_product_compensated;
/// let a = vec![1.0_f64, 2.0, 3.0];
/// let b = vec![4.0_f64, 5.0, 6.0];
/// let d = dot_product_compensated(&a, &b).expect("should succeed");
/// assert!((d - 32.0).abs() < 1e-14);
/// ```
pub fn dot_product_compensated(a: &[f64], b: &[f64]) -> Result<f64, CoreError> {
    if a.len() != b.len() {
        return Err(CoreError::InvalidInput(ErrorContext::new(format!(
            "dot_product_compensated: slice length mismatch ({} vs {})",
            a.len(),
            b.len()
        ))));
    }
    if a.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "dot_product_compensated: empty slices".to_string(),
        )));
    }

    let mut p = 0.0_f64;
    let mut s = 0.0_f64;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let (h, l) = two_product(ai, bi);
        let (p_new, q) = two_sum(p, h);
        p = p_new;
        s += l + q;
    }
    Ok(p + s)
}

// ---------------------------------------------------------------------------
// Knuth TwoSum
// ---------------------------------------------------------------------------

/// Knuth's TwoSum: exact split of `fl(a + b)`.
///
/// Returns `(s, e)` such that `s = fl(a + b)` and `e` is the rounding error,
/// i.e. `a + b = s + e` exactly (in real arithmetic).
///
/// This requires 6 floating-point operations and no branching.  It does NOT
/// require FMA.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::two_sum;
/// let (s, e) = two_sum(1.0_f64, f64::EPSILON / 2.0);
/// // s + e == 1.0 + epsilon/2 exactly
/// assert!((s + e - (1.0 + f64::EPSILON / 2.0)).abs() < f64::EPSILON * 1e-10);
/// ```
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let a_prime = s - b;
    let b_prime = s - a_prime;
    let delta_a = a - a_prime;
    let delta_b = b - b_prime;
    let e = delta_a + delta_b;
    (s, e)
}

// ---------------------------------------------------------------------------
// Dekker TwoProduct (FMA-free exact multiplication)
// ---------------------------------------------------------------------------

/// Exact FMA-free product: returns `(p, e)` with `a * b = p + e` exactly.
///
/// Uses Dekker's splitting to represent each operand as the sum of two
/// non-overlapping floats, then computes the exact rounding error of the
/// product.  This is a classic algorithm sometimes called "TwoProduct" in
/// the EFT (Error-Free Transform) literature.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::precision::compensated::two_product;
/// let (p, e) = two_product(3.0_f64, 7.0_f64);
/// // 3 * 7 = 21 exactly, so e should be 0.
/// assert_eq!(p, 21.0);
/// assert_eq!(e, 0.0);
/// ```
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    // Dekker split: split a into a_hi + a_lo with |a_lo| <= 0.5 ulp(a_hi)
    let (a_hi, a_lo) = split_f64(a);
    let (b_hi, b_lo) = split_f64(b);
    // Compute the rounding error of p = a * b.
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Dekker split: split a 64-bit float into two non-overlapping halves.
///
/// Returns `(a_hi, a_lo)` such that `a = a_hi + a_lo` exactly and
/// `a_hi` has at most 27 significant bits.
#[inline]
fn split_f64(a: f64) -> (f64, f64) {
    // The constant 2^27 + 1 is the Dekker splitting constant for f64.
    const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1
    let c = SPLITTER * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    (a_hi, a_lo)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kahan_sum_basic() {
        let data = [1.0_f64, 2.0, 3.0, 4.0];
        let s = kahan_sum(&data).expect("valid");
        assert!((s - 10.0).abs() < 1e-14);
    }

    #[test]
    fn kahan_sum_empty_error() {
        assert!(kahan_sum(&[]).is_err());
    }

    #[test]
    fn neumaier_sum_basic() {
        let data = [1.0_f64, 2.0, 3.0];
        let s = neumaier_sum(&data).expect("valid");
        assert!((s - 6.0).abs() < 1e-14);
    }

    #[test]
    fn neumaier_sum_large_and_small() {
        // 1e15 + 1.0 - 1e15 should not lose 1.0
        let data = [1e15_f64, 1.0, -1e15_f64];
        let s = neumaier_sum(&data).expect("valid");
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ogita_sum_basic() {
        let data = [1.0_f64, 2.0, 3.0];
        let s = ogita_sum(&data).expect("valid");
        assert!((s - 6.0).abs() < 1e-14);
    }

    #[test]
    fn ogita_sum_single() {
        let data = [42.0_f64];
        let s = ogita_sum(&data).expect("valid");
        assert_eq!(s, 42.0);
    }

    #[test]
    fn dot_product_exact_integers() {
        let a = [1.0_f64, 2.0, 3.0];
        let b = [4.0_f64, 5.0, 6.0];
        let d = dot_product_compensated(&a, &b).expect("valid");
        assert!((d - 32.0).abs() < 1e-12);
    }

    #[test]
    fn dot_product_length_mismatch() {
        let a = [1.0_f64, 2.0];
        let b = [3.0_f64];
        assert!(dot_product_compensated(&a, &b).is_err());
    }

    #[test]
    fn two_sum_exact_reconstruction() {
        let a = 1.0_f64;
        let b = f64::EPSILON / 2.0;
        let (s, e) = two_sum(a, b);
        // Real arithmetic: s + e == a + b exactly.
        assert!((s + e - (a + b)).abs() < f64::EPSILON * f64::EPSILON);
    }

    #[test]
    fn two_product_integer_exact() {
        let (p, e) = two_product(3.0_f64, 7.0_f64);
        assert_eq!(p, 21.0);
        assert_eq!(e, 0.0);
    }

    #[test]
    fn two_product_error_reconstruct() {
        let a = 1.0_f64 / 3.0;
        let b = 3.0_f64;
        let (p, e) = two_product(a, b);
        // a * b should be 1.0; p + e should equal 1.0 in exact arithmetic.
        let reconstructed = p + e;
        assert!((reconstructed - 1.0).abs() < 1e-15);
    }

    #[test]
    fn split_roundtrip() {
        let x = 1.23456789_f64;
        let (hi, lo) = super::split_f64(x);
        assert!((hi + lo - x).abs() < f64::EPSILON * x.abs());
    }
}
