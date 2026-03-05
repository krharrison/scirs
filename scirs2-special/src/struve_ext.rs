//! Extended Struve function interface
//!
//! Provides top-level public wrappers matching the requested API:
//!
//! * `struve_h(nu, x)` – Struve H_ν(x)
//! * `struve_l(nu, x)` – Modified Struve L_ν(x)
//! * `struve_h0(x)` – H_0(x) specialisation
//! * `struve_h1(x)` – H_1(x) specialisation
//!
//! All functions delegate to the high-quality implementations in `crate::struve`.

use crate::error::SpecialResult;

/// Struve function H_ν(x).
///
/// The Struve function is defined by:
/// ```text
/// H_ν(x) = (x/2)^{ν+1} ∑_{m=0}^∞  (-1)^m (x/2)^{2m} / [Γ(m+3/2) Γ(m+ν+3/2)]
/// ```
///
/// For small |x| a power series is used; for large |x| an asymptotic form
/// relative to Bessel functions of the second kind is used.
///
/// # Arguments
///
/// * `nu` – Order (real)
/// * `x`  – Argument (real, x ≥ 0 recommended)
///
/// # Returns
///
/// * `Ok(f64)` – value of H_ν(x)
/// * `Err` – if inputs are NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::struve_h;
///
/// // H_0(0) = 0
/// let h0_0 = struve_h(0.0, 0.0).expect("H_0(0)");
/// assert!((h0_0 - 0.0).abs() < 1e-14);
///
/// // H_1(0) = 0
/// let h1_0 = struve_h(1.0, 0.0).expect("H_1(0)");
/// assert!((h1_0 - 0.0).abs() < 1e-14);
/// ```
#[inline]
pub fn struve_h(nu: f64, x: f64) -> SpecialResult<f64> {
    crate::struve::struve(nu, x)
}

/// Modified Struve function L_ν(x).
///
/// The modified Struve function is defined by:
/// ```text
/// L_ν(x) = -i^{-ν} H_ν(ix)
///          = (x/2)^{ν+1} ∑_{m=0}^∞ (x/2)^{2m} / [Γ(m+3/2) Γ(m+ν+3/2)]
/// ```
///
/// All terms are positive, so L_ν diverges for large x (unlike H_ν which
/// oscillates like Bessel Y).
///
/// # Arguments
///
/// * `nu` – Order (real)
/// * `x`  – Argument (real, x ≥ 0 recommended)
///
/// # Returns
///
/// * `Ok(f64)` – value of L_ν(x)
/// * `Err` – if inputs are NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::struve_l;
///
/// // L_0(0) = 0
/// let l0_0 = struve_l(0.0, 0.0).expect("L_0(0)");
/// assert!((l0_0 - 0.0).abs() < 1e-14);
///
/// // L should grow for large x
/// let l0_5 = struve_l(0.0, 5.0).expect("L_0(5)");
/// assert!(l0_5 > 0.0 && l0_5.is_finite());
/// ```
#[inline]
pub fn struve_l(nu: f64, x: f64) -> SpecialResult<f64> {
    crate::struve::mod_struve(nu, x)
}

/// Struve function H_0(x) — order-0 specialisation.
///
/// H_0(x) = (2/π) [1 - cos x / x - ∫_0^x sin t / t dt]  (for large x)
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `Ok(f64)` – value of H_0(x)
/// * `Err` – if `x` is NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::struve_h0;
///
/// // H_0(0) = 0
/// let h = struve_h0(0.0).expect("H_0(0)");
/// assert!(h.abs() < 1e-14);
/// ```
#[inline]
pub fn struve_h0(x: f64) -> SpecialResult<f64> {
    crate::struve::struve(0.0, x)
}

/// Struve function H_1(x) — order-1 specialisation.
///
/// H_1(x) = (2/π) [1 - J_0(x)] + (2x/π) [- 1/3 + x²/15 - …]
///
/// # Arguments
///
/// * `x` – Real argument
///
/// # Returns
///
/// * `Ok(f64)` – value of H_1(x)
/// * `Err` – if `x` is NaN
///
/// # Examples
///
/// ```
/// use scirs2_special::struve_h1;
///
/// // H_1(0) = 0
/// let h = struve_h1(0.0).expect("H_1(0)");
/// assert!(h.abs() < 1e-14);
/// ```
#[inline]
pub fn struve_h1(x: f64) -> SpecialResult<f64> {
    crate::struve::struve(1.0, x)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_struve_h0_at_zero() {
        let val = struve_h0(0.0).expect("H_0(0)");
        assert_relative_eq!(val, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_struve_h1_at_zero() {
        let val = struve_h1(0.0).expect("H_1(0)");
        assert_relative_eq!(val, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_struve_h_consistency_with_h0() {
        // struve_h(0, x) should equal struve_h0(x)
        for x in [0.5_f64, 1.0, 2.0, 5.0] {
            let h = struve_h(0.0, x).expect("H");
            let h0 = struve_h0(x).expect("H0");
            assert!((h - h0).abs() < 1e-14, "at x={x}: struve_h={h}, struve_h0={h0}");
        }
    }

    #[test]
    fn test_struve_h_consistency_with_h1() {
        for x in [0.5_f64, 1.0, 2.0, 5.0] {
            let h = struve_h(1.0, x).expect("H");
            let h1 = struve_h1(x).expect("H1");
            assert!((h - h1).abs() < 1e-14, "at x={x}: struve_h={h}, struve_h1={h1}");
        }
    }

    #[test]
    fn test_struve_l0_positive() {
        // L_0(x) > 0 for x > 0
        for x in [0.5_f64, 1.0, 2.0, 5.0] {
            let l = struve_l(0.0, x).expect("L_0");
            assert!(l > 0.0, "L_0({x}) should be positive, got {l}");
        }
    }

    #[test]
    fn test_struve_l_consistency_with_mod_struve() {
        // struve_l should be consistent with the existing mod_struve export
        for x in [1.0_f64, 2.0, 3.0] {
            let l = struve_l(0.0, x).expect("struve_l");
            let ms = crate::struve::mod_struve(0.0, x).expect("mod_struve");
            assert!((l - ms).abs() < 1e-14, "at x={x}: struve_l={l}, mod_struve={ms}");
        }
    }

    #[test]
    fn test_struve_h_small_argument() {
        // For small x, H_0(x) ≈ 2x/π
        let x = 0.1_f64;
        let h0 = struve_h0(x).expect("H_0(0.1)");
        let approx = 2.0 * x / std::f64::consts::PI;
        assert_relative_eq!(h0, approx, epsilon = 1e-4);
    }

    #[test]
    fn test_struve_h_h1_small_argument() {
        // For small x, H_1(x) ≈ 2x²/(3π)
        let x = 0.1_f64;
        let h1 = struve_h1(x).expect("H_1(0.1)");
        let approx = 2.0 * x * x / (3.0 * std::f64::consts::PI);
        assert_relative_eq!(h1, approx, epsilon = 1e-4);
    }

    #[test]
    fn test_struve_h_finite() {
        // All calls with finite input should return finite values
        for nu in [0.0_f64, 1.0, 2.0, 0.5, 1.5] {
            for x in [0.0_f64, 1.0, 5.0, 10.0] {
                let val = struve_h(nu, x).expect("struve_h");
                assert!(val.is_finite(), "H_{nu}({x}) should be finite, got {val}");
            }
        }
    }

    #[test]
    fn test_struve_l_monotone() {
        // L_0 is monotonically increasing for x > 0
        let l1 = struve_l(0.0, 1.0).expect("L_0(1)");
        let l2 = struve_l(0.0, 2.0).expect("L_0(2)");
        let l3 = struve_l(0.0, 3.0).expect("L_0(3)");
        assert!(l2 > l1, "L_0 should increase: L(1)={l1}, L(2)={l2}");
        assert!(l3 > l2, "L_0 should increase: L(2)={l2}, L(3)={l3}");
    }
}
