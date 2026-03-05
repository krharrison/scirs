//! Verified mathematical functions and Taylor models with interval remainders.
//!
//! This module provides rigorously-enclosing evaluations of mathematical
//! functions on interval arguments, as well as Taylor-model enclosures of
//! smooth functions.
//!
//! # Overview
//!
//! * [`verified_sqrt`] / [`verified_exp`] / [`verified_ln`] — thin wrappers
//!   that forward to the outward-rounded methods on `Interval<f64>`, provided
//!   here for a uniform function-call API.
//! * [`verified_sin`] / [`verified_cos`] / [`verified_atan`] — trigonometric
//!   and arctangent enclosures.
//! * [`enclose_polynomial`] — evaluates a polynomial using the Horner scheme
//!   in interval arithmetic.
//! * [`taylor_verified`] — computes a Taylor model: a polynomial approximation
//!   together with an enclosing remainder interval.

use crate::error::{CoreError, CoreResult};
use super::interval::Interval;
// Vec, String, format are available from std prelude

// ---------------------------------------------------------------------------
// Verified elementary functions
// ---------------------------------------------------------------------------

/// Compute `√x` with outward rounding.
///
/// Returns an interval `r` such that for every `x₀ ∈ x`, `√x₀ ∈ r`.
///
/// Returns `Err` if `x.hi < 0`.
#[inline]
pub fn verified_sqrt(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.hi < 0.0 {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_sqrt: argument is entirely negative"),
        ));
    }
    Ok(x.sqrt())
}

/// Compute `eˣ` with outward rounding.
///
/// Returns an interval `r` such that for every `x₀ ∈ x`, `eˣ⁰ ∈ r`.
#[inline]
pub fn verified_exp(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_exp: empty interval argument"),
        ));
    }
    Ok(x.exp())
}

/// Compute `ln(x)` with outward rounding.
///
/// Returns `Err` if `x.hi <= 0` (logarithm not defined for non-positive reals).
#[inline]
pub fn verified_ln(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.hi <= 0.0 {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new(
                "verified_ln: argument must have a positive upper bound",
            ),
        ));
    }
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_ln: empty interval argument"),
        ));
    }
    Ok(x.ln())
}

/// Compute `sin(x)` with outward rounding.
#[inline]
pub fn verified_sin(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_sin: empty interval argument"),
        ));
    }
    Ok(x.sin())
}

/// Compute `cos(x)` with outward rounding.
#[inline]
pub fn verified_cos(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_cos: empty interval argument"),
        ));
    }
    Ok(x.cos())
}

/// Compute `atan(x)` with outward rounding.
#[inline]
pub fn verified_atan(x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("verified_atan: empty interval argument"),
        ));
    }
    Ok(x.atan())
}

// ---------------------------------------------------------------------------
// Polynomial enclosure — Horner scheme
// ---------------------------------------------------------------------------

/// Evaluate a polynomial at an interval argument using the Horner scheme.
///
/// Given coefficients `c = [c₀, c₁, …, cₙ]` the polynomial is
///
/// ```text
/// p(x) = c[0] + c[1]*x + c[2]*x² + … + c[n]*xⁿ
///       = c[0] + x*(c[1] + x*(c[2] + … + x*c[n])…)
/// ```
///
/// All arithmetic is performed in `Interval<f64>` with outward rounding so
/// the result provably contains every `p(x₀)` for `x₀ ∈ x`.
///
/// # Arguments
///
/// * `coeffs` — polynomial coefficients in **increasing** degree order
///   (index 0 = constant term).
/// * `x` — the evaluation point as an interval.
///
/// # Returns
///
/// An interval enclosing `p(x)`.
///
/// Returns `Err` if `coeffs` is empty or `x` is empty.
pub fn enclose_polynomial(coeffs: &[f64], x: Interval<f64>) -> CoreResult<Interval<f64>> {
    if coeffs.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("enclose_polynomial: empty coefficient slice"),
        ));
    }
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("enclose_polynomial: empty interval argument"),
        ));
    }

    // Horner evaluation from highest degree down to lowest
    let mut acc = Interval::point(*coeffs.last().expect("non-empty checked above"));
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = acc * x + Interval::point(c);
    }
    Ok(acc)
}

/// Evaluate a polynomial with interval coefficients at an interval argument.
///
/// Same as [`enclose_polynomial`] but accepts `Interval<f64>` coefficients.
pub fn enclose_polynomial_iv(
    coeffs: &[Interval<f64>],
    x: Interval<f64>,
) -> CoreResult<Interval<f64>> {
    if coeffs.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("enclose_polynomial_iv: empty coefficient slice"),
        ));
    }
    if x.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("enclose_polynomial_iv: empty interval argument"),
        ));
    }

    let mut acc = *coeffs.last().expect("non-empty checked above");
    for &c in coeffs[..coeffs.len() - 1].iter().rev() {
        acc = acc * x + c;
    }
    Ok(acc)
}

// ---------------------------------------------------------------------------
// Taylor model
// ---------------------------------------------------------------------------

/// A Taylor model for a smooth function on an interval `[lo, hi]`.
///
/// A Taylor model of order `n` consists of:
///
/// * A degree-`n` polynomial `p(x)` (stored as coefficients at the expansion
///   point `x₀`).
/// * A remainder interval `r` such that `f(x) ∈ p(x - x₀) + r` for all
///   `x ∈ domain`.
///
/// The polynomial is stored in the basis `(x - x₀)^k` so evaluating it at
/// a point `x` requires shifting first.
#[derive(Clone, Debug)]
pub struct TaylorModel {
    /// The expansion centre.
    pub centre: f64,
    /// Polynomial coefficients `[a₀, a₁, …, aₙ]` such that
    /// `p(t) = Σ aₖ * t^k` where `t = x - centre`.
    pub coeffs: Vec<f64>,
    /// Enclosing remainder interval: `f(x) - p(x - centre) ∈ remainder`
    /// for all `x ∈ domain`.
    pub remainder: Interval<f64>,
    /// The domain on which the model is valid.
    pub domain: Interval<f64>,
}

impl TaylorModel {
    /// Construct a Taylor model.
    pub fn new(
        centre: f64,
        coeffs: Vec<f64>,
        remainder: Interval<f64>,
        domain: Interval<f64>,
    ) -> Self {
        Self {
            centre,
            coeffs,
            remainder,
            domain,
        }
    }

    /// Evaluate the Taylor model at an interval `x ⊆ domain`.
    ///
    /// Returns an interval enclosing `{f(x₀) : x₀ ∈ x}`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `x` is not a subset of `self.domain`.
    pub fn evaluate(&self, x: Interval<f64>) -> CoreResult<Interval<f64>> {
        if !self.domain.contains_interval(&x) {
            return Err(CoreError::InvalidInput(
                crate::error::ErrorContext::new(
                    "TaylorModel::evaluate: x is not a subset of the model domain",
                ),
            ));
        }
        let t = x - Interval::point(self.centre);
        let poly_val = enclose_polynomial(&self.coeffs, t)?;
        Ok(poly_val + self.remainder)
    }

    /// Evaluate the polynomial part only (no remainder correction).
    pub fn evaluate_poly(&self, x: f64) -> f64 {
        let t = x - self.centre;
        // Horner with plain f64
        let mut acc = *self.coeffs.last().unwrap_or(&0.0);
        for &c in self.coeffs[..self.coeffs.len().saturating_sub(1)]
            .iter()
            .rev()
        {
            acc = acc * t + c;
        }
        acc
    }

    /// Order of the Taylor model (degree of the polynomial).
    pub fn order(&self) -> usize {
        self.coeffs.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// taylor_verified: build Taylor models for well-known functions
// ---------------------------------------------------------------------------

/// Compute a Taylor model for `exp(x)` centred at `centre` of order `n` on
/// `domain`.
///
/// The Taylor expansion of `exp(t)` at `t = 0` is:
/// ```text
/// exp(t) = 1 + t + t²/2! + … + tⁿ/n! + R(t)
/// ```
/// where the remainder `R(t) ∈ exp(domain − centre) * [tⁿ⁺¹/(n+1)!, tⁿ⁺¹/(n+1)!]`.
///
/// We bound the remainder by evaluating the `(n+1)`-th derivative at the
/// entire domain (which is `exp(x)` itself) and multiplying by the worst-case
/// `(domain − centre)^{n+1} / (n+1)!`.
pub fn taylor_exp(centre: f64, n: usize, domain: Interval<f64>) -> CoreResult<TaylorModel> {
    if domain.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_exp: empty domain"),
        ));
    }
    // Taylor expansion: exp(centre + t) = exp(centre) * sum_{k=0}^{n} t^k / k!
    // Coefficients a_k = exp(centre) / k!
    let exp_c = centre.exp();
    let mut coeffs = Vec::with_capacity(n + 1);
    let mut factorial = 1.0_f64;
    for k in 0..=n {
        if k > 0 {
            factorial *= k as f64;
        }
        coeffs.push(exp_c / factorial);
    }

    // Remainder: R(t) = exp(xi) * t^{n+1} / (n+1)! for some xi in [centre + t_lo, centre + t_hi]
    // We bound exp(xi) by exp(domain) and t^{n+1} by (domain - centre)^{n+1}
    let t_domain = domain - Interval::point(centre);
    let t_power = t_domain.powi((n + 1) as i32);
    let exp_bound = domain.exp();

    let nplus1_factorial = {
        let mut f = 1.0_f64;
        for k in 1..=(n + 1) {
            f *= k as f64;
        }
        f
    };

    let remainder = exp_bound * t_power / Interval::point(nplus1_factorial);

    Ok(TaylorModel::new(centre, coeffs, remainder, domain))
}

/// Compute a Taylor model for `ln(x)` centred at `centre > 0` of order `n`
/// on `domain ⊆ (0, ∞)`.
///
/// The Taylor expansion of `ln(x)` at `x₀` is:
/// ```text
/// ln(x₀ + t) = ln(x₀) + t/x₀ − t²/(2x₀²) + … + (−1)^{k+1} t^k / (k x₀^k)
/// ```
///
/// We compute the coefficients analytically and bound the remainder using
/// the `(n+1)`-th derivative bound `n! / ξ^{n+1}` over `domain`.
pub fn taylor_ln(centre: f64, n: usize, domain: Interval<f64>) -> CoreResult<TaylorModel> {
    if centre <= 0.0 {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_ln: centre must be positive"),
        ));
    }
    if domain.lo <= 0.0 {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_ln: domain must be a subset of (0, ∞)"),
        ));
    }
    if domain.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_ln: empty domain"),
        ));
    }

    // k = 0: ln(x₀)
    // k ≥ 1: (-1)^{k+1} / (k * x₀^k)
    let mut coeffs = Vec::with_capacity(n + 1);
    coeffs.push(centre.ln()); // k = 0
    let mut xpow = centre; // x₀^k
    for k in 1..=n {
        let sign = if k % 2 == 1 { 1.0_f64 } else { -1.0_f64 };
        coeffs.push(sign / (k as f64 * xpow));
        xpow *= centre;
    }

    // Remainder: derivative of order n+1 of ln is (-1)^n * n! / x^{n+1}
    // We bound x over domain (take min for magnitude since domain ⊆ (0,∞))
    let n_factorial = {
        let mut f = 1.0_f64;
        for k in 1..=n {
            f *= k as f64;
        }
        f
    };
    let t_domain = domain - Interval::point(centre);
    let t_power = t_domain.powi((n + 1) as i32);
    // Worst case denominator: x_min^{n+1} where x_min = domain.lo
    let x_min_pow = Interval::point(domain.lo.powi((n + 1) as i32));
    let deriv_bound = Interval::point(n_factorial) / x_min_pow;
    let nplus1_factorial = n_factorial * (n + 1) as f64;
    let remainder = deriv_bound * t_power / Interval::point(nplus1_factorial);

    Ok(TaylorModel::new(centre, coeffs, remainder, domain))
}

/// Compute a Taylor model for `sin(x)` centred at `centre` of order `n` on
/// `domain`.
///
/// Uses the Taylor expansion of `sin` with alternating terms; the remainder
/// is bounded by `|domain − centre|^{n+1} / (n+1)!` since all derivatives
/// of `sin` have magnitude ≤ 1.
pub fn taylor_sin(centre: f64, n: usize, domain: Interval<f64>) -> CoreResult<TaylorModel> {
    if domain.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_sin: empty domain"),
        ));
    }

    // sin(c + t) = sin(c) + cos(c)*t - sin(c)*t²/2! - cos(c)*t³/3! + …
    let mut coeffs = Vec::with_capacity(n + 1);
    let mut factorial = 1.0_f64;
    let s0 = centre.sin();
    let c0 = centre.cos();
    for k in 0..=n {
        if k > 0 {
            factorial *= k as f64;
        }
        // k-th derivative of sin at centre:
        // k mod 4 == 0: sin(centre)
        // k mod 4 == 1: cos(centre)
        // k mod 4 == 2: -sin(centre)
        // k mod 4 == 3: -cos(centre)
        let deriv = match k % 4 {
            0 => s0,
            1 => c0,
            2 => -s0,
            3 => -c0,
            _ => unreachable!(),
        };
        coeffs.push(deriv / factorial);
    }

    // Remainder: |R| ≤ |t|^{n+1} / (n+1)! (since |sin^{(n+1)}| ≤ 1)
    let t_domain = domain - Interval::point(centre);
    let t_power = t_domain.abs().powi((n + 1) as i32);
    let nplus1_factorial = {
        let mut f = 1.0_f64;
        for k in 1..=(n + 1) {
            f *= k as f64;
        }
        f
    };
    let remainder_mag = t_power / Interval::point(nplus1_factorial);
    let remainder = Interval::new(-remainder_mag.hi, remainder_mag.hi);

    Ok(TaylorModel::new(centre, coeffs, remainder, domain))
}

/// Compute a Taylor model for `cos(x)` centred at `centre` of order `n` on
/// `domain`.
pub fn taylor_cos(centre: f64, n: usize, domain: Interval<f64>) -> CoreResult<TaylorModel> {
    if domain.is_empty() {
        return Err(CoreError::InvalidInput(
            crate::error::ErrorContext::new("taylor_cos: empty domain"),
        ));
    }

    let mut coeffs = Vec::with_capacity(n + 1);
    let mut factorial = 1.0_f64;
    let s0 = centre.sin();
    let c0 = centre.cos();
    for k in 0..=n {
        if k > 0 {
            factorial *= k as f64;
        }
        // k-th derivative of cos at centre:
        // k mod 4 == 0: cos(centre)
        // k mod 4 == 1: -sin(centre)
        // k mod 4 == 2: -cos(centre)
        // k mod 4 == 3: sin(centre)
        let deriv = match k % 4 {
            0 => c0,
            1 => -s0,
            2 => -c0,
            3 => s0,
            _ => unreachable!(),
        };
        coeffs.push(deriv / factorial);
    }

    let t_domain = domain - Interval::point(centre);
    let t_power = t_domain.abs().powi((n + 1) as i32);
    let nplus1_factorial = {
        let mut f = 1.0_f64;
        for k in 1..=(n + 1) {
            f *= k as f64;
        }
        f
    };
    let remainder_mag = t_power / Interval::point(nplus1_factorial);
    let remainder = Interval::new(-remainder_mag.hi, remainder_mag.hi);

    Ok(TaylorModel::new(centre, coeffs, remainder, domain))
}

// ---------------------------------------------------------------------------
// Convenience: generic Taylor model builder
// ---------------------------------------------------------------------------

/// Tag enum for functions supported by [`taylor_verified`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaylorFunction {
    /// Natural exponential `eˣ`.
    Exp,
    /// Natural logarithm `ln(x)`.  Requires `domain ⊆ (0, ∞)`.
    Ln,
    /// Sine function.
    Sin,
    /// Cosine function.
    Cos,
}

/// Compute a Taylor model of order `n` for `func` centred at `centre` on
/// `domain`.
///
/// This is a unified entry point that dispatches to the appropriate
/// `taylor_*` function based on the `TaylorFunction` variant.
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_core::interval::functions::{taylor_verified, TaylorFunction};
/// use scirs2_core::interval::Interval;
///
/// let domain = Interval::new(0.0_f64, 1.0);
/// let tm = taylor_verified(TaylorFunction::Exp, 0.5, 5, domain).expect("should succeed");
/// // Evaluate at x = 0.7
/// let result = tm.evaluate(Interval::point(0.7)).expect("should succeed");
/// assert!(result.contains(0.7_f64.exp()));
/// ```
pub fn taylor_verified(
    func: TaylorFunction,
    centre: f64,
    n: usize,
    domain: Interval<f64>,
) -> CoreResult<TaylorModel> {
    match func {
        TaylorFunction::Exp => taylor_exp(centre, n, domain),
        TaylorFunction::Ln => taylor_ln(centre, n, domain),
        TaylorFunction::Sin => taylor_sin(centre, n, domain),
        TaylorFunction::Cos => taylor_cos(centre, n, domain),
    }
}

// ---------------------------------------------------------------------------
// Higher-order verified composition utilities
// ---------------------------------------------------------------------------

/// Evaluate the composition `f(g(x))` where `g` is given as a `TaylorModel`
/// and `f` is an elementary function tag, returning a tighter enclosure than
/// naive substitution when the Taylor model of `g` is tight.
///
/// This wraps: compute `y = g.evaluate(x)` and then apply the verified
/// function to `y`.
pub fn compose_verified(
    func: TaylorFunction,
    g: &TaylorModel,
    x: Interval<f64>,
) -> CoreResult<Interval<f64>> {
    let y = g.evaluate(x)?;
    match func {
        TaylorFunction::Exp => verified_exp(y),
        TaylorFunction::Ln => verified_ln(y),
        TaylorFunction::Sin => verified_sin(y),
        TaylorFunction::Cos => verified_cos(y),
    }
}

/// Bound the range of a polynomial over an interval using Horner-based
/// interval evaluation.
///
/// This is a direct application of [`enclose_polynomial`] and is provided
/// for a more descriptive API.
#[inline]
pub fn polynomial_range(coeffs: &[f64], domain: Interval<f64>) -> CoreResult<Interval<f64>> {
    enclose_polynomial(coeffs, domain)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts;

    fn iv(lo: f64, hi: f64) -> Interval<f64> {
        Interval::new(lo, hi)
    }

    #[test]
    fn test_verified_sqrt() {
        let r = verified_sqrt(iv(4.0, 9.0)).expect("sqrt ok");
        assert!(r.lo <= 2.0 && r.hi >= 3.0, "sqrt([4,9]) = {:?}", r);
    }

    #[test]
    fn test_verified_sqrt_negative_error() {
        assert!(verified_sqrt(iv(-2.0, -1.0)).is_err());
    }

    #[test]
    fn test_verified_exp() {
        let r = verified_exp(iv(0.0, 1.0)).expect("exp ok");
        assert!(r.lo <= 1.0 && r.hi >= consts::E, "exp([0,1]) = {:?}", r);
    }

    #[test]
    fn test_verified_ln() {
        let r = verified_ln(iv(1.0, consts::E)).expect("ln ok");
        assert!(r.lo <= 0.0 && r.hi >= 1.0, "ln([1,e]) = {:?}", r);
    }

    #[test]
    fn test_verified_ln_non_positive_error() {
        assert!(verified_ln(iv(-1.0, 0.0)).is_err());
    }

    #[test]
    fn test_verified_sin() {
        let r = verified_sin(iv(0.0, consts::FRAC_PI_2)).expect("sin ok");
        assert!(r.lo <= 0.0 && r.hi >= 1.0, "sin([0, π/2]) = {:?}", r);
    }

    #[test]
    fn test_verified_cos() {
        let r = verified_cos(iv(0.0, consts::FRAC_PI_2)).expect("cos ok");
        // cos decreases from 1 to 0 on [0, π/2]
        assert!(r.lo <= 0.0 && r.hi >= 1.0, "cos([0, π/2]) = {:?}", r);
    }

    #[test]
    fn test_verified_atan() {
        let r = verified_atan(iv(0.0, 1.0)).expect("atan ok");
        // atan increases from 0 to π/4
        assert!(r.lo <= 0.0 && r.hi >= consts::FRAC_PI_4, "atan([0,1]) = {:?}", r);
    }

    #[test]
    fn test_enclose_polynomial_constant() {
        // p(x) = 5 (constant)
        let r = enclose_polynomial(&[5.0], iv(0.0, 10.0)).expect("poly ok");
        assert!(r.contains(5.0), "const poly = {:?}", r);
    }

    #[test]
    fn test_enclose_polynomial_linear() {
        // p(x) = 2 + 3x on [1, 2] => [5, 8]
        let r = enclose_polynomial(&[2.0, 3.0], iv(1.0, 2.0)).expect("poly ok");
        assert!(r.lo <= 5.0 && r.hi >= 8.0, "linear poly = {:?}", r);
    }

    #[test]
    fn test_enclose_polynomial_quadratic() {
        // p(x) = 1 + x + x² on [0, 1] => [1, 3]
        let r = enclose_polynomial(&[1.0, 1.0, 1.0], iv(0.0, 1.0)).expect("poly ok");
        assert!(r.lo <= 1.0 && r.hi >= 3.0, "quadratic poly = {:?}", r);
    }

    #[test]
    fn test_polynomial_range() {
        // x^2 on [-1, 2] => [0, 4]  (actually min at 0, max at 2)
        // interval evaluation gives [-1,2]^2 = [0, 4] via powi
        let r = polynomial_range(&[0.0, 0.0, 1.0], iv(-1.0, 2.0)).expect("range ok");
        assert!(r.lo <= 0.0 && r.hi >= 4.0, "x^2 range = {:?}", r);
    }

    #[test]
    fn test_taylor_exp_containment() {
        // 5th-order Taylor model of exp at centre=0, domain=[0, 0.5]
        let domain = iv(0.0, 0.5);
        let tm = taylor_exp(0.0, 5, domain).expect("taylor_exp ok");
        // Evaluate at x=0.3
        let x = iv(0.3, 0.3);
        let r = tm.evaluate(x).expect("evaluate ok");
        let exact = 0.3_f64.exp();
        assert!(
            r.contains(exact),
            "taylor_exp(0.3) = {:?}, exact = {}",
            r,
            exact
        );
    }

    #[test]
    fn test_taylor_ln_containment() {
        // 4th-order Taylor model of ln at centre=1, domain=[0.5, 2]
        let domain = iv(0.5, 2.0);
        let tm = taylor_ln(1.0, 4, domain).expect("taylor_ln ok");
        // Evaluate at x=1.5
        let x = iv(1.5, 1.5);
        let r = tm.evaluate(x).expect("evaluate ok");
        let exact = 1.5_f64.ln();
        assert!(
            r.contains(exact),
            "taylor_ln(1.5) = {:?}, exact = {}",
            r,
            exact
        );
    }

    #[test]
    fn test_taylor_sin_containment() {
        let domain = iv(0.0, 1.0);
        let tm = taylor_sin(0.5, 6, domain).expect("taylor_sin ok");
        let x = iv(0.8, 0.8);
        let r = tm.evaluate(x).expect("evaluate ok");
        let exact = 0.8_f64.sin();
        assert!(
            r.contains(exact),
            "taylor_sin(0.8) = {:?}, exact = {}",
            r,
            exact
        );
    }

    #[test]
    fn test_taylor_cos_containment() {
        let domain = iv(0.0, 1.0);
        let tm = taylor_cos(0.5, 6, domain).expect("taylor_cos ok");
        let x = iv(0.2, 0.2);
        let r = tm.evaluate(x).expect("evaluate ok");
        let exact = 0.2_f64.cos();
        assert!(
            r.contains(exact),
            "taylor_cos(0.2) = {:?}, exact = {}",
            r,
            exact
        );
    }

    #[test]
    fn test_taylor_model_order() {
        let tm = taylor_exp(0.0, 4, iv(0.0, 1.0)).expect("taylor model ok");
        assert_eq!(tm.order(), 4);
    }

    #[test]
    fn test_taylor_verified_dispatch() {
        let domain = iv(0.0, 0.5);
        let tm = taylor_verified(TaylorFunction::Exp, 0.25, 4, domain).expect("dispatch ok");
        let r = tm.evaluate(iv(0.4, 0.4)).expect("eval ok");
        assert!(r.contains(0.4_f64.exp()), "dispatch exp: {:?}", r);
    }

    #[test]
    fn test_compose_verified() {
        // f = exp, g = linear: g(x) = x on [0, 0.5]
        // Taylor model of identity: just coeffs [0, 1] (centre=0)
        let domain = iv(0.0, 0.5);
        let g = TaylorModel::new(
            0.0,
            vec![0.0, 1.0],         // p(t) = t
            Interval::point(0.0),   // no remainder for identity
            domain,
        );
        let x = iv(0.3, 0.3);
        let r = compose_verified(TaylorFunction::Exp, &g, x).expect("compose ok");
        assert!(
            r.contains(0.3_f64.exp()),
            "compose exp(identity)(0.3) = {:?}",
            r
        );
    }
}
