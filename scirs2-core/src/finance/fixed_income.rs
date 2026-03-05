//! Fixed income pricing: bonds, duration, convexity, yield-to-maturity
//!
//! Implements plain-vanilla bond analytics using standard textbook formulas.
//! All time periods are expressed in consistent units (e.g. semi-annual periods
//! with a semi-annual coupon rate, or annual periods with annual coupon rate).

use crate::error::{CoreError, CoreResult};

// ============================================================
// Bond price
// ============================================================

/// Price a plain-vanilla bond with periodic coupon payments.
///
/// The bond makes `n_periods` coupon payments of `face * coupon_rate` each,
/// plus the face value at maturity.  All rates are expressed per-period
/// (e.g. pass semi-annual rates for semi-annual bonds).
///
/// # Arguments
/// * `face` - Face (par) value of the bond
/// * `coupon_rate` - Coupon rate per period (e.g. 0.03 for 3% semi-annual)
/// * `ytm` - Yield-to-maturity per period
/// * `n_periods` - Total number of coupon periods to maturity
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for non-positive face or coupon/ytm ≤ -1.
pub fn bond_price(face: f64, coupon_rate: f64, ytm: f64, n_periods: usize) -> CoreResult<f64> {
    validate_bond_params(face, ytm)?;
    if n_periods == 0 {
        return Ok(face); // Bond has matured; worth face value at maturity
    }

    let coupon = face * coupon_rate;
    let disc = 1.0 + ytm;

    // Present value of coupons + present value of principal
    // PV = C * [1 - (1+y)^{-n}] / y  when y ≠ 0
    let pv_coupons = if ytm.abs() < 1e-12 {
        coupon * n_periods as f64
    } else {
        coupon * (1.0 - disc.powi(-(n_periods as i32))) / ytm
    };
    let pv_principal = face / disc.powi(n_periods as i32);

    Ok(pv_coupons + pv_principal)
}

// ============================================================
// Duration (Macaulay + Modified)
// ============================================================

/// Compute the Macaulay duration and Modified duration of a bond.
///
/// * Macaulay duration: weighted average time of cash flows (in periods).
/// * Modified duration: Macaulay duration / (1 + ytm), measures % price change per 1-unit yield move.
///
/// # Arguments
/// * `face` - Face (par) value
/// * `coupon_rate` - Coupon rate per period
/// * `ytm` - Yield-to-maturity per period
/// * `n_periods` - Total coupon periods
///
/// # Returns
/// `(macaulay_duration, modified_duration)` in period units.
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for invalid parameters or zero bond price.
pub fn bond_duration(
    face: f64,
    coupon_rate: f64,
    ytm: f64,
    n_periods: usize,
) -> CoreResult<(f64, f64)> {
    validate_bond_params(face, ytm)?;
    if n_periods == 0 {
        return Ok((0.0, 0.0));
    }

    let coupon = face * coupon_rate;
    let disc = 1.0 + ytm;

    let mut weighted_pv = 0.0_f64;
    let mut total_pv = 0.0_f64;

    for t in 1..=n_periods {
        let cf = if t == n_periods {
            coupon + face
        } else {
            coupon
        };
        let pv = cf / disc.powi(t as i32);
        weighted_pv += t as f64 * pv;
        total_pv += pv;
    }

    if total_pv.abs() < 1e-12 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Bond price is (near) zero; cannot compute duration"),
        ));
    }

    let macaulay = weighted_pv / total_pv;
    let modified = macaulay / disc;

    Ok((macaulay, modified))
}

// ============================================================
// Convexity
// ============================================================

/// Compute the convexity of a bond.
///
/// Convexity captures the curvature of the price-yield relationship,
/// providing a second-order correction to the duration-based approximation:
///
/// `ΔP/P ≈ -D_mod * Δy + ½ * Convexity * (Δy)²`
///
/// # Arguments
/// * `face` - Face value
/// * `coupon_rate` - Coupon rate per period
/// * `ytm` - Yield-to-maturity per period
/// * `n_periods` - Total coupon periods
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for invalid parameters or zero bond price.
pub fn bond_convexity(face: f64, coupon_rate: f64, ytm: f64, n_periods: usize) -> CoreResult<f64> {
    validate_bond_params(face, ytm)?;
    if n_periods == 0 {
        return Ok(0.0);
    }

    let coupon = face * coupon_rate;
    let disc = 1.0 + ytm;
    let price = bond_price(face, coupon_rate, ytm, n_periods)?;

    if price.abs() < 1e-12 {
        return Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Bond price is (near) zero; cannot compute convexity"),
        ));
    }

    let mut conv_sum = 0.0_f64;
    for t in 1..=n_periods {
        let cf = if t == n_periods {
            coupon + face
        } else {
            coupon
        };
        let t_f = t as f64;
        conv_sum += t_f * (t_f + 1.0) * cf / disc.powi(t as i32 + 2);
    }

    Ok(conv_sum / price)
}

// ============================================================
// Yield-to-Maturity solver
// ============================================================

/// Solve for the yield-to-maturity given a bond's market price.
///
/// Uses Newton-Raphson iteration starting from an initial guess of
/// `coupon_rate` (a reasonable first approximation for par/near-par bonds).
///
/// # Arguments
/// * `price` - Observed market price of the bond
/// * `face` - Face (par) value
/// * `coupon_rate` - Coupon rate per period
/// * `n_periods` - Total coupon periods
///
/// # Returns
/// The per-period yield-to-maturity.
///
/// # Errors
/// * [`CoreError::InvalidArgument`] if `price` ≤ 0 or `face` ≤ 0.
/// * [`CoreError::ConvergenceError`] if Newton-Raphson does not converge within 200 iterations.
pub fn yield_to_maturity(
    price: f64,
    face: f64,
    coupon_rate: f64,
    n_periods: usize,
) -> CoreResult<f64> {
    if price <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Bond price must be positive",
        )));
    }
    if face <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Face value must be positive",
        )));
    }
    if n_periods == 0 {
        // Zero coupon / matured bond
        return Ok(0.0);
    }

    let coupon = face * coupon_rate;
    // Initial guess: approximate yield as coupon rate adjusted for premium/discount
    let approx = (coupon + (face - price) / n_periods as f64) / ((face + price) / 2.0);
    let mut y = approx.max(1e-6);

    const MAX_ITER: usize = 200;
    const TOL: f64 = 1e-10;

    for _ in 0..MAX_ITER {
        let price_y = bond_price_raw(coupon, face, y, n_periods);
        let residual = price_y - price;
        if residual.abs() < TOL {
            return Ok(y);
        }
        // Derivative dP/dy
        let dpdy = bond_price_deriv(coupon, face, y, n_periods);
        if dpdy.abs() < 1e-15 {
            break;
        }
        let step = residual / dpdy;
        y -= step;
        // Keep y bounded to (-1, ∞) for numerical stability
        if y <= -1.0 {
            y = -0.9999;
        }
        if residual.abs() < TOL {
            return Ok(y);
        }
    }

    // Final convergence check
    let final_price = bond_price_raw(coupon, face, y, n_periods);
    if (final_price - price).abs() < 1e-6 * price {
        Ok(y)
    } else {
        Err(CoreError::ConvergenceError(
            crate::error::ErrorContext::new(format!(
                "YTM solver did not converge (residual={:.6e})",
                (final_price - price).abs()
            )),
        ))
    }
}

// ============================================================
// Internal helpers
// ============================================================

fn validate_bond_params(face: f64, ytm: f64) -> CoreResult<()> {
    if face <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Face value must be positive",
        )));
    }
    if ytm <= -1.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "YTM must be > -1 (i.e., discount factor must be positive)",
        )));
    }
    Ok(())
}

/// Raw bond price computation (avoids re-validation in inner loops)
fn bond_price_raw(coupon: f64, face: f64, ytm: f64, n_periods: usize) -> f64 {
    let disc = 1.0 + ytm;
    let pv_coupons = if ytm.abs() < 1e-12 {
        coupon * n_periods as f64
    } else {
        coupon * (1.0 - disc.powi(-(n_periods as i32))) / ytm
    };
    let pv_principal = face / disc.powi(n_periods as i32);
    pv_coupons + pv_principal
}

/// Analytical derivative dP/dy used by Newton-Raphson
fn bond_price_deriv(coupon: f64, face: f64, ytm: f64, n_periods: usize) -> f64 {
    let disc = 1.0 + ytm;
    let n = n_periods as i32;
    let mut deriv = 0.0_f64;
    for t in 1..=n_periods {
        let cf = if t == n_periods {
            coupon + face
        } else {
            coupon
        };
        deriv -= t as f64 * cf / disc.powi(t as i32 + 1);
    }
    deriv
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- bond_price ---
    #[test]
    fn test_bond_price_at_par() {
        // When coupon rate == ytm, price == face value
        let price = bond_price(1000.0, 0.05, 0.05, 10).expect("should succeed");
        assert!((price - 1000.0).abs() < 1e-6, "Bond at par: {price:.6}");
    }

    #[test]
    fn test_bond_price_premium() {
        // coupon > ytm => price > face
        let price = bond_price(1000.0, 0.08, 0.05, 10).expect("should succeed");
        assert!(price > 1000.0, "Premium bond: {price:.4}");
    }

    #[test]
    fn test_bond_price_discount() {
        // coupon < ytm => price < face
        let price = bond_price(1000.0, 0.03, 0.06, 10).expect("should succeed");
        assert!(price < 1000.0, "Discount bond: {price:.4}");
    }

    #[test]
    fn test_bond_price_zero_coupon() {
        // Zero coupon: P = face / (1+ytm)^n
        let price = bond_price(1000.0, 0.0, 0.05, 5).expect("should succeed");
        let expected = 1000.0 / 1.05_f64.powi(5);
        assert!((price - expected).abs() < 1e-8, "Zero coupon: {price:.6}");
    }

    #[test]
    fn test_bond_price_matured_returns_face() {
        let price = bond_price(500.0, 0.05, 0.07, 0).expect("should succeed");
        assert!((price - 500.0).abs() < 1e-10);
    }

    #[test]
    fn test_bond_price_invalid_face() {
        assert!(bond_price(0.0, 0.05, 0.05, 10).is_err());
        assert!(bond_price(-100.0, 0.05, 0.05, 10).is_err());
    }

    #[test]
    fn test_bond_price_invalid_ytm() {
        assert!(bond_price(1000.0, 0.05, -1.5, 10).is_err());
    }

    // --- bond_duration ---
    #[test]
    fn test_macaulay_duration_zero_coupon() {
        // Zero coupon bond: Macaulay duration == n
        let (mac, _) = bond_duration(1000.0, 0.0, 0.05, 10).expect("should succeed");
        assert!(
            (mac - 10.0).abs() < 1e-8,
            "Zero coupon Macaulay duration = n: {mac:.8}"
        );
    }

    #[test]
    fn test_modified_duration_relationship() {
        let (mac, modd) = bond_duration(1000.0, 0.05, 0.05, 10).expect("should succeed");
        let expected_mod = mac / 1.05;
        assert!(
            (modd - expected_mod).abs() < 1e-10,
            "Modified = Macaulay / (1+y)"
        );
    }

    #[test]
    fn test_duration_coupon_bond_less_than_maturity() {
        // Coupon bond: Macaulay duration < n_periods
        let (mac, _) = bond_duration(1000.0, 0.05, 0.05, 10).expect("should succeed");
        assert!(mac < 10.0, "Coupon bond duration < maturity: {mac:.4}");
    }

    #[test]
    fn test_duration_zero_periods() {
        let (mac, modd) = bond_duration(1000.0, 0.05, 0.05, 0).expect("should succeed");
        assert_eq!(mac, 0.0);
        assert_eq!(modd, 0.0);
    }

    // --- bond_convexity ---
    #[test]
    fn test_convexity_positive() {
        let conv = bond_convexity(1000.0, 0.05, 0.05, 10).expect("should succeed");
        assert!(conv > 0.0, "Convexity must be positive: {conv:.4}");
    }

    #[test]
    fn test_convexity_zero_coupon() {
        // Zero coupon convexity = n*(n+1) / (1+y)^2
        let (n, y) = (5usize, 0.05_f64);
        let conv = bond_convexity(1000.0, 0.0, y, n).expect("should succeed");
        let expected = (n as f64) * (n as f64 + 1.0) / (1.0 + y).powi(2);
        assert!(
            (conv - expected).abs() < 1e-6,
            "Zero coupon convexity: {conv:.6} vs {expected:.6}"
        );
    }

    // --- yield_to_maturity ---
    #[test]
    fn test_ytm_at_par() {
        // If price == face, ytm == coupon_rate
        let ytm = yield_to_maturity(1000.0, 1000.0, 0.05, 10).expect("should succeed");
        assert!((ytm - 0.05).abs() < 1e-8, "YTM at par: {ytm:.10}");
    }

    #[test]
    fn test_ytm_roundtrip() {
        // Price -> YTM -> re-price should recover original price
        let (face, cr, target_ytm, n) = (1000.0, 0.06, 0.08, 20);
        let price = bond_price(face, cr, target_ytm, n).expect("should succeed");
        let solved_ytm = yield_to_maturity(price, face, cr, n).expect("should succeed");
        assert!(
            (solved_ytm - target_ytm).abs() < 1e-8,
            "YTM roundtrip: {solved_ytm:.10} vs {target_ytm}"
        );
    }

    #[test]
    fn test_ytm_discount_bond() {
        // Discount bond: ytm > coupon_rate
        let price = 900.0_f64;
        let ytm = yield_to_maturity(price, 1000.0, 0.05, 10).expect("should succeed");
        assert!(
            ytm > 0.05,
            "Discount bond: ytm {ytm:.4} should exceed coupon rate 0.05"
        );
    }

    #[test]
    fn test_ytm_premium_bond() {
        // Premium bond: ytm < coupon_rate
        let price = 1100.0_f64;
        let ytm = yield_to_maturity(price, 1000.0, 0.07, 10).expect("should succeed");
        assert!(
            ytm < 0.07,
            "Premium bond: ytm {ytm:.4} should be below coupon rate 0.07"
        );
    }

    #[test]
    fn test_ytm_invalid_price() {
        assert!(yield_to_maturity(0.0, 1000.0, 0.05, 10).is_err());
        assert!(yield_to_maturity(-100.0, 1000.0, 0.05, 10).is_err());
    }
}
