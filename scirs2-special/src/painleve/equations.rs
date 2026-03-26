//! Right-hand sides of the six Painleve equations
//!
//! Each Painleve equation is a second-order ODE of the form y'' = f(t, y, y').
//! This module provides the function `painleve_rhs` that evaluates f for each
//! equation type, as well as the first-order system form used by the ODE solver.
//!
//! ## Mathematical Definitions
//!
//! The six Painleve equations are:
//!
//! - **P-I**:   y'' = 6y^2 + t
//! - **P-II**:  y'' = 2y^3 + ty + alpha
//! - **P-III**: y'' = (y')^2/y - y'/t + (alpha*y^2+beta)/t + gamma*y^3 + delta/y
//! - **P-IV**:  y'' = (y')^2/(2y) + (3/2)y^3 + 4ty^2 + 2(t^2-alpha)y + beta/y
//! - **P-V**:   y'' = ((3y-1)/(2y(y-1)))*(y')^2 - y'/t
//!   + ((y-1)^2/t^2)*(alpha*y + beta/y) + gamma*y/t + delta*y(y+1)/(y-1)
//! - **P-VI**:  y'' = (1/2)(1/y + 1/(y-1) + 1/(y-t))*(y')^2
//!   - (1/t + 1/(t-1) + 1/(y-t))*y'
//!   + y(y-1)(y-t)/(t^2(t-1)^2) * [alpha + beta*t/y^2
//!   + gamma*(t-1)/(y-1)^2 + delta*t(t-1)/(y-t)^2]
//!
//! # References
//!
//! - DLMF Chapter 32: <https://dlmf.nist.gov/32>
//! - Ince, E.L. (1956), *Ordinary Differential Equations*, Chapter XIV

use crate::error::{SpecialError, SpecialResult};
use crate::painleve::types::PainleveEquation;

/// Small threshold to detect near-singularity in denominators.
const SING_TOL: f64 = 1e-30;

/// Evaluate the right-hand side f of the Painleve equation y'' = f(t, y, y').
///
/// # Arguments
///
/// * `eq`  - Which Painleve equation
/// * `t`   - Independent variable
/// * `y`   - Solution value y(t)
/// * `dy`  - Derivative y'(t)
///
/// # Errors
///
/// Returns `SpecialError::DomainError` if the evaluation encounters a singularity
/// (e.g. y = 0 in P-III which has a 1/y term).
pub fn painleve_rhs(eq: &PainleveEquation, t: f64, y: f64, dy: f64) -> SpecialResult<f64> {
    match eq {
        PainleveEquation::PI => painleve_i_rhs(t, y),
        PainleveEquation::PII { alpha } => painleve_ii_rhs(t, y, *alpha),
        PainleveEquation::PIII {
            alpha,
            beta,
            gamma,
            delta,
        } => painleve_iii_rhs(t, y, dy, *alpha, *beta, *gamma, *delta),
        PainleveEquation::PIV { alpha, beta } => painleve_iv_rhs(t, y, dy, *alpha, *beta),
        PainleveEquation::PV {
            alpha,
            beta,
            gamma,
            delta,
        } => painleve_v_rhs(t, y, dy, *alpha, *beta, *gamma, *delta),
        PainleveEquation::PVI {
            alpha,
            beta,
            gamma,
            delta,
        } => painleve_vi_rhs(t, y, dy, *alpha, *beta, *gamma, *delta),
    }
}

/// Evaluate the system form: given state (y, dy), return (dy, ddy).
///
/// This is the form used by the RK45 solver: the second-order ODE is written as
/// the system u1' = u2, u2' = f(t, u1, u2).
pub fn painleve_system(
    eq: &PainleveEquation,
    t: f64,
    y: f64,
    dy: f64,
) -> SpecialResult<(f64, f64)> {
    let ddy = painleve_rhs(eq, t, y, dy)?;
    Ok((dy, ddy))
}

// ---------------------------------------------------------------------------
// Individual equation RHS implementations
// ---------------------------------------------------------------------------

/// P-I: y'' = 6y^2 + t
fn painleve_i_rhs(t: f64, y: f64) -> SpecialResult<f64> {
    Ok(6.0 * y * y + t)
}

/// P-II: y'' = 2y^3 + ty + alpha
fn painleve_ii_rhs(t: f64, y: f64, alpha: f64) -> SpecialResult<f64> {
    Ok(2.0 * y * y * y + t * y + alpha)
}

/// P-III: y'' = (y')^2/y - y'/t + (alpha*y^2 + beta)/t + gamma*y^3 + delta/y
///
/// Singular when y = 0 or t = 0.
fn painleve_iii_rhs(
    t: f64,
    y: f64,
    dy: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> SpecialResult<f64> {
    if y.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve III: y is near zero (singular point)".to_string(),
        ));
    }
    if t.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve III: t is near zero (singular point)".to_string(),
        ));
    }

    let term1 = dy * dy / y;
    let term2 = -dy / t;
    let term3 = (alpha * y * y + beta) / t;
    let term4 = gamma * y * y * y;
    let term5 = delta / y;

    Ok(term1 + term2 + term3 + term4 + term5)
}

/// P-IV: y'' = (y')^2/(2y) + (3/2)y^3 + 4ty^2 + 2(t^2 - alpha)y + beta/y
///
/// Singular when y = 0.
fn painleve_iv_rhs(t: f64, y: f64, dy: f64, alpha: f64, beta: f64) -> SpecialResult<f64> {
    if y.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve IV: y is near zero (singular point)".to_string(),
        ));
    }

    let term1 = dy * dy / (2.0 * y);
    let term2 = 1.5 * y * y * y;
    let term3 = 4.0 * t * y * y;
    let term4 = 2.0 * (t * t - alpha) * y;
    let term5 = beta / y;

    Ok(term1 + term2 + term3 + term4 + term5)
}

/// P-V: y'' = ((3y-1)/(2y(y-1))) * (y')^2 - y'/t
///   + ((y-1)^2 / t^2) * (alpha*y + beta/y) + gamma*y/t + delta*y(y+1)/(y-1)
///
/// Singular when y = 0, y = 1, or t = 0.
fn painleve_v_rhs(
    t: f64,
    y: f64,
    dy: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> SpecialResult<f64> {
    if y.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve V: y is near zero (singular point)".to_string(),
        ));
    }
    if (y - 1.0).abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve V: y is near 1 (singular point)".to_string(),
        ));
    }
    if t.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve V: t is near zero (singular point)".to_string(),
        ));
    }

    let ym1 = y - 1.0;
    let term1 = (3.0 * y - 1.0) / (2.0 * y * ym1) * dy * dy;
    let term2 = -dy / t;
    let term3 = ym1 * ym1 / (t * t) * (alpha * y + beta / y);
    let term4 = gamma * y / t;
    let term5 = delta * y * (y + 1.0) / ym1;

    Ok(term1 + term2 + term3 + term4 + term5)
}

/// P-VI: y'' = (1/2)(1/y + 1/(y-1) + 1/(y-t)) * (y')^2
///   - (1/t + 1/(t-1) + 1/(y-t)) * y'
///   + y(y-1)(y-t) / (t^2(t-1)^2)
///     * [alpha + beta*t/y^2 + gamma*(t-1)/(y-1)^2 + delta*t(t-1)/(y-t)^2]
///
/// Singular when y = 0, y = 1, y = t, t = 0, or t = 1.
fn painleve_vi_rhs(
    t: f64,
    y: f64,
    dy: f64,
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
) -> SpecialResult<f64> {
    if y.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve VI: y is near zero (singular point)".to_string(),
        ));
    }
    if (y - 1.0).abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve VI: y is near 1 (singular point)".to_string(),
        ));
    }
    if (y - t).abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve VI: y is near t (singular point)".to_string(),
        ));
    }
    if t.abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve VI: t is near zero (singular point)".to_string(),
        ));
    }
    if (t - 1.0).abs() < SING_TOL {
        return Err(SpecialError::DomainError(
            "Painleve VI: t is near 1 (singular point)".to_string(),
        ));
    }

    let ym1 = y - 1.0;
    let ymt = y - t;
    let tm1 = t - 1.0;

    // First group: (1/2)(1/y + 1/(y-1) + 1/(y-t)) * (y')^2
    let coeff_dy2 = 0.5 * (1.0 / y + 1.0 / ym1 + 1.0 / ymt);
    let group1 = coeff_dy2 * dy * dy;

    // Second group: -(1/t + 1/(t-1) + 1/(y-t)) * y'
    let coeff_dy = -(1.0 / t + 1.0 / tm1 + 1.0 / ymt);
    let group2 = coeff_dy * dy;

    // Third group: y(y-1)(y-t) / (t^2*(t-1)^2) * [...]
    let prefactor = y * ym1 * ymt / (t * t * tm1 * tm1);
    let bracket =
        alpha + beta * t / (y * y) + gamma * tm1 / (ym1 * ym1) + delta * t * tm1 / (ymt * ymt);
    // Note: delta term uses (1/2 - delta) in some references; here we follow DLMF 32.2.6
    let group3 = prefactor * bracket;

    Ok(group1 + group2 + group3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_rhs_at_zero() {
        // P-I at (t=0, y=0): 6*0^2 + 0 = 0
        let val = painleve_rhs(&PainleveEquation::PI, 0.0, 0.0, 0.0);
        assert!((val.unwrap_or(f64::NAN) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_pi_rhs_known_value() {
        // P-I at (t=1, y=1): 6*1 + 1 = 7
        let val = painleve_rhs(&PainleveEquation::PI, 1.0, 1.0, 0.0);
        assert!((val.unwrap_or(f64::NAN) - 7.0).abs() < 1e-14);
    }

    #[test]
    fn test_pii_rhs_alpha_zero() {
        // P-II with alpha=0 at (t=0, y=1): 2*1 + 0 + 0 = 2
        let eq = PainleveEquation::PII { alpha: 0.0 };
        let val = painleve_rhs(&eq, 0.0, 1.0, 0.0);
        assert!((val.unwrap_or(f64::NAN) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_pii_rhs_general() {
        // P-II at (t=2, y=0.5, alpha=1): 2*(0.125) + 2*0.5 + 1 = 0.25 + 1.0 + 1.0 = 2.25
        let eq = PainleveEquation::PII { alpha: 1.0 };
        let val = painleve_rhs(&eq, 2.0, 0.5, 0.0);
        assert!((val.unwrap_or(f64::NAN) - 2.25).abs() < 1e-14);
    }

    #[test]
    fn test_piii_singular_y_zero() {
        let eq = PainleveEquation::PIII {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 1.0,
        };
        let val = painleve_rhs(&eq, 1.0, 0.0, 1.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_piii_singular_t_zero() {
        let eq = PainleveEquation::PIII {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 1.0,
        };
        let val = painleve_rhs(&eq, 0.0, 1.0, 1.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_piv_rhs() {
        // P-IV at (t=0, y=1, dy=0, alpha=0, beta=0):
        //   0/(2*1) + 1.5*1 + 0 + 0 + 0 = 1.5
        let eq = PainleveEquation::PIV {
            alpha: 0.0,
            beta: 0.0,
        };
        let val = painleve_rhs(&eq, 0.0, 1.0, 0.0);
        assert!((val.unwrap_or(f64::NAN) - 1.5).abs() < 1e-14);
    }

    #[test]
    fn test_piv_singular() {
        let eq = PainleveEquation::PIV {
            alpha: 1.0,
            beta: 1.0,
        };
        let val = painleve_rhs(&eq, 1.0, 0.0, 1.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_pv_singular_y_one() {
        let eq = PainleveEquation::PV {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 1.0,
        };
        let val = painleve_rhs(&eq, 1.0, 1.0, 0.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_pvi_singular_y_equals_t() {
        let eq = PainleveEquation::PVI {
            alpha: 1.0,
            beta: 1.0,
            gamma: 1.0,
            delta: 1.0,
        };
        // y = t = 0.5
        let val = painleve_rhs(&eq, 0.5, 0.5, 0.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_pvi_singular_t_zero() {
        let eq = PainleveEquation::PVI {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 0.0,
        };
        let val = painleve_rhs(&eq, 0.0, 0.5, 0.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_pvi_singular_t_one() {
        let eq = PainleveEquation::PVI {
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
            delta: 0.0,
        };
        let val = painleve_rhs(&eq, 1.0, 0.5, 0.0);
        assert!(val.is_err());
    }

    #[test]
    fn test_painleve_system_pi() {
        let eq = PainleveEquation::PI;
        let (u1p, u2p) = painleve_system(&eq, 1.0, 0.0, 1.0).unwrap_or((f64::NAN, f64::NAN));
        // u1' = dy = 1.0
        assert!((u1p - 1.0).abs() < 1e-14);
        // u2' = 6*0^2 + 1 = 1.0
        assert!((u2p - 1.0).abs() < 1e-14);
    }
}
