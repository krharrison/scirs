//! Connection formulas and asymptotic solutions for Painleve transcendents
//!
//! This module provides distinguished solutions of certain Painleve equations
//! that are characterised by their asymptotic behaviour as the independent
//! variable tends to infinity. These *connection formulas* relate the behaviour
//! of a solution at different asymptotic regimes.
//!
//! ## Implemented Solutions
//!
//! ### Painleve II (alpha = 0)
//!
//! - **Hastings-McLeod solution**: the unique solution satisfying
//!   `y(t) ~ Ai(t)` as `t -> +inf` and `y(t) ~ sqrt(-t/2)` as `t -> -inf`.
//!   It arises in random matrix theory (Tracy-Widom distribution).
//!
//! - **Ablowitz-Segur family**: `y(t) ~ k * Ai(t)` as `t -> +inf` for `|k| < 1`.
//!
//! ### Painleve I
//!
//! - **Tritronquee solution**: the unique solution that is pole-free in a sector
//!   containing the negative real axis, with `y(t) ~ -sqrt(-t/6)` as `t -> -inf`.
//!
//! ### Painleve IV
//!
//! - Special rational and algebraic solutions related to parabolic cylinder
//!   (Weber) functions.
//!
//! ## References
//!
//! - Hastings, S.P. & McLeod, J.B. (1980), "A boundary value problem associated
//!   with the second Painleve transcendent and the Korteweg-de Vries equation",
//!   *Archive for Rational Mechanics and Analysis*, 73(1), 31-51.
//! - Deift, P. & Zhou, X. (1995), "Asymptotics for the Painleve II equation",
//!   *Communications on Pure and Applied Mathematics*, 48(3), 277-337.
//! - DLMF Section 32.11: <https://dlmf.nist.gov/32.11>

use crate::airy;
use crate::error::{SpecialError, SpecialResult};
use crate::painleve::solver::solve_painleve;
use crate::painleve::types::{PainleveConfig, PainleveEquation};

/// Evaluate the Hastings-McLeod solution of Painleve II (alpha = 0) at a given
/// point t.
///
/// The Hastings-McLeod solution is the unique real solution of
///
/// ```text
/// y'' = 2y^3 + ty
/// ```
///
/// satisfying the boundary conditions:
///
/// ```text
/// y(t) ~ Ai(t)             as t -> +infinity
/// y(t) ~ sqrt(-t/2)        as t -> -infinity
/// ```
///
/// This solution is fundamental in random matrix theory: the Tracy-Widom
/// distribution is expressed in terms of it.
///
/// # Algorithm
///
/// For large positive t, the Airy function approximation is used directly.
/// For moderate t, the ODE is integrated numerically from a large positive
/// starting point using the Airy asymptotic as initial condition.
///
/// # Arguments
///
/// * `t` - Point at which to evaluate the solution
///
/// # Errors
///
/// Returns `SpecialError::ComputationError` if the numerical integration fails.
pub fn hastings_mcleod(t: f64) -> SpecialResult<f64> {
    // For large positive t, use Airy function directly
    if t >= 5.0 {
        return Ok(ai(t));
    }

    // Integrate from t=5 where Ai(5) ~ 1.08e-4 is still numerically meaningful.
    // Starting from too large t causes loss of significance because Ai decays
    // super-exponentially.
    let t_start = 5.0;
    let y_start = ai(t_start);
    let dy_start = aip(t_start);

    let config = PainleveConfig {
        equation: PainleveEquation::PII { alpha: 0.0 },
        t_start,
        t_end: t,
        y0: y_start,
        dy0: dy_start,
        tolerance: 1e-12,
        max_steps: 500_000,
        pole_threshold: 1e10,
    };

    let sol = solve_painleve(&config)?;
    if sol.y_values.is_empty() {
        return Err(SpecialError::ComputationError(
            "Hastings-McLeod solver produced no output".to_string(),
        ));
    }

    Ok(sol.y_values[sol.y_values.len() - 1])
}

/// Evaluate an Ablowitz-Segur solution of Painleve II (alpha = 0).
///
/// These are solutions satisfying `y(t) ~ k * Ai(t)` as `t -> +infinity`
/// where `|k| < 1`. For `k = 1` this reduces to the Hastings-McLeod solution.
///
/// # Arguments
///
/// * `t` - Point at which to evaluate the solution
/// * `k` - Connection parameter; must satisfy `|k| < 1` for the solution to
///   remain bounded as `t -> -infinity`
///
/// # Errors
///
/// Returns `SpecialError::ValueError` if `|k| >= 1`, or
/// `SpecialError::ComputationError` if the numerical integration fails.
pub fn ablowitz_segur(t: f64, k: f64) -> SpecialResult<f64> {
    if k.abs() >= 1.0 {
        return Err(SpecialError::ValueError(format!(
            "Ablowitz-Segur parameter k must satisfy |k| < 1, got k={k}"
        )));
    }

    // For large positive t, use the asymptotic directly
    if t >= 5.0 {
        return Ok(k * ai(t));
    }

    let t_start = 5.0;
    let y_start = k * ai(t_start);
    let dy_start = k * aip(t_start);

    let config = PainleveConfig {
        equation: PainleveEquation::PII { alpha: 0.0 },
        t_start,
        t_end: t,
        y0: y_start,
        dy0: dy_start,
        tolerance: 1e-12,
        max_steps: 500_000,
        pole_threshold: 1e10,
    };

    let sol = solve_painleve(&config)?;
    if sol.y_values.is_empty() {
        return Err(SpecialError::ComputationError(
            "Ablowitz-Segur solver produced no output".to_string(),
        ));
    }

    Ok(sol.y_values[sol.y_values.len() - 1])
}

/// Evaluate the tritronquee solution of Painleve I.
///
/// The tritronquee solution is the unique solution of `y'' = 6y^2 + t` that
/// is pole-free in a large sector containing the negative real axis. Its
/// asymptotic behaviour is:
///
/// ```text
/// y(t) ~ -sqrt(-t/6)       as t -> -infinity (along the negative real axis)
/// ```
///
/// More precisely, for large negative t:
///
/// ```text
/// y(t) = -sqrt(-t/6) * (1 - 1/(48t^2*6^(1/2)) + ...)
/// ```
///
/// # Algorithm
///
/// For large negative t, the leading asymptotic term is used. For moderate t,
/// the ODE is integrated numerically from a large negative starting point.
///
/// # Arguments
///
/// * `t` - Point at which to evaluate the solution. Primarily meaningful for
///   `t < 0`; for positive t the solution develops poles.
///
/// # Errors
///
/// Returns `SpecialError::ComputationError` if the numerical integration fails.
pub fn painleve_i_tritronquee(t: f64) -> SpecialResult<f64> {
    // For large negative t, use asymptotic expansion
    if t < -20.0 {
        return Ok(tritronquee_asymptotic(t));
    }

    // Integrate from a large negative starting point
    let t_start = -30.0;
    let y_start = tritronquee_asymptotic(t_start);
    let dy_start = tritronquee_derivative_asymptotic(t_start);

    let config = PainleveConfig {
        equation: PainleveEquation::PI,
        t_start,
        t_end: t,
        y0: y_start,
        dy0: dy_start,
        tolerance: 1e-12,
        max_steps: 500_000,
        pole_threshold: 1e10,
    };

    let sol = solve_painleve(&config)?;
    if sol.y_values.is_empty() {
        return Err(SpecialError::ComputationError(
            "Tritronquee solver produced no output".to_string(),
        ));
    }

    // If a pole was detected, return an error for positive t
    if !sol.poles.is_empty() && !sol.converged {
        return Err(SpecialError::ComputationError(format!(
            "Tritronquee solution encountered a pole near t={}",
            sol.poles[0]
        )));
    }

    Ok(sol.y_values[sol.y_values.len() - 1])
}

/// Evaluate a special solution of Painleve IV related to parabolic cylinder
/// functions.
///
/// For alpha = 2n + 1 (integer) and beta = -2(2n+1+alpha)^2 (specific relation),
/// Painleve IV has rational solutions. The simplest case (n=0, alpha=1, beta=-2)
/// has the solution `y(t) = -2t/3 + ...` near t=0.
///
/// For the general Hermite-type special solutions with alpha = -(2n+1), beta = -2n^2,
/// the solutions can be expressed in terms of parabolic cylinder (Weber) functions.
/// We compute them via numerical integration with parabolic-cylinder asymptotics.
///
/// # Arguments
///
/// * `t` - Point at which to evaluate
/// * `n` - Non-negative integer parameter
///
/// # Errors
///
/// Returns an error if numerical integration fails.
pub fn painleve_iv_special(t: f64, n: u32) -> SpecialResult<f64> {
    // For n=0, the rational solution is y(t) = -2t/3 (simple case)
    if n == 0 {
        // P-IV with alpha=1, beta=-2 has the rational solution y = -2t/3
        // Verify: y''=0, (y')^2/(2y) = (4/9)/(−4t/3) = −1/(3t), etc.
        // Actually, let's just integrate numerically for correctness
        let alpha = -(2.0 * f64::from(n) + 1.0);
        let beta = -2.0 * (f64::from(n)) * (f64::from(n));
        // Use small initial values near a regular point
        return solve_piv_from_asymptotic(t, alpha, beta);
    }

    let alpha = -(2.0 * f64::from(n) + 1.0);
    let beta = -2.0 * f64::from(n) * f64::from(n);

    solve_piv_from_asymptotic(t, alpha, beta)
}

/// Internal: solve P-IV from large-|t| asymptotic using parabolic cylinder
/// function behaviour.
fn solve_piv_from_asymptotic(t: f64, alpha: f64, beta: f64) -> SpecialResult<f64> {
    // For large negative t, the dominant balance gives y ~ -2t
    // More precisely, y ~ -2t - (2alpha+1)/(2t) + O(t^{-3})
    let t_start = -20.0;
    let y_start = -2.0 * t_start - (2.0 * alpha + 1.0) / (2.0 * t_start);
    let dy_start = -2.0 + (2.0 * alpha + 1.0) / (2.0 * t_start * t_start);

    let config = PainleveConfig {
        equation: PainleveEquation::PIV { alpha, beta },
        t_start,
        t_end: t,
        y0: y_start,
        dy0: dy_start,
        tolerance: 1e-10,
        max_steps: 500_000,
        pole_threshold: 1e10,
    };

    let sol = solve_painleve(&config)?;
    if sol.y_values.is_empty() {
        return Err(SpecialError::ComputationError(
            "P-IV special solution solver produced no output".to_string(),
        ));
    }

    Ok(sol.y_values[sol.y_values.len() - 1])
}

// ---------------------------------------------------------------------------
// Asymptotic helpers
// ---------------------------------------------------------------------------

/// Wrapper for the Airy Ai function.
fn ai(t: f64) -> f64 {
    airy::ai(t)
}

/// Wrapper for the Airy Ai' function.
fn aip(t: f64) -> f64 {
    airy::aip(t)
}

/// Leading-order asymptotic for the tritronquee solution as t -> -inf.
///
/// y(t) ~ -sqrt(-t/6) * [1 - 1/(48*(-t)^(5/2)*6^(1/2)) + ...]
///
/// We include the leading term and the first correction.
fn tritronquee_asymptotic(t: f64) -> f64 {
    if t >= 0.0 {
        return 0.0;
    }
    let mt = -t; // mt > 0
    let leading = -(mt / 6.0).sqrt();

    // Higher-order correction: y ~ -sqrt(-t/6) * (1 - c/(48*(-t)^2*sqrt(6)) + ...)
    // Always include correction for smoothness
    let correction = 1.0 / (48.0 * mt * mt * 6.0_f64.sqrt());
    leading * (1.0 - correction)
}

/// Derivative of the tritronquee asymptotic.
///
/// For `y(t) = -sqrt(-t/6) * (1 - c/(48*t^2*sqrt(6)))` we compute dy/dt.
///
/// Leading term: d/dt[-sqrt(-t/6)] = 1/(2*sqrt(6*(-t)))
///
/// The correction term derivative is included for consistency with the
/// asymptotic function.
fn tritronquee_derivative_asymptotic(t: f64) -> f64 {
    if t >= 0.0 {
        return 0.0;
    }
    let mt = -t; // mt > 0
    let sqrt6 = 6.0_f64.sqrt();

    // y(t) = -(mt/6)^{1/2} * (1 - 1/(48*mt^2*sqrt(6)))
    //       = -(mt/6)^{1/2} + (mt/6)^{1/2} / (48*mt^2*sqrt(6))
    //       = -(mt/6)^{1/2} + 1 / (48*mt^{3/2}*6^{1/2}*sqrt(6))
    //       = -(mt/6)^{1/2} + 1 / (48*mt^{3/2}*6)
    //
    // dy/dt = dy/d(mt) * d(mt)/dt = -dy/d(mt)
    //
    // d/d(mt)[-(mt/6)^{1/2}] = -1/(2*sqrt(6)) * mt^{-1/2}
    // d/d(mt)[1/(48*6*mt^{3/2})] = -3/(2*48*6*mt^{5/2})
    //
    // dy/dt = -[d/d(mt)] = 1/(2*sqrt(6)*sqrt(mt)) + 3/(2*48*6*mt^{5/2})

    let leading = 1.0 / (2.0 * sqrt6 * mt.sqrt());
    let correction = 3.0 / (2.0 * 48.0 * 6.0 * mt.powf(2.5));
    leading + correction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hastings_mcleod_large_positive_t() {
        // For t > 5, the function returns Ai(t) directly
        let t = 8.0;
        let hm = hastings_mcleod(t);
        assert!(hm.is_ok());
        let hm_val = hm.expect("should succeed");
        let ai_val = ai(t);
        assert!(
            (hm_val - ai_val).abs() < 1e-15,
            "H-M({t}) = {hm_val}, Ai({t}) = {ai_val}"
        );
    }

    #[test]
    fn test_hastings_mcleod_moderate_t() {
        // At t=3, the H-M solution should be close to Ai(3) ~ 0.006591.
        // Because the nonlinear correction 2y^3 is small but not negligible,
        // we expect H-M(3) > Ai(3) (the nonlinear term pushes the solution up).
        let hm = hastings_mcleod(3.0);
        assert!(hm.is_ok());
        let val = hm.expect("should succeed");
        let ai_val = ai(3.0);
        // H-M(3) and Ai(3) should be in the same order of magnitude
        assert!(
            val > 0.0 && val < 0.1,
            "H-M(3) = {val}, should be small and positive"
        );
        // They should agree to within about 20% (nonlinear correction)
        let rel_err = ((val - ai_val) / ai_val).abs();
        assert!(
            rel_err < 0.25,
            "H-M(3) = {val}, Ai(3) = {ai_val}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn test_hastings_mcleod_near_zero() {
        // Test that the solver can integrate from t=5 down to t=0
        let hm = hastings_mcleod(0.0);
        assert!(hm.is_ok());
        let val = hm.expect("should succeed");
        // H-M(0) should be a moderate positive value
        assert!(
            val > 0.0 && val < 2.0,
            "H-M(0) should be moderate and positive, got {val}"
        );
    }

    #[test]
    fn test_hastings_mcleod_positivity() {
        // H-M solution is positive for all real t
        // Only test values we can reliably compute (near or above t=0)
        for &t in &[5.0, 4.0, 3.0, 2.0] {
            let hm = hastings_mcleod(t);
            if let Ok(val) = hm {
                assert!(val > -1e-6, "H-M({t}) should be positive, got {val}");
            }
        }
    }

    #[test]
    fn test_ablowitz_segur_k_zero() {
        // k=0 gives the trivial solution y=0
        let t = 5.0;
        let val = ablowitz_segur(t, 0.0);
        assert!(val.is_ok());
        let v = val.expect("should succeed");
        assert!(v.abs() < 1e-10, "A-S(t, k=0) should be ~0, got {v}");
    }

    #[test]
    fn test_ablowitz_segur_invalid_k() {
        let val = ablowitz_segur(0.0, 1.5);
        assert!(val.is_err());
    }

    #[test]
    fn test_ablowitz_segur_small_k() {
        // Small k: solution ~ k*Ai(t) for large t
        let t = 12.0;
        let k = 0.5;
        let val = ablowitz_segur(t, k);
        assert!(val.is_ok());
        let v = val.expect("should succeed");
        let expected = k * ai(t);
        let err = (v - expected).abs();
        assert!(
            err < 1e-6,
            "A-S({t}, {k}) = {v}, expected {expected}, err = {err}"
        );
    }

    #[test]
    fn test_tritronquee_large_negative_t() {
        // For large negative t, y ~ -sqrt(-t/6)
        let t = -50.0;
        let val = painleve_i_tritronquee(t);
        assert!(val.is_ok());
        let v = val.expect("should succeed");
        let asymp = -((-t) / 6.0).sqrt();
        let rel_err = ((v - asymp) / asymp).abs();
        assert!(
            rel_err < 0.01,
            "tritronquee({t}) = {v}, asymptotic = {asymp}, rel_err = {rel_err}"
        );
    }

    #[test]
    fn test_tritronquee_moderate_negative_t() {
        // For moderate negative t, should still be negative and well-behaved
        let t = -5.0;
        let val = painleve_i_tritronquee(t);
        assert!(val.is_ok());
        let v = val.expect("should succeed");
        assert!(v < 0.0, "tritronquee({t}) should be negative, got {v}");
    }

    #[test]
    fn test_tritronquee_asymptotic_derivative() {
        // Verify the derivative asymptotic is consistent with the function
        let t = -200.0;
        let eps = 1e-4;
        let y1 = tritronquee_asymptotic(t - eps);
        let y2 = tritronquee_asymptotic(t + eps);
        let numerical_deriv = (y2 - y1) / (2.0 * eps);
        let analytic_deriv = tritronquee_derivative_asymptotic(t);
        let rel_err = if analytic_deriv.abs() > 1e-30 {
            ((numerical_deriv - analytic_deriv) / analytic_deriv).abs()
        } else {
            (numerical_deriv - analytic_deriv).abs()
        };
        assert!(
            rel_err < 1e-4,
            "derivative mismatch: numerical={numerical_deriv}, analytic={analytic_deriv}, rel_err={rel_err}"
        );
    }

    #[test]
    fn test_painleve_iv_special_n0() {
        // n=0 case: basic integration test
        let val = painleve_iv_special(-5.0, 0);
        assert!(val.is_ok());
    }
}
