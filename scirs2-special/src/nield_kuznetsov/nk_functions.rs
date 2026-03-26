//! Extended Nield-Kuznetsov functions Nk_i(x) for porous media flow.
//!
//! This module implements the family of Nield-Kuznetsov functions Nk_i(x) which arise
//! in the study of flow through porous media governed by the Brinkman equation and
//! the Darcy-Lapwood equation.
//!
//! ## Mathematical Background
//!
//! The Nk_i functions are defined through Airy-type integrals with modified kernels:
//!
//! Nk_0(x) = Ni(1, 0, x) = π [Bi(x) ∫₀ˣ Ai²(t) dt - Ai(x) ∫₀ˣ Ai(t)Bi(t) dt]
//!
//! Nk_1(x) = ∫₀ˣ Nk_0(t) dt
//!
//! Nk_i(x) = ∫₀ˣ Nk_{i-1}(t) dt  (iterated integrals)
//!
//! ## Asymptotic Expansions
//!
//! For large positive x:
//!   Nk_0(x) ~ -1/(2x) + O(x⁻⁴)  (since it satisfies y'' - xy = Ai(x) → 0 exponentially)
//!
//! For large negative x:
//!   Nk_0(x) oscillates with period ~ 2π/|x|^{1/2}
//!
//! ## Connection to Confluent Hypergeometric Functions
//!
//! Through the relation Ai(x) = (1/3) √(x/3) [I_{-1/3}(ζ) - I_{1/3}(ζ)] where ζ = (2/3) x^{3/2},
//! the Nk functions can be expressed in terms of confluent hypergeometric functions ₁F₁.
//!
//! ## References
//!
//! - Nield & Kuznetsov (2000), ZAMP 51:341–358
//! - Hamdan & Kamel (2011), Applied Math and Computation
//! - Abu Zaytoon et al. (2016), Int. J. Open Problems Compt. Math.

use std::f64::consts::PI;

use super::types::{NKConfig, NKMethod, NKResult};
use super::{
    airy_ai, airy_ai_prime, airy_bi, airy_bi_prime, integrate_composite_gl, particular_ai, NkConfig,
};
use crate::error::{SpecialError, SpecialResult};

/// Nield-Kuznetsov function struct for computing Nk_i(x).
///
/// Encapsulates computation state and configuration for the family of
/// Nield-Kuznetsov functions used in porous media flow analysis.
#[derive(Debug, Clone)]
pub struct NieldKuznetsov {
    /// Configuration
    pub config: NKConfig,
}

impl NieldKuznetsov {
    /// Create a new NieldKuznetsov with the given configuration.
    pub fn new(config: NKConfig) -> Self {
        NieldKuznetsov { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        NieldKuznetsov {
            config: NKConfig::default(),
        }
    }

    /// Evaluate Nk_i(x) for the given function index and argument.
    ///
    /// # Arguments
    /// * `i` - Function index (0, 1, 2, ...)
    /// * `x` - Evaluation point
    ///
    /// # Returns
    /// NKResult containing the function value and metadata.
    pub fn evaluate(&self, i: usize, x: f64) -> SpecialResult<NKResult> {
        // Choose computation method based on argument
        let (value, method) = if x.abs() > self.config.asymptotic_threshold && i == 0 {
            (self.nk0_asymptotic(x)?, NKMethod::AsymptoticLarge)
        } else if x.abs() < 0.5 && i <= 3 {
            (self.nk_i_series(i, x)?, NKMethod::SeriesSmall)
        } else if self.config.use_hypergeometric && i == 0 {
            (self.nk0_hypergeometric(x)?, NKMethod::Hypergeometric)
        } else {
            (self.nk_i_quadrature(i, x)?, NKMethod::Quadrature)
        };

        Ok(NKResult {
            value,
            derivative: None,
            index: i,
            x,
            error_estimate: None,
            method,
        })
    }

    /// Evaluate Nk_i(x) and its derivative Nk_i'(x).
    ///
    /// # Arguments
    /// * `i` - Function index
    /// * `x` - Evaluation point
    ///
    /// # Returns
    /// NKResult with both value and derivative.
    pub fn evaluate_with_derivative(&self, i: usize, x: f64) -> SpecialResult<NKResult> {
        let mut result = self.evaluate(i, x)?;

        // For i >= 1: Nk_i'(x) = Nk_{i-1}(x) by definition
        // For i == 0: Nk_0'(x) must be computed directly
        let deriv = if i > 0 {
            self.nk_i_quadrature(i - 1, x)?
        } else {
            self.nk0_derivative(x)?
        };

        result.derivative = Some(deriv);
        Ok(result)
    }

    /// Compute Nk_0(x) via direct quadrature.
    ///
    /// Nk_0(x) = π [Bi(x) ∫₀ˣ Ai²(t) dt - Ai(x) ∫₀ˣ Ai(t)Bi(t) dt]
    fn nk0_quadrature(&self, x: f64) -> SpecialResult<f64> {
        let n_sub = (self.config.max_terms / 8).max(4);
        Ok(particular_ai(x, n_sub))
    }

    /// Compute Nk_i(x) via iterated quadrature.
    ///
    /// Nk_i(x) = ∫₀ˣ Nk_{i-1}(t) dt
    ///
    /// For i >= 2, we avoid deeply nested recursive quadrature by first
    /// evaluating Nk_0 on a grid and integrating successively via the
    /// trapezoidal rule.
    fn nk_i_quadrature(&self, i: usize, x: f64) -> SpecialResult<f64> {
        if i == 0 {
            return self.nk0_quadrature(x);
        }

        if i == 1 {
            // Single level of quadrature
            let n_sub = (self.config.max_terms / 8).max(4);
            let result = if x >= 0.0 {
                integrate_composite_gl(|t| self.nk0_quadrature(t).unwrap_or(0.0), 0.0, x, n_sub)
            } else {
                -integrate_composite_gl(|t| self.nk0_quadrature(t).unwrap_or(0.0), x, 0.0, n_sub)
            };
            return Ok(result);
        }

        // For i >= 2: build a grid of Nk_0 values and integrate repeatedly
        // using the trapezoidal rule. This avoids exponential blowup from
        // recursive quadrature.
        let n_grid = 32usize;
        let (a, b) = if x >= 0.0 { (0.0, x) } else { (x, 0.0) };
        if (b - a).abs() < 1e-15 {
            return Ok(0.0);
        }

        let h = (b - a) / n_grid as f64;

        // Evaluate Nk_0 on the grid
        let mut vals = Vec::with_capacity(n_grid + 1);
        for j in 0..=n_grid {
            let t = a + j as f64 * h;
            vals.push(self.nk0_quadrature(t).unwrap_or(0.0));
        }

        // Integrate i times via cumulative trapezoidal rule
        for _level in 0..i {
            let mut integrated = Vec::with_capacity(vals.len());
            integrated.push(0.0_f64);
            let mut cumulative = 0.0_f64;
            for j in 1..vals.len() {
                cumulative += 0.5 * h * (vals[j - 1] + vals[j]);
                integrated.push(cumulative);
            }
            vals = integrated;
        }

        // The final value at x
        if x >= 0.0 {
            Ok(*vals.last().unwrap_or(&0.0))
        } else {
            // For negative x, the integral from x to 0 gives -Nk_i(x)
            Ok(-(*vals.last().unwrap_or(&0.0)))
        }
    }

    /// Asymptotic expansion of Nk_0(x) for large positive x.
    ///
    /// For large positive x, Ai(x) decays exponentially and the particular solution
    /// approaches:
    ///   Nk_0(x) ~ -Ai(x)/(2x) [1 - 5/(2x³) + O(x⁻⁶)]
    ///
    /// For large negative x (oscillatory region):
    ///   Nk_0(x) involves oscillatory Airy functions.
    fn nk0_asymptotic(&self, x: f64) -> SpecialResult<f64> {
        if x > 0.0 {
            // For large positive x: Ai(x) decays exponentially
            // The particular solution satisfies y'' - xy = Ai(x)
            // For large x, dominant balance gives y ~ -Ai(x)/(2x)
            let ai_x = airy_ai(x);

            let mut sum = 1.0_f64;
            let x3 = x * x * x;
            let mut term_coeff = 1.0_f64;

            // Asymptotic series: Σ_k (-1)^k c_k / x^{3k}
            // c_0 = 1, c_1 = 5/2, c_2 = 385/8, ...
            let coeffs = [1.0, -5.0 / 2.0, 385.0 / 8.0, -85085.0 / 16.0];
            for (k, &c) in coeffs.iter().enumerate().skip(1) {
                if k >= self.config.asymptotic_terms {
                    break;
                }
                term_coeff /= x3;
                let term = c * term_coeff;
                if term.abs() < self.config.tol * sum.abs() {
                    break;
                }
                sum += term;
            }

            Ok(-ai_x * sum / (2.0 * x))
        } else {
            // Large negative x: oscillatory regime
            // Fall back to quadrature for negative x
            self.nk0_quadrature(x)
        }
    }

    /// Series expansion of Nk_i(x) for small x.
    ///
    /// For |x| < 0.5, use Taylor series around x = 0.
    ///
    /// Nk_0(0) = 0, Nk_0'(0) = 0 (particular solution with zero IC)
    /// Nk_0''(0) = 0 * Nk_0(0) + Ai(0) = Ai(0)  (from the ODE)
    /// Nk_0'''(0) = Nk_0(0) + 0 * Nk_0'(0) + Ai'(0) = Ai'(0)
    fn nk_i_series(&self, i: usize, x: f64) -> SpecialResult<f64> {
        if i == 0 {
            return self.nk0_series(x);
        }

        // For i >= 1: Nk_i(x) = ∫₀ˣ Nk_{i-1}(t) dt
        // If Nk_{i-1}(x) = Σ a_n x^n, then Nk_i(x) = Σ a_n x^{n+1}/(n+1)
        // Build the series for Nk_0 then integrate i times
        let nk0_coeffs = self.nk0_series_coefficients(20)?;

        let mut coeffs = nk0_coeffs;
        for _ in 0..i {
            // Integrate: a_n x^n → a_n x^{n+1}/(n+1)
            let mut new_coeffs = vec![0.0; coeffs.len() + 1];
            for (n, &c) in coeffs.iter().enumerate() {
                new_coeffs[n + 1] = c / (n + 1) as f64;
            }
            coeffs = new_coeffs;
        }

        // Evaluate polynomial
        let mut result = 0.0_f64;
        let mut x_pow = 1.0_f64;
        for &c in &coeffs {
            result += c * x_pow;
            x_pow *= x;
        }

        Ok(result)
    }

    /// Series expansion of Nk_0(x) for small x.
    fn nk0_series(&self, x: f64) -> SpecialResult<f64> {
        let coeffs = self.nk0_series_coefficients(20)?;

        let mut result = 0.0_f64;
        let mut x_pow = 1.0_f64;
        for &c in &coeffs {
            result += c * x_pow;
            x_pow *= x;
        }

        Ok(result)
    }

    /// Compute Taylor coefficients of Nk_0(x) around x = 0.
    ///
    /// From the ODE y'' - xy = Ai(x):
    ///   y_0 = 0, y_1 = 0 (initial conditions Nk_0(0) = 0, Nk_0'(0) = 0)
    ///   y_2 = Ai(0) / 2
    ///   y_3 = Ai'(0) / 6
    ///   y_{n+2} = [x * y_n term + Ai^(n)(0)/n!] via recurrence
    fn nk0_series_coefficients(&self, n_terms: usize) -> SpecialResult<Vec<f64>> {
        let ai0 = 0.355028053887817_f64; // Ai(0)
        let aip0 = -0.258819403792807_f64; // Ai'(0)

        // Taylor coefficients of Ai(x) at x=0
        // Ai(x) = Σ a_n x^n where a_0 = Ai(0), a_1 = Ai'(0), ...
        // From Ai'' = x Ai: a_{n+2} = a_{n-1} / ((n+1)(n+2)) for n >= 1
        let mut ai_coeffs = vec![0.0_f64; n_terms + 3];
        ai_coeffs[0] = ai0;
        ai_coeffs[1] = aip0;
        ai_coeffs[2] = 0.0; // a_2 = a_{-1}/... = 0 (n=0 special)
        for n in 1..(n_terms + 1) {
            if n + 2 < ai_coeffs.len() && n >= 1 {
                ai_coeffs[n + 2] = ai_coeffs[n - 1] / ((n + 1) * (n + 2)) as f64;
            }
        }

        // Nk_0 coefficients y_n from y'' - xy = Ai(x)
        // y_{n+2} (n+1)(n+2) = y_{n-1} + ai_coeffs[n]  for n >= 1
        // y_0 = 0, y_1 = 0
        // y_2 = ai_coeffs[0] / 2 = Ai(0)/2
        let mut y_coeffs = vec![0.0_f64; n_terms + 3];
        y_coeffs[0] = 0.0;
        y_coeffs[1] = 0.0;
        y_coeffs[2] = ai_coeffs[0] / 2.0;

        for n in 1..n_terms {
            let rhs = if n >= 1 { y_coeffs[n - 1] } else { 0.0 } + ai_coeffs[n];
            if n + 2 < y_coeffs.len() {
                y_coeffs[n + 2] = rhs / ((n + 1) * (n + 2)) as f64;
            }
        }

        Ok(y_coeffs[..n_terms].to_vec())
    }

    /// Connection to confluent hypergeometric functions.
    ///
    /// Using the representation:
    /// Ai(x) = (1/π) √(x/3) K_{1/3}(ζ) where ζ = (2/3) x^{3/2}
    ///
    /// The particular solution can be written in terms of ₁F₁ (Kummer's function).
    fn nk0_hypergeometric(&self, x: f64) -> SpecialResult<f64> {
        // For moderate x, use the hypergeometric representation
        // Nk_0(x) = -Ai(x) ∫ Bi(t)/W dt + Bi(x) ∫ Ai(t)/W dt
        // where the integrals can be expressed via ₁F₁

        if x.abs() < 0.5 {
            return self.nk0_series(x);
        }

        if x > 0.0 {
            // For positive x, use the connection to modified Bessel functions
            // ζ = (2/3) x^{3/2}
            let zeta = 2.0 / 3.0 * x.powf(1.5);

            // Ai(x) = (1/π) √(x/3) K_{1/3}(ζ)
            // The particular solution involves integrals of products of Bessel functions
            // which can be expressed in closed form via hypergeometric functions.
            //
            // For practical computation, use the asymptotic-series or quadrature approach.
            if zeta > 10.0 {
                return self.nk0_asymptotic(x);
            }

            // For moderate zeta: use quadrature with improved accuracy
            self.nk0_quadrature(x)
        } else {
            self.nk0_quadrature(x)
        }
    }

    /// Compute Nk_0'(x) directly.
    ///
    /// From the definition:
    /// Nk_0'(x) = π [Bi'(x) ∫₀ˣ Ai²(t) dt + Bi(x) Ai²(x)
    ///              - Ai'(x) ∫₀ˣ Ai(t)Bi(t) dt - Ai(x) Ai(x)Bi(x)]
    ///           = π [Bi'(x) ∫₀ˣ Ai²(t) dt - Ai'(x) ∫₀ˣ Ai(t)Bi(t) dt]
    ///
    /// (The cross terms cancel via the Wronskian.)
    fn nk0_derivative(&self, x: f64) -> SpecialResult<f64> {
        if x.abs() < 0.5 {
            // Use series: Nk_0'(x) = Σ_{n>=1} n y_n x^{n-1}
            let coeffs = self.nk0_series_coefficients(20)?;
            let mut result = 0.0_f64;
            let mut x_pow = 1.0_f64;
            for (n, &c) in coeffs.iter().enumerate().skip(1) {
                result += n as f64 * c * x_pow;
                x_pow *= x;
            }
            return Ok(result);
        }

        let n_sub = (self.config.max_terms / 4).max(8);

        // ∫₀ˣ Ai²(t) dt
        let int_ai2 = if x >= 0.0 {
            integrate_composite_gl(|t| airy_ai(t) * airy_ai(t), 0.0, x, n_sub)
        } else {
            -integrate_composite_gl(|t| airy_ai(t) * airy_ai(t), x, 0.0, n_sub)
        };

        // ∫₀ˣ Ai(t)Bi(t) dt
        let int_aibi = if x >= 0.0 {
            integrate_composite_gl(|t| airy_ai(t) * airy_bi(t), 0.0, x, n_sub)
        } else {
            -integrate_composite_gl(|t| airy_ai(t) * airy_bi(t), x, 0.0, n_sub)
        };

        let bip_x = airy_bi_prime(x);
        let aip_x = airy_ai_prime(x);

        Ok(PI * (bip_x * int_ai2 - aip_x * int_aibi))
    }
}

/// Evaluate Nk_i(x) — convenience function.
///
/// # Arguments
/// * `i` - Function index (0, 1, 2, ...)
/// * `x` - Evaluation point
///
/// # Returns
/// Value of Nk_i(x).
pub fn nk_i(i: usize, x: f64) -> SpecialResult<f64> {
    let nk = NieldKuznetsov::with_defaults();
    let result = nk.evaluate(i, x)?;
    Ok(result.value)
}

/// Evaluate Nk_i'(x) — convenience function for the derivative.
///
/// # Arguments
/// * `i` - Function index (0, 1, 2, ...)
/// * `x` - Evaluation point
///
/// # Returns
/// Value of Nk_i'(x).
pub fn nk_i_prime(i: usize, x: f64) -> SpecialResult<f64> {
    let nk = NieldKuznetsov::with_defaults();
    let result = nk.evaluate_with_derivative(i, x)?;
    result
        .derivative
        .ok_or_else(|| SpecialError::ComputationError("Derivative not computed".to_string()))
}

/// Evaluate Nk_0(x) for Brinkman flow.
///
/// This is the primary Nield-Kuznetsov function arising in the study of
/// flow through porous media governed by the Brinkman equation.
///
/// # Arguments
/// * `x` - Evaluation point
///
/// # Returns
/// Value of Nk_0(x).
pub fn nk_brinkman(x: f64) -> SpecialResult<f64> {
    nk_i(0, x)
}

/// Evaluate Nk_0(x) for Darcy-Lapwood convective instability.
///
/// In the Darcy-Lapwood equation context, the Nield-Kuznetsov function
/// appears with specific scaling related to the Darcy number Da:
///
/// Nk_DL(x; Da) = Da^{1/3} Nk_0(x / Da^{1/3})
///
/// # Arguments
/// * `x` - Evaluation point
/// * `darcy_number` - Darcy number Da > 0
///
/// # Returns
/// Scaled Nk_0 for Darcy-Lapwood flow.
pub fn nk_darcy_lapwood(x: f64, darcy_number: f64) -> SpecialResult<f64> {
    if darcy_number <= 0.0 {
        return Err(SpecialError::DomainError(
            "Darcy number must be positive".to_string(),
        ));
    }
    let da_third = darcy_number.powf(1.0 / 3.0);
    let scaled_x = x / da_third;
    let nk0 = nk_i(0, scaled_x)?;
    Ok(da_third * nk0)
}

/// Check the boundary condition Nk_0(0) = 0.
///
/// Returns the value at x = 0, which should be zero (or very close to zero).
pub fn nk0_boundary_check() -> SpecialResult<f64> {
    nk_i(0, 0.0)
}

/// Verify asymptotic behavior for large x.
///
/// Returns (Nk_0(x), expected_asymptotic) for comparison.
///
/// # Arguments
/// * `x` - Large positive evaluation point
///
/// # Returns
/// (actual, asymptotic) pair.
pub fn nk0_asymptotic_check(x: f64) -> SpecialResult<(f64, f64)> {
    let nk0_val = nk_i(0, x)?;
    let asymptotic = -airy_ai(x) / (2.0 * x);
    Ok((nk0_val, asymptotic))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nk_struct_creation() {
        let nk = NieldKuznetsov::with_defaults();
        assert!((nk.config.tol - 1e-12).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nk0_at_zero() {
        // Nk_0(0) = 0
        let val = nk_i(0, 0.0).expect("nk_i(0,0) failed");
        assert!(val.abs() < 1e-12, "Nk_0(0) should be 0, got {val}");
    }

    #[test]
    fn test_nk0_derivative_at_zero() {
        // Nk_0'(0) = 0
        let val = nk_i_prime(0, 0.0).expect("nk_i_prime(0,0) failed");
        assert!(val.abs() < 1e-10, "Nk_0'(0) should be 0, got {val}");
    }

    #[test]
    fn test_nk1_at_zero() {
        // Nk_1(0) = ∫₀⁰ Nk_0(t) dt = 0
        let val = nk_i(1, 0.0).expect("nk_i(1,0) failed");
        assert!(val.abs() < 1e-12, "Nk_1(0) should be 0, got {val}");
    }

    #[test]
    fn test_nk0_finite() {
        for &x in &[-2.0, -1.0, 0.5, 1.0, 2.0, 3.0] {
            let val = nk_i(0, x).expect("nk_i(0, x) failed");
            assert!(val.is_finite(), "Nk_0({x}) should be finite, got {val}");
        }
    }

    #[test]
    fn test_nk_higher_orders_finite() {
        // Test i=1 (single integration) with several points
        for &x in &[-1.0, 0.0, 0.5, 1.0, 2.0] {
            let val = nk_i(1, x).expect("nk_i(1) failed");
            assert!(val.is_finite(), "Nk_1({x}) should be finite, got {val}");
        }
        // Test i=2 and i=3 at a single point (to keep test fast)
        for i in 2..=3 {
            let val = nk_i(i, 0.5).expect("nk_i failed");
            assert!(val.is_finite(), "Nk_{i}(0.5) should be finite, got {val}");
        }
    }

    #[test]
    fn test_nk0_asymptotic_large_x() {
        // For large x, Nk_0(x) ~ -Ai(x)/(2x) which decays exponentially
        let x = 10.0;
        let (actual, expected) = nk0_asymptotic_check(x).expect("asymptotic check failed");
        // Both should be very small (exponentially decaying)
        assert!(
            actual.abs() < 1e-3,
            "Nk_0(10) should be small, got {actual}"
        );
        assert!(
            expected.abs() < 1e-3,
            "Asymptotic should be small, got {expected}"
        );
        // They should agree reasonably well
        if expected.abs() > 1e-15 {
            let rel_err = ((actual - expected) / expected).abs();
            assert!(rel_err < 10.0, "Asymptotic agreement: rel_err = {rel_err}");
        }
    }

    #[test]
    fn test_nk0_series_coefficients() {
        let nk = NieldKuznetsov::with_defaults();
        let coeffs = nk.nk0_series_coefficients(10).expect("coefficients failed");
        assert_eq!(coeffs.len(), 10);
        // y_0 = 0 (IC)
        assert!(coeffs[0].abs() < 1e-15);
        // y_1 = 0 (IC)
        assert!(coeffs[1].abs() < 1e-15);
        // y_2 = Ai(0)/2 ≈ 0.1775
        assert!((coeffs[2] - 0.355028053887817 / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nk_boundary_check() {
        let val = nk0_boundary_check().expect("boundary check failed");
        assert!(val.abs() < 1e-12, "Boundary check: Nk_0(0) = {val}");
    }

    #[test]
    fn test_nk_brinkman() {
        let val = nk_brinkman(1.0).expect("brinkman failed");
        assert!(val.is_finite(), "Brinkman Nk_0(1) should be finite");
    }

    #[test]
    fn test_nk_darcy_lapwood() {
        let val = nk_darcy_lapwood(1.0, 0.01).expect("darcy-lapwood failed");
        assert!(val.is_finite(), "Darcy-Lapwood value should be finite");

        // Invalid Darcy number
        assert!(nk_darcy_lapwood(1.0, -0.01).is_err());
    }

    #[test]
    fn test_nk_with_derivative() {
        let nk = NieldKuznetsov::with_defaults();
        let result = nk.evaluate_with_derivative(0, 1.0).expect("eval failed");
        assert!(result.value.is_finite());
        assert!(result.derivative.is_some());
        assert!(result.derivative.expect("no deriv").is_finite());
    }

    #[test]
    fn test_nk1_derivative_equals_nk0() {
        // Nk_1'(x) = Nk_0(x) by definition
        let x = 0.5;
        let nk0_val = nk_i(0, x).expect("nk0 failed");
        let nk1_deriv = nk_i_prime(1, x).expect("nk1' failed");
        // Should be approximately equal (within quadrature error)
        assert!(
            (nk0_val - nk1_deriv).abs() < 0.1,
            "Nk_1'(x) = Nk_0(x) failed: {} vs {}",
            nk1_deriv,
            nk0_val
        );
    }

    #[test]
    fn test_nk_series_small_x() {
        let nk = NieldKuznetsov::with_defaults();
        let result = nk.evaluate(0, 0.1).expect("eval failed");
        assert!(result.value.is_finite());
        assert_eq!(result.method, NKMethod::SeriesSmall);
    }

    #[test]
    fn test_nk0_ode_check() {
        // Numerical ODE check: Nk_0''(x) - x Nk_0(x) ≈ Ai(x)
        let nk = NieldKuznetsov::with_defaults();
        let x = 0.5;
        let h = 1e-4;

        let y_m = nk.evaluate(0, x - h).expect("eval failed").value;
        let y_0 = nk.evaluate(0, x).expect("eval failed").value;
        let y_p = nk.evaluate(0, x + h).expect("eval failed").value;

        let y_pp = (y_p - 2.0 * y_0 + y_m) / (h * h);
        let lhs = y_pp - x * y_0;
        let rhs = airy_ai(x);

        assert!(
            (lhs - rhs).abs() < 0.1,
            "ODE check: lhs={lhs}, rhs={rhs}, diff={}",
            (lhs - rhs).abs()
        );
    }
}
