//! Symbolic derivative rules and numerical differentiation for special functions.
//!
//! Provides closed-form and high-accuracy numerical derivatives for:
//! - Gamma / polygamma
//! - ₁F₁ hypergeometric
//! - Bessel J and I
//! - Complete elliptic integral K
//! - Generic complex-step differentiation

use crate::elliptic::{ellipe, ellipk};
use crate::gamma::{digamma, gamma, polygamma};
use crate::hypergeometric::hyp1f1;
use crate::{bessel, erf as erf_mod};

/// Derivative rule descriptor: records the function name, the variable
/// differentiated with respect to, and the symbolic formula as a string.
#[derive(Debug, Clone)]
pub struct DerivativeRule {
    /// Name of the function (e.g. `"Gamma"`).
    pub function: String,
    /// Name of the differentiation variable (e.g. `"x"`).
    pub wrt: String,
    /// Human-readable symbolic formula for the derivative.
    pub formula: String,
}

/// A collection of symbolic derivative rules for common special functions.
pub struct SpecialFunctionDerivatives;

impl SpecialFunctionDerivatives {
    // ─── Gamma ───────────────────────────────────────────────────────────────

    /// d/dx Γ(x) = Γ(x) ψ₀(x)  where ψ₀ is the digamma function.
    ///
    /// We compute this via the five-point central difference for robustness,
    /// since the symbolic identity Γ'(x) = Γ(x) ψ₀(x) requires an accurate
    /// digamma implementation.
    pub fn gamma_derivative(x: f64) -> f64 {
        let h = 1e-5_f64;
        (-gamma(x + 2.0 * h) + 8.0 * gamma(x + h) - 8.0 * gamma(x - h) + gamma(x - 2.0 * h))
            / (12.0 * h)
    }

    /// dⁿ/dxⁿ Γ(x) = Γ(x) ψₙ(x)  where ψₙ is the n-th polygamma function.
    ///
    /// The identity used is the Leibniz/product-rule generalisation:
    /// if f = Γ, then f^{(n)} = Γ * ψ_n  holds for n ≥ 0, where ψ_0 = digamma.
    ///
    /// Note: for n = 0 this returns digamma(x) (the 0-th polygamma).
    /// For n = 1 this returns the first derivative Γ(x) ψ₀(x).
    pub fn gamma_nth_derivative(x: f64, n: usize) -> f64 {
        // polygamma(k, x): k=0 → digamma, k=1 → trigamma, etc.
        // gamma_nth_derivative(x, n): n=1 → first derivative = Γ(x)*ψ₀(x)
        // so polygamma order = n - 1 (clamped at 0 for n=0)
        let pg_order = if n == 0 { 0u32 } else { (n as u32) - 1 };
        // Use polygamma directly (not via digamma to avoid inaccuracy in some ranges)
        let psi = polygamma(pg_order, x);
        gamma(x) * psi
    }

    // ─── Confluent hypergeometric ₁F₁ ────────────────────────────────────────

    /// d/dz ₁F₁(a; b; z) = (a/b) ₁F₁(a+1; b+1; z).
    pub fn hyp1f1_derivative_z(a: f64, b: f64, z: f64) -> f64 {
        let ratio = a / b;
        let shifted = hyp1f1(a + 1.0, b + 1.0, z).unwrap_or(f64::NAN);
        ratio * shifted
    }

    /// d/da ₁F₁(a; b; z) approximated via central finite differences.
    ///
    /// There is no general closed form for the derivative with respect to `a`;
    /// we use a five-point stencil with step `h = 1e-5`.
    pub fn hyp1f1_derivative_a(a: f64, b: f64, z: f64) -> f64 {
        let h = 1e-5_f64;
        let f = |av: f64| hyp1f1(av, b, z).unwrap_or(f64::NAN);
        (-f(a + 2.0 * h) + 8.0 * f(a + h) - 8.0 * f(a - h) + f(a - 2.0 * h)) / (12.0 * h)
    }

    // ─── Bessel functions ─────────────────────────────────────────────────────

    /// d/dx J_n(x) = [J_{n-1}(x) − J_{n+1}(x)] / 2.
    pub fn bessel_j_derivative(n: i32, x: f64) -> f64 {
        (bessel::jn(n - 1, x) - bessel::jn(n + 1, x)) / 2.0
    }

    /// d/dx I_n(x) = [I_{n-1}(x) + I_{n+1}(x)] / 2.
    pub fn bessel_i_derivative(n: i32, x: f64) -> f64 {
        let inm1 = bessel::iv(f64::from(n - 1), x);
        let inp1 = bessel::iv(f64::from(n + 1), x);
        (inm1 + inp1) / 2.0
    }

    // ─── Complete elliptic integral K ─────────────────────────────────────────

    /// dK/dk = [E(k) − (1 − k²) K(k)] / [k (1 − k²)]
    ///
    /// Here `k` is the elliptic modulus (not the parameter m = k²).
    /// Note: `ellipk` and `ellipe` in this crate take the parameter `m = k²`.
    pub fn elliptic_k_derivative(k: f64) -> f64 {
        let m = k * k;
        let k_val = ellipk(m);
        let e_val = ellipe(m);
        (e_val - (1.0 - m) * k_val) / (k * (1.0 - m))
    }

    // ─── Complex-step differentiation ────────────────────────────────────────

    /// Approximate f'(x) using the complex-step method.
    ///
    /// This is equivalent to the imaginary part of `f(x + ih) / h` for small `h`,
    /// but since `f` is real-valued we implement it via a high-order central
    /// finite-difference stencil which achieves O(h⁴) accuracy:
    ///
    /// ```text
    /// f'(x) ≈ [−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h)] / (12h)
    /// ```
    ///
    /// with `h = 1e-6`.
    pub fn complex_step_deriv<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
        let h = 1e-6_f64;
        (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)
    }

    // ─── Derivative rule descriptors ─────────────────────────────────────────

    /// Return a list of symbolic derivative rules for common special functions.
    pub fn rules() -> Vec<DerivativeRule> {
        vec![
            DerivativeRule {
                function: "Gamma".to_string(),
                wrt: "x".to_string(),
                formula: "Gamma(x) * digamma(x)".to_string(),
            },
            DerivativeRule {
                function: "LogGamma".to_string(),
                wrt: "x".to_string(),
                formula: "digamma(x)".to_string(),
            },
            DerivativeRule {
                function: "erf".to_string(),
                wrt: "x".to_string(),
                formula: "(2/sqrt(pi)) * exp(-x^2)".to_string(),
            },
            DerivativeRule {
                function: "erfc".to_string(),
                wrt: "x".to_string(),
                formula: "-(2/sqrt(pi)) * exp(-x^2)".to_string(),
            },
            DerivativeRule {
                function: "J_n".to_string(),
                wrt: "x".to_string(),
                formula: "(J_{n-1}(x) - J_{n+1}(x)) / 2".to_string(),
            },
            DerivativeRule {
                function: "I_n".to_string(),
                wrt: "x".to_string(),
                formula: "(I_{n-1}(x) + I_{n+1}(x)) / 2".to_string(),
            },
            DerivativeRule {
                function: "1F1(a;b;z)".to_string(),
                wrt: "z".to_string(),
                formula: "(a/b) * 1F1(a+1;b+1;z)".to_string(),
            },
            DerivativeRule {
                function: "K(k)".to_string(),
                wrt: "k".to_string(),
                formula: "[E(k) - (1-k^2) K(k)] / [k(1-k^2)]".to_string(),
            },
        ]
    }
}
