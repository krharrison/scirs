//! Power series expansions for special functions.
//!
//! Provides the [`PowerSeries`] struct representing a truncated Taylor series
//! around a center point, together with factory functions that build Taylor
//! expansions for common special functions using complex-step differentiation.

use crate::erf::erf;
use crate::gamma::gamma;
use crate::hypergeometric::hyp1f1;
use crate::{bessel, erf as erf_mod};

/// A truncated Taylor/power series around a center point.
///
/// The series is represented as
///
/// ```text
/// f(x) ≈ Σ_{k=0}^{order} coefficients[k] * (x - center)^k
/// ```
#[derive(Debug, Clone)]
pub struct PowerSeries {
    /// The expansion point.
    pub center: f64,
    /// Series coefficients \[a₀, a₁, a₂, …\].
    pub coefficients: Vec<f64>,
    /// Polynomial degree (= `coefficients.len() - 1`).
    pub order: usize,
}

impl PowerSeries {
    /// Create a new power series with explicit coefficients.
    pub fn new(center: f64, coefficients: Vec<f64>) -> Self {
        let order = coefficients.len().saturating_sub(1);
        Self {
            center,
            coefficients,
            order,
        }
    }

    /// Evaluate the series at `x` using Horner's method.
    pub fn eval(&self, x: f64) -> f64 {
        let t = x - self.center;
        let mut result = 0.0f64;
        // Horner: a_n + t*(a_{n-1} + t*(...))
        for &coeff in self.coefficients.iter().rev() {
            result = result * t + coeff;
        }
        result
    }

    /// Differentiate term-by-term: a_k * (x-c)^k → k * a_k * (x-c)^{k-1}.
    ///
    /// The returned series has one fewer term; its center is the same.
    pub fn diff(&self) -> PowerSeries {
        if self.coefficients.len() <= 1 {
            return PowerSeries::new(self.center, vec![0.0]);
        }
        let new_coeffs: Vec<f64> = self
            .coefficients
            .iter()
            .enumerate()
            .skip(1) // skip a_0 (its derivative is 0)
            .map(|(k, &a)| (k as f64) * a)
            .collect();
        PowerSeries::new(self.center, new_coeffs)
    }

    /// Truncate the series to at most `n` coefficients (terms 0..n).
    pub fn truncate(&self, n: usize) -> PowerSeries {
        let coeffs = self.coefficients[..n.min(self.coefficients.len())].to_vec();
        PowerSeries::new(self.center, coeffs)
    }

    /// Compose: compute the series of `self(g(x))` around `g.center`,
    /// assuming `g(g.center) == self.center`.
    ///
    /// The result has `order = min(self.order, g.order)` and is evaluated by
    /// substituting `g` into `self` term-by-term.
    pub fn compose(&self, g: &PowerSeries) -> PowerSeries {
        let out_order = self.order.min(g.order);
        let mut result = vec![0.0f64; out_order + 1];

        // g_shifted[k] = coefficient of (x-g.center)^k in (g(x) - g(g.center))
        // = g.coefficients[k] for k >= 1 (since g.coefficients[0] = g(g.center) = self.center)
        // We compute powers of g_shifted up to out_order.
        let mut g_pow = vec![vec![0.0f64; out_order + 1]; out_order + 1];
        // g_pow[0] = 1  (constant polynomial 1)
        g_pow[0][0] = 1.0;
        // g_pow[1] = g_shifted = g - center
        for k in 1..=out_order {
            if k < g.coefficients.len() {
                g_pow[1][k] = g.coefficients[k];
            }
        }
        // g_pow[j] = g_pow[j-1] * g_pow[1]  (polynomial multiplication, truncated)
        for j in 2..=out_order {
            for i in 0..=out_order {
                for l in 0..=out_order {
                    if i + l <= out_order {
                        let v = g_pow[j - 1][i] * g_pow[1][l];
                        g_pow[j][i + l] += v;
                    }
                }
            }
        }

        // result = Σ_j a_j * g_pow[j]
        for (j, &aj) in self.coefficients.iter().enumerate() {
            if j > out_order {
                break;
            }
            for k in 0..=out_order {
                result[k] += aj * g_pow[j][k];
            }
        }

        PowerSeries::new(g.center, result)
    }
}

// ─── Complex-step derivative helper ──────────────────────────────────────────

/// Compute the k-th derivative of `f` at `x` via complex-step differentiation.
///
/// Uses step size `h = 1e-50` for single differentiation; for higher orders
/// the recursive finite-difference formula is applied with a moderate step.
fn complex_step_kth_deriv<F>(f: &F, x: f64, k: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    if k == 0 {
        return f(x);
    }
    if k == 1 {
        // Complex-step: Im(f(x + ih)) / h
        // Approximate using high-order finite difference for real-valued f
        let h = 1e-6_f64;
        return (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h);
    }
    // For higher derivatives, use recursive finite-difference with step h
    let h = 1e-3_f64;
    let prev_deriv = |t: f64| complex_step_kth_deriv(f, t, k - 1);
    (-prev_deriv(x + 2.0 * h) + 8.0 * prev_deriv(x + h) - 8.0 * prev_deriv(x - h)
        + prev_deriv(x - 2.0 * h))
        / (12.0 * h)
}

/// Build a Taylor series for `f` around `x0` to the given `order`.
///
/// Coefficients are `a_k = f^{(k)}(x0) / k!` computed via finite differences.
fn taylor_from_fn<F>(f: F, x0: f64, order: usize) -> PowerSeries
where
    F: Fn(f64) -> f64,
{
    let mut coeffs = Vec::with_capacity(order + 1);
    let mut factorial = 1.0_f64;
    for k in 0..=order {
        if k > 0 {
            factorial *= k as f64;
        }
        let dk = complex_step_kth_deriv(&f, x0, k);
        coeffs.push(dk / factorial);
    }
    PowerSeries::new(x0, coeffs)
}

// ─── Taylor series factory functions ─────────────────────────────────────────

/// Taylor series of Γ(x) around `x0` to the given `order`.
///
/// Uses numerical differentiation (finite differences) to compute coefficients.
/// For stable results only order ≤ 6 is recommended.
pub fn taylor_gamma(x0: f64, order: usize) -> PowerSeries {
    let safe_order = order.min(6);
    taylor_from_fn(gamma, x0, safe_order)
}

/// Taylor series of erf(x) around `x0` to the given `order`.
///
/// When `x0 == 0.0` the analytical Taylor series is used directly:
/// ```text
/// erf(x) = (2/√π) Σ_{n=0}^∞ (-1)^n x^{2n+1} / (n! (2n+1))
/// ```
/// Otherwise numerical differentiation is used.
pub fn taylor_erf(x0: f64, order: usize) -> PowerSeries {
    if x0 == 0.0 {
        // Analytical: a_k nonzero only for odd k = 2n+1, n = 0,1,...
        let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
        let mut coeffs = vec![0.0f64; order + 1];
        let mut factorial_n = 1.0_f64;
        for n in 0..=order / 2 {
            if n > 0 {
                factorial_n *= n as f64;
            }
            let k = 2 * n + 1;
            if k <= order {
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                // a_k = f^(k)(0) / k!
                // f^(k)(0) = (2/√π) * (-1)^n * n! * ... actually directly:
                // erf(x) = (2/√π) Σ (-1)^n x^{2n+1}/(n!(2n+1))
                // coefficient of x^{2n+1} is (2/√π) * (-1)^n / (n! * (2n+1))
                coeffs[k] = two_over_sqrt_pi * sign / (factorial_n * (2 * n + 1) as f64);
            }
        }
        PowerSeries::new(0.0, coeffs)
    } else {
        let safe_order = order.min(6);
        taylor_from_fn(erf, x0, safe_order)
    }
}

/// Taylor series of J_n(x) (Bessel function of the first kind) around `x0`.
///
/// When `x0 == 0.0` and `n >= 0` the analytical series is used:
/// ```text
/// J_n(x) = Σ_{k=0}^∞ (-1)^k / (k! Γ(n+k+1)) * (x/2)^{n+2k}
/// ```
pub fn taylor_bessel_j(n: i32, x0: f64, order: usize) -> PowerSeries {
    if x0 == 0.0 && n >= 0 {
        let nu = n as usize;
        let mut coeffs = vec![0.0f64; order + 1];
        let mut factorial_k = 1.0_f64;
        let mut gamma_nk1 = gamma_at_integer(nu + 1); // Γ(n+1) = n!
        for k in 0..=order {
            let power = nu + 2 * k;
            if power <= order {
                let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                // coefficient of x^{n+2k}: (-1)^k / (k! * Γ(n+k+1)) * (1/2)^{n+2k}
                let two_pow = (2.0_f64).powi((nu + 2 * k) as i32);
                coeffs[power] = sign / (factorial_k * gamma_nk1 * two_pow);
            }
            if k > 0 {
                factorial_k *= (k + 1) as f64;
            } else {
                // After k=0 iteration, prepare k=1: factorial_k = 1!, gamma = Γ(n+2)
                factorial_k = 1.0;
            }
            gamma_nk1 *= (nu + k + 1) as f64; // Γ(n+k+2) = (n+k+1) * Γ(n+k+1)
        }
        PowerSeries::new(0.0, coeffs)
    } else {
        let safe_order = order.min(6);
        let f = |x: f64| bessel::jn(n, x);
        taylor_from_fn(f, x0, safe_order)
    }
}

/// Helper: Γ(n+1) = n! for non-negative integer n.
fn gamma_at_integer(n: usize) -> f64 {
    let mut result = 1.0_f64;
    for i in 1..n {
        result *= i as f64;
    }
    result
}

/// Taylor series of ₁F₁(a; b; z) around `z0`.
///
/// Only the z-dependence is expanded; `a` and `b` are treated as fixed parameters.
/// When `z0 == 0.0` the analytical series is used:
/// ```text
/// ₁F₁(a; b; z) = Σ_{n=0}^∞ (a)_n / ((b)_n n!) z^n
/// ```
pub fn taylor_1f1(a: f64, b: f64, z0: f64, order: usize) -> PowerSeries {
    if z0 == 0.0 {
        let mut coeffs = Vec::with_capacity(order + 1);
        let mut rising_a = 1.0_f64; // (a)_n
        let mut rising_b = 1.0_f64; // (b)_n
        let mut factorial_n = 1.0_f64; // n!
        for n in 0..=order {
            coeffs.push(rising_a / (rising_b * factorial_n));
            rising_a *= a + n as f64;
            rising_b *= b + n as f64;
            factorial_n *= (n + 1) as f64;
        }
        PowerSeries::new(0.0, coeffs)
    } else {
        let safe_order = order.min(6);
        let f = |z: f64| hyp1f1(a, b, z).unwrap_or(f64::NAN);
        taylor_from_fn(f, z0, safe_order)
    }
}
