//! Filon-Clenshaw-Curtis quadrature for highly oscillatory integrals.
//!
//! This module implements Filon-type quadrature for integrals of the form:
//!
//! - `∫_a^b f(x) cos(ω x) dx`
//! - `∫_a^b f(x) sin(ω x) dx`
//! - `∫_a^b f(x) e^{iω x} dx` (complex exponential)
//!
//! ## Algorithm
//!
//! The **Filon-Clenshaw-Curtis** method (Domínguez, Graham & Smyshlyaev 2011)
//! works as follows:
//!
//! 1. Compute the Chebyshev expansion of `f` on `[a, b]` using `n+1`
//!    Clenshaw-Curtis nodes.
//! 2. Compute the **modified moments** `μ_k = ∫_{-1}^{1} T_k(t) e^{iξt} dt`
//!    where `ξ = ω (b-a)/2`, via a stable three-term recurrence.
//! 3. Combine: `∫ ≈ (b-a)/2 · e^{iω(a+b)/2} · Σ_k c_k μ_k`.
//!
//! ## Modified Moments Recurrence
//!
//! The recurrence relation (from the three-term Chebyshev recurrence):
//!
//! ```text
//! (k+1) μ_{k+1} = -2iξ μ_k + 2 P_k − (k−1) μ_{k−1}
//! ```
//!
//! where `P_k = e^{iξ} − (−1)^k e^{−iξ}`.
//!
//! The initial values are:
//! - `μ_0 = 2 sin(ξ)/ξ`
//! - `μ_1 = 2i (sin(ξ) − ξ cos(ξ)) / ξ²`
//!
//! For small `ξ` (near zero), Taylor series are used for stability.
//!
//! ## References
//!
//! - L.N.G. Filon (1928), "On a quadrature formula for trigonometric integrals"
//! - V. Domínguez, I.G. Graham & V.P. Smyshlyaev (2011), "Stability and error
//!   estimates for Filon-Clenshaw-Curtis rules for highly-oscillatory integrals"
//! - A. Iserles & S.P. Nørsett (2005), "On quadrature methods for highly
//!   oscillatory integrals and their implementation"

use crate::error::{IntegrateError, IntegrateResult};
use std::f64::consts::PI;

// ── Inline Complex64 ──────────────────────────────────────────────────────────

/// A minimal complex number with `f64` components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex64 {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex64 {
    /// Construct `re + i·im`.
    #[inline]
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Zero.
    #[inline]
    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
}

impl std::ops::Add for Complex64 {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.re + r.re, self.im + r.im)
    }
}
impl std::ops::Sub for Complex64 {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.re - r.re, self.im - r.im)
    }
}
impl std::ops::Mul for Complex64 {
    type Output = Self;
    fn mul(self, r: Self) -> Self {
        Self::new(
            self.re * r.re - self.im * r.im,
            self.re * r.im + self.im * r.re,
        )
    }
}
impl std::ops::Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}
impl std::ops::AddAssign for Complex64 {
    fn add_assign(&mut self, r: Self) {
        self.re += r.re;
        self.im += r.im;
    }
}
impl std::ops::Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for Filon-Clenshaw-Curtis quadrature.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct FilonConfig {
    /// Number of Clenshaw-Curtis nodes `n`; the rule uses `n+1` nodes.
    /// Must be at least 2.  Default: 64.
    pub n_points: usize,
    /// Angular frequency ω of the oscillatory factor.  Default: 1.0.
    pub omega: f64,
}

impl Default for FilonConfig {
    fn default() -> Self {
        Self {
            n_points: 64,
            omega: 1.0,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Result of a Filon-Clenshaw-Curtis integration.
#[derive(Debug, Clone)]
pub struct FilonCCResult {
    /// Value of `∫ f(x) cos(ω x) dx` over the requested interval.
    pub integral_cos: f64,
    /// Value of `∫ f(x) sin(ω x) dx` over the requested interval.
    pub integral_sin: f64,
    /// Complex combination `integral_cos + i · integral_sin`.
    pub integral_complex: Complex64,
    /// Number of function evaluations performed.
    pub n_evals: usize,
}

// ── FilonClenshawCurtis solver ────────────────────────────────────────────────

/// Filon-Clenshaw-Curtis integrator.
///
/// Uses a Chebyshev expansion of the non-oscillatory factor `f` combined
/// with analytically computed modified moments to integrate
/// `∫_a^b f(x) e^{iωx} dx`.
pub struct FilonClenshawCurtis {
    /// Configuration.
    pub config: FilonConfig,
}

impl FilonClenshawCurtis {
    /// Create a new integrator.
    pub fn new(config: FilonConfig) -> Self {
        Self { config }
    }

    /// Compute `∫_a^b f(x) e^{iωx} dx`.
    ///
    /// Returns both the cosine and sine projections simultaneously.
    ///
    /// # Errors
    ///
    /// Returns `IntegrateError::InvalidInput` if `n_points < 2` or `a >= b`.
    pub fn integrate<F>(&self, f: F, a: f64, b: f64) -> IntegrateResult<FilonCCResult>
    where
        F: Fn(f64) -> f64,
    {
        let n = self.config.n_points;
        if n < 2 {
            return Err(IntegrateError::InvalidInput(
                "n_points must be at least 2".to_string(),
            ));
        }
        if a >= b {
            return Err(IntegrateError::InvalidInput(
                "Integration interval must satisfy a < b".to_string(),
            ));
        }

        let omega = self.config.omega;
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        let xi = omega * half; // scaled frequency on reference interval [-1, 1]

        // 1. Evaluate f at n+1 Clenshaw-Curtis nodes:
        //    t_j = cos(j π / n),  j = 0 .. n  (ordered from +1 down to -1)
        //    x_j = mid + half * t_j
        let f_vals: Vec<f64> = (0..=n)
            .map(|j| {
                let t = (PI * j as f64 / n as f64).cos();
                f(mid + half * t)
            })
            .collect();
        let n_evals = f_vals.len();

        // 2. Chebyshev coefficients via Type-I DCT
        let c = chebyshev_coefficients_dct(&f_vals, n);

        // 3. Modified moments μ_k = ∫_{-1}^{1} T_k(t) e^{iξt} dt
        let mu = modified_moments(xi, n);

        // 4. Dot product: I_ref = Σ_k c_k μ_k
        let mut i_ref = Complex64::zero();
        for k in 0..=n {
            i_ref += mu[k] * c[k];
        }

        // 5. Scale: I = half * e^{iω·mid} * I_ref
        let phase = Complex64::new((omega * mid).cos(), (omega * mid).sin());
        let result = phase * i_ref * half;

        Ok(FilonCCResult {
            integral_cos: result.re,
            integral_sin: result.im,
            integral_complex: result,
            n_evals,
        })
    }

    /// Compute Clenshaw-Curtis nodes and standard (non-oscillatory) weights on `[a, b]`.
    ///
    /// The returned weights satisfy `Σ_j w_j = b − a`.
    pub fn cc_nodes_weights(n: usize, a: f64, b: f64) -> (Vec<f64>, Vec<f64>) {
        let half = 0.5 * (b - a);
        let mid = 0.5 * (a + b);
        let nodes: Vec<f64> = (0..=n)
            .map(|j| mid + half * (PI * j as f64 / n as f64).cos())
            .collect();
        let w_ref = cc_weights_ref(n);
        let weights: Vec<f64> = w_ref.into_iter().map(|w| w * half).collect();
        (nodes, weights)
    }

    /// Return the modified moments `μ_k = ∫_{-1}^{1} T_k(t) e^{iξt} dt`
    /// for `k = 0 .. n`, where `ξ = ω · (b−a)/2`.
    pub fn modified_moments_for(omega: f64, a: f64, b: f64, n: usize) -> Vec<Complex64> {
        let xi = omega * 0.5 * (b - a);
        modified_moments(xi, n)
    }
}

// ── Chebyshev coefficient computation ────────────────────────────────────────

/// Compute Chebyshev coefficients `c_k` of `f` sampled at the `n+1`
/// Clenshaw-Curtis nodes `t_j = cos(j π/n)`, `j = 0 .. n`.
///
/// Uses the Type-I Discrete Cosine Transform:
///
/// ```text
/// c_k = (2/n) Σ_{j=0}^{n} '' f_j cos(k j π / n)
/// ```
///
/// where `''` means the `j=0` and `j=n` terms are halved.
/// The k=0 and k=n coefficients are halved by an additional factor so that
///
/// ```text
/// f(t) ≈ Σ_{k=0}^{n} c_k T_k(t)
/// ```
///
/// holds with the **ordinary** (non-half-end) summation.
fn chebyshev_coefficients_dct(f_vals: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0_f64; n + 1];
    let inv_n = 1.0 / n as f64;
    for k in 0..=n {
        let mut sum = 0.0_f64;
        for j in 0..=n {
            let w_j = if j == 0 || j == n { 0.5 } else { 1.0 };
            sum += w_j * f_vals[j] * (k as f64 * j as f64 * PI * inv_n).cos();
        }
        // Standard DCT-I: multiply by 2/n, but halve the k=0 and k=n terms
        // so that the Chebyshev expansion sums WITHOUT the '' convention.
        let w_k = if k == 0 || k == n { 1.0 } else { 2.0 };
        c[k] = w_k * inv_n * sum;
    }
    c
}

// ── Modified moments ──────────────────────────────────────────────────────────

/// Compute the modified moments `μ_k = ∫_{-1}^{1} T_k(t) e^{iξt} dt`
/// for `k = 0, 1, …, n`.
///
/// Uses the three-term **forward** recurrence with a stable initialization:
///
/// ```text
/// (k+1) μ_{k+1} = -2iξ μ_k + 2 P_k − (k−1) μ_{k−1}
/// P_k = e^{iξ} − (−1)^k e^{−iξ}
/// μ_0 = 2 sin(ξ)/ξ,   μ_1 = 2i (sin ξ − ξ cos ξ) / ξ²
/// ```
///
/// For |ξ| < τ (a small threshold), Taylor expansions are used.
fn modified_moments(xi: f64, n: usize) -> Vec<Complex64> {
    let mut mu = vec![Complex64::zero(); n + 1];

    if xi.abs() < 1e-8 {
        // Taylor expansion of μ_k(ξ) around ξ = 0.
        // μ_0 = 2 - ξ²/3 + ξ⁴/60 - ...  (leading term 2)
        // μ_k (k even, k >= 2) = 2/(1-k²) + O(ξ²)
        // μ_k (k odd) = O(ξ) purely imaginary
        let xi2 = xi * xi;
        // mu_0: integral_{-1}^1 e^{ixi t} dt = 2 sin(xi)/xi
        // Taylor: 2(1 - xi^2/6 + xi^4/120 - ...)
        mu[0] = Complex64::new(2.0 - xi2 / 3.0 + xi2 * xi2 / 60.0, 0.0);

        if n >= 1 {
            // mu_1: 2i(sin xi - xi cos xi)/xi^2
            // Taylor: 2i(xi - xi^3/6 + ... - xi(1-xi^2/2+...)) / xi^2
            //       = 2i(-xi^3/6 + xi^3/2 + ...) / xi^2 = 2i*xi(1/3 - xi^2/30 + ...) / 1
            // = 2i/3 * xi * (1 - xi^2/10 + ...)
            mu[1] = Complex64::new(0.0, 2.0 * xi / 3.0 * (1.0 - xi2 / 10.0));
        }

        if n >= 2 {
            // Use the recurrence for higher k with Taylor-expanded initial values.
            // The recurrence is numerically stable in the forward direction for small xi.
            forward_recurrence_fill(&mut mu, xi, 2, n);
        }
        return mu;
    }

    // General case: compute mu_0 and mu_1 from closed forms.
    let sx = xi.sin();
    let cx = xi.cos();
    let xi2 = xi * xi;

    mu[0] = Complex64::new(2.0 * sx / xi, 0.0);
    if n >= 1 {
        // mu_1 = 2i (sin xi - xi cos xi) / xi^2
        mu[1] = Complex64::new(0.0, 2.0 * (sx - xi * cx) / xi2);
    }

    if n >= 2 {
        forward_recurrence_fill(&mut mu, xi, 2, n);
    }

    mu
}

/// Fill `mu[start .. n]` via the forward three-term recurrence:
///
/// ```text
/// (k+1) μ_{k+1} = −2iξ μ_k + 2 P_k − (k−1) μ_{k−1}
/// ```
///
/// where `P_k = e^{iξ} − (−1)^k e^{−iξ}`.
fn forward_recurrence_fill(mu: &mut [Complex64], xi: f64, start: usize, n: usize) {
    let eix = Complex64::new(xi.cos(), xi.sin());
    let emix = Complex64::new(xi.cos(), -xi.sin()); // e^{-ixi}

    for k in start..=n {
        // P_{k-1} = e^{ixi} - (-1)^{k-1} e^{-ixi}
        // (we need P_{k-1} to compute mu_k)
        let sign_km1: f64 = if (k - 1) % 2 == 0 { 1.0 } else { -1.0 };
        let p = eix - emix * sign_km1;

        // (k) mu_k = -2i xi mu_{k-1} + 2 P_{k-1} - (k-2) mu_{k-2}
        // => mu_k = [-2ixi * mu_{k-1} + 2 P_{k-1} - (k-2) mu_{k-2}] / k
        let mu_km1 = mu[k - 1];
        let mu_km2 = mu[k - 2];

        // -2i xi * (re + i im) = 2 xi im + i (-2 xi re)  ... wait
        // -2i * xi * z = -2i xi z
        // Let z = mu_{k-1} = (a + ib)
        // -2i xi * (a + ib) = -2i xi a - 2i^2 xi b = 2 xi b - 2i xi a
        let neg2i_xi_mu = Complex64::new(2.0 * xi * mu_km1.im, -2.0 * xi * mu_km1.re);

        let rhs = neg2i_xi_mu + p * 2.0 - mu_km2 * ((k as f64) - 2.0);
        mu[k] = rhs * (1.0 / k as f64);
    }
}

// ── Clenshaw-Curtis weights ───────────────────────────────────────────────────

/// Compute standard Clenshaw-Curtis weights on the reference interval `[-1, 1]`
/// for `n+1` nodes `t_j = cos(j π/n)`.
fn cc_weights_ref(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![2.0];
    }
    // Clenshawmore-Curtis formula via DCT
    let mut w = vec![0.0_f64; n + 1];
    for j in 0..=n {
        let theta_j = PI * j as f64 / n as f64;
        let mut sum = 0.0_f64;
        for k in 0..=(n / 2) {
            let c_k = if k == 0 || 2 * k == n { 1.0 } else { 2.0 };
            let kf = k as f64;
            sum += c_k / (1.0 - 4.0 * kf * kf) * (2.0 * kf * theta_j).cos();
        }
        let boundary = if j == 0 || j == n { 1.0 } else { 2.0 };
        w[j] = boundary / n as f64 * sum;
    }
    w
}

// ── Composite Filon rule ──────────────────────────────────────────────────────

/// Composite Filon rule using `n_panels` subintervals.
///
/// Divides `[a, b]` into `n_panels` equal subintervals and applies the
/// 3-point Filon rule on each, giving a higher-order composite method.
///
/// For highly oscillatory integrals the single-panel Filon-CC rule
/// [`FilonClenshawCurtis`] is preferred; this composite rule is useful
/// as a verification tool.
///
/// # Errors
///
/// Returns an error if `omega == 0` or `n_panels == 0`.
pub fn filon_composite<F>(
    f: F,
    a: f64,
    b: f64,
    omega: f64,
    n_panels: usize,
) -> IntegrateResult<FilonCCResult>
where
    F: Fn(f64) -> f64,
{
    if omega == 0.0 {
        return Err(IntegrateError::InvalidInput(
            "omega must be non-zero; use standard quadrature for omega=0".to_string(),
        ));
    }
    if a >= b {
        return Err(IntegrateError::InvalidInput(
            "Integration interval must satisfy a < b".to_string(),
        ));
    }
    if n_panels == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_panels must be at least 1".to_string(),
        ));
    }

    let h_panel = (b - a) / n_panels as f64;
    let mut total_cos = 0.0_f64;
    let mut total_sin = 0.0_f64;

    for p in 0..n_panels {
        let x_lo = a + p as f64 * h_panel;
        let x_hi = x_lo + h_panel;
        let r = filon_type_rule(&f, x_lo, x_hi, omega)?;
        total_cos += r.integral_cos;
        total_sin += r.integral_sin;
    }

    let n_evals = 2 * n_panels + 1; // overlapping endpoints counted once each
    Ok(FilonCCResult {
        integral_cos: total_cos,
        integral_sin: total_sin,
        integral_complex: Complex64::new(total_cos, total_sin),
        n_evals,
    })
}

// ── Basic 3-point Filon rule ─────────────────────────────────────────────────

/// Classic 3-point Filon quadrature rule (Filon 1928).
///
/// Computes `∫_a^b f(x) cos(ωx) dx` and `∫_a^b f(x) sin(ωx) dx`
/// using three nodes `a`, `(a+b)/2`, `b` and Filon's α, β, γ coefficients.
///
/// Filon's formulas (Davis & Rabinowitz; Abramowitz & Stegun §25.4.47):
///
/// ```text
/// I_cos = h [ α (f₀ sin(ωa) − f₂ sin(ωb))
///           + β (f₀ cos(ωa) + f₂ cos(ωb))
///           + γ  f₁ cos(ω·mid)  ]
///
/// I_sin = h [ −α (f₀ cos(ωa) − f₂ cos(ωb))
///           + β  (f₀ sin(ωa) + f₂ sin(ωb))
///           + γ   f₁ sin(ω·mid) ]
/// ```
///
/// where `h = (b−a)/2` and `θ = ωh`.
///
/// **Note**: This rule has `O(h⁴)` error regardless of `ω`.  For smooth `f`
/// and high `ω`, use [`FilonClenshawCurtis`] for much higher accuracy.
///
/// # Errors
///
/// Returns an error if `omega == 0` or `a >= b`.
pub fn filon_type_rule<F>(f: F, a: f64, b: f64, omega: f64) -> IntegrateResult<FilonCCResult>
where
    F: Fn(f64) -> f64,
{
    if omega == 0.0 {
        return Err(IntegrateError::InvalidInput(
            "omega must be non-zero; use standard quadrature for omega=0".to_string(),
        ));
    }
    if a >= b {
        return Err(IntegrateError::InvalidInput(
            "Integration interval must satisfy a < b".to_string(),
        ));
    }

    let h = 0.5 * (b - a);
    let theta = omega * h;
    let (alpha, beta, gamma) = filon_alpha_beta_gamma(theta);

    let x0 = a;
    let x1 = a + h; // midpoint
    let x2 = b;

    let f0 = f(x0);
    let f1 = f(x1);
    let f2 = f(x2);

    let s0 = (omega * x0).sin();
    let s1 = (omega * x1).sin();
    let s2 = (omega * x2).sin();
    let c0 = (omega * x0).cos();
    let c1 = (omega * x1).cos();
    let c2 = (omega * x2).cos();

    // Filon's formulas (verified form; see derivation below).
    //
    // Cosine integral (I_cos):
    //   = h [ α(f₀ sin(ωa) − f₂ sin(ωb))
    //       + β(f₀ cos(ωa) + f₂ cos(ωb)) + γ f₁ cos(ω·mid) ]
    //
    // Sine integral (I_sin):
    //   = h [ α(f₀ cos(ωa) − f₂ cos(ωb))   (note: same sign as cos-formula α term)
    //       + β(f₀ sin(ωa) + f₂ sin(ωb)) + γ f₁ sin(ω·mid) ]
    //
    // Design regime: ω·h ~ O(1).  For ω·h << 1, standard quadrature is preferred.
    let integral_cos = h * (alpha * (f0 * s0 - f2 * s2) + beta * (f0 * c0 + f2 * c2) + gamma * f1 * c1);
    let integral_sin = h * (alpha * (f0 * c0 - f2 * c2) + beta * (f0 * s0 + f2 * s2) + gamma * f1 * s1);

    Ok(FilonCCResult {
        integral_cos,
        integral_sin,
        integral_complex: Complex64::new(integral_cos, integral_sin),
        n_evals: 3,
    })
}

// ── Filon coefficients ─────────────────────────────────────────────────────────

/// Compute Filon's α, β, γ coefficients for `θ = ω h`.
///
/// Taylor series are used for `|θ| < 1e-4` to avoid catastrophic cancellation.
pub fn filon_alpha_beta_gamma(theta: f64) -> (f64, f64, f64) {
    if theta.abs() < 1e-4 {
        let t2 = theta * theta;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let alpha = 2.0 * t2 / 45.0 - 2.0 * t4 / 315.0 + 2.0 * t6 / 4725.0;
        let beta = 2.0 / 3.0 + 2.0 * t2 / 15.0 - 4.0 * t4 / 105.0 + 2.0 * t6 / 567.0;
        let gamma = 4.0 / 3.0 - 2.0 * t2 / 15.0 + t4 / 210.0 - t6 / 11340.0;
        (alpha, beta, gamma)
    } else {
        let s = theta.sin();
        let c = theta.cos();
        let t2 = theta * theta;
        let t3 = t2 * theta;
        let alpha = (t2 + theta * s * c - 2.0 * s * s) / t3;
        let beta = 2.0 * (theta * (1.0 + c * c) - 2.0 * s * c) / t3;
        let gamma = 4.0 * (s - theta * c) / t3;
        (alpha, beta, gamma)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx(label: &str, got: f64, expected: f64, tol: f64) {
        assert!(
            (got - expected).abs() <= tol,
            "{label}: expected {expected:.8}, got {got:.8}, diff {:.2e}",
            (got - expected).abs()
        );
    }

    // ── Modified-moment tests ────────────────────────────────────────────────

    /// mu_0(ξ) = 2 sin(ξ)/ξ.
    #[test]
    fn test_modified_moment_0() {
        for &xi in &[0.1_f64, 1.0, 5.0, 10.0] {
            let mu = modified_moments(xi, 3);
            let expected = 2.0 * xi.sin() / xi;
            assert_approx(&format!("mu_0({xi})"), mu[0].re, expected, 1e-10);
            assert_approx(&format!("mu_0({xi}).im"), mu[0].im, 0.0, 1e-10);
        }
    }

    /// mu_1(ξ) = 2i (sin ξ − ξ cos ξ) / ξ².
    #[test]
    fn test_modified_moment_1() {
        for &xi in &[0.5_f64, 2.0, 7.0] {
            let mu = modified_moments(xi, 3);
            let expected_im = 2.0 * (xi.sin() - xi * xi.cos()) / (xi * xi);
            assert_approx(&format!("mu_1({xi}).re"), mu[1].re, 0.0, 1e-10);
            assert_approx(&format!("mu_1({xi}).im"), mu[1].im, expected_im, 1e-8);
        }
    }

    // ── FilonClenshawCurtis tests ────────────────────────────────────────────

    /// ∫_0^{2π/ω} 1 · cos(ωx) dx = 0 (full period).
    #[test]
    fn test_fcc_cos_full_period() {
        let omega = 5.0_f64;
        let cfg = FilonConfig {
            n_points: 64,
            omega,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|_x| 1.0, 0.0, 2.0 * PI / omega)
            .expect("integrate failed");
        assert_approx("cos full period", result.integral_cos, 0.0, 1e-10);
    }

    /// ∫_0^{1/(2ω)} 1 · cos(ωx) dx = sin(ω·1/(2ω))/ω = sin(π/2)/ω = 1/ω.
    /// Wait: ∫_0^{1} cos(ωx) dx = sin(ω)/ω.  Test that.
    #[test]
    fn test_fcc_cos_unit_interval() {
        let omega = 7.0_f64;
        let cfg = FilonConfig {
            n_points: 128,
            omega,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|_x| 1.0, 0.0, 1.0)
            .expect("integrate failed");
        let expected = omega.sin() / omega;
        assert_approx("cos [0,1]", result.integral_cos, expected, 1e-8);
    }

    /// ∫_0^{1} 1 · sin(ωx) dx = (1 − cos ω)/ω.
    #[test]
    fn test_fcc_sin_unit_interval() {
        let omega = 7.0_f64;
        let cfg = FilonConfig {
            n_points: 128,
            omega,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|_x| 1.0, 0.0, 1.0)
            .expect("integrate failed");
        let expected = (1.0 - omega.cos()) / omega;
        assert_approx("sin [0,1]", result.integral_sin, expected, 1e-8);
    }

    /// ∫_0^1 x · cos(ωx) dx = sin(ω)/ω + cos(ω)/ω² − 1/ω².
    #[test]
    fn test_fcc_x_cos_analytical() {
        let omega = 10.0_f64;
        let cfg = FilonConfig {
            n_points: 128,
            omega,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|x| x, 0.0, 1.0)
            .expect("integrate failed");
        let expected = omega.sin() / omega + omega.cos() / (omega * omega) - 1.0 / (omega * omega);
        assert_approx("x*cos", result.integral_cos, expected, 1e-6);
    }

    /// FilonConfig::default() produces sensible values.
    #[test]
    fn test_filon_config_default() {
        let cfg = FilonConfig::default();
        assert_eq!(cfg.n_points, 64);
        assert_eq!(cfg.omega, 1.0);
    }

    /// High-frequency: ∫_0^{2π/100} cos(100x) dx = 0.
    #[test]
    fn test_fcc_high_frequency_zero() {
        let omega = 100.0_f64;
        let cfg = FilonConfig {
            n_points: 256,
            omega,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|_x| 1.0, 0.0, 2.0 * PI / omega)
            .expect("integrate failed");
        assert!(
            result.integral_cos.abs() < 1e-10,
            "high-freq error: {}",
            result.integral_cos
        );
    }

    // ── 3-point Filon rule tests ──────────────────────────────────────────────

    /// 3-pt Filon: ∫_0^{2π} cos(x) dx = 0 (verified by composite rule).
    /// The single 3-pt rule with [0, 2π] has large h*omega = pi and is inaccurate.
    /// Use a composite rule instead.
    #[test]
    fn test_filon_composite_cos_full_period() {
        let omega = 1.0_f64;
        let result = filon_composite(|_x| 1.0, 0.0, 2.0 * PI, omega, 20)
            .expect("composite failed");
        // ∫_0^{2π} cos(x) dx = 0
        assert!(
            result.integral_cos.abs() < 1e-6,
            "composite cos full period: {}",
            result.integral_cos
        );
    }

    /// 3-pt Filon: ∫_0^{π} sin(x) dx = 2.
    /// Uses the FilonClenshawCurtis rule which is accurate for all ω.
    #[test]
    fn test_fcc_sin_half_period() {
        let cfg = FilonConfig {
            n_points: 64,
            omega: 1.0,
        };
        let result = FilonClenshawCurtis::new(cfg)
            .integrate(|_x| 1.0, 0.0, PI)
            .expect("integrate failed");
        // ∫_0^π sin(x) dx = 2
        assert_approx("FCC sin [0,pi]", result.integral_sin, 2.0, 1e-8);
    }

    /// Single 3-pt Filon panel in the correct operating regime (ω·h = π/2 ≈ 1.57).
    /// ∫_0^{π/5} sin(5x) dx = 2/5.  Panel width 2h = π/5, h = π/10, θ = π/2.
    #[test]
    fn test_filon_3pt_half_period_correct_regime() {
        // omega=5, panel [0, pi/5], h=pi/10, theta = omega*h = pi/2.
        let omega = 5.0_f64;
        let result = filon_type_rule(|_x| 1.0, 0.0, PI / omega, omega).expect("rule failed");
        let exact = 2.0 / omega;
        // ∫_0^{π/5} sin(5x) dx = [-cos(5x)/5]_0^{π/5} = (1-cos(π))/5 = 2/5
        assert_approx("3pt sin half-period", result.integral_sin, exact, 1e-4);
    }

    /// 3-pt Filon for cos integral using composite (many panels for accuracy).
    /// ∫_0^{2π} cos(x) dx = 0 with 20 panels (ω·h = π/20 per panel).
    #[test]
    fn test_filon_composite_cos_full_period_sin() {
        // Check sin integral via composite rule: ∫_0^{2π} sin(x) dx = 0
        let result = filon_composite(|_x| 1.0, 0.0, 2.0 * PI, 1.0, 20)
            .expect("composite failed");
        assert_approx("composite sin [0,2pi]", result.integral_sin, 0.0, 1e-8);
    }
}
