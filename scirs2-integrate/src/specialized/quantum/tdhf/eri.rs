//! Two-electron repulsion integrals (ERI) for s-type Gaussian basis functions.
//!
//! # Theory
//! The electron repulsion integral is:
//!
//!   (μν|λσ) = ∫∫ φ_μ(r₁)φ_ν(r₁) [1/r₁₂] φ_λ(r₂)φ_σ(r₂) dr₁dr₂
//!
//! For pure s-type (l=0) contracted Gaussians the integral reduces to
//! Gaussian product pairs evaluated via the Boys F₀ function:
//!
//!   (ss|ss) = (2π^{5/2}) / (ζ·η·√(ζ+η)) · F₀(T) ·
//!              exp(-αβ|A-B|²/ζ) · exp(-γδ|C-D|²/η)
//!
//! where ζ=α+β, η=γ+δ, T = ζη/(ζ+η) |P-Q|², P and Q are the Gaussian
//! product centers.

use super::super::gaussian_integrals::{boys_fn, GaussianBasis};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Boys F₀ helper (re-exported for convenience; uses gaussian_integrals impl)
// ---------------------------------------------------------------------------

/// Boys function F₀(x).
///
/// For x > 1e-6: F₀(x) = √(π/(4x)) · erf(√x)
/// For x ≤ 1e-6: Taylor series  F₀(x) ≈ 1 − x/3 + x²/10 − …
#[inline]
pub fn boys_function_n0(x: f64) -> f64 {
    boys_fn(0, x)
}

// ---------------------------------------------------------------------------
// Core (ss|ss) integral
// ---------------------------------------------------------------------------

/// Compute the (ss|ss) electron repulsion integral between four s-type
/// primitive Gaussians using the Obara-Saika / Boys-function approach.
///
/// # Arguments
/// * `alpha`, `a` – exponent and center of basis function μ
/// * `beta`,  `b` – exponent and center of basis function ν
/// * `gamma`, `c` – exponent and center of basis function λ
/// * `delta`, `d` – exponent and center of basis function σ
///
/// # Returns
/// The integral value (μν|λσ) in atomic units.
pub fn compute_eri_ssss(
    alpha: f64,
    a: [f64; 3],
    beta: f64,
    b: [f64; 3],
    gamma: f64,
    c: [f64; 3],
    delta: f64,
    d: [f64; 3],
) -> f64 {
    let zeta = alpha + beta; // ζ
    let eta = gamma + delta; // η

    // Gaussian product centers
    let p = [
        (alpha * a[0] + beta * b[0]) / zeta,
        (alpha * a[1] + beta * b[1]) / zeta,
        (alpha * a[2] + beta * b[2]) / zeta,
    ];
    let q = [
        (gamma * c[0] + delta * d[0]) / eta,
        (gamma * c[1] + delta * d[1]) / eta,
        (gamma * c[2] + delta * d[2]) / eta,
    ];

    // Auxiliary exponent ρ = ζη/(ζ+η)
    let rho = zeta * eta / (zeta + eta);

    // Boys function argument T = ρ|P-Q|²
    let pq2: f64 = (0..3).map(|i| (p[i] - q[i]).powi(2)).sum();
    let t = rho * pq2;

    // Gaussian overlap pre-factors
    let ab2: f64 = (0..3).map(|i| (a[i] - b[i]).powi(2)).sum();
    let cd2: f64 = (0..3).map(|i| (c[i] - d[i]).powi(2)).sum();
    let exp_ab = (-alpha * beta * ab2 / zeta).exp();
    let exp_cd = (-gamma * delta * cd2 / eta).exp();

    // (2π^{5/2}) / (ζ · η · √(ζ+η))
    let prefactor = 2.0 * PI.powf(2.5) / (zeta * eta * (zeta + eta).sqrt());

    prefactor * boys_function_n0(t) * exp_ab * exp_cd
}

// ---------------------------------------------------------------------------
// ERI tensor builder
// ---------------------------------------------------------------------------

/// Build the full 4-index ERI tensor for a set of s-type Gaussian basis
/// functions.
///
/// The result is stored in a flat `Vec<f64>` with index ordering
/// `data[μ·n³ + ν·n² + λ·n + σ]`.
///
/// For non-s-type functions (l > 0) the integral is set to 0.0 — the higher-
/// angular-momentum case requires the full Obara-Saika 4-center recurrence
/// which is out of scope here.
pub fn build_eri_tensor(basis: &[GaussianBasis]) -> Vec<f64> {
    let n = basis.len();
    let n4 = n * n * n * n;
    let mut eri = vec![0.0_f64; n4];

    for mu in 0..n {
        for nu in 0..mu + 1 {
            for lam in 0..n {
                for sig in 0..lam + 1 {
                    let b_mu = &basis[mu];
                    let b_nu = &basis[nu];
                    let b_lam = &basis[lam];
                    let b_sig = &basis[sig];

                    // Only handle s-type primitives
                    if b_mu.l() != 0 || b_nu.l() != 0 || b_lam.l() != 0 || b_sig.l() != 0 {
                        continue;
                    }

                    let val = compute_eri_ssss(
                        b_mu.exponent,
                        b_mu.center,
                        b_nu.exponent,
                        b_nu.center,
                        b_lam.exponent,
                        b_lam.center,
                        b_sig.exponent,
                        b_sig.center,
                    ) * b_mu.coefficient
                        * b_nu.coefficient
                        * b_lam.coefficient
                        * b_sig.coefficient;

                    // Exploit 8-fold symmetry to fill all equivalent indices
                    let indices = [
                        (mu, nu, lam, sig),
                        (nu, mu, lam, sig),
                        (mu, nu, sig, lam),
                        (nu, mu, sig, lam),
                        (lam, sig, mu, nu),
                        (sig, lam, mu, nu),
                        (lam, sig, nu, mu),
                        (sig, lam, nu, mu),
                    ];
                    for (i, j, k, l) in indices {
                        eri[i * n * n * n + j * n * n + k * n + l] = val;
                    }
                }
            }
        }
    }

    eri
}

// ---------------------------------------------------------------------------
// ERI accessor with 8-fold symmetry
// ---------------------------------------------------------------------------

/// Retrieve the ERI (μν|λσ) from the pre-computed tensor, exploiting 8-fold
/// permutational symmetry.
///
/// Symmetries used:
/// (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = (σλ|μν) = …
#[inline]
pub fn get_eri(eri: &[f64], n: usize, mu: usize, nu: usize, lam: usize, sig: usize) -> f64 {
    eri[mu * n * n * n + nu * n * n + lam * n + sig]
}

// ---------------------------------------------------------------------------
// Cauchy-Schwarz screening
// ---------------------------------------------------------------------------

/// Cauchy-Schwarz screening: returns `true` if the integral |(μν|λσ)| could
/// be significant (i.e. it passes the bound).
///
/// The bound is: |(μν|λσ)| ≤ √[(μν|μν)(λσ|λσ)]
///
/// Returns `false` (screened out) when the bound is below 1e-12.
pub fn schwarz_screening(
    eri: &[f64],
    n: usize,
    mu: usize,
    nu: usize,
    lam: usize,
    sig: usize,
) -> bool {
    let q_munu = get_eri(eri, n, mu, nu, mu, nu);
    let q_lamsig = get_eri(eri, n, lam, sig, lam, sig);
    let bound = (q_munu.abs() * q_lamsig.abs()).sqrt();
    bound >= 1e-12
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::specialized::quantum::gaussian_integrals::normalized_s_gto;

    const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

    // Test that F₀(0) = 1.0
    #[test]
    fn test_boys_function_f0_at_zero() {
        let f = boys_function_n0(0.0);
        assert!((f - 1.0).abs() < 1e-8, "F₀(0) should be 1.0, got {f}");
    }

    // Test continuity of F₀ near x=0
    #[test]
    fn test_boys_function_continuity() {
        let eps = 1e-9;
        let f_eps = boys_function_n0(eps);
        let f_zero = boys_function_n0(0.0);
        assert!(
            (f_eps - f_zero).abs() < 1e-5,
            "F₀ should be continuous near 0: f(ε)={f_eps}, f(0)={f_zero}"
        );
    }

    // Test (ss|ss) with known value: identical s-type at origin
    #[test]
    fn test_eri_ssss_known_value() {
        // Two identical unit-exponent s-type Gaussians at origin
        // ζ = η = 2, P = Q = 0, T = 0 → F₀(0) = 1
        // (ss|ss) = 2π^{5/2} / (2 * 2 * √4) * 1 = 2π^{5/2}/8
        let val = compute_eri_ssss(1.0, ORIGIN, 1.0, ORIGIN, 1.0, ORIGIN, 1.0, ORIGIN);
        let expected = 2.0 * PI.powf(2.5) / (2.0 * 2.0 * (4.0_f64).sqrt());
        assert!(
            (val - expected).abs() < 1e-10,
            "ERI (ss|ss): {val}, expected {expected}"
        );
    }

    // Test 8-fold symmetry of ERI tensor
    #[test]
    fn test_eri_8fold_symmetry() {
        let a = normalized_s_gto(ORIGIN, 1.0);
        let b = normalized_s_gto([1.4, 0.0, 0.0], 0.8);
        let basis = vec![a, b];
        let n = basis.len();
        let eri = build_eri_tensor(&basis);

        // (01|10) == (10|01) etc.
        let v0110 = get_eri(&eri, n, 0, 1, 1, 0);
        let v1001 = get_eri(&eri, n, 1, 0, 0, 1);
        assert!(
            (v0110 - v1001).abs() < 1e-12,
            "(01|10)={v0110} != (10|01)={v1001}"
        );

        // (00|11) == (11|00)
        let v0011 = get_eri(&eri, n, 0, 0, 1, 1);
        let v1100 = get_eri(&eri, n, 1, 1, 0, 0);
        assert!(
            (v0011 - v1100).abs() < 1e-12,
            "(00|11)={v0011} != (11|00)={v1100}"
        );
    }

    // Diagonal ERI (μμ|μμ) must be positive (self-repulsion)
    #[test]
    fn test_eri_self_repulsion_positive() {
        let a = normalized_s_gto(ORIGIN, 1.0);
        let basis = vec![a];
        let eri = build_eri_tensor(&basis);
        let v = get_eri(&eri, 1, 0, 0, 0, 0);
        assert!(
            v > 0.0,
            "Self-repulsion (00|00) should be positive, got {v}"
        );
    }
}
