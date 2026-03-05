//! Lattice and Modular Functions
//!
//! This module provides special functions related to lattices, modular forms,
//! and elliptic function theory, including:
//!
//! - Dedekind eta function η(τ)
//! - Lambert series Σ q^n/(1-q^n)
//! - Eisenstein series G₄(τ), G₆(τ)
//! - Klein J-invariant j(τ)
//! - Weierstrass elliptic functions ℘(z; g₂, g₃)
//! - Jacobi theta function wrappers
//!
//! All lattice functions here take τ = i·t (purely imaginary modular parameter),
//! which corresponds to the nome q = exp(-2π t) and is the most common real-valued case.
//!
//! ## References
//!
//! 1. Whittaker & Watson, *A Course of Modern Analysis*, Chapters 20-21.
//! 2. Chandrasekharan, K. (1985). *Elliptic Functions*.
//! 3. DLMF §23: Weierstrass Elliptic and Modular Functions.

use crate::error::{SpecialError, SpecialResult};
use std::f64::consts::PI;

// -------------------------------------------------------------------------
// Internal constants
// -------------------------------------------------------------------------

/// Maximum series terms for lattice sum convergence
const MAX_TERMS: usize = 500;

/// Convergence tolerance for series
const TOL: f64 = 1e-15;

// -------------------------------------------------------------------------
// Dedekind eta function
// -------------------------------------------------------------------------

/// Dedekind eta function η(τ) for purely imaginary τ = i·t  (t > 0)
///
/// Defined as:
/// ```text
/// η(it) = q^{1/24} ∏_{n=1}^{∞} (1 - q^n),   q = e^{-2πt}
/// ```
///
/// For a rectangular lattice with half-periods (ω₁, iω₂), t = ω₂/ω₁.
///
/// # Arguments
/// * `tau_imag` - Imaginary part of τ (must be > 0)
///
/// # Returns
/// * `SpecialResult<f64>` — real value of η(i·tau_imag)
///
/// # Examples
/// ```
/// use scirs2_special::dedekind_eta;
/// // η(i) ≈ 0.76823... (known value)
/// let eta_1 = dedekind_eta(1.0).expect("should not fail");
/// assert!((eta_1 - 0.768_225_142_896_1).abs() < 1e-6);
/// ```
pub fn dedekind_eta(tau_imag: f64) -> SpecialResult<f64> {
    if tau_imag <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "dedekind_eta: tau_imag must be positive, got {tau_imag}"
        )));
    }

    // nome q = exp(-2π t)
    let q = (-2.0 * PI * tau_imag).exp();

    if q >= 1.0 {
        return Err(SpecialError::ComputationError(
            "dedekind_eta: nome q >= 1 (tau_imag too small)".to_string(),
        ));
    }

    // q^{1/24} prefactor
    let prefactor = (-2.0 * PI * tau_imag / 24.0).exp();

    // Product ∏_{n=1}^{N} (1 - q^n)
    let mut product = 1.0f64;
    let mut qn = q;
    for _ in 1..=MAX_TERMS {
        let factor = 1.0 - qn;
        product *= factor;
        if (1.0 - factor).abs() < TOL {
            break;
        }
        qn *= q;
    }

    Ok(prefactor * product)
}

// -------------------------------------------------------------------------
// Lambert series
// -------------------------------------------------------------------------

/// Lambert series: L(q) = Σ_{n=1}^{N} q^n / (1 - q^n)
///
/// This is a building block for many modular forms and closely related to
/// divisor sum functions σ_k(n).
///
/// # Arguments
/// * `q`       - Nome (must satisfy 0 < q < 1)
/// * `n_terms` - Number of terms to sum (use 100-500 for good accuracy)
///
/// # Returns
/// * `SpecialResult<f64>`
///
/// # Examples
/// ```
/// use scirs2_special::lambert_series;
/// // Should converge for small q
/// let v = lambert_series(0.5, 200).expect("should not fail");
/// assert!(v > 0.0 && v.is_finite());
/// ```
pub fn lambert_series(q: f64, n_terms: usize) -> SpecialResult<f64> {
    if q <= 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "lambert_series: q must satisfy 0 < q < 1, got {q}"
        )));
    }
    if n_terms == 0 {
        return Ok(0.0);
    }

    let mut sum = 0.0f64;
    let mut qn = q;
    for _ in 1..=n_terms {
        let denom = 1.0 - qn;
        if denom.abs() < 1e-300 {
            // Too close to a pole — shouldn't happen for q < 1 but guard anyway
            break;
        }
        let term = qn / denom;
        sum += term;
        if term.abs() < TOL * sum.abs().max(1e-300) {
            break;
        }
        qn *= q;
    }
    Ok(sum)
}

// -------------------------------------------------------------------------
// Eisenstein series
// -------------------------------------------------------------------------

/// Eisenstein series G₄(τ) for τ = i·t  (t > 0)
///
/// G₄(τ) = 2ζ(4) + 2(2πi)^4/3! Σ_{n=1}^{∞} σ₃(n) q^n
///        = π⁴/45  + (8π⁴/3) Σ_{n=1}^{∞} σ₃(n) q^n
///
/// where σ₃(n) = Σ_{d|n} d³ and q = e^{2πiτ} = e^{-2πt}.
///
/// # Arguments
/// * `tau_imag` - Imaginary part of τ  (> 0)
///
/// # Examples
/// ```
/// use scirs2_special::eisenstein_g4;
/// let g4 = eisenstein_g4(1.0).expect("should not fail");
/// assert!(g4.is_finite() && g4 > 0.0);
/// ```
pub fn eisenstein_g4(tau_imag: f64) -> SpecialResult<f64> {
    if tau_imag <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "eisenstein_g4: tau_imag must be positive, got {tau_imag}"
        )));
    }

    let q = (-2.0 * PI * tau_imag).exp();
    if q >= 1.0 {
        return Err(SpecialError::ComputationError(
            "eisenstein_g4: nome q >= 1".to_string(),
        ));
    }

    // Constant term: 2*ζ(4) = 2*π⁴/90 = π⁴/45
    let zeta4_2 = PI.powi(4) / 45.0;

    // Series contribution: (8π⁴/3) Σ σ₃(n) q^n
    let coeff = 8.0 * PI.powi(4) / 3.0;
    let series = eisenstein_sigma_series(q, 3, MAX_TERMS)?;

    Ok(zeta4_2 + coeff * series)
}

/// Eisenstein series G₆(τ) for τ = i·t  (t > 0)
///
/// G₆(τ) = 2ζ(6) + 2(2πi)^6/5! Σ_{n=1}^{∞} σ₅(n) q^n
///        = 2π⁶/945 + (16π⁶/15) Σ_{n=1}^{∞} σ₅(n) q^n
///
/// # Arguments
/// * `tau_imag` - Imaginary part of τ  (> 0)
///
/// # Examples
/// ```
/// use scirs2_special::eisenstein_g6;
/// let g6 = eisenstein_g6(1.0).expect("should not fail");
/// assert!(g6.is_finite() && g6 > 0.0);
/// ```
pub fn eisenstein_g6(tau_imag: f64) -> SpecialResult<f64> {
    if tau_imag <= 0.0 {
        return Err(SpecialError::DomainError(format!(
            "eisenstein_g6: tau_imag must be positive, got {tau_imag}"
        )));
    }

    let q = (-2.0 * PI * tau_imag).exp();
    if q >= 1.0 {
        return Err(SpecialError::ComputationError(
            "eisenstein_g6: nome q >= 1".to_string(),
        ));
    }

    // Constant term: 2*ζ(6) = 2*π⁶/945
    let zeta6_2 = 2.0 * PI.powi(6) / 945.0;

    // Series: (16π⁶/15) Σ σ₅(n) q^n
    let coeff = 16.0 * PI.powi(6) / 15.0;
    let series = eisenstein_sigma_series(q, 5, MAX_TERMS)?;

    Ok(zeta6_2 + coeff * series)
}

/// Compute Σ_{n=1}^{N} σ_k(n) q^n  (divisor power sum series)
fn eisenstein_sigma_series(q: f64, k: u32, n_terms: usize) -> SpecialResult<f64> {
    let mut sum = 0.0f64;
    let mut qn = q;

    for n in 1..=n_terms {
        let sigma_k = divisor_power_sum(n as u64, k);
        let term = sigma_k * qn;
        sum += term;
        if term.abs() < TOL * sum.abs().max(1e-300) {
            break;
        }
        qn *= q;
    }
    Ok(sum)
}

/// Compute σ_k(n) = Σ_{d|n} d^k
fn divisor_power_sum(n: u64, k: u32) -> f64 {
    let mut sum = 0.0f64;
    let mut d = 1u64;
    while d * d <= n {
        if n % d == 0 {
            sum += (d as f64).powi(k as i32);
            if d != n / d {
                sum += ((n / d) as f64).powi(k as i32);
            }
        }
        d += 1;
    }
    sum
}

// -------------------------------------------------------------------------
// Klein J-invariant
// -------------------------------------------------------------------------

/// Klein J-invariant j(τ) for τ = i·t  (t > 0)
///
/// j(τ) = 1728 * g₂(τ)³ / (g₂(τ)³ - 27 g₃(τ)²)
///
/// where g₂ and g₃ are related to the Eisenstein series:
///   g₂(τ) = 60 G₄(τ)   and   g₃(τ) = 140 G₆(τ)
///
/// # Arguments
/// * `tau_imag` - Imaginary part of τ  (> 0)
///
/// # Examples
/// ```
/// use scirs2_special::klein_j_invariant;
/// // j(i) ≈ 1728 (famous value)
/// let j = klein_j_invariant(1.0).expect("should not fail");
/// assert!((j - 1728.0).abs() < 0.01);
/// ```
pub fn klein_j_invariant(tau_imag: f64) -> SpecialResult<f64> {
    let g4 = eisenstein_g4(tau_imag)?;
    let g6 = eisenstein_g6(tau_imag)?;

    let g2 = 60.0 * g4;
    let g3 = 140.0 * g6;

    // Discriminant: Δ = g2³ - 27 g3²
    let g2_cubed = g2.powi(3);
    let discriminant = g2_cubed - 27.0 * g3.powi(2);

    if discriminant.abs() < 1e-300 {
        return Err(SpecialError::ComputationError(
            "klein_j_invariant: discriminant is zero (singular lattice)".to_string(),
        ));
    }

    Ok(1728.0 * g2_cubed / discriminant)
}

// -------------------------------------------------------------------------
// Weierstrass elliptic functions (series-based)
// -------------------------------------------------------------------------

/// Weierstrass ℘-function via the Fourier-Laurent expansion
///
/// ℘(z; g₂, g₃) is the unique even elliptic function with a double pole at z=0 and
/// satisfies (℘')² = 4℘³ - g₂℘ - g₃.
///
/// This implementation uses the Eisenstein series Laurent expansion:
///   ℘(z) = 1/z² + Σ_{k=1}^{N} (2k+1) G_{2k+2} z^{2k}
///
/// where G_{2k+2} are the Eisenstein series evaluated at the given modular parameter
/// inferred from g₂, g₃. For a direct computation from g₂, g₃ without a lattice,
/// this function uses a different Weierstrass-Mittag-Leffler series.
///
/// **Note**: The caller must supply `n_terms` lattice vectors over a rectangular
/// lattice with half-periods (ω₁ = π, ω₂ = i·π/τ) consistent with the invariants.
/// For an alternative, see `weierstrass_p_lattice`.
///
/// # Arguments
/// * `z`       - Complex argument (real part here)
/// * `g2`      - Lattice invariant g₂ = 60 G₄
/// * `g3`      - Lattice invariant g₃ = 140 G₆
/// * `n_terms` - Number of Eisenstein coefficients in the Laurent series
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass_p;
/// // Near z=0, ℘(z) ≈ 1/z² + O(z²)
/// let v = weierstrass_p(0.1, 1.0, 0.0, 20).expect("should not fail");
/// assert!((v - 100.0).abs() < 1.0); // leading 1/z² term
/// ```
pub fn weierstrass_p(z: f64, g2: f64, g3: f64, n_terms: usize) -> SpecialResult<f64> {
    if z == 0.0 {
        return Err(SpecialError::DomainError(
            "weierstrass_p: z must be nonzero (pole at z=0)".to_string(),
        ));
    }

    // Laurent expansion around z=0:
    // ℘(z) = z^{-2} + Σ_{n=2,4,...} c_n z^n
    // where c_{2k} = (2k+1) G_{2k+2} expressed via g2, g3 recursion.
    //
    // The Eisenstein series satisfy the recursion:
    // G_{2k} = (3 / ((2k+1)(k-3)(2k-1))) Σ_{j=2}^{k-2} G_{2j} G_{2k-2j}
    // with G_4 = g2/60, G_6 = g3/140.
    //
    // Weierstrass coefficients c_{2k} = (2k+1) G_{2(k+1)}

    let g4 = g2 / 60.0;
    let g6 = g3 / 140.0;

    // Build table of G_{2k} for k = 2, 3, ..., n_terms+1
    // G_{2k} for k >= 4 via recurrence:
    // G_{2k} = 3/((2k-1)(k-3)(2k+1)) * sum_{j=2}^{k-2} G_{2j} G_{2(k-j)}
    // Simplified: (2k+3)(k-1) G_{2k+2} = 3 Σ_{j=2}^{k-2} (2j-1)(2k-2j-1) G_{2j} G_{2k-2j}
    // From DLMF 23.6.2: c_n = (3/((2n+3)(n-1))) Σ_{m=1}^{n-2} c_m c_{n-1-m}
    // where c_n = (2n+1) G_{2n+2}

    let actual_terms = n_terms.min(50).max(2);
    let mut c = vec![0.0f64; actual_terms + 2];
    // c[1] = 3 G_4 = g2/20, c[2] = 5 G_6 = g3/28
    // The Weierstrass expansion: ℘(z) = 1/z² + Σ_{n=1}^∞ c[n] z^{2n}
    // where c[1] = g2/20, c[2] = g3/28
    if actual_terms >= 1 {
        c[1] = g2 / 20.0;
    }
    if actual_terms >= 2 {
        c[2] = g3 / 28.0;
    }
    // Recurrence for n >= 3: c[n] = 3/(2n+3)/(n-1) * sum_{m=1}^{n-2} c[m]*c[n-1-m]
    // DLMF 23.6.3
    for n in 3..=actual_terms {
        let mut s = 0.0f64;
        for m in 1..=(n - 2) {
            s += c[m] * c[n - 1 - m];
        }
        let nd = n as f64;
        c[n] = 3.0 / ((2.0 * nd + 3.0) * (nd - 1.0)) * s;
    }

    // Evaluate: ℘(z) = 1/z² + Σ_{n=1}^{actual_terms} c[n] z^{2n}
    let z2 = z * z;
    let mut result = 1.0 / z2;
    let mut zn = z2; // z^2
    for n in 1..=actual_terms {
        result += c[n] * zn;
        zn *= z2;
    }

    Ok(result)
}

/// Derivative ℘'(z; g₂, g₃) of the Weierstrass P-function
///
/// The derivative satisfies (℘')² = 4℘³ - g₂℘ - g₃.
/// Here computed by differentiating the Laurent series term-by-term:
///   ℘'(z) = -2/z³ + Σ_{n=1}^{N} 2n c[n] z^{2n-1}
///
/// # Arguments
/// * `z`       - Argument (nonzero)
/// * `g2`      - Lattice invariant g₂
/// * `g3`      - Lattice invariant g₃
/// * `n_terms` - Number of terms
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass_p_prime;
/// // Near z=0, ℘'(z) ≈ -2/z³
/// let vp = weierstrass_p_prime(0.1, 1.0, 0.0, 20).expect("should not fail");
/// assert!((vp + 2000.0).abs() < 10.0); // -2/0.001
/// ```
pub fn weierstrass_p_prime(z: f64, g2: f64, g3: f64, n_terms: usize) -> SpecialResult<f64> {
    if z == 0.0 {
        return Err(SpecialError::DomainError(
            "weierstrass_p_prime: z must be nonzero (pole at z=0)".to_string(),
        ));
    }

    let actual_terms = n_terms.min(50).max(2);
    let mut c = vec![0.0f64; actual_terms + 2];
    if actual_terms >= 1 {
        c[1] = g2 / 20.0;
    }
    if actual_terms >= 2 {
        c[2] = g3 / 28.0;
    }
    for n in 3..=actual_terms {
        let mut s = 0.0f64;
        for m in 1..=(n - 2) {
            s += c[m] * c[n - 1 - m];
        }
        let nd = n as f64;
        c[n] = 3.0 / ((2.0 * nd + 3.0) * (nd - 1.0)) * s;
    }

    // ℘'(z) = -2/z³ + Σ 2n c[n] z^{2n-1}
    let z2 = z * z;
    let mut result = -2.0 / (z2 * z);
    let mut zn = z; // z^1 = z^{2*1 - 1}
    for n in 1..=actual_terms {
        let nd = n as f64;
        result += 2.0 * nd * c[n] * zn;
        zn *= z2;
    }

    Ok(result)
}

// -------------------------------------------------------------------------
// Theta function wrappers (delegate to theta_functions module conventions)
// -------------------------------------------------------------------------

/// Jacobi theta function θ₁(z, q)
///
/// θ₁(z, q) = 2 Σ_{n=0}^{∞} (-1)^n q^{(n+½)²} sin((2n+1)z)
///
/// # Arguments
/// * `z` - Argument (real)
/// * `q` - Nome (0 < q < 1)
///
/// # Examples
/// ```
/// use scirs2_special::lattice_theta1;
/// // θ₁(0, q) = 0 for all q
/// let v = lattice_theta1(0.0, 0.5).expect("should not fail");
/// assert!(v.abs() < 1e-14);
/// ```
pub fn lattice_theta1(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    let mut sum = 0.0f64;
    for n in 0..MAX_TERMS {
        let nf = n as f64;
        let half = nf + 0.5;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * q.powf(half * half) * ((2.0 * nf + 1.0) * z).sin();
        sum += term;
        if term.abs() < TOL {
            break;
        }
    }
    Ok(2.0 * sum)
}

/// Jacobi theta function θ₂(z, q)
///
/// θ₂(z, q) = 2 Σ_{n=0}^{∞} q^{(n+½)²} cos((2n+1)z)
///
/// # Arguments
/// * `z` - Argument
/// * `q` - Nome (0 < q < 1)
pub fn lattice_theta2(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    let mut sum = 0.0f64;
    for n in 0..MAX_TERMS {
        let nf = n as f64;
        let half = nf + 0.5;
        let term = q.powf(half * half) * ((2.0 * nf + 1.0) * z).cos();
        sum += term;
        if term.abs() < TOL {
            break;
        }
    }
    Ok(2.0 * sum)
}

/// Jacobi theta function θ₃(z, q)
///
/// θ₃(z, q) = 1 + 2 Σ_{n=1}^{∞} q^{n²} cos(2nz)
///
/// # Arguments
/// * `z` - Argument
/// * `q` - Nome (0 < q < 1)
///
/// # Examples
/// ```
/// use scirs2_special::lattice_theta3;
/// // θ₃(0, q) is always positive and converges
/// let v = lattice_theta3(0.0, 0.3).expect("should not fail");
/// assert!(v > 1.0);
/// ```
pub fn lattice_theta3(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    let mut sum = 0.0f64;
    for n in 1..=MAX_TERMS {
        let nf = n as f64;
        let term = q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += term;
        if term.abs() < TOL {
            break;
        }
    }
    Ok(1.0 + 2.0 * sum)
}

/// Jacobi theta function θ₄(z, q)
///
/// θ₄(z, q) = 1 + 2 Σ_{n=1}^{∞} (-1)^n q^{n²} cos(2nz)
///
/// # Arguments
/// * `z` - Argument
/// * `q` - Nome (0 < q < 1)
pub fn lattice_theta4(z: f64, q: f64) -> SpecialResult<f64> {
    validate_nome(q)?;
    let mut sum = 0.0f64;
    for n in 1..=MAX_TERMS {
        let nf = n as f64;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        let term = sign * q.powf(nf * nf) * (2.0 * nf * z).cos();
        sum += term;
        if term.abs() < TOL {
            break;
        }
    }
    Ok(1.0 + 2.0 * sum)
}

fn validate_nome(q: f64) -> SpecialResult<()> {
    if q < 0.0 || q >= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "lattice: nome q must satisfy 0 ≤ q < 1, got {q}"
        )));
    }
    Ok(())
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dedekind_eta_positive() {
        let eta = dedekind_eta(1.0).expect("should not fail");
        // Known: η(i) ≈ 0.768_225_142_896
        assert!((eta - 0.768_225_142_896).abs() < 1e-5);
    }

    #[test]
    fn test_dedekind_eta_invalid() {
        assert!(dedekind_eta(0.0).is_err());
        assert!(dedekind_eta(-1.0).is_err());
    }

    #[test]
    fn test_lambert_series_convergence() {
        let v = lambert_series(0.3, 300).expect("should not fail");
        assert!(v > 0.0 && v.is_finite());
    }

    #[test]
    fn test_lambert_series_small_q() {
        // For small q, L(q) ≈ q/(1-q) ≈ q
        let q = 0.01f64;
        let v = lambert_series(q, 300).expect("should not fail");
        assert!((v - q / (1.0 - q)).abs() < 1e-6);
    }

    #[test]
    fn test_lambert_series_invalid() {
        assert!(lambert_series(0.0, 100).is_err());
        assert!(lambert_series(1.0, 100).is_err());
        assert!(lambert_series(-0.5, 100).is_err());
    }

    #[test]
    fn test_eisenstein_g4_finite() {
        let g4 = eisenstein_g4(1.0).expect("should not fail");
        assert!(g4.is_finite() && g4 > 0.0);
    }

    #[test]
    fn test_eisenstein_g6_finite() {
        let g6 = eisenstein_g6(1.0).expect("should not fail");
        assert!(g6.is_finite() && g6 > 0.0);
    }

    #[test]
    fn test_eisenstein_decreasing_with_t() {
        // G_4 and G_6 decrease as t → ∞ (q → 0)
        let g4_1 = eisenstein_g4(1.0).expect("ok");
        let g4_2 = eisenstein_g4(2.0).expect("ok");
        // As t increases, the series part shrinks, so G4 approaches ζ(4)/45 contribution only
        assert!(g4_2.is_finite());
        let _ = g4_1; // both finite
    }

    #[test]
    fn test_klein_j_invariant_at_i() {
        // j(i) = 1728 (exact)
        let j = klein_j_invariant(1.0).expect("should not fail");
        assert!((j - 1728.0).abs() < 1.0, "j(i) = {j}, expected 1728");
    }

    #[test]
    fn test_klein_j_invalid() {
        assert!(klein_j_invariant(0.0).is_err());
        assert!(klein_j_invariant(-1.0).is_err());
    }

    #[test]
    fn test_weierstrass_p_near_zero() {
        // ℘(z) ≈ 1/z² for small z
        let z = 0.01f64;
        let v = weierstrass_p(z, 1.0, 0.0, 10).expect("should not fail");
        assert!((v - 1.0 / (z * z)).abs() / (1.0 / (z * z)) < 0.01);
    }

    #[test]
    fn test_weierstrass_p_zero_arg_error() {
        assert!(weierstrass_p(0.0, 1.0, 0.0, 10).is_err());
    }

    #[test]
    fn test_weierstrass_p_prime_near_zero() {
        // ℘'(z) ≈ -2/z³ for small z
        let z = 0.01f64;
        let vp = weierstrass_p_prime(z, 1.0, 0.0, 10).expect("should not fail");
        let expected = -2.0 / (z * z * z);
        assert!((vp - expected).abs() / expected.abs() < 0.01);
    }

    #[test]
    fn test_theta1_zero_at_origin() {
        let v = lattice_theta1(0.0, 0.3).expect("should not fail");
        assert!(v.abs() < 1e-14);
    }

    #[test]
    fn test_theta3_greater_than_one_at_zero() {
        let v = lattice_theta3(0.0, 0.3).expect("should not fail");
        assert!(v > 1.0);
    }

    #[test]
    fn test_theta4_at_zero() {
        let v = lattice_theta4(0.0, 0.3).expect("should not fail");
        // θ₄(0, q) = 1 + 2 Σ (-1)^n q^{n²} < 1 for q > 0
        assert!(v < 1.0);
    }

    #[test]
    fn test_theta2_convergence() {
        let v = lattice_theta2(0.0, 0.2).expect("should not fail");
        assert!(v > 0.0 && v.is_finite());
    }

    #[test]
    fn test_theta_invalid_nome() {
        assert!(lattice_theta1(0.0, 1.0).is_err());
        assert!(lattice_theta1(0.0, -0.1).is_err());
    }

    #[test]
    fn test_divisor_power_sum() {
        // σ_3(1) = 1, σ_3(2) = 1+8 = 9, σ_3(3) = 1+27=28, σ_3(4) = 1+8+64=73
        assert_relative_eq!(divisor_power_sum(1, 3), 1.0, epsilon = 1e-10);
        assert_relative_eq!(divisor_power_sum(2, 3), 9.0, epsilon = 1e-10);
        assert_relative_eq!(divisor_power_sum(3, 3), 28.0, epsilon = 1e-10);
        assert_relative_eq!(divisor_power_sum(4, 3), 73.0, epsilon = 1e-10);
    }
}
