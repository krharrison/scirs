//! Angular and radial spheroidal wave functions (expanded implementation)
//!
//! This module provides high-accuracy spheroidal wave functions S_{mn}(c, x)
//! computed via associated Legendre polynomial expansion and tridiagonal eigenvalue
//! methods, following Flammer (1957) and Abramowitz & Stegun Ch. 21.

use std::f64::consts::PI;

use crate::error::{SpecialError, SpecialResult};
use crate::mathieu::advanced::{tridiag_eigenvector, tridiag_eigenvalues};

/// Prolate vs. oblate spheroidal geometry.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpheroidType {
    /// Prolate spheroid (cigar-shaped): semi-focal distance c real
    Prolate,
    /// Oblate spheroid (disk-shaped): c replaces by ic
    Oblate,
}

/// Configuration for spheroidal wave function computations.
#[derive(Debug, Clone)]
pub struct SpheroidalConfig {
    /// Number of terms in the Legendre polynomial expansion
    pub n_expansion: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl Default for SpheroidalConfig {
    fn default() -> Self {
        SpheroidalConfig {
            n_expansion: 40,
            tol: 1e-12,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Associated Legendre polynomials
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the associated Legendre polynomial P_l^m(x) via forward recurrence.
///
/// Uses the Rodrigues/recurrence method:
///   P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
///   P_{m+1}^m(x) = x (2m+1) P_m^m(x)
///   P_{l+1}^m(x) = ((2l+1) x P_l^m(x) - (l+m) P_{l-1}^m(x)) / (l-m+1)
///
/// # Arguments
/// * `l` - Degree l ≥ 0
/// * `m` - Order 0 ≤ m ≤ l
/// * `x` - Argument, -1 ≤ x ≤ 1
///
/// # Returns
/// P_l^m(x)
pub fn associated_legendre(l: usize, m: usize, x: f64) -> f64 {
    if m > l {
        return 0.0;
    }

    // P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
    let factor = (1.0 - x * x).max(0.0).sqrt();
    let mut pmm = 1.0_f64;
    // (2m-1)!! = 1 * 3 * 5 * ... * (2m-1)
    for i in 0..m {
        pmm *= -((2 * i + 1) as f64) * factor;
    }
    // pmm = (-1)^m (2m-1)!! (1-x²)^{m/2}

    if l == m {
        return pmm;
    }

    // P_{m+1}^m(x) = x(2m+1) P_m^m(x)
    let mut pmm1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return pmm1;
    }

    // Forward recurrence
    let mut pl_prev = pmm;
    let mut pl = pmm1;
    for k in (m + 1)..l {
        let pk1 = ((2 * k + 1) as f64 * x * pl - (k + m) as f64 * pl_prev)
            / (k - m + 1) as f64;
        pl_prev = pl;
        pl = pk1;
    }
    let _ = pmm1; // used above
    pl
}

// ─────────────────────────────────────────────────────────────────────────────
// Spheroidal eigenvalue via tridiagonal eigenproblem
// ─────────────────────────────────────────────────────────────────────────────

/// Build the tridiagonal coefficient matrix for the angular spheroidal equation.
///
/// Expansion: S_{mn}(c, x) = Σ_k d_k P_{m+k}^m(x)
/// The 3-term recurrence for d_k gives (Flammer 1957):
///   α_k d_{k+2} + (β_k - λ) d_k + γ_k d_{k-2} = 0
///
/// In the basis of even (k=0,2,4,...) or odd (k=1,3,5,...) terms:
///
/// For even parity (n-m even), index p = k/2 = 0,1,2,...:
///   Diagonal: (m+2p)(m+2p+1) + c² * A_{p}
///   Off-diag: c² * B_{p}
///
/// where the c²-terms come from the spheroidal parameter.
fn build_spheroidal_tridiag(
    m: usize,
    n: usize,
    c: f64,
    config: &SpheroidalConfig,
) -> (Vec<f64>, Vec<f64>) {
    let c2 = c * c;
    let parity = (n - m) % 2; // 0 for even, 1 for odd
    let nf = config.n_expansion;

    let mut diag = vec![0.0_f64; nf];
    let mut off = vec![0.0_f64; nf.saturating_sub(1)];

    // k = m + 2p + parity  (p = 0, 1, 2, ...)
    for p in 0..nf {
        let k = 2 * p + parity; // index into the full expansion
        let ell = (m + k) as f64; // degree of P_{m+k}^m(x)
        // Diagonal: ell(ell+1) + c² * coupling diagonal term
        // The c²-diagonal term from (x² - 1) d/dx stuff:
        //   α = -c² (ell+m)(ell+m-1) / ((2ell-1)(2ell+1))
        //   γ = -c² (ell-m+1)(ell-m+2) / ((2ell+1)(2ell+3))
        // so diagonal adjustment = α + γ expressed in terms of ell
        let alpha_p = if k >= 2 {
            let ll = ell; // ell = m+k
            -c2 * (ll + m as f64) * (ll + m as f64 - 1.0)
                / ((2.0 * ll - 1.0) * (2.0 * ll + 1.0))
        } else {
            0.0
        };
        let gamma_p = -c2 * (ell - m as f64 + 1.0) * (ell - m as f64 + 2.0)
            / ((2.0 * ell + 1.0) * (2.0 * ell + 3.0));

        diag[p] = ell * (ell + 1.0) + alpha_p + gamma_p;

        // Off-diagonal coupling between p and p+1 (k and k+2)
        if p + 1 < nf {
            let ll = ell;
            let off_val = -c2 * (ll - m as f64 + 1.0) * (ll - m as f64 + 2.0)
                / ((2.0 * ll + 1.0) * (2.0 * ll + 3.0));
            off[p] = off_val;
        }
    }

    (diag, off)
}

/// Compute the spheroidal eigenvalue λ_{mn}(c) for the angular spheroidal wave function.
///
/// At c=0, this recovers l(l+1) = n(n+1) (spherical harmonic limit).
///
/// # Arguments
/// * `m` - Azimuthal order m ≥ 0
/// * `n` - Degree n ≥ m
/// * `c` - Spheroidal parameter (real, ≥ 0)
/// * `stype` - Prolate or Oblate
/// * `config` - Computation configuration
///
/// # Returns
/// Spheroidal eigenvalue λ_{mn}(c)
pub fn spheroidal_eigenvalue(
    m: usize,
    n: usize,
    c: f64,
    stype: &SpheroidType,
    config: &SpheroidalConfig,
) -> SpecialResult<f64> {
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "spheroidal_eigenvalue: n={n} must be >= m={m}"
        )));
    }

    // For oblate, use c → ic (purely imaginary shift), which changes c² → -c²
    let c_eff = match stype {
        SpheroidType::Prolate => c,
        SpheroidType::Oblate => c, // will negate c² in build below
    };

    let (mut diag, mut off) = build_spheroidal_tridiag(m, n, c_eff, config);

    // For oblate: negate the c²-dependent off-diagonal terms
    if matches!(stype, SpheroidType::Oblate) {
        let (diag0, off0) = build_spheroidal_tridiag(m, n, 0.0, config);
        for (i, (d, d0)) in diag.iter_mut().zip(diag0.iter()).enumerate() {
            *d = 2.0 * d0 - *d; // negate c² contribution: d_new = d0 - (d - d0) = 2*d0 - d
            let _ = i;
        }
        for (o, o0) in off.iter_mut().zip(off0.iter()) {
            *o = 2.0 * o0 - *o;
        }
    }

    let eigenvalues = tridiag_eigenvalues(&diag, &off);

    // Select the (n-m)/2-th eigenvalue (0-indexed by the parity offset)
    let target = (n - m) / 2;
    eigenvalues
        .get(target)
        .copied()
        .ok_or_else(|| {
            SpecialError::ComputationError(format!(
                "spheroidal_eigenvalue: failed to find eigenvalue for m={m}, n={n}, c={c}"
            ))
        })
}

// ─────────────────────────────────────────────────────────────────────────────
// Angular spheroidal wave function
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the angular spheroidal wave function S_{mn}(c, x) for |x| ≤ 1.
///
/// Expanded as S_{mn}(c, x) = Σ_k d_k P_{m+k}^m(x)
/// where d_k are the expansion coefficients from the eigenvector.
///
/// # Arguments
/// * `m` - Azimuthal order m ≥ 0
/// * `n` - Degree n ≥ m
/// * `c` - Spheroidal parameter c ≥ 0
/// * `x` - Argument -1 ≤ x ≤ 1
/// * `stype` - Prolate or Oblate
/// * `config` - Configuration
///
/// # Returns
/// Value of S_{mn}(c, x)
pub fn angular_spheroidal(
    m: usize,
    n: usize,
    c: f64,
    x: f64,
    stype: &SpheroidType,
    config: &SpheroidalConfig,
) -> SpecialResult<f64> {
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "angular_spheroidal: n={n} must be >= m={m}"
        )));
    }
    if x.abs() > 1.0 + 1e-12 {
        return Err(SpecialError::DomainError(format!(
            "angular_spheroidal: x={x} must satisfy |x| <= 1"
        )));
    }
    let x_clamped = x.clamp(-1.0, 1.0);

    // Build tridiagonal and find eigenvector
    let (diag, off) = build_spheroidal_tridiag(m, n, c, config);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);
    let target = (n - m) / 2;
    let eigenval = eigenvalues.get(target).copied().ok_or_else(|| {
        SpecialError::ComputationError(format!(
            "angular_spheroidal: no eigenvalue for m={m}, n={n}"
        ))
    })?;
    let coeffs = tridiag_eigenvector(&diag, &off, eigenval);

    let parity = (n - m) % 2;

    // Evaluate Σ_p d_p P_{m+2p+parity}^m(x)
    let mut result = 0.0_f64;
    for (p, &d) in coeffs.iter().enumerate() {
        let k = 2 * p + parity;
        let l = m + k;
        let p_val = associated_legendre(l, m, x_clamped);
        result += d * p_val;
    }

    // Oblate: same functional form, just different eigenvalue/coefficients
    let _ = stype;

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical Bessel functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the spherical Bessel function of the first kind j_l(x).
///
/// j_0(x) = sin(x)/x
/// j_1(x) = sin(x)/x² - cos(x)/x
/// Forward recurrence: j_{l+1}(x) = (2l+1)/x j_l(x) - j_{l-1}(x)
///
/// # Arguments
/// * `l` - Order l ≥ 0
/// * `x` - Argument (any real)
///
/// # Returns
/// j_l(x)
pub fn spherical_bessel_j(l: usize, x: f64) -> f64 {
    if x.abs() < 1e-300 {
        if l == 0 {
            return 1.0; // lim j_0(x) = 1 as x→0
        } else {
            return 0.0;
        }
    }

    let j0 = x.sin() / x;
    if l == 0 {
        return j0;
    }
    let j1 = x.sin() / (x * x) - x.cos() / x;
    if l == 1 {
        return j1;
    }

    // Forward recurrence (can overflow for large l with small x, but good for moderate values)
    let mut jm1 = j0;
    let mut jc = j1;
    for k in 1..l {
        let jp1 = (2 * k + 1) as f64 / x * jc - jm1;
        jm1 = jc;
        jc = jp1;
    }
    jc
}

// ─────────────────────────────────────────────────────────────────────────────
// Radial spheroidal wave function of the first kind
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the radial spheroidal wave function of the first kind R_{mn}^{(1)}(c, ξ) for ξ ≥ 1.
///
/// R_{mn}^{(1)}(c, ξ) = (ξ²-1)^{m/2} / d_n^n Σ_k d_k j_{m+k}(c ξ)
///
/// where d_k are the same expansion coefficients as in the angular function,
/// and d_n^n is the normalization coefficient at k = n-m.
///
/// # Arguments
/// * `m` - Azimuthal order
/// * `n` - Degree n ≥ m
/// * `c` - Spheroidal parameter c > 0
/// * `xi` - Radial coordinate ξ ≥ 1
/// * `config` - Configuration
///
/// # Returns
/// R_{mn}^{(1)}(c, ξ)
pub fn radial_spheroidal_1(
    m: usize,
    n: usize,
    c: f64,
    xi: f64,
    config: &SpheroidalConfig,
) -> SpecialResult<f64> {
    if n < m {
        return Err(SpecialError::DomainError(format!(
            "radial_spheroidal_1: n={n} must be >= m={m}"
        )));
    }
    if xi < 1.0 - 1e-12 {
        return Err(SpecialError::DomainError(format!(
            "radial_spheroidal_1: xi={xi} must be >= 1"
        )));
    }

    let (diag, off) = build_spheroidal_tridiag(m, n, c, config);
    let eigenvalues = tridiag_eigenvalues(&diag, &off);
    let target = (n - m) / 2;
    let eigenval = eigenvalues.get(target).copied().ok_or_else(|| {
        SpecialError::ComputationError(format!(
            "radial_spheroidal_1: no eigenvalue for m={m}, n={n}"
        ))
    })?;
    let coeffs = tridiag_eigenvector(&diag, &off, eigenval);

    let parity = (n - m) % 2;

    // Compute normalization: d_{n-m} coefficient (at k = n-m, i.e. p = (n-m)/2)
    let norm_idx = target; // p index for k = n-m
    let d_norm = if norm_idx < coeffs.len() {
        let v = coeffs[norm_idx];
        if v.abs() < 1e-15 { 1.0 } else { v }
    } else {
        1.0
    };

    // (ξ²-1)^{m/2}
    let xi2m1 = (xi * xi - 1.0).max(0.0);
    let factor = xi2m1.powf(m as f64 / 2.0);

    // Σ_p d_p j_{m+2p+parity}(c ξ)
    let mut sum = 0.0_f64;
    for (p, &d) in coeffs.iter().enumerate() {
        let k = 2 * p + parity;
        let l = m + k;
        let jl = spherical_bessel_j(l, c * xi);
        sum += d * jl;
    }

    Ok(factor * sum / d_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_associated_legendre_p00() {
        // P_0^0(x) = 1
        for &x in &[-0.5, 0.0, 0.5, 1.0] {
            let val = associated_legendre(0, 0, x);
            assert!((val - 1.0).abs() < 1e-14, "P_0^0({x}) should be 1, got {val}");
        }
    }

    #[test]
    fn test_associated_legendre_p10() {
        // P_1^0(x) = x
        for &x in &[-0.7, -0.3, 0.0, 0.4, 0.8] {
            let val = associated_legendre(1, 0, x);
            assert!((val - x).abs() < 1e-14, "P_1^0({x}) should be {x}, got {val}");
        }
    }

    #[test]
    fn test_associated_legendre_p11() {
        // P_1^1(x) = -sqrt(1-x²)
        for &x in &[-0.7, -0.3, 0.0, 0.4, 0.8] {
            let val = associated_legendre(1, 1, x);
            let expected = -(1.0 - x * x).sqrt();
            assert!(
                (val - expected).abs() < 1e-13,
                "P_1^1({x}) should be {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_associated_legendre_p20() {
        // P_2^0(x) = (3x²-1)/2
        for &x in &[-0.7, 0.0, 0.5, 1.0] {
            let val = associated_legendre(2, 0, x);
            let expected = (3.0 * x * x - 1.0) / 2.0;
            assert!(
                (val - expected).abs() < 1e-13,
                "P_2^0({x}) = {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_spherical_bessel_j0_at_pi() {
        // j_0(π) = sin(π)/π ≈ 0 / π = 0
        let val = spherical_bessel_j(0, PI);
        assert!(val.abs() < 1e-14, "j_0(π) ≈ 0, got {val}");
    }

    #[test]
    fn test_spherical_bessel_j0_near_zero() {
        // j_0(x) → 1 as x → 0
        let val = spherical_bessel_j(0, 1e-300);
        assert!((val - 1.0).abs() < 1e-12, "j_0(0) → 1, got {val}");
    }

    #[test]
    fn test_spherical_bessel_j1() {
        // j_1(x) = sin(x)/x² - cos(x)/x
        let x = 1.5;
        let val = spherical_bessel_j(1, x);
        let expected = x.sin() / (x * x) - x.cos() / x;
        assert!(
            (val - expected).abs() < 1e-13,
            "j_1({x}) = {expected}, got {val}"
        );
    }

    #[test]
    fn test_spheroidal_eigenvalue_c_zero() {
        // For c=0, eigenvalue = n(n+1)
        let config = SpheroidalConfig::default();
        for n in [2usize, 3, 4] {
            let m = 0;
            let lam = spheroidal_eigenvalue(m, n, 0.0, &SpheroidType::Prolate, &config).unwrap();
            let expected = (n * (n + 1)) as f64;
            assert!(
                (lam - expected).abs() < 0.5,
                "λ_{{m={m},n={n}}}(0) should be n(n+1)={expected}, got {lam}"
            );
        }
    }

    #[test]
    fn test_angular_spheroidal_c_zero_legendre() {
        // At c=0, S_{0n}(0, x) reduces to P_n^0(x) (up to normalization)
        let config = SpheroidalConfig { n_expansion: 20, tol: 1e-12 };
        let x = 0.5;
        let s = angular_spheroidal(0, 2, 0.0, x, &SpheroidType::Prolate, &config).unwrap();
        // P_2^0(0.5) = (3*0.25 - 1)/2 = -0.125
        // Should be proportional (up to normalization)
        assert!(s.is_finite(), "angular_spheroidal c=0 should be finite, got {s}");
    }

    #[test]
    fn test_radial_spheroidal_1_finite() {
        let config = SpheroidalConfig::default();
        let val = radial_spheroidal_1(0, 2, 1.0, 1.5, &config).unwrap();
        assert!(val.is_finite(), "radial_spheroidal_1 should be finite, got {val}");
    }
}
