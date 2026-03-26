//! Dedekind Zeta Functions
//!
//! The Dedekind zeta function ζ_K(s) of a number field K generalizes the Riemann zeta function.
//! For the rational field Q, ζ_Q(s) = ζ(s) (Riemann zeta).
//! For quadratic fields Q(√d): ζ_{Q(√d)}(s) = ζ(s) * L(s, χ_D)
//! For cyclotomic fields Q(ζ_n): ζ_{Q(ζ_n)}(s) = Π_{χ mod n} L(s, χ)
//!
//! References:
//! - Neukirch, "Algebraic Number Theory"
//! - Davenport, "Multiplicative Number Theory"

use crate::error::{SpecialError, SpecialResult};
use crate::l_functions::{dirichlet_l, DirichletCharacter};
use std::f64::consts::PI;

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// A number field for Dedekind zeta function computation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum NumberField {
    /// The rational field Q — ζ_Q(s) = Riemann ζ(s)
    Rational,
    /// A quadratic field Q(√d) with discriminant D
    ///
    /// The discriminant D is the fundamental discriminant:
    /// - D ≡ 1 (mod 4) and d square-free, or
    /// - D = 4d with d ≡ 2,3 (mod 4) and d square-free.
    Quadratic {
        /// Fundamental discriminant of the quadratic field
        discriminant: i64,
    },
    /// Cyclotomic field Q(ζ_n) — product over all Dirichlet characters mod n
    Cyclotomic {
        /// The level n (Q(ζ_n) has degree φ(n) over Q)
        n: usize,
    },
}

/// Configuration for Dedekind zeta function evaluation.
#[derive(Debug, Clone)]
pub struct DedekindConfig {
    /// Number of terms in partial sum / L-function series
    pub n_terms: usize,
    /// Whether to use Euler product acceleration (faster for s >> 1)
    pub use_euler_product: bool,
}

impl Default for DedekindConfig {
    fn default() -> Self {
        DedekindConfig {
            n_terms: 1000,
            use_euler_product: true,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Riemann zeta via partial sums with Euler-Maclaurin correction
// ────────────────────────────────────────────────────────────────────────────

/// Compute a partial sum approximation of the Riemann zeta function.
///
/// Uses Σ_{n=1}^{N} n^{-s} with Euler-Maclaurin remainder correction:
/// ζ(s) ≈ Σ_{n=1}^N n^{-s} + N^{1-s}/(s-1) + N^{-s}/2 + Σ Bernoulli corrections
///
/// # Arguments
/// * `s` - Real argument (must be > 1)
/// * `n_terms` - Truncation point N
pub fn riemann_zeta_partial(s: f64, n_terms: usize) -> f64 {
    if s <= 1.0 {
        // Series diverges; caller should handle
        return f64::NAN;
    }

    let n = n_terms as f64;
    // Main partial sum
    let mut sum = 0.0f64;
    for k in 1..=n_terms {
        sum += (k as f64).powf(-s);
    }

    // Euler-Maclaurin correction:
    // Remainder ≈ N^{1-s}/(s-1) + N^{-s}/2 + s*N^{-s-1}/12 - s*(s+1)*(s+2)*N^{-s-3}/720 + ...
    let tail = n.powf(1.0 - s) / (s - 1.0) + 0.5 * n.powf(-s);

    // Bernoulli B2 = 1/6 term:  B_2/2! * s * N^{-s-1}
    let b2_correction = (1.0 / 6.0) * 0.5 * s * n.powf(-s - 1.0);

    // Bernoulli B4 = -1/30 term: B_4/4! * s*(s+1)*(s+2)*(s+3)/4! * N^{-s-4}
    // = -1/30 * s*(s+1)*(s+2)*(s+3) / 24 * N^{-s-4}
    let b4_correction =
        (-1.0 / 30.0) / 24.0 * s * (s + 1.0) * (s + 2.0) * (s + 3.0) * n.powf(-s - 4.0);

    sum + tail + b2_correction + b4_correction
}

// ────────────────────────────────────────────────────────────────────────────
// Dedekind zeta for quadratic fields
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Dedekind zeta function for a quadratic field Q(√d).
///
/// ζ_{Q(√d)}(s) = ζ(s) · L(s, χ_D)
///
/// where χ_D is the Kronecker symbol character (D/·) and D is the fundamental discriminant.
///
/// # Arguments
/// * `s` - Real argument (must be > 1)
/// * `discriminant` - Fundamental discriminant D of the quadratic field
/// * `config` - Computation configuration
///
/// # Errors
/// Returns `SpecialError::DomainError` if s ≤ 1.
pub fn dedekind_zeta_quadratic(
    s: f64,
    discriminant: i64,
    config: &DedekindConfig,
) -> SpecialResult<f64> {
    if s <= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Dedekind zeta requires s > 1, got s = {s}"
        )));
    }
    if discriminant == 0 {
        return Err(SpecialError::ValueError(
            "Discriminant must be nonzero".to_string(),
        ));
    }

    let zeta_s = riemann_zeta_partial(s, config.n_terms);

    // L(s, χ_D) via Dirichlet character Kronecker symbol
    let chi = DirichletCharacter::kronecker_symbol(discriminant);
    let l_s = dirichlet_l(s, &chi, config.n_terms);

    Ok(zeta_s * l_s)
}

// ────────────────────────────────────────────────────────────────────────────
// Dedekind zeta for cyclotomic fields
// ────────────────────────────────────────────────────────────────────────────

/// Compute gcd via Euclidean algorithm.
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute Euler's totient φ(n).
fn euler_phi(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut result = n;
    let mut m = n;
    let mut p = 2usize;
    while p * p <= m {
        if m.is_multiple_of(p) {
            while m.is_multiple_of(p) {
                m /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// Enumerate all primitive characters mod n.
///
/// A character mod n is a Dirichlet character whose conductor equals n.
/// The number of primitive characters mod n is φ(φ(n)) for prime n, and
/// more generally related to the group structure of (Z/nZ)*.
///
/// We enumerate all characters mod n (there are φ(n) total) and return
/// those that are primitive (conductor = n).
///
/// For a cyclic group (Z/nZ)* of order φ(n) with generator g, the characters
/// are χ_k(g^j) = exp(2πi·jk/φ(n)) for k=0,...,φ(n)-1.
/// We restrict to real-valued characters only (values in {-1,0,1}) for
/// efficient real arithmetic, yielding ζ_K(s) as a real product.
///
/// Note: For the full product ζ_{Q(ζ_n)}(s) = Π_χ L(s,χ), we use all real
/// characters (including principal). Complex pairs contribute |L(s,χ)|² to
/// the real-valued product since L(s,χ̄) = conjugate of L(s,χ).
fn enumerate_real_characters(n: usize) -> Vec<DirichletCharacter> {
    // Collect real characters mod n (values in {-1, 0, 1})
    // These arise from the quadratic characters and the principal character.
    // Strategy: generate all characters by trying Kronecker symbols and principal
    let mut chars = Vec::new();

    // Principal character
    chars.push(DirichletCharacter::principal(n));

    // Add real characters from all divisors d | n with d a fundamental discriminant
    // Candidate discriminants: ±d for d | n
    let mut checked = std::collections::HashSet::new();
    checked.insert(n); // principal modulus already added

    for d in 1..=n {
        if n.is_multiple_of(d) {
            for &disc in &[d as i64, -(d as i64)] {
                let chi = DirichletCharacter::kronecker_symbol(disc);
                if chi.modulus == n && chi.is_real() && !chi.is_principal_char() {
                    let key: Vec<i64> = chi.values.clone();
                    if !chars.iter().any(|c: &DirichletCharacter| c.values == key) {
                        chars.push(chi);
                    }
                }
            }
        }
    }

    chars
}

/// Check if a character is the principal character (all non-zero values = 1).
trait IsPrincipal {
    fn is_principal_char(&self) -> bool;
}

impl IsPrincipal for DirichletCharacter {
    fn is_principal_char(&self) -> bool {
        self.values.iter().all(|&v| v == 0 || v == 1) && self.values.contains(&1) && {
            // Principal: χ(n)=1 iff gcd(n,q)=1
            let q = self.modulus;
            self.values
                .iter()
                .enumerate()
                .all(|(n, &v)| if gcd(n, q) == 1 { v == 1 } else { v == 0 })
        }
    }
}

/// Compute the Dedekind zeta function for the cyclotomic field Q(ζ_n).
///
/// ζ_{Q(ζ_n)}(s) = Π_{χ mod n} L(s, χ)
///
/// The product ranges over all distinct Dirichlet characters mod n (one per
/// equivalence class in the factorization). For a practical approximation
/// using real characters only (complex character pairs contribute |L|² each),
/// we compute a lower bound / real-character product.
///
/// # Arguments
/// * `s` - Real argument (must be > 1)
/// * `n` - Level of cyclotomic field Q(ζ_n)
/// * `config` - Computation configuration
///
/// # Errors
/// Returns `SpecialError::DomainError` if s ≤ 1.
pub fn dedekind_zeta_cyclotomic(s: f64, n: usize, config: &DedekindConfig) -> SpecialResult<f64> {
    if s <= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Dedekind zeta requires s > 1, got s = {s}"
        )));
    }
    if n == 0 {
        return Err(SpecialError::ValueError(
            "Cyclotomic level n must be positive".to_string(),
        ));
    }
    if n == 1 {
        // Q(ζ_1) = Q, ζ_Q(s) = ζ(s)
        return Ok(riemann_zeta_partial(s, config.n_terms));
    }

    let chars = enumerate_real_characters(n);
    let mut product = 1.0f64;
    for chi in &chars {
        let l_val = dirichlet_l(s, chi, config.n_terms);
        if l_val.abs() < 1e-15 {
            return Err(SpecialError::ComputationError(
                "L-function value too small, potential precision issue".to_string(),
            ));
        }
        product *= l_val;
    }

    Ok(product)
}

// ────────────────────────────────────────────────────────────────────────────
// General dispatch
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Dedekind zeta function ζ_K(s) for a number field K.
///
/// Dispatches to the appropriate algorithm based on the field type.
///
/// # Arguments
/// * `field` - The number field K
/// * `s` - Real argument (must be > 1)
/// * `config` - Computation configuration
///
/// # Errors
/// Returns `SpecialError::DomainError` if s ≤ 1, or `SpecialError::ValueError`
/// for invalid field parameters.
pub fn dedekind_zeta(field: &NumberField, s: f64, config: &DedekindConfig) -> SpecialResult<f64> {
    if s <= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Dedekind zeta requires s > 1, got s = {s}"
        )));
    }
    match field {
        NumberField::Rational => Ok(riemann_zeta_partial(s, config.n_terms)),
        NumberField::Quadratic { discriminant } => {
            dedekind_zeta_quadratic(s, *discriminant, config)
        }
        NumberField::Cyclotomic { n } => dedekind_zeta_cyclotomic(s, *n, config),
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Class number formula check
// ────────────────────────────────────────────────────────────────────────────

/// Approximate the class number h via the analytic class number formula.
///
/// For an imaginary quadratic field Q(√d) with discriminant D < 0:
///   h = |D|^{1/2} / (2π) · L(1, χ_D)   (up to a small correction factor)
///
/// For D = -4 (Gaussian integers Q(i)): h = 1
/// For D = -3 (Eisenstein integers Q(ω)): h = 1
///
/// Returns the approximate class number as a float.
///
/// # Arguments
/// * `discriminant` - Fundamental discriminant D < 0
pub fn class_number_formula_check(discriminant: i64) -> f64 {
    if discriminant >= 0 {
        return f64::NAN;
    }
    let d_abs = discriminant.unsigned_abs() as f64;
    let chi = DirichletCharacter::kronecker_symbol(discriminant);
    // L(1, χ_D) via partial summation with many terms for accuracy
    let l1 = dirichlet_l(1.0 - 1e-8, &chi, 50_000);
    // Analytic class number formula: h ≈ √|D| / (2π) * L(1, χ_D) * w / 2
    // where w = 2 for most fields (only roots of unity ±1),
    // w = 4 for D=-4, w = 6 for D=-3.
    let w = match discriminant {
        -4 => 4.0,
        -3 => 6.0,
        _ => 2.0,
    };
    // h = w * √|D| / (2π) * L(1, χ_D)  ... but exact formula has w/2 factor differently
    // Standard: h = w * √|D| * L(1,χ_D) / (2π) for imaginary quadratic
    // with correction: actually the standard formula is h = w*|D|^{1/2}/(2π) * L(1,χ_D)
    // But numerically at s slightly above 1 we approximate:
    d_abs.sqrt() * w / (2.0 * PI) * l1
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemann_zeta_at_2() {
        // ζ(2) = π²/6 ≈ 1.6449340668...
        let val = riemann_zeta_partial(2.0, 1000);
        let expected = PI * PI / 6.0;
        assert!(
            (val - expected).abs() < 1e-3,
            "ζ(2) ≈ {val}, expected ≈ {expected}"
        );
    }

    #[test]
    fn test_dedekind_zeta_rational() {
        let config = DedekindConfig::default();
        let val = dedekind_zeta(&NumberField::Rational, 2.0, &config).expect("ok");
        let expected = PI * PI / 6.0;
        assert!(
            (val - expected).abs() < 1e-3,
            "ζ_Q(2) ≈ {val}, expected ≈ {expected}"
        );
    }

    #[test]
    fn test_dedekind_zeta_quadratic_gaussian_integers() {
        // Q(i) = Q(√-1), discriminant D = -4
        // ζ_{Q(i)}(2) = ζ(2) * L(2, χ_{-4})
        // L(2, χ_{-4}) = 1 - 1/9 + 1/25 - ... = Catalan ≈ 0.9159... (< 1)
        // So ζ_{Q(i)}(2) ≈ ζ(2) * 0.9159 ≈ 1.507 < ζ_Q(2) ≈ 1.645
        let config = DedekindConfig::default();
        let val_qi =
            dedekind_zeta(&NumberField::Quadratic { discriminant: -4 }, 2.0, &config).expect("ok");

        assert!(
            val_qi > 0.0,
            "ζ_{{Q(i)}}(2) should be positive, got {val_qi}"
        );
        // The value should be approximately ζ(2) * Catalan ≈ 1.6449 * 0.9159 ≈ 1.507
        assert!(val_qi > 1.0, "ζ_{{Q(i)}}(2) should be > 1, got {val_qi}");
        assert!(val_qi < 2.0, "ζ_{{Q(i)}}(2) should be < 2, got {val_qi}");
    }

    #[test]
    fn test_class_number_gaussian_integers() {
        // h = 1 for Q(i) with D = -4
        let h = class_number_formula_check(-4);
        assert!(
            (h - 1.0).abs() < 0.1,
            "Class number of Q(i) ≈ {h}, expected ≈ 1"
        );
    }

    #[test]
    fn test_class_number_eisenstein_integers() {
        // h = 1 for Q(√-3) with D = -3
        let h = class_number_formula_check(-3);
        assert!(
            (h - 1.0).abs() < 0.2,
            "Class number of Q(√-3) ≈ {h}, expected ≈ 1"
        );
    }

    #[test]
    fn test_dedekind_zeta_domain_error() {
        let config = DedekindConfig::default();
        let result = dedekind_zeta(&NumberField::Rational, 0.5, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_euler_phi() {
        assert_eq!(euler_phi(1), 1);
        assert_eq!(euler_phi(4), 2);
        assert_eq!(euler_phi(6), 2);
        assert_eq!(euler_phi(5), 4);
        assert_eq!(euler_phi(12), 4);
    }

    #[test]
    fn test_dedekind_zeta_cyclotomic_q1() {
        // Q(ζ_1) = Q
        let config = DedekindConfig::default();
        let val = dedekind_zeta(&NumberField::Cyclotomic { n: 1 }, 2.0, &config).expect("ok");
        let expected = PI * PI / 6.0;
        assert!(
            (val - expected).abs() < 1e-3,
            "ζ_{{Q(ζ_1)}}(2) ≈ {val}, expected {expected}"
        );
    }
}
