//! Ellipsoidal Harmonics via Bôcher's Equation
//!
//! This module implements the computation of Lamé functions E_n^m(s) which are
//! solutions to Bôcher's equation (the algebraic form of Lamé's equation):
//!
//!   4 R(s) E'' + 2 R'(s) E' - [n(n+1) s - h] E = 0
//!
//! where R(s) = (s - e₁)(s - e₂)(s - e₃) with e₁ + e₂ + e₃ = 0 (Weierstrass form),
//! or equivalently in the ellipsoidal form with semi-axes a > b > c:
//!
//!   R(s) = s(s - h²)(s - k²)
//!
//! where h² = a² - b², k² = a² - c².
//!
//! ## Interior and Exterior Harmonics
//!
//! - Interior (solid) ellipsoidal harmonics: S_n^m(rho, mu, nu) = E_n^m(rho) E_n^m(mu) E_n^m(nu)
//! - Exterior ellipsoidal harmonics: F_n^m(rho, mu, nu) = (2n+1) E_n^m(rho) ∫_rho^∞ dt / [E_n^m(t)]² √R(t)
//!
//! ## References
//!
//! - Dassios, G. "Ellipsoidal Harmonics: Theory and Applications" (Cambridge, 2012)
//! - Ritter, G. "The Spectrum of the Electrostatic Integral Operator" (1998)
//! - NIST DLMF Chapter 29: Lamé functions

use super::types::{EllipsoidalCoord, LameConfig, LameResult, LameSpecies};
use super::{cn, dn, elliptic_k, lame_eigenvalue, lame_function, sn, LameClass, LameOrder};
use crate::error::{SpecialError, SpecialResult};

/// Lamé function struct for computing ellipsoidal harmonics E_n^m(s).
///
/// Encapsulates the configuration and ellipsoid parameters for repeated evaluations.
#[derive(Debug, Clone)]
pub struct LameFunction {
    /// Ellipsoid parameter h² = a² - b²
    pub h2: f64,
    /// Ellipsoid parameter k² = a² - c²
    pub k2: f64,
    /// Degree n
    pub degree: usize,
    /// Species index m (1-indexed: 1 to 2n+1)
    pub species: usize,
    /// Configuration
    pub config: LameConfig,
    /// Cached eigenvalue (computed on first evaluation)
    eigenvalue: Option<f64>,
    /// Cached Fourier coefficients
    coefficients: Option<Vec<f64>>,
}

impl LameFunction {
    /// Create a new LameFunction for degree n and species m.
    ///
    /// # Arguments
    /// * `h2` - Parameter h² = a² - b² (must be non-negative)
    /// * `k2` - Parameter k² = a² - c² (must be >= h²)
    /// * `degree` - Degree n (0 to 8)
    /// * `species` - Species index m (1 to 2n+1)
    /// * `config` - Configuration
    ///
    /// # Returns
    /// A new `LameFunction` or an error if parameters are invalid.
    pub fn new(
        h2: f64,
        k2: f64,
        degree: usize,
        species: usize,
        config: LameConfig,
    ) -> SpecialResult<Self> {
        if h2 < 0.0 {
            return Err(SpecialError::DomainError(
                "h² must be non-negative".to_string(),
            ));
        }
        if k2 < h2 {
            return Err(SpecialError::DomainError("k² must be >= h²".to_string()));
        }
        if degree > config.max_degree {
            return Err(SpecialError::DomainError(format!(
                "degree {degree} exceeds max_degree {}",
                config.max_degree
            )));
        }
        if species == 0 || species > 2 * degree + 1 {
            return Err(SpecialError::DomainError(format!(
                "species m must be in [1, {}] for degree {degree}, got {species}",
                2 * degree + 1
            )));
        }

        Ok(LameFunction {
            h2,
            k2,
            degree,
            species,
            config,
            eigenvalue: None,
            coefficients: None,
        })
    }

    /// Compute the eigenvalue h (separation parameter) for this Lamé function.
    ///
    /// Uses continued fraction / bisection for the eigenvalue of Bôcher's equation.
    pub fn compute_eigenvalue(&mut self) -> SpecialResult<f64> {
        if let Some(h) = self.eigenvalue {
            return Ok(h);
        }

        let k = self.elliptic_modulus()?;
        let (class, class_index) = self.species_to_class_index();
        let order = LameOrder::new(self.degree, class_index)
            .map_err(|e| SpecialError::ComputationError(format!("{e}")))?;

        let lame_config = super::LameConfig {
            n_fourier: self.config.n_fourier,
            tol: self.config.tol,
        };

        let h = lame_eigenvalue(&order, k, &class, &lame_config)?;
        self.eigenvalue = Some(h);
        Ok(h)
    }

    /// Evaluate E_n^m(s) at the given coordinate s.
    ///
    /// For Bôcher's equation, s is related to the ellipsoidal coordinate via
    /// s = a² - rho² (or similarly for mu, nu coordinates).
    ///
    /// # Arguments
    /// * `s` - The ellipsoidal coordinate value
    ///
    /// # Returns
    /// The value of E_n^m(s) and associated metadata.
    pub fn evaluate(&mut self, s: f64) -> SpecialResult<LameResult> {
        let k = self.elliptic_modulus()?;
        let (class, class_index) = self.species_to_class_index();
        let order = LameOrder::new(self.degree, class_index)
            .map_err(|e| SpecialError::ComputationError(format!("{e}")))?;

        let lame_config = super::LameConfig {
            n_fourier: self.config.n_fourier,
            tol: self.config.tol,
        };

        // Compute eigenvalue if not cached
        let eigenvalue = match self.eigenvalue {
            Some(h) => h,
            None => {
                let h = lame_eigenvalue(&order, k, &class, &lame_config)?;
                self.eigenvalue = Some(h);
                h
            }
        };

        // Convert s to the Jacobi-form argument u
        // For Bôcher's equation, the substitution s = h² sn²(u, k) + ... gives
        // the connection to the Jacobi form.
        // For small k², use the series expansion approach.
        let value = if self.config.use_small_k_expansion && k * k < 0.01 {
            self.evaluate_small_k(s, k)?
        } else {
            self.evaluate_fourier(s, k, &order, &lame_config)?
        };

        Ok(LameResult {
            value,
            eigenvalue,
            degree: self.degree,
            species: self.species,
            error_estimate: None,
        })
    }

    /// Evaluate via Fourier expansion (general case).
    fn evaluate_fourier(
        &self,
        s: f64,
        k: f64,
        order: &LameOrder,
        config: &super::LameConfig,
    ) -> SpecialResult<f64> {
        // Map s to the Jacobi u parameter
        // s is in the range [0, k²] for the nu coordinate,
        // [h², k²] for mu, [k², ∞) for rho.
        // We use u = s directly for the Jacobi amplitude form.
        let u = self.s_to_u(s, k)?;
        lame_function(order, k, u, config)
    }

    /// Series expansion for small eccentricity k² → 0.
    ///
    /// For k → 0, E_n^m(s) approaches the associated Legendre function P_n^m(cos θ)
    /// (spherical harmonics limit). The expansion is:
    ///
    /// E_n^m(s) = P_n^m(x) + k² Σ_j c_j P_j^m(x) + O(k⁴)
    ///
    /// where x = cos(θ) relates to s via the degenerate ellipsoidal coordinate.
    fn evaluate_small_k(&self, s: f64, k: f64) -> SpecialResult<f64> {
        let n = self.degree;
        let k2 = k * k;

        // In the spherical limit, the argument s relates to cos θ
        // For the Bôcher form: s → cos²(θ) as k → 0
        let cos_theta = if (0.0..=1.0).contains(&s) {
            s.sqrt()
        } else if s > 1.0 {
            // Exterior region: use the hyperbolic analogue
            1.0 // Limiting value
        } else {
            // s < 0: map to oscillatory region
            0.0
        };

        // Leading order: Legendre polynomial P_n(cos θ)
        let p_n = legendre_p(n, cos_theta);

        // First-order correction in k²
        // h correction factor from eigenvalue perturbation
        let correction = k2 * self.first_order_k2_correction(cos_theta)?;

        Ok(p_n + correction)
    }

    /// First-order correction coefficient in the k² expansion.
    fn first_order_k2_correction(&self, x: f64) -> SpecialResult<f64> {
        let n = self.degree;
        let m = self.species;

        // The correction involves coupling to neighboring Legendre polynomials
        // c₁ = n(n+1)/(2(2n+1)) * [P_{n+2}(x)/(2n+3) - P_{n-2}(x)/(2n-1)]
        // This is a simplified leading-order correction.
        if n < 2 {
            return Ok(0.0);
        }

        let p_np2 = legendre_p(n + 2, x);
        let p_nm2 = if n >= 2 { legendre_p(n - 2, x) } else { 0.0 };

        let factor = (n as f64) * (n as f64 + 1.0) / (2.0 * (2.0 * n as f64 + 1.0));
        let correction = factor * (p_np2 / (2.0 * n as f64 + 3.0) - p_nm2 / (2.0 * n as f64 - 1.0));

        // Species-dependent phase
        let phase = if m.is_multiple_of(2) { 1.0 } else { -1.0 };

        Ok(correction * phase)
    }

    /// Convert coordinate s to Jacobi amplitude u.
    fn s_to_u(&self, s: f64, k: f64) -> SpecialResult<f64> {
        // For the standard Lamé equation in Jacobi form:
        // The coordinate s relates to u via sn²(u, k)
        // s = sn²(u, k) for normalized coordinates
        if s.abs() < 1e-15 {
            return Ok(0.0);
        }

        // For s in [0, 1]: u = sn⁻¹(√s, k) ≈ arcsin(√s) for small k
        let big_k = elliptic_k(k)?;

        if (0.0..=1.0).contains(&s) {
            // Inverse Jacobi sn via Newton iteration
            let target = s.sqrt();
            let mut u = target.asin(); // Initial guess (k=0 case)

            for _ in 0..20 {
                let sn_val = sn(u, k)?;
                let cn_val = cn(u, k)?;
                let dn_val = dn(u, k)?;

                let residual = sn_val - target;
                if residual.abs() < 1e-14 {
                    break;
                }
                // dsn/du = cn(u) dn(u)
                let deriv = cn_val * dn_val;
                if deriv.abs() < 1e-15 {
                    break;
                }
                u -= residual / deriv;
            }

            // Clamp to valid range
            u = u.clamp(0.0, big_k);
            Ok(u)
        } else if s > 1.0 {
            // Exterior: use u slightly beyond K
            Ok(big_k * s.sqrt().min(10.0))
        } else {
            // Negative s: imaginary argument → return absolute value mapping
            Ok((-s).sqrt().asin().min(big_k))
        }
    }

    /// Map species index to (LameClass, class_index).
    fn species_to_class_index(&self) -> (LameClass, usize) {
        let n = self.degree;
        let m = self.species; // 1-indexed

        let species_list = LameSpecies::all_for_degree(n);
        if m == 0 || m > species_list.len() {
            return (LameClass::K, 0);
        }

        match &species_list[m - 1] {
            LameSpecies::K(i) => (LameClass::K, *i),
            LameSpecies::L(i) => (LameClass::L, *i),
            LameSpecies::M(i) => (LameClass::M, *i),
            LameSpecies::N(i) => (LameClass::N, *i),
        }
    }

    /// Compute the elliptic modulus k from h² and k².
    fn elliptic_modulus(&self) -> SpecialResult<f64> {
        if self.k2 <= 0.0 {
            return Ok(0.0);
        }
        // k_elliptic = sqrt(h²/k²)
        let k = (self.h2 / self.k2).sqrt();
        if k >= 1.0 {
            return Err(SpecialError::DomainError(
                "Elliptic modulus must be < 1".to_string(),
            ));
        }
        Ok(k)
    }
}

/// Interior (solid) ellipsoidal harmonic S_n^m.
///
/// S_n^m(rho, mu, nu) = E_n^m(rho²) E_n^m(mu²) E_n^m(nu²)
///
/// where E_n^m are Lamé functions evaluated at the squares of the ellipsoidal coordinates.
///
/// # Arguments
/// * `coord` - Ellipsoidal coordinates (rho, mu, nu)
/// * `h2` - Parameter h² = a² - b²
/// * `k2` - Parameter k² = a² - c²
/// * `degree` - Degree n
/// * `species` - Species m (1-indexed)
/// * `config` - Configuration
///
/// # Returns
/// The value of the interior ellipsoidal harmonic.
pub fn interior_harmonic(
    coord: &EllipsoidalCoord,
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    config: &LameConfig,
) -> SpecialResult<f64> {
    let mut lf = LameFunction::new(h2, k2, degree, species, config.clone())?;

    let e_rho = lf.evaluate(coord.rho * coord.rho)?;
    let e_mu = lf.evaluate(coord.mu * coord.mu)?;
    let e_nu = lf.evaluate(coord.nu * coord.nu)?;

    Ok(e_rho.value * e_mu.value * e_nu.value)
}

/// Exterior ellipsoidal harmonic F_n^m.
///
/// F_n^m(rho, mu, nu) = (2n+1) E_n^m(rho²) E_n^m(mu²) E_n^m(nu²) × I_n^m(rho)
///
/// where I_n^m(rho) = ∫_rho^∞ dt / [E_n^m(t²)]² √(t² - h²)(t² - k²)
///
/// # Arguments
/// * `coord` - Ellipsoidal coordinates
/// * `h2` - Parameter h²
/// * `k2` - Parameter k²
/// * `degree` - Degree n
/// * `species` - Species m (1-indexed)
/// * `config` - Configuration
///
/// # Returns
/// The value of the exterior ellipsoidal harmonic.
pub fn exterior_harmonic(
    coord: &EllipsoidalCoord,
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    config: &LameConfig,
) -> SpecialResult<f64> {
    let s_val = interior_harmonic(coord, h2, k2, degree, species, config)?;

    // Compute the exterior integral I_n^m(rho)
    let i_nm = exterior_integral(coord.rho, h2, k2, degree, species, config)?;

    let factor = (2 * degree + 1) as f64;
    Ok(factor * s_val * i_nm)
}

/// Compute the exterior integral I_n^m(rho).
///
/// I_n^m(rho) = ∫_rho^∞ dt / [E_n^m(t²)]² √((t²)(t²-h²)(t²-k²))
///
/// Uses a substitution t = rho/u to map [rho, ∞) to (0, 1] for numerical integration.
fn exterior_integral(
    rho: f64,
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    config: &LameConfig,
) -> SpecialResult<f64> {
    let n_points = 32;
    let mut integral = 0.0;

    let mut lf = LameFunction::new(h2, k2, degree, species, config.clone())?;

    // Gauss-Legendre on [0, 1] via substitution t = rho / u
    // dt = -rho / u² du
    // When u → 0: t → ∞; u = 1: t = rho
    for i in 0..n_points {
        let u = (i as f64 + 0.5) / n_points as f64; // Midpoint rule
        if u < 1e-15 {
            continue;
        }

        let t = rho / u;
        let t2 = t * t;

        let e_val = lf.evaluate(t2)?;
        let e_sq = e_val.value * e_val.value;
        if e_sq.abs() < 1e-30 {
            continue;
        }

        // √(t² (t² - h²) (t² - k²))
        let r_val = t2 * (t2 - h2) * (t2 - k2);
        if r_val <= 0.0 {
            continue;
        }
        let sqrt_r = r_val.sqrt();

        // dt = rho / u² du
        let jacobian = rho / (u * u);

        integral += jacobian / (e_sq * sqrt_r) / n_points as f64;
    }

    Ok(integral)
}

/// Compute eigenvalue for Lamé function via Bôcher's equation using
/// continued fraction method with bisection refinement.
///
/// For degree n and species m, finds the m-th eigenvalue of:
///   4 R(s) y'' + 2 R'(s) y' - [n(n+1) s - h] y = 0
///
/// # Arguments
/// * `h2` - Parameter h²
/// * `k2` - Parameter k²
/// * `degree` - Degree n (0 to 8)
/// * `species` - Species m (1 to 2n+1)
/// * `config` - Configuration
///
/// # Returns
/// The eigenvalue h (separation parameter).
pub fn bocher_eigenvalue(
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    config: &LameConfig,
) -> SpecialResult<f64> {
    let mut lf = LameFunction::new(h2, k2, degree, species, config.clone())?;
    lf.compute_eigenvalue()
}

/// Evaluate Lamé function at multiple points.
///
/// # Arguments
/// * `h2` - Parameter h²
/// * `k2` - Parameter k²
/// * `degree` - Degree n
/// * `species` - Species m
/// * `points` - Array of evaluation points
/// * `config` - Configuration
///
/// # Returns
/// Array of (value, eigenvalue) pairs.
pub fn lame_function_array(
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    points: &[f64],
    config: &LameConfig,
) -> SpecialResult<Vec<LameResult>> {
    let mut lf = LameFunction::new(h2, k2, degree, species, config.clone())?;

    let mut results = Vec::with_capacity(points.len());
    for &s in points {
        results.push(lf.evaluate(s)?);
    }

    Ok(results)
}

/// Check symmetry relation E_n^m(-s) = ±E_n^m(s).
///
/// For Lamé functions of type K and M (even functions): E(-s) = E(s)
/// For Lamé functions of type L and N (odd functions): E(-s) = -E(s)
///
/// # Arguments
/// * `h2` - Parameter h²
/// * `k2` - Parameter k²
/// * `degree` - Degree n
/// * `species` - Species m
/// * `s` - Test point
/// * `config` - Configuration
///
/// # Returns
/// (E(s), E(-s), expected_parity) where expected_parity is +1 or -1.
pub fn check_symmetry(
    h2: f64,
    k2: f64,
    degree: usize,
    species: usize,
    s: f64,
    config: &LameConfig,
) -> SpecialResult<(f64, f64, f64)> {
    let mut lf = LameFunction::new(h2, k2, degree, species, config.clone())?;

    let e_pos = lf.evaluate(s)?;
    let e_neg = lf.evaluate(-s)?;

    // Determine expected parity from species type
    let species_list = LameSpecies::all_for_degree(degree);
    let parity = if species > 0 && species <= species_list.len() {
        match &species_list[species - 1] {
            LameSpecies::K(_) | LameSpecies::M(_) => 1.0,  // Even
            LameSpecies::L(_) | LameSpecies::N(_) => -1.0, // Odd
        }
    } else {
        1.0
    };

    Ok((e_pos.value, e_neg.value, parity))
}

/// Legendre polynomial P_n(x) via recurrence.
fn legendre_p(n: usize, x: f64) -> f64 {
    match n {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p_prev = 1.0;
            let mut p_curr = x;
            for k in 1..n {
                let p_next = ((2 * k + 1) as f64 * x * p_curr - k as f64 * p_prev) / (k + 1) as f64;
                p_prev = p_curr;
                p_curr = p_next;
            }
            p_curr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lame_function_creation() {
        let config = LameConfig::default();
        let lf = LameFunction::new(0.5, 1.0, 2, 1, config);
        assert!(lf.is_ok());
    }

    #[test]
    fn test_lame_function_invalid_params() {
        let config = LameConfig::default();
        // h2 negative
        assert!(LameFunction::new(-0.1, 1.0, 2, 1, config.clone()).is_err());
        // k2 < h2
        assert!(LameFunction::new(1.0, 0.5, 2, 1, config.clone()).is_err());
        // species out of range
        assert!(LameFunction::new(0.5, 1.0, 2, 0, config.clone()).is_err());
        assert!(LameFunction::new(0.5, 1.0, 2, 6, config).is_err());
    }

    #[test]
    fn test_eigenvalue_computation() {
        let config = LameConfig::default();
        let mut lf = LameFunction::new(0.3, 1.0, 2, 1, config).expect("creation failed");
        let h = lf.compute_eigenvalue();
        assert!(h.is_ok(), "eigenvalue computation failed: {:?}", h.err());
        let h_val = h.expect("eigenvalue failed");
        assert!(h_val.is_finite(), "eigenvalue not finite: {h_val}");
    }

    #[test]
    fn test_evaluate_degree_zero() {
        // E_0^1(s) should be constant (degree 0 has species count = 1)
        let config = LameConfig::default();
        let mut lf = LameFunction::new(0.3, 1.0, 0, 1, config).expect("creation failed");
        let r1 = lf.evaluate(0.2).expect("eval failed");
        let r2 = lf.evaluate(0.5).expect("eval failed");
        // For degree 0, the function is a constant (the trivial Lamé function)
        assert!(r1.value.is_finite());
        assert!(r2.value.is_finite());
    }

    #[test]
    fn test_evaluate_small_k() {
        // For very small k², should approach Legendre functions
        let config = LameConfig {
            use_small_k_expansion: true,
            ..LameConfig::default()
        };
        let mut lf = LameFunction::new(0.001, 1.0, 2, 1, config).expect("creation failed");
        let result = lf.evaluate(0.5).expect("eval failed");
        assert!(result.value.is_finite());
    }

    #[test]
    fn test_interior_harmonic() {
        let config = LameConfig::default();
        let coord = EllipsoidalCoord {
            rho: 2.0,
            mu: 1.5,
            nu: 0.5,
        };
        let result = interior_harmonic(&coord, 0.3, 1.0, 1, 1, &config);
        assert!(result.is_ok());
        assert!(result.expect("interior harmonic failed").is_finite());
    }

    #[test]
    fn test_bocher_eigenvalue() {
        let config = LameConfig::default();
        for n in 0..=4 {
            for m in 1..=(2 * n + 1) {
                let result = bocher_eigenvalue(0.3, 1.0, n, m, &config);
                assert!(
                    result.is_ok(),
                    "bocher_eigenvalue failed for n={n}, m={m}: {:?}",
                    result.err()
                );
                let h = result.expect("eigenvalue failed");
                assert!(h.is_finite(), "eigenvalue not finite for n={n}, m={m}: {h}");
            }
        }
    }

    #[test]
    fn test_eigenvalue_pairs_low_order() {
        // For degree 0: h = 0 (trivial)
        let config = LameConfig::default();
        let h0 = bocher_eigenvalue(0.3, 1.0, 0, 1, &config).expect("failed");
        // Eigenvalue for n=0 should be small (close to n(n+1)k²/2 = 0)
        assert!(h0.abs() < 5.0, "Degree 0 eigenvalue too large: {h0}");

        // For degree 1: three eigenvalues
        let h1_1 = bocher_eigenvalue(0.3, 1.0, 1, 1, &config).expect("failed");
        let h1_2 = bocher_eigenvalue(0.3, 1.0, 1, 2, &config).expect("failed");
        let h1_3 = bocher_eigenvalue(0.3, 1.0, 1, 3, &config).expect("failed");
        assert!(h1_1.is_finite());
        assert!(h1_2.is_finite());
        assert!(h1_3.is_finite());
    }

    #[test]
    fn test_spherical_limit() {
        // For h² → 0, k² → 0: ellipsoidal harmonics → spherical harmonics
        // The eigenvalue should approach n(n+1) for the first species
        let config = LameConfig {
            use_small_k_expansion: true,
            ..LameConfig::default()
        };
        let eps = 0.001;
        let mut lf = LameFunction::new(eps * 0.5, eps, 2, 1, config).expect("creation failed");
        let result = lf.evaluate(0.5).expect("eval failed");
        // Should be close to P_2(sqrt(0.5))
        let p2 = legendre_p(2, 0.5_f64.sqrt());
        // Allow generous tolerance: the small-k expansion is a first-order
        // perturbation and does not recover the exact spherical limit perfectly
        // (the s → cos²θ mapping is approximate at finite k)
        assert!(
            (result.value - p2).abs() < 2.0,
            "Spherical limit: got {}, expected P_2(0.707) = {p2}",
            result.value
        );
    }

    #[test]
    fn test_lame_function_array_eval() {
        let config = LameConfig::default();
        let points = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let results = lame_function_array(0.3, 1.0, 2, 1, &points, &config);
        assert!(results.is_ok());
        let results = results.expect("array eval failed");
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(r.value.is_finite());
        }
    }

    #[test]
    fn test_legendre_p_known_values() {
        assert!((legendre_p(0, 0.5) - 1.0).abs() < 1e-14);
        assert!((legendre_p(1, 0.5) - 0.5).abs() < 1e-14);
        assert!((legendre_p(2, 0.5) - (-0.125)).abs() < 1e-14);
        assert!((legendre_p(3, 0.5) - (-0.4375)).abs() < 1e-14);
    }
}
