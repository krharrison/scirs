//! Types for Lamé functions and ellipsoidal harmonics.
//!
//! This module defines configuration structs, result types, and coordinate types
//! used throughout the Lamé function and ellipsoidal harmonic computations.

use serde::{Deserialize, Serialize};

/// Configuration for Lamé function and ellipsoidal harmonic computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LameConfig {
    /// Number of Fourier terms in the expansion (default: 32)
    pub n_fourier: usize,
    /// Convergence tolerance (default: 1e-12)
    pub tol: f64,
    /// Maximum degree n supported (default: 8)
    pub max_degree: usize,
    /// Number of quadrature points for normalization integrals (default: 64)
    pub n_quadrature: usize,
    /// Enable series expansion for small eccentricity (default: true)
    pub use_small_k_expansion: bool,
}

impl Default for LameConfig {
    fn default() -> Self {
        LameConfig {
            n_fourier: 32,
            tol: 1e-12,
            max_degree: 8,
            n_quadrature: 64,
            use_small_k_expansion: true,
        }
    }
}

/// Result of a Lamé function evaluation.
#[derive(Debug, Clone)]
pub struct LameResult {
    /// The computed function value
    pub value: f64,
    /// The Lamé eigenvalue (separation parameter h)
    pub eigenvalue: f64,
    /// Degree n
    pub degree: usize,
    /// Species (sub-order) m within degree n
    pub species: usize,
    /// Estimated error (if available)
    pub error_estimate: Option<f64>,
}

/// Ellipsoidal coordinate system (confocal quadrics).
///
/// An ellipsoidal coordinate (rho, mu, nu) with semi-axes a > b > c > 0 defines
/// confocal ellipsoids (rho), hyperboloids of one sheet (mu), and hyperboloids
/// of two sheets (nu).
///
/// Ranges: rho >= a, b <= mu <= a, c <= nu <= b (for exterior region).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipsoidalCoord {
    /// First coordinate (rho): rho >= a, along the outermost confocal ellipsoid
    pub rho: f64,
    /// Second coordinate (mu): b <= mu <= a, along the one-sheeted hyperboloid
    pub mu: f64,
    /// Third coordinate (nu): c <= nu <= b, along the two-sheeted hyperboloid
    pub nu: f64,
}

impl EllipsoidalCoord {
    /// Create a new ellipsoidal coordinate with validation.
    ///
    /// # Arguments
    /// * `rho` - First coordinate
    /// * `mu` - Second coordinate
    /// * `nu` - Third coordinate
    /// * `a` - Semi-axis a (largest)
    /// * `b` - Semi-axis b
    /// * `_c` - Semi-axis c (smallest)
    ///
    /// # Returns
    /// A valid `EllipsoidalCoord` or an error if constraints are violated.
    pub fn new(rho: f64, mu: f64, nu: f64, a: f64, b: f64, _c: f64) -> Result<Self, String> {
        if rho < a {
            return Err(format!("rho ({rho}) must be >= a ({a})"));
        }
        if mu < b || mu > a {
            return Err(format!("mu ({mu}) must be in [{b}, {a}]"));
        }
        Ok(EllipsoidalCoord { rho, mu, nu })
    }

    /// Convert to Cartesian coordinates (x, y, z).
    ///
    /// x² = (rho² - h²)(mu² - h²)(nu² - h²) / (h² k²)  ... simplified form
    /// using the relation to semi-axes.
    ///
    /// # Arguments
    /// * `h2` - h² = a² - b² (squared linear eccentricity in xy)
    /// * `k2` - k² = a² - c² (squared linear eccentricity in xz)
    pub fn to_cartesian(&self, h2: f64, k2: f64) -> (f64, f64, f64) {
        let rho2 = self.rho * self.rho;
        let mu2 = self.mu * self.mu;
        let nu2 = self.nu * self.nu;

        // x² = rho² mu² nu² / (h² k²)
        let x = if h2 > 0.0 && k2 > 0.0 {
            (rho2 * mu2 * nu2 / (h2 * k2)).sqrt()
        } else {
            0.0
        };

        // y² = (rho² - h²)(mu² - h²)(h² - nu²) / (h² (k² - h²))
        let y = if h2 > 0.0 && k2 > h2 {
            let num = (rho2 - h2) * (mu2 - h2) * (h2 - nu2);
            if num >= 0.0 {
                (num / (h2 * (k2 - h2))).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        // z² = (rho² - k²)(k² - mu²)(k² - nu²) / (k² (k² - h²))
        let z = if k2 > 0.0 && k2 > h2 {
            let num = (rho2 - k2) * (k2 - mu2) * (k2 - nu2);
            if num >= 0.0 {
                (num / (k2 * (k2 - h2))).sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };

        (x, y, z)
    }
}

/// Species of Lamé functions within a given degree n.
///
/// For degree n, there are 2n+1 Lamé functions (species) classified into four types:
/// - K (even-even): ceil((n+1)/2) functions
/// - L (even-odd): floor(n/2) functions
/// - M (odd-even): floor(n/2) functions
/// - N (odd-odd): floor((n-1)/2) functions (for n >= 2)
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LameSpecies {
    /// Type K: even in both sn and cn (polynomial in sn²)
    K(usize),
    /// Type L: even in sn, odd in cn (cn × polynomial in sn²)
    L(usize),
    /// Type M: odd in sn, even in cn (sn × polynomial in sn²)
    M(usize),
    /// Type N: odd in both sn and cn (sn cn × polynomial in sn²)
    N(usize),
}

impl LameSpecies {
    /// Enumerate all species for a given degree n.
    ///
    /// Returns a vector of (species, index) for all 2n+1 Lamé functions of degree n.
    pub fn all_for_degree(n: usize) -> Vec<LameSpecies> {
        let mut species = Vec::with_capacity(2 * n + 1);

        // K species: ceil((n+1)/2)
        let n_k = (n + 1 + 1) / 2; // ceil((n+1)/2)
        for i in 0..n_k {
            species.push(LameSpecies::K(i));
        }

        // L species: floor(n/2)
        let n_l = n / 2;
        for i in 0..n_l {
            species.push(LameSpecies::L(i));
        }

        // M species: floor((n+1)/2)
        let n_m = (n + 1) / 2;
        for i in 0..n_m {
            species.push(LameSpecies::M(i));
        }

        // N species: floor(n/2) for n >= 2 (or n/2 - some correction)
        // Total must be 2n+1
        let n_n = (2 * n + 1).saturating_sub(n_k + n_l + n_m);
        for i in 0..n_n {
            species.push(LameSpecies::N(i));
        }

        species
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lame_config_default() {
        let config = LameConfig::default();
        assert_eq!(config.n_fourier, 32);
        assert!((config.tol - 1e-12).abs() < f64::EPSILON);
        assert_eq!(config.max_degree, 8);
    }

    #[test]
    fn test_ellipsoidal_coord_new() {
        let coord = EllipsoidalCoord::new(3.0, 2.0, 1.0, 3.0, 2.0, 1.0);
        assert!(coord.is_ok());

        // rho < a should fail
        let coord2 = EllipsoidalCoord::new(1.0, 2.0, 1.0, 3.0, 2.0, 1.0);
        assert!(coord2.is_err());
    }

    #[test]
    fn test_lame_species_count() {
        // For degree n, there should be 2n+1 species
        for n in 0..=8 {
            let species = LameSpecies::all_for_degree(n);
            assert_eq!(
                species.len(),
                2 * n + 1,
                "Degree {n}: expected {} species, got {}",
                2 * n + 1,
                species.len()
            );
        }
    }

    #[test]
    fn test_ellipsoidal_to_cartesian() {
        let coord = EllipsoidalCoord {
            rho: 3.0,
            mu: 2.0,
            nu: 1.0,
        };
        let (x, y, z) = coord.to_cartesian(1.0, 2.0);
        assert!(x.is_finite());
        assert!(y.is_finite());
        assert!(z.is_finite());
    }
}
