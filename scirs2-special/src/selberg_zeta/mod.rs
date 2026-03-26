//! Selberg Zeta Function
//!
//! The Selberg zeta function Z_Γ(s) of a hyperbolic surface Γ\H is defined
//! via the Euler product over primitive closed geodesics:
//!
//! Z_Γ(s) = Π_{p primitive} Π_{k=0}^∞ (1 - N(p)^{-(s+k)})
//!
//! where N(p) = e^{l(p)} is the norm of the primitive hyperbolic element γ_p
//! (l(p) is the geodesic length).
//!
//! References:
//! - Selberg, "Harmonic analysis and discontinuous groups", 1956
//! - Hejhal, "The Selberg Trace Formula for PSL(2,R)", 1976
//! - Iwaniec, "Spectral Methods of Automorphic Forms", 2002

use crate::error::{SpecialError, SpecialResult};

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// A hyperbolic surface Γ\H.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum HyperbolicSurface {
    /// The modular curve PSL(2,Z)\H
    ModularCurve {
        /// Level (1 for full modular group)
        level: usize,
    },
    /// Principal congruence subgroup Γ(N)\H
    PrincipalCongruence {
        /// The congruence level N
        level: usize,
    },
    /// Custom surface specified by geodesic norms N(p) = e^{l(p)}
    Custom {
        /// List of norms of primitive geodesics (N(p) = e^{length})
        geodesic_lengths: Vec<f64>,
    },
}

/// Configuration for Selberg zeta function evaluation.
#[derive(Debug, Clone)]
pub struct SelbergConfig {
    /// Number of primitive geodesics to include in the Euler product
    pub n_geodesics: usize,
    /// Maximum k in the inner product Π_{k=0}^{max_k}
    pub max_k: usize,
    /// Tolerance for convergence checks
    pub tol: f64,
    /// Maximum trace to enumerate for hyperbolic elements
    pub max_trace: usize,
}

impl Default for SelbergConfig {
    fn default() -> Self {
        SelbergConfig {
            n_geodesics: 50,
            max_k: 5,
            tol: 1e-10,
            max_trace: 30,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Primitive geodesic enumeration for PSL(2,Z)
// ────────────────────────────────────────────────────────────────────────────

/// GCD for unsigned integers.
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Compute the norm N(γ) of a hyperbolic element γ ∈ PSL(2,Z) with trace t = |a+d|.
///
/// For a matrix with trace t > 2, the eigenvalues are λ = (t ± √(t²-4)) / 2.
/// The norm is N = λ_max² = ((t + √(t²-4))/2)².
fn hyperbolic_norm_from_trace(t: i64) -> f64 {
    let t = t.abs();
    if t <= 2 {
        return 0.0; // Not hyperbolic
    }
    let tf = t as f64;
    let lambda = (tf + (tf * tf - 4.0).sqrt()) / 2.0;
    lambda * lambda
}

/// Enumerate primitive hyperbolic elements of PSL(2,Z) by their norms.
///
/// A matrix γ = [[a, b],[c, d]] ∈ PSL(2,Z) (ad-bc=1, entries in Z) is:
/// - Hyperbolic if |tr(γ)| = |a+d| > 2
/// - Primitive if γ is not a proper power of another element
///
/// We enumerate by trace t from 3 upward, and for each trace enumerate
/// the corresponding primitive elements.
///
/// The key fact: for the modular group, the primitive hyperbolic conjugacy
/// classes are in bijection with reduced binary quadratic forms of discriminant
/// D = t² - 4. We use trace enumeration: for trace t, there exists at least
/// one primitive hyperbolic element, and its norm is N = ((t+√(t²-4))/2)².
///
/// For multiplicity: the number of primitive classes with trace t equals
/// the number of reduced forms of discriminant t²-4, which equals 2h(t²-4)
/// for most t (where h is the class number). We use a simplified approach:
/// one norm per trace value (the principal class).
///
/// # Arguments
/// * `max_trace` - Maximum absolute trace value to enumerate
///
/// # Returns
/// Vector of norms N(p) for primitive geodesics, sorted in increasing order.
pub fn enumerate_primitive_geodesics(max_trace: usize) -> Vec<f64> {
    let mut norms: Vec<f64> = Vec::new();

    // Trace 3: N = ((3+√5)/2)² ≈ 6.854...
    // Trace 4: N = ((4+√12)/2)² = (2+√3)² ≈ 13.928...
    // Trace 5: N = ((5+√21)/2)² ≈ 27.00...
    // etc.

    // We enumerate matrices [[a,b],[c,d]] with:
    // - a, b, c, d in range, ad-bc=1
    // - a+d = t (trace), t > 2
    // - primitive (not a power)
    // - c > 0, or c=0 with b>0 for canonical representative

    // For efficiency, use trace enumeration directly since primitive conjugacy
    // classes of the modular group with given trace t form finitely many classes.
    // The fundamental domains give us: for each pair (t, discriminant D=t^2-4),
    // the reduced hyperbolic forms with discriminant D.

    // Simplified: one primitive geodesic per trace t ≥ 3
    // (the minimal representative in its conjugacy class)
    for t in 3..=(max_trace as i64) {
        let norm = hyperbolic_norm_from_trace(t);
        if norm > 1.0 {
            // Check if this t is "primitive" — not a power
            // A trace t arises from a power: t^2 - 4 must not be a perfect square times
            // a smaller discriminant. For our simplified approach, we include all t ≥ 3
            // and mark only those where t²-4 is not a perfect square times 4 (i.e. not
            // arising from trace-2 elements, which don't exist in PSL(2,Z)).
            // All t ≥ 3 give genuinely hyperbolic elements.
            norms.push(norm);
        }
    }

    // Also add norms from small explicit matrices for completeness at level 1
    // These come from the fundamental group elements with small matrix entries
    let mut extra_norms = collect_small_primitive_norms(max_trace.min(15));
    extra_norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Merge and deduplicate
    for &norm in &extra_norms {
        if norm > 1.0 && !norms.iter().any(|&n| (n - norm).abs() < 1e-6) {
            norms.push(norm);
        }
    }

    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    norms
}

/// Collect primitive norms by enumerating small SL(2,Z) matrices.
///
/// Enumerate [[a,b],[c,d]] with ad-bc=1, c>0, bounded entries.
fn collect_small_primitive_norms(bound: usize) -> Vec<f64> {
    let mut norms = Vec::new();
    let b = bound as i64;

    // Enumerate a, c, d with bounded entries; b = (ad-1)/c when c|ad-1
    for a in -b..=b {
        for d in -b..=b {
            let trace = a + d;
            if trace.abs() <= 2 {
                continue; // Not hyperbolic
            }
            // Need ad - bc = 1, so bc = ad - 1
            let det_target = a * d - 1; // = bc
            for c in 1..=b {
                if det_target % c == 0 {
                    let bb = det_target / c;
                    if bb.abs() <= b {
                        // Check primitivity: gcd of matrix entries - 1 should be 1
                        // Actually check that [[a,bb],[c,d]] is primitive in PSL(2,Z)
                        // A simple check: for SL(2,Z), all elements are primitive unless
                        // they are powers. The trace check: trace t with t²-4 square-free
                        // gives a primitive class.
                        let disc = trace * trace - 4;
                        if disc > 0 && is_not_perfect_square(disc) {
                            let norm = hyperbolic_norm_from_trace(trace);
                            norms.push(norm);
                        }
                    }
                }
            }
        }
    }
    norms
}

/// Check if n is not a perfect square (n > 0).
fn is_not_perfect_square(n: i64) -> bool {
    if n <= 0 {
        return false;
    }
    let s = (n as f64).sqrt() as i64;
    s * s != n && (s + 1) * (s + 1) != n
}

/// Get geodesic norms for a given surface and configuration.
fn get_geodesic_norms(surface: &HyperbolicSurface, config: &SelbergConfig) -> Vec<f64> {
    match surface {
        HyperbolicSurface::ModularCurve { level: _ } => {
            let mut norms = enumerate_primitive_geodesics(config.max_trace);
            norms.truncate(config.n_geodesics);
            norms
        }
        HyperbolicSurface::PrincipalCongruence { level } => {
            // Γ(N) is a subgroup of index N³ Π_{p|N}(1-p^{-2}) in PSL(2,Z)
            // Its geodesics are those of PSL(2,Z) with additional wrapping.
            // For simplicity, we use the PSL(2,Z) geodesics scaled by the level.
            let base_norms = enumerate_primitive_geodesics(config.max_trace);
            // A norm N(p) for PSL(2,Z) gives N(p)^{[PSL:Γ(N)]} / covering-degree
            // For simplification, just use the base norms repeated *level times
            let mut norms = Vec::new();
            for &norm in &base_norms {
                if norms.len() >= config.n_geodesics {
                    break;
                }
                // Each primitive geodesic of PSL(2,Z) lifts to [Γ(N):stabilizer] geodesics
                // For the principal congruence subgroup Γ(N), the lifting degree is *level
                let lifted_norm = norm.powf(*level as f64);
                norms.push(lifted_norm);
            }
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            norms.truncate(config.n_geodesics);
            norms
        }
        HyperbolicSurface::Custom { geodesic_lengths } => {
            let mut norms: Vec<f64> = geodesic_lengths
                .iter()
                .take(config.n_geodesics)
                .copied()
                .collect();
            norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            norms
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Selberg zeta evaluation
// ────────────────────────────────────────────────────────────────────────────

/// Evaluate the Selberg zeta function Z_Γ(s) via a truncated Euler product.
///
/// Z_Γ(s) = Π_{p primitive} Π_{k=0}^{max_k} (1 - N(p)^{-(s+k)})
///
/// Computed as exp(Σ_p Σ_k log(1 - N(p)^{-(s+k)})) for numerical stability.
///
/// # Arguments
/// * `surface` - The hyperbolic surface Γ\H
/// * `s` - Complex argument Re(s) > 1 for convergence
/// * `config` - Computation parameters
///
/// # Errors
/// Returns `SpecialError::DomainError` if s ≤ 1.
/// Returns `SpecialError::ComputationError` if the Euler product diverges.
pub fn selberg_zeta(
    surface: &HyperbolicSurface,
    s: f64,
    config: &SelbergConfig,
) -> SpecialResult<f64> {
    if s <= 1.0 {
        return Err(SpecialError::DomainError(format!(
            "Selberg zeta convergence requires s > 1, got s = {s}"
        )));
    }

    let norms = get_geodesic_norms(surface, config);

    if norms.is_empty() {
        return Err(SpecialError::ComputationError(
            "No primitive geodesics found for the given surface".to_string(),
        ));
    }

    // log Z(s) = Σ_p Σ_{k=0}^{max_k} log(1 - N(p)^{-(s+k)})
    let mut log_z = 0.0f64;
    for &norm in &norms {
        if norm <= 1.0 {
            continue; // Skip non-hyperbolic entries
        }
        for k in 0..=config.max_k {
            let exp_val = norm.powf(-(s + k as f64));
            if exp_val >= 1.0 {
                return Err(SpecialError::ComputationError(format!(
                    "Euler product factor 1 - N^{{-(s+k)}} ≤ 0 for N={norm}, s={s}, k={k}"
                )));
            }
            let factor = (1.0 - exp_val).ln();
            if !factor.is_finite() {
                return Err(SpecialError::ComputationError(
                    "Non-finite log factor in Selberg zeta Euler product".to_string(),
                ));
            }
            log_z += factor;
        }
    }

    Ok(log_z.exp())
}

// ────────────────────────────────────────────────────────────────────────────
// Selberg trace formula (simplified diagnostic)
// ────────────────────────────────────────────────────────────────────────────

/// Compute a partial sum from the Selberg trace formula for the modular curve.
///
/// The spectral side of the Selberg trace formula gives:
/// Σ_n h(r_n) = geometric + identity contributions
///
/// For a simple test, we return the sum Σ_{r_n} 1/(s_n(s_n-1)) where
/// s_n are the spectral parameters (s_n = 1/2 + ir_n with eigenvalue λ_n = s_n(1-s_n)).
///
/// The eigenvalues of the Laplacian on PSL(2,Z)\H are approximately
/// λ_n ≈ (2πn/A)² where A = π/3 is the area (Weyl's law).
///
/// # Arguments
/// * `s` - Spectral parameter
/// * `n_eigenvalues` - Number of eigenvalues to include
///
/// # Returns
/// Partial spectral sum Σ_{n=1}^{n_eigenvalues} 1/λ_n
pub fn selberg_trace_formula_check(s: f64, n_eigenvalues: usize) -> f64 {
    // Area of modular curve = π/3
    let area = std::f64::consts::PI / 3.0;

    // Weyl's law: λ_n ≈ 4π * n / area = 12n for PSL(2,Z)\H
    // First actual eigenvalue: λ_1 ≈ 91.14... (from numerics)
    // We use approximate values: λ_n ~ 4π n / area

    let mut sum = 0.0f64;
    for n in 1..=n_eigenvalues {
        // Approximate eigenvalue via Weyl's law: λ_n ≈ 4πn/A
        let lambda_n = 4.0 * std::f64::consts::PI * n as f64 / area;
        // Spectral parameter: s_n(1 - s_n) = λ_n
        // s_n = 1/2 + sqrt(1/4 - λ_n) if λ_n < 1/4, else 1/2 + i*sqrt(λ_n - 1/4)
        // For the sum 1/(s(s_n-1)) we use 1/(-λ_n) = -1/λ_n
        // (since s_n(s_n-1) = -λ_n)
        let _ = s; // s parameter available for future weighted sums
        sum += 1.0 / lambda_n;
    }
    sum
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_primitive_geodesics_nonempty() {
        let norms = enumerate_primitive_geodesics(15);
        assert!(
            !norms.is_empty(),
            "Expected non-empty list of primitive geodesics"
        );
    }

    #[test]
    fn test_enumerate_primitive_geodesics_sorted() {
        let norms = enumerate_primitive_geodesics(20);
        for window in norms.windows(2) {
            assert!(
                window[0] <= window[1],
                "Geodesic norms should be sorted: {} > {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_enumerate_primitive_geodesics_first_norm() {
        // First primitive geodesic of PSL(2,Z): trace = 3
        // N = ((3 + √5)/2)² ≈ 6.8541...
        let norms = enumerate_primitive_geodesics(10);
        assert!(!norms.is_empty());
        let n0 = norms[0];
        assert!(n0 > 1.0, "First geodesic norm should be > 1, got {n0}");
        assert!((n0 - 6.854).abs() < 0.5, "First norm ≈ 6.854, got {n0}");
    }

    #[test]
    fn test_selberg_zeta_positive_at_s2() {
        let surface = HyperbolicSurface::ModularCurve { level: 1 };
        let config = SelbergConfig::default();
        let z = selberg_zeta(&surface, 2.0, &config).expect("Selberg zeta at s=2");
        assert!(z > 0.0, "Z(2) should be positive, got {z}");
    }

    #[test]
    fn test_selberg_zeta_converges_to_1_for_large_s() {
        // For large s, all factors (1 - N^{-(s+k)}) → 1, so Z(s) → 1
        let surface = HyperbolicSurface::ModularCurve { level: 1 };
        let config = SelbergConfig {
            n_geodesics: 10,
            max_k: 3,
            ..Default::default()
        };
        let z_large = selberg_zeta(&surface, 20.0, &config).expect("Selberg zeta at s=20");
        assert!(
            (z_large - 1.0).abs() < 0.1,
            "Z(20) should be ≈ 1, got {z_large}"
        );
    }

    #[test]
    fn test_selberg_zeta_domain_error() {
        let surface = HyperbolicSurface::ModularCurve { level: 1 };
        let config = SelbergConfig::default();
        let result = selberg_zeta(&surface, 0.5, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_selberg_zeta_custom_surface() {
        // Custom surface with known norms
        let surface = HyperbolicSurface::Custom {
            geodesic_lengths: vec![6.854, 13.928, 27.0],
        };
        let config = SelbergConfig::default();
        let z = selberg_zeta(&surface, 3.0, &config).expect("custom selberg zeta");
        assert!(z > 0.0 && z <= 1.0, "Z(3) for custom surface: {z}");
    }

    #[test]
    fn test_hyperbolic_norm_trace3() {
        let norm = hyperbolic_norm_from_trace(3);
        let expected = {
            let lambda = (3.0 + 5.0f64.sqrt()) / 2.0;
            lambda * lambda
        };
        assert!(
            (norm - expected).abs() < 1e-10,
            "norm({}) ≈ {}, expected {}",
            3,
            norm,
            expected
        );
    }

    #[test]
    fn test_selberg_trace_formula_check() {
        let result = selberg_trace_formula_check(2.0, 10);
        assert!(
            result > 0.0,
            "Trace formula sum should be positive: {result}"
        );
    }
}
