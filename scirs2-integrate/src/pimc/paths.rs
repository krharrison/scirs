//! Ring polymer path representation for Path Integral Monte Carlo.
//!
//! A ring polymer stores bead positions `beads[particle][slice][dim]`.
//! The `n_slices` beads are connected cyclically in imaginary time with
//! harmonic spring constants `m / tau`.

use crate::pimc::config::PimcConfig;
use scirs2_core::random::{Distribution, Normal, Rng, SeedableRng};

/// Ring polymer representing `N` quantum particles, each discretised into
/// `M` imaginary-time beads living in `d`-dimensional space.
///
/// Indexing: `beads[particle_index][slice_index][dimension_index]`.
#[derive(Debug, Clone)]
pub struct RingPolymer {
    /// Bead positions: `beads[p][s][d]`.
    pub beads: Vec<Vec<Vec<f64>>>,
    /// Number of particles `N`.
    pub n_particles: usize,
    /// Number of imaginary-time slices `M`.
    pub n_slices: usize,
    /// Spatial dimensionality `d`.
    pub dimension: usize,
}

impl RingPolymer {
    /// Construct a ring polymer with all beads initialised to small random
    /// displacements around the origin using a seeded RNG.
    ///
    /// # Arguments
    /// * `config` — PIMC configuration providing `n_particles`, `n_slices`, `dimension`.
    /// * `rng`    — Mutable reference to any `Rng`-implementing generator.
    pub fn new_random<R: Rng>(config: &PimcConfig, rng: &mut R) -> Self {
        let normal = Normal::new(0.0_f64, 0.5_f64).expect("valid normal params");
        let beads = (0..config.n_particles)
            .map(|_| {
                (0..config.n_slices)
                    .map(|_| {
                        (0..config.dimension)
                            .map(|_| normal.sample(rng))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();

        Self {
            beads,
            n_particles: config.n_particles,
            n_slices: config.n_slices,
            dimension: config.dimension,
        }
    }

    /// Kinetic (spring) action of the entire ring polymer.
    ///
    /// ```text
    /// S_K = Σ_{p,s}  (m / (2 τ)) · |r_{p,s+1} − r_{p,s}|²
    /// ```
    /// where slice indices are taken modulo `M` (periodic boundary).
    ///
    /// # Arguments
    /// * `mass` — Particle mass `m`.
    /// * `tau`  — Imaginary-time step `β / M`.
    pub fn kinetic_action(&self, mass: f64, tau: f64) -> f64 {
        let prefactor = mass / (2.0 * tau);
        let mut sum = 0.0_f64;
        for p in 0..self.n_particles {
            for s in 0..self.n_slices {
                let s_next = (s + 1) % self.n_slices;
                let dist_sq = squared_distance(&self.beads[p][s], &self.beads[p][s_next]);
                sum += dist_sq;
            }
        }
        prefactor * sum
    }

    /// Potential action summed over all particles and slices.
    ///
    /// ```text
    /// S_V = τ · Σ_{p,s}  V(r_{p,s})
    /// ```
    ///
    /// # Arguments
    /// * `potential` — External potential `V : ℝ^d → ℝ`.
    /// * `tau`       — Imaginary-time step `β / M`.
    pub fn potential_action(&self, potential: &dyn Fn(&[f64]) -> f64, tau: f64) -> f64 {
        let mut sum = 0.0_f64;
        for p in 0..self.n_particles {
            for s in 0..self.n_slices {
                sum += potential(&self.beads[p][s]);
            }
        }
        tau * sum
    }

    /// Total imaginary-time action `S = S_K + S_V`.
    pub fn total_action(&self, potential: &dyn Fn(&[f64]) -> f64, mass: f64, tau: f64) -> f64 {
        self.kinetic_action(mass, tau) + self.potential_action(potential, tau)
    }

    /// Sample a Lévy bridge between `start` and `end` with `n_beads` intermediate
    /// points (not including the endpoints themselves).
    ///
    /// For the free particle, the conditional distribution of intermediate bead
    /// position `r_k` (at imaginary-time step `k` out of `n_total` steps) given
    /// the two endpoints is Gaussian:
    ///
    /// ```text
    /// mean_k  = start + k/n_total * (end − start)
    /// var_k   = (τ/mass) · k · (n_total − k) / n_total
    /// ```
    ///
    /// The returned `Vec<Vec<f64>>` has length `n_beads`; each element is a
    /// `d`-dimensional position vector.
    ///
    /// # Arguments
    /// * `start`   — Bead position at imaginary-time 0.
    /// * `end`     — Bead position at imaginary-time `(n_beads+1) * tau`.
    /// * `n_beads` — Number of beads to generate (*not* counting endpoints).
    /// * `mass`    — Particle mass.
    /// * `tau`     — Imaginary-time step between consecutive beads.
    /// * `rng`     — Mutable reference to an RNG.
    pub fn levy_bridge<R: Rng>(
        start: &[f64],
        end: &[f64],
        n_beads: usize,
        mass: f64,
        tau: f64,
        rng: &mut R,
    ) -> Vec<Vec<f64>> {
        let dim = start.len();
        let n_total = n_beads + 1; // total number of steps from start to end

        (1..=n_beads)
            .map(|k| {
                let k_f = k as f64;
                let n_f = n_total as f64;
                // Linear interpolation weight
                let alpha = k_f / n_f;
                // Free-particle variance σ² = (τ/m) · k·(n_total−k)/n_total
                let variance = (tau / mass) * k_f * (n_f - k_f) / n_f;
                let std_dev = variance.max(0.0).sqrt();

                (0..dim)
                    .map(|d| {
                        let mean = start[d] + alpha * (end[d] - start[d]);
                        if std_dev > 0.0 {
                            let normal =
                                Normal::new(mean, std_dev).expect("valid bridge normal params");
                            normal.sample(rng)
                        } else {
                            mean
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two `d`-dimensional points stored as slices.
#[inline]
pub(crate) fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi) * (ai - bi))
        .sum()
}

// We expose `seeded_rng` usage pattern needed by other submodules.
pub(crate) use scirs2_core::random::seeded_rng;

/// Type alias for the seeded RNG used throughout the PIMC module.
///
/// Uses `StdRng` (wrapped in `CoreRandom<StdRng>`) exposed by `scirs2_core::random`.
pub(crate) type PimcRng =
    scirs2_core::random::CoreRandom<scirs2_core::random::rand_prelude::StdRng>;

/// Construct a new `PimcRng` seeded deterministically from `seed`.
pub(crate) fn make_rng(seed: u64) -> PimcRng {
    seeded_rng(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pimc::config::PimcConfig;

    fn default_rng() -> PimcRng {
        make_rng(0)
    }

    #[test]
    fn test_ring_polymer_dimensions() {
        let cfg = PimcConfig {
            n_particles: 2,
            n_slices: 8,
            dimension: 3,
            ..Default::default()
        };
        let mut rng = default_rng();
        let poly = RingPolymer::new_random(&cfg, &mut rng);
        assert_eq!(poly.beads.len(), 2);
        assert_eq!(poly.beads[0].len(), 8);
        assert_eq!(poly.beads[0][0].len(), 3);
    }

    #[test]
    fn test_levy_bridge_endpoints_not_included() {
        let start = vec![0.0_f64];
        let end = vec![2.0_f64];
        let mut rng = default_rng();
        let bridge = RingPolymer::levy_bridge(&start, &end, 3, 1.0, 0.1, &mut rng);
        // n_beads = 3 intermediate points
        assert_eq!(bridge.len(), 3);
        // endpoints themselves are NOT in the returned slice
    }

    #[test]
    fn test_levy_bridge_zero_beads() {
        let start = vec![1.0_f64, 2.0_f64];
        let end = vec![3.0_f64, 4.0_f64];
        let mut rng = default_rng();
        let bridge = RingPolymer::levy_bridge(&start, &end, 0, 1.0, 0.05, &mut rng);
        assert!(bridge.is_empty());
    }

    #[test]
    fn test_kinetic_action_zero_for_constant_path() {
        // All beads at the same position → all spring displacements = 0 → S_K = 0
        let cfg = PimcConfig {
            n_particles: 1,
            n_slices: 4,
            dimension: 2,
            ..Default::default()
        };
        let mut poly = RingPolymer {
            beads: vec![vec![vec![1.0_f64, 2.0_f64]; 4]],
            n_particles: cfg.n_particles,
            n_slices: cfg.n_slices,
            dimension: cfg.dimension,
        };
        let _ = &mut poly; // silence warning
        let s_k = poly.kinetic_action(1.0, 0.1);
        assert!(s_k.abs() < 1e-12, "expected zero kinetic action, got {s_k}");
    }

    #[test]
    fn test_potential_action_constant() {
        // V(r) = 1.0 everywhere, tau = 0.25, N=1, M=4 → S_V = 1.0
        let cfg = PimcConfig {
            n_particles: 1,
            n_slices: 4,
            dimension: 1,
            ..Default::default()
        };
        let poly = RingPolymer {
            beads: vec![vec![vec![0.0_f64]; 4]],
            n_particles: cfg.n_particles,
            n_slices: cfg.n_slices,
            dimension: cfg.dimension,
        };
        let tau = 1.0 / 4.0; // beta=1, M=4
        let sv = poly.potential_action(&|_: &[f64]| 1.0, tau);
        // tau * N * M * 1.0 = 0.25 * 1 * 4 = 1.0
        assert!((sv - 1.0).abs() < 1e-12, "expected S_V=1.0, got {sv}");
    }
}
