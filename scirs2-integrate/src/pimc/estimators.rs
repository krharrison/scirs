//! Energy estimators for Path Integral Monte Carlo.
//!
//! Implements the **thermodynamic estimator** for total, kinetic, and potential
//! energy.  Accumulators are updated after every accepted (or proposed) sweep
//! at the caller's discretion.
//!
//! ## Thermodynamic energy estimator
//!
//! For a path integral with `N` particles, `M` slices, imaginary-time step
//! `τ = β/M`, and dimension `d`:
//!
//! ```text
//! ⟨E_kinetic⟩  ≈  d·N / (2τ) − (1/M) ∑_{p,s} (m/τ²) |r_{p,s+1}−r_{p,s}|² / 2
//! ⟨E_potential⟩ ≈  (1/M) ∑_{p,s} V(r_{p,s})
//! ⟨E_total⟩    ≈  ⟨E_kinetic⟩ + ⟨E_potential⟩
//! ```
//!
//! This is the standard thermodynamic (primitive) estimator.  The spring-energy
//! contribution gives the "quantum pressure" correction to the classical kinetic
//! energy `d·N·k_BT/2`.

use crate::pimc::paths::{squared_distance, RingPolymer};

/// Accumulates energy samples and computes running averages.
#[derive(Debug, Clone)]
pub struct EnergyEstimator {
    energy_sum: f64,
    energy_sq_sum: f64, // for variance
    kinetic_sum: f64,
    potential_sum: f64,
    /// Number of accumulated samples.
    pub count: u64,
}

impl Default for EnergyEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyEstimator {
    /// Create a fresh estimator with all accumulators at zero.
    pub fn new() -> Self {
        Self {
            energy_sum: 0.0,
            energy_sq_sum: 0.0,
            kinetic_sum: 0.0,
            potential_sum: 0.0,
            count: 0,
        }
    }

    /// Accumulate one energy sample from the current polymer configuration.
    ///
    /// # Arguments
    /// * `polymer`   — Current ring-polymer state.
    /// * `potential` — External potential `V : ℝ^d → ℝ`.
    /// * `mass`      — Particle mass `m`.
    /// * `tau`       — Imaginary-time step `β/M`.
    pub fn accumulate(
        &mut self,
        polymer: &RingPolymer,
        potential: &dyn Fn(&[f64]) -> f64,
        mass: f64,
        tau: f64,
    ) {
        let n = polymer.n_particles as f64;
        let m = polymer.n_slices as f64;
        let d = polymer.dimension as f64;

        // ── Kinetic (thermodynamic estimator) ───────────────────────────────
        // Classical part: d·N/(2τ)
        // Quantum correction: −(m / (2 τ²)) · (1/M) · ∑_{p,s} |Δr|²
        let spring_sq_sum: f64 = (0..polymer.n_particles)
            .flat_map(|p| {
                (0..polymer.n_slices).map(move |s| {
                    let s_next = (s + 1) % polymer.n_slices;
                    squared_distance(&polymer.beads[p][s], &polymer.beads[p][s_next])
                })
            })
            .sum();

        let kinetic = d * n / (2.0 * tau) - (mass / (2.0 * tau * tau)) * spring_sq_sum / m;

        // ── Potential ────────────────────────────────────────────────────────
        // ⟨V⟩ = (1/M) ∑_{p,s} V(r_{p,s})
        let pot_sum: f64 = (0..polymer.n_particles)
            .flat_map(|p| (0..polymer.n_slices).map(move |s| potential(&polymer.beads[p][s])))
            .sum();
        let potential_energy = pot_sum / m;

        let energy = kinetic + potential_energy;

        self.kinetic_sum += kinetic;
        self.potential_sum += potential_energy;
        self.energy_sum += energy;
        self.energy_sq_sum += energy * energy;
        self.count += 1;
    }

    /// Mean total energy over all accumulated samples.
    ///
    /// Returns `0.0` if no samples have been collected.
    pub fn mean_energy(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.energy_sum / self.count as f64
    }

    /// Mean kinetic energy over all accumulated samples.
    pub fn mean_kinetic(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.kinetic_sum / self.count as f64
    }

    /// Mean potential energy over all accumulated samples.
    pub fn mean_potential(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.potential_sum / self.count as f64
    }

    /// Sample variance of the total energy: `⟨E²⟩ − ⟨E⟩²`.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.energy_sum / n;
        let mean_sq = self.energy_sq_sum / n;
        (mean_sq - mean * mean).max(0.0)
    }

    /// Standard error of the mean total energy: `sqrt(Var / n_samples)`.
    ///
    /// `n_samples` is the number of independent samples used in the estimate.
    /// Passing `self.count` gives the naive standard error.
    pub fn std_energy(&self, n_samples: u64) -> f64 {
        if n_samples < 2 {
            return 0.0;
        }
        (self.variance() / n_samples as f64).sqrt()
    }

    /// Reset all accumulators to zero.
    pub fn reset(&mut self) {
        self.energy_sum = 0.0;
        self.energy_sq_sum = 0.0;
        self.kinetic_sum = 0.0;
        self.potential_sum = 0.0;
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pimc::{config::PimcConfig, paths::RingPolymer};

    fn flat_polymer(n_particles: usize, n_slices: usize, dim: usize, val: f64) -> RingPolymer {
        RingPolymer {
            beads: vec![vec![vec![val; dim]; n_slices]; n_particles],
            n_particles,
            n_slices,
            dimension: dim,
        }
    }

    #[test]
    fn test_estimator_initial_count() {
        let est = EnergyEstimator::new();
        assert_eq!(est.count, 0);
    }

    #[test]
    fn test_estimator_after_one_accumulate() {
        let mut est = EnergyEstimator::new();
        let poly = flat_polymer(1, 4, 1, 0.0);
        let tau = 0.25;
        est.accumulate(&poly, &|_| 0.0, 1.0, tau);
        assert_eq!(est.count, 1);
    }

    #[test]
    fn test_estimator_reset() {
        let mut est = EnergyEstimator::new();
        let poly = flat_polymer(1, 4, 1, 0.0);
        est.accumulate(&poly, &|_| 0.5, 1.0, 0.25);
        assert_ne!(est.count, 0);
        est.reset();
        assert_eq!(est.count, 0);
        assert_eq!(est.mean_energy(), 0.0);
    }

    #[test]
    fn test_estimator_constant_path_zero_potential() {
        // All beads at origin, V=0 → spring terms = 0
        // Kinetic = d*N/(2*tau) = 1*1/(2*0.25) = 2.0
        let mut est = EnergyEstimator::new();
        let poly = flat_polymer(1, 4, 1, 0.0);
        let tau = 0.25; // beta=1, M=4
        est.accumulate(&poly, &|_| 0.0, 1.0, tau);
        let expected_kin = 1.0 * 1.0 / (2.0 * tau); // d*N/(2*tau)
        assert!(
            (est.mean_kinetic() - expected_kin).abs() < 1e-10,
            "expected kinetic {expected_kin}, got {}",
            est.mean_kinetic()
        );
        assert!(est.mean_potential().abs() < 1e-12);
    }
}
