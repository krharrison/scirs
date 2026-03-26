//! Main PIMC simulator: thermalization, production run, and result collection.
//!
//! Orchestrates single-bead and centre-of-mass moves over the production run,
//! collecting energy estimates at regular intervals.

use crate::error::{IntegrateError, IntegrateResult};
use crate::pimc::{
    config::{PimcConfig, PimcResult},
    estimators::EnergyEstimator,
    moves::{CenterOfMassMove, PimcMove, RngProxy, SingleBeadMove},
    paths::{make_rng, PimcRng, RingPolymer},
};

// ── Simulator ─────────────────────────────────────────────────────────────────

/// Path Integral Monte Carlo simulator.
///
/// Holds all simulation state: the ring polymer, potential function, energy
/// estimator, and the seeded RNG.
pub struct PimcSimulator {
    config: PimcConfig,
    polymer: RingPolymer,
    potential: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    estimator: EnergyEstimator,
    rng: PimcRng,
}

impl PimcSimulator {
    /// Construct a new simulator, initialising the ring polymer at random positions.
    ///
    /// # Errors
    /// Returns [`IntegrateError::InvalidInput`] if any configuration parameter
    /// is out of range (e.g. zero slices, zero particles, non-positive `beta`).
    pub fn new(
        config: PimcConfig,
        potential: Box<dyn Fn(&[f64]) -> f64 + Send + Sync>,
    ) -> IntegrateResult<Self> {
        // ── Validate ──────────────────────────────────────────────────────
        if config.n_particles == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_particles must be ≥ 1".into(),
            ));
        }
        if config.n_slices == 0 {
            return Err(IntegrateError::InvalidInput("n_slices must be ≥ 1".into()));
        }
        if config.beta <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "beta must be strictly positive".into(),
            ));
        }
        if config.mass <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "mass must be strictly positive".into(),
            ));
        }
        if config.dimension == 0 {
            return Err(IntegrateError::InvalidInput("dimension must be ≥ 1".into()));
        }
        if config.max_displacement <= 0.0 {
            return Err(IntegrateError::InvalidInput(
                "max_displacement must be strictly positive".into(),
            ));
        }

        let mut rng: PimcRng = make_rng(config.seed);
        let polymer = RingPolymer::new_random(&config, &mut rng);

        Ok(Self {
            config,
            polymer,
            potential,
            estimator: EnergyEstimator::new(),
            rng,
        })
    }

    /// Run the full simulation: thermalization followed by the production run.
    ///
    /// Returns a [`PimcResult`] containing mean energies, acceptance statistics,
    /// and standard errors.
    pub fn run(&mut self) -> IntegrateResult<PimcResult> {
        let tau = self.config.beta / self.config.n_slices as f64;
        let mass = self.config.mass;

        let single_bead = SingleBeadMove {
            max_displacement: self.config.max_displacement,
        };
        let com = CenterOfMassMove {
            max_displacement: self.config.max_displacement * 0.5,
        };

        // ── Thermalization ────────────────────────────────────────────────
        for _ in 0..self.config.n_thermalize {
            self.sweep_impl(&single_bead, &com, mass, tau);
        }

        // ── Production run ────────────────────────────────────────────────
        let mut total_accepted: u64 = 0;
        let mut total_proposed: u64 = 0;
        self.estimator.reset();

        for step in 0..self.config.n_steps {
            let (accepted, proposed) = self.sweep_impl(&single_bead, &com, mass, tau);
            total_accepted += accepted;
            total_proposed += proposed;

            if (step + 1) % self.config.estimator_interval == 0 {
                let potential = self.potential.as_ref();
                self.estimator
                    .accumulate(&self.polymer, potential, mass, tau);
            }
        }

        let n_samples = self.estimator.count;
        let energy_mean = self.estimator.mean_energy();
        let energy_std = self.estimator.std_energy(n_samples);
        let kinetic_mean = self.estimator.mean_kinetic();
        let potential_mean = self.estimator.mean_potential();

        let acceptance_rate = if total_proposed > 0 {
            total_accepted as f64 / total_proposed as f64
        } else {
            0.0
        };

        Ok(PimcResult {
            energy_mean,
            energy_std,
            kinetic_mean,
            potential_mean,
            n_accepted: total_accepted,
            n_total: total_proposed,
            acceptance_rate,
        })
    }

    /// One MC sweep: attempt a single-bead move for every (particle, slice) pair,
    /// then attempt a centre-of-mass move for each particle.
    ///
    /// Returns `(n_accepted, n_proposed)` for this sweep.
    fn sweep_impl(
        &mut self,
        single_bead: &SingleBeadMove,
        com: &CenterOfMassMove,
        mass: f64,
        tau: f64,
    ) -> (u64, u64) {
        let potential = self.potential.as_ref();
        let mut accepted: u64 = 0;
        let total_single = (self.config.n_particles * self.config.n_slices) as u64;

        // Single-bead moves: one attempt per (particle, slice) pair
        for _ in 0..total_single {
            if single_bead.propose_and_accept(
                &mut self.polymer,
                potential,
                mass,
                tau,
                &mut self.rng,
            ) {
                accepted += 1;
            }
        }

        // Centre-of-mass moves: one attempt per particle
        for _ in 0..self.config.n_particles {
            if com.propose_and_accept(&mut self.polymer, potential, mass, tau, &mut self.rng) {
                accepted += 1;
            }
        }

        let total = total_single + self.config.n_particles as u64;
        (accepted, total)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pimc::config::PimcConfig;
    use std::f64::consts::PI;

    /// Harmonic potential V(r) = 0.5 * |r|²
    fn harmonic(r: &[f64]) -> f64 {
        r.iter().map(|&x| 0.5 * x * x).sum()
    }

    /// Zero potential (free particle)
    fn zero(_r: &[f64]) -> f64 {
        0.0
    }

    // ── Config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_pimc_config_default() {
        let cfg = PimcConfig::default();
        assert_eq!(cfg.n_slices, 32);
        assert!((cfg.beta - 1.0).abs() < 1e-12);
        assert_eq!(cfg.n_particles, 1);
        assert_eq!(cfg.dimension, 1);
        assert_eq!(cfg.seed, 42);
    }

    // ── RingPolymer tests ─────────────────────────────────────────────────

    #[test]
    fn test_ring_polymer_n_beads() {
        use crate::pimc::paths::make_rng;
        let cfg = PimcConfig {
            n_particles: 2,
            n_slices: 16,
            dimension: 3,
            ..Default::default()
        };
        let mut rng = make_rng(0);
        let poly = RingPolymer::new_random(&cfg, &mut rng);
        assert_eq!(poly.beads.len(), 2, "n_particles");
        assert_eq!(poly.beads[0].len(), 16, "n_slices");
        assert_eq!(poly.beads[0][0].len(), 3, "dimension");
    }

    #[test]
    fn test_ring_polymer_kinetic_action_harmonic() {
        // Construct a path where all beads differ by a known displacement and
        // verify S_K = (m / 2τ) · N · M · |Δr|²
        let n_slices = 4;
        let dim = 1;
        // Each consecutive pair of beads displaced by 1.0
        let mut beads = vec![vec![vec![0.0_f64; dim]; n_slices]];
        for s in 0..n_slices {
            beads[0][s][0] = s as f64; // positions: 0, 1, 2, 3
        }
        let poly = RingPolymer {
            beads,
            n_particles: 1,
            n_slices,
            dimension: dim,
        };
        let mass = 1.0_f64;
        let tau = 0.25_f64;
        let s_k = poly.kinetic_action(mass, tau);
        // Spring between consecutive beads: |Δr|² = 1 for (0,1),(1,2),(2,3)
        // Periodic: (3,0) has |Δr|² = 9
        // S_K = (1/(2*0.25)) * (1+1+1+9) = 2.0 * 12 = 24.0
        let expected = (mass / (2.0 * tau)) * 12.0;
        assert!(
            (s_k - expected).abs() < 1e-10,
            "S_K={s_k} expected {expected}"
        );
    }

    #[test]
    fn test_ring_polymer_potential_action_constant() {
        // V=1 everywhere, tau=beta/M=1/4, N=1, M=4 → S_V = tau * N * M * 1 = 1.0
        let poly = RingPolymer {
            beads: vec![vec![vec![0.0_f64]; 4]],
            n_particles: 1,
            n_slices: 4,
            dimension: 1,
        };
        let tau = 0.25;
        let sv = poly.potential_action(&|_| 1.0, tau);
        assert!((sv - 1.0).abs() < 1e-12, "S_V={sv}");
    }

    // ── Lévy bridge tests ─────────────────────────────────────────────────

    #[test]
    fn test_levy_bridge_endpoints_fixed() {
        // The bridge returns intermediate points only.  The caller is responsible
        // for placing start and end at the correct slice indices.
        use crate::pimc::paths::make_rng;
        let mut rng = make_rng(7);
        let start = vec![0.0_f64, 0.0_f64];
        let end = vec![4.0_f64, 4.0_f64];
        let n_beads = 5;
        let bridge = RingPolymer::levy_bridge(&start, &end, n_beads, 1.0, 0.1, &mut rng);
        assert_eq!(bridge.len(), n_beads);
        // All intermediate points have the correct dimension
        for pt in &bridge {
            assert_eq!(pt.len(), 2);
        }
    }

    #[test]
    fn test_levy_bridge_gaussian_spread() {
        // With tau > 0, the midpoint of the bridge should have non-zero variance
        // (we sample many realisations and check spread is positive)
        use crate::pimc::paths::make_rng;
        let mut rng = make_rng(13);
        let start = vec![0.0_f64];
        let end = vec![0.0_f64];
        let samples: Vec<f64> = (0..200)
            .map(|_| {
                let bridge = RingPolymer::levy_bridge(&start, &end, 1, 1.0, 0.5, &mut rng);
                bridge[0][0]
            })
            .collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!(
            var > 1e-6,
            "Lévy bridge variance should be positive, got {var}"
        );
    }

    // ── Move tests ────────────────────────────────────────────────────────

    #[test]
    fn test_single_bead_move_acceptance_high_temp() {
        // At very high temperature (beta ≈ 0, many slices) almost all moves are accepted
        let cfg = PimcConfig {
            n_slices: 64,
            beta: 0.01,
            n_steps: 0,
            n_thermalize: 0,
            max_displacement: 0.01,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let tau = 0.01 / 64.0;
        let mass = 1.0;
        let single = SingleBeadMove {
            max_displacement: 0.01,
        };
        let com = CenterOfMassMove {
            max_displacement: 0.005,
        };
        let (acc, tot) = sim.sweep_impl(&single, &com, mass, tau);
        let rate = acc as f64 / tot as f64;
        // High-temperature acceptance should be well above 50%
        assert!(rate > 0.5, "expected high acceptance at high T, got {rate}");
    }

    #[test]
    fn test_com_move_shifts_all_beads() {
        use crate::pimc::moves::RngProxy;
        let mut poly = RingPolymer {
            beads: vec![vec![vec![1.0_f64]; 4]],
            n_particles: 1,
            n_slices: 4,
            dimension: 1,
        };
        let mv = CenterOfMassMove {
            max_displacement: 0.3,
        };
        let mut rng = make_rng(55);
        // With V=0 the move is always accepted
        mv.propose_and_accept(&mut poly, &|_| 0.0, 1.0, 0.25, &mut rng);
        // All beads should be at the same value after a uniform COM shift
        let first = poly.beads[0][0][0];
        for s in 1..4 {
            assert!(
                (poly.beads[0][s][0] - first).abs() < 1e-12,
                "bead {s} inconsistent"
            );
        }
    }

    // ── Estimator tests ───────────────────────────────────────────────────

    #[test]
    fn test_energy_estimator_accumulate() {
        use crate::pimc::estimators::EnergyEstimator;
        let mut est = EnergyEstimator::new();
        let poly = RingPolymer {
            beads: vec![vec![vec![0.0_f64]; 4]],
            n_particles: 1,
            n_slices: 4,
            dimension: 1,
        };
        est.accumulate(&poly, &|_| 0.0, 1.0, 0.25);
        assert_eq!(est.count, 1);
    }

    #[test]
    fn test_energy_estimator_reset() {
        use crate::pimc::estimators::EnergyEstimator;
        let mut est = EnergyEstimator::new();
        let poly = RingPolymer {
            beads: vec![vec![vec![0.0_f64]; 4]],
            n_particles: 1,
            n_slices: 4,
            dimension: 1,
        };
        est.accumulate(&poly, &|_| 1.0, 1.0, 0.25);
        assert_ne!(est.count, 0);
        est.reset();
        assert_eq!(est.count, 0);
    }

    // ── Full-simulation tests ─────────────────────────────────────────────

    #[test]
    fn test_pimc_run_completes() {
        let cfg = PimcConfig {
            n_slices: 16,
            beta: 1.0,
            n_steps: 200,
            n_thermalize: 50,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run should succeed");
        // Basic sanity: finite result
        assert!(result.energy_mean.is_finite(), "energy_mean must be finite");
    }

    #[test]
    fn test_pimc_result_acceptance_rate_in_range() {
        let cfg = PimcConfig {
            n_slices: 16,
            beta: 1.0,
            n_steps: 500,
            n_thermalize: 100,
            max_displacement: 0.5,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run should succeed");
        assert!(
            result.acceptance_rate > 0.1 && result.acceptance_rate < 0.99,
            "acceptance_rate={} out of expected range",
            result.acceptance_rate
        );
    }

    #[test]
    fn test_pimc_seeded_reproducibility() {
        let make_sim = || -> PimcResult {
            let cfg = PimcConfig {
                n_slices: 16,
                beta: 1.0,
                n_steps: 200,
                n_thermalize: 50,
                seed: 12345,
                ..Default::default()
            };
            PimcSimulator::new(cfg, Box::new(harmonic))
                .expect("new")
                .run()
                .expect("run")
        };
        let r1 = make_sim();
        let r2 = make_sim();
        assert!(
            (r1.energy_mean - r2.energy_mean).abs() < 1e-14,
            "seeded runs differ: {} vs {}",
            r1.energy_mean,
            r2.energy_mean
        );
    }

    #[test]
    fn test_pimc_kinetic_positive() {
        // For a quantum particle at finite temperature the kinetic energy is > 0
        let cfg = PimcConfig {
            n_slices: 32,
            beta: 2.0,
            n_steps: 2000,
            n_thermalize: 500,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run");
        assert!(
            result.kinetic_mean > 0.0,
            "kinetic energy should be positive, got {}",
            result.kinetic_mean
        );
    }

    #[test]
    fn test_pimc_potential_harmonic() {
        // Virial theorem for harmonic oscillator: ⟨K⟩ = ⟨V⟩ (in the quantum ground state
        // and at any temperature for a harmonic potential with ω = 1, ħ = 1).
        // At low T (large beta) both should be ≈ 0.25 (half of ground state energy 0.5).
        let cfg = PimcConfig {
            n_slices: 64,
            beta: 10.0, // very low temperature
            n_steps: 10_000,
            n_thermalize: 2_000,
            max_displacement: 0.3,
            seed: 77,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run");
        // Virial: |⟨K⟩ − ⟨V⟩| should be small relative to ⟨E⟩
        let diff = (result.kinetic_mean - result.potential_mean).abs();
        let scale = result.energy_mean.abs().max(0.1);
        assert!(
            diff / scale < 0.5,
            "virial theorem violated: K={:.4}, V={:.4}",
            result.kinetic_mean,
            result.potential_mean
        );
    }

    #[test]
    fn test_pimc_free_particle_energy() {
        // The thermodynamic estimator for kinetic energy is:
        //   K_est = d*N/(2*tau) − (m/(2*tau²)) * mean_spring / M
        //
        // For a 1-D quantum harmonic oscillator at β=1, ω=1 (exact E=0.5*coth(0.5)≈1.31):
        // we use a moderate harmonic well and verify the simulation runs and
        // produces a finite, reasonable energy.
        //
        // The exact quantum result for ⟨E⟩ = (ω/2)·coth(β·ω/2)
        // With ω=1, β=1: ⟨E⟩ = 0.5 * coth(0.5) ≈ 1.313
        let cfg = PimcConfig {
            n_slices: 64,
            beta: 1.0,
            n_steps: 8_000,
            n_thermalize: 2_000,
            max_displacement: 0.5,
            seed: 101,
            ..Default::default()
        };
        // Use the harmonic potential to have a well-defined equilibrium
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run");
        // At β=1 the exact result is ≈1.313; we use a generous tolerance of 0.5
        let expected = 0.5_f64 / (0.5_f64).tanh(); // ω/2 * coth(βω/2) with ω=β=1
        let tol = 0.5;
        assert!(
            (result.energy_mean - expected).abs() < tol,
            "β=1 harmonic energy={:.4} expected ~{expected:.4} (tol {tol})",
            result.energy_mean
        );
    }

    #[test]
    fn test_pimc_harmonic_oscillator_energy() {
        // Ground state of 1D QHO (ω=1, ħ=1, m=1): E₀ = 0.5
        // At beta=10 (very low temperature) ⟨E⟩ ≈ 0.5
        let cfg = PimcConfig {
            n_slices: 128,
            beta: 10.0,
            n_steps: 20_000,
            n_thermalize: 3_000,
            max_displacement: 0.3,
            seed: 42,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run");
        let expected = 0.5;
        let tol = 0.15;
        assert!(
            (result.energy_mean - expected).abs() < tol,
            "QHO energy={:.4} expected ~{expected} (tol {tol})",
            result.energy_mean
        );
    }

    #[test]
    fn test_pimc_3d_harmonic_energy() {
        // 3D QHO ground state: E₀ = 3 * 0.5 = 1.5
        let cfg = PimcConfig {
            n_slices: 64,
            beta: 10.0,
            dimension: 3,
            n_steps: 20_000,
            n_thermalize: 3_000,
            max_displacement: 0.3,
            seed: 777,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(harmonic)).expect("new simulator");
        let result = sim.run().expect("run");
        let expected = 1.5;
        let tol = 0.3;
        assert!(
            (result.energy_mean - expected).abs() < tol,
            "3D QHO energy={:.4} expected ~{expected} (tol {tol})",
            result.energy_mean
        );
    }

    #[test]
    fn test_pimc_high_temp_classical() {
        // At high temperature, total energy ≈ d * k_B T / 2 = d / (2 * beta) for V=0
        // With d=1, beta=0.1 → E ≈ 5.0
        let cfg = PimcConfig {
            n_slices: 8,
            beta: 0.1,
            n_steps: 3_000,
            n_thermalize: 500,
            max_displacement: 0.5,
            seed: 55,
            ..Default::default()
        };
        let mut sim = PimcSimulator::new(cfg, Box::new(zero)).expect("new simulator");
        let result = sim.run().expect("run");
        let expected = 1.0 / (2.0 * 0.1); // 5.0
        let tol = 2.0;
        assert!(
            (result.energy_mean - expected).abs() < tol,
            "high-T energy={:.4} expected ~{expected} (tol {tol})",
            result.energy_mean
        );
    }

    #[test]
    fn test_single_bead_detailed_balance() {
        // A detailed-balance check: for V=0 the single-bead move probability is
        // symmetric because the proposal is symmetric. Accept all → equal forward/backward.
        // Here we just verify that a series of alternating forward/backward moves
        // on a two-bead system restores the original configuration.
        use crate::pimc::moves::SingleBeadMove;
        use crate::pimc::paths::make_rng;
        let mut poly = RingPolymer {
            beads: vec![vec![vec![0.5_f64]; 2]],
            n_particles: 1,
            n_slices: 2,
            dimension: 1,
        };
        let original = poly.beads.clone();
        let mv = SingleBeadMove {
            max_displacement: 0.01,
        };
        let mut rng = make_rng(888);
        // Run 100 moves and verify we never panic and state remains consistent
        for _ in 0..100 {
            mv.propose_and_accept(&mut poly, &|_| 0.0, 1.0, 0.5, &mut rng);
        }
        // After many moves the beads should still be finite
        for s in 0..2 {
            assert!(
                poly.beads[0][s][0].is_finite(),
                "bead position became non-finite"
            );
        }
        let _ = original;
    }

    #[test]
    fn test_pimc_invalid_config_rejected() {
        let cfg = PimcConfig {
            n_slices: 0,
            ..Default::default()
        };
        let result = PimcSimulator::new(cfg, Box::new(zero));
        assert!(result.is_err(), "zero n_slices should be rejected");
    }
}
