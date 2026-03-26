//! Configuration types for Path Integral Monte Carlo (PIMC)
//!
//! Defines `PimcConfig` (simulation parameters) and `PimcResult` (output statistics).

use serde::{Deserialize, Serialize};

/// Configuration for a Path Integral Monte Carlo simulation.
///
/// PIMC represents a quantum particle (or collection of particles) at finite
/// temperature `1/beta` as a ring polymer with `n_slices` beads in imaginary time.
/// Each bead lives in `dimension`-dimensional space.
///
/// # Defaults
///
/// ```
/// use scirs2_integrate::pimc::config::PimcConfig;
/// let cfg = PimcConfig::default();
/// assert_eq!(cfg.n_slices, 32);
/// assert_eq!(cfg.beta, 1.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PimcConfig {
    /// Number of distinguishable particles. Default: 1.
    pub n_particles: usize,
    /// Number of imaginary-time slices `M`. Default: 32.
    pub n_slices: usize,
    /// Inverse temperature `β = 1/(k_B T)`. Default: 1.0.
    pub beta: f64,
    /// Particle mass `m`. Default: 1.0.
    pub mass: f64,
    /// Spatial dimensionality `d`. Default: 1.
    pub dimension: usize,
    /// Number of MC steps after thermalization. Default: 10 000.
    pub n_steps: usize,
    /// Number of thermalization (burn-in) steps. Default: 1 000.
    pub n_thermalize: usize,
    /// Maximum single-bead displacement per dimension. Default: 0.5.
    pub max_displacement: f64,
    /// Collect energy estimators every this many MC sweeps. Default: 10.
    pub estimator_interval: usize,
    /// RNG seed for reproducibility. Default: 42.
    pub seed: u64,
}

impl Default for PimcConfig {
    fn default() -> Self {
        Self {
            n_particles: 1,
            n_slices: 32,
            beta: 1.0,
            mass: 1.0,
            dimension: 1,
            n_steps: 10_000,
            n_thermalize: 1_000,
            max_displacement: 0.5,
            estimator_interval: 10,
            seed: 42,
        }
    }
}

/// Results returned by a completed PIMC simulation.
#[derive(Debug, Clone, Default)]
pub struct PimcResult {
    /// Mean total energy `⟨E⟩` over the production run.
    pub energy_mean: f64,
    /// Standard error (not std dev) of the total energy estimate.
    pub energy_std: f64,
    /// Mean kinetic energy `⟨K⟩`.
    pub kinetic_mean: f64,
    /// Mean potential energy `⟨V⟩`.
    pub potential_mean: f64,
    /// Number of accepted MC moves during production.
    pub n_accepted: u64,
    /// Total number of proposed MC moves during production.
    pub n_total: u64,
    /// Fraction of accepted moves: `n_accepted / n_total`.
    pub acceptance_rate: f64,
}
