//! # Path Integral Monte Carlo (PIMC) for Quantum Statistical Mechanics
//!
//! This module implements the Path Integral Monte Carlo method, which evaluates
//! quantum-statistical partition functions and observables by representing each
//! particle as a **ring polymer** of `M` imaginary-time beads.
//!
//! ## Background
//!
//! The quantum partition function at inverse temperature `β = 1/(k_B T)` can be
//! written as a path integral (Feynman 1953):
//!
//! ```text
//! Z = ∫ D[r(τ)] exp(−S[r] / ħ)
//! ```
//!
//! where the imaginary-time action is
//!
//! ```text
//! S = ∫₀^β dτ [ (m/2)|ṙ|² + V(r) ]
//! ```
//!
//! Discretising into `M` slices of width `τ = β/M` gives the **primitive
//! approximation** used here.
//!
//! ## Modules
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`config`] | [`PimcConfig`], [`PimcResult`] |
//! | [`paths`]  | [`RingPolymer`] ring-polymer paths + Lévy bridge |
//! | [`moves`]  | [`SingleBeadMove`], [`CenterOfMassMove`], [`BisectionMove`], [`PimcMove`] |
//! | [`estimators`] | [`EnergyEstimator`] (thermodynamic estimator) |
//! | [`simulator`]  | [`PimcSimulator`] main driver |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use scirs2_integrate::pimc::{
//!     config::PimcConfig,
//!     simulator::PimcSimulator,
//! };
//!
//! // 1-D quantum harmonic oscillator (ω = 1, ħ = 1, m = 1)
//! // Ground-state energy = ħω/2 = 0.5
//! let cfg = PimcConfig {
//!     n_slices: 64,
//!     beta: 10.0,       // low temperature → ground state
//!     n_steps: 10_000,
//!     n_thermalize: 1_000,
//!     max_displacement: 0.3,
//!     seed: 42,
//!     ..Default::default()
//! };
//!
//! let mut sim = PimcSimulator::new(
//!     cfg,
//!     Box::new(|r: &[f64]| 0.5 * r[0] * r[0]),
//! ).unwrap();
//!
//! let result = sim.run().unwrap();
//! println!("Ground-state energy ≈ {:.4}", result.energy_mean);
//! // Expect ~ 0.5
//! ```

pub mod config;
pub mod estimators;
pub mod moves;
pub mod paths;
pub mod simulator;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use config::{PimcConfig, PimcResult};
pub use estimators::EnergyEstimator;
pub use moves::{BisectionMove, CenterOfMassMove, PimcMove, RngProxy, SingleBeadMove};
pub use paths::RingPolymer;
pub use simulator::PimcSimulator;
