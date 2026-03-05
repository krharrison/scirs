//! # Computational Physics Tools
//!
//! This module provides a comprehensive set of computational physics routines for
//! the SciRS2 scientific computing ecosystem, organised into four major domains:
//!
//! | Submodule | Domain |
//! |-----------|--------|
//! | [`classical`] | Classical mechanics — kinematics, projectile motion, harmonic oscillators |
//! | [`thermodynamics`] | Thermodynamics — ideal gas, statistical mechanics, radiation |
//! | [`electrodynamics`] | Electrodynamics — Coulomb, Lorentz, special relativity |
//! | [`quantum`] | Quantum mechanics — de Broglie, uncertainty, hydrogen atom, particle-in-box |
//!
//! ## Design Principles
//!
//! * **SI units** throughout unless a function's documentation states otherwise.
//! * **Error handling** — every computation that can fail returns a [`PhysicsResult`].
//! * **Physical constants** are taken from [`crate::constants::physical`] (NIST CODATA values).
//! * **Pure Rust** — no external physics crate dependencies.
//!
//! ## Quick Start
//!
//! ```rust
//! use scirs2_core::physics::classical::kinetic_energy;
//! use scirs2_core::physics::thermodynamics::ideal_gas_pressure;
//! use scirs2_core::physics::electrodynamics::{coulomb_force, lorentz_factor};
//! use scirs2_core::physics::quantum::hydrogen_energy_level;
//!
//! // Classical mechanics
//! let ke = kinetic_energy(1.0, 10.0).expect("should succeed");   // ½ · 1 kg · (10 m/s)² = 50 J
//! assert!((ke - 50.0).abs() < 1e-12);
//!
//! // Ideal gas law
//! let p = ideal_gas_pressure(1.0, 273.15, 0.022_413_969_54).expect("should succeed"); // ≈ 101 325 Pa
//! assert!((p - 101_325.0).abs() < 1.0);
//!
//! // Coulomb force between two protons 1 Å apart
//! use scirs2_core::constants::physical::ELEMENTARY_CHARGE;
//! let f = coulomb_force(ELEMENTARY_CHARGE, ELEMENTARY_CHARGE, 1e-10).expect("should succeed");
//! assert!(f > 0.0);
//!
//! // Hydrogen ground-state energy ≈ −13.6 eV
//! let e1 = hydrogen_energy_level(1).expect("should succeed");
//! assert!(e1 < 0.0);
//! ```

pub mod classical;
pub mod electrodynamics;
pub mod error;
pub mod quantum;
pub mod thermodynamics;

// Convenient top-level re-exports
pub use error::{PhysicsError, PhysicsResult};

// Re-export the most commonly used functions at the module root for ergonomic access.

// Classical mechanics
pub use classical::{
    angular_momentum, centripetal_acceleration, escape_velocity, gravitational_force,
    kinetic_energy, momentum, orbital_period, potential_energy_gravity, projectile_motion,
    sho_displacement, simple_harmonic_oscillator, OscillatorResult, ProjectileResult,
};

// Thermodynamics
pub use thermodynamics::{
    black_body_spectral_radiance, boltzmann_distribution, bose_einstein_distribution,
    carnot_efficiency, fermi_dirac_distribution, heat_capacity_debye, ideal_gas_pressure,
    maxwell_mean_speed, maxwell_most_probable_speed, maxwell_speed_distribution,
    stefan_boltzmann_radiance, thermal_de_broglie_wavelength, wien_wavelength, zero_point_energy,
};

// Electrodynamics
pub use electrodynamics::{
    biot_savart_wire, capacitor_energy, compton_wavelength_shift, coulomb_force, cyclotron_radius,
    electric_field, electric_potential, electrostatic_potential_energy, impedance_of_free_space,
    lorentz_factor, magnetic_force, ohm_resistance, relativistic_energy,
    relativistic_kinetic_energy, relativistic_momentum, relativistic_velocity_addition,
    resistor_power, rest_energy,
};

// Quantum mechanics
pub use quantum::{
    bohr_magneton, de_broglie_wavelength, de_broglie_wavelength_from_energy,
    energy_time_uncertainty, fine_structure_constant, heisenberg_uncertainty,
    hydrogen_energy_level, hydrogen_orbit_radius, hydrogen_transition_energy,
    particle_in_box_energy, particle_in_box_wavefunction, photoelectric_kinetic_energy,
    photon_energy, photon_frequency, photon_wavelength, qho_energy, spin_angular_momentum,
    thermal_wavelength, tunnel_transmission_wkb,
};
