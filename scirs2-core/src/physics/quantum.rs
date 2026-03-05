//! Quantum mechanics: wave-particle duality, uncertainty principle, atomic models,
//! potential wells, tunnelling probability, and spin statistics.
//!
//! All functions operate in SI units unless otherwise stated.
//!
//! # Reference
//!
//! * Griffiths — *Introduction to Quantum Mechanics* (3rd ed.)
//! * Shankar — *Principles of Quantum Mechanics* (2nd ed.)
//! * Cohen-Tannoudji, Diu & Laloë — *Quantum Mechanics* (Vols. 1 & 2)

use std::f64::consts::PI;

use crate::constants::physical::{
    BOHR_RADIUS, BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE, FINE_STRUCTURE, PLANCK,
    REDUCED_PLANCK, RYDBERG, SPEED_OF_LIGHT,
};

use super::error::{PhysicsError, PhysicsResult};

// ─── Wave-particle duality ───────────────────────────────────────────────────

/// de Broglie wavelength of a particle.
///
/// `λ = h / (mv)`
///
/// # Arguments
///
/// * `mass`     – particle rest mass in kg (must be > 0)
/// * `velocity` – particle speed in m/s (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `velocity` is not positive.
pub fn de_broglie_wavelength(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if velocity <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "velocity",
            reason: format!("velocity must be positive, got {velocity}"),
        });
    }
    Ok(PLANCK / (mass * velocity))
}

/// de Broglie wavelength given kinetic energy (non-relativistic).
///
/// `λ = h / √(2mK)`
///
/// # Arguments
///
/// * `mass`           – particle rest mass in kg (must be > 0)
/// * `kinetic_energy` – kinetic energy in J (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `kinetic_energy` is not positive.
pub fn de_broglie_wavelength_from_energy(mass: f64, kinetic_energy: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if kinetic_energy <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "kinetic_energy",
            reason: format!("kinetic energy must be positive, got {kinetic_energy}"),
        });
    }
    Ok(PLANCK / (2.0 * mass * kinetic_energy).sqrt())
}

// ─── Heisenberg uncertainty principle ────────────────────────────────────────

/// Minimum momentum uncertainty from position uncertainty.
///
/// The position-momentum uncertainty relation `ΔxΔp ≥ ℏ/2` gives the minimum
/// momentum spread:
///
/// `Δp_min = ℏ / (2·Δx)`
///
/// # Arguments
///
/// * `delta_x` – position uncertainty in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `delta_x` is not positive.
pub fn heisenberg_uncertainty(delta_x: f64) -> PhysicsResult<f64> {
    if delta_x <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "delta_x",
            reason: format!("position uncertainty must be positive, got {delta_x}"),
        });
    }
    Ok(REDUCED_PLANCK / (2.0 * delta_x))
}

/// Minimum energy uncertainty from lifetime uncertainty.
///
/// `ΔE_min = ℏ / (2·Δt)`
///
/// # Arguments
///
/// * `delta_t` – lifetime uncertainty in s (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `delta_t` is not positive.
pub fn energy_time_uncertainty(delta_t: f64) -> PhysicsResult<f64> {
    if delta_t <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "delta_t",
            reason: format!("time uncertainty must be positive, got {delta_t}"),
        });
    }
    Ok(REDUCED_PLANCK / (2.0 * delta_t))
}

// ─── Hydrogen atom (Bohr model) ───────────────────────────────────────────────

/// Energy of the nth energy level of a hydrogen atom (Bohr model).
///
/// `E_n = −13.6 eV / n²  =  −(m_e·e⁴) / (2·ħ²·n²)  =  −Ry/n²`
///
/// Returns energy in Joules (negative, as it is a bound state).
///
/// # Arguments
///
/// * `n` – principal quantum number (must be ≥ 1)
///
/// # Errors
///
/// Returns [`PhysicsError::QuantumNumberOutOfRange`] if `n` is 0.
pub fn hydrogen_energy_level(n: usize) -> PhysicsResult<f64> {
    if n == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "principal quantum number n must be ≥ 1".to_string(),
        ));
    }
    // E_n = -R_∞ · h · c / n²  (in Joules)
    // R_∞ in m⁻¹; energy = hcR/n²
    let rydberg_energy = PLANCK * SPEED_OF_LIGHT * RYDBERG;
    Ok(-rydberg_energy / (n * n) as f64)
}

/// Bohr radius of the nth orbit of a hydrogen atom.
///
/// `r_n = n²·a₀`  where `a₀` is the Bohr radius (≈ 0.529 Å).
///
/// # Arguments
///
/// * `n` – principal quantum number (must be ≥ 1)
///
/// # Errors
///
/// Returns [`PhysicsError::QuantumNumberOutOfRange`] if `n` is 0.
pub fn hydrogen_orbit_radius(n: usize) -> PhysicsResult<f64> {
    if n == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "principal quantum number n must be ≥ 1".to_string(),
        ));
    }
    Ok((n * n) as f64 * BOHR_RADIUS)
}

/// Photon energy emitted/absorbed during a transition between hydrogen levels.
///
/// `ΔE = E_initial − E_final = Ry·(1/n_f² − 1/n_i²)`
///
/// A positive value means emission (initial > final level); negative means absorption.
///
/// # Arguments
///
/// * `n_initial` – initial principal quantum number (must be ≥ 1)
/// * `n_final`   – final principal quantum number (must be ≥ 1)
///
/// # Errors
///
/// Returns [`PhysicsError::QuantumNumberOutOfRange`] if either quantum number is 0.
/// Returns [`PhysicsError::DomainError`] if `n_initial == n_final` (no transition).
pub fn hydrogen_transition_energy(n_initial: usize, n_final: usize) -> PhysicsResult<f64> {
    if n_initial == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "n_initial must be ≥ 1".to_string(),
        ));
    }
    if n_final == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "n_final must be ≥ 1".to_string(),
        ));
    }
    if n_initial == n_final {
        return Err(PhysicsError::DomainError(
            "n_initial and n_final must differ for a spectral transition".to_string(),
        ));
    }
    let e_i = hydrogen_energy_level(n_initial)?;
    let e_f = hydrogen_energy_level(n_final)?;
    Ok(e_i - e_f) // positive for emission (n_i > n_f)
}

// ─── Particle-in-a-box ───────────────────────────────────────────────────────

/// Energy eigenvalue for a particle in a 1-D infinite square well (particle in a box).
///
/// `E_n = n²·π²·ħ² / (2·m·L²)`
///
/// # Arguments
///
/// * `n`      – quantum number (must be ≥ 1)
/// * `length` – box length in m (must be > 0)
/// * `mass`   – particle mass in kg (must be > 0)
///
/// # Errors
///
/// * [`PhysicsError::QuantumNumberOutOfRange`] if `n` is 0.
/// * [`PhysicsError::InvalidParameter`] if `length` or `mass` is not positive.
pub fn particle_in_box_energy(n: usize, length: f64, mass: f64) -> PhysicsResult<f64> {
    if n == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "quantum number n must be ≥ 1".to_string(),
        ));
    }
    if length <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "length",
            reason: format!("box length must be positive, got {length}"),
        });
    }
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    let n_sq = (n * n) as f64;
    Ok(n_sq * PI * PI * REDUCED_PLANCK * REDUCED_PLANCK / (2.0 * mass * length * length))
}

/// Wavefunction ψₙ(x) for a particle in a 1-D infinite square well.
///
/// `ψₙ(x) = √(2/L) · sin(nπx/L)`  for `x ∈ [0, L]`, zero otherwise.
///
/// # Arguments
///
/// * `n`      – quantum number (must be ≥ 1)
/// * `length` – box length in m (must be > 0)
/// * `x`      – position in m
/// * `mass`   – particle mass in kg (must be > 0 — unused but retained for API consistency)
///
/// # Errors
///
/// * [`PhysicsError::QuantumNumberOutOfRange`] if `n` is 0.
/// * [`PhysicsError::InvalidParameter`] if `length` or `mass` is not positive.
pub fn particle_in_box_wavefunction(
    n: usize,
    length: f64,
    x: f64,
    mass: f64,
) -> PhysicsResult<f64> {
    if n == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "quantum number n must be ≥ 1".to_string(),
        ));
    }
    if length <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "length",
            reason: format!("box length must be positive, got {length}"),
        });
    }
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if x < 0.0 || x > length {
        return Ok(0.0); // outside the well
    }
    Ok((2.0 / length).sqrt() * (n as f64 * PI * x / length).sin())
}

// ─── Tunnelling ───────────────────────────────────────────────────────────────

/// WKB approximation for transmission probability through a rectangular barrier.
///
/// For a particle with energy `E` incident on a rectangular potential barrier of
/// height `V₀ > E` and width `a`, the WKB transmission probability is:
///
/// `T ≈ exp(−2κa)`  where  `κ = √(2m(V₀−E)) / ħ`
///
/// # Arguments
///
/// * `mass`          – particle mass in kg (must be > 0)
/// * `energy`        – particle energy in J (must be in `(0, v0)`)
/// * `barrier_height`– barrier potential V₀ in J (must be > `energy`)
/// * `barrier_width` – barrier width `a` in m (must be > 0)
///
/// # Errors
///
/// * [`PhysicsError::InvalidParameter`] if `mass`, `barrier_width`, or `energy` is not positive.
/// * [`PhysicsError::DomainError`] if `energy` ≥ `barrier_height` (classical transmission).
pub fn tunnel_transmission_wkb(
    mass: f64,
    energy: f64,
    barrier_height: f64,
    barrier_width: f64,
) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if energy <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "energy",
            reason: format!("particle energy must be positive, got {energy}"),
        });
    }
    if barrier_width <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "barrier_width",
            reason: format!("barrier width must be positive, got {barrier_width}"),
        });
    }
    if energy >= barrier_height {
        return Err(PhysicsError::DomainError(format!(
            "particle energy ({energy:.4e} J) must be less than barrier height ({barrier_height:.4e} J) \
             for sub-barrier tunnelling"
        )));
    }
    let kappa = (2.0 * mass * (barrier_height - energy)).sqrt() / REDUCED_PLANCK;
    Ok((-2.0 * kappa * barrier_width).exp())
}

// ─── Photon properties ────────────────────────────────────────────────────────

/// Energy of a photon given its wavelength.
///
/// `E = hc/λ`
///
/// # Arguments
///
/// * `wavelength` – photon wavelength in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `wavelength` is not positive.
pub fn photon_energy(wavelength: f64) -> PhysicsResult<f64> {
    if wavelength <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "wavelength",
            reason: format!("wavelength must be positive, got {wavelength}"),
        });
    }
    Ok(PLANCK * SPEED_OF_LIGHT / wavelength)
}

/// Wavelength corresponding to a given photon energy.
///
/// `λ = hc/E`
///
/// # Arguments
///
/// * `energy` – photon energy in J (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `energy` is not positive.
pub fn photon_wavelength(energy: f64) -> PhysicsResult<f64> {
    if energy <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "energy",
            reason: format!("photon energy must be positive, got {energy}"),
        });
    }
    Ok(PLANCK * SPEED_OF_LIGHT / energy)
}

/// Frequency of a photon given its energy.
///
/// `ν = E/h`
///
/// # Arguments
///
/// * `energy` – photon energy in J (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `energy` is not positive.
pub fn photon_frequency(energy: f64) -> PhysicsResult<f64> {
    if energy <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "energy",
            reason: format!("photon energy must be positive, got {energy}"),
        });
    }
    Ok(energy / PLANCK)
}

// ─── Quantum harmonic oscillator ──────────────────────────────────────────────

/// Energy eigenvalue of a quantum harmonic oscillator.
///
/// `E_n = ħω(n + ½)`
///
/// # Arguments
///
/// * `n`              – vibrational quantum number (≥ 0)
/// * `angular_freq`   – angular frequency ω in rad/s (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `angular_freq` is not positive.
pub fn qho_energy(n: usize, angular_freq: f64) -> PhysicsResult<f64> {
    if angular_freq <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "angular_freq",
            reason: format!("angular frequency must be positive, got {angular_freq}"),
        });
    }
    Ok(REDUCED_PLANCK * angular_freq * (n as f64 + 0.5))
}

/// Photoelectric effect: kinetic energy of emitted electron.
///
/// `K = hν − φ`
///
/// Returns `None` (wrapped as `Err`) if the photon energy is insufficient to overcome
/// the work function.
///
/// # Arguments
///
/// * `frequency`     – photon frequency ν in Hz (must be > 0)
/// * `work_function` – work function φ in J (must be > 0)
///
/// # Errors
///
/// * [`PhysicsError::InvalidParameter`] if `frequency` or `work_function` is not positive.
/// * [`PhysicsError::DomainError`] if photon energy < work function.
pub fn photoelectric_kinetic_energy(frequency: f64, work_function: f64) -> PhysicsResult<f64> {
    if frequency <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "frequency",
            reason: format!("frequency must be positive, got {frequency}"),
        });
    }
    if work_function <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "work_function",
            reason: format!("work function must be positive, got {work_function}"),
        });
    }
    let photon_e = PLANCK * frequency;
    if photon_e < work_function {
        return Err(PhysicsError::DomainError(format!(
            "photon energy ({photon_e:.4e} J) < work function ({work_function:.4e} J): \
             no photoelectric emission"
        )));
    }
    Ok(photon_e - work_function)
}

/// Spin angular momentum magnitude `S = ħ√(s(s+1))`.
///
/// For a spin-1/2 particle `s = 1/2` and `S = ħ√3/2`.
///
/// # Arguments
///
/// * `spin` – spin quantum number as a half-integer multiple of ½ (e.g. 1 for spin-1/2, 2 for spin-1)
///   The actual spin quantum number is `s = spin / 2.0`.
///
/// # Errors
///
/// Returns [`PhysicsError::QuantumNumberOutOfRange`] if `spin` is 0.
pub fn spin_angular_momentum(spin_twice: usize) -> PhysicsResult<f64> {
    if spin_twice == 0 {
        return Err(PhysicsError::QuantumNumberOutOfRange(
            "spin_twice must be ≥ 1 (representing spin ≥ 1/2)".to_string(),
        ));
    }
    let s = spin_twice as f64 / 2.0;
    Ok(REDUCED_PLANCK * (s * (s + 1.0)).sqrt())
}

/// Bohr magneton — magnetic moment of a ground-state hydrogen atom.
///
/// `μ_B = eħ / (2m_e)`
///
/// Returns the value in J/T (matches `physical::BOHR_MAGNETON` ≈ 9.274×10⁻²⁴ J/T).
#[must_use]
pub fn bohr_magneton() -> f64 {
    ELEMENTARY_CHARGE * REDUCED_PLANCK / (2.0 * ELECTRON_MASS)
}

/// Fine structure constant (dimensionless).
///
/// `α = e² / (4πε₀ħc) ≈ 1/137`
///
/// This is a re-export of the NIST CODATA value.
#[must_use]
pub fn fine_structure_constant() -> f64 {
    FINE_STRUCTURE
}

/// Thermal wavelength of a particle at temperature T.
///
/// `λ_th = h / √(2πmkT)`
///
/// Used to determine the quantum/classical crossover:
/// the gas is quantum when `n·λ_th³ ≳ 1`.
///
/// # Arguments
///
/// * `mass`        – particle mass in kg (must be > 0)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn thermal_wavelength(mass: f64, temperature: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    Ok(PLANCK / (2.0 * PI * mass * BOLTZMANN * temperature).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-30; // appropriate for atomic-scale energies

    // ─── Wave-particle duality ────────────────────────────────────────────────

    #[test]
    fn test_de_broglie_electron_thermal() {
        // Thermal electron at 300 K: λ ≈ 4.3 nm
        let ke = 1.5 * BOLTZMANN * 300.0; // average KE of ideal gas particle
        let lambda = de_broglie_wavelength_from_energy(ELECTRON_MASS, ke).expect("should succeed");
        // Should be in nm range
        assert!(lambda > 1e-9 && lambda < 20e-9, "λ = {lambda:.4e} m");
    }

    #[test]
    fn test_de_broglie_invalid() {
        assert!(de_broglie_wavelength(0.0, 1.0).is_err());
        assert!(de_broglie_wavelength(1.0, 0.0).is_err());
        assert!(de_broglie_wavelength_from_energy(0.0, 1.0).is_err());
        assert!(de_broglie_wavelength_from_energy(1.0, 0.0).is_err());
    }

    // ─── Uncertainty principle ────────────────────────────────────────────────

    #[test]
    fn test_heisenberg_uncertainty_bohr_radius() {
        // Δp_min for Δx = a₀ (Bohr radius)
        let dp = heisenberg_uncertainty(BOHR_RADIUS).expect("should succeed");
        let expected = REDUCED_PLANCK / (2.0 * BOHR_RADIUS);
        assert!((dp - expected).abs() < 1e-35);
    }

    #[test]
    fn test_heisenberg_uncertainty_invalid() {
        assert!(heisenberg_uncertainty(0.0).is_err());
        assert!(heisenberg_uncertainty(-1.0).is_err());
    }

    #[test]
    fn test_energy_time_uncertainty() {
        let de = energy_time_uncertainty(1e-9).expect("should succeed");
        let expected = REDUCED_PLANCK / (2.0 * 1e-9);
        assert!((de - expected).abs() < 1e-40);
    }

    // ─── Hydrogen atom ────────────────────────────────────────────────────────

    #[test]
    fn test_hydrogen_ground_state_energy() {
        // E_1 = −13.6 eV
        let e1 = hydrogen_energy_level(1).expect("should succeed");
        let ev = crate::constants::physical::ELEMENTARY_CHARGE; // 1 eV in J
        let expected_ev = -13.606; // eV
        let actual_ev = e1 / ev;
        assert!(
            (actual_ev - expected_ev).abs() < 0.01,
            "E_1 = {actual_ev:.3} eV"
        );
    }

    #[test]
    fn test_hydrogen_energy_levels_ordering() {
        // Energies should increase with n (less negative)
        let e1 = hydrogen_energy_level(1).expect("should succeed");
        let e2 = hydrogen_energy_level(2).expect("should succeed");
        let e3 = hydrogen_energy_level(3).expect("should succeed");
        assert!(e1 < e2 && e2 < e3, "E_1 < E_2 < E_3");
    }

    #[test]
    fn test_hydrogen_quantum_number_zero() {
        assert!(hydrogen_energy_level(0).is_err());
    }

    #[test]
    fn test_hydrogen_transition_lyman_alpha() {
        // n=2 → n=1 emission (Lyman α, ~10.2 eV)
        let de = hydrogen_transition_energy(2, 1).expect("should succeed");
        let ev = ELEMENTARY_CHARGE;
        let de_ev = de / ev;
        assert!((de_ev - 10.2).abs() < 0.1, "Lyman α = {de_ev:.3} eV");
    }

    #[test]
    fn test_hydrogen_orbit_radius() {
        // r_1 = a₀, r_2 = 4·a₀, r_3 = 9·a₀
        let r1 = hydrogen_orbit_radius(1).expect("should succeed");
        let r2 = hydrogen_orbit_radius(2).expect("should succeed");
        let r3 = hydrogen_orbit_radius(3).expect("should succeed");
        assert!((r1 - BOHR_RADIUS).abs() < 1e-20);
        assert!((r2 - 4.0 * BOHR_RADIUS).abs() < 1e-20);
        assert!((r3 - 9.0 * BOHR_RADIUS).abs() < 1e-20);
    }

    // ─── Particle in box ──────────────────────────────────────────────────────

    #[test]
    fn test_particle_in_box_energy_ratio() {
        // E_n ∝ n², so E_2/E_1 = 4, E_3/E_1 = 9
        let e1 = particle_in_box_energy(1, 1e-9, ELECTRON_MASS).expect("should succeed");
        let e2 = particle_in_box_energy(2, 1e-9, ELECTRON_MASS).expect("should succeed");
        let e3 = particle_in_box_energy(3, 1e-9, ELECTRON_MASS).expect("should succeed");
        assert!((e2 / e1 - 4.0).abs() < 1e-12, "E_2/E_1 = {}", e2 / e1);
        assert!((e3 / e1 - 9.0).abs() < 1e-12, "E_3/E_1 = {}", e3 / e1);
    }

    #[test]
    fn test_particle_in_box_wavefunction_boundary() {
        // ψ must vanish at x=0 and x=L (up to floating-point rounding of sin(nπ))
        let l = 1e-9_f64;
        let psi_0 = particle_in_box_wavefunction(1, l, 0.0, ELECTRON_MASS).expect("should succeed");
        let psi_l = particle_in_box_wavefunction(1, l, l, ELECTRON_MASS).expect("should succeed");
        // sin(0) = 0 exactly; sin(π) ≈ 1.2e-16 (floating-point), normalisation ≈ √(2/L)
        assert!(psi_0.abs() < 1e-20, "ψ(0) = {psi_0}");
        // At x = L: sin(n·π) is never exactly 0 in f64 — accept values < machine epsilon × norm
        let norm = (2.0 / l).sqrt(); // ≈ 4.47e4
        assert!(psi_l.abs() < norm * 2e-15, "ψ(L) = {psi_l:.4e}");
    }

    #[test]
    fn test_particle_in_box_wavefunction_outside() {
        // ψ = 0 outside [0, L]
        let psi =
            particle_in_box_wavefunction(1, 1e-9, 2e-9, ELECTRON_MASS).expect("should succeed");
        assert_eq!(psi, 0.0);
    }

    #[test]
    fn test_particle_in_box_invalid() {
        assert!(particle_in_box_energy(0, 1e-9, ELECTRON_MASS).is_err());
        assert!(particle_in_box_energy(1, 0.0, ELECTRON_MASS).is_err());
        assert!(particle_in_box_energy(1, 1e-9, 0.0).is_err());
    }

    // ─── Tunnelling ───────────────────────────────────────────────────────────

    #[test]
    fn test_tunnel_transmission_small_barrier() {
        // Wide barrier: nearly zero transmission
        let t = tunnel_transmission_wkb(ELECTRON_MASS, 1e-20, 2e-20, 1e-9).expect("should succeed");
        assert!(t > 0.0 && t < 1.0, "T = {t:.4e}");
    }

    #[test]
    fn test_tunnel_transmission_thin_barrier() {
        // Very thin barrier: higher transmission
        let t_wide =
            tunnel_transmission_wkb(ELECTRON_MASS, 1e-19, 2e-19, 1e-10).expect("should succeed");
        let t_thin =
            tunnel_transmission_wkb(ELECTRON_MASS, 1e-19, 2e-19, 1e-11).expect("should succeed");
        assert!(t_thin > t_wide, "Thinner barrier should transmit more");
    }

    #[test]
    fn test_tunnel_transmission_above_barrier_fails() {
        // Energy above barrier: classical domain, WKB sub-barrier formula invalid
        assert!(tunnel_transmission_wkb(ELECTRON_MASS, 3e-19, 2e-19, 1e-10).is_err());
    }

    // ─── Photon ──────────────────────────────────────────────────────────────

    #[test]
    fn test_photon_energy_visible_light() {
        // Green light ≈ 532 nm => E ≈ 3.73e-19 J
        let e = photon_energy(532e-9).expect("should succeed");
        assert!((e - 3.73e-19).abs() < 0.05e-19, "E = {e:.4e} J");
    }

    #[test]
    fn test_photon_energy_wavelength_roundtrip() {
        let lambda_in = 500e-9_f64;
        let e = photon_energy(lambda_in).expect("should succeed");
        let lambda_out = photon_wavelength(e).expect("should succeed");
        assert!((lambda_in - lambda_out).abs() < 1e-25);
    }

    // ─── Quantum harmonic oscillator ──────────────────────────────────────────

    #[test]
    fn test_qho_ground_state_zero_point() {
        // E_0 = ħω/2
        let omega = 1e14_f64;
        let e0 = qho_energy(0, omega).expect("should succeed");
        let expected = 0.5 * REDUCED_PLANCK * omega;
        assert!((e0 - expected).abs() < 1e-40);
    }

    #[test]
    fn test_qho_energy_spacing() {
        // ΔE = ħω between adjacent levels
        let omega = 1e12_f64;
        let e0 = qho_energy(0, omega).expect("should succeed");
        let e1 = qho_energy(1, omega).expect("should succeed");
        let spacing = e1 - e0;
        let expected = REDUCED_PLANCK * omega;
        // Relative tolerance: machine-precision-limited for f64 subtraction
        let rel_err = (spacing - expected).abs() / expected;
        assert!(rel_err < 1e-12, "ΔE rel error = {rel_err:.4e}");
    }

    // ─── Misc quantum ─────────────────────────────────────────────────────────

    #[test]
    fn test_bohr_magneton_matches_constant() {
        let mu_b = bohr_magneton();
        let expected = crate::constants::physical::BOHR_MAGNETON;
        // Computed value vs NIST constant
        assert!(
            (mu_b - expected).abs() / expected < 1e-6,
            "μ_B = {mu_b:.6e} J/T"
        );
    }

    #[test]
    fn test_spin_half_angular_momentum() {
        // S = ħ√(3)/2  for spin-1/2
        let s = spin_angular_momentum(1).expect("should succeed"); // spin_twice=1 => s=1/2
        let expected = REDUCED_PLANCK * (3.0_f64).sqrt() / 2.0;
        assert!((s - expected).abs() < 1e-50);
    }

    #[test]
    fn test_fine_structure_constant() {
        let alpha = fine_structure_constant();
        // α ≈ 1/137.036
        assert!((alpha - 1.0 / 137.036).abs() < 1e-5, "α = {alpha:.8}");
    }
}
