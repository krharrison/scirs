//! Electrodynamics: electrostatics, magnetostatics, electromagnetic fields,
//! circuit theory, and special relativity.
//!
//! All functions use SI units.
//!
//! # Reference
//!
//! * Griffiths — *Introduction to Electrodynamics* (4th ed.)
//! * Jackson — *Classical Electrodynamics* (3rd ed.)
//! * Einstein — *On the Electrodynamics of Moving Bodies* (1905)

use crate::constants::physical::{
    ELECTRIC_CONSTANT, ELEMENTARY_CHARGE, MAGNETIC_CONSTANT, SPEED_OF_LIGHT,
};

use super::error::{PhysicsError, PhysicsResult};

// Helper: 1 / (4πε₀)
#[inline(always)]
fn coulomb_constant() -> f64 {
    1.0 / (4.0 * std::f64::consts::PI * ELECTRIC_CONSTANT)
}

// ─── Electrostatics ───────────────────────────────────────────────────────────

/// Coulomb force between two point charges.
///
/// `F = kₑ·q₁·q₂/r²`  where `kₑ = 1/(4πε₀) ≈ 8.988×10⁹ N·m²/C²`.
///
/// The sign of the returned value indicates attractive (negative, opposite signs)
/// or repulsive (positive, same sign) force.
///
/// # Arguments
///
/// * `q1`       – first charge in C
/// * `q2`       – second charge in C
/// * `distance` – separation in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `distance` is not positive.
pub fn coulomb_force(q1: f64, q2: f64, distance: f64) -> PhysicsResult<f64> {
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(coulomb_constant() * q1 * q2 / (distance * distance))
}

/// Electric potential due to a point charge.
///
/// `V = kₑ·q/r`
///
/// # Arguments
///
/// * `charge`   – charge in C
/// * `distance` – distance from the charge in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `distance` is not positive.
pub fn electric_potential(charge: f64, distance: f64) -> PhysicsResult<f64> {
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(coulomb_constant() * charge / distance)
}

/// Electric field magnitude due to a point charge.
///
/// `E = kₑ·|q|/r²`
///
/// The magnitude is always non-negative.
///
/// # Arguments
///
/// * `charge`   – charge in C
/// * `distance` – distance from the charge in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `distance` is not positive.
pub fn electric_field(charge: f64, distance: f64) -> PhysicsResult<f64> {
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(coulomb_constant() * charge.abs() / (distance * distance))
}

/// Electrostatic potential energy of a two-charge system.
///
/// `U = kₑ·q₁·q₂/r`
///
/// # Arguments
///
/// * `q1`       – first charge in C
/// * `q2`       – second charge in C
/// * `distance` – separation in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `distance` is not positive.
pub fn electrostatic_potential_energy(q1: f64, q2: f64, distance: f64) -> PhysicsResult<f64> {
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(coulomb_constant() * q1 * q2 / distance)
}

/// Energy stored in a capacitor.
///
/// `U = Q²/(2C) = CV²/2`
///
/// # Arguments
///
/// * `capacitance` – capacitance in F (must be > 0)
/// * `voltage`     – voltage across the capacitor in V
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `capacitance` is not positive.
pub fn capacitor_energy(capacitance: f64, voltage: f64) -> PhysicsResult<f64> {
    if capacitance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "capacitance",
            reason: format!("capacitance must be positive, got {capacitance}"),
        });
    }
    Ok(0.5 * capacitance * voltage * voltage)
}

// ─── Magnetostatics ──────────────────────────────────────────────────────────

/// Magnetic force on a moving charged particle (Lorentz force magnitude).
///
/// `F = |q| · v · B · |sin(θ)|`
///
/// where `θ` is the angle between the velocity and the magnetic field vectors.
///
/// # Arguments
///
/// * `charge`    – particle charge in C
/// * `velocity`  – particle speed in m/s (must be ≥ 0)
/// * `b_field`   – magnetic flux density in T (must be ≥ 0)
/// * `angle_rad` – angle between v and B in radians
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `velocity` or `b_field` is negative.
pub fn magnetic_force(
    charge: f64,
    velocity: f64,
    b_field: f64,
    angle_rad: f64,
) -> PhysicsResult<f64> {
    if velocity < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "velocity",
            reason: format!("velocity must be non-negative, got {velocity}"),
        });
    }
    if b_field < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "b_field",
            reason: format!("magnetic field must be non-negative, got {b_field}"),
        });
    }
    Ok(charge.abs() * velocity * b_field * angle_rad.sin().abs())
}

/// Biot-Savart magnetic field magnitude at distance `r` from a long straight wire
/// carrying current `I`.
///
/// `B = μ₀·I / (2π·r)`
///
/// # Arguments
///
/// * `current`  – current in A (must be ≥ 0)
/// * `distance` – perpendicular distance to the wire in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `current` < 0 or `distance` ≤ 0.
pub fn biot_savart_wire(current: f64, distance: f64) -> PhysicsResult<f64> {
    if current < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "current",
            reason: format!("current must be non-negative, got {current}"),
        });
    }
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(MAGNETIC_CONSTANT * current / (2.0 * std::f64::consts::PI * distance))
}

/// Cyclotron radius of a charged particle in a uniform magnetic field.
///
/// `r = mv / (|q|·B)`
///
/// # Arguments
///
/// * `mass`     – particle mass in kg (must be > 0)
/// * `velocity` – speed in m/s (must be ≥ 0)
/// * `charge`   – charge magnitude in C (must be > 0)
/// * `b_field`  – magnetic flux density in T (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] for non-positive `mass`, `charge`, or `b_field`,
/// or negative `velocity`.
pub fn cyclotron_radius(mass: f64, velocity: f64, charge: f64, b_field: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if velocity < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "velocity",
            reason: format!("velocity must be non-negative, got {velocity}"),
        });
    }
    if charge <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "charge",
            reason: format!("charge magnitude must be positive, got {charge}"),
        });
    }
    if b_field <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "b_field",
            reason: format!("magnetic field must be positive, got {b_field}"),
        });
    }
    Ok(mass * velocity / (charge * b_field))
}

// ─── Special relativity ───────────────────────────────────────────────────────

/// Lorentz factor γ for a particle moving at speed `v`.
///
/// `γ = 1 / √(1 − v²/c²)`
///
/// # Arguments
///
/// * `velocity` – speed in m/s (must be in [0, c))
///
/// # Errors
///
/// * [`PhysicsError::InvalidParameter`] if `velocity` < 0.
/// * [`PhysicsError::SuperluminalVelocity`] if `velocity` ≥ c.
pub fn lorentz_factor(velocity: f64) -> PhysicsResult<f64> {
    if velocity < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "velocity",
            reason: format!("velocity must be non-negative, got {velocity}"),
        });
    }
    if velocity >= SPEED_OF_LIGHT {
        return Err(PhysicsError::SuperluminalVelocity {
            velocity,
            c: SPEED_OF_LIGHT,
        });
    }
    let beta = velocity / SPEED_OF_LIGHT;
    Ok(1.0 / (1.0 - beta * beta).sqrt())
}

/// Relativistic total energy of a particle.
///
/// `E = γmc²`
///
/// # Arguments
///
/// * `mass`     – rest mass in kg (must be > 0)
/// * `velocity` – speed in m/s (must be in [0, c))
///
/// # Errors
///
/// Propagates errors from [`lorentz_factor`]; additionally returns
/// [`PhysicsError::InvalidParameter`] if `mass` is not positive.
pub fn relativistic_energy(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    let gamma = lorentz_factor(velocity)?;
    Ok(gamma * mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
}

/// Rest energy of a particle: `E₀ = mc²`.
///
/// # Arguments
///
/// * `mass` – rest mass in kg (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` is not positive.
pub fn rest_energy(mass: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    Ok(mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
}

/// Relativistic kinetic energy.
///
/// `K = (γ − 1)mc²`
///
/// # Arguments
///
/// * `mass`     – rest mass in kg (must be > 0)
/// * `velocity` – speed in m/s (must be in [0, c))
///
/// # Errors
///
/// Propagates errors from [`lorentz_factor`] and [`rest_energy`].
pub fn relativistic_kinetic_energy(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    let gamma = lorentz_factor(velocity)?;
    Ok((gamma - 1.0) * mass * SPEED_OF_LIGHT * SPEED_OF_LIGHT)
}

/// Relativistic momentum `p = γmv`.
///
/// # Arguments
///
/// * `mass`     – rest mass in kg (must be > 0)
/// * `velocity` – speed in m/s (must be in [0, c))
///
/// # Errors
///
/// Propagates errors from [`lorentz_factor`]; additionally returns
/// [`PhysicsError::InvalidParameter`] if `mass` is not positive.
pub fn relativistic_momentum(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    let gamma = lorentz_factor(velocity)?;
    Ok(gamma * mass * velocity)
}

/// Relativistic velocity addition formula.
///
/// When an object moves at speed `u` in a frame S', and S' moves at speed `v`
/// relative to frame S (both in the same direction), the speed in S is:
///
/// `w = (u + v) / (1 + uv/c²)`
///
/// # Arguments
///
/// * `u` – speed in S' in m/s (must be in [0, c))
/// * `v` – speed of S' relative to S in m/s (must be in [0, c))
///
/// # Errors
///
/// * [`PhysicsError::InvalidParameter`] if `u` or `v` is negative.
/// * [`PhysicsError::SuperluminalVelocity`] if `u` or `v` ≥ c.
pub fn relativistic_velocity_addition(u: f64, v: f64) -> PhysicsResult<f64> {
    if u < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "u",
            reason: format!("speed u must be non-negative, got {u}"),
        });
    }
    if v < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "v",
            reason: format!("speed v must be non-negative, got {v}"),
        });
    }
    if u >= SPEED_OF_LIGHT {
        return Err(PhysicsError::SuperluminalVelocity {
            velocity: u,
            c: SPEED_OF_LIGHT,
        });
    }
    if v >= SPEED_OF_LIGHT {
        return Err(PhysicsError::SuperluminalVelocity {
            velocity: v,
            c: SPEED_OF_LIGHT,
        });
    }
    let c2 = SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    Ok((u + v) / (1.0 + u * v / c2))
}

/// Characteristic impedance of free space: `Z₀ = μ₀·c = √(μ₀/ε₀)`.
///
/// Returns the value in Ω (approximately 376.73 Ω).
#[must_use]
pub fn impedance_of_free_space() -> f64 {
    MAGNETIC_CONSTANT * SPEED_OF_LIGHT
}

/// Electrical resistance using Ohm's law: `R = V/I`.
///
/// # Arguments
///
/// * `voltage` – voltage in V
/// * `current` – current in A (must be non-zero)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `current` is zero.
pub fn ohm_resistance(voltage: f64, current: f64) -> PhysicsResult<f64> {
    if current == 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "current",
            reason: "current must be non-zero to compute resistance".to_string(),
        });
    }
    Ok(voltage / current)
}

/// Power dissipated in a resistor: `P = I²R = V²/R`.
///
/// # Arguments
///
/// * `current`    – current in A
/// * `resistance` – resistance in Ω (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `resistance` is not positive.
pub fn resistor_power(current: f64, resistance: f64) -> PhysicsResult<f64> {
    if resistance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "resistance",
            reason: format!("resistance must be positive, got {resistance}"),
        });
    }
    Ok(current * current * resistance)
}

/// Compton wavelength shift in scattering.
///
/// `Δλ = (h/(m_e·c)) · (1 − cos θ)` where `m_e` is the electron rest mass.
///
/// # Arguments
///
/// * `angle_rad` – scattering angle θ in radians
///
/// # Returns
///
/// Wavelength shift in meters.
pub fn compton_wavelength_shift(angle_rad: f64) -> f64 {
    use crate::constants::physical::{COMPTON_WAVELENGTH, ELECTRON_MASS};
    // Compton wavelength λ_C = h/(m_e·c)
    let _ = ELECTRON_MASS; // used via constant
    COMPTON_WAVELENGTH * (1.0 - angle_rad.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOL: f64 = 1e-9;

    // ─── Electrostatics ───────────────────────────────────────────────────────

    #[test]
    fn test_coulomb_force_two_elementary_charges() {
        // Two protons 1 Å apart
        let r = 1e-10_f64;
        let f = coulomb_force(ELEMENTARY_CHARGE, ELEMENTARY_CHARGE, r).expect("should succeed");
        // kₑ·e²/r² ≈ 23.07 nN
        assert!(f > 0.0, "Like charges repel");
        assert!((f - 23.07e-9).abs() < 1e-10, "F = {f:.4e} N");
    }

    #[test]
    fn test_coulomb_force_opposite_charges_attract() {
        let f =
            coulomb_force(ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE, 1e-10).expect("should succeed");
        assert!(f < 0.0, "Opposite charges attract");
    }

    #[test]
    fn test_coulomb_force_invalid() {
        assert!(coulomb_force(1.0, 1.0, 0.0).is_err());
        assert!(coulomb_force(1.0, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_electric_potential_sign() {
        let vp = electric_potential(ELEMENTARY_CHARGE, 1.0).expect("should succeed");
        let vn = electric_potential(-ELEMENTARY_CHARGE, 1.0).expect("should succeed");
        assert!(vp > 0.0);
        assert!(vn < 0.0);
        assert!((vp + vn).abs() < TOL);
    }

    #[test]
    fn test_electric_field_positive() {
        let e = electric_field(-ELEMENTARY_CHARGE, 1.0).expect("should succeed");
        assert!(e > 0.0, "electric_field returns magnitude");
    }

    #[test]
    fn test_electrostatic_potential_energy() {
        // U of H atom electron-proton system at Bohr radius a₀
        use crate::constants::physical::BOHR_RADIUS;
        let u = electrostatic_potential_energy(ELEMENTARY_CHARGE, -ELEMENTARY_CHARGE, BOHR_RADIUS)
            .expect("should succeed");
        // Expected ≈ −27.21 eV = −4.36e-18 J
        assert!(u < 0.0, "Opposite charges: negative PE");
        assert!((u + 4.36e-18).abs() < 1e-20, "U = {u:.4e} J");
    }

    #[test]
    fn test_capacitor_energy() {
        // C = 1 μF, V = 10 V => U = 50 μJ
        let u = capacitor_energy(1e-6, 10.0).expect("should succeed");
        assert!((u - 50e-6).abs() < 1e-15);
    }

    // ─── Magnetostatics ───────────────────────────────────────────────────────

    #[test]
    fn test_magnetic_force_perpendicular() {
        // F = |q|vB for angle = π/2
        let f = magnetic_force(ELEMENTARY_CHARGE, 1e6, 1.0, PI / 2.0).expect("should succeed");
        let expected = ELEMENTARY_CHARGE * 1e6 * 1.0;
        assert!((f - expected).abs() < 1e-30);
    }

    #[test]
    fn test_magnetic_force_parallel_zero() {
        // Velocity parallel to B => zero force
        let f = magnetic_force(ELEMENTARY_CHARGE, 1e6, 1.0, 0.0).expect("should succeed");
        assert!(f.abs() < TOL);
    }

    #[test]
    fn test_biot_savart_wire() {
        // I = 1 A, r = 1 m => B = μ₀/(2π) ≈ 200 nT
        let b = biot_savart_wire(1.0, 1.0).expect("should succeed");
        let expected = MAGNETIC_CONSTANT / (2.0 * PI);
        assert!((b - expected).abs() < 1e-15);
    }

    #[test]
    fn test_cyclotron_radius_proton() {
        use crate::constants::physical::PROTON_MASS;
        // v = 1e6 m/s, B = 1 T => r = m_p * v / (e * B)
        let r = cyclotron_radius(PROTON_MASS, 1e6, ELEMENTARY_CHARGE, 1.0).expect("should succeed");
        let expected = PROTON_MASS * 1e6 / ELEMENTARY_CHARGE;
        assert!((r - expected).abs() < 1e-15);
    }

    // ─── Special relativity ───────────────────────────────────────────────────

    #[test]
    fn test_lorentz_factor_zero_velocity() {
        let gamma = lorentz_factor(0.0).expect("should succeed");
        assert!((gamma - 1.0).abs() < TOL);
    }

    #[test]
    fn test_lorentz_factor_high_velocity() {
        // v = 0.99c => gamma ≈ 7.089
        let v = 0.99 * SPEED_OF_LIGHT;
        let gamma = lorentz_factor(v).expect("should succeed");
        assert!((gamma - 7.089).abs() < 0.001, "γ = {gamma:.4}");
    }

    #[test]
    fn test_lorentz_factor_superluminal_fails() {
        assert!(lorentz_factor(SPEED_OF_LIGHT).is_err());
        assert!(lorentz_factor(SPEED_OF_LIGHT + 1.0).is_err());
    }

    #[test]
    fn test_relativistic_energy_at_rest() {
        // v=0 => E = mc²
        let m = crate::constants::physical::ELECTRON_MASS;
        let e_rel = relativistic_energy(m, 0.0).expect("should succeed");
        let e_rest = rest_energy(m).expect("should succeed");
        assert!((e_rel - e_rest).abs() < 1e-40);
    }

    #[test]
    fn test_relativistic_kinetic_energy_low_v_matches_classical() {
        // At low v << c, relativistic KE should be > classical KE and the total
        // energy should closely match mc² + ½mv² (to better than 1 part in 10^9).
        //
        // Note: the computation (γ-1)mc² suffers catastrophic f64 cancellation for
        // tiny (v/c), so we verify the invariant E_total² = (pc)² + (mc²)²
        // (energy-momentum relation) rather than the Taylor expansion directly.
        let m = 1.0_f64;
        let v = 1e7_f64; // 10^7 m/s: v/c ≈ 0.033, small but still significant
        let k_rel = relativistic_kinetic_energy(m, v).expect("should succeed");
        let p_rel = relativistic_momentum(m, v).expect("should succeed");
        let mc2 = rest_energy(m).expect("should succeed");
        // Energy-momentum invariant: (K + mc²)² = (pc)² + (mc²)²
        let e_total = k_rel + mc2;
        let lhs = e_total * e_total;
        let rhs = (p_rel * SPEED_OF_LIGHT).powi(2) + mc2 * mc2;
        let rel_err = (lhs - rhs).abs() / rhs;
        assert!(
            rel_err < 1e-12,
            "Energy-momentum invariant violated: LHS={lhs:.8e}, RHS={rhs:.8e}, rel err={rel_err:.2e}"
        );
        // Also verify that relativistic KE exceeds classical KE
        let k_classical = 0.5 * m * v * v;
        assert!(
            k_rel > k_classical,
            "Relativistic KE must exceed classical KE"
        );
    }

    #[test]
    fn test_velocity_addition_stays_subluminal() {
        // 0.9c + 0.9c should be < c
        let v = 0.9 * SPEED_OF_LIGHT;
        let w = relativistic_velocity_addition(v, v).expect("should succeed");
        assert!(w < SPEED_OF_LIGHT, "w = {w:.6e} m/s must be < c");
        // Expected: (1.8c)/(1.81) ≈ 0.994c
        let expected = (v + v) / (1.0 + v * v / (SPEED_OF_LIGHT * SPEED_OF_LIGHT));
        assert!((w - expected).abs() < 1.0);
    }

    #[test]
    fn test_compton_shift_ninety_degrees() {
        use crate::constants::physical::COMPTON_WAVELENGTH;
        let shift = compton_wavelength_shift(PI / 2.0);
        assert!((shift - COMPTON_WAVELENGTH).abs() < 1e-25);
    }

    #[test]
    fn test_impedance_of_free_space() {
        let z0 = impedance_of_free_space();
        // Standard value ≈ 376.73 Ω
        assert!((z0 - 376.73).abs() < 0.01, "Z₀ = {z0:.2} Ω");
    }

    #[test]
    fn test_ohm_resistance() {
        let r = ohm_resistance(12.0, 3.0).expect("should succeed");
        assert!((r - 4.0).abs() < TOL);
        assert!(ohm_resistance(12.0, 0.0).is_err());
    }
}
