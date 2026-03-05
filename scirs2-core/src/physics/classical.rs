//! Classical mechanics: kinematics, dynamics, projectile motion, and oscillators.
//!
//! All functions operate in SI units unless otherwise stated.
//!
//! # Reference
//!
//! * Goldstein, Poole & Safko — *Classical Mechanics* (3rd ed.)
//! * Kibble & Berkshire — *Classical Mechanics* (5th ed.)

use std::f64::consts::PI;

use super::error::{PhysicsError, PhysicsResult};

// ─── Result types ────────────────────────────────────────────────────────────

/// Result of a projectile motion calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectileResult {
    /// Maximum height above the launch point (m).
    pub max_height: f64,
    /// Horizontal range (m) — distance from launch to landing on level ground.
    pub range: f64,
    /// Total time of flight (s).
    pub time_of_flight: f64,
    /// Horizontal velocity component (m/s).
    pub vx: f64,
    /// Initial vertical velocity component (m/s).
    pub vy0: f64,
}

/// Result of a simple harmonic oscillator characterisation.
#[derive(Debug, Clone, PartialEq)]
pub struct OscillatorResult {
    /// Angular frequency ω (rad/s).
    pub angular_frequency: f64,
    /// Period T (s).
    pub period: f64,
    /// Frequency f (Hz).
    pub frequency: f64,
}

// ─── Kinematic quantities ────────────────────────────────────────────────────

/// Kinetic energy of a non-relativistic particle.
///
/// `KE = ½mv²`
///
/// # Arguments
///
/// * `mass`     – mass in kg (must be > 0)
/// * `velocity` – speed in m/s
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` is not positive.
pub fn kinetic_energy(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    Ok(0.5 * mass * velocity * velocity)
}

/// Gravitational potential energy near Earth's surface (or any uniform field).
///
/// `PE = mgh`
///
/// # Arguments
///
/// * `mass`   – mass in kg (must be > 0)
/// * `height` – height above the reference level in m
/// * `g`      – gravitational acceleration in m/s² (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `g` is not positive.
pub fn potential_energy_gravity(mass: f64, height: f64, g: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if g <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "g",
            reason: format!("gravitational acceleration must be positive, got {g}"),
        });
    }
    Ok(mass * g * height)
}

/// Linear momentum `p = mv`.
///
/// # Arguments
///
/// * `mass`     – mass in kg (must be > 0)
/// * `velocity` – velocity in m/s
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` is not positive.
pub fn momentum(mass: f64, velocity: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    Ok(mass * velocity)
}

/// Angular momentum `L = mvr` for circular motion.
///
/// # Arguments
///
/// * `mass`     – mass in kg (must be > 0)
/// * `velocity` – tangential speed in m/s
/// * `radius`   – orbital radius in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `radius` is not positive.
pub fn angular_momentum(mass: f64, velocity: f64, radius: f64) -> PhysicsResult<f64> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if radius <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "radius",
            reason: format!("radius must be positive, got {radius}"),
        });
    }
    Ok(mass * velocity * radius)
}

/// Centripetal acceleration `a = v²/r`.
///
/// # Arguments
///
/// * `velocity` – tangential speed in m/s
/// * `radius`   – radius of curvature in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `radius` is not positive.
pub fn centripetal_acceleration(velocity: f64, radius: f64) -> PhysicsResult<f64> {
    if radius <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "radius",
            reason: format!("radius must be positive, got {radius}"),
        });
    }
    Ok(velocity * velocity / radius)
}

// ─── Projectile motion ───────────────────────────────────────────────────────

/// Full projectile-motion analysis on a flat surface.
///
/// The launch point is taken as the origin; the projectile is fired with initial
/// speed `v0` at an angle `angle_deg` above the horizontal in a uniform gravitational
/// field `g`.  Air resistance is neglected.
///
/// # Arguments
///
/// * `v0`        – initial speed in m/s (must be ≥ 0)
/// * `angle_deg` – launch angle in degrees above the horizontal (0–90)
/// * `g`         – gravitational acceleration in m/s² (must be > 0)
///
/// # Errors
///
/// * [`PhysicsError::InvalidParameter`] if `g` ≤ 0 or `v0` < 0.
/// * [`PhysicsError::DomainError`] if `angle_deg` is outside [0, 90].
pub fn projectile_motion(v0: f64, angle_deg: f64, g: f64) -> PhysicsResult<ProjectileResult> {
    if v0 < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "v0",
            reason: format!("initial speed must be non-negative, got {v0}"),
        });
    }
    if g <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "g",
            reason: format!("gravitational acceleration must be positive, got {g}"),
        });
    }
    if !(0.0..=90.0).contains(&angle_deg) {
        return Err(PhysicsError::DomainError(format!(
            "launch angle must be in [0, 90] degrees, got {angle_deg}"
        )));
    }

    let angle_rad = angle_deg * PI / 180.0;
    let vx = v0 * angle_rad.cos();
    let vy0 = v0 * angle_rad.sin();

    // Time of flight: projectile returns to y = 0  →  t = 2·vy0/g
    let time_of_flight = 2.0 * vy0 / g;
    // Maximum height: reached at t = vy0/g
    let max_height = vy0 * vy0 / (2.0 * g);
    // Horizontal range
    let range = vx * time_of_flight;

    Ok(ProjectileResult {
        max_height,
        range,
        time_of_flight,
        vx,
        vy0,
    })
}

/// Position of a projectile at time `t` after launch.
///
/// Returns `(x, y)` in meters.
///
/// # Arguments
///
/// * `v0`        – initial speed (m/s, must be ≥ 0)
/// * `angle_deg` – launch angle in degrees (0–90)
/// * `g`         – gravitational acceleration (m/s², must be > 0)
/// * `t`         – elapsed time in seconds (must be ≥ 0)
///
/// # Errors
///
/// Propagates errors from [`projectile_motion`]; additionally returns
/// [`PhysicsError::InvalidParameter`] if `t` < 0.
pub fn projectile_position(v0: f64, angle_deg: f64, g: f64, t: f64) -> PhysicsResult<(f64, f64)> {
    if t < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "t",
            reason: format!("time must be non-negative, got {t}"),
        });
    }
    let proj = projectile_motion(v0, angle_deg, g)?;
    let x = proj.vx * t;
    let y = proj.vy0 * t - 0.5 * g * t * t;
    Ok((x, y))
}

// ─── Simple harmonic oscillator ──────────────────────────────────────────────

/// Characterise a mass-spring simple harmonic oscillator.
///
/// `ω = √(k/m)`,  `T = 2π/ω`,  `f = ω/(2π)`
///
/// # Arguments
///
/// * `mass`           – mass in kg (must be > 0)
/// * `spring_constant` – spring constant in N/m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn simple_harmonic_oscillator(
    mass: f64,
    spring_constant: f64,
) -> PhysicsResult<OscillatorResult> {
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if spring_constant <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "spring_constant",
            reason: format!("spring constant must be positive, got {spring_constant}"),
        });
    }
    let angular_frequency = (spring_constant / mass).sqrt();
    let period = 2.0 * PI / angular_frequency;
    let frequency = angular_frequency / (2.0 * PI);
    Ok(OscillatorResult {
        angular_frequency,
        period,
        frequency,
    })
}

/// Displacement of a simple harmonic oscillator at time `t`.
///
/// `x(t) = A · cos(ωt + φ)`
///
/// # Arguments
///
/// * `amplitude`     – amplitude in m (must be ≥ 0)
/// * `angular_freq`  – angular frequency ω in rad/s (must be > 0)
/// * `time`          – time in s (must be ≥ 0)
/// * `phase`         – initial phase φ in radians
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] for non-positive `angular_freq`, negative `amplitude`, or negative `time`.
pub fn sho_displacement(
    amplitude: f64,
    angular_freq: f64,
    time: f64,
    phase: f64,
) -> PhysicsResult<f64> {
    if amplitude < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "amplitude",
            reason: format!("amplitude must be non-negative, got {amplitude}"),
        });
    }
    if angular_freq <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "angular_freq",
            reason: format!("angular frequency must be positive, got {angular_freq}"),
        });
    }
    if time < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "time",
            reason: format!("time must be non-negative, got {time}"),
        });
    }
    Ok(amplitude * (angular_freq * time + phase).cos())
}

/// Gravitational force between two point masses (Newton's law of gravitation).
///
/// `F = G·m₁·m₂/r²`  where `G` is Newton's gravitational constant.
///
/// # Arguments
///
/// * `m1`       – first mass in kg (must be > 0)
/// * `m2`       – second mass in kg (must be > 0)
/// * `distance` – separation in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if any argument is not positive.
pub fn gravitational_force(m1: f64, m2: f64, distance: f64) -> PhysicsResult<f64> {
    use crate::constants::physical::GRAVITATIONAL_CONSTANT;
    if m1 <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "m1",
            reason: format!("mass m1 must be positive, got {m1}"),
        });
    }
    if m2 <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "m2",
            reason: format!("mass m2 must be positive, got {m2}"),
        });
    }
    if distance <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "distance",
            reason: format!("distance must be positive, got {distance}"),
        });
    }
    Ok(GRAVITATIONAL_CONSTANT * m1 * m2 / (distance * distance))
}

/// Escape velocity from the surface of a spherical body.
///
/// `v_esc = √(2GM/R)`
///
/// # Arguments
///
/// * `mass`   – mass of the body in kg (must be > 0)
/// * `radius` – radius of the body in m (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `radius` is not positive.
pub fn escape_velocity(mass: f64, radius: f64) -> PhysicsResult<f64> {
    use crate::constants::physical::GRAVITATIONAL_CONSTANT;
    if mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "mass",
            reason: format!("mass must be positive, got {mass}"),
        });
    }
    if radius <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "radius",
            reason: format!("radius must be positive, got {radius}"),
        });
    }
    Ok((2.0 * GRAVITATIONAL_CONSTANT * mass / radius).sqrt())
}

/// Orbital period of a circular orbit (Kepler's third law).
///
/// `T = 2π · √(r³ / (GM))`
///
/// # Arguments
///
/// * `orbital_radius` – semi-major axis / orbital radius in m (must be > 0)
/// * `central_mass`   – mass of the central body in kg (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn orbital_period(orbital_radius: f64, central_mass: f64) -> PhysicsResult<f64> {
    use crate::constants::physical::GRAVITATIONAL_CONSTANT;
    if orbital_radius <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "orbital_radius",
            reason: format!("orbital radius must be positive, got {orbital_radius}"),
        });
    }
    if central_mass <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "central_mass",
            reason: format!("central mass must be positive, got {central_mass}"),
        });
    }
    Ok(2.0 * PI * (orbital_radius.powi(3) / (GRAVITATIONAL_CONSTANT * central_mass)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    #[test]
    fn test_kinetic_energy_basic() {
        // KE = 0.5 * 2.0 * 3.0^2 = 9.0
        let ke = kinetic_energy(2.0, 3.0).expect("should succeed");
        assert!((ke - 9.0).abs() < TOL, "KE = {ke}");
    }

    #[test]
    fn test_kinetic_energy_zero_velocity() {
        let ke = kinetic_energy(5.0, 0.0).expect("should succeed");
        assert_eq!(ke, 0.0);
    }

    #[test]
    fn test_kinetic_energy_invalid_mass() {
        assert!(kinetic_energy(-1.0, 1.0).is_err());
        assert!(kinetic_energy(0.0, 1.0).is_err());
    }

    #[test]
    fn test_potential_energy_gravity() {
        // PE = 2 * 9.81 * 10 = 196.2
        let pe = potential_energy_gravity(2.0, 10.0, 9.81).expect("should succeed");
        assert!((pe - 196.2).abs() < 1e-10);
    }

    #[test]
    fn test_potential_energy_negative_height() {
        // Negative height (below reference) is allowed — gives negative PE.
        let pe = potential_energy_gravity(1.0, -5.0, 9.81).expect("should succeed");
        assert!(pe < 0.0);
    }

    #[test]
    fn test_momentum() {
        let p = momentum(3.0, 4.0).expect("should succeed");
        assert!((p - 12.0).abs() < TOL);
    }

    #[test]
    fn test_angular_momentum() {
        // L = 1.0 * 2.0 * 3.0 = 6.0
        let l = angular_momentum(1.0, 2.0, 3.0).expect("should succeed");
        assert!((l - 6.0).abs() < TOL);
    }

    #[test]
    fn test_centripetal_acceleration() {
        // a = v^2/r = 4 / 2 = 2
        let a = centripetal_acceleration(2.0, 2.0).expect("should succeed");
        assert!((a - 2.0).abs() < TOL);
    }

    #[test]
    fn test_projectile_45_degrees() {
        // At 45° launch angle, range is maximised: R = v0²/g
        let v0 = 20.0;
        let g = 9.81;
        let result = projectile_motion(v0, 45.0, g).expect("should succeed");
        let expected_range = v0 * v0 / g;
        assert!(
            (result.range - expected_range).abs() < 1e-10,
            "range = {}, expected = {}",
            result.range,
            expected_range
        );
    }

    #[test]
    fn test_projectile_90_degrees_zero_range() {
        // Straight up: range = 0, max height = v0²/(2g)
        let v0 = 10.0;
        let g = 9.81;
        let result = projectile_motion(v0, 90.0, g).expect("should succeed");
        assert!(result.range.abs() < 1e-10, "range should be ~0");
        let expected_height = v0 * v0 / (2.0 * g);
        assert!((result.max_height - expected_height).abs() < 1e-10);
    }

    #[test]
    fn test_projectile_invalid_angle() {
        assert!(projectile_motion(10.0, 95.0, 9.81).is_err());
        assert!(projectile_motion(10.0, -1.0, 9.81).is_err());
    }

    #[test]
    fn test_simple_harmonic_oscillator() {
        // k=4, m=1  => ω=2, T=π, f=1/π
        let result = simple_harmonic_oscillator(1.0, 4.0).expect("should succeed");
        assert!((result.angular_frequency - 2.0).abs() < TOL);
        assert!((result.period - PI).abs() < TOL);
        assert!((result.frequency - 1.0 / PI).abs() < TOL);
    }

    #[test]
    fn test_sho_displacement_at_zero() {
        // x(0) = A * cos(φ)
        let d = sho_displacement(3.0, 2.0, 0.0, 0.0).expect("should succeed");
        assert!((d - 3.0).abs() < TOL);
    }

    #[test]
    fn test_escape_velocity_earth() {
        use crate::constants::physical::{EARTH_MASS, EARTH_RADIUS};
        let ve = escape_velocity(EARTH_MASS, EARTH_RADIUS).expect("should succeed");
        // Expected ~11.2 km/s
        assert!(
            (ve - 11_186.0).abs() < 200.0,
            "escape velocity = {ve:.1} m/s"
        );
    }

    #[test]
    fn test_gravitational_force_known() {
        use crate::constants::physical::GRAVITATIONAL_CONSTANT;
        // Two 1 kg masses 1 m apart: F = G
        let f = gravitational_force(1.0, 1.0, 1.0).expect("should succeed");
        assert!((f - GRAVITATIONAL_CONSTANT).abs() < 1e-20);
    }

    #[test]
    fn test_orbital_period_earth_approx() {
        use crate::constants::physical::{EARTH_RADIUS, SOLAR_MASS};
        // Earth at 1 AU around the Sun (use AU constant)
        use crate::constants::physical::ASTRONOMICAL_UNIT;
        let period = orbital_period(ASTRONOMICAL_UNIT, SOLAR_MASS).expect("should succeed");
        // One year ≈ 3.156e7 s
        let one_year = 3.156e7_f64;
        assert!(
            (period - one_year).abs() / one_year < 0.01,
            "orbital period = {period:.3e} s"
        );
        // Ensure radius/mass validation works
        assert!(orbital_period(0.0, SOLAR_MASS).is_err());
        assert!(orbital_period(EARTH_RADIUS, 0.0).is_err());
    }
}
