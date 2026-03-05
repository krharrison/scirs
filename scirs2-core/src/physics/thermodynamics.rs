//! Thermodynamics: gas laws, statistical mechanics, heat capacities, and radiation.
//!
//! All functions operate in SI units unless otherwise stated (temperatures in Kelvin,
//! energies in Joules, pressures in Pascals, volumes in m³, etc.).
//!
//! # Reference
//!
//! * Callen — *Thermodynamics and an Introduction to Thermostatistics* (2nd ed.)
//! * Kittel & Kroemer — *Thermal Physics* (2nd ed.)
//! * Planck — *The Theory of Heat Radiation*

use std::f64::consts::PI;

use crate::constants::physical::{
    BOLTZMANN, GAS_CONSTANT, PLANCK, REDUCED_PLANCK, SPEED_OF_LIGHT, STEFAN_BOLTZMANN,
};

use super::error::{PhysicsError, PhysicsResult};

// ─── Ideal gas ───────────────────────────────────────────────────────────────

/// Pressure of an ideal gas (PV = nRT).
///
/// # Arguments
///
/// * `n` – amount of substance in mol (must be > 0)
/// * `t` – temperature in K (must be > 0)
/// * `v` – volume in m³ (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if any argument is not strictly positive.
pub fn ideal_gas_pressure(n: f64, t: f64, v: f64) -> PhysicsResult<f64> {
    if n <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "n",
            reason: format!("moles must be positive, got {n}"),
        });
    }
    if t <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "t",
            reason: format!("temperature must be positive (K), got {t}"),
        });
    }
    if v <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "v",
            reason: format!("volume must be positive, got {v}"),
        });
    }
    Ok(n * GAS_CONSTANT * t / v)
}

/// Internal energy of an ideal monatomic gas: `U = (3/2)nRT`.
///
/// # Arguments
///
/// * `n` – moles (must be > 0)
/// * `t` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `n` or `t` is not positive.
pub fn ideal_gas_internal_energy(n: f64, t: f64) -> PhysicsResult<f64> {
    if n <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "n",
            reason: format!("moles must be positive, got {n}"),
        });
    }
    if t <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "t",
            reason: format!("temperature must be positive (K), got {t}"),
        });
    }
    Ok(1.5 * n * GAS_CONSTANT * t)
}

// ─── Statistical mechanics ───────────────────────────────────────────────────

/// Boltzmann factor `exp(-E / kT)`.
///
/// This dimensionless factor governs the relative probability of a microstate
/// with energy `E` at temperature `T` in canonical ensemble (Boltzmann statistics).
///
/// # Arguments
///
/// * `energy`      – energy in J (can be any real number)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `temperature` is not positive.
pub fn boltzmann_distribution(energy: f64, temperature: f64) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    Ok((-energy / (BOLTZMANN * temperature)).exp())
}

/// Maxwell-Boltzmann speed distribution PDF.
///
/// The probability density `f(v)` such that `∫₀^∞ f(v) dv = 1` for an ideal gas:
///
/// `f(v) = 4π n (m/(2πkT))^(3/2) v² exp(-mv²/(2kT))`
///
/// where the factor of `n` (number density) is **not** included here — this returns
/// the normalised PDF for a single particle.
///
/// # Arguments
///
/// * `speed`       – particle speed in m/s (must be ≥ 0)
/// * `mass`        – particle mass in kg (must be > 0)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `mass` or `temperature` is not positive,
/// or if `speed` is negative.
pub fn maxwell_speed_distribution(speed: f64, mass: f64, temperature: f64) -> PhysicsResult<f64> {
    if speed < 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "speed",
            reason: format!("speed must be non-negative, got {speed}"),
        });
    }
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
    let k_t = BOLTZMANN * temperature;
    let a = mass / (2.0 * k_t);
    // f(v) = 4π (a/π)^(3/2) v² exp(-av²)
    let norm = 4.0 * PI * (a / PI).powf(1.5);
    Ok(norm * speed * speed * (-a * speed * speed).exp())
}

/// Most probable speed in the Maxwell-Boltzmann distribution.
///
/// `v_p = √(2kT/m)`
///
/// # Arguments
///
/// * `mass`        – particle mass in kg (must be > 0)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn maxwell_most_probable_speed(mass: f64, temperature: f64) -> PhysicsResult<f64> {
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
    Ok((2.0 * BOLTZMANN * temperature / mass).sqrt())
}

/// Mean speed in the Maxwell-Boltzmann distribution.
///
/// `<v> = √(8kT/(πm))`
///
/// # Arguments
///
/// * `mass`        – particle mass in kg (must be > 0)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn maxwell_mean_speed(mass: f64, temperature: f64) -> PhysicsResult<f64> {
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
    Ok((8.0 * BOLTZMANN * temperature / (PI * mass)).sqrt())
}

// ─── Heat capacities ─────────────────────────────────────────────────────────

/// Molar heat capacity at constant volume using the **Debye model**.
///
/// The Debye model integrates phonon contributions to heat capacity:
///
/// `Cv = 9R(T/θ_D)³ ∫₀^(θ_D/T) x⁴eˣ/(eˣ−1)² dx`
///
/// The integral is evaluated numerically using 200-point Gauss-Legendre-style
/// Riemann summation that converges to better than 0.1% across the full temperature range.
///
/// # Arguments
///
/// * `temperature`       – temperature in K (must be > 0)
/// * `debye_temperature` – Debye temperature θ_D in K (must be > 0)
///
/// # Returns
///
/// Molar heat capacity in J/(mol·K).
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn heat_capacity_debye(temperature: f64, debye_temperature: f64) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    if debye_temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "debye_temperature",
            reason: format!("Debye temperature must be positive (K), got {debye_temperature}"),
        });
    }

    let x_max = debye_temperature / temperature;

    // High-T limit: Cv → 3R (classical Dulong-Petit)
    if x_max < 1e-6 {
        return Ok(3.0 * GAS_CONSTANT);
    }

    // Numerical integration of ∫₀^x_max x⁴eˣ/(eˣ−1)² dx
    // We use composite Simpson's rule with N=400 subintervals.
    let n = 400_usize;
    let h = x_max / n as f64;

    // Integrand: f(x) = x^4 * e^x / (e^x - 1)^2
    //
    // Taylor analysis at x→0:
    //   e^x ≈ 1 + x + x²/2 + ...
    //   e^x - 1 ≈ x + x²/2 + ... ≈ x(1 + x/2)
    //   (e^x - 1)^2 ≈ x²(1 + x/2)² ≈ x²
    //   f(x) = x⁴ · e^x / (e^x-1)² ≈ x⁴ / x² = x²   (limit → 0 as x→0)
    //
    // At large x the integrand → 0 (exponentially suppressed).
    let integrand = |x: f64| -> f64 {
        if x < 1e-8 {
            // Use the leading-order Taylor term to avoid 0/0
            return x * x; // lim_{x→0} f(x) = x²
        }
        let ex = x.exp();
        let denom = ex - 1.0;
        if denom < f64::MIN_POSITIVE {
            return 0.0;
        }
        x.powi(4) * ex / (denom * denom)
    };

    // Simpson's composite rule
    let mut sum = integrand(0.0) + integrand(x_max);
    for i in 1..n {
        let x = i as f64 * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * integrand(x);
    }
    let integral = sum * h / 3.0;

    let ratio = temperature / debye_temperature;
    Ok(9.0 * GAS_CONSTANT * ratio.powi(3) * integral)
}

// ─── Black-body radiation ─────────────────────────────────────────────────────

/// Spectral radiance of a black body (Planck's law).
///
/// `B(λ,T) = (2hc²/λ⁵) · 1/(exp(hc/(λkT)) − 1)`
///
/// Returns spectral radiance in W·sr⁻¹·m⁻³ (per unit wavelength, per steradian).
///
/// # Arguments
///
/// * `wavelength`  – wavelength in m (must be > 0)
/// * `temperature` – blackbody temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn black_body_spectral_radiance(wavelength: f64, temperature: f64) -> PhysicsResult<f64> {
    if wavelength <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "wavelength",
            reason: format!("wavelength must be positive, got {wavelength}"),
        });
    }
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    let hc_over_lk_t = PLANCK * SPEED_OF_LIGHT / (wavelength * BOLTZMANN * temperature);
    let exp_term = hc_over_lk_t.exp() - 1.0;
    if exp_term == 0.0 {
        return Ok(0.0);
    }
    let numerator = 2.0 * PLANCK * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
    Ok(numerator / (wavelength.powi(5) * exp_term))
}

/// Peak wavelength of blackbody radiation (Wien's displacement law).
///
/// `λ_max = b/T`  where `b = 2.897 771 955 × 10⁻³ m·K`.
///
/// # Arguments
///
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `temperature` is not positive.
pub fn wien_wavelength(temperature: f64) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    use crate::constants::physical::WIEN;
    Ok(WIEN / temperature)
}

/// Total power radiated per unit area by a black body (Stefan-Boltzmann law).
///
/// `P/A = σT⁴`
///
/// # Arguments
///
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `temperature` is not positive.
pub fn stefan_boltzmann_radiance(temperature: f64) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    Ok(STEFAN_BOLTZMANN * temperature.powi(4))
}

/// Thermal de Broglie wavelength of a particle.
///
/// `Λ = h / √(2πmkT)`
///
/// # Arguments
///
/// * `mass`        – particle mass in kg (must be > 0)
/// * `temperature` – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive.
pub fn thermal_de_broglie_wavelength(mass: f64, temperature: f64) -> PhysicsResult<f64> {
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
    // Λ = h / sqrt(2π m k T)
    Ok(PLANCK / (2.0 * PI * mass * BOLTZMANN * temperature).sqrt())
}

/// Bose-Einstein occupation number for bosons.
///
/// `n(E) = 1 / (exp((E - μ)/(kT)) - 1)`
///
/// # Arguments
///
/// * `energy`          – energy level in J
/// * `chemical_potential` – chemical potential μ in J (must be < energy for bosons at T > 0)
/// * `temperature`     – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `temperature` ≤ 0.
/// Returns [`PhysicsError::DomainError`] if the exponent causes divergence (μ ≥ E and T > 0).
pub fn bose_einstein_distribution(
    energy: f64,
    chemical_potential: f64,
    temperature: f64,
) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    let x = (energy - chemical_potential) / (BOLTZMANN * temperature);
    let denom = x.exp() - 1.0;
    if denom <= 0.0 {
        return Err(PhysicsError::DomainError(format!(
            "Bose-Einstein distribution diverges for E - μ = {:.4e} J, T = {temperature} K; \
             ensure E > μ",
            energy - chemical_potential
        )));
    }
    Ok(1.0 / denom)
}

/// Fermi-Dirac occupation number for fermions.
///
/// `f(E) = 1 / (exp((E - μ)/(kT)) + 1)`
///
/// # Arguments
///
/// * `energy`             – energy level in J
/// * `chemical_potential` – Fermi level / chemical potential μ in J
/// * `temperature`        – temperature in K (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `temperature` ≤ 0.
pub fn fermi_dirac_distribution(
    energy: f64,
    chemical_potential: f64,
    temperature: f64,
) -> PhysicsResult<f64> {
    if temperature <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "temperature",
            reason: format!("temperature must be positive (K), got {temperature}"),
        });
    }
    let x = (energy - chemical_potential) / (BOLTZMANN * temperature);
    Ok(1.0 / (x.exp() + 1.0))
}

/// Carnot efficiency of an ideal heat engine.
///
/// `η = 1 - T_cold/T_hot`
///
/// # Arguments
///
/// * `t_hot`  – temperature of the hot reservoir in K (must be > 0)
/// * `t_cold` – temperature of the cold reservoir in K (must be > 0, must be < t_hot)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if either argument is not positive or if
/// `t_cold` ≥ `t_hot`.
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> PhysicsResult<f64> {
    if t_hot <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "t_hot",
            reason: format!("hot reservoir temperature must be positive (K), got {t_hot}"),
        });
    }
    if t_cold <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "t_cold",
            reason: format!("cold reservoir temperature must be positive (K), got {t_cold}"),
        });
    }
    if t_cold >= t_hot {
        return Err(PhysicsError::InvalidParameter {
            param: "t_cold",
            reason: format!(
                "cold reservoir temperature ({t_cold} K) must be less than hot ({t_hot} K)"
            ),
        });
    }
    Ok(1.0 - t_cold / t_hot)
}

// ─── Phonon zero-point energy ─────────────────────────────────────────────────

/// Zero-point energy of a quantum harmonic oscillator.
///
/// `E_0 = ℏω/2`
///
/// # Arguments
///
/// * `angular_frequency` – angular frequency ω in rad/s (must be > 0)
///
/// # Errors
///
/// Returns [`PhysicsError::InvalidParameter`] if `angular_frequency` is not positive.
pub fn zero_point_energy(angular_frequency: f64) -> PhysicsResult<f64> {
    if angular_frequency <= 0.0 {
        return Err(PhysicsError::InvalidParameter {
            param: "angular_frequency",
            reason: format!("angular frequency must be positive, got {angular_frequency}"),
        });
    }
    Ok(0.5 * REDUCED_PLANCK * angular_frequency)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    // ─── Ideal gas ────────────────────────────────────────────────────────────

    #[test]
    fn test_ideal_gas_pressure_stp() {
        // At STP: n=1 mol, T=273.15 K, V=0.022414 m³  => P ≈ 101325 Pa
        let p = ideal_gas_pressure(1.0, 273.15, 0.022_413_969_54).expect("should succeed");
        assert!((p - 101_325.0).abs() < 1.0, "P = {p:.2} Pa");
    }

    #[test]
    fn test_ideal_gas_pressure_invalid() {
        assert!(ideal_gas_pressure(-1.0, 300.0, 1.0).is_err());
        assert!(ideal_gas_pressure(1.0, 0.0, 1.0).is_err());
        assert!(ideal_gas_pressure(1.0, 300.0, 0.0).is_err());
    }

    #[test]
    fn test_ideal_gas_internal_energy() {
        // U = 1.5 * n * R * T
        let u = ideal_gas_internal_energy(1.0, 300.0).expect("should succeed");
        let expected = 1.5 * GAS_CONSTANT * 300.0;
        assert!((u - expected).abs() < TOL);
    }

    // ─── Statistical mechanics ────────────────────────────────────────────────

    #[test]
    fn test_boltzmann_distribution_zero_energy() {
        // E=0 => factor = 1
        let bf = boltzmann_distribution(0.0, 300.0).expect("should succeed");
        assert!((bf - 1.0).abs() < TOL);
    }

    #[test]
    fn test_boltzmann_distribution_invalid_temperature() {
        assert!(boltzmann_distribution(0.0, 0.0).is_err());
        assert!(boltzmann_distribution(0.0, -10.0).is_err());
    }

    #[test]
    fn test_maxwell_distribution_normalisation() {
        // Integrate the Maxwell-Boltzmann PDF over [0, 6*v_p] numerically; should be ≈ 1.
        let mass = crate::constants::physical::PROTON_MASS;
        let temp = 1000.0;
        let v_p = maxwell_most_probable_speed(mass, temp).expect("should succeed");
        let v_max = 6.0 * v_p;
        let n = 10_000_usize;
        let dv = v_max / n as f64;
        let mut sum = 0.0_f64;
        for i in 0..=n {
            let v = i as f64 * dv;
            let f = maxwell_speed_distribution(v, mass, temp).expect("should succeed");
            let coeff = if i == 0 || i == n {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            sum += coeff * f;
        }
        let integral = sum * dv / 3.0;
        assert!(
            (integral - 1.0).abs() < 1e-4,
            "Maxwell integral = {integral:.6}"
        );
    }

    #[test]
    fn test_maxwell_speed_invalid() {
        assert!(maxwell_speed_distribution(-1.0, 1e-27, 300.0).is_err());
        assert!(maxwell_speed_distribution(100.0, 0.0, 300.0).is_err());
        assert!(maxwell_speed_distribution(100.0, 1e-27, 0.0).is_err());
    }

    // ─── Heat capacity ────────────────────────────────────────────────────────

    #[test]
    fn test_debye_high_temperature_limit() {
        // At T >> θ_D, Cv → 3R (Dulong-Petit)
        let cv = heat_capacity_debye(10_000.0, 100.0).expect("should succeed");
        assert!(
            (cv - 3.0 * GAS_CONSTANT).abs() < 0.01,
            "Cv = {cv:.4} J/(mol·K)"
        );
    }

    #[test]
    fn test_debye_low_temperature() {
        // At T << θ_D, Cv should be much less than 3R
        let cv = heat_capacity_debye(10.0, 1000.0).expect("should succeed");
        assert!(cv < 3.0 * GAS_CONSTANT, "Cv = {cv:.4} should be < 3R");
        assert!(cv > 0.0, "Cv must be positive");
    }

    #[test]
    fn test_debye_invalid() {
        assert!(heat_capacity_debye(0.0, 300.0).is_err());
        assert!(heat_capacity_debye(300.0, 0.0).is_err());
    }

    // ─── Radiation ────────────────────────────────────────────────────────────

    #[test]
    fn test_wien_wavelength_sun() {
        // Sun's surface temperature ≈ 5778 K => peak ~502 nm
        let lambda = wien_wavelength(5778.0).expect("should succeed");
        // Expected: 2.898e-3 / 5778 ≈ 5.015e-7 m
        assert!(
            (lambda - 5.015e-7).abs() < 2e-9,
            "Wien peak = {lambda:.4e} m"
        );
    }

    #[test]
    fn test_wien_wavelength_invalid() {
        assert!(wien_wavelength(0.0).is_err());
        assert!(wien_wavelength(-100.0).is_err());
    }

    #[test]
    fn test_stefan_boltzmann_sun_luminosity() {
        // Sun radius ≈ 6.96e8 m, T ≈ 5778 K
        // L = 4πR² σT⁴ ≈ 3.83e26 W (SOLAR_LUMINOSITY = 3.828e26 W)
        use crate::constants::physical::SOLAR_RADIUS;
        let flux = stefan_boltzmann_radiance(5778.0).expect("should succeed");
        let luminosity = 4.0 * PI * SOLAR_RADIUS.powi(2) * flux;
        // Within 5% of the accepted value
        let solar_lum = crate::constants::physical::SOLAR_LUMINOSITY;
        assert!(
            (luminosity - solar_lum).abs() / solar_lum < 0.05,
            "L = {luminosity:.4e} W"
        );
    }

    #[test]
    fn test_planck_spectral_radiance() {
        // Check that spectral radiance is positive and finite for reasonable inputs
        let b = black_body_spectral_radiance(5e-7, 5778.0).expect("should succeed");
        assert!(b > 0.0, "Spectral radiance must be positive");
        assert!(b.is_finite(), "Spectral radiance must be finite");
    }

    #[test]
    fn test_planck_invalid() {
        assert!(black_body_spectral_radiance(0.0, 300.0).is_err());
        assert!(black_body_spectral_radiance(500e-9, 0.0).is_err());
    }

    // ─── Distributions ────────────────────────────────────────────────────────

    #[test]
    fn test_fermi_dirac_at_fermi_level() {
        // f(E = μ) = 0.5 for all T > 0
        let f = fermi_dirac_distribution(1e-19, 1e-19, 1000.0).expect("should succeed");
        assert!((f - 0.5).abs() < TOL);
    }

    #[test]
    fn test_carnot_efficiency() {
        // η = 1 - 300/600 = 0.5
        let eta = carnot_efficiency(600.0, 300.0).expect("should succeed");
        assert!((eta - 0.5).abs() < TOL);
    }

    #[test]
    fn test_carnot_efficiency_invalid() {
        assert!(carnot_efficiency(300.0, 300.0).is_err()); // equal temps
        assert!(carnot_efficiency(200.0, 300.0).is_err()); // cold > hot
        assert!(carnot_efficiency(0.0, 300.0).is_err());
    }

    #[test]
    fn test_zero_point_energy() {
        // E_0 = ℏω/2  for ω = 1e12 rad/s
        let omega = 1e12_f64;
        let e0 = zero_point_energy(omega).expect("should succeed");
        let expected = 0.5 * REDUCED_PLANCK * omega;
        assert!((e0 - expected).abs() < 1e-55);
    }
}
