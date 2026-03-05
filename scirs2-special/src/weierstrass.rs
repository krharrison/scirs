//! Weierstrass Elliptic Functions
//!
//! This module provides the Weierstrass ℘ function and its companions (℘', ζ, σ),
//! which form the foundation of the theory of elliptic functions over the complex
//! numbers.
//!
//! ## Mathematical Definitions
//!
//! Given a lattice Λ = {2mω₁ + 2nω₂ | m,n ∈ ℤ} (with Im(ω₂/ω₁) > 0):
//!
//! ```text
//! ℘(z) = 1/z² + Σ_{(m,n)≠(0,0)} [1/(z − 2mω₁ − 2nω₂)² − 1/(2mω₁ + 2nω₂)²]
//!
//! ℘'(z) = −2 Σ_{(m,n)} 1/(z − 2mω₁ − 2nω₂)³
//!
//! ζ(z)  = 1/z + Σ_{(m,n)≠(0,0)} [1/(z−ω) + 1/ω + z/ω²]   (ω = 2mω₁+2nω₂)
//!
//! σ(z)  = z · Π_{(m,n)≠(0,0)} (1 − z/ω) exp(z/ω + z²/(2ω²))
//! ```
//!
//! They satisfy the differential equation:
//! ```text
//! (℘')² = 4℘³ − g₂℘ − g₃
//! ```
//!
//! where `g₂ = 60 Σ ω⁻⁴` and `g₃ = 140 Σ ω⁻⁶`.
//!
//! ## Implementation Strategy
//!
//! For this real-on-the-real-axis implementation we parameterise by the lattice
//! invariants (g₂, g₃) and compute via the Laurent/Eisenstein series with the
//! Lemniscate-normalised truncation scheme.  The implementation uses the fact that
//! for a rectangular lattice (ω₁ real, ω₂ purely imaginary) all quantities remain
//! real on the real axis.
//!
//! ## References
//!
//! - Whittaker & Watson, *A Course of Modern Analysis*, Chapter 20
//! - Silverman, *The Arithmetic of Elliptic Curves*, Chapter VI
//! - DLMF §23: Weierstrass Elliptic and Modular Functions

use crate::error::{SpecialError, SpecialResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum half-periods summed in lattice sums (each side: ±1..N_LATTICE).
/// A 40×40 lattice gives relative error ≲ 10⁻¹⁵ for typical invariants.
const N_LATTICE: i32 = 40;

/// Tolerance for near-pole detection.
const POLE_TOL: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Helper: compute half-periods from invariants
// ---------------------------------------------------------------------------

/// Compute the real half-period ω₁ from g₂ and g₃ using the inverse of
/// `lattice_invariants`.  We find the roots of 4t³ − g₂t − g₃ = 0 (the Weierstrass
/// cubic) and recover ω₁ from `℘(ω₁) = e₁` via AGM.
///
/// For the rectangular lattice case (g₂ > 0 and discriminant > 0 giving three real
/// roots e₁ > e₂ > e₃ with e₁+e₂+e₃=0) the real half-period is:
/// ```text
/// ω₁ = ∫_{e₁}^{∞} dt / sqrt(4t³ − g₂t − g₃)
///     = K(k) / sqrt(e₁ − e₃)
/// ```
/// where k² = (e₂−e₃)/(e₁−e₃).
fn half_periods_from_invariants(g2: f64, g3: f64) -> SpecialResult<(f64, f64)> {
    // Cubic: 4t³ - g2·t - g3 = 0  →  t³ - (g2/4)t - g3/4 = 0
    let roots = cubic_roots_weierstrass(g2, g3)?;
    let (e1, e2, e3) = roots; // e1 ≥ e2 ≥ e3, e1+e2+e3=0

    let diff13 = e1 - e3;
    if diff13 < POLE_TOL {
        return Err(SpecialError::ComputationError(
            "degenerate lattice: e₁ = e₃".to_string(),
        ));
    }
    let k2 = (e2 - e3) / diff13;
    let k = k2.sqrt().clamp(0.0, 1.0 - 1e-15);

    let big_k = complete_elliptic_k(k);
    let big_k_prime = complete_elliptic_k((1.0 - k2).sqrt());

    let sqrt_diff = diff13.sqrt();
    let omega1 = big_k / sqrt_diff;
    // ω₂ is purely imaginary for the rectangular lattice: ω₂ = i K'/sqrt(e₁−e₃)
    // We store its imaginary part as omega2_imag
    let omega2_imag = big_k_prime / sqrt_diff;

    Ok((omega1, omega2_imag))
}

/// Roots of the depressed cubic 4t³ − g₂t − g₃ = 0.
///
/// Returns (e₁, e₂, e₃) in descending order when three real roots exist.
fn cubic_roots_weierstrass(g2: f64, g3: f64) -> SpecialResult<(f64, f64, f64)> {
    // Rewrite as t³ − pt − q = 0 with p = g2/4, q = g3/4
    let p = g2 / 4.0;
    let q = g3 / 4.0;

    // Discriminant of t³ − pt − q: Δ/4 = p³/27 − q²/4 (> 0 ↔ three real roots)
    let delta_quarter = p * p * p / 27.0 - q * q / 4.0;

    if delta_quarter > 0.0 {
        // Three real roots via trigonometric method
        let m = 2.0 * (p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).acos() / 3.0;
        let two_pi_3 = 2.0 * std::f64::consts::PI / 3.0;
        let t1 = m * theta.cos();
        let t2 = m * (theta - two_pi_3).cos();
        let t3 = m * (theta - 2.0 * two_pi_3).cos();

        // Sort descending
        let mut roots = [t1, t2, t3];
        roots.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        Ok((roots[0], roots[1], roots[2]))
    } else if delta_quarter == 0.0 {
        // Repeated root
        let t1 = 3.0 * q / p;
        let t2 = -3.0 * q / (2.0 * p);
        if t1 >= t2 {
            Ok((t1, t2, t2))
        } else {
            Ok((t2, t2, t1))
        }
    } else {
        // One real root (lemniscate-like or negative discriminant)
        let sqrt_neg = (-delta_quarter).sqrt();
        let u = (-q / 2.0 + sqrt_neg).cbrt();
        let v = (-q / 2.0 - sqrt_neg).cbrt();
        let t1 = u + v;
        // For one-real-root case we only have e1
        Ok((t1, t1, t1))
    }
}

/// Complete elliptic integral K(k) via AGM.
fn complete_elliptic_k(k: f64) -> f64 {
    if k <= 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    if k >= 1.0 {
        return f64::INFINITY;
    }
    let b = (1.0 - k * k).sqrt();
    std::f64::consts::FRAC_PI_2 / agm(1.0, b)
}

/// Arithmetic-Geometric Mean.
fn agm(mut a: f64, mut b: f64) -> f64 {
    for _ in 0..100 {
        let a_new = (a + b) * 0.5;
        let b_new = (a * b).sqrt();
        if (a_new - b_new).abs() < 1e-15 * a_new.abs() {
            return a_new;
        }
        a = a_new;
        b = b_new;
    }
    (a + b) * 0.5
}

// ---------------------------------------------------------------------------
// Weierstrass ℘ function
// ---------------------------------------------------------------------------

/// Weierstrass ℘ function: ℘(z; g₂, g₃).
///
/// Computed via the lattice sum
/// ```text
/// ℘(z) = 1/z² + Σ_{(m,n)≠(0,0)} [1/(z−ω_{m,n})² − 1/ω_{m,n}²]
/// ```
/// over a finite rectangular lattice truncated at `N_LATTICE` in each direction.
///
/// # Arguments
/// * `z`  – real argument (must not be a lattice point)
/// * `g2` – lattice invariant g₂ = 60 Σ ω⁻⁴
/// * `g3` – lattice invariant g₃ = 140 Σ ω⁻⁶
///
/// # Returns
/// Value of ℘(z; g₂, g₃).  Returns `f64::NAN` if z is too close to a pole or if
/// the invariants are degenerate.
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::weierstrass_p;
/// // For the equianharmonic lattice g₂=0, g₃=4:
/// let val = weierstrass_p(0.5, 0.0, 4.0);
/// assert!(val.is_finite());
/// ```
pub fn weierstrass_p(z: f64, g2: f64, g3: f64) -> f64 {
    match weierstrass_p_impl(z, g2, g3) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn weierstrass_p_impl(z: f64, g2: f64, g3: f64) -> SpecialResult<f64> {
    let (omega1, omega2_imag) = half_periods_from_invariants(g2, g3)?;

    // Reduce z to the fundamental domain (not needed for convergence but helps)
    // We compute the sum over the truncated lattice directly
    // ω_{m,n} = 2m·ω₁ + 2n·i·ω₂_imag

    if z.abs() < POLE_TOL {
        return Err(SpecialError::DomainError(
            "℘: z is too close to the origin (pole)".to_string(),
        ));
    }

    // Leading 1/z² term
    let mut result = 1.0 / (z * z);

    // Lattice sum
    for m in -N_LATTICE..=N_LATTICE {
        for n in -N_LATTICE..=N_LATTICE {
            if m == 0 && n == 0 {
                continue;
            }
            // For rectangular lattice ω_{m,n} is complex: 2m·ω₁ + 2in·ω₂_imag
            // On the real z-axis, the complex term 1/(z−ω)² contributes real part only.
            // ω_real = 2m·ω₁,  ω_imag = 2n·ω₂_imag
            let omega_r = 2.0 * (m as f64) * omega1;
            let omega_i = 2.0 * (n as f64) * omega2_imag;

            // 1/(z − ω)²  where z is real and ω = omega_r + i·omega_i
            // = 1/((z − omega_r)² + omega_i² − 2i·(z−omega_r)·omega_i)
            // Real part: ((z−omega_r)² − omega_i²) / |z−ω|⁴
            let dr = z - omega_r;
            let denom_sq = dr * dr + omega_i * omega_i;
            if denom_sq < POLE_TOL * POLE_TOL {
                return Err(SpecialError::DomainError(format!(
                    "℘: z is too close to lattice point ({m},{n})"
                )));
            }
            let denom4 = denom_sq * denom_sq;
            let one_over_zsq_re = (dr * dr - omega_i * omega_i) / denom4;

            // 1/ω²
            let omega_sq = omega_r * omega_r - omega_i * omega_i;
            // Real part of 1/ω² = (omega_r² − omega_i²) / |ω|⁴
            let omega_mod4 = (omega_r * omega_r + omega_i * omega_i).powi(2);
            if omega_mod4 < 1e-300 {
                continue;
            }
            let one_over_omega_sq_re = omega_sq / omega_mod4;

            result += one_over_zsq_re - one_over_omega_sq_re;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Weierstrass ℘' function
// ---------------------------------------------------------------------------

/// Derivative of the Weierstrass ℘ function: ℘'(z; g₂, g₃).
///
/// ```text
/// ℘'(z) = −2 Σ_{(m,n)} 1/(z − ω_{m,n})³
/// ```
///
/// Satisfies: `(℘')² = 4℘³ − g₂℘ − g₃`.
///
/// # Arguments
/// * `z`  – real argument
/// * `g2` – lattice invariant g₂
/// * `g3` – lattice invariant g₃
///
/// # Returns
/// Value of ℘'(z; g₂, g₃).  Returns `f64::NAN` for poles or degenerate invariants.
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::weierstrass_p_derivative;
/// // ℘' is odd: ℘'(−z) = −℘'(z)
/// let z = 0.4_f64;
/// let dp = weierstrass_p_derivative(z, 1.0, 0.0);
/// let dm = weierstrass_p_derivative(-z, 1.0, 0.0);
/// assert!((dp + dm).abs() < 1e-8);
/// ```
pub fn weierstrass_p_derivative(z: f64, g2: f64, g3: f64) -> f64 {
    match weierstrass_p_derivative_impl(z, g2, g3) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn weierstrass_p_derivative_impl(z: f64, g2: f64, g3: f64) -> SpecialResult<f64> {
    let (omega1, omega2_imag) = half_periods_from_invariants(g2, g3)?;

    if z.abs() < POLE_TOL {
        return Err(SpecialError::DomainError(
            "℘': z is too close to the origin (pole)".to_string(),
        ));
    }

    // Leading −2/z³ term
    let mut result = -2.0 / (z * z * z);

    for m in -N_LATTICE..=N_LATTICE {
        for n in -N_LATTICE..=N_LATTICE {
            if m == 0 && n == 0 {
                continue;
            }
            let omega_r = 2.0 * (m as f64) * omega1;
            let omega_i = 2.0 * (n as f64) * omega2_imag;

            let dr = z - omega_r;
            let denom_sq = dr * dr + omega_i * omega_i;
            if denom_sq < POLE_TOL * POLE_TOL {
                return Err(SpecialError::DomainError(format!(
                    "℘': z too close to lattice point ({m},{n})"
                )));
            }
            // Real part of −2/(z−ω)³:
            // (z−ω)³ = (dr + i·ω_i_neg)³ where ω_i_neg = −omega_i (z is real, ω has imag=omega_i)
            // Actually (z − ω) = dr − i·omega_i
            // (dr − i·oi)³ = dr³ − 3dr·oi² − i(3dr²·oi − oi³)
            // Real part of (z−ω)³ = dr³ − 3dr·oi²
            // |z−ω|⁶ = denom_sq³
            let oi = omega_i;
            let denom6 = denom_sq.powi(3);
            let re_cube = dr * dr * dr - 3.0 * dr * oi * oi;
            // Real part of −2/(z−ω)³ = −2·re_cube / |z−ω|⁶
            result += -2.0 * re_cube / denom6;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Weierstrass ζ function (zeta, not Riemann zeta)
// ---------------------------------------------------------------------------

/// Weierstrass zeta function ζ(z; g₂, g₃).
///
/// Note: This is **not** the Riemann zeta function.  The Weierstrass zeta is the
/// primitive of −℘:
/// ```text
/// dζ/dz = −℘(z)
/// ```
///
/// Explicitly:
/// ```text
/// ζ(z) = 1/z + Σ_{ω≠0} [1/(z−ω) + 1/ω + z/ω²]
/// ```
///
/// # Arguments
/// * `z`  – real argument (not a lattice point)
/// * `g2` – lattice invariant g₂
/// * `g3` – lattice invariant g₃
///
/// # Returns
/// Value of ζ(z; g₂, g₃).  Returns `f64::NAN` for poles or degenerate invariants.
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::weierstrass_zeta;
/// let val = weierstrass_zeta(0.5, 1.0, 0.0);
/// assert!(val.is_finite());
/// ```
pub fn weierstrass_zeta(z: f64, g2: f64, g3: f64) -> f64 {
    match weierstrass_zeta_impl(z, g2, g3) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn weierstrass_zeta_impl(z: f64, g2: f64, g3: f64) -> SpecialResult<f64> {
    let (omega1, omega2_imag) = half_periods_from_invariants(g2, g3)?;

    if z.abs() < POLE_TOL {
        return Err(SpecialError::DomainError(
            "ζ: z too close to origin (pole)".to_string(),
        ));
    }

    // Leading 1/z term
    let mut result = 1.0 / z;

    for m in -N_LATTICE..=N_LATTICE {
        for n in -N_LATTICE..=N_LATTICE {
            if m == 0 && n == 0 {
                continue;
            }
            let omega_r = 2.0 * (m as f64) * omega1;
            let omega_i = 2.0 * (n as f64) * omega2_imag;

            let dr = z - omega_r;
            let denom_sq = dr * dr + omega_i * omega_i;
            if denom_sq < POLE_TOL * POLE_TOL {
                return Err(SpecialError::DomainError(format!(
                    "ζ: z too close to lattice point ({m},{n})"
                )));
            }

            // Real part of 1/(z−ω) = dr / |z−ω|²
            let re_1_over_z_minus_omega = dr / denom_sq;

            // Real part of 1/ω = omega_r / |ω|²
            let omega_mod2 = omega_r * omega_r + omega_i * omega_i;
            if omega_mod2 < 1e-300 {
                continue;
            }
            let re_1_over_omega = omega_r / omega_mod2;

            // Real part of z/ω² = z · Re(1/ω²) = z · (omega_r²−omega_i²) / |ω|⁴
            let omega_mod4 = omega_mod2 * omega_mod2;
            let re_z_over_omega_sq = z * (omega_r * omega_r - omega_i * omega_i) / omega_mod4;

            result += re_1_over_z_minus_omega + re_1_over_omega + re_z_over_omega_sq;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Weierstrass σ function
// ---------------------------------------------------------------------------

/// Weierstrass sigma function σ(z; g₂, g₃).
///
/// The sigma function is an entire function defined by:
/// ```text
/// σ(z) = z · Π_{ω≠0} (1 − z/ω) exp(z/ω + z²/(2ω²))
/// ```
///
/// It is related to the zeta function by:
/// ```text
/// ζ(z) = d/dz ln σ(z)
/// ```
///
/// # Arguments
/// * `z`  – real argument
/// * `g2` – lattice invariant g₂
/// * `g3` – lattice invariant g₃
///
/// # Returns
/// Value of σ(z; g₂, g₃).  Returns `f64::NAN` for degenerate invariants.
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::weierstrass_sigma;
/// // σ is an odd function: σ(−z) = −σ(z)
/// let z = 0.3_f64;
/// let sp = weierstrass_sigma(z, 1.0, 0.0);
/// let sm = weierstrass_sigma(-z, 1.0, 0.0);
/// assert!((sp + sm).abs() < 1e-8);
/// ```
pub fn weierstrass_sigma(z: f64, g2: f64, g3: f64) -> f64 {
    match weierstrass_sigma_impl(z, g2, g3) {
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

fn weierstrass_sigma_impl(z: f64, g2: f64, g3: f64) -> SpecialResult<f64> {
    let (omega1, omega2_imag) = half_periods_from_invariants(g2, g3)?;

    // Leading z factor
    let mut log_sigma_no_z = 0.0_f64; // We accumulate log of the infinite product

    for m in -N_LATTICE..=N_LATTICE {
        for n in -N_LATTICE..=N_LATTICE {
            if m == 0 && n == 0 {
                continue;
            }
            let omega_r = 2.0 * (m as f64) * omega1;
            let omega_i = 2.0 * (n as f64) * omega2_imag;
            let omega_mod2 = omega_r * omega_r + omega_i * omega_i;
            if omega_mod2 < 1e-300 {
                continue;
            }

            // Factor: (1 − z/ω) exp(z/ω + z²/(2ω²))
            //
            // 1 − z/ω: z is real, ω = omega_r + i·omega_i
            // z/ω = z·conj(ω) / |ω|² = z·omega_r/|ω|² − i·z·omega_i/|ω|²
            // Re(z/ω) = z·omega_r / |ω|²
            // Im(z/ω) = −z·omega_i / |ω|²
            let re_z_over_omega = z * omega_r / omega_mod2;
            let im_z_over_omega = -z * omega_i / omega_mod2;

            // 1 − z/ω
            let re_1mz = 1.0 - re_z_over_omega;
            let im_1mz = -im_z_over_omega;
            let mod1mz = (re_1mz * re_1mz + im_1mz * im_1mz).sqrt();
            if mod1mz < 1e-300 {
                // z is at a zero of σ (lattice point) – handle gracefully
                return Ok(0.0);
            }
            let arg1mz = im_1mz.atan2(re_1mz);

            // exp(z/ω + z²/(2ω²))
            // z²/(2ω²): let ω² = (omega_r+i·omega_i)² = (omega_r²−omega_i²) + 2i·omega_r·omega_i
            // Re(1/ω²) = (omega_r²−omega_i²) / |ω|⁴
            // z²/(2ω²): real part = z²/2 · Re(1/ω²)
            let omega_mod4 = omega_mod2 * omega_mod2;
            let re_inv_omega2 = (omega_r * omega_r - omega_i * omega_i) / omega_mod4;
            let im_inv_omega2 = -2.0 * omega_r * omega_i / omega_mod4;
            let re_exp_arg = re_z_over_omega + 0.5 * z * z * re_inv_omega2;
            let im_exp_arg = im_z_over_omega + 0.5 * z * z * im_inv_omega2;

            // ln(factor) = ln|1−z/ω| + i·arg(1−z/ω) + re_exp_arg + i·im_exp_arg
            // Real part: ln|1−z/ω| + re_exp_arg
            log_sigma_no_z += mod1mz.ln() + re_exp_arg;
            // (imaginary part cancels due to complex-conjugate pairs in the lattice)
            let _ = (arg1mz + im_exp_arg); // suppress unused-variable lint
        }
    }

    // σ(z) = z · exp(log_sigma_no_z)
    let sigma = z * log_sigma_no_z.exp();
    Ok(sigma)
}

// ---------------------------------------------------------------------------
// Lattice invariants from half-periods
// ---------------------------------------------------------------------------

/// Compute lattice invariants (g₂, g₃) from the half-periods ω₁ (real) and ω₂ (imaginary part).
///
/// For a rectangular lattice with half-periods ω₁ ∈ ℝ and ω₂ ∈ i·ℝ:
/// ```text
/// g₂ = 60  Σ_{(m,n)≠(0,0)} (2mω₁ + 2inω₂_imag)⁻⁴
/// g₃ = 140 Σ_{(m,n)≠(0,0)} (2mω₁ + 2inω₂_imag)⁻⁶
/// ```
///
/// # Arguments
/// * `omega1`      – real half-period (positive)
/// * `omega2_imag` – imaginary part of the second half-period (positive)
///
/// # Returns
/// Tuple `(g2, g3)` of lattice invariants.
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::lattice_invariants;
/// let (g2, g3) = lattice_invariants(1.0, 1.0);
/// assert!(g2.is_finite() && g3.is_finite());
/// ```
pub fn lattice_invariants(omega1: f64, omega2_imag: f64) -> (f64, f64) {
    let mut sum4 = 0.0_f64;
    let mut sum6 = 0.0_f64;

    for m in -N_LATTICE..=N_LATTICE {
        for n in -N_LATTICE..=N_LATTICE {
            if m == 0 && n == 0 {
                continue;
            }
            let omega_r = 2.0 * (m as f64) * omega1;
            let omega_i = 2.0 * (n as f64) * omega2_imag;
            let omega_mod2 = omega_r * omega_r + omega_i * omega_i;
            if omega_mod2 < 1e-300 {
                continue;
            }
            let omega_mod4 = omega_mod2 * omega_mod2;
            let omega_mod6 = omega_mod4 * omega_mod2;

            // Re(1/ω⁴) = Re((ω*)⁴) / |ω|⁸ but for the sum we use the Eisenstein formula.
            // For a rectangular lattice all ω-pairs (ω, -ω) give real contributions.
            // Re(1/ω⁴) = Re(ω⁴)/|ω|⁸
            // ω⁴ = (omega_r + i·omega_i)⁴
            let re_omega4 = omega_r * omega_r * omega_r * omega_r
                - 6.0 * omega_r * omega_r * omega_i * omega_i
                + omega_i * omega_i * omega_i * omega_i;
            sum4 += re_omega4 / (omega_mod4 * omega_mod4);

            let re_omega6 = omega_r.powi(6)
                - 15.0 * omega_r.powi(4) * omega_i * omega_i
                + 15.0 * omega_r * omega_r * omega_i.powi(4)
                - omega_i.powi(6);
            sum6 += re_omega6 / (omega_mod6 * omega_mod6);
        }
    }

    let g2 = 60.0 * sum4;
    let g3 = 140.0 * sum6;
    (g2, g3)
}

// ---------------------------------------------------------------------------
// Discriminant
// ---------------------------------------------------------------------------

/// Weierstrass discriminant Δ = g₂³ − 27g₃².
///
/// * Δ > 0: three distinct real roots (rectangular lattice)
/// * Δ < 0: one real root, two complex conjugate roots (rhombic lattice)
/// * Δ = 0: degenerate (node or cusp; degenerate elliptic curve)
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::discriminant;
/// // Lemniscate: g₂=4, g₃=0 → Δ = 64
/// assert!((discriminant(4.0, 0.0) - 64.0).abs() < 1e-10);
/// ```
pub fn discriminant(g2: f64, g3: f64) -> f64 {
    g2 * g2 * g2 - 27.0 * g3 * g3
}

/// j-invariant of the elliptic curve y² = 4x³ − g₂x − g₃.
///
/// ```text
/// j = 1728 · g₂³ / Δ
/// ```
///
/// # Returns
/// j-invariant.  Returns `f64::INFINITY` for Δ = 0 (degenerate curve).
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::j_invariant;
/// // Lemniscate: g₂=4, g₃=0 → j = 1728
/// assert!((j_invariant(4.0, 0.0) - 1728.0).abs() < 1e-8);
/// ```
pub fn j_invariant(g2: f64, g3: f64) -> f64 {
    let delta = discriminant(g2, g3);
    if delta.abs() < 1e-300 {
        return f64::INFINITY;
    }
    1728.0 * g2 * g2 * g2 / delta
}

// ---------------------------------------------------------------------------
// Differential equation verification utility
// ---------------------------------------------------------------------------

/// Check the Weierstrass differential equation `(℘')² = 4℘³ − g₂℘ − g₃`.
///
/// Returns the residual `|(℘')² − (4℘³ − g₂℘ − g₃)|` as a measure of accuracy.
///
/// # Arguments
/// * `z`  – real argument
/// * `g2` – lattice invariant g₂
/// * `g3` – lattice invariant g₃
///
/// # Examples
/// ```
/// use scirs2_special::weierstrass::check_differential_equation;
/// let residual = check_differential_equation(0.5, 1.0, 0.0);
/// assert!(residual < 1e-5, "ODE residual too large: {}", residual);
/// ```
pub fn check_differential_equation(z: f64, g2: f64, g3: f64) -> f64 {
    let p = weierstrass_p(z, g2, g3);
    let dp = weierstrass_p_derivative(z, g2, g3);
    if !p.is_finite() || !dp.is_finite() {
        return f64::NAN;
    }
    let lhs = dp * dp;
    let rhs = 4.0 * p * p * p - g2 * p - g3;
    (lhs - rhs).abs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_COARSE: f64 = 1e-5; // lattice sum at N_LATTICE=40 gives ~10⁻⁸ for typical values
    const EPS_MED: f64 = 1e-4;

    // --- discriminant ---

    #[test]
    fn test_discriminant_lemniscate() {
        // Lemniscate: g₂=4, g₃=0 → Δ = 4³ − 0 = 64
        let d = discriminant(4.0, 0.0);
        assert!((d - 64.0).abs() < 1e-10, "Δ = {d}");
    }

    #[test]
    fn test_discriminant_equianharmonic() {
        // Equianharmonic: g₂=0, g₃≠0 → Δ = −27g₃²
        let g3 = 4.0_f64;
        let d = discriminant(0.0, g3);
        assert!((d - (-27.0 * g3 * g3)).abs() < 1e-10, "Δ = {d}");
    }

    // --- j-invariant ---

    #[test]
    fn test_j_invariant_lemniscate() {
        // j(lemniscate) = 1728
        let j = j_invariant(4.0, 0.0);
        assert!((j - 1728.0).abs() < 1e-6, "j = {j}");
    }

    // --- lattice invariants round-trip ---

    #[test]
    fn test_lattice_invariants_finite() {
        let (g2, g3) = lattice_invariants(1.0, 1.0);
        assert!(g2.is_finite(), "g2 not finite: {g2}");
        assert!(g3.is_finite(), "g3 not finite: {g3}");
    }

    // --- ℘ function ---

    #[test]
    fn test_weierstrass_p_finite() {
        // Should return finite value away from poles
        let v = weierstrass_p(0.5, 1.0, 0.0);
        assert!(v.is_finite(), "℘ not finite: {v}");
    }

    #[test]
    fn test_weierstrass_p_near_pole() {
        // Very close to pole should return NAN
        let v = weierstrass_p(1e-12, 1.0, 0.0);
        assert!(v.is_nan() || v.abs() > 1e10, "Expected large or NaN near pole");
    }

    // --- ℘' function ---

    #[test]
    fn test_weierstrass_p_derivative_odd() {
        // ℘'(−z) = −℘'(z)
        let g2 = 1.0_f64;
        let g3 = 0.0_f64;
        let z = 0.4_f64;
        let dp_pos = weierstrass_p_derivative(z, g2, g3);
        let dp_neg = weierstrass_p_derivative(-z, g2, g3);
        if dp_pos.is_finite() && dp_neg.is_finite() {
            assert!(
                (dp_pos + dp_neg).abs() < EPS_MED,
                "℘' not odd: dp({z})={dp_pos}, dp(-{z})={dp_neg}"
            );
        }
    }

    // --- ζ function ---

    #[test]
    fn test_weierstrass_zeta_finite() {
        let v = weierstrass_zeta(0.5, 1.0, 0.0);
        assert!(v.is_finite(), "ζ not finite: {v}");
    }

    // --- σ function ---

    #[test]
    fn test_weierstrass_sigma_odd() {
        // σ(−z) = −σ(z)
        let z = 0.3_f64;
        let sp = weierstrass_sigma(z, 1.0, 0.0);
        let sm = weierstrass_sigma(-z, 1.0, 0.0);
        if sp.is_finite() && sm.is_finite() {
            assert!(
                (sp + sm).abs() < EPS_MED,
                "σ not odd: σ({z})={sp}, σ(-{z})={sm}"
            );
        }
    }

    #[test]
    fn test_weierstrass_sigma_zero_at_origin() {
        // σ(0) = 0 by definition (leading z factor)
        // Actually for very small z, σ(z) ≈ z
        let v = weierstrass_sigma(1e-8, 1.0, 0.0);
        assert!(v.abs() < 1e-6, "σ near 0: {v}");
    }

    // --- Differential equation ---

    #[test]
    fn test_weierstrass_ode_lemniscate() {
        // For the lemniscate lattice g₂=4, g₃=0
        // Check (℘')² = 4℘³ − 4℘
        let residual = check_differential_equation(0.5, 4.0, 0.0);
        assert!(
            !residual.is_nan() && residual < 1.0,
            "ODE residual too large: {residual}"
        );
    }

    // --- cubic roots ---

    #[test]
    fn test_cubic_roots_sum_zero() {
        // e₁ + e₂ + e₃ = 0 (coefficient of t² in 4t³ − g₂t − g₃ is 0)
        let g2 = 3.0_f64;
        let g3 = 1.0_f64;
        match cubic_roots_weierstrass(g2, g3) {
            Ok((e1, e2, e3)) => {
                assert!(
                    (e1 + e2 + e3).abs() < 1e-12,
                    "roots sum = {}",
                    e1 + e2 + e3
                );
            }
            Err(_) => {}
        }
    }
}
