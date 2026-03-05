//! Enhanced Transfer Function Analysis
//!
//! Comprehensive analysis tools for LTI transfer functions including:
//!
//! - **Pole-zero analysis**: Extract and classify poles and zeros
//! - **Root locus**: Trace pole trajectories as gain varies
//! - **Nyquist diagram**: Frequency response in polar form
//! - **Nichols chart**: Frequency response in dB-degree form
//! - **Stability margins**: Gain margin and phase margin computation
//! - **Sensitivity functions**: S(s), T(s), KS(s), CS(s)
//!
//! All functions return data suitable for plotting rather than
//! performing the plotting themselves.

use crate::error::{SignalError, SignalResult};
use crate::lti::systems::{LtiSystem, TransferFunction};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ============================================================================
// Pole-Zero Analysis
// ============================================================================

/// Result of pole-zero analysis
#[derive(Debug, Clone)]
pub struct PoleZeroResult {
    /// Poles of the system (roots of denominator)
    pub poles: Vec<Complex64>,
    /// Zeros of the system (roots of numerator)
    pub zeros: Vec<Complex64>,
    /// DC gain (H(0) for continuous, H(1) for discrete)
    pub dc_gain: f64,
    /// Whether the system is stable
    pub is_stable: bool,
    /// Whether the system is minimum phase (all zeros in LHP / inside unit circle)
    pub is_minimum_phase: bool,
    /// Natural frequencies of poles (rad/s)
    pub natural_frequencies: Vec<f64>,
    /// Damping ratios of poles
    pub damping_ratios: Vec<f64>,
}

/// Analyze poles and zeros of a transfer function
///
/// Computes poles, zeros, stability, minimum-phase property,
/// natural frequencies and damping ratios.
///
/// # Arguments
/// * `tf` - Transfer function to analyze
///
/// # Returns
/// * `PoleZeroResult` with comprehensive analysis
///
/// # Example
/// ```rust
/// use scirs2_signal::lti::systems::TransferFunction;
/// use scirs2_signal::tf_analysis::pole_zero_analysis;
///
/// // Second-order system: 1 / (s^2 + 2*0.7*s + 1)
/// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.4, 1.0], None)
///     .expect("TF creation failed");
/// let pz = pole_zero_analysis(&tf).expect("Analysis failed");
/// assert!(pz.is_stable);
/// assert_eq!(pz.poles.len(), 2);
/// ```
pub fn pole_zero_analysis(tf: &TransferFunction) -> SignalResult<PoleZeroResult> {
    let poles = find_polynomial_roots(&tf.den)?;
    let zeros = find_polynomial_roots(&tf.num)?;

    // Stability check
    let is_stable = if tf.dt {
        poles.iter().all(|p| p.norm() < 1.0 - 1e-10)
    } else {
        poles.iter().all(|p| p.re < 1e-10)
    };

    // Minimum phase check
    let is_minimum_phase = if tf.dt {
        zeros.iter().all(|z| z.norm() < 1.0 - 1e-10)
    } else {
        zeros.iter().all(|z| z.re < 1e-10)
    };

    // DC gain
    let dc_gain = if tf.dt {
        // H(z=1)
        let z1 = Complex64::new(1.0, 0.0);
        tf.evaluate(z1).re
    } else {
        // H(s=0)
        let s0 = Complex64::new(0.0, 0.0);
        let val = tf.evaluate(s0);
        if val.norm() < 1e30 {
            val.re
        } else {
            f64::INFINITY
        }
    };

    // Natural frequencies and damping ratios for continuous-time
    let mut natural_frequencies = Vec::with_capacity(poles.len());
    let mut damping_ratios = Vec::with_capacity(poles.len());

    for p in &poles {
        if tf.dt {
            // For discrete time: convert pole to continuous equivalent
            // p = exp(s*T), approximate omega_n and zeta
            let mag = p.norm();
            let angle = p.arg();
            natural_frequencies.push(angle.abs()); // normalized frequency
            damping_ratios.push(if mag > 1e-15 {
                -mag.ln() / (angle.powi(2) + mag.ln().powi(2)).sqrt()
            } else {
                1.0
            });
        } else {
            // omega_n = |p|, zeta = -Re(p)/|p|
            let omega_n = p.norm();
            natural_frequencies.push(omega_n);
            if omega_n > 1e-15 {
                damping_ratios.push(-p.re / omega_n);
            } else {
                damping_ratios.push(0.0);
            }
        }
    }

    Ok(PoleZeroResult {
        poles,
        zeros,
        dc_gain,
        is_stable,
        is_minimum_phase,
        natural_frequencies,
        damping_ratios,
    })
}

// ============================================================================
// Root Locus
// ============================================================================

/// Data point on the root locus
#[derive(Debug, Clone)]
pub struct RootLocusPoint {
    /// Gain value
    pub gain: f64,
    /// Pole locations at this gain
    pub poles: Vec<Complex64>,
}

/// Result of root locus computation
#[derive(Debug, Clone)]
pub struct RootLocusResult {
    /// Root locus data points (gain, poles) for each gain value
    pub points: Vec<RootLocusPoint>,
    /// Open-loop poles (gain = 0)
    pub open_loop_poles: Vec<Complex64>,
    /// Open-loop zeros (gain = infinity asymptotes)
    pub open_loop_zeros: Vec<Complex64>,
    /// Breakaway/break-in points on the real axis (approximate)
    pub breakaway_points: Vec<f64>,
}

/// Compute root locus data for a transfer function
///
/// Traces the closed-loop pole locations as the loop gain K varies
/// from 0 to `max_gain`. The characteristic equation is:
///   1 + K * G(s) = 0
///   den(s) + K * num(s) = 0
///
/// # Arguments
/// * `tf` - Open-loop transfer function G(s)
/// * `gains` - Gain values at which to compute pole locations
///
/// # Returns
/// * `RootLocusResult` with pole trajectories
///
/// # Example
/// ```rust
/// use scirs2_signal::lti::systems::TransferFunction;
/// use scirs2_signal::tf_analysis::root_locus;
///
/// let tf = TransferFunction::new(vec![1.0], vec![1.0, 3.0, 2.0], None)
///     .expect("TF creation failed");
/// let gains: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
/// let rl = root_locus(&tf, &gains).expect("Root locus failed");
/// assert_eq!(rl.points.len(), 100);
/// ```
pub fn root_locus(tf: &TransferFunction, gains: &[f64]) -> SignalResult<RootLocusResult> {
    let open_loop_poles = find_polynomial_roots(&tf.den)?;
    let open_loop_zeros = find_polynomial_roots(&tf.num)?;

    let mut points = Vec::with_capacity(gains.len());

    for &k in gains {
        // Characteristic polynomial: den + K * num
        // Pad to same length
        let max_len = tf.den.len().max(tf.num.len());
        let mut char_poly = vec![0.0; max_len];

        // Add denominator (aligned to highest power)
        let den_offset = max_len - tf.den.len();
        for (i, &c) in tf.den.iter().enumerate() {
            char_poly[den_offset + i] += c;
        }

        // Add K * numerator
        let num_offset = max_len - tf.num.len();
        for (i, &c) in tf.num.iter().enumerate() {
            char_poly[num_offset + i] += k * c;
        }

        let poles = find_polynomial_roots(&char_poly)?;
        points.push(RootLocusPoint { gain: k, poles });
    }

    // Find approximate breakaway points (where dpoly/ds = 0 on real axis)
    let breakaway_points = find_breakaway_points(&tf.num, &tf.den);

    Ok(RootLocusResult {
        points,
        open_loop_poles,
        open_loop_zeros,
        breakaway_points,
    })
}

// ============================================================================
// Nyquist Diagram
// ============================================================================

/// Data for Nyquist diagram
#[derive(Debug, Clone)]
pub struct NyquistResult {
    /// Frequencies (rad/s)
    pub frequencies: Vec<f64>,
    /// Real part of G(jw)
    pub real_parts: Vec<f64>,
    /// Imaginary part of G(jw)
    pub imag_parts: Vec<f64>,
    /// Magnitude |G(jw)|
    pub magnitudes: Vec<f64>,
    /// Phase angle(G(jw)) in radians
    pub phases: Vec<f64>,
    /// Number of encirclements of -1+j0 (approximate)
    pub encirclements: i32,
}

/// Compute Nyquist diagram data for a transfer function
///
/// Evaluates G(jw) for positive frequencies and returns real/imaginary
/// parts for plotting the Nyquist contour. Also estimates the number
/// of encirclements of the critical point -1+j0.
///
/// # Arguments
/// * `tf` - Transfer function
/// * `frequencies` - Frequency points (rad/s), or None for auto-generation
///
/// # Returns
/// * `NyquistResult` with frequency response data
///
/// # Example
/// ```rust
/// use scirs2_signal::lti::systems::TransferFunction;
/// use scirs2_signal::tf_analysis::nyquist_diagram;
///
/// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None)
///     .expect("TF creation failed");
/// let result = nyquist_diagram(&tf, None).expect("Nyquist failed");
/// assert!(!result.frequencies.is_empty());
/// ```
pub fn nyquist_diagram(
    tf: &TransferFunction,
    frequencies: Option<&[f64]>,
) -> SignalResult<NyquistResult> {
    let freqs = match frequencies {
        Some(f) => f.to_vec(),
        None => generate_log_frequencies(1e-3, 1e3, 500),
    };

    let response = tf.frequency_response(&freqs)?;

    let mut real_parts = Vec::with_capacity(freqs.len());
    let mut imag_parts = Vec::with_capacity(freqs.len());
    let mut magnitudes = Vec::with_capacity(freqs.len());
    let mut phases = Vec::with_capacity(freqs.len());

    for &h in &response {
        real_parts.push(h.re);
        imag_parts.push(h.im);
        magnitudes.push(h.norm());
        phases.push(h.arg());
    }

    // Estimate encirclements of -1+j0 using winding number
    let encirclements = estimate_encirclements(&real_parts, &imag_parts);

    Ok(NyquistResult {
        frequencies: freqs,
        real_parts,
        imag_parts,
        magnitudes,
        phases,
        encirclements,
    })
}

// ============================================================================
// Nichols Chart
// ============================================================================

/// Data for Nichols chart (dB vs degrees)
#[derive(Debug, Clone)]
pub struct NicholsResult {
    /// Frequencies (rad/s)
    pub frequencies: Vec<f64>,
    /// Open-loop magnitude in dB
    pub magnitude_db: Vec<f64>,
    /// Open-loop phase in degrees
    pub phase_deg: Vec<f64>,
}

/// Compute Nichols chart data for a transfer function
///
/// Returns open-loop magnitude (dB) vs phase (degrees) for plotting.
///
/// # Arguments
/// * `tf` - Open-loop transfer function
/// * `frequencies` - Frequency points (rad/s), or None for auto
///
/// # Returns
/// * `NicholsResult` with mag/phase data
pub fn nichols_chart(
    tf: &TransferFunction,
    frequencies: Option<&[f64]>,
) -> SignalResult<NicholsResult> {
    let freqs = match frequencies {
        Some(f) => f.to_vec(),
        None => generate_log_frequencies(1e-3, 1e3, 500),
    };

    let response = tf.frequency_response(&freqs)?;

    let mut magnitude_db = Vec::with_capacity(freqs.len());
    let mut phase_deg = Vec::with_capacity(freqs.len());

    for &h in &response {
        let mag = h.norm();
        magnitude_db.push(if mag > 1e-30 {
            20.0 * mag.log10()
        } else {
            -600.0
        });
        phase_deg.push(h.arg() * 180.0 / PI);
    }

    Ok(NicholsResult {
        frequencies: freqs,
        magnitude_db,
        phase_deg,
    })
}

// ============================================================================
// Stability Margins
// ============================================================================

/// Stability margin results
#[derive(Debug, Clone)]
pub struct StabilityMargins {
    /// Gain margin in dB (positive means stable)
    pub gain_margin_db: f64,
    /// Phase margin in degrees (positive means stable)
    pub phase_margin_deg: f64,
    /// Frequency at which phase = -180 deg (gain crossover)
    pub phase_crossover_freq: f64,
    /// Frequency at which |G(jw)| = 1 (0 dB, gain crossover)
    pub gain_crossover_freq: f64,
    /// Delay margin in seconds (phase_margin / gain_crossover_freq)
    pub delay_margin: f64,
}

/// Compute gain and phase margins
///
/// - **Gain margin**: How much the gain can increase before instability
/// - **Phase margin**: How much additional phase lag before instability
///
/// # Arguments
/// * `tf` - Open-loop transfer function
/// * `frequencies` - Frequency points, or None for auto
///
/// # Returns
/// * `StabilityMargins`
///
/// # Example
/// ```rust
/// use scirs2_signal::lti::systems::TransferFunction;
/// use scirs2_signal::tf_analysis::stability_margins;
///
/// // First-order system: 10 / (s+1)
/// let tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None)
///     .expect("TF creation failed");
/// let margins = stability_margins(&tf, None).expect("Margins failed");
/// assert!(margins.phase_margin_deg > 0.0); // stable
/// ```
pub fn stability_margins(
    tf: &TransferFunction,
    frequencies: Option<&[f64]>,
) -> SignalResult<StabilityMargins> {
    let freqs = match frequencies {
        Some(f) => f.to_vec(),
        None => generate_log_frequencies(1e-4, 1e4, 2000),
    };

    let response = tf.frequency_response(&freqs)?;

    // Find gain crossover frequency (|G(jw)| = 1, i.e., 0 dB)
    let mut gain_crossover_freq = f64::NAN;
    let mut phase_at_gc = f64::NAN;

    for i in 1..response.len() {
        let mag_prev = response[i - 1].norm();
        let mag_curr = response[i].norm();

        // Check for 0 dB crossing (magnitude crosses 1.0)
        if (mag_prev - 1.0) * (mag_curr - 1.0) < 0.0 {
            // Linear interpolation
            let t = (1.0 - mag_prev) / (mag_curr - mag_prev);
            gain_crossover_freq = freqs[i - 1] + t * (freqs[i] - freqs[i - 1]);

            let phase_prev = response[i - 1].arg() * 180.0 / PI;
            let phase_curr = response[i].arg() * 180.0 / PI;
            phase_at_gc = phase_prev + t * (phase_curr - phase_prev);
            break;
        }
    }

    // Phase margin = 180 + phase at gain crossover
    let phase_margin_deg = if phase_at_gc.is_finite() {
        180.0 + phase_at_gc
    } else {
        f64::INFINITY // No gain crossover found, infinite margin
    };

    // Find phase crossover frequency (phase = -180 deg)
    let mut phase_crossover_freq = f64::NAN;
    let mut mag_at_pc = f64::NAN;

    for i in 1..response.len() {
        let phase_prev = response[i - 1].arg() * 180.0 / PI;
        let phase_curr = response[i].arg() * 180.0 / PI;

        // Check for -180 degree crossing
        if (phase_prev + 180.0) * (phase_curr + 180.0) < 0.0 {
            let t = (-180.0 - phase_prev) / (phase_curr - phase_prev);
            phase_crossover_freq = freqs[i - 1] + t * (freqs[i] - freqs[i - 1]);

            let mag_prev = response[i - 1].norm();
            let mag_curr = response[i].norm();
            mag_at_pc = mag_prev + t * (mag_curr - mag_prev);
            break;
        }
    }

    // Gain margin = 1 / |G(jw)| at phase crossover, in dB
    let gain_margin_db = if mag_at_pc.is_finite() && mag_at_pc > 1e-30 {
        -20.0 * mag_at_pc.log10()
    } else {
        f64::INFINITY // No phase crossover, infinite gain margin
    };

    // Delay margin
    let delay_margin = if gain_crossover_freq.is_finite() && gain_crossover_freq > 1e-15 {
        phase_margin_deg.abs() * PI / (180.0 * gain_crossover_freq)
    } else {
        f64::INFINITY
    };

    Ok(StabilityMargins {
        gain_margin_db,
        phase_margin_deg,
        phase_crossover_freq,
        gain_crossover_freq,
        delay_margin,
    })
}

// ============================================================================
// Sensitivity Functions
// ============================================================================

/// Sensitivity function data at each frequency
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// Frequencies (rad/s)
    pub frequencies: Vec<f64>,
    /// S(jw) = 1/(1+G(jw)) magnitude in dB
    pub s_magnitude_db: Vec<f64>,
    /// T(jw) = G(jw)/(1+G(jw)) magnitude in dB
    pub t_magnitude_db: Vec<f64>,
    /// KS(jw) = K(jw)/(1+G(jw)) magnitude in dB (if controller provided)
    pub ks_magnitude_db: Option<Vec<f64>>,
    /// CS(jw) = G_plant(jw)/(1+G(jw)) magnitude in dB (if plant provided)
    pub cs_magnitude_db: Option<Vec<f64>>,
    /// Peak of |S(jw)| in dB (sensitivity peak, Ms)
    pub ms_db: f64,
    /// Peak of |T(jw)| in dB (complementary sensitivity peak, Mt)
    pub mt_db: f64,
    /// Bandwidth (frequency where |T| drops below -3dB)
    pub bandwidth: f64,
}

/// Compute sensitivity functions for a feedback system
///
/// Given open-loop transfer function L(s) = G(s)*K(s):
/// - S(s) = 1/(1+L(s))         — sensitivity function
/// - T(s) = L(s)/(1+L(s))      — complementary sensitivity
/// - KS(s) = K(s)*S(s)         — noise sensitivity (optional)
/// - CS(s) = G(s)*S(s)         — load sensitivity (optional)
///
/// # Arguments
/// * `loop_tf` - Open-loop transfer function L(s) = G(s)*K(s)
/// * `controller` - Optional controller K(s) for KS computation
/// * `plant` - Optional plant G(s) for CS computation
/// * `frequencies` - Optional frequency vector
///
/// # Returns
/// * `SensitivityResult`
pub fn sensitivity_functions(
    loop_tf: &TransferFunction,
    controller: Option<&TransferFunction>,
    plant: Option<&TransferFunction>,
    frequencies: Option<&[f64]>,
) -> SignalResult<SensitivityResult> {
    let freqs = match frequencies {
        Some(f) => f.to_vec(),
        None => generate_log_frequencies(1e-3, 1e3, 500),
    };

    let l_response = loop_tf.frequency_response(&freqs)?;

    let k_response = match controller {
        Some(k) => Some(k.frequency_response(&freqs)?),
        None => None,
    };
    let g_response = match plant {
        Some(g) => Some(g.frequency_response(&freqs)?),
        None => None,
    };

    let mut s_mag_db = Vec::with_capacity(freqs.len());
    let mut t_mag_db = Vec::with_capacity(freqs.len());
    let mut ks_mag_db = k_response.as_ref().map(|_| Vec::with_capacity(freqs.len()));
    let mut cs_mag_db = g_response.as_ref().map(|_| Vec::with_capacity(freqs.len()));

    let one = Complex64::new(1.0, 0.0);
    let mut ms = 0.0_f64;
    let mut mt = 0.0_f64;
    let mut bandwidth = f64::INFINITY;
    let mut bw_found = false;

    for (i, &l) in l_response.iter().enumerate() {
        let one_plus_l = one + l;
        let s = one / one_plus_l;
        let t = l / one_plus_l;

        let s_mag = s.norm();
        let t_mag = t.norm();

        let s_db = if s_mag > 1e-30 {
            20.0 * s_mag.log10()
        } else {
            -600.0
        };
        let t_db = if t_mag > 1e-30 {
            20.0 * t_mag.log10()
        } else {
            -600.0
        };

        s_mag_db.push(s_db);
        t_mag_db.push(t_db);

        ms = ms.max(s_mag);
        mt = mt.max(t_mag);

        // Bandwidth: where |T| drops below -3dB (0.707)
        if !bw_found && t_mag < 0.7071 && i > 0 {
            bandwidth = freqs[i];
            bw_found = true;
        }

        if let (Some(ref mut ks_vec), Some(ref k_resp)) = (&mut ks_mag_db, &k_response) {
            let ks = k_resp[i] * s;
            let ks_m = ks.norm();
            ks_vec.push(if ks_m > 1e-30 {
                20.0 * ks_m.log10()
            } else {
                -600.0
            });
        }

        if let (Some(ref mut cs_vec), Some(ref g_resp)) = (&mut cs_mag_db, &g_response) {
            let cs = g_resp[i] * s;
            let cs_m = cs.norm();
            cs_vec.push(if cs_m > 1e-30 {
                20.0 * cs_m.log10()
            } else {
                -600.0
            });
        }
    }

    let ms_db = if ms > 1e-30 {
        20.0 * ms.log10()
    } else {
        -600.0
    };
    let mt_db = if mt > 1e-30 {
        20.0 * mt.log10()
    } else {
        -600.0
    };

    Ok(SensitivityResult {
        frequencies: freqs,
        s_magnitude_db: s_mag_db,
        t_magnitude_db: t_mag_db,
        ks_magnitude_db: ks_mag_db,
        cs_magnitude_db: cs_mag_db,
        ms_db,
        mt_db,
        bandwidth,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Find roots of a polynomial using the companion matrix method
///
/// The polynomial is a_0 * x^n + a_1 * x^{n-1} + ... + a_n
/// (highest power first).
fn find_polynomial_roots(coeffs: &[f64]) -> SignalResult<Vec<Complex64>> {
    // Remove leading zeros
    let mut start_idx = 0;
    while start_idx < coeffs.len() && coeffs[start_idx].abs() < 1e-15 {
        start_idx += 1;
    }
    let c = &coeffs[start_idx..];

    if c.is_empty() || c.len() == 1 {
        return Ok(Vec::new());
    }

    let n = c.len() - 1; // degree

    if n == 0 {
        return Ok(Vec::new());
    }

    // Normalize by leading coefficient
    let lead = c[0];
    if lead.abs() < 1e-30 {
        return Err(SignalError::ComputationError(
            "Leading coefficient is zero".into(),
        ));
    }

    // Build companion matrix
    let mut companion = scirs2_core::ndarray::Array2::<f64>::zeros((n, n));

    // Sub-diagonal of 1s
    for i in 1..n {
        companion[[i, i - 1]] = 1.0;
    }

    // Last column = -coefficients (reversed)
    for i in 0..n {
        companion[[i, n - 1]] = -c[n - i] / lead;
    }

    // Compute eigenvalues of companion matrix
    match scirs2_linalg::eig(&companion.view(), None) {
        Ok((eigenvalues, _)) => Ok(eigenvalues.to_vec()),
        Err(_) => {
            // Fallback: for simple cases
            if n == 1 {
                Ok(vec![Complex64::new(-c[1] / c[0], 0.0)])
            } else if n == 2 {
                let a = c[0];
                let b = c[1];
                let cc = c[2];
                let disc = b * b - 4.0 * a * cc;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    Ok(vec![
                        Complex64::new((-b + sq) / (2.0 * a), 0.0),
                        Complex64::new((-b - sq) / (2.0 * a), 0.0),
                    ])
                } else {
                    let sq = (-disc).sqrt();
                    Ok(vec![
                        Complex64::new(-b / (2.0 * a), sq / (2.0 * a)),
                        Complex64::new(-b / (2.0 * a), -sq / (2.0 * a)),
                    ])
                }
            } else {
                Err(SignalError::ComputationError(
                    "Eigenvalue computation failed for polynomial root finding".into(),
                ))
            }
        }
    }
}

/// Generate logarithmically spaced frequencies
fn generate_log_frequencies(f_min: f64, f_max: f64, n: usize) -> Vec<f64> {
    if n == 0 || f_min <= 0.0 || f_max <= f_min {
        return Vec::new();
    }
    let log_min = f_min.ln();
    let log_max = f_max.ln();
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1).max(1) as f64;
            (log_min + t * (log_max - log_min)).exp()
        })
        .collect()
}

/// Find approximate breakaway points on the real axis for root locus
fn find_breakaway_points(num: &[f64], den: &[f64]) -> Vec<f64> {
    // Breakaway points satisfy: num'(s)*den(s) - num(s)*den'(s) = 0
    // This is a simplified search along the real axis
    let mut breakaway = Vec::new();

    // Evaluate K = -den(s)/num(s) along real axis and find local extrema
    let n_points = 1000;
    let s_min = -10.0;
    let s_max = 10.0;
    let ds = (s_max - s_min) / n_points as f64;

    let eval_k = |s: f64| -> Option<f64> {
        let n_val = eval_real_polynomial(num, s);
        if n_val.abs() < 1e-15 {
            return None;
        }
        let d_val = eval_real_polynomial(den, s);
        Some(-d_val / n_val)
    };

    let mut prev_k = eval_k(s_min);
    let mut prev_dk: Option<f64> = None;

    for i in 1..=n_points {
        let s = s_min + i as f64 * ds;
        let curr_k = eval_k(s);

        if let (Some(pk), Some(ck)) = (prev_k, curr_k) {
            let dk = ck - pk;
            if let Some(pdk) = prev_dk {
                // Sign change in derivative means extremum
                if pdk * dk < 0.0 && pk.is_finite() && ck.is_finite() {
                    breakaway.push(s - ds / 2.0);
                }
            }
            prev_dk = Some(dk);
        }
        prev_k = curr_k;
    }

    breakaway
}

/// Evaluate a polynomial at a real value
fn eval_real_polynomial(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for (i, &c) in coeffs.iter().enumerate() {
        let power = (coeffs.len() - 1 - i) as u32;
        result += c * x.powi(power as i32);
    }
    result
}

/// Estimate number of encirclements of -1+j0 using winding number
fn estimate_encirclements(real_parts: &[f64], imag_parts: &[f64]) -> i32 {
    if real_parts.len() < 2 {
        return 0;
    }

    // Shift to origin at -1+j0
    let shifted_re: Vec<f64> = real_parts.iter().map(|&r| r + 1.0).collect();
    let shifted_im: Vec<f64> = imag_parts.to_vec();

    // Compute winding number via angle accumulation
    let mut total_angle = 0.0;

    for i in 1..shifted_re.len() {
        let angle_prev = shifted_im[i - 1].atan2(shifted_re[i - 1]);
        let angle_curr = shifted_im[i].atan2(shifted_re[i]);

        let mut d_angle = angle_curr - angle_prev;

        // Handle wrapping
        if d_angle > PI {
            d_angle -= 2.0 * PI;
        } else if d_angle < -PI {
            d_angle += 2.0 * PI;
        }

        total_angle += d_angle;
    }

    // Include the negative frequency part (mirror)
    // For a real system, the Nyquist contour is symmetric
    let encirclements = (total_angle / PI).round() as i32;
    encirclements
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pole_zero_first_order() {
        // H(s) = 1/(s+1) -> pole at -1, no zeros
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("TF failed");
        let pz = pole_zero_analysis(&tf).expect("PZ failed");

        assert_eq!(pz.poles.len(), 1);
        assert_relative_eq!(pz.poles[0].re, -1.0, epsilon = 0.01);
        assert_relative_eq!(pz.poles[0].im, 0.0, epsilon = 0.01);
        assert!(pz.zeros.is_empty());
        assert!(pz.is_stable);
        assert!(pz.is_minimum_phase);
        assert_relative_eq!(pz.dc_gain, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_pole_zero_second_order() {
        // H(s) = 1/(s^2 + 1.4s + 1) -> zeta=0.7, wn=1
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.4, 1.0], None).expect("TF failed");
        let pz = pole_zero_analysis(&tf).expect("PZ failed");

        assert_eq!(pz.poles.len(), 2);
        assert!(pz.is_stable);
        // Natural frequency should be near 1.0
        for wn in &pz.natural_frequencies {
            assert_relative_eq!(*wn, 1.0, epsilon = 0.1);
        }
        // Damping ratio should be near 0.7
        for zeta in &pz.damping_ratios {
            assert_relative_eq!(*zeta, 0.7, epsilon = 0.1);
        }
    }

    #[test]
    fn test_pole_zero_unstable() {
        // H(s) = 1/(s-1) -> pole at +1, unstable
        let tf = TransferFunction::new(vec![1.0], vec![1.0, -1.0], None).expect("TF failed");
        let pz = pole_zero_analysis(&tf).expect("PZ failed");

        assert!(!pz.is_stable);
    }

    #[test]
    fn test_root_locus_basic() {
        // G(s) = 1/(s(s+2)) -> poles at 0 and -2
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 2.0, 0.0], None).expect("TF failed");
        let gains: Vec<f64> = (0..50).map(|i| i as f64 * 0.2).collect();
        let rl = root_locus(&tf, &gains).expect("RL failed");

        assert_eq!(rl.points.len(), 50);
        assert_eq!(rl.open_loop_poles.len(), 2);
    }

    #[test]
    fn test_nyquist_first_order() {
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("TF failed");
        let result = nyquist_diagram(&tf, None).expect("Nyquist failed");

        assert!(!result.frequencies.is_empty());
        assert_eq!(result.real_parts.len(), result.frequencies.len());
        assert_eq!(result.imag_parts.len(), result.frequencies.len());
    }

    #[test]
    fn test_nichols_chart_basic() {
        let tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).expect("TF failed");
        let result = nichols_chart(&tf, None).expect("Nichols failed");

        assert!(!result.frequencies.is_empty());
        assert_eq!(result.magnitude_db.len(), result.frequencies.len());
        assert_eq!(result.phase_deg.len(), result.frequencies.len());
    }

    #[test]
    fn test_stability_margins_stable() {
        // G(s) = 10/(s+1) -> stable with good margins
        let tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).expect("TF failed");
        let margins = stability_margins(&tf, None).expect("Margins failed");

        // First-order system has infinite gain margin (no -180 crossing)
        assert!(margins.gain_margin_db > 0.0 || margins.gain_margin_db.is_infinite());
        // Phase margin should be positive for stable system
        if margins.phase_margin_deg.is_finite() {
            assert!(margins.phase_margin_deg > 0.0);
        }
    }

    #[test]
    fn test_sensitivity_functions_basic() {
        // L(s) = 10/(s+1)
        let loop_tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).expect("TF failed");
        let result = sensitivity_functions(&loop_tf, None, None, None).expect("Sensitivity failed");

        assert!(!result.frequencies.is_empty());
        assert_eq!(result.s_magnitude_db.len(), result.frequencies.len());
        assert_eq!(result.t_magnitude_db.len(), result.frequencies.len());
        assert!(result.ks_magnitude_db.is_none());
        assert!(result.cs_magnitude_db.is_none());
        // For high loop gain, Ms should be reasonable
        assert!(result.ms_db.is_finite());
        assert!(result.mt_db.is_finite());
    }

    #[test]
    fn test_sensitivity_with_controller() {
        let controller = TransferFunction::new(vec![10.0], vec![1.0], None).expect("K failed");
        let plant = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).expect("G failed");
        // L = K*G = 10/(s+1)
        let loop_tf = TransferFunction::new(vec![10.0], vec![1.0, 1.0], None).expect("L failed");

        let result = sensitivity_functions(&loop_tf, Some(&controller), Some(&plant), None)
            .expect("Sensitivity failed");

        assert!(result.ks_magnitude_db.is_some());
        assert!(result.cs_magnitude_db.is_some());
    }

    #[test]
    fn test_generate_log_frequencies() {
        let freqs = generate_log_frequencies(0.01, 100.0, 50);
        assert_eq!(freqs.len(), 50);
        assert!(freqs[0] > 0.009);
        assert!(freqs[49] < 101.0);
        // Should be monotonically increasing
        for i in 1..freqs.len() {
            assert!(freqs[i] > freqs[i - 1]);
        }
    }

    #[test]
    fn test_polynomial_roots_quadratic() {
        // s^2 + 3s + 2 = (s+1)(s+2) -> roots at -1, -2
        let roots = find_polynomial_roots(&[1.0, 3.0, 2.0]).expect("Roots failed");
        assert_eq!(roots.len(), 2);

        let mut real_parts: Vec<f64> = roots.iter().map(|r| r.re).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_relative_eq!(real_parts[0], -2.0, epsilon = 0.01);
        assert_relative_eq!(real_parts[1], -1.0, epsilon = 0.01);
    }

    #[test]
    fn test_polynomial_roots_complex() {
        // s^2 + 2s + 5 -> roots at -1 +/- 2j
        let roots = find_polynomial_roots(&[1.0, 2.0, 5.0]).expect("Roots failed");
        assert_eq!(roots.len(), 2);

        for r in &roots {
            assert_relative_eq!(r.re, -1.0, epsilon = 0.01);
            assert_relative_eq!(r.im.abs(), 2.0, epsilon = 0.01);
        }
    }
}
