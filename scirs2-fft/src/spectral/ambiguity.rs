//! Ambiguity function computation.
//!
//! The ambiguity function characterises a signal's joint time-delay and
//! frequency-shift behaviour and is a fundamental tool in radar, sonar,
//! and communications signal design.
//!
//! # Narrowband ambiguity function
//!
//! ```text
//! A(tau, nu) = integral x(t + tau/2) * conj(x(t - tau/2)) * exp(-j 2 pi nu t) dt
//! ```
//!
//! # Wideband ambiguity function
//!
//! Uses time-scale (dilation) rather than frequency shift:
//! ```text
//! A_wb(tau, s) = sqrt(|s|) * integral x(t) * conj(x(s*(t - tau))) dt
//! ```
//!
//! # References
//!
//! * Woodward, P. M. "Probability and Information Theory, with Applications
//!   to Radar." Pergamon Press, 1953.
//! * Auger, F. & Flandrin, P. "Improving the readability of time-frequency
//!   and time-scale representations." IEEE Trans. SP, 43(5), 1995.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
//  Configuration
// ---------------------------------------------------------------------------

/// Configuration for ambiguity function computation.
#[derive(Debug, Clone)]
pub struct AmbiguityConfig {
    /// Maximum delay (tau) in samples (symmetric: -max_delay..+max_delay).
    pub max_delay: usize,
    /// Maximum Doppler shift in normalised frequency bins.
    pub max_doppler: usize,
    /// Sampling frequency (Hz). Default 1.0.
    pub fs: f64,
}

impl Default for AmbiguityConfig {
    fn default() -> Self {
        Self {
            max_delay: 64,
            max_doppler: 64,
            fs: 1.0,
        }
    }
}

/// Result of the narrowband ambiguity function.
#[derive(Debug, Clone)]
pub struct AmbiguityResult {
    /// Delay axis (seconds).
    pub delays: Vec<f64>,
    /// Doppler axis (Hz).
    pub dopplers: Vec<f64>,
    /// 2-D ambiguity surface `|A(tau, nu)|`: `surface[delay_idx][doppler_idx]`.
    pub surface: Vec<Vec<f64>>,
}

/// Result of the wideband ambiguity function.
#[derive(Debug, Clone)]
pub struct WidebandAmbiguityResult {
    /// Delay axis (seconds).
    pub delays: Vec<f64>,
    /// Scale factors.
    pub scales: Vec<f64>,
    /// 2-D ambiguity surface `|A(tau, s)|`.
    pub surface: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
//  Narrowband auto-ambiguity
// ---------------------------------------------------------------------------

/// Compute the narrowband auto-ambiguity function of a signal.
///
/// Uses FFT-based computation: for each delay `tau`, compute
/// `x(t + tau/2) * conj(x(t - tau/2))` and take its FFT over `t` to get
/// the Doppler axis.
///
/// # Errors
///
/// Returns an error when the signal is empty or the FFT fails.
pub fn auto_ambiguity(signal: &[f64], config: &AmbiguityConfig) -> FFTResult<AmbiguityResult> {
    let x: Vec<Complex64> = signal.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    auto_ambiguity_complex(&x, config)
}

/// Compute |A(tau, nu)|^2 (the ambiguity surface).
///
/// # Errors
///
/// Returns an error when the underlying ambiguity computation fails.
pub fn auto_ambiguity_surface(
    signal: &[f64],
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let mut result = auto_ambiguity(signal, config)?;
    for row in &mut result.surface {
        for v in row.iter_mut() {
            *v = *v * *v;
        }
    }
    Ok(result)
}

/// Auto-ambiguity for complex signals.
fn auto_ambiguity_complex(x: &[Complex64], config: &AmbiguityConfig) -> FFTResult<AmbiguityResult> {
    let n = x.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal is empty".to_string()));
    }

    let max_delay = config.max_delay.min(n - 1);
    let max_doppler = config.max_doppler.min(n);

    // Delay axis: -max_delay .. +max_delay
    let num_delays = 2 * max_delay + 1;
    let delays: Vec<f64> = (0..num_delays)
        .map(|i| {
            let tau = i as i64 - max_delay as i64;
            tau as f64 / config.fs
        })
        .collect();

    // Doppler axis
    let doppler_fft_len = 2 * max_doppler;
    let doppler_fft_len = if doppler_fft_len == 0 {
        1
    } else {
        doppler_fft_len
    };
    let dopplers: Vec<f64> = (0..doppler_fft_len)
        .map(|k| {
            let k_shifted = if k < doppler_fft_len / 2 {
                k as f64
            } else {
                k as f64 - doppler_fft_len as f64
            };
            k_shifted * config.fs / doppler_fft_len as f64
        })
        .collect();

    let mut surface: Vec<Vec<f64>> = Vec::with_capacity(num_delays);

    for delay_idx in 0..num_delays {
        let tau = delay_idx as i64 - max_delay as i64;

        // Build the product x(t + tau/2) * conj(x(t - tau/2))
        // We handle fractional tau by using integer shifts with interpolation
        // For simplicity, use integer tau directly (half-sample shifts are
        // approximated by (tau+1)/2 and tau/2).
        let half_tau_pos = tau.div_euclid(2);
        let half_tau_neg = tau - half_tau_pos;

        let mut product = vec![Complex64::new(0.0, 0.0); doppler_fft_len];

        for t in 0..doppler_fft_len.min(n) {
            let idx_plus = t as i64 + half_tau_pos;
            let idx_minus = t as i64 - half_tau_neg;

            if idx_plus >= 0
                && (idx_plus as usize) < n
                && idx_minus >= 0
                && (idx_minus as usize) < n
            {
                product[t] = x[idx_plus as usize] * x[idx_minus as usize].conj();
            }
        }

        // FFT over the t-axis to get the Doppler dimension
        let spectrum = fft(&product, None)?;
        let row: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        surface.push(row);
    }

    Ok(AmbiguityResult {
        delays,
        dopplers,
        surface,
    })
}

// ---------------------------------------------------------------------------
//  Cross-ambiguity
// ---------------------------------------------------------------------------

/// Compute the narrowband cross-ambiguity function between two signals.
///
/// ```text
/// A_xy(tau, nu) = integral x(t + tau/2) * conj(y(t - tau/2)) * exp(-j 2 pi nu t) dt
/// ```
///
/// # Errors
///
/// Returns an error when either signal is empty or the FFT fails.
pub fn cross_ambiguity(
    x: &[f64],
    y: &[f64],
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let xc: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let yc: Vec<Complex64> = y.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    cross_ambiguity_complex(&xc, &yc, config)
}

/// Compute |A_xy(tau, nu)|^2.
///
/// # Errors
///
/// Returns an error when the underlying cross-ambiguity computation fails.
pub fn cross_ambiguity_surface(
    x: &[f64],
    y: &[f64],
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let mut result = cross_ambiguity(x, y, config)?;
    for row in &mut result.surface {
        for v in row.iter_mut() {
            *v = *v * *v;
        }
    }
    Ok(result)
}

/// Cross-ambiguity for complex signals.
fn cross_ambiguity_complex(
    x: &[Complex64],
    y: &[Complex64],
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 {
        return Err(FFTError::ValueError(
            "Signals must be non-empty".to_string(),
        ));
    }
    let n = nx.max(ny);

    let max_delay = config.max_delay.min(n - 1);
    let max_doppler = config.max_doppler.min(n);

    let num_delays = 2 * max_delay + 1;
    let delays: Vec<f64> = (0..num_delays)
        .map(|i| {
            let tau = i as i64 - max_delay as i64;
            tau as f64 / config.fs
        })
        .collect();

    let doppler_fft_len = (2 * max_doppler).max(1);
    let dopplers: Vec<f64> = (0..doppler_fft_len)
        .map(|k| {
            let k_shifted = if k < doppler_fft_len / 2 {
                k as f64
            } else {
                k as f64 - doppler_fft_len as f64
            };
            k_shifted * config.fs / doppler_fft_len as f64
        })
        .collect();

    let mut surface: Vec<Vec<f64>> = Vec::with_capacity(num_delays);

    for delay_idx in 0..num_delays {
        let tau = delay_idx as i64 - max_delay as i64;
        let half_tau_pos = tau.div_euclid(2);
        let half_tau_neg = tau - half_tau_pos;

        let mut product = vec![Complex64::new(0.0, 0.0); doppler_fft_len];

        for t in 0..doppler_fft_len.min(n) {
            let idx_x = t as i64 + half_tau_pos;
            let idx_y = t as i64 - half_tau_neg;

            if idx_x >= 0 && (idx_x as usize) < nx && idx_y >= 0 && (idx_y as usize) < ny {
                product[t] = x[idx_x as usize] * y[idx_y as usize].conj();
            }
        }

        let spectrum = fft(&product, None)?;
        let row: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
        surface.push(row);
    }

    Ok(AmbiguityResult {
        delays,
        dopplers,
        surface,
    })
}

// ---------------------------------------------------------------------------
//  Wideband ambiguity
// ---------------------------------------------------------------------------

/// Compute the wideband (time-scale) ambiguity function.
///
/// ```text
/// A_wb(tau, s) = sqrt(|s|) * sum_t x(t) * conj(x(s * (t - tau)))
/// ```
///
/// Evaluates on a grid of delays and scale factors. The scales range from
/// `1/max_scale_ratio` to `max_scale_ratio` in logarithmic steps.
///
/// # Arguments
///
/// * `signal`          - Input real-valued signal
/// * `max_delay`       - Maximum delay in samples
/// * `num_scales`      - Number of scale factors to evaluate
/// * `max_scale_ratio` - Maximum scale ratio (>1). Scales span `[1/r, r]`.
/// * `fs`              - Sampling frequency (Hz)
///
/// # Errors
///
/// Returns an error when the signal is empty.
pub fn wideband_ambiguity(
    signal: &[f64],
    max_delay: usize,
    num_scales: usize,
    max_scale_ratio: f64,
    fs: f64,
) -> FFTResult<WidebandAmbiguityResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal is empty".to_string()));
    }
    if max_scale_ratio <= 0.0 {
        return Err(FFTError::ValueError(
            "max_scale_ratio must be positive".to_string(),
        ));
    }
    if num_scales == 0 {
        return Err(FFTError::ValueError(
            "num_scales must be positive".to_string(),
        ));
    }

    let max_delay = max_delay.min(n - 1);
    let num_delays = 2 * max_delay + 1;

    let delays: Vec<f64> = (0..num_delays)
        .map(|i| (i as i64 - max_delay as i64) as f64 / fs)
        .collect();

    // Logarithmically spaced scales
    let log_min = -(max_scale_ratio.ln());
    let log_max = max_scale_ratio.ln();
    let scales: Vec<f64> = if num_scales == 1 {
        vec![1.0]
    } else {
        (0..num_scales)
            .map(|i| {
                let frac = i as f64 / (num_scales - 1) as f64;
                (log_min + frac * (log_max - log_min)).exp()
            })
            .collect()
    };

    let mut surface: Vec<Vec<f64>> = Vec::with_capacity(num_delays);

    for delay_idx in 0..num_delays {
        let tau = delay_idx as i64 - max_delay as i64;
        let mut row = Vec::with_capacity(num_scales);

        for &s in &scales {
            let sqrt_s = s.abs().sqrt();
            let mut accum = Complex64::new(0.0, 0.0);

            for t in 0..n {
                // Scaled and delayed index: s * (t - tau)
                let scaled_idx = s * (t as f64 - tau as f64);
                let idx = scaled_idx.round() as i64;

                if idx >= 0 && (idx as usize) < n {
                    let x_t = Complex64::new(signal[t], 0.0);
                    let x_scaled = Complex64::new(signal[idx as usize], 0.0);
                    accum += x_t * x_scaled.conj();
                }
            }

            row.push((accum * sqrt_s).norm());
        }

        surface.push(row);
    }

    Ok(WidebandAmbiguityResult {
        delays,
        scales,
        surface,
    })
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// |A(0,0)|^2 should equal the signal energy squared.
    #[test]
    fn test_ambiguity_origin_is_energy() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).sin())
            .collect();

        let energy: f64 = signal.iter().map(|&v| v * v).sum();

        let config = AmbiguityConfig {
            max_delay: 32,
            max_doppler: 64,
            fs: 1.0,
        };

        let result = auto_ambiguity(&signal, &config).expect("Ambiguity should succeed");

        // Find the (tau=0, nu=0) entry
        let tau_zero_idx = config.max_delay; // centre of delay axis
        let nu_zero_idx = 0; // DC bin

        let a_00 = result.surface[tau_zero_idx][nu_zero_idx];

        // |A(0,0)| should be close to the signal energy
        // (the integral of |x(t)|^2 = energy)
        let ratio = a_00 / energy;
        assert!(
            (ratio - 1.0).abs() < 0.3,
            "|A(0,0)| = {}, energy = {}, ratio = {}",
            a_00,
            energy,
            ratio
        );
    }

    /// The auto-ambiguity surface should be symmetric in delay.
    #[test]
    fn test_ambiguity_delay_symmetry() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).cos())
            .collect();

        let config = AmbiguityConfig {
            max_delay: 16,
            max_doppler: 32,
            fs: 1.0,
        };

        let result = auto_ambiguity(&signal, &config).expect("Ambiguity should succeed");

        let centre = config.max_delay;
        // Check approximate symmetry: |A(tau, nu)| ~ |A(-tau, -nu)|
        for d in 1..=config.max_delay.min(10) {
            let row_pos = &result.surface[centre + d];
            let row_neg = &result.surface[centre - d];
            // At nu=0, should be approximately symmetric
            let diff = (row_pos[0] - row_neg[0]).abs();
            let max_val = row_pos[0].max(row_neg[0]).max(1e-15);
            assert!(
                diff / max_val < 0.5,
                "Symmetry broken at delay {}: pos={}, neg={}",
                d,
                row_pos[0],
                row_neg[0]
            );
        }
    }

    /// Cross-ambiguity of a signal with itself should match auto-ambiguity.
    #[test]
    fn test_cross_ambiguity_equals_auto() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / 64.0).sin())
            .collect();

        let config = AmbiguityConfig {
            max_delay: 16,
            max_doppler: 32,
            fs: 1.0,
        };

        let auto_result = auto_ambiguity(&signal, &config).expect("Auto ambiguity should succeed");
        let cross_result =
            cross_ambiguity(&signal, &signal, &config).expect("Cross ambiguity should succeed");

        // Should match
        for (auto_row, cross_row) in auto_result.surface.iter().zip(cross_result.surface.iter()) {
            for (&a, &c) in auto_row.iter().zip(cross_row.iter()) {
                let diff = (a - c).abs();
                let max_val = a.max(c).max(1e-15);
                assert!(
                    diff / max_val < 1e-10 || diff < 1e-10,
                    "Auto/cross mismatch: {} vs {}",
                    a,
                    c
                );
            }
        }
    }

    /// Wideband ambiguity at scale=1 and delay=0 should relate to signal energy.
    #[test]
    fn test_wideband_ambiguity_origin() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).sin())
            .collect();

        let energy: f64 = signal.iter().map(|&v| v * v).sum();

        let result =
            wideband_ambiguity(&signal, 16, 11, 2.0, 1.0).expect("Wideband should succeed");

        // Find scale closest to 1.0
        let scale_1_idx = result
            .scales
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (*a - 1.0).abs();
                let db = (*b - 1.0).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .expect("scales should not be empty");

        // Delay=0 is at the centre
        let tau_0_idx = 16; // max_delay

        let a_origin = result.surface[tau_0_idx][scale_1_idx];
        let ratio = a_origin / energy;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Wideband A(0,1) = {}, energy = {}, ratio = {}",
            a_origin,
            energy,
            ratio
        );
    }

    #[test]
    fn test_ambiguity_empty_signal() {
        let config = AmbiguityConfig::default();
        let empty: Vec<f64> = vec![];
        assert!(auto_ambiguity(&empty, &config).is_err());
        assert!(cross_ambiguity(&empty, &[1.0], &config).is_err());
        assert!(wideband_ambiguity(&empty, 10, 5, 2.0, 1.0).is_err());
    }

    /// The auto-ambiguity surface should have its maximum at (tau=0, nu=0).
    #[test]
    fn test_ambiguity_max_at_origin() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).sin())
            .collect();

        let config = AmbiguityConfig {
            max_delay: 32,
            max_doppler: 64,
            fs: 1.0,
        };

        let result = auto_ambiguity_surface(&signal, &config).expect("Surface should succeed");

        let tau_zero_idx = config.max_delay;
        let origin_val = result.surface[tau_zero_idx][0];

        // Check that origin is the global maximum (or very close)
        let global_max = result
            .surface
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            origin_val >= global_max * 0.9,
            "Origin ({}) should be near global max ({})",
            origin_val,
            global_max
        );
    }
}
