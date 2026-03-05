//! Hilbert Transform and analytic signal utilities
//!
//! This module provides the Hilbert transform and related functions for
//! computing the analytic signal, instantaneous amplitude (envelope),
//! instantaneous phase, and instantaneous frequency.
//!
//! The Hilbert transform is a linear operator that takes a real-valued signal
//! and produces its analytic signal representation, where the real part is
//! the original signal and the imaginary part is the Hilbert transform.
//!
//! # References
//!
//! * Marple, S. L. "Computing the Discrete-Time Analytic Signal via FFT."
//!   IEEE Transactions on Signal Processing, Vol. 47, No. 9, 1999.
//! * Boashash, B. "Estimating and interpreting the instantaneous frequency
//!   of a signal." Proceedings of the IEEE, Vol. 80, No. 4, 1992.

use crate::{FFTError, FFTResult};
use scirs2_core::numeric::{Complex64, NumCast};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Compute the analytic signal using the Hilbert transform.
///
/// The analytic signal `xa[n]` of a real signal `x[n]` is defined as:
///   `xa[n] = x[n] + j * H{x[n]}`
/// where `H{}` denotes the Hilbert transform.
///
/// This is computed via the FFT method:
/// 1. Compute FFT of the input signal
/// 2. Zero out negative frequency components
/// 3. Double positive frequency components
/// 4. Compute inverse FFT
///
/// # Arguments
///
/// * `x` - Input real-valued signal
///
/// # Returns
///
/// Complex-valued analytic signal where the real part is the original signal
/// and the imaginary part is the Hilbert transform.
///
/// # Errors
///
/// Returns an error if the FFT computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::analytic_signal;
/// use std::f64::consts::PI;
///
/// let n = 100;
/// let freq = 5.0;
/// let dt = 0.01;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).cos()).collect();
///
/// let xa = analytic_signal(&signal).expect("Hilbert transform should succeed");
///
/// // The envelope of a pure cosine should be approximately 1
/// let mid = n / 2;
/// let envelope = (xa[mid].re.powi(2) + xa[mid].im.powi(2)).sqrt();
/// assert!((envelope - 1.0).abs() < 0.15);
/// ```
pub fn analytic_signal<T>(x: &[T]) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let n = x.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "Input signal cannot be empty".to_string(),
        ));
    }

    // Convert input to f64
    let signal: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val).ok_or_else(|| {
                FFTError::ValueError(format!("Could not convert {val:?} to numeric type"))
            })
        })
        .collect::<FFTResult<Vec<_>>>()?;

    // Compute FFT
    let spectrum = crate::fft::fft(&signal, None)?;

    // Build the analytic signal filter h[k]:
    // h[0] = 1 (DC)
    // h[k] = 2 for 0 < k < N/2 (positive frequencies)
    // h[N/2] = 1 for even N (Nyquist)
    // h[k] = 0 for k > N/2 (negative frequencies)
    let mut h = vec![Complex64::new(0.0, 0.0); n];
    h[0] = Complex64::new(1.0, 0.0);

    if n % 2 == 0 {
        // Even length
        for k in 1..n / 2 {
            h[k] = Complex64::new(2.0, 0.0);
        }
        h[n / 2] = Complex64::new(1.0, 0.0);
        // h[n/2+1..n] remain 0
    } else {
        // Odd length
        for k in 1..=(n - 1) / 2 {
            h[k] = Complex64::new(2.0, 0.0);
        }
        // h[(n+1)/2..n] remain 0
    }

    // Multiply spectrum by the filter
    let filtered: Vec<Complex64> = spectrum
        .iter()
        .zip(h.iter())
        .map(|(&s, &hk)| s * hk)
        .collect();

    // Inverse FFT to get the analytic signal
    let result = crate::fft::ifft(&filtered, None)?;

    Ok(result)
}

/// Compute the instantaneous envelope (amplitude) of a signal.
///
/// The envelope is the magnitude of the analytic signal:
///   `A[n] = |xa[n]| = sqrt(x[n]^2 + H{x[n]}^2)`
///
/// # Arguments
///
/// * `x` - Input real-valued signal
///
/// # Returns
///
/// The instantaneous amplitude envelope of the signal.
///
/// # Errors
///
/// Returns an error if the Hilbert transform computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::envelope;
/// use std::f64::consts::PI;
///
/// // AM-modulated signal: carrier * (1 + m * cos(modulation))
/// let n = 200;
/// let dt = 0.005;
/// let carrier_freq = 50.0;
/// let mod_freq = 5.0;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 * dt;
///     (1.0 + 0.5 * (2.0 * PI * mod_freq * t).cos()) * (2.0 * PI * carrier_freq * t).cos()
/// }).collect();
///
/// let env = envelope(&signal).expect("Envelope should succeed");
/// assert_eq!(env.len(), n);
/// // Envelope should be positive
/// for &val in &env[10..n-10] {
///     assert!(val >= 0.0);
/// }
/// ```
pub fn envelope<T>(x: &[T]) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let xa = analytic_signal(x)?;
    Ok(xa.iter().map(|c| c.norm()).collect())
}

/// Compute the instantaneous phase of a signal.
///
/// The instantaneous phase is the argument (angle) of the analytic signal:
///   `phi[n] = atan2(Im(xa[n]), Re(xa[n]))`
///
/// The result is in radians, in the range `[-pi, pi]`.
///
/// # Arguments
///
/// * `x` - Input real-valued signal
///
/// # Returns
///
/// The instantaneous phase of the signal in radians.
///
/// # Errors
///
/// Returns an error if the Hilbert transform computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::instantaneous_phase;
/// use std::f64::consts::PI;
///
/// let n = 100;
/// let freq = 5.0;
/// let dt = 0.01;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).cos()).collect();
///
/// let phase = instantaneous_phase(&signal).expect("Phase computation should succeed");
/// assert_eq!(phase.len(), n);
/// ```
pub fn instantaneous_phase<T>(x: &[T]) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let xa = analytic_signal(x)?;
    Ok(xa.iter().map(|c| c.im.atan2(c.re)).collect())
}

/// Compute the unwrapped instantaneous phase of a signal.
///
/// Unwrapping removes discontinuities (jumps greater than pi) from the phase,
/// producing a continuous phase trajectory.
///
/// # Arguments
///
/// * `x` - Input real-valued signal
///
/// # Returns
///
/// The unwrapped instantaneous phase in radians.
///
/// # Errors
///
/// Returns an error if the Hilbert transform computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::instantaneous_phase_unwrapped;
/// use std::f64::consts::PI;
///
/// let n = 200;
/// let freq = 5.0;
/// let dt = 0.005;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * freq * i as f64 * dt).cos()).collect();
///
/// let phase = instantaneous_phase_unwrapped(&signal).expect("Unwrapped phase should succeed");
/// // For a positive frequency cosine, phase should be monotonically increasing
/// // Check a section away from boundaries
/// for i in 20..n-20 {
///     assert!(phase[i] >= phase[i-1] - 0.1, "Phase should generally increase for positive freq");
/// }
/// ```
pub fn instantaneous_phase_unwrapped<T>(x: &[T]) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let wrapped_phase = instantaneous_phase(x)?;
    Ok(unwrap_phase(&wrapped_phase))
}

/// Compute the instantaneous frequency of a signal.
///
/// The instantaneous frequency is the time derivative of the instantaneous phase:
///   `f[n] = (1 / (2*pi)) * d(phi)/dt`
///
/// This is approximated using finite differences on the unwrapped phase:
///   `f[n] = (phi[n+1] - phi[n]) / (2*pi*dt)`
///
/// # Arguments
///
/// * `x` - Input real-valued signal
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// The instantaneous frequency in Hz. The output has length `n-1` since
/// it's computed via finite differences.
///
/// # Errors
///
/// Returns an error if the signal has fewer than 2 elements, or if
/// the Hilbert transform computation fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::instantaneous_frequency;
/// use std::f64::consts::PI;
///
/// let n = 200;
/// let fs = 1000.0;
/// let freq = 50.0;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * PI * freq * t).cos()
/// }).collect();
///
/// let inst_freq = instantaneous_frequency(&signal, fs).expect("Inst freq should succeed");
/// assert_eq!(inst_freq.len(), n - 1);
///
/// // Check that the instantaneous frequency is close to 50 Hz in the middle
/// let mid_start = n / 4;
/// let mid_end = 3 * n / 4;
/// for &f in &inst_freq[mid_start..mid_end] {
///     assert!((f - freq).abs() < 5.0, "Instantaneous freq should be near {freq} Hz, got {f}");
/// }
/// ```
pub fn instantaneous_frequency<T>(x: &[T], fs: f64) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    if x.len() < 2 {
        return Err(FFTError::ValueError(
            "Signal must have at least 2 samples for instantaneous frequency".to_string(),
        ));
    }

    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let phase = instantaneous_phase_unwrapped(x)?;

    // Compute instantaneous frequency via finite differences
    let dt = 1.0 / fs;
    let mut freq = Vec::with_capacity(phase.len() - 1);
    for i in 0..phase.len() - 1 {
        let dphi = phase[i + 1] - phase[i];
        freq.push(dphi / (2.0 * PI * dt));
    }

    Ok(freq)
}

/// Compute the instantaneous frequency using central differences for better accuracy.
///
/// Uses central differences `(phi[n+1] - phi[n-1]) / (2*dt)` which gives
/// O(dt^2) accuracy instead of O(dt) from forward differences.
///
/// # Arguments
///
/// * `x` - Input real-valued signal
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// The instantaneous frequency in Hz. The output has length `n-2` since
/// central differences cannot be computed at the boundaries.
///
/// # Errors
///
/// Returns an error if the signal has fewer than 3 elements.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert::instantaneous_frequency_central;
/// use std::f64::consts::PI;
///
/// let n = 200;
/// let fs = 1000.0;
/// let freq = 50.0;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * PI * freq * t).cos()
/// }).collect();
///
/// let inst_freq = instantaneous_frequency_central(&signal, fs).expect("Central diff ok");
/// assert_eq!(inst_freq.len(), n - 2);
/// ```
pub fn instantaneous_frequency_central<T>(x: &[T], fs: f64) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    if x.len() < 3 {
        return Err(FFTError::ValueError(
            "Signal must have at least 3 samples for central difference frequency".to_string(),
        ));
    }

    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let phase = instantaneous_phase_unwrapped(x)?;

    // Central differences: f[n] = (phi[n+1] - phi[n-1]) / (4*pi*dt)
    let dt = 1.0 / fs;
    let mut freq = Vec::with_capacity(phase.len() - 2);
    for i in 1..phase.len() - 1 {
        let dphi = phase[i + 1] - phase[i - 1];
        freq.push(dphi / (4.0 * PI * dt));
    }

    Ok(freq)
}

/// Unwrap phase by adding/subtracting 2*pi to remove discontinuities.
///
/// This function detects jumps in the phase that are greater than pi and
/// corrects them to produce a continuous phase signal.
fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = Vec::with_capacity(phase.len());
    unwrapped.push(phase[0]);

    let mut cumulative_correction = 0.0;

    for i in 1..phase.len() {
        let mut diff = phase[i] - phase[i - 1];

        // Wrap the difference to [-pi, pi]
        while diff > PI {
            diff -= 2.0 * PI;
            cumulative_correction -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
            cumulative_correction += 2.0 * PI;
        }

        unwrapped.push(phase[i] + cumulative_correction);
    }

    unwrapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_analytic_signal_cosine() {
        // For a cosine wave, the analytic signal should have magnitude ~1
        let n = 256;
        let freq = 10.0;
        let dt = 1.0 / 256.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 * dt).cos())
            .collect();

        let xa = analytic_signal(&signal).expect("Analytic signal should succeed");
        assert_eq!(xa.len(), n);

        // Check envelope in the middle (avoid boundary effects)
        for i in n / 4..3 * n / 4 {
            let mag = xa[i].norm();
            assert_relative_eq!(mag, 1.0, epsilon = 0.05);
        }
    }

    #[test]
    fn test_analytic_signal_real_part_preserved() {
        // The real part of the analytic signal should be the original signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let xa = analytic_signal(&signal).expect("Analytic signal should succeed");

        for (i, &val) in signal.iter().enumerate() {
            assert_relative_eq!(xa[i].re, val, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_envelope_am_signal() {
        // AM signal: envelope should match the modulation
        let n = 512;
        let fs = 1000.0;
        let carrier_freq = 100.0;
        let mod_freq = 10.0;
        let mod_depth = 0.5;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let modulation = 1.0 + mod_depth * (2.0 * PI * mod_freq * t).cos();
                modulation * (2.0 * PI * carrier_freq * t).cos()
            })
            .collect();

        let env = envelope(&signal).expect("Envelope should succeed");
        assert_eq!(env.len(), n);

        // Envelope should be non-negative
        for &val in &env {
            assert!(val >= -1e-10, "Envelope should be non-negative");
        }

        // Envelope should vary between approximately (1 - mod_depth) and (1 + mod_depth)
        // away from boundaries
        let env_slice = &env[n / 4..3 * n / 4];
        let max_env = env_slice.iter().copied().fold(f64::MIN, f64::max);
        let min_env = env_slice.iter().copied().fold(f64::MAX, f64::min);

        assert!(max_env > 1.0, "Max envelope should be > 1");
        assert!(min_env < 1.0, "Min envelope should be < 1");
    }

    #[test]
    fn test_instantaneous_phase_linear() {
        // For a single-frequency cosine, the unwrapped phase should be approximately linear
        let n = 256;
        let freq = 5.0;
        let dt = 1.0 / 256.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 * dt).cos())
            .collect();

        let phase = instantaneous_phase_unwrapped(&signal).expect("Unwrapped phase should succeed");
        assert_eq!(phase.len(), n);

        // Check that phase is approximately linear in the middle
        // (avoid boundary artifacts)
        let start = n / 4;
        let end = 3 * n / 4;

        // Compute average phase rate (should be ~2*pi*freq per sample)
        let expected_rate = 2.0 * PI * freq * dt;
        let mut rates = Vec::new();
        for i in start..end - 1 {
            rates.push(phase[i + 1] - phase[i]);
        }
        let avg_rate: f64 = rates.iter().sum::<f64>() / rates.len() as f64;
        assert_relative_eq!(avg_rate, expected_rate, epsilon = 0.1);
    }

    #[test]
    fn test_instantaneous_frequency_constant() {
        // For a single-frequency signal, inst freq should be approximately constant
        let n = 512;
        let fs = 1000.0;
        let freq = 50.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * freq * t).cos()
            })
            .collect();

        let inst_freq = instantaneous_frequency(&signal, fs).expect("Inst freq should succeed");
        assert_eq!(inst_freq.len(), n - 1);

        // In the middle section, frequency should be close to the expected value
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        for &f in &inst_freq[mid_start..mid_end] {
            assert!(
                (f - freq).abs() < 3.0,
                "Frequency should be near {freq} Hz, got {f}"
            );
        }
    }

    #[test]
    fn test_instantaneous_frequency_central_accuracy() {
        // Central differences should be more accurate than forward differences
        let n = 512;
        let fs = 1000.0;
        let freq = 50.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * freq * t).cos()
            })
            .collect();

        let freq_forward =
            instantaneous_frequency(&signal, fs).expect("Forward inst freq should succeed");
        let freq_central =
            instantaneous_frequency_central(&signal, fs).expect("Central inst freq should succeed");

        assert_eq!(freq_central.len(), n - 2);

        // Compare errors in the middle section
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;

        let forward_errors: Vec<f64> = freq_forward[mid_start..mid_end]
            .iter()
            .map(|&f| (f - freq).abs())
            .collect();
        let central_errors: Vec<f64> = freq_central[mid_start - 1..mid_end - 1]
            .iter()
            .map(|&f| (f - freq).abs())
            .collect();

        let avg_forward_error: f64 =
            forward_errors.iter().sum::<f64>() / forward_errors.len() as f64;
        let avg_central_error: f64 =
            central_errors.iter().sum::<f64>() / central_errors.len() as f64;

        // Central should be at least as good or better
        assert!(
            avg_central_error <= avg_forward_error + 0.5,
            "Central ({avg_central_error:.4}) should be at least as accurate as forward ({avg_forward_error:.4})"
        );
    }

    #[test]
    fn test_chirp_instantaneous_frequency() {
        // For a chirp signal (linear frequency sweep), inst freq should increase linearly
        let n = 1024;
        let fs = 1000.0;
        let f0 = 10.0; // Start frequency
        let f1 = 100.0; // End frequency
        let duration = n as f64 / fs;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let freq_at_t = f0 + (f1 - f0) * t / duration;
                let phase = 2.0 * PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration));
                phase.cos()
            })
            .collect();

        let inst_freq =
            instantaneous_frequency(&signal, fs).expect("Chirp inst freq should succeed");

        // Check that frequency increases in the middle region
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;

        // Average frequency should increase over time
        let first_quarter_avg: f64 =
            inst_freq[mid_start..n / 2].iter().sum::<f64>() / (n / 2 - mid_start) as f64;
        let second_quarter_avg: f64 =
            inst_freq[n / 2..mid_end].iter().sum::<f64>() / (mid_end - n / 2) as f64;

        assert!(
            second_quarter_avg > first_quarter_avg,
            "Frequency should increase: second half ({second_quarter_avg:.1}) > first half ({first_quarter_avg:.1})"
        );
    }

    #[test]
    fn test_empty_signal_error() {
        let empty: Vec<f64> = vec![];
        assert!(analytic_signal(&empty).is_err());
        assert!(envelope(&empty).is_err());
        assert!(instantaneous_phase(&empty).is_err());
    }

    #[test]
    fn test_short_signal_frequency_error() {
        let short = vec![1.0];
        assert!(instantaneous_frequency(&short, 100.0).is_err());

        let very_short = vec![1.0, 2.0];
        assert!(instantaneous_frequency_central(&very_short, 100.0).is_err());
    }

    #[test]
    fn test_negative_fs_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        assert!(instantaneous_frequency(&signal, -100.0).is_err());
        assert!(instantaneous_frequency(&signal, 0.0).is_err());
    }

    #[test]
    fn test_unwrap_phase() {
        // Test phase unwrapping
        let wrapped = vec![
            0.0, 1.0, 2.0, 3.0, -2.8, -1.8, -0.8, 0.2, 1.2, 2.2, 3.2, -2.6,
        ];
        let unwrapped = unwrap_phase(&wrapped);

        // After unwrapping, the phase should be monotonically increasing
        for i in 1..unwrapped.len() {
            let diff = unwrapped[i] - unwrapped[i - 1];
            // Original differences should be approximately 1.0 (with wrapping jumps corrected)
            assert!(
                diff > -0.5 && diff < 2.0,
                "Unwrapped difference should be smooth, got {diff} at index {i}"
            );
        }
    }

    #[test]
    fn test_dc_signal_envelope() {
        // A constant signal should have constant envelope
        let signal = vec![3.0; 64];
        let env = envelope(&signal).expect("DC envelope should succeed");

        // DC signal envelope should be approximately 3.0
        // (except at boundaries due to spectral leakage)
        for &val in &env[8..56] {
            assert_relative_eq!(val, 3.0, epsilon = 0.5);
        }
    }
}
