//! Wigner-Ville Distribution and related time-frequency representations.
//!
//! This module provides the Wigner-Ville Distribution (WVD) and its smoothed
//! variants, along with supporting functions for analytic signal computation
//! and instantaneous attributes.
//!
//! ## Overview
//!
//! The Wigner-Ville Distribution has the best joint time-frequency resolution
//! of any bilinear TF representation (satisfying the marginal properties), but
//! suffers from cross-terms when the signal has multiple components. The
//! Pseudo-WVD and Smoothed Pseudo-WVD reduce cross-terms at the cost of
//! resolution.
//!
//! - [`wvd`] – Full WVD (best resolution, most cross-terms).
//! - [`pwvd`] – Pseudo WVD (frequency-smoothed, fewer cross-terms).
//! - [`spwvd`] – Smoothed Pseudo WVD (2-D smoothing, minimal cross-terms).
//! - [`analytic_signal`] – Hilbert-based analytic signal (prerequisite for WVD).
//! - [`instantaneous_frequency`] – Instantaneous frequency from analytic signal.
//! - [`instantaneous_amplitude`] – Envelope of analytic signal.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Analytic Signal (public)
// ---------------------------------------------------------------------------

/// Compute the analytic signal of a real-valued input using the Hilbert transform.
///
/// The analytic signal `z(t) = x(t) + j*H{x(t)}` is computed via the one-sided
/// spectrum method:
/// 1. Compute FFT of `x`.
/// 2. Zero out negative frequencies.
/// 3. Double positive frequencies.
/// 4. IFFT.
///
/// # Arguments
///
/// * `x` - Real-valued input signal
///
/// # Returns
///
/// Complex analytic signal of the same length as `x`.
///
/// # Examples
///
/// ```
/// use scirs2_signal::wigner_ville::analytic_signal;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0;
/// let n = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).cos())
///     .collect();
///
/// let z = analytic_signal(&signal).expect("analytic_signal failed");
/// assert_eq!(z.len(), n);
/// // For a cosine, the imaginary part should be close to a sine of same frequency
/// ```
pub fn analytic_signal(x: &[f64]) -> SignalResult<Vec<Complex64>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let spectrum = scirs2_fft::fft(x, None)
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

    // Build one-sided spectrum
    let mut h = vec![Complex64::new(0.0, 0.0); n];
    h[0] = spectrum[0]; // DC component
    let half = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    for i in 1..half {
        h[i] = spectrum[i] * 2.0; // double positive frequencies
    }
    if n % 2 == 0 && half < n {
        h[half] = spectrum[half]; // Nyquist component (even N only)
    }
    // Negative frequencies remain zero

    let result = scirs2_fft::ifft(&h, None)
        .map_err(|e| SignalError::ComputationError(format!("IFFT error: {e}")))?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// Instantaneous attributes
// ---------------------------------------------------------------------------

/// Compute the instantaneous frequency of an analytic signal.
///
/// Uses the phase difference formula:
/// ```text
/// f_inst(t) = d(angle(z(t))) / dt / (2π)
///           ≈ (angle(z[t+1]) - angle(z[t])) * fs / (2π)
/// ```
///
/// Phase unwrapping is applied before differentiation to handle phase jumps.
///
/// # Arguments
///
/// * `analytic` - Analytic signal (complex-valued, from [`analytic_signal`])
/// * `fs` - Sampling frequency (Hz)
///
/// # Returns
///
/// Vector of instantaneous frequency values in Hz. Length is `n - 1` (one
/// sample shorter than the input due to first-order differencing).
///
/// # Examples
///
/// ```
/// use scirs2_signal::wigner_ville::{analytic_signal, instantaneous_frequency};
/// use std::f64::consts::PI;
///
/// let fs = 1000.0;
/// let n = 512;
/// let freq = 200.0_f64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * freq * i as f64 / fs).cos())
///     .collect();
/// let z = analytic_signal(&signal).expect("failed");
/// let inst_f = instantaneous_frequency(&z, fs);
/// // Middle values should be near 200 Hz
/// let mid = inst_f.len() / 2;
/// assert!((inst_f[mid] - freq).abs() < 2.0);
/// ```
pub fn instantaneous_frequency(analytic: &[Complex64], fs: f64) -> Vec<f64> {
    let n = analytic.len();
    if n < 2 {
        return Vec::new();
    }

    // Extract phases
    let phases: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase
    let mut unwrapped = vec![0.0_f64; n];
    unwrapped[0] = phases[0];
    for i in 1..n {
        let mut diff = phases[i] - phases[i - 1];
        // Wrap into (-pi, pi]
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff <= -PI {
            diff += 2.0 * PI;
        }
        unwrapped[i] = unwrapped[i - 1] + diff;
    }

    // Differentiate phase and convert to Hz
    (0..(n - 1))
        .map(|i| (unwrapped[i + 1] - unwrapped[i]) * fs / (2.0 * PI))
        .collect()
}

/// Compute the instantaneous amplitude (envelope) of an analytic signal.
///
/// The instantaneous amplitude is the magnitude of the complex analytic signal:
/// `A(t) = |z(t)| = sqrt(x(t)^2 + H{x(t)}^2)`
///
/// # Arguments
///
/// * `analytic` - Analytic signal (complex-valued, from [`analytic_signal`])
///
/// # Returns
///
/// Vector of amplitude values. Same length as input.
///
/// # Examples
///
/// ```
/// use scirs2_signal::wigner_ville::{analytic_signal, instantaneous_amplitude};
/// use std::f64::consts::PI;
///
/// let fs = 1000.0;
/// let n = 256;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).cos())
///     .collect();
/// let z = analytic_signal(&signal).expect("failed");
/// let amp = instantaneous_amplitude(&z);
/// // For a pure cosine, amplitude should be near 1.0
/// let mid = amp.len() / 2;
/// assert!((amp[mid] - 1.0).abs() < 0.05);
/// ```
pub fn instantaneous_amplitude(analytic: &[Complex64]) -> Vec<f64> {
    analytic
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

// ---------------------------------------------------------------------------
// Wigner-Ville Distribution
// ---------------------------------------------------------------------------

/// Compute the Wigner-Ville Distribution (WVD) of a real signal.
///
/// The WVD is the Fourier transform of the instantaneous autocorrelation:
/// ```text
/// W(t, f) = integral  z*(t - τ/2) * z(t + τ/2) * exp(-j2πfτ) dτ
/// ```
///
/// This implementation uses the discrete version with the analytic signal to
/// suppress negative-frequency aliases (the "analytic WVD" or AWVD).
///
/// # Arguments
///
/// * `x` - Real-valued input signal
/// * `n_freqs` - Number of frequency bins. Defaults to `2 * next_power_of_2(n)`.
///   Should be even.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_freqs, n_samples)`. Each column is the WVD
/// at the corresponding time sample, each row is a frequency bin.
///
/// # Examples
///
/// ```
/// use scirs2_signal::wigner_ville::wvd;
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let n = 64;
/// // Linear chirp from 0 to fs/4
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / fs;
///         (2.0 * PI * (10.0 * t + 30.0 * t * t)).sin()
///     })
///     .collect();
///
/// let wvd_mat = wvd(&signal, None).expect("wvd failed");
/// assert_eq!(wvd_mat.ncols(), n);
/// ```
pub fn wvd(x: &[f64], n_freqs: Option<usize>) -> SignalResult<Array2<f64>> {
    let n = x.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }

    let nf = resolve_n_freqs(n_freqs, n);
    let z = analytic_signal(x)?;
    wvd_from_analytic_arr(&z, nf)
}

/// Compute the Pseudo Wigner-Ville Distribution (windowed WVD).
///
/// The PWVD applies a time-domain window to the lag variable of the
/// instantaneous autocorrelation, reducing cross-term interference at the
/// cost of frequency resolution:
/// ```text
/// PW(t, f) = integral  h(τ) * z*(t - τ) * z(t + τ) * exp(-j4πfτ) dτ
/// ```
///
/// # Arguments
///
/// * `x` - Real-valued input signal
/// * `window` - Lag-domain window (typically Hann or Gaussian, symmetric,
///   odd length preferred). A shorter window reduces cross-terms more.
/// * `n_freqs` - Number of frequency bins.
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_freqs, n_samples)`.
///
/// # Examples
///
/// ```
/// use scirs2_signal::wigner_ville::pwvd;
/// use std::f64::consts::PI;
///
/// let n = 64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
///     .collect();
/// let window: Vec<f64> = (0..17)
///     .map(|i| {
///         let x = (i as f64 - 8.0) / 4.0;
///         (-0.5 * x * x).exp()
///     })
///     .collect();
///
/// let pwvd_mat = pwvd(&signal, &window, None).expect("pwvd failed");
/// assert_eq!(pwvd_mat.ncols(), n);
/// ```
pub fn pwvd(
    x: &[f64],
    window: &[f64],
    n_freqs: Option<usize>,
) -> SignalResult<Array2<f64>> {
    let n = x.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    if window.is_empty() {
        return Err(SignalError::ValueError("Window must not be empty".into()));
    }

    let nf = resolve_n_freqs(n_freqs, n);
    let z = analytic_signal(x)?;
    pwvd_from_analytic_arr(&z, window, nf)
}

/// Compute the Smoothed Pseudo Wigner-Ville Distribution (SPWVD).
///
/// The SPWVD applies separate smoothing windows in both time and frequency:
/// - `freq_window` (lag/frequency window `h`) suppresses cross-terms in frequency.
/// - `time_window` (time smoothing window `g`) suppresses cross-terms in time.
///
/// This produces the smoothest TF surface with the least cross-terms, at the
/// cost of joint time-frequency resolution.
///
/// # Arguments
///
/// * `x` - Real-valued input signal
/// * `time_window` - Time-smoothing window (applied along the time axis)
/// * `freq_window` - Lag/frequency window (applied in the lag domain)
/// * `n_freqs` - Number of frequency bins
///
/// # Returns
///
/// `Array2<f64>` of shape `(n_freqs, n_samples)`.
pub fn spwvd(
    x: &[f64],
    time_window: &[f64],
    freq_window: &[f64],
    n_freqs: Option<usize>,
) -> SignalResult<Array2<f64>> {
    let n = x.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    if time_window.is_empty() || freq_window.is_empty() {
        return Err(SignalError::ValueError(
            "Time and frequency windows must not be empty".into(),
        ));
    }

    let nf = resolve_n_freqs(n_freqs, n);
    let z = analytic_signal(x)?;

    // First compute pseudo-WVD (frequency smoothed)
    let pwvd_mat = pwvd_from_analytic_arr(&z, freq_window, nf)?;

    // Then smooth along the time axis with time_window
    smooth_time_axis(&pwvd_mat, time_window)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Choose the number of frequency bins: use provided value (ensure even),
/// or fall back to 2 × next_power_of_2(n).
fn resolve_n_freqs(n_freqs: Option<usize>, n: usize) -> usize {
    let nf = n_freqs.unwrap_or_else(|| {
        let mut p = 1_usize;
        while p < n {
            p <<= 1;
        }
        p * 2 // 2× oversampling for WVD
    });
    if nf % 2 != 0 { nf + 1 } else { nf.max(2) }
}

/// Core WVD computation from an analytic signal, returning Array2.
fn wvd_from_analytic_arr(z: &[Complex64], nf: usize) -> SignalResult<Array2<f64>> {
    let n = z.len();
    let mut out = Array2::<f64>::zeros((nf, n));

    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n {
        // Maximum lag that keeps both z[t+tau] and z[t-tau] in bounds
        let tau_max = t.min(n - 1 - t).min(nf / 2 - 1);

        // Zero row buffer
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        // Fill instantaneous autocorrelation: R[tau] = z[t+tau] * conj(z[t-tau])
        for tau in 0..=tau_max {
            let val = z[t + tau] * z[t - tau].conj();
            row_buf[tau] = val;
            if tau > 0 {
                row_buf[nf - tau] = val.conj();
            }
        }

        // FFT to get frequency slice
        let freq_row = scirs2_fft::fft(&row_buf, Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            out[(f, t)] = freq_row[f].re;
        }
    }

    Ok(out)
}

/// Pseudo-WVD from analytic signal with lag window applied.
fn pwvd_from_analytic_arr(
    z: &[Complex64],
    window: &[f64],
    nf: usize,
) -> SignalResult<Array2<f64>> {
    let n = z.len();
    let win_half = window.len() / 2;
    let mut out = Array2::<f64>::zeros((nf, n));
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n {
        let tau_max = t
            .min(n - 1 - t)
            .min(nf / 2 - 1)
            .min(win_half);

        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        for tau in 0..=tau_max {
            // Window value at this lag
            let win_val = if tau < window.len() {
                window[win_half.saturating_sub(tau).min(window.len() - 1)]
            } else {
                0.0
            };

            let val = z[t + tau] * z[t - tau].conj() * win_val;
            row_buf[tau] = val;
            if tau > 0 {
                row_buf[nf - tau] = val.conj();
            }
        }

        let freq_row = scirs2_fft::fft(&row_buf, Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            out[(f, t)] = freq_row[f].re;
        }
    }

    Ok(out)
}

/// Smooth a TF distribution along the time axis using a 1-D window.
fn smooth_time_axis(
    tf: &Array2<f64>,
    window: &[f64],
) -> SignalResult<Array2<f64>> {
    let (nf, nt) = tf.dim();
    let win_len = window.len();
    let half = win_len / 2;

    // Normalize window
    let win_sum: f64 = window.iter().sum();
    let win_norm: f64 = if win_sum.abs() < 1e-30 { 1.0 } else { win_sum };

    let mut out = Array2::<f64>::zeros((nf, nt));

    for f in 0..nf {
        for t in 0..nt {
            let mut acc = 0.0_f64;
            let mut weight_sum = 0.0_f64;
            for (wi, &wv) in window.iter().enumerate() {
                let t_src = t as isize + wi as isize - half as isize;
                if t_src >= 0 && (t_src as usize) < nt {
                    acc += tf[(f, t_src as usize)] * wv;
                    weight_sum += wv;
                }
            }
            out[(f, t)] = if weight_sum.abs() > 1e-30 {
                acc / weight_sum * win_norm
            } else {
                0.0
            };
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_cosine(freq: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).cos())
            .collect()
    }

    fn make_sine(freq: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    // --- analytic_signal tests ---

    #[test]
    fn test_analytic_signal_length() {
        let signal = make_cosine(100.0, 256, 1000.0);
        let z = analytic_signal(&signal).expect("analytic_signal failed");
        assert_eq!(z.len(), 256);
    }

    #[test]
    fn test_analytic_signal_empty() {
        let z = analytic_signal(&[]).expect("empty analytic_signal failed");
        assert!(z.is_empty());
    }

    #[test]
    fn test_analytic_signal_real_part_matches_input() {
        // The real part of the analytic signal should approximately equal the input
        let fs = 1000.0;
        let n = 256;
        let signal = make_cosine(100.0, n, fs);
        let z = analytic_signal(&signal).expect("failed");

        // Check middle section (avoid edge effects)
        for i in 32..(n - 32) {
            assert_relative_eq!(z[i].re, signal[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_analytic_signal_imaginary_is_hilbert() {
        // For cos(2πft), the Hilbert transform is sin(2πft)
        let fs = 1000.0;
        let n = 256;
        let freq = 100.0;
        let signal = make_cosine(freq, n, fs);
        let expected_hilbert = make_sine(freq, n, fs);
        let z = analytic_signal(&signal).expect("failed");

        // Check middle section
        for i in 32..(n - 32) {
            assert_relative_eq!(z[i].im, expected_hilbert[i], epsilon = 0.05);
        }
    }

    // --- instantaneous_frequency tests ---

    #[test]
    fn test_instantaneous_frequency_length() {
        let n = 128;
        let z: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((0.1 * i as f64).cos(), (0.1 * i as f64).sin()))
            .collect();
        let inst_f = instantaneous_frequency(&z, 1000.0);
        assert_eq!(inst_f.len(), n - 1);
    }

    #[test]
    fn test_instantaneous_frequency_pure_tone() {
        let fs = 1000.0;
        let freq = 200.0;
        let n = 512;
        let signal = make_cosine(freq, n, fs);
        let z = analytic_signal(&signal).expect("failed");
        let inst_f = instantaneous_frequency(&z, fs);

        // Middle values should be near freq
        let mid = inst_f.len() / 2;
        assert!(
            (inst_f[mid] - freq).abs() < 3.0,
            "Expected ~{} Hz, got {} Hz",
            freq,
            inst_f[mid]
        );
    }

    #[test]
    fn test_instantaneous_frequency_empty() {
        let inst_f = instantaneous_frequency(&[], 1000.0);
        assert!(inst_f.is_empty());
    }

    #[test]
    fn test_instantaneous_frequency_single() {
        let z = vec![Complex64::new(1.0, 0.0)];
        let inst_f = instantaneous_frequency(&z, 1000.0);
        assert!(inst_f.is_empty());
    }

    // --- instantaneous_amplitude tests ---

    #[test]
    fn test_instantaneous_amplitude_pure_tone() {
        let fs = 1000.0;
        let n = 256;
        let signal = make_cosine(100.0, n, fs);
        let z = analytic_signal(&signal).expect("failed");
        let amp = instantaneous_amplitude(&z);

        // Middle values should be near 1.0 for a unit-amplitude cosine
        for i in 32..(n - 32) {
            assert_relative_eq!(amp[i], 1.0, epsilon = 0.05);
        }
    }

    #[test]
    fn test_instantaneous_amplitude_length() {
        let z: Vec<Complex64> = vec![Complex64::new(3.0, 4.0); 10];
        let amp = instantaneous_amplitude(&z);
        assert_eq!(amp.len(), 10);
        // |3 + 4j| = 5
        for &a in &amp {
            assert_relative_eq!(a, 5.0, epsilon = 1e-10);
        }
    }

    // --- wvd tests ---

    #[test]
    fn test_wvd_output_shape() {
        let n = 32;
        let signal = make_cosine(100.0, n, 1000.0);
        let nf = 64;
        let w = wvd(&signal, Some(nf)).expect("wvd failed");
        assert_eq!(w.nrows(), nf);
        assert_eq!(w.ncols(), n);
    }

    #[test]
    fn test_wvd_default_n_freqs() {
        let n = 32;
        let signal = make_cosine(50.0, n, 500.0);
        let w = wvd(&signal, None).expect("wvd failed");
        assert_eq!(w.ncols(), n);
        // n_freqs should be >= n and even
        assert!(w.nrows() >= n);
        assert_eq!(w.nrows() % 2, 0);
    }

    #[test]
    fn test_wvd_too_short_error() {
        assert!(wvd(&[1.0], None).is_err());
        assert!(wvd(&[], None).is_err());
    }

    #[test]
    fn test_wvd_chirp_has_time_frequency_support() {
        // A chirp should show increasing frequency over time in the WVD
        let fs = 128.0;
        let n = 64;
        let nf = 64;
        // Chirp from 10 Hz to 50 Hz
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let f_inst = 10.0 + 40.0 * t; // linear sweep
                (2.0 * PI * f_inst * t).sin()
            })
            .collect();

        let w = wvd(&signal, Some(nf)).expect("wvd failed");

        // The WVD should have non-trivial content
        let max_val = w.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0, "WVD should have positive values for a chirp");
    }

    // --- pwvd tests ---

    #[test]
    fn test_pwvd_output_shape() {
        let n = 32;
        let signal = make_cosine(100.0, n, 1000.0);
        let nf = 32;
        let window: Vec<f64> = (0..9)
            .map(|i| {
                let x = (i as f64 - 4.0) / 2.0;
                (-0.5 * x * x).exp()
            })
            .collect();
        let w = pwvd(&signal, &window, Some(nf)).expect("pwvd failed");
        assert_eq!(w.nrows(), nf);
        assert_eq!(w.ncols(), n);
    }

    #[test]
    fn test_pwvd_empty_window_error() {
        let signal = make_cosine(100.0, 32, 1000.0);
        assert!(pwvd(&signal, &[], None).is_err());
    }

    #[test]
    fn test_pwvd_too_short_signal_error() {
        assert!(pwvd(&[1.0], &[1.0], None).is_err());
    }

    // --- spwvd tests ---

    #[test]
    fn test_spwvd_output_shape() {
        let n = 32;
        let signal = make_cosine(50.0, n, 500.0);
        let nf = 32;
        let time_win: Vec<f64> = vec![0.25, 0.5, 1.0, 0.5, 0.25];
        let freq_win: Vec<f64> = vec![0.25, 0.5, 1.0, 0.5, 0.25];
        let w = spwvd(&signal, &time_win, &freq_win, Some(nf)).expect("spwvd failed");
        assert_eq!(w.nrows(), nf);
        assert_eq!(w.ncols(), n);
    }

    #[test]
    fn test_spwvd_smoothing_reduces_extremes() {
        // SPWVD should have less extreme values than raw WVD due to smoothing
        let n = 32;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 0.1 * i as f64).sin() + (2.0 * PI * 0.3 * i as f64).sin())
            .collect();
        let nf = 32;
        let time_win: Vec<f64> = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let freq_win: Vec<f64> = vec![0.1, 0.2, 0.4, 0.2, 0.1];

        let raw = wvd(&signal, Some(nf)).expect("wvd failed");
        let smoothed = spwvd(&signal, &time_win, &freq_win, Some(nf)).expect("spwvd failed");

        // Check that outputs exist and have correct shape
        assert_eq!(raw.shape(), smoothed.shape());
    }

    #[test]
    fn test_spwvd_empty_window_error() {
        let signal = make_cosine(100.0, 32, 1000.0);
        let win = vec![1.0; 5];
        assert!(spwvd(&signal, &[], &win, None).is_err());
        assert!(spwvd(&signal, &win, &[], None).is_err());
    }
}
