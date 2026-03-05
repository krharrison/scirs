//! Extended cepstral analysis: inverse complex cepstrum, minimum phase reconstruction,
//! liftering (cepstral windowing), and cepstral pitch detection.
//!
//! This module supplements `cepstral.rs` (which provides `real_cepstrum`,
//! `complex_cepstrum`, and MFCC functionality) with higher-level operations:
//!
//! - **Inverse complex cepstrum** – reconstruct a signal from its complex cepstrum.
//! - **Minimum-phase reconstruction** – build the minimum-phase signal whose
//!   magnitude spectrum matches the given one.
//! - **Liftering** – apply a cepstral window (sinusoidal or exponential) to
//!   emphasise or de-emphasise certain quefrency ranges.
//! - **Cepstral pitch detection** – find the fundamental period of a voiced
//!   speech/audio frame via the real cepstrum peak.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Inverse complex cepstrum
// ---------------------------------------------------------------------------

/// Reconstruct a signal from its complex cepstrum and a delay value.
///
/// Performs the inverse operation of `complex_cepstrum`:
/// ```text
/// X̂ = FFT(cepstrum)          # complex log-spectrum
/// x = IFFT(exp(X̂))           # back to time domain
/// ```
/// The `ndelay` parameter accounts for a linear phase ramp that was removed
/// during forward cepstrum computation. Positive `ndelay` means the original
/// signal was delayed (non-minimum phase); negative indicates it was advanced.
///
/// # Arguments
///
/// * `cepstrum` - Complex cepstrum coefficients (real part, length N)
/// * `ndelay` - Integer delay that was subtracted during `complex_cepstrum`.
///              Pass 0 if the delay is unknown or not applicable.
///
/// # Returns
///
/// Reconstructed time-domain signal of length N.
///
/// # Examples
///
/// ```
/// use scirs2_signal::cepstrum::inverse_complex_cepstrum;
///
/// let cepstrum = vec![0.1, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let reconstructed = inverse_complex_cepstrum(&cepstrum, 0)
///     .expect("inverse_complex_cepstrum failed");
/// assert_eq!(reconstructed.len(), 8);
/// ```
pub fn inverse_complex_cepstrum(
    cepstrum: &[f64],
    ndelay: isize,
) -> SignalResult<Vec<f64>> {
    if cepstrum.is_empty() {
        return Err(SignalError::ValueError(
            "Cepstrum must not be empty".into(),
        ));
    }

    let n = cepstrum.len();

    // Build the complex-valued cepstrum with optional linear phase ramp for ndelay
    // The complex log-spectrum is obtained by FFT of the real cepstrum,
    // and we add back j*2π*ndelay*k/N to account for the removed delay.
    let cepstrum_complex: Vec<Complex64> = cepstrum
        .iter()
        .map(|&c| Complex64::new(c, 0.0))
        .collect();

    // Forward FFT of cepstrum to get the complex log-spectrum
    let log_spectrum = scirs2_fft::fft(&cepstrum, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Add linear phase correction for ndelay
    // Phase correction: theta[k] = 2*pi*ndelay*k/N
    let log_spectrum_corrected: Vec<Complex64> = log_spectrum
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            if ndelay == 0 {
                c
            } else {
                let phase = 2.0 * PI * ndelay as f64 * k as f64 / n as f64;
                // Rotate the imaginary part by adding the phase correction
                // log X_corrected = log|X| + j*(unwrapped_phase + 2*pi*ndelay*k/N)
                // We need to combine: c = re + j*im
                // Adding phase: c_new = c + j*phase_correction
                Complex64::new(c.re, c.im + phase)
            }
        })
        .collect();

    // Exponentiate: X[k] = exp(log_spectrum[k])
    let spectrum: Vec<Complex64> = log_spectrum_corrected
        .iter()
        .map(|&c| {
            // exp(a + jb) = e^a * (cos(b) + j*sin(b))
            let mag = c.re.exp();
            Complex64::new(mag * c.im.cos(), mag * c.im.sin())
        })
        .collect();

    // IFFT to obtain time-domain signal
    let result = scirs2_fft::ifft(&spectrum, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    // Return real part (imaginary should be near zero for a valid real signal)
    Ok(result.iter().map(|c| c.re).collect())
}

// ---------------------------------------------------------------------------
// Minimum-phase reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct the minimum-phase signal whose magnitude spectrum matches
/// the provided magnitude values.
///
/// Uses the cepstral method:
/// 1. Take the log of the magnitude spectrum.
/// 2. Compute the real cepstrum (IFFT of log-magnitude).
/// 3. Apply the minimum-phase lifter (double the causal part).
/// 4. Exponentiate and IFFT back to time domain.
///
/// The result is the minimum-phase FIR filter whose frequency response
/// magnitude equals the input magnitude spectrum.
///
/// # Arguments
///
/// * `magnitude` - One-sided or full magnitude spectrum (length must be ≥ 2).
///   Typically this is the output of `|FFT(x)[:]|` for a length-N signal.
///
/// # Returns
///
/// Minimum-phase time-domain signal of the same length as `magnitude`.
///
/// # Examples
///
/// ```
/// use scirs2_signal::cepstrum::min_phase_reconstruction;
///
/// let magnitude = vec![1.0, 0.8, 0.5, 0.2, 0.1, 0.2, 0.5, 0.8];
/// let min_phase = min_phase_reconstruction(&magnitude)
///     .expect("min_phase_reconstruction failed");
/// assert_eq!(min_phase.len(), 8);
/// ```
pub fn min_phase_reconstruction(magnitude: &[f64]) -> SignalResult<Vec<f64>> {
    if magnitude.len() < 2 {
        return Err(SignalError::ValueError(
            "Magnitude spectrum must have at least 2 elements".into(),
        ));
    }

    let n = magnitude.len();

    // Step 1: log-magnitude spectrum (guard against log(0))
    let log_mag: Vec<Complex64> = magnitude
        .iter()
        .map(|&m| Complex64::new(m.max(1e-20).ln(), 0.0))
        .collect();

    // Step 2: IFFT(log-magnitude) = real cepstrum
    let cepstrum_complex = scirs2_fft::ifft(&log_mag, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;
    let cepstrum: Vec<f64> = cepstrum_complex.iter().map(|c| c.re).collect();

    // Step 3: Minimum-phase lifter (double causal part, zero anti-causal)
    // For N-point DFT:
    //   c_min[0]     = c[0]
    //   c_min[k]     = 2*c[k]   for 1 <= k < N/2
    //   c_min[N/2]   = c[N/2]   (only if N is even)
    //   c_min[k]     = 0        for N/2 < k < N
    let half = n / 2;
    let mut c_min: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];
    c_min[0] = Complex64::new(cepstrum[0], 0.0);
    for k in 1..half {
        c_min[k] = Complex64::new(2.0 * cepstrum[k], 0.0);
    }
    if n % 2 == 0 && half < n {
        c_min[half] = Complex64::new(cepstrum[half], 0.0);
    }

    // Step 4: FFT of c_min gives the complex minimum-phase log-spectrum
    let log_spec_min = scirs2_fft::fft(&c_min, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Exponentiate: H_min[k] = exp(log_spec_min[k])
    let h_min: Vec<Complex64> = log_spec_min
        .iter()
        .map(|&c| {
            let mag = c.re.exp();
            Complex64::new(mag * c.im.cos(), mag * c.im.sin())
        })
        .collect();

    // IFFT to time domain
    let result = scirs2_fft::ifft(&h_min, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    Ok(result.iter().map(|c| c.re).collect())
}

// ---------------------------------------------------------------------------
// Liftering (cepstral windowing)
// ---------------------------------------------------------------------------

/// Lifter type for cepstral windowing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifterType {
    /// Sinusoidal lifter: `w[n] = 1 + (L/2) * sin(pi * n / L)` for `n < lifter_order`.
    Sinusoidal,
    /// Exponential lifter: `w[n] = exp(n / lifter_order)` for the first `lifter_order` coefficients.
    Exponential,
}

/// Apply a cepstral window (lifter) to cepstral coefficients.
///
/// Liftering emphasises higher-order cepstral coefficients (which carry fine
/// spectral structure) and de-emphasises lower-order ones (which carry the
/// spectral envelope), or vice versa depending on the window type.
///
/// - **Sinusoidal lifter** (L = `lifter_order`):
///   `w[n] = 1 + (L/2) * sin(π * n / L)` for n in [0, lifter_order),
///   and 1 elsewhere.
/// - **Exponential lifter**:
///   `w[n] = n / lifter_order` for n in [0, lifter_order),
///   and 1 elsewhere. (normalised exponential ramp).
///
/// # Arguments
///
/// * `cepstrum` - Input cepstral coefficients
/// * `lifter_order` - Order of the lifter (typically 22 for speech)
/// * `ltype` - Lifter type
///
/// # Returns
///
/// Liftered cepstral coefficients (same length as input).
///
/// # Examples
///
/// ```
/// use scirs2_signal::cepstrum::{lifter, LifterType};
///
/// let cepstrum = vec![1.0, 0.5, 0.3, 0.1, 0.05, 0.02];
/// let liftered = lifter(&cepstrum, 3, LifterType::Sinusoidal);
/// assert_eq!(liftered.len(), 6);
/// ```
pub fn lifter(cepstrum: &[f64], lifter_order: usize, ltype: LifterType) -> Vec<f64> {
    if cepstrum.is_empty() || lifter_order == 0 {
        return cepstrum.to_vec();
    }

    let n = cepstrum.len();
    let l = lifter_order as f64;

    (0..n)
        .map(|i| {
            let weight = if i < lifter_order {
                match ltype {
                    LifterType::Sinusoidal => {
                        1.0 + (l / 2.0) * (PI * i as f64 / l).sin()
                    }
                    LifterType::Exponential => {
                        // Ramp from 0 to 1 over lifter_order coefficients
                        if lifter_order > 1 {
                            i as f64 / (lifter_order - 1) as f64
                        } else {
                            1.0
                        }
                    }
                }
            } else {
                1.0 // identity outside the window
            };
            cepstrum[i] * weight
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Cepstral pitch detection
// ---------------------------------------------------------------------------

/// Detect the fundamental frequency (pitch) of a voiced frame using cepstral analysis.
///
/// The algorithm:
/// 1. Compute the real cepstrum of the frame.
/// 2. Apply a rectangular lifter to the range corresponding to `[f_min, f_max]`.
/// 3. Find the quefrency peak within `[1/f_max, 1/f_min]` (in samples).
/// 4. Convert the quefrency to frequency in Hz.
///
/// This method works best on short frames (10–30 ms) from voiced speech or
/// audio with a clear fundamental frequency.
///
/// # Arguments
///
/// * `x` - Input signal frame (typically 10–30 ms of audio)
/// * `fs` - Sampling frequency (Hz)
/// * `f_min` - Minimum expected fundamental frequency (Hz), e.g. 80 Hz
/// * `f_max` - Maximum expected fundamental frequency (Hz), e.g. 400 Hz
///
/// # Returns
///
/// Estimated fundamental frequency in Hz.
///
/// # Examples
///
/// ```
/// use scirs2_signal::cepstrum::cepstral_pitch_detect;
/// use std::f64::consts::PI;
///
/// let fs = 16000.0_f64;
/// let f0 = 150.0_f64; // 150 Hz fundamental
/// let n = 1024;
/// // Simulate voiced speech with harmonic series
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / fs;
///         (2.0 * PI * f0 * t).sin()
///             + 0.5 * (2.0 * PI * 2.0 * f0 * t).sin()
///             + 0.25 * (2.0 * PI * 3.0 * f0 * t).sin()
///     })
///     .collect();
///
/// let pitch = cepstral_pitch_detect(&signal, fs, 80.0, 400.0)
///     .expect("pitch detection failed");
/// // Detected pitch should be near 150 Hz
/// assert!((pitch - f0).abs() < 5.0, "Expected ~150 Hz, got {}", pitch);
/// ```
pub fn cepstral_pitch_detect(
    x: &[f64],
    fs: f64,
    f_min: f64,
    f_max: f64,
) -> SignalResult<f64> {
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".into(),
        ));
    }
    if f_min <= 0.0 || f_max <= f_min {
        return Err(SignalError::ValueError(
            "Frequency bounds must satisfy 0 < f_min < f_max".into(),
        ));
    }
    if f_max > fs / 2.0 {
        return Err(SignalError::ValueError(
            "f_max must not exceed the Nyquist frequency (fs/2)".into(),
        ));
    }

    // Compute real cepstrum
    let cepstrum = crate::cepstral::real_cepstrum(x)?;
    let n = cepstrum.len();

    // Quefrency search range in samples: [fs/f_max, fs/f_min]
    let q_min = (fs / f_max).ceil() as usize;
    let q_max = (fs / f_min).floor() as usize;

    if q_min >= q_max || q_min >= n {
        return Err(SignalError::ValueError(format!(
            "Quefrency search range [{}, {}] is invalid for signal length {}",
            q_min, q_max, n
        )));
    }

    let q_max_clamped = q_max.min(n - 1);

    if q_min > q_max_clamped {
        return Err(SignalError::ValueError(
            "Quefrency search window is empty after clamping".into(),
        ));
    }

    // Find peak in the quefrency range [q_min, q_max_clamped]
    let (peak_idx, _) = cepstrum[q_min..=q_max_clamped]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| SignalError::ComputationError("Empty quefrency window".into()))?;

    let quefrency_samples = (q_min + peak_idx) as f64;
    let pitch_hz = fs / quefrency_samples;

    Ok(pitch_hz)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // --- inverse_complex_cepstrum tests ---

    #[test]
    fn test_inverse_complex_cepstrum_length() {
        let cepstrum = vec![0.1, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = inverse_complex_cepstrum(&cepstrum, 0)
            .expect("inverse failed");
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_inverse_complex_cepstrum_empty_error() {
        assert!(inverse_complex_cepstrum(&[], 0).is_err());
    }

    #[test]
    fn test_inverse_complex_cepstrum_with_delay() {
        let cepstrum = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let r0 = inverse_complex_cepstrum(&cepstrum, 0).expect("failed");
        let r1 = inverse_complex_cepstrum(&cepstrum, 1).expect("failed");
        // With ndelay=1 the result should differ from ndelay=0
        let diff: f64 = r0.iter().zip(r1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-10, "Results with different delays should differ");
    }

    // --- min_phase_reconstruction tests ---

    #[test]
    fn test_min_phase_reconstruction_length() {
        let magnitude = vec![1.0, 0.8, 0.5, 0.2, 0.1, 0.2, 0.5, 0.8];
        let result = min_phase_reconstruction(&magnitude)
            .expect("min_phase failed");
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_min_phase_reconstruction_too_short_error() {
        assert!(min_phase_reconstruction(&[1.0]).is_err());
    }

    #[test]
    fn test_min_phase_reconstruction_constant_magnitude() {
        // A flat (white noise) magnitude spectrum should produce a delta-like min-phase response
        let n = 16;
        let magnitude = vec![1.0_f64; n];
        let result = min_phase_reconstruction(&magnitude)
            .expect("min_phase failed");
        // The energy should be concentrated early (minimum-phase property)
        let total_energy: f64 = result.iter().map(|&x| x * x).sum();
        let early_energy: f64 = result[..4].iter().map(|&x| x * x).sum();
        assert!(
            early_energy / total_energy > 0.8,
            "Min-phase filter should concentrate energy early"
        );
    }

    // --- lifter tests ---

    #[test]
    fn test_lifter_output_length() {
        let cepstrum = vec![1.0, 0.5, 0.3, 0.1, 0.05, 0.02];
        let liftered = lifter(&cepstrum, 3, LifterType::Sinusoidal);
        assert_eq!(liftered.len(), 6);
    }

    #[test]
    fn test_lifter_identity_beyond_order() {
        let cepstrum = vec![1.0, 0.5, 0.3, 0.1, 0.05, 0.02];
        let liftered = lifter(&cepstrum, 2, LifterType::Sinusoidal);
        // Beyond lifter_order=2, coefficients should be unchanged
        for i in 2..cepstrum.len() {
            assert_relative_eq!(liftered[i], cepstrum[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lifter_sinusoidal() {
        // Zeroth coeff: w[0] = 1 + L/2 * sin(0) = 1
        // First coeff: w[1] = 1 + L/2 * sin(pi/L)
        let cepstrum = vec![1.0, 1.0, 1.0, 1.0];
        let l = 4_usize;
        let liftered = lifter(&cepstrum, l, LifterType::Sinusoidal);
        let l_f = l as f64;
        for i in 0..l {
            let expected = 1.0 + (l_f / 2.0) * (PI * i as f64 / l_f).sin();
            assert_relative_eq!(liftered[i], expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_lifter_exponential_first_is_zero() {
        // Exponential lifter: w[0] = 0/(L-1) = 0
        let cepstrum = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let liftered = lifter(&cepstrum, 4, LifterType::Exponential);
        assert_relative_eq!(liftered[0], 0.0, epsilon = 1e-12);
        // w[3] = 3/3 = 1
        assert_relative_eq!(liftered[3], 1.0, epsilon = 1e-12);
        // w[4] (beyond order) = 1
        assert_relative_eq!(liftered[4], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_lifter_empty_cepstrum() {
        let liftered = lifter(&[], 5, LifterType::Sinusoidal);
        assert!(liftered.is_empty());
    }

    #[test]
    fn test_lifter_zero_order() {
        let cepstrum = vec![1.0, 2.0, 3.0];
        let liftered = lifter(&cepstrum, 0, LifterType::Sinusoidal);
        assert_eq!(liftered, cepstrum);
    }

    // --- cepstral_pitch_detect tests ---

    #[test]
    fn test_cepstral_pitch_detect_tone() {
        let fs = 16000.0;
        let f0 = 150.0;
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * f0 * t).sin()
                    + 0.5 * (2.0 * PI * 2.0 * f0 * t).sin()
                    + 0.25 * (2.0 * PI * 3.0 * f0 * t).sin()
            })
            .collect();

        let pitch = cepstral_pitch_detect(&signal, fs, 80.0, 400.0)
            .expect("pitch detection failed");

        // Should detect near 150 Hz (within 10 Hz tolerance for frame-based method)
        assert!(
            (pitch - f0).abs() < 10.0,
            "Expected pitch ~{} Hz, got {} Hz",
            f0,
            pitch
        );
    }

    #[test]
    fn test_cepstral_pitch_detect_empty_error() {
        assert!(cepstral_pitch_detect(&[], 16000.0, 80.0, 400.0).is_err());
    }

    #[test]
    fn test_cepstral_pitch_detect_invalid_fs_error() {
        let signal = vec![0.0; 256];
        assert!(cepstral_pitch_detect(&signal, 0.0, 80.0, 400.0).is_err());
    }

    #[test]
    fn test_cepstral_pitch_detect_invalid_freqs_error() {
        let signal = vec![0.0; 256];
        assert!(cepstral_pitch_detect(&signal, 16000.0, 400.0, 80.0).is_err()); // f_min > f_max
        assert!(cepstral_pitch_detect(&signal, 16000.0, 0.0, 400.0).is_err()); // f_min = 0
    }
}
