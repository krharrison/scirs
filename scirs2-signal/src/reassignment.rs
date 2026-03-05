//! Time-Frequency Reassignment Methods
//!
//! Reassignment sharpens time-frequency representations by relocating each
//! STFT coefficient to the centroid of the local energy distribution rather
//! than the geometric centre of the analysis window.  This produces
//! dramatically more readable spectrograms for chirp and AM/FM signals.
//!
//! # Theory
//!
//! Given the STFT `S(t,ω) = ∫ x(τ) h(τ-t) e^{-jωτ} dτ`, three auxiliary
//! STFTs are formed using the original window `h`, the time-derivative window
//! `Dh(n) = h(n) · n`, and the frequency-derivative window `Th(n) = h'(n)`:
//!
//! ```text
//! t̂(t,ω)  = t + Re{ S_{Th}(t,ω) / S_h(t,ω) }          (group delay)
//! f̂(t,ω)  = ω - Im{ S_{Dh}(t,ω) / S_h(t,ω) } / (2π)   (inst. freq.)
//! ```
//!
//! The reassigned spectrogram accumulates `|S_h(t,ω)|²` at `(t̂, f̂)`.
//!
//! # References
//!
//! - Auger & Flandrin (1995), IEEE Trans. Signal Process. 43(5), 1068–1089.
//! - Fulop & Fitz (2006), JASA 119(1), 360–371.
//! - Chassande-Mottin, Auger & Flandrin (2003), *Reassignment*, in
//!   *Time-Frequency Analysis*, ISTE.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the STFT-based reassignment.
///
/// Returns three arrays of shape `(n_freq_bins, n_frames)`:
///
/// 1. `magnitude` – `|STFT(t,ω)|` (not squared).
/// 2. `reassigned_time` – Reassigned time coordinate in seconds for every bin.
/// 3. `reassigned_freq` – Reassigned frequency (instantaneous frequency) in Hz.
///
/// # Arguments
///
/// * `signal`   – Real-valued input signal.
/// * `window`   – Analysis window (e.g. Hann).  Length determines the STFT
///   resolution.
/// * `hop`      – Frame shift in samples (>0).
/// * `fft_size` – FFT size (≥ `window.len()`).  Pass `None` to use the next
///   power-of-two ≥ `window.len()`.
/// * `fs`       – Sampling frequency in Hz (used to convert axes to seconds /
///   Hz).
///
/// # Errors
///
/// Returns `SignalError::ValueError` if `signal` is shorter than `window`,
/// `hop_size` is zero, or `fft_size < window.len()`.
///
/// # Example
///
/// ```
/// use scirs2_signal::reassignment::stft_reassignment;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 512_usize;
/// let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()).collect();
/// let window: Vec<f64> = (0..64).map(|k| {
///     0.5 * (1.0 - (2.0 * PI * k as f64 / 63.0).cos())
/// }).collect();
/// let (mag, t_hat, f_hat) = stft_reassignment(&signal, &window, 16, None, fs).expect("operation should succeed");
/// assert_eq!(mag.shape(), t_hat.shape());
/// assert_eq!(mag.shape(), f_hat.shape());
/// ```
pub fn stft_reassignment(
    signal: &[f64],
    window: &[f64],
    hop: usize,
    fft_size: Option<usize>,
    fs: f64,
) -> SignalResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let win_len = window.len();
    if signal.len() < win_len {
        return Err(SignalError::ValueError(
            "signal must be at least as long as the window".to_string(),
        ));
    }
    if hop == 0 {
        return Err(SignalError::InvalidArgument(
            "hop must be > 0".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "fs must be positive".to_string(),
        ));
    }

    let nfft = match fft_size {
        Some(n) if n >= win_len => n,
        Some(n) => {
            return Err(SignalError::ValueError(format!(
                "fft_size ({n}) must be >= window length ({win_len})"
            )))
        }
        None => next_power_of_two(win_len),
    };
    let nfft = if nfft % 2 != 0 { nfft + 1 } else { nfft };

    // Build derivative windows.
    // Time-ramped window:  Th[k] = window[k] * k    (group delay)
    let t_window: Vec<f64> = window
        .iter()
        .enumerate()
        .map(|(k, &w)| w * (k as f64 - (win_len - 1) as f64 / 2.0))
        .collect();

    // Frequency-derivative window: Dh[k] = window'[k] (inst. freq.)
    let mut d_window = vec![0.0f64; win_len];
    if win_len >= 2 {
        d_window[0] = window[1] - window[0];
        d_window[win_len - 1] = window[win_len - 1] - window[win_len - 2];
        for k in 1..win_len - 1 {
            d_window[k] = (window[k + 1] - window[k - 1]) / 2.0;
        }
    }

    let n_frames = (signal.len().saturating_sub(win_len)) / hop + 1;
    let n_bins = nfft / 2 + 1;

    let mut magnitude = Array2::<f64>::zeros((n_bins, n_frames));
    let mut reassigned_time = Array2::<f64>::zeros((n_bins, n_frames));
    let mut reassigned_freq = Array2::<f64>::zeros((n_bins, n_frames));

    let threshold = f64::EPSILON * 1e6;

    for frame in 0..n_frames {
        let start = frame * hop;
        let end = (start + win_len).min(signal.len());
        let actual = end - start;

        let mut buf_h = vec![0.0f64; nfft];
        let mut buf_th = vec![0.0f64; nfft];
        let mut buf_dh = vec![0.0f64; nfft];

        for k in 0..actual {
            let x = signal[start + k];
            buf_h[k] = x * window[k];
            buf_th[k] = x * t_window[k];
            buf_dh[k] = x * d_window[k];
        }

        let stft_h = real_fft(&buf_h, nfft)?;
        let stft_th = real_fft(&buf_th, nfft)?;
        let stft_dh = real_fft(&buf_dh, nfft)?;

        let center_sample = start as f64 + (win_len as f64 - 1.0) / 2.0;

        for f in 0..n_bins {
            let sh = stft_h[f];
            let mag_sq = sh.norm_sqr();
            let mag = mag_sq.sqrt();
            magnitude[[f, frame]] = mag;

            if mag_sq < threshold {
                // No reassignment; keep original coordinates
                reassigned_time[[f, frame]] = center_sample / fs;
                reassigned_freq[[f, frame]] = f as f64 * fs / nfft as f64;
            } else {
                // Group delay  → reassigned time
                let sth = stft_th[f];
                let t_hat = center_sample + (sh.conj() * sth).re / mag_sq;
                reassigned_time[[f, frame]] = t_hat / fs;

                // Instantaneous frequency  → reassigned frequency
                let sdh = stft_dh[f];
                let f_nominal = f as f64 * fs / nfft as f64;
                let if_correction = (sh.conj() * sdh).im / mag_sq;
                reassigned_freq[[f, frame]] = f_nominal - if_correction * fs / (2.0 * PI);
            }
        }
    }

    Ok((magnitude, reassigned_time, reassigned_freq))
}

/// Compute a sharpened reassigned spectrogram image.
///
/// Identical to calling [`stft_reassignment`] and then accumulating the
/// squared magnitude at each reassigned `(t̂, f̂)` coordinate.  The output
/// shape is `(n_freq_bins, n_frames)` – the same grid as a standard
/// spectrogram.
///
/// # Arguments
///
/// * `signal`   – Real-valued input.
/// * `window`   – Analysis window.
/// * `hop`      – Frame hop in samples.
/// * `fft_size` – FFT size (or `None` for auto).
/// * `fs`       – Sampling frequency.
pub fn reassigned_spectrogram(
    signal: &[f64],
    window: &[f64],
    hop: usize,
    fft_size: Option<usize>,
    fs: f64,
) -> SignalResult<Array2<f64>> {
    let (mag, t_hat, f_hat) = stft_reassignment(signal, window, hop, fft_size, fs)?;
    let (n_bins, n_frames) = (mag.shape()[0], mag.shape()[1]);
    let nfft = match fft_size {
        Some(n) => n,
        None => next_power_of_two(window.len()),
    };
    let nfft = if nfft % 2 != 0 { nfft + 1 } else { nfft };

    // Build output grid axes
    let dt = hop as f64 / fs;
    let df = fs / nfft as f64;
    let t0 = (window.len() as f64 - 1.0) / 2.0 / fs;

    let mut output = Array2::<f64>::zeros((n_bins, n_frames));

    for f in 0..n_bins {
        for t in 0..n_frames {
            let energy = mag[[f, t]] * mag[[f, t]];
            if energy < f64::EPSILON {
                continue;
            }

            // Map reassigned coordinates to nearest bin indices
            let rt = t_hat[[f, t]];
            let rf = f_hat[[f, t]];

            let t_bin = ((rt - t0) / dt).round() as i64;
            let f_bin = (rf / df).round() as i64;

            if t_bin >= 0
                && t_bin < n_frames as i64
                && f_bin >= 0
                && f_bin < n_bins as i64
            {
                output[[f_bin as usize, t_bin as usize]] += energy;
            }
        }
    }

    Ok(output)
}

/// Multitaper reassigned spectrogram.
///
/// Applies DPSS tapers and averages the reassigned spectrogram across tapers.
/// The multitaper approach reduces variance while the reassignment sharpens
/// the time-frequency representation.
///
/// # Arguments
///
/// * `signal`    – Real-valued input signal.
/// * `n_tapers`  – Number of DPSS tapers (≥1, must satisfy `n_tapers ≤ 2*nw`
///   where `nw = (n_tapers + 1) / 2`).
/// * `hop`       – Frame shift in samples.
/// * `fft_size`  – FFT size.  Pass `None` for automatic selection.
/// * `win_len`   – Window / taper length.
/// * `fs`        – Sampling frequency.
///
/// # Returns
///
/// Averaged reassigned spectrogram of shape `(n_freq_bins, n_frames)`.
pub fn multitaper_reassignment(
    signal: &[f64],
    n_tapers: usize,
    hop: usize,
    fft_size: Option<usize>,
    win_len: usize,
    fs: f64,
) -> SignalResult<Array2<f64>> {
    if n_tapers == 0 {
        return Err(SignalError::ValueError(
            "n_tapers must be at least 1".to_string(),
        ));
    }

    // Determine time-bandwidth product: choose smallest nw s.t. 2*nw >= n_tapers
    let nw = (n_tapers as f64 / 2.0).ceil().max(1.0);

    // Get DPSS tapers using the multitaper module
    let (tapers, _): (scirs2_core::ndarray::Array2<f64>, _) = crate::multitaper::dpss(win_len, nw, n_tapers, false)?;

    let nfft = fft_size.unwrap_or_else(|| next_power_of_two(win_len));
    let nfft = if nfft < win_len {
        next_power_of_two(win_len)
    } else {
        nfft
    };
    let nfft = if nfft % 2 != 0 { nfft + 1 } else { nfft };

    let n_bins = nfft / 2 + 1;
    let n_frames = (signal.len().saturating_sub(win_len)) / hop + 1;

    let mut accumulated = Array2::<f64>::zeros((n_bins, n_frames));

    for taper_idx in 0..n_tapers {
        let taper: Vec<f64> = tapers.row(taper_idx).iter().cloned().collect();
        let rs = reassigned_spectrogram(signal, &taper, hop, Some(nfft), fs)?;
        for f in 0..n_bins {
            for t in 0..n_frames {
                accumulated[[f, t]] += rs[[f, t]];
            }
        }
    }

    // Average over tapers
    let inv = 1.0 / n_tapers as f64;
    for v in accumulated.iter_mut() {
        *v *= inv;
    }

    Ok(accumulated)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the one-sided FFT of a real-valued buffer (already padded to `nfft`).
/// Returns the full complex spectrum of length `nfft` (not just the one-sided
/// half) so that the caller can index freely.
fn real_fft(buf: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    let spectrum = scirs2_fft::fft(buf, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;
    Ok(spectrum)
}

/// Next power of two ≥ n.
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn hann_window(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos()))
            .collect()
    }

    fn sinusoid(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_stft_reassignment_shape() {
        let fs = 1000.0;
        let signal = sinusoid(100.0, fs, 512);
        let window = hann_window(64);
        let (mag, t_hat, f_hat) = stft_reassignment(&signal, &window, 16, None, fs).expect("unexpected None or Err");
        assert_eq!(mag.shape(), t_hat.shape());
        assert_eq!(mag.shape(), f_hat.shape());
        // n_bins = nfft/2 + 1; nfft = 64 -> 33
        assert_eq!(mag.shape()[0], 33);
    }

    #[test]
    fn test_stft_reassignment_non_negative_magnitude() {
        let fs = 1000.0;
        let signal = sinusoid(50.0, fs, 256);
        let window = hann_window(32);
        let (mag, _, _) = stft_reassignment(&signal, &window, 8, None, fs).expect("unexpected None or Err");
        assert!(mag.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_stft_reassignment_freq_in_range() {
        let fs = 1000.0;
        let signal = sinusoid(100.0, fs, 512);
        let window = hann_window(64);
        let nfft = 64;
        let (mag, _, f_hat) = stft_reassignment(&signal, &window, 16, Some(nfft), fs).expect("unexpected None or Err");
        // All reassigned frequencies with non-negligible magnitude should be in [0, fs/2]
        for f in 0..mag.shape()[0] {
            for t in 0..mag.shape()[1] {
                if mag[[f, t]] > 1e-6 {
                    let fv = f_hat[[f, t]];
                    assert!(
                        fv >= -fs * 0.1 && fv <= fs * 0.6,
                        "Reassigned freq {fv} out of expected range"
                    );
                }
            }
        }
    }

    #[test]
    fn test_stft_reassignment_instantaneous_frequency() {
        // A pure sinusoid at f0 Hz should have most reassigned frequency near f0
        let fs = 4000.0;
        let f0 = 400.0;
        let signal = sinusoid(f0, fs, 2048);
        let window = hann_window(128);
        let (mag, _, f_hat) = stft_reassignment(&signal, &window, 32, None, fs).expect("unexpected None or Err");

        // Find bin with maximum magnitude
        let mut best_f = 0.0;
        let mut best_mag = 0.0;
        for f in 0..mag.shape()[0] {
            for t in 0..mag.shape()[1] {
                if mag[[f, t]] > best_mag {
                    best_mag = mag[[f, t]];
                    best_f = f_hat[[f, t]];
                }
            }
        }
        // Reassigned frequency should be within 5% of f0
        let rel_err = (best_f - f0).abs() / f0;
        assert!(
            rel_err < 0.05,
            "Reassigned IF {best_f:.1} Hz, expected near {f0} Hz (rel_err={rel_err:.3})"
        );
    }

    #[test]
    fn test_stft_reassignment_error_signal_too_short() {
        let signal = vec![1.0, 2.0, 3.0];
        let window = hann_window(32);
        assert!(stft_reassignment(&signal, &window, 8, None, 1000.0).is_err());
    }

    #[test]
    fn test_stft_reassignment_error_zero_hop() {
        let signal: Vec<f64> = vec![0.0; 256];
        let window = hann_window(32);
        assert!(stft_reassignment(&signal, &window, 0, None, 1000.0).is_err());
    }

    #[test]
    fn test_stft_reassignment_error_fft_too_small() {
        let signal: Vec<f64> = vec![0.0; 256];
        let window = hann_window(64);
        // fft_size < win_len should fail
        assert!(stft_reassignment(&signal, &window, 16, Some(32), 1000.0).is_err());
    }

    #[test]
    fn test_reassigned_spectrogram_shape_and_non_negative() {
        let fs = 1000.0;
        let signal = sinusoid(100.0, fs, 512);
        let window = hann_window(64);
        let rs = reassigned_spectrogram(&signal, &window, 16, None, fs).expect("failed to create rs");
        assert!(rs.iter().all(|&v| v >= 0.0));
        assert_eq!(rs.shape()[0], 33); // nfft/2+1 = 64/2+1
    }

    #[test]
    fn test_reassigned_spectrogram_energy_preserved() {
        // Total energy in reassigned spectrogram should roughly equal total energy
        // in the standard spectrogram (energy is just redistributed, not changed)
        let fs = 1000.0;
        let signal = sinusoid(100.0, fs, 512);
        let window = hann_window(64);
        let (mag, _, _) = stft_reassignment(&signal, &window, 16, None, fs).expect("unexpected None or Err");
        let rs = reassigned_spectrogram(&signal, &window, 16, None, fs).expect("failed to create rs");

        let orig_energy: f64 = mag.iter().map(|&v| v * v).sum();
        let rs_energy: f64 = rs.iter().sum();

        // Energies may differ slightly due to boundary effects but should be same order
        if orig_energy > 1e-10 {
            let ratio = rs_energy / orig_energy;
            assert!(
                ratio > 0.5 && ratio < 2.0,
                "Energy ratio {ratio:.3} out of expected range [0.5, 2.0]"
            );
        }
    }

    #[test]
    fn test_multitaper_reassignment_shape() {
        let fs = 1000.0;
        let signal = sinusoid(100.0, fs, 512);
        let rs = multitaper_reassignment(&signal, 4, 16, None, 64, fs).expect("failed to create rs");
        assert_eq!(rs.shape()[0], 33); // nfft=64 -> 33 bins
        assert!(rs.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_multitaper_reassignment_error_zero_tapers() {
        let signal: Vec<f64> = vec![0.0; 256];
        assert!(multitaper_reassignment(&signal, 0, 16, None, 64, 1000.0).is_err());
    }
}
