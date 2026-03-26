//! Frequency Domain Decomposition (FDD) for Operational Modal Analysis.
//!
//! FDD identifies structural modes by:
//! 1. Estimating the cross-power spectral density (CPSD) matrix via Welch's method.
//! 2. Performing an SVD at every discrete frequency line.
//! 3. Peak-picking from the first singular value curve.
//! 4. Extracting the modal frequency (peak location) and mode shape (first left
//!    singular vector at the peak).
//!
//! ## References
//! - Brincker, R., Zhang, L. & Andersen, P. (2000). "Modal identification from
//!   ambient responses using frequency domain decomposition." Proc. IMAC 18.
//! - Brincker, R. & Ventura, C. (2015). *Introduction to Operational Modal Analysis.*
//!   Wiley.

use super::types::{ModalMode, OmaConfig, OmaMethod, OmaResult};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Complex number type used internally (real, imag) as a pair.
type Cplx = (f64, f64);

/// Multiply two complex numbers.
#[inline]
fn cmul(a: Cplx, b: Cplx) -> Cplx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// Complex conjugate.
#[inline]
fn cconj(a: Cplx) -> Cplx {
    (a.0, -a.1)
}

/// Compute the DFT (decimation-in-time Cooley-Tukey) for power-of-two lengths.
/// Returns complex spectrum of length `n`.
fn fft_radix2(signal: &[f64]) -> Vec<Cplx> {
    let n = signal.len();
    // If not a power of two, zero-pad to next power of two
    let n_fft = next_pow2(n);
    let mut buf: Vec<Cplx> = signal
        .iter()
        .map(|&v| (v, 0.0))
        .chain(std::iter::repeat((0.0_f64, 0.0_f64)))
        .take(n_fft)
        .collect();
    fft_in_place(&mut buf);
    buf
}

/// In-place iterative Cooley-Tukey FFT (power-of-two size).
fn fft_in_place(buf: &mut Vec<Cplx>) {
    let n = buf.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let bits = (n as f64).log2() as usize;
    for i in 0..n {
        let rev = bit_reverse(i, bits);
        if rev > i {
            buf.swap(i, rev);
        }
    }

    // Butterfly stages
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wlen = (angle.cos(), angle.sin());
        for i in (0..n).step_by(len) {
            let mut w: Cplx = (1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let t = cmul(w, buf[i + j + half]);
                buf[i + j] = (u.0 + t.0, u.1 + t.1);
                buf[i + j + half] = (u.0 - t.0, u.1 - t.1);
                w = cmul(w, wlen);
            }
        }
        len <<= 1;
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0_usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1_usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Cross-PSD via Welch
// ---------------------------------------------------------------------------

/// Estimate the cross-power spectral density matrix at each frequency bin
/// using Welch's averaged periodogram method with a Hann window.
///
/// # Arguments
/// * `data` — shape `[n_samples, n_channels]`
/// * `n_seg` — samples per segment (FFT length; zero-padded to next power of 2)
/// * `overlap` — fractional overlap in `[0, 1)`
///
/// # Returns
/// `(freq_bins, cpsd_stack)` where `cpsd_stack` has shape
/// `[n_freq_bins, n_channels, n_channels]` (complex as real/imag interleaved).
///
/// The returned array stores Re and Im of CPSD at `[k, i, j]` in positions
/// `[k, i, j]` (Re) and an accompanying imaginary array.
#[allow(clippy::type_complexity)]
fn welch_cpsd(
    data: &Array2<f64>,
    n_seg: usize,
    overlap: f64,
    fs: f64,
) -> SignalResult<(Vec<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let (n_samples, n_ch) = (data.nrows(), data.ncols());
    if n_samples == 0 || n_ch == 0 {
        return Err(SignalError::InvalidArgument(
            "data must have at least one sample and one channel".to_string(),
        ));
    }
    if n_seg == 0 {
        return Err(SignalError::InvalidArgument(
            "n_seg must be > 0".to_string(),
        ));
    }
    let n_fft = next_pow2(n_seg);
    let step = ((n_seg as f64) * (1.0 - overlap.clamp(0.0, 0.99))) as usize;
    let step = step.max(1);

    // Hann window
    let hann: Vec<f64> = (0..n_seg)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n_seg - 1) as f64).cos()))
        .collect();
    let win_norm: f64 = hann.iter().map(|v| v * v).sum::<f64>().sqrt();

    let n_bins = n_fft / 2 + 1; // one-sided
    let mut cpsd_re = vec![Array2::<f64>::zeros((n_ch, n_ch)); n_bins];
    let mut cpsd_im = vec![Array2::<f64>::zeros((n_ch, n_ch)); n_bins];
    let mut n_avg: usize = 0;

    let mut start = 0_usize;
    loop {
        let end = start + n_seg;
        if end > n_samples {
            break;
        }
        // Extract and window each channel segment
        let mut spectra: Vec<Vec<Cplx>> = Vec::with_capacity(n_ch);
        for ch in 0..n_ch {
            let seg: Vec<f64> = (start..end)
                .map(|i| data[[i, ch]] * hann[i - start])
                .collect();
            let sp = fft_radix2(&seg);
            spectra.push(sp);
        }
        // Accumulate cross-products for one-sided bins
        for k in 0..n_bins {
            for i in 0..n_ch {
                for j in 0..n_ch {
                    let xi = spectra[i][k];
                    let xj_conj = cconj(spectra[j][k]);
                    let prod = cmul(xi, xj_conj);
                    cpsd_re[k][[i, j]] += prod.0;
                    cpsd_im[k][[i, j]] += prod.1;
                }
            }
        }
        n_avg += 1;
        start += step;
    }

    if n_avg == 0 {
        return Err(SignalError::InvalidArgument(
            "data too short for the requested segment length".to_string(),
        ));
    }

    // Normalise: average and apply window/frequency scaling
    let scale = 1.0 / (n_avg as f64 * win_norm * win_norm);
    for k in 0..n_bins {
        cpsd_re[k].mapv_inplace(|v| v * scale);
        cpsd_im[k].mapv_inplace(|v| v * scale);
    }

    // Frequency axis
    let freq_res = fs / n_fft as f64;
    let freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * freq_res).collect();

    Ok((freqs, cpsd_re, cpsd_im))
}

// ---------------------------------------------------------------------------
// 2 × 2 SVD helper (Golub-Kahan for real symmetric positive semi-definite matrix)
// Generalised SVD for n_ch × n_ch: power-iteration based
// ---------------------------------------------------------------------------

/// Compute the first singular value and left singular vector of a real symmetric
/// PSD matrix `re` via a few steps of power iteration.
///
/// `re` is the real part of the CPSD at one frequency bin (the imaginary part
/// is skew-symmetric and its contribution to the first singular value is often
/// small; we use only the real part for robustness).
fn first_sv_power_iter(re: &Array2<f64>, n_iter: usize) -> (f64, Array1<f64>) {
    let n = re.nrows();
    if n == 0 {
        return (0.0, Array1::zeros(0));
    }
    // Start with uniform vector
    let mut v: Array1<f64> = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
    for _ in 0..n_iter {
        // w = re * v
        let mut w = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                w[i] += re[[i, j]] * v[j];
            }
        }
        // Normalise
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            return (0.0, v);
        }
        v = w.mapv(|x| x / norm);
    }
    // Compute Rayleigh quotient for the eigenvalue
    let mut rv: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n {
            rv[i] += re[[i, j]] * v[j];
        }
    }
    let sigma: f64 = v.iter().zip(rv.iter()).map(|(a, b)| a * b).sum::<f64>();
    (sigma.abs().sqrt(), v)
}

// ---------------------------------------------------------------------------
// Peak detection
// ---------------------------------------------------------------------------

/// Find local maxima in `signal`.  Returns indices of all peaks.
fn find_peaks(signal: &[f64]) -> Vec<usize> {
    let n = signal.len();
    let mut peaks = Vec::new();
    for i in 1..n.saturating_sub(1) {
        if signal[i] > signal[i - 1] && signal[i] >= signal[i + 1] {
            peaks.push(i);
        }
    }
    peaks
}

/// Keep only the `k` strongest peaks (by amplitude), sorted by frequency index.
fn top_k_peaks(peaks: &[usize], signal: &[f64], k: usize) -> Vec<usize> {
    let mut pairs: Vec<(usize, f64)> = peaks.iter().map(|&i| (i, signal[i])).collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    pairs.sort_by_key(|(i, _)| *i);
    pairs.into_iter().map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Identify structural modes via Frequency Domain Decomposition.
///
/// # Arguments
/// * `data` — measured vibration data, shape `[n_samples, n_channels]`
/// * `config` — OMA configuration (sampling frequency, number of modes, etc.)
///
/// # Returns
/// [`OmaResult`] containing the identified [`ModalMode`]s.
///
/// # Algorithm
/// 1. Segment `data` into overlapping blocks of length `config.n_lags` (treated
///    as the Welch segment size).
/// 2. Estimate the CPSD matrix at each one-sided frequency bin.
/// 3. Compute the first singular value (and left singular vector) at each bin
///    via power iteration on the real part of the CPSD.
/// 4. Find the `config.n_modes` strongest local peaks in the first-SV curve
///    within `[freq_min, freq_max]`.
/// 5. Return each peak's frequency and corresponding mode shape.
///
/// # Errors
/// Returns [`crate::error::SignalError`] if data dimensions are invalid or
/// the segment length is too short.
pub fn fdd_identify(data: &Array2<f64>, config: &OmaConfig) -> SignalResult<OmaResult> {
    let (n_samples, n_ch) = (data.nrows(), data.ncols());
    if n_samples < 2 {
        return Err(SignalError::InvalidArgument(
            "data must have at least 2 samples".to_string(),
        ));
    }
    if n_ch == 0 {
        return Err(SignalError::InvalidArgument(
            "data must have at least one channel".to_string(),
        ));
    }
    let n_seg = config.n_lags.max(4);
    let fs = config.fs;
    let freq_max = config.freq_max.unwrap_or(fs / 2.0);

    // Step 1: Welch cross-PSD
    let (freqs, cpsd_re, _cpsd_im) = welch_cpsd(data, n_seg, config.overlap, fs)?;
    let n_bins = freqs.len();
    if n_bins < 2 {
        return Err(SignalError::ComputationError(
            "too few frequency bins; increase n_lags or data length".to_string(),
        ));
    }

    // Step 2-3: Power-iteration SVD at each bin
    let n_power_iter = 20;
    let mut sv1: Vec<f64> = Vec::with_capacity(n_bins);
    let mut mode_shapes: Vec<Array1<f64>> = Vec::with_capacity(n_bins);
    for k in 0..n_bins {
        let (sigma, u1) = first_sv_power_iter(&cpsd_re[k], n_power_iter);
        sv1.push(sigma);
        mode_shapes.push(u1);
    }

    // Step 4: Restrict to frequency range and find peaks
    let i_min = freqs
        .iter()
        .position(|&f| f >= config.freq_min)
        .unwrap_or(0);
    let i_max = freqs
        .iter()
        .rposition(|&f| f <= freq_max)
        .unwrap_or(n_bins - 1)
        + 1;
    let i_max = i_max.min(n_bins);

    let sv1_slice = &sv1[i_min..i_max];
    let peaks_local = find_peaks(sv1_slice);
    let peaks_global: Vec<usize> = peaks_local.iter().map(|&p| p + i_min).collect();
    let top_peaks = top_k_peaks(&peaks_global, &sv1, config.n_modes);

    // Step 5: Build result
    let n_found = top_peaks.len();
    let mut modes = Vec::with_capacity(n_found);
    for &peak_idx in &top_peaks {
        let freq = freqs[peak_idx];
        // Damping: crude half-power bandwidth estimate
        let peak_sv = sv1[peak_idx];
        let half_power = peak_sv / 2.0_f64.sqrt();
        let i_left = (0..peak_idx)
            .rev()
            .find(|&i| sv1[i] <= half_power)
            .unwrap_or(0);
        let i_right = (peak_idx..n_bins)
            .find(|&i| sv1[i] <= half_power)
            .unwrap_or(peak_idx);
        let bw = (freqs[i_right] - freqs[i_left]).abs();
        let damping = if freq > 0.0 { bw / (2.0 * freq) } else { 0.0 };

        let mut mode_shape = mode_shapes[peak_idx].clone();
        // Normalise to unit norm
        let norm: f64 = mode_shape.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > f64::EPSILON {
            mode_shape.mapv_inplace(|v| v / norm);
        }

        modes.push(ModalMode::new(freq, damping.clamp(0.0, 1.0), mode_shape));
    }

    let freq_resolution = if n_bins > 1 { freqs[1] - freqs[0] } else { 0.0 };
    Ok(OmaResult::new(modes, OmaMethod::Fdd, freq_resolution))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use std::f64::consts::PI;

    fn sine_data(freqs_hz: &[f64], fs: f64, n: usize, n_ch: usize) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, n_ch));
        for (ch, &f) in freqs_hz.iter().take(n_ch).enumerate() {
            for i in 0..n {
                let t = i as f64 / fs;
                data[[i, ch]] = (2.0 * PI * f * t).sin();
            }
        }
        data
    }

    #[test]
    fn test_fdd_basic_two_modes() {
        // Two-channel signal with modes at 5 Hz and 15 Hz
        let fs = 200.0_f64;
        let n = 2048_usize;
        let freqs = [5.0, 15.0];
        let mut data = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 / fs;
            data[[i, 0]] = (2.0 * PI * freqs[0] * t).sin() + (2.0 * PI * freqs[1] * t).sin();
            data[[i, 1]] =
                (2.0 * PI * freqs[0] * t).sin() * 0.8 + (2.0 * PI * freqs[1] * t).sin() * 1.2;
        }
        let config = OmaConfig {
            n_modes: 2,
            fs,
            n_lags: 256,
            freq_min: 1.0,
            freq_max: Some(50.0),
            overlap: 0.5,
            ..Default::default()
        };
        let result = fdd_identify(&data, &config).expect("fdd_identify should succeed");
        assert_eq!(result.method, OmaMethod::Fdd);
        assert!(result.n_modes() <= 2);
    }

    #[test]
    fn test_fdd_single_channel() {
        let fs = 100.0_f64;
        let n = 1024_usize;
        let mut data = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let t = i as f64 / fs;
            data[[i, 0]] = (2.0 * PI * 10.0 * t).sin();
        }
        let config = OmaConfig {
            n_modes: 1,
            fs,
            n_lags: 128,
            freq_min: 1.0,
            freq_max: Some(40.0),
            overlap: 0.5,
            ..Default::default()
        };
        let result = fdd_identify(&data, &config).expect("single-channel fdd should succeed");
        // n_modes() returns usize, always non-negative; just verify the call succeeds
        let _n = result.n_modes();
    }

    #[test]
    fn test_fdd_error_empty() {
        let data = Array2::<f64>::zeros((0, 2));
        let config = OmaConfig::default();
        assert!(fdd_identify(&data, &config).is_err());
    }

    #[test]
    fn test_fdd_mode_shapes_unit_norm() {
        let fs = 200.0_f64;
        let n = 2048_usize;
        let freqs_hz = [5.0, 20.0];
        let data = sine_data(&freqs_hz, fs, n, 2);
        let config = OmaConfig {
            n_modes: 2,
            fs,
            n_lags: 256,
            freq_min: 1.0,
            freq_max: Some(80.0),
            overlap: 0.5,
            ..Default::default()
        };
        let result = fdd_identify(&data, &config).expect("fdd_identify should succeed");
        for mode in &result.modes {
            let norm: f64 = mode.mode_shape.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "mode shape norm={norm}");
        }
    }

    #[test]
    fn test_fdd_freq_resolution() {
        let fs = 512.0_f64;
        let n = 4096_usize;
        let data = sine_data(&[50.0], fs, n, 1);
        let config = OmaConfig {
            n_modes: 1,
            fs,
            n_lags: 256,
            freq_min: 10.0,
            freq_max: Some(200.0),
            overlap: 0.5,
            ..Default::default()
        };
        let result = fdd_identify(&data, &config).expect("fdd should succeed");
        assert!(result.freq_resolution > 0.0);
    }

    #[test]
    fn test_next_pow2() {
        assert_eq!(next_pow2(1), 1);
        assert_eq!(next_pow2(2), 2);
        assert_eq!(next_pow2(3), 4);
        assert_eq!(next_pow2(255), 256);
        assert_eq!(next_pow2(256), 256);
    }

    #[test]
    fn test_find_peaks_simple() {
        let sig = vec![0.0, 1.0, 0.5, 2.0, 0.5, 0.0];
        let peaks = find_peaks(&sig);
        assert!(peaks.contains(&1) || peaks.contains(&3));
    }

    #[test]
    fn test_fdd_many_channels() {
        let fs = 200.0_f64;
        let n = 2048;
        let n_ch = 4;
        let freqs_hz = [5.0, 12.0, 25.0, 40.0];
        let data = sine_data(&freqs_hz, fs, n, n_ch);
        let config = OmaConfig {
            n_modes: 4,
            fs,
            n_lags: 256,
            freq_min: 1.0,
            freq_max: Some(90.0),
            overlap: 0.5,
            ..Default::default()
        };
        let result = fdd_identify(&data, &config).expect("fdd_identify should succeed");
        assert!(result.n_modes() <= 4);
        assert!(!result.modes.is_empty());
    }
}
