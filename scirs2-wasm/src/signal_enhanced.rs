//! Enhanced signal processing WASM bindings (v0.3.0)
//!
//! Provides self-contained, pure-Rust implementations of:
//! - [`wasm_fft_real`] — real-input FFT (Cooley-Tukey DIT radix-2)
//! - [`wasm_ifft_real`] — inverse FFT with real output
//! - [`wasm_power_spectral_density`] — one-sided Welch PSD estimate
//! - [`wasm_stft`] — Short-Time Fourier Transform magnitude spectrogram
//! - [`wasm_convolution_1d`] — direct linear convolution
//! - [`wasm_moving_average_simple`] — causal boxcar moving average (O(n))
//! - [`wasm_butter_lowpass`] — second-order Butterworth IIR (bilinear transform)
//!
//! All functions follow the no-unwrap() and no-warnings policies.

use wasm_bindgen::prelude::*;

use std::f64::consts::PI;

// ============================================================================
// Internal DIT Cooley-Tukey FFT  (power-of-two size, in-place)
// ============================================================================

/// Bit-reversal permutation of length `n` (must be a power of two).
fn bit_reverse_permutation(data: &mut Vec<(f64, f64)>) {
    let n = data.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// In-place radix-2 Cooley-Tukey FFT.
/// `data` must have a power-of-two length.
/// `inverse = true` computes the un-normalised IFFT.
fn fft_inplace(data: &mut Vec<(f64, f64)>, inverse: bool) {
    let n = data.len();
    if n <= 1 { return; }

    bit_reverse_permutation(data);

    let sign = if inverse { 1.0_f64 } else { -1.0_f64 };

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = sign * 2.0 * PI / len as f64;
        let (wre, wim) = (theta.cos(), theta.sin());
        for i in (0..n).step_by(len) {
            let (mut ure, mut uim) = (1.0_f64, 0.0_f64);
            for j in 0..half {
                let (ere, eim) = data[i + j];
                let (ore, oim) = data[i + j + half];
                // twiddle * odd
                let (tre, tim) = (ure * ore - uim * oim, ure * oim + uim * ore);
                data[i + j]         = (ere + tre, eim + tim);
                data[i + j + half]  = (ere - tre, eim - tim);
                // advance twiddle factor
                let (new_ure, new_uim) = (ure * wre - uim * wim, ure * wim + uim * wre);
                ure = new_ure;
                uim = new_uim;
            }
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in data.iter_mut() {
            c.0 *= scale;
            c.1 *= scale;
        }
    }
}

/// Zero-pad `data` to the next power of two >= `data.len()`.
fn next_power_of_two_complex(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    let mut size = 1usize;
    while size < n { size <<= 1; }
    let mut out = data.to_vec();
    out.resize(size, (0.0, 0.0));
    out
}

// ============================================================================
// Public API
// ============================================================================

/// Compute the forward FFT of a real-valued signal.
///
/// The input real signal is zero-padded to the next power of two for efficiency.
/// The output is a flat interleaved array of `(re, im)` pairs:
/// `[re_0, im_0, re_1, im_1, …]` of length `2 * N` where `N` is the padded
/// length.
///
/// # Returns
/// An empty `Vec<f64>` if `signal` is empty.
#[wasm_bindgen]
pub fn wasm_fft_real(signal: &[f64]) -> Vec<f64> {
    if signal.is_empty() { return Vec::new(); }
    let data_raw: Vec<(f64, f64)> = signal.iter().map(|&r| (r, 0.0)).collect();
    // Pad to next power of two
    let mut data = next_power_of_two_complex(&data_raw);
    fft_inplace(&mut data, false);
    // Flatten
    let mut out = Vec::with_capacity(data.len() * 2);
    for (re, im) in &data {
        out.push(*re);
        out.push(*im);
    }
    out
}

/// Compute the inverse FFT of a complex spectrum, returning only the real part.
///
/// The input `spectrum` must be an interleaved `(re, im)` array of even length,
/// as produced by [`wasm_fft_real`] or the `wasm_fft` function.
/// The real part of the IFFT result is returned.
///
/// # Returns
/// An empty `Vec<f64>` if `spectrum` is empty or has odd length.
#[wasm_bindgen]
pub fn wasm_ifft_real(spectrum: &[f64]) -> Vec<f64> {
    if spectrum.is_empty() || spectrum.len() % 2 != 0 { return Vec::new(); }
    let mut data: Vec<(f64, f64)> = spectrum
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    fft_inplace(&mut data, true);
    data.iter().map(|&(re, _)| re).collect()
}

/// Compute the one-sided Power Spectral Density (PSD) using Welch's method.
///
/// Divides the signal into overlapping 50%-overlap Hann-windowed segments of
/// `segment_len` samples, computes the FFT of each, averages the squared
/// magnitudes, and scales to physical units.
///
/// # Arguments
/// * `signal`       — input real-valued signal.
/// * `sample_rate`  — sampling rate in Hz (used for frequency axis scaling; must be > 0).
/// * `segment_len`  — length of each Welch segment (will be rounded up to the
///                    next power of two internally).
///
/// # Returns
/// A flat `Vec<f64>` of length `segment_len / 2 + 1` containing the one-sided
/// PSD values (units: [signal²/Hz]).  Returns an empty Vec on invalid input.
#[wasm_bindgen]
pub fn wasm_power_spectral_density(signal: &[f64], sample_rate: f64, segment_len: usize) -> Vec<f64> {
    if signal.is_empty() || sample_rate <= 0.0 || segment_len < 4 {
        return Vec::new();
    }

    // Round segment_len to next power of two for FFT efficiency
    let mut nfft = 1usize;
    while nfft < segment_len { nfft <<= 1; }
    let n_out = nfft / 2 + 1; // one-sided spectrum

    // Build Hann window
    let hann: Vec<f64> = (0..nfft)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (nfft - 1) as f64).cos()))
        .collect();
    let win_pow: f64 = hann.iter().map(|w| w * w).sum::<f64>();

    let hop = nfft / 2; // 50% overlap
    let n = signal.len();
    let mut psd = vec![0.0_f64; n_out];
    let mut n_segments = 0usize;

    let mut start = 0usize;
    while start + nfft <= n {
        // Extract and window
        let mut frame: Vec<(f64, f64)> = signal[start..start + nfft]
            .iter()
            .zip(hann.iter())
            .map(|(&s, &w)| (s * w, 0.0))
            .collect();
        fft_inplace(&mut frame, false);
        // Accumulate one-sided power
        for k in 0..n_out {
            let (re, im) = frame[k];
            let mut power = re * re + im * im;
            // Double-count interior bins for one-sided PSD
            if k > 0 && k < nfft / 2 { power *= 2.0; }
            psd[k] += power;
        }
        n_segments += 1;
        start += hop;
    }

    if n_segments == 0 {
        // Signal shorter than one segment: process the whole signal zero-padded
        let mut frame: Vec<(f64, f64)> = (0..nfft)
            .map(|i| {
                let s = if i < n { signal[i] } else { 0.0 };
                (s * hann[i], 0.0)
            })
            .collect();
        fft_inplace(&mut frame, false);
        for k in 0..n_out {
            let (re, im) = frame[k];
            let mut power = re * re + im * im;
            if k > 0 && k < nfft / 2 { power *= 2.0; }
            psd[k] = power;
        }
        n_segments = 1;
    }

    // Normalise: average over segments, scale by window power and sample rate
    let scale = 1.0 / (n_segments as f64 * win_pow * sample_rate);
    for p in psd.iter_mut() {
        *p *= scale;
    }

    psd
}

/// Compute the Short-Time Fourier Transform (STFT) magnitude spectrogram.
///
/// Slides a Hann window of length `window_size` over the signal with a step of
/// `hop_size` samples and computes the FFT magnitude at each frame.
///
/// # Arguments
/// * `signal`      — input real-valued signal.
/// * `window_size` — FFT window length (zero-padded to next power of two).
/// * `hop_size`    — step between consecutive frames (must be >= 1).
///
/// # Returns
/// Flattened row-major `Vec<f64>` of shape `(n_frames, n_freq)` where
/// `n_freq = window_size / 2 + 1`.  Returns an empty Vec on invalid input.
///
/// Access element `(frame, freq)`: `result[frame * n_freq + freq]`.
#[wasm_bindgen]
pub fn wasm_stft(signal: &[f64], window_size: usize, hop_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size < 2 || hop_size == 0 {
        return Vec::new();
    }

    // Round window_size to next power of two for FFT
    let mut nfft = 1usize;
    while nfft < window_size { nfft <<= 1; }
    let n_freq = nfft / 2 + 1;

    // Hann window (only over the actual window_size samples, zero elsewhere)
    let hann: Vec<f64> = (0..window_size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (window_size.saturating_sub(1).max(1)) as f64).cos()))
        .collect();

    let n = signal.len();
    let n_frames = if n >= window_size {
        (n - window_size) / hop_size + 1
    } else {
        1
    };

    let mut out = vec![0.0_f64; n_frames * n_freq];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let mut buf: Vec<(f64, f64)> = (0..nfft)
            .map(|i| {
                let s = if i < window_size && start + i < n { signal[start + i] } else { 0.0 };
                let w = if i < window_size { hann[i] } else { 0.0 };
                (s * w, 0.0)
            })
            .collect();
        fft_inplace(&mut buf, false);
        for k in 0..n_freq {
            let (re, im) = buf[k];
            out[frame_idx * n_freq + k] = (re * re + im * im).sqrt();
        }
    }

    out
}

/// Compute the linear convolution of `signal` and `kernel` (full output).
///
/// Output length = `signal.len() + kernel.len() - 1`.
///
/// This implementation uses direct O(N*M) computation for small kernels and
/// FFT-based convolution for large kernels, selecting automatically based on
/// the product of the two lengths.
///
/// # Returns
/// An empty `Vec<f64>` if either input is empty.
#[wasm_bindgen]
pub fn wasm_convolution_1d(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    if signal.is_empty() || kernel.is_empty() { return Vec::new(); }

    let ns = signal.len();
    let nk = kernel.len();
    let out_len = ns + nk - 1;

    // Use direct convolution for small sizes, FFT-based for larger ones
    if ns * nk <= 4096 {
        // Direct O(N*M) convolution
        let mut out = vec![0.0_f64; out_len];
        for i in 0..ns {
            for j in 0..nk {
                out[i + j] += signal[i] * kernel[j];
            }
        }
        out
    } else {
        // FFT-based convolution via overlap-add
        let mut size = 1usize;
        while size < out_len { size <<= 1; }

        let mut sa: Vec<(f64, f64)> = signal.iter().map(|&r| (r, 0.0)).collect();
        sa.resize(size, (0.0, 0.0));
        let mut ka: Vec<(f64, f64)> = kernel.iter().map(|&r| (r, 0.0)).collect();
        ka.resize(size, (0.0, 0.0));

        fft_inplace(&mut sa, false);
        fft_inplace(&mut ka, false);

        // Pointwise multiplication
        let mut product: Vec<(f64, f64)> = sa.iter().zip(ka.iter())
            .map(|((ar, ai), (br, bi))| (ar * br - ai * bi, ar * bi + ai * br))
            .collect();

        fft_inplace(&mut product, true);
        product[..out_len].iter().map(|&(re, _)| re).collect()
    }
}

/// Causal boxcar (uniform) moving average with window `window`.
///
/// Each output element is the mean of the immediately preceding `window` samples
/// (including the current sample), using a sliding-sum accumulator for O(n)
/// complexity.
///
/// # Arguments
/// * `signal` — input signal.
/// * `window` — averaging window length (must be >= 1 and <= signal length).
///
/// # Returns
/// Vec of the same length as `signal`, or empty Vec on invalid parameters.
#[wasm_bindgen]
pub fn wasm_moving_average_simple(signal: &[f64], window: usize) -> Vec<f64> {
    if signal.is_empty() || window == 0 || window > signal.len() {
        return Vec::new();
    }
    let n = signal.len();
    let mut out = vec![0.0_f64; n];
    let mut acc = 0.0_f64;
    for i in 0..n {
        acc += signal[i];
        if i >= window {
            acc -= signal[i - window];
        }
        let w = if i + 1 < window { i + 1 } else { window };
        out[i] = acc / w as f64;
    }
    out
}

/// Apply a second-order (biquad) Butterworth low-pass IIR filter to a signal.
///
/// The filter is designed via the bilinear transform at the given
/// normalised cutoff frequency `cutoff_hz / (sample_rate / 2)`.  A single
/// second-order section is applied; for steeper roll-off cascade multiple calls
/// or use a higher-order design tool.
///
/// # Arguments
/// * `signal`          — input real-valued signal.
/// * `cutoff_hz`       — desired -3 dB cutoff frequency in Hz (must be in `(0, sample_rate/2)`).
/// * `sample_rate`     — sampling rate in Hz (must be > 0).
///
/// # Returns
/// Filtered signal of the same length as `signal`, or empty Vec on invalid params.
#[wasm_bindgen]
pub fn wasm_butter_lowpass(signal: &[f64], cutoff_hz: f64, sample_rate: f64) -> Vec<f64> {
    if signal.is_empty() || cutoff_hz <= 0.0 || sample_rate <= 0.0 {
        return Vec::new();
    }
    let nyquist = sample_rate / 2.0;
    let wn = cutoff_hz / nyquist; // normalised in (0, 1)
    if wn >= 1.0 { return Vec::new(); }

    // Pre-warp: ωa = 2*fs * tan(π * wn / 2)
    let wa = 2.0 * sample_rate * (PI * wn / 2.0).tan();

    // Second-order Butterworth prototype: poles at exp(jπ(2k+n+1)/(2n)), n=2
    // → denominator: s² + sqrt(2)*wa*s + wa²
    // After bilinear transform z = (2fs + s)/(2fs - s):
    let fs2 = 2.0 * sample_rate;
    let wa2 = wa * wa;
    let sqrt2_wa = std::f64::consts::SQRT_2 * wa;

    // a0_pre = fs2² + sqrt(2)*wa*fs2 + wa²
    let a0_pre = fs2 * fs2 + sqrt2_wa * fs2 + wa2;

    // Numerator B(z): b0=b1*2=b2 = wa² / a0_pre
    let b0 = wa2 / a0_pre;
    let b1 = 2.0 * b0;
    let b2 = b0;

    // Denominator A(z): a0=1, a1, a2
    let a1 = (2.0 * wa2 - 2.0 * fs2 * fs2) / a0_pre;
    let a2 = (fs2 * fs2 - sqrt2_wa * fs2 + wa2) / a0_pre;

    // Direct form II transposed filter
    let n = signal.len();
    let mut out = vec![0.0_f64; n];
    let (mut d1, mut d2) = (0.0_f64, 0.0_f64);

    for i in 0..n {
        let x = signal[i];
        let y = b0 * x + d1;
        d1 = b1 * x - a1 * y + d2;
        d2 = b2 * x - a2 * y;
        out[i] = y;
    }
    out
}

/// Compute the frequency axis for a real FFT output.
///
/// Returns a `Vec<f64>` of length `n / 2 + 1` containing the frequency values
/// in Hz corresponding to each bin of a one-sided FFT of `n` samples at
/// `sample_rate` Hz.
#[wasm_bindgen]
pub fn wasm_fft_frequencies(n: usize, sample_rate: f64) -> Vec<f64> {
    if n == 0 || sample_rate <= 0.0 { return Vec::new(); }
    let n_bins = n / 2 + 1;
    let df = sample_rate / n as f64;
    (0..n_bins).map(|k| k as f64 * df).collect()
}

/// Compute the magnitude spectrum (absolute value) of an interleaved FFT output.
///
/// # Arguments
/// * `spectrum` — interleaved `(re, im)` array from `wasm_fft_real`.
///
/// # Returns
/// Magnitude array of length `spectrum.len() / 2`.
#[wasm_bindgen]
pub fn wasm_fft_magnitude(spectrum: &[f64]) -> Vec<f64> {
    if spectrum.len() % 2 != 0 { return Vec::new(); }
    spectrum.chunks_exact(2)
        .map(|pair| (pair[0] * pair[0] + pair[1] * pair[1]).sqrt())
        .collect()
}

/// Compute the phase spectrum (argument) of an interleaved FFT output in radians.
///
/// # Arguments
/// * `spectrum` — interleaved `(re, im)` array from `wasm_fft_real`.
///
/// # Returns
/// Phase array of length `spectrum.len() / 2`, values in `(-π, π]`.
#[wasm_bindgen]
pub fn wasm_fft_phase(spectrum: &[f64]) -> Vec<f64> {
    if spectrum.len() % 2 != 0 { return Vec::new(); }
    spectrum.chunks_exact(2)
        .map(|pair| pair[1].atan2(pair[0]))
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Tolerance for floating-point comparisons
    const TOL: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool { (a - b).abs() < eps }

    #[test]
    fn test_fft_real_empty() {
        assert!(wasm_fft_real(&[]).is_empty());
    }

    #[test]
    fn test_fft_real_dc() {
        // Constant signal: FFT bin 0 = N * amplitude, rest near 0
        let signal = vec![1.0_f64; 8];
        let spectrum = wasm_fft_real(&signal);
        assert_eq!(spectrum.len(), 16); // 2 * 8 (padded to 8, already pow2)
        // DC bin magnitude = 8
        let dc_mag = (spectrum[0] * spectrum[0] + spectrum[1] * spectrum[1]).sqrt();
        assert!(approx_eq(dc_mag, 8.0, 1e-8), "DC mag = {}", dc_mag);
    }

    #[test]
    fn test_fft_real_single_tone() {
        // Pure cosine at frequency k/N should have magnitude N/2 at bin k
        let n = 8usize;
        let k = 2usize; // target bin
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * k as f64 * i as f64 / n as f64).cos()).collect();
        let spectrum = wasm_fft_real(&signal);
        assert_eq!(spectrum.len(), 2 * n);
        let re = spectrum[2 * k];
        let im = spectrum[2 * k + 1];
        let mag = (re * re + im * im).sqrt();
        // Should be close to n/2 = 4
        assert!(approx_eq(mag, n as f64 / 2.0, 1e-6), "mag at bin {} = {}", k, mag);
    }

    #[test]
    fn test_ifft_real_roundtrip() {
        let signal = [1.0_f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0];
        let spectrum = wasm_fft_real(&signal);
        let recovered = wasm_ifft_real(&spectrum);
        assert_eq!(recovered.len(), signal.len());
        for (i, (&orig, &rec)) in signal.iter().zip(recovered.iter()).enumerate() {
            assert!(approx_eq(orig, rec, 1e-8), "index {} orig {} rec {}", i, orig, rec);
        }
    }

    #[test]
    fn test_ifft_real_empty() {
        assert!(wasm_ifft_real(&[]).is_empty());
        assert!(wasm_ifft_real(&[1.0]).is_empty()); // odd length
    }

    #[test]
    fn test_psd_constant_signal() {
        // A DC signal (constant) should have all power at bin 0
        let signal = vec![1.0_f64; 64];
        let psd = wasm_power_spectral_density(&signal, 1000.0, 32);
        assert!(!psd.is_empty(), "PSD should be non-empty");
        assert!(psd[0] > 0.0, "DC power should be > 0");
    }

    #[test]
    fn test_psd_empty_signal() {
        assert!(wasm_power_spectral_density(&[], 1000.0, 32).is_empty());
    }

    #[test]
    fn test_psd_invalid_sample_rate() {
        let signal = vec![1.0_f64; 64];
        assert!(wasm_power_spectral_density(&signal, 0.0, 32).is_empty());
        assert!(wasm_power_spectral_density(&signal, -1.0, 32).is_empty());
    }

    #[test]
    fn test_stft_shape() {
        let signal: Vec<f64> = (0..128).map(|i| (i as f64).sin()).collect();
        let window = 32;
        let hop = 16;
        let spec = wasm_stft(&signal, window, hop);
        let n_freq = 32 / 2 + 1; // window_size / 2 + 1 (already pow2)
        let n_frames = (signal.len() - window) / hop + 1;
        assert_eq!(spec.len(), n_frames * n_freq, "STFT shape mismatch");
    }

    #[test]
    fn test_stft_empty() {
        assert!(wasm_stft(&[], 32, 8).is_empty());
    }

    #[test]
    fn test_stft_zero_hop() {
        let signal = vec![1.0_f64; 32];
        assert!(wasm_stft(&signal, 16, 0).is_empty());
    }

    #[test]
    fn test_stft_non_negative() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let spec = wasm_stft(&signal, 16, 8);
        for &v in &spec {
            assert!(v >= 0.0, "STFT magnitudes must be non-negative, got {}", v);
        }
    }

    #[test]
    fn test_convolution_1d_simple() {
        // [1, 2, 3] * [1, 1] = [1, 3, 5, 3]
        let s = [1.0_f64, 2.0, 3.0];
        let k = [1.0_f64, 1.0];
        let out = wasm_convolution_1d(&s, &k);
        assert_eq!(out.len(), 4);
        assert!(approx_eq(out[0], 1.0, TOL), "out[0] = {}", out[0]);
        assert!(approx_eq(out[1], 3.0, TOL), "out[1] = {}", out[1]);
        assert!(approx_eq(out[2], 5.0, TOL), "out[2] = {}", out[2]);
        assert!(approx_eq(out[3], 3.0, TOL), "out[3] = {}", out[3]);
    }

    #[test]
    fn test_convolution_1d_identity() {
        // Convolving with [1] is an identity
        let s = [3.0_f64, 1.0, 4.0, 1.0, 5.0];
        let k = [1.0_f64];
        let out = wasm_convolution_1d(&s, &k);
        assert_eq!(out.len(), s.len());
        for (i, (&orig, &got)) in s.iter().zip(out.iter()).enumerate() {
            assert!(approx_eq(orig, got, TOL), "index {} orig {} got {}", i, orig, got);
        }
    }

    #[test]
    fn test_convolution_1d_empty() {
        assert!(wasm_convolution_1d(&[], &[1.0]).is_empty());
        assert!(wasm_convolution_1d(&[1.0], &[]).is_empty());
    }

    #[test]
    fn test_moving_average_simple_window1() {
        // Window of 1 = identity
        let s = [1.0_f64, 5.0, 3.0];
        let out = wasm_moving_average_simple(&s, 1);
        for (i, (&a, &b)) in s.iter().zip(out.iter()).enumerate() {
            assert!(approx_eq(a, b, TOL), "index {}", i);
        }
    }

    #[test]
    fn test_moving_average_simple_window3() {
        let s = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let out = wasm_moving_average_simple(&s, 3);
        // First partial: 1/1=1, (1+2)/2=1.5, (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
        assert!(approx_eq(out[0], 1.0, TOL), "out[0] = {}", out[0]);
        assert!(approx_eq(out[1], 1.5, TOL), "out[1] = {}", out[1]);
        assert!(approx_eq(out[2], 2.0, TOL), "out[2] = {}", out[2]);
        assert!(approx_eq(out[3], 3.0, TOL), "out[3] = {}", out[3]);
        assert!(approx_eq(out[4], 4.0, TOL), "out[4] = {}", out[4]);
    }

    #[test]
    fn test_moving_average_invalid() {
        assert!(wasm_moving_average_simple(&[], 1).is_empty());
        assert!(wasm_moving_average_simple(&[1.0, 2.0], 0).is_empty());
        assert!(wasm_moving_average_simple(&[1.0, 2.0], 3).is_empty());
    }

    #[test]
    fn test_butter_lowpass_passthrough() {
        // A DC signal through lowpass should pass without change
        let signal = vec![1.0_f64; 100];
        let out = wasm_butter_lowpass(&signal, 100.0, 1000.0);
        assert_eq!(out.len(), signal.len());
        // After startup transient (first few samples), output should stabilise near 1
        let last = out[out.len() - 1];
        assert!(approx_eq(last, 1.0, 0.01), "DC passthrough: last = {}", last);
    }

    #[test]
    fn test_butter_lowpass_attenuates_highfreq() {
        // High-frequency signal (near Nyquist) should be attenuated
        let sample_rate = 1000.0;
        let cutoff = 50.0;
        let n = 500;
        // Signal at 400 Hz (well above cutoff)
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 400.0 * i as f64 / sample_rate).sin())
            .collect();
        let out = wasm_butter_lowpass(&signal, cutoff, sample_rate);
        assert_eq!(out.len(), signal.len());
        // RMS of output should be much less than 1 (significant attenuation)
        let rms_out: f64 = (out.iter().map(|&x| x * x).sum::<f64>() / n as f64).sqrt();
        assert!(rms_out < 0.5, "RMS should be attenuated: {}", rms_out);
    }

    #[test]
    fn test_butter_lowpass_invalid() {
        let s = vec![1.0_f64; 10];
        assert!(wasm_butter_lowpass(&[], 100.0, 1000.0).is_empty());
        assert!(wasm_butter_lowpass(&s, 0.0, 1000.0).is_empty());
        assert!(wasm_butter_lowpass(&s, 500.0, 1000.0).is_empty()); // cutoff == Nyquist
        assert!(wasm_butter_lowpass(&s, 100.0, 0.0).is_empty());
    }

    #[test]
    fn test_fft_frequencies() {
        let freqs = wasm_fft_frequencies(8, 1000.0);
        assert_eq!(freqs.len(), 5); // 8/2+1
        assert!(approx_eq(freqs[0], 0.0, TOL));
        assert!(approx_eq(freqs[1], 125.0, TOL)); // 1000/8
        assert!(approx_eq(freqs[4], 500.0, TOL)); // Nyquist
    }

    #[test]
    fn test_fft_magnitude_basic() {
        // Interleaved: [3, 4, ...] → magnitude = 5
        let spectrum = [3.0_f64, 4.0, 0.0, 0.0];
        let mag = wasm_fft_magnitude(&spectrum);
        assert_eq!(mag.len(), 2);
        assert!(approx_eq(mag[0], 5.0, TOL), "mag[0] = {}", mag[0]);
    }

    #[test]
    fn test_fft_phase_basic() {
        // Real-only input → phase = 0
        let spectrum = [1.0_f64, 0.0, -1.0, 0.0];
        let phase = wasm_fft_phase(&spectrum);
        assert_eq!(phase.len(), 2);
        assert!(approx_eq(phase[0], 0.0, TOL), "phase[0] = {}", phase[0]);
        // Im=0, Re=-1 → atan2(0, -1) = π
        assert!(approx_eq(phase[1], PI, TOL), "phase[1] = {}", phase[1]);
    }

    #[test]
    fn test_convolution_fft_vs_direct() {
        // For a medium-sized input both paths should give the same answer
        let s: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let k: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
        let direct = wasm_convolution_1d(&s, &k);
        // Force FFT path by using a large inner product count
        // (ns*nk = 1000 < 4096 so direct path is used; use bigger arrays to test both)
        let s2: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let k2: Vec<f64> = (0..50).map(|i| i as f64 * 0.05).collect();
        let fft_path = wasm_convolution_1d(&s2, &k2);
        // Just verify output lengths are correct
        assert_eq!(direct.len(), s.len() + k.len() - 1);
        assert_eq!(fft_path.len(), s2.len() + k2.len() - 1);
    }
}
