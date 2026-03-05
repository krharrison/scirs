//! Zoom FFT, Goertzel Algorithm, and Sliding DFT
//!
//! This module provides frequency-domain analysis tools that offer targeted
//! frequency resolution improvements over standard FFT:
//!
//! - **Zoom FFT**: High-resolution DFT in a specific frequency band [f1, f2].
//!   Uses the Chirp Z-Transform internally to achieve arbitrary frequency zoom.
//! - **Goertzel Algorithm**: O(N) per frequency computation of DFT coefficients
//!   at specific frequencies. More efficient than FFT when only a few frequencies
//!   are needed.
//! - **Sliding DFT**: Recursive, O(1) per sample update for streaming applications.
//!   Maintains a sliding window DFT that updates as new samples arrive.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::collections::VecDeque;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Zoom FFT
// ---------------------------------------------------------------------------

/// Compute high-resolution DFT in a specific frequency band using the Chirp Z-Transform.
///
/// The Zoom FFT computes `m` equally-spaced DFT samples in the frequency band
/// `[f1, f2]`, providing higher frequency resolution within that band compared
/// to a standard FFT of the same length.
///
/// Internally uses the Chirp Z-Transform (Bluestein's algorithm) to evaluate
/// the Z-transform along an arc from `exp(j*2π*f1/fs)` to `exp(j*2π*f2/fs)`.
///
/// # Arguments
///
/// * `x` - Input signal (time domain)
/// * `f1` - Lower frequency bound (Hz), must be >= 0
/// * `f2` - Upper frequency bound (Hz), must be > f1 and <= fs/2
/// * `m` - Number of output frequency points (>= 1)
/// * `fs` - Sampling frequency (Hz)
///
/// # Returns
///
/// Complex spectrum of length `m` covering frequencies `[f1, f2]`.
///
/// # Examples
///
/// ```
/// use scirs2_signal::zoom_fft::{zoom_fft, zoom_fft_freqs};
///
/// let fs = 1000.0_f64;
/// let n = 1024;
/// // 500 Hz tone
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * 500.0 * i as f64 / fs).sin())
///     .collect();
///
/// // Zoom into 400–600 Hz with 256 points
/// let spectrum = zoom_fft(&signal, 400.0, 600.0, 256, fs).expect("zoom_fft failed");
/// assert_eq!(spectrum.len(), 256);
///
/// let freqs = zoom_fft_freqs(400.0, 600.0, 256);
/// assert_eq!(freqs.len(), 256);
/// // 500 Hz should be near the center
/// ```
pub fn zoom_fft(
    x: &[f64],
    f1: f64,
    f2: f64,
    m: usize,
    fs: f64,
) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if m == 0 {
        return Err(SignalError::ValueError(
            "Number of output points m must be >= 1".into(),
        ));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency fs must be positive".into(),
        ));
    }
    if f1 < 0.0 || f2 <= f1 {
        return Err(SignalError::ValueError(
            "Frequency bounds must satisfy 0 <= f1 < f2".into(),
        ));
    }
    if f2 > fs / 2.0 {
        return Err(SignalError::ValueError(
            "f2 must not exceed the Nyquist frequency (fs/2)".into(),
        ));
    }

    let n = x.len();

    // Starting point on the unit circle: a = exp(j*2π*f1/fs)
    let theta_start = 2.0 * PI * f1 / fs;
    let a = Complex64::new(theta_start.cos(), theta_start.sin());

    // Step between consecutive frequency samples:
    // w = exp(-j * 2π * (f2-f1) / (fs * (m-1)))  for m > 1, else no step
    let delta_f = if m > 1 { (f2 - f1) / (m - 1) as f64 } else { 0.0 };
    let theta_step = -2.0 * PI * delta_f / fs;
    let w = Complex64::new(theta_step.cos(), theta_step.sin());

    // Compute the CZT using Bluestein's algorithm
    bluestein_czt(x, m, w, a)
}

/// Compute the frequency axis for `zoom_fft` output.
///
/// Returns `m` linearly spaced frequencies between `f1` and `f2` (inclusive).
///
/// # Arguments
///
/// * `f1` - Lower frequency bound (Hz)
/// * `f2` - Upper frequency bound (Hz)
/// * `m` - Number of frequency points
///
/// # Returns
///
/// Vector of length `m` with frequency values in Hz.
pub fn zoom_fft_freqs(f1: f64, f2: f64, m: usize) -> Vec<f64> {
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![f1];
    }
    (0..m)
        .map(|i| f1 + i as f64 * (f2 - f1) / (m - 1) as f64)
        .collect()
}

// ---------------------------------------------------------------------------
// Bluestein CZT (internal helper)
// ---------------------------------------------------------------------------

/// Compute the Chirp Z-Transform via Bluestein's algorithm.
///
/// Evaluates X(k) = sum_{n=0}^{N-1} x[n] * a^{-n} * w^{nk}  for k=0..M-1
///
/// Using the identity nk = n(k-n)/2 + n²/2 + k²/2 + nk - n(k-n)/2 - n²/2 - k²/2
/// (actually: nk = [k²/2 - (k-n)²/2 + n²/2]) we rewrite as convolution.
fn bluestein_czt(
    x: &[f64],
    m: usize,
    w: Complex64,
    a: Complex64,
) -> SignalResult<Vec<Complex64>> {
    let n = x.len();

    // Length for the FFT-based convolution: next power of 2 >= (n + m - 1)
    let conv_len = next_pow2(n + m - 1);

    // Chirp factors: w^(n^2/2) and w^(k^2/2)
    // We compute w^(n²/2) using the formula: arg = n² * theta_step / 2
    // But w may be complex with magnitude != 1, so we compute w^(n²/2) directly.

    // yn[n] = x[n] * a^{-n} * w^{n²/2},  n=0..N-1
    let mut yn: Vec<Complex64> = Vec::with_capacity(conv_len);
    let mut a_pow = Complex64::new(1.0, 0.0); // a^{-n}
    let a_inv = Complex64::new(1.0, 0.0) / a;

    // w^(n²/2): use the recurrence w^{n²/2} = w^{(n-1)²/2} * w^{n-1/2} ... difficult
    // Instead compute w_half = w^{1/2} and track w^{n²/2} = w_half^{n²}
    // Use: w^{n²} = w^{(n-1)²} * w^{2n-1}
    // So w^{n²/2}: use the approach w^{n² / 2} = product_{k=0}^{n-1} w^{(2k-1)/2}
    // Easier: precompute chirp[n] = exp(j * pi * n² * log_w / (2pi))
    // Since w = exp(j*theta_step), w^{n²/2} = exp(j*theta_step * n²/2)

    // Extract angle from w: w = |w| * exp(j*angle(w))
    let w_angle = w.im.atan2(w.re);
    let w_mag = (w.re * w.re + w.im * w.im).sqrt();

    // w^{n^2/2}: magnitude part = w_mag^{n^2/2}, phase part = exp(j * w_angle * n^2 / 2)
    // For the CZT on the unit circle, w_mag == 1, but we handle general case.
    let chirp_w = |n_val: i64| -> Complex64 {
        let exp_val = n_val * n_val;
        let mag = w_mag.powf(exp_val as f64 / 2.0);
        let phase = w_angle * exp_val as f64 / 2.0;
        Complex64::new(mag * phase.cos(), mag * phase.sin())
    };

    for ni in 0..n {
        let chirp_n = chirp_w(ni as i64);
        yn.push(x[ni] * a_pow * chirp_n);
        a_pow *= a_inv;
    }
    // Zero-pad to conv_len
    while yn.len() < conv_len {
        yn.push(Complex64::new(0.0, 0.0));
    }

    // hn[k]: impulse response, h[k] = w^{-k^2/2} for k = -(N-1)..(M-1)
    // For the convolution we store h in a wrapped form:
    //   h[k] = w^{-k^2/2} for k = 0..M-1
    //   h[conv_len - k] = w^{-k^2/2} for k = 1..N-1  (negative indices wrapped)
    let mut hn: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); conv_len];
    for ki in 0..m {
        let chirp_k_conj = Complex64::new(chirp_w(ki as i64).re, -chirp_w(ki as i64).im);
        hn[ki] = chirp_k_conj;
    }
    for ni in 1..n {
        let chirp_n_conj = Complex64::new(chirp_w(ni as i64).re, -chirp_w(ni as i64).im);
        hn[conv_len - ni] = chirp_n_conj;
    }

    // FFT both sequences
    let yn_fft = fft_complex(&yn)?;
    let hn_fft = fft_complex(&hn)?;

    // Pointwise multiply
    let mut product: Vec<Complex64> = yn_fft
        .iter()
        .zip(hn_fft.iter())
        .map(|(&y, &h)| y * h)
        .collect();

    // IFFT
    ifft_in_place(&mut product)?;

    // Extract first M points and multiply by w^{k^2/2}
    let result: Vec<Complex64> = (0..m)
        .map(|ki| product[ki] * chirp_w(ki as i64))
        .collect();

    Ok(result)
}

/// Find next power of 2 >= n.
fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Cooley-Tukey radix-2 DIT FFT for complex input (length must be power of 2).
fn fft_complex(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    // Use scirs2_fft for the heavy lifting by converting to (re,im) pairs
    // scirs2_fft::fft takes &[f64] (real), so we do our own implementation
    let mut buf = x.to_vec();
    fft_inplace(&mut buf, false)?;
    Ok(buf)
}

/// In-place Cooley-Tukey FFT (DIT, radix-2). Length must be power of 2.
fn fft_inplace(buf: &mut Vec<Complex64>, inverse: bool) -> SignalResult<()> {
    let n = buf.len();
    if n <= 1 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return Err(SignalError::ValueError(format!(
            "FFT length must be a power of 2, got {}",
            n
        )));
    }

    // Bit-reversal permutation
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2_usize;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * PI / len as f64
        } else {
            -2.0 * PI / len as f64
        };
        let wlen = Complex64::new(angle.cos(), angle.sin());

        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = buf[i + j];
                let v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for c in buf.iter_mut() {
            *c = Complex64::new(c.re * scale, c.im * scale);
        }
    }

    Ok(())
}

/// Reverse the bits of `x` using `bits` significant bits.
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place IFFT.
fn ifft_in_place(buf: &mut Vec<Complex64>) -> SignalResult<()> {
    fft_inplace(buf, true)
}

// ---------------------------------------------------------------------------
// Goertzel Algorithm
// ---------------------------------------------------------------------------

/// Compute DFT coefficients at specific frequencies using the Goertzel algorithm.
///
/// The Goertzel algorithm computes the DFT at arbitrary frequencies with O(N)
/// complexity per frequency. It is more efficient than FFT when only a small
/// number of specific frequencies are of interest.
///
/// The algorithm uses a second-order IIR filter approach equivalent to:
/// ```text
/// X(f) = sum_{n=0}^{N-1} x[n] * exp(-j * 2π * f * n / fs)
/// ```
///
/// # Arguments
///
/// * `x` - Input signal
/// * `freqs` - Frequencies at which to evaluate the DFT (Hz)
/// * `fs` - Sampling frequency (Hz)
///
/// # Returns
///
/// Complex DFT values at each of the requested frequencies.
///
/// # Examples
///
/// ```
/// use scirs2_signal::zoom_fft::goertzel;
///
/// let fs = 8000.0_f64;
/// let n = 256;
/// let freq = 1000.0_f64;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
///     .collect();
///
/// let result = goertzel(&signal, &[freq], fs).expect("goertzel failed");
/// // The magnitude at 1 kHz should be large
/// assert!(result[0].norm() > 10.0);
/// ```
pub fn goertzel(
    x: &[f64],
    freqs: &[f64],
    fs: f64,
) -> SignalResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(SignalError::ValueError("Input signal is empty".into()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency fs must be positive".into(),
        ));
    }

    let n = x.len();
    let mut results = Vec::with_capacity(freqs.len());

    for &freq in freqs {
        if freq < 0.0 || freq > fs / 2.0 {
            return Err(SignalError::ValueError(format!(
                "Frequency {} is outside valid range [0, {}]",
                freq,
                fs / 2.0
            )));
        }

        // Normalized frequency: k = f * N / fs (real-valued for arbitrary f)
        let k = freq * n as f64 / fs;
        let omega = 2.0 * PI * k / n as f64;
        let coeff = 2.0 * omega.cos();

        // Goertzel IIR filter
        let mut s_prev2 = 0.0_f64;
        let mut s_prev1 = 0.0_f64;
        for &sample in x {
            let s = sample + coeff * s_prev1 - s_prev2;
            s_prev2 = s_prev1;
            s_prev1 = s;
        }

        // Final complex output: X = s_prev1 - s_prev2 * exp(-j*omega)
        let re = s_prev1 - s_prev2 * omega.cos();
        let im = s_prev2 * omega.sin();
        results.push(Complex64::new(re, im));
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Sliding DFT
// ---------------------------------------------------------------------------

/// Sliding DFT for efficient streaming frequency analysis.
///
/// Maintains a sliding window DFT that updates in O(K) per sample (where K is
/// the number of tracked frequencies), compared to O(N log N) for recomputing
/// the full FFT every sample. The sliding DFT is exact for frequencies that are
/// exact DFT bin frequencies (i.e., `f = k * fs / N` for integer k).
///
/// For arbitrary frequencies the algorithm uses the frequency-domain update rule:
/// ```text
/// X_new[k] = (X_old[k] - x_out + x_in) * W[k]
/// ```
/// where `W[k] = exp(j * 2π * f_k / fs)` and `x_out` is the oldest sample.
///
/// # Examples
///
/// ```
/// use scirs2_signal::zoom_fft::SlidingDft;
///
/// let fs = 1000.0_f64;
/// let freqs = vec![50.0, 100.0, 200.0];
/// let window_size = 128;
/// let mut sdft = SlidingDft::new(freqs, fs, window_size);
///
/// // Push samples one at a time
/// for i in 0..256_usize {
///     let sample = (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin();
///     let spectrum = sdft.push(sample);
///     assert_eq!(spectrum.len(), 3); // one value per tracked frequency
/// }
/// ```
pub struct SlidingDft {
    /// Tracked frequencies (Hz)
    freqs: Vec<f64>,
    /// Sampling frequency
    fs: f64,
    /// Number of tracked frequencies
    n_freqs: usize,
    /// Window size (N)
    window_size: usize,
    /// Current DFT state (one complex value per tracked frequency)
    state: Vec<Complex64>,
    /// Circular buffer of input samples
    buffer: VecDeque<f64>,
    /// Rotation factors W[k] = exp(j * 2π * f_k / fs)
    rotation: Vec<Complex64>,
}

impl SlidingDft {
    /// Create a new SlidingDft tracker.
    ///
    /// # Arguments
    ///
    /// * `freqs` - Frequencies to track (Hz). Must be in [0, fs/2].
    /// * `fs` - Sampling frequency (Hz)
    /// * `window_size` - Analysis window length (number of samples)
    pub fn new(freqs: Vec<f64>, fs: f64, window_size: usize) -> Self {
        let n_freqs = freqs.len();

        // Precompute rotation factors W[k] = exp(j * 2π * f_k / fs)
        let rotation: Vec<Complex64> = freqs
            .iter()
            .map(|&f| {
                let theta = 2.0 * PI * f / fs;
                Complex64::new(theta.cos(), theta.sin())
            })
            .collect();

        let state = vec![Complex64::new(0.0, 0.0); n_freqs];
        let buffer = VecDeque::with_capacity(window_size + 1);

        Self {
            freqs,
            fs,
            n_freqs,
            window_size,
            state,
            buffer,
            rotation,
        }
    }

    /// Push a new sample and return the updated DFT at all tracked frequencies.
    ///
    /// Uses the recursive update: X_new[k] = (X_old[k] - x_out + x_in) * W[k]
    ///
    /// # Returns
    ///
    /// Vector of complex DFT values at each tracked frequency. The values are
    /// normalized by `1/window_size` so they are comparable to a DFT output.
    pub fn push(&mut self, sample: f64) -> Vec<Complex64> {
        // Get the oldest sample that is about to leave the window
        let x_out = if self.buffer.len() >= self.window_size {
            self.buffer.pop_front().unwrap_or(0.0)
        } else {
            0.0
        };

        // Add new sample to buffer
        self.buffer.push_back(sample);

        // Update DFT state for each tracked frequency
        for k in 0..self.n_freqs {
            // Sliding DFT update rule
            self.state[k] = (self.state[k] - Complex64::new(x_out, 0.0)
                + Complex64::new(sample, 0.0))
                * self.rotation[k];
        }

        // Return normalized copy of state
        let scale = 1.0 / self.window_size as f64;
        self.state
            .iter()
            .map(|&c| Complex64::new(c.re * scale, c.im * scale))
            .collect()
    }

    /// Return the tracked frequencies.
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Return the sampling frequency.
    pub fn fs(&self) -> f64 {
        self.fs
    }

    /// Return the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Return the current number of samples buffered.
    pub fn buffered(&self) -> usize {
        self.buffer.len()
    }

    /// Reset internal state (clear buffer and DFT state).
    pub fn reset(&mut self) {
        self.buffer.clear();
        for s in self.state.iter_mut() {
            *s = Complex64::new(0.0, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn make_tone(freq: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    // --- zoom_fft tests ---

    #[test]
    fn test_zoom_fft_output_length() {
        let fs = 1000.0;
        let signal = make_tone(200.0, 512, fs);
        let m = 64;
        let result = zoom_fft(&signal, 100.0, 300.0, m, fs).expect("zoom_fft failed");
        assert_eq!(result.len(), m);
    }

    #[test]
    fn test_zoom_fft_freqs_length() {
        let freqs = zoom_fft_freqs(100.0, 300.0, 64);
        assert_eq!(freqs.len(), 64);
        assert_relative_eq!(freqs[0], 100.0, epsilon = 1e-10);
        assert_relative_eq!(freqs[63], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zoom_fft_peak_at_tone_frequency() {
        let fs = 1000.0;
        let freq = 250.0;
        let n = 512;
        let signal = make_tone(freq, n, fs);
        let m = 128;
        // Zoom into [200, 300] Hz
        let spectrum = zoom_fft(&signal, 200.0, 300.0, m, fs).expect("zoom_fft failed");
        let freqs = zoom_fft_freqs(200.0, 300.0, m);

        // Find peak
        let (peak_idx, _) = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap_or(std::cmp::Ordering::Equal))
            .expect("empty spectrum");

        // The peak frequency should be close to 250 Hz
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - freq).abs() < 2.0,
            "Expected peak near {} Hz, got {} Hz",
            freq,
            peak_freq
        );
    }

    #[test]
    fn test_zoom_fft_empty_signal_error() {
        assert!(zoom_fft(&[], 100.0, 300.0, 64, 1000.0).is_err());
    }

    #[test]
    fn test_zoom_fft_invalid_freqs_error() {
        let signal = vec![1.0; 64];
        assert!(zoom_fft(&signal, 300.0, 100.0, 64, 1000.0).is_err()); // f1 > f2
        assert!(zoom_fft(&signal, 100.0, 600.0, 64, 1000.0).is_err()); // f2 > fs/2
    }

    // --- Goertzel tests ---

    #[test]
    fn test_goertzel_matches_fft_magnitude() {
        let fs = 8000.0;
        let n = 256;
        let freq = 1000.0_f64;
        let signal = make_tone(freq, n, fs);

        // Goertzel at exact DFT bin frequency
        let bin_freq = (freq * n as f64 / fs).round() * fs / n as f64;
        let goertzel_result =
            goertzel(&signal, &[bin_freq], fs).expect("goertzel failed");

        // Compute DFT reference using our own FFT
        let complex_signal: Vec<Complex64> = signal
            .iter()
            .map(|&s| Complex64::new(s, 0.0))
            .collect();
        let mut buf = complex_signal;
        // pad to power of 2 = 256
        fft_inplace(&mut buf, false).expect("fft failed");

        // Find the bin corresponding to bin_freq
        let bin_idx = (bin_freq * n as f64 / fs).round() as usize;
        let fft_mag = buf[bin_idx].norm();
        let goertzel_mag = goertzel_result[0].norm();

        // Magnitudes should match within 0.1%
        assert_relative_eq!(goertzel_mag, fft_mag, epsilon = fft_mag * 0.001 + 0.01);
    }

    #[test]
    fn test_goertzel_output_length() {
        let signal = make_tone(1000.0, 256, 8000.0);
        let freqs = [500.0, 1000.0, 2000.0, 3000.0];
        let result = goertzel(&signal, &freqs, 8000.0).expect("goertzel failed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_goertzel_empty_signal_error() {
        assert!(goertzel(&[], &[1000.0], 8000.0).is_err());
    }

    #[test]
    fn test_goertzel_out_of_range_freq_error() {
        let signal = vec![0.0; 64];
        assert!(goertzel(&signal, &[5000.0], 8000.0).is_err()); // f > fs/2
    }

    #[test]
    fn test_goertzel_dc_component() {
        // DC signal: all ones
        let signal = vec![1.0_f64; 64];
        let result = goertzel(&signal, &[0.0], 1000.0).expect("goertzel failed");
        // DC DFT value should be N (sum of all samples)
        assert_relative_eq!(result[0].re, 64.0, epsilon = 1e-8);
        assert_relative_eq!(result[0].im, 0.0, epsilon = 1e-8);
    }

    // --- SlidingDft tests ---

    #[test]
    fn test_sliding_dft_output_length() {
        let mut sdft = SlidingDft::new(vec![100.0, 200.0, 300.0], 1000.0, 64);
        let spectrum = sdft.push(1.0);
        assert_eq!(spectrum.len(), 3);
    }

    #[test]
    fn test_sliding_dft_window_fills() {
        let fs = 1000.0;
        let window = 32;
        let freq = 100.0;
        let mut sdft = SlidingDft::new(vec![freq], fs, window);

        // Push a full window of a 100 Hz tone
        for i in 0..(window * 2) {
            let s = (2.0 * PI * freq * i as f64 / fs).sin();
            let spectrum = sdft.push(s);
            assert_eq!(spectrum.len(), 1);
        }
        // After a full window, the DFT at 100 Hz should have non-zero magnitude
        let final_spectrum = sdft.push(0.0);
        let mag = final_spectrum[0].norm();
        // Should detect significant energy at 100 Hz
        assert!(mag > 0.0, "SlidingDft should have non-zero output after window fills");
    }

    #[test]
    fn test_sliding_dft_reset() {
        let mut sdft = SlidingDft::new(vec![100.0], 1000.0, 32);
        for i in 0..32_usize {
            sdft.push(i as f64 * 0.1);
        }
        sdft.reset();
        assert_eq!(sdft.buffered(), 0);
        let spectrum = sdft.push(0.0);
        assert_relative_eq!(spectrum[0].norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sliding_dft_accessors() {
        let freqs = vec![50.0, 100.0];
        let fs = 500.0;
        let window = 64;
        let sdft = SlidingDft::new(freqs.clone(), fs, window);
        assert_eq!(sdft.freqs(), freqs.as_slice());
        assert_relative_eq!(sdft.fs(), fs, epsilon = 1e-10);
        assert_eq!(sdft.window_size(), window);
        assert_eq!(sdft.buffered(), 0);
    }

    #[test]
    fn test_zoom_fft_freqs_single_point() {
        let freqs = zoom_fft_freqs(300.0, 500.0, 1);
        assert_eq!(freqs.len(), 1);
        assert_relative_eq!(freqs[0], 300.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zoom_fft_freqs_empty() {
        let freqs = zoom_fft_freqs(100.0, 200.0, 0);
        assert!(freqs.is_empty());
    }
}
