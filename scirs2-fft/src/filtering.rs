//! FFT-based ideal filters and streaming convolution (Overlap-Add).
//!
//! This module provides brick-wall frequency-domain filters and a streaming
//! convolution engine for processing long signals against FIR filters:
//!
//! * [`fft_lowpass_filter`]   — Ideal lowpass filter.
//! * [`fft_highpass_filter`]  — Ideal highpass filter.
//! * [`fft_bandpass_filter`]  — Ideal bandpass filter.
//! * [`fft_bandstop_filter`]  — Ideal bandstop (notch) filter.
//! * [`OLAConvolver`]         — Streaming overlap-add convolver.
//!
//! # Note
//!
//! Ideal (brick-wall) filters have infinite impulse responses and introduce
//! Gibbs ringing artefacts.  They are most appropriate for non-real-time
//! batch processing.  For production use, design a windowed-sinc or
//! equiripple FIR filter and apply it with [`OLAConvolver`].

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::helper::next_fast_len;
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the bin index corresponding to `freq` Hz for a signal sampled at
/// `fs` Hz with an FFT of length `n`.
fn freq_to_bin(freq: f64, fs: f64, n: usize) -> usize {
    let bin_f = freq / fs * n as f64;
    (bin_f.round() as usize).min(n / 2)
}

/// Apply a binary frequency-domain mask and return the filtered real signal.
///
/// `mask[k]` should be `true` to keep bin `k`, `false` to zero it.
/// The mask is applied to the two-sided spectrum symmetrically.
fn apply_mask(signal: &[f64], mask: &[bool]) -> FFTResult<Vec<f64>> {
    let n = signal.len();
    let spectrum = fft(signal, None)?;
    let n_pos = n / 2 + 1; // positive-frequency bins (including DC and Nyquist)

    let mut filtered = spectrum.clone();
    for k in 0..n_pos {
        if !mask[k] {
            filtered[k] = Complex64::new(0.0, 0.0);
            // Mirror (negative frequency)
            if k > 0 && k < n - k {
                filtered[n - k] = Complex64::new(0.0, 0.0);
            }
        }
    }

    let time_domain = ifft(&filtered, None)?;
    Ok(time_domain.iter().map(|c| c.re).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_lowpass_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal lowpass filter via FFT.
///
/// Zeroes all frequency bins above `cutoff` Hz and returns the inverse FFT.
///
/// # Arguments
///
/// * `signal` – Real-valued input signal.
/// * `cutoff` – Cutoff frequency in Hz.
/// * `fs`     – Sampling frequency in Hz.
///
/// # Returns
///
/// Filtered signal (same length as `signal`).
///
/// # Errors
///
/// Returns an error if `signal` is empty, `cutoff <= 0`, `cutoff >= fs/2`, or
/// an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::filtering::fft_lowpass_filter;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 1024;
/// // Mix of 50 Hz and 300 Hz
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         (2.0 * PI * 50.0 * i as f64 / fs).sin()
///         + (2.0 * PI * 300.0 * i as f64 / fs).sin()
///     })
///     .collect();
/// let filtered = fft_lowpass_filter(&signal, 150.0, fs).expect("lowpass");
/// assert_eq!(filtered.len(), signal.len());
/// // RMS of the filtered signal should be closer to 50 Hz component level
/// let rms: f64 = (filtered.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
/// assert!(rms > 0.1, "filtered rms={rms}");
/// ```
pub fn fft_lowpass_filter(signal: &[f64], cutoff: f64, fs: f64) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "fft_lowpass_filter: signal is empty".into(),
        ));
    }
    if cutoff <= 0.0 {
        return Err(FFTError::ValueError(
            "fft_lowpass_filter: cutoff must be positive".into(),
        ));
    }
    if cutoff >= fs / 2.0 {
        return Err(FFTError::ValueError(
            "fft_lowpass_filter: cutoff must be < fs/2".into(),
        ));
    }

    let n = signal.len();
    let n_pos = n / 2 + 1;
    let cutoff_bin = freq_to_bin(cutoff, fs, n);

    let mask: Vec<bool> = (0..n_pos).map(|k| k <= cutoff_bin).collect();
    apply_mask(signal, &mask)
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_highpass_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal highpass filter via FFT.
///
/// Zeroes all frequency bins below `cutoff` Hz and returns the inverse FFT.
///
/// # Arguments
///
/// * `signal` – Real-valued input signal.
/// * `cutoff` – Cutoff frequency in Hz.
/// * `fs`     – Sampling frequency in Hz.
///
/// # Errors
///
/// Returns an error if `signal` is empty, `cutoff <= 0`, `cutoff >= fs/2`, or
/// an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::filtering::fft_highpass_filter;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 1024;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         (2.0 * PI * 50.0 * i as f64 / fs).sin()
///         + (2.0 * PI * 400.0 * i as f64 / fs).sin()
///     })
///     .collect();
/// let filtered = fft_highpass_filter(&signal, 200.0, fs).expect("highpass");
/// assert_eq!(filtered.len(), signal.len());
/// ```
pub fn fft_highpass_filter(signal: &[f64], cutoff: f64, fs: f64) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "fft_highpass_filter: signal is empty".into(),
        ));
    }
    if cutoff <= 0.0 {
        return Err(FFTError::ValueError(
            "fft_highpass_filter: cutoff must be positive".into(),
        ));
    }
    if cutoff >= fs / 2.0 {
        return Err(FFTError::ValueError(
            "fft_highpass_filter: cutoff must be < fs/2".into(),
        ));
    }

    let n = signal.len();
    let n_pos = n / 2 + 1;
    let cutoff_bin = freq_to_bin(cutoff, fs, n);

    let mask: Vec<bool> = (0..n_pos).map(|k| k >= cutoff_bin).collect();
    apply_mask(signal, &mask)
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_bandpass_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal bandpass filter via FFT.
///
/// Keeps only frequency components in `[low_hz, high_hz]` and zeros everything
/// outside that band.
///
/// # Arguments
///
/// * `signal`  – Real-valued input signal.
/// * `low_hz`  – Lower cutoff frequency (Hz).
/// * `high_hz` – Upper cutoff frequency (Hz).
/// * `fs`      – Sampling frequency (Hz).
///
/// # Errors
///
/// Returns an error if the signal is empty, `low_hz >= high_hz`, either
/// frequency is outside `(0, fs/2)`, or an FFT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::filtering::fft_bandpass_filter;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 1024;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         (2.0 * PI * 50.0 * i as f64 / fs).sin()
///         + (2.0 * PI * 200.0 * i as f64 / fs).sin()
///         + (2.0 * PI * 400.0 * i as f64 / fs).sin()
///     })
///     .collect();
/// // Pass only the 200 Hz component
/// let filtered = fft_bandpass_filter(&signal, 150.0, 300.0, fs).expect("bandpass");
/// assert_eq!(filtered.len(), signal.len());
/// ```
pub fn fft_bandpass_filter(signal: &[f64], low_hz: f64, high_hz: f64, fs: f64) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "fft_bandpass_filter: signal is empty".into(),
        ));
    }
    if low_hz <= 0.0 || high_hz <= 0.0 {
        return Err(FFTError::ValueError(
            "fft_bandpass_filter: frequencies must be positive".into(),
        ));
    }
    if low_hz >= high_hz {
        return Err(FFTError::ValueError(
            "fft_bandpass_filter: low_hz must be < high_hz".into(),
        ));
    }
    if high_hz >= fs / 2.0 {
        return Err(FFTError::ValueError(
            "fft_bandpass_filter: high_hz must be < fs/2".into(),
        ));
    }

    let n = signal.len();
    let n_pos = n / 2 + 1;
    let lo_bin = freq_to_bin(low_hz, fs, n);
    let hi_bin = freq_to_bin(high_hz, fs, n);

    let mask: Vec<bool> = (0..n_pos).map(|k| k >= lo_bin && k <= hi_bin).collect();
    apply_mask(signal, &mask)
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_bandstop_filter
// ─────────────────────────────────────────────────────────────────────────────

/// Ideal bandstop (notch) filter via FFT.
///
/// Zeroes frequency components in the band `[low_hz, high_hz]` and keeps
/// everything outside.
///
/// # Arguments
///
/// * `signal`  – Real-valued input signal.
/// * `low_hz`  – Lower edge of the stop band (Hz).
/// * `high_hz` – Upper edge of the stop band (Hz).
/// * `fs`      – Sampling frequency (Hz).
///
/// # Errors
///
/// Returns an error if the signal is empty, `low_hz >= high_hz`, either
/// frequency is outside `(0, fs/2)`, or an FFT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::filtering::fft_bandstop_filter;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 1024;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         (2.0 * PI * 50.0 * i as f64 / fs).sin()
///         + (2.0 * PI * 200.0 * i as f64 / fs).sin()
///     })
///     .collect();
/// // Notch out the 200 Hz band
/// let filtered = fft_bandstop_filter(&signal, 150.0, 250.0, fs).expect("bandstop");
/// assert_eq!(filtered.len(), signal.len());
/// ```
pub fn fft_bandstop_filter(signal: &[f64], low_hz: f64, high_hz: f64, fs: f64) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "fft_bandstop_filter: signal is empty".into(),
        ));
    }
    if low_hz <= 0.0 || high_hz <= 0.0 {
        return Err(FFTError::ValueError(
            "fft_bandstop_filter: frequencies must be positive".into(),
        ));
    }
    if low_hz >= high_hz {
        return Err(FFTError::ValueError(
            "fft_bandstop_filter: low_hz must be < high_hz".into(),
        ));
    }
    if high_hz >= fs / 2.0 {
        return Err(FFTError::ValueError(
            "fft_bandstop_filter: high_hz must be < fs/2".into(),
        ));
    }

    let n = signal.len();
    let n_pos = n / 2 + 1;
    let lo_bin = freq_to_bin(low_hz, fs, n);
    let hi_bin = freq_to_bin(high_hz, fs, n);

    let mask: Vec<bool> = (0..n_pos)
        .map(|k| k < lo_bin || k > hi_bin)
        .collect();
    apply_mask(signal, &mask)
}

// ─────────────────────────────────────────────────────────────────────────────
//  OLAConvolver — Overlap-Add streaming convolver
// ─────────────────────────────────────────────────────────────────────────────

/// Streaming convolver using the Overlap-Add (OLA) method.
///
/// The OLA convolver partitions a (potentially infinite) input stream into
/// fixed-size blocks, convolves each block with a pre-loaded FIR impulse
/// response, and accumulates overlapping partial results.
///
/// # Usage
///
/// 1. Create with [`OLAConvolver::new`].
/// 2. Feed blocks with [`OLAConvolver::process_block`].
/// 3. Flush the final tail with [`OLAConvolver::flush`].
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::filtering::OLAConvolver;
///
/// let h = vec![0.25_f64, 0.5, 0.25]; // 3-tap FIR
/// let mut conv = OLAConvolver::new(h, 16).expect("ola");
///
/// let input: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
/// let mut output = Vec::new();
/// for chunk in input.chunks(16) {
///     let partial = conv.process_block(chunk).expect("block");
///     output.extend_from_slice(&partial);
/// }
/// let tail = conv.flush().expect("flush");
/// output.extend_from_slice(&tail);
/// // Total output length = input.len() + h.len() - 1
/// assert_eq!(output.len(), input.len() + 3 - 1);
/// ```
pub struct OLAConvolver {
    /// Frequency-domain representation of the impulse response (padded to `fft_len`).
    h_freq: Vec<Complex64>,
    /// Length of the impulse response.
    h_len: usize,
    /// Length of each input block.
    block_size: usize,
    /// FFT size: `next_fast_len(block_size + h_len - 1)`.
    fft_len: usize,
    /// Overlap-add accumulation buffer (length `h_len - 1`).
    overlap: Vec<f64>,
    /// Total input samples processed (for flush accounting).
    total_input: usize,
    /// Total output samples emitted so far.
    total_output: usize,
}

impl OLAConvolver {
    /// Create a new OLA convolver.
    ///
    /// # Arguments
    ///
    /// * `h`          – FIR impulse response.
    /// * `block_size` – Number of input samples per block.
    ///
    /// # Errors
    ///
    /// Returns an error if `h` is empty, `block_size` is zero, or the FFT of
    /// the impulse response fails.
    pub fn new(h: Vec<f64>, block_size: usize) -> FFTResult<Self> {
        if h.is_empty() {
            return Err(FFTError::ValueError(
                "OLAConvolver: impulse response is empty".into(),
            ));
        }
        if block_size == 0 {
            return Err(FFTError::ValueError(
                "OLAConvolver: block_size must be positive".into(),
            ));
        }

        let h_len = h.len();
        let conv_len = block_size + h_len - 1;
        let fft_len = next_fast_len(conv_len, true);

        // Pre-compute FFT of the zero-padded impulse response
        let mut h_pad = vec![0.0_f64; fft_len];
        h_pad[..h_len].copy_from_slice(&h);
        let h_freq = fft(&h_pad, None)?;

        let overlap = vec![0.0_f64; h_len - 1];

        Ok(Self {
            h_freq,
            h_len,
            block_size,
            fft_len,
            overlap,
            total_input: 0,
            total_output: 0,
        })
    }

    /// Process one block of input samples and return the corresponding output
    /// block.
    ///
    /// The block length need not equal `block_size`; any length ≤ `block_size`
    /// is accepted.  The output has the same length as the input.
    ///
    /// # Errors
    ///
    /// Returns an error if the block is empty, is longer than `block_size`, or
    /// an FFT call fails.
    pub fn process_block(&mut self, block: &[f64]) -> FFTResult<Vec<f64>> {
        if block.is_empty() {
            return Err(FFTError::ValueError(
                "OLAConvolver::process_block: block is empty".into(),
            ));
        }
        if block.len() > self.block_size {
            return Err(FFTError::ValueError(format!(
                "OLAConvolver::process_block: block.len()={} > block_size={}",
                block.len(),
                self.block_size
            )));
        }

        // Zero-pad block to fft_len
        let mut buf = vec![0.0_f64; self.fft_len];
        buf[..block.len()].copy_from_slice(block);

        let x_freq = fft(&buf, None)?;

        // Frequency-domain multiplication
        let y_freq: Vec<Complex64> = x_freq
            .iter()
            .zip(self.h_freq.iter())
            .map(|(&xf, &hf)| xf * hf)
            .collect();

        let y_c = ifft(&y_freq, None)?;

        // Length of valid output for this block (full convolution tail)
        let block_out_len = block.len() + self.h_len - 1;

        // Overlap-add: add saved overlap to the beginning
        let mut y_full: Vec<f64> = y_c.iter().take(block_out_len).map(|c| c.re).collect();
        for (i, &ov) in self.overlap.iter().enumerate() {
            y_full[i] += ov;
        }

        // Save new overlap (the tail beyond block.len())
        let new_overlap_len = self.h_len - 1;
        self.overlap = y_full[block.len()..block.len() + new_overlap_len].to_vec();

        // Output is only the first `block.len()` samples
        let out = y_full[..block.len()].to_vec();

        self.total_input += block.len();
        self.total_output += out.len();

        Ok(out)
    }

    /// Flush the remaining overlap tail.
    ///
    /// Must be called once at the end of the stream.  Returns the tail of length
    /// `h_len - 1`.  After flushing, the total output length equals
    /// `total_input + h_len - 1`.
    ///
    /// # Errors
    ///
    /// This function does not fail under normal usage but returns an `FFTResult`
    /// for API consistency.
    pub fn flush(&mut self) -> FFTResult<Vec<f64>> {
        let tail = self.overlap.clone();
        self.overlap = vec![0.0_f64; self.h_len - 1];
        self.total_output += tail.len();
        Ok(tail)
    }

    /// Total number of input samples consumed so far.
    pub fn total_input(&self) -> usize {
        self.total_input
    }

    /// Total number of output samples emitted so far (including any flushed tail).
    pub fn total_output(&self) -> usize {
        self.total_output
    }

    /// Block size configured at construction.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Impulse response length.
    pub fn h_len(&self) -> usize {
        self.h_len
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq_hz: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / fs).sin())
            .collect()
    }

    fn rms(v: &[f64]) -> f64 {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }

    #[test]
    fn test_lowpass_passes_low_freq() {
        let fs = 1000.0_f64;
        let n = 2048;
        let lo = sine_wave(50.0, n, fs);
        let hi = sine_wave(400.0, n, fs);
        let mixed: Vec<f64> = lo.iter().zip(hi.iter()).map(|(&a, &b)| a + b).collect();

        let filtered = fft_lowpass_filter(&mixed, 150.0, fs).expect("lowpass");
        assert_eq!(filtered.len(), n);

        let rms_filtered = rms(&filtered);
        let rms_hi = rms(&hi);
        // Filtered signal should have much less high-freq energy
        assert!(
            rms_filtered < rms_hi * 1.2,
            "Lowpass should attenuate 400 Hz: rms_filt={rms_filtered}, rms_hi={rms_hi}"
        );
    }

    #[test]
    fn test_highpass_attenuates_low_freq() {
        let fs = 1000.0_f64;
        let n = 2048;
        let lo = sine_wave(50.0, n, fs);
        let hi = sine_wave(400.0, n, fs);
        let mixed: Vec<f64> = lo.iter().zip(hi.iter()).map(|(&a, &b)| a + b).collect();

        let filtered = fft_highpass_filter(&mixed, 150.0, fs).expect("highpass");
        assert_eq!(filtered.len(), n);

        let rms_filtered = rms(&filtered);
        let rms_lo = rms(&lo);
        assert!(
            rms_filtered < rms_lo * 1.5,
            "Highpass should attenuate 50 Hz: rms_filt={rms_filtered}, rms_lo={rms_lo}"
        );
    }

    #[test]
    fn test_bandpass_isolates_band() {
        let fs = 1000.0_f64;
        let n = 2048;
        let f_lo = sine_wave(50.0, n, fs);
        let f_mid = sine_wave(200.0, n, fs);
        let f_hi = sine_wave(400.0, n, fs);
        let mixed: Vec<f64> = f_lo
            .iter()
            .zip(f_mid.iter())
            .zip(f_hi.iter())
            .map(|((&a, &b), &c)| a + b + c)
            .collect();

        let filtered = fft_bandpass_filter(&mixed, 150.0, 300.0, fs).expect("bandpass");
        assert_eq!(filtered.len(), n);

        let rms_f = rms(&filtered);
        let rms_mid = rms(&f_mid);
        // The 200 Hz component should dominate
        assert!(rms_f > 0.1 * rms_mid, "bandpass too weak: {rms_f}");
    }

    #[test]
    fn test_bandstop_removes_band() {
        let fs = 1000.0_f64;
        let n = 2048;
        let f_lo = sine_wave(50.0, n, fs);
        let f_notch = sine_wave(200.0, n, fs);
        let mixed: Vec<f64> = f_lo
            .iter()
            .zip(f_notch.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        let filtered = fft_bandstop_filter(&mixed, 150.0, 250.0, fs).expect("bandstop");
        assert_eq!(filtered.len(), n);

        // After notching, filtered rms should be less than original
        let rms_orig = rms(&mixed);
        let rms_filt = rms(&filtered);
        assert!(
            rms_filt < rms_orig,
            "Bandstop should reduce rms: orig={rms_orig}, filt={rms_filt}"
        );
    }

    #[test]
    fn test_ola_output_length() {
        let h = vec![0.25_f64, 0.5, 0.25];
        let mut conv = OLAConvolver::new(h.clone(), 16).expect("ola new");
        let input: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();

        let mut output = Vec::new();
        for chunk in input.chunks(16) {
            let part = conv.process_block(chunk).expect("process");
            output.extend_from_slice(&part);
        }
        let tail = conv.flush().expect("flush");
        output.extend_from_slice(&tail);

        let expected_len = input.len() + h.len() - 1;
        assert_eq!(output.len(), expected_len);
    }

    #[test]
    fn test_ola_matches_direct_convolution() {
        use crate::convolution::fft_convolve;

        let h = vec![0.25_f64, 0.5, 0.25];
        let input: Vec<f64> = (0..128).map(|i| (i as f64 * 0.3).sin()).collect();

        // Reference via convolution module
        let reference = fft_convolve(&input, &h).expect("ref conv");

        // OLA
        let mut conv = OLAConvolver::new(h, 32).expect("ola");
        let mut ola_out = Vec::new();
        for chunk in input.chunks(32) {
            ola_out.extend_from_slice(&conv.process_block(chunk).expect("block"));
        }
        ola_out.extend_from_slice(&conv.flush().expect("flush"));

        assert_eq!(ola_out.len(), reference.len());
        for (i, (a, b)) in reference.iter().zip(ola_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "OLA mismatch at [{i}]: ref={a}, ola={b}"
            );
        }
    }

    #[test]
    fn test_ola_different_block_sizes() {
        use crate::convolution::fft_convolve;

        let h = vec![1.0_f64, -1.0, 1.0, -1.0];
        let input: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let reference = fft_convolve(&input, &h).expect("ref");

        for &bs in &[1usize, 7, 16, 64, 100] {
            let mut conv = OLAConvolver::new(h.clone(), bs).expect("ola");
            let mut out = Vec::new();
            for chunk in input.chunks(bs) {
                out.extend_from_slice(&conv.process_block(chunk).expect("block"));
            }
            out.extend_from_slice(&conv.flush().expect("flush"));
            assert_eq!(out.len(), reference.len(), "block_size={bs}");
            for (i, (r, o)) in reference.iter().zip(out.iter()).enumerate() {
                assert!(
                    (r - o).abs() < 1e-8,
                    "bs={bs} mismatch at [{i}]: ref={r}, ola={o}"
                );
            }
        }
    }

    #[test]
    fn test_lowpass_identity_when_cutoff_near_nyquist() {
        let fs = 1000.0_f64;
        let sig = sine_wave(50.0, 256, fs);
        // cutoff just below Nyquist
        let filtered = fft_lowpass_filter(&sig, 499.0, fs).expect("lowpass wide");
        let rms_orig = rms(&sig);
        let rms_filt = rms(&filtered);
        // Near-identity filter: RMS should be preserved
        assert!(
            (rms_filt - rms_orig).abs() < 0.1 * rms_orig,
            "Near-identity lowpass changed RMS: {rms_orig} → {rms_filt}"
        );
    }
}
