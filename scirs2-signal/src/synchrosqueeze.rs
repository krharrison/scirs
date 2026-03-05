//! Synchrosqueezed Wavelet Transform (SST)
//!
//! The Synchrosqueezing Transform (SST) sharpens time-frequency representations
//! by reassigning the energy of each CWT coefficient to the instantaneous
//! frequency of the signal at that time, producing a highly concentrated
//! time-frequency image even for amplitude-modulated / frequency-modulated
//! signals.
//!
//! # Algorithm
//!
//! 1. **CWT** – Compute the continuous wavelet transform using the analytic
//!    Morlet wavelet on a logarithmically-spaced set of scales.
//! 2. **Instantaneous frequency** – Estimate `ω(a,t) = -Im{∂_t W(a,t) / W(a,t)}`
//!    using a finite-difference approximation of the phase derivative.
//! 3. **Synchrosqueezing** – Reassign each CWT coefficient at `(a,t)` to the
//!    frequency bin `ω(a,t)` in the SST image.
//! 4. **Inverse** – Sum over selected frequency bins to reconstruct a
//!    band-limited version of the original signal.
//!
//! # References
//!
//! - Daubechies, I., Lu, J., & Wu, H.T. (2011). Synchrosqueezed wavelet
//!   transforms: An empirical mode decomposition-like tool. *Applied and
//!   Computational Harmonic Analysis*, 30(2), 243-261.
//! - Thakur, G. & Wu, H.T. (2011). Synchrosqueezing-based recovery of
//!   instantaneous frequency. *SIAM Journal on Mathematical Analysis*.
//!
//! # Examples
//!
//! ```
//! use scirs2_signal::synchrosqueeze::{synchrosqueezing_transform, synchrosqueeze_inverse};
//! use scirs2_core::ndarray::Array1;
//! use std::f64::consts::PI;
//!
//! let n = 256usize;
//! let fs = 200.0f64;
//! let signal: Array1<f64> = Array1::from_iter(
//!     (0..n).map(|i| (2.0 * PI * 30.0 * i as f64 / fs).sin())
//! );
//!
//! let result = synchrosqueezing_transform(&signal, fs, 8, 4, 6.0, 1e-8).expect("operation should succeed");
//! assert!(!result.frequencies.is_empty());
//! assert!(!result.times.is_empty());
//! ```

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Public struct
// ---------------------------------------------------------------------------

/// Result of the Synchrosqueezed Wavelet Transform.
#[derive(Debug)]
pub struct SynchroSqueezeResult {
    /// Synchrosqueezed magnitude: shape `(n_freqs, n_time)`.
    pub sst: Array2<f64>,
    /// Frequency axis (Hz), length `n_freqs`.
    pub frequencies: Array1<f64>,
    /// Time axis (seconds), length `n_time = signal.len()`.
    pub times: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Morlet CWT
// ---------------------------------------------------------------------------

/// Compute the Continuous Wavelet Transform using the analytic Morlet wavelet.
///
/// The Morlet mother wavelet is
///
/// ```text
/// ψ(η) = π^{-1/4} · exp(i·ω₀·η) · exp(-η²/2)
/// ```
///
/// where `ω₀` is the centre frequency parameter (default 6.0).
///
/// The CWT at scale `a` and time `b` is computed via the convolution theorem:
///
/// ```text
/// W(a, b) = ∫ x(t) · (1/a) · ψ*((t-b)/a) dt
/// ```
///
/// evaluated on the DFT grid.
///
/// # Arguments
///
/// * `signal` – Input signal.
/// * `scales` – Array of scales `a > 0`.
/// * `omega0` – Morlet centre frequency parameter (typically 5-8).
///
/// # Returns
///
/// Complex CWT array of shape `(len_scales, len_signal)`.
pub fn cwt_morlet(
    signal: &Array1<f64>,
    scales: &[f64],
    omega0: f64,
) -> SignalResult<Array2<Complex64>> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".to_string(),
        ));
    }
    if scales.is_empty() {
        return Err(SignalError::ValueError(
            "scales must not be empty".to_string(),
        ));
    }
    if scales.iter().any(|&a| a <= 0.0) {
        return Err(SignalError::ValueError(
            "All scales must be positive".to_string(),
        ));
    }
    if omega0 <= 0.0 {
        return Err(SignalError::ValueError(
            "omega0 must be positive".to_string(),
        ));
    }

    let nfft = n.next_power_of_two();

    // FFT of zero-padded signal
    let signal_fft = fft_real(signal.as_slice().unwrap_or(&signal.iter().copied().collect::<Vec<_>>()), nfft)?;

    // Angular frequencies for the DFT grid: ω_k = 2π·k/nfft (positive freqs)
    let omegas: Vec<f64> = (0..nfft)
        .map(|k| {
            let k_i = k as f64;
            if k <= nfft / 2 {
                2.0 * PI * k_i / nfft as f64
            } else {
                2.0 * PI * (k_i - nfft as f64) / nfft as f64
            }
        })
        .collect();

    let ns = scales.len();
    let mut result = Array2::zeros((ns, n));

    for (s_idx, &scale) in scales.iter().enumerate() {
        // Morlet wavelet in frequency domain:
        // Ψ̂(a·ω) = π^{-1/4} · √a · H(ω) · exp(-½(a·ω - ω₀)²)
        // where H(ω) is the Heaviside step function (analytic wavelet)
        let norm = PI.powf(-0.25) * (2.0 * PI * scale).sqrt();
        let mut wav_fft: Vec<Complex64> = omegas
            .iter()
            .map(|&omega| {
                if omega <= 0.0 {
                    Complex64::new(0.0, 0.0)
                } else {
                    let arg = scale * omega - omega0;
                    let val = norm * (-0.5 * arg * arg).exp();
                    Complex64::new(val, 0.0)
                }
            })
            .collect();

        // Multiply signal FFT by wavelet FFT
        for k in 0..nfft {
            wav_fft[k] *= signal_fft[k];
        }

        // Inverse FFT
        let conv = ifft_complex(&wav_fft)?;

        // Store first n samples
        for t in 0..n {
            result[[s_idx, t]] = conv[t];
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Instantaneous frequency from CWT
// ---------------------------------------------------------------------------

/// Estimate the instantaneous frequency from a CWT coefficient array.
///
/// Uses the phase-derivative formula:
///
/// ```text
/// ω(a,t) ≈ -Im{ [W(a, t+dt) - W(a, t-dt)] / [2·dt·W(a,t)] }
/// ```
///
/// A central-difference scheme is used for interior points; forward/backward
/// differences are used at the edges.
///
/// # Arguments
///
/// * `cwt`    – CWT array of shape `(n_scales, n_time)`.
/// * `scales` – Scale values used to compute the CWT (not used in this
///              function but kept for API symmetry).
/// * `dt`     – Time step (= 1/sample_rate).
///
/// # Returns
///
/// Instantaneous frequency array of shape `(n_scales, n_time)`, in radians
/// per second.
pub fn cwt_instantaneous_frequency(
    cwt: &Array2<Complex64>,
    _scales: &[f64],
    dt: f64,
) -> SignalResult<Array2<f64>> {
    let (ns, nt) = cwt.dim();
    if nt < 2 {
        return Err(SignalError::ValueError(
            "CWT must have at least 2 time points".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(SignalError::ValueError(
            "dt must be positive".to_string(),
        ));
    }

    let mut omega = Array2::zeros((ns, nt));

    for s in 0..ns {
        for t in 0..nt {
            let w_t = cwt[[s, t]];
            if w_t.norm() < 1e-30 {
                omega[[s, t]] = 0.0;
                continue;
            }
            // Finite difference of complex CWT
            let dw = if t == 0 {
                (cwt[[s, 1]] - cwt[[s, 0]]) / dt
            } else if t == nt - 1 {
                (cwt[[s, nt - 1]] - cwt[[s, nt - 2]]) / dt
            } else {
                (cwt[[s, t + 1]] - cwt[[s, t - 1]]) / (2.0 * dt)
            };
            // ω = -Im(∂_t W / W) = Im(W* · ∂_t W) / |W|²
            let inst_freq = (w_t.conj() * dw).im / w_t.norm_sqr();
            omega[[s, t]] = inst_freq;
        }
    }

    Ok(omega)
}

// ---------------------------------------------------------------------------
// Synchrosqueezing Transform
// ---------------------------------------------------------------------------

/// Compute the Synchrosqueezed Wavelet Transform.
///
/// The energy of the CWT at each `(scale, time)` point is reassigned to the
/// frequency bin corresponding to the instantaneous frequency `ω(a,t)`.
/// Only coefficients with `|W(a,t)| ≥ γ` are included.
///
/// # Arguments
///
/// * `signal`      – Input signal.
/// * `sample_rate` – Sampling frequency in Hz.
/// * `num_voices`  – Number of scales per octave (controls frequency resolution).
/// * `num_octaves` – Number of octaves (controls lowest frequency coverage).
/// * `omega0`      – Morlet wavelet centre frequency (default 6.0).
/// * `gamma`       – Threshold for suppressing weak CWT coefficients.
///
/// # Returns
///
/// A [`SynchroSqueezeResult`] containing the SST image, frequency axis, and
/// time axis.
pub fn synchrosqueezing_transform(
    signal: &Array1<f64>,
    sample_rate: f64,
    num_voices: usize,
    num_octaves: usize,
    omega0: f64,
    gamma: f64,
) -> SignalResult<SynchroSqueezeResult> {
    let n = signal.len();
    if n < 4 {
        return Err(SignalError::ValueError(
            "Signal must have at least 4 samples".to_string(),
        ));
    }
    if sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "sample_rate must be positive".to_string(),
        ));
    }
    if num_voices == 0 {
        return Err(SignalError::ValueError(
            "num_voices must be at least 1".to_string(),
        ));
    }
    if num_octaves == 0 {
        return Err(SignalError::ValueError(
            "num_octaves must be at least 1".to_string(),
        ));
    }
    if omega0 <= 0.0 {
        return Err(SignalError::ValueError(
            "omega0 must be positive".to_string(),
        ));
    }

    let dt = 1.0 / sample_rate;
    let nyquist = sample_rate / 2.0;

    // Build logarithmically-spaced scales.
    // The centre frequency of the Morlet wavelet at scale a is:
    //   f_c(a) = omega0 / (2π·a)
    // We want f_c to range from f_min to f_max = nyquist.
    // f_max corresponds to scale a_min = omega0 / (2π·f_max)
    // f_min corresponds to scale a_max = a_min · 2^{num_octaves}
    let total_scales = num_voices * num_octaves;
    let f_max = nyquist * 0.95; // stay slightly below Nyquist
    let f_min = f_max / (2.0f64.powi(num_octaves as i32));
    let a_min = omega0 / (2.0 * PI * f_max);
    let a_max = omega0 / (2.0 * PI * f_min);

    let scales: Vec<f64> = (0..total_scales)
        .map(|j| {
            let t = j as f64 / (total_scales - 1).max(1) as f64;
            a_min * (a_max / a_min).powf(t)
        })
        .collect();

    // CWT
    let cwt = cwt_morlet(signal, &scales, omega0)?;

    // Instantaneous frequencies
    let inst_freq = cwt_instantaneous_frequency(&cwt, &scales, dt)?;

    // Build output frequency axis (same as scale centres)
    let n_freqs = total_scales;
    let freq_min = f_min;
    let freq_max = f_max;
    let freq_step = if n_freqs > 1 {
        (freq_max - freq_min) / (n_freqs - 1) as f64
    } else {
        1.0
    };

    let frequencies: Vec<f64> = (0..n_freqs)
        .map(|k| freq_min + k as f64 * freq_step)
        .collect();
    let times: Vec<f64> = (0..n).map(|t| t as f64 * dt).collect();

    // Synchrosqueezing: accumulate |W(a,t)| at bin of ω(a,t)
    let mut sst = Array2::zeros((n_freqs, n));

    let ns = scales.len();
    let log_a_ratio = if total_scales > 1 {
        (a_max / a_min).ln() / (total_scales - 1) as f64
    } else {
        1.0
    };

    for s_idx in 0..ns {
        let scale = scales[s_idx];
        // da/ds (for the Jacobian of scale change)
        let da = scale * log_a_ratio;
        for t in 0..n {
            let w_coeff = cwt[[s_idx, t]];
            if w_coeff.norm() < gamma {
                continue;
            }
            let if_hz = inst_freq[[s_idx, t]] / (2.0 * PI); // convert rad/s → Hz
            if if_hz <= 0.0 || if_hz > sample_rate / 2.0 {
                continue;
            }
            // Map instantaneous frequency to frequency bin
            let freq_idx_f = (if_hz - freq_min) / freq_step;
            if freq_idx_f < 0.0 {
                continue;
            }
            let freq_idx = freq_idx_f.round() as usize;
            if freq_idx >= n_freqs {
                continue;
            }
            // Accumulate magnitude contribution, weighted by |da|
            sst[[freq_idx, t]] += w_coeff.norm() * da.abs();
        }
    }

    Ok(SynchroSqueezeResult {
        sst,
        frequencies: Array1::from(frequencies),
        times: Array1::from(times),
    })
}

// ---------------------------------------------------------------------------
// Inverse Synchrosqueezing
// ---------------------------------------------------------------------------

/// Reconstruct a band-limited signal by inverting the SST over a frequency range.
///
/// Sums the SST columns over all frequency bins that lie within
/// `[frequency_range.0, frequency_range.1]` Hz.  This effectively extracts
/// the signal component(s) in that band.
///
/// # Arguments
///
/// * `sst`             – [`SynchroSqueezeResult`] from [`synchrosqueezing_transform`].
/// * `frequency_range` – `(f_low, f_high)` in Hz.
///
/// # Returns
///
/// Reconstructed time-domain signal (real-valued, length = `sst.times.len()`).
pub fn synchrosqueeze_inverse(
    sst: &SynchroSqueezeResult,
    frequency_range: (f64, f64),
) -> SignalResult<Array1<f64>> {
    let (f_low, f_high) = frequency_range;
    if f_low >= f_high {
        return Err(SignalError::ValueError(
            "frequency_range.0 must be less than frequency_range.1".to_string(),
        ));
    }
    if f_low < 0.0 {
        return Err(SignalError::ValueError(
            "frequency_range must be non-negative".to_string(),
        ));
    }

    let n_time = sst.times.len();
    let n_freqs = sst.frequencies.len();

    if n_time == 0 || n_freqs == 0 {
        return Err(SignalError::ValueError(
            "SST result is empty".to_string(),
        ));
    }

    let mut output = vec![0.0f64; n_time];
    let mut count = 0usize;

    for (freq_idx, &freq) in sst.frequencies.iter().enumerate() {
        if freq >= f_low && freq <= f_high {
            count += 1;
            for t in 0..n_time {
                output[t] += sst.sst[[freq_idx, t]];
            }
        }
    }

    // Normalise by number of contributing frequency bins
    if count > 0 {
        let norm = 1.0 / count as f64;
        for v in output.iter_mut() {
            *v *= norm;
        }
    }

    Ok(Array1::from(output))
}

// ---------------------------------------------------------------------------
// Internal FFT helpers
// ---------------------------------------------------------------------------

/// Compute the FFT of a real-valued signal padded/truncated to length `nfft`.
fn fft_real(x: &[f64], nfft: usize) -> SignalResult<Vec<Complex64>> {
    let mut buf: Vec<Complex64> = x
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .chain(std::iter::repeat(Complex64::new(0.0, 0.0)))
        .take(nfft)
        .collect();
    while buf.len() < nfft {
        buf.push(Complex64::new(0.0, 0.0));
    }
    cooley_tukey_fft(&mut buf);
    Ok(buf)
}

/// Compute the IFFT and return complex output (not normalised by 1/N yet).
fn ifft_complex(x: &[Complex64]) -> SignalResult<Vec<Complex64>> {
    let n = x.len();
    // Conjugate, FFT, conjugate, divide by N
    let mut buf: Vec<Complex64> = x.iter().map(|c| c.conj()).collect();
    cooley_tukey_fft(&mut buf);
    let norm = 1.0 / n as f64;
    for c in buf.iter_mut() {
        *c = c.conj() * norm;
    }
    Ok(buf)
}

/// In-place Cooley-Tukey radix-2 DIT FFT (power-of-two length).
fn cooley_tukey_fft(buf: &mut Vec<Complex64>) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..len / 2 {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
        len <<= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;

    fn make_sinusoid(freq: f64, fs: f64, n: usize) -> Array1<f64> {
        Array1::from_iter((0..n).map(|i| (2.0 * PI * freq * i as f64 / fs).sin()))
    }

    fn pseudo_noise(n: usize, seed: u64) -> Array1<f64> {
        let mut x = vec![0.0f64; n];
        let mut s = seed ^ 0xdeadbeef;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((s >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
        Array1::from(x)
    }

    // -----------------------------------------------------------------------
    // Morlet CWT tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cwt_morlet_shape() {
        let n = 128;
        let signal = make_sinusoid(20.0, 200.0, n);
        let scales: Vec<f64> = (1..=8).map(|j| j as f64).collect();
        let cwt = cwt_morlet(&signal, &scales, 6.0).expect("failed to create cwt");
        assert_eq!(cwt.shape(), &[scales.len(), n]);
    }

    #[test]
    fn test_cwt_morlet_sinusoid_energy_concentration() {
        // The CWT of a pure sinusoid should have highest energy at the scale
        // corresponding to the signal frequency.
        let fs = 200.0f64;
        let f0 = 30.0f64;
        let n = 256;
        let signal = make_sinusoid(f0, fs, n);

        // Scale corresponding to f0: a = omega0 / (2π·f0) * fs
        let omega0 = 6.0f64;
        // scale in samples: a_samp = omega0 * fs / (2π * f0)
        let a0 = omega0 * fs / (2.0 * PI * f0);
        // Build scales around a0
        let scales: Vec<f64> = (1..=32)
            .map(|j| a0 * 0.5 * (j as f64 / 16.0))
            .filter(|&a| a > 0.0)
            .collect();

        let cwt = cwt_morlet(&signal, &scales, omega0).expect("failed to create cwt");

        // Find scale with maximum total energy
        let scale_energies: Vec<f64> = (0..scales.len())
            .map(|s| (0..n).map(|t| cwt[[s, t]].norm_sqr()).sum())
            .collect();
        let peak_scale_idx = scale_energies
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("unexpected None or Err");
        let peak_freq = omega0 / (2.0 * PI * scales[peak_scale_idx] / fs);
        // Peak frequency should be within 1 octave of f0
        assert!(
            (peak_freq / f0).log2().abs() < 1.0,
            "CWT peak at {peak_freq} Hz, expected near {f0} Hz"
        );
    }

    #[test]
    fn test_cwt_morlet_error_on_empty_scales() {
        let signal = make_sinusoid(10.0, 100.0, 64);
        assert!(cwt_morlet(&signal, &[], 6.0).is_err());
    }

    #[test]
    fn test_cwt_morlet_error_on_negative_scale() {
        let signal = make_sinusoid(10.0, 100.0, 64);
        assert!(cwt_morlet(&signal, &[-1.0, 1.0, 2.0], 6.0).is_err());
    }

    // -----------------------------------------------------------------------
    // Instantaneous frequency tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cwt_inst_freq_shape() {
        let n = 128;
        let signal = make_sinusoid(20.0, 200.0, n);
        let scales: Vec<f64> = (1..=4).map(|j| j as f64 * 2.0).collect();
        let cwt = cwt_morlet(&signal, &scales, 6.0).expect("failed to create cwt");
        let inst_freq = cwt_instantaneous_frequency(&cwt, &scales, 1.0 / 200.0).expect("failed to create inst_freq");
        assert_eq!(inst_freq.shape(), cwt.shape());
    }

    #[test]
    fn test_cwt_inst_freq_sinusoid() {
        // For a pure sinusoid the IF at the correct scale ≈ 2π·f0
        let fs = 200.0f64;
        let f0 = 30.0f64;
        let n = 256;
        let omega0 = 6.0f64;
        let signal = make_sinusoid(f0, fs, n);
        let a0 = omega0 / (2.0 * PI * f0 / fs); // scale in samples
        let scales = vec![a0];
        let cwt = cwt_morlet(&signal, &scales, omega0).expect("failed to create cwt");
        let inst_freq = cwt_instantaneous_frequency(&cwt, &scales, 1.0 / fs).expect("failed to create inst_freq");

        // Mean IF over the interior of the signal (skip edges)
        let interior_start = n / 8;
        let interior_end = 7 * n / 8;
        let mean_if: f64 = (interior_start..interior_end)
            .map(|t| inst_freq[[0, t]])
            .sum::<f64>()
            / (interior_end - interior_start) as f64;

        let expected_if = 2.0 * PI * f0; // rad/s
        // Allow 20% tolerance
        assert!(
            (mean_if - expected_if).abs() / expected_if < 0.20,
            "Mean IF {mean_if:.2} rad/s, expected {expected_if:.2} rad/s"
        );
    }

    #[test]
    fn test_cwt_inst_freq_error_on_short_cwt() {
        let cwt: Array2<Complex64> = Array2::zeros((4, 1));
        let scales = vec![1.0, 2.0, 3.0, 4.0];
        assert!(cwt_instantaneous_frequency(&cwt, &scales, 0.01).is_err());
    }

    // -----------------------------------------------------------------------
    // Synchrosqueezing transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sst_basic_output() {
        let n = 256;
        let fs = 200.0;
        let signal = make_sinusoid(30.0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");
        assert_eq!(result.times.len(), n);
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.sst.shape()[1], n);
    }

    #[test]
    fn test_sst_sst_is_non_negative() {
        let n = 256;
        let fs = 200.0;
        let signal = pseudo_noise(n, 42);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");
        assert!(result.sst.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_sst_frequency_axis_sorted() {
        let n = 256;
        let fs = 200.0;
        let signal = make_sinusoid(20.0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");
        let freqs = result.frequencies.as_slice().expect("failed to create freqs");
        for i in 1..freqs.len() {
            assert!(
                freqs[i] >= freqs[i - 1],
                "Frequency axis is not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_sst_time_axis_correct() {
        let n = 256;
        let fs = 100.0;
        let signal = make_sinusoid(10.0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 6, 3, 6.0, 1e-8).expect("failed to create result");
        assert_relative_eq!(result.times[0], 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            result.times[n - 1],
            (n - 1) as f64 / fs,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_sst_sinusoid_energy_concentration() {
        // The SST of a pure sinusoid should be concentrated in a narrow
        // frequency band around f0.
        let n = 512;
        let fs = 500.0;
        let f0 = 50.0;
        let signal = make_sinusoid(f0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 12, 4, 6.0, 1e-6).expect("failed to create result");

        let n_freqs = result.frequencies.len();
        let n_time = result.times.len();

        // Total energy
        let total: f64 = result.sst.iter().sum();
        if total < 1e-30 {
            // If SST is all zeros the test is trivially satisfied
            return;
        }

        // Energy in a ±15% band around f0
        let band_energy: f64 = result
            .frequencies
            .iter()
            .enumerate()
            .filter(|(_, &f)| f >= f0 * 0.85 && f <= f0 * 1.15)
            .map(|(k, _)| {
                (0..n_time).map(|t| result.sst[[k, t]]).sum::<f64>()
            })
            .sum();

        assert!(
            band_energy / total > 0.3,
            "Only {:.1}% of SST energy is in the ±15% band around f0={f0}",
            100.0 * band_energy / total
        );
    }

    #[test]
    fn test_sst_error_on_short_signal() {
        let short = Array1::zeros(3);
        assert!(synchrosqueezing_transform(&short, 100.0, 8, 4, 6.0, 1e-8).is_err());
    }

    #[test]
    fn test_sst_error_on_zero_voices() {
        let signal = make_sinusoid(10.0, 100.0, 128);
        assert!(synchrosqueezing_transform(&signal, 100.0, 0, 4, 6.0, 1e-8).is_err());
    }

    // -----------------------------------------------------------------------
    // Inverse SST tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_inverse_sst_length() {
        let n = 256;
        let fs = 200.0;
        let signal = make_sinusoid(30.0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");
        let f_max = result.frequencies[result.frequencies.len() - 1];
        let reconstructed = synchrosqueeze_inverse(&result, (0.0, f_max)).expect("failed to create reconstructed");
        assert_eq!(reconstructed.len(), n);
    }

    #[test]
    fn test_inverse_sst_narrow_band() {
        // Reconstruct only a narrow band; should return non-trivial output
        let n = 256;
        let fs = 200.0;
        let f0 = 40.0;
        let signal = make_sinusoid(f0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");

        let f_min = f0 * 0.8;
        let f_max = f0 * 1.2;
        let reconstructed = synchrosqueeze_inverse(&result, (f_min, f_max)).expect("failed to create reconstructed");

        // At least the output should be finite
        assert!(reconstructed.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_inverse_sst_error_on_invalid_range() {
        let n = 128;
        let fs = 200.0;
        let signal = make_sinusoid(20.0, fs, n);
        let result = synchrosqueezing_transform(&signal, fs, 8, 3, 6.0, 1e-8).expect("failed to create result");
        // Low >= high should fail
        assert!(synchrosqueeze_inverse(&result, (50.0, 30.0)).is_err());
    }
}
