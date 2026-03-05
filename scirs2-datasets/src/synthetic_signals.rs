//! Synthetic Signal Datasets.
//!
//! Generates realistic 1-D signal datasets for benchmarking signal-processing,
//! frequency-analysis, and machine-learning algorithms.
//!
//! # Available generators
//!
//! | Function | Description |
//! |---|---|
//! | [`ecg_signal`] | Synthetic ECG waveform with configurable heart rate |
//! | [`seismic_trace`] | Synthetic seismic trace with random events |
//! | [`chirp_signal`] | Frequency-sweep (chirp) signal |
//! | [`am_signal`] | Amplitude-modulated carrier |
//! | [`fm_signal`] | Frequency-modulated carrier |
//! | [`sinusoidal_mixture`] | Sum of sinusoids with additive white noise |

use crate::error::{DatasetsError, Result};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::*;
use scirs2_core::random::rand_distributions::Distribution;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
// Chirp method
// ─────────────────────────────────────────────────────────────────────────────

/// Method used to sweep frequency in [`chirp_signal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChirpMethod {
    /// Frequency increases linearly from `f0` to `f1`.
    Linear,
    /// Frequency sweeps logarithmically (geometric sweep).
    Logarithmic,
    /// Frequency sweeps hyperbolically (constant time-bandwidth product).
    Hyperbolic,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn make_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn normal_dist(std: f64) -> Result<scirs2_core::random::rand_distributions::Normal<f64>> {
    scirs2_core::random::rand_distributions::Normal::new(0.0_f64, std).map_err(|e| {
        DatasetsError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })
}

fn uniform_dist(
    lo: f64,
    hi: f64,
) -> Result<scirs2_core::random::rand_distributions::Uniform<f64>> {
    scirs2_core::random::rand_distributions::Uniform::new(lo, hi).map_err(|e| {
        DatasetsError::ComputationError(format!("Uniform distribution creation failed: {e}"))
    })
}

/// Build a uniform time axis from 0 to `(n-1) / fs`.
fn time_axis(n: usize, fs: f64) -> Array1<f64> {
    Array1::from_vec((0..n).map(|i| i as f64 / fs).collect())
}

fn check_n_fs(func: &str, n: usize, fs: f64) -> Result<()> {
    if n == 0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{func}: n_samples must be > 0"
        )));
    }
    if fs <= 0.0 {
        return Err(DatasetsError::InvalidFormat(format!(
            "{func}: fs must be > 0"
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ECG signal
// ─────────────────────────────────────────────────────────────────────────────

/// Synthesise a single-lead ECG signal.
///
/// Each heartbeat is modelled as a sum of Gaussian bumps that approximate the
/// P, Q, R, S, T morphological waves.  Gaussian white noise is added at the
/// requested SNR.
///
/// # Arguments
///
/// * `n_samples`   – Number of samples.
/// * `fs`          – Sampling frequency in Hz (default: `500.0`).
/// * `heart_rate`  – Heart rate in beats per minute (default: `70.0`).
/// * `noise_level` – Standard deviation of additive Gaussian noise (≥ 0).
/// * `seed`        – Random seed.
///
/// # Returns
///
/// `(time, signal)` – both `Array1<f64>` of length `n_samples`.
///
/// # Errors
///
/// Returns an error when `n_samples == 0`, `fs ≤ 0`, `heart_rate ≤ 0`, or
/// `noise_level < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::ecg_signal;
///
/// let (t, sig) = ecg_signal(2000, 500.0, 70.0, 0.02, 42).expect("ecg failed");
/// assert_eq!(t.len(), 2000);
/// assert_eq!(sig.len(), 2000);
/// ```
pub fn ecg_signal(
    n_samples: usize,
    fs: f64,
    heart_rate: f64,
    noise_level: f64,
    seed: u64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    check_n_fs("ecg_signal", n_samples, fs)?;
    if heart_rate <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "ecg_signal: heart_rate must be > 0".to_string(),
        ));
    }
    if noise_level < 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "ecg_signal: noise_level must be >= 0".to_string(),
        ));
    }

    let mut rng = make_rng(seed);
    let t = time_axis(n_samples, fs);
    let rr_interval = 60.0 / heart_rate; // seconds between beats

    // PQRST Gaussian wave parameters: (relative_time, amplitude, width)
    // times are relative to the R-peak.
    let waves: &[(f64, f64, f64)] = &[
        (-0.20, 0.20, 0.020), // P wave
        (-0.05, -0.10, 0.010), // Q deflection
        (0.00, 1.00, 0.008),  // R peak
        (0.05, -0.15, 0.010), // S deflection
        (0.20, 0.35, 0.040),  // T wave
    ];

    let mut signal = vec![0.0_f64; n_samples];

    // R-peak locations: 0, rr, 2*rr, …
    let duration = n_samples as f64 / fs;
    let mut beat_t = rr_interval / 2.0; // offset by half RR so signal starts rising
    while beat_t < duration {
        for &(dt_rel, amp, sigma) in waves {
            let center = beat_t + dt_rel;
            for i in 0..n_samples {
                let ti = t[i];
                let arg = (ti - center) / sigma;
                signal[i] += amp * (-0.5 * arg * arg).exp();
            }
        }
        beat_t += rr_interval;
    }

    // Add Gaussian noise.
    if noise_level > 0.0 {
        let noise_dist = normal_dist(noise_level)?;
        for s in signal.iter_mut() {
            *s += noise_dist.sample(&mut rng);
        }
    }

    Ok((t, Array1::from_vec(signal)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Seismic trace
// ─────────────────────────────────────────────────────────────────────────────

/// Synthesise a seismic trace with random Ricker-wavelet events.
///
/// Each seismic event is modelled as a Ricker (Mexican-hat) wavelet with
/// random arrival time, random polarity, and random amplitude.
/// Background noise is added proportional to the peak amplitude.
///
/// # Arguments
///
/// * `n_samples` – Number of samples.
/// * `fs`        – Sampling frequency in Hz.
/// * `n_events`  – Number of seismic reflection events.
/// * `seed`      – Random seed.
///
/// # Returns
///
/// `(time, signal)` – both `Array1<f64>` of length `n_samples`.
///
/// # Errors
///
/// Returns an error when `n_samples == 0` or `fs ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::seismic_trace;
///
/// let (t, sig) = seismic_trace(1000, 250.0, 5, 42).expect("seismic failed");
/// assert_eq!(sig.len(), 1000);
/// ```
pub fn seismic_trace(
    n_samples: usize,
    fs: f64,
    n_events: usize,
    seed: u64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    check_n_fs("seismic_trace", n_samples, fs)?;

    let mut rng = make_rng(seed);
    let t = time_axis(n_samples, fs);
    let duration = n_samples as f64 / fs;

    // Ricker wavelet dominant frequency: 20 Hz (typical seismic band).
    let f_dom = 20.0_f64;

    let uniform_t = uniform_dist(0.0, duration)?;
    let amp_dist = uniform_dist(0.5, 1.5)?;
    let polarity_dist = uniform_dist(0.0, 1.0)?;
    let noise_dist = normal_dist(0.05)?;

    let mut signal = vec![0.0_f64; n_samples];

    for _ in 0..n_events {
        let t0 = uniform_t.sample(&mut rng);
        let amp = amp_dist.sample(&mut rng);
        // Random polarity.
        let polarity = if polarity_dist.sample(&mut rng) < 0.5 { 1.0_f64 } else { -1.0_f64 };

        for i in 0..n_samples {
            let tau = t[i] - t0;
            // Ricker wavelet: (1 - 2*(π f τ)²) exp(-(π f τ)²)
            let pi_f_tau = PI * f_dom * tau;
            signal[i] += polarity * amp
                * (1.0 - 2.0 * pi_f_tau * pi_f_tau)
                * (-pi_f_tau * pi_f_tau).exp();
        }
    }

    // Background noise.
    let peak = signal
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let noise_scale = peak * 0.05;
    if noise_scale > 0.0 {
        let bg_noise = normal_dist(noise_scale)?;
        for s in signal.iter_mut() {
            *s += bg_noise.sample(&mut rng);
        }
    } else {
        for s in signal.iter_mut() {
            *s += noise_dist.sample(&mut rng);
        }
    }

    Ok((t, Array1::from_vec(signal)))
}

// ─────────────────────────────────────────────────────────────────────────────
// Chirp signal
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a chirp (frequency-sweep) signal.
///
/// The phase is integrated exactly for each method, so the instantaneous
/// frequency tracks the requested sweep law precisely.
///
/// # Arguments
///
/// * `n_samples` – Number of samples.
/// * `fs`        – Sampling frequency in Hz.
/// * `f0`        – Start frequency in Hz (must be > 0).
/// * `f1`        – End frequency in Hz (must be > 0; may be < `f0` for down-sweep).
/// * `method`    – Frequency sweep law.
///
/// # Returns
///
/// `(time, signal)` – both `Array1<f64>` of length `n_samples`.
///
/// # Errors
///
/// Returns an error when `n_samples == 0`, `fs ≤ 0`, or `f0 ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::{chirp_signal, ChirpMethod};
///
/// let (t, sig) = chirp_signal(4096, 1000.0, 10.0, 400.0, ChirpMethod::Linear)
///     .expect("chirp failed");
/// assert_eq!(sig.len(), 4096);
/// ```
pub fn chirp_signal(
    n_samples: usize,
    fs: f64,
    f0: f64,
    f1: f64,
    method: ChirpMethod,
) -> Result<(Array1<f64>, Array1<f64>)> {
    check_n_fs("chirp_signal", n_samples, fs)?;
    if f0 <= 0.0 {
        return Err(DatasetsError::InvalidFormat(
            "chirp_signal: f0 must be > 0".to_string(),
        ));
    }

    let t = time_axis(n_samples, fs);
    let t_end = (n_samples - 1) as f64 / fs;

    let signal: Vec<f64> = (0..n_samples)
        .map(|i| {
            let ti = t[i];
            let phase = match method {
                ChirpMethod::Linear => {
                    // f(t) = f0 + (f1-f0)/T * t
                    // φ(t) = 2π [f0 t + (f1-f0)/(2T) t²]
                    let k = (f1 - f0) / t_end;
                    2.0 * PI * (f0 * ti + 0.5 * k * ti * ti)
                }
                ChirpMethod::Logarithmic => {
                    // f(t) = f0 * (f1/f0)^(t/T)
                    // φ(t) = 2π f0 T / ln(f1/f0) * [(f1/f0)^(t/T) - 1]
                    if (f1 - f0).abs() < 1e-12 {
                        2.0 * PI * f0 * ti
                    } else {
                        let ratio = f1 / f0;
                        let ln_ratio = ratio.ln();
                        2.0 * PI * f0 * t_end / ln_ratio * (ratio.powf(ti / t_end) - 1.0)
                    }
                }
                ChirpMethod::Hyperbolic => {
                    // f(t) = f0 f1 T / (f1 T - (f1-f0) t) for f0≠f1
                    // φ(t) = -2π f0 f1 T / (f1-f0) * ln(1 - (f1-f0)/(f1 T) * t)
                    if (f1 - f0).abs() < 1e-12 {
                        2.0 * PI * f0 * ti
                    } else {
                        let coeff = f0 * f1 * t_end / (f1 - f0);
                        let arg = 1.0 - (f1 - f0) / (f1 * t_end) * ti;
                        let arg_clamped = arg.max(1e-15); // avoid log(0)
                        -2.0 * PI * coeff * arg_clamped.ln()
                    }
                }
            };
            phase.sin()
        })
        .collect();

    Ok((t, Array1::from_vec(signal)))
}

// ─────────────────────────────────────────────────────────────────────────────
// AM signal
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an amplitude-modulated (AM) signal.
///
/// ```text
/// s(t) = [1 + cos(2π f_m t)] · cos(2π f_c t)
/// ```
///
/// # Arguments
///
/// * `carrier_freq`    – Carrier frequency `f_c` in Hz.
/// * `modulation_freq` – Modulation frequency `f_m` in Hz.
/// * `fs`              – Sampling frequency in Hz.
/// * `n_samples`       – Number of samples.
///
/// # Errors
///
/// Returns an error when `n_samples == 0` or `fs ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::am_signal;
///
/// let sig = am_signal(100.0, 5.0, 2000.0, 1000).expect("am failed");
/// assert_eq!(sig.len(), 1000);
/// ```
pub fn am_signal(
    carrier_freq: f64,
    modulation_freq: f64,
    fs: f64,
    n_samples: usize,
) -> Result<Array1<f64>> {
    check_n_fs("am_signal", n_samples, fs)?;

    let sig: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / fs;
            let envelope = 1.0 + (2.0 * PI * modulation_freq * t).cos();
            envelope * (2.0 * PI * carrier_freq * t).cos()
        })
        .collect();

    Ok(Array1::from_vec(sig))
}

// ─────────────────────────────────────────────────────────────────────────────
// FM signal
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a frequency-modulated (FM) signal.
///
/// ```text
/// s(t) = cos(2π f_c t + β sin(2π f_m t))
/// ```
///
/// where `β` is the modulation index.
///
/// # Arguments
///
/// * `carrier_freq`    – Carrier frequency in Hz.
/// * `modulation_freq` – Modulation frequency in Hz.
/// * `beta`            – Modulation index (ratio of frequency deviation to `f_m`).
/// * `fs`              – Sampling frequency in Hz.
/// * `n_samples`       – Number of samples.
///
/// # Errors
///
/// Returns an error when `n_samples == 0` or `fs ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::fm_signal;
///
/// let sig = fm_signal(200.0, 10.0, 2.5, 4000.0, 2000).expect("fm failed");
/// assert_eq!(sig.len(), 2000);
/// ```
pub fn fm_signal(
    carrier_freq: f64,
    modulation_freq: f64,
    beta: f64,
    fs: f64,
    n_samples: usize,
) -> Result<Array1<f64>> {
    check_n_fs("fm_signal", n_samples, fs)?;

    let sig: Vec<f64> = (0..n_samples)
        .map(|i| {
            let t = i as f64 / fs;
            let phase = 2.0 * PI * carrier_freq * t + beta * (2.0 * PI * modulation_freq * t).sin();
            phase.cos()
        })
        .collect();

    Ok(Array1::from_vec(sig))
}

// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal mixture
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a mixture of sinusoids with additive white Gaussian noise.
///
/// ```text
/// s(t) = Σ_k A_k sin(2π f_k t + φ_k) + N(0, σ²)
/// ```
///
/// The noise standard deviation `σ` is computed from the requested SNR:
/// `SNR_dB = 20 log₁₀(A_rms / σ)`.
///
/// # Arguments
///
/// * `frequencies` – Frequencies in Hz (must have same length as `amplitudes` and `phases`).
/// * `amplitudes`  – Amplitudes of each component.
/// * `phases`      – Initial phases in radians.
/// * `noise_snr_db` – Signal-to-noise ratio in dB (positive = less noise).
/// * `fs`          – Sampling frequency in Hz.
/// * `n_samples`   – Number of samples.
/// * `seed`        – Random seed.
///
/// # Errors
///
/// Returns an error when slice lengths mismatch, `n_samples == 0`, or `fs ≤ 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_datasets::synthetic_signals::sinusoidal_mixture;
///
/// let sig = sinusoidal_mixture(
///     &[50.0, 120.0, 300.0],
///     &[1.0, 0.5, 0.3],
///     &[0.0, 0.5, 1.0],
///     30.0,
///     2000.0,
///     4096,
///     0,
/// ).expect("mixture failed");
/// assert_eq!(sig.len(), 4096);
/// ```
pub fn sinusoidal_mixture(
    frequencies: &[f64],
    amplitudes: &[f64],
    phases: &[f64],
    noise_snr_db: f64,
    fs: f64,
    n_samples: usize,
    seed: u64,
) -> Result<Array1<f64>> {
    check_n_fs("sinusoidal_mixture", n_samples, fs)?;
    if frequencies.len() != amplitudes.len() || amplitudes.len() != phases.len() {
        return Err(DatasetsError::InvalidFormat(
            "sinusoidal_mixture: frequencies, amplitudes, and phases must have the same length"
                .to_string(),
        ));
    }
    if frequencies.is_empty() {
        return Err(DatasetsError::InvalidFormat(
            "sinusoidal_mixture: at least one frequency component is required".to_string(),
        ));
    }

    // Build clean signal.
    let mut signal = vec![0.0_f64; n_samples];
    for i in 0..n_samples {
        let t = i as f64 / fs;
        for k in 0..frequencies.len() {
            signal[i] += amplitudes[k] * (2.0 * PI * frequencies[k] * t + phases[k]).sin();
        }
    }

    // Compute RMS amplitude of the clean signal.
    let rms = {
        let sum_sq: f64 = signal.iter().map(|v| v * v).sum();
        (sum_sq / n_samples as f64).sqrt()
    };

    // Noise std from SNR: SNR = 20 log10(rms / sigma) → sigma = rms / 10^(SNR/20).
    let sigma = if rms > 1e-15 {
        rms / 10.0_f64.powf(noise_snr_db / 20.0)
    } else {
        1e-6
    };

    let mut rng = make_rng(seed);
    let noise_dist = normal_dist(sigma)?;
    for s in signal.iter_mut() {
        *s += noise_dist.sample(&mut rng);
    }

    Ok(Array1::from_vec(signal))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── ECG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_ecg_shape() {
        let (t, sig) = ecg_signal(2000, 500.0, 70.0, 0.02, 42).expect("ecg failed");
        assert_eq!(t.len(), 2000);
        assert_eq!(sig.len(), 2000);
    }

    #[test]
    fn test_ecg_time_axis() {
        let fs = 250.0;
        let n = 500;
        let (t, _) = ecg_signal(n, fs, 60.0, 0.0, 0).expect("ecg failed");
        assert!((t[0] - 0.0).abs() < 1e-12);
        assert!((t[n - 1] - (n - 1) as f64 / fs).abs() < 1e-10);
    }

    #[test]
    fn test_ecg_periodicity() {
        // Heart rate = 60 bpm → 1 beat/s.  With fs=500, period = 500 samples.
        // The R-peak should appear every ~500 samples.
        let fs = 500.0;
        let n = 3000;
        let hr = 60.0;
        let (_, sig) = ecg_signal(n, fs, hr, 0.0, 42).expect("ecg failed");

        // Find the two largest local maxima.
        let mut peaks = vec![];
        for i in 1..n - 1 {
            if sig[i] > sig[i - 1] && sig[i] > sig[i + 1] && sig[i] > 0.5 {
                peaks.push(i);
            }
        }
        assert!(peaks.len() >= 2, "Expected at least 2 R-peaks, got {}", peaks.len());
        let interval = (peaks[1] - peaks[0]) as f64;
        let expected = fs * 60.0 / hr;
        let rel = (interval - expected).abs() / expected;
        assert!(rel < 0.05, "Peak interval {interval} vs expected {expected}");
    }

    #[test]
    fn test_ecg_error_zero_samples() {
        assert!(ecg_signal(0, 500.0, 70.0, 0.0, 0).is_err());
    }

    #[test]
    fn test_ecg_error_negative_noise() {
        assert!(ecg_signal(100, 500.0, 70.0, -0.1, 0).is_err());
    }

    // ── Seismic ─────────────────────────────────────────────────────────────

    #[test]
    fn test_seismic_shape() {
        let (t, sig) = seismic_trace(1024, 200.0, 4, 0).expect("seismic failed");
        assert_eq!(t.len(), 1024);
        assert_eq!(sig.len(), 1024);
    }

    #[test]
    fn test_seismic_non_zero() {
        // With events the signal should be non-trivial.
        let (_, sig) = seismic_trace(500, 100.0, 3, 7).expect("seismic failed");
        let max_abs = sig.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(max_abs > 0.0, "seismic signal is all zeros");
    }

    // ── Chirp ────────────────────────────────────────────────────────────────

    #[test]
    fn test_chirp_shape() {
        let (t, sig) = chirp_signal(4096, 1000.0, 10.0, 400.0, ChirpMethod::Linear)
            .expect("chirp failed");
        assert_eq!(t.len(), 4096);
        assert_eq!(sig.len(), 4096);
    }

    #[test]
    fn test_chirp_unit_amplitude() {
        // All methods should produce |s(t)| ≤ 1.
        for method in [ChirpMethod::Linear, ChirpMethod::Logarithmic, ChirpMethod::Hyperbolic] {
            let (_, sig) =
                chirp_signal(2048, 2000.0, 10.0, 800.0, method).expect("chirp failed");
            let max_abs = sig.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
            assert!(max_abs <= 1.0 + 1e-10, "method={method:?}, max={max_abs}");
        }
    }

    #[test]
    fn test_chirp_linear_frequency_sweep() {
        // For a linear chirp from f0 to f1, the instantaneous frequency should
        // increase.  Verify by measuring the zero-crossing interval at the start
        // vs the end of the signal.
        let fs = 8000.0_f64;
        let n = 8000_usize;
        let f0 = 100.0_f64;
        let f1 = 1000.0_f64;
        let (_, sig) = chirp_signal(n, fs, f0, f1, ChirpMethod::Linear).expect("chirp failed");

        // Measure period at start (first 20 zero-crossings) and at end.
        let crossing_interval = |start: usize, count: usize| -> Option<f64> {
            let mut crossings = vec![];
            let mut i = start + 1;
            while i < n && crossings.len() < count {
                if sig[i - 1] < 0.0 && sig[i] >= 0.0 {
                    crossings.push(i);
                }
                i += 1;
            }
            if crossings.len() < 2 {
                return None;
            }
            let intervals: Vec<f64> = crossings.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            Some(mean)
        };

        let period_start = crossing_interval(0, 10).expect("not enough crossings at start");
        let period_end =
            crossing_interval(n - n / 4, 5).expect("not enough crossings at end");
        // Frequency at start ≈ f0, at end ≈ f1, so period should be smaller at end.
        assert!(
            period_end < period_start,
            "Expected period_end ({period_end}) < period_start ({period_start})"
        );
    }

    #[test]
    fn test_chirp_error_zero_f0() {
        assert!(chirp_signal(100, 1000.0, 0.0, 400.0, ChirpMethod::Linear).is_err());
    }

    // ── AM ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_am_shape() {
        let sig = am_signal(100.0, 5.0, 2000.0, 1000).expect("am failed");
        assert_eq!(sig.len(), 1000);
    }

    #[test]
    fn test_am_amplitude_bounds() {
        // Envelope 1 + cos ∈ [0, 2], so |s| ≤ 2.
        let sig = am_signal(100.0, 5.0, 2000.0, 2000).expect("am failed");
        let max_abs = sig.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(max_abs <= 2.0 + 1e-10, "AM amplitude > 2: {max_abs}");
    }

    // ── FM ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_fm_shape() {
        let sig = fm_signal(200.0, 10.0, 2.5, 4000.0, 2000).expect("fm failed");
        assert_eq!(sig.len(), 2000);
    }

    #[test]
    fn test_fm_unit_amplitude() {
        // FM: |s(t)| = |cos(...)| ≤ 1.
        let sig = fm_signal(200.0, 10.0, 5.0, 4000.0, 4000).expect("fm failed");
        let max_abs = sig.iter().copied().fold(0.0_f64, |a, v| a.max(v.abs()));
        assert!(max_abs <= 1.0 + 1e-10, "FM amplitude > 1: {max_abs}");
    }

    // ── Sinusoidal mixture ───────────────────────────────────────────────────

    #[test]
    fn test_mixture_shape() {
        let sig = sinusoidal_mixture(
            &[50.0, 120.0],
            &[1.0, 0.5],
            &[0.0, 0.0],
            30.0,
            2000.0,
            4096,
            99,
        )
        .expect("mixture failed");
        assert_eq!(sig.len(), 4096);
    }

    #[test]
    fn test_mixture_snr_improves_noise() {
        // High SNR should yield lower RMS noise than low SNR.
        let make = |snr: f64| {
            sinusoidal_mixture(&[100.0], &[1.0], &[0.0], snr, 4000.0, 8000, 1)
                .expect("mixture failed")
        };
        let high = make(40.0);
        let low = make(5.0);
        // Variance of the high-SNR signal should be closer to 0.5 (pure sine variance).
        let rms = |s: &Array1<f64>| {
            (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt()
        };
        let rms_high = rms(&high);
        let rms_low = rms(&low);
        // Both should be positive and the high-SNR one should be closer to √0.5 ≈ 0.707.
        let pure_rms = (0.5_f64).sqrt();
        let diff_high = (rms_high - pure_rms).abs();
        let diff_low = (rms_low - pure_rms).abs();
        assert!(
            diff_high <= diff_low,
            "High-SNR signal deviates more from ideal: high={diff_high}, low={diff_low}"
        );
    }

    #[test]
    fn test_mixture_error_mismatched_slices() {
        assert!(sinusoidal_mixture(&[100.0, 200.0], &[1.0], &[0.0], 30.0, 2000.0, 1024, 0).is_err());
    }

    #[test]
    fn test_mixture_error_empty_frequencies() {
        assert!(sinusoidal_mixture(&[], &[], &[], 30.0, 2000.0, 1024, 0).is_err());
    }

    // ── Determinism ──────────────────────────────────────────────────────────

    #[test]
    fn test_reproducibility() {
        let a = ecg_signal(500, 250.0, 60.0, 0.1, 77).expect("ecg failed");
        let b = ecg_signal(500, 250.0, 60.0, 0.1, 77).expect("ecg failed");
        assert_eq!(a.1, b.1, "ECG should be reproducible");

        let c = sinusoidal_mixture(&[100.0], &[1.0], &[0.0], 20.0, 2000.0, 1000, 5)
            .expect("mixture failed");
        let d = sinusoidal_mixture(&[100.0], &[1.0], &[0.0], 20.0, 2000.0, 1000, 5)
            .expect("mixture failed");
        assert_eq!(c, d, "Mixture should be reproducible");
    }
}
