//! Radar Ambiguity Function.
//!
//! The ambiguity function χ(τ, ν) characterises the response of a matched
//! filter to a time-delayed (by τ samples) and Doppler-shifted (by ν) echo of
//! the transmitted waveform.  It is defined as:
//!
//! ```text
//! χ(τ, ν) = Σ_{n} s[n + τ] · s*[n] · exp(j 2π ν n / N)
//! ```
//!
//! for a discrete signal `s[n]` of length N.
//!
//! # References
//! - A. W. Rihaczek, *Principles of High-Resolution Radar*, McGraw-Hill, 1969.
//! - M. A. Richards, J. A. Scheer, W. A. Holm, *Principles of Modern Radar*,
//!   SciTech Publishing, 2010.

use std::f64::consts::PI;

use scirs2_core::numeric::Complex64;

use crate::error::{FFTError, FFTResult};

// ── Simple in-place radix-2 FFT (pure Rust, no external dependency) ──────────

/// In-place radix-2 Cooley-Tukey FFT.  `n` must be a power of two.
fn fft_radix2(data: &mut [Complex64]) -> FFTResult<()> {
    let n = data.len();
    if n == 0 {
        return Ok(());
    }
    if n & (n - 1) != 0 {
        return Err(FFTError::ValueError(format!(
            "ambiguity FFT requires power-of-2 length, got {n}"
        )));
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
            data.swap(i, j);
        }
    }
    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wlen = Complex64::new(angle.cos(), angle.sin());
        for i in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = data[i + k];
                let v = data[i + k + half] * w;
                data[i + k] = u + v;
                data[i + k + half] = u - v;
                w *= wlen;
            }
        }
        len <<= 1;
    }
    Ok(())
}

/// Next power of two ≥ `n`.
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

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for [`compute_ambiguity_function`].
#[derive(Debug, Clone)]
pub struct AmbiguityConfig {
    /// Normalise the output so that the peak |χ(0,0)|² = 1.  Default: `true`.
    pub normalize: bool,
    /// Number of Doppler bins.  If 0, the signal length N is used.  Default: 0.
    pub n_doppler_bins: usize,
}

impl Default for AmbiguityConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            n_doppler_bins: 0,
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

/// Result of an ambiguity function computation.
#[derive(Debug, Clone)]
pub struct AmbiguityResult {
    /// |χ(delay, Doppler)|²,  shape `[n_delay_bins][n_doppler_bins]`.
    pub data: Vec<Vec<f64>>,
    /// Delay axis values in samples: −(N−1), ..., −1, 0, 1, ..., N−1.
    pub delay_bins: Vec<f64>,
    /// Normalised Doppler frequency axis: 0, 1/N_d, 2/N_d, ..., (N_d−1)/N_d.
    pub doppler_bins: Vec<f64>,
    /// Row index in `data` of the maximum value.
    pub peak_delay: usize,
    /// Column index in `data` of the maximum value.
    pub peak_doppler: usize,
    /// Maximum value of |χ|² in `data`.
    pub peak_magnitude: f64,
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute the full discrete ambiguity function for a complex signal.
///
/// For each delay lag k ∈ [−(N−1), N−1], the cross-correlation vector
/// `z[n] = s[n + k] · s*[n]` (zero-padded at boundaries) is formed, and then
/// its DFT is taken over `n_doppler_bins` frequency bins.
///
/// # Arguments
/// * `signal` — Input complex signal of length N ≥ 2.
/// * `config`  — Configuration (normalise flag, number of Doppler bins).
///
/// # Returns
/// An [`AmbiguityResult`] with `data[d][f]` = |chi(delay\_bins\[d\], doppler\_bins\[f\])|^2.
pub fn compute_ambiguity_function(
    signal: &[Complex64],
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let n = signal.len();
    if n < 2 {
        return Err(FFTError::ValueError(format!(
            "signal length must be ≥ 2, got {n}"
        )));
    }

    let nd_raw = if config.n_doppler_bins == 0 {
        n
    } else {
        config.n_doppler_bins
    };
    // Pad Doppler dimension to power of two for FFT
    let nd_fft = next_pow2(nd_raw.max(n));

    // Delay range: k = -(N-1) .. (N-1), total 2N-1 bins
    let n_delay = 2 * n - 1;
    let mut data: Vec<Vec<f64>> = Vec::with_capacity(n_delay);

    for d in 0..n_delay {
        let k: i64 = d as i64 - (n as i64 - 1); // lag value

        // Build cross-correlation vector z[n] = s[n+k] · s*[n]
        let mut z: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nd_fft];
        for nn in 0..n {
            let m = nn as i64 + k;
            if m >= 0 && m < n as i64 {
                let s_m = signal[m as usize];
                let s_n_conj = Complex64::new(signal[nn].re, -signal[nn].im);
                z[nn] = s_m * s_n_conj;
            }
        }

        // DFT along Doppler dimension
        fft_radix2(&mut z)?;

        // Store |·|² for the requested number of Doppler bins
        let row: Vec<f64> = (0..nd_raw)
            .map(|f| {
                let idx = f % nd_fft; // in case nd_raw was not padded
                z[idx].re * z[idx].re + z[idx].im * z[idx].im
            })
            .collect();
        data.push(row);
    }

    // Find peak
    let (mut peak_delay, mut peak_doppler, mut peak_magnitude) = (0usize, 0usize, 0.0_f64);
    for (di, row) in data.iter().enumerate() {
        for (fi, &val) in row.iter().enumerate() {
            if val > peak_magnitude {
                peak_magnitude = val;
                peak_delay = di;
                peak_doppler = fi;
            }
        }
    }

    // Normalise
    if config.normalize && peak_magnitude > 1e-30 {
        let inv = 1.0 / peak_magnitude;
        for row in data.iter_mut() {
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        peak_magnitude = 1.0;
    }

    let delay_bins: Vec<f64> = (0..n_delay).map(|d| d as f64 - (n as f64 - 1.0)).collect();
    let doppler_bins: Vec<f64> = (0..nd_raw).map(|f| f as f64 / nd_raw as f64).collect();

    Ok(AmbiguityResult {
        data,
        delay_bins,
        doppler_bins,
        peak_delay,
        peak_doppler,
        peak_magnitude,
    })
}

/// Compute a *narrow* (restricted delay range) ambiguity function.
///
/// Same as [`compute_ambiguity_function`] but only evaluates
/// `n_delay_bins` delay lags centred at zero: k ∈ [−H, H] where
/// H = (n_delay_bins − 1) / 2.
pub fn narrow_ambiguity_function(
    signal: &[Complex64],
    n_delay_bins: usize,
    config: &AmbiguityConfig,
) -> FFTResult<AmbiguityResult> {
    let n = signal.len();
    if n < 2 {
        return Err(FFTError::ValueError(format!(
            "signal length must be ≥ 2, got {n}"
        )));
    }
    let half_d = (n_delay_bins / 2) as i64;
    let nd_raw = if config.n_doppler_bins == 0 {
        n
    } else {
        config.n_doppler_bins
    };
    let nd_fft = next_pow2(nd_raw.max(n));

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(n_delay_bins);
    let mut delay_bins_vec: Vec<f64> = Vec::with_capacity(n_delay_bins);

    for d in 0..n_delay_bins {
        let k: i64 = d as i64 - half_d;
        delay_bins_vec.push(k as f64);

        let mut z: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); nd_fft];
        for nn in 0..n {
            let m = nn as i64 + k;
            if m >= 0 && m < n as i64 {
                let s_m = signal[m as usize];
                let s_n_conj = Complex64::new(signal[nn].re, -signal[nn].im);
                z[nn] = s_m * s_n_conj;
            }
        }
        fft_radix2(&mut z)?;

        let row: Vec<f64> = (0..nd_raw)
            .map(|f| {
                let idx = f % nd_fft;
                z[idx].re * z[idx].re + z[idx].im * z[idx].im
            })
            .collect();
        data.push(row);
    }

    let (mut peak_delay, mut peak_doppler, mut peak_magnitude) = (0usize, 0usize, 0.0_f64);
    for (di, row) in data.iter().enumerate() {
        for (fi, &val) in row.iter().enumerate() {
            if val > peak_magnitude {
                peak_magnitude = val;
                peak_delay = di;
                peak_doppler = fi;
            }
        }
    }

    if config.normalize && peak_magnitude > 1e-30 {
        let inv = 1.0 / peak_magnitude;
        for row in data.iter_mut() {
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        peak_magnitude = 1.0;
    }

    let doppler_bins: Vec<f64> = (0..nd_raw).map(|f| f as f64 / nd_raw as f64).collect();

    Ok(AmbiguityResult {
        data,
        delay_bins: delay_bins_vec,
        doppler_bins,
        peak_delay,
        peak_doppler,
        peak_magnitude,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_chirp(n: usize, rate: f64) -> Vec<Complex64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let phase = PI * rate * t * t;
                Complex64::new(phase.cos(), phase.sin())
            })
            .collect()
    }

    #[test]
    fn test_config_default() {
        let cfg = AmbiguityConfig::default();
        assert!(cfg.normalize);
        assert_eq!(cfg.n_doppler_bins, 0);
    }

    #[test]
    fn test_result_shape() {
        let n = 16;
        let signal: Vec<Complex64> = (0..n).map(|_| Complex64::new(1.0, 0.0)).collect();
        let cfg = AmbiguityConfig::default();
        let result = compute_ambiguity_function(&signal, &cfg).expect("ambiguity ok");
        // Delay dimension: 2N - 1 = 31
        assert_eq!(result.data.len(), 2 * n - 1);
        // Doppler dimension: N (default)
        assert_eq!(result.data[0].len(), n);
        assert_eq!(result.delay_bins.len(), 2 * n - 1);
        assert_eq!(result.doppler_bins.len(), n);
    }

    #[test]
    fn test_peak_at_origin_for_constant_signal() {
        // A constant (DC) signal should have maximum at delay=0, Doppler=0
        let n = 16;
        let signal: Vec<Complex64> = (0..n).map(|_| Complex64::new(1.0, 0.0)).collect();
        let cfg = AmbiguityConfig::default();
        let result = compute_ambiguity_function(&signal, &cfg).expect("ambiguity ok");
        // Zero delay is at index N-1
        let zero_delay_idx = n - 1;
        assert_eq!(
            result.peak_delay, zero_delay_idx,
            "peak delay bin should be {zero_delay_idx} (zero lag)"
        );
        assert_eq!(result.peak_doppler, 0);
    }

    #[test]
    fn test_peak_magnitude_normalised() {
        let n = 8;
        let signal = make_chirp(n, 4.0);
        let cfg = AmbiguityConfig {
            normalize: true,
            ..Default::default()
        };
        let result = compute_ambiguity_function(&signal, &cfg).expect("ambiguity ok");
        // Normalised peak must be 1.0
        assert!(
            (result.peak_magnitude - 1.0).abs() < 1e-9,
            "peak={}",
            result.peak_magnitude
        );
    }

    #[test]
    fn test_symmetry_property() {
        // |χ(τ, ν)|² should equal |χ(-τ, -ν)|² (modular doppler)
        // For a real signal the ambiguity function satisfies |χ(τ,ν)| = |χ(-τ,-ν)|
        let n = 8;
        let signal: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((2.0 * PI * i as f64 / n as f64).cos(), 0.0))
            .collect();
        let cfg = AmbiguityConfig {
            normalize: false,
            ..Default::default()
        };
        let result = compute_ambiguity_function(&signal, &cfg).expect("ambiguity ok");
        // delay index d and (2N-2 - d) correspond to +k and -k
        let nd_d = result.doppler_bins.len();
        for d in 0..(2 * n - 1) {
            let d_mirror = (2 * n - 2) - d;
            for f in 0..nd_d {
                let f_mirror = (nd_d - f) % nd_d;
                let v1 = result.data[d][f];
                let v2 = result.data[d_mirror][f_mirror];
                assert!(
                    (v1 - v2).abs() < 1e-6,
                    "symmetry failed at d={d} f={f}: {v1} vs {v2}"
                );
            }
        }
    }

    #[test]
    fn test_narrow_ambiguity_shape() {
        let n = 16;
        let signal: Vec<Complex64> = (0..n).map(|_| Complex64::new(1.0, 0.0)).collect();
        let cfg = AmbiguityConfig::default();
        let n_delay = 5;
        let result = narrow_ambiguity_function(&signal, n_delay, &cfg).expect("narrow ok");
        assert_eq!(result.data.len(), n_delay);
        assert_eq!(result.data[0].len(), n);
    }

    #[test]
    fn test_short_signal_error() {
        let signal = vec![Complex64::new(1.0, 0.0)]; // length 1
        let cfg = AmbiguityConfig::default();
        assert!(compute_ambiguity_function(&signal, &cfg).is_err());
    }
}
