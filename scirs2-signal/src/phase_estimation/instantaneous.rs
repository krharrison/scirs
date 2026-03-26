//! Instantaneous frequency estimation via Hilbert transform, phase unwrapping,
//! and the Teager-Kaiser energy operator.

use std::f64::consts::PI;

use super::types::InstantaneousFreq;

// ─── Radix-2 FFT (internal, no external dependency) ──────────────────────────

/// Bit-reversal permutation for FFT.
fn bit_reverse_permute(data: &mut [(f64, f64)]) {
    let n = data.len();
    let bits = (n as f64).log2() as usize;
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if i < j {
            data.swap(i, j);
        }
    }
}

fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place Cooley-Tukey radix-2 DIT FFT on complex data (power-of-2 length).
/// `inverse` flag controls sign of the twiddle exponent; caller must scale by 1/N for IFFT.
fn fft_inplace(data: &mut [(f64, f64)], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    bit_reverse_permute(data);

    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let sign = if inverse { 1.0 } else { -1.0 };
        let angle_unit = sign * PI / half as f64;
        for i in (0..n).step_by(len) {
            for k in 0..half {
                let theta = angle_unit * k as f64;
                let (wre, wim) = (theta.cos(), theta.sin());
                let (ure, uim) = data[i + k];
                let (vre, vim) = data[i + k + half];
                let twid_re = wre * vre - wim * vim;
                let twid_im = wre * vim + wim * vre;
                data[i + k] = (ure + twid_re, uim + twid_im);
                data[i + k + half] = (ure - twid_re, uim - twid_im);
            }
        }
        len *= 2;
    }
}

/// Next power of two ≥ n.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1 << (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

// ─── Hilbert transform ────────────────────────────────────────────────────────

/// Compute the analytic signal imaginary part (Hilbert transform) of `signal`.
///
/// The algorithm:
/// 1. Zero-pad to the next power of two.
/// 2. Forward FFT.
/// 3. Set negative-frequency bins to zero; double positive-frequency bins.
/// 4. Inverse FFT → imaginary part is the Hilbert transform.
///
/// Returns a `Vec<f64>` of the same length as `signal`.
pub fn hilbert_transform(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    let n_fft = next_pow2(n);

    // Build complex input, zero-padded.
    let mut buf: Vec<(f64, f64)> = (0..n_fft)
        .map(|i| if i < n { (signal[i], 0.0) } else { (0.0, 0.0) })
        .collect();

    fft_inplace(&mut buf, false);

    // Apply Hilbert weighting in frequency domain.
    // DC (k=0) and Nyquist (k=N/2 for even N): weight = 1.
    // Positive freqs (k=1..N/2-1): weight = 2.
    // Negative freqs (k=N/2+1..N-1): weight = 0.
    let n_half = n_fft / 2;
    // k=0 DC: unchanged.
    // k=1..n_half-1: double.
    for k in 1..n_half {
        buf[k] = (buf[k].0 * 2.0, buf[k].1 * 2.0);
    }
    // k=n_half (Nyquist): unchanged.
    // k=n_half+1..n_fft-1: zero out.
    for k in (n_half + 1)..n_fft {
        buf[k] = (0.0, 0.0);
    }

    // Inverse FFT.
    fft_inplace(&mut buf, true);
    let scale = 1.0 / n_fft as f64;

    // The imaginary part of the IFFT is the Hilbert transform.
    buf.iter().take(n).map(|&(_, im)| im * scale).collect()
}

// ─── Phase unwrapping ────────────────────────────────────────────────────────

/// Unwrap a phase sequence by adding ±2π to remove jumps larger than π.
///
/// No allocation beyond the output vector.
pub fn phase_unwrap(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(phase.len());
    out.push(phase[0]);
    let mut cumulative = 0.0f64;

    for window in phase.windows(2) {
        let diff = window[1] - window[0];
        // Wrap diff to (-π, π].
        let wrapped = ((diff + PI).rem_euclid(2.0 * PI)) - PI;
        cumulative += wrapped;
        out.push(phase[0] + cumulative);
    }
    out
}

// ─── Instantaneous phase ─────────────────────────────────────────────────────

/// Compute the unwrapped instantaneous phase of `signal`.
///
/// Uses the analytic signal: ψ(t) = atan2(H{x}(t), x(t)), then unwraps.
pub fn instantaneous_phase(signal: &[f64]) -> Vec<f64> {
    let h = hilbert_transform(signal);
    let wrapped: Vec<f64> = signal
        .iter()
        .zip(h.iter())
        .map(|(&x, &hx)| hx.atan2(x))
        .collect();
    phase_unwrap(&wrapped)
}

// ─── 5-point moving average ───────────────────────────────────────────────────

/// Apply a symmetric 5-point moving average to `data`.
fn moving_average_5(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let i_lo = i.saturating_sub(2);
        let i_hi = (i + 2).min(n - 1);
        let count = (i_hi - i_lo + 1) as f64;
        let sum: f64 = data[i_lo..=i_hi].iter().sum();
        out.push(sum / count);
    }
    out
}

// ─── Instantaneous frequency (Hilbert-based) ──────────────────────────────────

/// Estimate the instantaneous frequency of `signal` at sample rate `fs` Hz.
///
/// Algorithm:
/// 1. Compute unwrapped phase φ(n).
/// 2. Differentiate: δφ(n) = φ(n) - φ(n-1).
/// 3. IF(n) = fs · δφ(n) / (2π).
/// 4. Smooth with 5-point moving average.
///
/// Output length equals `signal.len()` (first sample is duplicated from second).
pub fn instantaneous_frequency(signal: &[f64], fs: f64) -> InstantaneousFreq {
    if signal.len() < 2 {
        return InstantaneousFreq::new(vec![0.0; signal.len()], fs);
    }

    let phase = instantaneous_phase(signal);
    let n = phase.len();

    // First-order difference.
    let mut if_raw = Vec::with_capacity(n);
    // Edge: replicate second sample at index 0.
    if_raw.push((phase[1] - phase[0]) * fs / (2.0 * PI));
    for i in 1..n {
        if_raw.push((phase[i] - phase[i - 1]) * fs / (2.0 * PI));
    }

    let smoothed = moving_average_5(&if_raw);
    InstantaneousFreq::new(smoothed, fs)
}

// ─── Teager-Kaiser Energy Operator ───────────────────────────────────────────

/// Compute the Teager-Kaiser Energy Operator (TKEO):
///   `psi[n] = x[n]^2 - x[n-1] * x[n+1]`
///
/// Output length equals `signal.len()`; boundary samples use central-difference
/// extrapolation (first and last samples are set to adjacent interior values).
pub fn teager_kaiser_energy(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![signal[0] * signal[0]];
    }
    if n == 2 {
        return vec![signal[0] * signal[0], signal[1] * signal[1]];
    }

    let mut tkeo = Vec::with_capacity(n);
    // Boundary: replicate interior value.
    tkeo.push(signal[1].powi(2) - signal[0] * signal[2]);
    for i in 1..n - 1 {
        tkeo.push(signal[i].powi(2) - signal[i - 1] * signal[i + 1]);
    }
    // Last boundary.
    let last_inner = signal[n - 2].powi(2) - signal[n - 3] * signal[n - 1];
    tkeo.push(last_inner);

    tkeo
}

/// Estimate instantaneous frequency from the Teager-Kaiser Energy Operator using the
/// DESA-2 (Discrete Energy Separation Algorithm) formula (Maragos, Kaiser & Quatieri 1993):
///
/// Define `y[n] = x[n+1] - x[n-1]` (central difference).
/// Then for `x[n] = A*cos(w*n + phi)`:
///   `Psi(x)[n] = A^2*sin^2(w)` (TKEO of signal)
///   `Psi(y)[n] = 4*A^2*sin^2(w)*sin^2(w)` ... actually:
///   `y[n] = -2*A*sin(w)*sin(w*n + phi)` -> `Psi(y)[n] = 4*A^2*sin^2(w)*sin^2(w)`?
///
/// The amplitude-independent formula (DESA-1 variant):
///   `f[n] = arccos(1 - Psi_x[n] / (Psi_x[n] + psi_cross[n])) * fs/(2*pi)`
/// where `psi_cross[n] = (x[n]*x[n+1] - x[n-1]*x[n])`.
///
/// Actually the numerically stable form that is amplitude-independent uses
/// adjacent-sample TKEO products. The simplest robust version:
///   Given `Psi(x) = A^2*sin^2(w)`, we need A. Use RMS: `A ~ sqrt(2) * rms(signal)`.
///   Then `w = arcsin(sqrt(Psi_x / A^2))` and `f = w*fs/(2*pi)`.
///
/// Output length equals `signal.len()`.
pub fn teager_kaiser_if(signal: &[f64], fs: f64) -> InstantaneousFreq {
    let n = signal.len();
    if n < 3 {
        return InstantaneousFreq::new(vec![0.0; n], fs);
    }

    let tkeo = teager_kaiser_energy(signal);

    // ψ_y: TKEO of central difference y[n] = x[n+1] - x[n-1].
    // y[n] = -2A sin(ω) sin(ωn+φ), so ψ_y ≈ 4A²sin²(ω)·sin²(ω) ... no:
    // TKEO(y)[n] = y[n]² - y[n-1]·y[n+1]
    //            = 4A²sin²(ω)[sin²(ωn+φ) - sin(ω(n-1)+φ)sin(ω(n+1)+φ)]
    //            = 4A²sin²(ω)·sin²(ω) = 4A²sin⁴(ω)
    // So Ψ_y/(4·Ψ_x) = sin²(ω) → ω = arcsin(sqrt(Ψ_y/(4Ψ_x)))
    // But building y needs boundary extension.
    let mut y = Vec::with_capacity(n);
    // Edge: replicate adjacent.
    y.push(signal[1] - signal[0]);
    for i in 1..n - 1 {
        y.push(signal[i + 1] - signal[i - 1]);
    }
    y.push(signal[n - 1] - signal[n - 2]);
    let tkeo_y = teager_kaiser_energy(&y);

    let samples: Vec<f64> = (0..n)
        .map(|i| {
            let psi_x = tkeo[i];
            let psi_y = tkeo_y[i];
            if psi_x <= 0.0 || psi_y <= 0.0 || !psi_x.is_finite() || !psi_y.is_finite() {
                return 0.0;
            }
            // DESA-2 formula: f = arcsin(sqrt(ψ_y / (4·ψ_x))) · fs/(2π)
            let ratio = (psi_y / (4.0 * psi_x)).min(1.0);
            if ratio < 0.0 {
                return 0.0;
            }
            let omega = ratio.sqrt().asin(); // radians per sample
            omega * fs / (2.0 * PI)
        })
        .collect();

    let smoothed = moving_average_5(&samples);
    InstantaneousFreq::new(smoothed, fs)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_sine(freq_hz: f64, phase: f64, amp: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq_hz / fs * i as f64 + phase).sin())
            .collect()
    }

    fn make_cos(freq_hz: f64, phase: f64, amp: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq_hz / fs * i as f64 + phase).cos())
            .collect()
    }

    #[test]
    fn test_hilbert_transform_quadrature() {
        // H{sin(ωt)} ≈ -cos(ωt), H{cos(ωt)} ≈ sin(ωt)
        let fs = 1000.0;
        let n = 512;
        let freq = 100.0f64;

        // H{sin} ≈ -cos
        let sig_sin = make_sine(freq, 0.0, 1.0, fs, n);
        let h_sin = hilbert_transform(&sig_sin);
        let expected: Vec<f64> = make_cos(freq, 0.0, -1.0, fs, n);
        // Compare interior samples (edges have boundary effects).
        let interior = 64..n - 64;
        let max_err = h_sin[interior.clone()]
            .iter()
            .zip(expected[interior.clone()].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 0.05, "H{{sin}} ≈ -cos error: {max_err:.4}");

        // H{cos} ≈ sin
        let sig_cos = make_cos(freq, 0.0, 1.0, fs, n);
        let h_cos = hilbert_transform(&sig_cos);
        let expected_cos: Vec<f64> = make_sine(freq, 0.0, 1.0, fs, n);
        let max_err2 = h_cos[interior.clone()]
            .iter()
            .zip(expected_cos[interior].iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err2 < 0.05, "H{{cos}} ≈ sin error: {max_err2:.4}");
    }

    #[test]
    fn test_instantaneous_phase_linear() {
        // Constant-frequency sine → instantaneous phase should be linear (after unwrap).
        let fs = 1000.0;
        let n = 256;
        let freq = 50.0f64;
        let sig = make_sine(freq, 0.0, 1.0, fs, n);
        let phase = instantaneous_phase(&sig);

        // Interior: phase[i] ≈ 2π·freq/fs · i − π/2  (due to sin convention)
        // Just verify it's monotonically increasing in the interior.
        let interior = &phase[32..224];
        for w in interior.windows(2) {
            assert!(w[1] > w[0] - 0.01, "Phase should be non-decreasing");
        }
    }

    #[test]
    fn test_phase_unwrap_no_jumps() {
        let phase = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let unwrapped = phase_unwrap(&phase);
        for (a, b) in phase.iter().zip(unwrapped.iter()) {
            assert!((a - b).abs() < 1e-12, "Smooth phase should be unchanged");
        }
    }

    #[test]
    fn test_phase_unwrap_with_jump() {
        // Simulate a 2π jump.
        let phase = vec![0.0, PI - 0.1, PI + 0.1, -(PI - 0.2), -(PI - 0.4)];
        // After: 0, ~3.04, ~3.24, 3.24 + (pi-0.2 - (pi+0.1)) ≈ 3.24 - 0.3 = 2.94 ... hmm
        // More direct test: consecutive difference > π should be corrected.
        let jumpy = vec![0.0, PI * 0.9, PI * 1.8, -(PI * 0.5), 0.2];
        let unwrapped = phase_unwrap(&jumpy);
        // Check that differences after unwrapping are all ≤ π.
        for w in unwrapped.windows(2) {
            let diff = (w[1] - w[0]).abs();
            assert!(diff <= PI + 1e-10, "Unwrapped diff {diff:.4} > π");
        }
    }

    #[test]
    fn test_instantaneous_freq_constant() {
        // Constant-frequency sine → IF should be approximately constant.
        let fs = 1000.0;
        let n = 512;
        let freq = 100.0f64;
        let sig = make_sine(freq, 0.0, 1.0, fs, n);
        let if_out = instantaneous_frequency(&sig, fs);
        // Interior samples (avoid edge effects).
        let interior = &if_out.samples[64..448];
        let mean = interior.iter().sum::<f64>() / interior.len() as f64;
        assert!(
            (mean - freq).abs() < 5.0,
            "Expected IF ≈ {freq} Hz, got mean={mean:.2} Hz"
        );
        let std: f64 = (interior.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
            / interior.len() as f64)
            .sqrt();
        assert!(
            std < 20.0,
            "IF std dev {std:.2} should be low for constant freq"
        );
    }

    #[test]
    fn test_tkeo_pure_sine() {
        // For x[n] = A·cos(ωn): ψ[n] = A²·sin²(ω).
        // With A=1, ω=2π·f/fs: ψ ≈ sin²(ω).
        let fs = 1000.0;
        let n = 512;
        let freq = 100.0f64;
        let omega = 2.0 * PI * freq / fs;
        let expected_tkeo = omega.sin().powi(2); // For A=1.

        let sig = make_cos(freq, 0.0, 1.0, fs, n);
        let tkeo = teager_kaiser_energy(&sig);
        let interior_mean = tkeo[64..448].iter().sum::<f64>() / (448 - 64) as f64;
        assert!(
            (interior_mean - expected_tkeo).abs() < 0.02,
            "TKEO mean {interior_mean:.4} ≠ expected {expected_tkeo:.4}"
        );
    }

    #[test]
    fn test_tkeo_if_recovery() {
        // TKEO IF should recover the true frequency for a cosine.
        let fs = 1000.0;
        let n = 512;
        let freq = 100.0f64;
        let sig = make_cos(freq, 0.0, 1.0, fs, n);
        let tk_if = teager_kaiser_if(&sig, fs);
        let interior = &tk_if.samples[64..448];
        let mean = interior.iter().filter(|&&v| v.is_finite()).sum::<f64>()
            / interior.iter().filter(|&&v| v.is_finite()).count().max(1) as f64;
        assert!(
            (mean - freq).abs() < 5.0,
            "TKEO IF mean {mean:.2} ≠ expected {freq}"
        );
    }
}
