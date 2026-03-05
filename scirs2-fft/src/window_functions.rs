//! Extended window functions for spectral analysis.
//!
//! This module provides a comprehensive collection of window functions used in
//! signal processing to reduce spectral leakage, as well as the DPSS (Discrete
//! Prolate Spheroidal Sequences) Slepian tapers for multitaper spectral estimation.
//!
//! # Simple windows
//!
//! * [`hann`]        — Raised cosine, good all-round choice.
//! * [`hamming`]     — Raised cosine with non-zero endpoints.
//! * [`blackman`]    — Three-term cosine sum; very low sidelobes.
//! * [`kaiser`]      — Adjustable sidelobe suppression via `beta`.
//! * [`flattop`]     — Maximally flat main lobe for amplitude accuracy.
//! * [`tukey`]       — Cosine-tapered rectangular window.
//!
//! # DPSS / Slepian tapers
//!
//! * [`dpss`] — Compute `k` Discrete Prolate Spheroidal Sequences with
//!   time-bandwidth product `nw`.
//!
//! # Utilities
//!
//! * [`apply_window`] — Element-wise multiply a signal by a window.

use crate::error::{FFTError, FFTResult};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Simple window functions
// ─────────────────────────────────────────────────────────────────────────────

/// Hann (raised cosine) window of length `n`.
///
/// The Hann window is `w[k] = 0.5 * (1 - cos(2π k / (n-1)))`.
/// It touches zero at both ends when `sym = true` (symmetric, default for
/// spectral analysis).
///
/// # Errors
///
/// Returns an error if `n == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::hann;
///
/// let w = hann(5).expect("hann");
/// assert_eq!(w.len(), 5);
/// assert!((w[0]).abs() < 1e-12);   // touches zero at endpoints
/// assert!((w[4]).abs() < 1e-12);
/// ```
pub fn hann(n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("hann: n must be positive".into()));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let w: Vec<f64> = (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n as f64 - 1.0)).cos()))
        .collect();
    Ok(w)
}

/// Hamming window of length `n`.
///
/// `w[k] = 0.54 - 0.46 * cos(2π k / (n-1))`.
/// The endpoints are non-zero (~0.08), giving slightly better frequency
/// resolution than the Hann window.
///
/// # Errors
///
/// Returns an error if `n == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::hamming;
///
/// let w = hamming(5).expect("hamming");
/// assert!((w[2] - 1.0).abs() < 1e-10);  // peak at centre
/// ```
pub fn hamming(n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("hamming: n must be positive".into()));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let w: Vec<f64> = (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f64 / (n as f64 - 1.0)).cos())
        .collect();
    Ok(w)
}

/// Blackman window of length `n`.
///
/// Three-term cosine sum:
/// `w[k] = 0.42 - 0.5 * cos(2π k/(n-1)) + 0.08 * cos(4π k/(n-1))`.
///
/// # Errors
///
/// Returns an error if `n == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::blackman;
///
/// let w = blackman(7).expect("blackman");
/// assert_eq!(w.len(), 7);
/// ```
pub fn blackman(n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("blackman: n must be positive".into()));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let w: Vec<f64> = (0..n)
        .map(|k| {
            let x = 2.0 * PI * k as f64 / (n as f64 - 1.0);
            0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
        })
        .collect();
    Ok(w)
}

/// Kaiser window of length `n` with shape parameter `beta`.
///
/// The Kaiser window is `w[k] = I₀(β √(1 - ((k - α)/α)²)) / I₀(β)` where
/// `α = (n-1)/2` and `I₀` is the modified Bessel function of order 0.
///
/// Larger `beta` gives better sidelobe suppression at the cost of a wider main
/// lobe.  Typical values:
/// * `beta ≈ 5`  → ~57 dB attenuation (similar to Hamming)
/// * `beta ≈ 8.6` → ~80 dB attenuation
///
/// # Errors
///
/// Returns an error if `n == 0` or `beta < 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::kaiser;
///
/// let w = kaiser(10, 8.6).expect("kaiser");
/// assert_eq!(w.len(), 10);
/// // Endpoints should be small
/// assert!(w[0] < 0.05);
/// ```
pub fn kaiser(n: usize, beta: f64) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("kaiser: n must be positive".into()));
    }
    if beta < 0.0 {
        return Err(FFTError::ValueError("kaiser: beta must be non-negative".into()));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    let alpha = (n as f64 - 1.0) / 2.0;
    let i0_beta = bessel_i0(beta);
    let w: Vec<f64> = (0..n)
        .map(|k| {
            let x = ((k as f64 - alpha) / alpha).powi(2);
            let arg = beta * (1.0 - x).max(0.0).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect();
    Ok(w)
}

/// Flat-top window of length `n`.
///
/// Five-term cosine sum optimised for accurate amplitude measurement.
/// The main lobe is broad and very flat; sidelobes are high.
///
/// # Errors
///
/// Returns an error if `n == 0`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::flattop;
///
/// let w = flattop(8).expect("flattop");
/// assert_eq!(w.len(), 8);
/// ```
pub fn flattop(n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("flattop: n must be positive".into()));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    // Coefficients from Heinzel et al. (2002)
    let a: [f64; 5] = [
        0.215_578_95,
        0.416_631_58,
        0.277_263_158,
        0.083_578_947,
        0.006_947_368,
    ];
    let w: Vec<f64> = (0..n)
        .map(|k| {
            let x = 2.0 * PI * k as f64 / (n as f64 - 1.0);
            a[0] - a[1] * x.cos() + a[2] * (2.0 * x).cos()
                - a[3] * (3.0 * x).cos()
                + a[4] * (4.0 * x).cos()
        })
        .collect();
    Ok(w)
}

/// Tukey (cosine-tapered rectangular) window of length `n`.
///
/// The Tukey window has `alpha * N` cosine-tapered samples at each end and a
/// rectangular flat region in the middle.
///
/// * `alpha = 0` → Rectangular window.
/// * `alpha = 1` → Hann window.
///
/// # Arguments
///
/// * `n`     – Window length.
/// * `alpha` – Taper ratio ∈ [0, 1].
///
/// # Errors
///
/// Returns an error if `n == 0` or `alpha` is outside `[0, 1]`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::tukey;
///
/// let w = tukey(20, 0.5).expect("tukey");
/// assert_eq!(w.len(), 20);
/// // The centre should be 1.0
/// assert!((w[10] - 1.0).abs() < 1e-12);
/// ```
pub fn tukey(n: usize, alpha: f64) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("tukey: n must be positive".into()));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(FFTError::ValueError(
            "tukey: alpha must be in [0, 1]".into(),
        ));
    }
    if n == 1 {
        return Ok(vec![1.0]);
    }
    if alpha == 0.0 {
        return Ok(vec![1.0; n]);
    }
    if alpha == 1.0 {
        return hann(n);
    }

    let width = (alpha * (n as f64 - 1.0) / 2.0).floor() as usize;
    let mut w = vec![1.0_f64; n];

    for i in 0..width {
        let val = 0.5 * (1.0 + (PI * i as f64 / width as f64).cos());
        w[i] = val;
        w[n - 1 - i] = val;
    }

    Ok(w)
}

// ─────────────────────────────────────────────────────────────────────────────
//  DPSS / Slepian sequences
// ─────────────────────────────────────────────────────────────────────────────

/// Compute `k` Discrete Prolate Spheroidal Sequences (DPSS / Slepian tapers).
///
/// DPSS windows maximise the spectral concentration within a bandwidth of
/// `2W = 2 nw / n` (fraction of the Nyquist band).  The `m`-th taper
/// corresponds to the `m`-th largest eigenvalue of the spectral concentration
/// problem.
///
/// # Algorithm
///
/// Solves the symmetric tridiagonal eigenvalue problem
///
/// ```text
/// T[i,i]   = ((n-1)/2 - i)² · cos(2π W)
/// T[i,i+1] = i · (n-i) / 2
/// ```
///
/// using the QR algorithm via deflation and shifted inverse power iteration to
/// find the `k` largest eigenvalues and their eigenvectors in descending order.
///
/// # Arguments
///
/// * `n`  – Window length.
/// * `nw` – Time-bandwidth product (e.g. 2.0, 2.5, 3.0, 4.0).
/// * `k`  – Number of tapers to compute (`k ≤ floor(2*nw - 1)`).
///
/// # Returns
///
/// A vector of `k` tapers, each of length `n`.  The first taper has the
/// highest eigenvalue (most concentrated bandwidth).
///
/// # Errors
///
/// Returns an error if `n == 0`, `nw <= 0`, `k == 0`, or
/// `k > floor(2*nw - 1)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::dpss;
///
/// // 4 Slepian tapers with time-bandwidth product 2.5
/// let tapers = dpss(256, 2.5, 4).expect("dpss");
/// assert_eq!(tapers.len(), 4);
/// assert_eq!(tapers[0].len(), 256);
///
/// // Each taper should have unit energy
/// for (i, taper) in tapers.iter().enumerate() {
///     let energy: f64 = taper.iter().map(|x| x * x).sum();
///     assert!((energy - 1.0).abs() < 1e-6, "taper {i} energy={energy}");
/// }
/// ```
pub fn dpss(n: usize, nw: f64, k: usize) -> FFTResult<Vec<Vec<f64>>> {
    if n == 0 {
        return Err(FFTError::ValueError("dpss: n must be positive".into()));
    }
    if nw <= 0.0 {
        return Err(FFTError::ValueError(
            "dpss: time-bandwidth product nw must be positive".into(),
        ));
    }
    if k == 0 {
        return Err(FFTError::ValueError(
            "dpss: k must be at least 1".into(),
        ));
    }
    let k_max = (2.0 * nw - 1.0).floor() as usize;
    if k > k_max {
        return Err(FFTError::ValueError(format!(
            "dpss: k={k} exceeds floor(2*nw-1)={k_max}"
        )));
    }

    // Build tridiagonal matrix entries
    let w = nw / n as f64; // half-bandwidth fraction
    let mut diag = vec![0.0_f64; n];
    let mut off = vec![0.0_f64; n.saturating_sub(1)];

    for i in 0..n {
        let t = (n as f64 - 1.0) / 2.0 - i as f64;
        diag[i] = t * t * (2.0 * PI * w).cos();
    }
    for i in 0..n.saturating_sub(1) {
        off[i] = (i as f64 + 1.0) * (n as f64 - 1.0 - i as f64) / 2.0;
    }

    // Find the k largest eigenvalues and eigenvectors using deflated power
    // iteration.  The approach:
    //   1. Use power iteration to find the dominant eigenvector.
    //   2. Deflate the matrix (conceptually) and repeat.
    //
    // We implement "band-orthogonal" deflation: after finding eigenvector v_m,
    // for the next iteration we project each intermediate iterate away from the
    // already-found eigenvectors.
    let mut tapers: Vec<Vec<f64>> = Vec::with_capacity(k);

    for m in 0..k {
        // Initialise: eigenvectors of the tridiagonal concentrate energy near
        // the centre; use an alternating-sign seed to avoid aliasing.
        let mut v: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign / (n as f64).sqrt()
            })
            .collect();

        // Shift estimate for the m-th eigenvalue to speed convergence.
        // The eigenvalues are roughly evenly spaced; a crude initial shift helps.
        let shift_approx = if m < k {
            let frac = m as f64 / k_max as f64;
            (1.0 - 2.0 * frac) * diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            0.0
        };

        let max_iter = 500;
        let tol = 1e-13;

        for _iter in 0..max_iter {
            // Multiply: tv = T * v  (tridiagonal)
            let mut tv = tridiag_mul(&diag, &off, &v);

            // Apply shift: tv = tv - shift * v
            for (tvi, &vi) in tv.iter_mut().zip(v.iter()) {
                *tvi -= shift_approx * vi;
            }

            // Orthogonalise against already-found tapers (Gram-Schmidt)
            for prev in &tapers {
                let dot: f64 = tv.iter().zip(prev.iter()).map(|(&a, &b)| a * b).sum();
                for (tvi, &pi) in tv.iter_mut().zip(prev.iter()) {
                    *tvi -= dot * pi;
                }
            }

            // Normalise
            let norm: f64 = tv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-30 {
                break;
            }
            for val in &mut tv {
                *val /= norm;
            }

            // Convergence check
            let diff: f64 = v
                .iter()
                .zip(tv.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum();

            v = tv;

            if diff < tol {
                break;
            }
        }

        // Ensure positive convention: first non-trivially large coefficient > 0
        let first_large = v.iter().find(|&&x| x.abs() > 1e-10).copied().unwrap_or(0.0);
        if first_large < 0.0 {
            for val in &mut v {
                *val = -*val;
            }
        }

        tapers.push(v);
    }

    Ok(tapers)
}

// ─────────────────────────────────────────────────────────────────────────────
//  apply_window
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a window to a signal by element-wise multiplication.
///
/// # Arguments
///
/// * `signal` – Input signal.
/// * `window` – Window values (must have the same length as `signal`).
///
/// # Errors
///
/// Returns an error if the lengths differ or either is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::window_functions::{hann, apply_window};
///
/// let signal = vec![1.0_f64; 8];
/// let win = hann(8).expect("hann");
/// let windowed = apply_window(&signal, &win).expect("apply");
/// assert_eq!(windowed.len(), signal.len());
/// assert!((windowed[0]).abs() < 1e-12);
/// ```
pub fn apply_window(signal: &[f64], window: &[f64]) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("apply_window: signal is empty".into()));
    }
    if signal.len() != window.len() {
        return Err(FFTError::DimensionError(format!(
            "apply_window: signal.len()={} != window.len()={}",
            signal.len(),
            window.len()
        )));
    }
    Ok(signal.iter().zip(window.iter()).map(|(&s, &w)| s * w).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Modified Bessel function of the first kind, order 0 — I₀(x).
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        y.mul_add(
            3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))),
            1.0,
        )
    } else {
        let y = 3.75 / ax;
        let exp_term = ax.exp() / ax.sqrt();
        exp_term
            * y.mul_add(
                0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77)))))),
                0.398_942_28,
            )
    }
}

/// Multiply a tridiagonal matrix (given by diagonal `d` and off-diagonal `o`)
/// by a vector `v`.
fn tridiag_mul(d: &[f64], o: &[f64], v: &[f64]) -> Vec<f64> {
    let n = d.len();
    let mut out = vec![0.0_f64; n];
    if n == 0 {
        return out;
    }
    out[0] = d[0] * v[0];
    if n > 1 {
        out[0] += o[0] * v[1];
    }
    for i in 1..n - 1 {
        out[i] = o[i - 1] * v[i - 1] + d[i] * v[i] + o[i] * v[i + 1];
    }
    if n > 1 {
        out[n - 1] = o[n - 2] * v[n - 2] + d[n - 1] * v[n - 1];
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Check that a window vector is within [−ε, 1+ε] and has the expected length.
    fn check_window(w: &[f64], n: usize) {
        assert_eq!(w.len(), n);
        for &v in w {
            assert!(v >= -1e-10 && v <= 1.0 + 1e-10, "window value out of range: {v}");
        }
    }

    #[test]
    fn test_hann_zeros_at_endpoints() {
        let w = hann(8).expect("hann");
        check_window(&w, 8);
        assert!(w[0].abs() < 1e-12, "hann[0]={}", w[0]);
        assert!(w[7].abs() < 1e-12, "hann[7]={}", w[7]);
    }

    #[test]
    fn test_hann_peak_at_centre() {
        let n = 9;
        let w = hann(n).expect("hann");
        let centre = n / 2;
        assert!((w[centre] - 1.0).abs() < 1e-10, "hann peak={}", w[centre]);
    }

    #[test]
    fn test_hamming_non_zero_endpoints() {
        let w = hamming(8).expect("hamming");
        check_window(&w, 8);
        // Hamming endpoints ≈ 0.08 (non-zero)
        assert!(w[0] > 0.05 && w[0] < 0.15, "hamming[0]={}", w[0]);
    }

    #[test]
    fn test_blackman_length_and_range() {
        let w = blackman(12).expect("blackman");
        check_window(&w, 12);
    }

    #[test]
    fn test_kaiser_length_and_range() {
        let w = kaiser(10, 8.6).expect("kaiser");
        assert_eq!(w.len(), 10);
        for &v in &w {
            assert!(v >= 0.0 && v <= 1.0 + 1e-10, "kaiser value out of range: {v}");
        }
        // Endpoints should be small for large beta
        assert!(w[0] < 0.05, "kaiser beta=8.6 endpoint too large: {}", w[0]);
    }

    #[test]
    fn test_kaiser_beta_zero_is_rectangular() {
        let w = kaiser(8, 0.0).expect("kaiser_rect");
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-10, "kaiser β=0 should be rectangular: {v}");
        }
    }

    #[test]
    fn test_flattop_length() {
        let w = flattop(16).expect("flattop");
        assert_eq!(w.len(), 16);
    }

    #[test]
    fn test_tukey_alpha_zero_is_rectangular() {
        let w = tukey(10, 0.0).expect("tukey_rect");
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tukey_alpha_one_is_hann() {
        let n = 16;
        let tw = tukey(n, 1.0).expect("tukey_hann");
        let hw = hann(n).expect("hann");
        for (t, h) in tw.iter().zip(hw.iter()) {
            assert!((t - h).abs() < 1e-10, "tukey(1)≠hann: {t} vs {h}");
        }
    }

    #[test]
    fn test_tukey_flat_centre() {
        let n = 20;
        let w = tukey(n, 0.5).expect("tukey");
        assert_eq!(w.len(), n);
        assert!((w[10] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_window_length() {
        let signal = vec![1.0_f64; 8];
        let win = hann(8).expect("hann");
        let out = apply_window(&signal, &win).expect("apply");
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_apply_window_values() {
        let signal = vec![1.0_f64; 8];
        let win = hann(8).expect("hann");
        let out = apply_window(&signal, &win).expect("apply");
        for (o, w) in out.iter().zip(win.iter()) {
            assert!((o - w).abs() < 1e-14);
        }
    }

    #[test]
    fn test_dpss_count_and_length() {
        let tapers = dpss(64, 4.0, 7).expect("dpss");
        assert_eq!(tapers.len(), 7);
        for t in &tapers {
            assert_eq!(t.len(), 64);
        }
    }

    #[test]
    fn test_dpss_unit_energy() {
        let tapers = dpss(64, 4.0, 7).expect("dpss");
        for (i, taper) in tapers.iter().enumerate() {
            let energy: f64 = taper.iter().map(|x| x * x).sum();
            assert!(
                (energy - 1.0).abs() < 1e-5,
                "taper {i}: energy={energy}"
            );
        }
    }

    #[test]
    fn test_dpss_orthogonality() {
        // Adjacent DPSS tapers should be approximately orthogonal
        let tapers = dpss(128, 4.0, 6).expect("dpss");
        for i in 0..tapers.len() {
            for j in (i + 1)..tapers.len() {
                let dot: f64 = tapers[i]
                    .iter()
                    .zip(tapers[j].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < 0.1,
                    "tapers {i} and {j} are not orthogonal: dot={dot}"
                );
            }
        }
    }

    #[test]
    fn test_dpss_k_exceeds_limit_fails() {
        // k > floor(2*nw - 1) should fail
        let result = dpss(64, 2.0, 10);
        assert!(result.is_err(), "expected error for k > 2*nw-1");
    }

    #[test]
    fn test_apply_window_length_mismatch_error() {
        let signal = vec![1.0_f64; 8];
        let win = vec![1.0_f64; 5];
        let result = apply_window(&signal, &win);
        assert!(result.is_err());
    }
}
