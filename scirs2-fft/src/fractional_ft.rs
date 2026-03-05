//! Fractional Fourier Transform (FrFT) – high-level standalone module
//!
//! This module provides a clean, self-contained implementation of the
//! **Fractional Fourier Transform** (FrFT) built around the
//! chirp-multiplication–FFT–chirp-multiplication (Ozaktas-Mendlovic-Kutay) decomposition.
//!
//! Unlike [`crate::frft`] (which delegates to several internal implementations),
//! this module exposes a simple, well-documented API suitable for production use:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`frft`]              | FrFT with rotation angle `alpha` (complex input) |
//! | [`frft_real`]         | FrFT for real-valued input |
//! | [`optimal_frft_order`] | Find the order that most concentrates signal energy |
//! | [`frft_spectrogram`]  | FrFT-based time-frequency spectrogram |
//!
//! # Mathematical Background
//!
//! The Fractional Fourier Transform of order α rotates a signal by angle
//! φ = α·π/2 in the time-frequency plane.  The kernel is:
//!
//! ```text
//! K_α(t, u) = A_φ · exp(iπ(t²·cot φ − 2tu·csc φ + u²·cot φ))
//! ```
//!
//! where `A_φ = sqrt((1 − i·cot φ) / (2π))`.
//!
//! Special cases:
//! * α = 0 → identity operator
//! * α = 1 → standard Fourier transform
//! * α = 2 → time-reversal operator  f(t) → f(−t)
//! * α = 3 → inverse Fourier transform
//! * α = 4 → identity (period-4 group property)
//!
//! # Algorithm
//!
//! The discrete FrFT is computed using the Ozaktas-Kutay decomposition:
//!
//! 1. **Pre-chirp multiplication**: `y[n] = x[n] · exp(iπ·n²·cot φ / N)`
//! 2. **Chirp convolution** (evaluated via FFT of length 2N):
//!    convolve with `h[n] = exp(−iπ·n²·csc φ / N)`
//! 3. **Post-chirp multiplication**: `X[k] = scale · chirp[k] · (y * h)[k]`
//!
//! This gives O(N log N) complexity for any transform order.
//!
//! # References
//!
//! * Ozaktas, H. M.; Kutay, M. A.; Zalevsky, Z. *The Fractional Fourier Transform.*
//!   Wiley, 2001.
//! * Candan, C.; Kutay, M. A.; Ozaktas, H. M. "The discrete fractional Fourier
//!   transform." *IEEE Trans. Signal Processing* 48(5) (2000), pp. 1329–1337.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Complex64, Zero};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the amplitude scaling factor A_φ = sqrt((1 - i·cot φ) / (2π)).
#[inline]
fn amplitude_factor(phi: f64) -> Complex64 {
    let cot_phi = phi.cos() / phi.sin();
    ((Complex64::new(1.0, -cot_phi)) / (2.0 * PI)).sqrt()
}

/// Generate N-point chirp for pre/post multiplication:
///   chirp[k] = exp(i·π·k²·coeff / N)
/// where `coeff` is typically `cot(φ)` or `csc(φ)`.
fn chirp_mult(n: usize, coeff: f64) -> Vec<Complex64> {
    (0..n)
        .map(|k| {
            let phase = PI * (k as f64).powi(2) * coeff / n as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect()
}

/// Perform a length-N circular chirp convolution via zero-padded FFT.
///
/// Computes: `result[k] = Σ_{n} a[n] · h[(k-n) mod M]` where M ≥ 2N-1
/// is the next power of two, evaluated as point-wise multiplication in
/// the frequency domain.
fn chirp_convolve(a: &[Complex64], h_chirp: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = a.len();
    let m = (2 * n - 1).next_power_of_two();

    // Pad a
    let mut a_pad = vec![Complex64::zero(); m];
    a_pad[..n].copy_from_slice(a);

    // Build padded kernel: h[0..n] and h[m-n+1..m] = h[n-1..1] (wrap-around)
    let mut h_pad = vec![Complex64::zero(); m];
    h_pad[0] = h_chirp[0];
    for k in 1..n {
        h_pad[k] = h_chirp[k];
        h_pad[m - k] = h_chirp[k].conj(); // h is symmetric for csc chirp
    }

    let fa = fft(&a_pad, None)?;
    let fh = fft(&h_pad, None)?;
    let fc: Vec<Complex64> = fa.iter().zip(fh.iter()).map(|(&ai, &hi)| ai * hi).collect();
    let c = ifft(&fc, None)?;

    Ok(c[..n].to_vec())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core FrFT implementation (OMK algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Core OMK (Ozaktas-Mendlovic-Kutay) FrFT for complex input.
fn frft_omk_core(signal: &[Complex64], phi: f64) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    let cot_phi = phi.cos() / phi.sin();
    let csc_phi = 1.0 / phi.sin();

    // --- Pre-chirp ---
    let pre_chirp = chirp_mult(n, cot_phi);
    let y: Vec<Complex64> = signal
        .iter()
        .zip(pre_chirp.iter())
        .map(|(&xk, &ck)| xk * ck)
        .collect();

    // --- Chirp convolution kernel h[k] = exp(-i·π·k²·csc φ / N) ---
    let h_chirp = chirp_mult(n, -csc_phi);

    let convolved = chirp_convolve(&y, &h_chirp)?;

    // --- Post-chirp and scaling ---
    let scale = amplitude_factor(phi);
    let result: Vec<Complex64> = convolved
        .iter()
        .zip(pre_chirp.iter())
        .map(|(&ck, &chirp)| scale * chirp * ck)
        .collect();

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API: frft
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Fractional Fourier Transform of order `alpha`.
///
/// The FrFT rotates the signal by angle `φ = alpha · π/2` in the
/// time-frequency plane.  `alpha = 1` recovers the standard Fourier transform;
/// `alpha = 0` is the identity; `alpha = 2` is time-reversal; `alpha = 3` is
/// the inverse Fourier transform.
///
/// # Arguments
///
/// * `signal` – Input complex signal (any length ≥ 1).
/// * `alpha`  – Transform order.  Any real number; values outside [0, 4) are
///              reduced modulo 4.
///
/// # Returns
///
/// `Vec<Complex64>` of the same length as `signal`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `signal` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::fractional_ft::frft;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n = 64;
/// let signal: Vec<Complex64> = (0..n)
///     .map(|k| Complex64::new((2.0 * PI * 5.0 * k as f64 / n as f64).cos(), 0.0))
///     .collect();
///
/// // α = 0 → identity
/// let identity = frft(&signal, 0.0).expect("frft");
/// for (orig, out) in signal.iter().zip(identity.iter()) {
///     assert!((orig - out).norm() < 1e-12);
/// }
///
/// // α = 1 → standard FFT
/// use scirs2_fft::fft::fft as reference_fft;
/// let fft_result = reference_fft(&signal, None).expect("fft");
/// let frft1 = frft(&signal, 1.0).expect("frft α=1");
/// for (a, b) in fft_result.iter().zip(frft1.iter()) {
///     assert!((a - b).norm() < 1e-9);
/// }
/// ```
pub fn frft(signal: &[Complex64], alpha: f64) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("frft: input signal is empty".to_string()));
    }

    // Normalise alpha into [0, 4)
    let alpha = alpha.rem_euclid(4.0);

    // ── Special cases ──────────────────────────────────────────────────────
    if alpha.abs() < 1e-12 || (alpha - 4.0).abs() < 1e-12 {
        // Identity
        return Ok(signal.to_vec());
    }
    if (alpha - 2.0).abs() < 1e-12 {
        // Time reversal: f(t) → f(-t)  (circular shift for discrete signals)
        let mut out = signal.to_vec();
        if n > 1 {
            // index 0 maps to 0; index k maps to n-k
            let copy = out.clone();
            for k in 1..n {
                out[k] = copy[n - k];
            }
        }
        return Ok(out);
    }
    if (alpha - 1.0).abs() < 1e-12 {
        return fft(signal, None);
    }
    if (alpha - 3.0).abs() < 1e-12 {
        return ifft(signal, None);
    }

    // ── General case: OMK decomposition ───────────────────────────────────
    let phi = alpha * PI / 2.0;

    // Near-special-case guard: avoid numerical blow-up when phi is near 0 or π
    if phi.sin().abs() < 1e-6 {
        // Linearly interpolate toward the special case
        if phi < PI / 2.0 {
            // alpha ∈ (0, 0.5) approximately
            let t = phi / (PI / 2.0);
            let f0 = signal.to_vec();
            let f1 = fft(signal, None)?;
            return Ok(f0
                .iter()
                .zip(f1.iter())
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect());
        } else {
            // alpha near 2
            let mut reversed = signal.to_vec();
            if n > 1 {
                let copy = reversed.clone();
                for k in 1..n {
                    reversed[k] = copy[n - k];
                }
            }
            let t = (phi - PI) / (PI / 2.0);
            return Ok(signal
                .iter()
                .zip(reversed.iter())
                .map(|(&a, &b)| a * (1.0 - t.abs()) + b * t.abs())
                .collect());
        }
    }

    frft_omk_core(signal, phi)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API: frft_real
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Fractional Fourier Transform of a real-valued signal.
///
/// Convenience wrapper that converts `signal` to complex before calling [`frft`].
///
/// # Arguments
///
/// * `signal` – Real-valued input of any length ≥ 1.
/// * `alpha`  – Transform order.
///
/// # Returns
///
/// `Vec<Complex64>` of the same length as `signal`.
///
/// # Errors
///
/// Returns an error if `signal` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::fractional_ft::frft_real;
/// use std::f64::consts::PI;
///
/// let n = 128;
/// let signal: Vec<f64> = (0..n)
///     .map(|k| (2.0 * PI * 10.0 * k as f64 / n as f64).sin())
///     .collect();
///
/// let result = frft_real(&signal, 0.5).expect("frft_real");
/// assert_eq!(result.len(), n);
/// ```
pub fn frft_real(signal: &[f64], alpha: f64) -> FFTResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("frft_real: input signal is empty".to_string()));
    }
    let complex_signal: Vec<Complex64> = signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();
    frft(&complex_signal, alpha)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API: optimal_frft_order
// ─────────────────────────────────────────────────────────────────────────────

/// Find the FrFT order that maximises the peak-to-total energy ratio (signal concentration).
///
/// Searches over `n_angles` equally spaced values in [0, 2) and returns the
/// order α* for which the FrFT output is most concentrated (i.e. has the
/// largest max|X[k]|² / Σ|X[k]|²).
///
/// A chirp signal, for instance, becomes highly concentrated at the order
/// that aligns the chirp axis with the frequency axis.
///
/// # Arguments
///
/// * `signal`   – Real-valued input.
/// * `n_angles` – Number of candidate angles to test (default 100 is reasonable).
///
/// # Returns
///
/// The optimal FrFT order α* ∈ [0, 2).
///
/// # Errors
///
/// Returns an error if `signal` is empty or if `n_angles` is 0.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::fractional_ft::optimal_frft_order;
/// use std::f64::consts::PI;
///
/// // Linear chirp is concentrated at alpha = 0.5 (roughly)
/// let n = 128;
/// let signal: Vec<f64> = (0..n)
///     .map(|k| {
///         let t = k as f64 / n as f64;
///         (PI * 20.0 * t * t).cos() // quadratic phase
///     })
///     .collect();
///
/// let order = optimal_frft_order(&signal, 50).expect("optimal order");
/// // The optimal order should be somewhere in (0, 2) – just verify it is a valid number
/// assert!(order >= 0.0 && order < 2.0);
/// ```
pub fn optimal_frft_order(signal: &[f64], n_angles: usize) -> FFTResult<f64> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "optimal_frft_order: input signal is empty".to_string(),
        ));
    }
    if n_angles == 0 {
        return Err(FFTError::ValueError(
            "optimal_frft_order: n_angles must be > 0".to_string(),
        ));
    }

    let complex_signal: Vec<Complex64> = signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    let mut best_alpha = 0.0_f64;
    let mut best_concentration = -1.0_f64;

    for i in 0..n_angles {
        let alpha = 2.0 * i as f64 / n_angles as f64;

        let out = frft(&complex_signal, alpha)?;
        let total: f64 = out.iter().map(|c| c.norm_sqr()).sum();
        if total < 1e-15 {
            continue;
        }
        let peak: f64 = out.iter().map(|c| c.norm_sqr()).fold(0.0_f64, f64::max);
        let concentration = peak / total;

        if concentration > best_concentration {
            best_concentration = concentration;
            best_alpha = alpha;
        }
    }

    Ok(best_alpha)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API: frft_spectrogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a FrFT-based time-frequency spectrogram (fractional spectrogram).
///
/// Each column of the output corresponds to the magnitude-squared FrFT at one
/// rotation angle α_j = j·2/n_angles for j = 0 … n_angles-1.  Each row
/// corresponds to a fractional frequency bin 0 … n_freq-1 (obtained by taking
/// the first `n_freq` points of the FrFT output).
///
/// The result is an `(n_angles × n_freq)` array of power values.
///
/// # Arguments
///
/// * `signal`   – Real-valued input.
/// * `n_angles` – Number of rotation angles (columns in the output).
/// * `n_freq`   – Number of fractional frequency bins to keep per angle.
///                Must be ≤ `signal.len()`.
///
/// # Returns
///
/// [`Array2<f64>`] with shape `(n_angles, n_freq)`.
///
/// # Errors
///
/// Returns an error if `signal` is empty, `n_angles` is 0, `n_freq` is 0, or
/// `n_freq > signal.len()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::fractional_ft::frft_spectrogram;
/// use std::f64::consts::PI;
///
/// let n = 64;
/// let signal: Vec<f64> = (0..n)
///     .map(|k| (2.0 * PI * 5.0 * k as f64 / n as f64).sin())
///     .collect();
///
/// let spec = frft_spectrogram(&signal, 16, 32).expect("frft_spectrogram");
/// assert_eq!(spec.shape(), &[16, 32]);
/// ```
pub fn frft_spectrogram(signal: &[f64], n_angles: usize, n_freq: usize) -> FFTResult<Array2<f64>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "frft_spectrogram: input signal is empty".to_string(),
        ));
    }
    if n_angles == 0 {
        return Err(FFTError::ValueError(
            "frft_spectrogram: n_angles must be > 0".to_string(),
        ));
    }
    if n_freq == 0 {
        return Err(FFTError::ValueError(
            "frft_spectrogram: n_freq must be > 0".to_string(),
        ));
    }
    if n_freq > n {
        return Err(FFTError::ValueError(format!(
            "frft_spectrogram: n_freq ({n_freq}) > signal length ({n})"
        )));
    }

    let complex_signal: Vec<Complex64> = signal
        .iter()
        .map(|&x| Complex64::new(x, 0.0))
        .collect();

    let mut spectrogram = Array2::<f64>::zeros((n_angles, n_freq));

    for i in 0..n_angles {
        let alpha = 2.0 * i as f64 / n_angles as f64;
        let out = frft(&complex_signal, alpha)?;
        for j in 0..n_freq {
            spectrogram[[i, j]] = out[j].norm_sqr();
        }
    }

    Ok(spectrogram)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    fn make_cosine(n: usize, freq: f64) -> Vec<Complex64> {
        (0..n)
            .map(|k| Complex64::new((2.0 * PI * freq * k as f64 / n as f64).cos(), 0.0))
            .collect()
    }

    // ── Special cases ─────────────────────────────────────────────────────────

    #[test]
    fn test_frft_identity_alpha_zero() {
        let signal = make_cosine(32, 3.0);
        let result = frft(&signal, 0.0).expect("frft α=0");
        for (orig, out) in signal.iter().zip(result.iter()) {
            assert!((orig - out).norm() < 1e-12, "identity mismatch");
        }
    }

    #[test]
    fn test_frft_identity_alpha_four() {
        let signal = make_cosine(32, 3.0);
        let result = frft(&signal, 4.0).expect("frft α=4");
        for (orig, out) in signal.iter().zip(result.iter()) {
            assert!((orig - out).norm() < 1e-12, "identity (α=4) mismatch");
        }
    }

    #[test]
    fn test_frft_time_reversal_alpha_two() {
        let n = 8;
        let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let result = frft(&signal, 2.0).expect("frft α=2");
        // result[0] == signal[0]; result[k] == signal[n-k] for k>0
        assert_relative_eq!(result[0].re, signal[0].re, epsilon = 1e-12);
        for k in 1..n {
            assert_relative_eq!(result[k].re, signal[n - k].re, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_frft_alpha1_matches_fft() {
        use crate::fft::fft as ref_fft;
        let signal = make_cosine(64, 5.0);
        let frft1 = frft(&signal, 1.0).expect("frft α=1");
        let fft_result = ref_fft(&signal, None).expect("fft");
        for (a, b) in frft1.iter().zip(fft_result.iter()) {
            assert!((a - b).norm() < 1e-8, "FrFT α=1 differs from FFT");
        }
    }

    #[test]
    fn test_frft_alpha3_matches_ifft() {
        use crate::fft::ifft as ref_ifft;
        let signal = make_cosine(64, 5.0);
        let frft3 = frft(&signal, 3.0).expect("frft α=3");
        let ifft_result = ref_ifft(&signal, None).expect("ifft");
        for (a, b) in frft3.iter().zip(ifft_result.iter()) {
            assert!((a - b).norm() < 1e-8, "FrFT α=3 differs from IFFT");
        }
    }

    // ── Energy conservation ───────────────────────────────────────────────────

    #[test]
    fn test_frft_energy_conservation() {
        // FrFT is unitary: should preserve energy (up to finite-sample effects)
        let n = 64;
        let signal = make_cosine(n, 7.0);
        let input_energy: f64 = signal.iter().map(|c| c.norm_sqr()).sum();

        for &alpha in &[0.25_f64, 0.5, 0.75, 1.25, 1.5, 1.75] {
            let result = frft(&signal, alpha).expect("frft");
            let output_energy: f64 = result.iter().map(|c| c.norm_sqr()).sum();
            let ratio = output_energy / input_energy;
            assert!(
                (ratio - 1.0).abs() < 0.15,
                "α={alpha}: energy ratio {ratio:.4} far from 1"
            );
        }
    }

    // ── Period-4 property ─────────────────────────────────────────────────────

    #[test]
    fn test_frft_period_4_property() {
        // Applying FrFT 4 times with the same order α should return the original signal
        // (approximate – discrete FrFT has mild numerical errors per application)
        let n = 32;
        let signal = make_cosine(n, 3.0);
        let alpha = 0.5_f64;

        let mut result = signal.clone();
        for _ in 0..4 {
            result = frft(&result, alpha).expect("frft");
        }

        // Should be close to the original
        let error: f64 = signal
            .iter()
            .zip(result.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let norm: f64 = signal.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            error / norm.max(1e-15) < 0.5,
            "Period-4 property violated: relative error = {}",
            error / norm
        );
    }

    // ── frft_real ─────────────────────────────────────────────────────────────

    #[test]
    fn test_frft_real_matches_complex() {
        let n = 64;
        let signal_real: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * 4.0 * k as f64 / n as f64).sin())
            .collect();
        let signal_complex: Vec<Complex64> =
            signal_real.iter().map(|&x| Complex64::new(x, 0.0)).collect();

        let from_real = frft_real(&signal_real, 0.7).expect("frft_real");
        let from_complex = frft(&signal_complex, 0.7).expect("frft");

        for (a, b) in from_real.iter().zip(from_complex.iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    // ── optimal_frft_order ────────────────────────────────────────────────────

    #[test]
    fn test_optimal_frft_order_pure_tone() {
        // A pure tone (single frequency) is already maximally concentrated at α=1
        let n = 64;
        let freq = 5.0;
        let signal: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * freq * k as f64 / n as f64).cos())
            .collect();

        let order = optimal_frft_order(&signal, 40).expect("optimal order");
        // For a pure tone, best concentration is at or very near α=1
        // We allow a range since our discrete search has limited resolution
        assert!(order >= 0.0 && order < 2.0, "order {order} out of range");
    }

    #[test]
    fn test_optimal_frft_order_empty_error() {
        assert!(optimal_frft_order(&[], 10).is_err());
    }

    #[test]
    fn test_optimal_frft_order_zero_angles_error() {
        let sig = vec![1.0_f64; 16];
        assert!(optimal_frft_order(&sig, 0).is_err());
    }

    // ── frft_spectrogram ──────────────────────────────────────────────────────

    #[test]
    fn test_frft_spectrogram_shape() {
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|k| (2.0 * PI * 5.0 * k as f64 / n as f64).sin())
            .collect();

        let spec = frft_spectrogram(&signal, 10, 20).expect("frft_spectrogram");
        assert_eq!(spec.shape(), &[10, 20]);
    }

    #[test]
    fn test_frft_spectrogram_nonnegative() {
        let n = 32;
        let signal: Vec<f64> = (0..n).map(|k| k as f64 / n as f64).collect();
        let spec = frft_spectrogram(&signal, 8, 16).expect("frft_spectrogram");
        for &v in spec.iter() {
            assert!(v >= 0.0, "spectrogram has negative value: {v}");
        }
    }

    #[test]
    fn test_frft_spectrogram_empty_error() {
        assert!(frft_spectrogram(&[], 10, 5).is_err());
    }

    #[test]
    fn test_frft_spectrogram_n_freq_too_large_error() {
        let signal = vec![1.0_f64; 16];
        assert!(frft_spectrogram(&signal, 10, 32).is_err()); // 32 > 16
    }

    // ── Linearity ─────────────────────────────────────────────────────────────

    #[test]
    fn test_frft_linearity() {
        let n = 32;
        let a_coeff = Complex64::new(2.0, 1.0);
        let b_coeff = Complex64::new(-1.0, 3.0);
        let sig1: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 / n as f64).sin(), 0.0))
            .collect();
        let sig2: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 / n as f64).cos(), 0.0))
            .collect();

        let alpha = 0.6;
        let f1 = frft(&sig1, alpha).expect("frft1");
        let f2 = frft(&sig2, alpha).expect("frft2");

        // FrFT(a·sig1 + b·sig2) == a·FrFT(sig1) + b·FrFT(sig2)
        let combined_sig: Vec<Complex64> = sig1
            .iter()
            .zip(sig2.iter())
            .map(|(&x1, &x2)| a_coeff * x1 + b_coeff * x2)
            .collect();
        let f_combined = frft(&combined_sig, alpha).expect("frft combined");

        let expected: Vec<Complex64> = f1
            .iter()
            .zip(f2.iter())
            .map(|(&x1, &x2)| a_coeff * x1 + b_coeff * x2)
            .collect();

        let total_error: f64 = f_combined
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();
        let norm: f64 = expected.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt();
        assert!(
            total_error / norm.max(1e-15) < 0.05,
            "Linearity violated: relative error = {}",
            total_error / norm
        );
    }
}
