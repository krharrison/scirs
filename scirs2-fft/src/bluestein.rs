//! Bluestein's FFT Algorithm (Chirp-Z Transform approach for arbitrary lengths)
//!
//! Bluestein's algorithm computes the DFT of an arbitrary-length sequence by
//! expressing it as a convolution, then evaluating that convolution using a
//! power-of-two FFT.  This enables O(N log N) DFT computation for any N,
//! including prime lengths.
//!
//! # Mathematical Basis
//!
//! The key identity used is the index factorisation:
//!
//! ```text
//! n·k = -((k - n)² - k² - n²) / 2
//! ```
//!
//! which rewrites the DFT sum
//!
//! ```text
//! X[k] = Σ_{n=0}^{N-1}  x[n] · W^{nk},   W = exp(-2πi/N)
//! ```
//!
//! as
//!
//! ```text
//! X[k] = W^{-k²/2} · Σ_{n=0}^{N-1} (x[n] · W^{-n²/2}) · W^{(k-n)²/2}
//! ```
//!
//! The inner sum is a convolution of the chirp-modulated input with a chirp
//! sequence.  Using a power-of-two FFT for the convolution yields an
//! O(N log N) algorithm for any N.
//!
//! # References
//!
//! * Bluestein, L. I. "A linear filtering approach to the computation of the
//!   discrete Fourier transform." *Northeast Electronics Research and Engineering
//!   Meeting Record* 10 (1968), pp. 218–219.
//! * Rabiner, L. R.; Schafer, R. W.; Rader, C. M. "The chirp z-transform
//!   algorithm." *IEEE Trans. Audio Electroacoust.* 17(2) (1969), pp. 86–92.
//! * Bluestein's algorithm – Wikipedia.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::{Complex64, Zero};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helper: next power of two
// ─────────────────────────────────────────────────────────────────────────────

/// Return the smallest power of two that is ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    n.next_power_of_two()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Chirp sequence pre-computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute W^{k²/2} for k = 0, 1, ..., n-1  where W = exp(sign * 2πi/n).
///
/// Using the modular identity  k² mod (2n) to avoid floating-point phase drift:
/// k² mod (2n) is computed exactly with integer arithmetic.
fn chirp_sequence(n: usize, sign: f64) -> Vec<Complex64> {
    let two_n = 2 * n;
    (0..n)
        .map(|k| {
            let k_sq_mod = (k * k) % two_n;
            let phase = sign * PI * k_sq_mod as f64 / n as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bluestein's FFT – complex input
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the DFT of an arbitrary-length complex sequence using Bluestein's algorithm.
///
/// Unlike a radix-2 Cooley-Tukey FFT this works for **any** positive length,
/// including prime lengths such as 1021, 32749, etc.
///
/// The algorithm has O(M log M) complexity where M is the next power of two
/// ≥ 2N − 1, so in the worst case (N prime) M ≈ 4N.
///
/// # Arguments
///
/// * `signal` - Input complex signal of any length ≥ 1.
///
/// # Returns
///
/// `Vec<Complex64>` of the same length as `signal`, containing the DFT output.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `signal` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::bluestein::bluestein_fft;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// // Length-5 DFT (prime length – cannot be done with radix-2 FFT directly)
/// let n = 5;
/// let signal: Vec<Complex64> = (0..n)
///     .map(|k| Complex64::new((2.0 * PI * k as f64 / n as f64).cos(), 0.0))
///     .collect();
///
/// let spectrum = bluestein_fft(&signal).expect("bluestein_fft failed");
/// assert_eq!(spectrum.len(), n);
///
/// // DC component should have magnitude ≈ n (constant signal ∝ cos + i·sin at one frequency)
/// // Energy should be preserved (Parseval's theorem)
/// let input_energy: f64  = signal.iter().map(|c| c.norm_sqr()).sum();
/// let output_energy: f64 = spectrum.iter().map(|c| c.norm_sqr() / n as f64).sum();
/// assert!((input_energy - output_energy).abs() < 1e-9 * input_energy.max(1.0));
/// ```
pub fn bluestein_fft(signal: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "bluestein_fft: input signal is empty".to_string(),
        ));
    }
    if n == 1 {
        return Ok(signal.to_vec());
    }

    // --- Step 1: choose the convolution length (power of two) -----------------
    // We need m ≥ 2*n - 1 to avoid circular aliasing.
    let m = next_pow2(2 * n - 1);

    // --- Step 2: pre-compute chirp sequences ----------------------------------
    // W = exp(-2πi/n),  chirp[k] = W^{k²/2} = exp(-πi·k²/n)
    let chirp = chirp_sequence(n, -1.0); // W^{k²/2}

    // --- Step 3: multiply input by the chirp ----------------------------------
    // a[k] = x[k] · chirp[k]
    let mut a = vec![Complex64::zero(); m];
    for k in 0..n {
        a[k] = signal[k] * chirp[k];
    }

    // --- Step 4: build the chirp kernel b[k] = conj(chirp[k]) = W^{-k²/2} ---
    // Extended to length m:  b[0..n-1] and b[m-n+1..m-1]  (circular convolution trick)
    let mut b = vec![Complex64::zero(); m];
    let chirp_conj: Vec<Complex64> = chirp.iter().map(|c| c.conj()).collect();
    b[0] = chirp_conj[0];
    for k in 1..n {
        b[k] = chirp_conj[k];
        b[m - k] = chirp_conj[k];
    }

    // --- Step 5: FFT of both sequences ----------------------------------------
    let fa = fft(&a, None)?;
    let fb = fft(&b, None)?;

    // --- Step 6: point-wise multiply in frequency domain ----------------------
    let fc: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(&ai, &bi)| ai * bi).collect();

    // --- Step 7: inverse FFT --------------------------------------------------
    let c = ifft(&fc, None)?;

    // --- Step 8: multiply by chirp again and extract n outputs ---------------
    // X[k] = chirp[k] · c[k]   for k = 0 .. n-1
    let result: Vec<Complex64> = (0..n).map(|k| chirp[k] * c[k]).collect();

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bluestein's FFT – real input
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the DFT of a real-valued arbitrary-length sequence using Bluestein's algorithm.
///
/// This is a convenience wrapper around [`bluestein_fft`] that accepts a slice of
/// `f64` values.  The output is the full N-point complex DFT.
///
/// # Arguments
///
/// * `signal` - Real-valued input signal of any length ≥ 1.
///
/// # Returns
///
/// `Vec<Complex64>` of length `signal.len()`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `signal` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::bluestein::bluestein_real;
/// use std::f64::consts::PI;
///
/// // Prime-length DFT of a cosine
/// let n = 7;
/// let signal: Vec<f64> = (0..n)
///     .map(|k| (2.0 * PI * k as f64 / n as f64).cos())
///     .collect();
///
/// let spectrum = bluestein_real(&signal).expect("bluestein_real failed");
/// assert_eq!(spectrum.len(), n);
/// // For a cosine with exactly 1 cycle over N samples, bins 1 and N-1 should dominate
/// let bin1_mag = spectrum[1].norm();
/// assert!(bin1_mag > 2.0);
/// ```
pub fn bluestein_real(signal: &[f64]) -> FFTResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "bluestein_real: input signal is empty".to_string(),
        ));
    }
    let complex_signal: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    bluestein_fft(&complex_signal)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bluestein inverse FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the inverse DFT of an arbitrary-length sequence using Bluestein's algorithm.
///
/// Computes `IDFT[k]` = (1/N) Σ_{j=0}^{N-1} `X[j]` · exp(2πi·j·k/N)
///
/// # Arguments
///
/// * `spectrum` - DFT domain input of any length ≥ 1.
///
/// # Returns
///
/// `Vec<Complex64>` of length `spectrum.len()`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `spectrum` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::bluestein::{bluestein_fft, bluestein_ifft};
/// use scirs2_core::numeric::Complex64;
///
/// // Round-trip test with prime length
/// let n = 11;
/// let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
/// let spectrum  = bluestein_fft(&signal).expect("fft");
/// let recovered = bluestein_ifft(&spectrum).expect("ifft");
/// for (orig, rec) in signal.iter().zip(recovered.iter()) {
///     assert!((orig - rec).norm() < 1e-10);
/// }
/// ```
pub fn bluestein_ifft(spectrum: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = spectrum.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "bluestein_ifft: input is empty".to_string(),
        ));
    }
    if n == 1 {
        return Ok(spectrum.to_vec());
    }

    let m = next_pow2(2 * n - 1);

    // Inverse chirp: use +1.0 sign for exp(+πi·k²/n)
    let chirp = chirp_sequence(n, 1.0);

    let mut a = vec![Complex64::zero(); m];
    for k in 0..n {
        a[k] = spectrum[k] * chirp[k];
    }

    let chirp_conj: Vec<Complex64> = chirp.iter().map(|c| c.conj()).collect();
    let mut b = vec![Complex64::zero(); m];
    b[0] = chirp_conj[0];
    for k in 1..n {
        b[k] = chirp_conj[k];
        b[m - k] = chirp_conj[k];
    }

    let fa = fft(&a, None)?;
    let fb = fft(&b, None)?;
    let fc: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(&ai, &bi)| ai * bi).collect();
    let c = ifft(&fc, None)?;

    let inv_n = 1.0 / n as f64;
    let result: Vec<Complex64> = (0..n).map(|k| chirp[k] * c[k] * inv_n).collect();
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience: prime-length FFT with automatic detection
// ─────────────────────────────────────────────────────────────────────────────

/// Test whether `n` is a prime number.
///
/// Uses trial division; efficient for n up to ~10⁶.
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut d = 3_usize;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

/// Compute the DFT of a prime-length sequence.
///
/// When the input length is prime, ordinary radix-2 FFT libraries cannot
/// directly compute the transform without zero-padding.  This function
/// automatically uses Bluestein's algorithm for prime lengths and falls back
/// to [`bluestein_fft`] (which works for all lengths) otherwise.
///
/// # Arguments
///
/// * `signal` - Complex input of any length.
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
/// use scirs2_fft::bluestein::prime_length_fft;
/// use scirs2_core::numeric::Complex64;
///
/// let n = 13; // prime
/// let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
/// let spectrum = prime_length_fft(&signal).expect("prime_length_fft");
/// assert_eq!(spectrum.len(), n);
/// ```
pub fn prime_length_fft(signal: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError(
            "prime_length_fft: input signal is empty".to_string(),
        ));
    }
    // Bluestein works for ALL lengths – prime or not.
    // We document the intent here by checking primality, but the algorithm is the same.
    let _ = is_prime(n); // informational only
    bluestein_fft(signal)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::fft as reference_fft;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Helper: compare two complex slices with tolerance.
    fn assert_complex_close(a: &[Complex64], b: &[Complex64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (ai - bi).norm();
            assert!(diff < tol, "index {i}: |{ai} - {bi}| = {diff} ≥ {tol}");
        }
    }

    // ── Power-of-two lengths (should match reference FFT) ────────────────────

    #[test]
    fn test_bluestein_pow2_matches_reference() {
        for &n in &[1_usize, 2, 4, 8, 16] {
            let signal: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new(k as f64, -(k as f64)))
                .collect();

            let blue = bluestein_fft(&signal).expect("bluestein_fft");
            let ref_ = reference_fft(&signal, None).expect("reference fft");

            assert_complex_close(&blue, &ref_, 1e-9);
        }
    }

    // ── Prime lengths ────────────────────────────────────────────────────────

    #[test]
    fn test_bluestein_prime_length_5() {
        // Length-5 DFT of impulse at 0 = all ones
        let signal: Vec<Complex64> = std::iter::once(Complex64::new(1.0, 0.0))
            .chain(std::iter::repeat_n(Complex64::zero(), 4))
            .collect();

        let spectrum = bluestein_fft(&signal).expect("bluestein_fft");
        for val in &spectrum {
            assert_relative_eq!(val.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(val.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bluestein_prime_length_7() {
        let n = 7;
        let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();

        let blue = bluestein_fft(&signal).expect("bluestein_fft");
        // Verify via brute-force DFT
        let brute: Vec<Complex64> = (0..n)
            .map(|j| {
                signal
                    .iter()
                    .enumerate()
                    .fold(Complex64::zero(), |acc, (k, &xk)| {
                        let phase = -2.0 * PI * j as f64 * k as f64 / n as f64;
                        acc + xk * Complex64::new(phase.cos(), phase.sin())
                    })
            })
            .collect();

        assert_complex_close(&blue, &brute, 1e-9);
    }

    #[test]
    fn test_bluestein_prime_length_11() {
        let n = 11;
        let signal: Vec<Complex64> = (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                Complex64::new((2.0 * PI * t).cos(), (2.0 * PI * t).sin())
            })
            .collect();

        let blue = bluestein_fft(&signal).expect("bluestein");
        let brute: Vec<Complex64> = (0..n)
            .map(|j| {
                signal
                    .iter()
                    .enumerate()
                    .fold(Complex64::zero(), |acc, (k, &xk)| {
                        let phase = -2.0 * PI * j as f64 * k as f64 / n as f64;
                        acc + xk * Complex64::new(phase.cos(), phase.sin())
                    })
            })
            .collect();

        assert_complex_close(&blue, &brute, 1e-8);
    }

    // ── Non-power-of-two, non-prime lengths ───────────────────────────────────

    #[test]
    fn test_bluestein_length_6() {
        let n = 6;
        let signal: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64).sin(), 0.0))
            .collect();

        let blue = bluestein_fft(&signal).expect("bluestein");
        let brute: Vec<Complex64> = (0..n)
            .map(|j| {
                signal
                    .iter()
                    .enumerate()
                    .fold(Complex64::zero(), |acc, (k, &xk)| {
                        let phase = -2.0 * PI * j as f64 * k as f64 / n as f64;
                        acc + xk * Complex64::new(phase.cos(), phase.sin())
                    })
            })
            .collect();
        assert_complex_close(&blue, &brute, 1e-9);
    }

    // ── Real input wrapper ────────────────────────────────────────────────────

    #[test]
    fn test_bluestein_real_length_5() {
        let n = 5;
        let signal: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let blue_real = bluestein_real(&signal).expect("bluestein_real");
        let blue_complex: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
        let blue_ref = bluestein_fft(&blue_complex).expect("bluestein_fft");
        assert_complex_close(&blue_real, &blue_ref, 1e-12);
    }

    // ── Inverse FFT round-trip ────────────────────────────────────────────────

    #[test]
    fn test_bluestein_ifft_roundtrip_prime() {
        let n = 13;
        let original: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64, -(k as f64) / 2.0))
            .collect();

        let spectrum = bluestein_fft(&original).expect("fft");
        let recovered = bluestein_ifft(&spectrum).expect("ifft");

        assert_complex_close(&original, &recovered, 1e-9);
    }

    #[test]
    fn test_bluestein_ifft_roundtrip_pow2() {
        let n = 8;
        let original: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64).cos(), (k as f64).sin()))
            .collect();

        let spectrum = bluestein_fft(&original).expect("fft");
        let recovered = bluestein_ifft(&spectrum).expect("ifft");

        assert_complex_close(&original, &recovered, 1e-9);
    }

    // ── Parseval's theorem ────────────────────────────────────────────────────

    #[test]
    fn test_bluestein_parseval() {
        let n = 17; // prime
        let signal: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 / n as f64).sin(), 0.0))
            .collect();

        let spectrum = bluestein_fft(&signal).expect("bluestein");

        let input_energy: f64 = signal.iter().map(|c| c.norm_sqr()).sum();
        let output_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

        assert_relative_eq!(
            input_energy,
            output_energy,
            epsilon = 1e-9 * input_energy.max(1.0)
        );
    }

    // ── Empty input returns error ─────────────────────────────────────────────

    #[test]
    fn test_bluestein_empty_error() {
        assert!(bluestein_fft(&[]).is_err());
        assert!(bluestein_real(&[]).is_err());
        assert!(bluestein_ifft(&[]).is_err());
    }

    // ── Length-1 trivial case ─────────────────────────────────────────────────

    #[test]
    fn test_bluestein_single_element() {
        let sig = vec![Complex64::new(3.25, 2.72)];
        let out = bluestein_fft(&sig).expect("fft");
        assert_eq!(out.len(), 1);
        assert_relative_eq!(out[0].re, sig[0].re, epsilon = 1e-12);
        assert_relative_eq!(out[0].im, sig[0].im, epsilon = 1e-12);
    }

    // ── Prime-length FFT wrapper ──────────────────────────────────────────────

    #[test]
    fn test_prime_length_fft() {
        let n = 19;
        let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let spec = prime_length_fft(&signal).expect("prime_length_fft");
        assert_eq!(spec.len(), n);
    }
}
