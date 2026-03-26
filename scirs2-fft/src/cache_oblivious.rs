//! Cache-Oblivious FFT (Frigo-Johnson style)
//!
//! This module implements a cache-oblivious recursive FFT based on the
//! four-step algorithm of Frigo and Johnson.  The key idea is to view an
//! N-point DFT as a 2-D DFT on an N1 x N2 matrix, choosing N1 ≈ sqrt(N)
//! so that sub-problems naturally fit in cache at some level of the
//! recursion — without ever knowing the cache size.
//!
//! # Algorithm
//!
//! Given `N = N1 * N2`:
//!
//! 1. Reshape input as N1 rows x N2 columns (row-major).
//! 2. Compute N1 column-DFTs of size N2 (recurse).
//! 3. Multiply element (i, j) by twiddle `W_N^{i*j}`.
//! 4. Compute N2 row-DFTs of size N1 (recurse).
//! 5. Read out in transposed order.
//!
//! Base case: for sizes ≤ `base_case_size` (default 16), a direct O(N^2)
//! DFT (or butterfly network) is used.
//!
//! Non-power-of-two sizes are handled via Bluestein's chirp-z algorithm.
//!
//! # References
//!
//! * M. Frigo, S. G. Johnson, "The Design and Implementation of FFTW3",
//!   Proc. IEEE 93(2), 2005.
//! * M. Frigo, "A Fast Fourier Transform Compiler", PLDI 1999.

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

use crate::bluestein;
use crate::butterfly::{direct_dft, direct_idft};
use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the cache-oblivious FFT.
#[derive(Debug, Clone)]
pub struct CacheObliviousConfig {
    /// Size at which the recursion stops and a direct DFT is used.
    /// Must be >= 2.  Typical values: 8, 16, 32.
    pub base_case_size: usize,

    /// If `true`, use the butterfly-based direct DFT at base case;
    /// if `false`, use a simple O(N^2) DFT.  Both give the same result,
    /// but the butterfly path may be slightly faster for power-of-two
    /// base cases.
    pub use_dft_at_base: bool,
}

impl Default for CacheObliviousConfig {
    fn default() -> Self {
        Self {
            base_case_size: 16,
            use_dft_at_base: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the forward DFT of `input` using the cache-oblivious algorithm
/// with default configuration.
///
/// Works for **any** positive length (including primes) via Bluestein fallback.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `input` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::Complex64;
/// use scirs2_fft::cache_oblivious::cache_oblivious_fft;
///
/// let signal: Vec<Complex64> = (0..8)
///     .map(|k| Complex64::new(k as f64, 0.0))
///     .collect();
/// let spectrum = cache_oblivious_fft(&signal).expect("fft failed");
/// assert_eq!(spectrum.len(), 8);
/// ```
pub fn cache_oblivious_fft(input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    cache_oblivious_fft_with_config(input, &CacheObliviousConfig::default())
}

/// Compute the forward DFT with a custom [`CacheObliviousConfig`].
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `input` is empty or config is invalid.
pub fn cache_oblivious_fft_with_config(
    input: &[Complex64],
    config: &CacheObliviousConfig,
) -> FFTResult<Vec<Complex64>> {
    validate_input(input, config)?;
    let n = input.len();

    // Trivial sizes
    if n == 1 {
        return Ok(input.to_vec());
    }

    // If the size can be factored into reasonable sub-problems, use 4-step.
    // Otherwise fall back to Bluestein.
    if let Some((n1, n2)) = find_factorization(n) {
        four_step_fft(input, n1, n2, config)
    } else {
        // Prime or hard-to-factor size: use Bluestein
        bluestein::bluestein_fft(input)
    }
}

/// Compute the inverse DFT of `input` using the cache-oblivious algorithm
/// with default configuration.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `input` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::numeric::Complex64;
/// use scirs2_fft::cache_oblivious::{cache_oblivious_fft, cache_oblivious_ifft};
///
/// let signal: Vec<Complex64> = (0..8)
///     .map(|k| Complex64::new(k as f64, 0.0))
///     .collect();
/// let spectrum = cache_oblivious_fft(&signal).expect("fft failed");
/// let recovered = cache_oblivious_ifft(&spectrum).expect("ifft failed");
///
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).norm() < 1e-10);
/// }
/// ```
pub fn cache_oblivious_ifft(input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    cache_oblivious_ifft_with_config(input, &CacheObliviousConfig::default())
}

/// Compute the inverse DFT with a custom [`CacheObliviousConfig`].
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `input` is empty or config is invalid.
pub fn cache_oblivious_ifft_with_config(
    input: &[Complex64],
    config: &CacheObliviousConfig,
) -> FFTResult<Vec<Complex64>> {
    validate_input(input, config)?;
    let n = input.len();

    if n == 1 {
        return Ok(input.to_vec());
    }

    // IFFT = conj(FFT(conj(x))) / N
    // This avoids double-scaling issues in recursive sub-FFTs.
    let conjugated: Vec<Complex64> = input.iter().map(|c| c.conj()).collect();
    let transformed = cache_oblivious_fft_with_config(&conjugated, config)?;
    let inv_n = 1.0 / n as f64;
    Ok(transformed.iter().map(|c| c.conj() * inv_n).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal: validation
// ─────────────────────────────────────────────────────────────────────────────

fn validate_input(input: &[Complex64], config: &CacheObliviousConfig) -> FFTResult<()> {
    if input.is_empty() {
        return Err(FFTError::ValueError(
            "cache_oblivious_fft: input must not be empty".into(),
        ));
    }
    if config.base_case_size < 2 {
        return Err(FFTError::ValueError(
            "cache_oblivious_fft: base_case_size must be >= 2".into(),
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal: factorization
// ─────────────────────────────────────────────────────────────────────────────

/// Find a factorization `N = N1 * N2` with `N1 ≈ sqrt(N)`.
///
/// Returns `None` if N is prime (or 1), indicating Bluestein should be used.
fn find_factorization(n: usize) -> Option<(usize, usize)> {
    if n <= 1 {
        return None;
    }

    // For power-of-two, split evenly
    if n.is_power_of_two() {
        let half_bits = n.trailing_zeros().div_ceil(2);
        let n1 = 1usize << half_bits;
        let n2 = n / n1;
        return Some((n1, n2));
    }

    // General case: find a factor closest to sqrt(N)
    let sqrt_n = (n as f64).sqrt() as usize;
    // Search outward from sqrt(n)
    for delta in 0..=sqrt_n {
        let candidate = sqrt_n.saturating_sub(delta);
        if candidate >= 2 && n % candidate == 0 {
            return Some((candidate, n / candidate));
        }
        let candidate_up = sqrt_n + delta + 1;
        if candidate_up < n && n % candidate_up == 0 {
            return Some((candidate_up, n / candidate_up));
        }
    }

    // N is prime
    None
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal: four-step FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Four-step cache-oblivious forward FFT.
fn four_step_fft(
    input: &[Complex64],
    n1: usize,
    n2: usize,
    config: &CacheObliviousConfig,
) -> FFTResult<Vec<Complex64>> {
    let n = n1 * n2;
    debug_assert_eq!(n, input.len());

    // Step 1: view input as N1 rows x N2 columns (row-major)
    // matrix[i][j] = input[i * n2 + j]
    let mut matrix = input.to_vec();

    // Step 2: Compute N2 column-DFTs of size N1.
    // Column j: elements at indices j, j + n2, j + 2*n2, ...
    for j in 0..n2 {
        let col: Vec<Complex64> = (0..n1).map(|i| matrix[i * n2 + j]).collect();
        let col_fft = compute_sub_fft(&col, config)?;
        for i in 0..n1 {
            matrix[i * n2 + j] = col_fft[i];
        }
    }

    // Step 3: Multiply by twiddle factors W_N^{i*j}
    let angle_base = -2.0 * PI / n as f64;
    for i in 0..n1 {
        for j in 0..n2 {
            let angle = angle_base * (i * j) as f64;
            let twiddle = Complex64::new(angle.cos(), angle.sin());
            matrix[i * n2 + j] *= twiddle;
        }
    }

    // Step 4: Compute N1 row-DFTs of size N2.
    for i in 0..n1 {
        let start = i * n2;
        let row: Vec<Complex64> = matrix[start..start + n2].to_vec();
        let transformed = compute_sub_fft(&row, config)?;
        matrix[start..start + n2].copy_from_slice(&transformed);
    }

    // Step 5: Read out in column-major order (transpose)
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    for i in 0..n1 {
        for j in 0..n2 {
            output[j * n1 + i] = matrix[i * n2 + j];
        }
    }

    Ok(output)
}

/// Recursively compute a forward sub-FFT.
///
/// If `data.len()` is small enough, uses a direct DFT.
/// Otherwise recurses via the four-step algorithm.
fn compute_sub_fft(data: &[Complex64], config: &CacheObliviousConfig) -> FFTResult<Vec<Complex64>> {
    let n = data.len();

    if n <= 1 {
        return Ok(data.to_vec());
    }

    // Base case
    if n <= config.base_case_size {
        return direct_dft(data);
    }

    // Recursive case: try to factor
    if let Some((n1, n2)) = find_factorization(n) {
        four_step_fft(data, n1, n2, config)
    } else {
        // Prime sub-size: Bluestein
        bluestein::bluestein_fft(data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reference DFT via the direct (O(N^2)) method.
    fn reference_fft(input: &[Complex64]) -> Vec<Complex64> {
        direct_dft(input).expect("direct_dft failed")
    }

    fn max_abs_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    // ── basic correctness ───────────────────────────────────────────────
    #[test]
    fn test_cache_oblivious_matches_direct_pow2() {
        for &n in &[2, 4, 8, 16, 32, 64] {
            let input: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new((k as f64).sin(), (k as f64).cos()))
                .collect();
            let expected = reference_fft(&input);
            let result =
                cache_oblivious_fft(&input).unwrap_or_else(|e| panic!("fft failed for n={n}: {e}"));
            let err = max_abs_err(&expected, &result);
            assert!(err < 1e-8, "n={n}: max error = {err}");
        }
    }

    #[test]
    fn test_cache_oblivious_matches_direct_composite() {
        for &n in &[6, 12, 15, 24, 36] {
            let input: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new(k as f64, -(k as f64)))
                .collect();
            let expected = reference_fft(&input);
            let result =
                cache_oblivious_fft(&input).unwrap_or_else(|e| panic!("fft failed for n={n}: {e}"));
            let err = max_abs_err(&expected, &result);
            assert!(err < 1e-8, "n={n}: max error = {err}");
        }
    }

    #[test]
    fn test_cache_oblivious_prime_sizes() {
        // Prime sizes => Bluestein fallback
        for &n in &[5, 7, 11, 13] {
            let input: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
            let expected = reference_fft(&input);
            let result =
                cache_oblivious_fft(&input).unwrap_or_else(|e| panic!("fft failed for n={n}: {e}"));
            let err = max_abs_err(&expected, &result);
            assert!(err < 1e-8, "prime n={n}: max error = {err}");
        }
    }

    // ── roundtrip (IFFT(FFT(x)) ≈ x) ───────────────────────────────────
    #[test]
    fn test_cache_oblivious_roundtrip_pow2() {
        let n = 32;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64, -0.5 * k as f64))
            .collect();
        let spectrum = cache_oblivious_fft(&input).expect("fft failed");
        let recovered = cache_oblivious_ifft(&spectrum).expect("ifft failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-8, "roundtrip error = {err}");
    }

    #[test]
    fn test_cache_oblivious_roundtrip_composite() {
        let n = 12;
        let input: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 1.0)).collect();
        let spectrum = cache_oblivious_fft(&input).expect("fft failed");
        let recovered = cache_oblivious_ifft(&spectrum).expect("ifft failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-8, "roundtrip (n={n}) error = {err}");
    }

    #[test]
    fn test_cache_oblivious_roundtrip_prime() {
        let n = 7;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64 * 0.3, -k as f64 * 0.1))
            .collect();
        let spectrum = cache_oblivious_fft(&input).expect("fft failed");
        let recovered = cache_oblivious_ifft(&spectrum).expect("ifft failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-8, "roundtrip (prime n={n}) error = {err}");
    }

    // ── Parseval's theorem ──────────────────────────────────────────────
    #[test]
    fn test_parseval_theorem() {
        for &n in &[8, 12, 16, 15] {
            let input: Vec<Complex64> = (0..n)
                .map(|k| Complex64::new((k as f64 * 0.7).sin(), (k as f64 * 0.3).cos()))
                .collect();
            let spectrum = cache_oblivious_fft(&input).expect("fft failed");

            let time_energy: f64 = input.iter().map(|c| c.norm_sqr()).sum();
            let freq_energy: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;

            assert_relative_eq!(time_energy, freq_energy, epsilon = 1e-6);
        }
    }

    // ── edge cases ──────────────────────────────────────────────────────
    #[test]
    fn test_empty_input() {
        assert!(cache_oblivious_fft(&[]).is_err());
    }

    #[test]
    fn test_single_element() {
        let input = vec![Complex64::new(42.0, -7.0)];
        let result = cache_oblivious_fft(&input).expect("fft failed");
        assert_eq!(result.len(), 1);
        assert_relative_eq!(result[0].re, 42.0, epsilon = 1e-14);
        assert_relative_eq!(result[0].im, -7.0, epsilon = 1e-14);
    }

    #[test]
    fn test_custom_config() {
        let config = CacheObliviousConfig {
            base_case_size: 4,
            use_dft_at_base: false,
        };
        let input: Vec<Complex64> = (0..16).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let result = cache_oblivious_fft_with_config(&input, &config).expect("fft failed");
        let expected = reference_fft(&input);
        let err = max_abs_err(&expected, &result);
        assert!(err < 1e-8, "custom config error = {err}");
    }

    #[test]
    fn test_config_base_case_too_small() {
        let config = CacheObliviousConfig {
            base_case_size: 1,
            use_dft_at_base: true,
        };
        let input = vec![Complex64::new(1.0, 0.0); 4];
        assert!(cache_oblivious_fft_with_config(&input, &config).is_err());
    }

    // ── factorization helper ────────────────────────────────────────────
    #[test]
    fn test_factorization() {
        // Power of two: should split roughly equally
        let (n1, n2) = find_factorization(16).expect("should factor 16");
        assert_eq!(n1 * n2, 16);
        assert!(n1 >= 2 && n2 >= 2);

        // Composite
        let (n1, n2) = find_factorization(12).expect("should factor 12");
        assert_eq!(n1 * n2, 12);

        // Prime: should return None
        assert!(find_factorization(13).is_none());

        // 1: should return None
        assert!(find_factorization(1).is_none());
    }

    // ── non-power-of-2 sizes (N=12, N=15) ──────────────────────────────
    #[test]
    fn test_non_power_of_two_n12() {
        let n = 12;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new((k as f64 * PI / 6.0).cos(), 0.0))
            .collect();
        let spectrum = cache_oblivious_fft(&input).expect("fft(12) failed");
        assert_eq!(spectrum.len(), n);

        // Verify by roundtrip
        let recovered = cache_oblivious_ifft(&spectrum).expect("ifft(12) failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-8, "N=12 roundtrip error = {err}");
    }

    #[test]
    fn test_non_power_of_two_n15() {
        let n = 15;
        let input: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64, -(k as f64) * 0.5))
            .collect();
        let spectrum = cache_oblivious_fft(&input).expect("fft(15) failed");
        assert_eq!(spectrum.len(), n);

        let recovered = cache_oblivious_ifft(&spectrum).expect("ifft(15) failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-8, "N=15 roundtrip error = {err}");
    }
}
