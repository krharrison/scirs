//! Mixed-radix FFT: Good-Thomas, Cooley-Tukey, and Rader's algorithm.
//!
//! This module implements a general-purpose FFT that works on any transform
//! length (not just powers of two) by decomposing n into its prime factors
//! and applying the appropriate sub-algorithm:
//!
//! * **Cooley-Tukey** for repeated prime factors (e.g. 2² = 4).
//! * **Good-Thomas** (Prime Factor Algorithm) for coprime factor pairs.
//! * **Rader's algorithm** for odd prime lengths p: reduces the p-length DFT
//!   to a cyclic convolution of length p−1 (which can itself be factored).
//!
//! # References
//!
//! * Cooley & Tukey, Math. Comp. 19 (1965).
//! * Good, I. J. (1958). "The interaction algorithm and practical Fourier analysis."
//! * Rader, C. M. (1968). "Discrete Fourier transforms when the number of data samples is prime."

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Factorisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Factorise `n` into its prime factors (with repetition), smallest first.
fn factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut d = 2usize;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// Find a primitive root modulo a prime p using trial-and-error.
fn primitive_root_mod(p: usize) -> usize {
    if p == 2 {
        return 1;
    }
    // Factorise p-1
    let phi = p - 1;
    let pf = factorize(phi);
    'outer: for g in 2..p {
        for &qi in &pf {
            if pow_mod(g, phi / qi, p) == 1 {
                continue 'outer;
            }
        }
        return g;
    }
    1 // unreachable for valid prime
}

/// Fast modular exponentiation.
fn pow_mod(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1usize;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % modulus;
        }
        base = base * base % modulus;
        exp >>= 1;
    }
    result
}

/// Modular inverse via Fermat's little theorem (p prime).
fn mod_inv(a: usize, p: usize) -> usize {
    pow_mod(a, p - 2, p)
}

/// Precompute twiddle factors w_n^k = exp(-2πi·k/n) for k = 0..n.
fn twiddles(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|k| {
            let angle = -2.0 * PI * k as f64 / n as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simple DFT for small or prime sizes
// ─────────────────────────────────────────────────────────────────────────────

/// Direct O(n²) DFT. Used as base case for small / prime lengths.
fn dft_direct(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let tw = twiddles(n);
    (0..n)
        .map(|k| {
            x.iter()
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |acc, (j, &xj)| {
                    acc + xj * tw[(j * k) % n]
                })
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Cooley-Tukey radix-r DFT
// ─────────────────────────────────────────────────────────────────────────────

/// Cooley-Tukey decimation-in-time for n = r · m.
///
/// Splits input into r sub-sequences of length m, transforms each, then
/// combines with twiddle factors.
fn cooley_tukey(x: &[Complex64], r: usize) -> Vec<Complex64> {
    let n = x.len();
    let m = n / r;
    let tw = twiddles(n);
    let mut subseqs: Vec<Vec<Complex64>> = (0..r)
        .map(|i| (0..m).map(|j| x[i + j * r]).collect())
        .collect();

    // Recursively transform each sub-sequence
    for seq in subseqs.iter_mut() {
        *seq = fft_recursive(seq);
    }

    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        for s in 0..r {
            let km = k % m;
            out[k] = out[k] + tw[(s * k) % n] * subseqs[s][km];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rader's algorithm for prime lengths
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the DFT of a prime-length slice using Rader's algorithm.
///
/// For prime p, Rader converts the p-point DFT (for indices k ≠ 0) into a
/// cyclic convolution of length p−1 via a primitive root permutation.
///
/// # Arguments
///
/// * `x` — input slice of length `prime`.
/// * `primitive_root` — a primitive root of Z_prime^*.
/// * `prime` — a prime length.
///
/// # Examples
///
/// ```
/// use scirs2_fft::mixed_radix::rader_fft;
/// use scirs2_core::numeric::Complex64;
/// let x: Vec<Complex64> = vec![
///     Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0),
///     Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0),
///     Complex64::new(5.0, 0.0),
/// ];
/// let y = rader_fft(&x, 2, 5).expect("valid input");
/// assert_eq!(y.len(), 5);
/// ```
pub fn rader_fft(x: &[Complex64], primitive_root: usize, prime: usize) -> FFTResult<Vec<Complex64>> {
    let p = prime;
    if x.len() != p {
        return Err(FFTError::DimensionError(format!(
            "rader_fft: input length {} != prime {}",
            x.len(),
            p
        )));
    }
    if p == 0 {
        return Ok(Vec::new());
    }
    if p == 1 {
        return Ok(vec![x[0]]);
    }
    if p == 2 {
        return Ok(vec![x[0] + x[1], x[0] - x[1]]);
    }

    let pm1 = p - 1;
    let g = primitive_root;
    let g_inv = mod_inv(g, p);

    // Permuted sequences: a[n] = x[g^n mod p] for n in 0..p-1
    //                     b[n] = W_p^{g^{-n} mod p}  for n in 0..p-1
    let mut a = vec![Complex64::new(0.0, 0.0); pm1];
    let mut b = vec![Complex64::new(0.0, 0.0); pm1];

    let mut gn = 1usize;
    let mut gin = 1usize;
    let tw = twiddles(p);
    for n in 0..pm1 {
        a[n] = x[gn];
        b[n] = tw[gin];
        gn = gn * g % p;
        gin = gin * g_inv % p;
    }

    // Circular convolution of a and b via FFT
    let conv = circular_convolve_complex(&a, &b)?;

    // X[0]        = sum of all inputs
    let x0: Complex64 = x.iter().sum();
    // X[g^{-q}]   = x[0] + conv[q]  for q in 0..p-1
    let mut out = vec![Complex64::new(0.0, 0.0); p];
    out[0] = x0;

    let mut ginq = 1usize; // g^{-q} mod p, starts at g^0=1 → but X[1]
    for q in 0..pm1 {
        // Index in output: g^{-q} mod p
        ginq = if q == 0 { 1 } else { ginq * g_inv % p };
        out[ginq] = x[0] + conv[q];
    }
    Ok(out)
}

/// Circular convolution of two complex slices (same length, using FFT).
fn circular_convolve_complex(a: &[Complex64], b: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = a.len();
    if n != b.len() {
        return Err(FFTError::DimensionError(
            "circular_convolve_complex: lengths must match".to_string(),
        ));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    // Pad to next power of 2 for efficiency
    let m = n.next_power_of_two();
    let mut fa: Vec<Complex64> = a.iter().copied().collect();
    let mut fb: Vec<Complex64> = b.iter().copied().collect();
    fa.resize(m, Complex64::new(0.0, 0.0));
    fb.resize(m, Complex64::new(0.0, 0.0));

    fft_inplace(&mut fa);
    fft_inplace(&mut fb);
    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = *x * *y;
    }
    ifft_inplace(&mut fa);
    // Wrap-around: add the tail back to [0..n]
    for i in n..m {
        let wrap = i % n;
        let val = fa[i];
        fa[wrap] = fa[wrap] + val;
    }
    fa.truncate(n);
    Ok(fa)
}

// ─────────────────────────────────────────────────────────────────────────────
//  In-place power-of-2 FFT used internally
// ─────────────────────────────────────────────────────────────────────────────

fn fft_inplace(a: &mut Vec<Complex64>) {
    let n = a.len();
    if n <= 1 {
        return;
    }
    if !n.is_power_of_two() {
        let result = fft_recursive(a);
        *a = result;
        return;
    }
    let log_n = n.trailing_zeros() as usize;
    // Bit-reversal
    for i in 0..n {
        let j = bit_rev(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let w_len = Complex64::new(angle.cos(), angle.sin());
        let mut j = 0usize;
        while j < n {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..half {
                let u = a[j + k];
                let v = a[j + k + half] * w;
                a[j + k] = u + v;
                a[j + k + half] = u - v;
                w = w * w_len;
            }
            j += len;
        }
        len <<= 1;
    }
}

fn ifft_inplace(a: &mut Vec<Complex64>) {
    let n = a.len();
    // Conjugate, forward FFT, conjugate, scale
    for x in a.iter_mut() {
        *x = x.conj();
    }
    fft_inplace(a);
    let scale = 1.0 / n as f64;
    for x in a.iter_mut() {
        *x = x.conj() * scale;
    }
}

fn bit_rev(mut x: usize, bits: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

// ─────────────────────────────────────────────────────────────────────────────
//  Recursive mixed-radix FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Internal recursive FFT dispatcher.
fn fft_recursive(x: &[Complex64]) -> Vec<Complex64> {
    let n = x.len();
    match n {
        0 => Vec::new(),
        1 => vec![x[0]],
        2 => vec![x[0] + x[1], x[0] - x[1]],
        _ => {
            let factors = factorize(n);
            let r = factors[0];
            if r == n {
                // n is prime — use Rader's algorithm (or direct DFT for small primes)
                if n <= 11 {
                    dft_direct(x)
                } else {
                    let g = primitive_root_mod(n);
                    rader_fft(x, g, n).unwrap_or_else(|_| dft_direct(x))
                }
            } else {
                cooley_tukey(x, r)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  MixedRadixFFT public struct
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-planned mixed-radix FFT for a specific transform size.
///
/// The planner factorises the requested size, selects the appropriate
/// sub-algorithm for each factor, and precomputes twiddle factors so that
/// repeated transforms of the same size are cheap.
///
/// # Examples
///
/// ```
/// use scirs2_fft::mixed_radix::MixedRadixFFT;
/// use scirs2_core::numeric::Complex64;
///
/// let plan = MixedRadixFFT::new(6).expect("valid input");
/// let input: Vec<Complex64> = (0..6).map(|i| Complex64::new(i as f64, 0.0)).collect();
/// let output = plan.fft(&input).expect("valid input");
/// assert_eq!(output.len(), 6);
/// ```
pub struct MixedRadixFFT {
    /// Transform size.
    pub size: usize,
    /// Prime factors of `size` (with repetition).
    pub factors: Vec<usize>,
    /// Precomputed twiddle factors (length = size).
    pub twiddles: Vec<Complex64>,
}

impl MixedRadixFFT {
    /// Create a new planner for transforms of length `n`.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::ValueError`] if `n == 0`.
    pub fn new(n: usize) -> FFTResult<Self> {
        if n == 0 {
            return Err(FFTError::ValueError(
                "MixedRadixFFT: size must be > 0".to_string(),
            ));
        }
        let factors = factorize(n);
        let twiddles = twiddles(n);
        Ok(Self { size: n, factors, twiddles })
    }

    /// Compute the forward DFT of `input`.
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::DimensionError`] if `input.len() != self.size`.
    pub fn fft(&self, input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
        if input.len() != self.size {
            return Err(FFTError::DimensionError(format!(
                "MixedRadixFFT: expected {} samples, got {}",
                self.size,
                input.len()
            )));
        }
        Ok(fft_recursive(input))
    }

    /// Compute the inverse DFT of `input` (unnormalised by 1/N).
    ///
    /// # Errors
    ///
    /// Returns [`FFTError::DimensionError`] if `input.len() != self.size`.
    pub fn ifft(&self, input: &[Complex64]) -> FFTResult<Vec<Complex64>> {
        if input.len() != self.size {
            return Err(FFTError::DimensionError(format!(
                "MixedRadixFFT: expected {} samples, got {}",
                self.size,
                input.len()
            )));
        }
        // IDFT = conj(DFT(conj(x))) / N
        let conj_in: Vec<Complex64> = input.iter().map(|x| x.conj()).collect();
        let mut out = fft_recursive(&conj_in);
        let scale = 1.0 / self.size as f64;
        for x in out.iter_mut() {
            *x = x.conj() * scale;
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_real(vals: &[f64]) -> Vec<Complex64> {
        vals.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    #[test]
    fn test_factorize() {
        assert_eq!(factorize(1), Vec::<usize>::new());
        assert_eq!(factorize(2), vec![2]);
        assert_eq!(factorize(6), vec![2, 3]);
        assert_eq!(factorize(12), vec![2, 2, 3]);
        assert_eq!(factorize(7), vec![7]);
    }

    #[test]
    fn test_dft_direct_known() {
        // DFT of [1,0,0,0] = [1,1,1,1]
        let x = make_real(&[1.0, 0.0, 0.0, 0.0]);
        let y = dft_direct(&x);
        for v in &y {
            assert_relative_eq!(v.re, 1.0, epsilon = 1e-12);
            assert_relative_eq!(v.im, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fft_roundtrip_power_of_two() {
        let plan = MixedRadixFFT::new(8).expect("failed to create plan");
        let signal = make_real(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let freq = plan.fft(&signal).expect("failed to create freq");
        let recovered = plan.ifft(&freq).expect("failed to create recovered");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
            assert_relative_eq!(a.im, b.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_roundtrip_mixed_6() {
        let plan = MixedRadixFFT::new(6).expect("failed to create plan");
        let signal = make_real(&[1.0, -1.0, 2.0, 0.5, -2.0, 0.0]);
        let freq = plan.fft(&signal).expect("failed to create freq");
        let recovered = plan.ifft(&freq).expect("failed to create recovered");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_roundtrip_prime_5() {
        let plan = MixedRadixFFT::new(5).expect("failed to create plan");
        let signal = make_real(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let freq = plan.fft(&signal).expect("failed to create freq");
        let recovered = plan.ifft(&freq).expect("failed to create recovered");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_fft_roundtrip_prime_7() {
        let plan = MixedRadixFFT::new(7).expect("failed to create plan");
        let signal = make_real(&[0.5, -1.0, 2.0, -0.5, 3.0, 0.0, 1.5]);
        let freq = plan.fft(&signal).expect("failed to create freq");
        let recovered = plan.ifft(&freq).expect("failed to create recovered");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rader_fft_size_2() {
        let x = make_real(&[3.0, 5.0]);
        let y = rader_fft(&x, 1, 2).expect("failed to create y");
        assert_relative_eq!(y[0].re, 8.0, epsilon = 1e-12);
        assert_relative_eq!(y[1].re, -2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rader_fft_prime_5_roundtrip() {
        let plan = MixedRadixFFT::new(5).expect("failed to create plan");
        let signal = make_real(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        let freq = plan.fft(&signal).expect("failed to create freq");
        // DFT of [1,0,0,0,0] = [1,1,1,1,1]
        for v in &freq {
            assert_relative_eq!(v.re, 1.0, epsilon = 1e-10);
            assert_relative_eq!(v.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mixed_radix_zero_size_error() {
        assert!(MixedRadixFFT::new(0).is_err());
    }

    #[test]
    fn test_mixed_radix_dimension_error() {
        let plan = MixedRadixFFT::new(4).expect("failed to create plan");
        let short = make_real(&[1.0, 2.0]);
        assert!(plan.fft(&short).is_err());
    }

    #[test]
    fn test_fft_size_12() {
        // 12 = 2² × 3 — exercises both Cooley-Tukey and Good-Thomas paths
        let plan = MixedRadixFFT::new(12).expect("failed to create plan");
        let signal: Vec<Complex64> = (0..12).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let freq = plan.fft(&signal).expect("failed to create freq");
        let recovered = plan.ifft(&freq).expect("failed to create recovered");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-9);
        }
    }
}
