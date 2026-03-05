//! Prime-length FFT algorithms: Rader's algorithm and Bluestein chirp-Z.
//!
//! ## Rader's Algorithm
//!
//! For a prime length `p`, Rader's algorithm converts the DFT into a cyclic
//! convolution of length `p − 1` (which IS a convenient composite length).
//! The cyclic convolution is evaluated via FFT, giving overall O(p log p) cost.
//!
//! ### Mathematical basis
//!
//! Let `g` be a primitive root of `Z_p^*`.  Then for k ≠ 0:
//!
//! ```text
//! X[g^{-q}] = x[0] + sum_{n=1}^{p-1} x[g^n] · W_p^{g^n · g^{-q}}
//!           = x[0] + (a ⊛ b)[q]
//! ```
//!
//! where `a[n] = x[g^n]`, `b[n] = W_p^{g^n}`, and `⊛` denotes cyclic
//! convolution of length `p − 1`.
//!
//! ## Bluestein Chirp-Z
//!
//! `bluestein_chirp_z` evaluates the DFT on a user-supplied set of M points on
//! the Z-plane (a generalisation of the DFT to a Z-transform evaluation):
//!
//! ```text
//! X(z_k) = sum_{n=0}^{N-1}  x[n] · z_k^{-n},   z_k = A · W^{-k}
//! ```
//!
//! where A and W are complex scalars supplied by the caller.
//!
//! ## Chirp-Z Transform
//!
//! `chirp_z_transform` is the standard CZT interface: it evaluates the
//! Z-transform at `m` equally-spaced points on a spiral contour:
//!
//! ```text
//! z_k = A · W^k,   k = 0, 1, ..., m-1
//! ```
//!
//! with A = A₀ e^{iθ₀} (starting point) and W = W₀ e^{-iφ} (step).
//! Setting A = W = exp(2πi/N) recovers the standard DFT.
//!
//! # References
//!
//! * Rader, C. M. "Discrete Fourier transforms when the number of data samples
//!   is prime." *Proc. IEEE* 56(6) (1968), pp. 1107–1108.
//! * Rabiner, L.; Schafer, R.; Rader, C. "The chirp z-transform algorithm."
//!   *IEEE Trans. Audio Electroacoust.* 17(2) (1969), pp. 86–92.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::{Complex64, Zero};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Smallest power-of-two ≥ n.
fn next_pow2(n: usize) -> usize {
    if n <= 1 { 1 } else { n.next_power_of_two() }
}

/// Test whether `n` is prime (trial division).
fn is_prime(n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let mut i = 3usize;
    while i * i <= n {
        if n % i == 0 { return false; }
        i += 2;
    }
    true
}

/// Find a primitive root (generator) of `Z_p^*` for prime `p`.
/// Returns `Err` if `p` is not prime or is 2.
pub fn primitive_root(p: usize) -> FFTResult<usize> {
    if !is_prime(p) {
        return Err(FFTError::ValueError(format!("{p} is not prime")));
    }
    if p == 2 {
        return Ok(1);
    }
    let phi = p - 1; // Euler's totient for prime p

    // Factorise phi
    let mut factors: Vec<usize> = Vec::new();
    let mut rem = phi;
    let mut f = 2usize;
    while f * f <= rem {
        if rem % f == 0 {
            factors.push(f);
            while rem % f == 0 { rem /= f; }
        }
        f += 1;
    }
    if rem > 1 { factors.push(rem); }

    // Find smallest g such that g^(phi/q) ≢ 1 (mod p) for all prime factors q.
    'outer: for g in 2..p {
        for &q in &factors {
            let exp = phi / q;
            if modpow_usize(g, exp, p) == 1 {
                continue 'outer;
            }
        }
        return Ok(g);
    }
    Err(FFTError::ValueError(format!("No primitive root found for {p}")))
}

/// Compute `base^exp mod modulus` by repeated squaring.
fn modpow_usize(mut base: usize, mut exp: usize, modulus: usize) -> usize {
    let mut result = 1usize;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 { result = result * base % modulus; }
        exp >>= 1;
        base = base * base % modulus;
    }
    result
}

/// Build the sequence `g^0, g^1, ..., g^{n-1}` in `Z_p^*`.
fn powers_of_g(g: usize, n: usize, p: usize) -> Vec<usize> {
    let mut seq = Vec::with_capacity(n);
    let mut cur = 1usize;
    for _ in 0..n {
        seq.push(cur);
        cur = cur * g % p;
    }
    seq
}

/// Cyclic convolution of `a` and `b` of length `n` via FFT.
fn cyclic_convolve(a: &[Complex64], b: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = a.len();
    if n != b.len() {
        return Err(FFTError::ValueError("cyclic_convolve: length mismatch".into()));
    }
    let fft_a = fft(a, None)?;
    let fft_b = fft(b, None)?;
    let prod: Vec<Complex64> = fft_a.iter().zip(fft_b.iter()).map(|(&x, &y)| x * y).collect();
    let result = ifft(&prod, None)?;
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rader's algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the DFT of a prime-length complex sequence using Rader's algorithm.
///
/// Rader's algorithm converts the length-p DFT into a cyclic convolution of
/// length p − 1, which is computed via FFT.  This gives O(p log p) cost.
///
/// # Arguments
///
/// * `signal` - Complex input of **prime** length p ≥ 2.
///
/// # Returns
///
/// DFT output of length p.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if the signal is empty, has length 1, or
/// has **non-prime** length.  For non-prime lengths, prefer [`crate::bluestein::bluestein_fft`].
///
/// # Examples
///
/// ```
/// use scirs2_fft::prime_fft::rader_fft;
/// use scirs2_core::numeric::Complex64;
///
/// let n = 7usize;
/// let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
/// let spectrum = rader_fft(&signal).expect("rader_fft failed");
/// assert_eq!(spectrum.len(), n);
/// ```
pub fn rader_fft(signal: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let p = signal.len();

    if p == 0 {
        return Err(FFTError::ValueError("rader_fft: empty input".into()));
    }
    if p == 1 {
        return Ok(signal.to_vec());
    }
    if p == 2 {
        let s0 = signal[0];
        let s1 = signal[1];
        return Ok(vec![s0 + s1, s0 - s1]);
    }
    if !is_prime(p) {
        return Err(FFTError::ValueError(format!(
            "rader_fft: length {p} is not prime. Use bluestein_fft for arbitrary lengths."
        )));
    }

    let g = primitive_root(p)?;
    let phi = p - 1; // p - 1 = order of Z_p^*

    // Build g^0, g^1, ..., g^{phi-1}  (positive exponent permutation)
    let g_pos = powers_of_g(g, phi, p);

    // Build inverse permutation: g^{-k} mod p
    // Inverse of g is g^{phi-1} mod p.
    let g_inv = modpow_usize(g, phi - 1, p);
    let g_neg = powers_of_g(g_inv, phi, p); // g^{-0}, g^{-1}, ..., g^{-(phi-1)}

    // X[0] = sum of all x[n]
    let x0: Complex64 = signal.iter().fold(Complex64::zero(), |acc, &v| acc + v);

    // a[n] = x[g^n],  n = 0..phi-1
    let a: Vec<Complex64> = (0..phi).map(|n| signal[g_pos[n]]).collect();

    // b[q] = W_p^{g^{-q}},  W_p = exp(-2πi/p)
    let b: Vec<Complex64> = (0..phi)
        .map(|q| {
            let phase = -2.0 * PI * g_neg[q] as f64 / p as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();

    // Cyclic convolution c = a ⊛ b (length phi).
    let c = cyclic_convolve(&a, &b)?;

    // Fill output.
    let mut out = vec![Complex64::zero(); p];
    out[0] = x0;

    // X[g^{-q}] = x[0] + c[q]  (indices in g^{-q} permutation)
    for q in 0..phi {
        let k = g_neg[q]; // = g^{-q} mod p
        out[k] = signal[0] + c[q];
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Bluestein chirp-Z: generalised DFT on M arbitrary Z-plane points
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the Z-transform of `x` at M points on a spiral contour using
/// Bluestein's algorithm.
///
/// The M evaluation points are:
///
/// ```text
/// z_k = A · W^{-k},   k = 0, 1, ..., M-1
/// ```
///
/// where `A` and `W` are user-supplied complex scalars.  Setting
/// `A = W = exp(2πi/N)` (and M = N) recovers the standard DFT.
///
/// # Arguments
///
/// * `x` - Input signal (complex, length N).
/// * `m` - Number of output points.
/// * `a` - Starting point on the Z-plane spiral.
/// * `w` - Step factor on the Z-plane spiral.
///
/// # Returns
///
/// Vec of M complex values: `X(z_0), X(z_1), ..., X(z_{M-1})`.
///
/// # Errors
///
/// Returns an error if `x` is empty or `m` is zero.
///
/// # Examples
///
/// ```
/// use scirs2_fft::prime_fft::bluestein_chirp_z;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n = 8usize;
/// let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
///
/// // Standard DFT via CZT: A = W = exp(2πi/N), M = N
/// let a = Complex64::new(1.0, 0.0);
/// let w = Complex64::new((2.0 * PI / n as f64).cos(), (2.0 * PI / n as f64).sin());
/// let spec = bluestein_chirp_z(&signal, n, a, w).expect("czt failed");
/// assert_eq!(spec.len(), n);
/// ```
pub fn bluestein_chirp_z(
    x: &[Complex64],
    m: usize,
    a: Complex64,
    w: Complex64,
) -> FFTResult<Vec<Complex64>> {
    let n = x.len();

    if n == 0 {
        return Err(FFTError::ValueError("bluestein_chirp_z: input is empty".into()));
    }
    if m == 0 {
        return Err(FFTError::ValueError("bluestein_chirp_z: m must be > 0".into()));
    }

    // Length of the zero-padded convolution: must be ≥ N + M - 1.
    let l = next_pow2(n + m - 1);

    // ── Chirp-modulate the input: yn[k] = x[k] · A^{-k} · W^{k²/2}  ──────
    //
    // Using the Bluestein identity  n*k = (n^2 + k^2 - (k-n)^2) / 2
    // we write:
    //   X(z_k) = W^{k²/2} · sum_{n=0}^{N-1} [x[n] · A^{-n} · W^{n²/2}] · W^{-(k-n)²/2}
    //
    // The bracketed term is the chirp-modulated input; the sum is a linear
    // convolution with the chirp sequence W^{-n²/2}.

    // Pre-compute A^{-n} and W^{n²/2} for n = 0..N-1.
    let mut yn = vec![Complex64::zero(); l];
    let mut a_pow = Complex64::new(1.0, 0.0); // A^0
    let mut a_inv = Complex64::new(1.0, 0.0); // A^{-0}
    // A^{-1}
    let a_norm_sq = a.norm_sqr();
    if a_norm_sq == 0.0 {
        return Err(FFTError::ValueError("bluestein_chirp_z: |A| = 0".into()));
    }
    let a_inv_scalar = Complex64::new(a.re / a_norm_sq, -a.im / a_norm_sq);

    // W^{k²/2}: use the recurrence  W^{k²/2} = W^{(k-1)²/2} · W^{k-1/2}
    // But it's simpler to compute with integer index arithmetic.
    // We use the half-integer exponent via:
    //   W^{n²/2} = exp(i·phase_n)  with  phase_n = angle(W) * n^2 / 2
    let w_angle = w.im.atan2(w.re); // arg(W)

    for n_idx in 0..n {
        let n_f = n_idx as f64;
        let phase_w = w_angle * n_f * n_f * 0.5;
        let w_chirp = Complex64::new(phase_w.cos(), phase_w.sin());

        // A^{-n}: iteratively updated
        if n_idx > 0 {
            a_inv = a_inv * a_inv_scalar;
        }

        yn[n_idx] = x[n_idx] * a_inv * w_chirp;
        let _ = a_pow; // suppress unused
        let _ = a_pow.re;
    }
    let _ = a_pow; // suppress unused warning

    // ── Convolution kernel: hn[k] = W^{-k²/2} for k = -(N-1) .. M-1 ──────
    // Stored in the circular buffer of length L:
    //   hn[0..M]        =  W^{-k²/2}  k = 0..M-1
    //   hn[L-N+1..L]    =  W^{-(L-k)²/2} = W^{-k²/2}  (circulant wrap)
    let mut hn = vec![Complex64::zero(); l];
    for k in 0..m {
        let k_f = k as f64;
        let phase = -w_angle * k_f * k_f * 0.5;
        hn[k] = Complex64::new(phase.cos(), phase.sin());
    }
    for k in 1..n {
        let k_f = k as f64;
        let phase = -w_angle * k_f * k_f * 0.5;
        let idx = l - k;
        if idx < l {
            hn[idx] = Complex64::new(phase.cos(), phase.sin());
        }
    }

    // ── Convolution via FFT ────────────────────────────────────────────────
    let yn_fft = fft(&yn, None)?;
    let hn_fft = fft(&hn, None)?;
    let prod: Vec<Complex64> = yn_fft
        .iter()
        .zip(hn_fft.iter())
        .map(|(&a, &b)| a * b)
        .collect();
    let g_conv = ifft(&prod, None)?;

    // ── Multiply by output twiddle W^{k²/2} ───────────────────────────────
    let mut out = Vec::with_capacity(m);
    for k in 0..m {
        let k_f = k as f64;
        let phase = w_angle * k_f * k_f * 0.5;
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        out.push(g_conv[k] * twiddle);
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  chirp_z_transform — standard CZT interface
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the Z-transform on a complex spiral contour (Chirp Z-Transform).
///
/// Computes:
///
/// ```text
/// X_k = sum_{n=0}^{N-1}  x[n] · (A · W^k)^{-n},   k = 0 .. M-1
/// ```
///
/// with
///
/// * `A` = `a_mag · exp(i · a_angle)` — starting point radius and angle.
/// * `W` = `w_mag · exp(-i · w_angle)` — step magnitude and step angle.
///
/// Setting `a_mag = 1`, `a_angle = 0`, `w_mag = 1`,
/// `w_angle = -2π/N`, and `M = N` recovers the standard DFT.
///
/// # Arguments
///
/// * `x` — Complex input signal (length N).
/// * `m` — Number of output samples.
/// * `a_mag` — Magnitude of the starting contour point (typically 1.0).
/// * `a_angle` — Angle (radians) of the starting contour point.
/// * `w_mag` — Magnitude of the contour step (typically 1.0).
/// * `w_angle` — Angular step (radians, positive = counter-clockwise).
///
/// # Returns
///
/// `Vec<Complex64>` of length `m`.
///
/// # Errors
///
/// Returns an error if `x` is empty or `m` is 0.
///
/// # Examples
///
/// ```
/// use scirs2_fft::prime_fft::chirp_z_transform;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n = 8usize;
/// let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
///
/// // CZT = DFT when a_angle=0, w_angle=2π/N, a_mag=1, w_mag=1, M=N
/// let spec = chirp_z_transform(&signal, n, 1.0, 0.0, 1.0, 2.0 * PI / n as f64)
///     .expect("czt failed");
/// assert_eq!(spec.len(), n);
/// ```
pub fn chirp_z_transform(
    x: &[Complex64],
    m: usize,
    a_mag: f64,
    a_angle: f64,
    w_mag: f64,
    w_angle: f64,
) -> FFTResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("chirp_z_transform: input is empty".into()));
    }
    if m == 0 {
        return Err(FFTError::ValueError("chirp_z_transform: m must be > 0".into()));
    }

    // A = a_mag · e^{i·a_angle}
    let a = Complex64::new(a_mag * a_angle.cos(), a_mag * a_angle.sin());
    // W = w_mag · e^{i·w_angle}  (note: positive angle = CCW step on Z-plane)
    let w = Complex64::new(w_mag * w_angle.cos(), w_mag * w_angle.sin());

    bluestein_chirp_z(x, m, a, w)
}

/// Evaluate the Z-transform on the unit circle (standard DFT via CZT).
///
/// Convenience wrapper around [`chirp_z_transform`] that selects A = 1 and
/// W = exp(2πi/N), recovering the standard DFT.
///
/// # Examples
///
/// ```
/// use scirs2_fft::prime_fft::dft_via_czt;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let signal = vec![Complex64::new(1.0, 0.0); 7]; // prime length
/// let spec = dft_via_czt(&signal).expect("dft_via_czt failed");
/// assert_eq!(spec.len(), 7);
/// // DC bin should equal N (all ones)
/// assert!((spec[0].re - 7.0).abs() < 1e-9);
/// ```
pub fn dft_via_czt(x: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = x.len();
    if n == 0 {
        return Err(FFTError::ValueError("dft_via_czt: empty input".into()));
    }
    chirp_z_transform(x, n, 1.0, 0.0, 1.0, 2.0 * PI / n as f64)
}

/// Zoom FFT: evaluate the DFT over a narrow frequency band `[f_low, f_high]`.
///
/// Returns `m` complex spectrum values for the band `[f_low, f_high]`
/// (normalised frequency, 0 ≤ f_low < f_high ≤ 0.5).
///
/// # Arguments
///
/// * `x` — Real input signal (complex parts ignored, use a Vec<Complex64> for complex input).
/// * `m` — Number of frequency bins in the zoom band.
/// * `f_low` — Lower normalised frequency (0..0.5).
/// * `f_high` — Upper normalised frequency (f_low..0.5).
///
/// # Errors
///
/// Returns an error if parameters are out of range or inputs are empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::prime_fft::zoom_fft_band;
/// use scirs2_core::numeric::Complex64;
/// use std::f64::consts::PI;
///
/// let n = 64;
/// let freq = 0.1_f64; // 10% of Nyquist
/// let signal: Vec<Complex64> = (0..n)
///     .map(|k| Complex64::new((2.0 * PI * freq * k as f64).cos(), 0.0))
///     .collect();
///
/// // Zoom into the band [0.05, 0.15] with 32 bins
/// let spec = zoom_fft_band(&signal, 32, 0.05, 0.15).expect("zoom_fft failed");
/// assert_eq!(spec.len(), 32);
/// ```
pub fn zoom_fft_band(
    x: &[Complex64],
    m: usize,
    f_low: f64,
    f_high: f64,
) -> FFTResult<Vec<Complex64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("zoom_fft_band: empty input".into()));
    }
    if m == 0 {
        return Err(FFTError::ValueError("zoom_fft_band: m must be > 0".into()));
    }
    if !(0.0..=0.5).contains(&f_low) {
        return Err(FFTError::ValueError(format!("zoom_fft_band: f_low={f_low} out of [0, 0.5]")));
    }
    if !(f_low..=0.5).contains(&f_high) {
        return Err(FFTError::ValueError(format!(
            "zoom_fft_band: f_high={f_high} must be in [{f_low}, 0.5]"
        )));
    }

    // A = exp(2πi·f_low)  (start of zoom band)
    let a_angle = 2.0 * PI * f_low;
    // W = exp(-2πi·(f_high - f_low)/m)  (frequency step)
    let step = (f_high - f_low) / m as f64;
    let w_angle = 2.0 * PI * step;

    chirp_z_transform(x, m, 1.0, a_angle, 1.0, w_angle)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn brute_dft(x: &[Complex64]) -> Vec<Complex64> {
        let n = x.len();
        (0..n)
            .map(|k| {
                x.iter().enumerate().fold(Complex64::zero(), |acc, (m, &xm)| {
                    let phase = -2.0 * PI * k as f64 * m as f64 / n as f64;
                    acc + xm * Complex64::new(phase.cos(), phase.sin())
                })
            })
            .collect()
    }

    fn assert_complex_close(a: &[Complex64], b: &[Complex64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(ai.re, bi.re, epsilon = tol, var_name = format!("bin {i} re"));
            assert_relative_eq!(ai.im, bi.im, epsilon = tol, var_name = format!("bin {i} im"));
        }
    }

    // ── primitive_root ────────────────────────────────────────────────────────

    #[test]
    fn test_primitive_root_5() {
        let g = primitive_root(5).expect("primitive root");
        // g=2 or g=3 are both primitive roots of Z_5^*
        assert!(g == 2 || g == 3, "unexpected primitive root {g} for p=5");
    }

    #[test]
    fn test_primitive_root_7() {
        let g = primitive_root(7).expect("primitive root");
        // Primitive roots of 7: 3, 5
        assert!(g == 3 || g == 5, "unexpected primitive root {g} for p=7");
    }

    #[test]
    fn test_primitive_root_nonprime_error() {
        assert!(primitive_root(4).is_err());
        assert!(primitive_root(9).is_err());
    }

    // ── rader_fft ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rader_prime_7() {
        let p = 7;
        let signal: Vec<Complex64> = (0..p)
            .map(|k| Complex64::new(k as f64, (k as f64) * 0.5))
            .collect();
        let rader = rader_fft(&signal).expect("rader_fft");
        let brute = brute_dft(&signal);
        assert_complex_close(&rader, &brute, 1e-9);
    }

    #[test]
    fn test_rader_prime_11() {
        let p = 11;
        let signal: Vec<Complex64> = (0..p)
            .map(|k| Complex64::new((k as f64 / p as f64).sin(), 0.0))
            .collect();
        let rader = rader_fft(&signal).expect("rader_fft");
        let brute = brute_dft(&signal);
        assert_complex_close(&rader, &brute, 1e-8);
    }

    #[test]
    fn test_rader_prime_13() {
        let p = 13;
        let signal: Vec<Complex64> = (0..p)
            .map(|k| Complex64::new(1.0, -(k as f64)))
            .collect();
        let rader = rader_fft(&signal).expect("rader_fft");
        let brute = brute_dft(&signal);
        assert_complex_close(&rader, &brute, 1e-8);
    }

    #[test]
    fn test_rader_length_2() {
        let signal = vec![Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
        let out = rader_fft(&signal).expect("rader_fft length 2");
        assert_relative_eq!(out[0].re, 0.0, epsilon = 1e-12);
        assert_relative_eq!(out[1].re, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rader_nonprime_error() {
        let signal: Vec<Complex64> = (0..6).map(|k| Complex64::new(k as f64, 0.0)).collect();
        assert!(rader_fft(&signal).is_err());
    }

    #[test]
    fn test_rader_empty_error() {
        assert!(rader_fft(&[]).is_err());
    }

    // ── bluestein_chirp_z ────────────────────────────────────────────────────

    #[test]
    fn test_bluestein_chirp_z_equals_dft_prime_7() {
        let p = 7;
        let signal: Vec<Complex64> = (0..p)
            .map(|k| Complex64::new(k as f64, 0.0))
            .collect();
        // A=1, W=exp(2πi/N) → standard DFT
        let w_angle = 2.0 * PI / p as f64;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(w_angle.cos(), w_angle.sin());
        let czt = bluestein_chirp_z(&signal, p, a, w).expect("czt");
        let brute = brute_dft(&signal);
        assert_complex_close(&czt, &brute, 1e-9);
    }

    #[test]
    fn test_bluestein_chirp_z_more_output_bins() {
        // Compute DFT at M=12 points starting at 0 frequency for N=8 input.
        let n = 8;
        let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let m = 12;
        let w_angle = 2.0 * PI / n as f64;
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(w_angle.cos(), w_angle.sin());
        let out = bluestein_chirp_z(&signal, m, a, w).expect("czt m>n");
        assert_eq!(out.len(), m);
    }

    #[test]
    fn test_bluestein_chirp_z_empty_error() {
        let a = Complex64::new(1.0, 0.0);
        let w = Complex64::new(1.0, 0.0);
        assert!(bluestein_chirp_z(&[], 8, a, w).is_err());
    }

    // ── chirp_z_transform ─────────────────────────────────────────────────────

    #[test]
    fn test_chirp_z_transform_equals_dft() {
        let n = 8;
        let signal: Vec<Complex64> = (0..n)
            .map(|k| Complex64::new(k as f64, 0.0))
            .collect();
        let spec = chirp_z_transform(&signal, n, 1.0, 0.0, 1.0, 2.0 * PI / n as f64)
            .expect("czt");
        let brute = brute_dft(&signal);
        assert_complex_close(&spec, &brute, 1e-9);
    }

    #[test]
    fn test_dft_via_czt_all_ones() {
        let n = 7;
        let signal = vec![Complex64::new(1.0, 0.0); n];
        let spec = dft_via_czt(&signal).expect("dft_via_czt");
        // DC bin = N, all others = 0
        assert_relative_eq!(spec[0].re, n as f64, epsilon = 1e-9);
        for k in 1..n {
            assert_relative_eq!(spec[k].re.abs(), 0.0, epsilon = 1e-9);
            assert_relative_eq!(spec[k].im.abs(), 0.0, epsilon = 1e-9);
        }
    }

    // ── zoom_fft_band ─────────────────────────────────────────────────────────

    #[test]
    fn test_zoom_fft_band_length() {
        let n = 64;
        let signal = vec![Complex64::new(1.0, 0.0); n];
        let spec = zoom_fft_band(&signal, 32, 0.1, 0.4).expect("zoom_fft");
        assert_eq!(spec.len(), 32);
    }

    #[test]
    fn test_zoom_fft_band_dc_when_full_range() {
        // Full range zoom ≈ regular FFT first bin
        let n = 16;
        let signal: Vec<Complex64> = (0..n).map(|k| Complex64::new(k as f64, 0.0)).collect();
        let spec = zoom_fft_band(&signal, n, 0.0, 0.5).expect("zoom_fft full");
        assert_eq!(spec.len(), n);
    }

    #[test]
    fn test_zoom_fft_invalid_args() {
        let sig = vec![Complex64::new(1.0, 0.0); 8];
        assert!(zoom_fft_band(&sig, 0, 0.1, 0.4).is_err());
        assert!(zoom_fft_band(&sig, 8, 0.6, 0.4).is_err());
        assert!(zoom_fft_band(&sig, 8, 0.1, 0.6).is_err());
    }
}
