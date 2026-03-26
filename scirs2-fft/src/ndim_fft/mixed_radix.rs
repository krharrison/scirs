//! Mixed-radix FFT engine.
//!
//! Implements Cooley-Tukey DIT with mixed radices {2, 3, 4, 5, 7, 8}
//! and Bluestein's algorithm for arbitrary (prime) sizes.
//!
//! The core strategy:
//! - Factor N into small primes, preferring larger factors first (4 > 2 > 3 > 5 > 7).
//! - For primes > 7 (or when no good factorization exists), use Bluestein's CZT.
//! - The iterative mixed-radix DIT FFT applies butterfly stages in-order after
//!   a digit-reversal permutation derived from the factorization.

use std::f64::consts::PI;

/// Type alias for a radix-4 butterfly result (4 complex tuples).
pub type Butterfly4Result = ((f64, f64), (f64, f64), (f64, f64), (f64, f64));

// ─────────────────────────────────────────────────────────────────────────────
// Inline complex arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

#[inline(always)]
fn cadd(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

#[inline(always)]
fn csub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

#[inline(always)]
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline(always)]
fn cscale(a: (f64, f64), s: f64) -> (f64, f64) {
    (a.0 * s, a.1 * s)
}

/// Multiply by -i (rotate 90° clockwise in negative direction).
#[inline(always)]
fn cmul_neg_i(a: (f64, f64)) -> (f64, f64) {
    (a.1, -a.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterfly operations
// ─────────────────────────────────────────────────────────────────────────────

/// Radix-2 DIT butterfly.
///
/// Computes:
///   a' = a + w·b
///   b' = a - w·b
#[inline(always)]
pub fn butterfly_2(a: (f64, f64), b: (f64, f64), w: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    let wb = cmul(w, b);
    (cadd(a, wb), csub(a, wb))
}

/// Radix-4 DIT butterfly.
///
/// Computes the length-4 DFT of (a, b, c, d) with twiddle factors (w1, w2, w3).
/// Requires 3 complex multiplications (vs 6 for two radix-2 stages).
#[allow(clippy::many_single_char_names)]
#[inline(always)]
pub fn butterfly_4(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    d: (f64, f64),
    w1: (f64, f64),
    w2: (f64, f64),
    w3: (f64, f64),
) -> Butterfly4Result {
    let wb1 = cmul(w1, b);
    let wb2 = cmul(w2, c);
    let wb3 = cmul(w3, d);

    let t0 = cadd(a, wb2); // a + w2·c
    let t1 = csub(a, wb2); // a - w2·c
    let t2 = cadd(wb1, wb3); // w1·b + w3·d
    let t3 = cmul_neg_i(csub(wb1, wb3)); // -i·(w1·b - w3·d)

    (cadd(t0, t2), cadd(t1, t3), csub(t0, t2), csub(t1, t3))
}

// ─────────────────────────────────────────────────────────────────────────────
// Factorization
// ─────────────────────────────────────────────────────────────────────────────

/// Factor `n` into small primes preferred by the mixed-radix engine.
///
/// Preference order: 4 first (fewer multiplies), then 2, then 3, 5, 7.
/// If a large prime > 7 remains, it is appended as-is (Bluestein handles it).
///
/// # Examples
///
/// ```
/// use scirs2_fft::ndim_fft::mixed_radix::factorize;
/// let f8 = factorize(8);
/// assert_eq!(f8.iter().product::<usize>(), 8);
/// let f12 = factorize(12);
/// assert_eq!(f12.iter().product::<usize>(), 12);
/// assert!(f12.contains(&3));
/// ```
pub fn factorize(mut n: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while n % 4 == 0 {
        factors.push(4);
        n /= 4;
    }
    if n % 2 == 0 {
        factors.push(2);
        n /= 2;
    }
    for &p in &[3usize, 5, 7] {
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
// Twiddle factor precomputation
// ─────────────────────────────────────────────────────────────────────────────

/// Precompute all forward twiddle factors `e^{-2πi k/n}` for `k = 0..n-1`.
///
/// Uses a recurrence for numerical stability rather than calling `sin`/`cos`
/// for every element.
pub fn compute_twiddles(n: usize) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    let mut table = Vec::with_capacity(n);
    let theta = -2.0 * PI / n as f64;
    let (sin_t, cos_t) = theta.sin_cos();
    let w1 = (cos_t, sin_t);
    let mut w = (1.0_f64, 0.0_f64);
    for _ in 0..n {
        table.push(w);
        w = cmul(w, w1);
    }
    table
}

/// Precompute inverse twiddle factors `e^{+2πi k/n}` for `k = 0..n-1`.
pub fn compute_twiddles_inv(n: usize) -> Vec<(f64, f64)> {
    if n == 0 {
        return Vec::new();
    }
    let mut table = Vec::with_capacity(n);
    let theta = 2.0 * PI / n as f64;
    let (sin_t, cos_t) = theta.sin_cos();
    let w1 = (cos_t, sin_t);
    let mut w = (1.0_f64, 0.0_f64);
    for _ in 0..n {
        table.push(w);
        w = cmul(w, w1);
    }
    table
}

// ─────────────────────────────────────────────────────────────────────────────
// Pure radix-2 FFT (power-of-2 only)
// ─────────────────────────────────────────────────────────────────────────────

/// Iterative Cooley-Tukey radix-2 DIT FFT.
///
/// `data` must have power-of-2 length.  Used internally for Bluestein convolution
/// and also as the fallback for power-of-2 input sizes.
pub fn fft_1d_pow2(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    debug_assert!(
        n.is_power_of_two(),
        "fft_1d_pow2 requires power-of-2 length"
    );

    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![data[0]];
    }

    // Bit-reversal permutation
    let mut output = data.to_vec();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let rev = bit_reverse(i, bits);
        if rev > i {
            output.swap(i, rev);
        }
    }

    // Cooley-Tukey butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let theta = -PI / half as f64;
        let w_step = (theta.cos(), theta.sin());
        let mut j = 0;
        while j < n {
            let mut w = (1.0f64, 0.0f64);
            for k in 0..half {
                let u = output[j + k];
                let v = cmul(w, output[j + k + half]);
                output[j + k] = cadd(u, v);
                output[j + k + half] = csub(u, v);
                w = cmul(w, w_step);
            }
            j += len;
        }
        len <<= 1;
    }
    output
}

/// Inverse radix-2 FFT (unnormalized — caller must divide by N).
pub fn ifft_1d_pow2_raw(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    // IDFT via conjugate trick: IDFT(x) = conj(FFT(conj(x)))
    let conj: Vec<(f64, f64)> = data.iter().map(|&(re, im)| (re, -im)).collect();
    let fft_out = fft_1d_pow2(&conj);
    fft_out.into_iter().map(|(re, im)| (re, -im)).collect()
}

#[inline(always)]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Bluestein's algorithm for arbitrary N
// ─────────────────────────────────────────────────────────────────────────────

/// Check whether `n` requires Bluestein (has a prime factor > 7).
fn needs_bluestein(n: usize) -> bool {
    let factors = factorize(n);
    factors.iter().any(|&f| f > 7)
}

/// Next power of 2 ≥ n.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n.next_power_of_two()
    }
}

/// Bluestein chirp Z-transform for arbitrary-length FFT.
///
/// Computes `X[k] = Sum_n x[n] * e^{-2*pi*i*n*k / N}` using the identity
/// nk = n²/2 + k²/2 - (k-n)²/2, converting the sum to a convolution.
pub fn bluestein_fft(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n <= 1 {
        return input.to_vec();
    }

    // Chirp sequence: chirp[k] = e^{-πi k²/N}
    let chirp: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let theta = -PI * (k * k) as f64 / n as f64;
            let (s, c) = theta.sin_cos();
            (c, s)
        })
        .collect();

    let m = next_pow2(2 * n - 1);

    // a[k] = x[k] * chirp[k]
    let mut a_padded = vec![(0.0f64, 0.0f64); m];
    for k in 0..n {
        a_padded[k] = cmul(input[k], chirp[k]);
    }

    // b[k] = conj(chirp[k]) for k = 0..n-1
    // b[m-k] = conj(chirp[k]) for k = 1..n-1
    let mut b_padded = vec![(0.0f64, 0.0f64); m];
    for k in 0..n {
        b_padded[k] = (chirp[k].0, -chirp[k].1); // conj(chirp[k])
    }
    for k in 1..n {
        b_padded[m - k] = b_padded[k];
    }

    // Convolve a * b via power-of-2 FFT
    let a_fft = fft_1d_pow2(&a_padded);
    let b_fft = fft_1d_pow2(&b_padded);
    let prod_fft: Vec<(f64, f64)> = a_fft
        .iter()
        .zip(b_fft.iter())
        .map(|(&a, &b)| cmul(a, b))
        .collect();

    // IFFT of product (unnormalized)
    let conv_raw = ifft_1d_pow2_raw(&prod_fft);
    let scale = 1.0 / m as f64;
    let conv: Vec<(f64, f64)> = conv_raw.into_iter().map(|x| cscale(x, scale)).collect();

    // Post-multiply by chirp and take first N elements
    conv[..n]
        .iter()
        .enumerate()
        .map(|(k, &c)| cmul(c, chirp[k]))
        .collect()
}

/// Bluestein inverse FFT (raw, unnormalized — caller divides by N).
pub fn bluestein_ifft_raw(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    // IDFT(x) = conj(DFT(conj(x)))
    let conj_in: Vec<(f64, f64)> = input.iter().map(|&(re, im)| (re, -im)).collect();
    let fft_out = bluestein_fft(&conj_in);
    fft_out.into_iter().map(|(re, im)| (re, -im)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Mixed-radix FFT (Cooley-Tukey, Gentleman-Sande decomposition)
// ─────────────────────────────────────────────────────────────────────────────
// We use a Stockham-style auto-sort algorithm which avoids the digit-reversal
// permutation entirely.  The algorithm processes the factors left-to-right,
// and each stage reads from `src` and writes to `dst` in natural order.

/// Recursive mixed-radix FFT.
///
/// Uses Cooley-Tukey decomposition: split N = radix × m, compute `radix`
/// sub-transforms of length m, then combine with twiddle factors.
///
/// `sign = -1.0` for forward FFT, `+1.0` for inverse (unnormalized).
fn mixed_radix_rec(data: &[(f64, f64)], sign: f64) -> Vec<(f64, f64)> {
    let n = data.len();
    if n <= 1 {
        return data.to_vec();
    }
    if n.is_power_of_two() {
        if sign < 0.0 {
            return fft_1d_pow2(data);
        } else {
            return ifft_1d_pow2_raw(data);
        }
    }

    // Choose the first factor as the outermost radix
    let factors = factorize(n);
    let radix = factors[0];
    let m = n / radix; // sub-transform length

    // Split input into `radix` interleaved sub-sequences of length m
    // sub[j][k] = data[k * radix + j]  for j=0..radix, k=0..m
    let sub_ffts: Vec<Vec<(f64, f64)>> = (0..radix)
        .map(|j| {
            let sub_seq: Vec<(f64, f64)> = (0..m).map(|k| data[k * radix + j]).collect();
            mixed_radix_rec(&sub_seq, sign)
        })
        .collect();

    // Combine: X[k + m*j'] = Σ_j  W_N^{j*k} * sub_fft[j][k % m]
    // where W_N = e^{sign * 2πi/N}
    let mut out = vec![(0.0f64, 0.0f64); n];
    for k in 0..m {
        for j_out in 0..radix {
            let out_idx = k + m * j_out;
            let mut sum = (0.0f64, 0.0f64);
            for j in 0..radix {
                // Twiddle: W_N^{j * out_idx} = e^{sign * 2πi * j * out_idx / N}
                let exp = j * out_idx;
                let theta = sign * 2.0 * PI * exp as f64 / n as f64;
                let (sin_t, cos_t) = theta.sin_cos();
                let w = (cos_t, sin_t);
                sum = cadd(sum, cmul(w, sub_ffts[j][k]));
            }
            out[out_idx] = sum;
        }
    }
    out
}

/// Small DFT of a scratch buffer of length `r` (r ∈ {2, 3, 4, 5, 7, ...}).
///
/// The `sign` parameter controls the sign of the exponent
/// (`-1.0` = forward, `+1.0` = inverse).
fn small_dft(scratch: &[(f64, f64)], sign: f64) -> Vec<(f64, f64)> {
    let r = scratch.len();
    match r {
        1 => vec![scratch[0]],
        2 => {
            let a = scratch[0];
            let b = scratch[1];
            vec![cadd(a, b), csub(a, b)]
        }
        3 => {
            // Radix-3 DFT
            let w = {
                let theta = sign * 2.0 * PI / 3.0;
                let (s, c) = theta.sin_cos();
                (c, s)
            };
            let w2 = cmul(w, w);
            let x0 = scratch[0];
            let x1 = scratch[1];
            let x2 = scratch[2];
            vec![
                cadd(x0, cadd(x1, x2)),
                cadd(x0, cadd(cmul(w, x1), cmul(w2, x2))),
                cadd(x0, cadd(cmul(w2, x1), cmul(cmul(w2, w), x2))),
            ]
        }
        4 => {
            // Radix-4 DFT: 3 multiplications
            let x0 = scratch[0];
            let x1 = scratch[1];
            let x2 = scratch[2];
            let x3 = scratch[3];

            let t0 = cadd(x0, x2);
            let t1 = csub(x0, x2);
            let t2 = cadd(x1, x3);
            // For forward (sign=-1): multiply x1-x3 by -i = (im, -re)
            // For inverse (sign=+1): multiply x1-x3 by +i = (-im, re)
            let t3_raw = csub(x1, x3);
            let t3 = if sign < 0.0 {
                cmul_neg_i(t3_raw)
            } else {
                (-t3_raw.1, t3_raw.0) // multiply by +i
            };

            vec![cadd(t0, t2), cadd(t1, t3), csub(t0, t2), csub(t1, t3)]
        }
        _ => {
            // Generic DFT: O(r²)
            (0..r)
                .map(|k| {
                    scratch
                        .iter()
                        .enumerate()
                        .fold((0.0f64, 0.0f64), |acc, (j, &x)| {
                            let theta = sign * 2.0 * PI * (k * j) as f64 / r as f64;
                            let (s, c) = theta.sin_cos();
                            cadd(acc, cmul(x, (c, s)))
                        })
                })
                .collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public 1-D FFT API
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 1-D forward FFT (unnormalized).
///
/// Automatically selects Bluestein for sizes with large prime factors,
/// radix-2 iterative FFT for power-of-2 sizes, and recursive mixed-radix
/// Cooley-Tukey for composite sizes.
pub fn fft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![input[0]];
    }

    if needs_bluestein(n) {
        return bluestein_fft(input);
    }

    if n.is_power_of_two() {
        return fft_1d_pow2(input);
    }

    mixed_radix_rec(input, -1.0)
}

/// Compute the 1-D inverse FFT (raw, unnormalized).
///
/// The caller is responsible for dividing by N.
pub fn ifft_1d_raw(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![input[0]];
    }

    if needs_bluestein(n) {
        return bluestein_ifft_raw(input);
    }

    if n.is_power_of_two() {
        return ifft_1d_pow2_raw(input);
    }

    mixed_radix_rec(input, 1.0)
}

/// Compute the 1-D inverse FFT (normalized by 1/N).
pub fn ifft_1d(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = input.len();
    let raw = ifft_1d_raw(input);
    let scale = 1.0 / n as f64;
    raw.into_iter().map(|x| cscale(x, scale)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn naive_dft(input: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .fold((0.0f64, 0.0f64), |acc, (j, &x)| {
                        let theta = -2.0 * PI * k as f64 * j as f64 / n as f64;
                        let (s, c) = theta.sin_cos();
                        cadd(acc, cmul(x, (c, s)))
                    })
            })
            .collect()
    }

    #[test]
    fn test_radix2_power_of_2() {
        let input: Vec<(f64, f64)> = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
            (7.0, 0.0),
            (8.0, 0.0),
        ];
        let fft_out = fft_1d(&input);
        let ref_out = naive_dft(&input);
        assert_eq!(fft_out.len(), 8);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-9);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_radix4_size_16() {
        let input: Vec<(f64, f64)> = (0..16).map(|i| (i as f64, 0.0)).collect();
        let fft_out = fft_1d(&input);
        let ref_out = naive_dft(&input);
        assert_eq!(fft_out.len(), 16);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-8);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_mixed_radix_size_12() {
        // 12 = 4 × 3
        let input: Vec<(f64, f64)> = (0..12).map(|i| (i as f64, 0.0)).collect();
        let fft_out = fft_1d(&input);
        let ref_out = naive_dft(&input);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-8);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_mixed_radix_size_30() {
        // 30 = 2 × 3 × 5
        let input: Vec<(f64, f64)> = (0..30).map(|i| (i as f64, 0.0)).collect();
        let fft_out = fft_1d(&input);
        let ref_out = naive_dft(&input);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-7);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_bluestein_prime_size_7() {
        let input: Vec<(f64, f64)> = (0..7).map(|i| (i as f64 + 1.0, 0.0)).collect();
        let fft_out = bluestein_fft(&input);
        let ref_out = naive_dft(&input);
        assert_eq!(fft_out.len(), 7);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-8);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_bluestein_prime_size_11() {
        let input: Vec<(f64, f64)> = (0..11).map(|i| ((i + 1) as f64, 0.0)).collect();
        let fft_out = bluestein_fft(&input);
        let ref_out = naive_dft(&input);
        assert_eq!(fft_out.len(), 11);
        for (a, b) in fft_out.iter().zip(ref_out.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-7);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_fft_1d_roundtrip() {
        let input: Vec<(f64, f64)> = (0..32).map(|i| (i as f64 * 0.1, 0.0)).collect();
        let freq = fft_1d(&input);
        let recovered = ifft_1d(&freq);
        for (a, b) in input.iter().zip(recovered.iter()) {
            assert_relative_eq!(a.0, b.0, epsilon = 1e-10);
            assert_relative_eq!(a.1, b.1, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_twiddle_precomputation() {
        let n = 8;
        let twiddles = compute_twiddles(n);
        for (k, &(re, im)) in twiddles.iter().enumerate() {
            let theta = -2.0 * PI * k as f64 / n as f64;
            assert_relative_eq!(re, theta.cos(), epsilon = 1e-14);
            assert_relative_eq!(im, theta.sin(), epsilon = 1e-14);
        }
    }

    #[test]
    fn test_factorize_powers_of_2() {
        let f = factorize(8);
        // 8 = 4 × 2
        assert_eq!(f.iter().product::<usize>(), 8);
        for &x in &f {
            assert!(matches!(x, 2 | 3 | 4 | 5 | 7));
        }
    }

    #[test]
    fn test_factorize_composite() {
        let f = factorize(12);
        assert_eq!(f.iter().product::<usize>(), 12);
        assert!(f.contains(&3));
    }
}
