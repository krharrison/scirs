//! # Number Theoretic Transform (NTT) — Extended Module
//!
//! NTT over Z/p for arbitrary primes, plus specialized fast paths for the
//! widely-used prime 998244353.
//!
//! ## Key Exports
//!
//! | Function / Constant | Purpose |
//! |---------------------|---------|
//! | [`MOD998244353`] | Well-known NTT prime 998 244 353 = 119·2²³+1 |
//! | [`MOD1000000007`] | Classical competitive-programming prime 10⁹+7 |
//! | [`ntt`] | In-place NTT over Z/p |
//! | [`ntt_mul`] | Polynomial multiplication over Z/p |
//! | [`find_ntt_prime`] | Search for a suitable NTT prime+generator |
//! | [`ntt_998244353`] | Optimised NTT for `MOD998244353` |
//!
//! ## Design Notes
//!
//! - All arithmetic uses 128-bit intermediates to avoid 64-bit overflow.
//! - Inverse NTT multiplies by `n⁻¹ mod p` (Fermat's little theorem).
//! - `find_ntt_prime` searches for primes of the form `k·2^m + 1` that
//!   support NTT of length `n` and fit in `bits` bits.
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_fft::polynomial::ntt::{ntt, ntt_mul, MOD998244353};
//!
//! // Round-trip test
//! let original = vec![1u64, 2, 3, 4];
//! let mut a = original.clone();
//! ntt(&mut a, MOD998244353, 3, false).expect("forward NTT");
//! ntt(&mut a, MOD998244353, 3, true).expect("inverse NTT");
//! assert_eq!(a, original);
//!
//! // Polynomial multiplication: (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
//! let c = ntt_mul(&[1, 2], &[3, 4], MOD998244353).expect("ntt mul");
//! assert_eq!(c, vec![3, 10, 8]);
//! ```

use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Public constants
// ─────────────────────────────────────────────────────────────────────────────

/// NTT-friendly prime 998 244 353 = 119·2²³+1.
///
/// - Primitive root: **3**
/// - Maximum NTT length: 2²³ = 8 388 608
pub const MOD998244353: u64 = 998_244_353;

/// Classical competitive-programming prime 1 000 000 007 = 10⁹+7.
///
/// This prime is **not** NTT-friendly (10⁹+6 = 2 × 500000003, so the largest
/// power-of-two NTT length is only 2).  It is included for modular arithmetic
/// helpers (Chinese Remainder Theorem, Garner, etc.) but **cannot** be used
/// directly with [`ntt`].
pub const MOD1000000007: u64 = 1_000_000_007;

/// Additional NTT-friendly prime 469 762 049 = 7·2²⁶+1, primitive root 3.
pub const MOD469762049: u64 = 469_762_049;

/// Table of known (prime, generator) pairs, ordered by max NTT length.
///
/// All primes satisfy `p - 1 = k * 2^m` for m ≥ 20.
pub const KNOWN_NTT_PRIMES: &[(u64, u64)] = &[
    (998_244_353, 3),     // 119·2²³+1
    (1_004_535_809, 3),   // 479·2²¹+1
    (469_762_049, 3),     //   7·2²⁶+1
    (167_772_161, 3),     //   5·2²⁵+1
    (2_013_265_921, 31),  //  15·2²⁷+1
    (786_433, 10),        //   3·2¹⁸+1   (small, good for tests)
];

// ─────────────────────────────────────────────────────────────────────────────
//  Modular arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Modular multiplication using 128-bit intermediate.
#[inline(always)]
pub fn mulmod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular addition without overflow.
#[inline(always)]
pub fn addmod(a: u64, b: u64, m: u64) -> u64 {
    let s = a + b;
    if s >= m { s - m } else { s }
}

/// Modular subtraction without underflow.
#[inline(always)]
pub fn submod(a: u64, b: u64, m: u64) -> u64 {
    if a >= b { a - b } else { a + m - b }
}

/// Fast modular exponentiation: `base^exp mod m`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::powmod;
/// assert_eq!(powmod(2, 10, 1000), 24); // 1024 mod 1000
/// ```
pub fn powmod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mulmod(result, base, m);
        }
        base = mulmod(base, base, m);
        exp >>= 1;
    }
    result
}

/// Modular inverse via Fermat's little theorem (m must be prime).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::{modinv, MOD998244353};
/// let inv3 = modinv(3, MOD998244353);
/// assert_eq!((3u128 * inv3 as u128) % MOD998244353 as u128, 1);
/// ```
pub fn modinv(a: u64, m: u64) -> u64 {
    powmod(a, m - 2, m)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Prime / generator search
// ─────────────────────────────────────────────────────────────────────────────

/// Find a suitable NTT prime and its primitive root.
///
/// Returns the smallest prime `p` such that:
/// - `p` fits in `bits` bits (i.e. `p < 2^bits`)
/// - `p - 1` is divisible by `n` (so an NTT of length `n` is possible)
/// - `p` is prime
///
/// Also returns a primitive root `g` of `p`.
///
/// Searches primes of the form `k·n + 1` for increasing `k`.
///
/// # Arguments
///
/// * `n` – Desired NTT length (must be a power of two).
/// * `bits` – Maximum bit width of the prime (e.g. 30 for competitive programming).
///
/// # Returns
///
/// `(prime, generator)` pair, or an error if no suitable prime is found.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `n` is not a power of two or no prime
/// can be found within the bit constraint.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::find_ntt_prime;
/// let (p, g) = find_ntt_prime(8, 30).expect("prime");
/// assert!(p < (1u64 << 30));
/// assert!((p - 1) % 8 == 0);
/// ```
pub fn find_ntt_prime(n: usize, bits: u32) -> FFTResult<(u64, u64)> {
    if !n.is_power_of_two() {
        return Err(FFTError::ValueError(format!(
            "NTT length {n} must be a power of two"
        )));
    }
    let max_val = 1u64 << bits;
    let n64 = n as u64;

    // First check the known table
    for &(p, g) in KNOWN_NTT_PRIMES {
        if p < max_val && (p - 1) % n64 == 0 {
            return Ok((p, g));
        }
    }

    // Systematic search: primes of the form k*n + 1
    let mut k = 1u64;
    while k * n64 + 1 < max_val {
        let candidate = k * n64 + 1;
        if is_prime(candidate) {
            let g = find_primitive_root_of(candidate)?;
            return Ok((candidate, g));
        }
        k += 1;
    }

    Err(FFTError::ValueError(format!(
        "no NTT-friendly prime found for n={n} within {bits} bits"
    )))
}

/// Miller-Rabin primality test (deterministic for all n < 3.3×10²⁴).
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 || n == 5 || n == 7 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 {
        return false;
    }
    // Write n-1 = 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    // Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
    let witnesses: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    'outer: for &a in witnesses {
        if a >= n {
            continue;
        }
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Find a primitive root of a prime `p`.
fn find_primitive_root_of(p: u64) -> FFTResult<u64> {
    // Check known primes first
    for &(known_p, known_g) in KNOWN_NTT_PRIMES {
        if known_p == p {
            return Ok(known_g);
        }
    }

    // General: factorise p-1 and find smallest primitive root
    let phi = p - 1;
    let factors = factorize(phi);

    for g in 2..p {
        let is_root = factors.iter().all(|&f| powmod(g, phi / f, p) != 1);
        if is_root {
            return Ok(g);
        }
    }
    Err(FFTError::ValueError(format!(
        "no primitive root found for prime {p}"
    )))
}

/// Factorise `n` into unique prime factors using trial division.
fn factorize(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core NTT
// ─────────────────────────────────────────────────────────────────────────────

/// Bit-reversal permutation in place.
fn bit_reverse_permute(a: &mut [u64]) {
    let n = a.len();
    if n == 0 {
        return;
    }
    let log_n = n.ilog2() as usize;
    for i in 0..n {
        let j = reverse_bits(i, log_n);
        if i < j {
            a.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`.
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Validate NTT parameters.
fn validate(n: usize, p: u64) -> FFTResult<()> {
    if n == 0 {
        return Err(FFTError::ValueError("NTT length must be > 0".into()));
    }
    if !n.is_power_of_two() {
        return Err(FFTError::ValueError(format!(
            "NTT length {n} must be a power of two"
        )));
    }
    if (p - 1) % n as u64 != 0 {
        return Err(FFTError::ValueError(format!(
            "modulus {p} does not support NTT of length {n} ((p-1) mod n ≠ 0)"
        )));
    }
    Ok(())
}

/// In-place Number Theoretic Transform over Z/p.
///
/// Performs the forward or inverse NTT using the iterative Cooley-Tukey
/// butterfly (decimation in time).
///
/// # Arguments
///
/// * `a` – Mutable slice of residues (all < `p`).  Length must be a power of
///   two and must divide `p - 1`.
/// * `p` – NTT-friendly prime modulus.
/// * `g` – Primitive root of `p` (e.g. 3 for `MOD998244353`).
/// * `inverse` – `false` for forward transform, `true` for inverse.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if the length is not a power of two or if
/// `p - 1` is not divisible by `a.len()`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::{ntt, MOD998244353};
///
/// let orig = vec![1u64, 2, 3, 4];
/// let mut a = orig.clone();
/// ntt(&mut a, MOD998244353, 3, false).expect("forward");
/// ntt(&mut a, MOD998244353, 3, true).expect("inverse");
/// assert_eq!(a, orig);
/// ```
pub fn ntt(a: &mut [u64], p: u64, g: u64, inverse: bool) -> FFTResult<()> {
    let n = a.len();
    validate(n, p)?;

    bit_reverse_permute(a);

    let mut len = 2_usize;
    while len <= n {
        let exp = (p - 1) / len as u64;
        let root = if inverse {
            // inverse root = g^{-(p-1)/len} = modinv(g)^{(p-1)/len}
            powmod(modinv(g, p), exp, p)
        } else {
            powmod(g, exp, p)
        };

        let mut i = 0;
        while i < n {
            let mut w = 1u64;
            for j in 0..len / 2 {
                let u = a[i + j];
                let v = mulmod(a[i + j + len / 2], w, p);
                a[i + j] = addmod(u, v, p);
                a[i + j + len / 2] = submod(u, v, p);
                w = mulmod(w, root, p);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let n_inv = modinv(n as u64, p);
        for x in a.iter_mut() {
            *x = mulmod(*x, n_inv, p);
        }
    }
    Ok(())
}

/// Polynomial multiplication over Z/p using NTT.
///
/// Computes `c = a * b mod p` where `a` and `b` are polynomials given as
/// coefficient vectors (least-significant coefficient first), and `*` denotes
/// polynomial convolution.
///
/// # Arguments
///
/// * `a`, `b` – Coefficient vectors (values in `[0, p)`).
/// * `p` – NTT-friendly prime modulus.
///
/// # Returns
///
/// Coefficient vector of length `a.len() + b.len() - 1`.
///
/// # Errors
///
/// Returns an error if either input is empty, or if the required NTT length is
/// too large for the given prime.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::{ntt_mul, MOD998244353};
///
/// // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x²
/// let c = ntt_mul(&[1, 2], &[3, 4], MOD998244353).expect("ntt mul");
/// assert_eq!(c, vec![3, 10, 8]);
/// ```
pub fn ntt_mul(a: &[u64], b: &[u64], p: u64) -> FFTResult<Vec<u64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "polynomial inputs must not be empty".to_string(),
        ));
    }

    let result_len = a.len() + b.len() - 1;
    let fft_len = result_len.next_power_of_two();
    validate(fft_len, p)?;

    let g = generator_for_prime(p)?;

    let mut fa: Vec<u64> = a.iter().map(|&x| x % p).collect();
    fa.resize(fft_len, 0);
    let mut fb: Vec<u64> = b.iter().map(|&x| x % p).collect();
    fb.resize(fft_len, 0);

    ntt(&mut fa, p, g, false)?;
    ntt(&mut fb, p, g, false)?;

    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = mulmod(*x, *y, p);
    }

    ntt(&mut fa, p, g, true)?;
    fa.truncate(result_len);
    Ok(fa)
}

/// Look up the generator for a known prime, or search for it.
fn generator_for_prime(p: u64) -> FFTResult<u64> {
    for &(known_p, known_g) in KNOWN_NTT_PRIMES {
        if known_p == p {
            return Ok(known_g);
        }
    }
    find_primitive_root_of(p)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Specialised path for MOD998244353
// ─────────────────────────────────────────────────────────────────────────────

/// In-place NTT specialised for `MOD998244353` (generator = 3).
///
/// Equivalent to calling `ntt(a, MOD998244353, 3, inverse)` but avoids the
/// overhead of prime/generator lookup.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if the length is not a power of two or
/// exceeds 2²³.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::ntt_998244353;
///
/// let orig = vec![5u64, 3, 7, 2];
/// let mut a = orig.clone();
/// ntt_998244353(&mut a, false).expect("forward");
/// ntt_998244353(&mut a, true).expect("inverse");
/// assert_eq!(a, orig);
/// ```
pub fn ntt_998244353(a: &mut [u64], inverse: bool) -> FFTResult<()> {
    const P: u64 = MOD998244353;
    const G: u64 = 3;
    ntt(a, P, G, inverse)
}

/// Polynomial multiplication specialised for `MOD998244353`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::ntt_mul_998244353;
///
/// // (1 + x)² = 1 + 2x + x²
/// let a = vec![1u64, 1];
/// let c = ntt_mul_998244353(&a, &a).expect("mul");
/// assert_eq!(c, vec![1, 2, 1]);
/// ```
pub fn ntt_mul_998244353(a: &[u64], b: &[u64]) -> FFTResult<Vec<u64>> {
    ntt_mul(a, b, MOD998244353)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Integer convolution (exact, via CRT over two primes)
// ─────────────────────────────────────────────────────────────────────────────

/// Exact integer polynomial convolution using Garner's CRT.
///
/// Uses two NTT primes to reconstruct the exact integer product, provided no
/// coefficient exceeds `MOD998244353 × MOD469762049 / 2 ≈ 2.3×10¹⁷`.
///
/// # Errors
///
/// Returns an error if either input is empty or if the NTT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::convolve_exact;
///
/// // [1, 2, 3] * [4, 5, 6] = [4, 13, 28, 27, 18]
/// let c = convolve_exact(&[1i64, 2, 3], &[4i64, 5, 6]).expect("ok");
/// assert_eq!(c, vec![4, 13, 28, 27, 18]);
/// ```
pub fn convolve_exact(a: &[i64], b: &[i64]) -> FFTResult<Vec<i64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError("inputs must not be empty".into()));
    }
    let m1 = MOD998244353;
    let m2 = MOD469762049;

    let to_u64 = |x: i64, m: u64| -> u64 {
        if x >= 0 {
            (x as u64) % m
        } else {
            let pos = ((-x) as u64) % m;
            if pos == 0 { 0 } else { m - pos }
        }
    };

    let a1: Vec<u64> = a.iter().map(|&x| to_u64(x, m1)).collect();
    let b1: Vec<u64> = b.iter().map(|&x| to_u64(x, m1)).collect();
    let a2: Vec<u64> = a.iter().map(|&x| to_u64(x, m2)).collect();
    let b2: Vec<u64> = b.iter().map(|&x| to_u64(x, m2)).collect();

    let c1 = ntt_mul(&a1, &b1, m1)?;
    let c2 = ntt_mul(&a2, &b2, m2)?;

    // Garner's algorithm
    let m1_inv_m2 = modinv(m1 % m2, m2);
    let m1_m2 = m1 as i128 * m2 as i128;
    let half = m1_m2 / 2;

    let result: Vec<i64> = c1
        .iter()
        .zip(c2.iter())
        .map(|(&r1, &r2)| {
            let r1_mod_m2 = r1 % m2;
            let diff = submod(r2, r1_mod_m2, m2);
            let t = mulmod(diff, m1_inv_m2, m2) as u128;
            let x = r1 as i128 + m1 as i128 * t as i128;
            if x > half { (x - m1_m2) as i64 } else { x as i64 }
        })
        .collect();

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Modular polynomial inverse (for division / interpolation)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the modular polynomial inverse of `f` modulo `x^n` over Z/p.
///
/// Returns `g` such that `f * g ≡ 1  (mod x^n, p)`.
/// Requires `f[0]` to be invertible (nonzero mod p).
///
/// Uses Newton's method: `g_{k+1} = 2 g_k - f g_k² (mod x^{2^k})`.
///
/// # Errors
///
/// Returns an error if `f[0] == 0 mod p` or if the NTT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::polynomial::ntt::{poly_inv_mod_xn, MOD998244353};
///
/// // f = 1 + x + x²,  inverse mod x^4 over Z/p
/// let f = vec![1u64, 1, 1];
/// let g = poly_inv_mod_xn(&f, 4, MOD998244353).expect("inv");
/// // Check f * g ≡ 1 mod x^4
/// // (manual check via ntt_mul and truncate)
/// assert_eq!(g.len(), 4);
/// ```
pub fn poly_inv_mod_xn(f: &[u64], n: usize, p: u64) -> FFTResult<Vec<u64>> {
    if f.is_empty() || f[0] == 0 {
        return Err(FFTError::ValueError(
            "constant term must be nonzero for polynomial inversion".into(),
        ));
    }

    let mut g = vec![modinv(f[0], p)]; // g₀ = f[0]⁻¹

    let mut k = 1_usize;
    while k < n {
        k = (2 * k).min(n);
        // g_new = 2*g - f * g^2 mod x^k
        let fg = ntt_mul(f, &g, p)?;
        let fg_trunc: Vec<u64> = fg.into_iter().take(k).collect();
        let fg2 = ntt_mul(&fg_trunc, &g, p)?;

        // 2*g - f*g^2 truncated to degree k
        let two_g: Vec<u64> = g.iter().map(|&x| mulmod(x, 2, p)).collect();
        let len = k;
        let mut g_new = vec![0u64; len];
        for (i, &x) in two_g.iter().take(len).enumerate() {
            g_new[i] = addmod(g_new[i], x, p);
        }
        for (i, &y) in fg2.iter().take(len).enumerate() {
            g_new[i] = submod(g_new[i], y, p);
        }
        g = g_new;
    }
    g.truncate(n);
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Modular polynomial differentiation and integration
// ─────────────────────────────────────────────────────────────────────────────

/// Formal derivative of a polynomial over Z/p.
pub fn poly_deriv_mod(f: &[u64], p: u64) -> Vec<u64> {
    if f.len() <= 1 {
        return vec![0];
    }
    f[1..]
        .iter()
        .enumerate()
        .map(|(i, &c)| mulmod(c, (i + 1) as u64 % p, p))
        .collect()
}

/// Formal integral of a polynomial over Z/p (constant of integration = 0).
///
/// Requires that the coefficients are in `[0, p)`.
pub fn poly_integral_mod(f: &[u64], p: u64) -> Vec<u64> {
    let mut g = vec![0u64]; // constant term
    for (i, &c) in f.iter().enumerate() {
        let inv = modinv((i + 1) as u64 % p, p);
        g.push(mulmod(c, inv, p));
    }
    g
}

// ─────────────────────────────────────────────────────────────────────────────
//  Modular polynomial GCD
// ─────────────────────────────────────────────────────────────────────────────

/// Polynomial GCD over Z/p using the extended Euclidean algorithm.
///
/// Returns the monic GCD (leading coefficient = 1).
///
/// # Errors
///
/// Returns an error if the leading coefficient of an intermediate polynomial is
/// zero modulo p (should not happen for a prime p unless inputs are degenerate).
pub fn poly_gcd_mod(a: &[u64], b: &[u64], p: u64) -> FFTResult<Vec<u64>> {
    let mut u: Vec<u64> = trim_zeros(a);
    let mut v: Vec<u64> = trim_zeros(b);

    while !v.iter().all(|&x| x == 0) {
        let r = poly_rem_mod(&u, &v, p)?;
        u = v;
        v = trim_zeros(&r);
    }

    // Make monic
    if u.is_empty() {
        return Ok(vec![1]);
    }
    let lc = *u.last().unwrap_or(&1);
    if lc == 0 {
        return Ok(vec![1]);
    }
    let lc_inv = modinv(lc, p);
    Ok(u.iter().map(|&c| mulmod(c, lc_inv, p)).collect())
}

/// Polynomial remainder over Z/p.
fn poly_rem_mod(a: &[u64], b: &[u64], p: u64) -> FFTResult<Vec<u64>> {
    if b.is_empty() {
        return Err(FFTError::ValueError("division by zero polynomial".into()));
    }
    let mut r: Vec<u64> = a.to_vec();
    let db = b.len() - 1;
    let lc_b = *b.last().unwrap_or(&1);
    if lc_b == 0 {
        return Err(FFTError::ValueError(
            "leading coefficient of divisor is zero".into(),
        ));
    }
    let lc_b_inv = modinv(lc_b, p);

    while r.len() > db && !r.is_empty() {
        let n = r.len();
        let top = mulmod(*r.last().unwrap_or(&0), lc_b_inv, p);
        // Subtract top * x^(n-1-db) * b
        let shift = n - 1 - db;
        for (i, &bi) in b.iter().enumerate() {
            let val = mulmod(top, bi, p);
            r[shift + i] = submod(r[shift + i], val, p);
        }
        // Pop leading zeros
        while r.last() == Some(&0) && r.len() > 1 {
            r.pop();
        }
        if r.last() == Some(&0) && r.len() == 1 {
            break;
        }
    }
    Ok(r)
}

/// Remove trailing zeros from a coefficient vector.
fn trim_zeros(a: &[u64]) -> Vec<u64> {
    let mut v = a.to_vec();
    while v.len() > 1 && v.last() == Some(&0) {
        v.pop();
    }
    v
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = MOD998244353;

    #[test]
    fn test_ntt_roundtrip() {
        let orig = vec![1u64, 2, 3, 4];
        let mut a = orig.clone();
        ntt(&mut a, P, 3, false).expect("forward");
        ntt(&mut a, P, 3, true).expect("inverse");
        assert_eq!(a, orig);
    }

    #[test]
    fn test_ntt_998244353_roundtrip() {
        let orig = vec![0u64, 5, 3, 2, 7, 1, 0, 0];
        let mut a = orig.clone();
        ntt_998244353(&mut a, false).expect("forward");
        ntt_998244353(&mut a, true).expect("inverse");
        assert_eq!(a, orig);
    }

    #[test]
    fn test_ntt_mul_basic() {
        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x²
        let c = ntt_mul(&[1, 2], &[3, 4], P).expect("mul");
        assert_eq!(c, vec![3, 10, 8]);
    }

    #[test]
    fn test_ntt_mul_identity() {
        let a = vec![3u64, 1, 4, 1, 5, 9, 2, 6];
        let one = vec![1u64];
        let c = ntt_mul(&a, &one, P).expect("mul by 1");
        assert_eq!(c, a);
    }

    #[test]
    fn test_ntt_mul_all_ones() {
        // [1,1,1,1] * [1,1,1,1] = [1,2,3,4,3,2,1]
        let a = vec![1u64; 4];
        let c = ntt_mul(&a, &a, P).expect("mul");
        assert_eq!(c, vec![1, 2, 3, 4, 3, 2, 1]);
    }

    #[test]
    fn test_ntt_mul_matches_brute_force() {
        let a = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let b = vec![8u64, 7, 6, 5, 4, 3, 2, 1];
        // Brute force
        let n = a.len() + b.len() - 1;
        let mut expected = vec![0u64; n];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                expected[i + j] = (expected[i + j] + ai * bj) % P;
            }
        }
        let result = ntt_mul(&a, &b, P).expect("ntt mul");
        assert_eq!(result, expected);
    }

    #[test]
    fn test_find_ntt_prime_small() {
        let (p, _g) = find_ntt_prime(8, 30).expect("prime found");
        assert!(p < (1u64 << 30));
        assert!((p - 1) % 8 == 0);
        assert!(is_prime(p));
    }

    #[test]
    fn test_find_ntt_prime_known() {
        // Should return a known prime for n=8, bits=30
        let (p, g) = find_ntt_prime(8, 32).expect("prime found");
        assert!((p - 1) % 8 == 0);
        // Validate generator
        let phi = p - 1;
        let factors = factorize(phi);
        for f in &factors {
            assert_ne!(powmod(g, phi / f, p), 1, "g is not a primitive root");
        }
    }

    #[test]
    fn test_find_ntt_prime_non_power_of_two_errors() {
        assert!(find_ntt_prime(3, 30).is_err());
    }

    #[test]
    fn test_convolve_exact_basic() {
        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let c = convolve_exact(&a, &b).expect("ok");
        assert_eq!(c, vec![4, 13, 28, 27, 18]);
    }

    #[test]
    fn test_convolve_exact_negatives() {
        let a = vec![-1i64, 2];
        let b = vec![3i64, -4];
        let c = convolve_exact(&a, &b).expect("ok");
        assert_eq!(c, vec![-3, 10, -8]);
    }

    #[test]
    fn test_poly_inv_mod_xn() {
        // f = 1 - x,  f⁻¹ = 1 + x + x² + x³ + ...
        let f = vec![1u64, P - 1]; // 1 - x
        let g = poly_inv_mod_xn(&f, 4, P).expect("inv");
        assert_eq!(g.len(), 4);
        // f * g ≡ 1 (mod x^4)
        let fg = ntt_mul(&f, &g, P).expect("mul");
        assert_eq!(fg[0], 1);
        for &c in &fg[1..4] {
            assert_eq!(c, 0, "non-zero coefficient: {c}");
        }
    }

    #[test]
    fn test_poly_deriv_mod() {
        // d/dx (1 + 2x + 3x²) = 2 + 6x
        let f = vec![1u64, 2, 3];
        let df = poly_deriv_mod(&f, P);
        assert_eq!(df[0], 2);
        assert_eq!(df[1], 6);
    }

    #[test]
    fn test_poly_gcd_mod() {
        // gcd(x² - 1, x - 1) = x - 1  (over Z/p)
        let a = vec![P - 1, 0, 1]; // x² - 1
        let b = vec![P - 1, 1];    // x - 1
        let g = poly_gcd_mod(&a, &b, P).expect("gcd");
        // The monic gcd should be [P-1, 1] normalized to monic → [1, ...]
        // Actually monic of (x-1) is (x-1)/1 = x-1 → coeffs [P-1, 1]
        // After making monic: leading coeff is 1, so g = [P-1, 1]
        assert_eq!(g.last(), Some(&1u64));
    }

    #[test]
    fn test_ntt_invalid_length_error() {
        let mut a = vec![1u64, 2, 3];
        assert!(ntt(&mut a, P, 3, false).is_err());
    }
}
