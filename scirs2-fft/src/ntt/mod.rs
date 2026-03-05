//! Number Theoretic Transform (NTT) over finite fields Z_p.
//!
//! The NTT is the analogue of the FFT in a prime field Z_p.  All arithmetic is
//! exact integer arithmetic, so NTT-based convolution is free of floating-point
//! rounding errors.
//!
//! # NTT-friendly primes
//!
//! An NTT prime `p` must satisfy `p - 1 ≡ 0 (mod 2^k)` for a sufficiently
//! large `k` so that transforms of length up to `2^k` are possible.
//!
//! | Prime | Value | Primitive root | Max length |
//! |-------|-------|----------------|------------|
//! | `MOD_998244353` | 998 244 353 = 119·2²³+1 | 3 | 2²³ |
//! | `MOD_469762049` | 469 762 049 =   7·2²⁶+1 | 3 | 2²⁶ |
//! | `MOD_167772161` | 167 772 161 =   5·2²⁵+1 | 3 | 2²⁵ |
//!
//! # References
//!
//! * cp-algorithms.com/algebra/fft.html

use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Public constants
// ─────────────────────────────────────────────────────────────────────────────

/// NTT-friendly prime 998 244 353 = 119·2²³+1. Primitive root: 3.
pub const MOD_998244353: u64 = 998_244_353;
/// NTT-friendly prime 469 762 049 = 7·2²⁶+1. Primitive root: 3.
pub const MOD_469762049: u64 = 469_762_049;
/// NTT-friendly prime 167 772 161 = 5·2²⁵+1. Primitive root: 3.
pub const MOD_167772161: u64 = 167_772_161;

/// Primitive root for `MOD_998244353`.
pub const PRIM_ROOT_998244353: u64 = 3;
/// Primitive root for `MOD_469762049`.
pub const PRIM_ROOT_469762049: u64 = 3;
/// Primitive root for `MOD_167772161`.
pub const PRIM_ROOT_167772161: u64 = 3;

/// Configuration for an NTT transform.
#[derive(Debug, Clone, Copy)]
pub struct NTTConfig {
    /// The prime modulus p.
    pub modulus: u64,
    /// A primitive root g of Z_p^*.
    pub primitive_root: u64,
}

impl NTTConfig {
    /// Create a config from a known prime.
    pub const fn new(modulus: u64, primitive_root: u64) -> Self {
        Self { modulus, primitive_root }
    }
}

/// List of (prime, primitive_root) pairs suitable for NTT.
pub const NTT_PRIMES: &[(u64, u64)] = &[
    (998_244_353, 3),
    (469_762_049, 3),
    (167_772_161, 3),
    (754_974_721, 11), // 45·2²⁴+1
    (985_661_441, 3),  // 235·2²²+1
];

// ─────────────────────────────────────────────────────────────────────────────
//  Modular arithmetic helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Fast modular exponentiation: base^exp mod modulus.
///
/// Uses binary exponentiation in O(log exp) multiplications.
///
/// # Examples
/// ```
/// use scirs2_fft::ntt::mod_pow;
/// assert_eq!(mod_pow(2, 10, 1_000_000_007), 1024);
/// ```
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp >>= 1;
    }
    result
}

/// Modular inverse of `a` mod `p` via Fermat's little theorem (p must be prime).
///
/// Returns a^(p-2) mod p.
pub fn mod_inv(a: u64, p: u64) -> u64 {
    mod_pow(a, p - 2, p)
}

/// Multiplication avoiding 64-bit overflow via u128.
#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Bit-reversal permutation of length n = 2^log_n.
fn bit_reverse_permute(a: &mut Vec<u64>, log_n: u32) {
    let n = a.len();
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - log_n);
        let j = j as usize;
        if i < j {
            a.swap(i, j);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core NTT
// ─────────────────────────────────────────────────────────────────────────────

/// In-place iterative Cooley-Tukey NTT over Z_p.
///
/// `a` must have a length that is a power of two. The transform is done
/// in-place. When `invert` is true the inverse NTT is computed (result is
/// divided by n using a Fermat inverse so that `ntt(ntt(a, false), true) == a`).
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `len(a)` is not a power of two or
/// exceeds the maximum supported by the prime.
///
/// # Examples
///
/// ```
/// use scirs2_fft::ntt::{ntt, MOD_998244353, PRIM_ROOT_998244353};
/// let mut a = vec![1u64, 2, 3, 4];
/// ntt(&mut a, false, MOD_998244353, PRIM_ROOT_998244353).expect("valid input");
/// ntt(&mut a, true,  MOD_998244353, PRIM_ROOT_998244353).expect("valid input");
/// assert_eq!(a, vec![1, 2, 3, 4]);
/// ```
pub fn ntt(a: &mut Vec<u64>, invert: bool, modulus: u64, primitive_root: u64) -> FFTResult<()> {
    let n = a.len();
    if n == 0 {
        return Ok(());
    }
    if !n.is_power_of_two() {
        return Err(FFTError::ValueError(format!(
            "NTT length must be a power of two, got {n}"
        )));
    }
    let log_n = n.trailing_zeros();

    // Verify length is supported by this prime
    let max_log = (modulus - 1).trailing_zeros();
    if log_n > max_log {
        return Err(FFTError::ValueError(format!(
            "NTT length 2^{log_n} exceeds max supported by prime {modulus} (max 2^{max_log})"
        )));
    }

    bit_reverse_permute(a, log_n);

    let mut len = 2usize;
    while len <= n {
        // Compute w = primitive_root^((p-1)/len) mod p
        let w = if invert {
            mod_inv(
                mod_pow(primitive_root, (modulus - 1) / len as u64, modulus),
                modulus,
            )
        } else {
            mod_pow(primitive_root, (modulus - 1) / len as u64, modulus)
        };

        let half = len / 2;
        let mut j = 0usize;
        while j < n {
            let mut wn = 1u64;
            for k in 0..half {
                let u = a[j + k];
                let v = mul_mod(a[j + k + half], wn, modulus);
                a[j + k] = (u + v) % modulus;
                a[j + k + half] = (u + modulus - v) % modulus;
                wn = mul_mod(wn, w, modulus);
            }
            j += len;
        }
        len <<= 1;
    }

    if invert {
        let n_inv = mod_inv(n as u64, modulus);
        for x in a.iter_mut() {
            *x = mul_mod(*x, n_inv, modulus);
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polynomial multiplication mod p
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply two polynomials mod a prime p using NTT.
///
/// Computes the polynomial product `c = a * b` over Z_p. Coefficients of `a`
/// and `b` must all be in `[0, p)`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if either input is empty or the product
/// length would exceed the NTT limit for this prime.
///
/// # Examples
///
/// ```
/// use scirs2_fft::ntt::{ntt_multiply, MOD_998244353};
/// // [1,2,3] * [1,1] = [1,3,5,3]
/// let c = ntt_multiply(&[1,2,3], &[1,1], MOD_998244353).expect("valid input");
/// assert_eq!(c, vec![1, 3, 5, 3]);
/// ```
pub fn ntt_multiply(a: &[u64], b: &[u64], modulus: u64) -> FFTResult<Vec<u64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "ntt_multiply: inputs must be non-empty".to_string(),
        ));
    }
    // Find primitive root for this prime
    let primitive_root = find_primitive_root(modulus)?;
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut fa: Vec<u64> = a.iter().map(|&x| x % modulus).collect();
    let mut fb: Vec<u64> = b.iter().map(|&x| x % modulus).collect();
    fa.resize(n, 0);
    fb.resize(n, 0);

    ntt(&mut fa, false, modulus, primitive_root)?;
    ntt(&mut fb, false, modulus, primitive_root)?;

    for (x, y) in fa.iter_mut().zip(fb.iter()) {
        *x = mul_mod(*x, *y, modulus);
    }

    ntt(&mut fa, true, modulus, primitive_root)?;
    fa.truncate(result_len);
    Ok(fa)
}

/// Find a known primitive root for the given prime, or use 3 as a fallback.
fn find_primitive_root(modulus: u64) -> FFTResult<u64> {
    for &(p, g) in NTT_PRIMES {
        if p == modulus {
            return Ok(g);
        }
    }
    // Heuristic fallback: try 3 for unknown primes
    Ok(3)
}

// ─────────────────────────────────────────────────────────────────────────────
//  CRT-based arbitrary integer polynomial multiplication
// ─────────────────────────────────────────────────────────────────────────────

/// Multiply two integer-coefficient polynomials exactly using 3-prime NTT + CRT.
///
/// Unlike `ntt_multiply`, the inputs can have arbitrary `i64` coefficients
/// (possibly negative).  Three NTT-friendly primes p1, p2, p3 are used such
/// that p1·p2·p3 > n·max_coeff² to avoid overflow in CRT reconstruction.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if inputs are empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::ntt::ntt_multiply_arbitrary;
/// // [1,-1] * [1,1] = [1,0,-1]
/// let c = ntt_multiply_arbitrary(&[1,-1], &[1,1]).expect("valid input");
/// assert_eq!(c, vec![1, 0, -1]);
/// ```
pub fn ntt_multiply_arbitrary(a: &[i64], b: &[i64]) -> FFTResult<Vec<i64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "ntt_multiply_arbitrary: inputs must be non-empty".to_string(),
        ));
    }

    // Shift negative values to [0, p) by adding p
    let shift_pos = |x: i64, p: u64| -> u64 {
        if x < 0 {
            (x.rem_euclid(p as i64)) as u64
        } else {
            (x as u64) % p
        }
    };

    let p1 = MOD_998244353;
    let p2 = MOD_469762049;
    let p3 = MOD_167772161;

    let a1: Vec<u64> = a.iter().map(|&x| shift_pos(x, p1)).collect();
    let b1: Vec<u64> = b.iter().map(|&x| shift_pos(x, p1)).collect();
    let a2: Vec<u64> = a.iter().map(|&x| shift_pos(x, p2)).collect();
    let b2: Vec<u64> = b.iter().map(|&x| shift_pos(x, p2)).collect();
    let a3: Vec<u64> = a.iter().map(|&x| shift_pos(x, p3)).collect();
    let b3: Vec<u64> = b.iter().map(|&x| shift_pos(x, p3)).collect();

    let c1 = ntt_multiply(&a1, &b1, p1)?;
    let c2 = ntt_multiply(&a2, &b2, p2)?;
    let c3 = ntt_multiply(&a3, &b3, p3)?;

    // CRT reconstruction using 128-bit intermediates
    // M = p1 * p2 * p3
    // M1 = p2 * p3,  M2 = p1 * p3,  M3 = p1 * p2
    // y_i = M_i^{-1} mod p_i
    let inv12 = mod_inv(p2 % p3, p3);
    let inv21 = mod_inv(p1 % p3, p3);
    let _ = (inv12, inv21); // suppress unused warnings

    // Simpler Garner's algorithm for 3-prime CRT
    // r1 = c1[i]
    // r2 = (c2[i] - r1) * inv(p1, p2) mod p2
    // r3 = ((c3[i] - r1 - r2*p1) * inv(p1*p2, p3)) mod p3
    // result = r1 + r2*p1 + r3*p1*p2  (may be > M/2 → subtract M)
    let inv_p1_p2 = mod_inv(p1 % p2, p2);
    let p1_mod_p3 = p1 % p3;
    let p2_mod_p3 = p2 % p3;
    let p1p2_mod_p3 = mul_mod(p1_mod_p3, p2_mod_p3, p3);
    let inv_p1p2_p3 = mod_inv(p1p2_mod_p3, p3);

    let result_len = c1.len();
    let mut result = Vec::with_capacity(result_len);

    for i in 0..result_len {
        let r1 = c1[i] % p1;
        // r2 = (c2 - r1) / p1  mod p2
        let r2 = mul_mod((c2[i] + p2 - r1 % p2) % p2, inv_p1_p2, p2);
        // r3 = (c3 - r1 - r2*p1) / (p1*p2)  mod p3
        let r2p1_mod_p3 = mul_mod(r2 % p3, p1_mod_p3, p3);
        let tmp = (c3[i] + p3 * 2 - r1 % p3 - r2p1_mod_p3) % p3;
        let r3 = mul_mod(tmp, inv_p1p2_p3, p3);

        // Reconstruct: value = r1 + r2*p1 + r3*p1*p2
        // Use i128 for reconstruction to avoid overflow
        let p1i = p1 as i128;
        let p2i = p2 as i128;
        let val = r1 as i128 + r2 as i128 * p1i + r3 as i128 * p1i * p2i;

        // Center: if val > M/2, subtract M so result can be negative
        let m = p1i * p2i * p3 as i128;
        let centered = if val > m / 2 { val - m } else { val };
        result.push(centered as i64);
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow_basic() {
        assert_eq!(mod_pow(2, 10, 1_000_000_007), 1024);
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(3, 1, 7), 3);
        assert_eq!(mod_pow(0, 5, 7), 0);
    }

    #[test]
    fn test_mod_inv() {
        // 3 * modinv(3,7) ≡ 1 (mod 7)
        let inv = mod_inv(3, 7);
        assert_eq!((3 * inv) % 7, 1);
        // Fermat: a^(p-1) ≡ 1 (mod p)
        let inv_998 = mod_inv(7, MOD_998244353);
        assert_eq!(mul_mod(7, inv_998, MOD_998244353), 1);
    }

    #[test]
    fn test_ntt_roundtrip() {
        let original = vec![1u64, 2, 3, 4];
        let mut a = original.clone();
        ntt(&mut a, false, MOD_998244353, PRIM_ROOT_998244353).expect("unexpected None or Err");
        ntt(&mut a, true,  MOD_998244353, PRIM_ROOT_998244353).expect("unexpected None or Err");
        assert_eq!(a, original);
    }

    #[test]
    fn test_ntt_multiply_basic() {
        // [1,1] * [1,1] = [1,2,1]
        let c = ntt_multiply(&[1, 1], &[1, 1], MOD_998244353).expect("failed to create c");
        assert_eq!(c, vec![1, 2, 1]);
    }

    #[test]
    fn test_ntt_multiply_larger() {
        // [1,2,3] * [1,1] = [1,3,5,3]
        let c = ntt_multiply(&[1, 2, 3], &[1, 1], MOD_998244353).expect("failed to create c");
        assert_eq!(c, vec![1, 3, 5, 3]);
    }

    #[test]
    fn test_ntt_multiply_primes_table() {
        // Test with MOD_469762049
        let c = ntt_multiply(&[1, 2], &[3, 4], MOD_469762049).expect("failed to create c");
        assert_eq!(c, vec![3, 10, 8]);
    }

    #[test]
    fn test_ntt_multiply_arbitrary_basic() {
        // [1,-1] * [1,1] = [1,0,-1]
        let c = ntt_multiply_arbitrary(&[1, -1], &[1, 1]).expect("failed to create c");
        assert_eq!(c, vec![1, 0, -1]);
    }

    #[test]
    fn test_ntt_multiply_arbitrary_negatives() {
        // [-1,-2,-3] * [1] = [-1,-2,-3]
        let c = ntt_multiply_arbitrary(&[-1, -2, -3], &[1]).expect("failed to create c");
        assert_eq!(c, vec![-1, -2, -3]);
    }

    #[test]
    fn test_ntt_not_power_of_two() {
        let mut a = vec![1u64, 2, 3];
        assert!(ntt(&mut a, false, MOD_998244353, PRIM_ROOT_998244353).is_err());
    }

    #[test]
    fn test_ntt_config() {
        let cfg = NTTConfig::new(MOD_998244353, PRIM_ROOT_998244353);
        assert_eq!(cfg.modulus, MOD_998244353);
        assert_eq!(cfg.primitive_root, PRIM_ROOT_998244353);
    }

    #[test]
    fn test_ntt_primes_list() {
        assert!(!NTT_PRIMES.is_empty());
        for &(p, _g) in NTT_PRIMES {
            // p must be odd prime > 2
            assert!(p > 2);
            assert!(p % 2 == 1);
        }
    }
}
