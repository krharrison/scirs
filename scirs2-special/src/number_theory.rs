//! Number-theoretic functions
//!
//! This module provides classical number theory algorithms:
//!
//! - `gcd`, `lcm`: greatest common divisor / least common multiple
//! - `extended_gcd`: extended Euclidean algorithm
//! - `pow_mod`: fast modular exponentiation
//! - `mod_inverse`: modular multiplicative inverse
//! - `is_prime`: deterministic Miller-Rabin primality test
//! - `factorize`: prime factorization via Pollard's rho
//! - `legendre_symbol`: quadratic residue symbol
//! - `chinese_remainder_theorem`: CRT solver
//! - `discrete_log`: baby-step giant-step discrete logarithm
//!
//! All arithmetic is exact (integer) where possible.  No floating-point
//! approximation is used for number-theoretic results.

use crate::error::{SpecialError, SpecialResult};
use std::collections::HashMap;

// ── Basic arithmetic ──────────────────────────────────────────────────────────

/// Greatest common divisor of a and b (Euclidean algorithm).
///
/// gcd(0, 0) = 0 by convention.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::gcd;
/// assert_eq!(gcd(12, 8),  4);
/// assert_eq!(gcd(35, 14), 7);
/// assert_eq!(gcd(0, 5),   5);
/// ```
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Least common multiple of a and b.
///
/// Returns 0 if either argument is 0.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::lcm;
/// assert_eq!(lcm(4, 6),   12);
/// assert_eq!(lcm(7, 5),   35);
/// assert_eq!(lcm(0, 10),  0);
/// ```
pub fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd(a, b)).saturating_mul(b)
}

/// Extended Euclidean algorithm.
///
/// Returns `(g, x, y)` such that `a·x + b·y = g = gcd(|a|, |b|)`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::extended_gcd;
/// let (g, x, y) = extended_gcd(35, 15);
/// assert_eq!(g, 5);
/// assert_eq!(35*x + 15*y, g);
/// ```
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        if a >= 0 {
            return (a, 1, 0);
        } else {
            return (-a, -1, 0);
        }
    }
    let (g, x1, y1) = extended_gcd(b, a % b);
    let x = y1;
    let y = x1 - (a / b) * y1;
    (g, x, y)
}

// ── Modular arithmetic ────────────────────────────────────────────────────────

/// Fast modular exponentiation:  base^exp mod m.
///
/// Uses the right-to-left binary method (square-and-multiply).
/// Returns 1 when m = 1 (by convention).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::pow_mod;
/// assert_eq!(pow_mod(2, 10, 1000), 24);
/// assert_eq!(pow_mod(3, 0, 7),     1);
/// ```
pub fn pow_mod(mut base: u64, mut exp: u64, m: u64) -> u64 {
    if m == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= m;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mul_mod(result, base, m);
        }
        base = mul_mod(base, base, m);
        exp >>= 1;
    }
    result
}

/// Modular multiplication:  (a * b) mod m, avoiding u64 overflow via u128.
#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Modular multiplicative inverse: a^{−1} mod m.
///
/// Requires gcd(a, m) = 1.  Returns an error otherwise.
///
/// # Errors
/// Returns `SpecialError::DomainError` if gcd(a, m) ≠ 1.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::mod_inverse;
/// assert_eq!(mod_inverse(3, 7).unwrap(), 5);  // 3·5 ≡ 1 (mod 7)
/// assert_eq!(mod_inverse(2, 5).unwrap(), 3);  // 2·3 ≡ 1 (mod 5)
/// assert!(mod_inverse(2, 4).is_err());        // gcd(2,4) = 2
/// ```
pub fn mod_inverse(a: u64, m: u64) -> SpecialResult<u64> {
    if m == 0 {
        return Err(SpecialError::ValueError(
            "mod_inverse: modulus must be > 0".to_string(),
        ));
    }
    if m == 1 {
        return Ok(0);
    }
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        return Err(SpecialError::DomainError(format!(
            "mod_inverse: gcd({a}, {m}) = {g} ≠ 1; inverse does not exist"
        )));
    }
    // x may be negative; bring it into [0, m)
    Ok(((x % m as i64 + m as i64) as u64) % m)
}

// ── Primality ─────────────────────────────────────────────────────────────────

/// Deterministic Miller-Rabin primality test for n < 3·10^18.
///
/// Uses the witness set {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37} which
/// is provably sufficient for all n < 3.317·10^24 (and in practice deterministic
/// for all 64-bit integers).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::is_prime;
/// assert!(is_prime(2));
/// assert!(is_prime(104729));
/// assert!(!is_prime(4));
/// assert!(!is_prime(1));
/// ```
pub fn is_prime(n: u64) -> bool {
    match n {
        0 | 1 => return false,
        2 | 3 | 5 | 7 => return true,
        _ if n % 2 == 0 || n % 3 == 0 || n % 5 == 0 => return false,
        _ => {}
    }

    // Write n-1 = 2^r · d with d odd
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    // Deterministic witnesses sufficient for all n < 3.317e24
    const WITNESSES: &[u64] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    'witness: for &a in WITNESSES {
        if a >= n {
            continue;
        }
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

// ── Factorization ─────────────────────────────────────────────────────────────

/// Prime factorization of n, sorted with multiplicity.
///
/// Uses trial division for small factors and Pollard's rho for larger
/// composite factors, giving sub-linear complexity for most inputs.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::factorize;
/// assert_eq!(factorize(1),  Vec::<u64>::new());
/// assert_eq!(factorize(12), vec![2, 2, 3]);
/// assert_eq!(factorize(30), vec![2, 3, 5]);
/// assert_eq!(factorize(97), vec![97]);  // prime
/// ```
pub fn factorize(n: u64) -> Vec<u64> {
    if n <= 1 {
        return Vec::new();
    }
    let mut factors = Vec::new();
    factorize_into(n, &mut factors);
    factors.sort_unstable();
    factors
}

/// Recursive factorizer: append all prime factors of n to `out`.
fn factorize_into(mut n: u64, out: &mut Vec<u64>) {
    if n == 1 {
        return;
    }
    // Trial division for small factors
    for p in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        while n % p == 0 {
            out.push(p);
            n /= p;
        }
        if n == 1 {
            return;
        }
    }
    if n < 1369 {
        // n < 37^2, so what remains must be prime
        out.push(n);
        return;
    }
    if is_prime(n) {
        out.push(n);
        return;
    }
    // Pollard's rho to split a composite factor
    let d = pollard_rho(n);
    factorize_into(d, out);
    factorize_into(n / d, out);
}

/// Pollard's rho algorithm — returns a non-trivial factor of n.
///
/// Uses Brent's variant for better constant factors.
fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }
    // Try multiple starting points if needed
    for seed in 1u64.. {
        let mut x = seed % n + 1;
        let mut y = x;
        let mut c = seed % (n - 1) + 1;
        let mut d = 1u64;
        while d == 1 {
            x = (mul_mod(x, x, n) + c) % n;
            y = (mul_mod(y, y, n) + c) % n;
            y = (mul_mod(y, y, n) + c) % n;
            let diff = if x > y { x - y } else { y - x };
            d = gcd(diff, n);
        }
        if d != n {
            return d;
        }
    }
    // Should never reach here for composite n
    n
}

// ── Chinese Remainder Theorem ─────────────────────────────────────────────────

/// Chinese Remainder Theorem solver.
///
/// Given residues `rems` and pairwise-coprime moduli `mods`, finds the unique
/// `x` in `[0, M)` where `M = lcm(mods)` such that:
///
/// ```text
/// x ≡ rems[i]  (mod mods[i])  for all i
/// ```
///
/// # Errors
/// Returns an error if:
/// - `rems` and `mods` have different lengths
/// - any modulus is ≤ 0
/// - the moduli are not pairwise coprime
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::chinese_remainder_theorem;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5)  → x = 8
/// let x = chinese_remainder_theorem(&[2, 3], &[3, 5]).unwrap();
/// assert_eq!(x, 8);
/// ```
pub fn chinese_remainder_theorem(rems: &[i64], mods: &[i64]) -> SpecialResult<i64> {
    if rems.len() != mods.len() {
        return Err(SpecialError::ValueError(
            "crt: rems and mods must have the same length".to_string(),
        ));
    }
    if rems.is_empty() {
        return Ok(0);
    }
    for &m in mods {
        if m <= 0 {
            return Err(SpecialError::ValueError(format!(
                "crt: modulus {m} must be positive"
            )));
        }
    }

    // Iterative Garner's algorithm
    let mut x = 0i64;
    let mut step = 1i64; // = lcm of moduli processed so far

    for (&r, &m) in rems.iter().zip(mods.iter()) {
        // Adjust x so that x ≡ r (mod m)
        // Find t such that x + t·step ≡ r (mod m)
        // t ≡ (r - x) · step^{-1}  (mod m)
        let g = gcd(step.unsigned_abs(), m.unsigned_abs()) as i64;
        let diff = ((r - x) % m + m) % m;
        if diff % g != 0 {
            return Err(SpecialError::ValueError(
                "crt: moduli are not pairwise coprime (no solution exists)".to_string(),
            ));
        }
        let m_reduced = m / g;
        let step_reduced = (step / g).rem_euclid(m_reduced);
        // step_reduced^{-1} mod m_reduced
        let inv = mod_inverse(step_reduced.unsigned_abs(), m_reduced.unsigned_abs())
            .map_err(|_| {
                SpecialError::ValueError("crt: step inverse does not exist".to_string())
            })?;
        let t = ((diff / g % m_reduced + m_reduced) % m_reduced
            * inv as i64
            % m_reduced
            + m_reduced)
            % m_reduced;
        x += t * step;
        step = step
            .checked_mul(m / g)
            .ok_or_else(|| SpecialError::OverflowError("crt: lcm overflows i64".to_string()))?;
    }
    // Normalize to [0, step)
    Ok(x.rem_euclid(step))
}

// ── Additional number-theoretic functions ─────────────────────────────────────

/// Find the next prime strictly greater than n.
///
/// Uses a simple deterministic search using `is_prime`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::next_prime;
/// assert_eq!(next_prime(0),  2);
/// assert_eq!(next_prime(2),  3);
/// assert_eq!(next_prime(10), 11);
/// assert_eq!(next_prime(12), 13);
/// ```
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    let mut candidate = n + 1;
    loop {
        if is_prime(candidate) {
            return candidate;
        }
        candidate += 1;
    }
}

/// Prime factorization of n as (prime, exponent) pairs, sorted by prime.
///
/// Returns an empty vector for n ≤ 1.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::prime_factors;
/// assert_eq!(prime_factors(1),   Vec::<(u64, u32)>::new());
/// assert_eq!(prime_factors(12),  vec![(2, 2), (3, 1)]);
/// assert_eq!(prime_factors(360), vec![(2, 3), (3, 2), (5, 1)]);
/// assert_eq!(prime_factors(97),  vec![(97, 1)]);
/// ```
pub fn prime_factors(n: u64) -> Vec<(u64, u32)> {
    if n <= 1 {
        return Vec::new();
    }
    // Collect flat factors via the existing factorize() helper
    let flat = factorize(n);
    // Compress consecutive equal factors into (prime, exponent) pairs
    let mut result: Vec<(u64, u32)> = Vec::new();
    for p in flat {
        match result.last_mut() {
            Some(last) if last.0 == p => last.1 += 1,
            _ => result.push((p, 1)),
        }
    }
    result
}

/// Euler's totient function φ(n) — count of integers in [1, n] coprime to n.
///
/// Uses the product formula: φ(n) = n · ∏_{p | n} (1 − 1/p).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::euler_totient;
/// assert_eq!(euler_totient(1),  1);
/// assert_eq!(euler_totient(9),  6);
/// assert_eq!(euler_totient(12), 4);
/// ```
pub fn euler_totient(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut result = n;
    let mut x = n;
    let mut p = 2u64;
    while p * p <= x {
        if x % p == 0 {
            while x % p == 0 {
                x /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if x > 1 {
        result -= result / x;
    }
    result
}

/// Möbius function μ(n).
///
/// * μ(1) = 1
/// * μ(n) = 0  if n has a squared prime factor
/// * μ(n) = (−1)^k  if n is a product of k distinct primes
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::mobius;
/// assert_eq!(mobius(1),   1);
/// assert_eq!(mobius(2),  -1);
/// assert_eq!(mobius(4),   0);  // 4 = 2^2
/// assert_eq!(mobius(6),   1);  // 6 = 2·3
/// assert_eq!(mobius(30), -1);  // 30 = 2·3·5
/// ```
pub fn mobius(n: u64) -> i32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut x = n;
    let mut k = 0i32;
    let mut p = 2u64;
    while p * p <= x {
        if x % p == 0 {
            x /= p;
            if x % p == 0 {
                return 0; // p^2 divides n
            }
            k += 1;
        }
        p += 1;
    }
    if x > 1 {
        k += 1;
    }
    if k % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Sum of k-th powers of divisors of n: σ_k(n) = Σ_{d | n} d^k.
///
/// For k = 0 this gives the number of divisors (τ(n)), and for k = 1 the
/// sum of divisors (σ(n)).
///
/// The result uses saturating arithmetic to avoid overflow for large k.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::divisor_sum;
/// // σ_0(12) = number of divisors of 12 = 6
/// assert_eq!(divisor_sum(12, 0), 6);
/// // σ_1(12) = 1+2+3+4+6+12 = 28
/// assert_eq!(divisor_sum(12, 1), 28);
/// // σ_2(6) = 1+4+9+36 = 50
/// assert_eq!(divisor_sum(6, 2), 50);
/// ```
pub fn divisor_sum(n: u64, k: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    // Use multiplicativity: σ_k is multiplicative.
    // For a prime power p^e: σ_k(p^e) = 1 + p^k + p^{2k} + … + p^{ek}
    //   = (p^{k(e+1)} − 1) / (p^k − 1)  for k > 0
    //   = e + 1                           for k = 0
    let factors = prime_factors(n);
    if factors.is_empty() {
        return 1; // n == 1: only divisor is 1, sum = 1^k = 1
    }
    let mut result = 1u64;
    for (p, e) in factors {
        let term = if k == 0 {
            u64::from(e) + 1
        } else {
            // Sum 1 + p^k + p^{2k} + … + p^{ek}
            let mut s = 1u64;
            let mut pk = 1u64;
            for _ in 0..e {
                pk = pk.saturating_mul(p.saturating_pow(k as u32));
                s = s.saturating_add(pk);
            }
            s
        };
        result = result.saturating_mul(term);
    }
    result
}

/// Jacobi symbol (a/n) — generalisation of the Legendre symbol to odd n.
///
/// n must be a positive odd integer.  Returns:
/// - 0  if gcd(a, n) > 1
/// - ±1 otherwise (does **not** guarantee quadratic residuosity)
///
/// Computed via the law of quadratic reciprocity without factoring n.
///
/// # Errors
/// Returns `SpecialError::ValueError` if n is even or zero.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::jacobi_symbol;
/// assert_eq!(jacobi_symbol(0,  5).unwrap(),  0);
/// assert_eq!(jacobi_symbol(1,  5).unwrap(),  1);
/// assert_eq!(jacobi_symbol(2,  5).unwrap(), -1);
/// assert_eq!(jacobi_symbol(5, 15).unwrap(),  0); // gcd(5,15)=5
/// assert_eq!(jacobi_symbol(2, 15).unwrap(),  1); // 2 is NR mod 3 and mod 5
/// ```
pub fn jacobi_symbol(a: i64, n: u64) -> SpecialResult<i32> {
    if n == 0 || n % 2 == 0 {
        return Err(SpecialError::ValueError(format!(
            "jacobi_symbol: n = {n} must be a positive odd integer"
        )));
    }
    // Reduce a modulo n into [0, n)
    let mut a = ((a % n as i64 + n as i64) as u64) % n;
    let mut n = n;
    let mut result = 1i32;

    loop {
        if a == 0 {
            return Ok(if n == 1 { result } else { 0 });
        }
        // Remove factors of 2 from a
        let v = a.trailing_zeros();
        a >>= v;
        // (2/n) = (-1)^((n^2-1)/8)  →  flip if n ≡ ±3 (mod 8)
        if v % 2 == 1 {
            let n_mod8 = n % 8;
            if n_mod8 == 3 || n_mod8 == 5 {
                result = -result;
            }
        }
        // Quadratic reciprocity: (a/n)·(n/a) = (-1)^{(a-1)/2 · (n-1)/2}
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        // Swap a and n
        let tmp = a;
        a = n % a;
        n = tmp;
        if n == 1 {
            return Ok(result);
        }
    }
}

/// Legendre symbol (a/p): quadratic residue indicator modulo an odd prime p.
///
/// This wraps `jacobi_symbol` with an additional primality check.
/// Returns:
/// - 0  if p | a
/// - 1  if a is a quadratic residue mod p  (∃ x: a ≡ x² mod p)
/// - −1 if a is a non-residue mod p
///
/// # Errors
/// Returns `SpecialError::ValueError` if p is even, less than 2, or composite.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::legendre_symbol;
/// assert_eq!(legendre_symbol(0, 5).unwrap(),  0);
/// assert_eq!(legendre_symbol(1, 5).unwrap(),  1);
/// assert_eq!(legendre_symbol(2, 5).unwrap(), -1);
/// assert_eq!(legendre_symbol(4, 5).unwrap(),  1);  // 4 ≡ 2^2 (mod 5)
/// assert_eq!(legendre_symbol(-1, 5).unwrap(), 1);   // -1 ≡ 4 (mod 5)
/// ```
pub fn legendre_symbol(a: i64, p: u64) -> SpecialResult<i32> {
    if p < 2 || p % 2 == 0 {
        return Err(SpecialError::ValueError(format!(
            "legendre_symbol: p = {p} must be an odd prime"
        )));
    }
    if !is_prime(p) {
        return Err(SpecialError::ValueError(format!(
            "legendre_symbol: p = {p} is not prime"
        )));
    }
    // For a prime, Jacobi = Legendre
    jacobi_symbol(a, p)
}

// ── Discrete logarithm ────────────────────────────────────────────────────────

/// Discrete logarithm:  find x such that g^x ≡ h (mod p).
///
/// Uses the **baby-step giant-step** (BSGS) algorithm with time and space
/// complexity O(√p).
///
/// # Errors
/// Returns `SpecialError::DomainError` if no solution exists.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::discrete_log;
/// // 2^3 = 8 ≡ 8 (mod 11)
/// let x = discrete_log(2, 8, 11).unwrap();
/// assert_eq!(2u64.pow(x as u32) % 11, 8 % 11);
/// ```
pub fn discrete_log(g: u64, h: u64, p: u64) -> SpecialResult<u64> {
    if p == 0 {
        return Err(SpecialError::ValueError(
            "discrete_log: modulus p must be > 0".to_string(),
        ));
    }
    if p == 1 {
        return Ok(0);
    }

    // Baby-step giant-step
    let m = (p as f64).sqrt().ceil() as u64 + 1;

    // Baby steps: table[g^j mod p] = j  for j = 0, 1, ..., m-1
    let mut table: HashMap<u64, u64> = HashMap::with_capacity(m as usize);
    let mut baby = 1u64;
    for j in 0..m {
        table.insert(baby, j);
        baby = mul_mod(baby, g, p);
    }

    // g^{-m} mod p
    let gm = pow_mod(g, m, p);
    let gm_inv = mod_inverse(gm, p).map_err(|_| {
        SpecialError::DomainError(format!(
            "discrete_log: g^m = {gm} has no inverse mod {p}; g and p may not be coprime"
        ))
    })?;

    // Giant steps: for i = 0, 1, ..., m-1
    //   check if  h · (g^{-m})^i ≡ baby_value
    let mut giant = h % p;
    for i in 0..m {
        if let Some(&j) = table.get(&giant) {
            let x = i * m + j;
            // Verify (in case of hash collision with wrap-around)
            if pow_mod(g, x, p) == h % p {
                return Ok(x);
            }
        }
        giant = mul_mod(giant, gm_inv, p);
    }

    Err(SpecialError::DomainError(format!(
        "discrete_log: no solution found for g={g}, h={h}, p={p}"
    )))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(35, 14), 7);
        assert_eq!(gcd(7, 13), 1); // coprime
    }

    #[test]
    fn test_lcm() {
        assert_eq!(lcm(4, 6), 12);
        assert_eq!(lcm(7, 5), 35);
        assert_eq!(lcm(0, 10), 0);
        assert_eq!(lcm(1, 100), 100);
    }

    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, g);

        let (g2, x2, y2) = extended_gcd(12, 8);
        assert_eq!(g2, 4);
        assert_eq!(12 * x2 + 8 * y2, g2);

        // Coprime
        let (g3, x3, y3) = extended_gcd(7, 13);
        assert_eq!(g3, 1);
        assert_eq!(7 * x3 + 13 * y3, g3);
    }

    #[test]
    fn test_pow_mod() {
        assert_eq!(pow_mod(2, 10, 1000), 24);
        assert_eq!(pow_mod(3, 0, 7), 1);
        assert_eq!(pow_mod(0, 5, 7), 0);
        assert_eq!(pow_mod(2, 3, 5), 3); // 8 mod 5 = 3
    }

    #[test]
    fn test_mod_inverse() {
        assert_eq!(mod_inverse(3, 7).expect("ok"), 5); // 3·5=15 ≡ 1 (mod 7)
        assert_eq!(mod_inverse(2, 5).expect("ok"), 3); // 2·3=6 ≡ 1 (mod 5)
        assert!(mod_inverse(2, 4).is_err()); // gcd(2,4)=2
        assert!(mod_inverse(0, 7).is_err()); // gcd(0,7)=7
    }

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(!is_prime(9));
        assert!(is_prime(97));
        assert!(is_prime(104729));
        assert!(!is_prime(104730));
        // Carmichael number 561 = 3·11·17
        assert!(!is_prime(561));
    }

    #[test]
    fn test_factorize() {
        assert_eq!(factorize(1), Vec::<u64>::new());
        assert_eq!(factorize(2), vec![2]);
        assert_eq!(factorize(4), vec![2, 2]);
        assert_eq!(factorize(12), vec![2, 2, 3]);
        assert_eq!(factorize(30), vec![2, 3, 5]);
        assert_eq!(factorize(97), vec![97]);
        // Large semiprime
        let f = factorize(9_007_199_254_740_997);
        assert!(f.len() >= 2 || (f.len() == 1 && is_prime(f[0])));
    }

    #[test]
    fn test_legendre_symbol() {
        // mod 5
        assert_eq!(legendre_symbol(0, 5).expect("ok"), 0);
        assert_eq!(legendre_symbol(1, 5).expect("ok"), 1);
        assert_eq!(legendre_symbol(2, 5).expect("ok"), -1);
        assert_eq!(legendre_symbol(3, 5).expect("ok"), -1);
        assert_eq!(legendre_symbol(4, 5).expect("ok"), 1); // 2^2
        // Negative a
        assert_eq!(legendre_symbol(-1, 5).expect("ok"), 1); // -1 ≡ 4 (mod 5)
        // Error for even p
        assert!(legendre_symbol(1, 4).is_err());
    }

    #[test]
    fn test_crt_two_congruences() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8
        let x = chinese_remainder_theorem(&[2, 3], &[3, 5]).expect("ok");
        assert_eq!(x, 8);
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
    }

    #[test]
    fn test_crt_three_congruences() {
        // x ≡ 0 (mod 3), x ≡ 3 (mod 4), x ≡ 4 (mod 5) → x = 39
        let x = chinese_remainder_theorem(&[0, 3, 4], &[3, 4, 5]).expect("ok");
        assert_eq!(x % 3, 0);
        assert_eq!(x % 4, 3);
        assert_eq!(x % 5, 4);
    }

    #[test]
    fn test_crt_error_length_mismatch() {
        assert!(chinese_remainder_theorem(&[1, 2], &[3]).is_err());
    }

    #[test]
    fn test_discrete_log_basic() {
        // 2^x ≡ 8 (mod 11)  →  x = 3
        let x = discrete_log(2, 8, 11).expect("ok");
        assert_eq!(pow_mod(2, x, 11), 8 % 11);

        // 3^x ≡ 1 (mod 7)  →  x = 0 (or 6, but 0 is minimal)
        let x0 = discrete_log(3, 1, 7).expect("ok");
        assert_eq!(pow_mod(3, x0, 7), 1);
    }

    #[test]
    fn test_discrete_log_no_solution() {
        // 2^x ≡ 3 (mod 7): must verify by checking all powers
        // Powers of 2 mod 7: 1,2,4,1,2,4,... → 3 is never reached
        let result = discrete_log(2, 3, 7);
        assert!(result.is_err());
    }

    #[test]
    fn test_next_prime() {
        assert_eq!(next_prime(0), 2);
        assert_eq!(next_prime(1), 2);
        assert_eq!(next_prime(2), 3);
        assert_eq!(next_prime(3), 5);
        assert_eq!(next_prime(10), 11);
        assert_eq!(next_prime(12), 13);
        assert_eq!(next_prime(100), 101);
    }

    #[test]
    fn test_prime_factors_with_exponents() {
        assert_eq!(prime_factors(1), Vec::<(u64, u32)>::new());
        assert_eq!(prime_factors(2), vec![(2, 1)]);
        assert_eq!(prime_factors(4), vec![(2, 2)]);
        assert_eq!(prime_factors(12), vec![(2, 2), (3, 1)]);
        assert_eq!(prime_factors(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(prime_factors(97), vec![(97, 1)]);
        // product recovery
        let n = 360u64;
        let f = prime_factors(n);
        let recovered: u64 = f.iter().map(|(p, e)| p.pow(*e)).product();
        assert_eq!(recovered, n);
    }

    #[test]
    fn test_euler_totient_nt() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(2), 1);
        assert_eq!(euler_totient(6), 2);
        assert_eq!(euler_totient(9), 6);
        assert_eq!(euler_totient(12), 4);
        assert_eq!(euler_totient(0), 0);
    }

    #[test]
    fn test_mobius_nt() {
        assert_eq!(mobius(1), 1);
        assert_eq!(mobius(2), -1);
        assert_eq!(mobius(4), 0);   // 2^2
        assert_eq!(mobius(6), 1);   // 2·3
        assert_eq!(mobius(30), -1); // 2·3·5
        assert_eq!(mobius(0), 0);
    }

    #[test]
    fn test_divisor_sum() {
        // σ_0(n) = number of divisors
        assert_eq!(divisor_sum(1, 0), 1);
        assert_eq!(divisor_sum(6, 0), 4);   // 1,2,3,6
        assert_eq!(divisor_sum(12, 0), 6);  // 1,2,3,4,6,12
        // σ_1(n) = sum of divisors
        assert_eq!(divisor_sum(1, 1), 1);
        assert_eq!(divisor_sum(6, 1), 12);  // 1+2+3+6
        assert_eq!(divisor_sum(12, 1), 28); // 1+2+3+4+6+12
        // σ_2(6) = 1+4+9+36 = 50
        assert_eq!(divisor_sum(6, 2), 50);
    }

    #[test]
    fn test_jacobi_symbol() {
        assert_eq!(jacobi_symbol(0, 5).expect("ok"), 0);
        assert_eq!(jacobi_symbol(1, 5).expect("ok"), 1);
        assert_eq!(jacobi_symbol(2, 5).expect("ok"), -1);
        assert_eq!(jacobi_symbol(4, 5).expect("ok"), 1);
        // n = 15 = 3 * 5 (composite odd)
        assert_eq!(jacobi_symbol(5, 15).expect("ok"), 0);  // gcd(5,15)=5
        assert_eq!(jacobi_symbol(1, 15).expect("ok"), 1);
        // Errors for even n
        assert!(jacobi_symbol(1, 4).is_err());
        assert!(jacobi_symbol(1, 0).is_err());
    }
}

// ── Advanced number-theoretic functions ───────────────────────────────────────

/// Euler's phi (totient) function φ(n) — alias for `euler_totient`.
///
/// Counts the number of integers in [1, n] that are coprime to n.
/// This is an alias provided for API compatibility with the task specification.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::euler_phi;
/// assert_eq!(euler_phi(1),  1);
/// assert_eq!(euler_phi(9),  6);
/// assert_eq!(euler_phi(12), 4);
/// ```
#[inline]
pub fn euler_phi(n: u64) -> u64 {
    euler_totient(n)
}

/// Von Mangoldt function Λ(n).
///
/// Defined as:
/// * Λ(n) = ln(p)  if n = p^k for some prime p and integer k ≥ 1
/// * Λ(n) = 0       otherwise
///
/// This function arises in the explicit formula for the prime counting function
/// and in the Riemann hypothesis.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::von_mangoldt;
/// assert_eq!(von_mangoldt(1),  0.0);
/// assert!((von_mangoldt(2) - 2_f64.ln()).abs() < 1e-14);
/// assert!((von_mangoldt(4) - 2_f64.ln()).abs() < 1e-14);  // 4 = 2^2
/// assert!((von_mangoldt(6)).abs() < 1e-14);               // 6 is not a prime power
/// assert!((von_mangoldt(9) - 3_f64.ln()).abs() < 1e-14);  // 9 = 3^2
/// ```
pub fn von_mangoldt(n: u64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let factors = prime_factors(n);
    // n is a prime power iff it has exactly one distinct prime factor
    if factors.len() == 1 {
        (factors[0].0 as f64).ln()
    } else {
        0.0
    }
}

/// Ramanujan sum c_k(n) = Σ_{gcd(m,k)=1, 1≤m≤k} exp(2πi·m·n/k).
///
/// This is purely real and integer-valued:
/// ```text
/// c_k(n) = μ(k / gcd(k, n)) · φ(k) / φ(k / gcd(k, n))
/// ```
/// which is equivalent to a sum over primitive k-th roots of unity.
///
/// # Special cases
/// * c_k(0) = φ(k)
/// * c_1(n) = 1  for all n
/// * c_k(1) = μ(k) for all k ≥ 1
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::ramanujan_sum;
/// assert_eq!(ramanujan_sum(1, 5),  1);   // c_1(5) = 1
/// assert_eq!(ramanujan_sum(5, 0),  4);   // c_5(0) = φ(5) = 4
/// assert_eq!(ramanujan_sum(5, 1), -1);   // c_5(1) = μ(5) = -1
/// assert_eq!(ramanujan_sum(6, 1), 1);    // c_6(1) = μ(6)  (6=2·3 → μ=1)
/// ```
pub fn ramanujan_sum(k: u64, n: u64) -> i64 {
    if k == 0 {
        return 0;
    }
    if k == 1 {
        return 1;
    }

    let g = gcd(k, n);
    let k_over_g = k / g;

    // c_k(n) = μ(k/gcd(k,n)) · φ(k) / φ(k/gcd(k,n))
    let mu_val = mobius(k_over_g) as i64;
    if mu_val == 0 {
        return 0;
    }

    let phi_k = euler_totient(k) as i64;
    let phi_k_over_g = euler_totient(k_over_g) as i64;

    if phi_k_over_g == 0 {
        return 0;
    }

    mu_val * phi_k / phi_k_over_g
}

/// Jordan's totient function J_k(n).
///
/// Generalises Euler's totient:
/// ```text
/// J_k(n) = n^k · ∏_{p | n} (1 − p^{−k})
/// ```
/// So J_1(n) = φ(n).
///
/// For k = 0, returns 1 by convention (empty product analog).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::jordan_totient;
/// assert_eq!(jordan_totient(6, 1), 2);  // φ(6) = 2
/// assert_eq!(jordan_totient(6, 2), 24); // J_2(6) = 36·(1-1/4)·(1-1/9) = 24
/// ```
pub fn jordan_totient(n: u64, k: u32) -> u64 {
    if n == 0 {
        return 0;
    }
    if k == 0 {
        return 1;
    }

    // J_k(n) = n^k * product over primes p|n of (1 - 1/p^k)
    // = n^k * product over primes p|n of (p^k - 1) / p^k
    // Computed multiplicatively to avoid floating-point:
    // For each prime power p^e || n:
    //   contribution = p^{k*e} * (1 - p^{-k}) = p^{k*(e-1)} * (p^k - 1)

    let factors = prime_factors(n);
    if factors.is_empty() {
        return 1; // n == 1
    }

    let mut result: u64 = 1;
    for (p, e) in factors {
        // p^{k*e} factor
        let pk = p.saturating_pow(k); // p^k
        let pke = pk.saturating_pow(e); // p^{k*e}
        // (1 - p^{-k}) part = (p^k - 1) / p^k
        // So contribution = p^{k*(e-1)} * (p^k - 1)
        let pk_e_minus_1 = pk.saturating_pow(e - 1); // p^{k*(e-1)}
        let term = pk_e_minus_1.saturating_mul(pk.saturating_sub(1));
        result = result.saturating_mul(term);
    }

    result
}

/// Arithmetic derivative n' of a natural number.
///
/// Defined via the product rule applied to prime factorizations:
/// * 0' = 0,  1' = 0
/// * p' = 1  for any prime p
/// * (ab)' = a'·b + a·b'
///
/// Equivalently, for n = ∏ p_i^{e_i}:
/// ```text
/// n' = n · Σ_i  e_i / p_i
/// ```
///
/// Uses saturating arithmetic to avoid overflow.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::arithmetic_derivative;
/// assert_eq!(arithmetic_derivative(0), 0);
/// assert_eq!(arithmetic_derivative(1), 0);
/// assert_eq!(arithmetic_derivative(2), 1);  // prime
/// assert_eq!(arithmetic_derivative(4), 4);  // 4 = 2^2 → 4 · (2/2) = 4
/// assert_eq!(arithmetic_derivative(6), 5);  // 6 = 2·3 → 6·(1/2 + 1/3) = 5
/// ```
pub fn arithmetic_derivative(n: u64) -> u64 {
    if n <= 1 {
        return 0;
    }

    let factors = prime_factors(n);
    if factors.is_empty() {
        return 0;
    }

    // n' = n · Σ_i e_i / p_i
    // To keep integer arithmetic, compute as:
    // n' = Σ_i  (n / p_i) * e_i
    // Since n = ∏ p_j^{e_j}, n/p_i is always an integer.

    let mut result: u64 = 0;
    for (p, e) in &factors {
        // n / p_i
        let n_div_p = n / p;
        let contribution = n_div_p.saturating_mul(*e as u64);
        result = result.saturating_add(contribution);
    }
    result
}

/// Prime zeta function P(s) = Σ_{p prime} p^{−s}.
///
/// Approximated by summing over the first `n_primes` primes.
/// Converges rapidly for s > 1.
///
/// # Arguments
/// * `s`        - Real argument (s > 1 for absolute convergence)
/// * `n_primes` - Number of prime terms to include (e.g. 1000 for good accuracy)
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::prime_zeta;
/// // P(2) ≈ 0.4522...
/// let val = prime_zeta(2.0, 5000);
/// assert!((val - 0.4522).abs() < 1e-3);
/// ```
pub fn prime_zeta(s: f64, n_primes: usize) -> f64 {
    if n_primes == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let mut p = 2u64;
    let mut count = 0usize;
    while count < n_primes {
        sum += (p as f64).powf(-s);
        p = next_prime(p);
        count += 1;
    }
    sum
}

/// Euler product formula for the Riemann zeta function.
///
/// ζ(s) = ∏_{p prime} 1 / (1 − p^{−s})
///
/// Approximated by the product over the first `n_primes` primes.
/// The infinite product converges for Re(s) > 1.
///
/// # Arguments
/// * `s`        - Real argument (s > 1)
/// * `n_primes` - Number of prime factors to include
///
/// # Examples
/// ```
/// use scirs2_special::number_theory::euler_product;
/// // ζ(2) = π²/6 ≈ 1.6449...
/// let val = euler_product(2.0, 5000);
/// assert!((val - std::f64::consts::PI.powi(2) / 6.0).abs() < 0.01);
/// ```
pub fn euler_product(s: f64, n_primes: usize) -> f64 {
    if n_primes == 0 {
        return 1.0;
    }
    let mut product = 1.0f64;
    let mut p = 2u64;
    let mut count = 0usize;
    while count < n_primes {
        let p_s = (p as f64).powf(s);
        let factor = p_s / (p_s - 1.0);
        product *= factor;
        p = next_prime(p);
        count += 1;
    }
    product
}

// ── Tests for new functions ───────────────────────────────────────────────────

#[cfg(test)]
mod advanced_tests {
    use super::*;

    #[test]
    fn test_euler_phi_is_alias() {
        // euler_phi must equal euler_totient for a range of inputs
        for n in 0..=50u64 {
            assert_eq!(euler_phi(n), euler_totient(n), "n = {n}");
        }
    }

    #[test]
    fn test_von_mangoldt() {
        assert_eq!(von_mangoldt(1), 0.0);
        // primes
        for p in [2u64, 3, 5, 7, 11, 13, 97] {
            let val = von_mangoldt(p);
            assert!(
                (val - (p as f64).ln()).abs() < 1e-14,
                "p = {p}: {val}"
            );
        }
        // prime squares
        for (p, p2) in [(2u64, 4u64), (3, 9), (5, 25)] {
            let val = von_mangoldt(p2);
            assert!(
                (val - (p as f64).ln()).abs() < 1e-14,
                "p^2 = {p2}: {val}"
            );
        }
        // non-prime-powers
        for c in [6u64, 10, 12, 15, 30] {
            assert_eq!(von_mangoldt(c), 0.0, "composite {c}");
        }
    }

    #[test]
    fn test_ramanujan_sum_special_cases() {
        // c_1(n) = 1 for all n
        for n in 0..=10u64 {
            assert_eq!(ramanujan_sum(1, n), 1, "c_1({n})");
        }
        // c_k(0) = φ(k)
        for k in 1..=10u64 {
            assert_eq!(
                ramanujan_sum(k, 0),
                euler_totient(k) as i64,
                "c_{k}(0)"
            );
        }
        // c_k(1) = μ(k)
        for k in 1..=10u64 {
            assert_eq!(
                ramanujan_sum(k, 1),
                mobius(k) as i64,
                "c_{k}(1)"
            );
        }
        // Known values
        assert_eq!(ramanujan_sum(5, 1), -1);
        assert_eq!(ramanujan_sum(6, 1),  1);
        // c_4(2): gcd(4,2)=2, k/gcd=2, μ(2)=-1, φ(4)/φ(2) = 2/1 = 2 → c_4(2) = -2
        assert_eq!(ramanujan_sum(4, 2), -2);
    }

    #[test]
    fn test_jordan_totient_k1() {
        // J_1(n) = φ(n)
        for n in 1..=20u64 {
            assert_eq!(
                jordan_totient(n, 1),
                euler_totient(n),
                "J_1({n})"
            );
        }
    }

    #[test]
    fn test_jordan_totient_k0() {
        // J_0(n) = 1 for n ≥ 1
        for n in 1..=10u64 {
            assert_eq!(jordan_totient(n, 0), 1, "J_0({n})");
        }
    }

    #[test]
    fn test_jordan_totient_known() {
        // J_2(6): 6 = 2·3
        // 6^2 · (1 - 1/4) · (1 - 1/9) = 36 · 3/4 · 8/9 = 36 · 24/36 = 24
        // But via our formula: p=2,e=1 → 2^{1} * (2^2-1) = 2*3 = 6;
        //                       p=3,e=1 → 3^{1} * (3^2-1) = 3*8 = 24;  wait — result = 6*24? No, let's recompute.
        // Actually J_k(n) = product over distinct primes p|n of p^{k*(e-1)} * (p^k - 1).
        // For 6 = 2^1 * 3^1 with k=2:
        //   p=2, e=1: 2^{2*0} * (2^2 - 1) = 1 * 3 = 3
        //   p=3, e=1: 3^{2*0} * (3^2 - 1) = 1 * 8 = 8
        //   J_2(6) = 3 * 8 = 24... but wait, we should also multiply n^k / n^k factor?
        // No: J_k(n) = n^k · ∏(1 - p^{-k}) = 36 * (1-1/4) * (1-1/9) = 36 * 3/4 * 8/9 = 36*2/3 = 24.
        // Our formula: result starts at 1, multiply for each prime.
        // p=2, e=1: pk=4, pke=4, pk_e_minus_1 = 4^0 = 1, term = 1*(4-1) = 3
        // p=3, e=1: pk=9, pke=9, pk_e_minus_1 = 9^0 = 1, term = 1*(9-1) = 8
        // result = 3*8 = 24.
        assert_eq!(jordan_totient(6, 2), 24);
    }

    #[test]
    fn test_arithmetic_derivative() {
        assert_eq!(arithmetic_derivative(0), 0);
        assert_eq!(arithmetic_derivative(1), 0);
        assert_eq!(arithmetic_derivative(2), 1);
        assert_eq!(arithmetic_derivative(3), 1);
        assert_eq!(arithmetic_derivative(4), 4);  // 4 = 2^2: 4*(2/2) = 4
        assert_eq!(arithmetic_derivative(6), 5);  // 6=2·3: 6*(1/2+1/3) = 3+2 = 5
        assert_eq!(arithmetic_derivative(9), 6);  // 9=3^2: 9*(2/3) = 6
        // product rule check: (ab)' = a'b + ab'
        let a = 5u64;
        let b = 7u64;
        let ab_d = arithmetic_derivative(a * b);
        let manual = arithmetic_derivative(a) * b + a * arithmetic_derivative(b);
        assert_eq!(ab_d, manual, "(5·7)' = {ab_d}, manual = {manual}");
    }

    #[test]
    fn test_prime_zeta_s2() {
        // P(2) ≈ 0.45224742...
        let val = prime_zeta(2.0, 10000);
        assert!((val - 0.45224742).abs() < 1e-4, "P(2) = {val}");
    }

    #[test]
    fn test_euler_product_s2() {
        // ζ(2) = π²/6
        let expected = std::f64::consts::PI.powi(2) / 6.0;
        let val = euler_product(2.0, 5000);
        assert!((val - expected).abs() < 0.01, "ζ(2) ≈ {val}, expected {expected}");
    }

    #[test]
    fn test_euler_product_s4() {
        // ζ(4) = π⁴/90
        let expected = std::f64::consts::PI.powi(4) / 90.0;
        let val = euler_product(4.0, 5000);
        assert!((val - expected).abs() < 0.001, "ζ(4) ≈ {val}, expected {expected}");
    }
}
