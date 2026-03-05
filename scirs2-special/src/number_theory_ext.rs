//! Extended number-theory functions providing square-root modular arithmetic,
//! the Chinese Remainder Theorem, quadratic residue symbols, deterministic
//! Miller-Rabin primality testing, and multiplicative arithmetic functions.
//!
//! These complement the existing `number_theory` module.  Where the same
//! concept appears in both modules the signatures may differ (e.g. this
//! module uses signed `i64` for Legendre/Jacobi symbols so that callers
//! need not cast).

use std::collections::HashMap;

// ── Modular arithmetic primitives ────────────────────────────────────────────

/// Modular exponentiation: `base^exp mod modulus`.
///
/// Returns 0 when `modulus == 1`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::mod_pow;
/// assert_eq!(mod_pow(2, 10, 1000), 24);
/// assert_eq!(mod_pow(3, 4, 7), 4);
/// ```
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result.wrapping_mul(base) % modulus;
        }
        exp >>= 1;
        base = base.wrapping_mul(base) % modulus;
    }
    result
}

/// Extended Euclidean algorithm: returns `(gcd, x, y)` satisfying `a·x + b·y = gcd`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::extended_gcd_signed;
/// let (g, x, y) = extended_gcd_signed(35, 15);
/// assert_eq!(g, 5);
/// assert_eq!(35 * x + 15 * y, 5);
/// ```
pub fn extended_gcd_signed(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        return (a, 1, 0);
    }
    let (g, x, y) = extended_gcd_signed(b, a % b);
    (g, y, x - (a / b) * y)
}

/// Modular multiplicative inverse: `a⁻¹ mod m`.
///
/// Returns `None` when `gcd(a, m) ≠ 1`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::mod_inverse_signed;
/// assert_eq!(mod_inverse_signed(3, 7), Some(5));   // 3·5 ≡ 1 (mod 7)
/// assert_eq!(mod_inverse_signed(2, 4), None);       // not invertible
/// ```
pub fn mod_inverse_signed(a: i64, m: i64) -> Option<i64> {
    let (g, x, _) = extended_gcd_signed(a.rem_euclid(m), m);
    if g != 1 {
        None
    } else {
        Some(x.rem_euclid(m))
    }
}

// ── Quadratic residue symbols ─────────────────────────────────────────────────

/// Legendre symbol (a/p) for an odd prime `p`.
///
/// Returns `1` if `a` is a quadratic residue mod `p`, `-1` if it is a
/// non-residue, and `0` if `p` divides `a`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::legendre_symbol;
/// assert_eq!(legendre_symbol(2, 7),  1);   // 2 ≡ 3² (mod 7)
/// assert_eq!(legendre_symbol(3, 7), -1);
/// assert_eq!(legendre_symbol(7, 7),  0);
/// ```
pub fn legendre_symbol(a: i64, p: i64) -> i64 {
    debug_assert!(p > 2, "p must be an odd prime");
    let a_red = a.rem_euclid(p) as u64;
    if a_red == 0 {
        return 0;
    }
    let exp = (p as u64 - 1) / 2;
    let result = mod_pow(a_red, exp, p as u64) as i64;
    if result == p - 1 {
        -1
    } else {
        result
    }
}

/// Jacobi symbol (a/n) — a generalisation of the Legendre symbol to odd
/// positive integers `n`.
///
/// When `n` is an odd prime the result agrees with `legendre_symbol`.
///
/// # Panics
/// Panics if `n` is even or non-positive (debug builds only).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::jacobi_symbol;
/// assert_eq!(jacobi_symbol(2, 15),  1);
/// assert_eq!(jacobi_symbol(5, 9),   1);
/// assert_eq!(jacobi_symbol(3, 9),   0);
/// ```
pub fn jacobi_symbol(mut a: i64, mut n: i64) -> i64 {
    debug_assert!(n > 0 && n % 2 == 1, "n must be an odd positive integer");
    let mut result = 1i64;
    a = a.rem_euclid(n);

    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let r = n % 8;
            if r == 3 || r == 5 {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    if n == 1 {
        result
    } else {
        0
    }
}

// ── Tonelli-Shanks modular square root ───────────────────────────────────────

/// Compute a square root of `n` modulo an odd prime `p` using the
/// Tonelli-Shanks algorithm.
///
/// Returns `Some(r)` with `r² ≡ n (mod p)` when a root exists, or `None`
/// when `n` is a quadratic non-residue mod `p`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::sqrt_mod_prime;
/// let r = sqrt_mod_prime(2, 7).unwrap();
/// assert!((r * r) % 7 == 2);
/// assert_eq!(sqrt_mod_prime(3, 7), None);  // 3 is a NQR mod 7
/// ```
pub fn sqrt_mod_prime(n: u64, p: u64) -> Option<u64> {
    let n = n % p;
    if n == 0 {
        return Some(0);
    }
    if p == 2 {
        return Some(n % 2);
    }
    if legendre_symbol(n as i64, p as i64) != 1 {
        return None;
    }

    // p ≡ 3 (mod 4): simple closed form
    if p % 4 == 3 {
        return Some(mod_pow(n, (p + 1) / 4, p));
    }

    // General Tonelli-Shanks
    // Write p−1 = Q · 2^S
    let mut q = p - 1;
    let mut s = 0u32;
    while q % 2 == 0 {
        q /= 2;
        s += 1;
    }

    // Find a quadratic non-residue z
    let mut z = 2u64;
    while legendre_symbol(z as i64, p as i64) != -1 {
        z += 1;
    }

    let mut m = s;
    let mut c = mod_pow(z, q, p);
    let mut t = mod_pow(n, q, p);
    let mut r = mod_pow(n, (q + 1) / 2, p);

    loop {
        if t == 1 {
            return Some(r);
        }

        // Find least i > 0 such that t^(2^i) ≡ 1 (mod p)
        let mut i = 1u32;
        let mut tmp = (t * t) % p;
        while tmp != 1 {
            tmp = (tmp * tmp) % p;
            i += 1;
        }

        // b = c^(2^(m-i-1)) mod p
        let b = mod_pow(c, 1u64 << (m - i - 1), p);
        m = i;
        c = (b * b) % p;
        t = (t * c) % p;
        r = (r * b) % p;
    }
}

// ── Chinese Remainder Theorem ─────────────────────────────────────────────────

/// Solve a system of simultaneous congruences by the Chinese Remainder Theorem.
///
/// `residues` is a slice of `(a_i, m_i)` pairs representing `x ≡ a_i (mod m_i)`.
/// The moduli must be pairwise coprime.
///
/// Returns `Some((x, M))` where `M = ∏ m_i` and `0 ≤ x < M`, or `None` if
/// any modular inverse does not exist (i.e. moduli are not pairwise coprime).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::crt;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 (mod 15)
/// let (x, m) = crt(&[(2, 3), (3, 5)]).unwrap();
/// assert_eq!(m, 15);
/// assert_eq!(x, 8);
/// ```
pub fn crt(residues: &[(i64, i64)]) -> Option<(i64, i64)> {
    if residues.is_empty() {
        return Some((0, 1));
    }
    let big_m: i64 = residues.iter().map(|(_, m)| m).product();
    let mut x = 0i64;
    for &(a, mi) in residues {
        let mi_bar = big_m / mi;
        let inv = mod_inverse_signed(mi_bar, mi)?;
        x = (x + a.rem_euclid(mi) * mi_bar % big_m * inv) % big_m;
    }
    Some((x.rem_euclid(big_m), big_m))
}

// ── Primality and factorization ───────────────────────────────────────────────

/// Deterministic Miller-Rabin primality test for all `n < 3_215_031_751`.
///
/// Uses the witness set `{2, 3, 5, 7}`.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::is_prime_miller_rabin;
/// assert!( is_prime_miller_rabin(2));
/// assert!( is_prime_miller_rabin(97));
/// assert!(!is_prime_miller_rabin(1));
/// assert!(!is_prime_miller_rabin(100));
/// assert!(!is_prime_miller_rabin(561));  // Carmichael number
/// ```
pub fn is_prime_miller_rabin(n: u64) -> bool {
    match n {
        0 | 1 => return false,
        2 | 3 | 5 | 7 => return true,
        _ if n % 2 == 0 => return false,
        _ => {}
    }

    // Write n−1 = d · 2^r
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }

    'outer: for &a in &[2u64, 3, 5, 7] {
        if a >= n {
            continue;
        }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..r - 1 {
            x = x.wrapping_mul(x) % n;
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Greatest common divisor (iterative Euclidean algorithm).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::gcd_u64;
/// assert_eq!(gcd_u64(12, 8), 4);
/// assert_eq!(gcd_u64(0, 5),  5);
/// ```
pub fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Trial-division prime factorization (with small primes first).
///
/// Returns the prime factors of `n` in non-decreasing order with repetition.
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::factorize_ext;
/// assert_eq!(factorize_ext(12), vec![2, 2, 3]);
/// assert_eq!(factorize_ext(1),  vec![]);
/// ```
pub fn factorize_ext(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    while n % 2 == 0 {
        factors.push(2u64);
        n /= 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        while n % d == 0 {
            factors.push(d);
            n /= d;
        }
        d += 2;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

// ── Multiplicative arithmetic functions ──────────────────────────────────────

/// Euler's totient function φ(n): count of integers in 1..n coprime to n.
///
/// Uses the formula  φ(n) = n · ∏_{p | n} (1 − 1/p).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::euler_phi_ext;
/// assert_eq!(euler_phi_ext(1),  1);
/// assert_eq!(euler_phi_ext(12), 4);
/// assert_eq!(euler_phi_ext(36), 12);
/// ```
pub fn euler_phi_ext(n: u64) -> u64 {
    let factors = factorize_ext(n);
    // Collect distinct prime factors
    let mut primes: Vec<u64> = factors.clone();
    primes.dedup();
    let mut result = n;
    for p in &primes {
        result = result / p * (p - 1);
    }
    result
}

/// Divisor function σ_k(n): sum of k-th powers of divisors of n.
///
/// `k = 0` gives the divisor count τ(n); `k = 1` gives the divisor sum.
///
/// Computed from the prime factorisation using the multiplicative formula
/// σ_k(p^e) = (p^{k(e+1)} − 1) / (p^k − 1).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::sigma;
/// assert_eq!(sigma(6, 0), 4);   // τ(6) = 4 (divisors: 1,2,3,6)
/// assert_eq!(sigma(6, 1), 12);  // σ(6) = 12
/// assert_eq!(sigma(4, 1), 7);   // σ(4) = 1+2+4
/// ```
pub fn sigma(n: u64, k: u32) -> u128 {
    let factors = factorize_ext(n);
    let mut prime_powers: HashMap<u64, u32> = HashMap::new();
    for p in factors {
        *prime_powers.entry(p).or_insert(0) += 1;
    }
    prime_powers
        .iter()
        .map(|(&p, &e)| {
            if k == 0 {
                (e + 1) as u128
            } else {
                let pk = (p as u128).pow(k);
                // (pk^(e+1) - 1) / (pk - 1)
                (pk.pow(e + 1) - 1) / (pk - 1)
            }
        })
        .product()
}

/// Number-of-divisors function τ(n) = σ_0(n).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::tau;
/// assert_eq!(tau(1),  1);
/// assert_eq!(tau(6),  4);
/// assert_eq!(tau(12), 6);
/// ```
pub fn tau(n: u64) -> u64 {
    sigma(n, 0) as u64
}

/// Sum of divisors σ(n) = σ_1(n).
///
/// # Examples
/// ```
/// use scirs2_special::number_theory_ext::sum_of_divisors;
/// assert_eq!(sum_of_divisors(6),  12);
/// assert_eq!(sum_of_divisors(12), 28);
/// ```
pub fn sum_of_divisors(n: u64) -> u64 {
    sigma(n, 1) as u64
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 4, 7), 4);
        assert_eq!(mod_pow(5, 0, 13), 1);
        assert_eq!(mod_pow(7, 1, 13), 7);
    }

    #[test]
    fn test_extended_gcd_signed() {
        let (g, x, y) = extended_gcd_signed(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }

    #[test]
    fn test_mod_inverse_signed() {
        assert_eq!(mod_inverse_signed(3, 7), Some(5));
        assert_eq!(mod_inverse_signed(2, 4), None);
        // Verify: 3·5 mod 7 = 15 mod 7 = 1
        assert_eq!((3 * 5) % 7, 1);
    }

    #[test]
    fn test_legendre_symbol() {
        // Quadratic residues mod 7: {1, 2, 4}
        assert_eq!(legendre_symbol(1, 7), 1);
        assert_eq!(legendre_symbol(2, 7), 1);
        assert_eq!(legendre_symbol(3, 7), -1);
        assert_eq!(legendre_symbol(4, 7), 1);
        assert_eq!(legendre_symbol(5, 7), -1);
        assert_eq!(legendre_symbol(6, 7), -1);
        assert_eq!(legendre_symbol(7, 7), 0);
        assert_eq!(legendre_symbol(0, 7), 0);
    }

    #[test]
    fn test_jacobi_symbol() {
        // Jacobi symbol agrees with Legendre for prime modulus
        assert_eq!(jacobi_symbol(2, 7), legendre_symbol(2, 7));
        assert_eq!(jacobi_symbol(3, 7), legendre_symbol(3, 7));
        // Composite examples
        assert_eq!(jacobi_symbol(2, 15), 1);
        assert_eq!(jacobi_symbol(3, 9), 0);
    }

    #[test]
    fn test_sqrt_mod_prime() {
        // 3² = 9 ≡ 2 (mod 7)
        let r = sqrt_mod_prime(2, 7).expect("root should exist");
        assert_eq!((r * r) % 7, 2);
        // 3 is NQR mod 7
        assert_eq!(sqrt_mod_prime(3, 7), None);
        // sqrt mod 13
        let r13 = sqrt_mod_prime(3, 13).expect("root should exist");
        assert_eq!((r13 * r13) % 13, 3);
    }

    #[test]
    fn test_crt() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x ≡ 8 (mod 15)
        let (x, m) = crt(&[(2, 3), (3, 5)]).expect("CRT should solve");
        assert_eq!(m, 15);
        assert_eq!(x, 8);
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
        // Empty residues
        let (x0, m0) = crt(&[]).expect("empty CRT");
        assert_eq!((x0, m0), (0, 1));
    }

    #[test]
    fn test_is_prime_miller_rabin() {
        for &p in &[2u64, 3, 5, 7, 11, 13, 17, 19, 23, 97, 101, 9973] {
            assert!(is_prime_miller_rabin(p), "{} should be prime", p);
        }
        for &c in &[0u64, 1, 4, 9, 15, 100, 561, 1105] {
            assert!(!is_prime_miller_rabin(c), "{} should be composite", c);
        }
    }

    #[test]
    fn test_euler_phi_ext() {
        assert_eq!(euler_phi_ext(1), 1);
        assert_eq!(euler_phi_ext(2), 1);
        assert_eq!(euler_phi_ext(6), 2);
        assert_eq!(euler_phi_ext(12), 4);
        assert_eq!(euler_phi_ext(36), 12);
    }

    #[test]
    fn test_sigma() {
        assert_eq!(sigma(6, 0), 4);
        assert_eq!(sigma(6, 1), 12);
        assert_eq!(sigma(4, 1), 7);
        assert_eq!(sigma(1, 1), 1);
    }

    #[test]
    fn test_tau() {
        assert_eq!(tau(1), 1);
        assert_eq!(tau(6), 4);
        assert_eq!(tau(12), 6);
    }

    #[test]
    fn test_sum_of_divisors() {
        assert_eq!(sum_of_divisors(6), 12);
        assert_eq!(sum_of_divisors(12), 28);
    }

    #[test]
    fn test_factorize_ext() {
        assert_eq!(factorize_ext(1), Vec::<u64>::new());
        assert_eq!(factorize_ext(12), vec![2u64, 2, 3]);
        assert_eq!(factorize_ext(13), vec![13u64]);
        assert_eq!(factorize_ext(36), vec![2u64, 2, 3, 3]);
    }
}
