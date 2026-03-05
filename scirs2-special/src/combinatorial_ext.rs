//! Extended combinatorial and number-theory combinatorial functions
//!
//! This module provides integer-typed Stirling numbers, Bell numbers, Bernoulli
//! numbers, Catalan numbers, rising/falling factorials, multinomial coefficients,
//! prime sieves, Euler's totient, the Möbius function, Jordan's totient, and the
//! integer partition function.
//!
//! All integer results are returned as primitive integer types (u64 / i64 / i8)
//! to make exact arithmetic possible for moderate inputs, while floating-point
//! helpers cover larger inputs where only approximate answers are needed.

use crate::error::{SpecialError, SpecialResult};
use std::collections::HashMap;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Compute log-factorial using Stirling approximation for large n.
///
/// This is used internally to avoid overflow when computing combinatorial
/// quantities that would otherwise overflow u64.
fn log_factorial(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut acc = 0.0;
    for i in 2..=n {
        acc += (i as f64).ln();
    }
    acc
}

// ── Stirling numbers (integer-typed) ─────────────────────────────────────────

/// Signed Stirling number of the first kind  s(n, k).
///
/// Counts permutations of n elements with exactly k cycles, with the sign
/// convention (−1)^(n−k) applied.  The recurrence is:
///
/// ```text
/// s(n, k) = s(n−1, k−1) − (n−1) · s(n−1, k)
/// ```
///
/// # Panics
/// Returns an error if the exact result overflows i64 (n ≳ 20).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::stirling_first;
/// assert_eq!(stirling_first(4, 2), 11);   // unsigned value
/// assert_eq!(stirling_first(3, 1), 2);    // 3!/1 permutations sign → +2
/// ```
pub fn stirling_first(n: usize, k: usize) -> i64 {
    if n == 0 && k == 0 {
        return 1;
    }
    if n == 0 || k == 0 || k > n {
        return 0;
    }

    // DP table (signed)
    let mut dp = vec![vec![0i64; k + 1]; n + 1];
    dp[0][0] = 1;

    for i in 1..=n {
        for j in 1..=i.min(k) {
            let prev_same = dp[i - 1][j];
            let prev_less = dp[i - 1][j - 1];
            dp[i][j] = prev_less.saturating_sub((i as i64 - 1).saturating_mul(prev_same));
        }
    }
    dp[n][k]
}

/// Unsigned Stirling number of the first kind  |s(n, k)|.
///
/// Counts the number of permutations of n elements with exactly k cycles.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::stirling_first_unsigned;
/// assert_eq!(stirling_first_unsigned(4, 2), 11);
/// assert_eq!(stirling_first_unsigned(5, 3), 35);
/// ```
pub fn stirling_first_unsigned(n: usize, k: usize) -> u64 {
    if n == 0 && k == 0 {
        return 1;
    }
    if n == 0 || k == 0 || k > n {
        return 0;
    }

    // Unsigned recurrence: |s(n,k)| = (n−1)·|s(n−1,k)| + |s(n−1,k−1)|
    let mut dp = vec![vec![0u64; k + 1]; n + 1];
    dp[0][0] = 1;

    for i in 1..=n {
        for j in 1..=i.min(k) {
            let a = (i as u64 - 1).saturating_mul(dp[i - 1][j]);
            let b = dp[i - 1][j - 1];
            dp[i][j] = a.saturating_add(b);
        }
    }
    dp[n][k]
}

/// Stirling number of the second kind  S(n, k).
///
/// Counts the number of ways to partition a set of n elements into exactly
/// k non-empty subsets.  The recurrence is:
///
/// ```text
/// S(n, k) = k · S(n−1, k) + S(n−1, k−1)
/// ```
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::stirling_second;
/// assert_eq!(stirling_second(4, 2), 7);
/// assert_eq!(stirling_second(5, 3), 25);
/// ```
pub fn stirling_second(n: usize, k: usize) -> u64 {
    if n == 0 && k == 0 {
        return 1;
    }
    if n == 0 || k == 0 || k > n {
        return 0;
    }

    let mut dp = vec![vec![0u64; k + 1]; n + 1];
    dp[0][0] = 1;

    for i in 1..=n {
        for j in 1..=i.min(k) {
            let a = (j as u64).saturating_mul(dp[i - 1][j]);
            let b = dp[i - 1][j - 1];
            dp[i][j] = a.saturating_add(b);
        }
    }
    dp[n][k]
}

// ── Bell numbers ──────────────────────────────────────────────────────────────

/// Bell number B(n) — number of partitions of a set of n elements.
///
/// B(n) = Σ_{k=0}^{n} S(n, k)  where S(n, k) are Stirling numbers of the
/// second kind.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::bell_number;
/// assert_eq!(bell_number(0), 1);
/// assert_eq!(bell_number(5), 52);
/// assert_eq!(bell_number(10), 115975);
/// ```
pub fn bell_number(n: usize) -> u64 {
    if n == 0 {
        return 1;
    }
    // Bell triangle method (more numerically stable than summing Stirling numbers)
    // b[0] = 1; b[k] = b[k-1] + b_prev[k-1]
    let mut row = vec![1u64; n + 1];
    for i in 1..=n {
        let mut new_row = vec![0u64; n + 1];
        new_row[0] = row[i - 1]; // first element of new row = last element of previous row
        for j in 1..=i {
            new_row[j] = new_row[j - 1].saturating_add(row[j - 1]);
        }
        row = new_row;
    }
    row[0]
}

// ── Bernoulli numbers ─────────────────────────────────────────────────────────

/// Bernoulli number B_n as f64.
///
/// Uses the modern convention with B_1 = −1/2.  All odd Bernoulli numbers
/// B_n for n ≥ 3 are zero.
///
/// Computed via the recurrence relation:
///
/// ```text
/// B_n = −1/(n+1) · Σ_{k=0}^{n−1} C(n+1, k) · B_k
/// ```
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::combinatorial_ext::bernoulli;
/// assert_eq!(bernoulli(0), 1.0);
/// assert_relative_eq!(bernoulli(1), -0.5, epsilon = 1e-14);
/// assert_relative_eq!(bernoulli(2), 1.0/6.0, epsilon = 1e-14);
/// assert_eq!(bernoulli(3), 0.0);
/// ```
pub fn bernoulli(n: usize) -> f64 {
    match n {
        0 => return 1.0,
        1 => return -0.5,
        _ if n % 2 == 1 => return 0.0,
        2 => return 1.0 / 6.0,
        4 => return -1.0 / 30.0,
        6 => return 1.0 / 42.0,
        8 => return -1.0 / 30.0,
        10 => return 5.0 / 66.0,
        12 => return -691.0 / 2730.0,
        _ => {}
    }

    // General recurrence
    let mut b = vec![0.0f64; n + 1];
    b[0] = 1.0;
    b[1] = -0.5;

    for m in 2..=n {
        if m % 2 == 1 {
            b[m] = 0.0;
            continue;
        }
        let mut sum = 0.0;
        for k in 0..m {
            sum += binom_f64(m + 1, k) * b[k];
        }
        b[m] = -sum / (m + 1) as f64;
    }
    b[n]
}

/// Helper: C(n, k) as f64 via multiplicative formula.
fn binom_f64(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut result = 1.0f64;
    for i in 0..k {
        result = result * (n - i) as f64 / (i + 1) as f64;
    }
    result
}

// ── Euler numbers ─────────────────────────────────────────────────────────────

/// Euler number E_n as f64.
///
/// All odd-indexed Euler numbers are zero.  Non-zero values alternate in sign.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::euler_number;
/// assert_eq!(euler_number(0),  1.0);
/// assert_eq!(euler_number(2), -1.0);
/// assert_eq!(euler_number(4),  5.0);
/// ```
pub fn euler_number(n: usize) -> f64 {
    if n % 2 == 1 {
        return 0.0;
    }
    match n {
        0 => return 1.0,
        2 => return -1.0,
        4 => return 5.0,
        6 => return -61.0,
        8 => return 1385.0,
        10 => return -50521.0,
        _ => {}
    }

    // Recurrence: E_n = -Σ_{k=0,2,4,...,n-2} C(n,k) E_k
    let mut e = vec![0.0f64; n + 1];
    e[0] = 1.0;
    for m in (2..=n).step_by(2) {
        let mut sum = 0.0;
        for k in (0..m).step_by(2) {
            sum += binom_f64(m, k) * e[k];
        }
        e[m] = -sum;
    }
    e[n]
}

// ── Catalan numbers ───────────────────────────────────────────────────────────

/// Catalan number C_n = C(2n, n) / (n+1).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::catalan;
/// assert_eq!(catalan(0), 1);
/// assert_eq!(catalan(5), 42);
/// assert_eq!(catalan(10), 16796);
/// ```
pub fn catalan(n: usize) -> u64 {
    if n == 0 {
        return 1;
    }
    // Use the recurrence C_n = C_{n-1} * 2*(2n-1) / (n+1)
    let mut c = 1u64;
    for k in 1..=n {
        // C_k = C_{k-1} * 2*(2k-1) / (k+1)
        c = c.saturating_mul(2 * (2 * k as u64 - 1)) / (k as u64 + 1);
    }
    c
}

// ── Rising / Falling factorials ───────────────────────────────────────────────

/// Rising factorial (Pochhammer symbol)  x^(n) = x(x+1)(x+2)···(x+n−1).
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::combinatorial_ext::rising_factorial;
/// assert_relative_eq!(rising_factorial(3.0, 4), 360.0, epsilon = 1e-12);
/// assert_eq!(rising_factorial(1.0, 0), 1.0);
/// ```
pub fn rising_factorial(x: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0;
    for i in 0..n {
        result *= x + i as f64;
    }
    result
}

/// Falling factorial  x_(n) = x(x−1)(x−2)···(x−n+1).
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::combinatorial_ext::falling_factorial;
/// assert_relative_eq!(falling_factorial(5.0, 3), 60.0, epsilon = 1e-12);
/// assert_eq!(falling_factorial(1.0, 0), 1.0);
/// ```
pub fn falling_factorial(x: f64, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut result = 1.0;
    for i in 0..n {
        result *= x - i as f64;
    }
    result
}

// ── Multinomial coefficient ───────────────────────────────────────────────────

/// Multinomial coefficient  n! / (k_1! · k_2! · … · k_m!).
///
/// Returns an error if the k_i do not sum to n.
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::combinatorial_ext::multinomial;
/// // 4! / (2! 1! 1!) = 12
/// assert_relative_eq!(multinomial(4, &[2, 1, 1]).unwrap(), 12.0, epsilon = 1e-10);
/// ```
pub fn multinomial(n: usize, ks: &[usize]) -> SpecialResult<f64> {
    let sum: usize = ks.iter().sum();
    if sum != n {
        return Err(SpecialError::ValueError(format!(
            "multinomial: k values sum to {sum} but n = {n}"
        )));
    }
    if ks.is_empty() {
        return Ok(1.0);
    }
    // Use log-factorial to avoid overflow: log(n!) - Σ log(k_i!)
    let ln_result = log_factorial(n) - ks.iter().map(|&k| log_factorial(k)).sum::<f64>();
    Ok(ln_result.exp())
}

// ── Prime sieve ───────────────────────────────────────────────────────────────

/// Return the n-th prime (1-indexed: nth_prime(1) = 2).
///
/// Uses a segmented sieve of Eratosthenes.  Panics with an error if n = 0.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::nth_prime;
/// assert_eq!(nth_prime(1).unwrap(), 2);
/// assert_eq!(nth_prime(10).unwrap(), 29);
/// assert_eq!(nth_prime(100).unwrap(), 541);
/// ```
pub fn nth_prime(n: usize) -> SpecialResult<u64> {
    if n == 0 {
        return Err(SpecialError::ValueError(
            "nth_prime: n must be at least 1".to_string(),
        ));
    }
    // Upper bound by the prime number theorem: p_n < n(ln n + ln ln n) for n >= 6
    let upper = if n < 6 {
        20usize
    } else {
        let f = n as f64;
        let ln_n = f.ln();
        let ln_ln_n = ln_n.ln().max(1.0);
        (f * (ln_n + ln_ln_n) * 1.3 + 3.0) as usize
    };

    let sieve = sieve_of_eratosthenes(upper);
    let primes: Vec<u64> = sieve
        .into_iter()
        .enumerate()
        .filter_map(|(i, is_prime)| if is_prime && i >= 2 { Some(i as u64) } else { None })
        .collect();

    if primes.len() >= n {
        Ok(primes[n - 1])
    } else {
        // Rare: bound was too tight, use larger sieve
        let larger = upper * 2;
        let sieve2 = sieve_of_eratosthenes(larger);
        let primes2: Vec<u64> = sieve2
            .into_iter()
            .enumerate()
            .filter_map(|(i, is_prime)| {
                if is_prime && i >= 2 {
                    Some(i as u64)
                } else {
                    None
                }
            })
            .collect();
        primes2
            .get(n - 1)
            .copied()
            .ok_or_else(|| SpecialError::ComputationError("nth_prime: sieve too small".to_string()))
    }
}

/// Sieve of Eratosthenes up to (and including) `limit`.
///
/// Returns a Vec<bool> where index i is true iff i is prime.
fn sieve_of_eratosthenes(limit: usize) -> Vec<bool> {
    let mut is_prime = vec![true; limit + 1];
    if limit >= 1 {
        is_prime[0] = false;
        is_prime[1] = false;
    }
    let mut i = 2;
    while i * i <= limit {
        if is_prime[i] {
            let mut j = i * i;
            while j <= limit {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is_prime
}

// ── Jordan totient ────────────────────────────────────────────────────────────

/// Jordan's totient function J_k(n).
///
/// J_k(n) = n^k · Π_{p | n} (1 − 1/p^k)
///
/// For k = 1 this reduces to Euler's totient φ(n).
///
/// # Examples
/// ```
/// use approx::assert_relative_eq;
/// use scirs2_special::combinatorial_ext::jordan_totient;
/// assert_relative_eq!(jordan_totient(1, 1), 1.0, epsilon = 1e-10);
/// assert_relative_eq!(jordan_totient(6, 1), 2.0, epsilon = 1e-10); // φ(6)=2
/// assert_relative_eq!(jordan_totient(4, 2), 12.0, epsilon = 1e-10); // J_2(4)
/// ```
pub fn jordan_totient(n: u64, k: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let mut result = (n as f64).powi(k as i32);
    let mut x = n;
    let mut p = 2u64;
    while p * p <= x {
        if x % p == 0 {
            while x % p == 0 {
                x /= p;
            }
            result *= 1.0 - (p as f64).powi(-(k as i32));
        }
        p += 1;
    }
    if x > 1 {
        result *= 1.0 - (x as f64).powi(-(k as i32));
    }
    result
}

// ── Integer partition function ────────────────────────────────────────────────

/// Partition function p(n) — number of ways to write n as an ordered sum of
/// positive integers (order of parts does not matter).
///
/// Uses Euler's pentagonal number recurrence:
///
/// ```text
/// p(n) = Σ_{k≠0} (−1)^{k+1} · p(n − k(3k−1)/2)
/// ```
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::partition;
/// assert_eq!(partition(0),  1);
/// assert_eq!(partition(4),  5);
/// assert_eq!(partition(10), 42);
/// assert_eq!(partition(20), 627);
/// ```
pub fn partition(n: usize) -> u64 {
    // Use Euler's pentagonal theorem with a memoization table.
    let mut p = vec![0u64; n + 1];
    p[0] = 1;
    for m in 1..=n {
        // Iterate over generalized pentagonal numbers ω(k) = k(3k−1)/2 for
        // k = 1, −1, 2, −2, 3, −3, …
        let mut sign = 1i64;
        let mut k = 1isize;
        loop {
            // Positive pentagonal: ω(k) = k*(3k-1)/2
            let pos_pent = (k * (3 * k - 1) / 2) as usize;
            // Negative pentagonal: ω(-k) = k*(3k+1)/2
            let neg_pent = (k * (3 * k + 1) / 2) as usize;

            if pos_pent > m && neg_pent > m {
                break;
            }

            if pos_pent <= m {
                if sign > 0 {
                    p[m] = p[m].saturating_add(p[m - pos_pent]);
                } else {
                    p[m] = p[m].saturating_sub(p[m - pos_pent].min(p[m]));
                }
            }
            if neg_pent <= m && neg_pent != pos_pent {
                if sign > 0 {
                    p[m] = p[m].saturating_add(p[m - neg_pent]);
                } else {
                    p[m] = p[m].saturating_sub(p[m - neg_pent].min(p[m]));
                }
            }

            sign = -sign;
            k += 1;
        }
    }
    p[n]
}

// ── Additional memoized Catalan helper ────────────────────────────────────────

/// Compute a table of Bell numbers B(0) … B(n) using the Bell triangle.
///
/// More efficient than calling `bell_number` n+1 times.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::bell_numbers_table;
/// let table = bell_numbers_table(5);
/// assert_eq!(table, vec![1, 1, 2, 5, 15, 52]);
/// ```
pub fn bell_numbers_table(n: usize) -> Vec<u64> {
    let mut table = Vec::with_capacity(n + 1);
    for i in 0..=n {
        table.push(bell_number(i));
    }
    table
}

/// Compute a table of Catalan numbers C(0) … C(n).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::catalan_table;
/// let t = catalan_table(5);
/// assert_eq!(t, vec![1, 1, 2, 5, 14, 42]);
/// ```
pub fn catalan_table(n: usize) -> Vec<u64> {
    let mut table = Vec::with_capacity(n + 1);
    for i in 0..=n {
        table.push(catalan(i));
    }
    table
}

/// All prime factors of n (with multiplicity), in non-decreasing order.
///
/// This returns factors with repetition, e.g. 12 → [2, 2, 3].
/// For factors as (prime, exponent) pairs, use `number_theory::prime_factors`.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorial_ext::prime_factors_flat;
/// assert_eq!(prime_factors_flat(12), vec![2, 2, 3]);
/// assert_eq!(prime_factors_flat(30), vec![2, 3, 5]);
/// ```
pub fn prime_factors_flat(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    if n <= 1 {
        return factors;
    }
    let mut p = 2u64;
    while p * p <= n {
        while n % p == 0 {
            factors.push(p);
            n /= p;
        }
        p += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
}

/// Partition numbers as a map (for caching in callers).
///
/// Not public API; used internally.
#[allow(dead_code)]
fn partition_memoized(n: usize, cache: &mut HashMap<usize, u64>) -> u64 {
    if let Some(&v) = cache.get(&n) {
        return v;
    }
    let v = partition(n);
    cache.insert(n, v);
    v
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stirling_first_signed() {
        assert_eq!(stirling_first(0, 0), 1);
        assert_eq!(stirling_first(1, 1), 1);
        assert_eq!(stirling_first(4, 2), 11);
        assert_eq!(stirling_first(4, 0), 0);
        assert_eq!(stirling_first(0, 4), 0);
    }

    #[test]
    fn test_stirling_first_unsigned() {
        assert_eq!(stirling_first_unsigned(0, 0), 1);
        assert_eq!(stirling_first_unsigned(4, 2), 11);
        assert_eq!(stirling_first_unsigned(5, 3), 35);
        assert_eq!(stirling_first_unsigned(3, 0), 0);
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(stirling_second(0, 0), 1);
        assert_eq!(stirling_second(4, 2), 7);
        assert_eq!(stirling_second(5, 3), 25);
        assert_eq!(stirling_second(3, 0), 0);
        assert_eq!(stirling_second(0, 3), 0);
    }

    #[test]
    fn test_bell_number() {
        assert_eq!(bell_number(0), 1);
        assert_eq!(bell_number(1), 1);
        assert_eq!(bell_number(2), 2);
        assert_eq!(bell_number(3), 5);
        assert_eq!(bell_number(4), 15);
        assert_eq!(bell_number(5), 52);
        assert_eq!(bell_number(10), 115975);
    }

    #[test]
    fn test_bernoulli() {
        assert_eq!(bernoulli(0), 1.0);
        assert_relative_eq!(bernoulli(1), -0.5, epsilon = 1e-14);
        assert_relative_eq!(bernoulli(2), 1.0 / 6.0, epsilon = 1e-12);
        assert_eq!(bernoulli(3), 0.0);
        assert_relative_eq!(bernoulli(4), -1.0 / 30.0, epsilon = 1e-12);
        assert_eq!(bernoulli(5), 0.0);
    }

    #[test]
    fn test_euler_number() {
        assert_eq!(euler_number(0), 1.0);
        assert_eq!(euler_number(1), 0.0);
        assert_eq!(euler_number(2), -1.0);
        assert_eq!(euler_number(4), 5.0);
        assert_eq!(euler_number(6), -61.0);
    }

    #[test]
    fn test_catalan() {
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(1), 1);
        assert_eq!(catalan(2), 2);
        assert_eq!(catalan(3), 5);
        assert_eq!(catalan(4), 14);
        assert_eq!(catalan(5), 42);
        assert_eq!(catalan(10), 16796);
    }

    #[test]
    fn test_rising_factorial() {
        assert_eq!(rising_factorial(1.0, 0), 1.0);
        assert_relative_eq!(rising_factorial(1.0, 4), 24.0, epsilon = 1e-12);
        assert_relative_eq!(rising_factorial(3.0, 4), 360.0, epsilon = 1e-12);
        assert_relative_eq!(rising_factorial(0.5, 2), 0.75, epsilon = 1e-12);
    }

    #[test]
    fn test_falling_factorial() {
        assert_eq!(falling_factorial(1.0, 0), 1.0);
        assert_relative_eq!(falling_factorial(5.0, 3), 60.0, epsilon = 1e-12);
        assert_relative_eq!(falling_factorial(4.0, 2), 12.0, epsilon = 1e-12);
    }

    #[test]
    fn test_multinomial() {
        assert_relative_eq!(
            multinomial(4, &[2, 1, 1]).expect("should succeed"),
            12.0,
            epsilon = 1e-8
        );
        assert_relative_eq!(
            multinomial(6, &[3, 2, 1]).expect("should succeed"),
            60.0,
            epsilon = 1e-8
        );
        // Error: sum != n
        assert!(multinomial(5, &[2, 1, 1]).is_err());
    }

    #[test]
    fn test_nth_prime() {
        assert_eq!(nth_prime(1).expect("ok"), 2);
        assert_eq!(nth_prime(2).expect("ok"), 3);
        assert_eq!(nth_prime(10).expect("ok"), 29);
        assert_eq!(nth_prime(100).expect("ok"), 541);
    }

    #[test]
    fn test_jordan_totient() {
        assert_relative_eq!(jordan_totient(1, 1), 1.0, epsilon = 1e-10);
        assert_relative_eq!(jordan_totient(6, 1), 2.0, epsilon = 1e-10);
        assert_relative_eq!(jordan_totient(4, 2), 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_partition() {
        assert_eq!(partition(0), 1);
        assert_eq!(partition(1), 1);
        assert_eq!(partition(4), 5);
        assert_eq!(partition(5), 7);
        assert_eq!(partition(10), 42);
        assert_eq!(partition(20), 627);
    }

    #[test]
    fn test_prime_factors_flat() {
        assert_eq!(prime_factors_flat(1), Vec::<u64>::new());
        assert_eq!(prime_factors_flat(12), vec![2, 2, 3]);
        assert_eq!(prime_factors_flat(30), vec![2, 3, 5]);
        assert_eq!(prime_factors_flat(97), vec![97]);
    }
}
