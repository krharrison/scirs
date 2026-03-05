//! Advanced combinatorial mathematics: partitions, permutations, Bell numbers,
//! Stirling numbers, Catalan numbers, derangements, Bernoulli and Euler numbers.
//!
//! This module uses wide integer types (`u128`/`i128`) to extend the range of
//! exact results compared to the `combinatorial_ext` module.

// ── Integer partition count ───────────────────────────────────────────────────

/// Number of integer partitions p(n) via Euler's pentagonal theorem.
///
/// Uses 128-bit arithmetic so exact results are available up to roughly n ≈ 90
/// (p(90) ≈ 5.6 × 10²³).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::partition_count;
/// assert_eq!(partition_count(0),  1);
/// assert_eq!(partition_count(10), 42);
/// assert_eq!(partition_count(20), 627);
/// ```
pub fn partition_count(n: usize) -> u128 {
    let mut p = vec![0u128; n + 1];
    p[0] = 1;
    for i in 1..=n {
        let mut k = 1i64;
        loop {
            let g1 = k * (3 * k - 1) / 2;
            if g1 as usize > i {
                break;
            }
            let g1u = g1 as usize;
            let sign_pos = k % 2 == 1;
            if sign_pos {
                p[i] = p[i].wrapping_add(p[i - g1u]);
            } else {
                p[i] = p[i].wrapping_sub(p[i - g1u]);
            }
            let g2 = k * (3 * k + 1) / 2;
            if g2 as usize <= i {
                let g2u = g2 as usize;
                if sign_pos {
                    p[i] = p[i].wrapping_add(p[i - g2u]);
                } else {
                    p[i] = p[i].wrapping_sub(p[i - g2u]);
                }
            }
            k += 1;
        }
    }
    p[n]
}

/// Generate all integer partitions of `n` in lexicographically decreasing order.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::partitions;
/// let parts = partitions(4);
/// assert_eq!(parts.len(), 5);
/// assert!(parts.contains(&vec![4]));
/// assert!(parts.contains(&vec![1, 1, 1, 1]));
/// ```
pub fn partitions(n: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    partitions_rec(n, n, &mut current, &mut result);
    result
}

fn partitions_rec(
    n: usize,
    max_part: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if n == 0 {
        result.push(current.clone());
        return;
    }
    for part in (1..=max_part.min(n)).rev() {
        current.push(part);
        partitions_rec(n - part, part, current, result);
        current.pop();
    }
}

// ── Bell numbers ─────────────────────────────────────────────────────────────

/// Bell number B(n): number of ways to partition a set of `n` elements.
///
/// Uses the Bell triangle algorithm.  Exact for `n` ≤ 25 with 128-bit
/// arithmetic (B(25) ≈ 4.6 × 10¹⁸).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::bell_number;
/// assert_eq!(bell_number(0), 1);
/// assert_eq!(bell_number(1), 1);
/// assert_eq!(bell_number(5), 52);
/// assert_eq!(bell_number(6), 203);
/// ```
pub fn bell_number(n: usize) -> u128 {
    if n == 0 {
        return 1;
    }
    // Bell triangle: each row starts with the last element of the previous row.
    let mut row = vec![0u128; n + 1];
    row[0] = 1; // B(0) = 1

    for _i in 1..=n {
        let mut new_row = vec![0u128; n + 1];
        // First element of new row = last element of previous row (= B(i-1 if i>=1))
        let prev_last = row[_i - 1];
        new_row[0] = prev_last;
        for j in 1..=_i {
            new_row[j] = new_row[j - 1].wrapping_add(row[j - 1]);
        }
        row = new_row;
    }
    row[0]
}

// ── Stirling numbers ─────────────────────────────────────────────────────────

/// Stirling number of the second kind S(n, k): ways to partition an n-element
/// set into exactly k non-empty subsets.
///
/// Uses the recurrence S(n, k) = k·S(n−1, k) + S(n−1, k−1).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::stirling_second;
/// assert_eq!(stirling_second(0, 0), 1);
/// assert_eq!(stirling_second(4, 2), 7);
/// assert_eq!(stirling_second(5, 3), 25);
/// ```
pub fn stirling_second(n: usize, k: usize) -> i128 {
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if k > n {
        return 0;
    }
    if k == n {
        return 1;
    }
    let mut dp = vec![vec![0i128; n + 1]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        for j in 1..=i {
            dp[i][j] = j as i128 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }
    dp[n][k]
}

/// Unsigned Stirling number of the first kind c(n, k): number of permutations
/// of n elements with exactly k disjoint cycles.
///
/// Uses the recurrence c(n, k) = (n−1)·c(n−1, k) + c(n−1, k−1).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::stirling_first;
/// assert_eq!(stirling_first(0, 0), 1);
/// assert_eq!(stirling_first(3, 2), 3);
/// assert_eq!(stirling_first(4, 2), 11);
/// ```
pub fn stirling_first(n: usize, k: usize) -> i128 {
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    if k > n {
        return 0;
    }
    if k == n {
        return 1;
    }
    let mut dp = vec![vec![0i128; n + 1]; n + 1];
    dp[0][0] = 1;
    for i in 1..=n {
        for j in 1..=i {
            dp[i][j] = (i as i128 - 1) * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }
    dp[n][k]
}

// ── Catalan numbers ───────────────────────────────────────────────────────────

/// Catalan number C(n) = binomial(2n, n) / (n+1).
///
/// Exact for n ≤ 33 with 128-bit arithmetic.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::catalan;
/// assert_eq!(catalan(0), 1);
/// assert_eq!(catalan(1), 1);
/// assert_eq!(catalan(5), 42);
/// assert_eq!(catalan(10), 16796);
/// ```
pub fn catalan(n: usize) -> u128 {
    if n == 0 {
        return 1;
    }
    let mut c = 1u128;
    for i in 0..n {
        c = c * (2 * (2 * i as u128 + 1)) / (i as u128 + 2);
    }
    c
}

// ── Derangements ─────────────────────────────────────────────────────────────

/// Number of derangements D(n): permutations of n elements with no fixed points.
///
/// Uses the recurrence D(n) = (n−1)·(D(n−2) + D(n−1)).
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::derangement;
/// assert_eq!(derangement(0), 1);
/// assert_eq!(derangement(1), 0);
/// assert_eq!(derangement(2), 1);
/// assert_eq!(derangement(4), 9);
/// assert_eq!(derangement(6), 265);
/// ```
pub fn derangement(n: usize) -> i128 {
    if n == 0 {
        return 1;
    }
    if n == 1 {
        return 0;
    }
    let mut dm2 = 1i128; // D(0)
    let mut dm1 = 0i128; // D(1)
    for k in 2..=n {
        let dk = (k as i128 - 1) * (dm2 + dm1);
        dm2 = dm1;
        dm1 = dk;
    }
    dm1
}

// ── Bernoulli numbers ─────────────────────────────────────────────────────────

/// Bernoulli number B(n) as a floating-point approximation.
///
/// Uses the Akiyama-Tanigawa algorithm.  B(1) = −1/2 by convention;
/// B(n) = 0 for all odd n > 1.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::bernoulli;
/// assert!((bernoulli(0) - 1.0).abs() < 1e-12);
/// assert!((bernoulli(2) - 1.0/6.0).abs() < 1e-12);
/// assert!(bernoulli(3).abs() < 1e-14);          // odd → 0
/// assert!((bernoulli(4) - (-1.0/30.0)).abs() < 1e-12);
/// ```
pub fn bernoulli(n: usize) -> f64 {
    if n == 1 {
        return -0.5;
    }
    if n % 2 == 1 {
        return 0.0;
    }
    // Akiyama-Tanigawa algorithm
    let mut a: Vec<f64> = (0..=n).map(|k| 1.0 / (k as f64 + 1.0)).collect();
    for m in 1..=n {
        for j in 0..=(n - m) {
            a[j] = (j as f64 + 1.0) * (a[j] - a[j + 1]);
        }
    }
    a[0]
}

// ── Euler numbers ─────────────────────────────────────────────────────────────

/// Euler number E(n).
///
/// E(n) = 0 for all odd n.  Exact for small even n using the recurrence
/// derived from the Taylor series of sech.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::euler_number;
/// assert_eq!(euler_number(0),  1);
/// assert_eq!(euler_number(2), -1);
/// assert_eq!(euler_number(4),  5);
/// assert_eq!(euler_number(6), -61);
/// ```
pub fn euler_number(n: usize) -> i128 {
    if n % 2 == 1 {
        return 0;
    }
    let mut e = vec![0i128; n + 1];
    e[0] = 1;
    for k in (2..=n).step_by(2) {
        for j in (0..k).step_by(2) {
            let binom = binomial_i128(k, j);
            e[k] -= binom * e[j];
        }
    }
    e[n]
}

// ── Helper: integer binomial coefficient ─────────────────────────────────────

/// Integer binomial coefficient C(n, k) as i128.
///
/// Returns 0 when k > n.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::binomial_i128;
/// assert_eq!(binomial_i128(5, 2), 10);
/// assert_eq!(binomial_i128(10, 3), 120);
/// assert_eq!(binomial_i128(3, 5), 0);
/// ```
pub fn binomial_i128(n: usize, k: usize) -> i128 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut c = 1i128;
    for i in 0..k {
        c = c * (n - i) as i128 / (i + 1) as i128;
    }
    c
}

// ── Sum-of-squares representations ───────────────────────────────────────────

/// Count the number of representations of `n` as a sum of exactly `k` perfect
/// squares (order and sign matter: r_k(n)).
///
/// Each summand may be any non-negative integer whose square does not exceed
/// `n`; the function counts ordered tuples (a_1, …, a_k) with a_i ≥ 0 and
/// a_1² + … + a_k² = n.
///
/// # Examples
/// ```
/// use scirs2_special::combinatorics::sum_of_squares_count;
/// // n=1, k=2: (±1,0),(0,±1) → 4 representations
/// assert_eq!(sum_of_squares_count(1, 2), 4);
/// // n=0, k=3: only (0,0,0)
/// assert_eq!(sum_of_squares_count(0, 3), 1);
/// ```
pub fn sum_of_squares_count(n: usize, k: usize) -> usize {
    if k == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    let sqrt_n = (n as f64).sqrt() as usize + 1;
    let mut count = 0usize;
    sum_of_squares_rec(n, k, sqrt_n, &mut count);
    count
}

fn sum_of_squares_rec(remaining: usize, squares_left: usize, max_s: usize, count: &mut usize) {
    if squares_left == 0 {
        if remaining == 0 {
            *count += 1;
        }
        return;
    }
    // Each summand s can be −max_s … max_s; count signed representations.
    let limit = remaining.min(max_s * max_s);
    let s_max = (limit as f64).sqrt() as usize + 1;
    for s in 0..=s_max.min(max_s) {
        let sq = s * s;
        if sq > remaining {
            break;
        }
        // Multiplicity: s=0 contributes once; s>0 contributes twice (±s).
        let mult = if s == 0 { 1 } else { 2 };
        let mut sub_count = 0usize;
        sum_of_squares_rec(remaining - sq, squares_left - 1, max_s, &mut sub_count);
        *count += mult * sub_count;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_count() {
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
        assert_eq!(partition_count(4), 5);
        assert_eq!(partition_count(10), 42);
        assert_eq!(partition_count(20), 627);
    }

    #[test]
    fn test_partitions_list() {
        let p4 = partitions(4);
        assert_eq!(p4.len(), 5);
        assert!(p4.contains(&vec![4]));
        assert!(p4.contains(&vec![3, 1]));
        assert!(p4.contains(&vec![2, 2]));
        assert!(p4.contains(&vec![2, 1, 1]));
        assert!(p4.contains(&vec![1, 1, 1, 1]));
    }

    #[test]
    fn test_bell_numbers() {
        assert_eq!(bell_number(0), 1);
        assert_eq!(bell_number(1), 1);
        assert_eq!(bell_number(2), 2);
        assert_eq!(bell_number(3), 5);
        assert_eq!(bell_number(4), 15);
        assert_eq!(bell_number(5), 52);
        assert_eq!(bell_number(6), 203);
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(stirling_second(0, 0), 1);
        assert_eq!(stirling_second(1, 1), 1);
        assert_eq!(stirling_second(2, 1), 1);
        assert_eq!(stirling_second(4, 2), 7);
        assert_eq!(stirling_second(5, 3), 25);
    }

    #[test]
    fn test_stirling_first() {
        assert_eq!(stirling_first(0, 0), 1);
        assert_eq!(stirling_first(1, 1), 1);
        assert_eq!(stirling_first(3, 2), 3);
        assert_eq!(stirling_first(4, 2), 11);
    }

    #[test]
    fn test_catalan() {
        assert_eq!(catalan(0), 1);
        assert_eq!(catalan(1), 1);
        assert_eq!(catalan(2), 2);
        assert_eq!(catalan(3), 5);
        assert_eq!(catalan(5), 42);
        assert_eq!(catalan(10), 16796);
    }

    #[test]
    fn test_derangement() {
        assert_eq!(derangement(0), 1);
        assert_eq!(derangement(1), 0);
        assert_eq!(derangement(2), 1);
        assert_eq!(derangement(3), 2);
        assert_eq!(derangement(4), 9);
        assert_eq!(derangement(6), 265);
    }

    #[test]
    fn test_bernoulli() {
        assert!((bernoulli(0) - 1.0).abs() < 1e-12);
        assert!((bernoulli(1) - (-0.5)).abs() < 1e-12);
        assert!((bernoulli(2) - 1.0 / 6.0).abs() < 1e-10);
        assert!(bernoulli(3).abs() < 1e-14);
        assert!((bernoulli(4) - (-1.0 / 30.0)).abs() < 1e-10);
    }

    #[test]
    fn test_euler_number() {
        assert_eq!(euler_number(0), 1);
        assert_eq!(euler_number(1), 0);
        assert_eq!(euler_number(2), -1);
        assert_eq!(euler_number(4), 5);
        assert_eq!(euler_number(6), -61);
    }

    #[test]
    fn test_binomial_i128() {
        assert_eq!(binomial_i128(0, 0), 1);
        assert_eq!(binomial_i128(5, 2), 10);
        assert_eq!(binomial_i128(10, 3), 120);
        assert_eq!(binomial_i128(3, 5), 0);
    }

    #[test]
    fn test_sum_of_squares_count() {
        // n=0, k=0: 1 (empty sum = 0)
        assert_eq!(sum_of_squares_count(0, 0), 1);
        // n=1, k=0: impossible
        assert_eq!(sum_of_squares_count(1, 0), 0);
        // n=0, k=3: (0,0,0) only
        assert_eq!(sum_of_squares_count(0, 3), 1);
        // n=1, k=2: (±1,0), (0,±1) → 4
        assert_eq!(sum_of_squares_count(1, 2), 4);
    }
}
