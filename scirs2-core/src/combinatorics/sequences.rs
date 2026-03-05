//! Mathematical integer sequences used in combinatorics.
//!
//! All functions are implemented in pure Rust with no `unwrap()` calls.
//! Where results can overflow `u64` / `i64`, the functions return
//! `Option<T>` or `u128` variants.

/// Compute the n-th Fibonacci number F(n) using the fast-doubling algorithm.
///
/// F(0) = 0, F(1) = 1, F(2) = 1, F(3) = 2, …
///
/// Returns `None` on overflow (n ≥ 94 overflows `u64`).
///
/// # Algorithm
///
/// The fast-doubling identities are:
///   F(2k)   = F(k) * (2*F(k+1) - F(k))
///   F(2k+1) = F(k)^2 + F(k+1)^2
///
/// This runs in O(log n) arithmetic operations.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::fibonacci;
/// assert_eq!(fibonacci(10), Some(55));
/// assert_eq!(fibonacci(0), Some(0));
/// assert_eq!(fibonacci(1), Some(1));
/// ```
pub fn fibonacci(n: u64) -> Option<u64> {
    fibonacci_pair(n).map(|(f, _)| f)
}

/// Return (F(n), F(n+1)) using the fast-doubling algorithm.
///
/// Returns `None` if either value overflows `u64`.
pub fn fibonacci_pair(n: u64) -> Option<(u64, u64)> {
    if n == 0 {
        return Some((0, 1));
    }
    let (f, f1) = fibonacci_pair(n / 2)?;
    // c = F(k) * (2*F(k+1) - F(k))
    // d = F(k)^2 + F(k+1)^2
    let two_f1 = f1.checked_mul(2)?;
    let two_f1_minus_f = two_f1.checked_sub(f)?;
    let c = f.checked_mul(two_f1_minus_f)?;
    let d = f.checked_mul(f)?.checked_add(f1.checked_mul(f1)?)?;
    if n % 2 == 0 {
        Some((c, d))
    } else {
        Some((d, c.checked_add(d)?))
    }
}

/// Compute the n-th Fibonacci number as a `u128` (valid for n ≤ 186).
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::fibonacci_u128;
/// assert_eq!(fibonacci_u128(93), Some(12200160415121876738));
/// ```
pub fn fibonacci_u128(n: u64) -> Option<u128> {
    fibonacci_pair_u128(n).map(|(f, _)| f)
}

/// Return (F(n), F(n+1)) as `u128`.
pub fn fibonacci_pair_u128(n: u64) -> Option<(u128, u128)> {
    if n == 0 {
        return Some((0, 1));
    }
    let (f, f1) = fibonacci_pair_u128(n / 2)?;
    let two_f1 = f1.checked_mul(2)?;
    let two_f1_minus_f = two_f1.checked_sub(f)?;
    let c = f.checked_mul(two_f1_minus_f)?;
    let d = f.checked_mul(f)?.checked_add(f1.checked_mul(f1)?)?;
    if n % 2 == 0 {
        Some((c, d))
    } else {
        Some((d, c.checked_add(d)?))
    }
}

/// Compute the n-th Lucas number L(n).
///
/// L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2).
///
/// Uses the identity L(n) = F(n-1) + F(n+1) = 2*F(n+1) - F(n).
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::lucas_number;
/// assert_eq!(lucas_number(0), Some(2));
/// assert_eq!(lucas_number(1), Some(1));
/// assert_eq!(lucas_number(6), Some(18));
/// ```
pub fn lucas_number(n: u64) -> Option<u64> {
    // L(n) = F(n-1) + F(n+1)
    // Using L(n) = 2*F(n+1) - F(n):
    let (f, f1) = fibonacci_pair(n)?;
    let two_f1 = f1.checked_mul(2)?;
    two_f1.checked_sub(f)
}

/// Compute the n-th Catalan number C(n).
///
/// C(n) = C(2n, n) / (n+1) = (2n)! / ((n+1)! * n!)
///
/// Computed iteratively to avoid large intermediate factorials.
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::catalan_number;
/// assert_eq!(catalan_number(0), Some(1));
/// assert_eq!(catalan_number(5), Some(42));
/// assert_eq!(catalan_number(10), Some(16796));
/// ```
pub fn catalan_number(n: u64) -> Option<u64> {
    // C(n) = product_{k=2}^{n} (n+k)/k  for n >= 1, C(0)=1.
    if n == 0 {
        return Some(1);
    }
    // Use the recurrence via multiplication/division staying in u128.
    // C(n) = C(n-1) * 2*(2n-1) / (n+1)
    let mut c: u128 = 1;
    for k in 1..=n {
        // C(k) = C(k-1) * 2*(2k-1) / (k+1)
        let num = 2 * (2 * k as u128 - 1);
        let den = k as u128 + 1;
        // Multiply first, then divide (the result is always an integer).
        c = c.checked_mul(num)? / den;
    }
    u64::try_from(c).ok()
}

/// Compute the n-th Bell number B(n) via the Bell triangle.
///
/// B(0) = 1, B(1) = 1, B(2) = 2, B(3) = 5, …
///
/// The Bell triangle approach builds successive rows; this uses O(n) memory
/// per row and runs in O(n²) time.
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::bell_number;
/// assert_eq!(bell_number(0), Some(1));
/// assert_eq!(bell_number(5), Some(52));
/// ```
pub fn bell_number(n: usize) -> Option<u64> {
    if n == 0 {
        return Some(1);
    }
    // The Bell triangle:
    // Row 0: [B(0)] = [1]
    // Row k+1: start with the last element of row k, then each subsequent
    //          element is the sum of the previous element in the same row
    //          and the element directly above it.
    // B(k+1) = first element of row k+1 = last element of row k.
    let mut row: Vec<u128> = vec![1]; // Row 0
    for _row_idx in 0..n {
        let mut next_row = Vec::with_capacity(row.len() + 1);
        // First element of next row = last element of current row.
        let first = *row.last().expect("row is non-empty");
        next_row.push(first);
        for prev in &row {
            let last = *next_row.last().expect("next_row is non-empty");
            next_row.push(last.checked_add(*prev)?);
        }
        row = next_row;
    }
    // B(n) is the first element of the n-th row.
    let bell = *row.first().expect("row is non-empty");
    u64::try_from(bell).ok()
}

/// Compute the (signed) Stirling number of the first kind s(n, k).
///
/// s(n, k) counts the number of permutations of n elements with exactly k
/// disjoint cycles (with sign: s(n,k) = (-1)^{n-k} * |s(n,k)|).
///
/// Uses the recurrence:
///   s(n, k) = s(n-1, k-1) - (n-1) * s(n-1, k)
///
/// with s(0,0)=1 and s(n,0)=0 for n>0, s(0,k)=0 for k>0.
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::stirling_first;
/// assert_eq!(stirling_first(4, 2), Some(-11));
/// assert_eq!(stirling_first(4, 4), Some(1));
/// assert_eq!(stirling_first(0, 0), Some(1));
/// ```
pub fn stirling_first(n: usize, k: usize) -> Option<i64> {
    if k > n {
        return Some(0);
    }
    // Build the table row by row; we only need the current row.
    let mut row = vec![0i64; n + 1];
    row[0] = 1; // s(0, 0) = 1
    for i in 1..=n {
        let mut new_row = vec![0i64; n + 1];
        for j in 1..=i {
            // s(i, j) = s(i-1, j-1) - (i-1) * s(i-1, j)
            let term1 = row[j - 1];
            let term2 = (i as i64 - 1).checked_mul(row[j])?;
            new_row[j] = term1.checked_sub(term2)?;
        }
        row = new_row;
    }
    Some(row[k])
}

/// Compute the unsigned Stirling number of the first kind |s(n, k)|.
///
/// |s(n, k)| counts the number of permutations of {1, …, n} with exactly k
/// disjoint cycles (ignoring sign).
///
/// Returns `None` on overflow.
pub fn stirling_first_unsigned(n: usize, k: usize) -> Option<u64> {
    stirling_first(n, k).map(|v| v.unsigned_abs())
}

/// Compute the Stirling number of the second kind S(n, k).
///
/// S(n, k) counts the number of ways to partition a set of n elements into
/// exactly k non-empty subsets.
///
/// Uses the recurrence:
///   S(n, k) = k * S(n-1, k) + S(n-1, k-1)
///
/// with S(0,0)=1 and S(n,0)=0 for n>0, S(0,k)=0 for k>0.
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::stirling_second;
/// assert_eq!(stirling_second(4, 2), Some(7));
/// assert_eq!(stirling_second(4, 4), Some(1));
/// assert_eq!(stirling_second(0, 0), Some(1));
/// ```
pub fn stirling_second(n: usize, k: usize) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    if k == 0 {
        return if n == 0 { Some(1) } else { Some(0) };
    }
    // Build the table row by row.
    let mut row = vec![0u128; n + 1];
    row[0] = 1; // S(0, 0) = 1
    for i in 1..=n {
        let mut new_row = vec![0u128; n + 1];
        for j in 1..=i {
            // S(i, j) = j * S(i-1, j) + S(i-1, j-1)
            let term1 = (j as u128).checked_mul(row[j])?;
            let term2 = row[j - 1];
            new_row[j] = term1.checked_add(term2)?;
        }
        row = new_row;
    }
    u64::try_from(row[k]).ok()
}

/// Compute the number of derangements D(n) (subfactorial !n).
///
/// A derangement is a permutation with no fixed points.
///
/// D(0) = 1, D(1) = 0, D(n) = (n-1) * (D(n-2) + D(n-1)).
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::derangement_count;
/// assert_eq!(derangement_count(0), Some(1));
/// assert_eq!(derangement_count(1), Some(0));
/// assert_eq!(derangement_count(4), Some(9));
/// assert_eq!(derangement_count(5), Some(44));
/// ```
pub fn derangement_count(n: usize) -> Option<u64> {
    match n {
        0 => Some(1),
        1 => Some(0),
        _ => {
            let mut d_prev2: u128 = 1; // D(0)
            let mut d_prev1: u128 = 0; // D(1)
            for i in 2..=n {
                let d = ((i as u128) - 1).checked_mul(d_prev2.checked_add(d_prev1)?)?;
                d_prev2 = d_prev1;
                d_prev1 = d;
            }
            u64::try_from(d_prev1).ok()
        }
    }
}

/// Compute the Euler number E(n) (for even n).
///
/// The Euler numbers satisfy the Taylor expansion of sec(x).
/// E(0)=1, E(2)=-1, E(4)=5, E(6)=-61, …; odd Euler numbers are zero.
///
/// Returns `None` on overflow or when n is odd and non-zero.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::euler_number;
/// assert_eq!(euler_number(0), Some(1));
/// assert_eq!(euler_number(2), Some(-1));
/// assert_eq!(euler_number(4), Some(5));
/// ```
pub fn euler_number(n: usize) -> Option<i64> {
    if n % 2 != 0 {
        return Some(0);
    }
    // Compute via the recurrence on the Euler triangle / secant series.
    // E(n) = -sum_{k=0}^{n/2-1} C(n, 2k) * E(2k)  for n >= 2 even.
    let half = n / 2;
    let mut e: Vec<i128> = vec![0; half + 1];
    e[0] = 1; // E(0) = 1
    for m in 1..=half {
        let two_m = 2 * m;
        let mut s: i128 = 0;
        // binomial(2m, 2k) * E(2k) for k = 0 .. m-1
        // We compute binomial coefficients iteratively.
        let mut binom: i128 = 1;
        for k in 0..m {
            s = s.checked_add(binom.checked_mul(e[k])?)?;
            // Advance binom: C(2m, 2k) → C(2m, 2(k+1))
            // C(2m, 2k+2) = C(2m, 2k) * (2m-2k) * (2m-2k-1) / ((2k+1) * (2k+2))
            let num = (two_m as i128 - 2 * k as i128)
                .checked_mul(two_m as i128 - 2 * k as i128 - 1)?;
            let den = (2 * k as i128 + 1).checked_mul(2 * k as i128 + 2)?;
            binom = binom.checked_mul(num)? / den;
        }
        e[m] = s.checked_neg()?;
    }
    i64::try_from(e[half]).ok()
}

/// Compute the n-th Bernoulli number numerator (as `i64`) and denominator
/// (as `u64`) for B(n).
///
/// Uses the recurrence:
///   B(0) = 1
///   B(n) = -1/(n+1) * sum_{k=0}^{n-1} C(n+1, k) * B(k)
///
/// This returns the result in exact rational form (numerator, denominator)
/// using i128 arithmetic.
///
/// Returns `None` on overflow.
///
/// # Example
///
/// ```
/// use scirs2_core::combinatorics::sequences::bernoulli_fraction;
/// // B(0) = 1/1, B(1) = -1/2, B(2) = 1/6
/// assert_eq!(bernoulli_fraction(0), Some((1, 1)));
/// assert_eq!(bernoulli_fraction(1), Some((-1, 2)));
/// assert_eq!(bernoulli_fraction(2), Some((1, 6)));
/// ```
pub fn bernoulli_fraction(n: usize) -> Option<(i64, u64)> {
    // We track each B(k) as (numerator, denominator) in reduced form.
    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            a %= b;
            std::mem::swap(&mut a, &mut b);
        }
        a
    }
    fn reduce(num: i64, den: u64) -> (i64, u64) {
        if num == 0 {
            return (0, 1);
        }
        let g = gcd(num.unsigned_abs(), den);
        (num / g as i64, den / g)
    }

    let mut b_num = vec![0i128; n + 1];
    let mut b_den = vec![1u128; n + 1];
    b_num[0] = 1;
    b_den[0] = 1;

    for m in 1..=n {
        // B(m) = -1/(m+1) * sum_{k=0}^{m-1} C(m+1, k) * B(k)
        let mut s_num: i128 = 0;
        let mut s_den: u128 = 1;
        let mut binom: u128 = 1; // C(m+1, 0) = 1
        for k in 0..m {
            // Add binom * B(k) to the running sum (rational arithmetic).
            // term = binom * b_num[k] / b_den[k]
            let term_num = binom as i128 * b_num[k];
            let term_den = b_den[k];
            // s + term: s_num/s_den + term_num/term_den
            s_num = s_num * term_den as i128 + term_num * s_den as i128;
            s_den = s_den.checked_mul(term_den)?;
            // Reduce to keep numbers manageable.
            let g = {
                let a = s_num.unsigned_abs() as u128;
                let b = s_den;
                let mut aa = a;
                let mut bb = b;
                while bb != 0 {
                    aa %= bb;
                    std::mem::swap(&mut aa, &mut bb);
                }
                aa
            };
            if g > 1 {
                s_num /= g as i128;
                s_den /= g;
            }
            // Advance binomial C(m+1, k) → C(m+1, k+1)
            binom = binom.checked_mul((m as u128 + 1 - k as u128))? / (k as u128 + 1);
        }
        // B(m) = -s / (m+1)
        b_num[m] = -s_num;
        b_den[m] = s_den.checked_mul(m as u128 + 1)?;
        // Reduce.
        let g = {
            let a = b_num[m].unsigned_abs() as u128;
            let b = b_den[m];
            let mut aa = a;
            let mut bb = b;
            while bb != 0 {
                aa %= bb;
                std::mem::swap(&mut aa, &mut bb);
            }
            aa
        };
        if g > 1 {
            b_num[m] /= g as i128;
            b_den[m] /= g;
        }
    }
    let num = i64::try_from(b_num[n]).ok()?;
    let den = u64::try_from(b_den[n]).ok()?;
    Some(reduce(num, den))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_small() {
        let expected = [0u64, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                fibonacci(i as u64),
                Some(exp),
                "F({i}) should be {exp}"
            );
        }
    }

    #[test]
    fn test_fibonacci_93() {
        // F(93) = 12200160415121876738, last value fitting u64.
        assert_eq!(fibonacci(93), Some(12200160415121876738u64));
    }

    #[test]
    fn test_fibonacci_overflow() {
        // F(94) overflows u64.
        assert!(fibonacci(94).is_none());
    }

    #[test]
    fn test_lucas() {
        let expected = [2u64, 1, 3, 4, 7, 11, 18, 29, 47];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(lucas_number(i as u64), Some(exp), "L({i}) should be {exp}");
        }
    }

    #[test]
    fn test_catalan() {
        let expected = [1u64, 1, 2, 5, 14, 42, 132, 429, 1430, 4862];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                catalan_number(i as u64),
                Some(exp),
                "C({i}) should be {exp}"
            );
        }
    }

    #[test]
    fn test_bell() {
        let expected = [1u64, 1, 2, 5, 15, 52, 203, 877, 4140, 21147];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(bell_number(i), Some(exp), "B({i}) should be {exp}");
        }
    }

    #[test]
    fn test_stirling_first() {
        // Verify a few values from a reference table.
        assert_eq!(stirling_first(0, 0), Some(1));
        assert_eq!(stirling_first(1, 1), Some(1));
        assert_eq!(stirling_first(2, 1), Some(-1));
        assert_eq!(stirling_first(3, 2), Some(3));
        assert_eq!(stirling_first(4, 2), Some(-11));
        assert_eq!(stirling_first(4, 3), Some(6));
        assert_eq!(stirling_first(4, 4), Some(1));
        // k > n → 0
        assert_eq!(stirling_first(2, 5), Some(0));
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(stirling_second(0, 0), Some(1));
        assert_eq!(stirling_second(1, 1), Some(1));
        assert_eq!(stirling_second(3, 2), Some(3));
        assert_eq!(stirling_second(4, 2), Some(7));
        assert_eq!(stirling_second(4, 4), Some(1));
        assert_eq!(stirling_second(5, 3), Some(25));
        // k > n → 0
        assert_eq!(stirling_second(2, 5), Some(0));
    }

    #[test]
    fn test_derangements() {
        let expected = [1u64, 0, 1, 2, 9, 44, 265, 1854, 14833];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(
                derangement_count(i),
                Some(exp),
                "D({i}) should be {exp}"
            );
        }
    }

    #[test]
    fn test_euler_numbers() {
        assert_eq!(euler_number(0), Some(1));
        assert_eq!(euler_number(2), Some(-1));
        assert_eq!(euler_number(4), Some(5));
        assert_eq!(euler_number(6), Some(-61));
        // Odd index returns 0.
        assert_eq!(euler_number(3), Some(0));
    }

    #[test]
    fn test_bernoulli() {
        assert_eq!(bernoulli_fraction(0), Some((1, 1)));
        assert_eq!(bernoulli_fraction(1), Some((-1, 2)));
        assert_eq!(bernoulli_fraction(2), Some((1, 6)));
        // B(3) = 0
        assert_eq!(bernoulli_fraction(3), Some((0, 1)));
        // B(4) = -1/30
        assert_eq!(bernoulli_fraction(4), Some((-1, 30)));
    }
}
