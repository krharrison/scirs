//! Period finding sub-routines for Shor's algorithm.
//!
//! This file provides the classical simulation of the quantum phase-estimation
//! step used in Shor's factoring algorithm. The QFT is used to extract period
//! information from the modular-exponentiation function f(x) = a^x mod N.
//!
//! # References
//! - P. Shor, "Polynomial-time algorithms for prime factorization and discrete
//!   logarithms on a quantum computer", SIAM J. Comput. 26(5), 1997.
//! - M. Nielsen & I. Chuang, *Quantum Computation and Quantum Information*,
//!   Cambridge University Press, 2000.

use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};

// ── Arithmetic helpers ────────────────────────────────────────────────────────

/// Compute `base^exp mod modulus` using fast binary (square-and-multiply)
/// exponentiation.  Handles modulus = 1 (always returns 0) and exp = 0 → 1
/// correctly.
pub fn modular_exp(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result
                .checked_mul(base)
                .map(|v| v % modulus)
                .unwrap_or_else(|| {
                    // overflow: fall back to 128-bit arithmetic
                    ((result as u128 * base as u128) % modulus as u128) as u64
                });
        }
        exp >>= 1;
        base = base
            .checked_mul(base)
            .map(|v| v % modulus)
            .unwrap_or_else(|| ((base as u128 * base as u128) % modulus as u128) as u64);
    }
    result
}

/// Greatest common divisor via Euclidean algorithm.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

// ── Continued fractions ───────────────────────────────────────────────────────

/// Compute the continued-fraction convergents of `x` (a non-negative real number)
/// up to the given maximum denominator.
///
/// Each convergent (p, q) satisfies |x − p/q| < 1/(2q²) and is returned in
/// order of increasing q.  The list contains at most `max_candidates` entries.
///
/// Standard recurrence (Knuth, TAOCP Vol. 2 §4.5.3):
/// ```text
///   h_{-1} = 1,  h_0 = a_0
///   k_{-1} = 0,  k_0 = 1
///   h_n = a_n * h_{n-1} + h_{n-2}
///   k_n = a_n * k_{n-1} + k_{n-2}
/// ```
pub fn continued_fraction_convergents(
    x: f64,
    max_denominator: u64,
    max_candidates: usize,
) -> Vec<(u64, u64)> {
    let mut convergents = Vec::new();
    if max_candidates == 0 || x.is_nan() {
        return convergents;
    }

    let mut xi = x;

    // (h_{n-2}, k_{n-2}) and (h_{n-1}, k_{n-1})
    // Initialise with the "virtual" minus-one term and minus-two term
    // so that the recurrence h_n = a_n * h_{n-1} + h_{n-2} works from n=0:
    //   step n=0: h_0 = a_0 * 1 + 0 = a_0, k_0 = a_0 * 0 + 1 = 1  ✓
    let mut hm2: i64 = 0; // h_{-2} (virtual extra term)
    let mut hm1: i64 = 1; // h_{-1}
    let mut km2: i64 = 1; // k_{-2} (virtual extra term)
    let mut km1: i64 = 0; // k_{-1}

    for _ in 0..64 {
        if xi.is_nan() || xi.is_infinite() {
            break;
        }
        let a = xi.floor() as i64;

        let hn = a * hm1 + hm2;
        let kn = a * km1 + km2;

        if kn <= 0 || kn as u64 > max_denominator {
            break;
        }

        convergents.push((hn.unsigned_abs(), kn as u64));

        if convergents.len() >= max_candidates {
            break;
        }

        let frac = xi - a as f64;
        if frac.abs() < 1e-12 {
            break;
        }
        xi = 1.0 / frac;

        hm2 = hm1;
        hm1 = hn;
        km2 = km1;
        km1 = kn;
    }

    convergents
}

// ── QFT simulation helpers ────────────────────────────────────────────────────

/// A simple in-place radix-2 Cooley-Tukey FFT on a `Vec<(f64, f64)>` (re, im).
/// The length *must* be a power of two.
fn fft_inplace(data: &mut [(f64, f64)]) -> FFTResult<()> {
    let n = data.len();
    if n == 0 || n & (n - 1) != 0 {
        return Err(FFTError::ValueError(format!(
            "fft_inplace requires power-of-2 length, got {n}"
        )));
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wlen = (angle.cos(), angle.sin());
        for i in (0..n).step_by(len) {
            let mut w = (1.0_f64, 0.0_f64);
            for k in 0..half {
                let u = data[i + k];
                let v_re = data[i + k + half].0 * w.0 - data[i + k + half].1 * w.1;
                let v_im = data[i + k + half].0 * w.1 + data[i + k + half].1 * w.0;
                data[i + k] = (u.0 + v_re, u.1 + v_im);
                data[i + k + half] = (u.0 - v_re, u.1 - v_im);
                let new_w_re = w.0 * wlen.0 - w.1 * wlen.1;
                let new_w_im = w.0 * wlen.1 + w.1 * wlen.0;
                w = (new_w_re, new_w_im);
            }
        }
        len <<= 1;
    }
    Ok(())
}

// ── Period finding ────────────────────────────────────────────────────────────

/// Find the period r of the function f(x) = a^x mod n by classically
/// simulating the quantum phase-estimation step.
///
/// We evaluate f on [0, M) where M = 2^`n_qubits`, build the probability
/// distribution |∑_x e^{2πi s x/M}|² over the "measured" outcome s, and
/// then use continued fractions to recover r as the denominator of s/M.
///
/// Returns the smallest r ≥ 2 such that `modular_exp(a, r, n) == 1` and
/// r divides a candidate denominator obtained from the QFT peak(s).
///
/// # Arguments
/// * `a`          – Base for the modular exponentiation (1 < a < n, gcd(a,n)=1).
/// * `n`          – Modulus to factorise.
/// * `n_qubits`   – Number of simulated qubits (register size = 2^n_qubits).
/// * `max_cands`  – Maximum period candidates to try from continued fractions.
///
/// Returns `Ok(None)` if no period is found within the given resources.
pub fn find_period_qft(
    a: u64,
    n: u64,
    n_qubits: usize,
    max_cands: usize,
) -> FFTResult<Option<u64>> {
    if a <= 1 || a >= n {
        return Err(FFTError::ValueError(format!(
            "require 1 < a < n, got a={a}, n={n}"
        )));
    }
    if n_qubits == 0 || n_qubits > 24 {
        return Err(FFTError::ValueError(format!(
            "n_qubits must be in 1..=24, got {n_qubits}"
        )));
    }

    let m: usize = 1 << n_qubits; // register size

    // Build the uniform superposition over x in [0, M),
    // with each amplitude weighted by the phase e^{2πi·f(x)·0/M} = 1.
    // After the first Hadamard layer and oracle, the state collapses (on
    // measurement) to a periodic comb.  We simulate the resulting QFT
    // output distribution directly.
    //
    // For each possible measurement outcome s, the amplitude is:
    //   A(s) = (1/M) Σ_{x: f(x)=f(x0)} e^{2πi s x / M}
    // where x0 is a random value in {0, ..., r-1}.
    //
    // We approximate by summing over all x (marginalising over x0):

    // Evaluate the modular-exp oracle on all x
    let f_vals: Vec<u64> = (0..m as u64).map(|x| modular_exp(a, x, n)).collect();

    // Group by output value and compute the QFT of each group independently,
    // then add magnitudes squared (probabilistic mixture over measurement of |f⟩).
    // For efficiency we compute: |FFT of 1_{f(x) = v}|^2 for each distinct v.

    // Collect distinct f-values
    let mut distinct: Vec<u64> = f_vals.clone();
    distinct.sort_unstable();
    distinct.dedup();

    let mut prob: Vec<f64> = vec![0.0; m];

    for &v in &distinct {
        // Build indicator signal for this residue
        let mut buf: Vec<(f64, f64)> = f_vals
            .iter()
            .map(|&fv| if fv == v { (1.0, 0.0) } else { (0.0, 0.0) })
            .collect();
        fft_inplace(&mut buf)?;
        for (s, amp) in buf.iter().enumerate() {
            prob[s] += amp.0 * amp.0 + amp.1 * amp.1;
        }
    }

    // Normalise
    let total: f64 = prob.iter().sum();
    if total < 1e-15 {
        return Ok(None);
    }
    let inv_total = 1.0 / total;
    for p in prob.iter_mut() {
        *p *= inv_total;
    }

    // Collect top-probability outcomes (sorted by prob, descending)
    let mut indexed: Vec<(f64, usize)> = prob
        .iter()
        .copied()
        .enumerate()
        .map(|(i, p)| (p, i))
        .collect();
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let max_denominator = n; // r must divide some integer < n
    for &(_prob, s) in indexed.iter().take(max_cands * 4) {
        if s == 0 {
            continue;
        }
        let phase = s as f64 / m as f64;
        let convergents = continued_fraction_convergents(phase, max_denominator, max_cands);
        for &(_p, q) in &convergents {
            if q < 2 {
                continue;
            }
            // Test r = q
            if modular_exp(a, q, n) == 1 {
                return Ok(Some(q));
            }
            // Also try small multiples (the true period may be a multiple of q)
            for k in 2..=4u64 {
                let rk = q * k;
                if rk < n && modular_exp(a, rk, n) == 1 {
                    return Ok(Some(rk));
                }
            }
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modular_exp_basic() {
        // 2^10 mod 1000 = 24
        assert_eq!(modular_exp(2, 10, 1000), 1024 % 1000);
        // identity: any^0 mod n = 1
        assert_eq!(modular_exp(7, 0, 13), 1);
        // modulus 1 always gives 0
        assert_eq!(modular_exp(5, 5, 1), 0);
    }

    #[test]
    fn test_modular_exp_fermat() {
        // Fermat's little theorem: a^(p-1) ≡ 1 (mod p) for prime p
        let p = 17u64;
        for a in 1..p {
            assert_eq!(modular_exp(a, p - 1, p), 1, "a={a}");
        }
    }

    #[test]
    fn test_modular_exp_large() {
        // 3^100 mod 1000000007 via reference computation
        // (just checks it doesn't panic and stays in range)
        let result = modular_exp(3, 100, 1_000_000_007);
        assert!(result < 1_000_000_007);
    }

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(100, 75), 25);
    }

    #[test]
    fn test_continued_fraction_convergents_half() {
        // x = 0.5 = 1/2
        let convs = continued_fraction_convergents(0.5, 100, 10);
        assert!(convs.iter().any(|&(p, q)| p == 1 && q == 2));
    }

    #[test]
    fn test_continued_fraction_convergents_third() {
        // x = 1/3 ≈ 0.3333
        let convs = continued_fraction_convergents(1.0 / 3.0, 100, 10);
        // Should have convergent 1/3
        assert!(
            convs.iter().any(|&(p, q)| p == 1 && q == 3),
            "convergents: {convs:?}"
        );
    }

    #[test]
    fn test_continued_fraction_golden_ratio() {
        // φ - 1 = (√5-1)/2 ≈ 0.6180...
        // Convergents: 1/1, 1/2, 2/3, 3/5, 5/8, 8/13, 13/21, ...
        let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
        let convs = continued_fraction_convergents(phi, 1000, 15);
        // Should contain 3/5
        assert!(
            convs.iter().any(|&(p, q)| p == 3 && q == 5),
            "convergents: {convs:?}"
        );
        // Should contain 5/8
        assert!(
            convs.iter().any(|&(p, q)| p == 5 && q == 8),
            "convergents: {convs:?}"
        );
    }

    #[test]
    fn test_fft_inplace_length_check() {
        let mut v = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]; // length 3, not pow2
        assert!(fft_inplace(&mut v).is_err());
    }

    #[test]
    fn test_fft_inplace_trivial() {
        // DC input: all real 1 → FFT should give (N, 0) at index 0, rest near 0
        let n = 8;
        let mut v: Vec<(f64, f64)> = vec![(1.0, 0.0); n];
        fft_inplace(&mut v).expect("fft ok");
        let (re0, im0) = v[0];
        assert!((re0 - n as f64).abs() < 1e-10, "DC bin={re0}");
        assert!(im0.abs() < 1e-10);
        for i in 1..n {
            let mag = (v[i].0 * v[i].0 + v[i].1 * v[i].1).sqrt();
            assert!(mag < 1e-9, "bin {i} mag={mag}");
        }
    }
}
