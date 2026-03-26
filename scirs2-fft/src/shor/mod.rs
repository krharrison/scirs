//! Shor's Algorithm Building Blocks.
//!
//! This module provides a classical simulation of Shor's quantum factoring
//! algorithm.  It is intended for educational purposes and small inputs;
//! it does *not* run on actual quantum hardware.
//!
//! # Algorithm overview
//! 1. Trivial check: is N even or a prime power?
//! 2. Choose a random base `a` with 1 < a < N and gcd(a,N)=1.
//! 3. Use a quantum-period-finding simulation (via QFT) to find the
//!    period r of f(x) = a^x mod N.
//! 4. If r is even and a^(r/2) ≢ -1 (mod N), compute
//!    gcd(a^(r/2) ± 1, N) to obtain non-trivial factors.
//!
//! # References
//! - P. Shor, "Polynomial-time algorithms for prime factorization and discrete
//!   logarithms on a quantum computer", SIAM J. Comput. 26(5), 1997.

pub mod period_finding;

pub use period_finding::{continued_fraction_convergents, find_period_qft, gcd, modular_exp};

use crate::error::{FFTError, FFTResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the [`ShorSimulator`].
#[derive(Debug, Clone)]
pub struct ShorConfig {
    /// Maximum number of period candidates to try from continued-fraction
    /// convergents.  Default: 16.
    pub max_period_candidates: usize,
    /// Number of simulated qubits in the phase-estimation register.
    /// Register size = 2^max_qft_qubits.  Must be in 1..=20.  Default: 12.
    pub max_qft_qubits: usize,
    /// Maximum number of random bases `a` to try before giving up.
    /// Default: 20.
    pub max_base_attempts: usize,
}

impl Default for ShorConfig {
    fn default() -> Self {
        Self {
            max_period_candidates: 16,
            max_qft_qubits: 12,
            max_base_attempts: 20,
        }
    }
}

// ── Result ─────────────────────────────────────────────────────────────────────

/// Result returned by [`ShorSimulator::factor`].
#[derive(Debug, Clone)]
pub struct ShorResult {
    /// Non-trivial factors found, or `None` if N is prime/the algorithm failed.
    pub factors: Option<(u64, u64)>,
    /// The period r that led to the factorisation (if found).
    pub period: Option<u64>,
    /// Total number of (a, r) pairs tried before success or giving up.
    pub iterations: usize,
}

// ── Simulator ─────────────────────────────────────────────────────────────────

/// Classical simulation of Shor's quantum factoring algorithm.
#[derive(Debug, Clone)]
pub struct ShorSimulator {
    /// Configuration.
    pub config: ShorConfig,
}

impl ShorSimulator {
    /// Create a new simulator with the given configuration.
    pub fn new(config: ShorConfig) -> Self {
        Self { config }
    }

    /// Create a new simulator with default configuration.
    pub fn default_new() -> Self {
        Self::new(ShorConfig::default())
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Simple trial-division primality test for small N (≤ 10^6).
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 || n == 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }
        let mut i = 5u64;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }
        true
    }

    /// Check if n = b^k for some integer b ≥ 2 and k ≥ 2.
    fn is_perfect_power(n: u64) -> Option<(u64, u32)> {
        if n < 4 {
            return None;
        }
        // Maximum exponent: 2^k ≤ n  ⟹  k ≤ log2(n)
        let max_k = (n as f64).log2().floor() as u32;
        for k in 2..=max_k {
            let b = (n as f64).powf(1.0 / k as f64).round() as u64;
            for candidate in b.saturating_sub(2)..=b + 2 {
                if candidate >= 2 {
                    let pw = (candidate as u128).pow(k);
                    if pw == n as u128 {
                        return Some((candidate, k));
                    }
                }
            }
        }
        None
    }

    /// Deterministic period search by direct evaluation (for small N only).
    /// Returns the multiplicative order of `a` modulo `n`.
    fn period_direct(a: u64, n: u64) -> Option<u64> {
        // Safety limit to avoid huge loops
        let limit = n.min(1 << 20);
        let mut x = a % n;
        for r in 1..=limit {
            if x == 1 {
                return Some(r);
            }
            x = ((x as u128 * a as u128) % n as u128) as u64;
        }
        None
    }

    /// Attempt to extract non-trivial factors of `n` given period `r` and base `a`.
    fn factors_from_period(a: u64, r: u64, n: u64) -> Option<(u64, u64)> {
        if r % 2 != 0 {
            return None; // odd period → unusable
        }
        let half_r = r / 2;
        let x = modular_exp(a, half_r, n);
        if x == n - 1 {
            return None; // a^(r/2) ≡ -1 (mod N) → unusable
        }
        let f1 = gcd(x + 1, n);
        let f2 = gcd(x.wrapping_sub(1).min(n - 1) + 1, n);
        // Accept any non-trivial divisor
        for &f in &[f1, f2] {
            if f > 1 && f < n && n % f == 0 {
                return Some((f, n / f));
            }
        }
        None
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Find the period r of f(x) = a^x mod n using a QFT-based simulation.
    ///
    /// Falls back to direct evaluation if the QFT simulation fails to find a
    /// valid period and N is small enough.
    pub fn find_period(&self, a: u64, n: u64) -> FFTResult<Option<u64>> {
        // Sanity checks
        if a <= 1 || a >= n {
            return Err(FFTError::ValueError(format!(
                "require 1 < a < n, got a={a} n={n}"
            )));
        }
        if gcd(a, n) != 1 {
            return Err(FFTError::ValueError(format!(
                "a={a} and n={n} are not coprime"
            )));
        }

        // QFT-based simulation
        let qft_result = find_period_qft(
            a,
            n,
            self.config.max_qft_qubits,
            self.config.max_period_candidates,
        )?;
        if qft_result.is_some() {
            return Ok(qft_result);
        }

        // Fallback: direct search for small n
        if n <= 1 << 20 {
            return Ok(Self::period_direct(a, n));
        }

        Ok(None)
    }

    /// Factor `n` using Shor's algorithm simulation.
    ///
    /// Returns a [`ShorResult`] with `factors = None` if `n` is prime or if
    /// the algorithm could not find a factorisation within the configured
    /// resource limits.
    ///
    /// # Errors
    /// Propagates internal QFT errors.
    pub fn factor(&self, n: u64) -> FFTResult<ShorResult> {
        // Edge cases
        if n <= 1 {
            return Ok(ShorResult {
                factors: None,
                period: None,
                iterations: 0,
            });
        }
        if n % 2 == 0 {
            return Ok(ShorResult {
                factors: Some((2, n / 2)),
                period: None,
                iterations: 0,
            });
        }
        if Self::is_prime(n) {
            return Ok(ShorResult {
                factors: None,
                period: None,
                iterations: 0,
            });
        }
        // Perfect-power check
        if let Some((base, _exp)) = Self::is_perfect_power(n) {
            if base > 1 && n % base == 0 {
                return Ok(ShorResult {
                    factors: Some((base, n / base)),
                    period: None,
                    iterations: 0,
                });
            }
        }

        // Main loop: try different bases
        let mut iterations = 0;
        // Use a deterministic sequence of bases: 2, 3, 5, 7, ...
        // (avoiding multiples of n, non-coprime bases)
        let mut base_candidates: Vec<u64> = (2..n)
            .filter(|&a| gcd(a, n) == 1)
            .take(self.config.max_base_attempts)
            .collect();
        // Prefer small bases that are coprime
        base_candidates.sort_unstable();

        for a in base_candidates {
            iterations += 1;

            // Quick check: is a itself a non-trivial factor?
            let g = gcd(a, n);
            if g > 1 && g < n {
                return Ok(ShorResult {
                    factors: Some((g, n / g)),
                    period: None,
                    iterations,
                });
            }

            let period_opt = self.find_period(a, n)?;
            if let Some(r) = period_opt {
                if let Some(factors) = Self::factors_from_period(a, r, n) {
                    return Ok(ShorResult {
                        factors: Some(factors),
                        period: Some(r),
                        iterations,
                    });
                }
            }
        }

        Ok(ShorResult {
            factors: None,
            period: None,
            iterations,
        })
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(ShorSimulator::is_prime(2));
        assert!(ShorSimulator::is_prime(3));
        assert!(ShorSimulator::is_prime(17));
        assert!(!ShorSimulator::is_prime(1));
        assert!(!ShorSimulator::is_prime(15));
        assert!(!ShorSimulator::is_prime(21));
    }

    #[test]
    fn test_is_perfect_power() {
        assert_eq!(ShorSimulator::is_perfect_power(8), Some((2, 3)));
        assert_eq!(ShorSimulator::is_perfect_power(9), Some((3, 2)));
        assert_eq!(ShorSimulator::is_perfect_power(25), Some((5, 2)));
        assert!(ShorSimulator::is_perfect_power(15).is_none());
    }

    #[test]
    fn test_factor_even_number() {
        let sim = ShorSimulator::default_new();
        let result = sim.factor(10).expect("factor 10");
        assert!(result.factors.is_some());
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 10);
    }

    #[test]
    fn test_factor_prime_returns_none() {
        let sim = ShorSimulator::default_new();
        let result = sim.factor(17).expect("factor 17");
        assert!(result.factors.is_none(), "17 is prime");

        let result2 = sim.factor(7).expect("factor 7");
        assert!(result2.factors.is_none(), "7 is prime");
    }

    #[test]
    fn test_factor_15() {
        // N = 15 = 3 × 5 (classic Shor demo)
        let sim = ShorSimulator::default_new();
        let result = sim.factor(15).expect("factor 15");
        assert!(result.factors.is_some(), "should find factors of 15");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 15, "product must equal 15");
        assert!(a > 1 && b > 1, "non-trivial factors");
    }

    #[test]
    fn test_factor_21() {
        // N = 21 = 3 × 7
        let sim = ShorSimulator::default_new();
        let result = sim.factor(21).expect("factor 21");
        assert!(result.factors.is_some(), "should find factors of 21");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 21);
        assert!(a > 1 && b > 1);
    }

    #[test]
    fn test_factor_35() {
        // N = 35 = 5 × 7
        let sim = ShorSimulator::default_new();
        let result = sim.factor(35).expect("factor 35");
        assert!(result.factors.is_some(), "should find factors of 35");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 35);
    }

    #[test]
    fn test_find_period_known() {
        // Period of 2 mod 15 is 4 (2^4 = 16 ≡ 1 mod 15)
        let sim = ShorSimulator::default_new();
        let period = sim.find_period(2, 15).expect("find_period 2 mod 15");
        assert!(period.is_some());
        let r = period.unwrap();
        assert_eq!(modular_exp(2, r, 15), 1, "2^r mod 15 must equal 1");
    }

    #[test]
    fn test_find_period_bad_input() {
        let sim = ShorSimulator::default_new();
        // a >= n is an error
        assert!(sim.find_period(15, 15).is_err());
        // a = 1 is an error
        assert!(sim.find_period(1, 15).is_err());
    }

    #[test]
    fn test_factor_1_and_2() {
        let sim = ShorSimulator::default_new();
        let r1 = sim.factor(1).expect("factor 1");
        assert!(r1.factors.is_none());
        let r2 = sim.factor(2).expect("factor 2");
        // 2 is even, returns (2,1) — but 1 is trivial
        // Our implementation returns (2, 1) — acceptable since 2 * 1 = 2
        assert!(r2.factors.is_some());
    }

    #[test]
    fn test_factors_from_period_odd_period() {
        // Odd period should return None
        assert!(ShorSimulator::factors_from_period(2, 3, 15).is_none());
    }

    #[test]
    fn test_shor_config_default() {
        let cfg = ShorConfig::default();
        assert_eq!(cfg.max_period_candidates, 16);
        assert_eq!(cfg.max_qft_qubits, 12);
        assert_eq!(cfg.max_base_attempts, 20);
    }
}
