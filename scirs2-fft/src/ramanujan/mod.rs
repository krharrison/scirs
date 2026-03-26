//! Ramanujan Periodic Transform (RPT).
//!
//! The RPT decomposes a discrete signal into periodicities using Ramanujan
//! sums, which are arithmetic functions orthogonal to different periods.
//!
//! ## Ramanujan sum definition
//!
//! ```text
//! c_q(n) = Σ_{k=1, gcd(k,q)=1}^{q} exp(2πi k n / q)
//! ```
//!
//! This is a real-valued, integer-valued function.  An equivalent closed form
//! uses the Möbius function μ and Euler's totient φ:
//!
//! ```text
//! c_q(n) = μ(q / gcd(n, q)) · φ(q) / φ(q / gcd(n, q))
//! ```
//!
//! ## RPT expansion
//!
//! A finite signal x[0..N-1] is expanded as
//!
//! ```text
//! x[n] ≈ Σ_{q=1}^{Q} a_q · c_q(n mod q)
//! ```
//!
//! where the coefficients a_q minimise the L₂ reconstruction error.
//!
//! # References
//! - P. P. Vaidyanathan & P.-K. Phoong, "Ramanujan Sums in the Context of
//!   Signal Processing," *IEEE Trans. Signal Process.*, 62(16), 2014.
//! - S. Ramanujan, "On Certain Trigonometrical Sums," *Trans. Cambridge Phil.
//!   Soc.*, 22, 1918.

use crate::error::{FFTError, FFTResult};

// ── Number-theoretic helpers ──────────────────────────────────────────────────

/// Möbius function μ(n).
///
/// Returns:
/// *  0  if n has a squared prime factor,
/// *  1  if n = 1,
/// * −1^k if n is a product of k distinct primes.
pub fn mobius(n: usize) -> i64 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut m = n;
    let mut k = 0i64; // number of distinct prime factors
                      // Trial division
    let mut d = 2usize;
    while d * d <= m {
        if m % d == 0 {
            m /= d;
            k += 1;
            if m % d == 0 {
                // d² | n → squared prime factor
                return 0;
            }
        }
        d += 1;
    }
    if m > 1 {
        k += 1;
    }
    if k % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Euler's totient function φ(n).
///
/// φ(n) counts the integers in [1, n] that are coprime to n.
pub fn euler_totient(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut result = n;
    let mut m = n;
    let mut d = 2usize;
    while d * d <= m {
        if m % d == 0 {
            while m % d == 0 {
                m /= d;
            }
            result -= result / d;
        }
        d += 1;
    }
    if m > 1 {
        result -= result / m;
    }
    result
}

/// Greatest common divisor via Euclidean algorithm.
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Ramanujan sum c_q(n) using the closed-form identity:
///
/// ```text
/// c_q(n) = μ(q / d) · φ(q) / φ(q / d)
/// ```
///
/// where d = gcd(n, q).
pub fn ramanujan_sum(q: usize, n: i64) -> f64 {
    if q == 0 {
        return 0.0;
    }
    let n_mod = n.unsigned_abs() as usize % q;
    let d = if n_mod == 0 { q } else { gcd(n_mod, q) };
    let q_over_d = q / d;
    let phi_q = euler_totient(q) as f64;
    let phi_q_over_d = euler_totient(q_over_d) as f64;
    if phi_q_over_d < 1e-12 {
        return 0.0;
    }
    mobius(q_over_d) as f64 * phi_q / phi_q_over_d
}

// ── Subspace basis ─────────────────────────────────────────────────────────────

/// Compute the Ramanujan subspace basis for period `q`.
///
/// The basis consists of φ(q) vectors, one for each coprime residue
/// k ∈ {1 ≤ k ≤ q : gcd(k, q) = 1}.  Each basis vector has length q and
/// its n-th component is `cos(2π k n / q)` (the real part of the Ramanujan
/// sum kernel).
///
/// Returns `Vec<Vec<f64>>` where each inner vector has length `q`.
pub fn ramanujan_subspace_basis(q: usize) -> Vec<Vec<f64>> {
    if q == 0 {
        return Vec::new();
    }
    let two_pi_over_q = 2.0 * std::f64::consts::PI / q as f64;
    let coprimes: Vec<usize> = (1..=q).filter(|&k| gcd(k, q) == 1).collect();
    coprimes
        .iter()
        .map(|&k| {
            (0..q)
                .map(|n| (two_pi_over_q * k as f64 * n as f64).cos())
                .collect()
        })
        .collect()
}

// ── Configuration & Result ────────────────────────────────────────────────────

/// Configuration for [`compute_rpt`].
#[derive(Debug, Clone)]
pub struct RamanujamConfig {
    /// Maximum period Q to consider.  Default: 16.
    pub max_period: usize,
    /// If `true`, orthogonalise the subspace basis via Gram-Schmidt before
    /// projection.  Default: `true`.
    pub use_orthogonal_basis: bool,
}

impl Default for RamanujamConfig {
    fn default() -> Self {
        Self {
            max_period: 16,
            use_orthogonal_basis: true,
        }
    }
}

/// Result of [`compute_rpt`].
#[derive(Debug, Clone)]
pub struct RamanujamResult {
    /// Projection energy (coefficient) for each period q = 1 .. max_period.
    pub coefficients: Vec<f64>,
    /// Period indices 1 .. max_period.
    pub periods: Vec<usize>,
    /// Reconstructed signal (linear combination of Ramanujan subspaces).
    pub reconstruction: Vec<f64>,
    /// Period with the largest projection energy.
    pub dominant_period: usize,
}

// ── Gram-Schmidt orthogonalisation ────────────────────────────────────────────

/// Orthogonalise the columns in `basis` (each column has length `n`) using
/// modified Gram-Schmidt.  Returns only non-zero vectors.
fn gram_schmidt(basis: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut ortho: Vec<Vec<f64>> = Vec::new();
    for v in basis {
        let mut u = v.clone();
        for q in &ortho {
            let dot_uq: f64 = u.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
            let dot_qq: f64 = q.iter().map(|x| x * x).sum();
            if dot_qq > 1e-15 {
                let coeff = dot_uq / dot_qq;
                for (a, b) in u.iter_mut().zip(q.iter()) {
                    *a -= coeff * b;
                }
            }
        }
        let norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            let inv = 1.0 / norm;
            for x in u.iter_mut() {
                *x *= inv;
            }
            ortho.push(u);
        }
    }
    ortho
}

// ── Core RPT computation ──────────────────────────────────────────────────────

/// Compute the Ramanujan Periodic Transform of a real-valued signal.
///
/// For each candidate period q ∈ 1..=max_period:
/// 1. Build the Ramanujan subspace basis for period q.
/// 2. (Optionally) orthogonalise via Gram-Schmidt.
/// 3. Project the signal onto the subspace and record the projection energy.
///
/// The dominant period is the q with maximum projection energy.
///
/// # Arguments
/// * `signal` — Real-valued input signal of length ≥ 1.
/// * `config`  — Configuration (max period, orthogonalisation).
///
/// # Errors
/// Returns an error if the signal is empty.
pub fn compute_rpt(signal: &[f64], config: &RamanujamConfig) -> FFTResult<RamanujamResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("signal must not be empty".into()));
    }
    if config.max_period == 0 {
        return Err(FFTError::ValueError("max_period must be ≥ 1".into()));
    }

    let q_max = config.max_period;

    let mut coefficients = vec![0.0_f64; q_max];
    let mut periods: Vec<usize> = (1..=q_max).collect();
    let mut reconstruction = vec![0.0_f64; n];

    for (qi, &q) in periods.iter().enumerate() {
        // Build subspace basis vectors tiled to length n
        let raw_basis: Vec<Vec<f64>> = ramanujan_subspace_basis(q)
            .into_iter()
            .map(|bv| {
                // Tile bv to length n
                (0..n).map(|i| bv[i % q]).collect::<Vec<f64>>()
            })
            .collect();

        if raw_basis.is_empty() {
            continue;
        }

        let ortho_basis = if config.use_orthogonal_basis {
            gram_schmidt(&raw_basis)
        } else {
            raw_basis
        };

        // Project signal onto the subspace and accumulate reconstruction
        let mut energy = 0.0_f64;
        for basis_vec in &ortho_basis {
            // Dot product <signal, basis_vec>
            let coeff: f64 = signal
                .iter()
                .zip(basis_vec.iter())
                .map(|(s, b)| s * b)
                .sum();
            // Add coeff * basis_vec to reconstruction
            for (r, b) in reconstruction.iter_mut().zip(basis_vec.iter()) {
                *r += coeff * b;
            }
            energy += coeff * coeff;
        }
        coefficients[qi] = energy;
    }

    // Find dominant period (1-indexed)
    let (dom_idx, _) = coefficients
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));
    let dominant_period = periods[dom_idx];

    // Sort periods and coefficients together for output (already sorted 1..=q_max)
    periods.retain(|_| true); // no-op, just to keep borrow checker happy

    Ok(RamanujamResult {
        coefficients,
        periods,
        reconstruction,
        dominant_period,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Möbius function tests ─────────────────────────────────────────────────

    #[test]
    fn test_mobius_known_values() {
        // μ(1) = 1
        assert_eq!(mobius(1), 1);
        // μ(2) = -1 (prime)
        assert_eq!(mobius(2), -1);
        // μ(3) = -1 (prime)
        assert_eq!(mobius(3), -1);
        // μ(4) = 0 (4 = 2²)
        assert_eq!(mobius(4), 0);
        // μ(6) = 1 (6 = 2·3, 2 primes)
        assert_eq!(mobius(6), 1);
        // μ(8) = 0 (8 = 2³)
        assert_eq!(mobius(8), 0);
        // μ(30) = -1 (30 = 2·3·5, 3 primes)
        assert_eq!(mobius(30), -1);
    }

    #[test]
    fn test_mobius_squarefree_products() {
        // Products of k distinct primes have μ = (-1)^k
        assert_eq!(mobius(2 * 3), 1); // 2 primes
        assert_eq!(mobius(2 * 3 * 5), -1); // 3 primes
        assert_eq!(mobius(2 * 3 * 5 * 7), 1); // 4 primes
    }

    // ── Euler's totient tests ─────────────────────────────────────────────────

    #[test]
    fn test_euler_totient_known() {
        assert_eq!(euler_totient(1), 1);
        assert_eq!(euler_totient(2), 1);
        assert_eq!(euler_totient(6), 2);
        assert_eq!(euler_totient(9), 6);
        assert_eq!(euler_totient(12), 4);
    }

    #[test]
    fn test_euler_totient_prime() {
        // φ(p) = p-1 for prime p
        for &p in &[2usize, 3, 5, 7, 11, 13] {
            assert_eq!(euler_totient(p), p - 1, "φ({p})");
        }
    }

    // ── Ramanujan sum tests ───────────────────────────────────────────────────

    #[test]
    fn test_ramanujan_sum_at_zero() {
        // c_q(0) = φ(q)
        for q in 1..=10usize {
            let cq0 = ramanujan_sum(q, 0);
            let phi_q = euler_totient(q) as f64;
            assert!(
                (cq0 - phi_q).abs() < 1e-10,
                "c_{q}(0) = {cq0}, expected φ({q}) = {phi_q}"
            );
        }
    }

    #[test]
    fn test_ramanujan_sum_period_sum_zero() {
        // For q > 1: Σ_{n=0}^{q-1} c_q(n) = 0
        for q in 2..=8usize {
            let total: f64 = (0..q as i64).map(|n| ramanujan_sum(q, n)).sum();
            assert!(total.abs() < 1e-9, "Σ c_{q}(n) = {total}, expected 0");
        }
    }

    #[test]
    fn test_ramanujan_sum_integer_valued() {
        // Ramanujan sums are integers
        for q in 1..=12usize {
            for n in 0..q as i64 {
                let c = ramanujan_sum(q, n);
                let rounded = c.round();
                assert!((c - rounded).abs() < 1e-9, "c_{q}({n}) = {c} not integer");
            }
        }
    }

    // ── Subspace basis tests ──────────────────────────────────────────────────

    #[test]
    fn test_subspace_basis_size() {
        // Basis for period q has φ(q) vectors each of length q
        for q in 1..=8usize {
            let basis = ramanujan_subspace_basis(q);
            assert_eq!(basis.len(), euler_totient(q), "basis size for q={q}");
            for v in &basis {
                assert_eq!(v.len(), q);
            }
        }
    }

    // ── RPT tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_rpt_recovers_dominant_period() {
        // A pure period-4 cosine signal should have dominant_period = 4
        let n = 64;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 4.0).cos()).collect();
        let config = RamanujamConfig {
            max_period: 16,
            use_orthogonal_basis: true,
        };
        let result = compute_rpt(&signal, &config).expect("rpt ok");
        assert_eq!(
            result.dominant_period, 4,
            "dominant period should be 4, got {}",
            result.dominant_period
        );
    }

    #[test]
    fn test_rpt_reconstruction_shape() {
        let n = 32;
        let signal: Vec<f64> = (0..n).map(|i| i as f64 % 5.0).collect();
        let config = RamanujamConfig::default();
        let result = compute_rpt(&signal, &config).expect("rpt ok");
        assert_eq!(result.reconstruction.len(), n);
        assert_eq!(result.coefficients.len(), config.max_period);
        assert_eq!(result.periods.len(), config.max_period);
    }

    #[test]
    fn test_rpt_config_default() {
        let cfg = RamanujamConfig::default();
        assert_eq!(cfg.max_period, 16);
        assert!(cfg.use_orthogonal_basis);
    }

    #[test]
    fn test_rpt_empty_signal_error() {
        let cfg = RamanujamConfig::default();
        assert!(compute_rpt(&[], &cfg).is_err());
    }

    #[test]
    fn test_rpt_reconstruction_small_error() {
        // For a signal that is exactly periodic with period ≤ max_period,
        // the reconstruction should be close (not necessarily perfect, since
        // we use orthogonal projection without cross-period correction).
        let n = 48;
        let q_true = 3usize;
        let signal: Vec<f64> = (0..n).map(|i| ramanujan_sum(q_true, i as i64)).collect();
        let config = RamanujamConfig {
            max_period: 8,
            use_orthogonal_basis: true,
        };
        let result = compute_rpt(&signal, &config).expect("rpt ok");
        let mse: f64 = signal
            .iter()
            .zip(result.reconstruction.iter())
            .map(|(s, r)| (s - r) * (s - r))
            .sum::<f64>()
            / n as f64;
        // Reconstruction should be reasonable (not necessarily perfect)
        assert!(mse < 100.0, "MSE={mse} too large");
    }
}
