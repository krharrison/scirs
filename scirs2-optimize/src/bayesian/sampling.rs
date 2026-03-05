//! Sampling strategies for initial experimental design in Bayesian optimization.
//!
//! This module provides various sampling methods for generating initial points
//! in the search space before the surrogate model takes over. Proper space-filling
//! designs are critical for Bayesian optimization performance.
//!
//! # Available Methods
//!
//! - **Latin Hypercube Sampling (LHS)**: Stratified sampling with maximin optimization
//! - **Sobol sequences**: Quasi-random low-discrepancy sequences
//! - **Halton sequences**: Multi-dimensional quasi-random sequences using prime bases
//! - **Random sampling**: Uniform random baseline

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};

use crate::error::{OptimizeError, OptimizeResult};

/// Strategy for generating initial sample points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Uniform random sampling
    Random,
    /// Latin Hypercube Sampling with optional maximin optimization
    LatinHypercube,
    /// Sobol quasi-random sequence
    Sobol,
    /// Halton quasi-random sequence
    Halton,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::LatinHypercube
    }
}

/// Configuration for sampling methods.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Number of maximin iterations for LHS optimization (default: 100)
    pub lhs_maximin_iters: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Scramble Sobol/Halton sequences for better uniformity
    pub scramble: bool,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            lhs_maximin_iters: 100,
            seed: None,
            scramble: true,
        }
    }
}

/// Generate sample points within given bounds.
///
/// # Arguments
/// * `n_samples` - Number of points to generate
/// * `bounds` - Lower and upper bounds for each dimension: `[(low, high), ...]`
/// * `strategy` - Sampling strategy to use
/// * `config` - Optional sampling configuration
///
/// # Returns
/// A 2D array of shape `(n_samples, n_dims)` with sample points.
pub fn generate_samples(
    n_samples: usize,
    bounds: &[(f64, f64)],
    strategy: SamplingStrategy,
    config: Option<SamplingConfig>,
) -> OptimizeResult<Array2<f64>> {
    let config = config.unwrap_or_default();
    let n_dims = bounds.len();

    if n_samples == 0 {
        return Ok(Array2::zeros((0, n_dims)));
    }
    if n_dims == 0 {
        return Err(OptimizeError::InvalidInput(
            "Bounds must have at least one dimension".to_string(),
        ));
    }

    // Validate bounds
    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInput(format!(
                "Lower bound must be strictly less than upper bound for dimension {} (got [{}, {}])",
                i, lo, hi
            )));
        }
        if !lo.is_finite() || !hi.is_finite() {
            return Err(OptimizeError::InvalidInput(format!(
                "Bounds must be finite for dimension {} (got [{}, {}])",
                i, lo, hi
            )));
        }
    }

    match strategy {
        SamplingStrategy::Random => random_sampling(n_samples, bounds, &config),
        SamplingStrategy::LatinHypercube => latin_hypercube_sampling(n_samples, bounds, &config),
        SamplingStrategy::Sobol => sobol_sampling(n_samples, bounds, &config),
        SamplingStrategy::Halton => halton_sampling(n_samples, bounds, &config),
    }
}

// ---------------------------------------------------------------------------
// Random sampling
// ---------------------------------------------------------------------------

fn random_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    config: &SamplingConfig,
) -> OptimizeResult<Array2<f64>> {
    let n_dims = bounds.len();
    let mut rng = make_rng(config.seed);
    let mut samples = Array2::zeros((n_samples, n_dims));

    for i in 0..n_samples {
        for (j, &(lo, hi)) in bounds.iter().enumerate() {
            samples[[i, j]] = lo + rng.random_range(0.0..1.0) * (hi - lo);
        }
    }

    Ok(samples)
}

// ---------------------------------------------------------------------------
// Latin Hypercube Sampling with maximin optimization
// ---------------------------------------------------------------------------

/// Latin Hypercube Sampling (LHS) with maximin distance optimization.
///
/// LHS divides each dimension into `n_samples` equal strata and places exactly
/// one sample in each stratum per dimension. The maximin optimization then
/// iteratively swaps elements to maximize the minimum pairwise distance,
/// yielding better space-filling properties.
fn latin_hypercube_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    config: &SamplingConfig,
) -> OptimizeResult<Array2<f64>> {
    let n_dims = bounds.len();
    let mut rng = make_rng(config.seed);

    // Step 1: Generate basic LHS in [0,1]^d
    // For each dimension, create a random permutation of {0, 1, ..., n-1}
    let mut unit_samples = Array2::zeros((n_samples, n_dims));

    for j in 0..n_dims {
        let mut perm: Vec<usize> = (0..n_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let swap_idx = rng.random_range(0..=i);
            perm.swap(i, swap_idx);
        }
        for i in 0..n_samples {
            // Place sample uniformly within its stratum
            let u: f64 = rng.random_range(0.0..1.0);
            unit_samples[[i, j]] = (perm[i] as f64 + u) / n_samples as f64;
        }
    }

    // Step 2: Maximin optimization by column-wise pair swaps
    if config.lhs_maximin_iters > 0 && n_samples > 2 {
        let mut best_min_dist = compute_min_distance(&unit_samples);

        for _ in 0..config.lhs_maximin_iters {
            // Pick a random dimension
            let dim = rng.random_range(0..n_dims);
            // Pick two random rows
            let r1 = rng.random_range(0..n_samples);
            let mut r2 = rng.random_range(0..n_samples.saturating_sub(1));
            if r2 >= r1 {
                r2 += 1;
            }

            // Tentatively swap
            let tmp = unit_samples[[r1, dim]];
            unit_samples[[r1, dim]] = unit_samples[[r2, dim]];
            unit_samples[[r2, dim]] = tmp;

            let new_min_dist = compute_min_distance(&unit_samples);
            if new_min_dist > best_min_dist {
                best_min_dist = new_min_dist;
            } else {
                // Revert swap
                let tmp = unit_samples[[r1, dim]];
                unit_samples[[r1, dim]] = unit_samples[[r2, dim]];
                unit_samples[[r2, dim]] = tmp;
            }
        }
    }

    // Step 3: Scale to bounds
    let mut result = Array2::zeros((n_samples, n_dims));
    for i in 0..n_samples {
        for (j, &(lo, hi)) in bounds.iter().enumerate() {
            result[[i, j]] = lo + unit_samples[[i, j]] * (hi - lo);
        }
    }

    Ok(result)
}

/// Compute the minimum pairwise Euclidean distance in a sample set.
fn compute_min_distance(samples: &Array2<f64>) -> f64 {
    let n = samples.nrows();
    if n < 2 {
        return f64::INFINITY;
    }
    let mut min_dist = f64::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq_dist = 0.0;
            for k in 0..samples.ncols() {
                let d = samples[[i, k]] - samples[[j, k]];
                sq_dist += d * d;
            }
            if sq_dist < min_dist {
                min_dist = sq_dist;
            }
        }
    }
    min_dist.sqrt()
}

// ---------------------------------------------------------------------------
// Sobol quasi-random sequence
// ---------------------------------------------------------------------------

/// Sobol sequence generator using direction numbers.
///
/// Implements the Joe-Kuo direction numbers for up to 21201 dimensions.
/// Here we provide a compact implementation for the first several dimensions
/// using hardcoded primitive polynomials and initial direction numbers.
fn sobol_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    config: &SamplingConfig,
) -> OptimizeResult<Array2<f64>> {
    let n_dims = bounds.len();
    let mut samples = Array2::zeros((n_samples, n_dims));

    // We need direction numbers for each dimension.
    // Dimension 0 uses the Van der Corput sequence in base 2.
    // Higher dimensions use Joe-Kuo direction numbers.
    let direction_numbers = get_sobol_direction_numbers(n_dims)?;

    for j in 0..n_dims {
        let dirs = &direction_numbers[j];
        let mut x: u64 = 0;
        for i in 0..n_samples {
            if j == 0 {
                // Dimension 0: Van der Corput in base 2 using gray code
                x = gray_code_sobol(i as u64 + 1);
            } else {
                // Use direction numbers with gray code enumeration
                if i == 0 {
                    x = 0;
                } else {
                    // Find the rightmost zero bit of i
                    let c = rightmost_zero_bit(i as u64);
                    let dir_idx = c.min(dirs.len() - 1);
                    x ^= dirs[dir_idx];
                }
            }

            let value = x as f64 / (1u64 << 32) as f64;

            // Optional scramble: Owen's scrambling approximation via random shift
            let scrambled = if config.scramble {
                let mut rng = make_rng(config.seed.map(|s| s.wrapping_add(j as u64 * 1000 + 7)));
                let shift: f64 = rng.random_range(0.0..1.0);
                (value + shift) % 1.0
            } else {
                value
            };

            let (lo, hi) = bounds[j];
            samples[[i, j]] = lo + scrambled * (hi - lo);
        }
    }

    Ok(samples)
}

/// Gray code based Sobol index for dimension 0.
fn gray_code_sobol(n: u64) -> u64 {
    // For dimension 0, Sobol sequence is just the bit-reversed fraction.
    // We use 32-bit precision.
    let mut result: u64 = 0;
    let mut val = n;
    let mut bit = 1u64 << 31;
    while val > 0 {
        if val & 1 != 0 {
            result ^= bit;
        }
        val >>= 1;
        bit >>= 1;
    }
    result
}

/// Find the index of the rightmost zero bit (0-indexed).
fn rightmost_zero_bit(n: u64) -> usize {
    let mut val = n;
    let mut c = 0usize;
    while val & 1 != 0 {
        val >>= 1;
        c += 1;
    }
    c
}

/// Get Sobol direction numbers for up to `n_dims` dimensions.
///
/// Dimension 0 is handled by the Van der Corput sequence.
/// For dimensions 1..n_dims, we use hardcoded Joe-Kuo direction numbers
/// for the first 20 dimensions, and fall back to a deterministic
/// construction for higher dimensions.
fn get_sobol_direction_numbers(n_dims: usize) -> OptimizeResult<Vec<Vec<u64>>> {
    // Maximum bits of precision (32-bit)
    let max_bits = 32usize;

    let mut all_dirs = Vec::with_capacity(n_dims);

    // Dimension 0: placeholder (handled specially)
    all_dirs.push(vec![0u64; max_bits]);

    if n_dims <= 1 {
        return Ok(all_dirs);
    }

    // Primitive polynomials (degree, polynomial coefficients as bits)
    // These are from the Joe-Kuo tables.
    // Format: (degree, poly_coeffs_bits)
    // The polynomial x^s + c_{s-1}*x^{s-1} + ... + c_1*x + 1
    // is stored as the integer with bit pattern c_{s-1}...c_1
    let primitive_polys: &[(u32, u32)] = &[
        (1, 0),  // x + 1
        (2, 1),  // x^2 + x + 1
        (3, 1),  // x^3 + x + 1
        (3, 2),  // x^3 + x^2 + 1
        (4, 1),  // x^4 + x + 1
        (4, 4),  // x^4 + x^3 + 1
        (5, 2),  // x^5 + x^2 + 1
        (5, 4),  // x^5 + x^3 + 1
        (5, 7),  // x^5 + x^3 + x^2 + x + 1
        (5, 11), // x^5 + x^4 + x^2 + x + 1
        (5, 13), // x^5 + x^4 + x^3 + x + 1
        (5, 14), // x^5 + x^4 + x^3 + x^2 + 1
        (6, 1),  // x^6 + x + 1
        (6, 13), // x^6 + x^4 + x^3 + x + 1
        (6, 16), // x^6 + x^5 + 1
        (6, 19), // x^6 + x^5 + x^2 + x + 1
        (6, 22), // x^6 + x^5 + x^3 + x^2 + 1
        (6, 25), // x^6 + x^5 + x^4 + x + 1
        (7, 1),  // x^7 + x + 1
        (7, 4),  // x^7 + x^3 + 1
    ];

    // Initial direction numbers (m_i values, 1-indexed) from Joe-Kuo
    // Each row corresponds to a dimension (starting from dimension 1).
    // The first `degree` values are the initial direction numbers.
    let initial_m: &[&[u64]] = &[
        &[1],                   // dim 1
        &[1, 1],                // dim 2
        &[1, 1, 1],             // dim 3
        &[1, 3, 1],             // dim 4
        &[1, 1, 1, 1],          // dim 5
        &[1, 1, 3, 1],          // dim 6
        &[1, 3, 5, 1, 3],       // dim 7
        &[1, 3, 3, 1, 1],       // dim 8
        &[1, 3, 7, 7, 5],       // dim 9
        &[1, 1, 5, 1, 15],      // dim 10
        &[1, 3, 1, 3, 5],       // dim 11
        &[1, 3, 7, 7, 5],       // dim 12
        &[1, 1, 1, 1, 1, 1],    // dim 13
        &[1, 1, 5, 3, 13, 7],   // dim 14
        &[1, 3, 3, 1, 1, 1],    // dim 15
        &[1, 1, 1, 5, 7, 11],   // dim 16
        &[1, 1, 7, 3, 29, 3],   // dim 17
        &[1, 3, 7, 7, 21, 25],  // dim 18
        &[1, 1, 1, 1, 1, 1, 1], // dim 19
        &[1, 3, 1, 1, 1, 7, 1], // dim 20
    ];

    for dim_idx in 1..n_dims {
        let poly_idx = if dim_idx - 1 < primitive_polys.len() {
            dim_idx - 1
        } else {
            (dim_idx - 1) % primitive_polys.len()
        };

        let (degree, poly_bits) = primitive_polys[poly_idx];
        let s = degree as usize;

        let mut dirs = vec![0u64; max_bits];

        // Set initial direction numbers
        let init = if dim_idx - 1 < initial_m.len() {
            initial_m[dim_idx - 1]
        } else {
            // Fall back to all-ones
            &[1u64; 1][..] // Will be extended below
        };

        for k in 0..s.min(max_bits) {
            let m_k = if k < init.len() { init[k] } else { 1 };
            // Direction number v_k = m_k * 2^(32 - k - 1)
            dirs[k] = m_k << (max_bits - k - 1);
        }

        // Generate remaining direction numbers using the recurrence:
        // v_k = c_1 * v_{k-1} XOR c_2 * v_{k-2} XOR ... XOR c_{s-1} * v_{k-s+1}
        //       XOR v_{k-s} XOR (v_{k-s} >> s)
        for k in s..max_bits {
            let mut new_v = dirs[k - s] ^ (dirs[k - s] >> s);
            for j in 1..s {
                if (poly_bits >> (s - 1 - j)) & 1 == 1 {
                    new_v ^= dirs[k - j];
                }
            }
            dirs[k] = new_v;
        }

        all_dirs.push(dirs);
    }

    Ok(all_dirs)
}

// ---------------------------------------------------------------------------
// Halton quasi-random sequence
// ---------------------------------------------------------------------------

/// Halton sequence using prime bases per dimension.
///
/// The Halton sequence is a generalization of the Van der Corput sequence
/// to multiple dimensions, using a different prime base for each dimension.
fn halton_sampling(
    n_samples: usize,
    bounds: &[(f64, f64)],
    config: &SamplingConfig,
) -> OptimizeResult<Array2<f64>> {
    let n_dims = bounds.len();
    let primes = first_n_primes(n_dims);
    let mut samples = Array2::zeros((n_samples, n_dims));

    // Optional random shift for scrambling
    let shifts: Vec<f64> = if config.scramble {
        let mut rng = make_rng(config.seed);
        (0..n_dims).map(|_| rng.random_range(0.0..1.0)).collect()
    } else {
        vec![0.0; n_dims]
    };

    for i in 0..n_samples {
        for j in 0..n_dims {
            let raw = radical_inverse(i as u64 + 1, primes[j]);
            let value = if config.scramble {
                (raw + shifts[j]) % 1.0
            } else {
                raw
            };
            let (lo, hi) = bounds[j];
            samples[[i, j]] = lo + value * (hi - lo);
        }
    }

    Ok(samples)
}

/// Compute the radical inverse of `n` in the given `base`.
///
/// The radical inverse is the fraction formed by reflecting the digits of `n`
/// about the decimal point in the given base.
fn radical_inverse(n: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    let mut val = n;

    while val > 0 {
        denom *= base as f64;
        result += (val % base) as f64 / denom;
        val /= base;
    }
    result
}

/// Return the first `n` prime numbers.
fn first_n_primes(n: usize) -> Vec<u64> {
    if n == 0 {
        return Vec::new();
    }
    let mut primes = Vec::with_capacity(n);
    let mut candidate = 2u64;

    while primes.len() < n {
        let is_prime = primes
            .iter()
            .take_while(|&&p| p * p <= candidate)
            .all(|&p| candidate % p != 0);
        if is_prime {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let s: u64 = scirs2_core::random::rng().random();
            StdRng::seed_from_u64(s)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bounds_2d() -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0), (0.0, 10.0)]
    }

    fn bounds_5d() -> Vec<(f64, f64)> {
        vec![
            (0.0, 1.0),
            (-1.0, 1.0),
            (0.0, 100.0),
            (-10.0, 10.0),
            (5.0, 15.0),
        ]
    }

    // ---- random sampling ----

    #[test]
    fn test_random_sampling_shape() {
        let samples = generate_samples(20, &bounds_2d(), SamplingStrategy::Random, None)
            .expect("should succeed");
        assert_eq!(samples.nrows(), 20);
        assert_eq!(samples.ncols(), 2);
    }

    #[test]
    fn test_random_sampling_within_bounds() {
        let b = bounds_2d();
        let samples =
            generate_samples(100, &b, SamplingStrategy::Random, None).expect("should succeed");
        for i in 0..samples.nrows() {
            for (j, &(lo, hi)) in b.iter().enumerate() {
                assert!(
                    samples[[i, j]] >= lo && samples[[i, j]] <= hi,
                    "sample[{},{}] = {} not in [{}, {}]",
                    i,
                    j,
                    samples[[i, j]],
                    lo,
                    hi
                );
            }
        }
    }

    // ---- LHS ----

    #[test]
    fn test_lhs_shape_and_bounds() {
        let b = bounds_5d();
        let samples = generate_samples(30, &b, SamplingStrategy::LatinHypercube, None)
            .expect("should succeed");
        assert_eq!(samples.nrows(), 30);
        assert_eq!(samples.ncols(), 5);

        for i in 0..samples.nrows() {
            for (j, &(lo, hi)) in b.iter().enumerate() {
                assert!(
                    samples[[i, j]] >= lo && samples[[i, j]] <= hi,
                    "LHS sample[{},{}] = {} not in [{}, {}]",
                    i,
                    j,
                    samples[[i, j]],
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn test_lhs_stratification() {
        // Each dimension should have exactly one sample per stratum.
        let n = 10;
        let bounds = vec![(0.0, 1.0); 3];
        let cfg = SamplingConfig {
            lhs_maximin_iters: 0, // No optimization, raw LHS
            seed: Some(42),
            scramble: false,
        };
        let samples = generate_samples(n, &bounds, SamplingStrategy::LatinHypercube, Some(cfg))
            .expect("should succeed");

        for j in 0..3 {
            let mut strata = vec![false; n];
            for i in 0..n {
                let stratum = (samples[[i, j]] * n as f64).floor() as usize;
                let stratum = stratum.min(n - 1);
                strata[stratum] = true;
            }
            // Every stratum should be occupied
            for (s, &occupied) in strata.iter().enumerate() {
                assert!(occupied, "Stratum {} in dimension {} is unoccupied", s, j);
            }
        }
    }

    #[test]
    fn test_lhs_maximin_improves_spacing() {
        let n = 15;
        let bounds = vec![(0.0, 1.0); 2];

        // Without maximin
        let cfg0 = SamplingConfig {
            lhs_maximin_iters: 0,
            seed: Some(123),
            scramble: false,
        };
        let s0 = generate_samples(n, &bounds, SamplingStrategy::LatinHypercube, Some(cfg0))
            .expect("should succeed");

        // With maximin
        let cfg1 = SamplingConfig {
            lhs_maximin_iters: 500,
            seed: Some(123),
            scramble: false,
        };
        let s1 = generate_samples(n, &bounds, SamplingStrategy::LatinHypercube, Some(cfg1))
            .expect("should succeed");

        let d0 = compute_min_distance(&s0);
        let d1 = compute_min_distance(&s1);

        // Maximin should give equal or better minimum distance
        assert!(
            d1 >= d0 - 1e-12,
            "Maximin LHS should not decrease min distance: d_opt={} < d_raw={}",
            d1,
            d0
        );
    }

    // ---- Sobol ----

    #[test]
    fn test_sobol_shape_and_bounds() {
        let b = bounds_2d();
        let samples =
            generate_samples(32, &b, SamplingStrategy::Sobol, None).expect("should succeed");
        assert_eq!(samples.nrows(), 32);
        assert_eq!(samples.ncols(), 2);

        for i in 0..samples.nrows() {
            for (j, &(lo, hi)) in b.iter().enumerate() {
                assert!(
                    samples[[i, j]] >= lo && samples[[i, j]] <= hi,
                    "Sobol sample[{},{}] = {} not in [{}, {}]",
                    i,
                    j,
                    samples[[i, j]],
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn test_sobol_reproducibility() {
        let b = bounds_2d();
        let cfg = SamplingConfig {
            seed: Some(99),
            scramble: true,
            ..Default::default()
        };
        let s1 = generate_samples(16, &b, SamplingStrategy::Sobol, Some(cfg.clone()))
            .expect("should succeed");
        let s2 =
            generate_samples(16, &b, SamplingStrategy::Sobol, Some(cfg)).expect("should succeed");
        assert_eq!(s1, s2);
    }

    // ---- Halton ----

    #[test]
    fn test_halton_shape_and_bounds() {
        let b = bounds_5d();
        let samples =
            generate_samples(50, &b, SamplingStrategy::Halton, None).expect("should succeed");
        assert_eq!(samples.nrows(), 50);
        assert_eq!(samples.ncols(), 5);

        for i in 0..samples.nrows() {
            for (j, &(lo, hi)) in b.iter().enumerate() {
                assert!(
                    samples[[i, j]] >= lo && samples[[i, j]] <= hi,
                    "Halton sample[{},{}] = {} not in [{}, {}]",
                    i,
                    j,
                    samples[[i, j]],
                    lo,
                    hi
                );
            }
        }
    }

    #[test]
    fn test_halton_low_discrepancy() {
        // Halton in 1D with base 2 should produce Van der Corput sequence:
        // 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, ...
        let bounds = vec![(0.0, 1.0)];
        let cfg = SamplingConfig {
            seed: None,
            scramble: false,
            ..Default::default()
        };
        let samples = generate_samples(4, &bounds, SamplingStrategy::Halton, Some(cfg))
            .expect("should succeed");

        let expected = [0.5, 0.25, 0.75, 0.125];
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (samples[[i, 0]] - exp).abs() < 1e-10,
                "Halton[{}] = {}, expected {}",
                i,
                samples[[i, 0]],
                exp
            );
        }
    }

    // ---- edge cases ----

    #[test]
    fn test_zero_samples() {
        let samples = generate_samples(0, &bounds_2d(), SamplingStrategy::Random, None)
            .expect("should succeed");
        assert_eq!(samples.nrows(), 0);
    }

    #[test]
    fn test_single_sample() {
        for strategy in &[
            SamplingStrategy::Random,
            SamplingStrategy::LatinHypercube,
            SamplingStrategy::Sobol,
            SamplingStrategy::Halton,
        ] {
            let samples =
                generate_samples(1, &bounds_2d(), *strategy, None).expect("should succeed");
            assert_eq!(samples.nrows(), 1);
            assert_eq!(samples.ncols(), 2);
        }
    }

    #[test]
    fn test_invalid_bounds_rejected() {
        // lo >= hi
        let result = generate_samples(10, &[(5.0, 5.0)], SamplingStrategy::Random, None);
        assert!(result.is_err());

        // infinite
        let result = generate_samples(
            10,
            &[(f64::NEG_INFINITY, 1.0)],
            SamplingStrategy::Random,
            None,
        );
        assert!(result.is_err());
    }

    // ---- prime generation ----

    #[test]
    fn test_first_n_primes() {
        let p = first_n_primes(10);
        assert_eq!(p, vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29]);
    }

    #[test]
    fn test_radical_inverse() {
        // radical_inverse(1, 2) = 0.5
        assert!((radical_inverse(1, 2) - 0.5).abs() < 1e-15);
        // radical_inverse(2, 2) = 0.25
        assert!((radical_inverse(2, 2) - 0.25).abs() < 1e-15);
        // radical_inverse(3, 2) = 0.75
        assert!((radical_inverse(3, 2) - 0.75).abs() < 1e-15);
        // radical_inverse(1, 3) = 1/3
        assert!((radical_inverse(1, 3) - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_high_dimensional_sampling() {
        let bounds: Vec<(f64, f64)> = (0..15).map(|_| (0.0, 1.0)).collect();
        for strategy in &[
            SamplingStrategy::Random,
            SamplingStrategy::LatinHypercube,
            SamplingStrategy::Sobol,
            SamplingStrategy::Halton,
        ] {
            let samples = generate_samples(20, &bounds, *strategy, None).expect("should succeed");
            assert_eq!(samples.nrows(), 20);
            assert_eq!(samples.ncols(), 15);
        }
    }
}
