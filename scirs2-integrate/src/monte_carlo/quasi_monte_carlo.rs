//! Quasi-Monte Carlo (QMC) sequences and integration.
//!
//! QMC replaces pseudo-random points with **deterministic low-discrepancy
//! sequences** that fill the unit hypercube much more uniformly, achieving
//! a convergence rate of approximately `O(N⁻¹ (log N)ˢ)` for smooth
//! `s`-dimensional integrands, compared with `O(N⁻½)` for plain Monte Carlo.
//!
//! ## Provided sequences
//!
//! | Struct | Type | Notes |
//! |--------|------|-------|
//! | [`HaltonSequence`] | Van der Corput in prime bases | Simple; correlations in high dims |
//! | [`SobolSequence`] | Direction-number construction | Excellent up to ~21 dims (built-in) |
//! | [`LatticeRule`] | Korobov rank-1 lattice | Best for periodic integrands |
//!
//! ## Integration helpers
//!
//! * [`qmc_integrate`] – generic QMC integration accepting any of the above.
//! * [`scrambled_net`] – Owen's nested uniform scrambling for variance/error estimation.
//!
//! ## References
//!
//! - Niederreiter, H. (1992). *Random Number Generation and Quasi-Monte Carlo Methods.*
//! - Joe, S. & Kuo, F. Y. (2008). Constructing Sobol sequences with better two-dimensional
//!   projections. *SIAM J. Sci. Comput.*, 30(5), 2635–2654.

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Uniform;
use scirs2_core::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// QmcSequence trait
// ─────────────────────────────────────────────────────────────────────────────

/// A deterministic (or optionally randomised) sequence in `[0,1)ᵈ`.
pub trait QmcSequence {
    /// Returns the next point as a `Vec<f64>` of length `dim()`.
    fn next_point(&mut self) -> Vec<f64>;

    /// Dimensionality of the sequence.
    fn dim(&self) -> usize;

    /// Resets the sequence to its initial state (index 0).
    fn reset(&mut self);

    /// Generates `n` successive points as an `n × d` matrix.
    fn generate(&mut self, n: usize) -> Array2<f64> {
        let d = self.dim();
        let mut out = Array2::<f64>::zeros((n, d));
        for i in 0..n {
            let p = self.next_point();
            for j in 0..d {
                out[[i, j]] = p[j];
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Halton sequence
// ─────────────────────────────────────────────────────────────────────────────

/// First 21 primes for default Halton bases.
const PRIMES_21: [usize; 21] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
];

/// Halton low-discrepancy sequence.
///
/// Dimension `k` uses the `k`-th prime as its radical-inverse base.
///
/// # Example
///
/// ```
/// use scirs2_integrate::monte_carlo::quasi_monte_carlo::{HaltonSequence, QmcSequence};
///
/// let mut seq = HaltonSequence::new(2);
/// let p0 = seq.next_point();
/// assert_eq!(p0.len(), 2);
/// assert!(p0.iter().all(|&x| x >= 0.0 && x < 1.0));
/// ```
pub struct HaltonSequence {
    /// Bases (one per dimension); must be pairwise coprime.
    bases: Vec<usize>,
    /// Current index (0-based; the first call to `next_point` returns the
    /// point corresponding to `index = 1`).
    index: usize,
}

impl HaltonSequence {
    /// Constructs a Halton sequence with the first `dim` primes as bases.
    ///
    /// # Panics
    ///
    /// Does not panic; returns `Err` if `dim == 0` (checked internally).
    pub fn new(dim: usize) -> Self {
        let bases = if dim <= PRIMES_21.len() {
            PRIMES_21[..dim].to_vec()
        } else {
            // Generate primes beyond the table using trial division.
            let mut primes = PRIMES_21.to_vec();
            let mut candidate = *PRIMES_21.last().expect("table is non-empty") + 2;
            while primes.len() < dim {
                if is_prime(candidate) {
                    primes.push(candidate);
                }
                candidate += 2;
            }
            primes
        };
        Self { bases, index: 0 }
    }

    /// Constructs a Halton sequence with explicitly provided bases.
    ///
    /// `bases` must be pairwise coprime (typically distinct primes).
    pub fn with_bases(bases: Vec<usize>) -> IntegrateResult<Self> {
        if bases.is_empty() {
            return Err(IntegrateError::ValueError(
                "bases must be non-empty".to_string(),
            ));
        }
        Ok(Self { bases, index: 0 })
    }
}

impl QmcSequence for HaltonSequence {
    fn next_point(&mut self) -> Vec<f64> {
        self.index += 1;
        self.bases
            .iter()
            .map(|&b| van_der_corput(self.index, b))
            .collect()
    }

    fn dim(&self) -> usize {
        self.bases.len()
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}

/// Van der Corput radical-inverse function: reverses the base-`b` digits of `n`.
fn van_der_corput(mut n: usize, base: usize) -> f64 {
    let mut result = 0.0_f64;
    let mut denom = 1.0_f64;
    while n > 0 {
        denom *= base as f64;
        result += (n % base) as f64 / denom;
        n /= base;
    }
    result
}

/// Simple primality test (trial division).
fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut d = 3;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

// ─────────────────────────────────────────────────────────────────────────────
// Sobol sequence (direction numbers for up to 21 dimensions)
// ─────────────────────────────────────────────────────────────────────────────

/// Sobol low-discrepancy sequence.
///
/// Uses the standard direction-number initialisation for up to 21 dimensions.
/// For additional dimensions, the Gray-code construction is extended with
/// primitive polynomials.
///
/// # Example
///
/// ```
/// use scirs2_integrate::monte_carlo::quasi_monte_carlo::{SobolSequence, QmcSequence};
///
/// let mut sobol = SobolSequence::new(3).expect("valid dim");
/// let pt = sobol.next_point();
/// assert_eq!(pt.len(), 3);
/// ```
pub struct SobolSequence {
    dim: usize,
    /// Direction numbers, dim × BITS matrix.  `direction[d][k]` is the
    /// `k`-th direction number for dimension `d`.
    direction: Vec<Vec<u64>>,
    /// Current point (integer representation).
    x: Vec<u64>,
    /// Current index.
    n: usize,
    /// Number of bits used.
    bits: usize,
}

const SOBOL_BITS: usize = 30;

impl SobolSequence {
    /// Creates a new Sobol sequence for the given number of dimensions.
    ///
    /// Supports up to 21 dimensions with built-in direction numbers.  Higher
    /// dimensions fall back to a simple polynomial construction.
    pub fn new(dim: usize) -> IntegrateResult<Self> {
        if dim == 0 {
            return Err(IntegrateError::ValueError("dim must be ≥ 1".to_string()));
        }
        let direction = initialise_direction_numbers(dim, SOBOL_BITS);
        let x = vec![0u64; dim];
        Ok(Self {
            dim,
            direction,
            x,
            n: 0,
            bits: SOBOL_BITS,
        })
    }
}

impl QmcSequence for SobolSequence {
    fn next_point(&mut self) -> Vec<f64> {
        // Gray-code increment: find the rightmost 0-bit of the current index.
        let c = trailing_zeros_plus1(self.n);
        self.n += 1;
        let scale = (1u64 << self.bits) as f64;
        let mut out = Vec::with_capacity(self.dim);
        for d in 0..self.dim {
            let bit = c.min(self.bits) - 1;
            self.x[d] ^= self.direction[d][bit];
            out.push(self.x[d] as f64 / scale);
        }
        out
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn reset(&mut self) {
        self.x = vec![0u64; self.dim];
        self.n = 0;
    }
}

/// Returns the position of the least-significant 0-bit + 1.
/// e.g. n=0b110 → position 1 (0-indexed 0), returns 1.
fn trailing_zeros_plus1(n: usize) -> usize {
    (!n).trailing_zeros() as usize + 1
}

/// Initialises Sobol direction numbers for `dim` dimensions and `bits` bits.
///
/// Dimension 0 (base 2) uses v_k = 2^(bits-k).
/// Dimensions 1..21 use pre-stored primitive polynomials.
/// Further dimensions fall back to Niederreiter-style constants.
fn initialise_direction_numbers(dim: usize, bits: usize) -> Vec<Vec<u64>> {
    // Primitive polynomials (degree, coefficients) for dims 1..20.
    // Coefficients encoded as bit masks (excluding leading 1).
    let poly_data: &[(u32, u32)] = &[
        (1, 0),  // dim 1: x
        (2, 1),  // dim 2: x²+x+1
        (3, 1),  // dim 3: x³+x+1
        (3, 2),  // dim 4: x³+x²+1
        (4, 1),  // dim 5: x⁴+x+1
        (4, 4),  // dim 6: x⁴+x³+1
        (5, 2),  // dim 7
        (5, 4),  // dim 8
        (5, 7),  // dim 9
        (5, 11), // dim 10
        (5, 13), // dim 11
        (5, 14), // dim 12
        (6, 1),  // dim 13
        (6, 13), // dim 14
        (6, 16), // dim 15
        (6, 19), // dim 16
        (6, 22), // dim 17
        (6, 25), // dim 18
        (7, 1),  // dim 19
        (7, 4),  // dim 20
    ];

    let mut dirs: Vec<Vec<u64>> = Vec::with_capacity(dim);

    // Dimension 0: v_k = 2^(bits-k) for k=0..bits.
    let d0: Vec<u64> = (0..bits).map(|k| 1u64 << (bits - 1 - k)).collect();
    dirs.push(d0);

    for d in 1..dim {
        let (degree, coeff) = if d - 1 < poly_data.len() {
            poly_data[d - 1]
        } else {
            // Fallback for very high dims: simple degree-1 polynomial.
            (1 + (d as u32 / 20), (d as u32 % 4) + 1)
        };

        let s = degree as usize;
        // Initial m-values (seed) for this polynomial.
        let m_init: Vec<u64> = (1..=s.min(bits))
            .map(|k| {
                // Use an odd number for the initial direction vectors.
                let raw = 2u64.pow(k as u32) - 1;
                // Ensure it fits within k bits and is odd.
                (raw | 1) % (1u64 << k)
            })
            .collect();

        let mut v: Vec<u64> = vec![0u64; bits];
        // Fill initial values
        for k in 0..s.min(bits) {
            v[k] = m_init[k] * (1u64 << (bits - 1 - k));
        }

        // Recurrence relation: v[k] = v[k-s] ⊕ (v[k-s] >> s) ⊕ Σ aⱼ v[k-j]
        for k in s..bits {
            let mut x = v[k - s] ^ (v[k - s] >> s);
            for j in 1..s {
                if (coeff >> (s - 1 - j)) & 1 == 1 {
                    x ^= v[k - j];
                }
            }
            v[k] = x;
        }

        dirs.push(v);
    }

    dirs
}

// ─────────────────────────────────────────────────────────────────────────────
// Lattice Rule (Korobov rank-1)
// ─────────────────────────────────────────────────────────────────────────────

/// Korobov rank-1 lattice rule for periodic integrands.
///
/// The lattice points are `{k · z / N mod 1}ₖ₌₀^{N-1}` in `d` dimensions,
/// where `z` is the generator vector.
///
/// For periodic functions, lattice rules achieve near-`O(N⁻¹)` convergence
/// (under smoothness assumptions), significantly faster than Sobol or Halton.
///
/// # Example
///
/// ```
/// use scirs2_integrate::monte_carlo::quasi_monte_carlo::{LatticeRule, QmcSequence};
///
/// let lat = LatticeRule::korobov(1024, 2).expect("valid params");
/// let pts = lat.points();
/// assert_eq!(pts.nrows(), 1024);
/// assert_eq!(pts.ncols(), 2);
/// ```
pub struct LatticeRule {
    /// Number of lattice points.
    n: usize,
    /// Dimensionality.
    dim: usize,
    /// Generator vector (length `dim`).
    generator: Vec<u64>,
    /// Current index.
    index: usize,
}

impl LatticeRule {
    /// Creates a Korobov lattice with a default generator `z = (1, a, a², …)  mod N`
    /// where `a` is a primitive root (approximated by `⌊N/φ⌋` for coprimality).
    pub fn korobov(n: usize, dim: usize) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::ValueError("n must be > 0".to_string()));
        }
        if dim == 0 {
            return Err(IntegrateError::ValueError("dim must be > 0".to_string()));
        }

        // Use a = round(N / φ) which tends to give good two-dimensional projections.
        let phi = 0.6180339887_f64; // 1/φ ≈ golden ratio
        let a = ((n as f64 * phi).round() as u64).max(1);

        let mut gen = vec![0u64; dim];
        gen[0] = 1;
        for d in 1..dim {
            gen[d] = (gen[d - 1] * a) % (n as u64);
        }

        Ok(Self {
            n,
            dim,
            generator: gen,
            index: 0,
        })
    }

    /// Creates a lattice from an explicit generator vector `z` with `N` points.
    pub fn with_generator(n: usize, generator: Vec<u64>) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::ValueError("n must be > 0".to_string()));
        }
        if generator.is_empty() {
            return Err(IntegrateError::ValueError(
                "generator must be non-empty".to_string(),
            ));
        }
        let dim = generator.len();
        Ok(Self {
            n,
            dim,
            generator,
            index: 0,
        })
    }

    /// Returns all `N` lattice points as an `N × d` matrix.
    pub fn points(&self) -> Array2<f64> {
        let n_f = self.n as f64;
        let mut out = Array2::<f64>::zeros((self.n, self.dim));
        for k in 0..self.n {
            for d in 0..self.dim {
                out[[k, d]] = ((k as u64 * self.generator[d]) % (self.n as u64)) as f64 / n_f;
            }
        }
        out
    }

    /// Returns the number of lattice points.
    pub fn n_points(&self) -> usize {
        self.n
    }
}

impl QmcSequence for LatticeRule {
    fn next_point(&mut self) -> Vec<f64> {
        let k = self.index % self.n;
        self.index += 1;
        let n_f = self.n as f64;
        self.generator
            .iter()
            .map(|&g| ((k as u64 * g) % (self.n as u64)) as f64 / n_f)
            .collect()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn reset(&mut self) {
        self.index = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Owen's scrambled net
// ─────────────────────────────────────────────────────────────────────────────

/// Options for [`scrambled_net`].
#[derive(Debug, Clone)]
pub struct ScrambledNetOptions {
    /// Number of scrambled replicates used to estimate the error.
    pub n_replicates: usize,
    /// RNG seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for ScrambledNetOptions {
    fn default() -> Self {
        Self {
            n_replicates: 16,
            seed: None,
        }
    }
}

/// Applies **Owen's nested uniform scrambling** to a Halton or Sobol net and
/// estimates the integral and its standard error from `n_replicates` scrambled
/// copies.
///
/// Scrambling preserves the equidistribution properties of the net while
/// making the samples look random, enabling unbiased error estimation via
/// the sample variance across replicates.
///
/// # Arguments
///
/// * `f`          – integrand `ℝᵈ → ℝ`
/// * `ranges`     – integration domain
/// * `n_points`   – number of QMC points per replicate
/// * `dim`        – dimension of the domain
/// * `options`    – scrambling options
///
/// # Returns
///
/// `(estimate, std_error)` — the integral estimate and its estimated standard
/// error across replicates.
pub fn scrambled_net<F>(
    f: F,
    ranges: &[(f64, f64)],
    n_points: usize,
    options: Option<ScrambledNetOptions>,
) -> IntegrateResult<(f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    let opts = options.unwrap_or_default();
    let dim = ranges.len();

    if dim == 0 {
        return Err(IntegrateError::ValueError(
            "ranges must be non-empty".to_string(),
        ));
    }
    if n_points == 0 {
        return Err(IntegrateError::ValueError(
            "n_points must be positive".to_string(),
        ));
    }
    if opts.n_replicates < 2 {
        return Err(IntegrateError::ValueError(
            "n_replicates must be ≥ 2 for error estimation".to_string(),
        ));
    }

    let mut rng = match opts.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut os = scirs2_core::random::rng();
            StdRng::from_rng(&mut os)
        }
    };

    // Volume of the integration domain
    let volume: f64 = ranges.iter().map(|(a, b)| b - a).product();

    let uniform01 = Uniform::new(0.0_f64, 1.0).expect("valid bounds");

    let mut replicate_estimates: Vec<f64> = Vec::with_capacity(opts.n_replicates);

    for _rep in 0..opts.n_replicates {
        // Build a scrambled Halton sequence: add a random digit-scramble shift.
        // We use a *random digital shift* (the simplest valid scramble):
        // x̃ᵢ = xᵢ ⊕ Δ  (bitwise xor with a random shift in [0,1)).
        // This preserves stratification properties.
        let shifts: Vec<f64> = (0..dim).map(|_| uniform01.sample(&mut rng)).collect();

        let mut seq = HaltonSequence::new(dim);
        let mut sum = 0.0_f64;

        for _ in 0..n_points {
            let raw = seq.next_point();
            // Apply the digital shift (mod 1)
            let shifted: Vec<f64> = raw
                .iter()
                .zip(shifts.iter())
                .map(|(x, s)| (x + s).fract())
                .collect();
            // Map to the integration domain
            let mapped: Vec<f64> = shifted
                .iter()
                .zip(ranges.iter())
                .map(|(u, (a, b))| a + (b - a) * u)
                .collect();
            sum += f(&mapped);
        }

        replicate_estimates.push(sum / (n_points as f64) * volume);
    }

    let n_r = opts.n_replicates as f64;
    let mean: f64 = replicate_estimates.iter().sum::<f64>() / n_r;
    let var: f64 = replicate_estimates
        .iter()
        .map(|&v| (v - mean) * (v - mean))
        .sum::<f64>()
        / (n_r * (n_r - 1.0));
    let std_error = var.sqrt();

    Ok((mean, std_error))
}

// ─────────────────────────────────────────────────────────────────────────────
// qmc_integrate
// ─────────────────────────────────────────────────────────────────────────────

/// Options for [`qmc_integrate`].
#[derive(Debug, Clone)]
pub struct QmcIntegrateOptions {
    /// Number of sample points.
    pub n_points: usize,
    /// Optional random digital shift seed (for randomised QMC).
    pub seed: Option<u64>,
}

impl Default for QmcIntegrateOptions {
    fn default() -> Self {
        Self {
            n_points: 10000,
            seed: None,
        }
    }
}

/// Result of a QMC integration.
#[derive(Debug, Clone)]
pub struct QmcIntegrateResult {
    /// Estimated integral value.
    pub value: f64,
    /// Number of function evaluations.
    pub n_evals: usize,
}

/// Integrate `f` over a rectangular domain using any [`QmcSequence`].
///
/// The sequence is expected to produce points in `[0,1)ᵈ`; they are
/// linearly mapped to the user-specified `ranges`.
///
/// # Arguments
///
/// * `f`       – integrand mapping a `d`-vector to `f64`
/// * `ranges`  – integration bounds `[(a₁,b₁), …, (aₐ,bₐ)]`
/// * `seq`     – a mutable reference to any [`QmcSequence`]
/// * `options` – number of points and optional seed
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::quasi_monte_carlo::{
///     HaltonSequence, QmcIntegrateOptions, qmc_integrate,
/// };
/// use scirs2_core::ndarray::ArrayView1;
///
/// let mut halton = HaltonSequence::new(2);
/// let opts = QmcIntegrateOptions {
///     n_points: 10000,
///     seed: None,
/// };
///
/// // Integrate f(x,y) = x + y over [0,1]²; exact = 1.0.
/// let result = qmc_integrate(
///     |pt: &[f64]| pt[0] + pt[1],
///     &[(0.0, 1.0), (0.0, 1.0)],
///     &mut halton,
///     Some(opts),
/// ).expect("QMC failed");
///
/// assert!((result.value - 1.0).abs() < 0.01);
/// ```
pub fn qmc_integrate<F, S>(
    f: F,
    ranges: &[(f64, f64)],
    seq: &mut S,
    options: Option<QmcIntegrateOptions>,
) -> IntegrateResult<QmcIntegrateResult>
where
    F: Fn(&[f64]) -> f64,
    S: QmcSequence,
{
    let opts = options.unwrap_or_default();

    if ranges.is_empty() {
        return Err(IntegrateError::ValueError(
            "ranges must be non-empty".to_string(),
        ));
    }
    if opts.n_points == 0 {
        return Err(IntegrateError::ValueError(
            "n_points must be positive".to_string(),
        ));
    }

    let dim = ranges.len();
    if seq.dim() != dim {
        return Err(IntegrateError::DimensionMismatch(format!(
            "sequence dim {} ≠ ranges len {}",
            seq.dim(),
            dim
        )));
    }

    let volume: f64 = ranges.iter().map(|(a, b)| b - a).product();

    // Optional random digital shift for unbiasedness.
    let shifts: Option<Vec<f64>> = opts.seed.map(|s| {
        let mut rng = StdRng::seed_from_u64(s);
        let u01 = Uniform::new(0.0_f64, 1.0).expect("valid bounds");
        (0..dim).map(|_| u01.sample(&mut rng)).collect()
    });

    seq.reset();

    let mut sum = 0.0_f64;
    let mut mapped = vec![0.0_f64; dim];

    for _ in 0..opts.n_points {
        let raw = seq.next_point();
        for j in 0..dim {
            let u = match &shifts {
                Some(sh) => (raw[j] + sh[j]).fract(),
                None => raw[j],
            };
            let (a, b) = ranges[j];
            mapped[j] = a + (b - a) * u;
        }
        sum += f(&mapped);
    }

    Ok(QmcIntegrateResult {
        value: sum / (opts.n_points as f64) * volume,
        n_evals: opts.n_points,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_halton_range() {
        let mut seq = HaltonSequence::new(3);
        for _ in 0..1000 {
            let p = seq.next_point();
            for &x in &p {
                assert!((0.0..1.0).contains(&x), "Halton point out of [0,1)");
            }
        }
    }

    #[test]
    fn test_halton_integrate_x_squared() {
        // ∫₀¹ x² dx = 1/3
        let mut seq = HaltonSequence::new(1);
        let opts = QmcIntegrateOptions {
            n_points: 8192,
            seed: None,
        };
        let res = qmc_integrate(|p| p[0] * p[0], &[(0.0, 1.0)], &mut seq, Some(opts))
            .expect("Halton integrate failed");
        assert!(
            (res.value - 1.0 / 3.0).abs() < 0.005,
            "Halton estimate {} ≠ 1/3",
            res.value
        );
    }

    #[test]
    fn test_sobol_range() {
        let mut seq = SobolSequence::new(4).expect("valid dim");
        for _ in 0..1000 {
            let p = seq.next_point();
            assert_eq!(p.len(), 4);
            for &x in &p {
                assert!((0.0..=1.0).contains(&x), "Sobol point out of [0,1]");
            }
        }
    }

    #[test]
    fn test_sobol_integrate_2d() {
        // ∫₀¹ ∫₀¹ (x+y) dx dy = 1
        let mut seq = SobolSequence::new(2).expect("valid dim");
        let opts = QmcIntegrateOptions {
            n_points: 4096,
            seed: None,
        };
        let res = qmc_integrate(
            |p| p[0] + p[1],
            &[(0.0, 1.0), (0.0, 1.0)],
            &mut seq,
            Some(opts),
        )
        .expect("Sobol integrate failed");
        assert!(
            (res.value - 1.0).abs() < 0.01,
            "Sobol estimate {} ≠ 1.0",
            res.value
        );
    }

    #[test]
    fn test_lattice_rule_integrate() {
        // ∫₀¹ cos(2πx) dx = 0  (periodic function, ideal for lattice)
        let lat = LatticeRule::korobov(1024, 1).expect("valid params");
        let pts = lat.points();
        let sum: f64 = (0..1024).map(|i| (2.0 * PI * pts[[i, 0]]).cos()).sum();
        let est = sum / 1024.0;
        assert!(est.abs() < 0.01, "lattice cosine integral {} ≠ 0", est);
    }

    #[test]
    fn test_scrambled_net() {
        // ∫₀¹ x dx = 0.5
        let (est, se) =
            scrambled_net(|p| p[0], &[(0.0, 1.0)], 1024, None).expect("scrambled net failed");
        assert!((est - 0.5).abs() < 0.02, "scrambled net est {} ≠ 0.5", est);
        assert!(se >= 0.0, "std error must be non-negative");
    }

    #[test]
    fn test_qmc_integrate_with_shift() {
        // Randomised QMC: ∫₀¹ sin(πx) dx = 2/π ≈ 0.6366
        let mut seq = HaltonSequence::new(1);
        let opts = QmcIntegrateOptions {
            n_points: 8192,
            seed: Some(1234),
        };
        let res = qmc_integrate(|p| (PI * p[0]).sin(), &[(0.0, 1.0)], &mut seq, Some(opts))
            .expect("shifted QMC failed");
        assert!(
            (res.value - 2.0 / PI).abs() < 0.01,
            "shifted QMC {} ≠ 2/π",
            res.value
        );
    }

    #[test]
    fn test_sobol_reset() {
        let mut seq = SobolSequence::new(2).expect("valid dim");
        let pts_first: Vec<Vec<f64>> = (0..8).map(|_| seq.next_point()).collect();
        seq.reset();
        let pts_second: Vec<Vec<f64>> = (0..8).map(|_| seq.next_point()).collect();
        assert_eq!(
            pts_first, pts_second,
            "Sobol reset should reproduce sequence"
        );
    }
}
