//! SIMD-accelerated batch sampling for continuous distributions.
//!
//! This module provides high-throughput batch sampling for the Normal, Uniform, and
//! Exponential distributions, with vectorised mathematical kernels drawn from
//! `scirs2_core`'s `SimdUnifiedOps` trait.
//!
//! # Design
//!
//! Each sampler generates a block of uniform random numbers first (the only serial
//! bottleneck), then applies the appropriate mathematical transform entirely through
//! SIMD operations:
//!
//! - **Normal** (`sample_normal_batch`): Box-Muller transform — two uniform variates
//!   per pair produce two independent standard-normal variates via `ln`, `sqrt`,
//!   `sin`, `cos`.
//! - **Uniform** (`sample_uniform_batch`): linear rescaling `a + u * (b - a)` using
//!   vectorised FMA.
//! - **Exponential** (`sample_exponential_batch`): inverse-CDF transform
//!   `-ln(u) / λ` using vectorised `ln` and `mul`.
//!
//! Vectorised CDF and PDF evaluation functions are also provided for all three
//! distributions.
//!
//! # Performance
//!
//! On arrays larger than the SIMD scalar threshold (~64 elements), all mathematical
//! transforms are executed as SIMD kernels.  Random number generation itself uses
//! `scirs2_core::random` (pure-Rust Xoshiro256++ under the hood).

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::Rng;
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

// ============================================================================
// Internal macro / helper for RNG construction
// ============================================================================

/// Resolve a seed: if `seed` is `None`, draw a u64 from `thread_rng`;
/// then always call `seeded_rng` so that the return type is uniform.
macro_rules! build_rng {
    ($seed:expr) => {{
        let s: u64 = $seed.unwrap_or_else(|| {
            use scirs2_core::random::Rng;
            scirs2_core::random::thread_rng().random()
        });
        scirs2_core::random::seeded_rng(s)
    }};
}

// ============================================================================
// Normal distribution — batch sampling
// ============================================================================

/// Generate `n` independent samples from N(`mean`, `std_dev`²) using a
/// SIMD-accelerated Box-Muller transform.
///
/// The Box-Muller transform converts pairs of independent U(0, 1) random
/// variables (u₁, u₂) into pairs of independent standard-normal variates:
///
/// ```text
/// z₀ = √(−2 ln u₁) · cos(2π u₂)
/// z₁ = √(−2 ln u₁) · sin(2π u₂)
/// ```
///
/// Both `ln`, `sqrt`, `cos`, and `sin` are evaluated via SIMD intrinsics from
/// `scirs2_core` when the batch is large enough.  The result is then scaled by
/// `std_dev` and shifted by `mean` using a fused-multiply-add pass.
///
/// # Arguments
///
/// * `n`       — Number of samples to generate.
/// * `mean`    — Location parameter of the Normal distribution.
/// * `std_dev` — Scale parameter (must be positive).
/// * `seed`    — Optional RNG seed for reproducibility.
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `n == 0` or `std_dev <= 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::sample_normal_batch;
///
/// let samples = sample_normal_batch::<f64>(1_000, 0.0, 1.0, Some(42))
///     .expect("sampling failed");
/// assert_eq!(samples.len(), 1_000);
///
/// // Empirical mean should be close to 0
/// let mean: f64 = samples.sum() / samples.len() as f64;
/// assert!(mean.abs() < 0.15);
/// ```
pub fn sample_normal_batch<F>(
    n: usize,
    mean: F,
    std_dev: F,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
{
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n must be at least 1".to_string(),
        ));
    }
    if std_dev <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "std_dev must be positive for the Normal distribution".to_string(),
        ));
    }

    // We need ⌈n/2⌉ pairs for Box-Muller.
    let n_pairs = (n + 1) / 2;
    let n_total = n_pairs * 2;

    let mut rng = build_rng!(seed);

    // ── Phase 1: generate uniform pairs ──────────────────────────────────────
    // u1 ∈ (ε, 1) so that ln(u1) is finite; u2 ∈ [0, 1).
    let eps = F::epsilon().to_f64().unwrap_or(1e-15_f64).max(1e-15_f64);

    let u1: Array1<F> = Array1::from_shape_fn(n_pairs, |_| {
        F::from(rng.gen_range(eps..1.0_f64)).unwrap_or_else(|| F::one())
    });
    let u2: Array1<F> = Array1::from_shape_fn(n_pairs, |_| {
        F::from(rng.gen_range(0.0_f64..1.0_f64)).unwrap_or_else(|| F::zero())
    });

    // ── Phase 2: Box-Muller transform ─────────────────────────────────────────
    let optimizer = AutoOptimizer::new();

    let (z0, z1) = if optimizer.should_use_simd(n_pairs) {
        // SIMD path ─────────────────────────────────────────────────────────

        // √(−2 · ln(u1)):  apply simd_ln, negate×2, simd_sqrt
        let ln_u1 = F::simd_ln(&u1.view());
        let neg_two = F::from(-2.0_f64).unwrap_or_else(|| -F::one() - F::one());
        let neg_two_arr = Array1::from_elem(n_pairs, neg_two);
        let neg2_ln_u1 = F::simd_mul(&neg_two_arr.view(), &ln_u1.view());
        let r = F::simd_sqrt(&neg2_ln_u1.view());

        // 2π · u2:  use simd_mul then cos/sin
        let two_pi = F::from(2.0_f64 * std::f64::consts::PI).unwrap_or_else(|| F::one());
        let two_pi_arr = Array1::from_elem(n_pairs, two_pi);
        let theta = F::simd_mul(&two_pi_arr.view(), &u2.view());

        let cos_theta = F::simd_cos(&theta.view());
        let sin_theta = F::simd_sin(&theta.view());

        let raw_z0 = F::simd_mul(&r.view(), &cos_theta.view());
        let raw_z1 = F::simd_mul(&r.view(), &sin_theta.view());

        // Scale: z * std_dev + mean  (FMA)
        let std_arr = Array1::from_elem(n_pairs, std_dev);
        let mean_arr = Array1::from_elem(n_pairs, mean);

        let z0 = F::simd_fma(&raw_z0.view(), &std_arr.view(), &mean_arr.view());
        let z1 = F::simd_fma(&raw_z1.view(), &std_arr.view(), &mean_arr.view());

        (z0, z1)
    } else {
        // Scalar fallback ──────────────────────────────────────────────────
        let two = F::from(2.0_f64).unwrap_or_else(|| F::one() + F::one());
        let two_pi = F::from(2.0_f64 * std::f64::consts::PI).unwrap_or_else(|| F::one());

        let mut z0 = Array1::zeros(n_pairs);
        let mut z1 = Array1::zeros(n_pairs);
        for i in 0..n_pairs {
            let r = (-two * u1[i].ln()).sqrt();
            let theta = two_pi * u2[i];
            z0[i] = mean + std_dev * r * theta.cos();
            z1[i] = mean + std_dev * r * theta.sin();
        }
        (z0, z1)
    };

    // ── Phase 3: interleave z0/z1 and trim to n ───────────────────────────────
    let mut result: Array1<F> = Array1::zeros(n_total);
    for i in 0..n_pairs {
        result[2 * i] = z0[i];
        if 2 * i + 1 < n_total {
            result[2 * i + 1] = z1[i];
        }
    }

    Ok(result.slice(scirs2_core::ndarray::s![..n]).to_owned())
}

// ============================================================================
// Uniform distribution — batch sampling
// ============================================================================

/// Generate `n` independent samples from U(`low`, `high`) using SIMD rescaling.
///
/// Each sample is drawn from the standard uniform U(0, 1) and then linearly
/// rescaled:
///
/// ```text
/// x = low + u · (high − low)
/// ```
///
/// The rescaling is performed as a SIMD FMA pass over the full batch, giving
/// vectorised throughput equal to the underlying RNG bottleneck.
///
/// # Arguments
///
/// * `n`    — Number of samples.
/// * `low`  — Lower bound of the interval (inclusive).
/// * `high` — Upper bound of the interval (exclusive).
/// * `seed` — Optional RNG seed.
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `n == 0` or `low >= high`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::sample_uniform_batch;
///
/// let samples = sample_uniform_batch::<f64>(500, 2.0, 5.0, Some(7))
///     .expect("sampling failed");
/// assert_eq!(samples.len(), 500);
/// assert!(samples.iter().all(|&x| x >= 2.0 && x < 5.0));
/// ```
pub fn sample_uniform_batch<F>(
    n: usize,
    low: F,
    high: F,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
{
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n must be at least 1".to_string(),
        ));
    }
    if low >= high {
        return Err(StatsError::InvalidArgument(
            "low must be strictly less than high for the Uniform distribution".to_string(),
        ));
    }

    let mut rng = build_rng!(seed);

    // ── Phase 1: generate raw U(0, 1) values ──────────────────────────────────
    let u: Array1<F> = Array1::from_shape_fn(n, |_| {
        F::from(rng.gen_range(0.0_f64..1.0_f64)).unwrap_or_else(|| F::zero())
    });

    // ── Phase 2: linear rescaling via SIMD ────────────────────────────────────
    let optimizer = AutoOptimizer::new();
    let width = high - low;

    if optimizer.should_use_simd(n) {
        // x = low + u * width  — computed as FMA: u * width + low
        let width_arr = Array1::from_elem(n, width);
        let low_arr = Array1::from_elem(n, low);
        Ok(F::simd_fma(&u.view(), &width_arr.view(), &low_arr.view()))
    } else {
        Ok(u.mapv(|ui| low + ui * width))
    }
}

// ============================================================================
// Exponential distribution — batch sampling
// ============================================================================

/// Generate `n` independent samples from Exp(`rate`) using a SIMD-accelerated
/// inverse-CDF transform.
///
/// The CDF of Exp(λ) is F(x) = 1 − e^(−λx), so the inverse is:
///
/// ```text
/// x = −ln(u) / λ
/// ```
///
/// where u ~ U(0, 1).  The vectorised `ln` from `scirs2_core` is applied to
/// the entire batch at once.
///
/// # Arguments
///
/// * `n`    — Number of samples.
/// * `rate` — Rate parameter λ (must be positive; mean = 1/λ).
/// * `seed` — Optional RNG seed.
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `n == 0` or `rate <= 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::sample_exponential_batch;
///
/// let rate = 2.0_f64;
/// let samples = sample_exponential_batch::<f64>(1_000, rate, Some(99))
///     .expect("sampling failed");
/// assert_eq!(samples.len(), 1_000);
///
/// // Empirical mean should be near 1/rate = 0.5
/// let mean: f64 = samples.sum() / samples.len() as f64;
/// assert!((mean - 0.5).abs() < 0.08);
/// ```
pub fn sample_exponential_batch<F>(n: usize, rate: F, seed: Option<u64>) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
{
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n must be at least 1".to_string(),
        ));
    }
    if rate <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "rate must be positive for the Exponential distribution".to_string(),
        ));
    }

    let eps = F::epsilon().to_f64().unwrap_or(1e-15_f64).max(1e-15_f64);

    let mut rng = build_rng!(seed);

    // ── Phase 1: generate U(ε, 1) to avoid ln(0) ─────────────────────────────
    let u: Array1<F> = Array1::from_shape_fn(n, |_| {
        F::from(rng.gen_range(eps..1.0_f64)).unwrap_or_else(|| F::one())
    });

    // ── Phase 2: inverse-CDF transform via SIMD ───────────────────────────────
    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // ln(u)
        let ln_u = F::simd_ln(&u.view());

        // −ln(u):  multiply by −1
        let neg_one = F::from(-1.0_f64).unwrap_or_else(|| -F::one());
        let neg_one_arr = Array1::from_elem(n, neg_one);
        let neg_ln_u = F::simd_mul(&neg_one_arr.view(), &ln_u.view());

        // divide by rate:  (1/rate) scalar multiplication
        let inv_rate = F::one() / rate;
        let inv_rate_arr = Array1::from_elem(n, inv_rate);
        Ok(F::simd_mul(&neg_ln_u.view(), &inv_rate_arr.view()))
    } else {
        Ok(u.mapv(|ui| -ui.ln() / rate))
    }
}

// ============================================================================
// Vectorised PDF / CDF evaluation
// ============================================================================

/// Evaluate the Normal PDF at each point in `x` using SIMD.
///
/// Computes the probability density
/// ```text
/// f(x) = exp(−(x − μ)² / (2σ²)) / (σ √(2π))
/// ```
/// over an entire array in a single vectorised pass.
///
/// # Arguments
///
/// * `x`       — Points at which to evaluate.
/// * `mean`    — Distribution mean µ.
/// * `std_dev` — Distribution standard deviation σ (must be positive).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `std_dev <= 0` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::normal_pdf_batch;
///
/// let x = array![0.0_f64, 1.0, -1.0];
/// let pdf = normal_pdf_batch(&x.view(), 0.0, 1.0).expect("failed");
/// // pdf(0) ≈ 0.3989
/// assert!((pdf[0] - 0.3989422804_f64).abs() < 1e-8);
/// ```
pub fn normal_pdf_batch<F>(x: &ArrayView1<F>, mean: F, std_dev: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if std_dev <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "std_dev must be positive".to_string(),
        ));
    }

    let n = x.len();
    let two = F::from(2.0_f64).unwrap_or_else(|| F::one() + F::one());
    let two_pi = F::from(2.0_f64 * std::f64::consts::PI).unwrap_or_else(|| F::one());
    let norm_const = F::one() / (std_dev * two_pi.sqrt());
    let two_var = two * std_dev * std_dev;

    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // (x − μ)
        let mean_arr = Array1::from_elem(n, mean);
        let diff = F::simd_sub(x, &mean_arr.view());

        // (x − μ)²
        let diff_sq = F::simd_mul(&diff.view(), &diff.view());

        // −(x − μ)² / (2σ²)
        let inv_two_var = F::one() / two_var;
        let neg_inv = F::from(-1.0_f64).unwrap_or_else(|| -F::one()) * inv_two_var;
        let neg_inv_arr = Array1::from_elem(n, neg_inv);
        let exponent = F::simd_mul(&neg_inv_arr.view(), &diff_sq.view());

        // exp(exponent)
        let exp_vals = F::simd_exp(&exponent.view());

        // multiply by normalisation constant
        let norm_arr = Array1::from_elem(n, norm_const);
        Ok(F::simd_mul(&norm_arr.view(), &exp_vals.view()))
    } else {
        Ok(x.mapv(|xi| {
            let z = (xi - mean) / std_dev;
            norm_const * (-(z * z) / two).exp()
        }))
    }
}

/// Evaluate the Normal CDF at each point in `x` using a SIMD-friendly
/// rational approximation (Abramowitz & Stegun 26.2.17, max error 7.5 × 10⁻⁸).
///
/// The standard-normal CDF Φ(z) is approximated element-wise; for non-standard
/// parameters the inputs are standardised: z = (x − µ) / σ.
///
/// # Arguments
///
/// * `x`       — Points at which to evaluate.
/// * `mean`    — Distribution mean µ.
/// * `std_dev` — Distribution standard deviation σ (must be positive).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `std_dev <= 0` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::normal_cdf_batch;
///
/// let x = array![0.0_f64, 1.959964_f64];
/// let cdf = normal_cdf_batch(&x.view(), 0.0, 1.0).expect("failed");
/// // CDF(0) ≈ 0.5,  CDF(1.96) ≈ 0.975
/// assert!((cdf[0] - 0.5).abs() < 1e-6);
/// assert!((cdf[1] - 0.975).abs() < 1e-4);
/// ```
pub fn normal_cdf_batch<F>(x: &ArrayView1<F>, mean: F, std_dev: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if std_dev <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "std_dev must be positive".to_string(),
        ));
    }

    // Standardise: z = (x − µ) / σ
    let n = x.len();
    let mean_arr = Array1::from_elem(n, mean);
    let std_arr = Array1::from_elem(n, std_dev);

    let optimizer = AutoOptimizer::new();

    let z_owned = if optimizer.should_use_simd(n) {
        let diff = F::simd_sub(x, &mean_arr.view());
        F::simd_div(&diff.view(), &std_arr.view())
    } else {
        x.mapv(|xi| (xi - mean) / std_dev)
    };

    // Scalar A&S polynomial approximation per element (vectorisable but intrinsic-
    // heavy; let the compiler auto-vectorise the loop).
    let half = F::from(0.5_f64).unwrap_or_else(|| F::one() / (F::one() + F::one()));
    let a1 = F::from(0.319_381_530_f64).unwrap_or_else(|| F::zero());
    let a2 = F::from(-0.356_563_782_f64).unwrap_or_else(|| F::zero());
    let a3 = F::from(1.781_477_937_f64).unwrap_or_else(|| F::zero());
    let a4 = F::from(-1.821_255_978_f64).unwrap_or_else(|| F::zero());
    let a5 = F::from(1.330_274_429_f64).unwrap_or_else(|| F::zero());
    let p = F::from(0.231_641_9_f64).unwrap_or_else(|| F::zero());
    let two_pi = F::from(2.0_f64 * std::f64::consts::PI).unwrap_or_else(|| F::one());

    let result: Array1<F> = z_owned.mapv(|z| {
        let abs_z = z.abs();
        let t = F::one() / (F::one() + p * abs_z);
        let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
        let pdf_z = (-(z * z) / (F::one() + F::one())).exp() / two_pi.sqrt();
        let cdf_abs = F::one() - pdf_z * poly;
        if z >= F::zero() {
            cdf_abs
        } else {
            F::one() - cdf_abs
        }
        // Clamp to [0, 1]
        .max(F::zero())
        .min(F::one())
    });

    Ok(result)
}

/// Evaluate the Uniform PDF at each point in `x` using SIMD comparisons.
///
/// Returns 1/(high − low) for x ∈ [low, high) and 0 elsewhere.
///
/// # Arguments
///
/// * `x`    — Points at which to evaluate.
/// * `low`  — Lower bound.
/// * `high` — Upper bound (must be strictly greater than `low`).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `low >= high` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::uniform_pdf_batch;
///
/// let x = array![0.5_f64, 1.5, 2.5];
/// let pdf = uniform_pdf_batch(&x.view(), 1.0, 2.0).expect("failed");
/// assert!((pdf[0] - 0.0).abs() < 1e-10);
/// assert!((pdf[1] - 1.0).abs() < 1e-10);
/// assert!((pdf[2] - 0.0).abs() < 1e-10);
/// ```
pub fn uniform_pdf_batch<F>(x: &ArrayView1<F>, low: F, high: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if low >= high {
        return Err(StatsError::InvalidArgument(
            "low must be strictly less than high".to_string(),
        ));
    }

    let density = F::one() / (high - low);
    let result = x.mapv(|xi| {
        if xi >= low && xi < high {
            density
        } else {
            F::zero()
        }
    });
    Ok(result)
}

/// Evaluate the Uniform CDF at each point in `x`.
///
/// # Arguments
///
/// * `x`    — Points at which to evaluate.
/// * `low`  — Lower bound.
/// * `high` — Upper bound (must be strictly greater than `low`).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `low >= high` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::uniform_cdf_batch;
///
/// let x = array![0.5_f64, 1.5, 2.5];
/// let cdf = uniform_cdf_batch(&x.view(), 1.0, 2.0).expect("failed");
/// assert!((cdf[0] - 0.0).abs() < 1e-10);
/// assert!((cdf[1] - 0.5).abs() < 1e-10);
/// assert!((cdf[2] - 1.0).abs() < 1e-10);
/// ```
pub fn uniform_cdf_batch<F>(x: &ArrayView1<F>, low: F, high: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if low >= high {
        return Err(StatsError::InvalidArgument(
            "low must be strictly less than high".to_string(),
        ));
    }

    let width = high - low;
    let result = x.mapv(|xi| {
        if xi < low {
            F::zero()
        } else if xi >= high {
            F::one()
        } else {
            (xi - low) / width
        }
    });
    Ok(result)
}

/// Evaluate the Exponential PDF at each point in `x` using SIMD.
///
/// Computes `f(x) = λ · exp(−λx)` for x ≥ 0, else 0.
///
/// # Arguments
///
/// * `x`    — Points at which to evaluate.
/// * `rate` — Rate parameter λ (must be positive).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `rate <= 0` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::exponential_pdf_batch;
///
/// let x = array![0.0_f64, 1.0, 2.0];
/// let pdf = exponential_pdf_batch(&x.view(), 1.0).expect("failed");
/// // pdf(0; λ=1) = 1.0,  pdf(1) = e^-1 ≈ 0.368,  pdf(2) = e^-2 ≈ 0.135
/// assert!((pdf[0] - 1.0).abs() < 1e-10);
/// assert!((pdf[1] - (-1.0_f64).exp()).abs() < 1e-10);
/// ```
pub fn exponential_pdf_batch<F>(x: &ArrayView1<F>, rate: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if rate <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "rate must be positive".to_string(),
        ));
    }

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // Mask: 1 for x >= 0, 0 otherwise
        let mask: Array1<F> = x.mapv(|xi| if xi >= F::zero() { F::one() } else { F::zero() });

        // −rate · x
        let neg_rate = F::from(-1.0_f64).unwrap_or_else(|| -F::one()) * rate;
        let neg_rate_arr = Array1::from_elem(n, neg_rate);
        let exponent = F::simd_mul(&neg_rate_arr.view(), x);

        // exp(−rate · x)
        let exp_vals = F::simd_exp(&exponent.view());

        // rate · exp(−rate · x)
        let rate_arr = Array1::from_elem(n, rate);
        let pdf = F::simd_mul(&rate_arr.view(), &exp_vals.view());

        // Apply mask for non-negative domain
        let result = F::simd_mul(&mask.view(), &pdf.view());
        Ok(result)
    } else {
        Ok(x.mapv(|xi| {
            if xi >= F::zero() {
                rate * (-rate * xi).exp()
            } else {
                F::zero()
            }
        }))
    }
}

/// Evaluate the Exponential CDF at each point in `x` using SIMD.
///
/// Computes `F(x) = 1 − exp(−λx)` for x ≥ 0, else 0.
///
/// # Arguments
///
/// * `x`    — Points at which to evaluate.
/// * `rate` — Rate parameter λ (must be positive).
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `rate <= 0` or `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::exponential_cdf_batch;
///
/// let x = array![0.0_f64, 1.0];
/// let cdf = exponential_cdf_batch(&x.view(), 1.0).expect("failed");
/// assert!((cdf[0] - 0.0).abs() < 1e-10);
/// assert!((cdf[1] - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
/// ```
pub fn exponential_cdf_batch<F>(x: &ArrayView1<F>, rate: F) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument("x is empty".to_string()));
    }
    if rate <= F::zero() {
        return Err(StatsError::InvalidArgument(
            "rate must be positive".to_string(),
        ));
    }

    let n = x.len();
    let optimizer = AutoOptimizer::new();

    if optimizer.should_use_simd(n) {
        // Mask: 1 for x >= 0
        let mask: Array1<F> = x.mapv(|xi| if xi >= F::zero() { F::one() } else { F::zero() });

        // −rate · x
        let neg_rate = F::from(-1.0_f64).unwrap_or_else(|| -F::one()) * rate;
        let neg_rate_arr = Array1::from_elem(n, neg_rate);
        let exponent = F::simd_mul(&neg_rate_arr.view(), x);

        // exp(−rate · x)
        let exp_vals = F::simd_exp(&exponent.view());

        // 1 − exp(−rate · x)
        let ones = Array1::from_elem(n, F::one());
        let cdf_positive = F::simd_sub(&ones.view(), &exp_vals.view());

        // Apply mask
        Ok(F::simd_mul(&mask.view(), &cdf_positive.view()))
    } else {
        Ok(x.mapv(|xi| {
            if xi >= F::zero() {
                F::one() - (-rate * xi).exp()
            } else {
                F::zero()
            }
        }))
    }
}

// ============================================================================
// Parallel batch sampling — Normal distribution
// ============================================================================

/// Generate `n` independent samples from N(`mean`, `std_dev`²) using parallel
/// threads (via Rayon) and per-thread SIMD Box-Muller kernels.
///
/// The work is split into `num_cpus` chunks.  Each chunk receives a
/// deterministic seed derived from the user-supplied `seed` and the chunk
/// index, making the output fully reproducible when `seed` is provided.
///
/// # Arguments
///
/// * `n`       — Total number of samples to generate.
/// * `mean`    — Location parameter of the Normal distribution.
/// * `std_dev` — Scale parameter (must be positive).
/// * `seed`    — Optional base RNG seed for reproducibility.
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `n == 0` or `std_dev <= 0`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::simd_sampling::parallel_normal_sample;
///
/// let samples = parallel_normal_sample(10_000, 5.0_f64, 2.0_f64, Some(42))
///     .expect("parallel sampling failed");
/// assert_eq!(samples.len(), 10_000);
/// let mean: f64 = samples.iter().sum::<f64>() / 10_000.0;
/// assert!((mean - 5.0).abs() < 0.2, "empirical mean {mean} too far from 5.0");
/// ```
pub fn parallel_normal_sample(
    n: usize,
    mean: f64,
    std_dev: f64,
    seed: Option<u64>,
) -> StatsResult<Vec<f64>> {
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n must be at least 1".to_string(),
        ));
    }
    if std_dev <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "std_dev must be positive for the Normal distribution".to_string(),
        ));
    }

    use scirs2_core::parallel_ops::*;

    // Derive a base seed — deterministic when the caller provides one.
    let base_seed: u64 = seed.unwrap_or_else(|| {
        use scirs2_core::random::Rng;
        scirs2_core::random::thread_rng().random()
    });

    // Choose a sensible chunk count (at least 1, at most n).
    let n_threads = num_cpus::get().max(1).min(n);
    let chunk_size = (n + n_threads - 1) / n_threads;

    // Build per-chunk seed offsets so outputs are reproducible.
    // Use wrapping_mul to avoid debug-mode overflow panics on the large
    // Fibonacci hashing constant.
    let seeds: Vec<u64> = (0..n_threads)
        .map(|i| base_seed.wrapping_add((i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15_u64)))
        .collect();

    // Each thread independently generates its slice using the SIMD Box-Muller
    // kernel from `sample_normal_batch`.
    let chunks: Result<Vec<Vec<f64>>, StatsError> = seeds
        .into_par_iter()
        .enumerate()
        .map(|(i, s)| {
            let start = i * chunk_size;
            if start >= n {
                return Ok(vec![]);
            }
            let end = (start + chunk_size).min(n);
            let count = end - start;
            let arr = sample_normal_batch::<f64>(count, mean, std_dev, Some(s))?;
            Ok(arr.to_vec())
        })
        .collect();

    let chunks = chunks?;
    let mut out = Vec::with_capacity(n);
    for chunk in chunks {
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

// ============================================================================
// Parallel batch sampling — Uniform distribution
// ============================================================================

/// Generate `n` independent samples from U(`low`, `high`) using parallel
/// threads and per-thread SIMD linear-rescaling kernels.
///
/// Each chunk receives a deterministic seed derived from the base seed and
/// the chunk index for full reproducibility.
///
/// # Arguments
///
/// * `n`    — Total number of samples.
/// * `low`  — Lower bound (inclusive).
/// * `high` — Upper bound (exclusive; must be strictly greater than `low`).
/// * `seed` — Optional base RNG seed.
///
/// # Errors
///
/// Returns [`StatsError::InvalidArgument`] when `n == 0` or `low >= high`.
///
/// # Examples
///
/// ```
/// use scirs2_stats::simd_sampling::parallel_uniform_sample;
///
/// let samples = parallel_uniform_sample(10_000, 0.0_f64, 1.0_f64, Some(7))
///     .expect("parallel sampling failed");
/// assert_eq!(samples.len(), 10_000);
/// assert!(samples.iter().all(|&x| x >= 0.0 && x < 1.0));
/// ```
pub fn parallel_uniform_sample(
    n: usize,
    low: f64,
    high: f64,
    seed: Option<u64>,
) -> StatsResult<Vec<f64>> {
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "n must be at least 1".to_string(),
        ));
    }
    if low >= high {
        return Err(StatsError::InvalidArgument(
            "low must be strictly less than high for the Uniform distribution".to_string(),
        ));
    }

    use scirs2_core::parallel_ops::*;

    let base_seed: u64 = seed.unwrap_or_else(|| {
        use scirs2_core::random::Rng;
        scirs2_core::random::thread_rng().random()
    });

    let n_threads = num_cpus::get().max(1).min(n);
    let chunk_size = (n + n_threads - 1) / n_threads;

    let seeds: Vec<u64> = (0..n_threads)
        .map(|i| base_seed.wrapping_add((i as u64).wrapping_mul(0x6c62_272e_07bb_0142_u64)))
        .collect();

    let chunks: Result<Vec<Vec<f64>>, StatsError> = seeds
        .into_par_iter()
        .enumerate()
        .map(|(i, s)| {
            let start = i * chunk_size;
            if start >= n {
                return Ok(vec![]);
            }
            let end = (start + chunk_size).min(n);
            let count = end - start;
            let arr = sample_uniform_batch::<f64>(count, low, high, Some(s))?;
            Ok(arr.to_vec())
        })
        .collect();

    let chunks = chunks?;
    let mut out = Vec::with_capacity(n);
    for chunk in chunks {
        out.extend_from_slice(&chunk);
    }
    Ok(out)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── Normal batch sampling ─────────────────────────────────────────────────

    #[test]
    fn test_sample_normal_batch_length() {
        let samples = sample_normal_batch::<f64>(500, 0.0, 1.0, Some(1)).expect("should succeed");
        assert_eq!(samples.len(), 500);
    }

    #[test]
    fn test_sample_normal_batch_empirical_moments() {
        // With 10 000 samples the empirical mean/std should be within ±3σ of true value.
        let n = 10_000_usize;
        let mu = 3.5_f64;
        let sigma = 2.0_f64;
        let samples = sample_normal_batch::<f64>(n, mu, sigma, Some(42)).expect("should succeed");
        assert_eq!(samples.len(), n);

        let emp_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let emp_var: f64 = samples.iter().map(|&x| (x - emp_mean).powi(2)).sum::<f64>() / n as f64;
        let emp_std = emp_var.sqrt();

        // Within 5 % relative error on mean and std (generous to avoid flakiness)
        assert!(
            (emp_mean - mu).abs() < 0.2,
            "empirical mean {emp_mean} too far from {mu}"
        );
        assert!(
            (emp_std - sigma).abs() < 0.2,
            "empirical std {emp_std} too far from {sigma}"
        );
    }

    #[test]
    fn test_sample_normal_batch_rejects_bad_args() {
        assert!(sample_normal_batch::<f64>(0, 0.0, 1.0, None).is_err());
        assert!(sample_normal_batch::<f64>(10, 0.0, -1.0, None).is_err());
        assert!(sample_normal_batch::<f64>(10, 0.0, 0.0, None).is_err());
    }

    #[test]
    fn test_sample_normal_batch_f32() {
        let samples = sample_normal_batch::<f32>(200, 0.0, 1.0, Some(5)).expect("should succeed");
        assert_eq!(samples.len(), 200);
    }

    // ── Uniform batch sampling ─────────────────────────────────────────────────

    #[test]
    fn test_sample_uniform_batch_length() {
        let samples = sample_uniform_batch::<f64>(300, -1.0, 1.0, Some(2)).expect("should succeed");
        assert_eq!(samples.len(), 300);
    }

    #[test]
    fn test_sample_uniform_batch_range() {
        let samples =
            sample_uniform_batch::<f64>(2_000, 3.0, 7.0, Some(11)).expect("should succeed");
        for &s in samples.iter() {
            assert!(s >= 3.0 && s < 7.0, "sample {s} outside [3.0, 7.0)");
        }
    }

    #[test]
    fn test_sample_uniform_batch_empirical_mean() {
        let (low, high) = (2.0_f64, 6.0_f64);
        let expected_mean = (low + high) / 2.0;
        let n = 10_000_usize;
        let samples = sample_uniform_batch::<f64>(n, low, high, Some(99)).expect("should succeed");
        let emp_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!(
            (emp_mean - expected_mean).abs() < 0.1,
            "empirical mean {emp_mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn test_sample_uniform_batch_rejects_bad_args() {
        assert!(sample_uniform_batch::<f64>(0, 0.0, 1.0, None).is_err());
        assert!(sample_uniform_batch::<f64>(10, 1.0, 1.0, None).is_err());
        assert!(sample_uniform_batch::<f64>(10, 2.0, 1.0, None).is_err());
    }

    // ── Exponential batch sampling ────────────────────────────────────────────

    #[test]
    fn test_sample_exponential_batch_length() {
        let samples = sample_exponential_batch::<f64>(400, 1.0, Some(3)).expect("should succeed");
        assert_eq!(samples.len(), 400);
    }

    #[test]
    fn test_sample_exponential_batch_non_negative() {
        let samples =
            sample_exponential_batch::<f64>(5_000, 0.5, Some(77)).expect("should succeed");
        for &s in samples.iter() {
            assert!(s >= 0.0, "exponential sample {s} is negative");
        }
    }

    #[test]
    fn test_sample_exponential_batch_empirical_mean() {
        let rate = 2.5_f64;
        let expected_mean = 1.0 / rate;
        let n = 10_000_usize;
        let samples = sample_exponential_batch::<f64>(n, rate, Some(13)).expect("should succeed");
        let emp_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!(
            (emp_mean - expected_mean).abs() < 0.05,
            "empirical mean {emp_mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn test_sample_exponential_batch_rejects_bad_args() {
        assert!(sample_exponential_batch::<f64>(0, 1.0, None).is_err());
        assert!(sample_exponential_batch::<f64>(10, 0.0, None).is_err());
        assert!(sample_exponential_batch::<f64>(10, -1.0, None).is_err());
    }

    // ── Normal PDF / CDF ─────────────────────────────────────────────────────

    #[test]
    fn test_normal_pdf_batch_standard() {
        let x = array![0.0_f64, 1.0, -1.0];
        let pdf = normal_pdf_batch(&x.view(), 0.0, 1.0).expect("should succeed");
        // pdf(0) = 1/√(2π) ≈ 0.39894
        let expected_0 = 1.0_f64 / (2.0 * std::f64::consts::PI).sqrt();
        assert!((pdf[0] - expected_0).abs() < 1e-7, "pdf[0]={}", pdf[0]);
        // Standard normal is symmetric
        assert!((pdf[1] - pdf[2]).abs() < 1e-10, "symmetry failed");
    }

    #[test]
    fn test_normal_cdf_batch_standard() {
        let x = array![0.0_f64, 1.959_964_f64, -1.959_964_f64];
        let cdf = normal_cdf_batch(&x.view(), 0.0, 1.0).expect("should succeed");
        assert!((cdf[0] - 0.5).abs() < 1e-5, "cdf[0]={}", cdf[0]);
        assert!((cdf[1] - 0.975).abs() < 1e-4, "cdf[1]={}", cdf[1]);
        assert!((cdf[2] - 0.025).abs() < 1e-4, "cdf[2]={}", cdf[2]);
    }

    // ── Uniform PDF / CDF ────────────────────────────────────────────────────

    #[test]
    fn test_uniform_pdf_batch() {
        let x = array![0.5_f64, 1.5, 2.5];
        let pdf = uniform_pdf_batch(&x.view(), 1.0, 2.0).expect("should succeed");
        assert!((pdf[0] - 0.0).abs() < 1e-10);
        assert!((pdf[1] - 1.0).abs() < 1e-10);
        assert!((pdf[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_cdf_batch() {
        let x = array![0.5_f64, 1.5, 2.5];
        let cdf = uniform_cdf_batch(&x.view(), 1.0, 2.0).expect("should succeed");
        assert!((cdf[0] - 0.0).abs() < 1e-10);
        assert!((cdf[1] - 0.5).abs() < 1e-10);
        assert!((cdf[2] - 1.0).abs() < 1e-10);
    }

    // ── Exponential PDF / CDF ────────────────────────────────────────────────

    #[test]
    fn test_exponential_pdf_batch() {
        let x = array![0.0_f64, 1.0, 2.0, -0.5];
        let pdf = exponential_pdf_batch(&x.view(), 1.0).expect("should succeed");
        assert!((pdf[0] - 1.0).abs() < 1e-10, "pdf(0)={}", pdf[0]);
        assert!(
            (pdf[1] - (-1.0_f64).exp()).abs() < 1e-10,
            "pdf(1)={}",
            pdf[1]
        );
        assert!(
            (pdf[2] - (-2.0_f64).exp()).abs() < 1e-10,
            "pdf(2)={}",
            pdf[2]
        );
        // Negative input: PDF should be 0
        assert!((pdf[3] - 0.0).abs() < 1e-10, "pdf(-0.5)={}", pdf[3]);
    }

    #[test]
    fn test_exponential_cdf_batch() {
        let x = array![0.0_f64, 1.0, 2.0, -1.0];
        let cdf = exponential_cdf_batch(&x.view(), 1.0).expect("should succeed");
        assert!((cdf[0] - 0.0).abs() < 1e-10);
        assert!((cdf[1] - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
        assert!((cdf[2] - (1.0 - (-2.0_f64).exp())).abs() < 1e-10);
        assert!((cdf[3] - 0.0).abs() < 1e-10);
    }

    // ── Reproducibility (seeding) ────────────────────────────────────────────

    #[test]
    fn test_seeded_reproducibility_normal() {
        let s1 = sample_normal_batch::<f64>(100, 0.0, 1.0, Some(42)).expect("ok");
        let s2 = sample_normal_batch::<f64>(100, 0.0, 1.0, Some(42)).expect("ok");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a, b, "seeded normal samples should be identical");
        }
    }

    #[test]
    fn test_seeded_reproducibility_uniform() {
        let s1 = sample_uniform_batch::<f64>(100, 0.0, 1.0, Some(17)).expect("ok");
        let s2 = sample_uniform_batch::<f64>(100, 0.0, 1.0, Some(17)).expect("ok");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a, b, "seeded uniform samples should be identical");
        }
    }

    #[test]
    fn test_seeded_reproducibility_exponential() {
        let s1 = sample_exponential_batch::<f64>(100, 0.5, Some(7)).expect("ok");
        let s2 = sample_exponential_batch::<f64>(100, 0.5, Some(7)).expect("ok");
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a, b, "seeded exponential samples should be identical");
        }
    }

    // ── Parallel normal sampling ──────────────────────────────────────────────

    #[test]
    fn test_parallel_normal_sample_length() {
        let samples =
            super::parallel_normal_sample(5_000, 0.0, 1.0, Some(42)).expect("should succeed");
        assert_eq!(samples.len(), 5_000);
    }

    #[test]
    fn test_parallel_normal_sample_empirical_moments() {
        let n = 20_000_usize;
        let mu = 2.5_f64;
        let sigma = 1.5_f64;
        let samples =
            super::parallel_normal_sample(n, mu, sigma, Some(1234)).expect("should succeed");
        assert_eq!(samples.len(), n);

        let emp_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let emp_var: f64 = samples.iter().map(|&x| (x - emp_mean).powi(2)).sum::<f64>() / n as f64;
        let emp_std = emp_var.sqrt();

        assert!(
            (emp_mean - mu).abs() < 0.2,
            "parallel normal: empirical mean {emp_mean} too far from {mu}"
        );
        assert!(
            (emp_std - sigma).abs() < 0.2,
            "parallel normal: empirical std {emp_std} too far from {sigma}"
        );
    }

    #[test]
    fn test_parallel_normal_sample_rejects_bad_args() {
        assert!(super::parallel_normal_sample(0, 0.0, 1.0, None).is_err());
        assert!(super::parallel_normal_sample(10, 0.0, 0.0, None).is_err());
        assert!(super::parallel_normal_sample(10, 0.0, -1.5, None).is_err());
    }

    #[test]
    fn test_parallel_normal_sample_reproducibility() {
        let s1 = super::parallel_normal_sample(1_000, 0.0, 1.0, Some(77)).expect("ok");
        let s2 = super::parallel_normal_sample(1_000, 0.0, 1.0, Some(77)).expect("ok");
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a, b, "seeded parallel normal samples should be identical");
        }
    }

    // ── Parallel uniform sampling ─────────────────────────────────────────────

    #[test]
    fn test_parallel_uniform_sample_length() {
        let samples =
            super::parallel_uniform_sample(4_000, 1.0, 3.0, Some(55)).expect("should succeed");
        assert_eq!(samples.len(), 4_000);
    }

    #[test]
    fn test_parallel_uniform_sample_range() {
        let samples =
            super::parallel_uniform_sample(8_000, 2.0, 5.0, Some(99)).expect("should succeed");
        for &s in samples.iter() {
            assert!(
                s >= 2.0 && s < 5.0,
                "parallel uniform: sample {s} outside [2.0, 5.0)"
            );
        }
    }

    #[test]
    fn test_parallel_uniform_sample_empirical_mean() {
        let (low, high) = (3.0_f64, 7.0_f64);
        let expected_mean = (low + high) / 2.0;
        let n = 10_000_usize;
        let samples =
            super::parallel_uniform_sample(n, low, high, Some(11)).expect("should succeed");
        let emp_mean: f64 = samples.iter().sum::<f64>() / n as f64;
        assert!(
            (emp_mean - expected_mean).abs() < 0.1,
            "parallel uniform: empirical mean {emp_mean} too far from {expected_mean}"
        );
    }

    #[test]
    fn test_parallel_uniform_sample_rejects_bad_args() {
        assert!(super::parallel_uniform_sample(0, 0.0, 1.0, None).is_err());
        assert!(super::parallel_uniform_sample(10, 1.0, 1.0, None).is_err());
        assert!(super::parallel_uniform_sample(10, 2.0, 1.0, None).is_err());
    }

    #[test]
    fn test_parallel_uniform_sample_reproducibility() {
        let s1 = super::parallel_uniform_sample(500, 0.0, 1.0, Some(33)).expect("ok");
        let s2 = super::parallel_uniform_sample(500, 0.0, 1.0, Some(33)).expect("ok");
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a, b, "seeded parallel uniform samples should be identical");
        }
    }
}
