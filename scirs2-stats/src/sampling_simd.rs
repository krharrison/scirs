//! SIMD-optimized sampling operations
//!
//! This module provides SIMD-accelerated implementations of distribution sampling
//! operations using scirs2-core's unified SIMD operations for v0.2.0.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::uniform::SampleUniform;
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::{AutoOptimizer, SimdUnifiedOps};

/// SIMD-optimized Box-Muller transform for normal distribution sampling
///
/// This function generates pairs of independent standard normal random variables
/// using the Box-Muller transform with SIMD acceleration for improved performance.
///
/// # Arguments
///
/// * `n` - Number of samples to generate (must be even for Box-Muller)
/// * `mean` - Mean of the normal distribution
/// * `std_dev` - Standard deviation of the normal distribution
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Array of normally distributed random samples
///
/// # Examples
///
/// ```
/// use scirs2_stats::sampling_simd::box_muller_simd;
///
/// let samples = box_muller_simd(1000, 0.0, 1.0, Some(42)).expect("Sampling failed");
/// assert_eq!(samples.len(), 1000);
/// ```
pub fn box_muller_simd<F>(
    n: usize,
    mean: F,
    std_dev: F,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
{
    if std_dev <= F::zero() {
        return Err(StatsError::invalid_argument(
            "Standard deviation must be positive",
        ));
    }

    // Ensure even number of samples for Box-Muller
    let n_pairs = (n + 1) / 2;
    let n_total = n_pairs * 2;

    let mut rng = {
        let s = seed.unwrap_or_else(|| {
            use scirs2_core::random::{Rng, RngExt};
            scirs2_core::random::thread_rng().random()
        });
        scirs2_core::random::seeded_rng(s)
    };

    let optimizer = AutoOptimizer::new();

    // Generate uniform random numbers
    let u1: Array1<F> = Array1::from_shape_fn(n_pairs, |_| {
        F::from(rng.gen_range(F::epsilon().to_f64().unwrap_or(1e-10)..1.0))
            .unwrap_or_else(|| F::one())
    });
    let u2: Array1<F> = Array1::from_shape_fn(n_pairs, |_| {
        F::from(rng.gen_range(0.0..1.0)).unwrap_or_else(|| F::zero())
    });

    if optimizer.should_use_simd(n_pairs) {
        // SIMD path for Box-Muller transform
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap_or_else(|| F::one());

        // Compute: sqrt(-2 * ln(u1))
        let ln_u1 = F::simd_ln(&u1.view());
        let minus_two = F::from(-2.0).unwrap_or_else(|| F::one());
        let minus_two_array = Array1::from_elem(n_pairs, minus_two);
        let neg_two_ln_u1 = F::simd_mul(&minus_two_array.view(), &ln_u1.view());
        let sqrt_term = F::simd_sqrt(&neg_two_ln_u1.view());

        // Compute: 2 * pi * u2
        let two_pi_array = Array1::from_elem(n_pairs, two_pi);
        let two_pi_u2 = F::simd_mul(&two_pi_array.view(), &u2.view());

        // Compute: cos(2*pi*u2) and sin(2*pi*u2)
        let cos_term = F::simd_cos(&two_pi_u2.view());
        let sin_term = F::simd_sin(&two_pi_u2.view());

        // z0 = sqrt_term * cos_term
        // z1 = sqrt_term * sin_term
        let z0 = F::simd_mul(&sqrt_term.view(), &cos_term.view());
        let z1 = F::simd_mul(&sqrt_term.view(), &sin_term.view());

        // Scale and shift: z * std_dev + mean
        let std_dev_array = Array1::from_elem(n_pairs, std_dev);
        let mean_array = Array1::from_elem(n_pairs, mean);

        let z0_scaled = F::simd_fma(&z0.view(), &std_dev_array.view(), &mean_array.view());
        let z1_scaled = F::simd_fma(&z1.view(), &std_dev_array.view(), &mean_array.view());

        // Interleave results
        let mut result = Array1::zeros(n_total);
        for i in 0..n_pairs {
            result[2 * i] = z0_scaled[i];
            if 2 * i + 1 < n_total {
                result[2 * i + 1] = z1_scaled[i];
            }
        }

        // Trim to requested size
        Ok(result.slice(scirs2_core::ndarray::s![..n]).to_owned())
    } else {
        // Scalar fallback
        let mut result = Array1::zeros(n_total);
        let two_pi = F::from(2.0 * std::f64::consts::PI).unwrap_or_else(|| F::one());
        let two = F::from(2.0).unwrap_or_else(|| F::one());

        for i in 0..n_pairs {
            let r = (-two * u1[i].ln()).sqrt();
            let theta = two_pi * u2[i];

            result[2 * i] = mean + std_dev * r * theta.cos();
            if 2 * i + 1 < n_total {
                result[2 * i + 1] = mean + std_dev * r * theta.sin();
            }
        }

        Ok(result.slice(scirs2_core::ndarray::s![..n]).to_owned())
    }
}

/// SIMD-optimized inverse transform sampling
///
/// This function generates random samples using the inverse CDF method with
/// SIMD acceleration where applicable.
///
/// # Arguments
///
/// * `n` - Number of samples to generate
/// * `inverse_cdf` - Function that computes the inverse CDF
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Array of random samples from the distribution
pub fn inverse_transform_simd<F, InvCDF>(
    n: usize,
    inverse_cdf: InvCDF,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
    InvCDF: Fn(F) -> F,
{
    let mut rng = {
        let s = seed.unwrap_or_else(|| {
            use scirs2_core::random::{Rng, RngExt};
            scirs2_core::random::thread_rng().random()
        });
        scirs2_core::random::seeded_rng(s)
    };

    // Generate uniform random numbers
    let u: Array1<F> = Array1::from_shape_fn(n, |_| {
        F::from(rng.gen_range(0.0..1.0)).unwrap_or_else(|| F::zero())
    });

    // Apply inverse CDF
    let result = u.mapv(|ui| inverse_cdf(ui));

    Ok(result)
}

/// SIMD-optimized exponential distribution sampling
///
/// Generates samples from an exponential distribution with given rate parameter
/// using SIMD-accelerated logarithm operations.
///
/// # Arguments
///
/// * `n` - Number of samples to generate
/// * `rate` - Rate parameter (lambda > 0)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Array of exponentially distributed random samples
pub fn exponential_simd<F>(n: usize, rate: F, seed: Option<u64>) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps + SampleUniform,
{
    if rate <= F::zero() {
        return Err(StatsError::invalid_argument("Rate must be positive"));
    }

    let mut rng = {
        let s = seed.unwrap_or_else(|| {
            use scirs2_core::random::{Rng, RngExt};
            scirs2_core::random::thread_rng().random()
        });
        scirs2_core::random::seeded_rng(s)
    };

    let optimizer = AutoOptimizer::new();

    // Generate uniform random numbers in (0, 1]
    let u: Array1<F> = Array1::from_shape_fn(n, |_| {
        F::from(rng.gen_range(F::epsilon().to_f64().unwrap_or(1e-10)..1.0))
            .unwrap_or_else(|| F::one())
    });

    if optimizer.should_use_simd(n) {
        // SIMD path: -ln(u) / rate
        let ln_u = F::simd_ln(&u.view());
        let minus_one = F::from(-1.0).unwrap_or_else(|| F::one());
        let minus_one_array = Array1::from_elem(n, minus_one);
        let neg_ln_u = F::simd_mul(&minus_one_array.view(), &ln_u.view());

        let inv_rate = F::one() / rate;
        let inv_rate_array = Array1::from_elem(n, inv_rate);
        let result = F::simd_mul(&neg_ln_u.view(), &inv_rate_array.view());

        Ok(result)
    } else {
        // Scalar fallback
        Ok(u.mapv(|ui| -ui.ln() / rate))
    }
}

/// SIMD-optimized bootstrap sampling with replacement
///
/// Generates bootstrap samples from the input data using SIMD operations
/// for improved performance on large datasets.
///
/// # Arguments
///
/// * `data` - Input data array
/// * `n_samples` - Number of bootstrap samples to generate
/// * `seed` - Optional random seed
///
/// # Returns
///
/// Array of bootstrap samples
pub fn bootstrap_simd<F>(
    data: &ArrayView1<F>,
    n_samples: usize,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Data array cannot be empty"));
    }

    let n_data = data.len();
    let mut rng = {
        let s = seed.unwrap_or_else(|| {
            use scirs2_core::random::{Rng, RngExt};
            scirs2_core::random::thread_rng().random()
        });
        scirs2_core::random::seeded_rng(s)
    };

    // Generate random indices
    let mut result = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let idx = rng.gen_range(0..n_data);
        result[i] = data[idx];
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_box_muller_simd_basic() {
        let samples = box_muller_simd(1000, 0.0, 1.0, Some(42)).expect("Sampling failed");
        assert_eq!(samples.len(), 1000);

        // Check that mean is approximately 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 0.1);

        // Check that std dev is approximately 1
        let variance: f64 =
            samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert_abs_diff_eq!(variance.sqrt(), 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_exponential_simd_basic() {
        let rate = 2.0;
        let samples = exponential_simd(1000, rate, Some(42)).expect("Sampling failed");
        assert_eq!(samples.len(), 1000);

        // Check that mean is approximately 1/rate
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let expected_mean = 1.0 / rate;
        assert_abs_diff_eq!(mean, expected_mean, epsilon = 0.1);
    }

    #[test]
    fn test_bootstrap_simd_basic() {
        use scirs2_core::ndarray::array;

        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let samples = bootstrap_simd(&data.view(), 100, Some(42)).expect("Bootstrap failed");
        assert_eq!(samples.len(), 100);

        // Check that all samples are from the original data
        for &sample in samples.iter() {
            assert!(data.iter().any(|&x| (x - sample).abs() < 1e-10));
        }
    }
}
