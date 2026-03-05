//! Bootstrap resampling methods for statistical inference
//!
//! This module provides focused implementations of the most important bootstrap
//! methods for constructing confidence intervals and performing statistical tests.
//!
//! ## Methods provided
//!
//! - **Percentile bootstrap CI**: Simple and widely used
//! - **BCa bootstrap**: Bias-corrected and accelerated (Efron 1987)
//! - **Parametric bootstrap**: Resample from a fitted parametric model
//! - **Block bootstrap**: For dependent / time series data

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Draw `n` samples with replacement from `data` using a simple LCG RNG.
/// Returns a new `Array1<F>`.
fn resample_with_replacement<F: Float + NumCast>(data: &[F], rng_state: &mut u64) -> Vec<F> {
    let n = data.len();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let idx = lcg_next(rng_state) as usize % n;
        out.push(data[idx]);
    }
    out
}

/// Simple LCG pseudo-random number generator (period 2^64).
/// Returns a positive u64 and updates state in place.
fn lcg_next(state: &mut u64) -> u64 {
    // Knuth MMIX LCG parameters
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state >> 1 // make positive
}

fn sorted_f64_vec(v: &[f64]) -> Vec<f64> {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

fn quantile_f64(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = q * (sorted.len() as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo.min(sorted.len() - 1)] * (1.0 - frac) + sorted[hi.min(sorted.len() - 1)] * frac
}

/// Standard normal CDF (Abramowitz & Stegun approximation).
fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * x.abs());
    let d = 0.398_942_28 * (-0.5 * x * x).exp();
    let p = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    if x >= 0.0 {
        1.0 - d * p
    } else {
        d * p
    }
}

/// Standard normal quantile (probit).
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < f64::EPSILON {
        return 0.0;
    }
    let (sign, pp) = if p < 0.5 { (-1.0, p) } else { (1.0, 1.0 - p) };
    let t = (-2.0 * pp.ln()).sqrt();
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;
    sign * (t - (c0 + t * (c1 + t * c2)) / (1.0 + t * (d1 + t * (d2 + t * d3))))
}

// ===========================================================================
// Result types
// ===========================================================================

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapCI<F> {
    /// Point estimate (statistic applied to the original sample)
    pub estimate: F,
    /// Lower bound of the confidence interval
    pub ci_lower: F,
    /// Upper bound of the confidence interval
    pub ci_upper: F,
    /// Confidence level (e.g. 0.95)
    pub confidence_level: f64,
    /// Standard error estimate (std dev of bootstrap distribution)
    pub standard_error: F,
    /// Bootstrap distribution (all bootstrap replicates)
    pub replicates: Vec<F>,
}

/// BCa bootstrap result (extends BootstrapCI with bias/acceleration info).
#[derive(Debug, Clone)]
pub struct BcaBootstrapResult<F> {
    /// The confidence interval
    pub ci: BootstrapCI<F>,
    /// Bias correction factor (z0)
    pub bias_correction: f64,
    /// Acceleration factor (a)
    pub acceleration: f64,
}

// ===========================================================================
// Percentile bootstrap
// ===========================================================================

/// Compute a percentile bootstrap confidence interval.
///
/// The percentile method (Efron 1979) is the simplest bootstrap CI method.
/// It takes the alpha/2 and 1-alpha/2 quantiles of the bootstrap distribution
/// as the CI bounds.
///
/// # Arguments
///
/// * `x` - Input data
/// * `statistic` - A function that computes the statistic of interest from a slice
/// * `n_bootstrap` - Number of bootstrap resamples (default 1000)
/// * `confidence_level` - Confidence level (default 0.95)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// A `BootstrapCI` with the confidence interval and bootstrap distribution.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::percentile_bootstrap;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = percentile_bootstrap(
///     &data.view(),
///     |s| s.iter().sum::<f64>() / s.len() as f64,  // mean
///     Some(2000),
///     Some(0.95),
///     Some(42),
/// ).expect("bootstrap failed");
/// assert!(result.ci_lower < 5.5);
/// assert!(result.ci_upper > 5.5);
/// ```
pub fn percentile_bootstrap<F, S>(
    x: &ArrayView1<F>,
    statistic: S,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BootstrapCI<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
    S: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    let n_boot = n_bootstrap.unwrap_or(1000);
    let conf = confidence_level.unwrap_or(0.95);
    if conf <= 0.0 || conf >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "confidence_level must be in (0, 1)".to_string(),
        ));
    }

    let data_f64: Vec<f64> = x
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .collect();

    let theta_hat = statistic(&data_f64);

    let mut rng_state = seed.unwrap_or(12345);
    let mut replicates_f64 = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let sample = resample_with_replacement(&data_f64, &mut rng_state);
        replicates_f64.push(statistic(&sample));
    }

    let sorted_reps = sorted_f64_vec(&replicates_f64);
    let alpha = 1.0 - conf;
    let ci_lower_f64 = quantile_f64(&sorted_reps, alpha / 2.0);
    let ci_upper_f64 = quantile_f64(&sorted_reps, 1.0 - alpha / 2.0);

    // Standard error
    let mean_rep = replicates_f64.iter().sum::<f64>() / replicates_f64.len() as f64;
    let var_rep = replicates_f64
        .iter()
        .map(|&r| (r - mean_rep) * (r - mean_rep))
        .sum::<f64>()
        / (replicates_f64.len() as f64 - 1.0);
    let se = var_rep.sqrt();

    let replicates: Result<Vec<F>, _> = replicates_f64
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();

    Ok(BootstrapCI {
        estimate: F::from(theta_hat).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_lower: F::from(ci_lower_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_upper: F::from(ci_upper_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        confidence_level: conf,
        standard_error: F::from(se).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        replicates: replicates?,
    })
}

// ===========================================================================
// BCa bootstrap
// ===========================================================================

/// Compute a BCa (bias-corrected and accelerated) bootstrap confidence interval.
///
/// The BCa method (Efron 1987) adjusts the percentile bootstrap CI for both
/// bias and skewness in the bootstrap distribution. It uses:
///
/// 1. **Bias correction (z0)**: the proportion of bootstrap replicates less
///    than the original estimate, converted to a z-score.
/// 2. **Acceleration (a)**: estimated from the jackknife influence values.
///
/// BCa intervals are second-order accurate and generally preferred over
/// simple percentile intervals.
///
/// # Arguments
///
/// * `x` - Input data
/// * `statistic` - Function computing the statistic from a slice
/// * `n_bootstrap` - Number of bootstrap resamples (default 2000)
/// * `confidence_level` - Confidence level (default 0.95)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// A `BcaBootstrapResult` with the CI, bias correction, and acceleration.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::bca_bootstrap;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = bca_bootstrap(
///     &data.view(),
///     |s| s.iter().sum::<f64>() / s.len() as f64,
///     Some(2000),
///     Some(0.95),
///     Some(42),
/// ).expect("BCa failed");
/// assert!(result.ci.ci_lower < 5.5);
/// assert!(result.ci.ci_upper > 5.5);
/// ```
pub fn bca_bootstrap<F, S>(
    x: &ArrayView1<F>,
    statistic: S,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BcaBootstrapResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
    S: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "Need at least 2 observations for BCa bootstrap".to_string(),
        ));
    }

    let n_boot = n_bootstrap.unwrap_or(2000);
    let conf = confidence_level.unwrap_or(0.95);
    if conf <= 0.0 || conf >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "confidence_level must be in (0, 1)".to_string(),
        ));
    }

    let data_f64: Vec<f64> = x
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .collect();

    let theta_hat = statistic(&data_f64);

    // Generate bootstrap replicates
    let mut rng_state = seed.unwrap_or(12345);
    let mut replicates_f64 = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        let sample = resample_with_replacement(&data_f64, &mut rng_state);
        replicates_f64.push(statistic(&sample));
    }

    // Bias correction: z0 = Phi^{-1}(proportion of replicates < theta_hat)
    let prop_less =
        replicates_f64.iter().filter(|&&r| r < theta_hat).count() as f64 / n_boot as f64;
    let z0 = norm_ppf(prop_less.max(0.001).min(0.999));

    // Acceleration: a = sum(d_i^3) / (6 * (sum(d_i^2))^{3/2})
    // where d_i = theta_hat_dot - theta_hat_jack_i (jackknife influence values)
    let mut jackknife_vals = Vec::with_capacity(n);
    for i in 0..n {
        let jack_sample: Vec<f64> = data_f64
            .iter()
            .enumerate()
            .filter_map(|(j, &v)| if j != i { Some(v) } else { None })
            .collect();
        jackknife_vals.push(statistic(&jack_sample));
    }

    let theta_dot = jackknife_vals.iter().sum::<f64>() / n as f64;
    let d: Vec<f64> = jackknife_vals.iter().map(|&t| theta_dot - t).collect();

    let sum_d2: f64 = d.iter().map(|&di| di * di).sum();
    let sum_d3: f64 = d.iter().map(|&di| di * di * di).sum();

    let acceleration = if sum_d2 > f64::EPSILON {
        sum_d3 / (6.0 * sum_d2.powf(1.5))
    } else {
        0.0
    };

    // Adjusted quantiles
    let alpha = 1.0 - conf;
    let z_alpha_lower = norm_ppf(alpha / 2.0);
    let z_alpha_upper = norm_ppf(1.0 - alpha / 2.0);

    let adjusted_lower = {
        let num = z0 + z_alpha_lower;
        let denom = 1.0 - acceleration * num;
        if denom.abs() > f64::EPSILON {
            norm_cdf(z0 + num / denom)
        } else {
            alpha / 2.0
        }
    };

    let adjusted_upper = {
        let num = z0 + z_alpha_upper;
        let denom = 1.0 - acceleration * num;
        if denom.abs() > f64::EPSILON {
            norm_cdf(z0 + num / denom)
        } else {
            1.0 - alpha / 2.0
        }
    };

    let sorted_reps = sorted_f64_vec(&replicates_f64);
    let ci_lower_f64 = quantile_f64(&sorted_reps, adjusted_lower.max(0.0).min(1.0));
    let ci_upper_f64 = quantile_f64(&sorted_reps, adjusted_upper.max(0.0).min(1.0));

    // Standard error
    let mean_rep = replicates_f64.iter().sum::<f64>() / replicates_f64.len() as f64;
    let var_rep = replicates_f64
        .iter()
        .map(|&r| (r - mean_rep) * (r - mean_rep))
        .sum::<f64>()
        / (replicates_f64.len() as f64 - 1.0);
    let se = var_rep.sqrt();

    let replicates: Result<Vec<F>, _> = replicates_f64
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();

    Ok(BcaBootstrapResult {
        ci: BootstrapCI {
            estimate: F::from(theta_hat)
                .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
            ci_lower: F::from(ci_lower_f64)
                .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
            ci_upper: F::from(ci_upper_f64)
                .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
            confidence_level: conf,
            standard_error: F::from(se)
                .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
            replicates: replicates?,
        },
        bias_correction: z0,
        acceleration,
    })
}

// ===========================================================================
// Parametric bootstrap
// ===========================================================================

/// Compute a parametric bootstrap confidence interval.
///
/// Instead of resampling the data directly, the parametric bootstrap fits a
/// parametric model to the data and then draws samples from that model.
/// This version assumes a normal model: the mean and standard deviation are
/// estimated from the data, and new samples are drawn from N(mean, sd^2).
///
/// # Arguments
///
/// * `x` - Input data
/// * `statistic` - Function computing the statistic from a slice
/// * `n_bootstrap` - Number of bootstrap resamples (default 1000)
/// * `confidence_level` - Confidence level (default 0.95)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// A `BootstrapCI` with the confidence interval.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::parametric_bootstrap;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = parametric_bootstrap(
///     &data.view(),
///     |s| s.iter().sum::<f64>() / s.len() as f64,
///     Some(1000),
///     Some(0.95),
///     Some(42),
/// ).expect("parametric bootstrap failed");
/// assert!(result.ci_lower < 5.5);
/// assert!(result.ci_upper > 5.5);
/// ```
pub fn parametric_bootstrap<F, S>(
    x: &ArrayView1<F>,
    statistic: S,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BootstrapCI<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
    S: Fn(&[f64]) -> f64,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "Need at least 2 observations for parametric bootstrap".to_string(),
        ));
    }

    let n_boot = n_bootstrap.unwrap_or(1000);
    let conf = confidence_level.unwrap_or(0.95);

    let data_f64: Vec<f64> = x
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .collect();

    let theta_hat = statistic(&data_f64);

    // Fit normal model: estimate mean and std dev
    let sample_mean = data_f64.iter().sum::<f64>() / n as f64;
    let sample_var = data_f64
        .iter()
        .map(|&v| (v - sample_mean) * (v - sample_mean))
        .sum::<f64>()
        / (n as f64 - 1.0);
    let sample_sd = sample_var.sqrt();

    // Generate parametric bootstrap samples using Box-Muller transform
    let mut rng_state = seed.unwrap_or(12345);
    let mut replicates_f64 = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let mut sample = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            // Box-Muller
            let u1 = (lcg_next(&mut rng_state) as f64) / (u64::MAX as f64 / 2.0);
            let u2 = (lcg_next(&mut rng_state) as f64) / (u64::MAX as f64 / 2.0);
            let u1 = u1.max(1e-300); // avoid log(0)
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;

            let z1 = r * theta.cos();
            sample.push(sample_mean + sample_sd * z1);
            i += 1;

            if i < n {
                let z2 = r * theta.sin();
                sample.push(sample_mean + sample_sd * z2);
                i += 1;
            }
        }
        replicates_f64.push(statistic(&sample));
    }

    let sorted_reps = sorted_f64_vec(&replicates_f64);
    let alpha = 1.0 - conf;
    let ci_lower_f64 = quantile_f64(&sorted_reps, alpha / 2.0);
    let ci_upper_f64 = quantile_f64(&sorted_reps, 1.0 - alpha / 2.0);

    let mean_rep = replicates_f64.iter().sum::<f64>() / replicates_f64.len() as f64;
    let var_rep = replicates_f64
        .iter()
        .map(|&r| (r - mean_rep) * (r - mean_rep))
        .sum::<f64>()
        / (replicates_f64.len() as f64 - 1.0);
    let se = var_rep.sqrt();

    let replicates: Result<Vec<F>, _> = replicates_f64
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();

    Ok(BootstrapCI {
        estimate: F::from(theta_hat).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_lower: F::from(ci_lower_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_upper: F::from(ci_upper_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        confidence_level: conf,
        standard_error: F::from(se).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        replicates: replicates?,
    })
}

// ===========================================================================
// Block bootstrap (for time series)
// ===========================================================================

/// Compute a block bootstrap confidence interval for dependent data.
///
/// The moving block bootstrap (Kunsch 1989, Liu & Singh 1992) divides the
/// time series into overlapping blocks of a fixed length and resamples entire
/// blocks. This preserves the serial dependence within each block.
///
/// # Arguments
///
/// * `x` - Input time series data
/// * `statistic` - Function computing the statistic from a slice
/// * `block_length` - Length of each block (default: floor(n^(1/3)))
/// * `n_bootstrap` - Number of bootstrap resamples (default 1000)
/// * `confidence_level` - Confidence level (default 0.95)
/// * `seed` - Optional random seed
///
/// # Returns
///
/// A `BootstrapCI` with the confidence interval.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::block_bootstrap_ci;
///
/// // Simulated time series
/// let data = array![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
///                    6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5];
/// let result = block_bootstrap_ci(
///     &data.view(),
///     |s| s.iter().sum::<f64>() / s.len() as f64,
///     None,       // auto block length
///     Some(1000),
///     Some(0.95),
///     Some(42),
/// ).expect("block bootstrap failed");
/// assert!(result.ci_lower < result.ci_upper);
/// ```
pub fn block_bootstrap_ci<F, S>(
    x: &ArrayView1<F>,
    statistic: S,
    block_length: Option<usize>,
    n_bootstrap: Option<usize>,
    confidence_level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BootstrapCI<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
    S: Fn(&[f64]) -> f64,
{
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "Need at least 3 observations for block bootstrap".to_string(),
        ));
    }

    let block_len = block_length.unwrap_or_else(|| {
        let auto = (n as f64).powf(1.0 / 3.0).ceil() as usize;
        auto.max(1).min(n)
    });

    if block_len == 0 || block_len > n {
        return Err(StatsError::InvalidArgument(format!(
            "block_length must be between 1 and {} (n)",
            n
        )));
    }

    let n_boot = n_bootstrap.unwrap_or(1000);
    let conf = confidence_level.unwrap_or(0.95);

    let data_f64: Vec<f64> = x
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .collect();

    let theta_hat = statistic(&data_f64);

    // Number of possible starting positions for overlapping blocks
    let n_blocks_possible = n - block_len + 1;
    // Number of blocks needed to fill n observations
    let n_blocks_needed = (n as f64 / block_len as f64).ceil() as usize;

    let mut rng_state = seed.unwrap_or(12345);
    let mut replicates_f64 = Vec::with_capacity(n_boot);

    for _ in 0..n_boot {
        let mut sample = Vec::with_capacity(n_blocks_needed * block_len);

        for _ in 0..n_blocks_needed {
            let start = lcg_next(&mut rng_state) as usize % n_blocks_possible;
            for j in 0..block_len {
                sample.push(data_f64[start + j]);
            }
        }

        // Trim to exactly n observations
        sample.truncate(n);

        replicates_f64.push(statistic(&sample));
    }

    let sorted_reps = sorted_f64_vec(&replicates_f64);
    let alpha = 1.0 - conf;
    let ci_lower_f64 = quantile_f64(&sorted_reps, alpha / 2.0);
    let ci_upper_f64 = quantile_f64(&sorted_reps, 1.0 - alpha / 2.0);

    let mean_rep = replicates_f64.iter().sum::<f64>() / replicates_f64.len() as f64;
    let var_rep = replicates_f64
        .iter()
        .map(|&r| (r - mean_rep) * (r - mean_rep))
        .sum::<f64>()
        / (replicates_f64.len() as f64 - 1.0);
    let se = var_rep.sqrt();

    let replicates: Result<Vec<F>, _> = replicates_f64
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();

    Ok(BootstrapCI {
        estimate: F::from(theta_hat).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_lower: F::from(ci_lower_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        ci_upper: F::from(ci_upper_f64)
            .ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        confidence_level: conf,
        standard_error: F::from(se).ok_or_else(|| StatsError::ComputationError("cast".into()))?,
        replicates: replicates?,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    fn sample_mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn sample_median(data: &[f64]) -> f64 {
        let mut s = data.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 0 {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        } else {
            s[n / 2]
        }
    }

    // -------------------------------------------------------------------
    // Percentile bootstrap
    // -------------------------------------------------------------------

    #[test]
    fn test_percentile_bootstrap_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result =
            percentile_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42))
                .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        // True mean is 5.5; CI should contain it
        assert!(lower < 5.5);
        assert!(upper > 5.5);
    }

    #[test]
    fn test_percentile_bootstrap_median() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = percentile_bootstrap(
            &data.view(),
            sample_median,
            Some(2000),
            Some(0.95),
            Some(42),
        )
        .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        assert!(lower < upper);
    }

    #[test]
    fn test_percentile_bootstrap_se() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result =
            percentile_bootstrap(&data.view(), sample_mean, Some(1000), Some(0.95), Some(42))
                .expect("should succeed");
        let se: f64 = NumCast::from(result.standard_error).expect("cast");
        assert!(se > 0.0);
        assert!(se < 3.0); // SE of mean should be roughly sqrt(var/n) ~ 0.96
    }

    #[test]
    fn test_percentile_bootstrap_replicates_count() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let n_boot = 500;
        let result = percentile_bootstrap(&data.view(), sample_mean, Some(n_boot), None, Some(42))
            .expect("should succeed");
        assert_eq!(result.replicates.len(), n_boot);
    }

    #[test]
    fn test_percentile_bootstrap_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(percentile_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    #[test]
    fn test_percentile_bootstrap_single_element() {
        let data = array![5.0];
        let result = percentile_bootstrap(&data.view(), sample_mean, Some(100), None, Some(42))
            .expect("should succeed");
        // All resamples are [5.0], so CI should be [5, 5]
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        assert!((lower - 5.0).abs() < 1e-10);
        assert!((upper - 5.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // BCa bootstrap
    // -------------------------------------------------------------------

    #[test]
    fn test_bca_bootstrap_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = bca_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42))
            .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci.ci_upper).expect("cast");
        assert!(lower < 5.5);
        assert!(upper > 5.5);
    }

    #[test]
    fn test_bca_bootstrap_bias_correction() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = bca_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42))
            .expect("should succeed");
        // z0 should be near 0 for symmetric distribution and mean statistic
        assert!(result.bias_correction.abs() < 1.0);
    }

    #[test]
    fn test_bca_bootstrap_acceleration() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = bca_bootstrap(&data.view(), sample_mean, Some(1000), Some(0.95), Some(42))
            .expect("should succeed");
        // Acceleration should be small for mean of uniform-ish data
        assert!(result.acceleration.abs() < 0.5);
    }

    #[test]
    fn test_bca_bootstrap_skewed_data() {
        // Skewed data: BCa should adjust
        let data = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 10.0];
        let result = bca_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42))
            .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci.ci_upper).expect("cast");
        assert!(lower < upper);
    }

    #[test]
    fn test_bca_bootstrap_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(bca_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    #[test]
    fn test_bca_bootstrap_single_error() {
        let data = array![5.0];
        assert!(bca_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    // -------------------------------------------------------------------
    // Parametric bootstrap
    // -------------------------------------------------------------------

    #[test]
    fn test_parametric_bootstrap_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result =
            parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42))
                .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        assert!(lower < 5.5);
        assert!(upper > 5.5);
    }

    #[test]
    fn test_parametric_bootstrap_ci_width() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci_90 =
            parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.90), Some(42))
                .expect("90%");
        let ci_99 =
            parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.99), Some(42))
                .expect("99%");

        let width_90: f64 = ci_90.ci_upper - ci_90.ci_lower;
        let width_99: f64 = ci_99.ci_upper - ci_99.ci_lower;
        // 99% CI should be wider than 90% CI
        assert!(width_99 > width_90);
    }

    #[test]
    fn test_parametric_bootstrap_se_positive() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = parametric_bootstrap(&data.view(), sample_mean, Some(500), None, Some(42))
            .expect("should succeed");
        let se: f64 = NumCast::from(result.standard_error).expect("cast");
        assert!(se > 0.0);
    }

    #[test]
    fn test_parametric_bootstrap_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(parametric_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    #[test]
    fn test_parametric_bootstrap_single_error() {
        let data = array![5.0];
        assert!(parametric_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    // -------------------------------------------------------------------
    // Block bootstrap
    // -------------------------------------------------------------------

    #[test]
    fn test_block_bootstrap_basic() {
        let data = array![
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
            9.5, 10.0, 10.5
        ];
        let result = block_bootstrap_ci(
            &data.view(),
            sample_mean,
            None,
            Some(1000),
            Some(0.95),
            Some(42),
        )
        .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        assert!(lower < upper);
    }

    #[test]
    fn test_block_bootstrap_custom_block_length() {
        let data =
            array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let result = block_bootstrap_ci(
            &data.view(),
            sample_mean,
            Some(3),
            Some(500),
            Some(0.95),
            Some(42),
        )
        .expect("should succeed");
        let lower: f64 = NumCast::from(result.ci_lower).expect("cast");
        let upper: f64 = NumCast::from(result.ci_upper).expect("cast");
        assert!(lower < upper);
    }

    #[test]
    fn test_block_bootstrap_block_length_1() {
        // Block length 1 should behave like ordinary bootstrap
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = block_bootstrap_ci(
            &data.view(),
            sample_mean,
            Some(1),
            Some(500),
            Some(0.95),
            Some(42),
        )
        .expect("should succeed");
        assert_eq!(result.replicates.len(), 500);
    }

    #[test]
    fn test_block_bootstrap_replicates() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let n_boot = 300;
        let result = block_bootstrap_ci(
            &data.view(),
            sample_mean,
            Some(2),
            Some(n_boot),
            None,
            Some(42),
        )
        .expect("should succeed");
        assert_eq!(result.replicates.len(), n_boot);
    }

    #[test]
    fn test_block_bootstrap_too_small_error() {
        let data = array![1.0, 2.0];
        assert!(block_bootstrap_ci(&data.view(), sample_mean, None, None, None, None).is_err());
    }

    #[test]
    fn test_block_bootstrap_invalid_block_length() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(block_bootstrap_ci(&data.view(), sample_mean, Some(0), None, None, None).is_err());
        assert!(block_bootstrap_ci(&data.view(), sample_mean, Some(6), None, None, None).is_err());
    }

    // -------------------------------------------------------------------
    // Helper tests
    // -------------------------------------------------------------------

    #[test]
    fn test_norm_cdf_ppf_roundtrip() {
        for &p in &[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99] {
            let z = norm_ppf(p);
            let p_back = norm_cdf(z);
            assert!(
                (p - p_back).abs() < 0.02,
                "roundtrip failed for p={}: z={}, p_back={}",
                p,
                z,
                p_back
            );
        }
    }

    #[test]
    fn test_lcg_different_seeds() {
        let mut s1 = 1u64;
        let mut s2 = 2u64;
        let v1 = lcg_next(&mut s1);
        let v2 = lcg_next(&mut s2);
        assert_ne!(v1, v2);
    }
}
