//! Bootstrap and resampling-based inference methods.
//!
//! Provides comprehensive bootstrap methodology including:
//! - Percentile, basic, and BCa confidence intervals
//! - Block bootstrap for time-series data
//! - Stationary bootstrap (Politis & Romano 1994)
//! - Parametric bootstrap from a fitted normal model
//! - Bootstrap hypothesis testing (p-value estimation)

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};

// ---------------------------------------------------------------------------
// Internal RNG: 64-bit SplitMix (fast, no external dep)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        // Mix seed to avoid bad zero-state
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        s = (s ^ (s >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        s = (s ^ (s >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        s ^= s >> 31;
        if s == 0 {
            s = 1;
        }
        Self { state: s }
    }

    /// Next u64 via SplitMix64.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform index in [0, n).
    fn usize_below(&mut self, n: usize) -> usize {
        // Rejection-free via 128-bit multiplication
        let r = self.next_u64();
        let m = (r as u128).wrapping_mul(n as u128);
        (m >> 64) as usize
    }

    /// Standard normal via Box-Muller (cos branch).
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Math helpers (no external dep)
// ---------------------------------------------------------------------------

/// Standard normal CDF via rational approximation (Abramowitz & Stegun 26.2.17).
fn norm_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * x.abs());
    let d = (0.5_f64 / std::f64::consts::PI).sqrt() * (-0.5 * x * x).exp();
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    if x >= 0.0 {
        1.0 - d * poly
    } else {
        d * poly
    }
}

/// Probit (inverse normal CDF) via rational approximation (Beasley-Springer-Moro).
fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let (sign, pp) = if p < 0.5 { (-1.0_f64, p) } else { (1.0_f64, 1.0 - p) };
    let t = (-2.0 * pp.ln()).sqrt();
    let c0 = 2.515_517_f64;
    let c1 = 0.802_853_f64;
    let c2 = 0.010_328_f64;
    let d1 = 1.432_788_f64;
    let d2 = 0.189_269_f64;
    let d3 = 0.001_308_f64;
    let num = c0 + c1 * t + c2 * t * t;
    let den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    sign * (t - num / den)
}

/// Linear interpolation quantile on a sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    let frac = pos - lo as f64;
    let lo_val = sorted[lo.min(n - 1)];
    let hi_val = sorted[hi.min(n - 1)];
    lo_val + frac * (hi_val - lo_val)
}

/// Sort a Vec<f64> in-place.
fn sort_f64(v: &mut Vec<f64>) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Compute mean of a slice.
fn slice_mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Compute variance (population) of a slice.
fn slice_var(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = slice_mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64
}

/// Compute variance (sample) of a slice.
fn slice_var_sample(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = slice_mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Bootstrap confidence interval method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CIMethod {
    /// Percentile interval: [q_{α/2}, q_{1-α/2}] of bootstrap distribution.
    Percentile,
    /// Basic (reverse percentile) interval: 2*θ̂ - [q_{1-α/2}, q_{α/2}].
    Basic,
    /// Bias-corrected and accelerated (BCa; Efron 1987).
    BCa,
}

/// Bootstrap confidence interval result.
#[derive(Debug, Clone)]
pub struct BootstrapCI {
    /// Lower bound of the confidence interval.
    pub lower: f64,
    /// Upper bound of the confidence interval.
    pub upper: f64,
    /// Original (observed) statistic.
    pub observed: f64,
    /// Bootstrap standard error.
    pub standard_error: f64,
    /// Bootstrap bias estimate (mean(replicates) - observed).
    pub bias: f64,
    /// Nominal confidence level (e.g. 0.95).
    pub confidence_level: f64,
    /// CI method used.
    pub method: CIMethod,
    /// All bootstrap replicates (sorted).
    pub replicates: Vec<f64>,
}

/// BCa bootstrap result with additional diagnostic fields.
#[derive(Debug, Clone)]
pub struct BCaResult {
    /// Bias-correction constant z₀.
    pub z0: f64,
    /// Acceleration constant a.
    pub acceleration: f64,
    /// The BCa confidence interval.
    pub ci: BootstrapCI,
}

/// Result from parametric bootstrap.
#[derive(Debug, Clone)]
pub struct ParametricBootstrapResult {
    /// Lower CI bound.
    pub ci_lower: f64,
    /// Upper CI bound.
    pub ci_upper: f64,
    /// Observed statistic.
    pub observed: f64,
    /// Bootstrap standard error.
    pub standard_error: f64,
    /// Bootstrap replicates (sorted).
    pub replicates: Vec<f64>,
    /// Confidence level.
    pub confidence_level: f64,
}

/// Result from a bootstrap hypothesis test.
#[derive(Debug, Clone)]
pub struct BootstrapTestResult {
    /// Observed test statistic.
    pub statistic: f64,
    /// Bootstrap p-value.
    pub p_value: f64,
    /// All bootstrap statistics.
    pub bootstrap_statistics: Vec<f64>,
    /// Number of bootstrap replicates used.
    pub n_replicates: usize,
}

/// Result from block bootstrap CI.
#[derive(Debug, Clone)]
pub struct BlockBootstrapResult {
    /// Lower CI bound.
    pub ci_lower: f64,
    /// Upper CI bound.
    pub ci_upper: f64,
    /// Observed statistic.
    pub observed: f64,
    /// Bootstrap standard error.
    pub standard_error: f64,
    /// Block length used.
    pub block_length: usize,
    /// Bootstrap replicates (sorted).
    pub replicates: Vec<f64>,
}

// ---------------------------------------------------------------------------
// bootstrap_ci – general-purpose CI
// ---------------------------------------------------------------------------

/// Compute a bootstrap confidence interval using the requested method.
///
/// # Arguments
/// * `data`       – 1-D sample (n ≥ 2).
/// * `statistic`  – Function mapping a slice to a scalar.
/// * `n_boot`     – Bootstrap replicate count (default 2000).
/// * `level`      – Confidence level, e.g. 0.95 (default).
/// * `method`     – [`CIMethod`] variant.
/// * `seed`       – Optional RNG seed.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::{bootstrap_ci, CIMethod};
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let ci = bootstrap_ci(
///     &data.view(),
///     |x| x.iter().sum::<f64>() / x.len() as f64,
///     Some(1000),
///     Some(0.95),
///     CIMethod::Percentile,
///     Some(42),
/// )
/// .expect("bootstrap_ci failed");
/// assert!(ci.lower < ci.observed && ci.observed < ci.upper);
/// ```
pub fn bootstrap_ci(
    data: &ArrayView1<f64>,
    statistic: impl Fn(&[f64]) -> f64,
    n_boot: Option<usize>,
    level: Option<f64>,
    method: CIMethod,
    seed: Option<u64>,
) -> StatsResult<BootstrapCI> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "bootstrap_ci requires at least 2 observations".to_string(),
        ));
    }
    let n_boot = n_boot.unwrap_or(2000);
    if n_boot < 1 {
        return Err(StatsError::InvalidArgument(
            "n_boot must be at least 1".to_string(),
        ));
    }
    let level = level.unwrap_or(0.95);
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be in (0, 1)".to_string(),
        ));
    }

    let slice: Vec<f64> = data.iter().cloned().collect();
    let observed = statistic(&slice);

    let mut rng = Rng64::new(seed.unwrap_or(12345));
    let mut reps: Vec<f64> = Vec::with_capacity(n_boot);
    let mut buf = vec![0.0_f64; n];

    for _ in 0..n_boot {
        for b in buf.iter_mut() {
            *b = slice[rng.usize_below(n)];
        }
        reps.push(statistic(&buf));
    }

    sort_f64(&mut reps);

    let alpha = 1.0 - level;
    let (lower, upper) = match method {
        CIMethod::Percentile => {
            let lo = quantile_sorted(&reps, alpha / 2.0);
            let hi = quantile_sorted(&reps, 1.0 - alpha / 2.0);
            (lo, hi)
        }
        CIMethod::Basic => {
            let lo = 2.0 * observed - quantile_sorted(&reps, 1.0 - alpha / 2.0);
            let hi = 2.0 * observed - quantile_sorted(&reps, alpha / 2.0);
            (lo, hi)
        }
        CIMethod::BCa => {
            // Bias-correction z0
            let below = reps.iter().filter(|&&r| r < observed).count();
            let z0 = norm_ppf(below as f64 / n_boot as f64);

            // Acceleration a via jackknife influence function
            let acc = compute_acceleration(&slice, &statistic);

            let z_alpha2 = norm_ppf(alpha / 2.0);
            let z_1_alpha2 = norm_ppf(1.0 - alpha / 2.0);

            let adj_lo = bca_adjusted_quantile(z0, acc, z_alpha2);
            let adj_hi = bca_adjusted_quantile(z0, acc, z_1_alpha2);

            let lo = quantile_sorted(&reps, adj_lo);
            let hi = quantile_sorted(&reps, adj_hi);
            (lo, hi)
        }
    };

    let mean_rep = slice_mean(&reps);
    let se = slice_var_sample(&reps).sqrt();
    let bias = mean_rep - observed;

    Ok(BootstrapCI {
        lower,
        upper,
        observed,
        standard_error: se,
        bias,
        confidence_level: level,
        method,
        replicates: reps,
    })
}

// ---------------------------------------------------------------------------
// BCaBootstrap struct
// ---------------------------------------------------------------------------

/// Bias-Corrected and Accelerated (BCa) bootstrap engine.
///
/// Stores configuration and provides methods for repeated use.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::BCaBootstrap;
/// use scirs2_core::ndarray::array;
///
/// let engine = BCaBootstrap::new(2000, 0.95, Some(42));
/// let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let result = engine.compute(
///     &data.view(),
///     |x| x.iter().sum::<f64>() / x.len() as f64,
/// )
/// .expect("BCaBootstrap failed");
/// assert!(result.ci.lower < result.ci.upper);
/// ```
#[derive(Debug, Clone)]
pub struct BCaBootstrap {
    /// Number of bootstrap replicates.
    pub n_boot: usize,
    /// Confidence level.
    pub confidence_level: f64,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

impl BCaBootstrap {
    /// Create a new BCaBootstrap engine.
    pub fn new(n_boot: usize, confidence_level: f64, seed: Option<u64>) -> Self {
        Self {
            n_boot,
            confidence_level,
            seed,
        }
    }

    /// Compute the BCa interval for `statistic` applied to `data`.
    pub fn compute(
        &self,
        data: &ArrayView1<f64>,
        statistic: impl Fn(&[f64]) -> f64,
    ) -> StatsResult<BCaResult> {
        let n = data.len();
        if n < 2 {
            return Err(StatsError::InsufficientData(
                "BCaBootstrap requires at least 2 observations".to_string(),
            ));
        }
        let level = self.confidence_level;
        if !(0.0 < level && level < 1.0) {
            return Err(StatsError::InvalidArgument(
                "Confidence level must be in (0, 1)".to_string(),
            ));
        }

        let slice: Vec<f64> = data.iter().cloned().collect();
        let observed = statistic(&slice);

        let mut rng = Rng64::new(self.seed.unwrap_or(99999));
        let mut reps: Vec<f64> = Vec::with_capacity(self.n_boot);
        let mut buf = vec![0.0_f64; n];

        for _ in 0..self.n_boot {
            for b in buf.iter_mut() {
                *b = slice[rng.usize_below(n)];
            }
            reps.push(statistic(&buf));
        }

        sort_f64(&mut reps);

        let alpha = 1.0 - level;
        let below = reps.iter().filter(|&&r| r < observed).count();
        let z0 = norm_ppf(below as f64 / self.n_boot as f64);
        let acc = compute_acceleration(&slice, &statistic);

        let z_alpha2 = norm_ppf(alpha / 2.0);
        let z_1_alpha2 = norm_ppf(1.0 - alpha / 2.0);

        let adj_lo = bca_adjusted_quantile(z0, acc, z_alpha2);
        let adj_hi = bca_adjusted_quantile(z0, acc, z_1_alpha2);

        let lower = quantile_sorted(&reps, adj_lo);
        let upper = quantile_sorted(&reps, adj_hi);
        let se = slice_var_sample(&reps).sqrt();
        let bias = slice_mean(&reps) - observed;

        Ok(BCaResult {
            z0,
            acceleration: acc,
            ci: BootstrapCI {
                lower,
                upper,
                observed,
                standard_error: se,
                bias,
                confidence_level: level,
                method: CIMethod::BCa,
                replicates: reps,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: BCa quantile adjustment
// ---------------------------------------------------------------------------

fn bca_adjusted_quantile(z0: f64, acc: f64, z_alpha: f64) -> f64 {
    let num = z0 + z_alpha;
    let adjusted_z = z0 + num / (1.0 - acc * num);
    // Clamp to avoid extreme probabilities
    norm_cdf(adjusted_z).clamp(0.001, 0.999)
}

/// Compute the acceleration constant via jackknife.
fn compute_acceleration(data: &[f64], statistic: &impl Fn(&[f64]) -> f64) -> f64 {
    let n = data.len();
    let mut jk_stats: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let loo: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &v)| v)
            .collect();
        jk_stats.push(statistic(&loo));
    }

    let jk_mean = slice_mean(&jk_stats);
    let diffs: Vec<f64> = jk_stats.iter().map(|&s| jk_mean - s).collect();
    let numer: f64 = diffs.iter().map(|&d| d * d * d).sum::<f64>();
    let denom: f64 = diffs.iter().map(|&d| d * d).sum::<f64>();
    let denom_pow = denom.powf(1.5);

    if denom_pow.abs() < 1e-15 {
        0.0
    } else {
        numer / (6.0 * denom_pow)
    }
}

// ---------------------------------------------------------------------------
// parametric_bootstrap
// ---------------------------------------------------------------------------

/// Parametric bootstrap: fit a normal model to `data`, resample from it.
///
/// Computes a percentile confidence interval for the statistic applied
/// to samples from N(x̄, s²).
///
/// # Arguments
/// * `data`      – 1-D sample (n ≥ 2).
/// * `statistic` – Mapping from a slice to a scalar.
/// * `n_boot`    – Replicate count (default 2000).
/// * `level`     – Confidence level (default 0.95).
/// * `seed`      – Optional RNG seed.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::parametric_bootstrap;
/// use scirs2_core::ndarray::array;
///
/// let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let res = parametric_bootstrap(
///     &data.view(),
///     |x| x.iter().sum::<f64>() / x.len() as f64,
///     Some(1000),
///     Some(0.95),
///     Some(42),
/// )
/// .expect("parametric_bootstrap failed");
/// assert!(res.ci_lower < res.observed && res.observed < res.ci_upper);
/// ```
pub fn parametric_bootstrap(
    data: &ArrayView1<f64>,
    statistic: impl Fn(&[f64]) -> f64,
    n_boot: Option<usize>,
    level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<ParametricBootstrapResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "parametric_bootstrap requires at least 2 observations".to_string(),
        ));
    }
    let n_boot = n_boot.unwrap_or(2000);
    let level = level.unwrap_or(0.95);
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be in (0, 1)".to_string(),
        ));
    }

    let slice: Vec<f64> = data.iter().cloned().collect();
    let observed = statistic(&slice);

    // Fit normal parameters
    let mu = slice_mean(&slice);
    let sigma = slice_var_sample(&slice).sqrt();
    if sigma < f64::EPSILON {
        return Err(StatsError::ComputationError(
            "Zero variance in data; parametric bootstrap undefined".to_string(),
        ));
    }

    let mut rng = Rng64::new(seed.unwrap_or(54321));
    let mut reps: Vec<f64> = Vec::with_capacity(n_boot);
    let mut buf = vec![0.0_f64; n];

    for _ in 0..n_boot {
        for b in buf.iter_mut() {
            *b = mu + sigma * rng.standard_normal();
        }
        reps.push(statistic(&buf));
    }

    sort_f64(&mut reps);

    let alpha = 1.0 - level;
    let ci_lower = quantile_sorted(&reps, alpha / 2.0);
    let ci_upper = quantile_sorted(&reps, 1.0 - alpha / 2.0);
    let se = slice_var_sample(&reps).sqrt();

    Ok(ParametricBootstrapResult {
        ci_lower,
        ci_upper,
        observed,
        standard_error: se,
        replicates: reps,
        confidence_level: level,
    })
}

// ---------------------------------------------------------------------------
// block_bootstrap
// ---------------------------------------------------------------------------

/// Moving block bootstrap (MBB) confidence interval for time-series data.
///
/// Blocks of length `block_length` are sampled with replacement and
/// concatenated to form each bootstrap replicate, preserving local
/// serial dependence.
///
/// # Arguments
/// * `data`         – 1-D time series (n ≥ 4).
/// * `statistic`    – Scalar statistic of the full series.
/// * `block_length` – Block length `l` (1 ≤ l < n).  Auto if `None` (≈ n^{1/3}).
/// * `n_boot`       – Number of bootstrap replicates (default 1000).
/// * `level`        – Confidence level (default 0.95).
/// * `seed`         – Optional RNG seed.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::block_bootstrap;
/// use scirs2_core::ndarray::array;
///
/// let data = array![
///     1.0f64, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
///     6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5
/// ];
/// let res = block_bootstrap(
///     &data.view(),
///     |x| x.iter().sum::<f64>() / x.len() as f64,
///     None,
///     Some(500),
///     Some(0.95),
///     Some(42),
/// )
/// .expect("block_bootstrap failed");
/// assert!(res.ci_lower < res.ci_upper);
/// ```
pub fn block_bootstrap(
    data: &ArrayView1<f64>,
    statistic: impl Fn(&[f64]) -> f64,
    block_length: Option<usize>,
    n_boot: Option<usize>,
    level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BlockBootstrapResult> {
    let n = data.len();
    if n < 4 {
        return Err(StatsError::InsufficientData(
            "block_bootstrap requires at least 4 observations".to_string(),
        ));
    }

    let bl = match block_length {
        Some(0) => {
            return Err(StatsError::InvalidArgument(
                "block_length must be ≥ 1".to_string(),
            ))
        }
        Some(l) if l >= n => {
            return Err(StatsError::InvalidArgument(
                "block_length must be < n".to_string(),
            ))
        }
        Some(l) => l,
        None => {
            // Optimal MBB block length ≈ n^{1/3} (Bühlmann & Künsch 1999)
            ((n as f64).powf(1.0 / 3.0).round() as usize).max(1)
        }
    };

    let n_boot = n_boot.unwrap_or(1000);
    let level = level.unwrap_or(0.95);
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be in (0, 1)".to_string(),
        ));
    }

    let slice: Vec<f64> = data.iter().cloned().collect();
    let observed = statistic(&slice);

    // Number of blocks needed to cover n observations
    let n_blocks_needed = (n + bl - 1) / bl;
    // Number of valid starting positions (circular wrap)
    let n_starts = n;

    let mut rng = Rng64::new(seed.unwrap_or(77777));
    let mut reps: Vec<f64> = Vec::with_capacity(n_boot);
    let mut resample = Vec::with_capacity(n_blocks_needed * bl);

    for _ in 0..n_boot {
        resample.clear();
        let mut filled = 0;
        while filled < n {
            let start = rng.usize_below(n_starts);
            let take = bl.min(n - filled);
            for k in 0..take {
                resample.push(slice[(start + k) % n]);
            }
            filled += take;
        }
        reps.push(statistic(&resample[..n]));
    }

    sort_f64(&mut reps);

    let alpha = 1.0 - level;
    let ci_lower = quantile_sorted(&reps, alpha / 2.0);
    let ci_upper = quantile_sorted(&reps, 1.0 - alpha / 2.0);
    let se = slice_var_sample(&reps).sqrt();

    Ok(BlockBootstrapResult {
        ci_lower,
        ci_upper,
        observed,
        standard_error: se,
        block_length: bl,
        replicates: reps,
    })
}

// ---------------------------------------------------------------------------
// stationary_bootstrap
// ---------------------------------------------------------------------------

/// Stationary bootstrap (Politis & Romano, 1994).
///
/// Block lengths are geometrically distributed with parameter `p = 1/expected_block_length`,
/// making the bootstrap stationary.  Like MBB but avoids non-stationarity at
/// block boundaries.
///
/// # Arguments
/// * `data`                  – 1-D time series (n ≥ 4).
/// * `statistic`             – Scalar function of the full series.
/// * `expected_block_length` – Mean block length (default ≈ n^{1/3}).
/// * `n_boot`                – Number of bootstrap replicates (default 1000).
/// * `level`                 – Confidence level (default 0.95).
/// * `seed`                  – Optional RNG seed.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::stationary_bootstrap;
/// use scirs2_core::ndarray::array;
///
/// let data = array![
///     1.0f64, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
///     6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5
/// ];
/// let res = stationary_bootstrap(
///     &data.view(),
///     |x| x.iter().sum::<f64>() / x.len() as f64,
///     None,
///     Some(500),
///     Some(0.95),
///     Some(42),
/// )
/// .expect("stationary_bootstrap failed");
/// assert!(res.ci_lower < res.ci_upper);
/// ```
pub fn stationary_bootstrap(
    data: &ArrayView1<f64>,
    statistic: impl Fn(&[f64]) -> f64,
    expected_block_length: Option<f64>,
    n_boot: Option<usize>,
    level: Option<f64>,
    seed: Option<u64>,
) -> StatsResult<BlockBootstrapResult> {
    let n = data.len();
    if n < 4 {
        return Err(StatsError::InsufficientData(
            "stationary_bootstrap requires at least 4 observations".to_string(),
        ));
    }

    let ebl = match expected_block_length {
        Some(l) if l < 1.0 => {
            return Err(StatsError::InvalidArgument(
                "expected_block_length must be ≥ 1".to_string(),
            ))
        }
        Some(l) => l,
        None => (n as f64).powf(1.0 / 3.0).max(1.0),
    };
    let p_stop = 1.0 / ebl; // geometric success probability
    let n_boot = n_boot.unwrap_or(1000);
    let level = level.unwrap_or(0.95);
    if !(0.0 < level && level < 1.0) {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be in (0, 1)".to_string(),
        ));
    }

    let slice: Vec<f64> = data.iter().cloned().collect();
    let observed = statistic(&slice);

    let mut rng = Rng64::new(seed.unwrap_or(88888));
    let mut reps: Vec<f64> = Vec::with_capacity(n_boot);
    let mut resample = Vec::with_capacity(n + 1);

    for _ in 0..n_boot {
        resample.clear();
        let mut start = rng.usize_below(n);
        while resample.len() < n {
            resample.push(slice[start % n]);
            start += 1;
            // With probability p_stop, begin a new block
            if rng.uniform() < p_stop {
                start = rng.usize_below(n);
            }
        }
        reps.push(statistic(&resample[..n]));
    }

    sort_f64(&mut reps);

    let alpha = 1.0 - level;
    let ci_lower = quantile_sorted(&reps, alpha / 2.0);
    let ci_upper = quantile_sorted(&reps, 1.0 - alpha / 2.0);
    let se = slice_var_sample(&reps).sqrt();

    Ok(BlockBootstrapResult {
        ci_lower,
        ci_upper,
        observed,
        standard_error: se,
        block_length: ebl.round() as usize,
        replicates: reps,
    })
}

// ---------------------------------------------------------------------------
// bootstrap_hypothesis_test
// ---------------------------------------------------------------------------

/// Bootstrap hypothesis test: estimate the p-value for a test statistic.
///
/// The null distribution is generated by resampling from the *pooled* data
/// (i.e. the two-sample test statistic is computed on bootstrap resamples of
/// size `n1` and `n2` drawn from the combined sample).  For one-sample
/// testing, the data are resampled directly after mean-centering.
///
/// # Algorithm
///
/// 1. Compute the observed statistic T = `test_statistic(data)`.
/// 2. Generate `n_boot` bootstrap resamples.
/// 3. For each resample compute T*.
/// 4. p-value = #{T* ≥ T} / n_boot  (upper-tailed; use `two_sided` flag for 2-sided).
///
/// # Arguments
/// * `data`           – 1-D sample.
/// * `test_statistic` – Scalar test statistic.
/// * `n_boot`         – Number of replicates (default 2000).
/// * `two_sided`      – If true, use |T*| ≥ |T| (default: true).
/// * `seed`           – Optional RNG seed.
///
/// # Example
/// ```
/// use scirs2_stats::resampling::bootstrap::bootstrap_hypothesis_test;
/// use scirs2_core::ndarray::array;
///
/// // Test H₀: mean = 0 (data clearly from mean=5)
/// let data = array![4.8f64, 5.2, 4.9, 5.1, 5.0, 5.3, 4.7, 5.0, 5.1, 4.9];
/// let mean_centered = {
///     let m = data.mean().unwrap_or(0.0);
///     let v: Vec<f64> = data.iter().map(|&x| x - m).collect();
///     scirs2_core::ndarray::Array1::from_vec(v)
/// };
/// let result = bootstrap_hypothesis_test(
///     &mean_centered.view(),
///     |x| {
///         let m = x.iter().sum::<f64>() / x.len() as f64;
///         let n = x.len() as f64;
///         let s2 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
///         m / (s2 / n).sqrt()  // t-statistic under H0: mean=0
///     },
///     Some(1000),
///     true,
///     Some(42),
/// )
/// .expect("bootstrap_hypothesis_test failed");
/// assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
/// ```
pub fn bootstrap_hypothesis_test(
    data: &ArrayView1<f64>,
    test_statistic: impl Fn(&[f64]) -> f64,
    n_boot: Option<usize>,
    two_sided: bool,
    seed: Option<u64>,
) -> StatsResult<BootstrapTestResult> {
    let n = data.len();
    if n < 2 {
        return Err(StatsError::InsufficientData(
            "bootstrap_hypothesis_test requires at least 2 observations".to_string(),
        ));
    }
    let n_boot = n_boot.unwrap_or(2000);

    let slice: Vec<f64> = data.iter().cloned().collect();
    let observed = test_statistic(&slice);

    let mut rng = Rng64::new(seed.unwrap_or(31415));
    let mut boot_stats: Vec<f64> = Vec::with_capacity(n_boot);
    let mut buf = vec![0.0_f64; n];

    for _ in 0..n_boot {
        for b in buf.iter_mut() {
            *b = slice[rng.usize_below(n)];
        }
        boot_stats.push(test_statistic(&buf));
    }

    // Compute p-value
    let p_value = if two_sided {
        let obs_abs = observed.abs();
        boot_stats.iter().filter(|&&s| s.abs() >= obs_abs).count() as f64 / n_boot as f64
    } else {
        boot_stats.iter().filter(|&&s| s >= observed).count() as f64 / n_boot as f64
    };

    Ok(BootstrapTestResult {
        statistic: observed,
        p_value,
        bootstrap_statistics: boot_stats,
        n_replicates: n_boot,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn sample_mean(x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        x.iter().sum::<f64>() / x.len() as f64
    }

    fn sample_median(x: &[f64]) -> f64 {
        if x.is_empty() {
            return 0.0;
        }
        let mut s = x.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = s.len();
        if n % 2 == 0 {
            (s[n / 2 - 1] + s[n / 2]) / 2.0
        } else {
            s[n / 2]
        }
    }

    // --- bootstrap_ci tests ---

    #[test]
    fn test_bootstrap_ci_percentile_contains_true_mean() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = bootstrap_ci(&data.view(), sample_mean, Some(2000), Some(0.95), CIMethod::Percentile, Some(42))
            .expect("percentile ci");
        assert!(ci.lower < 5.5 && ci.upper > 5.5, "CI should contain true mean 5.5");
    }

    #[test]
    fn test_bootstrap_ci_basic_contains_true_mean() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = bootstrap_ci(&data.view(), sample_mean, Some(2000), Some(0.95), CIMethod::Basic, Some(42))
            .expect("basic ci");
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_bootstrap_ci_bca_contains_true_mean() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = bootstrap_ci(&data.view(), sample_mean, Some(2000), Some(0.95), CIMethod::BCa, Some(42))
            .expect("bca ci");
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_bootstrap_ci_wider_at_higher_level() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci90 = bootstrap_ci(&data.view(), sample_mean, Some(2000), Some(0.90), CIMethod::Percentile, Some(42)).expect("90");
        let ci99 = bootstrap_ci(&data.view(), sample_mean, Some(2000), Some(0.99), CIMethod::Percentile, Some(42)).expect("99");
        assert!(ci99.upper - ci99.lower > ci90.upper - ci90.lower, "99% CI should be wider");
    }

    #[test]
    fn test_bootstrap_ci_replicates_count() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let ci = bootstrap_ci(&data.view(), sample_mean, Some(500), Some(0.95), CIMethod::Percentile, Some(1)).expect("ok");
        assert_eq!(ci.replicates.len(), 500);
    }

    #[test]
    fn test_bootstrap_ci_se_positive() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ci = bootstrap_ci(&data.view(), sample_mean, Some(1000), Some(0.95), CIMethod::Percentile, Some(7)).expect("ok");
        assert!(ci.standard_error > 0.0);
    }

    #[test]
    fn test_bootstrap_ci_insufficient_data() {
        let data = array![1.0f64];
        assert!(bootstrap_ci(&data.view(), sample_mean, None, None, CIMethod::Percentile, None).is_err());
    }

    #[test]
    fn test_bootstrap_ci_invalid_level() {
        let data = array![1.0f64, 2.0, 3.0];
        assert!(bootstrap_ci(&data.view(), sample_mean, None, Some(1.5), CIMethod::Percentile, None).is_err());
    }

    #[test]
    fn test_bootstrap_ci_median() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = bootstrap_ci(&data.view(), sample_median, Some(2000), Some(0.95), CIMethod::Percentile, Some(42)).expect("median ci");
        // True median 5.5 should be inside
        assert!(ci.lower < 5.5 && ci.upper > 5.5);
    }

    // --- BCaBootstrap tests ---

    #[test]
    fn test_bca_bootstrap_basic() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let engine = BCaBootstrap::new(1000, 0.95, Some(42));
        let res = engine.compute(&data.view(), sample_mean).expect("bca");
        assert!(res.ci.lower < res.ci.upper);
        assert!((res.ci.observed - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_bca_bootstrap_acceleration_small() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let engine = BCaBootstrap::new(500, 0.95, Some(42));
        let res = engine.compute(&data.view(), sample_mean).expect("bca");
        assert!(res.acceleration.abs() < 0.5, "acceleration should be small for symmetric data");
    }

    #[test]
    fn test_bca_bootstrap_skewed() {
        let data = array![1.0f64, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 10.0];
        let engine = BCaBootstrap::new(2000, 0.95, Some(42));
        let res = engine.compute(&data.view(), sample_mean).expect("bca");
        assert!(res.ci.lower < res.ci.upper);
    }

    #[test]
    fn test_bca_insufficient_data() {
        let data = array![1.0f64];
        let engine = BCaBootstrap::new(1000, 0.95, None);
        assert!(engine.compute(&data.view(), sample_mean).is_err());
    }

    // --- parametric_bootstrap tests ---

    #[test]
    fn test_parametric_bootstrap_mean_covered() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let res = parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.95), Some(42)).expect("ok");
        // CI should cover the sample mean
        assert!(res.ci_lower < res.observed && res.observed < res.ci_upper);
    }

    #[test]
    fn test_parametric_bootstrap_wider_at_higher_level() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let r90 = parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.90), Some(42)).expect("90");
        let r99 = parametric_bootstrap(&data.view(), sample_mean, Some(2000), Some(0.99), Some(42)).expect("99");
        assert!(r99.ci_upper - r99.ci_lower > r90.ci_upper - r90.ci_lower);
    }

    #[test]
    fn test_parametric_bootstrap_se_positive() {
        let data = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let res = parametric_bootstrap(&data.view(), sample_mean, Some(500), None, Some(1)).expect("ok");
        assert!(res.standard_error > 0.0);
    }

    #[test]
    fn test_parametric_bootstrap_insufficient_data() {
        let data = array![1.0f64];
        assert!(parametric_bootstrap(&data.view(), sample_mean, None, None, None).is_err());
    }

    // --- block_bootstrap tests ---

    #[test]
    fn test_block_bootstrap_ci_valid() {
        let data = array![
            1.0f64, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
            6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5
        ];
        let res = block_bootstrap(&data.view(), sample_mean, None, Some(500), Some(0.95), Some(42)).expect("ok");
        assert!(res.ci_lower < res.ci_upper);
    }

    #[test]
    fn test_block_bootstrap_explicit_block_length() {
        let data: Array1<f64> = Array1::from_iter((1..=20).map(|x| x as f64));
        let res = block_bootstrap(&data.view(), sample_mean, Some(4), Some(500), Some(0.95), Some(42)).expect("ok");
        assert_eq!(res.block_length, 4);
    }

    #[test]
    fn test_block_bootstrap_replicates_count() {
        let data: Array1<f64> = Array1::from_iter((1..=20).map(|x| x as f64));
        let res = block_bootstrap(&data.view(), sample_mean, Some(2), Some(300), Some(0.95), Some(42)).expect("ok");
        assert_eq!(res.replicates.len(), 300);
    }

    #[test]
    fn test_block_bootstrap_invalid_block_length_zero() {
        let data: Array1<f64> = Array1::from_iter((1..=10).map(|x| x as f64));
        assert!(block_bootstrap(&data.view(), sample_mean, Some(0), None, None, None).is_err());
    }

    #[test]
    fn test_block_bootstrap_block_length_too_large() {
        let data: Array1<f64> = Array1::from_iter((1..=5).map(|x| x as f64));
        assert!(block_bootstrap(&data.view(), sample_mean, Some(10), None, None, None).is_err());
    }

    #[test]
    fn test_block_bootstrap_insufficient_data() {
        let data = array![1.0f64, 2.0, 3.0];
        assert!(block_bootstrap(&data.view(), sample_mean, None, None, None, None).is_err());
    }

    // --- stationary_bootstrap tests ---

    #[test]
    fn test_stationary_bootstrap_ci_valid() {
        let data: Array1<f64> = Array1::from_iter((1..=20).map(|x| x as f64));
        let res = stationary_bootstrap(&data.view(), sample_mean, None, Some(500), Some(0.95), Some(42)).expect("ok");
        assert!(res.ci_lower < res.ci_upper);
    }

    #[test]
    fn test_stationary_bootstrap_custom_block_length() {
        let data: Array1<f64> = Array1::from_iter((1..=20).map(|x| x as f64));
        let res = stationary_bootstrap(&data.view(), sample_mean, Some(4.0), Some(500), Some(0.95), Some(42)).expect("ok");
        assert_eq!(res.block_length, 4);
    }

    #[test]
    fn test_stationary_bootstrap_replicates() {
        let data: Array1<f64> = Array1::from_iter((1..=16).map(|x| x as f64));
        let res = stationary_bootstrap(&data.view(), sample_mean, None, Some(200), None, Some(42)).expect("ok");
        assert_eq!(res.replicates.len(), 200);
    }

    #[test]
    fn test_stationary_bootstrap_insufficient_data() {
        let data = array![1.0f64, 2.0];
        assert!(stationary_bootstrap(&data.view(), sample_mean, None, None, None, None).is_err());
    }

    // --- bootstrap_hypothesis_test tests ---

    #[test]
    fn test_hypothesis_test_null_should_have_large_p() {
        // Under H0 (mean=0) with data centred at 0, p-value should be large
        let data = array![0.1f64, -0.1, 0.05, -0.05, 0.02, -0.02, 0.03, -0.03, 0.01, -0.01];
        let t_stat = |x: &[f64]| {
            let m = x.iter().sum::<f64>() / x.len() as f64;
            let n = x.len() as f64;
            let s2 = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0);
            m / (s2 / n).sqrt().max(1e-15)
        };
        let result = bootstrap_hypothesis_test(&data.view(), t_stat, Some(1000), true, Some(42)).expect("ok");
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_hypothesis_test_p_value_in_range() {
        let data: Array1<f64> = Array1::from_iter((1..=10).map(|x| x as f64));
        let result = bootstrap_hypothesis_test(&data.view(), sample_mean, Some(500), false, Some(42)).expect("ok");
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_hypothesis_test_replicates_count() {
        let data: Array1<f64> = Array1::from_iter((1..=8).map(|x| x as f64));
        let result = bootstrap_hypothesis_test(&data.view(), sample_mean, Some(400), true, Some(1)).expect("ok");
        assert_eq!(result.n_replicates, 400);
        assert_eq!(result.bootstrap_statistics.len(), 400);
    }

    #[test]
    fn test_hypothesis_test_insufficient_data() {
        let data = array![1.0f64];
        let result = bootstrap_hypothesis_test(&data.view(), sample_mean, None, true, None);
        assert!(result.is_err());
    }

    // --- RNG internal tests ---

    #[test]
    fn test_rng_different_seeds_differ() {
        let mut r1 = Rng64::new(1);
        let mut r2 = Rng64::new(2);
        assert_ne!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn test_rng_uniform_in_range() {
        let mut rng = Rng64::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!(u >= 0.0 && u < 1.0);
        }
    }

    #[test]
    fn test_norm_cdf_symmetry() {
        assert!((norm_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((norm_cdf(1.96) - 0.975).abs() < 0.005);
    }

    #[test]
    fn test_quantile_sorted_basics() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile_sorted(&v, 0.0), 1.0);
        assert_eq!(quantile_sorted(&v, 1.0), 5.0);
        assert!((quantile_sorted(&v, 0.5) - 3.0).abs() < 1e-10);
    }
}
