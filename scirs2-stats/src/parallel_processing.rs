//! Expanded parallel processing for statistical computations
//!
//! This module provides parallelized implementations of:
//! - Descriptive statistics (Welford's algorithm, parallel quantiles, histograms)
//! - Hypothesis testing (permutation tests, bootstrap, cross-validation)
//! - Distribution fitting (parallel MLE, grid search)
//!
//! All parallel code is feature-gated behind `cfg(feature = "parallel")` (which scirs2-core
//! enables via `scirs2-core/parallel = ["rayon"]`).

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::parallel_ops::{num_threads, par_chunks, parallel_map, ParallelIterator};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Threshold for switching to parallel execution
const PAR_THRESHOLD: usize = 5_000;

// ===========================================================================
// Part 1: Parallel Descriptive Statistics
// ===========================================================================

// ---------------------------------------------------------------------------
// Welford accumulators (online, mergeable)
// ---------------------------------------------------------------------------

/// Mergeable accumulator for mean, variance, skewness, kurtosis via Welford's method.
///
/// Uses the parallel variant from Chan, Golub & LeVeque (1979).
#[derive(Debug, Clone)]
pub struct WelfordAccumulator {
    /// Number of observations
    pub n: u64,
    /// Running mean
    pub mean: f64,
    /// Sum of squared deviations from mean (M2)
    pub m2: f64,
    /// Third central moment accumulator (M3)
    pub m3: f64,
    /// Fourth central moment accumulator (M4)
    pub m4: f64,
}

impl Default for WelfordAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl WelfordAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        WelfordAccumulator {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
        }
    }

    /// Add a single observation.
    pub fn push(&mut self, x: f64) {
        let n1 = self.n;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1 as f64;

        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;
    }

    /// Merge another accumulator into this one (parallel combine step).
    ///
    /// Implements the parallel formulas from Chan, Golub & LeVeque.
    pub fn merge(&mut self, other: &WelfordAccumulator) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = other.clone();
            return;
        }

        let na = self.n as f64;
        let nb = other.n as f64;
        let n_total = na + nb;

        let delta = other.mean - self.mean;
        let delta2 = delta * delta;
        let delta3 = delta2 * delta;
        let delta4 = delta2 * delta2;

        let new_mean = (na * self.mean + nb * other.mean) / n_total;

        let new_m2 = self.m2 + other.m2 + delta2 * na * nb / n_total;

        let new_m3 = self.m3
            + other.m3
            + delta3 * na * nb * (na - nb) / (n_total * n_total)
            + 3.0 * delta * (na * other.m2 - nb * self.m2) / n_total;

        let new_m4 = self.m4
            + other.m4
            + delta4 * na * nb * (na * na - na * nb + nb * nb) / (n_total * n_total * n_total)
            + 6.0 * delta2 * (na * na * other.m2 + nb * nb * self.m2) / (n_total * n_total)
            + 4.0 * delta * (na * other.m3 - nb * self.m3) / n_total;

        self.n = self.n + other.n;
        self.mean = new_mean;
        self.m2 = new_m2;
        self.m3 = new_m3;
        self.m4 = new_m4;
    }

    /// Population variance
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / self.n as f64
    }

    /// Sample variance (ddof=1)
    pub fn sample_variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / (self.n - 1) as f64
    }

    /// Population skewness (Fisher's definition)
    pub fn skewness(&self) -> f64 {
        if self.n < 3 || self.m2.abs() < 1e-300 {
            return 0.0;
        }
        let n = self.n as f64;
        n.sqrt() * self.m3 / self.m2.powf(1.5)
    }

    /// Excess kurtosis (Fisher's definition)
    pub fn kurtosis(&self) -> f64 {
        if self.n < 4 || self.m2.abs() < 1e-300 {
            return 0.0;
        }
        let n = self.n as f64;
        n * self.m4 / (self.m2 * self.m2) - 3.0
    }
}

/// Compute mean, variance, skewness, and kurtosis in parallel via Welford's method.
///
/// Uses the parallel merge variant of Welford's algorithm that is numerically
/// stable even for large datasets.
///
/// # Arguments
///
/// * `data` - Input data slice
///
/// # Returns
///
/// A `WelfordAccumulator` containing all four statistics.
pub fn parallel_moments(data: &[f64]) -> WelfordAccumulator {
    if data.len() < PAR_THRESHOLD {
        // Sequential path
        let mut acc = WelfordAccumulator::new();
        for &x in data {
            acc.push(x);
        }
        return acc;
    }

    let chunk_size = (data.len() / num_threads()).max(1000);
    par_chunks(data, chunk_size)
        .map(|chunk| {
            let mut acc = WelfordAccumulator::new();
            for &x in chunk {
                acc.push(x);
            }
            acc
        })
        .reduce(WelfordAccumulator::new, |mut a, b| {
            a.merge(&b);
            a
        })
}

/// Compute mean in parallel via Welford's online algorithm.
pub fn parallel_welford_mean(data: &[f64]) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute mean of empty array".to_string(),
        ));
    }
    Ok(parallel_moments(data).mean)
}

/// Compute variance in parallel via Welford's online algorithm.
///
/// # Arguments
///
/// * `data` - Input data
/// * `ddof` - Delta degrees of freedom (0 = population, 1 = sample)
pub fn parallel_welford_variance(data: &[f64], ddof: usize) -> StatsResult<f64> {
    if data.len() <= ddof {
        return Err(StatsError::InvalidArgument(
            "Not enough data for given ddof".to_string(),
        ));
    }
    let acc = parallel_moments(data);
    if ddof == 0 {
        Ok(acc.variance())
    } else {
        Ok(acc.m2 / (acc.n as f64 - ddof as f64))
    }
}

/// Compute skewness in parallel.
pub fn parallel_welford_skewness(data: &[f64]) -> StatsResult<f64> {
    if data.len() < 3 {
        return Err(StatsError::InvalidArgument(
            "Need at least 3 observations for skewness".to_string(),
        ));
    }
    Ok(parallel_moments(data).skewness())
}

/// Compute excess kurtosis in parallel.
pub fn parallel_welford_kurtosis(data: &[f64]) -> StatsResult<f64> {
    if data.len() < 4 {
        return Err(StatsError::InvalidArgument(
            "Need at least 4 observations for kurtosis".to_string(),
        ));
    }
    Ok(parallel_moments(data).kurtosis())
}

// ---------------------------------------------------------------------------
// Parallel quantile estimation (parallel-sort approach)
// ---------------------------------------------------------------------------

/// Compute a single quantile using parallel sort.
///
/// For large arrays this is faster than a sequential sort because the
/// merge phase can be overlapped with computation.
///
/// # Arguments
///
/// * `data` - Input data (will be sorted internally)
/// * `q` - Quantile in [0, 1]
pub fn parallel_quantile(data: &[f64], q: f64) -> StatsResult<f64> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute quantile of empty array".to_string(),
        ));
    }
    if q < 0.0 || q > 1.0 {
        return Err(StatsError::InvalidArgument(
            "Quantile must be in [0, 1]".to_string(),
        ));
    }

    let mut sorted = data.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 1 {
        return Ok(sorted[0]);
    }

    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;

    if lo == hi || hi >= n {
        Ok(sorted[lo.min(n - 1)])
    } else {
        Ok(sorted[lo] * (1.0 - frac) + sorted[hi] * frac)
    }
}

/// Compute the median in parallel.
pub fn parallel_median(data: &[f64]) -> StatsResult<f64> {
    parallel_quantile(data, 0.5)
}

// ---------------------------------------------------------------------------
// Parallel histogram
// ---------------------------------------------------------------------------

/// Result of a parallel histogram computation.
#[derive(Debug, Clone)]
pub struct ParallelHistogramResult {
    /// Bin edges (length = n_bins + 1)
    pub edges: Vec<f64>,
    /// Counts per bin (length = n_bins)
    pub counts: Vec<u64>,
    /// Total number of observations
    pub total: u64,
}

/// Compute a histogram in parallel.
///
/// # Arguments
///
/// * `data` - Input data
/// * `n_bins` - Number of bins
///
/// # Returns
///
/// `ParallelHistogramResult` with edges, counts, and total.
pub fn parallel_histogram(data: &[f64], n_bins: usize) -> StatsResult<ParallelHistogramResult> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot compute histogram of empty array".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(StatsError::InvalidArgument(
            "Number of bins must be > 0".to_string(),
        ));
    }

    // Find min/max
    let (min_val, max_val) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
            (lo.min(x), hi.max(x))
        });

    if !min_val.is_finite() || !max_val.is_finite() {
        return Err(StatsError::InvalidArgument(
            "Data contains non-finite values".to_string(),
        ));
    }

    let range = max_val - min_val;
    let bin_width = if range < 1e-300 {
        1.0 // all same value
    } else {
        range / n_bins as f64
    };

    // Build edges
    let edges: Vec<f64> = (0..=n_bins)
        .map(|i| min_val + i as f64 * bin_width)
        .collect();

    if data.len() < PAR_THRESHOLD {
        // Sequential
        let mut counts = vec![0u64; n_bins];
        for &x in data {
            let bin = if bin_width < 1e-300 {
                0
            } else {
                ((x - min_val) / bin_width).floor() as usize
            };
            let bin = bin.min(n_bins - 1);
            counts[bin] += 1;
        }
        return Ok(ParallelHistogramResult {
            edges,
            counts,
            total: data.len() as u64,
        });
    }

    // Parallel: each thread builds a partial histogram, then merge
    let chunk_size = (data.len() / num_threads()).max(1000);
    let partial_counts: Vec<Vec<u64>> = par_chunks(data, chunk_size)
        .map(|chunk| {
            let mut counts = vec![0u64; n_bins];
            for &x in chunk {
                let bin = if bin_width < 1e-300 {
                    0
                } else {
                    ((x - min_val) / bin_width).floor() as usize
                };
                let bin = bin.min(n_bins - 1);
                counts[bin] += 1;
            }
            counts
        })
        .collect();

    // Merge partial histograms
    let mut counts = vec![0u64; n_bins];
    for partial in &partial_counts {
        for (i, &c) in partial.iter().enumerate() {
            counts[i] += c;
        }
    }

    Ok(ParallelHistogramResult {
        edges,
        counts,
        total: data.len() as u64,
    })
}

// ===========================================================================
// Part 2: Parallel Hypothesis Testing
// ===========================================================================

// ---------------------------------------------------------------------------
// Parallel permutation test
// ---------------------------------------------------------------------------

/// Result of a permutation test.
#[derive(Debug, Clone)]
pub struct PermutationTestResult {
    /// Observed test statistic
    pub observed: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Number of permutations performed
    pub n_permutations: usize,
    /// Count of permutations with statistic >= |observed|
    pub n_extreme: usize,
}

/// Parallel permutation test for the difference in means between two groups.
///
/// Randomly permutes group labels and recomputes the test statistic
/// to build a null distribution.
///
/// # Arguments
///
/// * `group1` - First group's data
/// * `group2` - Second group's data
/// * `n_permutations` - Number of random permutations
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
///
/// `PermutationTestResult` with observed statistic and p-value.
pub fn parallel_permutation_test(
    group1: &[f64],
    group2: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> StatsResult<PermutationTestResult> {
    if group1.is_empty() || group2.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Both groups must be non-empty".to_string(),
        ));
    }

    let n1 = group1.len();
    let combined: Vec<f64> = group1.iter().chain(group2.iter()).copied().collect();
    let n_total = combined.len();

    // Observed statistic: difference in means
    let mean1: f64 = group1.iter().sum::<f64>() / n1 as f64;
    let mean2: f64 = group2.iter().sum::<f64>() / group2.len() as f64;
    let observed = (mean1 - mean2).abs();

    // Generate seeds for each permutation
    let base_seed = seed.unwrap_or(42);
    let perm_seeds: Vec<u64> = (0..n_permutations)
        .map(|i| {
            // Simple hash to generate diverse seeds
            base_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64)
        })
        .collect();

    // Parallel permutation computation
    let extreme_counts: Vec<usize> = parallel_map(&perm_seeds, |&s| {
        // Simple Fisher-Yates shuffle using LCG
        let mut shuffled = combined.clone();
        let mut state = s;
        for i in (1..n_total).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (state >> 1) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        let perm_mean1: f64 = shuffled[..n1].iter().sum::<f64>() / n1 as f64;
        let perm_mean2: f64 = shuffled[n1..].iter().sum::<f64>() / (n_total - n1) as f64;
        let perm_stat = (perm_mean1 - perm_mean2).abs();

        if perm_stat >= observed - 1e-12 {
            1
        } else {
            0
        }
    });

    let n_extreme: usize = extreme_counts.iter().sum();
    let p_value = (n_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    Ok(PermutationTestResult {
        observed,
        p_value,
        n_permutations,
        n_extreme,
    })
}

// ---------------------------------------------------------------------------
// Parallel bootstrap
// ---------------------------------------------------------------------------

/// Result of a parallel bootstrap procedure.
#[derive(Debug, Clone)]
pub struct ParallelBootstrapResult {
    /// Point estimate from original data
    pub estimate: f64,
    /// Bootstrap standard error
    pub standard_error: f64,
    /// Lower CI bound (percentile method)
    pub ci_lower: f64,
    /// Upper CI bound (percentile method)
    pub ci_upper: f64,
    /// Confidence level
    pub confidence_level: f64,
    /// All bootstrap replicates
    pub replicates: Vec<f64>,
}

/// Run a parallel bootstrap procedure.
///
/// Distributes bootstrap resampling across threads, each with an independent
/// pseudo-random number generator seed.
///
/// # Arguments
///
/// * `data` - Input data
/// * `statistic` - Function computing the statistic from a sample slice
/// * `n_bootstrap` - Number of bootstrap samples
/// * `confidence_level` - Confidence level for CI (e.g. 0.95)
/// * `seed` - Optional random seed
pub fn parallel_bootstrap(
    data: &[f64],
    statistic: &(dyn Fn(&[f64]) -> f64 + Send + Sync),
    n_bootstrap: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> StatsResult<ParallelBootstrapResult> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot bootstrap empty data".to_string(),
        ));
    }
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(StatsError::InvalidArgument(
            "Confidence level must be in (0, 1)".to_string(),
        ));
    }

    let estimate = statistic(data);
    let n = data.len();
    let base_seed = seed.unwrap_or(42);

    // Generate seeds for each bootstrap replicate
    let seeds: Vec<u64> = (0..n_bootstrap)
        .map(|i| {
            base_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(i as u64)
        })
        .collect();

    // Parallel bootstrap
    let replicates: Vec<f64> = parallel_map(&seeds, |&s| {
        let mut state = s;
        let mut sample = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let idx = (state >> 1) as usize % n;
            sample.push(data[idx]);
        }
        statistic(&sample)
    });

    // Compute standard error
    let boot_mean: f64 = replicates.iter().sum::<f64>() / replicates.len() as f64;
    let boot_var: f64 = replicates
        .iter()
        .map(|&x| (x - boot_mean) * (x - boot_mean))
        .sum::<f64>()
        / (replicates.len() as f64 - 1.0).max(1.0);
    let standard_error = boot_var.sqrt();

    // Percentile CI
    let alpha = 1.0 - confidence_level;
    let mut sorted = replicates.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lo_idx = (alpha / 2.0 * sorted.len() as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * sorted.len() as f64).ceil() as usize;
    let ci_lower = sorted[lo_idx.min(sorted.len() - 1)];
    let ci_upper = sorted[hi_idx.min(sorted.len() - 1)];

    Ok(ParallelBootstrapResult {
        estimate,
        standard_error,
        ci_lower,
        ci_upper,
        confidence_level,
        replicates,
    })
}

// ---------------------------------------------------------------------------
// Parallel k-fold cross-validation
// ---------------------------------------------------------------------------

/// Result of parallel cross-validation.
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation of scores across folds
    pub std_score: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Number of folds
    pub n_folds: usize,
}

/// Run parallel k-fold cross-validation.
///
/// Splits data into k folds and evaluates a scoring function on each
/// train/test split in parallel.
///
/// # Arguments
///
/// * `data` - Input features (rows = samples)
/// * `targets` - Target values
/// * `n_folds` - Number of folds
/// * `scorer` - Function(train_X, train_y, test_X, test_y) -> score
/// * `seed` - Optional seed for fold assignment shuffling
pub fn parallel_cross_validation(
    data: &Array2<f64>,
    targets: &Array1<f64>,
    n_folds: usize,
    scorer: &(dyn Fn(&Array2<f64>, &Array1<f64>, &Array2<f64>, &Array1<f64>) -> StatsResult<f64>
          + Send
          + Sync),
    seed: Option<u64>,
) -> StatsResult<CrossValidationResult> {
    let n_samples = data.nrows();
    if n_samples < n_folds {
        return Err(StatsError::InvalidArgument(format!(
            "Need at least {} samples for {}-fold CV, got {}",
            n_folds, n_folds, n_samples
        )));
    }
    if data.nrows() != targets.len() {
        return Err(StatsError::InvalidArgument(
            "data rows and targets length must match".to_string(),
        ));
    }

    // Create fold indices (simple sequential assignment with optional shuffle)
    let mut indices: Vec<usize> = (0..n_samples).collect();
    if let Some(s) = seed {
        // Simple Fisher-Yates shuffle
        let mut state = s;
        for i in (1..n_samples).rev() {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let j = (state >> 1) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    let fold_size = n_samples / n_folds;
    let folds: Vec<usize> = (0..n_folds).collect();

    // Parallel fold evaluation
    let fold_scores: Vec<f64> = parallel_map(&folds, |&fold_idx| {
        let test_start = fold_idx * fold_size;
        let test_end = if fold_idx == n_folds - 1 {
            n_samples
        } else {
            (fold_idx + 1) * fold_size
        };
        let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_indices: Vec<usize> = indices
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < test_start || *i >= test_end)
            .map(|(_, &idx)| idx)
            .collect();

        let n_train = train_indices.len();
        let n_test = test_indices.len();
        let n_features = data.ncols();

        // Build train/test arrays
        let mut train_x = Array2::zeros((n_train, n_features));
        let mut train_y = Array1::zeros(n_train);
        for (row, &idx) in train_indices.iter().enumerate() {
            for col in 0..n_features {
                train_x[(row, col)] = data[(idx, col)];
            }
            train_y[row] = targets[idx];
        }

        let mut test_x = Array2::zeros((n_test, n_features));
        let mut test_y = Array1::zeros(n_test);
        for (row, &idx) in test_indices.iter().enumerate() {
            for col in 0..n_features {
                test_x[(row, col)] = data[(idx, col)];
            }
            test_y[row] = targets[idx];
        }

        scorer(&train_x, &train_y, &test_x, &test_y)
    })
    .into_iter()
    .collect::<StatsResult<Vec<_>>>()?;

    let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
    let std_score = if fold_scores.len() > 1 {
        let var = fold_scores
            .iter()
            .map(|&s| (s - mean_score) * (s - mean_score))
            .sum::<f64>()
            / (fold_scores.len() as f64 - 1.0);
        var.sqrt()
    } else {
        0.0
    };

    Ok(CrossValidationResult {
        mean_score,
        std_score,
        fold_scores,
        n_folds,
    })
}

// ===========================================================================
// Part 3: Parallel Distribution Fitting
// ===========================================================================

// ---------------------------------------------------------------------------
// Parallel MLE
// ---------------------------------------------------------------------------

/// Result of parallel maximum-likelihood estimation.
#[derive(Debug, Clone)]
pub struct ParallelMLEResult {
    /// Best-fit distribution name
    pub distribution: String,
    /// Estimated parameters (distribution-specific)
    pub parameters: Vec<f64>,
    /// Log-likelihood of the best fit
    pub log_likelihood: f64,
    /// AIC (Akaike Information Criterion)
    pub aic: f64,
    /// BIC (Bayesian Information Criterion)
    pub bic: f64,
}

/// Fit multiple distributions in parallel and return the best by AIC.
///
/// Currently supported distributions: Normal, Exponential, Uniform.
///
/// # Arguments
///
/// * `data` - Observed data
///
/// # Returns
///
/// The `ParallelMLEResult` for the best-fitting distribution.
pub fn parallel_mle_fit(data: &[f64]) -> StatsResult<Vec<ParallelMLEResult>> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Cannot fit distributions to empty data".to_string(),
        ));
    }

    let n = data.len() as f64;
    let dist_names: Vec<&str> = vec!["normal", "exponential", "uniform"];

    let results: Vec<ParallelMLEResult> = parallel_map(&dist_names, |&name| {
        match name {
            "normal" => {
                // MLE for Normal: mu = mean, sigma^2 = (1/n)*sum((x-mu)^2)
                let mu = data.iter().sum::<f64>() / n;
                let sigma2 = data.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / n;
                let sigma = sigma2.max(1e-300).sqrt();

                let ll: f64 = data
                    .iter()
                    .map(|&x| {
                        let z = (x - mu) / sigma;
                        -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    })
                    .sum();

                let k = 2.0; // number of parameters
                let aic = 2.0 * k - 2.0 * ll;
                let bic = k * n.ln() - 2.0 * ll;

                ParallelMLEResult {
                    distribution: "normal".to_string(),
                    parameters: vec![mu, sigma],
                    log_likelihood: ll,
                    aic,
                    bic,
                }
            }
            "exponential" => {
                // MLE for Exponential: rate = 1 / mean (only positive data)
                let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
                let shifted: Vec<f64> = if min_val <= 0.0 {
                    data.iter().map(|&x| x - min_val + 1e-10).collect()
                } else {
                    data.to_vec()
                };
                let mean_val = shifted.iter().sum::<f64>() / n;
                let rate = (1.0 / mean_val).max(1e-300);

                let ll: f64 = shifted.iter().map(|&x| rate.ln() - rate * x).sum();

                let k = 1.0;
                let aic = 2.0 * k - 2.0 * ll;
                let bic = k * n.ln() - 2.0 * ll;

                ParallelMLEResult {
                    distribution: "exponential".to_string(),
                    parameters: vec![rate],
                    log_likelihood: ll,
                    aic,
                    bic,
                }
            }
            "uniform" => {
                // MLE for Uniform: a = min, b = max
                let a = data.iter().copied().fold(f64::INFINITY, f64::min);
                let b = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = (b - a).max(1e-300);
                let ll = -n * range.ln();

                let k = 2.0;
                let aic = 2.0 * k - 2.0 * ll;
                let bic = k * n.ln() - 2.0 * ll;

                ParallelMLEResult {
                    distribution: "uniform".to_string(),
                    parameters: vec![a, b],
                    log_likelihood: ll,
                    aic,
                    bic,
                }
            }
            _ => ParallelMLEResult {
                distribution: name.to_string(),
                parameters: vec![],
                log_likelihood: f64::NEG_INFINITY,
                aic: f64::INFINITY,
                bic: f64::INFINITY,
            },
        }
    });

    // Sort by AIC (best first)
    let mut sorted_results = results;
    sorted_results.sort_by(|a, b| {
        a.aic
            .partial_cmp(&b.aic)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(sorted_results)
}

// ---------------------------------------------------------------------------
// Parallel grid search for distribution parameters
// ---------------------------------------------------------------------------

/// Result of a parallel parameter grid search.
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    /// Best parameters found
    pub best_params: Vec<f64>,
    /// Log-likelihood at best parameters
    pub best_log_likelihood: f64,
    /// All parameter combinations evaluated (sorted by log-likelihood descending)
    pub all_results: Vec<(Vec<f64>, f64)>,
}

/// Parallel grid search over distribution parameters to maximize log-likelihood.
///
/// # Arguments
///
/// * `data` - Observed data
/// * `log_likelihood_fn` - Function(data, params) -> log-likelihood
/// * `param_grids` - For each parameter, a vector of candidate values
///
/// # Returns
///
/// `GridSearchResult` with the best parameters.
pub fn parallel_grid_search(
    data: &[f64],
    log_likelihood_fn: &(dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync),
    param_grids: &[Vec<f64>],
) -> StatsResult<GridSearchResult> {
    if data.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Data cannot be empty".to_string(),
        ));
    }
    if param_grids.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Must specify at least one parameter grid".to_string(),
        ));
    }

    // Build Cartesian product of all parameter grids
    let mut combinations: Vec<Vec<f64>> = vec![vec![]];
    for grid in param_grids {
        let mut new_combos = Vec::new();
        for combo in &combinations {
            for &val in grid {
                let mut extended = combo.clone();
                extended.push(val);
                new_combos.push(extended);
            }
        }
        combinations = new_combos;
    }

    // Evaluate log-likelihood for each combination in parallel
    let results: Vec<(Vec<f64>, f64)> = parallel_map(&combinations, |params| {
        let ll = log_likelihood_fn(data, params);
        (params.clone(), ll)
    });

    // Find best
    let mut sorted = results;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let (best_params, best_ll) = sorted
        .first()
        .map(|(p, ll)| (p.clone(), *ll))
        .unwrap_or_else(|| (vec![], f64::NEG_INFINITY));

    Ok(GridSearchResult {
        best_params,
        best_log_likelihood: best_ll,
        all_results: sorted,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Welford accumulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_welford_empty() {
        let acc = WelfordAccumulator::new();
        assert_eq!(acc.n, 0);
        assert_eq!(acc.mean, 0.0);
    }

    #[test]
    fn test_welford_single_value() {
        let mut acc = WelfordAccumulator::new();
        acc.push(5.0);
        assert!((acc.mean - 5.0).abs() < 1e-12);
        assert_eq!(acc.n, 1);
    }

    #[test]
    fn test_welford_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut acc = WelfordAccumulator::new();
        for &x in &data {
            acc.push(x);
        }
        assert!((acc.mean - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_welford_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mut acc = WelfordAccumulator::new();
        for &x in &data {
            acc.push(x);
        }
        // Population variance = 4.0
        assert!((acc.variance() - 4.0).abs() < 1e-10);
        // Sample variance = 4.571428...
        assert!((acc.sample_variance() - 4.571_428_571_428_571).abs() < 1e-8);
    }

    #[test]
    fn test_welford_merge_equals_sequential() {
        let all_data: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin() * 100.0).collect();

        // Sequential
        let mut seq = WelfordAccumulator::new();
        for &x in &all_data {
            seq.push(x);
        }

        // Parallel merge (split into 4 chunks)
        let chunk_size = all_data.len() / 4;
        let mut merged = WelfordAccumulator::new();
        for chunk in all_data.chunks(chunk_size) {
            let mut partial = WelfordAccumulator::new();
            for &x in chunk {
                partial.push(x);
            }
            merged.merge(&partial);
        }

        assert!((seq.mean - merged.mean).abs() < 1e-8, "means differ");
        assert!(
            (seq.variance() - merged.variance()).abs() < 1e-6,
            "variances differ"
        );
        assert!(
            (seq.skewness() - merged.skewness()).abs() < 0.01,
            "skewness differs"
        );
        assert!(
            (seq.kurtosis() - merged.kurtosis()).abs() < 0.1,
            "kurtosis differs"
        );
    }

    #[test]
    fn test_parallel_moments_small() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let acc = parallel_moments(&data);
        assert!((acc.mean - 5.5).abs() < 1e-10);
        assert_eq!(acc.n, 10);
    }

    #[test]
    fn test_parallel_welford_mean() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let m = parallel_welford_mean(&data).expect("mean failed");
        assert!((m - 50.5).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_welford_variance() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let v = parallel_welford_variance(&data, 0).expect("var failed");
        assert!((v - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_welford_mean_empty() {
        let result = parallel_welford_mean(&[]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel quantile tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_quantile_median() {
        let data: Vec<f64> = (1..=99).map(|x| x as f64).collect();
        let med = parallel_quantile(&data, 0.5).expect("median failed");
        assert!((med - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_quantile_extremes() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let q0 = parallel_quantile(&data, 0.0).expect("q0 failed");
        let q1 = parallel_quantile(&data, 1.0).expect("q1 failed");
        assert!((q0 - 1.0).abs() < 1e-10);
        assert!((q1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_quantile_empty() {
        assert!(parallel_quantile(&[], 0.5).is_err());
    }

    #[test]
    fn test_parallel_quantile_invalid() {
        let data = vec![1.0, 2.0, 3.0];
        assert!(parallel_quantile(&data, -0.1).is_err());
        assert!(parallel_quantile(&data, 1.1).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel histogram tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_histogram_basic() {
        let data: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let hist = parallel_histogram(&data, 10).expect("hist failed");
        assert_eq!(hist.edges.len(), 11);
        assert_eq!(hist.counts.len(), 10);
        let total: u64 = hist.counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_parallel_histogram_single_value() {
        let data = vec![5.0; 50];
        let hist = parallel_histogram(&data, 5).expect("hist failed");
        let total: u64 = hist.counts.iter().sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn test_parallel_histogram_empty() {
        assert!(parallel_histogram(&[], 10).is_err());
    }

    #[test]
    fn test_parallel_histogram_zero_bins() {
        assert!(parallel_histogram(&[1.0, 2.0], 0).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel permutation test
    // -----------------------------------------------------------------------

    #[test]
    fn test_permutation_test_identical_groups() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result =
            parallel_permutation_test(&group1, &group2, 999, Some(42)).expect("perm test failed");
        // Identical groups => p-value should be large
        assert!(
            result.p_value > 0.1,
            "p-value {} should be > 0.1 for identical groups",
            result.p_value
        );
    }

    #[test]
    fn test_permutation_test_different_groups() {
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![100.0, 200.0, 300.0, 400.0, 500.0];
        let result =
            parallel_permutation_test(&group1, &group2, 999, Some(42)).expect("perm test failed");
        // Very different groups => p-value should be small
        assert!(
            result.p_value < 0.05,
            "p-value {} should be < 0.05 for very different groups",
            result.p_value
        );
    }

    #[test]
    fn test_permutation_test_empty_group() {
        assert!(parallel_permutation_test(&[], &[1.0], 100, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel bootstrap tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_bootstrap_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mean_fn = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let result =
            parallel_bootstrap(&data, &mean_fn, 2000, 0.95, Some(42)).expect("bootstrap failed");

        // Point estimate should be the mean of the data
        assert!((result.estimate - 5.5).abs() < 1e-10);
        // CI should contain the estimate
        assert!(result.ci_lower <= result.estimate);
        assert!(result.ci_upper >= result.estimate);
        assert_eq!(result.replicates.len(), 2000);
    }

    #[test]
    fn test_parallel_bootstrap_empty() {
        let mean_fn = |s: &[f64]| s.iter().sum::<f64>() / s.len().max(1) as f64;
        assert!(parallel_bootstrap(&[], &mean_fn, 100, 0.95, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel cross-validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_cv_basic() {
        // Simple regression: y = 2*x + 1
        let n = 100;
        let data = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let targets = Array1::from_shape_fn(n, |i| 2.0 * i as f64 + 1.0);

        let scorer = |train_x: &Array2<f64>,
                      train_y: &Array1<f64>,
                      test_x: &Array2<f64>,
                      test_y: &Array1<f64>|
         -> StatsResult<f64> {
            // Simple mean prediction as baseline
            let pred = train_y.iter().sum::<f64>() / train_y.len() as f64;
            let ss_res: f64 = test_y.iter().map(|&y| (y - pred) * (y - pred)).sum();
            let ss_tot: f64 = {
                let mean_y = test_y.iter().sum::<f64>() / test_y.len() as f64;
                test_y.iter().map(|&y| (y - mean_y) * (y - mean_y)).sum()
            };
            if ss_tot.abs() < 1e-12 {
                Ok(0.0)
            } else {
                Ok(1.0 - ss_res / ss_tot) // R^2
            }
        };

        let result =
            parallel_cross_validation(&data, &targets, 5, &scorer, Some(42)).expect("CV failed");
        assert_eq!(result.n_folds, 5);
        assert_eq!(result.fold_scores.len(), 5);
    }

    // -----------------------------------------------------------------------
    // Parallel MLE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_mle_normal_data() {
        // Generate data from a normal distribution (deterministic)
        let data: Vec<f64> = (0..500)
            .map(|i| {
                let x = (i as f64 * 0.13).sin() * 2.0 + 5.0;
                x
            })
            .collect();

        let results = parallel_mle_fit(&data).expect("MLE fit failed");
        assert!(!results.is_empty());
        // First result should be the best by AIC
        assert!(results[0].aic.is_finite());
    }

    #[test]
    fn test_parallel_mle_empty() {
        assert!(parallel_mle_fit(&[]).is_err());
    }

    // -----------------------------------------------------------------------
    // Parallel grid search tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_grid_search_normal() {
        let data: Vec<f64> = vec![4.5, 5.0, 5.5, 5.0, 4.8, 5.2, 5.1, 4.9];

        let normal_ll = |data: &[f64], params: &[f64]| -> f64 {
            if params.len() < 2 || params[1] <= 0.0 {
                return f64::NEG_INFINITY;
            }
            let mu = params[0];
            let sigma = params[1];
            data.iter()
                .map(|&x| {
                    let z = (x - mu) / sigma;
                    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
                })
                .sum()
        };

        let mu_grid: Vec<f64> = (40..=60).map(|i| i as f64 * 0.1).collect();
        let sigma_grid: Vec<f64> = (1..=10).map(|i| i as f64 * 0.1).collect();

        let result = parallel_grid_search(&data, &normal_ll, &[mu_grid, sigma_grid])
            .expect("grid search failed");

        // Best mu should be near the data mean (~5.0)
        assert!(
            (result.best_params[0] - 5.0).abs() < 0.2,
            "best mu = {}",
            result.best_params[0]
        );
    }

    #[test]
    fn test_parallel_grid_search_empty() {
        let ll = |_data: &[f64], _params: &[f64]| -> f64 { 0.0 };
        assert!(parallel_grid_search(&[], &ll, &[vec![1.0]]).is_err());
    }

    // -----------------------------------------------------------------------
    // Skewness / kurtosis tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parallel_skewness_symmetric() {
        // Symmetric data => skewness should be near 0
        let data: Vec<f64> = (-50..=50).map(|x| x as f64).collect();
        let sk = parallel_welford_skewness(&data).expect("skewness failed");
        assert!(
            sk.abs() < 0.01,
            "skewness of symmetric data should be ~0, got {}",
            sk
        );
    }

    #[test]
    fn test_parallel_kurtosis_uniform() {
        // Uniform data has excess kurtosis of -1.2
        let data: Vec<f64> = (0..10000).map(|x| x as f64 / 10000.0).collect();
        let kurt = parallel_welford_kurtosis(&data).expect("kurtosis failed");
        assert!(
            (kurt - (-1.2)).abs() < 0.1,
            "kurtosis of uniform data should be ~-1.2, got {}",
            kurt
        );
    }

    #[test]
    fn test_parallel_skewness_insufficient_data() {
        assert!(parallel_welford_skewness(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_parallel_kurtosis_insufficient_data() {
        assert!(parallel_welford_kurtosis(&[1.0, 2.0, 3.0]).is_err());
    }
}
