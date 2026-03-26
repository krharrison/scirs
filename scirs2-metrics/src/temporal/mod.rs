//! Temporal Metrics: Dynamic Time Warping and Probabilistic Forecast Skill Scores
//!
//! This module provides metrics for evaluating time series models and probabilistic
//! forecasting systems:
//!
//! - **Dynamic Time Warping (DTW)**: Elastic similarity measure for time series
//!   - Full O(n·m) DP computation
//!   - Sakoe-Chiba band constraint for O(n·w) speedup
//!   - Normalized DTW (divide by optimal path length)
//! - **Brier Score**: MSE for probabilistic binary forecasts
//! - **Brier Skill Score (BSS)**: Relative improvement over climatology
//! - **CRPS**: Continuous Ranked Probability Score for distributional forecasts
//!   - Closed-form for Gaussian predictive distributions
//!   - Empirical approximation for general distributions
//! - **Directional Accuracy**: Fraction of forecasts with correct sign of change
//! - **Diebold-Mariano Test**: Statistical test for comparing two forecast series
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::temporal::{
//!     dtw, dtw_windowed, brier_score_temporal, crps_gaussian, directional_accuracy,
//! };
//!
//! // DTW of identical series = 0
//! let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
//! let d = dtw(&x, &x).expect("should succeed");
//! assert!(d.abs() < 1e-10);
//!
//! // CRPS for Gaussian predictive
//! let mu = vec![0.0, 1.0, 2.0];
//! let sigma = vec![1.0, 1.0, 1.0];
//! let obs = vec![0.5, 1.5, 1.8];
//! let crps = crps_gaussian(&mu, &sigma, &obs).expect("should succeed");
//! assert!(crps >= 0.0);
//! ```

use crate::error::{MetricsError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Dynamic Time Warping
// ────────────────────────────────────────────────────────────────────────────

/// Computes the Dynamic Time Warping (DTW) distance between two time series.
///
/// DTW finds the optimal alignment between two sequences by allowing elastic
/// stretching/compression in the time dimension. The standard O(n·m) DP
/// recurrence is:
///
/// ```text
/// DTW[i][j] = dist(x[i], y[j]) + min(DTW[i-1][j], DTW[i][j-1], DTW[i-1][j-1])
/// ```
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
///
/// # Returns
///
/// The DTW distance (non-negative).
pub fn dtw(x: &[f64], y: &[f64]) -> Result<f64> {
    dtw_full(x, y, None)
}

/// Computes DTW with a Sakoe-Chiba band constraint.
///
/// The Sakoe-Chiba band restricts warping to a window of ±`window` steps,
/// reducing complexity from O(n·m) to O(n·w).
///
/// When `window` is large enough to cover the full matrix, this is equivalent
/// to unconstrained DTW.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `window` - Half-width of Sakoe-Chiba band
///
/// # Returns
///
/// The constrained DTW distance.
pub fn dtw_windowed(x: &[f64], y: &[f64], window: usize) -> Result<f64> {
    dtw_full(x, y, Some(window))
}

/// Computes normalized DTW distance (divided by warping path length).
///
/// Normalization makes DTW comparable across time series of different lengths.
///
/// # Arguments
///
/// * `x` - First time series
/// * `y` - Second time series
/// * `window` - Optional Sakoe-Chiba window
pub fn dtw_normalized(x: &[f64], y: &[f64], window: Option<usize>) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }

    let (dist, path_len) = dtw_with_path_length(x, y, window)?;

    if path_len == 0 {
        return Ok(0.0);
    }

    Ok(dist / path_len as f64)
}

/// Full DTW computation returning both distance and path length.
fn dtw_with_path_length(x: &[f64], y: &[f64], window: Option<usize>) -> Result<(f64, usize)> {
    let n = x.len();
    let m = y.len();

    if n == 0 || m == 0 {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }

    // Effective window size: if None, use full matrix
    let w = window.unwrap_or(n.max(m));
    let w = w.max((n as isize - m as isize).unsigned_abs()); // must cover length difference

    let inf = f64::INFINITY;
    // Use flattened 2D array: dp[i][j] = DTW(x[0..=i], y[0..=j])
    let mut dp = vec![inf; n * m];

    let idx = |i: usize, j: usize| i * m + j;

    for i in 0..n {
        for j in 0..m {
            // Sakoe-Chiba constraint
            if (i as isize - j as isize).unsigned_abs() > w {
                continue;
            }

            let cost = (x[i] - y[j]).abs();

            let prev = if i == 0 && j == 0 {
                0.0
            } else if i == 0 {
                dp[idx(0, j - 1)]
            } else if j == 0 {
                dp[idx(i - 1, 0)]
            } else {
                let d_ij = dp[idx(i - 1, j - 1)];
                let d_i1j = dp[idx(i - 1, j)];
                let d_ij1 = dp[idx(i, j - 1)];
                d_ij.min(d_i1j).min(d_ij1)
            };

            if prev.is_infinite() && !(i == 0 && j == 0) {
                dp[idx(i, j)] = inf;
            } else {
                dp[idx(i, j)] = cost + if i == 0 && j == 0 { 0.0 } else { prev };
            }
        }
    }

    let total = dp[idx(n - 1, m - 1)];
    if total.is_infinite() {
        return Err(MetricsError::CalculationError(
            "DTW path not found within Sakoe-Chiba window; increase window size".to_string(),
        ));
    }

    // Traceback to find path length
    let mut path_len = 0usize;
    let mut i = n - 1;
    let mut j = m - 1;

    loop {
        path_len += 1;
        if i == 0 && j == 0 {
            break;
        }
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let d_ij = dp[idx(i - 1, j - 1)];
            let d_i1j = dp[idx(i - 1, j)];
            let d_ij1 = dp[idx(i, j - 1)];
            let min_prev = d_ij.min(d_i1j).min(d_ij1);
            if min_prev == d_ij {
                i -= 1;
                j -= 1;
            } else if min_prev == d_i1j {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }

    Ok((total, path_len))
}

/// Internal full DTW (handles window=None as unconstrained).
fn dtw_full(x: &[f64], y: &[f64], window: Option<usize>) -> Result<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MetricsError::InvalidInput(
            "time series must not be empty".to_string(),
        ));
    }
    let (dist, _) = dtw_with_path_length(x, y, window)?;
    Ok(dist)
}

// ────────────────────────────────────────────────────────────────────────────
// Brier Score and Brier Skill Score
// ────────────────────────────────────────────────────────────────────────────

/// Computes the Brier Score for probabilistic binary forecasts.
///
/// BS = (1/n) * sum_i (p_i - o_i)²
///
/// where p_i is the forecast probability and o_i ∈ {0, 1} is the observation.
///
/// # Arguments
///
/// * `forecasts` - Predicted probabilities in [0, 1]
/// * `observations` - Observed binary outcomes (0.0 or 1.0)
///
/// # Returns
///
/// Brier score in [0, 1]. 0 = perfect, 0.25 = no skill (climatology at 0.5).
pub fn brier_score_temporal(forecasts: &[f64], observations: &[f64]) -> Result<f64> {
    if forecasts.len() != observations.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "forecasts ({}) and observations ({}) must have the same length",
            forecasts.len(),
            observations.len()
        )));
    }
    if forecasts.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let n = forecasts.len();
    let bs: f64 = (0..n)
        .map(|i| (forecasts[i] - observations[i]).powi(2))
        .sum::<f64>()
        / n as f64;
    Ok(bs)
}

/// Computes the Brier Skill Score (BSS) relative to a climatological reference.
///
/// BSS = 1 - BS(model) / BS(reference)
///
/// The reference is the climatological frequency (base rate) for all forecasts.
/// BSS = 1: perfect forecast. BSS = 0: no skill over climatology. BSS < 0: worse.
///
/// # Arguments
///
/// * `forecasts` - Predicted probabilities
/// * `observations` - Observed binary outcomes
pub fn brier_skill_score_temporal(forecasts: &[f64], observations: &[f64]) -> Result<f64> {
    let bs_model = brier_score_temporal(forecasts, observations)?;
    let base_rate = observations.iter().sum::<f64>() / observations.len() as f64;
    let ref_fc: Vec<f64> = vec![base_rate; observations.len()];
    let bs_ref = brier_score_temporal(&ref_fc, observations)?;

    if bs_ref <= f64::EPSILON {
        // Degenerate case: all observations same
        if bs_model <= f64::EPSILON {
            return Ok(1.0);
        }
        return Ok(f64::NEG_INFINITY);
    }

    Ok(1.0 - bs_model / bs_ref)
}

// ────────────────────────────────────────────────────────────────────────────
// CRPS (Continuous Ranked Probability Score)
// ────────────────────────────────────────────────────────────────────────────

/// Computes the mean CRPS for Gaussian predictive distributions.
///
/// For a Gaussian forecast N(μ, σ²), the CRPS has a closed-form solution:
/// ```text
/// CRPS(N(μ,σ), y) = σ * [ (y-μ)/σ * (2Φ((y-μ)/σ) - 1)
///                         + 2φ((y-μ)/σ) - 1/√π ]
/// ```
/// where Φ is the standard normal CDF and φ is the PDF.
///
/// # Arguments
///
/// * `mu` - Predictive means, one per observation
/// * `sigma` - Predictive standard deviations (must be > 0)
/// * `observations` - Observed values
///
/// # Returns
///
/// Mean CRPS (lower is better; 0 is optimal).
pub fn crps_gaussian(mu: &[f64], sigma: &[f64], observations: &[f64]) -> Result<f64> {
    let n = mu.len();
    if n != sigma.len() || n != observations.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "mu ({}), sigma ({}), observations ({}) must have the same length",
            n,
            sigma.len(),
            observations.len()
        )));
    }
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut total = 0.0f64;
    for i in 0..n {
        if sigma[i] <= 0.0 {
            return Err(MetricsError::InvalidInput(format!(
                "sigma[{i}] must be positive, got {}",
                sigma[i]
            )));
        }
        let z = (observations[i] - mu[i]) / sigma[i];
        // CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
        let phi_z = standard_normal_pdf(z);
        let big_phi_z = standard_normal_cdf(z);
        let crps_i = sigma[i]
            * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - 1.0 / std::f64::consts::PI.sqrt());
        total += crps_i;
    }

    Ok(total / n as f64)
}

/// Computes CRPS for an empirical predictive distribution (ensemble).
///
/// Given an ensemble of forecasts for each observation, the CRPS is:
/// ```text
/// CRPS = E[|X - y|] - (1/2) * E[|X - X'|]
/// ```
/// where X, X' are independent draws from the ensemble.
///
/// # Arguments
///
/// * `ensemble` - Shape (n_observations, n_members) ensemble forecasts
/// * `observations` - Observed values, length n_observations
pub fn crps_ensemble(ensemble: &[Vec<f64>], observations: &[f64]) -> Result<f64> {
    let n = ensemble.len();
    if n != observations.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "ensemble ({}) and observations ({}) must have the same length",
            n,
            observations.len()
        )));
    }
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let mut total = 0.0f64;
    for i in 0..n {
        let members = &ensemble[i];
        let m = members.len();
        if m == 0 {
            return Err(MetricsError::InvalidInput(format!(
                "ensemble[{i}] must not be empty"
            )));
        }

        // E[|X - y|]
        let e_xy: f64 = members
            .iter()
            .map(|&x| (x - observations[i]).abs())
            .sum::<f64>()
            / m as f64;

        // E[|X - X'|] using sorted trick
        let mut sorted = members.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut prefix = 0.0f64;
        let mut e_xx = 0.0f64;
        for (k, &xk) in sorted.iter().enumerate() {
            e_xx += xk * k as f64 - prefix;
            prefix += xk;
        }
        let pairs = m as f64 * (m as f64 - 1.0) / 2.0;
        let e_xx_mean = if pairs > 0.0 { e_xx / pairs } else { 0.0 };

        total += e_xy - 0.5 * e_xx_mean;
    }

    Ok(total / n as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Directional Accuracy
// ────────────────────────────────────────────────────────────────────────────

/// Computes the directional accuracy of forecasts.
///
/// Directional accuracy measures the fraction of forecasts that correctly
/// predict the sign of change from the previous period:
/// ```text
/// DA = (1/(n-1)) * sum_{i=1}^{n-1} 1[sign(pred[i] - pred[i-1]) == sign(obs[i] - obs[i-1])]
/// ```
///
/// # Arguments
///
/// * `forecasts` - Forecast values (continuous)
/// * `observations` - Observed values (same length as forecasts)
///
/// # Returns
///
/// Directional accuracy in [0, 1]. 1.0 = all directions correct.
pub fn directional_accuracy(forecasts: &[f64], observations: &[f64]) -> Result<f64> {
    let n = forecasts.len();
    if n != observations.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "forecasts ({}) and observations ({}) must have the same length",
            n,
            observations.len()
        )));
    }
    if n < 2 {
        return Err(MetricsError::InvalidInput(
            "at least 2 data points required for directional accuracy".to_string(),
        ));
    }

    let correct = (1..n)
        .filter(|&i| {
            let obs_dir = (observations[i] - observations[i - 1]).signum();
            let fc_dir = (forecasts[i] - forecasts[i - 1]).signum();
            obs_dir == fc_dir
        })
        .count();

    Ok(correct as f64 / (n - 1) as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Diebold-Mariano Test
// ────────────────────────────────────────────────────────────────────────────

/// Result of the Diebold-Mariano test.
#[derive(Debug, Clone)]
pub struct DieboldMarianoResult {
    /// DM test statistic
    pub statistic: f64,
    /// Two-sided p-value (approximate, from t-distribution with n-1 df)
    pub p_value: f64,
    /// Loss differential series d_t = L(e1_t) - L(e2_t)
    pub loss_differentials: Vec<f64>,
    /// Mean of loss differentials
    pub mean_differential: f64,
}

/// Loss function type for the Diebold-Mariano test.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DmLossFunction {
    /// Squared error loss: L(e) = e²
    SquaredError,
    /// Absolute error loss: L(e) = |e|
    AbsoluteError,
    /// Asymmetric loss with parameter a: L(e) = e² + a*e
    Asymmetric(f64),
}

/// Performs the Diebold-Mariano test to compare predictive accuracy of two forecasters.
///
/// The DM statistic tests H₀: E\[d_t\] = 0 vs H₁: E\[d_t\] ≠ 0, where
/// d_t = L(e1_t) - L(e2_t) is the loss differential.
///
/// Test statistic: DM = d̄ / sqrt(V̂(d̄))
/// where V̂(d̄) is the long-run variance of d̄ estimated via HAC (Newey-West).
///
/// Under H₀, DM ~ t(n-1) approximately.
///
/// # Arguments
///
/// * `actual` - Actual observed values
/// * `forecast1` - Forecasts from model 1
/// * `forecast2` - Forecasts from model 2
/// * `loss_fn` - Loss function to use
/// * `h` - Forecast horizon (for Newey-West bandwidth selection: `h - 1`)
///
/// # Returns
///
/// `DieboldMarianoResult` with test statistic and p-value.
pub fn diebold_mariano_test(
    actual: &[f64],
    forecast1: &[f64],
    forecast2: &[f64],
    loss_fn: DmLossFunction,
    h: usize,
) -> Result<DieboldMarianoResult> {
    let n = actual.len();
    if n != forecast1.len() || n != forecast2.len() {
        return Err(MetricsError::DimensionMismatch(
            "actual, forecast1, forecast2 must have the same length".to_string(),
        ));
    }
    if n < 3 {
        return Err(MetricsError::InvalidInput(
            "at least 3 observations required for DM test".to_string(),
        ));
    }

    let loss = |e: f64| -> f64 {
        match loss_fn {
            DmLossFunction::SquaredError => e * e,
            DmLossFunction::AbsoluteError => e.abs(),
            DmLossFunction::Asymmetric(a) => e * e + a * e,
        }
    };

    // Compute loss differentials d_t = L(e1_t) - L(e2_t)
    let d: Vec<f64> = (0..n)
        .map(|i| {
            let e1 = actual[i] - forecast1[i];
            let e2 = actual[i] - forecast2[i];
            loss(e1) - loss(e2)
        })
        .collect();

    let d_mean = d.iter().sum::<f64>() / n as f64;

    // HAC variance estimate (Newey-West with bandwidth M = h - 1)
    let m = h.saturating_sub(1);

    let gamma_0: f64 = d.iter().map(|&di| (di - d_mean).powi(2)).sum::<f64>() / n as f64;

    let mut variance = gamma_0;
    for lag in 1..=m {
        if lag >= n {
            break;
        }
        let gamma_l: f64 = (lag..n)
            .map(|t| (d[t] - d_mean) * (d[t - lag] - d_mean))
            .sum::<f64>()
            / n as f64;
        let weight = 1.0 - lag as f64 / (m + 1) as f64; // Bartlett kernel
        variance += 2.0 * weight * gamma_l;
    }

    // Ensure positive variance
    if variance <= 0.0 {
        variance = gamma_0.max(f64::EPSILON);
    }

    let se = (variance / n as f64).sqrt();
    if se <= f64::EPSILON {
        return Err(MetricsError::CalculationError(
            "variance of loss differentials is effectively zero".to_string(),
        ));
    }

    let dm_stat = d_mean / se;

    // Two-sided p-value from t-distribution with (n-1) degrees of freedom
    let p_value = two_sided_t_pvalue(dm_stat, n - 1);

    Ok(DieboldMarianoResult {
        statistic: dm_stat,
        p_value,
        loss_differentials: d,
        mean_differential: d_mean,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Forecast Skill Summary Struct
// ────────────────────────────────────────────────────────────────────────────

/// Summary of probabilistic forecast skill metrics.
#[derive(Debug, Clone)]
pub struct ForecastSkillMetrics {
    /// Brier Score (binary probabilistic forecasts)
    pub brier_score: f64,
    /// Brier Skill Score
    pub bss: f64,
    /// Continuous Ranked Probability Score (Gaussian)
    pub crps: f64,
    /// Directional accuracy (sign of change)
    pub directional_accuracy: f64,
}

impl ForecastSkillMetrics {
    /// Compute all forecast skill metrics for Gaussian predictive distributions.
    ///
    /// # Arguments
    ///
    /// * `prob_forecasts` - Probabilistic binary forecasts (for Brier score)
    /// * `binary_obs` - Binary observations for Brier score
    /// * `mu` - Predictive means (for CRPS)
    /// * `sigma` - Predictive standard deviations (for CRPS)
    /// * `point_forecasts` - Point forecasts (for directional accuracy)
    /// * `observations` - Continuous observations
    pub fn compute(
        prob_forecasts: &[f64],
        binary_obs: &[f64],
        mu: &[f64],
        sigma: &[f64],
        point_forecasts: &[f64],
        observations: &[f64],
    ) -> Result<Self> {
        let brier_score = brier_score_temporal(prob_forecasts, binary_obs)?;
        let bss = brier_skill_score_temporal(prob_forecasts, binary_obs)?;
        let crps = crps_gaussian(mu, sigma, observations)?;
        let da = directional_accuracy(point_forecasts, observations)?;

        Ok(Self {
            brier_score,
            bss,
            crps,
            directional_accuracy: da,
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Statistical utilities
// ────────────────────────────────────────────────────────────────────────────

/// Standard normal PDF φ(z)
fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF Φ(z) using Abramowitz & Stegun 26.2.17 approximation.
fn standard_normal_cdf(z: f64) -> f64 {
    // Using rational approximation (max error 7.5e-8)
    if z >= 0.0 {
        1.0 - standard_normal_cdf_positive(z)
    } else {
        standard_normal_cdf_positive(-z)
    }
}

fn standard_normal_cdf_positive(z: f64) -> f64 {
    // Complementary CDF for z >= 0
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    standard_normal_pdf(z) * poly
}

/// Approximate two-sided p-value for t-distribution.
/// Uses normal approximation for df > 30, otherwise Bailey (1994) approximation.
fn two_sided_t_pvalue(t: f64, df: usize) -> f64 {
    let t_abs = t.abs();
    if df == 0 {
        return 1.0;
    }

    let p_one_sided = if df as f64 > 30.0 {
        // Normal approximation
        standard_normal_cdf(-t_abs)
    } else {
        // Incomplete beta function approximation for t-distribution
        let x = df as f64 / (df as f64 + t_abs * t_abs);
        0.5 * regularized_incomplete_beta(df as f64 / 2.0, 0.5, x)
    };

    (2.0 * p_one_sided).min(1.0).max(0.0)
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    // Lentz's continued fraction
    let lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;

    // Continued fraction via Lentz's method
    let max_iter = 200;
    let tol = 1e-10;
    let mut c = 1.0f64;
    let raw_d = 1.0 - (a + b) * x / (a + 1.0);
    let mut d = if raw_d.abs() < f64::MIN_POSITIVE {
        f64::MIN_POSITIVE
    } else {
        1.0 / raw_d
    };
    let mut f = d;

    for m in 1..=max_iter {
        let m = m as f64;
        // Even step
        let num_even = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num_even * d;
        d = if d.abs() < f64::MIN_POSITIVE {
            f64::MIN_POSITIVE
        } else {
            d
        };
        c = 1.0 + num_even / c;
        c = if c.abs() < f64::MIN_POSITIVE {
            f64::MIN_POSITIVE
        } else {
            c
        };
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let num_odd = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num_odd * d;
        d = if d.abs() < f64::MIN_POSITIVE {
            f64::MIN_POSITIVE
        } else {
            d
        };
        c = 1.0 + num_odd / c;
        c = if c.abs() < f64::MIN_POSITIVE {
            f64::MIN_POSITIVE
        } else {
            c
        };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < tol {
            break;
        }
    }

    front * f
}

/// Log-gamma function via Lanczos approximation.
fn lgamma(x: f64) -> f64 {
    // Lanczos approximation
    let g = 7.0;
    let c: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        std::f64::consts::PI.ln() - ((std::f64::consts::PI * x).sin().abs()).ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut s = c[0];
        for (i, &ci) in c[1..].iter().enumerate() {
            s += ci / (x + (i + 1) as f64);
        }
        let t = x + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── DTW tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_dtw_identical() {
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let d = dtw(&x, &x).expect("should succeed");
        assert!(d.abs() < 1e-10, "DTW(x,x) should be 0, got {d}");
    }

    #[test]
    fn test_dtw_shift() {
        // DTW should handle time-shifted signals well
        let x = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.0, 2.0, 1.0]; // shifted by 1
        let d = dtw(&x, &y).expect("should succeed");
        // DTW is allowed to warp, so distance should be small
        assert!(d >= 0.0, "DTW must be non-negative");
        assert!(d < 2.0, "DTW should handle shifted signals: got {d}");
    }

    #[test]
    fn test_dtw_windowed_matches_full_for_large_window() {
        let x = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let y = vec![1.5, 2.5, 3.5, 2.5, 1.5];
        let d_full = dtw(&x, &y).expect("full DTW");
        let d_win = dtw_windowed(&x, &y, 5).expect("windowed DTW");
        assert!(
            (d_full - d_win).abs() < 1e-10,
            "large window should match full DTW"
        );
    }

    #[test]
    fn test_dtw_windowed_constraint() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // same — DTW should be 0
        let d = dtw_windowed(&x, &y, 2).expect("windowed DTW");
        assert!(d.abs() < 1e-10, "DTW(x,x) with window should be 0");
    }

    #[test]
    fn test_dtw_normalized() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let d = dtw_normalized(&x, &y, None).expect("normalized DTW");
        assert!(d.abs() < 1e-10, "normalized DTW(x,x) should be 0");
    }

    #[test]
    fn test_dtw_different_lengths() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 1.5, 2.0, 2.5, 3.0];
        let d = dtw(&x, &y).expect("DTW with different lengths");
        assert!(d >= 0.0);
    }

    // ── Brier Score tests ──────────────────────────────────────────────────

    #[test]
    fn test_brier_score_perfect() {
        let obs = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let fc = vec![1.0, 0.0, 1.0, 0.0, 1.0]; // perfect
        let bs = brier_score_temporal(&fc, &obs).expect("should succeed");
        assert!(
            bs.abs() < 1e-10,
            "perfect Brier Score should be 0, got {bs}"
        );
    }

    #[test]
    fn test_brier_score_worst() {
        let obs = vec![1.0, 0.0, 1.0, 0.0];
        let fc = vec![0.0, 1.0, 0.0, 1.0]; // completely wrong
        let bs = brier_score_temporal(&fc, &obs).expect("should succeed");
        assert!(
            (bs - 1.0).abs() < 1e-10,
            "worst Brier Score should be 1.0, got {bs}"
        );
    }

    #[test]
    fn test_brier_skill_score_no_skill() {
        // Climatological forecast should have BSS = 0
        let obs = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let base_rate = 0.5;
        let fc = vec![base_rate; obs.len()];
        let bss = brier_skill_score_temporal(&fc, &obs).expect("should succeed");
        assert!(
            bss.abs() < 1e-10,
            "climatological BSS should be 0, got {bss}"
        );
    }

    // ── CRPS tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_crps_gaussian_perfect() {
        // When obs exactly at mu and sigma→0, CRPS → 0
        let mu = vec![1.0, 2.0, 3.0];
        let sigma = vec![0.001, 0.001, 0.001]; // very sharp
        let obs = vec![1.0, 2.0, 3.0]; // exact
        let crps = crps_gaussian(&mu, &sigma, &obs).expect("should succeed");
        assert!(crps >= 0.0);
        assert!(
            crps < 0.01,
            "CRPS for near-perfect Gaussian should be small, got {crps}"
        );
    }

    #[test]
    fn test_crps_gaussian_nonnegative() {
        let mu = vec![0.0, 1.0, 2.0, -1.0];
        let sigma = vec![1.0, 2.0, 0.5, 1.5];
        let obs = vec![0.5, 1.5, 1.8, -0.5];
        let crps = crps_gaussian(&mu, &sigma, &obs).expect("should succeed");
        assert!(crps >= 0.0, "CRPS must be non-negative, got {crps}");
    }

    #[test]
    fn test_crps_gaussian_known_value() {
        // For N(0, 1) and observation y=0:
        // CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
        //      = 1 * [ 0 + 2*(1/sqrt(2*pi)) - 1/sqrt(pi) ]
        //      = 2/sqrt(2*pi) - 1/sqrt(pi)
        //      = (sqrt(2) - 1) / sqrt(pi)
        //      ≈ 0.2338
        let mu = vec![0.0];
        let sigma = vec![1.0];
        let obs = vec![0.0];
        let crps = crps_gaussian(&mu, &sigma, &obs).expect("should succeed");
        let expected = (2.0_f64.sqrt() - 1.0) / std::f64::consts::PI.sqrt();
        assert!(
            (crps - expected).abs() < 1e-4,
            "CRPS(N(0,1), y=0) ≈ {expected:.4}, got {crps:.4}"
        );
    }

    #[test]
    fn test_crps_ensemble_identical() {
        // Ensemble forecast exactly at observation
        let ensemble = vec![vec![2.0, 2.0, 2.0], vec![5.0, 5.0, 5.0]];
        let obs = vec![2.0, 5.0];
        let crps = crps_ensemble(&ensemble, &obs).expect("should succeed");
        assert!(crps >= 0.0);
        assert!(
            crps < 1e-6,
            "perfect ensemble CRPS should be ~0, got {crps}"
        );
    }

    // ── Directional Accuracy tests ─────────────────────────────────────────

    #[test]
    fn test_directional_accuracy_all_correct() {
        let obs = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let fc = vec![1.1, 2.1, 3.1, 1.9, 0.9]; // same directions
        let da = directional_accuracy(&fc, &obs).expect("should succeed");
        assert!(
            (da - 1.0).abs() < 1e-10,
            "all correct DA should be 1.0, got {da}"
        );
    }

    #[test]
    fn test_directional_accuracy_all_wrong() {
        let obs = vec![1.0, 2.0, 3.0, 4.0];
        let fc = vec![4.0, 3.0, 2.0, 1.0]; // opposite directions
        let da = directional_accuracy(&fc, &obs).expect("should succeed");
        assert!(
            (da - 0.0).abs() < 1e-10,
            "all wrong DA should be 0.0, got {da}"
        );
    }

    #[test]
    fn test_directional_accuracy_half() {
        // Design: n=5, 4 changes, need exactly 2 correct
        // obs: 1->2 (+), 2->1 (-), 1->2 (+), 2->1 (-)
        // fc:  1->2 (+), 2->2 (0), 1->2 (+), 2->1 (-)  →  +,0,+,- vs +,-,+,- → 3 correct
        // Better: forecast exactly opposite for first 2 changes, same for last 2
        // obs: 0->1(+), 1->0(-), 0->1(+), 1->0(-)
        // fc:  0->0(0), 0->1(+), 1->0(-), 0->1(+) →  0,+,-,+ vs +,-,+,- → 0 matches
        // Simplest: 2 correct, 2 wrong
        // obs changes: -, +, -, +  (obs=[4,2,6,1,5])
        // fc  changes: -, +, +, -  (fc =[4,2,6,8,3]) → match: +,+,-,- vs correct: -,+,-,+
        // i=1: obs=-,fc=- MATCH; i=2: obs=+,fc=+ MATCH; i=3: obs=-,fc=+ NO; i=4: obs=+,fc=- NO
        let obs = vec![4.0, 2.0, 6.0, 1.0, 5.0];
        let fc = vec![4.0, 2.0, 6.0, 8.0, 3.0];
        let da = directional_accuracy(&fc, &obs).expect("should succeed");
        assert!(
            (da - 0.5).abs() < 1e-10,
            "half-correct DA should be 0.5, got {da}"
        );
    }

    // ── Diebold-Mariano tests ──────────────────────────────────────────────

    #[test]
    fn test_diebold_mariano_identical_forecasts() {
        // If both forecasts are identical, d_t = 0 for all t
        // Mean differential should be exactly 0
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fc1 = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let fc2 = fc1.clone();
        let result = diebold_mariano_test(&actual, &fc1, &fc2, DmLossFunction::SquaredError, 1);
        // Loss differentials are all 0 → variance is 0 → error, or statistic is 0
        match result {
            Err(_) => {} // expected: zero variance triggers error
            Ok(r) => {
                // If it doesn't error, mean differential must be 0
                assert!(
                    r.mean_differential.abs() < 1e-10,
                    "mean differential for identical forecasts should be 0"
                );
            }
        }
    }

    #[test]
    fn test_diebold_mariano_clearly_different() {
        let actual: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // Model 1: very poor forecasts
        let fc1: Vec<f64> = actual.iter().map(|&x| x + 5.0).collect();
        // Model 2: much better
        let fc2: Vec<f64> = actual.iter().map(|&x| x + 0.1).collect();
        let result = diebold_mariano_test(&actual, &fc1, &fc2, DmLossFunction::SquaredError, 1)
            .expect("should succeed");
        // fc1 has larger loss, so mean differential should be positive
        assert!(
            result.mean_differential > 0.0,
            "model1 should have higher loss"
        );
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_forecast_skill_metrics_compute() {
        let n = 10;
        let prob_fc: Vec<f64> = (0..n).map(|i| if i < 5 { 0.9 } else { 0.1 }).collect();
        let bin_obs: Vec<f64> = (0..n).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();
        let mu: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sigma = vec![1.0; n];
        let point_fc: Vec<f64> = mu.iter().map(|&x| x + 0.1).collect();
        let obs: Vec<f64> = mu.clone();

        let metrics =
            ForecastSkillMetrics::compute(&prob_fc, &bin_obs, &mu, &sigma, &point_fc, &obs)
                .expect("should succeed");

        assert!(metrics.brier_score >= 0.0);
        assert!(metrics.crps >= 0.0);
        assert!(metrics.directional_accuracy >= 0.0 && metrics.directional_accuracy <= 1.0);
    }
}
