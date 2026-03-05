//! Statistical anomaly detection methods for time series
//!
//! Provides:
//! - `STLAnomalyDetector`: STL decomposition-based residual outlier detection
//! - `GESD`: Generalized Extreme Studentized Deviate test
//! - `SHEWHARTChart`: Shewhart control chart with UCL/LCL
//! - `EWMAChart`: Exponentially Weighted Moving Average control chart
//! - `CUSUMChart`: Cumulative Sum control chart (two-sided)
//! - `IsolationForestTS`: Isolation forest adapted for time series windows

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive, NumCast};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

// ============================================================================
// Common result types
// ============================================================================

/// Result of statistical anomaly detection
#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult {
    /// Anomaly scores for each point (higher = more anomalous)
    pub scores: Vec<f64>,
    /// Boolean flags for anomalous points
    pub is_anomaly: Vec<bool>,
    /// Threshold used for anomaly classification
    pub threshold: f64,
    /// Control limit information (if applicable)
    pub control_limits: Option<(f64, f64)>,
}

// ============================================================================
// STL Anomaly Detector
// ============================================================================

/// Configuration for STL-based anomaly detection
#[derive(Debug, Clone)]
pub struct STLAnomalyConfig {
    /// Seasonal period
    pub period: usize,
    /// Trend window for STL (must be odd)
    pub trend_window: usize,
    /// Seasonal window for STL (must be odd)
    pub seasonal_window: usize,
    /// Number of sigma for outlier threshold
    pub n_sigma: f64,
    /// Use robust STL fitting
    pub robust: bool,
}

impl Default for STLAnomalyConfig {
    fn default() -> Self {
        Self {
            period: 12,
            trend_window: 21,
            seasonal_window: 13,
            n_sigma: 3.0,
            robust: true,
        }
    }
}

/// STL Anomaly Detector
///
/// Decomposes the series via STL into trend + seasonal + residual.
/// Anomalies are points where the residual exceeds `n_sigma` standard deviations.
pub struct STLAnomalyDetector {
    config: STLAnomalyConfig,
}

impl STLAnomalyDetector {
    /// Create a new STL anomaly detector
    pub fn new(config: STLAnomalyConfig) -> Self {
        Self { config }
    }

    /// Perform simple moving average trend (LOESS approximation for speed)
    fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
        let n = data.len();
        let half = window / 2;
        let mut trend = vec![0.0; n];
        for i in 0..n {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(n);
            let slice = &data[start..end];
            trend[i] = slice.iter().sum::<f64>() / slice.len() as f64;
        }
        trend
    }

    /// Compute seasonal component using period averaging
    fn seasonal_component(residual_no_season: &[f64], period: usize) -> Vec<f64> {
        let n = residual_no_season.len();
        let mut seasonal = vec![0.0; n];

        // Compute average for each phase in the period
        let mut phase_sums = vec![0.0; period];
        let mut phase_counts = vec![0usize; period];

        for (i, &val) in residual_no_season.iter().enumerate() {
            let phase = i % period;
            phase_sums[phase] += val;
            phase_counts[phase] += 1;
        }

        // Center the seasonal component
        let phase_means: Vec<f64> = phase_sums
            .iter()
            .zip(phase_counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();
        let seasonal_mean: f64 = phase_means.iter().sum::<f64>() / period as f64;
        let centered_means: Vec<f64> = phase_means.iter().map(|&m| m - seasonal_mean).collect();

        for (i, s) in seasonal.iter_mut().enumerate() {
            *s = centered_means[i % period];
        }
        seasonal
    }

    /// Detect anomalies via STL decomposition + residual analysis
    pub fn detect(&self, data: &[f64]) -> Result<AnomalyDetectionResult> {
        let n = data.len();
        if n < self.config.period * 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 full seasonal periods for STL".to_string(),
                required: self.config.period * 2,
                actual: n,
            });
        }

        // Step 1: Compute trend via moving average
        let trend = Self::moving_average(data, self.config.trend_window);

        // Step 2: Detrend
        let detrended: Vec<f64> = data
            .iter()
            .zip(trend.iter())
            .map(|(&d, &t)| d - t)
            .collect();

        // Step 3: Compute seasonal component
        let seasonal = Self::seasonal_component(&detrended, self.config.period);

        // Step 4: Compute residuals
        let residuals: Vec<f64> = data
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .map(|((&d, &t), &s)| d - t - s)
            .collect();

        // Step 5: Robust scale estimate (MAD)
        let mut sorted_res: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
        sorted_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_abs_dev = sorted_res[n / 2];
        let scale = (median_abs_dev * 1.4826).max(1e-15); // Consistent estimator

        // Step 6: Flag anomalies
        let threshold = self.config.n_sigma * scale;
        let scores: Vec<f64> = residuals.iter().map(|&r| r.abs() / scale).collect();
        let is_anomaly: Vec<bool> = scores.iter().map(|&s| s > self.config.n_sigma).collect();

        Ok(AnomalyDetectionResult {
            scores,
            is_anomaly,
            threshold,
            control_limits: None,
        })
    }
}

// ============================================================================
// GESD Test
// ============================================================================

/// Result of the GESD test
#[derive(Debug, Clone)]
pub struct GESDResult {
    /// Detected outlier indices
    pub outlier_indices: Vec<usize>,
    /// Test statistics for each removed observation
    pub test_statistics: Vec<f64>,
    /// Critical values for each step
    pub critical_values: Vec<f64>,
    /// Number of outliers detected
    pub n_outliers: usize,
}

/// Generalized Extreme Studentized Deviate (GESD) test
///
/// Tests for up to `max_outliers` outliers in data assumed to come from
/// a normal distribution. Reference: Rosner (1983).
pub struct GESD {
    /// Maximum number of outliers to test for
    pub max_outliers: usize,
    /// Significance level (e.g., 0.05)
    pub alpha: f64,
}

impl GESD {
    /// Create a new GESD test
    pub fn new(max_outliers: usize, alpha: f64) -> Self {
        Self {
            max_outliers,
            alpha,
        }
    }

    /// Approximate critical value using Student's t distribution
    /// lambda_i = (n - i) * t / sqrt((n - i - 1 + t^2)(n - i + 1))
    fn critical_value(&self, n: usize, i: usize) -> f64 {
        let ni = (n - i) as f64; // remaining sample size
        let p = 1.0 - self.alpha / (2.0 * ni);
        // Approximate t quantile using normal approximation for large n
        let df = ni - 2.0; // degrees of freedom
        if df <= 0.0 {
            return f64::INFINITY;
        }
        let t = Self::approx_t_quantile(p, df);
        // Standard GESD critical value: lambda = (n_i - 1) * t / sqrt((n_i - 2 + t^2) * n_i)
        let denom = ((ni - 2.0 + t * t) * ni).sqrt().max(1e-15);
        (ni - 1.0) * t / denom
    }

    /// Approximate t quantile using Wilson-Hilferty transformation
    fn approx_t_quantile(p: f64, df: f64) -> f64 {
        // Normal quantile approximation (Abramowitz & Stegun)
        let z = Self::normal_quantile(p);
        let g1 = (z.powi(3) + z) / (4.0 * df);
        let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df * df);
        (z + g1 + g2).max(0.0)
    }

    /// Standard normal quantile (Beasley-Springer-Moro algorithm)
    fn normal_quantile(p: f64) -> f64 {
        let p = p.clamp(1e-10, 1.0 - 1e-10);
        // Rational approximation (Abramowitz & Stegun 26.2.17)
        let t = (-2.0 * (p.min(1.0 - p)).ln()).sqrt();
        let c = [2.515517, 0.802853, 0.010328];
        let d = [1.432788, 0.189269, 0.001308];
        let num = c[0] + c[1] * t + c[2] * t * t;
        let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
        let z = t - num / den;
        if p >= 0.5 {
            z
        } else {
            -z
        }
    }

    /// Run the GESD test
    pub fn test(&self, data: &[f64]) -> Result<GESDResult> {
        let n = data.len();
        if n < self.max_outliers + 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient data for GESD test".to_string(),
                required: self.max_outliers + 3,
                actual: n,
            });
        }

        let mut working: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
        let mut test_stats = Vec::new();
        let mut crit_vals = Vec::new();
        let mut removed_indices = Vec::new();

        for i in 0..self.max_outliers {
            let current_n = working.len();
            let mean: f64 = working.iter().map(|&(_, v)| v).sum::<f64>() / current_n as f64;
            let std: f64 = {
                let var = working
                    .iter()
                    .map(|&(_, v)| (v - mean).powi(2))
                    .sum::<f64>()
                    / current_n as f64;
                var.sqrt().max(1e-15)
            };

            // Find point with maximum |x - mean| / std
            let (max_idx, max_val) = working
                .iter()
                .enumerate()
                .map(|(j, &(orig_idx, v))| (j, orig_idx, (v - mean).abs() / std))
                .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, orig_idx, stat)| (j, (orig_idx, stat)))
                .unwrap_or((0, (0, 0.0)));

            test_stats.push(max_val.1);
            crit_vals.push(self.critical_value(n, i));
            removed_indices.push(max_val.0);
            working.remove(max_idx);
        }

        // Determine the number of outliers (largest i where stat > crit_val)
        let n_outliers = test_stats
            .iter()
            .zip(crit_vals.iter())
            .enumerate()
            .filter(|(_, (stat, crit))| stat > crit)
            .map(|(i, _)| i + 1)
            .last()
            .unwrap_or(0);

        let outlier_indices = removed_indices[..n_outliers].to_vec();

        Ok(GESDResult {
            outlier_indices,
            test_statistics: test_stats,
            critical_values: crit_vals,
            n_outliers,
        })
    }
}

// ============================================================================
// Shewhart Control Chart
// ============================================================================

/// Configuration for the Shewhart chart
#[derive(Debug, Clone)]
pub struct ShewhartConfig {
    /// Number of sigma for control limits (typically 3.0)
    pub n_sigma: f64,
    /// Baseline period for estimating mean/std (if None, use all data)
    pub baseline_n: Option<usize>,
}

impl Default for ShewhartConfig {
    fn default() -> Self {
        Self {
            n_sigma: 3.0,
            baseline_n: None,
        }
    }
}

/// Shewhart control chart with Upper Control Limit (UCL) and Lower Control Limit (LCL)
///
/// Points outside [LCL, UCL] = [mean - k*sigma, mean + k*sigma] are flagged.
pub struct SHEWHARTChart {
    config: ShewhartConfig,
}

impl SHEWHARTChart {
    /// Create a new Shewhart chart
    pub fn new(config: ShewhartConfig) -> Self {
        Self { config }
    }

    /// Run the control chart on the data
    pub fn detect(&self, data: &[f64]) -> Result<AnomalyDetectionResult> {
        let n = data.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }

        let baseline_n = self.config.baseline_n.unwrap_or(n);
        let baseline = &data[..baseline_n.min(n)];

        let mean: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let std: f64 = {
            let var =
                baseline.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / baseline.len() as f64;
            var.sqrt().max(1e-15)
        };

        let ucl = mean + self.config.n_sigma * std;
        let lcl = mean - self.config.n_sigma * std;

        let scores: Vec<f64> = data.iter().map(|&x| (x - mean).abs() / std).collect();
        let is_anomaly: Vec<bool> = data.iter().map(|&x| x > ucl || x < lcl).collect();

        Ok(AnomalyDetectionResult {
            scores,
            is_anomaly,
            threshold: self.config.n_sigma,
            control_limits: Some((lcl, ucl)),
        })
    }
}

// ============================================================================
// EWMA Control Chart
// ============================================================================

/// Configuration for the EWMA chart
#[derive(Debug, Clone)]
pub struct EWMAConfig {
    /// Smoothing parameter (0 < lambda <= 1)
    pub lambda: f64,
    /// Control limit multiplier (L)
    pub l_factor: f64,
    /// Baseline for estimating sigma (if None, use the full data)
    pub baseline_n: Option<usize>,
}

impl Default for EWMAConfig {
    fn default() -> Self {
        Self {
            lambda: 0.2,
            l_factor: 3.0,
            baseline_n: None,
        }
    }
}

/// Exponentially Weighted Moving Average (EWMA) control chart
///
/// Z_t = lambda * X_t + (1 - lambda) * Z_{t-1}
/// UCL/LCL = mu ± L * sigma * sqrt(lambda / (2 - lambda) * (1 - (1 - lambda)^{2t}))
pub struct EWMAChart {
    config: EWMAConfig,
}

impl EWMAChart {
    /// Create a new EWMA chart
    pub fn new(config: EWMAConfig) -> Self {
        Self { config }
    }

    /// Run the EWMA chart on the data
    pub fn detect(&self, data: &[f64]) -> Result<AnomalyDetectionResult> {
        let n = data.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }

        let baseline_n = self.config.baseline_n.unwrap_or(n);
        let baseline = &data[..baseline_n.min(n)];
        let mu: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let sigma: f64 = {
            let var =
                baseline.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / baseline.len() as f64;
            var.sqrt().max(1e-15)
        };

        let lambda = self.config.lambda;
        let l = self.config.l_factor;

        let mut z = mu;
        let mut scores = Vec::with_capacity(n);
        let mut is_anomaly = Vec::with_capacity(n);

        for (t, &x) in data.iter().enumerate() {
            z = lambda * x + (1.0 - lambda) * z;
            let t1 = (t + 1) as f64;
            let var_factor = (lambda / (2.0 - lambda)) * (1.0 - (1.0 - lambda).powf(2.0 * t1));
            let cl_half = l * sigma * var_factor.sqrt();
            let ucl = mu + cl_half;
            let lcl = mu - cl_half;

            let score = (z - mu).abs() / (sigma * var_factor.sqrt().max(1e-15));
            scores.push(score);
            is_anomaly.push(z > ucl || z < lcl);
        }

        // Report last UCL/LCL
        let t = n as f64;
        let var_factor = (lambda / (2.0 - lambda)) * (1.0 - (1.0 - lambda).powf(2.0 * t));
        let cl_half = l * sigma * var_factor.sqrt();
        let ucl = mu + cl_half;
        let lcl = mu - cl_half;

        Ok(AnomalyDetectionResult {
            scores,
            is_anomaly,
            threshold: l,
            control_limits: Some((lcl, ucl)),
        })
    }
}

// ============================================================================
// CUSUM Control Chart (two-sided)
// ============================================================================

/// Configuration for the CUSUM chart
#[derive(Debug, Clone)]
pub struct CUSUMConfig {
    /// Reference value k (allowable slack, typically 0.5 * delta where delta is shift in sigma units)
    pub k: f64,
    /// Decision threshold h
    pub h: f64,
    /// Baseline for estimating mean/sigma
    pub baseline_n: Option<usize>,
}

impl Default for CUSUMConfig {
    fn default() -> Self {
        Self {
            k: 0.5,
            h: 4.0,
            baseline_n: None,
        }
    }
}

/// CUSUM state for diagnostic output
#[derive(Debug, Clone)]
pub struct CUSUMState {
    /// Upper CUSUM C+
    pub c_pos: Vec<f64>,
    /// Lower CUSUM C-
    pub c_neg: Vec<f64>,
}

/// Two-sided CUSUM (Cumulative Sum) control chart
///
/// C+_t = max(0, C+_{t-1} + (X_t - mu)/sigma - k)
/// C-_t = max(0, C-_{t-1} - (X_t - mu)/sigma - k)
/// Signal when C+_t > h or C-_t > h
pub struct CUSUMChart {
    config: CUSUMConfig,
}

impl CUSUMChart {
    /// Create a new CUSUM chart
    pub fn new(config: CUSUMConfig) -> Self {
        Self { config }
    }

    /// Run the CUSUM chart and return result with internal state
    pub fn detect_with_state(&self, data: &[f64]) -> Result<(AnomalyDetectionResult, CUSUMState)> {
        let n = data.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 4 observations".to_string(),
                required: 4,
                actual: n,
            });
        }

        let baseline_n = self.config.baseline_n.unwrap_or(n);
        let baseline = &data[..baseline_n.min(n)];
        let mu: f64 = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let sigma: f64 = {
            let var =
                baseline.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / baseline.len() as f64;
            var.sqrt().max(1e-15)
        };

        let k = self.config.k;
        let h = self.config.h;

        let mut c_pos = 0.0_f64;
        let mut c_neg = 0.0_f64;
        let mut c_pos_vec = Vec::with_capacity(n);
        let mut c_neg_vec = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut is_anomaly = Vec::with_capacity(n);

        for &x in data {
            let z = (x - mu) / sigma;
            c_pos = (c_pos + z - k).max(0.0);
            c_neg = (c_neg - z - k).max(0.0);
            c_pos_vec.push(c_pos);
            c_neg_vec.push(c_neg);

            let score = c_pos.max(c_neg);
            scores.push(score / h);
            is_anomaly.push(c_pos > h || c_neg > h);
        }

        Ok((
            AnomalyDetectionResult {
                scores,
                is_anomaly,
                threshold: h,
                control_limits: Some((mu - h * sigma, mu + h * sigma)),
            },
            CUSUMState {
                c_pos: c_pos_vec,
                c_neg: c_neg_vec,
            },
        ))
    }

    /// Detect anomalies (convenience method)
    pub fn detect(&self, data: &[f64]) -> Result<AnomalyDetectionResult> {
        let (result, _) = self.detect_with_state(data)?;
        Ok(result)
    }
}

// ============================================================================
// Isolation Forest for Time Series
// ============================================================================

/// Configuration for IsolationForestTS
#[derive(Debug, Clone)]
pub struct IsolationForestTSConfig {
    /// Window size for feature extraction
    pub window_size: usize,
    /// Stride between windows
    pub stride: usize,
    /// Number of trees in the forest
    pub n_trees: usize,
    /// Subsampling size for each tree
    pub subsample_size: usize,
    /// Contamination rate (fraction of anomalies)
    pub contamination: f64,
    /// Random seed
    pub seed: u64,
}

impl Default for IsolationForestTSConfig {
    fn default() -> Self {
        Self {
            window_size: 10,
            stride: 1,
            n_trees: 100,
            subsample_size: 256,
            contamination: 0.1,
            seed: 42,
        }
    }
}

/// Isolation tree node
#[derive(Debug, Clone)]
enum ITree {
    Leaf {
        size: usize,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<ITree>,
        right: Box<ITree>,
    },
}

/// Isolation Forest adapted for time series windows
///
/// Extracts sliding window features (mean, std, min, max, autocorr)
/// and applies isolation forest to detect anomalous windows.
pub struct IsolationForestTS {
    config: IsolationForestTSConfig,
    trees: Vec<ITree>,
    fitted: bool,
    n_features: usize,
}

impl IsolationForestTS {
    /// Create a new isolation forest for time series
    pub fn new(config: IsolationForestTSConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            fitted: false,
            n_features: 0,
        }
    }

    /// Extract features from a window of data
    fn extract_features(window: &[f64]) -> Vec<f64> {
        let n = window.len() as f64;
        let mean = window.iter().sum::<f64>() / n;
        let var = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let min = window.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Lag-1 autocorrelation
        let autocorr = if window.len() > 1 {
            let cov: f64 = window
                .iter()
                .zip(window[1..].iter())
                .map(|(&a, &b)| (a - mean) * (b - mean))
                .sum::<f64>()
                / (n - 1.0);
            if var > 1e-15 {
                cov / var
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Slope via simple linear regression
        let xs: Vec<f64> = (0..window.len()).map(|i| i as f64).collect();
        let x_mean = (n - 1.0) / 2.0;
        let slope = {
            let num: f64 = xs
                .iter()
                .zip(window.iter())
                .map(|(&x, &y)| (x - x_mean) * (y - mean))
                .sum();
            let den: f64 = xs.iter().map(|&x| (x - x_mean).powi(2)).sum();
            if den > 1e-15 {
                num / den
            } else {
                0.0
            }
        };

        vec![mean, std, min, max, autocorr, slope]
    }

    /// Build an isolation tree from a subsample using LCG PRNG
    fn build_tree(data: &[Vec<f64>], state: &mut u64, max_depth: usize) -> ITree {
        if data.is_empty() || max_depth == 0 {
            return ITree::Leaf { size: data.len() };
        }

        let n_features = data[0].len();

        // Random feature selection
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let feature = ((*state >> 33) as usize) % n_features;

        // Find min/max for the feature
        let (feat_min, feat_max) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), row| {
                (mn.min(row[feature]), mx.max(row[feature]))
            });

        if (feat_max - feat_min).abs() < 1e-15 {
            return ITree::Leaf { size: data.len() };
        }

        // Random split threshold
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let frac = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
        let threshold = feat_min + frac * (feat_max - feat_min);

        let left_data: Vec<Vec<f64>> = data
            .iter()
            .filter(|row| row[feature] < threshold)
            .cloned()
            .collect();
        let right_data: Vec<Vec<f64>> = data
            .iter()
            .filter(|row| row[feature] >= threshold)
            .cloned()
            .collect();

        if left_data.is_empty() || right_data.is_empty() {
            return ITree::Leaf { size: data.len() };
        }

        ITree::Internal {
            feature,
            threshold,
            left: Box::new(Self::build_tree(&left_data, state, max_depth - 1)),
            right: Box::new(Self::build_tree(&right_data, state, max_depth - 1)),
        }
    }

    /// Compute path length for a sample in an isolation tree
    fn path_length(tree: &ITree, sample: &[f64], depth: usize) -> f64 {
        match tree {
            ITree::Leaf { size } => depth as f64 + Self::c_n(*size),
            ITree::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    Self::path_length(left, sample, depth + 1)
                } else {
                    Self::path_length(right, sample, depth + 1)
                }
            }
        }
    }

    /// Expected path length for a random BST with n nodes
    fn c_n(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        let n = n as f64;
        2.0 * (n - 1.0).ln() + 0.5772156649 - 2.0 * (n - 1.0) / n
    }

    /// Fit the isolation forest on training features
    fn fit_features(&mut self, features: &[Vec<f64>]) -> Result<()> {
        let n = features.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 windows to fit isolation forest".to_string(),
                required: 2,
                actual: n,
            });
        }

        let subsample_size = self.config.subsample_size.min(n);
        let max_depth = (subsample_size as f64).log2().ceil() as usize;
        let mut state = self.config.seed;

        self.trees = Vec::with_capacity(self.config.n_trees);
        for _ in 0..self.config.n_trees {
            // Draw subsample
            let subsample: Vec<Vec<f64>> = (0..subsample_size)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let idx = (state >> 33) as usize % n;
                    features[idx].clone()
                })
                .collect();

            self.trees
                .push(Self::build_tree(&subsample, &mut state, max_depth));
        }

        self.n_features = features[0].len();
        self.fitted = true;
        Ok(())
    }

    /// Score features using the fitted forest
    fn score_features(&self, features: &[Vec<f64>]) -> Result<Vec<f64>> {
        if !self.fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "IsolationForestTS not fitted".to_string(),
            ));
        }
        let c_n = Self::c_n(self.config.subsample_size);
        let scores: Vec<f64> = features
            .iter()
            .map(|feat| {
                let avg_path: f64 = self
                    .trees
                    .iter()
                    .map(|tree| Self::path_length(tree, feat, 0))
                    .sum::<f64>()
                    / self.trees.len() as f64;
                2.0_f64.powf(-avg_path / c_n)
            })
            .collect();
        Ok(scores)
    }

    /// Fit and detect anomalies in the time series
    pub fn fit_detect(&mut self, data: &[f64]) -> Result<AnomalyDetectionResult> {
        let n = data.len();
        let w = self.config.window_size;
        let s = self.config.stride;

        if n < w {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series shorter than window size".to_string(),
                required: w,
                actual: n,
            });
        }

        // Extract sliding window features
        let windows: Vec<usize> = (0..=n - w).step_by(s).collect();
        let features: Vec<Vec<f64>> = windows
            .iter()
            .map(|&start| Self::extract_features(&data[start..start + w]))
            .collect();

        self.fit_features(&features)?;
        let window_scores = self.score_features(&features)?;

        // Map window scores back to points (each point gets the score of its covering windows)
        let mut point_scores = vec![0.0_f64; n];
        let mut point_counts = vec![0usize; n];

        for (wi, &start) in windows.iter().enumerate() {
            for i in start..(start + w).min(n) {
                point_scores[i] += window_scores[wi];
                point_counts[i] += 1;
            }
        }

        let scores: Vec<f64> = point_scores
            .iter()
            .zip(point_counts.iter())
            .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
            .collect();

        // Determine threshold based on contamination
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx =
            ((1.0 - self.config.contamination) * sorted_scores.len() as f64) as usize;
        let threshold = sorted_scores
            .get(threshold_idx.min(sorted_scores.len() - 1))
            .cloned()
            .unwrap_or(0.5);

        let is_anomaly: Vec<bool> = scores.iter().map(|&s| s > threshold).collect();

        Ok(AnomalyDetectionResult {
            scores,
            is_anomaly,
            threshold,
            control_limits: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_series(n: usize, anomaly_pos: usize) -> Vec<f64> {
        let mut data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        if anomaly_pos < n {
            data[anomaly_pos] = 100.0;
        }
        data
    }

    #[test]
    fn test_stl_anomaly_detector() {
        let mut data = make_test_series(100, 50);
        data[50] = 20.0; // Clear anomaly
        let detector = STLAnomalyDetector::new(STLAnomalyConfig {
            period: 10,
            ..Default::default()
        });
        let result = detector
            .detect(&data)
            .expect("STL anomaly detection failed");
        assert_eq!(result.scores.len(), 100);
        assert_eq!(result.is_anomaly.len(), 100);
        // Index 50 should be flagged
        assert!(result.is_anomaly[50], "Anomaly at index 50 not detected");
    }

    #[test]
    fn test_gesd() {
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[50] = 1000.0;
        let gesd = GESD::new(5, 0.05);
        let result = gesd.test(&data).expect("GESD failed");
        assert!(result.n_outliers > 0, "GESD should detect the outlier");
    }

    #[test]
    fn test_shewhart_chart() {
        let data = make_test_series(100, 50);
        let chart = SHEWHARTChart::new(ShewhartConfig::default());
        let result = chart.detect(&data).expect("Shewhart failed");
        assert_eq!(result.scores.len(), 100);
        assert!(result.control_limits.is_some());
    }

    #[test]
    fn test_ewma_chart() {
        let data = make_test_series(100, 50);
        let chart = EWMAChart::new(EWMAConfig::default());
        let result = chart.detect(&data).expect("EWMA failed");
        assert_eq!(result.scores.len(), 100);
    }

    #[test]
    fn test_cusum_chart() {
        let mut data: Vec<f64> = (0..100).map(|_| 0.0_f64).collect();
        // Introduce a shift at t=50
        for i in 50..100 {
            data[i] = 3.0;
        }
        let chart = CUSUMChart::new(CUSUMConfig::default());
        let (result, state) = chart.detect_with_state(&data).expect("CUSUM failed");
        assert_eq!(result.scores.len(), 100);
        assert_eq!(state.c_pos.len(), 100);
        // After the shift, C+ should eventually exceed h
        assert!(
            result.is_anomaly.iter().skip(50).any(|&a| a),
            "CUSUM should detect shift"
        );
    }

    #[test]
    fn test_isolation_forest_ts() {
        let mut data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        data[100] = 50.0; // Anomaly
        let mut forest = IsolationForestTS::new(IsolationForestTSConfig {
            window_size: 10,
            n_trees: 10,
            subsample_size: 50,
            ..Default::default()
        });
        let result = forest.fit_detect(&data).expect("IsolationForestTS failed");
        assert_eq!(result.scores.len(), 200);
    }
}
