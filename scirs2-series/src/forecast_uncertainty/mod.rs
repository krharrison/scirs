//! Forecast uncertainty quantification for time series
//!
//! Provides:
//! - `ConformalPrediction`: Split conformal prediction intervals
//! - `EnbPI`: Ensemble conformal prediction for online prediction intervals
//! - `BootstrapPI`: Bootstrap prediction intervals (residual/block bootstrap)
//! - `BayesianPI`: Bayesian posterior predictive intervals
//! - `CoverageTest`: Empirical coverage test for prediction intervals

use crate::error::{Result, TimeSeriesError};

// ============================================================================
// Common types
// ============================================================================

/// A prediction interval at a given time index
#[derive(Debug, Clone)]
pub struct PredictionInterval {
    /// Lower bound of the interval
    pub lower: f64,
    /// Point prediction (center)
    pub prediction: f64,
    /// Upper bound of the interval
    pub upper: f64,
    /// Nominal coverage level (e.g., 0.95 for 95% PI)
    pub coverage_level: f64,
}

/// Collection of prediction intervals
#[derive(Debug, Clone)]
pub struct PredictionIntervals {
    /// Intervals for each forecast horizon
    pub intervals: Vec<PredictionInterval>,
    /// Method used to compute the intervals
    pub method: String,
    /// Actual empirical coverage (if evaluation data available)
    pub empirical_coverage: Option<f64>,
}

// ============================================================================
// Conformal Prediction
// ============================================================================

/// Configuration for split conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Coverage level (1 - alpha), e.g., 0.95
    pub coverage: f64,
    /// Fraction of data to use as calibration set
    pub calibration_fraction: f64,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            calibration_fraction: 0.2,
        }
    }
}

/// Split Conformal Prediction for time series
///
/// Uses a held-out calibration set to compute the conformal quantile.
/// Given point predictions from any forecaster, constructs prediction intervals
/// with guaranteed marginal coverage under exchangeability.
///
/// Algorithm:
/// 1. Split data into train / calibration
/// 2. Fit model on train, compute residuals on calibration
/// 3. Compute quantile q_{ceil((n+1)(1-alpha))/n} of |residuals|
/// 4. PI for new x: [ŷ - q, ŷ + q]
pub struct ConformalPrediction {
    config: ConformalConfig,
    /// Calibration scores (absolute residuals), set after calibration
    calibration_scores: Option<Vec<f64>>,
    /// Conformal quantile
    quantile: Option<f64>,
}

impl ConformalPrediction {
    /// Create a new conformal predictor
    pub fn new(config: ConformalConfig) -> Self {
        Self {
            config,
            calibration_scores: None,
            quantile: None,
        }
    }

    /// Calibrate using residuals on a held-out calibration set
    ///
    /// `actuals`: true values in calibration set
    /// `predictions`: model predictions for calibration set
    pub fn calibrate(&mut self, actuals: &[f64], predictions: &[f64]) -> Result<()> {
        if actuals.len() != predictions.len() {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: actuals.len(),
                actual: predictions.len(),
            });
        }
        if actuals.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "Calibration set is empty".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let scores: Vec<f64> = actuals.iter().zip(predictions.iter())
            .map(|(&a, &p)| (a - p).abs())
            .collect();

        let n = scores.len();
        let alpha = 1.0 - self.config.coverage;
        // Conformal quantile at level ceil((n+1)(1-alpha))/n
        let q_idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let q_idx = q_idx.min(n); // Clamp to n (returns +inf if out of range)

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let quantile = if q_idx >= n {
            f64::INFINITY
        } else {
            sorted[q_idx - 1]
        };

        self.calibration_scores = Some(scores);
        self.quantile = Some(quantile);
        Ok(())
    }

    /// Predict intervals for new point predictions
    pub fn predict_intervals(&self, predictions: &[f64]) -> Result<PredictionIntervals> {
        let q = self.quantile.ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("ConformalPrediction not calibrated".to_string())
        })?;

        let intervals: Vec<PredictionInterval> = predictions.iter().map(|&pred| {
            PredictionInterval {
                lower: pred - q,
                prediction: pred,
                upper: pred + q,
                coverage_level: self.config.coverage,
            }
        }).collect();

        Ok(PredictionIntervals {
            intervals,
            method: "SplitConformal".to_string(),
            empirical_coverage: None,
        })
    }

    /// Calibrate and predict in one step
    pub fn fit_predict(
        &mut self,
        train_actuals: &[f64],
        train_preds: &[f64],
        test_preds: &[f64],
    ) -> Result<PredictionIntervals> {
        self.calibrate(train_actuals, train_preds)?;
        self.predict_intervals(test_preds)
    }
}

// ============================================================================
// EnbPI: Ensemble Conformal Prediction with Bonus
// ============================================================================

/// Configuration for EnbPI
#[derive(Debug, Clone)]
pub struct EnbPIConfig {
    /// Coverage level (1 - alpha)
    pub coverage: f64,
    /// Number of bootstrap ensembles
    pub n_ensembles: usize,
    /// Exponential forgetting factor (beta) for online update
    pub beta: f64,
    /// Random seed
    pub seed: u64,
}

impl Default for EnbPIConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            n_ensembles: 50,
            beta: 0.95,
            seed: 42,
        }
    }
}

/// EnbPI: Ensemble Conformal Prediction for Online Prediction Intervals
///
/// Uses a jackknife+ style ensemble of in-sample predictions to form
/// conformal scores. When new observations arrive, the quantile is
/// updated online using an exponentially decaying window.
///
/// Reference: Xu & Xie (2021) "Conformal prediction interval for dynamic time-series"
pub struct EnbPI {
    config: EnbPIConfig,
    /// Stored conformal scores (sliding window)
    scores: Vec<f64>,
    /// Ensemble predictions on training data
    ensemble_preds: Option<Vec<f64>>,
}

impl EnbPI {
    /// Create a new EnbPI predictor
    pub fn new(config: EnbPIConfig) -> Self {
        Self {
            config,
            scores: Vec::new(),
            ensemble_preds: None,
        }
    }

    /// Initialize with training residuals
    ///
    /// In a real implementation, this would use leave-one-out ensemble predictions.
    /// Here we use the provided residuals directly as calibration scores.
    pub fn initialize(&mut self, residuals: &[f64]) -> Result<()> {
        if residuals.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "Residuals for EnbPI initialization cannot be empty".to_string(),
                required: 1,
                actual: 0,
            });
        }
        self.scores = residuals.iter().map(|&r| r.abs()).collect();
        Ok(())
    }

    /// Compute the current conformal quantile from stored scores
    fn current_quantile(&self) -> f64 {
        let n = self.scores.len();
        if n == 0 {
            return f64::INFINITY;
        }
        let alpha = 1.0 - self.config.coverage;
        let q_idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let q_idx = q_idx.min(n);

        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        if q_idx == 0 || q_idx > sorted.len() {
            f64::INFINITY
        } else {
            sorted[q_idx - 1]
        }
    }

    /// Predict an interval for the next timestep, then update with the observed value
    pub fn predict_and_update(&mut self, prediction: f64, actual: Option<f64>) -> Result<PredictionInterval> {
        let q = self.current_quantile();

        let interval = PredictionInterval {
            lower: prediction - q,
            prediction,
            upper: prediction + q,
            coverage_level: self.config.coverage,
        };

        // If we observe the actual value, update the score sequence
        if let Some(obs) = actual {
            let new_score = (obs - prediction).abs();
            self.scores.push(new_score);

            // Exponential forgetting: remove the oldest score with probability (1 - beta)
            // Simplified: keep the scores weighted by recency
            if self.scores.len() > 1 {
                // Trim scores: keep approximately beta * T scores
                let target_len = (self.scores.len() as f64 * self.config.beta).ceil() as usize;
                let target_len = target_len.max(1);
                if self.scores.len() > target_len * 2 {
                    // Remove oldest entries
                    let remove_n = self.scores.len() - target_len;
                    self.scores.drain(0..remove_n);
                }
            }
        }

        Ok(interval)
    }

    /// Batch predict intervals (offline mode, no online updates)
    pub fn predict_batch(&self, predictions: &[f64]) -> Result<PredictionIntervals> {
        let q = self.current_quantile();

        let intervals: Vec<PredictionInterval> = predictions.iter().map(|&pred| {
            PredictionInterval {
                lower: pred - q,
                prediction: pred,
                upper: pred + q,
                coverage_level: self.config.coverage,
            }
        }).collect();

        Ok(PredictionIntervals {
            intervals,
            method: "EnbPI".to_string(),
            empirical_coverage: None,
        })
    }
}

// ============================================================================
// Bootstrap Prediction Intervals
// ============================================================================

/// Bootstrap method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Resample individual residuals (assumes i.i.d. residuals)
    Residual,
    /// Block bootstrap (preserves local dependence structure)
    Block,
}

/// Configuration for bootstrap prediction intervals
#[derive(Debug, Clone)]
pub struct BootstrapPIConfig {
    /// Coverage level (1 - alpha)
    pub coverage: f64,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Block length for block bootstrap
    pub block_length: usize,
    /// Bootstrap method
    pub method: BootstrapMethod,
    /// Random seed
    pub seed: u64,
}

impl Default for BootstrapPIConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            n_bootstrap: 1000,
            block_length: 10,
            method: BootstrapMethod::Residual,
            seed: 42,
        }
    }
}

/// Bootstrap Prediction Intervals
///
/// Constructs prediction intervals by resampling residuals from a fitted model.
/// Supports both i.i.d. residual bootstrap and block bootstrap for dependent data.
pub struct BootstrapPI {
    config: BootstrapPIConfig,
    residuals: Option<Vec<f64>>,
}

impl BootstrapPI {
    /// Create a new BootstrapPI instance
    pub fn new(config: BootstrapPIConfig) -> Self {
        Self {
            config,
            residuals: None,
        }
    }

    /// Set the training residuals from a fitted model
    pub fn set_residuals(&mut self, residuals: Vec<f64>) -> Result<()> {
        if residuals.is_empty() {
            return Err(TimeSeriesError::InsufficientData {
                message: "Residuals cannot be empty".to_string(),
                required: 1,
                actual: 0,
            });
        }
        self.residuals = Some(residuals);
        Ok(())
    }

    /// LCG random number generator
    fn lcg_sample(state: &mut u64, n: usize) -> usize {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as usize) % n
    }

    /// Sample residuals using block bootstrap
    fn block_bootstrap_residuals(residuals: &[f64], block_len: usize, state: &mut u64) -> Vec<f64> {
        let n = residuals.len();
        let n_blocks = (n + block_len - 1) / block_len;
        let mut sample = Vec::with_capacity(n);

        for _ in 0..n_blocks {
            let start = Self::lcg_sample(state, n.saturating_sub(block_len).max(1));
            for j in 0..block_len {
                if sample.len() >= n {
                    break;
                }
                sample.push(residuals[(start + j) % n]);
            }
        }
        sample.truncate(n);
        sample
    }

    /// Compute prediction intervals for a given set of point predictions
    ///
    /// `predictions`: Point predictions for each forecast horizon
    /// Returns: PI for each prediction
    pub fn predict_intervals(&self, predictions: &[f64]) -> Result<PredictionIntervals> {
        let residuals = self.residuals.as_ref().ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("BootstrapPI residuals not set".to_string())
        })?;

        let n_res = residuals.len();
        if n_res == 0 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Empty residuals".to_string(),
                required: 1,
                actual: 0,
            });
        }

        let alpha = 1.0 - self.config.coverage;
        let lower_q = (alpha / 2.0 * self.config.n_bootstrap as f64) as usize;
        let upper_q = ((1.0 - alpha / 2.0) * self.config.n_bootstrap as f64) as usize;

        let mut state = self.config.seed;

        let intervals: Vec<PredictionInterval> = predictions.iter().map(|&pred| {
            // For each forecast horizon, bootstrap the residuals
            let mut boot_vals: Vec<f64> = (0..self.config.n_bootstrap)
                .map(|_| {
                    let sampled_res = match self.config.method {
                        BootstrapMethod::Residual => {
                            let idx = Self::lcg_sample(&mut state, n_res);
                            residuals[idx]
                        }
                        BootstrapMethod::Block => {
                            let boot = Self::block_bootstrap_residuals(residuals, self.config.block_length, &mut state);
                            let idx = Self::lcg_sample(&mut state, boot.len().max(1));
                            boot.get(idx).cloned().unwrap_or(0.0)
                        }
                    };
                    pred + sampled_res
                })
                .collect();

            boot_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let lower = boot_vals.get(lower_q.max(1) - 1).cloned().unwrap_or(pred);
            let upper = boot_vals.get(upper_q.min(boot_vals.len()) - 1).cloned().unwrap_or(pred);

            PredictionInterval {
                lower,
                prediction: pred,
                upper,
                coverage_level: self.config.coverage,
            }
        }).collect();

        Ok(PredictionIntervals {
            intervals,
            method: format!("Bootstrap({})", match self.config.method {
                BootstrapMethod::Residual => "Residual",
                BootstrapMethod::Block => "Block",
            }),
            empirical_coverage: None,
        })
    }
}

// ============================================================================
// Bayesian Prediction Intervals
// ============================================================================

/// Configuration for Bayesian prediction intervals
#[derive(Debug, Clone)]
pub struct BayesianPIConfig {
    /// Coverage level (1 - alpha)
    pub coverage: f64,
    /// Prior variance for the mean parameter
    pub prior_var: f64,
    /// Number of Monte Carlo samples for posterior predictive
    pub n_samples: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for BayesianPIConfig {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            prior_var: 1.0,
            n_samples: 10000,
            seed: 42,
        }
    }
}

/// Bayesian Posterior Predictive Intervals
///
/// Uses a Normal-Normal conjugate model:
/// - Prior: mu ~ N(mu_0, tau^2)
/// - Likelihood: X_i | mu ~ N(mu, sigma^2)
/// - Posterior: mu | X ~ N(mu_n, tau_n^2)
/// - Posterior predictive: X_new | X ~ N(mu_n, sigma^2 + tau_n^2)
///
/// The predictive interval is the HPD (highest posterior density) interval
/// from the posterior predictive distribution.
pub struct BayesianPI {
    config: BayesianPIConfig,
    /// Estimated noise variance
    sigma_sq: Option<f64>,
    /// Posterior mean
    posterior_mean: Option<f64>,
    /// Posterior variance of the mean
    posterior_var: Option<f64>,
}

impl BayesianPI {
    /// Create a new BayesianPI instance
    pub fn new(config: BayesianPIConfig) -> Self {
        Self {
            config,
            sigma_sq: None,
            posterior_mean: None,
            posterior_var: None,
        }
    }

    /// Fit the Bayesian model on training residuals
    pub fn fit(&mut self, residuals: &[f64]) -> Result<()> {
        let n = residuals.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 2 residuals for Bayesian PI".to_string(),
                required: 2,
                actual: n,
            });
        }

        let mean: f64 = residuals.iter().sum::<f64>() / n as f64;
        let var: f64 = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let sigma_sq = var.max(1e-15);

        // Prior: mu_0 = 0, tau^2 = prior_var
        let tau_sq = self.config.prior_var;
        let mu_0 = 0.0_f64;

        // Posterior update (Normal-Normal conjugate):
        // tau_n^2 = 1 / (1/tau^2 + n/sigma^2)
        let tau_n_sq = 1.0 / (1.0 / tau_sq + n as f64 / sigma_sq);
        // mu_n = tau_n^2 * (mu_0 / tau^2 + sum(x_i) / sigma^2)
        let sum_res: f64 = residuals.iter().sum();
        let mu_n = tau_n_sq * (mu_0 / tau_sq + sum_res / sigma_sq);

        self.sigma_sq = Some(sigma_sq);
        self.posterior_mean = Some(mu_n);
        self.posterior_var = Some(tau_n_sq);

        Ok(())
    }

    /// Normal quantile function (Beasley-Springer-Moro)
    fn normal_quantile(p: f64) -> f64 {
        let p = p.clamp(1e-10, 1.0 - 1e-10);
        let t = (-2.0 * (p.min(1.0 - p)).ln()).sqrt();
        let c = [2.515517_f64, 0.802853, 0.010328];
        let d = [1.432788_f64, 0.189269, 0.001308];
        let num = c[0] + c[1] * t + c[2] * t * t;
        let den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t;
        let z = t - num / den;
        if p >= 0.5 { z } else { -z }
    }

    /// Compute posterior predictive intervals for point predictions
    pub fn predict_intervals(&self, predictions: &[f64]) -> Result<PredictionIntervals> {
        let sigma_sq = self.sigma_sq.ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("BayesianPI not fitted".to_string())
        })?;
        let posterior_mean = self.posterior_mean.ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("BayesianPI not fitted".to_string())
        })?;
        let posterior_var = self.posterior_var.ok_or_else(|| {
            TimeSeriesError::ModelNotFitted("BayesianPI not fitted".to_string())
        })?;

        // Posterior predictive variance = sigma^2 + tau_n^2
        let pred_var = sigma_sq + posterior_var;
        let pred_std = pred_var.sqrt();

        let alpha = 1.0 - self.config.coverage;
        let z_lower = Self::normal_quantile(alpha / 2.0);
        let z_upper = Self::normal_quantile(1.0 - alpha / 2.0);

        let intervals: Vec<PredictionInterval> = predictions.iter().map(|&point_pred| {
            // The predictive mean shifts by the posterior residual mean
            let pred_center = point_pred + posterior_mean;
            PredictionInterval {
                lower: pred_center + z_lower * pred_std,
                prediction: pred_center,
                upper: pred_center + z_upper * pred_std,
                coverage_level: self.config.coverage,
            }
        }).collect();

        Ok(PredictionIntervals {
            intervals,
            method: "BayesianPosteriorPredictive".to_string(),
            empirical_coverage: None,
        })
    }
}

// ============================================================================
// Coverage Test
// ============================================================================

/// Result of an empirical coverage test
#[derive(Debug, Clone)]
pub struct CoverageTestResult {
    /// Empirical coverage rate
    pub empirical_coverage: f64,
    /// Nominal coverage level
    pub nominal_coverage: f64,
    /// Number of observations tested
    pub n_observations: usize,
    /// Number of observations covered by the interval
    pub n_covered: usize,
    /// Coverage deficit (positive if over-covering, negative if under-covering)
    pub coverage_deficit: f64,
    /// Winkler score (interval score; penalizes width)
    pub winkler_score: f64,
    /// Mean interval width
    pub mean_width: f64,
    /// Whether the coverage is acceptable at the given tolerance
    pub passes: bool,
    /// Tolerance used for pass/fail decision
    pub tolerance: f64,
}

/// Coverage Test for prediction intervals
///
/// Computes empirical coverage and related diagnostics for evaluating
/// the validity of prediction intervals.
pub struct CoverageTest {
    /// Tolerance for acceptable deviation from nominal coverage
    pub tolerance: f64,
}

impl CoverageTest {
    /// Create a new coverage test with given tolerance
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Evaluate coverage of prediction intervals against actual observations
    ///
    /// `actuals`: True observed values
    /// `intervals`: Prediction intervals to evaluate
    pub fn evaluate(
        &self,
        actuals: &[f64],
        intervals: &PredictionIntervals,
    ) -> Result<CoverageTestResult> {
        let n = actuals.len();
        if n == 0 {
            return Err(TimeSeriesError::InsufficientData {
                message: "No observations to evaluate".to_string(),
                required: 1,
                actual: 0,
            });
        }
        if intervals.intervals.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: intervals.intervals.len(),
            });
        }

        let n_covered: usize = actuals.iter().zip(intervals.intervals.iter())
            .filter(|(&a, pi)| a >= pi.lower && a <= pi.upper)
            .count();

        let empirical_coverage = n_covered as f64 / n as f64;
        let nominal_coverage = intervals.intervals.first().map(|pi| pi.coverage_level).unwrap_or(0.95);

        let alpha = 1.0 - nominal_coverage;
        let mut winkler_total = 0.0_f64;
        let mut total_width = 0.0_f64;

        for (&a, pi) in actuals.iter().zip(intervals.intervals.iter()) {
            let width = pi.upper - pi.lower;
            total_width += width;
            // Winkler interval score: width + 2/alpha * penalty for misses
            let penalty = if a < pi.lower {
                2.0 / alpha * (pi.lower - a)
            } else if a > pi.upper {
                2.0 / alpha * (a - pi.upper)
            } else {
                0.0
            };
            winkler_total += width + penalty;
        }

        let winkler_score = winkler_total / n as f64;
        let mean_width = total_width / n as f64;
        let coverage_deficit = empirical_coverage - nominal_coverage;
        let passes = coverage_deficit.abs() <= self.tolerance;

        Ok(CoverageTestResult {
            empirical_coverage,
            nominal_coverage,
            n_observations: n,
            n_covered,
            coverage_deficit,
            winkler_score,
            mean_width,
            passes,
            tolerance: self.tolerance,
        })
    }

    /// Compute conditional coverage (rolling window coverage)
    pub fn rolling_coverage(
        &self,
        actuals: &[f64],
        intervals: &PredictionIntervals,
        window: usize,
    ) -> Result<Vec<f64>> {
        let n = actuals.len();
        if n < window {
            return Err(TimeSeriesError::InsufficientData {
                message: "Data too short for rolling coverage".to_string(),
                required: window,
                actual: n,
            });
        }
        if intervals.intervals.len() != n {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: n,
                actual: intervals.intervals.len(),
            });
        }

        let covered: Vec<bool> = actuals.iter().zip(intervals.intervals.iter())
            .map(|(&a, pi)| a >= pi.lower && a <= pi.upper)
            .collect();

        let rolling: Vec<f64> = (0..=n - window)
            .map(|i| {
                covered[i..i + window].iter().filter(|&&c| c).count() as f64 / window as f64
            })
            .collect();

        Ok(rolling)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_noisy_sine(n: usize, noise_std: f64) -> (Vec<f64>, Vec<f64>) {
        let mut state = 42_u64;
        let actuals: Vec<f64> = (0..n).map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 2.0 * noise_std;
            (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin() + noise
        }).collect();
        let predictions: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin())
            .collect();
        (actuals, predictions)
    }

    #[test]
    fn test_conformal_prediction_calibrate_predict() {
        let (actuals, preds) = make_noisy_sine(100, 0.5);
        let cal_actuals = &actuals[..80];
        let cal_preds = &preds[..80];
        let test_preds = &preds[80..];

        let mut cp = ConformalPrediction::new(ConformalConfig { coverage: 0.9, ..Default::default() });
        cp.calibrate(cal_actuals, cal_preds).expect("Calibration failed");
        let intervals = cp.predict_intervals(test_preds).expect("Prediction failed");
        assert_eq!(intervals.intervals.len(), 20);
        for pi in &intervals.intervals {
            assert!(pi.lower <= pi.upper, "Lower bound exceeds upper bound");
            assert!((pi.coverage_level - 0.9).abs() < 1e-10);
        }
    }

    #[test]
    fn test_enbpi_predict_update() {
        let (actuals, preds) = make_noisy_sine(100, 0.3);
        let train_residuals: Vec<f64> = actuals[..80].iter().zip(preds[..80].iter())
            .map(|(&a, &p)| a - p)
            .collect();

        let mut enbpi = EnbPI::new(EnbPIConfig::default());
        enbpi.initialize(&train_residuals).expect("Initialize failed");

        for i in 80..100 {
            let pi = enbpi.predict_and_update(preds[i], Some(actuals[i])).expect("Predict failed");
            assert!(pi.lower <= pi.upper);
        }
    }

    #[test]
    fn test_bootstrap_pi_residual() {
        let (actuals, preds) = make_noisy_sine(100, 0.5);
        let residuals: Vec<f64> = actuals.iter().zip(preds.iter()).map(|(&a, &p)| a - p).collect();

        let mut bpi = BootstrapPI::new(BootstrapPIConfig {
            n_bootstrap: 100,
            ..Default::default()
        });
        bpi.set_residuals(residuals).expect("Set residuals failed");

        let test_preds = vec![0.0, 0.5, 1.0, 0.5];
        let intervals = bpi.predict_intervals(&test_preds).expect("Prediction failed");
        assert_eq!(intervals.intervals.len(), 4);
        for pi in &intervals.intervals {
            assert!(pi.lower <= pi.upper, "Lower bound exceeds upper bound");
        }
    }

    #[test]
    fn test_bootstrap_pi_block() {
        let (actuals, preds) = make_noisy_sine(100, 0.5);
        let residuals: Vec<f64> = actuals.iter().zip(preds.iter()).map(|(&a, &p)| a - p).collect();

        let mut bpi = BootstrapPI::new(BootstrapPIConfig {
            method: BootstrapMethod::Block,
            block_length: 5,
            n_bootstrap: 100,
            ..Default::default()
        });
        bpi.set_residuals(residuals).expect("Set residuals failed");

        let test_preds = vec![0.0, 0.5];
        let intervals = bpi.predict_intervals(&test_preds).expect("Prediction failed");
        assert_eq!(intervals.intervals.len(), 2);
    }

    #[test]
    fn test_bayesian_pi() {
        let (actuals, preds) = make_noisy_sine(100, 0.5);
        let residuals: Vec<f64> = actuals.iter().zip(preds.iter()).map(|(&a, &p)| a - p).collect();

        let mut bpi = BayesianPI::new(BayesianPIConfig::default());
        bpi.fit(&residuals).expect("Fit failed");

        let test_preds = &preds[80..100];
        let intervals = bpi.predict_intervals(test_preds).expect("Prediction failed");
        assert_eq!(intervals.intervals.len(), 20);
        for pi in &intervals.intervals {
            assert!(pi.lower <= pi.upper, "Lower bound exceeds upper bound");
        }
    }

    #[test]
    fn test_coverage_test_evaluate() {
        let (actuals, preds) = make_noisy_sine(100, 0.3);
        let residuals: Vec<f64> = actuals.iter().zip(preds.iter()).map(|(&a, &p)| a - p).collect();

        let mut bpi = BootstrapPI::new(BootstrapPIConfig { n_bootstrap: 200, ..Default::default() });
        bpi.set_residuals(residuals).expect("Set residuals failed");

        let test_actuals = &actuals[..50];
        let test_preds_slice = &preds[..50];
        let intervals = bpi.predict_intervals(test_preds_slice).expect("Prediction failed");

        let ct = CoverageTest::new(0.15); // 15% tolerance
        let result = ct.evaluate(test_actuals, &intervals).expect("Coverage test failed");
        assert!(result.empirical_coverage >= 0.0 && result.empirical_coverage <= 1.0);
        assert_eq!(result.n_observations, 50);
        assert!(result.mean_width >= 0.0);
        assert!(result.winkler_score >= 0.0);
    }

    #[test]
    fn test_rolling_coverage() {
        let (actuals, preds) = make_noisy_sine(100, 0.3);
        let residuals: Vec<f64> = actuals.iter().zip(preds.iter()).map(|(&a, &p)| a - p).collect();

        let mut bpi = BootstrapPI::new(BootstrapPIConfig { n_bootstrap: 100, ..Default::default() });
        bpi.set_residuals(residuals).expect("Set residuals failed");

        let intervals = bpi.predict_intervals(&preds).expect("Prediction failed");
        let ct = CoverageTest::new(0.1);
        let rolling = ct.rolling_coverage(&actuals, &intervals, 20).expect("Rolling coverage failed");
        assert_eq!(rolling.len(), 81); // 100 - 20 + 1
        for &cov in &rolling {
            assert!(cov >= 0.0 && cov <= 1.0);
        }
    }
}
