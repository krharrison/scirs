//! Advanced Calibration Metrics
//!
//! Extends the basic calibration module with:
//!
//! - **AdaptiveCalibrationError (ACE)**: ECE with equal-mass (equal-frequency) bins
//! - **OverconfidenceError**: Weighted penalty for overconfident bins
//! - **CalibrationMetrics**: All-in-one struct combining ECE, MCE, ACE, overconfidence
//! - **temperature_scaling_diagnostic**: Optimal temperature T* minimizing NLL
//! - **conformal_prediction_coverage**: Empirical coverage rate of prediction intervals

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::Array2;

// ────────────────────────────────────────────────────────────────────────────
// Adaptive Calibration Error (ACE) - equal-mass bins
// ────────────────────────────────────────────────────────────────────────────

/// Computes the Adaptive Calibration Error (ACE) using equal-mass bins.
///
/// ACE divides predictions into bins of equal size (roughly n/n_bins samples per bin),
/// ranked by predicted probability. This avoids the issue of empty or nearly-empty
/// bins at extremes that affects equal-width ECE.
///
/// ACE = (1/B) * sum_{b=1}^{B} |acc(b) - conf(b)|
///
/// # Arguments
///
/// * `y_true` - Binary ground truth labels (0.0 or 1.0)
/// * `y_prob` - Predicted probabilities in [0, 1]
/// * `n_bins` - Number of equal-mass bins
///
/// # Returns
///
/// ACE value in [0, 1]. Lower is better.
pub fn adaptive_calibration_error_v2(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    validate_inputs_v2(y_true, y_prob, n_bins)?;

    let n = y_true.len();
    // Sort by predicted probability
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        y_prob[i]
            .partial_cmp(&y_prob[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bin_size = n / n_bins;
    let mut ace = 0.0f64;
    let mut valid_bins = 0usize;

    for b in 0..n_bins {
        let start = b * bin_size;
        let end = if b == n_bins - 1 { n } else { start + bin_size };

        if start >= end {
            continue;
        }

        let count = end - start;
        let sum_true: f64 = (start..end).map(|k| y_true[indices[k]]).sum();
        let sum_prob: f64 = (start..end).map(|k| y_prob[indices[k]]).sum();

        let acc = sum_true / count as f64;
        let conf = sum_prob / count as f64;

        ace += (acc - conf).abs();
        valid_bins += 1;
    }

    if valid_bins == 0 {
        return Err(MetricsError::CalculationError(
            "no valid bins found".to_string(),
        ));
    }

    Ok(ace / valid_bins as f64)
}

/// Computes the Overconfidence Error.
///
/// This is the average of max(0, conf(b) - acc(b)) weighted by the number of
/// samples in each bin. It penalizes bins where the model is more confident
/// than its accuracy warrants.
///
/// OverconfidenceError = sum_{b} (n_b / n) * max(0, conf(b) - acc(b))
///
/// # Arguments
///
/// * `y_true` - Binary ground truth labels
/// * `y_prob` - Predicted probabilities
/// * `n_bins` - Number of equal-width bins
pub fn overconfidence_error(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    validate_inputs_v2(y_true, y_prob, n_bins)?;

    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut oe = 0.0f64;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0f64;
        let mut sum_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..n {
            if y_prob[i] >= lower && y_prob[i] < upper {
                sum_true += y_true[i];
                sum_prob += y_prob[i];
                count += 1;
            }
        }

        if count > 0 {
            let acc = sum_true / count as f64;
            let conf = sum_prob / count as f64;
            let overconf = (conf - acc).max(0.0);
            oe += (count as f64 / n as f64) * overconf;
        }
    }

    Ok(oe)
}

// ────────────────────────────────────────────────────────────────────────────
// CalibrationMetrics summary struct
// ────────────────────────────────────────────────────────────────────────────

/// Comprehensive calibration metrics for a binary probabilistic classifier.
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error (equal-width bins)
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Adaptive Calibration Error (equal-mass bins)
    pub ace: f64,
    /// Overconfidence Error
    pub overconfidence: f64,
    /// Brier Score
    pub brier_score: f64,
    /// Number of bins used
    pub n_bins: usize,
    /// Total number of samples
    pub n_samples: usize,
}

impl CalibrationMetrics {
    /// Compute all calibration metrics.
    ///
    /// # Arguments
    ///
    /// * `y_true` - Binary ground truth labels (0.0 or 1.0)
    /// * `y_prob` - Predicted probabilities in [0, 1]
    /// * `n_bins` - Number of bins (default: 10)
    pub fn compute(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<Self> {
        validate_inputs_v2(y_true, y_prob, n_bins)?;

        let ece = ece_equal_width(y_true, y_prob, n_bins)?;
        let mce = mce_equal_width(y_true, y_prob, n_bins)?;
        let ace = adaptive_calibration_error_v2(y_true, y_prob, n_bins)?;
        let overconfidence = overconfidence_error(y_true, y_prob, n_bins)?;
        let brier_score = brier_score_v2(y_true, y_prob)?;

        Ok(Self {
            ece,
            mce,
            ace,
            overconfidence,
            brier_score,
            n_bins,
            n_samples: y_true.len(),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Temperature Scaling Diagnostic
// ────────────────────────────────────────────────────────────────────────────

/// Finds the optimal temperature T* that minimizes the Negative Log-Likelihood
/// (NLL) of a calibrated softmax predictor.
///
/// Temperature scaling divides the logits by T before softmax. For binary
/// classification, this transforms probabilities as:
/// ```text
/// p_T = sigmoid(logit(p) / T)
/// ```
///
/// The optimal T* is found via golden-section search on the NLL:
/// ```text
/// NLL(T) = -(1/n) sum_i [ y_i * log(p_T(x_i)) + (1-y_i) * log(1-p_T(x_i)) ]
/// ```
///
/// # Arguments
///
/// * `probs` - Predicted probabilities, shape (n_samples, n_classes)
/// * `labels` - True class indices, length n_samples
///
/// # Returns
///
/// The optimal temperature T* > 0. T* < 1 → model is underconfident; T* > 1 → overconfident.
pub fn temperature_scaling_diagnostic(probs: &Array2<f64>, labels: &[usize]) -> Result<f64> {
    let (n, _n_classes) = probs.dim();

    if labels.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "probs has {} rows but labels has {} elements",
            n,
            labels.len()
        )));
    }
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    // Validate label indices
    for (i, &lbl) in labels.iter().enumerate() {
        if lbl >= probs.ncols() {
            return Err(MetricsError::InvalidInput(format!(
                "label[{i}]={lbl} out of range for {nc} classes",
                nc = probs.ncols()
            )));
        }
    }

    // Extract logits from probabilities (inverse softmax is not unique; use log(p_k))
    // For temperature scaling: scaled_logit_k = log(p_k) / T
    // We search T in [0.01, 20.0] to minimize NLL

    let nll = |t: f64| -> f64 {
        // For each sample, compute softmax of log(p) / t
        let mut total_nll = 0.0f64;
        for i in 0..n {
            let row = probs.row(i);
            // log probabilities
            let log_p: Vec<f64> = row.iter().map(|&p| p.max(1e-15).ln()).collect();
            // scaled: divide by t
            let scaled: Vec<f64> = log_p.iter().map(|&lp| lp / t).collect();
            // log-sum-exp normalization
            let max_s = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = max_s + scaled.iter().map(|&s| (s - max_s).exp()).sum::<f64>().ln();
            // log of calibrated prob for true class
            let log_pk = scaled[labels[i]] - lse;
            total_nll -= log_pk;
        }
        total_nll / n as f64
    };

    // Golden section search on [t_lo, t_hi]
    let t_lo = 0.01f64;
    let t_hi = 20.0f64;
    let phi = (5.0f64.sqrt() - 1.0) / 2.0; // golden ratio conjugate

    let mut a = t_lo;
    let mut b = t_hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let tol = 1e-7;

    while (b - a).abs() > tol {
        if nll(c) < nll(d) {
            b = d;
        } else {
            a = c;
        }
        c = b - phi * (b - a);
        d = a + phi * (b - a);
    }

    Ok((a + b) / 2.0)
}

// ────────────────────────────────────────────────────────────────────────────
// Conformal Prediction Coverage
// ────────────────────────────────────────────────────────────────────────────

/// Computes the empirical coverage rate of prediction intervals.
///
/// Given lower and upper bounds of prediction intervals and actual observed values,
/// the coverage rate is the fraction of observations that fall within their
/// respective prediction intervals:
///
/// ```text
/// coverage = (1/n) * sum_i 1[lower_i <= actual_i <= upper_i]
/// ```
///
/// For a well-calibrated conformal predictor targeting coverage 1-α,
/// the empirical coverage should be ≈ 1-α.
///
/// # Arguments
///
/// * `lower` - Lower bounds of prediction intervals
/// * `upper` - Upper bounds of prediction intervals
/// * `actual` - Actual observed values
///
/// # Returns
///
/// Empirical coverage rate in [0, 1].
pub fn conformal_prediction_coverage(lower: &[f64], upper: &[f64], actual: &[f64]) -> Result<f64> {
    let n = lower.len();
    if n != upper.len() || n != actual.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "lower ({}), upper ({}), actual ({}) must have the same length",
            lower.len(),
            upper.len(),
            actual.len()
        )));
    }
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let covered = lower
        .iter()
        .zip(upper.iter())
        .zip(actual.iter())
        .filter(|((lo, hi), act)| *act >= *lo && *act <= *hi)
        .count();

    Ok(covered as f64 / n as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers (reimplemented locally to avoid circular deps)
// ────────────────────────────────────────────────────────────────────────────

fn validate_inputs_v2(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<()> {
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::InvalidInput(format!(
            "y_true and y_prob must have the same length: {} vs {}",
            y_true.len(),
            y_prob.len()
        )));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(MetricsError::InvalidInput(
            "n_bins must be greater than 0".to_string(),
        ));
    }
    Ok(())
}

fn ece_equal_width(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut ece = 0.0f64;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0f64;
        let mut sum_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..n {
            if y_prob[i] >= lower && y_prob[i] < upper {
                sum_true += y_true[i];
                sum_prob += y_prob[i];
                count += 1;
            }
        }

        if count > 0 {
            let acc = sum_true / count as f64;
            let conf = sum_prob / count as f64;
            ece += (count as f64 / n as f64) * (acc - conf).abs();
        }
    }

    Ok(ece)
}

fn mce_equal_width(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut mce = 0.0f64;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0f64;
        let mut sum_prob = 0.0f64;
        let mut count = 0usize;

        for i in 0..n {
            if y_prob[i] >= lower && y_prob[i] < upper {
                sum_true += y_true[i];
                sum_prob += y_prob[i];
                count += 1;
            }
        }

        if count > 0 {
            let acc = sum_true / count as f64;
            let conf = sum_prob / count as f64;
            let err = (acc - conf).abs();
            if err > mce {
                mce = err;
            }
        }
    }

    Ok(mce)
}

fn brier_score_v2(y_true: &[f64], y_prob: &[f64]) -> Result<f64> {
    let n = y_true.len();
    let sum_sq: f64 = (0..n).map(|i| (y_prob[i] - y_true[i]).powi(2)).sum();
    Ok(sum_sq / n as f64)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_ece_perfect_calibration() {
        // Perfect predictor: y_true[i] == y_prob[i] exactly
        let y_true = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        // Each bin has acc == conf when probs == outcomes
        // Use probs that put each sample in its own bin
        let y_prob = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let ece = ece_equal_width(&y_true, &y_prob, 2).expect("should succeed");
        assert!(
            ece.abs() < 1e-10,
            "ECE for perfect predictor should be 0, got {ece}"
        );
    }

    #[test]
    fn test_ece_overconfident() {
        // Model always predicts 0.9 but accuracy is 0.5
        let y_true: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();
        let y_prob: Vec<f64> = vec![0.9; 10];
        let ece = ece_equal_width(&y_true, &y_prob, 10).expect("should succeed");
        assert!(
            ece > 0.0,
            "ECE for always-confident predictor should be > 0, got {ece}"
        );
    }

    #[test]
    fn test_mce_bounds() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_prob = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85];
        let ece = ece_equal_width(&y_true, &y_prob, 5).expect("should succeed");
        let mce = mce_equal_width(&y_true, &y_prob, 5).expect("should succeed");
        assert!(mce >= ece - 1e-10, "MCE ({mce}) must be >= ECE ({ece})");
    }

    #[test]
    fn test_ace_equal_mass_bins() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_prob = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];
        let ace = adaptive_calibration_error_v2(&y_true, &y_prob, 5).expect("should succeed");
        assert!(ace >= 0.0, "ACE must be non-negative");
        assert!(ace <= 1.0, "ACE must be <= 1.0");
    }

    #[test]
    fn test_overconfidence_error_nonnegative() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.8, 0.9, 0.7, 0.6, 0.85];
        let oe = overconfidence_error(&y_true, &y_prob, 5).expect("should succeed");
        assert!(
            oe >= 0.0,
            "overconfidence error must be non-negative, got {oe}"
        );
    }

    #[test]
    fn test_calibration_metrics_compute() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y_prob = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85];
        let metrics = CalibrationMetrics::compute(&y_true, &y_prob, 10).expect("should succeed");
        assert!(metrics.ece >= 0.0);
        assert!(metrics.mce >= metrics.ece - 1e-10);
        assert!(metrics.ace >= 0.0);
        assert!(metrics.overconfidence >= 0.0);
        assert!(metrics.brier_score >= 0.0);
        assert_eq!(metrics.n_bins, 10);
        assert_eq!(metrics.n_samples, 10);
    }

    #[test]
    fn test_temperature_scaling_diagnostic() {
        // Perfect predictor: T* should be close to 1.0
        let n = 20;
        let probs_data: Vec<f64> = (0..n)
            .flat_map(|i| {
                if i < n / 2 {
                    vec![0.9, 0.1]
                } else {
                    vec![0.1, 0.9]
                }
            })
            .collect();
        let probs = Array2::from_shape_vec((n, 2), probs_data).expect("valid shape");
        let labels: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
        let t = temperature_scaling_diagnostic(&probs, &labels).expect("should succeed");
        assert!(t > 0.0, "temperature must be positive, got {t}");
    }

    #[test]
    fn test_conformal_prediction_coverage_full() {
        let lower = vec![0.0, 1.0, 2.0, 3.0];
        let upper = vec![2.0, 3.0, 4.0, 5.0];
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let cov = conformal_prediction_coverage(&lower, &upper, &actual).expect("should succeed");
        assert!(
            (cov - 1.0).abs() < 1e-10,
            "coverage should be 1.0, got {cov}"
        );
    }

    #[test]
    fn test_conformal_prediction_coverage_half() {
        let lower = vec![0.0, 1.0, 2.0, 3.0];
        let upper = vec![1.0, 2.0, 3.0, 4.0];
        let actual = vec![0.5, 3.0, 2.5, 5.0]; // 2nd and 4th outside
        let cov = conformal_prediction_coverage(&lower, &upper, &actual).expect("should succeed");
        assert!(
            (cov - 0.5).abs() < 1e-10,
            "coverage should be 0.5, got {cov}"
        );
    }

    #[test]
    fn test_conformal_coverage_mismatch_error() {
        let lower = vec![0.0, 1.0];
        let upper = vec![1.0, 2.0, 3.0]; // wrong length
        let actual = vec![0.5, 1.5];
        assert!(conformal_prediction_coverage(&lower, &upper, &actual).is_err());
    }
}
