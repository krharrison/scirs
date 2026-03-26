//! Calibration Metrics
//!
//! Metrics for evaluating how well predicted probabilities match observed
//! frequencies. A well-calibrated model predicts P(Y=1|p) = p for all p.
//!
//! # Metrics
//!
//! - **Expected Calibration Error (ECE)**: Weighted average of bin-level calibration errors
//! - **Maximum Calibration Error (MCE)**: Worst-case bin-level calibration error
//! - **Brier Score**: Mean squared difference between predicted probabilities and outcomes
//! - **Reliability Diagram Data**: Bin-level data for plotting reliability diagrams
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::calibration::{
//!     expected_calibration_error, maximum_calibration_error, brier_score,
//!     reliability_diagram_data,
//! };
//!
//! // Ground truth binary labels
//! let y_true = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
//! // Predicted probabilities
//! let y_prob = vec![0.1, 0.2, 0.8, 0.7, 0.9, 0.3, 0.6, 0.4, 0.85, 0.95];
//!
//! let ece = expected_calibration_error(&y_true, &y_prob, 10).expect("Failed");
//! let mce = maximum_calibration_error(&y_true, &y_prob, 10).expect("Failed");
//! let brier = brier_score(&y_true, &y_prob).expect("Failed");
//! let diagram = reliability_diagram_data(&y_true, &y_prob, 10).expect("Failed");
//! ```

use crate::error::{MetricsError, Result};

/// Data for a single bin in a reliability diagram.
#[derive(Debug, Clone)]
pub struct ReliabilityBin {
    /// Lower boundary of this bin
    pub bin_lower: f64,
    /// Upper boundary of this bin
    pub bin_upper: f64,
    /// Mean predicted probability in this bin
    pub mean_predicted: f64,
    /// Fraction of positive outcomes in this bin (observed frequency)
    pub fraction_positive: f64,
    /// Number of samples in this bin
    pub count: usize,
    /// Calibration error for this bin: |fraction_positive - mean_predicted|
    pub calibration_error: f64,
}

/// Complete reliability diagram data.
#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    /// Per-bin data
    pub bins: Vec<ReliabilityBin>,
    /// Expected Calibration Error
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Total number of samples
    pub total_samples: usize,
}

/// Computes Expected Calibration Error (ECE).
///
/// ECE measures the average difference between predicted probabilities
/// and actual frequencies, weighted by the number of samples in each bin.
///
/// ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|
///
/// where acc(b) is the fraction of positives in bin b,
/// conf(b) is the mean predicted probability in bin b,
/// n_b is the number of samples in bin b, and N is the total.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0).
/// * `y_prob` - Predicted probabilities in [0, 1].
/// * `n_bins` - Number of equally-spaced bins to use.
///
/// # Returns
///
/// The ECE value in [0, 1]. Lower is better.
///
/// # Errors
///
/// Returns error if inputs have different lengths, are empty, or n_bins is 0.
pub fn expected_calibration_error(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    validate_inputs(y_true, y_prob, n_bins)?;

    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut ece = 0.0;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0;
        let mut sum_prob = 0.0;
        let mut count = 0;

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

/// Computes Maximum Calibration Error (MCE).
///
/// MCE is the maximum calibration error across all bins.
///
/// MCE = max_{b=1}^{B} |acc(b) - conf(b)|
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0).
/// * `y_prob` - Predicted probabilities in [0, 1].
/// * `n_bins` - Number of equally-spaced bins to use.
///
/// # Returns
///
/// The MCE value in [0, 1]. Lower is better.
pub fn maximum_calibration_error(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    validate_inputs(y_true, y_prob, n_bins)?;

    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut mce = 0.0_f64;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0;
        let mut sum_prob = 0.0;
        let mut count = 0;

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
            let bin_error = (acc - conf).abs();
            if bin_error > mce {
                mce = bin_error;
            }
        }
    }

    Ok(mce)
}

/// Computes the Brier score.
///
/// The Brier score is the mean squared difference between predicted
/// probabilities and the actual binary outcomes. It decomposes into
/// calibration (reliability) and refinement (resolution) components.
///
/// Brier = (1/N) * sum_{i=1}^{N} (p_i - y_i)^2
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0).
/// * `y_prob` - Predicted probabilities in [0, 1].
///
/// # Returns
///
/// The Brier score in [0, 1]. Lower is better.
/// A perfect model has Brier = 0, random guessing has Brier ~ 0.25.
///
/// # Examples
///
/// ```
/// use scirs2_metrics::calibration::brier_score;
///
/// let y_true = vec![1.0, 0.0, 1.0, 0.0];
/// let y_prob = vec![0.9, 0.1, 0.8, 0.2];
///
/// let brier = brier_score(&y_true, &y_prob).expect("Failed");
/// assert!(brier < 0.05); // Good predictions
/// ```
pub fn brier_score(y_true: &[f64], y_prob: &[f64]) -> Result<f64> {
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

    let n = y_true.len();
    let mut sum_sq = 0.0;

    for i in 0..n {
        let diff = y_prob[i] - y_true[i];
        sum_sq += diff * diff;
    }

    Ok(sum_sq / n as f64)
}

/// Computes the Brier Skill Score (BSS).
///
/// BSS compares the Brier score of the model to the Brier score of a
/// reference model (usually climatological frequency).
///
/// BSS = 1 - Brier(model) / Brier(reference)
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels.
/// * `y_prob` - Predicted probabilities.
///
/// # Returns
///
/// The BSS value. 1.0 = perfect, 0.0 = same as climatology, negative = worse.
pub fn brier_skill_score(y_true: &[f64], y_prob: &[f64]) -> Result<f64> {
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_prob must have the same length".to_string(),
        ));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }

    let model_brier = brier_score(y_true, y_prob)?;

    // Reference Brier score: predict the base rate for every sample
    let base_rate: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ref_prob = vec![base_rate; y_true.len()];
    let ref_brier = brier_score(y_true, &ref_prob)?;

    if ref_brier <= 0.0 {
        // All samples in same class, Brier reference is 0
        if model_brier <= 0.0 {
            return Ok(1.0); // both perfect
        }
        return Ok(f64::NEG_INFINITY);
    }

    Ok(1.0 - model_brier / ref_brier)
}

/// Generates reliability diagram data for plotting.
///
/// Returns bin-level statistics for constructing a reliability diagram
/// (calibration curve).
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0).
/// * `y_prob` - Predicted probabilities in [0, 1].
/// * `n_bins` - Number of equally-spaced bins to use.
///
/// # Returns
///
/// A [`ReliabilityDiagram`] containing per-bin statistics plus ECE and MCE.
pub fn reliability_diagram_data(
    y_true: &[f64],
    y_prob: &[f64],
    n_bins: usize,
) -> Result<ReliabilityDiagram> {
    validate_inputs(y_true, y_prob, n_bins)?;

    let n = y_true.len();
    let bin_width = 1.0 / n_bins as f64;
    let mut bins = Vec::with_capacity(n_bins);
    let mut ece = 0.0;
    let mut mce = 0.0_f64;

    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = if b == n_bins - 1 {
            1.0 + f64::EPSILON
        } else {
            (b + 1) as f64 * bin_width
        };

        let mut sum_true = 0.0;
        let mut sum_prob = 0.0;
        let mut count = 0;

        for i in 0..n {
            if y_prob[i] >= lower && y_prob[i] < upper {
                sum_true += y_true[i];
                sum_prob += y_prob[i];
                count += 1;
            }
        }

        let (mean_predicted, fraction_positive, calibration_error) = if count > 0 {
            let mp = sum_prob / count as f64;
            let fp = sum_true / count as f64;
            let ce = (fp - mp).abs();
            (mp, fp, ce)
        } else {
            (0.0, 0.0, 0.0)
        };

        if count > 0 {
            ece += (count as f64 / n as f64) * calibration_error;
            if calibration_error > mce {
                mce = calibration_error;
            }
        }

        bins.push(ReliabilityBin {
            bin_lower: lower,
            bin_upper: if b == n_bins - 1 { 1.0 } else { upper },
            mean_predicted,
            fraction_positive,
            count,
            calibration_error,
        });
    }

    Ok(ReliabilityDiagram {
        bins,
        ece,
        mce,
        total_samples: n,
    })
}

/// Computes ECE using adaptive (equal-mass) binning instead of equal-width.
///
/// Equal-mass bins each contain approximately the same number of samples,
/// which can be more reliable than equal-width bins when probability
/// predictions are not uniformly distributed.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels.
/// * `y_prob` - Predicted probabilities in [0, 1].
/// * `n_bins` - Number of bins. Each bin will contain approximately N/n_bins samples.
///
/// # Returns
///
/// The adaptive ECE value in [0, 1].
pub fn adaptive_calibration_error(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<f64> {
    if y_true.len() != y_prob.len() {
        return Err(MetricsError::InvalidInput(
            "y_true and y_prob must have the same length".to_string(),
        ));
    }
    if y_true.is_empty() {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }

    let n = y_true.len();

    // Sort samples by predicted probability
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        y_prob[a]
            .partial_cmp(&y_prob[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bin_size = n / n_bins;
    let remainder = n % n_bins;
    let mut ece = 0.0;
    let mut offset = 0;

    for b in 0..n_bins {
        // Distribute remainder evenly
        let current_bin_size = if b < remainder {
            bin_size + 1
        } else {
            bin_size
        };

        if current_bin_size == 0 {
            continue;
        }

        let bin_indices = &indices[offset..offset + current_bin_size];

        let mut sum_true = 0.0;
        let mut sum_prob = 0.0;

        for &idx in bin_indices {
            sum_true += y_true[idx];
            sum_prob += y_prob[idx];
        }

        let acc = sum_true / current_bin_size as f64;
        let conf = sum_prob / current_bin_size as f64;
        ece += (current_bin_size as f64 / n as f64) * (acc - conf).abs();

        offset += current_bin_size;
    }

    Ok(ece)
}

/// Computes classwise ECE for multi-class classification.
///
/// For K classes, computes the ECE for each class separately (one-vs-rest)
/// and returns the mean.
///
/// # Arguments
///
/// * `y_true` - Ground truth class labels (integer-valued as f64).
/// * `y_prob_matrix` - Predicted probability matrix, flattened row-major.
///   Row i is the probability distribution over classes for sample i.
///   Shape: [n_samples, n_classes].
/// * `n_classes` - Number of classes.
/// * `n_bins` - Number of bins for each class's calibration.
///
/// # Returns
///
/// The classwise ECE, averaged over all classes.
pub fn classwise_ece(
    y_true: &[f64],
    y_prob_matrix: &[f64],
    n_classes: usize,
    n_bins: usize,
) -> Result<f64> {
    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "n_classes must be > 0".to_string(),
        ));
    }
    if n_bins == 0 {
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }
    if y_prob_matrix.len() != n_samples * n_classes {
        return Err(MetricsError::InvalidInput(format!(
            "y_prob_matrix length {} does not match n_samples({}) * n_classes({})",
            y_prob_matrix.len(),
            n_samples,
            n_classes
        )));
    }

    let mut total_ece = 0.0;

    for class in 0..n_classes {
        // One-vs-rest: binary labels for this class
        let y_binary: Vec<f64> = y_true
            .iter()
            .map(|&y| if (y as usize) == class { 1.0 } else { 0.0 })
            .collect();

        // Predicted probabilities for this class
        let y_class_prob: Vec<f64> = (0..n_samples)
            .map(|i| y_prob_matrix[i * n_classes + class])
            .collect();

        let class_ece = expected_calibration_error(&y_binary, &y_class_prob, n_bins)?;
        total_ece += class_ece;
    }

    Ok(total_ece / n_classes as f64)
}

/// Validates common inputs for calibration functions.
fn validate_inputs(y_true: &[f64], y_prob: &[f64], n_bins: usize) -> Result<()> {
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
        return Err(MetricsError::InvalidInput("n_bins must be > 0".to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ECE tests ----

    #[test]
    fn test_ece_perfectly_calibrated() {
        // Each bin has exactly matching predicted prob and outcome rate
        let y_true = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y_prob = vec![0.05, 0.05, 0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95];
        let ece = expected_calibration_error(&y_true, &y_prob, 10).expect("should succeed");
        // Bin [0.0, 0.1): 5 samples, acc=0.0, conf=0.05 -> error=0.05
        // Bin [0.9, 1.0]: 5 samples, acc=1.0, conf=0.95 -> error=0.05
        // ECE = 0.5*0.05 + 0.5*0.05 = 0.05
        assert!((ece - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_ece_poorly_calibrated() {
        // Predict high probability for negative samples
        let y_true = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let y_prob = vec![0.9, 0.9, 0.9, 0.9, 0.9];
        let ece = expected_calibration_error(&y_true, &y_prob, 10).expect("should succeed");
        // Bin [0.9, 1.0]: 5 samples, acc=0.0, conf=0.9 -> error=0.9
        // ECE = 1.0 * 0.9 = 0.9
        assert!((ece - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_ece_empty() {
        let y_true: Vec<f64> = vec![];
        let y_prob: Vec<f64> = vec![];
        assert!(expected_calibration_error(&y_true, &y_prob, 10).is_err());
    }

    #[test]
    fn test_ece_mismatched_length() {
        let y_true = vec![0.0, 1.0];
        let y_prob = vec![0.5];
        assert!(expected_calibration_error(&y_true, &y_prob, 10).is_err());
    }

    #[test]
    fn test_ece_zero_bins() {
        let y_true = vec![0.0, 1.0];
        let y_prob = vec![0.5, 0.5];
        assert!(expected_calibration_error(&y_true, &y_prob, 0).is_err());
    }

    // ---- MCE tests ----

    #[test]
    fn test_mce_perfectly_calibrated() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_prob = vec![0.05, 0.05, 0.95, 0.95];
        let mce = maximum_calibration_error(&y_true, &y_prob, 10).expect("should succeed");
        // Max error across bins = 0.05
        assert!((mce - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_mce_poorly_calibrated() {
        let y_true = vec![0.0, 0.0, 0.0, 0.0];
        let y_prob = vec![0.9, 0.9, 0.9, 0.9];
        let mce = maximum_calibration_error(&y_true, &y_prob, 10).expect("should succeed");
        assert!((mce - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_mce_multiple_bins() {
        // Two non-empty bins with different errors
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_prob = vec![0.1, 0.2, 0.8, 0.9];
        let mce = maximum_calibration_error(&y_true, &y_prob, 10).expect("should succeed");
        assert!(mce > 0.0 && mce <= 1.0);
    }

    #[test]
    fn test_mce_empty() {
        assert!(maximum_calibration_error(&[], &[], 10).is_err());
    }

    // ---- Brier score tests ----

    #[test]
    fn test_brier_perfect() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![1.0, 0.0, 1.0, 0.0];
        let brier = brier_score(&y_true, &y_prob).expect("should succeed");
        assert!((brier - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brier_worst() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.0, 1.0, 0.0, 1.0];
        let brier = brier_score(&y_true, &y_prob).expect("should succeed");
        assert!((brier - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brier_random() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.5, 0.5, 0.5, 0.5];
        let brier = brier_score(&y_true, &y_prob).expect("should succeed");
        assert!((brier - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_brier_good_predictions() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.9, 0.1, 0.8, 0.2];
        let brier = brier_score(&y_true, &y_prob).expect("should succeed");
        assert!(brier < 0.05);
    }

    #[test]
    fn test_brier_empty() {
        assert!(brier_score(&[], &[]).is_err());
    }

    #[test]
    fn test_brier_mismatched() {
        assert!(brier_score(&[1.0], &[0.5, 0.5]).is_err());
    }

    // ---- Brier Skill Score tests ----

    #[test]
    fn test_bss_perfect() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![1.0, 0.0, 1.0, 0.0];
        let bss = brier_skill_score(&y_true, &y_prob).expect("should succeed");
        assert!((bss - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bss_climatology() {
        // Predicting base rate for every sample = same as reference
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.5, 0.5, 0.5, 0.5]; // base rate = 0.5
        let bss = brier_skill_score(&y_true, &y_prob).expect("should succeed");
        assert!((bss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_bss_better_than_reference() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.9, 0.1, 0.8, 0.2];
        let bss = brier_skill_score(&y_true, &y_prob).expect("should succeed");
        assert!(bss > 0.0);
    }

    #[test]
    fn test_bss_worse_than_reference() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.1, 0.9, 0.2, 0.8]; // inverted predictions
        let bss = brier_skill_score(&y_true, &y_prob).expect("should succeed");
        assert!(bss < 0.0);
    }

    // ---- Reliability diagram tests ----

    #[test]
    fn test_reliability_diagram_structure() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let y_prob = vec![0.1, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4, 0.85, 0.95, 0.9];
        let diagram = reliability_diagram_data(&y_true, &y_prob, 5).expect("should succeed");

        assert_eq!(diagram.bins.len(), 5);
        assert_eq!(diagram.total_samples, 10);
        assert!(diagram.ece >= 0.0 && diagram.ece <= 1.0);
        assert!(diagram.mce >= 0.0 && diagram.mce <= 1.0);
        assert!(diagram.mce >= diagram.ece);
    }

    #[test]
    fn test_reliability_diagram_bin_boundaries() {
        let y_true = vec![0.0, 1.0];
        let y_prob = vec![0.1, 0.9];
        let diagram = reliability_diagram_data(&y_true, &y_prob, 4).expect("should succeed");

        assert!((diagram.bins[0].bin_lower - 0.0).abs() < 1e-10);
        assert!((diagram.bins[3].bin_upper - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reliability_diagram_counts() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_prob = vec![0.1, 0.2, 0.8, 0.9];
        let diagram = reliability_diagram_data(&y_true, &y_prob, 2).expect("should succeed");

        // Bin [0, 0.5): samples at 0.1, 0.2
        // Bin [0.5, 1.0]: samples at 0.8, 0.9
        assert_eq!(diagram.bins[0].count, 2);
        assert_eq!(diagram.bins[1].count, 2);
    }

    #[test]
    fn test_reliability_diagram_empty() {
        assert!(reliability_diagram_data(&[], &[], 5).is_err());
    }

    // ---- Adaptive ECE tests ----

    #[test]
    fn test_adaptive_ece_perfect() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_prob = vec![0.0, 0.0, 1.0, 1.0];
        let ece = adaptive_calibration_error(&y_true, &y_prob, 2).expect("should succeed");
        assert!((ece - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_ece_poor() {
        let y_true = vec![0.0, 0.0, 0.0, 0.0];
        let y_prob = vec![0.8, 0.85, 0.9, 0.95];
        let ece = adaptive_calibration_error(&y_true, &y_prob, 2).expect("should succeed");
        assert!(ece > 0.7);
    }

    #[test]
    fn test_adaptive_ece_empty() {
        assert!(adaptive_calibration_error(&[], &[], 2).is_err());
    }

    #[test]
    fn test_adaptive_ece_single_bin() {
        let y_true = vec![0.0, 1.0, 0.0, 1.0];
        let y_prob = vec![0.3, 0.7, 0.4, 0.6];
        let ece = adaptive_calibration_error(&y_true, &y_prob, 1).expect("should succeed");
        assert!(ece >= 0.0);
    }

    // ---- Classwise ECE tests ----

    #[test]
    fn test_classwise_ece_perfect() {
        // 4 samples, 2 classes
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        // Perfect predictions: [1,0] for class 0, [0,1] for class 1
        let y_prob_matrix = vec![
            1.0, 0.0, // sample 0: class 0
            1.0, 0.0, // sample 1: class 0
            0.0, 1.0, // sample 2: class 1
            0.0, 1.0, // sample 3: class 1
        ];
        let ece = classwise_ece(&y_true, &y_prob_matrix, 2, 10).expect("should succeed");
        assert!((ece - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_classwise_ece_moderate() {
        let y_true = vec![0.0, 0.0, 1.0, 1.0];
        let y_prob_matrix = vec![0.8, 0.2, 0.7, 0.3, 0.3, 0.7, 0.2, 0.8];
        let ece = classwise_ece(&y_true, &y_prob_matrix, 2, 5).expect("should succeed");
        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_classwise_ece_invalid_matrix() {
        let y_true = vec![0.0, 1.0];
        let y_prob_matrix = vec![0.5]; // wrong size
        assert!(classwise_ece(&y_true, &y_prob_matrix, 2, 5).is_err());
    }

    #[test]
    fn test_classwise_ece_empty() {
        assert!(classwise_ece(&[], &[], 2, 5).is_err());
    }
}
