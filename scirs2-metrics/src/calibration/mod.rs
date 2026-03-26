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
//! - **Brier Skill Score (BSS)**: Improvement over climatological reference
//! - **Log-Loss (Cross-Entropy)**: Logarithmic scoring rule
//! - **Reliability Diagram Data**: Bin-level data for plotting reliability diagrams
//! - **Adaptive Calibration Error**: ECE with equal-mass bins
//! - **Classwise ECE**: Per-class ECE for multi-class problems
//!
//! # Examples
//!
//! ```
//! use scirs2_metrics::calibration::{
//!     expected_calibration_error, maximum_calibration_error, brier_score,
//!     reliability_diagram_data, log_loss, cross_entropy,
//! };
//!
//! let y_true = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
//! let y_prob = vec![0.1, 0.2, 0.8, 0.7, 0.9, 0.3, 0.6, 0.4, 0.85, 0.95];
//!
//! let ece = expected_calibration_error(&y_true, &y_prob, 10).expect("Failed");
//! let mce = maximum_calibration_error(&y_true, &y_prob, 10).expect("Failed");
//! let brier = brier_score(&y_true, &y_prob).expect("Failed");
//! let diagram = reliability_diagram_data(&y_true, &y_prob, 10).expect("Failed");
//! let ll = log_loss(&y_true, &y_prob).expect("Failed");
//! let ce = cross_entropy(&y_true, &y_prob).expect("Failed");
//! ```

pub mod advanced;
mod metrics;
pub mod reliability;

pub use metrics::{
    adaptive_calibration_error, brier_score, brier_skill_score, classwise_ece,
    expected_calibration_error, maximum_calibration_error, reliability_diagram_data,
    ReliabilityBin, ReliabilityDiagram,
};

use crate::error::{MetricsError, Result};

/// Computes the log-loss (binary cross-entropy) for binary classification.
///
/// Log-loss = -(1/N) * sum_{i=1}^{N} [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
///
/// This is the negative log-likelihood of the predicted probabilities under
/// the true labels. Lower is better; a perfect model has log-loss = 0.
///
/// Predicted probabilities are clipped to [eps, 1 - eps] to avoid log(0).
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0)
/// * `y_prob` - Predicted probabilities in [0, 1]
///
/// # Returns
///
/// The log-loss value (non-negative). Lower is better.
///
/// # Examples
///
/// ```
/// use scirs2_metrics::calibration::log_loss;
///
/// let y_true = vec![1.0, 0.0, 1.0, 0.0];
/// let y_prob = vec![0.9, 0.1, 0.8, 0.2];
///
/// let ll = log_loss(&y_true, &y_prob).expect("Failed");
/// assert!(ll < 0.3); // Good predictions
/// ```
pub fn log_loss(y_true: &[f64], y_prob: &[f64]) -> Result<f64> {
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

    let eps = 1e-15;
    let n = y_true.len();
    let mut total = 0.0;

    for i in 0..n {
        let p = y_prob[i].clamp(eps, 1.0 - eps);
        let y = y_true[i];
        total += y * p.ln() + (1.0 - y) * (1.0 - p).ln();
    }

    Ok(-total / n as f64)
}

/// Computes the cross-entropy loss for binary classification.
///
/// This is identical to log-loss for binary classification:
///
/// CE = -(1/N) * sum_{i} [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
///
/// For multi-class, use `multiclass_cross_entropy` instead.
///
/// # Arguments
///
/// * `y_true` - Ground truth binary labels (0.0 or 1.0)
/// * `y_prob` - Predicted probabilities in [0, 1]
///
/// # Returns
///
/// The cross-entropy value (non-negative).
pub fn cross_entropy(y_true: &[f64], y_prob: &[f64]) -> Result<f64> {
    log_loss(y_true, y_prob)
}

/// Computes the multi-class cross-entropy loss.
///
/// CE = -(1/N) * sum_{i} sum_{c} y_{i,c} * log(p_{i,c})
///
/// # Arguments
///
/// * `y_true` - Ground truth class labels (integer-valued as f64, 0-indexed)
/// * `y_prob_matrix` - Predicted probability matrix, flattened row-major.
///   Row i is the probability distribution over classes for sample i.
///   Shape: [n_samples, n_classes].
/// * `n_classes` - Number of classes
///
/// # Returns
///
/// The multi-class cross-entropy value (non-negative).
///
/// # Examples
///
/// ```
/// use scirs2_metrics::calibration::multiclass_cross_entropy;
///
/// let y_true = vec![0.0, 1.0, 2.0];
/// // Perfect predictions
/// let y_prob = vec![
///     1.0, 0.0, 0.0,
///     0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0,
/// ];
/// let ce = multiclass_cross_entropy(&y_true, &y_prob, 3).expect("should succeed");
/// assert!(ce < 1e-10); // Perfect predictions have CE ~ 0
/// ```
pub fn multiclass_cross_entropy(
    y_true: &[f64],
    y_prob_matrix: &[f64],
    n_classes: usize,
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
    if y_prob_matrix.len() != n_samples * n_classes {
        return Err(MetricsError::InvalidInput(format!(
            "y_prob_matrix length {} does not match n_samples({}) * n_classes({})",
            y_prob_matrix.len(),
            n_samples,
            n_classes
        )));
    }

    let eps = 1e-15;
    let mut total = 0.0;

    for i in 0..n_samples {
        let class = y_true[i] as usize;
        if class >= n_classes {
            return Err(MetricsError::InvalidInput(format!(
                "class label {} exceeds n_classes={n_classes}",
                class
            )));
        }
        let p = y_prob_matrix[i * n_classes + class].clamp(eps, 1.0);
        total -= p.ln();
    }

    Ok(total / n_samples as f64)
}

/// Computes the multi-class log-loss with sample weights.
///
/// Weighted CE = -sum_{i} w_i * log(p_{i, y_i}) / sum_{i} w_i
///
/// # Arguments
///
/// * `y_true` - Ground truth class labels (integer-valued as f64)
/// * `y_prob_matrix` - Predicted probability matrix, flattened row-major
/// * `n_classes` - Number of classes
/// * `sample_weights` - Per-sample weights
///
/// # Returns
///
/// The weighted multi-class cross-entropy value.
pub fn weighted_cross_entropy(
    y_true: &[f64],
    y_prob_matrix: &[f64],
    n_classes: usize,
    sample_weights: &[f64],
) -> Result<f64> {
    let n_samples = y_true.len();
    if n_samples == 0 {
        return Err(MetricsError::InvalidInput(
            "inputs must not be empty".to_string(),
        ));
    }
    if sample_weights.len() != n_samples {
        return Err(MetricsError::InvalidInput(format!(
            "sample_weights length {} does not match n_samples {}",
            sample_weights.len(),
            n_samples
        )));
    }
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "n_classes must be > 0".to_string(),
        ));
    }
    if y_prob_matrix.len() != n_samples * n_classes {
        return Err(MetricsError::InvalidInput(format!(
            "y_prob_matrix length mismatch: {} vs {}",
            y_prob_matrix.len(),
            n_samples * n_classes
        )));
    }

    let eps = 1e-15;
    let mut total = 0.0;
    let mut weight_sum = 0.0;

    for i in 0..n_samples {
        let class = y_true[i] as usize;
        if class >= n_classes {
            return Err(MetricsError::InvalidInput(format!(
                "class label {} exceeds n_classes={n_classes}",
                class
            )));
        }
        let p = y_prob_matrix[i * n_classes + class].clamp(eps, 1.0);
        let w = sample_weights[i];
        total -= w * p.ln();
        weight_sum += w;
    }

    if weight_sum <= 0.0 {
        return Err(MetricsError::DivisionByZero);
    }

    Ok(total / weight_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_loss_perfect() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![1.0, 0.0, 1.0, 0.0];
        let ll = log_loss(&y_true, &y_prob).expect("should succeed");
        assert!(
            ll < 1e-10,
            "perfect predictions should have log-loss ~0, got {ll}"
        );
    }

    #[test]
    fn test_log_loss_worst() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.0, 1.0, 0.0, 1.0];
        let ll = log_loss(&y_true, &y_prob).expect("should succeed");
        // Should be very large (clipped to avoid infinity)
        assert!(
            ll > 10.0,
            "worst predictions should have very high log-loss, got {ll}"
        );
    }

    #[test]
    fn test_log_loss_random() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.5, 0.5, 0.5, 0.5];
        let ll = log_loss(&y_true, &y_prob).expect("should succeed");
        let expected = -(0.5_f64.ln()); // -ln(0.5) ~ 0.693
        assert!(
            (ll - expected).abs() < 1e-10,
            "random guessing should give log-loss=ln(2)={expected}, got {ll}"
        );
    }

    #[test]
    fn test_log_loss_good_predictions() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.9, 0.1, 0.8, 0.2];
        let ll = log_loss(&y_true, &y_prob).expect("should succeed");
        assert!(
            ll < 0.3,
            "good predictions should have low log-loss, got {ll}"
        );
    }

    #[test]
    fn test_log_loss_empty() {
        assert!(log_loss(&[], &[]).is_err());
    }

    #[test]
    fn test_log_loss_mismatched() {
        assert!(log_loss(&[1.0], &[0.5, 0.5]).is_err());
    }

    #[test]
    fn test_cross_entropy_equals_log_loss() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_prob = vec![0.9, 0.1, 0.8, 0.2];
        let ll = log_loss(&y_true, &y_prob).expect("should succeed");
        let ce = cross_entropy(&y_true, &y_prob).expect("should succeed");
        assert!(
            (ll - ce).abs() < 1e-15,
            "cross_entropy should equal log_loss"
        );
    }

    #[test]
    fn test_multiclass_cross_entropy_perfect() {
        let y_true = vec![0.0, 1.0, 2.0];
        let y_prob = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let ce = multiclass_cross_entropy(&y_true, &y_prob, 3).expect("should succeed");
        assert!(
            ce < 1e-10,
            "perfect predictions should have CE ~0, got {ce}"
        );
    }

    #[test]
    fn test_multiclass_cross_entropy_uniform() {
        let y_true = vec![0.0, 1.0, 2.0];
        let y_prob = vec![
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ];
        let ce = multiclass_cross_entropy(&y_true, &y_prob, 3).expect("should succeed");
        let expected = -(1.0_f64 / 3.0).ln(); // ln(3) ~ 1.0986
        assert!(
            (ce - expected).abs() < 1e-10,
            "uniform predictions should give CE=ln(3)={expected}, got {ce}"
        );
    }

    #[test]
    fn test_multiclass_cross_entropy_bad_label() {
        let y_true = vec![5.0]; // class 5, but only 3 classes
        let y_prob = vec![0.33, 0.33, 0.34];
        assert!(multiclass_cross_entropy(&y_true, &y_prob, 3).is_err());
    }

    #[test]
    fn test_multiclass_cross_entropy_empty() {
        assert!(multiclass_cross_entropy(&[], &[], 3).is_err());
    }

    #[test]
    fn test_weighted_cross_entropy_uniform_weights() {
        let y_true = vec![0.0, 1.0];
        let y_prob = vec![0.9, 0.1, 0.1, 0.9];
        let weights = vec![1.0, 1.0];
        let wce = weighted_cross_entropy(&y_true, &y_prob, 2, &weights).expect("should succeed");
        let ce = multiclass_cross_entropy(&y_true, &y_prob, 2).expect("should succeed");
        assert!(
            (wce - ce).abs() < 1e-10,
            "uniform weights should give same result as unweighted: {wce} vs {ce}"
        );
    }

    #[test]
    fn test_weighted_cross_entropy_mismatched() {
        let y_true = vec![0.0];
        let y_prob = vec![1.0, 0.0];
        let weights = vec![1.0, 1.0]; // wrong length
        assert!(weighted_cross_entropy(&y_true, &y_prob, 2, &weights).is_err());
    }
}
