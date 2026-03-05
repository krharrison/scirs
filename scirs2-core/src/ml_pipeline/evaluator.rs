//! Model evaluation metrics for the ML Pipeline API.
//!
//! This module provides standard evaluation functions for regression and
//! classification models:
//!
//! **Regression**
//! - [`r2_score`] — coefficient of determination R²
//! - [`mean_squared_error`] — MSE
//! - [`mean_absolute_error`] — MAE
//! - [`root_mean_squared_error`] — RMSE
//! - [`explained_variance_score`] — fraction of variance explained
//!
//! **Classification**
//! - [`accuracy_score`] — fraction of correctly classified samples
//! - [`precision_recall_f1`] — precision, recall, and F1 score (micro/macro/weighted)
//! - [`confusion_matrix`] — per-class confusion matrix
//!
//! All functions return sensible defaults (0.0 or NaN) for degenerate inputs
//! rather than panicking.

use ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// Averaging strategy for classification metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy for aggregating per-class metrics into a single scalar.
#[derive(Debug, Clone, PartialEq)]
pub enum Average {
    /// Compute metrics globally by counting total true positives, etc.
    Micro,
    /// Compute metrics for each label and take the unweighted mean.
    Macro,
    /// Compute metrics for each label weighted by support (number of true instances).
    Weighted,
}

// ─────────────────────────────────────────────────────────────────────────────
// Regression metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the coefficient of determination R².
///
/// R² = 1 - SS_res / SS_tot
///
/// Returns `f64::NAN` if `y_true` is constant (SS_tot = 0).
/// Returns `1.0` for a perfect prediction.
///
/// # Panics
///
/// Does not panic; returns `f64::NAN` for degenerate inputs.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::evaluator::r2_score;
/// use ndarray::Array1;
///
/// let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y_pred = y_true.clone();
/// assert!((r2_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
/// # }
/// ```
pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() {
        return f64::NAN;
    }
    let mean_y: f64 = y_true.sum() / n as f64;
    let ss_tot: f64 = y_true.iter().map(|&y| (y - mean_y).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();

    if ss_tot < f64::EPSILON {
        if ss_res < f64::EPSILON {
            1.0
        } else {
            f64::NEG_INFINITY
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Compute mean squared error (MSE).
///
/// MSE = Σ (y_true - y_pred)² / n
///
/// Returns `f64::NAN` if inputs are empty or have mismatched lengths.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::evaluator::mean_squared_error;
/// use ndarray::Array1;
///
/// let y_true = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// let y_pred = Array1::from_vec(vec![1.0, 2.0, 3.0]);
/// assert!((mean_squared_error(&y_true, &y_pred)).abs() < 1e-10);
/// # }
/// ```
pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() {
        return f64::NAN;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum::<f64>()
        / n as f64
}

/// Compute mean absolute error (MAE).
///
/// MAE = Σ |y_true - y_pred| / n
///
/// Returns `f64::NAN` if inputs are empty or mismatched.
pub fn mean_absolute_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() {
        return f64::NAN;
    }
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).abs())
        .sum::<f64>()
        / n as f64
}

/// Compute root mean squared error (RMSE).
///
/// RMSE = √MSE
pub fn root_mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    mean_squared_error(y_true, y_pred).sqrt()
}

/// Compute the explained variance score.
///
/// EVS = 1 - Var(y_true - y_pred) / Var(y_true)
///
/// Unlike R², this is invariant to additive bias in predictions.
/// Returns `f64::NAN` for degenerate inputs.
pub fn explained_variance_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() {
        return f64::NAN;
    }

    // Variance of residuals
    let residuals: Array1<f64> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| yt - yp)
        .collect::<Vec<_>>()
        .into();
    let res_mean = residuals.sum() / n as f64;
    let var_res: f64 = residuals.iter().map(|&r| (r - res_mean).powi(2)).sum::<f64>() / n as f64;

    // Variance of y_true
    let y_mean = y_true.sum() / n as f64;
    let var_y: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum::<f64>() / n as f64;

    if var_y < f64::EPSILON {
        if var_res < f64::EPSILON {
            1.0
        } else {
            f64::NAN
        }
    } else {
        1.0 - var_res / var_y
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classification metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Compute classification accuracy.
///
/// accuracy = number of correct predictions / total predictions
///
/// Returns 0.0 for empty inputs.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::evaluator::accuracy_score;
///
/// let y_true = &[0usize, 1, 2, 0, 1];
/// let y_pred = &[0usize, 0, 2, 0, 1];
/// assert!((accuracy_score(y_true, y_pred) - 0.8).abs() < 1e-10);
/// # }
/// ```
pub fn accuracy_score(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    correct as f64 / n as f64
}

/// Compute precision, recall, and F1 score with a given averaging strategy.
///
/// Returns `(precision, recall, f1)`.
///
/// For `Average::Micro`:
/// - precision = recall = f1 = accuracy (for multiclass)
///
/// For `Average::Macro`:
/// - unweighted mean of per-class precision, recall, and F1
///
/// For `Average::Weighted`:
/// - weighted mean by support (number of true instances per class)
///
/// Classes with zero support are excluded from the average.
/// Returns `(0.0, 0.0, 0.0)` for empty inputs.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::evaluator::{precision_recall_f1, Average};
///
/// let y_true = &[0usize, 0, 1, 1, 2, 2];
/// let y_pred = &[0usize, 1, 1, 1, 0, 2];
/// let (p, r, f1) = precision_recall_f1(y_true, y_pred, 3, Average::Macro);
/// assert!(f1 > 0.0 && f1 <= 1.0);
/// # }
/// ```
pub fn precision_recall_f1(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
    average: Average,
) -> (f64, f64, f64) {
    let n = y_true.len();
    if n == 0 || n != y_pred.len() || n_classes == 0 {
        return (0.0, 0.0, 0.0);
    }

    // Per-class counts: tp, fp, fn
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_ = vec![0usize; n_classes];

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t < n_classes && p < n_classes {
            if t == p {
                tp[t] += 1;
            } else {
                fp[p] += 1;
                fn_[t] += 1;
            }
        }
    }

    let support: Vec<usize> = (0..n_classes).map(|c| tp[c] + fn_[c]).collect();

    match average {
        Average::Micro => {
            let total_tp: usize = tp.iter().sum();
            let total_fp: usize = fp.iter().sum();
            let total_fn: usize = fn_.iter().sum();
            let p = total_tp as f64 / (total_tp + total_fp).max(1) as f64;
            let r = total_tp as f64 / (total_tp + total_fn).max(1) as f64;
            let f1 = f1_from_pr(p, r);
            (p, r, f1)
        }
        Average::Macro => {
            let active_classes: Vec<usize> = (0..n_classes)
                .filter(|&c| support[c] > 0)
                .collect();
            if active_classes.is_empty() {
                return (0.0, 0.0, 0.0);
            }
            let count = active_classes.len() as f64;
            let p_mean = active_classes
                .iter()
                .map(|&c| tp[c] as f64 / (tp[c] + fp[c]).max(1) as f64)
                .sum::<f64>()
                / count;
            let r_mean = active_classes
                .iter()
                .map(|&c| tp[c] as f64 / support[c].max(1) as f64)
                .sum::<f64>()
                / count;
            let f1_mean = active_classes
                .iter()
                .map(|&c| {
                    let pc = tp[c] as f64 / (tp[c] + fp[c]).max(1) as f64;
                    let rc = tp[c] as f64 / support[c].max(1) as f64;
                    f1_from_pr(pc, rc)
                })
                .sum::<f64>()
                / count;
            (p_mean, r_mean, f1_mean)
        }
        Average::Weighted => {
            let total_support: usize = support.iter().sum();
            if total_support == 0 {
                return (0.0, 0.0, 0.0);
            }
            let ts = total_support as f64;
            let p_w = (0..n_classes)
                .map(|c| {
                    let pc = tp[c] as f64 / (tp[c] + fp[c]).max(1) as f64;
                    pc * support[c] as f64
                })
                .sum::<f64>()
                / ts;
            let r_w = (0..n_classes)
                .map(|c| {
                    let rc = tp[c] as f64 / support[c].max(1) as f64;
                    rc * support[c] as f64
                })
                .sum::<f64>()
                / ts;
            let f1_w = (0..n_classes)
                .map(|c| {
                    let pc = tp[c] as f64 / (tp[c] + fp[c]).max(1) as f64;
                    let rc = tp[c] as f64 / support[c].max(1) as f64;
                    f1_from_pr(pc, rc) * support[c] as f64
                })
                .sum::<f64>()
                / ts;
            (p_w, r_w, f1_w)
        }
    }
}

/// Compute the confusion matrix.
///
/// Returns an `n_classes × n_classes` matrix where `matrix[[i, j]]` is the
/// number of samples with true label `i` predicted as label `j`.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::evaluator::confusion_matrix;
///
/// let y_true = &[0usize, 0, 1, 1];
/// let y_pred = &[0usize, 1, 0, 1];
/// let cm = confusion_matrix(y_true, y_pred, 2);
/// assert_eq!(cm[[0, 0]], 1); // true 0 predicted as 0
/// assert_eq!(cm[[0, 1]], 1); // true 0 predicted as 1
/// assert_eq!(cm[[1, 0]], 1); // true 1 predicted as 0
/// assert_eq!(cm[[1, 1]], 1); // true 1 predicted as 1
/// # }
/// ```
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> Array2<usize> {
    let mut cm = Array2::<usize>::zeros((n_classes, n_classes));
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t < n_classes && p < n_classes {
            cm[[t, p]] += 1;
        }
    }
    cm
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute F1 from precision and recall: 2*p*r / (p+r), or 0 if denominator is 0.
fn f1_from_pr(p: f64, r: f64) -> f64 {
    let denom = p + r;
    if denom < f64::EPSILON {
        0.0
    } else {
        2.0 * p * r / denom
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Regression metrics ────────────────────────────────────────────────────

    #[test]
    fn test_r2_perfect() {
        let y = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        assert!((r2_score(&y, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r2_mean_predictor() {
        // Predicting the mean gives R² = 0
        let y_true = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let mean = 2.0;
        let y_pred = Array1::from_vec(vec![mean, mean, mean]);
        assert!(r2_score(&y_true, &y_pred).abs() < 1e-10);
    }

    #[test]
    fn test_r2_negative_for_bad_predictor() {
        let y_true = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![3.0, 2.0, 1.0]); // reversed
        assert!(r2_score(&y_true, &y_pred) < 0.0);
    }

    #[test]
    fn test_r2_constant_target() {
        let y_true = Array1::from_vec(vec![5.0f64, 5.0, 5.0]);
        let y_pred = Array1::from_vec(vec![5.0f64, 5.0, 5.0]);
        assert!((r2_score(&y_true, &y_pred) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_zero() {
        let y = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        assert!(mean_squared_error(&y, &y).abs() < 1e-10);
    }

    #[test]
    fn test_mse_known_value() {
        let y_true = Array1::from_vec(vec![0.0f64, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        // MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
        let mse = mean_squared_error(&y_true, &y_pred);
        assert!((mse - 14.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mae_known_value() {
        let y_true = Array1::from_vec(vec![0.0f64, 0.0, 0.0]);
        let y_pred = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        // MAE = (1 + 2 + 3) / 3 = 2.0
        let mae = mean_absolute_error(&y_true, &y_pred);
        assert!((mae - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_equals_sqrt_mse() {
        let y_true = Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let y_pred = Array1::from_vec(vec![1.1f64, 1.9, 3.1, 3.9]);
        let mse = mean_squared_error(&y_true, &y_pred);
        let rmse = root_mean_squared_error(&y_true, &y_pred);
        assert!((rmse - mse.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_explained_variance_perfect() {
        let y = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        assert!((explained_variance_score(&y, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_explained_variance_with_bias() {
        // EVS is invariant to constant offset; R² is not
        let y_true = Array1::from_vec(vec![1.0f64, 2.0, 3.0]);
        let y_pred = Array1::from_vec(vec![2.0f64, 3.0, 4.0]); // y_true + 1 (perfect shape)
        let evs = explained_variance_score(&y_true, &y_pred);
        // residuals are constant (all 1.0), var = 0 → EVS = 1.0
        assert!((evs - 1.0).abs() < 1e-10, "EVS with constant offset: {evs}");
    }

    // ── Classification metrics ────────────────────────────────────────────────

    #[test]
    fn test_accuracy_perfect() {
        let y = &[0usize, 1, 2, 0];
        assert!((accuracy_score(y, y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_accuracy_partial() {
        let y_true = &[0usize, 0, 1, 1];
        let y_pred = &[0usize, 1, 0, 1]; // 2 correct / 4
        assert!((accuracy_score(y_true, y_pred) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_macro_perfect() {
        let y = &[0usize, 0, 1, 1, 2, 2];
        let (p, r, f1) = precision_recall_f1(y, y, 3, Average::Macro);
        assert!((p - 1.0).abs() < 1e-10);
        assert!((r - 1.0).abs() < 1e-10);
        assert!((f1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_micro() {
        let y_true = &[0usize, 0, 1, 1];
        let y_pred = &[0usize, 1, 0, 1];
        let (p, r, f1) = precision_recall_f1(y_true, y_pred, 2, Average::Micro);
        // 2 correct / 4 total → precision = recall = f1 = 0.5
        assert!((p - 0.5).abs() < 1e-10, "micro precision: {p}");
        assert!((r - 0.5).abs() < 1e-10, "micro recall: {r}");
        assert!((f1 - 0.5).abs() < 1e-10, "micro f1: {f1}");
    }

    #[test]
    fn test_precision_recall_f1_weighted() {
        let y_true = &[0usize, 0, 0, 1]; // imbalanced: 3 class-0, 1 class-1
        let y_pred = &[0usize, 0, 1, 1];
        let (p, r, f1) = precision_recall_f1(y_true, y_pred, 2, Average::Weighted);
        assert!(p >= 0.0 && p <= 1.0, "weighted p: {p}");
        assert!(r >= 0.0 && r <= 1.0, "weighted r: {r}");
        assert!(f1 >= 0.0 && f1 <= 1.0, "weighted f1: {f1}");
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let y_true = &[0usize, 0, 1, 1];
        let y_pred = &[0usize, 1, 0, 1];
        let cm = confusion_matrix(y_true, y_pred, 2);
        // [[1,1],[1,1]]
        assert_eq!(cm[[0, 0]], 1);
        assert_eq!(cm[[0, 1]], 1);
        assert_eq!(cm[[1, 0]], 1);
        assert_eq!(cm[[1, 1]], 1);
    }

    #[test]
    fn test_confusion_matrix_trace_equals_correct() {
        let y_true = &[0usize, 1, 2, 0, 1, 2];
        let y_pred = &[0usize, 1, 1, 0, 2, 2]; // 4 correct
        let cm = confusion_matrix(y_true, y_pred, 3);
        let trace: usize = (0..3).map(|i| cm[[i, i]]).sum();
        assert_eq!(trace, 4);
    }

    #[test]
    fn test_confusion_matrix_empty_inputs() {
        let cm = confusion_matrix(&[], &[], 3);
        assert_eq!(cm.sum(), 0);
    }

    #[test]
    fn test_r2_empty_inputs() {
        let empty = Array1::<f64>::zeros(0);
        assert!(r2_score(&empty, &empty).is_nan());
    }

    #[test]
    fn test_mse_empty_inputs() {
        let empty = Array1::<f64>::zeros(0);
        assert!(mean_squared_error(&empty, &empty).is_nan());
    }

    #[test]
    fn test_f1_from_pr_zero_denominator() {
        assert_eq!(f1_from_pr(0.0, 0.0), 0.0);
    }
}
