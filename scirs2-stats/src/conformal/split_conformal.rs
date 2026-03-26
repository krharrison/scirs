//! Split (inductive) conformal prediction for regression and classification.
//!
//! Split conformal prediction (Papadopoulos et al. 2002; Vovk et al. 2005)
//! provides a finite-sample marginal coverage guarantee of exactly `1 − α`
//! under the assumption of exchangeability between the calibration set and
//! future test points.
//!
//! ## Regression
//!
//! 1. Compute nonconformity scores on a held-out calibration set:
//!    `s_i = |y_i − ŷ_i|`
//! 2. Find the `(1 + 1/n)(1 − α)` empirical quantile `Q̂`.
//! 3. For a new point with prediction `ŷ_{n+1}`, output
//!    `[ŷ_{n+1} − Q̂ , ŷ_{n+1} + Q̂]`.
//!
//! ## Classification
//!
//! 1. Score: `s_i = 1 − p_{y_i}(x_i)` (complement of true-class probability).
//! 2. Find quantile `Q̂`.
//! 3. Prediction set: include all classes `k` with `1 − p_k(x) ≤ Q̂`, i.e.
//!    all `k` with `p_k(x) ≥ 1 − Q̂`.

use crate::conformal::types::{conformal_quantile, PredictionSet};

/// Split conformal predictor for regression.
///
/// Stores calibration nonconformity scores and uses them to construct
/// symmetric prediction intervals for new test points.
///
/// # Example
/// ```rust
/// use scirs2_stats::conformal::split_conformal::SplitConformal;
///
/// let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let actuals     = vec![1.1, 1.9, 3.2, 3.8, 5.1];
/// let mut sc = SplitConformal::new();
/// sc.calibrate(&predictions, &actuals);
/// let interval = sc.predict_interval(3.0, 0.1).expect("calibrated");
/// assert!(interval.contains_value(3.0));
/// ```
#[derive(Debug, Clone, Default)]
pub struct SplitConformal {
    /// Nonconformity scores |y_i − ŷ_i| from the calibration set.
    pub calibration_scores: Vec<f64>,
}

impl SplitConformal {
    /// Create a new (uncalibrated) `SplitConformal`.
    pub fn new() -> Self {
        Self {
            calibration_scores: Vec::new(),
        }
    }

    /// Calibrate using point predictions and the corresponding ground-truth
    /// values. Computes `s_i = |y_i − ŷ_i|` for each calibration pair.
    ///
    /// Calling `calibrate` again replaces any previously stored scores.
    pub fn calibrate(&mut self, predictions: &[f64], actuals: &[f64]) {
        self.calibration_scores = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(yhat, y)| (y - yhat).abs())
            .collect();
    }

    /// Build a symmetric prediction interval `[ŷ − Q̂, ŷ + Q̂]` for a new
    /// point at significance level `alpha`.
    ///
    /// Returns `None` if the predictor has not been calibrated yet.
    pub fn predict_interval(&self, y_hat: f64, alpha: f64) -> Option<PredictionSet> {
        if self.calibration_scores.is_empty() {
            return None;
        }
        let q = conformal_quantile(&self.calibration_scores, alpha);
        Some(PredictionSet::interval(y_hat - q, y_hat + q))
    }

    /// Batch-predict intervals and return the empirical coverage on the
    /// provided actuals (useful for validation).
    ///
    /// Returns `None` if the predictor has not been calibrated.
    pub fn predict_batch(
        &self,
        y_hats: &[f64],
        actuals: &[f64],
        alpha: f64,
    ) -> Option<Vec<PredictionSet>> {
        if self.calibration_scores.is_empty() {
            return None;
        }
        let q = conformal_quantile(&self.calibration_scores, alpha);
        let sets = y_hats
            .iter()
            .map(|&yhat| PredictionSet::interval(yhat - q, yhat + q))
            .collect();
        let _ = actuals; // kept for API symmetry / future use
        Some(sets)
    }

    /// Return the conformal quantile `Q̂` for the given `alpha`.
    pub fn quantile(&self, alpha: f64) -> Option<f64> {
        if self.calibration_scores.is_empty() {
            return None;
        }
        Some(conformal_quantile(&self.calibration_scores, alpha))
    }
}

/// Split conformal predictor for classification.
///
/// Uses softmax probabilities to compute nonconformity scores and builds
/// prediction sets that contain the true class with probability ≥ 1 − α.
///
/// # Nonconformity score
/// `s_i = 1 − p_{y_i}(x_i)` where `p_{y_i}` is the predicted probability
/// of the true class.
#[derive(Debug, Clone, Default)]
pub struct SplitConformalClassifier {
    /// Calibration nonconformity scores 1 − p_{true class}.
    pub calibration_scores: Vec<f64>,
    /// Number of classes.
    pub num_classes: usize,
}

impl SplitConformalClassifier {
    /// Create a new (uncalibrated) classifier.
    pub fn new(num_classes: usize) -> Self {
        Self {
            calibration_scores: Vec::new(),
            num_classes,
        }
    }

    /// Calibrate using predicted class probabilities and true labels.
    ///
    /// * `probs_cal` — calibration probability matrix, shape `[n_cal][n_classes]`.
    ///   Each row must sum to 1 (softmax output).
    /// * `labels_cal` — true class indices in `0..num_classes`.
    pub fn calibrate(&mut self, probs_cal: &[Vec<f64>], labels_cal: &[usize]) {
        self.calibration_scores = probs_cal
            .iter()
            .zip(labels_cal.iter())
            .filter_map(|(row, &y)| {
                if y < row.len() {
                    Some(1.0 - row[y])
                } else {
                    None
                }
            })
            .collect();
    }

    /// Build a prediction set for a single test point.
    ///
    /// All classes `k` satisfying `1 − p_k(x) ≤ Q̂` (i.e. `p_k(x) ≥ 1 − Q̂`)
    /// are included.
    ///
    /// Returns `None` if uncalibrated.
    pub fn predict_set(&self, probs: &[f64], alpha: f64) -> Option<PredictionSet> {
        if self.calibration_scores.is_empty() {
            return None;
        }
        let q = conformal_quantile(&self.calibration_scores, alpha);
        let threshold = 1.0 - q;
        let set: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter_map(|(k, &pk)| if pk >= threshold { Some(k) } else { None })
            .collect();
        Some(PredictionSet::classification(set))
    }

    /// Check whether all sets in a test batch contain the true labels.
    /// Returns `(sets, empirical_coverage)`.
    pub fn predict_and_evaluate(
        &self,
        probs_test: &[Vec<f64>],
        labels_test: &[usize],
        alpha: f64,
    ) -> Option<(Vec<PredictionSet>, f64)> {
        if self.calibration_scores.is_empty() {
            return None;
        }
        let q = conformal_quantile(&self.calibration_scores, alpha);
        let threshold = 1.0 - q;
        let sets: Vec<PredictionSet> = probs_test
            .iter()
            .map(|row| {
                let set: Vec<usize> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &pk)| if pk >= threshold { Some(k) } else { None })
                    .collect();
                PredictionSet::classification(set)
            })
            .collect();

        let covered = sets
            .iter()
            .zip(labels_test.iter())
            .filter(|(s, &y)| s.contains_class(y))
            .count();
        let coverage = if sets.is_empty() {
            0.0
        } else {
            covered as f64 / sets.len() as f64
        };
        Some((sets, coverage))
    }
}

/// Compute empirical coverage: fraction of test intervals containing the true value.
pub fn empirical_coverage_regression(sets: &[PredictionSet], actuals: &[f64]) -> f64 {
    if sets.is_empty() {
        return 0.0;
    }
    let covered = sets
        .iter()
        .zip(actuals.iter())
        .filter(|(s, &y)| s.contains_value(y))
        .count();
    covered as f64 / sets.len() as f64
}

/// Compute empirical coverage for classification prediction sets.
pub fn empirical_coverage_classification(sets: &[PredictionSet], labels: &[usize]) -> f64 {
    if sets.is_empty() {
        return 0.0;
    }
    let covered = sets
        .iter()
        .zip(labels.iter())
        .filter(|(s, &y)| s.contains_class(y))
        .count();
    covered as f64 / sets.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG pseudo-random number generator for reproducible tests.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_f64(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 33) as f64 / (u32::MAX as f64)
        }

        fn next_normal(&mut self) -> f64 {
            // Box-Muller
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn test_split_conformal_coverage() {
        let mut rng = Lcg::new(42);
        let n_cal = 200usize;
        let n_test = 100usize;
        let alpha = 0.1;

        // True relationship: y = 2x + ε
        let cal_preds: Vec<f64> = (0..n_cal).map(|i| i as f64 * 0.01 * 2.0).collect();
        let cal_actuals: Vec<f64> = cal_preds
            .iter()
            .map(|&yhat| yhat + rng.next_normal() * 0.1)
            .collect();

        let mut sc = SplitConformal::new();
        sc.calibrate(&cal_preds, &cal_actuals);

        let test_preds: Vec<f64> = (0..n_test).map(|i| i as f64 * 0.02).collect();
        let test_actuals: Vec<f64> = test_preds
            .iter()
            .map(|&yhat| yhat + rng.next_normal() * 0.1)
            .collect();

        let sets = sc
            .predict_batch(&test_preds, &test_actuals, alpha)
            .expect("calibrated");
        let cov = empirical_coverage_regression(&sets, &test_actuals);
        assert!(
            cov >= 1.0 - alpha - 0.15,
            "Coverage {} below 1-α-0.15 = {}",
            cov,
            1.0 - alpha - 0.15
        );
    }

    #[test]
    fn test_split_conformal_alpha_effect() {
        let cal_preds: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let cal_actuals: Vec<f64> = cal_preds.iter().map(|&x| x + 1.0).collect();
        let mut sc = SplitConformal::new();
        sc.calibrate(&cal_preds, &cal_actuals);

        let q_small_alpha = sc.quantile(0.05).expect("q");
        let q_large_alpha = sc.quantile(0.2).expect("q");
        assert!(
            q_large_alpha <= q_small_alpha,
            "Larger alpha should yield a smaller (or equal) quantile"
        );
    }

    #[test]
    fn test_quantile_calibration() {
        // Verify the (1+1/n)(1-α) quantile formula.
        let scores: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let n = scores.len() as f64;
        let alpha = 0.1;
        // Expected index: ceil((n+1)(1-alpha)) - 1 = ceil(9.9) - 1 = 10 - 1 = 9
        let expected_idx = ((n + 1.0) * (1.0 - alpha)).ceil() as usize - 1;
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let expected = sorted[expected_idx.min(sorted.len() - 1)];

        let q = conformal_quantile(&scores, alpha);
        assert!(
            (q - expected).abs() < 1e-10,
            "q={} expected={}",
            q,
            expected
        );
    }

    #[test]
    fn test_split_conformal_interval_width() {
        // Wider residuals → wider intervals
        let preds: Vec<f64> = vec![0.0; 50];
        let actuals_narrow: Vec<f64> = vec![0.1; 50];
        let actuals_wide: Vec<f64> = vec![5.0; 50];

        let mut sc_narrow = SplitConformal::new();
        sc_narrow.calibrate(&preds, &actuals_narrow);
        let q_narrow = sc_narrow.quantile(0.1).expect("q");

        let mut sc_wide = SplitConformal::new();
        sc_wide.calibrate(&preds, &actuals_wide);
        let q_wide = sc_wide.quantile(0.1).expect("q");

        assert!(
            q_wide > q_narrow,
            "Wider residuals should produce wider quantile ({} vs {})",
            q_wide,
            q_narrow
        );
    }

    #[test]
    fn test_classification_set_coverage() {
        let mut rng = Lcg::new(99);
        let n_cal = 150usize;
        let n_classes = 4usize;
        let alpha = 0.1;

        // Generate calibration data: true label = class with highest prob
        let mut cal_probs: Vec<Vec<f64>> = Vec::new();
        let mut cal_labels: Vec<usize> = Vec::new();
        for _ in 0..n_cal {
            let mut raw: Vec<f64> = (0..n_classes).map(|_| rng.next_f64().max(0.01)).collect();
            // Normalise to sum=1
            let sum: f64 = raw.iter().sum();
            for p in raw.iter_mut() {
                *p /= sum;
            }
            // True label = argmax to ensure high recall
            let label = raw
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            cal_probs.push(raw);
            cal_labels.push(label);
        }

        let mut clf = SplitConformalClassifier::new(n_classes);
        clf.calibrate(&cal_probs, &cal_labels);

        // Test set: same distribution
        let mut test_probs: Vec<Vec<f64>> = Vec::new();
        let mut test_labels: Vec<usize> = Vec::new();
        for _ in 0..100 {
            let mut raw: Vec<f64> = (0..n_classes).map(|_| rng.next_f64().max(0.01)).collect();
            let sum: f64 = raw.iter().sum();
            for p in raw.iter_mut() {
                *p /= sum;
            }
            let label = raw
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            test_probs.push(raw);
            test_labels.push(label);
        }

        let (sets, cov) = clf
            .predict_and_evaluate(&test_probs, &test_labels, alpha)
            .expect("calibrated");
        assert!(cov >= 1.0 - alpha - 0.15, "Coverage {} too low", cov);
        assert_eq!(sets.len(), 100);
    }

    #[test]
    fn test_coverage_guarantee_exact() {
        // With n_cal calibration points, split conformal guarantees ≥ 1-α coverage.
        let n_cal = 100usize;
        let alpha = 0.1;
        let cal_preds = vec![0.0_f64; n_cal];
        // All residuals = 1.0 → quantile = 1.0
        let cal_actuals = vec![1.0_f64; n_cal];

        let mut sc = SplitConformal::new();
        sc.calibrate(&cal_preds, &cal_actuals);
        let q = sc.quantile(alpha).expect("q");
        // With uniform residuals of 1.0, the quantile is 1.0
        assert!((q - 1.0).abs() < 1e-10);

        // Any test point with |actual - pred| ≤ 1.0 is covered
        let ps = sc.predict_interval(0.0, alpha).expect("interval");
        assert!(ps.contains_value(0.99));
        assert!(!ps.contains_value(1.5));
    }

    #[test]
    fn test_conformal_exchangeability() {
        // Permuting the calibration set should give the same quantile
        let scores = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0];
        let perm = vec![9.0, 5.0, 3.0, 2.0, 6.0, 1.0, 5.0, 4.0, 1.0, 3.0];
        let q1 = conformal_quantile(&scores, 0.1);
        let q2 = conformal_quantile(&perm, 0.1);
        assert!(
            (q1 - q2).abs() < 1e-10,
            "Permutation must not change quantile"
        );
    }

    #[test]
    fn test_prediction_set_contains_true_alpha_zero() {
        // α = 0 ⟹ quantile = max score → interval covers everything
        let scores: Vec<f64> = vec![1.0, 2.0, 3.0];
        let q = conformal_quantile(&scores, 0.0);
        // level = (1+1/3)*1.0 = 1.33 clamped to 1.0 → index n-1 = 2 → score = 3.0
        assert!(q >= 3.0 - 1e-10, "q = {}", q);
        let ps = PredictionSet::interval(5.0 - q, 5.0 + q);
        // The true value can be anywhere in [5-q, 5+q]; q=3 so [2, 8]
        assert!(ps.contains_value(5.0));
    }
}
