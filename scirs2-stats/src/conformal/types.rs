//! Core types for Conformal Prediction
//!
//! Defines configuration structures, prediction set representations, and result
//! types used throughout the conformal prediction framework.

/// Nonconformity score type for conformal prediction
///
/// Each variant determines how nonconformity scores are computed during
/// calibration and inference.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ScoreType {
    /// |y - ŷ| for regression tasks
    AbsResidual,
    /// Conformal Quantile Regression (Romano et al. 2019):
    /// s_i = max(q̂_lo(x_i) - y_i, y_i - q̂_hi(x_i))
    QuantileRegression,
    /// |y - ŷ| / σ̂  where σ̂ is a local difficulty estimate
    NormalizedResidual,
    /// Highest predictive density score: s = 1 - p(y | x)
    Hpd,
    /// Regularized Adaptive Prediction Sets (RAPS, Angelopoulos 2021)
    /// for multi-class classification
    Raps,
}

impl Default for ScoreType {
    fn default() -> Self {
        ScoreType::AbsResidual
    }
}

/// Configuration for split/inductive conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalConfig {
    /// Significance level α ∈ (0, 1).  Coverage target is 1 − α.
    pub alpha: f64,
    /// Nonconformity score type to use.
    pub score_fn: ScoreType,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            score_fn: ScoreType::AbsResidual,
        }
    }
}

/// A prediction set for a single test point.
///
/// For regression tasks the set is the interval `[lower, upper]`.
/// For classification tasks the set is a collection of class indices.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionSet {
    /// Lower bound of the prediction interval (regression).
    pub lower: f64,
    /// Upper bound of the prediction interval (regression).
    pub upper: f64,
    /// Predicted class indices included in the set (classification).
    pub set: Vec<usize>,
}

impl PredictionSet {
    /// Create a regression interval prediction set.
    pub fn interval(lower: f64, upper: f64) -> Self {
        Self {
            lower,
            upper,
            set: Vec::new(),
        }
    }

    /// Create a classification prediction set.
    pub fn classification(set: Vec<usize>) -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            set,
        }
    }

    /// Return `true` if `value` is inside the regression interval.
    pub fn contains_value(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }

    /// Return `true` if `class` is in the classification set.
    pub fn contains_class(&self, class: usize) -> bool {
        self.set.contains(&class)
    }

    /// Width of the regression interval.  Returns `f64::INFINITY` for
    /// classification sets.
    pub fn width(&self) -> f64 {
        if self.set.is_empty() {
            self.upper - self.lower
        } else {
            f64::INFINITY
        }
    }
}

/// Aggregated results for a batch of conformal predictions.
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// One [`PredictionSet`] per test point.
    pub sets: Vec<PredictionSet>,
    /// Empirical coverage of the prediction sets over the test batch
    /// (fraction of sets that contain the true label).
    pub coverage: f64,
    /// Average width (regression) or average set size (classification).
    pub avg_width: f64,
}

/// Configuration specific to RAPS (Regularized Adaptive Prediction Sets).
#[derive(Debug, Clone)]
pub struct RapsConfig {
    /// Regularization threshold: classes ranked beyond `k_reg` incur a penalty.
    pub k_reg: usize,
    /// Regularization strength λ.  Larger values encourage smaller sets.
    pub lambda: f64,
}

impl Default for RapsConfig {
    fn default() -> Self {
        Self {
            k_reg: 5,
            lambda: 0.01,
        }
    }
}

/// High-level configuration for adaptive conformal prediction.
#[derive(Debug, Clone)]
pub struct CpConfig {
    /// Desired marginal coverage probability (e.g. 0.9 for 90% coverage).
    pub coverage_target: f64,
    /// When `true`, locally-adaptive scores (normalized / RAPS) are used.
    pub adaptive: bool,
}

impl Default for CpConfig {
    fn default() -> Self {
        Self {
            coverage_target: 0.9,
            adaptive: false,
        }
    }
}

/// Compute the empirical (1−α)-quantile with the finite-sample correction
/// (1 + 1/n) used in split conformal inference.
///
/// Returns `f64::INFINITY` if `scores` is empty.
pub fn conformal_quantile(scores: &[f64], alpha: f64) -> f64 {
    if scores.is_empty() {
        return f64::INFINITY;
    }
    let n = scores.len();
    // Level: ceil((n+1)(1-alpha)) / n  ≡  (1+1/n)(1-alpha) quantile
    let level = ((n + 1) as f64 * (1.0 - alpha) / n as f64).min(1.0);
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((level * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_config_default() {
        let cfg = ConformalConfig::default();
        assert!((cfg.alpha - 0.1).abs() < 1e-10);
        assert_eq!(cfg.score_fn, ScoreType::AbsResidual);
    }

    #[test]
    fn test_cp_config_default() {
        let cfg = CpConfig::default();
        assert!((cfg.coverage_target - 0.9).abs() < 1e-10);
        assert!(!cfg.adaptive);
    }

    #[test]
    fn test_raps_config_default() {
        let cfg = RapsConfig::default();
        assert_eq!(cfg.k_reg, 5);
        assert!(cfg.lambda > 0.0);
    }

    #[test]
    fn test_prediction_set_contains_value() {
        let ps = PredictionSet::interval(1.0, 3.0);
        assert!(ps.contains_value(2.0));
        assert!(!ps.contains_value(0.5));
        assert!((ps.width() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_set_classification() {
        let ps = PredictionSet::classification(vec![0, 2]);
        assert!(ps.contains_class(0));
        assert!(!ps.contains_class(1));
    }

    #[test]
    fn test_conformal_quantile_basic() {
        let scores: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let q = conformal_quantile(&scores, 0.1);
        // For n=10, level = (11*0.9/10) = 0.99 → index ceil(9.9)-1 = 9 → score = 10.0
        assert!(q <= 10.0);
    }

    #[test]
    fn test_conformal_quantile_empty() {
        let q = conformal_quantile(&[], 0.1);
        assert!(q.is_infinite());
    }
}
