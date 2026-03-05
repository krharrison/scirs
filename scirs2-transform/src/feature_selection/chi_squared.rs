//! Chi-squared test for categorical feature selection
//!
//! Evaluates the chi-squared statistic between each non-negative feature and the
//! class label. Features with higher chi-squared scores are more likely to be
//! independent of class and thus less useful for classification. This test requires
//! non-negative feature values (e.g., term frequencies, counts).

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// Result of a chi-squared test for a single feature
#[derive(Debug, Clone)]
pub struct Chi2Result {
    /// Chi-squared statistic
    pub chi2_statistic: f64,
    /// Approximate p-value from chi-squared distribution
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
}

/// Compute chi-squared statistics between each feature and the target class.
///
/// This test computes the chi-squared statistic between each non-negative
/// feature and class, as well as an approximate p-value. Features are
/// discretized into bins before computing the contingency table.
///
/// # Arguments
/// * `x` - Feature matrix, shape (n_samples, n_features). Values must be non-negative.
/// * `y` - Class labels, shape (n_samples,)
/// * `n_bins` - Number of bins to discretize continuous features
///
/// # Returns
/// A tuple of (chi2_statistics, p_values) as Array1<f64>
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::chi_squared::chi2_scores;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 2),
///     vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]);
///
/// let (chi2, pvals) = chi2_scores(&x, &y, 5).expect("should succeed");
/// assert_eq!(chi2.len(), 2);
/// ```
pub fn chi2_scores(
    x: &Array2<f64>,
    y: &Array1<f64>,
    n_bins: usize,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "X has {} samples but y has {} samples",
            n_samples,
            y.len()
        )));
    }

    if n_samples < 2 {
        return Err(TransformError::InvalidInput(
            "At least 2 samples required".to_string(),
        ));
    }

    // Verify non-negative features
    for i in 0..n_samples {
        for j in 0..n_features {
            if x[[i, j]] < 0.0 {
                return Err(TransformError::InvalidInput(format!(
                    "Chi-squared requires non-negative features, found {} at ({}, {})",
                    x[[i, j]],
                    i,
                    j
                )));
            }
        }
    }

    // Determine unique classes
    let mut class_set: Vec<i64> = Vec::new();
    for &val in y.iter() {
        let key = val.round() as i64;
        if !class_set.contains(&key) {
            class_set.push(key);
        }
    }
    class_set.sort();
    let n_classes = class_set.len();

    if n_classes < 2 {
        return Err(TransformError::InvalidInput(
            "At least 2 classes required for chi-squared test".to_string(),
        ));
    }

    let class_map: HashMap<i64, usize> =
        class_set.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let mut chi2_stats = Array1::zeros(n_features);
    let mut p_values = Array1::ones(n_features);

    let effective_bins = n_bins.max(2);

    for j in 0..n_features {
        // Find min/max for feature j
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for i in 0..n_samples {
            if x[[i, j]] < min_val {
                min_val = x[[i, j]];
            }
            if x[[i, j]] > max_val {
                max_val = x[[i, j]];
            }
        }

        let range = max_val - min_val;
        let n_actual_bins = if range < 1e-15 { 1 } else { effective_bins };
        let bin_width = if n_actual_bins > 1 {
            range / n_actual_bins as f64
        } else {
            1.0
        };

        // Build contingency table: rows = bins, cols = classes
        let mut observed = vec![vec![0usize; n_classes]; n_actual_bins];

        for i in 0..n_samples {
            let bin = if n_actual_bins == 1 {
                0
            } else {
                let b = ((x[[i, j]] - min_val) / bin_width).floor() as usize;
                b.min(n_actual_bins - 1)
            };
            let class_idx = class_map.get(&(y[i].round() as i64)).copied().unwrap_or(0);
            observed[bin][class_idx] += 1;
        }

        // Compute chi-squared statistic
        let result = chi2_from_contingency(&observed, n_samples);
        chi2_stats[j] = result.chi2_statistic;
        p_values[j] = result.p_value;
    }

    Ok((chi2_stats, p_values))
}

/// Chi-squared feature selector
///
/// Selects the k best features based on chi-squared scores.
#[derive(Debug, Clone)]
pub struct Chi2Selector {
    /// Number of features to select
    k: usize,
    /// Number of bins for discretization
    n_bins: usize,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// Chi-squared scores
    scores_: Option<Array1<f64>>,
    /// P-values
    p_values_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl Chi2Selector {
    /// Create a new chi-squared feature selector
    ///
    /// # Arguments
    /// * `k` - Number of top features to select
    pub fn new(k: usize) -> Self {
        Chi2Selector {
            k,
            n_bins: 10,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        }
    }

    /// Set number of bins for discretization
    pub fn with_n_bins(mut self, n_bins: usize) -> Self {
        self.n_bins = n_bins.max(2);
        self
    }

    /// Fit the selector
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.shape()[1];

        if self.k > n_features {
            return Err(TransformError::InvalidInput(format!(
                "k={} must be <= n_features={}",
                self.k, n_features
            )));
        }

        let (chi2_stats, p_values) = chi2_scores(x, y, self.n_bins)?;

        // Select top k features by chi2 statistic (higher is better)
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            chi2_stats[b]
                .partial_cmp(&chi2_stats[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected: Vec<usize> = indices.into_iter().take(self.k).collect();

        self.scores_ = Some(chi2_stats);
        self.p_values_ = Some(p_values);
        self.selected_features_ = Some(selected);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Chi2Selector has not been fitted".to_string())
        })?;

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_features_in = self.n_features_in_.unwrap_or(0);

        if n_features != n_features_in {
            return Err(TransformError::InvalidInput(format!(
                "x has {} features, expected {}",
                n_features, n_features_in
            )));
        }

        let mut transformed = Array2::zeros((n_samples, selected.len()));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            for i in 0..n_samples {
                transformed[[i, new_idx]] = x[[i, old_idx]];
            }
        }

        Ok(transformed)
    }

    /// Fit and transform
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn get_support(&self) -> Option<&Vec<usize>> {
        self.selected_features_.as_ref()
    }

    /// Get chi-squared scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }

    /// Get p-values
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values_.as_ref()
    }
}

/// Compute chi-squared statistic from a contingency table
fn chi2_from_contingency(observed: &[Vec<usize>], n_total: usize) -> Chi2Result {
    let n_rows = observed.len();
    if n_rows == 0 {
        return Chi2Result {
            chi2_statistic: 0.0,
            p_value: 1.0,
            degrees_of_freedom: 0,
        };
    }
    let n_cols = observed[0].len();

    // Compute row and column totals
    let mut row_totals = vec![0usize; n_rows];
    let mut col_totals = vec![0usize; n_cols];

    for (i, row) in observed.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            row_totals[i] += val;
            col_totals[j] += val;
        }
    }

    let n = n_total as f64;
    let mut chi2 = 0.0;

    // Count actual non-empty rows and columns for dof
    let non_empty_rows = row_totals.iter().filter(|&&t| t > 0).count();
    let non_empty_cols = col_totals.iter().filter(|&&t| t > 0).count();

    for (i, row) in observed.iter().enumerate() {
        for (j, &obs) in row.iter().enumerate() {
            if row_totals[i] > 0 && col_totals[j] > 0 {
                let expected = (row_totals[i] as f64 * col_totals[j] as f64) / n;
                if expected > 0.0 {
                    let diff = obs as f64 - expected;
                    chi2 += (diff * diff) / expected;
                }
            }
        }
    }

    let dof = if non_empty_rows > 1 && non_empty_cols > 1 {
        (non_empty_rows - 1) * (non_empty_cols - 1)
    } else {
        1
    };

    // Approximate p-value using the regularized incomplete gamma function
    // For chi-squared with k degrees of freedom:
    // p-value = 1 - CDF(chi2, k) = 1 - gamma_inc(k/2, chi2/2) / Gamma(k/2)
    let p_value = chi2_survival(chi2, dof);

    Chi2Result {
        chi2_statistic: chi2,
        p_value,
        degrees_of_freedom: dof,
    }
}

/// Compute survival function (1 - CDF) for chi-squared distribution
/// Uses a series expansion of the regularized incomplete gamma function
fn chi2_survival(x: f64, k: usize) -> f64 {
    if x <= 0.0 || k == 0 {
        return 1.0;
    }

    let a = k as f64 / 2.0;
    let half_x = x / 2.0;

    // Use the regularized upper incomplete gamma function Q(a, x)
    // Q(a, x) = 1 - P(a, x) where P is the regularized lower incomplete gamma
    upper_incomplete_gamma_reg(a, half_x)
}

/// Regularized upper incomplete gamma function Q(a, x) = Gamma(a, x) / Gamma(a)
/// Uses series expansion for small x, continued fraction for large x
fn upper_incomplete_gamma_reg(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // Use series expansion for P(a,x), then Q = 1 - P
        1.0 - lower_gamma_series(a, x)
    } else {
        // Use continued fraction for Q(a,x)
        upper_gamma_cf(a, x)
    }
}

/// Lower regularized incomplete gamma via series expansion
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    // P(a,x) = exp(-x) * x^a * sum / Gamma(a)
    let log_val = a * x.ln() - x - ln_gamma(a);
    if log_val < -500.0 {
        return 0.0;
    }
    log_val.exp() * sum
}

/// Upper regularized incomplete gamma via continued fraction (Lentz's method)
fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut b_n = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b_n;
    let mut h = d;

    for n in 1..max_iter {
        let a_n = -(n as f64) * (n as f64 - a);
        b_n += 2.0;
        d = a_n * d + b_n;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b_n + a_n / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    let log_val = a * x.ln() - x - ln_gamma(a);
    if log_val < -500.0 {
        return 1.0;
    }
    log_val.exp() * h
}

/// Lanczos approximation for log-gamma function
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let coefficients = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -5.395_239_384_953_e-6,
    ];

    let g = 5.0;
    let xx = x - 1.0;
    let mut t = 1.000_000_000_190_015_f64;
    for (i, &c) in coefficients.iter().enumerate() {
        t += c / (xx + i as f64 + 1.0);
    }

    let tmp = xx + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (xx + 0.5) * tmp.ln() - tmp + t.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_chi2_independent_features() {
        // Feature 0: perfectly separates classes; feature 1: uniform
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 5.0, 1.0, 5.0, 2.0, 5.0, 1.0, 5.0, 2.0,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (chi2, _pvals) = chi2_scores(&x, &y, 5).expect("chi2_scores");
        // Feature 0 should have much higher chi2 than feature 1
        assert!(
            chi2[0] > chi2[1],
            "chi2[0]={} > chi2[1]={}",
            chi2[0],
            chi2[1]
        );
    }

    #[test]
    fn test_chi2_selector() {
        let x = Array::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.5, 0.0, 1.1, 0.4, 0.0, 1.0, 0.6, 0.0, 1.1, 0.5, 0.0, 5.0, 0.4, 0.0, 5.1,
                0.5, 0.0, 5.0, 0.6, 0.0, 5.1, 0.4, 0.0,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let mut selector = Chi2Selector::new(1);
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(transformed.shape(), &[8, 1]);

        let selected = selector.get_support().expect("support");
        // Feature 0 should be selected (most discriminative)
        assert_eq!(selected, &[0]);
    }

    #[test]
    fn test_chi2_negative_features_error() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, -1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        assert!(chi2_scores(&x, &y, 5).is_err());
    }

    #[test]
    fn test_chi2_single_class_error() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);

        assert!(chi2_scores(&x, &y, 5).is_err());
    }

    #[test]
    fn test_chi2_p_values() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 10.0, 0.5, 10.0, 0.5, 10.0, 0.5, 10.0, 0.5,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        let (_chi2, pvals) = chi2_scores(&x, &y, 5).expect("chi2_scores");

        // Feature 0 discriminates classes -> small p-value
        // Feature 1 is constant -> large p-value
        assert!(pvals[0] < pvals[1], "p[0]={} < p[1]={}", pvals[0], pvals[1]);
    }

    #[test]
    fn test_chi2_selector_not_fitted() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let selector = Chi2Selector::new(1);
        assert!(selector.transform(&x).is_err());
    }

    #[test]
    fn test_ln_gamma_values() {
        // ln(Gamma(1)) = 0
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        // ln(Gamma(2)) = ln(1!) = 0
        assert!((ln_gamma(2.0)).abs() < 1e-10);
        // ln(Gamma(3)) = ln(2!) = ln(2) ~ 0.6931
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
    }
}
