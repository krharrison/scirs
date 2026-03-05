//! ANOVA F-test for feature ranking
//!
//! Computes the one-way ANOVA F-statistic for each feature against the target
//! class labels. Features with higher F-values have greater between-class variance
//! relative to within-class variance, indicating stronger discriminative power.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::{Result, TransformError};

/// Result of an F-test for a single feature
#[derive(Debug, Clone)]
pub struct FTestResult {
    /// F-statistic
    pub f_statistic: f64,
    /// P-value from the F-distribution
    pub p_value: f64,
    /// Between-group degrees of freedom
    pub df_between: usize,
    /// Within-group degrees of freedom
    pub df_within: usize,
}

/// Compute ANOVA F-test statistics for each feature against the class label.
///
/// The F-statistic measures the ratio of between-class variance to within-class
/// variance. Higher values indicate more discriminative features.
///
/// # Arguments
/// * `x` - Feature matrix, shape (n_samples, n_features)
/// * `y` - Class labels, shape (n_samples,)
///
/// # Returns
/// A tuple of (f_statistics, p_values) as Array1<f64>
///
/// # Examples
///
/// ```
/// use scirs2_transform::feature_selection::f_test::f_classif;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let x = Array2::from_shape_vec(
///     (6, 2),
///     vec![1.0, 5.0, 2.0, 5.1, 1.5, 5.0,
///          8.0, 5.0, 9.0, 5.1, 8.5, 5.0],
/// ).expect("should succeed");
/// let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
///
/// let (f_stats, p_vals) = f_classif(&x, &y).expect("should succeed");
/// // Feature 0 should have much higher F-statistic
/// assert!(f_stats[0] > f_stats[1]);
/// ```
pub fn f_classif(x: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "X has {} samples but y has {} samples",
            n_samples,
            y.len()
        )));
    }

    if n_samples < 3 {
        return Err(TransformError::InvalidInput(
            "At least 3 samples required for F-test".to_string(),
        ));
    }

    // Determine unique classes and their indices
    let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
    for (i, &val) in y.iter().enumerate() {
        let key = val.round() as i64;
        class_indices.entry(key).or_default().push(i);
    }

    let n_classes = class_indices.len();
    if n_classes < 2 {
        return Err(TransformError::InvalidInput(
            "At least 2 classes required for F-test".to_string(),
        ));
    }

    let df_between = n_classes - 1;
    let df_within = n_samples - n_classes;

    if df_within == 0 {
        return Err(TransformError::InvalidInput(
            "Not enough samples within groups for F-test".to_string(),
        ));
    }

    let mut f_stats = Array1::zeros(n_features);
    let mut p_values = Array1::ones(n_features);

    for j in 0..n_features {
        // Compute overall mean for feature j
        let grand_mean = x.column(j).iter().sum::<f64>() / n_samples as f64;

        // Compute between-group sum of squares (SSB) and within-group sum of squares (SSW)
        let mut ssb = 0.0;
        let mut ssw = 0.0;

        for (_, indices) in &class_indices {
            let n_k = indices.len() as f64;
            let group_mean = indices.iter().map(|&i| x[[i, j]]).sum::<f64>() / n_k;

            // Between-group: n_k * (group_mean - grand_mean)^2
            ssb += n_k * (group_mean - grand_mean).powi(2);

            // Within-group: sum of (x_i - group_mean)^2
            for &i in indices {
                ssw += (x[[i, j]] - group_mean).powi(2);
            }
        }

        // F-statistic = (SSB / df_between) / (SSW / df_within)
        let msb = ssb / df_between as f64;
        let msw = ssw / df_within as f64;

        let f_stat = if msw > 1e-15 { msb / msw } else { 0.0 };
        f_stats[j] = f_stat;

        // Compute p-value from F-distribution
        p_values[j] = f_survival(f_stat, df_between, df_within);
    }

    Ok((f_stats, p_values))
}

/// Compute ANOVA F-test for regression (continuous target).
///
/// Uses correlation-based F-statistic: F = (r^2 / 1) / ((1-r^2) / (n-2))
/// where r is the Pearson correlation coefficient.
///
/// # Arguments
/// * `x` - Feature matrix, shape (n_samples, n_features)
/// * `y` - Continuous target, shape (n_samples,)
pub fn f_regression(x: &Array2<f64>, y: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    if n_samples != y.len() {
        return Err(TransformError::InvalidInput(format!(
            "X has {} samples but y has {} samples",
            n_samples,
            y.len()
        )));
    }

    if n_samples < 3 {
        return Err(TransformError::InvalidInput(
            "At least 3 samples required for F-test".to_string(),
        ));
    }

    let y_mean = y.iter().sum::<f64>() / n_samples as f64;
    let y_var: f64 = y.iter().map(|&v| (v - y_mean).powi(2)).sum();

    if y_var < 1e-15 {
        return Err(TransformError::InvalidInput(
            "Target variable has zero variance".to_string(),
        ));
    }

    let mut f_stats = Array1::zeros(n_features);
    let mut p_values = Array1::ones(n_features);

    for j in 0..n_features {
        let x_col = x.column(j);
        let x_mean = x_col.iter().sum::<f64>() / n_samples as f64;
        let x_var: f64 = x_col.iter().map(|&v| (v - x_mean).powi(2)).sum();

        if x_var < 1e-15 {
            // Zero-variance feature
            f_stats[j] = 0.0;
            p_values[j] = 1.0;
            continue;
        }

        // Pearson correlation
        let mut cov = 0.0;
        for i in 0..n_samples {
            cov += (x_col[i] - x_mean) * (y[i] - y_mean);
        }

        let r = cov / (x_var.sqrt() * y_var.sqrt());
        let r2 = r * r;

        // F = (r^2 / 1) / ((1 - r^2) / (n - 2))
        let df1 = 1;
        let df2 = n_samples - 2;

        let f_stat = if (1.0 - r2) > 1e-15 {
            (r2 * df2 as f64) / (1.0 - r2)
        } else {
            f64::MAX
        };

        f_stats[j] = f_stat;
        p_values[j] = f_survival(f_stat, df1, df2);
    }

    Ok((f_stats, p_values))
}

/// F-test feature selector
///
/// Selects the k best features based on ANOVA F-test scores.
#[derive(Debug, Clone)]
pub struct FTestSelector {
    /// Number of features to select
    k: usize,
    /// Whether the task is classification (true) or regression (false)
    classification: bool,
    /// Selected feature indices
    selected_features_: Option<Vec<usize>>,
    /// F-test scores
    scores_: Option<Array1<f64>>,
    /// P-values
    p_values_: Option<Array1<f64>>,
    /// Number of features at fit time
    n_features_in_: Option<usize>,
}

impl FTestSelector {
    /// Create a new F-test feature selector for classification
    pub fn classification(k: usize) -> Self {
        FTestSelector {
            k,
            classification: true,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        }
    }

    /// Create a new F-test feature selector for regression
    pub fn regression(k: usize) -> Self {
        FTestSelector {
            k,
            classification: false,
            selected_features_: None,
            scores_: None,
            p_values_: None,
            n_features_in_: None,
        }
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

        let (f_stats, p_values) = if self.classification {
            f_classif(x, y)?
        } else {
            f_regression(x, y)?
        };

        // Select top k by F-statistic
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            f_stats[b]
                .partial_cmp(&f_stats[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected: Vec<usize> = indices.into_iter().take(self.k).collect();

        self.scores_ = Some(f_stats);
        self.p_values_ = Some(p_values);
        self.selected_features_ = Some(selected);
        self.n_features_in_ = Some(n_features);

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features_.as_ref().ok_or_else(|| {
            TransformError::NotFitted("FTestSelector has not been fitted".to_string())
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

    /// Get F-test scores
    pub fn scores(&self) -> Option<&Array1<f64>> {
        self.scores_.as_ref()
    }

    /// Get p-values
    pub fn p_values(&self) -> Option<&Array1<f64>> {
        self.p_values_.as_ref()
    }
}

/// Compute the survival function (1 - CDF) of the F-distribution
/// using the regularized incomplete beta function.
///
/// F-distribution CDF: I_x(d1/2, d2/2) where x = d1*f / (d1*f + d2)
fn f_survival(f: f64, d1: usize, d2: usize) -> f64 {
    if f <= 0.0 || d1 == 0 || d2 == 0 {
        return 1.0;
    }

    let a = d1 as f64 / 2.0;
    let b = d2 as f64 / 2.0;
    let x = d1 as f64 * f / (d1 as f64 * f + d2 as f64);

    // P(F > f) = 1 - I_x(a, b) = I_{1-x}(b, a)
    regularized_incomplete_beta(b, a, 1.0 - x)
}

/// Regularized incomplete beta function I_x(a, b)
/// Uses continued fraction expansion (Lentz's method)
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();
    if log_prefix < -500.0 {
        return 0.0;
    }
    let prefix = log_prefix.exp();

    // Continued fraction expansion
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(tiny);
    let mut h = d;

    for m in 1..=max_iter {
        let m_f64 = m as f64;

        // Even step
        let a_even = m_f64 * (b - m_f64) * x / ((a + 2.0 * m_f64 - 1.0) * (a + 2.0 * m_f64));
        d = 1.0 + a_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        let a_odd =
            -((a + m_f64) * (a + b + m_f64) * x) / ((a + 2.0 * m_f64) * (a + 2.0 * m_f64 + 1.0));
        d = 1.0 + a_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_odd / c;
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

    prefix * h
}

/// Log of the beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
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
    fn test_f_classif_basic() {
        // Feature 0 strongly separates classes, feature 1 doesn't
        let x = Array::from_shape_vec(
            (6, 2),
            vec![1.0, 5.0, 2.0, 5.1, 1.5, 5.0, 8.0, 5.0, 9.0, 5.1, 8.5, 5.0],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let (f_stats, p_vals) = f_classif(&x, &y).expect("f_classif");

        assert!(
            f_stats[0] > f_stats[1],
            "f[0]={} should be > f[1]={}",
            f_stats[0],
            f_stats[1]
        );
        assert!(
            p_vals[0] < p_vals[1],
            "p[0]={} should be < p[1]={}",
            p_vals[0],
            p_vals[1]
        );
    }

    #[test]
    fn test_f_classif_three_classes() {
        let x = Array::from_shape_vec(
            (9, 2),
            vec![
                1.0, 5.0, 1.1, 5.1, 0.9, 4.9, 5.0, 5.0, 5.1, 5.1, 4.9, 4.9, 9.0, 5.0, 9.1, 5.1,
                8.9, 4.9,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let (f_stats, _p_vals) = f_classif(&x, &y).expect("f_classif");
        assert!(f_stats[0] > f_stats[1]);
    }

    #[test]
    fn test_f_regression_basic() {
        let n = 20;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..n {
            let t = i as f64 / n as f64;
            x_data.push(t); // linearly related
            x_data.push(0.5); // constant
            y_data.push(3.0 * t + 1.0);
        }

        let x = Array::from_shape_vec((n, 2), x_data).expect("test data");
        let y = Array::from_vec(y_data);

        let (f_stats, p_vals) = f_regression(&x, &y).expect("f_regression");
        assert!(f_stats[0] > f_stats[1]);
        assert!(p_vals[0] < 0.01); // Should be highly significant
    }

    #[test]
    fn test_f_test_selector_classification() {
        let x = Array::from_shape_vec(
            (6, 3),
            vec![
                1.0, 5.0, 0.5, 2.0, 5.1, 0.6, 1.5, 5.0, 0.4, 8.0, 5.0, 0.5, 9.0, 5.1, 0.6, 8.5,
                5.0, 0.4,
            ],
        )
        .expect("test data");
        let y = Array::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut selector = FTestSelector::classification(1);
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(transformed.shape(), &[6, 1]);

        let selected = selector.get_support().expect("support");
        assert_eq!(selected, &[0]); // Feature 0 is most discriminative
    }

    #[test]
    fn test_f_test_selector_regression() {
        let n = 20;
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for i in 0..n {
            let t = i as f64 / n as f64;
            x_data.push(t);
            x_data.push(0.5);
            x_data.push(2.0 * t);
            y_data.push(t + 0.1);
        }

        let x = Array::from_shape_vec((n, 3), x_data).expect("test data");
        let y = Array::from_vec(y_data);

        let mut selector = FTestSelector::regression(2);
        let transformed = selector.fit_transform(&x, &y).expect("fit_transform");
        assert_eq!(transformed.shape(), &[n, 2]);
    }

    #[test]
    fn test_f_test_errors() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let y_wrong_len = Array::from_vec(vec![0.0, 1.0]);

        assert!(f_classif(&x, &y_wrong_len).is_err());

        // Single class
        let y_single = Array::from_vec(vec![0.0, 0.0, 0.0, 0.0]);
        assert!(f_classif(&x, &y_single).is_err());
    }

    #[test]
    fn test_f_test_selector_not_fitted() {
        let x = Array::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("test data");
        let selector = FTestSelector::classification(1);
        assert!(selector.transform(&x).is_err());
    }

    #[test]
    fn test_f_survival_known_values() {
        // For very large F, p-value should be near 0
        let p = f_survival(100.0, 1, 10);
        assert!(p < 0.001, "p={} should be very small", p);

        // For F=0, p-value should be 1
        let p = f_survival(0.0, 1, 10);
        assert!((p - 1.0).abs() < 1e-10);
    }
}
