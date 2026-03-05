//! Extended feature transformers for the ML Pipeline API.
//!
//! This module provides additional transformers beyond the basic scalers:
//!
//! - [`RobustScaler`] — median/IQR-based scaling, resistant to outliers
//! - [`PolynomialFeatures`] — polynomial and interaction feature expansion
//!
//! All implementations satisfy the [`FeatureTransformer`] trait contract.

use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt;

use super::builder::PipelineError;
use super::traits::FeatureTransformer;

// ─────────────────────────────────────────────────────────────────────────────
// RobustScaler
// ─────────────────────────────────────────────────────────────────────────────

/// Scale features using statistics that are robust to outliers.
///
/// This scaler removes the median and scales according to the interquartile
/// range (IQR). The transformation is:
///
/// ```text
/// z = (x - median) / IQR
/// ```
///
/// where `IQR = Q(q_high) - Q(q_low)` (default: 25th to 75th percentile).
/// If IQR is zero, the feature is left at zero (analogous to `StandardScaler`
/// behaviour for zero-variance columns).
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::transformer::RobustScaler;
/// use scirs2_core::ml_pipeline::traits::FeatureTransformer;
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec(
///     (5, 1),
///     vec![1.0f64, 2.0, 3.0, 4.0, 100.0],
/// ).expect("shape ok");
///
/// let mut scaler = RobustScaler::new();
/// let out = scaler.fit_transform(&data).expect("fit_transform ok");
/// // Median is 3.0; IQR ~ 2.0; outlier 100 does not dominate
/// assert!(out[[2, 0]].abs() < 1e-10, "median maps to 0");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct RobustScaler {
    median: Option<Array1<f64>>,
    iqr: Option<Array1<f64>>,
    /// Quantile range (q_low, q_high) as percentages in [0, 100].
    quantile_range: (f64, f64),
    n_features: Option<usize>,
    is_fitted: bool,
}

impl Default for RobustScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustScaler {
    /// Create a `RobustScaler` with default quantile range `(25.0, 75.0)`.
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            quantile_range: (25.0, 75.0),
            n_features: None,
            is_fitted: false,
        }
    }

    /// Create a `RobustScaler` with a custom quantile range.
    ///
    /// # Arguments
    ///
    /// * `q_low` — lower percentile in `[0, 100)`, must be less than `q_high`.
    /// * `q_high` — upper percentile in `(0, 100]`, must be greater than `q_low`.
    pub fn with_quantile_range(q_low: f64, q_high: f64) -> Self {
        Self {
            median: None,
            iqr: None,
            quantile_range: (q_low, q_high),
            n_features: None,
            is_fitted: false,
        }
    }

    /// Return the fitted median (one value per feature), or `None` if not yet fitted.
    pub fn median(&self) -> Option<&Array1<f64>> {
        self.median.as_ref()
    }

    /// Return the fitted IQR (one value per feature), or `None` if not yet fitted.
    pub fn iqr(&self) -> Option<&Array1<f64>> {
        self.iqr.as_ref()
    }
}

/// Compute the `q`-th percentile (0-100) of a sorted slice using the
/// Hyndman-Fan method 6 ("exclusive") interpolation, which is the method
/// used by R's default `quantile()` and many statistical textbooks.
///
/// rank = q/100 * (n + 1), then linear-interpolate between adjacent values.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    // H&F method 6: rank = q/100 * (n+1), 1-indexed
    let rank = q / 100.0 * (n as f64 + 1.0);
    // Clamp to valid range [1, n]
    let rank = rank.clamp(1.0, n as f64);
    // Convert to 0-indexed
    let idx = rank - 1.0;
    let lo = idx.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

impl FeatureTransformer<f64> for RobustScaler {
    fn fit(&mut self, data: &Array2<f64>) -> Result<(), PipelineError> {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        if nrows == 0 || ncols == 0 {
            return Err(PipelineError::EmptyInput("RobustScaler".to_string()));
        }

        let (q_low, q_high) = self.quantile_range;
        let mut median_vals = Vec::with_capacity(ncols);
        let mut iqr_vals = Vec::with_capacity(ncols);

        for j in 0..ncols {
            let mut col: Vec<f64> = data.column(j).to_vec();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = percentile_sorted(&col, 50.0);
            let q1 = percentile_sorted(&col, q_low);
            let q3 = percentile_sorted(&col, q_high);
            median_vals.push(med);
            iqr_vals.push(q3 - q1);
        }

        self.median = Some(Array1::from_vec(median_vals));
        self.iqr = Some(Array1::from_vec(iqr_vals));
        self.n_features = Some(ncols);
        self.is_fitted = true;
        Ok(())
    }

    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PipelineError> {
        if !self.is_fitted {
            return Err(PipelineError::NotFitted("RobustScaler".to_string()));
        }
        let n_features = self.n_features.unwrap_or(0);
        if data.ncols() != n_features {
            return Err(PipelineError::FeatureCountMismatch {
                step: "RobustScaler".to_string(),
                expected: n_features,
                actual: data.ncols(),
            });
        }
        if data.nrows() == 0 {
            return Err(PipelineError::EmptyInput("RobustScaler".to_string()));
        }

        let median = self.median.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("RobustScaler.median".to_string())
        })?;
        let iqr = self.iqr.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("RobustScaler.iqr".to_string())
        })?;

        let mut out = data.to_owned();
        for j in 0..n_features {
            let med = median[j];
            let scale = iqr[j];
            for i in 0..data.nrows() {
                out[[i, j]] = if scale.abs() < f64::EPSILON {
                    0.0
                } else {
                    (data[[i, j]] - med) / scale
                };
            }
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "RobustScaler"
    }

    fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PolynomialFeatures
// ─────────────────────────────────────────────────────────────────────────────

/// Generate polynomial and interaction features.
///
/// For input matrix `X` with `d` features, this transformer produces all
/// monomials of degree ≤ `degree`. For `degree=2` with 2 input features
/// `[x0, x1]` the output (with `include_bias=true`) is:
///
/// ```text
/// [1, x0, x1, x0², x0·x1, x1²]
/// ```
///
/// With `interaction_only=true`, pure powers (e.g. `x0²`) are excluded,
/// leaving only cross-terms.
///
/// With `include_bias=false`, the constant 1 column is omitted.
///
/// # Performance note
///
/// The number of output features grows combinatorially:
/// `C(d + degree, degree)` without interaction-only;
/// `C(d, 2) + ... + C(d, degree)` with interaction-only.
/// Keep `degree` small for high-dimensional data.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::transformer::PolynomialFeatures;
/// use scirs2_core::ml_pipeline::traits::FeatureTransformer;
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((2, 2), vec![2.0f64, 3.0, 4.0, 5.0]).expect("shape ok");
/// let mut poly = PolynomialFeatures::new(2);
/// let out = poly.fit_transform(&data).expect("ok");
/// // Output shape: (2, 6) = [1, x0, x1, x0^2, x0*x1, x1^2]
/// assert_eq!(out.ncols(), 6);
/// // Row 0: [1, 2, 3, 4, 6, 9]
/// assert!((out[[0, 0]] - 1.0).abs() < 1e-10);
/// assert!((out[[0, 3]] - 4.0).abs() < 1e-10);
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFeatures {
    degree: usize,
    include_bias: bool,
    interaction_only: bool,
    /// Indices (into input columns) for each output column; None = bias term.
    combinations: Option<Vec<Vec<usize>>>,
    n_features_in: Option<usize>,
    is_fitted: bool,
}

impl PolynomialFeatures {
    /// Create a `PolynomialFeatures` transformer of the given degree.
    ///
    /// By default `include_bias=true` and `interaction_only=false`.
    pub fn new(degree: usize) -> Self {
        Self {
            degree,
            include_bias: true,
            interaction_only: false,
            combinations: None,
            n_features_in: None,
            is_fitted: false,
        }
    }

    /// Disable the constant bias column.
    pub fn without_bias(mut self) -> Self {
        self.include_bias = false;
        self
    }

    /// Only generate interaction terms (exclude pure powers like x²).
    pub fn interaction_only(mut self) -> Self {
        self.interaction_only = true;
        self
    }

    /// Number of output features (after fitting), or `None` if not fitted.
    pub fn n_output_features(&self) -> Option<usize> {
        self.combinations.as_ref().map(|c| c.len())
    }

    /// Build the list of index combinations for the given parameters.
    ///
    /// Combinations are grouped by degree (degree 0 = bias, degree 1, degree 2, ...)
    /// matching the scikit-learn convention.
    fn build_combinations(
        n_features: usize,
        degree: usize,
        include_bias: bool,
        interaction_only: bool,
    ) -> Vec<Vec<usize>> {
        let mut result: Vec<Vec<usize>> = Vec::new();

        if include_bias {
            result.push(Vec::new()); // bias = product of zero factors = 1
        }

        // Generate combinations for each degree level separately so that
        // all degree-1 terms appear before degree-2 terms, etc.
        for d in 1..=degree {
            Self::combinations_of_degree(
                n_features,
                d,
                interaction_only,
                0,
                &mut Vec::new(),
                &mut result,
            );
        }

        result
    }

    /// Generate all combinations of exactly `target_degree` elements from
    /// `0..n_features` (with repetition unless `interaction_only`).
    fn combinations_of_degree(
        n_features: usize,
        target_degree: usize,
        interaction_only: bool,
        min_idx: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == target_degree {
            result.push(current.clone());
            return;
        }
        let start = if interaction_only {
            // For interaction_only, next index must be strictly greater than last
            current.last().map(|&x| x + 1).unwrap_or(min_idx)
        } else {
            // Allow repetition: next index >= last
            current.last().copied().unwrap_or(min_idx)
        };

        for i in start..n_features {
            current.push(i);
            Self::combinations_of_degree(
                n_features,
                target_degree,
                interaction_only,
                min_idx,
                current,
                result,
            );
            current.pop();
        }
    }
}

impl FeatureTransformer<f64> for PolynomialFeatures {
    fn fit(&mut self, data: &Array2<f64>) -> Result<(), PipelineError> {
        let (nrows, ncols) = (data.nrows(), data.ncols());
        if nrows == 0 || ncols == 0 {
            return Err(PipelineError::EmptyInput("PolynomialFeatures".to_string()));
        }
        if self.degree == 0 {
            return Err(PipelineError::StepError {
                step: "PolynomialFeatures".to_string(),
                message: "degree must be >= 1".to_string(),
            });
        }

        let combos = Self::build_combinations(
            ncols,
            self.degree,
            self.include_bias,
            self.interaction_only,
        );
        self.combinations = Some(combos);
        self.n_features_in = Some(ncols);
        self.is_fitted = true;
        Ok(())
    }

    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>, PipelineError> {
        if !self.is_fitted {
            return Err(PipelineError::NotFitted("PolynomialFeatures".to_string()));
        }
        let n_features_in = self.n_features_in.unwrap_or(0);
        if data.ncols() != n_features_in {
            return Err(PipelineError::FeatureCountMismatch {
                step: "PolynomialFeatures".to_string(),
                expected: n_features_in,
                actual: data.ncols(),
            });
        }
        if data.nrows() == 0 {
            return Err(PipelineError::EmptyInput("PolynomialFeatures".to_string()));
        }

        let combinations = self.combinations.as_ref().ok_or_else(|| {
            PipelineError::NotFitted("PolynomialFeatures.combinations".to_string())
        })?;

        let n_out = combinations.len();
        let nrows = data.nrows();
        let mut out = Array2::<f64>::zeros((nrows, n_out));

        for (col_idx, combo) in combinations.iter().enumerate() {
            for row in 0..nrows {
                let val = if combo.is_empty() {
                    1.0 // bias term
                } else {
                    combo.iter().fold(1.0, |acc, &feat| acc * data[[row, feat]])
                };
                out[[row, col_idx]] = val;
            }
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "PolynomialFeatures"
    }

    fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RobustScaler ──────────────────────────────────────────────────────────

    #[test]
    fn test_robust_scaler_basic() {
        // Column values: [1,2,3,4,5]
        // Median=3, Q1=1.5, Q3=4.5, IQR=3.0
        let data = Array2::from_shape_vec(
            (5, 1),
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        )
        .expect("shape ok");
        let mut scaler = RobustScaler::new();
        let out = scaler.fit_transform(&data).expect("ok");
        // Median (3.0) should map to 0
        assert!((out[[2, 0]]).abs() < 1e-10, "median → 0, got {}", out[[2, 0]]);
        // 5.0 → (5 - 3) / 3 ≈ 0.667
        assert!((out[[4, 0]] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_robust_scaler_outlier_resistance() {
        // Adding extreme outlier: should not dominate scaling
        let data = Array2::from_shape_vec(
            (5, 1),
            vec![1.0f64, 2.0, 3.0, 4.0, 1000.0],
        )
        .expect("shape ok");
        let mut scaler = RobustScaler::new();
        scaler.fit(&data).expect("fit ok");
        let median = scaler.median().expect("fitted");
        // median of [1,2,3,4,1000] = 3.0
        assert!((median[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_robust_scaler_zero_iqr() {
        // Constant column: IQR=0, should map to 0
        let data = Array2::from_shape_vec(
            (3, 1),
            vec![5.0f64, 5.0, 5.0],
        )
        .expect("shape ok");
        let mut scaler = RobustScaler::new();
        let out = scaler.fit_transform(&data).expect("ok");
        for &v in out.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_robust_scaler_custom_quantile_range() {
        let data = Array2::from_shape_vec(
            (5, 1),
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
        )
        .expect("shape ok");
        let mut scaler = RobustScaler::with_quantile_range(10.0, 90.0);
        scaler.fit(&data).expect("fit ok");
        let iqr = scaler.iqr().expect("fitted");
        // Q10 ≈ 1.4, Q90 ≈ 4.6  → IQR ≈ 3.2
        assert!(iqr[0] > 3.0, "IQR with 10-90 range: {}", iqr[0]);
    }

    #[test]
    fn test_robust_scaler_not_fitted_error() {
        let data = Array2::from_shape_vec((2, 1), vec![1.0f64, 2.0]).expect("shape ok");
        let scaler = RobustScaler::new();
        let result = scaler.transform(&data);
        assert!(matches!(result, Err(PipelineError::NotFitted(_))));
    }

    #[test]
    fn test_robust_scaler_empty_data_error() {
        let data: Array2<f64> = Array2::zeros((0, 2));
        let mut scaler = RobustScaler::new();
        let result = scaler.fit(&data);
        assert!(matches!(result, Err(PipelineError::EmptyInput(_))));
    }

    #[test]
    fn test_robust_scaler_feature_mismatch_error() {
        let train = Array2::from_shape_vec((3, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape ok");
        let test = Array2::from_shape_vec((2, 1), vec![1.0f64, 2.0]).expect("shape ok");
        let mut scaler = RobustScaler::new();
        scaler.fit(&train).expect("fit ok");
        let result = scaler.transform(&test);
        assert!(matches!(
            result,
            Err(PipelineError::FeatureCountMismatch { .. })
        ));
    }

    // ── PolynomialFeatures ────────────────────────────────────────────────────

    #[test]
    fn test_poly_features_degree2_shape() {
        // d=2 features, degree=2, with bias: C(4,2)=6 output features
        let data = Array2::from_shape_vec(
            (3, 2),
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("shape ok");
        let mut poly = PolynomialFeatures::new(2);
        let out = poly.fit_transform(&data).expect("ok");
        // [1, x0, x1, x0^2, x0*x1, x1^2] → 6 columns
        assert_eq!(out.ncols(), 6);
        assert_eq!(out.nrows(), 3);
    }

    #[test]
    fn test_poly_features_degree2_values() {
        let data = Array2::from_shape_vec(
            (1, 2),
            vec![2.0f64, 3.0],
        )
        .expect("shape ok");
        let mut poly = PolynomialFeatures::new(2);
        let out = poly.fit_transform(&data).expect("ok");
        // Expected: [1, 2, 3, 4, 6, 9]
        let expected = [1.0, 2.0, 3.0, 4.0, 6.0, 9.0];
        for (j, &exp) in expected.iter().enumerate() {
            assert!(
                (out[[0, j]] - exp).abs() < 1e-10,
                "col {j}: expected {exp}, got {}",
                out[[0, j]]
            );
        }
    }

    #[test]
    fn test_poly_features_without_bias() {
        let data = Array2::from_shape_vec(
            (1, 2),
            vec![2.0f64, 3.0],
        )
        .expect("shape ok");
        let mut poly = PolynomialFeatures::new(2).without_bias();
        let out = poly.fit_transform(&data).expect("ok");
        // Expected: [2, 3, 4, 6, 9] → 5 columns
        assert_eq!(out.ncols(), 5);
        assert!((out[[0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_features_interaction_only() {
        let data = Array2::from_shape_vec(
            (1, 2),
            vec![2.0f64, 3.0],
        )
        .expect("shape ok");
        let mut poly = PolynomialFeatures::new(2).without_bias().interaction_only();
        let out = poly.fit_transform(&data).expect("ok");
        // degree 1: [x0, x1]; degree 2 interactions: [x0*x1] → 3 columns
        assert_eq!(out.ncols(), 3);
        // Values: [2, 3, 6]
        assert!((out[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((out[[0, 1]] - 3.0).abs() < 1e-10);
        assert!((out[[0, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_features_degree1_identity() {
        // degree=1 without bias → identity transform
        let data = Array2::from_shape_vec(
            (2, 3),
            vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("shape ok");
        let mut poly = PolynomialFeatures::new(1).without_bias();
        let out = poly.fit_transform(&data).expect("ok");
        assert_eq!(out, data);
    }

    #[test]
    fn test_poly_features_not_fitted_error() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, 3.0, 4.0]).expect("shape ok");
        let poly = PolynomialFeatures::new(2);
        let result = poly.transform(&data);
        assert!(matches!(result, Err(PipelineError::NotFitted(_))));
    }

    #[test]
    fn test_poly_features_empty_data_error() {
        let data: Array2<f64> = Array2::zeros((0, 2));
        let mut poly = PolynomialFeatures::new(2);
        let result = poly.fit(&data);
        assert!(matches!(result, Err(PipelineError::EmptyInput(_))));
    }

    #[test]
    fn test_percentile_sorted_edge_cases() {
        let single = vec![42.0f64];
        assert_eq!(percentile_sorted(&single, 50.0), 42.0);
        assert_eq!(percentile_sorted(&[], 50.0), 0.0);
    }
}
