//! Concrete feature transformer implementations.
//!
//! This module provides production-quality implementations of common feature
//! scaling and normalisation techniques:
//!
//! - [`StandardScaler`] — zero-mean, unit-variance standardisation (z-score)
//! - [`MinMaxScaler`] — linear rescaling to a [0, 1] (or custom) range
//! - [`NormalizerTransform`] — per-row L1 or L2 normalisation
//!
//! All implementations satisfy the [`FeatureTransformer`] trait contract:
//! they are `Send + Sync`, handle edge cases (zero variance, zero norm, empty
//! input) without panicking, and propagate errors via [`PipelineError`].

use ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive};
use std::fmt;

use super::builder::PipelineError;
use super::traits::FeatureTransformer;

// ──────────────────────────────────────────────────────────────────────────────
// StandardScaler
// ──────────────────────────────────────────────────────────────────────────────

/// Standardise features to have zero mean and unit variance.
///
/// The transformation is:
///
/// ```text
/// z = (x - μ) / σ
/// ```
///
/// where μ is the per-column mean and σ is the per-column standard deviation
/// computed over the training set. If a column has zero variance (σ = 0),
/// all values in that column are mapped to 0.0 rather than NaN.
///
/// # Fitted state
///
/// - `mean_`: per-column means (shape `[n_features]`)
/// - `std_`:  per-column standard deviations (shape `[n_features]`)
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{FeatureTransformer, StandardScaler};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((3, 2), vec![1.0f64, 10.0, 2.0, 20.0, 3.0, 30.0]).expect("should succeed");
/// let mut scaler = StandardScaler::new();
/// let transformed = scaler.fit_transform(&data).expect("should succeed");
///
/// // Column means should be ~0
/// for col in 0..transformed.ncols() {
///     let col_mean: f64 = transformed.column(col).sum() / transformed.nrows() as f64;
///     assert!(col_mean.abs() < 1e-10);
/// }
/// # }
/// ```
pub struct StandardScaler<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    /// Per-column means, populated after fitting.
    mean_: Option<Array1<T>>,
    /// Per-column standard deviations, populated after fitting.
    std_: Option<Array1<T>>,
    /// Number of columns seen at fit time.
    n_features_in_: Option<usize>,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> StandardScaler<T> {
    /// Create a new, unfitted `StandardScaler`.
    pub fn new() -> Self {
        Self {
            mean_: None,
            std_: None,
            n_features_in_: None,
        }
    }

    /// Access the fitted per-column means.
    pub fn mean(&self) -> Option<&Array1<T>> {
        self.mean_.as_ref()
    }

    /// Access the fitted per-column standard deviations.
    pub fn std(&self) -> Option<&Array1<T>> {
        self.std_.as_ref()
    }
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> FeatureTransformer<T>
    for StandardScaler<T>
{
    fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }

        let n = T::from_usize(data.nrows()).ok_or_else(|| {
            PipelineError::NumericError(
                self.name().to_string(),
                "cannot convert row count to T".to_string(),
            )
        })?;

        let ncols = data.ncols();
        let mut means = Vec::with_capacity(ncols);
        let mut stds = Vec::with_capacity(ncols);

        for col_idx in 0..ncols {
            let col = data.column(col_idx);

            // Compute mean
            let sum = col.fold(T::zero(), |acc, &x| acc + x);
            let mean = sum / n;
            means.push(mean);

            // Compute population standard deviation
            let sq_diff_sum = col.fold(T::zero(), |acc, &x| {
                let diff = x - mean;
                acc + diff * diff
            });
            let variance = sq_diff_sum / n;
            let std_dev = variance.sqrt();
            stds.push(std_dev);
        }

        self.mean_ = Some(Array1::from_vec(means));
        self.std_ = Some(Array1::from_vec(stds));
        self.n_features_in_ = Some(ncols);
        Ok(())
    }

    fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        let (mean, std_arr) = match (&self.mean_, &self.std_) {
            (Some(m), Some(s)) => (m, s),
            _ => return Err(PipelineError::NotFitted(self.name().to_string())),
        };

        let expected = self.n_features_in_.expect("n_features_in set with mean_");
        if data.ncols() != expected {
            return Err(PipelineError::FeatureCountMismatch {
                step: self.name().to_string(),
                expected,
                actual: data.ncols(),
            });
        }

        if data.nrows() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }

        let mut out = data.clone();
        for col_idx in 0..data.ncols() {
            let mu = mean[col_idx];
            let sigma = std_arr[col_idx];
            let mut col_out = out.column_mut(col_idx);
            if sigma == T::zero() {
                // Zero-variance column → map all to 0
                col_out.fill(T::zero());
            } else {
                col_out.mapv_inplace(|x| (x - mu) / sigma);
            }
        }

        Ok(out)
    }

    fn name(&self) -> &str {
        "StandardScaler"
    }

    fn is_fitted(&self) -> bool {
        self.mean_.is_some()
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MinMaxScaler
// ──────────────────────────────────────────────────────────────────────────────

/// Rescale features to the range `[feature_range.0, feature_range.1]`.
///
/// The transformation is:
///
/// ```text
/// x_scaled = (x - x_min) / (x_max - x_min) * (max - min) + min
/// ```
///
/// where `(min, max)` is `feature_range` (defaults to `(0, 1)`).
/// If a column has a constant value (range = 0), all values in that column
/// are mapped to 0.0 rather than NaN.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{FeatureTransformer, MinMaxScaler};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_vec((3, 1), vec![0.0f64, 5.0, 10.0]).expect("should succeed");
/// let mut scaler = MinMaxScaler::new();
/// let transformed = scaler.fit_transform(&data).expect("should succeed");
/// assert!((transformed[[0, 0]] - 0.0).abs() < 1e-10);
/// assert!((transformed[[1, 0]] - 0.5).abs() < 1e-10);
/// assert!((transformed[[2, 0]] - 1.0).abs() < 1e-10);
/// # }
/// ```
pub struct MinMaxScaler<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    /// Per-column minimum values from fitting.
    data_min_: Option<Array1<T>>,
    /// Per-column maximum values from fitting.
    data_max_: Option<Array1<T>>,
    /// Target output range `(min, max)`.
    feature_range: (T, T),
    /// Number of features seen at fit time.
    n_features_in_: Option<usize>,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> MinMaxScaler<T> {
    /// Create a new `MinMaxScaler` with the default feature range `[0, 1]`.
    pub fn new() -> Self {
        Self {
            data_min_: None,
            data_max_: None,
            feature_range: (T::zero(), T::one()),
            n_features_in_: None,
        }
    }

    /// Create a `MinMaxScaler` with a custom output range.
    ///
    /// # Errors
    ///
    /// Returns `PipelineError::ConfigurationError` if `min >= max`.
    pub fn with_range(min: T, max: T) -> Result<Self, PipelineError> {
        if min >= max {
            return Err(PipelineError::ConfigurationError(format!(
                "MinMaxScaler: feature_range min ({:?}) must be < max ({:?})",
                min, max
            )));
        }
        Ok(Self {
            data_min_: None,
            data_max_: None,
            feature_range: (min, max),
            n_features_in_: None,
        })
    }

    /// Access fitted per-column minima.
    pub fn data_min(&self) -> Option<&Array1<T>> {
        self.data_min_.as_ref()
    }

    /// Access fitted per-column maxima.
    pub fn data_max(&self) -> Option<&Array1<T>> {
        self.data_max_.as_ref()
    }
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> FeatureTransformer<T>
    for MinMaxScaler<T>
{
    fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }

        let ncols = data.ncols();
        let mut mins = Vec::with_capacity(ncols);
        let mut maxs = Vec::with_capacity(ncols);

        for col_idx in 0..ncols {
            let col = data.column(col_idx);
            let mut col_min = T::infinity();
            let mut col_max = T::neg_infinity();
            for &v in col.iter() {
                if v < col_min {
                    col_min = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            mins.push(col_min);
            maxs.push(col_max);
        }

        self.data_min_ = Some(Array1::from_vec(mins));
        self.data_max_ = Some(Array1::from_vec(maxs));
        self.n_features_in_ = Some(ncols);
        Ok(())
    }

    fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        let (data_min, data_max) = match (&self.data_min_, &self.data_max_) {
            (Some(mn), Some(mx)) => (mn, mx),
            _ => return Err(PipelineError::NotFitted(self.name().to_string())),
        };

        let expected = self
            .n_features_in_
            .expect("n_features_in set with data_min_");
        if data.ncols() != expected {
            return Err(PipelineError::FeatureCountMismatch {
                step: self.name().to_string(),
                expected,
                actual: data.ncols(),
            });
        }

        if data.nrows() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }

        let (out_min, out_max) = self.feature_range;
        let out_range = out_max - out_min;

        let mut out = data.clone();
        for col_idx in 0..data.ncols() {
            let x_min = data_min[col_idx];
            let x_max = data_max[col_idx];
            let x_range = x_max - x_min;

            let mut col_out = out.column_mut(col_idx);
            if x_range == T::zero() {
                // Constant column → map to out_min (conventionally 0 for [0,1])
                col_out.fill(out_min);
            } else {
                col_out.mapv_inplace(|x| {
                    let x_std = (x - x_min) / x_range;
                    x_std * out_range + out_min
                });
            }
        }

        Ok(out)
    }

    fn name(&self) -> &str {
        "MinMaxScaler"
    }

    fn is_fitted(&self) -> bool {
        self.data_min_.is_some()
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// NormalizerTransform
// ──────────────────────────────────────────────────────────────────────────────

/// Norm type used by [`NormalizerTransform`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm (sum of absolute values).
    L1,
    /// L2 norm (Euclidean norm).
    L2,
    /// L∞ norm (maximum absolute value).
    LInf,
}

/// Normalise each sample (row) independently to unit norm.
///
/// Unlike [`StandardScaler`] and [`MinMaxScaler`], this transformer
/// operates **row-wise**: each row is divided by its own norm. This is
/// useful when the magnitude of individual samples is irrelevant and only
/// the direction (relative feature ratios) matter.
///
/// Rows with a norm of zero are left unchanged (all-zero output).
///
/// Because this transformation does not learn from training data, `fit` is
/// a no-op and `transform` can be called without a prior `fit`. However,
/// `fit` still validates the input and records the expected feature count.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "ml_pipeline")]
/// # {
/// use scirs2_core::ml_pipeline::{FeatureTransformer, NormalizerTransform};
/// use ndarray::Array2;
///
/// // Row [3, 4, 0] has L2 norm 5.0 → normalised to [0.6, 0.8, 0.0]
/// let data = Array2::from_shape_vec((1, 3), vec![3.0f64, 4.0, 0.0]).expect("should succeed");
/// let mut norm = NormalizerTransform::l2();
/// let result = norm.fit_transform(&data).expect("should succeed");
/// assert!((result[[0, 0]] - 0.6).abs() < 1e-10);
/// assert!((result[[0, 1]] - 0.8).abs() < 1e-10);
/// assert!(result[[0, 2]].abs() < 1e-10);
/// # }
/// ```
pub struct NormalizerTransform<T: Clone + fmt::Debug + Float + FromPrimitive + 'static> {
    norm_type: NormType,
    /// Number of features seen at fit time (or None if not fitted).
    n_features_in_: Option<usize>,
    /// Phantom to associate T with the struct.
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> NormalizerTransform<T> {
    /// Create a normalizer using L2 norm.
    pub fn l2() -> Self {
        Self {
            norm_type: NormType::L2,
            n_features_in_: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a normalizer using L1 norm.
    pub fn l1() -> Self {
        Self {
            norm_type: NormType::L1,
            n_features_in_: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a normalizer using L∞ norm.
    pub fn linf() -> Self {
        Self {
            norm_type: NormType::LInf,
            n_features_in_: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a normalizer with an explicit norm type.
    pub fn with_norm(norm_type: NormType) -> Self {
        Self {
            norm_type,
            n_features_in_: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// The norm type used by this transformer.
    pub fn norm_type(&self) -> NormType {
        self.norm_type
    }

    /// Compute the norm for a single row.
    fn compute_row_norm(&self, row: ndarray::ArrayView1<T>) -> T {
        match self.norm_type {
            NormType::L1 => row.fold(T::zero(), |acc, &x| acc + x.abs()),
            NormType::L2 => {
                let sq_sum = row.fold(T::zero(), |acc, &x| acc + x * x);
                sq_sum.sqrt()
            }
            NormType::LInf => row.fold(T::zero(), |acc, &x| {
                let abs_x = x.abs();
                if abs_x > acc {
                    abs_x
                } else {
                    acc
                }
            }),
        }
    }
}

impl<T: Clone + fmt::Debug + Float + FromPrimitive + Send + Sync + 'static> FeatureTransformer<T>
    for NormalizerTransform<T>
{
    /// Fit records the feature count. The transformation itself is stateless.
    fn fit(&mut self, data: &Array2<T>) -> Result<(), PipelineError> {
        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }
        self.n_features_in_ = Some(data.ncols());
        Ok(())
    }

    fn transform(&self, data: &Array2<T>) -> Result<Array2<T>, PipelineError> {
        // Allow transform without fit (stateless transformer), but if fitted,
        // validate feature count.
        if let Some(expected) = self.n_features_in_ {
            if data.ncols() != expected {
                return Err(PipelineError::FeatureCountMismatch {
                    step: self.name().to_string(),
                    expected,
                    actual: data.ncols(),
                });
            }
        }

        if data.nrows() == 0 || data.ncols() == 0 {
            return Err(PipelineError::EmptyInput(self.name().to_string()));
        }

        let mut out = data.clone();
        for row_idx in 0..data.nrows() {
            let row = data.row(row_idx);
            let norm = self.compute_row_norm(row);
            if norm != T::zero() {
                let mut row_out = out.row_mut(row_idx);
                row_out.mapv_inplace(|x| x / norm);
            }
            // rows with norm == 0 are left as-is (all zeros)
        }

        Ok(out)
    }

    fn name(&self) -> &str {
        match self.norm_type {
            NormType::L1 => "NormalizerL1",
            NormType::L2 => "NormalizerL2",
            NormType::LInf => "NormalizerLInf",
        }
    }

    fn is_fitted(&self) -> bool {
        self.n_features_in_.is_some()
    }

    fn n_features_in(&self) -> Option<usize> {
        self.n_features_in_
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod scaler_tests {
    use super::*;
    use crate::ml_pipeline::builder::PipelineError;
    use ndarray::Array2;

    // ── StandardScaler tests ──────────────────────────────────────────────────

    #[test]
    fn test_standard_scaler_basic() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0])
            .expect("shape is valid");

        let mut scaler = StandardScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        assert_eq!(result.shape(), &[3, 2]);

        // Column means should be ~0
        for col in 0..result.ncols() {
            let mean: f64 = result.column(col).sum() / result.nrows() as f64;
            assert!(mean.abs() < 1e-10, "col {col} mean: {mean}");
        }
    }

    #[test]
    fn test_standard_scaler_zero_variance_column() {
        let data = Array2::from_shape_vec((3, 2), vec![5.0f64, 1.0, 5.0, 2.0, 5.0, 3.0])
            .expect("shape is valid");

        let mut scaler = StandardScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        // All values in column 0 map to 0 (zero variance)
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 0]], 0.0);
        assert_eq!(result[[2, 0]], 0.0);
    }

    #[test]
    fn test_standard_scaler_not_fitted_error() {
        let scaler = StandardScaler::<f64>::new();
        let data =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape is valid");
        let result = scaler.transform(&data);
        assert!(matches!(result, Err(PipelineError::NotFitted(_))));
    }

    #[test]
    fn test_standard_scaler_feature_count_mismatch() {
        let train =
            Array2::from_shape_vec((3, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("shape is valid");
        let test =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape is valid");

        let mut scaler = StandardScaler::new();
        scaler.fit(&train).expect("fit should succeed");
        let result = scaler.transform(&test);
        assert!(matches!(
            result,
            Err(PipelineError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn test_standard_scaler_empty_input_error() {
        let data: Array2<f64> = Array2::zeros((0, 3));
        let mut scaler = StandardScaler::new();
        let result = scaler.fit(&data);
        assert!(matches!(result, Err(PipelineError::EmptyInput(_))));
    }

    #[test]
    fn test_standard_scaler_fitted_state() {
        let data = Array2::from_shape_vec((4, 2), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape is valid");
        let mut scaler = StandardScaler::new();
        assert!(!scaler.is_fitted());
        assert_eq!(scaler.n_features_in(), None);
        scaler.fit(&data).expect("fit should succeed");
        assert!(scaler.is_fitted());
        assert_eq!(scaler.n_features_in(), Some(2));
        assert!(scaler.mean().is_some());
        assert!(scaler.std().is_some());
    }

    // ── MinMaxScaler tests ────────────────────────────────────────────────────

    #[test]
    fn test_min_max_scaler_basic() {
        let data = Array2::from_shape_vec((3, 1), vec![0.0f64, 5.0, 10.0]).expect("shape is valid");
        let mut scaler = MinMaxScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        assert!((result[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((result[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scaler_custom_range() {
        let data = Array2::from_shape_vec((3, 1), vec![0.0f64, 5.0, 10.0]).expect("shape is valid");
        let mut scaler = MinMaxScaler::with_range(-1.0_f64, 1.0).expect("valid range");
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");

        assert!((result[[0, 0]] - (-1.0)).abs() < 1e-10);
        assert!((result[[1, 0]] - 0.0).abs() < 1e-10);
        assert!((result[[2, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_max_scaler_invalid_range_returns_error() {
        let result = MinMaxScaler::<f64>::with_range(1.0, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_min_max_scaler_uniform_column() {
        let data = Array2::from_shape_vec((3, 1), vec![7.0f64, 7.0, 7.0]).expect("shape is valid");
        let mut scaler = MinMaxScaler::new();
        let result = scaler
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // All values map to out_min (0.0 for default range)
        for &v in result.iter() {
            assert_eq!(v, 0.0, "uniform column should map to out_min");
        }
    }

    #[test]
    fn test_min_max_scaler_not_fitted_error() {
        let scaler = MinMaxScaler::<f64>::new();
        let data =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("shape is valid");
        assert!(matches!(
            scaler.transform(&data),
            Err(PipelineError::NotFitted(_))
        ));
    }

    #[test]
    fn test_min_max_scaler_data_min_max_accessors() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0f64, 10.0, 5.0, 50.0, 3.0, 30.0])
            .expect("shape is valid");
        let mut scaler = MinMaxScaler::new();
        scaler.fit(&data).expect("fit should succeed");

        let data_min = scaler.data_min().expect("should be fitted");
        let data_max = scaler.data_max().expect("should be fitted");

        assert!((data_min[0] - 1.0).abs() < 1e-10);
        assert!((data_min[1] - 10.0).abs() < 1e-10);
        assert!((data_max[0] - 5.0).abs() < 1e-10);
        assert!((data_max[1] - 50.0).abs() < 1e-10);
    }

    // ── NormalizerTransform tests ─────────────────────────────────────────────

    #[test]
    fn test_normalizer_l2() {
        let data = Array2::from_shape_vec((1, 3), vec![3.0f64, 4.0, 0.0]).expect("shape is valid");
        let mut norm = NormalizerTransform::l2();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // norm = 5
        assert!((result[[0, 0]] - 0.6).abs() < 1e-10);
        assert!((result[[0, 1]] - 0.8).abs() < 1e-10);
        assert!(result[[0, 2]].abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_l1() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0f64, 2.0, 3.0]).expect("shape is valid");
        let mut norm = NormalizerTransform::l1();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // L1 norm = 6
        assert!((result[[0, 0]] - 1.0 / 6.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 2.0 / 6.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 3.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_linf() {
        let data = Array2::from_shape_vec((1, 3), vec![1.0f64, -5.0, 3.0]).expect("shape is valid");
        let mut norm = NormalizerTransform::linf();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // L∞ norm = 5.0
        assert!((result[[0, 0]] - 0.2).abs() < 1e-10);
        assert!((result[[0, 1]] - (-1.0)).abs() < 1e-10);
        assert!((result[[0, 2]] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_zero_row() {
        let data = Array2::from_shape_vec((2, 3), vec![0.0f64, 0.0, 0.0, 1.0, 0.0, 0.0])
            .expect("shape is valid");
        let mut norm = NormalizerTransform::l2();
        let result = norm
            .fit_transform(&data)
            .expect("fit_transform should succeed");
        // Row 0 is all-zero → unchanged
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[0, 1]], 0.0);
        assert_eq!(result[[0, 2]], 0.0);
        // Row 1: [1,0,0] / 1 = [1,0,0]
        assert!((result[[1, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalizer_stateless_transform_without_fit() {
        let data =
            Array2::from_shape_vec((2, 2), vec![3.0f64, 4.0, 1.0, 0.0]).expect("shape is valid");
        // NormalizerTransform is stateless – transform can be called without fit
        let norm = NormalizerTransform::<f64>::l2();
        let result = norm
            .transform(&data)
            .expect("transform should succeed even without fit");
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_normalizer_feature_count_mismatch_after_fit() {
        let train = Array2::from_shape_vec((2, 3), vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("shape is valid");
        let test =
            Array2::from_shape_vec((2, 2), vec![1.0f64, 2.0, 3.0, 4.0]).expect("shape is valid");
        let mut norm = NormalizerTransform::<f64>::l2();
        norm.fit(&train).expect("fit should succeed");
        let result = norm.transform(&test);
        assert!(matches!(
            result,
            Err(PipelineError::FeatureCountMismatch { .. })
        ));
    }

    #[test]
    fn test_normalizer_name_reflects_norm_type() {
        assert_eq!(NormalizerTransform::<f64>::l1().name(), "NormalizerL1");
        assert_eq!(NormalizerTransform::<f64>::l2().name(), "NormalizerL2");
        assert_eq!(NormalizerTransform::<f64>::linf().name(), "NormalizerLInf");
    }
}
