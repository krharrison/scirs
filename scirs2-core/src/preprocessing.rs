//! Data preprocessing utilities for scientific computing
//!
//! This module provides common data preprocessing operations used across
//! the SciRS2 ecosystem, including scaling, encoding, imputation, and
//! outlier detection.
//!
//! # Scalers
//!
//! - [`StandardScaler`] - Standardize features by removing the mean and scaling to unit variance
//! - [`MinMaxScaler`] - Scale features to a given range (default [0, 1])
//! - [`RobustScaler`] - Scale features using statistics robust to outliers (median, IQR)
//! - [`MaxAbsScaler`] - Scale each feature by its maximum absolute value
//!
//! # Encoders
//!
//! - [`LabelEncoder`] - Encode string labels as integers
//! - [`OneHotEncoder`] - Encode categorical features as one-hot numeric arrays
//! - [`OrdinalEncoder`] - Encode categorical features as ordinal integers
//!
//! # Imputation
//!
//! - [`Imputer`] - Fill missing values using various strategies
//!
//! # Outlier Detection
//!
//! - [`OutlierDetector`] - Detect outliers using Z-score or IQR methods

use crate::error::{CoreError, CoreResult, ErrorContext};
use ::ndarray::{Array1, Array2, Axis};
use num_traits::{Float, FromPrimitive, NumCast, Zero};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::hash::Hash;

// ---------------------------------------------------------------------------
// StandardScaler
// ---------------------------------------------------------------------------

/// Standardize features by removing the mean and scaling to unit variance.
///
/// z = (x - mean) / std
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::StandardScaler;
/// use ndarray::array;
///
/// let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let mut scaler = StandardScaler::<f64>::new();
/// scaler.fit(&data).expect("fit failed");
/// let transformed = scaler.transform(&data).expect("transform failed");
/// ```
#[derive(Debug, Clone)]
pub struct StandardScaler<F: Float> {
    /// Per-feature mean
    pub mean: Option<Array1<F>>,
    /// Per-feature standard deviation
    pub std_dev: Option<Array1<F>>,
    /// Whether to center the data (subtract mean)
    pub with_mean: bool,
    /// Whether to scale to unit variance
    pub with_std: bool,
}

impl<F: Float + FromPrimitive + Debug + Display + std::iter::Sum> StandardScaler<F> {
    /// Create a new StandardScaler with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: None,
            std_dev: None,
            with_mean: true,
            with_std: true,
        }
    }

    /// Create with explicit centering/scaling options
    #[must_use]
    pub fn with_options(with_mean: bool, with_std: bool) -> Self {
        Self {
            mean: None,
            std_dev: None,
            with_mean,
            with_std,
        }
    }

    /// Fit the scaler by computing mean and std from the data.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        let n_samples = data.nrows();
        if n_samples == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit StandardScaler on empty data",
            )));
        }
        let n_f = F::from_usize(n_samples).ok_or_else(|| {
            CoreError::ValueError(ErrorContext::new("Failed to convert n_samples to float"))
        })?;
        let n_cols = data.ncols();
        let mut mean_arr = Array1::<F>::zeros(n_cols);
        let mut std_arr = Array1::<F>::zeros(n_cols);

        for j in 0..n_cols {
            let col = data.column(j);
            let sum: F = col.iter().copied().sum();
            let m = sum / n_f;
            mean_arr[j] = m;

            let var_sum: F = col.iter().map(|&x| (x - m) * (x - m)).sum();
            let var = var_sum / n_f;
            std_arr[j] = var.sqrt();
        }

        self.mean = Some(mean_arr);
        self.std_dev = Some(std_arr);
        Ok(())
    }

    /// Transform the data using fitted parameters.
    pub fn transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let mean = self.mean.as_ref().ok_or_else(|| {
            CoreError::InvalidState(ErrorContext::new("StandardScaler not fitted"))
        })?;
        let std_dev = self.std_dev.as_ref().ok_or_else(|| {
            CoreError::InvalidState(ErrorContext::new("StandardScaler not fitted"))
        })?;
        if data.ncols() != mean.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                mean.len(),
                data.ncols()
            ))));
        }
        let mut result = data.clone();
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        for j in 0..data.ncols() {
            for i in 0..data.nrows() {
                let mut val = result[[i, j]];
                if self.with_mean {
                    val = val - mean[j];
                }
                if self.with_std {
                    let s = if std_dev[j] < eps {
                        F::one()
                    } else {
                        std_dev[j]
                    };
                    val = val / s;
                }
                result[[i, j]] = val;
            }
        }
        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse-transform scaled data back to original scale.
    pub fn inverse_transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let mean = self.mean.as_ref().ok_or_else(|| {
            CoreError::InvalidState(ErrorContext::new("StandardScaler not fitted"))
        })?;
        let std_dev = self.std_dev.as_ref().ok_or_else(|| {
            CoreError::InvalidState(ErrorContext::new("StandardScaler not fitted"))
        })?;
        if data.ncols() != mean.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                mean.len(),
                data.ncols()
            ))));
        }
        let mut result = data.clone();
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        for j in 0..data.ncols() {
            for i in 0..data.nrows() {
                let mut val = result[[i, j]];
                if self.with_std {
                    let s = if std_dev[j] < eps {
                        F::one()
                    } else {
                        std_dev[j]
                    };
                    val = val * s;
                }
                if self.with_mean {
                    val = val + mean[j];
                }
                result[[i, j]] = val;
            }
        }
        Ok(result)
    }
}

impl<F: Float + FromPrimitive + Debug + Display + std::iter::Sum> Default for StandardScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MinMaxScaler
// ---------------------------------------------------------------------------

/// Scale features to a given range [feature_min, feature_max] (default [0, 1]).
///
/// X_scaled = (X - X_min) / (X_max - X_min) * (feature_max - feature_min) + feature_min
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::MinMaxScaler;
/// use ndarray::array;
///
/// let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
/// let mut scaler = MinMaxScaler::<f64>::new(0.0, 1.0);
/// scaler.fit(&data).expect("fit failed");
/// let scaled = scaler.transform(&data).expect("transform failed");
/// ```
#[derive(Debug, Clone)]
pub struct MinMaxScaler<F: Float> {
    /// Per-feature minimum from training data
    pub data_min: Option<Array1<F>>,
    /// Per-feature maximum from training data
    pub data_max: Option<Array1<F>>,
    /// Target range minimum
    pub feature_min: F,
    /// Target range maximum
    pub feature_max: F,
}

impl<F: Float + FromPrimitive + Debug + Display> MinMaxScaler<F> {
    /// Create a new MinMaxScaler with target range [feature_min, feature_max].
    #[must_use]
    pub fn new(feature_min: F, feature_max: F) -> Self {
        Self {
            data_min: None,
            data_max: None,
            feature_min,
            feature_max,
        }
    }

    /// Create a scaler mapping to [0, 1].
    #[must_use]
    pub fn unit_range() -> Self {
        Self::new(F::zero(), F::one())
    }

    /// Fit the scaler from data.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        if data.nrows() == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit MinMaxScaler on empty data",
            )));
        }
        let n_cols = data.ncols();
        let mut mins = Array1::<F>::zeros(n_cols);
        let mut maxs = Array1::<F>::zeros(n_cols);
        for j in 0..n_cols {
            let col = data.column(j);
            let mut col_min = F::infinity();
            let mut col_max = F::neg_infinity();
            for &v in col.iter() {
                if v < col_min {
                    col_min = v;
                }
                if v > col_max {
                    col_max = v;
                }
            }
            mins[j] = col_min;
            maxs[j] = col_max;
        }
        self.data_min = Some(mins);
        self.data_max = Some(maxs);
        Ok(())
    }

    /// Transform data.
    pub fn transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let d_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MinMaxScaler not fitted")))?;
        let d_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MinMaxScaler not fitted")))?;
        if data.ncols() != d_min.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                d_min.len(),
                data.ncols()
            ))));
        }
        let range = self.feature_max - self.feature_min;
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        let mut result = data.clone();
        for j in 0..data.ncols() {
            let data_range = d_max[j] - d_min[j];
            let scale = if data_range.abs() < eps {
                F::zero()
            } else {
                range / data_range
            };
            for i in 0..data.nrows() {
                result[[i, j]] = (result[[i, j]] - d_min[j]) * scale + self.feature_min;
            }
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let d_min = self
            .data_min
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MinMaxScaler not fitted")))?;
        let d_max = self
            .data_max
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MinMaxScaler not fitted")))?;
        if data.ncols() != d_min.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                d_min.len(),
                data.ncols()
            ))));
        }
        let range = self.feature_max - self.feature_min;
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        let mut result = data.clone();
        for j in 0..data.ncols() {
            let data_range = d_max[j] - d_min[j];
            let scale = if range.abs() < eps {
                F::zero()
            } else {
                data_range / range
            };
            for i in 0..data.nrows() {
                result[[i, j]] = (result[[i, j]] - self.feature_min) * scale + d_min[j];
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// RobustScaler
// ---------------------------------------------------------------------------

/// Scale features using statistics robust to outliers.
///
/// Uses the median and interquartile range (IQR = Q3 - Q1) so that
/// outliers have less influence than StandardScaler.
///
/// X_scaled = (X - median) / IQR
#[derive(Debug, Clone)]
pub struct RobustScaler<F: Float> {
    /// Per-feature median
    pub median: Option<Array1<F>>,
    /// Per-feature interquartile range
    pub iqr: Option<Array1<F>>,
    /// Whether to center the data
    pub with_centering: bool,
    /// Whether to scale the data
    pub with_scaling: bool,
}

impl<F: Float + FromPrimitive + Debug + Display> RobustScaler<F> {
    /// Create a new RobustScaler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            median: None,
            iqr: None,
            with_centering: true,
            with_scaling: true,
        }
    }

    /// Create with explicit options.
    #[must_use]
    pub fn with_options(with_centering: bool, with_scaling: bool) -> Self {
        Self {
            median: None,
            iqr: None,
            with_centering,
            with_scaling,
        }
    }

    /// Fit the scaler.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        if data.nrows() == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit RobustScaler on empty data",
            )));
        }
        let n_cols = data.ncols();
        let mut median_arr = Array1::<F>::zeros(n_cols);
        let mut iqr_arr = Array1::<F>::zeros(n_cols);
        for j in 0..n_cols {
            let mut col_vals: Vec<F> = data.column(j).iter().copied().collect();
            col_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col_vals.len();
            median_arr[j] = compute_quantile(&col_vals, F::from_f64(0.5).unwrap_or_else(F::zero));
            let q1 = compute_quantile(&col_vals, F::from_f64(0.25).unwrap_or_else(F::zero));
            let q3 = compute_quantile(&col_vals, F::from_f64(0.75).unwrap_or_else(F::zero));
            iqr_arr[j] = q3 - q1;
            let _ = n; // suppress unused warning
        }
        self.median = Some(median_arr);
        self.iqr = Some(iqr_arr);
        Ok(())
    }

    /// Transform data.
    pub fn transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let med = self
            .median
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("RobustScaler not fitted")))?;
        let iqr = self
            .iqr
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("RobustScaler not fitted")))?;
        if data.ncols() != med.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                med.len(),
                data.ncols()
            ))));
        }
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        let mut result = data.clone();
        for j in 0..data.ncols() {
            for i in 0..data.nrows() {
                let mut val = result[[i, j]];
                if self.with_centering {
                    val = val - med[j];
                }
                if self.with_scaling {
                    let s = if iqr[j].abs() < eps { F::one() } else { iqr[j] };
                    val = val / s;
                }
                result[[i, j]] = val;
            }
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        self.fit(data)?;
        self.transform(data)
    }
}

impl<F: Float + FromPrimitive + Debug + Display> Default for RobustScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MaxAbsScaler
// ---------------------------------------------------------------------------

/// Scale each feature by its maximum absolute value so values are in [-1, 1].
///
/// X_scaled = X / max(|X|)
#[derive(Debug, Clone)]
pub struct MaxAbsScaler<F: Float> {
    /// Per-feature maximum absolute value
    pub max_abs: Option<Array1<F>>,
}

impl<F: Float + FromPrimitive + Debug + Display> MaxAbsScaler<F> {
    /// Create a new MaxAbsScaler.
    #[must_use]
    pub fn new() -> Self {
        Self { max_abs: None }
    }

    /// Fit the scaler.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        if data.nrows() == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit MaxAbsScaler on empty data",
            )));
        }
        let n_cols = data.ncols();
        let mut max_abs_arr = Array1::<F>::zeros(n_cols);
        for j in 0..n_cols {
            let mut ma = F::zero();
            for &v in data.column(j).iter() {
                let av = v.abs();
                if av > ma {
                    ma = av;
                }
            }
            max_abs_arr[j] = ma;
        }
        self.max_abs = Some(max_abs_arr);
        Ok(())
    }

    /// Transform data.
    pub fn transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let ma = self
            .max_abs
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MaxAbsScaler not fitted")))?;
        if data.ncols() != ma.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                ma.len(),
                data.ncols()
            ))));
        }
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        let mut result = data.clone();
        for j in 0..data.ncols() {
            let s = if ma[j].abs() < eps { F::one() } else { ma[j] };
            for i in 0..data.nrows() {
                result[[i, j]] = result[[i, j]] / s;
            }
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let ma = self
            .max_abs
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("MaxAbsScaler not fitted")))?;
        if data.ncols() != ma.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                ma.len(),
                data.ncols()
            ))));
        }
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        let mut result = data.clone();
        for j in 0..data.ncols() {
            let s = if ma[j].abs() < eps { F::one() } else { ma[j] };
            for i in 0..data.nrows() {
                result[[i, j]] = result[[i, j]] * s;
            }
        }
        Ok(result)
    }
}

impl<F: Float + FromPrimitive + Debug + Display> Default for MaxAbsScaler<F> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LabelEncoder
// ---------------------------------------------------------------------------

/// Encode string (or any hashable) labels as integer indices.
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::LabelEncoder;
///
/// let labels = vec!["cat", "dog", "cat", "bird"];
/// let mut enc = LabelEncoder::new();
/// enc.fit(&labels);
/// let encoded = enc.transform(&labels).expect("transform failed");
/// assert_eq!(encoded.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct LabelEncoder<L: Eq + Hash + Clone> {
    /// Mapping from label to integer
    pub label_to_int: HashMap<L, usize>,
    /// Mapping from integer back to label
    pub int_to_label: Vec<L>,
}

impl<L: Eq + Hash + Clone + Debug> LabelEncoder<L> {
    /// Create a new empty LabelEncoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            label_to_int: HashMap::new(),
            int_to_label: Vec::new(),
        }
    }

    /// Fit the encoder by learning the unique labels.
    /// Labels are assigned indices in the order they first appear.
    pub fn fit(&mut self, labels: &[L]) {
        self.label_to_int.clear();
        self.int_to_label.clear();
        for label in labels {
            if !self.label_to_int.contains_key(label) {
                let idx = self.int_to_label.len();
                self.label_to_int.insert(label.clone(), idx);
                self.int_to_label.push(label.clone());
            }
        }
    }

    /// Transform labels to integer indices.
    pub fn transform(&self, labels: &[L]) -> CoreResult<Vec<usize>> {
        if self.int_to_label.is_empty() {
            return Err(CoreError::InvalidState(ErrorContext::new(
                "LabelEncoder not fitted",
            )));
        }
        let mut result = Vec::with_capacity(labels.len());
        for label in labels {
            let idx = self.label_to_int.get(label).ok_or_else(|| {
                CoreError::ValueError(ErrorContext::new(format!(
                    "Unknown label encountered: {:?}",
                    label
                )))
            })?;
            result.push(*idx);
        }
        Ok(result)
    }

    /// Inverse-transform integer indices back to labels.
    pub fn inverse_transform(&self, indices: &[usize]) -> CoreResult<Vec<L>> {
        let mut result = Vec::with_capacity(indices.len());
        for &idx in indices {
            if idx >= self.int_to_label.len() {
                return Err(CoreError::IndexError(ErrorContext::new(format!(
                    "Label index {} out of range (max {})",
                    idx,
                    self.int_to_label.len().saturating_sub(1)
                ))));
            }
            result.push(self.int_to_label[idx].clone());
        }
        Ok(result)
    }

    /// Fit and transform in one step.
    pub fn fit_transform(&mut self, labels: &[L]) -> Vec<usize> {
        self.fit(labels);
        // After fit, transform should always succeed for the training data
        labels.iter().map(|l| self.label_to_int[l]).collect()
    }

    /// Return the number of unique classes.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.int_to_label.len()
    }
}

impl<L: Eq + Hash + Clone + Debug> Default for LabelEncoder<L> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OneHotEncoder
// ---------------------------------------------------------------------------

/// One-hot encode categorical features.
///
/// Each unique category for a feature becomes a binary column.
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::OneHotEncoder;
///
/// let data = vec![vec!["red", "small"], vec!["blue", "large"], vec!["red", "large"]];
/// let mut enc = OneHotEncoder::new();
/// enc.fit(&data);
/// let encoded = enc.transform(&data).expect("transform failed");
/// assert_eq!(encoded.ncols(), 4); // red, blue, small, large
/// ```
#[derive(Debug, Clone)]
pub struct OneHotEncoder<L: Eq + Hash + Clone> {
    /// Per-feature category encoders
    pub encoders: Vec<LabelEncoder<L>>,
    /// Number of features
    pub n_features: usize,
}

impl<L: Eq + Hash + Clone + Debug> OneHotEncoder<L> {
    /// Create a new OneHotEncoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit from a 2D vector of labels (rows x features).
    pub fn fit(&mut self, data: &[Vec<L>]) {
        if data.is_empty() {
            self.n_features = 0;
            self.encoders.clear();
            return;
        }
        self.n_features = data[0].len();
        self.encoders.clear();
        for j in 0..self.n_features {
            let mut enc = LabelEncoder::new();
            let col: Vec<L> = data.iter().map(|row| row[j].clone()).collect();
            enc.fit(&col);
            self.encoders.push(enc);
        }
    }

    /// Transform data to a one-hot encoded `Array2<f64>`.
    pub fn transform(&self, data: &[Vec<L>]) -> CoreResult<Array2<f64>> {
        if self.encoders.is_empty() {
            return Err(CoreError::InvalidState(ErrorContext::new(
                "OneHotEncoder not fitted",
            )));
        }
        let total_cols: usize = self.encoders.iter().map(|e| e.n_classes()).sum();
        let n_rows = data.len();
        let mut result = Array2::<f64>::zeros((n_rows, total_cols));
        let mut col_offset = 0;
        for (j, enc) in self.encoders.iter().enumerate() {
            let col_labels: Vec<L> = data.iter().map(|row| row[j].clone()).collect();
            let indices = enc.transform(&col_labels)?;
            for (i, idx) in indices.into_iter().enumerate() {
                result[[i, col_offset + idx]] = 1.0;
            }
            col_offset += enc.n_classes();
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &[Vec<L>]) -> CoreResult<Array2<f64>> {
        self.fit(data);
        self.transform(data)
    }

    /// Total number of output columns.
    #[must_use]
    pub fn n_output_features(&self) -> usize {
        self.encoders.iter().map(|e| e.n_classes()).sum()
    }
}

impl<L: Eq + Hash + Clone + Debug> Default for OneHotEncoder<L> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OrdinalEncoder
// ---------------------------------------------------------------------------

/// Encode categorical features as ordinal integers.
///
/// Each feature's categories are mapped to 0, 1, 2, ... in the order
/// they first appear during fitting.
#[derive(Debug, Clone)]
pub struct OrdinalEncoder<L: Eq + Hash + Clone> {
    /// Per-feature label encoders
    pub encoders: Vec<LabelEncoder<L>>,
    /// Number of features
    pub n_features: usize,
}

impl<L: Eq + Hash + Clone + Debug> OrdinalEncoder<L> {
    /// Create a new OrdinalEncoder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoders: Vec::new(),
            n_features: 0,
        }
    }

    /// Fit from 2D data.
    pub fn fit(&mut self, data: &[Vec<L>]) {
        if data.is_empty() {
            self.n_features = 0;
            self.encoders.clear();
            return;
        }
        self.n_features = data[0].len();
        self.encoders.clear();
        for j in 0..self.n_features {
            let mut enc = LabelEncoder::new();
            let col: Vec<L> = data.iter().map(|row| row[j].clone()).collect();
            enc.fit(&col);
            self.encoders.push(enc);
        }
    }

    /// Transform to ordinal-encoded `Array2<usize>`.
    pub fn transform(&self, data: &[Vec<L>]) -> CoreResult<Vec<Vec<usize>>> {
        if self.encoders.is_empty() {
            return Err(CoreError::InvalidState(ErrorContext::new(
                "OrdinalEncoder not fitted",
            )));
        }
        let n_rows = data.len();
        let mut result = vec![vec![0usize; self.n_features]; n_rows];
        for (j, enc) in self.encoders.iter().enumerate() {
            let col_labels: Vec<L> = data.iter().map(|row| row[j].clone()).collect();
            let indices = enc.transform(&col_labels)?;
            for (i, idx) in indices.into_iter().enumerate() {
                result[i][j] = idx;
            }
        }
        Ok(result)
    }

    /// Inverse transform.
    pub fn inverse_transform(&self, data: &[Vec<usize>]) -> CoreResult<Vec<Vec<L>>> {
        if self.encoders.is_empty() {
            return Err(CoreError::InvalidState(ErrorContext::new(
                "OrdinalEncoder not fitted",
            )));
        }
        let n_rows = data.len();
        let mut result: Vec<Vec<L>> = Vec::with_capacity(n_rows);
        for row in data {
            let mut out_row = Vec::with_capacity(self.n_features);
            for (j, enc) in self.encoders.iter().enumerate() {
                let labels = enc.inverse_transform(&[row[j]])?;
                out_row.push(labels.into_iter().next().ok_or_else(|| {
                    CoreError::ValueError(ErrorContext::new("Empty inverse_transform result"))
                })?);
            }
            result.push(out_row);
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &[Vec<L>]) -> CoreResult<Vec<Vec<usize>>> {
        self.fit(data);
        self.transform(data)
    }
}

impl<L: Eq + Hash + Clone + Debug> Default for OrdinalEncoder<L> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Imputer
// ---------------------------------------------------------------------------

/// Strategy for imputing missing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputeStrategy {
    /// Replace with the column mean
    Mean,
    /// Replace with the column median
    Median,
    /// Replace with the column mode (most frequent value, discretized)
    Mode,
    /// Replace with a constant value
    Constant,
}

/// Impute missing values in numeric data.
///
/// Missing values are represented as NaN. The imputer learns fill values
/// during `fit` and applies them during `transform`.
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::{Imputer, ImputeStrategy};
/// use ndarray::array;
///
/// let data = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
/// let mut imp = Imputer::<f64>::new(ImputeStrategy::Mean, None);
/// imp.fit(&data).expect("fit failed");
/// let filled = imp.transform(&data).expect("transform failed");
/// assert!(!filled[[0, 1]].is_nan());
/// ```
#[derive(Debug, Clone)]
pub struct Imputer<F: Float> {
    /// Imputation strategy
    pub strategy: ImputeStrategy,
    /// Fill values per feature (computed during fit)
    pub fill_values: Option<Array1<F>>,
    /// Constant fill value (used when strategy == Constant)
    pub fill_constant: F,
}

impl<F: Float + FromPrimitive + Debug + Display + std::iter::Sum> Imputer<F> {
    /// Create a new Imputer.
    ///
    /// `fill_constant` is only used when `strategy == ImputeStrategy::Constant`.
    #[must_use]
    pub fn new(strategy: ImputeStrategy, fill_constant: Option<F>) -> Self {
        Self {
            strategy,
            fill_values: None,
            fill_constant: fill_constant.unwrap_or_else(F::zero),
        }
    }

    /// Fit the imputer.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        if data.nrows() == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit Imputer on empty data",
            )));
        }
        let n_cols = data.ncols();
        let mut fill_vals = Array1::<F>::zeros(n_cols);
        for j in 0..n_cols {
            let col = data.column(j);
            let valid: Vec<F> = col.iter().copied().filter(|v| !v.is_nan()).collect();
            if valid.is_empty() {
                fill_vals[j] = self.fill_constant;
                continue;
            }
            match self.strategy {
                ImputeStrategy::Mean => {
                    let n = F::from_usize(valid.len()).unwrap_or_else(F::one);
                    let s: F = valid.iter().copied().sum();
                    fill_vals[j] = s / n;
                }
                ImputeStrategy::Median => {
                    let mut sorted = valid.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    fill_vals[j] =
                        compute_quantile(&sorted, F::from_f64(0.5).unwrap_or_else(F::zero));
                }
                ImputeStrategy::Mode => {
                    // Discretize to find mode: bucket by rounding to ~6 decimal places
                    let factor = F::from_f64(1e6).unwrap_or_else(F::one);
                    let mut counts: HashMap<i64, (usize, F)> = HashMap::new();
                    for &v in &valid {
                        let key = NumCast::from(v * factor)
                            .map(|x: f64| x.round() as i64)
                            .unwrap_or(0);
                        let entry = counts.entry(key).or_insert((0, v));
                        entry.0 += 1;
                    }
                    let mode_val = counts
                        .values()
                        .max_by_key(|(count, _)| *count)
                        .map(|(_, v)| *v)
                        .unwrap_or_else(F::zero);
                    fill_vals[j] = mode_val;
                }
                ImputeStrategy::Constant => {
                    fill_vals[j] = self.fill_constant;
                }
            }
        }
        self.fill_values = Some(fill_vals);
        Ok(())
    }

    /// Transform data by replacing NaN with imputed values.
    pub fn transform(&self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        let fill = self
            .fill_values
            .as_ref()
            .ok_or_else(|| CoreError::InvalidState(ErrorContext::new("Imputer not fitted")))?;
        if data.ncols() != fill.len() {
            return Err(CoreError::DimensionError(ErrorContext::new(format!(
                "Expected {} features, got {}",
                fill.len(),
                data.ncols()
            ))));
        }
        let mut result = data.clone();
        for j in 0..data.ncols() {
            for i in 0..data.nrows() {
                if result[[i, j]].is_nan() {
                    result[[i, j]] = fill[j];
                }
            }
        }
        Ok(result)
    }

    /// Fit and transform.
    pub fn fit_transform(&mut self, data: &Array2<F>) -> CoreResult<Array2<F>> {
        self.fit(data)?;
        self.transform(data)
    }
}

// ---------------------------------------------------------------------------
// Outlier Detection
// ---------------------------------------------------------------------------

/// Method used for outlier detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierMethod {
    /// Z-score method: outliers have |z| > threshold (default 3.0)
    ZScore,
    /// IQR method: outliers are below Q1 - factor*IQR or above Q3 + factor*IQR (default factor 1.5)
    Iqr,
}

/// Detect outliers in numeric data.
///
/// # Example
///
/// ```
/// use scirs2_core::preprocessing::{OutlierDetector, OutlierMethod};
/// use ndarray::array;
///
/// // IQR method: Q1=1.5, Q3=3.0, IQR=1.5, upper fence=3+1.5*1.5=5.25 → 100.0 is outlier
/// let data = array![[1.0f64], [2.0], [3.0], [100.0]];
/// let mut det = OutlierDetector::<f64>::new(OutlierMethod::Iqr, 1.5);
/// det.fit(&data).expect("fit failed");
/// let mask = det.detect(&data).expect("detect failed");
/// assert!(mask[3]); // 100.0 is an outlier (above upper fence of 5.25)
/// ```
#[derive(Debug, Clone)]
pub struct OutlierDetector<F: Float> {
    /// Detection method
    pub method: OutlierMethod,
    /// Threshold / factor parameter
    pub threshold: F,
    /// Fitted parameters for ZScore: (mean, std) per feature
    zscore_params: Option<Vec<(F, F)>>,
    /// Fitted parameters for IQR: (Q1, Q3, IQR) per feature
    iqr_params: Option<Vec<(F, F, F)>>,
}

impl<F: Float + FromPrimitive + Debug + Display + std::iter::Sum> OutlierDetector<F> {
    /// Create a new OutlierDetector.
    ///
    /// For ZScore, `threshold` is the z-score threshold (e.g. 3.0).
    /// For IQR, `threshold` is the IQR factor (e.g. 1.5).
    #[must_use]
    pub fn new(method: OutlierMethod, threshold: F) -> Self {
        Self {
            method,
            threshold,
            zscore_params: None,
            iqr_params: None,
        }
    }

    /// Fit the detector.
    pub fn fit(&mut self, data: &Array2<F>) -> CoreResult<()> {
        if data.nrows() == 0 {
            return Err(CoreError::ValueError(ErrorContext::new(
                "Cannot fit OutlierDetector on empty data",
            )));
        }
        let n_cols = data.ncols();
        match self.method {
            OutlierMethod::ZScore => {
                let mut params = Vec::with_capacity(n_cols);
                for j in 0..n_cols {
                    let col = data.column(j);
                    let n = F::from_usize(col.len()).unwrap_or_else(F::one);
                    let sum: F = col.iter().copied().sum();
                    let mean = sum / n;
                    let var_sum: F = col.iter().map(|&x| (x - mean) * (x - mean)).sum();
                    let std_dev = (var_sum / n).sqrt();
                    params.push((mean, std_dev));
                }
                self.zscore_params = Some(params);
            }
            OutlierMethod::Iqr => {
                let mut params = Vec::with_capacity(n_cols);
                for j in 0..n_cols {
                    let mut sorted: Vec<F> = data.column(j).iter().copied().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let q1 = compute_quantile(&sorted, F::from_f64(0.25).unwrap_or_else(F::zero));
                    let q3 = compute_quantile(&sorted, F::from_f64(0.75).unwrap_or_else(F::zero));
                    let iqr = q3 - q1;
                    params.push((q1, q3, iqr));
                }
                self.iqr_params = Some(params);
            }
        }
        Ok(())
    }

    /// Detect outliers. Returns a boolean mask where `true` means outlier.
    ///
    /// A sample is considered an outlier if ANY of its features is an outlier.
    pub fn detect(&self, data: &Array2<F>) -> CoreResult<Vec<bool>> {
        let n_rows = data.nrows();
        let mut mask = vec![false; n_rows];
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        match self.method {
            OutlierMethod::ZScore => {
                let params = self.zscore_params.as_ref().ok_or_else(|| {
                    CoreError::InvalidState(ErrorContext::new("OutlierDetector not fitted"))
                })?;
                for j in 0..data.ncols() {
                    let (mean, std_dev) = params[j];
                    let s = if std_dev.abs() < eps {
                        F::one()
                    } else {
                        std_dev
                    };
                    for i in 0..n_rows {
                        let z = (data[[i, j]] - mean) / s;
                        if z.abs() > self.threshold {
                            mask[i] = true;
                        }
                    }
                }
            }
            OutlierMethod::Iqr => {
                let params = self.iqr_params.as_ref().ok_or_else(|| {
                    CoreError::InvalidState(ErrorContext::new("OutlierDetector not fitted"))
                })?;
                for j in 0..data.ncols() {
                    let (q1, q3, iqr) = params[j];
                    let lower = q1 - self.threshold * iqr;
                    let upper = q3 + self.threshold * iqr;
                    for i in 0..n_rows {
                        let v = data[[i, j]];
                        if v < lower || v > upper {
                            mask[i] = true;
                        }
                    }
                }
            }
        }
        Ok(mask)
    }

    /// Detect per-feature outlier masks. Returns `Array2<bool>` with same shape as data.
    pub fn detect_per_feature(&self, data: &Array2<F>) -> CoreResult<Array2<bool>> {
        let n_rows = data.nrows();
        let n_cols = data.ncols();
        let mut mask = Array2::<bool>::default((n_rows, n_cols));
        let eps = F::from_f64(1e-10).unwrap_or_else(F::epsilon);
        match self.method {
            OutlierMethod::ZScore => {
                let params = self.zscore_params.as_ref().ok_or_else(|| {
                    CoreError::InvalidState(ErrorContext::new("OutlierDetector not fitted"))
                })?;
                for j in 0..n_cols {
                    let (mean, std_dev) = params[j];
                    let s = if std_dev.abs() < eps {
                        F::one()
                    } else {
                        std_dev
                    };
                    for i in 0..n_rows {
                        let z = (data[[i, j]] - mean) / s;
                        mask[[i, j]] = z.abs() > self.threshold;
                    }
                }
            }
            OutlierMethod::Iqr => {
                let params = self.iqr_params.as_ref().ok_or_else(|| {
                    CoreError::InvalidState(ErrorContext::new("OutlierDetector not fitted"))
                })?;
                for j in 0..n_cols {
                    let (q1, q3, iqr) = params[j];
                    let lower = q1 - self.threshold * iqr;
                    let upper = q3 + self.threshold * iqr;
                    for i in 0..n_rows {
                        let v = data[[i, j]];
                        mask[[i, j]] = v < lower || v > upper;
                    }
                }
            }
        }
        Ok(mask)
    }
}

// ---------------------------------------------------------------------------
// Helper: quantile computation
// ---------------------------------------------------------------------------

/// Compute a quantile (0..1) from a sorted slice using linear interpolation.
fn compute_quantile<F: Float + FromPrimitive>(sorted: &[F], q: F) -> F {
    if sorted.is_empty() {
        return F::zero();
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let n = sorted.len();
    let idx_f = q * F::from_usize(n - 1).unwrap_or_else(F::zero);
    let lower = NumCast::from(idx_f.floor()).unwrap_or(0usize);
    let upper = NumCast::from(idx_f.ceil()).unwrap_or(n - 1);
    let lower = lower.min(n - 1);
    let upper = upper.min(n - 1);
    if lower == upper {
        return sorted[lower];
    }
    let frac = idx_f - F::from_usize(lower).unwrap_or_else(F::zero);
    sorted[lower] * (F::one() - frac) + sorted[upper] * frac
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ::ndarray::array;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_standard_scaler_basic() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = StandardScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let transformed = scaler.transform(&data).expect("transform");
        // Mean of each column should be ~0
        for j in 0..2 {
            let col_mean: f64 = transformed.column(j).iter().sum::<f64>() / 3.0;
            assert!(col_mean.abs() < EPS, "col {} mean = {}", j, col_mean);
        }
    }

    #[test]
    fn test_standard_scaler_inverse() {
        let data = array![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]];
        let mut scaler = StandardScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        let inv = scaler.inverse_transform(&t).expect("inverse");
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    (inv[[i, j]] - data[[i, j]]).abs() < EPS,
                    "mismatch at [{}, {}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_standard_scaler_empty_error() {
        let data = Array2::<f64>::zeros((0, 3));
        let mut scaler = StandardScaler::<f64>::new();
        assert!(scaler.fit(&data).is_err());
    }

    #[test]
    fn test_minmax_scaler_basic() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let mut scaler = MinMaxScaler::<f64>::new(0.0, 1.0);
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        assert!((t[[0, 0]] - 0.0).abs() < EPS);
        assert!((t[[2, 0]] - 1.0).abs() < EPS);
        assert!((t[[0, 1]] - 0.0).abs() < EPS);
        assert!((t[[2, 1]] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_minmax_scaler_custom_range() {
        let data = array![[0.0], [5.0], [10.0]];
        let mut scaler = MinMaxScaler::<f64>::new(-1.0, 1.0);
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        assert!((t[[0, 0]] - (-1.0)).abs() < EPS);
        assert!((t[[1, 0]] - 0.0).abs() < EPS);
        assert!((t[[2, 0]] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_minmax_scaler_inverse() {
        let data = array![[2.0, 4.0], [6.0, 8.0]];
        let mut scaler = MinMaxScaler::<f64>::new(0.0, 1.0);
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        let inv = scaler.inverse_transform(&t).expect("inverse");
        for i in 0..2 {
            for j in 0..2 {
                assert!((inv[[i, j]] - data[[i, j]]).abs() < EPS);
            }
        }
    }

    #[test]
    fn test_robust_scaler_basic() {
        let data = array![[1.0], [2.0], [3.0], [4.0], [100.0]];
        let mut scaler = RobustScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        // The median is 3.0 so the third element should be 0
        assert!((t[[2, 0]]).abs() < EPS);
    }

    #[test]
    fn test_max_abs_scaler_basic() {
        let data = array![[-3.0, 2.0], [1.0, -5.0]];
        let mut scaler = MaxAbsScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        // Max abs of col0 = 3, col1 = 5
        assert!((t[[0, 0]] - (-1.0)).abs() < EPS);
        assert!((t[[1, 1]] - (-1.0)).abs() < EPS);
    }

    #[test]
    fn test_max_abs_scaler_inverse() {
        let data = array![[4.0, -8.0], [-2.0, 6.0]];
        let mut scaler = MaxAbsScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        let inv = scaler.inverse_transform(&t).expect("inverse");
        for i in 0..2 {
            for j in 0..2 {
                assert!((inv[[i, j]] - data[[i, j]]).abs() < EPS);
            }
        }
    }

    #[test]
    fn test_label_encoder() {
        let labels = vec!["cat", "dog", "cat", "bird", "dog"];
        let mut enc = LabelEncoder::new();
        enc.fit(&labels);
        assert_eq!(enc.n_classes(), 3);
        let encoded = enc.transform(&labels).expect("transform");
        assert_eq!(encoded[0], encoded[2]); // cat == cat
        assert_eq!(encoded[1], encoded[4]); // dog == dog
        let decoded = enc.inverse_transform(&encoded).expect("inverse");
        assert_eq!(decoded, labels);
    }

    #[test]
    fn test_label_encoder_unknown() {
        let labels = vec!["a", "b"];
        let mut enc = LabelEncoder::new();
        enc.fit(&labels);
        let result = enc.transform(&["c"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoder() {
        let data = vec![
            vec!["red", "small"],
            vec!["blue", "large"],
            vec!["red", "large"],
        ];
        let mut enc = OneHotEncoder::new();
        enc.fit(&data);
        let encoded = enc.transform(&data).expect("transform");
        assert_eq!(encoded.nrows(), 3);
        assert_eq!(encoded.ncols(), 4); // 2 colors + 2 sizes
                                        // Each row should sum to 2 (one per feature)
        for i in 0..3 {
            let row_sum: f64 = encoded.row(i).iter().sum();
            assert!((row_sum - 2.0).abs() < EPS);
        }
    }

    #[test]
    fn test_ordinal_encoder() {
        let data = vec![vec!["a", "x"], vec!["b", "y"], vec!["a", "y"]];
        let mut enc = OrdinalEncoder::new();
        let encoded = enc.fit_transform(&data).expect("transform");
        assert_eq!(encoded[0][0], encoded[2][0]); // a == a
        let decoded = enc.inverse_transform(&encoded).expect("inverse");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_imputer_mean() {
        let data = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];
        let mut imp = Imputer::<f64>::new(ImputeStrategy::Mean, None);
        imp.fit(&data).expect("fit");
        let filled = imp.transform(&data).expect("transform");
        assert!(!filled[[0, 1]].is_nan());
        // Mean of col1 valid values: (4+6)/2 = 5
        assert!((filled[[0, 1]] - 5.0).abs() < EPS);
    }

    #[test]
    fn test_imputer_median() {
        let data = array![[f64::NAN, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
        let mut imp = Imputer::<f64>::new(ImputeStrategy::Median, None);
        imp.fit(&data).expect("fit");
        let filled = imp.transform(&data).expect("transform");
        assert!(!filled[[0, 0]].is_nan());
        // Median of [2,4,6] = 4
        assert!((filled[[0, 0]] - 4.0).abs() < EPS);
    }

    #[test]
    fn test_imputer_constant() {
        let data = array![[1.0, f64::NAN], [f64::NAN, 4.0]];
        let mut imp = Imputer::<f64>::new(ImputeStrategy::Constant, Some(-999.0));
        imp.fit(&data).expect("fit");
        let filled = imp.transform(&data).expect("transform");
        assert!((filled[[0, 1]] - (-999.0)).abs() < EPS);
        assert!((filled[[1, 0]] - (-999.0)).abs() < EPS);
    }

    #[test]
    fn test_outlier_zscore() {
        // Use enough normal-range samples so the outlier has a clear z-score
        let data = array![
            [1.0],
            [2.0],
            [3.0],
            [2.0],
            [1.5],
            [2.5],
            [3.0],
            [2.0],
            [1.0],
            [2.0],
            [3.0],
            [2.5],
            [1.5],
            [2.0],
            [2.5],
            [100.0]
        ];
        let mut det = OutlierDetector::<f64>::new(OutlierMethod::ZScore, 2.0);
        det.fit(&data).expect("fit");
        let mask = det.detect(&data).expect("detect");
        // 100 should be an outlier (last element, index 15)
        assert!(mask[15]);
        // Normal values should not be outliers
        assert!(!mask[0]);
        assert!(!mask[1]);
    }

    #[test]
    fn test_outlier_iqr() {
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0], [100.0]];
        let mut det = OutlierDetector::<f64>::new(OutlierMethod::Iqr, 1.5);
        det.fit(&data).expect("fit");
        let mask = det.detect(&data).expect("detect");
        assert!(mask[5]); // 100 is outlier
        assert!(!mask[0]);
    }

    #[test]
    fn test_outlier_per_feature() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 100.0]];
        let mut det = OutlierDetector::<f64>::new(OutlierMethod::ZScore, 1.0);
        det.fit(&data).expect("fit");
        let mask = det.detect_per_feature(&data).expect("detect");
        assert_eq!(mask.nrows(), 3);
        assert_eq!(mask.ncols(), 2);
    }

    #[test]
    fn test_standard_scaler_f32() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = StandardScaler::<f32>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        let col_mean: f32 = t.column(0).iter().sum::<f32>() / 3.0;
        assert!(col_mean.abs() < 1e-4);
    }

    #[test]
    fn test_compute_quantile() {
        let sorted = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert!((compute_quantile(&sorted, 0.0) - 1.0).abs() < EPS);
        assert!((compute_quantile(&sorted, 0.5) - 3.0).abs() < EPS);
        assert!((compute_quantile(&sorted, 1.0) - 5.0).abs() < EPS);
        assert!((compute_quantile(&sorted, 0.25) - 2.0).abs() < EPS);
    }

    #[test]
    fn test_constant_feature_standard_scaler() {
        // All values the same => std=0, should not produce NaN
        let data = array![[5.0], [5.0], [5.0]];
        let mut scaler = StandardScaler::<f64>::new();
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        assert!(!t[[0, 0]].is_nan());
    }

    #[test]
    fn test_constant_feature_minmax() {
        let data = array![[5.0], [5.0], [5.0]];
        let mut scaler = MinMaxScaler::<f64>::new(0.0, 1.0);
        scaler.fit(&data).expect("fit");
        let t = scaler.transform(&data).expect("transform");
        assert!(!t[[0, 0]].is_nan());
    }

    #[test]
    fn test_fit_transform_shortcut() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut scaler = StandardScaler::<f64>::new();
        let t = scaler.fit_transform(&data).expect("fit_transform");
        assert_eq!(t.shape(), &[2, 2]);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let train = array![[1.0, 2.0], [3.0, 4.0]];
        let test_data = array![[1.0, 2.0, 3.0]];
        let mut scaler = StandardScaler::<f64>::new();
        scaler.fit(&train).expect("fit");
        assert!(scaler.transform(&test_data).is_err());
    }

    #[test]
    fn test_not_fitted_error() {
        let data = array![[1.0]];
        let scaler = StandardScaler::<f64>::new();
        assert!(scaler.transform(&data).is_err());
    }

    #[test]
    fn test_label_encoder_fit_transform() {
        let labels = vec![10, 20, 30, 20, 10];
        let mut enc = LabelEncoder::new();
        let encoded = enc.fit_transform(&labels);
        assert_eq!(encoded[0], encoded[4]);
        assert_eq!(encoded[1], encoded[3]);
        assert_ne!(encoded[0], encoded[1]);
    }

    #[test]
    fn test_imputer_mode() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0], [2.0, 3.0], [3.0, 5.0]];
        let mut imp = Imputer::<f64>::new(ImputeStrategy::Mode, None);
        imp.fit(&data).expect("fit");
        let filled = imp.transform(&data).expect("transform");
        // Mode of col0 valid: 2 appears twice
        assert!(
            (filled[[0, 0]] - 1.0).abs() < EPS
                || (filled[[0, 0]] - 2.0).abs() < EPS
                || (filled[[0, 0]] - 3.0).abs() < EPS
        );
        // Mode of col1 valid: 3 appears twice
        assert!((filled[[0, 1]] - 3.0).abs() < EPS);
    }
}
