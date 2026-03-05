//! Python bindings for scirs2-transform
//!
//! This module provides Python bindings for data transformation operations,
//! including normalization, feature engineering, dimensionality reduction,
//! categorical encoding, imputation, and preprocessing pipelines.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// Direct imports from scirs2-transform
use scirs2_transform::{
    // Encoding
    encoding::{OneHotEncoder, OrdinalEncoder},
    // Features
    features::{
        binarize, discretize_equal_frequency, discretize_equal_width, log_transform,
        power_transform, PolynomialFeatures, PowerTransformer,
    },
    // Imputation
    impute::{ImputeStrategy, KNNImputer, SimpleImputer},
    // Normalization
    normalize::{normalize_array, normalize_vector, NormalizationMethod, Normalizer},
    // Reduction
    reduction::{PCA, TSNE, UMAP},
    // Scaling
    scaling::{MaxAbsScaler, QuantileTransformer},
};

// ========================================
// NORMALIZATION
// ========================================

/// Normalize array using specified method
#[pyfunction]
#[pyo3(signature = (array, method="zscore", axis=0))]
fn normalize_array_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    method: &str,
    axis: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();

    let norm_method = parse_normalization_method(method)?;

    let result = normalize_array(&arr, norm_method, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("Normalization failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Normalize vector using specified method
#[pyfunction]
#[pyo3(signature = (array, method="zscore"))]
fn normalize_vector_py(
    py: Python,
    array: &Bound<'_, PyArray1<f64>>,
    method: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();

    let norm_method = parse_normalization_method(method)?;

    let result = normalize_vector(&arr, norm_method)
        .map_err(|e| PyRuntimeError::new_err(format!("Normalization failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Normalizer class for fit/transform pattern
#[pyclass(name = "Normalizer")]
pub struct PyNormalizer {
    inner: Normalizer,
}

#[pymethods]
impl PyNormalizer {
    #[new]
    #[pyo3(signature = (method="zscore", axis=0))]
    fn new(method: &str, axis: usize) -> PyResult<Self> {
        let norm_method = parse_normalization_method(method)?;
        Ok(Self {
            inner: Normalizer::new(norm_method, axis),
        })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

// Helper to parse normalization method
fn parse_normalization_method(method: &str) -> PyResult<NormalizationMethod> {
    match method.to_lowercase().as_str() {
        "minmax" => Ok(NormalizationMethod::MinMax),
        "zscore" => Ok(NormalizationMethod::ZScore),
        "maxabs" => Ok(NormalizationMethod::MaxAbs),
        "l1" => Ok(NormalizationMethod::L1),
        "l2" => Ok(NormalizationMethod::L2),
        "robust" => Ok(NormalizationMethod::Robust),
        _ => Err(PyValueError::new_err(format!(
            "Unknown normalization method: {}",
            method
        ))),
    }
}

// ========================================
// FEATURE ENGINEERING
// ========================================

/// Polynomial Features generator
#[pyclass(name = "PolynomialFeatures")]
pub struct PyPolynomialFeatures {
    inner: PolynomialFeatures,
}

#[pymethods]
impl PyPolynomialFeatures {
    #[new]
    #[pyo3(signature = (degree=2, interaction_only=false, include_bias=true))]
    fn new(degree: usize, interaction_only: bool, include_bias: bool) -> Self {
        Self {
            inner: PolynomialFeatures::new(degree, interaction_only, include_bias),
        }
    }

    fn n_output_features(&self, n_features: usize) -> usize {
        self.inner.n_output_features(n_features)
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Polynomial transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// Power Transformer (Box-Cox, Yeo-Johnson)
#[pyclass(name = "PowerTransformer")]
pub struct PyPowerTransformer {
    inner: PowerTransformer,
}

#[pymethods]
impl PyPowerTransformer {
    #[new]
    #[pyo3(signature = (method="yeo-johnson", standardize=true))]
    fn new(method: &str, standardize: bool) -> PyResult<Self> {
        let transformer = PowerTransformer::new(method, standardize).map_err(|e| {
            PyRuntimeError::new_err(format!("PowerTransformer creation failed: {}", e))
        })?;
        Ok(Self { inner: transformer })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn inverse_transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .inverse_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Inverse transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// Binarize array with threshold
#[pyfunction]
#[pyo3(signature = (array, threshold=0.0))]
fn binarize_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    threshold: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();
    let result = binarize(&arr, threshold)
        .map_err(|e| PyRuntimeError::new_err(format!("Binarization failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Discretize with equal width bins
#[pyfunction]
#[pyo3(signature = (array, n_bins, encode="ordinal", axis=0))]
fn discretize_equal_width_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    n_bins: usize,
    encode: &str,
    axis: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();
    let result = discretize_equal_width(&arr, n_bins, encode, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("Discretization failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Discretize with equal frequency bins
#[pyfunction]
#[pyo3(signature = (array, n_bins, encode="ordinal", axis=0))]
fn discretize_equal_frequency_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    n_bins: usize,
    encode: &str,
    axis: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();
    let result = discretize_equal_frequency(&arr, n_bins, encode, axis)
        .map_err(|e| PyRuntimeError::new_err(format!("Discretization failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Log transform
#[pyfunction]
#[pyo3(signature = (array, epsilon=1e-10))]
fn log_transform_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    epsilon: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();
    let result = log_transform(&arr, epsilon)
        .map_err(|e| PyRuntimeError::new_err(format!("Log transform failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Power transform (Box-Cox or Yeo-Johnson)
#[pyfunction]
#[pyo3(signature = (array, method="yeo-johnson", standardize=true))]
fn power_transform_py(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    method: &str,
    standardize: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = array.readonly();
    let arr = binding.as_array();
    let result = power_transform(&arr, method, standardize)
        .map_err(|e| PyRuntimeError::new_err(format!("Power transform failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

// ========================================
// DIMENSIONALITY REDUCTION
// ========================================

/// PCA dimensionality reduction
#[pyclass(name = "PCA")]
pub struct PyPCA {
    inner: PCA,
}

#[pymethods]
impl PyPCA {
    #[new]
    #[pyo3(signature = (n_components=2, center=true, scale=false))]
    fn new(n_components: usize, center: bool, scale: bool) -> Self {
        Self {
            inner: PCA::new(n_components, center, scale),
        }
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("PCA fit_transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn components(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match self.inner.components() {
            Some(comp) => Ok(Some(comp.clone().into_pyarray(py).unbind())),
            None => Ok(None),
        }
    }

    fn explained_variance_ratio(&self, py: Python) -> PyResult<Option<Py<PyArray1<f64>>>> {
        match self.inner.explained_variance_ratio() {
            Some(evr) => Ok(Some(evr.clone().into_pyarray(py).unbind())),
            None => Ok(None),
        }
    }
}

/// t-SNE dimensionality reduction
#[pyclass(name = "TSNE")]
pub struct PyTSNE {
    inner: TSNE,
}

#[pymethods]
impl PyTSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0, max_iter=1000))]
    fn new(n_components: usize, perplexity: f64, max_iter: usize) -> Self {
        Self {
            inner: TSNE::new()
                .with_n_components(n_components)
                .with_perplexity(perplexity)
                .with_max_iter(max_iter),
        }
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("TSNE fit_transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// UMAP dimensionality reduction
#[pyclass(name = "UMAP")]
pub struct PyUMAP {
    inner: UMAP,
}

#[pymethods]
impl PyUMAP {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=15, min_dist=0.1, learning_rate=1.0, n_epochs=200))]
    fn new(
        n_components: usize,
        n_neighbors: usize,
        min_dist: f64,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Self {
        Self {
            inner: UMAP::new(n_neighbors, n_components, min_dist, learning_rate, n_epochs),
        }
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("UMAP fit_transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

// ========================================
// CATEGORICAL ENCODING
// ========================================

/// One-Hot Encoder
#[pyclass(name = "OneHotEncoder")]
pub struct PyOneHotEncoder {
    inner: OneHotEncoder,
}

#[pymethods]
impl PyOneHotEncoder {
    #[new]
    #[pyo3(signature = (drop=None, handle_unknown="error", sparse=false))]
    fn new(drop: Option<String>, handle_unknown: &str, sparse: bool) -> PyResult<Self> {
        let encoder = OneHotEncoder::new(drop, handle_unknown, sparse).map_err(|e| {
            PyRuntimeError::new_err(format!("OneHotEncoder creation failed: {}", e))
        })?;
        Ok(Self { inner: encoder })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.to_dense().into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.to_dense().into_pyarray(py).unbind())
    }
}

/// Ordinal Encoder
#[pyclass(name = "OrdinalEncoder")]
pub struct PyOrdinalEncoder {
    inner: OrdinalEncoder,
}

#[pymethods]
impl PyOrdinalEncoder {
    #[new]
    #[pyo3(signature = (handle_unknown="error", unknown_value=None))]
    fn new(handle_unknown: &str, unknown_value: Option<f64>) -> PyResult<Self> {
        let encoder = OrdinalEncoder::new(handle_unknown, unknown_value).map_err(|e| {
            PyRuntimeError::new_err(format!("OrdinalEncoder creation failed: {}", e))
        })?;
        Ok(Self { inner: encoder })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

// ========================================
// IMPUTATION
// ========================================

/// Simple Imputer for missing values
#[pyclass(name = "SimpleImputer")]
pub struct PySimpleImputer {
    inner: SimpleImputer,
}

#[pymethods]
impl PySimpleImputer {
    #[new]
    #[pyo3(signature = (strategy="mean", missing_values=f64::NAN))]
    fn new(strategy: &str, missing_values: f64) -> PyResult<Self> {
        let impute_strategy = match strategy.to_lowercase().as_str() {
            "mean" => ImputeStrategy::Mean,
            "median" => ImputeStrategy::Median,
            "most_frequent" => ImputeStrategy::MostFrequent,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown impute strategy: {}",
                    strategy
                )))
            }
        };

        Ok(Self {
            inner: SimpleImputer::new(impute_strategy, missing_values),
        })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// KNN Imputer for missing values
#[pyclass(name = "KNNImputer")]
pub struct PyKNNImputer {
    inner: KNNImputer,
}

#[pymethods]
impl PyKNNImputer {
    #[new]
    #[pyo3(signature = (n_neighbors=5, missing_values=f64::NAN))]
    fn new(n_neighbors: usize, missing_values: f64) -> Self {
        use scirs2_transform::impute::{DistanceMetric, WeightingScheme};
        Self {
            inner: KNNImputer::new(
                n_neighbors,
                DistanceMetric::Euclidean,
                WeightingScheme::Uniform,
                missing_values,
            ),
        }
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

// ========================================
// SCALING
// ========================================

/// Max Absolute Scaler
#[pyclass(name = "MaxAbsScaler")]
pub struct PyMaxAbsScaler {
    inner: MaxAbsScaler,
}

#[pymethods]
impl PyMaxAbsScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: MaxAbsScaler::new(),
        }
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// Quantile Transformer
#[pyclass(name = "QuantileTransformer")]
pub struct PyQuantileTransformer {
    inner: QuantileTransformer,
}

#[pymethods]
impl PyQuantileTransformer {
    #[new]
    #[pyo3(signature = (n_quantiles=1000, output_distribution="uniform", clip=false))]
    fn new(n_quantiles: usize, output_distribution: &str, clip: bool) -> PyResult<Self> {
        let transformer = QuantileTransformer::new(n_quantiles, output_distribution, clip)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("QuantileTransformer creation failed: {}", e))
            })?;
        Ok(Self { inner: transformer })
    }

    fn fit(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = array.readonly();
        let arr = binding.as_array();
        self.inner
            .fit(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit failed: {}", e)))
    }

    fn transform(
        &self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }

    fn fit_transform(
        &mut self,
        py: Python,
        array: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let binding = array.readonly();
        let arr = binding.as_array();
        let result = self
            .inner
            .fit_transform(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("Fit transform failed: {}", e)))?;
        Ok(result.into_pyarray(py).unbind())
    }
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Normalization
    m.add_function(wrap_pyfunction!(normalize_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_vector_py, m)?)?;
    m.add_class::<PyNormalizer>()?;

    // Feature engineering
    m.add_class::<PyPolynomialFeatures>()?;
    m.add_class::<PyPowerTransformer>()?;
    m.add_function(wrap_pyfunction!(binarize_py, m)?)?;
    m.add_function(wrap_pyfunction!(discretize_equal_width_py, m)?)?;
    m.add_function(wrap_pyfunction!(discretize_equal_frequency_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(power_transform_py, m)?)?;

    // Dimensionality reduction
    m.add_class::<PyPCA>()?;
    m.add_class::<PyTSNE>()?;
    m.add_class::<PyUMAP>()?;

    // Categorical encoding
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyOrdinalEncoder>()?;

    // Imputation
    m.add_class::<PySimpleImputer>()?;
    m.add_class::<PyKNNImputer>()?;

    // Scaling
    m.add_class::<PyMaxAbsScaler>()?;
    m.add_class::<PyQuantileTransformer>()?;

    Ok(())
}
