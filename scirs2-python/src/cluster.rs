//! Python bindings for scirs2-cluster
//!
//! This module provides Python bindings that make scirs2-cluster algorithms
//! accessible from Python with scikit-learn compatible APIs.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core
use scirs2_core::{Array1, Array2};

// Direct imports from scirs2-cluster (native ndarray 0.17 support)
use scirs2_cluster::kmeans;
use scirs2_cluster::{calinski_harabasz_score, davies_bouldin_score, silhouette_score};
use scirs2_cluster::{normalize, standardize, NormType};

/// Python-compatible K-means clustering implementation
#[pyclass(name = "KMeans")]
pub struct PyKMeans {
    /// Number of clusters
    n_clusters: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed
    random_state: Option<u64>,
    /// Number of initializations
    n_init: usize,
    /// Initialization method
    init: String,
    /// Fitted cluster centers
    cluster_centers_: Option<Vec<Vec<f64>>>,
    /// Labels of each point
    labels_: Option<Vec<usize>>,
    /// Sum of squared distances to centroids
    inertia_: Option<f64>,
}

#[pymethods]
impl PyKMeans {
    /// Create new K-means clustering instance
    #[new]
    #[pyo3(signature = (n_clusters=8, *, init="k-means++", n_init=10, max_iter=300, tol=1e-4, random_state=None))]
    fn new(
        n_clusters: usize,
        init: &str,
        n_init: usize,
        max_iter: usize,
        tol: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_clusters,
            max_iter,
            tol,
            random_state,
            n_init,
            init: init.to_string(),
            cluster_centers_: None,
            labels_: None,
            inertia_: None,
        }
    }

    /// Fit K-means clustering to data
    fn fit(&mut self, _py: Python, x: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let binding = x.readonly();
        let data = binding.as_array();

        // Run K-means using scirs2_cluster directly
        let (centroids, inertia) = kmeans(
            data,
            self.n_clusters,
            Some(self.max_iter),
            Some(self.tol),
            Some(true), // check_finite
            self.random_state,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("K-means fitting failed: {}", e)))?;

        // Assign labels by finding nearest centroid for each point
        let n_samples = data.nrows();
        let mut labels = Vec::with_capacity(n_samples);

        for sample in data.rows() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (j, centroid) in centroids.rows().into_iter().enumerate() {
                let dist: f64 = sample
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels.push(best_cluster);
        }

        // Store results
        self.cluster_centers_ = Some(
            centroids
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect(),
        );
        self.labels_ = Some(labels);
        self.inertia_ = Some(inertia);

        Ok(())
    }

    /// Fit and predict cluster labels
    fn fit_predict(
        &mut self,
        py: Python,
        x: &Bound<'_, PyArray2<f64>>,
    ) -> PyResult<Py<PyArray1<i32>>> {
        self.fit(py, x)?;
        self.labels(py)
    }

    /// Predict cluster labels for new data
    fn predict(&self, py: Python, x: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyArray1<i32>>> {
        if self.cluster_centers_.is_none() {
            return Err(PyRuntimeError::new_err("Model not fitted yet"));
        }

        let binding = x.readonly();
        let data = binding.as_array();
        let centers = self.cluster_centers_.as_ref().expect("Operation failed");

        let n_samples = data.nrows();
        let mut labels = Vec::with_capacity(n_samples);

        for sample in data.rows() {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for (j, center) in centers.iter().enumerate() {
                let dist: f64 = sample
                    .iter()
                    .zip(center.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels.push(best_cluster as i32);
        }

        let labels_array = Array1::from_vec(labels);
        Ok(labels_array.into_pyarray(py).unbind())
    }

    /// Get cluster centers
    #[getter]
    fn cluster_centers_(&self, py: Python) -> PyResult<Option<Py<PyArray2<f64>>>> {
        match &self.cluster_centers_ {
            Some(centers) => {
                let n_clusters = centers.len();
                let n_features = centers.first().map(|c| c.len()).unwrap_or(0);
                let flat: Vec<f64> = centers.iter().flatten().copied().collect();
                let array = Array2::from_shape_vec((n_clusters, n_features), flat)
                    .map_err(|e| PyRuntimeError::new_err(format!("Array reshape error: {}", e)))?;
                Ok(Some(array.into_pyarray(py).unbind()))
            }
            None => Ok(None),
        }
    }

    /// Get labels
    #[getter]
    fn labels(&self, py: Python) -> PyResult<Py<PyArray1<i32>>> {
        match &self.labels_ {
            Some(labels) => {
                let labels_i32: Vec<i32> = labels.iter().map(|&x| x as i32).collect();
                let array = Array1::from_vec(labels_i32);
                Ok(array.into_pyarray(py).unbind())
            }
            None => Err(PyRuntimeError::new_err("Model not fitted yet")),
        }
    }

    /// Get inertia (sum of squared distances to centroids)
    #[getter]
    fn inertia_(&self) -> Option<f64> {
        self.inertia_
    }

    /// Set parameters
    fn set_params(&mut self, params: &Bound<'_, PyDict>) -> PyResult<()> {
        for (key, value) in params.iter() {
            let key_str: String = key.extract()?;
            match key_str.as_str() {
                "n_clusters" => self.n_clusters = value.extract()?,
                "max_iter" => self.max_iter = value.extract()?,
                "tol" => self.tol = value.extract()?,
                "random_state" => self.random_state = value.extract()?,
                "n_init" => self.n_init = value.extract()?,
                "init" => self.init = value.extract()?,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Unknown parameter: {}",
                        key_str
                    )))
                }
            }
        }
        Ok(())
    }

    /// Get parameters
    fn get_params(&self, py: Python, _deep: Option<bool>) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("n_clusters", self.n_clusters)?;
        dict.set_item("max_iter", self.max_iter)?;
        dict.set_item("tol", self.tol)?;
        dict.set_item("random_state", self.random_state)?;
        dict.set_item("n_init", self.n_init)?;
        dict.set_item("init", &self.init)?;
        Ok(dict.into_any().unbind())
    }
}

/// Calculate silhouette score
#[pyfunction]
fn silhouette_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<i32>>,
) -> PyResult<f64> {
    let binding = x.readonly();
    let data = binding.as_array();
    let labels_binding = labels.readonly();
    let labels_arr = labels_binding.as_array();

    let score = silhouette_score(data, labels_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Silhouette score failed: {}", e)))?;

    Ok(score)
}

/// Calculate Davies-Bouldin score
#[pyfunction]
fn davies_bouldin_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<i32>>,
) -> PyResult<f64> {
    let binding = x.readonly();
    let data = binding.as_array();
    let labels_binding = labels.readonly();
    let labels_arr = labels_binding.as_array();

    let score = davies_bouldin_score(data, labels_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Davies-Bouldin score failed: {}", e)))?;

    Ok(score)
}

/// Calculate Calinski-Harabasz score
#[pyfunction]
fn calinski_harabasz_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<i32>>,
) -> PyResult<f64> {
    let binding = x.readonly();
    let data = binding.as_array();
    let labels_binding = labels.readonly();
    let labels_arr = labels_binding.as_array();

    let score = calinski_harabasz_score(data, labels_arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Calinski-Harabasz score failed: {}", e)))?;

    Ok(score)
}

/// Standardize data to zero mean and unit variance
#[pyfunction]
fn standardize_py(py: Python, x: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let binding = x.readonly();
    let data = binding.as_array();

    let result = standardize(data, true)  // check_finite=true
        .map_err(|e| PyRuntimeError::new_err(format!("Standardization failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Normalize data to unit norm
#[pyfunction]
fn normalize_py(
    py: Python,
    x: &Bound<'_, PyArray2<f64>>,
    norm: Option<&str>,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = x.readonly();
    let data = binding.as_array();

    let norm_type = match norm.unwrap_or("l2") {
        "l1" => NormType::L1,
        "l2" => NormType::L2,
        "max" => NormType::Max,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown norm type: {}",
                other
            )))
        }
    };

    let result =
        normalize(data, norm_type, true) // check_finite=true
            .map_err(|e| PyRuntimeError::new_err(format!("Normalization failed: {}", e)))?;

    Ok(result.into_pyarray(py).unbind())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyKMeans>()?;

    // Metrics
    m.add_function(wrap_pyfunction!(silhouette_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score_py, m)?)?;

    // Preprocessing
    m.add_function(wrap_pyfunction!(standardize_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_py, m)?)?;

    Ok(())
}
