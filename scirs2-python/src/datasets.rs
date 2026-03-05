//! Python bindings for scirs2-datasets
//!
//! This module provides Python bindings for dataset loading and generation,
//! including toy datasets, synthetic data generators, and utilities.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core (used indirectly via Dataset)
#[allow(unused_imports)]
use scirs2_core::ndarray::{Array1, Array2};

// Direct imports from scirs2-datasets
use scirs2_datasets::{
    // Manifold datasets
    generators::manifold::{make_s_curve, make_swiss_roll},
    // Generators
    generators::{
        make_blobs, make_circles, make_classification, make_moons, make_regression, make_spirals,
    },
    // Toy datasets
    toy::{load_boston, load_breast_cancer, load_diabetes, load_digits, load_iris},
    // Utilities
    utils::{k_fold_split, min_max_scale, normalize, train_test_split},
    // Dataset structure
    Dataset,
};

// ========================================
// HELPER FUNCTION
// ========================================

/// Convert Dataset to Python dict
fn dataset_to_pydict(py: Python, dataset: Dataset) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    let n_samples = dataset.data.nrows();
    let n_features = dataset.data.ncols();
    dict.set_item("data", dataset.data.into_pyarray(py).unbind())?;

    if let Some(target) = dataset.target {
        dict.set_item("target", target.into_pyarray(py).unbind())?;
    }

    if let Some(featurenames) = dataset.featurenames {
        dict.set_item("feature_names", featurenames)?;
    }

    if let Some(targetnames) = dataset.targetnames {
        dict.set_item("target_names", targetnames)?;
    }

    if let Some(description) = dataset.description {
        dict.set_item("description", description)?;
    }

    dict.set_item("n_samples", n_samples)?;
    dict.set_item("n_features", n_features)?;

    Ok(dict.into())
}

// ========================================
// TOY DATASETS
// ========================================

/// Load Iris dataset
#[pyfunction]
fn load_iris_py(py: Python) -> PyResult<Py<PyAny>> {
    let dataset =
        load_iris().map_err(|e| PyRuntimeError::new_err(format!("Failed to load iris: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Load Boston housing dataset
#[pyfunction]
fn load_boston_py(py: Python) -> PyResult<Py<PyAny>> {
    let dataset = load_boston()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load boston: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Load Diabetes dataset
#[pyfunction]
fn load_diabetes_py(py: Python) -> PyResult<Py<PyAny>> {
    let dataset = load_diabetes()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load diabetes: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Load Breast Cancer dataset
#[pyfunction]
fn load_breast_cancer_py(py: Python) -> PyResult<Py<PyAny>> {
    let dataset = load_breast_cancer()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load breast cancer: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Load Digits dataset
#[pyfunction]
fn load_digits_py(py: Python) -> PyResult<Py<PyAny>> {
    let dataset = load_digits()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load digits: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

// ========================================
// DATA GENERATORS
// ========================================

/// Generate synthetic classification dataset
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_clusters_per_class=2, random_seed=None))]
fn make_classification_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    n_redundant: usize,
    n_clusters_per_class: usize,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_classification(
        n_samples,
        n_features,
        n_informative,
        n_redundant,
        n_clusters_per_class,
        random_seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to make classification: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate synthetic regression dataset
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=10, n_informative=5, noise=0.1, random_seed=None))]
fn make_regression_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    n_informative: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_regression(n_samples, n_features, n_informative, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make regression: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate blob clusters
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_features=2, n_clusters=3, std_dev=1.0, random_seed=None))]
fn make_blobs_py(
    py: Python,
    n_samples: usize,
    n_features: usize,
    n_clusters: usize,
    std_dev: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_blobs(n_samples, n_features, n_clusters, std_dev, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make blobs: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate two interleaving half circles (moons)
#[pyfunction]
#[pyo3(signature = (n_samples=100, noise=0.1, random_seed=None))]
fn make_moons_py(
    py: Python,
    n_samples: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_moons(n_samples, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make moons: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate two concentric circles
#[pyfunction]
#[pyo3(signature = (n_samples=100, factor=0.8, noise=0.1, random_seed=None))]
fn make_circles_py(
    py: Python,
    n_samples: usize,
    factor: f64,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_circles(n_samples, factor, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make circles: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate spiral clusters
#[pyfunction]
#[pyo3(signature = (n_samples=100, n_spirals=2, noise=0.1, random_seed=None))]
fn make_spirals_py(
    py: Python,
    n_samples: usize,
    n_spirals: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_spirals(n_samples, n_spirals, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make spirals: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

// ========================================
// MANIFOLD DATASETS
// ========================================

/// Generate Swiss roll dataset
#[pyfunction]
#[pyo3(signature = (n_samples=1000, noise=0.0, random_seed=None))]
fn make_swiss_roll_py(
    py: Python,
    n_samples: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_swiss_roll(n_samples, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make swiss roll: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

/// Generate S-curve dataset
#[pyfunction]
#[pyo3(signature = (n_samples=1000, noise=0.0, random_seed=None))]
fn make_s_curve_py(
    py: Python,
    n_samples: usize,
    noise: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let dataset = make_s_curve(n_samples, noise, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to make s-curve: {}", e)))?;
    dataset_to_pydict(py, dataset)
}

// ========================================
// DATA UTILITIES
// ========================================

/// Split arrays into train and test subsets
#[pyfunction]
#[pyo3(signature = (x, y, test_size=0.25, random_seed=None))]
fn train_test_split_py(
    py: Python,
    x: &Bound<'_, PyArray2<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    test_size: f64,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let x_binding = x.readonly();
    let y_binding = y.readonly();
    let x_data = x_binding.as_array().to_owned();
    let y_data = y_binding.as_array().to_owned();

    let dataset = Dataset {
        data: x_data,
        target: Some(y_data),
        featurenames: None,
        targetnames: None,
        feature_descriptions: None,
        description: None,
        metadata: std::collections::HashMap::new(),
    };

    let (train, test) = train_test_split(&dataset, test_size, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("Train-test split failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x_train", train.data.into_pyarray(py).unbind())?;
    dict.set_item("x_test", test.data.into_pyarray(py).unbind())?;

    if let Some(y_train) = train.target {
        dict.set_item("y_train", y_train.into_pyarray(py).unbind())?;
    }
    if let Some(y_test) = test.target {
        dict.set_item("y_test", y_test.into_pyarray(py).unbind())?;
    }

    Ok(dict.into())
}

/// K-fold cross-validation split indices
#[pyfunction]
#[pyo3(signature = (n_samples, n_folds=5, shuffle=true, random_seed=None))]
fn k_fold_split_py(
    py: Python,
    n_samples: usize,
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let folds = k_fold_split(n_samples, n_folds, shuffle, random_seed)
        .map_err(|e| PyRuntimeError::new_err(format!("K-fold split failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("n_folds", n_folds)?;

    // folds is already Vec<(Vec<usize>, Vec<usize>)>
    dict.set_item("folds", folds)?;

    Ok(dict.into())
}

/// Min-max scale array to [0, 1] range
#[pyfunction]
#[pyo3(signature = (data, min_val=0.0, max_val=1.0))]
fn min_max_scale_py(
    py: Python,
    data: &Bound<'_, PyArray2<f64>>,
    min_val: f64,
    max_val: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let binding = data.readonly();
    let mut arr = binding.as_array().to_owned();

    min_max_scale(&mut arr, (min_val, max_val));

    Ok(arr.into_pyarray(py).unbind())
}

/// Normalize array (L2 norm per row)
#[pyfunction]
fn normalize_py(py: Python, data: &Bound<'_, PyArray2<f64>>) -> PyResult<Py<PyArray2<f64>>> {
    let binding = data.readonly();
    let mut arr = binding.as_array().to_owned();

    normalize(&mut arr);

    Ok(arr.into_pyarray(py).unbind())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Toy datasets
    m.add_function(wrap_pyfunction!(load_iris_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_boston_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_diabetes_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_breast_cancer_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_digits_py, m)?)?;

    // Data generators
    m.add_function(wrap_pyfunction!(make_classification_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_regression_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_blobs_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_moons_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_circles_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_spirals_py, m)?)?;

    // Manifold datasets
    m.add_function(wrap_pyfunction!(make_swiss_roll_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_s_curve_py, m)?)?;

    // Utilities
    m.add_function(wrap_pyfunction!(train_test_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(k_fold_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(min_max_scale_py, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_py, m)?)?;

    Ok(())
}
