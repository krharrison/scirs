//! Python bindings for scirs2-metrics
//!
//! This module provides Python bindings for machine learning evaluation metrics,
//! including classification, regression, and clustering metrics.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core
use scirs2_core::ndarray::{Array1, Array2};

// Direct imports from scirs2-metrics submodules
use scirs2_metrics::classification::advanced::{
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
};
use scirs2_metrics::classification::curves::roc_curve;
use scirs2_metrics::classification::{
    accuracy_score, binary_log_loss, confusion_matrix, f1_score, fbeta_score, precision_score,
    recall_score, roc_auc_score,
};
use scirs2_metrics::clustering::{
    adjusted_rand_index, calinski_harabasz_score, davies_bouldin_score,
    normalized_mutual_info_score, silhouette_score,
};
use scirs2_metrics::ranking::{mean_reciprocal_rank, ndcg_score};
use scirs2_metrics::regression::{
    explained_variance_score, mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score,
};

// ========================================
// CLASSIFICATION METRICS
// ========================================

/// Calculate accuracy score
#[pyfunction]
fn accuracy_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    accuracy_score(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Accuracy score failed: {}", e)))
}

/// Calculate precision score (binary classification)
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, pos_label=1))]
fn precision_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
    pos_label: i64,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    precision_score(&y_true_data, &y_pred_data, pos_label)
        .map_err(|e| PyRuntimeError::new_err(format!("Precision score failed: {}", e)))
}

/// Calculate recall score (binary classification)
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, pos_label=1))]
fn recall_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
    pos_label: i64,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    recall_score(&y_true_data, &y_pred_data, pos_label)
        .map_err(|e| PyRuntimeError::new_err(format!("Recall score failed: {}", e)))
}

/// Calculate F1 score (binary classification)
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, pos_label=1))]
fn f1_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
    pos_label: i64,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    f1_score(&y_true_data, &y_pred_data, pos_label)
        .map_err(|e| PyRuntimeError::new_err(format!("F1 score failed: {}", e)))
}

/// Calculate F-beta score
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, beta, pos_label=1))]
fn fbeta_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
    beta: f64,
    pos_label: i64,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    fbeta_score(&y_true_data, &y_pred_data, pos_label, beta)
        .map_err(|e| PyRuntimeError::new_err(format!("F-beta score failed: {}", e)))
}

/// Calculate confusion matrix
#[pyfunction]
fn confusion_matrix_py(
    py: Python,
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<Py<PyAny>> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    let (matrix, classes): (Array2<u64>, Array1<i64>) =
        confusion_matrix(&y_true_data, &y_pred_data, None)
            .map_err(|e| PyRuntimeError::new_err(format!("Confusion matrix failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("matrix", matrix.into_pyarray(py).unbind())?;
    dict.set_item("classes", classes.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

/// Calculate ROC curve
#[pyfunction]
fn roc_curve_py(
    py: Python,
    y_true: &Bound<'_, PyArray1<i32>>,
    y_score: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let y_true_binding = y_true.readonly();
    let y_score_binding = y_score.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_score_data = y_score_binding.as_array();

    // roc_curve requires S1::Elem: Into<f64>, i32 implements this
    // Returns ROCCurveResult which is a struct with fpr, tpr, thresholds fields
    let result = roc_curve(&y_true_data, &y_score_data)
        .map_err(|e| PyRuntimeError::new_err(format!("ROC curve failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("fpr", result.0.into_pyarray(py).unbind())?;
    dict.set_item("tpr", result.1.into_pyarray(py).unbind())?;
    dict.set_item("thresholds", result.2.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

/// Calculate ROC AUC score
#[pyfunction]
fn roc_auc_score_py(
    y_true: &Bound<'_, PyArray1<u32>>,
    y_score: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_score_binding = y_score.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_score_data = y_score_binding.as_array();

    // roc_auc_score requires S1: Data<Elem = u32>
    roc_auc_score(&y_true_data, &y_score_data)
        .map_err(|e| PyRuntimeError::new_err(format!("ROC AUC score failed: {}", e)))
}

/// Calculate log loss
#[pyfunction]
#[pyo3(signature = (y_true, y_prob, eps=1e-15))]
fn log_loss_py(
    y_true: &Bound<'_, PyArray1<u32>>,
    y_prob: &Bound<'_, PyArray1<f64>>,
    eps: f64,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_prob_binding = y_prob.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_prob_data = y_prob_binding.as_array();

    // binary_log_loss requires S1: Data<Elem = u32> and takes eps parameter
    binary_log_loss(&y_true_data, &y_prob_data, eps)
        .map_err(|e| PyRuntimeError::new_err(format!("Log loss failed: {}", e)))
}

/// Calculate Matthews correlation coefficient
#[pyfunction]
fn matthews_corrcoef_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    matthews_corrcoef(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Matthews correlation failed: {}", e)))
}

/// Calculate balanced accuracy score
#[pyfunction]
fn balanced_accuracy_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    balanced_accuracy_score(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Balanced accuracy failed: {}", e)))
}

/// Calculate Cohen's kappa score
#[pyfunction]
fn cohen_kappa_score_py(
    y_true: &Bound<'_, PyArray1<i64>>,
    y_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    cohen_kappa_score(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Cohen's kappa failed: {}", e)))
}

// ========================================
// REGRESSION METRICS
// ========================================

/// Calculate mean squared error
#[pyfunction]
fn mean_squared_error_py(
    y_true: &Bound<'_, PyArray1<f64>>,
    y_pred: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    mean_squared_error(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("MSE failed: {}", e)))
}

/// Calculate mean absolute error
#[pyfunction]
fn mean_absolute_error_py(
    y_true: &Bound<'_, PyArray1<f64>>,
    y_pred: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    mean_absolute_error(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("MAE failed: {}", e)))
}

/// Calculate R² score
#[pyfunction]
fn r2_score_py(
    y_true: &Bound<'_, PyArray1<f64>>,
    y_pred: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    r2_score(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("R² score failed: {}", e)))
}

/// Calculate mean absolute percentage error
#[pyfunction]
fn mape_py(y_true: &Bound<'_, PyArray1<f64>>, y_pred: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    mean_absolute_percentage_error(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("MAPE failed: {}", e)))
}

/// Calculate explained variance score
#[pyfunction]
fn explained_variance_score_py(
    y_true: &Bound<'_, PyArray1<f64>>,
    y_pred: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_pred_binding = y_pred.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_pred_data = y_pred_binding.as_array();

    explained_variance_score(&y_true_data, &y_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Explained variance failed: {}", e)))
}

// ========================================
// CLUSTERING METRICS
// ========================================

/// Calculate silhouette score
#[pyfunction]
#[pyo3(signature = (x, labels, metric="euclidean"))]
fn silhouette_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<usize>>,
    metric: &str,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let labels_binding = labels.readonly();
    let x_data = x_binding.as_array();
    let labels_data = labels_binding.as_array();

    silhouette_score(&x_data, &labels_data, metric)
        .map_err(|e| PyRuntimeError::new_err(format!("Silhouette score failed: {}", e)))
}

/// Calculate Davies-Bouldin score
#[pyfunction]
fn davies_bouldin_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<usize>>,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let labels_binding = labels.readonly();
    let x_data = x_binding.as_array();
    let labels_data = labels_binding.as_array();

    davies_bouldin_score(&x_data, &labels_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Davies-Bouldin score failed: {}", e)))
}

/// Calculate Calinski-Harabasz score
#[pyfunction]
fn calinski_harabasz_score_py(
    x: &Bound<'_, PyArray2<f64>>,
    labels: &Bound<'_, PyArray1<usize>>,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let labels_binding = labels.readonly();
    let x_data = x_binding.as_array();
    let labels_data = labels_binding.as_array();

    calinski_harabasz_score(&x_data, &labels_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Calinski-Harabasz score failed: {}", e)))
}

/// Calculate adjusted Rand index
#[pyfunction]
fn adjusted_rand_index_py(
    labels_true: &Bound<'_, PyArray1<i64>>,
    labels_pred: &Bound<'_, PyArray1<i64>>,
) -> PyResult<f64> {
    let labels_true_binding = labels_true.readonly();
    let labels_pred_binding = labels_pred.readonly();
    let labels_true_data = labels_true_binding.as_array();
    let labels_pred_data = labels_pred_binding.as_array();

    adjusted_rand_index(&labels_true_data, &labels_pred_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Adjusted Rand index failed: {}", e)))
}

/// Calculate normalized mutual information score
#[pyfunction]
#[pyo3(signature = (labels_true, labels_pred, average_method="arithmetic"))]
fn nmi_score_py(
    labels_true: &Bound<'_, PyArray1<i64>>,
    labels_pred: &Bound<'_, PyArray1<i64>>,
    average_method: &str,
) -> PyResult<f64> {
    let labels_true_binding = labels_true.readonly();
    let labels_pred_binding = labels_pred.readonly();
    let labels_true_data = labels_true_binding.as_array();
    let labels_pred_data = labels_pred_binding.as_array();

    normalized_mutual_info_score(&labels_true_data, &labels_pred_data, average_method)
        .map_err(|e| PyRuntimeError::new_err(format!("NMI score failed: {}", e)))
}

// ========================================
// RANKING METRICS
// ========================================

/// Calculate NDCG score
/// y_true and y_score are 2D arrays where each row represents a query
#[pyfunction]
#[pyo3(signature = (y_true, y_score, k=None))]
fn ndcg_score_py(
    y_true: &Bound<'_, PyArray2<f64>>,
    y_score: &Bound<'_, PyArray2<f64>>,
    k: Option<usize>,
) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_score_binding = y_score.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_score_data = y_score_binding.as_array();

    // Convert 2D arrays to Vec of 1D arrays (each row is a query)
    let y_true_vec: Vec<Array1<f64>> = y_true_data
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect();
    let y_score_vec: Vec<Array1<f64>> = y_score_data
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect();

    ndcg_score(&y_true_vec, &y_score_vec, k)
        .map_err(|e| PyRuntimeError::new_err(format!("NDCG score failed: {}", e)))
}

/// Calculate mean reciprocal rank
/// y_true and y_score are 2D arrays where each row represents a query
#[pyfunction]
fn mrr_py(y_true: &Bound<'_, PyArray2<f64>>, y_score: &Bound<'_, PyArray2<f64>>) -> PyResult<f64> {
    let y_true_binding = y_true.readonly();
    let y_score_binding = y_score.readonly();
    let y_true_data = y_true_binding.as_array();
    let y_score_data = y_score_binding.as_array();

    // Convert 2D arrays to Vec of 1D arrays (each row is a query)
    let y_true_vec: Vec<Array1<f64>> = y_true_data
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect();
    let y_score_vec: Vec<Array1<f64>> = y_score_data
        .rows()
        .into_iter()
        .map(|row| row.to_owned())
        .collect();

    mean_reciprocal_rank(&y_true_vec, &y_score_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("MRR failed: {}", e)))
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classification metrics
    m.add_function(wrap_pyfunction!(accuracy_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(fbeta_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(confusion_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(roc_curve_py, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(log_loss_py, m)?)?;
    m.add_function(wrap_pyfunction!(matthews_corrcoef_py, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_accuracy_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(cohen_kappa_score_py, m)?)?;

    // Regression metrics
    m.add_function(wrap_pyfunction!(mean_squared_error_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_absolute_error_py, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(mape_py, m)?)?;
    m.add_function(wrap_pyfunction!(explained_variance_score_py, m)?)?;

    // Clustering metrics
    m.add_function(wrap_pyfunction!(silhouette_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(davies_bouldin_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(calinski_harabasz_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(adjusted_rand_index_py, m)?)?;
    m.add_function(wrap_pyfunction!(nmi_score_py, m)?)?;

    // Ranking metrics
    m.add_function(wrap_pyfunction!(ndcg_score_py, m)?)?;
    m.add_function(wrap_pyfunction!(mrr_py, m)?)?;

    Ok(())
}
