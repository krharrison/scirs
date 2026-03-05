//! Python bindings for scirs2-sparse
//!
//! This module provides Python bindings for sparse matrix operations,
//! including CSR, CSC, COO formats and basic sparse operations.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core
use scirs2_core::ndarray::Array1;

// Direct imports from scirs2-sparse
use scirs2_sparse::{diag_matrix, eye, CooArray, CscArray, CsrArray, SparseArray};

// ========================================
// SPARSE ARRAY CREATION
// ========================================

/// Create a CSR sparse array from triplets (row, col, data)
#[pyfunction]
#[pyo3(signature = (rows, cols, data, shape, sum_duplicates=false))]
#[allow(clippy::too_many_arguments)]
fn csr_array_from_triplets(
    py: Python,
    rows: Vec<usize>,
    cols: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
    sum_duplicates: bool,
) -> PyResult<Py<PyAny>> {
    let csr = CsrArray::from_triplets(&rows, &cols, &data, shape, sum_duplicates)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CSR array: {}", e)))?;

    // Return as dict with internal representation
    let dict = PyDict::new(py);
    dict.set_item("format", "csr")?;
    dict.set_item("shape", shape)?;
    dict.set_item("nnz", csr.nnz())?;

    // Get CSR components
    dict.set_item("indptr", csr.get_indptr().to_vec())?;
    dict.set_item("indices", csr.get_indices().to_vec())?;
    dict.set_item("data", csr.get_data().to_vec())?;

    Ok(dict.into())
}

/// Create a COO sparse array from triplets
#[pyfunction]
#[pyo3(signature = (rows, cols, data, shape, sum_duplicates=false))]
#[allow(clippy::too_many_arguments)]
fn coo_array_from_triplets(
    py: Python,
    rows: Vec<usize>,
    cols: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
    sum_duplicates: bool,
) -> PyResult<Py<PyAny>> {
    let coo = CooArray::from_triplets(&rows, &cols, &data, shape, sum_duplicates)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create COO array: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("format", "coo")?;
    dict.set_item("shape", shape)?;
    dict.set_item("nnz", coo.nnz())?;

    // Get COO components
    dict.set_item("row", coo.get_rows().to_vec())?;
    dict.set_item("col", coo.get_cols().to_vec())?;
    dict.set_item("data", coo.get_data().to_vec())?;

    Ok(dict.into())
}

/// Create a CSC sparse array from triplets
#[pyfunction]
#[pyo3(signature = (rows, cols, data, shape, sum_duplicates=false))]
#[allow(clippy::too_many_arguments)]
fn csc_array_from_triplets(
    py: Python,
    rows: Vec<usize>,
    cols: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
    sum_duplicates: bool,
) -> PyResult<Py<PyAny>> {
    let csc = CscArray::from_triplets(&rows, &cols, &data, shape, sum_duplicates)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CSC array: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("format", "csc")?;
    dict.set_item("shape", shape)?;
    dict.set_item("nnz", csc.nnz())?;

    // Get CSC components
    dict.set_item("indptr", csc.get_indptr().to_vec())?;
    dict.set_item("indices", csc.get_indices().to_vec())?;
    dict.set_item("data", csc.get_data().to_vec())?;

    Ok(dict.into())
}

/// Create sparse identity matrix
#[pyfunction]
fn sparse_eye_py(py: Python, n: usize) -> PyResult<Py<PyAny>> {
    let csr = eye::<f64>(n)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create identity matrix: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("format", "csr")?;
    dict.set_item("shape", (n, n))?;
    dict.set_item("nnz", csr.data.len())?;

    // CsrMatrix has public fields
    dict.set_item("indptr", csr.indptr.clone())?;
    dict.set_item("indices", csr.indices.clone())?;
    dict.set_item("data", csr.data.clone())?;

    Ok(dict.into())
}

/// Create sparse diagonal matrix from vector
#[pyfunction]
fn sparse_diag_py(py: Python, diag: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let binding = diag.readonly();
    let diag_data = binding.as_array();

    // diag_matrix takes &[F] and Option<usize>, returns CsrMatrix
    let diag_slice: Vec<f64> = diag_data.iter().copied().collect();
    let csr = diag_matrix::<f64>(&diag_slice, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create diagonal matrix: {}", e)))?;

    let n = diag_data.len();
    let dict = PyDict::new(py);
    dict.set_item("format", "csr")?;
    dict.set_item("shape", (n, n))?;
    dict.set_item("nnz", csr.data.len())?;

    // CsrMatrix has public fields
    dict.set_item("indptr", csr.indptr.clone())?;
    dict.set_item("indices", csr.indices.clone())?;
    dict.set_item("data", csr.data.clone())?;

    Ok(dict.into())
}

// ========================================
// SPARSE ARRAY OPERATIONS
// ========================================

/// Convert sparse array to dense
#[pyfunction]
fn sparse_to_dense_py(
    py: Python,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
) -> PyResult<Py<PyArray2<f64>>> {
    // Reconstruct CSR array - CsrArray::new expects (data, indices, indptr, shape)
    let csr = CsrArray::new(
        Array1::from_vec(data),
        Array1::from_vec(indices),
        Array1::from_vec(indptr),
        shape,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Invalid CSR data: {}", e)))?;

    let dense = csr.to_array();
    Ok(dense.into_pyarray(py).unbind())
}

/// Sparse matrix-vector multiplication
#[pyfunction]
fn sparse_matvec_py(
    py: Python,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
    x: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let csr = CsrArray::new(
        Array1::from_vec(data),
        Array1::from_vec(indices),
        Array1::from_vec(indptr),
        shape,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Invalid CSR data: {}", e)))?;

    let x_binding = x.readonly();
    let x_data = x_binding.as_array();

    let result = csr.dot_vector(&x_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Matrix-vector multiplication failed: {}", e))
    })?;

    Ok(result.into_pyarray(py).unbind())
}

/// Sparse matrix-matrix multiplication (returns CSR)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn sparse_matmul_py(
    py: Python,
    a_indptr: Vec<usize>,
    a_indices: Vec<usize>,
    a_data: Vec<f64>,
    a_shape: (usize, usize),
    b_indptr: Vec<usize>,
    b_indices: Vec<usize>,
    b_data: Vec<f64>,
    b_shape: (usize, usize),
) -> PyResult<Py<PyAny>> {
    let a = CsrArray::new(
        Array1::from_vec(a_data),
        Array1::from_vec(a_indices),
        Array1::from_vec(a_indptr),
        a_shape,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Invalid CSR data for A: {}", e)))?;
    let b = CsrArray::new(
        Array1::from_vec(b_data),
        Array1::from_vec(b_indices),
        Array1::from_vec(b_indptr),
        b_shape,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Invalid CSR data for B: {}", e)))?;

    let result = a
        .dot(&b)
        .map_err(|e| PyRuntimeError::new_err(format!("Matrix multiplication failed: {}", e)))?;

    // Use find() to get row, col, data from the boxed SparseArray
    let (rows, cols, data) = result.find();
    let shape = result.shape();
    let nnz = result.nnz();

    // Convert result to CSR format for return
    let result_csr = CsrArray::from_triplets(
        rows.as_slice().unwrap_or(&[]),
        cols.as_slice().unwrap_or(&[]),
        data.as_slice().unwrap_or(&[]),
        shape,
        false,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create result CSR: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("format", "csr")?;
    dict.set_item("shape", shape)?;
    dict.set_item("nnz", nnz)?;

    dict.set_item("indptr", result_csr.get_indptr().to_vec())?;
    dict.set_item("indices", result_csr.get_indices().to_vec())?;
    dict.set_item("data", result_csr.get_data().to_vec())?;

    Ok(dict.into())
}

/// Transpose sparse matrix
#[pyfunction]
fn sparse_transpose_py(
    py: Python,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
) -> PyResult<Py<PyAny>> {
    let csr = CsrArray::new(
        Array1::from_vec(data),
        Array1::from_vec(indices),
        Array1::from_vec(indptr),
        shape,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Invalid CSR data: {}", e)))?;

    let transposed = csr
        .transpose()
        .map_err(|e| PyRuntimeError::new_err(format!("Transpose failed: {}", e)))?;

    // Use find() to get row, col, data from the boxed SparseArray
    let (rows, cols, data) = transposed.find();
    let t_shape = transposed.shape();
    let nnz = transposed.nnz();

    // Convert result to CSR format for return
    let result_csr = CsrArray::from_triplets(
        rows.as_slice().unwrap_or(&[]),
        cols.as_slice().unwrap_or(&[]),
        data.as_slice().unwrap_or(&[]),
        t_shape,
        false,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to create transposed CSR: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("format", "csr")?;
    dict.set_item("shape", t_shape)?;
    dict.set_item("nnz", nnz)?;

    dict.set_item("indptr", result_csr.get_indptr().to_vec())?;
    dict.set_item("indices", result_csr.get_indices().to_vec())?;
    dict.set_item("data", result_csr.get_data().to_vec())?;

    Ok(dict.into())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Array creation
    m.add_function(wrap_pyfunction!(csr_array_from_triplets, m)?)?;
    m.add_function(wrap_pyfunction!(coo_array_from_triplets, m)?)?;
    m.add_function(wrap_pyfunction!(csc_array_from_triplets, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_eye_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_diag_py, m)?)?;

    // Array operations
    m.add_function(wrap_pyfunction!(sparse_to_dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_matvec_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_matmul_py, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_transpose_py, m)?)?;

    Ok(())
}
