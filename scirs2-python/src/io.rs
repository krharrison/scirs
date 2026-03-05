//! Python bindings for scirs2-io
//!
//! This module provides Python bindings for file I/O operations,
//! including CSV, MATLAB, HDF5, and other format support.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core
#[allow(unused_imports)]
use scirs2_core::ndarray::{Array1, Array2};

// Direct imports from scirs2-io
use scirs2_io::{
    // CSV operations
    csv::{read_csv_numeric, write_csv, CsvReaderConfig, CsvWriterConfig},
    // Matrix Market format
    matrix_market::{
        read_dense_matrix, read_sparse_matrix, write_dense_matrix, write_sparse_matrix, MMDataType,
        MMDenseMatrix, MMFormat, MMHeader, MMSparseMatrix, MMSymmetry, SparseEntry,
    },
    // Serialization
    serialize::{deserialize_array, serialize_array, SerializationFormat},
    // WAV files
    wavfile::{read_wav, write_wav},
};

// ========================================
// CSV OPERATIONS
// ========================================

/// Read CSV file into array
#[pyfunction]
#[pyo3(signature = (path, has_header=true, delimiter=","))]
fn read_csv_py(py: Python, path: &str, has_header: bool, delimiter: &str) -> PyResult<Py<PyAny>> {
    let config = CsvReaderConfig {
        has_header,
        delimiter: delimiter.chars().next().unwrap_or(','),
        ..Default::default()
    };

    let (headers, data) = read_csv_numeric(path, Some(config))
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read CSV: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("data", data.into_pyarray(py).unbind())?;
    dict.set_item("headers", headers)?;

    Ok(dict.into())
}

/// Write array to CSV file
#[pyfunction]
#[pyo3(signature = (path, data, headers=None))]
fn write_csv_py(
    path: &str,
    data: &Bound<'_, PyArray2<f64>>,
    headers: Option<Vec<String>>,
) -> PyResult<()> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let arr_owned = arr.to_owned();

    let default_headers: Vec<String> = match &headers {
        Some(h) => h.clone(),
        None => (0..arr.ncols()).map(|i| format!("col_{}", i)).collect(),
    };

    write_csv(
        path,
        &arr_owned,
        Some(&default_headers),
        None::<CsvWriterConfig>,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Failed to write CSV: {}", e)))?;

    Ok(())
}

// ========================================
// MATRIX MARKET FORMAT
// ========================================

/// Read Matrix Market sparse matrix
#[pyfunction]
fn read_matrix_market_sparse_py(py: Python, path: &str) -> PyResult<Py<PyAny>> {
    let matrix = read_sparse_matrix(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read sparse matrix: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("shape", (matrix.rows, matrix.cols))?;
    dict.set_item("nnz", matrix.nnz)?;

    // Extract COO format from SparseEntry structs
    let rows: Vec<usize> = matrix.entries.iter().map(|e| e.row).collect();
    let cols: Vec<usize> = matrix.entries.iter().map(|e| e.col).collect();
    let data: Vec<f64> = matrix.entries.iter().map(|e| e.value).collect();

    dict.set_item("row", rows)?;
    dict.set_item("col", cols)?;
    dict.set_item("data", data)?;

    Ok(dict.into())
}

/// Write sparse matrix in Matrix Market format
#[pyfunction]
fn write_matrix_market_sparse_py(
    path: &str,
    rows: Vec<usize>,
    cols: Vec<usize>,
    data: Vec<f64>,
    shape: (usize, usize),
) -> PyResult<()> {
    let entries: Vec<SparseEntry<f64>> = rows
        .into_iter()
        .zip(cols)
        .zip(data)
        .map(|((r, c), v)| SparseEntry {
            row: r,
            col: c,
            value: v,
        })
        .collect();

    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Coordinate,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: Vec::new(),
    };

    let matrix = MMSparseMatrix {
        header,
        rows: shape.0,
        cols: shape.1,
        nnz: entries.len(),
        entries,
    };

    write_sparse_matrix(path, &matrix)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to write sparse matrix: {}", e)))?;

    Ok(())
}

/// Read Matrix Market dense matrix
#[pyfunction]
fn read_matrix_market_dense_py(py: Python, path: &str) -> PyResult<Py<PyArray2<f64>>> {
    let matrix = read_dense_matrix(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read dense matrix: {}", e)))?;

    Ok(matrix.data.into_pyarray(py).unbind())
}

/// Write dense matrix in Matrix Market format
#[pyfunction]
fn write_matrix_market_dense_py(path: &str, data: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let header = MMHeader {
        object: "matrix".to_string(),
        format: MMFormat::Array,
        data_type: MMDataType::Real,
        symmetry: MMSymmetry::General,
        comments: Vec::new(),
    };

    let matrix = MMDenseMatrix {
        header,
        rows: arr.nrows(),
        cols: arr.ncols(),
        data: arr.to_owned(),
    };

    write_dense_matrix(path, &matrix)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to write dense matrix: {}", e)))?;

    Ok(())
}

// ========================================
// SERIALIZATION
// ========================================

/// Save array to binary file
#[pyfunction]
#[pyo3(signature = (path, data, format="binary"))]
fn save_array_py(path: &str, data: &Bound<'_, PyArray2<f64>>, format: &str) -> PyResult<()> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let ser_format = match format {
        "binary" => SerializationFormat::Binary,
        "json" => SerializationFormat::JSON,
        "messagepack" | "msgpack" => SerializationFormat::MessagePack,
        _ => return Err(PyValueError::new_err(format!("Unknown format: {}", format))),
    };

    // Convert to dynamic dimensionality for serialize_array
    let arr_dyn = arr.to_owned().into_dyn();
    serialize_array::<_, f64, _>(path, &arr_dyn, ser_format)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to save array: {}", e)))?;

    Ok(())
}

/// Load array from binary file
#[pyfunction]
#[pyo3(signature = (path, format="binary"))]
fn load_array_py(py: Python, path: &str, format: &str) -> PyResult<Py<PyArray2<f64>>> {
    let ser_format = match format {
        "binary" => SerializationFormat::Binary,
        "json" => SerializationFormat::JSON,
        "messagepack" | "msgpack" => SerializationFormat::MessagePack,
        _ => return Err(PyValueError::new_err(format!("Unknown format: {}", format))),
    };

    let arr: scirs2_core::ndarray::ArrayD<f64> = deserialize_array(path, ser_format)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to load array: {}", e)))?;

    // Convert to 2D
    let shape = arr.shape();
    if shape.len() == 2 {
        let arr2d = arr
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| PyRuntimeError::new_err(format!("Shape conversion failed: {}", e)))?;
        Ok(arr2d.into_pyarray(py).unbind())
    } else {
        Err(PyValueError::new_err("Array is not 2-dimensional"))
    }
}

// ========================================
// WAV FILE OPERATIONS
// ========================================

/// Read WAV audio file
#[pyfunction]
fn read_wav_py(py: Python, path: &str) -> PyResult<Py<PyAny>> {
    let (header, data) = read_wav(path)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read WAV: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("sample_rate", header.sample_rate)?;
    dict.set_item("channels", header.channels)?;
    dict.set_item("bits_per_sample", header.bits_per_sample)?;

    // Convert to f64 for consistency
    let data_f64: Array1<f64> = data.mapv(|x| x as f64).iter().cloned().collect();
    dict.set_item("data", data_f64.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

/// Write WAV audio file
#[pyfunction]
fn write_wav_py(path: &str, samplerate: u32, data: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Convert to f32
    let data_f32: scirs2_core::ndarray::ArrayD<f32> = arr.mapv(|x| x as f32).into_dyn();

    write_wav(path, samplerate, &data_f32)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to write WAV: {}", e)))?;

    Ok(())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // CSV operations
    m.add_function(wrap_pyfunction!(read_csv_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv_py, m)?)?;

    // Matrix Market format
    m.add_function(wrap_pyfunction!(read_matrix_market_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_matrix_market_sparse_py, m)?)?;
    m.add_function(wrap_pyfunction!(read_matrix_market_dense_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_matrix_market_dense_py, m)?)?;

    // Serialization
    m.add_function(wrap_pyfunction!(save_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_array_py, m)?)?;

    // WAV files
    m.add_function(wrap_pyfunction!(read_wav_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_wav_py, m)?)?;

    Ok(())
}
