//! Type conversion utilities for Python integration
//!
//! This module provides comprehensive conversion utilities between Python types
//! and scirs2-core types, with a focus on performance and correctness.

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use scirs2_numpy::{
    Element, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods,
};

// IMPORTANT: Python integration uses scirs2-numpy with ndarray 0.17
#[cfg(feature = "python")]
use ::ndarray::{Array1, Array2, ArrayD};

// ========================================
// SCALAR CONVERSIONS
// ========================================

/// Convert Python float to Rust f32
#[cfg(feature = "python")]
pub fn py_float_to_f32(obj: &Bound<'_, PyAny>) -> PyResult<f32> {
    obj.extract::<f64>().map(|v| v as f32)
}

/// Convert Python float to Rust f64
#[cfg(feature = "python")]
pub fn py_float_to_f64(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
    obj.extract::<f64>()
}

/// Convert Python int to Rust i32
#[cfg(feature = "python")]
pub fn py_int_to_i32(obj: &Bound<'_, PyAny>) -> PyResult<i32> {
    obj.extract::<i64>().map(|v| v as i32)
}

/// Convert Python int to Rust i64
#[cfg(feature = "python")]
pub fn py_int_to_i64(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    obj.extract::<i64>()
}

// ========================================
// LIST/TUPLE CONVERSIONS
// ========================================

/// Convert Python list to Rust `Vec<f32>`
#[cfg(feature = "python")]
pub fn py_list_to_vec_f32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    obj.extract::<Vec<f64>>()
        .map(|v| v.into_iter().map(|x| x as f32).collect())
}

/// Convert Python list to Rust `Vec<i32>`
#[cfg(feature = "python")]
pub fn py_list_to_vec_i32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    obj.extract::<Vec<i64>>()
        .map(|v| v.into_iter().map(|x| x as i32).collect())
}

/// Convert Python list to Rust `Vec<usize>`
#[cfg(feature = "python")]
pub fn py_list_to_vec_usize(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    obj.extract::<Vec<i64>>()
        .map(|v| v.into_iter().map(|x| x as usize).collect())
}

// ========================================
// SHAPE CONVERSIONS
// ========================================

/// Convert Python shape tuple to Rust `Vec<usize>`
#[cfg(feature = "python")]
pub fn py_shape_to_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    py_list_to_vec_usize(obj)
}

/// Convert Rust shape Vec to Python tuple
#[cfg(feature = "python")]
pub fn shape_to_py_tuple(shape: &[usize], py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
    Ok(pyo3::types::PyTuple::new(py, shape)?.unbind())
}

// ========================================
// NUMPY ARRAY CONVERSIONS (GENERIC)
// ========================================

/// Convert NumPy array to scirs2 Array1
#[cfg(feature = "python")]
pub fn numpy_to_scirs_array1<T: Element + Clone>(
    array: &Bound<'_, scirs2_numpy::PyArray1<T>>,
) -> PyResult<Array1<T>> {
    let readonly = array.readonly();
    Ok(readonly.as_array().to_owned())
}

/// Convert NumPy array to scirs2 Array2
#[cfg(feature = "python")]
pub fn numpy_to_scirs_array2<T: Element + Clone>(
    array: &Bound<'_, scirs2_numpy::PyArray2<T>>,
) -> PyResult<Array2<T>> {
    let readonly = array.readonly();
    Ok(readonly.as_array().to_owned())
}

/// Convert scirs2 Array1 to NumPy
#[cfg(feature = "python")]
pub fn scirs_array1_to_numpy<T: Element>(
    array: Array1<T>,
    py: Python<'_>,
) -> PyResult<Py<scirs2_numpy::PyArray1<T>>> {
    Ok(scirs2_numpy::PyArray1::from_owned_array(py, array).unbind())
}

/// Convert scirs2 Array2 to NumPy
#[cfg(feature = "python")]
pub fn scirs_array2_to_numpy<T: Element>(
    array: Array2<T>,
    py: Python<'_>,
) -> PyResult<Py<scirs2_numpy::PyArray2<T>>> {
    Ok(scirs2_numpy::PyArray2::from_owned_array(py, array).unbind())
}

// ========================================
// DTYPE UTILITIES
// ========================================

/// Get NumPy dtype string for a Rust type
#[cfg(feature = "python")]
pub fn rust_dtype_to_numpy_str<T: 'static>() -> &'static str {
    use std::any::TypeId;

    match TypeId::of::<T>() {
        t if t == TypeId::of::<f32>() => "float32",
        t if t == TypeId::of::<f64>() => "float64",
        t if t == TypeId::of::<i8>() => "int8",
        t if t == TypeId::of::<i16>() => "int16",
        t if t == TypeId::of::<i32>() => "int32",
        t if t == TypeId::of::<i64>() => "int64",
        t if t == TypeId::of::<u8>() => "uint8",
        t if t == TypeId::of::<u16>() => "uint16",
        t if t == TypeId::of::<u32>() => "uint32",
        t if t == TypeId::of::<u64>() => "uint64",
        _ => "unknown",
    }
}

// ========================================
// VALIDATION UTILITIES
// ========================================

/// Validate NumPy array meets scirs2 requirements
#[cfg(feature = "python")]
pub fn validate_numpy_array<T: Element>(array: &Bound<'_, PyArrayDyn<T>>) -> PyResult<()> {
    let shape = array.shape();

    // Check for valid dimensions
    if shape.is_empty() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Array must have at least one dimension",
        ));
    }

    // Check for zero-size dimensions
    if shape.contains(&0) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Array dimensions must be non-zero",
        ));
    }

    Ok(())
}

// ========================================
// MEMORY LAYOUT UTILITIES
// ========================================

/// Information about NumPy array memory layout
#[cfg(feature = "python")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLayoutInfo {
    pub is_c_contiguous: bool,
    pub is_f_contiguous: bool,
    pub itemsize: usize,
    pub ndim: usize,
}

/// Get detailed memory layout information from NumPy array
#[cfg(feature = "python")]
pub fn get_memory_layout_info<T: Element>(array: &Bound<'_, PyArrayDyn<T>>) -> MemoryLayoutInfo {
    MemoryLayoutInfo {
        is_c_contiguous: array.is_c_contiguous(),
        is_f_contiguous: array.is_fortran_contiguous(),
        itemsize: std::mem::size_of::<T>(),
        ndim: array.ndim(),
    }
}

// ========================================
// TESTS
// ========================================

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_strings() {
        assert_eq!(rust_dtype_to_numpy_str::<f32>(), "float32");
        assert_eq!(rust_dtype_to_numpy_str::<f64>(), "float64");
        assert_eq!(rust_dtype_to_numpy_str::<i32>(), "int32");
        assert_eq!(rust_dtype_to_numpy_str::<i64>(), "int64");
    }

    #[test]
    #[allow(deprecated)]
    fn test_shape_conversion() {
        pyo3::Python::attach(|py| {
            let shape = vec![2, 3, 4];
            let py_tuple = shape_to_py_tuple(&shape, py).expect("Operation failed");
            let bound_tuple = py_tuple.bind(py);

            assert_eq!(bound_tuple.len(), 3);
            assert_eq!(
                bound_tuple
                    .get_item(0)
                    .expect("Operation failed")
                    .extract::<usize>()
                    .expect("Operation failed"),
                2
            );
            assert_eq!(
                bound_tuple
                    .get_item(1)
                    .expect("Operation failed")
                    .extract::<usize>()
                    .expect("Operation failed"),
                3
            );
            assert_eq!(
                bound_tuple
                    .get_item(2)
                    .expect("Operation failed")
                    .extract::<usize>()
                    .expect("Operation failed"),
                4
            );
        });
    }
}
