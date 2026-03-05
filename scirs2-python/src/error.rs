//! Error handling for scirs2-python
//!
//! This module provides comprehensive error types and conversions for Python exceptions.

use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::fmt;

/// Main error type for scirs2-python operations
#[derive(Debug)]
pub enum SciRS2Error {
    /// Array operation error (dimension mismatch, invalid shape, etc.)
    ArrayError(String),
    /// Numerical computation error (NaN, infinity, convergence failure)
    ComputationError(String),
    /// Invalid input parameters
    ValueError(String),
    /// Type conversion error
    TypeError(String),
    /// Index out of bounds
    IndexError(String),
    /// Memory allocation error
    MemoryError(String),
    /// Generic runtime error
    RuntimeError(String),
}

impl fmt::Display for SciRS2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArrayError(msg) => write!(f, "Array error: {}", msg),
            Self::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            Self::ValueError(msg) => write!(f, "Value error: {}", msg),
            Self::TypeError(msg) => write!(f, "Type error: {}", msg),
            Self::IndexError(msg) => write!(f, "Index error: {}", msg),
            Self::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            Self::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for SciRS2Error {}

impl From<SciRS2Error> for PyErr {
    fn from(err: SciRS2Error) -> PyErr {
        match err {
            SciRS2Error::ArrayError(msg)
            | SciRS2Error::ComputationError(msg)
            | SciRS2Error::RuntimeError(msg) => PyRuntimeError::new_err(msg),
            SciRS2Error::ValueError(msg) => PyValueError::new_err(msg),
            SciRS2Error::TypeError(msg) => PyTypeError::new_err(msg),
            SciRS2Error::IndexError(msg) => PyIndexError::new_err(msg),
            SciRS2Error::MemoryError(msg) => {
                PyRuntimeError::new_err(format!("Memory error: {}", msg))
            }
        }
    }
}

/// Helper trait for converting numpy array access errors
pub trait ArrayAccessExt<T> {
    fn to_scirs2_err(self, context: &str) -> Result<T, SciRS2Error>;
}

impl<T> ArrayAccessExt<T> for Option<T> {
    fn to_scirs2_err(self, context: &str) -> Result<T, SciRS2Error> {
        self.ok_or_else(|| SciRS2Error::ArrayError(format!("{}: array access failed", context)))
    }
}

/// Helper function to safely get array slice with context
pub fn get_array_slice<'a, T>(
    arr: &'a ndarray::ArrayView1<'_, T>,
    context: &str,
) -> Result<&'a [T], SciRS2Error>
where
    T: ndarray::NdFloat,
{
    arr.as_slice()
        .ok_or_else(|| SciRS2Error::ArrayError(format!("{}: array is not contiguous", context)))
}

/// Helper function to safely get 2D array slice with context
pub fn get_array_slice_2d<'a, T>(
    arr: &'a ndarray::ArrayView2<'_, T>,
    context: &str,
) -> Result<&'a [T], SciRS2Error>
where
    T: ndarray::NdFloat,
{
    arr.as_slice().ok_or_else(|| {
        SciRS2Error::ArrayError(format!(
            "{}: array is not contiguous or not in standard layout",
            context
        ))
    })
}

/// Helper function to validate array is not empty
pub fn check_not_empty<T>(
    arr: &ndarray::ArrayView1<'_, T>,
    operation: &str,
) -> Result<(), SciRS2Error> {
    if arr.is_empty() {
        Err(SciRS2Error::ValueError(format!(
            "{}: array must not be empty",
            operation
        )))
    } else {
        Ok(())
    }
}

/// Helper function to validate arrays have same length
pub fn check_same_length<T, U>(
    arr1: &ndarray::ArrayView1<'_, T>,
    arr2: &ndarray::ArrayView1<'_, U>,
    operation: &str,
) -> Result<(), SciRS2Error> {
    if arr1.len() != arr2.len() {
        Err(SciRS2Error::ValueError(format!(
            "{}: arrays must have same length (got {} and {})",
            operation,
            arr1.len(),
            arr2.len()
        )))
    } else {
        Ok(())
    }
}

/// Helper function to validate minimum array length
pub fn check_min_length<T>(
    arr: &ndarray::ArrayView1<'_, T>,
    min_len: usize,
    operation: &str,
) -> Result<(), SciRS2Error> {
    if arr.len() < min_len {
        Err(SciRS2Error::ValueError(format!(
            "{}: array must have at least {} elements (got {})",
            operation,
            min_len,
            arr.len()
        )))
    } else {
        Ok(())
    }
}
