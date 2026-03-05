//! Python bindings for scirs2-fft
//!
//! This module provides Python bindings for Fast Fourier Transform operations.
//!
//! FFT functions return NumPy complex128 arrays for optimal performance and
//! compatibility with NumPy's FFT functions.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
// Complex64 from scirs2_numpy maps to NumPy's complex128 (cdouble)
use scirs2_numpy::{Complex64 as NumpyComplex64, IntoPyArray, PyArray1, PyArrayMethods};

// ndarray types from scirs2-core
use scirs2_core::{numeric::Complex64, Array1};

// Direct imports from scirs2-fft (native ndarray 0.17 support)
use scirs2_fft::{dct, fftfreq, fftshift, idct, ifftshift, next_fast_len, rfftfreq, DCTType};

// Fallback imports for non-OxiFFT builds
#[cfg(not(feature = "oxifft"))]
use scirs2_fft::{fft, ifft, irfft};

// ========================================
// CORE FFT FUNCTIONS
// ========================================

/// 1D FFT (OxiFFT-optimized for f64!)
/// Returns NumPy complex128 array (compatible with np.fft.fft output)
#[pyfunction]
fn fft_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for f64 (Pure Rust high-performance!)
    #[cfg(feature = "oxifft")]
    {
        // Convert real input to complex for fft_oxifft
        let complex_input: scirs2_core::ndarray::Array1<Complex64> =
            arr.iter().map(|&r| Complex64::new(r, 0.0)).collect();

        let result = scirs2_fft::oxifft_backend::fft_oxifft(&complex_input.view())
            .map_err(|e| PyRuntimeError::new_err(format!("FFT (OxiFFT) failed: {}", e)))?;

        // Convert to NumPy complex128 array
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind())
    }

    // Fallback to pure Rust
    #[cfg(not(feature = "oxifft"))]
    {
        let vec_data: Vec<f64> = arr.to_vec();

        let result = fft(&vec_data, None)
            .map_err(|e| PyRuntimeError::new_err(format!("FFT failed: {}", e)))?;

        // Convert to NumPy complex128 array
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        return Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind());
    }
}

/// 1D inverse FFT (OxiFFT-optimized!)
/// Accepts and returns NumPy complex128 arrays (compatible with np.fft.ifft)
#[pyfunction]
fn ifft_py(
    py: Python,
    data: &Bound<'_, PyArray1<NumpyComplex64>>,
) -> PyResult<Py<PyArray1<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for f64 (Pure Rust high-performance!)
    #[cfg(feature = "oxifft")]
    {
        // Convert NumPy complex to scirs2 Complex64
        let complex_input: scirs2_core::ndarray::Array1<Complex64> =
            arr.iter().map(|c| Complex64::new(c.re, c.im)).collect();

        let result = scirs2_fft::oxifft_backend::ifft_oxifft(&complex_input.view())
            .map_err(|e| PyRuntimeError::new_err(format!("IFFT (OxiFFT) failed: {}", e)))?;

        // Convert back to NumPy complex128
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind())
    }

    // Fallback to pure Rust
    #[cfg(not(feature = "oxifft"))]
    {
        // Convert NumPy complex to Vec<Complex64>
        let complex_input: Vec<Complex64> =
            arr.iter().map(|c| Complex64::new(c.re, c.im)).collect();

        let result = ifft(&complex_input, None)
            .map_err(|e| PyRuntimeError::new_err(format!("IFFT failed: {}", e)))?;

        // Convert back to NumPy complex128
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        return Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind());
    }
}

/// Real FFT - FFT of real-valued input (OxiFFT-optimized!)
/// Returns NumPy complex128 array (compatible with np.fft.rfft output)
#[pyfunction]
fn rfft_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for f64 (OxiFFT + plan caching = high performance!)
    #[cfg(feature = "oxifft")]
    {
        let result = scirs2_fft::oxifft_backend::rfft_oxifft(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("RFFT (OxiFFT) failed: {}", e)))?;

        // Convert to NumPy complex128 array
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind())
    }

    // Fallback to pure Rust
    #[cfg(not(feature = "oxifft"))]
    {
        let vec_data: Vec<f64> = arr.to_vec();

        let result = rfft(&vec_data, None)
            .map_err(|e| PyRuntimeError::new_err(format!("RFFT failed: {}", e)))?;

        // Convert to NumPy complex128 array
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        return Ok(Array1::from_vec(complex_result).into_pyarray(py).unbind());
    }
}

/// Inverse real FFT (OxiFFT-optimized!)
/// Accepts NumPy complex128 array, returns real f64 array (compatible with np.fft.irfft)
#[pyfunction]
#[pyo3(signature = (data, n=None))]
fn irfft_py(
    py: Python,
    data: &Bound<'_, PyArray1<NumpyComplex64>>,
    n: Option<usize>,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for f64 (Pure Rust high-performance!)
    #[cfg(feature = "oxifft")]
    {
        // Convert NumPy complex to scirs2 Complex64
        let complex_input: scirs2_core::ndarray::Array1<Complex64> =
            arr.iter().map(|c| Complex64::new(c.re, c.im)).collect();

        // Infer output size: if n is None, assume n = 2*(input_len - 1)
        let output_len = n.unwrap_or_else(|| 2 * (complex_input.len() - 1));

        let result = scirs2_fft::oxifft_backend::irfft_oxifft(&complex_input.view(), output_len)
            .map_err(|e| PyRuntimeError::new_err(format!("IRFFT (OxiFFT) failed: {}", e)))?;

        Ok(result.into_pyarray(py).unbind())
    }

    // Fallback to pure Rust
    #[cfg(not(feature = "oxifft"))]
    {
        // Convert NumPy complex to Vec<Complex64>
        let complex_input: Vec<Complex64> =
            arr.iter().map(|c| Complex64::new(c.re, c.im)).collect();

        let result = irfft(&complex_input, n)
            .map_err(|e| PyRuntimeError::new_err(format!("IRFFT failed: {}", e)))?;

        return Ok(Array1::from_vec(result).into_pyarray(py).unbind());
    }
}

// ========================================
// DCT FUNCTIONS
// ========================================

/// Discrete Cosine Transform (OxiFFT-optimized for Type 2!)
/// dct_type: 1, 2, 3, or 4
#[pyfunction]
#[pyo3(signature = (data, dct_type=2))]
fn dct_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    dct_type: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for DCT Type 2 (most common)
    #[cfg(feature = "oxifft")]
    if dct_type == 2 {
        let result = scirs2_fft::oxifft_backend::dct2_oxifft(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("DCT-II (OxiFFT) failed: {}", e)))?;
        return Ok(result.into_pyarray(py).unbind());
    }

    // Fallback to pure Rust for other types
    let vec_data: Vec<f64> = arr.to_vec();
    let dct_type_enum = match dct_type {
        1 => DCTType::Type1,
        2 => DCTType::Type2,
        3 => DCTType::Type3,
        4 => DCTType::Type4,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid DCT type: {}",
                dct_type
            )))
        }
    };

    let result = dct(&vec_data, Some(dct_type_enum), None)
        .map_err(|e| PyRuntimeError::new_err(format!("DCT failed: {}", e)))?;

    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// Inverse Discrete Cosine Transform (OxiFFT-optimized for Type 2!)
/// dct_type: 1, 2, 3, or 4
#[pyfunction]
#[pyo3(signature = (data, dct_type=2))]
fn idct_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    dct_type: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    // Use OxiFFT for IDCT Type 2 (most common)
    #[cfg(feature = "oxifft")]
    if dct_type == 2 {
        let result = scirs2_fft::oxifft_backend::idct2_oxifft(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("IDCT-II (OxiFFT) failed: {}", e)))?;
        return Ok(result.into_pyarray(py).unbind());
    }

    // Fallback to pure Rust for other types
    let vec_data: Vec<f64> = arr.to_vec();
    let dct_type_enum = match dct_type {
        1 => DCTType::Type1,
        2 => DCTType::Type2,
        3 => DCTType::Type3,
        4 => DCTType::Type4,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid DCT type: {}",
                dct_type
            )))
        }
    };

    let result = idct(&vec_data, Some(dct_type_enum), None)
        .map_err(|e| PyRuntimeError::new_err(format!("IDCT failed: {}", e)))?;

    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

// ========================================
// 2D FFT FUNCTIONS (OxiFFT-optimized!)
// ========================================

/// 2D FFT - Returns NumPy complex128 2D array (compatible with np.fft.fft2)
/// Optimized: Uses rfft2 + Hermitian symmetry to reconstruct full spectrum
/// This avoids the slow real→complex conversion for the input
#[pyfunction]
fn fft2_py(
    py: Python,
    data: &Bound<'_, scirs2_numpy::PyArray2<f64>>,
) -> PyResult<Py<scirs2_numpy::PyArray2<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    #[cfg(feature = "oxifft")]
    {
        let (rows, cols) = arr.dim();

        // Use rfft2 which is much faster for real input (no complex conversion needed)
        let half_result = scirs2_fft::oxifft_backend::rfft2_oxifft(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("FFT2 (OxiFFT) failed: {}", e)))?;

        // Reconstruct full spectrum using Hermitian symmetry
        // rfft2 gives us columns 0 to cols/2, we need to fill cols/2+1 to cols-1
        let half_cols = cols / 2 + 1;
        let mut full_result: Vec<NumpyComplex64> = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            // Copy the first half (cols 0 to half_cols-1)
            for col in 0..half_cols {
                let c = half_result[[row, col]];
                full_result.push(NumpyComplex64::new(c.re, c.im));
            }

            // Reconstruct the second half using Hermitian symmetry
            // For real input: X[k1, k2] = conj(X[N1-k1, N2-k2])
            for col in half_cols..cols {
                let conj_row = if row == 0 { 0 } else { rows - row };
                let conj_col = cols - col;
                let c = half_result[[conj_row, conj_col]];
                full_result.push(NumpyComplex64::new(c.re, -c.im)); // conjugate
            }
        }

        let result_array = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), full_result)
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;

        Ok(result_array.into_pyarray(py).unbind())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = arr;
        Err(PyRuntimeError::new_err("FFT2 requires oxifft feature"))
    }
}

/// 2D real FFT - Returns NumPy complex128 2D array (compatible with np.fft.rfft2)
#[pyfunction]
fn rfft2_py(
    py: Python,
    data: &Bound<'_, scirs2_numpy::PyArray2<f64>>,
) -> PyResult<Py<scirs2_numpy::PyArray2<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    #[cfg(feature = "oxifft")]
    {
        let result = scirs2_fft::oxifft_backend::rfft2_oxifft(&arr)
            .map_err(|e| PyRuntimeError::new_err(format!("RFFT2 (OxiFFT) failed: {}", e)))?;

        // Convert to NumPy complex128 2D array
        let (rows, cols) = result.dim();
        let complex_result: Vec<NumpyComplex64> = result
            .iter()
            .map(|c| NumpyComplex64::new(c.re, c.im))
            .collect();

        let result_array =
            scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), complex_result)
                .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;

        Ok(result_array.into_pyarray(py).unbind())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = arr;
        Err(PyRuntimeError::new_err("RFFT2 requires oxifft feature"))
    }
}

/// 2D inverse FFT - Accepts and returns NumPy complex128 2D arrays (compatible with np.fft.ifft2)
/// Optimized: Direct allocation without intermediate conversions
#[pyfunction]
fn ifft2_py(
    py: Python,
    data: &Bound<'_, scirs2_numpy::PyArray2<NumpyComplex64>>,
) -> PyResult<Py<scirs2_numpy::PyArray2<NumpyComplex64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    #[cfg(feature = "oxifft")]
    {
        let (rows, cols) = arr.dim();
        let n = rows * cols;

        // Optimized: Direct allocation with capacity (no intermediate Array1)
        let mut complex_vec: Vec<Complex64> = Vec::with_capacity(n);
        for c in arr.iter() {
            complex_vec.push(Complex64::new(c.re, c.im));
        }
        let complex_input = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), complex_vec)
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;

        let result = scirs2_fft::oxifft_backend::ifft2_oxifft(&complex_input.view())
            .map_err(|e| PyRuntimeError::new_err(format!("IFFT2 (OxiFFT) failed: {}", e)))?;

        // Direct allocation for output
        let mut result_vec: Vec<NumpyComplex64> = Vec::with_capacity(n);
        for c in result.iter() {
            result_vec.push(NumpyComplex64::new(c.re, c.im));
        }

        let result_array = scirs2_core::ndarray::Array2::from_shape_vec((rows, cols), result_vec)
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;

        Ok(result_array.into_pyarray(py).unbind())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = arr;
        Err(PyRuntimeError::new_err("IFFT2 requires oxifft feature"))
    }
}

/// 2D inverse real FFT - Accepts NumPy complex128 2D array, returns real 2D array
#[pyfunction]
#[pyo3(signature = (data, shape))]
fn irfft2_py(
    py: Python,
    data: &Bound<'_, scirs2_numpy::PyArray2<NumpyComplex64>>,
    shape: (usize, usize),
) -> PyResult<Py<scirs2_numpy::PyArray2<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    #[cfg(feature = "oxifft")]
    {
        let (in_rows, in_cols) = arr.dim();

        // Convert NumPy complex to scirs2 Complex64
        let complex_input: scirs2_core::ndarray::Array2<Complex64> = arr
            .iter()
            .map(|c| Complex64::new(c.re, c.im))
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<scirs2_core::ndarray::Array1<_>>()
            .into_shape_with_order((in_rows, in_cols))
            .map_err(|e| PyRuntimeError::new_err(format!("Shape error: {}", e)))?;

        let result = scirs2_fft::oxifft_backend::irfft2_oxifft(&complex_input.view(), shape)
            .map_err(|e| PyRuntimeError::new_err(format!("IRFFT2 (OxiFFT) failed: {}", e)))?;

        Ok(result.into_pyarray(py).unbind())
    }

    #[cfg(not(feature = "oxifft"))]
    {
        let _ = (arr, shape);
        Err(PyRuntimeError::new_err("IRFFT2 requires oxifft feature"))
    }
}

// ========================================
// HELPER FUNCTIONS
// ========================================

/// FFT sample frequencies
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
fn fftfreq_py(py: Python, n: usize, d: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result =
        fftfreq(n, d).map_err(|e| PyRuntimeError::new_err(format!("FFT freq failed: {}", e)))?;
    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// Real FFT sample frequencies
#[pyfunction]
#[pyo3(signature = (n, d=1.0))]
fn rfftfreq_py(py: Python, n: usize, d: f64) -> PyResult<Py<PyArray1<f64>>> {
    let result =
        rfftfreq(n, d).map_err(|e| PyRuntimeError::new_err(format!("RFFT freq failed: {}", e)))?;
    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// FFT shift - shift zero-frequency component to center
#[pyfunction]
fn fftshift_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array().to_owned();

    let result =
        fftshift(&arr).map_err(|e| PyRuntimeError::new_err(format!("FFT shift failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Inverse FFT shift
#[pyfunction]
fn ifftshift_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array().to_owned();

    let result = ifftshift(&arr)
        .map_err(|e| PyRuntimeError::new_err(format!("Inverse FFT shift failed: {}", e)))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Find next fast length for FFT
/// Returns the smallest size >= n that can be efficiently transformed
#[pyfunction]
#[pyo3(signature = (n, real=false))]
fn next_fast_len_py(n: usize, real: bool) -> usize {
    next_fast_len(n, real)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core FFT functions
    m.add_function(wrap_pyfunction!(fft_py, m)?)?;
    m.add_function(wrap_pyfunction!(ifft_py, m)?)?;
    m.add_function(wrap_pyfunction!(rfft_py, m)?)?;
    m.add_function(wrap_pyfunction!(irfft_py, m)?)?;

    // DCT functions
    m.add_function(wrap_pyfunction!(dct_py, m)?)?;
    m.add_function(wrap_pyfunction!(idct_py, m)?)?;

    // 2D FFT functions
    m.add_function(wrap_pyfunction!(fft2_py, m)?)?;
    m.add_function(wrap_pyfunction!(ifft2_py, m)?)?;
    m.add_function(wrap_pyfunction!(rfft2_py, m)?)?;
    m.add_function(wrap_pyfunction!(irfft2_py, m)?)?;

    // Helper functions
    m.add_function(wrap_pyfunction!(fftfreq_py, m)?)?;
    m.add_function(wrap_pyfunction!(rfftfreq_py, m)?)?;
    m.add_function(wrap_pyfunction!(fftshift_py, m)?)?;
    m.add_function(wrap_pyfunction!(ifftshift_py, m)?)?;
    m.add_function(wrap_pyfunction!(next_fast_len_py, m)?)?;

    Ok(())
}
