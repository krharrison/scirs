//! Python bindings for scirs2-signal
//!
//! Provides signal processing functions similar to scipy.signal

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::python::numpy_compat::{scirs_to_numpy_array1, Array1};
use scirs2_numpy::{PyArray1, PyReadonlyArray1};

// Import signal functions
use scirs2_signal::hilbert::hilbert;

// Import filter design functions
use scirs2_signal::filter::fir::firwin;
use scirs2_signal::filter::iir::{butter, cheby1};
use scirs2_signal::filter::FilterType;

// =============================================================================
// Convolution and Correlation
// =============================================================================

/// Convolve two 1-D arrays - optimized direct implementation
///
/// Parameters:
/// - a: First input array
/// - v: Second input array (kernel)
/// - mode: 'full', 'same', or 'valid'
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
fn convolve_py(
    py: Python,
    a: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    mode: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let a_arr = a.as_array();
    let v_arr = v.as_array();
    let a_slice = a_arr.as_slice().expect("Operation failed");
    let v_slice = v_arr.as_slice().expect("Operation failed");
    let n = a_slice.len();
    let m = v_slice.len();

    if n == 0 || m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must not be empty",
        ));
    }

    // Calculate output size based on mode
    let (out_len, offset) = match mode {
        "full" => (n + m - 1, 0),
        "same" => (n, (m - 1) / 2),
        "valid" => {
            if n < m {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "For 'valid' mode, first array must be at least as long as second",
                ));
            }
            (n - m + 1, m - 1)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be 'full', 'same', or 'valid'",
            ))
        }
    };

    // Direct convolution (optimized for small kernels)
    let mut result = vec![0.0f64; out_len];

    for (i, res) in result.iter_mut().enumerate() {
        let full_idx = i + offset;
        let mut sum = 0.0f64;
        for (j, &vj) in v_slice.iter().enumerate() {
            let ai = full_idx as isize - j as isize;
            if ai >= 0 && (ai as usize) < n {
                sum += a_slice[ai as usize] * vj;
            }
        }
        *res = sum;
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

/// Cross-correlation of two 1-D arrays - optimized direct implementation
///
/// Parameters:
/// - a: First input array
/// - v: Second input array
/// - mode: 'full', 'same', or 'valid'
#[pyfunction]
#[pyo3(signature = (a, v, mode="full"))]
fn correlate_py(
    py: Python,
    a: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
    mode: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let a_arr = a.as_array();
    let v_arr = v.as_array();
    let a_slice = a_arr.as_slice().expect("Operation failed");
    let v_slice = v_arr.as_slice().expect("Operation failed");
    let n = a_slice.len();
    let m = v_slice.len();

    if n == 0 || m == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must not be empty",
        ));
    }

    // Reverse kernel for correlation (correlation = convolution with reversed kernel)
    // Calculate output size based on mode
    let (out_len, offset) = match mode {
        "full" => (n + m - 1, 0),
        "same" => (n, (m - 1) / 2),
        "valid" => {
            if n < m {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "For 'valid' mode, first array must be at least as long as second",
                ));
            }
            (n - m + 1, m - 1)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mode must be 'full', 'same', or 'valid'",
            ))
        }
    };

    // Direct correlation
    let mut result = vec![0.0f64; out_len];

    for (i, res) in result.iter_mut().enumerate() {
        let full_idx = i + offset;
        let mut sum = 0.0f64;
        for (j, &vj) in v_slice.iter().rev().enumerate() {
            let ai = full_idx as isize - j as isize;
            if ai >= 0 && (ai as usize) < n {
                sum += a_slice[ai as usize] * vj;
            }
        }
        *res = sum;
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

// =============================================================================
// Hilbert Transform
// =============================================================================

/// Compute the analytic signal using Hilbert transform
///
/// Returns the analytic signal (real and imaginary parts separately)
#[pyfunction]
fn hilbert_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyAny>> {
    let x_slice = x.as_array().to_vec();

    let result =
        hilbert(&x_slice).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    // Extract real and imaginary parts
    let real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let imag: Vec<f64> = result.iter().map(|c| c.im).collect();

    let dict = PyDict::new(py);
    dict.set_item("real", scirs_to_numpy_array1(Array1::from_vec(real), py)?)?;
    dict.set_item("imag", scirs_to_numpy_array1(Array1::from_vec(imag), py)?)?;

    Ok(dict.into())
}

// =============================================================================
// Window Functions
// =============================================================================

/// Hann window
#[pyfunction]
fn hann_py(py: Python, n: usize) -> PyResult<Py<PyArray1<f64>>> {
    let mut window = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos());
        window.push(val);
    }
    scirs_to_numpy_array1(Array1::from_vec(window), py)
}

/// Hamming window
#[pyfunction]
fn hamming_py(py: Python, n: usize) -> PyResult<Py<PyArray1<f64>>> {
    let mut window = Vec::with_capacity(n);
    for i in 0..n {
        let val = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
        window.push(val);
    }
    scirs_to_numpy_array1(Array1::from_vec(window), py)
}

/// Blackman window
#[pyfunction]
fn blackman_py(py: Python, n: usize) -> PyResult<Py<PyArray1<f64>>> {
    let mut window = Vec::with_capacity(n);
    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
        let val = 0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos();
        window.push(val);
    }
    scirs_to_numpy_array1(Array1::from_vec(window), py)
}

/// Bartlett (triangular) window
#[pyfunction]
fn bartlett_py(py: Python, n: usize) -> PyResult<Py<PyArray1<f64>>> {
    let mut window = Vec::with_capacity(n);
    let half = (n - 1) as f64 / 2.0;
    for i in 0..n {
        let val = 1.0 - ((i as f64 - half) / half).abs();
        window.push(val);
    }
    scirs_to_numpy_array1(Array1::from_vec(window), py)
}

/// Kaiser window
#[pyfunction]
fn kaiser_py(py: Python, n: usize, beta: f64) -> PyResult<Py<PyArray1<f64>>> {
    let mut window = Vec::with_capacity(n);

    // Simple approximation of I0 (modified Bessel function)
    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        for k in 1..50 {
            term *= (x / 2.0).powi(2) / (k as f64).powi(2);
            sum += term;
            if term < 1e-12 {
                break;
            }
        }
        sum
    }

    let denom = bessel_i0(beta);
    for i in 0..n {
        let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        let arg = beta * (1.0 - t * t).sqrt();
        let val = bessel_i0(arg) / denom;
        window.push(val);
    }

    scirs_to_numpy_array1(Array1::from_vec(window), py)
}

// =============================================================================
// Filter Design
// =============================================================================

/// Design a Butterworth digital filter
///
/// Parameters:
/// - order: Filter order
/// - cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
/// - filter_type: 'lowpass', 'highpass'
///
/// Returns:
/// - Dict with 'b' (numerator) and 'a' (denominator) coefficients
#[pyfunction]
#[pyo3(signature = (order, cutoff, filter_type="lowpass"))]
fn butter_py(py: Python, order: usize, cutoff: f64, filter_type: &str) -> PyResult<Py<PyAny>> {
    let ftype = match filter_type.to_lowercase().as_str() {
        "lowpass" | "low" => FilterType::Lowpass,
        "highpass" | "high" => FilterType::Highpass,
        "bandpass" | "band" => FilterType::Bandpass,
        "bandstop" | "stop" => FilterType::Bandstop,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid filter type. Use 'lowpass', 'highpass', 'bandpass', or 'bandstop'",
            ));
        }
    };

    let (b, a) = butter(order, cutoff, ftype)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("b", scirs_to_numpy_array1(Array1::from_vec(b), py)?)?;
    dict.set_item("a", scirs_to_numpy_array1(Array1::from_vec(a), py)?)?;

    Ok(dict.into())
}

/// Design a Chebyshev Type I digital filter
///
/// Parameters:
/// - order: Filter order
/// - ripple: Passband ripple in dB
/// - cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
/// - filter_type: 'lowpass', 'highpass'
///
/// Returns:
/// - Dict with 'b' (numerator) and 'a' (denominator) coefficients
#[pyfunction]
#[pyo3(signature = (order, ripple, cutoff, filter_type="lowpass"))]
fn cheby1_py(
    py: Python,
    order: usize,
    ripple: f64,
    cutoff: f64,
    filter_type: &str,
) -> PyResult<Py<PyAny>> {
    let ftype = match filter_type.to_lowercase().as_str() {
        "lowpass" | "low" => FilterType::Lowpass,
        "highpass" | "high" => FilterType::Highpass,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid filter type for cheby1. Use 'lowpass' or 'highpass'",
            ));
        }
    };

    let (b, a) = cheby1(order, ripple, cutoff, ftype)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("b", scirs_to_numpy_array1(Array1::from_vec(b), py)?)?;
    dict.set_item("a", scirs_to_numpy_array1(Array1::from_vec(a), py)?)?;

    Ok(dict.into())
}

/// Design a FIR filter using window method
///
/// Parameters:
/// - numtaps: Number of filter taps (filter order + 1)
/// - cutoff: Cutoff frequency (normalized 0-1, where 1 is Nyquist)
/// - window: Window function ('hamming', 'hann', 'blackman', 'kaiser')
/// - pass_zero: If true, lowpass; if false, highpass
///
/// Returns:
/// - Filter coefficients as numpy array
#[pyfunction]
#[pyo3(signature = (numtaps, cutoff, window="hamming", pass_zero=true))]
fn firwin_py(
    py: Python,
    numtaps: usize,
    cutoff: f64,
    window: &str,
    pass_zero: bool,
) -> PyResult<Py<PyArray1<f64>>> {
    let coeffs = firwin(numtaps, cutoff, window, pass_zero)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    scirs_to_numpy_array1(Array1::from_vec(coeffs), py)
}

// =============================================================================
// Peak Finding
// =============================================================================

/// Find peaks in a 1-D array
///
/// Returns indices of peaks
#[pyfunction]
#[pyo3(signature = (x, height=None, distance=None))]
fn find_peaks_py(
    py: Python,
    x: PyReadonlyArray1<f64>,
    height: Option<f64>,
    distance: Option<usize>,
) -> PyResult<Py<PyArray1<i64>>> {
    let x_arr = x.as_array();
    let n = x_arr.len();

    if n < 3 {
        return scirs_to_numpy_array1(Array1::from_vec(vec![]), py);
    }

    let mut peaks: Vec<i64> = Vec::new();

    // Find local maxima
    for i in 1..n - 1 {
        if x_arr[i] > x_arr[i - 1] && x_arr[i] > x_arr[i + 1] {
            // Check height threshold
            if let Some(h) = height {
                if x_arr[i] < h {
                    continue;
                }
            }
            peaks.push(i as i64);
        }
    }

    // Apply distance filter
    if let Some(dist) = distance {
        let mut filtered = Vec::new();
        for &peak in &peaks {
            let keep = filtered
                .iter()
                .all(|&p: &i64| (peak - p).unsigned_abs() >= dist as u64);
            if keep {
                filtered.push(peak);
            }
        }
        peaks = filtered;
    }

    scirs_to_numpy_array1(Array1::from_vec(peaks), py)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Convolution/correlation
    m.add_function(wrap_pyfunction!(convolve_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlate_py, m)?)?;

    // Hilbert transform
    m.add_function(wrap_pyfunction!(hilbert_py, m)?)?;

    // Window functions
    m.add_function(wrap_pyfunction!(hann_py, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_py, m)?)?;
    m.add_function(wrap_pyfunction!(blackman_py, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_py, m)?)?;
    m.add_function(wrap_pyfunction!(kaiser_py, m)?)?;

    // Filter design
    m.add_function(wrap_pyfunction!(butter_py, m)?)?;
    m.add_function(wrap_pyfunction!(cheby1_py, m)?)?;
    m.add_function(wrap_pyfunction!(firwin_py, m)?)?;

    // Peak finding
    m.add_function(wrap_pyfunction!(find_peaks_py, m)?)?;

    Ok(())
}
