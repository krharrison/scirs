//! Extended Python bindings for scirs2-signal
//!
//! Provides bindings for v0.3.0 features:
//! - Spectral analysis (STFT, periodogram, Welch's method)
//! - Continuous Wavelet Transform
//! - Signal generation (chirp, ricker wavelet)

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2};
use scirs2_core::ndarray::{Array1, Array2};

// =============================================================================
// Spectral Analysis
// =============================================================================

/// Compute Short-Time Fourier Transform (STFT).
///
/// The STFT represents a signal in the time-frequency domain by dividing
/// it into overlapping segments and computing the FFT of each segment.
///
/// Parameters:
///     x: Input signal (1D array)
///     fs: Sampling frequency (default: 1.0)
///     window: Window function - 'hann', 'hamming', 'blackman', etc. (default: 'hann')
///     nperseg: Length of each segment (default: 256)
///     noverlap: Number of points to overlap (default: nperseg // 2)
///     nfft: Length of the FFT (default: nperseg)
///
/// Returns:
///     Dict with:
///     - 'freqs': Frequency array
///     - 'times': Time array
///     - 'stft_real': Real part of STFT (2D array, shape: [freq, time])
///     - 'stft_imag': Imaginary part of STFT (2D array, shape: [freq, time])
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None))]
pub fn stft_py(
    py: Python,
    x: Vec<f64>,
    fs: f64,
    window: &str,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if x.is_empty() {
        return Err(PyValueError::new_err("x must not be empty"));
    }

    let result = scirs2_signal::stft(
        &x,
        Some(fs),
        Some(window),
        nperseg,
        noverlap,
        nfft,
        None,
        None,
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("STFT failed: {}", e)))?;

    let (freqs, times, stft_complex) = result;

    let n_freqs = freqs.len();
    let n_times = times.len();

    let mut stft_real: Vec<f64> = Vec::with_capacity(n_freqs * n_times);
    let mut stft_imag: Vec<f64> = Vec::with_capacity(n_freqs * n_times);

    for row in &stft_complex {
        for c in row {
            stft_real.push(c.re);
            stft_imag.push(c.im);
        }
    }

    let freqs_arr = Array1::from_vec(freqs);
    let times_arr = Array1::from_vec(times);
    let stft_real_arr = Array2::from_shape_vec((n_freqs, n_times), stft_real)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape STFT real: {}", e)))?;
    let stft_imag_arr = Array2::from_shape_vec((n_freqs, n_times), stft_imag)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape STFT imag: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("freqs", freqs_arr.into_pyarray(py))?;
    dict.set_item("times", times_arr.into_pyarray(py))?;
    dict.set_item("stft_real", stft_real_arr.into_pyarray(py))?;
    dict.set_item("stft_imag", stft_imag_arr.into_pyarray(py))?;

    Ok(dict.into())
}

/// Compute power spectral density using Welch's method.
///
/// Welch's method averages periodograms of overlapping segments to
/// produce a lower-variance estimate of the PSD.
///
/// Parameters:
///     x: Input signal (1D array)
///     fs: Sampling frequency (default: 1.0)
///     window: Window function (default: 'hann')
///     nperseg: Length of each segment (default: 256)
///     noverlap: Points to overlap between segments (default: None = nperseg//2)
///     nfft: FFT length (default: nperseg)
///     scaling: 'density' or 'spectrum' (default: 'density')
///
/// Returns:
///     Dict with:
///     - 'freqs': Frequency array
///     - 'psd': Power spectral density
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None, scaling="density"))]
pub fn welch_py(
    py: Python,
    x: Vec<f64>,
    fs: f64,
    window: &str,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    scaling: &str,
) -> PyResult<Py<PyAny>> {
    if x.is_empty() {
        return Err(PyValueError::new_err("x must not be empty"));
    }

    let result = scirs2_signal::welch(
        &x,
        Some(fs),
        Some(window),
        nperseg,
        noverlap,
        nfft,
        None,
        Some(scaling),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Welch PSD failed: {}", e)))?;

    let (freqs, psd) = result;

    let dict = PyDict::new(py);
    dict.set_item("freqs", Array1::from_vec(freqs).into_pyarray(py))?;
    dict.set_item("psd", Array1::from_vec(psd).into_pyarray(py))?;

    Ok(dict.into())
}

/// Estimate power spectral density using a periodogram.
///
/// Computes the periodogram for the given signal using the discrete Fourier
/// transform. Considers a simple windowing approach.
///
/// Parameters:
///     x: Input signal (1D array)
///     fs: Sampling frequency (default: 1.0)
///     window: Window function (default: 'boxcar')
///     nfft: FFT length (default: len(x))
///     scaling: 'density' or 'spectrum' (default: 'density')
///
/// Returns:
///     Dict with:
///     - 'freqs': Frequency array
///     - 'psd': Power spectral density
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, window=None, nfft=None, scaling="density"))]
pub fn periodogram_py(
    py: Python,
    x: Vec<f64>,
    fs: f64,
    window: Option<&str>,
    nfft: Option<usize>,
    scaling: &str,
) -> PyResult<Py<PyAny>> {
    if x.is_empty() {
        return Err(PyValueError::new_err("x must not be empty"));
    }

    let result = scirs2_signal::periodogram(
        &x,
        Some(fs),
        window,
        nfft,
        None,
        Some(scaling),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Periodogram failed: {}", e)))?;

    let (freqs, psd) = result;

    let dict = PyDict::new(py);
    dict.set_item("freqs", Array1::from_vec(freqs).into_pyarray(py))?;
    dict.set_item("psd", Array1::from_vec(psd).into_pyarray(py))?;

    Ok(dict.into())
}

/// Compute spectrogram (squared magnitude of STFT).
///
/// Parameters:
///     x: Input signal (1D array)
///     fs: Sampling frequency (default: 1.0)
///     window: Window function (default: 'hann')
///     nperseg: Length of each segment (default: 256)
///     noverlap: Number of overlapping points (default: nperseg//2)
///     nfft: FFT length (default: nperseg)
///
/// Returns:
///     Dict with:
///     - 'freqs': Frequency array
///     - 'times': Time array
///     - 'Sxx': Power spectral density (2D array: [freq, time])
#[pyfunction]
#[pyo3(signature = (x, fs=1.0, window="hann", nperseg=None, noverlap=None, nfft=None))]
pub fn spectrogram_py(
    py: Python,
    x: Vec<f64>,
    fs: f64,
    window: &str,
    nperseg: Option<usize>,
    noverlap: Option<usize>,
    nfft: Option<usize>,
) -> PyResult<Py<PyAny>> {
    if x.is_empty() {
        return Err(PyValueError::new_err("x must not be empty"));
    }

    let result = scirs2_signal::spectrogram(
        &x,
        Some(fs),
        Some(window),
        nperseg,
        noverlap,
        nfft,
        None,
        None,
        None,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Spectrogram failed: {}", e)))?;

    let (freqs, times, sxx) = result;
    let n_freqs = freqs.len();
    let n_times = times.len();

    let sxx_flat: Vec<f64> = sxx.into_iter().flatten().collect();
    let sxx_arr = Array2::from_shape_vec((n_freqs, n_times), sxx_flat)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to reshape Sxx: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("freqs", Array1::from_vec(freqs).into_pyarray(py))?;
    dict.set_item("times", Array1::from_vec(times).into_pyarray(py))?;
    dict.set_item("Sxx", sxx_arr.into_pyarray(py))?;

    Ok(dict.into())
}

// =============================================================================
// Continuous Wavelet Transform
// =============================================================================

/// Compute the Ricker wavelet (Mexican hat wavelet).
///
/// Parameters:
///     points: Number of points in the wavelet
///     a: Width parameter
///
/// Returns:
///     1D numpy array containing the wavelet values
#[pyfunction]
pub fn ricker_py(
    py: Python,
    points: usize,
    a: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if points == 0 {
        return Err(PyValueError::new_err("points must be > 0"));
    }
    if a <= 0.0 {
        return Err(PyValueError::new_err("a must be positive"));
    }
    let wav = scirs2_signal::ricker(points, a)
        .map_err(|e| PyRuntimeError::new_err(format!("Ricker wavelet failed: {}", e)))?;
    Ok(Array1::from_vec(wav).into_pyarray(py).unbind())
}

// =============================================================================
// Signal Generation
// =============================================================================

/// Generate a chirp signal (frequency-swept cosine).
///
/// Parameters:
///     t: Time array (list of floats)
///     f0: Frequency at time 0 in Hz
///     t1: Time at which f1 is specified
///     f1: Frequency at time t1 in Hz
///     method: Sweep method - 'linear', 'quadratic', 'logarithmic', 'hyperbolic' (default: 'linear')
///     phi: Phase offset in degrees (default: 0.0)
///
/// Returns:
///     1D numpy array of chirp signal values
#[pyfunction]
#[pyo3(signature = (t, f0, t1, f1, method="linear", phi=0.0))]
pub fn chirp_py(
    py: Python,
    t: Vec<f64>,
    f0: f64,
    t1: f64,
    f1: f64,
    method: &str,
    phi: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if t.is_empty() {
        return Err(PyValueError::new_err("t must not be empty"));
    }
    let result = scirs2_signal::chirp(&t, f0, t1, f1, method, phi)
        .map_err(|e| PyRuntimeError::new_err(format!("Chirp generation failed: {}", e)))?;
    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// Generate a square wave.
///
/// Parameters:
///     t: Time array (list of floats)
///     duty: Duty cycle between 0 and 1 (default: 0.5)
///
/// Returns:
///     1D numpy array of square wave values (-1.0 or 1.0)
#[pyfunction]
#[pyo3(signature = (t, duty=0.5))]
pub fn square_py(
    py: Python,
    t: Vec<f64>,
    duty: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if t.is_empty() {
        return Err(PyValueError::new_err("t must not be empty"));
    }
    if !(0.0..=1.0).contains(&duty) {
        return Err(PyValueError::new_err("duty must be between 0 and 1"));
    }
    let result = scirs2_signal::square(&t, duty)
        .map_err(|e| PyRuntimeError::new_err(format!("Square wave generation failed: {}", e)))?;
    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// Generate a sawtooth wave.
///
/// Parameters:
///     t: Time array (list of floats)
///     width: Width of the rising ramp (default: 1.0 for pure sawtooth)
///
/// Returns:
///     1D numpy array of sawtooth wave values (-1.0 to 1.0)
#[pyfunction]
#[pyo3(signature = (t, width=1.0))]
pub fn sawtooth_py(
    py: Python,
    t: Vec<f64>,
    width: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    if t.is_empty() {
        return Err(PyValueError::new_err("t must not be empty"));
    }
    let result = scirs2_signal::sawtooth(&t, width)
        .map_err(|e| PyRuntimeError::new_err(format!("Sawtooth wave generation failed: {}", e)))?;
    Ok(Array1::from_vec(result).into_pyarray(py).unbind())
}

/// Python module registration for signal extensions
pub fn register_signal_ext_module(m: &Bound<'_, pyo3::PyModule>) -> pyo3::PyResult<()> {
    // Spectral analysis
    m.add_function(wrap_pyfunction!(stft_py, m)?)?;
    m.add_function(wrap_pyfunction!(welch_py, m)?)?;
    m.add_function(wrap_pyfunction!(periodogram_py, m)?)?;
    m.add_function(wrap_pyfunction!(spectrogram_py, m)?)?;

    // Wavelets
    m.add_function(wrap_pyfunction!(ricker_py, m)?)?;

    // Signal generation
    m.add_function(wrap_pyfunction!(chirp_py, m)?)?;
    m.add_function(wrap_pyfunction!(square_py, m)?)?;
    m.add_function(wrap_pyfunction!(sawtooth_py, m)?)?;

    Ok(())
}
