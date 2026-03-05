//! Python bindings for scirs2-special
//!
//! Provides special mathematical functions similar to scipy.special

use pyo3::prelude::*;
use scirs2_core::python::numpy_compat::{scirs_to_numpy_array1, Array1};
use scirs2_numpy::{PyArray1, PyReadonlyArray1};

// Import special functions from scirs2-special
use scirs2_special::{bessel, erf as erf_mod, gamma as gamma_fn};

// =============================================================================
// Gamma Functions
// =============================================================================

/// Gamma function
#[pyfunction]
fn gamma_py(x: f64) -> PyResult<f64> {
    Ok(gamma_fn(x))
}

/// Log-gamma function
#[pyfunction]
fn lgamma_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::gammaln(x))
}

/// Digamma (psi) function
#[pyfunction]
fn digamma_py(x: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::digamma(x))
}

/// Beta function
#[pyfunction]
fn beta_py(a: f64, b: f64) -> PyResult<f64> {
    Ok(scirs2_special::gamma::beta(a, b))
}

// =============================================================================
// Bessel Functions
// =============================================================================

/// Bessel function of the first kind, J0
#[pyfunction]
fn j0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::j0(x))
}

/// Bessel function of the first kind, J1
#[pyfunction]
fn j1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::j1(x))
}

/// Bessel function of the first kind, Jn
#[pyfunction]
fn jn_py(n: i32, x: f64) -> PyResult<f64> {
    Ok(bessel::jn(n, x))
}

/// Bessel function of the second kind, Y0
#[pyfunction]
fn y0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::y0(x))
}

/// Bessel function of the second kind, Y1
#[pyfunction]
fn y1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::y1(x))
}

/// Bessel function of the second kind, Yn
#[pyfunction]
fn yn_py(n: i32, x: f64) -> PyResult<f64> {
    Ok(bessel::yn(n, x))
}

/// Modified Bessel function of the first kind, I0
#[pyfunction]
fn i0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::i0(x))
}

/// Modified Bessel function of the first kind, I1
#[pyfunction]
fn i1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::i1(x))
}

/// Modified Bessel function of the second kind, K0
#[pyfunction]
fn k0_py(x: f64) -> PyResult<f64> {
    Ok(bessel::k0(x))
}

/// Modified Bessel function of the second kind, K1
#[pyfunction]
fn k1_py(x: f64) -> PyResult<f64> {
    Ok(bessel::k1(x))
}

// =============================================================================
// Error Functions
// =============================================================================

/// Error function
#[pyfunction]
fn erf_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erf(x))
}

/// Complementary error function
#[pyfunction]
fn erfc_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfc(x))
}

/// Inverse error function
#[pyfunction]
fn erfinv_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfinv(x))
}

/// Inverse complementary error function
#[pyfunction]
fn erfcinv_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfcinv(x))
}

/// Scaled complementary error function
#[pyfunction]
fn erfcx_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfcx(x))
}

/// Imaginary error function
#[pyfunction]
fn erfi_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::erfi(x))
}

/// Dawson's integral
#[pyfunction]
fn dawsn_py(x: f64) -> PyResult<f64> {
    Ok(erf_mod::dawsn(x))
}

// =============================================================================
// Combinatorial Functions
// =============================================================================

/// Factorial function
#[pyfunction]
fn factorial_py(n: u32) -> PyResult<f64> {
    scirs2_special::factorial(n)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Binomial coefficient (n choose k)
#[pyfunction]
fn comb_py(n: u32, k: u32) -> PyResult<f64> {
    scirs2_special::comb(n, k)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

/// Permutations (n permute k)
#[pyfunction]
fn perm_py(n: u32, k: u32) -> PyResult<f64> {
    scirs2_special::perm(n, k)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
}

// =============================================================================
// Elliptic Integrals
// =============================================================================

/// Complete elliptic integral of the first kind
#[pyfunction]
fn ellipk_py(m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_k(m))
}

/// Complete elliptic integral of the second kind
#[pyfunction]
fn ellipe_py(m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_e(m))
}

/// Incomplete elliptic integral of the first kind
#[pyfunction]
fn ellipkinc_py(phi: f64, m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_f(phi, m))
}

/// Incomplete elliptic integral of the second kind
#[pyfunction]
fn ellipeinc_py(phi: f64, m: f64) -> PyResult<f64> {
    Ok(scirs2_special::elliptic_e_inc(phi, m))
}

// =============================================================================
// Vectorized versions (for arrays)
// =============================================================================

/// Vectorized gamma function
#[pyfunction]
fn gamma_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| gamma_fn(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized error function
#[pyfunction]
fn erf_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| erf_mod::erf(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Vectorized J0 Bessel function
#[pyfunction]
fn j0_array_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
    let input = x.as_array();
    let output: Vec<f64> = input.iter().map(|&v| bessel::j0(v)).collect();
    scirs_to_numpy_array1(Array1::from_vec(output), py)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Gamma functions
    m.add_function(wrap_pyfunction!(gamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(lgamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(digamma_py, m)?)?;
    m.add_function(wrap_pyfunction!(beta_py, m)?)?;

    // Bessel functions
    m.add_function(wrap_pyfunction!(j0_py, m)?)?;
    m.add_function(wrap_pyfunction!(j1_py, m)?)?;
    m.add_function(wrap_pyfunction!(jn_py, m)?)?;
    m.add_function(wrap_pyfunction!(y0_py, m)?)?;
    m.add_function(wrap_pyfunction!(y1_py, m)?)?;
    m.add_function(wrap_pyfunction!(yn_py, m)?)?;
    m.add_function(wrap_pyfunction!(i0_py, m)?)?;
    m.add_function(wrap_pyfunction!(i1_py, m)?)?;
    m.add_function(wrap_pyfunction!(k0_py, m)?)?;
    m.add_function(wrap_pyfunction!(k1_py, m)?)?;

    // Error functions
    m.add_function(wrap_pyfunction!(erf_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfc_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfinv_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfcinv_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfcx_py, m)?)?;
    m.add_function(wrap_pyfunction!(erfi_py, m)?)?;
    m.add_function(wrap_pyfunction!(dawsn_py, m)?)?;

    // Combinatorial functions
    m.add_function(wrap_pyfunction!(factorial_py, m)?)?;
    m.add_function(wrap_pyfunction!(comb_py, m)?)?;
    m.add_function(wrap_pyfunction!(perm_py, m)?)?;

    // Elliptic integrals
    m.add_function(wrap_pyfunction!(ellipk_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipe_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipkinc_py, m)?)?;
    m.add_function(wrap_pyfunction!(ellipeinc_py, m)?)?;

    // Vectorized versions
    m.add_function(wrap_pyfunction!(gamma_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(erf_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(j0_array_py, m)?)?;

    Ok(())
}
