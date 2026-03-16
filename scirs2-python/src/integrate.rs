//! Python bindings for scirs2-integrate
//!
//! Provides numerical integration similar to scipy.integrate

// Callbacks acquire the GIL via Python::attach (pyo3 0.28+) since they are invoked
// from Rust integration routines where the caller's Python<'_> token is not in scope.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::{Array1 as Array1_17, ArrayView1};
use scirs2_core::python::numpy_compat::{scirs_to_numpy_array1, Array1};
use scirs2_numpy::{PyArray1, PyReadonlyArray1};

use scirs2_integrate::ode::{solve_ivp, ODEMethod, ODEOptions};
use scirs2_integrate::quad::{quad, QuadOptions};

// =============================================================================
// Array-based Integration (works without callbacks)
// =============================================================================

/// Integrate using array data (y values at x points) - trapezoidal rule
///
/// Similar to scipy.integrate.trapezoid
#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0))]
fn trapezoid_array_py(
    y: PyReadonlyArray1<f64>,
    x: Option<PyReadonlyArray1<f64>>,
    dx: f64,
) -> PyResult<f64> {
    let y_arr = y.as_array();

    if y_arr.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 points",
        ));
    }

    let result = if let Some(x_py) = x {
        let x_arr = x_py.as_array();
        if x_arr.len() != y_arr.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "x and y must have same length",
            ));
        }
        // Non-uniform spacing
        let mut total = 0.0;
        for i in 0..y_arr.len() - 1 {
            let dx = x_arr[i + 1] - x_arr[i];
            total += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx;
        }
        total
    } else {
        // Uniform spacing with provided dx
        let mut total = 0.0;
        for i in 0..y_arr.len() - 1 {
            total += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx;
        }
        total
    };

    Ok(result)
}

/// Integrate using array data - Simpson's rule
///
/// Similar to scipy.integrate.simpson
#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0))]
fn simpson_array_py(
    y: PyReadonlyArray1<f64>,
    x: Option<PyReadonlyArray1<f64>>,
    dx: f64,
) -> PyResult<f64> {
    let y_arr = y.as_array();
    let n = y_arr.len();

    if n < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 3 points",
        ));
    }

    // Use Simpson's rule for even number of intervals, fall back to trapezoid for odd
    let result = if let Some(x_py) = x {
        let x_arr = x_py.as_array();
        if x_arr.len() != y_arr.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "x and y must have same length",
            ));
        }

        let mut total = 0.0;
        let mut i = 0;
        while i + 2 < n {
            let h = (x_arr[i + 2] - x_arr[i]) / 2.0;
            total += h / 3.0 * (y_arr[i] + 4.0 * y_arr[i + 1] + y_arr[i + 2]);
            i += 2;
        }
        // Handle remaining interval with trapezoid
        if i + 1 < n {
            let h = x_arr[i + 1] - x_arr[i];
            total += 0.5 * (y_arr[i] + y_arr[i + 1]) * h;
        }
        total
    } else {
        let mut total = 0.0;
        let mut i = 0;
        while i + 2 < n {
            total += dx / 3.0 * (y_arr[i] + 4.0 * y_arr[i + 1] + y_arr[i + 2]);
            i += 2;
        }
        // Handle remaining interval with trapezoid
        if i + 1 < n {
            total += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx;
        }
        total
    };

    Ok(result)
}

/// Cumulative trapezoidal integration
///
/// Similar to scipy.integrate.cumulative_trapezoid
#[pyfunction]
#[pyo3(signature = (y, x=None, dx=1.0, initial=None))]
fn cumulative_trapezoid_py(
    py: Python,
    y: PyReadonlyArray1<f64>,
    x: Option<PyReadonlyArray1<f64>>,
    dx: f64,
    initial: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let y_arr = y.as_array();

    if y_arr.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 2 points",
        ));
    }

    let n = y_arr.len();
    let has_initial = initial.is_some();
    let mut result = Vec::with_capacity(if has_initial { n } else { n - 1 });

    if let Some(init) = initial {
        result.push(init);
    }

    let mut cumsum = initial.unwrap_or(0.0);

    if let Some(x_py) = x {
        let x_arr = x_py.as_array();
        for i in 0..y_arr.len() - 1 {
            let dx_i = x_arr[i + 1] - x_arr[i];
            cumsum += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx_i;
            result.push(cumsum);
        }
    } else {
        for i in 0..y_arr.len() - 1 {
            cumsum += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx;
            result.push(cumsum);
        }
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

/// Romberg integration using array data
#[pyfunction]
fn romberg_array_py(y: PyReadonlyArray1<f64>, dx: f64) -> PyResult<f64> {
    let y_arr = y.as_array();
    let n = y_arr.len();

    if n < 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Need at least 3 points",
        ));
    }

    // Simple implementation using available data points
    // This is essentially Simpson's rule as a good approximation
    let mut total = 0.0;
    let mut i = 0;
    while i + 2 < n {
        total += dx / 3.0 * (y_arr[i] + 4.0 * y_arr[i + 1] + y_arr[i + 2]);
        i += 2;
    }
    if i + 1 < n {
        total += 0.5 * (y_arr[i] + y_arr[i + 1]) * dx;
    }

    Ok(total)
}

// =============================================================================
// Adaptive Quadrature
// =============================================================================

/// Adaptive quadrature integration
///
/// Parameters:
/// - fun: Function to integrate
/// - a: Lower bound
/// - b: Upper bound
/// - epsabs: Absolute error tolerance (default 1.49e-8)
/// - epsrel: Relative error tolerance (default 1.49e-8)
/// - maxiter: Maximum function evaluations (default 500)
///
/// Returns:
/// - Dict with 'value' (integral), 'error' (estimated error), 'neval', 'success'
#[pyfunction]
#[pyo3(signature = (fun, a, b, epsabs=1.49e-8, epsrel=1.49e-8, maxiter=500))]
fn quad_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    maxiter: usize,
) -> PyResult<Py<PyAny>> {
    let fun_clone = fun.clone().unbind();
    let f = |x: f64| -> f64 {
        Python::attach(|py| {
            let result = fun_clone
                .bind(py)
                .call1((x,))
                .expect("Failed to call function");
            result.extract().expect("Failed to extract result")
        })
    };

    let options = QuadOptions {
        abs_tol: epsabs,
        rel_tol: epsrel,
        max_evals: maxiter,
        ..Default::default()
    };

    let result = quad(f, a, b, Some(options))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("value", result.value)?;
    dict.set_item("error", result.abs_error)?;
    dict.set_item("neval", result.n_evals)?;
    dict.set_item("success", result.converged)?;

    Ok(dict.into())
}

// =============================================================================
// ODE Solvers
// =============================================================================

/// Solve an initial value problem for a system of ODEs
///
/// Parameters:
/// - fun: Function computing dy/dt = f(t, y)
/// - t_span: Tuple (t0, tf) for integration interval
/// - y0: Initial state
/// - method: 'RK45' (default), 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA'
/// - rtol: Relative tolerance (default 1e-3)
/// - atol: Absolute tolerance (default 1e-6)
/// - max_step: Maximum step size (optional)
///
/// Returns:
/// - Dict with 't' (times), 'y' (solutions), 'nfev', 'success', 'message'
#[pyfunction]
#[pyo3(signature = (fun, t_span, y0, method="RK45", rtol=1e-3, atol=1e-6, max_step=None))]
fn solve_ivp_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    t_span: (f64, f64),
    y0: Vec<f64>,
    method: &str,
    rtol: f64,
    atol: f64,
    max_step: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let fun_arc = std::sync::Arc::new(fun.clone().unbind());
    let f = move |t: f64, y: ArrayView1<f64>| -> Array1_17<f64> {
        let fun_clone = fun_arc.clone();
        Python::attach(|py| {
            let y_vec: Vec<f64> = y.to_vec();
            let result = fun_clone
                .bind(py)
                .call1((t, y_vec))
                .expect("Failed to call ODE function");
            let result_vec: Vec<f64> = result.extract().expect("Failed to extract result");
            Array1_17::from_vec(result_vec)
        })
    };

    let ode_method = match method.to_uppercase().as_str() {
        "EULER" => ODEMethod::Euler,
        "RK4" => ODEMethod::RK4,
        "RK23" => ODEMethod::RK23,
        "RK45" => ODEMethod::RK45,
        "DOP853" => ODEMethod::DOP853,
        "BDF" => ODEMethod::Bdf,
        "RADAU" => ODEMethod::Radau,
        "LSODA" => ODEMethod::LSODA,
        _ => ODEMethod::RK45,
    };

    let options = ODEOptions {
        method: ode_method,
        rtol,
        atol,
        max_step,
        ..Default::default()
    };

    let y0_arr = Array1_17::from_vec(y0);
    let result = solve_ivp(f, [t_span.0, t_span.1], y0_arr, Some(options))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    // Convert results to Python
    let t_vec: Vec<f64> = result.t.to_vec();

    // Convert y (Vec<Array1>) to 2D array
    let n_points = result.y.len();
    let n_dim = if n_points > 0 { result.y[0].len() } else { 0 };
    let mut y_flat = Vec::with_capacity(n_points * n_dim);
    for arr in &result.y {
        for &val in arr.iter() {
            y_flat.push(val);
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("t", scirs_to_numpy_array1(Array1::from_vec(t_vec), py)?)?;

    // Create 2D array for y
    let y_arr = scirs2_core::python::numpy_compat::Array2::from_shape_vec((n_dim, n_points), {
        let mut transposed = Vec::with_capacity(n_points * n_dim);
        for j in 0..n_dim {
            for i in 0..n_points {
                transposed.push(y_flat[i * n_dim + j]);
            }
        }
        transposed
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    dict.set_item(
        "y",
        scirs2_core::python::numpy_compat::scirs_to_numpy_array2(y_arr, py)?,
    )?;

    dict.set_item("nfev", result.n_eval)?;
    dict.set_item("success", result.success)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(trapezoid_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(simpson_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(cumulative_trapezoid_py, m)?)?;
    m.add_function(wrap_pyfunction!(romberg_array_py, m)?)?;
    m.add_function(wrap_pyfunction!(quad_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ivp_py, m)?)?;

    Ok(())
}
