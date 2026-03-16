//! Python bindings for scirs2-optimize
//!
//! Provides optimization algorithms similar to scipy.optimize

// Callbacks acquire the GIL via Python::attach (pyo3 0.28+) since they are invoked
// from Rust optimization routines where the caller's Python<'_> token is not in scope.

use pyo3::prelude::*;
use pyo3::types::PyDict;

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
use scirs2_numpy::IntoPyArray;

// ndarray types from scirs2-core
use scirs2_core::{ndarray::ArrayView1, Array1};

// Direct imports from scirs2-optimize (native ndarray 0.17 support)
use scirs2_optimize::global::{differential_evolution, DifferentialEvolutionOptions};
use scirs2_optimize::scalar::{minimize_scalar, Method as ScalarMethod, Options as ScalarOptions};
use scirs2_optimize::unconstrained::{minimize, Bounds, Method, Options};

/// Minimize a scalar function of one variable
///
/// Parameters:
/// - fun: The objective function to minimize
/// - bracket: (a, b) interval to search
/// - method: 'brent', 'golden', or 'bounded'
/// - options: Dict with 'maxiter', 'tol'
#[pyfunction]
#[pyo3(signature = (fun, bracket, method="brent", options=None))]
fn minimize_scalar_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    bracket: (f64, f64),
    method: &str,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let maxiter = options
        .and_then(|o| o.get_item("maxiter").ok().flatten())
        .and_then(|v| v.extract().ok());
    let tol = options
        .and_then(|o| o.get_item("tol").ok().flatten())
        .and_then(|v| v.extract().ok());

    let fun_clone = fun.clone().unbind();
    let f = move |x: f64| -> f64 {
        Python::attach(|py| {
            let result = fun_clone
                .bind(py)
                .call1((x,))
                .expect("Failed to call objective function");
            result.extract().expect("Failed to extract result")
        })
    };

    // Parse method
    let scalar_method = match method {
        "brent" => ScalarMethod::Brent,
        "golden" => ScalarMethod::Golden,
        "bounded" => ScalarMethod::Bounded,
        _ => ScalarMethod::Brent,
    };

    // Set up options
    let mut options = ScalarOptions::default();
    if let Some(mi) = maxiter {
        options.max_iter = mi;
    }
    if let Some(t) = tol {
        options.xatol = t;
    }

    let result = minimize_scalar(f, Some(bracket), scalar_method, Some(options))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("x", result.x)?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;

    Ok(dict.into())
}

/// Find root of a scalar function using Brent's method
///
/// Parameters:
/// - fun: The function for which to find the root
/// - a: Lower bound of the bracket
/// - b: Upper bound of the bracket
/// - xtol: Absolute tolerance (default 1e-12)
/// - maxiter: Maximum iterations (default 100)
///
/// Returns:
/// - Dict with 'x' (root location), 'fun' (function value at root),
///   'iterations', 'success'
#[pyfunction]
#[pyo3(signature = (fun, a, b, xtol=1e-12, maxiter=100))]
fn brentq_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    a: f64,
    b: f64,
    xtol: f64,
    maxiter: usize,
) -> PyResult<Py<PyAny>> {
    let fun_clone = fun.clone().unbind();
    let f = |x: f64| -> f64 {
        Python::attach(|py| {
            let result = fun_clone
                .bind(py)
                .call1((x,))
                .expect("Failed to call objective function");
            result.extract().expect("Failed to extract result")
        })
    };

    // Brent's method implementation
    let mut a = a;
    let mut b = b;
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "f(a) and f(b) must have opposite signs",
        ));
    }

    // Ensure |f(a)| >= |f(b)|
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let mut iter = 0;

    while iter < maxiter {
        if fb.abs() < xtol {
            let dict = PyDict::new(py);
            dict.set_item("x", b)?;
            dict.set_item("fun", fb)?;
            dict.set_item("iterations", iter)?;
            dict.set_item("success", true)?;
            return Ok(dict.into());
        }

        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
            c = a;
            fc = fa;
        }

        let tol = 2.0 * f64::EPSILON * b.abs() + xtol;
        let m = (c - b) / 2.0;

        if m.abs() <= tol {
            let dict = PyDict::new(py);
            dict.set_item("x", b)?;
            dict.set_item("fun", fb)?;
            dict.set_item("iterations", iter)?;
            dict.set_item("success", true)?;
            return Ok(dict.into());
        }

        // Use bisection or interpolation
        let mut use_bisection = true;

        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if (a - c).abs() < 1e-14 {
                // Linear interpolation
                (2.0 * m * s, 1.0 - s)
            } else {
                // Inverse quadratic interpolation
                let q = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q * (q - r) - (b - a) * (r - 1.0)),
                    (q - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };

            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };

            if 2.0 * p < 3.0 * m * q - (tol * q).abs() && p < (e * q / 2.0).abs() {
                e = d;
                d = p / q;
                use_bisection = false;
            }
        }

        if use_bisection {
            d = m;
            e = m;
        }

        a = b;
        fa = fb;

        if d.abs() > tol {
            b += d;
        } else {
            b += if m > 0.0 { tol } else { -tol };
        }

        fb = f(b);

        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }

        iter += 1;
    }

    let dict = PyDict::new(py);
    dict.set_item("x", b)?;
    dict.set_item("fun", fb)?;
    dict.set_item("iterations", iter)?;
    dict.set_item("success", false)?;
    dict.set_item("message", "Maximum iterations reached")?;
    Ok(dict.into())
}

/// Minimize a function of one or more variables
///
/// Parameters:
/// - fun: The objective function to minimize
/// - x0: Initial guess as array
/// - method: Optimization method ('nelder-mead', 'bfgs', 'cg', 'powell', 'lbfgs', etc.)
/// - options: Dict with 'maxiter', 'ftol', 'gtol'
/// - bounds: Optional list of (min, max) bounds for each variable
///
/// Returns:
/// - Dict with 'x' (solution), 'fun' (function value), 'success', 'nit', 'nfev', 'message'
#[pyfunction]
#[pyo3(signature = (fun, x0, method="bfgs", options=None, bounds=None))]
fn minimize_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    x0: Vec<f64>,
    method: &str,
    options: Option<&Bound<'_, PyDict>>,
    bounds: Option<Vec<(f64, f64)>>,
) -> PyResult<Py<PyAny>> {
    // Parse method
    let opt_method = match method.to_lowercase().as_str() {
        "nelder-mead" | "neldermead" => Method::NelderMead,
        "powell" => Method::Powell,
        "cg" | "conjugate-gradient" => Method::CG,
        "bfgs" => Method::BFGS,
        "lbfgs" | "l-bfgs" => Method::LBFGS,
        "lbfgsb" | "l-bfgs-b" => Method::LBFGSB,
        "newton-cg" => Method::NewtonCG,
        "trust-ncg" => Method::TrustNCG,
        "sr1" => Method::SR1,
        "dfp" => Method::DFP,
        _ => Method::BFGS, // Default to BFGS
    };

    // Parse options
    let maxiter = options
        .and_then(|o| o.get_item("maxiter").ok().flatten())
        .and_then(|v| v.extract().ok());
    let ftol = options
        .and_then(|o| o.get_item("ftol").ok().flatten())
        .and_then(|v| v.extract().ok());
    let gtol = options
        .and_then(|o| o.get_item("gtol").ok().flatten())
        .and_then(|v| v.extract().ok());

    let mut opt_options = Options::default();
    if let Some(mi) = maxiter {
        opt_options.max_iter = mi;
    }
    if let Some(ft) = ftol {
        opt_options.ftol = ft;
    }
    if let Some(gt) = gtol {
        opt_options.gtol = gt;
    }

    // Parse bounds
    if let Some(b) = bounds {
        let n = x0.len();
        let mut lower = vec![None; n];
        let mut upper = vec![None; n];
        for (i, (l, u)) in b.iter().enumerate() {
            if i < n {
                lower[i] = Some(*l);
                upper[i] = Some(*u);
            }
        }
        opt_options.bounds = Some(Bounds { lower, upper });
    }

    // Create closure for the objective function
    let fun_arc = std::sync::Arc::new(fun.clone().unbind());
    let f = move |x: &ArrayView1<f64>| -> f64 {
        let fun_clone = fun_arc.clone();
        Python::attach(|py| {
            let x_vec: Vec<f64> = x.to_vec();
            let result = fun_clone
                .bind(py)
                .call1((x_vec,))
                .expect("Failed to call objective function");
            result.extract().expect("Failed to extract result")
        })
    };

    // Run optimization
    let result = minimize(f, &x0, opt_method, Some(opt_options))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    // Return result as dict
    let dict = PyDict::new(py);
    dict.set_item("x", result.x.into_pyarray(py).unbind())?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("message", result.message)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.func_evals)?;

    Ok(dict.into())
}

/// Global optimization using differential evolution
///
/// Parameters:
/// - fun: The objective function to minimize
/// - bounds: List of (min, max) bounds for each variable
/// - options: Dict with 'maxiter', 'popsize', 'tol', 'seed'
#[pyfunction]
#[pyo3(signature = (fun, bounds, options=None))]
fn differential_evolution_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let maxiter = options
        .and_then(|o| o.get_item("maxiter").ok().flatten())
        .and_then(|v| v.extract().ok());
    let popsize = options
        .and_then(|o| o.get_item("popsize").ok().flatten())
        .and_then(|v| v.extract().ok());
    let tol = options
        .and_then(|o| o.get_item("tol").ok().flatten())
        .and_then(|v| v.extract().ok());
    let seed = options
        .and_then(|o| o.get_item("seed").ok().flatten())
        .and_then(|v| v.extract().ok());

    let fun_arc = std::sync::Arc::new(fun.clone().unbind());
    let f = move |x: &ArrayView1<f64>| -> f64 {
        let fun_clone = fun_arc.clone();
        Python::attach(|py| {
            let x_vec: Vec<f64> = x.to_vec();
            let result = fun_clone
                .bind(py)
                .call1((x_vec,))
                .expect("Failed to call objective function");
            result.extract().expect("Failed to extract result")
        })
    };

    // Set up options
    let mut de_options = DifferentialEvolutionOptions::default();
    if let Some(mi) = maxiter {
        de_options.maxiter = mi;
    }
    if let Some(ps) = popsize {
        de_options.popsize = ps;
    }
    if let Some(t) = tol {
        de_options.tol = t;
    }
    if let Some(s) = seed {
        de_options.seed = Some(s);
    }

    // Use bounds vector directly (differential_evolution expects Vec<(f64, f64)>)
    let result = differential_evolution(f, bounds.to_vec(), Some(de_options), None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("x", result.x.into_pyarray(py).unbind())?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("message", result.message)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.func_evals)?;

    Ok(dict.into())
}

/// Curve fitting using non-linear least squares
///
/// Use non-linear least squares to fit a function to data.
/// This is similar to scipy.optimize.curve_fit.
///
/// Parameters:
/// - f: Model function f(x, *params) that takes independent variable(s) and parameters
/// - xdata: Independent variable where data is measured (array or scalar for each point)
/// - ydata: Dependent data to fit
/// - p0: Initial guess for parameters (optional, defaults to ones)
/// - method: Optimization method ('lm', 'trf', or 'dogbox')
/// - maxfev: Maximum number of function evaluations (default: 1000)
///
/// Returns:
/// - Dict with 'popt' (optimized parameters), 'success', 'nfev', 'message'
///
/// Example:
/// ```python
/// import numpy as np
/// import scirs2
///
/// # Define exponential model: f(x, a, b) = a * exp(b * x)
/// def model(x, a, b):
///     return a * np.exp(b * x)
///
/// # Generate noisy data
/// xdata = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
/// ydata = np.array([1.0, 2.7, 7.4, 20.1, 54.6])  # ≈ 1.0 * exp(1.0 * x) with noise
///
/// # Fit the curve
/// result = scirs2.curve_fit_py(model, xdata, ydata, p0=[1.0, 1.0])
/// print(f"Optimized parameters: {result['popt']}")
/// ```
#[pyfunction]
#[pyo3(signature = (f, xdata, ydata, p0=None, method="lm", maxfev=1000))]
fn curve_fit_py(
    py: Python,
    f: &Bound<'_, PyAny>,
    xdata: Vec<f64>,
    ydata: Vec<f64>,
    p0: Option<Vec<f64>>,
    method: &str,
    maxfev: usize,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::least_squares::{least_squares, Method as LSMethod, Options as LSOptions};

    if xdata.len() != ydata.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xdata and ydata must have the same length",
        ));
    }

    let n_data = xdata.len();

    // Use p0 or default to ones
    let params_init = p0.unwrap_or_else(|| vec![1.0; 2]);

    // Parse method
    let ls_method = match method.to_lowercase().as_str() {
        "lm" => LSMethod::LevenbergMarquardt,
        "trf" => LSMethod::TrustRegionReflective,
        "dogbox" => LSMethod::Dogbox,
        _ => LSMethod::LevenbergMarquardt,
    };

    // Create copies for closure
    let xdata_clone = xdata.clone();
    let ydata_clone = ydata.clone();
    let f_arc = std::sync::Arc::new(f.clone().unbind());

    // Define residual function
    let residual_fn = move |params: &[f64], _data: &[f64]| -> Array1<f64> {
        let f_clone = f_arc.clone();
        let xdata_ref = &xdata_clone;
        let ydata_ref = &ydata_clone;

        Python::attach(|py| {
            let mut residuals = Vec::with_capacity(n_data);

            for i in 0..n_data {
                // Call f(x, *params)
                let mut args = vec![xdata_ref[i]];
                args.extend_from_slice(params);

                let f_val: f64 = f_clone
                    .bind(py)
                    .call1(pyo3::types::PyTuple::new(py, &args).expect("Operation failed"))
                    .expect("Failed to call model function")
                    .extract()
                    .expect("Failed to extract model result");

                residuals.push(ydata_ref[i] - f_val);
            }

            Array1::from_vec(residuals)
        })
    };

    // Set up options
    let options = LSOptions {
        max_nfev: Some(maxfev),
        ..Default::default()
    };

    // Run least squares optimization
    let empty_data = Array1::from_vec(vec![]);

    let result = least_squares(
        residual_fn,
        &Array1::from_vec(params_init),
        ls_method,
        None::<fn(&[f64], &[f64]) -> scirs2_core::ndarray::Array2<f64>>, // No jacobian provided
        &empty_data,                                                     // No additional data
        Some(options),
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Curve fitting failed: {}", e))
    })?;

    // Return results
    let dict = PyDict::new(py);
    dict.set_item("popt", result.x.into_pyarray(py).unbind())?;
    dict.set_item("success", result.success)?;
    dict.set_item("nfev", result.nfev)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_scalar_py, m)?)?;
    m.add_function(wrap_pyfunction!(brentq_py, m)?)?;
    m.add_function(wrap_pyfunction!(differential_evolution_py, m)?)?;
    m.add_function(wrap_pyfunction!(curve_fit_py, m)?)?;

    Ok(())
}
