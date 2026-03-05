//! Extended Python bindings for scirs2-optimize
//!
//! Provides bindings for v0.3.0 features:
//! - Nelder-Mead simplex method
//! - L-BFGS-B quasi-Newton method
//! - Simulated Annealing
//! - Particle Swarm Optimization
//! - Dual Annealing

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_numpy::IntoPyArray;

// =============================================================================
// Nelder-Mead Simplex Method
// =============================================================================

/// Minimize a scalar function using the Nelder-Mead simplex algorithm.
///
/// The Nelder-Mead method is a derivative-free optimization algorithm
/// suitable for nonsmooth or noisy objective functions.
///
/// Parameters:
///     fun: Objective function fun(x: list[float]) -> float
///     x0: Initial guess (list of floats)
///     max_iter: Maximum iterations (default: 1000)
///     tol: Convergence tolerance (default: 1e-8)
///     bounds: List of (lower, upper) bounds per parameter (default: None)
///
/// Returns:
///     Dict with 'x', 'fun', 'success', 'nit', 'nfev', 'message'
#[pyfunction]
#[pyo3(signature = (fun, x0, max_iter=1000, tol=1e-8, bounds=None))]
pub fn minimize_nelder_mead_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    x0: Vec<f64>,
    max_iter: usize,
    tol: f64,
    bounds: Option<Vec<(f64, f64)>>,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::unconstrained::{minimize_nelder_mead, Bounds, Options};

    if x0.is_empty() {
        return Err(PyValueError::new_err("x0 must not be empty"));
    }

    let fn_obj = fun.clone().unbind();
    let obj_fn = move |x: &ArrayView1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        #[allow(deprecated)]
        Python::with_gil(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
    };

    let x0_arr = Array1::from_vec(x0);

    let opt_bounds = bounds.map(|b| {
        let bound_pairs: Vec<(Option<f64>, Option<f64>)> = b
            .iter()
            .map(|&(lo, hi)| (Some(lo), Some(hi)))
            .collect();
        Bounds::new(&bound_pairs)
    });

    let options = Options {
        max_iter,
        ftol: tol,
        xtol: tol,
        gtol: tol,
        bounds: opt_bounds,
        ..Default::default()
    };

    let result = minimize_nelder_mead(obj_fn, x0_arr, &options)
        .map_err(|e| PyRuntimeError::new_err(format!("Nelder-Mead failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x", Array1::from_vec(result.x.to_vec()).into_pyarray(py))?;
    dict.set_item("fun", result.fun.into())?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

// =============================================================================
// L-BFGS Quasi-Newton Method
// =============================================================================

/// Minimize a scalar function using L-BFGS (limited-memory BFGS).
///
/// L-BFGS is a quasi-Newton gradient-based optimizer that is efficient
/// for large-scale problems.
///
/// Parameters:
///     fun: Objective function fun(x: list[float]) -> float
///     x0: Initial guess (list of floats)
///     max_iter: Maximum iterations (default: 1000)
///     tol: Convergence tolerance (default: 1e-8)
///     bounds: List of (lower, upper) bounds per parameter (default: None)
///
/// Returns:
///     Dict with 'x', 'fun', 'success', 'nit', 'nfev', 'message'
#[pyfunction]
#[pyo3(signature = (fun, x0, max_iter=1000, tol=1e-8, bounds=None))]
pub fn minimize_lbfgsb_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    x0: Vec<f64>,
    max_iter: usize,
    tol: f64,
    bounds: Option<Vec<(f64, f64)>>,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::unconstrained::{minimize_lbfgs, Bounds, Options};

    if x0.is_empty() {
        return Err(PyValueError::new_err("x0 must not be empty"));
    }

    let fn_obj = fun.clone().unbind();
    let obj_fn = move |x: &ArrayView1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        #[allow(deprecated)]
        Python::with_gil(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
    };

    let x0_arr = Array1::from_vec(x0);

    let opt_bounds = bounds.map(|b| {
        let bound_pairs: Vec<(Option<f64>, Option<f64>)> = b
            .iter()
            .map(|&(lo, hi)| (Some(lo), Some(hi)))
            .collect();
        Bounds::new(&bound_pairs)
    });

    let options = Options {
        max_iter,
        ftol: tol,
        xtol: tol,
        gtol: tol,
        bounds: opt_bounds,
        ..Default::default()
    };

    let result = minimize_lbfgs(obj_fn, x0_arr, &options)
        .map_err(|e| PyRuntimeError::new_err(format!("L-BFGS failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x", Array1::from_vec(result.x.to_vec()).into_pyarray(py))?;
    dict.set_item("fun", result.fun.into())?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

// =============================================================================
// Simulated Annealing
// =============================================================================

/// Minimize a function using Simulated Annealing.
///
/// A probabilistic global optimization algorithm that accepts worse solutions
/// with decreasing probability (analogous to cooling of a material).
///
/// Parameters:
///     fun: Objective function fun(x: list[float]) -> float
///     x0: Initial guess (list of floats)
///     bounds: List of (lower, upper) bounds per parameter (default: None)
///     max_iter: Maximum iterations (default: 10000)
///     initial_temp: Initial temperature (default: 100.0)
///     final_temp: Final temperature (default: 1e-3)
///     step_size: Step size for random perturbation (default: 0.1)
///     seed: Random seed (default: None)
///
/// Returns:
///     Dict with 'x', 'fun', 'success', 'nit', 'nfev'
#[pyfunction]
#[pyo3(signature = (fun, x0, bounds=None, max_iter=10000, initial_temp=100.0, final_temp=1e-3, step_size=0.1, seed=None))]
pub fn simulated_annealing_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    x0: Vec<f64>,
    bounds: Option<Vec<(f64, f64)>>,
    max_iter: usize,
    initial_temp: f64,
    final_temp: f64,
    step_size: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::global::{simulated_annealing, SimulatedAnnealingOptions};

    if x0.is_empty() {
        return Err(PyValueError::new_err("x0 must not be empty"));
    }

    let fn_obj = fun.clone().unbind();
    let obj_fn = move |x: &ArrayView1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        #[allow(deprecated)]
        Python::with_gil(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
    };

    let x0_arr = Array1::from_vec(x0);

    let options = SimulatedAnnealingOptions {
        maxiter: max_iter,
        initial_temp,
        final_temp,
        step_size,
        seed,
        ..Default::default()
    };

    let result = simulated_annealing(obj_fn, x0_arr, bounds, Some(options))
        .map_err(|e| PyRuntimeError::new_err(format!("Simulated Annealing failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x", Array1::from_vec(result.x.to_vec()).into_pyarray(py))?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

// =============================================================================
// Particle Swarm Optimization
// =============================================================================

/// Minimize a function using Particle Swarm Optimization (PSO).
///
/// PSO is a population-based global optimizer inspired by bird flocking.
/// Each particle moves through the search space guided by personal and
/// swarm-wide best solutions.
///
/// Parameters:
///     fun: Objective function fun(x: list[float]) -> float
///     bounds: List of (lower, upper) bounds per parameter (required)
///     swarm_size: Number of particles (default: 30)
///     max_iter: Maximum iterations (default: 1000)
///     tol: Convergence tolerance (default: 1e-8)
///     seed: Random seed (default: None)
///
/// Returns:
///     Dict with 'x', 'fun', 'success', 'nit', 'nfev'
#[pyfunction]
#[pyo3(signature = (fun, bounds, swarm_size=30, max_iter=1000, tol=1e-8, seed=None))]
pub fn particle_swarm_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    swarm_size: usize,
    max_iter: usize,
    tol: f64,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::global::{particle_swarm, ParticleSwarmOptions};

    if bounds.is_empty() {
        return Err(PyValueError::new_err("bounds must not be empty"));
    }
    for (i, &(lb, ub)) in bounds.iter().enumerate() {
        if lb >= ub {
            return Err(PyValueError::new_err(format!(
                "bounds[{}]: lower bound {} must be < upper bound {}",
                i, lb, ub
            )));
        }
    }

    let fn_obj = fun.clone().unbind();
    let obj_fn = move |x: &ArrayView1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        #[allow(deprecated)]
        Python::with_gil(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
    };

    let options = ParticleSwarmOptions {
        swarm_size,
        maxiter: max_iter,
        tol,
        seed,
        ..Default::default()
    };

    let result = particle_swarm(obj_fn, bounds, Some(options))
        .map_err(|e| PyRuntimeError::new_err(format!("Particle Swarm failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x", Array1::from_vec(result.x.to_vec()).into_pyarray(py))?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

// =============================================================================
// Dual Annealing
// =============================================================================

/// Minimize a function using Dual Annealing.
///
/// Dual Annealing combines classical Simulated Annealing with fast simulated
/// annealing to achieve better exploration and faster convergence.
///
/// Parameters:
///     fun: Objective function fun(x: list[float]) -> float
///     bounds: List of (lower, upper) bounds per parameter (required)
///     x0: Initial guess (default: None - midpoint of bounds)
///     max_iter: Maximum iterations (default: 1000)
///     seed: Random seed (default: None)
///
/// Returns:
///     Dict with 'x', 'fun', 'success', 'nit', 'nfev'
#[pyfunction]
#[pyo3(signature = (fun, bounds, x0=None, max_iter=1000, seed=None))]
pub fn dual_annealing_py(
    py: Python,
    fun: &Bound<'_, PyAny>,
    bounds: Vec<(f64, f64)>,
    x0: Option<Vec<f64>>,
    max_iter: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    use scirs2_optimize::global::{dual_annealing, DualAnnealingOptions};

    if bounds.is_empty() {
        return Err(PyValueError::new_err("bounds must not be empty"));
    }

    let fn_obj = fun.clone().unbind();
    let obj_fn = move |x: &ArrayView1<f64>| -> f64 {
        let x_vec: Vec<f64> = x.to_vec();
        #[allow(deprecated)]
        Python::with_gil(|py| {
            fn_obj
                .bind(py)
                .call1((x_vec,))
                .and_then(|r| r.extract::<f64>())
                .unwrap_or(f64::INFINITY)
        })
    };

    let x0_arr = x0
        .map(Array1::from_vec)
        .unwrap_or_else(|| {
            Array1::from_vec(
                bounds.iter().map(|&(lb, ub)| (lb + ub) / 2.0).collect(),
            )
        });

    let options = DualAnnealingOptions {
        maxiter: max_iter,
        seed,
        bounds: bounds.clone(),
        ..Default::default()
    };

    let result = dual_annealing(obj_fn, x0_arr, bounds, Some(options))
        .map_err(|e| PyRuntimeError::new_err(format!("Dual Annealing failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("x", Array1::from_vec(result.x.to_vec()).into_pyarray(py))?;
    dict.set_item("fun", result.fun)?;
    dict.set_item("success", result.success)?;
    dict.set_item("nit", result.nit)?;
    dict.set_item("nfev", result.function_evals)?;
    dict.set_item("message", result.message)?;

    Ok(dict.into())
}

/// Python module registration for optimize extensions
pub fn register_optimize_ext_module(m: &Bound<'_, pyo3::PyModule>) -> pyo3::PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize_nelder_mead_py, m)?)?;
    m.add_function(wrap_pyfunction!(minimize_lbfgsb_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulated_annealing_py, m)?)?;
    m.add_function(wrap_pyfunction!(particle_swarm_py, m)?)?;
    m.add_function(wrap_pyfunction!(dual_annealing_py, m)?)?;

    Ok(())
}
