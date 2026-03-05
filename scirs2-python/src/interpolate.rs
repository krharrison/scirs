//! Python bindings for scirs2-interpolate
//!
//! Provides interpolation methods similar to scipy.interpolate

use pyo3::prelude::*;
use scirs2_core::ndarray::{Array1 as Array1_17, Array2 as Array2_17};
use scirs2_core::python::numpy_compat::{
    scirs_to_numpy_array1, scirs_to_numpy_array2, Array1, Array2,
};
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use scirs2_interpolate::interp1d::{ExtrapolateMode, Interp1d, InterpolationMethod};
use scirs2_interpolate::interp2d::{Interp2d, Interp2dKind};
use scirs2_interpolate::spline::CubicSpline;

/// 1D interpolation class
#[pyclass(name = "Interp1d")]
pub struct PyInterp1d {
    interp: Interp1d<f64>,
}

#[pymethods]
impl PyInterp1d {
    /// Create a new 1D interpolator
    ///
    /// Parameters:
    /// - x: x coordinates (must be sorted)
    /// - y: y coordinates
    /// - method: 'linear', 'nearest', 'cubic', or 'pchip'
    /// - extrapolate: 'error', 'const', or 'extrapolate'
    #[new]
    #[pyo3(signature = (x, y, method="linear", extrapolate="error"))]
    fn new(
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        method: &str,
        extrapolate: &str,
    ) -> PyResult<Self> {
        // Convert from numpy arrays to scirs2-core ndarray17 (single copy)
        let x_arr = x.as_array().to_owned();
        let y_arr = y.as_array().to_owned();

        let method = match method.to_lowercase().as_str() {
            "nearest" => InterpolationMethod::Nearest,
            "linear" => InterpolationMethod::Linear,
            "cubic" => InterpolationMethod::Cubic,
            "pchip" => InterpolationMethod::Pchip,
            _ => InterpolationMethod::Linear,
        };

        let extrapolate_mode = match extrapolate.to_lowercase().as_str() {
            "nearest" => ExtrapolateMode::Nearest,
            "extrapolate" => ExtrapolateMode::Extrapolate,
            _ => ExtrapolateMode::Error,
        };

        let interp = Interp1d::new(&x_arr.view(), &y_arr.view(), method, extrapolate_mode)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        Ok(PyInterp1d { interp })
    }

    /// Evaluate the interpolator at new points
    fn __call__(&self, py: Python, x_new: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x_vec: Vec<f64> = x_new.as_array().to_vec();
        let x_arr = Array1_17::from_vec(x_vec);
        let result = self
            .interp
            .evaluate_array(&x_arr.view())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        // Convert back to numpy-compatible array
        scirs_to_numpy_array1(Array1::from_vec(result.to_vec()), py)
    }

    /// Evaluate at a single point
    fn eval_single(&self, x: f64) -> PyResult<f64> {
        self.interp
            .evaluate(x)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }
}

/// Cubic spline interpolation class
#[pyclass(name = "CubicSpline")]
pub struct PyCubicSpline {
    spline: CubicSpline<f64>,
}

#[pymethods]
impl PyCubicSpline {
    /// Create a new cubic spline interpolator
    ///
    /// Parameters:
    /// - x: x coordinates (must be sorted, strictly increasing)
    /// - y: y coordinates
    /// - bc_type: boundary condition type ('natural', 'not-a-knot', 'clamped', 'periodic')
    #[new]
    #[pyo3(signature = (x, y, bc_type="natural"))]
    fn new(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>, bc_type: &str) -> PyResult<Self> {
        let x_vec: Vec<f64> = x.as_array().to_vec();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        let x_arr = Array1_17::from_vec(x_vec);
        let y_arr = Array1_17::from_vec(y_vec);

        let spline = match bc_type.to_lowercase().as_str() {
            "natural" | "not-a-knot" | "periodic" => CubicSpline::new(&x_arr.view(), &y_arr.view())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported boundary condition: {}",
                    bc_type
                )));
            }
        };

        Ok(PyCubicSpline { spline })
    }

    /// Evaluate the spline at new points
    fn __call__(&self, py: Python, x_new: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray1<f64>>> {
        let x_vec: Vec<f64> = x_new.as_array().to_vec();
        let mut result = Vec::with_capacity(x_vec.len());

        for &x in &x_vec {
            let y = self
                .spline
                .evaluate(x)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            result.push(y);
        }

        scirs_to_numpy_array1(Array1::from_vec(result), py)
    }

    /// Evaluate at a single point
    fn eval_single(&self, x: f64) -> PyResult<f64> {
        self.spline
            .evaluate(x)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Compute derivative at new points
    ///
    /// Parameters:
    /// - x_new: points to evaluate derivative
    /// - nu: derivative order (default: 1)
    #[pyo3(signature = (x_new, nu=1))]
    fn derivative(
        &self,
        py: Python,
        x_new: PyReadonlyArray1<f64>,
        nu: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let x_vec: Vec<f64> = x_new.as_array().to_vec();
        let mut result = Vec::with_capacity(x_vec.len());

        for &x in &x_vec {
            let y = self
                .spline
                .derivative_n(x, nu)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
            result.push(y);
        }

        scirs_to_numpy_array1(Array1::from_vec(result), py)
    }

    /// Integrate the spline over an interval
    ///
    /// Parameters:
    /// - a: lower bound
    /// - b: upper bound
    fn integrate(&self, a: f64, b: f64) -> PyResult<f64> {
        self.spline
            .integrate(a, b)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }
}

/// 2D interpolation class
#[pyclass(name = "Interp2d")]
pub struct PyInterp2d {
    interp: Interp2d<f64>,
}

#[pymethods]
impl PyInterp2d {
    /// Create a new 2D interpolator
    ///
    /// Parameters:
    /// - x: x coordinates (must be sorted)
    /// - y: y coordinates (must be sorted)
    /// - z: z values with shape (len(y), len(x))
    /// - kind: interpolation method ('linear', 'cubic', 'quintic')
    #[new]
    #[pyo3(signature = (x, y, z, kind="linear"))]
    fn new(
        x: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
        z: PyReadonlyArray2<f64>,
        kind: &str,
    ) -> PyResult<Self> {
        let x_vec: Vec<f64> = x.as_array().to_vec();
        let y_vec: Vec<f64> = y.as_array().to_vec();
        let z_arr = z.as_array();

        let x_arr = Array1_17::from_vec(x_vec);
        let y_arr = Array1_17::from_vec(y_vec);

        // Convert to Array2_17
        let z_shape = z_arr.shape();
        let z_vec: Vec<f64> = z_arr.iter().copied().collect();
        let z_arr_17 = Array2_17::from_shape_vec((z_shape[0], z_shape[1]), z_vec).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid z array: {e}"))
        })?;

        let interp_kind = match kind.to_lowercase().as_str() {
            "linear" => Interp2dKind::Linear,
            "cubic" => Interp2dKind::Cubic,
            "quintic" => Interp2dKind::Quintic,
            _ => Interp2dKind::Linear,
        };

        let interp = Interp2d::new(&x_arr.view(), &y_arr.view(), &z_arr_17.view(), interp_kind)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        Ok(PyInterp2d { interp })
    }

    /// Evaluate at a single point (x, y)
    fn __call__(&self, x: f64, y: f64) -> PyResult<f64> {
        self.interp
            .evaluate(x, y)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Evaluate at multiple points
    ///
    /// Parameters:
    /// - x_new: x coordinates
    /// - y_new: y coordinates (must have same length as x_new)
    fn eval_array(
        &self,
        py: Python,
        x_new: PyReadonlyArray1<f64>,
        y_new: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let x_vec: Vec<f64> = x_new.as_array().to_vec();
        let y_vec: Vec<f64> = y_new.as_array().to_vec();
        let x_arr = Array1_17::from_vec(x_vec);
        let y_arr = Array1_17::from_vec(y_vec);

        let result = self
            .interp
            .evaluate_array(&x_arr.view(), &y_arr.view())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        scirs_to_numpy_array1(Array1::from_vec(result.to_vec()), py)
    }

    /// Evaluate on a regular grid
    ///
    /// Parameters:
    /// - x_new: x coordinates for output grid
    /// - y_new: y coordinates for output grid
    ///
    /// Returns:
    /// - 2D array with shape (len(y_new), len(x_new))
    fn eval_grid(
        &self,
        py: Python,
        x_new: PyReadonlyArray1<f64>,
        y_new: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let x_vec: Vec<f64> = x_new.as_array().to_vec();
        let y_vec: Vec<f64> = y_new.as_array().to_vec();
        let x_arr = Array1_17::from_vec(x_vec);
        let y_arr = Array1_17::from_vec(y_vec);

        let result = self
            .interp
            .evaluate_grid(&x_arr.view(), &y_arr.view())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        // Convert Array2_17 to Array2 (ndarray 0.16)
        let shape = result.dim();
        let vec: Vec<f64> = result.into_iter().collect();
        scirs_to_numpy_array2(
            Array2::from_shape_vec(shape, vec).expect("Operation failed"),
            py,
        )
    }
}

// =============================================================================
// Simple interpolation functions
// =============================================================================

/// Linear interpolation - optimized direct implementation
#[pyfunction]
fn interp_py(
    py: Python,
    x: PyReadonlyArray1<f64>,
    xp: PyReadonlyArray1<f64>,
    fp: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_arr = x.as_array();
    let xp_arr = xp.as_array();
    let fp_arr = fp.as_array();

    let n = xp_arr.len();
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xp must not be empty",
        ));
    }
    if n != fp_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xp and fp must have same length",
        ));
    }

    let xp_slice = xp_arr.as_slice().expect("Operation failed");
    let fp_slice = fp_arr.as_slice().expect("Operation failed");

    // Pre-allocate result
    let mut result = Vec::with_capacity(x_arr.len());

    for &xi in x_arr.iter() {
        let yi = if xi <= xp_slice[0] {
            fp_slice[0]
        } else if xi >= xp_slice[n - 1] {
            fp_slice[n - 1]
        } else {
            // Binary search for interval
            let idx = xp_slice.partition_point(|&v| v < xi);
            let i = if idx > 0 { idx - 1 } else { 0 };

            // Linear interpolation
            let x0 = xp_slice[i];
            let x1 = xp_slice[i + 1];
            let y0 = fp_slice[i];
            let y1 = fp_slice[i + 1];
            let t = (xi - x0) / (x1 - x0);
            y0 + t * (y1 - y0)
        };
        result.push(yi);
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

/// Piecewise linear interpolation with boundary handling
#[pyfunction]
#[pyo3(signature = (x, xp, fp, left=None, right=None))]
fn interp_with_bounds_py(
    py: Python,
    x: PyReadonlyArray1<f64>,
    xp: PyReadonlyArray1<f64>,
    fp: PyReadonlyArray1<f64>,
    left: Option<f64>,
    right: Option<f64>,
) -> PyResult<Py<PyArray1<f64>>> {
    let x_arr = x.as_array();
    let xp_arr = xp.as_array();
    let fp_arr = fp.as_array();

    let n = xp_arr.len();
    if n == 0 || fp_arr.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Invalid input arrays",
        ));
    }

    let mut result = Vec::with_capacity(x_arr.len());

    for &xi in x_arr.iter() {
        let yi = if xi < xp_arr[0] {
            left.unwrap_or(fp_arr[0])
        } else if xi > xp_arr[n - 1] {
            right.unwrap_or(fp_arr[n - 1])
        } else {
            // Binary search for interval
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if xp_arr[mid] <= xi {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // Linear interpolation
            let t = (xi - xp_arr[lo]) / (xp_arr[hi] - xp_arr[lo]);
            fp_arr[lo] * (1.0 - t) + fp_arr[hi] * t
        };
        result.push(yi);
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInterp1d>()?;
    m.add_class::<PyCubicSpline>()?;
    m.add_class::<PyInterp2d>()?;
    m.add_function(wrap_pyfunction!(interp_py, m)?)?;
    m.add_function(wrap_pyfunction!(interp_with_bounds_py, m)?)?;

    Ok(())
}
