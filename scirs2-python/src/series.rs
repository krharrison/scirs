//! Python bindings for scirs2-series using PyO3
//!
//! This module provides Python bindings for seamless integration with pandas,
//! statsmodels, and other Python time series analysis libraries.

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
use scirs2_numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, ToPyArray};

// ndarray types from scirs2-core
use scirs2_core::Array1;

// Direct imports from scirs2-series (native ndarray 0.17 support)
use scirs2_series::arima_models::ArimaModel;
use scirs2_series::decomposition::stl::stl_decomposition;
use scirs2_series::transformations::{adf_test, box_cox_transform, inverse_box_cox_transform};
use scirs2_series::utils::{difference_series, seasonal_difference_series};

use std::collections::HashMap;

/// Python wrapper for time series data
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PyTimeSeries {
    pub(crate) values: Array1<f64>,
    pub(crate) timestamps: Option<Array1<f64>>,
    pub(crate) frequency: Option<f64>,
}

impl PyTimeSeries {
    /// Create a new time series from Rust-owned arrays (crate-internal)
    pub(crate) fn from_arrays(values: Array1<f64>, timestamps: Option<Array1<f64>>) -> Self {
        PyTimeSeries {
            values,
            timestamps,
            frequency: None,
        }
    }

    /// Get values as an owned array (crate-internal)
    pub(crate) fn values_owned(&self) -> Array1<f64> {
        self.values.clone()
    }

    /// Get timestamps as an owned array (crate-internal)
    pub(crate) fn timestamps_owned(&self) -> Option<Array1<f64>> {
        self.timestamps.clone()
    }
}

#[pymethods]
impl PyTimeSeries {
    /// Create a new time series from Python list or numpy array
    #[new]
    fn new(
        values: PyReadonlyArray1<f64>,
        timestamps: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Self> {
        let values_array = values.as_array().to_owned();
        let timestamps_array = timestamps.map(|ts| ts.as_array().to_owned());

        Ok(PyTimeSeries {
            values: values_array,
            timestamps: timestamps_array,
            frequency: None,
        })
    }

    /// Set the frequency of the time series
    fn set_frequency(&mut self, frequency: f64) {
        self.frequency = Some(frequency);
    }

    /// Get the length of the time series
    fn __len__(&self) -> usize {
        self.values.len()
    }

    /// Get values as numpy array
    fn get_values<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray1<f64>>> {
        Ok(self.values.clone().into_pyarray(py).unbind())
    }

    /// Get timestamps as numpy array (if available)
    fn get_timestamps<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyArray1<f64>>>> {
        Ok(self
            .timestamps
            .as_ref()
            .map(|ts| ts.clone().into_pyarray(py).unbind()))
    }

    /// Convert to pandas-compatible dictionary
    fn to_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        dict.set_item("values", self.values.clone().into_pyarray(py).unbind())?;

        if let Some(ref timestamps) = self.timestamps {
            dict.set_item("timestamps", timestamps.clone().into_pyarray(py).unbind())?;
        }

        if let Some(freq) = self.frequency {
            dict.set_item("frequency", freq)?;
        }

        Ok(dict.into())
    }

    /// Create from pandas Series
    #[classmethod]
    fn from_pandas(_cls: &Bound<'_, PyType>, series: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Extract values from pandas Series
        let values = series.getattr("values")?;
        let values_array: PyReadonlyArray1<f64> = values.extract()?;

        // Try to extract index (timestamps) if available
        let index = series.getattr("index")?;
        let timestamps = if index.hasattr("values")? {
            index
                .getattr("values")?
                .extract::<PyReadonlyArray1<f64>>()
                .ok()
        } else {
            None
        };

        Self::new(values_array, timestamps)
    }

    /// Statistical summary
    fn describe(&self) -> PyResult<HashMap<String, f64>> {
        let mut stats = HashMap::new();
        let values = &self.values;

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        stats.insert("count".to_string(), n);
        stats.insert("mean".to_string(), mean);
        stats.insert("std".to_string(), std);
        stats.insert("min".to_string(), min);
        stats.insert("max".to_string(), max);

        // Calculate quantiles
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
        let len = sorted_values.len();

        stats.insert("25%".to_string(), sorted_values[len / 4]);
        stats.insert("50%".to_string(), sorted_values[len / 2]);
        stats.insert("75%".to_string(), sorted_values[3 * len / 4]);

        Ok(stats)
    }
}

/// Python wrapper for ARIMA models
#[pyclass]
pub struct PyARIMA {
    p: usize,
    d: usize,
    q: usize,
    model: Option<ArimaModel<f64>>,
    data: Option<Array1<f64>>,
}

#[pymethods]
impl PyARIMA {
    /// Create a new ARIMA model
    #[new]
    fn new(p: usize, d: usize, q: usize) -> Self {
        PyARIMA {
            p,
            d,
            q,
            model: None,
            data: None,
        }
    }

    /// Fit the ARIMA model
    fn fit(&mut self, data: &PyTimeSeries) -> PyResult<()> {
        let mut model = ArimaModel::new(self.p, self.d, self.q)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        model
            .fit(&data.values)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        self.model = Some(model);
        self.data = Some(data.values.clone());
        Ok(())
    }

    /// Generate forecasts
    fn forecast(&self, py: Python, steps: usize) -> PyResult<Py<PyArray1<f64>>> {
        match (&self.model, &self.data) {
            (Some(model), Some(data)) => {
                let forecasts = model
                    .forecast(steps, data)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                Ok(forecasts.into_pyarray(py).unbind())
            }
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get model parameters
    fn get_params(&self) -> PyResult<HashMap<String, f64>> {
        let mut params = HashMap::new();
        params.insert("p".to_string(), self.p as f64);
        params.insert("d".to_string(), self.d as f64);
        params.insert("q".to_string(), self.q as f64);

        if let Some(ref model) = self.model {
            params.insert("aic".to_string(), model.aic());
            params.insert("bic".to_string(), model.bic());
        }

        Ok(params)
    }

    /// Get AR coefficients
    fn get_ar_coefficients(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        match &self.model {
            Some(model) => Ok(model.ar_coeffs.to_pyarray(py).unbind()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get MA coefficients
    fn get_ma_coefficients(&self, py: Python) -> PyResult<Py<PyArray1<f64>>> {
        match &self.model {
            Some(model) => Ok(model.ma_coeffs.to_pyarray(py).unbind()),
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }

    /// Get model summary (similar to statsmodels)
    fn summary(&self) -> PyResult<String> {
        match &self.model {
            Some(model) => {
                let mut summary =
                    format!("ARIMA({},{},{}) Model Results\n", self.p, self.d, self.q);
                summary.push_str("=====================================\n");
                summary.push_str(&format!("AIC:                  {:10.4}\n", model.aic()));
                summary.push_str(&format!("BIC:                  {:10.4}\n", model.bic()));

                let ar_coeffs = &model.ar_coeffs;
                if !ar_coeffs.is_empty() {
                    summary.push_str("\nAR Coefficients:\n");
                    for (i, coef) in ar_coeffs.iter().enumerate() {
                        summary.push_str(&format!("  ar.L{}: {:10.4}\n", i + 1, coef));
                    }
                }

                let ma_coeffs = &model.ma_coeffs;
                if !ma_coeffs.is_empty() {
                    summary.push_str("\nMA Coefficients:\n");
                    for (i, coef) in ma_coeffs.iter().enumerate() {
                        summary.push_str(&format!("  ma.L{}: {:10.4}\n", i + 1, coef));
                    }
                }

                Ok(summary)
            }
            None => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not fitted. Call fit() first.",
            )),
        }
    }
}

/// Apply differencing to a time series
#[pyfunction]
fn apply_differencing(
    py: Python,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = difference_series(&data.values, periods)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Apply seasonal differencing to a time series
#[pyfunction]
fn apply_seasonal_differencing(
    py: Python,
    data: &PyTimeSeries,
    periods: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    let result = seasonal_difference_series(&data.values, periods)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Perform STL decomposition
#[pyfunction]
fn stl_decomposition_py(py: Python, data: &PyTimeSeries, period: usize) -> PyResult<Py<PyAny>> {
    use scirs2_series::decomposition::stl::STLOptions;

    let options = STLOptions::default();
    let result = stl_decomposition(&data.values, period, &options)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("trend", result.trend.into_pyarray(py).unbind())?;
    dict.set_item("seasonal", result.seasonal.into_pyarray(py).unbind())?;
    dict.set_item("residual", result.residual.into_pyarray(py).unbind())?;

    Ok(dict.into())
}

/// Perform Augmented Dickey-Fuller test for stationarity
#[pyfunction]
#[pyo3(signature = (data, max_lags=None, regression="c"))]
fn adf_test_py(
    data: &PyTimeSeries,
    max_lags: Option<usize>,
    regression: &str,
) -> PyResult<HashMap<String, f64>> {
    let result = adf_test(&data.values, max_lags, regression)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let mut output = HashMap::new();
    output.insert("statistic".to_string(), result.statistic);
    output.insert("p_value".to_string(), result.p_value);
    output.insert(
        "is_stationary".to_string(),
        if result.is_stationary { 1.0 } else { 0.0 },
    );

    Ok(output)
}

/// Apply Box-Cox transformation
#[pyfunction]
fn boxcox_transform_py(
    py: Python,
    data: &PyTimeSeries,
    lambda: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let (transformed, transform_info) = box_cox_transform(&data.values, lambda)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    let dict = PyDict::new(py);
    dict.set_item("transformed", transformed.into_pyarray(py).unbind())?;
    dict.set_item("lambda", transform_info.lambda)?;

    Ok(dict.into())
}

/// Apply inverse Box-Cox transformation
#[pyfunction]
fn boxcox_inverse_py(
    py: Python,
    data: PyReadonlyArray1<f64>,
    lambda: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let data_array = data.as_array();
    // Create BoxCoxTransform struct with the lambda parameter
    use scirs2_series::transformations::BoxCoxTransform;
    let transform = BoxCoxTransform {
        lambda,
        lambda_estimated: false,
        min_adjustment: 0.0,
    };
    let result = inverse_box_cox_transform(&data_array, &transform)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    Ok(result.into_pyarray(py).unbind())
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeSeries>()?;
    m.add_class::<PyARIMA>()?;

    m.add_function(wrap_pyfunction!(apply_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(apply_seasonal_differencing, m)?)?;
    m.add_function(wrap_pyfunction!(stl_decomposition_py, m)?)?;
    m.add_function(wrap_pyfunction!(adf_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(boxcox_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(boxcox_inverse_py, m)?)?;

    Ok(())
}

// Helper functions for pandas integration

/// Creates a pandas DataFrame from a HashMap of Array1<f64> data
#[allow(dead_code)]
pub fn create_pandas_dataframe(
    py: Python,
    data: HashMap<String, Array1<f64>>,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let dict = PyDict::new(py);

    for (key, values) in data {
        dict.set_item(key, values.into_pyarray(py).unbind())?;
    }

    let df = pandas.call_method1("DataFrame", (dict,))?;
    Ok(df.into())
}

/// Creates a pandas Series from a Rust Array1<f64>
#[allow(dead_code)]
pub fn create_pandas_series(
    py: Python,
    data: Array1<f64>,
    name: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;
    let args = (data.into_pyarray(py).unbind(),);
    let kwargs = PyDict::new(py);

    if let Some(name) = name {
        kwargs.set_item("name", name)?;
    }

    let series = pandas.call_method("Series", args, Some(&kwargs))?;
    Ok(series.into())
}
