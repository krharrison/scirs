//! Pandas DataFrame and Series integration
//!
//! This module provides utilities for converting between pandas DataFrames/Series
//! and scirs2 data structures with zero-copy where possible.
//!
//! # Example (Python)
//! ```python
//! import pandas as pd
//! import scirs2
//! import numpy as np
//!
//! # Create pandas Series
//! s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
//!
//! # Convert to TimeSeries
//! ts = scirs2.pandas_to_timeseries(s)
//!
//! # Perform operations
//! arima = scirs2.PyARIMA(1, 1, 0)
//! arima.fit(ts)
//! forecast = arima.forecast(5)
//!
//! # Convert back to pandas
//! forecast_series = pd.Series(forecast)
//! ```

use crate::error::SciRS2Error;
use crate::series::PyTimeSeries;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

/// Convert pandas Series to PyTimeSeries
///
/// This function extracts the values and index from a pandas Series
/// and creates a PyTimeSeries object.
///
/// # Arguments
/// * `series` - A pandas Series object
///
/// # Returns
/// A PyTimeSeries object with values and timestamps
#[pyfunction]
pub fn pandas_to_timeseries(py: Python, series: Py<PyAny>) -> PyResult<PyTimeSeries> {
    // Get values as numpy array
    let values_obj = series.getattr(py, "values")?;
    let values_array = values_obj.cast_bound::<PyArray1<f64>>(py)?;

    // Get index as numpy array (if datetime-like, convert to float timestamps)
    let index = series.getattr(py, "index")?;

    // Try to convert index to numpy array
    let timestamps = if let Ok(index_values) = index.getattr(py, "values") {
        // Check if it's a DatetimeIndex by trying to_numpy method
        if index.getattr(py, "to_numpy").is_ok() {
            // Convert datetime to timestamp (seconds since epoch)
            let timestamp_method = index.getattr(py, "astype")?;
            let timestamps_ns = timestamp_method.call1(py, ("int64",))?;
            let ts_values = timestamps_ns.getattr(py, "values")?;
            let timestamps_array = ts_values.cast_bound::<PyArray1<i64>>(py)?;

            // Convert from nanoseconds to seconds
            let binding = timestamps_array.readonly();
            let ts_arr = binding.as_array();
            let ts_vec: Vec<f64> = ts_arr.iter().map(|&ns| ns as f64 / 1e9).collect();

            Some(scirs2_core::Array1::from_vec(ts_vec))
        } else if let Ok(index_array) = index_values.cast_bound::<PyArray1<f64>>(py) {
            // Already numeric
            let binding = index_array.readonly();
            let idx_arr = binding.as_array();
            Some(idx_arr.to_owned())
        } else {
            None
        }
    } else {
        None
    };

    // Create PyTimeSeries using crate-internal constructor
    let binding = values_array.readonly();
    let values_arr = binding.as_array();
    let values_owned = values_arr.to_owned();

    Ok(PyTimeSeries::from_arrays(values_owned, timestamps))
}

/// Convert PyTimeSeries to pandas Series
///
/// # Arguments
/// * `ts` - A PyTimeSeries object
///
/// # Returns
/// A pandas Series object
#[pyfunction]
pub fn timeseries_to_pandas(py: Python, ts: &PyTimeSeries) -> PyResult<Py<PyAny>> {
    // Import pandas
    let pandas = py.import("pandas")?;

    // Get values using crate-internal accessor
    let values = ts.values_owned().into_pyarray(py).unbind();

    // Create Series
    let series = if let Some(timestamps) = ts.timestamps_owned() {
        // Create DatetimeIndex from timestamps
        let timestamps_ns: Vec<i64> = timestamps.iter().map(|&s| (s * 1e9) as i64).collect();

        let datetime_index = pandas
            .getattr("DatetimeIndex")?
            .call1((PyList::new(py, &timestamps_ns)?,))?;

        // Create Series with datetime index
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", datetime_index)?;
        pandas.getattr("Series")?.call((values,), Some(&kwargs))
    } else {
        // Create Series without index
        pandas.getattr("Series")?.call1((values,))
    }?;

    Ok(series.into())
}

/// Convert pandas DataFrame to numpy array (2D)
///
/// This is a convenience function for extracting numeric data from DataFrames
/// for use with scirs2 functions.
///
/// # Arguments
/// * `df` - A pandas DataFrame
///
/// # Returns
/// A 2D numpy array
#[pyfunction]
pub fn dataframe_to_array(py: Python, df: Py<PyAny>) -> PyResult<Py<PyArray2<f64>>> {
    // Get values as numpy array
    let values = df.getattr(py, "values")?;
    let array = values.cast_bound::<PyArray2<f64>>(py)?;

    Ok(array.to_owned().unbind())
}

/// Convert numpy array to pandas DataFrame
///
/// # Arguments
/// * `array` - A 2D numpy array
/// * `columns` - Optional column names (list of strings)
/// * `index` - Optional index values
///
/// # Returns
/// A pandas DataFrame
#[pyfunction]
#[pyo3(signature = (array, columns=None, index=None))]
pub fn array_to_dataframe(
    py: Python,
    array: &Bound<'_, PyArray2<f64>>,
    columns: Option<Vec<String>>,
    index: Option<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    // Import pandas
    let pandas = py.import("pandas")?;

    // Create DataFrame
    let kwargs = PyDict::new(py);
    if let Some(cols) = columns {
        kwargs.set_item("columns", cols)?;
    }
    if let Some(idx) = index {
        kwargs.set_item("index", idx)?;
    }

    let df = pandas.getattr("DataFrame")?.call((array,), Some(&kwargs))?;

    Ok(df.into())
}

/// Apply a scirs2 function to each column of a DataFrame
///
/// # Example (Python)
/// ```python
/// import pandas as pd
/// import scirs2
///
/// df = pd.DataFrame({
///     'A': [1, 2, 3, 4, 5],
///     'B': [2, 4, 6, 8, 10],
///     'C': [1, 3, 5, 7, 9]
/// })
///
/// # Calculate mean of each column
/// means = scirs2.apply_to_dataframe(df, scirs2.mean_py)
/// ```
#[pyfunction]
pub fn apply_to_dataframe(py: Python, df: Py<PyAny>, func: Py<PyAny>) -> PyResult<Py<PyAny>> {
    // Import pandas
    let pandas = py.import("pandas")?;

    // Get column names
    let columns = df.getattr(py, "columns")?;
    let col_list: Vec<String> = columns.extract(py)?;

    // Apply function to each column
    let results = PyDict::new(py);
    for col_name in col_list {
        let column = df.call_method1(py, "__getitem__", (&col_name,))?;
        let values = column.getattr(py, "values")?;
        let result = func.call1(py, (values,))?;
        results.set_item(&col_name, result)?;
    }

    // Convert results dict to pandas Series
    let series = pandas.getattr("Series")?.call1((results,))?;
    Ok(series.into())
}

/// Apply a scirs2 function row-wise or column-wise to a DataFrame
///
/// # Arguments
/// * `df` - A pandas DataFrame
/// * `func` - A scirs2 function that takes a 1D array
/// * `axis` - 0 for column-wise (default), 1 for row-wise
///
/// # Returns
/// A pandas Series with results
#[pyfunction]
#[pyo3(signature = (df, func, axis=0))]
pub fn apply_along_axis(
    py: Python,
    df: Py<PyAny>,
    func: Py<PyAny>,
    axis: usize,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;

    if axis == 0 {
        // Column-wise (same as apply_to_dataframe)
        apply_to_dataframe(py, df, func)
    } else {
        // Row-wise
        let values = df.getattr(py, "values")?;
        let array = values.cast_bound::<PyArray2<f64>>(py)?;
        let binding = array.readonly();
        let arr = binding.as_array();

        let results: Vec<f64> = arr
            .rows()
            .into_iter()
            .map(|row| {
                let row_array = row.to_owned().into_pyarray(py);
                func.call1(py, (row_array,))
                    .and_then(|r| r.extract::<f64>(py))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let series = pandas.getattr("Series")?.call1((results,))?;
        Ok(series.into())
    }
}

/// Rolling window operations on pandas Series with scirs2 functions
///
/// # Example (Python)
/// ```python
/// import pandas as pd
/// import scirs2
///
/// s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
///
/// # Calculate rolling mean with window size 3
/// rolling_mean = scirs2.rolling_apply(s, 3, scirs2.mean_py)
/// ```
#[pyfunction]
pub fn rolling_apply(
    py: Python,
    series: Py<PyAny>,
    window: usize,
    func: Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let pandas = py.import("pandas")?;

    // Get values
    let values = series.getattr(py, "values")?;
    let array = values.cast_bound::<PyArray1<f64>>(py)?;
    let binding = array.readonly();
    let arr = binding.as_array();

    if arr.len() < window {
        return Err(SciRS2Error::ValueError(format!(
            "Window size {} is larger than array length {}",
            window,
            arr.len()
        ))
        .into());
    }

    // Calculate rolling statistics
    let mut results = Vec::with_capacity(arr.len() - window + 1);
    for i in 0..=(arr.len() - window) {
        let window_slice = arr.slice(ndarray::s![i..i + window]);
        let window_array = window_slice.to_owned().into_pyarray(py);
        let result: f64 = func.call1(py, (window_array,))?.extract(py)?;
        results.push(result);
    }

    // Pad with NaN at the beginning
    let mut padded = vec![f64::NAN; window - 1];
    padded.extend(results);

    // Create pandas Series
    let series_result = pandas.getattr("Series")?.call1((padded,))?;
    Ok(series_result.into())
}

/// Register pandas compatibility functions with Python module
pub fn register_pandas_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pandas_to_timeseries, m)?)?;
    m.add_function(wrap_pyfunction!(timeseries_to_pandas, m)?)?;
    m.add_function(wrap_pyfunction!(dataframe_to_array, m)?)?;
    m.add_function(wrap_pyfunction!(array_to_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(apply_to_dataframe, m)?)?;
    m.add_function(wrap_pyfunction!(apply_along_axis, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_apply, m)?)?;
    Ok(())
}
