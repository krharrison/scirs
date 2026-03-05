//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{PyArray1, PyArrayMethods};
use scirs2_stats::tests::ttest::{ttest_rel, Alternative};
use scirs2_stats::{
    boxplot_stats, coef_variation, cross_entropy, data_range, entropy, gini_coefficient,
    kl_divergence, kurtosis_ci, mean_abs_deviation, median_abs_deviation, moment, quartiles,
    quintiles, skewness_ci, weighted_mean, winsorized_mean, winsorized_variance,
    QuantileInterpolation,
};

/// Paired (related samples) t-test.
///
/// Tests whether the means of two related/paired samples differ.
/// This is a parametric test for paired observations (e.g., before/after measurements).
///
/// Parameters:
///     a: First array of observations (e.g., before treatment)
///     b: Second array of observations (e.g., after treatment)
///         Must be same length as a (paired observations)
///     alternative: Alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary with 'statistic' (t), 'pvalue', and 'df' (degrees of freedom)
#[pyfunction]
#[pyo3(signature = (a, b, alternative = "two-sided"))]
pub fn ttest_rel_py(
    py: Python,
    a: &Bound<'_, PyArray1<f64>>,
    b: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let a_binding = a.readonly();
    let a_arr = a_binding.as_array();
    let b_binding = b.readonly();
    let b_arr = b_binding.as_array();
    let alt = match alternative.to_lowercase().as_str() {
        "two-sided" | "two_sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid alternative: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )));
        }
    };
    let result = ttest_rel(&a_arr.view(), &b_arr.view(), alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("Paired t-test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;
    Ok(dict.into())
}
/// Skewness of the data
#[pyfunction]
pub fn skew_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;
    if n < 3.0 {
        return Err(PyRuntimeError::new_err(
            "Need at least 3 data points for skewness",
        ));
    }
    let mean: f64 = arr.iter().sum::<f64>() / n;
    let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = arr.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    Ok(m3 / m2.powf(1.5))
}
/// Kurtosis of the data
#[pyfunction]
pub fn kurtosis_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;
    if n < 4.0 {
        return Err(PyRuntimeError::new_err(
            "Need at least 4 data points for kurtosis",
        ));
    }
    let mean: f64 = arr.iter().sum::<f64>() / n;
    let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = arr.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    Ok(m4 / m2.powi(2) - 3.0)
}
/// Mode of the data (most frequent value)
#[pyfunction]
pub fn mode_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    if arr.is_empty() {
        return Err(PyRuntimeError::new_err(
            "Cannot compute mode of empty array",
        ));
    }
    let mut values: Vec<f64> = arr.iter().copied().collect();
    values.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
    let mut max_count = 0;
    let mut mode = values[0];
    let mut current_count = 1;
    let mut current_val = values[0];
    for &v in values.iter().skip(1) {
        if (v - current_val).abs() < 1e-10 {
            current_count += 1;
        } else {
            if current_count > max_count {
                max_count = current_count;
                mode = current_val;
            }
            current_count = 1;
            current_val = v;
        }
    }
    if current_count > max_count {
        mode = current_val;
    }
    Ok(mode)
}
/// Geometric mean
#[pyfunction]
pub fn gmean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err(
            "Cannot compute geometric mean of empty array",
        ));
    }
    if arr.iter().any(|&x| x <= 0.0) {
        return Err(PyRuntimeError::new_err(
            "Geometric mean requires all positive values",
        ));
    }
    let log_sum: f64 = arr.iter().map(|x| x.ln()).sum();
    Ok((log_sum / n as f64).exp())
}
/// Harmonic mean
#[pyfunction]
pub fn hmean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err(
            "Cannot compute harmonic mean of empty array",
        ));
    }
    if arr.iter().any(|&x| x <= 0.0) {
        return Err(PyRuntimeError::new_err(
            "Harmonic mean requires all positive values",
        ));
    }
    let reciprocal_sum: f64 = arr.iter().map(|x| 1.0 / x).sum();
    Ok(n as f64 / reciprocal_sum)
}
/// Z-score normalization
#[pyfunction]
pub fn zscore_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;
    let mean: f64 = arr.iter().sum::<f64>() / n;
    let std: f64 = (arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std == 0.0 {
        return Err(PyRuntimeError::new_err(
            "Standard deviation is zero, cannot compute z-scores",
        ));
    }
    let zscores: Vec<f64> = arr.iter().map(|x| (x - mean) / std).collect();
    Ok(PyArray1::from_vec(py, zscores).into())
}
/// Mean absolute deviation
///
/// Calculates the average absolute deviation from the mean (or specified center).
///
/// Parameters:
/// - data: Input data array
/// - center: Optional center value (default: None, uses mean)
///
/// Returns:
/// - Mean absolute deviation value
#[pyfunction]
#[pyo3(signature = (data, center = None))]
pub fn mean_abs_deviation_py(
    data: &Bound<'_, PyArray1<f64>>,
    center: Option<f64>,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    mean_abs_deviation(&arr.view(), center)
        .map_err(|e| PyRuntimeError::new_err(format!("Mean absolute deviation failed: {}", e)))
}
/// Median absolute deviation
///
/// Calculates the median absolute deviation from the median.
/// This is a robust measure of variability.
///
/// Parameters:
/// - data: Input data array
/// - center: Optional center value (default: None, uses median)
/// - scale: Optional scale factor (default: None, uses 1.0)
///
/// Returns:
/// - Median absolute deviation value
#[pyfunction]
#[pyo3(signature = (data, center = None, scale = None))]
pub fn median_abs_deviation_py(
    data: &Bound<'_, PyArray1<f64>>,
    center: Option<f64>,
    scale: Option<f64>,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    median_abs_deviation(&arr.view(), center, scale)
        .map_err(|e| PyRuntimeError::new_err(format!("Median absolute deviation failed: {}", e)))
}
/// Data range
///
/// Calculates the range of the data (maximum - minimum).
///
/// Parameters:
/// - data: Input data array
///
/// Returns:
/// - Range value (max - min)
#[pyfunction]
pub fn data_range_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    data_range(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Data range calculation failed: {}", e)))
}
/// Coefficient of variation
///
/// Calculates the coefficient of variation (standard deviation / mean).
/// This is a unitless measure of relative variability.
///
/// Parameters:
/// - data: Input data array
/// - ddof: Degrees of freedom for standard deviation (default: 1)
///
/// Returns:
/// - Coefficient of variation
#[pyfunction]
#[pyo3(signature = (data, ddof = 1))]
pub fn coef_variation_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    coef_variation(&arr.view(), ddof).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Coefficient of variation calculation failed: {}",
            e
        ))
    })
}
/// Gini coefficient
///
/// Calculates the Gini coefficient, a measure of inequality.
/// Values range from 0 (perfect equality) to 1 (maximum inequality).
///
/// Parameters:
/// - data: Input data array (must be non-negative)
///
/// Returns:
/// - Gini coefficient
#[pyfunction]
pub fn gini_coefficient_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    gini_coefficient(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Gini coefficient calculation failed: {}", e)))
}
/// Compute boxplot statistics (five-number summary + outliers)
#[pyfunction]
#[pyo3(signature = (data, whis = 1.5))]
pub fn boxplot_stats_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    whis: f64,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let (q1, median, q3, whislo, whishi, outliers) =
        boxplot_stats::<f64>(&arr.view(), Some(whis), QuantileInterpolation::Linear).map_err(
            |e| PyRuntimeError::new_err(format!("Boxplot stats calculation failed: {}", e)),
        )?;
    let dict = PyDict::new(py);
    dict.set_item("q1", q1)?;
    dict.set_item("median", median)?;
    dict.set_item("q3", q3)?;
    dict.set_item("whislo", whislo)?;
    dict.set_item("whishi", whishi)?;
    dict.set_item("outliers", outliers)?;
    Ok(dict.into())
}
/// Compute quartiles (Q1, Q2, Q3)
#[pyfunction]
pub fn quartiles_py(
    py: Python<'_>,
    data: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let result = quartiles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Quartiles calculation failed: {}", e)))?;
    Ok(PyArray1::from_vec(py, result.to_vec()).into())
}
/// Compute winsorized mean (robust mean)
#[pyfunction]
#[pyo3(signature = (data, limits = 0.1))]
pub fn winsorized_mean_py(data: &Bound<'_, PyArray1<f64>>, limits: f64) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    winsorized_mean::<f64>(&arr.view(), limits)
        .map_err(|e| PyRuntimeError::new_err(format!("Winsorized mean calculation failed: {}", e)))
}
/// Compute winsorized variance (robust variance)
#[pyfunction]
#[pyo3(signature = (data, limits = 0.1, ddof = 1))]
pub fn winsorized_variance_py(
    data: &Bound<'_, PyArray1<f64>>,
    limits: f64,
    ddof: usize,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    winsorized_variance::<f64>(&arr.view(), limits, ddof).map_err(|e| {
        PyRuntimeError::new_err(format!("Winsorized variance calculation failed: {}", e))
    })
}
/// Compute Shannon entropy of discrete data
#[pyfunction]
#[pyo3(signature = (data, base = None))]
pub fn entropy_py(data: &Bound<'_, PyArray1<i64>>, base: Option<f64>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    entropy::<i64>(&arr.view(), base)
        .map_err(|e| PyRuntimeError::new_err(format!("Entropy calculation failed: {}", e)))
}
/// Compute Kullback-Leibler divergence between two probability distributions
#[pyfunction]
pub fn kl_divergence_py(
    p: &Bound<'_, PyArray1<f64>>,
    q: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let p_binding = p.readonly();
    let p_arr = p_binding.as_array();
    let q_binding = q.readonly();
    let q_arr = q_binding.as_array();
    kl_divergence::<f64>(&p_arr.view(), &q_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("KL divergence calculation failed: {}", e)))
}
/// Compute cross-entropy between two probability distributions
#[pyfunction]
pub fn cross_entropy_py(
    p: &Bound<'_, PyArray1<f64>>,
    q: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let p_binding = p.readonly();
    let p_arr = p_binding.as_array();
    let q_binding = q.readonly();
    let q_arr = q_binding.as_array();
    cross_entropy::<f64>(&p_arr.view(), &q_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Cross-entropy calculation failed: {}", e)))
}
/// Compute weighted mean
#[pyfunction]
pub fn weighted_mean_py(
    data: &Bound<'_, PyArray1<f64>>,
    weights: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let data_binding = data.readonly();
    let data_arr = data_binding.as_array();
    let weights_binding = weights.readonly();
    let weights_arr = weights_binding.as_array();
    weighted_mean::<f64>(&data_arr.view(), &weights_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Weighted mean calculation failed: {}", e)))
}
/// Compute statistical moment
#[pyfunction]
#[pyo3(signature = (data, order, center = true))]
pub fn moment_py(data: &Bound<'_, PyArray1<f64>>, order: usize, center: bool) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    moment::<f64>(&arr.view(), order, center, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Moment calculation failed: {}", e)))
}
/// Compute quintiles (20th, 40th, 60th, 80th percentiles) of a dataset
#[pyfunction]
pub fn quintiles_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let result = quintiles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Quintiles calculation failed: {}", e)))?;
    Ok(PyArray1::from_vec(data.py(), result.to_vec()).into())
}
/// Compute skewness with bootstrap confidence interval
#[pyfunction]
#[pyo3(
    signature = (data, bias = false, confidence = 0.95, n_bootstrap = 1000, seed = None)
)]
pub fn skewness_ci_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    bias: bool,
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let result = skewness_ci::<f64>(&arr.view(), bias, Some(confidence), Some(n_bootstrap), seed)
        .map_err(|e| {
        PyRuntimeError::new_err(format!("Skewness CI calculation failed: {}", e))
    })?;
    let dict = PyDict::new(py);
    dict.set_item("estimate", result.estimate)?;
    dict.set_item("lower", result.lower)?;
    dict.set_item("upper", result.upper)?;
    dict.set_item("confidence", result.confidence)?;
    Ok(dict.into())
}
/// Compute kurtosis with bootstrap confidence interval
#[pyfunction]
#[pyo3(
    signature = (
        data,
        fisher = true,
        bias = false,
        confidence = 0.95,
        n_bootstrap = 1000,
        seed = None
    )
)]
pub fn kurtosis_ci_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    fisher: bool,
    bias: bool,
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let result = kurtosis_ci::<f64>(
        &arr.view(),
        fisher,
        bias,
        Some(confidence),
        Some(n_bootstrap),
        seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Kurtosis CI calculation failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("estimate", result.estimate)?;
    dict.set_item("lower", result.lower)?;
    dict.set_item("upper", result.upper)?;
    dict.set_item("confidence", result.confidence)?;
    Ok(dict.into())
}
