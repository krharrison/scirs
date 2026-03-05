//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{PyArray1, PyArrayMethods};
use scirs2_stats::tests::ttest::{ttest_1samp, ttest_ind, Alternative};
use scirs2_stats::{covariance_simd, pearsonr};

/// Compute descriptive statistics - uses optimized implementations
/// Returns dict with mean, std, var, min, max, median, count
#[pyfunction]
pub fn describe_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    let slice = arr.as_slice().expect("Operation failed");
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean_val = sum / n as f64;
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let d0 = chunk[0] - mean_val;
        let d1 = chunk[1] - mean_val;
        let d2 = chunk[2] - mean_val;
        let d3 = chunk[3] - mean_val;
        let d4 = chunk[4] - mean_val;
        let d5 = chunk[5] - mean_val;
        let d6 = chunk[6] - mean_val;
        let d7 = chunk[7] - mean_val;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
        min_val = min_val
            .min(chunk[0])
            .min(chunk[1])
            .min(chunk[2])
            .min(chunk[3])
            .min(chunk[4])
            .min(chunk[5])
            .min(chunk[6])
            .min(chunk[7]);
        max_val = max_val
            .max(chunk[0])
            .max(chunk[1])
            .max(chunk[2])
            .max(chunk[3])
            .max(chunk[4])
            .max(chunk[5])
            .max(chunk[6])
            .max(chunk[7]);
    }
    for &val in remainder {
        let d = val - mean_val;
        sq0 += d * d;
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }
    let sq_sum = sq0 + sq1 + sq2 + sq3;
    let var_val = if n > 1 { sq_sum / (n - 1) as f64 } else { 0.0 };
    let std_val = var_val.sqrt();
    let mut vec: Vec<f64> = arr.iter().cloned().collect();
    let median_val = if n % 2 == 1 {
        let mid = n / 2;
        let (_, val, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *val
    } else {
        let mid = n / 2;
        let (lower, val_at_mid, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_mid = *val_at_mid;
        let val_mid_minus_1 = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_mid);
        (val_mid_minus_1 + val_mid) / 2.0
    };
    let dict = PyDict::new(py);
    dict.set_item("mean", mean_val)?;
    dict.set_item("std", std_val)?;
    dict.set_item("var", var_val)?;
    dict.set_item("min", min_val)?;
    dict.set_item("max", max_val)?;
    dict.set_item("median", median_val)?;
    dict.set_item("count", n)?;
    Ok(dict.into())
}
/// Calculate mean - optimized with 8-way unrolling and multiple accumulators
#[pyfunction]
pub fn mean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    let slice = arr.as_slice().expect("Operation failed");
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    Ok(sum / n as f64)
}
/// Calculate standard deviation - optimized two-pass with multi-accumulator
#[pyfunction]
#[pyo3(signature = (data, ddof = 0))]
pub fn std_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    if n <= ddof {
        return Err(PyRuntimeError::new_err(
            "Not enough data points for given ddof",
        ));
    }
    let slice = arr.as_slice().expect("Operation failed");
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean = sum / n as f64;
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let d0 = chunk[0] - mean;
        let d1 = chunk[1] - mean;
        let d2 = chunk[2] - mean;
        let d3 = chunk[3] - mean;
        let d4 = chunk[4] - mean;
        let d5 = chunk[5] - mean;
        let d6 = chunk[6] - mean;
        let d7 = chunk[7] - mean;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
    }
    let mut sq_sum = sq0 + sq1 + sq2 + sq3;
    for &val in remainder {
        let d = val - mean;
        sq_sum += d * d;
    }
    let variance = sq_sum / (n - ddof) as f64;
    Ok(variance.sqrt())
}
/// Calculate variance - optimized two-pass with multi-accumulator
#[pyfunction]
#[pyo3(signature = (data, ddof = 0))]
pub fn var_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    if n <= ddof {
        return Err(PyRuntimeError::new_err(
            "Not enough data points for given ddof",
        ));
    }
    let slice = arr.as_slice().expect("Operation failed");
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean = sum / n as f64;
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let d0 = chunk[0] - mean;
        let d1 = chunk[1] - mean;
        let d2 = chunk[2] - mean;
        let d3 = chunk[3] - mean;
        let d4 = chunk[4] - mean;
        let d5 = chunk[5] - mean;
        let d6 = chunk[6] - mean;
        let d7 = chunk[7] - mean;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
    }
    let mut sq_sum = sq0 + sq1 + sq2 + sq3;
    for &val in remainder {
        let d = val - mean;
        sq_sum += d * d;
    }
    Ok(sq_sum / (n - ddof) as f64)
}
/// Calculate percentile using optimized partial sort
/// q: percentile value (0-100)
#[pyfunction]
pub fn percentile_py(data: &Bound<'_, PyArray1<f64>>, q: f64) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    if !(0.0..=100.0).contains(&q) {
        return Err(PyRuntimeError::new_err(
            "Percentile must be between 0 and 100",
        ));
    }
    let mut vec: Vec<f64> = arr.iter().cloned().collect();
    let q_norm = q / 100.0;
    let virtual_index = q_norm * (n - 1) as f64;
    let i = virtual_index.floor() as usize;
    let fraction = virtual_index - i as f64;
    if fraction == 0.0 || i >= n - 1 {
        let idx = i.min(n - 1);
        let (_, val, _) = vec.select_nth_unstable_by(idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(*val)
    } else {
        let (lower, val_i1, _) = vec.select_nth_unstable_by(i + 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_upper = *val_i1;
        let val_lower = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_upper);
        Ok(val_lower + fraction * (val_upper - val_lower))
    }
}
/// Calculate correlation coefficient
#[pyfunction]
pub fn correlation_py(x: &Bound<'_, PyArray1<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let x_binding = x.readonly();
    let x_arr = x_binding.as_array();
    let y_binding = y.readonly();
    let y_arr = y_binding.as_array();
    let (r, _p) = pearsonr(&x_arr, &y_arr, "two-sided")
        .map_err(|e| PyRuntimeError::new_err(format!("Correlation failed: {}", e)))?;
    Ok(r)
}
/// Calculate covariance
#[pyfunction]
#[pyo3(signature = (x, y, ddof = 1))]
pub fn covariance_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    ddof: usize,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let x_arr = x_binding.as_array();
    let y_binding = y.readonly();
    let y_arr = y_binding.as_array();
    covariance_simd(&x_arr, &y_arr, ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("Covariance failed: {}", e)))
}
/// Calculate median using optimized partial sort (O(n) instead of O(n log n))
#[pyfunction]
pub fn median_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    let mut vec: Vec<f64> = arr.iter().cloned().collect();
    if n % 2 == 1 {
        let mid = n / 2;
        let (_, median_val, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(*median_val)
    } else {
        let mid = n / 2;
        let (lower, val_at_mid, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_mid = *val_at_mid;
        let val_mid_minus_1 = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_mid);
        Ok((val_mid_minus_1 + val_mid) / 2.0)
    }
}
/// Calculate interquartile range (IQR) using optimized partial sort
#[pyfunction]
pub fn iqr_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();
    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    let mut vec: Vec<f64> = arr.iter().cloned().collect();
    let get_percentile = |vec: &mut [f64], q: f64| -> f64 {
        let virtual_index = q * (n - 1) as f64;
        let i = virtual_index.floor() as usize;
        let fraction = virtual_index - i as f64;
        if fraction == 0.0 || i >= n - 1 {
            let idx = i.min(n - 1);
            let (_, val, _) = vec.select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            *val
        } else {
            let (lower, val_i1, _) = vec.select_nth_unstable_by(i + 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let val_upper = *val_i1;
            let val_lower = lower
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(val_upper);
            val_lower + fraction * (val_upper - val_lower)
        }
    };
    let q75 = get_percentile(&mut vec, 0.75);
    let mut vec2: Vec<f64> = arr.iter().cloned().collect();
    let q25 = get_percentile(&mut vec2, 0.25);
    Ok(q75 - q25)
}
/// One-sample t-test
///
/// Test whether the mean of a sample is different from a given value.
///
/// Parameters:
/// - data: Input data
/// - popmean: Population mean for null hypothesis
/// - alternative: "two-sided", "less", or "greater"
#[pyfunction]
#[pyo3(signature = (data, popmean, alternative = "two-sided"))]
pub fn ttest_1samp_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    popmean: f64,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
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
    let result = ttest_1samp(&arr.view(), popmean, alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;
    Ok(dict.into())
}
/// Two-sample independent t-test
///
/// Test whether two independent samples have different means.
///
/// Parameters:
/// - a: First sample
/// - b: Second sample
/// - equal_var: If true, perform standard t-test assuming equal variance
/// - alternative: "two-sided", "less", or "greater"
#[pyfunction]
#[pyo3(signature = (a, b, equal_var = true, alternative = "two-sided"))]
pub fn ttest_ind_py(
    py: Python,
    a: &Bound<'_, PyArray1<f64>>,
    b: &Bound<'_, PyArray1<f64>>,
    equal_var: bool,
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
    let result = ttest_ind(&a_arr.view(), &b_arr.view(), equal_var, alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;
    Ok(dict.into())
}
