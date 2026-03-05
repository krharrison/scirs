//! Batch/vectorized Python APIs for statistics
//!
//! Provides batch operations that reduce FFI overhead when calling many small
//! statistical computations from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use scirs2_core::ndarray::Array1;
use scirs2_stats::distributions::beta::Beta as RustBeta;
use scirs2_stats::distributions::exponential::Exponential as RustExponential;
use scirs2_stats::distributions::gamma::Gamma as RustGamma;
use scirs2_stats::distributions::normal::Normal as RustNormal;
use scirs2_stats::distributions::uniform::Uniform as RustUniform;
use scirs2_stats::pearsonr;

// ============================================================
// Internal helpers (no PyO3 dependency)
// ============================================================

/// Compute mean of a slice in a single pass.
fn slice_mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &v in remainder {
        sum += v;
    }
    Some(sum / data.len() as f64)
}

/// Compute (mean, variance_sample, std_sample) in two passes.
fn slice_mean_var_std(data: &[f64]) -> Option<(f64, f64, f64)> {
    if data.is_empty() {
        return None;
    }
    let n = data.len();
    let mean = slice_mean(data)?;
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    let chunks = data.chunks_exact(8);
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
    for &v in remainder {
        let d = v - mean;
        sq_sum += d * d;
    }
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let var = sq_sum / denom;
    let std = var.sqrt();
    Some((mean, std, var))
}

/// Compute percentile from a sorted slice via linear interpolation.
fn sorted_percentile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let virtual_index = p * (n - 1) as f64;
    let i = virtual_index.floor() as usize;
    let frac = virtual_index - i as f64;
    if frac == 0.0 || i >= n - 1 {
        sorted[i.min(n - 1)]
    } else {
        sorted[i] + frac * (sorted[i + 1] - sorted[i])
    }
}

/// Full descriptive stats dict for a single slice.
fn descriptive_stats_for_slice(
    data: &[f64],
) -> Result<std::collections::HashMap<String, f64>, String> {
    let n = data.len();
    if n == 0 {
        return Err("Empty array".to_string());
    }
    let (mean, std, var) =
        slice_mean_var_std(data).ok_or_else(|| "Failed to compute mean/std".to_string())?;
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_percentile(&sorted, 0.5);
    let q25 = sorted_percentile(&sorted, 0.25);
    let q75 = sorted_percentile(&sorted, 0.75);

    let mut map = std::collections::HashMap::new();
    map.insert("n".to_string(), n as f64);
    map.insert("mean".to_string(), mean);
    map.insert("std".to_string(), std);
    map.insert("var".to_string(), var);
    map.insert("min".to_string(), min);
    map.insert("max".to_string(), max);
    map.insert("median".to_string(), median);
    map.insert("q25".to_string(), q25);
    map.insert("q75".to_string(), q75);
    Ok(map)
}

// ============================================================
// Public batch #[pyfunction]s
// ============================================================

/// Compute mean, std, and variance in a single batch call.
///
/// Avoids 3 separate FFI calls by computing all three in a single pass.
///
/// Parameters:
///     data: Input data array
///
/// Returns:
///     Tuple (mean, std, variance) — sample variance (ddof=1)
#[pyfunction]
pub fn stats_summary(data: Vec<f64>) -> PyResult<(f64, f64, f64)> {
    if data.is_empty() {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }
    let (mean, std, var) = slice_mean_var_std(&data)
        .ok_or_else(|| PyRuntimeError::new_err("Failed to compute stats"))?;
    Ok((mean, std, var))
}

/// Batch descriptive stats for multiple arrays.
///
/// For each array computes: n, mean, std, var, min, max, median, q25, q75.
///
/// Parameters:
///     arrays: List of data arrays
///
/// Returns:
///     List of dicts, one per input array
#[pyfunction]
pub fn batch_descriptive_stats(
    arrays: Vec<Vec<f64>>,
) -> PyResult<Vec<std::collections::HashMap<String, f64>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }
    let results: Vec<Result<std::collections::HashMap<String, f64>, String>> = arrays
        .par_iter()
        .map(|arr| descriptive_stats_for_slice(arr))
        .collect();

    results
        .into_iter()
        .map(|r| r.map_err(|e| PyRuntimeError::new_err(format!("Descriptive stats failed: {}", e))))
        .collect()
}

/// Batch Pearson correlation matrix for a list of arrays.
///
/// Computes the full correlation matrix for the provided arrays.
/// Entry [i][j] is the Pearson correlation between arrays[i] and arrays[j].
///
/// Parameters:
///     arrays: List of arrays (all must have the same length)
///
/// Returns:
///     Correlation matrix as Vec<Vec<f64>>
#[pyfunction]
pub fn batch_correlation(arrays: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let k = arrays.len();
    if k == 0 {
        return Ok(vec![]);
    }
    let n = arrays[0].len();
    for (i, arr) in arrays.iter().enumerate() {
        if arr.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "Array {} has length {} but expected {}",
                i,
                arr.len(),
                n
            )));
        }
        if arr.is_empty() {
            return Err(PyRuntimeError::new_err(format!("Array {} is empty", i)));
        }
    }

    // Build index pairs for upper triangle (including diagonal)
    let pairs: Vec<(usize, usize)> = (0..k).flat_map(|i| (i..k).map(move |j| (i, j))).collect();

    // Compute correlations in parallel for upper triangle
    let corr_values: Vec<((usize, usize), f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            if i == j {
                return Ok(((i, j), 1.0_f64));
            }
            let x_arr = Array1::from_vec(arrays[i].clone());
            let y_arr = Array1::from_vec(arrays[j].clone());
            pearsonr(&x_arr.view(), &y_arr.view(), "two-sided")
                .map(|(r, _p)| ((i, j), r))
                .map_err(|e| format!("Pearson correlation ({},{}) failed: {}", i, j, e))
        })
        .collect::<Vec<Result<((usize, usize), f64), String>>>()
        .into_iter()
        .collect::<Result<Vec<((usize, usize), f64)>, String>>()
        .map_err(|e| PyRuntimeError::new_err(e))?;

    // Fill symmetric matrix
    let mut matrix = vec![vec![0.0f64; k]; k];
    for ((i, j), val) in corr_values {
        matrix[i][j] = val;
        matrix[j][i] = val;
    }
    Ok(matrix)
}

/// Evaluate the PDF of a named distribution at each data point.
///
/// Supported distributions: "normal", "exponential", "uniform", "gamma", "beta"
///
/// Parameters:
///     data: Points at which to evaluate the PDF
///     distribution: Distribution name (e.g., "normal")
///     params: Distribution parameters
///         - normal: [mu, sigma]
///         - exponential: [lambda] (rate = 1/scale)
///         - uniform: [low, high]
///         - gamma: [shape, scale]
///         - beta: [alpha, beta_param]
///
/// Returns:
///     Vec<f64> of PDF values
#[pyfunction]
pub fn batch_pdf_eval(data: Vec<f64>, distribution: &str, params: Vec<f64>) -> PyResult<Vec<f64>> {
    if data.is_empty() {
        return Ok(vec![]);
    }

    match distribution.to_lowercase().as_str() {
        "normal" => {
            if params.len() < 2 {
                return Err(PyRuntimeError::new_err(
                    "Normal distribution requires [mu, sigma] params",
                ));
            }
            let mu = params[0];
            let sigma = params[1];
            if sigma <= 0.0 {
                return Err(PyRuntimeError::new_err("sigma must be positive"));
            }
            // Normal::new(loc, scale) where loc=mu, scale=sigma
            let dist = RustNormal::new(mu, sigma).map_err(|e| {
                PyRuntimeError::new_err(format!("Normal distribution failed: {}", e))
            })?;
            let result: Vec<f64> = data.par_iter().map(|&x| dist.pdf(x)).collect();
            Ok(result)
        }
        "exponential" => {
            if params.is_empty() {
                return Err(PyRuntimeError::new_err(
                    "Exponential distribution requires [lambda] params",
                ));
            }
            let lambda = params[0];
            if lambda <= 0.0 {
                return Err(PyRuntimeError::new_err("lambda must be positive"));
            }
            // Exponential::new(rate, loc) where rate=lambda, loc=0
            let dist = RustExponential::new(lambda, 0.0).map_err(|e| {
                PyRuntimeError::new_err(format!("Exponential distribution failed: {}", e))
            })?;
            let result: Vec<f64> = data.par_iter().map(|&x| dist.pdf(x)).collect();
            Ok(result)
        }
        "uniform" => {
            if params.len() < 2 {
                return Err(PyRuntimeError::new_err(
                    "Uniform distribution requires [low, high] params",
                ));
            }
            let low = params[0];
            let high = params[1];
            if high <= low {
                return Err(PyRuntimeError::new_err("high must be greater than low"));
            }
            let dist = RustUniform::new(low, high).map_err(|e| {
                PyRuntimeError::new_err(format!("Uniform distribution failed: {}", e))
            })?;
            let result: Vec<f64> = data.par_iter().map(|&x| dist.pdf(x)).collect();
            Ok(result)
        }
        "gamma" => {
            if params.len() < 2 {
                return Err(PyRuntimeError::new_err(
                    "Gamma distribution requires [shape, scale] params",
                ));
            }
            let shape = params[0];
            let scale = params[1];
            if shape <= 0.0 || scale <= 0.0 {
                return Err(PyRuntimeError::new_err("shape and scale must be positive"));
            }
            let dist = RustGamma::new(shape, scale, 0.0).map_err(|e| {
                PyRuntimeError::new_err(format!("Gamma distribution failed: {}", e))
            })?;
            let result: Vec<f64> = data.par_iter().map(|&x| dist.pdf(x)).collect();
            Ok(result)
        }
        "beta" => {
            if params.len() < 2 {
                return Err(PyRuntimeError::new_err(
                    "Beta distribution requires [alpha, beta] params",
                ));
            }
            let alpha = params[0];
            let beta_param = params[1];
            if alpha <= 0.0 || beta_param <= 0.0 {
                return Err(PyRuntimeError::new_err("alpha and beta must be positive"));
            }
            let dist = RustBeta::new(alpha, beta_param, 0.0, 1.0)
                .map_err(|e| PyRuntimeError::new_err(format!("Beta distribution failed: {}", e)))?;
            let result: Vec<f64> = data.par_iter().map(|&x| dist.pdf(x)).collect();
            Ok(result)
        }
        other => Err(PyRuntimeError::new_err(format!(
            "Unknown distribution: '{}'. Supported: normal, exponential, uniform, gamma, beta",
            other
        ))),
    }
}

/// Register batch stats functions into the Python module.
pub fn register_batch_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(stats_summary, m)?)?;
    m.add_function(wrap_pyfunction!(batch_descriptive_stats, m)?)?;
    m.add_function(wrap_pyfunction!(batch_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(batch_pdf_eval, m)?)?;
    Ok(())
}
