//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_core::{ndarray::ArrayView1, Array1};
use scirs2_numpy::{PyArray1, PyArrayMethods};
use scirs2_stats::{
    anderson_darling, bartlett, brown_forsythe, chi2_gof, covariance_simd, dagostino_k2, deciles,
    kruskal_wallis, kurtosis_simd, levene, mann_whitney, mean_simd, moment_simd, one_way_anova,
    pearson_r_simd, percentile_range_simd, sem_simd, shapiro_wilk, skewness_simd, std_simd,
    variance_simd, wilcoxon, QuantileInterpolation,
};

/// Compute deciles (10th, 20th, ..., 90th percentiles) of a dataset
#[pyfunction]
pub fn deciles_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let result = deciles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Deciles calculation failed: {}", e)))?;
    Ok(PyArray1::from_vec(data.py(), result.to_vec()).into())
}
/// Compute standard error of the mean (SEM)
#[pyfunction]
#[pyo3(signature = (data, ddof = 1))]
pub fn sem_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    sem_simd::<f64, _>(&arr, ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SEM calculation failed: {}", e)))
}
/// Compute the range between two percentiles
#[pyfunction]
#[pyo3(signature = (data, lower_pct, upper_pct, interpolation = "linear"))]
pub fn percentile_range_py(
    data: &Bound<'_, PyArray1<f64>>,
    lower_pct: f64,
    upper_pct: f64,
    interpolation: &str,
) -> PyResult<f64> {
    let binding = data.readonly();
    let mut arr = binding.as_array().to_owned();
    percentile_range_simd::<f64, _>(&mut arr, lower_pct, upper_pct, interpolation)
        .map_err(|e| PyRuntimeError::new_err(format!("Percentile range calculation failed: {}", e)))
}
/// Compute SIMD-optimized skewness (third standardized moment)
#[pyfunction]
#[pyo3(signature = (data, bias = false))]
pub fn skewness_simd_py(data: &Bound<'_, PyArray1<f64>>, bias: bool) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    skewness_simd::<f64, _>(&arr.view(), bias)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD skewness calculation failed: {}", e)))
}
/// Compute SIMD-optimized kurtosis (fourth standardized moment)
#[pyfunction]
#[pyo3(signature = (data, fisher = true, bias = false))]
pub fn kurtosis_simd_py(
    data: &Bound<'_, PyArray1<f64>>,
    fisher: bool,
    bias: bool,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    kurtosis_simd::<f64, _>(&arr.view(), fisher, bias)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD kurtosis calculation failed: {}", e)))
}
/// Compute SIMD-optimized Pearson correlation coefficient
#[pyfunction]
pub fn pearson_r_simd_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let y_binding = y.readonly();
    let x_arr = x_binding.as_array();
    let y_arr = y_binding.as_array();
    pearson_r_simd::<f64, _>(&x_arr.view(), &y_arr.view()).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "SIMD Pearson correlation calculation failed: {}",
            e
        ))
    })
}
/// Compute SIMD-optimized covariance
#[pyfunction]
#[pyo3(signature = (x, y, ddof = 1))]
pub fn covariance_simd_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    ddof: usize,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let y_binding = y.readonly();
    let x_arr = x_binding.as_array();
    let y_arr = y_binding.as_array();
    covariance_simd::<f64, _>(&x_arr.view(), &y_arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD covariance calculation failed: {}", e)))
}
/// Compute SIMD-optimized nth statistical moment
#[pyfunction]
#[pyo3(signature = (data, moment_order, center = true))]
pub fn moment_simd_py(
    data: &Bound<'_, PyArray1<f64>>,
    moment_order: usize,
    center: bool,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    moment_simd::<f64, _>(&arr.view(), moment_order, center)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD moment calculation failed: {}", e)))
}
/// Compute SIMD-optimized mean
#[pyfunction]
pub fn mean_simd_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    mean_simd::<f64, _>(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD mean calculation failed: {}", e)))
}
/// Compute SIMD-optimized standard deviation
#[pyfunction]
#[pyo3(signature = (data, ddof = 1))]
pub fn std_simd_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    std_simd::<f64, _>(&arr.view(), ddof).map_err(|e| {
        PyRuntimeError::new_err(format!("SIMD standard deviation calculation failed: {}", e))
    })
}
/// Compute SIMD-optimized variance
#[pyfunction]
#[pyo3(signature = (data, ddof = 1))]
pub fn variance_simd_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    variance_simd::<f64, _>(&arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD variance calculation failed: {}", e)))
}
/// Shapiro-Wilk test for normality
///
/// Tests the null hypothesis that the data was drawn from a normal distribution.
///
/// Parameters:
/// - data: Input data array
///
/// Returns:
/// - Dict with 'statistic' (W statistic) and 'pvalue'
#[pyfunction]
pub fn shapiro_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let (statistic, pvalue) = shapiro_wilk(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Shapiro-Wilk test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Chi-square goodness-of-fit test
///
/// Tests whether observed frequencies differ from expected frequencies.
///
/// Parameters:
/// - observed: Observed frequencies (integers or floats)
/// - expected: Expected frequencies (optional, defaults to uniform)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue', 'dof' (degrees of freedom)
#[pyfunction]
#[pyo3(signature = (observed, expected = None))]
pub fn chisquare_py(
    py: Python,
    observed: &Bound<'_, PyArray1<f64>>,
    expected: Option<&Bound<'_, PyArray1<f64>>>,
) -> PyResult<Py<PyAny>> {
    let obs_binding = observed.readonly();
    let obs_arr = obs_binding.as_array();
    let obs_int: Vec<i64> = obs_arr.iter().map(|&x| x.round() as i64).collect();
    let obs_int_arr = Array1::from_vec(obs_int);
    let exp_opt = expected.map(|e| {
        let e_binding = e.readonly();
        let e_arr = e_binding.as_array();
        e_arr.to_owned()
    });
    let result = chi2_gof(&obs_int_arr.view(), exp_opt.as_ref().map(|e| e.view()))
        .map_err(|e| PyRuntimeError::new_err(format!("Chi-square test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("dof", result.df)?;
    Ok(dict.into())
}
/// One-way ANOVA (Analysis of Variance)
///
/// Tests whether the means of multiple groups are equal.
///
/// Parameters:
/// - *args: Variable number of arrays, each representing a group
///
/// Returns:
/// - Dict with 'f_statistic', 'pvalue', 'df_between', 'df_within',
///   'ss_between', 'ss_within', 'ms_between', 'ms_within'
#[pyfunction(signature = (*args))]
pub fn f_oneway_py(py: Python, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err("Need at least 2 groups for ANOVA"));
    }
    let mut group_arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let binding = arr.readonly();
        group_arrays.push(binding.as_array().to_owned());
    }
    let group_views: Vec<ArrayView1<f64>> = group_arrays.iter().map(|a| a.view()).collect();
    let group_refs: Vec<&ArrayView1<f64>> = group_views.iter().collect();
    let result = one_way_anova(&group_refs)
        .map_err(|e| PyRuntimeError::new_err(format!("ANOVA failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("f_statistic", result.f_statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df_between", result.df_treatment)?;
    dict.set_item("df_within", result.df_error)?;
    dict.set_item("ss_between", result.ss_treatment)?;
    dict.set_item("ss_within", result.ss_error)?;
    dict.set_item("ms_between", result.ms_treatment)?;
    dict.set_item("ms_within", result.ms_error)?;
    Ok(dict.into())
}
/// Wilcoxon signed-rank test for paired samples.
///
/// Parameters:
/// - x: First array of observations
/// - y: Second array of observations (paired with x)
/// - zero_method: How to handle zero differences: "wilcox" (default), "pratt"
/// - correction: Whether to apply continuity correction (default: True)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction]
#[pyo3(signature = (x, y, zero_method = "wilcox", correction = true))]
pub fn wilcoxon_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    zero_method: &str,
    correction: bool,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();
    let y_data = y.readonly();
    let y_arr = y_data.as_array();
    let (statistic, pvalue) = wilcoxon(&x_arr.view(), &y_arr.view(), zero_method, correction)
        .map_err(|e| PyRuntimeError::new_err(format!("Wilcoxon test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Mann-Whitney U test for independent samples.
///
/// Parameters:
/// - x: First array of observations
/// - y: Second array of observations
/// - alternative: Alternative hypothesis: "two-sided" (default), "less", or "greater"
/// - use_continuity: Whether to apply continuity correction (default: True)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction]
#[pyo3(signature = (x, y, alternative = "two-sided", use_continuity = true))]
pub fn mannwhitneyu_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
    use_continuity: bool,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();
    let y_data = y.readonly();
    let y_arr = y_data.as_array();
    let (statistic, pvalue) =
        mann_whitney(&x_arr.view(), &y_arr.view(), alternative, use_continuity)
            .map_err(|e| PyRuntimeError::new_err(format!("Mann-Whitney U test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Kruskal-Wallis H-test for independent samples.
///
/// Parameters:
/// - *args: Variable number of arrays, one for each group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
pub fn kruskal_py(py: Python, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Kruskal-Wallis test",
        ));
    }
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let (statistic, pvalue) = kruskal_wallis(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Kruskal-Wallis test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Levene's test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
/// - center: Which function to use: "mean", "median" (default), or "trimmed"
/// - proportion_to_cut: When using "trimmed", the proportion to cut from each end (default: 0.05)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args, center = "median", proportion_to_cut = 0.05))]
pub fn levene_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
    center: &str,
    proportion_to_cut: f64,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Levene's test",
        ));
    }
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let (statistic, pvalue) = levene(&views, center, proportion_to_cut)
        .map_err(|e| PyRuntimeError::new_err(format!("Levene's test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Bartlett's test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
pub fn bartlett_test_py(py: Python, args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Bartlett's test",
        ));
    }
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let (statistic, pvalue) = bartlett(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Bartlett's test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Brown-Forsythe test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
pub fn brown_forsythe_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Brown-Forsythe test",
        ));
    }
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let (statistic, pvalue) = brown_forsythe(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Brown-Forsythe test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Anderson-Darling test for normality.
///
/// Tests whether a sample comes from a normal distribution using the
/// Anderson-Darling statistic. More sensitive to deviations in the tails
/// than the Shapiro-Wilk test.
///
/// Parameters:
///     x: Array of sample data
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
pub fn anderson_darling_py(py: Python, x: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();
    let (statistic, pvalue) = anderson_darling(&x_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Anderson-Darling test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// D'Agostino's K-squared test for normality.
///
/// Tests whether a sample comes from a normal distribution using the
/// D'Agostino-Pearson K² test, which combines tests for skewness and kurtosis.
///
/// Parameters:
///     x: Array of sample data (minimum 20 observations)
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
pub fn dagostino_k2_py(py: Python, x: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();
    let (statistic, pvalue) = dagostino_k2(&x_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("D'Agostino K² test failed: {}", e)))?;
    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
