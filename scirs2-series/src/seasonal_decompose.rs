//! Convenience functions for seasonal decomposition
//!
//! Provides a simple `seasonal_decompose` function similar to Python's
//! `statsmodels.tsa.seasonal.seasonal_decompose` and `STL`.
//!
//! This module unifies the different decomposition approaches into a single
//! easy-to-use interface.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::decomposition::common::{DecompositionModel, DecompositionResult};
use crate::decomposition::seasonal::decompose_seasonal;
use crate::decomposition::stl::{stl_decomposition, STLOptions};
use crate::error::{Result, TimeSeriesError};

/// Decomposition method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecomposeMethod {
    /// Classical decomposition using centered moving averages
    Classical,
    /// STL (Seasonal-Trend decomposition using LOESS)
    STL,
}

impl Default for DecomposeMethod {
    fn default() -> Self {
        Self::STL
    }
}

/// Decompose a time series into trend, seasonal, and residual components
///
/// This is a convenience function that provides a simple interface to the
/// different decomposition methods available in the library.
///
/// # Arguments
///
/// * `data` - Time series data
/// * `period` - Seasonal period (e.g., 12 for monthly data with yearly seasonality)
/// * `model` - Decomposition model ("additive" or "multiplicative")
///
/// # Returns
///
/// A tuple of (trend, seasonal, residual) arrays
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::seasonal_decompose::seasonal_decompose;
///
/// let data = array![
///     1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///     1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0
/// ];
/// let (trend, seasonal, residual) = seasonal_decompose(&data, 4, "additive")
///     .expect("Decomposition failed");
///
/// assert_eq!(trend.len(), data.len());
/// assert_eq!(seasonal.len(), data.len());
/// assert_eq!(residual.len(), data.len());
/// ```
pub fn seasonal_decompose<F>(
    data: &Array1<F>,
    period: usize,
    model: &str,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    seasonal_decompose_with_method(data, period, model, DecomposeMethod::STL)
}

/// Decompose a time series with a specific method
///
/// # Arguments
///
/// * `data` - Time series data
/// * `period` - Seasonal period
/// * `model` - Decomposition model ("additive" or "multiplicative")
/// * `method` - Decomposition method (Classical or STL)
///
/// # Returns
///
/// A tuple of (trend, seasonal, residual) arrays
pub fn seasonal_decompose_with_method<F>(
    data: &Array1<F>,
    period: usize,
    model: &str,
    method: DecomposeMethod,
) -> Result<(Array1<F>, Array1<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug,
{
    if period < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "Seasonal period must be at least 2".to_string(),
        });
    }

    if data.len() < 2 * period {
        return Err(TimeSeriesError::InsufficientData {
            message: "Data length must be at least twice the seasonal period".to_string(),
            required: 2 * period,
            actual: data.len(),
        });
    }

    let decomp_model = match model.to_lowercase().as_str() {
        "additive" | "add" | "a" => DecompositionModel::Additive,
        "multiplicative" | "mul" | "m" => DecompositionModel::Multiplicative,
        _ => {
            return Err(TimeSeriesError::InvalidParameter {
                name: "model".to_string(),
                message: format!(
                    "Unknown model '{}'. Use 'additive' or 'multiplicative'",
                    model
                ),
            });
        }
    };

    match method {
        DecomposeMethod::Classical => {
            let result = decompose_seasonal(data, period, decomp_model)?;
            Ok((result.trend, result.seasonal, result.residual))
        }
        DecomposeMethod::STL => {
            if decomp_model == DecompositionModel::Multiplicative {
                // For multiplicative STL: transform to log space, decompose, transform back
                // Check all positive
                if data.iter().any(|&x| x <= F::zero()) {
                    return Err(TimeSeriesError::InvalidInput(
                        "Multiplicative decomposition requires strictly positive data".to_string(),
                    ));
                }

                let log_data = data.mapv(|x| x.ln());
                let options = STLOptions {
                    trend_window: compute_trend_window(period),
                    seasonal_window: compute_seasonal_window(period),
                    n_inner: 2,
                    n_outer: if data.len() > 50 { 1 } else { 0 },
                    robust: data.len() > 50,
                };

                let result = stl_decomposition(&log_data, period, &options)?;

                // Transform back to original scale
                let trend = result.trend.mapv(|x| x.exp());
                let seasonal = result.seasonal.mapv(|x| x.exp());
                let n = data.len();
                let mut residual = Array1::zeros(n);
                for i in 0..n {
                    if trend[i] > F::zero() && seasonal[i] > F::zero() {
                        residual[i] = data[i] / (trend[i] * seasonal[i]);
                    } else {
                        residual[i] = F::one();
                    }
                }

                Ok((trend, seasonal, residual))
            } else {
                let options = STLOptions {
                    trend_window: compute_trend_window(period),
                    seasonal_window: compute_seasonal_window(period),
                    n_inner: 2,
                    n_outer: if data.len() > 50 { 1 } else { 0 },
                    robust: data.len() > 50,
                };

                let result = stl_decomposition(data, period, &options)?;
                Ok((result.trend, result.seasonal, result.residual))
            }
        }
    }
}

/// Compute an appropriate trend window size for the given period
///
/// Following Cleveland et al. (1990), the trend window should be
/// the smallest odd integer >= 1.5 * period / (1 - 1.5 / seasonal_window)
fn compute_trend_window(period: usize) -> usize {
    // A reasonable default: period + 1 if even, period if odd, with minimum of 7
    let base = ((1.5 * period as f64).ceil() as usize).max(7);
    if base % 2 == 0 {
        base + 1
    } else {
        base
    }
}

/// Compute an appropriate seasonal window size for the given period
fn compute_seasonal_window(period: usize) -> usize {
    // Default: 7 or period + 1, whichever is larger, ensured odd
    let base = (period + 1).max(7);
    if base % 2 == 0 {
        base + 1
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn make_seasonal_data() -> Array1<f64> {
        // Data with clear seasonal pattern (period 4) and upward trend
        let mut data = Vec::with_capacity(20);
        for i in 0..20 {
            let trend = 10.0 + 0.5 * i as f64;
            let seasonal = match i % 4 {
                0 => 2.0,
                1 => -1.0,
                2 => 3.0,
                3 => -4.0,
                _ => 0.0,
            };
            data.push(trend + seasonal);
        }
        Array1::from_vec(data)
    }

    #[test]
    fn test_seasonal_decompose_additive_stl() {
        let data = make_seasonal_data();
        let (trend, seasonal, residual) =
            seasonal_decompose(&data, 4, "additive").expect("STL decomposition failed");

        assert_eq!(trend.len(), data.len());
        assert_eq!(seasonal.len(), data.len());
        assert_eq!(residual.len(), data.len());

        // Trend + Seasonal + Residual should reconstruct original
        for i in 0..data.len() {
            let reconstructed = trend[i] + seasonal[i] + residual[i];
            assert!(
                (reconstructed - data[i]).abs() < 1e-6,
                "Reconstruction error at index {}: {} vs {}",
                i,
                reconstructed,
                data[i]
            );
        }
    }

    #[test]
    fn test_seasonal_decompose_classical() {
        let data = make_seasonal_data();
        let (trend, seasonal, residual) =
            seasonal_decompose_with_method(&data, 4, "additive", DecomposeMethod::Classical)
                .expect("Classical decomposition failed");

        assert_eq!(trend.len(), data.len());
        assert_eq!(seasonal.len(), data.len());
        assert_eq!(residual.len(), data.len());
    }

    #[test]
    fn test_seasonal_decompose_multiplicative() {
        // Multiplicative data (all positive)
        let mut data = Vec::with_capacity(20);
        for i in 0..20 {
            let trend = 10.0 + 0.5 * i as f64;
            let seasonal = match i % 4 {
                0 => 1.2,
                1 => 0.9,
                2 => 1.3,
                3 => 0.6,
                _ => 1.0,
            };
            data.push(trend * seasonal);
        }
        let data = Array1::from_vec(data);

        let result = seasonal_decompose(&data, 4, "multiplicative");
        assert!(
            result.is_ok(),
            "Multiplicative decomposition should succeed"
        );

        let (trend, seasonal, _residual) = result.expect("Should not fail");
        assert_eq!(trend.len(), data.len());
        assert_eq!(seasonal.len(), data.len());
    }

    #[test]
    fn test_seasonal_decompose_insufficient_data() {
        let data = array![1.0, 2.0, 3.0];
        let result = seasonal_decompose(&data, 4, "additive");
        assert!(result.is_err(), "Should fail with insufficient data");
    }

    #[test]
    fn test_seasonal_decompose_invalid_model() {
        let data = make_seasonal_data();
        let result = seasonal_decompose(&data, 4, "invalid_model");
        assert!(result.is_err(), "Should fail with invalid model");
    }

    #[test]
    fn test_seasonal_decompose_period_too_small() {
        let data = make_seasonal_data();
        let result = seasonal_decompose(&data, 1, "additive");
        assert!(result.is_err(), "Should fail with period < 2");
    }

    #[test]
    fn test_seasonal_decompose_captures_trend() {
        // Use longer data series and classical method for reliable trend extraction
        let mut data = Vec::with_capacity(40);
        for i in 0..40 {
            let trend_val = 10.0 + 0.5 * i as f64;
            let seasonal = match i % 4 {
                0 => 2.0,
                1 => -1.0,
                2 => 3.0,
                3 => -4.0,
                _ => 0.0,
            };
            data.push(trend_val + seasonal);
        }
        let data = Array1::from_vec(data);

        // Use Classical method which handles trend extraction well for short series
        let (trend, _seasonal, _residual) =
            seasonal_decompose_with_method(&data, 4, "additive", DecomposeMethod::Classical)
                .expect("Decomposition failed");

        // Find non-NaN values for comparison
        let valid_first: Vec<f64> = trend
            .iter()
            .take(15)
            .copied()
            .filter(|x| !x.is_nan())
            .collect();
        let valid_last: Vec<f64> = trend
            .iter()
            .skip(25)
            .copied()
            .filter(|x| !x.is_nan())
            .collect();

        if !valid_first.is_empty() && !valid_last.is_empty() {
            let first_mean = valid_first.iter().sum::<f64>() / valid_first.len() as f64;
            let last_mean = valid_last.iter().sum::<f64>() / valid_last.len() as f64;
            assert!(
                last_mean > first_mean,
                "Trend should be increasing: first={}, last={}",
                first_mean,
                last_mean
            );
        }
    }

    #[test]
    fn test_compute_window_sizes() {
        // Trend window should be odd
        let tw = compute_trend_window(4);
        assert!(tw % 2 == 1, "Trend window must be odd");
        assert!(tw >= 7, "Trend window should be at least 7");

        // Seasonal window should be odd
        let sw = compute_seasonal_window(4);
        assert!(sw % 2 == 1, "Seasonal window must be odd");
        assert!(sw >= 5, "Seasonal window should be at least 5");
    }

    #[test]
    fn test_multiplicative_negative_data_fails() {
        let data = array![10.0, 15.0, -5.0, 8.0, 11.0, 16.0, 13.0, 9.0, 12.0, 17.0, 14.0, 10.0];
        let result = seasonal_decompose(&data, 4, "multiplicative");
        assert!(
            result.is_err(),
            "Multiplicative with negative data should fail"
        );
    }
}
