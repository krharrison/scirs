//! STL (Seasonal-Trend decomposition using LOESS) and MSTL implementation
//!
//! This is a proper implementation following Cleveland et al. (1990):
//! "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
//!
//! The algorithm decomposes a time series Y into three components:
//! Y_t = T_t + S_t + R_t
//! where T_t is the trend, S_t is the seasonal component, and R_t is the residual.

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use super::common::DecompositionResult;
use crate::error::{Result, TimeSeriesError};
use crate::loess::{loess_smooth, LoessConfig};

/// Options for STL decomposition
#[derive(Debug, Clone)]
pub struct STLOptions {
    /// Trend window size (must be odd)
    pub trend_window: usize,
    /// Seasonal window size (must be odd)
    pub seasonal_window: usize,
    /// Number of inner loop iterations
    pub n_inner: usize,
    /// Number of outer loop iterations
    pub n_outer: usize,
    /// Whether to use robust weighting
    pub robust: bool,
}

impl Default for STLOptions {
    fn default() -> Self {
        Self {
            trend_window: 21,
            seasonal_window: 13,
            n_inner: 2,
            n_outer: 1,
            robust: false,
        }
    }
}

/// Options for Multiple Seasonal-Trend decomposition using LOESS (MSTL)
#[derive(Debug, Clone)]
pub struct MSTLOptions {
    /// Seasonal periods (e.g., [7, 30, 365] for weekly, monthly, and yearly seasonality)
    pub seasonal_periods: Vec<usize>,
    /// Trend window size (must be odd)
    pub trend_window: usize,
    /// Seasonal window sizes for each seasonal period (must be odd)
    pub seasonal_windows: Option<Vec<usize>>,
    /// Number of inner loop iterations
    pub n_inner: usize,
    /// Number of outer loop iterations
    pub n_outer: usize,
    /// Whether to use robust weighting
    pub robust: bool,
}

impl Default for MSTLOptions {
    fn default() -> Self {
        Self {
            seasonal_periods: Vec::new(),
            trend_window: 21,
            seasonal_windows: None,
            n_inner: 2,
            n_outer: 1,
            robust: false,
        }
    }
}

/// Result of multiple seasonal time series decomposition
#[derive(Debug, Clone)]
pub struct MultiSeasonalDecompositionResult<F> {
    /// Trend component
    pub trend: Array1<F>,
    /// Multiple seasonal components
    pub seasonal_components: Vec<Array1<F>>,
    /// Residual component
    pub residual: Array1<F>,
    /// Original time series
    pub original: Array1<F>,
}

/// Performs STL (Seasonal and Trend decomposition using LOESS) on a time series
///
/// STL decomposition uses locally weighted regression (LOESS) to extract trend
/// and seasonal components. This implementation follows the original algorithm
/// by Cleveland et al. (1990).
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `period` - The seasonal period
/// * `options` - Options for STL decomposition
///
/// # Returns
///
/// * Decomposition result containing trend, seasonal, and residual components
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::decomposition::{stl_decomposition, STLOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let options = STLOptions::default();
/// let result = stl_decomposition(&ts, 4, &options).expect("Operation failed");
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal: {:?}", result.seasonal);
/// println!("Residual: {:?}", result.residual);
/// ```
#[allow(dead_code)]
pub fn stl_decomposition<F>(
    ts: &Array1<F>,
    period: usize,
    options: &STLOptions,
) -> Result<DecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    if ts.len() < 2 * period {
        return Err(TimeSeriesError::DecompositionError(format!(
            "Time series length ({}) must be at least twice the seasonal period ({})",
            ts.len(),
            period
        )));
    }

    // Validate options
    if options.trend_window % 2 == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "trend_window".to_string(),
            message: "Trend window size must be odd".to_string(),
        });
    }
    if options.seasonal_window % 2 == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "seasonal_window".to_string(),
            message: "Seasonal window size must be odd".to_string(),
        });
    }

    let n = ts.len();
    let mut seasonal = Array1::zeros(n);
    let mut trend = Array1::zeros(n);
    let mut weights = Array1::from_elem(n, F::one());
    let original = ts.clone();

    // LOESS configs for seasonal and trend smoothing
    let seasonal_loess = LoessConfig {
        span: options.seasonal_window as f64,
        degree: 1,
        robustness_iters: 0,
    };

    let trend_loess = LoessConfig {
        span: options.trend_window as f64,
        degree: 1,
        robustness_iters: 0,
    };

    // Low-pass filter config (applied to the seasonal component)
    // Uses 3-pass moving average of lengths period, period, 3
    let low_pass_loess = LoessConfig {
        span: (period as f64).max(3.0),
        degree: 1,
        robustness_iters: 0,
    };

    // STL Outer Loop
    for _outer in 0..options.n_outer {
        // STL Inner Loop
        for _inner in 0..options.n_inner {
            // Step 1: Detrending
            let detrended = if trend.iter().all(|&x| x == F::zero()) {
                original.clone()
            } else {
                &original - &trend
            };

            // Step 2: Cycle-subseries smoothing
            let mut cycle_subseries = vec![Vec::new(); period];
            let mut smoothed_seasonal = Array1::zeros(n);

            // Group data by seasonal position
            for i in 0..n {
                cycle_subseries[i % period].push((i, detrended[i], weights[i]));
            }

            // Apply LOESS to each subseries
            for subseries in cycle_subseries.iter() {
                if subseries.len() < 3 {
                    // Too few points for LOESS, use weighted average
                    let mut sum_w = F::zero();
                    let mut sum_wy = F::zero();
                    for &(_, val, w) in subseries {
                        sum_w = sum_w + w;
                        sum_wy = sum_wy + w * val;
                    }
                    let avg = if sum_w > F::zero() {
                        sum_wy / sum_w
                    } else {
                        F::zero()
                    };
                    for &(idx, _, _) in subseries {
                        smoothed_seasonal[idx] = avg;
                    }
                    continue;
                }

                let values: Vec<F> = subseries.iter().map(|&(_, v, _)| v).collect();
                let sub_weights: Vec<F> = subseries.iter().map(|&(_, _, w)| w).collect();
                let indices: Vec<usize> = subseries.iter().map(|&(i, _, _)| i).collect();

                let y_arr = Array1::from_vec(values);
                let w_arr = Array1::from_vec(sub_weights);

                match loess_smooth(&y_arr, &seasonal_loess, Some(&w_arr)) {
                    Ok(smoothed) => {
                        for (k, &idx) in indices.iter().enumerate() {
                            smoothed_seasonal[idx] = smoothed[k];
                        }
                    }
                    Err(_) => {
                        // Fallback: use original values
                        for &(idx, val, _) in subseries {
                            smoothed_seasonal[idx] = val;
                        }
                    }
                }
            }

            // Step 3: Low-pass filter on the raw seasonal component
            // Cleveland et al. use a 3-pass moving average (period, period, 3) followed by LOESS
            let filtered_seasonal =
                match apply_low_pass_filter(&smoothed_seasonal, period, &low_pass_loess) {
                    Ok(filtered) => {
                        // Subtract the low-pass filtered version from the raw seasonal
                        &smoothed_seasonal - &filtered
                    }
                    Err(_) => {
                        // Fallback: center the seasonal to sum to approximately zero
                        let mean = smoothed_seasonal.iter().fold(F::zero(), |acc, &x| acc + x)
                            / F::from(n).unwrap_or(F::one());
                        smoothed_seasonal.mapv(|x| x - mean)
                    }
                };

            // Step 4: Deseasonalize
            let deseasonalized = &original - &filtered_seasonal;

            // Step 5: Trend smoothing using LOESS
            let new_trend = match loess_smooth(&deseasonalized, &trend_loess, Some(&weights)) {
                Ok(t) => t,
                Err(_) => {
                    // Fallback: simple weighted moving average
                    weighted_moving_average(&deseasonalized, &weights, options.trend_window)?
                }
            };

            // Step 6: Update components
            trend = new_trend;
            seasonal = filtered_seasonal;
        }

        // Update robustness weights (bisquare weights on residuals)
        if options.robust {
            let residual = &original - &trend - &seasonal;
            let abs_residuals = residual.mapv(|x| x.abs());

            // Compute median absolute residual
            let mut sorted_abs: Vec<F> = abs_residuals.to_vec();
            sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let median_abs = if n % 2 == 0 {
                (sorted_abs[n / 2 - 1] + sorted_abs[n / 2]) / F::from(2.0).unwrap_or(F::one())
            } else {
                sorted_abs[n / 2]
            };

            let h = F::from(6.0).unwrap_or(F::one()) * median_abs;

            if h > F::zero() {
                for i in 0..n {
                    let u = abs_residuals[i] / h;
                    if u >= F::one() {
                        weights[i] = F::zero();
                    } else {
                        let one_minus_u2 = F::one() - u * u;
                        weights[i] = one_minus_u2 * one_minus_u2; // Bisquare
                    }
                }
            }
        }
    }

    // Calculate final residual
    let residual = &original - &trend - &seasonal;

    Ok(DecompositionResult {
        trend,
        seasonal,
        residual,
        original,
    })
}

/// Apply the STL low-pass filter: 3-pass moving average (period, period, 3) then LOESS
fn apply_low_pass_filter<F>(
    data: &Array1<F>,
    period: usize,
    loess_config: &LoessConfig,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = data.len();
    if n < period + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Insufficient data for low-pass filter".to_string(),
            required: period + 2,
            actual: n,
        });
    }

    // First pass: moving average of length period
    let ma1 = moving_average_centered(data, period)?;

    // Second pass: moving average of length period
    let ma2 = if ma1.len() >= period {
        moving_average_centered(&ma1, period)?
    } else {
        ma1.clone()
    };

    // Third pass: moving average of length 3
    let ma3 = if ma2.len() >= 3 {
        moving_average_centered(&ma2, 3)?
    } else {
        ma2.clone()
    };

    // Apply LOESS to the result
    if ma3.len() >= 3 {
        loess_smooth(&ma3, loess_config, None)
    } else {
        Ok(ma3)
    }
}

/// Centered moving average preserving array length
fn moving_average_centered<F>(data: &Array1<F>, window: usize) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = data.len();
    if n < window {
        return Ok(data.clone());
    }

    let half = window / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let actual_window = end - start;

        let mut sum = F::zero();
        for j in start..end {
            sum = sum + data[j];
        }
        result[i] = sum
            / F::from(actual_window).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Conversion failed".to_string())
            })?;
    }

    Ok(result)
}

/// Weighted moving average preserving array length
fn weighted_moving_average<F>(
    data: &Array1<F>,
    weights: &Array1<F>,
    window: usize,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = data.len();
    let half = window / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);

        let mut sum_w = F::zero();
        let mut sum_wy = F::zero();
        for j in start..end {
            sum_w = sum_w + weights[j];
            sum_wy = sum_wy + weights[j] * data[j];
        }

        result[i] = if sum_w > F::zero() {
            sum_wy / sum_w
        } else {
            data[i]
        };
    }

    Ok(result)
}

/// Performs Multiple STL decomposition on a time series with multiple seasonal components
///
/// MSTL (Multiple Seasonal-Trend decomposition using LOESS) extends STL to handle
/// multiple seasonal periods.
///
/// # Arguments
///
/// * `ts` - The time series to decompose
/// * `options` - Options for MSTL decomposition
///
/// # Returns
///
/// * MultiSeasonalDecompositionResult containing trend, multiple seasonal components, and residual
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::decomposition::{mstl_decomposition, MSTLOptions};
///
/// let ts = array![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
///                 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5, 1.5, 2.5, 3.5, 2.5];
///
/// let mut options = MSTLOptions::default();
/// options.seasonal_periods = vec![4, 12]; // Weekly and monthly patterns
///
/// let result = mstl_decomposition(&ts, &options).expect("Operation failed");
/// println!("Trend: {:?}", result.trend);
/// println!("Seasonal Components: {}", result.seasonal_components.len());
/// println!("Residual: {:?}", result.residual);
/// ```
#[allow(dead_code)]
pub fn mstl_decomposition<F>(
    ts: &Array1<F>,
    options: &MSTLOptions,
) -> Result<MultiSeasonalDecompositionResult<F>>
where
    F: Float + FromPrimitive + Debug,
{
    // Validate options
    if options.seasonal_periods.is_empty() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "seasonal_periods".to_string(),
            message: "At least one seasonal period must be specified".to_string(),
        });
    }

    let n_seasons = options.seasonal_periods.len();
    if let Some(ref windows) = options.seasonal_windows {
        if windows.len() != n_seasons {
            return Err(TimeSeriesError::InvalidParameter {
                name: "seasonal_windows".to_string(),
                message: format!(
                    "Number of seasonal windows ({}) must match number of seasonal periods ({})",
                    windows.len(),
                    n_seasons
                ),
            });
        }
    }

    // For each seasonal period, check that the time series is long enough
    for &period in &options.seasonal_periods {
        if ts.len() < 2 * period {
            return Err(TimeSeriesError::DecompositionError(format!(
                "Time series length ({}) must be at least twice the seasonal period ({})",
                ts.len(),
                period
            )));
        }
    }

    let n = ts.len();
    let original = ts.clone();
    let mut seasonal_components = Vec::with_capacity(n_seasons);
    let _weights = Array1::from_elem(n, F::one());
    let mut deseasonal = original.clone();

    // For each seasonal component
    for (i, &period) in options.seasonal_periods.iter().enumerate() {
        let seasonal_window = if let Some(ref windows) = options.seasonal_windows {
            windows[i]
        } else {
            // Default to 7 or (period / 2), whichever is larger
            std::cmp::max(7, period / 2) | 1 // Ensure odd
        };

        let stl_options = STLOptions {
            trend_window: options.trend_window,
            seasonal_window,
            n_inner: options.n_inner,
            n_outer: options.n_outer,
            robust: options.robust,
        };

        // Apply STL with current data
        let result = stl_decomposition(&deseasonal, period, &stl_options)?;

        // Save this seasonal component
        seasonal_components.push(result.seasonal);

        // Remove this seasonal component
        deseasonal = deseasonal - &seasonal_components[i];
    }

    // The remaining deseasonal series is the trend
    let trend = deseasonal.clone();

    // Calculate final residual
    let mut residual = original.clone();
    residual = residual - &trend;
    for seasonal in &seasonal_components {
        residual = residual - seasonal;
    }

    Ok(MultiSeasonalDecompositionResult {
        trend,
        seasonal_components,
        residual,
        original,
    })
}
