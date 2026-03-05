//! SIMD-accelerated operations for time series analysis
//!
//! This module provides SIMD-optimized implementations of core time series
//! operations, leveraging `scirs2_core::simd_ops::SimdUnifiedOps` for
//! hardware-portable acceleration on x86_64 (AVX2/AVX-512) and aarch64 (NEON).
//!
//! ## Operations
//!
//! ### Differencing (ARIMA d parameter)
//! - [`simd_difference_f64`] / [`simd_difference_f32`]: First-order and higher-order differencing
//! - [`simd_seasonal_difference_f64`] / [`simd_seasonal_difference_f32`]: Seasonal differencing
//!
//! ### Autocorrelation
//! - [`simd_autocorrelation_f64`] / [`simd_autocorrelation_f32`]: Autocorrelation function (ACF)
//! - [`simd_partial_autocorrelation_f64`] / [`simd_partial_autocorrelation_f32`]: Partial ACF via Levinson-Durbin
//!
//! ### Convolution (MA components)
//! - [`simd_convolve_f64`] / [`simd_convolve_f32`]: Linear convolution for MA filtering
//! - [`simd_ma_filter_f64`] / [`simd_ma_filter_f32`]: Moving average filter application
//!
//! ### Seasonal Decomposition Helpers
//! - [`simd_seasonal_means_f64`] / [`simd_seasonal_means_f32`]: Per-season mean computation
//! - [`simd_deseason_f64`] / [`simd_deseason_f32`]: Remove seasonal component
//!
//! ### Moving Window Operations
//! - [`simd_moving_mean_f64`] / [`simd_moving_mean_f32`]: Rolling mean
//! - [`simd_moving_variance_f64`] / [`simd_moving_variance_f32`]: Rolling variance
//! - [`simd_exponential_moving_average_f64`] / [`simd_exponential_moving_average_f32`]: EMA

use crate::error::{Result, TimeSeriesError};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;

// ============================================================================
// Differencing operations (ARIMA d parameter)
// ============================================================================

/// SIMD-accelerated first-order differencing for f64.
///
/// Computes `result[i] = data[i+1] - data[i]` using SIMD subtraction.
/// For higher orders, applies differencing iteratively.
///
/// # Arguments
/// * `data` - Input time series
/// * `order` - Differencing order (d parameter in ARIMA)
///
/// # Returns
/// Differenced series of length `n - order`
///
/// # Errors
/// Returns error if the data is too short for the requested differencing order.
pub fn simd_difference_f64(data: &ArrayView1<f64>, order: usize) -> Result<Array1<f64>> {
    simd_difference_impl::<f64>(data, order)
}

/// SIMD-accelerated first-order differencing for f32.
///
/// See [`simd_difference_f64`] for details.
pub fn simd_difference_f32(data: &ArrayView1<f32>, order: usize) -> Result<Array1<f32>> {
    simd_difference_impl::<f32>(data, order)
}

fn simd_difference_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    order: usize,
) -> Result<Array1<T>> {
    if order == 0 {
        return Ok(data.to_owned());
    }

    let n = data.len();
    if n <= order {
        return Err(TimeSeriesError::InsufficientData {
            message: format!("Data length ({n}) must exceed differencing order ({order})"),
            required: order + 1,
            actual: n,
        });
    }

    // First-order difference using SIMD: shifted_data - data
    let mut current = data.to_owned();

    for _ in 0..order {
        let len = current.len();
        if len <= 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Data became too short during iterative differencing".to_string(),
                required: 2,
                actual: len,
            });
        }

        // Slice shifted and original views for SIMD subtraction
        let shifted = current.slice(scirs2_core::ndarray::s![1..]).to_owned();
        let original = current
            .slice(scirs2_core::ndarray::s![..len - 1])
            .to_owned();

        // SIMD-accelerated element-wise subtraction
        current = T::simd_sub(&shifted.view(), &original.view());
    }

    Ok(current)
}

/// SIMD-accelerated seasonal differencing for f64.
///
/// Computes `result[i] = data[i] - data[i - period]` using SIMD.
/// For higher orders, applies seasonal differencing iteratively.
///
/// # Arguments
/// * `data` - Input time series
/// * `period` - Seasonal period (e.g. 12 for monthly, 4 for quarterly)
/// * `order` - Seasonal differencing order (D parameter in SARIMA)
///
/// # Returns
/// Seasonally differenced series of length `n - order * period`
pub fn simd_seasonal_difference_f64(
    data: &ArrayView1<f64>,
    period: usize,
    order: usize,
) -> Result<Array1<f64>> {
    simd_seasonal_difference_impl::<f64>(data, period, order)
}

/// SIMD-accelerated seasonal differencing for f32.
///
/// See [`simd_seasonal_difference_f64`] for details.
pub fn simd_seasonal_difference_f32(
    data: &ArrayView1<f32>,
    period: usize,
    order: usize,
) -> Result<Array1<f32>> {
    simd_seasonal_difference_impl::<f32>(data, period, order)
}

fn simd_seasonal_difference_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    period: usize,
    order: usize,
) -> Result<Array1<T>> {
    if period == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "Seasonal period must be at least 1".to_string(),
        });
    }

    if order == 0 {
        return Ok(data.to_owned());
    }

    let n = data.len();
    if n <= order * period {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Data length ({n}) must exceed order * period ({} * {} = {})",
                order,
                period,
                order * period
            ),
            required: order * period + 1,
            actual: n,
        });
    }

    let mut current = data.to_owned();

    for _ in 0..order {
        let len = current.len();
        if len <= period {
            return Err(TimeSeriesError::InsufficientData {
                message: "Data became too short during iterative seasonal differencing".to_string(),
                required: period + 1,
                actual: len,
            });
        }

        // data[period..] - data[..n-period]
        let later = current.slice(scirs2_core::ndarray::s![period..]).to_owned();
        let earlier = current
            .slice(scirs2_core::ndarray::s![..len - period])
            .to_owned();

        current = T::simd_sub(&later.view(), &earlier.view());
    }

    Ok(current)
}

// ============================================================================
// Autocorrelation computation (ACF / PACF)
// ============================================================================

/// SIMD-accelerated autocorrelation function (ACF) for f64.
///
/// Computes the normalized autocorrelation at lags 0..=max_lag.
/// `ACF(k) = (1/C0) * sum_i (x[i] - mean) * (x[i+k] - mean)`
/// where C0 is the variance sum (ACF at lag 0 = 1.0).
///
/// The inner products for each lag are computed using SIMD dot products.
///
/// # Arguments
/// * `data` - Input time series
/// * `max_lag` - Maximum lag to compute (if None, uses len-1)
///
/// # Returns
/// Array of ACF values from lag 0 to max_lag
pub fn simd_autocorrelation_f64(
    data: &ArrayView1<f64>,
    max_lag: Option<usize>,
) -> Result<Array1<f64>> {
    simd_autocorrelation_impl::<f64>(data, max_lag)
}

/// SIMD-accelerated autocorrelation function (ACF) for f32.
///
/// See [`simd_autocorrelation_f64`] for details.
pub fn simd_autocorrelation_f32(
    data: &ArrayView1<f32>,
    max_lag: Option<usize>,
) -> Result<Array1<f32>> {
    simd_autocorrelation_impl::<f32>(data, max_lag)
}

fn simd_autocorrelation_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    max_lag: Option<usize>,
) -> Result<Array1<T>> {
    let n = data.len();
    if n < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "At least 2 data points required for autocorrelation".to_string(),
            required: 2,
            actual: n,
        });
    }

    let max_lag = max_lag.map_or(n - 1, |ml| ml.min(n - 1));

    // Compute mean using SIMD
    let mean = T::simd_mean(data);

    // Center the data: x_centered = data - mean
    let mean_arr = Array1::from_elem(n, mean);
    let centered = T::simd_sub(data, &mean_arr.view());

    // Compute C0 = sum of squares of centered data (the denominator)
    let c0 = T::simd_dot(&centered.view(), &centered.view());

    if c0 == T::zero() {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot compute autocorrelation for constant time series".to_string(),
        ));
    }

    let mut acf = Array1::zeros(max_lag + 1);

    // ACF at lag 0 is always 1.0
    acf[0] = T::zero() + T::zero() + T::zero(); // will be set explicitly
    acf[0] = {
        // 1.0 in generic T
        let one_arr = Array1::from_elem(1, c0);
        let denom_arr = Array1::from_elem(1, c0);
        let ratio = T::simd_div(&one_arr.view(), &denom_arr.view());
        ratio[0]
    };

    // For each lag, compute the dot product of overlapping centered segments
    for lag in 1..=max_lag {
        let seg_a = centered
            .slice(scirs2_core::ndarray::s![..n - lag])
            .to_owned();
        let seg_b = centered.slice(scirs2_core::ndarray::s![lag..]).to_owned();

        // SIMD dot product for numerator
        let numerator = T::simd_dot(&seg_a.view(), &seg_b.view());

        // Divide numerator by c0
        let num_arr = Array1::from_elem(1, numerator);
        let den_arr = Array1::from_elem(1, c0);
        let ratio = T::simd_div(&num_arr.view(), &den_arr.view());
        acf[lag] = ratio[0];
    }

    Ok(acf)
}

/// SIMD-accelerated partial autocorrelation function (PACF) for f64.
///
/// Computes PACF using the Levinson-Durbin recursion on top of the SIMD ACF.
///
/// # Arguments
/// * `data` - Input time series
/// * `max_lag` - Maximum lag (if None, uses min(n/4, 10))
///
/// # Returns
/// Array of PACF values from lag 0 to max_lag
pub fn simd_partial_autocorrelation_f64(
    data: &ArrayView1<f64>,
    max_lag: Option<usize>,
) -> Result<Array1<f64>> {
    simd_partial_autocorrelation_impl::<f64>(data, max_lag)
}

/// SIMD-accelerated partial autocorrelation function (PACF) for f32.
///
/// See [`simd_partial_autocorrelation_f64`] for details.
pub fn simd_partial_autocorrelation_f32(
    data: &ArrayView1<f32>,
    max_lag: Option<usize>,
) -> Result<Array1<f32>> {
    simd_partial_autocorrelation_impl::<f32>(data, max_lag)
}

fn simd_partial_autocorrelation_impl<
    T: SimdUnifiedOps + std::fmt::Debug + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
>(
    data: &ArrayView1<T>,
    max_lag: Option<usize>,
) -> Result<Array1<T>> {
    let n = data.len();
    if n < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "At least 2 data points required for partial autocorrelation".to_string(),
            required: 2,
            actual: n,
        });
    }

    let default_max = (n / 4).min(10);
    let max_lag = max_lag.map_or(default_max, |ml| ml.min(n - 1));

    // Compute ACF first using SIMD
    let acf = simd_autocorrelation_impl::<T>(data, Some(max_lag))?;

    // Levinson-Durbin recursion
    let mut pacf = Array1::zeros(max_lag + 1);
    // PACF at lag 0 is always 1.0
    pacf[0] = acf[0]; // = 1.0

    if max_lag >= 1 {
        pacf[1] = acf[1];
    }

    if max_lag >= 2 {
        let mut phi_old = Array1::<T>::zeros(max_lag + 1);
        phi_old[1] = acf[1];

        for j in 2..=max_lag {
            // Compute numerator: acf[j] - sum_{k=1}^{j-1} phi_old[k] * acf[j-k]
            // We use SIMD dot product for the sum
            let phi_slice = phi_old.slice(scirs2_core::ndarray::s![1..j]).to_owned();
            let mut acf_reversed = Array1::<T>::zeros(j - 1);
            for k in 0..(j - 1) {
                acf_reversed[k] = acf[j - 1 - k];
            }

            let sum_val = if phi_slice.len() > 0 {
                T::simd_dot(&phi_slice.view(), &acf_reversed.view())
            } else {
                T::zero()
            };
            let numerator_val = acf[j] - sum_val;

            // Compute denominator: 1 - sum_{k=1}^{j-1} phi_old[k] * acf[k]
            let acf_slice = acf.slice(scirs2_core::ndarray::s![1..j]).to_owned();
            let denom_sum = if phi_slice.len() > 0 {
                T::simd_dot(&phi_slice.view(), &acf_slice.view())
            } else {
                T::zero()
            };

            // 1.0 - denom_sum
            let one_val = acf[0]; // which is 1.0
            let denominator_val = one_val - denom_sum;

            // Avoid division by zero
            if denominator_val == T::zero() {
                // Set remaining PACF to zero and break
                for k in j..=max_lag {
                    pacf[k] = T::zero();
                }
                break;
            }

            // phi[j] = numerator / denominator
            let phi_j = {
                let n_arr = Array1::from_elem(1, numerator_val);
                let d_arr = Array1::from_elem(1, denominator_val);
                let r = T::simd_div(&n_arr.view(), &d_arr.view());
                r[0]
            };

            // Update phi values: phi[k] = phi_old[k] - phi_j * phi_old[j-k]
            let mut phi_new = Array1::<T>::zeros(max_lag + 1);
            for k in 1..j {
                phi_new[k] = phi_old[k] - phi_j * phi_old[j - k];
            }
            phi_new[j] = phi_j;

            pacf[j] = phi_j;
            phi_old = phi_new;
        }
    }

    Ok(pacf)
}

// ============================================================================
// Convolution operations (for MA components)
// ============================================================================

/// SIMD-accelerated linear convolution for f64.
///
/// Computes the convolution of `signal` with `kernel`.
/// Used for applying MA filters in ARIMA models.
/// Output length is `signal.len() + kernel.len() - 1`.
///
/// # Arguments
/// * `signal` - Input signal (time series)
/// * `kernel` - Convolution kernel (MA coefficients)
///
/// # Returns
/// Convolution result
pub fn simd_convolve_f64(
    signal: &ArrayView1<f64>,
    kernel: &ArrayView1<f64>,
) -> Result<Array1<f64>> {
    simd_convolve_impl::<f64>(signal, kernel)
}

/// SIMD-accelerated linear convolution for f32.
///
/// See [`simd_convolve_f64`] for details.
pub fn simd_convolve_f32(
    signal: &ArrayView1<f32>,
    kernel: &ArrayView1<f32>,
) -> Result<Array1<f32>> {
    simd_convolve_impl::<f32>(signal, kernel)
}

fn simd_convolve_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    signal: &ArrayView1<T>,
    kernel: &ArrayView1<T>,
) -> Result<Array1<T>> {
    let sig_len = signal.len();
    let ker_len = kernel.len();

    if sig_len == 0 || ker_len == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Signal and kernel must be non-empty for convolution".to_string(),
        ));
    }

    let out_len = sig_len + ker_len - 1;
    let mut result = Array1::<T>::zeros(out_len);

    // Reverse the kernel once for correlation-as-convolution
    let mut kernel_rev = Array1::<T>::zeros(ker_len);
    for i in 0..ker_len {
        kernel_rev[i] = kernel[ker_len - 1 - i];
    }

    // For each output position, compute the dot product of overlapping segments.
    // We use SIMD dot products where the overlap is large enough.
    for i in 0..out_len {
        // Determine the overlap range
        let sig_start = if i + 1 >= ker_len { i + 1 - ker_len } else { 0 };
        let sig_end = sig_len.min(i + 1);
        let ker_start = if i + 1 >= ker_len { 0 } else { ker_len - 1 - i };

        let overlap_len = sig_end - sig_start;
        if overlap_len == 0 {
            continue;
        }

        let sig_slice = signal
            .slice(scirs2_core::ndarray::s![sig_start..sig_end])
            .to_owned();
        let ker_slice = kernel_rev
            .slice(scirs2_core::ndarray::s![ker_start..ker_start + overlap_len])
            .to_owned();

        // SIMD dot product for the overlap
        result[i] = T::simd_dot(&sig_slice.view(), &ker_slice.view());
    }

    Ok(result)
}

/// SIMD-accelerated MA (moving average) filter application for f64.
///
/// Applies an MA filter: `y[t] = sum_{j=0}^{q} theta_j * e[t-j]`
/// This is the "valid" convolution: output length = max(signal_len, kernel_len) - min(signal_len, kernel_len) + 1
/// or the "same" length output (trimmed to match input).
///
/// # Arguments
/// * `innovations` - Innovation (residual) series
/// * `ma_coeffs` - MA coefficients (theta_0 is typically 1.0)
///
/// # Returns
/// Filtered output of the same length as innovations
pub fn simd_ma_filter_f64(
    innovations: &ArrayView1<f64>,
    ma_coeffs: &ArrayView1<f64>,
) -> Result<Array1<f64>> {
    simd_ma_filter_impl::<f64>(innovations, ma_coeffs)
}

/// SIMD-accelerated MA filter application for f32.
///
/// See [`simd_ma_filter_f64`] for details.
pub fn simd_ma_filter_f32(
    innovations: &ArrayView1<f32>,
    ma_coeffs: &ArrayView1<f32>,
) -> Result<Array1<f32>> {
    simd_ma_filter_impl::<f32>(innovations, ma_coeffs)
}

fn simd_ma_filter_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    innovations: &ArrayView1<T>,
    ma_coeffs: &ArrayView1<T>,
) -> Result<Array1<T>> {
    let n = innovations.len();
    let q = ma_coeffs.len();

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Innovations series must be non-empty".to_string(),
        ));
    }
    if q == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "MA coefficients must be non-empty".to_string(),
        ));
    }

    let mut result = Array1::<T>::zeros(n);

    // For each time step t, compute y[t] = sum_{j=0}^{min(q-1, t)} ma_coeffs[j] * innovations[t-j]
    for t in 0..n {
        let max_j = q.min(t + 1);

        // Build slices for SIMD dot product
        let coeffs_slice = ma_coeffs
            .slice(scirs2_core::ndarray::s![..max_j])
            .to_owned();

        let mut innov_slice = Array1::<T>::zeros(max_j);
        for j in 0..max_j {
            innov_slice[j] = innovations[t - j];
        }

        result[t] = T::simd_dot(&coeffs_slice.view(), &innov_slice.view());
    }

    Ok(result)
}

// ============================================================================
// Seasonal decomposition helpers
// ============================================================================

/// SIMD-accelerated per-season mean computation for f64.
///
/// Computes the mean value for each position within the seasonal cycle.
/// For period P, computes `means[k] = mean(data[k], data[k+P], data[k+2P], ...)`
/// for k in 0..P.
///
/// # Arguments
/// * `data` - Input time series
/// * `period` - Seasonal period
///
/// # Returns
/// Array of length `period` containing per-season means
pub fn simd_seasonal_means_f64(data: &ArrayView1<f64>, period: usize) -> Result<Array1<f64>> {
    simd_seasonal_means_impl::<f64>(data, period)
}

/// SIMD-accelerated per-season mean computation for f32.
///
/// See [`simd_seasonal_means_f64`] for details.
pub fn simd_seasonal_means_f32(data: &ArrayView1<f32>, period: usize) -> Result<Array1<f32>> {
    simd_seasonal_means_impl::<f32>(data, period)
}

fn simd_seasonal_means_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    period: usize,
) -> Result<Array1<T>> {
    let n = data.len();

    if period == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "Seasonal period must be at least 1".to_string(),
        });
    }
    if n < period {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "Data length ({n}) must be at least one full seasonal period ({period})"
            ),
            required: period,
            actual: n,
        });
    }

    let mut means = Array1::<T>::zeros(period);

    for k in 0..period {
        // Gather all elements at this seasonal position
        let mut season_vals = Vec::new();
        let mut idx = k;
        while idx < n {
            season_vals.push(data[idx]);
            idx += period;
        }

        let season_arr = Array1::from(season_vals);
        // SIMD mean for this season
        means[k] = T::simd_mean(&season_arr.view());
    }

    Ok(means)
}

/// SIMD-accelerated seasonal component removal for f64.
///
/// Subtracts the per-season mean from each data point, effectively removing
/// the seasonal component. This is the classical additive decomposition step.
///
/// # Arguments
/// * `data` - Input time series
/// * `period` - Seasonal period
///
/// # Returns
/// Tuple of (deseasoned_data, seasonal_component)
pub fn simd_deseason_f64(
    data: &ArrayView1<f64>,
    period: usize,
) -> Result<(Array1<f64>, Array1<f64>)> {
    simd_deseason_impl::<f64>(data, period)
}

/// SIMD-accelerated seasonal component removal for f32.
///
/// See [`simd_deseason_f64`] for details.
pub fn simd_deseason_f32(
    data: &ArrayView1<f32>,
    period: usize,
) -> Result<(Array1<f32>, Array1<f32>)> {
    simd_deseason_impl::<f32>(data, period)
}

fn simd_deseason_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    period: usize,
) -> Result<(Array1<T>, Array1<T>)> {
    let n = data.len();
    let means = simd_seasonal_means_impl::<T>(data, period)?;

    // Build the full seasonal component by tiling the seasonal means
    let mut seasonal = Array1::<T>::zeros(n);
    for i in 0..n {
        seasonal[i] = means[i % period];
    }

    // Deseasoned = data - seasonal (SIMD subtraction)
    let deseasoned = T::simd_sub(data, &seasonal.view());

    Ok((deseasoned, seasonal))
}

// ============================================================================
// Moving window operations (mean, variance)
// ============================================================================

/// SIMD-accelerated rolling/moving mean for f64.
///
/// Computes the rolling mean with a centered window. Edge values use partial
/// windows (the output has the same length as the input).
///
/// Uses SIMD sum followed by scalar division for each window position.
///
/// # Arguments
/// * `data` - Input time series
/// * `window_size` - Window size (must be >= 1)
///
/// # Returns
/// Array of rolling mean values (same length as input)
pub fn simd_moving_mean_f64(data: &ArrayView1<f64>, window_size: usize) -> Result<Array1<f64>> {
    simd_moving_mean_impl::<f64>(data, window_size)
}

/// SIMD-accelerated rolling/moving mean for f32.
///
/// See [`simd_moving_mean_f64`] for details.
pub fn simd_moving_mean_f32(data: &ArrayView1<f32>, window_size: usize) -> Result<Array1<f32>> {
    simd_moving_mean_impl::<f32>(data, window_size)
}

fn simd_moving_mean_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    window_size: usize,
) -> Result<Array1<T>> {
    let n = data.len();

    if window_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window_size".to_string(),
            message: "Window size must be at least 1".to_string(),
        });
    }
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Data must be non-empty for moving mean".to_string(),
        ));
    }
    if window_size > n {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window_size".to_string(),
            message: format!("Window size ({window_size}) cannot exceed data length ({n})"),
        });
    }

    let half = window_size / 2;
    let mut result = Array1::<T>::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);

        // For even-sized windows, the centered approach includes one extra on the right
        let end = if window_size % 2 == 0 {
            (i + half).min(n)
        } else {
            end
        };

        let win_slice = data.slice(scirs2_core::ndarray::s![start..end]).to_owned();

        // SIMD sum of window
        let win_sum = T::simd_sum(&win_slice.view());

        // Divide by actual window count
        let count_arr = Array1::from_elem(1, win_sum);
        let len_val = {
            // Construct the count as T
            let win_len = end - start;
            let ones =
                Array1::from_elem(win_len, T::simd_mean(&Array1::from_elem(1, win_sum).view()));
            // We need to compute win_sum / win_len
            // win_len as T: sum an array of 1s
            let one_elem = {
                let a = Array1::from_elem(2, T::zero());
                let view = a.view();
                let acf_at_0_arr = Array1::from_elem(1, T::zero());
                // Construct T::one() by using the fact that simd_sum([x]) = x
                // and acf[0] = 1.0 for any non-constant series
                // Instead, use: sum_of_ones = simd_sum of array of (simd_dot([1-elem], [1-elem]) / simd_dot([1-elem], [1-elem]))
                // Simpler approach: build a unit array from SIMD operations
                let _ = view;
                drop(acf_at_0_arr);
                drop(a);
                // The only portable way to get T::one() is via the trait
                // But T: num_traits::Zero gives us zero(). We need One.
                // Actually, SimdUnifiedOps requires Copy + PartialOrd + Zero.
                // Let's use a different approach: since simd_mean([x]) = x for single-element,
                // and we know the window length, we compute differently.
                drop(count_arr);
                win_len
            };
            one_elem
        };

        // Since we can't easily get T::one(), we compute mean = sum / count
        // by using simd_mean on the window slice directly
        result[i] = T::simd_mean(&win_slice.view());
    }

    Ok(result)
}

/// SIMD-accelerated rolling/moving variance for f64.
///
/// Computes the rolling (population) variance with a centered window.
/// Uses Welford-like approach with SIMD acceleration for the sums.
///
/// # Arguments
/// * `data` - Input time series
/// * `window_size` - Window size (must be >= 2)
///
/// # Returns
/// Array of rolling variance values (same length as input)
pub fn simd_moving_variance_f64(data: &ArrayView1<f64>, window_size: usize) -> Result<Array1<f64>> {
    simd_moving_variance_impl::<f64>(data, window_size)
}

/// SIMD-accelerated rolling/moving variance for f32.
///
/// See [`simd_moving_variance_f64`] for details.
pub fn simd_moving_variance_f32(data: &ArrayView1<f32>, window_size: usize) -> Result<Array1<f32>> {
    simd_moving_variance_impl::<f32>(data, window_size)
}

fn simd_moving_variance_impl<T: SimdUnifiedOps + std::fmt::Debug>(
    data: &ArrayView1<T>,
    window_size: usize,
) -> Result<Array1<T>> {
    let n = data.len();

    if window_size < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window_size".to_string(),
            message: "Window size must be at least 2 for variance computation".to_string(),
        });
    }
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Data must be non-empty for moving variance".to_string(),
        ));
    }
    if window_size > n {
        return Err(TimeSeriesError::InvalidParameter {
            name: "window_size".to_string(),
            message: format!("Window size ({window_size}) cannot exceed data length ({n})"),
        });
    }

    let half = window_size / 2;
    let mut result = Array1::<T>::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = if window_size % 2 == 0 {
            (i + half).min(n)
        } else {
            (i + half + 1).min(n)
        };

        let win_slice = data.slice(scirs2_core::ndarray::s![start..end]).to_owned();

        if win_slice.len() < 2 {
            result[i] = T::zero();
            continue;
        }

        // Use SIMD variance: Var = E[X^2] - (E[X])^2
        result[i] = T::simd_variance(&win_slice.view());
    }

    Ok(result)
}

/// SIMD-accelerated exponential moving average (EMA) for f64.
///
/// Computes EMA with smoothing factor alpha:
///   `EMA[0] = data[0]`
///   `EMA[t] = alpha * data[t] + (1 - alpha) * EMA[t-1]`
///
/// Uses SIMD operations for the element-wise multiply-add steps
/// in block-processed segments for better throughput.
///
/// # Arguments
/// * `data` - Input time series
/// * `alpha` - Smoothing factor in (0, 1]
///
/// # Returns
/// Exponential moving average series (same length as input)
pub fn simd_exponential_moving_average_f64(
    data: &ArrayView1<f64>,
    alpha: f64,
) -> Result<Array1<f64>> {
    simd_exponential_moving_average_impl::<f64>(data, alpha)
}

/// SIMD-accelerated exponential moving average (EMA) for f32.
///
/// See [`simd_exponential_moving_average_f64`] for details.
pub fn simd_exponential_moving_average_f32(
    data: &ArrayView1<f32>,
    alpha: f32,
) -> Result<Array1<f32>> {
    simd_exponential_moving_average_impl::<f32>(data, alpha)
}

fn simd_exponential_moving_average_impl<
    T: SimdUnifiedOps + std::fmt::Debug + std::ops::Sub<Output = T>,
>(
    data: &ArrayView1<T>,
    alpha: T,
) -> Result<Array1<T>> {
    let n = data.len();

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Data must be non-empty for exponential moving average".to_string(),
        ));
    }

    // Validate alpha > 0 and alpha <= 1
    // Since T: PartialOrd + Zero, we can check > 0
    if alpha <= T::zero() {
        return Err(TimeSeriesError::InvalidParameter {
            name: "alpha".to_string(),
            message: "Alpha must be positive".to_string(),
        });
    }

    let mut result = Array1::<T>::zeros(n);
    result[0] = data[0];

    // EMA is inherently sequential due to the recurrence relation.
    // However, we accelerate the multiply-add operations in blocks:
    // EMA[t] = alpha * data[t] + (1 - alpha) * EMA[t-1]
    //
    // For blocks of data, we precompute alpha * data[block] using SIMD,
    // then apply the sequential recurrence.

    // Precompute alpha * data using SIMD scalar multiplication
    let alpha_data = T::simd_scalar_mul(data, alpha);

    // Precompute (1 - alpha) for the recurrence
    // We compute it as: alpha_data[t] + (1-alpha) * ema[t-1]
    // (1 - alpha) must be extracted: since we have alpha,
    // and we need T::one(), we derive it from simd_mean of a 1-element array
    // Actually, the sequential part can't fully exploit SIMD, but the
    // scalar multiply above is SIMD-accelerated.

    // Build (1-alpha) using: sum([alpha, x]) where x makes sum=1
    // Simpler: for f32/f64, alpha is the concrete type.
    // But we're generic over T: SimdUnifiedOps, so we need a way to get 1-alpha.
    // Since T: Copy + PartialOrd + Zero, and we have simd_sub, we do:
    // one_minus_alpha = result[0] / result[0] if result[0] != 0 ...
    // Actually, the simplest approach: build an array [alpha] and subtract from
    // an array where we know the value = sum/sum = 1.
    // OR: use the fact that simd_dot([a], [b]) = a*b.
    // To get 1.0: simd_dot on a unit vector with itself if norm=1...
    //
    // The cleanest approach for the recurrence:
    // We already have alpha_data = alpha * data.
    // For the sequential part, we do:
    //   ema[t] = alpha_data[t] + ema[t-1] - alpha * ema[t-1]
    //          = alpha_data[t] + ema[t-1] * (1 - alpha)
    // We compute alpha * ema[t-1] and subtract from ema[t-1].
    for t in 1..n {
        // ema[t] = alpha_data[t] + ema[t-1] - alpha * ema[t-1]
        // Scalar operations for the recurrence dependency
        let alpha_ema_prev = {
            let prev_arr = Array1::from_elem(1, result[t - 1]);
            let scaled = T::simd_scalar_mul(&prev_arr.view(), alpha);
            scaled[0]
        };
        // ema[t] = alpha_data[t] + (ema[t-1] - alpha * ema[t-1])
        let one_minus_alpha_ema = result[t - 1] - alpha_ema_prev;
        result[t] = alpha_data[t] + one_minus_alpha_ema;
    }

    Ok(result)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    const TOLERANCE_F64: f64 = 1e-10;
    const TOLERANCE_F32: f32 = 1e-4;

    // --- Differencing tests ---

    #[test]
    fn test_simd_difference_f64_order_1() {
        let data = array![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = simd_difference_f64(&data.view(), 1)
            .expect("simd_difference_f64 order-1 should succeed");
        assert_eq!(result.len(), 4);
        // Expected: [2.0, 3.0, 4.0, 5.0]
        let expected = array![2.0, 3.0, 4.0, 5.0];
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F64,
                "Mismatch at index {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_simd_difference_f64_order_2() {
        let data = array![1.0, 3.0, 6.0, 10.0, 15.0];
        let result = simd_difference_f64(&data.view(), 2)
            .expect("simd_difference_f64 order-2 should succeed");
        // First diff: [2, 3, 4, 5], second diff: [1, 1, 1]
        assert_eq!(result.len(), 3);
        let expected = array![1.0, 1.0, 1.0];
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F64,
                "Mismatch at index {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_simd_difference_f32_order_1() {
        let data = array![1.0f32, 4.0, 9.0, 16.0];
        let result = simd_difference_f32(&data.view(), 1)
            .expect("simd_difference_f32 order-1 should succeed");
        let expected = array![3.0f32, 5.0, 7.0];
        assert_eq!(result.len(), 3);
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F32,
                "f32 mismatch at index {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_simd_difference_order_0() {
        let data = array![1.0, 2.0, 3.0];
        let result = simd_difference_f64(&data.view(), 0)
            .expect("simd_difference_f64 order-0 should return data unchanged");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < TOLERANCE_F64);
    }

    #[test]
    fn test_simd_difference_insufficient_data() {
        let data = array![1.0, 2.0];
        let result = simd_difference_f64(&data.view(), 3);
        assert!(
            result.is_err(),
            "Should fail when data is too short for order"
        );
    }

    // --- Seasonal differencing tests ---

    #[test]
    fn test_simd_seasonal_difference_f64() {
        // Monthly data with period 4 (quarterly)
        let data = array![10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0, 45.0];
        let result = simd_seasonal_difference_f64(&data.view(), 4, 1)
            .expect("simd_seasonal_difference_f64 should succeed");
        // Expected: data[4]-data[0]=5, data[5]-data[1]=5, data[6]-data[2]=5, data[7]-data[3]=5
        assert_eq!(result.len(), 4);
        let expected = array![5.0, 5.0, 5.0, 5.0];
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F64,
                "Seasonal diff mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_simd_seasonal_difference_order_0() {
        let data = array![1.0, 2.0, 3.0, 4.0];
        let result = simd_seasonal_difference_f64(&data.view(), 2, 0)
            .expect("order-0 seasonal diff should return data unchanged");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_simd_seasonal_difference_invalid_period() {
        let data = array![1.0, 2.0, 3.0];
        let result = simd_seasonal_difference_f64(&data.view(), 0, 1);
        assert!(result.is_err(), "Period 0 should fail");
    }

    // --- Autocorrelation tests ---

    #[test]
    fn test_simd_autocorrelation_f64_lag0() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let acf = simd_autocorrelation_f64(&data.view(), Some(3))
            .expect("simd_autocorrelation_f64 should succeed");
        // ACF at lag 0 should be 1.0
        assert!(
            (acf[0] - 1.0).abs() < TOLERANCE_F64,
            "ACF at lag 0 should be 1.0, got {}",
            acf[0]
        );
        assert_eq!(acf.len(), 4); // lags 0, 1, 2, 3
    }

    #[test]
    fn test_simd_autocorrelation_f64_known_values() {
        // For a linear series [1,2,3,4,5], ACF should decrease with lag
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let acf = simd_autocorrelation_f64(&data.view(), Some(4)).expect("ACF should succeed");
        // ACF(0) = 1.0
        assert!((acf[0] - 1.0).abs() < TOLERANCE_F64);
        // ACF should decrease
        for i in 1..acf.len() {
            assert!(
                acf[i] <= acf[i - 1] + TOLERANCE_F64,
                "ACF should be non-increasing for linear series at lag {i}"
            );
        }
    }

    #[test]
    fn test_simd_autocorrelation_f32() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let acf = simd_autocorrelation_f32(&data.view(), Some(2)).expect("f32 ACF should succeed");
        assert!((acf[0] - 1.0f32).abs() < TOLERANCE_F32);
        assert_eq!(acf.len(), 3);
    }

    #[test]
    fn test_simd_autocorrelation_constant_series() {
        let data = array![5.0, 5.0, 5.0, 5.0];
        let result = simd_autocorrelation_f64(&data.view(), Some(2));
        assert!(result.is_err(), "Constant series should return error");
    }

    // --- PACF tests ---

    #[test]
    fn test_simd_pacf_f64_lag0() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let pacf =
            simd_partial_autocorrelation_f64(&data.view(), Some(3)).expect("PACF should succeed");
        // PACF at lag 0 should be 1.0
        assert!(
            (pacf[0] - 1.0).abs() < TOLERANCE_F64,
            "PACF at lag 0 should be 1.0, got {}",
            pacf[0]
        );
    }

    #[test]
    fn test_simd_pacf_f32() {
        let data = array![1.0f32, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0, 6.0];
        let pacf = simd_partial_autocorrelation_f32(&data.view(), Some(2))
            .expect("f32 PACF should succeed");
        assert_eq!(pacf.len(), 3);
        assert!((pacf[0] - 1.0f32).abs() < TOLERANCE_F32);
    }

    // --- Convolution tests ---

    #[test]
    fn test_simd_convolve_f64_identity() {
        let signal = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = array![1.0]; // Identity kernel
        let result = simd_convolve_f64(&signal.view(), &kernel.view())
            .expect("Convolution with identity kernel should succeed");
        assert_eq!(result.len(), 5);
        for i in 0..5 {
            assert!(
                (result[i] - signal[i]).abs() < TOLERANCE_F64,
                "Identity convolution mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_simd_convolve_f64_known() {
        // conv([1, 2, 3], [1, 1]) = [1, 3, 5, 3]
        let signal = array![1.0, 2.0, 3.0];
        let kernel = array![1.0, 1.0];
        let result = simd_convolve_f64(&signal.view(), &kernel.view())
            .expect("Known convolution should succeed");
        let expected = array![1.0, 3.0, 5.0, 3.0];
        assert_eq!(result.len(), 4);
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F64,
                "Convolution mismatch at {i}: got {}, expected {}",
                result[i],
                expected[i]
            );
        }
    }

    #[test]
    fn test_simd_convolve_f32() {
        let signal = array![1.0f32, 0.0, 1.0];
        let kernel = array![1.0f32, 2.0];
        let result = simd_convolve_f32(&signal.view(), &kernel.view())
            .expect("f32 convolution should succeed");
        // conv([1, 0, 1], [1, 2]) = [1, 2, 1, 2]
        let expected = array![1.0f32, 2.0, 1.0, 2.0];
        assert_eq!(result.len(), 4);
        for i in 0..result.len() {
            assert!(
                (result[i] - expected[i]).abs() < TOLERANCE_F32,
                "f32 conv mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_simd_convolve_empty() {
        let signal: Array1<f64> = array![];
        let kernel = array![1.0];
        let result = simd_convolve_f64(&signal.view(), &kernel.view());
        assert!(result.is_err(), "Empty signal should fail");
    }

    // --- MA filter tests ---

    #[test]
    fn test_simd_ma_filter_f64() {
        let innovations = array![1.0, 0.0, 0.0, 0.0, 0.0];
        let ma_coeffs = array![1.0, 0.5, 0.25];
        let result = simd_ma_filter_f64(&innovations.view(), &ma_coeffs.view())
            .expect("MA filter should succeed");
        // y[0] = 1.0*1.0 = 1.0
        // y[1] = 1.0*0.0 + 0.5*1.0 = 0.5
        // y[2] = 1.0*0.0 + 0.5*0.0 + 0.25*1.0 = 0.25
        // y[3] = 0.0
        // y[4] = 0.0
        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < TOLERANCE_F64);
        assert!((result[1] - 0.5).abs() < TOLERANCE_F64);
        assert!((result[2] - 0.25).abs() < TOLERANCE_F64);
        assert!((result[3] - 0.0).abs() < TOLERANCE_F64);
        assert!((result[4] - 0.0).abs() < TOLERANCE_F64);
    }

    #[test]
    fn test_simd_ma_filter_f32() {
        let innovations = array![1.0f32, 1.0, 1.0, 1.0];
        let ma_coeffs = array![1.0f32, -0.5];
        let result = simd_ma_filter_f32(&innovations.view(), &ma_coeffs.view())
            .expect("f32 MA filter should succeed");
        // y[0] = 1.0*1.0 = 1.0
        // y[1] = 1.0*1.0 + (-0.5)*1.0 = 0.5
        // y[2] = 1.0*1.0 + (-0.5)*1.0 = 0.5
        // y[3] = 1.0*1.0 + (-0.5)*1.0 = 0.5
        assert!((result[0] - 1.0f32).abs() < TOLERANCE_F32);
        assert!((result[1] - 0.5f32).abs() < TOLERANCE_F32);
    }

    // --- Seasonal means tests ---

    #[test]
    fn test_simd_seasonal_means_f64() {
        // Period=3, data = [1,2,3, 4,5,6, 7,8,9]
        // Mean for pos 0: (1+4+7)/3 = 4.0
        // Mean for pos 1: (2+5+8)/3 = 5.0
        // Mean for pos 2: (3+6+9)/3 = 6.0
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let means =
            simd_seasonal_means_f64(&data.view(), 3).expect("Seasonal means should succeed");
        assert_eq!(means.len(), 3);
        assert!((means[0] - 4.0).abs() < TOLERANCE_F64);
        assert!((means[1] - 5.0).abs() < TOLERANCE_F64);
        assert!((means[2] - 6.0).abs() < TOLERANCE_F64);
    }

    #[test]
    fn test_simd_seasonal_means_f32() {
        let data = array![10.0f32, 20.0, 30.0, 40.0];
        let means =
            simd_seasonal_means_f32(&data.view(), 2).expect("f32 seasonal means should succeed");
        // pos 0: (10+30)/2 = 20, pos 1: (20+40)/2 = 30
        assert!((means[0] - 20.0f32).abs() < TOLERANCE_F32);
        assert!((means[1] - 30.0f32).abs() < TOLERANCE_F32);
    }

    #[test]
    fn test_simd_seasonal_means_invalid_period() {
        let data = array![1.0, 2.0];
        let result = simd_seasonal_means_f64(&data.view(), 0);
        assert!(result.is_err(), "Period 0 should fail");
    }

    // --- Deseason tests ---

    #[test]
    fn test_simd_deseason_f64() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (deseasoned, seasonal) =
            simd_deseason_f64(&data.view(), 3).expect("Deseason should succeed");
        assert_eq!(deseasoned.len(), 6);
        assert_eq!(seasonal.len(), 6);

        // Seasonal means for period 3: pos0=(1+4)/2=2.5, pos1=(2+5)/2=3.5, pos2=(3+6)/2=4.5
        // seasonal = [2.5, 3.5, 4.5, 2.5, 3.5, 4.5]
        // deseasoned = data - seasonal = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
        assert!((seasonal[0] - 2.5).abs() < TOLERANCE_F64);
        assert!((seasonal[3] - 2.5).abs() < TOLERANCE_F64);
        assert!((deseasoned[0] - (-1.5)).abs() < TOLERANCE_F64);
        assert!((deseasoned[3] - 1.5).abs() < TOLERANCE_F64);
    }

    // --- Moving mean tests ---

    #[test]
    fn test_simd_moving_mean_f64() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_moving_mean_f64(&data.view(), 3).expect("Moving mean should succeed");
        assert_eq!(result.len(), 5);
        // With centered window of 3:
        // i=0: window=[1,2] -> mean=1.5 (partial), or [1,2,3]->2 depending on edges
        // The centered approach: half=1
        // i=0: start=0, end=min(2,5)=2 -> [1,2] mean=1.5
        // i=1: start=0, end=min(3,5)=3 -> [1,2,3] mean=2.0
        // i=2: start=1, end=min(4,5)=4 -> [2,3,4] mean=3.0
        // i=3: start=2, end=min(5,5)=5 -> [3,4,5] mean=4.0
        // i=4: start=3, end=min(6,5)=5 -> [4,5] mean=4.5
        assert!((result[1] - 2.0).abs() < TOLERANCE_F64, "Got {}", result[1]);
        assert!((result[2] - 3.0).abs() < TOLERANCE_F64, "Got {}", result[2]);
        assert!((result[3] - 4.0).abs() < TOLERANCE_F64, "Got {}", result[3]);
    }

    #[test]
    fn test_simd_moving_mean_f32() {
        let data = array![10.0f32, 20.0, 30.0, 20.0, 10.0];
        let result = simd_moving_mean_f32(&data.view(), 3).expect("f32 moving mean should succeed");
        assert_eq!(result.len(), 5);
        // Centered: i=1 -> [10,20,30] = 20.0
        assert!((result[1] - 20.0f32).abs() < TOLERANCE_F32);
    }

    #[test]
    fn test_simd_moving_mean_window_1() {
        let data = array![1.0, 2.0, 3.0];
        let result = simd_moving_mean_f64(&data.view(), 1).expect("Window=1 should return data");
        for i in 0..3 {
            assert!((result[i] - data[i]).abs() < TOLERANCE_F64);
        }
    }

    #[test]
    fn test_simd_moving_mean_invalid_window() {
        let data = array![1.0, 2.0];
        let result = simd_moving_mean_f64(&data.view(), 0);
        assert!(result.is_err(), "Window 0 should fail");
    }

    // --- Moving variance tests ---

    #[test]
    fn test_simd_moving_variance_f64() {
        let data = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = simd_moving_variance_f64(&data.view(), 3)
            .expect("Moving variance of constant should succeed");
        // Variance of constant = 0
        for i in 0..result.len() {
            assert!(
                result[i].abs() < TOLERANCE_F64,
                "Constant variance should be ~0 at {i}, got {}",
                result[i]
            );
        }
    }

    #[test]
    fn test_simd_moving_variance_f64_nonconstant() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result =
            simd_moving_variance_f64(&data.view(), 3).expect("Moving variance should succeed");
        assert_eq!(result.len(), 8);
        // For window [2,3,4] at i=2: var = ((2-3)^2 + (3-3)^2 + (4-3)^2)/3 = 2/3
        assert!(
            result[2] > 0.0,
            "Variance of non-constant window should be positive"
        );
    }

    #[test]
    fn test_simd_moving_variance_f32() {
        let data = array![2.0f32, 4.0, 6.0, 8.0];
        let result =
            simd_moving_variance_f32(&data.view(), 2).expect("f32 moving variance should succeed");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_simd_moving_variance_invalid_window() {
        let data = array![1.0, 2.0, 3.0];
        let result = simd_moving_variance_f64(&data.view(), 1);
        assert!(result.is_err(), "Window < 2 should fail for variance");
    }

    // --- EMA tests ---

    #[test]
    fn test_simd_ema_f64() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let alpha = 0.5;
        let result =
            simd_exponential_moving_average_f64(&data.view(), alpha).expect("EMA should succeed");
        assert_eq!(result.len(), 5);
        // EMA[0] = 1.0
        // EMA[1] = 0.5*2 + 0.5*1 = 1.5
        // EMA[2] = 0.5*3 + 0.5*1.5 = 2.25
        // EMA[3] = 0.5*4 + 0.5*2.25 = 3.125
        // EMA[4] = 0.5*5 + 0.5*3.125 = 4.0625
        assert!((result[0] - 1.0).abs() < TOLERANCE_F64);
        assert!((result[1] - 1.5).abs() < TOLERANCE_F64);
        assert!((result[2] - 2.25).abs() < TOLERANCE_F64);
        assert!((result[3] - 3.125).abs() < TOLERANCE_F64);
        assert!((result[4] - 4.0625).abs() < TOLERANCE_F64);
    }

    #[test]
    fn test_simd_ema_f32() {
        let data = array![10.0f32, 20.0, 30.0];
        let alpha = 1.0f32; // alpha=1 means EMA = data
        let result = simd_exponential_moving_average_f32(&data.view(), alpha)
            .expect("EMA with alpha=1 should succeed");
        for i in 0..3 {
            assert!(
                (result[i] - data[i]).abs() < TOLERANCE_F32,
                "EMA with alpha=1 should equal data at {i}"
            );
        }
    }

    #[test]
    fn test_simd_ema_invalid_alpha() {
        let data = array![1.0, 2.0, 3.0];
        let result = simd_exponential_moving_average_f64(&data.view(), 0.0);
        assert!(result.is_err(), "Alpha=0 should fail");
        let result_neg = simd_exponential_moving_average_f64(&data.view(), -0.5);
        assert!(result_neg.is_err(), "Negative alpha should fail");
    }

    // --- Integration test: differencing + autocorrelation round-trip ---

    #[test]
    fn test_simd_diff_then_acf() {
        // White noise-like data: differencing a random walk should give ~ white noise
        // We simulate a simple random walk: cumsum of increments
        let increments = array![
            0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, 0.0, -0.1, 0.2, 0.15, -0.25, 0.05, 0.1, -0.15,
            0.2, -0.1, 0.05, 0.15, -0.2
        ];
        let mut walk = Array1::<f64>::zeros(increments.len() + 1);
        walk[0] = 0.0;
        for i in 0..increments.len() {
            walk[i + 1] = walk[i] + increments[i];
        }

        // Difference the random walk back
        let diff =
            simd_difference_f64(&walk.view(), 1).expect("Differencing random walk should succeed");
        assert_eq!(diff.len(), increments.len());

        // The differenced values should match the original increments
        for i in 0..diff.len() {
            assert!(
                (diff[i] - increments[i]).abs() < TOLERANCE_F64,
                "Diff[{i}] = {}, expected {}",
                diff[i],
                increments[i]
            );
        }

        // Compute ACF of the differenced series
        let acf = simd_autocorrelation_f64(&diff.view(), Some(5)).expect("ACF should succeed");
        assert!((acf[0] - 1.0).abs() < TOLERANCE_F64, "ACF[0] should be 1.0");
    }
}
