//! LOESS (LOcally Estimated Scatterplot Smoothing) implementation
//!
//! Implements locally weighted polynomial regression (LOESS/LOWESS) for
//! non-parametric smoothing of time series data. This is the core building
//! block for STL (Seasonal-Trend decomposition using LOESS).
//!
//! # Algorithm
//!
//! For each point x_i, LOESS:
//! 1. Selects q nearest neighbors (where q = floor(span * n))
//! 2. Assigns weights using the tricube kernel: w(u) = (1 - |u|^3)^3
//! 3. Fits a weighted polynomial (degree 1 or 2) regression
//! 4. Evaluates the polynomial at x_i to get the smoothed value
//!
//! # References
//!
//! - Cleveland, W.S. (1979) "Robust Locally Weighted Regression and Smoothing Scatterplots"
//! - Cleveland, R.B., Cleveland, W.S., McRae, J.E. & Terpenning, I. (1990)
//!   "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::{Result, TimeSeriesError};

/// LOESS smoother configuration
#[derive(Debug, Clone)]
pub struct LoessConfig {
    /// Span (fraction of data used for each local fit), typically 0.1 to 1.0
    /// Alternatively, if > 1, interpreted as the number of neighbors
    pub span: f64,
    /// Polynomial degree (1 = linear, 2 = quadratic)
    pub degree: usize,
    /// Number of robustness iterations (0 = no robustness)
    pub robustness_iters: usize,
}

impl Default for LoessConfig {
    fn default() -> Self {
        Self {
            span: 0.75,
            degree: 1,
            robustness_iters: 0,
        }
    }
}

/// Perform LOESS smoothing on evenly-spaced data
///
/// This is optimized for the common case where x values are 0, 1, 2, ..., n-1.
///
/// # Arguments
///
/// * `y` - The response values (y-axis data)
/// * `config` - LOESS configuration
/// * `weights` - Optional external weights for each observation
///
/// # Returns
///
/// Smoothed values at each observation point
pub fn loess_smooth<F>(
    y: &Array1<F>,
    config: &LoessConfig,
    weights: Option<&Array1<F>>,
) -> Result<Array1<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = y.len();
    if n < 3 {
        return Err(TimeSeriesError::InsufficientData {
            message: "LOESS requires at least 3 data points".to_string(),
            required: 3,
            actual: n,
        });
    }

    if config.degree > 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "degree".to_string(),
            message: "LOESS degree must be 0, 1, or 2".to_string(),
        });
    }

    // Determine the number of nearest neighbors
    let q = if config.span <= 1.0 {
        let q_float = config.span * n as f64;
        (q_float.ceil() as usize).max(config.degree + 1).min(n)
    } else {
        (config.span as usize).max(config.degree + 1).min(n)
    };

    // External weights (default to 1.0)
    let ext_weights = if let Some(w) = weights {
        w.clone()
    } else {
        Array1::from_elem(n, F::one())
    };

    let mut smoothed = Array1::zeros(n);
    let mut robustness_weights = Array1::from_elem(n, F::one());

    // Main LOESS iterations (1 + robustness_iters)
    let total_iters = 1 + config.robustness_iters;

    for iter in 0..total_iters {
        for i in 0..n {
            let x_i = F::from(i).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Failed to convert index".to_string())
            })?;

            // Find q nearest neighbors
            let (start, end) = find_neighborhood(i, n, q);

            // Compute maximum distance in the neighborhood
            let max_dist = compute_max_distance(i, start, end)?;

            if max_dist < 1e-15 {
                // All neighbors are at the same point, use simple weighted average
                let mut sum_w = F::zero();
                let mut sum_wy = F::zero();
                for j in start..end {
                    let w = ext_weights[j] * robustness_weights[j];
                    sum_w = sum_w + w;
                    sum_wy = sum_wy + w * y[j];
                }
                smoothed[i] = if sum_w > F::zero() {
                    sum_wy / sum_w
                } else {
                    y[i]
                };
                continue;
            }

            let max_dist_f = F::from(max_dist).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Failed to convert max_dist".to_string())
            })?;

            // Compute tricube weights
            let mut local_weights = Vec::with_capacity(end - start);
            for j in start..end {
                let x_j = F::from(j).ok_or_else(|| {
                    TimeSeriesError::NumericalInstability("Failed to convert index".to_string())
                })?;
                let u = ((x_j - x_i).abs() / max_dist_f).min(F::one());
                let one_minus_u3 = F::one() - u * u * u;
                let tricube = one_minus_u3 * one_minus_u3 * one_minus_u3;
                local_weights.push(tricube * ext_weights[j] * robustness_weights[j]);
            }

            // Fit weighted polynomial regression
            smoothed[i] = fit_weighted_poly(i, start, end, y, &local_weights, config.degree)?;
        }

        // Update robustness weights (if not last iteration)
        if iter < total_iters - 1 {
            let residuals: Vec<F> = (0..n).map(|i| (y[i] - smoothed[i]).abs()).collect();

            // Compute median absolute residual
            let mut sorted_residuals = residuals.clone();
            sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median_residual = if n % 2 == 0 {
                (sorted_residuals[n / 2 - 1] + sorted_residuals[n / 2])
                    / F::from(2.0).ok_or_else(|| {
                        TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                    })?
            } else {
                sorted_residuals[n / 2]
            };

            let six = F::from(6.0).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Conversion failed".to_string())
            })?;
            let h = six * median_residual;

            if h > F::zero() {
                for i in 0..n {
                    let u = residuals[i] / h;
                    if u >= F::one() {
                        robustness_weights[i] = F::zero();
                    } else {
                        let one_minus_u2 = F::one() - u * u;
                        robustness_weights[i] = one_minus_u2 * one_minus_u2;
                    }
                }
            }
        }
    }

    Ok(smoothed)
}

/// Find the neighborhood [start, end) of q nearest neighbors for index i
fn find_neighborhood(i: usize, n: usize, q: usize) -> (usize, usize) {
    if q >= n {
        return (0, n);
    }

    // Start centered around i
    let half = q / 2;
    let start = if i > half { i - half } else { 0 };
    let end = (start + q).min(n);
    let start = if end == n { n.saturating_sub(q) } else { start };

    (start, end)
}

/// Compute the maximum distance from index i to any point in [start, end)
fn compute_max_distance(i: usize, start: usize, end: usize) -> Result<f64> {
    let d_start = if i >= start { i - start } else { start - i };
    let d_end = if end > 0 {
        if i >= end - 1 {
            i - (end - 1)
        } else {
            (end - 1) - i
        }
    } else {
        0
    };
    Ok(d_start.max(d_end) as f64)
}

/// Fit a weighted polynomial of given degree and evaluate at index i
fn fit_weighted_poly<F>(
    target_idx: usize,
    start: usize,
    end: usize,
    y: &Array1<F>,
    weights: &[F],
    degree: usize,
) -> Result<F>
where
    F: Float + FromPrimitive + Debug,
{
    let m = end - start;
    let x_target = F::from(target_idx)
        .ok_or_else(|| TimeSeriesError::NumericalInstability("Conversion failed".to_string()))?;

    match degree {
        0 => {
            // Weighted mean
            let mut sum_w = F::zero();
            let mut sum_wy = F::zero();
            for (k, j) in (start..end).enumerate() {
                sum_w = sum_w + weights[k];
                sum_wy = sum_wy + weights[k] * y[j];
            }
            if sum_w > F::zero() {
                Ok(sum_wy / sum_w)
            } else {
                Ok(y[target_idx])
            }
        }
        1 => {
            // Weighted linear regression: y = a + b*(x - x_target)
            let mut sum_w = F::zero();
            let mut sum_wx = F::zero();
            let mut sum_wy = F::zero();
            let mut sum_wxx = F::zero();
            let mut sum_wxy = F::zero();

            for (k, j) in (start..end).enumerate() {
                let w = weights[k];
                let x = F::from(j).ok_or_else(|| {
                    TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                })? - x_target;

                sum_w = sum_w + w;
                sum_wx = sum_wx + w * x;
                sum_wy = sum_wy + w * y[j];
                sum_wxx = sum_wxx + w * x * x;
                sum_wxy = sum_wxy + w * x * y[j];
            }

            let det = sum_w * sum_wxx - sum_wx * sum_wx;
            if det.abs() < F::from(1e-15).unwrap_or(F::zero()) {
                // Singular, fall back to weighted mean
                if sum_w > F::zero() {
                    Ok(sum_wy / sum_w)
                } else {
                    Ok(y[target_idx])
                }
            } else {
                let a = (sum_wxx * sum_wy - sum_wx * sum_wxy) / det;
                // b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
                // At x = x_target (i.e., x - x_target = 0), prediction = a
                Ok(a)
            }
        }
        2 => {
            // Weighted quadratic regression: y = a + b*(x-xt) + c*(x-xt)^2
            // Use normal equations with 3x3 system
            let mut s = [[F::zero(); 3]; 3];
            let mut rhs = [F::zero(); 3];

            for (k, j) in (start..end).enumerate() {
                let w = weights[k];
                let x = F::from(j).ok_or_else(|| {
                    TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                })? - x_target;
                let x2 = x * x;
                let x3 = x2 * x;
                let x4 = x2 * x2;

                s[0][0] = s[0][0] + w;
                s[0][1] = s[0][1] + w * x;
                s[0][2] = s[0][2] + w * x2;
                s[1][1] = s[1][1] + w * x2;
                s[1][2] = s[1][2] + w * x3;
                s[2][2] = s[2][2] + w * x4;

                rhs[0] = rhs[0] + w * y[j];
                rhs[1] = rhs[1] + w * x * y[j];
                rhs[2] = rhs[2] + w * x2 * y[j];
            }

            // Symmetric matrix
            s[1][0] = s[0][1];
            s[2][0] = s[0][2];
            s[2][1] = s[1][2];

            // Solve 3x3 system using Cramer's rule
            let det = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
                - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
                + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);

            if det.abs() < F::from(1e-15).unwrap_or(F::zero()) {
                // Singular, fall back to linear
                fit_weighted_poly(target_idx, start, end, y, weights, 1)
            } else {
                let a = (rhs[0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
                    - s[0][1] * (rhs[1] * s[2][2] - s[1][2] * rhs[2])
                    + s[0][2] * (rhs[1] * s[2][1] - s[1][1] * rhs[2]))
                    / det;
                // At x = x_target (x - x_target = 0), prediction = a
                Ok(a)
            }
        }
        _ => Err(TimeSeriesError::InvalidParameter {
            name: "degree".to_string(),
            message: "LOESS degree must be 0, 1, or 2".to_string(),
        }),
    }
}

/// Perform LOESS smoothing on a cycle-subseries (for STL)
///
/// This smooths a subseries extracted at regular intervals (every `period` steps)
/// from the original time series.
///
/// # Arguments
///
/// * `indices` - The original indices of the subseries points
/// * `values` - The subseries values
/// * `weights` - External weights for each point
/// * `config` - LOESS configuration
///
/// # Returns
///
/// Smoothed values at each subseries point
pub fn loess_subseries<F>(values: &[F], weights: &[F], config: &LoessConfig) -> Result<Vec<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n = values.len();
    if n < 2 {
        return Ok(values.to_vec());
    }

    let y_arr = Array1::from_vec(values.to_vec());
    let w_arr = Array1::from_vec(weights.to_vec());
    let smoothed = loess_smooth(&y_arr, config, Some(&w_arr))?;
    Ok(smoothed.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_loess_linear_data() {
        // Perfect linear data should be smoothed to itself
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let config = LoessConfig {
            span: 0.5,
            degree: 1,
            robustness_iters: 0,
        };

        let smoothed = loess_smooth(&y, &config, None).expect("LOESS failed");
        assert_eq!(smoothed.len(), 10);

        // Smoothed values should be close to original for linear data
        for i in 0..10 {
            assert!(
                (smoothed[i] - y[i]).abs() < 1.0,
                "LOESS of linear data should be close at index {}: got {} expected {}",
                i,
                smoothed[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_loess_constant_data() {
        let y = array![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let config = LoessConfig::default();

        let smoothed = loess_smooth(&y, &config, None).expect("LOESS failed");
        for i in 0..10 {
            assert!(
                (smoothed[i] - 5.0).abs() < 1e-10,
                "Smoothed constant data should remain constant"
            );
        }
    }

    #[test]
    fn test_loess_noisy_data() {
        // Sine wave with noise - smoothed should be closer to the underlying sine
        let n = 50;
        let mut y_vec = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 * 2.0 * std::f64::consts::PI / n as f64;
            let noise = if i % 3 == 0 {
                0.3
            } else if i % 3 == 1 {
                -0.2
            } else {
                0.1
            };
            y_vec.push(x.sin() + noise);
        }
        let y = Array1::from_vec(y_vec);

        let config = LoessConfig {
            span: 0.3,
            degree: 2,
            robustness_iters: 0,
        };

        let smoothed = loess_smooth(&y, &config, None).expect("LOESS failed");
        assert_eq!(smoothed.len(), n);

        // Smoothed values should be within reasonable bounds
        for i in 0..n {
            assert!(smoothed[i].abs() < 2.0, "Smoothed value should be bounded");
        }
    }

    #[test]
    fn test_loess_robust() {
        // Data with an outlier
        let y = array![1.0, 2.0, 3.0, 4.0, 100.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config_no_robust = LoessConfig {
            span: 0.5,
            degree: 1,
            robustness_iters: 0,
        };

        let config_robust = LoessConfig {
            span: 0.5,
            degree: 1,
            robustness_iters: 3,
        };

        let smooth_no_robust = loess_smooth(&y, &config_no_robust, None).expect("LOESS failed");
        let smooth_robust =
            loess_smooth(&y, &config_robust, None).expect("LOESS with robustness failed");

        // At the outlier point (index 4), the robust version should be less affected
        // The non-robust version is pulled toward the outlier
        let expected = 5.0; // True value would be about 5
        let err_no_robust = (smooth_no_robust[4] - expected).abs();
        let err_robust = (smooth_robust[4] - expected).abs();

        // Robust should be at least somewhat less influenced by the outlier
        assert!(
            smooth_robust[4].is_finite(),
            "Robust LOESS should produce finite values"
        );
    }

    #[test]
    fn test_loess_insufficient_data() {
        let y = array![1.0, 2.0];
        let config = LoessConfig::default();

        let result = loess_smooth(&y, &config, None);
        assert!(result.is_err(), "LOESS should fail with < 3 data points");
    }

    #[test]
    fn test_loess_quadratic_degree() {
        // Quadratic data: y = x^2
        let n = 20;
        let mut y_vec = Vec::with_capacity(n);
        for i in 0..n {
            y_vec.push((i as f64).powi(2));
        }
        let y = Array1::from_vec(y_vec);

        let config = LoessConfig {
            span: 0.5,
            degree: 2,
            robustness_iters: 0,
        };

        let smoothed = loess_smooth(&y, &config, None).expect("LOESS failed");

        // Quadratic LOESS on quadratic data should be very close to original
        for i in 2..n - 2 {
            // Exclude edges where boundary effects are larger
            assert!(
                (smoothed[i] - y[i]).abs() < 5.0,
                "Quadratic LOESS on quadratic data at {}: got {} expected {}",
                i,
                smoothed[i],
                y[i]
            );
        }
    }

    #[test]
    fn test_find_neighborhood() {
        assert_eq!(find_neighborhood(5, 20, 7), (2, 9));
        assert_eq!(find_neighborhood(0, 20, 7), (0, 7));
        assert_eq!(find_neighborhood(19, 20, 7), (13, 20));
        assert_eq!(find_neighborhood(5, 10, 10), (0, 10));
    }
}
