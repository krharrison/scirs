//! Robust statistical estimators
//!
//! This module provides robust estimators for location and scale parameters
//! that are resistant to outliers and violations of distributional assumptions.
//!
//! ## Estimators provided
//!
//! - **Trimmed mean**: Discards a proportion of extreme values before averaging
//! - **Huber M-estimator**: Location estimator using Huber's influence function
//! - **Tukey biweight M-estimator**: Location estimator using Tukey's bisquare function
//! - **IRLS for location**: Generic Iteratively Reweighted Least Squares framework
//!
//! ## Comparison with existing functions
//!
//! - `median_abs_deviation` (in `dispersion.rs`) provides MAD
//! - `winsorized_mean` (in `quantile.rs`) replaces extremes rather than discarding them

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};

// ---------------------------------------------------------------------------
// Helper: sort a slice copy
// ---------------------------------------------------------------------------
fn sorted_copy<F: Float>(x: &ArrayView1<F>) -> Vec<F> {
    let mut v: Vec<F> = x.iter().cloned().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

// ---------------------------------------------------------------------------
// Helper: median of a sorted slice
// ---------------------------------------------------------------------------
fn median_sorted<F: Float>(sorted: &[F]) -> F {
    let n = sorted.len();
    if n == 0 {
        return F::zero();
    }
    let mid = n / 2;
    if n % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / F::from(2.0).unwrap_or_else(|| F::one() + F::one())
    } else {
        sorted[mid]
    }
}

// ---------------------------------------------------------------------------
// Helper: MAD (median absolute deviation) of a sorted slice from a centre
// ---------------------------------------------------------------------------
fn mad_from_center<F: Float>(sorted: &[F], center: F) -> F {
    let mut abs_devs: Vec<F> = sorted.iter().map(|&v| (v - center).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    median_sorted(&abs_devs)
}

// ===========================================================================
// Trimmed mean
// ===========================================================================

/// Compute the trimmed mean of a dataset.
///
/// The trimmed mean discards a proportion of the smallest and largest values
/// before computing the arithmetic mean. This makes it more robust to outliers
/// than the ordinary mean while retaining higher efficiency than the median.
///
/// # Arguments
///
/// * `x` - Input data
/// * `proportiontocut` - Fraction (0.0 to < 0.5) of data to trim from each tail.
///   For example, 0.1 removes the lowest 10 % and highest 10 % of values.
///
/// # Returns
///
/// The trimmed mean as `F`.
///
/// # Errors
///
/// Returns an error if `x` is empty, `proportiontocut` is not in [0, 0.5),
/// or trimming would remove all observations.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::trimmed_mean;
///
/// let data = array![1.0_f64, 2.0, 3.0, 4.0, 100.0];
/// let tm = trimmed_mean(&data.view(), 0.2).expect("trimmed mean failed");
/// // Trimming 20% from each end removes 1 value from each side: mean of [2,3,4] = 3
/// assert!((tm - 3.0_f64).abs() < 1e-10);
/// ```
pub fn trimmed_mean<F>(x: &ArrayView1<F>, proportiontocut: f64) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    if proportiontocut < 0.0 || proportiontocut >= 0.5 {
        return Err(StatsError::InvalidArgument(
            "proportiontocut must be in [0.0, 0.5)".to_string(),
        ));
    }

    let n = x.len();
    let k = (n as f64 * proportiontocut).floor() as usize;

    // After removing k from each end we need at least 1 value
    if 2 * k >= n {
        return Err(StatsError::InvalidArgument(format!(
            "Trimming {} from each end of {} observations leaves no data",
            k, n
        )));
    }

    let sorted = sorted_copy(x);
    let trimmed = &sorted[k..n - k];
    let count = trimmed.len();

    let sum: F = trimmed.iter().cloned().sum();
    let count_f = F::from(count).ok_or_else(|| {
        StatsError::ComputationError("Failed to convert count to float".to_string())
    })?;

    Ok(sum / count_f)
}

// ===========================================================================
// Weight functions for M-estimators
// ===========================================================================

/// Weight function types for M-estimators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MEstimatorWeight {
    /// Huber weight: w(u) = 1 if |u| <= c, else c/|u|
    /// Default c = 1.345 gives 95% efficiency at the normal model.
    Huber {
        /// Tuning constant
        c: f64,
    },
    /// Tukey bisquare (biweight): w(u) = (1 - u^2)^2 if |u| <= c, else 0
    /// Default c = 4.685 gives 95% efficiency at the normal model.
    TukeyBisquare {
        /// Tuning constant
        c: f64,
    },
}

impl MEstimatorWeight {
    /// Huber weight with default tuning constant (1.345).
    pub fn huber_default() -> Self {
        MEstimatorWeight::Huber { c: 1.345 }
    }

    /// Tukey bisquare weight with default tuning constant (4.685).
    pub fn tukey_default() -> Self {
        MEstimatorWeight::TukeyBisquare { c: 4.685 }
    }

    /// Evaluate the weight for a standardised residual `u`.
    fn weight(&self, u: f64) -> f64 {
        match *self {
            MEstimatorWeight::Huber { c } => {
                let abs_u = u.abs();
                if abs_u <= c {
                    1.0
                } else {
                    c / abs_u
                }
            }
            MEstimatorWeight::TukeyBisquare { c } => {
                let abs_u = u.abs();
                if abs_u <= c {
                    let ratio = u / c;
                    let one_minus_sq = 1.0 - ratio * ratio;
                    one_minus_sq * one_minus_sq
                } else {
                    0.0
                }
            }
        }
    }
}

// ===========================================================================
// IRLS for location estimation (generic)
// ===========================================================================

/// Configuration for the IRLS (Iteratively Reweighted Least Squares) location estimator.
#[derive(Debug, Clone)]
pub struct IrlsConfig {
    /// Weight function to use
    pub weight_fn: MEstimatorWeight,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance (relative change in estimate)
    pub tol: f64,
    /// Whether to use MAD for initial scale estimate (true) or std dev (false)
    pub use_mad_scale: bool,
}

impl Default for IrlsConfig {
    fn default() -> Self {
        Self {
            weight_fn: MEstimatorWeight::huber_default(),
            max_iter: 50,
            tol: 1e-8,
            use_mad_scale: true,
        }
    }
}

/// Result of the IRLS location estimation.
#[derive(Debug, Clone)]
pub struct IrlsResult<F> {
    /// Estimated location
    pub location: F,
    /// Scale estimate used
    pub scale: F,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final weights assigned to each observation
    pub weights: Array1<F>,
}

/// Estimate the location (centre) of a sample using Iteratively Reweighted Least
/// Squares (IRLS) with a configurable weight function.
///
/// The algorithm iteratively computes a weighted mean, where the weights are
/// derived from the weight function evaluated at the standardised residuals
/// (residual / scale). The initial estimate is the median and the scale is
/// estimated from the MAD (by default).
///
/// # Arguments
///
/// * `x` - Input data
/// * `config` - IRLS configuration (weight function, max iterations, tolerance, ...)
///
/// # Returns
///
/// An `IrlsResult` containing the estimated location, scale, iteration count,
/// convergence flag, and final weights.
///
/// # Errors
///
/// Returns an error if the input is empty or the scale estimate is zero (constant data).
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::robust_estimators::{irls_location, IrlsConfig, MEstimatorWeight};
///
/// let data = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 100.0]; // outlier at 100
/// let config = IrlsConfig::default(); // Huber weights
/// let result = irls_location(&data.view(), &config).expect("IRLS failed");
/// // The robust location should be close to the median (3.5), not the mean (19.17)
/// assert!((result.location - 3.0_f64).abs() < 1.5);
/// assert!(result.converged);
/// ```
pub fn irls_location<F>(x: &ArrayView1<F>, config: &IrlsConfig) -> StatsResult<IrlsResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }

    let n = x.len();
    if n == 1 {
        return Ok(IrlsResult {
            location: x[0],
            scale: F::zero(),
            iterations: 0,
            converged: true,
            weights: Array1::ones(1),
        });
    }

    let sorted = sorted_copy(x);
    let med = median_sorted(&sorted);

    // Initial scale estimate
    let scale_f64 = if config.use_mad_scale {
        let mad = mad_from_center(&sorted, med);
        let mad_f64: f64 =
            NumCast::from(mad).ok_or_else(|| StatsError::ComputationError("cast failed".into()))?;
        // Normalise to be consistent with std dev under normality
        mad_f64 / 0.6745
    } else {
        // Use standard deviation
        let mean_f64: f64 = NumCast::from(
            x.iter().cloned().sum::<F>()
                / F::from(n)
                    .ok_or_else(|| StatsError::ComputationError("cast failed".to_string()))?,
        )
        .ok_or_else(|| StatsError::ComputationError("cast failed".into()))?;
        let var: f64 = x
            .iter()
            .map(|&v| {
                let vf: f64 = NumCast::from(v).unwrap_or(0.0);
                (vf - mean_f64) * (vf - mean_f64)
            })
            .sum::<f64>()
            / (n as f64 - 1.0);
        var.sqrt()
    };

    if scale_f64 <= f64::EPSILON {
        // All values are essentially the same
        return Ok(IrlsResult {
            location: med,
            scale: F::zero(),
            iterations: 0,
            converged: true,
            weights: Array1::ones(n),
        });
    }

    let scale = F::from(scale_f64).ok_or_else(|| {
        StatsError::ComputationError("Failed to convert scale to float".to_string())
    })?;

    // IRLS iteration
    let mut mu = med;
    let mut weights = Array1::<F>::ones(n);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        // Compute weights from standardised residuals
        let mut weight_sum = 0.0_f64;
        let mut weighted_sum = 0.0_f64;

        for (i, &xi) in x.iter().enumerate() {
            let xi_f64: f64 =
                NumCast::from(xi).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
            let mu_f64: f64 =
                NumCast::from(mu).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
            let u = (xi_f64 - mu_f64) / scale_f64;
            let w = config.weight_fn.weight(u);

            weights[i] = F::from(w).ok_or_else(|| {
                StatsError::ComputationError("Failed to convert weight".to_string())
            })?;
            weight_sum += w;
            weighted_sum += w * xi_f64;
        }

        if weight_sum <= f64::EPSILON {
            // All observations got zero weight — fall back to median
            return Ok(IrlsResult {
                location: med,
                scale,
                iterations,
                converged: false,
                weights,
            });
        }

        let new_mu_f64 = weighted_sum / weight_sum;
        let new_mu = F::from(new_mu_f64)
            .ok_or_else(|| StatsError::ComputationError("Failed to convert mu".to_string()))?;

        // Check convergence
        let mu_f64: f64 =
            NumCast::from(mu).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
        let rel_change = if mu_f64.abs() > f64::EPSILON {
            ((new_mu_f64 - mu_f64) / mu_f64).abs()
        } else {
            (new_mu_f64 - mu_f64).abs()
        };

        mu = new_mu;

        if rel_change < config.tol {
            converged = true;
            break;
        }
    }

    Ok(IrlsResult {
        location: mu,
        scale,
        iterations,
        converged,
        weights,
    })
}

// ===========================================================================
// Huber M-estimator for location
// ===========================================================================

/// Compute the Huber M-estimator of location.
///
/// The Huber M-estimator uses Huber's weight function which gives full weight
/// to observations within `c` standard deviations of the centre and linearly
/// decreasing weight beyond that. The default tuning constant `c = 1.345`
/// gives 95 % asymptotic efficiency relative to the mean at the normal model.
///
/// Internally this calls [`irls_location`] with `MEstimatorWeight::Huber`.
///
/// # Arguments
///
/// * `x` - Input data
/// * `c` - Optional tuning constant (default 1.345)
/// * `max_iter` - Optional maximum iterations (default 50)
/// * `tol` - Optional convergence tolerance (default 1e-8)
///
/// # Returns
///
/// The Huber M-estimate of location.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::huber_location;
///
/// let data = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 100.0];
/// let loc = huber_location(&data.view(), None, None, None).expect("Huber failed");
/// assert!((loc - 3.0_f64).abs() < 1.5);  // much closer to 3 than to 19.2
/// ```
pub fn huber_location<F>(
    x: &ArrayView1<F>,
    c: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let config = IrlsConfig {
        weight_fn: MEstimatorWeight::Huber {
            c: c.unwrap_or(1.345),
        },
        max_iter: max_iter.unwrap_or(50),
        tol: tol.unwrap_or(1e-8),
        use_mad_scale: true,
    };
    let result = irls_location(x, &config)?;
    Ok(result.location)
}

// ===========================================================================
// Tukey biweight (bisquare) M-estimator for location
// ===========================================================================

/// Compute the Tukey biweight (bisquare) M-estimator of location.
///
/// The Tukey biweight uses a redescending weight function that assigns zero
/// weight to observations more than `c` standardised deviations from the
/// centre. This makes it extremely robust to gross outliers. The default
/// tuning constant `c = 4.685` gives 95 % asymptotic efficiency at the
/// normal model.
///
/// Internally this calls [`irls_location`] with `MEstimatorWeight::TukeyBisquare`.
///
/// # Arguments
///
/// * `x` - Input data
/// * `c` - Optional tuning constant (default 4.685)
/// * `max_iter` - Optional maximum iterations (default 50)
/// * `tol` - Optional convergence tolerance (default 1e-8)
///
/// # Returns
///
/// The biweight M-estimate of location.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::tukey_biweight_location;
///
/// let data = array![1.0_f64, 2.0, 3.0, 4.0, 5.0, 100.0];
/// let loc = tukey_biweight_location(&data.view(), None, None, None).expect("biweight failed");
/// assert!((loc - 3.0_f64).abs() < 1.5);
/// ```
pub fn tukey_biweight_location<F>(
    x: &ArrayView1<F>,
    c: Option<f64>,
    max_iter: Option<usize>,
    tol: Option<f64>,
) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let config = IrlsConfig {
        weight_fn: MEstimatorWeight::TukeyBisquare {
            c: c.unwrap_or(4.685),
        },
        max_iter: max_iter.unwrap_or(50),
        tol: tol.unwrap_or(1e-8),
        use_mad_scale: true,
    };
    let result = irls_location(x, &config)?;
    Ok(result.location)
}

// ===========================================================================
// Biweight midvariance
// ===========================================================================

/// Compute the biweight midvariance, a robust scale estimator.
///
/// The biweight midvariance is the robust analogue of the variance, using
/// Tukey's biweight function. It is highly resistant to outliers.
///
/// The formula is:
/// ```text
///   s^2 = n * sum(u_i < 1: (x_i - M)^2 (1-u_i^2)^4) / (sum(u_i < 1: (1-u_i^2)(1-5u_i^2)))^2
/// ```
/// where M is the median, u_i = (x_i - M) / (c * MAD), and c = 9.0 by default.
///
/// # Arguments
///
/// * `x` - Input data
/// * `c` - Optional tuning constant (default 9.0)
///
/// # Returns
///
/// The biweight midvariance.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::biweight_midvariance;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
/// let bwv = biweight_midvariance(&data.view(), None).expect("bw midvar failed");
/// // Should be much smaller than the ordinary variance (which is inflated by the outlier)
/// assert!(bwv > 0.0);
/// assert!(bwv < 2000.0); // ordinary var is ~1470
/// ```
pub fn biweight_midvariance<F>(x: &ArrayView1<F>, c: Option<f64>) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required for biweight midvariance".to_string(),
        ));
    }

    let c_val = c.unwrap_or(9.0);
    let sorted = sorted_copy(x);
    let med = median_sorted(&sorted);
    let mad = mad_from_center(&sorted, med);

    let mad_f64: f64 =
        NumCast::from(mad).ok_or_else(|| StatsError::ComputationError("cast failed".into()))?;

    if mad_f64 <= f64::EPSILON {
        // All values effectively identical
        return Ok(F::zero());
    }

    let med_f64: f64 =
        NumCast::from(med).ok_or_else(|| StatsError::ComputationError("cast failed".into()))?;
    let denom_scale = c_val * mad_f64;

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    let n_f = n as f64;

    for &xi in x.iter() {
        let xi_f64: f64 =
            NumCast::from(xi).ok_or_else(|| StatsError::ComputationError("cast failed".into()))?;
        let diff = xi_f64 - med_f64;
        let u = diff / denom_scale;
        let u2 = u * u;

        if u2 < 1.0 {
            let one_minus_u2 = 1.0 - u2;
            numerator += diff * diff * one_minus_u2.powi(4);
            denominator += one_minus_u2 * (1.0 - 5.0 * u2);
        }
    }

    if denominator.abs() <= f64::EPSILON {
        return Err(StatsError::ComputationError(
            "Biweight midvariance denominator is zero".to_string(),
        ));
    }

    let result = n_f * numerator / (denominator * denominator);

    F::from(result).ok_or_else(|| {
        StatsError::ComputationError("Failed to convert result to float".to_string())
    })
}

// ===========================================================================
// Biweight midcovariance
// ===========================================================================

/// Compute the biweight midcovariance between two variables.
///
/// This is the robust analogue of the covariance, using Tukey's biweight
/// function applied independently to each variable.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable
/// * `c` - Optional tuning constant (default 9.0)
///
/// # Returns
///
/// The biweight midcovariance.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::biweight_midcovariance;
///
/// let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
/// let bwc = biweight_midcovariance(&x.view(), &y.view(), None).expect("bw midcov failed");
/// assert!(bwc > 0.0);  // positive association
/// ```
pub fn biweight_midcovariance<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    c: Option<f64>,
) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() || y.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input arrays cannot be empty".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(StatsError::DimensionMismatch(
            "Input arrays must have the same length".to_string(),
        ));
    }

    let n = x.len();
    if n < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 observations required".to_string(),
        ));
    }

    let c_val = c.unwrap_or(9.0);

    let sorted_x = sorted_copy(x);
    let sorted_y = sorted_copy(y);
    let med_x = median_sorted(&sorted_x);
    let med_y = median_sorted(&sorted_y);
    let mad_x = mad_from_center(&sorted_x, med_x);
    let mad_y = mad_from_center(&sorted_y, med_y);

    let mad_x_f64: f64 =
        NumCast::from(mad_x).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let mad_y_f64: f64 =
        NumCast::from(mad_y).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let med_x_f64: f64 =
        NumCast::from(med_x).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let med_y_f64: f64 =
        NumCast::from(med_y).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    if mad_x_f64 <= f64::EPSILON || mad_y_f64 <= f64::EPSILON {
        return Ok(F::zero());
    }

    let scale_x = c_val * mad_x_f64;
    let scale_y = c_val * mad_y_f64;
    let n_f = n as f64;

    let mut numerator = 0.0_f64;
    let mut denom_x = 0.0_f64;
    let mut denom_y = 0.0_f64;

    for i in 0..n {
        let xi_f64: f64 =
            NumCast::from(x[i]).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
        let yi_f64: f64 =
            NumCast::from(y[i]).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

        let diff_x = xi_f64 - med_x_f64;
        let diff_y = yi_f64 - med_y_f64;
        let ux = diff_x / scale_x;
        let uy = diff_y / scale_y;
        let ux2 = ux * ux;
        let uy2 = uy * uy;

        if ux2 < 1.0 && uy2 < 1.0 {
            let wx = (1.0 - ux2) * (1.0 - ux2);
            let wy = (1.0 - uy2) * (1.0 - uy2);
            numerator += diff_x * wx * diff_y * wy;
            denom_x += (1.0 - ux2) * (1.0 - 5.0 * ux2);
            denom_y += (1.0 - uy2) * (1.0 - 5.0 * uy2);
        }
    }

    let denom = denom_x * denom_y;
    if denom.abs() <= f64::EPSILON {
        return Ok(F::zero());
    }

    let result = n_f * numerator / denom;
    F::from(result).ok_or_else(|| {
        StatsError::ComputationError("Failed to convert result to float".to_string())
    })
}

// ===========================================================================
// Biweight midcorrelation
// ===========================================================================

/// Compute the biweight midcorrelation between two variables.
///
/// This is the robust analogue of Pearson's correlation coefficient:
/// `r_bw = cov_bw(x,y) / sqrt(var_bw(x) * var_bw(y))`.
///
/// # Arguments
///
/// * `x` - First variable
/// * `y` - Second variable
/// * `c` - Optional tuning constant (default 9.0)
///
/// # Returns
///
/// The biweight midcorrelation in [-1, 1].
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::biweight_midcorrelation;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let y = array![2.0_f64, 4.0, 6.0, 8.0, 10.0];
/// let r = biweight_midcorrelation(&x.view(), &y.view(), None).expect("corr failed");
/// assert!((r - 1.0_f64).abs() < 0.01);  // perfect linear relationship
/// ```
pub fn biweight_midcorrelation<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    c: Option<f64>,
) -> StatsResult<F>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let cov = biweight_midcovariance(x, y, c)?;
    let var_x = biweight_midvariance(x, c)?;
    let var_y = biweight_midvariance(y, c)?;

    let var_x_f64: f64 =
        NumCast::from(var_x).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let var_y_f64: f64 =
        NumCast::from(var_y).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    if var_x_f64 <= f64::EPSILON || var_y_f64 <= f64::EPSILON {
        return Ok(F::zero());
    }

    let cov_f64: f64 =
        NumCast::from(cov).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let r = cov_f64 / (var_x_f64 * var_y_f64).sqrt();
    // Clamp to [-1, 1]
    let r_clamped = r.max(-1.0).min(1.0);

    F::from(r_clamped).ok_or_else(|| {
        StatsError::ComputationError("Failed to convert correlation to float".to_string())
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    // -----------------------------------------------------------------------
    // Trimmed mean tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trimmed_mean_no_trimming() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let tm = trimmed_mean(&data.view(), 0.0).expect("should succeed");
        assert!((tm - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_symmetric() {
        // 20% trim from each end of 10 values = remove 2 from each end
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tm = trimmed_mean(&data.view(), 0.2).expect("should succeed");
        // Trim removes 1,2 and 9,10 -> mean of [3,4,5,6,7,8] = 5.5
        assert!((tm - 5.5).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_with_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let tm = trimmed_mean(&data.view(), 0.2).expect("should succeed");
        // 20% of 6 = 1.2 -> floor = 1, remove 1 from each end
        // sorted: [1, 2, 3, 4, 5, 100] -> trim to [2, 3, 4, 5] = mean 3.5
        assert!((tm - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_empty_error() {
        let data = Array1::<f64>::zeros(0);
        let result = trimmed_mean(&data.view(), 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_trimmed_mean_invalid_proportion() {
        let data = array![1.0, 2.0, 3.0];
        assert!(trimmed_mean(&data.view(), -0.1).is_err());
        assert!(trimmed_mean(&data.view(), 0.5).is_err());
        assert!(trimmed_mean(&data.view(), 0.7).is_err());
    }

    #[test]
    fn test_trimmed_mean_single_element() {
        let data = array![42.0];
        let tm = trimmed_mean(&data.view(), 0.0).expect("should succeed");
        assert!((tm - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_trimmed_mean_all_same() {
        let data = array![5.0, 5.0, 5.0, 5.0, 5.0];
        let tm = trimmed_mean(&data.view(), 0.2).expect("should succeed");
        assert!((tm - 5.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Huber location tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_huber_location_no_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let loc = huber_location(&data.view(), None, None, None).expect("should succeed");
        // Without outliers, should be close to the mean (3.0)
        assert!((loc - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_huber_location_with_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let loc = huber_location(&data.view(), None, None, None).expect("should succeed");
        // Should be much closer to 3.0 than to the mean (19.17)
        assert!((loc - 3.0).abs() < 2.0);
    }

    #[test]
    fn test_huber_location_symmetric() {
        let data = array![-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let loc = huber_location(&data.view(), None, None, None).expect("should succeed");
        // Symmetric outliers: centre should be near 3.0
        assert!((loc - 3.0).abs() < 1.0);
    }

    #[test]
    fn test_huber_location_constant() {
        let data = array![7.0, 7.0, 7.0, 7.0, 7.0];
        let loc = huber_location(&data.view(), None, None, None).expect("should succeed");
        assert!((loc - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_huber_location_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(huber_location(&data.view(), None, None, None).is_err());
    }

    #[test]
    fn test_huber_location_single() {
        let data = array![42.0];
        let loc = huber_location(&data.view(), None, None, None).expect("should succeed");
        assert!((loc - 42.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // Tukey biweight location tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tukey_biweight_no_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let loc = tukey_biweight_location(&data.view(), None, None, None).expect("should succeed");
        assert!((loc - 3.0).abs() < 0.5);
    }

    #[test]
    fn test_tukey_biweight_with_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let loc = tukey_biweight_location(&data.view(), None, None, None).expect("should succeed");
        assert!((loc - 3.0).abs() < 2.0);
    }

    #[test]
    fn test_tukey_biweight_extreme_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 1000.0, -1000.0];
        let loc = tukey_biweight_location(&data.view(), None, None, None).expect("should succeed");
        // Tukey bisquare gives zero weight to extreme outliers
        assert!((loc - 3.0).abs() < 1.5);
    }

    #[test]
    fn test_tukey_biweight_constant() {
        let data = array![3.0, 3.0, 3.0, 3.0];
        let loc = tukey_biweight_location(&data.view(), None, None, None).expect("should succeed");
        assert!((loc - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_tukey_biweight_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(tukey_biweight_location(&data.view(), None, None, None).is_err());
    }

    // -----------------------------------------------------------------------
    // IRLS generic tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_irls_huber_converges() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 50.0];
        let config = IrlsConfig::default();
        let result = irls_location(&data.view(), &config).expect("should succeed");
        assert!(result.converged);
        assert!(result.iterations > 0);
        assert_eq!(result.weights.len(), 6);
    }

    #[test]
    fn test_irls_tukey_converges() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 50.0];
        let config = IrlsConfig {
            weight_fn: MEstimatorWeight::tukey_default(),
            ..Default::default()
        };
        let result = irls_location(&data.view(), &config).expect("should succeed");
        assert!(result.converged);
    }

    #[test]
    fn test_irls_weights_less_for_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let config = IrlsConfig::default();
        let result = irls_location(&data.view(), &config).expect("should succeed");
        // Outlier at index 5 should have lower weight than inliers
        let w_inlier: f64 = NumCast::from(result.weights[2]).expect("cast");
        let w_outlier: f64 = NumCast::from(result.weights[5]).expect("cast");
        assert!(w_outlier < w_inlier);
    }

    #[test]
    fn test_irls_scale_reported() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = IrlsConfig::default();
        let result = irls_location(&data.view(), &config).expect("should succeed");
        let scale_f64: f64 = NumCast::from(result.scale).expect("cast");
        assert!(scale_f64 > 0.0);
    }

    #[test]
    fn test_irls_custom_tuning() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        // Very small c makes Huber more robust
        let config = IrlsConfig {
            weight_fn: MEstimatorWeight::Huber { c: 0.5 },
            max_iter: 100,
            tol: 1e-10,
            use_mad_scale: true,
        };
        let result = irls_location(&data.view(), &config).expect("should succeed");
        let loc_f64: f64 = NumCast::from(result.location).expect("cast");
        // With very small c, should be very close to median
        assert!((loc_f64 - 3.5).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // Biweight midvariance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_biweight_midvariance_basic() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let bwv = biweight_midvariance(&data.view(), None).expect("should succeed");
        assert!(bwv > 0.0);
    }

    #[test]
    fn test_biweight_midvariance_with_outlier() {
        let clean = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let dirty = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

        let bwv_clean = biweight_midvariance(&clean.view(), None).expect("should succeed");
        let bwv_dirty = biweight_midvariance(&dirty.view(), None).expect("should succeed");

        // Biweight midvariance should not blow up with the outlier
        // (ordinary variance goes from ~2.5 to ~1470)
        assert!(bwv_dirty < 50.0);
        // And the clean one should be reasonable
        assert!(bwv_clean > 0.5 && bwv_clean < 10.0);
    }

    #[test]
    fn test_biweight_midvariance_constant() {
        let data = array![3.0, 3.0, 3.0, 3.0];
        let bwv = biweight_midvariance(&data.view(), None).expect("should succeed");
        assert!(bwv.abs() < 1e-10);
    }

    #[test]
    fn test_biweight_midvariance_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(biweight_midvariance(&data.view(), None).is_err());
    }

    #[test]
    fn test_biweight_midvariance_single_error() {
        let data = array![5.0];
        assert!(biweight_midvariance(&data.view(), None).is_err());
    }

    // -----------------------------------------------------------------------
    // Biweight midcovariance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_biweight_midcovariance_positive() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let bwc = biweight_midcovariance(&x.view(), &y.view(), None).expect("should succeed");
        assert!(bwc > 0.0);
    }

    #[test]
    fn test_biweight_midcovariance_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];
        let bwc = biweight_midcovariance(&x.view(), &y.view(), None).expect("should succeed");
        assert!(bwc < 0.0);
    }

    #[test]
    fn test_biweight_midcovariance_dim_mismatch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        assert!(biweight_midcovariance(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn test_biweight_midcovariance_with_outlier() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 200.0];
        let bwc = biweight_midcovariance(&x.view(), &y.view(), None).expect("should succeed");
        // Should still be positive and not dominated by the outlier
        assert!(bwc > 0.0);
    }

    #[test]
    fn test_biweight_midcovariance_empty() {
        let x = Array1::<f64>::zeros(0);
        let y = Array1::<f64>::zeros(0);
        assert!(biweight_midcovariance(&x.view(), &y.view(), None).is_err());
    }

    // -----------------------------------------------------------------------
    // Biweight midcorrelation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_biweight_midcorrelation_perfect() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = biweight_midcorrelation(&x.view(), &y.view(), None).expect("should succeed");
        assert!((r - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_biweight_midcorrelation_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = biweight_midcorrelation(&x.view(), &y.view(), None).expect("should succeed");
        assert!(r < -0.9);
    }

    #[test]
    fn test_biweight_midcorrelation_bounds() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let y = array![5.0, 3.0, 8.0, 1.0, 9.0, 200.0];
        let r = biweight_midcorrelation(&x.view(), &y.view(), None).expect("should succeed");
        assert!(r >= -1.0 && r <= 1.0);
    }

    #[test]
    fn test_biweight_midcorrelation_unrelated() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 1.0, 4.0, 2.0];
        let r = biweight_midcorrelation(&x.view(), &y.view(), None).expect("should succeed");
        assert!(r.abs() < 1.0);
    }

    #[test]
    fn test_biweight_midcorrelation_dim_mismatch() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        assert!(biweight_midcorrelation(&x.view(), &y.view(), None).is_err());
    }

    // -----------------------------------------------------------------------
    // MEstimatorWeight tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_huber_weight_function() {
        let w = MEstimatorWeight::Huber { c: 1.345 };
        assert!((w.weight(0.0) - 1.0).abs() < 1e-12);
        assert!((w.weight(1.0) - 1.0).abs() < 1e-12); // |1.0| < 1.345
        assert!((w.weight(2.0) - 1.345 / 2.0).abs() < 1e-12); // |2.0| > 1.345
        assert!((w.weight(-2.0) - 1.345 / 2.0).abs() < 1e-12); // symmetric
    }

    #[test]
    fn test_tukey_weight_function() {
        let w = MEstimatorWeight::TukeyBisquare { c: 4.685 };
        // At u = 0, weight = 1.0
        assert!((w.weight(0.0) - 1.0).abs() < 1e-12);
        // At u = c, weight should be 0 (boundary)
        assert!(w.weight(4.685).abs() < 1e-10);
        // Beyond c, weight = 0
        assert!(w.weight(5.0).abs() < 1e-12);
        // At moderate u, weight should be < 1 but > 0
        let w_mid = w.weight(2.0);
        assert!(w_mid > 0.0 && w_mid < 1.0);
    }

    #[test]
    fn test_weight_fn_defaults() {
        let h = MEstimatorWeight::huber_default();
        let t = MEstimatorWeight::tukey_default();
        match h {
            MEstimatorWeight::Huber { c } => assert!((c - 1.345).abs() < 1e-10),
            _ => panic!("Expected Huber"),
        }
        match t {
            MEstimatorWeight::TukeyBisquare { c } => assert!((c - 4.685).abs() < 1e-10),
            _ => panic!("Expected Tukey"),
        }
    }
}
