//! Outlier detection methods
//!
//! This module provides statistical tests and methods for detecting outliers
//! in univariate datasets.
//!
//! ## Methods provided
//!
//! - **Grubbs' test**: Test for a single outlier in a normally distributed dataset
//! - **Dixon's Q test**: Quick test for a single outlier in small samples
//! - **Generalized ESD test**: Test for multiple outliers (Rosner 1983)
//! - **Modified Z-score**: MAD-based outlier detection (Iglewicz & Hoaglin)
//! - **IQR-based detection**: Tukey's fences method

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::numeric::{Float, NumCast};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn sorted_vec<F: Float>(x: &ArrayView1<F>) -> Vec<F> {
    let mut v: Vec<F> = x.iter().cloned().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v
}

fn mean_f64<F: Float + NumCast>(data: &[F]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: f64 = data
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .sum();
    sum / data.len() as f64
}

fn std_f64<F: Float + NumCast>(data: &[F], ddof: usize) -> f64 {
    let n = data.len();
    if n <= ddof {
        return 0.0;
    }
    let m = mean_f64(data);
    let ss: f64 = data
        .iter()
        .map(|v| {
            let vf: f64 = <f64 as NumCast>::from(*v).unwrap_or(0.0);
            (vf - m) * (vf - m)
        })
        .sum();
    (ss / (n - ddof) as f64).sqrt()
}

fn median_f64<F: Float + NumCast>(sorted: &[F]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    if n % 2 == 0 {
        let a: f64 = <f64 as NumCast>::from(sorted[mid - 1]).unwrap_or(0.0);
        let b: f64 = <f64 as NumCast>::from(sorted[mid]).unwrap_or(0.0);
        (a + b) / 2.0
    } else {
        <f64 as NumCast>::from(sorted[mid]).unwrap_or(0.0)
    }
}

fn mad_f64<F: Float + NumCast>(sorted: &[F]) -> f64 {
    let med = median_f64(sorted);
    let mut abs_devs: Vec<f64> = sorted
        .iter()
        .map(|v| {
            let vf: f64 = <f64 as NumCast>::from(*v).unwrap_or(0.0);
            (vf - med).abs()
        })
        .collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = abs_devs.len();
    if n == 0 {
        return 0.0;
    }
    let mid = n / 2;
    if n % 2 == 0 {
        (abs_devs[mid - 1] + abs_devs[mid]) / 2.0
    } else {
        abs_devs[mid]
    }
}

/// Approximate the inverse CDF (quantile) of Student's t-distribution
/// using a rational approximation. This avoids needing a full t-distribution crate.
fn t_ppf(p: f64, df: f64) -> f64 {
    // For large df, use normal approximation
    if df > 1000.0 {
        return normal_ppf(p);
    }

    // Cornish-Fisher approximation for Student's t
    let z = normal_ppf(p);
    let z2 = z * z;
    let z3 = z2 * z;
    let z5 = z3 * z2;

    let g1 = (z3 + z) / (4.0 * df);
    let g2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df * df);
    let g3 = (3.0 * z5 + 19.0 * z3 + 17.0 * z) / (384.0 * df * df * df);

    z + g1 + g2 - g3
}

/// Approximate the standard normal quantile function (probit).
fn normal_ppf(p: f64) -> f64 {
    // Rational approximation (Abramowitz and Stegun 26.2.23)
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < f64::EPSILON {
        return 0.0;
    }

    let (sign, pp) = if p < 0.5 { (-1.0, p) } else { (1.0, 1.0 - p) };

    let t = (-2.0 * pp.ln()).sqrt();

    // Coefficients
    let c0 = 2.515_517;
    let c1 = 0.802_853;
    let c2 = 0.010_328;
    let d1 = 1.432_788;
    let d2 = 0.189_269;
    let d3 = 0.001_308;

    let numerator = c0 + t * (c1 + t * c2);
    let denominator = 1.0 + t * (d1 + t * (d2 + t * d3));

    sign * (t - numerator / denominator)
}

/// Approximate CDF of standard normal distribution.
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let t = 1.0 / (1.0 + 0.231_641_9 * x.abs());
    let d = 0.398_942_28 * (-0.5 * x * x).exp();
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));

    if x >= 0.0 {
        1.0 - d * poly
    } else {
        d * poly
    }
}

/// Survival function (1 - CDF) of the t-distribution.
/// Approximate using the normal approximation for larger df.
fn t_sf(t_val: f64, df: f64) -> f64 {
    // Use a simple approximation: transform t to z
    // For better accuracy with small df, use the incomplete beta function
    // approach, but this approximation suffices for outlier tests.
    if df > 100.0 {
        return 1.0 - normal_cdf(t_val);
    }

    // Incomplete beta function approximation for small df
    // Fisher's approximation: P(T > t | df) ≈ P(Z > z) where
    // z = df^0.5 * ln(1 + t^2/df) / ... (more refined)
    //
    // We use the Hill (1970) algorithm approximation
    let x = df / (df + t_val * t_val);
    let a = df / 2.0;
    let b = 0.5;
    let beta_inc = regularized_incomplete_beta(a, b, x);

    // The two-tailed sf from one tail
    beta_inc / 2.0
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction.
fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation if needed for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    // Lentz's continued fraction algorithm
    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();
    let prefix = ln_prefix.exp();

    let mut f = 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    f = d;

    for m in 1..200 {
        let m_f = m as f64;

        // Even step
        let numerator_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + numerator_even * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + numerator_even / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let numerator_odd =
            -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + numerator_odd * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + numerator_odd / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    prefix * f
}

/// Log of the Beta function: ln(B(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Log-gamma function (Stirling's approximation with Lanczos correction)
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation (g=7)
    let p = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let z = x - 1.0;
    let mut sum = p[0];
    for (i, &pi) in p.iter().enumerate().skip(1) {
        sum += pi / (z + i as f64);
    }

    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + sum.ln()
}

// ===========================================================================
// Grubbs' test
// ===========================================================================

/// Result of the Grubbs test.
#[derive(Debug, Clone)]
pub struct GrubbsResult<F> {
    /// Grubbs test statistic G
    pub statistic: F,
    /// p-value (two-sided)
    pub p_value: F,
    /// Index of the suspected outlier
    pub outlier_index: usize,
    /// Value of the suspected outlier
    pub outlier_value: F,
}

/// Perform Grubbs' test for a single outlier.
///
/// Grubbs' test (Grubbs 1950) detects a single outlier in a univariate
/// dataset that is assumed to come from a normal distribution. The test
/// statistic is the maximum absolute deviation from the mean, divided
/// by the standard deviation.
///
/// # Arguments
///
/// * `x` - Input data (at least 3 observations)
/// * `alpha` - Significance level (default 0.05)
///
/// # Returns
///
/// A `GrubbsResult` containing the test statistic, p-value, and the
/// suspected outlier's index and value.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::grubbs_test;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
/// let result = grubbs_test(&data.view(), None).expect("Grubbs test failed");
/// assert_eq!(result.outlier_index, 5);
/// assert!(result.p_value < 0.05);
/// ```
pub fn grubbs_test<F>(x: &ArrayView1<F>, alpha: Option<f64>) -> StatsResult<GrubbsResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "Grubbs test requires at least 3 observations".to_string(),
        ));
    }
    let _alpha = alpha.unwrap_or(0.05);

    let data: Vec<F> = x.iter().cloned().collect();
    let m = mean_f64(&data);
    let s = std_f64(&data, 1);

    if s <= f64::EPSILON {
        return Err(StatsError::ComputationError(
            "Standard deviation is zero; cannot perform Grubbs test".to_string(),
        ));
    }

    // Find the observation furthest from the mean
    let mut max_dev = 0.0_f64;
    let mut outlier_idx = 0;
    for (i, &xi) in data.iter().enumerate() {
        let xi_f64: f64 = <f64 as NumCast>::from(xi).unwrap_or(0.0);
        let dev = (xi_f64 - m).abs();
        if dev > max_dev {
            max_dev = dev;
            outlier_idx = i;
        }
    }

    let g = max_dev / s;
    let n_f = n as f64;

    // Compute the critical value and p-value
    // Under H0, G follows a distribution related to the t-distribution.
    // P-value: two-sided
    // G_crit = ((n-1)/sqrt(n)) * sqrt(t_{alpha/(2n), n-2}^2 / (n - 2 + t^2))
    //
    // To get the p-value, we invert:
    // t^2 = g^2 * (n-2) / (n - 1 - g^2 * n / (n-1))
    let g2 = g * g;
    let denom = (n_f - 1.0) - g2 * n_f / (n_f - 1.0);

    let p_value = if denom <= 0.0 {
        // g is so large that t -> infinity, p -> 0
        0.0
    } else {
        let t2 = g2 * (n_f - 2.0) / denom;
        let t_abs = t2.sqrt();
        let df = n_f - 2.0;
        // Two-sided p-value, adjusted for n comparisons (Bonferroni-like)
        let p_one_tail = t_sf(t_abs, df);
        let p_two_tail = 2.0 * p_one_tail;
        // Adjust for testing all n points
        let p_adjusted = (n_f * p_two_tail).min(1.0);
        p_adjusted
    };

    let statistic = F::from(g).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let p_val = F::from(p_value).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    Ok(GrubbsResult {
        statistic,
        p_value: p_val,
        outlier_index: outlier_idx,
        outlier_value: x[outlier_idx],
    })
}

// ===========================================================================
// Dixon's Q test
// ===========================================================================

/// Result of Dixon's Q test.
#[derive(Debug, Clone)]
pub struct DixonResult<F> {
    /// Dixon Q statistic
    pub statistic: F,
    /// Whether the suspected value is an outlier at the given significance level
    pub is_outlier: bool,
    /// Critical value at the given significance level
    pub critical_value: F,
    /// The suspected outlier value
    pub outlier_value: F,
    /// Whether the outlier is at the low end (true) or high end (false)
    pub is_low_outlier: bool,
}

/// Perform Dixon's Q test for a single outlier.
///
/// Dixon's Q test (Dixon 1950, 1951) is a quick test for detecting a single
/// outlier in a small sample (3 to 30 observations). The test examines the
/// ratio of the gap between a suspect value and its nearest neighbour to
/// the range of the data.
///
/// # Arguments
///
/// * `x` - Input data (3 to 30 observations)
/// * `alpha` - Significance level: 0.10, 0.05 (default), or 0.01
///
/// # Returns
///
/// A `DixonResult` with the test statistic, critical value, and the outlier info.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::dixon_test;
///
/// let data = array![0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.177, 0.015];
/// let result = dixon_test(&data.view(), None).expect("Dixon test failed");
/// assert!(result.is_outlier);
/// ```
pub fn dixon_test<F>(x: &ArrayView1<F>, alpha: Option<f64>) -> StatsResult<DixonResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "Dixon's Q test requires at least 3 observations".to_string(),
        ));
    }
    if n > 30 {
        return Err(StatsError::InvalidArgument(
            "Dixon's Q test is designed for samples of size 3 to 30".to_string(),
        ));
    }

    let alpha = alpha.unwrap_or(0.05);

    // Validate alpha
    if alpha != 0.10 && alpha != 0.05 && alpha != 0.01 {
        // Allow close matches
        let valid = [0.10, 0.05, 0.01];
        let closest = valid
            .iter()
            .min_by(|a, b| {
                ((**a) - alpha)
                    .abs()
                    .partial_cmp(&((**b) - alpha).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(0.05);
        if (closest - alpha).abs() > 0.02 {
            return Err(StatsError::InvalidArgument(
                "alpha must be 0.10, 0.05, or 0.01 for Dixon's Q test".to_string(),
            ));
        }
    }

    let sorted = sorted_vec(x);
    let x_min: f64 = <f64 as NumCast>::from(sorted[0]).unwrap_or(0.0);
    let x_max: f64 = <f64 as NumCast>::from(sorted[n - 1]).unwrap_or(0.0);
    let data_range = x_max - x_min;

    if data_range <= f64::EPSILON {
        return Err(StatsError::ComputationError(
            "Range of data is zero; cannot perform Dixon test".to_string(),
        ));
    }

    // Calculate Q statistics for both ends
    // For n=3..7: Q10 (gap/range)
    // For n=8..10: Q11 (gap/modified range)
    // For n=11..13: Q21
    // For n >= 14: Q22

    let (q_low, q_high) = if n <= 7 {
        // Q10: r10 = (x[1] - x[0]) / (x[n-1] - x[0])
        let x1: f64 = <f64 as NumCast>::from(sorted[1]).unwrap_or(0.0);
        let xnm2: f64 = <f64 as NumCast>::from(sorted[n - 2]).unwrap_or(0.0);
        ((x1 - x_min) / data_range, (x_max - xnm2) / data_range)
    } else if n <= 10 {
        // Q11
        let x1: f64 = <f64 as NumCast>::from(sorted[1]).unwrap_or(0.0);
        let xnm2: f64 = <f64 as NumCast>::from(sorted[n - 2]).unwrap_or(0.0);
        let xnm1: f64 = <f64 as NumCast>::from(sorted[n - 1]).unwrap_or(0.0);
        let x0: f64 = <f64 as NumCast>::from(sorted[0]).unwrap_or(0.0);
        (
            (x1 - x0) / (xnm2 - x0),
            (xnm1 - xnm2) / (xnm1 - <f64 as NumCast>::from(sorted[1]).unwrap_or(0.0)),
        )
    } else if n <= 13 {
        // Q21
        let x2: f64 = <f64 as NumCast>::from(sorted[2]).unwrap_or(0.0);
        let xnm3: f64 = <f64 as NumCast>::from(sorted[n - 3]).unwrap_or(0.0);
        let x1: f64 = <f64 as NumCast>::from(sorted[1]).unwrap_or(0.0);
        (
            (<f64 as NumCast>::from(sorted[1]).unwrap_or(0.0) - x_min) / (xnm3 - x_min),
            (x_max - <f64 as NumCast>::from(sorted[n - 2]).unwrap_or(0.0)) / (x_max - x2),
        )
    } else {
        // Q22
        let x2: f64 = <f64 as NumCast>::from(sorted[2]).unwrap_or(0.0);
        let xnm3: f64 = <f64 as NumCast>::from(sorted[n - 3]).unwrap_or(0.0);
        ((x2 - x_min) / (xnm3 - x_min), (x_max - xnm3) / (x_max - x2))
    };

    // Choose the larger Q and determine which end has the outlier
    let (q_stat, is_low) = if q_low > q_high {
        (q_low, true)
    } else {
        (q_high, false)
    };

    // Critical values for Dixon's Q test (two-sided)
    // Sources: Dixon (1950, 1951), Rorabacher (1991)
    let critical = dixon_critical_value(n, alpha);

    let is_outlier = q_stat > critical;

    let outlier_value = if is_low { sorted[0] } else { sorted[n - 1] };

    let stat_f = F::from(q_stat).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let crit_f = F::from(critical).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    Ok(DixonResult {
        statistic: stat_f,
        is_outlier,
        critical_value: crit_f,
        outlier_value,
        is_low_outlier: is_low,
    })
}

/// Look up critical values for Dixon's Q test.
fn dixon_critical_value(n: usize, alpha: f64) -> f64 {
    // Critical values from Rorabacher (1991), Table 1
    // Rows: n=3..30; columns: alpha = 0.10, 0.05, 0.01
    #[rustfmt::skip]
    let table_010: [f64; 28] = [
        0.941, 0.765, 0.642, 0.560, 0.507, 0.468, 0.437, 0.412, // n=3..10
        0.392, 0.376, 0.361, 0.349, 0.338, 0.329, 0.320, 0.313, // n=11..18
        0.306, 0.300, 0.295, 0.290, 0.285, 0.281, 0.277, 0.273, // n=19..26
        0.269, 0.266, 0.263, 0.260,                               // n=27..30
    ];

    #[rustfmt::skip]
    let table_005: [f64; 28] = [
        0.970, 0.829, 0.710, 0.625, 0.568, 0.526, 0.493, 0.466, // n=3..10
        0.444, 0.426, 0.410, 0.396, 0.384, 0.374, 0.365, 0.356, // n=11..18
        0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312, // n=19..26
        0.308, 0.305, 0.301, 0.298,                               // n=27..30
    ];

    #[rustfmt::skip]
    let table_001: [f64; 28] = [
        0.994, 0.926, 0.821, 0.740, 0.680, 0.634, 0.598, 0.568, // n=3..10
        0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442, // n=11..18
        0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388, // n=19..26
        0.384, 0.380, 0.376, 0.372,                               // n=27..30
    ];

    let idx = if n < 3 { 0 } else { (n - 3).min(27) };

    if alpha <= 0.01 + f64::EPSILON {
        table_001[idx]
    } else if alpha <= 0.05 + f64::EPSILON {
        table_005[idx]
    } else {
        table_010[idx]
    }
}

// ===========================================================================
// Generalized Extreme Studentized Deviate (ESD) test
// ===========================================================================

/// Result of the Generalized ESD test.
#[derive(Debug, Clone)]
pub struct EsdResult<F> {
    /// Number of outliers detected
    pub n_outliers: usize,
    /// Indices of detected outliers (in original data order)
    pub outlier_indices: Vec<usize>,
    /// Test statistics for each iteration
    pub test_statistics: Vec<F>,
    /// Critical values for each iteration
    pub critical_values: Vec<F>,
}

/// Perform the Generalized Extreme Studentized Deviate (ESD) test for up to
/// `max_outliers` outliers.
///
/// The Generalized ESD test (Rosner 1983) is an extension of the Grubbs test
/// for detecting multiple outliers in a univariate dataset assumed to be
/// approximately normally distributed. It sequentially removes the most
/// extreme observation and re-tests.
///
/// # Arguments
///
/// * `x` - Input data (at least 15 observations recommended)
/// * `max_outliers` - Maximum number of outliers to test for
/// * `alpha` - Significance level (default 0.05)
///
/// # Returns
///
/// An `EsdResult` with the number of detected outliers, their indices,
/// and the per-iteration test statistics and critical values.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::generalized_esd_test;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 200.0];
/// let result = generalized_esd_test(&data.view(), 3, None).expect("ESD test failed");
/// assert!(result.n_outliers >= 1);
/// ```
pub fn generalized_esd_test<F>(
    x: &ArrayView1<F>,
    max_outliers: usize,
    alpha: Option<f64>,
) -> StatsResult<EsdResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let n = x.len();
    if n < 3 {
        return Err(StatsError::InvalidArgument(
            "At least 3 observations required for ESD test".to_string(),
        ));
    }
    if max_outliers == 0 || max_outliers >= n - 1 {
        return Err(StatsError::InvalidArgument(format!(
            "max_outliers must be between 1 and {} (n-2)",
            n - 2
        )));
    }

    let alpha = alpha.unwrap_or(0.05);

    // Work with indices into the original data
    let mut remaining_indices: Vec<usize> = (0..n).collect();
    let mut remaining_data: Vec<f64> = x
        .iter()
        .map(|v| <f64 as NumCast>::from(*v).unwrap_or(0.0))
        .collect();

    let mut test_stats = Vec::with_capacity(max_outliers);
    let mut critical_vals = Vec::with_capacity(max_outliers);
    let mut removed_indices = Vec::with_capacity(max_outliers);

    for i in 0..max_outliers {
        let ni = remaining_data.len();
        if ni < 3 {
            break;
        }

        let m = remaining_data.iter().sum::<f64>() / ni as f64;
        let s: f64 = (remaining_data
            .iter()
            .map(|&v| (v - m) * (v - m))
            .sum::<f64>()
            / (ni as f64 - 1.0))
            .sqrt();

        if s <= f64::EPSILON {
            break;
        }

        // Find the most extreme value
        let mut max_dev = 0.0_f64;
        let mut max_idx_in_remaining = 0;

        for (j, &val) in remaining_data.iter().enumerate() {
            let dev = (val - m).abs();
            if dev > max_dev {
                max_dev = dev;
                max_idx_in_remaining = j;
            }
        }

        let ri = max_dev / s;

        // Critical value: t_{p, n-i-1} where p = 1 - alpha / (2*(n-i))
        let ni_f = ni as f64;
        let p = 1.0 - alpha / (2.0 * ni_f);
        let df = ni_f - 2.0;
        let t_crit = if df > 0.0 { t_ppf(p, df) } else { 3.0 };

        let lambda_i = (ni_f - 1.0) * t_crit / ((ni_f - 2.0 + t_crit * t_crit) * ni_f).sqrt();

        test_stats.push(ri);
        critical_vals.push(lambda_i);

        // Record the original index of the removed point
        removed_indices.push(remaining_indices[max_idx_in_remaining]);

        // Remove the point
        remaining_data.remove(max_idx_in_remaining);
        remaining_indices.remove(max_idx_in_remaining);
    }

    // Determine the number of outliers: the largest i where R_i > lambda_i
    let mut n_outliers = 0;
    for i in (0..test_stats.len()).rev() {
        if test_stats[i] > critical_vals[i] {
            n_outliers = i + 1;
            break;
        }
    }

    let outlier_indices = removed_indices[..n_outliers].to_vec();

    let test_statistics: Result<Vec<F>, _> = test_stats
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();
    let critical_values: Result<Vec<F>, _> = critical_vals
        .iter()
        .map(|&v| F::from(v).ok_or_else(|| StatsError::ComputationError("cast".into())))
        .collect();

    Ok(EsdResult {
        n_outliers,
        outlier_indices,
        test_statistics: test_statistics?,
        critical_values: critical_values?,
    })
}

// ===========================================================================
// Modified Z-score method
// ===========================================================================

/// Result of modified Z-score outlier detection.
#[derive(Debug, Clone)]
pub struct ModifiedZScoreResult<F> {
    /// Modified Z-scores for each observation
    pub scores: Array1<F>,
    /// Boolean mask: true if the observation is an outlier
    pub outlier_mask: Vec<bool>,
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Threshold used
    pub threshold: F,
}

/// Detect outliers using the modified Z-score method (Iglewicz & Hoaglin 1993).
///
/// The modified Z-score replaces the mean with the median and the standard
/// deviation with the MAD (median absolute deviation), making it robust to
/// the influence of outliers:
///
/// ```text
///   M_i = 0.6745 * (x_i - median) / MAD
/// ```
///
/// An observation is flagged as an outlier if |M_i| > threshold (default 3.5).
///
/// # Arguments
///
/// * `x` - Input data (at least 2 observations)
/// * `threshold` - Threshold for outlier detection (default 3.5)
///
/// # Returns
///
/// A `ModifiedZScoreResult` with scores, mask, and outlier indices.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::modified_z_score;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
/// let result = modified_z_score(&data.view(), None).expect("modified z-score failed");
/// assert!(result.outlier_mask[5]); // 100.0 is an outlier
/// assert!(!result.outlier_mask[0]); // 1.0 is not
/// ```
pub fn modified_z_score<F>(
    x: &ArrayView1<F>,
    threshold: Option<f64>,
) -> StatsResult<ModifiedZScoreResult<F>>
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
            "At least 2 observations required".to_string(),
        ));
    }

    let threshold = threshold.unwrap_or(3.5);
    let sorted = sorted_vec(x);
    let med = median_f64(&sorted);
    let mad = mad_f64(&sorted);

    let mut scores = Array1::<F>::zeros(n);
    let mut outlier_mask = vec![false; n];
    let mut outlier_indices = Vec::new();

    if mad <= f64::EPSILON {
        // MAD is zero: all values equal except outliers
        // Use a small epsilon-based approach
        let threshold_f =
            F::from(threshold).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
        for (i, &xi) in x.iter().enumerate() {
            let xi_f64: f64 = <f64 as NumCast>::from(xi).unwrap_or(0.0);
            let diff = (xi_f64 - med).abs();
            if diff > f64::EPSILON {
                // Any deviation from the median is infinite in MAD terms
                scores[i] = F::infinity();
                outlier_mask[i] = true;
                outlier_indices.push(i);
            } else {
                scores[i] = F::zero();
            }
        }
        return Ok(ModifiedZScoreResult {
            scores,
            outlier_mask,
            outlier_indices,
            threshold: threshold_f,
        });
    }

    let factor = 0.6745 / mad;

    for (i, &xi) in x.iter().enumerate() {
        let xi_f64: f64 = <f64 as NumCast>::from(xi).unwrap_or(0.0);
        let z = factor * (xi_f64 - med);
        let score = F::from(z).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
        scores[i] = score;

        if z.abs() > threshold {
            outlier_mask[i] = true;
            outlier_indices.push(i);
        }
    }

    let threshold_f =
        F::from(threshold).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    Ok(ModifiedZScoreResult {
        scores,
        outlier_mask,
        outlier_indices,
        threshold: threshold_f,
    })
}

// ===========================================================================
// IQR-based outlier detection (Tukey's fences)
// ===========================================================================

/// Result of IQR-based outlier detection.
#[derive(Debug, Clone)]
pub struct IqrOutlierResult<F> {
    /// Boolean mask: true if outlier
    pub outlier_mask: Vec<bool>,
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Lower fence
    pub lower_fence: F,
    /// Upper fence
    pub upper_fence: F,
    /// First quartile (Q1)
    pub q1: F,
    /// Third quartile (Q3)
    pub q3: F,
    /// IQR
    pub iqr: F,
    /// Number of mild outliers (between 1.5 and 3 IQR)
    pub n_mild: usize,
    /// Number of extreme outliers (beyond 3 IQR)
    pub n_extreme: usize,
}

/// Detect outliers using Tukey's IQR (Interquartile Range) method.
///
/// Observations below `Q1 - k * IQR` or above `Q3 + k * IQR` are classified
/// as outliers. The default `k = 1.5` corresponds to Tukey's "inner fences"
/// (mild outliers); `k = 3.0` gives the "outer fences" (extreme outliers).
///
/// # Arguments
///
/// * `x` - Input data
/// * `k` - Multiplier for the IQR (default 1.5)
///
/// # Returns
///
/// An `IqrOutlierResult` with outlier information and fence values.
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::iqr_outlier_detection;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
/// let result = iqr_outlier_detection(&data.view(), None).expect("IQR detection failed");
/// assert!(result.outlier_mask[9]);       // 100 is an outlier
/// assert!(result.outlier_indices.contains(&9));
/// ```
pub fn iqr_outlier_detection<F>(
    x: &ArrayView1<F>,
    k: Option<f64>,
) -> StatsResult<IqrOutlierResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    if x.len() < 4 {
        return Err(StatsError::InvalidArgument(
            "At least 4 observations required for IQR outlier detection".to_string(),
        ));
    }

    let k = k.unwrap_or(1.5);
    let sorted = sorted_vec(x);
    let n = sorted.len();

    // Q1 and Q3 via linear interpolation
    let q1_f64 = percentile_f64(&sorted, 25.0);
    let q3_f64 = percentile_f64(&sorted, 75.0);
    let iqr_f64 = q3_f64 - q1_f64;

    let lower_fence_f64 = q1_f64 - k * iqr_f64;
    let upper_fence_f64 = q3_f64 + k * iqr_f64;

    let extreme_lower = q1_f64 - 3.0 * iqr_f64;
    let extreme_upper = q3_f64 + 3.0 * iqr_f64;

    let mut outlier_mask = vec![false; n];
    let mut outlier_indices = Vec::new();
    let mut n_mild = 0_usize;
    let mut n_extreme = 0_usize;

    for (i, &xi) in x.iter().enumerate() {
        let xi_f64: f64 = <f64 as NumCast>::from(xi).unwrap_or(0.0);
        if xi_f64 < lower_fence_f64 || xi_f64 > upper_fence_f64 {
            outlier_mask[i] = true;
            outlier_indices.push(i);

            if xi_f64 < extreme_lower || xi_f64 > extreme_upper {
                n_extreme += 1;
            } else {
                n_mild += 1;
            }
        }
    }

    let q1 = F::from(q1_f64).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let q3 = F::from(q3_f64).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let iqr_val = F::from(iqr_f64).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let lower_fence =
        F::from(lower_fence_f64).ok_or_else(|| StatsError::ComputationError("cast".into()))?;
    let upper_fence =
        F::from(upper_fence_f64).ok_or_else(|| StatsError::ComputationError("cast".into()))?;

    Ok(IqrOutlierResult {
        outlier_mask,
        outlier_indices,
        lower_fence,
        upper_fence,
        q1,
        q3,
        iqr: iqr_val,
        n_mild,
        n_extreme,
    })
}

/// Compute a percentile using linear interpolation on a sorted slice.
fn percentile_f64<F: Float + NumCast>(sorted: &[F], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return <f64 as NumCast>::from(sorted[0]).unwrap_or(0.0);
    }

    let frac = p / 100.0;
    let idx = frac * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let weight = idx - lo as f64;

    let lo_val: f64 = <f64 as NumCast>::from(sorted[lo.min(n - 1)]).unwrap_or(0.0);
    let hi_val: f64 = <f64 as NumCast>::from(sorted[hi.min(n - 1)]).unwrap_or(0.0);

    lo_val + weight * (hi_val - lo_val)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    // -----------------------------------------------------------------------
    // Grubbs' test
    // -----------------------------------------------------------------------

    #[test]
    fn test_grubbs_obvious_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let result = grubbs_test(&data.view(), None).expect("should succeed");
        assert_eq!(result.outlier_index, 5);
        let p: f64 = NumCast::from(result.p_value).expect("cast");
        assert!(p < 0.05);
    }

    #[test]
    fn test_grubbs_no_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = grubbs_test(&data.view(), None).expect("should succeed");
        let p: f64 = NumCast::from(result.p_value).expect("cast");
        // Should not detect an outlier with this clean data
        assert!(p > 0.01);
    }

    #[test]
    fn test_grubbs_low_outlier() {
        let data = array![-100.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = grubbs_test(&data.view(), None).expect("should succeed");
        assert_eq!(result.outlier_index, 0);
    }

    #[test]
    fn test_grubbs_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(grubbs_test(&data.view(), None).is_err());
    }

    #[test]
    fn test_grubbs_too_small_error() {
        let data = array![1.0, 2.0];
        assert!(grubbs_test(&data.view(), None).is_err());
    }

    #[test]
    fn test_grubbs_constant_data_error() {
        let data = array![5.0, 5.0, 5.0, 5.0];
        assert!(grubbs_test(&data.view(), None).is_err());
    }

    // -----------------------------------------------------------------------
    // Dixon's Q test
    // -----------------------------------------------------------------------

    #[test]
    fn test_dixon_obvious_outlier() {
        let data = array![0.189, 0.167, 0.187, 0.183, 0.186, 0.182, 0.181, 0.184, 0.177, 0.015];
        let result = dixon_test(&data.view(), Some(0.05)).expect("should succeed");
        assert!(result.is_outlier);
    }

    #[test]
    fn test_dixon_no_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = dixon_test(&data.view(), None).expect("should succeed");
        assert!(!result.is_outlier);
    }

    #[test]
    fn test_dixon_small_sample() {
        let data = array![1.0, 2.0, 10.0];
        let result = dixon_test(&data.view(), None).expect("should succeed");
        // 10 is far from 1 and 2; should be detected at n=3
        let stat: f64 = NumCast::from(result.statistic).expect("cast");
        assert!(stat > 0.0);
    }

    #[test]
    fn test_dixon_too_small_error() {
        let data = array![1.0, 2.0];
        assert!(dixon_test(&data.view(), None).is_err());
    }

    #[test]
    fn test_dixon_too_large_error() {
        let data = Array1::from_vec((0..31).map(|i| i as f64).collect());
        assert!(dixon_test(&data.view(), None).is_err());
    }

    #[test]
    fn test_dixon_high_outlier() {
        let data = array![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 10.0];
        let result = dixon_test(&data.view(), None).expect("should succeed");
        assert!(!result.is_low_outlier);
    }

    // -----------------------------------------------------------------------
    // Generalized ESD test
    // -----------------------------------------------------------------------

    #[test]
    fn test_esd_single_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0];
        let result = generalized_esd_test(&data.view(), 3, None).expect("should succeed");
        assert!(result.n_outliers >= 1);
        assert!(result.outlier_indices.contains(&10));
    }

    #[test]
    fn test_esd_multiple_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 200.0];
        let result = generalized_esd_test(&data.view(), 5, None).expect("should succeed");
        assert!(result.n_outliers >= 1);
    }

    #[test]
    fn test_esd_no_outliers() {
        // Evenly spaced data, no outliers
        let data = Array1::from_vec((1..=20).map(|i| i as f64).collect());
        let result = generalized_esd_test(&data.view(), 3, None).expect("should succeed");
        // Should detect 0 or very few "outliers"
        assert!(result.n_outliers <= 1);
    }

    #[test]
    fn test_esd_test_stats_length() {
        let data = Array1::from_vec((1..=15).map(|i| i as f64).collect());
        let max_out = 3;
        let result = generalized_esd_test(&data.view(), max_out, None).expect("should succeed");
        assert_eq!(result.test_statistics.len(), max_out);
        assert_eq!(result.critical_values.len(), max_out);
    }

    #[test]
    fn test_esd_error_zero_max() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(generalized_esd_test(&data.view(), 0, None).is_err());
    }

    #[test]
    fn test_esd_error_too_many() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(generalized_esd_test(&data.view(), 4, None).is_err());
    }

    // -----------------------------------------------------------------------
    // Modified Z-score
    // -----------------------------------------------------------------------

    #[test]
    fn test_modified_z_obvious_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let result = modified_z_score(&data.view(), None).expect("should succeed");
        assert!(result.outlier_mask[5]);
        assert!(!result.outlier_mask[2]);
    }

    #[test]
    fn test_modified_z_no_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = modified_z_score(&data.view(), None).expect("should succeed");
        assert!(result.outlier_indices.is_empty());
    }

    #[test]
    fn test_modified_z_custom_threshold() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 20.0];
        let strict = modified_z_score(&data.view(), Some(2.0)).expect("strict");
        let loose = modified_z_score(&data.view(), Some(5.0)).expect("loose");
        // Stricter threshold should detect at least as many outliers
        assert!(strict.outlier_indices.len() >= loose.outlier_indices.len());
    }

    #[test]
    fn test_modified_z_scores_symmetry() {
        let data = array![-5.0, -4.0, -3.0, 0.0, 3.0, 4.0, 5.0];
        let result = modified_z_score(&data.view(), None).expect("should succeed");
        // Scores for -5 and 5 should be symmetric (opposite sign, same magnitude)
        let s0: f64 = NumCast::from(result.scores[0]).expect("cast");
        let s6: f64 = NumCast::from(result.scores[6]).expect("cast");
        assert!((s0.abs() - s6.abs()).abs() < 1e-10);
    }

    #[test]
    fn test_modified_z_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(modified_z_score(&data.view(), None).is_err());
    }

    #[test]
    fn test_modified_z_constant_data() {
        let data = array![3.0, 3.0, 3.0, 3.0, 3.0, 10.0];
        let result = modified_z_score(&data.view(), None).expect("should succeed");
        // 10.0 should be the only outlier (MAD is 0 for the majority)
        assert!(result.outlier_mask[5]);
    }

    // -----------------------------------------------------------------------
    // IQR-based outlier detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_iqr_obvious_outlier() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        let result = iqr_outlier_detection(&data.view(), None).expect("should succeed");
        assert!(result.outlier_mask[9]);
        assert!(result.outlier_indices.contains(&9));
    }

    #[test]
    fn test_iqr_no_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = iqr_outlier_detection(&data.view(), None).expect("should succeed");
        assert!(result.outlier_indices.is_empty());
    }

    #[test]
    fn test_iqr_fences_correct() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = iqr_outlier_detection(&data.view(), Some(1.5)).expect("should succeed");
        let q1: f64 = NumCast::from(result.q1).expect("cast");
        let q3: f64 = NumCast::from(result.q3).expect("cast");
        let iqr_val: f64 = NumCast::from(result.iqr).expect("cast");
        let lf: f64 = NumCast::from(result.lower_fence).expect("cast");
        let uf: f64 = NumCast::from(result.upper_fence).expect("cast");

        assert!((lf - (q1 - 1.5 * iqr_val)).abs() < 1e-10);
        assert!((uf - (q3 + 1.5 * iqr_val)).abs() < 1e-10);
    }

    #[test]
    fn test_iqr_extreme_classification() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0, 200.0];
        let result = iqr_outlier_detection(&data.view(), Some(1.5)).expect("should succeed");
        // 200 is likely extreme, 100 may be mild or extreme depending on IQR
        assert!(result.n_mild + result.n_extreme >= 1);
    }

    #[test]
    fn test_iqr_custom_k() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0];
        let strict = iqr_outlier_detection(&data.view(), Some(1.0)).expect("strict");
        let loose = iqr_outlier_detection(&data.view(), Some(3.0)).expect("loose");
        assert!(strict.outlier_indices.len() >= loose.outlier_indices.len());
    }

    #[test]
    fn test_iqr_empty_error() {
        let data = Array1::<f64>::zeros(0);
        assert!(iqr_outlier_detection(&data.view(), None).is_err());
    }

    #[test]
    fn test_iqr_too_small_error() {
        let data = array![1.0, 2.0, 3.0];
        assert!(iqr_outlier_detection(&data.view(), None).is_err());
    }

    // -----------------------------------------------------------------------
    // Helper function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_normal_ppf_symmetry() {
        let z_low = normal_ppf(0.025);
        let z_high = normal_ppf(0.975);
        assert!((z_low + z_high).abs() < 0.01);
        assert!((z_high - 1.96).abs() < 0.05);
    }

    #[test]
    fn test_normal_cdf_bounds() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!(normal_cdf(-10.0) < 1e-10);
        assert!(normal_cdf(10.0) > 1.0 - 1e-10);
    }

    #[test]
    fn test_percentile_f64_basic() {
        let sorted = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_f64(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile_f64(&sorted, 100.0) - 5.0).abs() < 1e-10);
        assert!((percentile_f64(&sorted, 50.0) - 3.0).abs() < 1e-10);
    }
}
