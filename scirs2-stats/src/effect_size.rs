//! Effect size measures for statistical analysis
//!
//! Effect sizes quantify the magnitude of a phenomenon, independent of sample size.
//! They are essential for meta-analysis, power analysis, and reporting practical
//! significance alongside statistical significance.
//!
//! ## Measures Provided
//!
//! ### Standardized Mean Differences
//! - **Cohen's d** - Standardized difference between two group means (pooled SD)
//! - **Hedges' g** - Bias-corrected version of Cohen's d for small samples
//! - **Glass's delta** - Uses only the control group's SD as denominator
//!
//! ### Correlation-based
//! - **Point-biserial correlation** - Correlation between a binary and continuous variable
//!
//! ### Variance-explained
//! - **Eta-squared** - Proportion of variance explained (ANOVA)
//! - **Partial eta-squared** - Proportion of variance explained controlling for other factors
//! - **Omega-squared** - Less biased version of eta-squared
//!
//! ### Confidence Intervals
//! - **CI for Cohen's d** - Noncentral t-distribution based
//! - **CI for correlation** - Fisher z-transformation based

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::ArrayView1;
use scirs2_core::numeric::{Float, NumCast};

/// Result containing an effect size estimate with optional confidence interval
#[derive(Debug, Clone)]
pub struct EffectSizeResult<F: Float> {
    /// Point estimate of the effect size
    pub estimate: F,
    /// Name of the effect size measure
    pub measure: String,
    /// Lower bound of the confidence interval (if computed)
    pub ci_lower: Option<F>,
    /// Upper bound of the confidence interval (if computed)
    pub ci_upper: Option<F>,
    /// Confidence level for the interval
    pub confidence_level: Option<F>,
}

// ========================================================================
// Helper: standard normal quantile (inverse CDF)
// ========================================================================

/// Inverse of the standard normal CDF (quantile function) using
/// the rational approximation of Beasley & Springer (1977) / Moro (1995).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Standard normal CDF (Abramowitz & Stegun approximation)
fn normal_cdf_f64(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * abs_x);
    let d = 0.39894228040143268 * (-0.5 * x * x).exp();
    let p = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 {
        1.0 - d * p
    } else {
        d * p
    }
}

// ========================================================================
// Cohen's d
// ========================================================================

/// Computes Cohen's d, the standardized mean difference between two groups.
///
/// d = (mean_x - mean_y) / pooled_sd
///
/// where pooled_sd = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
///
/// # Returns
///
/// An `EffectSizeResult` with Cohen's d.
///
/// # Interpretation
///
/// |d| < 0.2 = small, |d| ~ 0.5 = medium, |d| > 0.8 = large (Cohen, 1988)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::effect_size::cohens_d;
///
/// let x = array![5.0, 6.0, 7.0, 8.0, 9.0];
/// let y = array![3.0, 4.0, 5.0, 6.0, 7.0];
///
/// let result = cohens_d(&x.view(), &y.view()).expect("Cohen's d failed");
/// assert!(result.estimate > 0.0); // x has larger mean
/// ```
pub fn cohens_d<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.len() < 2 || y.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "Both samples must have at least 2 observations for Cohen's d".to_string(),
        ));
    }

    let (mean_x, var_x, n1) = sample_stats(x)?;
    let (mean_y, var_y, n2) = sample_stats(y)?;

    let pooled_var = ((n1 - 1.0) * var_x + (n2 - 1.0) * var_y) / (n1 + n2 - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd < 1e-30 {
        return Err(StatsError::ComputationError(
            "Pooled standard deviation is zero; Cohen's d is undefined".to_string(),
        ));
    }

    let d = (mean_x - mean_y) / pooled_sd;

    Ok(EffectSizeResult {
        estimate: F::from(d).unwrap_or(F::zero()),
        measure: "Cohen's d".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Hedges' g (bias-corrected Cohen's d)
// ========================================================================

/// Computes Hedges' g, a bias-corrected version of Cohen's d.
///
/// g = d * J, where J = 1 - 3/(4*(n1+n2-2) - 1) is the correction factor.
///
/// For small samples (n < 20), Cohen's d has a slight upward bias.
/// Hedges' g corrects for this.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
///
/// # Returns
///
/// An `EffectSizeResult` with Hedges' g.
pub fn hedges_g<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if x.len() < 2 || y.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "Both samples must have at least 2 observations for Hedges' g".to_string(),
        ));
    }

    let d_result = cohens_d(x, y)?;
    let d = <f64 as NumCast>::from(d_result.estimate).unwrap_or(0.0);
    let df = (x.len() + y.len() - 2) as f64;

    // Hedges' correction factor J
    let j = 1.0 - 3.0 / (4.0 * df - 1.0);
    let g = d * j;

    Ok(EffectSizeResult {
        estimate: F::from(g).unwrap_or(F::zero()),
        measure: "Hedges' g".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Glass's delta
// ========================================================================

/// Computes Glass's delta, using the control group's standard deviation.
///
/// delta = (mean_treatment - mean_control) / sd_control
///
/// This is appropriate when the variances of the two groups are not expected
/// to be equal, and there is a clear control group.
///
/// # Arguments
///
/// * `treatment` - Treatment group sample
/// * `control` - Control group sample (its SD is used as the denominator)
///
/// # Returns
///
/// An `EffectSizeResult` with Glass's delta.
pub fn glass_delta<F>(
    treatment: &ArrayView1<F>,
    control: &ArrayView1<F>,
) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if treatment.len() < 2 || control.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "Both samples must have at least 2 observations for Glass's delta".to_string(),
        ));
    }

    let (mean_t, _, _) = sample_stats(treatment)?;
    let (mean_c, var_c, _) = sample_stats(control)?;
    let sd_c = var_c.sqrt();

    if sd_c < 1e-30 {
        return Err(StatsError::ComputationError(
            "Control group standard deviation is zero; Glass's delta is undefined".to_string(),
        ));
    }

    let delta = (mean_t - mean_c) / sd_c;

    Ok(EffectSizeResult {
        estimate: F::from(delta).unwrap_or(F::zero()),
        measure: "Glass's delta".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Point-biserial correlation
// ========================================================================

/// Computes the point-biserial correlation between a binary variable and a continuous variable.
///
/// r_pb = (M1 - M0) / Sp * sqrt(n0 * n1 / n^2)
///
/// where M1, M0 are means of the continuous variable for the two groups,
/// Sp is the pooled standard deviation, and n0, n1 are the group sizes.
///
/// # Arguments
///
/// * `binary` - Binary group labels (each element should be 0 or 1)
/// * `continuous` - Continuous variable values
///
/// # Returns
///
/// An `EffectSizeResult` with the point-biserial correlation in [-1, 1].
pub fn point_biserial_correlation<F>(
    binary: &ArrayView1<F>,
    continuous: &ArrayView1<F>,
) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if binary.len() != continuous.len() {
        return Err(StatsError::DimensionMismatch(
            "Binary and continuous arrays must have equal length".to_string(),
        ));
    }
    if binary.len() < 3 {
        return Err(StatsError::InvalidArgument(
            "At least 3 observations required for point-biserial correlation".to_string(),
        ));
    }

    let n = binary.len();
    let nf = n as f64;

    // Separate into two groups
    let mut group0: Vec<f64> = Vec::new();
    let mut group1: Vec<f64> = Vec::new();

    for i in 0..n {
        let b = <f64 as NumCast>::from(binary[i]).unwrap_or(0.0);
        let c = <f64 as NumCast>::from(continuous[i]).unwrap_or(0.0);
        if (b - 0.0).abs() < 1e-10 {
            group0.push(c);
        } else if (b - 1.0).abs() < 1e-10 {
            group1.push(c);
        } else {
            return Err(StatsError::InvalidArgument(format!(
                "Binary variable must contain only 0 and 1, got {} at index {}",
                b, i
            )));
        }
    }

    if group0.is_empty() || group1.is_empty() {
        return Err(StatsError::InvalidArgument(
            "Both groups must have at least one observation".to_string(),
        ));
    }

    let n0 = group0.len() as f64;
    let n1 = group1.len() as f64;
    let mean0: f64 = group0.iter().sum::<f64>() / n0;
    let mean1: f64 = group1.iter().sum::<f64>() / n1;

    // Overall standard deviation (sample)
    let overall_mean: f64 = continuous
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .sum::<f64>()
        / nf;
    let overall_var: f64 = continuous
        .iter()
        .map(|&v| {
            let vf = <f64 as NumCast>::from(v).unwrap_or(0.0);
            (vf - overall_mean) * (vf - overall_mean)
        })
        .sum::<f64>()
        / (nf - 1.0);
    let overall_sd = overall_var.sqrt();

    if overall_sd < 1e-30 {
        return Err(StatsError::ComputationError(
            "Standard deviation is zero; point-biserial correlation is undefined".to_string(),
        ));
    }

    let rpb = (mean1 - mean0) / overall_sd * (n0 * n1).sqrt() / nf;

    Ok(EffectSizeResult {
        estimate: F::from(rpb).unwrap_or(F::zero()),
        measure: "Point-biserial correlation".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Eta-squared
// ========================================================================

/// Computes eta-squared from ANOVA-style sums of squares.
///
/// eta^2 = SS_between / SS_total
///
/// This is the proportion of total variance explained by the grouping factor.
///
/// # Arguments
///
/// * `groups` - Slice of array views, one per group
///
/// # Returns
///
/// An `EffectSizeResult` with eta-squared in [0, 1].
///
/// # Interpretation
///
/// 0.01 = small, 0.06 = medium, 0.14 = large (Cohen, 1988)
pub fn eta_squared<F>(groups: &[ArrayView1<F>]) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if groups.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 groups required for eta-squared".to_string(),
        ));
    }
    for (i, g) in groups.iter().enumerate() {
        if g.is_empty() {
            return Err(StatsError::InvalidArgument(format!("Group {} is empty", i)));
        }
    }

    let (ss_between, ss_total) = compute_ss(groups)?;

    if ss_total < 1e-30 {
        return Err(StatsError::ComputationError(
            "Total sum of squares is zero; eta-squared is undefined".to_string(),
        ));
    }

    let eta2 = ss_between / ss_total;

    Ok(EffectSizeResult {
        estimate: F::from(eta2).unwrap_or(F::zero()),
        measure: "Eta-squared".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Partial eta-squared
// ========================================================================

/// Computes partial eta-squared.
///
/// partial_eta^2 = SS_effect / (SS_effect + SS_error)
///
/// This is the proportion of variance attributable to the factor,
/// after removing variance attributable to other factors.
///
/// # Arguments
///
/// * `ss_effect` - Sum of squares for the effect of interest
/// * `ss_error` - Sum of squares for the error (residual)
///
/// # Returns
///
/// An `EffectSizeResult` with partial eta-squared in [0, 1].
pub fn partial_eta_squared<F>(ss_effect: F, ss_error: F) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    let ef = <f64 as NumCast>::from(ss_effect).unwrap_or(0.0);
    let er = <f64 as NumCast>::from(ss_error).unwrap_or(0.0);

    if ef < 0.0 || er < 0.0 {
        return Err(StatsError::InvalidArgument(
            "Sums of squares must be non-negative".to_string(),
        ));
    }
    let denom = ef + er;
    if denom < 1e-30 {
        return Err(StatsError::ComputationError(
            "SS_effect + SS_error is zero; partial eta-squared is undefined".to_string(),
        ));
    }

    let partial_eta2 = ef / denom;

    Ok(EffectSizeResult {
        estimate: F::from(partial_eta2).unwrap_or(F::zero()),
        measure: "Partial eta-squared".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Omega-squared
// ========================================================================

/// Computes omega-squared, a less biased alternative to eta-squared.
///
/// omega^2 = (SS_between - df_between * MS_error) / (SS_total + MS_error)
///
/// where MS_error = SS_within / df_within.
///
/// # Arguments
///
/// * `groups` - Slice of array views, one per group
///
/// # Returns
///
/// An `EffectSizeResult` with omega-squared.
pub fn omega_squared<F>(groups: &[ArrayView1<F>]) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    if groups.len() < 2 {
        return Err(StatsError::InvalidArgument(
            "At least 2 groups required for omega-squared".to_string(),
        ));
    }
    for (i, g) in groups.iter().enumerate() {
        if g.is_empty() {
            return Err(StatsError::InvalidArgument(format!("Group {} is empty", i)));
        }
    }

    let k = groups.len();
    let n_total: usize = groups.iter().map(|g| g.len()).sum();

    let (ss_between, ss_total) = compute_ss(groups)?;
    let ss_within = ss_total - ss_between;

    let df_between = (k - 1) as f64;
    let df_within = (n_total - k) as f64;

    if df_within < 1.0 {
        return Err(StatsError::InvalidArgument(
            "Insufficient degrees of freedom for omega-squared".to_string(),
        ));
    }

    let ms_error = ss_within / df_within;
    let denom = ss_total + ms_error;

    if denom < 1e-30 {
        return Err(StatsError::ComputationError(
            "Denominator is zero; omega-squared is undefined".to_string(),
        ));
    }

    let omega2 = (ss_between - df_between * ms_error) / denom;
    // omega-squared can be negative (set to 0 if so)
    let omega2 = omega2.max(0.0);

    Ok(EffectSizeResult {
        estimate: F::from(omega2).unwrap_or(F::zero()),
        measure: "Omega-squared".to_string(),
        ci_lower: None,
        ci_upper: None,
        confidence_level: None,
    })
}

// ========================================================================
// Confidence interval for Cohen's d
// ========================================================================

/// Computes Cohen's d with a confidence interval.
///
/// The CI is based on the noncentral t-distribution approximation:
/// SE(d) ~ sqrt(1/n1 + 1/n2 + d^2/(2*(n1+n2)))
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample
/// * `confidence_level` - Confidence level (e.g., 0.95 for 95% CI)
///
/// # Returns
///
/// An `EffectSizeResult` with Cohen's d and confidence interval bounds.
pub fn cohens_d_ci<F>(
    x: &ArrayView1<F>,
    y: &ArrayView1<F>,
    confidence_level: F,
) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + std::iter::Sum<F> + NumCast + std::fmt::Display,
{
    let cl = <f64 as NumCast>::from(confidence_level).unwrap_or(0.95);
    if cl <= 0.0 || cl >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "confidence_level must be in (0, 1), got {}",
            cl
        )));
    }

    let d_result = cohens_d(x, y)?;
    let d = <f64 as NumCast>::from(d_result.estimate).unwrap_or(0.0);
    let n1 = x.len() as f64;
    let n2 = y.len() as f64;

    // Approximate SE of d (Hedges & Olkin, 1985)
    let se = (1.0 / n1 + 1.0 / n2 + d * d / (2.0 * (n1 + n2))).sqrt();

    let alpha = 1.0 - cl;
    let z = normal_quantile(1.0 - alpha / 2.0);

    let lower = d - z * se;
    let upper = d + z * se;

    Ok(EffectSizeResult {
        estimate: F::from(d).unwrap_or(F::zero()),
        measure: "Cohen's d".to_string(),
        ci_lower: Some(F::from(lower).unwrap_or(F::zero())),
        ci_upper: Some(F::from(upper).unwrap_or(F::zero())),
        confidence_level: Some(confidence_level),
    })
}

// ========================================================================
// Confidence interval for correlation
// ========================================================================

/// Computes a confidence interval for a Pearson correlation coefficient
/// using Fisher's z-transformation.
///
/// z = atanh(r), SE(z) = 1/sqrt(n-3), CI on z, then transform back.
///
/// # Arguments
///
/// * `r` - The sample correlation coefficient
/// * `n` - The sample size
/// * `confidence_level` - Confidence level (e.g., 0.95)
///
/// # Returns
///
/// An `EffectSizeResult` with the correlation and CI bounds.
pub fn correlation_ci<F>(r: F, n: usize, confidence_level: F) -> StatsResult<EffectSizeResult<F>>
where
    F: Float + NumCast + std::fmt::Display,
{
    let rf = <f64 as NumCast>::from(r).unwrap_or(0.0);
    let cl = <f64 as NumCast>::from(confidence_level).unwrap_or(0.95);

    if rf < -1.0 || rf > 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "Correlation must be in [-1, 1], got {}",
            rf
        )));
    }
    if n < 4 {
        return Err(StatsError::InvalidArgument(
            "Sample size must be at least 4 for correlation CI".to_string(),
        ));
    }
    if cl <= 0.0 || cl >= 1.0 {
        return Err(StatsError::InvalidArgument(format!(
            "confidence_level must be in (0, 1), got {}",
            cl
        )));
    }

    // Fisher z-transformation
    let z = 0.5 * ((1.0 + rf) / (1.0 - rf)).ln();
    let se_z = 1.0 / ((n as f64 - 3.0).sqrt());

    let alpha = 1.0 - cl;
    let z_crit = normal_quantile(1.0 - alpha / 2.0);

    let z_lower = z - z_crit * se_z;
    let z_upper = z + z_crit * se_z;

    // Back-transform
    let r_lower = z_lower.tanh();
    let r_upper = z_upper.tanh();

    Ok(EffectSizeResult {
        estimate: r,
        measure: "Pearson correlation".to_string(),
        ci_lower: Some(F::from(r_lower).unwrap_or(F::zero())),
        ci_upper: Some(F::from(r_upper).unwrap_or(F::zero())),
        confidence_level: Some(confidence_level),
    })
}

// ========================================================================
// Conversion between effect size measures
// ========================================================================

/// Converts Cohen's d to the correlation coefficient r.
///
/// r = d / sqrt(d^2 + 4)  (when n1 = n2)
///
/// More generally: r = d / sqrt(d^2 + (n1+n2)^2 / (n1*n2))
///
/// # Arguments
///
/// * `d` - Cohen's d value
/// * `n1` - Size of first group
/// * `n2` - Size of second group
///
/// # Returns
///
/// The equivalent Pearson correlation coefficient.
pub fn d_to_r(d: f64, n1: usize, n2: usize) -> f64 {
    let n1f = n1 as f64;
    let n2f = n2 as f64;
    let a = (n1f + n2f) * (n1f + n2f) / (n1f * n2f);
    d / (d * d + a).sqrt()
}

/// Converts a Pearson correlation r to Cohen's d.
///
/// d = 2*r / sqrt(1 - r^2)  (when n1 = n2)
///
/// # Arguments
///
/// * `r` - Pearson correlation
///
/// # Returns
///
/// The equivalent Cohen's d.
pub fn r_to_d(r: f64) -> f64 {
    if (1.0 - r * r) < 1e-15 {
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        2.0 * r / (1.0 - r * r).sqrt()
    }
}

// ========================================================================
// Internal helpers
// ========================================================================

/// Compute sample mean, sample variance (ddof=1), and n as f64.
fn sample_stats<F>(x: &ArrayView1<F>) -> StatsResult<(f64, f64, f64)>
where
    F: Float + std::iter::Sum<F> + NumCast,
{
    let n = x.len();
    if n == 0 {
        return Err(StatsError::InvalidArgument(
            "Input array cannot be empty".to_string(),
        ));
    }
    let nf = n as f64;
    let mean: f64 = x
        .iter()
        .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
        .sum::<f64>()
        / nf;
    let var: f64 = x
        .iter()
        .map(|&v| {
            let vf = <f64 as NumCast>::from(v).unwrap_or(0.0);
            (vf - mean) * (vf - mean)
        })
        .sum::<f64>()
        / (nf - 1.0).max(1.0);
    Ok((mean, var, nf))
}

/// Compute SS_between and SS_total for ANOVA-style calculations.
fn compute_ss<F>(groups: &[ArrayView1<F>]) -> StatsResult<(f64, f64)>
where
    F: Float + std::iter::Sum<F> + NumCast,
{
    // Grand mean
    let mut grand_sum = 0.0_f64;
    let mut n_total = 0_usize;
    for g in groups {
        for &v in g.iter() {
            grand_sum += <f64 as NumCast>::from(v).unwrap_or(0.0);
            n_total += 1;
        }
    }

    if n_total == 0 {
        return Err(StatsError::InvalidArgument(
            "All groups are empty".to_string(),
        ));
    }

    let grand_mean = grand_sum / n_total as f64;

    // SS_total = sum of (x_ij - grand_mean)^2
    let mut ss_total = 0.0_f64;
    for g in groups {
        for &v in g.iter() {
            let vf = <f64 as NumCast>::from(v).unwrap_or(0.0);
            ss_total += (vf - grand_mean) * (vf - grand_mean);
        }
    }

    // SS_between = sum of n_i * (mean_i - grand_mean)^2
    let mut ss_between = 0.0_f64;
    for g in groups {
        let ni = g.len() as f64;
        let group_mean: f64 = g
            .iter()
            .map(|&v| <f64 as NumCast>::from(v).unwrap_or(0.0))
            .sum::<f64>()
            / ni;
        ss_between += ni * (group_mean - grand_mean) * (group_mean - grand_mean);
    }

    Ok((ss_between, ss_total))
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cohens_d_basic() {
        let x = array![5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let result = cohens_d(&x.view(), &y.view()).expect("Cohen's d failed");
        // Mean diff = 2.0, pooled_sd = sqrt(2.5) ~ 1.58, d ~ 1.26
        assert!(result.estimate > 1.0);
        assert!(result.estimate < 2.0);
    }

    #[test]
    fn test_cohens_d_equal_means() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cohens_d(&x.view(), &y.view()).expect("Cohen's d failed");
        assert!(result.estimate.abs() < 1e-10);
    }

    #[test]
    fn test_cohens_d_too_small() {
        let x = array![1.0];
        let y = array![2.0, 3.0];
        assert!(cohens_d(&x.view(), &y.view()).is_err());
    }

    #[test]
    fn test_hedges_g_correction() {
        let x = array![5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let d = cohens_d(&x.view(), &y.view()).expect("d failed");
        let g = hedges_g(&x.view(), &y.view()).expect("g failed");

        // Hedges' g should be slightly smaller than Cohen's d (bias correction)
        assert!(g.estimate.abs() < d.estimate.abs() + 1e-10);
        assert!(g.estimate.abs() > 0.0);
    }

    #[test]
    fn test_glass_delta_basic() {
        let treatment = array![6.0, 7.0, 8.0, 9.0, 10.0];
        let control = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = glass_delta(&treatment.view(), &control.view()).expect("Glass's delta failed");
        // mean_t = 8, mean_c = 3, sd_c = sqrt(2.5) ~ 1.58, delta ~ 3.16
        assert!(result.estimate > 2.0);
    }

    #[test]
    fn test_glass_delta_symmetric() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![3.0, 4.0, 5.0, 6.0, 7.0];

        let delta_ab = glass_delta(&a.view(), &b.view()).expect("delta AB failed");
        let delta_ba = glass_delta(&b.view(), &a.view()).expect("delta BA failed");

        // Note: Glass's delta is NOT symmetric because it uses only the control group's SD
        assert!(delta_ab.estimate < 0.0);
        assert!(delta_ba.estimate > 0.0);
    }

    #[test]
    fn test_point_biserial_basic() {
        let binary = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let continuous = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result =
            point_biserial_correlation(&binary.view(), &continuous.view()).expect("rpb failed");
        assert!(result.estimate > 0.0);
        assert!(result.estimate <= 1.0);
    }

    #[test]
    fn test_point_biserial_no_correlation() {
        let binary = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let continuous = array![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];

        let result = point_biserial_correlation(&binary.view(), &continuous.view());
        // All values same => SD = 0 => error
        assert!(result.is_err());
    }

    #[test]
    fn test_point_biserial_invalid_binary() {
        let binary = array![0.0, 1.0, 2.0];
        let continuous = array![1.0, 2.0, 3.0];
        let result = point_biserial_correlation(&binary.view(), &continuous.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_eta_squared_basic() {
        let g1 = array![1.0, 2.0, 3.0];
        let g2 = array![4.0, 5.0, 6.0];
        let g3 = array![7.0, 8.0, 9.0];

        let groups = vec![g1.view(), g2.view(), g3.view()];
        let result = eta_squared(&groups).expect("eta2 failed");

        assert!(result.estimate > 0.0);
        assert!(result.estimate <= 1.0);
        // Groups are well separated, eta2 should be high
        assert!(result.estimate > 0.8);
    }

    #[test]
    fn test_eta_squared_no_effect() {
        let g1 = array![5.0, 5.0, 5.0];
        let g2 = array![5.0, 5.0, 5.0];

        let groups = vec![g1.view(), g2.view()];
        let result = eta_squared(&groups);
        // All values identical => SS_total = 0 => error
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_eta_squared() {
        let result = partial_eta_squared(10.0_f64, 40.0_f64).expect("partial eta2 failed");
        assert!((result.estimate - 0.2).abs() < 1e-10); // 10/(10+40) = 0.2
    }

    #[test]
    fn test_partial_eta_squared_zero() {
        let result = partial_eta_squared(0.0_f64, 0.0_f64);
        assert!(result.is_err());
    }

    #[test]
    fn test_omega_squared_basic() {
        let g1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = array![6.0, 7.0, 8.0, 9.0, 10.0];

        let groups = vec![g1.view(), g2.view()];
        let result = omega_squared(&groups).expect("omega2 failed");

        assert!(result.estimate > 0.0);
        assert!(result.estimate <= 1.0);
        // Should be slightly less than eta-squared
        let eta2_result = eta_squared(&groups).expect("eta2 failed");
        assert!(result.estimate <= eta2_result.estimate + 1e-10);
    }

    #[test]
    fn test_cohens_d_ci() {
        let x = array![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let y = array![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = cohens_d_ci(&x.view(), &y.view(), 0.95).expect("CI failed");

        assert!(result.ci_lower.is_some());
        assert!(result.ci_upper.is_some());

        let lower = result.ci_lower.expect("no lower bound");
        let upper = result.ci_upper.expect("no upper bound");
        assert!(lower < result.estimate);
        assert!(upper > result.estimate);
        assert!((result.confidence_level.expect("no CL") - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_ci() {
        let result = correlation_ci(0.5_f64, 100, 0.95).expect("corr CI failed");
        let lower = result.ci_lower.expect("no lower");
        let upper = result.ci_upper.expect("no upper");

        assert!(lower < 0.5);
        assert!(upper > 0.5);
        assert!(lower > 0.0); // r=0.5 with n=100 should have positive lower bound
    }

    #[test]
    fn test_correlation_ci_invalid() {
        assert!(correlation_ci(1.5_f64, 100, 0.95).is_err());
        assert!(correlation_ci(0.5_f64, 2, 0.95).is_err());
        assert!(correlation_ci(0.5_f64, 100, 0.0).is_err());
    }

    #[test]
    fn test_d_to_r_and_back() {
        let d = 0.8;
        let r = d_to_r(d, 50, 50);
        assert!(r > 0.0 && r < 1.0);
        // Approximate round-trip
        let d_back = r_to_d(r);
        assert!((d_back - d).abs() < 0.1); // approximate due to different formulas
    }

    #[test]
    fn test_r_to_d_extremes() {
        assert!(r_to_d(0.0).abs() < 1e-10);
        assert!(r_to_d(0.999) > 10.0);
        assert!(r_to_d(-0.999) < -10.0);
    }

    #[test]
    fn test_normal_quantile() {
        assert!((normal_quantile(0.5) - 0.0).abs() < 1e-10);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) + 1.96).abs() < 0.01);
    }

    #[test]
    fn test_effect_size_result_fields() {
        let x = array![5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![3.0, 4.0, 5.0, 6.0, 7.0];
        let result = cohens_d(&x.view(), &y.view()).expect("d failed");
        assert_eq!(result.measure, "Cohen's d");
        assert!(result.ci_lower.is_none());
        assert!(result.ci_upper.is_none());
    }
}
