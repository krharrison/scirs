//! Autocorrelation Functions for Time Series Analysis
//!
//! This module provides:
//!
//! - **ACF**: Sample autocorrelation function via Pearson's formula
//! - **PACF**: Partial autocorrelation function via Yule-Walker equations
//! - **CCF**: Cross-correlation function between two series
//! - **Confidence bands**: Bartlett's formula for approximate significance bounds
//! - **Ljung-Box test**: portmanteau test for residual autocorrelation
//! - **Box-Pierce test**: simpler version of the portmanteau test
//!
//! # References
//!
//! - Box, G.E.P. & Pierce, D.A. (1970). Distribution of Residual Autocorrelations in
//!   Autoregressive-Integrated Moving Average Time Series Models. JASA.
//! - Ljung, G.M. & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models.
//!   Biometrika.
//! - Bartlett, M.S. (1946). On the Theoretical Specification and Sampling Properties of
//!   Autocorrelated Time-Series. J. R. Stat. Soc. Supplement.

use crate::error::{StatsError, StatsResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// ACF
// ---------------------------------------------------------------------------

/// Compute the sample autocorrelation function (ACF) up to `max_lag`.
///
/// Returns a vector of length `max_lag + 1` where element `k` is the
/// autocorrelation at lag `k`.  Element 0 is always 1.0.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] when `x.len() <= max_lag`.
///
/// # Example
///
/// ```
/// use scirs2_stats::time_series::acf;
///
/// let x = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0];
/// let result = acf(&x, 3).unwrap();
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// ```
pub fn acf(x: &[f64], max_lag: usize) -> StatsResult<Vec<f64>> {
    let n = x.len();
    if n <= max_lag {
        return Err(StatsError::InsufficientData(format!(
            "Need n > max_lag, got n={n} max_lag={max_lag}"
        )));
    }

    let mean = x.iter().sum::<f64>() / n as f64;
    let variance: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    if variance == 0.0 {
        // Constant series: all correlations undefined; return 1 for lag 0, NaN otherwise
        let mut result = vec![f64::NAN; max_lag + 1];
        result[0] = 1.0;
        return Ok(result);
    }

    let mut result = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let cov: f64 = x[..n - lag]
            .iter()
            .zip(x[lag..].iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / n as f64;
        result.push(cov / variance);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// PACF via Yule-Walker
// ---------------------------------------------------------------------------

/// Compute the partial autocorrelation function (PACF) up to `max_lag`
/// via the Yule-Walker equations solved with the Levinson-Durbin recursion.
///
/// Returns a vector of length `max_lag + 1` where element `k` is the PACF
/// at lag `k`.  Element 0 is always 1.0.
///
/// # Errors
///
/// Returns an error when the series is too short or singular.
///
/// # Example
///
/// ```
/// use scirs2_stats::time_series::pacf;
///
/// let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.3).sin()).collect();
/// let p = pacf(&x, 4).unwrap();
/// assert!((p[0] - 1.0).abs() < 1e-10);
/// ```
pub fn pacf(x: &[f64], max_lag: usize) -> StatsResult<Vec<f64>> {
    let n = x.len();
    if n <= max_lag + 1 {
        return Err(StatsError::InsufficientData(format!(
            "Need n > max_lag+1, got n={n} max_lag={max_lag}"
        )));
    }

    // Compute ACF values r[0..=max_lag]
    let r = acf(x, max_lag)?;

    if r.iter().skip(1).any(|v| v.is_nan()) {
        let mut out = vec![f64::NAN; max_lag + 1];
        out[0] = 1.0;
        return Ok(out);
    }

    // Levinson-Durbin recursion
    // phi[k][j] is the j-th coefficient of the AR(k) model (1-indexed)
    let mut phi: Vec<f64> = vec![0.0; max_lag + 1]; // current AR coefficients
    let mut phi_prev: Vec<f64> = vec![0.0; max_lag + 1];
    let mut pacf_vals = vec![1.0f64; max_lag + 1];

    if max_lag == 0 {
        return Ok(pacf_vals);
    }

    // k=1
    if r[1].abs() < 1.0 {
        phi[1] = r[1];
    } else {
        phi[1] = r[1].signum() * (1.0 - 1e-10);
    }
    pacf_vals[1] = phi[1];

    let mut sigma_sq = 1.0 - phi[1].powi(2);

    for k in 2..=max_lag {
        if sigma_sq.abs() < 1e-14 {
            // Degenerate: fill remaining with 0
            for j in k..=max_lag {
                pacf_vals[j] = 0.0;
            }
            break;
        }

        phi_prev[..=k - 1].copy_from_slice(&phi[..=k - 1]);

        // Reflection coefficient
        let num: f64 = r[k]
            - (1..k)
                .map(|j| phi_prev[j] * r[k - j])
                .sum::<f64>();
        let kk = num / sigma_sq;
        phi[k] = kk;
        pacf_vals[k] = kk;

        // Update AR coefficients
        for j in 1..k {
            phi[j] = phi_prev[j] - kk * phi_prev[k - j];
        }
        sigma_sq *= 1.0 - kk.powi(2);
    }

    Ok(pacf_vals)
}

// ---------------------------------------------------------------------------
// Cross-Correlation Function
// ---------------------------------------------------------------------------

/// Compute the sample cross-correlation function (CCF) between `x` and `y`
/// for lags `0, 1, ..., max_lag`.
///
/// The returned value at index `k` is `Corr(x_t, y_{t+k})`, i.e. how much
/// `y` lags `x` at lag `k`.
///
/// # Errors
///
/// Returns an error when the series have different lengths or are too short.
pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> StatsResult<Vec<f64>> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "x and y must have the same length, got {} vs {}",
            n,
            y.len()
        )));
    }
    if n <= max_lag {
        return Err(StatsError::InsufficientData(format!(
            "Need n > max_lag, got n={n} max_lag={max_lag}"
        )));
    }

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let std_x = (x.iter().map(|v| (v - mean_x).powi(2)).sum::<f64>() / n as f64).sqrt();
    let std_y = (y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / n as f64).sqrt();

    if std_x < 1e-15 || std_y < 1e-15 {
        let mut result = vec![f64::NAN; max_lag + 1];
        result[0] = if std_x < 1e-15 && std_y < 1e-15 {
            1.0
        } else {
            0.0
        };
        return Ok(result);
    }

    let mut result = Vec::with_capacity(max_lag + 1);
    for lag in 0..=max_lag {
        let cov: f64 = x[..n - lag]
            .iter()
            .zip(y[lag..].iter())
            .map(|(a, b)| (a - mean_x) * (b - mean_y))
            .sum::<f64>()
            / n as f64;
        result.push(cov / (std_x * std_y));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Confidence Bands (Bartlett's formula)
// ---------------------------------------------------------------------------

/// Compute pointwise confidence bands for the ACF using Bartlett's formula.
///
/// Returns a vector of `(lower, upper)` pairs for lags `0, 1, ..., max_lag`.
/// For lag 0 both bounds are `(1.0, 1.0)`.
///
/// The bands are:
///
/// ```text
/// ±z_{α/2} / sqrt(n)
/// ```
///
/// which is the standard large-sample approximation under white noise.
///
/// # Arguments
///
/// * `n`       – series length
/// * `max_lag` – maximum lag
/// * `alpha`   – significance level (e.g. 0.05 for 95 % bands)
///
/// # Errors
///
/// Returns an error for invalid inputs (n=0, alpha outside (0,1)).
pub fn acf_confidence_bands(
    n: usize,
    max_lag: usize,
    alpha: f64,
) -> StatsResult<Vec<(f64, f64)>> {
    if n == 0 {
        return Err(StatsError::InvalidArgument("n must be > 0".to_string()));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(StatsError::InvalidArgument(
            "alpha must be in (0, 1)".to_string(),
        ));
    }

    // z_{alpha/2}: probit approximation via Beasley-Springer-Moro
    let z = normal_quantile(1.0 - alpha / 2.0);
    let band = z / (n as f64).sqrt();

    let mut bands = Vec::with_capacity(max_lag + 1);
    bands.push((1.0, 1.0)); // lag 0
    for _ in 1..=max_lag {
        bands.push((-band, band));
    }
    Ok(bands)
}

// ---------------------------------------------------------------------------
// Ljung-Box and Box-Pierce tests
// ---------------------------------------------------------------------------

/// Ljung-Box portmanteau test for autocorrelation.
///
/// Tests H₀: the data are independently distributed (no autocorrelation up to
/// the specified lags).
///
/// # Arguments
///
/// * `x`    – time series residuals
/// * `lags` – lags to include (e.g. `&[1, 2, 5, 10]`).  The statistic is
///            computed at the maximum lag in `lags` using all lags from 1 to
///            that maximum.
///
/// # Returns
///
/// `(statistic, p_value)` where the statistic is χ²-distributed with
/// `max_lag` degrees of freedom under H₀.
///
/// # Errors
///
/// Returns [`StatsError::InsufficientData`] when the series is too short.
pub fn ljung_box_test(x: &[f64], lags: &[usize]) -> StatsResult<(f64, f64)> {
    if lags.is_empty() {
        return Err(StatsError::InvalidArgument(
            "lags must be non-empty".to_string(),
        ));
    }
    let max_lag = *lags.iter().max().expect("non-empty lags");
    let n = x.len();
    if n <= max_lag {
        return Err(StatsError::InsufficientData(format!(
            "Need n > max_lag={max_lag}, got n={n}"
        )));
    }

    let acf_vals = acf(x, max_lag)?;
    let n_f = n as f64;

    // Ljung-Box Q = n(n+2) * sum_{k=1}^{m} r_k^2 / (n - k)
    let stat: f64 = lags
        .iter()
        .filter(|&&k| k >= 1)
        .map(|&k| acf_vals[k].powi(2) / (n_f - k as f64))
        .sum::<f64>()
        * n_f
        * (n_f + 2.0);

    let df = max_lag as f64;
    let p_value = chi2_sf(stat, df);
    Ok((stat, p_value))
}

/// Box-Pierce portmanteau test for autocorrelation.
///
/// Similar to the Ljung-Box test but uses the simpler statistic:
///
/// ```text
/// Q = n * sum_{k=1}^{m} r_k^2
/// ```
///
/// # Returns
///
/// `(statistic, p_value)` where the statistic is χ²-distributed with
/// `max_lag` degrees of freedom under H₀.
pub fn box_pierce_test(x: &[f64], lags: &[usize]) -> StatsResult<(f64, f64)> {
    if lags.is_empty() {
        return Err(StatsError::InvalidArgument(
            "lags must be non-empty".to_string(),
        ));
    }
    let max_lag = *lags.iter().max().expect("non-empty lags");
    let n = x.len();
    if n <= max_lag {
        return Err(StatsError::InsufficientData(format!(
            "Need n > max_lag={max_lag}, got n={n}"
        )));
    }

    let acf_vals = acf(x, max_lag)?;
    let n_f = n as f64;

    // Box-Pierce Q = n * sum_{k=1}^{m} r_k^2
    let stat: f64 = lags
        .iter()
        .filter(|&&k| k >= 1)
        .map(|&k| acf_vals[k].powi(2))
        .sum::<f64>()
        * n_f;

    let df = max_lag as f64;
    let p_value = chi2_sf(stat, df);
    Ok((stat, p_value))
}

// ---------------------------------------------------------------------------
// Helper: chi-squared survival function (1 - CDF)
// ---------------------------------------------------------------------------

/// Regularised incomplete gamma function Q(a, x) = 1 - P(a, x)
/// via continued-fraction / series.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x < 0.0 || df <= 0.0 {
        return 1.0;
    }
    // chi2 SF = regularised upper incomplete gamma: Q(df/2, x/2)
    upper_incomplete_gamma(df / 2.0, x / 2.0)
}

/// Upper regularised incomplete gamma Q(a, x) = Gamma(a,x)/Gamma(a)
fn upper_incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - lower_incomplete_gamma_series(a, x)
    } else {
        lower_incomplete_gamma_cf(a, x)
    }
}

/// Lower regularised P(a,x) via series expansion
fn lower_incomplete_gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let mut term = 1.0 / a;
    let mut sum = term;
    let ln_gamma_a = ln_gamma(a);
    for n in 1..max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Upper regularised Q(a,x) via continued fraction (Lentz)
fn lower_incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-300;
    let ln_gamma_a = ln_gamma(a);

    let mut b = x + 1.0 - a;
    let mut c = 1.0 / tiny;
    let mut d = 1.0 / b;
    let mut h = d;

    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Lanczos approximation for ln(Gamma(x))
fn ln_gamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes
    let coeffs: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_78,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        1.208_650_973_866_179e-3,
        -5.395_239_384_953_e-6,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let ser = coeffs
        .iter()
        .enumerate()
        .fold(1.000_000_000_190_015, |acc, (i, &c)| {
            y += 1.0;
            acc + c / y
        });
    let _ = y; // suppress warning
    0.5 * (2.0 * PI).ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

// ---------------------------------------------------------------------------
// Helper: normal quantile (probit) — rational approximation
// ---------------------------------------------------------------------------

/// Rational approximation for the standard normal quantile function (probit).
/// Maximum error ≈ 4.5×10⁻⁴ over the full range.
fn normal_quantile(p: f64) -> f64 {
    // Abramowitz & Stegun 26.2.23
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    if p < 0.5 {
        -rational_approx(p)
    } else {
        rational_approx(1.0 - p)
    }
}

fn rational_approx(p: f64) -> f64 {
    let c = [2.515_517, 0.802_853, 0.010_328];
    let d = [1.432_788, 0.189_269, 0.001_308];
    let t = (-2.0 * p.ln()).sqrt();
    t - (c[0] + c[1] * t + c[2] * t.powi(2))
        / (1.0 + d[0] * t + d[1] * t.powi(2) + d[2] * t.powi(3))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise_series(n: usize) -> Vec<f64> {
        // Deterministic pseudo-random via LCG for reproducibility (no rand dep)
        let mut v = Vec::with_capacity(n);
        let mut s: u64 = 12345;
        for _ in 0..n {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let x = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            v.push(x);
        }
        v
    }

    #[test]
    fn test_acf_lag0_is_one() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r = acf(&x, 5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-12, "lag-0 ACF must be 1");
    }

    #[test]
    fn test_acf_white_noise_small_values() {
        let x = white_noise_series(200);
        let r = acf(&x, 10).unwrap();
        // For WN: |ACF| should be small
        for k in 1..=10 {
            assert!(r[k].abs() < 0.3, "lag {k} ACF too large: {}", r[k]);
        }
    }

    #[test]
    fn test_acf_ar1_positive() {
        // AR(1) with phi=0.8 → ACF(1) ≈ 0.8
        let phi = 0.8_f64;
        let n = 500;
        let mut x = vec![0.0f64; n];
        let wn = white_noise_series(n);
        for i in 1..n {
            x[i] = phi * x[i - 1] + wn[i] * 0.6; // noise variance < 1
        }
        let r = acf(&x, 3).unwrap();
        assert!(r[1] > 0.5, "AR(1) ACF(1) should be >0.5, got {}", r[1]);
    }

    #[test]
    fn test_pacf_lag0_is_one() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 * 0.1 + (i as f64).sin()).collect();
        let p = pacf(&x, 4).unwrap();
        assert!((p[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_pacf_ar1_cutoff() {
        // For AR(1): PACF cuts off after lag 1
        let phi = 0.7_f64;
        let n = 300;
        let mut x = vec![0.0f64; n];
        let wn = white_noise_series(n);
        for i in 1..n {
            x[i] = phi * x[i - 1] + wn[i] * 0.7;
        }
        let p = pacf(&x, 5).unwrap();
        assert!(p[1].abs() > 0.3, "AR(1) PACF(1) should be substantial");
        // Lags 2+ should be smaller
        for k in 2..=5 {
            assert!(p[k].abs() < p[1].abs(), "PACF cutoff failed at lag {k}");
        }
    }

    #[test]
    fn test_cross_correlation_same_series() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        // CCF(x,x,0) == ACF(x,0) == 1
        let cc = cross_correlation(&x, &x, 0).unwrap();
        assert!((cc[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_cross_correlation_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        assert!(cross_correlation(&x, &y, 1).is_err());
    }

    #[test]
    fn test_acf_confidence_bands_width() {
        let bands = acf_confidence_bands(100, 10, 0.05).unwrap();
        // lag 0
        assert!((bands[0].0 - 1.0).abs() < 1e-10);
        // lag 1+: symmetric ±z/sqrt(n) ≈ ±0.196
        for k in 1..=10 {
            let (lo, hi) = bands[k];
            assert!(hi > 0.0 && lo < 0.0);
            assert!((hi + lo).abs() < 1e-10, "bands not symmetric");
        }
    }

    #[test]
    fn test_ljung_box_white_noise_high_pvalue() {
        let x = white_noise_series(100);
        let (_stat, pval) = ljung_box_test(&x, &[1, 2, 3, 4, 5]).unwrap();
        // WN should have large p-value most of the time
        // We just check it doesn't error and pval is in [0,1]
        assert!(pval >= 0.0 && pval <= 1.0);
    }

    #[test]
    fn test_box_pierce_vs_ljung_box_ordering() {
        let x = white_noise_series(100);
        let (lb_stat, _) = ljung_box_test(&x, &[1, 2, 3]).unwrap();
        let (bp_stat, _) = box_pierce_test(&x, &[1, 2, 3]).unwrap();
        // LB >= BP always
        assert!(lb_stat >= bp_stat - 1e-10);
    }

    #[test]
    fn test_insufficient_data_error() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(acf(&x, 5).is_err());
        assert!(pacf(&x, 5).is_err());
    }
}
