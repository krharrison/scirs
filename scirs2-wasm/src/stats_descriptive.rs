//! Typed descriptive-statistics WASM structs and functions (v0.3.0)
//!
//! This module exposes `DescriptiveStats`, `WasmLinearModel`, `WasmTTestResult`,
//! `WasmKSResult`, and `WasmHistogram` as proper `wasm_bindgen` structs with
//! getter methods, so TypeScript consumers get full type safety without parsing
//! opaque JSON objects.
//!
//! All functions follow the no-unwrap() policy: any computation that can fail
//! returns `Result<_, JsValue>` or falls back to `NaN`/empty values.

use wasm_bindgen::prelude::*;

use std::f64::consts::PI;

// ============================================================================
// DescriptiveStats struct
// ============================================================================

/// Full descriptive statistics for a numeric dataset.
///
/// Construct via [`compute_descriptive_stats`].
///
/// # Example (JavaScript)
/// ```js
/// const ds = compute_descriptive_stats(new Float64Array([1,2,3,4,5]));
/// console.log(ds.mean, ds.std_dev, ds.skewness);
/// ds.free();
/// ```
#[wasm_bindgen]
pub struct DescriptiveStats {
    mean: f64,
    variance: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    median: f64,
    skewness: f64,
    kurtosis: f64,
    count: usize,
    sum: f64,
    q25: f64,
    q75: f64,
    iqr: f64,
}

#[wasm_bindgen]
impl DescriptiveStats {
    /// Arithmetic mean.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 { self.mean }

    /// Population variance (ddof=0).
    #[wasm_bindgen(getter)]
    pub fn variance(&self) -> f64 { self.variance }

    /// Population standard deviation (ddof=0).
    #[wasm_bindgen(getter)]
    pub fn std_dev(&self) -> f64 { self.std_dev }

    /// Minimum value.
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 { self.min }

    /// Maximum value.
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 { self.max }

    /// Median (50th percentile).
    #[wasm_bindgen(getter)]
    pub fn median(&self) -> f64 { self.median }

    /// Fisher's skewness (third standardised moment).
    #[wasm_bindgen(getter)]
    pub fn skewness(&self) -> f64 { self.skewness }

    /// Excess kurtosis (fourth standardised moment minus 3).
    #[wasm_bindgen(getter)]
    pub fn kurtosis(&self) -> f64 { self.kurtosis }

    /// Number of observations.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize { self.count }

    /// Sum of all observations.
    #[wasm_bindgen(getter)]
    pub fn sum(&self) -> f64 { self.sum }

    /// First quartile (25th percentile).
    #[wasm_bindgen(getter)]
    pub fn q25(&self) -> f64 { self.q25 }

    /// Third quartile (75th percentile).
    #[wasm_bindgen(getter)]
    pub fn q75(&self) -> f64 { self.q75 }

    /// Interquartile range (q75 - q25).
    #[wasm_bindgen(getter)]
    pub fn iqr(&self) -> f64 { self.iqr }
}

/// Compute comprehensive descriptive statistics for a dataset.
///
/// # Arguments
/// * `data` — non-empty slice of finite f64 values.
///
/// # Returns
/// A [`DescriptiveStats`] struct.  All fields are `NaN` if `data` is empty.
#[wasm_bindgen]
pub fn compute_descriptive_stats(data: &[f64]) -> DescriptiveStats {
    let n = data.len();
    if n == 0 {
        return DescriptiveStats {
            mean: f64::NAN, variance: f64::NAN, std_dev: f64::NAN,
            min: f64::NAN, max: f64::NAN, median: f64::NAN,
            skewness: f64::NAN, kurtosis: f64::NAN,
            count: 0, sum: 0.0, q25: f64::NAN, q75: f64::NAN, iqr: f64::NAN,
        };
    }

    let n_f = n as f64;
    let sum: f64 = data.iter().sum();
    let mean = sum / n_f;

    // Population variance / std (ddof=0)
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_f;
    let std_dev = variance.sqrt();

    // Skewness + excess kurtosis
    let (skewness, kurtosis) = if std_dev > 0.0 {
        let m3 = data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n_f;
        let m4 = data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / n_f;
        (m3, m4 - 3.0)
    } else {
        (0.0, 0.0)
    };

    // Sort for order statistics
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[n - 1];
    let median = quantile_linear(&sorted, 0.50);
    let q25 = quantile_linear(&sorted, 0.25);
    let q75 = quantile_linear(&sorted, 0.75);
    let iqr = q75 - q25;

    DescriptiveStats {
        mean, variance, std_dev, min, max, median,
        skewness, kurtosis, count: n, sum, q25, q75, iqr,
    }
}

// ============================================================================
// WasmLinearModel struct
// ============================================================================

/// Result of a simple OLS linear regression y = slope·x + intercept.
///
/// Construct via [`wasm_linear_regression_typed`].
#[wasm_bindgen]
pub struct WasmLinearModel {
    slope: f64,
    intercept: f64,
    r_squared: f64,
    std_err_slope: f64,
    std_err_intercept: f64,
    n: usize,
}

#[wasm_bindgen]
impl WasmLinearModel {
    /// Regression slope (β₁).
    #[wasm_bindgen(getter)]
    pub fn slope(&self) -> f64 { self.slope }

    /// Regression intercept (β₀).
    #[wasm_bindgen(getter)]
    pub fn intercept(&self) -> f64 { self.intercept }

    /// Coefficient of determination R².
    #[wasm_bindgen(getter)]
    pub fn r_squared(&self) -> f64 { self.r_squared }

    /// Standard error of the slope estimate.
    #[wasm_bindgen(getter)]
    pub fn std_err_slope(&self) -> f64 { self.std_err_slope }

    /// Standard error of the intercept estimate.
    #[wasm_bindgen(getter)]
    pub fn std_err_intercept(&self) -> f64 { self.std_err_intercept }

    /// Number of observations used.
    #[wasm_bindgen(getter)]
    pub fn n(&self) -> usize { self.n }

    /// Predict y for a new x value.
    pub fn predict(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }

    /// Predict y for an array of x values.
    pub fn predict_batch(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter().map(|&x| self.slope * x + self.intercept).collect()
    }
}

/// Fit simple OLS linear regression y = slope·x + intercept and return
/// a typed [`WasmLinearModel`].
///
/// # Returns
/// A model with all fields `NaN` / 0 on invalid input (fewer than 2 points,
/// length mismatch, or constant x).
#[wasm_bindgen]
pub fn wasm_linear_regression_typed(x: &[f64], y: &[f64]) -> WasmLinearModel {
    let n = x.len();
    let nan_model = WasmLinearModel {
        slope: f64::NAN, intercept: f64::NAN, r_squared: f64::NAN,
        std_err_slope: f64::NAN, std_err_intercept: f64::NAN, n: 0,
    };

    if n < 2 || n != y.len() {
        return nan_model;
    }

    let n_f = n as f64;
    let x_mean = x.iter().sum::<f64>() / n_f;
    let y_mean = y.iter().sum::<f64>() / n_f;

    let mut ss_xx = 0.0_f64;
    let mut ss_xy = 0.0_f64;
    let mut ss_yy = 0.0_f64;

    for i in 0..n {
        let dx = x[i] - x_mean;
        let dy = y[i] - y_mean;
        ss_xx += dx * dx;
        ss_xy += dx * dy;
        ss_yy += dy * dy;
    }

    if ss_xx == 0.0 {
        return nan_model;
    }

    let slope = ss_xy / ss_xx;
    let intercept = y_mean - slope * x_mean;

    // R²
    let ss_res: f64 = (0..n).map(|i| (y[i] - (slope * x[i] + intercept)).powi(2)).sum();
    let r_squared = if ss_yy == 0.0 { 1.0 } else { (1.0 - ss_res / ss_yy).clamp(0.0, 1.0) };

    // Standard errors (unbiased residual variance s² = SS_res / (n-2))
    let (std_err_slope, std_err_intercept) = if n > 2 {
        let s2 = ss_res / (n_f - 2.0);
        let se_slope = (s2 / ss_xx).sqrt();
        let se_intercept = (s2 * (1.0 / n_f + x_mean * x_mean / ss_xx)).sqrt();
        (se_slope, se_intercept)
    } else {
        (f64::NAN, f64::NAN)
    };

    WasmLinearModel { slope, intercept, r_squared, std_err_slope, std_err_intercept, n }
}

// ============================================================================
// WasmTTestResult struct
// ============================================================================

/// Result of a Student's t-test.
///
/// Construct via [`wasm_t_test_one_sample_typed`] or [`wasm_t_test_two_sample_typed`].
#[wasm_bindgen]
pub struct WasmTTestResult {
    t_stat: f64,
    p_value: f64,
    df: f64,
    significant_at_05: bool,
    significant_at_01: bool,
}

#[wasm_bindgen]
impl WasmTTestResult {
    /// t-statistic.
    #[wasm_bindgen(getter)]
    pub fn t_stat(&self) -> f64 { self.t_stat }

    /// Two-tailed p-value.
    #[wasm_bindgen(getter)]
    pub fn p_value(&self) -> f64 { self.p_value }

    /// Degrees of freedom.
    #[wasm_bindgen(getter)]
    pub fn df(&self) -> f64 { self.df }

    /// Whether the result is significant at α=0.05.
    #[wasm_bindgen(getter)]
    pub fn significant_at_05(&self) -> bool { self.significant_at_05 }

    /// Whether the result is significant at α=0.01.
    #[wasm_bindgen(getter)]
    pub fn significant_at_01(&self) -> bool { self.significant_at_01 }
}

/// One-sample t-test: is the sample mean significantly different from `mu`?
///
/// # Returns
/// [`WasmTTestResult`] with all fields `NaN` / `false` on insufficient data.
#[wasm_bindgen]
pub fn wasm_t_test_one_sample_typed(data: &[f64], mu: f64) -> WasmTTestResult {
    let nan_result = WasmTTestResult {
        t_stat: f64::NAN, p_value: f64::NAN, df: f64::NAN,
        significant_at_05: false, significant_at_01: false,
    };
    let n = data.len();
    if n < 2 { return nan_result; }

    let n_f = n as f64;
    let mean = data.iter().sum::<f64>() / n_f;
    let s2 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_f - 1.0);
    let se = (s2 / n_f).sqrt();
    if se == 0.0 { return nan_result; }

    let t_stat = (mean - mu) / se;
    let df = n_f - 1.0;
    let p_value = t_two_tailed_pvalue(t_stat.abs(), df);

    WasmTTestResult {
        t_stat, p_value, df,
        significant_at_05: p_value < 0.05,
        significant_at_01: p_value < 0.01,
    }
}

/// Two-sample Welch t-test: are the means of `x` and `y` significantly different?
///
/// Uses the Welch–Satterthwaite degrees-of-freedom approximation.
///
/// # Returns
/// [`WasmTTestResult`] with all fields `NaN` / `false` on insufficient data.
#[wasm_bindgen]
pub fn wasm_t_test_two_sample_typed(x: &[f64], y: &[f64]) -> WasmTTestResult {
    let nan_result = WasmTTestResult {
        t_stat: f64::NAN, p_value: f64::NAN, df: f64::NAN,
        significant_at_05: false, significant_at_01: false,
    };
    if x.len() < 2 || y.len() < 2 { return nan_result; }

    let nx = x.len() as f64;
    let ny = y.len() as f64;
    let mx = x.iter().sum::<f64>() / nx;
    let my = y.iter().sum::<f64>() / ny;
    let vx = x.iter().map(|&v| (v - mx).powi(2)).sum::<f64>() / (nx - 1.0);
    let vy = y.iter().map(|&v| (v - my).powi(2)).sum::<f64>() / (ny - 1.0);

    let se_x = vx / nx;
    let se_y = vy / ny;
    let se = (se_x + se_y).sqrt();
    if se == 0.0 { return nan_result; }

    let t_stat = (mx - my) / se;
    let df = {
        let num = (se_x + se_y).powi(2);
        let denom = se_x.powi(2) / (nx - 1.0) + se_y.powi(2) / (ny - 1.0);
        if denom == 0.0 { 1.0 } else { num / denom }
    };
    let p_value = t_two_tailed_pvalue(t_stat.abs(), df);

    WasmTTestResult {
        t_stat, p_value, df,
        significant_at_05: p_value < 0.05,
        significant_at_01: p_value < 0.01,
    }
}

// ============================================================================
// WasmKSResult struct
// ============================================================================

/// Result of the Kolmogorov–Smirnov normality test.
///
/// Construct via [`wasm_ks_test_normality`].
#[wasm_bindgen]
pub struct WasmKSResult {
    statistic: f64,
    p_value: f64,
    is_normal_at_05: bool,
}

#[wasm_bindgen]
impl WasmKSResult {
    /// KS test statistic D (maximum absolute deviation from normal CDF).
    #[wasm_bindgen(getter)]
    pub fn statistic(&self) -> f64 { self.statistic }

    /// Approximate two-tailed p-value.
    #[wasm_bindgen(getter)]
    pub fn p_value(&self) -> f64 { self.p_value }

    /// True when the data is consistent with normality at α=0.05
    /// (i.e. p_value > 0.05 ⟹ we fail to reject H₀ of normality).
    #[wasm_bindgen(getter)]
    pub fn is_normal_at_05(&self) -> bool { self.is_normal_at_05 }
}

/// Kolmogorov–Smirnov test for normality.
///
/// Tests whether `data` is consistent with a Normal distribution by comparing
/// the empirical CDF against the Gaussian CDF with the same sample mean and
/// standard deviation (Lilliefors variant — but we use KS critical values as an
/// approximation suitable for exploratory use).
///
/// # Returns
/// [`WasmKSResult`] with `statistic = NaN` if `data.len() < 3`.
#[wasm_bindgen]
pub fn wasm_ks_test_normality(data: &[f64]) -> WasmKSResult {
    let nan_result = WasmKSResult { statistic: f64::NAN, p_value: f64::NAN, is_normal_at_05: false };
    let n = data.len();
    if n < 3 { return nan_result; }

    let n_f = n as f64;
    let mean = data.iter().sum::<f64>() / n_f;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_f;
    let std_dev = var.sqrt();
    if std_dev == 0.0 { return nan_result; }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // KS statistic: max |F_n(x) - Phi(x)|
    let mut d: f64 = 0.0;
    for (i, &xi) in sorted.iter().enumerate() {
        let phi = normal_cdf_inner(xi, mean, std_dev);
        let fn_plus = (i + 1) as f64 / n_f;
        let fn_minus = i as f64 / n_f;
        let di = (fn_plus - phi).abs().max((phi - fn_minus).abs());
        if di > d { d = di; }
    }

    // Asymptotic p-value approximation via the Kolmogorov distribution
    // p ≈ 2 * sum_{k=1}^∞ (-1)^(k+1) exp(-2 k² λ²)
    // where λ = D * (sqrt(n) + 0.12 + 0.11/sqrt(n)) — Miller (1956) correction
    let lambda = d * (n_f.sqrt() + 0.12 + 0.11 / n_f.sqrt());
    let p_value = kolmogorov_pvalue(lambda);

    WasmKSResult {
        statistic: d,
        p_value,
        is_normal_at_05: p_value > 0.05,
    }
}

// ============================================================================
// WasmHistogram struct
// ============================================================================

/// Result of [`wasm_histogram`].
#[wasm_bindgen]
pub struct WasmHistogram {
    edges: Vec<f64>,
    counts: Vec<u32>,
    density: Vec<f64>,
    n_bins: usize,
    total: usize,
}

#[wasm_bindgen]
impl WasmHistogram {
    /// Bin edges vector (length = n_bins + 1).
    #[wasm_bindgen(getter)]
    pub fn edges(&self) -> Vec<f64> { self.edges.clone() }

    /// Count of observations in each bin (length = n_bins).
    #[wasm_bindgen(getter)]
    pub fn counts(&self) -> Vec<u32> { self.counts.clone() }

    /// Probability density in each bin: count / (total * bin_width).
    #[wasm_bindgen(getter)]
    pub fn density(&self) -> Vec<f64> { self.density.clone() }

    /// Number of bins.
    #[wasm_bindgen(getter)]
    pub fn n_bins(&self) -> usize { self.n_bins }

    /// Total number of observations.
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> usize { self.total }

    /// Centre of each bin (length = n_bins).
    pub fn bin_centers(&self) -> Vec<f64> {
        (0..self.n_bins)
            .map(|i| (self.edges[i] + self.edges[i + 1]) / 2.0)
            .collect()
    }

    /// Width of each bin (uniform spacing → all equal).
    pub fn bin_width(&self) -> f64 {
        if self.n_bins == 0 { return f64::NAN; }
        self.edges[1] - self.edges[0]
    }
}

/// Compute a histogram of `data` with `n_bins` equally-spaced bins.
///
/// Bins span `[min, max]`; the last bin is closed on the right.
/// Returns an empty histogram if `data` is empty or `n_bins == 0`.
#[wasm_bindgen]
pub fn wasm_histogram(data: &[f64], n_bins: usize) -> WasmHistogram {
    let empty = WasmHistogram { edges: vec![], counts: vec![], density: vec![], n_bins: 0, total: 0 };
    if data.is_empty() || n_bins == 0 { return empty; }

    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if !min.is_finite() || !max.is_finite() { return empty; }

    // If all values are identical, create a single-bin histogram
    let (range, adjusted_min) = if (max - min).abs() < f64::EPSILON {
        (1.0, min - 0.5)
    } else {
        (max - min, min)
    };

    let bin_width = range / n_bins as f64;

    // Build edges
    let edges: Vec<f64> = (0..=n_bins).map(|i| adjusted_min + i as f64 * bin_width).collect();

    let mut counts = vec![0u32; n_bins];
    for &v in data {
        if v.is_nan() { continue; }
        let bin_idx = ((v - adjusted_min) / bin_width).floor() as isize;
        // Clamp: last value falls exactly on right edge → last bin
        let bin_idx = bin_idx.clamp(0, n_bins as isize - 1) as usize;
        counts[bin_idx] += 1;
    }

    let total: usize = counts.iter().map(|&c| c as usize).sum();
    let density: Vec<f64> = counts.iter()
        .map(|&c| if total > 0 { c as f64 / (total as f64 * bin_width) } else { 0.0 })
        .collect();

    WasmHistogram { edges, counts, density, n_bins, total }
}

// ============================================================================
// Spearman correlation (typed version)
// ============================================================================

/// Compute the Spearman rank correlation coefficient between `x` and `y`.
///
/// Converts each array to ranks (with average ranks for ties), then computes
/// Pearson correlation on those ranks.
///
/// # Returns
/// The Spearman ρ in [-1, 1], or `NaN` for empty / mismatched inputs.
#[wasm_bindgen]
pub fn wasm_spearman_correlation_typed(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || n != y.len() { return f64::NAN; }

    let rx = rank_vector(x);
    let ry = rank_vector(y);

    pearson_on_ranks(&rx, &ry)
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Linear-interpolation quantile on a pre-sorted slice.
fn quantile_linear(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return sorted[0]; }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi { sorted[lo] }
    else { sorted[lo] * (1.0 - (h - lo as f64)) + sorted[hi] * (h - lo as f64) }
}

/// Normal CDF via rational-approximation error function.
fn normal_cdf_inner(x: f64, mean: f64, std_dev: f64) -> f64 {
    let z = (x - mean) / (std_dev * std::f64::consts::SQRT_2);
    (1.0 + erf_approx(z)) / 2.0
}

/// Error function via Abramowitz & Stegun 7.1.26 (max err ~1.5e-7).
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0_f64 } else { 1.0_f64 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
        + t * (1.421_413_741
        + t * (-1.453_152_027
        + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Regularised incomplete beta via continued-fraction (Lentz method).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    let mut c = 1.0_f64;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    let mut f = d;
    for m in 1_usize..=200 {
        let m_f = m as f64;
        let num_e = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_e * d; if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_e / c; if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d; f *= d * c;
        let num_o = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_o * d; if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_o / c; if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-14 { break; }
    }
    front * f
}

/// Lanczos approximation to ln(Γ(x)).
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if x < 0.5 {
        let v = PI / ((PI * x).sin() * ln_gamma(1.0 - x).exp());
        return v.ln();
    }
    let xm1 = x - 1.0;
    let mut a = C[0];
    let t = xm1 + G + 0.5;
    for (i, &c) in C[1..].iter().enumerate() {
        a += c / (xm1 + i as f64 + 1.0);
    }
    (2.0 * PI).sqrt().ln() + a.ln() + (xm1 + 0.5) * t.ln() - t
}

/// Two-tailed p-value for Student's t with `df` degrees of freedom.
fn t_two_tailed_pvalue(t_abs: f64, df: f64) -> f64 {
    let x = df / (df + t_abs * t_abs);
    2.0 * regularized_incomplete_beta(x, df / 2.0, 0.5)
}

/// Kolmogorov asymptotic distribution p-value: P(K > λ).
fn kolmogorov_pvalue(lambda: f64) -> f64 {
    if lambda <= 0.0 { return 1.0; }
    let mut p = 0.0_f64;
    for k in 1_i64..=100 {
        let term = (-2.0 * (k as f64).powi(2) * lambda * lambda).exp();
        let sign = if k % 2 == 0 { -1.0_f64 } else { 1.0_f64 };
        p += sign * term;
        if term < 1e-14 { break; }
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// Assign fractional ranks (average of tied ranks) to a slice.
fn rank_vector(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    // Create (value, original_index) pairs and sort by value
    let mut indexed: Vec<(f64, usize)> = data.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    indexed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let val = indexed[i].0;
        let mut j = i;
        while j < n && indexed[j].0 == val { j += 1; }
        // Average rank for the tied group [i, j)
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average rank
        for item in indexed[i..j].iter() {
            ranks[item.1] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation directly on pre-computed rank vectors.
fn pearson_on_ranks(rx: &[f64], ry: &[f64]) -> f64 {
    let n = rx.len();
    if n == 0 { return f64::NAN; }
    let n_f = n as f64;
    let mx = rx.iter().sum::<f64>() / n_f;
    let my = ry.iter().sum::<f64>() / n_f;
    let mut cov = 0.0_f64;
    let mut vx = 0.0_f64;
    let mut vy = 0.0_f64;
    for i in 0..n {
        let dx = rx[i] - mx;
        let dy = ry[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 { f64::NAN } else { cov / (vx * vy).sqrt() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_stats_basic() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let ds = compute_descriptive_stats(&data);
        assert!((ds.mean() - 3.0).abs() < 1e-10, "mean = {}", ds.mean());
        assert!((ds.min() - 1.0).abs() < 1e-10);
        assert!((ds.max() - 5.0).abs() < 1e-10);
        assert!((ds.median() - 3.0).abs() < 1e-10);
        assert_eq!(ds.count(), 5);
        assert!((ds.sum() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_descriptive_stats_empty() {
        let ds = compute_descriptive_stats(&[]);
        assert!(ds.mean().is_nan());
        assert_eq!(ds.count(), 0);
    }

    #[test]
    fn test_descriptive_stats_skewness_symmetric() {
        // Symmetric distribution → skewness ≈ 0
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let ds = compute_descriptive_stats(&data);
        assert!(ds.skewness().abs() < 1e-10, "skewness = {}", ds.skewness());
    }

    #[test]
    fn test_descriptive_stats_iqr() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let ds = compute_descriptive_stats(&data);
        // q25 = 2, q75 = 4, iqr = 2 (linear interpolation)
        assert!(ds.iqr() > 0.0, "iqr = {}", ds.iqr());
    }

    #[test]
    fn test_linear_regression_typed_perfect() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let model = wasm_linear_regression_typed(&x, &y);
        assert!((model.slope() - 2.0).abs() < 1e-10, "slope = {}", model.slope());
        assert!(model.intercept().abs() < 1e-10, "intercept = {}", model.intercept());
        assert!((model.r_squared() - 1.0).abs() < 1e-10, "r2 = {}", model.r_squared());
    }

    #[test]
    fn test_linear_regression_predict() {
        let x = [0.0_f64, 1.0, 2.0];
        let y = [1.0_f64, 3.0, 5.0];
        let model = wasm_linear_regression_typed(&x, &y);
        let p = model.predict(3.0);
        assert!((p - 7.0).abs() < 1e-8, "predict(3) = {}", p);
    }

    #[test]
    fn test_t_test_one_sample() {
        // Data drawn from N(0,1), test against mu=0 → should NOT reject
        let data = [-0.5_f64, 0.2, 0.1, -0.3, 0.4, 0.0, 0.15, -0.1];
        let res = wasm_t_test_one_sample_typed(&data, 0.0);
        assert!(!res.t_stat().is_nan());
        assert!(res.p_value() > 0.05, "p_value = {}", res.p_value());
    }

    #[test]
    fn test_t_test_two_sample_distinct() {
        // Data from clearly different populations
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..20).map(|i| 10.0 + i as f64 * 0.1).collect();
        let res = wasm_t_test_two_sample_typed(&x, &y);
        assert!(res.p_value() < 0.001, "p_value = {}", res.p_value());
        assert!(res.significant_at_01());
    }

    #[test]
    fn test_ks_test_normality_gaussian() {
        // Large sample from a symmetric distribution → should not reject normality
        let data: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
        let res = wasm_ks_test_normality(&data);
        assert!(!res.statistic().is_nan(), "KS statistic should be computable");
    }

    #[test]
    fn test_ks_test_too_small() {
        let res = wasm_ks_test_normality(&[1.0, 2.0]);
        assert!(res.statistic().is_nan());
    }

    #[test]
    fn test_histogram_basic() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let h = wasm_histogram(&data, 5);
        assert_eq!(h.n_bins(), 5);
        assert_eq!(h.edges().len(), 6);
        assert_eq!(h.counts().len(), 5);
        assert_eq!(h.total(), 5);
    }

    #[test]
    fn test_histogram_empty() {
        let h = wasm_histogram(&[], 5);
        assert_eq!(h.n_bins(), 0);
    }

    #[test]
    fn test_histogram_density_sums_to_one() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let h = wasm_histogram(&data, 10);
        let bw = h.bin_width();
        let area: f64 = h.density().iter().sum::<f64>() * bw;
        assert!((area - 1.0).abs() < 1e-10, "density area = {}", area);
    }

    #[test]
    fn test_spearman_perfect_monotone() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let rho = wasm_spearman_correlation_typed(&x, &y);
        assert!((rho - 1.0).abs() < 1e-10, "rho = {}", rho);
    }

    #[test]
    fn test_spearman_perfect_anti_monotone() {
        let x = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0_f64, 4.0, 3.0, 2.0, 1.0];
        let rho = wasm_spearman_correlation_typed(&x, &y);
        assert!((rho + 1.0).abs() < 1e-10, "rho = {}", rho);
    }

    #[test]
    fn test_spearman_mismatch() {
        let x = [1.0_f64, 2.0];
        let y = [3.0_f64];
        assert!(wasm_spearman_correlation_typed(&x, &y).is_nan());
    }

    #[test]
    fn test_quantile_edge_cases() {
        let sorted = [1.0_f64, 3.0, 5.0];
        assert!((quantile_linear(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((quantile_linear(&sorted, 1.0) - 5.0).abs() < 1e-10);
        assert!((quantile_linear(&sorted, 0.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_vector_no_ties() {
        let data = [3.0_f64, 1.0, 4.0, 1.5, 9.0];
        let ranks = rank_vector(&data);
        // Sorted: 1.0(idx 1)→rank1, 1.5(idx 3)→rank2, 3.0(idx 0)→rank3,
        //         4.0(idx 2)→rank4, 9.0(idx 4)→rank5
        assert!((ranks[0] - 3.0).abs() < 1e-10, "rank[0] = {}", ranks[0]);
        assert!((ranks[1] - 1.0).abs() < 1e-10, "rank[1] = {}", ranks[1]);
        assert!((ranks[4] - 5.0).abs() < 1e-10, "rank[4] = {}", ranks[4]);
    }

    #[test]
    fn test_rank_vector_ties() {
        let data = [1.0_f64, 1.0, 3.0];
        let ranks = rank_vector(&data);
        // First two values tied → average rank = (1+2)/2 = 1.5
        assert!((ranks[0] - 1.5).abs() < 1e-10, "rank[0] = {}", ranks[0]);
        assert!((ranks[1] - 1.5).abs() < 1e-10, "rank[1] = {}", ranks[1]);
        assert!((ranks[2] - 3.0).abs() < 1e-10, "rank[2] = {}", ranks[2]);
    }
}
