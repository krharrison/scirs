//! Conditional Independence Tests for Time Series
//!
//! This module provides conditional independence (CI) tests adapted for time series data,
//! used as building blocks for causal discovery algorithms like PC-stable and PCMCI.
//!
//! ## Available Tests
//!
//! - **ParCorr**: Partial correlation test with Fisher's z-transform
//! - **RobustParCorr**: Rank-based partial correlation (Spearman)
//! - **GaussianCMI**: Conditional mutual information assuming Gaussian distribution
//!
//! ## Effective Sample Size
//!
//! All tests adjust for autocorrelation in time series via effective sample size
//! estimation, which accounts for reduced degrees of freedom due to serial dependence.

use crate::error::TimeSeriesError;
use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::{normal_cdf, CausalityResult};

/// Result of a conditional independence test
#[derive(Debug, Clone)]
pub struct CITestResult {
    /// Test statistic value
    pub statistic: f64,
    /// p-value for the test
    pub p_value: f64,
    /// Effective sample size used
    pub effective_n: f64,
    /// Whether the null hypothesis (independence) is rejected
    pub dependent: bool,
    /// Significance level used for the decision
    pub alpha: f64,
}

/// A lagged variable reference: (variable_index, lag)
/// For example, (0, 2) means variable 0 at time t-2
pub type LaggedVar = (usize, usize);

/// Trait for conditional independence tests on time series data
pub trait TimeSeriesCITest {
    /// Test conditional independence: X_{t-tau_x} _||_ Y_t | Z_set
    ///
    /// # Arguments
    /// * `data` - Multivariate time series, shape (T, n_vars)
    /// * `x` - Source variable (index, lag)
    /// * `y` - Target variable (index, lag=0 typically)
    /// * `z_set` - Conditioning set of (variable_index, lag) pairs
    /// * `alpha` - Significance level
    ///
    /// # Returns
    /// `CITestResult` with test statistic and p-value
    fn test(
        &self,
        data: &Array2<f64>,
        x: LaggedVar,
        y: LaggedVar,
        z_set: &[LaggedVar],
        alpha: f64,
    ) -> CausalityResult<CITestResult>;
}

/// Partial correlation test using Fisher's z-transform
///
/// Tests conditional independence via partial correlation. Under the null hypothesis
/// of conditional independence (given Gaussianity), the Fisher z-transformed partial
/// correlation follows a standard normal distribution.
#[derive(Debug, Clone)]
pub struct ParCorr {
    /// Whether to adjust effective sample size for autocorrelation
    pub adjust_autocorrelation: bool,
}

impl Default for ParCorr {
    fn default() -> Self {
        Self {
            adjust_autocorrelation: true,
        }
    }
}

impl ParCorr {
    /// Create a new ParCorr test
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with autocorrelation adjustment setting
    pub fn with_autocorrelation_adjustment(adjust: bool) -> Self {
        Self {
            adjust_autocorrelation: adjust,
        }
    }
}

impl TimeSeriesCITest for ParCorr {
    fn test(
        &self,
        data: &Array2<f64>,
        x: LaggedVar,
        y: LaggedVar,
        z_set: &[LaggedVar],
        alpha: f64,
    ) -> CausalityResult<CITestResult> {
        let columns = extract_lagged_columns(data, x, y, z_set)?;
        let n = columns.nrows();

        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Too few observations for partial correlation test".to_string(),
                required: 4,
                actual: n,
            });
        }

        // Compute partial correlation between column 0 (x) and column 1 (y)
        // conditioning on columns 2.. (z_set)
        let parcorr = compute_partial_correlation(&columns, 0, 1)?;

        // Effective sample size
        let eff_n = if self.adjust_autocorrelation {
            let x_col = columns.column(0).to_owned();
            let y_col = columns.column(1).to_owned();
            effective_sample_size(&x_col, &y_col)
        } else {
            n as f64
        };

        // Degrees of freedom: n_eff - |Z| - 2
        let z_size = z_set.len() as f64;
        let df = eff_n - z_size - 2.0;

        if df < 1.0 {
            return Ok(CITestResult {
                statistic: 0.0,
                p_value: 1.0,
                effective_n: eff_n,
                dependent: false,
                alpha,
            });
        }

        // Fisher's z-transform
        let clamped = parcorr.clamp(-0.9999, 0.9999);
        let z_stat = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * df.sqrt();

        // Two-sided p-value from standard normal
        let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

        Ok(CITestResult {
            statistic: parcorr,
            p_value,
            effective_n: eff_n,
            dependent: p_value < alpha,
            alpha,
        })
    }
}

/// Robust partial correlation test using Spearman ranks
///
/// Uses rank transformation before computing partial correlation,
/// making it robust to non-Gaussian distributions and outliers.
#[derive(Debug, Clone)]
pub struct RobustParCorr {
    /// Whether to adjust effective sample size for autocorrelation
    pub adjust_autocorrelation: bool,
}

impl Default for RobustParCorr {
    fn default() -> Self {
        Self {
            adjust_autocorrelation: true,
        }
    }
}

impl RobustParCorr {
    /// Create a new RobustParCorr test
    pub fn new() -> Self {
        Self::default()
    }
}

impl TimeSeriesCITest for RobustParCorr {
    fn test(
        &self,
        data: &Array2<f64>,
        x: LaggedVar,
        y: LaggedVar,
        z_set: &[LaggedVar],
        alpha: f64,
    ) -> CausalityResult<CITestResult> {
        let columns = extract_lagged_columns(data, x, y, z_set)?;
        let n = columns.nrows();

        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Too few observations for robust partial correlation test".to_string(),
                required: 4,
                actual: n,
            });
        }

        // Rank-transform each column
        let ranked = rank_transform(&columns);

        let parcorr = compute_partial_correlation(&ranked, 0, 1)?;

        let eff_n = if self.adjust_autocorrelation {
            let x_col = columns.column(0).to_owned();
            let y_col = columns.column(1).to_owned();
            effective_sample_size(&x_col, &y_col)
        } else {
            n as f64
        };

        let z_size = z_set.len() as f64;
        let df = eff_n - z_size - 2.0;

        if df < 1.0 {
            return Ok(CITestResult {
                statistic: 0.0,
                p_value: 1.0,
                effective_n: eff_n,
                dependent: false,
                alpha,
            });
        }

        let clamped = parcorr.clamp(-0.9999, 0.9999);
        let z_stat = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * df.sqrt();
        let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

        Ok(CITestResult {
            statistic: parcorr,
            p_value,
            effective_n: eff_n,
            dependent: p_value < alpha,
            alpha,
        })
    }
}

/// Gaussian Conditional Mutual Information test
///
/// Under Gaussian assumptions, the CMI can be computed from partial correlations:
///   CMI(X; Y | Z) = -0.5 * ln(1 - parcorr(X,Y|Z)^2)
///
/// The test statistic is 2*n_eff*CMI, which follows chi-squared(1) under H0.
#[derive(Debug, Clone)]
pub struct GaussianCMI {
    /// Whether to adjust effective sample size for autocorrelation
    pub adjust_autocorrelation: bool,
}

impl Default for GaussianCMI {
    fn default() -> Self {
        Self {
            adjust_autocorrelation: true,
        }
    }
}

impl GaussianCMI {
    /// Create a new GaussianCMI test
    pub fn new() -> Self {
        Self::default()
    }
}

impl TimeSeriesCITest for GaussianCMI {
    fn test(
        &self,
        data: &Array2<f64>,
        x: LaggedVar,
        y: LaggedVar,
        z_set: &[LaggedVar],
        alpha: f64,
    ) -> CausalityResult<CITestResult> {
        let columns = extract_lagged_columns(data, x, y, z_set)?;
        let n = columns.nrows();

        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Too few observations for CMI test".to_string(),
                required: 4,
                actual: n,
            });
        }

        let parcorr = compute_partial_correlation(&columns, 0, 1)?;

        let eff_n = if self.adjust_autocorrelation {
            let x_col = columns.column(0).to_owned();
            let y_col = columns.column(1).to_owned();
            effective_sample_size(&x_col, &y_col)
        } else {
            n as f64
        };

        // CMI = -0.5 * ln(1 - r^2)
        let r_sq = parcorr * parcorr;
        let cmi = if r_sq < 1.0 {
            -0.5 * (1.0 - r_sq).ln()
        } else {
            f64::INFINITY
        };

        // Test statistic: 2 * n_eff * CMI ~ chi2(1) under H0
        let test_stat = 2.0 * eff_n * cmi;

        // Chi-squared(1) p-value
        let p_value = super::chi_squared_p_value(test_stat, 1);

        Ok(CITestResult {
            statistic: cmi,
            p_value,
            effective_n: eff_n,
            dependent: p_value < alpha,
            alpha,
        })
    }
}

// ---- Internal helpers ----

/// Extract lagged columns from multivariate time series data.
///
/// Given data of shape (T, n_vars), extract columns for x, y, and each z in z_set,
/// properly aligned by their lags.
pub(crate) fn extract_lagged_columns(
    data: &Array2<f64>,
    x: LaggedVar,
    y: LaggedVar,
    z_set: &[LaggedVar],
) -> CausalityResult<Array2<f64>> {
    let t = data.nrows();
    let n_vars = data.ncols();

    // Validate variable indices
    let all_vars: Vec<&LaggedVar> = std::iter::once(&x)
        .chain(std::iter::once(&y))
        .chain(z_set.iter())
        .collect();

    for &&(var_idx, _lag) in &all_vars {
        if var_idx >= n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Variable index {} out of range (n_vars={})",
                var_idx, n_vars
            )));
        }
    }

    // Find max lag to determine the usable time range
    let max_lag = all_vars.iter().map(|&&(_, lag)| lag).max().unwrap_or(0);

    if max_lag >= t {
        return Err(TimeSeriesError::InsufficientData {
            message: "Max lag exceeds time series length".to_string(),
            required: max_lag + 1,
            actual: t,
        });
    }

    let usable_t = t - max_lag;
    let n_cols = 2 + z_set.len();
    let mut result = Array2::zeros((usable_t, n_cols));

    // Fill x column (col 0)
    for row in 0..usable_t {
        let time_idx = max_lag + row - x.1;
        result[[row, 0]] = data[[time_idx, x.0]];
    }

    // Fill y column (col 1)
    for row in 0..usable_t {
        let time_idx = max_lag + row - y.1;
        result[[row, 1]] = data[[time_idx, y.0]];
    }

    // Fill z columns (cols 2..)
    for (z_idx, &(var_idx, lag)) in z_set.iter().enumerate() {
        for row in 0..usable_t {
            let time_idx = max_lag + row - lag;
            result[[row, 2 + z_idx]] = data[[time_idx, var_idx]];
        }
    }

    Ok(result)
}

/// Compute partial correlation between columns `i` and `j` in `data`,
/// conditioning on all other columns.
///
/// Uses recursive formula: residualize on one conditioning variable at a time.
fn compute_partial_correlation(data: &Array2<f64>, i: usize, j: usize) -> CausalityResult<f64> {
    let n_cols = data.ncols();

    if n_cols <= 2 {
        // No conditioning variables, just Pearson correlation
        let x_vec: Vec<f64> = data.column(i).to_vec();
        let y_vec: Vec<f64> = data.column(j).to_vec();
        return Ok(correlation_1d(&x_vec, &y_vec));
    }

    // Compute via precision matrix (inverse of correlation matrix)
    // parcorr(i,j|rest) = -P[i,j] / sqrt(P[i,i] * P[j,j])
    // where P = inv(Sigma)

    let n = data.nrows();
    let k = data.ncols();

    // Compute covariance matrix
    let means: Vec<f64> = (0..k)
        .map(|col| data.column(col).sum() / n as f64)
        .collect();

    let mut cov = Array2::zeros((k, k));
    for a in 0..k {
        for b in a..k {
            let mut s = 0.0;
            for row in 0..n {
                s += (data[[row, a]] - means[a]) * (data[[row, b]] - means[b]);
            }
            let val = s / (n as f64 - 1.0).max(1.0);
            cov[[a, b]] = val;
            cov[[b, a]] = val;
        }
    }

    // Regularize for numerical stability
    for idx in 0..k {
        cov[[idx, idx]] += 1e-10;
    }

    // Invert via Gauss-Jordan elimination
    let precision = invert_matrix(&cov)?;

    let denom = (precision[[i, i]] * precision[[j, j]]).sqrt();
    if denom.abs() < 1e-15 {
        return Ok(0.0);
    }

    Ok(-precision[[i, j]] / denom)
}

/// Invert a symmetric positive-definite matrix using Cholesky-like approach
/// Falls back to Gauss-Jordan if needed.
fn invert_matrix(mat: &Array2<f64>) -> CausalityResult<Array2<f64>> {
    let n = mat.nrows();
    let mut augmented = Array2::zeros((n, 2 * n));

    // Set up [A | I]
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = mat[[i, j]];
        }
        augmented[[i, n + i]] = 1.0;
    }

    // Gauss-Jordan elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = augmented[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = augmented[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in partial correlation computation".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = augmented[[col, j]];
                augmented[[col, j]] = augmented[[max_row, j]];
                augmented[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = augmented[[col, col]];
        for j in 0..(2 * n) {
            augmented[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = augmented[[row, col]];
                for j in 0..(2 * n) {
                    augmented[[row, j]] -= factor * augmented[[col, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = augmented[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Pearson correlation between two slices
fn correlation_1d(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mx;
        let dy = yi - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }

    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Estimate effective sample size for autocorrelated time series.
///
/// Uses Bartlett's formula: n_eff = n / (1 + 2 * sum_{k=1}^{K} rho_x(k) * rho_y(k))
/// where rho_x(k) and rho_y(k) are the autocorrelation functions at lag k.
pub(crate) fn effective_sample_size(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len();
    if n < 5 {
        return n as f64;
    }

    let max_lag = (n / 4).min(20);
    let acf_x = autocorrelation(x, max_lag);
    let acf_y = autocorrelation(y, max_lag);

    let mut correction = 0.0;
    for k in 1..=max_lag {
        if k < acf_x.len() && k < acf_y.len() {
            correction += acf_x[k] * acf_y[k];
        }
    }

    let denom = 1.0 + 2.0 * correction;
    if denom <= 0.0 {
        return n as f64;
    }

    (n as f64 / denom).max(4.0)
}

/// Compute autocorrelation function up to max_lag
fn autocorrelation(x: &Array1<f64>, max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mean = x.sum() / n as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        return vec![1.0; max_lag + 1];
    }

    let mut acf = Vec::with_capacity(max_lag + 1);
    acf.push(1.0); // lag 0

    for lag in 1..=max_lag {
        if lag >= n {
            acf.push(0.0);
            continue;
        }
        let mut s = 0.0;
        for t in lag..n {
            s += (x[t] - mean) * (x[t - lag] - mean);
        }
        acf.push(s / (n as f64 * var));
    }

    acf
}

/// Rank-transform each column of a matrix (ties get average rank)
fn rank_transform(data: &Array2<f64>) -> Array2<f64> {
    let n = data.nrows();
    let k = data.ncols();
    let mut ranked = Array2::zeros((n, k));

    for col in 0..k {
        let column = data.column(col);
        let mut indexed: Vec<(usize, f64)> = column.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks, handling ties with average
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n - 1 && (indexed[j + 1].1 - indexed[i].1).abs() < 1e-15 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 1.0;
            for k_idx in i..=j {
                ranked[[indexed[k_idx].0, col]] = avg_rank;
            }
            i = j + 1;
        }
    }

    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn generate_var1_data(n: usize, seed: u64) -> Array2<f64> {
        // Simple VAR(1): x_t = 0.5*x_{t-1} + e1, y_t = 0.3*x_{t-1} + 0.2*y_{t-1} + e2
        let mut data = Array2::zeros((n, 2));
        // Simple deterministic pseudo-random
        let mut state = seed;
        let next_rand = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };

        for t in 1..n {
            let e1 = next_rand(&mut state) * 0.1;
            let e2 = next_rand(&mut state) * 0.1;
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + e1;
            data[[t, 1]] = 0.3 * data[[t - 1, 0]] + 0.2 * data[[t - 1, 1]] + e2;
        }
        data
    }

    #[test]
    fn test_parcorr_dependent_pair() {
        let data = generate_var1_data(500, 42);
        let test = ParCorr::new();
        // x_{t-1} -> y_t should be dependent
        let result = test
            .test(&data, (0, 1), (1, 0), &[], 0.05)
            .expect("ParCorr test failed");
        assert!(
            result.dependent,
            "x_{{t-1}} -> y_t should be detected as dependent, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_parcorr_independent_pair() {
        // Two independent AR(1) processes
        let n = 500;
        let mut data = Array2::zeros((n, 2));
        let mut s1: u64 = 123;
        let mut s2: u64 = 456;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for t in 1..n {
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + next(&mut s1) * 0.3;
            data[[t, 1]] = 0.4 * data[[t - 1, 1]] + next(&mut s2) * 0.3;
        }

        let test = ParCorr::new();
        let result = test
            .test(&data, (0, 1), (1, 0), &[], 0.05)
            .expect("ParCorr test failed");
        // Should NOT be detected as dependent (at alpha=0.05, with high probability)
        // We use a lenient check since it's a statistical test
        assert!(
            result.p_value > 0.001,
            "Independent series should have high p-value, got p={}",
            result.p_value
        );
    }

    #[test]
    fn test_parcorr_with_conditioning() {
        let data = generate_var1_data(500, 99);
        let test = ParCorr::new();

        // Test y_{t-1} -> y_t conditioning on x_{t-1}
        let result = test
            .test(&data, (1, 1), (1, 0), &[(0, 1)], 0.05)
            .expect("ParCorr conditional test failed");
        // y has autocorrelation coefficient 0.2, so it should be weakly dependent
        assert!(result.statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_robust_parcorr_dependent() {
        let data = generate_var1_data(500, 77);
        let test = RobustParCorr::new();
        let result = test
            .test(&data, (0, 1), (1, 0), &[], 0.05)
            .expect("RobustParCorr test failed");
        assert!(
            result.dependent,
            "Robust test should detect dependency, p={}",
            result.p_value
        );
    }

    #[test]
    fn test_gaussian_cmi_dependent() {
        let data = generate_var1_data(500, 55);
        let test = GaussianCMI::new();
        let result = test
            .test(&data, (0, 1), (1, 0), &[], 0.05)
            .expect("GaussianCMI test failed");
        assert!(
            result.dependent,
            "CMI test should detect dependency, p={}",
            result.p_value
        );
        assert!(result.statistic >= 0.0, "CMI should be non-negative");
    }

    #[test]
    fn test_gaussian_cmi_independent() {
        let n = 500;
        let mut data = Array2::zeros((n, 2));
        let mut s1: u64 = 789;
        let mut s2: u64 = 101;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for t in 1..n {
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + next(&mut s1) * 0.3;
            data[[t, 1]] = 0.4 * data[[t - 1, 1]] + next(&mut s2) * 0.3;
        }

        let test = GaussianCMI::new();
        let result = test
            .test(&data, (0, 1), (1, 0), &[], 0.05)
            .expect("GaussianCMI test failed");
        assert!(
            result.p_value > 0.001,
            "Independent series should have high p-value, got p={}",
            result.p_value
        );
    }

    #[test]
    fn test_effective_sample_size() {
        // Highly autocorrelated series should have reduced effective N
        let n = 200;
        let mut x = Array1::zeros(n);
        let mut y = Array1::zeros(n);
        for t in 1..n {
            x[t] = 0.95 * x[t - 1] + 0.05;
            y[t] = 0.95 * y[t - 1] + 0.05;
        }
        let eff_n = effective_sample_size(&x, &y);
        assert!(
            eff_n < n as f64,
            "Effective N ({}) should be less than actual N ({})",
            eff_n,
            n
        );
    }

    #[test]
    fn test_extract_lagged_columns() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
        )
        .expect("shape error");

        // x=(0,1), y=(1,0), no z
        let cols = extract_lagged_columns(&data, (0, 1), (1, 0), &[]).expect("extract failed");
        assert_eq!(cols.nrows(), 4); // max_lag=1, so 5-1=4 rows
                                     // At row 0: time_idx for x = 1-1=0, for y = 1-0=1
        assert!((cols[[0, 0]] - 1.0).abs() < 1e-10); // x at t=0
        assert!((cols[[0, 1]] - 20.0).abs() < 1e-10); // y at t=1
    }
}
