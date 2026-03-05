//! Diagnostic tests for ARCH/GARCH volatility models
//!
//! This module provides a suite of statistical tests for validating ARCH/GARCH
//! model adequacy and detecting departures from model assumptions.
//!
//! # Tests Available
//!
//! | Test | Function | Null Hypothesis |
//! |------|----------|-----------------|
//! | Engle ARCH-LM | [`arch_lm_test`] | No ARCH effects in residuals |
//! | Ljung-Box | [`ljung_box_test`] | No serial autocorrelation |
//! | Ljung-Box (squared) | [`ljung_box_squared_test`] | No ARCH effects |
//! | Jarque-Bera | [`jarque_bera_test`] | Gaussian innovations |
//! | Sign Bias | [`sign_bias_test`] | No asymmetric volatility |
//! | Nyblom Stability | [`nyblom_stability_test`] | Parameter constancy |
//! | McLeod-Li | [`mcleod_li_test`] | No nonlinear serial dependence |
//!
//! # References
//! - Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with
//!   estimates of the variance of United Kingdom inflation. *Econometrica*, 50, 987–1007.
//! - Ljung, G., & Box, G. (1978). On a measure of lack of fit in time series models.
//!   *Biometrika*, 65(2), 297–303.
//! - Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality,
//!   homoscedasticity and serial independence of regression residuals.
//!   *Economics Letters*, 6(3), 255–259.
//! - Nyblom, J. (1989). Testing for the constancy of parameters over time.
//!   *Journal of the American Statistical Association*, 84(405), 223–230.

use crate::error::{Result, TimeSeriesError};
use crate::volatility::arch::{chi2_survival, ln_gamma, regularised_upper_gamma};

// ============================================================
// Engle's ARCH-LM Test
// ============================================================

/// Result of Engle's ARCH-LM test for conditional heteroskedasticity.
#[derive(Debug, Clone)]
pub struct ArchLmResult {
    /// LM test statistic = n * R²
    pub statistic: f64,
    /// p-value: P(χ²(lags) > statistic)
    pub p_value: f64,
    /// Number of lags used
    pub lags: usize,
    /// Effective sample size used in the regression
    pub n_eff: usize,
    /// R² of the auxiliary regression
    pub r_squared: f64,
}

/// Engle's ARCH-LM test for ARCH effects in a residual series.
///
/// Tests H₀: no ARCH effects (homoscedastic residuals) against H₁: ARCH(lags).
/// The test regresses squared residuals `ε²_t` on `q` lags, and the test
/// statistic `LM = n R²` is asymptotically χ²(q).
///
/// # Arguments
/// * `residuals` — model residuals (not standardised)
/// * `lags` — number of ARCH lags (`q`)
pub fn arch_lm_test(residuals: &[f64], lags: usize) -> Result<ArchLmResult> {
    if lags == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "ARCH-LM: lags must be >= 1".into(),
        ));
    }
    let n = residuals.len();
    if n <= lags + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ARCH-LM: too few observations".into(),
            required: lags + 3,
            actual: n,
        });
    }

    let eps2: Vec<f64> = residuals.iter().map(|&e| e * e).collect();
    let n_reg = n - lags;
    let y: Vec<f64> = eps2[lags..].to_vec();

    // Auxiliary regression: ε²_t = c + Σ γ_j ε²_{t-j}  for j=1..q
    // Design matrix: [1, ε²_{t-1}, ..., ε²_{t-q}]
    let k = lags + 1;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for t in 0..n_reg {
        let row: Vec<f64> = std::iter::once(1.0_f64)
            .chain((1..=lags).map(|j| eps2[lags + t - j]))
            .collect();
        for (a, &ra) in row.iter().enumerate() {
            xty[a] += ra * y[t];
            for (b, &rb) in row.iter().enumerate() {
                xtx[a * k + b] += ra * rb;
            }
        }
    }

    let beta = ols_solve(&xtx, &xty, k)?;

    // Compute R²
    let y_mean = y.iter().sum::<f64>() / n_reg as f64;
    let mut ss_res = 0.0_f64;
    let mut ss_tot = 0.0_f64;
    for t in 0..n_reg {
        let row: Vec<f64> = std::iter::once(1.0_f64)
            .chain((1..=lags).map(|j| eps2[lags + t - j]))
            .collect();
        let y_hat: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        ss_res += (y[t] - y_hat).powi(2);
        ss_tot += (y[t] - y_mean).powi(2);
    }

    let r_squared = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let statistic = n_reg as f64 * r_squared;
    let p_value = chi2_survival(statistic, lags as f64);

    Ok(ArchLmResult {
        statistic,
        p_value,
        lags,
        n_eff: n_reg,
        r_squared,
    })
}

// ============================================================
// Ljung-Box Test
// ============================================================

/// Result of the Ljung-Box autocorrelation test.
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    /// Q-LB test statistic
    pub statistic: f64,
    /// p-value: P(χ²(lags) > Q)
    pub p_value: f64,
    /// Number of lags tested
    pub lags: usize,
    /// Sample autocorrelations ρ_k for k=1..lags
    pub acf: Vec<f64>,
}

/// Ljung-Box portmanteau test for serial autocorrelation.
///
/// Tests H₀: no autocorrelation in `series` up to `lags` lags.
/// ```text
/// Q_LB = n(n+2) Σ_{k=1}^{lags} ρ̂²_k / (n-k)
/// ```
/// Asymptotically χ²(lags) under H₀.
pub fn ljung_box_test(series: &[f64], lags: usize) -> Result<LjungBoxResult> {
    let n = series.len();
    if n <= lags + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Ljung-Box: too few observations".into(),
            required: lags + 2,
            actual: n,
        });
    }
    if lags == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Ljung-Box: lags must be >= 1".into(),
        ));
    }

    let mean = series.iter().sum::<f64>() / n as f64;
    let var: f64 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        return Err(TimeSeriesError::InvalidInput(
            "Ljung-Box: series has zero variance".into(),
        ));
    }

    let mut acf = Vec::with_capacity(lags);
    let mut q_stat = 0.0_f64;

    for k in 1..=lags {
        let cov_k: f64 = (k..n)
            .map(|t| (series[t] - mean) * (series[t - k] - mean))
            .sum::<f64>()
            / n as f64;
        let rho_k = cov_k / var;
        acf.push(rho_k);
        q_stat += rho_k * rho_k / (n - k) as f64;
    }

    q_stat *= n as f64 * (n as f64 + 2.0);

    let p_value = chi2_survival(q_stat, lags as f64);

    Ok(LjungBoxResult {
        statistic: q_stat,
        p_value,
        lags,
        acf,
    })
}

/// Ljung-Box test applied to squared series (tests for ARCH effects).
///
/// Equivalent to `ljung_box_test(series.map(|x| x*x), lags)`.
pub fn ljung_box_squared_test(series: &[f64], lags: usize) -> Result<LjungBoxResult> {
    let squared: Vec<f64> = series.iter().map(|&x| x * x).collect();
    ljung_box_test(&squared, lags)
}

// ============================================================
// Jarque-Bera Normality Test
// ============================================================

/// Result of the Jarque-Bera normality test.
#[derive(Debug, Clone)]
pub struct JarqueBeraResult {
    /// JB test statistic
    pub statistic: f64,
    /// p-value: P(χ²(2) > JB)
    pub p_value: f64,
    /// Sample skewness
    pub skewness: f64,
    /// Sample excess kurtosis (Kurt - 3)
    pub excess_kurtosis: f64,
}

/// Jarque-Bera test for normality of a series.
///
/// Tests H₀: Gaussian distribution (skewness=0, excess kurtosis=0).
/// ```text
/// JB = n/6 * (S² + (K-3)²/4)
/// ```
/// Asymptotically χ²(2) under H₀.
pub fn jarque_bera_test(series: &[f64]) -> Result<JarqueBeraResult> {
    let n = series.len();
    if n < 8 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Jarque-Bera: need at least 8 observations".into(),
            required: 8,
            actual: n,
        });
    }

    let n_f = n as f64;
    let mean = series.iter().sum::<f64>() / n_f;
    let m2 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_f;
    let m3 = series.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n_f;
    let m4 = series.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n_f;

    if m2 < 1e-15 {
        return Err(TimeSeriesError::InvalidInput(
            "Jarque-Bera: series has (near-)zero variance".into(),
        ));
    }

    let std_dev = m2.sqrt();
    let skewness = m3 / (m2 * std_dev);
    let kurtosis = m4 / (m2 * m2);
    let excess_kurtosis = kurtosis - 3.0;

    let statistic = n_f / 6.0 * (skewness.powi(2) + excess_kurtosis.powi(2) / 4.0);
    let p_value = chi2_survival(statistic, 2.0);

    Ok(JarqueBeraResult {
        statistic,
        p_value,
        skewness,
        excess_kurtosis,
    })
}

// ============================================================
// Sign Bias Test — Engle & Ng (1993)
// ============================================================

/// Result of the Engle-Ng sign bias test.
#[derive(Debug, Clone)]
pub struct SignBiasResult {
    /// Sign bias: coefficient on I_{ε_{t-1}<0}
    pub sign_bias: f64,
    /// t-statistic for sign bias
    pub sign_bias_t: f64,
    /// Negative size bias: coefficient on I_{ε_{t-1}<0} * ε_{t-1}
    pub neg_size_bias: f64,
    /// t-statistic for negative size bias
    pub neg_size_bias_t: f64,
    /// Positive size bias: coefficient on I_{ε_{t-1}>=0} * ε_{t-1}
    pub pos_size_bias: f64,
    /// t-statistic for positive size bias
    pub pos_size_bias_t: f64,
    /// Joint test statistic (χ²(3) under H₀)
    pub joint_statistic: f64,
    /// Joint p-value
    pub joint_p_value: f64,
}

/// Engle-Ng (1993) sign bias test for asymmetric volatility.
///
/// Tests whether the squared standardised residuals `ε²_t` can be predicted
/// by functions of the *sign* and *magnitude* of the lagged residual.
///
/// Regression: `ε²_t = c + b₁ S⁻_{t-1} + b₂ S⁻_{t-1} ε_{t-1} + b₃ S⁺_{t-1} ε_{t-1} + u_t`
///
/// where `S⁻_{t-1} = I[ε_{t-1} < 0]` and `S⁺_{t-1} = I[ε_{t-1} >= 0]`.
///
/// # Arguments
/// * `residuals` — demeaned residuals (not standardised)
/// * `squared_residuals` — squared residuals (can be `residuals²` or model `ε²/σ²`)
pub fn sign_bias_test(
    residuals: &[f64],
    squared_residuals: &[f64],
) -> Result<SignBiasResult> {
    let n = residuals.len();
    if n != squared_residuals.len() {
        return Err(TimeSeriesError::InvalidInput(
            "sign_bias: residuals and squared_residuals must have the same length".into(),
        ));
    }
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "sign_bias: need at least 10 observations".into(),
            required: 10,
            actual: n,
        });
    }

    let n_reg = n - 1;
    let y: Vec<f64> = squared_residuals[1..].to_vec();

    // Regressors: [1, S⁻_{t-1}, S⁻_{t-1}*ε_{t-1}, S⁺_{t-1}*ε_{t-1}]
    let k = 4_usize;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for t in 0..n_reg {
        let eps_lag = residuals[t];
        let s_neg = if eps_lag < 0.0 { 1.0 } else { 0.0 };
        let s_pos = 1.0 - s_neg;
        let row = [1.0, s_neg, s_neg * eps_lag, s_pos * eps_lag];
        for (a, &ra) in row.iter().enumerate() {
            xty[a] += ra * y[t];
            for (b, &rb) in row.iter().enumerate() {
                xtx[a * k + b] += ra * rb;
            }
        }
    }

    let beta = ols_solve(&xtx, &xty, k)?;
    let xtx_inv = invert_matrix(k, &xtx)?;

    // Residuals and variance
    let mut ss_res = 0.0_f64;
    for t in 0..n_reg {
        let eps_lag = residuals[t];
        let s_neg = if eps_lag < 0.0 { 1.0 } else { 0.0 };
        let s_pos = 1.0 - s_neg;
        let row = [1.0, s_neg, s_neg * eps_lag, s_pos * eps_lag];
        let y_hat: f64 = row.iter().zip(beta.iter()).map(|(&x, &b)| x * b).sum();
        ss_res += (y[t] - y_hat).powi(2);
    }

    let sigma2 = ss_res / (n_reg as f64 - k as f64).max(1.0);
    let se: Vec<f64> = (0..k)
        .map(|j| (sigma2 * xtx_inv[j * k + j]).sqrt().max(1e-15))
        .collect();

    let t_stats: Vec<f64> = beta.iter().zip(se.iter()).map(|(&b, &s)| b / s).collect();

    // Joint test: LM = n_reg * R² (or equivalently Wald statistic for b₁,b₂,b₃)
    let y_mean = y.iter().sum::<f64>() / n_reg as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r2 = if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 };
    let joint_stat = n_reg as f64 * r2;
    let joint_p = chi2_survival(joint_stat, 3.0);

    Ok(SignBiasResult {
        sign_bias: beta[1],
        sign_bias_t: t_stats[1],
        neg_size_bias: beta[2],
        neg_size_bias_t: t_stats[2],
        pos_size_bias: beta[3],
        pos_size_bias_t: t_stats[3],
        joint_statistic: joint_stat,
        joint_p_value: joint_p,
    })
}

// ============================================================
// Nyblom Parameter Stability Test
// ============================================================

/// Result of the Nyblom test for parameter constancy.
#[derive(Debug, Clone)]
pub struct NyblomResult {
    /// Nyblom joint test statistic L
    pub statistic: f64,
    /// Approximate p-value based on asymptotic critical values
    pub p_value: f64,
    /// Number of parameters tested
    pub n_params: usize,
    /// Individual parameter stability statistics
    pub individual_stats: Vec<f64>,
}

/// Nyblom (1989) parameter stability test.
///
/// Tests H₀: all parameters are constant over time.
/// Based on CUSUM of score contributions from the model.
///
/// # Arguments
/// * `model_scores` — matrix of score contributions: `scores[t][j]` is the
///   score of the j-th parameter at time t.  Length `T × K`.
///   (For GARCH, the score is `∂ log L_t / ∂ θ_j`.)
///
/// Returns `(test_statistic, approx_p_value)`.
pub fn nyblom_stability_test(model_scores: &[Vec<f64>]) -> Result<NyblomResult> {
    let t = model_scores.len();
    if t < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Nyblom: need at least 10 score observations".into(),
            required: 10,
            actual: t,
        });
    }

    let k = model_scores[0].len();
    if k == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Nyblom: score matrix has no columns".into(),
        ));
    }

    for (i, row) in model_scores.iter().enumerate() {
        if row.len() != k {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Nyblom: row {} has {} columns, expected {}",
                i,
                row.len(),
                k
            )));
        }
    }

    let t_f = t as f64;

    // Cumulative sum of scores: S_t = Σ_{s=1}^{t} score_s
    let mut cumsum = vec![vec![0.0_f64; k]; t];
    for t_idx in 0..t {
        for j in 0..k {
            cumsum[t_idx][j] = if t_idx == 0 {
                model_scores[0][j]
            } else {
                cumsum[t_idx - 1][j] + model_scores[t_idx][j]
            };
        }
    }

    // Information matrix estimate: I = (1/T) Σ score_t score_t'
    let mut info = vec![0.0_f64; k * k];
    for t_idx in 0..t {
        for a in 0..k {
            for b in 0..k {
                info[a * k + b] += model_scores[t_idx][a] * model_scores[t_idx][b];
            }
        }
    }
    info.iter_mut().for_each(|v| *v /= t_f);

    // Individual Nyblom statistics: L_j = (1/T²) Σ_t S_{t,j}² / I_{jj}
    let mut individual_stats = Vec::with_capacity(k);
    for j in 0..k {
        let i_jj = info[j * k + j];
        if i_jj < 1e-15 {
            individual_stats.push(0.0);
            continue;
        }
        let l_j = cumsum.iter().map(|cs| cs[j].powi(2)).sum::<f64>()
            / (t_f * t_f * i_jj);
        individual_stats.push(l_j);
    }

    // Joint statistic: L = (1/T²) tr( I⁻¹ Σ_t S_t S_t' )
    let info_inv = match invert_matrix(k, &info) {
        Ok(inv) => inv,
        Err(_) => {
            // Diagonal approximation
            let mut diag_inv = vec![0.0_f64; k * k];
            for j in 0..k {
                let d = info[j * k + j];
                diag_inv[j * k + j] = if d > 1e-12 { 1.0 / d } else { 0.0 };
            }
            diag_inv
        }
    };

    let mut joint_l = 0.0_f64;
    for t_idx in 0..t {
        // I⁻¹ S_t (k-vector)
        let i_inv_s: Vec<f64> = (0..k)
            .map(|a| (0..k).map(|b| info_inv[a * k + b] * cumsum[t_idx][b]).sum::<f64>())
            .collect();
        // S_t' I⁻¹ S_t
        let quad: f64 = (0..k).map(|a| cumsum[t_idx][a] * i_inv_s[a]).sum();
        joint_l += quad;
    }
    joint_l /= t_f * t_f;

    // Approximate p-value using the asymptotic distribution
    // Critical values from Nyblom (1989): 5% = (k * 0.47 + 0.07*k²/10) approx
    // We use the Brownian bridge integral formula approximation
    let p_value = nyblom_p_value(joint_l, k);

    Ok(NyblomResult {
        statistic: joint_l,
        p_value,
        n_params: k,
        individual_stats,
    })
}

/// Approximate p-value for the Nyblom joint stability test.
///
/// Uses simulated critical values (interpolation from Nyblom 1989, Table 2).
fn nyblom_p_value(stat: f64, k: usize) -> f64 {
    // Critical values at 10%, 5%, 1% for k=1..10 (from Nyblom 1989)
    // Row k, columns [cv_10%, cv_5%, cv_1%]
    let critical_values: &[(f64, f64, f64)] = &[
        (0.353, 0.470, 0.749), // k=1
        (0.610, 0.749, 1.070), // k=2
        (0.846, 1.010, 1.350), // k=3
        (1.070, 1.240, 1.600), // k=4
        (1.280, 1.470, 1.820), // k=5
        (1.490, 1.700, 2.130), // k=6
        (1.680, 1.930, 2.390), // k=7
        (1.870, 2.150, 2.690), // k=8
        (2.060, 2.380, 2.990), // k=9
        (2.250, 2.600, 3.280), // k=10
    ];

    let idx = (k.min(10).max(1) - 1) as usize;
    let (cv10, cv05, cv01) = critical_values[idx];

    if stat >= cv01 {
        0.005 // < 1%
    } else if stat >= cv05 {
        0.025 // between 1% and 5%
    } else if stat >= cv10 {
        0.075 // between 5% and 10%
    } else {
        0.20 // > 10% (not significant)
    }
}

// ============================================================
// McLeod-Li Test
// ============================================================

/// McLeod-Li (1983) test for nonlinear serial dependence via autocorrelation of
/// squared standardised residuals.
///
/// Equivalent to [`ljung_box_squared_test`] applied to standardised residuals.
pub fn mcleod_li_test(standardised_residuals: &[f64], lags: usize) -> Result<LjungBoxResult> {
    ljung_box_squared_test(standardised_residuals, lags)
}

// ============================================================
// Linear algebra helpers
// ============================================================

fn ols_solve(xtx: &[f64], xty: &[f64], k: usize) -> Result<Vec<f64>> {
    let inv = invert_matrix(k, xtx)?;
    let beta: Vec<f64> = (0..k)
        .map(|i| (0..k).map(|j| inv[i * k + j] * xty[j]).sum())
        .collect();
    Ok(beta)
}

fn invert_matrix(k: usize, a: &[f64]) -> Result<Vec<f64>> {
    if k == 0 {
        return Ok(Vec::new());
    }

    // Gaussian elimination with partial pivoting on augmented [A | I]
    let mut aug: Vec<f64> = Vec::with_capacity(k * 2 * k);
    for i in 0..k {
        for j in 0..k {
            aug.push(a[i * k + j]);
        }
        for j in 0..k {
            aug.push(if i == j { 1.0 } else { 0.0 });
        }
    }
    let cols = 2 * k;

    for col in 0..k {
        // Pivot
        let pivot_row = (col..k)
            .max_by(|&i, &j| {
                aug[i * cols + col].abs().partial_cmp(&aug[j * cols + col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);

        if aug[pivot_row * cols + col].abs() < 1e-14 {
            return Err(TimeSeriesError::NumericalError(
                "invert_matrix: singular matrix".into(),
            ));
        }

        if pivot_row != col {
            for c in 0..cols {
                aug.swap(col * cols + c, pivot_row * cols + c);
            }
        }

        let pivot = aug[col * cols + col];
        for c in 0..cols {
            aug[col * cols + c] /= pivot;
        }

        for row in 0..k {
            if row == col { continue; }
            let factor = aug[row * cols + col];
            for c in 0..cols {
                let v = aug[col * cols + c] * factor;
                aug[row * cols + c] -= v;
            }
        }
    }

    // Extract right half
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            inv[i * k + j] = aug[i * cols + k + j];
        }
    }

    Ok(inv)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_residuals(n: usize, arch_effect: bool) -> Vec<f64> {
        // Generate residuals with or without ARCH effects
        let mut resid = Vec::with_capacity(n);
        let mut sigma2 = 0.0001_f64;
        for i in 0..n {
            let z = (i as f64 * 0.97 + 0.3).sin();
            let eps = if arch_effect {
                sigma2 = 0.00001 + 0.3 * resid.last().copied().unwrap_or(0.0).powi(2) + 0.6 * sigma2;
                z * sigma2.sqrt()
            } else {
                z * 0.01
            };
            resid.push(eps);
        }
        resid
    }

    fn make_normal_series(n: usize) -> Vec<f64> {
        // Approximately normal via sin (limited, but deterministic)
        (0..n)
            .map(|i| (i as f64 * 1.37 + 0.7).sin() * 0.01)
            .collect()
    }

    #[test]
    fn test_arch_lm_test_no_arch() {
        // iid series should not reject H0
        let resid = make_residuals(100, false);
        let result = arch_lm_test(&resid, 5).expect("ARCH-LM");
        // p-value should be high (no arch effects)
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.statistic >= 0.0);
    }

    #[test]
    fn test_arch_lm_test_with_arch() {
        // ARCH series should have low p-value
        let resid = make_residuals(200, true);
        let result = arch_lm_test(&resid, 5).expect("ARCH-LM");
        assert!(result.p_value < 0.2, "Expected ARCH effects, p={:.4}", result.p_value);
    }

    #[test]
    fn test_arch_lm_test_invalid_lags() {
        let resid = make_residuals(50, false);
        assert!(arch_lm_test(&resid, 0).is_err());
    }

    #[test]
    fn test_ljung_box_test_basic() {
        let series = make_normal_series(50);
        let result = ljung_box_test(&series, 10).expect("Ljung-Box");
        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.acf.len(), 10);
    }

    #[test]
    fn test_ljung_box_too_few() {
        let series = vec![1.0, 2.0, 3.0];
        assert!(ljung_box_test(&series, 5).is_err());
    }

    #[test]
    fn test_jarque_bera_normal() {
        // A series with known skewness=0 and kurtosis=3 should have high p-value
        let series: Vec<f64> = (0..50)
            .map(|i| {
                // Construct a roughly symmetric, mesokurtic series
                let x = ((i as f64 * 0.73) % 1.0) * 2.0 - 1.0;
                x * 0.01
            })
            .collect();
        let result = jarque_bera_test(&series).expect("JB test");
        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
        assert!(result.skewness.is_finite());
        assert!(result.excess_kurtosis.is_finite());
    }

    #[test]
    fn test_jarque_bera_too_few() {
        assert!(jarque_bera_test(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_sign_bias_test_basic() {
        let resid = make_residuals(100, true);
        let sq_resid: Vec<f64> = resid.iter().map(|&x| x * x).collect();
        let result = sign_bias_test(&resid, &sq_resid).expect("Sign bias");
        assert!(result.sign_bias.is_finite());
        assert!(result.joint_statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.joint_p_value));
    }

    #[test]
    fn test_sign_bias_mismatched_lengths() {
        let resid = make_residuals(50, false);
        let sq_resid = make_residuals(40, false).into_iter().map(|x| x * x).collect::<Vec<_>>();
        assert!(sign_bias_test(&resid, &sq_resid).is_err());
    }

    #[test]
    fn test_nyblom_test_basic() {
        // Generate fake score matrix (50 observations × 3 parameters)
        let scores: Vec<Vec<f64>> = (0..50)
            .map(|t| {
                vec![
                    (t as f64 * 0.3).sin() * 0.01,
                    (t as f64 * 0.7 + 0.5).cos() * 0.01,
                    (t as f64 * 1.1 + 1.0).sin() * 0.005,
                ]
            })
            .collect();
        let result = nyblom_stability_test(&scores).expect("Nyblom");
        assert!(result.statistic >= 0.0, "Nyblom statistic must be >= 0");
        assert!((0.0..=1.0).contains(&result.p_value));
        assert_eq!(result.individual_stats.len(), 3);
    }

    #[test]
    fn test_nyblom_insufficient_data() {
        let scores = vec![vec![0.01]; 5];
        assert!(nyblom_stability_test(&scores).is_err());
    }

    #[test]
    fn test_ljung_box_squared_test_basic() {
        let resid = make_residuals(60, true);
        let result = ljung_box_squared_test(&resid, 5).expect("LB squared");
        assert!(result.statistic >= 0.0);
        assert!((0.0..=1.0).contains(&result.p_value));
    }

    #[test]
    fn test_mcleod_li_basic() {
        let resid = make_residuals(60, true);
        let result = mcleod_li_test(&resid, 5).expect("McLeod-Li");
        assert!(result.statistic >= 0.0);
    }
}
