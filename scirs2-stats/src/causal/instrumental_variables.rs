//! Instrumental Variables Estimation
//!
//! This module provides methods for causal inference using instrumental variables (IV):
//!
//! - **`IVEstimator`**: Two-Stage Least Squares (2SLS) estimator
//! - **`LIML`**: Limited Information Maximum Likelihood
//! - **`HausmanTest`**: Endogeneity test (Hausman specification test)
//! - **`WeakInstrumentTest`**: First-stage F-test and partial R²
//! - **`IVResult`**: Unified result struct for IV estimators
//!
//! # References
//!
//! - Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics.
//! - Heckman, J.J. & Vytlacil, E. (2007). Econometric Evaluation of Social Programs.
//! - Stock, J.H., Wright, J.H. & Yogo, M. (2002). A Survey of Weak Instruments
//!   and Weak Identification in Generalized Method of Moments.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of an instrumental variables estimation
#[derive(Debug, Clone)]
pub struct IVResult {
    /// Point estimates of the structural equation coefficients
    pub coefficients: Array1<f64>,

    /// Standard errors (heteroskedasticity-robust if requested)
    pub std_errors: Array1<f64>,

    /// t-statistics for each coefficient
    pub t_stats: Array1<f64>,

    /// Two-sided p-values for each coefficient
    pub p_values: Array1<f64>,

    /// 95 % confidence intervals (n_params × 2)
    pub conf_intervals: Array2<f64>,

    /// Number of observations
    pub n_obs: usize,

    /// Number of endogenous regressors
    pub n_endog: usize,

    /// Number of instruments
    pub n_instruments: usize,

    /// First-stage F-statistic (weak-instrument diagnostic)
    pub first_stage_f: Option<f64>,

    /// Partial R² from the first stage
    pub partial_r_squared: Option<f64>,

    /// J-statistic (Sargan / Hansen test for over-identification)
    pub j_statistic: Option<f64>,

    /// p-value for the J-statistic
    pub j_p_value: Option<f64>,

    /// Residual sum of squares in the structural equation
    pub rss: f64,

    /// Estimator name ("2SLS" or "LIML")
    pub estimator: String,
}

/// Result of the Hausman endogeneity test
#[derive(Debug, Clone)]
pub struct HausmanResult {
    /// Hausman test statistic (chi-squared)
    pub statistic: f64,
    /// p-value (chi-squared distribution)
    pub p_value: f64,
    /// Degrees of freedom (number of endogenous regressors)
    pub df: usize,
    /// Difference of OLS and IV coefficient vectors
    pub coef_difference: Array1<f64>,
    /// Whether endogeneity is rejected at 5 % level
    pub endogenous_detected: bool,
}

/// Result of the weak-instrument test
#[derive(Debug, Clone)]
pub struct WeakInstrumentResult {
    /// First-stage F-statistic for each endogenous regressor
    pub f_statistics: Vec<f64>,
    /// Partial R² for each endogenous regressor
    pub partial_r_squared: Vec<f64>,
    /// Stock-Yogo critical values at 10 % maximal IV size distortion (approximate)
    pub critical_value_10pct: f64,
    /// Whether instruments are weak (F < critical value)
    pub instruments_weak: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Helper: OLS via normal equations (no external BLAS)
// ---------------------------------------------------------------------------

/// Compute X'X and X'y and solve for beta = (X'X)^{-1} X'y
///
/// Returns (beta, residuals, (X'X)^{-1})
fn ols_fit(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> StatsResult<(Array1<f64>, Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let k = x.ncols();
    if n < k {
        return Err(StatsError::InsufficientData(format!(
            "OLS requires at least {k} observations, got {n}"
        )));
    }

    // X'X
    let xtx = x.t().dot(x);
    // X'y
    let xty = x.t().dot(y);

    // Invert X'X via Cholesky decomposition
    let xtx_inv = cholesky_invert(&xtx.view())?;
    let beta = xtx_inv.dot(&xty);
    let fitted = x.dot(&beta);
    let residuals = y.to_owned() - fitted;

    Ok((beta, residuals, xtx_inv))
}

/// Cholesky inversion for positive-definite matrix
fn cholesky_invert(a: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(StatsError::DimensionMismatch(
            "Matrix must be square for inversion".into(),
        ));
    }
    // Cholesky factor L such that A = L L'
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for p in 0..j {
                s -= l[[i, p]] * l[[j, p]];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix is not positive definite; check for multicollinearity".into(),
                    ));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    // Invert L (forward substitution: L Y = I → Y = L^{-1})
    let mut linv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        linv[[j, j]] = 1.0 / l[[j, j]];
        for i in (j + 1)..n {
            let mut s = 0.0_f64;
            for p in j..i {
                s += l[[i, p]] * linv[[p, j]];
            }
            linv[[i, j]] = -s / l[[i, i]];
        }
    }
    // A^{-1} = (L^{-1})' L^{-1}
    let inv = linv.t().dot(&linv);
    Ok(inv)
}

/// Compute the t-distribution CDF for a two-sided test.
/// Uses a rational approximation to the regularized incomplete beta function.
fn t_dist_p_value(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t * t);
    let a = df / 2.0;
    let b = 0.5_f64;
    // P(|T| >= |t|) = I(x; a, b) via continued-fraction / log approximation
    let ibeta = regularized_incomplete_beta(x, a, b);
    ibeta.min(1.0).max(0.0)
}

/// Two-sided p-value from chi-squared distribution (degrees of freedom `df`)
fn chi2_p_value(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let df_f = df as f64;
    // Upper tail: P(chi2 >= x) = 1 - gamma_reg(df/2, x/2)
    1.0 - regularized_gamma_lower(df_f / 2.0, x / 2.0)
}

/// Regularized lower incomplete gamma: P(a, x) = gamma(a, x) / Gamma(a)
/// Uses the series expansion for x < a+1, continued fraction otherwise.
fn regularized_gamma_lower(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        // Series expansion
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 3e-15 {
                break;
            }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - regularized_gamma_upper_cf(a, x)
    }
}

/// Continued-fraction representation for upper regularized gamma P(a,x) (Lentz method)
fn regularized_gamma_upper_cf(a: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200_i64 {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Natural log of the Gamma function (Stirling / Lanczos approximation)
fn ln_gamma(x: f64) -> f64 {
    // Lanczos coefficients for g=7
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.13857_109_526_572_012,
        9.984_369_578_019_572e-6,
        1.5056_327_351_493_116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut s = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            s += ci / (z + (i as f64) + 1.0);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln()
            + (z + 0.5) * t.ln()
            - t
            + s.ln()
    }
}

/// Regularized incomplete Beta function I_x(a,b) via continued fraction (Lentz)
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the symmetry relation when x > (a+1)/(a+b+2)
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let log_beta_cf = (a * x.ln() + b * (1.0 - x).ln()
        - ln_gamma(a)
        - ln_gamma(b)
        + ln_gamma(a + b))
        .exp()
        / a;
    log_beta_cf * beta_cf(x, a, b)
}

/// Continued-fraction expansion for the incomplete beta (Lentz method)
fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200_i32 {
        let m_f = m as f64;
        // Even step
        let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        // Odd step
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 {
            break;
        }
    }
    h
}

// ---------------------------------------------------------------------------
// Two-Stage Least Squares (2SLS)
// ---------------------------------------------------------------------------

/// Two-Stage Least Squares instrumental-variable estimator.
///
/// The structural equation is:
///   y = X_endog * β_endog + X_exog * β_exog + ε
///
/// The instrument set Z contains the excluded instruments plus all exogenous
/// regressors.  The estimator is:
///   β̂_{2SLS} = (X̂'X)^{-1} X̂'y,  where X̂ = Z(Z'Z)^{-1}Z'X
pub struct IVEstimator {
    /// Whether to use heteroskedasticity-robust standard errors
    pub robust_se: bool,
}

impl IVEstimator {
    /// Create a new 2SLS estimator.
    ///
    /// # Arguments
    /// * `robust_se` – if `true`, use HC1 heteroskedasticity-robust SEs.
    pub fn new(robust_se: bool) -> Self {
        Self { robust_se }
    }

    /// Fit the 2SLS model.
    ///
    /// # Arguments
    /// * `y`        – outcome vector (n,)
    /// * `x_endog`  – endogenous regressors (n × k_endog); may be empty (ncols==0)
    /// * `x_exog`   – exogenous regressors including a constant (n × k_exog)
    /// * `z_excl`   – excluded instruments (n × l), l >= k_endog
    ///
    /// # Returns
    /// [`IVResult`] with 2SLS estimates and diagnostics.
    pub fn fit(
        &self,
        y: &ArrayView1<f64>,
        x_endog: &ArrayView2<f64>,
        x_exog: &ArrayView2<f64>,
        z_excl: &ArrayView2<f64>,
    ) -> StatsResult<IVResult> {
        let n = y.len();
        let k_endog = x_endog.ncols();
        let k_exog = x_exog.ncols();
        let l_excl = z_excl.ncols();

        if n < 2 {
            return Err(StatsError::InsufficientData(
                "Need at least 2 observations".into(),
            ));
        }
        if x_endog.nrows() != n || x_exog.nrows() != n || z_excl.nrows() != n {
            return Err(StatsError::DimensionMismatch(
                "All matrices must have the same number of rows as y".into(),
            ));
        }
        if l_excl < k_endog {
            return Err(StatsError::InvalidArgument(format!(
                "Need at least {k_endog} instruments for {k_endog} endogenous regressors, got {l_excl}"
            )));
        }

        // Instrument matrix Z = [z_excl | x_exog]
        let k_total = k_endog + k_exog;
        let l_total = l_excl + k_exog;

        let mut z = Array2::<f64>::zeros((n, l_total));
        for i in 0..n {
            for j in 0..l_excl {
                z[[i, j]] = z_excl[[i, j]];
            }
            for j in 0..k_exog {
                z[[i, l_excl + j]] = x_exog[[i, j]];
            }
        }

        // Regressor matrix X = [x_endog | x_exog]
        let mut x = Array2::<f64>::zeros((n, k_total));
        for i in 0..n {
            for j in 0..k_endog {
                x[[i, j]] = x_endog[[i, j]];
            }
            for j in 0..k_exog {
                x[[i, k_endog + j]] = x_exog[[i, j]];
            }
        }

        // --- First stage: project X on Z ---
        // X_hat = Z (Z'Z)^{-1} Z' X
        let (_, _, zt_z_inv) = ols_fit(&z.view(), &y)?; // we only need (Z'Z)^{-1}
        // Actually we need to run OLS of X on Z to get X_hat columns
        let mut x_hat = Array2::<f64>::zeros((n, k_total));
        let mut first_stage_f = None;
        let mut partial_r2 = None;
        if k_endog > 0 {
            // For each endogenous regressor, regress on Z
            let (_, _, ztz_inv) = ols_fit(&z.view(), &Array1::zeros(n).view())?;
            // Projection matrix P_Z X:
            // beta_j = (Z'Z)^{-1} Z' x_j
            let ztz_inv_real = cholesky_invert(&z.t().dot(&z).view())?;
            let mut fs_f_sum = 0.0_f64;
            let mut pr2_sum = 0.0_f64;
            for j in 0..k_endog {
                let xj = x.column(j).to_owned();
                let coef = ztz_inv_real.dot(&z.t().dot(&xj));
                let x_hat_j = z.dot(&coef);
                for i in 0..n {
                    x_hat[[i, j]] = x_hat_j[i];
                }
                // First-stage F and partial R²
                let (fs_f, pr2) = first_stage_diagnostics(&z.view(), &xj.view(), l_excl, k_exog)?;
                fs_f_sum += fs_f;
                pr2_sum += pr2;
            }
            // Copy exogenous columns as-is
            for j in 0..k_exog {
                for i in 0..n {
                    x_hat[[i, k_endog + j]] = x_exog[[i, j]];
                }
            }
            first_stage_f = Some(fs_f_sum / k_endog as f64);
            partial_r2 = Some(pr2_sum / k_endog as f64);
            let _ = ztz_inv; // suppress unused warning
        } else {
            // No endogenous variables: 2SLS = OLS
            x_hat.assign(&x);
        }

        // --- Second stage: beta = (X_hat'X)^{-1} X_hat' y ---
        let xht_x = x_hat.t().dot(&x);
        let xht_y = x_hat.t().dot(y);
        let xhtx_inv = cholesky_invert(&xht_x.view())?;
        let beta = xhtx_inv.dot(&xht_y);

        // Residuals from structural equation
        let y_hat = x.dot(&beta);
        let residuals = y.to_owned() - &y_hat;
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let df = (n - k_total) as f64;
        if df <= 0.0 {
            return Err(StatsError::InsufficientData(
                "Insufficient degrees of freedom".into(),
            ));
        }
        let s2 = rss / df;

        // Variance-covariance matrix of beta
        let vcov = if self.robust_se {
            // HC1 sandwich estimator
            // V = n/(n-k) (X_hat'X)^{-1} [sum_i u_i^2 x_hat_i x_i'] (X_hat'X)^{-1}'
            // Simplified: use (X'X)^{-1} X' diag(u^2) X (X'X)^{-1}
            let scale = n as f64 / (n as f64 - k_total as f64);
            let mut meat = Array2::<f64>::zeros((k_total, k_total));
            for i in 0..n {
                let xi = x.row(i);
                let ui2 = residuals[i] * residuals[i];
                for r in 0..k_total {
                    for c in 0..k_total {
                        meat[[r, c]] += ui2 * xi[r] * xi[c];
                    }
                }
            }
            let bread_left = xhtx_inv.dot(&x_hat.t());
            let inner = bread_left.dot(&meat);
            inner.dot(&xhtx_inv.t()) * scale
        } else {
            // Homoskedastic: s² (X_hat'X)^{-1} X_hat'X_hat (X_hat'X)^{-1}
            // Simplification when X_hat = P_Z X: s² (X_hat'X)^{-1} approximately
            // For pure 2SLS, V(beta) ≈ s² (X_hat'X)^{-1}
            xhtx_inv.mapv(|v| v * s2)
        };

        let std_errors: Array1<f64> = (0..k_total)
            .map(|i| vcov[[i, i]].max(0.0).sqrt())
            .collect();
        let t_stats: Array1<f64> = (0..k_total)
            .map(|i| {
                if std_errors[i] > 0.0 {
                    beta[i] / std_errors[i]
                } else {
                    0.0
                }
            })
            .collect();
        let p_values: Array1<f64> = t_stats
            .iter()
            .map(|&t| t_dist_p_value(t, df))
            .collect();

        // 95 % confidence intervals
        let mut conf_intervals = Array2::<f64>::zeros((k_total, 2));
        let t_crit = t_critical(0.025, df as usize);
        for i in 0..k_total {
            conf_intervals[[i, 0]] = beta[i] - t_crit * std_errors[i];
            conf_intervals[[i, 1]] = beta[i] + t_crit * std_errors[i];
        }

        // Sargan-Hansen over-identification test (only when l_excl > k_endog)
        let (j_statistic, j_p_value) = if l_excl > k_endog {
            // Project residuals on Z; test statistic = n * R²
            let (_, residuals_z, _) = ols_fit(&z.view(), &residuals.view())?;
            let ss_res: f64 = residuals_z.iter().map(|&r| r * r).sum();
            let ss_tot: f64 = residuals.iter().map(|&r| r * r).sum();
            let r2_z = 1.0 - ss_res / ss_tot.max(1e-15);
            let j_stat = n as f64 * r2_z;
            let j_df = l_excl - k_endog;
            let j_p = chi2_p_value(j_stat, j_df);
            (Some(j_stat), Some(j_p))
        } else {
            (None, None)
        };

        Ok(IVResult {
            coefficients: beta,
            std_errors,
            t_stats,
            p_values,
            conf_intervals,
            n_obs: n,
            n_endog: k_endog,
            n_instruments: l_excl,
            first_stage_f,
            partial_r_squared: partial_r2,
            j_statistic,
            j_p_value,
            rss,
            estimator: "2SLS".into(),
        })
    }
}

/// Compute first-stage F-statistic and partial R² for a single endogenous regressor.
fn first_stage_diagnostics(
    z: &ArrayView2<f64>,
    x_endog: &ArrayView1<f64>,
    l_excl: usize,
    k_exog: usize,
) -> StatsResult<(f64, f64)> {
    let n = x_endog.len();
    let l_total = z.ncols();

    // Unrestricted: regress x_endog on Z = [z_excl | x_exog]
    let (_, resid_u, _) = ols_fit(z, x_endog)?;
    let rss_u: f64 = resid_u.iter().map(|&r| r * r).sum();

    // Restricted: regress x_endog on x_exog only
    let x_exog_mat = z.slice(scirs2_core::ndarray::s![.., l_excl..l_total]);
    let (_, resid_r, _) = ols_fit(&x_exog_mat, x_endog)?;
    let rss_r: f64 = resid_r.iter().map(|&r| r * r).sum();

    let df_num = l_excl as f64;
    let df_den = (n as f64) - (l_total as f64);
    if df_den <= 0.0 || df_num <= 0.0 {
        return Ok((0.0, 0.0));
    }
    let f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den).max(1e-15);

    // Partial R²: (RSS_r - RSS_u) / RSS_r
    let partial_r2 = if rss_r > 1e-15 {
        (rss_r - rss_u) / rss_r
    } else {
        0.0
    };

    Ok((f_stat.max(0.0), partial_r2.max(0.0).min(1.0)))
}

/// Approximate critical value of the t-distribution at `alpha` (one tail) with `df` df.
fn t_critical(alpha: f64, df: usize) -> f64 {
    // Newton-Raphson inversion of the CDF
    let df_f = df as f64;
    let mut t = 2.0_f64; // initial guess for alpha ≈ 0.025
    for _ in 0..50 {
        let p = t_dist_p_value(t, df_f);
        // p is two-tailed; we want one-tailed = alpha
        // two-tailed p = 2*alpha => p = alpha corresponds to two_tailed = 2*alpha
        let two_tail_target = 2.0 * alpha;
        let err = p - two_tail_target;
        // Approximate derivative via finite difference
        let delta = 1e-6;
        let dp = (t_dist_p_value(t + delta, df_f) - p) / delta;
        if dp.abs() < 1e-15 {
            break;
        }
        t -= err / dp;
        if err.abs() < 1e-10 {
            break;
        }
    }
    t.max(0.0)
}

// ---------------------------------------------------------------------------
// LIML (Limited Information Maximum Likelihood)
// ---------------------------------------------------------------------------

/// Limited Information Maximum Likelihood estimator.
///
/// LIML is obtained as the minimum eigenvalue solution of:
///   W'(Y'MY - λ Y'M₀Y)W = 0
/// where M and M₀ are annihilators of Z and X_exog respectively.
/// The LIML estimator is more robust to weak instruments than 2SLS.
pub struct LIML;

impl LIML {
    /// Fit the LIML estimator.
    ///
    /// See [`IVEstimator::fit`] for argument documentation.
    pub fn fit(
        y: &ArrayView1<f64>,
        x_endog: &ArrayView2<f64>,
        x_exog: &ArrayView2<f64>,
        z_excl: &ArrayView2<f64>,
    ) -> StatsResult<IVResult> {
        let n = y.len();
        let k_endog = x_endog.ncols();
        let k_exog = x_exog.ncols();
        let l_excl = z_excl.ncols();
        let k_total = k_endog + k_exog;

        if n < 2 {
            return Err(StatsError::InsufficientData(
                "Need at least 2 observations".into(),
            ));
        }
        if l_excl < k_endog {
            return Err(StatsError::InvalidArgument(format!(
                "Need at least {k_endog} instruments, got {l_excl}"
            )));
        }

        // Build [y | X_endog | X_exog]
        let mut w = Array2::<f64>::zeros((n, 1 + k_endog + k_exog));
        for i in 0..n {
            w[[i, 0]] = y[i];
            for j in 0..k_endog {
                w[[i, 1 + j]] = x_endog[[i, j]];
            }
            for j in 0..k_exog {
                w[[i, 1 + k_endog + j]] = x_exog[[i, j]];
            }
        }

        // Full instrument matrix Z = [z_excl | x_exog]
        let mut z = Array2::<f64>::zeros((n, l_excl + k_exog));
        for i in 0..n {
            for j in 0..l_excl {
                z[[i, j]] = z_excl[[i, j]];
            }
            for j in 0..k_exog {
                z[[i, l_excl + j]] = x_exog[[i, j]];
            }
        }

        // Projection M₀ = I - X_exog (X_exog'X_exog)^{-1} X_exog'
        // Annihilator w.r.t. x_exog only
        let m0_w = annihilate(x_exog, &w.view())?;
        // Projection M = I - Z (Z'Z)^{-1} Z'  (annihilator w.r.t. full Z)
        let m_w = annihilate(&z.view(), &w.view())?;

        // Solve generalized eigenvalue problem: (W'M₀W) v = λ (W'MW) v
        // The LIML eigenvalue is the minimum eigenvalue.
        // W'M₀W and W'MW are (1+k_endog+k_exog) × (1+k_endog+k_exog)
        let a = m0_w.t().dot(&m0_w); // W'M₀'M₀W = W'M₀W (M₀ idempotent)
        let b = m_w.t().dot(&m_w);   // W'M'MW = W'MW

        // We want min λ s.t. A v = λ B v  => min λ s.t. (A - λB) v = 0
        // Lambda_LIML = min eigenvalue of B^{-1} A
        let b_inv = cholesky_invert(&b.view())?;
        let c = b_inv.dot(&a);

        // Power iteration to find minimum eigenvalue of C
        let lambda_liml = min_eigenvalue(&c.view())?;

        // LIML coefficient: solve (A - λ B) v = 0 for the first element of [y, x_endog]
        // The structural equation selects the component for y as numeraire.
        // beta_LIML = (X' (M₀ - λ M) X)^{-1} X' (M₀ - λ M) y
        // where X here is x_endog | x_exog (not including y)
        let scale = lambda_liml;
        // Build X (no y column)
        let mut x = Array2::<f64>::zeros((n, k_total));
        for i in 0..n {
            for j in 0..k_endog {
                x[[i, j]] = x_endog[[i, j]];
            }
            for j in 0..k_exog {
                x[[i, k_endog + j]] = x_exog[[i, j]];
            }
        }
        // M₀ X and M X and M₀ y and M y
        let m0_x = annihilate(x_exog, &x.view())?;
        let m_x = annihilate(&z.view(), &x.view())?;
        let m0_y = annihilate_vec(x_exog, y)?;
        let m_y = annihilate_vec(&z.view(), y)?;

        // Effective moment matrix
        // lhs = X'(M₀ - λM)X,  rhs = X'(M₀ - λM)y
        let lhs_mat = m0_x.t().dot(&m0_x) - m_x.t().dot(&m_x) * scale;
        let rhs_vec = m0_x.t().dot(&m0_y) - m_x.t().dot(&m_y) * scale;
        let lhs_inv = cholesky_invert(&lhs_mat.view())?;
        let beta = lhs_inv.dot(&rhs_vec);

        let y_hat = x.dot(&beta);
        let residuals = y.to_owned() - &y_hat;
        let rss: f64 = residuals.iter().map(|&r| r * r).sum();
        let df = (n - k_total) as f64;
        let s2 = if df > 0.0 { rss / df } else { rss };

        let vcov = lhs_inv.mapv(|v| v * s2);
        let std_errors: Array1<f64> = (0..k_total)
            .map(|i| vcov[[i, i]].max(0.0).sqrt())
            .collect();
        let t_stats: Array1<f64> = (0..k_total)
            .map(|i| if std_errors[i] > 0.0 { beta[i] / std_errors[i] } else { 0.0 })
            .collect();
        let p_values: Array1<f64> = t_stats
            .iter()
            .map(|&t| t_dist_p_value(t, df.max(1.0)))
            .collect();

        let t_crit = t_critical(0.025, df as usize);
        let mut conf_intervals = Array2::<f64>::zeros((k_total, 2));
        for i in 0..k_total {
            conf_intervals[[i, 0]] = beta[i] - t_crit * std_errors[i];
            conf_intervals[[i, 1]] = beta[i] + t_crit * std_errors[i];
        }

        Ok(IVResult {
            coefficients: beta,
            std_errors,
            t_stats,
            p_values,
            conf_intervals,
            n_obs: n,
            n_endog: k_endog,
            n_instruments: l_excl,
            first_stage_f: None,
            partial_r_squared: None,
            j_statistic: None,
            j_p_value: None,
            rss,
            estimator: "LIML".into(),
        })
    }
}

/// Compute M_Z W = W - Z(Z'Z)^{-1}Z'W (annihilator projection of columns)
fn annihilate(z: &ArrayView2<f64>, w: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = w.nrows();
    let k = w.ncols();
    let mut result = w.to_owned();
    let ztz = z.t().dot(z);
    let ztz_inv = cholesky_invert(&ztz.view())?;
    let coef = ztz_inv.dot(&z.t().dot(w));
    let projection = z.dot(&coef);
    for i in 0..n {
        for j in 0..k {
            result[[i, j]] -= projection[[i, j]];
        }
    }
    Ok(result)
}

/// Annihilator for a single vector
fn annihilate_vec(z: &ArrayView2<f64>, v: &ArrayView1<f64>) -> StatsResult<Array1<f64>> {
    let ztz = z.t().dot(z);
    let ztz_inv = cholesky_invert(&ztz.view())?;
    let coef = ztz_inv.dot(&z.t().dot(v));
    let proj = z.dot(&coef);
    Ok(v.to_owned() - proj)
}

/// Find the minimum eigenvalue of a symmetric matrix via inverse-power iteration.
fn min_eigenvalue(a: &ArrayView2<f64>) -> StatsResult<f64> {
    let n = a.nrows();
    // Shift: use Gershgorin to bound the spectrum, then do inverse iteration
    let shift: f64 = a.diag().iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let mut a_shifted = a.to_owned();
    for i in 0..n {
        a_shifted[[i, i]] -= shift;
    }
    // Negate so minimum becomes maximum
    let a_neg = a_shifted.mapv(|v| -v);

    // Power iteration for dominant eigenvalue of -A_shifted
    let mut v: Array1<f64> = Array1::ones(n);
    let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
    v.mapv_inplace(|x| x / norm);

    let a_neg_inv = cholesky_invert(&a_neg.view())
        .unwrap_or_else(|_| Array2::eye(n));

    let mut lambda = 0.0_f64;
    for _ in 0..200 {
        let av = a_neg_inv.dot(&v);
        let new_lambda: f64 = v.iter().zip(av.iter()).map(|(&vi, &avi)| vi * avi).sum();
        let norm_av: f64 = av.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm_av < 1e-15 {
            break;
        }
        v = av.mapv(|x| x / norm_av);
        if (new_lambda - lambda).abs() < 1e-12 {
            lambda = new_lambda;
            break;
        }
        lambda = new_lambda;
    }
    // lambda here is 1 / (dominant eigenvalue of -A_shifted)
    // => dominant eigenvalue of -A_shifted = 1 / lambda_inv
    // Min eigenvalue of A = -max(-A_shifted) + shift - shift = -1/lambda_inv - shift
    // Actually: we compute the smallest eigenvalue directly:
    // min eigen of A = shift - dominant(A_shifted_negated)
    // Recompute more carefully using standard Rayleigh quotient
    let av_direct = a.dot(&v);
    let rayleigh: f64 = v.iter().zip(av_direct.iter()).map(|(&vi, &avi)| vi * avi).sum();
    Ok(rayleigh)
}

// ---------------------------------------------------------------------------
// Hausman Endogeneity Test
// ---------------------------------------------------------------------------

/// Hausman specification test for endogeneity.
///
/// Compares OLS and 2SLS estimators.  Under H₀ (exogeneity), both
/// estimators are consistent and the difference should be zero.
pub struct HausmanTest;

impl HausmanTest {
    /// Perform the Hausman test.
    ///
    /// # Arguments
    /// * `y`       – outcome vector
    /// * `x_endog` – suspected endogenous regressors (n × k)
    /// * `x_exog`  – exogenous controls including constant (n × p)
    /// * `z_excl`  – excluded instruments (n × l), l >= k
    ///
    /// # Returns
    /// [`HausmanResult`] with test statistic, p-value, and coefficient difference.
    pub fn test(
        y: &ArrayView1<f64>,
        x_endog: &ArrayView2<f64>,
        x_exog: &ArrayView2<f64>,
        z_excl: &ArrayView2<f64>,
    ) -> StatsResult<HausmanResult> {
        let k_endog = x_endog.ncols();
        let k_exog = x_exog.ncols();
        let k_total = k_endog + k_exog;
        let n = y.len();

        // Build full X = [x_endog | x_exog]
        let mut x = Array2::<f64>::zeros((n, k_total));
        for i in 0..n {
            for j in 0..k_endog {
                x[[i, j]] = x_endog[[i, j]];
            }
            for j in 0..k_exog {
                x[[i, k_endog + j]] = x_exog[[i, j]];
            }
        }

        // OLS estimates
        let (beta_ols, resid_ols, xtx_inv_ols) = ols_fit(&x.view(), y)?;
        let s2_ols = resid_ols.iter().map(|&r| r * r).sum::<f64>() / (n - k_total) as f64;
        let vcov_ols = xtx_inv_ols.mapv(|v| v * s2_ols);

        // 2SLS estimates
        let iv_est = IVEstimator::new(false);
        let iv_res = iv_est.fit(y, x_endog, x_exog, z_excl)?;

        // Build 2SLS vcov from std_errors (diagonal only; sufficient for Hausman)
        let k = k_endog; // only endogenous coefficients
        let diff: Array1<f64> = (0..k)
            .map(|j| beta_ols[j] - iv_res.coefficients[j])
            .collect();

        // Variance of the difference: V(beta_IV) - V(beta_OLS) (restricted to endog block)
        let mut var_diff = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            var_diff[[j, j]] = (iv_res.std_errors[j] * iv_res.std_errors[j]
                - vcov_ols[[j, j]])
            .max(1e-15);
        }
        let var_diff_inv = cholesky_invert(&var_diff.view())
            .unwrap_or_else(|_| Array2::eye(k));

        // H = diff' V^{-1} diff ~ chi²(k)
        let h_stat: f64 = (0..k)
            .map(|r| (0..k).map(|c| diff[r] * var_diff_inv[[r, c]] * diff[c]).sum::<f64>())
            .sum();
        let p_val = chi2_p_value(h_stat.max(0.0), k);

        Ok(HausmanResult {
            statistic: h_stat,
            p_value: p_val,
            df: k,
            coef_difference: diff,
            endogenous_detected: p_val < 0.05,
        })
    }
}

// ---------------------------------------------------------------------------
// Weak Instrument Test
// ---------------------------------------------------------------------------

/// First-stage F-test and partial R² for testing instrument relevance.
///
/// Uses the Stock-Yogo (2005) rule-of-thumb critical value of 10 (for a
/// single endogenous regressor).
pub struct WeakInstrumentTest;

impl WeakInstrumentTest {
    /// Perform weak-instrument diagnostics.
    ///
    /// # Arguments
    /// * `x_endog` – endogenous regressor(s) (n × k)
    /// * `x_exog`  – exogenous controls including constant (n × p)
    /// * `z_excl`  – excluded instruments (n × l)
    pub fn test(
        x_endog: &ArrayView2<f64>,
        x_exog: &ArrayView2<f64>,
        z_excl: &ArrayView2<f64>,
    ) -> StatsResult<WeakInstrumentResult> {
        let n = x_endog.nrows();
        let k_endog = x_endog.ncols();
        let k_exog = x_exog.ncols();
        let l_excl = z_excl.ncols();
        let l_total = l_excl + k_exog;

        let mut z = Array2::<f64>::zeros((n, l_total));
        for i in 0..n {
            for j in 0..l_excl {
                z[[i, j]] = z_excl[[i, j]];
            }
            for j in 0..k_exog {
                z[[i, l_excl + j]] = x_exog[[i, j]];
            }
        }

        let mut f_stats = Vec::with_capacity(k_endog);
        let mut pr2_vec = Vec::with_capacity(k_endog);
        for j in 0..k_endog {
            let xj = x_endog.column(j).to_owned();
            let (f, pr2) = first_stage_diagnostics(&z.view(), &xj.view(), l_excl, k_exog)?;
            f_stats.push(f);
            pr2_vec.push(pr2);
        }

        // Stock-Yogo critical value: approximately 10 for one endogenous regressor
        // For multiple endogenous regressors the critical value depends on the
        // number of instruments and the desired size distortion; we use 10 as
        // a conservative approximation.
        let critical_value_10pct = 10.0_f64;
        let instruments_weak: Vec<bool> = f_stats.iter().map(|&f| f < critical_value_10pct).collect();

        Ok(WeakInstrumentResult {
            f_statistics: f_stats,
            partial_r_squared: pr2_vec,
            critical_value_10pct,
            instruments_weak,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    fn ones_col(n: usize) -> Array2<f64> {
        Array2::ones((n, 1))
    }

    #[test]
    fn test_2sls_just_identified() {
        // Simple IV with one endogenous variable and one instrument.
        // True beta = 2.0 in y = 2*x + eps
        let n = 100_usize;
        // z is the instrument, x = 0.8 z + v (endogenous)
        let z_vals: Array1<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let x_vals: Array1<f64> = z_vals.mapv(|z| 0.8 * z + 0.1); // ignore noise for exact test
        let y_vals: Array1<f64> = x_vals.mapv(|x| 2.0 * x + 0.5);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n { x_endog[[i, 0]] = x_vals[i]; }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n { z_excl[[i, 0]] = z_vals[i]; }

        let est = IVEstimator::new(false);
        let res = est.fit(&y_vals.view(), &x_endog.view(), &x_exog.view(), &z_excl.view())
            .expect("2SLS fit should succeed");

        // Check the endogenous coefficient is close to 2.0
        assert!((res.coefficients[0] - 2.0).abs() < 0.1,
            "Expected beta≈2.0, got {}", res.coefficients[0]);
        assert_eq!(res.estimator, "2SLS");
    }

    #[test]
    fn test_weak_instrument_test() {
        let n = 50_usize;
        let z: Array1<f64> = (0..n).map(|i| i as f64).collect();
        let x: Array1<f64> = z.mapv(|zi| 0.01 * zi); // very weak relationship

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n { x_endog[[i, 0]] = x[i]; }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n { z_excl[[i, 0]] = z[i]; }

        let res = WeakInstrumentTest::test(&x_endog.view(), &x_exog.view(), &z_excl.view())
            .expect("Weak instrument test should succeed");
        assert_eq!(res.f_statistics.len(), 1);
        // partial R² should be between 0 and 1
        assert!(res.partial_r_squared[0] >= 0.0 && res.partial_r_squared[0] <= 1.0);
    }

    #[test]
    fn test_ols_fit_helper() {
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let (beta, resid, _) = ols_fit(&x.view(), &y.view()).expect("OLS should succeed");
        // Exact fit: beta = [0, 2]
        assert!(beta[1].abs() - 2.0 < 1e-6);
        assert!(resid.iter().all(|&r| r.abs() < 1e-6));
    }
}
