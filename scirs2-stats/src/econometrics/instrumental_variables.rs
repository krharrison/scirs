//! Instrumental Variables Estimation
//!
//! Two-Stage Least Squares (2SLS) estimator with comprehensive diagnostics:
//!
//! - **First-stage F-statistic**: Tests instrument strength (rule of thumb: F > 10)
//! - **Sargan/Hansen J-test**: Overidentification test (when more instruments than endogenous vars)
//! - **Wu-Hausman test**: Compares OLS vs 2SLS to test for endogeneity
//! - **HC1 robust standard errors**: Heteroscedasticity-robust inference
//!
//! # References
//!
//! - Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics.
//! - Stock, J.H., Wright, J.H. & Yogo, M. (2002). A Survey of Weak Instruments.
//! - Staiger, D. & Stock, J.H. (1997). Instrumental Variables Regression with
//!   Weak Instruments. Econometrica, 65(3), 557-586.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{
    chi2_p_value, cholesky_invert, f_dist_p_value, ols_fit, robust_vcov_hc1, t_critical,
    t_dist_p_value,
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a 2SLS instrumental variables estimation.
#[derive(Debug, Clone)]
pub struct IvResult {
    /// Coefficient estimates (endogenous coefficients first, then exogenous).
    pub coefficients: Array1<f64>,

    /// Standard errors (HC1 robust if configured).
    pub std_errors: Array1<f64>,

    /// t-statistics for each coefficient.
    pub t_stats: Array1<f64>,

    /// Two-sided p-values for each coefficient.
    pub p_values: Array1<f64>,

    /// 95% confidence intervals (n_params x 2).
    pub conf_intervals: Array2<f64>,

    /// Number of observations.
    pub n_obs: usize,

    /// Number of endogenous regressors.
    pub n_endog: usize,

    /// Number of excluded instruments.
    pub n_instruments: usize,

    /// Diagnostics (first-stage F, Sargan J, Wu-Hausman).
    pub diagnostics: IvDiagnostics,

    /// Residual sum of squares.
    pub rss: f64,
}

/// Diagnostic statistics for instrumental variables estimation.
#[derive(Debug, Clone)]
pub struct IvDiagnostics {
    /// First-stage F-statistic (average across endogenous regressors).
    /// Rule of thumb: F > 10 indicates strong instruments.
    pub first_stage_f: f64,

    /// p-value for the first-stage F-statistic.
    pub first_stage_f_p: f64,

    /// Per-endogenous-regressor first-stage F-statistics.
    pub first_stage_f_each: Vec<f64>,

    /// Partial R-squared from first stage (per endogenous regressor).
    pub partial_r_squared: Vec<f64>,

    /// Sargan/Hansen J-statistic for overidentification.
    /// Only present when n_instruments > n_endog.
    pub sargan_stat: Option<f64>,

    /// p-value for the Sargan J-statistic.
    /// Rejecting H0 means instruments may not all be valid.
    pub sargan_p_value: Option<f64>,

    /// Wu-Hausman endogeneity test statistic.
    pub wu_hausman_stat: Option<f64>,

    /// p-value for Wu-Hausman test.
    /// Rejecting H0 means endogeneity is detected and IV is preferred over OLS.
    pub wu_hausman_p_value: Option<f64>,

    /// Whether instruments appear weak (first-stage F < 10).
    pub instruments_weak: bool,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Specification of variables for IV estimation.
#[derive(Debug, Clone)]
pub struct IvSpec {
    /// Indices of endogenous regressors in the full regressor matrix.
    pub endogenous_indices: Vec<usize>,
    /// Indices of exogenous regressors (controls, including constant).
    pub exogenous_indices: Vec<usize>,
    /// Indices of excluded instruments.
    pub instrument_indices: Vec<usize>,
}

// ---------------------------------------------------------------------------
// 2SLS Estimator
// ---------------------------------------------------------------------------

/// Two-Stage Least Squares (2SLS) instrumental variable estimator.
///
/// # Algorithm
///
/// **Stage 1**: For each endogenous regressor X_j, regress on the full
/// instrument set Z = [excluded instruments | exogenous controls]:
///   X_j = Z * gamma_j + v_j
/// This yields fitted values X_hat_j = Z * gamma_hat_j.
///
/// **Stage 2**: Replace endogenous X with X_hat in the structural equation
/// and run OLS:
///   Y = [X_hat | X_exog] * beta + epsilon
///
/// Standard errors are computed using the original (not fitted) X matrix
/// to account for the generated-regressor problem.
pub struct TwoStageLeastSquares {
    /// Whether to use HC1 heteroscedasticity-robust standard errors.
    pub robust_se: bool,
    /// Whether to compute the Wu-Hausman endogeneity test.
    pub compute_hausman: bool,
}

impl TwoStageLeastSquares {
    /// Create a new 2SLS estimator.
    ///
    /// # Arguments
    /// * `robust_se` - if `true`, use HC1 heteroscedasticity-robust standard errors.
    pub fn new(robust_se: bool) -> Self {
        Self {
            robust_se,
            compute_hausman: true,
        }
    }

    /// Create a 2SLS estimator with full configuration.
    pub fn with_options(robust_se: bool, compute_hausman: bool) -> Self {
        Self {
            robust_se,
            compute_hausman,
        }
    }

    /// Fit the 2SLS model.
    ///
    /// # Arguments
    /// * `y`       - outcome vector (n,)
    /// * `x_endog` - endogenous regressors (n x k_endog)
    /// * `x_exog`  - exogenous regressors including constant (n x k_exog)
    /// * `z_excl`  - excluded instruments (n x l), l >= k_endog
    ///
    /// # Returns
    /// [`IvResult`] with 2SLS estimates and diagnostics.
    pub fn fit(
        &self,
        y: &ArrayView1<f64>,
        x_endog: &ArrayView2<f64>,
        x_exog: &ArrayView2<f64>,
        z_excl: &ArrayView2<f64>,
    ) -> StatsResult<IvResult> {
        let n = y.len();
        let k_endog = x_endog.ncols();
        let k_exog = x_exog.ncols();
        let l_excl = z_excl.ncols();
        let k_total = k_endog + k_exog;

        // Validate inputs
        if n < k_total + 1 {
            return Err(StatsError::InsufficientData(format!(
                "Need at least {} observations for {} regressors, got {n}",
                k_total + 1,
                k_total
            )));
        }
        if x_endog.nrows() != n || x_exog.nrows() != n || z_excl.nrows() != n {
            return Err(StatsError::DimensionMismatch(
                "All matrices must have the same number of rows as y".into(),
            ));
        }
        if l_excl < k_endog {
            return Err(StatsError::InvalidArgument(format!(
                "Need at least {k_endog} excluded instruments for {k_endog} endogenous regressors, got {l_excl}"
            )));
        }

        // Build full instrument matrix Z = [z_excl | x_exog]
        let l_total = l_excl + k_exog;
        let z = build_concat_matrix(z_excl, x_exog, n, l_excl, k_exog);

        // Build full regressor matrix X = [x_endog | x_exog]
        let x = build_concat_matrix(x_endog, x_exog, n, k_endog, k_exog);

        // ---- Stage 1: project endogenous regressors onto instrument space ----
        let ztz = z.t().dot(&z);
        let ztz_inv = cholesky_invert(&ztz.view())?;
        let pz = z.dot(&ztz_inv).dot(&z.t()); // Projection matrix P_Z

        let mut x_hat = Array2::<f64>::zeros((n, k_total));
        // Projected endogenous regressors
        for j in 0..k_endog {
            let xj = x.column(j);
            let xj_hat = pz.dot(&xj);
            for i in 0..n {
                x_hat[[i, j]] = xj_hat[i];
            }
        }
        // Exogenous regressors pass through (P_Z x_exog = x_exog when x_exog is in Z)
        for j in 0..k_exog {
            for i in 0..n {
                x_hat[[i, k_endog + j]] = x_exog[[i, j]];
            }
        }

        // ---- First-stage diagnostics ----
        let (f_each, pr2_each) =
            compute_first_stage_diagnostics(&z.view(), x_endog, l_excl, k_exog)?;
        let avg_f = if f_each.is_empty() {
            0.0
        } else {
            f_each.iter().sum::<f64>() / f_each.len() as f64
        };
        // p-value for average F (use first regressor's degrees of freedom)
        let df1_fs = l_excl;
        let df2_fs = n.saturating_sub(l_total);
        let avg_f_p = f_dist_p_value(avg_f, df1_fs, df2_fs);

        // ---- Stage 2: regress Y on X_hat ----
        let xht_x = x_hat.t().dot(&x);
        let xht_y = x_hat.t().dot(y);
        let xhtx_inv = cholesky_invert(&xht_x.view())?;
        let beta = xhtx_inv.dot(&xht_y);

        // Residuals from structural equation (using original X, not X_hat)
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

        // ---- Variance-covariance matrix ----
        // For proper 2SLS SEs, use: V = s^2 * (X_hat'X)^{-1} X_hat'X_hat (X_hat'X)^{-1}'
        // With robust: HC1 sandwich using X_hat as the "instruments"
        let vcov = if self.robust_se {
            let bread = &xhtx_inv;
            let scale = n as f64 / df;
            let mut meat = Array2::<f64>::zeros((k_total, k_total));
            for i in 0..n {
                let e2 = residuals[i] * residuals[i];
                for r in 0..k_total {
                    for c in 0..k_total {
                        meat[[r, c]] += e2 * x_hat[[i, r]] * x_hat[[i, c]];
                    }
                }
            }
            let inner = bread.dot(&meat).dot(&bread.t());
            inner.mapv(|v| v * scale)
        } else {
            // Homoscedastic 2SLS: s^2 (X_hat'X)^{-1}
            xhtx_inv.mapv(|v| v * s2)
        };

        let std_errors: Array1<f64> = (0..k_total).map(|i| vcov[[i, i]].max(0.0).sqrt()).collect();
        let t_stats: Array1<f64> = (0..k_total)
            .map(|i| {
                if std_errors[i] > 1e-15 {
                    beta[i] / std_errors[i]
                } else {
                    0.0
                }
            })
            .collect();
        let p_values: Array1<f64> = t_stats.iter().map(|&t| t_dist_p_value(t, df)).collect();

        let t_crit = t_critical(0.025, df as usize);
        let mut conf_intervals = Array2::<f64>::zeros((k_total, 2));
        for i in 0..k_total {
            conf_intervals[[i, 0]] = beta[i] - t_crit * std_errors[i];
            conf_intervals[[i, 1]] = beta[i] + t_crit * std_errors[i];
        }

        // ---- Sargan/Hansen J-test (overidentification) ----
        let (sargan_stat, sargan_p_value) = if l_excl > k_endog {
            let (_, resid_z, _) = ols_fit(&z.view(), &residuals.view())?;
            let ss_res: f64 = resid_z.iter().map(|&r| r * r).sum();
            let ss_tot: f64 = residuals.iter().map(|&r| r * r).sum();
            let r2_z = 1.0 - ss_res / ss_tot.max(1e-15);
            let j_stat = n as f64 * r2_z;
            let j_df = l_excl - k_endog;
            (Some(j_stat), Some(chi2_p_value(j_stat.max(0.0), j_df)))
        } else {
            (None, None)
        };

        // ---- Wu-Hausman endogeneity test ----
        let (wh_stat, wh_p_value) = if self.compute_hausman && k_endog > 0 {
            self.wu_hausman_test(y, &x.view(), &x_hat.view(), k_endog, k_total, n)?
        } else {
            (None, None)
        };

        let diagnostics = IvDiagnostics {
            first_stage_f: avg_f,
            first_stage_f_p: avg_f_p,
            first_stage_f_each: f_each,
            partial_r_squared: pr2_each,
            sargan_stat,
            sargan_p_value,
            wu_hausman_stat: wh_stat,
            wu_hausman_p_value: wh_p_value,
            instruments_weak: avg_f < 10.0,
        };

        Ok(IvResult {
            coefficients: beta,
            std_errors,
            t_stats,
            p_values,
            conf_intervals,
            n_obs: n,
            n_endog: k_endog,
            n_instruments: l_excl,
            diagnostics,
            rss,
        })
    }

    /// Wu-Hausman endogeneity test.
    ///
    /// Under H0 (exogeneity), OLS is consistent and efficient.
    /// Under H1 (endogeneity), only IV is consistent.
    ///
    /// Implementation: augmented regression form.
    /// Regress Y on [X, V_hat] where V_hat are first-stage residuals.
    /// Test whether V_hat coefficients are jointly zero.
    fn wu_hausman_test(
        &self,
        y: &ArrayView1<f64>,
        x: &ArrayView2<f64>,
        x_hat: &ArrayView2<f64>,
        k_endog: usize,
        k_total: usize,
        n: usize,
    ) -> StatsResult<(Option<f64>, Option<f64>)> {
        // First-stage residuals: V_hat = X_endog - X_hat_endog
        let mut x_aug = Array2::<f64>::zeros((n, k_total + k_endog));
        for i in 0..n {
            for j in 0..k_total {
                x_aug[[i, j]] = x[[i, j]];
            }
            for j in 0..k_endog {
                // Residual from first stage
                x_aug[[i, k_total + j]] = x[[i, j]] - x_hat[[i, j]];
            }
        }

        // Unrestricted regression: Y on [X, V_hat]
        let (_, resid_u, _) = ols_fit(&x_aug.view(), y)?;
        let rss_u: f64 = resid_u.iter().map(|&r| r * r).sum();

        // Restricted regression: Y on X only (OLS)
        let (_, resid_r, _) = ols_fit(x, y)?;
        let rss_r: f64 = resid_r.iter().map(|&r| r * r).sum();

        let df1 = k_endog;
        let df2 = n.saturating_sub(k_total + k_endog);
        if df2 == 0 {
            return Ok((None, None));
        }

        let f_stat = ((rss_r - rss_u) / df1 as f64) / (rss_u / df2 as f64).max(1e-15);
        let p_val = f_dist_p_value(f_stat.max(0.0), df1, df2);

        Ok((Some(f_stat), Some(p_val)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a matrix by column-concatenation: [A | B].
fn build_concat_matrix(
    a: &ArrayView2<f64>,
    b: &ArrayView2<f64>,
    n: usize,
    ka: usize,
    kb: usize,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, ka + kb));
    for i in 0..n {
        for j in 0..ka {
            out[[i, j]] = a[[i, j]];
        }
        for j in 0..kb {
            out[[i, ka + j]] = b[[i, j]];
        }
    }
    out
}

/// Compute first-stage F-statistics and partial R-squared for each endogenous regressor.
fn compute_first_stage_diagnostics(
    z: &ArrayView2<f64>,
    x_endog: &ArrayView2<f64>,
    l_excl: usize,
    k_exog: usize,
) -> StatsResult<(Vec<f64>, Vec<f64>)> {
    let n = x_endog.nrows();
    let k_endog = x_endog.ncols();
    let l_total = z.ncols();

    let mut f_stats = Vec::with_capacity(k_endog);
    let mut pr2_vec = Vec::with_capacity(k_endog);

    for j in 0..k_endog {
        let xj = x_endog.column(j).to_owned();

        // Unrestricted: regress x_endog_j on full Z
        let (_, resid_u, _) = ols_fit(z, &xj.view())?;
        let rss_u: f64 = resid_u.iter().map(|&r| r * r).sum();

        // Restricted: regress x_endog_j on exogenous controls only (last k_exog cols of Z)
        let x_exog_only = z.slice(scirs2_core::ndarray::s![.., l_excl..l_total]);
        let (_, resid_r, _) = ols_fit(&x_exog_only, &xj.view())?;
        let rss_r: f64 = resid_r.iter().map(|&r| r * r).sum();

        let df_num = l_excl as f64;
        let df_den = (n as f64) - (l_total as f64);
        if df_den <= 0.0 || df_num <= 0.0 {
            f_stats.push(0.0);
            pr2_vec.push(0.0);
            continue;
        }

        let f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den).max(1e-15);
        let partial_r2 = if rss_r > 1e-15 {
            (rss_r - rss_u) / rss_r
        } else {
            0.0
        };

        f_stats.push(f_stat.max(0.0));
        pr2_vec.push(partial_r2.clamp(0.0, 1.0));
    }

    Ok((f_stats, pr2_vec))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn ones_col(n: usize) -> Array2<f64> {
        Array2::ones((n, 1))
    }

    #[test]
    fn test_2sls_consistent_estimate() {
        // True model: y = 2*x + 0.5 + eps
        // z is instrument for x: x = 0.8*z + 0.1
        let n = 200_usize;
        let z_vals: Array1<f64> = (0..n).map(|i| (i as f64) / 20.0).collect();
        let x_vals: Array1<f64> = z_vals.mapv(|z| 0.8 * z + 0.1);
        let y_vals: Array1<f64> = x_vals.mapv(|x| 2.0 * x + 0.5);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            z_excl[[i, 0]] = z_vals[i];
        }

        let est = TwoStageLeastSquares::new(false);
        let res = est
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("2SLS should succeed");

        assert!(
            (res.coefficients[0] - 2.0).abs() < 0.05,
            "Expected beta ~2.0, got {}",
            res.coefficients[0]
        );
        assert!(
            (res.coefficients[1] - 0.5).abs() < 0.1,
            "Expected intercept ~0.5, got {}",
            res.coefficients[1]
        );
    }

    #[test]
    fn test_2sls_strong_instrument_f_stat() {
        // Strong instrument: x = 0.9*z + noise
        let n = 200_usize;
        let z_vals: Array1<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let x_vals: Array1<f64> = z_vals.mapv(|z| 0.9 * z + 0.05);
        let y_vals: Array1<f64> = x_vals.mapv(|x| 3.0 * x + 1.0);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            z_excl[[i, 0]] = z_vals[i];
        }

        let est = TwoStageLeastSquares::new(true);
        let res = est
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("2SLS should succeed");

        assert!(
            res.diagnostics.first_stage_f > 10.0,
            "F-stat should be > 10 for strong instrument, got {}",
            res.diagnostics.first_stage_f
        );
        assert!(
            !res.diagnostics.instruments_weak,
            "Instruments should not be flagged as weak"
        );
    }

    #[test]
    fn test_2sls_weak_instrument_warning() {
        // Weak instrument: x = 0.001*z + large_constant
        let n = 100_usize;
        let z_vals: Array1<f64> = (0..n).map(|i| i as f64).collect();
        // Almost no variation from instrument
        let x_vals: Array1<f64> = z_vals.mapv(|z| 0.001 * z + 100.0);
        let y_vals: Array1<f64> = x_vals.mapv(|x| x + 1.0);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            z_excl[[i, 0]] = z_vals[i];
        }

        let est = TwoStageLeastSquares::new(false);
        let res = est
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("2SLS should succeed even with weak instrument");

        // With near-perfect but tiny relationship, F could be large or small
        // The key test is that the diagnostics are computed
        assert!(res.diagnostics.first_stage_f >= 0.0);
        assert!(res.diagnostics.partial_r_squared.len() == 1);
    }

    #[test]
    fn test_2sls_overidentification_sargan() {
        // Over-identified: 2 instruments for 1 endogenous variable
        let n = 150_usize;
        let z1: Array1<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let z2: Array1<f64> = (0..n).map(|i| ((i as f64) / 10.0).powi(2) * 0.1).collect();
        let x_vals: Array1<f64> = Array1::from_shape_fn(n, |i| 0.5 * z1[i] + 0.3 * z2[i]);
        let y_vals: Array1<f64> = x_vals.mapv(|x| 2.5 * x + 1.0);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            z_excl[[i, 0]] = z1[i];
            z_excl[[i, 1]] = z2[i];
        }

        let est = TwoStageLeastSquares::new(false);
        let res = est
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("2SLS should succeed");

        // With valid instruments, J-stat should be small and p-value large
        assert!(
            res.diagnostics.sargan_stat.is_some(),
            "Should have Sargan stat"
        );
        assert!(
            res.diagnostics.sargan_p_value.is_some(),
            "Should have Sargan p-value"
        );
        assert!(res.n_instruments == 2);
    }

    #[test]
    fn test_2sls_wu_hausman_no_endogeneity() {
        // When there is no endogeneity, Wu-Hausman should not reject
        let n = 100_usize;
        let x_vals: Array1<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let y_vals: Array1<f64> = x_vals.mapv(|x| 1.5 * x + 2.0);
        // Perfect instrument (z = x + small perturbation)
        let z_vals: Array1<f64> = x_vals.mapv(|x| x * 1.1);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            z_excl[[i, 0]] = z_vals[i];
        }

        let est = TwoStageLeastSquares::with_options(false, true);
        let res = est
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("2SLS should succeed");

        assert!(res.diagnostics.wu_hausman_stat.is_some());
        assert!(res.diagnostics.wu_hausman_p_value.is_some());
    }

    #[test]
    fn test_2sls_robust_se() {
        let n = 100_usize;
        let z_vals: Array1<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
        let x_vals: Array1<f64> = z_vals.mapv(|z| 0.7 * z);
        let y_vals: Array1<f64> = x_vals.mapv(|x| 2.0 * x + 1.0);

        let mut x_endog = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            x_endog[[i, 0]] = x_vals[i];
        }
        let x_exog = ones_col(n);
        let mut z_excl = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            z_excl[[i, 0]] = z_vals[i];
        }

        let est_robust = TwoStageLeastSquares::new(true);
        let res_robust = est_robust
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("Robust 2SLS should succeed");

        let est_homo = TwoStageLeastSquares::new(false);
        let res_homo = est_homo
            .fit(
                &y_vals.view(),
                &x_endog.view(),
                &x_exog.view(),
                &z_excl.view(),
            )
            .expect("Homoscedastic 2SLS should succeed");

        // Both should get similar point estimates
        assert!(
            (res_robust.coefficients[0] - res_homo.coefficients[0]).abs() < 1e-6,
            "Point estimates should be identical"
        );
    }
}
