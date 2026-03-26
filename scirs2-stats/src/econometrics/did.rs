//! Difference-in-Differences Methods
//!
//! Provides three DiD estimators:
//!
//! - **[`DidEstimator`]**: Classic 2x2 DiD and regression-based DiD
//! - **[`EventStudyEstimator`]**: Dynamic treatment effects with leads and lags
//! - **[`StaggeredDidEstimator`]**: Callaway-Sant'Anna style staggered treatment
//!
//! # Key Concepts
//!
//! The **ATT** (Average Treatment Effect on the Treated) is identified by the
//! parallel trends assumption: absent treatment, treated and control groups
//! would have followed the same trend.
//!
//! In the regression framework:
//!   Y = beta_0 + beta_1*Treat + beta_2*Post + beta_3*(Treat x Post) + eps
//!
//! beta_3 is the ATT.
//!
//! # References
//!
//! - Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods. Journal of Econometrics.
//! - Roth, J. et al. (2023). What's Trending in Difference-in-Differences?

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::{
    cholesky_invert, clustered_vcov, normal_p_value, normal_quantile, ols_fit, t_critical,
    t_dist_p_value,
};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a Difference-in-Differences estimation.
#[derive(Debug, Clone)]
pub struct DidResult {
    /// Average Treatment Effect on the Treated.
    pub att: f64,

    /// Standard error of the ATT.
    pub se: f64,

    /// t-statistic.
    pub t_stat: f64,

    /// Two-sided p-value.
    pub p_value: f64,

    /// 95% confidence interval [lower, upper].
    pub confidence_interval: [f64; 2],

    /// Parallel trends test statistic (F or t statistic).
    /// Large values / small p-values indicate violation of parallel trends.
    pub parallel_trends_stat: Option<f64>,

    /// p-value for parallel trends test.
    /// A large p-value (> 0.05) is consistent with parallel trends holding.
    pub parallel_trends_p: Option<f64>,

    /// Number of treated units.
    pub n_treated: usize,

    /// Number of control units.
    pub n_control: usize,

    /// Regression coefficients: [intercept, treat, post, treat*post].
    pub regression_coefs: Option<Array1<f64>>,

    /// Estimator description.
    pub estimator: String,
}

/// Coefficient from an event study specification.
#[derive(Debug, Clone)]
pub struct EventStudyCoefficient {
    /// Relative time (negative = pre-treatment, 0 = treatment, positive = post).
    pub relative_time: i64,
    /// Point estimate.
    pub estimate: f64,
    /// Standard error.
    pub std_error: f64,
    /// Two-sided p-value.
    pub p_value: f64,
    /// 95% confidence interval.
    pub conf_interval: [f64; 2],
}

/// Result of an event study specification.
#[derive(Debug, Clone)]
pub struct EventStudyResult {
    /// Coefficients for each relative time period.
    pub coefficients: Vec<EventStudyCoefficient>,
    /// Joint F-test for pre-treatment coefficients = 0.
    pub pre_trend_f: f64,
    /// p-value for the pre-trend F-test.
    pub pre_trend_p: f64,
}

/// ATT for a specific (cohort, time) pair in staggered DiD.
#[derive(Debug, Clone)]
pub struct StaggeredAttGt {
    /// Cohort (first treatment period).
    pub cohort: i64,
    /// Calendar period.
    pub period: i64,
    /// ATT estimate.
    pub att: f64,
    /// Standard error.
    pub std_error: f64,
    /// p-value.
    pub p_value: f64,
}

/// Result of staggered DiD estimation.
#[derive(Debug, Clone)]
pub struct StaggeredDidResult {
    /// Individual ATT(g,t) estimates.
    pub att_gt: Vec<StaggeredAttGt>,
    /// Aggregate ATT (weighted average of post-treatment ATT(g,t)).
    pub aggregate_att: f64,
    /// Standard error of aggregate ATT.
    pub aggregate_se: f64,
    /// p-value for aggregate ATT.
    pub aggregate_p: f64,
}

// ---------------------------------------------------------------------------
// Classic DiD Estimator
// ---------------------------------------------------------------------------

/// Difference-in-Differences estimator.
///
/// Supports both the classic 2x2 comparison and the regression-based approach.
pub struct DidEstimator {
    /// Whether to cluster standard errors by group.
    pub cluster_se: bool,
}

impl DidEstimator {
    /// Create a new DiD estimator.
    pub fn new(cluster_se: bool) -> Self {
        Self { cluster_se }
    }

    /// Estimate ATT using the classic 2x2 DiD formula.
    ///
    /// ATT = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
    ///
    /// # Arguments
    /// * `y_treat_pre`   - outcomes for treated group, pre-treatment
    /// * `y_treat_post`  - outcomes for treated group, post-treatment
    /// * `y_control_pre` - outcomes for control group, pre-treatment
    /// * `y_control_post`- outcomes for control group, post-treatment
    pub fn estimate_classic(
        &self,
        y_treat_pre: &[f64],
        y_treat_post: &[f64],
        y_control_pre: &[f64],
        y_control_post: &[f64],
    ) -> StatsResult<DidResult> {
        if y_treat_pre.is_empty() || y_treat_post.is_empty() {
            return Err(StatsError::InsufficientData(
                "Need at least one treated observation in each period".into(),
            ));
        }
        if y_control_pre.is_empty() || y_control_post.is_empty() {
            return Err(StatsError::InsufficientData(
                "Need at least one control observation in each period".into(),
            ));
        }

        let n_t = y_treat_pre.len() + y_treat_post.len();
        let n_c = y_control_pre.len() + y_control_post.len();

        let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let var = |s: &[f64], m: f64| {
            if s.len() <= 1 {
                return 0.0;
            }
            s.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (s.len() - 1) as f64
        };

        let m_tp = mean(y_treat_post);
        let m_t0 = mean(y_treat_pre);
        let m_cp = mean(y_control_post);
        let m_c0 = mean(y_control_pre);

        let att = (m_tp - m_t0) - (m_cp - m_c0);

        // Standard error via variance of the difference-in-differences
        let v_tp = var(y_treat_post, m_tp) / y_treat_post.len() as f64;
        let v_t0 = var(y_treat_pre, m_t0) / y_treat_pre.len() as f64;
        let v_cp = var(y_control_post, m_cp) / y_control_post.len() as f64;
        let v_c0 = var(y_control_pre, m_c0) / y_control_pre.len() as f64;
        let se = (v_tp + v_t0 + v_cp + v_c0).sqrt();

        let t_stat = if se > 1e-15 { att / se } else { 0.0 };
        let total_n = n_t + n_c;
        let df = (total_n.saturating_sub(4)) as f64;
        let p_value = if df > 0.0 {
            t_dist_p_value(t_stat, df)
        } else {
            normal_p_value(t_stat)
        };

        let z_crit = normal_quantile(0.975);
        let ci = [att - z_crit * se, att + z_crit * se];

        Ok(DidResult {
            att,
            se,
            t_stat,
            p_value,
            confidence_interval: ci,
            parallel_trends_stat: None,
            parallel_trends_p: None,
            n_treated: n_t / 2, // approximate per-period count
            n_control: n_c / 2,
            regression_coefs: None,
            estimator: "DiD-Classic".into(),
        })
    }

    /// Estimate ATT using regression-based DiD.
    ///
    /// Fits: Y = beta_0 + beta_1*Treat + beta_2*Post + beta_3*(Treat*Post) + eps
    ///
    /// beta_3 is the ATT.
    ///
    /// # Arguments
    /// * `y`            - outcome vector (n_obs,)
    /// * `treat`        - treatment indicator per observation (0 or 1)
    /// * `post`         - post-period indicator per observation (0 or 1)
    /// * `cluster_ids`  - optional cluster identifiers for clustered SEs
    pub fn estimate_regression(
        &self,
        y: &ArrayView1<f64>,
        treat: &ArrayView1<f64>,
        post: &ArrayView1<f64>,
        cluster_ids: Option<&[usize]>,
    ) -> StatsResult<DidResult> {
        let n = y.len();
        if treat.len() != n || post.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y, treat, and post must have the same length".into(),
            ));
        }
        if n < 5 {
            return Err(StatsError::InsufficientData(
                "Need at least 5 observations".into(),
            ));
        }

        // Design matrix: [1, treat, post, treat*post]
        let k = 4;
        let mut xmat = Array2::<f64>::zeros((n, k));
        for i in 0..n {
            xmat[[i, 0]] = 1.0;
            xmat[[i, 1]] = treat[i];
            xmat[[i, 2]] = post[i];
            xmat[[i, 3]] = treat[i] * post[i];
        }

        let (beta, resid, xtx_inv) = ols_fit(&xmat.view(), y)?;
        let att = beta[3]; // Treat*Post coefficient

        let df = (n - k) as f64;
        let vcov = if self.cluster_se {
            if let Some(cids) = cluster_ids {
                clustered_vcov(&xmat.view(), &resid, cids, &xtx_inv)
            } else {
                // Default: cluster by treatment group
                let cids: Vec<usize> = treat.iter().map(|&t| if t > 0.5 { 1 } else { 0 }).collect();
                clustered_vcov(&xmat.view(), &resid, &cids, &xtx_inv)
            }
        } else {
            let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
            xtx_inv.mapv(|v| v * s2)
        };

        let se = vcov[[3, 3]].max(0.0).sqrt();
        let t_stat = if se > 1e-15 { att / se } else { 0.0 };
        let p_value = t_dist_p_value(t_stat, df);
        let t_crit = t_critical(0.025, df as usize);
        let ci = [att - t_crit * se, att + t_crit * se];

        let n_treated = treat.iter().filter(|&&t| t > 0.5).count();
        let n_control = n - n_treated;

        Ok(DidResult {
            att,
            se,
            t_stat,
            p_value,
            confidence_interval: ci,
            parallel_trends_stat: None,
            parallel_trends_p: None,
            n_treated,
            n_control,
            regression_coefs: Some(beta),
            estimator: "DiD-Regression".into(),
        })
    }

    /// Estimate ATT from panel data with parallel trends test.
    ///
    /// # Arguments
    /// * `y`            - outcome vector (n_units * n_periods, row-major by unit)
    /// * `treated`      - binary indicator per unit (n_units,)
    /// * `n_units`      - number of units
    /// * `n_periods`    - number of time periods
    /// * `treat_period` - first treatment period (0-indexed)
    pub fn estimate_panel(
        &self,
        y: &ArrayView1<f64>,
        treated: &ArrayView1<f64>,
        n_units: usize,
        n_periods: usize,
        treat_period: usize,
    ) -> StatsResult<DidResult> {
        let n = n_units * n_periods;
        if y.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "y length {} != n_units * n_periods = {}",
                y.len(),
                n
            )));
        }
        if treated.len() != n_units {
            return Err(StatsError::DimensionMismatch(
                "treated must have length n_units".into(),
            ));
        }
        if treat_period == 0 || treat_period >= n_periods {
            return Err(StatsError::InvalidArgument(
                "treat_period must be between 1 and n_periods-1".into(),
            ));
        }

        // Build flat vectors for regression-based DiD
        let mut y_flat = Array1::<f64>::zeros(n);
        let mut treat_flat = Array1::<f64>::zeros(n);
        let mut post_flat = Array1::<f64>::zeros(n);
        let mut cluster_ids = Vec::with_capacity(n);

        for i in 0..n_units {
            for t in 0..n_periods {
                let idx = i * n_periods + t;
                y_flat[idx] = y[idx];
                treat_flat[idx] = treated[i];
                post_flat[idx] = if t >= treat_period { 1.0 } else { 0.0 };
                cluster_ids.push(i);
            }
        }

        let cids = if self.cluster_se {
            Some(cluster_ids.as_slice())
        } else {
            None
        };

        let mut result =
            self.estimate_regression(&y_flat.view(), &treat_flat.view(), &post_flat.view(), cids)?;

        // Parallel trends test: check pre-treatment trends
        if treat_period > 1 {
            let pt = self.parallel_trends_test(y, treated, n_units, n_periods, treat_period)?;
            result.parallel_trends_stat = Some(pt.0);
            result.parallel_trends_p = Some(pt.1);
        }

        result.estimator = "DiD-Panel".into();
        Ok(result)
    }

    /// Pre-treatment parallel trends test.
    ///
    /// Regresses pre-treatment outcomes on treat * time_trend.
    /// Under parallel trends, the interaction should be zero.
    fn parallel_trends_test(
        &self,
        y: &ArrayView1<f64>,
        treated: &ArrayView1<f64>,
        n_units: usize,
        n_periods: usize,
        treat_period: usize,
    ) -> StatsResult<(f64, f64)> {
        let n_pre = n_units * treat_period;
        if n_pre < 5 {
            return Ok((0.0, 1.0));
        }

        let k = 3; // [1, time, treat*time]
        let mut xmat = Array2::<f64>::zeros((n_pre, k));
        let mut y_pre = Array1::<f64>::zeros(n_pre);
        let mut row = 0;
        for i in 0..n_units {
            for t in 0..treat_period {
                y_pre[row] = y[i * n_periods + t];
                xmat[[row, 0]] = 1.0;
                xmat[[row, 1]] = t as f64;
                xmat[[row, 2]] = treated[i] * (t as f64);
                row += 1;
            }
        }

        let (beta, resid, xtx_inv) = ols_fit(&xmat.view(), &y_pre.view())?;
        let df = (n_pre - k) as f64;
        let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
        let se = (xtx_inv[[2, 2]] * s2).max(0.0).sqrt();
        let t = if se > 1e-15 { beta[2] / se } else { 0.0 };
        let p = t_dist_p_value(t, df);

        Ok((t.abs(), p))
    }
}

impl Default for DidEstimator {
    fn default() -> Self {
        Self::new(false)
    }
}

// ---------------------------------------------------------------------------
// Event Study Estimator
// ---------------------------------------------------------------------------

/// Event study specification for dynamic treatment effects.
///
/// Estimates leads and lags around the treatment date:
///   Y_{it} = sum_{l != -1} beta_l * D^l_{it} + alpha_i + delta_t + eps_{it}
///
/// where D^l_{it} = 1 if unit i is treated and is l periods from treatment.
/// Period l = -1 is the omitted reference period.
pub struct EventStudyEstimator {
    /// Number of pre-treatment leads (>= 1).
    pub n_leads: usize,
    /// Number of post-treatment lags (>= 1).
    pub n_lags: usize,
}

impl EventStudyEstimator {
    /// Create a new event study estimator.
    pub fn new(n_leads: usize, n_lags: usize) -> Self {
        Self {
            n_leads: n_leads.max(1),
            n_lags: n_lags.max(1),
        }
    }

    /// Estimate dynamic treatment effects.
    ///
    /// # Arguments
    /// * `y`            - outcome (n_units * n_periods, row-major by unit)
    /// * `treated`      - binary indicator per unit
    /// * `n_units`      - number of units
    /// * `n_periods`    - number of periods
    /// * `treat_period` - first treatment period (0-indexed)
    pub fn estimate(
        &self,
        y: &ArrayView1<f64>,
        treated: &ArrayView1<f64>,
        n_units: usize,
        n_periods: usize,
        treat_period: usize,
    ) -> StatsResult<EventStudyResult> {
        let n = n_units * n_periods;
        if y.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y length != n_units * n_periods".into(),
            ));
        }
        if treated.len() != n_units {
            return Err(StatsError::DimensionMismatch(
                "treated must have length n_units".into(),
            ));
        }

        // Event times: [-n_leads, ..., -2, 0, 1, ..., n_lags-1]
        // Omit l = -1 (reference)
        let event_times: Vec<i64> = (-(self.n_leads as i64)..=(self.n_lags as i64 - 1))
            .filter(|&l| l != -1)
            .collect();
        let n_event = event_times.len();

        // Design: [unit FE (n_units-1), time FE (n_periods-1), event dummies]
        let k = (n_units - 1) + (n_periods - 1) + n_event;
        if n <= k {
            return Err(StatsError::InsufficientData(format!(
                "Not enough observations ({n}) for {k} regressors"
            )));
        }

        let mut xmat = Array2::<f64>::zeros((n, k));
        let mut y_vec = Array1::<f64>::zeros(n);

        for i in 0..n_units {
            for t in 0..n_periods {
                let row = i * n_periods + t;
                y_vec[row] = y[row];

                // Unit FE (omit unit 0)
                if i > 0 {
                    xmat[[row, i - 1]] = 1.0;
                }
                // Time FE (omit period 0)
                if t > 0 {
                    xmat[[row, (n_units - 1) + t - 1]] = 1.0;
                }
                // Event-time dummies
                if treated[i] > 0.5 {
                    let rel_time = (t as i64) - (treat_period as i64);
                    for (d_idx, &et) in event_times.iter().enumerate() {
                        if rel_time == et {
                            xmat[[row, (n_units - 1) + (n_periods - 1) + d_idx]] = 1.0;
                        }
                    }
                }
            }
        }

        let (beta, resid, xtx_inv) = ols_fit(&xmat.view(), &y_vec.view())?;
        let df = (n - k) as f64;
        let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
        let t_crit = t_critical(0.025, df as usize);
        let fe_offset = (n_units - 1) + (n_periods - 1);

        let mut coefficients = Vec::with_capacity(n_event);
        for (d_idx, &et) in event_times.iter().enumerate() {
            let cidx = fe_offset + d_idx;
            let est = beta[cidx];
            let se = (xtx_inv[[cidx, cidx]] * s2).max(0.0).sqrt();
            let t = if se > 1e-15 { est / se } else { 0.0 };
            let p = t_dist_p_value(t, df);
            coefficients.push(EventStudyCoefficient {
                relative_time: et,
                estimate: est,
                std_error: se,
                p_value: p,
                conf_interval: [est - t_crit * se, est + t_crit * se],
            });
        }

        // Pre-trend F-test: joint test that all pre-treatment coefficients = 0
        let n_pre_coefs = event_times.iter().filter(|&&et| et < -1).count();
        let (pre_f, pre_p) = if n_pre_coefs > 0 {
            self.pre_trend_f_test(&xmat, &y_vec, &resid, fe_offset, n_pre_coefs, n, k, df)?
        } else {
            (0.0, 1.0)
        };

        Ok(EventStudyResult {
            coefficients,
            pre_trend_f: pre_f,
            pre_trend_p: pre_p,
        })
    }

    /// Joint F-test for pre-treatment coefficients.
    fn pre_trend_f_test(
        &self,
        xmat: &Array2<f64>,
        y_vec: &Array1<f64>,
        resid_ur: &Array1<f64>,
        fe_offset: usize,
        n_pre: usize,
        n: usize,
        k: usize,
        df: f64,
    ) -> StatsResult<(f64, f64)> {
        let rss_ur: f64 = resid_ur.iter().map(|&r| r * r).sum();

        // Restricted: drop pre-treatment dummies
        let pre_cols: Vec<usize> = (fe_offset..fe_offset + n_pre).collect();
        let cols_r: Vec<usize> = (0..k).filter(|c| !pre_cols.contains(c)).collect();
        let mut xr = Array2::<f64>::zeros((n, cols_r.len()));
        for (new_j, &old_j) in cols_r.iter().enumerate() {
            for i in 0..n {
                xr[[i, new_j]] = xmat[[i, old_j]];
            }
        }
        let (_, resid_r, _) = ols_fit(&xr.view(), &y_vec.view())?;
        let rss_r: f64 = resid_r.iter().map(|&r| r * r).sum();

        let f = ((rss_r - rss_ur) / n_pre as f64) / (rss_ur / df).max(1e-15);
        let p = super::f_dist_p_value(f.max(0.0), n_pre, df as usize);

        Ok((f, p))
    }
}

// ---------------------------------------------------------------------------
// Staggered DiD (Callaway-Sant'Anna style)
// ---------------------------------------------------------------------------

/// Staggered Difference-in-Differences estimator.
///
/// Handles multiple treatment cohorts (units treated at different times)
/// using the Callaway-Sant'Anna (2021) approach.
///
/// For each (cohort g, period t) pair, estimates ATT(g,t) using
/// a "not yet treated" comparison group.
pub struct StaggeredDidEstimator {
    /// Random seed for any randomization.
    pub seed: u64,
}

impl StaggeredDidEstimator {
    /// Create a new staggered DiD estimator.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Estimate ATT(g,t) for all cohort-period pairs.
    ///
    /// # Arguments
    /// * `y`        - outcome matrix (n_units x n_periods)
    /// * `cohorts`  - cohort vector: `cohorts[i]` = first treatment period for unit i.
    ///                Use i64::MAX for never-treated units.
    /// * `n_units`  - number of units
    /// * `n_periods`- number of calendar periods
    pub fn estimate(
        &self,
        y: &ArrayView2<f64>,
        cohorts: &[i64],
        n_units: usize,
        n_periods: usize,
    ) -> StatsResult<StaggeredDidResult> {
        if y.nrows() != n_units || y.ncols() != n_periods {
            return Err(StatsError::DimensionMismatch(
                "y must be (n_units x n_periods)".into(),
            ));
        }
        if cohorts.len() != n_units {
            return Err(StatsError::DimensionMismatch(
                "cohorts must have length n_units".into(),
            ));
        }

        // Collect unique treatment cohorts
        let mut unique_cohorts: Vec<i64> = cohorts
            .iter()
            .filter(|&&g| g < i64::MAX && g >= 0)
            .copied()
            .collect::<std::collections::HashSet<i64>>()
            .into_iter()
            .collect();
        unique_cohorts.sort();

        if unique_cohorts.is_empty() {
            return Err(StatsError::InvalidArgument(
                "No treated cohorts found (all units are never-treated)".into(),
            ));
        }

        let mut att_gt_vec: Vec<StaggeredAttGt> = Vec::new();

        for &g in &unique_cohorts {
            let treated_ids: Vec<usize> = (0..n_units).filter(|&i| cohorts[i] == g).collect();

            // Reference period: g - 1
            let t_ref = (g - 1) as usize;
            if t_ref >= n_periods {
                continue;
            }

            for t in 0..n_periods {
                let t_i64 = t as i64;

                // "Not yet treated" comparison group
                let control_ids: Vec<usize> = (0..n_units)
                    .filter(|&i| cohorts[i] == i64::MAX || cohorts[i] > t_i64)
                    .collect();

                if treated_ids.is_empty() || control_ids.is_empty() {
                    continue;
                }

                let (att, se) = self.compute_att_gt(y, &treated_ids, &control_ids, t, t_ref)?;

                let z = if se > 1e-15 { att / se } else { 0.0 };
                let p = normal_p_value(z);

                att_gt_vec.push(StaggeredAttGt {
                    cohort: g,
                    period: t_i64,
                    att,
                    std_error: se,
                    p_value: p,
                });
            }
        }

        if att_gt_vec.is_empty() {
            return Err(StatsError::InsufficientData(
                "No valid (cohort, period) pairs could be estimated".into(),
            ));
        }

        // Aggregate: average over post-treatment ATTs
        let post_atts: Vec<&StaggeredAttGt> = att_gt_vec
            .iter()
            .filter(|ag| ag.period >= ag.cohort)
            .collect();

        let (aggregate_att, aggregate_se) = if post_atts.is_empty() {
            (0.0, 0.0)
        } else {
            let n_post = post_atts.len() as f64;
            let agg = post_atts.iter().map(|ag| ag.att).sum::<f64>() / n_post;
            let var_sum: f64 = post_atts.iter().map(|ag| ag.std_error * ag.std_error).sum();
            let se = (var_sum / (n_post * n_post)).sqrt();
            (agg, se)
        };

        let aggregate_p = normal_p_value(if aggregate_se > 1e-15 {
            aggregate_att / aggregate_se
        } else {
            0.0
        });

        Ok(StaggeredDidResult {
            att_gt: att_gt_vec,
            aggregate_att,
            aggregate_se,
            aggregate_p,
        })
    }

    /// Compute ATT(g,t) for a specific cohort-period pair.
    fn compute_att_gt(
        &self,
        y: &ArrayView2<f64>,
        treated_ids: &[usize],
        control_ids: &[usize],
        t: usize,
        t_ref: usize,
    ) -> StatsResult<(f64, f64)> {
        let n_t = treated_ids.len();
        let n_c = control_ids.len();

        // Delta Y = y_t - y_{t_ref}
        let delta_t: Vec<f64> = treated_ids
            .iter()
            .map(|&i| y[[i, t]] - y[[i, t_ref]])
            .collect();
        let delta_c: Vec<f64> = control_ids
            .iter()
            .map(|&i| y[[i, t]] - y[[i, t_ref]])
            .collect();

        let mean_t = delta_t.iter().sum::<f64>() / n_t as f64;
        let mean_c = delta_c.iter().sum::<f64>() / n_c as f64;
        let att = mean_t - mean_c;

        let var_t = if n_t > 1 {
            delta_t.iter().map(|&v| (v - mean_t).powi(2)).sum::<f64>() / (n_t * (n_t - 1)) as f64
        } else {
            0.0
        };
        let var_c = if n_c > 1 {
            delta_c.iter().map(|&v| (v - mean_c).powi(2)).sum::<f64>() / (n_c * (n_c - 1)) as f64
        } else {
            0.0
        };
        let se = (var_t + var_c).sqrt();

        Ok((att, se))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_did_classic_known_effect() {
        // Treatment effect = 5
        let y_treat_pre = vec![10.0, 11.0, 12.0, 10.5];
        let y_treat_post = vec![15.0, 16.0, 17.0, 15.5]; // +5 from treatment
        let y_control_pre = vec![10.0, 11.0, 12.0, 10.5];
        let y_control_post = vec![10.0, 11.0, 12.0, 10.5]; // no change

        let est = DidEstimator::new(false);
        let res = est
            .estimate_classic(&y_treat_pre, &y_treat_post, &y_control_pre, &y_control_post)
            .expect("DiD classic should succeed");

        assert!(
            (res.att - 5.0).abs() < 0.5,
            "ATT should be ~5.0, got {}",
            res.att
        );
    }

    #[test]
    fn test_did_classic_no_effect() {
        let y_treat_pre = vec![10.0, 11.0, 12.0];
        let y_treat_post = vec![10.0, 11.0, 12.0];
        let y_control_pre = vec![10.0, 11.0, 12.0];
        let y_control_post = vec![10.0, 11.0, 12.0];

        let est = DidEstimator::new(false);
        let res = est
            .estimate_classic(&y_treat_pre, &y_treat_post, &y_control_pre, &y_control_post)
            .expect("DiD classic should succeed");

        assert!(
            res.att.abs() < 0.01,
            "ATT should be ~0 with no effect, got {}",
            res.att
        );
    }

    #[test]
    fn test_did_regression_known_effect() {
        // 20 obs: 10 treated (5 pre, 5 post), 10 control (5 pre, 5 post)
        let n = 20;
        let te = 3.0_f64;
        let mut y = Array1::<f64>::zeros(n);
        let mut treat = Array1::<f64>::zeros(n);
        let mut post = Array1::<f64>::zeros(n);

        // Treated pre (indices 0-4)
        for i in 0..5 {
            y[i] = 10.0;
            treat[i] = 1.0;
            post[i] = 0.0;
        }
        // Treated post (indices 5-9)
        for i in 5..10 {
            y[i] = 10.0 + te;
            treat[i] = 1.0;
            post[i] = 1.0;
        }
        // Control pre (indices 10-14)
        for i in 10..15 {
            y[i] = 10.0;
            treat[i] = 0.0;
            post[i] = 0.0;
        }
        // Control post (indices 15-19)
        for i in 15..20 {
            y[i] = 10.0;
            treat[i] = 0.0;
            post[i] = 1.0;
        }

        let est = DidEstimator::new(false);
        let res = est
            .estimate_regression(&y.view(), &treat.view(), &post.view(), None)
            .expect("DiD regression should succeed");

        assert!(
            (res.att - te).abs() < 0.1,
            "ATT should be ~{te}, got {}",
            res.att
        );
    }

    #[test]
    fn test_did_panel_parallel_trends() {
        let n_units = 6;
        let n_periods = 6;
        let treat_period = 3;
        let treated = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let te = 4.0_f64;

        let mut y = Array1::<f64>::zeros(n_units * n_periods);
        let unit_fe = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5];
        let time_fe = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

        for i in 0..n_units {
            for t in 0..n_periods {
                let effect = if treated[i] > 0.5 && t >= treat_period {
                    te
                } else {
                    0.0
                };
                y[i * n_periods + t] = unit_fe[i] + time_fe[t] + effect;
            }
        }

        let est = DidEstimator::new(false);
        let res = est
            .estimate_panel(&y.view(), &treated.view(), n_units, n_periods, treat_period)
            .expect("DiD panel should succeed");

        assert!(
            (res.att - te).abs() < 1.0,
            "ATT should be ~{te}, got {}",
            res.att
        );
        // Parallel trends should not be rejected (p > 0.05)
        if let Some(pt_p) = res.parallel_trends_p {
            assert!(
                pt_p > 0.01,
                "Parallel trends should hold (p > 0.01), got p = {pt_p}"
            );
        }
    }

    #[test]
    fn test_event_study_no_pre_trends() {
        let n_units = 8;
        let n_periods = 8;
        let treat_period = 4;
        let treated = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let te = 5.0_f64;

        let mut y = Array1::<f64>::zeros(n_units * n_periods);
        for i in 0..n_units {
            for t in 0..n_periods {
                let effect = if treated[i] > 0.5 && t >= treat_period {
                    te
                } else {
                    0.0
                };
                y[i * n_periods + t] = effect;
            }
        }

        let es = EventStudyEstimator::new(3, 4);
        let res = es
            .estimate(&y.view(), &treated.view(), n_units, n_periods, treat_period)
            .expect("Event study should succeed");

        // Pre-treatment coefficients should be near zero
        let pre_coefs: Vec<&EventStudyCoefficient> = res
            .coefficients
            .iter()
            .filter(|c| c.relative_time < -1)
            .collect();
        for c in &pre_coefs {
            assert!(
                c.estimate.abs() < 1.0,
                "Pre-treatment coef at t={} should be ~0, got {}",
                c.relative_time,
                c.estimate
            );
        }

        // Post-treatment coefficients should be positive
        let post_coefs: Vec<&EventStudyCoefficient> = res
            .coefficients
            .iter()
            .filter(|c| c.relative_time >= 0)
            .collect();
        assert!(!post_coefs.is_empty());
    }

    #[test]
    fn test_staggered_did() {
        let n_units = 9;
        let n_periods = 6;
        // 3 cohorts: treated at t=2, t=3, never-treated
        let cohorts: Vec<i64> = vec![2, 2, 2, 3, 3, 3, i64::MAX, i64::MAX, i64::MAX];
        let te = 2.0_f64;

        let mut y = Array2::<f64>::zeros((n_units, n_periods));
        for i in 0..n_units {
            for t in 0..n_periods {
                let effect = if cohorts[i] < i64::MAX && (t as i64) >= cohorts[i] {
                    te
                } else {
                    0.0
                };
                y[[i, t]] = effect;
            }
        }

        let est = StaggeredDidEstimator::new(42);
        let res = est
            .estimate(&y.view(), &cohorts, n_units, n_periods)
            .expect("Staggered DiD should succeed");

        assert!(
            (res.aggregate_att - te).abs() < 1.0,
            "Aggregate ATT should be ~{te}, got {}",
            res.aggregate_att
        );
        assert!(!res.att_gt.is_empty());
    }
}
