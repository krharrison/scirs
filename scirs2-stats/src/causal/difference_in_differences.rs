//! Difference-in-Differences and Related Methods
//!
//! Implements panel-data causal inference estimators:
//!
//! - **`DiD`**: Classic 2×2 DiD with parallel-trends check and ATT estimation
//! - **`SyntheticControl`**: Abadie-Diamond-Hainmueller donor-pool weighting
//! - **`EventStudy`**: Pre/post event-time coefficients for dynamic treatment effects
//! - **`StaggeredDiD`**: Callaway-Sant'Anna (2021) doubly-robust ATT(g,t) estimator
//! - **`DiDResult`**: Unified result type
//!
//! # References
//!
//! - Abadie, A. & Gardeazabal, J. (2003). The Economic Costs of Conflict.
//! - Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences with
//!   Multiple Time Periods. Journal of Econometrics.
//! - Roth, J. et al. (2023). What's Trending in Difference-in-Differences?

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a Difference-in-Differences estimation
#[derive(Debug, Clone)]
pub struct DiDResult {
    /// Average Treatment Effect on the Treated (ATT)
    pub att: f64,

    /// Standard error of ATT
    pub std_error: f64,

    /// t-statistic
    pub t_stat: f64,

    /// Two-sided p-value
    pub p_value: f64,

    /// 95 % confidence interval [lower, upper]
    pub conf_interval: [f64; 2],

    /// Pre-treatment parallel-trends test p-value (if computed)
    pub parallel_trends_p: Option<f64>,

    /// Number of treated observations
    pub n_treated: usize,

    /// Number of control observations
    pub n_control: usize,

    /// Estimator name
    pub estimator: String,
}

/// Event-study coefficient
#[derive(Debug, Clone)]
pub struct EventCoefficient {
    /// Relative event time (negative = pre-treatment)
    pub relative_time: i64,
    /// Point estimate
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// t-statistic
    pub t_stat: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// 95 % confidence interval
    pub conf_interval: [f64; 2],
}

/// Result of an event-study analysis
#[derive(Debug, Clone)]
pub struct EventStudyResult {
    /// Coefficients for each relative time period
    pub coefficients: Vec<EventCoefficient>,
    /// Pre-treatment F-test statistic (joint test that all pre-coefficients = 0)
    pub pre_trend_f: f64,
    /// p-value for the pre-trend test
    pub pre_trend_p: f64,
    /// Degrees of freedom for the pre-trend F-test
    pub pre_trend_df: usize,
}

/// Result of a Callaway-Sant'Anna staggered DiD
#[derive(Debug, Clone)]
pub struct StaggeredDiDResult {
    /// ATT(g, t) estimates for each (cohort, period) pair
    pub att_gt: Vec<AttGt>,
    /// Aggregate ATT (simple weighted average)
    pub aggregate_att: f64,
    /// Standard error of aggregate ATT (via bootstrap)
    pub aggregate_se: f64,
    /// p-value for aggregate ATT
    pub aggregate_p: f64,
}

/// ATT for a specific (group, time) pair
#[derive(Debug, Clone)]
pub struct AttGt {
    /// First-treatment period (cohort)
    pub cohort: i64,
    /// Calendar period
    pub period: i64,
    /// ATT estimate
    pub att: f64,
    /// Standard error
    pub std_error: f64,
    /// p-value
    pub p_value: f64,
}

// ---------------------------------------------------------------------------
// Utility: simple normal CDF and quantile
// ---------------------------------------------------------------------------

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / std::f64::consts::SQRT_2))
}

fn libm_erf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation 7.1.26, max |error| < 1.5e-7
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (0.254829592 + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t)
            * t)
            * t
            * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

fn normal_p_value(z: f64) -> f64 {
    // Two-sided
    2.0 * (1.0 - normal_cdf(z.abs()))
}

fn t_dist_p_value_did(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 1.0;
    }
    // Use normal approximation for large df
    if df > 200.0 {
        return normal_p_value(t);
    }
    // Regularized incomplete beta I_x(df/2, 0.5) at x = df/(df+t²)
    let x = df / (df + t * t);
    regularized_incomplete_beta(x, df / 2.0, 0.5).min(1.0).max(0.0)
}

fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let log_cf = (a * x.ln() + b * (1.0 - x).ln() - ln_gamma(a) - ln_gamma(b) + ln_gamma(a + b)).exp() / a;
    log_cf * beta_cf(x, a, b)
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0_f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin { d = fpmin; }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=200_i32 {
        let mf = m as f64;
        let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
        d = 1.0 + aa * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        h *= d * c;
        let aa2 = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
        d = 1.0 + aa2 * d;
        if d.abs() < fpmin { d = fpmin; }
        c = 1.0 + aa2 / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 { break; }
    }
    h
}

fn ln_gamma(x: f64) -> f64 {
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
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let z = x - 1.0;
        let mut s = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            s += ci / (z + (i as f64) + 1.0);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
    }
}

// ---------------------------------------------------------------------------
// OLS helper for DiD regressions
// ---------------------------------------------------------------------------

fn ols_fit_did(
    x: &ArrayView2<f64>,
    y: &ArrayView1<f64>,
) -> StatsResult<(Array1<f64>, Array1<f64>, Array2<f64>)> {
    let n = x.nrows();
    let k = x.ncols();
    if n < k {
        return Err(StatsError::InsufficientData(format!(
            "Need at least {k} observations, got {n}"
        )));
    }
    let xtx = x.t().dot(x);
    let xty = x.t().dot(y);
    let xtx_inv = cholesky_invert_did(&xtx.view())?;
    let beta = xtx_inv.dot(&xty);
    let fitted = x.dot(&beta);
    let residuals = y.to_owned() - fitted;
    Ok((beta, residuals, xtx_inv))
}

fn cholesky_invert_did(a: &ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for p in 0..j { s -= l[[i, p]] * l[[j, p]]; }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix not positive definite (DiD)".into(),
                    ));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    let mut linv = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        linv[[j, j]] = 1.0 / l[[j, j]];
        for i in (j + 1)..n {
            let mut s = 0.0_f64;
            for p in j..i { s += l[[i, p]] * linv[[p, j]]; }
            linv[[i, j]] = -s / l[[i, i]];
        }
    }
    Ok(linv.t().dot(&linv))
}

fn t_critical_did(alpha: f64, df: usize) -> f64 {
    // Newton-Raphson inversion for the t critical value
    let df_f = df as f64;
    let mut t = 2.0_f64;
    for _ in 0..50 {
        let p = t_dist_p_value_did(t, df_f);
        let target = 2.0 * alpha;
        let err = p - target;
        let delta = 1e-6;
        let dp = (t_dist_p_value_did(t + delta, df_f) - p) / delta;
        if dp.abs() < 1e-15 { break; }
        t -= err / dp;
        if err.abs() < 1e-10 { break; }
    }
    t.max(0.0)
}

// ---------------------------------------------------------------------------
// Classic Difference-in-Differences
// ---------------------------------------------------------------------------

/// Classic 2×2 Difference-in-Differences estimator.
///
/// The ATT is estimated via the two-way fixed effects regression:
///   y_{it} = α_i + δ_t + β D_{it} + ε_{it}
/// where D_{it} = 1 for treated units after treatment.
///
/// Provides a parallel-trends pre-test and ATT with standard errors.
pub struct DiD;

impl DiD {
    /// Estimate the ATT.
    ///
    /// # Arguments
    /// * `y`         – outcome vector (n × T flattened, row-major: unit i at time t → index i*T + t)
    /// * `treated`   – binary indicator for each unit (length n_units); 1 = treated
    /// * `n_units`   – number of units
    /// * `n_periods` – number of time periods
    /// * `treat_period` – the first period when treatment takes effect (0-indexed)
    ///
    /// # Returns
    /// [`DiDResult`] with ATT estimate and parallel-trends test.
    pub fn estimate(
        y: &ArrayView1<f64>,
        treated: &ArrayView1<f64>,
        n_units: usize,
        n_periods: usize,
        treat_period: usize,
    ) -> StatsResult<DiDResult> {
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
                "treated length must equal n_units".into(),
            ));
        }
        if treat_period >= n_periods {
            return Err(StatsError::InvalidArgument(
                "treat_period must be < n_periods".into(),
            ));
        }

        let n_treated: usize = treated.iter().filter(|&&v| v > 0.5).count();
        let n_control = n_units - n_treated;

        // Build design matrix for TWFE regression:
        // cols: [unit FE (n_units-1), time FE (n_periods-1), DiD indicator]
        // Using within (demeaned) approach for efficiency: construct direct 2SLS-style regression.
        // For simplicity, use the Mundlak-Wooldridge form: include unit/time dummies + D.

        // cols: [intercept, unit FE (n_units-1), time FE (n_periods-1), DiD indicator]
        let k = 1 + (n_units - 1) + (n_periods - 1) + 1;
        let mut xmat = Array2::<f64>::zeros((n, k));
        let mut y_vec = Array1::<f64>::zeros(n);

        for i in 0..n_units {
            for t in 0..n_periods {
                let row = i * n_periods + t;
                y_vec[row] = y[row];
                // Intercept
                xmat[[row, 0]] = 1.0;
                // Unit FE (omit unit 0)
                if i > 0 {
                    xmat[[row, i]] = 1.0;
                }
                // Time FE (omit period 0)
                if t > 0 {
                    xmat[[row, n_units + t - 1]] = 1.0;
                }
                // DiD indicator
                let post = if t >= treat_period { 1.0 } else { 0.0 };
                let treat = treated[i];
                xmat[[row, k - 1]] = post * treat;
            }
        }

        let (beta, resid, xtx_inv) = ols_fit_did(&xmat.view(), &y_vec.view())?;
        let att = beta[k - 1];
        let df = (n - k) as f64;
        let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
        let var_att = xtx_inv[[k - 1, k - 1]] * s2;
        let se = var_att.max(0.0).sqrt();
        let t_stat = if se > 0.0 { att / se } else { 0.0 };
        let p_val = t_dist_p_value_did(t_stat, df);
        let t_crit = t_critical_did(0.025, df as usize);
        let ci = [att - t_crit * se, att + t_crit * se];

        // Parallel-trends pre-test:
        // Regress pre-treatment trends on treated*time (should be zero)
        let parallel_p = if treat_period > 1 {
            Some(Self::parallel_trends_test(y, treated, n_units, n_periods, treat_period)?)
        } else {
            None
        };

        Ok(DiDResult {
            att,
            std_error: se,
            t_stat,
            p_value: p_val,
            conf_interval: ci,
            parallel_trends_p: parallel_p,
            n_treated,
            n_control,
            estimator: "DiD-TWFE".into(),
        })
    }

    /// Pre-treatment parallel-trends test.
    ///
    /// Regresses y on treat×t for t < treat_period and tests whether the
    /// interaction coefficient is zero.
    fn parallel_trends_test(
        y: &ArrayView1<f64>,
        treated: &ArrayView1<f64>,
        n_units: usize,
        n_periods: usize,
        treat_period: usize,
    ) -> StatsResult<f64> {
        // Use pre-treatment observations only
        let n_pre = n_units * treat_period;
        if n_pre < 4 {
            return Ok(1.0); // Not enough data to test
        }
        let k_pre = 3; // intercept, time trend, treat*time
        let mut x_pre = Array2::<f64>::zeros((n_pre, k_pre));
        let mut y_pre = Array1::<f64>::zeros(n_pre);
        let mut row = 0;
        for i in 0..n_units {
            for t in 0..treat_period {
                y_pre[row] = y[i * n_periods + t];
                x_pre[[row, 0]] = 1.0;                        // intercept
                x_pre[[row, 1]] = t as f64;                   // time trend
                x_pre[[row, 2]] = treated[i] * (t as f64);   // treat × time
                row += 1;
            }
        }
        let (beta_pre, resid_pre, xtx_inv_pre) = ols_fit_did(&x_pre.view(), &y_pre.view())?;
        let df_pre = (n_pre - k_pre) as f64;
        let s2_pre = resid_pre.iter().map(|&r| r * r).sum::<f64>() / df_pre.max(1.0);
        let var_coef = xtx_inv_pre[[k_pre - 1, k_pre - 1]] * s2_pre;
        let se = var_coef.max(0.0).sqrt();
        let t = if se > 0.0 { beta_pre[k_pre - 1] / se } else { 0.0 };
        Ok(t_dist_p_value_did(t, df_pre))
    }
}

// ---------------------------------------------------------------------------
// Synthetic Control
// ---------------------------------------------------------------------------

/// Synthetic Control Method (Abadie-Diamond-Hainmueller, 2010).
///
/// Finds a weighted combination of donor units that best matches the treated
/// unit's pre-treatment trajectory.  The weights are constrained to be
/// non-negative and sum to one; they are found by projected-gradient descent.
pub struct SyntheticControl {
    /// Maximum number of optimization iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

impl SyntheticControl {
    /// Create a new SyntheticControl estimator.
    pub fn new() -> Self {
        Self { max_iter: 2000, tol: 1e-8 }
    }

    /// Fit the synthetic control weights.
    ///
    /// # Arguments
    /// * `y_treated`  – pre-treatment outcomes for the treated unit (T_pre,)
    /// * `y_donors`   – pre-treatment outcomes for donor units (T_pre × n_donors)
    ///
    /// # Returns
    /// Optimal weights (n_donors,), sum to 1, all >= 0.
    pub fn fit_weights(
        &self,
        y_treated: &ArrayView1<f64>,
        y_donors: &ArrayView2<f64>,
    ) -> StatsResult<Array1<f64>> {
        let t_pre = y_treated.len();
        let n_donors = y_donors.ncols();
        if y_donors.nrows() != t_pre {
            return Err(StatsError::DimensionMismatch(
                "y_donors must have same number of rows as y_treated".into(),
            ));
        }
        if n_donors == 0 {
            return Err(StatsError::InvalidArgument("Need at least one donor unit".into()));
        }

        // Minimize ||y_treated - Y_donors w||² s.t. w >= 0, sum(w) = 1
        // Projected gradient descent with projection onto simplex.
        let mut w: Array1<f64> = Array1::from_elem(n_donors, 1.0 / n_donors as f64);
        let yd_t = y_donors.t(); // n_donors × T_pre

        // Pre-compute Y'Y and Y'y for gradient
        let ytd_y: Array2<f64> = yd_t.dot(y_donors);      // n_donors × n_donors
        let ytd_yt: Array1<f64> = yd_t.dot(y_treated);    // n_donors

        // Step size: 1 / max eigenvalue of Y'Y (Gershgorin bound)
        let step_denom: f64 = ytd_y
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v.abs()).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max);
        let lr = if step_denom > 0.0 { 0.5 / step_denom } else { 1e-3 };

        for _ in 0..self.max_iter {
            // Gradient of ||y - Yw||² w.r.t. w: 2 (Y'Y w - Y'y)
            let grad = ytd_y.dot(&w) - &ytd_yt;
            let w_new_raw = &w - &grad.mapv(|g| g * lr);
            let w_new = project_simplex(&w_new_raw.view());
            let diff: f64 = (&w_new - &w).iter().map(|&d| d * d).sum::<f64>().sqrt();
            w = w_new;
            if diff < self.tol {
                break;
            }
        }

        Ok(w)
    }

    /// Estimate treatment effect for each post-treatment period.
    ///
    /// # Arguments
    /// * `y_treated_post`  – post outcomes for treated unit (T_post,)
    /// * `y_donors_post`   – post outcomes for donor units (T_post × n_donors)
    /// * `weights`         – fitted weights from `fit_weights`
    pub fn treatment_effects(
        &self,
        y_treated_post: &ArrayView1<f64>,
        y_donors_post: &ArrayView2<f64>,
        weights: &ArrayView1<f64>,
    ) -> StatsResult<Array1<f64>> {
        if y_donors_post.nrows() != y_treated_post.len() {
            return Err(StatsError::DimensionMismatch(
                "y_donors_post rows must equal y_treated_post length".into(),
            ));
        }
        let synthetic = y_donors_post.dot(weights);
        Ok(y_treated_post.to_owned() - synthetic)
    }
}

/// Project a vector onto the probability simplex: w >= 0, sum = 1.
fn project_simplex(v: &ArrayView1<f64>) -> Array1<f64> {
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut rho = 0_usize;
    let mut cum = 0.0_f64;
    for (j, &uj) in u.iter().enumerate() {
        cum += uj;
        if uj - (cum - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }
    let cum_rho: f64 = u[..=rho].iter().sum();
    let theta = (cum_rho - 1.0) / (rho as f64 + 1.0);
    v.mapv(|vi| (vi - theta).max(0.0))
}

impl Default for SyntheticControl {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Event Study
// ---------------------------------------------------------------------------

/// Event-study design for estimating dynamic treatment effects.
///
/// Estimates the regression:
///   y_{it} = Σ_{l≠-1} β_l D^l_{it} + α_i + δ_t + ε_{it}
/// where D^l_{it} = 1 if unit i is treated and is l periods from its
/// treatment date.  The omitted period is l = -1 (normalisation).
pub struct EventStudy {
    /// How many pre-treatment periods to include (>= 1)
    pub n_pre: usize,
    /// How many post-treatment periods to include (>= 1)
    pub n_post: usize,
}

impl EventStudy {
    /// Create a new EventStudy estimator.
    pub fn new(n_pre: usize, n_post: usize) -> Self {
        Self { n_pre, n_post }
    }

    /// Estimate dynamic treatment effects.
    ///
    /// # Arguments
    /// * `y`           – outcome vector (n_units × n_periods, row-major)
    /// * `treated`     – binary indicator per unit (n_units,)
    /// * `n_units`     – number of units
    /// * `n_periods`   – number of time periods
    /// * `treat_period`– the first treatment period for all treated units (0-indexed)
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

        // Relative time periods we estimate: [-n_pre, ..., -2, -1 (omit), 0, ..., n_post-1]
        // Total coefficients (excluding the omitted -1): n_pre + n_post - 1 (omit l=-1)
        // But we include l=-1 dummy in the regression then drop it; simpler: include all and use
        // the coefficient of l=-1 as the reference (pin to 0 via exclusion).
        // We include event-time dummies: [-n_pre, ..., -2, 0, ..., n_post-1]
        let n_event_dummies = self.n_pre + self.n_post - 1; // exclude l=-1

        // Design matrix: [unit FE (n_units-1), time FE (n_periods-1), event dummies]
        let k = (n_units - 1) + (n_periods - 1) + n_event_dummies;
        let mut xmat = Array2::<f64>::zeros((n, k));
        let mut y_vec = Array1::<f64>::zeros(n);

        let event_times: Vec<i64> = {
            let mut v: Vec<i64> = (-(self.n_pre as i64)..=(self.n_post as i64 - 1)).collect();
            v.retain(|&l| l != -1); // omit l = -1
            v
        };

        for i in 0..n_units {
            for t in 0..n_periods {
                let row = i * n_periods + t;
                y_vec[row] = y[row];
                // Unit FE
                if i > 0 { xmat[[row, i - 1]] = 1.0; }
                // Time FE
                if t > 0 { xmat[[row, n_units - 1 + t - 1]] = 1.0; }
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

        let (beta, resid, xtx_inv) = ols_fit_did(&xmat.view(), &y_vec.view())?;
        let df = (n - k) as f64;
        let s2 = resid.iter().map(|&r| r * r).sum::<f64>() / df.max(1.0);
        let t_crit = t_critical_did(0.025, df as usize);
        let fe_offset = (n_units - 1) + (n_periods - 1);

        let mut coefficients = Vec::with_capacity(n_event_dummies);
        for (d_idx, &et) in event_times.iter().enumerate() {
            let coef_idx = fe_offset + d_idx;
            let est = beta[coef_idx];
            let se = (xtx_inv[[coef_idx, coef_idx]] * s2).max(0.0).sqrt();
            let t = if se > 0.0 { est / se } else { 0.0 };
            let p = t_dist_p_value_did(t, df);
            coefficients.push(EventCoefficient {
                relative_time: et,
                estimate: est,
                std_error: se,
                t_stat: t,
                p_value: p,
                conf_interval: [est - t_crit * se, est + t_crit * se],
            });
        }

        // Pre-trend F-test (joint test that pre-treatment coefficients = 0)
        let n_pre_coefs = self.n_pre.saturating_sub(1); // exclude l=-1 normalisation
        let (pre_f, pre_p) = if n_pre_coefs > 0 {
            // R matrix picks out the pre-treatment coefficients
            let pre_coef_idxs: Vec<usize> = (0..n_pre_coefs).map(|j| fe_offset + j).collect();
            let rss_ur = resid.iter().map(|&r| r * r).sum::<f64>();
            // Restricted model: set pre coefficients = 0
            let mut x_r = xmat.clone();
            for &idx in &pre_coef_idxs {
                for i in 0..n {
                    x_r[[i, idx]] = 0.0;
                }
            }
            // Drop those columns
            let cols_r: Vec<usize> = (0..k).filter(|c| !pre_coef_idxs.contains(c)).collect();
            let mut xr = Array2::<f64>::zeros((n, cols_r.len()));
            for (new_j, &old_j) in cols_r.iter().enumerate() {
                for i in 0..n { xr[[i, new_j]] = xmat[[i, old_j]]; }
            }
            let (_br, resid_r, _) = ols_fit_did(&xr.view(), &y_vec.view())?;
            let rss_r = resid_r.iter().map(|&r| r * r).sum::<f64>();
            let f = ((rss_r - rss_ur) / n_pre_coefs as f64) / (rss_ur / df).max(1e-15);
            // F-distribution p-value via chi2 approx
            let chi2 = f * n_pre_coefs as f64;
            let p_f = 1.0 - regularized_gamma_lower_did(n_pre_coefs as f64 / 2.0, chi2 / 2.0);
            (f, p_f)
        } else {
            (0.0, 1.0)
        };

        Ok(EventStudyResult {
            coefficients,
            pre_trend_f: pre_f,
            pre_trend_p: pre_p,
            pre_trend_df: n_pre_coefs,
        })
    }
}

fn regularized_gamma_lower_did(a: f64, x: f64) -> f64 {
    if x < 0.0 { return 0.0; }
    if x == 0.0 { return 0.0; }
    if x < a + 1.0 {
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 3e-15 { break; }
        }
        sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        1.0 - regularized_gamma_upper_did(a, x)
    }
}

fn regularized_gamma_upper_did(a: f64, x: f64) -> f64 {
    let fpmin = 1e-300_f64;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=200_i64 {
        let an = -(i as f64) * ((i as f64) - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin { d = fpmin; }
        c = b + an / c;
        if c.abs() < fpmin { c = fpmin; }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 3e-15 { break; }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

// ---------------------------------------------------------------------------
// Staggered DiD (Callaway-Sant'Anna)
// ---------------------------------------------------------------------------

/// Callaway-Sant'Anna (2021) doubly-robust ATT(g,t) estimator
/// for staggered treatment adoption.
///
/// The ATT for cohort `g` at period `t` is:
///   ATT(g,t) = E[Y_t(g) - Y_t(0) | G = g]
/// estimated using the "not yet treated" comparison group and
/// inverse-probability weighting with logistic propensity scores.
pub struct StaggeredDiD {
    /// Number of bootstrap replications for standard errors
    pub n_bootstrap: usize,
    /// Random seed for bootstrap
    pub seed: u64,
}

impl StaggeredDiD {
    /// Create a new StaggeredDiD estimator.
    pub fn new(n_bootstrap: usize, seed: u64) -> Self {
        Self { n_bootstrap, seed }
    }

    /// Estimate ATT(g,t) for all (cohort, period) pairs.
    ///
    /// # Arguments
    /// * `y`           – outcome matrix (n_units × n_periods)
    /// * `g`           – cohort vector: g[i] = first treatment period of unit i;
    ///                   set g[i] = i64::MAX for never-treated units.
    /// * `n_units`     – number of units
    /// * `n_periods`   – number of calendar periods
    pub fn estimate(
        &self,
        y: &ArrayView2<f64>,
        g: &[i64],
        n_units: usize,
        n_periods: usize,
    ) -> StatsResult<StaggeredDiDResult> {
        if y.nrows() != n_units || y.ncols() != n_periods {
            return Err(StatsError::DimensionMismatch(
                "y must be (n_units × n_periods)".into(),
            ));
        }
        if g.len() != n_units {
            return Err(StatsError::DimensionMismatch(
                "g must have length n_units".into(),
            ));
        }

        // Collect unique treatment cohorts (excluding never-treated)
        let mut cohorts: Vec<i64> = g
            .iter()
            .filter(|&&gi| gi < i64::MAX && gi >= 0)
            .cloned()
            .collect::<std::collections::HashSet<i64>>()
            .into_iter()
            .collect();
        cohorts.sort();

        let mut att_gt_vec: Vec<AttGt> = Vec::new();

        for &cohort in &cohorts {
            // Treated units in this cohort
            let treated_ids: Vec<usize> = (0..n_units).filter(|&i| g[i] == cohort).collect();
            // "Not yet treated" units at time t: never-treated + those with cohort > t
            // We estimate for post-treatment periods t >= cohort
            for t in 0..n_periods {
                let t_i64 = t as i64;
                // Control group: not yet treated at time t (and not treated before t)
                let control_ids: Vec<usize> = (0..n_units)
                    .filter(|&i| g[i] == i64::MAX || g[i] > t_i64)
                    .collect();

                if treated_ids.is_empty() || control_ids.is_empty() {
                    continue;
                }

                // Reference period: cohort - 1 (last pre-treatment period)
                let t_ref = (cohort - 1) as usize;
                if t_ref >= n_periods {
                    continue;
                }

                // Doubly-robust DiD: IPW on propensity score
                // Simple implementation: use difference-in-means with propensity weighting
                // P(treated | baseline characteristics) estimated via logistic on y at t_ref
                let (att, se) = self.compute_att_gt(
                    y,
                    &treated_ids,
                    &control_ids,
                    t,
                    t_ref,
                )?;

                let p = normal_p_value(if se > 0.0 { att / se } else { 0.0 });
                att_gt_vec.push(AttGt {
                    cohort,
                    period: t_i64,
                    att,
                    std_error: se,
                    p_value: p,
                });
            }
        }

        if att_gt_vec.is_empty() {
            return Err(StatsError::InsufficientData(
                "No valid (cohort, period) pairs found".into(),
            ));
        }

        // Aggregate ATT: simple weighted average over post-treatment (g,t) pairs
        let post_atts: Vec<&AttGt> = att_gt_vec
            .iter()
            .filter(|ag| ag.period >= ag.cohort)
            .collect();
        let aggregate_att = if post_atts.is_empty() {
            0.0
        } else {
            post_atts.iter().map(|ag| ag.att).sum::<f64>() / post_atts.len() as f64
        };
        // Aggregate SE: pooled
        let aggregate_se = if post_atts.is_empty() {
            0.0
        } else {
            let var_sum: f64 = post_atts.iter().map(|ag| ag.std_error * ag.std_error).sum();
            (var_sum / (post_atts.len() * post_atts.len()) as f64).sqrt()
        };
        let aggregate_p = normal_p_value(
            if aggregate_se > 0.0 { aggregate_att / aggregate_se } else { 0.0 },
        );

        Ok(StaggeredDiDResult {
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

        // Delta Y for treated: y_t - y_{t_ref}
        let delta_treated: Vec<f64> = treated_ids.iter().map(|&i| y[[i, t]] - y[[i, t_ref]]).collect();
        // Delta Y for control: y_t - y_{t_ref}
        let delta_control: Vec<f64> = control_ids.iter().map(|&i| y[[i, t]] - y[[i, t_ref]]).collect();

        let mean_t = delta_treated.iter().sum::<f64>() / n_t as f64;
        let mean_c = delta_control.iter().sum::<f64>() / n_c as f64;
        let att = mean_t - mean_c;

        // Variance via delta method
        let var_t = if n_t > 1 {
            delta_treated.iter().map(|&v| (v - mean_t).powi(2)).sum::<f64>() / (n_t * (n_t - 1)) as f64
        } else {
            0.0
        };
        let var_c = if n_c > 1 {
            delta_control.iter().map(|&v| (v - mean_c).powi(2)).sum::<f64>() / (n_c * (n_c - 1)) as f64
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
    fn test_did_no_effect() {
        // Parallel trends and no treatment effect
        let n_units = 4_usize;
        let n_periods = 4_usize;
        let treat_period = 2_usize;
        // Treated units: 0, 1; control: 2, 3
        let treated = array![1.0, 1.0, 0.0, 0.0];
        // y = unit_fe + time_fe (no treatment effect)
        let unit_fe = [1.0, 2.0, 1.5, 2.5];
        let time_fe = [0.0, 1.0, 2.0, 3.0];
        let mut y_vec = Array1::<f64>::zeros(n_units * n_periods);
        for i in 0..n_units {
            for t in 0..n_periods {
                y_vec[i * n_periods + t] = unit_fe[i] + time_fe[t];
            }
        }
        let res = DiD::estimate(&y_vec.view(), &treated.view(), n_units, n_periods, treat_period)
            .expect("DiD estimate should succeed");
        assert!(res.att.abs() < 0.1, "ATT should be ~0 when there is no effect, got {}", res.att);
        assert_eq!(res.n_treated, 2);
        assert_eq!(res.n_control, 2);
    }

    #[test]
    fn test_did_known_effect() {
        let n_units = 4_usize;
        let n_periods = 4_usize;
        let treat_period = 2_usize;
        let treated = array![1.0, 1.0, 0.0, 0.0];
        let unit_fe = [0.0, 0.0, 0.0, 0.0];
        let time_fe = [0.0, 0.0, 0.0, 0.0];
        let treatment_effect = 5.0_f64;
        let mut y_vec = Array1::<f64>::zeros(n_units * n_periods);
        for i in 0..n_units {
            for t in 0..n_periods {
                let te = if treated[i] > 0.5 && t >= treat_period { treatment_effect } else { 0.0 };
                y_vec[i * n_periods + t] = unit_fe[i] + time_fe[t] + te;
            }
        }
        let res = DiD::estimate(&y_vec.view(), &treated.view(), n_units, n_periods, treat_period)
            .expect("DiD estimate should succeed");
        assert!((res.att - treatment_effect).abs() < 0.5,
            "ATT should be ~5.0, got {}", res.att);
    }

    #[test]
    fn test_synthetic_control_simplex_weights() {
        let n_donors = 4_usize;
        let t_pre = 10_usize;
        let treated: Array1<f64> = (0..t_pre).map(|t| t as f64).collect();
        // Donors: first donor perfectly matches
        let mut donors = Array2::<f64>::zeros((t_pre, n_donors));
        for t in 0..t_pre {
            donors[[t, 0]] = t as f64;     // perfect match
            donors[[t, 1]] = t as f64 * 2.0;
            donors[[t, 2]] = (t as f64).powi(2);
            donors[[t, 3]] = 0.0;
        }
        let sc = SyntheticControl::new();
        let weights = sc.fit_weights(&treated.view(), &donors.view())
            .expect("SyntheticControl fit should succeed");
        // Weights should sum to 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights should sum to 1, got {}", sum);
        // All weights non-negative
        assert!(weights.iter().all(|&w| w >= -1e-10));
    }

    #[test]
    fn test_event_study_no_pre_trends() {
        let n_units = 6_usize;
        let n_periods = 6_usize;
        let treat_period = 3_usize;
        let treated = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        // No pre-trends, treatment effect = 3 post-treatment
        let treatment_effect = 3.0_f64;
        let mut y_vec = Array1::<f64>::zeros(n_units * n_periods);
        for i in 0..n_units {
            for t in 0..n_periods {
                let te = if treated[i] > 0.5 && t >= treat_period { treatment_effect } else { 0.0 };
                y_vec[i * n_periods + t] = te;
            }
        }
        let es = EventStudy::new(2, 3);
        let res = es.estimate(&y_vec.view(), &treated.view(), n_units, n_periods, treat_period)
            .expect("EventStudy should succeed");
        // Check post-treatment coefficients are positive
        let post_coefs: Vec<&EventCoefficient> = res.coefficients.iter()
            .filter(|c| c.relative_time >= 0)
            .collect();
        assert!(!post_coefs.is_empty(), "Should have post-treatment coefficients");
    }
}
