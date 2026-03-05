//! Difference-in-Differences (DiD) estimator
//!
//! DiD is a quasi-experimental causal inference technique that estimates the
//! average treatment effect on the treated (ATT) by comparing outcome changes
//! over time between a treatment group and a control group.
//!
//! # 2×2 DiD
//!
//! Standard DiD uses two groups × two time periods:
//! ```text
//! y_{it} = α + β₁ D_i + β₂ T_t + β_DiD (D_i × T_t) + ε_{it}
//! ```
//! where:
//! - `D_i = 1` if individual `i` is in the treatment group
//! - `T_t = 1` if observation is in the post-treatment period
//! - `β_DiD` — the treatment effect (ATT)
//!
//! # Parallel Trends Assumption
//! DiD requires that, absent treatment, both groups would have followed the same
//! trend (parallel trends).  This is a maintained assumption that cannot be tested
//! directly with only two time periods.  Pre-treatment placebo tests (event study)
//! are recommended.
//!
//! # Event Study / Staggered DiD
//! For settings with multiple pre/post periods, `event_study_did` returns
//! relative-time treatment effect estimates `δ_τ` for `τ ∈ {-k,...,-1, 0, 1,...,m}`.
//!
//! # References
//! - Card, D., & Krueger, A. B. (1994). Minimum wages and employment.
//!   *American Economic Review*, 84(4), 772–793.
//! - Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with
//!   multiple time periods. *Journal of Econometrics*, 225(2), 200–230.

use crate::error::{Result, TimeSeriesError};

// ============================================================
// 2×2 DiD
// ============================================================

/// Results of a 2×2 Difference-in-Differences regression.
#[derive(Debug, Clone)]
pub struct DifferenceInDifferences {
    /// DiD coefficient β_DiD (treatment effect ATT)
    pub beta_did: f64,
    /// OLS standard error of β_DiD
    pub se: f64,
    /// t-statistic for H₀: β_DiD = 0
    pub t_stat: f64,
    /// Approximate p-value (two-sided, large-sample normal)
    pub p_value: f64,
    /// Average outcome in control group, pre-period
    pub y_control_pre: f64,
    /// Average outcome in treatment group, pre-period
    pub y_treat_pre: f64,
    /// Average outcome in control group, post-period
    pub y_control_post: f64,
    /// Average outcome in treatment group, post-period
    pub y_treat_post: f64,
    /// Within-group trend for control: y_control_post - y_control_pre
    pub delta_control: f64,
    /// Within-group trend for treatment: y_treat_post - y_treat_pre
    pub delta_treat: f64,
    /// Number of observations
    pub n_obs: usize,
}

impl DifferenceInDifferences {
    /// Estimate a 2×2 DiD model.
    ///
    /// # Arguments
    /// * `outcomes` — outcome values for all units in both periods
    /// * `treatment` — binary treatment indicator (`true` = in treatment group)
    /// * `post` — binary period indicator (`true` = post-treatment period)
    ///
    /// Each slice must have the same length `N` (one observation per unit per period).
    ///
    /// # Returns
    /// Returns `Err` if the inputs are empty or if any group × period cell is empty.
    pub fn estimate(
        outcomes: &[f64],
        treatment: &[bool],
        post: &[bool],
    ) -> Result<Self> {
        let n = outcomes.len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "DiD: outcomes slice is empty".into(),
            ));
        }
        if treatment.len() != n || post.len() != n {
            return Err(TimeSeriesError::InvalidInput(
                "DiD: outcomes, treatment, and post must have the same length".into(),
            ));
        }

        // Group averages and cell counts
        let (mut y_cp, mut n_cp) = (0.0, 0usize); // control, pre
        let (mut y_tp, mut n_tp) = (0.0, 0usize); // treat, pre
        let (mut y_cq, mut n_cq) = (0.0, 0usize); // control, post
        let (mut y_tq, mut n_tq) = (0.0, 0usize); // treat, post

        for i in 0..n {
            match (treatment[i], post[i]) {
                (false, false) => { y_cp += outcomes[i]; n_cp += 1; }
                (true, false) => { y_tp += outcomes[i]; n_tp += 1; }
                (false, true) => { y_cq += outcomes[i]; n_cq += 1; }
                (true, true) => { y_tq += outcomes[i]; n_tq += 1; }
            }
        }

        if n_cp == 0 || n_tp == 0 || n_cq == 0 || n_tq == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "DiD: one or more group×period cells are empty".into(),
            ));
        }

        let y_cp = y_cp / n_cp as f64;
        let y_tp = y_tp / n_tp as f64;
        let y_cq = y_cq / n_cq as f64;
        let y_tq = y_tq / n_tq as f64;

        let delta_control = y_cq - y_cp;
        let delta_treat = y_tq - y_tp;
        let beta_did = delta_treat - delta_control;

        // OLS standard error via the regression formulation
        // y_i = a + b*D_i + c*T_i + beta_did*(D_i*T_i) + eps_i
        // SE(β_DiD) by OLS: use the Neyman (1990) variance formula for 2×2 DiD

        // Cell variances (sample, unbiased)
        let cell_var = |indices: Vec<usize>, mean: f64| -> f64 {
            let m = indices.len();
            if m < 2 {
                return 0.0;
            }
            indices.iter().map(|&i| (outcomes[i] - mean).powi(2)).sum::<f64>() / (m - 1) as f64
        };

        let idx_cp: Vec<usize> = (0..n).filter(|&i| !treatment[i] && !post[i]).collect();
        let idx_tp: Vec<usize> = (0..n).filter(|&i| treatment[i] && !post[i]).collect();
        let idx_cq: Vec<usize> = (0..n).filter(|&i| !treatment[i] && post[i]).collect();
        let idx_tq: Vec<usize> = (0..n).filter(|&i| treatment[i] && post[i]).collect();

        let var_cp = cell_var(idx_cp, y_cp);
        let var_tp = cell_var(idx_tp, y_tp);
        let var_cq = cell_var(idx_cq, y_cq);
        let var_tq = cell_var(idx_tq, y_tq);

        let var_did = var_cp / n_cp as f64
            + var_tp / n_tp as f64
            + var_cq / n_cq as f64
            + var_tq / n_tq as f64;

        let se = var_did.sqrt().max(1e-15);
        let t_stat = beta_did / se;

        // Two-sided p-value (large-sample normal)
        let p_value = 2.0 * standard_normal_sf(t_stat.abs());

        Ok(Self {
            beta_did,
            se,
            t_stat,
            p_value,
            y_control_pre: y_cp,
            y_treat_pre: y_tp,
            y_control_post: y_cq,
            y_treat_post: y_tq,
            delta_control,
            delta_treat,
            n_obs: n,
        })
    }

    /// Reject the null hypothesis H₀: β_DiD = 0 at the given significance level.
    pub fn reject_null(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

// ============================================================
// Event study DiD
// ============================================================

/// One relative-time coefficient from an event study regression.
#[derive(Debug, Clone)]
pub struct EventStudyCoef {
    /// Relative time to treatment (negative = pre-treatment)
    pub relative_time: i32,
    /// Estimated treatment effect at this time
    pub estimate: f64,
    /// Standard error
    pub se: f64,
    /// 95% CI lower bound
    pub ci_lower: f64,
    /// 95% CI upper bound
    pub ci_upper: f64,
}

/// Estimate an event study for DiD with multiple pre- and post-treatment periods.
///
/// The regression is:
/// ```text
/// y_{it} = α_i + λ_t + Σ_{τ≠-1} δ_τ * (D_i × I[t-g_i=τ]) + ε_{it}
/// ```
///
/// where `g_i` is the treatment cohort for unit `i`, `τ=-1` is the excluded
/// (reference) period, and both unit and time fixed effects `α_i`, `λ_t`
/// are included.
///
/// # Arguments
/// * `outcomes` — flat vector of outcomes, length `N*T`, ordered by unit within time
/// * `treated` — binary treatment indicator for each unit (length N)
/// * `treatment_period` — time period when treatment begins (same for all units)
/// * `n_units` — number of units N
/// * `n_periods` — number of time periods T
/// * `n_pre` — number of pre-treatment periods to include in event study
/// * `n_post` — number of post-treatment periods to include
///
/// Returns a vector of [`EventStudyCoef`] sorted by `relative_time`.
///
/// # Notes
/// This is a simplified "clean" DiD (2-group, staggered-free).  For general
/// staggered settings use the Callaway-Sant'Anna or Sun-Abraham estimator.
pub fn event_study_did(
    outcomes: &[f64],
    treated: &[bool],
    treatment_period: usize,
    n_units: usize,
    n_periods: usize,
    n_pre: usize,
    n_post: usize,
) -> Result<Vec<EventStudyCoef>> {
    let expected_len = n_units * n_periods;
    if outcomes.len() != expected_len {
        return Err(TimeSeriesError::InvalidInput(format!(
            "event_study_did: outcomes length {} != n_units*n_periods = {}",
            outcomes.len(),
            expected_len
        )));
    }
    if treated.len() != n_units {
        return Err(TimeSeriesError::InvalidInput(format!(
            "event_study_did: treated length {} != n_units = {}",
            treated.len(),
            n_units
        )));
    }
    if treatment_period == 0 || treatment_period > n_periods {
        return Err(TimeSeriesError::InvalidInput(format!(
            "event_study_did: treatment_period {} out of range [1,{}]",
            treatment_period, n_periods
        )));
    }

    // Build event-time indicator interactions: D_i * I[t = treatment_period + tau]
    // Relative times: -(n_pre)..=-1, 1..=n_post (exclude tau=-1 as reference)
    let rel_times: Vec<i32> = {
        let mut v: Vec<i32> = (-(n_pre as i32)..=n_post as i32)
            .filter(|&tau| tau != -1)
            .collect();
        v.sort();
        v
    };

    let n_tau = rel_times.len();
    let n_obs = n_units * n_periods;

    // Build design matrix: [unit FE dummies (N-1), time FE dummies (T-1), event indicators (n_tau)]
    // To keep this tractable we use a within-within transformation:
    // 1. Demean by unit (removes unit FE)
    // 2. Demean by time (removes time FE) -- iterative (Mundlak-style)
    // Then regress on event indicators (also double-demeaned).

    let n_fe_params = n_tau; // after demeaning, only event indicators remain

    // Compute unit means
    let mut unit_mean_y = vec![0.0_f64; n_units];
    let mut unit_mean_x = vec![vec![0.0_f64; n_tau]; n_units];

    for i in 0..n_units {
        for t_idx in 0..n_periods {
            let obs = i * n_periods + t_idx;
            unit_mean_y[i] += outcomes[obs];
            let rel_t = t_idx as i32 + 1 - treatment_period as i32;
            for (tau_idx, &tau) in rel_times.iter().enumerate() {
                if treated[i] && rel_t == tau {
                    unit_mean_x[i][tau_idx] += 1.0;
                }
            }
        }
        unit_mean_y[i] /= n_periods as f64;
        for tau_idx in 0..n_tau {
            unit_mean_x[i][tau_idx] /= n_periods as f64;
        }
    }

    // Compute time means (after within-unit demeaning)
    let mut time_mean_y = vec![0.0_f64; n_periods];
    let mut time_mean_x = vec![vec![0.0_f64; n_tau]; n_periods];

    for i in 0..n_units {
        for t_idx in 0..n_periods {
            let obs = i * n_periods + t_idx;
            time_mean_y[t_idx] += outcomes[obs] - unit_mean_y[i];
            let rel_t = t_idx as i32 + 1 - treatment_period as i32;
            for (tau_idx, &tau) in rel_times.iter().enumerate() {
                let x_it = if treated[i] && rel_t == tau { 1.0 } else { 0.0 };
                time_mean_x[t_idx][tau_idx] += x_it - unit_mean_x[i][tau_idx];
            }
        }
        for t_idx in 0..n_periods {
            time_mean_y[t_idx] /= n_units as f64;
            for tau_idx in 0..n_tau {
                time_mean_x[t_idx][tau_idx] /= n_units as f64;
            }
        }
    }

    // Double-demean: ỹ_{it} = y_{it} - ȳ_i - ȳ_t + ȳ
    let grand_mean_y: f64 = outcomes.iter().sum::<f64>() / n_obs as f64;

    let mut y_dd = Vec::with_capacity(n_obs);
    let mut x_dd = Vec::with_capacity(n_obs * n_tau);

    for i in 0..n_units {
        for t_idx in 0..n_periods {
            let obs = i * n_periods + t_idx;
            y_dd.push(outcomes[obs] - unit_mean_y[i] - time_mean_y[t_idx] + grand_mean_y);
            let rel_t = t_idx as i32 + 1 - treatment_period as i32;
            for (tau_idx, &tau) in rel_times.iter().enumerate() {
                let x_it = if treated[i] && rel_t == tau { 1.0 } else { 0.0 };
                let grand_mean_x_tau: f64 = unit_mean_x.iter().map(|v| v[tau_idx]).sum::<f64>() / n_units as f64;
                let x_dd_it = x_it - unit_mean_x[i][tau_idx]
                    - time_mean_x[t_idx][tau_idx]
                    + grand_mean_x_tau;
                x_dd.push(x_dd_it);
            }
        }
    }

    // OLS: β = (X̃'X̃)⁻¹ X̃'ỹ
    if n_fe_params == 0 {
        return Ok(Vec::new());
    }

    let k = n_fe_params;
    let mut xtx = vec![0.0_f64; k * k];
    let mut xty = vec![0.0_f64; k];

    for obs in 0..n_obs {
        for a in 0..k {
            xty[a] += x_dd[obs * k + a] * y_dd[obs];
            for b in 0..k {
                xtx[a * k + b] += x_dd[obs * k + a] * x_dd[obs * k + b];
            }
        }
    }

    let xtx_inv = crate::panel::fixed_effects::invert_sym(k, &xtx)?;
    let beta = crate::panel::fixed_effects::mat_vec_mul(k, &xtx_inv, &xty);

    // Residual variance
    let df = (n_obs as f64 - n_units as f64 - n_periods as f64 - k as f64).max(1.0);
    let ss_res: f64 = (0..n_obs).map(|obs| {
        let xb: f64 = (0..k).map(|j| x_dd[obs * k + j] * beta[j]).sum();
        (y_dd[obs] - xb).powi(2)
    }).sum();

    let sigma2 = ss_res / df;

    let coefs: Vec<EventStudyCoef> = rel_times
        .iter()
        .enumerate()
        .map(|(idx, &tau)| {
            let est = beta[idx];
            let se = (sigma2 * xtx_inv[idx * k + idx]).sqrt().max(1e-15);
            EventStudyCoef {
                relative_time: tau,
                estimate: est,
                se,
                ci_lower: est - 1.96 * se,
                ci_upper: est + 1.96 * se,
            }
        })
        .collect();

    Ok(coefs)
}

// ============================================================
// Standard normal survival function (local)
// ============================================================

fn standard_normal_sf(z: f64) -> f64 {
    0.5 * erfc_approx(z / std::f64::consts::SQRT_2)
}

fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 { return 2.0 - erfc_approx(-x); }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736
            + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (-x * x).exp() * poly
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_did_data() -> (Vec<f64>, Vec<bool>, Vec<bool>) {
        // 20 observations: 10 control (pre=5, post=5), 10 treatment (pre=5, post=5)
        // True ATT = 3.0
        let mut outcomes = Vec::new();
        let mut treatment = Vec::new();
        let mut post = Vec::new();

        // Control group: pre=10, post=12 (trend +2)
        for _ in 0..5 {
            outcomes.push(10.0); treatment.push(false); post.push(false);
        }
        for _ in 0..5 {
            outcomes.push(12.0); treatment.push(false); post.push(true);
        }
        // Treatment group: pre=10, post=15 (trend +5 = +2 common + 3 ATT)
        for _ in 0..5 {
            outcomes.push(10.0); treatment.push(true); post.push(false);
        }
        for _ in 0..5 {
            outcomes.push(15.0); treatment.push(true); post.push(true);
        }

        (outcomes, treatment, post)
    }

    #[test]
    fn test_did_estimate_basic() {
        let (outcomes, treatment, post) = make_did_data();
        let result = DifferenceInDifferences::estimate(&outcomes, &treatment, &post)
            .expect("DiD should estimate");
        // True ATT = 3.0
        assert!(
            (result.beta_did - 3.0).abs() < 1e-8,
            "β_DiD = {:.4}, expected 3.0",
            result.beta_did
        );
    }

    #[test]
    fn test_did_t_stat_significant() {
        let (outcomes, treatment, post) = make_did_data();
        let result = DifferenceInDifferences::estimate(&outcomes, &treatment, &post)
            .expect("DiD");
        assert!(result.reject_null(0.05), "ATT = 3 should be significant at 5%");
    }

    #[test]
    fn test_did_zero_effect() {
        // No treatment effect: both groups follow same trend
        let outcomes: Vec<f64> = vec![
            10.0, 10.0, 12.0, 12.0, // control pre, control post
            10.0, 10.0, 12.0, 12.0, // treat pre, treat post
        ];
        let treatment = vec![false, false, false, false, true, true, true, true];
        let post = vec![false, false, true, true, false, false, true, true];
        let result = DifferenceInDifferences::estimate(&outcomes, &treatment, &post)
            .expect("DiD");
        assert!(
            result.beta_did.abs() < 1e-8,
            "β_DiD = {:.6}, expected 0",
            result.beta_did
        );
    }

    #[test]
    fn test_did_empty_outcomes() {
        let result = DifferenceInDifferences::estimate(&[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_did_empty_cell() {
        // All observations are in control/pre — missing treatment/post cell
        let outcomes = vec![1.0, 2.0, 3.0];
        let treatment = vec![false, false, false];
        let post = vec![false, false, false];
        assert!(DifferenceInDifferences::estimate(&outcomes, &treatment, &post).is_err());
    }
}
