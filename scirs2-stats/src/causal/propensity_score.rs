//! Propensity Score Methods for Causal Inference
//!
//! Provides a suite of propensity-score-based estimators:
//!
//! - **`PropensityScoreModel`**: logistic-regression-based PS estimation
//! - **`IPW`**: inverse probability weighting (Horvitz-Thompson and
//!   normalised Hajek variants)
//! - **`PSMatching`**: nearest-neighbour, caliper, and kernel matching
//! - **`OverlapCheck`**: common-support trimming and overlap diagnostics
//! - Estimates ATE, ATT, and ATC
//!
//! # References
//!
//! - Rosenbaum, P.R. & Rubin, D.B. (1983). The Central Role of the Propensity Score
//!   in Observational Studies for Causal Effects. Biometrika.
//! - Hirano, K., Imbens, G.W. & Ridder, G. (2003). Efficient Estimation of Average
//!   Treatment Effects Using the Estimated Propensity Score.
//! - Heckman, J.J., Ichimura, H. & Todd, P. (1998). Matching As An Econometric
//!   Evaluation Estimator.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a propensity score-based causal effect estimation
#[derive(Debug, Clone)]
pub struct PSResult {
    /// Average Treatment Effect (population-wide)
    pub ate: f64,
    /// Standard error of ATE
    pub ate_se: f64,
    /// Average Treatment Effect on the Treated
    pub att: f64,
    /// Standard error of ATT
    pub att_se: f64,
    /// Average Treatment Effect on the Controls
    pub atc: f64,
    /// Standard error of ATC
    pub atc_se: f64,
    /// p-value for ATE test (H₀: ATE = 0)
    pub ate_p: f64,
    /// p-value for ATT test
    pub att_p: f64,
    /// p-value for ATC test
    pub atc_p: f64,
    /// Estimated propensity scores
    pub propensity_scores: Array1<f64>,
    /// Estimator name
    pub estimator: String,
}

/// Overlap diagnostics result
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// Estimated propensity scores for all units
    pub ps: Array1<f64>,
    /// Indices of units in the common support
    pub common_support_idx: Vec<usize>,
    /// Lower bound of the common support trimming rule
    pub ps_lower: f64,
    /// Upper bound of the common support trimming rule
    pub ps_upper: f64,
    /// Fraction of treated units inside common support
    pub frac_treated_in_support: f64,
    /// Fraction of control units inside common support
    pub frac_control_in_support: f64,
    /// Overlap coefficient (integral of min(f_t, f_c))
    pub overlap_coefficient: f64,
}

/// Matching result
#[derive(Debug, Clone)]
pub struct MatchingResult {
    /// ATT estimate
    pub att: f64,
    /// Standard error of ATT
    pub att_se: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// 95 % confidence interval
    pub conf_interval: [f64; 2],
    /// Number of treated units matched
    pub n_matched_treated: usize,
    /// Matching method used
    pub method: String,
}

// ---------------------------------------------------------------------------
// Utility: standard normal
// ---------------------------------------------------------------------------

fn normal_p_value(z: f64) -> f64 {
    2.0 * (1.0 - normal_cdf(z.abs()))
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (0.254829592 + (-0.284496736 + (1.421413741 + (-1.453152027 + 1.061405429 * t) * t) * t)
            * t)
            * t
            * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

// ---------------------------------------------------------------------------
// Propensity Score Model (logistic regression)
// ---------------------------------------------------------------------------

/// Logistic regression estimator for the propensity score.
///
/// Estimates P(W=1 | X) via logistic regression using Newton-Raphson (IRLS).
pub struct PropensityScoreModel {
    /// Maximum number of Newton-Raphson iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// L2 regularisation parameter (ridge penalty)
    pub lambda: f64,
}

impl PropensityScoreModel {
    /// Create a new propensity score model.
    pub fn new() -> Self {
        Self { max_iter: 200, tol: 1e-8, lambda: 1e-4 }
    }

    /// Fit the propensity score model via IRLS (Newton-Raphson).
    ///
    /// # Arguments
    /// * `x`   – covariate matrix (n × k); a constant column is prepended automatically
    /// * `w`   – binary treatment indicator (n,)
    ///
    /// # Returns
    /// Fitted coefficient vector (k+1,) including intercept.
    pub fn fit(
        &self,
        x: &ArrayView2<f64>,
        w: &ArrayView1<f64>,
    ) -> StatsResult<Array1<f64>> {
        let n = x.nrows();
        let k = x.ncols();
        if w.len() != n {
            return Err(StatsError::DimensionMismatch(
                "x rows must equal w length".into(),
            ));
        }
        // Prepend intercept column
        let mut xmat = Array2::<f64>::zeros((n, k + 1));
        for i in 0..n {
            xmat[[i, 0]] = 1.0;
            for j in 0..k { xmat[[i, j + 1]] = x[[i, j]]; }
        }
        let k1 = k + 1;
        let mut beta = Array1::<f64>::zeros(k1);

        for _iter in 0..self.max_iter {
            // mu = sigmoid(X beta)
            let eta: Array1<f64> = xmat.dot(&beta);
            let mu: Array1<f64> = eta.mapv(sigmoid);
            // Working weights: v_i = mu_i (1 - mu_i)
            let v: Array1<f64> = mu.mapv(|m| (m * (1.0 - m)).max(1e-8));
            // Gradient: X' (y - mu) - lambda * beta  (regularise all but intercept)
            let grad_data = xmat.t().dot(&(w.to_owned() - &mu));
            let mut grad = grad_data;
            for j in 1..k1 { grad[j] -= self.lambda * beta[j]; }
            // Hessian: X' diag(v) X + lambda * I  (except [0,0])
            // Build W^{1/2} X and solve H delta = grad
            let sqrt_v: Array1<f64> = v.mapv(|vi| vi.sqrt());
            let mut wxmat = Array2::<f64>::zeros((n, k1));
            for i in 0..n {
                for j in 0..k1 { wxmat[[i, j]] = sqrt_v[i] * xmat[[i, j]]; }
            }
            let mut hess = wxmat.t().dot(&wxmat);
            for j in 1..k1 { hess[[j, j]] += self.lambda; }
            let h_inv = cholesky_invert_ps(&hess.view())?;
            let delta = h_inv.dot(&grad);
            let step_norm: f64 = delta.iter().map(|&d| d * d).sum::<f64>().sqrt();
            beta = &beta + &delta;
            if step_norm < self.tol { break; }
        }
        Ok(beta)
    }

    /// Predict propensity scores for new covariates.
    ///
    /// # Arguments
    /// * `x`    – covariate matrix (n × k)
    /// * `beta` – fitted coefficients from `fit` (k+1,)
    pub fn predict(
        &self,
        x: &ArrayView2<f64>,
        beta: &ArrayView1<f64>,
    ) -> StatsResult<Array1<f64>> {
        let n = x.nrows();
        let k = x.ncols();
        if beta.len() != k + 1 {
            return Err(StatsError::DimensionMismatch(format!(
                "beta length {} != k+1 = {}",
                beta.len(),
                k + 1
            )));
        }
        let mut eta = Array1::<f64>::zeros(n);
        for i in 0..n {
            eta[i] = beta[0];
            for j in 0..k { eta[i] += beta[j + 1] * x[[i, j]]; }
        }
        Ok(eta.mapv(sigmoid))
    }
}

impl Default for PropensityScoreModel {
    fn default() -> Self {
        Self::new()
    }
}

fn sigmoid(x: f64) -> f64 {
    if x > 500.0 { return 1.0; }
    if x < -500.0 { return 0.0; }
    1.0 / (1.0 + (-x).exp())
}

fn cholesky_invert_ps(a: &scirs2_core::ndarray::ArrayView2<f64>) -> StatsResult<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for p in 0..j { s -= l[[i, p]] * l[[j, p]]; }
            if i == j {
                if s <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Hessian not positive definite (PS logistic)".into(),
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

// ---------------------------------------------------------------------------
// Inverse Probability Weighting (IPW)
// ---------------------------------------------------------------------------

/// Inverse Probability Weighting estimator.
///
/// ATE (Horvitz-Thompson):
///   τ̂_ATE = (1/n) Σ [ W_i Y_i / e_i - (1-W_i) Y_i / (1-e_i) ]
///
/// ATT (normalised):
///   τ̂_ATT = Σ_{W=1} Y_i / n_t  -  Σ_{W=0} Y_i e_i/(1-e_i) / Σ_{W=0} e_i/(1-e_i)
pub struct IPW;

impl IPW {
    /// Estimate ATE, ATT, and ATC via inverse probability weighting.
    ///
    /// # Arguments
    /// * `y`  – outcome vector
    /// * `w`  – binary treatment indicator
    /// * `ps` – estimated propensity scores
    /// * `trim_eps` – trim propensity scores to [trim_eps, 1 - trim_eps]
    pub fn estimate(
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        ps: &ArrayView1<f64>,
        trim_eps: f64,
    ) -> StatsResult<PSResult> {
        let n = y.len();
        if w.len() != n || ps.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y, w, ps must all have the same length".into(),
            ));
        }
        let eps = trim_eps.max(0.0).min(0.49);

        // Trim propensity scores
        let ps_trim: Array1<f64> = ps.mapv(|p| p.clamp(eps, 1.0 - eps));

        // ATE: Horvitz-Thompson estimator
        let ate_terms: Array1<f64> = (0..n).map(|i| {
            let wi = w[i];
            let yi = y[i];
            let pi = ps_trim[i];
            wi * yi / pi - (1.0 - wi) * yi / (1.0 - pi)
        }).collect();
        let ate = ate_terms.iter().sum::<f64>() / n as f64;

        // ATT: normalised IPW
        let n_treated: usize = w.iter().filter(|&&wi| wi > 0.5).count();
        let att_num: f64 = (0..n)
            .filter(|&i| w[i] > 0.5)
            .map(|i| y[i])
            .sum::<f64>();
        let att_denom_ctrl_num: f64 = (0..n)
            .filter(|&i| w[i] <= 0.5)
            .map(|i| y[i] * ps_trim[i] / (1.0 - ps_trim[i]))
            .sum::<f64>();
        let att_denom_ctrl_den: f64 = (0..n)
            .filter(|&i| w[i] <= 0.5)
            .map(|i| ps_trim[i] / (1.0 - ps_trim[i]))
            .sum::<f64>();
        let att = if n_treated > 0 && att_denom_ctrl_den > 1e-10 {
            att_num / n_treated as f64 - att_denom_ctrl_num / att_denom_ctrl_den
        } else { 0.0 };

        // ATC: normalised IPW
        let n_control = n - n_treated;
        let atc_ctrl_mean = if n_control > 0 {
            (0..n).filter(|&i| w[i] <= 0.5).map(|i| y[i]).sum::<f64>() / n_control as f64
        } else { 0.0 };
        let atc_trt_num: f64 = (0..n)
            .filter(|&i| w[i] > 0.5)
            .map(|i| y[i] * (1.0 - ps_trim[i]) / ps_trim[i])
            .sum::<f64>();
        let atc_trt_den: f64 = (0..n)
            .filter(|&i| w[i] > 0.5)
            .map(|i| (1.0 - ps_trim[i]) / ps_trim[i])
            .sum::<f64>();
        let atc = if atc_trt_den > 1e-10 {
            atc_trt_num / atc_trt_den - atc_ctrl_mean
        } else { 0.0 };

        // Sandwich standard errors (influence-function based)
        let ate_se = bootstrap_se_ipw_ate(y, w, &ps_trim.view(), ate, n)?;
        let att_se = bootstrap_se_ipw_att(y, w, &ps_trim.view(), att, n)?;
        let atc_se = ate_se; // simplified

        let ate_p = normal_p_value(if ate_se > 0.0 { ate / ate_se } else { 0.0 });
        let att_p = normal_p_value(if att_se > 0.0 { att / att_se } else { 0.0 });
        let atc_p = normal_p_value(if atc_se > 0.0 { atc / atc_se } else { 0.0 });

        Ok(PSResult {
            ate,
            ate_se,
            att,
            att_se,
            atc,
            atc_se,
            ate_p,
            att_p,
            atc_p,
            propensity_scores: ps_trim,
            estimator: "IPW".into(),
        })
    }
}

/// Influence-function-based SE for ATE
fn bootstrap_se_ipw_ate(
    y: &ArrayView1<f64>,
    w: &ArrayView1<f64>,
    ps: &ArrayView1<f64>,
    ate: f64,
    n: usize,
) -> StatsResult<f64> {
    let psi: Array1<f64> = (0..n).map(|i| {
        let wi = w[i];
        let yi = y[i];
        let pi = ps[i];
        wi * yi / pi - (1.0 - wi) * yi / (1.0 - pi) - ate
    }).collect();
    let var_psi: f64 = psi.iter().map(|&p| p * p).sum::<f64>() / (n * (n - 1).max(1)) as f64;
    Ok(var_psi.sqrt())
}

/// Influence-function-based SE for ATT
fn bootstrap_se_ipw_att(
    y: &ArrayView1<f64>,
    w: &ArrayView1<f64>,
    ps: &ArrayView1<f64>,
    att: f64,
    n: usize,
) -> StatsResult<f64> {
    let n_treated: f64 = w.iter().filter(|&&wi| wi > 0.5).count() as f64;
    if n_treated < 1.0 {
        return Ok(0.0);
    }
    let psi: Array1<f64> = (0..n).map(|i| {
        let wi = w[i];
        let yi = y[i];
        let pi = ps[i];
        // Influence function for ATT
        (wi * yi - (1.0 - wi) * pi * yi / (1.0 - pi)) / (n_treated / n as f64) - att
    }).collect();
    let var_psi: f64 = psi.iter().map(|&p| p * p).sum::<f64>() / (n * (n - 1).max(1)) as f64;
    Ok(var_psi.sqrt())
}

// ---------------------------------------------------------------------------
// Propensity Score Matching
// ---------------------------------------------------------------------------

/// Matching method options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchingMethod {
    /// 1-nearest-neighbour matching without replacement
    NearestNeighbour,
    /// Caliper matching (NN within caliper)
    Caliper,
    /// Kernel matching (weighted average of all controls)
    Kernel,
}

/// Propensity score matching estimator.
pub struct PSMatching {
    /// Matching method
    pub method: MatchingMethod,
    /// Caliper width (for NN and Caliper methods); if `None`, defaults to 0.2 * sd(logit(ps))
    pub caliper: Option<f64>,
    /// Number of nearest neighbours (k:1 matching)
    pub n_neighbours: usize,
    /// Kernel bandwidth for kernel matching
    pub kernel_bandwidth: Option<f64>,
}

impl PSMatching {
    /// Create a new PSMatching estimator.
    pub fn new(method: MatchingMethod) -> Self {
        Self {
            method,
            caliper: None,
            n_neighbours: 1,
            kernel_bandwidth: None,
        }
    }

    /// Estimate ATT via propensity score matching.
    ///
    /// # Arguments
    /// * `y`  – outcome
    /// * `w`  – binary treatment indicator
    /// * `ps` – estimated propensity scores
    pub fn estimate_att(
        &self,
        y: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
        ps: &ArrayView1<f64>,
    ) -> StatsResult<MatchingResult> {
        let n = y.len();
        if w.len() != n || ps.len() != n {
            return Err(StatsError::DimensionMismatch(
                "y, w, ps must have equal length".into(),
            ));
        }

        let treated_idx: Vec<usize> = (0..n).filter(|&i| w[i] > 0.5).collect();
        let control_idx: Vec<usize> = (0..n).filter(|&i| w[i] <= 0.5).collect();

        if treated_idx.is_empty() {
            return Err(StatsError::InsufficientData("No treated units".into()));
        }
        if control_idx.is_empty() {
            return Err(StatsError::InsufficientData("No control units".into()));
        }

        // Caliper in logit(ps) scale
        let logit_ps: Array1<f64> = ps.mapv(|p| logit(p.clamp(1e-8, 1.0 - 1e-8)));
        let logit_sd = std_dev_vec(&logit_ps.to_vec());
        let caliper_val = self.caliper.unwrap_or(0.2 * logit_sd);
        let bw = self.kernel_bandwidth.unwrap_or(0.1 * logit_sd);

        match self.method {
            MatchingMethod::NearestNeighbour | MatchingMethod::Caliper => {
                self.nn_matching_att(y, &treated_idx, &control_idx, &logit_ps.view(), caliper_val)
            }
            MatchingMethod::Kernel => {
                self.kernel_matching_att(y, &treated_idx, &control_idx, ps, bw)
            }
        }
    }

    fn nn_matching_att(
        &self,
        y: &ArrayView1<f64>,
        treated_idx: &[usize],
        control_idx: &[usize],
        logit_ps: &ArrayView1<f64>,
        caliper: f64,
    ) -> StatsResult<MatchingResult> {
        let mut matched_diffs: Vec<f64> = Vec::new();
        let use_caliper = self.method == MatchingMethod::Caliper;

        for &ti in treated_idx {
            let lps_t = logit_ps[ti];
            let best = control_idx
                .iter()
                .map(|&ci| (ci, (logit_ps[ci] - lps_t).abs()))
                .filter(|(_, dist)| !use_caliper || *dist <= caliper)
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((best_ci, _)) = best {
                matched_diffs.push(y[ti] - y[best_ci]);
            }
        }

        if matched_diffs.is_empty() {
            return Err(StatsError::InsufficientData(
                "No matches found; try increasing the caliper".into(),
            ));
        }

        let n_m = matched_diffs.len();
        let att = matched_diffs.iter().sum::<f64>() / n_m as f64;
        let se = if n_m > 1 {
            let var = matched_diffs.iter().map(|&d| (d - att).powi(2)).sum::<f64>()
                / (n_m * (n_m - 1)) as f64;
            var.sqrt()
        } else { 0.0 };
        let t = if se > 0.0 { att / se } else { 0.0 };
        let p = normal_p_value(t);
        let ci = [att - 1.96 * se, att + 1.96 * se];

        let method_name = if self.method == MatchingMethod::Caliper {
            "Caliper-matching"
        } else {
            "NN-matching"
        };

        Ok(MatchingResult {
            att,
            att_se: se,
            p_value: p,
            conf_interval: ci,
            n_matched_treated: n_m,
            method: method_name.into(),
        })
    }

    fn kernel_matching_att(
        &self,
        y: &ArrayView1<f64>,
        treated_idx: &[usize],
        control_idx: &[usize],
        ps: &ArrayView1<f64>,
        bw: f64,
    ) -> StatsResult<MatchingResult> {
        let mut diffs: Vec<f64> = Vec::with_capacity(treated_idx.len());
        for &ti in treated_idx {
            let psi = ps[ti];
            // Epanechnikov kernel weights
            let weights: Vec<f64> = control_idx
                .iter()
                .map(|&ci| {
                    let u = (ps[ci] - psi) / bw;
                    if u.abs() < 1.0 { 0.75 * (1.0 - u * u) } else { 0.0 }
                })
                .collect();
            let total_w: f64 = weights.iter().sum();
            if total_w < 1e-10 {
                continue;
            }
            let y_ctrl_wt: f64 = control_idx
                .iter()
                .zip(weights.iter())
                .map(|(&ci, &wi)| y[ci] * wi)
                .sum::<f64>()
                / total_w;
            diffs.push(y[ti] - y_ctrl_wt);
        }
        if diffs.is_empty() {
            return Err(StatsError::InsufficientData(
                "No matches with positive kernel weight; reduce bandwidth".into(),
            ));
        }
        let n_m = diffs.len();
        let att = diffs.iter().sum::<f64>() / n_m as f64;
        let se = if n_m > 1 {
            let var = diffs.iter().map(|&d| (d - att).powi(2)).sum::<f64>()
                / (n_m * (n_m - 1)) as f64;
            var.sqrt()
        } else { 0.0 };
        let t = if se > 0.0 { att / se } else { 0.0 };
        let p = normal_p_value(t);
        let ci = [att - 1.96 * se, att + 1.96 * se];
        Ok(MatchingResult {
            att,
            att_se: se,
            p_value: p,
            conf_interval: ci,
            n_matched_treated: n_m,
            method: "Kernel-matching".into(),
        })
    }
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

fn std_dev_vec(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 { return 1.0; }
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt().max(1e-15)
}

// ---------------------------------------------------------------------------
// Overlap Check / Common Support
// ---------------------------------------------------------------------------

/// Common-support and overlap diagnostics for propensity score analysis.
pub struct OverlapCheck {
    /// Trimming rule: exclude units with PS outside [min_treated + epsilon, max_control - epsilon]
    pub trim_method: TrimMethod,
}

/// Trimming method for common-support enforcement
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimMethod {
    /// Crump et al. (2009) optimal trimming: exclude if ps < α or ps > 1-α
    Crump,
    /// Min-max trimming: keep only the overlap range
    MinMax,
    /// Percentile trimming: trim the most extreme τ% on each side
    Percentile,
}

impl OverlapCheck {
    /// Create a new OverlapCheck analyser.
    pub fn new(trim_method: TrimMethod) -> Self {
        Self { trim_method }
    }

    /// Compute overlap diagnostics.
    ///
    /// # Arguments
    /// * `ps` – propensity scores for all units
    /// * `w`  – binary treatment indicator
    pub fn check(
        &self,
        ps: &ArrayView1<f64>,
        w: &ArrayView1<f64>,
    ) -> StatsResult<OverlapResult> {
        let n = ps.len();
        if w.len() != n {
            return Err(StatsError::DimensionMismatch("ps and w must have equal length".into()));
        }

        let treated_ps: Vec<f64> = (0..n).filter(|&i| w[i] > 0.5).map(|i| ps[i]).collect();
        let control_ps: Vec<f64> = (0..n).filter(|&i| w[i] <= 0.5).map(|i| ps[i]).collect();

        if treated_ps.is_empty() || control_ps.is_empty() {
            return Err(StatsError::InsufficientData(
                "Need both treated and control units".into(),
            ));
        }

        let (ps_lower, ps_upper) = match self.trim_method {
            TrimMethod::Crump => {
                // Crump optimal α ≈ 0.1 (simple approximation)
                (0.1_f64, 0.9_f64)
            }
            TrimMethod::MinMax => {
                let min_t = treated_ps.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_t = treated_ps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_c = control_ps.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_c = control_ps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (min_t.max(min_c), max_t.min(max_c))
            }
            TrimMethod::Percentile => {
                // Trim 5% on each side
                let alpha = 0.05_f64;
                let all_ps: Vec<f64> = ps.to_vec();
                let lower = quantile_val(&all_ps, alpha);
                let upper = quantile_val(&all_ps, 1.0 - alpha);
                (lower, upper)
            }
        };

        let common_support_idx: Vec<usize> = (0..n)
            .filter(|&i| ps[i] >= ps_lower && ps[i] <= ps_upper)
            .collect();

        let n_t = treated_ps.len() as f64;
        let n_c = control_ps.len() as f64;
        let frac_t = treated_ps.iter().filter(|&&p| p >= ps_lower && p <= ps_upper).count() as f64
            / n_t.max(1.0);
        let frac_c = control_ps.iter().filter(|&&p| p >= ps_lower && p <= ps_upper).count() as f64
            / n_c.max(1.0);

        // Overlap coefficient: approximate as fraction of total PS range covered by both groups
        let overlap_coefficient = overlap_coef(&treated_ps, &control_ps);

        Ok(OverlapResult {
            ps: ps.to_owned(),
            common_support_idx,
            ps_lower,
            ps_upper,
            frac_treated_in_support: frac_t,
            frac_control_in_support: frac_c,
            overlap_coefficient,
        })
    }
}

fn quantile_val(v: &[f64], q: f64) -> f64 {
    if v.is_empty() { return 0.5; }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

/// Overlap coefficient: 1 - total variation distance
fn overlap_coef(ps_t: &[f64], ps_c: &[f64]) -> f64 {
    if ps_t.is_empty() || ps_c.is_empty() { return 0.0; }
    // Grid-based approximation
    let all_min = ps_t.iter().chain(ps_c.iter()).cloned().fold(f64::INFINITY, f64::min);
    let all_max = ps_t.iter().chain(ps_c.iter()).cloned().fold(f64::NEG_INFINITY, f64::max);
    if (all_max - all_min).abs() < 1e-10 { return 1.0; }
    let n_bins = 100_usize;
    let step = (all_max - all_min) / n_bins as f64;
    let mut oc = 0.0_f64;
    for b in 0..n_bins {
        let lo = all_min + b as f64 * step;
        let hi = lo + step;
        let ft = ps_t.iter().filter(|&&p| p >= lo && p < hi).count() as f64 / ps_t.len() as f64;
        let fc = ps_c.iter().filter(|&&p| p >= lo && p < hi).count() as f64 / ps_c.len() as f64;
        oc += ft.min(fc);
    }
    oc
}

// ---------------------------------------------------------------------------
// Full estimation pipeline
// ---------------------------------------------------------------------------

/// Convenience function: estimate ATE/ATT/ATC using propensity score methods.
///
/// Fits a logistic propensity score model and applies IPW.
///
/// # Arguments
/// * `y`  – outcome vector (n,)
/// * `w`  – binary treatment indicator (n,)
/// * `x`  – covariate matrix (n × k)
/// * `trim_eps` – propensity score trimming threshold
pub fn ps_estimate(
    y: &ArrayView1<f64>,
    w: &ArrayView1<f64>,
    x: &ArrayView2<f64>,
    trim_eps: f64,
) -> StatsResult<PSResult> {
    let ps_model = PropensityScoreModel::new();
    let beta = ps_model.fit(x, w)?;
    let ps = ps_model.predict(x, &beta.view())?;
    IPW::estimate(y, w, &ps.view(), trim_eps)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_logistic_regression_ps() {
        // Binary outcome with one covariate; should converge
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0],
                       [5.0], [6.0], [7.0], [8.0], [9.0]];
        let w = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let model = PropensityScoreModel::new();
        let beta = model.fit(&x.view(), &w.view()).expect("Logistic fit should succeed");
        assert_eq!(beta.len(), 2);
        // Coefficient on x should be positive (larger x → more likely treated)
        assert!(beta[1] > 0.0, "Coefficient should be positive, got {}", beta[1]);
        // Predict: units with x>5 should have ps > 0.5
        let ps = model.predict(&x.view(), &beta.view()).expect("Predict should succeed");
        assert!(ps[9] > 0.5, "ps for x=9 should be > 0.5, got {}", ps[9]);
        assert!(ps[0] < 0.5, "ps for x=0 should be < 0.5, got {}", ps[0]);
    }

    #[test]
    fn test_ipw_zero_effect() {
        // No treatment effect: both groups have same outcome distribution
        let n = 100_usize;
        let ps: Array1<f64> = (0..n).map(|i| 0.3 + 0.4 * (i as f64 / n as f64)).collect();
        let w: Array1<f64> = ps.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
        // Outcomes equal to a constant (no effect)
        let y: Array1<f64> = Array1::ones(n);
        let res = IPW::estimate(&y.view(), &w.view(), &ps.view(), 0.05)
            .expect("IPW should succeed");
        assert!(res.ate.abs() < 0.1, "ATE should be ~0 when no effect, got {}", res.ate);
    }

    #[test]
    fn test_ps_matching_nn() {
        let n = 40_usize;
        let ps: Array1<f64> = (0..n).map(|i| 0.1 + 0.8 * i as f64 / n as f64).collect();
        let w: Array1<f64> = ps.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
        // Treatment effect = 2
        let y: Array1<f64> = (0..n).map(|i| if w[i] > 0.5 { 5.0 } else { 3.0 }).collect();
        let matcher = PSMatching::new(MatchingMethod::NearestNeighbour);
        let res = matcher.estimate_att(&y.view(), &w.view(), &ps.view())
            .expect("NN matching should succeed");
        assert!((res.att - 2.0).abs() < 0.5,
            "ATT should be ~2.0, got {}", res.att);
    }

    #[test]
    fn test_overlap_check_minmax() {
        // Ensure treated and control propensity scores overlap:
        // treated PS: 0.3, 0.5, 0.6, 0.7, 0.8  (min=0.3, max=0.8)
        // control PS: 0.1, 0.2, 0.4, 0.5, 0.9  (min=0.1, max=0.9)
        // MinMax common support: [max(0.3,0.1), min(0.8,0.9)] = [0.3, 0.8]
        let ps = array![0.1, 0.3, 0.4, 0.5, 0.5, 0.2, 0.6, 0.7, 0.8, 0.9];
        let w  = array![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let checker = OverlapCheck::new(TrimMethod::MinMax);
        let res = checker.check(&ps.view(), &w.view()).expect("Overlap check should succeed");
        assert!(res.ps_lower < res.ps_upper, "lower={} >= upper={}", res.ps_lower, res.ps_upper);
        assert!(!res.common_support_idx.is_empty());
    }

    #[test]
    fn test_ps_estimate_pipeline() {
        let n = 60_usize;
        let mut x_data = Array2::<f64>::zeros((n, 1));
        let mut w_data = Array1::<f64>::zeros(n);
        let mut y_data = Array1::<f64>::zeros(n);
        for i in 0..n {
            let xi = i as f64 / n as f64;
            x_data[[i, 0]] = xi;
            w_data[i] = if xi > 0.5 { 1.0 } else { 0.0 };
            y_data[i] = if w_data[i] > 0.5 { 3.0 + xi } else { 1.0 + xi };
        }
        let res = ps_estimate(&y_data.view(), &w_data.view(), &x_data.view(), 0.05)
            .expect("PS estimate pipeline should succeed");
        // Treatment effect ≈ 2
        assert!(res.ate.abs() > 0.0, "ATE should be non-zero");
    }
}
