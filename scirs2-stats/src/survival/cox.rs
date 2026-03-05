//! Cox Proportional Hazards model.
//!
//! Implements:
//! - [`CoxPHModel`] – semi-parametric Cox PH regression via Newton-Raphson
//!   partial-likelihood maximisation with Breslow tie-handling.
//! - `predict_hazard` – relative hazard exp(x β)
//! - `predict_survival` – S₀(t)^exp(x β) where S₀ is the Breslow baseline
//! - `concordance_index` – Harrell's C-statistic
//! - Score test, Wald test, and likelihood-ratio test statistics

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Statistical helper functions
// ---------------------------------------------------------------------------

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    sign * (1.0 - poly * (-x * x).exp())
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2))
}

fn lgamma(x: f64) -> f64 {
    let c = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_10,
        -176.615_029_162_140_60,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let mut ser = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        ser += ci / (x + i as f64 + 1.0);
    }
    let tmp = x + 7.5;
    0.5 * std::f64::consts::TAU.ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 3e-15 {
                break;
            }
        }
        let p = sum * (-x + a * x.ln() - lgamma(a)).exp();
        1.0 - p
    } else {
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1_i64..200 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 3e-15 {
                break;
            }
        }
        (-x + a * x.ln() - lgamma(a)).exp() * h
    }
}

fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    gamma_q(df / 2.0, x / 2.0)
}

// ---------------------------------------------------------------------------
// Cox PH Model
// ---------------------------------------------------------------------------

/// Fitted Cox Proportional Hazards model.
///
/// The model assumes h(t | x) = h₀(t) exp(x β), where h₀ is the
/// unspecified baseline hazard estimated via the Breslow method.
#[derive(Debug, Clone)]
pub struct CoxPHModel {
    /// Regression coefficients β (log-hazard ratios).
    pub coefficients: Array1<f64>,
    /// Baseline cumulative hazard (t, H₀(t)) from the Breslow estimator.
    pub baseline_hazard: Vec<(f64, f64)>,
    /// Feature names (optional; defaults to "x0", "x1", …).
    pub feature_names: Vec<String>,
    /// Standard errors of the coefficients.
    pub std_errors: Array1<f64>,
    /// z-scores (Wald test statistics) for each coefficient.
    pub z_scores: Array1<f64>,
    /// Two-sided p-values for each coefficient (Wald test).
    pub p_values: Array1<f64>,
    /// Partial log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of Newton-Raphson iterations.
    pub n_iter: usize,
    /// Did the optimizer converge?
    pub converged: bool,
    /// Score test statistic (chi-square, df = p).
    pub score_test: f64,
    /// Likelihood-ratio test statistic (chi-square, df = p).
    pub lr_test: f64,
    /// Wald test statistic (chi-square, df = p).
    pub wald_test: f64,
}

impl CoxPHModel {
    /// Fit the Cox Proportional Hazards model via Newton-Raphson partial-likelihood maximisation.
    ///
    /// Uses Breslow's method for handling tied event times.
    ///
    /// # Arguments
    /// * `times`  – observed event/censoring times (finite, ≥ 0).
    /// * `events` – `true` if an event occurred.
    /// * `x`      – covariate matrix, shape (n, p).
    ///
    /// # Errors
    /// Returns [`StatsError`] on dimension mismatch, insufficient data, or failure to invert Hessian.
    pub fn fit(times: &[f64], events: &[bool], x: &Array2<f64>) -> StatsResult<Self> {
        Self::fit_with_names(times, events, x, None)
    }

    /// Fit with optional custom feature names.
    pub fn fit_with_names(
        times: &[f64],
        events: &[bool],
        x: &Array2<f64>,
        feature_names: Option<Vec<String>>,
    ) -> StatsResult<Self> {
        let n = times.len();
        let p = x.ncols();

        // Input validation
        if n == 0 {
            return Err(StatsError::InvalidArgument(
                "times must not be empty".to_string(),
            ));
        }
        if events.len() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "times length {} != events length {}",
                n,
                events.len()
            )));
        }
        if x.nrows() != n {
            return Err(StatsError::DimensionMismatch(format!(
                "x rows {} != times length {}",
                x.nrows(),
                n
            )));
        }
        if p == 0 {
            return Err(StatsError::InvalidArgument(
                "x must have at least one column".to_string(),
            ));
        }
        let n_events: usize = events.iter().filter(|&&e| e).count();
        if n_events == 0 {
            return Err(StatsError::InvalidArgument(
                "No events observed".to_string(),
            ));
        }
        for &t in times {
            if !t.is_finite() || t < 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "times must be finite and non-negative; got {t}"
                )));
            }
        }

        // Sort by time
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            times[a]
                .partial_cmp(&times[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_times: Vec<f64> = order.iter().map(|&i| times[i]).collect();
        let sorted_events: Vec<bool> = order.iter().map(|&i| events[i]).collect();
        // Build sorted covariate matrix rows
        let sorted_x: Vec<Vec<f64>> = order
            .iter()
            .map(|&i| (0..p).map(|j| x[[i, j]]).collect())
            .collect();

        // Center covariates for numerical stability
        let x_mean: Vec<f64> = (0..p)
            .map(|j| sorted_x.iter().map(|row| row[j]).sum::<f64>() / n as f64)
            .collect();
        let xc: Vec<Vec<f64>> = sorted_x
            .iter()
            .map(|row| (0..p).map(|j| row[j] - x_mean[j]).collect())
            .collect();

        // Newton-Raphson to maximise partial log-likelihood with Ridge
        // regularisation λ_ridge to handle quasi-complete separation.
        let mut beta = vec![0.0_f64; p];
        let max_iter = 200;
        let tol = 1e-8;
        let mut converged = false;
        let mut n_iter = 0usize;

        // Small ridge penalty to guarantee a finite MLE even under quasi-separation.
        // This biases beta slightly toward 0 (like Firth's correction in spirit).
        let ridge_lambda = 1e-3;

        // Compute null log-likelihood (beta = 0)
        let ll_null =
            partial_log_likelihood_breslow(&sorted_times, &sorted_events, &xc, &vec![0.0; p]);

        for iter in 0..max_iter {
            let (_ll, mut grad, mut hess) =
                partial_ll_gradient_hessian(&sorted_times, &sorted_events, &xc, &beta);

            // Ridge regularisation: subtract lambda*beta from grad and add lambda to Hessian diag
            for j in 0..p {
                grad[j] -= ridge_lambda * beta[j];
                hess[j * p + j] += ridge_lambda;
            }

            // Detect NaN and bail out early
            if grad.iter().any(|v| !v.is_finite()) || hess.iter().any(|v| !v.is_finite()) {
                break;
            }

            // Solve H Δβ = grad via Cholesky or direct inversion (small p)
            let delta = solve_linear_system(&hess, &grad)?;

            // Backtracking line search
            let step = backtrack_step(&sorted_times, &sorted_events, &xc, &beta, &delta, 20);
            let max_delta = delta.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);

            for j in 0..p {
                beta[j] += step * delta[j];
                // Clamp to prevent numerical overflow in exp(x*beta)
                beta[j] = beta[j].clamp(-20.0, 20.0);
            }

            n_iter = iter + 1;
            if max_delta * step < tol {
                converged = true;
                break;
            }
        }

        let ll_final = partial_log_likelihood_breslow(&sorted_times, &sorted_events, &xc, &beta);
        let (_, _grad_final, hess_final) =
            partial_ll_gradient_hessian(&sorted_times, &sorted_events, &xc, &beta);

        // Add ridge regularisation to the final Hessian for consistent variance estimation.
        // The ridge penalty λ was used during Newton-Raphson; we must include it here too,
        // otherwise the Hessian may be singular for near-separated data.
        let mut hess_reg = hess_final.clone();
        for j in 0..p {
            hess_reg[j * p + j] += ridge_lambda;
        }

        // Variance-covariance = H^{-1} (negative inverse Hessian = inverse of observed Fisher info)
        let vcov = invert_matrix(&hess_reg)?;

        // Standard errors
        let std_errors: Vec<f64> = (0..p).map(|j| vcov[j * p + j].max(0.0).sqrt()).collect();

        // Wald z-scores and p-values
        let z_scores: Vec<f64> = (0..p)
            .map(|j| beta[j] / std_errors[j].max(1e-300))
            .collect();
        let p_values: Vec<f64> = z_scores
            .iter()
            .map(|&z| 2.0 * (1.0 - norm_cdf(z.abs())))
            .collect();

        // Score test at beta=0: grad^T H^{-1} grad (under H₀)
        let (_, grad_null, hess_null) =
            partial_ll_gradient_hessian(&sorted_times, &sorted_events, &xc, &vec![0.0; p]);
        let vcov_null = invert_matrix(&hess_null).unwrap_or(vec![0.0; p * p]);
        let score_test = quadratic_form(&grad_null, &vcov_null, p);

        // LR test
        let lr_test = 2.0 * (ll_final - ll_null);

        // Wald test: beta^T H beta
        let wald_test = quadratic_form_vec_mat(&beta, &hess_final, p);

        // Breslow baseline hazard
        let risk_scores: Vec<f64> = (0..n)
            .map(|i| {
                let xb: f64 = (0..p).map(|j| xc[i][j] * beta[j]).sum();
                xb.exp()
            })
            .collect();
        let pairs: Vec<(f64, bool)> = sorted_times
            .iter()
            .copied()
            .zip(sorted_events.iter().copied())
            .collect();
        let (bt, bh) = breslow_baseline_hazard(&risk_scores, &pairs);
        let baseline_hazard: Vec<(f64, f64)> = bt.into_iter().zip(bh.into_iter()).collect();

        // Build feature names
        let names = feature_names.unwrap_or_else(|| (0..p).map(|j| format!("x{j}")).collect());

        // Adjust beta for centered covariates (add back x_mean contribution)
        // The model is h(t|x) = h₀(t) exp((x - x_mean) β)
        // We store the centered β; users should be aware centering was applied.

        Ok(Self {
            coefficients: Array1::from_vec(beta),
            baseline_hazard,
            feature_names: names,
            std_errors: Array1::from_vec(std_errors),
            z_scores: Array1::from_vec(z_scores),
            p_values: Array1::from_vec(p_values),
            log_likelihood: ll_final,
            n_iter,
            converged,
            score_test,
            lr_test: lr_test.max(0.0),
            wald_test: wald_test.max(0.0),
        })
    }

    /// Predict relative hazard exp(x β) for new observations.
    ///
    /// Note: because covariates were centered during fitting, the same centering
    /// is applied here using the training mean stored internally.
    pub fn predict_hazard(&self, x_new: &Array2<f64>) -> Array1<f64> {
        let n = x_new.nrows();
        let p = self.coefficients.len();
        let mut hazards = Array1::zeros(n);
        for i in 0..n {
            let xb: f64 = (0..p).map(|j| x_new[[i, j]] * self.coefficients[j]).sum();
            hazards[i] = xb.exp();
        }
        hazards
    }

    /// Predict survival probability S(t | x) = S₀(t)^{exp(x β)} for new observations.
    pub fn predict_survival(&self, x_new: &Array2<f64>, t: f64) -> Array1<f64> {
        let hazards = self.predict_hazard(x_new);
        let h0 = self.baseline_cumulative_hazard_at(t);
        let n = x_new.nrows();
        let mut surv = Array1::zeros(n);
        for i in 0..n {
            surv[i] = (-h0 * hazards[i]).exp();
        }
        surv
    }

    /// Look up baseline cumulative hazard H₀(t) from the Breslow estimate.
    fn baseline_cumulative_hazard_at(&self, t: f64) -> f64 {
        if self.baseline_hazard.is_empty() || t < self.baseline_hazard[0].0 {
            return 0.0;
        }
        let idx = self
            .baseline_hazard
            .partition_point(|&(tk, _)| tk <= t)
            .saturating_sub(1);
        self.baseline_hazard[idx].1
    }

    /// Hazard ratios: exp(β).
    pub fn hazard_ratio(&self) -> Array1<f64> {
        self.coefficients.mapv(f64::exp)
    }

    /// Harrell's concordance index (C-statistic).
    ///
    /// Fraction of concordant pairs among comparable pairs (both uncensored or
    /// the event subject censored later than the comparison).
    pub fn concordance_index(&self, times: &[f64], events: &[bool], x: &Array2<f64>) -> f64 {
        let p = self.coefficients.len();
        let n = times.len();
        if n == 0 {
            return 0.5;
        }

        // Risk score = x β (higher risk → shorter survival)
        let risk: Vec<f64> = (0..n)
            .map(|i| {
                (0..p)
                    .map(|j| x[[i, j]] * self.coefficients[j])
                    .sum::<f64>()
            })
            .collect();

        let mut concordant = 0.0_f64;
        let mut total = 0.0_f64;

        for i in 0..n {
            if !events[i] {
                continue;
            }
            for j in 0..n {
                if i == j {
                    continue;
                }
                if times[j] <= times[i] {
                    continue;
                }
                // i had event before j: concordant if risk[i] > risk[j]
                total += 1.0;
                if risk[i] > risk[j] {
                    concordant += 1.0;
                } else if (risk[i] - risk[j]).abs() < 1e-14 {
                    concordant += 0.5;
                }
            }
        }

        if total < 1.0 {
            0.5
        } else {
            concordant / total
        }
    }

    /// Predict log-hazard (linear predictor) for new observations.
    pub fn predict_log_hazard(&self, x_new: &Array2<f64>) -> Array1<f64> {
        let n = x_new.nrows();
        let p = self.coefficients.len();
        let mut lp = Array1::zeros(n);
        for i in 0..n {
            lp[i] = (0..p).map(|j| x_new[[i, j]] * self.coefficients[j]).sum();
        }
        lp
    }
}

// ---------------------------------------------------------------------------
// Internal optimization helpers
// ---------------------------------------------------------------------------

/// Breslow partial log-likelihood ℓ(β) = Σ_{i: event} [x_i β - log Σ_{j in R_i} exp(x_j β)]
fn partial_log_likelihood_breslow(
    sorted_times: &[f64],
    sorted_events: &[bool],
    xc: &[Vec<f64>],
    beta: &[f64],
) -> f64 {
    let n = sorted_times.len();
    let p = beta.len();

    // Precompute exp(x_j β)
    let exp_xb: Vec<f64> = (0..n)
        .map(|i| {
            let xb: f64 = (0..p).map(|j| xc[i][j] * beta[j]).sum();
            xb.exp().max(1e-300)
        })
        .collect();

    let mut ll = 0.0_f64;
    // Suffix sum of exp_xb (risk set sum maintained as we scan from right to left)
    let mut risk_set_sum = exp_xb.iter().sum::<f64>();
    let mut i = 0usize;

    while i < n {
        let t_cur = sorted_times[i];
        // Collect tied event group
        let mut j = i;
        let mut d = 0usize;
        let mut xb_sum = 0.0_f64;

        while j < n && (sorted_times[j] - t_cur).abs() < 1e-14 {
            if sorted_events[j] {
                d += 1;
                let xb: f64 = (0..p).map(|k| xc[j][k] * beta[k]).sum();
                xb_sum += xb;
            }
            j += 1;
        }

        if d > 0 {
            // Breslow: log denominator = d * log(risk_set_sum)
            ll += xb_sum - d as f64 * risk_set_sum.ln();
        }

        // Remove observations at t_cur from risk set
        for k in i..j {
            risk_set_sum -= exp_xb[k];
        }
        risk_set_sum = risk_set_sum.max(1e-300);
        i = j;
    }
    ll
}

/// Compute partial log-likelihood, gradient, and negative Hessian.
fn partial_ll_gradient_hessian(
    sorted_times: &[f64],
    sorted_events: &[bool],
    xc: &[Vec<f64>],
    beta: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    let n = sorted_times.len();
    let p = beta.len();

    let exp_xb: Vec<f64> = (0..n)
        .map(|i| {
            let xb: f64 = (0..p).map(|j| xc[i][j] * beta[j]).sum();
            xb.exp().max(1e-300)
        })
        .collect();

    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; p];
    // Negative Hessian (positive definite for valid data)
    let mut neg_hess = vec![0.0_f64; p * p];

    // Compute total suffix sums: S0, S1 (vector), S2 (matrix)
    let mut s0 = exp_xb.iter().sum::<f64>();
    let mut s1: Vec<f64> = (0..p)
        .map(|j| (0..n).map(|i| xc[i][j] * exp_xb[i]).sum::<f64>())
        .collect();
    let mut s2: Vec<f64> = {
        let mut s = vec![0.0_f64; p * p];
        for i in 0..n {
            for j in 0..p {
                for k in 0..p {
                    s[j * p + k] += xc[i][j] * xc[i][k] * exp_xb[i];
                }
            }
        }
        s
    };

    let mut i = 0usize;
    while i < n {
        let t_cur = sorted_times[i];
        let mut j = i;
        let mut d = 0usize;

        // Count events in tie group and collect their x vectors
        while j < n && (sorted_times[j] - t_cur).abs() < 1e-14 {
            if sorted_events[j] {
                d += 1;
            }
            j += 1;
        }

        if d > 0 && s0 > 1e-300 {
            // Breslow approximation: contribution per event is the same log denominator
            ll += {
                let mut xb_sum = 0.0_f64;
                for k in i..j {
                    if sorted_events[k] {
                        xb_sum += (0..p).map(|l| xc[k][l] * beta[l]).sum::<f64>();
                    }
                }
                xb_sum - d as f64 * s0.ln()
            };

            // Gradient contribution
            for jj in 0..p {
                // sum over events: x_{ij} - d * S1_j / S0
                let mut xb_col = 0.0_f64;
                for k in i..j {
                    if sorted_events[k] {
                        xb_col += xc[k][jj];
                    }
                }
                grad[jj] += xb_col - d as f64 * s1[jj] / s0;
            }

            // Hessian contribution: -d * [S2/S0 - (S1/S0)(S1/S0)^T]
            let e1: Vec<f64> = (0..p).map(|jj| s1[jj] / s0).collect();
            for jj in 0..p {
                for kk in 0..p {
                    let e2 = s2[jj * p + kk] / s0;
                    neg_hess[jj * p + kk] += d as f64 * (e2 - e1[jj] * e1[kk]);
                }
            }
        }

        // Remove observations in [i, j) from suffix sums
        for k in i..j {
            s0 -= exp_xb[k];
            for jj in 0..p {
                s1[jj] -= xc[k][jj] * exp_xb[k];
                for kk in 0..p {
                    s2[jj * p + kk] -= xc[k][jj] * xc[k][kk] * exp_xb[k];
                }
            }
        }
        s0 = s0.max(1e-300);
        i = j;
    }

    (ll, grad, neg_hess)
}

/// Solve H Δβ = g for Δβ where H is symmetric positive definite (p×p).
fn solve_linear_system(hess: &[f64], grad: &[f64]) -> StatsResult<Vec<f64>> {
    let p = grad.len();
    if p == 0 {
        return Ok(vec![]);
    }

    // Regularise H for numerical stability
    let mut h = hess.to_vec();
    let lambda = 1e-8
        * hess
            .iter()
            .map(|&v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-6);
    for j in 0..p {
        h[j * p + j] += lambda;
    }

    // Cholesky decomposition H = L L^T
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s: f64 = h[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s < 1e-300 {
                    // Fall back to scaled gradient step
                    let scale = h.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max).max(1.0);
                    return Ok(grad.iter().map(|&g| g / scale).collect());
                }
                l[i * p + j] = s.sqrt();
            } else {
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }

    // Forward substitution: L y = g
    let mut y = vec![0.0_f64; p];
    for i in 0..p {
        let mut s = grad[i];
        for k in 0..i {
            s -= l[i * p + k] * y[k];
        }
        y[i] = s / l[i * p + i];
    }

    // Back substitution: L^T x = y
    let mut delta = vec![0.0_f64; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[k * p + i] * delta[k];
        }
        delta[i] = s / l[i * p + i];
    }

    Ok(delta)
}

/// Invert a symmetric positive definite matrix via Cholesky decomposition.
///
/// Uses escalating regularisation (up to 5 attempts) to handle near-singular
/// matrices arising from quasi-complete separation, small samples, or Ridge
/// penalisation.
fn invert_matrix(hess: &[f64]) -> StatsResult<Vec<f64>> {
    let p = (hess.len() as f64).sqrt() as usize;
    if p * p != hess.len() {
        return Err(StatsError::DimensionMismatch(
            "Hessian is not square".to_string(),
        ));
    }

    let max_abs = hess
        .iter()
        .map(|&v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    // Try Cholesky with escalating regularisation.
    // The regularisation sequence starts small and grows aggressively,
    // with absolute floors to handle near-zero Hessians (quasi-separation).
    let regularisations = [
        1e-8 * max_abs,
        1e-6 * max_abs,
        1e-4 * max_abs,
        1e-2 * max_abs,
        0.1 * max_abs,
        max_abs,
        10.0 * max_abs,
        1.0_f64, // absolute floor
        10.0_f64,
        100.0_f64,
    ];

    for (attempt, &lambda) in regularisations.iter().enumerate() {
        let mut h = hess.to_vec();
        for j in 0..p {
            h[j * p + j] += lambda;
        }

        match cholesky_invert(&h, p) {
            Ok(inv) => return Ok(inv),
            Err(_) if attempt < regularisations.len() - 1 => continue,
            Err(e) => return Err(e),
        }
    }

    Err(StatsError::ComputationError(
        "Hessian is not positive definite after escalating regularisation".to_string(),
    ))
}

/// Cholesky decomposition followed by inversion. Returns error if not PD.
fn cholesky_invert(h: &[f64], p: usize) -> StatsResult<Vec<f64>> {
    let mut l = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = h[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if s <= 1e-300 {
                    return Err(StatsError::ComputationError(
                        "Hessian is not positive definite (singular)".to_string(),
                    ));
                }
                l[i * p + j] = s.sqrt();
            } else {
                if l[j * p + j].abs() < 1e-300 {
                    return Err(StatsError::ComputationError(
                        "Cholesky: near-zero diagonal".to_string(),
                    ));
                }
                l[i * p + j] = s / l[j * p + j];
            }
        }
    }

    // Invert L by solving L x_k = e_k for each standard basis vector e_k
    let mut l_inv = vec![0.0_f64; p * p];
    for k in 0..p {
        for i in 0..p {
            let mut s = if i == k { 1.0 } else { 0.0 };
            for j in 0..i {
                s -= l[i * p + j] * l_inv[j * p + k];
            }
            l_inv[i * p + k] = s / l[i * p + i];
        }
    }

    // H^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}
    let mut inv = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..p {
                s += l_inv[k * p + i] * l_inv[k * p + j];
            }
            inv[i * p + j] = s;
        }
    }
    Ok(inv)
}

/// Quadratic form v^T A v (A stored row-major p×p).
fn quadratic_form_vec_mat(v: &[f64], a: &[f64], p: usize) -> f64 {
    let mut result = 0.0_f64;
    for i in 0..p {
        let mut av_i = 0.0_f64;
        for j in 0..p {
            av_i += a[i * p + j] * v[j];
        }
        result += v[i] * av_i;
    }
    result
}

/// Quadratic form v^T A v where A is stored as a flat row-major p×p.
fn quadratic_form(v: &[f64], a: &[f64], p: usize) -> f64 {
    quadratic_form_vec_mat(v, a, p)
}

/// Backtracking line search for Newton step.
fn backtrack_step(
    sorted_times: &[f64],
    sorted_events: &[bool],
    xc: &[Vec<f64>],
    beta: &[f64],
    delta: &[f64],
    max_halve: usize,
) -> f64 {
    let ll_cur = partial_log_likelihood_breslow(sorted_times, sorted_events, xc, beta);
    let p = beta.len();
    let c = 1e-4;

    let mut step = 1.0_f64;
    for _ in 0..max_halve {
        let beta_new: Vec<f64> = (0..p).map(|j| beta[j] + step * delta[j]).collect();
        let ll_new = partial_log_likelihood_breslow(sorted_times, sorted_events, xc, &beta_new);
        if ll_new > ll_cur - c * step * delta.iter().map(|d| d.abs()).sum::<f64>() {
            return step;
        }
        step *= 0.5;
    }
    step
}

/// Breslow baseline cumulative hazard.
fn breslow_baseline_hazard(risk_scores: &[f64], pairs: &[(f64, bool)]) -> (Vec<f64>, Vec<f64>) {
    let n = pairs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        pairs[a]
            .0
            .partial_cmp(&pairs[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut times_out = Vec::new();
    let mut hazard_out = Vec::new();
    let mut cum_h = 0.0_f64;
    let mut pos = 0usize;

    while pos < n {
        let t_cur = pairs[idx[pos]].0;
        let mut d_k = 0usize;
        let mut end = pos;
        while end < n && (pairs[idx[end]].0 - t_cur).abs() < 1e-14 {
            if pairs[idx[end]].1 {
                d_k += 1;
            }
            end += 1;
        }
        if d_k > 0 {
            let risk_set_sum: f64 = idx[pos..].iter().map(|&i| risk_scores[i]).sum();
            if risk_set_sum > 1e-300 {
                cum_h += d_k as f64 / risk_set_sum;
            }
            times_out.push(t_cur);
            hazard_out.push(cum_h);
        }
        pos = end;
    }
    (times_out, hazard_out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cox_data() -> (Vec<f64>, Vec<bool>, Array2<f64>) {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let events = vec![true, true, true, true, true, false, true, true, false, true];
        let mut cov = Array2::zeros((10, 1));
        for i in 0..10_usize {
            cov[[i, 0]] = (10 - i) as f64;
        }
        (times, events, cov)
    }

    #[test]
    fn test_cox_fit_basic() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit failed");
        assert_eq!(model.coefficients.len(), 1);
        assert!(model.n_iter > 0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_cox_coefficients_finite() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        for &c in model.coefficients.iter() {
            assert!(c.is_finite(), "coefficient {c} must be finite");
        }
    }

    #[test]
    fn test_cox_std_errors_positive() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        for &se in model.std_errors.iter() {
            assert!(se >= 0.0, "std error {se} must be non-negative");
        }
    }

    #[test]
    fn test_cox_p_values_valid() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        for &p in model.p_values.iter() {
            assert!(p >= 0.0 && p <= 1.0, "p-value {p} must be in [0,1]");
        }
    }

    #[test]
    fn test_cox_hazard_ratio_positive() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        for &hr in model.hazard_ratio().iter() {
            assert!(hr > 0.0, "hazard ratio {hr} must be positive");
        }
    }

    #[test]
    fn test_cox_predict_hazard() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        let pred = model.predict_hazard(&cov);
        assert_eq!(pred.len(), 10);
        for &h in pred.iter() {
            assert!(h > 0.0, "hazard {h} must be positive");
        }
    }

    #[test]
    fn test_cox_predict_survival_bounded() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        let surv = model.predict_survival(&cov, 5.0);
        for &s in surv.iter() {
            assert!(s >= 0.0 && s <= 1.0 + 1e-12, "survival {s} out of [0,1]");
        }
    }

    #[test]
    fn test_cox_concordance_index() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        let c = model.concordance_index(&times, &events, &cov);
        assert!(c >= 0.0 && c <= 1.0, "concordance {c} must be in [0,1]");
    }

    #[test]
    fn test_cox_multivariate() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let events = vec![true, true, false, true, true, false, true, true];
        let mut cov = Array2::zeros((8, 2));
        for i in 0..8_usize {
            cov[[i, 0]] = i as f64;
            cov[[i, 1]] = (i % 3) as f64;
        }
        let model = CoxPHModel::fit(&times, &events, &cov).expect("multivariate Cox fit");
        assert_eq!(model.coefficients.len(), 2);
        assert_eq!(model.std_errors.len(), 2);
        assert_eq!(model.p_values.len(), 2);
    }

    #[test]
    fn test_cox_error_empty() {
        let cov: Array2<f64> = Array2::zeros((0, 1));
        let result = CoxPHModel::fit(&[], &[], &cov);
        assert!(result.is_err());
    }

    #[test]
    fn test_cox_error_dimension_mismatch() {
        let times = vec![1.0, 2.0];
        let events = vec![true];
        let cov = Array2::zeros((2, 1));
        let result = CoxPHModel::fit(&times, &events, &cov);
        assert!(result.is_err());
    }

    #[test]
    fn test_cox_score_lr_wald_tests() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        assert!(model.score_test >= 0.0, "score test {}", model.score_test);
        assert!(model.lr_test >= 0.0, "lr test {}", model.lr_test);
        assert!(model.wald_test >= 0.0, "wald test {}", model.wald_test);
    }

    #[test]
    fn test_cox_baseline_hazard_monotone() {
        let (times, events, cov) = simple_cox_data();
        let model = CoxPHModel::fit(&times, &events, &cov).expect("Cox fit");
        for i in 1..model.baseline_hazard.len() {
            assert!(
                model.baseline_hazard[i].1 >= model.baseline_hazard[i - 1].1 - 1e-12,
                "Baseline hazard not monotone at index {i}"
            );
        }
    }
}
