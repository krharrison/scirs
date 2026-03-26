//! EGARCH (Exponential GARCH) volatility model — Nelson (1991)
//!
//! The EGARCH model parameterises the log-conditional variance rather than the
//! variance itself.  This removes the non-negativity constraints on the ARCH/GARCH
//! coefficients and, crucially, allows for asymmetric (leverage) effects via the
//! `γ` (gamma) parameter: negative shocks (bad news) tend to increase volatility
//! more than positive shocks of the same magnitude.
//!
//! # Model Specification
//!
//! ```text
//! ln σ²_t = ω
//!         + Σⱼ βⱼ ln σ²_{t-j}
//!         + Σᵢ [αᵢ (|z_{t-i}| − E|z|) + γᵢ z_{t-i}]
//! ```
//!
//! where `z_t = ε_t / σ_t` are the standardised residuals.
//! Under the assumption of standard normal innovations `E|z| = √(2/π)`.
//!
//! # Stationarity
//! The model is strictly stationary when `|Σⱼ βⱼ| < 1`.
//!
//! # Unconditional Variance
//! ```text
//! E[ln σ²_t] = ω / (1 − Σβⱼ)
//! ```
//! which gives `E[σ²_t] = exp(ω / (1 − Σβⱼ))` (not exact, by Jensen's inequality,
//! but the conventional point estimate).
//!
//! # References
//! - Nelson, D. B. (1991). Conditional Heteroskedasticity in Asset Returns: A New
//!   Approach. *Econometrica*, 59(2), 347–370.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::egarch::{EGARCHModel, fit_egarch};
//! use scirs2_core::ndarray::Array1;
//!
//! let returns: Array1<f64> = Array1::from(vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
//!     0.001, -0.009, 0.016, -0.002, 0.011, 0.006,
//! ]);
//! let model = fit_egarch(&returns, 1, 1).expect("EGARCH fitting failed");
//! println!("omega={:.4}, alpha={:.4}, gamma={:.4}, beta={:.4}",
//!          model.omega, model.alpha[0], model.gamma[0], model.beta[0]);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::arch::{chi2_survival, ln_gamma};

// E[|z|] for standard normal = sqrt(2/π)
const EXPECTED_ABS_Z: f64 = 0.797_884_560_802_865_4; // sqrt(2/pi)

// ============================================================
// EGARCH model struct
// ============================================================

/// EGARCH(p,q) model parameters
///
/// ```text
/// ln σ²_t = ω + Σ βⱼ ln σ²_{t-j}
///             + Σᵢ [αᵢ (|z_{t-i}| − E|z|) + γᵢ z_{t-i}]
/// ```
#[derive(Debug, Clone)]
pub struct EGARCHModel {
    /// ARCH order p
    pub p: usize,
    /// GARCH order q
    pub q: usize,
    /// Constant / intercept in log-variance equation
    pub omega: f64,
    /// Symmetric shock response coefficients αᵢ (length p)
    pub alpha: Vec<f64>,
    /// Asymmetry / leverage coefficients γᵢ (length p)
    pub gamma: Vec<f64>,
    /// Log-variance persistence coefficients βⱼ (length q)
    pub beta: Vec<f64>,
    /// Log-likelihood at the estimated parameters
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl EGARCHModel {
    /// Construct a new `EGARCHModel` from raw parameter vectors.
    ///
    /// # Errors
    /// Returns an error if any vector length does not match the declared order,
    /// or if `|Σβⱼ| >= 1` (non-stationary).
    pub fn new(
        p: usize,
        q: usize,
        omega: f64,
        alpha: Vec<f64>,
        gamma: Vec<f64>,
        beta: Vec<f64>,
    ) -> Result<Self> {
        if alpha.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "alpha length {} does not match ARCH order p={}",
                alpha.len(),
                p
            )));
        }
        if gamma.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "gamma length {} does not match ARCH order p={}",
                gamma.len(),
                p
            )));
        }
        if beta.len() != q {
            return Err(TimeSeriesError::InvalidModel(format!(
                "beta length {} does not match GARCH order q={}",
                beta.len(),
                q
            )));
        }
        let sum_beta: f64 = beta.iter().sum();
        if sum_beta.abs() >= 1.0 {
            return Err(TimeSeriesError::InvalidModel(format!(
                "|Σβ| = {:.4} >= 1.0; model is non-stationary",
                sum_beta.abs()
            )));
        }
        Ok(Self {
            p,
            q,
            omega,
            alpha,
            gamma,
            beta,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Persistence measure: `|Σⱼ βⱼ|`
    pub fn persistence(&self) -> f64 {
        self.beta.iter().sum::<f64>().abs()
    }

    /// Unconditional log-variance (point estimate via Jensen's inequality):
    /// `ω / (1 − Σβⱼ)`
    pub fn unconditional_log_variance(&self) -> Result<f64> {
        let sum_beta: f64 = self.beta.iter().sum();
        let denom = 1.0 - sum_beta;
        if denom.abs() < 1e-12 {
            return Err(TimeSeriesError::NumericalInstability(
                "EGARCH: Σβ ≈ 1 — denominator near zero in unconditional log-variance".into(),
            ));
        }
        Ok(self.omega / denom)
    }

    /// Unconditional variance (point estimate): `exp(ω / (1 − Σβⱼ))`
    pub fn unconditional_variance(&self) -> Result<f64> {
        Ok(self.unconditional_log_variance()?.exp())
    }

    /// AIC = -2 * log-likelihood + 2 * n_params
    pub fn aic(&self) -> f64 {
        -2.0 * self.log_likelihood + 2.0 * self.n_params() as f64
    }

    /// BIC = -2 * log-likelihood + n_params * log(n_obs)
    pub fn bic(&self) -> f64 {
        -2.0 * self.log_likelihood + self.n_params() as f64 * (self.n_obs as f64).ln()
    }

    /// Number of free parameters: 1 (ω) + p (α) + p (γ) + q (β)
    pub fn n_params(&self) -> usize {
        1 + self.p + self.p + self.q
    }
}

// ============================================================
// Log-variance recursion
// ============================================================

/// Compute the conditional log-variance series `h_t = ln σ²_t` for the
/// given EGARCH parameters.
///
/// Returns a vector of length `n` (one per observation in `returns`).
pub fn egarch_log_variance(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>> {
    let n = returns.len();
    let p = alpha.len();
    let q = beta.len();

    if n < p.max(q) + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "EGARCH: too few observations for model order".into(),
            required: p.max(q) + 2,
            actual: n,
        });
    }

    // Initial log-variance: log of sample variance
    let mean = returns.mean().unwrap_or(0.0);
    let sample_var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let init_log_var = if sample_var > 0.0 {
        sample_var.ln()
    } else {
        -10.0_f64
    };

    let mut h = vec![init_log_var; n]; // log-variance
    let mut z = vec![0.0_f64; n]; // standardised residuals

    for t in 1..n {
        // Compute z_{t-1} using current sigma
        let sigma_prev = h[t - 1].exp().sqrt().max(1e-12);
        z[t - 1] = returns[t - 1] / sigma_prev;

        let mut log_var_t = omega;

        // βⱼ * h_{t-j}
        for (j, &bj) in beta.iter().enumerate() {
            let lag = j + 1;
            if t >= lag {
                log_var_t += bj * h[t - lag];
            } else {
                log_var_t += bj * init_log_var;
            }
        }

        // αᵢ (|z_{t-i}| − E|z|) + γᵢ z_{t-i}
        for (i, (&ai, &gi)) in alpha.iter().zip(gamma.iter()).enumerate() {
            let lag = i + 1;
            let zt_i = if t >= lag { z[t - lag] } else { 0.0 };
            log_var_t += ai * (zt_i.abs() - EXPECTED_ABS_Z) + gi * zt_i;
        }

        h[t] = log_var_t;
    }

    Ok(h)
}

// ============================================================
// Log-likelihood
// ============================================================

/// Compute the Gaussian log-likelihood for EGARCH(p,q) parameters.
///
/// Returns `(log_likelihood, h_series)`.
pub fn egarch_log_likelihood(
    returns: &Array1<f64>,
    omega: f64,
    alpha: &[f64],
    gamma: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let h = egarch_log_variance(returns, omega, alpha, gamma, beta)?;
    let n = returns.len();
    let burn_in = alpha.len().max(beta.len()).max(1);

    let mut ll = 0.0_f64;
    for t in burn_in..n {
        let sigma2 = h[t].exp();
        if sigma2 <= 0.0 || !sigma2.is_finite() {
            return Err(TimeSeriesError::NumericalInstability(
                "EGARCH: non-positive or non-finite conditional variance".into(),
            ));
        }
        let eps = returns[t];
        ll += -0.5 * (std::f64::consts::TAU.ln() + h[t] + eps * eps / sigma2);
    }

    if !ll.is_finite() {
        return Err(TimeSeriesError::NumericalInstability(
            "EGARCH: log-likelihood is not finite".into(),
        ));
    }

    Ok((ll, h))
}

// ============================================================
// Fitting via Nelder-Mead
// ============================================================

/// Fit an EGARCH(p,q) model to the given return series using Nelder-Mead
/// maximisation of the Gaussian log-likelihood.
///
/// # Arguments
/// * `returns` — observed return series (demeaned internally)
/// * `p` — ARCH order (number of lagged standardised innovations)
/// * `q` — GARCH order (number of lagged log-variances)
///
/// # Errors
/// Returns an error if the data are insufficient or optimisation fails to
/// produce a stationary model.
pub fn fit_egarch(returns: &Array1<f64>, p: usize, q: usize) -> Result<EGARCHModel> {
    let n = returns.len();
    let min_obs = (p.max(q) + 1) * 5;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: "EGARCH: need at least 5*(max(p,q)+1) observations".into(),
            required: min_obs,
            actual: n,
        });
    }

    // Demean
    let mean = returns.mean().unwrap_or(0.0);
    let r: Array1<f64> = returns.mapv(|x| x - mean);

    // Initial parameter guess
    // theta layout: [omega, alpha_0..p-1, gamma_0..p-1, beta_0..q-1]
    let n_params = 1 + 2 * p + q;
    let sample_var = r.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    let init_log_var = if sample_var > 0.0 {
        sample_var.ln()
    } else {
        -8.0
    };

    let mut theta = vec![0.0_f64; n_params];
    theta[0] = init_log_var * 0.05; // omega
    for i in 0..p {
        theta[1 + i] = 0.10; // alpha
        theta[1 + p + i] = -0.05; // gamma (leverage, expect negative)
    }
    for j in 0..q {
        theta[1 + 2 * p + j] = 0.80 / q as f64; // beta
    }

    // Nelder-Mead minimisation of negative log-likelihood
    let objective = |th: &[f64]| -> f64 {
        let omega = th[0];
        let alpha: Vec<f64> = th[1..1 + p].to_vec();
        let gamma: Vec<f64> = th[1 + p..1 + 2 * p].to_vec();
        let beta: Vec<f64> = th[1 + 2 * p..].to_vec();

        // Enforce stationarity
        let sum_beta: f64 = beta.iter().sum::<f64>().abs();
        if sum_beta >= 0.9999 {
            return 1e15;
        }

        match egarch_log_likelihood(&r, omega, &alpha, &gamma, &beta) {
            Ok((ll, _)) => {
                if ll.is_finite() {
                    -ll
                } else {
                    1e15
                }
            }
            Err(_) => 1e15,
        }
    };

    let best_theta = nelder_mead(&theta, &objective, 2000, 1e-8)?;

    let omega = best_theta[0];
    let alpha: Vec<f64> = best_theta[1..1 + p].to_vec();
    let gamma: Vec<f64> = best_theta[1 + p..1 + 2 * p].to_vec();
    let beta: Vec<f64> = best_theta[1 + 2 * p..].to_vec();

    let sum_beta: f64 = beta.iter().sum::<f64>().abs();
    if sum_beta >= 1.0 {
        return Err(TimeSeriesError::FittingError(
            "EGARCH: fitted beta sum violates stationarity".into(),
        ));
    }

    let (ll, _) = egarch_log_likelihood(&r, omega, &alpha, &gamma, &beta)?;

    let mut model = EGARCHModel::new(p, q, omega, alpha, gamma, beta)?;
    model.log_likelihood = ll;
    model.n_obs = n;
    Ok(model)
}

// ============================================================
// Forecast
// ============================================================

/// Compute `h`-step-ahead conditional log-variance forecasts.
///
/// The multi-step forecast exploits the recursion:
/// ```text
/// E[ln σ²_{T+h}] = ω/(1-Σβ) + (Σβ)^h * (ln σ²_T − ω/(1-Σβ))
/// ```
///
/// Returns a vector of length `h` with the forecasted **variances** (not log-variances).
pub fn egarch_forecast(model: &EGARCHModel, returns: &Array1<f64>, h: usize) -> Result<Vec<f64>> {
    if h == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "forecast horizon h must be >= 1".into(),
        ));
    }

    let mean = returns.mean().unwrap_or(0.0);
    let r: Array1<f64> = returns.mapv(|x| x - mean);

    let log_var_series =
        egarch_log_variance(&r, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    let last_log_var = *log_var_series
        .last()
        .ok_or_else(|| TimeSeriesError::InsufficientData {
            message: "EGARCH forecast: empty log-variance series".into(),
            required: 1,
            actual: 0,
        })?;

    let sum_beta: f64 = model.beta.iter().sum();
    let uncond_log_var = model.unconditional_log_variance().unwrap_or(last_log_var);

    let mut forecasts = Vec::with_capacity(h);
    let mut current = last_log_var;

    for _ in 0..h {
        // E[ln σ²_{t+1}] = ω + Σβ * current  (α terms vanish in expectation)
        let next_log_var = model.omega + sum_beta * current;
        let _ = next_log_var; // suppress lint
                              // Mean-reverting recursion toward unconditional log-variance
        current = uncond_log_var + sum_beta * (current - uncond_log_var);
        forecasts.push(current.exp());
    }

    Ok(forecasts)
}

/// Compute the conditional volatility series (σ_t) from an EGARCH model.
pub fn egarch_conditional_volatility(
    model: &EGARCHModel,
    returns: &Array1<f64>,
) -> Result<Vec<f64>> {
    let mean = returns.mean().unwrap_or(0.0);
    let r: Array1<f64> = returns.mapv(|x| x - mean);
    let log_var = egarch_log_variance(&r, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    Ok(log_var.into_iter().map(|lv| lv.exp().sqrt()).collect())
}

/// Compute standardised residuals `z_t = ε_t / σ_t`.
pub fn egarch_standardised_residuals(
    model: &EGARCHModel,
    returns: &Array1<f64>,
) -> Result<Vec<f64>> {
    let mean = returns.mean().unwrap_or(0.0);
    let r: Array1<f64> = returns.mapv(|x| x - mean);
    let log_var = egarch_log_variance(&r, model.omega, &model.alpha, &model.gamma, &model.beta)?;
    let resid: Vec<f64> = r
        .iter()
        .zip(log_var.iter())
        .map(|(&eps, &lv)| eps / lv.exp().sqrt().max(1e-12))
        .collect();
    Ok(resid)
}

// ============================================================
// Nelder-Mead simplex optimiser (local, no external deps)
// ============================================================

/// Nelder-Mead downhill simplex minimisation.
///
/// Minimises `f(x)` starting from the initial point `x0`.  Returns the
/// approximate minimiser.
pub(crate) fn nelder_mead<F>(x0: &[f64], f: &F, max_iter: usize, tol: f64) -> Result<Vec<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Nelder-Mead: empty initial point".into(),
        ));
    }

    // Build initial simplex: x0 plus n perturbed vertices
    let step = 0.05_f64.max(x0.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) * 0.05);
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] += if vertex[i].abs() < 1e-10 {
            0.00025
        } else {
            step
        };
        simplex.push(vertex);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha_nm = 1.0_f64; // reflection
    let gamma_nm = 2.0_f64; // expansion
    let rho_nm = 0.5_f64; // contraction
    let sigma_nm = 0.5_f64; // shrink

    for _iter in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let (best_idx, worst_idx, second_worst_idx) = (order[0], order[n], order[n - 1]);

        // Convergence check
        let range = fvals[worst_idx] - fvals[best_idx];
        if range < tol {
            break;
        }

        // Centroid of all points except worst
        let mut centroid = vec![0.0_f64; n];
        for &idx in &order[0..n] {
            for (d, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][d];
            }
        }
        centroid.iter_mut().for_each(|c| *c /= n as f64);

        // Reflection
        let reflected: Vec<f64> = centroid
            .iter()
            .zip(simplex[worst_idx].iter())
            .map(|(&c, &w)| c + alpha_nm * (c - w))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best_idx] {
            // Expansion
            let expanded: Vec<f64> = centroid
                .iter()
                .zip(reflected.iter())
                .map(|(&c, &r)| c + gamma_nm * (r - c))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                fvals[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                fvals[worst_idx] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst_idx] {
            simplex[worst_idx] = reflected;
            fvals[worst_idx] = f_reflected;
        } else {
            // Contraction
            let (ref_pt, f_ref) = if f_reflected < fvals[worst_idx] {
                (reflected.clone(), f_reflected)
            } else {
                (simplex[worst_idx].clone(), fvals[worst_idx])
            };
            let contracted: Vec<f64> = centroid
                .iter()
                .zip(ref_pt.iter())
                .map(|(&c, &r)| c + rho_nm * (r - c))
                .collect();
            let f_contracted = f(&contracted);
            if f_contracted < f_ref {
                simplex[worst_idx] = contracted;
                fvals[worst_idx] = f_contracted;
            } else {
                // Shrink
                for i in 1..=n {
                    let idx = order[i];
                    let shrunk: Vec<f64> = simplex[best_idx]
                        .iter()
                        .zip(simplex[idx].iter())
                        .map(|(&b, &v)| b + sigma_nm * (v - b))
                        .collect();
                    fvals[idx] = f(&shrunk);
                    simplex[idx] = shrunk;
                }
            }
        }
    }

    // Return best vertex
    let best = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok(simplex[best].clone())
}

// ============================================================
// Re-export helper for arch tests
// ============================================================

/// Chi-squared survival function re-exported for use in arch_tests module.
pub(crate) use crate::volatility::arch::{
    chi2_survival as arch_chi2_survival, ln_gamma as arch_ln_gamma,
};

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_returns() -> Array1<f64> {
        // 50 synthetic return values with some volatility clustering
        Array1::from(vec![
            0.008, -0.015, 0.011, -0.007, 0.013, 0.005, -0.018, 0.009, -0.003, 0.007, 0.025,
            -0.014, 0.008, -0.006, 0.011, -0.019, 0.022, 0.003, -0.011, 0.017, -0.004, 0.008,
            -0.013, 0.019, 0.001, -0.009, 0.016, -0.002, 0.011, 0.006, 0.010, -0.020, 0.014,
            -0.009, 0.015, 0.003, -0.016, 0.010, -0.004, 0.008, 0.023, -0.012, 0.007, -0.007,
            0.012, -0.021, 0.018, 0.002, -0.010, 0.016,
        ])
    }

    #[test]
    fn test_egarch_log_variance_basic() {
        let r = make_returns();
        let h = egarch_log_variance(&r, -0.2, &[0.1], &[-0.05], &[0.85])
            .expect("Should compute log-variance");
        assert_eq!(h.len(), r.len());
        for &lv in &h {
            assert!(lv.is_finite(), "log-variance should be finite");
        }
    }

    #[test]
    fn test_egarch_log_likelihood_basic() {
        let r = make_returns();
        let (ll, h) = egarch_log_likelihood(&r, -0.2, &[0.1], &[-0.05], &[0.85])
            .expect("Should compute log-likelihood");
        assert!(ll.is_finite());
        assert_eq!(h.len(), r.len());
    }

    #[test]
    fn test_egarch_model_new_valid() {
        let m = EGARCHModel::new(1, 1, -0.2, vec![0.1], vec![-0.05], vec![0.85])
            .expect("Should create model");
        assert!((m.persistence() - 0.85).abs() < 1e-10);
        let uv = m.unconditional_variance().expect("Should compute");
        assert!(uv > 0.0);
    }

    #[test]
    fn test_egarch_model_new_non_stationary() {
        assert!(EGARCHModel::new(1, 1, -0.2, vec![0.1], vec![-0.05], vec![1.05]).is_err());
    }

    #[test]
    fn test_egarch_model_new_wrong_lengths() {
        assert!(EGARCHModel::new(1, 1, -0.2, vec![0.1, 0.05], vec![-0.05], vec![0.85]).is_err());
    }

    #[test]
    fn test_fit_egarch_11() {
        let r = make_returns();
        let model = fit_egarch(&r, 1, 1).expect("EGARCH(1,1) should fit");
        assert!(model.omega.is_finite());
        assert!(model.alpha[0].is_finite());
        assert!(model.gamma[0].is_finite());
        assert!(model.beta[0].is_finite());
        assert!(model.persistence() < 1.0);
        assert!(model.log_likelihood.is_finite());
        let aic = model.aic();
        let bic = model.bic();
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(bic >= aic); // BIC penalises more for n > e^2
    }

    #[test]
    fn test_egarch_forecast() {
        let r = make_returns();
        let model = fit_egarch(&r, 1, 1).expect("EGARCH(1,1) should fit");
        let forecasts = egarch_forecast(&model, &r, 5).expect("Should forecast");
        assert_eq!(forecasts.len(), 5);
        for &f in &forecasts {
            assert!(f > 0.0, "Forecasted variance must be positive: {f}");
            assert!(f.is_finite());
        }
    }

    #[test]
    fn test_egarch_conditional_volatility() {
        let r = make_returns();
        let model = EGARCHModel::new(1, 1, -0.2, vec![0.1], vec![-0.05], vec![0.85])
            .expect("Should create");
        let vol = egarch_conditional_volatility(&model, &r).expect("Should compute volatility");
        assert_eq!(vol.len(), r.len());
        for &v in &vol {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_egarch_standardised_residuals() {
        let r = make_returns();
        let model = EGARCHModel::new(1, 1, -0.2, vec![0.1], vec![-0.05], vec![0.85])
            .expect("Should create");
        let z = egarch_standardised_residuals(&model, &r).expect("Should compute");
        assert_eq!(z.len(), r.len());
        for &zi in &z {
            assert!(zi.is_finite());
        }
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        // Minimise (x-2)^2 + (y+1)^2 starting from (0,0)
        let x0 = vec![0.0, 0.0];
        let obj = |v: &[f64]| (v[0] - 2.0).powi(2) + (v[1] + 1.0).powi(2);
        let best = nelder_mead(&x0, &obj, 2000, 1e-10).expect("Should optimise");
        assert!((best[0] - 2.0).abs() < 1e-4, "x ≈ 2.0, got {}", best[0]);
        assert!((best[1] + 1.0).abs() < 1e-4, "y ≈ -1.0, got {}", best[1]);
    }
}
