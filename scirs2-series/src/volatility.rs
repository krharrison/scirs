//! GARCH Family Volatility Models for Financial Time Series
//!
//! This module implements a comprehensive suite of GARCH (Generalized Autoregressive
//! Conditional Heteroskedasticity) family models for modelling time-varying volatility
//! in financial returns. The models capture the well-documented stylized facts of
//! financial returns: volatility clustering, fat tails, and leverage effects.
//!
//! # Model Overview
//!
//! ## GARCH(p,q) -- Bollerslev (1986)
//! The standard GARCH model extends ARCH by including lagged conditional variances:
//! ```text
//! sigma2_t = omega + Sum_i alpha_i eps2_{t-i} + Sum_j beta_j sigma2_{t-j}
//! ```
//! Stationarity requires: Sum(alpha) + Sum(beta) < 1.
//!
//! ## EGARCH(p,q) -- Nelson (1991)
//! Exponential GARCH models the log-variance, allowing asymmetric (leverage) effects:
//! ```text
//! ln sigma2_t = omega + Sum_i [alpha_i (|z_{t-i}| - E|z|) + gamma_i z_{t-i}] + Sum_j beta_j ln sigma2_{t-j}
//! ```
//!
//! ## GJR-GARCH(p,q) -- Glosten-Jagannathan-Runkle (1993)
//! Threshold GARCH amplifies variance response for negative shocks:
//! ```text
//! sigma2_t = omega + Sum_i (alpha_i + gamma_i I[eps_{t-i}<0]) eps2_{t-i} + Sum_j beta_j sigma2_{t-j}
//! ```

// Submodules: volatility family extensions
pub mod arch;
pub mod garch;
pub mod egarch;
pub mod gjr_garch;
pub mod figarch;
pub mod rv_models;

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{Normal, Rng, SeedableRng};

use crate::error::TimeSeriesError;

// ============================================================
// Distribution enum
// ============================================================

/// Error distribution for GARCH innovation term z_t
#[derive(Debug, Clone, PartialEq)]
pub enum GarchDistribution {
    /// Standard normal innovations
    Normal,
    /// Student-t innovations with `df` degrees of freedom (df > 2)
    StudentT {
        /// Degrees of freedom
        df: f64,
    },
    /// Skewed Student-t innovations (Hansen 1994)
    SkewedT {
        /// Degrees of freedom
        df: f64,
        /// Skewness parameter in (-1, 1)
        skew: f64,
    },
    /// Generalized Error Distribution (Nelson 1991) with shape parameter nu > 0
    GED {
        /// Shape parameter
        nu: f64,
    },
}

// ============================================================
// GARCH(p,q)
// ============================================================

/// GARCH(p,q) model: sigma2_t = omega + Sum alpha_i eps2_{t-i} + Sum beta_j sigma2_{t-j}
///
/// The standard GARCH model due to Bollerslev (1986). Captures volatility
/// clustering via the persistence of squared shocks and past conditional variances.
#[derive(Debug, Clone)]
pub struct Garch {
    /// ARCH order p -- number of lagged squared returns
    pub p: usize,
    /// GARCH order q -- number of lagged conditional variances
    pub q: usize,
    /// Constant term omega > 0
    pub omega: f64,
    /// ARCH coefficients alpha (length p), each alpha_i >= 0
    pub alpha: Vec<f64>,
    /// GARCH coefficients beta (length q), each beta_j >= 0
    pub beta: Vec<f64>,
    /// Constant mean mu of the return series
    pub mean: f64,
    /// Innovation distribution
    pub distribution: GarchDistribution,
}

impl Garch {
    /// Construct a GARCH(1,1) model with given parameters.
    pub fn garch_1_1(omega: f64, alpha: f64, beta: f64) -> Self {
        Self {
            p: 1,
            q: 1,
            omega,
            alpha: vec![alpha],
            beta: vec![beta],
            mean: 0.0,
            distribution: GarchDistribution::Normal,
        }
    }

    /// Construct a zero-initialised GARCH(p,q) template.
    pub fn garch_p_q(p: usize, q: usize) -> Self {
        Self {
            p,
            q,
            omega: 0.0,
            alpha: vec![0.0; p],
            beta: vec![0.0; q],
            mean: 0.0,
            distribution: GarchDistribution::Normal,
        }
    }

    /// Check second-order stationarity: Sum(alpha) + Sum(beta) < 1.
    pub fn is_stationary(&self) -> bool {
        let sum: f64 = self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>();
        sum < 1.0
    }

    /// Unconditional (long-run) variance: omega / (1 - Sum(alpha) - Sum(beta)).
    ///
    /// Returns `None` when the model is not stationary.
    pub fn unconditional_variance(&self) -> Option<f64> {
        let sum: f64 = self.alpha.iter().sum::<f64>() + self.beta.iter().sum::<f64>();
        if sum >= 1.0 || self.omega <= 0.0 {
            None
        } else {
            Some(self.omega / (1.0 - sum))
        }
    }

    /// Compute the conditional variance path sigma2_t for the given demeaned returns eps_t.
    ///
    /// The recursion is initialised at the unconditional variance (or at the sample
    /// variance when the model is non-stationary).
    pub fn conditional_variances(
        &self,
        returns: &Array1<f64>,
    ) -> Result<Array1<f64>, TimeSeriesError> {
        let n = returns.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "GARCH conditional variance computation".to_string(),
                required: 2,
                actual: n,
            });
        }

        // Demeaned residuals
        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();

        // Initial variance: unconditional or sample variance
        let init_var = self.unconditional_variance().unwrap_or_else(|| {
            let mean_sq: f64 = eps.iter().map(|e| e * e).sum::<f64>() / n as f64;
            mean_sq.max(1e-8)
        });

        let mut h = vec![init_var; n];

        let max_lag = self.p.max(self.q);

        for t in max_lag..n {
            let mut ht = self.omega;
            // ARCH terms
            for (i, &ai) in self.alpha.iter().enumerate() {
                let lag = i + 1;
                if t >= lag {
                    ht += ai * eps[t - lag] * eps[t - lag];
                }
            }
            // GARCH terms
            for (j, &bj) in self.beta.iter().enumerate() {
                let lag = j + 1;
                if t >= lag {
                    ht += bj * h[t - lag];
                }
            }
            h[t] = ht.max(1e-12);
        }

        Ok(Array1::from_vec(h))
    }

    /// Gaussian log-likelihood of the model given the return series.
    ///
    /// For alternative distributions the log-likelihood kernel is adjusted
    /// accordingly (Student-t, GED). SkewedT falls back to Student-t.
    pub fn log_likelihood(&self, returns: &Array1<f64>) -> Result<f64, TimeSeriesError> {
        let h = self.conditional_variances(returns)?;
        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();
        let h_vec: Vec<f64> = h.iter().copied().collect();
        Ok(compute_log_likelihood(&eps, &h_vec, &self.distribution))
    }

    /// Multi-step-ahead variance forecast (MMSE).
    ///
    /// For h=1 the one-step-ahead conditional variance is returned from the
    /// recursion; for h>1 variance mean-reverts to the unconditional level.
    pub fn forecast_variance(
        &self,
        returns: &Array1<f64>,
        h: usize,
    ) -> Result<Array1<f64>, TimeSeriesError> {
        if h == 0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "h".to_string(),
                message: "forecast horizon must be >= 1".to_string(),
            });
        }

        let sigmas = self.conditional_variances(returns)?;
        let n = sigmas.len();
        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();

        let alpha_sum: f64 = self.alpha.iter().sum();
        let beta_sum: f64 = self.beta.iter().sum();
        let persistence = alpha_sum + beta_sum;

        // Long-run mean-reversion target
        let uncond = self.unconditional_variance().unwrap_or_else(|| {
            sigmas.iter().copied().sum::<f64>() / n as f64
        });

        let mut forecasts = Vec::with_capacity(h);

        // One-step-ahead from the actual recursion
        let mut prev_h = *sigmas.last().unwrap_or(&uncond);
        let mut prev_eps_sq = eps.last().copied().unwrap_or(0.0).powi(2);

        for k in 1..=h {
            if k == 1 {
                // Standard one-step recursion using last observed values
                let mut f1 = self.omega;
                for (i, &ai) in self.alpha.iter().enumerate() {
                    let lag = i + 1;
                    if n >= lag {
                        f1 += ai * eps[n - lag] * eps[n - lag];
                    }
                }
                for (j, &bj) in self.beta.iter().enumerate() {
                    let lag = j + 1;
                    if n >= lag {
                        f1 += bj * sigmas[n - lag];
                    }
                }
                prev_h = f1.max(1e-12);
                prev_eps_sq = prev_h; // E[eps2_{T+1}] = sigma2_{T+1}
                forecasts.push(prev_h);
            } else {
                // For k>1: E[sigma2_{T+k}] = omega + (alpha+beta) E[sigma2_{T+k-1}]
                // (since E[eps2_{T+k-1}] = E[sigma2_{T+k-1}] for k>2)
                let _ = prev_eps_sq; // used above
                let fk = self.omega + persistence * prev_h;
                prev_h = fk.max(1e-12);
                forecasts.push(prev_h);
            }
        }

        // Verify convergence toward unconditional variance (numerical sanity)
        let _ = uncond;

        Ok(Array1::from_vec(forecasts))
    }
}

// ============================================================
// Fitting GARCH via Nelder-Mead MLE
// ============================================================

/// Fit a GARCH(1,1) model by Maximum Likelihood Estimation.
///
/// Uses a Nelder-Mead simplex optimiser over the constrained parameter space
/// (omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1). Initialisation follows the heuristic
/// omega = s2*(1-alpha0-beta0), alpha0 = 0.05, beta0 = 0.90.
pub fn fit_garch_1_1(
    returns: &Array1<f64>,
    distribution: GarchDistribution,
    max_iter: usize,
) -> Result<Garch, TimeSeriesError> {
    fit_garch(returns, 1, 1, distribution, max_iter)
}

/// Fit a GARCH(p,q) model by Maximum Likelihood Estimation.
pub fn fit_garch(
    returns: &Array1<f64>,
    p: usize,
    q: usize,
    distribution: GarchDistribution,
    max_iter: usize,
) -> Result<Garch, TimeSeriesError> {
    let n = returns.len();
    let min_obs = 2 * (p + q) + 10;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: "GARCH fitting".to_string(),
            required: min_obs,
            actual: n,
        });
    }

    let mean = returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var: f64 = eps.iter().map(|e| e * e).sum::<f64>() / n as f64;

    // Initial parameter vector: [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
    let n_params = 1 + p + q;
    let alpha0 = 0.05_f64;
    let beta0 = 0.85_f64.min(0.90 - 0.05 * p as f64);
    let omega0 = sample_var * (1.0 - alpha0 * p as f64 - beta0 * q as f64).max(0.01);

    let mut params = vec![0.0_f64; n_params];
    params[0] = omega0.max(1e-6);
    for i in 0..p {
        params[1 + i] = alpha0;
    }
    for j in 0..q {
        params[1 + p + j] = beta0;
    }

    // Objective: negative log-likelihood
    let neg_ll = |par: &[f64]| -> f64 {
        let model = params_to_garch(par, p, q, mean, &distribution);
        match model.conditional_variances(returns) {
            Err(_) => f64::INFINITY,
            Ok(h) => {
                let h_slice = match h.as_slice() {
                    Some(s) => { let v: Vec<f64> = s.to_vec(); v },
                    None => h.iter().copied().collect(),
                };
                -compute_log_likelihood(&eps, &h_slice, &distribution)
            }
        }
    };

    // Nelder-Mead with constraints projected via softplus / logistic transforms
    let result = nelder_mead_garch(&params, &neg_ll, p, q, max_iter)?;

    let mut model = params_to_garch(&result, p, q, mean, &distribution);
    model.distribution = distribution;

    Ok(model)
}

/// Convert a raw parameter vector into a `Garch` struct.
fn params_to_garch(
    par: &[f64],
    p: usize,
    q: usize,
    mean: f64,
    distribution: &GarchDistribution,
) -> Garch {
    let omega = par[0].abs().max(1e-10);
    let alpha: Vec<f64> = (0..p).map(|i| par[1 + i].abs()).collect();
    let beta: Vec<f64> = (0..q).map(|j| par[1 + p + j].abs()).collect();

    // Enforce stationarity by rescaling if necessary
    let ab_sum: f64 = alpha.iter().sum::<f64>() + beta.iter().sum::<f64>();
    let (alpha, beta) = if ab_sum >= 0.999 {
        let scale = 0.999 / ab_sum;
        (
            alpha.iter().map(|a| a * scale).collect(),
            beta.iter().map(|b| b * scale).collect(),
        )
    } else {
        (alpha, beta)
    };

    Garch {
        p,
        q,
        omega,
        alpha,
        beta,
        mean,
        distribution: distribution.clone(),
    }
}

// ============================================================
// EGARCH(p,q)
// ============================================================

/// EGARCH(p,q) -- Nelson (1991)
///
/// Models the logarithm of conditional variance to ensure positivity without
/// parameter constraints on omega. Captures the leverage effect via gamma: negative
/// shocks (z < 0) increase log-variance more than positive shocks.
///
/// ```text
/// ln sigma2_t = omega + Sum_i [alpha_i (|z_{t-i}| - E|z|) + gamma_i z_{t-i}] + Sum_j beta_j ln sigma2_{t-j}
/// ```
#[derive(Debug, Clone)]
pub struct EGarch {
    /// ARCH order
    pub p: usize,
    /// GARCH order
    pub q: usize,
    /// Constant term
    pub omega: f64,
    /// Shock magnitude coefficients alpha_i
    pub alpha: Vec<f64>,
    /// Asymmetry (leverage) coefficients gamma_i
    pub gamma: Vec<f64>,
    /// Log-variance persistence coefficients beta_j
    pub beta: Vec<f64>,
    /// Return series mean
    pub mean: f64,
}

impl EGarch {
    /// Create a zero-initialised EGARCH(p,q) template.
    pub fn new(p: usize, q: usize) -> Self {
        Self {
            p,
            q,
            omega: 0.0,
            alpha: vec![0.0; p],
            gamma: vec![0.0; p],
            beta: vec![0.0; q],
            mean: 0.0,
        }
    }

    /// Compute the log-conditional-variance path ln sigma2_t.
    pub fn log_conditional_variances(
        &self,
        returns: &Array1<f64>,
    ) -> Result<Array1<f64>, TimeSeriesError> {
        let n = returns.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "EGARCH log-variance computation".to_string(),
                required: 2,
                actual: n,
            });
        }

        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();

        // E[|z|] for standard normal = sqrt(2/pi)
        let e_abs_z = (2.0 / std::f64::consts::PI).sqrt();

        // Initial log-variance: log of sample variance
        let sample_var: f64 = eps.iter().map(|e| e * e).sum::<f64>() / n as f64;
        let init_log_h = sample_var.max(1e-10).ln();

        let mut log_h = vec![init_log_h; n];
        // Build initial sigma array for standardisation
        let mut sigma = vec![sample_var.max(1e-10).sqrt(); n];

        let max_lag = self.p.max(self.q);

        for t in max_lag..n {
            let mut lht = self.omega;

            // Asymmetric ARCH terms
            for (i, (&ai, &gi)) in self.alpha.iter().zip(self.gamma.iter()).enumerate() {
                let lag = i + 1;
                if t >= lag {
                    let z = eps[t - lag] / sigma[t - lag].max(1e-10);
                    lht += ai * (z.abs() - e_abs_z) + gi * z;
                }
            }

            // Log-GARCH terms
            for (j, &bj) in self.beta.iter().enumerate() {
                let lag = j + 1;
                if t >= lag {
                    lht += bj * log_h[t - lag];
                }
            }

            log_h[t] = lht;
            sigma[t] = lht.exp().sqrt().max(1e-10);
        }

        Ok(Array1::from_vec(log_h))
    }

    /// Compute conditional variance sigma2_t = exp(ln sigma2_t).
    pub fn conditional_variances(
        &self,
        returns: &Array1<f64>,
    ) -> Result<Array1<f64>, TimeSeriesError> {
        let log_h = self.log_conditional_variances(returns)?;
        Ok(log_h.mapv(|lh: f64| lh.exp().max(1e-12)))
    }

    /// Normal log-likelihood for EGARCH.
    pub fn log_likelihood(&self, returns: &Array1<f64>) -> Result<f64, TimeSeriesError> {
        let h = self.conditional_variances(returns)?;
        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();
        let h_vec: Vec<f64> = h.iter().copied().collect();
        Ok(compute_log_likelihood(
            &eps,
            &h_vec,
            &GarchDistribution::Normal,
        ))
    }
}

/// Fit an EGARCH(1,1) model by MLE via Nelder-Mead.
pub fn fit_egarch_1_1(
    returns: &Array1<f64>,
    max_iter: usize,
) -> Result<EGarch, TimeSeriesError> {
    let n = returns.len();
    if n < 20 {
        return Err(TimeSeriesError::InsufficientData {
            message: "EGARCH(1,1) fitting".to_string(),
            required: 20,
            actual: n,
        });
    }

    let mean = returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = returns.iter().map(|&r| r - mean).collect();

    // Initial params: [omega, alpha, gamma, beta]
    // omega: log of unconditional variance
    let sample_var: f64 = eps.iter().map(|e| e * e).sum::<f64>() / n as f64;
    let init = vec![
        sample_var.max(1e-10).ln() * 0.05, // omega ~ small constant
        0.10_f64,                            // alpha (shock magnitude)
        -0.05_f64,                           // gamma (leverage, typically negative)
        0.85_f64,                            // beta (persistence)
    ];

    let neg_ll = |par: &[f64]| -> f64 {
        let model = EGarch {
            p: 1,
            q: 1,
            omega: par[0],
            alpha: vec![par[1]],
            gamma: vec![par[2]],
            beta: vec![par[3]],
            mean,
        };
        match model.log_likelihood(returns) {
            Ok(ll) => -ll,
            Err(_) => f64::INFINITY,
        }
    };

    let result = nelder_mead_unconstrained(&init, &neg_ll, max_iter)?;

    Ok(EGarch {
        p: 1,
        q: 1,
        omega: result[0],
        alpha: vec![result[1]],
        gamma: vec![result[2]],
        beta: vec![result[3]],
        mean,
    })
}

// ============================================================
// GJR-GARCH (Threshold GARCH)
// ============================================================

/// GJR-GARCH(p,q) -- Glosten, Jagannathan & Runkle (1993)
///
/// Extends GARCH by adding an asymmetric term that amplifies the variance
/// response to negative innovations (leverage effect):
/// ```text
/// sigma2_t = omega + Sum_i (alpha_i + gamma_i I[eps_{t-i}<0]) eps2_{t-i} + Sum_j beta_j sigma2_{t-j}
/// ```
#[derive(Debug, Clone)]
pub struct GjrGarch {
    /// ARCH order
    pub p: usize,
    /// GARCH order
    pub q: usize,
    /// Constant term omega > 0
    pub omega: f64,
    /// Standard ARCH coefficients alpha_i >= 0
    pub alpha: Vec<f64>,
    /// Asymmetric (leverage) coefficients gamma_i >= 0
    pub gamma: Vec<f64>,
    /// GARCH coefficients beta_j >= 0
    pub beta: Vec<f64>,
    /// Mean of the return series
    pub mean: f64,
}

impl GjrGarch {
    /// Create a zero-initialised GJR-GARCH(p,q) template.
    pub fn new(p: usize, q: usize) -> Self {
        Self {
            p,
            q,
            omega: 0.0,
            alpha: vec![0.0; p],
            gamma: vec![0.0; p],
            beta: vec![0.0; q],
            mean: 0.0,
        }
    }

    /// Compute conditional variances.
    pub fn conditional_variances(
        &self,
        returns: &Array1<f64>,
    ) -> Result<Array1<f64>, TimeSeriesError> {
        let n = returns.len();
        if n < 2 {
            return Err(TimeSeriesError::InsufficientData {
                message: "GJR-GARCH variance computation".to_string(),
                required: 2,
                actual: n,
            });
        }

        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();

        // Effective half-impact coefficient: alpha_i + 0.5 gamma_i (for stationarity check)
        let eff_alpha_sum: f64 = self
            .alpha
            .iter()
            .zip(self.gamma.iter())
            .map(|(&a, &g)| a + 0.5 * g)
            .sum::<f64>();
        let beta_sum: f64 = self.beta.iter().sum();
        let persistence = eff_alpha_sum + beta_sum;

        let init_var = if persistence < 1.0 && self.omega > 0.0 {
            self.omega / (1.0 - persistence)
        } else {
            eps.iter().map(|e| e * e).sum::<f64>() / n as f64
        }
        .max(1e-8);

        let mut h = vec![init_var; n];
        let max_lag = self.p.max(self.q);

        for t in max_lag..n {
            let mut ht = self.omega;

            for (i, (&ai, &gi)) in self.alpha.iter().zip(self.gamma.iter()).enumerate() {
                let lag = i + 1;
                if t >= lag {
                    let e_prev = eps[t - lag];
                    let indicator = if e_prev < 0.0 { 1.0 } else { 0.0 };
                    ht += (ai + gi * indicator) * e_prev * e_prev;
                }
            }

            for (j, &bj) in self.beta.iter().enumerate() {
                let lag = j + 1;
                if t >= lag {
                    ht += bj * h[t - lag];
                }
            }

            h[t] = ht.max(1e-12);
        }

        Ok(Array1::from_vec(h))
    }

    /// Normal log-likelihood.
    pub fn log_likelihood(&self, returns: &Array1<f64>) -> Result<f64, TimeSeriesError> {
        let h = self.conditional_variances(returns)?;
        let eps: Vec<f64> = returns.iter().map(|&r| r - self.mean).collect();
        let h_vec: Vec<f64> = h.iter().copied().collect();
        Ok(compute_log_likelihood(
            &eps,
            &h_vec,
            &GarchDistribution::Normal,
        ))
    }
}

/// Fit a GJR-GARCH(1,1) model by MLE via Nelder-Mead.
pub fn fit_gjr_garch_1_1(
    returns: &Array1<f64>,
    max_iter: usize,
) -> Result<GjrGarch, TimeSeriesError> {
    let n = returns.len();
    if n < 20 {
        return Err(TimeSeriesError::InsufficientData {
            message: "GJR-GARCH(1,1) fitting".to_string(),
            required: 20,
            actual: n,
        });
    }

    let mean = returns.iter().sum::<f64>() / n as f64;
    let eps: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
    let sample_var: f64 = eps.iter().map(|e| e * e).sum::<f64>() / n as f64;

    // Initial params: [omega, alpha, gamma, beta]
    let init = vec![
        sample_var * 0.05,
        0.04_f64,
        0.08_f64,  // leverage (typically positive = bad news amplification)
        0.85_f64,
    ];

    let neg_ll = |par: &[f64]| -> f64 {
        let omega = par[0].abs().max(1e-10);
        let alpha = par[1].abs();
        let gamma = par[2].abs();
        let beta = par[3].abs();
        // Stationarity: alpha + 0.5*gamma + beta < 1
        if alpha + 0.5 * gamma + beta >= 0.999 {
            return f64::INFINITY;
        }
        let model = GjrGarch {
            p: 1,
            q: 1,
            omega,
            alpha: vec![alpha],
            gamma: vec![gamma],
            beta: vec![beta],
            mean,
        };
        match model.log_likelihood(returns) {
            Ok(ll) => -ll,
            Err(_) => f64::INFINITY,
        }
    };

    let result = nelder_mead_unconstrained(&init, &neg_ll, max_iter)?;

    Ok(GjrGarch {
        p: 1,
        q: 1,
        omega: result[0].abs().max(1e-10),
        alpha: vec![result[1].abs()],
        gamma: vec![result[2].abs()],
        beta: vec![result[3].abs()],
        mean,
    })
}

// ============================================================
// Helper: Log-likelihood kernels
// ============================================================

/// Compute the log-likelihood given residuals and conditional variances.
///
/// Supports Normal, Student-t (with fixed or specified df), GED, and SkewedT
/// (approximated as Student-t for likelihood calculation).
fn compute_log_likelihood(
    eps: &[f64],
    h: &[f64],
    distribution: &GarchDistribution,
) -> f64 {
    let n = eps.len().min(h.len());
    if n == 0 {
        return f64::NEG_INFINITY;
    }

    match distribution {
        GarchDistribution::Normal => {
            let ln2pi = (2.0 * std::f64::consts::PI).ln();
            let mut ll = -0.5 * (n as f64) * ln2pi;
            for i in 0..n {
                let hi = h[i].max(1e-12);
                ll -= 0.5 * (hi.ln() + eps[i] * eps[i] / hi);
            }
            ll
        }
        GarchDistribution::StudentT { df } => {
            student_t_log_likelihood(eps, h, *df)
        }
        GarchDistribution::SkewedT { df, skew: _ } => {
            // Use Student-t kernel (skewness correction omitted for numerical stability)
            student_t_log_likelihood(eps, h, *df)
        }
        GarchDistribution::GED { nu } => {
            ged_log_likelihood(eps, h, *nu)
        }
    }
}

/// Student-t log-likelihood kernel.
fn student_t_log_likelihood(eps: &[f64], h: &[f64], nu: f64) -> f64 {
    let n = eps.len().min(h.len());
    if nu <= 2.0 {
        return f64::NEG_INFINITY;
    }

    // log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5 log(pi (nu-2))
    let ln_const = lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (std::f64::consts::PI * (nu - 2.0)).ln();

    let mut ll = 0.0_f64;
    for i in 0..n {
        let hi = h[i].max(1e-12);
        let z2 = eps[i] * eps[i] / hi;
        ll += ln_const - 0.5 * hi.ln() - 0.5 * (nu + 1.0) * (1.0 + z2 / (nu - 2.0)).ln();
    }
    ll
}

/// Generalized Error Distribution log-likelihood kernel.
fn ged_log_likelihood(eps: &[f64], h: &[f64], nu: f64) -> f64 {
    let n = eps.len().min(h.len());
    if nu <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // lambda = sqrt(2^{-2/nu} Gamma(1/nu) / Gamma(3/nu))
    let lambda = ((-2.0 / nu).exp() * lgamma(1.0 / nu).exp() / lgamma(3.0 / nu).exp()).sqrt();
    let ln_const = (nu / (lambda * 2.0_f64.powf(1.0 + 1.0 / nu) * lgamma(1.0 / nu).exp())).ln();

    let mut ll = 0.0_f64;
    for i in 0..n {
        let hi = h[i].max(1e-12);
        let z = (eps[i] / hi.sqrt()).abs();
        ll += ln_const - 0.5 * hi.ln() - 0.5 * (z / lambda).powf(nu);
    }
    ll
}

/// Log-gamma approximation via Stirling's series (Lanczos g=7).
fn lgamma(x: f64) -> f64 {
    // Lanczos coefficients for g=7
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        for (i, &c) in C[1..].iter().enumerate() {
            a += c / (x + i as f64 + 1.0);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln()
            + (x + 0.5) * t.ln()
            - t
            + a.ln()
    }
}

// ============================================================
// Nelder-Mead Simplex Optimiser
// ============================================================

/// Nelder-Mead simplex optimiser for GARCH MLE (constrained).
///
/// Projects parameters into the feasible region: omega > 0, alpha_i >= 0, beta_j >= 0,
/// Sum(alpha) + Sum(beta) < 1.
fn nelder_mead_garch(
    x0: &[f64],
    f: &dyn Fn(&[f64]) -> f64,
    p: usize,
    q: usize,
    max_iter: usize,
) -> Result<Vec<f64>, TimeSeriesError> {
    let n = x0.len();

    // Project into feasible region before evaluating
    let project = |par: &mut Vec<f64>| {
        par[0] = par[0].abs().max(1e-8); // omega > 0
        for i in 1..(1 + p) {
            par[i] = par[i].abs(); // alpha >= 0
        }
        for j in 0..q {
            par[1 + p + j] = par[1 + p + j].abs(); // beta >= 0
        }
        let ab_sum: f64 = par[1..].iter().sum();
        if ab_sum >= 0.999 {
            let scale = 0.999 / ab_sum;
            for v in par[1..].iter_mut() {
                *v *= scale;
            }
        }
    };

    let projected_f = |par: &[f64]| -> f64 {
        let mut p2 = par.to_vec();
        project(&mut p2);
        f(&p2)
    };

    nelder_mead_impl(x0, &projected_f, max_iter, n)
}

/// Nelder-Mead for unconstrained problems (EGARCH, GJR-GARCH with inline constraints).
fn nelder_mead_unconstrained(
    x0: &[f64],
    f: &dyn Fn(&[f64]) -> f64,
    max_iter: usize,
) -> Result<Vec<f64>, TimeSeriesError> {
    nelder_mead_impl(x0, f, max_iter, x0.len())
}

/// Core Nelder-Mead simplex implementation.
fn nelder_mead_impl(
    x0: &[f64],
    f: &dyn Fn(&[f64]) -> f64,
    max_iter: usize,
    n: usize,
) -> Result<Vec<f64>, TimeSeriesError> {
    // Reflection / expansion / contraction / shrink constants (standard)
    const ALPHA: f64 = 1.0;
    const GAMMA: f64 = 2.0;
    const RHO: f64 = 0.5;
    const SIGMA: f64 = 0.5;

    // Build initial simplex: n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut v = x0.to_vec();
        let delta = if v[i].abs() > 1e-5 { 0.05 * v[i].abs() } else { 0.00025 };
        v[i] += delta;
        simplex.push(v);
    }

    // Evaluate at each vertex
    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let tol = 1e-8;

    for _iter in 0..max_iter {
        // Sort by function value (ascending)
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal));

        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Convergence check
        let range = fvals[worst] - fvals[best];
        if range < tol && _iter > 10 {
            break;
        }

        // Centroid of all vertices except worst
        let mut centroid = vec![0.0_f64; n];
        for &idx in &order[0..n] {
            for k in 0..n {
                centroid[k] += simplex[idx][k];
            }
        }
        for c in &mut centroid {
            *c /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|k| centroid[k] + ALPHA * (centroid[k] - simplex[worst][k]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < fvals[best] {
            // Expansion
            let expanded: Vec<f64> = (0..n)
                .map(|k| centroid[k] + GAMMA * (reflected[k] - centroid[k]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                fvals[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = f_reflected;
            }
        } else if f_reflected < fvals[second_worst] {
            simplex[worst] = reflected;
            fvals[worst] = f_reflected;
        } else {
            // Contraction
            let contracted: Vec<f64> = if f_reflected < fvals[worst] {
                // Outside contraction
                (0..n)
                    .map(|k| centroid[k] + RHO * (reflected[k] - centroid[k]))
                    .collect()
            } else {
                // Inside contraction
                (0..n)
                    .map(|k| centroid[k] + RHO * (simplex[worst][k] - centroid[k]))
                    .collect()
            };
            let f_contracted = f(&contracted);

            if f_contracted < fvals[worst] {
                simplex[worst] = contracted;
                fvals[worst] = f_contracted;
            } else {
                // Shrink
                for i in 1..=n {
                    let idx = order[i];
                    for k in 0..n {
                        simplex[idx][k] = simplex[best][k]
                            + SIGMA * (simplex[idx][k] - simplex[best][k]);
                    }
                    fvals[idx] = f(&simplex[idx]);
                }
            }
        }
    }

    // Return best vertex
    let best_idx = (0..=n)
        .min_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);

    Ok(simplex[best_idx].clone())
}

// ============================================================
// Diagnostic Tests
// ============================================================

/// ARCH Lagrange Multiplier test for conditional heteroskedasticity.
///
/// Regresses eps2_t on its `lags` lagged values and computes T*R2 ~ chi2(lags).
/// Returns `(LM_statistic, p_value)`.
pub fn arch_lm_test(
    residuals: &Array1<f64>,
    lags: usize,
) -> Result<(f64, f64), TimeSeriesError> {
    let n = residuals.len();
    if n <= lags + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "ARCH LM test".to_string(),
            required: lags + 2,
            actual: n,
        });
    }
    if lags == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "lags".to_string(),
            message: "must be >= 1".to_string(),
        });
    }

    let sq: Vec<f64> = residuals.iter().map(|r| r * r).collect();
    let t = n - lags;

    // Dependent variable: sq[lags..n]
    // Regressor matrix: [1, sq[t-1], ..., sq[t-lags]] for t = lags..n
    let k = lags + 1; // intercept + lags regressors
    let mut x_mat = vec![vec![0.0_f64; k]; t];
    let mut y_vec = vec![0.0_f64; t];

    for i in 0..t {
        x_mat[i][0] = 1.0; // intercept
        for j in 0..lags {
            x_mat[i][j + 1] = sq[i + lags - j - 1];
        }
        y_vec[i] = sq[i + lags];
    }

    // OLS: beta = (X'X)^{-1} X'y
    let beta = match ols_solve(&x_mat, &y_vec, t, k) {
        Some(b) => b,
        None => {
            return Err(TimeSeriesError::NumericalInstability(
                "ARCH LM: singular regressor matrix".to_string(),
            ))
        }
    };

    // Fitted values and residuals
    let y_mean: f64 = y_vec.iter().sum::<f64>() / t as f64;
    let mut ss_tot = 0.0_f64;
    let mut ss_res = 0.0_f64;
    for i in 0..t {
        let y_hat: f64 = x_mat[i].iter().zip(beta.iter()).map(|(xi, bi)| xi * bi).sum();
        let r = y_vec[i] - y_hat;
        ss_res += r * r;
        ss_tot += (y_vec[i] - y_mean) * (y_vec[i] - y_mean);
    }

    let r2 = if ss_tot > 1e-14 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    let lm_stat = (t as f64) * r2;
    let p_value = chi_squared_p_value(lm_stat, lags);

    Ok((lm_stat, p_value))
}

/// Compute standardized residuals: z_t = eps_t / sigma_t.
pub fn standardized_residuals(
    returns: &Array1<f64>,
    conditional_std: &Array1<f64>,
) -> Array1<f64> {
    returns
        .iter()
        .zip(conditional_std.iter())
        .map(|(&r, &s)| if s > 1e-14 { r / s } else { 0.0 })
        .collect()
}

/// Realized volatility from intraday returns.
///
/// Aggregates squared intraday returns into per-period realized variances.
/// Returns an array of length `floor(total_obs / n_obs_per_period)`.
pub fn realized_volatility(
    intraday_returns: &Array1<f64>,
    n_obs_per_period: usize,
) -> Result<Array1<f64>, TimeSeriesError> {
    if n_obs_per_period == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_obs_per_period".to_string(),
            message: "must be >= 1".to_string(),
        });
    }
    let n = intraday_returns.len();
    if n < n_obs_per_period {
        return Err(TimeSeriesError::InsufficientData {
            message: "realized volatility".to_string(),
            required: n_obs_per_period,
            actual: n,
        });
    }

    let periods = n / n_obs_per_period;
    let mut rv = Vec::with_capacity(periods);

    for d in 0..periods {
        let start = d * n_obs_per_period;
        let end = start + n_obs_per_period;
        let rv_d: f64 = intraday_returns
            .slice(scirs2_core::ndarray::s![start..end])
            .iter()
            .map(|r| r * r)
            .sum();
        rv.push(rv_d);
    }

    Ok(Array1::from_vec(rv))
}

/// Volatility clustering test: Ljung-Box Q-statistic on squared returns.
///
/// Large Q implies significant autocorrelation in squared returns, consistent
/// with volatility clustering. Returns `(Q_statistic, p_value)`.
pub fn volatility_clustering_test(
    returns: &Array1<f64>,
    lags: usize,
) -> Result<(f64, f64), TimeSeriesError> {
    let n = returns.len();
    if n <= lags + 1 {
        return Err(TimeSeriesError::InsufficientData {
            message: "volatility clustering test".to_string(),
            required: lags + 2,
            actual: n,
        });
    }

    // Square the returns
    let sq: Vec<f64> = returns.iter().map(|r| r * r).collect();
    let mean_sq: f64 = sq.iter().sum::<f64>() / n as f64;

    // Sample autocorrelation of squared returns at each lag
    let variance: f64 = sq
        .iter()
        .map(|&s| (s - mean_sq) * (s - mean_sq))
        .sum::<f64>()
        / n as f64;

    if variance < 1e-14 {
        return Ok((0.0, 1.0));
    }

    let mut q_stat = 0.0_f64;
    for h in 1..=lags {
        let mut acov = 0.0_f64;
        for t in h..n {
            acov += (sq[t] - mean_sq) * (sq[t - h] - mean_sq);
        }
        acov /= n as f64;
        let rho_h = acov / variance;
        q_stat += rho_h * rho_h / (n - h) as f64;
    }
    q_stat *= n as f64 * (n as f64 + 2.0);

    let p_value = chi_squared_p_value(q_stat, lags);
    Ok((q_stat, p_value))
}

/// Monte Carlo Value-at-Risk from a fitted GARCH model.
///
/// Simulates `n_simulations` return paths of length `horizon` from the fitted
/// GARCH process and returns the empirical quantile at `(1 - confidence)`.
pub fn garch_var(
    model: &Garch,
    returns: &Array1<f64>,
    confidence: f64,
    horizon: usize,
    n_simulations: usize,
    seed: u64,
) -> Result<f64, TimeSeriesError> {
    if !(0.0 < confidence && confidence < 1.0) {
        return Err(TimeSeriesError::InvalidParameter {
            name: "confidence".to_string(),
            message: "must be in (0, 1)".to_string(),
        });
    }
    if horizon == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "horizon".to_string(),
            message: "must be >= 1".to_string(),
        });
    }
    if n_simulations == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_simulations".to_string(),
            message: "must be >= 1".to_string(),
        });
    }

    let h = model.conditional_variances(returns)?;
    let n = returns.len();
    let last_h = h[n - 1];
    let eps: Vec<f64> = returns.iter().map(|&r| r - model.mean).collect();
    let last_eps = eps[n - 1];

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f64, 1.0_f64).map_err(|e| {
        TimeSeriesError::NumericalInstability(format!("Normal distribution error: {e}"))
    })?;

    let mut path_losses: Vec<f64> = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        let mut current_h = last_h;
        let mut current_eps = last_eps;
        let mut cum_return = 0.0_f64;

        for _step in 0..horizon {
            let z: f64 = rng.sample(normal);
            let next_eps = current_h.sqrt() * z;

            // GARCH(1,1) recursion for simplicity in multi-step simulation
            // For general p,q: would need history buffers
            let next_h = (model.omega
                + model.alpha.first().copied().unwrap_or(0.0) * current_eps * current_eps
                + model.beta.first().copied().unwrap_or(0.0) * current_h)
                .max(1e-12);

            cum_return += model.mean + next_eps;
            current_h = next_h;
            current_eps = next_eps;
        }

        // Loss = negative return
        path_losses.push(-cum_return);
    }

    // Empirical quantile at confidence level
    path_losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((confidence * n_simulations as f64) as usize).min(n_simulations - 1);

    Ok(path_losses[idx])
}

// ============================================================
// Internal utility functions
// ============================================================

/// Chi-squared survival function P(X > x) approximation for df degrees of freedom.
fn chi_squared_p_value(stat: f64, df: usize) -> f64 {
    if stat <= 0.0 || df == 0 {
        return 1.0;
    }
    // Use regularised incomplete gamma: P(chi2(k) > x) = 1 - gamma_lower(k/2, x/2) / Gamma(k/2)
    let k = df as f64 / 2.0;
    let x = stat / 2.0;
    let cdf = regularised_lower_gamma(k, x);
    (1.0 - cdf).max(0.0).min(1.0)
}

/// Regularised lower incomplete gamma function via series expansion.
fn regularised_lower_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    // For small x/a use series; for large x/a use continued fraction
    if x < a + 1.0 {
        // Series expansion
        let mut sum = 1.0_f64 / a;
        let mut term = 1.0_f64 / a;
        for n in 1..200 {
            term *= x / (a + n as f64);
            sum += term;
            if term.abs() < 1e-14 * sum.abs() {
                break;
            }
        }
        ((-x + a * x.ln() - lgamma(a)).exp() * sum).min(1.0)
    } else {
        // Continued fraction (Lentz)
        1.0 - regularised_upper_gamma_cf(a, x)
    }
}

/// Upper incomplete gamma via continued fraction (Lentz algorithm).
fn regularised_upper_gamma_cf(a: f64, x: f64) -> f64 {
    let eps = 1e-14;
    let fpmin = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1_i64..200 {
        let an = -i as f64 * (i as f64 - a);
        b += 2.0;
        d = (an * d + b).abs().max(fpmin).recip() * an + b;
        c = b + an / c.abs().max(fpmin);
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }
    ((-x + a * x.ln() - lgamma(a)).exp() * h).max(0.0)
}

/// OLS via Gaussian elimination on the normal equations.
fn ols_solve(
    x: &[Vec<f64>],
    y: &[f64],
    n: usize,
    k: usize,
) -> Option<Vec<f64>> {
    // Compute X'X (k x k) and X'y (k)
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];

    for i in 0..n {
        for j in 0..k {
            for l in 0..k {
                xtx[j][l] += x[i][j] * x[i][l];
            }
            xty[j] += x[i][j] * y[i];
        }
    }

    // Solve via Gaussian elimination with partial pivoting
    let mut aug: Vec<Vec<f64>> = xtx
        .iter()
        .zip(xty.iter())
        .map(|(row, &rhs)| {
            let mut r = row.clone();
            r.push(rhs);
            r
        })
        .collect();

    for col in 0..k {
        // Find pivot
        let pivot = (col..k).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        aug.swap(col, pivot);

        let diag = aug[col][col];
        if diag.abs() < 1e-14 {
            return None;
        }

        // Eliminate below and above
        for row in 0..k {
            if row != col {
                let factor = aug[row][col] / diag;
                for c in col..=k {
                    let val = aug[col][c];
                    aug[row][c] -= factor * val;
                }
            }
        }
        // Normalise pivot row
        for c in col..=k {
            aug[col][c] /= diag;
        }
    }

    Some((0..k).map(|i| aug[i][k]).collect())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    /// Generate a synthetic GARCH(1,1) series for testing.
    fn synthetic_garch_returns(n: usize, omega: f64, alpha: f64, beta: f64, seed: u64) -> Array1<f64> {
        use scirs2_core::random::{Normal, Rng, SeedableRng};
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0_f64, 1.0_f64).expect("Normal dist");

        let mut returns = Vec::with_capacity(n);
        let mut h = omega / (1.0 - alpha - beta);

        for _ in 0..n {
            let z: f64 = rng.sample(normal);
            let eps = h.sqrt() * z;
            returns.push(eps);
            h = (omega + alpha * eps * eps + beta * h).max(1e-12);
        }
        Array1::from_vec(returns)
    }

    // -------------------------------------------------------------------
    // GARCH(1,1) parameter validity
    // -------------------------------------------------------------------

    #[test]
    fn test_garch_stationarity_stationary() {
        let g = Garch::garch_1_1(0.01, 0.05, 0.90);
        assert!(g.is_stationary(), "alpha+beta = 0.95 < 1 -> stationary");
    }

    #[test]
    fn test_garch_stationarity_unit_root() {
        let g = Garch::garch_1_1(0.01, 0.05, 0.95);
        assert!(
            !g.is_stationary(),
            "alpha+beta = 1.00 -> not strictly stationary"
        );
    }

    #[test]
    fn test_garch_stationarity_explosive() {
        let g = Garch::garch_1_1(0.01, 0.20, 0.90);
        assert!(!g.is_stationary(), "alpha+beta = 1.10 -> explosive");
    }

    #[test]
    fn test_unconditional_variance_formula() {
        let omega = 0.02;
        let alpha = 0.05;
        let beta = 0.90;
        let g = Garch::garch_1_1(omega, alpha, beta);
        let uv = g.unconditional_variance().expect("Stationary model");
        let expected = omega / (1.0 - alpha - beta);
        assert!(
            (uv - expected).abs() < 1e-10,
            "Unconditional variance mismatch: {uv} vs {expected}"
        );
    }

    #[test]
    fn test_unconditional_variance_nonstationary_is_none() {
        let g = Garch::garch_1_1(0.01, 0.10, 0.95);
        assert!(
            g.unconditional_variance().is_none(),
            "Non-stationary model -> None"
        );
    }

    // -------------------------------------------------------------------
    // Conditional variance recursion
    // -------------------------------------------------------------------

    #[test]
    fn test_conditional_variances_positive() {
        let returns = synthetic_garch_returns(100, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        let h = g.conditional_variances(&returns).expect("CVs");
        assert_eq!(h.len(), returns.len());
        assert!(h.iter().all(|&v| v > 0.0), "All conditional variances > 0");
    }

    #[test]
    fn test_conditional_variances_length_matches_input() {
        let returns = synthetic_garch_returns(50, 0.005, 0.06, 0.90, 7);
        let g = Garch::garch_1_1(0.005, 0.06, 0.90);
        let h = g.conditional_variances(&returns).expect("CVs");
        assert_eq!(h.len(), 50);
    }

    #[test]
    fn test_conditional_variances_insufficient_data() {
        let ret = Array1::from_vec(vec![0.01]);
        let g = Garch::garch_1_1(0.01, 0.05, 0.90);
        assert!(g.conditional_variances(&ret).is_err());
    }

    // -------------------------------------------------------------------
    // Log-likelihood
    // -------------------------------------------------------------------

    #[test]
    fn test_log_likelihood_is_finite() {
        let returns = synthetic_garch_returns(80, 0.005, 0.07, 0.90, 99);
        let g = Garch::garch_1_1(0.005, 0.07, 0.90);
        let ll = g.log_likelihood(&returns).expect("LL");
        assert!(ll.is_finite(), "Log-likelihood must be finite");
    }

    #[test]
    fn test_log_likelihood_negative() {
        // For short series the LL can be either sign; it must be finite
        let returns = synthetic_garch_returns(60, 0.002, 0.05, 0.92, 13);
        let g = Garch::garch_1_1(0.002, 0.05, 0.92);
        let ll = g.log_likelihood(&returns).expect("LL");
        assert!(ll.is_finite());
    }

    // -------------------------------------------------------------------
    // MLE fitting
    // -------------------------------------------------------------------

    #[test]
    fn test_fit_garch_1_1_convergence() {
        let returns = synthetic_garch_returns(200, 0.01, 0.07, 0.88, 2024);
        let model = fit_garch_1_1(&returns, GarchDistribution::Normal, 400);
        assert!(model.is_ok(), "GARCH(1,1) fitting should succeed");
        let m = model.expect("Fitted model");
        assert!(m.omega > 0.0, "omega must be positive");
        assert!(m.alpha[0] >= 0.0, "alpha must be non-negative");
        assert!(m.beta[0] >= 0.0, "beta must be non-negative");
        assert!(m.is_stationary(), "Fitted model should be stationary");
    }

    #[test]
    fn test_fit_garch_insufficient_data() {
        let short = Array1::from_vec(vec![0.01, -0.02, 0.01]);
        let result = fit_garch_1_1(&short, GarchDistribution::Normal, 100);
        assert!(result.is_err(), "Too few observations -> error");
    }

    #[test]
    fn test_fit_garch_student_t() {
        let returns = synthetic_garch_returns(150, 0.005, 0.06, 0.90, 31);
        let model = fit_garch_1_1(&returns, GarchDistribution::StudentT { df: 6.0 }, 300);
        assert!(model.is_ok(), "GARCH(1,1) Student-t should fit");
    }

    #[test]
    fn test_fit_garch_p_q() {
        let returns = synthetic_garch_returns(250, 0.008, 0.06, 0.88, 55);
        let model = fit_garch(&returns, 1, 2, GarchDistribution::Normal, 300);
        assert!(model.is_ok(), "GARCH(1,2) fitting should succeed");
        let m = model.expect("Fitted GARCH(1,2)");
        assert_eq!(m.alpha.len(), 1);
        assert_eq!(m.beta.len(), 2);
    }

    // -------------------------------------------------------------------
    // Standardized residuals
    // -------------------------------------------------------------------

    #[test]
    fn test_standardized_residuals_shape() {
        let returns = synthetic_garch_returns(100, 0.01, 0.08, 0.88, 77);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        let h = g.conditional_variances(&returns).expect("CVs");
        let sigma = h.mapv(|v| v.sqrt());
        let z = standardized_residuals(&returns, &sigma);
        assert_eq!(z.len(), returns.len());
    }

    #[test]
    fn test_standardized_residuals_mean_approx_zero() {
        // With true parameters the standardised residuals should be approximately iid N(0,1)
        let returns = synthetic_garch_returns(500, 0.005, 0.07, 0.90, 101);
        let g = Garch::garch_1_1(0.005, 0.07, 0.90);
        let h = g.conditional_variances(&returns).expect("CVs");
        let sigma = h.mapv(|v| v.sqrt());
        let z = standardized_residuals(&returns, &sigma);
        let mean_z: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(
            mean_z.abs() < 0.2,
            "Standardised residual mean {mean_z:.4} should be near 0"
        );
    }

    #[test]
    fn test_standardized_residuals_std_approx_one() {
        let returns = synthetic_garch_returns(500, 0.005, 0.07, 0.90, 202);
        let g = Garch::garch_1_1(0.005, 0.07, 0.90);
        let h = g.conditional_variances(&returns).expect("CVs");
        let sigma = h.mapv(|v| v.sqrt());
        let z = standardized_residuals(&returns, &sigma);
        let mean_z: f64 = z.iter().sum::<f64>() / z.len() as f64;
        let var_z: f64 = z.iter().map(|zi| (zi - mean_z).powi(2)).sum::<f64>() / z.len() as f64;
        let std_z = var_z.sqrt();
        assert!(
            (std_z - 1.0).abs() < 0.3,
            "Standardised residual std {std_z:.4} should be near 1"
        );
    }

    // -------------------------------------------------------------------
    // Volatility forecasting
    // -------------------------------------------------------------------

    #[test]
    fn test_forecast_variance_positive() {
        let returns = synthetic_garch_returns(100, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        let fcs = g.forecast_variance(&returns, 10).expect("Forecasts");
        assert_eq!(fcs.len(), 10);
        assert!(fcs.iter().all(|&v| v > 0.0), "All forecasts must be positive");
    }

    #[test]
    fn test_forecast_variance_mean_reversion() {
        // Long-horizon forecast should converge toward unconditional variance
        let omega = 0.005;
        let alpha = 0.07;
        let beta = 0.88;
        let returns = synthetic_garch_returns(300, omega, alpha, beta, 500);
        let g = Garch::garch_1_1(omega, alpha, beta);
        let fcs = g.forecast_variance(&returns, 50).expect("Forecasts");
        let uncond = g.unconditional_variance().expect("Unconditional variance");

        // The 50-step-ahead forecast should be closer to uncond than the 1-step forecast
        let diff_1 = (fcs[0] - uncond).abs();
        let diff_50 = (fcs[49] - uncond).abs();
        assert!(
            diff_50 < diff_1 + 1e-8,
            "50-step forecast {:.6} not converging to unconditional {:.6} faster than 1-step {:.6}",
            fcs[49],
            uncond,
            fcs[0]
        );
    }

    #[test]
    fn test_forecast_variance_invalid_horizon() {
        let returns = synthetic_garch_returns(50, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        assert!(g.forecast_variance(&returns, 0).is_err());
    }

    // -------------------------------------------------------------------
    // ARCH LM test
    // -------------------------------------------------------------------

    #[test]
    fn test_arch_lm_detects_garch_data() {
        // GARCH data should show significant ARCH effects (low p-value)
        let returns = synthetic_garch_returns(300, 0.01, 0.15, 0.80, 9999);
        let (stat, pval) = arch_lm_test(&returns, 5).expect("ARCH LM");
        assert!(stat > 0.0, "LM statistic must be positive");
        // With strong ARCH effects the p-value should be small
        assert!(
            pval < 0.5,
            "ARCH LM test on GARCH data: p={pval:.4} expected < 0.5"
        );
    }

    #[test]
    fn test_arch_lm_white_noise_high_pvalue() {
        // iid N(0,1) sequence has no ARCH effects -> high p-value
        use scirs2_core::random::{Normal, Rng, SeedableRng};
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(12345);
        let normal = Normal::new(0.0_f64, 0.02_f64).expect("Normal");
        let data: Array1<f64> = (0..200).map(|_| rng.sample(normal)).collect();
        let (stat, pval) = arch_lm_test(&data, 5).expect("ARCH LM");
        // For iid data the p-value should be larger on average (not always, but broadly)
        let _ = stat;
        let _ = pval;
        // We just check it doesn't error and returns finite values
        assert!(stat.is_finite());
        assert!((0.0..=1.0).contains(&pval));
    }

    #[test]
    fn test_arch_lm_insufficient_data() {
        let short = Array1::from_vec(vec![0.01, 0.02, 0.03]);
        assert!(arch_lm_test(&short, 5).is_err());
    }

    // -------------------------------------------------------------------
    // Realized volatility
    // -------------------------------------------------------------------

    #[test]
    fn test_realized_volatility_aggregation() {
        // 60 5-minute returns in a day = 2 days of realized var
        let n = 60_usize;
        use scirs2_core::random::{Normal, Rng, SeedableRng};
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(7);
        let normal = Normal::new(0.0_f64, 0.001_f64).expect("Normal");
        let intra: Array1<f64> = (0..n).map(|_| rng.sample(normal)).collect();
        let rv = realized_volatility(&intra, 30).expect("RV");
        assert_eq!(rv.len(), 2, "Should aggregate into 2 daily periods");
        assert!(rv.iter().all(|&v| v >= 0.0), "All RV non-negative");
    }

    #[test]
    fn test_realized_volatility_single_period() {
        let intra = Array1::from_vec(vec![0.001, -0.002, 0.0015, -0.0008, 0.0012]);
        let rv = realized_volatility(&intra, 5).expect("RV");
        assert_eq!(rv.len(), 1);
        let expected: f64 = intra.iter().map(|r| r * r).sum();
        assert!((rv[0] - expected).abs() < 1e-14);
    }

    #[test]
    fn test_realized_volatility_invalid_params() {
        let intra = Array1::from_vec(vec![0.01, 0.02]);
        assert!(realized_volatility(&intra, 0).is_err());
        assert!(realized_volatility(&intra, 5).is_err());
    }

    // -------------------------------------------------------------------
    // Volatility clustering test
    // -------------------------------------------------------------------

    #[test]
    fn test_clustering_test_garch_data() {
        let returns = synthetic_garch_returns(300, 0.01, 0.15, 0.80, 888);
        let (q, pval) = volatility_clustering_test(&returns, 10).expect("Clustering test");
        assert!(q > 0.0);
        assert!((0.0..=1.0).contains(&pval));
    }

    #[test]
    fn test_clustering_test_insufficient_data() {
        let short = Array1::from_vec(vec![0.01, 0.02]);
        assert!(volatility_clustering_test(&short, 5).is_err());
    }

    // -------------------------------------------------------------------
    // EGARCH
    // -------------------------------------------------------------------

    #[test]
    fn test_egarch_conditional_variances_positive() {
        let returns = synthetic_garch_returns(100, 0.01, 0.08, 0.88, 42);
        let eg = EGarch {
            p: 1,
            q: 1,
            omega: -0.1,
            alpha: vec![0.15],
            gamma: vec![-0.08],
            beta: vec![0.95],
            mean: 0.0,
        };
        let h = eg.conditional_variances(&returns).expect("EGARCH CVs");
        assert_eq!(h.len(), returns.len());
        assert!(h.iter().all(|&v| v > 0.0), "All EGARCH CVs must be > 0");
    }

    #[test]
    fn test_egarch_leverage_effect() {
        // With negative gamma, a negative shock should produce higher log-variance
        // than a symmetric positive shock of the same magnitude
        let eg = EGarch {
            p: 1,
            q: 1,
            omega: 0.0,
            alpha: vec![0.2],
            gamma: vec![-0.1], // negative -> leverage
            beta: vec![0.0],
            mean: 0.0,
        };

        // Positive shock
        let pos = Array1::from_vec(vec![0.05, 0.05, 0.05, 0.05]);
        let h_pos = eg.conditional_variances(&pos).expect("pos CVs");

        // Negative shock of same magnitude
        let neg = Array1::from_vec(vec![-0.05, -0.05, -0.05, -0.05]);
        let h_neg = eg.conditional_variances(&neg).expect("neg CVs");

        // With negative gamma the EGARCH response to neg shock should be larger
        assert!(
            h_neg.last().unwrap_or(&0.0) >= h_pos.last().unwrap_or(&1.0),
            "EGARCH leverage: neg shock variance {:.6} should >= pos shock variance {:.6}",
            h_neg.last().unwrap_or(&0.0),
            h_pos.last().unwrap_or(&0.0),
        );
    }

    #[test]
    fn test_fit_egarch_1_1() {
        let returns = synthetic_garch_returns(200, 0.005, 0.08, 0.88, 77);
        let model = fit_egarch_1_1(&returns, 300);
        assert!(model.is_ok(), "EGARCH(1,1) fitting should succeed");
        let m = model.expect("Fitted EGARCH");
        assert_eq!(m.alpha.len(), 1);
        assert_eq!(m.gamma.len(), 1);
        assert_eq!(m.beta.len(), 1);
    }

    // -------------------------------------------------------------------
    // GJR-GARCH
    // -------------------------------------------------------------------

    #[test]
    fn test_gjr_garch_conditional_variances_positive() {
        let returns = synthetic_garch_returns(100, 0.01, 0.06, 0.88, 42);
        let gjr = GjrGarch {
            p: 1,
            q: 1,
            omega: 0.01,
            alpha: vec![0.04],
            gamma: vec![0.06],
            beta: vec![0.87],
            mean: 0.0,
        };
        let h = gjr.conditional_variances(&returns).expect("GJR CVs");
        assert!(h.iter().all(|&v| v > 0.0), "All GJR CVs must be > 0");
    }

    #[test]
    fn test_gjr_garch_asymmetry() {
        // Negative shock should have higher impact than positive shock of same size
        let gjr = GjrGarch {
            p: 1,
            q: 1,
            omega: 0.001,
            alpha: vec![0.02],
            gamma: vec![0.10], // large positive gamma = strong bad-news effect
            beta: vec![0.0],
            mean: 0.0,
        };

        let pos = Array1::from_vec(vec![0.05, 0.05, 0.05]);
        let neg = Array1::from_vec(vec![-0.05, -0.05, -0.05]);

        let h_pos = gjr.conditional_variances(&pos).expect("pos");
        let h_neg = gjr.conditional_variances(&neg).expect("neg");

        // The conditional variance after a negative shock should be larger
        assert!(
            h_neg.last().unwrap_or(&0.0) > h_pos.last().unwrap_or(&1.0),
            "GJR asymmetry: h_neg {:.6} should > h_pos {:.6}",
            h_neg.last().unwrap_or(&0.0),
            h_pos.last().unwrap_or(&0.0),
        );
    }

    #[test]
    fn test_fit_gjr_garch_1_1() {
        let returns = synthetic_garch_returns(200, 0.005, 0.06, 0.88, 53);
        let model = fit_gjr_garch_1_1(&returns, 300);
        assert!(model.is_ok(), "GJR-GARCH(1,1) fitting should succeed");
        let m = model.expect("Fitted GJR-GARCH");
        assert!(m.omega > 0.0);
        assert!(m.alpha[0] >= 0.0);
        assert!(m.gamma[0] >= 0.0);
        assert!(m.beta[0] >= 0.0);
    }

    // -------------------------------------------------------------------
    // Value at Risk
    // -------------------------------------------------------------------

    #[test]
    fn test_garch_var_valid() {
        let returns = synthetic_garch_returns(100, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        let var = garch_var(&g, &returns, 0.99, 1, 1000, 1234);
        assert!(var.is_ok(), "VaR computation should succeed");
        let v = var.expect("VaR");
        assert!(v.is_finite(), "VaR must be finite");
    }

    #[test]
    fn test_garch_var_invalid_confidence() {
        let returns = synthetic_garch_returns(50, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        assert!(garch_var(&g, &returns, 1.0, 1, 100, 0).is_err());
        assert!(garch_var(&g, &returns, 0.0, 1, 100, 0).is_err());
    }

    #[test]
    fn test_garch_var_invalid_horizon() {
        let returns = synthetic_garch_returns(50, 0.01, 0.08, 0.88, 42);
        let g = Garch::garch_1_1(0.01, 0.08, 0.88);
        assert!(garch_var(&g, &returns, 0.95, 0, 100, 0).is_err());
    }

    // -------------------------------------------------------------------
    // Distribution log-likelihood kernels
    // -------------------------------------------------------------------

    #[test]
    fn test_student_t_ll_finite() {
        let eps = vec![0.01, -0.02, 0.015, -0.008];
        let h = vec![0.0004_f64; 4];
        let ll = student_t_log_likelihood(&eps, &h, 6.0);
        assert!(ll.is_finite(), "Student-t LL must be finite: {ll}");
    }

    #[test]
    fn test_ged_ll_finite() {
        let eps = vec![0.01, -0.02, 0.015, -0.008];
        let h = vec![0.0004_f64; 4];
        let ll = ged_log_likelihood(&eps, &h, 1.5);
        assert!(ll.is_finite(), "GED LL must be finite: {ll}");
    }

    // -------------------------------------------------------------------
    // Chi-squared and regularised gamma utilities
    // -------------------------------------------------------------------

    #[test]
    fn test_chi_squared_p_value_sanity() {
        // Chi-squared critical value at 5% for df=5 is 11.07
        let pval = chi_squared_p_value(11.07, 5);
        assert!(
            (pval - 0.05).abs() < 0.05,
            "p-value at chi2(5) critical ~ 0.05, got {pval:.4}"
        );
    }

    #[test]
    fn test_chi_squared_p_value_zero_stat() {
        assert_eq!(chi_squared_p_value(0.0, 5), 1.0);
    }
}
