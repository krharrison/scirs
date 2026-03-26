//! FIGARCH (Fractionally Integrated GARCH) model — Baillie, Bollerslev & Mikkelsen (1996)
//!
//! The FIGARCH model captures the long-memory property of financial volatility
//! by using a fractional differencing operator `(1-L)^d` applied to the squared
//! innovation process.  This is intermediate between GARCH (`d=0`, short memory)
//! and IGARCH (`d=1`, unit-root).
//!
//! # Model Specification
//!
//! The FIGARCH(p,d,q) conditional variance equation is:
//!
//! ```text
//! σ²_t = ω/(1-β(L)) + [1 - β(L)^{-1} φ(L) (1-L)^d] ε²_t
//! ```
//!
//! where:
//! - `d ∈ (0,1)` — fractional integration order
//! - `φ(L)` — ARCH polynomial of order p
//! - `β(L)` — GARCH polynomial of order q
//!
//! For practical fitting we use the truncated infinite-order AR representation:
//!
//! ```text
//! σ²_t = ω + Σₖ₌₁ᴹ λₖ(d,φ,β) ε²_{t-k}  +  Σⱼ₌₁ᵍ βⱼ σ²_{t-j}
//! ```
//!
//! with truncation order `M` (typically 100–1000 for practical computation).
//! The `λₖ` weights are computed via the recursive formula of BBM (1996).
//!
//! # References
//! - Baillie, R. T., Bollerslev, T., & Mikkelsen, H. O. (1996).
//!   Fractionally integrated generalized autoregressive conditional heteroskedasticity.
//!   *Journal of Econometrics*, 74(1), 3–30.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_series::volatility::figarch::{FIGARCHModel, fit_figarch};
//! use scirs2_core::ndarray::Array1;
//!
//! let returns: Array1<f64> = Array1::from(vec![
//!     0.01, -0.02, 0.015, -0.008, 0.012, 0.005, -0.018, 0.009,
//!     -0.003, 0.007, 0.025, -0.014, 0.008, -0.006, 0.011, -0.019,
//!     0.022, 0.003, -0.011, 0.017, -0.004, 0.008, -0.013, 0.019,
//!     0.001, -0.009, 0.016, -0.002, 0.011, 0.006, -0.017, 0.013,
//!     0.004, -0.021, 0.009, 0.014, -0.007, 0.018, -0.005, 0.023,
//!     -0.012, 0.006, 0.010, -0.015, 0.020, -0.003, 0.008, -0.011,
//!     0.016, 0.002, -0.008, 0.019, -0.006, 0.013, 0.007, -0.014,
//! ]);
//! let model = fit_figarch(&returns, 1, 0.4, 1, 50).expect("FIGARCH fitting failed");
//! println!("d={:.4}, omega={:.6}", model.d, model.omega);
//! ```

use scirs2_core::ndarray::Array1;

use crate::error::{Result, TimeSeriesError};
use crate::volatility::egarch::nelder_mead;

// ============================================================
// FIGARCH model struct
// ============================================================

/// FIGARCH(p,d,q) model parameters
///
/// The fractional integration parameter `d ∈ (0,1)` controls the long-memory
/// property of volatility.
#[derive(Debug, Clone)]
pub struct FIGARCHModel {
    /// ARCH order p
    pub p: usize,
    /// Fractional integration order d ∈ (0,1)
    pub d: f64,
    /// GARCH order q
    pub q: usize,
    /// Constant ω > 0
    pub omega: f64,
    /// ARCH-like coefficients φᵢ (length p)
    pub phi: Vec<f64>,
    /// GARCH persistence coefficients βⱼ (length q)
    pub beta: Vec<f64>,
    /// Truncation order M for the infinite-order AR representation
    pub truncation: usize,
    /// Log-likelihood at the fitted parameters
    pub log_likelihood: f64,
    /// Number of observations used in fitting
    pub n_obs: usize,
}

impl FIGARCHModel {
    /// Construct a new FIGARCH model (with validation).
    pub fn new(
        p: usize,
        d: f64,
        q: usize,
        omega: f64,
        phi: Vec<f64>,
        beta: Vec<f64>,
        truncation: usize,
    ) -> Result<Self> {
        if !(0.0 < d && d < 1.0) {
            return Err(TimeSeriesError::InvalidModel(format!(
                "FIGARCH: d must be in (0,1), got d={d:.4}"
            )));
        }
        if omega <= 0.0 {
            return Err(TimeSeriesError::InvalidModel(
                "FIGARCH: ω must be positive".into(),
            ));
        }
        if phi.len() != p {
            return Err(TimeSeriesError::InvalidModel(format!(
                "FIGARCH: phi length {} != p={}",
                phi.len(),
                p
            )));
        }
        if beta.len() != q {
            return Err(TimeSeriesError::InvalidModel(format!(
                "FIGARCH: beta length {} != q={}",
                beta.len(),
                q
            )));
        }
        if truncation < 1 {
            return Err(TimeSeriesError::InvalidModel(
                "FIGARCH: truncation must be >= 1".into(),
            ));
        }
        Ok(Self {
            p,
            d,
            q,
            omega,
            phi,
            beta,
            truncation,
            log_likelihood: f64::NEG_INFINITY,
            n_obs: 0,
        })
    }

    /// Compute the long-memory weight sequence `λₖ` for `k = 1..M`.
    ///
    /// The weights are derived from the expansion of `(1-L)^d φ(L)`:
    ///
    /// `π_k^d = d * (k-1-d) / k * π_{k-1}^d`  with  `π_1^d = d`
    ///
    /// Then `λ_k = π_k^d - β * λ_{k-1}` (for FIGARCH(1,d,1) simplification).
    pub fn lambda_weights(&self) -> Vec<f64> {
        let m = self.truncation;
        let d = self.d;

        // Fractional differencing weights: (1-L)^d coefficients
        let mut pi = vec![0.0_f64; m + 1];
        pi[0] = 1.0;
        for k in 1..=m {
            pi[k] = -pi[k - 1] * (d - (k as f64 - 1.0)) / k as f64;
        }

        // Compose with ARCH polynomial φ(L): (1-L)^d (1 - φ(L))
        // For brevity: λ_k = Σ_{j=0}^{p} φ_j * π_{k-j}^d
        //   where φ_0 = 1, φ_j = -phi[j-1]
        let mut lambda = vec![0.0_f64; m + 1];
        for k in 1..=m {
            let mut lk = pi[k]; // contribution from φ_0 = 1
            for (j, &phij) in self.phi.iter().enumerate() {
                let lag = j + 1;
                if k > lag {
                    lk += (-phij) * pi[k - lag];
                }
            }
            // Subtract β-weighted past lambda (for GARCH part)
            for (j, &bj) in self.beta.iter().enumerate() {
                let lag = j + 1;
                if k > lag {
                    lk -= bj * lambda[k - lag];
                }
            }
            lambda[k] = lk;
        }

        // Return λ_{1..M} (positive elements only — negative values truncated to 0
        // following Baillie-Bollerslev-Mikkelsen convention)
        lambda[1..].iter().map(|&v| v.max(0.0)).collect()
    }
}

// ============================================================
// Conditional variance recursion
// ============================================================

/// Compute the conditional variance series `σ²_t` for a FIGARCH model using
/// the truncated infinite-order AR representation.
pub fn figarch_variance(
    returns: &Array1<f64>,
    omega: f64,
    lambda: &[f64],
    beta: &[f64],
) -> Result<Vec<f64>> {
    let n = returns.len();
    let m = lambda.len();
    let q = beta.len();
    let burn_in = m.max(q);

    if n < burn_in + 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "FIGARCH: too few observations for truncation order M".into(),
            required: burn_in + 2,
            actual: n,
        });
    }

    let mean = returns.mean().unwrap_or(0.0);
    let sample_var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let init_var = sample_var.max(omega);

    let mut sigma2 = vec![init_var; n];
    let eps2: Vec<f64> = returns.iter().map(|&r| (r - mean).powi(2)).collect();

    for t in 1..n {
        let mut var_t = omega;

        // Long-memory ARCH terms: Σ λₖ ε²_{t-k}
        for (k, &lk) in lambda.iter().enumerate() {
            let lag = k + 1;
            let e2 = if t >= lag { eps2[t - lag] } else { init_var };
            var_t += lk * e2;
        }

        // GARCH terms: Σ βⱼ σ²_{t-j}
        for (j, &bj) in beta.iter().enumerate() {
            let lag = j + 1;
            let s2 = if t >= lag { sigma2[t - lag] } else { init_var };
            var_t += bj * s2;
        }

        sigma2[t] = var_t.max(1e-15);
    }

    Ok(sigma2)
}

// ============================================================
// Log-likelihood
// ============================================================

/// Gaussian log-likelihood for FIGARCH.
///
/// Returns `(log_likelihood, sigma2_series)`.
pub fn figarch_log_likelihood(
    returns: &Array1<f64>,
    omega: f64,
    lambda: &[f64],
    beta: &[f64],
) -> Result<(f64, Vec<f64>)> {
    let sigma2 = figarch_variance(returns, omega, lambda, beta)?;
    let n = returns.len();
    let burn_in = lambda.len().max(beta.len()).max(1);
    let mean = returns.mean().unwrap_or(0.0);

    let mut ll = 0.0_f64;
    for t in burn_in..n {
        let s2 = sigma2[t];
        if s2 <= 0.0 || !s2.is_finite() {
            return Err(TimeSeriesError::NumericalInstability(
                "FIGARCH: non-positive conditional variance".into(),
            ));
        }
        let eps = returns[t] - mean;
        ll += -0.5 * (std::f64::consts::TAU.ln() + s2.ln() + eps * eps / s2);
    }

    if !ll.is_finite() {
        return Err(TimeSeriesError::NumericalInstability(
            "FIGARCH: log-likelihood not finite".into(),
        ));
    }

    Ok((ll, sigma2))
}

// ============================================================
// Fitting
// ============================================================

/// Fit a FIGARCH(p,d,q) model via Nelder-Mead maximisation of the Gaussian log-likelihood.
///
/// # Arguments
/// * `returns` — observed return series
/// * `p` — ARCH order
/// * `d_init` — initial guess for fractional integration parameter (0 < d < 1)
/// * `q` — GARCH order
/// * `truncation` — number of lags M for the truncated AR(∞) approximation
///
/// # Notes
/// FIGARCH fitting is computationally intensive due to the long-lag weight evaluation.
/// For exploratory use, `truncation = 50` is often sufficient; for accurate estimation
/// use `truncation ≥ 100`.
pub fn fit_figarch(
    returns: &Array1<f64>,
    p: usize,
    d_init: f64,
    q: usize,
    truncation: usize,
) -> Result<FIGARCHModel> {
    let n = returns.len();
    // Need at least truncation + some burn-in
    let min_obs = truncation + (p.max(q) + 1) * 3;
    if n < min_obs {
        return Err(TimeSeriesError::InsufficientData {
            message: format!(
                "FIGARCH: need at least truncation + extra = {} observations",
                min_obs
            ),
            required: min_obs,
            actual: n,
        });
    }

    let d_init = d_init.clamp(0.05, 0.95);

    // Parameter layout: [d, omega, phi_0..p-1, beta_0..q-1]
    let n_params = 2 + p + q;
    let mean = returns.mean().unwrap_or(0.0);
    let sample_var = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

    let mut theta = vec![0.0_f64; n_params];
    theta[0] = d_init;
    theta[1] = sample_var * 0.05; // omega
    for i in 0..p {
        theta[2 + i] = 0.05; // phi
    }
    for j in 0..q {
        theta[2 + p + j] = 0.80 / q as f64; // beta
    }

    let objective = |th: &[f64]| -> f64 {
        let d = th[0];
        if !(0.01..0.99).contains(&d) {
            return 1e15;
        }
        let omega = th[1];
        if omega <= 0.0 {
            return 1e15;
        }
        let phi: Vec<f64> = th[2..2 + p].to_vec();
        let beta: Vec<f64> = th[2 + p..].to_vec();

        if beta.iter().any(|&b| b < 0.0 || b >= 1.0) {
            return 1e15;
        }

        // Build a temporary model to compute lambda weights
        let tmp = match FIGARCHModel::new(p, d, q, omega, phi.clone(), beta.clone(), truncation) {
            Ok(m) => m,
            Err(_) => return 1e15,
        };
        let lambda = tmp.lambda_weights();

        match figarch_log_likelihood(returns, omega, &lambda, &beta) {
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

    let best = nelder_mead(&theta, &objective, 2000, 1e-7)?;

    let d_fit = best[0].clamp(0.01, 0.99);
    let omega = best[1].max(1e-10);
    let phi: Vec<f64> = best[2..2 + p].to_vec();
    let beta: Vec<f64> = best[2 + p..].iter().map(|&v| v.max(0.0)).collect();

    let mut model = FIGARCHModel::new(p, d_fit, q, omega, phi, beta, truncation)?;
    let lambda = model.lambda_weights();
    let (ll, _) = figarch_log_likelihood(returns, omega, &lambda, &model.beta)?;
    model.log_likelihood = ll;
    model.n_obs = n;
    Ok(model)
}

// ============================================================
// Conditional volatility and residuals
// ============================================================

/// Compute the conditional volatility series (σ_t) from a FIGARCH model.
pub fn figarch_conditional_volatility(
    model: &FIGARCHModel,
    returns: &Array1<f64>,
) -> Result<Vec<f64>> {
    let lambda = model.lambda_weights();
    let sigma2 = figarch_variance(returns, model.omega, &lambda, &model.beta)?;
    Ok(sigma2.into_iter().map(|v| v.sqrt()).collect())
}

/// Compute standardised residuals from a FIGARCH model.
pub fn figarch_standardised_residuals(
    model: &FIGARCHModel,
    returns: &Array1<f64>,
) -> Result<Vec<f64>> {
    let mean = returns.mean().unwrap_or(0.0);
    let lambda = model.lambda_weights();
    let sigma2 = figarch_variance(returns, model.omega, &lambda, &model.beta)?;
    let z: Vec<f64> = returns
        .iter()
        .zip(sigma2.iter())
        .map(|(&r, &s2)| (r - mean) / s2.sqrt().max(1e-12))
        .collect();
    Ok(z)
}

// ============================================================
// Long-memory diagnostics
// ============================================================

/// Estimate the fractional integration parameter `d` using the
/// log-periodogram (GPH) estimator of Geweke & Porter-Hudak (1983).
///
/// The estimator regresses `log(I(ω_j))` on `log(4 sin²(ω_j/2))`
/// for the low-frequency periodogram ordinates `j = 1..m`.
///
/// # Arguments
/// * `series` — observed time series (e.g., absolute returns, log(RV))
/// * `m` — number of periodogram ordinates to use (typically `n^0.5`)
///
/// Returns `(d_hat, se_d)` — estimate and asymptotic standard error.
pub fn gph_estimator(series: &Array1<f64>, m: usize) -> Result<(f64, f64)> {
    let n = series.len();
    if n < 10 {
        return Err(TimeSeriesError::InsufficientData {
            message: "GPH estimator needs at least 10 observations".into(),
            required: 10,
            actual: n,
        });
    }
    let m = m.min(n / 2 - 1).max(2);

    let mean = series.mean().unwrap_or(0.0);
    let x: Vec<f64> = series.iter().map(|&v| v - mean).collect();

    // Compute periodogram via DFT at Fourier frequencies ω_j = 2πj/n
    let mut periodogram = Vec::with_capacity(m);
    for j in 1..=m {
        let omega = 2.0 * std::f64::consts::PI * j as f64 / n as f64;
        let (re, im) = x
            .iter()
            .enumerate()
            .fold((0.0_f64, 0.0_f64), |(re, im), (t, &xt)| {
                let angle = omega * t as f64;
                (re + xt * angle.cos(), im - xt * angle.sin())
            });
        let i_omega = (re * re + im * im) / n as f64;
        periodogram.push(i_omega.max(1e-30).ln());
    }

    // Regressor: log(4 sin²(ω_j/2))
    let regressors: Vec<f64> = (1..=m)
        .map(|j| {
            let omega = std::f64::consts::PI * j as f64 / n as f64;
            let sin_val = omega.sin();
            (4.0 * sin_val * sin_val).max(1e-30).ln()
        })
        .collect();

    // OLS: y = a + b*x  →  d = -b/2
    let m_f = m as f64;
    let x_mean: f64 = regressors.iter().sum::<f64>() / m_f;
    let y_mean: f64 = periodogram.iter().sum::<f64>() / m_f;

    let sxx: f64 = regressors.iter().map(|&x| (x - x_mean).powi(2)).sum();
    let sxy: f64 = regressors
        .iter()
        .zip(periodogram.iter())
        .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
        .sum();

    if sxx.abs() < 1e-15 {
        return Err(TimeSeriesError::NumericalInstability(
            "GPH: degenerate regressor matrix (all ω_j equal?)".into(),
        ));
    }

    let b_hat = sxy / sxx;
    let d_hat = -0.5 * b_hat;

    // Asymptotic SE: π² / (6 * sxx)  (c.f. Geweke & Porter-Hudak 1983)
    let se_b = (std::f64::consts::PI.powi(2) / 6.0 / sxx).sqrt();
    let se_d = 0.5 * se_b;

    Ok((d_hat, se_d))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_returns(n: usize) -> Array1<f64> {
        // Simple pseudo-returns with sign alternation to avoid constant series
        let vals: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 3 == 0 { 1.0 } else { -1.0 };
                sign * (0.005 + 0.003 * ((i as f64 * 0.37).sin()))
            })
            .collect();
        Array1::from(vals)
    }

    #[test]
    fn test_figarch_model_new_valid() {
        let m = FIGARCHModel::new(1, 0.4, 1, 1e-5, vec![0.2], vec![0.3], 50)
            .expect("Should create FIGARCH model");
        assert!((m.d - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_figarch_model_invalid_d() {
        assert!(FIGARCHModel::new(1, 0.0, 1, 1e-5, vec![0.2], vec![0.3], 50).is_err());
        assert!(FIGARCHModel::new(1, 1.0, 1, 1e-5, vec![0.2], vec![0.3], 50).is_err());
        assert!(FIGARCHModel::new(1, -0.2, 1, 1e-5, vec![0.2], vec![0.3], 50).is_err());
    }

    #[test]
    fn test_lambda_weights_positive() {
        let m =
            FIGARCHModel::new(1, 0.4, 1, 1e-5, vec![0.2], vec![0.3], 30).expect("Should create");
        let lam = m.lambda_weights();
        assert_eq!(lam.len(), 30);
        // All weights must be non-negative (after clamping)
        for &lk in &lam {
            assert!(lk >= 0.0, "lambda weight must be >= 0, got {lk}");
        }
    }

    #[test]
    fn test_figarch_variance_positive() {
        let r = make_returns(80);
        let model =
            FIGARCHModel::new(1, 0.4, 1, 1e-5, vec![0.1], vec![0.3], 20).expect("Should create");
        let lambda = model.lambda_weights();
        let s2 = figarch_variance(&r, model.omega, &lambda, &model.beta)
            .expect("Should compute variance");
        assert_eq!(s2.len(), r.len());
        for &v in &s2 {
            assert!(v > 0.0, "conditional variance must be positive: {v}");
        }
    }

    #[test]
    fn test_figarch_log_likelihood() {
        let r = make_returns(80);
        let model =
            FIGARCHModel::new(1, 0.4, 1, 1e-5, vec![0.1], vec![0.3], 20).expect("Should create");
        let lambda = model.lambda_weights();
        let (ll, _) = figarch_log_likelihood(&r, model.omega, &lambda, &model.beta)
            .expect("Should compute LL");
        assert!(ll.is_finite(), "Log-likelihood should be finite: {ll}");
    }

    #[test]
    fn test_fit_figarch_basic() {
        // Use smaller truncation and fewer obs for faster test
        let r = make_returns(80);
        let model = fit_figarch(&r, 1, 0.4, 1, 20).expect("FIGARCH should fit");
        assert!(model.d > 0.0 && model.d < 1.0);
        assert!(model.omega > 0.0);
        assert!(model.log_likelihood.is_finite());
    }

    #[test]
    fn test_figarch_conditional_volatility() {
        let r = make_returns(80);
        let model =
            FIGARCHModel::new(1, 0.4, 1, 1e-5, vec![0.1], vec![0.3], 20).expect("Should create");
        let vol = figarch_conditional_volatility(&model, &r).expect("Should compute");
        assert_eq!(vol.len(), r.len());
        for &v in &vol {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_gph_estimator_basic() {
        // A series with known long memory (d ≈ 0.3) cannot be verified exactly,
        // but we check the estimator runs and returns finite values.
        let r = make_returns(100);
        let abs_r: Array1<f64> = r.mapv(|x| x.abs());
        let m = (100.0_f64).sqrt() as usize;
        let (d_hat, se_d) = gph_estimator(&abs_r, m).expect("GPH should compute");
        assert!(d_hat.is_finite(), "d_hat should be finite: {d_hat}");
        assert!(se_d > 0.0, "se_d should be positive: {se_d}");
    }
}
