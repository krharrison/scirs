//! Generalized Pareto Distribution (GPD) for exceedances.
//!
//! The GPD models the distribution of values exceeding a high threshold u.
//! It arises as the limiting distribution of threshold exceedances in the
//! Peaks Over Threshold (POT) framework.
//!
//! # Parameterization
//! GPD(σ, ξ):
//! - σ > 0: scale
//! - ξ ∈ ℝ: shape
//! - Support: x ≥ 0 (ξ ≥ 0) or 0 ≤ x ≤ -σ/ξ (ξ < 0)
//!
//! # References
//! - Pickands, J. (1975). Statistical inference using extreme order statistics.
//! - Davison & Smith (1990). Models for exceedances over high thresholds.

use crate::error::{StatsError, StatsResult};

const XI_THRESHOLD: f64 = 1e-10;

// ---------------------------------------------------------------------------
// GPD struct
// ---------------------------------------------------------------------------

/// Generalized Pareto Distribution for modeling threshold exceedances.
#[derive(Debug, Clone, PartialEq)]
pub struct GPD {
    /// Scale parameter σ > 0
    pub sigma: f64,
    /// Shape parameter ξ
    pub xi: f64,
}

impl GPD {
    /// Create a new GPD, validating σ > 0.
    pub fn new(sigma: f64, xi: f64) -> StatsResult<Self> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "GPD scale σ must be positive".into(),
            ));
        }
        Ok(Self { sigma, xi })
    }

    /// Probability density function at `x` (exceedance above threshold).
    pub fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if self.xi.abs() < XI_THRESHOLD {
            // Exponential: f(x) = (1/σ) * exp(-x/σ)
            let val = (-x / self.sigma).exp() / self.sigma;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        } else {
            let t = 1.0 + self.xi * x / self.sigma;
            if t <= 0.0 {
                return 0.0;
            }
            // Check upper bound for ξ < 0
            if self.xi < 0.0 && x > -self.sigma / self.xi {
                return 0.0;
            }
            let val = t.powf(-1.0 / self.xi - 1.0) / self.sigma;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        }
    }

    /// Cumulative distribution function at `x`.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        if self.xi < 0.0 {
            let upper = -self.sigma / self.xi;
            if x >= upper {
                return 1.0;
            }
        }
        if self.xi.abs() < XI_THRESHOLD {
            // Exponential
            (1.0 - (-x / self.sigma).exp()).clamp(0.0, 1.0)
        } else {
            let t = 1.0 + self.xi * x / self.sigma;
            if t <= 0.0 {
                return if self.xi > 0.0 { 0.0 } else { 1.0 };
            }
            (1.0 - t.powf(-1.0 / self.xi)).clamp(0.0, 1.0)
        }
    }

    /// Quantile function: returns x such that F(x) = p.
    pub fn quantile(&self, p: f64) -> StatsResult<f64> {
        if !(0.0 < p && p < 1.0) {
            return Err(StatsError::InvalidArgument(
                "quantile probability must be in (0, 1)".into(),
            ));
        }
        let q = if self.xi.abs() < XI_THRESHOLD {
            -self.sigma * (1.0 - p).ln()
        } else {
            self.sigma * ((1.0 - p).powf(-self.xi) - 1.0) / self.xi
        };
        if q.is_finite() && q >= 0.0 {
            Ok(q)
        } else {
            Err(StatsError::ComputationError(
                "GPD quantile produced non-finite or negative value".into(),
            ))
        }
    }

    /// Log-likelihood for the given exceedances.
    pub fn log_likelihood(&self, exceedances: &[f64]) -> f64 {
        if exceedances.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut ll = 0.0;
        for &x in exceedances {
            let p = self.pdf(x);
            if p > 0.0 && p.is_finite() {
                ll += p.ln();
            } else {
                return f64::NEG_INFINITY;
            }
        }
        ll
    }

    /// Mean of the GPD (exists only for ξ < 1).
    pub fn mean(&self) -> Option<f64> {
        if self.xi >= 1.0 {
            None
        } else {
            Some(self.sigma / (1.0 - self.xi))
        }
    }

    /// Variance of the GPD (exists only for ξ < 0.5).
    pub fn variance(&self) -> Option<f64> {
        if self.xi >= 0.5 {
            None
        } else {
            Some(self.sigma * self.sigma / ((1.0 - self.xi).powi(2) * (1.0 - 2.0 * self.xi)))
        }
    }

    /// Fit GPD via Probability-Weighted Moments (PWM / L-moments).
    ///
    /// This is the method of moments estimator and is fast, closed-form, and
    /// robust for moderate sample sizes (n ≥ 15).
    ///
    /// # Errors
    /// Returns an error if `exceedances.len() < 5`.
    pub fn fit(exceedances: &[f64]) -> StatsResult<GPD> {
        if exceedances.len() < 5 {
            return Err(StatsError::InsufficientData(
                "GPD PWM fitting requires at least 5 exceedances".into(),
            ));
        }
        gpd_fit_pwm(exceedances)
    }

    /// Fit GPD via Maximum Likelihood Estimation (gradient-based numerical optimizer).
    ///
    /// Uses a simple gradient descent / line search approach.
    ///
    /// # Errors
    /// Returns an error if `exceedances.len() < 5`.
    pub fn fit_mle(exceedances: &[f64]) -> StatsResult<GPD> {
        if exceedances.len() < 5 {
            return Err(StatsError::InsufficientData(
                "GPD MLE fitting requires at least 5 exceedances".into(),
            ));
        }
        gpd_fit_mle_newton(exceedances)
    }
}

// ---------------------------------------------------------------------------
// PWM (Probability-Weighted Moments) fitting
// ---------------------------------------------------------------------------

/// Fit GPD via PWM (Hosking & Wallis 1987).
fn gpd_fit_pwm(exceedances: &[f64]) -> StatsResult<GPD> {
    let n = exceedances.len();
    let mut sorted = exceedances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Unbiased PWM: b_r = (1/n) Σ C(i-1, r) / C(n-1, r) * x_i:n
    let mut b0 = 0.0f64;
    let mut b1 = 0.0f64;

    for (i, &x) in sorted.iter().enumerate() {
        let i_f = i as f64;
        let n_f = n as f64;
        b0 += x;
        b1 += (i_f / (n_f - 1.0)) * x;
    }
    b0 /= n as f64;
    b1 /= n as f64;

    // GPD PWM estimators (Hosking & Wallis 1987, eq. 4)
    // ξ = 2 - b0/(b0 - 2*b1)
    // σ = 2*b0*b1/(b0 - 2*b1)
    let denom = b0 - 2.0 * b1;
    if denom.abs() < 1e-15 {
        // Default to exponential
        return Ok(GPD { sigma: b0, xi: 0.0 });
    }

    let xi = 2.0 - b0 / denom;
    let sigma = 2.0 * b0 * b1 / denom;

    if sigma <= 0.0 {
        // Fallback: exponential with mean = sample mean
        return Ok(GPD { sigma: b0, xi: 0.0 });
    }

    // Clamp xi to avoid extreme values
    let xi_clamped = xi.clamp(-5.0, 5.0);
    Ok(GPD {
        sigma,
        xi: xi_clamped,
    })
}

// ---------------------------------------------------------------------------
// MLE fitting via gradient descent
// ---------------------------------------------------------------------------

/// Fit GPD via MLE using gradient descent with line search.
fn gpd_fit_mle_newton(exceedances: &[f64]) -> StatsResult<GPD> {
    // Start from PWM estimates
    let initial = gpd_fit_pwm(exceedances)?;
    let (mut sigma, mut xi) = (initial.sigma, initial.xi);

    let neg_ll = |s: f64, x: f64| -> f64 {
        if s <= 0.0 {
            return 1e15;
        }
        match GPD::new(s, x) {
            Ok(gpd) => {
                let ll = gpd.log_likelihood(exceedances);
                if ll.is_finite() {
                    -ll
                } else {
                    1e15
                }
            }
            Err(_) => 1e15,
        }
    };

    // Numerical gradient
    let grad = |s: f64, x: f64| -> (f64, f64) {
        let h_s = s * 1e-5 + 1e-8;
        let h_x = 1e-5;
        let ds = (neg_ll(s + h_s, x) - neg_ll(s - h_s, x)) / (2.0 * h_s);
        let dx = (neg_ll(s, x + h_x) - neg_ll(s, x - h_x)) / (2.0 * h_x);
        (ds, dx)
    };

    let max_iter = 500;
    let mut step = 0.01;
    let mut f_current = neg_ll(sigma, xi);

    for _ in 0..max_iter {
        let (ds, dx) = grad(sigma, xi);
        let norm = (ds * ds + dx * dx).sqrt();
        if norm < 1e-8 {
            break;
        }
        let ds_n = ds / norm;
        let dx_n = dx / norm;

        // Line search
        let mut found = false;
        for _ in 0..20 {
            let s_new = (sigma - step * ds_n).max(1e-8);
            let x_new = (xi - step * dx_n).clamp(-5.0, 5.0);
            let f_new = neg_ll(s_new, x_new);
            if f_new < f_current {
                sigma = s_new;
                xi = x_new;
                f_current = f_new;
                step *= 1.2;
                found = true;
                break;
            }
            step *= 0.5;
        }
        if !found {
            break;
        }
    }

    if sigma <= 0.0 {
        return Err(StatsError::ComputationError(
            "GPD MLE converged to non-positive scale".into(),
        ));
    }
    Ok(GPD { sigma, xi })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_gpd_new_invalid_sigma() {
        assert!(GPD::new(0.0, 0.0).is_err());
        assert!(GPD::new(-1.0, 0.0).is_err());
    }

    #[test]
    fn test_gpd_exponential_case() {
        // ξ=0: GPD reduces to Exp(1/σ)
        let g = GPD::new(2.0, 0.0).unwrap();
        let x = 3.0;
        let expected_cdf = 1.0 - (-x / 2.0_f64).exp();
        assert!(approx_eq(g.cdf(x), expected_cdf, 1e-12));
        let expected_pdf = 0.5 * (-1.5_f64).exp();
        assert!(approx_eq(g.pdf(x), expected_pdf, 1e-10));
    }

    #[test]
    fn test_gpd_cdf_at_zero() {
        let g = GPD::new(1.0, 0.5).unwrap();
        assert_eq!(g.cdf(0.0), 0.0);
    }

    #[test]
    fn test_gpd_quantile_roundtrip() {
        let g = GPD::new(1.5, 0.2).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            let q = g.quantile(p).unwrap();
            assert!(approx_eq(g.cdf(q), p, 1e-8), "p={p}, q={q}");
        }
    }

    #[test]
    fn test_gpd_quantile_invalid() {
        let g = GPD::new(1.0, 0.0).unwrap();
        assert!(g.quantile(0.0).is_err());
        assert!(g.quantile(1.0).is_err());
    }

    #[test]
    fn test_gpd_bounded_support() {
        // ξ < 0: upper bound = -σ/ξ
        let g = GPD::new(1.0, -0.5).unwrap();
        let upper = 2.0;
        assert_eq!(g.cdf(upper + 0.1), 1.0);
        assert!(approx_eq(g.cdf(upper), 1.0, 1e-10));
    }

    #[test]
    fn test_gpd_mean() {
        let g = GPD::new(2.0, 0.3).unwrap();
        let m = g.mean().unwrap();
        assert!(approx_eq(m, 2.0 / 0.7, 1e-10));
    }

    #[test]
    fn test_gpd_mean_undefined() {
        let g = GPD::new(1.0, 1.0).unwrap();
        assert!(g.mean().is_none());
    }

    #[test]
    fn test_gpd_fit_pwm() {
        // Exponential data: GPD with ξ ≈ 0
        let data: Vec<f64> = (1..=100)
            .map(|i| -2.0 * (1.0 - i as f64 / 101.0).ln())
            .collect();
        let gpd = GPD::fit(&data).unwrap();
        assert!(gpd.sigma > 0.0);
    }

    #[test]
    fn test_gpd_fit_insufficient_data() {
        assert!(GPD::fit(&[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_gpd_fit_mle() {
        let data: Vec<f64> = (1..=50)
            .map(|i| -1.5 * (1.0 - i as f64 / 51.0).ln())
            .collect();
        let gpd = GPD::fit_mle(&data).unwrap();
        assert!(gpd.sigma > 0.0);
    }

    #[test]
    fn test_gpd_log_likelihood_empty() {
        let g = GPD::new(1.0, 0.0).unwrap();
        assert_eq!(g.log_likelihood(&[]), f64::NEG_INFINITY);
    }
}
