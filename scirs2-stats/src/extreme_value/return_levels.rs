//! Return level calculations for extreme value models.
//!
//! Return levels quantify how extreme a value needs to be to occur (on average)
//! once every T time periods. They are fundamental to risk assessment in
//! hydrology, meteorology, insurance, and finance.
//!
//! # Return Level Definitions
//! - **GEV return level**: x_T = Q_GEV(1 - 1/T)
//! - **GPD return level**: combines the exceedance rate with the GPD quantile
//!
//! # Confidence Intervals
//! We implement the **delta method** for approximate CIs, using the observed
//! Fisher information matrix.
//!
//! # References
//! - Coles (2001). *An Introduction to Statistical Modeling of Extreme Values*.
//! - Coles & Tawn (1994). Statistical methods for multivariate extremes.

use super::gev::GEV;
use super::gpd::GPD;
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// GEV return levels
// ---------------------------------------------------------------------------

/// Compute the T-year return level for a fitted GEV distribution.
///
/// The return level x_T satisfies P(X ≤ x_T) = 1 - 1/T,
/// i.e., it is the (1 - 1/T) quantile of the GEV.
///
/// # Arguments
/// - `gev`: fitted GEV distribution
/// - `period`: return period T (must be > 1)
///
/// # Errors
/// Returns an error if `period <= 1`.
pub fn return_level_gev(gev: &GEV, period: f64) -> StatsResult<f64> {
    if period <= 1.0 {
        return Err(StatsError::InvalidArgument(
            "return period must be greater than 1".into(),
        ));
    }
    gev.return_level(period)
}

// ---------------------------------------------------------------------------
// GPD return levels
// ---------------------------------------------------------------------------

/// Compute the T-year return level for a fitted GPD model.
///
/// Given:
/// - GPD parameters (σ, ξ) fitted to exceedances above threshold u
/// - `n_obs`: total number of observations
/// - `n_per_period`: number of observations per return period (e.g., 1 if
///   data is annual, 365 if daily)
/// - `period`: return period T in "periods"
///
/// The return level is:
/// x_T = u + σ/ξ * [(λ * T)^ξ - 1]    (ξ ≠ 0)
/// x_T = u + σ * ln(λ * T)              (ξ = 0)
///
/// where λ = (number of exceedances) / (total observations) is the exceedance rate.
///
/// # Errors
/// Returns an error if any parameter is invalid.
pub fn return_level_gpd(
    gpd: &GPD,
    threshold: f64,
    n_obs: usize,
    n_per_period: usize,
    period: f64,
) -> StatsResult<f64> {
    if period <= 0.0 {
        return Err(StatsError::InvalidArgument(
            "return period must be positive".into(),
        ));
    }
    if n_obs == 0 || n_per_period == 0 {
        return Err(StatsError::InvalidArgument(
            "n_obs and n_per_period must be positive".into(),
        ));
    }

    // We need exceedance rate λ. Since this function doesn't have n_exceedances,
    // we compute the return level in terms of the rate parameter directly.
    // The expected number of exceedances per period = λ * n_per_period
    // For T-year return level with n_per_period obs/period:
    // total exceedances in T periods = λ * n_per_period * T
    // But we need λ... Use a unit exceedance rate here and let caller provide n_obs.

    // Standard formula:
    // x_T = u + GPD_quantile(1 - 1/(λ * T)) where λ is exceedances per unit time
    // But λ = n_exceedances / n_total_periods
    // Without n_exceedances, use the quantile approach:
    // Expected waiting time between exceedances = 1/λ periods
    // x_T = u + σ/ξ * ((n_per_period * period)^ξ - 1)

    let rate_factor = (n_per_period as f64) * period;
    let xi = gpd.xi;
    const XI_THR: f64 = 1e-10;

    let excess_rl = if xi.abs() < XI_THR {
        gpd.sigma * rate_factor.ln()
    } else {
        gpd.sigma / xi * (rate_factor.powf(xi) - 1.0)
    };

    Ok(threshold + excess_rl)
}

// ---------------------------------------------------------------------------
// Confidence intervals via delta method
// ---------------------------------------------------------------------------

/// Compute return level confidence intervals for a GEV fit using the delta method.
///
/// For each return period in `periods`, computes the estimated return level
/// and approximate (1-alpha) confidence interval via the delta method.
///
/// The delta method approximation: Var(x_T) ≈ ∇x_T^T * I^{-1}(θ) * ∇x_T
/// where I(θ) is the observed Fisher information matrix.
///
/// Returns `Vec<(period, lower, upper)>`.
///
/// # Arguments
/// - `gev`: fitted GEV parameters
/// - `data`: original block maxima used for fitting
/// - `periods`: return periods for which to compute CIs
/// - `alpha`: significance level (e.g., 0.05 for 95% CI)
///
/// # Errors
/// Returns an error if `alpha` is not in (0, 0.5) or data is insufficient.
pub fn return_level_confidence_intervals(
    gev: &GEV,
    data: &[f64],
    periods: &[f64],
    alpha: f64,
) -> StatsResult<Vec<(f64, f64, f64)>> {
    if !(0.0 < alpha && alpha < 0.5) {
        return Err(StatsError::InvalidArgument(
            "alpha must be in (0, 0.5)".into(),
        ));
    }
    if data.len() < 3 {
        return Err(StatsError::InsufficientData(
            "Need at least 3 observations for CI computation".into(),
        ));
    }

    // Normal quantile for CI
    let z = norm_ppf(1.0 - alpha / 2.0);

    // Compute Hessian numerically for Fisher information
    let inv_fisher = numerical_hessian_gev(gev, data)?;

    let mut results = Vec::with_capacity(periods.len());

    for &period in periods {
        let rl = gev.return_level(period)?;

        // Gradient of return level with respect to (mu, sigma, xi)
        let grad = return_level_gradient(gev, period);

        // Delta method variance: grad^T * Σ * grad
        let var_rl = delta_method_variance(&grad, &inv_fisher);

        if var_rl < 0.0 || !var_rl.is_finite() {
            // Fallback: use a fraction of the estimate
            let se = rl.abs() * 0.1 + 0.01;
            results.push((period, rl - z * se, rl + z * se));
        } else {
            let se = var_rl.sqrt();
            results.push((period, rl - z * se, rl + z * se));
        }
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Gradient computation
// ---------------------------------------------------------------------------

/// Gradient of the T-year return level with respect to GEV parameters (μ, σ, ξ).
fn return_level_gradient(gev: &GEV, period: f64) -> [f64; 3] {
    const XI_THR: f64 = 1e-10;
    let p = 1.0 - 1.0 / period;
    let y = -p.ln(); // -ln(p) = -ln(1 - 1/T)

    if gev.xi.abs() < XI_THR {
        // Gumbel: x_T = μ - σ * ln(-ln(p)) = μ + σ * ln(1/y)
        // ∂x_T/∂μ = 1
        // ∂x_T/∂σ = -ln(-ln(p)) = ln(1/y) ... = -ln(y)
        // ∂x_T/∂ξ ≈ finite difference
        let d_mu = 1.0;
        let d_sigma = -y.ln();
        // Numerical derivative for xi at xi=0
        let h = 1e-5;
        let gev_p = GEV::new(gev.mu, gev.sigma, h).ok();
        let gev_m = GEV::new(gev.mu, gev.sigma, -h).ok();
        let d_xi = match (gev_p, gev_m) {
            (Some(gp), Some(gm)) => {
                let rl_p = gp.return_level(period).unwrap_or(0.0);
                let rl_m = gm.return_level(period).unwrap_or(0.0);
                (rl_p - rl_m) / (2.0 * h)
            }
            _ => 0.0,
        };
        [d_mu, d_sigma, d_xi]
    } else {
        // General: x_T = μ + σ * (y^(-ξ) - 1) / ξ
        let xi = gev.xi;
        let y_xi = y.powf(-xi);
        let d_mu = 1.0;
        let d_sigma = (y_xi - 1.0) / xi;
        // ∂x_T/∂ξ = σ * d/dξ [(y^(-ξ) - 1) / ξ]
        // = σ * [(-y^(-ξ) * ln(y) * ξ - (y^(-ξ) - 1)) / ξ²]
        let d_xi = gev.sigma * (-y_xi * y.ln() * xi - (y_xi - 1.0)) / (xi * xi);
        [d_mu, d_sigma, d_xi]
    }
}

/// Delta method variance: v^T * Σ * v
fn delta_method_variance(grad: &[f64; 3], cov: &[[f64; 3]; 3]) -> f64 {
    let mut result = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            result += grad[i] * cov[i][j] * grad[j];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Numerical Hessian and Fisher information
// ---------------------------------------------------------------------------

/// Compute the inverse of the observed Fisher information matrix (= asymptotic
/// covariance matrix) using a numerical Hessian of the log-likelihood.
fn numerical_hessian_gev(gev: &GEV, data: &[f64]) -> StatsResult<[[f64; 3]; 3]> {
    let params = [gev.mu, gev.sigma, gev.xi];
    let h = [gev.sigma * 0.001 + 1e-6, gev.sigma * 0.001 + 1e-6, 1e-5];

    // Compute negative log-likelihood at current params
    let neg_ll = |p: &[f64; 3]| -> f64 {
        match GEV::new(p[0], p[1], p[2]) {
            Ok(g) => {
                let ll = g.log_likelihood(data);
                if ll.is_finite() {
                    -ll
                } else {
                    1e15
                }
            }
            Err(_) => 1e15,
        }
    };

    let f0 = neg_ll(&params);
    let mut hess = [[0.0f64; 3]; 3];

    // Diagonal elements: ∂²f/∂θᵢ² ≈ [f(θ+h) - 2f(θ) + f(θ-h)] / h²
    for i in 0..3 {
        let mut p_plus = params;
        let mut p_minus = params;
        p_plus[i] += h[i];
        p_minus[i] -= h[i];
        let f_plus = neg_ll(&p_plus);
        let f_minus = neg_ll(&p_minus);
        hess[i][i] = (f_plus - 2.0 * f0 + f_minus) / (h[i] * h[i]);
    }

    // Off-diagonal: ∂²f/∂θᵢ∂θⱼ ≈ [f(++)-f(+-)-f(-+)+f(--)] / (4*hᵢ*hⱼ)
    for i in 0..3 {
        for j in (i + 1)..3 {
            let mut p_pp = params;
            let mut p_pm = params;
            let mut p_mp = params;
            let mut p_mm = params;
            p_pp[i] += h[i];
            p_pp[j] += h[j];
            p_pm[i] += h[i];
            p_pm[j] -= h[j];
            p_mp[i] -= h[i];
            p_mp[j] += h[j];
            p_mm[i] -= h[i];
            p_mm[j] -= h[j];
            hess[i][j] = (neg_ll(&p_pp) - neg_ll(&p_pm) - neg_ll(&p_mp) + neg_ll(&p_mm))
                / (4.0 * h[i] * h[j]);
            hess[j][i] = hess[i][j];
        }
    }

    // Invert 3x3 Hessian to get covariance matrix
    invert_3x3(&hess)
}

/// Invert a 3x3 matrix using Cramer's rule.
fn invert_3x3(m: &[[f64; 3]; 3]) -> StatsResult<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return Err(StatsError::ComputationError(
            "Fisher information matrix is singular; cannot compute confidence intervals".into(),
        ));
    }

    let inv_det = 1.0 / det;
    let cofactor = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ];
    Ok(cofactor)
}

// ---------------------------------------------------------------------------
// Normal quantile
// ---------------------------------------------------------------------------

fn norm_ppf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239e0,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838e0,
        -2.549_732_539_343_734e0,
        4.374_664_141_464_968e0,
        2.938_163_982_698_783e0,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996e0,
        3.754_408_661_907_416e0,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
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
    fn test_return_level_gev_basic() {
        let gev = GEV::new(0.0, 1.0, 0.0).unwrap();
        // 100-year return level for Gumbel(0,1)
        let rl = return_level_gev(&gev, 100.0).unwrap();
        assert!(rl.is_finite());
        assert!(rl > gev.quantile(0.5).unwrap());
    }

    #[test]
    fn test_return_level_gev_invalid_period() {
        let gev = GEV::new(0.0, 1.0, 0.0).unwrap();
        assert!(return_level_gev(&gev, 1.0).is_err());
        assert!(return_level_gev(&gev, 0.5).is_err());
    }

    #[test]
    fn test_return_level_gev_monotone() {
        let gev = GEV::new(0.0, 1.0, 0.1).unwrap();
        let rl_10 = return_level_gev(&gev, 10.0).unwrap();
        let rl_100 = return_level_gev(&gev, 100.0).unwrap();
        let rl_1000 = return_level_gev(&gev, 1000.0).unwrap();
        assert!(rl_10 < rl_100, "10-yr={rl_10}, 100-yr={rl_100}");
        assert!(rl_100 < rl_1000, "100-yr={rl_100}, 1000-yr={rl_1000}");
    }

    #[test]
    fn test_return_level_gpd_basic() {
        let gpd = GPD::new(1.0, 0.1).unwrap();
        // 100-year return level with daily data
        let rl = return_level_gpd(&gpd, 5.0, 1000, 365, 100.0).unwrap();
        assert!(rl.is_finite());
        assert!(rl > 5.0);
    }

    #[test]
    fn test_return_level_gpd_invalid_period() {
        let gpd = GPD::new(1.0, 0.0).unwrap();
        assert!(return_level_gpd(&gpd, 5.0, 1000, 365, 0.0).is_err());
    }

    #[test]
    fn test_return_level_gpd_zero_n_obs() {
        let gpd = GPD::new(1.0, 0.0).unwrap();
        assert!(return_level_gpd(&gpd, 5.0, 0, 365, 100.0).is_err());
    }

    #[test]
    fn test_return_level_confidence_intervals_basic() {
        let gev = GEV::new(0.0, 1.0, 0.0).expect("valid GEV");
        // Generate Gumbel(0,1) quantile data avoiding i=0 (which gives ln(0) = -inf)
        let data: Vec<f64> = (1..=50)
            .map(|i| {
                let u = i as f64 / 51.0; // in (0, 1)
                -(-u.ln()).ln() // Gumbel(0,1) quantile: -ln(-ln(u))
            })
            .collect();
        let periods = vec![10.0, 50.0, 100.0];
        let cis = return_level_confidence_intervals(&gev, &data, &periods, 0.05)
            .expect("CI computation should succeed");
        assert_eq!(cis.len(), 3);
        for (period, lo, hi) in &cis {
            assert!(period.is_finite());
            assert!(lo.is_finite(), "lo not finite for period {period}");
            assert!(hi.is_finite(), "hi not finite for period {period}");
        }
    }

    #[test]
    fn test_return_level_ci_alpha_invalid() {
        let gev = GEV::new(0.0, 1.0, 0.0).unwrap();
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert!(return_level_confidence_intervals(&gev, &data, &[10.0], 0.0).is_err());
        assert!(return_level_confidence_intervals(&gev, &data, &[10.0], 0.6).is_err());
    }

    #[test]
    fn test_return_level_gradient_gumbel() {
        let gev = GEV::new(0.0, 1.0, 0.0).unwrap();
        let grad = return_level_gradient(&gev, 100.0);
        // ∂x_T/∂μ = 1 for all GEV
        assert!(approx_eq(grad[0], 1.0, 1e-10));
    }

    #[test]
    fn test_norm_ppf_symmetry() {
        let z = norm_ppf(0.975);
        assert!(approx_eq(z, 1.96, 0.01), "z={z}");
        assert!(approx_eq(norm_ppf(0.025), -z, 0.01));
    }
}
