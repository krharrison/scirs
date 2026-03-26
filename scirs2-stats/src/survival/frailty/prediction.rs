//! Prediction utilities for frailty survival models.
//!
//! Provides conditional and marginal survival functions, intraclass correlation
//! coefficient (ICC), empirical Bayes frailty estimates, and median survival
//! computation.

use crate::error::{StatsError, StatsResult};

use super::types::{FrailtyDistribution, FrailtyResult};

// ---------------------------------------------------------------------------
// Log-gamma helper
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Conditional survival
// ---------------------------------------------------------------------------

/// Compute the conditional survival probability S(t | u, x, β).
///
/// Given a specific frailty value u, covariates x, regression coefficients β,
/// and baseline survival S₀(t):
///
///   S(t | u, x) = S₀(t)^{u · exp(x · β)}
///
/// # Arguments
/// * `baseline_survival` – S₀(t) at the desired time point (0 < S₀ ≤ 1)
/// * `frailty`           – frailty value u > 0
/// * `beta`              – regression coefficients
/// * `x`                 – covariate vector (same length as beta)
///
/// # Returns
/// The conditional survival probability in [0, 1].
pub fn conditional_survival(baseline_survival: f64, frailty: f64, beta: &[f64], x: &[f64]) -> f64 {
    let xb: f64 = beta.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
    let exponent = frailty.max(0.0) * xb.exp();
    // S₀(t)^{u * exp(xβ)}
    baseline_survival.clamp(0.0, 1.0).powf(exponent)
}

/// Compute the marginal survival probability by integrating over the frailty distribution.
///
/// For Gamma(1/θ, θ) frailty, the marginal survival has a closed form:
///   S_m(t | x) = (1 + θ · H₀(t) · exp(xβ))^{-1/θ}
///
/// For other distributions, numerical integration (Gauss-Hermite quadrature) is used.
///
/// # Arguments
/// * `cumulative_baseline_hazard` – H₀(t) at the desired time point (≥ 0)
/// * `theta`                      – frailty variance θ > 0
/// * `beta`                       – regression coefficients
/// * `x`                          – covariate vector
/// * `distribution`               – frailty distribution family
pub fn marginal_survival(
    cumulative_baseline_hazard: f64,
    theta: f64,
    beta: &[f64],
    x: &[f64],
    distribution: FrailtyDistribution,
) -> f64 {
    let xb: f64 = beta.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
    let h = cumulative_baseline_hazard.max(0.0) * xb.exp();

    match distribution {
        FrailtyDistribution::Gamma => {
            // Closed form: S_m(t) = (1 + θ H)^{-1/θ}
            let theta = theta.max(1e-15);
            (1.0 + theta * h).powf(-1.0 / theta)
        }
        FrailtyDistribution::LogNormal => {
            // Numerical integration via Gauss-Hermite quadrature (5-point)
            // log(u) ~ N(-σ²/2, σ²), σ² = ln(1+θ)
            let sigma2 = (1.0 + theta.max(0.0)).ln().max(1e-10);
            let sigma = sigma2.sqrt();
            let mu = -sigma2 / 2.0;

            // Gauss-Hermite nodes and weights (5-point)
            let nodes = [
                -2.020_182_870_456_085_6,
                -0.958_572_464_613_818_7,
                0.0,
                0.958_572_464_613_818_7,
                2.020_182_870_456_085_6,
            ];
            let weights = [
                0.019_953_242_059_045_913,
                0.393_619_323_152_241_16,
                0.945_308_720_482_941_9,
                0.393_619_323_152_241_16,
                0.019_953_242_059_045_913,
            ];

            let mut integral = 0.0_f64;
            for (&node, &weight) in nodes.iter().zip(weights.iter()) {
                // Transform: v = mu + sigma * sqrt(2) * node
                let v = mu + sigma * std::f64::consts::SQRT_2 * node;
                let u = v.exp();
                // S(t|u) = exp(-u * H)
                let s_cond = (-u * h).exp();
                integral += weight * s_cond;
            }
            // Gauss-Hermite normalisation: integral / sqrt(pi)
            (integral / std::f64::consts::PI.sqrt()).clamp(0.0, 1.0)
        }
        FrailtyDistribution::InverseGaussian => {
            // Closed form for IG(1, λ) frailty, λ = 1/θ:
            // S_m(t) = exp(λ (1 - sqrt(1 + 2θH)))
            let theta = theta.max(1e-15);
            let lambda = 1.0 / theta;
            let inner = (1.0 + 2.0 * theta * h).sqrt();
            (lambda * (1.0 - inner)).exp().clamp(0.0, 1.0)
        }
        _ => {
            // Fallback: no frailty effect
            (-h).exp().clamp(0.0, 1.0)
        }
    }
}

/// Return the empirical Bayes frailty estimates from a fitted result.
pub fn frailty_estimates(result: &FrailtyResult) -> &[f64] {
    &result.frailty_estimates
}

/// Compute the intraclass correlation coefficient (ICC) for a frailty model.
///
/// The ICC measures the proportion of total variance attributable to between-cluster
/// heterogeneity. For different distributions:
///
/// - **Gamma**: ICC = θ / (θ + π²/6)  (based on logistic variance)
/// - **LogNormal**: ICC = σ² / (σ² + π²/6) where σ² = ln(1+θ)
/// - **InverseGaussian**: ICC = θ / (θ + π²/6)  (same formula as Gamma)
///
/// # Arguments
/// * `theta`        – frailty variance (> 0)
/// * `distribution` – frailty distribution family
///
/// # Returns
/// ICC in [0, 1].
pub fn intraclass_correlation(theta: f64, distribution: FrailtyDistribution) -> f64 {
    let pi2_over_6 = std::f64::consts::PI.powi(2) / 6.0;
    let theta = theta.max(0.0);

    match distribution {
        FrailtyDistribution::Gamma | FrailtyDistribution::InverseGaussian => {
            theta / (theta + pi2_over_6)
        }
        FrailtyDistribution::LogNormal => {
            let sigma2 = (1.0 + theta).ln().max(0.0);
            sigma2 / (sigma2 + pi2_over_6)
        }
        _ => 0.0,
    }
}

/// Compute the median survival time for a given frailty and covariate profile.
///
/// Finds t such that S(t | u, x) = 0.5, i.e., S₀(t)^{u·exp(xβ)} = 0.5.
/// Uses bisection search over the provided baseline hazard table.
///
/// # Arguments
/// * `frailty`         – frailty value u > 0
/// * `beta`            – regression coefficients
/// * `x`               – covariate vector
/// * `baseline_hazard` – sorted (time, H₀(t)) pairs from the fitted model
///
/// # Errors
/// Returns `StatsError` if median is outside the observed time range.
pub fn median_survival(
    frailty: f64,
    beta: &[f64],
    x: &[f64],
    baseline_hazard: &[(f64, f64)],
) -> StatsResult<f64> {
    if baseline_hazard.is_empty() {
        return Err(StatsError::InvalidArgument(
            "baseline_hazard must not be empty".into(),
        ));
    }

    let xb: f64 = beta.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
    let exponent = frailty.max(1e-15) * xb.exp();

    // Target: S₀(t)^exponent = 0.5
    // => -H₀(t) * exponent = ln(0.5)
    // => H₀(t) = -ln(0.5) / exponent = ln(2) / exponent
    let target_h0 = std::f64::consts::LN_2 / exponent;

    // Find the time where H₀(t) crosses target_h0
    let last = baseline_hazard.last().map(|&(_, h)| h).unwrap_or(0.0);
    if target_h0 > last {
        return Err(StatsError::ComputationError(
            "Median survival time exceeds the observed time range".into(),
        ));
    }

    // Binary search / linear interpolation
    let mut lo = 0_usize;
    let mut hi = baseline_hazard.len() - 1;

    // Check if target is before first point
    if target_h0 <= baseline_hazard[0].1 {
        return Ok(baseline_hazard[0].0);
    }

    while lo < hi - 1 {
        let mid = (lo + hi) / 2;
        if baseline_hazard[mid].1 < target_h0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation between lo and hi
    let (t_lo, h_lo) = baseline_hazard[lo];
    let (t_hi, h_hi) = baseline_hazard[hi];
    let dh = h_hi - h_lo;
    if dh.abs() < 1e-30 {
        return Ok(t_lo);
    }
    let frac = (target_h0 - h_lo) / dh;
    Ok(t_lo + frac * (t_hi - t_lo))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conditional_survival_monotone_decreasing() {
        // As baseline survival decreases (earlier => higher S₀), conditional should decrease
        let beta = [0.5];
        let x = [1.0];
        let frailty = 1.0;

        let s_values: Vec<f64> = (1..=10)
            .map(|i| {
                let s0 = 1.0 - i as f64 * 0.08; // decreasing S₀
                conditional_survival(s0, frailty, &beta, &x)
            })
            .collect();

        for i in 1..s_values.len() {
            assert!(
                s_values[i] <= s_values[i - 1] + 1e-10,
                "Conditional survival should be monotone decreasing in cumulative hazard"
            );
        }
    }

    #[test]
    fn test_conditional_survival_no_frailty_effect() {
        // With frailty = 1 and no covariates, S(t|u=1, x=0) = S₀(t)
        let s0 = 0.7;
        let s = conditional_survival(s0, 1.0, &[], &[]);
        assert!((s - s0).abs() < 1e-10);
    }

    #[test]
    fn test_marginal_survival_gamma_closed_form() {
        // S_m(t) = (1 + θH)^{-1/θ}
        let theta: f64 = 0.5;
        let h: f64 = 1.0;
        let expected = (1.0 + theta * h).powf(-1.0 / theta);
        let result = marginal_survival(h, theta, &[], &[], FrailtyDistribution::Gamma);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_marginal_survival_smooth() {
        // Marginal survival should be monotone decreasing in cumulative hazard
        let theta = 0.5;
        let beta = [0.0];
        let x = [0.0];

        for dist in &[
            FrailtyDistribution::Gamma,
            FrailtyDistribution::LogNormal,
            FrailtyDistribution::InverseGaussian,
        ] {
            let mut prev = 1.0_f64;
            for h_int in 0..=20 {
                let h = h_int as f64 * 0.5;
                let s = marginal_survival(h, theta, &beta, &x, *dist);
                assert!(
                    s <= prev + 1e-10,
                    "Marginal survival should be non-increasing for {:?}: s={s}, prev={prev}, h={h}",
                    dist
                );
                assert!(s >= 0.0 && s <= 1.0, "Survival must be in [0,1]");
                prev = s;
            }
        }
    }

    #[test]
    fn test_icc_gamma_formula() {
        // ICC = θ / (θ + π²/6)
        let theta = 1.0;
        let pi2_6 = std::f64::consts::PI.powi(2) / 6.0;
        let expected = theta / (theta + pi2_6);
        let icc = intraclass_correlation(theta, FrailtyDistribution::Gamma);
        assert!(
            (icc - expected).abs() < 1e-10,
            "ICC should match θ/(θ+π²/6)"
        );
    }

    #[test]
    fn test_icc_zero_variance() {
        let icc = intraclass_correlation(0.0, FrailtyDistribution::Gamma);
        assert!((icc - 0.0).abs() < 1e-10, "ICC should be 0 when θ=0");
    }

    #[test]
    fn test_icc_large_variance() {
        let icc = intraclass_correlation(100.0, FrailtyDistribution::Gamma);
        assert!(icc > 0.9, "ICC should approach 1 for large θ");
    }

    #[test]
    fn test_icc_lognormal() {
        let theta: f64 = 1.0;
        let sigma2 = (1.0 + theta).ln();
        let pi2_6 = std::f64::consts::PI.powi(2) / 6.0;
        let expected = sigma2 / (sigma2 + pi2_6);
        let icc = intraclass_correlation(theta, FrailtyDistribution::LogNormal);
        assert!((icc - expected).abs() < 1e-10);
    }

    #[test]
    fn test_median_survival_basic() {
        // Simple baseline: H₀ increases linearly
        let baseline: Vec<(f64, f64)> = (1..=10).map(|i| (i as f64, i as f64 * 0.3)).collect();
        let result = median_survival(1.0, &[], &[], &baseline);
        assert!(result.is_ok());
        let median = result.expect("should succeed");
        // ln(2) / 1.0 ≈ 0.693, target H₀ = 0.693
        // H₀(t) = t * 0.3, so t ≈ 0.693/0.3 ≈ 2.31
        assert!(median > 1.0 && median < 4.0);
    }

    #[test]
    fn test_median_survival_empty_baseline_error() {
        let result = median_survival(1.0, &[], &[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_frailty_estimates_accessor() {
        let result = FrailtyResult {
            coefficients: vec![0.5],
            frailty_variance: 0.3,
            frailty_estimates: vec![1.1, 0.9, 1.0],
            log_likelihood_history: vec![-100.0, -95.0],
            converged: true,
            iterations: 10,
            baseline_hazard: vec![(1.0, 0.1)],
        };
        let estimates = frailty_estimates(&result);
        assert_eq!(estimates.len(), 3);
        assert!((estimates[0] - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_marginal_survival_inverse_gaussian() {
        // S_m(t) = exp(λ(1 - sqrt(1 + 2θH)))
        let theta: f64 = 0.5;
        let h: f64 = 1.0;
        let lambda: f64 = 1.0 / theta;
        let expected = (lambda * (1.0 - (1.0 + 2.0 * theta * h).sqrt())).exp();
        let result = marginal_survival(h, theta, &[], &[], FrailtyDistribution::InverseGaussian);
        assert!(
            (result - expected).abs() < 1e-10,
            "IG marginal survival should match closed form"
        );
    }
}
