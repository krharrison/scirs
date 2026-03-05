//! Generalized Extreme Value (GEV) distribution — unified Gumbel/Fréchet/Weibull.
//!
//! The GEV family unifies:
//! - ξ = 0: Gumbel (Type I, light-tailed)
//! - ξ > 0: Fréchet (Type II, heavy-tailed)  
//! - ξ < 0: Weibull/reversed-Weibull (Type III, bounded upper tail)
//!
//! # References
//! - Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*. Springer.

use crate::error::{StatsError, StatsResult};

/// Threshold below which |ξ| is treated as zero to avoid numerical issues.
const XI_THRESHOLD: f64 = 1e-10;

// ---------------------------------------------------------------------------
// GEV struct
// ---------------------------------------------------------------------------

/// Generalized Extreme Value distribution with parameters μ (location), σ (scale), ξ (shape).
///
/// - ξ → 0: Gumbel distribution  
/// - ξ > 0: Fréchet distribution  
/// - ξ < 0: Weibull (reversed) distribution
#[derive(Debug, Clone, PartialEq)]
pub struct GEV {
    /// Location parameter μ
    pub mu: f64,
    /// Scale parameter σ > 0
    pub sigma: f64,
    /// Shape parameter ξ
    pub xi: f64,
}

impl GEV {
    /// Create a new GEV distribution, validating σ > 0.
    pub fn new(mu: f64, sigma: f64, xi: f64) -> StatsResult<Self> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "GEV scale σ must be positive".into(),
            ));
        }
        Ok(Self { mu, sigma, xi })
    }

    /// Compute the GEV bracket term `t(x) = 1 + ξ*(x-μ)/σ`.
    /// Returns `None` if t ≤ 0 (outside support).
    #[inline]
    fn bracket(&self, x: f64) -> Option<f64> {
        let z = (x - self.mu) / self.sigma;
        if self.xi.abs() < XI_THRESHOLD {
            // Gumbel case: support is all reals
            Some(z) // we return z here; callers handle Gumbel separately
        } else {
            let t = 1.0 + self.xi * z;
            if t <= 0.0 {
                None
            } else {
                Some(t)
            }
        }
    }

    /// Probability density function.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        if self.xi.abs() < XI_THRESHOLD {
            // Gumbel: f(x) = (1/σ) * exp(-z - exp(-z))
            let val = (-z - (-z).exp()).exp() / self.sigma;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        } else {
            let t = 1.0 + self.xi * z;
            if t <= 0.0 {
                return 0.0;
            }
            // f(x) = (1/σ) * t^(-1/ξ - 1) * exp(-t^(-1/ξ))
            let inv_xi = 1.0 / self.xi;
            let t_pow = t.powf(-inv_xi - 1.0);
            let exp_term = (-t.powf(-inv_xi)).exp();
            let val = t_pow * exp_term / self.sigma;
            if val.is_finite() {
                val
            } else {
                0.0
            }
        }
    }

    /// Cumulative distribution function.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        if self.xi.abs() < XI_THRESHOLD {
            // Gumbel: F(x) = exp(-exp(-z))
            let val = (-(-z).exp()).exp();
            val.clamp(0.0, 1.0)
        } else {
            let t = 1.0 + self.xi * z;
            if t <= 0.0 {
                // Below support for Fréchet (ξ > 0), above support for Weibull (ξ < 0)
                return if self.xi > 0.0 { 0.0 } else { 1.0 };
            }
            // F(x) = exp(-t^(-1/ξ))
            let val = (-t.powf(-1.0 / self.xi)).exp();
            val.clamp(0.0, 1.0)
        }
    }

    /// Quantile function (inverse CDF). Returns the value x such that F(x) = p.
    ///
    /// # Errors
    /// Returns an error if p ∉ (0, 1).
    pub fn quantile(&self, p: f64) -> StatsResult<f64> {
        if !(0.0 < p && p < 1.0) {
            return Err(StatsError::InvalidArgument(
                "quantile probability must be in (0, 1)".into(),
            ));
        }
        // Q(p) = μ + σ * g(p) where:
        //   ξ = 0: g(p) = -ln(-ln(p))
        //   ξ ≠ 0: g(p) = ((-ln(p))^(-ξ) - 1) / ξ
        let q = if self.xi.abs() < XI_THRESHOLD {
            self.mu - self.sigma * (-p.ln()).ln()
        } else {
            self.mu + self.sigma * ((-p.ln()).powf(-self.xi) - 1.0) / self.xi
        };
        if q.is_finite() {
            Ok(q)
        } else {
            Err(StatsError::ComputationError(
                "quantile computation produced non-finite value".into(),
            ))
        }
    }

    /// T-year return level: the value exceeded on average once every T years.
    ///
    /// Equivalent to `quantile(1 - 1/T)`.
    pub fn return_level(&self, return_period: f64) -> StatsResult<f64> {
        if return_period <= 1.0 {
            return Err(StatsError::InvalidArgument(
                "return period must be > 1".into(),
            ));
        }
        self.quantile(1.0 - 1.0 / return_period)
    }

    /// Log-likelihood of the GEV distribution for the given data.
    pub fn log_likelihood(&self, data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NEG_INFINITY;
        }
        let mut ll = 0.0;
        for &x in data {
            let p = self.pdf(x);
            if p > 0.0 && p.is_finite() {
                ll += p.ln();
            } else {
                return f64::NEG_INFINITY;
            }
        }
        ll
    }

    /// Fit a GEV distribution to data using Nelder-Mead simplex maximization.
    ///
    /// Returns `(fitted_gev, log_likelihood)`.
    ///
    /// # Errors
    /// Returns an error if `data.len() < 3`.
    pub fn fit(data: &[f64]) -> StatsResult<(GEV, f64)> {
        if data.len() < 3 {
            return Err(StatsError::InsufficientData(
                "GEV fitting requires at least 3 observations".into(),
            ));
        }

        // Initial parameter estimates using L-moments
        let (mu0, sigma0, xi0) = gev_lmoment_estimates(data);

        // Try fitting with L-moment starting point
        let mut best_gev: Option<GEV> = None;
        let mut best_ll = f64::NEG_INFINITY;

        // Attempt 1: L-moment initial estimates
        if let Ok((mu, sigma, xi)) = nelder_mead_gev(data, mu0, sigma0, xi0) {
            if let Ok(gev) = GEV::new(mu, sigma, xi) {
                let ll = gev.log_likelihood(data);
                if ll.is_finite() && ll > best_ll {
                    best_ll = ll;
                    best_gev = Some(gev);
                }
            }
        }

        // Attempt 2: Gumbel starting point (xi = 0)
        if !best_ll.is_finite() {
            // Simple Gumbel MOM estimates
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            let sigma_gumbel = (6.0 * variance / (std::f64::consts::PI * std::f64::consts::PI))
                .sqrt()
                .max(1e-8);
            let mu_gumbel = mean - 0.577_215_664_901_532_9 * sigma_gumbel;

            if let Ok((mu, sigma, xi)) = nelder_mead_gev(data, mu_gumbel, sigma_gumbel, 0.0) {
                if let Ok(gev) = GEV::new(mu, sigma, xi) {
                    let ll = gev.log_likelihood(data);
                    if ll.is_finite() && ll > best_ll {
                        best_ll = ll;
                        best_gev = Some(gev);
                    }
                }
            }
        }

        // Attempt 3: Pure Gumbel (no shape parameter optimization)
        if !best_ll.is_finite() {
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            let sigma_gumbel = (6.0 * variance / (std::f64::consts::PI * std::f64::consts::PI))
                .sqrt()
                .max(1e-8);
            let mu_gumbel = mean - 0.577_215_664_901_532_9 * sigma_gumbel;

            if let Ok(gev) = GEV::new(mu_gumbel, sigma_gumbel, 0.0) {
                let ll = gev.log_likelihood(data);
                if ll.is_finite() && ll > best_ll {
                    best_ll = ll;
                    best_gev = Some(gev);
                }
            }
        }

        match best_gev {
            Some(gev) => Ok((gev, best_ll)),
            None => Err(StatsError::ComputationError(
                "GEV fitting failed: could not find valid parameters for the data".into(),
            )),
        }
    }

    /// Mean of the GEV distribution (exists only for ξ < 1).
    pub fn mean(&self) -> Option<f64> {
        if self.xi >= 1.0 {
            return None;
        }
        if self.xi.abs() < XI_THRESHOLD {
            // Gumbel: μ + σ * γ where γ = Euler-Mascheroni constant
            Some(self.mu + self.sigma * 0.577_215_664_901_532_9)
        } else {
            let g1 = gamma_fn(1.0 - self.xi)?;
            Some(self.mu + self.sigma * (g1 - 1.0) / self.xi)
        }
    }

    /// Variance of the GEV distribution (exists only for ξ < 0.5).
    pub fn variance(&self) -> Option<f64> {
        if self.xi >= 0.5 {
            return None;
        }
        if self.xi.abs() < XI_THRESHOLD {
            // Gumbel: σ² * π²/6
            Some(self.sigma * self.sigma * std::f64::consts::PI * std::f64::consts::PI / 6.0)
        } else {
            let g1 = gamma_fn(1.0 - self.xi)?;
            let g2 = gamma_fn(1.0 - 2.0 * self.xi)?;
            Some(self.sigma * self.sigma * (g2 - g1 * g1) / (self.xi * self.xi))
        }
    }
}

// ---------------------------------------------------------------------------
// L-moment initial estimates
// ---------------------------------------------------------------------------

/// Initial GEV parameter estimates via probability-weighted moments (L-moments).
///
/// Uses the Hosking (1997) / Donaldson (1996) rational-function approximation
/// of the GEV shape parameter from the L-skewness ratio tau3, as implemented
/// in the Fortran LMOMENTS package (pelgev routine).
fn gev_lmoment_estimates(data: &[f64]) -> (f64, f64, f64) {
    let n = data.len();
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Probability-weighted moments b0, b1, b2
    let mut b0 = 0.0f64;
    let mut b1 = 0.0f64;
    let mut b2 = 0.0f64;

    for (i, &x) in sorted.iter().enumerate() {
        let i_f = i as f64;
        let n_f = n as f64;
        let w1 = i_f / (n_f - 1.0);
        let w2 = i_f * (i_f - 1.0) / ((n_f - 1.0) * (n_f - 2.0));
        b0 += x;
        b1 += w1 * x;
        b2 += w2 * x;
    }
    b0 /= n as f64;
    b1 /= n as f64;
    b2 /= n as f64;

    // L-moments
    let l1 = b0;
    let l2 = 2.0 * b1 - b0;
    let l3 = 6.0 * b2 - 6.0 * b1 + b0;

    // L-skewness ratio
    let tau3 = if l2.abs() > 1e-15 { l3 / l2 } else { 0.0 };

    // ── Shape parameter estimation via Hosking/Donaldson rational functions ──
    //
    // Two branches depending on the sign of τ₃:
    //   τ₃ ≤ 0  → one rational function (Donaldson 1996), with Newton-Raphson
    //              refinement when τ₃ < -0.8.
    //   τ₃ > 0  → another rational function.
    //
    // Constants from Hosking's Fortran LMOMENTS package (pelgev).
    const SMALL_G: f64 = 1e-5;
    const A0: f64 = 0.28377530;
    const A1: f64 = -1.21096399;
    const A2: f64 = -2.50728214;
    const A3: f64 = -1.13455566;
    const A4: f64 = -0.07138022;
    const B1: f64 = 2.06189696;
    const B2: f64 = 1.31912239;
    const B3: f64 = 0.25077104;
    const C1: f64 = 1.59921491;
    const C2: f64 = -0.48832213;
    const C3: f64 = 0.01573152;
    const D1: f64 = -0.64363929;
    const D2: f64 = 0.08985247;

    let g = if tau3 <= 0.0 {
        // Initial rational approximation for τ₃ ≤ 0
        let mut g_val = (A0 + tau3 * (A1 + tau3 * (A2 + tau3 * (A3 + tau3 * A4))))
            / (1.0 + tau3 * (B1 + tau3 * (B2 + tau3 * B3)));

        if tau3 < -0.8 {
            // Refine with Newton-Raphson on the exact relation:
            //   τ₃ = (1 - 3^{-g}) / (1 - 2^{-g}) - 3
            // i.e. T0 = (τ₃ + 3)/2 and we solve T(g) = (1-3^{-g})/(1-2^{-g}) = T0.
            if tau3 <= -0.97 {
                // Better starting value near the boundary
                g_val = 1.0 - (1.0 + tau3).ln() / std::f64::consts::LN_2;
            }
            let t0 = (tau3 + 3.0) * 0.5;
            let dl2 = std::f64::consts::LN_2;
            let dl3 = 3.0_f64.ln();
            for _ in 0..20 {
                let x2 = 2.0_f64.powf(-g_val);
                let x3 = 3.0_f64.powf(-g_val);
                let xx2 = 1.0 - x2;
                let xx3 = 1.0 - x3;
                let t = xx3 / xx2;
                let deriv = (xx2 * x3 * dl3 - xx3 * x2 * dl2) / (xx2 * xx2);
                let g_old = g_val;
                g_val -= (t - t0) / deriv;
                if (g_val - g_old).abs() <= 1e-6 * g_val.abs() {
                    break;
                }
            }
        }
        g_val
    } else {
        // τ₃ > 0: different rational approximation
        let z = 1.0 - tau3;
        let g_val = (-1.0 + z * (C1 + z * (C2 + z * C3))) / (1.0 + z * (D1 + z * D2));
        g_val
    };

    // If |g| is very small, treat as Gumbel (g = 0)
    let xi = if g.abs() < SMALL_G { 0.0 } else { g };

    let xi_clamped = xi.clamp(-2.0, 2.0);

    // ── Scale and location from L-moments ──
    let eu = 0.577_215_664_901_532_9; // Euler-Mascheroni constant
    let dl2 = std::f64::consts::LN_2;

    let (sigma, mu) = if xi_clamped.abs() < SMALL_G {
        // Gumbel case
        let s = l2 / dl2;
        let m = l1 - eu * s;
        (s, m)
    } else {
        let gam = gamma_fn(1.0 + xi_clamped).unwrap_or(1.0);
        let s = l2 * xi_clamped / (gam * (1.0 - 2.0_f64.powf(-xi_clamped)));
        let m = l1 - s * (1.0 - gam) / xi_clamped;
        (s, m)
    };

    let sigma_pos = if sigma.is_finite() && sigma > 0.0 {
        sigma
    } else {
        // Fallback: use sample std dev
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        var.sqrt().max(1e-8)
    };

    let mu_finite = if mu.is_finite() {
        mu
    } else {
        data.iter().cloned().fold(f64::INFINITY, f64::min)
    };

    (mu_finite, sigma_pos, xi_clamped)
}

// ---------------------------------------------------------------------------
// Nelder-Mead optimizer for GEV fitting
// ---------------------------------------------------------------------------

/// Nelder-Mead simplex minimization of negative log-likelihood for GEV.
fn nelder_mead_gev(data: &[f64], mu0: f64, sigma0: f64, xi0: f64) -> StatsResult<(f64, f64, f64)> {
    // Objective: negative log-likelihood
    let neg_ll = |params: &[f64; 3]| -> f64 {
        let (mu, sigma, xi) = (params[0], params[1], params[2]);
        if sigma <= 0.0 {
            return 1e15;
        }
        match GEV::new(mu, sigma, xi) {
            Ok(gev) => {
                let ll = gev.log_likelihood(data);
                if ll.is_finite() {
                    -ll
                } else {
                    1e15
                }
            }
            Err(_) => 1e15,
        }
    };

    // Initial simplex: 4 vertices (3 parameters)
    let s = sigma0.max(1e-6);
    let x0 = [mu0, s, xi0.clamp(-2.0, 2.0)];
    let step = [s * 0.5 + 0.1, s * 0.2 + 0.01, 0.1];

    let mut simplex: [[f64; 3]; 4] = [
        x0,
        [x0[0] + step[0], x0[1], x0[2]],
        [x0[0], x0[1] + step[1], x0[2]],
        [x0[0], x0[1], x0[2] + step[2]],
    ];

    let mut fvals: [f64; 4] = [
        neg_ll(&simplex[0]),
        neg_ll(&simplex[1]),
        neg_ll(&simplex[2]),
        neg_ll(&simplex[3]),
    ];

    let max_iter = 10_000;
    let tol = 1e-8;
    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5; // contraction
    let sigma_nm = 0.5; // shrink

    for _ in 0..max_iter {
        // Sort by function value
        let mut idx: [usize; 4] = [0, 1, 2, 3];
        idx.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check convergence
        let f_best = fvals[idx[0]];
        let f_worst = fvals[idx[3]];
        if (f_worst - f_best).abs() < tol {
            break;
        }

        // Centroid of best 3
        let mut centroid = [0.0f64; 3];
        for i in 0..3 {
            for j in 0..3 {
                centroid[j] += simplex[idx[i]][j];
            }
        }
        for j in 0..3 {
            centroid[j] /= 3.0;
        }

        let worst = simplex[idx[3]];
        let f_worst_val = fvals[idx[3]];
        let f_second_worst = fvals[idx[2]];

        // Reflection
        let mut reflected = [0.0f64; 3];
        for j in 0..3 {
            reflected[j] = centroid[j] + alpha * (centroid[j] - worst[j]);
        }
        let f_reflected = neg_ll(&reflected);

        if f_reflected < f_best {
            // Try expansion
            let mut expanded = [0.0f64; 3];
            for j in 0..3 {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            let f_expanded = neg_ll(&expanded);
            if f_expanded < f_reflected {
                simplex[idx[3]] = expanded;
                fvals[idx[3]] = f_expanded;
            } else {
                simplex[idx[3]] = reflected;
                fvals[idx[3]] = f_reflected;
            }
        } else if f_reflected < f_second_worst {
            simplex[idx[3]] = reflected;
            fvals[idx[3]] = f_reflected;
        } else {
            // Contraction
            let use_reflected = f_reflected < f_worst_val;
            let contraction_point = if use_reflected { reflected } else { worst };
            let mut contracted = [0.0f64; 3];
            for j in 0..3 {
                contracted[j] = centroid[j] + rho * (contraction_point[j] - centroid[j]);
            }
            let f_contracted = neg_ll(&contracted);

            if f_contracted
                < (if use_reflected {
                    f_reflected
                } else {
                    f_worst_val
                })
            {
                simplex[idx[3]] = contracted;
                fvals[idx[3]] = f_contracted;
            } else {
                // Shrink
                let best_vertex = simplex[idx[0]];
                for i in 1..4 {
                    for j in 0..3 {
                        simplex[idx[i]][j] =
                            best_vertex[j] + sigma_nm * (simplex[idx[i]][j] - best_vertex[j]);
                    }
                    fvals[idx[i]] = neg_ll(&simplex[idx[i]]);
                }
            }
        }
    }

    // Find best vertex
    let best_idx = (0..4)
        .min_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let best = simplex[best_idx];
    if best[1] <= 0.0 {
        return Err(StatsError::ComputationError(
            "GEV fitting converged to non-positive scale".into(),
        ));
    }

    // Validate that the best solution gives a finite log-likelihood
    if fvals[best_idx] >= 1e14 {
        // The optimizer didn't find a valid solution; fall back to L-moment estimates
        // or try with Gumbel (xi=0) as a fallback
        let gumbel_candidate = [best[0], best[1], 0.0];
        let gumbel_val = neg_ll(&gumbel_candidate);
        if gumbel_val < 1e14 {
            return Ok((
                gumbel_candidate[0],
                gumbel_candidate[1],
                gumbel_candidate[2],
            ));
        }
        // Try with original L-moment estimates and xi clamped closer to 0
        let fallback = [mu0, sigma0.max(1e-6), 0.0];
        let fallback_val = neg_ll(&fallback);
        if fallback_val < 1e14 {
            return Ok((fallback[0], fallback[1], fallback[2]));
        }
        return Err(StatsError::ComputationError(
            "GEV fitting failed: no valid parameter combination found".into(),
        ));
    }

    Ok((best[0], best[1], best[2]))
}

// ---------------------------------------------------------------------------
// Gamma function approximation (Lanczos)
// ---------------------------------------------------------------------------

/// Lanczos approximation of the gamma function for x > 0.
pub(crate) fn gamma_fn(x: f64) -> Option<f64> {
    if x <= 0.0 {
        return None;
    }
    if x < 0.5 {
        // Reflection formula: Γ(x) = π / (sin(πx) * Γ(1-x))
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-300 {
            return None;
        }
        let gamma_1mx = gamma_fn(1.0 - x)?;
        return Some(std::f64::consts::PI / (sin_pi_x * gamma_1mx));
    }

    // Lanczos coefficients (g=7)
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_3,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let x_adj = x - 1.0;
    let mut sum = c[0];
    for (i, &ci) in c.iter().enumerate().skip(1) {
        sum += ci / (x_adj + i as f64);
    }

    let t = x_adj + g + 0.5;
    let val = (2.0 * std::f64::consts::PI).sqrt() * t.powf(x_adj + 0.5) * (-t).exp() * sum;

    if val.is_finite() && val > 0.0 {
        Some(val)
    } else {
        None
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
    fn test_gev_new_invalid_sigma() {
        assert!(GEV::new(0.0, 0.0, 0.0).is_err());
        assert!(GEV::new(0.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_gev_gumbel_pdf_at_mode() {
        // Mode of Gumbel(0,1) is at x=0, pdf = exp(-1)
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        let pdf = g.pdf(0.0);
        assert!(approx_eq(pdf, (-1.0_f64).exp(), 1e-8), "pdf={pdf}");
    }

    #[test]
    fn test_gev_gumbel_cdf() {
        // F(0) = exp(-exp(0)) = exp(-1) for Gumbel(0,1)
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        assert!(approx_eq(g.cdf(0.0), (-1.0_f64).exp(), 1e-8));
    }

    #[test]
    fn test_gev_quantile_inverse_of_cdf() {
        let g = GEV::new(1.0, 2.0, 0.1).unwrap();
        for &p in &[0.1, 0.5, 0.9, 0.99] {
            let q = g.quantile(p).unwrap();
            let cdf_q = g.cdf(q);
            assert!(approx_eq(cdf_q, p, 1e-8), "p={p}, q={q}, cdf(q)={cdf_q}");
        }
    }

    #[test]
    fn test_gev_return_level() {
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        let rl = g.return_level(100.0).unwrap();
        assert!(rl.is_finite());
        assert!(rl > g.quantile(0.5).unwrap());
    }

    #[test]
    fn test_gev_return_level_invalid_period() {
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        assert!(g.return_level(1.0).is_err());
        assert!(g.return_level(0.5).is_err());
    }

    #[test]
    fn test_gev_fit_basic() {
        // Generate Gumbel(5, 2) data
        let data: Vec<f64> = (0..200)
            .map(|i| {
                let u = (i as f64 + 0.5) / 200.0;
                5.0 - 2.0 * (-u.ln()).ln()
            })
            .collect();
        let (gev, ll) = GEV::fit(&data).unwrap();
        assert!(gev.sigma > 0.0);
        assert!(ll.is_finite());
        // Parameters should be roughly close to truth
        assert!((gev.mu - 5.0).abs() < 2.0, "mu={}", gev.mu);
    }

    #[test]
    fn test_gev_fit_insufficient_data() {
        assert!(GEV::fit(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_gev_log_likelihood_empty() {
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        assert_eq!(g.log_likelihood(&[]), f64::NEG_INFINITY);
    }

    #[test]
    fn test_gev_mean_gumbel() {
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        let m = g.mean().unwrap();
        // Gumbel mean = μ + σ*γ ≈ 0.5772
        assert!(approx_eq(m, 0.577_215_664_901_532_9, 1e-8));
    }

    #[test]
    fn test_gev_mean_undefined_for_heavy_tail() {
        let g = GEV::new(0.0, 1.0, 1.5).unwrap();
        assert!(g.mean().is_none());
    }

    #[test]
    fn test_gev_variance_gumbel() {
        let g = GEV::new(0.0, 1.0, 0.0).unwrap();
        let v = g.variance().unwrap();
        // Gumbel variance = π²/6 ≈ 1.6449
        assert!(approx_eq(v, std::f64::consts::PI.powi(2) / 6.0, 1e-8));
    }

    #[test]
    fn test_gamma_fn_known_values() {
        assert!(approx_eq(gamma_fn(1.0).unwrap(), 1.0, 1e-8));
        assert!(approx_eq(gamma_fn(2.0).unwrap(), 1.0, 1e-8));
        assert!(approx_eq(gamma_fn(3.0).unwrap(), 2.0, 1e-8));
        assert!(approx_eq(
            gamma_fn(0.5).unwrap(),
            std::f64::consts::PI.sqrt(),
            1e-7
        ));
    }
}
