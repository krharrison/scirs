//! Extreme Value distributions: GEV, Gumbel, Fréchet, and Generalized Pareto (GPD).
//!
//! # Mathematical Background
//!
//! The **Generalized Extreme Value** (GEV) distribution unifies three classical extreme value types:
//! - ξ > 0: Fréchet (heavy-tailed)
//! - ξ = 0: Gumbel (light-tailed / exponential tail)
//! - ξ < 0: Weibull / reversed Weibull (bounded tail)
//!
//! The **Generalized Pareto Distribution** (GPD) models exceedances above a threshold and arises
//! naturally as the limiting distribution of excesses in the Peaks Over Threshold (POT) framework.

use crate::error::StatsError;
use std::f64::consts::PI;

/// Euler–Mascheroni constant γ ≈ 0.5772156649…
pub const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

/// Threshold below which |ξ| is treated as zero (Gumbel case) to avoid numerical issues.
const XI_THRESHOLD: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Evaluate the GEV/GPD bracket term `t(x) = 1 + ξ*(x - μ)/σ`.
/// Returns `None` when the value is non-positive (outside support).
#[inline]
fn gev_bracket(x: f64, mu: f64, sigma: f64, xi: f64) -> Option<f64> {
    let t = 1.0 + xi * (x - mu) / sigma;
    if t <= 0.0 {
        None
    } else {
        Some(t)
    }
}

/// Simple linear congruential generator – used only for sampling, not cryptographic purposes.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Returns a value uniformly distributed in (0, 1) (exclusive on both ends).
    fn next_f64(&mut self) -> f64 {
        // Constants from Knuth / Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.state >> 11) as f64;
        // Map to (0, 1) – exclude exact 0 and 1 to keep log/quantile happy
        (bits + 0.5) / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// Generalized Extreme Value (GEV)
// ---------------------------------------------------------------------------

/// Generalized Extreme Value (GEV) distribution.
///
/// The CDF is:
/// - ξ ≠ 0: F(x) = exp(−\[1 + ξ(x−μ)/σ\]^{−1/ξ})  for 1 + ξ(x−μ)/σ > 0
/// - ξ = 0: F(x) = exp(−exp(−(x−μ)/σ))  (Gumbel)
///
/// # Parameters
/// - `mu`: location parameter (−∞ < μ < ∞)
/// - `sigma`: scale parameter (σ > 0)
/// - `xi`: shape parameter (−∞ < ξ < ∞)
#[derive(Debug, Clone)]
pub struct GeneralizedExtremeValue {
    /// Location parameter μ.
    pub mu: f64,
    /// Scale parameter σ > 0.
    pub sigma: f64,
    /// Shape parameter ξ.  ξ > 0 ⇒ Fréchet, ξ < 0 ⇒ reversed Weibull, ξ = 0 ⇒ Gumbel.
    pub xi: f64,
}

impl GeneralizedExtremeValue {
    /// Construct a new GEV distribution.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `sigma <= 0`.
    pub fn new(mu: f64, sigma: f64, xi: f64) -> Result<Self, StatsError> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "GEV scale parameter sigma must be positive, got {sigma}"
            )));
        }
        Ok(Self { mu, sigma, xi })
    }

    /// Probability density function (PDF).
    ///
    /// Returns 0.0 for values outside the support.
    pub fn pdf(&self, x: f64) -> f64 {
        let xi = self.xi;
        let sigma = self.sigma;
        let mu = self.mu;

        if xi.abs() < XI_THRESHOLD {
            // Gumbel case
            let z = (x - mu) / sigma;
            (1.0 / sigma) * (-z - (-z).exp()).exp()
        } else {
            match gev_bracket(x, mu, sigma, xi) {
                None => 0.0,
                Some(t) => {
                    let log_t = t.ln();
                    let exponent = -(1.0 + 1.0 / xi) * log_t - t.powf(-1.0 / xi);
                    (1.0 / sigma) * exponent.exp()
                }
            }
        }
    }

    /// Cumulative distribution function (CDF).
    ///
    /// Returns 0.0 below support, 1.0 above.
    pub fn cdf(&self, x: f64) -> f64 {
        let xi = self.xi;
        let sigma = self.sigma;
        let mu = self.mu;

        if xi.abs() < XI_THRESHOLD {
            // Gumbel
            let z = (x - mu) / sigma;
            (-(-z).exp()).exp()
        } else {
            match gev_bracket(x, mu, sigma, xi) {
                None => {
                    if xi > 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
                Some(t) => (-t.powf(-1.0 / xi)).exp(),
            }
        }
    }

    /// Quantile function (inverse CDF).
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `p` is not in (0, 1).
    pub fn quantile(&self, p: f64) -> Result<f64, StatsError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Probability p must be in (0, 1), got {p}"
            )));
        }
        let log_p = -p.ln();
        let xi = self.xi;
        let mu = self.mu;
        let sigma = self.sigma;

        let q = if xi.abs() < XI_THRESHOLD {
            mu - sigma * log_p.ln()
        } else {
            mu + sigma * (log_p.powf(-xi) - 1.0) / xi
        };
        Ok(q)
    }

    /// Mean of the distribution.
    ///
    /// Returns `None` when ξ ≥ 1 (mean is undefined) or when the Gamma function
    /// argument would be invalid.
    pub fn mean(&self) -> Option<f64> {
        let xi = self.xi;
        if xi >= 1.0 {
            return None;
        }
        if xi.abs() < XI_THRESHOLD {
            // Gumbel: μ + σ * γ
            Some(self.mu + self.sigma * EULER_MASCHERONI)
        } else {
            // μ + σ * (Γ(1−ξ) − 1) / ξ
            let g = gamma_approx(1.0 - xi)?;
            Some(self.mu + self.sigma * (g - 1.0) / xi)
        }
    }

    /// Variance of the distribution.
    ///
    /// Returns `None` when ξ ≥ 0.5 (variance is undefined).
    pub fn variance(&self) -> Option<f64> {
        let xi = self.xi;
        if xi >= 0.5 {
            return None;
        }
        if xi.abs() < XI_THRESHOLD {
            // Gumbel: σ² * π² / 6
            Some(self.sigma.powi(2) * PI * PI / 6.0)
        } else {
            let g1 = gamma_approx(1.0 - xi)?;
            let g2 = gamma_approx(1.0 - 2.0 * xi)?;
            Some(self.sigma.powi(2) * (g2 - g1.powi(2)) / xi.powi(2))
        }
    }

    /// Return level for a given return period T (in the same units as the block size).
    ///
    /// The return level x_T satisfies F(x_T) = 1 − 1/T, i.e. q = 1 − 1/T.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `return_period <= 1`.
    pub fn return_level(&self, return_period: f64) -> Result<f64, StatsError> {
        if return_period <= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Return period must be > 1, got {return_period}"
            )));
        }
        let p = 1.0 - 1.0 / return_period;
        self.quantile(p)
    }

    /// Generate `n` random samples using inverse-transform sampling.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .filter_map(|_| {
                let u = rng.next_f64();
                self.quantile(u).ok()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Gumbel (Extreme Value Type I)
// ---------------------------------------------------------------------------

/// Gumbel (Extreme Value Type I) distribution.
///
/// This is a special case of the GEV with ξ = 0:
/// - CDF: F(x) = exp(−exp(−(x−μ)/β))
/// - PDF: f(x) = (1/β) exp(−(x−μ)/β − exp(−(x−μ)/β))
///
/// Commonly used to model maximum of many independent RVs with exponential-like tails
/// (e.g. extreme rainfall, wind speeds, flood levels).
#[derive(Debug, Clone)]
pub struct Gumbel {
    /// Location parameter μ.
    pub mu: f64,
    /// Scale parameter β > 0.
    pub beta: f64,
}

impl Gumbel {
    /// Construct a new Gumbel distribution.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `beta <= 0`.
    pub fn new(mu: f64, beta: f64) -> Result<Self, StatsError> {
        if beta <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Gumbel scale parameter beta must be positive, got {beta}"
            )));
        }
        Ok(Self { mu, beta })
    }

    /// PDF of the Gumbel distribution.
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.beta;
        (1.0 / self.beta) * (-z - (-z).exp()).exp()
    }

    /// CDF of the Gumbel distribution.
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.beta;
        (-(-z).exp()).exp()
    }

    /// Quantile function (inverse CDF); defined for all p ∈ (0, 1).
    pub fn quantile(&self, p: f64) -> f64 {
        // x = μ − β * ln(−ln(p))
        self.mu - self.beta * (-(p.ln())).ln()
    }

    /// Mean: μ + β * γ where γ is the Euler–Mascheroni constant.
    pub fn mean(&self) -> f64 {
        self.mu + self.beta * EULER_MASCHERONI
    }

    /// Variance: β² * π² / 6.
    pub fn variance(&self) -> f64 {
        self.beta.powi(2) * PI * PI / 6.0
    }

    /// Generate `n` random samples.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| {
                let u = rng.next_f64();
                self.quantile(u)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Fréchet (Extreme Value Type II)
// ---------------------------------------------------------------------------

/// Fréchet (Extreme Value Type II) distribution.
///
/// This corresponds to GEV with ξ > 0, reparameterized as:
/// - F(x) = exp(−((x − m)/s)^{−α})  for x > m
///
/// It has heavy polynomial tails and is used for modelling returns in finance,
/// earthquake magnitudes, or annual maximum wind speeds.
#[derive(Debug, Clone)]
pub struct Frechet {
    /// Shape parameter α > 0.  The tail index ξ = 1/α in GEV notation.
    pub alpha: f64,
    /// Scale parameter s > 0.
    pub s: f64,
    /// Location (threshold) parameter m.
    pub m: f64,
}

impl Frechet {
    /// Construct a new Fréchet distribution.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `alpha <= 0` or `s <= 0`.
    pub fn new(alpha: f64, s: f64, m: f64) -> Result<Self, StatsError> {
        if alpha <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Fréchet shape alpha must be positive, got {alpha}"
            )));
        }
        if s <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Fréchet scale s must be positive, got {s}"
            )));
        }
        Ok(Self { alpha, s, m })
    }

    /// PDF of the Fréchet distribution (0 for x ≤ m).
    pub fn pdf(&self, x: f64) -> f64 {
        if x <= self.m {
            return 0.0;
        }
        let z = (x - self.m) / self.s;
        (self.alpha / self.s) * z.powf(-self.alpha - 1.0) * (-z.powf(-self.alpha)).exp()
    }

    /// CDF of the Fréchet distribution.
    pub fn cdf(&self, x: f64) -> f64 {
        if x <= self.m {
            return 0.0;
        }
        let z = (x - self.m) / self.s;
        (-z.powf(-self.alpha)).exp()
    }

    /// Quantile function.
    pub fn quantile(&self, p: f64) -> f64 {
        // x = m + s * (−ln p)^{−1/α}
        self.m + self.s * (-(p.ln())).powf(-1.0 / self.alpha)
    }

    /// Mean: m + s * Γ(1 − 1/α).
    ///
    /// Returns `None` when α ≤ 1 (mean is infinite).
    pub fn mean(&self) -> Option<f64> {
        if self.alpha <= 1.0 {
            return None;
        }
        let g = gamma_approx(1.0 - 1.0 / self.alpha)?;
        Some(self.m + self.s * g)
    }
}

// ---------------------------------------------------------------------------
// Generalized Pareto Distribution (GPD)
// ---------------------------------------------------------------------------

/// Generalized Pareto Distribution (GPD).
///
/// Models exceedances above a threshold μ.  The CDF is:
/// - ξ ≠ 0: F(x) = 1 − (1 + ξ(x−μ)/σ)^{−1/ξ}
/// - ξ = 0: F(x) = 1 − exp(−(x−μ)/σ)  (Exponential)
///
/// Support: x ≥ μ when ξ ≥ 0; μ ≤ x ≤ μ − σ/ξ when ξ < 0.
#[derive(Debug, Clone)]
pub struct GeneralizedPareto {
    /// Threshold (location) μ.
    pub mu: f64,
    /// Scale parameter σ > 0.
    pub sigma: f64,
    /// Shape parameter ξ.
    pub xi: f64,
}

impl GeneralizedPareto {
    /// Construct a new GPD.
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `sigma <= 0`.
    pub fn new(mu: f64, sigma: f64, xi: f64) -> Result<Self, StatsError> {
        if sigma <= 0.0 {
            return Err(StatsError::InvalidArgument(format!(
                "GPD scale parameter sigma must be positive, got {sigma}"
            )));
        }
        Ok(Self { mu, sigma, xi })
    }

    /// PDF of the GPD (0 outside support).
    pub fn pdf(&self, x: f64) -> f64 {
        if x < self.mu {
            return 0.0;
        }
        let xi = self.xi;
        let sigma = self.sigma;
        let mu = self.mu;

        if xi.abs() < XI_THRESHOLD {
            // Exponential
            let z = (x - mu) / sigma;
            (1.0 / sigma) * (-z).exp()
        } else {
            match gev_bracket(x, mu, sigma, xi) {
                None => 0.0,
                Some(t) => (1.0 / sigma) * t.powf(-1.0 / xi - 1.0),
            }
        }
    }

    /// CDF of the GPD.
    pub fn cdf(&self, x: f64) -> f64 {
        if x < self.mu {
            return 0.0;
        }
        let xi = self.xi;
        let sigma = self.sigma;
        let mu = self.mu;

        if xi.abs() < XI_THRESHOLD {
            let z = (x - mu) / sigma;
            1.0 - (-z).exp()
        } else {
            match gev_bracket(x, mu, sigma, xi) {
                None => {
                    if xi < 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                Some(t) => 1.0 - t.powf(-1.0 / xi),
            }
        }
    }

    /// Quantile function (p ∈ (0, 1)).
    ///
    /// # Errors
    /// Returns [`StatsError::InvalidArgument`] if `p` is not in (0, 1).
    pub fn quantile(&self, p: f64) -> Result<f64, StatsError> {
        if p <= 0.0 || p >= 1.0 {
            return Err(StatsError::InvalidArgument(format!(
                "Probability p must be in (0, 1), got {p}"
            )));
        }
        let xi = self.xi;
        let sigma = self.sigma;
        let mu = self.mu;

        let q = if xi.abs() < XI_THRESHOLD {
            mu - sigma * (1.0 - p).ln()
        } else {
            mu + sigma * ((1.0 - p).powf(-xi) - 1.0) / xi
        };
        Ok(q)
    }

    /// Mean of the GPD: μ + σ/(1−ξ).
    ///
    /// Returns `None` when ξ ≥ 1 (mean is undefined).
    pub fn mean(&self) -> Option<f64> {
        if self.xi >= 1.0 {
            None
        } else {
            Some(self.mu + self.sigma / (1.0 - self.xi))
        }
    }

    /// Variance of the GPD: σ²/((1−ξ)²(1−2ξ)).
    ///
    /// Returns `None` when ξ ≥ 0.5 (variance is undefined).
    pub fn variance(&self) -> Option<f64> {
        if self.xi >= 0.5 {
            None
        } else {
            let denom = (1.0 - self.xi).powi(2) * (1.0 - 2.0 * self.xi);
            Some(self.sigma.powi(2) / denom)
        }
    }

    /// Quantile of the *exceedance distribution* above the threshold.
    ///
    /// Equivalent to [`GeneralizedPareto::quantile`] since the GPD is already parameterized as an
    /// exceedance distribution, but provided for API clarity.
    pub fn exceedance_quantile(&self, p: f64) -> Result<f64, StatsError> {
        self.quantile(p)
    }

    /// Generate `n` random samples using inverse-transform sampling.
    pub fn sample(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .filter_map(|_| {
                let u = rng.next_f64();
                self.quantile(u).ok()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Gamma function approximation (Lanczos)
// ---------------------------------------------------------------------------

/// Lanczos approximation of the Gamma function for positive real arguments.
///
/// Accurate to ~15 significant figures for Re(z) > 0.5.
/// Returns `None` only when the argument would cause arithmetic failure.
pub(crate) fn gamma_approx(z: f64) -> Option<f64> {
    if z <= 0.0 {
        return None;
    }
    // Lanczos coefficients (g=7, n=9)
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_10,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let mut z = z;
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        let sin_piz = (PI * z).sin();
        let g = gamma_approx(1.0 - z)?;
        return Some(PI / (sin_piz * g));
    }
    z -= 1.0;

    let mut x = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        x += c / (z + i as f64 + 1.0);
    }

    let t = z + G + 0.5;
    let result = (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x;
    Some(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: absolute tolerance check
    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- GEV ---------------------------------------------------------------

    #[test]
    fn test_gev_new_invalid_sigma() {
        assert!(GeneralizedExtremeValue::new(0.0, 0.0, 0.0).is_err());
        assert!(GeneralizedExtremeValue::new(0.0, -1.0, 0.0).is_err());
    }

    #[test]
    fn test_gev_new_valid() {
        assert!(GeneralizedExtremeValue::new(0.0, 1.0, 0.0).is_ok());
        assert!(GeneralizedExtremeValue::new(0.0, 1.0, 0.5).is_ok());
        assert!(GeneralizedExtremeValue::new(0.0, 1.0, -0.5).is_ok());
    }

    #[test]
    fn test_gev_gumbel_case_cdf() {
        // Gumbel special case (ξ=0): F(0) = exp(-1) ≈ 0.3679
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        assert!(approx_eq(gev.cdf(0.0), 0.367_879_441, 1e-8));
    }

    #[test]
    fn test_gev_cdf_pdf_consistency() {
        // Numerical derivative of CDF ≈ PDF
        let gev = GeneralizedExtremeValue::new(2.0, 1.5, 0.3).unwrap();
        let h = 1e-5;
        let x = 3.0_f64;
        let numerical_pdf = (gev.cdf(x + h) - gev.cdf(x - h)) / (2.0 * h);
        assert!(approx_eq(numerical_pdf, gev.pdf(x), 1e-4));
    }

    #[test]
    fn test_gev_quantile_inverse_cdf() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.2).unwrap();
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let q = gev.quantile(p).unwrap();
            let cdf_q = gev.cdf(q);
            assert!(
                approx_eq(cdf_q, p, 1e-10),
                "p={p}, quantile={q}, CDF(quantile)={cdf_q}"
            );
        }
    }

    #[test]
    fn test_gev_quantile_invalid_p() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        assert!(gev.quantile(0.0).is_err());
        assert!(gev.quantile(1.0).is_err());
        assert!(gev.quantile(-0.1).is_err());
    }

    #[test]
    fn test_gev_mean_gumbel() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        let m = gev.mean().unwrap();
        assert!(approx_eq(m, EULER_MASCHERONI, 1e-10));
    }

    #[test]
    fn test_gev_variance_gumbel() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        let v = gev.variance().unwrap();
        let expected = PI * PI / 6.0;
        assert!(approx_eq(v, expected, 1e-10));
    }

    #[test]
    fn test_gev_mean_undefined_for_large_xi() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 1.5).unwrap();
        assert!(gev.mean().is_none());
    }

    #[test]
    fn test_gev_return_level() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        // 100-year return level: p = 0.99
        let rl = gev.return_level(100.0).unwrap();
        let expected = gev.quantile(0.99).unwrap();
        assert!(approx_eq(rl, expected, 1e-12));
    }

    #[test]
    fn test_gev_return_level_invalid() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        assert!(gev.return_level(0.5).is_err());
        assert!(gev.return_level(1.0).is_err());
    }

    #[test]
    fn test_gev_sample_length() {
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.0).unwrap();
        let s = gev.sample(50, 42);
        assert_eq!(s.len(), 50);
    }

    #[test]
    fn test_gev_frechet_support() {
        // Fréchet: ξ > 0, lower bound at μ − σ/ξ
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, 0.5).unwrap();
        // x = -2.1 is below lower bound (-2.0), CDF should be 0
        let below = gev.cdf(-2.1);
        assert_eq!(below, 0.0);
    }

    #[test]
    fn test_gev_weibull_support() {
        // Weibull (ξ < 0): upper bound at μ − σ/ξ
        let gev = GeneralizedExtremeValue::new(0.0, 1.0, -0.5).unwrap();
        // upper bound = 0 + 1/0.5 = 2.0
        let above = gev.cdf(2.1);
        assert_eq!(above, 1.0);
    }

    // ---- Gumbel ------------------------------------------------------------

    #[test]
    fn test_gumbel_new_invalid() {
        assert!(Gumbel::new(0.0, 0.0).is_err());
        assert!(Gumbel::new(0.0, -1.0).is_err());
    }

    #[test]
    fn test_gumbel_cdf_at_mu() {
        // F(μ) = exp(-exp(0)) = exp(-1) ≈ 0.3679
        let g = Gumbel::new(5.0, 2.0).unwrap();
        assert!(approx_eq(g.cdf(5.0), (-1.0_f64).exp(), 1e-10));
    }

    #[test]
    fn test_gumbel_quantile_roundtrip() {
        let g = Gumbel::new(1.0, 2.0).unwrap();
        for &p in &[0.05, 0.5, 0.95] {
            let q = g.quantile(p);
            assert!(approx_eq(g.cdf(q), p, 1e-12));
        }
    }

    #[test]
    fn test_gumbel_mean_variance() {
        let g = Gumbel::new(0.0, 1.0).unwrap();
        assert!(approx_eq(g.mean(), EULER_MASCHERONI, 1e-10));
        assert!(approx_eq(g.variance(), PI * PI / 6.0, 1e-10));
    }

    #[test]
    fn test_gumbel_sample() {
        let g = Gumbel::new(0.0, 1.0).unwrap();
        let s = g.sample(100, 7);
        assert_eq!(s.len(), 100);
        // All samples should be finite
        assert!(s.iter().all(|x| x.is_finite()));
    }

    // ---- Fréchet -----------------------------------------------------------

    #[test]
    fn test_frechet_new_invalid() {
        assert!(Frechet::new(0.0, 1.0, 0.0).is_err()); // alpha <= 0
        assert!(Frechet::new(2.0, 0.0, 0.0).is_err()); // s <= 0
    }

    #[test]
    fn test_frechet_cdf_below_location() {
        let f = Frechet::new(2.0, 1.0, 3.0).unwrap();
        assert_eq!(f.cdf(3.0), 0.0);
        assert_eq!(f.cdf(2.9), 0.0);
    }

    #[test]
    fn test_frechet_pdf_positive() {
        let f = Frechet::new(2.0, 1.0, 0.0).unwrap();
        let p = f.pdf(2.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_frechet_mean_undefined_alpha_le_1() {
        let f = Frechet::new(0.5, 1.0, 0.0).unwrap();
        assert!(f.mean().is_none());
        let f2 = Frechet::new(1.0, 1.0, 0.0).unwrap();
        assert!(f2.mean().is_none());
    }

    #[test]
    fn test_frechet_mean_defined() {
        let f = Frechet::new(2.0, 1.0, 0.0).unwrap();
        let m = f.mean();
        assert!(m.is_some());
        assert!(m.unwrap() > 0.0);
    }

    // ---- GPD ---------------------------------------------------------------

    #[test]
    fn test_gpd_new_invalid_sigma() {
        assert!(GeneralizedPareto::new(0.0, 0.0, 0.0).is_err());
        assert!(GeneralizedPareto::new(0.0, -1.0, 0.5).is_err());
    }

    #[test]
    fn test_gpd_exponential_case() {
        // ξ=0: exponential with scale σ
        let g = GeneralizedPareto::new(0.0, 2.0, 0.0).unwrap();
        let x = 3.0_f64;
        let expected_cdf = 1.0 - (-x / 2.0).exp();
        assert!(approx_eq(g.cdf(x), expected_cdf, 1e-12));
    }

    #[test]
    fn test_gpd_quantile_inverse() {
        let g = GeneralizedPareto::new(0.0, 1.5, 0.2).unwrap();
        for &p in &[0.1, 0.5, 0.9] {
            let q = g.quantile(p).unwrap();
            assert!(approx_eq(g.cdf(q), p, 1e-10), "p={p}");
        }
    }

    #[test]
    fn test_gpd_mean() {
        let g = GeneralizedPareto::new(0.0, 2.0, 0.3).unwrap();
        let m = g.mean().unwrap();
        let expected = 0.0 + 2.0 / (1.0 - 0.3);
        assert!(approx_eq(m, expected, 1e-12));
    }

    #[test]
    fn test_gpd_mean_undefined() {
        let g = GeneralizedPareto::new(0.0, 1.0, 1.0).unwrap();
        assert!(g.mean().is_none());
    }

    #[test]
    fn test_gpd_variance_undefined() {
        let g = GeneralizedPareto::new(0.0, 1.0, 0.5).unwrap();
        assert!(g.variance().is_none());
    }

    #[test]
    fn test_gpd_variance_defined() {
        let g = GeneralizedPareto::new(0.0, 2.0, 0.0).unwrap();
        let v = g.variance().unwrap();
        // Exponential variance: σ²
        assert!(approx_eq(v, 4.0, 1e-10));
    }

    #[test]
    fn test_gpd_sample() {
        let g = GeneralizedPareto::new(0.0, 1.0, 0.1).unwrap();
        let s = g.sample(200, 13);
        assert_eq!(s.len(), 200);
        assert!(s.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_gpd_bounded_support_negative_xi() {
        // ξ < 0: support is [μ, μ - σ/ξ]
        // upper = 0 - 1.0/(-0.5) = 2.0
        let g = GeneralizedPareto::new(0.0, 1.0, -0.5).unwrap();
        assert_eq!(g.cdf(2.1), 1.0); // beyond upper bound
        assert!(approx_eq(g.cdf(2.0), 1.0, 1e-10));
    }

    #[test]
    fn test_gamma_approx_known_values() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(0.5) = √π
        assert!(approx_eq(gamma_approx(1.0).unwrap(), 1.0, 1e-10));
        assert!(approx_eq(gamma_approx(2.0).unwrap(), 1.0, 1e-10));
        assert!(approx_eq(gamma_approx(3.0).unwrap(), 2.0, 1e-10));
        assert!(approx_eq(gamma_approx(0.5).unwrap(), PI.sqrt(), 1e-10));
    }
}
