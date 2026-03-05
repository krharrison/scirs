//! Conjugate hyperprior distributions for Bayesian hierarchical models.
//!
//! Implements:
//! - `NormalInverseGamma`: conjugate prior for (μ, σ²) in Normal model
//! - `NormalInverseWishart`: multivariate conjugate prior for (μ, Σ) in MVN
//! - `HyperPrior`: trait for hierarchical hyperprior distributions

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::random::{rngs::StdRng, Beta as RandBeta, Distribution, Gamma, Normal, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// HyperPrior trait
// ---------------------------------------------------------------------------

/// Trait for hyperprior distributions used in hierarchical models.
pub trait HyperPrior: Clone + std::fmt::Debug {
    /// Return the log normalizing constant (log marginal likelihood contribution).
    fn log_norm_const(&self) -> f64;
    /// Return the name / identifier of this hyperprior family.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// NormalInverseGamma
// ---------------------------------------------------------------------------

/// Normal-Inverse-Gamma distribution: conjugate prior for (μ, σ²) in
/// the Normal model with unknown mean and variance.
///
/// Parameterization:
/// ```text
///   σ²  ~ InvGamma(α₀, β₀)
///   μ | σ²  ~ Normal(μ₀, σ²/κ₀)
/// ```
///
/// This is the standard conjugate prior for the Bayesian normal model
/// y_i ~ N(μ, σ²) with both parameters unknown.
#[derive(Debug, Clone)]
pub struct NormalInverseGamma {
    /// Prior mean for μ.
    pub mu0: f64,
    /// Prior precision factor (number of pseudo-observations).
    pub kappa0: f64,
    /// Shape parameter of the inverse-gamma prior on σ².
    pub alpha0: f64,
    /// Scale parameter of the inverse-gamma prior on σ².
    pub beta0: f64,
}

impl NormalInverseGamma {
    /// Construct a new `NormalInverseGamma` hyperprior.
    ///
    /// # Parameters
    /// - `mu0`: prior mean for μ
    /// - `kappa0`: prior precision factor (> 0)
    /// - `alpha0`: shape of InvGamma on σ² (> 0)
    /// - `beta0`: scale of InvGamma on σ² (> 0)
    ///
    /// # Errors
    /// Returns `StatsError::DomainError` when any parameter is non-positive.
    pub fn new(mu0: f64, kappa0: f64, alpha0: f64, beta0: f64) -> Result<Self> {
        if kappa0 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "kappa0 must be > 0, got {kappa0}"
            )));
        }
        if alpha0 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "alpha0 must be > 0, got {alpha0}"
            )));
        }
        if beta0 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "beta0 must be > 0, got {beta0}"
            )));
        }
        if !mu0.is_finite() {
            return Err(StatsError::DomainError(format!(
                "mu0 must be finite, got {mu0}"
            )));
        }
        Ok(Self { mu0, kappa0, alpha0, beta0 })
    }

    /// Conjugate Bayesian update: return updated NIG hyperparameters
    /// given `n` i.i.d. observations `obs ~ N(μ, σ²)`.
    ///
    /// The sufficient statistics are `n`, `x̄ = mean(obs)`, and
    /// `S = Σ(obs_i - x̄)²`.
    pub fn update(&self, obs: &[f64]) -> Result<Self> {
        let n = obs.len();
        if n == 0 {
            return Ok(self.clone());
        }
        let n_f = n as f64;
        let x_bar = obs.iter().sum::<f64>() / n_f;
        let s: f64 = obs.iter().map(|&x| (x - x_bar).powi(2)).sum();

        let kappa_n = self.kappa0 + n_f;
        let mu_n = (self.kappa0 * self.mu0 + n_f * x_bar) / kappa_n;
        let alpha_n = self.alpha0 + n_f / 2.0;
        let beta_n = self.beta0
            + 0.5 * s
            + 0.5 * (self.kappa0 * n_f / kappa_n) * (x_bar - self.mu0).powi(2);

        Self::new(mu_n, kappa_n, alpha_n, beta_n)
    }

    /// Log marginal likelihood (model evidence) for observations `obs`
    /// under this NIG prior.
    ///
    /// Uses the closed-form NIG marginal:
    /// ```text
    /// log p(obs | NIG) = log Γ(αₙ) - log Γ(α₀)
    ///                   + α₀ log β₀ - αₙ log βₙ
    ///                   + 0.5 log(κ₀/κₙ)
    ///                   - n/2 log(2π)
    /// ```
    pub fn log_marginal_likelihood(&self, obs: &[f64]) -> Result<f64> {
        let n = obs.len();
        if n == 0 {
            return Ok(0.0);
        }
        let post = self.update(obs)?;
        let n_f = n as f64;

        let log_ml = lgamma(post.alpha0) - lgamma(self.alpha0)
            + self.alpha0 * self.beta0.ln()
            - post.alpha0 * post.beta0.ln()
            + 0.5 * (self.kappa0 / post.kappa0).ln()
            - (n_f / 2.0) * (2.0 * PI).ln();

        Ok(log_ml)
    }

    /// Posterior predictive PDF at point `x`.
    ///
    /// Under NIG, the posterior predictive is a Student-t distribution:
    /// ```text
    /// p(x | obs) = t_{2α₀}(x | μ₀, β₀(κ₀+1)/(κ₀ α₀))
    /// ```
    /// (evaluated at the current, potentially updated, parameters)
    pub fn posterior_predictive_pdf(&self, x: f64) -> f64 {
        let df = 2.0 * self.alpha0;
        let scale_sq = self.beta0 * (self.kappa0 + 1.0) / (self.kappa0 * self.alpha0);
        let scale = scale_sq.sqrt();

        // Student-t PDF: Γ((ν+1)/2) / (Γ(ν/2) √(νπ) σ) * (1 + z²/ν)^{-(ν+1)/2}
        let z = (x - self.mu0) / scale;
        let log_pdf = lgamma((df + 1.0) / 2.0)
            - lgamma(df / 2.0)
            - 0.5 * (df * PI).ln()
            - scale.ln()
            - ((df + 1.0) / 2.0) * (1.0 + z * z / df).ln();
        log_pdf.exp()
    }

    /// Sample (μ, σ²) from the NIG distribution.
    ///
    /// Algorithm:
    /// 1. Sample σ² ~ InvGamma(α₀, β₀)  (via Gamma: σ² = 1/g, g ~ Gamma(α₀, 1/β₀))
    /// 2. Sample μ | σ² ~ Normal(μ₀, σ²/κ₀)
    pub fn sample(&self, rng: &mut StdRng) -> Result<(f64, f64)> {
        // Sample precision τ = 1/σ² ~ Gamma(α₀, 1/β₀)
        let gamma = Gamma::new(self.alpha0, 1.0 / self.beta0).map_err(|e| {
            StatsError::ComputationError(format!("NIG Gamma sampling error: {e}"))
        })?;
        let tau = gamma.sample(rng);
        let sigma2 = if tau > 0.0 { 1.0 / tau } else { f64::MAX };

        // Sample μ | σ² ~ N(μ₀, σ²/κ₀)
        let std_mu = (sigma2 / self.kappa0).sqrt();
        let normal = Normal::new(self.mu0, std_mu).map_err(|e| {
            StatsError::ComputationError(format!("NIG Normal sampling error: {e}"))
        })?;
        let mu = normal.sample(rng);

        Ok((mu, sigma2))
    }

    /// Return the posterior mode of σ²: β₀ / (α₀ + 1).
    pub fn sigma2_mode(&self) -> f64 {
        self.beta0 / (self.alpha0 + 1.0)
    }

    /// Return the posterior mean of σ²: β₀ / (α₀ - 1)  (requires α₀ > 1).
    pub fn sigma2_mean(&self) -> Result<f64> {
        if self.alpha0 <= 1.0 {
            return Err(StatsError::DomainError(
                "sigma2_mean requires alpha0 > 1".into(),
            ));
        }
        Ok(self.beta0 / (self.alpha0 - 1.0))
    }
}

impl HyperPrior for NormalInverseGamma {
    fn log_norm_const(&self) -> f64 {
        lgamma(self.alpha0)
            + 0.5 * self.kappa0.ln()
            - self.alpha0 * self.beta0.ln()
            - 0.5 * (2.0 * PI).ln()
    }

    fn name(&self) -> &'static str {
        "NormalInverseGamma"
    }
}

// ---------------------------------------------------------------------------
// NormalInverseWishart
// ---------------------------------------------------------------------------

/// Normal-Inverse-Wishart distribution: multivariate conjugate prior for
/// (μ, Σ) in the multivariate Normal model.
///
/// Parameterization:
/// ```text
///   Σ      ~ InvWishart(ν₀, Ψ₀)
///   μ | Σ  ~ Normal(μ₀, Σ/κ₀)
/// ```
#[derive(Debug, Clone)]
pub struct NormalInverseWishart {
    /// Prior mean vector (dimension d).
    pub mu0: Vec<f64>,
    /// Prior precision scale (> 0).
    pub kappa0: f64,
    /// Degrees of freedom (> d - 1).
    pub nu0: f64,
    /// Scale matrix Ψ₀ (d×d, positive definite).
    pub psi0: Vec<Vec<f64>>,
    /// Dimensionality.
    pub dim: usize,
}

impl NormalInverseWishart {
    /// Construct a new `NormalInverseWishart`.
    ///
    /// # Errors
    /// Returns `StatsError::DomainError` when parameters are invalid.
    pub fn new(mu0: Vec<f64>, kappa0: f64, nu0: f64, psi0: Vec<Vec<f64>>) -> Result<Self> {
        let dim = mu0.len();
        if dim == 0 {
            return Err(StatsError::DomainError(
                "mu0 must be non-empty".into(),
            ));
        }
        if kappa0 <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "kappa0 must be > 0, got {kappa0}"
            )));
        }
        if nu0 < dim as f64 {
            return Err(StatsError::DomainError(format!(
                "nu0 ({nu0}) must be >= dim ({dim})"
            )));
        }
        if psi0.len() != dim || psi0.iter().any(|row| row.len() != dim) {
            return Err(StatsError::DimensionMismatch(format!(
                "psi0 must be {dim}×{dim}"
            )));
        }
        Ok(Self { mu0, kappa0, nu0, psi0, dim })
    }

    /// Conjugate Bayesian update given multivariate observations `obs` (N × d).
    pub fn update(&self, obs: &[Vec<f64>]) -> Result<Self> {
        let n = obs.len();
        if n == 0 {
            return Ok(self.clone());
        }
        let n_f = n as f64;
        let d = self.dim;

        // Validate observations
        for (i, row) in obs.iter().enumerate() {
            if row.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "obs[{i}] has length {}, expected {d}",
                    row.len()
                )));
            }
        }

        // Sample mean x̄
        let mut x_bar = vec![0.0_f64; d];
        for row in obs {
            for (k, &v) in row.iter().enumerate() {
                x_bar[k] += v;
            }
        }
        for k in 0..d {
            x_bar[k] /= n_f;
        }

        // Updated parameters
        let kappa_n = self.kappa0 + n_f;
        let nu_n = self.nu0 + n_f;

        let mut mu_n = vec![0.0_f64; d];
        for k in 0..d {
            mu_n[k] = (self.kappa0 * self.mu0[k] + n_f * x_bar[k]) / kappa_n;
        }

        // Psi_n = Psi0 + S + kappa0*n/(kappa0+n) * (x̄ - mu0)(x̄ - mu0)^T
        // where S = sum_i (x_i - x̄)(x_i - x̄)^T
        let mut psi_n = self.psi0.clone();

        // Add scatter matrix S
        for row in obs {
            for i in 0..d {
                for j in 0..d {
                    psi_n[i][j] += (row[i] - x_bar[i]) * (row[j] - x_bar[j]);
                }
            }
        }

        // Add rank-1 correction
        let scale = self.kappa0 * n_f / kappa_n;
        for i in 0..d {
            for j in 0..d {
                psi_n[i][j] += scale * (x_bar[i] - self.mu0[i]) * (x_bar[j] - self.mu0[j]);
            }
        }

        Self::new(mu_n, kappa_n, nu_n, psi_n)
    }

    /// Log marginal likelihood for multivariate observations.
    ///
    /// Uses the NIW closed-form marginal:
    /// ```text
    /// log p(X | NIW) = log Z(Ψₙ, νₙ, κₙ) - log Z(Ψ₀, ν₀, κ₀) - n*d/2 * log(π)
    /// ```
    /// where `log Z(Ψ, ν, κ) = log Γ_d(ν/2) - ν/2 * log|Ψ| + d/2 * log(κ)`.
    pub fn log_marginal_likelihood(&self, obs: &[Vec<f64>]) -> Result<f64> {
        let n = obs.len();
        if n == 0 {
            return Ok(0.0);
        }
        let n_f = n as f64;
        let d = self.dim as f64;
        let post = self.update(obs)?;

        let log_z_prior = log_niw_norm_const(&self.psi0, self.nu0, self.kappa0, self.dim)?;
        let log_z_post = log_niw_norm_const(&post.psi0, post.nu0, post.kappa0, self.dim)?;

        Ok(log_z_post - log_z_prior - n_f * d / 2.0 * PI.ln())
    }

    /// Return the mean of the marginal distribution of μ, which is simply μ₀.
    pub fn mean_of_mu(&self) -> &[f64] {
        &self.mu0
    }
}

impl HyperPrior for NormalInverseWishart {
    fn log_norm_const(&self) -> f64 {
        log_niw_norm_const(&self.psi0, self.nu0, self.kappa0, self.dim).unwrap_or(f64::NEG_INFINITY)
    }

    fn name(&self) -> &'static str {
        "NormalInverseWishart"
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Log normalizing constant for the NIW distribution.
fn log_niw_norm_const(psi: &[Vec<f64>], nu: f64, kappa: f64, d: usize) -> Result<f64> {
    let log_det = log_det_chol(psi, d)?;
    let log_gamma_d = multivariate_lgamma(nu / 2.0, d);
    Ok(log_gamma_d - nu / 2.0 * log_det + d as f64 / 2.0 * kappa.ln())
}

/// Multivariate log-gamma function: log Γ_d(x) = (d(d-1)/4) log(π) + Σ_{j=1}^{d} log Γ(x + (1-j)/2)
fn multivariate_lgamma(x: f64, d: usize) -> f64 {
    let mut result = (d * (d - 1)) as f64 / 4.0 * PI.ln();
    for j in 1..=d {
        result += lgamma(x + (1.0 - j as f64) / 2.0);
    }
    result
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky.
fn log_det_chol(m: &[Vec<f64>], d: usize) -> Result<f64> {
    // Simple Cholesky decomposition to get diagonal elements
    let mut l = vec![vec![0.0_f64; d]; d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = m[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(StatsError::ComputationError(
                        "Matrix is not positive definite".into(),
                    ));
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // log|M| = 2 * sum_i log(L[i][i])
    let log_det: f64 = (0..d).map(|i| l[i][i].ln()).sum::<f64>() * 2.0;
    Ok(log_det)
}

/// Natural log of the gamma function (Stirling-based approximation for large x,
/// exact recurrence for small x).
pub(crate) fn lgamma(x: f64) -> f64 {
    // Use the standard math library lgamma via Rust's f64 methods
    // Rust doesn't have lgamma directly, use approximation via Lanczos
    lanczos_lgamma(x)
}

/// Lanczos approximation for log Γ(x), accurate to ~15 significant figures.
fn lanczos_lgamma(x: f64) -> f64 {
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π/sin(πx)
        return PI.ln() - (PI * x).sin().abs().ln() - lanczos_lgamma(1.0 - x);
    }

    // Lanczos coefficients g=7
    let g = 7.0_f64;
    let c = [
        0.999_999_999_999_809_3_f64,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312_4e-7,
    ];

    let xm1 = x - 1.0;
    let mut series = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        series += ci / (xm1 + (i as f64 + 1.0));
    }

    let t = xm1 + g + 0.5;
    (2.0 * PI).sqrt().ln() + series.ln() + (xm1 + 0.5) * t.ln() - t
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nig_construction() {
        let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        assert_eq!(nig.mu0, 0.0);
        assert_eq!(nig.kappa0, 1.0);
        assert_eq!(nig.alpha0, 2.0);
        assert_eq!(nig.beta0, 1.0);
    }

    #[test]
    fn test_nig_invalid() {
        assert!(NormalInverseGamma::new(0.0, 0.0, 2.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(0.0, 1.0, 0.0, 1.0).is_err());
        assert!(NormalInverseGamma::new(0.0, 1.0, 2.0, 0.0).is_err());
        assert!(NormalInverseGamma::new(f64::NAN, 1.0, 2.0, 1.0).is_err());
    }

    #[test]
    fn test_nig_update() {
        let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();
        let obs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let post = prior.update(&obs).unwrap();

        // κ_n = κ_0 + n = 1 + 5 = 6
        assert!((post.kappa0 - 6.0).abs() < 1e-10);
        // μ_n = (κ_0 μ_0 + n x̄) / κ_n = (0 + 5*3) / 6 = 2.5
        assert!((post.mu0 - 2.5).abs() < 1e-10);
        // α_n = α_0 + n/2 = 1 + 2.5 = 3.5
        assert!((post.alpha0 - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_nig_update_empty() {
        let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();
        let post = prior.update(&[]).unwrap();
        assert_eq!(post.mu0, prior.mu0);
        assert_eq!(post.kappa0, prior.kappa0);
    }

    #[test]
    fn test_nig_sample() {
        let nig = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let (mu, sigma2) = nig.sample(&mut rng).unwrap();
        assert!(mu.is_finite());
        assert!(sigma2 > 0.0);
    }

    #[test]
    fn test_nig_posterior_predictive() {
        let nig = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let pdf_at_0 = nig.posterior_predictive_pdf(0.0);
        let pdf_at_10 = nig.posterior_predictive_pdf(10.0);
        // PDF should be higher near the mean
        assert!(pdf_at_0 > pdf_at_10);
        assert!(pdf_at_0 > 0.0);
    }

    #[test]
    fn test_nig_log_marginal_likelihood() {
        let prior = NormalInverseGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();
        let obs = vec![0.0, 0.1, -0.1, 0.2, -0.2];
        let lml = prior.log_marginal_likelihood(&obs).unwrap();
        assert!(lml.is_finite());
        // More data near the prior mean should have higher marginal likelihood
        // than data far from it
        let obs_far = vec![10.0, 10.1, 9.9, 10.2, 9.8];
        let lml_far = prior.log_marginal_likelihood(&obs_far).unwrap();
        assert!(lml > lml_far);
    }

    #[test]
    fn test_niw_construction() {
        let mu0 = vec![0.0, 0.0];
        let psi0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let niw = NormalInverseWishart::new(mu0, 1.0, 3.0, psi0).unwrap();
        assert_eq!(niw.dim, 2);
    }

    #[test]
    fn test_niw_invalid() {
        let mu0 = vec![0.0, 0.0];
        let psi0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // nu0 too small
        assert!(NormalInverseWishart::new(mu0.clone(), 1.0, 1.0, psi0.clone()).is_err());
        // kappa0 <= 0
        assert!(NormalInverseWishart::new(mu0.clone(), 0.0, 3.0, psi0.clone()).is_err());
        // Wrong psi0 size
        let bad_psi = vec![vec![1.0]];
        assert!(NormalInverseWishart::new(mu0, 1.0, 3.0, bad_psi).is_err());
    }

    #[test]
    fn test_niw_update() {
        let mu0 = vec![0.0, 0.0];
        let psi0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let prior = NormalInverseWishart::new(mu0, 1.0, 3.0, psi0).unwrap();
        let obs = vec![
            vec![1.0, 0.5],
            vec![2.0, 1.5],
            vec![-1.0, 0.0],
        ];
        let post = prior.update(&obs).unwrap();
        // kappa_n = kappa0 + n = 1 + 3 = 4
        assert!((post.kappa0 - 4.0).abs() < 1e-10);
        // nu_n = nu0 + n = 3 + 3 = 6
        assert!((post.nu0 - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_niw_log_marginal_likelihood() {
        let mu0 = vec![0.0, 0.0];
        let psi0 = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let prior = NormalInverseWishart::new(mu0, 1.0, 4.0, psi0).unwrap();
        let obs = vec![
            vec![0.1, -0.1],
            vec![-0.1, 0.1],
            vec![0.0, 0.0],
        ];
        let lml = prior.log_marginal_likelihood(&obs).unwrap();
        assert!(lml.is_finite());
    }

    #[test]
    fn test_lgamma() {
        // Known values
        assert!((lgamma(1.0) - 0.0).abs() < 1e-10);  // Γ(1) = 1
        assert!((lgamma(2.0) - 0.0).abs() < 1e-10);  // Γ(2) = 1
        assert!((lgamma(0.5) - (PI.sqrt().ln())).abs() < 1e-6);  // Γ(1/2) = √π
    }
}
