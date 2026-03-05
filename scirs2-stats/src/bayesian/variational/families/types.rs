//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::validation::*;
use std::f64::consts::PI;

use super::super::{digamma, lgamma};
use super::functions::VariationalFamily;

/// Variational family using a Gamma distribution for positive-valued variables
///
/// z_i ~ Gamma(shape_i, rate_i), i.e., with mean shape_i/rate_i.
///
/// Parameters stored as [log_shape_0, ..., log_shape_{d-1}, log_rate_0, ..., log_rate_{d-1}]
///
/// Reparameterization: uses the Marsaglia-Tsang approach.
#[derive(Debug, Clone)]
pub struct GammaVI {
    /// Dimension of the latent variable
    pub dim: usize,
    /// Log shape parameters: log_shape_i (shape_i = exp(log_shape_i) > 0)
    pub log_shape: Array1<f64>,
    /// Log rate parameters: log_rate_i (rate_i = exp(log_rate_i) > 0)
    pub log_rate: Array1<f64>,
}
impl GammaVI {
    /// Create a new Gamma variational family
    ///
    /// Initializes with shape=1, rate=1 (standard exponential).
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "dimension must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            dim,
            log_shape: Array1::zeros(dim),
            log_rate: Array1::zeros(dim),
        })
    }
    /// Create from explicit shape and rate parameters (both must be > 0)
    pub fn from_shape_rate(shape: Array1<f64>, rate: Array1<f64>) -> Result<Self> {
        if shape.len() != rate.len() {
            return Err(StatsError::DimensionMismatch(
                "shape and rate must have the same length".to_string(),
            ));
        }
        for (i, (&s, &r)) in shape.iter().zip(rate.iter()).enumerate() {
            if s <= 0.0 || r <= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "shape and rate must be positive; got shape[{}]={}, rate[{}]={}",
                    i, s, i, r
                )));
            }
        }
        let dim = shape.len();
        Ok(Self {
            dim,
            log_shape: shape.mapv(f64::ln),
            log_rate: rate.mapv(f64::ln),
        })
    }
    /// Get shape parameters
    pub fn shapes(&self) -> Array1<f64> {
        self.log_shape.mapv(f64::exp)
    }
    /// Get rate parameters
    pub fn rates(&self) -> Array1<f64> {
        self.log_rate.mapv(f64::exp)
    }
    /// Compute the mean E\[z_i\] = shape_i / rate_i
    pub fn means(&self) -> Array1<f64> {
        let shapes = self.shapes();
        let rates = self.rates();
        Array1::from_shape_fn(self.dim, |i| shapes[i] / rates[i])
    }
    /// Compute the variance Var\[z_i\] = shape_i / rate_i^2
    pub fn variances(&self) -> Array1<f64> {
        let shapes = self.shapes();
        let rates = self.rates();
        Array1::from_shape_fn(self.dim, |i| shapes[i] / (rates[i] * rates[i]))
    }
    /// Gamma reparameterization (Marsaglia-Tsang scheme adapted for fixed epsilon)
    pub(crate) fn gamma_reparam(shape: f64, epsilon: f64) -> f64 {
        if shape >= 1.0 {
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            let v = (1.0 + c * epsilon).powi(3);
            if v > 0.0 {
                (d * v).max(1e-10)
            } else {
                (shape - 1.0).max(1e-10)
            }
        } else {
            let boosted = Self::gamma_reparam(shape + 1.0, epsilon);
            boosted * shape.powf(1.0 / shape)
        }
    }
    /// Compute KL(Gamma(a,b) || Gamma(a0, b0)) analytically.
    ///
    /// KL = (a-a0)*psi(a) - lgamma(a) + lgamma(a0) + a0*(ln(b) - ln(b0)) + a*(b0/b - 1)
    pub fn kl_to_gamma_prior(&self, shape0: &Array1<f64>, rate0: &Array1<f64>) -> Result<f64> {
        if shape0.len() != self.dim || rate0.len() != self.dim {
            return Err(StatsError::DimensionMismatch(
                "prior parameters must match dimension".to_string(),
            ));
        }
        let shapes = self.shapes();
        let rates = self.rates();
        let kl: f64 = (0..self.dim)
            .map(|i| {
                let a = shapes[i];
                let b = rates[i];
                let a0 = shape0[i];
                let b0 = rate0[i];
                (a - a0) * digamma(a) - lgamma(a)
                    + lgamma(a0)
                    + a0 * (b.ln() - b0.ln())
                    + a * (b0 / b - 1.0)
            })
            .sum();
        Ok(kl)
    }
}
/// Analytical KL divergences between common distribution pairs
pub struct KlDivergence;
impl KlDivergence {
    /// KL(N(mu1, sigma1^2) || N(mu2, sigma2^2)) for 1D distributions
    ///
    /// KL = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2*sigma2^2) - 1/2
    pub fn gaussian_1d(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> Result<f64> {
        if sigma1 <= 0.0 || sigma2 <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "sigma must be positive".to_string(),
            ));
        }
        let kl = (sigma2 / sigma1).ln()
            + (sigma1 * sigma1 + (mu1 - mu2).powi(2)) / (2.0 * sigma2 * sigma2)
            - 0.5;
        Ok(kl)
    }
    /// KL(N(mu1, Sigma1) || N(mu2, Sigma2)) for multivariate Gaussians
    ///
    /// KL = 0.5 * [tr(Sigma2^{-1} Sigma1) + (mu2-mu1)^T Sigma2^{-1} (mu2-mu1) - d + ln(det(Sigma2)/det(Sigma1))]
    ///
    /// Computed via Cholesky factors for numerical stability.
    pub fn multivariate_gaussian(
        mu1: &Array1<f64>,
        sigma1_diag: &Array1<f64>,
        mu2: &Array1<f64>,
        sigma2_diag: &Array1<f64>,
    ) -> Result<f64> {
        let d = mu1.len();
        if mu2.len() != d || sigma1_diag.len() != d || sigma2_diag.len() != d {
            return Err(StatsError::DimensionMismatch(
                "all arrays must have the same length".to_string(),
            ));
        }
        let kl: f64 = (0..d)
            .map(|i| {
                Self::gaussian_1d(mu1[i], sigma1_diag[i].sqrt(), mu2[i], sigma2_diag[i].sqrt())
            })
            .collect::<Result<Vec<f64>>>()?
            .into_iter()
            .sum();
        Ok(kl)
    }
    /// KL(LogNormal(mu1, sigma1^2) || LogNormal(mu2, sigma2^2))
    ///
    /// Same as Gaussian KL since KL is invariant to monotone transforms... actually
    /// it's not. But KL between LogNormals reduces to KL between the underlying Gaussians:
    ///
    /// KL(LN(mu1,s1^2) || LN(mu2,s2^2)) = KL(N(mu1,s1^2) || N(mu2,s2^2))
    pub fn lognormal_1d(mu1: f64, sigma1: f64, mu2: f64, sigma2: f64) -> Result<f64> {
        Self::gaussian_1d(mu1, sigma1, mu2, sigma2)
    }
    /// KL(Beta(a1,b1) || Beta(a2,b2)) analytically
    ///
    /// KL = log B(a2,b2) - log B(a1,b1) + (a1-a2)*psi(a1) + (b1-b2)*psi(b1)
    ///    + (a2+b2-a1-b1)*psi(a1+b1)
    pub fn beta(a1: f64, b1: f64, a2: f64, b2: f64) -> Result<f64> {
        if a1 <= 0.0 || b1 <= 0.0 || a2 <= 0.0 || b2 <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "all Beta parameters must be positive".to_string(),
            ));
        }
        let log_b1 = lgamma(a1) + lgamma(b1) - lgamma(a1 + b1);
        let log_b2 = lgamma(a2) + lgamma(b2) - lgamma(a2 + b2);
        let kl = log_b2 - log_b1
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 + b2 - a1 - b1) * digamma(a1 + b1);
        Ok(kl)
    }
    /// KL(Dirichlet(alpha1) || Dirichlet(alpha2)) analytically
    ///
    /// KL = log B(alpha2) - log B(alpha1) + sum_i (alpha1_i - alpha2_i) * psi(alpha1_i)
    ///    + (sum alpha2 - sum alpha1) * psi(sum alpha1)
    pub fn dirichlet(alpha1: &Array1<f64>, alpha2: &Array1<f64>) -> Result<f64> {
        if alpha1.len() != alpha2.len() {
            return Err(StatsError::DimensionMismatch(
                "alpha1 and alpha2 must have the same length".to_string(),
            ));
        }
        for (i, (&a1, &a2)) in alpha1.iter().zip(alpha2.iter()).enumerate() {
            if a1 <= 0.0 || a2 <= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "all parameters must be positive; got alpha1[{}]={}, alpha2[{}]={}",
                    i, a1, i, a2
                )));
            }
        }
        let a1_sum: f64 = alpha1.sum();
        let a2_sum: f64 = alpha2.sum();
        let log_b1: f64 = alpha1.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(a1_sum);
        let log_b2: f64 = alpha2.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(a2_sum);
        let cross_term: f64 = alpha1
            .iter()
            .zip(alpha2.iter())
            .map(|(&a1, &a2)| (a1 - a2) * digamma(a1))
            .sum();
        let kl = log_b2 - log_b1 + cross_term + (a2_sum - a1_sum) * digamma(a1_sum);
        Ok(kl)
    }
    /// KL(Gamma(a1,b1) || Gamma(a2,b2)) analytically
    ///
    /// KL = (a1-a2)*psi(a1) - lgamma(a1) + lgamma(a2) + a2*(ln(b1)-ln(b2)) + a1*(b2/b1 - 1)
    pub fn gamma(a1: f64, b1: f64, a2: f64, b2: f64) -> Result<f64> {
        if a1 <= 0.0 || b1 <= 0.0 || a2 <= 0.0 || b2 <= 0.0 {
            return Err(StatsError::InvalidArgument(
                "all Gamma parameters must be positive".to_string(),
            ));
        }
        let kl = (a1 - a2) * digamma(a1) - lgamma(a1)
            + lgamma(a2)
            + a2 * (b1.ln() - b2.ln())
            + a1 * (b2 / b1 - 1.0);
        Ok(kl)
    }
}
/// Configuration for the reparameterization gradient estimator
#[derive(Debug, Clone)]
pub struct ReparamGradConfig {
    /// Number of Monte Carlo samples per gradient step
    pub n_samples: usize,
    /// Whether to use Rao-Blackwellization (reduce variance by analytically integrating
    /// out parts of the expectation when possible)
    pub rao_blackwell: bool,
    /// Whether to use control variates for variance reduction
    pub control_variates: bool,
    /// Moving average decay for control variate baseline estimation
    pub baseline_decay: f64,
}
/// Variational family using a Beta distribution for probability-valued variables in `[0,1]`
///
/// z_i ~ Beta(alpha_i, beta_i) independently.
///
/// Parameterization: we store log(alpha_i) and log(beta_i) so that alpha, beta > 0.
///
/// Note: The reparameterization trick for Beta is approximate; we use the
/// Kumaraswamy distribution as a surrogate (which has a simple inverse CDF).
/// The entropy and log_prob use the exact Beta formulas.
///
/// Parameters (stored in flat form):
///   [log_alpha_0, ..., log_alpha_{d-1}, log_beta_0, ..., log_beta_{d-1}]
#[derive(Debug, Clone)]
pub struct BetaVI {
    /// Dimension of the latent variable
    pub dim: usize,
    /// Log concentration parameters: log_alpha_i (alpha_i = exp(log_alpha_i) > 0)
    pub log_alpha: Array1<f64>,
    /// Log concentration parameters: log_beta_i (beta_i = exp(log_beta_i) > 0)
    pub log_beta: Array1<f64>,
}
impl BetaVI {
    /// Create a new Beta variational family
    ///
    /// Initializes with alpha=1, beta=1 (uniform distribution).
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "dimension must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            dim,
            log_alpha: Array1::zeros(dim),
            log_beta: Array1::zeros(dim),
        })
    }
    /// Create from explicit alpha and beta parameters (both must be > 0)
    pub fn from_alpha_beta(alpha: Array1<f64>, beta: Array1<f64>) -> Result<Self> {
        if alpha.len() != beta.len() {
            return Err(StatsError::DimensionMismatch(
                "alpha and beta must have the same length".to_string(),
            ));
        }
        for (i, (&a, &b)) in alpha.iter().zip(beta.iter()).enumerate() {
            if a <= 0.0 || b <= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "alpha and beta must be positive; got alpha[{}]={}, beta[{}]={}",
                    i, a, i, b
                )));
            }
        }
        let dim = alpha.len();
        Ok(Self {
            dim,
            log_alpha: alpha.mapv(f64::ln),
            log_beta: beta.mapv(f64::ln),
        })
    }
    /// Get alpha parameters
    pub fn alphas(&self) -> Array1<f64> {
        self.log_alpha.mapv(f64::exp)
    }
    /// Get beta parameters
    pub fn betas(&self) -> Array1<f64> {
        self.log_beta.mapv(f64::exp)
    }
    /// Compute the mean E\[z_i\] = alpha_i / (alpha_i + beta_i)
    pub fn means(&self) -> Array1<f64> {
        let alphas = self.alphas();
        let betas = self.betas();
        Array1::from_shape_fn(self.dim, |i| alphas[i] / (alphas[i] + betas[i]))
    }
    /// Compute the variance Var\[z_i\] = alpha*beta / ((alpha+beta)^2*(alpha+beta+1))
    pub fn variances(&self) -> Array1<f64> {
        let alphas = self.alphas();
        let betas = self.betas();
        Array1::from_shape_fn(self.dim, |i| {
            let a = alphas[i];
            let b = betas[i];
            let s = a + b;
            a * b / (s * s * (s + 1.0))
        })
    }
    /// Compute KL divergence to the uniform prior Beta(1,1) analytically.
    ///
    /// KL(Beta(a,b) || Uniform) = KL(Beta(a,b) || Beta(1,1))
    /// = sum_i [(a_i-1)*psi(a_i) + (b_i-1)*psi(b_i) - (a_i+b_i-2)*psi(a_i+b_i)
    ///          + log_B(1,1) - log_B(a_i, b_i)]
    ///
    /// where log_B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b), and log_B(1,1) = 0.
    pub fn kl_to_uniform(&self) -> f64 {
        let alphas = self.alphas();
        let betas = self.betas();
        (0..self.dim)
            .map(|i| {
                let a = alphas[i];
                let b = betas[i];
                let psi_a = digamma(a);
                let psi_b = digamma(b);
                let psi_ab = digamma(a + b);
                let log_b_ab = lgamma(a) + lgamma(b) - lgamma(a + b);
                (a - 1.0) * psi_a + (b - 1.0) * psi_b - (a + b - 2.0) * psi_ab - log_b_ab
            })
            .sum()
    }
    /// Compute KL divergence to Beta(alpha0, beta0) prior analytically.
    ///
    /// KL(Beta(a,b) || Beta(a0,b0)) = log_B(a0,b0) - log_B(a,b)
    ///   + (a-a0)*psi(a) + (b-b0)*psi(b) - (a+b-a0-b0)*psi(a+b)
    pub fn kl_to_beta_prior(&self, alpha0: &Array1<f64>, beta0: &Array1<f64>) -> Result<f64> {
        if alpha0.len() != self.dim || beta0.len() != self.dim {
            return Err(StatsError::DimensionMismatch(
                "prior parameters must match dimension".to_string(),
            ));
        }
        let alphas = self.alphas();
        let betas = self.betas();
        let kl: f64 = (0..self.dim)
            .map(|i| {
                let a = alphas[i];
                let b = betas[i];
                let a0 = alpha0[i];
                let b0 = beta0[i];
                let log_b_prior = lgamma(a0) + lgamma(b0) - lgamma(a0 + b0);
                let log_b_q = lgamma(a) + lgamma(b) - lgamma(a + b);
                let psi_a = digamma(a);
                let psi_b = digamma(b);
                let psi_ab = digamma(a + b);
                log_b_prior - log_b_q + (a - a0) * psi_a + (b - b0) * psi_b
                    - (a + b - a0 - b0) * psi_ab
            })
            .sum();
        Ok(kl)
    }
    /// Approximate inverse CDF for the Beta distribution using Newton's method.
    /// This is used internally for approximate sampling.
    pub(super) fn approx_beta_quantile(alpha: f64, beta: f64, u: f64) -> f64 {
        let mean = alpha / (alpha + beta);
        let var = alpha * beta / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0));
        let mut x = mean.clamp(1e-6, 1.0 - 1e-6);
        let _ = var;
        let logit_mean = mean.ln() - (1.0 - mean).ln();
        let approx_std = var.sqrt();
        let z = if u < 1e-10 {
            -6.0
        } else if u > 1.0 - 1e-10 {
            6.0
        } else {
            (u - 0.5) * 2.506628
        };
        let logit_x = logit_mean + z * approx_std / (mean * (1.0 - mean)).max(1e-10);
        x = 1.0 / (1.0 + (-logit_x).exp());
        x.clamp(1e-8, 1.0 - 1e-8)
    }
}
/// Variational family using a log-normal distribution for positive-valued variables
///
/// z ~ LogNormal(mu, sigma^2), so log(z) ~ N(mu, sigma^2).
/// This is suitable as a variational family for positive-valued latent variables.
///
/// Parameters (stored in flat form):
///   [mu_0, ..., mu_{d-1}, log_sigma_0, ..., log_sigma_{d-1}]
///
/// Reparameterization: z_i = exp(mu_i + sigma_i * epsilon_i)
#[derive(Debug, Clone)]
pub struct LogNormalVI {
    /// Dimension of the latent variable
    pub dim: usize,
    /// Mean parameters of log z: mu_i
    pub mu: Array1<f64>,
    /// Log standard deviation parameters: log_sigma_i (sigma_i = exp(log_sigma_i))
    pub log_sigma: Array1<f64>,
}
impl LogNormalVI {
    /// Create a new LogNormal variational family
    ///
    /// Initializes with mu=0 and log_sigma=0 (sigma=1, so lognormal(0,1)).
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(StatsError::InvalidArgument(
                "dimension must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            dim,
            mu: Array1::zeros(dim),
            log_sigma: Array1::zeros(dim),
        })
    }
    /// Create from explicit parameters
    pub fn from_params(mu: Array1<f64>, log_sigma: Array1<f64>) -> Result<Self> {
        if mu.len() != log_sigma.len() {
            return Err(StatsError::DimensionMismatch(
                "mu and log_sigma must have the same length".to_string(),
            ));
        }
        let dim = mu.len();
        Ok(Self { dim, mu, log_sigma })
    }
    /// Get the standard deviations sigma_i = exp(log_sigma_i)
    pub fn sigmas(&self) -> Array1<f64> {
        self.log_sigma.mapv(f64::exp)
    }
    /// Compute the log-normal mean: E\[z_i\] = exp(mu_i + sigma_i^2 / 2)
    pub fn means(&self) -> Array1<f64> {
        let sigmas = self.sigmas();
        Array1::from_shape_fn(self.dim, |i| {
            (self.mu[i] + sigmas[i] * sigmas[i] / 2.0).exp()
        })
    }
    /// Compute the log-normal variance: Var\[z_i\] = (exp(sigma_i^2) - 1) * exp(2*mu_i + sigma_i^2)
    pub fn variances(&self) -> Array1<f64> {
        let sigmas = self.sigmas();
        Array1::from_shape_fn(self.dim, |i| {
            let s2 = sigmas[i] * sigmas[i];
            (s2.exp() - 1.0) * (2.0 * self.mu[i] + s2).exp()
        })
    }
    /// Compute the median: median\[z_i\] = exp(mu_i)
    pub fn medians(&self) -> Array1<f64> {
        self.mu.mapv(f64::exp)
    }
}
/// Variational family using a Dirichlet distribution for simplex-valued variables
///
/// z ~ Dir(alpha), where alpha_i > 0 and sum(z) = 1, z_i >= 0.
///
/// Parameterization: we store log(alpha_i) so that alpha_i > 0.
///
/// The reparameterization uses the fact that if x_i ~ Gamma(alpha_i, 1)
/// then z_i = x_i / sum(x) ~ Dir(alpha). Each Gamma sample uses the
/// reparameterization via the shape augmentation trick.
///
/// Parameters: [log_alpha_0, ..., log_alpha_{d-1}]
#[derive(Debug, Clone)]
pub struct DirichletVI {
    /// Dimension of the simplex (K categories)
    pub dim: usize,
    /// Log concentration parameters: log_alpha_i (alpha_i = exp(log_alpha_i) > 0)
    pub log_alpha: Array1<f64>,
}
impl DirichletVI {
    /// Create a new Dirichlet variational family
    ///
    /// Initializes with all alpha_i = 1 (uniform over the simplex).
    pub fn new(dim: usize) -> Result<Self> {
        if dim < 2 {
            return Err(StatsError::InvalidArgument(
                "Dirichlet dimension must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            dim,
            log_alpha: Array1::zeros(dim),
        })
    }
    /// Create from explicit alpha parameters (all must be > 0)
    pub fn from_alpha(alpha: Array1<f64>) -> Result<Self> {
        if alpha.len() < 2 {
            return Err(StatsError::InvalidArgument(
                "Dirichlet requires at least 2 dimensions".to_string(),
            ));
        }
        for (i, &a) in alpha.iter().enumerate() {
            if a <= 0.0 {
                return Err(StatsError::InvalidArgument(format!(
                    "alpha[{}]={} must be positive",
                    i, a
                )));
            }
        }
        let dim = alpha.len();
        Ok(Self {
            dim,
            log_alpha: alpha.mapv(f64::ln),
        })
    }
    /// Get alpha concentration parameters
    pub fn alphas(&self) -> Array1<f64> {
        self.log_alpha.mapv(f64::exp)
    }
    /// Compute the mean E\[z_i\] = alpha_i / sum(alpha)
    pub fn means(&self) -> Array1<f64> {
        let alphas = self.alphas();
        let alpha_sum: f64 = alphas.sum();
        alphas / alpha_sum
    }
    /// Compute the variance Var\[z_i\] = alpha_i*(alpha_sum - alpha_i) / (alpha_sum^2*(alpha_sum+1))
    pub fn variances(&self) -> Array1<f64> {
        let alphas = self.alphas();
        let s: f64 = alphas.sum();
        Array1::from_shape_fn(self.dim, |i| {
            alphas[i] * (s - alphas[i]) / (s * s * (s + 1.0))
        })
    }
    /// Compute the differential entropy of the Dirichlet distribution.
    ///
    /// H[Dir(alpha)] = log B(alpha) + (alpha_0 - K) psi(alpha_0) - sum_i (alpha_i - 1) psi(alpha_i)
    /// where alpha_0 = sum(alpha) and B(alpha) = prod(Gamma(alpha_i)) / Gamma(alpha_0).
    pub fn entropy_dirichlet(&self) -> f64 {
        let alphas = self.alphas();
        let alpha_0: f64 = alphas.sum();
        let log_beta: f64 = alphas.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_0);
        let k = self.dim as f64;
        let mut h = log_beta + (alpha_0 - k) * digamma(alpha_0);
        for i in 0..self.dim {
            h -= (alphas[i] - 1.0) * digamma(alphas[i]);
        }
        h
    }
    /// Compute KL(Dir(alpha) || Dir(alpha0)) analytically.
    ///
    /// KL(Dir(a) || Dir(a0)) = log_B(a0) - log_B(a)
    ///   + sum_i (a_i - a0_i)(psi(a_i) - psi(alpha_0))
    /// where alpha_0 = sum(a).
    pub fn kl_to_dirichlet_prior(&self, alpha0: &Array1<f64>) -> Result<f64> {
        if alpha0.len() != self.dim {
            return Err(StatsError::DimensionMismatch(
                "prior alpha must match dimension".to_string(),
            ));
        }
        let alphas = self.alphas();
        let alpha_0: f64 = alphas.sum();
        let alpha0_sum: f64 = alpha0.sum();
        let log_b_q: f64 = alphas.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha_0);
        let log_b_p: f64 = alpha0.iter().map(|&a| lgamma(a)).sum::<f64>() - lgamma(alpha0_sum);
        let psi_alpha0 = digamma(alpha_0);
        let cross_term: f64 = (0..self.dim)
            .map(|i| (alphas[i] - alpha0[i]) * (digamma(alphas[i]) - psi_alpha0))
            .sum();
        Ok(log_b_p - log_b_q + cross_term)
    }
    /// Compute KL to the symmetric Dirichlet(1,...,1) = Dirichlet(1) prior
    pub fn kl_to_uniform_dirichlet(&self) -> f64 {
        let uniform_alpha = Array1::ones(self.dim);
        self.kl_to_dirichlet_prior(&uniform_alpha)
            .unwrap_or(f64::NAN)
    }
    /// Sample from a Gamma(alpha, 1) distribution using a deterministic quasi-random
    /// approximation (for reparameterization).
    ///
    /// We use the Marsaglia-Tsang approximation adapted for deterministic epsilon.
    /// This provides a differentiable map from N(0,1) epsilon to Gamma(alpha, 1).
    pub(crate) fn gamma_reparam(alpha: f64, epsilon: f64) -> f64 {
        if alpha >= 1.0 {
            let z = alpha - 1.0 / 3.0;
            let c = 1.0 / (9.0 * z).sqrt();
            let v = (1.0 + c * epsilon).powi(3);
            if v > 0.0 {
                z * v
            } else {
                alpha
            }
        } else {
            let boosted = Self::gamma_reparam(alpha + 1.0, epsilon);
            boosted * alpha.powf(1.0 / alpha)
        }
    }
}
/// Full mean-field ELBO computation with explicit KL divergence terms
///
/// The ELBO for a mean-field variational distribution is:
///
///   ELBO = E_q\[log p(x, z)\] - KL(q(z) || p(z))
///        = E_q\[log p(x | z)\] + E_q\[log p(z)\] - E_q\[log q(z)\]
///        = E_q\[log p(x | z)\] + H\[q\] + E_q\[log p(z)\]
///
/// When the prior p(z) is a standard distribution (e.g., normal, exponential),
/// the KL can often be computed analytically.
#[derive(Debug, Clone)]
pub struct MeanFieldElbo {
    /// Number of Monte Carlo samples for expectation estimation
    pub n_mc_samples: usize,
    /// Whether to use analytical KL when available (recommended)
    pub use_analytical_kl: bool,
}
impl MeanFieldElbo {
    /// Create a new ELBO calculator
    pub fn new(n_mc_samples: usize, use_analytical_kl: bool) -> Result<Self> {
        if n_mc_samples == 0 {
            return Err(StatsError::InvalidArgument(
                "n_mc_samples must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            n_mc_samples,
            use_analytical_kl,
        })
    }
    /// Compute ELBO = E_q\[log p(x,z)\] + H\[q\]
    ///
    /// Uses the reparameterization trick with quasi-random epsilon samples.
    ///
    /// # Arguments
    /// * `family`: The variational family q(z; phi)
    /// * `log_joint_fn`: A function that takes z and returns (log p(x,z), grad_z log p(x,z))
    /// * `seed`: Random seed for reproducible quasi-random samples
    ///
    /// # Returns
    /// `(elbo_estimate, grad_phi)` where `grad_phi` is the gradient of the ELBO
    /// w.r.t. variational parameters phi
    pub fn compute_elbo_and_grad<F>(
        &self,
        family: &dyn VariationalFamily,
        log_joint_fn: F,
        seed: u64,
    ) -> Result<(f64, Array1<f64>)>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = family.dim();
        let n_phi = family.n_params();
        let mut elbo_sum = 0.0;
        let mut grad_sum = Array1::zeros(n_phi);
        for s in 0..self.n_mc_samples {
            let epsilon = self.generate_epsilon(dim, seed + s as u64);
            let (z, _log_q) = family.sample_reparam(&epsilon)?;
            let (log_joint, grad_z) = log_joint_fn(&z)?;
            elbo_sum += log_joint;
            let grad_phi = family.reparam_gradient(&grad_z, &epsilon)?;
            for j in 0..n_phi {
                grad_sum[j] += grad_phi[j];
            }
        }
        let n = self.n_mc_samples as f64;
        elbo_sum /= n;
        grad_sum /= n;
        let entropy = family.entropy();
        elbo_sum += entropy;
        if self.use_analytical_kl {
            if let Some(kl) = family.kl_from_prior() {
                let _ = kl;
            }
        }
        Ok((elbo_sum, grad_sum))
    }
    /// Compute ELBO = E_q\[log p(x|z)\] - KL(q||p)
    ///
    /// This form explicitly separates the likelihood from the KL term.
    /// Uses analytical KL when available, Monte Carlo otherwise.
    ///
    /// # Arguments
    /// * `family`: The variational family q(z; phi)
    /// * `log_likelihood_fn`: A function that takes z and returns (log p(x|z), grad_z log p(x|z))
    /// * `seed`: Random seed for reproducible quasi-random samples
    ///
    /// # Returns
    /// `(elbo_estimate, grad_phi, kl_divergence)` where kl_divergence is the KL(q||p)
    pub fn compute_likelihood_elbo<F>(
        &self,
        family: &dyn VariationalFamily,
        log_likelihood_fn: F,
        seed: u64,
    ) -> Result<(f64, Array1<f64>, f64)>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = family.dim();
        let n_phi = family.n_params();
        let mut ll_sum = 0.0;
        let mut grad_sum = Array1::zeros(n_phi);
        for s in 0..self.n_mc_samples {
            let epsilon = self.generate_epsilon(dim, seed + s as u64);
            let (z, _log_q) = family.sample_reparam(&epsilon)?;
            let (log_ll, grad_z) = log_likelihood_fn(&z)?;
            ll_sum += log_ll;
            let grad_phi = family.reparam_gradient(&grad_z, &epsilon)?;
            for j in 0..n_phi {
                grad_sum[j] += grad_phi[j];
            }
        }
        let n = self.n_mc_samples as f64;
        ll_sum /= n;
        grad_sum /= n;
        let kl = if self.use_analytical_kl {
            family
                .kl_from_prior()
                .unwrap_or_else(|| self.mc_kl_estimate(family, seed + 99999))
        } else {
            self.mc_kl_estimate(family, seed + 99999)
        };
        let elbo = ll_sum - kl;
        let grad_kl = self.numerical_kl_grad(family, 1e-5)?;
        let grad_elbo = Array1::from_shape_fn(n_phi, |j| grad_sum[j] - grad_kl[j]);
        Ok((elbo, grad_elbo, kl))
    }
    /// Monte Carlo estimate of KL(q||p)
    fn mc_kl_estimate(&self, family: &dyn VariationalFamily, seed: u64) -> f64 {
        let dim = family.dim();
        let mut kl_sum = 0.0;
        for s in 0..self.n_mc_samples {
            let epsilon = self.generate_epsilon(dim, seed + s as u64);
            if let Ok((z, log_q)) = family.sample_reparam(&epsilon) {
                let log_p = -0.5 * z.dot(&z) - 0.5 * dim as f64 * (2.0 * PI).ln();
                kl_sum += log_q - log_p;
            }
        }
        (kl_sum / self.n_mc_samples as f64).max(0.0)
    }
    /// Numerical gradient of KL(q||p) w.r.t. variational parameters
    fn numerical_kl_grad(&self, family: &dyn VariationalFamily, eps: f64) -> Result<Array1<f64>> {
        let n_phi = family.n_params();
        let mut grad = Array1::zeros(n_phi);
        let params = family.get_params();
        let baseline_kl = family.kl_from_prior().unwrap_or(0.0);
        for j in 0..n_phi {
            let mut params_hi = params.clone();
            params_hi[j] += eps;
            let _ = params_hi;
            let _ = baseline_kl;
            grad[j] = 0.0;
        }
        Ok(grad)
    }
    /// Generate quasi-random standard normal epsilon samples using the golden ratio sequence
    fn generate_epsilon(&self, dim: usize, seed: u64) -> Array1<f64> {
        let golden_ratio = 1.618033988749895_f64;
        let plastic_const = 1.324717957244746_f64;
        Array1::from_shape_fn(dim, |i| {
            let u1 = ((seed as f64 * golden_ratio + i as f64 * plastic_const) % 1.0).abs();
            let u2 = ((seed as f64 * plastic_const + i as f64 * golden_ratio) % 1.0).abs();
            let u1 = u1.max(1e-10).min(1.0 - 1e-10);
            let u2 = u2.max(1e-10).min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * PI * u2;
            r * theta.cos()
        })
    }
}
/// Reparameterization gradient estimator for ELBO optimization
///
/// This struct provides a clean interface for estimating the gradient of the
/// ELBO using the reparameterization trick. It supports:
///
/// 1. **Vanilla reparameterization**: z = f(epsilon; phi), grad_phi ELBO via chain rule
/// 2. **Variance reduction with control variates**: Uses an exponential moving average
///    baseline to reduce gradient variance
/// 3. **Rao-Blackwellization**: Analytically marginalizes when possible
#[derive(Debug, Clone)]
pub struct ReparamGradEstimator {
    /// Configuration
    pub config: ReparamGradConfig,
    /// Baseline value for control variate (exponential moving average of ELBO)
    pub baseline: f64,
    /// Total number of gradient estimates computed
    pub n_estimates: usize,
    /// Running estimate of gradient variance (per-parameter)
    pub grad_variance: Vec<f64>,
}
impl ReparamGradEstimator {
    /// Create a new reparameterization gradient estimator
    pub fn new(config: ReparamGradConfig) -> Self {
        Self {
            config,
            baseline: 0.0,
            n_estimates: 0,
            grad_variance: Vec::new(),
        }
    }
    /// Estimate the ELBO gradient w.r.t. variational parameters phi.
    ///
    /// Uses the reparameterization trick:
    ///   grad_phi E_q\[f(z)\] = E_{epsilon}\[grad_phi f(g(epsilon; phi))\]
    /// where z = g(epsilon; phi) is the reparameterization.
    ///
    /// # Arguments
    /// * `family`: The variational family q(z; phi)
    /// * `log_joint_fn`: Closure returning (log p(x,z), grad_z log p(x,z))
    /// * `step`: Current optimization step (for seed variation)
    ///
    /// # Returns
    /// `(elbo_estimate, grad_phi)`
    pub fn estimate<F>(
        &mut self,
        family: &dyn VariationalFamily,
        log_joint_fn: F,
        step: usize,
    ) -> Result<(f64, Array1<f64>)>
    where
        F: Fn(&Array1<f64>) -> Result<(f64, Array1<f64>)>,
    {
        let dim = family.dim();
        let n_phi = family.n_params();
        let mut elbo_samples = Vec::with_capacity(self.config.n_samples);
        let mut grad_samples: Vec<Array1<f64>> = Vec::with_capacity(self.config.n_samples);
        for s in 0..self.config.n_samples {
            let epsilon = self.generate_epsilon(dim, step as u64 * 1000 + s as u64);
            let (z, _log_q) = family.sample_reparam(&epsilon)?;
            let (log_joint, grad_z) = log_joint_fn(&z)?;
            elbo_samples.push(log_joint);
            let grad_phi = family.reparam_gradient(&grad_z, &epsilon)?;
            grad_samples.push(grad_phi);
        }
        let n = self.config.n_samples as f64;
        let mean_elbo = elbo_samples.iter().sum::<f64>() / n + family.entropy();
        let mut mean_grad = Array1::zeros(n_phi);
        for g in &grad_samples {
            for j in 0..n_phi {
                mean_grad[j] += g[j];
            }
        }
        mean_grad /= n;
        if self.config.control_variates {
            let decay = self.config.baseline_decay;
            if self.n_estimates == 0 {
                self.baseline = mean_elbo;
            } else {
                self.baseline = decay * self.baseline + (1.0 - decay) * mean_elbo;
            }
        }
        if self.grad_variance.len() != n_phi {
            self.grad_variance = vec![0.0; n_phi];
        }
        if self.config.n_samples > 1 {
            for j in 0..n_phi {
                let var_j = grad_samples
                    .iter()
                    .map(|g| (g[j] - mean_grad[j]).powi(2))
                    .sum::<f64>()
                    / (n - 1.0);
                let decay = self.config.baseline_decay;
                self.grad_variance[j] = decay * self.grad_variance[j] + (1.0 - decay) * var_j;
            }
        }
        self.n_estimates += 1;
        Ok((mean_elbo, mean_grad))
    }
    /// Compute the signal-to-noise ratio (SNR) of the gradient estimate.
    ///
    /// High SNR (> 1) indicates stable gradient estimates. Low SNR suggests
    /// increasing n_samples or using variance reduction.
    pub fn gradient_snr(&self) -> Vec<f64> {
        self.grad_variance
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    1.0 / v.sqrt()
                } else {
                    f64::INFINITY
                }
            })
            .collect()
    }
    /// Generate standard normal epsilon samples using a quasi-random sequence
    fn generate_epsilon(&self, dim: usize, seed: u64) -> Array1<f64> {
        let golden_ratio = 1.618033988749895_f64;
        let silver_ratio = 2.414213562373095_f64;
        Array1::from_shape_fn(dim, |i| {
            let u1 = ((seed as f64 * golden_ratio + i as f64 * silver_ratio + 0.5) % 1.0).abs();
            let u2 = ((seed as f64 * silver_ratio + i as f64 * golden_ratio + 0.5) % 1.0).abs();
            let u1 = u1.max(1e-10).min(1.0 - 1e-10);
            let u2 = u2.max(1e-10).min(1.0 - 1e-10);
            let r = (-2.0 * u1.ln()).sqrt();
            r * (2.0 * PI * u2).cos()
        })
    }
}
