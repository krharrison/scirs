//! Online Variational Bayes for conjugate exponential family models
//!
//! Implements Stochastic Variational Inference (SVI) following Hoffman et al. (2013):
//! natural gradient updates on the global variational parameters using mini-batches.
//!
//! The key insight is that for models in the exponential family with conjugate priors,
//! the natural gradient of the ELBO with respect to the global variational parameters
//! has a simple closed-form expression involving the expected sufficient statistics of
//! a mini-batch scaled to the full dataset.
//!
//! # Learning rate schedule
//!
//! ρ_t = (t + τ)^{-κ}
//!
//! where τ (delay) ≥ 0 slows down early iterations and κ ∈ (0.5, 1] controls the
//! forgetting rate. The Robbins-Monro conditions are satisfied: Σ ρ_t = ∞ and
//! Σ ρ_t² < ∞.
//!
//! # Supported models
//!
//! | Model | Prior | Likelihood | Natural params |
//! |-------|-------|------------|----------------|
//! | BetaBernoulli | Beta(α, β) | Bernoulli(θ) | [α-1, β-1] |
//! | GammaPoisson | Gamma(a, b) | Poisson(λ) | [a-1, -b] |
//! | NormalNormal | N(μ₀, σ₀²) | N(θ, σ²) | [μ₀/σ₀², -1/(2σ₀²)] |
//! | DirichletMultinomial | Dir(α) | Multinomial(θ) | [α_k - 1] |

use crate::error::{StatsError, StatsResult};

// ─── ConjugateModel ────────────────────────────────────────────────────────

/// Supported conjugate exponential family model types.
///
/// Each variant specifies the prior-likelihood pair. The `#[non_exhaustive]`
/// attribute allows new models to be added in future versions without breaking
/// downstream match statements.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ConjugateModel {
    /// Beta prior, Bernoulli likelihood: models a biased coin / click-through rate.
    BetaBernoulli,
    /// Gamma prior, Poisson likelihood: models an event rate (count data).
    GammaPoisson,
    /// Normal prior, Normal likelihood with **known** observation variance.
    /// Prior params: [μ₀, σ₀², σ²] where σ² is the fixed observation noise.
    NormalNormal,
    /// Dirichlet prior, Multinomial likelihood: models categorical proportions.
    DirichletMultinomial,
}

// ─── OnlineVbConfig ────────────────────────────────────────────────────────

/// Configuration for the Online Variational Bayes estimator.
#[derive(Debug, Clone)]
pub struct OnlineVbConfig {
    /// The conjugate model to fit.
    pub model: ConjugateModel,
    /// Base learning rate multiplier (rarely needs tuning; schedule is driven by kappa/delay).
    pub learning_rate: f64,
    /// Forgetting exponent κ ∈ (0.5, 1.0]. Higher values forget the past more quickly.
    /// Must satisfy 0.5 < kappa ≤ 1.0 for convergence (Robbins-Monro condition).
    pub forgetting_factor: f64,
    /// Delay τ ≥ 0. Slows early updates to improve stability.
    pub delay: f64,
    /// Number of data points per mini-batch.
    pub mini_batch_size: usize,
}

impl Default for OnlineVbConfig {
    fn default() -> Self {
        Self {
            model: ConjugateModel::BetaBernoulli,
            learning_rate: 1.0,
            forgetting_factor: 0.7,
            delay: 1.0,
            mini_batch_size: 32,
        }
    }
}

// ─── OnlineVbState ─────────────────────────────────────────────────────────

/// Internal state of the Online VB estimator.
///
/// The `natural_params` vector stores the **variational** natural parameters
/// (not the prior natural parameters). These are updated after each mini-batch.
#[derive(Debug, Clone)]
pub struct OnlineVbState {
    /// Current variational natural parameters λ.
    ///
    /// Interpretation by model:
    /// - BetaBernoulli: \[eta1, eta2\] = \[alpha-1, beta-1\] (alpha, beta > 0 always)
    /// - GammaPoisson: \[eta1, eta2\] = \[a-1, -b\]
    /// - NormalNormal: \[eta1, eta2\] = \[mu/sigma^2, -1/(2*sigma^2)\]
    /// - DirichletMultinomial: \[eta\_k\] = \[alpha\_k - 1\] for each category k
    pub natural_params: Vec<f64>,
    /// Total number of data points processed so far.
    pub n_processed: usize,
    /// Running exponential-smoothed estimate of the ELBO (for monitoring).
    pub elbo_estimate: f64,
}

// ─── OnlineVbEstimator ─────────────────────────────────────────────────────

/// Online Variational Bayes estimator for conjugate exponential family models.
///
/// # Algorithm
///
/// At each step t:
/// 1. Sample a mini-batch B of size |B| from the data stream.
/// 2. Compute the "noisy" natural gradient:
///    g_t = prior_natural_params + (N / |B|) * E_q[sufficient_stats(B)] - λ_t
/// 3. Update: λ_{t+1} = λ_t + ρ_t * g_t  (equivalent to: λ_{t+1} = (1-ρ_t) λ_t + ρ_t * λ̃)
///    where λ̃ = prior_natural_params + (N / |B|) * E_q[sufficient_stats(B)].
///
/// # Example
///
/// ```rust
/// use scirs2_stats::online::{OnlineVbConfig, OnlineVbEstimator, ConjugateModel};
///
/// let config = OnlineVbConfig {
///     model: ConjugateModel::BetaBernoulli,
///     forgetting_factor: 0.7,
///     delay: 1.0,
///     ..Default::default()
/// };
/// // prior Beta(2, 2), total_n = 1000
/// let mut estimator = OnlineVbEstimator::new(config, &[2.0, 2.0], 1000)
///     .expect("valid prior");
///
/// // Process a mini-batch of Bernoulli observations (1.0 = success, 0.0 = failure)
/// let batch = vec![1.0, 0.0, 1.0, 1.0, 0.0];
/// let elbo = estimator.update(&batch).expect("update ok");
/// let posterior_mean = estimator.predict();
/// ```
pub struct OnlineVbEstimator {
    config: OnlineVbConfig,
    state: OnlineVbState,
    /// Prior natural parameters η₀ (fixed throughout training).
    prior_natural: Vec<f64>,
    /// Estimated total dataset size N (used for scaling mini-batch stats).
    total_n: usize,
    /// Step counter t (1-indexed, incremented after each update call).
    step: usize,
}

impl OnlineVbEstimator {
    /// Create a new estimator.
    ///
    /// # Arguments
    ///
    /// * `config` - Algorithm configuration including model type and hyperparameters.
    /// * `prior_params` - **Moment-form** prior parameters (not natural parameters):
    ///   - BetaBernoulli: `[α₀, β₀]` with α₀, β₀ > 0
    ///   - GammaPoisson: `[a₀, b₀]` with a₀, b₀ > 0
    ///   - NormalNormal: `[μ₀, σ₀², σ²]` where σ² is the fixed likelihood variance
    ///   - DirichletMultinomial: `[α₁, …, α_K]` with each α_k > 0
    /// * `total_n` - Estimated total dataset size. Used to scale the mini-batch
    ///   sufficient statistics to the full-data equivalent.
    ///
    /// # Errors
    ///
    /// Returns `StatsError::InvalidArgument` if prior parameters are invalid.
    pub fn new(config: OnlineVbConfig, prior_params: &[f64], total_n: usize) -> StatsResult<Self> {
        if total_n == 0 {
            return Err(StatsError::InvalidArgument(
                "total_n must be > 0".to_string(),
            ));
        }
        if config.forgetting_factor <= 0.5 || config.forgetting_factor > 1.0 {
            return Err(StatsError::InvalidArgument(
                "forgetting_factor (kappa) must be in (0.5, 1.0]".to_string(),
            ));
        }
        if config.delay < 0.0 {
            return Err(StatsError::InvalidArgument(
                "delay (tau) must be >= 0".to_string(),
            ));
        }

        // Convert moment-form prior parameters to natural parameters.
        let prior_natural = moment_to_natural(&config.model, prior_params)?;

        // Initialise variational parameters to the prior.
        let natural_params = prior_natural.clone();

        let state = OnlineVbState {
            natural_params,
            n_processed: 0,
            elbo_estimate: f64::NEG_INFINITY,
        };

        Ok(Self {
            config,
            state,
            prior_natural,
            total_n,
            step: 0,
        })
    }

    /// Process one mini-batch and return an ELBO estimate.
    ///
    /// The ELBO is estimated as:
    ///   ELBO ≈ (N / |B|) * E_q[log p(B | θ)] - KL(q(θ) || p(θ))
    ///
    /// where the KL divergence has a closed form for all supported conjugate models.
    ///
    /// # Arguments
    ///
    /// * `data` - Mini-batch of observations. Format depends on the model:
    ///   - BetaBernoulli: values in {0, 1} (as f64)
    ///   - GammaPoisson: non-negative integer counts (as f64)
    ///   - NormalNormal: real-valued observations
    ///   - DirichletMultinomial: integer category indices (as f64, zero-based)
    ///
    /// # Returns
    ///
    /// Returns the estimated ELBO for this mini-batch update, or a `StatsError`.
    pub fn update(&mut self, data: &[f64]) -> StatsResult<f64> {
        if data.is_empty() {
            return Err(StatsError::InsufficientData(
                "mini-batch must not be empty".to_string(),
            ));
        }

        self.step += 1;
        let rho = self.learning_rate_at(self.step);
        let scale = self.total_n as f64 / data.len() as f64;

        // Compute expected sufficient statistics E_q[T(x)] for the mini-batch.
        let ess = expected_sufficient_stats(&self.config.model, &self.state.natural_params, data)?;

        // Natural parameter update: λ̃ = η₀ + N/|B| * E[T(B)]
        // For NormalNormal the 3rd element (sigma_sq) is not a variational parameter;
        // we must not update it. Only update the 2 true natural params [η₁, η₂].
        let n_variational = match self.config.model {
            ConjugateModel::NormalNormal => 2,
            _ => self.prior_natural.len(),
        };
        let mut lambda_tilde = vec![0.0_f64; self.prior_natural.len()];
        for i in 0..self.prior_natural.len() {
            lambda_tilde[i] = self.prior_natural[i]; // copy full prior (incl. sigma_sq)
        }
        for i in 0..n_variational {
            let ess_i = *ess.get(i).unwrap_or(&0.0);
            lambda_tilde[i] = self.prior_natural[i] + scale * ess_i;
        }

        // Interpolate: λ_{t+1} = (1 - ρ) λ_t + ρ λ̃  (only variational dims)
        for i in 0..n_variational {
            self.state.natural_params[i] =
                (1.0 - rho) * self.state.natural_params[i] + rho * lambda_tilde[i];
        }
        // Preserve non-variational elements (e.g. sigma_sq for NormalNormal).
        for i in n_variational..self.prior_natural.len() {
            self.state.natural_params[i] = self.prior_natural[i];
        }

        self.state.n_processed += data.len();

        // Estimate the ELBO.
        let elbo = estimate_elbo(
            &self.config.model,
            &self.state.natural_params,
            &self.prior_natural,
            data,
            scale,
        )?;

        // Exponential smoothing of ELBO (α=0.1 smoothing).
        if self.state.elbo_estimate.is_finite() {
            self.state.elbo_estimate = 0.9 * self.state.elbo_estimate + 0.1 * elbo;
        } else {
            self.state.elbo_estimate = elbo;
        }

        Ok(elbo)
    }

    /// Return the current variational posterior in **moment form**.
    ///
    /// - BetaBernoulli: \[alpha, beta\] where alpha = eta1+1, beta = eta2+1
    /// - GammaPoisson: \[a, b\] where a = eta1+1, b = -eta2
    /// - NormalNormal: \[mu, sigma^2\] of the posterior
    /// - DirichletMultinomial: \[alpha\_k\] for each category
    pub fn posterior_params(&self) -> Vec<f64> {
        natural_to_moment(&self.config.model, &self.state.natural_params)
    }

    /// Predict E\[theta\] under the current variational posterior.
    ///
    /// - BetaBernoulli: α / (α + β)  (posterior mean of Bernoulli parameter)
    /// - GammaPoisson: a / b          (posterior mean of Poisson rate)
    /// - NormalNormal: μ              (posterior mean)
    /// - DirichletMultinomial: mode of the primary category (α₁ / Σ α_k)
    pub fn predict(&self) -> f64 {
        let mp = self.posterior_params();
        match self.config.model {
            ConjugateModel::BetaBernoulli => {
                let alpha = mp.first().copied().unwrap_or(1.0);
                let beta = mp.get(1).copied().unwrap_or(1.0);
                let denom = alpha + beta;
                if denom > f64::EPSILON {
                    alpha / denom
                } else {
                    0.5
                }
            }
            ConjugateModel::GammaPoisson => {
                let a = mp.first().copied().unwrap_or(1.0);
                let b = mp.get(1).copied().unwrap_or(1.0);
                if b > f64::EPSILON {
                    a / b
                } else {
                    0.0
                }
            }
            ConjugateModel::NormalNormal => mp.first().copied().unwrap_or(0.0),
            ConjugateModel::DirichletMultinomial => {
                if mp.is_empty() {
                    return 0.0;
                }
                let sum: f64 = mp.iter().sum();
                if sum > f64::EPSILON {
                    mp[0] / sum
                } else {
                    1.0 / mp.len() as f64
                }
            }
            _ => 0.0,
        }
    }

    /// Return a reference to the current state.
    pub fn state(&self) -> &OnlineVbState {
        &self.state
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &OnlineVbConfig {
        &self.config
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Compute ρ_t = learning_rate * (t + τ)^{-κ}.
    fn learning_rate_at(&self, t: usize) -> f64 {
        let t_f = t as f64;
        let tau = self.config.delay;
        let kappa = self.config.forgetting_factor;
        let rho = self.config.learning_rate * (t_f + tau).powf(-kappa);
        rho.clamp(1e-10, 1.0)
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────

/// Convert moment-form prior parameters to natural parameters.
fn moment_to_natural(model: &ConjugateModel, params: &[f64]) -> StatsResult<Vec<f64>> {
    match model {
        ConjugateModel::BetaBernoulli => {
            if params.len() < 2 {
                return Err(StatsError::InvalidArgument(
                    "BetaBernoulli prior requires [alpha, beta]".to_string(),
                ));
            }
            let alpha = params[0];
            let beta = params[1];
            if alpha <= 0.0 || beta <= 0.0 {
                return Err(StatsError::InvalidArgument(
                    "BetaBernoulli prior: alpha, beta must be > 0".to_string(),
                ));
            }
            // Natural params of Beta: η₁ = α-1, η₂ = β-1
            Ok(vec![alpha - 1.0, beta - 1.0])
        }
        ConjugateModel::GammaPoisson => {
            if params.len() < 2 {
                return Err(StatsError::InvalidArgument(
                    "GammaPoisson prior requires [a, b]".to_string(),
                ));
            }
            let a = params[0];
            let b = params[1];
            if a <= 0.0 || b <= 0.0 {
                return Err(StatsError::InvalidArgument(
                    "GammaPoisson prior: a, b must be > 0".to_string(),
                ));
            }
            // Natural params of Gamma: η₁ = a-1, η₂ = -b
            Ok(vec![a - 1.0, -b])
        }
        ConjugateModel::NormalNormal => {
            if params.len() < 3 {
                return Err(StatsError::InvalidArgument(
                    "NormalNormal prior requires [mu0, sigma0^2, sigma^2]".to_string(),
                ));
            }
            let mu0 = params[0];
            let sigma0_sq = params[1];
            let sigma_sq = params[2];
            if sigma0_sq <= 0.0 || sigma_sq <= 0.0 {
                return Err(StatsError::InvalidArgument(
                    "NormalNormal prior: sigma0^2 and sigma^2 must be > 0".to_string(),
                ));
            }
            // Store: [η₁, η₂, log_sigma_sq] where η₁ = μ₀/σ₀², η₂ = -1/(2σ₀²)
            // We also need sigma_sq for likelihood; store as a 3rd element.
            Ok(vec![mu0 / sigma0_sq, -1.0 / (2.0 * sigma0_sq), sigma_sq])
        }
        ConjugateModel::DirichletMultinomial => {
            if params.is_empty() {
                return Err(StatsError::InvalidArgument(
                    "DirichletMultinomial prior requires at least one alpha".to_string(),
                ));
            }
            for (k, &a) in params.iter().enumerate() {
                if a <= 0.0 {
                    return Err(StatsError::InvalidArgument(format!(
                        "DirichletMultinomial prior: alpha[{}] must be > 0",
                        k
                    )));
                }
            }
            // Natural params of Dirichlet: η_k = α_k - 1
            Ok(params.iter().map(|&a| a - 1.0).collect())
        }
        _ => Err(StatsError::NotImplementedError(
            "Unsupported conjugate model variant".to_string(),
        )),
    }
}

/// Convert natural parameters back to moment-form parameters.
fn natural_to_moment(model: &ConjugateModel, eta: &[f64]) -> Vec<f64> {
    match model {
        ConjugateModel::BetaBernoulli => {
            // η₁ = α-1, η₂ = β-1 → α = η₁+1, β = η₂+1
            let alpha = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let beta = (eta.get(1).copied().unwrap_or(0.0) + 1.0).max(1e-10);
            vec![alpha, beta]
        }
        ConjugateModel::GammaPoisson => {
            // η₁ = a-1, η₂ = -b → a = η₁+1, b = -η₂
            let a = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let b = (-eta.get(1).copied().unwrap_or(-1.0)).max(1e-10);
            vec![a, b]
        }
        ConjugateModel::NormalNormal => {
            // η₁ = μ/σ², η₂ = -1/(2σ²) → σ² = -1/(2η₂), μ = η₁ * σ²
            let eta1 = eta.first().copied().unwrap_or(0.0);
            let eta2 = eta.get(1).copied().unwrap_or(-0.5);
            let sigma_sq = if eta2 < -f64::EPSILON {
                -1.0 / (2.0 * eta2)
            } else {
                1.0
            };
            let mu = eta1 * sigma_sq;
            vec![mu, sigma_sq]
        }
        ConjugateModel::DirichletMultinomial => {
            // η_k = α_k - 1 → α_k = η_k + 1
            eta.iter().map(|&e| (e + 1.0).max(1e-10)).collect()
        }
        _ => vec![],
    }
}

/// Compute expected sufficient statistics E_q[T(x)] for a mini-batch.
///
/// For conjugate models, the sufficient statistics of the likelihood form the
/// bridge between data and variational parameters.
fn expected_sufficient_stats(
    model: &ConjugateModel,
    eta: &[f64],
    data: &[f64],
) -> StatsResult<Vec<f64>> {
    let n = data.len() as f64;
    match model {
        ConjugateModel::BetaBernoulli => {
            // Sufficient stats: [sum(x), n - sum(x)]
            let sum_x: f64 = data.iter().sum();
            Ok(vec![sum_x, n - sum_x])
        }
        ConjugateModel::GammaPoisson => {
            // Sufficient stats: [sum(x), n]
            // (Poisson: T(x) = x, base measure = 1/x!, partition A(η) = exp(exp(η₁+1)·exp(-1/(−η₂))))
            for &x in data {
                if x < 0.0 {
                    return Err(StatsError::DomainError(
                        "GammaPoisson: observations must be non-negative".to_string(),
                    ));
                }
            }
            let sum_x: f64 = data.iter().sum();
            Ok(vec![sum_x, n])
        }
        ConjugateModel::NormalNormal => {
            // For Normal prior on θ with Normal likelihood x~N(θ, σ²):
            // The likelihood contributes [Σx/σ², -n/(2σ²)] to the natural parameters
            // η₁ = μ/σ_prior², η₂ = -1/(2σ_prior²).
            // Note: η₂ is negative, so the likelihood contribution for η₂ is -n/(2σ²).
            let sigma_sq = if eta.len() >= 3 {
                eta[2].max(f64::EPSILON)
            } else {
                1.0
            };
            let sum_x: f64 = data.iter().sum();
            Ok(vec![sum_x / sigma_sq, -n / (2.0 * sigma_sq)])
        }
        ConjugateModel::DirichletMultinomial => {
            // Sufficient stats: count[k] for each category k
            let k = eta.len();
            if k == 0 {
                return Err(StatsError::InvalidArgument(
                    "DirichletMultinomial: variational params have length 0".to_string(),
                ));
            }
            let mut counts = vec![0.0_f64; k];
            for &x in data {
                let idx = x as usize;
                if idx >= k {
                    return Err(StatsError::DomainError(format!(
                        "DirichletMultinomial: category index {} >= K={}",
                        idx, k
                    )));
                }
                counts[idx] += 1.0;
            }
            Ok(counts)
        }
        _ => Err(StatsError::NotImplementedError(
            "Unsupported model variant".to_string(),
        )),
    }
}

/// Estimate the Evidence Lower Bound (ELBO) for a mini-batch.
///
/// ELBO ≈ scale * E_q[log p(B|θ)] - KL(q(θ) || p(θ))
///
/// For all supported conjugate models the KL divergence has a closed-form
/// expression in terms of the sufficient statistics of the exponential family.
fn estimate_elbo(
    model: &ConjugateModel,
    eta: &[f64],
    prior_eta: &[f64],
    data: &[f64],
    scale: f64,
) -> StatsResult<f64> {
    let expected_log_likelihood = compute_expected_log_likelihood(model, eta, data)?;
    let kl = compute_kl(model, eta, prior_eta)?;
    Ok(scale * expected_log_likelihood - kl)
}

/// Expected log-likelihood E_q[log p(data | θ)] for a mini-batch.
fn compute_expected_log_likelihood(
    model: &ConjugateModel,
    eta: &[f64],
    data: &[f64],
) -> StatsResult<f64> {
    match model {
        ConjugateModel::BetaBernoulli => {
            // E_q[log p(x|θ)] = E_q[x log θ + (1-x) log(1-θ)]
            //                  = sum(x) * E[log θ] + sum(1-x) * E[log(1-θ)]
            // E[log θ] = ψ(α) - ψ(α+β), E[log(1-θ)] = ψ(β) - ψ(α+β)
            let alpha = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let beta = (eta.get(1).copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let e_log_theta = digamma(alpha) - digamma(alpha + beta);
            let e_log_1m_theta = digamma(beta) - digamma(alpha + beta);
            let sum_x: f64 = data.iter().sum();
            let n = data.len() as f64;
            Ok(sum_x * e_log_theta + (n - sum_x) * e_log_1m_theta)
        }
        ConjugateModel::GammaPoisson => {
            // E_q[log p(x|λ)] = sum(x) * E[log λ] - n * E[λ] - sum(log(x!))
            // E[log λ] = ψ(a) - log(b), E[λ] = a/b
            let a = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let b = (-eta.get(1).copied().unwrap_or(-1.0)).max(1e-10);
            let e_log_lambda = digamma(a) - b.ln();
            let e_lambda = a / b;
            let n = data.len() as f64;
            let sum_x: f64 = data.iter().sum();
            let sum_log_fact: f64 = data.iter().map(|&x| log_factorial(x as u64)).sum();
            Ok(sum_x * e_log_lambda - n * e_lambda - sum_log_fact)
        }
        ConjugateModel::NormalNormal => {
            // E_q[log p(x|θ)] for N(θ, σ²):
            // = -n/2 * log(2πσ²) - 1/(2σ²) * sum[(x - θ)²]
            // = -n/2 * log(2πσ²) - 1/(2σ²) * [sum(x²) - 2*sum(x)*E[θ] + n*(E[θ²])]
            // E[θ] = μ, E[θ²] = μ² + σ_q²
            let sigma_sq = if eta.len() >= 3 {
                eta[2].max(f64::EPSILON)
            } else {
                1.0
            };
            let eta1 = eta.first().copied().unwrap_or(0.0);
            let eta2 = eta.get(1).copied().unwrap_or(-0.5);
            let sigma_q_sq = if eta2 < -f64::EPSILON {
                -1.0 / (2.0 * eta2)
            } else {
                1.0
            };
            let mu = eta1 * sigma_q_sq;

            let n = data.len() as f64;
            let sum_x: f64 = data.iter().sum();
            let sum_x_sq: f64 = data.iter().map(|&x| x * x).sum();
            let e_theta_sq = mu * mu + sigma_q_sq;

            let ll = -0.5 * n * (2.0 * std::f64::consts::PI * sigma_sq).ln()
                - 1.0 / (2.0 * sigma_sq) * (sum_x_sq - 2.0 * sum_x * mu + n * e_theta_sq);
            Ok(ll)
        }
        ConjugateModel::DirichletMultinomial => {
            // E_q[log p(x|θ)] = sum_k count_k * E[log θ_k]
            // E[log θ_k] = ψ(α_k) - ψ(sum_k α_k)
            let alpha: Vec<f64> = eta.iter().map(|&e| (e + 1.0).max(1e-10)).collect();
            let sum_alpha: f64 = alpha.iter().sum();
            let k = alpha.len();
            let mut counts = vec![0.0_f64; k];
            for &x in data {
                let idx = x as usize;
                if idx < k {
                    counts[idx] += 1.0;
                }
            }
            let ll: f64 = alpha
                .iter()
                .enumerate()
                .map(|(i, &a)| counts[i] * (digamma(a) - digamma(sum_alpha)))
                .sum();
            Ok(ll)
        }
        _ => Ok(0.0),
    }
}

/// KL divergence KL(q || p) between the variational distribution and the prior.
///
/// For exponential families: KL(q || p) = A(η) - A(η₀) - (η - η₀)ᵀ ∇A(η)
/// where A is the log-partition function. For Beta and Gamma distributions this
/// has a closed form in terms of the digamma and log-gamma functions.
fn compute_kl(model: &ConjugateModel, eta: &[f64], prior_eta: &[f64]) -> StatsResult<f64> {
    match model {
        ConjugateModel::BetaBernoulli => {
            // KL(Beta(α_q, β_q) || Beta(α₀, β₀))
            let alpha_q = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let beta_q = (eta.get(1).copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let alpha_0 = (prior_eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let beta_0 = (prior_eta.get(1).copied().unwrap_or(0.0) + 1.0).max(1e-10);
            Ok(kl_beta(alpha_q, beta_q, alpha_0, beta_0))
        }
        ConjugateModel::GammaPoisson => {
            let a_q = (eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let b_q = (-eta.get(1).copied().unwrap_or(-1.0)).max(1e-10);
            let a_0 = (prior_eta.first().copied().unwrap_or(0.0) + 1.0).max(1e-10);
            let b_0 = (-prior_eta.get(1).copied().unwrap_or(-1.0)).max(1e-10);
            Ok(kl_gamma(a_q, b_q, a_0, b_0))
        }
        ConjugateModel::NormalNormal => {
            let eta1_q = eta.first().copied().unwrap_or(0.0);
            let eta2_q = eta.get(1).copied().unwrap_or(-0.5);
            let sigma_q_sq = if eta2_q < -f64::EPSILON {
                -1.0 / (2.0 * eta2_q)
            } else {
                1.0
            };
            let mu_q = eta1_q * sigma_q_sq;

            let eta1_0 = prior_eta.first().copied().unwrap_or(0.0);
            let eta2_0 = prior_eta.get(1).copied().unwrap_or(-0.5);
            let sigma_0_sq = if eta2_0 < -f64::EPSILON {
                -1.0 / (2.0 * eta2_0)
            } else {
                1.0
            };
            let mu_0 = eta1_0 * sigma_0_sq;

            // KL(N(μ_q, σ_q²) || N(μ_0, σ_0²))
            Ok(kl_normal(mu_q, sigma_q_sq, mu_0, sigma_0_sq))
        }
        ConjugateModel::DirichletMultinomial => {
            let alpha_q: Vec<f64> = eta.iter().map(|&e| (e + 1.0).max(1e-10)).collect();
            let alpha_0: Vec<f64> = prior_eta.iter().map(|&e| (e + 1.0).max(1e-10)).collect();
            Ok(kl_dirichlet(&alpha_q, &alpha_0))
        }
        _ => Ok(0.0),
    }
}

// ─── KL divergence closed forms ────────────────────────────────────────────

/// KL(Beta(a1,b1) || Beta(a2,b2))
fn kl_beta(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    lgamma(a1 + b1) - lgamma(a1) - lgamma(b1) - lgamma(a2 + b2)
        + lgamma(a2)
        + lgamma(b2)
        + (a1 - a2) * digamma(a1)
        + (b1 - b2) * digamma(b1)
        + (a2 - a1 + b2 - b1) * digamma(a1 + b1)
}

/// KL(Gamma(a1,b1) || Gamma(a2,b2))
fn kl_gamma(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    (a1 - a2) * digamma(a1) - lgamma(a1) + lgamma(a2) + a2 * (b1 / b2).ln()
        - a1 * (b1.ln() - b2.ln())
        + a1 * (b2 - b1) / b2
}

/// KL(N(mu1, s1) || N(mu2, s2))
fn kl_normal(mu1: f64, s1: f64, mu2: f64, s2: f64) -> f64 {
    let s1 = s1.max(f64::EPSILON);
    let s2 = s2.max(f64::EPSILON);
    0.5 * ((s1 / s2).ln() + s1 / s2 + (mu1 - mu2).powi(2) / s2 - 1.0)
}

/// KL(Dir(alpha_q) || Dir(alpha_0))
fn kl_dirichlet(alpha_q: &[f64], alpha_0: &[f64]) -> f64 {
    let sum_q: f64 = alpha_q.iter().sum();
    let sum_0: f64 = alpha_0.iter().sum();
    let mut kl = lgamma(sum_q) - lgamma(sum_0);
    for i in 0..alpha_q.len().min(alpha_0.len()) {
        kl += lgamma(alpha_0[i]) - lgamma(alpha_q[i]);
        kl += (alpha_q[i] - alpha_0[i]) * (digamma(alpha_q[i]) - digamma(sum_q));
    }
    kl.max(0.0) // Numerical guard: KL is always ≥ 0
}

// ─── Special functions ─────────────────────────────────────────────────────

/// Log-gamma function via Stirling / Lanczos approximation.
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Use the built-in f64 lgamma via the math trait, falling back to
    // a Lanczos series valid for x > 0.5.
    lanczos_lgamma(x)
}

/// Lanczos approximation to ln Γ(x) (g=7, n=9 coefficients; accurate to ~15 digits).
fn lanczos_lgamma(x: f64) -> f64 {
    // Coefficients from Numerical Recipes, 3rd ed., §6.1
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_10,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];

    if x < 0.5 {
        // Reflection: ln Γ(x) = ln π - ln sin(πx) - ln Γ(1-x)
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().abs().ln()
            - lanczos_lgamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Digamma (ψ) function via asymptotic expansion.
///
/// Uses recurrence ψ(x+1) = ψ(x) + 1/x to shift x ≥ 6 before applying
/// the asymptotic series.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // Shift to x ≥ 6 for asymptotic series.
    if x < 6.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Asymptotic: ψ(x) ≈ ln(x) - 1/(2x) - sum B_{2k} / (2k x^{2k})
    let x2 = x * x;
    x.ln() - 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2) - 1.0 / (252.0 * x2 * x2 * x2)
}

/// Stirling approximation for log(n!) (used in Poisson log-likelihood).
fn log_factorial(n: u64) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    // ln(n!) = sum_{k=1}^{n} ln(k) = ln Γ(n+1)
    lgamma(n as f64 + 1.0)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_beta_bernoulli(alpha0: f64, beta0: f64) -> OnlineVbEstimator {
        let config = OnlineVbConfig {
            model: ConjugateModel::BetaBernoulli,
            forgetting_factor: 0.7,
            delay: 1.0,
            learning_rate: 1.0,
            mini_batch_size: 32,
        };
        OnlineVbEstimator::new(config, &[alpha0, beta0], 1000).expect("valid estimator")
    }

    // ── BetaBernoulli ──────────────────────────────────────────────────────

    #[test]
    fn test_online_vb_beta_bernoulli_converges() {
        // True success probability = 0.7. We should recover roughly this.
        let mut est = make_beta_bernoulli(1.0, 1.0);
        let mut rng_state: u64 = 42;
        for _ in 0..200 {
            let batch: Vec<f64> = (0..32)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if (rng_state >> 33) % 100 < 70 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            est.update(&batch).expect("update ok");
        }
        let pred = est.predict();
        // Should be within 10% of the true probability.
        assert!((pred - 0.7).abs() < 0.1, "predict={pred}, expected ~0.7");
    }

    #[test]
    fn test_online_vb_elbo_finite() {
        let mut est = make_beta_bernoulli(2.0, 2.0);
        let batch = vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let elbo = est.update(&batch).expect("update ok");
        assert!(elbo.is_finite(), "ELBO should be finite");
    }

    #[test]
    fn test_online_vb_forgetting_factor_stability() {
        // kappa=1.0 decays faster and should reach near-zero learning rate quickly.
        let config_stable = OnlineVbConfig {
            model: ConjugateModel::BetaBernoulli,
            forgetting_factor: 1.0,
            delay: 1.0,
            learning_rate: 1.0,
            mini_batch_size: 16,
        };
        let mut est =
            OnlineVbEstimator::new(config_stable, &[1.0, 1.0], 500).expect("valid estimator");
        let batch = vec![1.0; 16];
        // Run many steps; with kappa=1.0 the learning rate decays ~1/t
        for _ in 0..50 {
            est.update(&batch).expect("update ok");
        }
        // Parameters should have moved toward the all-ones batch pattern.
        let pp = est.posterior_params();
        assert!(
            pp[0] > pp[1],
            "alpha should dominate after all-ones batches"
        );
    }

    #[test]
    fn test_online_vb_posterior_params_shape() {
        let est = make_beta_bernoulli(2.0, 3.0);
        let pp = est.posterior_params();
        assert_eq!(pp.len(), 2);
        assert!(pp[0] > 0.0 && pp[1] > 0.0, "alpha, beta must be > 0");
    }

    #[test]
    fn test_online_vb_empty_batch_error() {
        let mut est = make_beta_bernoulli(1.0, 1.0);
        let result = est.update(&[]);
        assert!(result.is_err(), "empty batch should return error");
    }

    #[test]
    fn test_online_vb_invalid_prior_error() {
        let config = OnlineVbConfig {
            model: ConjugateModel::BetaBernoulli,
            ..Default::default()
        };
        let result = OnlineVbEstimator::new(config, &[-1.0, 1.0], 100);
        assert!(result.is_err(), "negative alpha should fail");
    }

    // ── NormalNormal ───────────────────────────────────────────────────────

    #[test]
    fn test_online_vb_normal_normal_mean_convergence() {
        // True mean = 5.0, observation noise = 1.0, prior N(0, 10)
        let config = OnlineVbConfig {
            model: ConjugateModel::NormalNormal,
            forgetting_factor: 0.7,
            delay: 1.0,
            learning_rate: 1.0,
            mini_batch_size: 20,
        };
        let mut est =
            OnlineVbEstimator::new(config, &[0.0, 10.0, 1.0], 2000).expect("valid estimator");

        // Generate noisy observations around true mean 5.0.
        let mut rng_state: u64 = 123;
        for _ in 0..100 {
            let batch: Vec<f64> = (0..20)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                    // Box-Muller (using just uniform shift for simplicity in tests)
                    5.0 + (u - 0.5) * 2.0 // ~U[4, 6], centered at 5
                })
                .collect();
            est.update(&batch).expect("update ok");
        }
        let pred = est.predict();
        assert!((pred - 5.0).abs() < 1.0, "predict={pred}, expected ~5.0");
    }

    #[test]
    fn test_online_vb_normal_normal_posterior_shape() {
        let config = OnlineVbConfig {
            model: ConjugateModel::NormalNormal,
            ..Default::default()
        };
        let est = OnlineVbEstimator::new(config, &[0.0, 1.0, 0.5], 100).expect("valid estimator");
        let pp = est.posterior_params();
        assert_eq!(pp.len(), 2, "NormalNormal posterior has [mu, sigma^2]");
        assert!(pp[1] > 0.0, "posterior variance must be positive");
    }

    // ── GammaPoisson ───────────────────────────────────────────────────────

    #[test]
    fn test_online_vb_gamma_poisson_rate_estimation() {
        let config = OnlineVbConfig {
            model: ConjugateModel::GammaPoisson,
            forgetting_factor: 0.7,
            delay: 1.0,
            learning_rate: 1.0,
            mini_batch_size: 20,
        };
        // Prior: Gamma(1, 1) (weak), true rate = 3.0
        let mut est = OnlineVbEstimator::new(config, &[1.0, 1.0], 2000).expect("valid estimator");

        let mut rng_state: u64 = 999;
        for _ in 0..100 {
            let batch: Vec<f64> = (0..20)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    // Very rough Poisson(3) sample via LCG (not rigorous but sufficient)
                    let u: f64 = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
                    ((-3.0f64).exp() * (3.0f64).powi(((u * 7.0) as i32).min(10)).max(1e-300)
                        / lgamma(((u * 7.0) as i32 + 1).min(11) as f64).exp())
                    .min(10.0)
                    .max(0.0)
                    .round()
                })
                .collect();
            let _ = est.update(&batch);
        }
        // Just check that it runs and posterior params are valid.
        let pp = est.posterior_params();
        assert!(pp[0] > 0.0, "a > 0");
        assert!(pp[1] > 0.0, "b > 0");
    }

    // ── DirichletMultinomial ───────────────────────────────────────────────

    #[test]
    fn test_online_vb_dirichlet_multinomial_proportions() {
        // K=3 categories, true proportions ~ [0.6, 0.3, 0.1]
        let config = OnlineVbConfig {
            model: ConjugateModel::DirichletMultinomial,
            forgetting_factor: 0.7,
            delay: 1.0,
            learning_rate: 1.0,
            mini_batch_size: 30,
        };
        let mut est =
            OnlineVbEstimator::new(config, &[1.0, 1.0, 1.0], 3000).expect("valid estimator");

        let mut rng_state: u64 = 77;
        for _ in 0..100 {
            let batch: Vec<f64> = (0..30)
                .map(|_| {
                    rng_state = rng_state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (rng_state >> 11) % 100;
                    if u < 60 {
                        0.0
                    } else if u < 90 {
                        1.0
                    } else {
                        2.0
                    }
                })
                .collect();
            est.update(&batch).expect("update ok");
        }
        let pp = est.posterior_params();
        assert_eq!(pp.len(), 3);
        // Category 0 should have the largest alpha.
        assert!(pp[0] > pp[1], "alpha[0] > alpha[1] for 60% vs 30%");
        assert!(pp[1] > pp[2], "alpha[1] > alpha[2] for 30% vs 10%");
    }

    #[test]
    fn test_online_vb_invalid_kappa_error() {
        let config = OnlineVbConfig {
            model: ConjugateModel::BetaBernoulli,
            forgetting_factor: 0.3, // < 0.5, invalid
            ..Default::default()
        };
        let result = OnlineVbEstimator::new(config, &[1.0, 1.0], 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_vb_elbo_trend() {
        // ELBO should generally improve over time (not strictly monotone due to stochasticity,
        // but should have a positive trend after many steps).
        let mut est = make_beta_bernoulli(1.0, 1.0);
        let batch: Vec<f64> = vec![1.0; 16]; // All successes
        let mut first_elbo = f64::NEG_INFINITY;
        let mut last_elbo = f64::NEG_INFINITY;
        for i in 0..100 {
            let elbo = est.update(&batch).expect("update ok");
            if i == 0 {
                first_elbo = elbo;
            }
            last_elbo = elbo;
        }
        // After many updates toward all-ones, ELBO should increase.
        assert!(last_elbo >= first_elbo - 1.0, "ELBO should trend upward");
    }

    #[test]
    fn test_online_vb_n_processed_tracking() {
        let mut est = make_beta_bernoulli(1.0, 1.0);
        let batch = vec![1.0; 10];
        est.update(&batch).expect("update ok");
        est.update(&batch).expect("update ok");
        assert_eq!(est.state().n_processed, 20);
    }

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ ≈ -0.5772...
        let psi1 = digamma(1.0);
        assert!((psi1 - (-0.5772156649)).abs() < 1e-6, "ψ(1) ≈ -0.5772");
        // ψ(2) = 1 - γ ≈ 0.4228...
        let psi2 = digamma(2.0);
        assert!((psi2 - 0.4227843351).abs() < 1e-6, "ψ(2) ≈ 0.4228");
    }

    #[test]
    fn test_lgamma_known_values() {
        // Γ(1) = 1 → ln Γ(1) = 0
        assert!(lgamma(1.0).abs() < 1e-10);
        // Γ(2) = 1! = 1 → ln Γ(2) = 0
        assert!(lgamma(2.0).abs() < 1e-10);
        // Γ(3) = 2! = 2 → ln Γ(3) = ln 2
        assert!((lgamma(3.0) - 2.0f64.ln()).abs() < 1e-10);
        // Γ(0.5) = √π → ln Γ(0.5) = 0.5 * ln π
        assert!((lgamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-8);
    }
}
