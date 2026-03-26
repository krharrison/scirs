//! Sequential Bayesian Model Comparison via SMC
//!
//! Implements Sequential Monte Carlo (SMC) samplers for comparing multiple
//! Bayesian models by computing their marginal likelihoods (evidences) and
//! Bayes factors.
//!
//! ## Algorithm
//!
//! For each model M_i:
//! 1. Initialise N particles from the prior: θ^j ~ p(θ | M_i)
//! 2. Anneal through a geometric temperature schedule β_1,...,β_T with β_T = 1
//! 3. At each step t:
//!    a. Reweight: log w^j += (β_t − β_{t-1}) · log p(x | θ^j, M_i)
//!    b. Estimate log evidence increment: Δ_t = log mean(exp(Δlog_w))
//!    c. Compute ESS; if ESS < α·N → multinomial resample
//!    d. MCMC refresh: n_mcmc_steps random-walk Metropolis-Hastings steps
//! 4. Accumulate: log p(x | M_i) ≈ Σ_t Δ_t
//!
//! ## Bayes Factors & Model Probabilities
//!
//! BF_{i,0} = p(x | M_i) / p(x | M_0)     (log BF = log p(x|M_i) − log p(x|M_0))
//! P(M_i | x) ∝ p(x | M_i) · P(M_i)        (softmax of log evidences under equal priors)
//!
//! # References
//! - Del Moral, Doucet & Jasra (2006). Sequential Monte Carlo samplers.
//! - Chopin & Robert (2010). Properties of nested sampling.
//! - Zhou, Johansen & Aston (2016). Towards automatic model comparison.

use crate::error::{StatsError, StatsResult};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{RngExt, SeedableRng};

// ============================================================================
// Public types
// ============================================================================

/// Result of a sequential Bayesian model comparison
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Log marginal likelihood log p(data | M_i) for each model (unnormalized)
    pub log_evidences: Vec<f64>,
    /// Log Bayes factors log BF_{i,0} w.r.t. the first model (model 0)
    pub log_bayes_factors: Vec<f64>,
    /// Posterior model probabilities P(M_i | data) (sum to 1)
    pub model_probabilities: Vec<f64>,
    /// Posterior samples for each model: `posterior_samples[model_i][particle_j]` = θ vector
    pub posterior_samples: Vec<Vec<Vec<f64>>>,
    /// ESS trajectory for each model (one value per temperature step)
    pub ess_trajectories: Vec<Vec<f64>>,
    /// Number of resamplings performed per model
    pub n_resamplings: Vec<usize>,
}

/// Configuration for SMC-based model comparison
#[derive(Debug, Clone)]
pub struct SmcModelComparisonConfig {
    /// Number of particles per model (default: 500)
    pub n_particles: usize,
    /// Number of MH steps per temperature for MCMC refresh (default: 5)
    pub n_mcmc_steps: usize,
    /// Relative ESS threshold triggering resampling (default: 0.5)
    pub ess_threshold: f64,
    /// Number of annealing temperatures (default: 10)
    pub n_temperatures: usize,
    /// Random seed (default: 42)
    pub seed: u64,
    /// MH proposal step size (default: 0.1)
    pub step_size: f64,
    /// Minimum temperature increment to avoid numerical issues (default: 1e-6)
    pub min_delta_beta: f64,
}

impl Default for SmcModelComparisonConfig {
    fn default() -> Self {
        SmcModelComparisonConfig {
            n_particles: 500,
            n_mcmc_steps: 5,
            ess_threshold: 0.5,
            n_temperatures: 10,
            seed: 42,
            step_size: 0.1,
            min_delta_beta: 1e-6,
        }
    }
}

/// Trait for Bayesian models used in SMC model comparison.
///
/// Each model defines its own prior and likelihood; the SMC algorithm handles
/// the tempering schedule and resampling.
pub trait BayesianModel: Send {
    /// Compute log p(θ | M) — the log prior density of parameters θ.
    fn log_prior(&self, params: &[f64]) -> f64;

    /// Compute log p(data | θ, M) — the log likelihood.
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64;

    /// Dimensionality of the parameter space.
    fn n_params(&self) -> usize;

    /// Draw a sample from the prior: θ ~ p(θ | M).
    fn prior_sample(&self, rng: &mut dyn DynRng) -> Vec<f64>;

    /// Propose a new parameter vector via a Gaussian random walk.
    ///
    /// `current` is the current state, `step_size` scales the proposal.
    fn proposal_step(&self, current: &[f64], step_size: f64, rng: &mut dyn DynRng) -> Vec<f64>;
}

/// Minimal RNG interface for `dyn` usage in trait methods.
pub trait DynRng {
    fn next_f64(&mut self) -> f64;
    fn next_normal(&mut self) -> f64;
}

/// Adapts any `R: RngExt` to `DynRng`.
pub struct DynRngAdapter<R: RngExt>(pub R);

impl<R: RngExt> DynRng for DynRngAdapter<R> {
    fn next_f64(&mut self) -> f64 {
        self.0.random::<f64>()
    }

    fn next_normal(&mut self) -> f64 {
        use std::f64::consts::TAU;
        let u1 = self.0.random::<f64>().max(1e-300);
        let u2 = self.0.random::<f64>();
        let r = (-2.0_f64 * u1.ln()).sqrt();
        r * (TAU * u2).cos()
    }
}

// ============================================================================
// Main SMC function
// ============================================================================

/// Run sequential Bayesian model comparison across multiple models.
///
/// # Arguments
/// * `models` - Slice of references to models implementing [`BayesianModel`]
/// * `data`   - Observed data passed to each model's log_likelihood
/// * `config` - SMC configuration
///
/// # Returns
/// A [`ModelComparison`] with log evidences, Bayes factors, posterior
/// model probabilities, and posterior samples.
pub fn smc_model_comparison(
    models: &[&dyn BayesianModel],
    data: &[f64],
    config: &SmcModelComparisonConfig,
) -> StatsResult<ModelComparison> {
    if models.is_empty() {
        return Err(StatsError::InvalidArgument(
            "At least one model is required".to_string(),
        ));
    }
    if config.n_particles < 2 {
        return Err(StatsError::InvalidArgument(
            "n_particles must be >= 2".to_string(),
        ));
    }
    if config.n_temperatures < 1 {
        return Err(StatsError::InvalidArgument(
            "n_temperatures must be >= 1".to_string(),
        ));
    }
    if config.ess_threshold <= 0.0 || config.ess_threshold > 1.0 {
        return Err(StatsError::InvalidArgument(
            "ess_threshold must be in (0, 1]".to_string(),
        ));
    }

    let n_models = models.len();
    let mut log_evidences = Vec::with_capacity(n_models);
    let mut posterior_samples_all = Vec::with_capacity(n_models);
    let mut ess_trajectories = Vec::with_capacity(n_models);
    let mut n_resamplings = Vec::with_capacity(n_models);

    // Run SMC independently for each model
    for (model_idx, &model) in models.iter().enumerate() {
        let seed = config.seed.wrapping_add(model_idx as u64 * 1_000_000);
        let (log_ev, particles, ess_traj, n_res) = run_smc_for_model(model, data, config, seed)?;

        log_evidences.push(log_ev);
        posterior_samples_all.push(particles);
        ess_trajectories.push(ess_traj);
        n_resamplings.push(n_res);
    }

    // Compute log Bayes factors relative to first model
    let log_ev_0 = log_evidences[0];
    let log_bayes_factors: Vec<f64> = log_evidences.iter().map(|&le| le - log_ev_0).collect();

    // Posterior model probabilities via softmax (equal model priors)
    let model_probabilities = softmax(&log_evidences);

    Ok(ModelComparison {
        log_evidences,
        log_bayes_factors,
        model_probabilities,
        posterior_samples: posterior_samples_all,
        ess_trajectories,
        n_resamplings,
    })
}

// ============================================================================
// Per-model SMC sampler
// ============================================================================

fn run_smc_for_model(
    model: &dyn BayesianModel,
    data: &[f64],
    config: &SmcModelComparisonConfig,
    seed: u64,
) -> StatsResult<(f64, Vec<Vec<f64>>, Vec<f64>, usize)> {
    let n = config.n_particles;
    let d = model.n_params();

    let mut rng_adapter = DynRngAdapter(SmallRng::seed_from_u64(seed));

    // Step 1: Initialize particles from prior
    let mut particles: Vec<Vec<f64>> = (0..n)
        .map(|_| model.prior_sample(&mut rng_adapter))
        .collect();

    // Log-weights (initially uniform ⟹ log(1/N) each)
    let mut log_weights: Vec<f64> = vec![0.0_f64; n]; // unnormalized, relative
    let mut log_evidence_total = 0.0_f64;

    // Geometric temperature schedule: 0 = β_0 < β_1 < ... < β_T = 1
    let betas = geometric_schedule(config.n_temperatures);

    let mut ess_traj = Vec::with_capacity(betas.len());
    let mut n_resamplings = 0_usize;

    let mut beta_prev = 0.0_f64;

    for &beta_curr in &betas {
        let delta_beta = (beta_curr - beta_prev).max(config.min_delta_beta);

        // Step 2: Reweight by incremental likelihood
        let mut log_incremental: Vec<f64> = Vec::with_capacity(n);
        for p in &particles {
            let ll = model.log_likelihood(p, data);
            log_incremental.push(delta_beta * ll);
        }

        // Accumulate log evidence increment: log(1/N Σ_i w_i * exp(Δlog p_i))
        // = log(Σ_i exp(log_w_i + Δlog_p_i)) - log(N) with current normalized weights
        // We track unnormalized log weights and gather the normalizer.
        for i in 0..n {
            log_weights[i] += log_incremental[i];
        }

        // Log normalizer increment: log mean weight = logsumexp(log_weights) - log N
        let lse = logsumexp(&log_weights);
        log_evidence_total += lse - (n as f64).ln();

        // Normalize log weights
        for lw in log_weights.iter_mut() {
            *lw -= lse;
        }

        // Compute ESS from normalized weights
        let weights: Vec<f64> = log_weights.iter().map(|&lw| lw.exp()).collect();
        let ess = effective_sample_size(&weights);
        ess_traj.push(ess);

        // Step 3: Resample if ESS < threshold * N
        let ess_rel = ess / n as f64;
        if ess_rel < config.ess_threshold {
            let indices = multinomial_resample(&weights, n, &mut rng_adapter);
            let new_particles: Vec<Vec<f64>> =
                indices.iter().map(|&i| particles[i].clone()).collect();
            particles = new_particles;
            log_weights = vec![0.0_f64; n]; // reset to uniform after resampling
            n_resamplings += 1;
        }

        // Step 4: MCMC refresh (Metropolis-Hastings at current temperature β_curr)
        for j in 0..n {
            for _ in 0..config.n_mcmc_steps {
                let proposed =
                    model.proposal_step(&particles[j], config.step_size, &mut rng_adapter);
                let log_accept =
                    mh_log_accept_ratio(model, data, &particles[j], &proposed, beta_curr);
                let u: f64 = rng_adapter.next_f64();
                if u.ln() < log_accept {
                    particles[j] = proposed;
                }
            }
        }

        beta_prev = beta_curr;
    }

    // If we haven't fully reached β=1 due to min_delta_beta, add a final step
    if beta_prev < 1.0 - 1e-10 {
        let delta_beta = 1.0 - beta_prev;
        let mut log_incremental_final: Vec<f64> = Vec::with_capacity(n);
        for p in &particles {
            let ll = model.log_likelihood(p, data);
            log_incremental_final.push(delta_beta * ll);
        }
        for i in 0..n {
            log_weights[i] += log_incremental_final[i];
        }
        let lse = logsumexp(&log_weights);
        log_evidence_total += lse - (n as f64).ln();
        for lw in log_weights.iter_mut() {
            *lw -= lse;
        }
    }

    // Final weighted particles for posterior samples
    // Return all particles (equally weighted after last resampling)
    let _ = d; // suppress unused warning
    Ok((log_evidence_total, particles, ess_traj, n_resamplings))
}

// ============================================================================
// MH acceptance ratio
// ============================================================================

fn mh_log_accept_ratio(
    model: &dyn BayesianModel,
    data: &[f64],
    current: &[f64],
    proposed: &[f64],
    beta: f64,
) -> f64 {
    let log_prior_curr = model.log_prior(current);
    let log_prior_prop = model.log_prior(proposed);
    let log_lik_curr = model.log_likelihood(current, data);
    let log_lik_prop = model.log_likelihood(proposed, data);

    // log π_β(θ') / π_β(θ) = log p(θ') + β log p(x|θ') - log p(θ) - β log p(x|θ)
    (log_prior_prop + beta * log_lik_prop) - (log_prior_curr + beta * log_lik_curr)
}

// ============================================================================
// Utility functions
// ============================================================================

/// Geometric temperature schedule: β_t = (t/T)^γ with γ=1 (linear).
fn geometric_schedule(n_temps: usize) -> Vec<f64> {
    if n_temps == 0 {
        return vec![1.0];
    }
    (1..=n_temps).map(|t| t as f64 / n_temps as f64).collect()
}

/// Numerically stable log-sum-exp.
fn logsumexp(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_x.is_finite() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = xs.iter().map(|&x| (x - max_x).exp()).sum();
    max_x + sum.ln()
}

/// Effective sample size from normalized weights: ESS = (Σw_i)² / Σw_i²
fn effective_sample_size(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|&w| w * w).sum();
    if sum_sq <= 0.0 {
        return 0.0;
    }
    let sum: f64 = weights.iter().sum();
    (sum * sum) / sum_sq
}

/// Multinomial resampling: draw N indices proportional to `weights`.
fn multinomial_resample(weights: &[f64], n: usize, rng: &mut dyn DynRng) -> Vec<usize> {
    // Build CDF
    let mut cdf = Vec::with_capacity(weights.len());
    let mut cumsum = 0.0_f64;
    for &w in weights {
        cumsum += w;
        cdf.push(cumsum);
    }

    let mut indices = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_f64();
        let idx = cdf.partition_point(|&c| c < u).min(weights.len() - 1);
        indices.push(idx);
    }
    indices
}

/// Softmax: exp(x_i) / Σ_j exp(x_j)
fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() {
        return vec![];
    }
    let max_x = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|&x| (x - max_x).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let n = xs.len();
        return vec![1.0 / n as f64; n];
    }
    exps.iter().map(|&e| e / sum).collect()
}

// ============================================================================
// Concrete model implementations for tests
// ============================================================================

/// Simple Gaussian observation model with known variance.
///
/// Likelihood: x_i ~ N(θ, σ²) where σ is fixed.
/// Prior:      θ ~ N(prior_mean, prior_std²).
#[derive(Debug, Clone)]
pub struct GaussianObservationModel {
    /// Prior mean for θ
    pub prior_mean: f64,
    /// Prior standard deviation for θ
    pub prior_std: f64,
    /// Fixed observation noise σ
    pub obs_std: f64,
}

impl GaussianObservationModel {
    /// Create a new GaussianObservationModel
    pub fn new(prior_mean: f64, prior_std: f64, obs_std: f64) -> Self {
        GaussianObservationModel {
            prior_mean,
            prior_std,
            obs_std,
        }
    }

    fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
        let d = (x - mu) / sigma;
        -0.5 * d * d - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

impl BayesianModel for GaussianObservationModel {
    fn log_prior(&self, params: &[f64]) -> f64 {
        if params.is_empty() {
            return f64::NEG_INFINITY;
        }
        Self::log_normal(params[0], self.prior_mean, self.prior_std)
    }

    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        if params.is_empty() || data.is_empty() {
            return 0.0;
        }
        let theta = params[0];
        data.iter()
            .map(|&x| Self::log_normal(x, theta, self.obs_std))
            .sum()
    }

    fn n_params(&self) -> usize {
        1
    }

    fn prior_sample(&self, rng: &mut dyn DynRng) -> Vec<f64> {
        let eps = rng.next_normal();
        vec![self.prior_mean + self.prior_std * eps]
    }

    fn proposal_step(&self, current: &[f64], step_size: f64, rng: &mut dyn DynRng) -> Vec<f64> {
        let eps = rng.next_normal();
        vec![current[0] + step_size * eps]
    }
}

/// Gaussian model with different noise level (for model comparison tests).
///
/// Acts as an alternative to [`GaussianObservationModel`] with a different σ.
#[derive(Debug, Clone)]
pub struct GaussianAlternativeModel {
    /// Fixed observation noise (different from the "true" model)
    pub obs_std: f64,
    /// Prior std for θ
    pub prior_std: f64,
}

impl GaussianAlternativeModel {
    /// Create a new GaussianAlternativeModel
    pub fn new(obs_std: f64, prior_std: f64) -> Self {
        GaussianAlternativeModel { obs_std, prior_std }
    }

    fn log_normal(x: f64, mu: f64, sigma: f64) -> f64 {
        let d = (x - mu) / sigma;
        -0.5 * d * d - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

impl BayesianModel for GaussianAlternativeModel {
    fn log_prior(&self, params: &[f64]) -> f64 {
        if params.is_empty() {
            return f64::NEG_INFINITY;
        }
        Self::log_normal(params[0], 0.0, self.prior_std)
    }

    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        if params.is_empty() || data.is_empty() {
            return 0.0;
        }
        let theta = params[0];
        data.iter()
            .map(|&x| Self::log_normal(x, theta, self.obs_std))
            .sum()
    }

    fn n_params(&self) -> usize {
        1
    }

    fn prior_sample(&self, rng: &mut dyn DynRng) -> Vec<f64> {
        let eps = rng.next_normal();
        vec![self.prior_std * eps]
    }

    fn proposal_step(&self, current: &[f64], step_size: f64, rng: &mut dyn DynRng) -> Vec<f64> {
        let eps = rng.next_normal();
        vec![current[0] + step_size * eps]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(n_particles: usize, n_temps: usize) -> SmcModelComparisonConfig {
        SmcModelComparisonConfig {
            n_particles,
            n_mcmc_steps: 3,
            ess_threshold: 0.5,
            n_temperatures: n_temps,
            seed: 42,
            step_size: 0.3,
            ..SmcModelComparisonConfig::default()
        }
    }

    /// Generate data from N(mu, sigma²)
    fn gen_data(n: usize, mu: f64, sigma: f64, seed: u64) -> Vec<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut adapter = DynRngAdapter(rng);
        (0..n).map(|_| mu + sigma * adapter.next_normal()).collect()
    }

    // ------------------------------------------------------------------
    // Basic functionality
    // ------------------------------------------------------------------

    #[test]
    fn test_smc_model_compare_basic() {
        let data = gen_data(20, 2.0, 0.5, 1);
        let m1 = GaussianObservationModel::new(0.0, 5.0, 0.5);
        let m2 = GaussianObservationModel::new(0.0, 5.0, 2.0);
        let config = make_config(100, 5);
        let result = smc_model_comparison(
            &[&m1 as &dyn BayesianModel, &m2 as &dyn BayesianModel],
            &data,
            &config,
        )
        .expect("smc_model_comparison failed");

        assert_eq!(result.log_evidences.len(), 2);
        assert_eq!(result.log_bayes_factors.len(), 2);
        assert_eq!(result.model_probabilities.len(), 2);
        // First BF is always 0 (reference model)
        assert!((result.log_bayes_factors[0]).abs() < 1e-10);
        for &le in &result.log_evidences {
            assert!(le.is_finite(), "log evidence not finite: {}", le);
        }
    }

    #[test]
    fn test_smc_bayes_factors_ordering() {
        // Data generated from N(2, 0.5²); model 1 (correct σ=0.5) should
        // have higher evidence than model 2 (wrong σ=2.0).
        let data = gen_data(50, 2.0, 0.5, 99);
        let m_correct = GaussianObservationModel::new(0.0, 5.0, 0.5);
        let m_wrong = GaussianObservationModel::new(0.0, 5.0, 2.0);
        let config = make_config(200, 10);

        let result = smc_model_comparison(
            &[
                &m_correct as &dyn BayesianModel,
                &m_wrong as &dyn BayesianModel,
            ],
            &data,
            &config,
        )
        .expect("SMC failed");

        // Correct model should be preferred (higher log evidence)
        assert!(
            result.log_evidences[0] > result.log_evidences[1],
            "Expected correct model to have higher evidence: {:.3} vs {:.3}",
            result.log_evidences[0],
            result.log_evidences[1]
        );
        // Positive Bayes factor for reference = log(1) = 0; model 1 BF > model 2 BF
        assert!(
            result.log_bayes_factors[1] < 0.0,
            "Expected wrong model to have negative log BF: {:.3}",
            result.log_bayes_factors[1]
        );
    }

    #[test]
    fn test_smc_ess_resampling() {
        // With a tight ESS threshold, resampling should trigger on most steps
        let data = gen_data(10, 0.0, 1.0, 7);
        let model = GaussianObservationModel::new(0.0, 1.0, 1.0);
        let config = SmcModelComparisonConfig {
            n_particles: 100,
            n_mcmc_steps: 2,
            ess_threshold: 0.99, // very aggressive: resample almost always
            n_temperatures: 5,
            seed: 5,
            ..SmcModelComparisonConfig::default()
        };
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("SMC failed");

        // With threshold 0.99, should have performed resampling
        assert!(
            result.n_resamplings[0] > 0,
            "Expected resampling to occur with aggressive ESS threshold"
        );
    }

    #[test]
    fn test_smc_posterior_samples_valid() {
        let data = gen_data(20, 1.0, 0.5, 13);
        let model = GaussianObservationModel::new(0.0, 3.0, 0.5);
        let config = make_config(200, 8);
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("SMC failed");

        let post = &result.posterior_samples[0];
        assert_eq!(post.len(), 200, "Expected 200 posterior particles");
        for p in post {
            assert_eq!(p.len(), 1, "Each particle should have 1 parameter");
            assert!(p[0].is_finite(), "Non-finite particle: {}", p[0]);
            // Posterior should be concentrated near the true mean
            assert!(p[0].abs() < 10.0, "Particle too extreme: {}", p[0]);
        }
    }

    #[test]
    fn test_smc_three_models() {
        let data = gen_data(20, 1.0, 0.5, 21);
        let m1 = GaussianObservationModel::new(0.0, 5.0, 0.5); // correct σ
        let m2 = GaussianObservationModel::new(0.0, 5.0, 1.0);
        let m3 = GaussianObservationModel::new(0.0, 5.0, 2.0);
        let config = make_config(150, 5);

        let result = smc_model_comparison(
            &[
                &m1 as &dyn BayesianModel,
                &m2 as &dyn BayesianModel,
                &m3 as &dyn BayesianModel,
            ],
            &data,
            &config,
        )
        .expect("3-model SMC failed");

        assert_eq!(result.model_probabilities.len(), 3);

        // Probabilities sum to 1
        let prob_sum: f64 = result.model_probabilities.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-10,
            "Probabilities don't sum to 1: {:.6}",
            prob_sum
        );

        // All probabilities in [0, 1]
        for &p in &result.model_probabilities {
            assert!(p >= 0.0 && p <= 1.0 + 1e-10, "Invalid probability: {}", p);
        }
    }

    #[test]
    fn test_smc_log_evidences_finite() {
        let data = gen_data(15, 0.0, 1.0, 3);
        let model = GaussianObservationModel::new(0.0, 2.0, 1.0);
        let config = make_config(100, 5);
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("SMC failed");

        assert!(result.log_evidences[0].is_finite());
        assert!((result.log_bayes_factors[0]).abs() < 1e-10);
        assert!((result.model_probabilities[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smc_empty_models_error() {
        let data = vec![1.0, 2.0];
        let config = make_config(10, 5);
        let result = smc_model_comparison(&[], &data, &config);
        assert!(result.is_err(), "Expected error for empty models");
    }

    #[test]
    fn test_smc_too_few_particles_error() {
        let data = gen_data(5, 0.0, 1.0, 0);
        let model = GaussianObservationModel::new(0.0, 1.0, 1.0);
        let config = SmcModelComparisonConfig {
            n_particles: 1, // too few
            ..SmcModelComparisonConfig::default()
        };
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config);
        assert!(result.is_err(), "Expected error for n_particles=1");
    }

    #[test]
    fn test_smc_ess_trajectory_length() {
        let data = gen_data(10, 0.0, 1.0, 4);
        let model = GaussianObservationModel::new(0.0, 1.0, 1.0);
        let n_temps = 8;
        let config = make_config(100, n_temps);
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("SMC failed");

        // ESS trajectory should have n_temperatures entries
        assert_eq!(
            result.ess_trajectories[0].len(),
            n_temps,
            "ESS trajectory length mismatch"
        );
        for &ess in &result.ess_trajectories[0] {
            assert!(ess >= 0.0, "Negative ESS: {}", ess);
            assert!(
                ess <= (config.n_particles + 1) as f64,
                "ESS too large: {}",
                ess
            );
        }
    }

    #[test]
    fn test_smc_single_temperature() {
        let data = gen_data(10, 0.0, 1.0, 8);
        let model = GaussianObservationModel::new(0.0, 1.0, 1.0);
        let config = make_config(100, 1);
        let result = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("SMC with 1 temperature failed");

        assert!(result.log_evidences[0].is_finite());
    }

    #[test]
    fn test_smc_alternative_model() {
        // Use GaussianAlternativeModel to ensure it works with the trait
        let data = gen_data(20, 0.0, 1.0, 44);
        let m1 = GaussianObservationModel::new(0.0, 2.0, 1.0);
        let m2 = GaussianAlternativeModel::new(1.0, 2.0);
        let config = make_config(150, 5);
        let result = smc_model_comparison(
            &[&m1 as &dyn BayesianModel, &m2 as &dyn BayesianModel],
            &data,
            &config,
        )
        .expect("SMC with alternative model failed");

        assert_eq!(result.log_evidences.len(), 2);
        let prob_sum: f64 = result.model_probabilities.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smc_reproducibility() {
        let data = gen_data(15, 1.0, 0.5, 77);
        let model = GaussianObservationModel::new(0.0, 3.0, 0.5);
        let config = make_config(100, 5);

        let r1 = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("first SMC failed");
        let r2 = smc_model_comparison(&[&model as &dyn BayesianModel], &data, &config)
            .expect("second SMC failed");

        // Same seed → same log evidence
        assert!(
            (r1.log_evidences[0] - r2.log_evidences[0]).abs() < 1e-10,
            "Non-reproducible: {:.6} vs {:.6}",
            r1.log_evidences[0],
            r2.log_evidences[0]
        );
    }
}
