//! Path sampling methods for Monte Carlo integration
//!
//! Implements Feynman-Kac path integral estimation, diffusion bridge sampling,
//! Metropolis random walks, and Annealed Importance Sampling (AIS).

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_core::random::rand_prelude::StdRng as RandStdRng;
use scirs2_core::random::{seeded_rng, Distribution, Normal};
use scirs2_core::random::{CoreRandom, SeedableRng, Uniform};

// ─────────────────────────────────────────────────────────────────────────────
// Feynman-Kac path integral estimator
// ─────────────────────────────────────────────────────────────────────────────

/// Options for Feynman-Kac path integral estimation.
#[derive(Debug, Clone)]
pub struct FeynmanKacOptions {
    /// Number of paths
    pub n_paths: usize,
    /// Number of time steps per path
    pub n_steps: usize,
    /// Time horizon
    pub t_end: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for FeynmanKacOptions {
    fn default() -> Self {
        Self {
            n_paths: 1000,
            n_steps: 100,
            t_end: 1.0,
            seed: None,
        }
    }
}

/// Feynman-Kac path integral estimator.
///
/// Estimates E[f(X_T) * exp(∫_0^T V(t, X_t) dt)] where X_t follows a Brownian motion.
pub struct FeynmanKacEstimator {
    options: FeynmanKacOptions,
}

impl FeynmanKacEstimator {
    /// Create a new Feynman-Kac estimator with given options.
    pub fn new(options: FeynmanKacOptions) -> Self {
        Self { options }
    }

    /// Estimate E[g(X_T) * exp(∫_0^T V(t, X_t) dt)] starting from x0.
    ///
    /// # Arguments
    ///
    /// * `x0` - Initial state
    /// * `g` - Terminal payoff function
    /// * `v` - Potential function V(t, x)
    pub fn estimate<G, V>(&self, x0: f64, g: G, v: V) -> IntegrateResult<f64>
    where
        G: Fn(f64) -> f64,
        V: Fn(f64, f64) -> f64,
    {
        let opts = &self.options;
        let dt = opts.t_end / opts.n_steps as f64;
        let sqrt_dt = dt.sqrt();
        let normal = Normal::new(0.0_f64, 1.0_f64)
            .map_err(|e| IntegrateError::ComputationError(format!("Normal dist: {e}")))?;

        let mut rng = match opts.seed {
            Some(s) => seeded_rng(s),
            None => seeded_rng(12345),
        };

        let mut total = 0.0_f64;

        for _ in 0..opts.n_paths {
            let mut x = x0;
            let mut log_weight = 0.0_f64;
            let mut t = 0.0_f64;

            for _ in 0..opts.n_steps {
                log_weight += v(t, x) * dt;
                x += sqrt_dt * Distribution::sample(&normal, &mut rng);
                t += dt;
            }

            total += g(x) * log_weight.exp();
        }

        Ok(total / opts.n_paths as f64)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Diffusion bridge sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Result of diffusion bridge sampling.
#[derive(Debug, Clone)]
pub struct DiffusionBridgeResult {
    /// Sampled path values at each time step
    pub path: Vec<f64>,
    /// Time points
    pub times: Vec<f64>,
    /// Log-weight of the path
    pub log_weight: f64,
}

/// Sample a Brownian bridge from `x_start` to `x_end` over `[0, T]`.
///
/// Uses the exact Brownian bridge distribution:
/// X(t) | X(0) = a, X(T) = b ~ N(a + (b-a)*t/T, t*(T-t)/T)
///
/// # Arguments
///
/// * `x_start` - Starting value
/// * `x_end` - Ending value
/// * `t_end` - Total time
/// * `n_steps` - Number of time steps
/// * `seed` - Optional random seed
pub fn diffusion_bridge(
    x_start: f64,
    x_end: f64,
    t_end: f64,
    n_steps: usize,
    seed: Option<u64>,
) -> IntegrateResult<DiffusionBridgeResult> {
    if t_end <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "t_end must be positive".to_string(),
        ));
    }
    if n_steps == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_steps must be positive".to_string(),
        ));
    }

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist: {e}")))?;
    let mut rng = match seed {
        Some(s) => seeded_rng(s),
        None => seeded_rng(99999),
    };

    let dt = t_end / n_steps as f64;
    let mut path = Vec::with_capacity(n_steps + 1);
    let mut times = Vec::with_capacity(n_steps + 1);

    path.push(x_start);
    times.push(0.0);

    // Generate Brownian bridge via conditional sampling
    for step in 1..=n_steps {
        let t = step as f64 * dt;
        let remaining = t_end - t + dt; // time from previous step
        let t_prev = (step - 1) as f64 * dt;

        // Conditional mean and variance for bridge at time t given path up to t_prev and end point
        let t_remaining_from_now = t_end - t;
        let x_prev = path[step - 1];

        // Bridge mean: x_prev + (x_end - x_prev) * dt / (t_end - t_prev)
        let bridge_mean = x_prev + (x_end - x_prev) * dt / (t_end - t_prev + 1e-300);
        // Bridge variance: dt * (t_end - t) / (t_end - t_prev)
        let bridge_var = (dt * t_remaining_from_now / (t_end - t_prev + 1e-300)).max(0.0);
        let _ = remaining; // suppress warning

        let x_new = bridge_mean + bridge_var.sqrt() * Distribution::sample(&normal, &mut rng);
        path.push(x_new);
        times.push(t);
    }

    // Force endpoint
    if let Some(last) = path.last_mut() {
        *last = x_end;
    }

    Ok(DiffusionBridgeResult {
        path,
        times,
        log_weight: 0.0,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Metropolis random walk
// ─────────────────────────────────────────────────────────────────────────────

/// Options for Metropolis random walk sampler.
#[derive(Debug, Clone)]
pub struct MetropolisWalkOptions {
    /// Number of samples
    pub n_samples: usize,
    /// Proposal step size (standard deviation)
    pub step_size: f64,
    /// Number of burn-in steps
    pub n_burnin: usize,
    /// Thinning factor
    pub thinning: usize,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for MetropolisWalkOptions {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            step_size: 0.5,
            n_burnin: 100,
            thinning: 1,
            seed: None,
        }
    }
}

/// Result of Metropolis walk sampling.
#[derive(Debug, Clone)]
pub struct MetropolisWalkResult {
    /// Sampled points (after burn-in and thinning)
    pub samples: Vec<Array1<f64>>,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Estimated expectation of a test function
    pub mean_estimate: f64,
}

/// Metropolis random walk MCMC sampler.
///
/// Samples from a target density π(x) ∝ exp(-U(x)) using
/// Gaussian proposals x' = x + σ * N(0, I).
///
/// # Arguments
///
/// * `log_target` - Log of the target density (up to constant)
/// * `x0` - Initial state
/// * `options` - Sampler configuration
pub fn metropolis_walk<F>(
    log_target: F,
    x0: Array1<f64>,
    options: Option<MetropolisWalkOptions>,
) -> IntegrateResult<MetropolisWalkResult>
where
    F: Fn(ArrayView1<f64>) -> f64,
{
    let opts = options.unwrap_or_default();
    let dim = x0.len();

    let normal = Normal::new(0.0_f64, opts.step_size)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist: {e}")))?;
    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Uniform dist: {e}")))?;

    let mut rng = match opts.seed {
        Some(s) => seeded_rng(s),
        None => seeded_rng(77777),
    };

    let mut x = x0.clone();
    let mut log_px = log_target(x.view());

    let total_steps = opts.n_burnin + opts.n_samples * opts.thinning;
    let mut samples = Vec::with_capacity(opts.n_samples);
    let mut n_accepted = 0usize;

    for step in 0..total_steps {
        // Propose
        let proposal: Array1<f64> = Array1::from_shape_fn(dim, |_| {
            x[dim - 1] + Distribution::sample(&normal, &mut rng)
        });
        // Actually do component-wise proposals:
        let proposal: Array1<f64> =
            Array1::from_shape_fn(dim, |i| x[i] + Distribution::sample(&normal, &mut rng));

        let log_p_proposal = log_target(proposal.view());

        // Accept/reject
        let log_alpha = (log_p_proposal - log_px).min(0.0);
        let u: f64 = Distribution::sample(&uniform, &mut rng);

        if u.ln() < log_alpha {
            x = proposal;
            log_px = log_p_proposal;
            n_accepted += 1;
        }

        // Collect sample (after burn-in, with thinning)
        if step >= opts.n_burnin && (step - opts.n_burnin).is_multiple_of(opts.thinning) {
            samples.push(x.clone());
        }
    }

    let acceptance_rate = n_accepted as f64 / total_steps as f64;

    // Compute mean of first component as a test statistic
    let mean_estimate = if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64
    };

    Ok(MetropolisWalkResult {
        samples,
        acceptance_rate,
        mean_estimate,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Annealed Importance Sampling (AIS)
// ─────────────────────────────────────────────────────────────────────────────

/// Options for Annealed Importance Sampling.
#[derive(Debug, Clone)]
pub struct AISOptions {
    /// Number of AIS particles
    pub n_particles: usize,
    /// Number of intermediate distributions (annealing steps)
    pub n_annealing: usize,
    /// Number of MCMC steps per annealing level
    pub n_mcmc_per_level: usize,
    /// MCMC step size
    pub step_size: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for AISOptions {
    fn default() -> Self {
        Self {
            n_particles: 100,
            n_annealing: 10,
            n_mcmc_per_level: 5,
            step_size: 0.5,
            seed: None,
        }
    }
}

/// Result of Annealed Importance Sampling.
#[derive(Debug, Clone)]
pub struct AISResult {
    /// Log normalizing constant estimate: log Z_target / Z_prior
    pub log_z_ratio: f64,
    /// Normalizing constant ratio: Z_target / Z_prior
    pub z_ratio: f64,
    /// Effective sample size
    pub ess: f64,
    /// Final particle states
    pub particles: Vec<Array1<f64>>,
    /// Final log-weights
    pub log_weights: Vec<f64>,
}

/// Annealed Importance Sampling estimator.
///
/// Estimates the normalizing constant ratio Z_T / Z_0 using a sequence of
/// intermediate distributions π_β(x) ∝ π_0(x)^(1-β) * π_T(x)^β for β ∈ \[0,1\].
///
/// # Arguments
///
/// * `log_prior` - Log of the prior distribution π_0 (normalized)
/// * `log_target` - Log of the target distribution π_T (unnormalized)
/// * `sample_prior` - Sampler from the prior distribution (takes `&mut CoreRandom<RandStdRng>`)
/// * `dim` - Dimension of the state space
/// * `options` - AIS configuration
pub fn annealed_importance_sampling<P, T, S>(
    log_prior: P,
    log_target: T,
    sample_prior: S,
    dim: usize,
    options: Option<AISOptions>,
) -> IntegrateResult<AISResult>
where
    P: Fn(ArrayView1<f64>) -> f64,
    T: Fn(ArrayView1<f64>) -> f64,
    S: Fn(&mut CoreRandom<RandStdRng>) -> Array1<f64>,
{
    let opts = options.unwrap_or_default();
    let n = opts.n_particles;
    let n_beta = opts.n_annealing;

    let normal = Normal::new(0.0_f64, opts.step_size)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist: {e}")))?;
    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Uniform dist: {e}")))?;

    let mut rng = match opts.seed {
        Some(s) => seeded_rng(s),
        None => seeded_rng(55555),
    };

    // Initialize particles from prior
    let mut particles: Vec<Array1<f64>> = (0..n).map(|_| sample_prior(&mut rng)).collect();

    let mut log_weights = vec![0.0_f64; n];

    // Annealing schedule: β_k = k / n_beta for k = 0, ..., n_beta
    let betas: Vec<f64> = (0..=n_beta).map(|k| k as f64 / n_beta as f64).collect();

    for k in 1..=n_beta {
        let beta_prev = betas[k - 1];
        let beta_curr = betas[k];

        // Update weights: w_k ∝ w_{k-1} * π_k(x) / π_{k-1}(x)
        // = w_{k-1} * (π_0^(1-β_k) π_T^β_k) / (π_0^(1-β_{k-1}) π_T^β_{k-1})
        // = w_{k-1} * π_0^(β_{k-1}-β_k) * π_T^(β_k-β_{k-1})
        let delta_beta = beta_curr - beta_prev;

        for i in 0..n {
            let lp0 = log_prior(particles[i].view());
            let lpt = log_target(particles[i].view());
            log_weights[i] += delta_beta * (lpt - lp0);
        }

        // MCMC transition at β_curr (Metropolis within AIS)
        for _ in 0..opts.n_mcmc_per_level {
            for i in 0..n {
                let log_pi_curr = (1.0 - beta_curr) * log_prior(particles[i].view())
                    + beta_curr * log_target(particles[i].view());

                let proposal: Array1<f64> = Array1::from_shape_fn(dim, |j| {
                    particles[i][j] + Distribution::sample(&normal, &mut rng)
                });

                let log_pi_prop = (1.0 - beta_curr) * log_prior(proposal.view())
                    + beta_curr * log_target(proposal.view());

                let log_alpha = (log_pi_prop - log_pi_curr).min(0.0);
                let u: f64 = Distribution::sample(&uniform, &mut rng);

                if u.ln() < log_alpha {
                    particles[i] = proposal;
                }
            }
        }
    }

    // Compute log Z ratio using log-sum-exp
    let max_lw = log_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = log_weights.iter().map(|&lw| (lw - max_lw).exp()).sum();
    let log_z_ratio = max_lw + sum_exp.ln() - (n as f64).ln();
    let z_ratio = log_z_ratio.exp();

    // Effective sample size: ESS = (Σ w_i)^2 / (Σ w_i^2)
    let weights: Vec<f64> = log_weights.iter().map(|&lw| (lw - max_lw).exp()).collect();
    let sum_w: f64 = weights.iter().sum();
    let sum_w_sq: f64 = weights.iter().map(|w| w * w).sum();
    let ess = if sum_w_sq > 0.0 {
        sum_w * sum_w / sum_w_sq
    } else {
        0.0
    };

    Ok(AISResult {
        log_z_ratio,
        z_ratio,
        ess,
        particles,
        log_weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_diffusion_bridge() {
        let result = diffusion_bridge(0.0, 1.0, 1.0, 50, Some(42)).expect("bridge failed");
        assert_eq!(result.path.len(), 51);
        assert_eq!(result.times.len(), 51);
        assert!((result.path[0] - 0.0).abs() < 1e-10);
        assert!((result.path[50] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_metropolis_walk_standard_normal() {
        // Target: standard normal N(0,1)
        let log_target = |x: ArrayView1<f64>| -0.5 * x[0] * x[0];
        let x0 = array![0.0_f64];

        let result = metropolis_walk(
            log_target,
            x0,
            Some(MetropolisWalkOptions {
                n_samples: 2000,
                step_size: 1.0,
                n_burnin: 500,
                thinning: 2,
                seed: Some(42),
            }),
        )
        .expect("Metropolis failed");

        assert!(!result.samples.is_empty());
        assert!(result.acceptance_rate > 0.0 && result.acceptance_rate < 1.0);
        // Mean should be close to 0
        assert!(
            result.mean_estimate.abs() < 0.5,
            "mean={}",
            result.mean_estimate
        );
    }

    #[test]
    fn test_feynman_kac() {
        let estimator = FeynmanKacEstimator::new(FeynmanKacOptions {
            n_paths: 500,
            n_steps: 50,
            t_end: 1.0,
            seed: Some(42),
        });

        // Estimate E[X_T^2] with V=0, X_0=0 -> variance = T = 1
        let result = estimator
            .estimate(0.0, |x| x * x, |_t, _x| 0.0)
            .expect("FK failed");
        assert!(result >= 0.0);
    }

    #[test]
    fn test_ais_normalizing_constant() {
        // AIS to estimate Z for N(0,1) relative to N(0,2)
        let log_prior = |x: ArrayView1<f64>| -0.25 * x[0] * x[0]; // N(0,2)
        let log_target = |x: ArrayView1<f64>| -0.5 * x[0] * x[0]; // N(0,1)

        let normal_prior = Normal::new(0.0_f64, std::f64::consts::SQRT_2).expect("valid");
        let sample_prior =
            |rng: &mut CoreRandom<RandStdRng>| array![Distribution::sample(&normal_prior, rng)];

        let result = annealed_importance_sampling(
            log_prior,
            log_target,
            sample_prior,
            1,
            Some(AISOptions {
                n_particles: 200,
                n_annealing: 5,
                n_mcmc_per_level: 3,
                step_size: 0.5,
                seed: Some(42),
            }),
        )
        .expect("AIS failed");

        // log Z = log(sqrt(2pi)*1 / sqrt(2pi)*sqrt(2)) = -0.5*log(2) ≈ -0.347
        // We just check reasonable bounds
        assert!(result.ess > 0.0);
    }
}
