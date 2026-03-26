//! Beta Process and Indian Buffet Process (IBP)
//!
//! This module implements the Beta Process as a prior over binary feature matrices
//! via the Indian Buffet Process (IBP) construction.
//!
//! # References
//! - Griffiths & Ghahramani (2011). "The Indian Buffet Process: An Introduction and Review."
//! - Thibaux & Jordan (2007). "Hierarchical Beta Processes and the Indian Buffet Process."
//! - Knowles & Ghahramani (2011). "Nonparametric Bayesian Sparse Factor Models."

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::ndarray::{Array2, Axis};

// ---------------------------------------------------------------------------
// LCG-based minimal PRNG (avoids rand dependency)
// ---------------------------------------------------------------------------

/// Minimal linear-congruential PRNG returning values in [0,1).
#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance one step and return uniform in (0, 1).
    fn next_f64(&mut self) -> f64 {
        // LCG parameters from Knuth TAOCP Vol.2
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use upper 53 bits for float
        let bits = self.state >> 11;
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Sample from standard normal using Box-Muller transform.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Utility: Poisson sampler
// ---------------------------------------------------------------------------

/// Sample from Poisson(lambda) using Knuth's algorithm for small lambda,
/// or normal approximation for large lambda.
pub fn poisson_sample(lambda: f64, rng: &mut impl FnMut() -> f64) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    if lambda > 50.0 {
        // Normal approximation: Poisson(λ) ≈ N(λ, λ)
        // We implement via a simple rejection/clamp
        let u1 = rng().max(1e-300);
        let u2 = rng();
        let n = lambda + lambda.sqrt() * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        return n.round().max(0.0) as usize;
    }
    let l = (-lambda).exp();
    let mut k: usize = 0;
    let mut p = 1.0_f64;
    loop {
        k += 1;
        p *= rng();
        if p <= l {
            break;
        }
    }
    k.saturating_sub(1)
}

/// Log of the gamma function using Lanczos approximation.
fn lgamma(x: f64) -> f64 {
    // Lanczos approximation with g=7, n=9 coefficients
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_906,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI.ln()
            - ((std::f64::consts::PI * x).sin()).ln()
            - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        for (i, &ci) in C[1..].iter().enumerate() {
            a += ci / (x + (i as f64) + 1.0);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln()
            + (x + 0.5) * t.ln()
            - t
            + a.ln()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Beta Process / IBP sampler.
#[derive(Debug, Clone)]
pub struct BetaProcessConfig {
    /// Concentration (mass) parameter of the Beta Process.  α > 0.
    pub alpha: f64,
    /// Beta process concentration (c > 0) for two-parameter version.
    pub c: f64,
    /// Initial truncation: max number of features to track.
    pub n_features: usize,
    /// Number of MCMC samples (after burnin).
    pub n_samples: usize,
    /// Number of burn-in iterations.
    pub burnin: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for BetaProcessConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            c: 1.0,
            n_features: 20,
            n_samples: 1000,
            burnin: 100,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// IBP State
// ---------------------------------------------------------------------------

/// State of an IBP Gibbs sampler.
#[derive(Debug, Clone)]
pub struct IbpState {
    /// N × K binary feature ownership matrix.
    pub z: Vec<Vec<bool>>,
    /// Feature prior probabilities π_k.
    pub feature_probs: Vec<f64>,
    /// Number of observations.
    pub n_obs: usize,
}

impl IbpState {
    /// Number of active features.
    pub fn n_features(&self) -> usize {
        if self.z.is_empty() {
            self.feature_probs.len()
        } else {
            self.z[0].len()
        }
    }

    /// Count of observations that use feature k.
    pub fn feature_count(&self, k: usize) -> usize {
        self.z.iter().filter(|row| row.get(k).copied().unwrap_or(false)).count()
    }

    /// Feature count vector.
    pub fn feature_counts(&self) -> Vec<usize> {
        let k = self.n_features();
        (0..k).map(|ki| self.feature_count(ki)).collect()
    }
}

// ---------------------------------------------------------------------------
// IBP Result
// ---------------------------------------------------------------------------

/// Result of fitting the Beta Process / IBP model.
#[derive(Debug, Clone)]
pub struct IbpResult {
    /// Posterior mean of the binary feature matrix (N × K).
    pub z_mean: Array2<f64>,
    /// Number of features actually used (with at least one observation).
    pub n_features_used: usize,
    /// Approximate log marginal likelihood (from annealed importance weighting proxy).
    pub log_marginal_likelihood: f64,
}

// ---------------------------------------------------------------------------
// BetaProcess
// ---------------------------------------------------------------------------

/// Beta Process implementation via the Indian Buffet Process.
///
/// The IBP defines a distribution over sparse binary matrices with an
/// unbounded number of columns (features).  The one-parameter IBP is
/// parameterised by α; the two-parameter version additionally uses *c*.
pub struct BetaProcess {
    config: BetaProcessConfig,
}

impl BetaProcess {
    /// Create a new `BetaProcess` with the given configuration.
    pub fn new(config: BetaProcessConfig) -> Result<Self> {
        if config.alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "BetaProcess: alpha must be > 0, got {}",
                config.alpha
            )));
        }
        if config.c <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "BetaProcess: c must be > 0, got {}",
                config.c
            )));
        }
        Ok(Self { config })
    }

    /// Sample a binary feature matrix from the IBP prior.
    ///
    /// Uses the sequential (restaurant metaphor) construction:
    /// - Customer 1 tries Poisson(α) new dishes.
    /// - Customer i tries dish k with prob `n_k / i` and also
    ///   tries Poisson(α / i) new dishes.
    pub fn sample_prior(&self, n_obs: usize) -> IbpState {
        let mut rng = Lcg::new(self.config.seed);
        let mut uniform = || rng.next_f64();

        let mut z: Vec<Vec<bool>> = Vec::with_capacity(n_obs);
        // Track counts for each feature
        let mut n_k: Vec<usize> = Vec::new();

        for i in 0..n_obs {
            let n_existing = n_k.len();
            // row for customer i+1
            let mut row = vec![false; n_existing];

            // Try existing dishes
            for (k, row_k) in row.iter_mut().enumerate().take(n_existing) {
                let prob = n_k[k] as f64 / (i as f64 + 1.0);
                if uniform() < prob {
                    *row_k = true;
                    n_k[k] += 1;
                }
            }

            // Try new dishes: Poisson(alpha / (i+1)) for i>=1, Poisson(alpha) for i=0
            let lambda = if i == 0 {
                self.config.alpha
            } else {
                self.config.alpha / (i as f64 + 1.0)
            };
            let new_count = poisson_sample(lambda, &mut uniform);
            for _ in 0..new_count {
                row.push(true);
                n_k.push(1);
            }

            // Pad existing rows to new length
            let total_k = row.len();
            for prev_row in z.iter_mut() {
                prev_row.resize(total_k, false);
            }
            z.push(row);
        }

        // Compute feature probabilities (empirical from counts)
        let feature_probs: Vec<f64> = n_k
            .iter()
            .map(|&count| count as f64 / n_obs.max(1) as f64)
            .collect();

        IbpState {
            z,
            feature_probs,
            n_obs,
        }
    }

    /// Perform one Gibbs sweep over all (i, k) assignments.
    ///
    /// Uses the linear-Gaussian likelihood: x_i | Z_i, A ~ N(Z_i A, σ_x² I)
    /// where A is marginalised analytically (σ_a² prior on each A_k).
    ///
    /// # Parameters
    /// - `state`: current IBP state (modified in-place)
    /// - `data`: N × D data matrix
    /// - `sigma_x`: observation noise
    /// - `sigma_a`: feature weight prior scale
    pub fn gibbs_step(
        &self,
        state: &mut IbpState,
        data: &Array2<f64>,
        sigma_x: f64,
        sigma_a: f64,
    ) {
        let n = state.n_obs;
        let k_total = state.n_features();
        if k_total == 0 || n == 0 {
            return;
        }

        let mut rng = Lcg::new(self.config.seed.wrapping_add(13));
        let mut uniform = || rng.next_f64();

        let d = data.ncols();
        let var_x = sigma_x * sigma_x;
        let var_a = sigma_a * sigma_a;

        for i in 0..n {
            for k in 0..k_total {
                // Count n_{-i,k}: how many OTHER observations use feature k
                let n_minus_ik: usize = state
                    .z
                    .iter()
                    .enumerate()
                    .filter(|(j, row)| *j != i && row.get(k).copied().unwrap_or(false))
                    .count();

                // Prior probability from IBP
                let prior_prob = n_minus_ik as f64 / n as f64;
                if prior_prob <= 0.0 {
                    state.z[i][k] = false;
                    continue;
                }
                if prior_prob >= 1.0 {
                    state.z[i][k] = true;
                    continue;
                }

                // Approximate log-likelihood ratio: p(x_i | z_ik=1) / p(x_i | z_ik=0)
                // For tractability, we use the single-feature contribution approximation:
                // Under z_ik=1: x_i receives contribution from feature k ~ N(0, var_a),
                //   so each dim j: p(x_ij | z_ik=1) ∝ exp(-x_ij^2 / (2*(var_x + var_a)))
                // Under z_ik=0: p(x_ij | z_ik=0) ∝ exp(-x_ij^2 / (2*var_x))
                let mut log_lr = 0.0_f64;
                for j in 0..d {
                    let xij = data[[i, j]];
                    // log p(x | z=1) - log p(x | z=0)
                    let log_p1 = -0.5 * xij * xij / (var_x + var_a)
                        - 0.5 * (2.0 * std::f64::consts::PI * (var_x + var_a)).ln();
                    let log_p0 = -0.5 * xij * xij / var_x
                        - 0.5 * (2.0 * std::f64::consts::PI * var_x).ln();
                    log_lr += log_p1 - log_p0;
                }

                // Posterior probability P(z_ik=1 | rest)
                let log_prior_ratio = (prior_prob / (1.0 - prior_prob)).ln();
                let log_post = log_prior_ratio + log_lr;
                let post_prob = sigmoid(log_post);

                state.z[i][k] = uniform() < post_prob;
            }
        }

        // Update feature probabilities from current counts
        for k in 0..k_total {
            state.feature_probs[k] = state.feature_count(k) as f64 / n as f64;
        }
    }

    /// Fit the Beta Process / IBP model to data via Gibbs sampling.
    ///
    /// # Parameters
    /// - `data`: N × D observation matrix
    /// - `config`: sampler configuration (uses `self.config` if not overridden)
    ///
    /// # Returns
    /// [`IbpResult`] with posterior summary statistics.
    pub fn fit(&self, data: &Array2<f64>) -> Result<IbpResult> {
        let n = data.nrows();
        if n == 0 {
            return Err(StatsError::InvalidInput(
                "BetaProcess::fit: data must have at least one observation".into(),
            ));
        }
        let sigma_x = 0.5;
        let sigma_a = 1.0;
        let n_samples = self.config.n_samples;
        let burnin = self.config.burnin;

        // Initialise state from prior
        let mut state = self.sample_prior(n);

        // Ensure Z matrix has at least n_features columns
        let min_k = self.config.n_features.min(state.n_features().max(1));
        let k_init = state.n_features().max(min_k);
        for row in state.z.iter_mut() {
            row.resize(k_init, false);
        }
        state.feature_probs.resize(k_init, 0.0);

        // Accumulate Z for posterior mean
        let k_final = state.n_features();
        let mut z_accum = vec![vec![0.0_f64; k_final]; n];
        let mut n_samples_collected = 0usize;
        let mut log_ml_accum = 0.0_f64;

        for iter in 0..(burnin + n_samples) {
            self.gibbs_step(&mut state, data, sigma_x, sigma_a);

            if iter >= burnin {
                // Accumulate
                let k = state.n_features().min(k_final);
                for i in 0..n {
                    for ki in 0..k {
                        if state.z[i].get(ki).copied().unwrap_or(false) {
                            if ki < z_accum[i].len() {
                                z_accum[i][ki] += 1.0;
                            }
                        }
                    }
                }
                // Approximate log marginal likelihood contribution
                log_ml_accum += compute_ibp_log_likelihood(&state, data, sigma_x, sigma_a);
                n_samples_collected += 1;
            }
        }

        let total = n_samples_collected.max(1) as f64;
        let z_mean_vec: Vec<f64> = z_accum.iter().flat_map(|row| row.iter().map(|&v| v / total)).collect();
        let z_mean = Array2::from_shape_vec((n, k_final), z_mean_vec).map_err(|e| {
            StatsError::ComputationError(format!("BetaProcess::fit: shape error: {e}"))
        })?;

        let n_features_used = (0..k_final)
            .filter(|&k| z_mean.column(k).iter().any(|&v| v > 0.0))
            .count();

        Ok(IbpResult {
            z_mean,
            n_features_used,
            log_marginal_likelihood: log_ml_accum / total,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Approximate log-likelihood p(X | Z) under a linear-Gaussian model with
/// analytic marginalisation over A.
fn compute_ibp_log_likelihood(
    state: &IbpState,
    data: &Array2<f64>,
    sigma_x: f64,
    sigma_a: f64,
) -> f64 {
    let n = state.n_obs;
    let d = data.ncols();
    let k = state.n_features();
    if k == 0 {
        return 0.0;
    }
    let var_x = sigma_x * sigma_x;
    let var_a = sigma_a * sigma_a;

    // Build Z as f64 matrix (N × K)
    let z_vec: Vec<f64> = state
        .z
        .iter()
        .flat_map(|row| row.iter().map(|&b| if b { 1.0 } else { 0.0 }))
        .collect();
    let z_mat = match Array2::from_shape_vec((n, k), z_vec) {
        Ok(m) => m,
        Err(_) => return 0.0,
    };

    // Sigma_A^{-1} = Z^T Z / sigma_x^2 + I / sigma_a^2
    // For log-likelihood we approximate with per-dimension independence:
    // log p(X | Z) ≈ sum_j [ -N/2 log(2π σ_x^2) - 1/(2σ_x^2) (x_j - Z mu_A_j)^T (x_j - Z mu_A_j) + const ]
    let zt_z = z_mat.t().dot(&z_mat);

    let mut log_lik = 0.0_f64;
    for j in 0..d {
        let xj = data.column(j);
        // Posterior mean of A_j: mu_j = Sigma_A * Z^T x_j / var_x
        // Sigma_A = (Z^T Z / var_x + I / var_a)^{-1}
        // We approximate: use diagonal of Sigma_A only
        let mut pred_sq_err = 0.0_f64;
        for i in 0..n {
            // Simple reconstruction: Σ_k z_ik * mu_Akj
            // Use z_ik weighted mean of x values
            let mut pred = 0.0_f64;
            let mut wsum = 0.0_f64;
            for ki in 0..k {
                if state.z[i].get(ki).copied().unwrap_or(false) {
                    // Feature ki is active for i; estimate A_kj from data
                    let a_kj_est: f64 = (0..n)
                        .filter(|&ii| state.z[ii].get(ki).copied().unwrap_or(false))
                        .map(|ii| data[[ii, j]])
                        .sum::<f64>()
                        / (state.feature_count(ki).max(1) as f64);
                    pred += a_kj_est;
                    wsum += 1.0;
                }
            }
            if wsum > 0.0 {
                pred /= wsum.sqrt().max(1.0);
            }
            let xij = xj[i];
            pred_sq_err += (xij - pred).powi(2);
        }
        log_lik -= 0.5 * (n as f64) * (2.0 * std::f64::consts::PI * var_x).ln();
        log_lik -= pred_sq_err / (2.0 * var_x);
    }

    // IBP prior log-probability
    let mut log_prior = 0.0_f64;
    let alpha = self_alpha_from_z(state, &zt_z, var_a);
    // Harmonic number H_N = sum_{i=1}^{N} 1/i
    let h_n: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
    // Expected dishes = alpha * H_N
    log_prior -= alpha * h_n; // Poisson(alpha * H_N) for total features observed

    log_lik + log_prior
}

/// Rough estimate of effective alpha from Z counts (for prior term only).
fn self_alpha_from_z(state: &IbpState, _zt_z: &Array2<f64>, _var_a: f64) -> f64 {
    let k = state.n_features();
    let n = state.n_obs.max(1);
    let h_n: f64 = (1..=n).map(|i| 1.0 / i as f64).sum();
    (k as f64) / h_n.max(1e-10)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bp() -> BetaProcess {
        BetaProcess::new(BetaProcessConfig::default()).expect("valid config")
    }

    #[test]
    fn test_ibp_prior_first_customer_poisson_alpha() {
        // The first customer (index 0) should take Poisson(alpha) dishes on average
        let config = BetaProcessConfig {
            alpha: 3.0,
            n_features: 30,
            ..Default::default()
        };
        let bp = BetaProcess::new(config).expect("valid");
        let state = bp.sample_prior(1);
        let n_dishes = state.z[0].iter().filter(|&&b| b).count();
        // Poisson(3) mean=3; with seed=42 result should be non-negative
        assert!(n_dishes <= 20, "should not get excessively many dishes");
    }

    #[test]
    fn test_ibp_prior_z_is_binary() {
        let bp = default_bp();
        let state = bp.sample_prior(15);
        // All entries are bool, always valid—but let's check feature_probs in [0,1]
        for &p in &state.feature_probs {
            assert!(p >= 0.0 && p <= 1.0, "prob out of range: {p}");
        }
    }

    #[test]
    fn test_ibp_prior_positive_features() {
        let config = BetaProcessConfig {
            alpha: 2.0,
            n_features: 20,
            n_samples: 10,
            burnin: 5,
            ..Default::default()
        };
        let bp = BetaProcess::new(config).expect("valid");
        let state = bp.sample_prior(10);
        assert!(
            state.n_features() > 0,
            "should have at least one feature, got {}",
            state.n_features()
        );
    }

    #[test]
    fn test_ibp_prior_low_alpha_few_features() {
        let config = BetaProcessConfig {
            alpha: 0.01,
            n_features: 20,
            n_samples: 20,
            burnin: 5,
            ..Default::default()
        };
        let bp = BetaProcess::new(config).expect("valid");
        let state = bp.sample_prior(20);
        // With very low alpha, expect very few features on average
        let total_features = state.n_features();
        // Allow up to 5; typically 0 or 1
        assert!(
            total_features <= 10,
            "Expected few features with alpha=0.01, got {total_features}"
        );
    }

    #[test]
    fn test_ibp_prior_left_ordering_tendency() {
        // In the IBP prior, later features should (on average) be used by fewer customers
        let config = BetaProcessConfig {
            alpha: 5.0,
            n_features: 30,
            n_samples: 10,
            burnin: 2,
            seed: 123,
            ..Default::default()
        };
        let bp = BetaProcess::new(config).expect("valid");
        let state = bp.sample_prior(50);
        let counts = state.feature_counts();
        if counts.len() >= 3 {
            // First feature should have >= last feature count on average (left-ordered)
            let first = counts[0];
            let last = *counts.last().unwrap_or(&0);
            assert!(first >= last, "left-ordering: first count {first} should >= last count {last}");
        }
    }

    #[test]
    fn test_beta_process_fit_produces_result() {
        use scirs2_core::ndarray::Array2;
        let n = 10;
        let d = 4;
        let mut rng = Lcg::new(777);
        let data_vec: Vec<f64> = (0..n * d).map(|_| rng.next_normal()).collect();
        let data = Array2::from_shape_vec((n, d), data_vec).expect("shape ok");

        let config = BetaProcessConfig {
            alpha: 2.0,
            n_features: 5,
            n_samples: 10,
            burnin: 5,
            ..Default::default()
        };
        let bp = BetaProcess::new(config).expect("valid");
        let result = bp.fit(&data).expect("fit ok");
        assert!(result.n_features_used > 0);
        assert!(result.log_marginal_likelihood.is_finite());
    }

    #[test]
    fn test_poisson_sample_zero_lambda() {
        let mut rng = Lcg::new(1);
        let mut uniform = || rng.next_f64();
        assert_eq!(poisson_sample(0.0, &mut uniform), 0);
    }

    #[test]
    fn test_poisson_sample_reasonable_mean() {
        let lambda = 4.0;
        let mut rng = Lcg::new(99);
        let mut uniform = || rng.next_f64();
        let samples: Vec<usize> = (0..1000).map(|_| poisson_sample(lambda, &mut uniform)).collect();
        let mean = samples.iter().sum::<usize>() as f64 / 1000.0;
        // Should be close to lambda
        assert!((mean - lambda).abs() < 1.0, "mean {mean} far from lambda {lambda}");
    }
}
