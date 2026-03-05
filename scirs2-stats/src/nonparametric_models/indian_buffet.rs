//! Indian Buffet Process for sparse latent feature models.
//!
//! The IBP generates an infinite binary matrix Z ∈ {0,1}^{N×∞} where
//! Z_{ik} = 1 indicates that observation i has feature k.
//!
//! The IBP metaphor: customers (observations) enter an Indian buffet one by
//! one.  The first customer tries Poisson(α) dishes.  Customer n:
//! - Tries each previously-sampled dish k with probability m_k / n
//!   (where m_k is the number of previous customers who tried dish k)
//! - Tries Poisson(α/n) new dishes
//!
//! The resulting Z matrix has:
//! - Expected total features: α * H_N  (H_N = n-th harmonic number)
//! - Power-law distribution on feature popularities
//!
//! IBP-Linear Gaussian model: X = Z A + ε, ε ~ N(0, σ²_n I)
//!                            A_{kd} ~ N(0, σ²_A)

use crate::error::{StatsError, StatsResult as Result};
use scirs2_core::random::{rngs::StdRng, Distribution, Normal, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// IBP sampler
// ---------------------------------------------------------------------------

/// Indian Buffet Process prior sampler.
///
/// Generates binary feature allocation matrices from the IBP(α) prior.
#[derive(Debug, Clone)]
pub struct IBPSampler {
    /// IBP strength parameter α > 0.  E[num dishes tried by customer 1] = α.
    pub alpha: f64,
    /// Feature assignments Z[i][k] for customer i, feature k.
    pub assignments: Vec<Vec<bool>>,
    /// How many customers have each feature: m_k = Σ_i Z_{ik}.
    pub feature_counts: Vec<usize>,
    /// Number of customers (observations) added so far.
    pub n_customers: usize,
    /// Total number of unique features generated.
    pub n_features: usize,
}

impl IBPSampler {
    /// Construct a new `IBPSampler` with strength parameter `alpha`.
    ///
    /// # Errors
    /// Returns an error when `alpha <= 0`.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "IBP alpha must be > 0, got {alpha}"
            )));
        }
        Ok(Self {
            alpha,
            assignments: Vec::new(),
            feature_counts: Vec::new(),
            n_customers: 0,
            n_features: 0,
        })
    }

    /// Add customer n+1 to the buffet (0-indexed: customer index `n_customers`).
    ///
    /// Algorithm:
    /// 1. For each existing feature k: take it with prob `feature_counts[k] / n`
    /// 2. Sample `Poisson(α/n)` new features and take all of them
    ///
    /// Returns the feature vector for the new customer.
    pub fn add_customer(&mut self, rng: &mut StdRng) -> Vec<bool> {
        let n = self.n_customers + 1; // 1-indexed customer number
        let n_f = n as f64;

        let mut row = vec![false; self.n_features];

        // Step 1: existing features
        for k in 0..self.n_features {
            let prob = self.feature_counts[k] as f64 / n_f;
            let u = sample_uniform_01(rng);
            if u < prob {
                row[k] = true;
                self.feature_counts[k] += 1;
            }
        }

        // Step 2: new features ~ Poisson(α/n)
        let rate = self.alpha / n_f;
        let new_features = sample_poisson(rng, rate);
        for _ in 0..new_features {
            row.push(true);
            self.feature_counts.push(1);
            self.n_features += 1;
        }

        // Pad previous customers' assignments to include the new features
        for prev_row in &mut self.assignments {
            prev_row.resize(self.n_features, false);
        }

        self.assignments.push(row.clone());
        self.n_customers += 1;
        row
    }

    /// Add `n` customers at once.
    pub fn add_n_customers(&mut self, n: usize, rng: &mut StdRng) -> Vec<Vec<bool>> {
        (0..n).map(|_| self.add_customer(rng)).collect()
    }

    /// The harmonic number H_n = Σ_{k=1}^{n} 1/k.
    pub fn harmonic(n: usize) -> f64 {
        (1..=n).map(|k| 1.0 / k as f64).sum()
    }

    /// Expected total number of features for `n` customers: `α * H_n`.
    pub fn expected_n_features(&self, n: usize) -> f64 {
        self.alpha * Self::harmonic(n)
    }

    /// Total number of customers.
    pub fn num_customers(&self) -> usize {
        self.n_customers
    }

    /// Total number of unique features generated.
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    /// Reference to the binary feature matrix.
    pub fn feature_matrix(&self) -> &[Vec<bool>] {
        &self.assignments
    }

    /// Feature density: fraction of (customer, feature) pairs that are 1.
    pub fn feature_density(&self) -> f64 {
        if self.n_customers == 0 || self.n_features == 0 {
            return 0.0;
        }
        let total_ones: usize = self.feature_counts.iter().sum();
        total_ones as f64 / (self.n_customers * self.n_features) as f64
    }

    /// Remove feature k from all customers (for Gibbs updates).
    ///
    /// # Errors
    /// Returns an error when `k >= n_features`.
    pub fn remove_feature(&mut self, k: usize) -> Result<()> {
        if k >= self.n_features {
            return Err(StatsError::InvalidArgument(format!(
                "feature {k} >= n_features {}",
                self.n_features
            )));
        }
        for row in &mut self.assignments {
            if k < row.len() {
                row.remove(k);
            }
        }
        self.feature_counts.remove(k);
        self.n_features -= 1;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// IBP-LGM: Linear Gaussian Model
// ---------------------------------------------------------------------------

/// Indian Buffet Process coupled with a Linear Gaussian observation model.
///
/// ```text
/// Z    ~ IBP(α)
/// A_k  ~ N(0, σ²_A * I_d)    (feature factors)
/// X_i  ~ N(Z_i A, σ²_n * I_d)  (observations)
/// ```
///
/// Inference via Gibbs sampling over Z and A.
#[derive(Debug, Clone)]
pub struct IndianBuffetProcess {
    /// IBP concentration parameter.
    pub alpha: f64,
    /// Feature factor variance.
    pub sigma2_a: f64,
    /// Observation noise variance.
    pub sigma2_n: f64,
    /// Feature matrix Z (N × K).
    pub feature_matrix: Vec<Vec<bool>>,
    /// Factor matrix A (K × D), row-major.
    pub factors: Vec<Vec<f64>>,
    /// Number of observations.
    pub n_obs: usize,
    /// Data dimensionality.
    pub dim: usize,
    /// Number of active features.
    pub n_features: usize,
    /// Log-likelihood at last Gibbs iteration.
    pub log_likelihood: f64,
    /// Whether sampling converged.
    pub converged: bool,
}

impl IndianBuffetProcess {
    /// Construct a new IBP linear Gaussian model.
    ///
    /// # Errors
    /// Returns an error on invalid hyperparameters.
    pub fn new(alpha: f64, sigma2_a: f64, sigma2_n: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "alpha must be > 0, got {alpha}"
            )));
        }
        if sigma2_a <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "sigma2_a must be > 0, got {sigma2_a}"
            )));
        }
        if sigma2_n <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "sigma2_n must be > 0, got {sigma2_n}"
            )));
        }
        Ok(Self {
            alpha,
            sigma2_a,
            sigma2_n,
            feature_matrix: Vec::new(),
            factors: Vec::new(),
            n_obs: 0,
            dim: 0,
            n_features: 0,
            log_likelihood: f64::NEG_INFINITY,
            converged: false,
        })
    }

    /// Fit the IBP-LGM via Gibbs sampling.
    ///
    /// # Parameters
    /// - `data`: Observation matrix (N × D).
    /// - `n_iter`: Number of Gibbs sweeps.
    /// - `seed`: Random seed.
    ///
    /// # Errors
    /// Returns an error on empty data or dimension mismatches.
    pub fn fit_gibbs(&mut self, data: &[Vec<f64>], n_iter: usize, seed: u64) -> Result<()> {
        let n = data.len();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "data must be non-empty".into(),
            ));
        }
        let d = data[0].len();
        if d == 0 {
            return Err(StatsError::InvalidArgument(
                "data dimensionality must be >= 1".into(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != d {
                return Err(StatsError::DimensionMismatch(format!(
                    "data[{i}] has {} cols, expected {d}",
                    row.len()
                )));
            }
        }

        self.n_obs = n;
        self.dim = d;

        let mut rng = StdRng::seed_from_u64(seed);

        // Initialize via the IBP prior
        let mut sampler = IBPSampler::new(self.alpha)?;
        sampler.add_n_customers(n, &mut rng);
        self.feature_matrix = sampler.assignments;
        self.n_features = sampler.n_features;

        // Ensure rectangular shape
        for row in &mut self.feature_matrix {
            row.resize(self.n_features, false);
        }

        // Initialize factor matrix A (K × D) from prior N(0, σ²_A)
        let normal_a = Normal::new(0.0, self.sigma2_a.sqrt()).map_err(|e| {
            StatsError::ComputationError(format!("Normal init error: {e}"))
        })?;
        self.factors = (0..self.n_features)
            .map(|_| (0..d).map(|_| normal_a.sample(&mut rng)).collect())
            .collect();

        let mut prev_ll = f64::NEG_INFINITY;
        let tol = 1e-4;

        for iter in 0..n_iter {
            // ---- Update Z (feature assignments) ----
            self.gibbs_update_z(data, &mut rng)?;

            // ---- Update A (factor values) ----
            self.gibbs_update_a(data, &mut rng)?;

            // Compute log-likelihood
            let ll = self.compute_log_likelihood(data);
            if iter > 5 && (ll - prev_ll).abs() < tol {
                self.converged = true;
                self.log_likelihood = ll;
                break;
            }
            prev_ll = ll;
            self.log_likelihood = ll;
        }

        Ok(())
    }

    /// Gibbs update for binary feature assignments Z_{ik}.
    fn gibbs_update_z(&mut self, data: &[Vec<f64>], rng: &mut StdRng) -> Result<()> {
        let n = self.n_obs;
        let d = self.dim;

        for i in 0..n {
            // Remove features that become empty after removal
            let mut k = 0;
            while k < self.n_features {
                // Check if all other customers have this feature
                let m_ik: usize = (0..n)
                    .filter(|&j| j != i && self.feature_matrix[j].get(k).copied().unwrap_or(false))
                    .count();

                if m_ik == 0 {
                    // No other customer has this feature: use IBP prior + likelihood
                    let log_prior_off = (n as f64 - 1.0 - 0.0).ln().max(f64::NEG_INFINITY)
                        - (n as f64).ln();
                    // P(z_ik=1 | m_{-i,k}) = m_ik / n = 0: no prior mass for taking it
                    // Remove this feature if current customer has it
                    if self.feature_matrix[i].get(k).copied().unwrap_or(false) {
                        // Drop this singleton feature
                        for row in &mut self.feature_matrix {
                            if k < row.len() {
                                row.remove(k);
                            }
                        }
                        if k < self.factors.len() {
                            self.factors.remove(k);
                        }
                        self.n_features -= 1;
                        continue; // don't advance k
                    }
                    k += 1;
                    continue;
                }

                // Compute likelihood ratio for z_ik = 1 vs z_ik = 0
                let z_on = {
                    let old = self.feature_matrix[i][k];
                    self.feature_matrix[i][k] = true;
                    let ll = self.obs_log_lik(data, i);
                    self.feature_matrix[i][k] = old;
                    ll
                };
                let z_off = {
                    let old = self.feature_matrix[i][k];
                    self.feature_matrix[i][k] = false;
                    let ll = self.obs_log_lik(data, i);
                    self.feature_matrix[i][k] = old;
                    ll
                };

                let log_prior_on = (m_ik as f64).ln() - (n as f64).ln();
                let log_prior_off_v = ((n - m_ik) as f64).ln() - (n as f64).ln();

                let log_p_on = log_prior_on + z_on;
                let log_p_off = log_prior_off_v + z_off;

                let max_lp = log_p_on.max(log_p_off);
                let p_on = (log_p_on - max_lp).exp();
                let p_off = (log_p_off - max_lp).exp();
                let prob_on = p_on / (p_on + p_off);

                let u = sample_uniform_01(rng);
                self.feature_matrix[i][k] = u < prob_on;
                k += 1;
            }

            // Sample new features ~ Poisson(α/n)
            let rate = self.alpha / n as f64;
            let new_k = sample_poisson(rng, rate);
            for _ in 0..new_k {
                // Add a new feature to all customers (false for others)
                for row in &mut self.feature_matrix {
                    row.push(false);
                }
                self.feature_matrix[i].push(true);
                // Sample factor from prior
                let normal_a = Normal::new(0.0, self.sigma2_a.sqrt())
                    .unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("Normal::new(0.0, 1.0) is always valid"));
                let new_factor: Vec<f64> = (0..d).map(|_| normal_a.sample(rng)).collect();
                self.factors.push(new_factor);
                self.n_features += 1;
            }
        }
        Ok(())
    }

    /// Gibbs update for factor matrix A (conjugate Normal posterior).
    fn gibbs_update_a(&mut self, data: &[Vec<f64>], rng: &mut StdRng) -> Result<()> {
        let n = self.n_obs;
        let d = self.dim;

        for k in 0..self.n_features {
            // Observations that use feature k
            let users: Vec<usize> = (0..n)
                .filter(|&i| self.feature_matrix[i].get(k).copied().unwrap_or(false))
                .collect();

            if users.is_empty() {
                // Sample from prior
                let normal_a = Normal::new(0.0, self.sigma2_a.sqrt())
                    .unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("Normal::new(0.0, 1.0) is always valid"));
                for j in 0..d {
                    self.factors[k][j] = normal_a.sample(rng);
                }
                continue;
            }

            // For each dimension j, compute posterior N(mu_post, sigma2_post)
            // Likelihood: x_{ij} = Σ_{k'≠k} z_{ik'} a_{k'j} + z_{ik} a_{kj} + ε
            // Residual r_{ij} = x_{ij} - Σ_{k'≠k} z_{ik'} a_{k'j}
            // Posterior: sigma2_post = 1 / (1/σ²_A + n_k/σ²_n)
            //            mu_post = sigma2_post * Σ_i r_{ij} / σ²_n

            let n_k = users.len() as f64;
            let sigma2_post = 1.0 / (1.0 / self.sigma2_a + n_k / self.sigma2_n);
            let std_post = sigma2_post.sqrt();

            for j in 0..d {
                // Compute residual sum
                let resid_sum: f64 = users.iter().map(|&i| {
                    let mut r = data[i][j];
                    for k2 in 0..self.n_features {
                        if k2 != k && self.feature_matrix[i].get(k2).copied().unwrap_or(false) {
                            r -= self.factors[k2].get(j).copied().unwrap_or(0.0);
                        }
                    }
                    r
                }).sum();

                let mu_post = sigma2_post * resid_sum / self.sigma2_n;
                let normal = Normal::new(mu_post, std_post).map_err(|e| {
                    StatsError::ComputationError(format!("Normal init error: {e}"))
                })?;
                self.factors[k][j] = normal.sample(rng);
            }
        }
        Ok(())
    }

    /// Log-likelihood contribution of observation i.
    fn obs_log_lik(&self, data: &[Vec<f64>], i: usize) -> f64 {
        let d = self.dim;
        let xi = &data[i];
        let std_n = self.sigma2_n.sqrt();
        (0..d)
            .map(|j| {
                let pred: f64 = (0..self.n_features)
                    .filter(|&k| self.feature_matrix[i].get(k).copied().unwrap_or(false))
                    .map(|k| self.factors[k].get(j).copied().unwrap_or(0.0))
                    .sum();
                let z = (xi[j] - pred) / std_n;
                -0.5 * z * z - std_n.ln() - 0.5 * (2.0 * PI).ln()
            })
            .sum()
    }

    /// Total log-likelihood of all observations.
    pub fn compute_log_likelihood(&self, data: &[Vec<f64>]) -> f64 {
        (0..self.n_obs).map(|i| self.obs_log_lik(data, i)).sum()
    }

    /// Reconstruct data as Z A.
    pub fn reconstruct(&self) -> Vec<Vec<f64>> {
        (0..self.n_obs)
            .map(|i| {
                (0..self.dim)
                    .map(|j| {
                        (0..self.n_features)
                            .filter(|&k| {
                                self.feature_matrix[i].get(k).copied().unwrap_or(false)
                            })
                            .map(|k| self.factors[k].get(j).copied().unwrap_or(0.0))
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sample_uniform_01(rng: &mut StdRng) -> f64 {
    use scirs2_core::random::Uniform;
    Uniform::new(0.0, 1.0)
        .map(|d| d.sample(rng))
        .unwrap_or(0.5)
}

/// Approximate Poisson sample using Knuth's algorithm for rate < 30,
/// and normal approximation for higher rates.
fn sample_poisson(rng: &mut StdRng, rate: f64) -> usize {
    if rate <= 0.0 {
        return 0;
    }
    if rate > 30.0 {
        // Normal approximation
        let normal = Normal::new(rate, rate.sqrt()).unwrap_or_else(|_| Normal::new(0.0, 1.0).expect("Normal::new(0.0, 1.0) is always valid"));
        let s = normal.sample(rng);
        return s.max(0.0).round() as usize;
    }
    // Knuth's algorithm
    let threshold = (-rate).exp();
    let mut k = 0usize;
    let mut p = 1.0_f64;
    loop {
        p *= sample_uniform_01(rng);
        if p <= threshold {
            break;
        }
        k += 1;
    }
    k
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibp_sampler_construction() {
        assert!(IBPSampler::new(1.0).is_ok());
        assert!(IBPSampler::new(0.0).is_err());
        assert!(IBPSampler::new(-1.0).is_err());
    }

    #[test]
    fn test_ibp_adds_customers() {
        let mut sampler = IBPSampler::new(2.0).unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        sampler.add_n_customers(10, &mut rng);
        assert_eq!(sampler.num_customers(), 10);
        assert!(sampler.num_features() >= 1);
        assert_eq!(sampler.assignments.len(), 10);
    }

    #[test]
    fn test_ibp_feature_matrix_shape() {
        let mut sampler = IBPSampler::new(1.0).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        sampler.add_n_customers(5, &mut rng);
        let k = sampler.num_features();
        assert!(sampler.feature_matrix().iter().all(|row| row.len() == k));
    }

    #[test]
    fn test_ibp_feature_counts_consistent() {
        let mut sampler = IBPSampler::new(2.0).unwrap();
        let mut rng = StdRng::seed_from_u64(99);
        sampler.add_n_customers(20, &mut rng);
        // feature_counts[k] == number of true entries in column k
        for k in 0..sampler.num_features() {
            let count = sampler
                .assignments
                .iter()
                .filter(|row| row.get(k).copied().unwrap_or(false))
                .count();
            assert_eq!(sampler.feature_counts[k], count, "feature {k}");
        }
    }

    #[test]
    fn test_ibp_expected_features() {
        let alpha = 2.0;
        let sampler = IBPSampler::new(alpha).unwrap();
        let expected = sampler.expected_n_features(10);
        // H_10 ≈ 2.928
        assert!((expected - 2.0 * 2.928).abs() < 0.5);
    }

    #[test]
    fn test_ibp_harmonic() {
        // H_1 = 1, H_2 = 1.5, H_3 = 11/6 ≈ 1.833
        assert!((IBPSampler::harmonic(1) - 1.0).abs() < 1e-10);
        assert!((IBPSampler::harmonic(2) - 1.5).abs() < 1e-10);
        assert!((IBPSampler::harmonic(3) - 11.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ibp_lgm_fit() {
        // Synthetic 2-feature data
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                if i < 10 {
                    vec![1.0, 0.0]
                } else {
                    vec![0.0, 1.0]
                }
            })
            .collect();
        let mut ibp = IndianBuffetProcess::new(1.0, 1.0, 0.1).unwrap();
        ibp.fit_gibbs(&data, 30, 42).unwrap();

        assert_eq!(ibp.n_obs, 20);
        assert_eq!(ibp.dim, 2);
        assert!(ibp.log_likelihood.is_finite());
        assert!(ibp.n_features >= 1);
    }

    #[test]
    fn test_ibp_lgm_invalid() {
        assert!(IndianBuffetProcess::new(0.0, 1.0, 0.1).is_err());
        assert!(IndianBuffetProcess::new(1.0, 0.0, 0.1).is_err());
        assert!(IndianBuffetProcess::new(1.0, 1.0, 0.0).is_err());

        let mut ibp = IndianBuffetProcess::new(1.0, 1.0, 0.1).unwrap();
        assert!(ibp.fit_gibbs(&[], 10, 0).is_err());
    }

    #[test]
    fn test_ibp_reconstruct_shape() {
        let data: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let mut ibp = IndianBuffetProcess::new(1.0, 1.0, 0.5).unwrap();
        ibp.fit_gibbs(&data, 20, 1).unwrap();
        let recon = ibp.reconstruct();
        assert_eq!(recon.len(), 3);
        assert!(recon.iter().all(|row| row.len() == 2));
    }
}
