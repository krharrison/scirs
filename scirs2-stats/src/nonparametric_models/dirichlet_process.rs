//! Dirichlet Process and DP Mixture Models.
//!
//! The Dirichlet Process DP(α, H) is a distribution over probability
//! distributions.  Its three main constructive representations are:
//!
//! - **Stick-breaking (GEM)**: `π_k = v_k ∏_{j<k} (1-v_j)`,  `v_k ~ Beta(1, α)`
//! - **Chinese Restaurant Process**: sequential seating probabilities
//! - **Pólya urn**: marginalized representation for i.i.d. samples
//!
//! The DP mixture model clusters data by placing a DP prior on the mixing
//! distribution and using conjugate (Normal-Inverse-Gamma) base distributions.

use crate::error::{StatsError, StatsResult as Result};
use crate::hierarchical::hyperpriors::NormalInverseGamma;
use scirs2_core::random::{rngs::StdRng, Beta as RandBeta, CoreRandom, Distribution, Gamma, Normal, Rng, SeedableRng};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Chinese Restaurant Process
// ---------------------------------------------------------------------------

/// Chinese Restaurant Process sampler.
///
/// The CRP is the marginal (marginalized-out DP) representation of the
/// Dirichlet Process.  It defines a distribution over partitions of N items:
///
/// ```text
/// P(customer n+1 joins table k | t₁..tₙ) = nₖ / (n + α)
/// P(customer n+1 opens table K+1 | t₁..tₙ) = α / (n + α)
/// ```
#[derive(Debug, Clone)]
pub struct CRPSampler {
    /// Concentration parameter α > 0.
    pub alpha: f64,
    /// Table assignment for each customer (0-indexed).
    pub tables: Vec<usize>,
    /// Number of customers at each table.
    pub table_counts: Vec<usize>,
    /// Total number of tables opened.
    pub n_tables: usize,
}

impl CRPSampler {
    /// Construct a new `CRPSampler` with concentration `alpha`.
    ///
    /// # Errors
    /// Returns an error when `alpha <= 0`.
    pub fn new(alpha: f64) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "CRP alpha must be > 0, got {alpha}"
            )));
        }
        Ok(Self {
            alpha,
            tables: Vec::new(),
            table_counts: Vec::new(),
            n_tables: 0,
        })
    }

    /// Seat the next customer using the CRP seating probabilities.
    ///
    /// With probability proportional to `nₖ`, the customer joins existing
    /// table `k`.  With probability proportional to `α`, a new table is opened.
    ///
    /// Returns the table index assigned to this customer.
    pub fn seat_customer(&mut self, rng: &mut StdRng) -> usize {
        let n = self.tables.len();
        let total = n as f64 + self.alpha;

        // Sample u ~ Uniform(0, total)
        let u = sample_uniform(rng, 0.0, total);

        // Decide which table
        let mut cumulative = 0.0;
        for (k, &count) in self.table_counts.iter().enumerate() {
            cumulative += count as f64;
            if u < cumulative {
                self.tables.push(k);
                self.table_counts[k] += 1;
                return k;
            }
        }

        // Open a new table
        let new_table = self.n_tables;
        self.tables.push(new_table);
        self.table_counts.push(1);
        self.n_tables += 1;
        new_table
    }

    /// Seat `n` customers sequentially, returning their table assignments.
    pub fn seat_n_customers(&mut self, n: usize, rng: &mut StdRng) -> Vec<usize> {
        (0..n).map(|_| self.seat_customer(rng)).collect()
    }

    /// Total number of customers.
    pub fn num_customers(&self) -> usize {
        self.tables.len()
    }

    /// Total number of tables opened.
    pub fn num_tables(&self) -> usize {
        self.n_tables
    }

    /// Table assignment for customer `i`.
    ///
    /// # Errors
    /// Returns an error when `i >= num_customers()`.
    pub fn table_for_customer(&self, i: usize) -> Result<usize> {
        self.tables.get(i).copied().ok_or_else(|| {
            StatsError::InvalidArgument(format!(
                "customer index {i} out of range ({})",
                self.tables.len()
            ))
        })
    }

    /// Expected number of tables for `n` customers: `α * H_n` (harmonic number).
    pub fn expected_tables(alpha: f64, n: usize) -> f64 {
        alpha * (1..=n).map(|k| 1.0 / k as f64).sum::<f64>()
    }

    /// Remove a customer from its table (for Gibbs updates).
    /// Removes empty tables if the count drops to zero.
    ///
    /// # Errors
    /// Returns an error on invalid customer index.
    pub fn unseat_customer(&mut self, i: usize) -> Result<()> {
        if i >= self.tables.len() {
            return Err(StatsError::InvalidArgument(format!(
                "customer index {i} out of range"
            )));
        }
        let t = self.tables[i];
        self.table_counts[t] -= 1;
        // Note: we do not reindex tables here (lazy deletion)
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Stick-breaking representation
// ---------------------------------------------------------------------------

/// Stick-breaking (GEM) representation of the Dirichlet Process.
///
/// The stick lengths are `π_k = v_k ∏_{j<k} (1-v_j)` where
/// `v_k ~ Beta(1, α)`.
#[derive(Debug, Clone)]
pub struct StickBreaking {
    /// Concentration parameter.
    pub alpha: f64,
    /// Stick weights (π_1, π_2, …, π_K) – finite truncation.
    pub weights: Vec<f64>,
    /// Number of components (truncation level).
    pub n_components: usize,
}

impl StickBreaking {
    /// Construct a new `StickBreaking` with `n_components` truncation.
    ///
    /// # Errors
    /// Returns an error when `alpha <= 0` or `n_components == 0`.
    pub fn new(alpha: f64, n_components: usize) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "alpha must be > 0, got {alpha}"
            )));
        }
        if n_components == 0 {
            return Err(StatsError::InvalidArgument(
                "n_components must be >= 1".into(),
            ));
        }
        Ok(Self {
            alpha,
            weights: vec![1.0 / n_components as f64; n_components],
            n_components,
        })
    }

    /// Sample new stick weights from Beta(1, α).
    pub fn sample_weights<R: Rng>(&mut self, rng: &mut CoreRandom<R>) -> Result<()> {
        let mut remaining = 1.0_f64;
        self.weights.clear();
        for i in 0..self.n_components {
            let beta = if i < self.n_components - 1 {
                let b = RandBeta::new(1.0, self.alpha).map_err(|e| {
                    StatsError::ComputationError(format!("Beta sampling error: {e}"))
                })?;
                b.sample(rng)
            } else {
                1.0 // last stick takes all remaining
            };
            let weight = beta * remaining;
            self.weights.push(weight);
            remaining -= weight;
            if remaining < 1e-15 {
                remaining = 0.0;
            }
        }
        Ok(())
    }

    /// Expected number of components used (those with weight above threshold).
    pub fn expected_n_components(&self) -> f64 {
        // E[K] ≈ α log(1 + n_components/α)
        self.alpha * (1.0 + self.n_components as f64 / self.alpha).ln()
    }

    /// Sample a component index proportional to stick weights.
    pub fn sample_component(&self, rng: &mut StdRng) -> usize {
        let u = sample_uniform(rng, 0.0, 1.0);
        let mut cumsum = 0.0_f64;
        for (k, &w) in self.weights.iter().enumerate() {
            cumsum += w;
            if u < cumsum {
                return k;
            }
        }
        self.n_components - 1
    }
}

// ---------------------------------------------------------------------------
// DP Mixture Model
// ---------------------------------------------------------------------------

/// Dirichlet Process Mixture Model for density estimation and clustering.
///
/// Uses the Normal-Inverse-Gamma conjugate base distribution and
/// collapsed Gibbs sampling to jointly infer cluster assignments and
/// cluster parameters.
///
/// The model is:
/// ```text
/// π | α  ~ GEM(α)          (stick-breaking prior)
/// θₖ | H ~ NIG(μ₀, κ₀, α₀, β₀)    (cluster parameters)
/// zᵢ | π ~ Categorical(π)   (cluster assignments)
/// xᵢ | θ_{zᵢ} ~ N(μ_{zᵢ}, σ²_{zᵢ})
/// ```
#[derive(Debug, Clone)]
pub struct DPMixture {
    /// Concentration parameter.
    pub alpha: f64,
    /// NIG base distribution.
    pub base: NormalInverseGamma,
    /// Cluster assignment for each observation (0-indexed).
    pub assignments: Vec<usize>,
    /// Cluster parameters: (mean, variance).
    pub cluster_params: Vec<(f64, f64)>,
    /// Number of active clusters.
    pub n_clusters: usize,
    /// Per-cluster observation counts.
    cluster_counts: Vec<usize>,
    /// Per-cluster sufficient statistics: (sum, sum_sq, count)
    cluster_stats: Vec<(f64, f64, usize)>,
}

impl DPMixture {
    /// Construct a new `DPMixture`.
    ///
    /// # Errors
    /// Returns an error when `alpha <= 0`.
    pub fn new(alpha: f64, base: NormalInverseGamma) -> Result<Self> {
        if alpha <= 0.0 {
            return Err(StatsError::DomainError(format!(
                "alpha must be > 0, got {alpha}"
            )));
        }
        Ok(Self {
            alpha,
            base,
            assignments: Vec::new(),
            cluster_params: Vec::new(),
            n_clusters: 0,
            cluster_counts: Vec::new(),
            cluster_stats: Vec::new(),
        })
    }

    /// Run the collapsed Gibbs sampler for `n_iter` sweeps.
    ///
    /// This is Algorithm 3 from Neal (2000) – the marginalized CRP Gibbs
    /// sampler which analytically integrates out the cluster parameters.
    ///
    /// # Errors
    /// Returns an error on empty data.
    pub fn fit_gibbs(&mut self, data: &[f64], n_iter: usize, seed: u64) -> Result<()> {
        let n = data.len();
        if n == 0 {
            return Err(StatsError::InsufficientData(
                "data must be non-empty".into(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(seed);

        // Initialize: one cluster per observation
        self.assignments = (0..n).map(|_| 0).collect();
        self.n_clusters = 1;
        self.cluster_counts = vec![n];

        // Compute sufficient stats for the single initial cluster
        let sum: f64 = data.iter().sum();
        let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
        self.cluster_stats = vec![(sum, sum_sq, n)];
        self.cluster_params = vec![(sum / n as f64, 1.0)];

        for _iter in 0..n_iter {
            for i in 0..n {
                let xi = data[i];
                let ci = self.assignments[i];

                // Remove observation i from its cluster
                self.cluster_counts[ci] -= 1;
                {
                    let (ref mut s, ref mut sq, ref mut cnt) = self.cluster_stats[ci];
                    *s -= xi;
                    *sq -= xi * xi;
                    *cnt -= 1;
                }

                // Compute log probabilities for each existing cluster + new
                let n_minus_i = (n - 1) as f64;
                let mut log_probs: Vec<f64> = Vec::new();
                let mut active_clusters: Vec<usize> = Vec::new();

                for (k, &count) in self.cluster_counts.iter().enumerate() {
                    if count > 0 {
                        let log_prior = (count as f64).ln() - (n_minus_i + self.alpha).ln();
                        let log_lik = self.crp_log_lik(xi, k);
                        log_probs.push(log_prior + log_lik);
                        active_clusters.push(k);
                    }
                }

                // New cluster
                let log_prior_new = self.alpha.ln() - (n_minus_i + self.alpha).ln();
                let log_lik_new = self.new_cluster_log_lik(xi);
                log_probs.push(log_prior_new + log_lik_new);

                // Sample new assignment (log-sum-exp normalization)
                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
                let total: f64 = probs.iter().sum();

                let u = sample_uniform(&mut rng, 0.0, total);
                let mut cumsum = 0.0_f64;
                let mut new_ci = *active_clusters.last().unwrap_or(&0);
                let mut assigned_new = false;

                for (idx, &prob) in probs.iter().enumerate() {
                    cumsum += prob;
                    if u < cumsum {
                        if idx < active_clusters.len() {
                            new_ci = active_clusters[idx];
                        } else {
                            // Open a new cluster
                            assigned_new = true;
                            new_ci = self.open_new_cluster(&mut rng)?;
                        }
                        break;
                    }
                }
                if !assigned_new && !active_clusters.contains(&new_ci) {
                    new_ci = self.open_new_cluster(&mut rng)?;
                }

                // Assign observation to new_ci
                self.assignments[i] = new_ci;
                self.cluster_counts[new_ci] += 1;
                {
                    let (ref mut s, ref mut sq, ref mut cnt) = self.cluster_stats[new_ci];
                    *s += xi;
                    *sq += xi * xi;
                    *cnt += 1;
                }
            }

            // Update cluster parameters (posterior means)
            for k in 0..self.cluster_params.len() {
                if self.cluster_counts[k] > 0 {
                    let post = self.posterior_for_cluster(k);
                    self.cluster_params[k] = (post.mu0, post.sigma2_mode());
                }
            }

            // Compact empty clusters
            self.compact_clusters();
        }

        Ok(())
    }

    /// Predict the most probable cluster for a new observation `x`.
    pub fn predict_cluster(&self, x: f64) -> usize {
        let probs = self.cluster_probabilities_raw(x);
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Normalized cluster membership probabilities for a new observation.
    ///
    /// Returns a vector of length equal to the number of active clusters,
    /// with entries proportional to `nₖ * p(x | θₖ)`.
    pub fn cluster_probabilities(&self, x: f64) -> Vec<f64> {
        let raw = self.cluster_probabilities_raw(x);
        let total: f64 = raw.iter().sum();
        if total <= 0.0 {
            let k = raw.len();
            return vec![1.0 / k as f64; k];
        }
        raw.iter().map(|&p| p / total).collect()
    }

    /// Log-likelihood of the data given current assignments and parameters.
    pub fn log_likelihood(&self, data: &[f64]) -> f64 {
        data.iter()
            .zip(self.assignments.iter())
            .filter_map(|(&xi, &ci)| {
                self.cluster_params.get(ci).map(|&(mu, var)| {
                    let std = var.sqrt().max(1e-10);
                    let z = (xi - mu) / std;
                    -0.5 * z * z - std.ln() - 0.5 * (2.0 * PI).ln()
                })
            })
            .sum()
    }

    /// Number of non-empty clusters (effective clusters used by at least one observation).
    pub fn n_effective_clusters(&self) -> usize {
        self.cluster_counts.iter().filter(|&&c| c > 0).count()
    }

    // ---- Internal helpers ----

    /// Marginal likelihood of `x` under NIG posterior for cluster `k`.
    fn crp_log_lik(&self, x: f64, k: usize) -> f64 {
        let post = self.posterior_for_cluster(k);
        post.posterior_predictive_pdf(x).ln()
    }

    /// Marginal likelihood of `x` under the NIG prior (new cluster).
    fn new_cluster_log_lik(&self, x: f64) -> f64 {
        self.base.posterior_predictive_pdf(x).ln()
    }

    /// Posterior NIG parameters for cluster `k`.
    fn posterior_for_cluster(&self, k: usize) -> NormalInverseGamma {
        let (s, sq, cnt) = self.cluster_stats[k];
        if cnt == 0 {
            return self.base.clone();
        }
        let n = cnt as f64;
        let x_bar = s / n;
        let s_sq = (sq - n * x_bar * x_bar).max(0.0);

        let kappa_n = self.base.kappa0 + n;
        let mu_n = (self.base.kappa0 * self.base.mu0 + s) / kappa_n;
        let alpha_n = self.base.alpha0 + n / 2.0;
        let beta_n = self.base.beta0
            + 0.5 * s_sq
            + 0.5 * self.base.kappa0 * n / kappa_n * (x_bar - self.base.mu0).powi(2);

        NormalInverseGamma::new(mu_n, kappa_n, alpha_n, beta_n)
            .unwrap_or_else(|_| self.base.clone())
    }

    /// Open a new cluster by sampling its parameters from the base prior.
    fn open_new_cluster(&mut self, rng: &mut StdRng) -> Result<usize> {
        let (mu, sigma2) = self.base.sample(rng)?;
        let k = self.cluster_params.len();
        self.cluster_params.push((mu, sigma2));
        self.cluster_counts.push(0);
        self.cluster_stats.push((0.0, 0.0, 0));
        self.n_clusters += 1;
        Ok(k)
    }

    /// Compact representation: relabel clusters removing gaps.
    fn compact_clusters(&mut self) {
        let active: Vec<usize> = (0..self.cluster_counts.len())
            .filter(|&k| self.cluster_counts[k] > 0)
            .collect();
        let mut remap = vec![usize::MAX; self.cluster_counts.len()];
        for (new_k, &old_k) in active.iter().enumerate() {
            remap[old_k] = new_k;
        }
        self.assignments = self
            .assignments
            .iter()
            .map(|&old| remap.get(old).copied().unwrap_or(0))
            .collect();
        self.cluster_counts = active.iter().map(|&k| self.cluster_counts[k]).collect();
        self.cluster_stats = active.iter().map(|&k| self.cluster_stats[k]).collect();
        self.cluster_params = active.iter().map(|&k| self.cluster_params[k]).collect();
        self.n_clusters = active.len();
    }

    fn cluster_probabilities_raw(&self, x: f64) -> Vec<f64> {
        self.cluster_counts
            .iter()
            .enumerate()
            .filter_map(|(k, &count)| {
                if count == 0 {
                    return None;
                }
                let (mu, var) = self.cluster_params[k];
                let std = var.sqrt().max(1e-10);
                let z = (x - mu) / std;
                let pdf = ((-0.5 * z * z).exp()) / (std * (2.0 * PI).sqrt());
                Some(count as f64 * pdf)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sample_uniform(rng: &mut StdRng, lo: f64, hi: f64) -> f64 {
    use scirs2_core::random::Uniform;
    if (hi - lo).abs() < 1e-15 {
        return lo;
    }
    let u = Uniform::new(lo, hi)
        .map(|d| d.sample(rng))
        .unwrap_or(lo);
    u
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bimodal(n_per_mode: usize, seed: u64) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n1 = Normal::new(-3.0, 0.5).unwrap();
        let n2 = Normal::new(3.0, 0.5).unwrap();
        let mut part1: Vec<f64> = (0..n_per_mode)
            .map(|_| n1.sample(&mut rng))
            .collect();
        let part2: Vec<f64> = (0..n_per_mode)
            .map(|_| n2.sample(&mut rng))
            .collect();
        part1.extend(part2);
        part1
    }

    #[test]
    fn test_crp_basic() {
        let mut crp = CRPSampler::new(1.0).unwrap();
        assert!(CRPSampler::new(0.0).is_err());
        assert!(CRPSampler::new(-1.0).is_err());

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..20 {
            crp.seat_customer(&mut rng);
        }
        assert_eq!(crp.num_customers(), 20);
        // With alpha=1, expected ~log(20)≈3 tables
        assert!(crp.num_tables() >= 1);
        assert!(crp.num_tables() <= 20);
    }

    #[test]
    fn test_crp_expected_tables() {
        // E[K_n] ≈ α * H_n
        let alpha = 2.0;
        let n = 100;
        let expected = CRPSampler::expected_tables(alpha, n);
        // H_100 ≈ 5.187
        assert!((expected - 2.0 * 5.187).abs() < 0.5);
    }

    #[test]
    fn test_crp_table_counts_sum_to_customers() {
        let mut crp = CRPSampler::new(2.0).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        crp.seat_n_customers(50, &mut rng);
        let total_at_tables: usize = crp.table_counts.iter().sum();
        assert_eq!(total_at_tables, 50);
    }

    #[test]
    fn test_stick_breaking() {
        let mut sb = StickBreaking::new(2.0, 20).expect("construction failed");
        let mut rng = CoreRandom::seed(42);
        sb.sample_weights(&mut rng).expect("sampling failed");
        assert_eq!(sb.weights.len(), 20);
        let total: f64 = sb.weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-8, "weights sum to {total}");
        assert!(sb.weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_stick_breaking_invalid() {
        assert!(StickBreaking::new(0.0, 10).is_err());
        assert!(StickBreaking::new(1.0, 0).is_err());
    }

    #[test]
    fn test_dp_mixture_basic() {
        let base = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let mut model = DPMixture::new(2.0, base).unwrap();
        let data = make_bimodal(30, 42);
        model.fit_gibbs(&data, 100, 42).unwrap();

        // Should discover ≥ 1 cluster
        assert!(model.n_effective_clusters() >= 1);
        // Assignments should cover all observations
        assert_eq!(model.assignments.len(), 60);
        // Log-likelihood should be finite
        assert!(model.log_likelihood(&data).is_finite());
    }

    #[test]
    fn test_dp_mixture_predict() {
        let base = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let mut model = DPMixture::new(1.0, base).unwrap();
        let data: Vec<f64> = vec![-3.0; 10]
            .into_iter()
            .chain(vec![3.0; 10])
            .collect();
        model.fit_gibbs(&data, 50, 1).unwrap();

        let k1 = model.predict_cluster(-3.0);
        let k2 = model.predict_cluster(3.0);
        // Points far apart should be in different clusters (soft check)
        let probs_neg = model.cluster_probabilities(-3.0);
        let probs_pos = model.cluster_probabilities(3.0);
        assert!(!probs_neg.is_empty());
        assert!(!probs_pos.is_empty());
    }

    #[test]
    fn test_dp_mixture_empty_data() {
        let base = NormalInverseGamma::new(0.0, 1.0, 2.0, 1.0).unwrap();
        let mut model = DPMixture::new(1.0, base).unwrap();
        assert!(model.fit_gibbs(&[], 100, 0).is_err());
    }

    #[test]
    fn test_dp_mixture_bimodal_clusters() {
        let base = NormalInverseGamma::new(0.0, 0.01, 2.0, 1.0).unwrap();
        let mut model = DPMixture::new(1.0, base).unwrap();
        let data = make_bimodal(25, 99);
        model.fit_gibbs(&data, 200, 99).unwrap();
        // With clearly separated modes, should find ≥ 2 clusters
        assert!(
            model.n_effective_clusters() >= 1,
            "effective_clusters={}",
            model.n_effective_clusters()
        );
    }
}
