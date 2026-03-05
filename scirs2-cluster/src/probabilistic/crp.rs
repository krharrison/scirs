//! Chinese Restaurant Process (CRP) and Pitman-Yor Process.
//!
//! The Chinese Restaurant Process is a constructive definition of the Dirichlet
//! Process that describes the sequential allocation of observations to clusters
//! (tables in the restaurant metaphor).
//!
//! # Chinese Restaurant Process
//!
//! Given `N` customers and concentration parameter `α > 0`:
//!
//! - Customer 1 always sits at table 1.
//! - Customer `n` (for n ≥ 2) sits at:
//!   - An existing table `k` with probability `n_k / (n - 1 + α)` where `n_k`
//!     is the current number of customers at table `k`.
//!   - A new table with probability `α / (n - 1 + α)`.
//!
//! # Pitman-Yor Process
//!
//! Generalises the CRP by adding a discount parameter `0 ≤ d < 1`:
//!
//! - Customer `n` sits at existing table `k` with prob `(n_k - d) / (n - 1 + α)`.
//! - Customer `n` opens a new table with prob `(α + K·d) / (n - 1 + α)` where `K`
//!   is the current number of non-empty tables.
//!
//! Setting `d = 0` recovers the standard CRP.
//!
//! # Gibbs Sampling
//!
//! `gibbs_sampler_crp` implements collapsed Gibbs sampling for a CRP mixture
//! model where the base measure is a Normal-Normal conjugate model.
//!
//! # References
//!
//! - Aldous (1985) "Exchangeability and related topics" in École d'Été de
//!   Probabilités de Saint-Flour XIII.
//! - Pitman & Yor (1997) "The two-parameter Poisson-Dirichlet distribution
//!   derived from a stable subordinator".
//! - Neal (2000) "Markov chain sampling methods for Dirichlet process mixture models".

use std::collections::HashMap;

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ────────────────────────────────────────────────────────────────────────────
// Simple pseudo-random utilities (no external rand crate per SciRS2 policy)
// ────────────────────────────────────────────────────────────────────────────

/// A simple linear congruential generator for reproducible sampling.
/// State is a u64 seed.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    /// Advance and return a value in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Numerical Recipes LCG parameters
        self.state = self.state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use top 53 bits for mantissa
        let bits = self.state >> 11;
        bits as f64 / (1u64 << 53) as f64
    }

    /// Sample a uniform integer in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        let u = self.next_f64();
        ((u * n as f64) as usize).min(n - 1)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// CRP Sampler
// ────────────────────────────────────────────────────────────────────────────

/// Chinese Restaurant Process sampler.
///
/// Generates table (cluster) assignments for a sequence of N customers
/// according to the CRP with concentration parameter `alpha`.
#[derive(Debug, Clone)]
pub struct CRPSampler {
    /// Concentration parameter (> 0).
    pub alpha: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl CRPSampler {
    /// Create a new CRP sampler.
    ///
    /// # Arguments
    /// * `alpha` – concentration parameter (> 0). Larger values create more tables.
    pub fn new(alpha: f64) -> Self {
        Self { alpha, seed: 42 }
    }

    /// Create a CRP sampler with a specific random seed.
    pub fn with_seed(alpha: f64, seed: u64) -> Self {
        Self { alpha, seed }
    }

    /// Sample table assignments for `n_customers` customers.
    ///
    /// Returns a vector of table indices (0-based) of length `n_customers`.
    /// The first customer always sits at table 0.
    pub fn sample_seating(&self, n_customers: usize) -> Result<Array1<usize>> {
        if n_customers == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_customers must be at least 1".to_string(),
            ));
        }
        if self.alpha <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        let mut rng = Lcg::new(self.seed);
        let mut assignments = Array1::<usize>::zeros(n_customers);
        let mut table_counts: Vec<usize> = Vec::new();

        // Customer 0 always goes to table 0
        assignments[0] = 0;
        table_counts.push(1);

        for i in 1..n_customers {
            let total = i as f64 + self.alpha;
            let u = rng.next_f64() * total;
            let mut cumulative = 0.0;
            let mut chosen_table = table_counts.len(); // default: new table

            for (k, &count) in table_counts.iter().enumerate() {
                cumulative += count as f64;
                if u < cumulative {
                    chosen_table = k;
                    break;
                }
            }

            assignments[i] = chosen_table;
            if chosen_table < table_counts.len() {
                table_counts[chosen_table] += 1;
            } else {
                table_counts.push(1);
            }
        }

        Ok(assignments)
    }

    /// Compute the CRP probability `P(z_n = k | z_{1:n-1})` for a given
    /// sequence of previous assignments.
    ///
    /// # Arguments
    /// * `prev_assignments` – assignments of the first n-1 customers.
    /// * `k` – table index to query. Pass `None` for "new table".
    ///
    /// Returns the (unnormalised) CRP probability.
    pub fn crp_probability(
        &self,
        prev_assignments: &[usize],
        k: Option<usize>,
    ) -> Result<f64> {
        if self.alpha <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        let n_prev = prev_assignments.len();
        let normaliser = n_prev as f64 + self.alpha;

        match k {
            None => {
                // Probability of opening a new table
                Ok(self.alpha / normaliser)
            }
            Some(table) => {
                // Count customers already at table `table`
                let count = prev_assignments.iter().filter(|&&t| t == table).count();
                if count == 0 {
                    // Table doesn't exist yet — treat as new table
                    Ok(self.alpha / normaliser)
                } else {
                    Ok(count as f64 / normaliser)
                }
            }
        }
    }

    /// Compute the full CRP predictive probability vector for a new customer.
    ///
    /// Returns `(probs, n_tables)` where `probs[k]` is the probability of
    /// joining existing table `k`, and the last entry is the new-table probability.
    pub fn predictive_distribution(
        &self,
        prev_assignments: &[usize],
    ) -> Result<(Vec<f64>, usize)> {
        if self.alpha <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "alpha must be positive".to_string(),
            ));
        }

        let n_prev = prev_assignments.len();
        let normaliser = n_prev as f64 + self.alpha;

        // Count table occupancies
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &t in prev_assignments {
            *counts.entry(t).or_insert(0) += 1;
        }
        let n_tables = counts.len();

        let mut probs: Vec<f64> = Vec::with_capacity(n_tables + 1);
        let mut tables: Vec<usize> = counts.keys().copied().collect();
        tables.sort_unstable();

        for t in &tables {
            probs.push(*counts.get(t).unwrap_or(&0) as f64 / normaliser);
        }
        // New table probability
        probs.push(self.alpha / normaliser);

        Ok((probs, n_tables))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Gibbs sampler for CRP mixture model
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for Gibbs sampling of a CRP mixture model.
#[derive(Debug, Clone)]
pub struct CRPGibbsConfig {
    /// DP concentration parameter.
    pub alpha: f64,
    /// Number of Gibbs sweeps (iterations).
    pub n_iter: usize,
    /// Number of burn-in sweeps to discard.
    pub n_burnin: usize,
    /// Prior mean for cluster parameters (Normal base measure).
    pub prior_mean: f64,
    /// Prior variance for cluster parameters.
    pub prior_var: f64,
    /// Likelihood variance (assumed known).
    pub likelihood_var: f64,
    /// Random seed.
    pub seed: u64,
}

impl CRPGibbsConfig {
    /// Create a new Gibbs sampler configuration.
    pub fn new(alpha: f64, n_iter: usize) -> Self {
        Self {
            alpha,
            n_iter,
            n_burnin: n_iter / 4,
            prior_mean: 0.0,
            prior_var: 1.0,
            likelihood_var: 0.1,
            seed: 42,
        }
    }
}

/// Result of Gibbs sampling for a CRP mixture.
#[derive(Debug, Clone)]
pub struct CRPGibbsResult {
    /// Final cluster assignments (N,).
    pub assignments: Array1<usize>,
    /// Number of active clusters.
    pub n_clusters: usize,
    /// Cluster means (averaged over post-burnin samples).
    pub cluster_means: HashMap<usize, f64>,
    /// Number of Gibbs iterations performed.
    pub n_iter: usize,
}

/// Gibbs sampler for a CRP mixture model with Normal-Normal conjugacy.
///
/// Each cluster `k` has a Gaussian likelihood `x_i ~ N(μ_k, σ²)` with a
/// Normal prior `μ_k ~ N(μ₀, τ²)`.  The cluster labels are marginalised out
/// by integrating over `μ_k`, yielding a marginal likelihood that can be
/// evaluated in closed form.
pub fn gibbs_sampler_crp(
    data: ArrayView2<f64>,
    config: &CRPGibbsConfig,
) -> Result<CRPGibbsResult> {
    let n = data.nrows();
    if n == 0 {
        return Err(ClusteringError::InvalidInput(
            "data must have at least one row".to_string(),
        ));
    }
    if config.alpha <= 0.0 {
        return Err(ClusteringError::InvalidInput(
            "alpha must be positive".to_string(),
        ));
    }

    // Use only the first feature dimension for this 1-D Gibbs sampler
    // (extensions to D-dimensional data follow the same pattern)
    let d = data.ncols();

    let mut rng = Lcg::new(config.seed);

    // Initialise: all points in one cluster
    let mut assignments: Vec<usize> = vec![0usize; n];
    // Track cluster membership: cluster_id -> sorted list of member indices
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    clusters.insert(0, (0..n).collect());
    let mut next_cluster_id = 1usize;

    let tau2 = config.prior_var;
    let sigma2 = config.likelihood_var;
    let mu0 = config.prior_mean;

    // Helper: log marginal likelihood of a cluster given its members.
    // Integrates out μ_k under Normal-Normal conjugacy.
    // log p(X_k) = Σ log N(x_i; μ₀, σ² + τ²) ... simplified predictive
    let log_marginal = |members: &[usize], x_new: f64| -> f64 {
        let n_k = members.len() as f64;
        // Posterior variance of μ given data
        let tau2_n = 1.0 / (1.0 / tau2 + n_k / sigma2);
        // Posterior mean
        let sum_x: f64 = members
            .iter()
            .map(|&i| (0..d).map(|j| data[[i, j]]).sum::<f64>() / d as f64)
            .sum();
        let mu_n = tau2_n * (mu0 / tau2 + sum_x / sigma2);
        // Predictive: N(x_new; mu_n, sigma2 + tau2_n)
        let pred_var = sigma2 + tau2_n;
        let x_mean = (0..d).map(|j| data[[(x_new as usize).min(n-1), j]]).sum::<f64>() / d as f64;
        -0.5 * (2.0 * std::f64::consts::PI * pred_var).ln()
            - 0.5 * (x_mean - mu_n).powi(2) / pred_var
    };

    // Accumulated assignments for averaging (post-burnin)
    let mut accumulated: Vec<Vec<usize>> = Vec::new();

    for iter in 0..config.n_iter {
        // One full Gibbs sweep
        for i in 0..n {
            // Remove i from its current cluster
            let old_k = assignments[i];
            let members = clusters.entry(old_k).or_default();
            members.retain(|&m| m != i);
            if members.is_empty() {
                clusters.remove(&old_k);
            }

            // Build probability vector over existing clusters + new cluster
            let mut log_probs: Vec<(usize, f64)> = Vec::new();
            let total_minus_i = n as f64 - 1.0;

            for (&k, members) in &clusters {
                let n_k = members.len() as f64;
                let crp_prior = n_k / (total_minus_i + config.alpha);
                let log_lik = log_marginal(members, i as f64);
                log_probs.push((k, crp_prior.ln() + log_lik));
            }

            // New cluster contribution
            let new_cluster_logprob = (config.alpha / (total_minus_i + config.alpha)).ln()
                + log_marginal(&[], i as f64);
            log_probs.push((next_cluster_id, new_cluster_logprob));

            // Normalise
            let log_vals: Vec<f64> = log_probs.iter().map(|(_, lp)| *lp).collect();
            let max_lp = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = log_vals.iter().map(|&lp| (lp - max_lp).exp()).sum();

            // Sample
            let u = rng.next_f64() * sum_exp;
            let mut cumsum = 0.0;
            let mut chosen_k = log_probs[0].0;
            for (k, lp) in &log_probs {
                cumsum += (lp - max_lp).exp();
                if u <= cumsum {
                    chosen_k = *k;
                    break;
                }
            }

            // Assign point i to chosen_k
            assignments[i] = chosen_k;
            clusters.entry(chosen_k).or_default().push(i);
            if chosen_k == next_cluster_id {
                next_cluster_id += 1;
            }
        }

        // Collect post-burnin samples
        if iter >= config.n_burnin {
            accumulated.push(assignments.clone());
        }
    }

    // Compute consensus assignment (majority vote) from accumulated samples
    let mut final_assignments = Array1::<usize>::zeros(n);
    for i in 0..n {
        let mut vote_counts: HashMap<usize, usize> = HashMap::new();
        for sample in &accumulated {
            *vote_counts.entry(sample[i]).or_insert(0) += 1;
        }
        let best = vote_counts
            .into_iter()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k)
            .unwrap_or(0);
        final_assignments[i] = best;
    }

    // Re-label clusters contiguously (0-based)
    let unique_labels: std::collections::BTreeSet<usize> =
        final_assignments.iter().copied().collect();
    let label_map: HashMap<usize, usize> = unique_labels
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();
    for v in final_assignments.iter_mut() {
        *v = *label_map.get(v).unwrap_or(v);
    }

    let n_clusters = label_map.len();

    // Compute per-cluster means from the final assignment
    let mut cluster_means: HashMap<usize, f64> = HashMap::new();
    let mut cluster_sums: HashMap<usize, (f64, usize)> = HashMap::new();
    for i in 0..n {
        let k = final_assignments[i];
        let x_mean: f64 = (0..d).map(|j| data[[i, j]]).sum::<f64>() / d as f64;
        let entry = cluster_sums.entry(k).or_insert((0.0, 0));
        entry.0 += x_mean;
        entry.1 += 1;
    }
    for (k, (sum, count)) in cluster_sums {
        cluster_means.insert(k, sum / count as f64);
    }

    Ok(CRPGibbsResult {
        assignments: final_assignments,
        n_clusters,
        cluster_means,
        n_iter: config.n_iter,
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Pitman-Yor Process
// ────────────────────────────────────────────────────────────────────────────

/// Pitman-Yor Process sampler.
///
/// Generalises the CRP by introducing a discount parameter `0 ≤ discount < 1`.
/// With `discount = 0` this reduces to the standard CRP (Dirichlet Process).
/// With `discount > 0` the process generates power-law distributed cluster sizes,
/// which is useful for modelling natural-language phenomena.
#[derive(Debug, Clone)]
pub struct PitmanYorProcess {
    /// Concentration parameter `α > -discount`.
    pub alpha: f64,
    /// Discount parameter `0 ≤ discount < 1`.
    pub discount: f64,
    /// RNG seed.
    pub seed: u64,
}

impl PitmanYorProcess {
    /// Create a new Pitman-Yor sampler.
    ///
    /// # Arguments
    /// * `alpha` – concentration parameter (`> -discount`).
    /// * `discount` – discount parameter (`0 ≤ discount < 1`).
    pub fn new(alpha: f64, discount: f64) -> Result<Self> {
        if discount < 0.0 || discount >= 1.0 {
            return Err(ClusteringError::InvalidInput(
                "discount must be in [0, 1)".to_string(),
            ));
        }
        if alpha <= -discount {
            return Err(ClusteringError::InvalidInput(
                "alpha must be > -discount".to_string(),
            ));
        }
        Ok(Self { alpha, discount, seed: 42 })
    }

    /// Create with a specific random seed.
    pub fn with_seed(alpha: f64, discount: f64, seed: u64) -> Result<Self> {
        let mut py = Self::new(alpha, discount)?;
        py.seed = seed;
        Ok(py)
    }

    /// Sample table assignments for `n_customers` customers under the PY process.
    ///
    /// Returns a vector of table indices (0-based) of length `n_customers`.
    pub fn sample_seating(&self, n_customers: usize) -> Result<Array1<usize>> {
        if n_customers == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_customers must be at least 1".to_string(),
            ));
        }

        let mut rng = Lcg::new(self.seed);
        let mut assignments = Array1::<usize>::zeros(n_customers);
        let mut table_counts: Vec<usize> = Vec::new();

        // Customer 0 always goes to table 0
        assignments[0] = 0;
        table_counts.push(1);

        for i in 1..n_customers {
            let n_tables = table_counts.len();
            let total = i as f64 + self.alpha;
            let new_table_prob = (self.alpha + n_tables as f64 * self.discount) / total;
            let u = rng.next_f64();

            if u < new_table_prob || table_counts.is_empty() {
                // New table
                assignments[i] = n_tables;
                table_counts.push(1);
            } else {
                // Existing table — sample proportional to (count - discount)
                let mut cumulative = 0.0;
                let denom: f64 = table_counts
                    .iter()
                    .map(|&c| (c as f64 - self.discount).max(0.0))
                    .sum();

                // Rescale u to [0, denom) accounting for new_table_prob
                let u_rescaled = rng.next_f64() * denom;
                let mut chosen = n_tables - 1; // fallback
                for (k, &count) in table_counts.iter().enumerate() {
                    cumulative += (count as f64 - self.discount).max(0.0);
                    if u_rescaled <= cumulative {
                        chosen = k;
                        break;
                    }
                }
                assignments[i] = chosen;
                table_counts[chosen] += 1;
            }
        }

        Ok(assignments)
    }

    /// Compute the expected number of tables (clusters) for `n` customers.
    ///
    /// For the PY process, `E[K_n] ≈ (Γ(α+1+n)/Γ(α+1) - 1) * discount / alpha`
    /// when `discount > 0`, or `α * ln(1 + n/α)` for the DP case.
    pub fn expected_n_tables(&self, n: usize) -> f64 {
        if self.discount < 1e-10 {
            // Standard CRP
            self.alpha * (1.0 + n as f64 / self.alpha).ln()
        } else {
            // PY approximation
            let d = self.discount;
            let a = self.alpha;
            // Rising factorial Γ(a+1+n)/Γ(a+1) ≈ n^d * Γ(a+1+d)/Γ(a+1) for large n
            let log_ratio = d * (n as f64).ln() + lgamma(a + 1.0 + d) - lgamma(a + 1.0);
            (log_ratio.exp() - 1.0) * d / (a + d).max(1e-10)
        }
    }

    /// Compute the Pitman-Yor predictive probability for customer n.
    ///
    /// Returns `(existing_probs, new_table_prob)` where `existing_probs[k]` is
    /// the probability of joining table `k`.
    pub fn pitman_yor_process(
        &self,
        table_counts: &[usize],
        n_prev: usize,
    ) -> Result<(Vec<f64>, f64)> {
        if table_counts.is_empty() {
            return Ok((vec![], 1.0));
        }

        let n_tables = table_counts.len();
        let total = n_prev as f64 + self.alpha;
        if total.abs() < 1e-14 {
            return Err(ClusteringError::ComputationError(
                "degenerate normaliser in PY process".to_string(),
            ));
        }

        let new_table_prob = (self.alpha + n_tables as f64 * self.discount) / total;
        let existing_probs: Vec<f64> = table_counts
            .iter()
            .map(|&c| (c as f64 - self.discount).max(0.0) / total)
            .collect();

        Ok((existing_probs, new_table_prob))
    }
}

/// Convenience function: compute `E[log π_k]` under the stick-breaking
/// representation of the Pitman-Yor process.
///
/// Approximates the expected log mixing weights using a truncated representation
/// with `T` terms. Uses the Beta(1-discount, alpha+k*discount) stick-breaking.
pub fn py_e_log_stick_breaking(
    alpha: f64,
    discount: f64,
    t: usize,
) -> Result<Array1<f64>> {
    if discount < 0.0 || discount >= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "discount must be in [0, 1)".to_string(),
        ));
    }

    // For the PY process, the k-th stick-breaking Beta has parameters
    // (1 - discount, alpha + k * discount)
    let mut e_log_pi = Array1::<f64>::zeros(t);
    let mut cumulative = 0.0;

    for k in 0..t {
        let a_k = 1.0 - discount;
        let b_k = alpha + k as f64 * discount;
        let ab = a_k + b_k;
        if ab < 1e-14 {
            break;
        }
        let e_log_v = digamma(a_k) - digamma(ab);
        let e_log_1mv = digamma(b_k) - digamma(ab);
        e_log_pi[k] = e_log_v + cumulative;
        cumulative += e_log_1mv;
    }

    Ok(e_log_pi)
}

/// Digamma function (same as in dpgmm.rs — kept local to avoid cross-module deps).
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let mut v = x;
    let mut result = 0.0;
    while v < 6.0 {
        result -= 1.0 / v;
        v += 1.0;
    }
    result += v.ln() - 0.5 / v;
    let iv2 = 1.0 / (v * v);
    result -= iv2 * (1.0 / 12.0 - iv2 * (1.0 / 120.0 - iv2 / 252.0));
    result
}

/// Log-gamma (same Lanczos approximation as dpgmm.rs).
fn lgamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_9,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    let _ = G; // used via index offsets
    if x < 0.5 {
        return std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - lgamma(1.0 - x);
    }
    let xm1 = x - 1.0;
    let mut sum = C[0];
    for (i, &c) in C[1..].iter().enumerate() {
        sum += c / (xm1 + i as f64 + 1.0);
    }
    let t = xm1 + G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + sum.ln() + (xm1 + 0.5) * t.ln() - t
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_crp_seating_length() {
        let sampler = CRPSampler::new(1.0);
        let assignments = sampler.sample_seating(20).expect("sample");
        assert_eq!(assignments.len(), 20);
    }

    #[test]
    fn test_crp_seating_contiguous() {
        let sampler = CRPSampler::new(1.0);
        let assignments = sampler.sample_seating(50).expect("sample");
        let max_k = *assignments.iter().max().expect("non-empty");
        // Assignments should be contiguous (no gaps)
        let unique: std::collections::BTreeSet<usize> = assignments.iter().copied().collect();
        for k in 0..=max_k {
            assert!(unique.contains(&k), "missing table {k}");
        }
    }

    #[test]
    fn test_crp_first_customer_table0() {
        let sampler = CRPSampler::new(2.0);
        let assignments = sampler.sample_seating(5).expect("sample");
        assert_eq!(assignments[0], 0, "first customer must sit at table 0");
    }

    #[test]
    fn test_crp_probability_new_table_empty() {
        let sampler = CRPSampler::new(2.0);
        let p_new = sampler.crp_probability(&[], None).expect("prob");
        // With no previous customers, new table probability = alpha/(0+alpha) = 1.0
        assert!((p_new - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crp_probability_existing() {
        let sampler = CRPSampler::new(1.0);
        let prev = vec![0usize, 0, 1];
        // P(k=0) = 2/(3+1) = 0.5
        let p0 = sampler.crp_probability(&prev, Some(0)).expect("prob");
        assert!((p0 - 0.5).abs() < 1e-10, "p0 = {p0}");
        // P(k=1) = 1/4 = 0.25
        let p1 = sampler.crp_probability(&prev, Some(1)).expect("prob");
        assert!((p1 - 0.25).abs() < 1e-10, "p1 = {p1}");
        // P(new) = 1/4 = 0.25
        let p_new = sampler.crp_probability(&prev, None).expect("prob");
        assert!((p_new - 0.25).abs() < 1e-10, "p_new = {p_new}");
    }

    #[test]
    fn test_crp_higher_alpha_more_tables() {
        let sampler_low = CRPSampler::with_seed(0.1, 123);
        let sampler_high = CRPSampler::with_seed(5.0, 123);
        let a_low = sampler_low.sample_seating(100).expect("low");
        let a_high = sampler_high.sample_seating(100).expect("high");
        let n_tables_low: std::collections::HashSet<_> = a_low.iter().copied().collect();
        let n_tables_high: std::collections::HashSet<_> = a_high.iter().copied().collect();
        assert!(
            n_tables_high.len() >= n_tables_low.len(),
            "low={} high={}",
            n_tables_low.len(),
            n_tables_high.len()
        );
    }

    #[test]
    fn test_crp_invalid_alpha() {
        let sampler = CRPSampler::new(-1.0);
        assert!(sampler.sample_seating(10).is_err());
    }

    #[test]
    fn test_crp_invalid_n_customers() {
        let sampler = CRPSampler::new(1.0);
        assert!(sampler.sample_seating(0).is_err());
    }

    #[test]
    fn test_gibbs_crp_basic() {
        let data = Array2::from_shape_vec(
            (10, 1),
            vec![0.0, 0.1, -0.1, 0.05, 0.0, 5.0, 4.9, 5.1, 5.0, 4.95],
        )
        .expect("data");
        let cfg = CRPGibbsConfig::new(1.0, 20);
        let result = gibbs_sampler_crp(data.view(), &cfg).expect("gibbs");
        assert_eq!(result.assignments.len(), 10);
        assert!(result.n_clusters >= 1);
        assert_eq!(result.n_iter, 20);
    }

    #[test]
    fn test_gibbs_crp_recovers_clusters() {
        // Two well-separated clusters
        let data = Array2::from_shape_vec(
            (8, 1),
            vec![0.0, 0.1, -0.1, 0.05, 10.0, 9.9, 10.1, 10.05],
        )
        .expect("data");
        let cfg = CRPGibbsConfig {
            alpha: 1.0,
            n_iter: 50,
            n_burnin: 10,
            prior_mean: 5.0,
            prior_var: 25.0,
            likelihood_var: 0.5,
            seed: 99,
        };
        let result = gibbs_sampler_crp(data.view(), &cfg).expect("gibbs");
        // The two groups should have different labels
        let label_low = result.assignments[0];
        let label_high = result.assignments[4];
        assert_ne!(
            label_low, label_high,
            "expected distinct clusters, got same label {label_low}"
        );
    }

    #[test]
    fn test_pitman_yor_seating_length() {
        let py = PitmanYorProcess::new(1.0, 0.5).expect("py");
        let assignments = py.sample_seating(30).expect("sample");
        assert_eq!(assignments.len(), 30);
    }

    #[test]
    fn test_pitman_yor_invalid_discount() {
        assert!(PitmanYorProcess::new(1.0, 1.0).is_err());
        assert!(PitmanYorProcess::new(1.0, -0.1).is_err());
    }

    #[test]
    fn test_pitman_yor_discount_zero_like_crp() {
        // discount=0 should behave like standard CRP
        let py = PitmanYorProcess::with_seed(1.0, 0.0, 42).expect("py");
        let crp = CRPSampler::with_seed(1.0, 42);
        let py_a = py.sample_seating(50).expect("py sample");
        let crp_a = crp.sample_seating(50).expect("crp sample");
        // Both should produce the same number of tables (stochastically similar)
        let py_tables: std::collections::HashSet<_> = py_a.iter().copied().collect();
        let crp_tables: std::collections::HashSet<_> = crp_a.iter().copied().collect();
        // Allow a small difference due to different algorithms (not same RNG path)
        let diff = (py_tables.len() as isize - crp_tables.len() as isize).abs();
        assert!(diff <= 5, "PY tables={}, CRP tables={}", py_tables.len(), crp_tables.len());
    }

    #[test]
    fn test_pitman_yor_predictive() {
        let py = PitmanYorProcess::new(1.0, 0.3).expect("py");
        let counts = vec![5usize, 3, 1];
        let (existing, new_p) = py.pitman_yor_process(&counts, 9).expect("predictive");
        // Probabilities should be non-negative
        for p in &existing {
            assert!(*p >= 0.0, "negative prob {p}");
        }
        assert!(new_p >= 0.0);
        // Sum should be approximately 1
        let total: f64 = existing.iter().sum::<f64>() + new_p;
        assert!((total - 1.0).abs() < 1e-10, "total = {total}");
    }

    #[test]
    fn test_py_e_log_stick_breaking() {
        let e_log = py_e_log_stick_breaking(1.0, 0.5, 5).expect("e_log");
        assert_eq!(e_log.len(), 5);
        for &v in e_log.iter() {
            assert!(v.is_finite(), "non-finite value {v}");
        }
    }

    #[test]
    fn test_predictive_distribution() {
        let sampler = CRPSampler::new(2.0);
        let prev = vec![0usize, 0, 1, 2];
        let (probs, n_tables) = sampler.predictive_distribution(&prev).expect("dist");
        assert_eq!(n_tables, 3);
        // n+1 entries (3 existing tables + 1 new-table entry)
        assert_eq!(probs.len(), 4);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10, "total = {total}");
    }
}
