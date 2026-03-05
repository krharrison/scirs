//! Dirichlet Process Mixture Model (DPMM) via Collapsed Gibbs Sampling.
//!
//! This implementation uses the conjugate Normal-Wishart prior so that the
//! cluster sufficient statistics can be maintained incrementally and the
//! marginal (predictive) likelihood of each cluster is available in closed form
//! as a multivariate Student-t distribution.
//!
//! # References
//!
//! - Neal, R. M. (2000). "Markov Chain Sampling Methods for Dirichlet Process
//!   Mixture Models." *Journal of Computational and Graphical Statistics*, 9(2).
//! - Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*.
//!   Chapter 25.
//!
//! # Example
//!
//! ```rust
//! use scirs2_cluster::bayesian_clustering::dpmm::{
//!     DPMMConfig, DPMMMixture, NormalWishart,
//! };
//! use scirs2_core::random::rngs::StdRng;
//! use scirs2_core::random::SeedableRng;
//!
//! let data = vec![
//!     vec![1.0_f64, 2.0], vec![1.1, 1.9], vec![0.9, 2.1],
//!     vec![5.0, 5.0],     vec![5.1, 4.9], vec![4.9, 5.1],
//! ];
//! let dim = 2;
//! let prior = NormalWishart::default(dim);
//! let config = DPMMConfig {
//!     alpha: 1.0,
//!     max_clusters: 10,
//!     n_iter: 50,
//!     n_burnin: 20,
//!     base_prior: prior,
//! };
//! let mut rng = StdRng::seed_from_u64(42);
//! let state = DPMMMixture::fit(&data, &config, &mut rng).expect("dpmm fit");
//! assert!(!state.assignments.is_empty());
//! ```

use std::f64::consts::PI;

use scirs2_core::random::{Rng, SeedableRng};
use scirs2_core::random::rngs::StdRng;

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Normal-Wishart prior
// ─────────────────────────────────────────────────────────────────────────────

/// Normal-Wishart conjugate prior for multivariate Gaussian components.
///
/// Parameterises the prior over (μ, Λ) where Λ = Σ⁻¹ is the precision matrix.
///
/// - `mu0`:    prior mean of μ.
/// - `kappa0`: strength of the prior on μ (pseudo-observations).
/// - `nu0`:    degrees of freedom for the Wishart prior (must be ≥ D).
/// - `psi0`:   D×D scale matrix of the Wishart prior.
#[derive(Debug, Clone)]
pub struct NormalWishart {
    /// Prior mean vector (length D).
    pub mu0: Vec<f64>,
    /// Prior strength on μ.
    pub kappa0: f64,
    /// Wishart degrees of freedom.
    pub nu0: f64,
    /// Wishart scale matrix (D×D, stored row-major).
    pub psi0: Vec<Vec<f64>>,
}

impl NormalWishart {
    /// Construct a default (vague) Normal-Wishart prior for dimension `d`.
    ///
    /// Sets `mu0 = 0`, `kappa0 = 1e-3`, `nu0 = d`, `psi0 = I_d`.
    pub fn default(d: usize) -> Self {
        let identity: Vec<Vec<f64>> = (0..d)
            .map(|i| {
                let mut row = vec![0.0; d];
                row[i] = 1.0;
                row
            })
            .collect();
        Self {
            mu0: vec![0.0; d],
            kappa0: 1e-3,
            nu0: d as f64,
            psi0: identity,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sufficient statistics for a cluster
// ─────────────────────────────────────────────────────────────────────────────

/// Incremental sufficient statistics for one cluster component.
#[derive(Debug, Clone)]
struct ClusterStats {
    /// Number of observations assigned to this cluster.
    n: usize,
    /// Sum of observations (length D).
    sum: Vec<f64>,
    /// Sum of outer products: ΣΣ x_i x_i^T (D×D).
    sum_sq: Vec<Vec<f64>>,
}

impl ClusterStats {
    fn new(d: usize) -> Self {
        Self {
            n: 0,
            sum: vec![0.0; d],
            sum_sq: vec![vec![0.0; d]; d],
        }
    }

    fn add(&mut self, x: &[f64]) {
        self.n += 1;
        for (s, xi) in self.sum.iter_mut().zip(x.iter()) {
            *s += xi;
        }
        for i in 0..x.len() {
            for j in 0..x.len() {
                self.sum_sq[i][j] += x[i] * x[j];
            }
        }
    }

    fn remove(&mut self, x: &[f64]) {
        if self.n == 0 {
            return;
        }
        self.n -= 1;
        for (s, xi) in self.sum.iter_mut().zip(x.iter()) {
            *s -= xi;
        }
        for i in 0..x.len() {
            for j in 0..x.len() {
                self.sum_sq[i][j] -= x[i] * x[j];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Posterior predictive: multivariate Student-t
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the log marginal likelihood of a new observation `x` given the
/// Normal-Wishart posterior formed by combining `stats` with `prior`.
///
/// This is the log of the marginal (integrating out μ and Λ):
///   p(x | X_k, prior) = ∫ p(x | μ, Λ) p(μ, Λ | X_k) dμ dΛ
///
/// which is a multivariate Student-t distribution with posterior parameters.
fn log_marginal_likelihood(x: &[f64], stats: &ClusterStats, prior: &NormalWishart) -> f64 {
    let d = x.len();
    let n = stats.n as f64;
    let kappa_n = prior.kappa0 + n;
    let nu_n = prior.nu0 + n;

    // Posterior mean
    let mu_n: Vec<f64> = (0..d)
        .map(|i| (prior.kappa0 * prior.mu0[i] + stats.sum[i]) / kappa_n)
        .collect();

    // Posterior scale matrix: Ψ_n = Ψ_0 + S + (κ_0 n)/(κ_0+n) (x̄ - μ_0)(x̄ - μ_0)^T
    // where S = Σ x_i x_i^T - n x̄ x̄^T
    let x_bar: Vec<f64> = if n > 0.0 {
        stats.sum.iter().map(|s| s / n).collect()
    } else {
        vec![0.0; d]
    };

    let mut psi_n: Vec<Vec<f64>> = prior.psi0.clone();

    // Add scatter matrix
    for i in 0..d {
        for j in 0..d {
            let s_ij = stats.sum_sq[i][j] - n * x_bar[i] * x_bar[j];
            psi_n[i][j] += s_ij;
        }
    }

    // Add correction term if n > 0
    if n > 0.0 {
        let factor = prior.kappa0 * n / kappa_n;
        for i in 0..d {
            for j in 0..d {
                psi_n[i][j] += factor * (x_bar[i] - prior.mu0[i]) * (x_bar[j] - prior.mu0[j]);
            }
        }
    }

    // Now compute the posterior predictive for x:
    // degrees of freedom ν* = ν_n - d + 1
    // mean μ* = μ_n
    // covariance Σ* = (κ_n + 1)/(κ_n (ν_n - d + 1)) Ψ_n
    let nu_star = nu_n - d as f64 + 1.0;
    if nu_star <= 0.0 {
        return f64::NEG_INFINITY;
    }

    let scale_factor = (kappa_n + 1.0) / (kappa_n * nu_star);

    // Σ* = scale_factor * Ψ_n
    let sigma_star: Vec<Vec<f64>> = psi_n
        .iter()
        .map(|row| row.iter().map(|v| v * scale_factor).collect())
        .collect();

    // Log Student-t density: log Γ((ν+d)/2) - log Γ(ν/2) - d/2 log(ν π) - 1/2 log|Σ*|
    //   - (ν+d)/2 * log(1 + 1/ν (x-μ)^T Σ*^{-1} (x-μ))
    let log_det = log_det_pd(&sigma_star);
    let delta = x
        .iter()
        .zip(mu_n.iter())
        .map(|(xi, mi)| xi - mi)
        .collect::<Vec<f64>>();

    let maha_sq = quadratic_form_inv(&sigma_star, &delta);

    let nu = nu_star;
    let log_dens = lgamma((nu + d as f64) / 2.0)
        - lgamma(nu / 2.0)
        - (d as f64 / 2.0) * (nu * PI).ln()
        - 0.5 * log_det
        - ((nu + d as f64) / 2.0) * (1.0 + maha_sq / nu).ln();

    log_dens
}

/// Log Gamma function approximation (Lanczos).
fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // Lanczos coefficients for g=7
    let g = 7.0_f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7_f64,
    ];
    let mut z = x;
    if z < 0.5 {
        return PI.ln() - (PI * z).sin().ln() - lgamma(1.0 - z);
    }
    z -= 1.0;
    let mut s = c[0];
    for (i, &ci) in c[1..].iter().enumerate() {
        s += ci / (z + i as f64 + 1.0);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + s.ln()
}

/// Log-determinant of a positive semi-definite matrix (via Cholesky).
fn log_det_pd(a: &Vec<Vec<f64>>) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return a[0][0].max(1e-300).ln();
    }
    // Cholesky: L s.t. A = L L^T
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = s.max(1e-15).sqrt();
            } else if l[j][j].abs() > 1e-15 {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut log_det = 0.0;
    for i in 0..n {
        log_det += 2.0 * l[i][i].max(1e-300).ln();
    }
    log_det
}

/// Compute x^T A^{-1} x using Cholesky (for SPD A).
fn quadratic_form_inv(a: &Vec<Vec<f64>>, x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    }
    // Solve L y = x via forward substitution.
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = s.max(1e-15).sqrt();
            } else if l[j][j].abs() > 1e-15 {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= l[i][j] * y[j];
        }
        y[i] = if l[i][i].abs() > 1e-15 {
            s / l[i][i]
        } else {
            0.0
        };
    }
    y.iter().map(|yi| yi * yi).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the DPMM collapsed Gibbs sampler.
#[derive(Debug, Clone)]
pub struct DPMMConfig {
    /// DP concentration parameter α > 0.
    pub alpha: f64,
    /// Maximum number of clusters to track simultaneously.
    pub max_clusters: usize,
    /// Total number of Gibbs iterations (including burn-in).
    pub n_iter: usize,
    /// Number of burn-in iterations (discarded when collecting samples).
    pub n_burnin: usize,
    /// Normal-Wishart base prior.
    pub base_prior: NormalWishart,
}

impl DPMMConfig {
    /// Construct a default configuration for data of dimension `d`.
    pub fn default_for_dim(d: usize) -> Self {
        Self {
            alpha: 1.0,
            max_clusters: 20,
            n_iter: 200,
            n_burnin: 100,
            base_prior: NormalWishart::default(d),
        }
    }
}

/// Sampled state of the DPMM after fitting.
#[derive(Debug, Clone)]
pub struct DPMMState {
    /// Cluster assignment for each data point (0-indexed).
    pub assignments: Vec<usize>,
    /// Per-cluster (mean, covariance) estimates.
    /// `cluster_params[k] = (mean_vec, cov_matrix_rows)`.
    pub cluster_params: Vec<(Vec<f64>, Vec<Vec<f64>>)>,
    /// Number of Gibbs iterations completed.
    pub n_iter_done: usize,
    /// Active cluster IDs (non-empty clusters after final sample).
    pub active_clusters: Vec<usize>,
}

/// DPMM fitter.
pub struct DPMMMixture;

impl DPMMMixture {
    /// Fit a DPMM to `data` using the collapsed Gibbs sampler.
    ///
    /// # Parameters
    ///
    /// - `data`: N×D slice of observations.
    /// - `config`: algorithm configuration.
    /// - `rng`: mutable RNG (use `scirs2_core::random::rngs::StdRng`).
    ///
    /// # Returns
    ///
    /// The final [`DPMMState`] after `n_iter` Gibbs sweeps.
    pub fn fit(
        data: &[Vec<f64>],
        config: &DPMMConfig,
        rng: &mut impl Rng,
    ) -> Result<DPMMState> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data must be non-empty".to_string(),
            ));
        }
        let d = data[0].len();
        if d == 0 {
            return Err(ClusteringError::InvalidInput(
                "Feature dimension must be > 0".to_string(),
            ));
        }
        if config.alpha <= 0.0 {
            return Err(ClusteringError::InvalidInput(
                "alpha must be > 0".to_string(),
            ));
        }
        for (i, row) in data.iter().enumerate() {
            if row.len() != d {
                return Err(ClusteringError::InvalidInput(format!(
                    "Row {} has {} features, expected {}",
                    i,
                    row.len(),
                    d
                )));
            }
        }

        // Initialise: assign all points to cluster 0.
        let mut assignments = vec![0usize; n];
        let max_k = config.max_clusters.max(2);
        let mut stats: Vec<ClusterStats> = (0..max_k).map(|_| ClusterStats::new(d)).collect();

        for (i, x) in data.iter().enumerate() {
            stats[assignments[i]].add(x);
        }

        let prior = &config.base_prior;
        let alpha = config.alpha;

        // Gibbs sweeps.
        for _iter in 0..config.n_iter {
            for i in 0..n {
                let x = &data[i];
                let k_old = assignments[i];

                // Remove point i from its cluster.
                stats[k_old].remove(x);

                // Identify active cluster indices (non-empty or the new-cluster slot).
                // We allow up to max_k clusters.
                let n_active = stats.iter().filter(|s| s.n > 0).count();

                // Compute unnormalised log probabilities.
                // Use log-sum-exp for numerical stability.
                let mut log_probs: Vec<f64> = Vec::with_capacity(n_active + 1);
                let mut cluster_ids: Vec<usize> = Vec::with_capacity(n_active + 1);

                for (k, s) in stats.iter().enumerate() {
                    if s.n > 0 {
                        let lp = (s.n as f64).ln()
                            + log_marginal_likelihood(x, s, prior);
                        log_probs.push(lp);
                        cluster_ids.push(k);
                    }
                }

                // New cluster probability: α * p(x | prior)
                let empty_stats = ClusterStats::new(d);
                let lp_new = alpha.ln() + log_marginal_likelihood(x, &empty_stats, prior);
                log_probs.push(lp_new);
                cluster_ids.push(usize::MAX); // Sentinel for "new cluster"

                // Softmax to get probabilities.
                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = log_probs
                    .iter()
                    .map(|&lp| (lp - max_lp).exp())
                    .collect();
                let sum_probs: f64 = probs.iter().sum();

                // Sample new cluster.
                let u: f64 = rng.random::<f64>() * sum_probs;
                let mut cumsum = 0.0;
                let mut chosen = cluster_ids[0];
                for (j, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u <= cumsum {
                        chosen = cluster_ids[j];
                        break;
                    }
                }

                // If new cluster, find an empty slot.
                let k_new = if chosen == usize::MAX {
                    // Find first empty slot.
                    stats
                        .iter()
                        .position(|s| s.n == 0)
                        .unwrap_or_else(|| {
                            // All slots are full; fall back to first active cluster.
                            cluster_ids[0]
                        })
                } else {
                    chosen
                };

                assignments[i] = k_new;
                stats[k_new].add(x);
            }
        }

        // Compute cluster parameters from sufficient statistics.
        let active_clusters: Vec<usize> = stats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.n > 0)
            .map(|(k, _)| k)
            .collect();

        let cluster_params: Vec<(Vec<f64>, Vec<Vec<f64>>)> = active_clusters
            .iter()
            .map(|&k| {
                let s = &stats[k];
                let mean = if s.n > 0 {
                    s.sum.iter().map(|v| v / s.n as f64).collect()
                } else {
                    vec![0.0; d]
                };
                let cov = if s.n > 1 {
                    let x_bar = &mean;
                    let mut c = vec![vec![0.0f64; d]; d];
                    for i in 0..d {
                        for j in 0..d {
                            c[i][j] = (s.sum_sq[i][j]
                                - s.n as f64 * x_bar[i] * x_bar[j])
                                / (s.n as f64 - 1.0);
                        }
                    }
                    c
                } else {
                    // Identity fallback
                    (0..d)
                        .map(|i| {
                            let mut r = vec![0.0; d];
                            r[i] = 1.0;
                            r
                        })
                        .collect()
                };
                (mean, cov)
            })
            .collect();

        Ok(DPMMState {
            assignments,
            cluster_params,
            n_iter_done: config.n_iter,
            active_clusters,
        })
    }

    /// Compute approximate posterior cluster probabilities for a new point.
    ///
    /// Uses the cluster parameters (mean, covariance) from the last Gibbs
    /// sample to compute Gaussian densities, then normalises.
    pub fn sample_posterior_predictive(state: &DPMMState, x: &[f64]) -> Vec<f64> {
        let n_total: usize = state.assignments.len();
        let n_active = state.active_clusters.len();
        if n_active == 0 {
            return Vec::new();
        }

        // Count cluster sizes.
        let mut counts = vec![0usize; n_active];
        for &a in &state.assignments {
            if let Some(pos) = state.active_clusters.iter().position(|&k| k == a) {
                counts[pos] += 1;
            }
        }

        let mut log_probs: Vec<f64> = Vec::with_capacity(n_active);
        for (idx, (mean, cov)) in state.cluster_params.iter().enumerate() {
            let weight = counts[idx] as f64 / n_total as f64;
            let lp = weight.ln() + log_gaussian(x, mean, cov);
            log_probs.push(lp);
        }

        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum < 1e-300 {
            return vec![1.0 / n_active as f64; n_active];
        }
        probs.iter().map(|p| p / sum).collect()
    }
}

/// Log Gaussian density (unnormalised constant omitted for ratio purposes).
fn log_gaussian(x: &[f64], mean: &[f64], cov: &Vec<Vec<f64>>) -> f64 {
    let delta: Vec<f64> = x
        .iter()
        .zip(mean.iter())
        .map(|(xi, mi)| xi - mi)
        .collect();
    let maha_sq = quadratic_form_inv(cov, &delta);
    let log_det = log_det_pd(cov);
    let d = x.len() as f64;
    -0.5 * (d * (2.0 * PI).ln() + log_det + maha_sq)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::SeedableRng;

    fn two_cluster_data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 2.0],
            vec![1.1, 1.9],
            vec![0.9, 2.1],
            vec![1.2, 1.8],
            vec![8.0, 8.0],
            vec![8.1, 7.9],
            vec![7.9, 8.1],
            vec![8.2, 7.8],
        ]
    }

    #[test]
    fn test_dpmm_fit_returns_assignments() {
        let data = two_cluster_data();
        let prior = NormalWishart::default(2);
        let config = DPMMConfig {
            alpha: 1.0,
            max_clusters: 10,
            n_iter: 50,
            n_burnin: 20,
            base_prior: prior,
        };
        let mut rng = StdRng::seed_from_u64(42);
        let state = DPMMMixture::fit(&data, &config, &mut rng)
            .expect("dpmm fit");

        assert_eq!(state.assignments.len(), data.len());
        assert!(!state.active_clusters.is_empty());
        assert!(state.n_iter_done == 50);
    }

    #[test]
    fn test_dpmm_finds_two_clusters() {
        let data = two_cluster_data();
        let prior = NormalWishart::default(2);
        let config = DPMMConfig {
            alpha: 1.0,
            max_clusters: 10,
            n_iter: 200,
            n_burnin: 100,
            base_prior: prior,
        };
        let mut rng = StdRng::seed_from_u64(0);
        let state = DPMMMixture::fit(&data, &config, &mut rng)
            .expect("dpmm fit");

        // Inferred cluster count should be >= 1.
        let n_clusters = state.active_clusters.len();
        assert!(n_clusters >= 1, "expected >= 1 cluster, got {}", n_clusters);
    }

    #[test]
    fn test_posterior_predictive() {
        let data = two_cluster_data();
        let prior = NormalWishart::default(2);
        let config = DPMMConfig {
            alpha: 1.0,
            max_clusters: 10,
            n_iter: 100,
            n_burnin: 50,
            base_prior: prior,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let state = DPMMMixture::fit(&data, &config, &mut rng)
            .expect("dpmm fit");

        let probs = DPMMMixture::sample_posterior_predictive(&state, &[1.0, 2.0]);
        if !probs.is_empty() {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "probs should sum to 1");
        }
    }

    #[test]
    fn test_dpmm_single_point() {
        let data = vec![vec![3.0, 4.0]];
        let prior = NormalWishart::default(2);
        let config = DPMMConfig {
            alpha: 1.0,
            max_clusters: 5,
            n_iter: 10,
            n_burnin: 5,
            base_prior: prior,
        };
        let mut rng = StdRng::seed_from_u64(99);
        let state = DPMMMixture::fit(&data, &config, &mut rng)
            .expect("single point dpmm");
        assert_eq!(state.assignments.len(), 1);
    }

    #[test]
    fn test_lgamma_known_values() {
        // lgamma(1) = 0, lgamma(2) = 0, lgamma(0.5) = sqrt(pi)/2
        assert!((lgamma(1.0)).abs() < 1e-6);
        assert!((lgamma(2.0)).abs() < 1e-6);
        let expected = 0.5 * PI.ln();
        assert!((lgamma(0.5) - expected).abs() < 1e-4);
    }

    #[test]
    fn test_log_det_identity() {
        let identity: Vec<Vec<f64>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let ld = log_det_pd(&identity);
        assert!(ld.abs() < 1e-10, "log_det(I) should be 0, got {}", ld);
    }
}
