//! Stochastic Block Model (SBM) for community detection.
//!
//! The SBM is a generative model for networks with community structure.
//! Given K blocks (communities), the probability of an edge between nodes i
//! and j depends only on their block assignments z_i and z_j through a
//! K x K probability matrix B.
//!
//! This module provides:
//!
//! - **Fitting via variational EM**: infer block assignments and the B matrix
//! - **Degree-corrected SBM**: accounts for degree heterogeneity within blocks
//! - **Model selection**: choose K via Integrated Classification Likelihood (ICL)
//! - **Network generation**: sample random graphs from a fitted or specified SBM

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{AdjacencyGraph, CommunityResult};
use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for SBM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticBlockModelConfig {
    /// Number of blocks (communities). If `None`, model selection is used.
    pub num_blocks: Option<usize>,
    /// Range of K values to try when `num_blocks` is `None`.
    pub k_range: (usize, usize),
    /// Maximum EM iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on log-likelihood change.
    pub convergence_threshold: f64,
    /// Whether to use degree-corrected SBM.
    pub degree_corrected: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for StochasticBlockModelConfig {
    fn default() -> Self {
        Self {
            num_blocks: None,
            k_range: (2, 8),
            max_iterations: 100,
            convergence_threshold: 1e-6,
            degree_corrected: false,
            seed: 42,
        }
    }
}

/// Result of SBM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SBMResult {
    /// Community detection result (labels, num_communities, quality_score).
    pub community: CommunityResult,
    /// Block probability matrix B (K x K), row-major.
    pub block_matrix: Vec<f64>,
    /// Number of blocks K.
    pub k: usize,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
    /// ICL score (for model comparison).
    pub icl_score: f64,
    /// Degree correction factors (only for degree-corrected SBM).
    pub degree_corrections: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// SBM Core
// ---------------------------------------------------------------------------

/// The Stochastic Block Model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticBlockModel {
    /// Configuration.
    pub config: StochasticBlockModelConfig,
}

impl StochasticBlockModel {
    /// Create a new SBM with the given configuration.
    pub fn new(config: StochasticBlockModelConfig) -> Self {
        Self { config }
    }

    /// Fit the SBM to the observed adjacency graph.
    ///
    /// If `config.num_blocks` is set, fits with that K.
    /// Otherwise, tries each K in `config.k_range` and picks the one
    /// with the best ICL score.
    pub fn fit(&self, graph: &AdjacencyGraph) -> Result<SBMResult> {
        let n = graph.n_nodes;
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "Graph has zero nodes".to_string(),
            ));
        }

        if let Some(k) = self.config.num_blocks {
            if k == 0 || k > n {
                return Err(ClusteringError::InvalidInput(format!(
                    "num_blocks ({}) must be in [1, {}]",
                    k, n
                )));
            }
            self.fit_k(graph, k)
        } else {
            let k_min = self.config.k_range.0.max(1);
            let k_max = self.config.k_range.1.min(n);
            if k_min > k_max {
                return Err(ClusteringError::InvalidInput("Invalid k_range".to_string()));
            }

            let mut best: Option<SBMResult> = None;
            for k in k_min..=k_max {
                let result = self.fit_k(graph, k)?;
                let is_better = best
                    .as_ref()
                    .map(|b| result.icl_score > b.icl_score)
                    .unwrap_or(true);
                if is_better {
                    best = Some(result);
                }
            }

            best.ok_or_else(|| ClusteringError::ComputationError("No valid K found".to_string()))
        }
    }

    /// Fit SBM with a specific K.
    fn fit_k(&self, graph: &AdjacencyGraph, k: usize) -> Result<SBMResult> {
        let n = graph.n_nodes;
        let mut rng = Xorshift64::new(self.config.seed.wrapping_add(k as u64));

        // Build dense adjacency for fast lookup.
        let mut adj_matrix = vec![0.0_f64; n * n];
        for i in 0..n {
            for &(j, w) in &graph.adjacency[i] {
                adj_matrix[i * n + j] = w;
            }
        }

        // Initialise tau (soft assignment matrix): n x k, row-major.
        // Start with a noisy K-means-like initialisation.
        let mut tau = vec![0.0_f64; n * k];
        for i in 0..n {
            let assigned = (i * k / n) % k; // spread nodes across blocks
            for r in 0..k {
                tau[i * k + r] = if r == assigned {
                    0.8
                } else {
                    0.2 / ((k - 1).max(1) as f64)
                };
            }
            // Add noise.
            let noise_sum: f64 = (0..k).map(|_| rng.next_f64() * 0.1).sum();
            for r in 0..k {
                tau[i * k + r] += rng.next_f64() * 0.1;
            }
            // Normalise.
            let row_sum: f64 = (0..k).map(|r| tau[i * k + r]).sum();
            if row_sum > 0.0 {
                for r in 0..k {
                    tau[i * k + r] /= row_sum;
                }
            }
            let _ = noise_sum; // suppress warning
        }

        // Block probability matrix B: k x k, row-major.
        let mut b_matrix = vec![0.0_f64; k * k];
        // Degree corrections.
        let mut theta = vec![1.0_f64; n];

        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..self.config.max_iterations {
            // --- M-step: update B and optionally theta ---
            self.m_step(graph, &adj_matrix, &tau, &mut b_matrix, &mut theta, n, k);

            // --- E-step: update tau ---
            self.e_step(graph, &adj_matrix, &b_matrix, &theta, &mut tau, n, k);

            // --- Compute log-likelihood ---
            let ll = self.log_likelihood(&adj_matrix, &b_matrix, &theta, &tau, n, k);

            if (ll - prev_ll).abs() < self.config.convergence_threshold {
                break;
            }
            prev_ll = ll;
        }

        // Hard assignment: argmax of tau.
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let mut best_r = 0;
            let mut best_val = f64::NEG_INFINITY;
            for r in 0..k {
                if tau[i * k + r] > best_val {
                    best_val = tau[i * k + r];
                    best_r = r;
                }
            }
            labels[i] = best_r;
        }

        // Compact labels (some blocks may be empty).
        let mut mapping: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0usize;
        for lbl in &labels {
            if !mapping.contains_key(lbl) {
                mapping.insert(*lbl, next_id);
                next_id += 1;
            }
        }
        let compacted: Vec<usize> = labels
            .iter()
            .map(|l| mapping.get(l).copied().unwrap_or(0))
            .collect();
        let num_communities = next_id;

        let ll = self.log_likelihood(&adj_matrix, &b_matrix, &theta, &tau, n, k);
        let icl = self.compute_icl(ll, &compacted, n, k);
        let quality = graph.modularity(&compacted);

        let degree_corrections = if self.config.degree_corrected {
            Some(theta)
        } else {
            None
        };

        Ok(SBMResult {
            community: CommunityResult {
                labels: compacted,
                num_communities,
                quality_score: Some(quality),
            },
            block_matrix: b_matrix,
            k,
            log_likelihood: ll,
            icl_score: icl,
            degree_corrections,
        })
    }

    /// M-step: update B matrix and degree corrections.
    fn m_step(
        &self,
        _graph: &AdjacencyGraph,
        adj_matrix: &[f64],
        tau: &[f64],
        b_matrix: &mut [f64],
        theta: &mut [f64],
        n: usize,
        k: usize,
    ) {
        // B_{rs} = sum_{i,j} tau_{ir} * A_{ij} * tau_{js} / sum_{i,j} tau_{ir} * tau_{js}
        for r in 0..k {
            for s in 0..k {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for i in 0..n {
                    let tau_ir = tau[i * k + r];
                    if tau_ir < 1e-15 {
                        continue;
                    }
                    for j in 0..n {
                        if i == j {
                            continue;
                        }
                        let tau_js = tau[j * k + s];
                        if tau_js < 1e-15 {
                            continue;
                        }
                        numerator += tau_ir * adj_matrix[i * n + j] * tau_js;
                        denominator += tau_ir * tau_js;
                    }
                }
                // Clamp to [epsilon, 1 - epsilon] for numerical stability.
                let val = if denominator > 1e-15 {
                    numerator / denominator
                } else {
                    0.5
                };
                b_matrix[r * k + s] = val.clamp(1e-10, 1.0 - 1e-10);
            }
        }

        // Degree corrections (degree-corrected SBM).
        if self.config.degree_corrected {
            // theta_i = (actual degree of i) / (expected degree under SBM)
            for i in 0..n {
                let actual_deg: f64 = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| adj_matrix[i * n + j])
                    .sum();

                let mut expected = 0.0;
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    for r in 0..k {
                        for s in 0..k {
                            expected += tau[i * k + r] * tau[j * k + s] * b_matrix[r * k + s];
                        }
                    }
                }
                theta[i] = if expected > 1e-15 {
                    (actual_deg / expected).max(1e-10)
                } else {
                    1.0
                };
            }
        }
    }

    /// E-step: update posterior block probabilities tau.
    fn e_step(
        &self,
        _graph: &AdjacencyGraph,
        adj_matrix: &[f64],
        b_matrix: &[f64],
        theta: &[f64],
        tau: &mut [f64],
        n: usize,
        k: usize,
    ) {
        // pi_r = proportion of nodes in block r.
        let mut pi = vec![0.0_f64; k];
        for i in 0..n {
            for r in 0..k {
                pi[r] += tau[i * k + r];
            }
        }
        let pi_sum: f64 = pi.iter().sum();
        if pi_sum > 0.0 {
            for r in 0..k {
                pi[r] = (pi[r] / pi_sum).max(1e-10);
            }
        }

        for i in 0..n {
            let mut log_probs = vec![0.0_f64; k];
            for r in 0..k {
                log_probs[r] = pi[r].ln();

                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    // Use the current hard assignment of j approximation
                    // for efficiency, or sum over s.
                    for s in 0..k {
                        let tau_js = tau[j * k + s];
                        if tau_js < 1e-15 {
                            continue;
                        }

                        let mut p_rs = b_matrix[r * k + s];
                        if self.config.degree_corrected {
                            p_rs *= theta[i] * theta[j];
                        }
                        p_rs = p_rs.clamp(1e-15, 1.0 - 1e-15);

                        let a_ij = adj_matrix[i * n + j];
                        if a_ij > 0.0 {
                            log_probs[r] += tau_js * (a_ij * p_rs.ln());
                        } else {
                            log_probs[r] += tau_js * ((1.0 - p_rs).ln());
                        }
                    }
                }
            }

            // Log-sum-exp normalisation.
            let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_exp = 0.0;
            for r in 0..k {
                log_probs[r] = (log_probs[r] - max_lp).exp();
                sum_exp += log_probs[r];
            }
            if sum_exp > 0.0 {
                for r in 0..k {
                    tau[i * k + r] = (log_probs[r] / sum_exp).max(1e-15);
                }
            }
        }
    }

    /// Compute the log-likelihood of the model.
    fn log_likelihood(
        &self,
        adj_matrix: &[f64],
        b_matrix: &[f64],
        theta: &[f64],
        tau: &[f64],
        n: usize,
        k: usize,
    ) -> f64 {
        let mut ll = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let a_ij = adj_matrix[i * n + j];
                for r in 0..k {
                    let tau_ir = tau[i * k + r];
                    if tau_ir < 1e-15 {
                        continue;
                    }
                    for s in 0..k {
                        let tau_js = tau[j * k + s];
                        if tau_js < 1e-15 {
                            continue;
                        }
                        let mut p = b_matrix[r * k + s];
                        if self.config.degree_corrected {
                            p *= theta[i] * theta[j];
                        }
                        p = p.clamp(1e-15, 1.0 - 1e-15);

                        if a_ij > 0.0 {
                            ll += tau_ir * tau_js * a_ij * p.ln();
                        } else {
                            ll += tau_ir * tau_js * (1.0 - p).ln();
                        }
                    }
                }
            }
        }
        ll
    }

    /// Compute the Integrated Classification Likelihood (ICL) score.
    ///
    /// ICL = LL - penalty
    /// penalty = (K*(K+1)/2) * ln(n*(n-1)/2) / 2 + (K-1) * ln(n) / 2
    fn compute_icl(&self, ll: f64, labels: &[usize], n: usize, k: usize) -> f64 {
        let n_f = n as f64;
        let k_f = k as f64;
        // Number of B matrix parameters.
        let n_b_params = k_f * (k_f + 1.0) / 2.0;
        // Number of possible edges.
        let n_pairs = n_f * (n_f - 1.0) / 2.0;
        let penalty =
            n_b_params * n_pairs.max(1.0).ln() / 2.0 + (k_f - 1.0) * n_f.max(1.0).ln() / 2.0;

        // Entropy of tau (classification entropy).
        // Since we use hard labels here, the classification entropy contribution
        // is captured by the block sizes.
        let mut block_sizes = vec![0usize; k];
        for &l in labels {
            if l < k {
                block_sizes[l] += 1;
            }
        }
        let entropy_correction: f64 = block_sizes
            .iter()
            .filter(|&&s| s > 0)
            .map(|&s| {
                let p = s as f64 / n_f;
                -(s as f64) * p.ln()
            })
            .sum();

        ll - penalty - entropy_correction
    }

    /// Predict block assignments for a given graph using a fitted model.
    ///
    /// This re-runs the E-step with the given B matrix to assign labels.
    pub fn predict(
        &self,
        graph: &AdjacencyGraph,
        b_matrix: &[f64],
        k: usize,
    ) -> Result<Vec<usize>> {
        let n = graph.n_nodes;
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "Graph has zero nodes".to_string(),
            ));
        }
        if b_matrix.len() != k * k {
            return Err(ClusteringError::InvalidInput(
                "B matrix size mismatch".to_string(),
            ));
        }

        // Build dense adjacency.
        let mut adj_matrix = vec![0.0_f64; n * n];
        for i in 0..n {
            for &(j, w) in &graph.adjacency[i] {
                adj_matrix[i * n + j] = w;
            }
        }

        // Uniform initialisation.
        let uniform = 1.0 / k as f64;
        let mut tau = vec![uniform; n * k];
        let theta = vec![1.0_f64; n];

        for _iter in 0..self.config.max_iterations {
            self.e_step(graph, &adj_matrix, b_matrix, &theta, &mut tau, n, k);
        }

        // Hard assignment.
        let mut labels = vec![0usize; n];
        for i in 0..n {
            let mut best_r = 0;
            let mut best_val = f64::NEG_INFINITY;
            for r in 0..k {
                if tau[i * k + r] > best_val {
                    best_val = tau[i * k + r];
                    best_r = r;
                }
            }
            labels[i] = best_r;
        }

        Ok(labels)
    }

    /// Generate a random graph from SBM parameters.
    ///
    /// - `n`: number of nodes
    /// - `k`: number of blocks
    /// - `b_matrix`: K x K probability matrix (row-major)
    /// - `block_sizes`: sizes of each block (must sum to n)
    pub fn generate(
        n: usize,
        k: usize,
        b_matrix: &[f64],
        block_sizes: &[usize],
        seed: u64,
    ) -> Result<(AdjacencyGraph, Vec<usize>)> {
        if b_matrix.len() != k * k {
            return Err(ClusteringError::InvalidInput(
                "B matrix size must be k*k".to_string(),
            ));
        }
        if block_sizes.len() != k {
            return Err(ClusteringError::InvalidInput(
                "block_sizes length must equal k".to_string(),
            ));
        }
        let total: usize = block_sizes.iter().sum();
        if total != n {
            return Err(ClusteringError::InvalidInput(format!(
                "block_sizes sum ({}) must equal n ({})",
                total, n
            )));
        }

        let mut rng = Xorshift64::new(seed);

        // Assign nodes to blocks.
        let mut labels = Vec::with_capacity(n);
        for (block, &size) in block_sizes.iter().enumerate() {
            for _ in 0..size {
                labels.push(block);
            }
        }

        // Generate edges.
        let mut graph = AdjacencyGraph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let r = labels[i];
                let s = labels[j];
                let p = b_matrix[r * k + s];
                if rng.next_f64() < p {
                    let _ = graph.add_edge(i, j, 1.0);
                }
            }
        }

        Ok((graph, labels))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a planted-partition graph via SBM and fit -> recover blocks.
    #[test]
    fn test_sbm_generate_and_fit() {
        let k = 2;
        let n = 20;
        // High intra-block probability, low inter-block.
        let b_matrix = vec![0.8, 0.05, 0.05, 0.8];
        let block_sizes = vec![10, 10];
        let (graph, true_labels) =
            StochasticBlockModel::generate(n, k, &b_matrix, &block_sizes, 123)
                .expect("generate should succeed");

        let config = StochasticBlockModelConfig {
            num_blocks: Some(2),
            max_iterations: 50,
            seed: 42,
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        let result = sbm.fit(&graph).expect("fit should succeed");

        assert_eq!(result.community.num_communities, 2);
        assert_eq!(result.community.labels.len(), n);

        // Check accuracy: at least 70% of nodes should be correctly assigned
        // (up to label permutation).
        let accuracy = compute_accuracy(&true_labels, &result.community.labels, k);
        assert!(accuracy >= 0.7, "Accuracy {} is too low", accuracy);
    }

    /// Degree-corrected SBM should handle heterogeneous degrees.
    #[test]
    fn test_sbm_degree_corrected() {
        let k = 2;
        let n = 20;
        let b_matrix = vec![0.7, 0.1, 0.1, 0.7];
        let block_sizes = vec![10, 10];
        let (graph, _) = StochasticBlockModel::generate(n, k, &b_matrix, &block_sizes, 456)
            .expect("generate should succeed");

        let config = StochasticBlockModelConfig {
            num_blocks: Some(2),
            degree_corrected: true,
            max_iterations: 30,
            seed: 789,
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        let result = sbm.fit(&graph).expect("fit should succeed");

        assert!(result.degree_corrections.is_some());
        let dc = result
            .degree_corrections
            .as_ref()
            .expect("should have degree corrections");
        assert_eq!(dc.len(), n);
        // All corrections should be positive.
        for &d in dc {
            assert!(d > 0.0);
        }
    }

    /// Model selection should pick roughly the right K.
    #[test]
    fn test_sbm_model_selection() {
        let k = 2;
        let n = 30;
        let b_matrix = vec![0.9, 0.05, 0.05, 0.9];
        let block_sizes = vec![15, 15];
        let (graph, _) = StochasticBlockModel::generate(n, k, &b_matrix, &block_sizes, 111)
            .expect("generate should succeed");

        let config = StochasticBlockModelConfig {
            num_blocks: None,
            k_range: (2, 5),
            max_iterations: 30,
            seed: 222,
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        let result = sbm.fit(&graph).expect("fit should succeed");

        // The selected K should be 2 or 3 (small range is fine).
        assert!(
            result.k >= 2 && result.k <= 3,
            "Selected K={} seems wrong",
            result.k
        );
    }

    /// Predict with a known B matrix.
    #[test]
    fn test_sbm_predict() {
        let k = 2;
        let n = 20;
        let b_matrix = vec![0.8, 0.05, 0.05, 0.8];
        let block_sizes = vec![10, 10];
        let (graph, true_labels) =
            StochasticBlockModel::generate(n, k, &b_matrix, &block_sizes, 333)
                .expect("generate should succeed");

        let config = StochasticBlockModelConfig {
            max_iterations: 30,
            seed: 444,
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        let predicted = sbm
            .predict(&graph, &b_matrix, k)
            .expect("predict should succeed");

        assert_eq!(predicted.len(), n);
        let accuracy = compute_accuracy(&true_labels, &predicted, k);
        assert!(accuracy >= 0.6, "Predict accuracy {} is too low", accuracy);
    }

    /// Generate with invalid parameters should error.
    #[test]
    fn test_sbm_generate_invalid() {
        // block_sizes don't sum to n.
        let result = StochasticBlockModel::generate(10, 2, &[0.5, 0.1, 0.1, 0.5], &[4, 4], 0);
        assert!(result.is_err());
    }

    /// Empty graph should error.
    #[test]
    fn test_sbm_empty_graph() {
        let g = AdjacencyGraph::new(0);
        let config = StochasticBlockModelConfig {
            num_blocks: Some(2),
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        assert!(sbm.fit(&g).is_err());
    }

    /// Single block.
    #[test]
    fn test_sbm_single_block() {
        let n = 10;
        let mut g = AdjacencyGraph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        let config = StochasticBlockModelConfig {
            num_blocks: Some(1),
            max_iterations: 20,
            seed: 555,
            ..Default::default()
        };
        let sbm = StochasticBlockModel::new(config);
        let result = sbm.fit(&g).expect("fit should succeed");
        assert_eq!(result.community.num_communities, 1);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Compute assignment accuracy accounting for label permutations.
    fn compute_accuracy(true_labels: &[usize], pred_labels: &[usize], k: usize) -> f64 {
        let n = true_labels.len();
        if n == 0 {
            return 1.0;
        }

        // Try all permutations (only feasible for small k).
        // For k <= 8 this is fine.
        let perms = generate_permutations(k);
        let mut best_correct = 0usize;
        for perm in &perms {
            let correct = (0..n)
                .filter(|&i| {
                    let mapped = if pred_labels[i] < perm.len() {
                        perm[pred_labels[i]]
                    } else {
                        pred_labels[i]
                    };
                    mapped == true_labels[i]
                })
                .count();
            if correct > best_correct {
                best_correct = correct;
            }
        }
        best_correct as f64 / n as f64
    }

    fn generate_permutations(k: usize) -> Vec<Vec<usize>> {
        if k == 0 {
            return vec![vec![]];
        }
        if k == 1 {
            return vec![vec![0]];
        }
        let mut result = Vec::new();
        let sub = generate_permutations(k - 1);
        for perm in sub {
            for pos in 0..k {
                let mut new_perm = perm.clone();
                new_perm.insert(pos, k - 1);
                result.push(new_perm);
            }
        }
        result
    }
}
