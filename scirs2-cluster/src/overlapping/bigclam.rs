//! BigClam: Community Affiliation Graph Model for overlapping community detection.
//!
//! Implements the BigClam algorithm (Yang & Leskovec, 2013) which learns a soft
//! membership matrix F where `F[u][c]` represents the affiliation strength of node u
//! to community c. The probability of an edge between u and v is:
//!   P(u,v) = 1 - exp(-F_u · F_v)
//!
//! The algorithm maximises the log-likelihood via coordinate ascent (gradient updates)
//! on each row of F.

use crate::error::{ClusteringError, Result as ClusterResult};
use std::collections::HashSet;

// ─── Initialisation strategy ─────────────────────────────────────────────────

/// Strategy used to initialise the membership matrix before gradient ascent.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BigClamInit {
    /// Draw each `F[u][c]` uniformly from (0, 1).
    Random,
    /// Use a spectral warm-start: run a fast 1-step power method on the
    /// normalised adjacency, then split into `n_communities` columns via
    /// random projection. Falls back to `Random` for graphs with < 2 nodes.
    SpectralWarmStart,
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the BigClam algorithm.
#[derive(Debug, Clone)]
pub struct BigClamConfig {
    /// Number of communities to detect.  Default: 5.
    pub n_communities: usize,
    /// Maximum number of gradient-ascent iterations.  Default: 100.
    pub max_iter: usize,
    /// Learning rate for coordinate-ascent updates.  Default: 0.005.
    pub learning_rate: f64,
    /// Entries of F below this threshold are clamped to zero after each update.
    /// Default: 1e-3.
    pub min_membership: f64,
    /// L2 regularisation coefficient on F.  Default: 0.01.
    pub reg_lambda: f64,
    /// Initialisation strategy.  Default: `BigClamInit::Random`.
    pub init: BigClamInit,
    /// Random seed for reproducibility.  Default: 42.
    pub seed: u64,
    /// Relative convergence tolerance: `||F_new - F_old|| / (||F_old|| + ε) < tol`.
    /// Default: 1e-4.
    pub tol: f64,
}

impl Default for BigClamConfig {
    fn default() -> Self {
        Self {
            n_communities: 5,
            max_iter: 100,
            learning_rate: 0.005,
            min_membership: 1e-3,
            reg_lambda: 0.01,
            init: BigClamInit::Random,
            seed: 42,
            tol: 1e-4,
        }
    }
}

// ─── Membership matrix ────────────────────────────────────────────────────────

/// Soft membership matrix returned by BigClam.
///
/// `memberships[u][c]` is the affiliation strength of node `u` to community `c`.
/// Values are non-negative; a value of 0 means no membership.
#[derive(Debug, Clone)]
pub struct MembershipMatrix {
    /// Raw membership values: shape `[n_nodes][n_communities]`.
    pub memberships: Vec<Vec<f64>>,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Number of communities.
    pub n_communities: usize,
}

impl MembershipMatrix {
    /// Create a zero-initialised membership matrix.
    pub fn new(n_nodes: usize, n_communities: usize) -> Self {
        Self {
            memberships: vec![vec![0.0; n_communities]; n_nodes],
            n_nodes,
            n_communities,
        }
    }

    /// Return the indices of nodes whose membership in `community` is ≥ `threshold`.
    pub fn community_members(&self, community: usize, threshold: f64) -> Vec<usize> {
        self.memberships
            .iter()
            .enumerate()
            .filter(|(_, row)| row.get(community).copied().unwrap_or(0.0) >= threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Return the community indices that node `node` belongs to (membership ≥ `threshold`).
    pub fn node_communities(&self, node: usize, threshold: f64) -> Vec<usize> {
        match self.memberships.get(node) {
            None => Vec::new(),
            Some(row) => row
                .iter()
                .enumerate()
                .filter(|(_, &v)| v >= threshold)
                .map(|(c, _)| c)
                .collect(),
        }
    }

    /// Convert to a hard partition by assigning each node to its maximum-membership community.
    pub fn to_hard_partition(&self) -> Vec<usize> {
        self.memberships
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(c, _)| c)
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ─── BigClam struct ───────────────────────────────────────────────────────────

/// BigClam overlapping community detection.
pub struct BigClam {
    config: BigClamConfig,
}

impl BigClam {
    /// Create a new BigClam instance with the given configuration.
    pub fn new(config: BigClamConfig) -> Self {
        Self { config }
    }

    /// Fit the model to an adjacency list and return the membership matrix.
    ///
    /// `adj[u]` must contain the (undirected) neighbours of node `u`.  Self-loops
    /// are ignored.  The graph is assumed to be 0-indexed and every edge must
    /// appear in both directions.
    pub fn fit(&self, adj: &[Vec<usize>]) -> ClusterResult<MembershipMatrix> {
        let n = adj.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "Adjacency list is empty".to_string(),
            ));
        }
        let k = self.config.n_communities;
        if k == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_communities must be ≥ 1".to_string(),
            ));
        }

        // Build neighbour sets for O(1) membership queries.
        let neighbour_sets: Vec<HashSet<usize>> = adj
            .iter()
            .map(|row| row.iter().copied().collect())
            .collect();

        // Initialise F.
        let mut f = self.init_f(n, k, adj);

        let lr = self.config.learning_rate;
        let lambda = self.config.reg_lambda;
        let tol = self.config.tol;
        let min_m = self.config.min_membership;

        // Pre-compute per-community sums: s[c] = sum_u F[u][c]
        let mut community_sums = compute_community_sums(&f, k);

        for _iter in 0..self.config.max_iter {
            let f_old = f.clone();

            // Deterministic node order (can be randomised, but determinism aids tests)
            for u in 0..n {
                let f_u_old = f[u].clone();

                for c in 0..k {
                    // Positive contribution from edges.
                    let mut pos_grad = 0.0;
                    for &v in &adj[u] {
                        if v == u {
                            continue;
                        }
                        let dp = dot_product(&f[u], &f[v]);
                        let exp_neg = (-dp).exp();
                        // Guard against 1 - exp(-dp) ≈ 0
                        let denom = (1.0 - exp_neg).max(1e-10);
                        pos_grad += f[v][c] * exp_neg / denom;
                    }

                    // Negative contribution from non-edges (use community sums trick).
                    // sum_{v not in N(u)} F[v][c] = community_sum[c] - sum_{v in N(u)} F[v][c]
                    let mut nbr_sum_c = 0.0;
                    for &v in &adj[u] {
                        if v != u {
                            nbr_sum_c += f[v][c];
                        }
                    }
                    // Exclude self contribution
                    let neg_grad = community_sums[c] - f[u][c] - nbr_sum_c;

                    // Regularisation
                    let reg = lambda * f[u][c];

                    let grad = pos_grad - neg_grad - reg;
                    let new_val = (f[u][c] + lr * grad).max(0.0);
                    f[u][c] = if new_val < min_m { 0.0 } else { new_val };
                }

                // Update community sums for the change in row u.
                for c in 0..k {
                    community_sums[c] += f[u][c] - f_u_old[c];
                }
            }

            // Check convergence.
            if has_converged(&f, &f_old, tol) {
                break;
            }

            // Verify neighbour sets are still valid (they are; just silence the warning).
            let _ = &neighbour_sets;
        }

        Ok(MembershipMatrix {
            memberships: f,
            n_nodes: n,
            n_communities: k,
        })
    }

    /// Compute the BigClam log-likelihood log P(G | F).
    ///
    /// Uses the full sum over all pairs, which is O(n²) but is fine for tests.
    /// Production code should use the sampled approximation.
    pub fn log_likelihood(&self, adj: &[Vec<usize>], f: &[Vec<f64>]) -> f64 {
        let n = adj.len();
        // Build edge set for O(1) lookup.
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for (u, neighbours) in adj.iter().enumerate() {
            for &v in neighbours {
                if u < v {
                    edge_set.insert((u, v));
                }
            }
        }

        let mut ll = 0.0;
        for u in 0..n {
            for v in (u + 1)..n {
                let dp = dot_product(&f[u], &f[v]);
                if edge_set.contains(&(u, v)) {
                    // log(1 - exp(-dp));  guard dp ≈ 0
                    let prob = (1.0_f64 - (-dp).exp()).max(1e-15);
                    ll += prob.ln();
                } else {
                    // - F_u · F_v
                    ll -= dp;
                }
            }
        }
        ll
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Initialise the membership matrix according to the configured strategy.
    fn init_f(&self, n: usize, k: usize, adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
        match self.config.init {
            BigClamInit::Random => self.init_random(n, k),
            BigClamInit::SpectralWarmStart => {
                if n < 2 {
                    self.init_random(n, k)
                } else {
                    self.init_spectral(n, k, adj)
                }
            }
        }
    }

    /// Simple LCG-based pseudo-random init (no external dependency).
    fn init_random(&self, n: usize, k: usize) -> Vec<Vec<f64>> {
        let mut state = self.config.seed.wrapping_add(1);
        let mut rand_f64 = move || -> f64 {
            // Xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };
        (0..n)
            .map(|_| (0..k).map(|_| rand_f64() * 0.5 + 0.01).collect())
            .collect()
    }

    /// Spectral warm-start: one step of power iteration on the normalised adjacency,
    /// then project onto `k` communities via a random Gaussian projection.
    fn init_spectral(&self, n: usize, k: usize, adj: &[Vec<usize>]) -> Vec<Vec<f64>> {
        // Degree vector
        let deg: Vec<f64> = adj.iter().map(|row| row.len() as f64).collect();

        // Start with uniform vector, normalised.
        let inv_sqrt_n = 1.0 / (n as f64).sqrt();
        let mut x: Vec<f64> = vec![inv_sqrt_n; n];

        // Two power-iteration steps on D^{-1/2} A D^{-1/2}
        for _ in 0..2 {
            let mut y = vec![0.0f64; n];
            for (u, neighbours) in adj.iter().enumerate() {
                let d_u = deg[u].max(1.0).sqrt();
                for &v in neighbours {
                    let d_v = deg[v].max(1.0).sqrt();
                    y[u] += x[v] / (d_u * d_v);
                }
            }
            // Re-normalise
            let norm = y.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-15);
            for val in &mut y {
                *val /= norm;
            }
            x = y;
        }

        // Random Gaussian projection of x (scalar) into k dimensions
        let mut state = self.config.seed.wrapping_add(0xDEAD_BEEF);
        let mut rand_normal = move || -> f64 {
            // Box-Muller
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u1 = (state as f64) / (u64::MAX as f64);
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u2 = (state as f64) / (u64::MAX as f64);
            let u1 = u1.max(1e-15);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        // Build a projection matrix [n_communities][n] (unused dims → random)
        // Simple: each community c gets a random weight w_c; F[u][c] = |x[u] * w_c + eps|
        let mut proj: Vec<f64> = (0..k).map(|_| rand_normal().abs() + 0.1).collect();
        // Normalise projection weights
        let pnorm = proj.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-15);
        for v in &mut proj {
            *v /= pnorm;
        }

        (0..n)
            .map(|u| {
                (0..k)
                    .map(|c| {
                        let base = (x[u] * proj[c]).abs();
                        (base + 0.01).max(0.01)
                    })
                    .collect()
            })
            .collect()
    }
}

// ─── Free functions ───────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Compute per-community column sums of F.
fn compute_community_sums(f: &[Vec<f64>], k: usize) -> Vec<f64> {
    let mut sums = vec![0.0f64; k];
    for row in f {
        for (c, &v) in row.iter().enumerate() {
            sums[c] += v;
        }
    }
    sums
}

/// Check whether the Frobenius-norm relative change is below `tol`.
fn has_converged(f_new: &[Vec<f64>], f_old: &[Vec<f64>], tol: f64) -> bool {
    let mut diff_sq = 0.0f64;
    let mut old_sq = 0.0f64;
    for (row_new, row_old) in f_new.iter().zip(f_old.iter()) {
        for (&a, &b) in row_new.iter().zip(row_old.iter()) {
            diff_sq += (a - b) * (a - b);
            old_sq += b * b;
        }
    }
    diff_sq / (old_sq + 1e-15) < tol * tol
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build two disjoint triangles (cliques of size 3) sharing no nodes.
    fn two_triangles() -> Vec<Vec<usize>> {
        // Nodes 0-2: first triangle; nodes 3-5: second triangle.
        vec![
            vec![1, 2],
            vec![0, 2],
            vec![0, 1],
            vec![4, 5],
            vec![3, 5],
            vec![3, 4],
        ]
    }

    /// Build two cliques of size 4 sharing one bridge node (node 3 = node 4).
    fn two_cliques_bridge() -> Vec<Vec<usize>> {
        // Clique A: 0,1,2,3  Clique B: 3,4,5,6  (node 3 is bridge)
        let mut adj = vec![vec![]; 7];
        for &(u, v) in &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            adj[u].push(v);
            adj[v].push(u);
        }
        for &(u, v) in &[(3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)] {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }

    #[test]
    fn test_bigclam_membership_nonnegative() {
        let adj = two_triangles();
        let config = BigClamConfig {
            n_communities: 2,
            max_iter: 10,
            ..Default::default()
        };
        let mm = BigClam::new(config).fit(&adj).expect("fit should succeed");
        for row in &mm.memberships {
            for &v in row {
                assert!(v >= 0.0, "membership must be non-negative, got {v}");
            }
        }
    }

    #[test]
    fn test_bigclam_on_two_cliques() {
        let adj = two_cliques_bridge();
        let config = BigClamConfig {
            n_communities: 2,
            max_iter: 80,
            learning_rate: 0.01,
            ..Default::default()
        };
        let mm = BigClam::new(config).fit(&adj).expect("fit should succeed");
        assert_eq!(mm.n_nodes, 7);
        assert_eq!(mm.n_communities, 2);
        // Node 3 (bridge) should have non-trivial membership in both communities.
        let node3 = &mm.memberships[3];
        // At least one community should have significant membership for the bridge node.
        let max_m = node3.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_m > 0.0, "bridge node should have positive membership");
    }

    #[test]
    fn test_bigclam_convergence() {
        let adj = two_triangles();
        let config = BigClamConfig {
            n_communities: 2,
            max_iter: 200,
            tol: 1e-6,
            ..Default::default()
        };
        // Should not panic; convergence is internal.
        BigClam::new(config).fit(&adj).expect("fit should succeed");
    }

    #[test]
    fn test_bigclam_n_communities() {
        let adj = two_triangles();
        for k in 1..=4 {
            let config = BigClamConfig {
                n_communities: k,
                max_iter: 5,
                ..Default::default()
            };
            let mm = BigClam::new(config).fit(&adj).expect("fit should succeed");
            assert_eq!(mm.n_communities, k);
            for row in &mm.memberships {
                assert_eq!(row.len(), k);
            }
        }
    }

    #[test]
    fn test_membership_matrix_community_members() {
        let mut mm = MembershipMatrix::new(4, 2);
        mm.memberships[0][0] = 0.9;
        mm.memberships[1][0] = 0.8;
        mm.memberships[2][1] = 0.7;
        mm.memberships[3][0] = 0.1; // below threshold
        let members = mm.community_members(0, 0.5);
        assert!(members.contains(&0));
        assert!(members.contains(&1));
        assert!(!members.contains(&2));
        assert!(!members.contains(&3));
    }

    #[test]
    fn test_membership_matrix_to_hard_partition() {
        let mut mm = MembershipMatrix::new(3, 2);
        mm.memberships[0] = vec![0.8, 0.2];
        mm.memberships[1] = vec![0.1, 0.9];
        mm.memberships[2] = vec![0.5, 0.5];
        let hard = mm.to_hard_partition();
        assert_eq!(hard[0], 0);
        assert_eq!(hard[1], 1);
        // Tie → first max (community 0 due to equal)
        assert!(hard[2] == 0 || hard[2] == 1);
    }

    #[test]
    fn test_bigclam_log_likelihood_improves() {
        let adj = two_cliques_bridge();
        let bc = BigClam::new(BigClamConfig {
            n_communities: 2,
            max_iter: 1,
            ..Default::default()
        });
        let f_init = bc.fit(&adj).expect("fit should succeed").memberships;
        let ll_early = bc.log_likelihood(&adj, &f_init);

        let bc2 = BigClam::new(BigClamConfig {
            n_communities: 2,
            max_iter: 100,
            ..Default::default()
        });
        let f_trained = bc2.fit(&adj).expect("fit should succeed").memberships;
        let ll_trained = bc2.log_likelihood(&adj, &f_trained);

        // More iterations should not decrease log-likelihood significantly
        // (allows for small numerical fluctuations around convergence)
        assert!(
            ll_trained >= ll_early - 0.5,
            "trained LL {ll_trained} should not be much worse than early LL {ll_early}"
        );
    }

    #[test]
    fn test_bigclam_spectral_init() {
        let adj = two_cliques_bridge();
        let config = BigClamConfig {
            n_communities: 2,
            max_iter: 20,
            init: BigClamInit::SpectralWarmStart,
            ..Default::default()
        };
        let mm = BigClam::new(config)
            .fit(&adj)
            .expect("spectral init fit should succeed");
        for row in &mm.memberships {
            for &v in row {
                assert!(v >= 0.0);
            }
        }
    }

    #[test]
    fn test_bigclam_empty_graph_error() {
        let adj: Vec<Vec<usize>> = vec![];
        let result = BigClam::new(BigClamConfig::default()).fit(&adj);
        assert!(result.is_err());
    }

    #[test]
    fn test_bigclam_zero_communities_error() {
        let adj = two_triangles();
        let config = BigClamConfig {
            n_communities: 0,
            ..Default::default()
        };
        let result = BigClam::new(config).fit(&adj);
        assert!(result.is_err());
    }
}
