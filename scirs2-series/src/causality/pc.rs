//! PC Algorithm for Causal Discovery
//!
//! Implements the classic PC (Peter-Clark) algorithm for discovering causal
//! structure from observational data. Unlike [`super::pc_stable`] which is
//! adapted for time series with lagged variables, this module implements the
//! standard cross-sectional PC algorithm.
//!
//! ## Algorithm Phases
//!
//! 1. **Skeleton** (Phase 1): Start with a complete undirected graph. For each
//!    pair (X, Y), test conditional independence X _||_ Y | S for increasing
//!    sizes of conditioning set S drawn from the adjacency set. Remove the edge
//!    if independent.
//!
//! 2. **V-structures** (Phase 2): For each unshielded triple X - Z - Y (i.e.
//!    X and Y not adjacent, both adjacent to Z), orient as X -> Z <- Y if Z
//!    was *not* in the separation set of (X, Y).
//!
//! 3. **Meek rules** (Phase 3): Apply Meek's orientation propagation rules
//!    to infer additional edge directions without creating new v-structures
//!    or directed cycles.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use scirs2_series::causality::pc::{PCAlgorithm, PCConfig, IndependenceTest};
//!
//! let data: Vec<Vec<f64>> = vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![2.0, 4.1, 5.9],
//!     // ... more observations
//! ];
//! let config = PCConfig::default();
//! let pc = PCAlgorithm::new(config);
//! let graph = pc.discover(&data).expect("discovery failed");
//! println!("Edges: {:?}", graph.edges);
//! ```

use super::CausalityResult;
use crate::error::TimeSeriesError;

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for the PC algorithm.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PCConfig {
    /// Significance level for conditional independence tests.
    pub significance_level: f64,
    /// Maximum conditioning set size to consider.
    /// The algorithm stops increasing the conditioning set size once it exceeds
    /// this value or when no edge is testable.
    pub max_cond_set_size: usize,
    /// Type of independence test to use.
    pub test_type: IndependenceTest,
}

impl Default for PCConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            max_cond_set_size: 4,
            test_type: IndependenceTest::PartialCorrelation,
        }
    }
}

/// Type of conditional independence test.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndependenceTest {
    /// Partial correlation test with Fisher's z-transform.
    PartialCorrelation,
    /// Mutual information test (Gaussian approximation).
    MutualInformation,
    /// Kernel-based independence test (HSIC-like, simplified).
    KernelBased,
}

/// Type of an edge in the causal graph.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Directed edge (arrow from `from` to `to`).
    Directed,
    /// Undirected edge.
    Undirected,
    /// Bidirected edge (latent common cause).
    Bidirected,
}

/// A single edge in the causal graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CausalEdge {
    /// Source node index.
    pub from: usize,
    /// Target node index.
    pub to: usize,
    /// Type of edge.
    pub edge_type: EdgeType,
}

/// Result of causal discovery: the estimated causal graph.
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Number of nodes (variables).
    pub nodes: usize,
    /// Discovered edges.
    pub edges: Vec<CausalEdge>,
    /// Separation sets for pairs that were separated.
    /// Key: (i, j) with i < j; Value: conditioning set that separated them.
    pub separation_sets: HashMap<(usize, usize), Vec<usize>>,
}

impl CausalGraph {
    /// Check if there is a directed edge from `from` to `to`.
    pub fn has_directed_edge(&self, from: usize, to: usize) -> bool {
        self.edges
            .iter()
            .any(|e| e.from == from && e.to == to && e.edge_type == EdgeType::Directed)
    }

    /// Check if there is any edge (directed or undirected) between `a` and `b`.
    pub fn has_edge(&self, a: usize, b: usize) -> bool {
        self.edges
            .iter()
            .any(|e| (e.from == a && e.to == b) || (e.from == b && e.to == a))
    }

    /// Count edges of a given type.
    pub fn count_edges(&self, edge_type: EdgeType) -> usize {
        self.edges
            .iter()
            .filter(|e| e.edge_type == edge_type)
            .count()
    }
}

// ---------------------------------------------------------------------------
// PC Algorithm
// ---------------------------------------------------------------------------

/// The PC algorithm for causal discovery from observational data.
#[derive(Debug, Clone)]
pub struct PCAlgorithm {
    config: PCConfig,
}

impl PCAlgorithm {
    /// Create a new PC algorithm instance.
    pub fn new(config: PCConfig) -> Self {
        Self { config }
    }

    /// Run causal discovery on cross-sectional data.
    ///
    /// # Arguments
    /// * `data` - Observations as a vector of samples, each sample is a vector
    ///   of variable values. All samples must have the same length.
    ///
    /// # Returns
    /// A [`CausalGraph`] with the discovered causal structure.
    pub fn discover(&self, data: &[Vec<f64>]) -> CausalityResult<CausalGraph> {
        let n_samples = data.len();
        if n_samples < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Need at least 4 samples for PC algorithm".to_string(),
                required: 4,
                actual: n_samples,
            });
        }

        let n_vars = data[0].len();
        if n_vars < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Need at least 2 variables for causal discovery".to_string(),
            ));
        }

        // Validate all samples have same length
        for (i, sample) in data.iter().enumerate() {
            if sample.len() != n_vars {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: n_vars,
                    actual: sample.len(),
                });
            }
            // Check for NaN / Inf
            for &v in sample {
                if !v.is_finite() {
                    return Err(TimeSeriesError::InvalidInput(format!(
                        "Non-finite value in sample {}",
                        i
                    )));
                }
            }
        }

        // Compute correlation/covariance matrix
        let cov_matrix = compute_covariance_matrix(data)?;

        // Phase 1: Skeleton discovery
        let (adjacency, separation_sets) =
            self.discover_skeleton(n_vars, n_samples, &cov_matrix)?;

        // Phase 2: Orient v-structures
        let mut edge_types = self.orient_v_structures(n_vars, &adjacency, &separation_sets);

        // Phase 3: Meek rules
        self.apply_meek_rules(n_vars, &adjacency, &mut edge_types);

        // Build the causal graph
        let mut edges = Vec::new();
        for i in 0..n_vars {
            for j in (i + 1)..n_vars {
                if adjacency[i].contains(&j) {
                    let key = (i, j);
                    let et = edge_types
                        .get(&key)
                        .copied()
                        .unwrap_or(EdgeType::Undirected);
                    match et {
                        EdgeType::Directed => {
                            // Check direction: was it i->j or j->i?
                            // We store the directed edge based on what was determined
                            if let Some(&dir) = edge_types.get(&(i, j)) {
                                if dir == EdgeType::Directed {
                                    edges.push(CausalEdge {
                                        from: i,
                                        to: j,
                                        edge_type: EdgeType::Directed,
                                    });
                                }
                            }
                        }
                        _ => {
                            edges.push(CausalEdge {
                                from: i,
                                to: j,
                                edge_type: et,
                            });
                        }
                    }
                }
            }
        }

        // Also add reverse directed edges that were stored with (j, i) key
        for (&(from, to), &et) in &edge_types {
            if et == EdgeType::Directed && from > to {
                // This is a j->i edge stored as (j, i) = Directed
                edges.push(CausalEdge {
                    from,
                    to,
                    edge_type: EdgeType::Directed,
                });
            }
        }

        // Deduplicate
        let mut seen = HashSet::new();
        let deduped: Vec<CausalEdge> = edges
            .into_iter()
            .filter(|e| {
                let key = (e.from, e.to, e.edge_type);
                seen.insert(key)
            })
            .collect();

        Ok(CausalGraph {
            nodes: n_vars,
            edges: deduped,
            separation_sets,
        })
    }

    // --- Phase 1: Skeleton ---

    fn discover_skeleton(
        &self,
        n_vars: usize,
        n_samples: usize,
        cov_matrix: &[Vec<f64>],
    ) -> CausalityResult<(Vec<HashSet<usize>>, HashMap<(usize, usize), Vec<usize>>)> {
        // Start with complete undirected graph
        let mut adjacency: Vec<HashSet<usize>> = (0..n_vars)
            .map(|i| (0..n_vars).filter(|&j| j != i).collect())
            .collect();

        let mut separation_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        let mut p = 0usize;
        loop {
            if p > self.config.max_cond_set_size {
                break;
            }

            let mut any_testable = false;
            let mut removals: Vec<(usize, usize, Vec<usize>)> = Vec::new();

            // Snapshot adjacency for stable iteration
            let adj_snapshot: Vec<Vec<usize>> = adjacency
                .iter()
                .map(|s| {
                    let mut v: Vec<usize> = s.iter().copied().collect();
                    v.sort();
                    v
                })
                .collect();

            for i in 0..n_vars {
                let neighbors_i = &adj_snapshot[i];
                for &j in neighbors_i {
                    if j <= i {
                        continue; // Only test each pair once
                    }

                    // Conditioning set candidates: neighbors of i, excluding j
                    let cond_candidates: Vec<usize> =
                        neighbors_i.iter().copied().filter(|&k| k != j).collect();

                    if cond_candidates.len() < p {
                        continue;
                    }
                    any_testable = true;

                    // Test all subsets of size p
                    let subsets = gen_combinations(&cond_candidates, p);
                    let mut found_independent = false;
                    let mut best_sep = Vec::new();

                    for subset in &subsets {
                        let p_value = self
                            .test_conditional_independence(i, j, subset, n_samples, cov_matrix)?;

                        if p_value > self.config.significance_level {
                            found_independent = true;
                            best_sep = subset.clone();
                            break;
                        }
                    }

                    if found_independent {
                        removals.push((i, j, best_sep));
                    }
                }
            }

            // Apply removals
            for (i, j, sep_set) in removals {
                adjacency[i].remove(&j);
                adjacency[j].remove(&i);
                let key = if i < j { (i, j) } else { (j, i) };
                separation_sets.insert(key, sep_set);
            }

            if !any_testable {
                break;
            }
            p += 1;
        }

        Ok((adjacency, separation_sets))
    }

    // --- Phase 2: V-structure orientation ---

    fn orient_v_structures(
        &self,
        n_vars: usize,
        adjacency: &[HashSet<usize>],
        separation_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) -> HashMap<(usize, usize), EdgeType> {
        let mut edge_types: HashMap<(usize, usize), EdgeType> = HashMap::new();

        // Initialize all remaining edges as undirected
        for i in 0..n_vars {
            for &j in &adjacency[i] {
                if j > i {
                    edge_types.insert((i, j), EdgeType::Undirected);
                }
            }
        }

        // For each unshielded triple X - Z - Y (X and Y not adjacent)
        for z in 0..n_vars {
            let neighbors_z: Vec<usize> = adjacency[z].iter().copied().collect();
            for idx_x in 0..neighbors_z.len() {
                for idx_y in (idx_x + 1)..neighbors_z.len() {
                    let x = neighbors_z[idx_x];
                    let y = neighbors_z[idx_y];

                    // Check if X and Y are NOT adjacent (unshielded)
                    if adjacency[x].contains(&y) {
                        continue;
                    }

                    // Look up separation set of (X, Y)
                    let key = if x < y { (x, y) } else { (y, x) };
                    let sep_set = separation_sets.get(&key);

                    // If Z is NOT in the separation set, orient as X -> Z <- Y
                    let z_in_sep = sep_set.map(|s| s.contains(&z)).unwrap_or(false);

                    if !z_in_sep {
                        // Orient X -> Z
                        edge_types.insert((x, z), EdgeType::Directed);
                        // Orient Y -> Z
                        edge_types.insert((y, z), EdgeType::Directed);
                        // Remove the undirected versions
                        let k1 = if x < z { (x, z) } else { (z, x) };
                        let k2 = if y < z { (y, z) } else { (z, y) };
                        edge_types.remove(&k1);
                        edge_types.remove(&k2);
                        edge_types.insert((x, z), EdgeType::Directed);
                        edge_types.insert((y, z), EdgeType::Directed);
                    }
                }
            }
        }

        edge_types
    }

    // --- Phase 3: Meek rules ---

    fn apply_meek_rules(
        &self,
        n_vars: usize,
        adjacency: &[HashSet<usize>],
        edge_types: &mut HashMap<(usize, usize), EdgeType>,
    ) {
        // Apply Meek's four rules iteratively until no more orientations change
        let max_iterations = n_vars * n_vars;
        for _ in 0..max_iterations {
            let mut changed = false;

            // Rule 1: If A -> B - C and A and C are not adjacent, orient B -> C
            for b in 0..n_vars {
                let neighbors_b: Vec<usize> = adjacency[b].iter().copied().collect();
                for &c in &neighbors_b {
                    // Check if B - C is undirected
                    if !is_undirected(edge_types, b, c) {
                        continue;
                    }

                    for &a in &neighbors_b {
                        if a == c {
                            continue;
                        }
                        // Check if A -> B (directed)
                        if !is_directed(edge_types, a, b) {
                            continue;
                        }
                        // Check if A and C are not adjacent
                        if adjacency[a].contains(&c) {
                            continue;
                        }

                        // Orient B -> C
                        orient_edge(edge_types, b, c);
                        changed = true;
                    }
                }
            }

            // Rule 2: If A -> C -> B and A - B, orient A -> B
            for a in 0..n_vars {
                let neighbors_a: Vec<usize> = adjacency[a].iter().copied().collect();
                for &b in &neighbors_a {
                    if !is_undirected(edge_types, a, b) {
                        continue;
                    }

                    // Look for C such that A -> C and C -> B
                    for &c in &neighbors_a {
                        if c == b {
                            continue;
                        }
                        if !is_directed(edge_types, a, c) {
                            continue;
                        }
                        if !adjacency[c].contains(&b) {
                            continue;
                        }
                        if !is_directed(edge_types, c, b) {
                            continue;
                        }

                        orient_edge(edge_types, a, b);
                        changed = true;
                    }
                }
            }

            // Rule 3: If A - C, A - D, C -> B, D -> B, and C and D are not adjacent,
            // orient A -> B
            for a in 0..n_vars {
                let neighbors_a: Vec<usize> = adjacency[a].iter().copied().collect();
                for &b in &neighbors_a {
                    if !is_undirected(edge_types, a, b) {
                        continue;
                    }

                    // Find two distinct C, D both adjacent to A (undirected) and both -> B
                    let mut oriented = false;
                    for idx_c in 0..neighbors_a.len() {
                        if oriented {
                            break;
                        }
                        let c = neighbors_a[idx_c];
                        if c == b {
                            continue;
                        }
                        if !is_undirected(edge_types, a, c) {
                            continue;
                        }
                        if !adjacency[c].contains(&b) || !is_directed(edge_types, c, b) {
                            continue;
                        }

                        for idx_d in (idx_c + 1)..neighbors_a.len() {
                            let d = neighbors_a[idx_d];
                            if d == b || d == c {
                                continue;
                            }
                            if !is_undirected(edge_types, a, d) {
                                continue;
                            }
                            if !adjacency[d].contains(&b) || !is_directed(edge_types, d, b) {
                                continue;
                            }
                            // C and D must not be adjacent
                            if adjacency[c].contains(&d) {
                                continue;
                            }

                            orient_edge(edge_types, a, b);
                            changed = true;
                            oriented = true;
                            break;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }

    // --- Conditional independence tests ---

    fn test_conditional_independence(
        &self,
        i: usize,
        j: usize,
        cond_set: &[usize],
        n_samples: usize,
        cov_matrix: &[Vec<f64>],
    ) -> CausalityResult<f64> {
        match self.config.test_type {
            IndependenceTest::PartialCorrelation => {
                partial_correlation_test(i, j, cond_set, n_samples, cov_matrix)
            }
            IndependenceTest::MutualInformation => {
                mutual_information_test(i, j, cond_set, n_samples, cov_matrix)
            }
            IndependenceTest::KernelBased => {
                // Simplified: falls back to partial correlation
                // A full kernel-based test (HSIC) would require access to raw data
                partial_correlation_test(i, j, cond_set, n_samples, cov_matrix)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical tests
// ---------------------------------------------------------------------------

/// Partial correlation test using Fisher's z-transform.
///
/// Computes the partial correlation between variables `i` and `j` conditioning
/// on `cond_set`, then applies Fisher's z-transform to compute a p-value.
fn partial_correlation_test(
    i: usize,
    j: usize,
    cond_set: &[usize],
    n_samples: usize,
    cov_matrix: &[Vec<f64>],
) -> CausalityResult<f64> {
    let parcorr = compute_partial_corr(i, j, cond_set, cov_matrix)?;

    // Fisher's z-transform
    let df = n_samples as f64 - cond_set.len() as f64 - 2.0;
    if df < 1.0 {
        return Ok(1.0); // Not enough degrees of freedom
    }

    let clamped = parcorr.clamp(-0.9999, 0.9999);
    let z_stat = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln() * df.sqrt();

    // Two-sided p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
    Ok(p_value)
}

/// Mutual information test (Gaussian approximation).
///
/// Under Gaussianity: MI(X;Y|Z) = -0.5 * ln(1 - parcorr^2)
/// Test statistic: 2 * n * MI ~ chi2(1) under H0
fn mutual_information_test(
    i: usize,
    j: usize,
    cond_set: &[usize],
    n_samples: usize,
    cov_matrix: &[Vec<f64>],
) -> CausalityResult<f64> {
    let parcorr = compute_partial_corr(i, j, cond_set, cov_matrix)?;

    let r_sq = parcorr * parcorr;
    let mi = if r_sq < 1.0 {
        -0.5 * (1.0 - r_sq).ln()
    } else {
        f64::INFINITY
    };

    let test_stat = 2.0 * n_samples as f64 * mi;
    let p_value = chi_squared_p_value_1df(test_stat);
    Ok(p_value)
}

/// Compute partial correlation between i and j given cond_set.
///
/// Uses the precision matrix approach: parcorr(i,j|S) = -P[i,j] / sqrt(P[i,i] * P[j,j])
/// where P = inv(Sigma_S) and Sigma_S is the submatrix of the covariance for {i,j} ∪ S.
fn compute_partial_corr(
    i: usize,
    j: usize,
    cond_set: &[usize],
    cov_matrix: &[Vec<f64>],
) -> CausalityResult<f64> {
    if cond_set.is_empty() {
        // Simple Pearson correlation
        let var_i = cov_matrix[i][i];
        let var_j = cov_matrix[j][j];
        let denom = (var_i * var_j).sqrt();
        if denom < 1e-15 {
            return Ok(0.0);
        }
        return Ok(cov_matrix[i][j] / denom);
    }

    // Build the sub-covariance matrix for variables {i, j} ∪ cond_set
    let mut indices = vec![i, j];
    indices.extend_from_slice(cond_set);
    let k = indices.len();

    let mut sub_cov = vec![vec![0.0; k]; k];
    for (a_idx, &a) in indices.iter().enumerate() {
        for (b_idx, &b) in indices.iter().enumerate() {
            sub_cov[a_idx][b_idx] = cov_matrix[a][b];
        }
    }

    // Regularize
    for idx in 0..k {
        sub_cov[idx][idx] += 1e-10;
    }

    // Invert
    let precision = invert_small_matrix(&sub_cov)?;

    let denom = (precision[0][0] * precision[1][1]).sqrt();
    if denom < 1e-15 {
        return Ok(0.0);
    }

    Ok(-precision[0][1] / denom)
}

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------

/// Compute the covariance matrix from data samples.
fn compute_covariance_matrix(data: &[Vec<f64>]) -> CausalityResult<Vec<Vec<f64>>> {
    let n = data.len();
    let p = data[0].len();

    // Compute means
    let mut means = vec![0.0; p];
    for sample in data {
        for (j, &v) in sample.iter().enumerate() {
            means[j] += v;
        }
    }
    for m in &mut means {
        *m /= n as f64;
    }

    // Compute covariance
    let mut cov = vec![vec![0.0; p]; p];
    for sample in data {
        for a in 0..p {
            let da = sample[a] - means[a];
            for b in a..p {
                let db = sample[b] - means[b];
                cov[a][b] += da * db;
            }
        }
    }

    let denom = (n as f64 - 1.0).max(1.0);
    for a in 0..p {
        for b in a..p {
            cov[a][b] /= denom;
            cov[b][a] = cov[a][b];
        }
    }

    Ok(cov)
}

/// Invert a small matrix using Gauss-Jordan elimination with partial pivoting.
fn invert_small_matrix(mat: &[Vec<f64>]) -> CausalityResult<Vec<Vec<f64>>> {
    let n = mat.len();
    let mut augmented = vec![vec![0.0; 2 * n]; n];

    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = mat[i][j];
        }
        augmented[i][n + i] = 1.0;
    }

    for col in 0..n {
        let mut max_val = augmented[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = augmented[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(TimeSeriesError::NumericalInstability(
                "Singular matrix in partial correlation computation".to_string(),
            ));
        }

        if max_row != col {
            augmented.swap(col, max_row);
        }

        let pivot = augmented[col][col];
        for j in 0..(2 * n) {
            augmented[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = augmented[row][col];
                for j in 0..(2 * n) {
                    augmented[row][j] -= factor * augmented[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = augmented[i][n + j];
        }
    }

    Ok(inv)
}

// ---------------------------------------------------------------------------
// Edge orientation helpers
// ---------------------------------------------------------------------------

fn is_directed(edge_types: &HashMap<(usize, usize), EdgeType>, from: usize, to: usize) -> bool {
    edge_types
        .get(&(from, to))
        .map(|&et| et == EdgeType::Directed)
        .unwrap_or(false)
}

fn is_undirected(edge_types: &HashMap<(usize, usize), EdgeType>, a: usize, b: usize) -> bool {
    let k1 = (a, b);
    let k2 = (b, a);
    let k_canon = if a < b { (a, b) } else { (b, a) };

    // If any directed orientation exists, it's not undirected
    if is_directed(edge_types, a, b) || is_directed(edge_types, b, a) {
        return false;
    }

    // Check if there is an undirected edge
    edge_types
        .get(&k_canon)
        .map(|&et| et == EdgeType::Undirected)
        .unwrap_or(false)
        || edge_types
            .get(&k1)
            .map(|&et| et == EdgeType::Undirected)
            .unwrap_or(false)
        || edge_types
            .get(&k2)
            .map(|&et| et == EdgeType::Undirected)
            .unwrap_or(false)
}

fn orient_edge(edge_types: &mut HashMap<(usize, usize), EdgeType>, from: usize, to: usize) {
    // Remove undirected versions
    let k_canon = if from < to { (from, to) } else { (to, from) };
    edge_types.remove(&k_canon);
    edge_types.remove(&(to, from));
    edge_types.remove(&(from, to));
    // Insert directed
    edge_types.insert((from, to), EdgeType::Directed);
}

// ---------------------------------------------------------------------------
// Combination generation
// ---------------------------------------------------------------------------

fn gen_combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();
    gen_combinations_rec(items, k, 0, &mut vec![], &mut result);
    result
}

fn gen_combinations_rec(
    items: &[usize],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    let available = items.len() - start;
    if available < remaining {
        return;
    }
    for i in start..items.len() {
        current.push(items[i]);
        gen_combinations_rec(items, k, i + 1, current, result);
        current.pop();
    }
}

// ---------------------------------------------------------------------------
// Standard normal CDF
// ---------------------------------------------------------------------------

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let p = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Chi-squared(1) p-value
fn chi_squared_p_value_1df(chi2: f64) -> f64 {
    if chi2 <= 0.0 {
        return 1.0;
    }
    // chi2(1) CDF = 2 * Phi(sqrt(chi2)) - 1
    // p-value = 1 - CDF = 2 * (1 - Phi(sqrt(chi2)))
    2.0 * (1.0 - normal_cdf(chi2.sqrt()))
}

// ---------------------------------------------------------------------------
// Fisher z-transform utility (public for tests)
// ---------------------------------------------------------------------------

/// Compute Fisher's z-transform of a correlation coefficient.
///
/// z = 0.5 * ln((1 + r) / (1 - r))
///
/// Under H0 (r = 0), z * sqrt(n - |S| - 3) ~ N(0, 1).
pub fn fisher_z_transform(r: f64, n: usize, cond_size: usize) -> (f64, f64) {
    let clamped = r.clamp(-0.9999, 0.9999);
    let z = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln();
    let df = n as f64 - cond_size as f64 - 3.0;
    let z_stat = if df > 0.0 { z * df.sqrt() } else { 0.0 };
    let p_value = if df > 0.0 {
        2.0 * (1.0 - normal_cdf(z_stat.abs()))
    } else {
        1.0
    };
    (z_stat, p_value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LCG pseudo-random for deterministic tests.
    fn next_rand(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 32) as f64) / (u32::MAX as f64) - 0.5
    }

    /// Generate N samples of (X, Y, Z) where X -> Y -> Z (chain)
    fn generate_chain(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let x = next_rand(&mut state);
            let y = 0.8 * x + next_rand(&mut state) * 0.3;
            let z = 0.8 * y + next_rand(&mut state) * 0.3;
            data.push(vec![x, y, z]);
        }
        data
    }

    /// Generate N samples of (X, Y, Z) where X -> Z <- Y (v-structure)
    fn generate_v_structure(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let x = next_rand(&mut state);
            let y = next_rand(&mut state);
            let z = 0.7 * x + 0.7 * y + next_rand(&mut state) * 0.2;
            data.push(vec![x, y, z]);
        }
        data
    }

    /// Generate independent variables
    fn generate_independent(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut state = seed;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let x = next_rand(&mut state);
            let y = next_rand(&mut state);
            let z = next_rand(&mut state);
            data.push(vec![x, y, z]);
        }
        data
    }

    #[test]
    fn test_pc_config_default() {
        let cfg = PCConfig::default();
        assert!((cfg.significance_level - 0.05).abs() < 1e-10);
        assert_eq!(cfg.max_cond_set_size, 4);
        assert_eq!(cfg.test_type, IndependenceTest::PartialCorrelation);
    }

    #[test]
    fn test_independent_variables_no_edge() {
        let data = generate_independent(500, 42);
        let config = PCConfig {
            significance_level: 0.05,
            max_cond_set_size: 2,
            test_type: IndependenceTest::PartialCorrelation,
        };
        let pc = PCAlgorithm::new(config);
        let graph = pc.discover(&data).expect("discovery");
        // Independent variables should have no edges (or very few due to chance)
        assert!(
            graph.edges.len() <= 1,
            "Independent vars should have ~0 edges, got {}",
            graph.edges.len()
        );
    }

    #[test]
    fn test_chain_skeleton_discovered() {
        // X -> Y -> Z: skeleton should have X-Y and Y-Z edges
        let data = generate_chain(1000, 123);
        let config = PCConfig {
            significance_level: 0.05,
            max_cond_set_size: 2,
            test_type: IndependenceTest::PartialCorrelation,
        };
        let pc = PCAlgorithm::new(config);
        let graph = pc.discover(&data).expect("discovery");

        // Should have edges involving Y (Y is connected to both X and Z)
        let has_xy = graph.has_edge(0, 1);
        let has_yz = graph.has_edge(1, 2);
        assert!(has_xy, "Should have X-Y edge in chain");
        assert!(has_yz, "Should have Y-Z edge in chain");

        // X-Z should NOT be present (conditional independence given Y)
        let has_xz = graph.has_edge(0, 2);
        assert!(!has_xz, "Should NOT have X-Z direct edge in chain");
    }

    #[test]
    fn test_v_structure_orientation() {
        // X -> Z <- Y: X and Y independent, both cause Z
        let data = generate_v_structure(1000, 456);
        let config = PCConfig {
            significance_level: 0.05,
            max_cond_set_size: 2,
            test_type: IndependenceTest::PartialCorrelation,
        };
        let pc = PCAlgorithm::new(config);
        let graph = pc.discover(&data).expect("discovery");

        // X and Y should NOT be adjacent
        assert!(
            !graph.has_edge(0, 1),
            "X and Y should not be adjacent in v-structure"
        );

        // Check that Z is connected to both X and Y
        let has_xz = graph.has_edge(0, 2);
        let has_yz = graph.has_edge(1, 2);
        assert!(has_xz, "Should have X-Z edge");
        assert!(has_yz, "Should have Y-Z edge");

        // V-structure should orient: X -> Z and Y -> Z
        let x_to_z = graph.has_directed_edge(0, 2);
        let y_to_z = graph.has_directed_edge(1, 2);
        assert!(x_to_z, "Should orient X -> Z in v-structure");
        assert!(y_to_z, "Should orient Y -> Z in v-structure");
    }

    #[test]
    fn test_causal_graph_node_edge_counts() {
        let data = generate_chain(500, 789);
        let config = PCConfig::default();
        let pc = PCAlgorithm::new(config);
        let graph = pc.discover(&data).expect("discovery");

        assert_eq!(graph.nodes, 3);
        // Chain should have 2 edges (X-Y and Y-Z), X-Z removed
        assert!(
            graph.edges.len() >= 2,
            "Chain should have at least 2 edges, got {}",
            graph.edges.len()
        );
    }

    #[test]
    fn test_partial_correlation_independent() {
        // Two truly independent variables: correlation should be near zero
        let data = generate_independent(500, 999);
        let cov = compute_covariance_matrix(&data).expect("cov");
        let parcorr = compute_partial_corr(0, 1, &[], &cov).expect("parcorr");
        assert!(
            parcorr.abs() < 0.15,
            "Independent vars should have near-zero partial corr, got {}",
            parcorr
        );
    }

    #[test]
    fn test_partial_correlation_dependent() {
        let data = generate_chain(500, 111);
        let cov = compute_covariance_matrix(&data).expect("cov");
        let parcorr = compute_partial_corr(0, 1, &[], &cov).expect("parcorr");
        assert!(
            parcorr.abs() > 0.3,
            "Dependent vars should have significant partial corr, got {}",
            parcorr
        );
    }

    #[test]
    fn test_partial_correlation_conditional_independence() {
        // In X->Y->Z, X and Z should be conditionally independent given Y
        let data = generate_chain(1000, 222);
        let cov = compute_covariance_matrix(&data).expect("cov");
        let parcorr_xz_given_y = compute_partial_corr(0, 2, &[1], &cov).expect("parcorr");
        assert!(
            parcorr_xz_given_y.abs() < 0.15,
            "X⊥Z|Y should hold in chain, parcorr={}",
            parcorr_xz_given_y
        );
    }

    #[test]
    fn test_fisher_z_transform_correct_pvalue() {
        // Zero correlation => p-value should be ~1
        let (_, p) = fisher_z_transform(0.0, 100, 0);
        assert!(
            (p - 1.0).abs() < 0.01,
            "Zero correlation should give p≈1, got {}",
            p
        );

        // Strong correlation => p-value should be small
        let (_, p2) = fisher_z_transform(0.9, 100, 0);
        assert!(
            p2 < 0.01,
            "Strong correlation should give small p-value, got {}",
            p2
        );
    }

    #[test]
    fn test_mutual_information_test() {
        let data = generate_chain(500, 333);
        let cov = compute_covariance_matrix(&data).expect("cov");
        let p = mutual_information_test(0, 1, &[], 500, &cov).expect("mi");
        assert!(
            p < 0.05,
            "MI test should detect dependence in chain, p={}",
            p
        );
    }

    #[test]
    fn test_pc_insufficient_data() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pc = PCAlgorithm::new(PCConfig::default());
        let result = pc.discover(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_type_non_exhaustive() {
        // Verify we can construct all edge types
        let _d = EdgeType::Directed;
        let _u = EdgeType::Undirected;
        let _b = EdgeType::Bidirected;
    }

    #[test]
    fn test_known_graph_recovery_synthetic() {
        // Generate a known DAG: X0 -> X1, X0 -> X2, X1 -> X3, X2 -> X3
        // This is a diamond/fork structure
        let n = 2000;
        let mut state: u64 = 42;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let x0 = next_rand(&mut state);
            let x1 = 0.8 * x0 + next_rand(&mut state) * 0.2;
            let x2 = 0.8 * x0 + next_rand(&mut state) * 0.2;
            let x3 = 0.5 * x1 + 0.5 * x2 + next_rand(&mut state) * 0.2;
            data.push(vec![x0, x1, x2, x3]);
        }

        let config = PCConfig {
            significance_level: 0.05,
            max_cond_set_size: 3,
            test_type: IndependenceTest::PartialCorrelation,
        };
        let pc = PCAlgorithm::new(config);
        let graph = pc.discover(&data).expect("discovery");

        assert_eq!(graph.nodes, 4);

        // Should have edges: 0-1, 0-2, 1-3, 2-3
        assert!(graph.has_edge(0, 1), "Should have X0-X1 edge");
        assert!(graph.has_edge(0, 2), "Should have X0-X2 edge");
        assert!(graph.has_edge(1, 3), "Should have X1-X3 edge");
        assert!(graph.has_edge(2, 3), "Should have X2-X3 edge");
    }
}
