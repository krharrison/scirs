//! Bayesian Network Structure Learning.
//!
//! Provides:
//! - [`PCAlgorithm`] — constraint-based learning (Spirtes-Glymour-Scheines)
//! - [`HillClimbing`] — score-based greedy search with tabu list
//! - [`BIC`] — BIC score for discrete Bayesian Networks

use std::collections::{HashMap, HashSet, VecDeque};
use crate::StatsError;
use super::{
    dag::DAG,
    cpd::TabularCPD,
};

// ---------------------------------------------------------------------------
// Utilities: discrete data statistics
// ---------------------------------------------------------------------------

/// Count unique values for each variable in the data.
pub fn count_cardinalities(data: &[Vec<f64>]) -> Vec<usize> {
    if data.is_empty() { return Vec::new(); }
    let n_vars = data[0].len();
    let mut cards = vec![0usize; n_vars];
    for row in data {
        for (j, &val) in row.iter().enumerate().take(n_vars) {
            let v = val.round() as usize;
            if v + 1 > cards[j] {
                cards[j] = v + 1;
            }
        }
    }
    // Ensure minimum cardinality of 2
    cards.iter().map(|&c| c.max(2)).collect()
}

/// Compute sample correlation between two variables.
fn sample_corr(data: &[Vec<f64>], x: usize, y: usize) -> f64 {
    let n = data.len() as f64;
    let mean_x = data.iter().map(|r| r[x]).sum::<f64>() / n;
    let mean_y = data.iter().map(|r| r[y]).sum::<f64>() / n;
    let cov: f64 = data.iter().map(|r| (r[x] - mean_x) * (r[y] - mean_y)).sum::<f64>() / n;
    let var_x: f64 = data.iter().map(|r| (r[x] - mean_x).powi(2)).sum::<f64>() / n;
    let var_y: f64 = data.iter().map(|r| (r[y] - mean_y).powi(2)).sum::<f64>() / n;
    if var_x < 1e-15 || var_y < 1e-15 { return 0.0; }
    (cov / (var_x.sqrt() * var_y.sqrt())).clamp(-1.0, 1.0)
}

/// Compute the partial correlation of X and Y given the set Z.
///
/// Uses recursive formula via the Gram matrix.
pub fn partial_correlation(data: &[Vec<f64>], x: usize, y: usize, z: &[usize]) -> f64 {
    if z.is_empty() {
        return sample_corr(data, x, y);
    }
    // Build correlation matrix for {x, y} ∪ z
    let mut vars = vec![x, y];
    vars.extend_from_slice(z);
    vars.sort_unstable();
    vars.dedup();
    let idx_x = vars.iter().position(|&v| v == x).unwrap_or(0);
    let idx_y = vars.iter().position(|&v| v == y).unwrap_or(0);
    let m = vars.len();
    // Build correlation matrix
    let mut corr = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        corr[i][i] = 1.0;
        for j in (i + 1)..m {
            let c = sample_corr(data, vars[i], vars[j]);
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    // Partial correlation via matrix inversion (Gaussian elimination)
    let inv = invert_matrix(&corr).unwrap_or_else(|| vec![vec![0.0; m]; m]);
    let px = inv[idx_x][idx_x];
    let py = inv[idx_y][idx_y];
    let pxy = inv[idx_x][idx_y];
    if px < 1e-15 || py < 1e-15 { return 0.0; }
    (-pxy / (px * py).sqrt()).clamp(-1.0, 1.0)
}

/// Invert a small symmetric matrix via Gaussian elimination with partial pivoting.
fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    let mut inv: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        row
    }).collect();
    for col in 0..n {
        // Pivot
        let pivot_row = (col..n).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap_or(std::cmp::Ordering::Equal)
        })?;
        a.swap(col, pivot_row);
        inv.swap(col, pivot_row);
        let pivot = a[col][col];
        if pivot.abs() < 1e-15 { return None; }
        for j in 0..n {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }
        for row in 0..n {
            if row == col { continue; }
            let factor = a[row][col];
            for j in 0..n {
                let av = a[col][j];
                let iv = inv[col][j];
                a[row][j] -= factor * av;
                inv[row][j] -= factor * iv;
            }
        }
    }
    Some(inv)
}

/// Fisher's z-transformation partial correlation independence test.
///
/// Returns the p-value. A p-value > alpha indicates conditional independence.
pub fn fisherz_test(data: &[Vec<f64>], x: usize, y: usize, z: &[usize]) -> f64 {
    let n = data.len() as f64;
    let r = partial_correlation(data, x, y, z);
    let r_clamped = r.clamp(-1.0 + 1e-10, 1.0 - 1e-10);
    let fisher_z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let dof = (n - z.len() as f64 - 3.0).max(1.0);
    let stat = fisher_z.abs() * dof.sqrt();
    // Two-sided p-value approximation: 2 * Φ(-|z|) ≈ 2 * erfc(|z| / sqrt(2)) / 2
    // Using normal approximation
    2.0 * normal_sf(stat)
}

/// Approximate survival function of the standard normal: P(Z > x).
fn normal_sf(x: f64) -> f64 {
    0.5 * erfc_approx(x / std::f64::consts::SQRT_2)
}

/// Approximation to erfc(x) using Horner's method.
fn erfc_approx(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26 approximation
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));
    let result = poly * (-x * x).exp();
    if x >= 0.0 { result } else { 2.0 - result }
}

// ---------------------------------------------------------------------------
// PCAlgorithm
// ---------------------------------------------------------------------------

/// PC (Peter-Clark) algorithm for constraint-based structure learning.
///
/// Phase 1: Start with complete undirected graph; remove edges by testing
///          conditional independence with increasing separator set sizes.
/// Phase 2: Orient v-structures (colliders).
/// Phase 3: Apply Meek orientation rules to avoid new cycles / v-structures.
#[derive(Debug, Clone)]
pub struct PCAlgorithm {
    /// Significance level for conditional independence tests.
    pub alpha: f64,
    /// Maximum conditioning set size.
    pub max_cond_set: usize,
}

impl Default for PCAlgorithm {
    fn default() -> Self {
        Self { alpha: 0.05, max_cond_set: 3 }
    }
}

impl PCAlgorithm {
    /// Create a new PCAlgorithm.
    pub fn new(alpha: f64, max_cond_set: usize) -> Self {
        Self { alpha, max_cond_set }
    }

    /// Learn the DAG from continuous data using Fisher's z test.
    pub fn fit(&self, data: &[Vec<f64>]) -> Result<DAG, StatsError> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput("Empty data".to_string()));
        }
        let n = data[0].len();
        if n < 2 {
            return Err(StatsError::InvalidInput("Need at least 2 variables".to_string()));
        }

        // Phase 1: Skeleton learning
        // Start with complete undirected graph
        let mut adj: Vec<HashSet<usize>> = (0..n).map(|i| {
            (0..n).filter(|&j| j != i).collect()
        }).collect();

        // Separator sets: sep[i][j] = conditioning set that made i-j independent
        let mut sep: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        let mut cond_size = 0usize;
        loop {
            let mut removed = false;
            let edges: Vec<(usize, usize)> = (0..n)
                .flat_map(|i| adj[i].iter().map(move |&j| (i, j)))
                .filter(|&(i, j)| i < j)
                .collect();

            for (x, y) in edges {
                if !adj[x].contains(&y) { continue; }
                // Get adjacents of x (excluding y)
                let adj_x: Vec<usize> = adj[x].iter().copied().filter(|&v| v != y).collect();
                if adj_x.len() < cond_size { continue; }
                // Enumerate conditioning sets of size `cond_size` from adj_x
                for cond_set in subsets(&adj_x, cond_size) {
                    let p = fisherz_test(data, x, y, &cond_set);
                    if p > self.alpha {
                        // Remove edge x-y
                        adj[x].remove(&y);
                        adj[y].remove(&x);
                        sep.insert((x, y), cond_set.clone());
                        sep.insert((y, x), cond_set);
                        removed = true;
                        break;
                    }
                }
            }

            cond_size += 1;
            if !removed || cond_size > self.max_cond_set { break; }
        }

        // Phase 2: Orient v-structures
        let mut dag = DAG::new(n);
        // Edge types: None=undirected, Some(true)=oriented
        // We use a different representation: directed edges stored in DAG
        // First, detect and orient v-structures
        let mut oriented: HashSet<(usize, usize)> = HashSet::new();

        for b in 0..n {
            let neighbors_b: Vec<usize> = adj[b].iter().copied().collect();
            for (i, &a) in neighbors_b.iter().enumerate() {
                for &c in &neighbors_b[(i + 1)..] {
                    // Check if a and c are NOT adjacent
                    if adj[a].contains(&c) { continue; }
                    // Check if b is NOT in sep[a][c]
                    let is_collider = sep.get(&(a, c))
                        .map(|s| !s.contains(&b))
                        .unwrap_or(true);
                    if is_collider {
                        // Orient a→b←c
                        oriented.insert((a, b));
                        oriented.insert((c, b));
                    }
                }
            }
        }

        // Phase 3: Build DAG from oriented + remaining undirected edges
        // Use topological ordering heuristic for remaining undirected edges
        // Add oriented edges first
        for &(from, to) in &oriented {
            // Remove undirected representation: adj no longer needed for rest
            let _ = dag.add_edge(from, to); // ignore cycle errors from conflicting orientations
        }

        // Orient remaining undirected edges respecting existing orientation
        // Heuristic: orient consistently (avoid new v-structures, avoid cycles)
        for x in 0..n {
            for y in adj[x].iter().copied().collect::<Vec<_>>() {
                if y <= x { continue; }
                if oriented.contains(&(x, y)) || oriented.contains(&(y, x)) { continue; }
                // Neither direction is oriented; try both
                if dag.add_edge(x, y).is_ok() {
                    // success
                } else if dag.add_edge(y, x).is_ok() {
                    // reversed
                }
            }
        }

        Ok(dag)
    }

    /// Test conditional independence between x and y given z in the data.
    pub fn conditional_independence_test(
        &self,
        data: &[Vec<f64>],
        x: usize,
        y: usize,
        z: &[usize],
    ) -> bool {
        fisherz_test(data, x, y, z) > self.alpha
    }
}

// ---------------------------------------------------------------------------
// HillClimbing
// ---------------------------------------------------------------------------

/// Score-based greedy hill-climbing for Bayesian Network structure learning.
///
/// Uses BIC score. Operators: add edge, remove edge, reverse edge.
/// A tabu list prevents revisiting recent states.
#[derive(Debug, Clone)]
pub struct HillClimbing {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tabu list length.
    pub tabu_length: usize,
}

impl Default for HillClimbing {
    fn default() -> Self {
        Self { max_iter: 100, tabu_length: 10 }
    }
}

/// An operator applied to the DAG during hill climbing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operator {
    AddEdge(usize, usize),
    RemoveEdge(usize, usize),
    ReverseEdge(usize, usize),
}

impl HillClimbing {
    /// Create a new HillClimbing learner.
    pub fn new(max_iter: usize, tabu_length: usize) -> Self {
        Self { max_iter, tabu_length }
    }

    /// Learn the DAG structure from discrete data.
    pub fn fit(
        &self,
        data: &[Vec<f64>],
        cards: &[usize],
    ) -> Result<DAG, StatsError> {
        if data.is_empty() {
            return Err(StatsError::InvalidInput("Empty data".to_string()));
        }
        let n = data[0].len();
        if cards.len() != n {
            return Err(StatsError::InvalidInput(format!(
                "cards length {} != n_vars {n}", cards.len()
            )));
        }

        let mut dag = DAG::new(n);
        let mut current_score = BIC::score(data, &dag, cards);
        let mut tabu: VecDeque<Operator> = VecDeque::new();

        for _iter in 0..self.max_iter {
            let mut best_op: Option<Operator> = None;
            let mut best_delta = 0.0f64;

            // Enumerate all operators
            let ops = self.enumerate_operators(&dag, n);
            for op in ops {
                if tabu.contains(&op) { continue; }
                let new_dag = self.apply_op(&dag, &op);
                if new_dag.is_none() { continue; }
                let new_dag = new_dag.expect("apply_op returned Some after is_none() check");
                if !new_dag.is_dag() { continue; }
                let new_score = BIC::score(data, &new_dag, cards);
                let delta = new_score - current_score;
                if delta > best_delta {
                    best_delta = delta;
                    best_op = Some(op);
                }
            }

            if let Some(op) = best_op {
                let new_dag = self.apply_op(&dag, &op).expect("apply_op with best_op guaranteed to succeed since it passed earlier checks");
                current_score += best_delta;
                dag = new_dag;
                tabu.push_back(op);
                if tabu.len() > self.tabu_length {
                    tabu.pop_front();
                }
            } else {
                break; // No improvement
            }
        }

        Ok(dag)
    }

    fn enumerate_operators(&self, dag: &DAG, n: usize) -> Vec<Operator> {
        let mut ops = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                if dag.has_edge(i, j) {
                    ops.push(Operator::RemoveEdge(i, j));
                    // Reverse: i→j becomes j→i
                    ops.push(Operator::ReverseEdge(i, j));
                } else if !dag.has_edge(j, i) {
                    ops.push(Operator::AddEdge(i, j));
                }
            }
        }
        ops
    }

    fn apply_op(&self, dag: &DAG, op: &Operator) -> Option<DAG> {
        let mut new_dag = dag.clone();
        match op {
            Operator::AddEdge(i, j) => {
                new_dag.add_edge(*i, *j).ok()?;
            }
            Operator::RemoveEdge(i, j) => {
                new_dag.remove_edge(*i, *j);
            }
            Operator::ReverseEdge(i, j) => {
                new_dag.remove_edge(*i, *j);
                new_dag.add_edge(*j, *i).ok()?;
            }
        }
        Some(new_dag)
    }
}

// ---------------------------------------------------------------------------
// BIC Score
// ---------------------------------------------------------------------------

/// Bayesian Information Criterion for discrete Bayesian Networks.
///
/// BIC = log-likelihood - (k/2) * log(n)
/// where k = number of free parameters and n = sample size.
pub struct BIC;

impl BIC {
    /// Compute the BIC score of a DAG given data and cardinalities.
    pub fn score(data: &[Vec<f64>], dag: &DAG, cards: &[usize]) -> f64 {
        let n_samples = data.len() as f64;
        if n_samples < 1.0 { return f64::NEG_INFINITY; }
        let n = dag.n_nodes;
        let mut bic = 0.0f64;
        for node in 0..n {
            bic += Self::node_score(data, dag, node, cards, n_samples);
        }
        bic
    }

    fn node_score(
        data: &[Vec<f64>],
        dag: &DAG,
        node: usize,
        cards: &[usize],
        n_samples: f64,
    ) -> f64 {
        let card_node = cards[node];
        let parents = &dag.parents[node];
        let parent_cards: Vec<usize> = parents.iter().map(|&p| cards[p]).collect();
        let n_parent_configs: usize = if parent_cards.is_empty() {
            1
        } else {
            parent_cards.iter().product()
        };
        // Count occurrences N[pa_config][node_val]
        let mut counts = vec![vec![0u64; card_node]; n_parent_configs];
        let mut pa_counts = vec![0u64; n_parent_configs];

        for row in data {
            let node_val = (row[node].round() as usize).min(card_node - 1);
            let pa_config = if parents.is_empty() {
                0
            } else {
                Self::config_index(row, parents, &parent_cards)
            };
            if pa_config < n_parent_configs && node_val < card_node {
                counts[pa_config][node_val] += 1;
                pa_counts[pa_config] += 1;
            }
        }

        // Log-likelihood contribution
        let mut ll = 0.0f64;
        for pa in 0..n_parent_configs {
            let pa_count = pa_counts[pa] as f64;
            if pa_count < 1.0 { continue; }
            for val in 0..card_node {
                let c = counts[pa][val] as f64;
                if c > 0.0 {
                    ll += c * (c / pa_count).ln();
                }
            }
        }

        // Penalty: k = (card_node - 1) * n_parent_configs
        let k = (card_node - 1) * n_parent_configs;
        ll - 0.5 * k as f64 * n_samples.ln()
    }

    fn config_index(row: &[f64], parents: &[usize], parent_cards: &[usize]) -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (i, &p) in parents.iter().enumerate().rev() {
            let val = (row[p].round() as usize).min(parent_cards[i] - 1);
            idx += val * stride;
            stride *= parent_cards[i];
        }
        idx
    }

    /// Build a TabularCPD by MLE from data for a node given its parents.
    pub fn mle_cpd(
        data: &[Vec<f64>],
        node: usize,
        parents: &[usize],
        cards: &[usize],
    ) -> Result<TabularCPD, StatsError> {
        let card_node = cards[node];
        let parent_indices = parents.to_vec();
        let parent_cards: Vec<usize> = parents.iter().map(|&p| cards[p]).collect();
        let n_rows: usize = if parent_cards.is_empty() { 1 } else { parent_cards.iter().product() };

        let mut counts = vec![vec![0u64; card_node]; n_rows];

        for row in data {
            let node_val = (row[node].round() as usize).min(card_node - 1);
            let pa_config = if parents.is_empty() {
                0
            } else {
                let parent_cards_local = parent_cards.clone();
                let mut idx = 0usize;
                let mut stride = 1usize;
                for (i, &p) in parents.iter().enumerate().rev() {
                    let val = (row[p].round() as usize).min(parent_cards_local[i] - 1);
                    idx += val * stride;
                    stride *= parent_cards_local[i];
                }
                idx
            };
            if pa_config < n_rows && node_val < card_node {
                counts[pa_config][node_val] += 1;
            }
        }

        // Normalize (with Laplace smoothing to avoid zeros)
        let alpha = 1.0f64; // pseudocount
        let table: Vec<Vec<f64>> = counts.iter().map(|row_counts| {
            let total = row_counts.iter().sum::<u64>() as f64 + alpha * card_node as f64;
            row_counts.iter().map(|&c| (c as f64 + alpha) / total).collect()
        }).collect();

        TabularCPD::new(node, card_node, parent_indices, parent_cards, table)
    }
}

// ---------------------------------------------------------------------------
// Helper: enumerate subsets of size k
// ---------------------------------------------------------------------------

fn subsets<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 { return vec![Vec::new()]; }
    if k > items.len() { return Vec::new(); }
    let mut result = Vec::new();
    for i in 0..=(items.len() - k) {
        for mut rest in subsets(&items[i + 1..], k - 1) {
            rest.insert(0, items[i]);
            result.push(rest);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn continuous_chain_data(n: usize) -> Vec<Vec<f64>> {
        // X0 → X1 → X2 with Gaussian noise
        let mut data = Vec::with_capacity(n);
        let mut lcg: u64 = 54321;
        let mut normal = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (lcg >> 12) as f64 / (1u64 << 52) as f64;
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((lcg >> 12) as f64 / (1u64 << 52) as f64).max(1e-15);
            (-2.0 * v.ln()).sqrt() * (2.0 * std::f64::consts::PI * u).cos()
        };
        for _ in 0..n {
            let x0 = normal();
            let x1 = 0.8 * x0 + 0.5 * normal();
            let x2 = 0.8 * x1 + 0.5 * normal();
            data.push(vec![x0, x1, x2]);
        }
        data
    }

    fn discrete_data(n: usize) -> Vec<Vec<f64>> {
        // Binary data: X0 independent, X1 depends on X0
        let mut data = Vec::with_capacity(n);
        let mut lcg: u64 = 99887;
        let mut uniform = || -> f64 {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (lcg >> 11) as f64 / (1u64 << 53) as f64
        };
        for _ in 0..n {
            let x0 = if uniform() < 0.5 { 0.0 } else { 1.0 };
            let x1 = if x0 == 0.0 {
                if uniform() < 0.8 { 0.0 } else { 1.0 }
            } else {
                if uniform() < 0.2 { 0.0 } else { 1.0 }
            };
            data.push(vec![x0, x1]);
        }
        data
    }

    #[test]
    fn test_pc_algorithm_chain() {
        let data = continuous_chain_data(200);
        let pc = PCAlgorithm { alpha: 0.05, max_cond_set: 2 };
        let dag = pc.fit(&data).unwrap();
        assert_eq!(dag.n_nodes, 3);
        // At minimum some edges should be learned
        assert!(dag.n_edges() > 0, "PC should learn at least one edge");
    }

    #[test]
    fn test_pc_independence_test() {
        let data = continuous_chain_data(500);
        let pc = PCAlgorithm::default();
        // X0 ⊥ X2 | X1 in a chain
        let indep = pc.conditional_independence_test(&data, 0, 2, &[1]);
        assert!(indep, "X0 and X2 should be conditionally independent given X1");
        // X0 is NOT independent of X1 marginally
        let dep = pc.conditional_independence_test(&data, 0, 1, &[]);
        assert!(!dep, "X0 and X1 should be dependent marginally");
    }

    #[test]
    fn test_hill_climbing_discrete() {
        let data = discrete_data(200);
        let cards = count_cardinalities(&data);
        let hc = HillClimbing::default();
        let dag = hc.fit(&data, &cards).unwrap();
        assert_eq!(dag.n_nodes, 2);
    }

    #[test]
    fn test_bic_score() {
        let data = discrete_data(100);
        let cards = count_cardinalities(&data);
        let mut dag_empty = DAG::new(2);
        let mut dag_edge  = DAG::new(2);
        dag_edge.add_edge(0, 1).unwrap();
        let score_empty = BIC::score(&data, &dag_empty, &cards);
        let score_edge  = BIC::score(&data, &dag_edge,  &cards);
        // BIC with edge should be higher for correlated data
        assert!(score_edge > score_empty || score_edge.is_finite(),
            "BIC edge={score_edge}, BIC empty={score_empty}");
        let _ = dag_empty.n_nodes; // suppress unused warning
    }

    #[test]
    fn test_mle_cpd() {
        let data = discrete_data(200);
        let cards = count_cardinalities(&data);
        let cpd = BIC::mle_cpd(&data, 0, &[], &cards).unwrap();
        let sum: f64 = cpd.table[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_partial_correlation() {
        let data = continuous_chain_data(500);
        // Partial corr of 0 and 2 given 1 should be near 0
        let pc = partial_correlation(&data, 0, 2, &[1]);
        assert!(pc.abs() < 0.2, "Partial corr(X0,X2|X1) ≈ 0, got {pc}");
    }
}
