//! PC Algorithm for Causal Discovery
//!
//! The Peter-Clark (PC) algorithm is a constraint-based method that learns
//! the Markov equivalence class of a DAG from observational data, represented
//! as a Completed Partially Directed Acyclic Graph (CPDAG).
//!
//! # Phases
//!
//! 1. **Skeleton discovery** (PC-stable variant): start with a fully connected
//!    undirected graph and iteratively remove edges when a conditional
//!    independence is found, recording separation sets.
//!
//! 2. **V-structure orientation**: for each triple X - Z - Y where X and Y
//!    are not adjacent, orient X -> Z <- Y if Z is not in sep(X, Y).
//!
//! 3. **Meek's rules** (R1-R4): propagate orientations to avoid new
//!    v-structures and cycles.
//!
//! # PC-stable variant
//!
//! In the standard PC algorithm, edge removals during skeleton discovery can
//! depend on the order in which edges are tested. The PC-stable variant
//! (Colombo & Maathuis, 2014) fixes this by computing all removals at each
//! conditioning-set size before applying them.
//!
//! # References
//!
//! - Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction,
//!   and Search* (2nd ed.). MIT Press.
//! - Colombo, D. & Maathuis, M.H. (2014). Order-independent constraint-based
//!   causal structure learning. *JMLR* 15, 3741-3782.
//! - Meek, C. (1995). Causal inference and causal explanation with background
//!   knowledge. *UAI 1995*, pp. 403-410.

use std::collections::HashMap;

use scirs2_core::ndarray::ArrayView2;

use super::conditional_independence::{ConditionalIndependenceTest, PartialCorrelationTest};
use super::{CausalGraph, EdgeMark};
use crate::error::{StatsError, StatsResult};

// ---------------------------------------------------------------------------
// PC Algorithm
// ---------------------------------------------------------------------------

/// Configuration for the PC algorithm.
#[derive(Debug, Clone)]
pub struct PcAlgorithm {
    /// Significance level alpha for CI tests (default 0.05).
    pub alpha: f64,
    /// Maximum conditioning set size (default 3).
    pub max_cond_set_size: usize,
    /// Whether to use the PC-stable variant (default true).
    pub stable: bool,
}

impl Default for PcAlgorithm {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            max_cond_set_size: 3,
            stable: true,
        }
    }
}

/// Result of the PC algorithm.
#[derive(Debug, Clone)]
pub struct PcResult {
    /// The learned CPDAG.
    pub graph: CausalGraph,
    /// Separation sets: sep_sets[(i,j)] = conditioning set that made i and j independent.
    pub sep_sets: HashMap<(usize, usize), Vec<usize>>,
    /// Number of CI tests performed.
    pub n_tests: usize,
}

impl PcAlgorithm {
    /// Create a PC algorithm with the given significance level.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            ..Default::default()
        }
    }

    /// Create a PC algorithm with custom parameters.
    pub fn with_params(alpha: f64, max_cond_set_size: usize, stable: bool) -> Self {
        Self {
            alpha,
            max_cond_set_size,
            stable,
        }
    }

    /// Run the PC algorithm using the default partial correlation CI test.
    pub fn fit(&self, data: ArrayView2<f64>, var_names: &[&str]) -> StatsResult<PcResult> {
        let ci_test = PartialCorrelationTest::new(self.alpha);
        self.fit_with_test(data, var_names, &ci_test)
    }

    /// Run the PC algorithm with a custom CI test.
    pub fn fit_with_test<T: ConditionalIndependenceTest>(
        &self,
        data: ArrayView2<f64>,
        var_names: &[&str],
        ci_test: &T,
    ) -> StatsResult<PcResult> {
        let p = data.ncols();
        if var_names.len() != p {
            return Err(StatsError::DimensionMismatch(
                "var_names length must match number of columns".to_owned(),
            ));
        }
        if p == 0 {
            return Ok(PcResult {
                graph: CausalGraph::new(var_names),
                sep_sets: HashMap::new(),
                n_tests: 0,
            });
        }

        // Phase 1: Skeleton discovery
        let (adj, sep_sets, n_tests) = if self.stable {
            self.skeleton_stable(data, p, ci_test)?
        } else {
            self.skeleton_standard(data, p, ci_test)?
        };

        // Phase 2: Orient v-structures
        let mut graph = CausalGraph::new(var_names);
        // Set up adjacency from skeleton
        for i in 0..p {
            for j in (i + 1)..p {
                if adj[i][j] {
                    graph.set_edge(i, j, EdgeMark::Tail, EdgeMark::Tail);
                }
            }
        }

        orient_v_structures(&mut graph, &adj, &sep_sets, p);

        // Phase 3: Apply Meek's rules R1-R4
        apply_meek_rules(&mut graph, p);

        Ok(PcResult {
            graph,
            sep_sets,
            n_tests,
        })
    }

    /// Standard (order-dependent) skeleton discovery.
    fn skeleton_standard<T: ConditionalIndependenceTest>(
        &self,
        data: ArrayView2<f64>,
        p: usize,
        ci_test: &T,
    ) -> StatsResult<(Vec<Vec<bool>>, HashMap<(usize, usize), Vec<usize>>, usize)> {
        let mut adj = vec![vec![true; p]; p];
        for i in 0..p {
            adj[i][i] = false;
        }
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut n_tests = 0usize;

        for ord in 0..=self.max_cond_set_size {
            let edges: Vec<(usize, usize)> = (0..p)
                .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
                .filter(|&(i, j)| adj[i][j])
                .collect();

            for (x, y) in edges {
                let z_candidates: Vec<usize> =
                    (0..p).filter(|&k| k != x && k != y && adj[x][k]).collect();
                if z_candidates.len() < ord {
                    continue;
                }

                for z_set in subsets(&z_candidates, ord) {
                    n_tests += 1;
                    if ci_test.is_independent(x, y, &z_set, data, self.alpha)? {
                        adj[x][y] = false;
                        adj[y][x] = false;
                        let key = (x.min(y), x.max(y));
                        sep_sets.insert(key, z_set);
                        break;
                    }
                }
            }
        }

        Ok((adj, sep_sets, n_tests))
    }

    /// PC-stable skeleton discovery (order-independent).
    ///
    /// At each conditioning-set size, all removals are computed on the
    /// adjacency from the *previous* level, then applied simultaneously.
    fn skeleton_stable<T: ConditionalIndependenceTest>(
        &self,
        data: ArrayView2<f64>,
        p: usize,
        ci_test: &T,
    ) -> StatsResult<(Vec<Vec<bool>>, HashMap<(usize, usize), Vec<usize>>, usize)> {
        let mut adj = vec![vec![true; p]; p];
        for i in 0..p {
            adj[i][i] = false;
        }
        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut n_tests = 0usize;

        for ord in 0..=self.max_cond_set_size {
            // Snapshot adjacency at start of this order
            let adj_snapshot = adj.clone();

            let edges: Vec<(usize, usize)> = (0..p)
                .flat_map(|i| ((i + 1)..p).map(move |j| (i, j)))
                .filter(|&(i, j)| adj_snapshot[i][j])
                .collect();

            // Collect all removals to apply at end
            let mut removals: Vec<(usize, usize, Vec<usize>)> = Vec::new();

            for (x, y) in edges {
                // Use adjacency from snapshot (PC-stable)
                let z_candidates: Vec<usize> = (0..p)
                    .filter(|&k| k != x && k != y && adj_snapshot[x][k])
                    .collect();
                if z_candidates.len() < ord {
                    continue;
                }

                // Also check neighbours of y (both sides, PC-stable)
                let z_candidates_y: Vec<usize> = (0..p)
                    .filter(|&k| k != x && k != y && adj_snapshot[y][k])
                    .collect();

                let mut found = false;
                // Test from x's side
                for z_set in subsets(&z_candidates, ord) {
                    n_tests += 1;
                    if ci_test.is_independent(x, y, &z_set, data, self.alpha)? {
                        removals.push((x, y, z_set));
                        found = true;
                        break;
                    }
                }
                if found {
                    continue;
                }
                // Test from y's side (PC-stable considers both neighbour sets)
                if z_candidates_y.len() >= ord {
                    for z_set in subsets(&z_candidates_y, ord) {
                        // Skip sets already tested from x's side
                        n_tests += 1;
                        if ci_test.is_independent(x, y, &z_set, data, self.alpha)? {
                            removals.push((x, y, z_set));
                            break;
                        }
                    }
                }
            }

            // Apply removals simultaneously
            for (x, y, z_set) in removals {
                adj[x][y] = false;
                adj[y][x] = false;
                let key = (x.min(y), x.max(y));
                sep_sets.insert(key, z_set);
            }
        }

        Ok((adj, sep_sets, n_tests))
    }
}

// ---------------------------------------------------------------------------
// V-structure orientation
// ---------------------------------------------------------------------------

/// Orient v-structures: for each X - Z - Y where X-Y not adjacent,
/// if Z not in sep(X,Y), orient X -> Z <- Y.
fn orient_v_structures(
    graph: &mut CausalGraph,
    adj: &[Vec<bool>],
    sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    p: usize,
) {
    for z in 0..p {
        let neighbours: Vec<usize> = (0..p).filter(|&k| k != z && adj[z][k]).collect();
        for i in 0..neighbours.len() {
            for j in (i + 1)..neighbours.len() {
                let x = neighbours[i];
                let y = neighbours[j];
                // x - y must not be adjacent
                if adj[x][y] {
                    continue;
                }
                let key = (x.min(y), x.max(y));
                let sep = sep_sets.get(&key).cloned().unwrap_or_default();
                if !sep.contains(&z) {
                    // Orient X -> Z <- Y
                    graph.set_edge(x, z, EdgeMark::Tail, EdgeMark::Arrow);
                    graph.set_edge(y, z, EdgeMark::Tail, EdgeMark::Arrow);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Meek's Rules R1-R4
// ---------------------------------------------------------------------------

/// Apply Meek's orientation rules R1-R4 until no more orientations change.
///
/// - R1: a -> b - c and a not adj c => orient b -> c
/// - R2: a -> b -> c and a - c => orient a -> c
/// - R3: a - b, a - c, a - d, b -> d, c -> d, b not adj c => orient a -> d
/// - R4: a - b, b -> c, a - c, a -> d not adj c, d -> c exists on some
///       directed path => orient a -> c (acyclicity preservation)
pub fn apply_meek_rules(graph: &mut CausalGraph, p: usize) {
    let max_iterations = p * p + 10;
    let mut changed = true;
    let mut iterations = 0;

    while changed && iterations < max_iterations {
        changed = false;
        iterations += 1;

        // R1: a -> b - c, a not adj c => b -> c
        changed |= meek_r1(graph, p);

        // R2: a -> b -> c, a - c => a -> c
        changed |= meek_r2(graph, p);

        // R3: a - d, b -> d, c -> d, a - b, a - c, b not adj c => a -> d
        changed |= meek_r3(graph, p);

        // R4: a - b, b -> c -> ... -> a (directed path), b not adj a through
        //     the path => orient a -> b to avoid cycle
        changed |= meek_r4(graph, p);
    }
}

/// R1: If a -> b - c and a is not adjacent to c, orient b -> c.
fn meek_r1(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for b in 0..p {
        for a in 0..p {
            if a == b {
                continue;
            }
            // Check a -> b (directed from a to b)
            if !graph.is_directed(a, b) {
                continue;
            }
            for c in 0..p {
                if c == a || c == b {
                    continue;
                }
                // Check b - c (undirected)
                if !graph.is_undirected(b, c) {
                    continue;
                }
                // Check a not adjacent to c
                if graph.is_adjacent(a, c) {
                    continue;
                }
                // Orient b -> c
                graph.set_edge(b, c, EdgeMark::Tail, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

/// R2: If a -> b -> c and a - c, orient a -> c.
fn meek_r2(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for b in 0..p {
            if a == b {
                continue;
            }
            if !graph.is_directed(a, b) {
                continue;
            }
            for c in 0..p {
                if c == a || c == b {
                    continue;
                }
                if !graph.is_directed(b, c) {
                    continue;
                }
                if !graph.is_undirected(a, c) {
                    continue;
                }
                graph.set_edge(a, c, EdgeMark::Tail, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

/// R3: If a - d, and there exist b, c such that b -> d, c -> d,
/// a - b, a - c, and b not adj c, orient a -> d.
fn meek_r3(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for d in 0..p {
            if a == d {
                continue;
            }
            // a - d (undirected)
            if !graph.is_undirected(a, d) {
                continue;
            }
            // Find b, c where: b -> d, c -> d, a - b, a - c, b not adj c
            let parents_of_d: Vec<usize> = (0..p)
                .filter(|&k| k != a && k != d && graph.is_directed(k, d))
                .collect();
            let mut orient = false;
            for i in 0..parents_of_d.len() {
                for j in (i + 1)..parents_of_d.len() {
                    let b = parents_of_d[i];
                    let c = parents_of_d[j];
                    if graph.is_undirected(a, b)
                        && graph.is_undirected(a, c)
                        && !graph.is_adjacent(b, c)
                    {
                        orient = true;
                        break;
                    }
                }
                if orient {
                    break;
                }
            }
            if orient {
                graph.set_edge(a, d, EdgeMark::Tail, EdgeMark::Arrow);
                changed = true;
            }
        }
    }
    changed
}

/// R4: If a - b and there exists a directed path from b to a through
/// some node c where b -> c and a - c, orient a -> b.
///
/// More precisely: if a - b, a - c, c -> ... -> b (directed path of length >= 1),
/// and b -> c, then orient a -> b. This is needed to prevent creating a new
/// v-structure or a directed cycle.
fn meek_r4(graph: &mut CausalGraph, p: usize) -> bool {
    let mut changed = false;
    for a in 0..p {
        for b in 0..p {
            if a == b {
                continue;
            }
            if !graph.is_undirected(a, b) {
                continue;
            }
            // Check: exists c such that a - c, b -> c, and directed path c ->* b
            // (which would create cycle if we orient a <- b)
            for c in 0..p {
                if c == a || c == b {
                    continue;
                }
                if !graph.is_undirected(a, c) {
                    continue;
                }
                if !graph.is_directed(b, c) {
                    continue;
                }
                // Check directed path from c to a
                if has_directed_path(graph, c, a, p) {
                    graph.set_edge(a, b, EdgeMark::Tail, EdgeMark::Arrow);
                    changed = true;
                    break;
                }
            }
        }
    }
    changed
}

/// Check if there is a directed path from `src` to `dst` in the graph.
fn has_directed_path(graph: &CausalGraph, src: usize, dst: usize, p: usize) -> bool {
    let mut visited = vec![false; p];
    let mut stack = vec![src];
    while let Some(cur) = stack.pop() {
        if cur == dst {
            return true;
        }
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        for next in 0..p {
            if !visited[next] && graph.is_directed(cur, next) {
                stack.push(next);
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate all subsets of `items` of size `k`.
pub(crate) fn subsets<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![Vec::new()];
    }
    if k > items.len() {
        return Vec::new();
    }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn lcg_uniform(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn lcg_normal(s: &mut u64) -> f64 {
        let u1 = lcg_uniform(s).max(1e-15);
        let u2 = lcg_uniform(s);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate chain X -> Y -> Z data.
    fn chain_data(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg = seed;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 1]] = 0.9 * data[[i, 0]] + lcg_normal(&mut lcg) * 0.3;
            data[[i, 2]] = 0.9 * data[[i, 1]] + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    /// Generate fork X <- Y -> Z data.
    fn fork_data(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg = seed;
        for i in 0..n {
            let y = lcg_normal(&mut lcg);
            data[[i, 0]] = 0.9 * y + lcg_normal(&mut lcg) * 0.3;
            data[[i, 1]] = y;
            data[[i, 2]] = 0.9 * y + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    /// Generate collider X -> Y <- Z data.
    fn collider_data(n: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        let mut lcg = seed;
        for i in 0..n {
            data[[i, 0]] = lcg_normal(&mut lcg);
            data[[i, 2]] = lcg_normal(&mut lcg);
            data[[i, 1]] = 0.7 * data[[i, 0]] + 0.7 * data[[i, 2]] + lcg_normal(&mut lcg) * 0.3;
        }
        data
    }

    #[test]
    fn test_pc_chain() {
        let data = chain_data(300, 12345);
        let pc = PcAlgorithm::new(0.05);
        let result = pc.fit(data.view(), &["X", "Y", "Z"]).expect("PC failed");
        // Chain: X - Y - Z, X not adj Z
        // X-Y should be adjacent
        assert!(
            result.graph.is_adjacent(0, 1),
            "X-Y should be adjacent in chain"
        );
        // Y-Z should be adjacent
        assert!(
            result.graph.is_adjacent(1, 2),
            "Y-Z should be adjacent in chain"
        );
        // X-Z should NOT be adjacent (conditionally independent given Y)
        assert!(
            !result.graph.is_adjacent(0, 2),
            "X-Z should not be adjacent in chain"
        );
    }

    #[test]
    fn test_pc_fork() {
        let data = fork_data(300, 54321);
        let pc = PcAlgorithm::new(0.05);
        let result = pc.fit(data.view(), &["X", "Y", "Z"]).expect("PC failed");
        // Fork: X <- Y -> Z
        // X-Y, Y-Z adjacent; X-Z not adjacent (given Y)
        assert!(result.graph.is_adjacent(0, 1), "X-Y should be adjacent");
        assert!(result.graph.is_adjacent(1, 2), "Y-Z should be adjacent");
        assert!(
            !result.graph.is_adjacent(0, 2),
            "X-Z should not be adjacent given Y"
        );
    }

    #[test]
    fn test_pc_collider() {
        let data = collider_data(300, 99999);
        let pc = PcAlgorithm::new(0.05);
        let result = pc.fit(data.view(), &["X", "Y", "Z"]).expect("PC failed");
        // Collider: X -> Y <- Z
        // All should be adjacent but X-Z not
        assert!(result.graph.is_adjacent(0, 1), "X-Y should be adjacent");
        assert!(result.graph.is_adjacent(1, 2), "Y-Z should be adjacent");
        // X and Z should be marginally independent
        assert!(
            !result.graph.is_adjacent(0, 2),
            "X-Z should not be adjacent (independent causes)"
        );
        // V-structure: X -> Y <- Z
        // Y should be oriented as collider
        assert!(
            result.graph.is_directed(0, 1) || result.graph.is_directed(2, 1),
            "At least one edge should point into Y (v-structure)"
        );
    }

    #[test]
    fn test_pc_meek_r1() {
        // Test Meek's R1: if a -> b - c and a not adj c, orient b -> c
        let mut graph = CausalGraph::new(&["A", "B", "C"]);
        // a -> b
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrow);
        // b - c
        graph.set_edge(1, 2, EdgeMark::Tail, EdgeMark::Tail);
        // a not adj c (no edge)

        apply_meek_rules(&mut graph, 3);

        // R1 should orient b -> c
        assert!(graph.is_directed(1, 2), "R1: b -> c expected");
    }

    #[test]
    fn test_pc_meek_r2() {
        // R2: a -> b -> c and a - c => a -> c
        let mut graph = CausalGraph::new(&["A", "B", "C"]);
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrow); // a -> b
        graph.set_edge(1, 2, EdgeMark::Tail, EdgeMark::Arrow); // b -> c
        graph.set_edge(0, 2, EdgeMark::Tail, EdgeMark::Tail); // a - c

        apply_meek_rules(&mut graph, 3);

        assert!(graph.is_directed(0, 2), "R2: a -> c expected");
    }

    #[test]
    fn test_pc_meek_r3() {
        // R3: a - d, b -> d, c -> d, a - b, a - c, b not adj c => a -> d
        let mut graph = CausalGraph::new(&["A", "B", "C", "D"]);
        graph.set_edge(0, 3, EdgeMark::Tail, EdgeMark::Tail); // a - d
        graph.set_edge(1, 3, EdgeMark::Tail, EdgeMark::Arrow); // b -> d
        graph.set_edge(2, 3, EdgeMark::Tail, EdgeMark::Arrow); // c -> d
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Tail); // a - b
        graph.set_edge(0, 2, EdgeMark::Tail, EdgeMark::Tail); // a - c
                                                              // b and c NOT adjacent

        apply_meek_rules(&mut graph, 4);

        assert!(graph.is_directed(0, 3), "R3: a -> d expected");
    }

    #[test]
    fn test_pc_stable_vs_standard() {
        let data = chain_data(200, 77777);
        let pc_stable = PcAlgorithm::with_params(0.05, 3, true);
        let pc_standard = PcAlgorithm::with_params(0.05, 3, false);
        let r1 = pc_stable
            .fit(data.view(), &["X", "Y", "Z"])
            .expect("stable failed");
        let r2 = pc_standard
            .fit(data.view(), &["X", "Y", "Z"])
            .expect("standard failed");
        // Both should find the same skeleton for a simple chain
        assert_eq!(
            r1.graph.is_adjacent(0, 2),
            r2.graph.is_adjacent(0, 2),
            "Skeleton should match for simple structures"
        );
    }

    #[test]
    fn test_pc_sep_sets() {
        let data = chain_data(300, 12345);
        let pc = PcAlgorithm::new(0.05);
        let result = pc.fit(data.view(), &["X", "Y", "Z"]).expect("PC failed");
        // X-Z should be separated by Y
        if let Some(sep) = result.sep_sets.get(&(0, 2)) {
            assert!(sep.contains(&1), "Sep set for X-Z should contain Y");
        }
        // If X-Z is adjacent, there's no sep set, which is also valid for some data
    }

    #[test]
    fn test_subsets() {
        let items = vec![0, 1, 2, 3];
        let s0 = subsets(&items, 0);
        assert_eq!(s0.len(), 1);
        assert!(s0[0].is_empty());

        let s1 = subsets(&items, 1);
        assert_eq!(s1.len(), 4);

        let s2 = subsets(&items, 2);
        assert_eq!(s2.len(), 6);

        let s3 = subsets(&items, 3);
        assert_eq!(s3.len(), 4);

        let s4 = subsets(&items, 4);
        assert_eq!(s4.len(), 1);

        let s5 = subsets(&items, 5);
        assert!(s5.is_empty());
    }

    #[test]
    fn test_directed_path_detection() {
        let mut graph = CausalGraph::new(&["A", "B", "C", "D"]);
        graph.set_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrow); // A -> B
        graph.set_edge(1, 2, EdgeMark::Tail, EdgeMark::Arrow); // B -> C
        graph.set_edge(2, 3, EdgeMark::Tail, EdgeMark::Arrow); // C -> D

        assert!(has_directed_path(&graph, 0, 3, 4), "A -> B -> C -> D");
        assert!(has_directed_path(&graph, 0, 2, 4), "A -> B -> C");
        assert!(!has_directed_path(&graph, 3, 0, 4), "No path D -> A");
        assert!(!has_directed_path(&graph, 1, 0, 4), "No path B -> A");
    }
}
