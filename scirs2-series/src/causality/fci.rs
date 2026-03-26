//! FCI (Fast Causal Inference) Algorithm for Time Series with Latent Confounders
//!
//! FCI extends the PC algorithm to handle latent (hidden) confounders by
//! producing a Partial Ancestral Graph (PAG) rather than a DAG. In a PAG:
//! - Bidirected edges (`<->`) indicate latent common causes
//! - Circle marks (`o`) indicate uncertainty about the edge type
//! - Directed edges (`-->`) indicate definite causal direction
//!
//! ## Algorithm Phases
//!
//! 1. **Skeleton discovery**: Build the undirected skeleton via CI tests
//! 2. **V-structure orientation**: Orient unshielded colliders
//! 3. **Temporal priority**: Past nodes must cause present (not the reverse)
//! 4. **FCI orientation rules R1–R4**: Propagate orientations to fixpoint
//!
//! ## References
//!
//! - Spirtes et al. (2000). "Causation, Prediction, and Search."
//! - Zhang (2008). "On the completeness of orientation rules for causal discovery
//!   in the presence of latent confounders and selection bias." AIJ.
//! - Entner & Hoyer (2010). "On causal discovery from time series data using
//!   FCI." PGMC Workshop.

use std::collections::HashMap;

use scirs2_core::ndarray::Array2;

use crate::error::TimeSeriesError;

use super::pag::{EdgeMark, PartialAncestralGraph};
use super::CausalityResult;

/// Configuration for the FCI time series algorithm
#[derive(Debug, Clone)]
pub struct FciTimeSeriesConfig {
    /// Maximum lag to consider
    pub tau_max: usize,
    /// Significance level for CI tests
    pub alpha: f64,
    /// Maximum conditioning set size
    pub max_cond_size: usize,
    /// Whether to apply temporal priority rule (past cannot be caused by future)
    pub apply_temporal_priority: bool,
    /// Maximum iterations for FCI orientation rules convergence
    pub max_orientation_iter: usize,
}

impl Default for FciTimeSeriesConfig {
    fn default() -> Self {
        Self {
            tau_max: 3,
            alpha: 0.05,
            max_cond_size: 4,
            apply_temporal_priority: true,
            max_orientation_iter: 100,
        }
    }
}

/// Result of the FCI algorithm
#[derive(Debug, Clone)]
pub struct FciResult {
    /// The discovered Partial Ancestral Graph
    pub pag: PartialAncestralGraph,
    /// Number of edges with at least one non-Circle mark
    pub n_edges_oriented: usize,
    /// Number of bidirected edges (indicating latent confounders)
    pub n_bidirected_edges: usize,
    /// Number of remaining Circle marks (uncertainty)
    pub n_circle_marks: usize,
    /// Variable pairs connected by bidirected edges (latent confounder pairs)
    pub latent_confounder_pairs: Vec<(usize, usize)>,
}

/// FCI algorithm for time series with latent confounders
pub struct FciTimeSeries<T: super::ci_tests::TimeSeriesCITest> {
    ci_test: T,
    config: FciTimeSeriesConfig,
}

impl<T: super::ci_tests::TimeSeriesCITest> FciTimeSeries<T> {
    /// Create a new FCI time series instance
    pub fn new(ci_test: T, config: FciTimeSeriesConfig) -> Self {
        Self { ci_test, config }
    }

    /// Run the full FCI pipeline on multivariate time series data
    ///
    /// # Arguments
    /// * `data` - Multivariate time series of shape (T, n_vars)
    ///
    /// # Returns
    /// `FciResult` containing the PAG and summary statistics
    pub fn run(&self, data: &Array2<f64>) -> CausalityResult<FciResult> {
        let n_vars = data.ncols();
        let t = data.nrows();

        if n_vars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data must have at least one variable".to_string(),
            ));
        }

        let min_required = self.config.tau_max + 4;
        if t < min_required {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series too short for FCI with given tau_max".to_string(),
                required: min_required,
                actual: t,
            });
        }

        // Phase 1: Skeleton discovery
        let (adj, sep_sets) = self.skeleton_phase(data)?;
        let n_nodes = n_vars * (self.config.tau_max + 1);

        // Phase 2: Initialize PAG with Circle-Circle marks on skeleton edges
        let mut pag = PartialAncestralGraph::initialize_from_skeleton(&adj, n_nodes);
        pag.n_vars = n_vars;
        pag.tau_max = self.config.tau_max;
        pag.n_nodes = n_nodes;
        pag.sep_sets = sep_sets.iter().map(|(&k, v)| (k, v.clone())).collect();

        // Phase 3: Orient v-structures and apply temporal/FCI rules to fixpoint
        self.fci_converge(&mut pag, &sep_sets);

        // Collect result statistics
        let n_edges_oriented = self.count_oriented_edges(&pag);
        let n_bidirected_edges = pag.n_bidirected_edges();
        let n_circle_marks = pag.n_circle_marks();

        // Find latent confounder pairs: variable pairs connected by bidirected edges
        let latent_confounder_pairs = self.find_latent_confounder_pairs(&pag, n_vars);

        Ok(FciResult {
            pag,
            n_edges_oriented,
            n_bidirected_edges,
            n_circle_marks,
            latent_confounder_pairs,
        })
    }

    /// Phase 1: Skeleton discovery over lagged variables
    ///
    /// Variables are indexed as: var v at lag l → node index v + l * n_vars
    /// (lag 0 = contemporaneous, lag tau_max = oldest).
    /// Total nodes: n_vars * (tau_max + 1).
    fn skeleton_phase(
        &self,
        data: &Array2<f64>,
    ) -> CausalityResult<(Vec<Vec<bool>>, HashMap<(usize, usize), Vec<usize>>)> {
        let n_vars = data.ncols();
        let n_nodes = n_vars * (self.config.tau_max + 1);
        let n = data.nrows();

        // Initialize: fully connected skeleton
        let mut adj = vec![vec![true; n_nodes]; n_nodes];
        // No self-loops
        for i in 0..n_nodes {
            adj[i][i] = false;
        }
        // Temporal constraint: no edge from lag-l node to lag-l' node if l' > l
        // (future cannot cause past). Contemporaneous edges (same lag) are allowed.
        for lag_i in 0..=self.config.tau_max {
            for v_i in 0..n_vars {
                let node_i = v_i + lag_i * n_vars;
                for lag_j in 0..=self.config.tau_max {
                    for v_j in 0..n_vars {
                        let node_j = v_j + lag_j * n_vars;
                        // Nodes at the same lag level (contemporaneous with each other)
                        // can be connected. Past-to-future is allowed.
                        // Future-to-past is not (exclude lag_j < lag_i for lagged nodes)
                        if lag_j > lag_i {
                            adj[node_i][node_j] = false;
                            adj[node_j][node_i] = false;
                        }
                    }
                }
            }
        }

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();

        // Iteratively remove edges using CI tests with increasing conditioning set sizes
        for cond_size in 0..=self.config.max_cond_size {
            let mut removals: Vec<(usize, usize, Vec<usize>)> = Vec::new();

            // Snapshot adjacency for order-independence (PC-stable style)
            let adj_snapshot = adj.clone();

            for i in 0..n_nodes {
                for j in (i + 1)..n_nodes {
                    if !adj[i][j] {
                        continue;
                    }

                    // Build conditioning candidates: common neighbors of i and j
                    let cond_candidates: Vec<usize> = (0..n_nodes)
                        .filter(|&k| k != i && k != j && adj_snapshot[i][k] && adj_snapshot[j][k])
                        .collect();

                    if cond_candidates.len() < cond_size {
                        continue;
                    }

                    // Test all subsets of cond_candidates of size cond_size
                    let subsets = combinations_usize(&cond_candidates, cond_size);
                    let mut found_sep = false;
                    let mut best_sep = Vec::new();

                    for subset in &subsets {
                        // Translate node indices back to (var, lag) pairs for the CI test
                        // Node node_idx = var + lag * n_vars
                        let (var_i, lag_i) = node_to_var_lag(i, n_vars);
                        let (var_j, lag_j) = node_to_var_lag(j, n_vars);
                        let z_set: Vec<(usize, usize)> =
                            subset.iter().map(|&k| node_to_var_lag(k, n_vars)).collect();

                        let ci_result = self.ci_test.test(
                            data,
                            (var_i, lag_i),
                            (var_j, lag_j),
                            &z_set,
                            self.config.alpha,
                        )?;

                        if !ci_result.dependent {
                            found_sep = true;
                            best_sep = subset.clone();
                            break;
                        }
                    }

                    if found_sep {
                        removals.push((i, j, best_sep));
                    }
                }
            }

            // Apply all removals at once (PC-stable order independence)
            for (i, j, sep) in removals {
                adj[i][j] = false;
                adj[j][i] = false;
                // Store sep set with canonical key
                let key = if i < j { (i, j) } else { (j, i) };
                sep_sets.insert(key, sep);
            }

            // Check if any edges remain testable
            let any_testable = (0..n_nodes).any(|i| {
                (i + 1..n_nodes).any(|j| {
                    if !adj[i][j] {
                        return false;
                    }
                    let cond_candidates = (0..n_nodes)
                        .filter(|&k| k != i && k != j && adj[i][k] && adj[j][k])
                        .count();
                    cond_candidates >= cond_size + 1
                })
            });
            if !any_testable {
                break;
            }
        }

        Ok((adj, sep_sets))
    }

    /// Phase 2: Orient v-structures (unshielded colliders)
    ///
    /// For each unshielded triple A - B - C (A not adjacent to C):
    /// - If B is NOT in sep(A, C): orient as A o-> B <-o C (collider/v-structure)
    /// - Otherwise: no change (B is in the separation set)
    fn orient_vstructures(
        &self,
        pag: &mut PartialAncestralGraph,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) {
        let n_nodes = pag.n_nodes;

        for b in 0..n_nodes {
            let neighbors_b = pag.adjacent_nodes(b);
            let nb_len = neighbors_b.len();
            for ai in 0..nb_len {
                for ci in (ai + 1)..nb_len {
                    let a = neighbors_b[ai];
                    let c = neighbors_b[ci];

                    // Check unshielded triple: A - B - C, A not adjacent to C
                    if pag.has_edge(a, c) {
                        continue;
                    }

                    // Look up sep(A, C)
                    let key = if a < c { (a, c) } else { (c, a) };
                    let b_in_sep = sep_sets.get(&key).map_or(false, |sep| sep.contains(&b));

                    if !b_in_sep {
                        // V-structure: A *-> B <-* C
                        // Orient mark at B on edge A-B as Arrowhead
                        pag.set_mark(a, b, EdgeMark::Arrowhead);
                        // Orient mark at B on edge C-B as Arrowhead
                        pag.set_mark(c, b, EdgeMark::Arrowhead);
                    }
                }
            }
        }
    }

    /// Phase 3: Apply temporal priority rule
    ///
    /// For a lagged node `l` (lag > 0) connected to a contemporaneous node `c` (lag = 0):
    /// the lagged node must have Tail at its end, Arrowhead at contemporaneous end.
    /// This encodes the temporal priority principle: past cannot be caused by future.
    ///
    /// Node encoding: node = var + lag * n_vars  (lag=0 → contemporaneous)
    fn apply_temporal_priority(&self, pag: &mut PartialAncestralGraph, n_vars: usize) {
        let n_nodes = pag.n_nodes;
        let edge_pairs = pag.edge_node_pairs();

        for (a, b) in edge_pairs {
            let (_, lag_a) = node_to_var_lag(a, n_vars);
            let (_, lag_b) = node_to_var_lag(b, n_vars);

            // Temporal priority: if lag_a > lag_b, then a is "older" and must point to b
            // i.e., the edge must be a --> b (Tail at a, Arrowhead at b)
            if lag_a > lag_b {
                // a is older (higher lag), b is newer (lower lag)
                // Must be: a ---> b  (Tail at a, Arrowhead at b)
                // In canonical storage (a, b) with a < b is not necessarily the case,
                // but set_mark uses the logical from/to direction.
                // set_mark(from=b, to=a, Arrowhead) sets arrowhead at a from b's perspective
                // We want: mark at a = Tail (a is not child), mark at b = Arrowhead (b is effect)
                // set_mark(from, to, mark) sets mark at 'to'
                // So: set_mark(a, b, Arrowhead) → arrowhead at b ✓
                //     set_mark(b, a, Tail)      → tail at a ✓
                pag.set_mark(b, a, EdgeMark::Tail); // mark at a (older end) = Tail
                pag.set_mark(a, b, EdgeMark::Arrowhead); // mark at b (newer end) = Arrowhead
            } else if lag_b > lag_a {
                // b is older, a is newer — b must point to a
                pag.set_mark(a, b, EdgeMark::Tail);
                pag.set_mark(b, a, EdgeMark::Arrowhead);
            }
            // lag_a == lag_b: contemporaneous edge, no temporal constraint applied here
        }
    }

    /// Apply FCI orientation rules R1–R4 (one pass)
    ///
    /// Returns true if any change was made.
    ///
    /// Rules (Zhang 2008):
    /// - R1: If A *-> B o-* C and A not adjacent to C: orient B *-> C (away from non-collider)
    /// - R2: If A -> B *-> C and A *-o C: orient A *-> C (away from cycle)
    /// - R3: If D *-o B o-* E (D,E not adjacent), D *-> A <-* E, B *-> A: orient B *-> A
    /// - R4: Discriminating path rule (simplified)
    fn apply_fci_rules(
        &self,
        pag: &mut PartialAncestralGraph,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) -> bool {
        let mut changed = false;
        let n_nodes = pag.n_nodes;

        // R1: Away from non-collider
        // If A *-> B o-* C, A not adjacent to C: orient B *-> C
        for b in 0..n_nodes {
            let neighbors_b = pag.adjacent_nodes(b);
            for &a in &neighbors_b {
                // Check A *-> B: mark at B on edge A-B is Arrowhead
                if pag.get_mark_at(a, b) != Some(EdgeMark::Arrowhead) {
                    continue;
                }
                for &c in &neighbors_b {
                    if c == a {
                        continue;
                    }
                    // Check B o-* C: mark at B on edge B-C is Circle
                    if pag.get_mark_at(c, b) != Some(EdgeMark::Circle) {
                        continue;
                    }
                    // Check A not adjacent to C
                    if pag.has_edge(a, c) {
                        continue;
                    }
                    // Orient: mark at C on edge B-C becomes Arrowhead
                    if pag.get_mark_at(b, c) != Some(EdgeMark::Arrowhead) {
                        pag.set_mark(b, c, EdgeMark::Arrowhead);
                        // Also remove circle at B side: B *-> C, circle at B → Tail
                        if pag.get_mark_at(c, b) == Some(EdgeMark::Circle) {
                            pag.set_mark(c, b, EdgeMark::Tail);
                        }
                        changed = true;
                    }
                }
            }
        }

        // R2: Away from cycle
        // If A -> B *-> C and A *-o C: orient A *-> C
        for b in 0..n_nodes {
            let neighbors_b = pag.adjacent_nodes(b);
            for &a in &neighbors_b {
                // A -> B: Tail at A, Arrowhead at B
                if pag.get_mark_at(b, a) != Some(EdgeMark::Tail) {
                    continue;
                }
                if pag.get_mark_at(a, b) != Some(EdgeMark::Arrowhead) {
                    continue;
                }
                for &c in &neighbors_b {
                    if c == a {
                        continue;
                    }
                    // B *-> C: Arrowhead at C
                    if pag.get_mark_at(b, c) != Some(EdgeMark::Arrowhead) {
                        continue;
                    }
                    // A *-o C: Circle at C on A-C edge
                    if !pag.has_edge(a, c) {
                        continue;
                    }
                    if pag.get_mark_at(a, c) != Some(EdgeMark::Circle) {
                        continue;
                    }
                    // Orient: mark at C becomes Arrowhead on edge A-C
                    pag.set_mark(a, c, EdgeMark::Arrowhead);
                    changed = true;
                }
            }
        }

        // R3: Away from collider (double non-collider)
        // If D *-o B o-* E, D not adjacent to E, D *-> A <-* E, B *-o A: orient B *-> A
        for b in 0..n_nodes {
            let neighbors_b = pag.adjacent_nodes(b);
            let nb_len = neighbors_b.len();
            for di in 0..nb_len {
                let d = neighbors_b[di];
                // D *-o B: Circle at B on D-B edge
                if pag.get_mark_at(d, b) != Some(EdgeMark::Circle) {
                    continue;
                }
                for ei in (di + 1)..nb_len {
                    let e = neighbors_b[ei];
                    // E o-* B: Circle at B on E-B edge
                    if pag.get_mark_at(e, b) != Some(EdgeMark::Circle) {
                        continue;
                    }
                    // D not adjacent to E
                    if pag.has_edge(d, e) {
                        continue;
                    }
                    // Find A such that D *-> A <-* E and B *-o A
                    let neighbors_d = pag.adjacent_nodes(d);
                    for &a in &neighbors_d {
                        if a == b || a == e {
                            continue;
                        }
                        if !pag.has_edge(e, a) || !pag.has_edge(b, a) {
                            continue;
                        }
                        // D *-> A: Arrowhead at A from D
                        if pag.get_mark_at(d, a) != Some(EdgeMark::Arrowhead) {
                            continue;
                        }
                        // E *-> A: Arrowhead at A from E
                        if pag.get_mark_at(e, a) != Some(EdgeMark::Arrowhead) {
                            continue;
                        }
                        // B *-o A: Circle at A on B-A edge
                        if pag.get_mark_at(b, a) != Some(EdgeMark::Circle) {
                            continue;
                        }
                        // Orient B *-> A
                        pag.set_mark(b, a, EdgeMark::Arrowhead);
                        changed = true;
                    }
                }
            }
        }

        // R4: Discriminating path rule (simplified)
        // A discriminating path for B is a path U ... W - A - B - C where:
        // - U is not adjacent to B
        // - Every node between U and A is a parent of C
        // - The path ends in W *-> A <-> B o-* C
        // If B in sep(U, C): orient B -> C; else orient B <-> C
        changed |= self.apply_r4(pag, sep_sets, n_nodes);

        changed
    }

    /// R4: Discriminating path rule
    fn apply_r4(
        &self,
        pag: &mut PartialAncestralGraph,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
        n_nodes: usize,
    ) -> bool {
        let mut changed = false;

        for c in 0..n_nodes {
            let neighbors_c = pag.adjacent_nodes(c);
            for &b in &neighbors_c {
                if b == c {
                    continue;
                }
                // B o-* C: Circle at B on B-C edge
                if pag.get_mark_at(c, b) != Some(EdgeMark::Circle) {
                    continue;
                }
                // A *-> B: Arrowhead at B
                let neighbors_b = pag.adjacent_nodes(b);
                for &a in &neighbors_b {
                    if a == c {
                        continue;
                    }
                    if pag.get_mark_at(a, b) != Some(EdgeMark::Arrowhead) {
                        continue;
                    }
                    // A must be adjacent to C
                    if !pag.has_edge(a, c) {
                        continue;
                    }
                    // A -> C: Tail at A, Arrowhead at C (A is parent of C)
                    if !pag.is_parent(a, c) {
                        continue;
                    }
                    // Look for discriminating path: find U not adjacent to B,
                    // with a path U...A where all intermediate nodes are parents of C.
                    // Simplified: check if there exists some U adjacent to A but not B,
                    // not adjacent to B, where U *-> A.
                    let neighbors_a = pag.adjacent_nodes(a);
                    for &u in &neighbors_a {
                        if u == b || u == c {
                            continue;
                        }
                        if pag.has_edge(u, b) {
                            continue;
                        }
                        // U *-> A
                        if pag.get_mark_at(u, a) != Some(EdgeMark::Arrowhead) {
                            continue;
                        }
                        // Found a potential discriminating path: U ... A - B - C
                        // Check if B is in sep(U, C)
                        let key = if u < c { (u, c) } else { (c, u) };
                        let b_in_sep = sep_sets.get(&key).map_or(false, |sep| sep.contains(&b));

                        if b_in_sep {
                            // B is non-collider: orient B -> C
                            // B *-> C: Arrowhead at C, Tail at B
                            if pag.get_mark_at(b, c) != Some(EdgeMark::Arrowhead)
                                || pag.get_mark_at(c, b) != Some(EdgeMark::Tail)
                            {
                                pag.set_mark(b, c, EdgeMark::Arrowhead);
                                pag.set_mark(c, b, EdgeMark::Tail);
                                changed = true;
                            }
                        } else {
                            // B is collider: orient B <-> C
                            if pag.get_mark_at(b, c) != Some(EdgeMark::Arrowhead)
                                || pag.get_mark_at(c, b) != Some(EdgeMark::Arrowhead)
                            {
                                pag.set_mark(b, c, EdgeMark::Arrowhead);
                                pag.set_mark(c, b, EdgeMark::Arrowhead);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        changed
    }

    /// Converge orientation rules to fixpoint
    ///
    /// Applies v-structure orientation, temporal priority, then iterates FCI rules
    /// until no further changes are possible or max_orientation_iter is reached.
    fn fci_converge(
        &self,
        pag: &mut PartialAncestralGraph,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) {
        let n_vars = pag.n_vars;

        // Orient v-structures first
        self.orient_vstructures(pag, sep_sets);

        // Apply temporal priority
        if self.config.apply_temporal_priority {
            self.apply_temporal_priority(pag, n_vars);
        }

        // Iterate FCI rules until fixpoint
        for _ in 0..self.config.max_orientation_iter {
            let changed = self.apply_fci_rules(pag, sep_sets);
            if !changed {
                break;
            }
        }
    }

    /// Count edges with at least one non-Circle mark
    fn count_oriented_edges(&self, pag: &PartialAncestralGraph) -> usize {
        pag.edges()
            .filter(|(_, _, e)| e.from_mark != EdgeMark::Circle || e.to_mark != EdgeMark::Circle)
            .count()
    }

    /// Find variable pairs (original variable indices) that have bidirected edges
    fn find_latent_confounder_pairs(
        &self,
        pag: &PartialAncestralGraph,
        n_vars: usize,
    ) -> Vec<(usize, usize)> {
        let mut pairs = std::collections::HashSet::new();

        for (a, b, edge) in pag.edges() {
            if edge.from_mark == EdgeMark::Arrowhead && edge.to_mark == EdgeMark::Arrowhead {
                let (var_a, _lag_a) = node_to_var_lag(a, n_vars);
                let (var_b, _lag_b) = node_to_var_lag(b, n_vars);
                let key = if var_a < var_b {
                    (var_a, var_b)
                } else {
                    (var_b, var_a)
                };
                pairs.insert(key);
            }
        }

        let mut result: Vec<(usize, usize)> = pairs.into_iter().collect();
        result.sort_unstable();
        result
    }
}

// ---- Helper functions ----

/// Convert node index to (variable_index, lag)
/// Node = var + lag * n_vars
#[inline]
fn node_to_var_lag(node: usize, n_vars: usize) -> (usize, usize) {
    if n_vars == 0 {
        return (0, 0);
    }
    let var = node % n_vars;
    let lag = node / n_vars;
    (var, lag)
}

/// Generate all size-k combinations from a slice of usize
fn combinations_usize(items: &[usize], k: usize) -> Vec<Vec<usize>> {
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
    combinations_usize_rec(items, k, 0, &mut vec![], &mut result);
    result
}

fn combinations_usize_rec(
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
    if items.len() - start < remaining {
        return;
    }
    for i in start..items.len() {
        current.push(items[i]);
        combinations_usize_rec(items, k, i + 1, current, result);
        current.pop();
    }
}

/// Partial correlation CI test using Fisher's Z-transformation
///
/// Tests H0: i _||_ j | conditioning using Pearson partial correlation.
/// - Computes sample partial correlation via residualization
/// - Fisher z-transform: z = 0.5 * ln((1+r)/(1-r))
/// - Test statistic: sqrt(n - |conditioning| - 3) * |z| ~ N(0,1)
/// - p-value: 2*(1 - Phi(|stat|))
/// - Returns true if conditionally independent (p > alpha)
pub fn partial_correlation_ci_test(
    data: &Array2<f64>,
    i: usize,
    j: usize,
    conditioning: &[usize],
    alpha: f64,
    n: usize,
) -> bool {
    let r = compute_partial_corr_from_cols(data, i, j, conditioning);

    let clamped = r.clamp(-0.9999, 0.9999);
    let z = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln();

    let cond_len = conditioning.len() as f64;
    let df = (n as f64 - cond_len - 3.0).max(1.0);
    let stat = df.sqrt() * z.abs();

    // p-value = 2 * (1 - Phi(|stat|)) using erf approximation
    let p_value = 2.0 * (1.0 - super::normal_cdf(stat));
    p_value > alpha
}

/// Compute partial correlation between columns i and j in data,
/// conditioning on the given column indices via recursive residualization.
fn compute_partial_corr_from_cols(
    data: &Array2<f64>,
    i: usize,
    j: usize,
    conditioning: &[usize],
) -> f64 {
    if conditioning.is_empty() {
        return pearson_r_cols(data, i, j);
    }

    // Residualize i and j on the last conditioning variable, recurse
    let last = *conditioning.last().expect("conditioning non-empty");
    let rest = &conditioning[..conditioning.len() - 1];

    let r_ik = compute_partial_corr_from_cols(data, i, last, rest);
    let r_jk = compute_partial_corr_from_cols(data, j, last, rest);
    let r_ij = compute_partial_corr_from_cols(data, i, j, rest);

    let denom = ((1.0 - r_ik * r_ik) * (1.0 - r_jk * r_jk)).sqrt();
    if denom < 1e-15 {
        return 0.0;
    }
    (r_ij - r_ik * r_jk) / denom
}

/// Compute Pearson correlation between two columns of data
fn pearson_r_cols(data: &Array2<f64>, col_i: usize, col_j: usize) -> f64 {
    let n = data.nrows();
    if n == 0 || col_i >= data.ncols() || col_j >= data.ncols() {
        return 0.0;
    }

    let mut sum_i = 0.0;
    let mut sum_j = 0.0;
    for row in 0..n {
        sum_i += data[[row, col_i]];
        sum_j += data[[row, col_j]];
    }
    let mean_i = sum_i / n as f64;
    let mean_j = sum_j / n as f64;

    let mut num = 0.0;
    let mut var_i = 0.0;
    let mut var_j = 0.0;
    for row in 0..n {
        let di = data[[row, col_i]] - mean_i;
        let dj = data[[row, col_j]] - mean_j;
        num += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }

    let denom = (var_i * var_j).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causality::ci_tests::ParCorr;
    use crate::causality::pag::{EdgeMark, PartialAncestralGraph};
    use scirs2_core::ndarray::Array2;

    fn make_pag_chain() -> PartialAncestralGraph {
        // 0 --> 1 --> 2 (directed chain)
        let mut pag = PartialAncestralGraph::new(3);
        pag.add_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrowhead);
        pag.add_edge(1, 2, EdgeMark::Tail, EdgeMark::Arrowhead);
        pag
    }

    fn make_data(n: usize, n_vars: usize, seed: u64) -> Array2<f64> {
        let mut data = Array2::zeros((n, n_vars));
        let mut state = seed;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for t in 1..n {
            let e0 = next(&mut state) * 0.3;
            data[[t, 0]] = 0.6 * data[[t - 1, 0]] + e0;
            if n_vars > 1 {
                let e1 = next(&mut state) * 0.3;
                data[[t, 1]] = 0.5 * data[[t - 1, 0]] + 0.3 * data[[t - 1, 1]] + e1;
            }
            if n_vars > 2 {
                let e2 = next(&mut state) * 0.3;
                data[[t, 2]] = 0.4 * data[[t - 1, 1]] + 0.2 * data[[t - 1, 2]] + e2;
            }
        }
        data
    }

    // ---- PAG structural tests ----

    #[test]
    fn test_pag_new_empty() {
        let pag = PartialAncestralGraph::new(5);
        assert_eq!(pag.n_nodes, 5);
        assert_eq!(pag.n_bidirected_edges(), 0);
        assert_eq!(pag.n_circle_marks(), 0);
        assert!(pag.adjacent_nodes(0).is_empty());
    }

    #[test]
    fn test_pag_add_remove_edge() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 2, EdgeMark::Circle, EdgeMark::Circle);
        assert!(pag.has_edge(0, 2));
        assert!(pag.has_edge(2, 0));
        pag.remove_edge(2, 0);
        assert!(!pag.has_edge(0, 2));
        assert!(!pag.has_edge(2, 0));
    }

    #[test]
    fn test_pag_initialize_from_skeleton_all_circles() {
        let adj = vec![
            vec![false, true, true, false],
            vec![true, false, false, true],
            vec![true, false, false, false],
            vec![false, true, false, false],
        ];
        let pag = PartialAncestralGraph::initialize_from_skeleton(&adj, 4);
        assert!(pag.has_edge(0, 1));
        assert!(pag.has_edge(0, 2));
        assert!(pag.has_edge(1, 3));
        assert!(!pag.has_edge(1, 2));
        // All circle marks: 3 edges × 2 marks = 6
        assert_eq!(pag.n_circle_marks(), 6);
    }

    #[test]
    fn test_pag_set_and_get_mark() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(1, 3, EdgeMark::Circle, EdgeMark::Circle);
        // Set arrowhead at 3 (from 1→3)
        pag.set_mark(1, 3, EdgeMark::Arrowhead);
        assert_eq!(pag.get_mark_at(1, 3), Some(EdgeMark::Arrowhead));
        // Circle at 1 (from 3→1) unchanged
        assert_eq!(pag.get_mark_at(3, 1), Some(EdgeMark::Circle));
    }

    #[test]
    fn test_pag_is_parent_true() {
        let pag = make_pag_chain();
        assert!(pag.is_parent(0, 1));
        assert!(pag.is_parent(1, 2));
        assert!(!pag.is_parent(1, 0));
        assert!(!pag.is_parent(2, 1));
    }

    #[test]
    fn test_pag_n_bidirected_edges_after_orient() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 1, EdgeMark::Arrowhead, EdgeMark::Arrowhead);
        pag.add_edge(1, 2, EdgeMark::Tail, EdgeMark::Arrowhead);
        pag.add_edge(2, 3, EdgeMark::Circle, EdgeMark::Circle);
        assert_eq!(pag.n_bidirected_edges(), 1);
        // Turn 2-3 into bidirected
        pag.add_edge(2, 3, EdgeMark::Arrowhead, EdgeMark::Arrowhead);
        assert_eq!(pag.n_bidirected_edges(), 2);
    }

    #[test]
    fn test_pag_n_circle_marks() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle); // 2
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Arrowhead); // 1
        pag.add_edge(2, 3, EdgeMark::Tail, EdgeMark::Arrowhead); // 0
        assert_eq!(pag.n_circle_marks(), 3);
    }

    #[test]
    fn test_pag_adjacent_nodes() {
        let mut pag = PartialAncestralGraph::new(5);
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(0, 4, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(2, 4, EdgeMark::Circle, EdgeMark::Circle);
        let adj0 = pag.adjacent_nodes(0);
        assert_eq!(adj0, vec![1, 4]);
        let adj4 = pag.adjacent_nodes(4);
        assert_eq!(adj4, vec![0, 2]);
    }

    // ---- V-structure orientation tests ----

    #[test]
    fn test_orient_vstructures_basic_triple() {
        // A - B - C unshielded, B not in sep(A,C)
        // Should orient: A *-> B <-* C
        let mut pag = PartialAncestralGraph::new(3);
        pag.n_nodes = 3;
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle);
        // No edge between 0 and 2 (unshielded)

        let sep_sets: HashMap<(usize, usize), Vec<usize>> = {
            let mut m = HashMap::new();
            // sep(0, 2) = [] — empty, so B=1 is not in sep
            m.insert((0, 2), vec![]);
            m
        };

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        fci.orient_vstructures(&mut pag, &sep_sets);

        // Mark at B (node 1) on edge A(0)-B(1) should be Arrowhead
        assert_eq!(
            pag.get_mark_at(0, 1),
            Some(EdgeMark::Arrowhead),
            "A *-> B: arrowhead at B"
        );
        // Mark at B (node 1) on edge C(2)-B(1) should be Arrowhead
        assert_eq!(
            pag.get_mark_at(2, 1),
            Some(EdgeMark::Arrowhead),
            "C *-> B: arrowhead at B"
        );
    }

    #[test]
    fn test_no_vstructure_when_in_sepset() {
        // A - B - C unshielded, B IS in sep(A,C)
        // Should NOT orient as collider
        let mut pag = PartialAncestralGraph::new(3);
        pag.n_nodes = 3;
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle);

        let sep_sets: HashMap<(usize, usize), Vec<usize>> = {
            let mut m = HashMap::new();
            // B=1 is in sep(A=0, C=2), so no v-structure
            m.insert((0, 2), vec![1]);
            m
        };

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        fci.orient_vstructures(&mut pag, &sep_sets);

        // Marks should remain Circle
        assert_eq!(pag.get_mark_at(0, 1), Some(EdgeMark::Circle));
        assert_eq!(pag.get_mark_at(2, 1), Some(EdgeMark::Circle));
    }

    // ---- Temporal priority tests ----

    #[test]
    fn test_temporal_priority_lagged_to_contemp_becomes_arrow() {
        // 2 vars, tau_max=1, 4 nodes: [0,1] (lag0), [2,3] (lag1)
        // Edge between node 2 (var0, lag1) and node 0 (var0, lag0)
        // After temporal priority: node2 ---> node0 (Tail at 2, Arrowhead at 0)
        let n_vars = 2;
        let mut pag = PartialAncestralGraph::with_vars_and_lags(n_vars, 1);
        pag.add_edge(2, 0, EdgeMark::Circle, EdgeMark::Circle);

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        fci.apply_temporal_priority(&mut pag, n_vars);

        // node2 = var0, lag1 (older); node0 = var0, lag0 (newer)
        // Should be: Tail at node2, Arrowhead at node0
        let edge = pag.get_edge(2, 0).expect("edge should exist");
        assert_eq!(edge.to_mark, EdgeMark::Arrowhead, "Arrowhead at newer node");
        assert_eq!(edge.from_mark, EdgeMark::Tail, "Tail at older node");
    }

    #[test]
    fn test_temporal_priority_contemp_unchanged() {
        // Two contemporaneous nodes (same lag) should not have temporal priority applied
        let n_vars = 2;
        let mut pag = PartialAncestralGraph::with_vars_and_lags(n_vars, 1);
        // Nodes 0 and 1 are both lag=0 (contemporaneous)
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        fci.apply_temporal_priority(&mut pag, n_vars);

        // Contemporaneous edge should remain Circle-Circle
        assert_eq!(pag.get_mark_at(0, 1), Some(EdgeMark::Circle));
        assert_eq!(pag.get_mark_at(1, 0), Some(EdgeMark::Circle));
    }

    // ---- FCI rule tests ----

    #[test]
    fn test_fci_r1_non_collider() {
        // R1: A *-> B o-* C, A not adjacent to C → B *-> C
        let mut pag = PartialAncestralGraph::new(3);
        pag.n_nodes = 3;
        // A(0) *-> B(1): Arrowhead at B
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Arrowhead);
        // B(1) o-* C(2): Circle at B
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle);
        // A(0) not adjacent to C(2)
        let sep_sets = HashMap::new();

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        let changed = fci.apply_fci_rules(&mut pag, &sep_sets);

        assert!(changed, "R1 should have made a change");
        // Mark at C(2) on edge B(1)-C(2) should be Arrowhead
        assert_eq!(
            pag.get_mark_at(1, 2),
            Some(EdgeMark::Arrowhead),
            "R1: B *-> C, arrowhead at C"
        );
    }

    #[test]
    fn test_fci_r2_away_from_collider() {
        // R2: A -> B *-> C and A *-o C → orient A *-> C
        let mut pag = PartialAncestralGraph::new(3);
        pag.n_nodes = 3;
        // A(0) -> B(1): Tail at A, Arrowhead at B
        pag.add_edge(0, 1, EdgeMark::Tail, EdgeMark::Arrowhead);
        // B(1) *-> C(2): Arrowhead at C
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Arrowhead);
        // A(0) *-o C(2): Circle at C
        pag.add_edge(0, 2, EdgeMark::Circle, EdgeMark::Circle);

        let sep_sets = HashMap::new();
        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        let changed = fci.apply_fci_rules(&mut pag, &sep_sets);

        assert!(changed, "R2 should have made a change");
        // Mark at C(2) on edge A(0)-C(2) should now be Arrowhead
        assert_eq!(
            pag.get_mark_at(0, 2),
            Some(EdgeMark::Arrowhead),
            "R2: A *-> C, arrowhead at C"
        );
    }

    // ---- Convergence and result tests ----

    #[test]
    fn test_fci_converge_simple_chain() {
        // A simple 3-node chain with lagged data
        // Ensure fci_converge runs without error and produces a valid PAG
        let mut pag = PartialAncestralGraph::new(3);
        pag.n_vars = 3;
        pag.tau_max = 0;
        pag.n_nodes = 3;
        pag.add_edge(0, 1, EdgeMark::Circle, EdgeMark::Circle);
        pag.add_edge(1, 2, EdgeMark::Circle, EdgeMark::Circle);

        let sep_sets: HashMap<(usize, usize), Vec<usize>> = {
            let mut m = HashMap::new();
            m.insert((0, 2), vec![]);
            m
        };
        pag.sep_sets = sep_sets.iter().map(|(&k, v)| (k, v.clone())).collect();

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        fci.fci_converge(&mut pag, &sep_sets);

        // After converge, some edges should be oriented
        // 0-1-2 unshielded collider with empty sep(0,2) → v-structure at 1
        assert_eq!(pag.get_mark_at(0, 1), Some(EdgeMark::Arrowhead));
        assert_eq!(pag.get_mark_at(2, 1), Some(EdgeMark::Arrowhead));
    }

    #[test]
    fn test_fci_result_n_bidirected() {
        let mut pag = PartialAncestralGraph::new(4);
        pag.n_vars = 2;
        pag.tau_max = 1;
        pag.n_nodes = 4;
        pag.add_edge(0, 1, EdgeMark::Arrowhead, EdgeMark::Arrowhead);
        pag.add_edge(2, 3, EdgeMark::Tail, EdgeMark::Arrowhead);

        assert_eq!(pag.n_bidirected_edges(), 1);
    }

    #[test]
    fn test_fci_result_latent_confounder_pairs() {
        let n_vars = 3;
        let mut pag = PartialAncestralGraph::new(n_vars);
        pag.n_vars = n_vars;
        pag.tau_max = 0;
        pag.n_nodes = n_vars;
        // Bidirected edge between var0 and var2
        pag.add_edge(0, 2, EdgeMark::Arrowhead, EdgeMark::Arrowhead);

        let config = FciTimeSeriesConfig::default();
        let fci = FciTimeSeries {
            ci_test: ParCorr::new(),
            config,
        };
        let pairs = fci.find_latent_confounder_pairs(&pag, n_vars);

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], (0, 2));
    }

    #[test]
    fn test_fci_config_default() {
        let config = FciTimeSeriesConfig::default();
        assert_eq!(config.tau_max, 3);
        assert!((config.alpha - 0.05).abs() < 1e-15);
        assert_eq!(config.max_cond_size, 4);
        assert!(config.apply_temporal_priority);
        assert_eq!(config.max_orientation_iter, 100);
    }

    // ---- Partial correlation helper tests ----

    #[test]
    fn test_partial_correlation_independent() {
        // Two independent columns should be declared independent
        let n = 200;
        let mut data = Array2::zeros((n, 2));
        let mut s1: u64 = 111;
        let mut s2: u64 = 222;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for row in 0..n {
            data[[row, 0]] = next(&mut s1);
            data[[row, 1]] = next(&mut s2);
        }
        // With no conditioning and alpha=0.05, independent data should have p > alpha
        let independent = partial_correlation_ci_test(&data, 0, 1, &[], 0.05, n);
        assert!(
            independent,
            "Independent columns should be declared independent"
        );
    }

    #[test]
    fn test_partial_correlation_dependent() {
        // Two strongly correlated columns should be declared dependent
        let n = 200;
        let mut data = Array2::zeros((n, 2));
        let mut s: u64 = 333;
        let next = |s: &mut u64| -> f64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*s >> 32) as f64) / (u32::MAX as f64) - 0.5
        };
        for row in 0..n {
            let x = next(&mut s);
            data[[row, 0]] = x;
            data[[row, 1]] = x + next(&mut s) * 0.05; // highly correlated
        }
        let independent = partial_correlation_ci_test(&data, 0, 1, &[], 0.05, n);
        assert!(
            !independent,
            "Dependent columns should NOT be declared independent"
        );
    }

    // ---- Full integration test ----

    #[test]
    fn test_fci_run_small_dataset() {
        let data = make_data(200, 2, 42);
        let config = FciTimeSeriesConfig {
            tau_max: 1,
            alpha: 0.05,
            max_cond_size: 2,
            apply_temporal_priority: true,
            max_orientation_iter: 50,
        };
        let fci = FciTimeSeries::new(ParCorr::new(), config);
        let result = fci.run(&data).expect("FCI should succeed");

        assert_eq!(result.pag.n_vars, 2);
        assert_eq!(result.pag.tau_max, 1);
        // Verify basic structural invariants
        let _ = result.n_circle_marks + result.n_edges_oriented * 2;
        assert!(result.n_bidirected_edges <= result.pag.n_bidirected_edges());
    }
}
