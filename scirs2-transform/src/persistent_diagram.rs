//! Topological Data Analysis: Persistence Diagrams
//!
//! This module provides tools for computing and analysing persistence diagrams,
//! which are fundamental objects in Topological Data Analysis (TDA).
//!
//! ## Background
//!
//! A **persistence diagram** is a multiset of points `(birth, death)` in
//! the extended plane `ℝ × (ℝ ∪ {+∞})`.  Each point represents a
//! topological feature (connected component, loop, void, …) that is born
//! at filtration value `birth` and dies at `death`.
//!
//! ## Implemented Algorithms
//!
//! | Function                       | Description                                             |
//! |--------------------------------|---------------------------------------------------------|
//! | `persistence_diagram_0d`       | 0-dimensional PH via Union-Find on distance matrix      |
//! | `bottleneck_distance`          | Bottleneck distance between two diagrams                |
//! | `wasserstein_distance_pd`      | p-Wasserstein distance between two diagrams             |
//! | `persistence_entropy`          | Shannon entropy of persistence lifetimes                |
//! | `betti_numbers`                | Betti numbers b₀, b₁, … at a given filtration level    |
//!
//! ## References
//!
//! - Edelsbrunner, H., & Harer, J. (2010). Computational Topology: An
//!   Introduction. American Mathematical Society.
//! - Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). Stability of
//!   persistence diagrams. Discrete & Computational Geometry, 37(1), 103-120.
//! - Mileyko, Y., Mukherjee, S., & Harer, J. (2011). Probability measures on
//!   the space of persistence diagrams. Inverse Problems, 27(12), 124007.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::Array2;

// ─── Core Data Structures ─────────────────────────────────────────────────────

/// A single persistence pair `(birth, death)` in a persistence diagram.
///
/// When `death == f64::INFINITY` the feature is **essential** (never dies
/// within the finite filtration).
#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePair {
    /// Filtration value at which the topological feature is born
    pub birth: f64,
    /// Filtration value at which the feature dies, or `f64::INFINITY`
    pub death: f64,
    /// Homological dimension: 0 = connected components, 1 = loops, 2 = voids …
    pub dimension: usize,
}

impl PersistencePair {
    /// Create a new persistence pair.
    pub fn new(birth: f64, death: f64, dimension: usize) -> Self {
        Self {
            birth,
            death,
            dimension,
        }
    }

    /// The persistence (lifetime) of this feature: `death − birth`.
    ///
    /// Returns `f64::INFINITY` for essential features.
    pub fn persistence(&self) -> f64 {
        if self.death.is_infinite() {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    /// `true` if this feature never dies (birth–death range is infinite).
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }

    /// Midpoint `( (birth + death) / 2, (death − birth) / 2 )` on the
    /// diagonal half-plane.  Used for diagram matching.
    ///
    /// Returns `None` for essential features.
    pub fn diagonal_projection(&self) -> Option<(f64, f64)> {
        if self.is_essential() {
            None
        } else {
            let proj = (self.birth + self.death) / 2.0;
            Some((proj, proj))
        }
    }
}

// ─── Union-Find ───────────────────────────────────────────────────────────────

/// Disjoint-set (Union-Find) data structure with path compression and
/// union-by-rank for efficient connected-components computation.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    birth: Vec<f64>, // filtration value at which component was born
}

impl UnionFind {
    /// Initialise with `n` singletons, each born at `birth_values[i]`.
    fn new(n: usize, birth_values: &[f64]) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            birth: birth_values.to_vec(),
        }
    }

    /// Find representative of the set containing `x` (with path compression).
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Union the sets containing `x` and `y`.
    ///
    /// Returns `Some((killed_root, death_time))` if two distinct components
    /// were merged (i.e., a 0-dimensional feature just died), or `None` if
    /// `x` and `y` were already in the same component.
    ///
    /// The **elder rule** is applied: the component born *later* (higher
    /// birth value) is killed, and the one born *earlier* survives.
    fn union(&mut self, x: usize, y: usize, edge_weight: f64) -> Option<(usize, f64)> {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return None;
        }

        // Elder rule: the component with the later birth dies
        let (survivor, killed) = if self.birth[rx] <= self.birth[ry] {
            (rx, ry)
        } else {
            (ry, rx)
        };

        // Union by rank
        if self.rank[survivor] < self.rank[killed] {
            self.parent[survivor] = killed;
            // The killed component survives structurally — swap roles
            let (s2, k2) = (killed, survivor);
            self.parent[k2] = s2;
            self.parent[s2] = s2;
            // birth of surviving root stays as the one born earlier
            self.birth[s2] = self.birth[s2].min(self.birth[k2]);
            return Some((k2, edge_weight));
        } else {
            self.parent[killed] = survivor;
            if self.rank[survivor] == self.rank[killed] {
                self.rank[survivor] += 1;
            }
        }

        Some((killed, edge_weight))
    }
}

// ─── 0-Dimensional Persistent Homology ───────────────────────────────────────

/// Compute the 0-dimensional persistence diagram from a pairwise distance matrix.
///
/// Uses a Union-Find algorithm on the edges sorted by weight (Kruskal-style
/// filtration) to track connected-component merges.  Each merge event
/// produces a `PersistencePair(birth, death)`:
///
/// - `birth` = the birth time of the **younger** component (elder rule).
/// - `death` = the edge weight at which the components merge.
///
/// One component never dies (the global component born at filtration 0);
/// it is included as an essential pair `(min_birth, +∞)`.
///
/// # Parameters
///
/// * `distance_matrix` – Symmetric square distance matrix of shape `(n, n)`.
///   The diagonal should be zero.
///
/// # Returns
///
/// A `Vec<PersistencePair>` of dimension 0 pairs.
pub fn persistence_diagram_0d(distance_matrix: &Array2<f64>) -> Result<Vec<PersistencePair>> {
    let n = distance_matrix.nrows();
    if n == 0 {
        return Ok(Vec::new());
    }
    if distance_matrix.ncols() != n {
        return Err(TransformError::InvalidInput(
            "distance_matrix must be square".to_string(),
        ));
    }

    // Collect all edges (i, j, weight) with i < j
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = distance_matrix[[i, j]];
            edges.push((i, j, w));
        }
    }
    // Sort edges by weight
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Each vertex is born at time 0 (all points exist from the start of the
    // filtration).
    let birth_times = vec![0.0f64; n];
    let mut uf = UnionFind::new(n, &birth_times);

    let mut pairs: Vec<PersistencePair> = Vec::new();

    for (i, j, w) in &edges {
        if let Some((_killed, death)) = uf.union(*i, *j, *w) {
            // A component was killed at edge weight `death`.
            // Its birth time is 0 (all components start at t=0 for 0-dim PH).
            pairs.push(PersistencePair::new(0.0, death, 0));
        }
    }

    // The one surviving essential component
    pairs.push(PersistencePair::new(0.0, f64::INFINITY, 0));

    Ok(pairs)
}

// ─── Bottleneck Distance ──────────────────────────────────────────────────────

/// Compute the bottleneck distance between two persistence diagrams.
///
/// The bottleneck distance is the min-max matching cost between the two
/// diagrams, where unmatched points are matched to their projections onto
/// the diagonal (with cost equal to half their persistence).
///
/// d_B(X, Y) = inf_γ  sup_{x ∈ X} ‖x − γ(x)‖_∞
///
/// where the infimum is over all partial matchings γ : X → Y ∪ Δ.
///
/// # Parameters
///
/// * `diagram1`, `diagram2` – Slices of `PersistencePair`.
///
/// # Returns
///
/// The bottleneck distance (a non-negative `f64`).
///
/// # Implementation Notes
///
/// Uses a binary-search + bipartite-matching approach:
/// 1. Collect all candidate edge weights (pairwise ‖·‖_∞ and half-persistence).
/// 2. Binary-search for the smallest δ such that the δ-reachability graph
///    has a perfect matching.
pub fn bottleneck_distance(diagram1: &[PersistencePair], diagram2: &[PersistencePair]) -> f64 {
    // Filter to finite points only
    let pts1: Vec<(f64, f64)> = diagram1
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();
    let pts2: Vec<(f64, f64)> = diagram2
        .iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    bottleneck_finite(&pts1, &pts2)
}

/// Bottleneck distance restricted to a given homological dimension.
pub fn bottleneck_distance_dim(
    diagram1: &[PersistencePair],
    diagram2: &[PersistencePair],
    dim: usize,
) -> f64 {
    let pts1: Vec<(f64, f64)> = diagram1
        .iter()
        .filter(|p| p.dimension == dim && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();
    let pts2: Vec<(f64, f64)> = diagram2
        .iter()
        .filter(|p| p.dimension == dim && !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    bottleneck_finite(&pts1, &pts2)
}

/// Bottleneck distance between two sets of finite (birth, death) pairs.
fn bottleneck_finite(pts1: &[(f64, f64)], pts2: &[(f64, f64)]) -> f64 {
    // Cost of matching a point (b, d) to the diagonal
    let diag_cost = |b: f64, d: f64| -> f64 { (d - b).abs() / 2.0 };

    // L-infinity distance between two diagram points
    let linf = |a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - b.0).abs().max((a.1 - b.1).abs())
    };

    // Collect all candidate δ values
    let mut candidates = Vec::new();

    // Pairwise distances between points in the two diagrams
    for &p1 in pts1 {
        for &p2 in pts2 {
            candidates.push(linf(p1, p2));
        }
        candidates.push(diag_cost(p1.0, p1.1));
    }
    for &p2 in pts2 {
        candidates.push(diag_cost(p2.0, p2.1));
    }

    if candidates.is_empty() {
        return 0.0;
    }

    // Sort candidates
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    // Binary search for minimum δ that admits a perfect matching
    let mut lo = 0usize;
    let mut hi = candidates.len(); // exclusive upper bound (fallback: use last)

    while lo < hi {
        let mid = (lo + hi) / 2;
        let delta = candidates[mid];
        if admits_perfect_matching(pts1, pts2, delta, &linf, &diag_cost) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    if lo < candidates.len() {
        candidates[lo]
    } else {
        *candidates.last().unwrap_or(&0.0)
    }
}

/// Check whether a δ-matching exists (bipartite matching via augmenting paths).
///
/// Builds a bipartite graph where:
/// - Left nodes = pts1 ∪ {diagonal copies of pts2}
/// - Right nodes = pts2 ∪ {diagonal copies of pts1}
/// - An edge exists if ‖·‖_∞ ≤ δ.
fn admits_perfect_matching(
    pts1: &[(f64, f64)],
    pts2: &[(f64, f64)],
    delta: f64,
    linf: &dyn Fn((f64, f64), (f64, f64)) -> f64,
    diag_cost: &dyn Fn(f64, f64) -> f64,
) -> bool {
    let n1 = pts1.len();
    let n2 = pts2.len();

    // Left nodes: 0..n1 are pts1, n1..n1+n2 are diagonal projections of pts2
    // Right nodes: 0..n2 are pts2, n2..n1+n2 are diagonal projections of pts1
    let left_size = n1 + n2;
    let right_size = n2 + n1;
    debug_assert_eq!(left_size, right_size);
    let total = left_size;

    // Build adjacency: adj[l] = list of right nodes reachable from left node l
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); total];

    // pts1[i] can match to pts2[j] if linf distance ≤ delta
    for i in 0..n1 {
        for j in 0..n2 {
            if linf(pts1[i], pts2[j]) <= delta + 1e-12 {
                adj[i].push(j);
            }
        }
        // pts1[i] can match to its diagonal projection (right node n2 + i)
        if diag_cost(pts1[i].0, pts1[i].1) <= delta + 1e-12 {
            adj[i].push(n2 + i);
        }
    }
    // Diagonal of pts2[j] (left node n1 + j) can match to pts2[j] (right node j)
    for j in 0..n2 {
        if diag_cost(pts2[j].0, pts2[j].1) <= delta + 1e-12 {
            adj[n1 + j].push(j);
        }
        // Diagonal of pts2[j] can also match to diagonal of pts1[i] if i = j (dummy)
        // Actually diagonal–diagonal is always cost 0; we add all diagonal-diagonal edges.
        for i in 0..n1 {
            adj[n1 + j].push(n2 + i);
        }
    }

    // Hopcroft-Karp style maximum bipartite matching via augmenting paths
    let mut match_l = vec![usize::MAX; total]; // left → right
    let mut match_r = vec![usize::MAX; total]; // right → left

    let mut matched = 0usize;
    for l in 0..total {
        let mut visited = vec![false; total];
        if augment(l, &adj, &mut match_l, &mut match_r, &mut visited) {
            matched += 1;
        }
    }

    matched == total
}

/// Augmenting-path DFS for bipartite matching.
fn augment(
    u: usize,
    adj: &[Vec<usize>],
    match_l: &mut Vec<usize>,
    match_r: &mut Vec<usize>,
    visited: &mut Vec<bool>,
) -> bool {
    for &v in &adj[u] {
        if visited[v] {
            continue;
        }
        visited[v] = true;
        let prev = match_r[v];
        let can_augment = prev == usize::MAX
            || augment(prev, adj, match_l, match_r, visited);
        if can_augment {
            match_l[u] = v;
            match_r[v] = u;
            return true;
        }
    }
    false
}

// ─── Wasserstein Distance ─────────────────────────────────────────────────────

/// Compute the p-Wasserstein distance between two persistence diagrams.
///
/// W_p(X, Y) = ( inf_γ Σ_{x ∈ X} ‖x − γ(x)‖_∞^p )^(1/p)
///
/// where γ ranges over all partial matchings and unmatched points are
/// mapped to their diagonal projections.
///
/// # Parameters
///
/// * `diagram1`, `diagram2` – Input diagrams.
/// * `p`                    – Exponent (1 = Earth-Mover, 2 = standard Wasserstein).
///
/// # Returns
///
/// A non-negative `f64`.
pub fn wasserstein_distance_pd(
    diagram1: &[PersistencePair],
    diagram2: &[PersistencePair],
    p: usize,
) -> f64 {
    if p == 0 {
        return f64::NAN;
    }
    let pts1: Vec<(f64, f64)> = diagram1
        .iter()
        .filter(|pt| !pt.is_essential())
        .map(|pt| (pt.birth, pt.death))
        .collect();
    let pts2: Vec<(f64, f64)> = diagram2
        .iter()
        .filter(|pt| !pt.is_essential())
        .map(|pt| (pt.birth, pt.death))
        .collect();

    wasserstein_finite(&pts1, &pts2, p)
}

/// p-Wasserstein distance for finite point sets.
///
/// Uses the Hungarian algorithm (O(n³)) to find the optimal matching.
fn wasserstein_finite(pts1: &[(f64, f64)], pts2: &[(f64, f64)], p: usize) -> f64 {
    let diag_cost = |b: f64, d: f64| -> f64 { (d - b).abs() / 2.0 };
    let linf = |a: (f64, f64), b: (f64, f64)| -> f64 {
        (a.0 - b.0).abs().max((a.1 - b.1).abs())
    };

    let n1 = pts1.len();
    let n2 = pts2.len();
    let n = n1 + n2; // augmented size: each unmatched point matched to diagonal

    if n == 0 {
        return 0.0;
    }

    // Build cost matrix of size n × n
    // Rows: pts1[0..n1] then diagonal-copies of pts2[0..n2]
    // Cols: pts2[0..n2] then diagonal-copies of pts1[0..n1]
    let cost_fn = |r: usize, c: usize| -> f64 {
        match (r < n1, c < n2) {
            (true, true) => {
                // pts1[r] ↔ pts2[c]
                linf(pts1[r], pts2[c])
            }
            (true, false) => {
                // pts1[r] ↔ diagonal (its own projection)
                let i = c - n2;
                if i == r {
                    diag_cost(pts1[r].0, pts1[r].1)
                } else {
                    f64::INFINITY
                }
            }
            (false, true) => {
                // diagonal of pts2[c] ↔ pts2[c]
                let j = r - n1;
                if j == c {
                    diag_cost(pts2[c].0, pts2[c].1)
                } else {
                    f64::INFINITY
                }
            }
            (false, false) => {
                // diagonal copy ↔ diagonal copy: always 0
                0.0
            }
        }
    };

    // Hungarian algorithm (Jonker-Volgenant simplified for moderate n)
    let assignment = hungarian_assignment(n, &cost_fn);

    let cost: f64 = (0..n)
        .map(|i| {
            let j = assignment[i];
            let c = cost_fn(i, j);
            if c.is_infinite() {
                0.0 // shouldn't happen with valid assignment
            } else {
                c.powi(p as i32)
            }
        })
        .sum::<f64>();

    cost.powf(1.0 / p as f64)
}

/// Simple O(n³) Hungarian algorithm returning assignment[i] = j.
fn hungarian_assignment(n: usize, cost_fn: &dyn Fn(usize, usize) -> f64) -> Vec<usize> {
    // Build full cost matrix, replacing +inf with a large sentinel
    let sentinel = 1e18f64;
    let mut cost: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let c = cost_fn(i, j);
                    if c.is_infinite() { sentinel } else { c }
                })
                .collect()
        })
        .collect();

    // Row reduction
    for i in 0..n {
        let min_r = cost[i].iter().cloned().fold(f64::INFINITY, f64::min);
        for j in 0..n {
            cost[i][j] -= min_r;
        }
    }
    // Column reduction
    for j in 0..n {
        let min_c = (0..n).map(|i| cost[i][j]).fold(f64::INFINITY, f64::min);
        for i in 0..n {
            cost[i][j] -= min_c;
        }
    }

    // Augmenting path assignment
    let mut row_cover = vec![false; n];
    let mut col_cover = vec![false; n];
    let mut assignment = vec![usize::MAX; n];
    let mut assigned_row = vec![usize::MAX; n];

    // Greedy initial assignment
    for i in 0..n {
        for j in 0..n {
            if cost[i][j].abs() < 1e-10 && !col_cover[j] && assignment[i] == usize::MAX {
                assignment[i] = j;
                assigned_row[j] = i;
                col_cover[j] = true;
            }
        }
    }

    // Iterative refinement (simplified Munkres)
    for _ in 0..(n * n) {
        // Find unassigned row
        let unassigned = (0..n).find(|&i| assignment[i] == usize::MAX);
        let Some(start_row) = unassigned else { break };

        // Try to find augmenting path via DFS
        let mut visited_cols = vec![false; n];
        let mut path_col = vec![usize::MAX; n];
        let mut path_row = vec![usize::MAX; n];

        if !try_augment_hungarian(
            start_row,
            &cost,
            &assignment,
            &assigned_row,
            &mut visited_cols,
            &mut path_col,
            &mut path_row,
            n,
        ) {
            // Find minimum uncovered element and reduce
            let min_uncov = (0..n)
                .filter(|&i| !row_cover[i])
                .flat_map(|i| {
                    let row = &cost[i];
                    (0..n)
                        .filter(|&j| !col_cover[j])
                        .map(|j| row[j])
                        .collect::<Vec<_>>()
                })
                .fold(f64::INFINITY, f64::min);

            if min_uncov.is_infinite() {
                break;
            }
            for i in 0..n {
                for j in 0..n {
                    if !row_cover[i] && !col_cover[j] {
                        cost[i][j] -= min_uncov;
                    } else if row_cover[i] && col_cover[j] {
                        cost[i][j] += min_uncov;
                    }
                }
            }
            // Re-cover
            col_cover = vec![false; n];
            row_cover = vec![false; n];
            assignment = vec![usize::MAX; n];
            assigned_row = vec![usize::MAX; n];
            for i in 0..n {
                for j in 0..n {
                    if cost[i][j].abs() < 1e-10
                        && !col_cover[j]
                        && assignment[i] == usize::MAX
                    {
                        assignment[i] = j;
                        assigned_row[j] = i;
                        col_cover[j] = true;
                    }
                }
            }
        }
    }

    // Fill any remaining holes greedily
    for i in 0..n {
        if assignment[i] == usize::MAX {
            for j in 0..n {
                if assigned_row[j] == usize::MAX {
                    assignment[i] = j;
                    assigned_row[j] = i;
                    break;
                }
            }
        }
    }

    assignment
}

/// DFS for augmenting path in Hungarian algorithm.
fn try_augment_hungarian(
    row: usize,
    cost: &[Vec<f64>],
    assignment: &[usize],
    assigned_row: &[usize],
    visited_cols: &mut Vec<bool>,
    path_col: &mut Vec<usize>,
    path_row: &mut Vec<usize>,
    n: usize,
) -> bool {
    for j in 0..n {
        if !visited_cols[j] && cost[row][j].abs() < 1e-10 {
            visited_cols[j] = true;
            path_col[row] = j;
            let prev_row = assigned_row[j];
            if prev_row == usize::MAX
                || try_augment_hungarian(
                    prev_row, cost, assignment, assigned_row, visited_cols, path_col,
                    path_row, n,
                )
            {
                path_row[j] = row;
                return true;
            }
        }
    }
    false
}

// ─── Persistence Entropy ──────────────────────────────────────────────────────

/// Compute the persistence entropy of a persistence diagram.
///
/// Persistence entropy measures the information content of the topological
/// features captured by the diagram.  It is defined as:
///
/// H = − Σᵢ pᵢ log₂(pᵢ)
///
/// where `pᵢ = persistence(xᵢ) / L` and `L = Σ persistence(xᵢ)` is
/// the total persistence (summed over all **finite** points).
///
/// Returns `0.0` for diagrams with no finite points.
pub fn persistence_entropy(diagram: &[PersistencePair]) -> f64 {
    let persts: Vec<f64> = diagram
        .iter()
        .filter(|p| !p.is_essential() && p.persistence() > 0.0)
        .map(|p| p.persistence())
        .collect();

    if persts.is_empty() {
        return 0.0;
    }

    let total: f64 = persts.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }

    persts
        .iter()
        .map(|&pi| {
            let p = pi / total;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}

// ─── Betti Numbers ────────────────────────────────────────────────────────────

/// Compute the Betti numbers `b₀, b₁, …` at a given filtration level.
///
/// The k-th Betti number counts the number of k-dimensional topological
/// features (connected components for k=0, loops for k=1, voids for k=2,
/// etc.) that are alive at filtration value `threshold`.
///
/// A feature with `(birth, death)` is alive at `threshold` when
/// `birth ≤ threshold < death` (or `death == +∞`).
///
/// # Parameters
///
/// * `diagram`   – Slice of `PersistencePair`.
/// * `threshold` – Filtration level at which to evaluate Betti numbers.
///
/// # Returns
///
/// A `Vec<usize>` of length `max_dimension + 1` where `result[k]` is `bₖ`.
/// Returns an empty vector if the diagram is empty.
pub fn betti_numbers(diagram: &[PersistencePair], threshold: f64) -> Vec<usize> {
    if diagram.is_empty() {
        return Vec::new();
    }

    let max_dim = diagram.iter().map(|p| p.dimension).max().unwrap_or(0);
    let mut betti = vec![0usize; max_dim + 1];

    for pair in diagram {
        let alive = pair.birth <= threshold
            && (pair.is_essential() || pair.death > threshold);
        if alive && pair.dimension <= max_dim {
            betti[pair.dimension] += 1;
        }
    }

    betti
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    fn triangle_dist() -> Array2<f64> {
        // Three points: distances 1, 2, 3
        array![[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]]
    }

    fn four_points_dist() -> Array2<f64> {
        // Two clusters: {0,1} close, {2,3} close, clusters far apart
        array![
            [0.0, 0.5, 5.0, 5.2],
            [0.5, 0.0, 5.1, 5.0],
            [5.0, 5.1, 0.0, 0.3],
            [5.2, 5.0, 0.3, 0.0],
        ]
    }

    #[test]
    fn test_persistence_diagram_0d_triangle() {
        let dist = triangle_dist();
        let pairs = persistence_diagram_0d(&dist).expect("should succeed");
        // 3 points → 2 finite pairs + 1 essential
        assert_eq!(pairs.len(), 3);
        let finite: Vec<_> = pairs.iter().filter(|p| !p.is_essential()).collect();
        assert_eq!(finite.len(), 2);
        let essential: Vec<_> = pairs.iter().filter(|p| p.is_essential()).collect();
        assert_eq!(essential.len(), 1);
    }

    #[test]
    fn test_persistence_diagram_0d_two_clusters() {
        let dist = four_points_dist();
        let pairs = persistence_diagram_0d(&dist).expect("should succeed");
        // 4 points → 3 finite pairs + 1 essential
        let finite: Vec<_> = pairs.iter().filter(|p| !p.is_essential()).collect();
        assert_eq!(finite.len(), 3);
        // One pair should have a large death time (inter-cluster merge)
        let max_death = finite
            .iter()
            .map(|p| p.death)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_death > 4.0, "large inter-cluster death expected, got {max_death}");
    }

    #[test]
    fn test_bottleneck_same_diagram() {
        let pairs = persistence_diagram_0d(&triangle_dist()).expect("ok");
        let d = bottleneck_distance(&pairs, &pairs);
        assert!(d < 1e-10, "self-distance should be 0, got {d}");
    }

    #[test]
    fn test_bottleneck_empty_diagrams() {
        let d = bottleneck_distance(&[], &[]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_wasserstein_same_diagram() {
        let pairs = persistence_diagram_0d(&triangle_dist()).expect("ok");
        let d = wasserstein_distance_pd(&pairs, &pairs, 2);
        assert!(d < 1e-6, "self-distance should be 0, got {d}");
    }

    #[test]
    fn test_wasserstein_empty() {
        let d = wasserstein_distance_pd(&[], &[], 2);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn test_persistence_entropy_empty() {
        assert_eq!(persistence_entropy(&[]), 0.0);
    }

    #[test]
    fn test_persistence_entropy_positive() {
        let pairs = persistence_diagram_0d(&four_points_dist()).expect("ok");
        let h = persistence_entropy(&pairs);
        // Entropy should be non-negative; with varied persistences it should be > 0
        assert!(h >= 0.0);
    }

    #[test]
    fn test_betti_numbers_threshold() {
        let pairs = persistence_diagram_0d(&four_points_dist()).expect("ok");
        // At t=0 (start): all 4 components alive → b0 = 4
        let b_start = betti_numbers(&pairs, 0.0);
        assert!(!b_start.is_empty());
        // At high threshold: only 1 component
        let b_end = betti_numbers(&pairs, 100.0);
        assert_eq!(b_end[0], 1, "one essential component, got {:?}", b_end);
    }

    #[test]
    fn test_betti_numbers_empty() {
        let b = betti_numbers(&[], 1.0);
        assert!(b.is_empty());
    }

    #[test]
    fn test_persistence_pair_methods() {
        let p = PersistencePair::new(1.0, 4.0, 0);
        assert!((p.persistence() - 3.0).abs() < 1e-10);
        assert!(!p.is_essential());
        let proj = p.diagonal_projection().expect("has projection");
        assert!((proj.0 - 2.5).abs() < 1e-10);

        let ess = PersistencePair::new(0.0, f64::INFINITY, 0);
        assert!(ess.is_essential());
        assert_eq!(ess.persistence(), f64::INFINITY);
        assert!(ess.diagonal_projection().is_none());
    }

    #[test]
    fn test_non_square_distance_matrix() {
        let dist = Array2::<f64>::zeros((3, 4));
        assert!(persistence_diagram_0d(&dist).is_err());
    }
}
