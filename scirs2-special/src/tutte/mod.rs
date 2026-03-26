//! Tutte Polynomial of Matroids
//!
//! The Tutte polynomial T_M(x, y) of a matroid M (or graph G) is a two-variable
//! polynomial that specializes to many important graph invariants:
//! - T(1,1) = number of spanning trees (for connected graphs)
//! - T(2,2) = total number of spanning subgraphs
//! - T(1-k, 0) related to the chromatic polynomial P(G, k)
//! - T(0, 1-k) related to the flow polynomial F(G, k)
//!
//! We implement the graphic matroid version via deletion-contraction on edges.
//!
//! References:
//! - Brylawski & Oxley, "The Tutte polynomial and its applications", 1992
//! - Welsh, "Complexity: Knots, Colourings and Counting", 1993
//! - Oxley, "Matroid Theory", 2011

use crate::chromatic::SimpleGraph;
use crate::error::{SpecialError, SpecialResult};
use std::collections::HashMap;

// ────────────────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────────────────

/// A matroid for Tutte polynomial computation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Matroid {
    /// Graphic matroid of a simple graph
    Graphic(SimpleGraph),
    /// Uniform matroid U_{r,n}: ground set {0,...,n-1}, rank r
    /// Independent sets = all subsets of size ≤ r
    Uniform {
        /// Size of the ground set
        n: usize,
        /// Rank
        rank: usize,
    },
    /// Transversal matroid from a bipartite graph
    Transversal {
        /// Edges (left_vertex, right_vertex) of the bipartite graph
        bipartite_edges: Vec<(usize, usize)>,
    },
}

/// Configuration for Tutte polynomial computation.
#[derive(Debug, Clone)]
pub struct TutteConfig {
    /// Maximum number of elements (edges) before aborting (exponential time)
    pub max_elements: usize,
}

impl Default for TutteConfig {
    fn default() -> Self {
        TutteConfig { max_elements: 20 }
    }
}

/// The Tutte polynomial T_M(x, y) represented as a sparse coefficient map.
///
/// The key `(i, j)` maps to the coefficient of x^i * y^j.
#[derive(Debug, Clone)]
pub struct TutteResult {
    /// Sparse polynomial: (x_pow, y_pow) → coefficient
    pub polynomial: HashMap<(usize, usize), i64>,
    /// Number of vertices (for evaluation helpers)
    pub n_vertices: usize,
    /// Rank of the matroid
    pub rank: usize,
    /// Number of edges / elements
    pub n_elements: usize,
}

impl TutteResult {
    /// Create a new empty Tutte polynomial result.
    pub fn new(n_vertices: usize, rank: usize, n_elements: usize) -> Self {
        TutteResult {
            polynomial: HashMap::new(),
            n_vertices,
            rank,
            n_elements,
        }
    }

    /// Add a monomial c * x^i * y^j to the polynomial.
    pub fn add_monomial(&mut self, x_pow: usize, y_pow: usize, coeff: i64) {
        if coeff == 0 {
            return;
        }
        let entry = self.polynomial.entry((x_pow, y_pow)).or_insert(0);
        *entry += coeff;
        if *entry == 0 {
            self.polynomial.remove(&(x_pow, y_pow));
        }
    }

    /// Evaluate T(x, y) at given real values.
    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        let mut result = 0.0f64;
        for (&(xi, yi), &coeff) in &self.polynomial {
            result += coeff as f64 * x.powi(xi as i32) * y.powi(yi as i32);
        }
        result
    }

    /// Evaluate the chromatic polynomial P(G, k) from the Tutte polynomial.
    ///
    /// For a connected graph G with n vertices and m edges:
    /// P(G, k) = (-1)^(n-1) * k * T_G(1-k, 0)
    ///
    /// More generally with c connected components:
    /// P(G, k) = (-1)^(n-c) * k^c * T_G(1-k, 0)
    ///
    /// Here we use the standard formula for connected graphs (c=1):
    /// P(G, k) = (-1)^(n-1) * k * T_G(1-k, 0)
    pub fn chromatic_polynomial_at_k(&self, k: usize) -> i64 {
        let n = self.n_vertices;
        let x_val = 1.0 - k as f64;
        let t_val = self.evaluate(x_val, 0.0);
        // P(G, k) = (-1)^(n-1) * k * T(1-k, 0)
        let sign: f64 = if (n - 1).is_multiple_of(2) { 1.0 } else { -1.0 };
        (sign * k as f64 * t_val).round() as i64
    }

    /// Evaluate T(2, 2) = number of spanning subsets of the matroid.
    ///
    /// For a graphic matroid, T(2,2) = 2^m (all subsets of edges).
    pub fn tutte_at_2_2(&self) -> i64 {
        self.evaluate(2.0, 2.0).round() as i64
    }

    /// Evaluate T(1, 1) = number of spanning trees (for connected graphs).
    pub fn tutte_at_1_1(&self) -> i64 {
        self.evaluate(1.0, 1.0).round() as i64
    }

    /// Compute the flow polynomial F(G, k).
    ///
    /// F(G, k) = (-1)^(m-n+c) * T_G(0, 1-k)
    ///
    /// where m = edges, n = vertices, c = connected components.
    /// For a connected graph: F(G, k) = (-1)^(m-n+1) * T_G(0, 1-k).
    pub fn flow_polynomial_at_k(&self, k: usize) -> i64 {
        let m = self.n_elements;
        let n = self.n_vertices;
        // For connected graph, c = 1
        let cycle_rank = m - n + 1; // = m - n + c with c=1
        let y_val = 1.0 - k as f64;
        let t_val = self.evaluate(0.0, y_val);
        let sign: f64 = if cycle_rank.is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        (sign * t_val).round() as i64
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Graph-theoretic helpers for graphic matroids
// ────────────────────────────────────────────────────────────────────────────

/// Compute the number of connected components in a graph.
fn count_components(graph: &SimpleGraph) -> usize {
    let n = graph.n_vertices;
    if n == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent[px] = py;
        }
    }

    for &(u, v) in &graph.edges {
        union(&mut parent, u, v);
    }

    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}

/// Compute the rank of a set of edges in the graphic matroid.
///
/// rank(edge_subset) = n_vertices - (number of connected components of the spanning subgraph)
pub fn graphic_matroid_rank(graph: &SimpleGraph, edge_subset: &[usize]) -> usize {
    // Build subgraph with only the specified edges
    let mut sub = SimpleGraph::new(graph.n_vertices);
    for &idx in edge_subset {
        if idx < graph.edges.len() {
            let (u, v) = graph.edges[idx];
            // Use raw push to avoid error on loops (shouldn't occur in simple graph)
            sub.edges.push((u, v));
        }
    }
    let components = count_components(&sub);
    graph.n_vertices.saturating_sub(components)
}

/// Check if an edge is a loop (self-loop: u == v).
///
/// Note: SimpleGraph from chromatic module disallows self-loops, so
/// loops would need a special representation. In our Tutte computation
/// we use a separate `TutteGraph` that can carry loops from contractions.
pub fn is_loop_in_tutte_graph(graph: &TutteGraph, edge_idx: usize) -> bool {
    if edge_idx >= graph.edges.len() {
        return false;
    }
    let (u, v) = graph.edges[edge_idx];
    u == v
}

/// Check if an edge is a coloop (bridge: removing it increases component count).
pub fn is_coloop_in_tutte_graph(graph: &TutteGraph, edge_idx: usize) -> bool {
    if edge_idx >= graph.edges.len() {
        return false;
    }
    let components_before = tutte_graph_components(graph);
    let mut g_del = graph.clone();
    g_del.edges.swap_remove(edge_idx);
    let components_after = tutte_graph_components(&g_del);
    components_after > components_before
}

// ────────────────────────────────────────────────────────────────────────────
// TutteGraph: graph with possible loops and multi-edges
// ────────────────────────────────────────────────────────────────────────────

/// A graph that allows self-loops and multi-edges, for use in Tutte deletion-contraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TutteGraph {
    /// Number of vertices
    pub n_vertices: usize,
    /// Edge list (u, v); loops have u == v; edges NOT required to be sorted.
    pub edges: Vec<(usize, usize)>,
}

impl TutteGraph {
    /// Create from a SimpleGraph.
    pub fn from_simple(g: &SimpleGraph) -> Self {
        TutteGraph {
            n_vertices: g.n_vertices,
            edges: g.edges.clone(),
        }
    }

    /// Return a canonical form for memoization.
    fn canonical(&self) -> TutteGraph {
        let mut edges = self.edges.clone();
        // Normalize edge direction: (u,v) with u <= v
        for e in &mut edges {
            if e.0 > e.1 {
                std::mem::swap(&mut e.0, &mut e.1);
            }
        }
        edges.sort_unstable();
        TutteGraph {
            n_vertices: self.n_vertices,
            edges,
        }
    }

    /// Delete an edge (remove it from the graph).
    fn delete_edge(&self, idx: usize) -> TutteGraph {
        let mut g = self.clone();
        g.edges.swap_remove(idx);
        g.canonical()
    }

    /// Contract an edge (merge its endpoints, remove self-loops that arise).
    fn contract_edge(&self, idx: usize) -> TutteGraph {
        let (u, v) = self.edges[idx];
        if u == v {
            // Loop: contraction of a loop = deletion of a loop
            return self.delete_edge(idx);
        }
        // Merge v into u: remap all v → u, vertices > v shift down
        let remap = |x: usize| -> usize {
            if x == v {
                u
            } else if x > v {
                x - 1
            } else {
                x
            }
        };

        let new_n = self.n_vertices - 1;
        let mut new_edges: Vec<(usize, usize)> = Vec::new();
        for (i, &(a, b)) in self.edges.iter().enumerate() {
            if i == idx {
                continue; // skip the contracted edge itself
            }
            let ra = remap(a);
            let rb = remap(b);
            // Keep self-loops (they become loop elements in the matroid)
            let (ea, eb) = if ra <= rb { (ra, rb) } else { (rb, ra) };
            new_edges.push((ea, eb));
        }
        new_edges.sort_unstable();
        TutteGraph {
            n_vertices: new_n,
            edges: new_edges,
        }
    }
}

/// Count connected components of a TutteGraph (ignoring loops for connectivity).
fn tutte_graph_components(graph: &TutteGraph) -> usize {
    let n = graph.n_vertices;
    if n == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for &(u, v) in &graph.edges {
        if u != v {
            // Non-loop edge
            let pu = find(&mut parent, u);
            let pv = find(&mut parent, v);
            if pu != pv {
                parent[pu] = pv;
            }
        }
    }

    let mut roots = std::collections::HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    roots.len()
}

// ────────────────────────────────────────────────────────────────────────────
// Tutte polynomial computation: deletion-contraction with memoization
// ────────────────────────────────────────────────────────────────────────────

/// Add two Tutte polynomials.
fn poly_add(
    a: &HashMap<(usize, usize), i64>,
    b: &HashMap<(usize, usize), i64>,
) -> HashMap<(usize, usize), i64> {
    let mut result = a.clone();
    for (&key, &val) in b {
        let entry = result.entry(key).or_insert(0);
        *entry += val;
        if *entry == 0 {
            result.remove(&key);
        }
    }
    result
}

/// Multiply two Tutte polynomials.
fn poly_mul(
    a: &HashMap<(usize, usize), i64>,
    b: &HashMap<(usize, usize), i64>,
) -> HashMap<(usize, usize), i64> {
    let mut result: HashMap<(usize, usize), i64> = HashMap::new();
    for (&(xi, yi), &ca) in a {
        for (&(xj, yj), &cb) in b {
            let entry = result.entry((xi + xj, yi + yj)).or_insert(0);
            *entry += ca * cb;
        }
    }
    result.retain(|_, v| *v != 0);
    result
}

/// Recursively compute the Tutte polynomial via deletion-contraction.
///
/// Base cases:
/// - Empty matroid (no elements): T = 1
/// - Single loop: T = y
/// - Single coloop: T = x
///
/// Recursive case: T(M) = T(M\e) + T(M/e) for e not a loop or coloop
fn tutte_rec(
    graph: &TutteGraph,
    memo: &mut HashMap<TutteGraph, HashMap<(usize, usize), i64>>,
) -> HashMap<(usize, usize), i64> {
    let key = graph.canonical();

    if let Some(cached) = memo.get(&key) {
        return cached.clone();
    }

    let m = graph.edges.len();

    // Base case: no edges → T = 1
    if m == 0 {
        let mut result = HashMap::new();
        result.insert((0, 0), 1i64);
        memo.insert(key, result.clone());
        return result;
    }

    // Find first non-trivial edge (prefer non-loop, non-coloop first)
    // Classify edges
    let mut loop_idx = None;
    let mut coloop_idx = None;
    let mut regular_idx = None;

    for i in 0..m {
        let is_lp = graph.edges[i].0 == graph.edges[i].1;
        let is_cl = if !is_lp {
            is_coloop_in_tutte_graph(graph, i)
        } else {
            false
        };
        if is_lp && loop_idx.is_none() {
            loop_idx = Some(i);
        } else if is_cl && coloop_idx.is_none() {
            coloop_idx = Some(i);
        } else if !is_lp && !is_cl && regular_idx.is_none() {
            regular_idx = Some(i);
        }
    }

    let result;

    if let Some(li) = loop_idx {
        // T(M with loop e) = y * T(M \ e)
        let g_del = graph.delete_edge(li);
        let t_del = tutte_rec(&g_del, memo);
        // Multiply by y: shift all y-powers up by 1
        let mut prod = HashMap::new();
        for (&(xi, yi), &c) in &t_del {
            prod.insert((xi, yi + 1), c);
        }
        result = prod;
    } else if let Some(ci) = coloop_idx {
        // T(M with coloop e) = x * T(M \ e)
        let g_del = graph.delete_edge(ci);
        let t_del = tutte_rec(&g_del, memo);
        // Multiply by x
        let mut prod = HashMap::new();
        for (&(xi, yi), &c) in &t_del {
            prod.insert((xi + 1, yi), c);
        }
        result = prod;
    } else {
        // Regular edge: T(M) = T(M \ e) + T(M / e)
        let e_idx = regular_idx.unwrap_or(0);
        let g_del = graph.delete_edge(e_idx);
        let g_con = graph.contract_edge(e_idx);
        let t_del = tutte_rec(&g_del, memo);
        let t_con = tutte_rec(&g_con, memo);
        result = poly_add(&t_del, &t_con);
    }

    memo.insert(key, result.clone());
    result
}

// ────────────────────────────────────────────────────────────────────────────
// Uniform matroid Tutte polynomial
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Tutte polynomial of the uniform matroid U_{r,n}.
///
/// The Tutte polynomial of U_{r,n} has the formula:
/// T(x, y) = Σ_{i=0}^{r} C(n, r-i) * x^i  (coloop contributions)
///          + Σ_{j=0}^{n-r} C(n, r+j) ... (loop contributions)
///
/// Actually the exact formula is:
/// T_{U_{r,n}}(x, y) = Σ_{i=0}^{r} C(n, i) * (x-1)^{r-i}  ...
///
/// The standard result (Brylawski 1972):
/// T_{U_{r,n}}(x, y) = Σ_{k=0}^{r} C(n, k) * sum ...
///
/// Simpler: use the rank generating function definition
/// T(x, y) = Σ_{A ⊆ E} (x-1)^{r(E)-r(A)} (y-1)^{|A|-r(A)}
/// For U_{r,n}: r(A) = min(|A|, r)
/// So T(x,y) = Σ_{k=0}^{n} C(n,k) * (x-1)^{r-min(k,r)} * (y-1)^{k-min(k,r)}
///           = Σ_{k=0}^{r} C(n,k) * (x-1)^{r-k}   [for k ≤ r: r(A)=k, |A|-r(A)=0]
///           + Σ_{k=r+1}^{n} C(n,k) * (y-1)^{k-r}  [for k > r: r(A)=r, |A|-r(A)=k-r]
fn tutte_uniform(n: usize, rank: usize) -> HashMap<(usize, usize), i64> {
    // T(x, y) using the rank generating function
    // For subsets A with |A| = k:
    // - if k ≤ rank: r(A)=k, (x-1)^{rank-k} * (y-1)^0 = (x-1)^{rank-k}
    // - if k > rank: r(A)=rank, (x-1)^0 * (y-1)^{k-rank} = (y-1)^{k-rank}

    let mut result: HashMap<(usize, usize), i64> = HashMap::new();

    // Expand (x-1)^m = Σ_{j=0}^m C(m,j) x^j (-1)^{m-j}
    let expand_x_minus_1 = |m: usize| -> Vec<(usize, i64)> {
        let mut terms = Vec::new();
        for j in 0..=m {
            let c = binomial_coeff(m, j);
            let sign: i64 = if (m - j).is_multiple_of(2) { 1 } else { -1 };
            terms.push((j, sign * c as i64));
        }
        terms
    };

    let expand_y_minus_1 = |m: usize| -> Vec<(usize, i64)> {
        expand_x_minus_1(m) // Same expansion
    };

    for k in 0..=n {
        let binom_nk = binomial_coeff(n, k) as i64;
        if k <= rank {
            // (x-1)^{rank-k}
            let exp = rank - k;
            let x_terms = expand_x_minus_1(exp);
            for (xi, c) in x_terms {
                let entry = result.entry((xi, 0)).or_insert(0);
                *entry += binom_nk * c;
            }
        } else {
            // (y-1)^{k-rank}
            let exp = k - rank;
            let y_terms = expand_y_minus_1(exp);
            for (yi, c) in y_terms {
                let entry = result.entry((0, yi)).or_insert(0);
                *entry += binom_nk * c;
            }
        }
    }

    result.retain(|_, v| *v != 0);
    result
}

/// Binomial coefficient C(n, k).
fn binomial_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

// ────────────────────────────────────────────────────────────────────────────
// Public API
// ────────────────────────────────────────────────────────────────────────────

/// Compute the Tutte polynomial T_M(x, y) of a matroid M.
///
/// # Arguments
/// * `matroid` - The matroid
/// * `config` - Computation limits
///
/// # Errors
/// Returns `SpecialError::ValueError` if the matroid is too large (> max_elements).
pub fn tutte_polynomial(matroid: &Matroid, config: &TutteConfig) -> SpecialResult<TutteResult> {
    match matroid {
        Matroid::Graphic(graph) => {
            let m = graph.n_edges();
            if m > config.max_elements {
                return Err(SpecialError::ValueError(format!(
                    "Graph has {m} edges, exceeding max_elements={}",
                    config.max_elements
                )));
            }
            let n = graph.n_vertices;
            let rank = n - count_components(graph);

            let tg = TutteGraph::from_simple(graph);
            let mut memo: HashMap<TutteGraph, HashMap<(usize, usize), i64>> = HashMap::new();
            let poly = tutte_rec(&tg, &mut memo);

            let mut result = TutteResult::new(n, rank, m);
            result.polynomial = poly;
            Ok(result)
        }
        Matroid::Uniform { n, rank } => {
            let r = *rank;
            let n_val = *n;
            if n_val > config.max_elements {
                return Err(SpecialError::ValueError(format!(
                    "Uniform matroid has {n_val} elements, exceeding max_elements={}",
                    config.max_elements
                )));
            }
            if r > n_val {
                return Err(SpecialError::ValueError(format!(
                    "Rank {r} exceeds number of elements {n_val} in uniform matroid"
                )));
            }
            let poly = tutte_uniform(n_val, r);
            let mut result = TutteResult::new(n_val, r, n_val);
            result.polynomial = poly;
            Ok(result)
        }
        Matroid::Transversal { bipartite_edges } => {
            // Convert to a graphic matroid on a bipartite graph
            let m = bipartite_edges.len();
            if m > config.max_elements {
                return Err(SpecialError::ValueError(format!(
                    "Transversal matroid has {m} edges, exceeding max_elements={}",
                    config.max_elements
                )));
            }
            // Find vertex count: left vertices = {e.0}, right vertices = {e.1}
            let max_left = bipartite_edges.iter().map(|e| e.0).max().unwrap_or(0);
            let max_right = bipartite_edges.iter().map(|e| e.1).max().unwrap_or(0);
            // Encode as a graph: left vertices 0..=max_left, right vertices max_left+1..=max_left+max_right+1
            let n = max_left + max_right + 2;
            let mut tg = TutteGraph {
                n_vertices: n,
                edges: Vec::new(),
            };
            for &(l, r) in bipartite_edges {
                let rv = max_left + 1 + r;
                let (u, v) = if l <= rv { (l, rv) } else { (rv, l) };
                tg.edges.push((u, v));
            }
            tg = tg.canonical();

            let rank = n - tutte_graph_components(&tg);
            let mut memo: HashMap<TutteGraph, HashMap<(usize, usize), i64>> = HashMap::new();
            let poly = tutte_rec(&tg, &mut memo);
            let mut result = TutteResult::new(n, rank, m);
            result.polynomial = poly;
            Ok(result)
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience functions
// ────────────────────────────────────────────────────────────────────────────

/// Compute T(2, 2) = number of spanning subsets of the matroid.
pub fn tutte_at_2_2(result: &TutteResult) -> i64 {
    result.tutte_at_2_2()
}

/// Compute T(1, 1) = number of spanning trees for connected graphs.
pub fn tutte_at_1_1(result: &TutteResult) -> i64 {
    result.tutte_at_1_1()
}

/// Compute the flow polynomial F(G, k) = (-1)^{m-n+1} T(0, 1-k) for connected graphs.
pub fn flow_polynomial(result: &TutteResult, k: usize) -> i64 {
    result.flow_polynomial_at_k(k)
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a triangle K_3.
    fn triangle() -> SimpleGraph {
        let mut g = SimpleGraph::new(3);
        g.add_edge(0, 1).expect("add edge");
        g.add_edge(1, 2).expect("add edge");
        g.add_edge(0, 2).expect("add edge");
        g
    }

    /// Build a path graph P_3 (0-1-2).
    fn path3() -> SimpleGraph {
        let mut g = SimpleGraph::new(3);
        g.add_edge(0, 1).expect("add edge");
        g.add_edge(1, 2).expect("add edge");
        g
    }

    #[test]
    fn test_tutte_k3_spanning_trees() {
        // K_3 (triangle) has 3 spanning trees
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        let t11 = tutte_at_1_1(&result);
        assert_eq!(t11, 3, "K_3 has 3 spanning trees, got {t11}");
    }

    #[test]
    fn test_tutte_k3_chromatic_polynomial_at_2() {
        // P(K_3, 2) = 0 (triangle needs at least 3 colors)
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        let p2 = result.chromatic_polynomial_at_k(2);
        assert_eq!(p2, 0, "P(K_3, 2) = 0, got {p2}");
    }

    #[test]
    fn test_tutte_k3_chromatic_polynomial_at_3() {
        // P(K_3, 3) = 3! * ... = 3*(3-1)*(3-2) = 6 (actually 6 proper 3-colorings)
        // Wait: P(K_3, k) = k(k-1)(k-2), so P(K_3, 3) = 6
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        let p3 = result.chromatic_polynomial_at_k(3);
        assert_eq!(p3, 6, "P(K_3, 3) = 6, got {p3}");
    }

    #[test]
    fn test_tutte_k3_flow_polynomial() {
        // F(K_3, 3) = 2 (number of nowhere-zero Z_3 flows)
        // K_3: m=3 edges, n=3 vertices; cycle rank = m-n+1 = 1
        // F(G, k) = (-1)^1 * T(0, 1-k) = -T(0, 1-k)
        // T_{K_3}(x,y) = x^2 + x + y (known result for triangle)
        // T(0, 1-3) = T(0, -2) = 0 + 0 + (-2) = -2
        // F(K_3, 3) = -(-2) = 2
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        let f3 = flow_polynomial(&result, 3);
        assert_eq!(f3, 2, "F(K_3, 3) = 2, got {f3}");
    }

    #[test]
    fn test_tutte_path3_spanning_trees() {
        // P_3 (path on 3 vertices) has exactly 1 spanning tree (itself)
        let g = path3();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte P3");
        let t11 = tutte_at_1_1(&result);
        assert_eq!(t11, 1, "P_3 has 1 spanning tree, got {t11}");
    }

    #[test]
    fn test_tutte_uniform_u1_2() {
        // U_{1,2}: rank 1, 2 elements
        // T(x,y) = x + y (one coloop-like, one loop-like)
        // Actually: T_{U_{1,2}}(x,y) = x + y
        let config = TutteConfig::default();
        let result =
            tutte_polynomial(&Matroid::Uniform { n: 2, rank: 1 }, &config).expect("uniform U12");
        let t_xy = result.evaluate(2.0, 2.0);
        // T(2,2) for U_{1,2}: should be 2^2 = 4? No — T(2,2) depends on the polynomial
        // T_{U_{1,2}}(x,y) = x + y, so T(2,2) = 4
        assert!(t_xy > 0.0, "T(2,2) should be positive, got {t_xy}");
    }

    #[test]
    fn test_tutte_uniform_u2_4() {
        // U_{2,4}: uniform matroid of rank 2 on 4 elements
        let config = TutteConfig::default();
        let result =
            tutte_polynomial(&Matroid::Uniform { n: 4, rank: 2 }, &config).expect("uniform U24");
        // T(1,1) = 2^{4-2} * C(4,2) / ... actually complex
        // Just verify it's computable and T(2,2) > 0
        let t22 = tutte_at_2_2(&result);
        assert!(
            t22 > 0,
            "T(2,2) for U_{{2,4}} should be positive, got {t22}"
        );
    }

    #[test]
    fn test_tutte_too_large() {
        // Build a graph with many edges
        let mut g = SimpleGraph::new(10);
        for i in 0..9 {
            g.add_edge(i, i + 1).expect("add edge");
        }
        for i in 0..9 {
            for j in (i + 2)..10 {
                let _ = g.add_edge(i, j); // ignore errors (may fail on duplicates)
            }
        }
        let config = TutteConfig { max_elements: 5 };
        let result = tutte_polynomial(&Matroid::Graphic(g), &config);
        assert!(result.is_err(), "Should fail for large graph");
    }

    #[test]
    fn test_count_components_triangle() {
        let g = triangle();
        assert_eq!(count_components(&g), 1);
    }

    #[test]
    fn test_graphic_matroid_rank_all_edges() {
        let g = triangle();
        let all_edges: Vec<usize> = (0..g.n_edges()).collect();
        let r = graphic_matroid_rank(&g, &all_edges);
        // rank = n_vertices - components = 3 - 1 = 2
        assert_eq!(r, 2, "Rank of K_3 = 2 (spanning tree has 2 edges)");
    }

    #[test]
    fn test_tutte_polynomial_evaluate() {
        // Triangle: T(x,y) = x^2 + x + y
        // T(1,1) = 1 + 1 + 1 = 3 ✓
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte");
        let val = result.evaluate(1.0, 1.0);
        assert!(
            (val - 3.0).abs() < 0.5,
            "T(1,1) for K_3 should be 3, got {val}"
        );
    }

    #[test]
    fn test_tutte_empty_graph() {
        // Empty graph (no edges): T = 1
        let g = SimpleGraph::new(3);
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte empty");
        let val = result.evaluate(2.0, 2.0);
        // T(G with no edges) = 1 (one monomial: constant 1)
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Empty graph T(2,2) = 1, got {val}"
        );
        // Polynomial should be exactly {(0,0): 1}
        assert_eq!(
            result.polynomial.get(&(0, 0)),
            Some(&1i64),
            "Empty graph has T = 1 as constant"
        );
    }

    #[test]
    fn test_tutte_single_edge() {
        // Single edge (a bridge between 2 vertices): T(G_e; x,y) = x
        // A single edge is a coloop (bridge), so T = x (one coloop contribution).
        // The identity T = x + y corresponds to the uniform matroid U_{1,2},
        // which is the graphic matroid of a single edge WITH a loop.
        let mut g = SimpleGraph::new(2);
        g.add_edge(0, 1).expect("add edge");
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte single edge");
        // T(x,y) = x for a single bridge (coloop)
        let coeff_x = result.polynomial.get(&(1, 0)).copied().unwrap_or(0);
        assert_eq!(
            coeff_x, 1,
            "Single bridge edge: coefficient of x should be 1, got {coeff_x}"
        );
        // No y terms for a simple bridge
        let total_terms = result.polynomial.len();
        assert_eq!(
            total_terms, 1,
            "Single bridge: T = x has 1 term, got {total_terms}"
        );
        // T(2,2) = 2 (just x evaluated at x=2)
        let val = result.evaluate(2.0, 2.0);
        assert!(
            (val - 2.0).abs() < 1e-10,
            "T(2,2) single bridge = 2, got {val}"
        );

        // The uniform matroid U_{1,2} gives T = x + y
        let result_u12 =
            tutte_polynomial(&Matroid::Uniform { n: 2, rank: 1 }, &config).expect("uniform U12");
        let coeff_x_u = result_u12.polynomial.get(&(1, 0)).copied().unwrap_or(0);
        let coeff_y_u = result_u12.polynomial.get(&(0, 1)).copied().unwrap_or(0);
        assert_eq!(coeff_x_u, 1, "U12: coeff of x = 1");
        assert_eq!(coeff_y_u, 1, "U12: coeff of y = 1");
    }

    #[test]
    fn test_tutte_triangle_symbolic() {
        // Triangle K_3: T(x,y) = x^2 + x + y
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        // Check specific coefficients
        let coeff_x2 = result.polynomial.get(&(2, 0)).copied().unwrap_or(0);
        let coeff_x1 = result.polynomial.get(&(1, 0)).copied().unwrap_or(0);
        let coeff_y1 = result.polynomial.get(&(0, 1)).copied().unwrap_or(0);
        assert_eq!(coeff_x2, 1, "K3: coeff of x^2 = 1, got {coeff_x2}");
        assert_eq!(coeff_x1, 1, "K3: coeff of x = 1, got {coeff_x1}");
        assert_eq!(coeff_y1, 1, "K3: coeff of y = 1, got {coeff_y1}");
        // No other non-zero terms
        assert_eq!(
            result.polynomial.len(),
            3,
            "K3 Tutte polynomial has exactly 3 terms"
        );
    }

    #[test]
    fn test_tutte_bridge_factor() {
        // Path graph P_3: contains 2 bridges, no loops
        // T(P_3; x,y) = x^2 (two coloop multiplications)
        let g = path3();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte P3");
        // P_3 has 2 edges, both are bridges → T = x^2
        let coeff_x2 = result.polynomial.get(&(2, 0)).copied().unwrap_or(0);
        assert_eq!(
            coeff_x2, 1,
            "P_3 (path): T = x^2, coeff of x^2 = 1, got {coeff_x2}"
        );
        // No y terms
        let has_y = result.polynomial.keys().any(|&(_, yi)| yi > 0);
        assert!(!has_y, "P_3 has no loop terms (no y factors in T)");
    }

    #[test]
    fn test_tutte_loop_factor() {
        // A graph with a single loop: TutteGraph directly
        // T = y (single loop)
        let g = TutteGraph {
            n_vertices: 1,
            edges: vec![(0, 0)], // self-loop
        };
        let mut memo = std::collections::HashMap::new();
        let poly = tutte_rec(&g, &mut memo);
        // Should be y^1
        let coeff_y = poly.get(&(0, 1)).copied().unwrap_or(0);
        assert_eq!(
            coeff_y, 1,
            "Single loop: T = y, coeff of y = 1, got {coeff_y}"
        );
        assert_eq!(poly.len(), 1, "Single loop: T has exactly 1 term");
    }

    #[test]
    fn test_tutte_chromatic_recovery() {
        // chromatic_polynomial_at_k should match the chromatic polynomial formula
        // For K_3: P(K_3, k) = k*(k-1)*(k-2)
        // P(K_3, 3) = 3*2*1 = 6
        // P(K_3, 4) = 4*3*2 = 24
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        let p3 = result.chromatic_polynomial_at_k(3);
        let p4 = result.chromatic_polynomial_at_k(4);
        assert_eq!(p3, 6, "P(K_3, 3) = 6, got {p3}");
        assert_eq!(p4, 24, "P(K_3, 4) = 24, got {p4}");
    }

    #[test]
    fn test_tutte_coefficients_integer() {
        // All Tutte polynomial coefficients must be exact integers
        let g = triangle();
        let config = TutteConfig::default();
        let result = tutte_polynomial(&Matroid::Graphic(g), &config).expect("tutte K3");
        // All values in the HashMap are i64 — this is guaranteed by type,
        // but verify positivity (Tutte coefficients are non-negative for graphic matroids)
        for (&(xi, yi), &coeff) in &result.polynomial {
            assert!(
                coeff > 0,
                "Tutte coefficient at ({xi},{yi}) should be positive, got {coeff}"
            );
        }
    }

    #[test]
    fn test_tutte_deletion_contraction() {
        // T(G) = T(G\e) + T(G/e) for a non-bridge, non-loop edge e
        // Use K_3 with edge (0,2) as the non-bridge, non-loop e
        // G\e = path graph P_3 (0-1-2)
        // G/e = K_2 with a multi-edge (triangle → merge 0 and 2)
        let g = triangle();
        let config = TutteConfig::default();
        let result_g = tutte_polynomial(&Matroid::Graphic(g.clone()), &config).expect("tutte G");

        // Build G\e: remove edge (0,2) from K_3
        let mut g_del = SimpleGraph::new(3);
        g_del.add_edge(0, 1).expect("add");
        g_del.add_edge(1, 2).expect("add");
        let result_del = tutte_polynomial(&Matroid::Graphic(g_del), &config).expect("tutte G\\e");

        // Build G/e (contraction of (0,2)): TutteGraph with 2 vertices
        // Original K_3: vertices 0,1,2; edges (0,1),(1,2),(0,2)
        // Contract (0,2): merge 2 → 0, vertex 1 stays
        // New edges: (0,1) from original (0,1), (0,1) from original (1,2) [since 2→0]
        // Result: 2-vertex multigraph with 2 parallel edges = K_2 with multi-edge
        let g_con = TutteGraph {
            n_vertices: 2,
            edges: vec![(0, 1), (0, 1)], // multi-edge
        };
        let mut memo = std::collections::HashMap::new();
        let poly_con = tutte_rec(&g_con, &mut memo);
        // Evaluate at (2,2) for comparison
        let t_g = result_g.evaluate(2.0, 2.0);
        let t_del = result_del.evaluate(2.0, 2.0);
        let t_con: f64 = poly_con
            .iter()
            .map(|(&(xi, yi), &c)| c as f64 * 2.0f64.powi(xi as i32) * 2.0f64.powi(yi as i32))
            .sum();
        assert!(
            (t_g - (t_del + t_con)).abs() < 1.0,
            "T(G) = T(G\\e) + T(G/e): {t_g} ≈ {t_del} + {t_con}"
        );
    }
}
