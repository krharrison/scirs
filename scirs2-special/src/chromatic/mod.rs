//! Chromatic Polynomials of Simple Graphs
//!
//! The chromatic polynomial P(G, k) counts the number of proper k-colorings
//! of a simple graph G (colorings where adjacent vertices receive different colors).
//!
//! This module implements:
//! - Simple undirected graphs
//! - Chromatic polynomial via deletion-contraction
//! - Chromatic number computation
//! - Coloring verification
//!
//! The deletion-contraction algorithm uses memoization by graph canonical form,
//! yielding complexity O(2^m * poly(n)) for m edges, n vertices.

use crate::error::{SpecialError, SpecialResult};
use std::collections::HashMap;

/// A simple undirected graph (no self-loops, no multi-edges).
#[derive(Debug, Clone)]
pub struct SimpleGraph {
    /// Number of vertices (vertices are labeled 0, 1, ..., n_vertices-1)
    pub n_vertices: usize,
    /// Edge list (u, v) with u < v for canonical form
    pub edges: Vec<(usize, usize)>,
}

impl SimpleGraph {
    /// Create a new empty graph with the given number of vertices.
    pub fn new(n_vertices: usize) -> Self {
        SimpleGraph {
            n_vertices,
            edges: Vec::new(),
        }
    }

    /// Add an edge between vertices u and v.
    ///
    /// # Errors
    /// Returns `SpecialError::ValueError` if u or v are out of bounds, or if
    /// (u, v) is a self-loop or already exists.
    pub fn add_edge(&mut self, u: usize, v: usize) -> SpecialResult<()> {
        if u >= self.n_vertices || v >= self.n_vertices {
            return Err(SpecialError::ValueError(format!(
                "Vertex index out of bounds: u={u}, v={v}, n_vertices={}",
                self.n_vertices
            )));
        }
        if u == v {
            return Err(SpecialError::ValueError(format!(
                "Self-loops not allowed: vertex {u}"
            )));
        }
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if self.edges.contains(&(a, b)) {
            return Err(SpecialError::ValueError(format!(
                "Edge ({a}, {b}) already exists"
            )));
        }
        self.edges.push((a, b));
        Ok(())
    }

    /// Number of edges in the graph.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if there is an edge between u and v.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        self.edges.contains(&(a, b))
    }

    /// Return a canonical key for memoization: (n_vertices, sorted_edges).
    fn canonical_key(&self) -> GraphKey {
        let mut edges = self.edges.clone();
        edges.sort_unstable();
        GraphKey {
            n_vertices: self.n_vertices,
            edges,
        }
    }

    /// Construct the graph obtained by deleting edge `idx`.
    fn delete_edge(&self, idx: usize) -> SimpleGraph {
        let mut g = SimpleGraph {
            n_vertices: self.n_vertices,
            edges: self.edges.clone(),
        };
        g.edges.swap_remove(idx);
        g.edges.sort_unstable();
        g
    }

    /// Construct the graph obtained by contracting edge `idx`.
    ///
    /// Merging vertices u and v (edge[idx] = (u,v)): keep vertex u, remove v.
    /// Replace all edges incident to v with edges to u, removing self-loops and duplicates.
    fn contract_edge(&self, idx: usize) -> SimpleGraph {
        let (u, v) = self.edges[idx]; // u < v guaranteed by canonical form
        let n = self.n_vertices;

        // Remap vertices: v → u, vertices > v get shifted down by 1
        let remap = |x: usize| -> usize {
            if x == v {
                u
            } else if x > v {
                x - 1
            } else {
                x
            }
        };

        let new_n = n - 1;
        let mut new_edges: Vec<(usize, usize)> = Vec::new();

        for (i, &(a, b)) in self.edges.iter().enumerate() {
            if i == idx {
                continue; // skip the contracted edge
            }
            let ra = remap(a);
            let rb = remap(b);
            if ra == rb {
                continue; // self-loop after contraction — discard
            }
            let (ea, eb) = if ra < rb { (ra, rb) } else { (rb, ra) };
            if !new_edges.contains(&(ea, eb)) {
                new_edges.push((ea, eb));
            }
        }
        new_edges.sort_unstable();
        SimpleGraph {
            n_vertices: new_n,
            edges: new_edges,
        }
    }
}

/// Canonical key for memoization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GraphKey {
    n_vertices: usize,
    edges: Vec<(usize, usize)>,
}

// ────────────────────────────────────────────────────────────────────────────
// Chromatic polynomial computation
// ────────────────────────────────────────────────────────────────────────────

/// Compute the coefficients of the chromatic polynomial P(G, k) = Σ_i c_i * k^i.
///
/// Returns a vector of length n_vertices+1 where index i holds the coefficient
/// of k^i.
///
/// Uses deletion-contraction with memoization. Works well for graphs up to
/// ~12 vertices.
///
/// # Examples
/// ```
/// use scirs2_special::chromatic::{SimpleGraph, chromatic_polynomial_coefficients};
///
/// let mut g = SimpleGraph::new(3);
/// g.add_edge(0, 1).unwrap();
/// g.add_edge(1, 2).unwrap();
/// g.add_edge(0, 2).unwrap();
/// let coeffs = chromatic_polynomial_coefficients(&g);
/// // K_3: P(K_3, k) = k(k-1)(k-2) = k^3 - 3k^2 + 2k
/// assert_eq!(coeffs[3], 1);
/// assert_eq!(coeffs[2], -3);
/// assert_eq!(coeffs[1], 2);
/// assert_eq!(coeffs[0], 0);
/// ```
pub fn chromatic_polynomial_coefficients(g: &SimpleGraph) -> Vec<i64> {
    let n = g.n_vertices;
    let mut memo: HashMap<GraphKey, Vec<i64>> = HashMap::new();
    chrom_poly_rec(g, n, &mut memo)
}

/// Evaluate P(G, k) at a specific integer k using the coefficient vector.
pub fn chromatic_polynomial_evaluate(g: &SimpleGraph, k: usize) -> i64 {
    let coeffs = chromatic_polynomial_coefficients(g);
    let mut result = 0i64;
    let mut k_pow = 1i64;
    for &c in &coeffs {
        result += c * k_pow;
        k_pow *= k as i64;
    }
    result
}

/// Compute the chromatic number χ(G) = minimum k such that P(G, k) > 0.
///
/// Uses binary search / sequential search starting from 1.
/// Returns n_vertices as upper bound (always valid since the identity coloring uses n colors).
pub fn chromatic_number(g: &SimpleGraph) -> usize {
    let n = g.n_vertices;
    // Edge case: no vertices
    if n == 0 {
        return 0;
    }
    // No edges: 1-colorable
    if g.edges.is_empty() {
        return 1;
    }
    // Search from 2 upward
    for k in 1..=n {
        let p = chromatic_polynomial_evaluate(g, k);
        if p > 0 {
            return k;
        }
    }
    n
}

/// Verify whether a given coloring is proper.
///
/// A coloring `coloring[v]` for each vertex v is proper if for every edge (u,v),
/// `coloring[u] != coloring[v]`.
///
/// # Arguments
/// * `g` - The graph
/// * `coloring` - Color assignment (`coloring[v]` = color of vertex v)
pub fn is_proper_coloring(g: &SimpleGraph, coloring: &[usize]) -> bool {
    if coloring.len() != g.n_vertices {
        return false;
    }
    for &(u, v) in &g.edges {
        if coloring[u] == coloring[v] {
            return false;
        }
    }
    true
}

// ────────────────────────────────────────────────────────────────────────────
// Internal: deletion-contraction recursion
// ────────────────────────────────────────────────────────────────────────────

/// Recursively compute chromatic polynomial coefficients via deletion-contraction.
fn chrom_poly_rec(
    g: &SimpleGraph,
    max_degree: usize,
    memo: &mut HashMap<GraphKey, Vec<i64>>,
) -> Vec<i64> {
    let key = g.canonical_key();
    if let Some(cached) = memo.get(&key) {
        return cached.clone();
    }

    let n = g.n_vertices;
    let m = g.edges.len();

    // Base case 1: no edges — P = k^n
    if m == 0 {
        let mut coeffs = vec![0i64; max_degree + 1];
        if n <= max_degree {
            coeffs[n] = 1;
        }
        memo.insert(key, coeffs.clone());
        return coeffs;
    }

    // Base case 2: complete graph K_n — P = k*(k-1)*...*(k-n+1)
    let expected_edges = n * (n - 1) / 2;
    if m == expected_edges {
        let coeffs = falling_factorial_coeffs(n, max_degree);
        memo.insert(key, coeffs.clone());
        return coeffs;
    }

    // Base case 3: disconnected graph — P(G) = product of P(components)
    if let Some(coeffs) = try_disconnect(g, max_degree, memo) {
        memo.insert(key, coeffs.clone());
        return coeffs;
    }

    // Deletion-contraction on first edge
    let g_del = g.delete_edge(0);
    let g_con = g.contract_edge(0);

    let p_del = chrom_poly_rec(&g_del, max_degree, memo);
    let p_con = chrom_poly_rec(&g_con, max_degree, memo);

    // P(G) = P(G-e) - P(G/e)
    // P(G/e) has n_vertices-1 so its degree is one less; pad to max_degree+1
    let mut coeffs = vec![0i64; max_degree + 1];
    for i in 0..=max_degree {
        let del_i = p_del.get(i).copied().unwrap_or(0);
        let con_i = p_con.get(i).copied().unwrap_or(0);
        coeffs[i] = del_i - con_i;
    }

    memo.insert(key, coeffs.clone());
    coeffs
}

/// Compute coefficients of k*(k-1)*...*(k-n+1) (falling factorial / Pochhammer).
fn falling_factorial_coeffs(n: usize, max_deg: usize) -> Vec<i64> {
    // Build product (k)(k-1)(k-2)...(k-n+1)
    // Start with [1] and multiply by (k - i) for i = 0, ..., n-1
    let mut poly = vec![0i64; max_deg + 1];
    poly[0] = 1; // constant polynomial "1"

    for i in 0..n {
        // Multiply poly by (k - i): new[j] = poly[j-1] - i*poly[j]
        let old = poly.clone();
        poly[0] = -(i as i64) * old[0];
        for j in 1..=max_deg {
            let prev = old.get(j - 1).copied().unwrap_or(0);
            poly[j] = prev - (i as i64) * old[j];
        }
    }
    poly
}

/// Try to decompose G into connected components and multiply their polynomials.
///
/// Returns `None` if the graph is connected.
fn try_disconnect(
    g: &SimpleGraph,
    max_degree: usize,
    memo: &mut HashMap<GraphKey, Vec<i64>>,
) -> Option<Vec<i64>> {
    let components = connected_components(g);
    if components.len() <= 1 {
        return None;
    }
    // Product of polynomials of individual components
    let mut product = vec![0i64; max_degree + 1];
    product[0] = 1;

    for comp_vertices in components {
        let sub = induced_subgraph(g, &comp_vertices);
        let sub_poly = chrom_poly_rec(&sub, max_degree, memo);
        product = poly_multiply(&product, &sub_poly, max_degree);
    }
    Some(product)
}

/// Find connected components via BFS.
fn connected_components(g: &SimpleGraph) -> Vec<Vec<usize>> {
    let n = g.n_vertices;
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in 0..n {
        if !visited[start] {
            let mut comp = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(v) = queue.pop_front() {
                comp.push(v);
                for &(a, b) in &g.edges {
                    let neighbor = if a == v {
                        b
                    } else if b == v {
                        a
                    } else {
                        continue;
                    };
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            components.push(comp);
        }
    }
    components
}

/// Construct the induced subgraph on a subset of vertices.
fn induced_subgraph(g: &SimpleGraph, vertices: &[usize]) -> SimpleGraph {
    let n_sub = vertices.len();
    // Build mapping from original vertex index to new index
    let mut vmap = vec![usize::MAX; g.n_vertices];
    for (new_idx, &v) in vertices.iter().enumerate() {
        vmap[v] = new_idx;
    }
    let mut sub = SimpleGraph::new(n_sub);
    for &(a, b) in &g.edges {
        let na = vmap[a];
        let nb = vmap[b];
        if na != usize::MAX && nb != usize::MAX {
            let (ea, eb) = if na < nb { (na, nb) } else { (nb, na) };
            sub.edges.push((ea, eb));
        }
    }
    sub.edges.sort_unstable();
    sub
}

/// Multiply two polynomials (coefficient vectors).
fn poly_multiply(a: &[i64], b: &[i64], max_deg: usize) -> Vec<i64> {
    let mut result = vec![0i64; max_deg + 1];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if i + j <= max_deg {
                result[i + j] += ai * bj;
            }
        }
    }
    result
}

// ────────────────────────────────────────────────────────────────────────────
// Graph constructors for common graphs
// ────────────────────────────────────────────────────────────────────────────

/// Construct the complete graph K_n.
pub fn complete_graph(n: usize) -> SimpleGraph {
    let mut g = SimpleGraph::new(n);
    for u in 0..n {
        for v in (u + 1)..n {
            g.edges.push((u, v));
        }
    }
    g
}

/// Construct a path graph P_n on n vertices (n-1 edges).
pub fn path_graph(n: usize) -> SimpleGraph {
    let mut g = SimpleGraph::new(n);
    for i in 0..(n - 1) {
        g.edges.push((i, i + 1));
    }
    g
}

/// Construct a cycle graph C_n.
pub fn cycle_graph(n: usize) -> SimpleGraph {
    let mut g = path_graph(n);
    if n >= 3 {
        g.edges.push((0, n - 1));
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_k3() -> SimpleGraph {
        let mut g = SimpleGraph::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g.add_edge(0, 2).expect("ok");
        g
    }

    fn make_path3() -> SimpleGraph {
        let mut g = SimpleGraph::new(3);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(1, 2).expect("ok");
        g
    }

    #[test]
    fn test_k3_chromatic_polynomial() {
        // P(K_3, k) = k(k-1)(k-2) = k^3 - 3k^2 + 2k
        let g = make_k3();
        let coeffs = chromatic_polynomial_coefficients(&g);
        assert_eq!(coeffs[3], 1, "k^3 coefficient");
        assert_eq!(coeffs[2], -3, "k^2 coefficient");
        assert_eq!(coeffs[1], 2, "k^1 coefficient");
        assert_eq!(coeffs[0], 0, "constant term");
    }

    #[test]
    fn test_k3_evaluate() {
        let g = make_k3();
        // P(K_3, 3) = 6
        assert_eq!(chromatic_polynomial_evaluate(&g, 3), 6);
        // P(K_3, 2) = 0
        assert_eq!(chromatic_polynomial_evaluate(&g, 2), 0);
        // P(K_3, 4) = 4*3*2 = 24
        assert_eq!(chromatic_polynomial_evaluate(&g, 4), 24);
    }

    #[test]
    fn test_path3_chromatic_polynomial() {
        // P(P_3, k) = k(k-1)^2 = k^3 - 2k^2 + k
        let g = make_path3();
        let coeffs = chromatic_polynomial_coefficients(&g);
        assert_eq!(coeffs[3], 1, "k^3 coefficient");
        assert_eq!(coeffs[2], -2, "k^2 coefficient");
        assert_eq!(coeffs[1], 1, "k^1 coefficient");
        assert_eq!(coeffs[0], 0, "constant term");
    }

    #[test]
    fn test_path3_evaluate() {
        let g = make_path3();
        // P(P_3, 2) = 2*1 = 2
        assert_eq!(chromatic_polynomial_evaluate(&g, 2), 2);
        // P(P_3, 3) = 3*4 = 12
        assert_eq!(chromatic_polynomial_evaluate(&g, 3), 12);
    }

    #[test]
    fn test_chromatic_number_k4() {
        let g = complete_graph(4);
        assert_eq!(chromatic_number(&g), 4);
    }

    #[test]
    fn test_chromatic_number_path() {
        let g = path_graph(5);
        // Path graph is bipartite → 2-colorable
        assert_eq!(chromatic_number(&g), 2);
    }

    #[test]
    fn test_chromatic_number_cycle_even() {
        // Even cycle is bipartite → 2-colorable
        let g = cycle_graph(4);
        assert_eq!(chromatic_number(&g), 2);
    }

    #[test]
    fn test_chromatic_number_cycle_odd() {
        // Odd cycle → 3-colorable
        let g = cycle_graph(5);
        assert_eq!(chromatic_number(&g), 3);
    }

    #[test]
    fn test_is_proper_coloring_valid() {
        let g = make_k3();
        let coloring = [0, 1, 2];
        assert!(is_proper_coloring(&g, &coloring));
    }

    #[test]
    fn test_is_proper_coloring_invalid() {
        let g = make_k3();
        let coloring = [0, 0, 1]; // vertices 0 and 1 both have color 0 but are adjacent
        assert!(!is_proper_coloring(&g, &coloring));
    }

    #[test]
    fn test_add_edge_self_loop_error() {
        let mut g = SimpleGraph::new(3);
        let result = g.add_edge(1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edge_duplicate_error() {
        let mut g = SimpleGraph::new(3);
        g.add_edge(0, 1).expect("ok");
        let result = g.add_edge(0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_graph() {
        // Empty graph (no edges): P(G, k) = k^n
        let g = SimpleGraph::new(3);
        let coeffs = chromatic_polynomial_coefficients(&g);
        assert_eq!(coeffs[3], 1);
        assert_eq!(coeffs[0], 0);
        assert_eq!(chromatic_number(&g), 1);
    }

    #[test]
    fn test_disconnected_graph() {
        // Two disjoint edges: P(G, k) = k(k-1) * k(k-1) = k²(k-1)²
        let mut g = SimpleGraph::new(4);
        g.add_edge(0, 1).expect("ok");
        g.add_edge(2, 3).expect("ok");
        // P(G, k) = [k(k-1)]^2
        let p2 = chromatic_polynomial_evaluate(&g, 2);
        assert_eq!(p2, 4); // 2*1*2*1 = 4
        let p3 = chromatic_polynomial_evaluate(&g, 3);
        assert_eq!(p3, 36); // 3*2*3*2 = 36
    }

    #[test]
    fn test_falling_factorial_k3() {
        // k*(k-1)*(k-2) at k=5 should be 60
        let poly = falling_factorial_coeffs(3, 5);
        let val: i64 = poly
            .iter()
            .enumerate()
            .map(|(i, &c)| c * 5i64.pow(i as u32))
            .sum();
        assert_eq!(val, 60);
    }
}
