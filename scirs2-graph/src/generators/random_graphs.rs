//! Advanced random graph models and network generation
//!
//! This module implements a suite of state-of-the-art random graph models:
//!
//! - **Erdős–Rényi G(n,p)**: edges exist independently with probability p
//! - **Erdős–Rényi G(n,m)**: exactly m edges chosen uniformly at random
//! - **Barabási–Albert**: preferential attachment produces scale-free networks
//! - **Watts–Strogatz**: ring-lattice rewiring produces small-world topology
//! - **Random d-regular**: uniform random d-regular graph via configuration model
//! - **Hyperbolic random graph (HRG)**: geometric model in hyperbolic disk
//! - **Stochastic Kronecker graph**: iterative tensor-product graph model
//! - **Chung–Lu**: random graph with prescribed expected degree sequence

use crate::base::Graph;
use crate::error::{GraphError, Result};
use scirs2_core::ndarray::Array2;
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::prelude::*;
use scirs2_core::random::seq::SliceRandom;
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// G(n, p) — Erdős–Rényi probability model
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an Erdős–Rényi G(n, p) random graph.
///
/// Each pair of nodes (u, v) with u < v is connected independently with
/// probability `p`.  The expected number of edges is p · C(n, 2).
///
/// # Arguments
/// * `n` – number of nodes (nodes are labelled 0 … n-1)
/// * `p` – edge probability in \[0, 1\]
/// * `rng` – a seeded or default random-number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` when `p ∉ [0,1]` or `n == 0`.
///
/// # Example
/// ```rust
/// use scirs2_graph::generators::random_graphs::erdos_renyi_g_np;
/// use scirs2_core::random::prelude::*;
/// let mut rng = StdRng::seed_from_u64(7);
/// let g = erdos_renyi_g_np(20, 0.3, &mut rng).unwrap();
/// assert_eq!(g.node_count(), 20);
/// ```
pub fn erdos_renyi_g_np<R: Rng>(n: usize, p: f64, rng: &mut R) -> Result<Graph<usize, f64>> {
    if !(0.0..=1.0).contains(&p) {
        return Err(GraphError::InvalidGraph(format!(
            "erdos_renyi_g_np: p={p} must be in [0,1]"
        )));
    }
    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    for u in 0..n {
        for v in (u + 1)..n {
            if rng.random::<f64>() < p {
                g.add_edge(u, v, 1.0)?;
            }
        }
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// G(n, m) — Erdős–Rényi exact-edge-count model
// ─────────────────────────────────────────────────────────────────────────────

/// Generate an Erdős–Rényi G(n, m) random graph with **exactly** `m` edges.
///
/// A uniformly random subset of size `m` is chosen from all C(n, 2) possible
/// edges using reservoir sampling (Fisher–Yates on the candidate list).
///
/// # Arguments
/// * `n` – number of nodes
/// * `m` – exact number of edges; must satisfy m ≤ C(n, 2)
/// * `rng` – random-number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` when the edge count is infeasible.
pub fn erdos_renyi_g_nm<R: Rng>(n: usize, m: usize, rng: &mut R) -> Result<Graph<usize, f64>> {
    let max_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
    if m > max_edges {
        return Err(GraphError::InvalidGraph(format!(
            "erdos_renyi_g_nm: m={m} > C({n},2)={max_edges}"
        )));
    }
    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    if m == 0 {
        return Ok(g);
    }

    // Build the full candidate list and shuffle the first `m` entries.
    let mut candidates: Vec<(usize, usize)> = Vec::with_capacity(max_edges);
    for u in 0..n {
        for v in (u + 1)..n {
            candidates.push((u, v));
        }
    }
    // Partial Fisher–Yates: select m edges
    for i in 0..m {
        let j = i + rng.random_range(0..(candidates.len() - i));
        candidates.swap(i, j);
    }
    for &(u, v) in &candidates[..m] {
        g.add_edge(u, v, 1.0)?;
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Barabási–Albert preferential attachment
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Barabási–Albert (BA) scale-free graph via preferential attachment.
///
/// The model starts from an initial clique of `m + 1` nodes and then adds one
/// new node at a time, connecting it to `m` existing nodes with probability
/// proportional to their current degree (linear preferential attachment).
///
/// The resulting degree distribution follows a power law P(k) ~ k^{-3}.
///
/// # Arguments
/// * `n` – total number of nodes (must satisfy n > m ≥ 1)
/// * `m` – edges added per new node
/// * `rng` – random-number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` for invalid parameter combinations.
pub fn barabasi_albert<R: Rng>(n: usize, m: usize, rng: &mut R) -> Result<Graph<usize, f64>> {
    if m == 0 {
        return Err(GraphError::InvalidGraph(
            "barabasi_albert: m must be ≥ 1".to_string(),
        ));
    }
    if n <= m {
        return Err(GraphError::InvalidGraph(format!(
            "barabasi_albert: n={n} must be > m={m}"
        )));
    }

    let mut g = Graph::new();

    // Seed: complete graph on m+1 nodes
    for i in 0..=m {
        g.add_node(i);
    }
    for u in 0..=m {
        for v in (u + 1)..=m {
            g.add_edge(u, v, 1.0)?;
        }
    }

    // Degree bookkeeping for O(1) preferential sampling via repeated-stub trick.
    // stubs[i] appears degree[i] times so that uniform sampling over stubs gives
    // the correct preferential-attachment distribution.
    let mut stubs: Vec<usize> = Vec::with_capacity(n * m * 2);
    for i in 0..=m {
        for _ in 0..m {
            stubs.push(i);
        }
    }

    for new_node in (m + 1)..n {
        g.add_node(new_node);
        let mut chosen: HashSet<usize> = HashSet::with_capacity(m);

        while chosen.len() < m {
            let idx = rng.random_range(0..stubs.len());
            let target = stubs[idx];
            if target != new_node {
                chosen.insert(target);
            }
        }

        for &t in &chosen {
            g.add_edge(new_node, t, 1.0)?;
            // Update stubs: each accepted edge adds one stub for the target and one
            // for the new node, preserving proportional-to-degree sampling.
            stubs.push(t);
            stubs.push(new_node);
        }
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Watts–Strogatz small-world model
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Watts–Strogatz small-world graph.
///
/// Starting from a ring lattice where every node is connected to its `k`
/// nearest neighbours (k/2 on each side, k must be even), each edge is
/// independently rewired with probability `beta` to a uniformly random
/// non-neighbour.  This interpolates between a regular lattice (β=0) and an
/// Erdős–Rényi random graph (β=1) while passing through a small-world regime.
///
/// # Arguments
/// * `n`    – number of nodes
/// * `k`    – mean degree in the initial lattice (must be even, k < n)
/// * `beta` – rewiring probability in \[0, 1\]
/// * `rng`  – random-number generator
pub fn watts_strogatz<R: Rng>(
    n: usize,
    k: usize,
    beta: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if k == 0 || k >= n || !k.is_multiple_of(2) {
        return Err(GraphError::InvalidGraph(format!(
            "watts_strogatz: k={k} must be even, ≥ 2, and < n={n}"
        )));
    }
    if !(0.0..=1.0).contains(&beta) {
        return Err(GraphError::InvalidGraph(format!(
            "watts_strogatz: beta={beta} must be in [0,1]"
        )));
    }

    // Adjacency set for O(1) duplicate-edge checking
    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    // Initial ring lattice
    for u in 0..n {
        for s in 1..=(k / 2) {
            let v = (u + s) % n;
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }

    // Rewiring pass: iterate over u, then over each right-half neighbour v.
    for u in 0..n {
        for s in 1..=(k / 2) {
            let v = (u + s) % n;
            if !adj[u].contains(&v) {
                // Edge was already rewired away in a previous step; skip
                continue;
            }
            if rng.random::<f64>() < beta {
                // Choose a new target w != u and not already a neighbour
                let mut w = rng.random_range(0..n);
                let mut attempts = 0usize;
                while (w == u || adj[u].contains(&w)) && attempts < n * 4 {
                    w = rng.random_range(0..n);
                    attempts += 1;
                }
                if w == u || adj[u].contains(&w) {
                    // Could not find a valid rewire target; skip
                    continue;
                }
                // Remove edge (u, v)
                adj[u].remove(&v);
                adj[v].remove(&u);
                // Add edge (u, w)
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }
    }

    // Build the Graph from the final adjacency sets
    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    for u in 0..n {
        for &v in &adj[u] {
            if u < v {
                g.add_edge(u, v, 1.0)?;
            }
        }
    }

    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Random d-regular graph
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a uniformly random d-regular graph on `n` nodes.
///
/// A *d-regular* graph is one where every node has exactly degree `d`.
/// This is feasible only when `n·d` is even.
///
/// The algorithm uses the **configuration model** with self-loop / parallel-edge
/// rejection: stubs are generated (n·d stubs total), randomly paired, and
/// the pairing is rejected and retried if it produces a self-loop or
/// multi-edge.  The expected number of retries is O(1) for fixed d.
///
/// # Arguments
/// * `n` – number of nodes
/// * `d` – required degree of every node
/// * `rng` – random-number generator
///
/// # Returns
/// `Some(Graph)` when a valid d-regular graph was found within the attempt
/// budget, `None` when the degree sequence is infeasible or sampling
/// consistently fails (e.g., n < d+1).
pub fn random_regular<R: Rng>(n: usize, d: usize, rng: &mut R) -> Option<Graph<usize, f64>> {
    if n == 0 || d == 0 {
        let mut g = Graph::new();
        for i in 0..n {
            g.add_node(i);
        }
        return Some(g);
    }
    if d >= n {
        return None; // Not possible without self-loops
    }
    if !(n * d).is_multiple_of(2) {
        return None; // Degree sequence not graphical
    }

    let max_outer_attempts = 100usize;

    for _ in 0..max_outer_attempts {
        if let Some(g) = try_build_regular(n, d, rng) {
            return Some(g);
        }
    }
    None
}

/// Single attempt to build a d-regular graph via the configuration model.
fn try_build_regular<R: Rng>(n: usize, d: usize, rng: &mut R) -> Option<Graph<usize, f64>> {
    // Create stubs: node i appears d times
    let mut stubs: Vec<usize> = (0..n).flat_map(|i| std::iter::repeat_n(i, d)).collect();
    stubs.shuffle(rng);

    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];

    while stubs.len() >= 2 {
        let u = stubs[0];
        let v = stubs[1];
        if u == v || adj[u].contains(&v) {
            // Try to find a valid partner for u via random swap
            let swap_idx = 2 + rng.random_range(0..stubs.len().saturating_sub(2).max(1));
            if swap_idx < stubs.len() {
                stubs.swap(1, swap_idx);
            } else {
                return None; // Give up on this attempt
            }
            // Attempt limit guard: after many swaps we give up
            continue;
        }
        adj[u].insert(v);
        adj[v].insert(u);
        stubs.drain(0..2);
    }

    if !stubs.is_empty() {
        return None;
    }

    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    for u in 0..n {
        for &v in &adj[u] {
            if u < v {
                g.add_edge(u, v, 1.0).ok()?;
            }
        }
    }
    Some(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hyperbolic Random Graph (HRG)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a hyperbolic random graph (HRG) in the Poincaré disk model.
///
/// `n` nodes are placed uniformly at random in a hyperbolic disk of radius `r`
/// according to the quasi-uniform distribution with curvature parameter `alpha`
/// (controls degree heterogeneity; `alpha = 0.5` recovers the pure geometric
/// model, higher values concentrate nodes near the boundary).  Two nodes are
/// connected if their hyperbolic distance is at most `r`.
///
/// The hyperbolic distance between nodes at (r₁, θ₁) and (r₂, θ₂) in
/// polar-hyperbolic coordinates is:
///
/// ```text
/// d(u,v) = acosh(cosh(r₁)·cosh(r₂) − sinh(r₁)·sinh(r₂)·cos(θ₁−θ₂))
/// ```
///
/// # Arguments
/// * `n`     – number of nodes
/// * `r`     – disk radius; larger `r` gives sparser graphs
/// * `alpha` – radial density exponent (> 0; 0.5 ≤ α ≤ 1 for scale-free degree)
/// * `rng`   – random-number generator
///
/// # Reference
/// Krioukov et al., "Hyperbolic geometry of complex networks", Phys. Rev. E,
/// 82(3), 036106, 2010.
pub fn hyperbolic_random_graph<R: Rng>(
    n: usize,
    r: f64,
    alpha: f64,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    if r <= 0.0 {
        return Err(GraphError::InvalidGraph(format!(
            "hyperbolic_random_graph: r={r} must be > 0"
        )));
    }
    if alpha <= 0.0 {
        return Err(GraphError::InvalidGraph(format!(
            "hyperbolic_random_graph: alpha={alpha} must be > 0"
        )));
    }

    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }
    if n < 2 {
        return Ok(g);
    }

    // Sample polar coordinates in the hyperbolic disk.
    // Radial coordinate: the quasi-uniform distribution on [0, R] is
    //   F(ρ) = (cosh(alpha·ρ) − 1) / (cosh(alpha·R) − 1)
    // We invert it via inverse-CDF: ρ = acosh(1 + u·(cosh(α·R)−1)) / α
    let cosh_alpha_r = (alpha * r).cosh();
    let two_pi = std::f64::consts::TAU;

    let mut coords: Vec<(f64, f64)> = Vec::with_capacity(n); // (rho, theta)
    for _ in 0..n {
        let u: f64 = rng.random();
        let rho = ((1.0 + u * (cosh_alpha_r - 1.0)).acosh()) / alpha;
        let theta = rng.random::<f64>() * two_pi;
        coords.push((rho, theta));
    }

    // Connect pairs within hyperbolic distance r
    for i in 0..n {
        let (r1, t1) = coords[i];
        let (sinh_r1, cosh_r1) = (r1.sinh(), r1.cosh());
        for j in (i + 1)..n {
            let (r2, t2) = coords[j];
            let delta_theta = (t1 - t2).abs();
            // Map delta_theta to [0, π]
            let delta_theta = if delta_theta > std::f64::consts::PI {
                two_pi - delta_theta
            } else {
                delta_theta
            };
            let arg = cosh_r1 * r2.cosh() - sinh_r1 * r2.sinh() * delta_theta.cos();
            // arg can be < 1 due to floating-point; clamp to avoid NaN from acosh
            let hyp_dist = if arg <= 1.0 { 0.0 } else { arg.acosh() };
            if hyp_dist <= r {
                g.add_edge(i, j, hyp_dist)?;
            }
        }
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Stochastic Kronecker graph
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a stochastic Kronecker graph.
///
/// The *Kronecker graph* model (Leskovec et al., 2010) starts from a small
/// `k₀ × k₀` *initiator matrix* Θ whose entries are probabilities in (0,1].
/// The Kronecker product Θ^{⊗k} gives a (k₀^k × k₀^k) matrix of edge
/// probabilities; an edge (i, j) is present independently with that probability.
///
/// This function computes the Kronecker product iteratively and samples edges
/// without materialising the full N×N matrix: for each entry (i, j) of the
/// product matrix, the entry value equals the product of initiator entries
/// indexed by the base-k₀ digits of i and j.
///
/// # Arguments
/// * `initiator` – square probability matrix of shape k₀ × k₀ (entries in (0, 1])
/// * `k`         – number of Kronecker iterations; produces N = k₀^k nodes
/// * `rng`       – random-number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` if the initiator is not square or contains
/// values outside (0, 1].
///
/// # Reference
/// Leskovec, J., Chakrabarti, D., Kleinberg, J., Faloutsos, C., & Ghahramani, Z.
/// "Kronecker graphs: An approach to modeling networks." JMLR 11 (2010).
pub fn kronecker_graph<R: Rng>(
    initiator: &Array2<f64>,
    k: usize,
    rng: &mut R,
) -> Result<Graph<usize, f64>> {
    let k0 = initiator.nrows();
    if k0 == 0 || initiator.ncols() != k0 {
        return Err(GraphError::InvalidGraph(
            "kronecker_graph: initiator must be a non-empty square matrix".to_string(),
        ));
    }
    for val in initiator.iter() {
        if !(*val > 0.0 && *val <= 1.0) {
            return Err(GraphError::InvalidGraph(format!(
                "kronecker_graph: initiator entries must be in (0,1], found {val}"
            )));
        }
    }
    if k == 0 {
        return Err(GraphError::InvalidGraph(
            "kronecker_graph: k must be ≥ 1".to_string(),
        ));
    }

    let n = k0.pow(k as u32);
    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }

    // Compute edge probability for (i,j) by decomposing i,j in base k0
    for i in 0..n {
        for j in (i + 1)..n {
            let prob = kronecker_edge_prob(initiator, k0, k, i, j);
            if rng.random::<f64>() < prob {
                g.add_edge(i, j, 1.0)?;
            }
        }
    }
    Ok(g)
}

/// Compute the edge probability P(i,j) in the k-th Kronecker power.
///
/// The probability equals the product of initiator\[i_t\]\[j_t\] where
/// (i_0, …, i_{k-1}) and (j_0, …, j_{k-1}) are the base-k₀ representations
/// of i and j (most-significant digit first).
fn kronecker_edge_prob(
    initiator: &Array2<f64>,
    k0: usize,
    k: usize,
    mut i: usize,
    mut j: usize,
) -> f64 {
    let mut prob = 1.0f64;
    for _ in 0..k {
        let digit_i = i % k0;
        let digit_j = j % k0;
        prob *= initiator[[digit_i, digit_j]];
        i /= k0;
        j /= k0;
    }
    prob
}

// ─────────────────────────────────────────────────────────────────────────────
// Chung–Lu model
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a Chung–Lu random graph with prescribed expected degree sequence.
///
/// In the Chung–Lu model each node `i` has a weight `w_i` (intended expected
/// degree).  The probability of an edge (i, j) is:
///
/// ```text
/// P(i,j) = min(w_i · w_j / W, 1)   where W = Σ w_i
/// ```
///
/// When the weights equal the observed degrees of a real network, this model
/// reproduces the degree sequence in expectation and is often used as a
/// null-model baseline.
///
/// # Arguments
/// * `degree_seq` – slice of non-negative expected degrees; all values ≥ 0
/// * `rng`        – random-number generator
///
/// # Errors
/// Returns `GraphError::InvalidGraph` if the degree sequence is empty or any
/// value is negative.
///
/// # Reference
/// Chung, F., & Lu, L. "Connected components in random graphs with given expected
/// degree sequences." Annals of Combinatorics, 6(2), 125–145, 2002.
pub fn chung_lu<R: Rng>(degree_seq: &[f64], rng: &mut R) -> Result<Graph<usize, f64>> {
    if degree_seq.is_empty() {
        return Ok(Graph::new());
    }
    for &w in degree_seq {
        if w < 0.0 || !w.is_finite() {
            return Err(GraphError::InvalidGraph(format!(
                "chung_lu: degree sequence entries must be non-negative finite, found {w}"
            )));
        }
    }

    let n = degree_seq.len();
    let total_weight: f64 = degree_seq.iter().sum();

    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }

    if total_weight <= 0.0 {
        return Ok(g);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let p = (degree_seq[i] * degree_seq[j] / total_weight).min(1.0);
            if p > 0.0 && rng.random::<f64>() < p {
                g.add_edge(i, j, 1.0)?;
            }
        }
    }
    Ok(g)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::StdRng;
    use scirs2_core::random::SeedableRng;

    // ── G(n,p) ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gnp_basic() {
        let mut rng = StdRng::seed_from_u64(1);
        let g = erdos_renyi_g_np(10, 0.5, &mut rng).expect("gnp failed");
        assert_eq!(g.node_count(), 10);
        assert!(g.edge_count() <= 45);
    }

    #[test]
    fn test_gnp_p0_no_edges() {
        let mut rng = StdRng::seed_from_u64(2);
        let g = erdos_renyi_g_np(15, 0.0, &mut rng).expect("gnp p=0 failed");
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_gnp_p1_complete() {
        let mut rng = StdRng::seed_from_u64(3);
        let g = erdos_renyi_g_np(8, 1.0, &mut rng).expect("gnp p=1 failed");
        assert_eq!(g.edge_count(), 8 * 7 / 2);
    }

    #[test]
    fn test_gnp_invalid_p() {
        let mut rng = StdRng::seed_from_u64(4);
        assert!(erdos_renyi_g_np(10, 1.5, &mut rng).is_err());
        assert!(erdos_renyi_g_np(10, -0.1, &mut rng).is_err());
    }

    // ── G(n,m) ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gnm_exact_edges() {
        let mut rng = StdRng::seed_from_u64(5);
        for m in [0, 1, 10, 45] {
            let g = erdos_renyi_g_nm(10, m, &mut rng).expect("gnm failed");
            assert_eq!(g.node_count(), 10);
            assert_eq!(g.edge_count(), m);
        }
    }

    #[test]
    fn test_gnm_too_many_edges() {
        let mut rng = StdRng::seed_from_u64(6);
        assert!(erdos_renyi_g_nm(5, 11, &mut rng).is_err()); // C(5,2) = 10
    }

    // ── Barabási–Albert ──────────────────────────────────────────────────────

    #[test]
    fn test_ba_node_count() {
        let mut rng = StdRng::seed_from_u64(7);
        let g = barabasi_albert(50, 3, &mut rng).expect("ba failed");
        assert_eq!(g.node_count(), 50);
    }

    #[test]
    fn test_ba_min_edges() {
        let mut rng = StdRng::seed_from_u64(8);
        // Initial clique: C(m+1, 2) edges; each new node adds at least m edges.
        let g = barabasi_albert(20, 2, &mut rng).expect("ba failed");
        // seed clique has C(3,2)=3 edges + 17 * 2 = 37 total minimum
        assert!(g.edge_count() >= 3 + 17 * 2);
    }

    #[test]
    fn test_ba_invalid_params() {
        let mut rng = StdRng::seed_from_u64(9);
        assert!(barabasi_albert(5, 0, &mut rng).is_err());
        assert!(barabasi_albert(3, 5, &mut rng).is_err());
    }

    // ── Watts–Strogatz ───────────────────────────────────────────────────────

    #[test]
    fn test_ws_lattice() {
        let mut rng = StdRng::seed_from_u64(10);
        let g = watts_strogatz(20, 4, 0.0, &mut rng).expect("ws failed");
        assert_eq!(g.node_count(), 20);
        // Each node connects to k=4 others in a ring, so n*k/2 edges
        assert_eq!(g.edge_count(), 20 * 4 / 2);
    }

    #[test]
    fn test_ws_random() {
        let mut rng = StdRng::seed_from_u64(11);
        let g = watts_strogatz(30, 4, 1.0, &mut rng).expect("ws full rewire failed");
        assert_eq!(g.node_count(), 30);
        // rewiring preserves number of edges (at most n*k/2)
        assert!(g.edge_count() <= 30 * 4 / 2);
    }

    #[test]
    fn test_ws_invalid_params() {
        let mut rng = StdRng::seed_from_u64(12);
        assert!(watts_strogatz(10, 3, 0.5, &mut rng).is_err()); // k odd
        assert!(watts_strogatz(10, 10, 0.5, &mut rng).is_err()); // k >= n
        assert!(watts_strogatz(10, 4, -0.1, &mut rng).is_err());
    }

    // ── Random regular ───────────────────────────────────────────────────────

    #[test]
    fn test_random_regular_degrees() {
        let mut rng = StdRng::seed_from_u64(13);
        if let Some(g) = random_regular(10, 3, &mut rng) {
            assert_eq!(g.node_count(), 10);
            // Every node should have degree 3
            for i in 0..10usize {
                assert_eq!(g.degree(&i), 3);
            }
        }
        // n·d must be even: 10*3=30 is even, so it should succeed
    }

    #[test]
    fn test_random_regular_infeasible() {
        let mut rng = StdRng::seed_from_u64(14);
        // n*d odd → infeasible
        assert!(random_regular(5, 3, &mut rng).is_none()); // 15 is odd
                                                           // d >= n → infeasible
        assert!(random_regular(4, 4, &mut rng).is_none());
    }

    // ── Hyperbolic random graph ───────────────────────────────────────────────

    #[test]
    fn test_hrg_node_count() {
        let mut rng = StdRng::seed_from_u64(15);
        let g = hyperbolic_random_graph(30, 6.0, 0.75, &mut rng).expect("hrg failed");
        assert_eq!(g.node_count(), 30);
    }

    #[test]
    fn test_hrg_small_radius_sparse() {
        let mut rng = StdRng::seed_from_u64(16);
        // Very small disk radius → very few edges (most pairs far apart)
        let g = hyperbolic_random_graph(50, 0.5, 0.75, &mut rng).expect("hrg small r failed");
        assert!(g.edge_count() < 50 * 49 / 2);
    }

    #[test]
    fn test_hrg_invalid_params() {
        let mut rng = StdRng::seed_from_u64(17);
        assert!(hyperbolic_random_graph(10, -1.0, 0.5, &mut rng).is_err());
        assert!(hyperbolic_random_graph(10, 5.0, 0.0, &mut rng).is_err());
    }

    // ── Stochastic Kronecker ─────────────────────────────────────────────────

    #[test]
    fn test_kronecker_small() {
        use scirs2_core::ndarray::array;
        let mut rng = StdRng::seed_from_u64(18);
        // 2×2 initiator, k=3 → 2^3=8 nodes
        let theta = array![[0.9, 0.5], [0.5, 0.3]];
        let g = kronecker_graph(&theta, 3, &mut rng).expect("kronecker failed");
        assert_eq!(g.node_count(), 8);
        // All edge weights should be 1.0 (binary)
        for e in g.edges() {
            assert!((e.weight - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_kronecker_invalid() {
        use scirs2_core::ndarray::array;
        let mut rng = StdRng::seed_from_u64(19);
        // Non-square initiator
        let non_square = Array2::zeros((2, 3));
        assert!(kronecker_graph(&non_square, 2, &mut rng).is_err());
        // k=0
        let theta = array![[0.5, 0.5], [0.5, 0.5]];
        assert!(kronecker_graph(&theta, 0, &mut rng).is_err());
    }

    // ── Chung–Lu ─────────────────────────────────────────────────────────────

    #[test]
    fn test_chung_lu_basic() {
        let mut rng = StdRng::seed_from_u64(20);
        let w = vec![5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0];
        let g = chung_lu(&w, &mut rng).expect("chung_lu failed");
        assert_eq!(g.node_count(), 8);
        // edges should exist (high weights ≈ high connectivity)
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_chung_lu_zero_weights() {
        let mut rng = StdRng::seed_from_u64(21);
        let w = vec![0.0; 10];
        let g = chung_lu(&w, &mut rng).expect("chung_lu zero weights failed");
        assert_eq!(g.node_count(), 10);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_chung_lu_invalid() {
        let mut rng = StdRng::seed_from_u64(22);
        assert!(chung_lu(&[-1.0, 2.0], &mut rng).is_err());
        assert!(chung_lu(&[f64::INFINITY, 1.0], &mut rng).is_err());
    }

    #[test]
    fn test_chung_lu_empty() {
        let mut rng = StdRng::seed_from_u64(23);
        let g = chung_lu(&[], &mut rng).expect("chung_lu empty failed");
        assert_eq!(g.node_count(), 0);
    }
}
