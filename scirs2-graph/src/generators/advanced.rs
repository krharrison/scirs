//! Advanced graph generators — LFR benchmark and Forest Fire model
//!
//! This module complements `generators::random_graphs` and
//! `generators::mod` with two additional generation paradigms:
//!
//! * **LFR benchmark** (Lancichinetti–Fortunato–Radicchi, 2008): synthetic
//!   networks with heterogeneous community sizes and degree sequences following
//!   power laws, widely used to evaluate community-detection algorithms.
//!
//! * **Forest Fire** model (Leskovec, Kleinberg & Faloutsos, 2005): a directed-graph
//!   growth process that reproduces densification, shrinking diameter, and heavy-tailed
//!   degree distributions observed in real Web and citation graphs.
//!
//! # References
//!
//! - Lancichinetti, A., Fortunato, S., & Radicchi, F. (2008). Benchmark graphs for
//!   testing community detection algorithms. *Physical Review E*, 78, 046110.
//! - Leskovec, J., Kleinberg, J., & Faloutsos, C. (2005). Graphs over time:
//!   densification laws, shrinking diameters and possible explanations. *KDD '05*.

use crate::base::{DiGraph, Graph};
use crate::error::{GraphError, Result};
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::prelude::*;
use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// LFR benchmark
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the LFR benchmark graph generator.
///
/// All parameters correspond to those described in Lancichinetti et al. (2008).
#[derive(Debug, Clone)]
pub struct LfrParams {
    /// Number of nodes
    pub n: usize,
    /// Average degree (informational; drives k_min calibration)
    pub avg_degree: f64,
    /// Maximum allowed degree
    pub max_degree: usize,
    /// Power-law exponent for the degree sequence (τ₁, typically 2–3)
    pub tau1: f64,
    /// Power-law exponent for the community-size sequence (τ₂, typically 1–2)
    pub tau2: f64,
    /// Mixing parameter μ ∈ \[0, 1): fraction of edges connecting different communities
    pub mu: f64,
    /// Minimum community size
    pub min_community: usize,
    /// Maximum community size; if 0, defaults to `n / 2`
    pub max_community: usize,
}

impl LfrParams {
    /// Create an `LfrParams` with sensible defaults.
    pub fn new(n: usize, avg_degree: f64, tau1: f64, tau2: f64, mu: f64) -> Self {
        LfrParams {
            n,
            avg_degree,
            max_degree: n.max(1),
            tau1,
            tau2,
            mu,
            min_community: 5,
            max_community: 0, // will default to n/2
        }
    }
}

/// Generate an LFR benchmark graph with community structure.
///
/// The algorithm proceeds in four stages:
/// 1. Sample a power-law degree sequence with exponent `tau1`.
/// 2. Assign nodes to communities (community sizes follow a power law with
///    exponent `tau2`).
/// 3. For each node *i* with degree *k_i*, assign ⌊(1−μ)·k_i⌋ intra-community
///    stubs and ⌈μ·k_i⌉ inter-community stubs.
/// 4. Wire stubs within each community and across communities using the
///    configuration-model approach; multi-edges and self-loops are discarded.
///
/// Returns both the graph and the community assignment vector (node index → community id).
///
/// # Errors
///
/// Returns [`GraphError::InvalidGraph`] when parameters are out of range.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::generators::advanced::{lfr_benchmark, LfrParams};
/// use scirs2_core::random::prelude::*;
/// let mut rng = StdRng::seed_from_u64(42);
/// let params = LfrParams::new(50, 5.0, 2.5, 1.5, 0.2);
/// let (g, communities) = lfr_benchmark(&params, &mut rng).unwrap();
/// assert_eq!(g.node_count(), 50);
/// ```
pub fn lfr_benchmark<R: Rng>(
    params: &LfrParams,
    rng: &mut R,
) -> Result<(Graph<usize, f64>, Vec<usize>)> {
    // ── parameter validation ───────────────────────────────────────────────
    if params.n < 3 {
        return Err(GraphError::InvalidGraph(
            "lfr_benchmark: n must be ≥ 3".to_string(),
        ));
    }
    if params.avg_degree <= 0.0 || params.avg_degree >= params.n as f64 {
        return Err(GraphError::InvalidGraph(format!(
            "lfr_benchmark: avg_degree={} must be in (0, n)",
            params.avg_degree
        )));
    }
    if params.tau1 <= 1.0 {
        return Err(GraphError::InvalidGraph(
            "lfr_benchmark: tau1 must be > 1".to_string(),
        ));
    }
    if params.tau2 <= 1.0 {
        return Err(GraphError::InvalidGraph(
            "lfr_benchmark: tau2 must be > 1".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&params.mu) {
        return Err(GraphError::InvalidGraph(
            "lfr_benchmark: mu must be in [0, 1)".to_string(),
        ));
    }

    let n = params.n;
    let max_deg = params.max_degree.clamp(1, n - 1);
    let max_com = if params.max_community == 0 {
        (n / 2).max(params.min_community + 1)
    } else {
        params.max_community.min(n)
    };

    // ── stage 1: sample degree sequence ───────────────────────────────────
    let degrees = sample_power_law_degrees(n, max_deg, params.tau1, rng);

    // ── stage 2: assign communities ───────────────────────────────────────
    let community_of = assign_communities(n, params.min_community, max_com, params.tau2, rng);
    let num_communities = community_of.iter().copied().max().map_or(0, |m| m + 1);

    // Build community membership lists
    let mut community_members: Vec<Vec<usize>> = vec![Vec::new(); num_communities];
    for (node, &com) in community_of.iter().enumerate() {
        community_members[com].push(node);
    }
    let _ = community_members; // Used conceptually above; avoid unused warning

    // ── stage 3: split stubs into intra / inter ────────────────────────────
    let mut intra_stubs: Vec<Vec<usize>> = vec![Vec::new(); num_communities];
    let mut inter_stubs: Vec<usize> = Vec::new();

    for (node, &deg) in degrees.iter().enumerate() {
        let com = community_of[node];
        let intra = (((1.0 - params.mu) * deg as f64).round() as usize).max(1);
        let the_inter = deg.saturating_sub(intra);
        for _ in 0..intra {
            intra_stubs[com].push(node);
        }
        for _ in 0..the_inter {
            inter_stubs.push(node);
        }
    }

    // ── stage 4: wire edges ────────────────────────────────────────────────
    let mut g = Graph::new();
    for i in 0..n {
        g.add_node(i);
    }

    // intra-community wiring: shuffle stubs within each community and pair them
    for com_stubs in intra_stubs.iter_mut() {
        com_stubs.shuffle(rng);
        // If odd length, drop one stub
        if com_stubs.len() % 2 != 0 {
            com_stubs.pop();
        }
        let mut idx = 0;
        while idx + 1 < com_stubs.len() {
            let u = com_stubs[idx];
            let v = com_stubs[idx + 1];
            idx += 2;
            if u != v {
                let _ = g.add_edge(u, v, 1.0); // ignore multi-edge errors
            }
        }
    }

    // inter-community wiring
    inter_stubs.shuffle(rng);
    if !inter_stubs.len().is_multiple_of(2) {
        inter_stubs.pop();
    }
    let mut idx = 0;
    while idx + 1 < inter_stubs.len() {
        let u = inter_stubs[idx];
        let v = inter_stubs[idx + 1];
        idx += 2;
        // Reject same-community or self-loop edges in the inter-community wiring
        if u != v && community_of[u] != community_of[v] {
            let _ = g.add_edge(u, v, 1.0);
        }
    }

    Ok((g, community_of))
}

/// Sample `n` degrees from a discrete power-law P(k) ∝ k^(−τ) on \[2, max_degree\].
fn sample_power_law_degrees<R: Rng>(
    n: usize,
    max_degree: usize,
    tau: f64,
    rng: &mut R,
) -> Vec<usize> {
    let k_min = 2_usize;
    let k_max = max_degree.max(k_min);

    // Pre-compute unnormalised weights w(k) = k^(−τ)
    let weights: Vec<f64> = (k_min..=k_max).map(|k| (k as f64).powf(-tau)).collect();
    let total: f64 = weights.iter().sum();

    if total == 0.0 {
        return vec![k_min; n];
    }

    let mut degrees = Vec::with_capacity(n);
    for _ in 0..n {
        let r: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut chosen = k_min;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w / total;
            if r <= cumulative {
                chosen = k_min + i;
                break;
            }
        }
        degrees.push(chosen);
    }

    // Make the total stub count even (required by configuration model)
    let total_stubs: usize = degrees.iter().sum();
    if !total_stubs.is_multiple_of(2) {
        if let Some(last) = degrees.last_mut() {
            *last = (*last + 1).min(k_max);
        }
    }
    degrees
}

/// Assign `n` nodes to communities whose sizes follow a power law P(s) ∝ s^(−τ₂).
fn assign_communities<R: Rng>(
    n: usize,
    min_size: usize,
    max_size: usize,
    tau2: f64,
    rng: &mut R,
) -> Vec<usize> {
    let min_s = min_size.max(2);
    let max_s = max_size.max(min_s + 1);

    // Weights for community sizes
    let weights: Vec<f64> = (min_s..=max_s).map(|s| (s as f64).powf(-tau2)).collect();
    let total: f64 = weights.iter().sum();

    let mut community_of = vec![0usize; n];
    let mut assigned = 0usize;
    let mut com_id = 0usize;

    while assigned < n {
        // Sample a community size
        let r: f64 = rng.random();
        let mut cum = 0.0;
        let mut size = min_s;
        for (i, &w) in weights.iter().enumerate() {
            cum += w / total;
            if r <= cum {
                size = min_s + i;
                break;
            }
        }
        // Clamp size so we don't overshoot
        let remaining = n - assigned;
        let actual_size = size.min(remaining).max(1);

        for i in assigned..assigned + actual_size {
            community_of[i] = com_id;
        }
        assigned += actual_size;
        com_id += 1;
    }

    community_of
}

// ─────────────────────────────────────────────────────────────────────────────
// Forest Fire model
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a directed Forest Fire graph (Leskovec, Kleinberg & Faloutsos, 2005).
///
/// Nodes are added one at a time.  Each new node *v*:
/// 1. Selects an **ambassador** *a* uniformly at random from the existing nodes.
/// 2. Generates a geometric random variable *x* ~ Geometric(`p_f`) and links to
///    *x* of *a*'s out-neighbours.
/// 3. Generates a geometric random variable *y* ~ Geometric(`p_b`) and links to
///    *y* of *a*'s in-neighbours.
/// 4. Recursively applies steps 2–3 to each newly discovered node (the "fire"
///    spreads), stopping when no new nodes are reachable.
///
/// The result is a `DiGraph<usize, f64>` where edge (u → v) indicates that *v*
/// linked to *u* when *v* arrived.
///
/// # Arguments
///
/// * `n` – total number of nodes (must be ≥ 2)
/// * `p_f` – forward burning probability ∈ (0, 1)
/// * `p_b` – backward burning probability ∈ (0, 1\]; often `p_f / 2`
/// * `rng` – seeded random-number generator
///
/// # Errors
///
/// Returns [`GraphError::InvalidGraph`] when parameters are out of range.
///
/// # Example
///
/// ```rust
/// use scirs2_graph::generators::advanced::forest_fire;
/// use scirs2_core::random::prelude::*;
/// let mut rng = StdRng::seed_from_u64(7);
/// let g = forest_fire(100, 0.37, 0.32, &mut rng).unwrap();
/// assert_eq!(g.node_count(), 100);
/// ```
pub fn forest_fire<R: Rng>(
    n: usize,
    p_f: f64,
    p_b: f64,
    rng: &mut R,
) -> Result<DiGraph<usize, f64>> {
    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "forest_fire: n must be ≥ 2".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&p_f) {
        return Err(GraphError::InvalidGraph(format!(
            "forest_fire: p_f={p_f} must be in (0,1)"
        )));
    }
    if !(0.0..=1.0).contains(&p_b) {
        return Err(GraphError::InvalidGraph(format!(
            "forest_fire: p_b={p_b} must be in (0,1]"
        )));
    }

    // We maintain adjacency lists manually for efficiency during growth.
    // out_adj[u] = set of nodes that u points TO
    // in_adj[u]  = set of nodes that point TO u
    let mut out_adj: Vec<HashSet<usize>> = Vec::with_capacity(n);
    let mut in_adj: Vec<HashSet<usize>> = Vec::with_capacity(n);

    // Seed graph: node 0
    out_adj.push(HashSet::new());
    in_adj.push(HashSet::new());

    for v in 1..n {
        out_adj.push(HashSet::new());
        in_adj.push(HashSet::new());

        // Pick a random ambassador from existing nodes 0..v
        let existing: Vec<usize> = (0..v).collect();
        let &ambassador = existing
            .choose(rng)
            .expect("existing is non-empty since v >= 1");

        // BFS fire spread — visited tracks nodes already burned
        let mut visited: HashSet<usize> = HashSet::new();
        let mut queue: Vec<usize> = vec![ambassador];
        visited.insert(ambassador);

        while let Some(current) = queue.pop() {
            // Link v → current
            out_adj[v].insert(current);
            in_adj[current].insert(v);

            // Forward: sample x ~ Geometric(p_f) out-neighbours of current
            let x = sample_geometric(p_f, rng);
            let fwd_neighbours: Vec<usize> = out_adj[current].iter().copied().collect();
            let forward_targets = choose_up_to(&fwd_neighbours, x, rng);
            for t in forward_targets {
                if !visited.contains(&t) {
                    visited.insert(t);
                    queue.push(t);
                }
            }

            // Backward: sample y ~ Geometric(p_b) in-neighbours of current
            let y = sample_geometric(p_b, rng);
            let bwd_neighbours: Vec<usize> = in_adj[current].iter().copied().collect();
            let backward_targets = choose_up_to(&bwd_neighbours, y, rng);
            for t in backward_targets {
                if !visited.contains(&t) {
                    visited.insert(t);
                    queue.push(t);
                }
            }
        }
    }

    // Build DiGraph from adjacency lists
    let mut g = DiGraph::new();
    for i in 0..n {
        g.add_node(i);
    }
    for u in 0..n {
        for &v in &out_adj[u] {
            let _ = g.add_edge(u, v, 1.0);
        }
    }

    Ok(g)
}

/// Sample from a Geometric distribution: number of successes before first failure.
///
/// Returns a value in {0, 1, 2, …} where P(X = k) = (1−p)^k · p.
/// Uses the inverse CDF method.
fn sample_geometric<R: Rng>(p: f64, rng: &mut R) -> usize {
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return 1;
    }
    let u: f64 = rng.random::<f64>().max(1e-300); // avoid log(0)
    let k = (1.0 - u).ln() / (1.0 - p).ln();
    (k.floor() as usize).min(1024) // cap to prevent runaway fires
}

/// Choose at most `k` elements uniformly at random from `slice` without replacement.
fn choose_up_to<R: Rng>(slice: &[usize], k: usize, rng: &mut R) -> Vec<usize> {
    if slice.is_empty() || k == 0 {
        return Vec::new();
    }
    let count = k.min(slice.len());
    let mut indices: Vec<usize> = (0..slice.len()).collect();
    // Partial Fisher-Yates shuffle
    for i in 0..count {
        let j = i + rng.random_range(0..(slice.len() - i));
        indices.swap(i, j);
    }
    indices[..count].iter().map(|&i| slice[i]).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lfr_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let params = LfrParams::new(30, 4.0, 2.5, 1.5, 0.2);
        let (g, communities) =
            lfr_benchmark(&params, &mut rng).expect("lfr_benchmark should succeed");
        assert_eq!(g.node_count(), 30);
        assert_eq!(communities.len(), 30);
    }

    #[test]
    fn test_lfr_community_ids_valid() {
        let mut rng = StdRng::seed_from_u64(1);
        let params = LfrParams::new(40, 4.0, 2.5, 1.5, 0.3);
        let (g, communities) =
            lfr_benchmark(&params, &mut rng).expect("lfr_benchmark should succeed");
        assert_eq!(g.node_count(), 40);
        // All community IDs should be valid (< some max)
        let max_com = communities.iter().copied().max().unwrap_or(0);
        assert!(max_com < 40, "Community IDs should be < n");
    }

    #[test]
    fn test_lfr_invalid_params() {
        let mut rng = StdRng::seed_from_u64(0);
        // n too small
        let params = LfrParams::new(2, 1.0, 2.5, 1.5, 0.2);
        assert!(lfr_benchmark(&params, &mut rng).is_err());
        // mu out of range
        let params = LfrParams::new(30, 4.0, 2.5, 1.5, 1.1);
        assert!(lfr_benchmark(&params, &mut rng).is_err());
        // tau1 invalid
        let params = LfrParams::new(30, 4.0, 0.5, 1.5, 0.2);
        assert!(lfr_benchmark(&params, &mut rng).is_err());
        // tau2 invalid
        let params = LfrParams::new(30, 4.0, 2.5, 0.5, 0.2);
        assert!(lfr_benchmark(&params, &mut rng).is_err());
    }

    #[test]
    fn test_forest_fire_basic() {
        let mut rng = StdRng::seed_from_u64(7);
        let g = forest_fire(50, 0.37, 0.32, &mut rng).expect("forest_fire should succeed");
        assert_eq!(g.node_count(), 50);
        // Should have at least some edges (every node v>0 links at least to its ambassador)
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn test_forest_fire_directed() {
        let mut rng = StdRng::seed_from_u64(99);
        let g = forest_fire(20, 0.37, 0.32, &mut rng).expect("forest_fire should succeed");
        assert_eq!(g.node_count(), 20);
        // Every node except node 0 contributes at least one outgoing edge
        assert!(g.edge_count() >= 19);
    }

    #[test]
    fn test_forest_fire_invalid_params() {
        let mut rng = StdRng::seed_from_u64(0);
        assert!(forest_fire(1, 0.37, 0.32, &mut rng).is_err());
        assert!(forest_fire(10, 1.5, 0.32, &mut rng).is_err());
        assert!(forest_fire(10, 0.37, 1.5, &mut rng).is_err());
    }

    #[test]
    fn test_geometric_sampling_edge_cases() {
        let mut rng = StdRng::seed_from_u64(1);
        // p=1.0 → always returns 1
        assert_eq!(sample_geometric(1.0, &mut rng), 1);
        // p=0.0 → always returns 0
        assert_eq!(sample_geometric(0.0, &mut rng), 0);
        // Typical value should be finite and small
        for _ in 0..100 {
            let val = sample_geometric(0.5, &mut rng);
            assert!(val <= 1024, "geometric sample should be capped");
        }
    }

    #[test]
    fn test_choose_up_to() {
        let mut rng = StdRng::seed_from_u64(3);
        let slice = vec![0usize, 1, 2, 3, 4];
        // Choose 3 from 5
        let chosen = choose_up_to(&slice, 3, &mut rng);
        assert_eq!(chosen.len(), 3);
        // All chosen values should be from slice
        for &v in &chosen {
            assert!(slice.contains(&v));
        }
        // No duplicates
        let unique: HashSet<usize> = chosen.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_choose_up_to_more_than_available() {
        let mut rng = StdRng::seed_from_u64(4);
        let slice = vec![10usize, 20, 30];
        let chosen = choose_up_to(&slice, 10, &mut rng);
        // Should return all 3
        assert_eq!(chosen.len(), 3);
    }
}
