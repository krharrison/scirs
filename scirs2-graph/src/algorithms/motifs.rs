//! Graph motif finding and subgraph analysis algorithms
//!
//! This module contains algorithms for finding small recurring subgraph patterns (motifs),
//! subgraph isomorphism, graph kernels, and graphlet analysis.
//!
//! # Algorithms
//! - **Motif counting**: Enumeration of 3-node and 4-node motifs
//! - **VF2 subgraph isomorphism**: State-space search for subgraph matching
//! - **Weisfeiler-Lehman subtree kernel**: Graph similarity via label refinement
//! - **Graphlet degree distribution**: Node-level graphlet statistics
//! - **Frequent subgraph mining**: gSpan-like approach for common patterns

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Motif types for enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotifType {
    /// Triangle (3-cycle)
    Triangle,
    /// Square (4-cycle)
    Square,
    /// Star with 3 leaves
    Star3,
    /// Clique of size 4
    Clique4,
    /// Path of length 3 (4 nodes)
    Path3,
    /// Bi-fan motif (2 nodes connected to 2 other nodes)
    BiFan,
    /// Feed-forward loop
    FeedForwardLoop,
    /// Bi-directional motif
    BiDirectional,
}

/// Result of VF2 subgraph isomorphism
#[derive(Debug, Clone)]
pub struct VF2Result<N: Node> {
    /// Whether a match was found
    pub is_match: bool,
    /// All found mappings from pattern to target
    pub mappings: Vec<HashMap<N, N>>,
    /// Number of states explored
    pub states_explored: usize,
}

/// Result of Weisfeiler-Lehman subtree kernel computation
#[derive(Debug, Clone)]
pub struct WLKernelResult {
    /// Kernel matrix (similarity between pairs of graphs)
    pub kernel_matrix: Vec<Vec<f64>>,
    /// Feature vectors for each graph
    pub feature_vectors: Vec<HashMap<String, usize>>,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Graphlet degree distribution for a node
#[derive(Debug, Clone)]
pub struct GraphletDegreeVector {
    /// Number of graphlet orbits the node participates in
    pub orbit_counts: Vec<usize>,
}

/// Result of graphlet degree distribution analysis
#[derive(Debug, Clone)]
pub struct GraphletDDResult<N: Node> {
    /// GDV for each node
    pub node_gdvs: HashMap<N, GraphletDegreeVector>,
    /// Agreement between GDVs (similarity metric)
    pub gdv_agreement: f64,
}

/// Frequent subgraph pattern
#[derive(Debug, Clone)]
pub struct FrequentPattern<N: Node> {
    /// Nodes in the pattern
    pub nodes: Vec<N>,
    /// Edges in the pattern (as index pairs into nodes)
    pub edges: Vec<(usize, usize)>,
    /// Support count (number of occurrences)
    pub support: usize,
}

// ============================================================================
// Motif finding
// ============================================================================

/// Find all occurrences of a specified motif in the graph.
pub fn find_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>, motiftype: MotifType) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    match motiftype {
        MotifType::Triangle => find_triangles(graph),
        MotifType::Square => find_squares(graph),
        MotifType::Star3 => find_star3s(graph),
        MotifType::Clique4 => find_clique4s(graph),
        MotifType::Path3 => find_path3s(graph),
        MotifType::BiFan => find_bi_fans(graph),
        MotifType::FeedForwardLoop => find_feed_forward_loops(graph),
        MotifType::BiDirectional => find_bidirectional_motifs(graph),
    }
}

/// Count all 3-node and 4-node motif types in the graph.
pub fn count_motif_frequencies<N, E, Ix>(graph: &Graph<N, E, Ix>) -> HashMap<MotifType, usize>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;

    let motif_types = vec![
        MotifType::Triangle,
        MotifType::Square,
        MotifType::Star3,
        MotifType::Clique4,
        MotifType::Path3,
        MotifType::BiFan,
        MotifType::FeedForwardLoop,
        MotifType::BiDirectional,
    ];

    motif_types
        .par_iter()
        .map(|motif_type| {
            let count = find_motifs(graph, *motif_type).len();
            (*motif_type, count)
        })
        .collect()
}

/// Count 3-node motifs efficiently: open triads vs closed triads (triangles).
pub fn count_3node_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>) -> (usize, usize)
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let triangles = find_triangles(graph).len();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Count open triads (paths of length 2)
    let mut open_triads = 0usize;
    for node in &nodes {
        let deg = graph.degree(node);
        if deg >= 2 {
            // Number of pairs of neighbors: C(deg, 2)
            open_triads += deg * (deg - 1) / 2;
        }
    }
    // Each triangle contributes 3 "closed" triads, so open = total - 3*triangles
    let open = open_triads.saturating_sub(3 * triangles);

    (triangles, open)
}

/// Count 4-node motifs: paths, stars, squares, cliques.
pub fn count_4node_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>) -> HashMap<&'static str, usize>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut counts = HashMap::new();
    counts.insert("path3", find_path3s(graph).len());
    counts.insert("star3", find_star3s(graph).len());
    counts.insert("square", find_squares(graph).len());
    counts.insert("clique4", find_clique4s(graph).len());
    counts
}

/// Efficient motif detection using sampling for large graphs.
pub fn sample_motif_frequencies<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    sample_size: usize,
    rng: &mut impl scirs2_core::random::Rng,
) -> HashMap<MotifType, f64>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::random::seq::SliceRandom;

    let all_nodes: Vec<_> = graph.nodes().into_iter().cloned().collect();
    if all_nodes.len() <= sample_size {
        return count_motif_frequencies(graph)
            .into_iter()
            .map(|(k, v)| (k, v as f64))
            .collect();
    }

    let mut sampled_nodes = all_nodes.clone();
    sampled_nodes.shuffle(rng);
    sampled_nodes.truncate(sample_size);

    let mut subgraph = crate::generators::create_graph::<N, E>();
    for node in &sampled_nodes {
        let _ = subgraph.add_node(node.clone());
    }

    for node1 in &sampled_nodes {
        if let Ok(neighbors) = graph.neighbors(node1) {
            for node2 in neighbors {
                if sampled_nodes.contains(&node2) && node1 != &node2 {
                    if let Ok(weight) = graph.edge_weight(node1, &node2) {
                        let _ = subgraph.add_edge(node1.clone(), node2, weight);
                    }
                }
            }
        }
    }

    let subgraph_counts = count_motif_frequencies(&subgraph);
    let scaling_factor = (all_nodes.len() as f64) / (sample_size as f64);

    subgraph_counts
        .into_iter()
        .map(|(motif_type, count)| (motif_type, count as f64 * scaling_factor))
        .collect()
}

// ============================================================================
// VF2 Subgraph Isomorphism
// ============================================================================

/// VF2 subgraph isomorphism algorithm.
///
/// Determines if `pattern` is isomorphic to a subgraph of `target`.
/// Uses the VF2 state-space representation for efficient pruning.
///
/// # Arguments
/// * `pattern` - The pattern graph to search for
/// * `target` - The target graph to search in
/// * `max_matches` - Maximum number of matches to find (0 = all)
pub fn vf2_subgraph_isomorphism<N, E, Ix>(
    pattern: &Graph<N, E, Ix>,
    target: &Graph<N, E, Ix>,
    max_matches: usize,
) -> VF2Result<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let p_nodes: Vec<N> = pattern.nodes().into_iter().cloned().collect();
    let t_nodes: Vec<N> = target.nodes().into_iter().cloned().collect();

    if p_nodes.is_empty() {
        return VF2Result {
            is_match: true,
            mappings: vec![HashMap::new()],
            states_explored: 0,
        };
    }

    if p_nodes.len() > t_nodes.len() {
        return VF2Result {
            is_match: false,
            mappings: vec![],
            states_explored: 0,
        };
    }

    // Build adjacency sets for fast lookup
    let p_adj = build_adj_set(pattern, &p_nodes);
    let t_adj = build_adj_set(target, &t_nodes);

    let p_idx: HashMap<N, usize> = p_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let t_idx: HashMap<N, usize> = t_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let mut state = VF2State {
        core_p: vec![None; p_nodes.len()],
        core_t: vec![None; t_nodes.len()],
        in_p: vec![0usize; p_nodes.len()],
        out_p: vec![0usize; p_nodes.len()],
        in_t: vec![0usize; t_nodes.len()],
        out_t: vec![0usize; t_nodes.len()],
        depth: 0,
    };

    let mut mappings = Vec::new();
    let mut states_explored = 0;
    let limit = if max_matches == 0 {
        usize::MAX
    } else {
        max_matches
    };

    vf2_match(
        &p_nodes,
        &t_nodes,
        &p_adj,
        &t_adj,
        &p_idx,
        &t_idx,
        &mut state,
        &mut mappings,
        &mut states_explored,
        limit,
    );

    VF2Result {
        is_match: !mappings.is_empty(),
        mappings,
        states_explored,
    }
}

/// Internal VF2 state
struct VF2State {
    core_p: Vec<Option<usize>>, // pattern node -> target node
    core_t: Vec<Option<usize>>, // target node -> pattern node
    in_p: Vec<usize>,
    out_p: Vec<usize>,
    in_t: Vec<usize>,
    out_t: Vec<usize>,
    depth: usize,
}

fn build_adj_set<N, E, Ix>(graph: &Graph<N, E, Ix>, nodes: &[N]) -> Vec<HashSet<usize>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let idx_map: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let mut adj = vec![HashSet::new(); nodes.len()];
    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&j) = idx_map.get(neighbor) {
                    adj[i].insert(j);
                }
            }
        }
    }
    adj
}

fn vf2_match<N: Node + Clone + Hash + Eq + std::fmt::Debug>(
    p_nodes: &[N],
    t_nodes: &[N],
    p_adj: &[HashSet<usize>],
    t_adj: &[HashSet<usize>],
    _p_idx: &HashMap<N, usize>,
    _t_idx: &HashMap<N, usize>,
    state: &mut VF2State,
    mappings: &mut Vec<HashMap<N, N>>,
    states_explored: &mut usize,
    limit: usize,
) {
    *states_explored += 1;

    if mappings.len() >= limit {
        return;
    }

    if state.depth == p_nodes.len() {
        // Complete matching found
        let mut mapping = HashMap::new();
        for (pi, ti_opt) in state.core_p.iter().enumerate() {
            if let Some(ti) = ti_opt {
                mapping.insert(p_nodes[pi].clone(), t_nodes[*ti].clone());
            }
        }
        mappings.push(mapping);
        return;
    }

    // Find candidate pairs
    let candidates = vf2_candidates(state, p_nodes.len(), t_nodes.len());

    for (p, t) in candidates {
        if vf2_feasible(p, t, p_adj, t_adj, state, p_nodes.len(), t_nodes.len()) {
            // Add pair to mapping
            state.core_p[p] = Some(t);
            state.core_t[t] = Some(p);
            let old_depth = state.depth;
            state.depth += 1;

            // Update in/out sets
            let mut in_p_changes = Vec::new();
            let mut out_p_changes = Vec::new();
            let mut in_t_changes = Vec::new();
            let mut out_t_changes = Vec::new();

            for &neighbor in &p_adj[p] {
                if state.in_p[neighbor] == 0 && state.core_p[neighbor].is_none() {
                    state.in_p[neighbor] = state.depth;
                    in_p_changes.push(neighbor);
                }
                if state.out_p[neighbor] == 0 && state.core_p[neighbor].is_none() {
                    state.out_p[neighbor] = state.depth;
                    out_p_changes.push(neighbor);
                }
            }
            for &neighbor in &t_adj[t] {
                if state.in_t[neighbor] == 0 && state.core_t[neighbor].is_none() {
                    state.in_t[neighbor] = state.depth;
                    in_t_changes.push(neighbor);
                }
                if state.out_t[neighbor] == 0 && state.core_t[neighbor].is_none() {
                    state.out_t[neighbor] = state.depth;
                    out_t_changes.push(neighbor);
                }
            }

            vf2_match(
                p_nodes,
                t_nodes,
                p_adj,
                t_adj,
                _p_idx,
                _t_idx,
                state,
                mappings,
                states_explored,
                limit,
            );

            // Restore state
            state.core_p[p] = None;
            state.core_t[t] = None;
            state.depth = old_depth;

            for idx in in_p_changes {
                state.in_p[idx] = 0;
            }
            for idx in out_p_changes {
                state.out_p[idx] = 0;
            }
            for idx in in_t_changes {
                state.in_t[idx] = 0;
            }
            for idx in out_t_changes {
                state.out_t[idx] = 0;
            }

            if mappings.len() >= limit {
                return;
            }
        }
    }
}

fn vf2_candidates(state: &VF2State, n_p: usize, n_t: usize) -> Vec<(usize, usize)> {
    // Find the first unmapped pattern node
    let p_node = (0..n_p).find(|&i| state.core_p[i].is_none());

    if let Some(p) = p_node {
        // Try all unmapped target nodes
        let candidates: Vec<(usize, usize)> = (0..n_t)
            .filter(|&t| state.core_t[t].is_none())
            .map(|t| (p, t))
            .collect();
        candidates
    } else {
        vec![]
    }
}

fn vf2_feasible(
    p: usize,
    t: usize,
    p_adj: &[HashSet<usize>],
    t_adj: &[HashSet<usize>],
    state: &VF2State,
    _n_p: usize,
    _n_t: usize,
) -> bool {
    // Check: for every neighbor of p that is mapped, t must be a neighbor of the mapping
    for &p_neighbor in &p_adj[p] {
        if let Some(t_mapped) = state.core_p[p_neighbor] {
            if !t_adj[t].contains(&t_mapped) {
                return false;
            }
        }
    }

    // Check: for every neighbor of t that is mapped, the reverse must hold
    for &t_neighbor in &t_adj[t] {
        if let Some(p_mapped) = state.core_t[t_neighbor] {
            if !p_adj[p].contains(&p_mapped) {
                return false;
            }
        }
    }

    // Lookahead: degree compatibility
    // The number of unmapped neighbors of p in pattern must not exceed
    // the number of unmapped neighbors of t in target
    let p_unmapped_neighbors = p_adj[p]
        .iter()
        .filter(|&&n| state.core_p[n].is_none())
        .count();
    let t_unmapped_neighbors = t_adj[t]
        .iter()
        .filter(|&&n| state.core_t[n].is_none())
        .count();

    p_unmapped_neighbors <= t_unmapped_neighbors
}

// ============================================================================
// Weisfeiler-Lehman Subtree Kernel
// ============================================================================

/// Compute the Weisfeiler-Lehman subtree kernel between a collection of graphs.
///
/// The WL kernel iteratively refines node labels by aggregating neighbor labels.
/// After `h` iterations, it computes a feature vector of label histograms
/// and returns the dot product as the kernel value.
///
/// # Arguments
/// * `graphs` - Collection of graphs to compare
/// * `iterations` - Number of WL iterations (depth of subtree consideration)
pub fn wl_subtree_kernel<N, E, Ix>(graphs: &[&Graph<N, E, Ix>], iterations: usize) -> WLKernelResult
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight,
    Ix: IndexType,
{
    let num_graphs = graphs.len();

    if num_graphs == 0 {
        return WLKernelResult {
            kernel_matrix: vec![],
            feature_vectors: vec![],
            iterations: 0,
        };
    }

    // Initialize labels: use degree as initial label
    let mut graph_labels: Vec<HashMap<N, String>> = graphs
        .iter()
        .map(|g| {
            let mut labels = HashMap::new();
            for node in g.nodes() {
                labels.insert(node.clone(), format!("{}", g.degree(node)));
            }
            labels
        })
        .collect();

    let mut all_features: Vec<HashMap<String, usize>> = vec![HashMap::new(); num_graphs];

    // Collect initial labels as features
    for (gi, labels) in graph_labels.iter().enumerate() {
        for label in labels.values() {
            *all_features[gi].entry(label.clone()).or_insert(0) += 1;
        }
    }

    // WL iterations
    for _iter in 0..iterations {
        let mut new_graph_labels: Vec<HashMap<N, String>> = Vec::with_capacity(num_graphs);

        for (gi, graph) in graphs.iter().enumerate() {
            let mut new_labels = HashMap::new();

            for node in graph.nodes() {
                let own_label = graph_labels[gi].get(node).cloned().unwrap_or_default();

                // Collect sorted neighbor labels
                let mut neighbor_labels: Vec<String> = Vec::new();
                if let Ok(neighbors) = graph.neighbors(node) {
                    for neighbor in &neighbors {
                        if let Some(label) = graph_labels[gi].get(neighbor) {
                            neighbor_labels.push(label.clone());
                        }
                    }
                }
                neighbor_labels.sort();

                // New label = hash of (own_label, sorted neighbor labels)
                let new_label = format!("{own_label}|{}", neighbor_labels.join(","));
                new_labels.insert(node.clone(), new_label.clone());

                *all_features[gi].entry(new_label).or_insert(0) += 1;
            }

            new_graph_labels.push(new_labels);
        }

        graph_labels = new_graph_labels;
    }

    // Compute kernel matrix from feature vectors
    let mut kernel_matrix = vec![vec![0.0f64; num_graphs]; num_graphs];

    for i in 0..num_graphs {
        for j in i..num_graphs {
            // Dot product of feature vectors
            let mut dot = 0.0;
            for (key, &count_i) in &all_features[i] {
                if let Some(&count_j) = all_features[j].get(key) {
                    dot += (count_i as f64) * (count_j as f64);
                }
            }
            kernel_matrix[i][j] = dot;
            kernel_matrix[j][i] = dot;
        }
    }

    WLKernelResult {
        kernel_matrix,
        feature_vectors: all_features,
        iterations,
    }
}

// ============================================================================
// Graphlet Degree Distribution
// ============================================================================

/// Compute graphlet degree vectors for all nodes.
///
/// For each node, counts how many times it participates in each
/// graphlet orbit. Currently supports up to 4-node graphlets:
/// - Orbit 0: Edge (degree)
/// - Orbit 1: Path of length 2 (center)
/// - Orbit 2: Path of length 2 (endpoint)
/// - Orbit 3: Triangle
/// - Orbit 4: Path of length 3 (endpoint)
/// - Orbit 5: Path of length 3 (internal)
/// - Orbit 6: Star with 3 leaves (center)
/// - Orbit 7: Star with 3 leaves (leaf)
/// - Orbit 8: 4-cycle
/// - Orbit 9: 4-clique
pub fn graphlet_degree_distribution<N, E, Ix>(graph: &Graph<N, E, Ix>) -> GraphletDDResult<N>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let num_orbits = 10;

    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build adjacency list as indices
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&j) = node_to_idx.get(neighbor) {
                    adj[i].push(j);
                }
            }
        }
    }

    let mut orbits = vec![vec![0usize; num_orbits]; n];

    // Orbit 0: Degree (number of edges)
    for i in 0..n {
        orbits[i][0] = adj[i].len();
    }

    // Orbit 1,2: Paths of length 2
    for i in 0..n {
        for &j in &adj[i] {
            for &k in &adj[j] {
                if k != i && !adj[i].contains(&k) {
                    orbits[j][1] += 1; // center of path
                    orbits[i][2] += 1; // endpoint of path
                }
            }
        }
    }
    // Correct for double counting
    for i in 0..n {
        orbits[i][1] /= 2;
    }

    // Orbit 3: Triangles
    for i in 0..n {
        for (idx_j, &j) in adj[i].iter().enumerate() {
            if j <= i {
                continue;
            }
            for &k in adj[i].iter().skip(idx_j + 1) {
                if k <= j {
                    continue;
                }
                if adj[j].contains(&k) {
                    orbits[i][3] += 1;
                    orbits[j][3] += 1;
                    orbits[k][3] += 1;
                }
            }
        }
    }

    // Orbit 6,7: Star with 3 leaves
    for i in 0..n {
        let deg = adj[i].len();
        if deg >= 3 {
            // C(deg, 3) = number of 3-leaf stars centered at i
            // (only count those where leaves are NOT connected to each other)
            let mut star_count = 0;
            for j_idx in 0..adj[i].len() {
                for k_idx in (j_idx + 1)..adj[i].len() {
                    for l_idx in (k_idx + 1)..adj[i].len() {
                        let j = adj[i][j_idx];
                        let k = adj[i][k_idx];
                        let l = adj[i][l_idx];
                        if !adj[j].contains(&k) && !adj[k].contains(&l) && !adj[j].contains(&l) {
                            star_count += 1;
                            orbits[j][7] += 1;
                            orbits[k][7] += 1;
                            orbits[l][7] += 1;
                        }
                    }
                }
            }
            orbits[i][6] = star_count;
        }
    }

    // Orbits 4,5: Path of length 3
    for i in 0..n {
        for &j in &adj[i] {
            for &k in &adj[j] {
                if k == i {
                    continue;
                }
                for &l in &adj[k] {
                    if l == j || l == i {
                        continue;
                    }
                    // Path i-j-k-l
                    if !adj[i].contains(&k) && !adj[i].contains(&l) && !adj[j].contains(&l) {
                        orbits[i][4] += 1; // endpoint
                        orbits[l][4] += 1; // endpoint
                        orbits[j][5] += 1; // internal
                        orbits[k][5] += 1; // internal
                    }
                }
            }
        }
    }
    // Correct for double-counting (each path counted from both endpoints)
    for i in 0..n {
        orbits[i][4] /= 2;
        orbits[i][5] /= 2;
    }

    // Build result
    let mut node_gdvs = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        node_gdvs.insert(
            node.clone(),
            GraphletDegreeVector {
                orbit_counts: orbits[i].clone(),
            },
        );
    }

    // Compute GDV agreement (average over all node pairs)
    let mut total_agreement = 0.0;
    let mut pair_count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut agree = 0.0;
            for o in 0..num_orbits {
                let a = orbits[i][o] as f64;
                let b = orbits[j][o] as f64;
                let max_val = a.max(b);
                if max_val > 0.0 {
                    agree += 1.0 - (a - b).abs() / max_val;
                } else {
                    agree += 1.0;
                }
            }
            total_agreement += agree / num_orbits as f64;
            pair_count += 1;
        }
    }

    let gdv_agreement = if pair_count > 0 {
        total_agreement / pair_count as f64
    } else {
        1.0
    };

    GraphletDDResult {
        node_gdvs,
        gdv_agreement,
    }
}

// ============================================================================
// Frequent Subgraph Mining (simplified gSpan-like)
// ============================================================================

/// Find frequent subgraph patterns via edge enumeration.
///
/// A simplified version of gSpan: enumerates small connected subgraphs
/// and counts their support. Returns patterns with support >= min_support.
///
/// # Arguments
/// * `graph` - The graph to mine
/// * `min_support` - Minimum number of occurrences for a pattern to be frequent
/// * `max_size` - Maximum number of nodes in a pattern
pub fn frequent_subgraph_mining<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    min_support: usize,
    max_size: usize,
) -> Vec<FrequentPattern<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 || max_size == 0 {
        return vec![];
    }

    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for (i, node) in nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(node) {
            for neighbor in &neighbors {
                if let Some(&j) = node_to_idx.get(neighbor) {
                    adj[i].insert(j);
                }
            }
        }
    }

    let mut patterns: Vec<FrequentPattern<N>> = Vec::new();

    // Level 1: Single edges
    let mut edge_canonical: HashMap<(usize, usize), usize> = HashMap::new();
    for i in 0..n {
        for &j in &adj[i] {
            if i < j {
                let deg_i = adj[i].len();
                let deg_j = adj[j].len();
                let canonical = if deg_i <= deg_j {
                    (deg_i, deg_j)
                } else {
                    (deg_j, deg_i)
                };
                *edge_canonical.entry(canonical).or_insert(0) += 1;
            }
        }
    }

    for (&(deg_a, deg_b), &count) in &edge_canonical {
        if count >= min_support {
            // Find one representative instance
            for i in 0..n {
                for &j in &adj[i] {
                    if i < j {
                        let di = adj[i].len();
                        let dj = adj[j].len();
                        let c = if di <= dj { (di, dj) } else { (dj, di) };
                        if c == (deg_a, deg_b) {
                            patterns.push(FrequentPattern {
                                nodes: vec![nodes[i].clone(), nodes[j].clone()],
                                edges: vec![(0, 1)],
                                support: count,
                            });
                            break;
                        }
                    }
                }
                if !patterns.is_empty()
                    && patterns.last().map(|p| p.support == count).unwrap_or(false)
                {
                    break;
                }
            }
        }
    }

    // Level 2: Triangles (3-node connected subgraphs with 3 edges)
    if max_size >= 3 {
        let triangles = find_triangles(graph);
        if triangles.len() >= min_support {
            if let Some(tri) = triangles.first() {
                patterns.push(FrequentPattern {
                    nodes: tri.clone(),
                    edges: vec![(0, 1), (1, 2), (0, 2)],
                    support: triangles.len(),
                });
            }
        }

        // Paths of length 2 (3 nodes, 2 edges)
        let mut path2_count = 0;
        let mut path2_example: Option<Vec<N>> = None;
        for i in 0..n {
            for &j in &adj[i] {
                for &k in &adj[j] {
                    if k > i && !adj[i].contains(&k) {
                        path2_count += 1;
                        if path2_example.is_none() {
                            path2_example =
                                Some(vec![nodes[i].clone(), nodes[j].clone(), nodes[k].clone()]);
                        }
                    }
                }
            }
        }
        path2_count /= 2; // Each path counted twice

        if path2_count >= min_support {
            if let Some(example) = path2_example {
                patterns.push(FrequentPattern {
                    nodes: example,
                    edges: vec![(0, 1), (1, 2)],
                    support: path2_count,
                });
            }
        }
    }

    // Level 3: 4-node patterns
    if max_size >= 4 {
        let squares = find_squares(graph);
        if squares.len() >= min_support {
            if let Some(sq) = squares.first() {
                patterns.push(FrequentPattern {
                    nodes: sq.clone(),
                    edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
                    support: squares.len(),
                });
            }
        }
    }

    // Sort by support descending
    patterns.sort_by(|a, b| b.support.cmp(&a.support));
    patterns
}

// ============================================================================
// Internal motif finding functions
// ============================================================================

fn find_triangles<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let triangles = Mutex::new(Vec::new());

    nodes.par_iter().enumerate().for_each(|(_i, node_i)| {
        if let Ok(neighbors_i) = graph.neighbors(node_i) {
            let neighbors_i: Vec<_> = neighbors_i;

            for (j, node_j) in neighbors_i.iter().enumerate() {
                for node_k in neighbors_i.iter().skip(j + 1) {
                    if graph.has_edge(node_j, node_k) {
                        if let Ok(mut triangles_guard) = triangles.lock() {
                            let mut triangle = vec![node_i.clone(), node_j.clone(), node_k.clone()];
                            triangle.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                            if !triangles_guard.iter().any(|t| t == &triangle) {
                                triangles_guard.push(triangle);
                            }
                        }
                    }
                }
            }
        }
    });

    triangles.into_inner().unwrap_or_default()
}

fn find_squares<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut squares = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[k], &nodes[l])
                        && graph.has_edge(&nodes[l], &nodes[i])
                        && !graph.has_edge(&nodes[i], &nodes[k])
                        && !graph.has_edge(&nodes[j], &nodes[l])
                    {
                        squares.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    squares
}

fn find_star3s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut stars = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    for center in &nodes {
        if let Ok(neighbors) = graph.neighbors(center) {
            let neighbor_list: Vec<N> = neighbors;

            if neighbor_list.len() >= 3 {
                for i in 0..neighbor_list.len() {
                    for j in i + 1..neighbor_list.len() {
                        for k in j + 1..neighbor_list.len() {
                            if !graph.has_edge(&neighbor_list[i], &neighbor_list[j])
                                && !graph.has_edge(&neighbor_list[j], &neighbor_list[k])
                                && !graph.has_edge(&neighbor_list[i], &neighbor_list[k])
                            {
                                stars.push(vec![
                                    center.clone(),
                                    neighbor_list[i].clone(),
                                    neighbor_list[j].clone(),
                                    neighbor_list[k].clone(),
                                ]);
                            }
                        }
                    }
                }
            }
        }
    }

    stars
}

fn find_clique4s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    let mut cliques = Vec::new();
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    for i in 0..nodes.len() {
        for j in i + 1..nodes.len() {
            if !graph.has_edge(&nodes[i], &nodes[j]) {
                continue;
            }
            for k in j + 1..nodes.len() {
                if !graph.has_edge(&nodes[i], &nodes[k]) || !graph.has_edge(&nodes[j], &nodes[k]) {
                    continue;
                }
                for l in k + 1..nodes.len() {
                    if graph.has_edge(&nodes[i], &nodes[l])
                        && graph.has_edge(&nodes[j], &nodes[l])
                        && graph.has_edge(&nodes[k], &nodes[l])
                    {
                        cliques.push(vec![
                            nodes[i].clone(),
                            nodes[j].clone(),
                            nodes[k].clone(),
                            nodes[l].clone(),
                        ]);
                    }
                }
            }
        }
    }

    cliques
}

fn find_path3s<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let paths = Mutex::new(Vec::new());

    nodes.par_iter().for_each(|start_node| {
        if let Ok(neighbors1) = graph.neighbors(start_node) {
            for middle1 in neighbors1 {
                if let Ok(neighbors2) = graph.neighbors(&middle1) {
                    for middle2 in neighbors2 {
                        if middle2 == *start_node {
                            continue;
                        }

                        if let Ok(neighbors3) = graph.neighbors(&middle2) {
                            for end_node in neighbors3 {
                                if end_node == middle1 || end_node == *start_node {
                                    continue;
                                }

                                if !graph.has_edge(start_node, &middle2)
                                    && !graph.has_edge(start_node, &end_node)
                                    && !graph.has_edge(&middle1, &end_node)
                                {
                                    let mut path = vec![
                                        start_node.clone(),
                                        middle1.clone(),
                                        middle2.clone(),
                                        end_node.clone(),
                                    ];
                                    path.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                                    if let Ok(mut paths_guard) = paths.lock() {
                                        if !paths_guard.iter().any(|p| p == &path) {
                                            paths_guard.push(path);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    paths.into_inner().unwrap_or_default()
}

fn find_bi_fans<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let bi_fans = Mutex::new(Vec::new());

    nodes.par_iter().enumerate().for_each(|(i, node1)| {
        for node2 in nodes.iter().skip(i + 1) {
            if let (Ok(neighbors1), Ok(neighbors2)) =
                (graph.neighbors(node1), graph.neighbors(node2))
            {
                let neighbors1: HashSet<_> = neighbors1.into_iter().collect();
                let neighbors2: HashSet<_> = neighbors2.into_iter().collect();

                let common: Vec<_> = neighbors1
                    .intersection(&neighbors2)
                    .filter(|&n| n != node1 && n != node2)
                    .cloned()
                    .collect();

                if common.len() >= 2 {
                    for (j, fan1) in common.iter().enumerate() {
                        for fan2 in common.iter().skip(j + 1) {
                            let mut bi_fan =
                                vec![node1.clone(), node2.clone(), fan1.clone(), fan2.clone()];
                            bi_fan.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                            if let Ok(mut bi_fans_guard) = bi_fans.lock() {
                                if !bi_fans_guard.iter().any(|bf| bf == &bi_fan) {
                                    bi_fans_guard.push(bi_fan);
                                }
                            }
                        }
                    }
                }
            }
        }
    });

    bi_fans.into_inner().unwrap_or_default()
}

fn find_feed_forward_loops<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let ffls = Mutex::new(Vec::new());

    nodes.par_iter().for_each(|node_a| {
        if let Ok(out_neighbors_a) = graph.neighbors(node_a) {
            let out_neighbors_a: Vec<_> = out_neighbors_a;

            for (i, node_b) in out_neighbors_a.iter().enumerate() {
                for node_c in out_neighbors_a.iter().skip(i + 1) {
                    if graph.has_edge(node_b, node_c)
                        && !graph.has_edge(node_b, node_a)
                        && !graph.has_edge(node_c, node_a)
                        && !graph.has_edge(node_c, node_b)
                    {
                        let mut ffl = vec![node_a.clone(), node_b.clone(), node_c.clone()];
                        ffl.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                        if let Ok(mut ffls_guard) = ffls.lock() {
                            if !ffls_guard.iter().any(|f| f == &ffl) {
                                ffls_guard.push(ffl);
                            }
                        }
                    }
                }
            }
        }
    });

    ffls.into_inner().unwrap_or_default()
}

fn find_bidirectional_motifs<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Vec<Vec<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug + Send + Sync,
    E: EdgeWeight + Send + Sync,
    Ix: IndexType + Send + Sync,
{
    use scirs2_core::parallel_ops::*;
    use std::sync::Mutex;

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let bidirectionals = Mutex::new(Vec::new());

    nodes.par_iter().enumerate().for_each(|(i, node1)| {
        for node2 in nodes.iter().skip(i + 1) {
            if graph.has_edge(node1, node2) && graph.has_edge(node2, node1) {
                let mut motif = vec![node1.clone(), node2.clone()];
                motif.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));

                if let Ok(mut bidirectionals_guard) = bidirectionals.lock() {
                    if !bidirectionals_guard.iter().any(|m| m == &motif) {
                        bidirectionals_guard.push(motif);
                    }
                }
            }
        }
    });

    bidirectionals.into_inner().unwrap_or_default()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result as GraphResult;
    use crate::generators::create_graph;

    #[test]
    fn test_find_triangles() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "A", ())?;
        graph.add_edge("A", "D", ())?;

        let triangles = find_motifs(&graph, MotifType::Triangle);
        assert_eq!(triangles.len(), 1);
        let triangle = &triangles[0];
        assert_eq!(triangle.len(), 3);
        assert!(triangle.contains(&"A"));
        assert!(triangle.contains(&"B"));
        assert!(triangle.contains(&"C"));
        Ok(())
    }

    #[test]
    fn test_find_squares() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();
        graph.add_edge("A", "B", ())?;
        graph.add_edge("B", "C", ())?;
        graph.add_edge("C", "D", ())?;
        graph.add_edge("D", "A", ())?;

        let squares = find_motifs(&graph, MotifType::Square);
        assert_eq!(squares.len(), 1);
        assert_eq!(squares[0].len(), 4);
        Ok(())
    }

    #[test]
    fn test_find_star3() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();
        graph.add_edge("A", "B", ())?;
        graph.add_edge("A", "C", ())?;
        graph.add_edge("A", "D", ())?;

        let stars = find_motifs(&graph, MotifType::Star3);
        assert_eq!(stars.len(), 1);
        assert_eq!(stars[0].len(), 4);
        assert!(stars[0].contains(&"A"));
        Ok(())
    }

    #[test]
    fn test_find_clique4() -> GraphResult<()> {
        let mut graph = create_graph::<&str, ()>();
        let nodes = ["A", "B", "C", "D"];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                graph.add_edge(nodes[i], nodes[j], ())?;
            }
        }

        let cliques = find_motifs(&graph, MotifType::Clique4);
        assert_eq!(cliques.len(), 1);
        assert_eq!(cliques[0].len(), 4);
        Ok(())
    }

    #[test]
    fn test_vf2_exact_match() -> GraphResult<()> {
        let mut pattern = create_graph::<i32, ()>();
        pattern.add_edge(0, 1, ())?;
        pattern.add_edge(1, 2, ())?;

        let mut target = create_graph::<i32, ()>();
        target.add_edge(10, 11, ())?;
        target.add_edge(11, 12, ())?;

        let result = vf2_subgraph_isomorphism(&pattern, &target, 0);
        assert!(result.is_match);
        assert!(!result.mappings.is_empty());
        Ok(())
    }

    #[test]
    fn test_vf2_no_match() -> GraphResult<()> {
        let mut pattern = create_graph::<i32, ()>();
        pattern.add_edge(0, 1, ())?;
        pattern.add_edge(1, 2, ())?;
        pattern.add_edge(2, 0, ())?; // Triangle

        let mut target = create_graph::<i32, ()>();
        target.add_edge(10, 11, ())?;
        target.add_edge(11, 12, ())?;
        // Path, not triangle

        let result = vf2_subgraph_isomorphism(&pattern, &target, 0);
        assert!(!result.is_match);
        Ok(())
    }

    #[test]
    fn test_vf2_subgraph() -> GraphResult<()> {
        let mut pattern = create_graph::<i32, ()>();
        pattern.add_edge(0, 1, ())?;
        pattern.add_edge(1, 2, ())?;

        let mut target = create_graph::<i32, ()>();
        target.add_edge(10, 11, ())?;
        target.add_edge(11, 12, ())?;
        target.add_edge(12, 13, ())?;
        target.add_edge(10, 13, ())?;

        let result = vf2_subgraph_isomorphism(&pattern, &target, 0);
        assert!(result.is_match);
        // Multiple matches possible
        assert!(result.mappings.len() >= 1);
        Ok(())
    }

    #[test]
    fn test_wl_kernel_identical() -> GraphResult<()> {
        let mut g1 = create_graph::<i32, ()>();
        g1.add_edge(0, 1, ())?;
        g1.add_edge(1, 2, ())?;

        let mut g2 = create_graph::<i32, ()>();
        g2.add_edge(10, 11, ())?;
        g2.add_edge(11, 12, ())?;

        let result = wl_subtree_kernel(&[&g1, &g2], 3);
        assert_eq!(result.kernel_matrix.len(), 2);
        // Identical structure should have equal kernel values
        assert!((result.kernel_matrix[0][0] - result.kernel_matrix[1][1]).abs() < 1e-6);
        assert!((result.kernel_matrix[0][1] - result.kernel_matrix[0][0]).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_wl_kernel_different() -> GraphResult<()> {
        let mut g1 = create_graph::<i32, ()>();
        g1.add_edge(0, 1, ())?;
        g1.add_edge(1, 2, ())?;

        let mut g2 = create_graph::<i32, ()>();
        g2.add_edge(0, 1, ())?;
        g2.add_edge(1, 2, ())?;
        g2.add_edge(2, 0, ())?; // Triangle

        let result = wl_subtree_kernel(&[&g1, &g2], 2);
        // Different structures should have different self-kernel values
        assert!(result.kernel_matrix[0][0] > 0.0);
        assert!(result.kernel_matrix[1][1] > 0.0);
        Ok(())
    }

    #[test]
    fn test_graphlet_degree_distribution() -> GraphResult<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;
        graph.add_edge(2, 3, ())?;

        let result = graphlet_degree_distribution(&graph);
        assert_eq!(result.node_gdvs.len(), 4);

        // Node 2 has degree 3 (connected to 0, 1, 3)
        let gdv_2 = &result.node_gdvs[&2];
        assert_eq!(gdv_2.orbit_counts[0], 3); // degree

        // All nodes participate in at least one triangle except node 3
        let gdv_3 = &result.node_gdvs[&3];
        assert_eq!(gdv_3.orbit_counts[3], 0); // no triangles for node 3

        assert!(result.gdv_agreement >= 0.0 && result.gdv_agreement <= 1.0);
        Ok(())
    }

    #[test]
    fn test_count_3node_motifs() -> GraphResult<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;

        let (triangles, open) = count_3node_motifs(&graph);
        assert_eq!(triangles, 1);
        assert_eq!(open, 0); // All triads are closed in a triangle
        Ok(())
    }

    #[test]
    fn test_count_4node_motifs() -> GraphResult<()> {
        let mut graph = create_graph::<i32, ()>();
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 3, ())?;

        let counts = count_4node_motifs(&graph);
        assert!(counts.contains_key("path3"));
        assert_eq!(counts["path3"], 1);
        Ok(())
    }

    #[test]
    fn test_frequent_subgraph_mining() -> GraphResult<()> {
        let mut graph = create_graph::<i32, ()>();
        // Create a graph with multiple triangles
        graph.add_edge(0, 1, ())?;
        graph.add_edge(1, 2, ())?;
        graph.add_edge(2, 0, ())?;
        graph.add_edge(2, 3, ())?;
        graph.add_edge(3, 4, ())?;
        graph.add_edge(4, 2, ())?;

        let patterns = frequent_subgraph_mining(&graph, 1, 3);
        assert!(!patterns.is_empty());
        // Should find edges and triangles
        Ok(())
    }

    #[test]
    fn test_frequent_subgraph_empty() {
        let graph = create_graph::<i32, ()>();
        let patterns = frequent_subgraph_mining(&graph, 1, 3);
        assert!(patterns.is_empty());
    }
}
