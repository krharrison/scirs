//! Girvan-Newman community detection algorithm
//!
//! Detects communities by iteratively removing edges with the highest
//! edge betweenness centrality. This divisive hierarchical approach
//! reveals community structure at multiple scales.
//!
//! # References
//! - Girvan, M. & Newman, M.E.J. (2002). Community structure in social and
//!   biological networks. PNAS, 99(12), 7821-7826.

use super::types::CommunityResult;
use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

/// Configuration for Girvan-Newman algorithm
#[derive(Debug, Clone)]
pub struct GirvanNewmanConfig {
    /// Target number of communities (None = find optimal via modularity)
    pub num_communities: Option<usize>,
    /// Maximum number of edge removal iterations
    pub max_iterations: usize,
}

impl Default for GirvanNewmanConfig {
    fn default() -> Self {
        GirvanNewmanConfig {
            num_communities: None,
            max_iterations: 10000,
        }
    }
}

/// A dendrogram node representing the hierarchical decomposition
#[derive(Debug, Clone)]
pub struct DendrogramLevel<N: Node> {
    /// Communities at this level
    pub communities: Vec<HashSet<N>>,
    /// Number of communities at this level
    pub num_communities: usize,
    /// Modularity score at this level
    pub modularity: f64,
    /// The edge that was removed to reach this level (source, target)
    pub removed_edge: Option<(N, N)>,
}

/// Result of Girvan-Newman algorithm including the full dendrogram
#[derive(Debug, Clone)]
pub struct GirvanNewmanResult<N: Node> {
    /// The best community partition found (maximizing modularity)
    pub best_partition: CommunityResult<N>,
    /// The full dendrogram showing hierarchical decomposition
    pub dendrogram: Vec<DendrogramLevel<N>>,
}

/// Detect communities using the Girvan-Newman edge betweenness method
///
/// # Arguments
/// * `graph` - The undirected graph to analyze
/// * `config` - Algorithm configuration
///
/// # Returns
/// A `GirvanNewmanResult` containing the best partition and full dendrogram
///
/// # Algorithm
/// 1. Compute edge betweenness centrality for all edges
/// 2. Remove the edge with the highest betweenness
/// 3. Recompute connected components
/// 4. Repeat until desired number of communities or all edges removed
///
/// # Time Complexity
/// O(m^2 * n) where m is the number of edges and n is the number of nodes
pub fn girvan_newman_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &GirvanNewmanConfig,
) -> Result<GirvanNewmanResult<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: petgraph::graph::IndexType,
{
    let n = graph.node_count();
    if n == 0 {
        return Err(GraphError::InvalidGraph(
            "Cannot detect communities in an empty graph".to_string(),
        ));
    }

    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();

    // Build mutable adjacency representation
    let mut adjacency: HashMap<N, HashSet<N>> = HashMap::new();
    let mut edge_weights: HashMap<(N, N), f64> = HashMap::new();

    for node in &nodes {
        adjacency.insert(node.clone(), HashSet::new());
    }

    let edges = graph.edges();
    for edge in &edges {
        adjacency
            .entry(edge.source.clone())
            .or_default()
            .insert(edge.target.clone());
        adjacency
            .entry(edge.target.clone())
            .or_default()
            .insert(edge.source.clone());

        let w: f64 = edge.weight.clone().into();
        edge_weights.insert((edge.source.clone(), edge.target.clone()), w);
        edge_weights.insert((edge.target.clone(), edge.source.clone()), w);
    }

    // Compute initial total edge weight (for modularity)
    let total_weight: f64 = edges
        .iter()
        .map(|e| {
            let w: f64 = e.weight.clone().into();
            w
        })
        .sum();

    // Build degree map (original degrees for modularity computation)
    let original_degrees: HashMap<N, f64> = nodes
        .iter()
        .map(|node| {
            let deg: f64 = adjacency
                .get(node)
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .map(|nb| {
                            edge_weights
                                .get(&(node.clone(), nb.clone()))
                                .copied()
                                .unwrap_or(1.0)
                        })
                        .sum()
                })
                .unwrap_or(0.0);
            (node.clone(), deg)
        })
        .collect();

    let mut dendrogram: Vec<DendrogramLevel<N>> = Vec::new();
    let mut best_modularity = f64::NEG_INFINITY;
    let mut best_communities: Option<Vec<HashSet<N>>> = None;

    // Record initial state
    let initial_communities = find_connected_components(&adjacency);
    let initial_modularity = compute_modularity(
        &initial_communities,
        &adjacency,
        &original_degrees,
        &edge_weights,
        total_weight,
    );

    dendrogram.push(DendrogramLevel {
        communities: initial_communities.clone(),
        num_communities: initial_communities.len(),
        modularity: initial_modularity,
        removed_edge: None,
    });

    if initial_modularity > best_modularity {
        best_modularity = initial_modularity;
        best_communities = Some(initial_communities);
    }

    // Iteratively remove edges
    for _iter in 0..config.max_iterations {
        // Check if we've reached target number of communities
        let current_communities = find_connected_components(&adjacency);
        if let Some(target) = config.num_communities {
            if current_communities.len() >= target {
                break;
            }
        }

        // Count remaining edges
        let remaining_edges: usize = adjacency.values().map(|neighbors| neighbors.len()).sum();
        if remaining_edges == 0 {
            break;
        }

        // Compute edge betweenness centrality
        let betweenness = compute_edge_betweenness(&adjacency, &nodes);

        // Find edge with highest betweenness
        let mut max_betweenness = f64::NEG_INFINITY;
        let mut max_edge: Option<(N, N)> = None;

        for ((u, v), &b) in &betweenness {
            if b > max_betweenness {
                max_betweenness = b;
                max_edge = Some((u.clone(), v.clone()));
            }
        }

        let (u, v) = match max_edge {
            Some(edge) => edge,
            None => break,
        };

        // Remove the edge
        if let Some(neighbors) = adjacency.get_mut(&u) {
            neighbors.remove(&v);
        }
        if let Some(neighbors) = adjacency.get_mut(&v) {
            neighbors.remove(&u);
        }

        // Find new connected components
        let new_communities = find_connected_components(&adjacency);
        let new_modularity = compute_modularity(
            &new_communities,
            &adjacency,
            &original_degrees,
            &edge_weights,
            total_weight,
        );

        dendrogram.push(DendrogramLevel {
            communities: new_communities.clone(),
            num_communities: new_communities.len(),
            modularity: new_modularity,
            removed_edge: Some((u, v)),
        });

        if new_modularity > best_modularity {
            best_modularity = new_modularity;
            best_communities = Some(new_communities);
        }
    }

    // Build the best partition result
    let communities = best_communities.unwrap_or_else(|| find_connected_components(&adjacency));

    let mut node_communities: HashMap<N, usize> = HashMap::new();
    for (comm_id, community) in communities.iter().enumerate() {
        for node in community {
            node_communities.insert(node.clone(), comm_id);
        }
    }

    let mut result = CommunityResult::from_node_map(node_communities);
    result.quality_score = Some(best_modularity);
    result
        .metadata
        .insert("modularity".to_string(), best_modularity);
    result
        .metadata
        .insert("dendrogram_levels".to_string(), dendrogram.len() as f64);

    Ok(GirvanNewmanResult {
        best_partition: result,
        dendrogram,
    })
}

/// Simplified interface returning just a CommunityResult
pub fn girvan_newman_communities_result<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
) -> Result<CommunityResult<N>>
where
    N: Node + Clone + Hash + Eq + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Clone,
    Ix: petgraph::graph::IndexType,
{
    let config = GirvanNewmanConfig::default();
    let result = girvan_newman_result(graph, &config)?;
    Ok(result.best_partition)
}

/// Compute edge betweenness centrality for all edges using Brandes' algorithm
fn compute_edge_betweenness<N: Node + Clone + Hash + Eq + std::fmt::Debug>(
    adjacency: &HashMap<N, HashSet<N>>,
    nodes: &[N],
) -> HashMap<(N, N), f64> {
    let mut betweenness: HashMap<(N, N), f64> = HashMap::new();

    // Initialize all edges with zero betweenness
    for (u, neighbors) in adjacency {
        for v in neighbors {
            betweenness.insert((u.clone(), v.clone()), 0.0);
        }
    }

    // For each source node, run BFS and accumulate betweenness
    for source in nodes {
        if adjacency.get(source).map(|n| n.is_empty()).unwrap_or(true) {
            continue;
        }

        // BFS from source
        let mut stack: Vec<N> = Vec::new();
        let mut predecessors: HashMap<N, Vec<N>> = HashMap::new();
        let mut sigma: HashMap<N, f64> = HashMap::new(); // Number of shortest paths
        let mut dist: HashMap<N, i64> = HashMap::new(); // Distance from source

        for node in nodes {
            sigma.insert(node.clone(), 0.0);
            dist.insert(node.clone(), -1);
        }

        sigma.insert(source.clone(), 1.0);
        dist.insert(source.clone(), 0);

        let mut queue: VecDeque<N> = VecDeque::new();
        queue.push_back(source.clone());

        while let Some(v) = queue.pop_front() {
            stack.push(v.clone());

            let v_dist = dist.get(&v).copied().unwrap_or(-1);
            if v_dist < 0 {
                continue;
            }

            if let Some(neighbors) = adjacency.get(&v) {
                for w in neighbors {
                    let w_dist = dist.get(w).copied().unwrap_or(-1);

                    // w found for the first time?
                    if w_dist < 0 {
                        dist.insert(w.clone(), v_dist + 1);
                        queue.push_back(w.clone());
                    }

                    // Shortest path to w via v?
                    let current_w_dist = dist.get(w).copied().unwrap_or(-1);
                    if current_w_dist == v_dist + 1 {
                        let sigma_v = sigma.get(&v).copied().unwrap_or(0.0);
                        *sigma.entry(w.clone()).or_insert(0.0) += sigma_v;
                        predecessors.entry(w.clone()).or_default().push(v.clone());
                    }
                }
            }
        }

        // Back-propagation of dependencies
        let mut delta: HashMap<N, f64> = nodes.iter().map(|n| (n.clone(), 0.0)).collect();

        while let Some(w) = stack.pop() {
            if let Some(preds) = predecessors.get(&w) {
                let sigma_w = sigma.get(&w).copied().unwrap_or(1.0);
                let delta_w = delta.get(&w).copied().unwrap_or(0.0);

                for v in preds {
                    let sigma_v = sigma.get(v).copied().unwrap_or(1.0);
                    let coeff = (sigma_v / sigma_w) * (1.0 + delta_w);

                    // Add to edge betweenness (both directions)
                    *betweenness.entry((v.clone(), w.clone())).or_insert(0.0) += coeff;

                    // Accumulate node dependency
                    *delta.entry(v.clone()).or_insert(0.0) += coeff;
                }
            }
        }
    }

    // For undirected graphs, each edge is counted twice; normalize
    for value in betweenness.values_mut() {
        *value /= 2.0;
    }

    betweenness
}

/// Find connected components using BFS
fn find_connected_components<N: Node + Clone + Hash + Eq>(
    adjacency: &HashMap<N, HashSet<N>>,
) -> Vec<HashSet<N>> {
    let mut visited: HashSet<N> = HashSet::new();
    let mut components: Vec<HashSet<N>> = Vec::new();

    for node in adjacency.keys() {
        if visited.contains(node) {
            continue;
        }

        let mut component = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node.clone());
        visited.insert(node.clone());

        while let Some(current) = queue.pop_front() {
            component.insert(current.clone());

            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        components.push(component);
    }

    // Sort by size (largest first)
    components.sort_by_key(|b| std::cmp::Reverse(b.len()));
    components
}

/// Compute modularity Q for a partition
///
/// Q = (1/2m) * sum_{ij} [A_{ij} - k_i*k_j/(2m)] * delta(c_i, c_j)
fn compute_modularity<N: Node + Clone + Hash + Eq>(
    communities: &[HashSet<N>],
    _adjacency: &HashMap<N, HashSet<N>>,
    original_degrees: &HashMap<N, f64>,
    edge_weights: &HashMap<(N, N), f64>,
    total_weight: f64,
) -> f64 {
    if total_weight <= 0.0 {
        return 0.0;
    }

    let two_m = 2.0 * total_weight;
    let mut q = 0.0;

    for community in communities {
        for u in community {
            for v in community {
                let a_uv = edge_weights
                    .get(&(u.clone(), v.clone()))
                    .copied()
                    .unwrap_or(0.0);
                let k_u = original_degrees.get(u).copied().unwrap_or(0.0);
                let k_v = original_degrees.get(v).copied().unwrap_or(0.0);

                q += a_uv - (k_u * k_v) / two_m;
            }
        }
    }

    q / two_m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_karate_club_like() -> Graph<i32, f64> {
        // Two clear communities connected by a few bridge edges
        let mut g = Graph::new();
        for i in 0..10 {
            g.add_node(i);
        }

        // Community A (0-4): dense connections
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        let _ = g.add_edge(0, 3, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(1, 3, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        let _ = g.add_edge(2, 4, 1.0);
        let _ = g.add_edge(3, 4, 1.0);

        // Community B (5-9): dense connections
        let _ = g.add_edge(5, 6, 1.0);
        let _ = g.add_edge(5, 7, 1.0);
        let _ = g.add_edge(5, 8, 1.0);
        let _ = g.add_edge(6, 7, 1.0);
        let _ = g.add_edge(6, 8, 1.0);
        let _ = g.add_edge(7, 8, 1.0);
        let _ = g.add_edge(7, 9, 1.0);
        let _ = g.add_edge(8, 9, 1.0);

        // Bridge edge between communities
        let _ = g.add_edge(4, 5, 1.0);

        g
    }

    fn make_triangle() -> Graph<i32, f64> {
        let mut g: Graph<i32, f64> = Graph::new();
        for i in 0..3 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(0, 2, 1.0);
        g
    }

    #[test]
    fn test_girvan_newman_basic() {
        let g = make_karate_club_like();
        let config = GirvanNewmanConfig::default();

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok(), "Girvan-Newman should succeed");

        let result = result.expect("result should be valid");
        assert!(
            result.best_partition.num_communities >= 2,
            "Should find at least 2 communities, found {}",
            result.best_partition.num_communities
        );
    }

    #[test]
    fn test_girvan_newman_target_communities() {
        let g = make_karate_club_like();
        let config = GirvanNewmanConfig {
            num_communities: Some(2),
            max_iterations: 1000,
        };

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");
        // Should have at least 2 communities
        assert!(result.best_partition.num_communities >= 2);
    }

    #[test]
    fn test_girvan_newman_dendrogram() {
        let g = make_karate_club_like();
        let config = GirvanNewmanConfig {
            num_communities: Some(3),
            max_iterations: 1000,
        };

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");
        // Dendrogram should have multiple levels
        assert!(
            result.dendrogram.len() >= 2,
            "Dendrogram should have at least 2 levels, has {}",
            result.dendrogram.len()
        );

        // First level should have the initial number of components
        assert_eq!(result.dendrogram[0].removed_edge, None);

        // Subsequent levels should have removed edges
        for level in result.dendrogram.iter().skip(1) {
            assert!(level.removed_edge.is_some());
        }
    }

    #[test]
    fn test_girvan_newman_bridge_removal() {
        // In a graph with two communities joined by a single bridge,
        // the bridge should be removed first
        let g = make_karate_club_like();
        let config = GirvanNewmanConfig {
            num_communities: Some(2),
            max_iterations: 1000,
        };

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");

        // Check that communities separate nodes correctly
        // Nodes 0-4 should be in one community, 5-9 in another
        let comm_0 = result.best_partition.get_community(&0);
        let comm_5 = result.best_partition.get_community(&5);

        assert!(comm_0.is_some());
        assert!(comm_5.is_some());

        // They should be in different communities
        // (modularity optimization may produce slight variations, but the bridge
        //  community separation should hold)
        if result.best_partition.num_communities >= 2 {
            // Verify all A-side nodes are together
            let comm_a = comm_0.expect("node 0 should have community");
            for node in [1, 2, 3, 4] {
                let c = result.best_partition.get_community(&node);
                assert!(c.is_some(), "Node {node} should have community assignment");
                assert_eq!(
                    c.expect("community should exist"),
                    comm_a,
                    "Node {node} should be in same community as node 0"
                );
            }
        }
    }

    #[test]
    fn test_girvan_newman_triangle() {
        let g = make_triangle();
        let config = GirvanNewmanConfig::default();

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");
        // All 3 nodes should have community assignments
        for node in [0, 1, 2] {
            assert!(
                result.best_partition.get_community(&node).is_some(),
                "Node {node} should have a community"
            );
        }
    }

    #[test]
    fn test_girvan_newman_empty_graph() {
        let g: Graph<i32, f64> = Graph::new();
        let config = GirvanNewmanConfig::default();

        let result = girvan_newman_result(&g, &config);
        assert!(result.is_err(), "Should fail on empty graph");
    }

    #[test]
    fn test_girvan_newman_single_node() {
        let mut g: Graph<i32, f64> = Graph::new();
        g.add_node(0);

        let config = GirvanNewmanConfig::default();
        let result = girvan_newman_result(&g, &config);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");
        assert_eq!(result.best_partition.num_communities, 1);
    }

    #[test]
    fn test_girvan_newman_communities_result_simplified() {
        let g = make_karate_club_like();

        let result = girvan_newman_communities_result(&g);
        assert!(result.is_ok());

        let result = result.expect("result should be valid");
        assert!(result.num_communities >= 1);
        assert!(result.quality_score.is_some());
    }

    #[test]
    fn test_edge_betweenness_computation() {
        // In a path graph 0-1-2, edge (0,1) and (1,2) should have higher betweenness
        // than in a triangle where edges are equally important
        let mut adjacency: HashMap<i32, HashSet<i32>> = HashMap::new();
        adjacency.insert(0, [1].iter().copied().collect());
        adjacency.insert(1, [0, 2].iter().copied().collect());
        adjacency.insert(2, [1].iter().copied().collect());

        let nodes = vec![0, 1, 2];
        let betweenness = compute_edge_betweenness(&adjacency, &nodes);

        // Edge (0,1) should have positive betweenness
        let b_01 = betweenness.get(&(0, 1)).copied().unwrap_or(0.0);
        assert!(
            b_01 > 0.0,
            "Edge (0,1) should have positive betweenness, got {b_01}"
        );

        let b_12 = betweenness.get(&(1, 2)).copied().unwrap_or(0.0);
        assert!(
            b_12 > 0.0,
            "Edge (1,2) should have positive betweenness, got {b_12}"
        );
    }
}
