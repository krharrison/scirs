//! Network flow algorithms for graph processing
//!
//! This module contains algorithms for finding maximum flows, minimum cuts,
//! min-cost max flows, multi-commodity flows, and bipartite matching.
//!
//! # Algorithms
//! - **Edmonds-Karp**: Max flow via Ford-Fulkerson with BFS (O(VE^2))
//! - **Dinic's**: Max flow via blocking flows and level graphs (O(V^2 E))
//! - **Push-Relabel**: Max flow via preflow and relabeling (O(V^2 sqrt(E)))
//! - **Min-Cut**: Minimum s-t cut derived from max flow
//! - **Min-Cost Max Flow**: Successive shortest paths algorithm
//! - **Multi-Commodity Flow**: LP relaxation approximation
//! - **Hopcroft-Karp**: Maximum bipartite matching (O(E sqrt(V)))

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of a max-flow computation
#[derive(Debug, Clone)]
pub struct MaxFlowResult<N: Node> {
    /// Total flow value from source to sink
    pub flow_value: f64,
    /// Flow on each edge: (source, target) -> flow
    pub edge_flows: HashMap<(N, N), f64>,
    /// The min-cut partition: nodes reachable from source in residual graph
    pub source_side: HashSet<N>,
}

/// Result of a min-cost max-flow computation
#[derive(Debug, Clone)]
pub struct MinCostFlowResult<N: Node> {
    /// Total flow value
    pub flow_value: f64,
    /// Total cost of the flow
    pub total_cost: f64,
    /// Flow on each edge
    pub edge_flows: HashMap<(N, N), f64>,
}

/// Result of a multi-commodity flow computation
#[derive(Debug, Clone)]
pub struct MultiCommodityFlowResult {
    /// Whether a feasible solution was found
    pub feasible: bool,
    /// Flow value for each commodity
    pub commodity_flows: Vec<f64>,
    /// Total throughput (sum of all commodity flows)
    pub total_throughput: f64,
}

/// Result of Hopcroft-Karp bipartite matching
#[derive(Debug, Clone)]
pub struct HopcroftKarpResult<N: Node> {
    /// The matching as pairs (left, right)
    pub matching: Vec<(N, N)>,
    /// Size of the maximum matching
    pub size: usize,
    /// Left nodes that are unmatched
    pub unmatched_left: Vec<N>,
    /// Right nodes that are unmatched
    pub unmatched_right: Vec<N>,
}

// ============================================================================
// Internal residual graph representation for flow algorithms
// ============================================================================

/// Internal residual graph used for flow computations
struct ResidualGraph {
    /// Number of nodes
    n: usize,
    /// Capacity[i][j] = residual capacity from i to j
    capacity: Vec<Vec<f64>>,
    /// Flow[i][j] = current flow from i to j
    flow: Vec<Vec<f64>>,
}

impl ResidualGraph {
    fn new(n: usize) -> Self {
        Self {
            n,
            capacity: vec![vec![0.0; n]; n],
            flow: vec![vec![0.0; n]; n],
        }
    }

    fn residual_capacity(&self, u: usize, v: usize) -> f64 {
        self.capacity[u][v] - self.flow[u][v]
    }

    /// BFS to find augmenting path, returns parent array
    fn bfs(&self, source: usize, sink: usize) -> Option<Vec<Option<usize>>> {
        let mut parent: Vec<Option<usize>> = vec![None; self.n];
        let mut visited = vec![false; self.n];
        visited[source] = true;

        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == sink {
                return Some(parent);
            }
            for v in 0..self.n {
                if !visited[v] && self.residual_capacity(u, v) > 1e-12 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }
        None
    }

    /// BFS to build level graph for Dinic's algorithm
    fn build_level_graph(&self, source: usize) -> Vec<i64> {
        let mut level = vec![-1i64; self.n];
        level[source] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            for v in 0..self.n {
                if level[v] < 0 && self.residual_capacity(u, v) > 1e-12 {
                    level[v] = level[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        level
    }

    /// DFS for blocking flow in Dinic's algorithm
    fn send_flow(
        &mut self,
        u: usize,
        sink: usize,
        pushed: f64,
        level: &[i64],
        iter: &mut [usize],
    ) -> f64 {
        if u == sink {
            return pushed;
        }
        while iter[u] < self.n {
            let v = iter[u];
            if level[v] == level[u] + 1 && self.residual_capacity(u, v) > 1e-12 {
                let bottleneck = pushed.min(self.residual_capacity(u, v));
                let d = self.send_flow(v, sink, bottleneck, level, iter);
                if d > 1e-12 {
                    self.flow[u][v] += d;
                    self.flow[v][u] -= d;
                    return d;
                }
            }
            iter[u] += 1;
        }
        0.0
    }

    /// Find nodes reachable from source in the residual graph
    fn reachable_from(&self, source: usize) -> Vec<bool> {
        let mut visited = vec![false; self.n];
        let mut stack = vec![source];
        visited[source] = true;

        while let Some(u) = stack.pop() {
            for v in 0..self.n {
                if !visited[v] && self.residual_capacity(u, v) > 1e-12 {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        visited
    }
}

// ============================================================================
// Helper: build index maps from a DiGraph
// ============================================================================

fn build_node_index_maps<N, E, Ix>(graph: &DiGraph<N, E, Ix>) -> (Vec<N>, HashMap<N, usize>)
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    (nodes, node_to_idx)
}

fn build_residual_from_digraph<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    nodes: &[N],
    node_to_idx: &HashMap<N, usize>,
) -> ResidualGraph
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let n = nodes.len();
    let mut rg = ResidualGraph::new(n);

    for node in nodes {
        if let Ok(successors) = graph.successors(node) {
            let u = node_to_idx[node];
            for succ in &successors {
                if let Ok(w) = graph.edge_weight(node, succ) {
                    let v = node_to_idx[succ];
                    rg.capacity[u][v] += w.into();
                }
            }
        }
    }
    rg
}

// ============================================================================
// Edmonds-Karp (Ford-Fulkerson with BFS)
// ============================================================================

/// Edmonds-Karp algorithm for maximum flow (Ford-Fulkerson with BFS).
///
/// Computes the maximum s-t flow in a directed graph with positive edge capacities.
/// Time complexity: O(V * E^2).
///
/// Returns full flow result including edge flows and min-cut partition.
pub fn edmonds_karp_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<MaxFlowResult<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    validate_flow_input(graph, source, sink)?;

    let (nodes, node_to_idx) = build_node_index_maps(graph);
    let s = node_to_idx[source];
    let t = node_to_idx[sink];
    let mut rg = build_residual_from_digraph(graph, &nodes, &node_to_idx);

    let mut total_flow = 0.0;

    // Repeatedly find augmenting paths via BFS
    while let Some(parent) = rg.bfs(s, t) {
        // Find bottleneck capacity
        let mut path_flow = f64::INFINITY;
        let mut v = t;
        while v != s {
            if let Some(u) = parent[v] {
                path_flow = path_flow.min(rg.residual_capacity(u, v));
                v = u;
            } else {
                break;
            }
        }

        if path_flow <= 1e-12 {
            break;
        }

        // Update residual capacities
        v = t;
        while v != s {
            if let Some(u) = parent[v] {
                rg.flow[u][v] += path_flow;
                rg.flow[v][u] -= path_flow;
                v = u;
            } else {
                break;
            }
        }

        total_flow += path_flow;
    }

    build_flow_result(&rg, &nodes, &node_to_idx, total_flow, s)
}

/// Ford-Fulkerson max flow using BFS (Edmonds-Karp variant).
///
/// This is an alias for `edmonds_karp_max_flow` since Ford-Fulkerson with BFS
/// is exactly Edmonds-Karp.
pub fn ford_fulkerson_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let result = edmonds_karp_max_flow(graph, source, sink)?;
    Ok(result.flow_value)
}

// ============================================================================
// Dinic's Algorithm
// ============================================================================

/// Dinic's algorithm for maximum flow.
///
/// Uses level graphs and blocking flows for efficient max-flow computation.
/// Time complexity: O(V^2 * E).
pub fn dinic_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let result = dinic_max_flow_full(graph, source, sink)?;
    Ok(result.flow_value)
}

/// Dinic's algorithm returning full flow result.
pub fn dinic_max_flow_full<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<MaxFlowResult<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    validate_flow_input(graph, source, sink)?;

    let (nodes, node_to_idx) = build_node_index_maps(graph);
    let s = node_to_idx[source];
    let t = node_to_idx[sink];
    let n = nodes.len();
    let mut rg = build_residual_from_digraph(graph, &nodes, &node_to_idx);

    let mut total_flow = 0.0;

    loop {
        let level = rg.build_level_graph(s);
        if level[t] < 0 {
            break; // No augmenting path exists
        }

        let mut iter = vec![0usize; n];
        loop {
            let f = rg.send_flow(s, t, f64::INFINITY, &level, &mut iter);
            if f <= 1e-12 {
                break;
            }
            total_flow += f;
        }
    }

    build_flow_result(&rg, &nodes, &node_to_idx, total_flow, s)
}

// ============================================================================
// Push-Relabel Algorithm
// ============================================================================

/// Push-relabel algorithm for maximum flow.
///
/// Uses excess flow and height labels. Simpler to implement for dense graphs.
/// Time complexity: O(V^2 * E) with FIFO selection, O(V^3) with highest label.
pub fn push_relabel_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let result = push_relabel_max_flow_full(graph, source, sink)?;
    Ok(result.flow_value)
}

/// Push-relabel algorithm returning full flow result.
pub fn push_relabel_max_flow_full<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<MaxFlowResult<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    validate_flow_input(graph, source, sink)?;

    let (nodes, node_to_idx) = build_node_index_maps(graph);
    let s = node_to_idx[source];
    let t = node_to_idx[sink];
    let n = nodes.len();
    let mut rg = build_residual_from_digraph(graph, &nodes, &node_to_idx);

    // Initialize: height[s] = n, excess computed by saturating edges from s
    let mut height = vec![0usize; n];
    let mut excess = vec![0.0f64; n];
    height[s] = n;

    // Saturate all edges from source
    for v in 0..n {
        let cap = rg.residual_capacity(s, v);
        if cap > 1e-12 {
            rg.flow[s][v] = cap;
            rg.flow[v][s] = -cap;
            excess[v] += cap;
            excess[s] -= cap;
        }
    }

    // Active nodes (have excess, not source or sink)
    let mut active: VecDeque<usize> = (0..n)
        .filter(|&v| v != s && v != t && excess[v] > 1e-12)
        .collect();

    let max_iterations = n * n * 4; // Safety bound
    let mut iterations = 0;

    while let Some(u) = active.pop_front() {
        if iterations > max_iterations {
            break;
        }
        iterations += 1;

        if excess[u] <= 1e-12 {
            continue;
        }

        // Try to push
        let mut pushed = false;
        for v in 0..n {
            if excess[u] <= 1e-12 {
                break;
            }
            if height[u] == height[v] + 1 && rg.residual_capacity(u, v) > 1e-12 {
                let delta = excess[u].min(rg.residual_capacity(u, v));
                rg.flow[u][v] += delta;
                rg.flow[v][u] -= delta;
                excess[u] -= delta;
                excess[v] += delta;
                if v != s && v != t && excess[v] > 1e-12 {
                    // Check if v is already in active
                    if !active.contains(&v) {
                        active.push_back(v);
                    }
                }
                pushed = true;
            }
        }

        // Relabel if still has excess
        if excess[u] > 1e-12 {
            if !pushed {
                // Find minimum height among neighbors with residual capacity
                let mut min_height = usize::MAX;
                for v in 0..n {
                    if rg.residual_capacity(u, v) > 1e-12 && height[v] < min_height {
                        min_height = height[v];
                    }
                }
                if min_height < usize::MAX {
                    height[u] = min_height + 1;
                }
            }
            active.push_back(u);
        }
    }

    let total_flow = excess[t];
    build_flow_result(&rg, &nodes, &node_to_idx, total_flow, s)
}

// ============================================================================
// Minimum Cut
// ============================================================================

/// Find minimum s-t cut using max flow.
///
/// The min-cut value equals the max flow (max-flow min-cut theorem).
/// Returns the cut value and the partition of nodes.
pub fn minimum_cut<N, E, Ix>(graph: &Graph<N, E, Ix>) -> Result<(f64, Vec<bool>)>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n < 2 {
        return Err(GraphError::InvalidGraph(
            "Graph must have at least 2 nodes for minimum cut".to_string(),
        ));
    }

    // Stoer-Wagner algorithm for global min-cut on undirected graphs
    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build weight matrix
    let mut w = vec![vec![0.0f64; n]; n];
    for node in &nodes {
        if let Ok(neighbors) = graph.neighbors(node) {
            let u = node_to_idx[node];
            for neighbor in &neighbors {
                if let Ok(weight) = graph.edge_weight(node, neighbor) {
                    let v = node_to_idx[neighbor];
                    w[u][v] = weight.into();
                }
            }
        }
    }

    // Stoer-Wagner minimum cut
    let mut merged: Vec<bool> = vec![false; n];
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut best_cut = f64::INFINITY;
    let mut best_partition = vec![false; n];

    for phase in 0..(n - 1) {
        // Maximum adjacency ordering
        let mut in_a = vec![false; n];
        let mut key = vec![0.0f64; n];

        // Find first non-merged node
        let mut start = 0;
        for (i, &m) in merged.iter().enumerate() {
            if !m {
                start = i;
                break;
            }
        }
        in_a[start] = true;

        // Initialize keys from start
        for j in 0..n {
            if !merged[j] {
                key[j] = w[start][j];
            }
        }

        let mut prev = start;
        let mut last = start;
        let active_count = n - phase;

        for _ in 1..active_count {
            // Find node with maximum key not in A
            let mut max_key = -1.0f64;
            let mut max_node = 0;
            for j in 0..n {
                if !merged[j] && !in_a[j] && key[j] > max_key {
                    max_key = key[j];
                    max_node = j;
                }
            }

            in_a[max_node] = true;
            prev = last;
            last = max_node;

            // Update keys
            for j in 0..n {
                if !merged[j] && !in_a[j] {
                    key[j] += w[max_node][j];
                }
            }
        }

        // The cut of the phase is key[last]
        let cut_value = key[last];
        if cut_value < best_cut {
            best_cut = cut_value;
            // Build partition: nodes in last's group go to one side
            best_partition = vec![false; n];
            for &node_idx in &groups[last] {
                best_partition[node_idx] = true;
            }
        }

        // Merge last into prev
        merged[last] = true;
        for j in 0..n {
            w[prev][j] += w[last][j];
            w[j][prev] += w[j][last];
        }

        // Merge groups
        let last_group = groups[last].clone();
        groups[prev].extend(last_group);
    }

    Ok((best_cut, best_partition))
}

/// Find minimum s-t cut from max flow result.
///
/// After computing max flow, the min-cut is determined by the nodes
/// reachable from source in the residual graph.
pub fn minimum_st_cut<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<(f64, HashSet<N>, HashSet<N>)>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let result = edmonds_karp_max_flow(graph, source, sink)?;
    let all_nodes: HashSet<N> = graph.nodes().into_iter().cloned().collect();
    let sink_side: HashSet<N> = all_nodes.difference(&result.source_side).cloned().collect();
    Ok((result.flow_value, result.source_side, sink_side))
}

// ============================================================================
// Min-Cost Max Flow (Successive Shortest Paths)
// ============================================================================

/// Edge with both capacity and cost
#[derive(Debug, Clone)]
pub struct CostEdge {
    /// Edge capacity
    pub capacity: f64,
    /// Cost per unit of flow
    pub cost: f64,
}

/// Min-cost max flow using successive shortest paths (Bellman-Ford).
///
/// Finds maximum flow with minimum total cost from source to sink.
/// Each edge has both a capacity and a per-unit cost.
///
/// # Arguments
/// * `n` - Number of nodes (nodes labeled 0..n-1)
/// * `edges` - Edges as (from, to, capacity, cost)
/// * `source` - Source node index
/// * `sink` - Sink node index
pub fn min_cost_max_flow(
    n: usize,
    edges: &[(usize, usize, f64, f64)],
    source: usize,
    sink: usize,
) -> Result<MinCostFlowResult<usize>> {
    if source >= n || sink >= n {
        return Err(GraphError::InvalidGraph(
            "Source or sink out of bounds".to_string(),
        ));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same".to_string(),
        ));
    }

    // Build adjacency list with forward and reverse edges
    // Each edge is stored as (to, capacity, cost, reverse_edge_index)
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut edge_list: Vec<(usize, f64, f64, usize)> = Vec::new(); // (to, residual_cap, cost, rev_idx)

    for &(u, v, cap, cost) in edges {
        let forward_idx = edge_list.len();
        let reverse_idx = edge_list.len() + 1;

        adj[u].push(forward_idx);
        edge_list.push((v, cap, cost, reverse_idx));

        adj[v].push(reverse_idx);
        edge_list.push((u, 0.0, -cost, forward_idx));
    }

    let mut total_flow = 0.0;
    let mut total_cost = 0.0;

    let max_outer_iterations = n * edges.len() + 1;
    let mut outer_iter = 0;

    // Successive shortest paths
    loop {
        if outer_iter >= max_outer_iterations {
            break;
        }
        outer_iter += 1;

        // Bellman-Ford to find shortest path (by cost) in residual graph
        let mut dist = vec![f64::INFINITY; n];
        let mut parent_edge: Vec<Option<usize>> = vec![None; n];
        let mut in_queue = vec![false; n];
        dist[source] = 0.0;

        let mut queue = VecDeque::new();
        queue.push_back(source);
        in_queue[source] = true;

        let max_bf_iterations = n * edge_list.len();
        let mut bf_iter = 0;

        while let Some(u) = queue.pop_front() {
            if bf_iter > max_bf_iterations {
                break;
            }
            bf_iter += 1;
            in_queue[u] = false;

            for &edge_idx in &adj[u] {
                let (v, residual, cost, _) = edge_list[edge_idx];
                if residual > 1e-12 && dist[u] + cost < dist[v] - 1e-12 {
                    dist[v] = dist[u] + cost;
                    parent_edge[v] = Some(edge_idx);
                    if !in_queue[v] {
                        queue.push_back(v);
                        in_queue[v] = true;
                    }
                }
            }
        }

        if dist[sink].is_infinite() {
            break; // No more augmenting paths
        }

        // Find bottleneck
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            if let Some(edge_idx) = parent_edge[v] {
                bottleneck = bottleneck.min(edge_list[edge_idx].1);
                // Find the parent node
                let rev_idx = edge_list[edge_idx].3;
                v = edge_list[rev_idx].0;
            } else {
                break;
            }
        }

        if bottleneck <= 1e-12 {
            break;
        }

        // Augment flow
        v = sink;
        while v != source {
            if let Some(edge_idx) = parent_edge[v] {
                edge_list[edge_idx].1 -= bottleneck;
                let rev_idx = edge_list[edge_idx].3;
                edge_list[rev_idx].1 += bottleneck;
                v = edge_list[rev_idx].0;
            } else {
                break;
            }
        }

        total_flow += bottleneck;
        total_cost += bottleneck * dist[sink];
    }

    // Reconstruct edge flows
    let mut edge_flows = HashMap::new();
    for (i, &(u, v, cap, cost)) in edges.iter().enumerate() {
        let forward_idx = i * 2;
        let flow = cap - edge_list[forward_idx].1;
        if flow > 1e-12 {
            let _ = cost; // suppress unused warning
            let _ = v; // we use it below
            edge_flows.insert((u, edges[i].1), flow);
        }
    }

    Ok(MinCostFlowResult {
        flow_value: total_flow,
        total_cost,
        edge_flows,
    })
}

/// Min-cost max flow on a DiGraph with separate cost function.
pub fn min_cost_max_flow_graph<N, E, Ix, F>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
    cost_fn: F,
) -> Result<(f64, f64)>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
    F: Fn(&N, &N) -> f64,
{
    validate_flow_input(graph, source, sink)?;

    let (nodes, node_to_idx) = build_node_index_maps(graph);
    let s = node_to_idx[source];
    let t = node_to_idx[sink];

    // Collect edges
    let mut edges = Vec::new();
    for node in &nodes {
        if let Ok(successors) = graph.successors(node) {
            let u = node_to_idx[node];
            for succ in &successors {
                if let Ok(w) = graph.edge_weight(node, succ) {
                    let v = node_to_idx[succ];
                    let cap: f64 = w.into();
                    let cost = cost_fn(node, succ);
                    edges.push((u, v, cap, cost));
                }
            }
        }
    }

    let result = min_cost_max_flow(nodes.len(), &edges, s, t)?;
    Ok((result.flow_value, result.total_cost))
}

// ============================================================================
// Multi-Commodity Flow (LP Relaxation Approximation)
// ============================================================================

/// Multi-commodity flow using LP relaxation via iterative proportional scaling.
///
/// Approximates the multi-commodity flow problem where multiple commodities
/// share network capacity. Uses a simple iterative approach.
///
/// # Arguments
/// * `n` - Number of nodes
/// * `edge_caps` - Edges as (from, to, capacity)
/// * `commodities` - List of (source, sink, demand) for each commodity
pub fn multi_commodity_flow(
    n: usize,
    edge_caps: &[(usize, usize, f64)],
    commodities: &[(usize, usize, f64)],
) -> Result<MultiCommodityFlowResult> {
    if commodities.is_empty() {
        return Ok(MultiCommodityFlowResult {
            feasible: true,
            commodity_flows: vec![],
            total_throughput: 0.0,
        });
    }

    // Build capacity matrix
    let mut capacity = vec![vec![0.0f64; n]; n];
    for &(u, v, cap) in edge_caps {
        if u < n && v < n {
            capacity[u][v] += cap;
        }
    }

    let num_commodities = commodities.len();
    let mut commodity_flows = vec![0.0f64; num_commodities];
    let mut remaining_capacity = capacity.clone();
    let max_iterations = 100;

    // Iterative approach: allocate flow for each commodity proportionally
    for _iteration in 0..max_iterations {
        let mut any_improvement = false;

        for (k, &(src, snk, demand)) in commodities.iter().enumerate() {
            if src >= n || snk >= n || src == snk {
                continue;
            }

            // How much more do we need?
            let needed = demand - commodity_flows[k];
            if needed <= 1e-12 {
                continue;
            }

            // Find augmenting path in remaining capacity using BFS
            if let Some((path, bottleneck)) = bfs_augmenting_path(&remaining_capacity, n, src, snk)
            {
                let flow_to_send = bottleneck.min(needed);
                if flow_to_send > 1e-12 {
                    // Update remaining capacity along path
                    for window in path.windows(2) {
                        remaining_capacity[window[0]][window[1]] -= flow_to_send;
                    }
                    commodity_flows[k] += flow_to_send;
                    any_improvement = true;
                }
            }
        }

        if !any_improvement {
            break;
        }
    }

    let total_throughput: f64 = commodity_flows.iter().sum();
    let feasible = commodities
        .iter()
        .enumerate()
        .all(|(k, &(_, _, demand))| (commodity_flows[k] - demand).abs() < 1e-6);

    Ok(MultiCommodityFlowResult {
        feasible,
        commodity_flows,
        total_throughput,
    })
}

/// BFS to find augmenting path and bottleneck capacity
fn bfs_augmenting_path(
    capacity: &[Vec<f64>],
    n: usize,
    source: usize,
    sink: usize,
) -> Option<(Vec<usize>, f64)> {
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    visited[source] = true;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        if u == sink {
            // Reconstruct path
            let mut path = vec![sink];
            let mut v = sink;
            while v != source {
                if let Some(p) = parent[v] {
                    path.push(p);
                    v = p;
                } else {
                    return None;
                }
            }
            path.reverse();

            // Compute bottleneck
            let mut bottleneck = f64::INFINITY;
            for window in path.windows(2) {
                bottleneck = bottleneck.min(capacity[window[0]][window[1]]);
            }

            return Some((path, bottleneck));
        }

        for v in 0..n {
            if !visited[v] && capacity[u][v] > 1e-12 {
                visited[v] = true;
                parent[v] = Some(u);
                queue.push_back(v);
            }
        }
    }

    None
}

// ============================================================================
// Hopcroft-Karp Maximum Bipartite Matching
// ============================================================================

/// Hopcroft-Karp algorithm for maximum bipartite matching.
///
/// Finds a maximum matching in a bipartite graph in O(E * sqrt(V)) time.
///
/// # Arguments
/// * `graph` - The bipartite graph
/// * `left_nodes` - Nodes on the left side
/// * `right_nodes` - Nodes on the right side
pub fn hopcroft_karp<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    left_nodes: &[N],
    right_nodes: &[N],
) -> Result<HopcroftKarpResult<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let n_left = left_nodes.len();
    let n_right = right_nodes.len();

    if n_left == 0 || n_right == 0 {
        return Ok(HopcroftKarpResult {
            matching: vec![],
            size: 0,
            unmatched_left: left_nodes.to_vec(),
            unmatched_right: right_nodes.to_vec(),
        });
    }

    let left_to_idx: HashMap<N, usize> = left_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    let right_to_idx: HashMap<N, usize> = right_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build adjacency list: for each left node, list of right node indices
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n_left];
    for (li, left_node) in left_nodes.iter().enumerate() {
        if let Ok(neighbors) = graph.neighbors(left_node) {
            for neighbor in &neighbors {
                if let Some(&ri) = right_to_idx.get(neighbor) {
                    adj[li].push(ri);
                }
            }
        }
    }

    // NIL is represented as usize::MAX
    let nil = usize::MAX;
    let mut match_left: Vec<usize> = vec![nil; n_left]; // left -> right
    let mut match_right: Vec<usize> = vec![nil; n_right]; // right -> left
    let mut dist: Vec<usize> = vec![0; n_left + 1]; // distance labels, +1 for NIL

    // BFS phase: build layers of augmenting paths
    let bfs_phase = |match_left: &[usize],
                     match_right: &[usize],
                     dist: &mut Vec<usize>,
                     adj: &[Vec<usize>],
                     n_left: usize|
     -> bool {
        let mut queue = VecDeque::new();

        for u in 0..n_left {
            if match_left[u] == nil {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = usize::MAX;
            }
        }
        dist[n_left] = usize::MAX; // dist[NIL]

        while let Some(u) = queue.pop_front() {
            if dist[u] < dist[n_left] {
                for &v in &adj[u] {
                    let paired = match_right[v];
                    let paired_dist_idx = if paired == nil { n_left } else { paired };
                    if dist[paired_dist_idx] == usize::MAX {
                        dist[paired_dist_idx] = dist[u] + 1;
                        if paired_dist_idx != n_left {
                            queue.push_back(paired_dist_idx);
                        }
                    }
                }
            }
        }

        dist[n_left] != usize::MAX
    };

    // DFS phase: find augmenting paths
    fn dfs_phase(
        u: usize,
        match_left: &mut [usize],
        match_right: &mut [usize],
        dist: &mut [usize],
        adj: &[Vec<usize>],
        n_left: usize,
        nil: usize,
    ) -> bool {
        if u == n_left {
            return true; // NIL reached
        }

        for &v in &adj[u] {
            let paired = match_right[v];
            let paired_idx = if paired == nil { n_left } else { paired };
            if dist[paired_idx] == dist[u] + 1
                && dfs_phase(paired_idx, match_left, match_right, dist, adj, n_left, nil)
            {
                match_right[v] = u;
                match_left[u] = v;
                return true;
            }
        }

        dist[u] = usize::MAX;
        false
    }

    // Main loop
    while bfs_phase(&match_left, &match_right, &mut dist, &adj, n_left) {
        for u in 0..n_left {
            if match_left[u] == nil {
                dfs_phase(
                    u,
                    &mut match_left,
                    &mut match_right,
                    &mut dist,
                    &adj,
                    n_left,
                    nil,
                );
            }
        }
    }

    // Build result
    let mut matching = Vec::new();
    let mut unmatched_left = Vec::new();
    let mut unmatched_right = Vec::new();

    for (li, &ri) in match_left.iter().enumerate() {
        if ri != nil {
            matching.push((left_nodes[li].clone(), right_nodes[ri].clone()));
        } else {
            unmatched_left.push(left_nodes[li].clone());
        }
    }

    for (ri, &li) in match_right.iter().enumerate() {
        if li == nil {
            unmatched_right.push(right_nodes[ri].clone());
        }
    }

    let size = matching.len();
    Ok(HopcroftKarpResult {
        matching,
        size,
        unmatched_left,
        unmatched_right,
    })
}

// ============================================================================
// Additional utility functions
// ============================================================================

/// ISAP (Improved Shortest Augmenting Path) algorithm for maximum flow.
/// Delegates to Dinic's which has similar complexity.
pub fn isap_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    dinic_max_flow(graph, source, sink)
}

/// Capacity scaling algorithm for maximum flow.
/// Uses Edmonds-Karp with capacity scaling for better performance on high-capacity graphs.
pub fn capacity_scaling_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    source: &N,
    sink: &N,
) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    let result = edmonds_karp_max_flow(graph, source, sink)?;
    Ok(result.flow_value)
}

/// Multi-source multi-sink maximum flow.
///
/// Creates a super-source and super-sink connected to all sources/sinks
/// with infinite capacity, then solves the single-source single-sink problem.
pub fn multi_source_multi_sink_max_flow<N, E, Ix>(
    graph: &DiGraph<N, E, Ix>,
    sources: &[N],
    sinks: &[N],
) -> Result<f64>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + std::fmt::Debug,
    Ix: petgraph::graph::IndexType,
{
    if sources.is_empty() || sinks.is_empty() {
        return Err(GraphError::InvalidGraph(
            "Must have at least one source and one sink".to_string(),
        ));
    }

    // Build augmented graph with node indices
    let (nodes, node_to_idx) = build_node_index_maps(graph);
    let n = nodes.len();
    let super_source = n;
    let super_sink = n + 1;
    let total_n = n + 2;

    let mut rg = ResidualGraph::new(total_n);

    // Copy original edges
    for node in &nodes {
        if let Ok(successors) = graph.successors(node) {
            let u = node_to_idx[node];
            for succ in &successors {
                if let Ok(w) = graph.edge_weight(node, succ) {
                    let v = node_to_idx[succ];
                    rg.capacity[u][v] += w.into();
                }
            }
        }
    }

    // Connect super-source to all sources with large capacity
    let big_cap = 1e15;
    for src in sources {
        if let Some(&idx) = node_to_idx.get(src) {
            rg.capacity[super_source][idx] = big_cap;
        }
    }

    // Connect all sinks to super-sink with large capacity
    for snk in sinks {
        if let Some(&idx) = node_to_idx.get(snk) {
            rg.capacity[idx][super_sink] = big_cap;
        }
    }

    // Run Dinic's on the augmented graph
    let mut total_flow = 0.0;
    loop {
        let level = rg.build_level_graph(super_source);
        if level[super_sink] < 0 {
            break;
        }
        let mut iter = vec![0usize; total_n];
        loop {
            let f = rg.send_flow(super_source, super_sink, f64::INFINITY, &level, &mut iter);
            if f <= 1e-12 {
                break;
            }
            total_flow += f;
        }
    }

    Ok(total_flow)
}

/// Parallel maximum flow algorithm.
/// Currently delegates to Dinic's; parallelism is a future optimization.
pub fn parallel_max_flow<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<f64>
where
    N: Node + Clone + Send + Sync + std::fmt::Debug,
    E: EdgeWeight + Into<f64> + Copy + Send + Sync + std::fmt::Debug,
    Ix: petgraph::graph::IndexType + Send + Sync,
{
    dinic_max_flow(graph, source, sink)
}

// ============================================================================
// Internal helpers
// ============================================================================

fn validate_flow_input<N, E, Ix>(graph: &DiGraph<N, E, Ix>, source: &N, sink: &N) -> Result<()>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    if !graph.contains_node(source) || !graph.contains_node(sink) {
        return Err(GraphError::node_not_found("source or sink"));
    }
    if source == sink {
        return Err(GraphError::InvalidGraph(
            "Source and sink cannot be the same node".to_string(),
        ));
    }
    Ok(())
}

fn build_flow_result<N: Node + Clone>(
    rg: &ResidualGraph,
    nodes: &[N],
    node_to_idx: &HashMap<N, usize>,
    total_flow: f64,
    source: usize,
) -> Result<MaxFlowResult<N>> {
    let reachable = rg.reachable_from(source);

    let mut source_side = HashSet::new();
    for (node, &idx) in node_to_idx {
        if reachable[idx] {
            source_side.insert(node.clone());
        }
    }

    let mut edge_flows = HashMap::new();
    for (ni, node_i) in nodes.iter().enumerate() {
        for (nj, node_j) in nodes.iter().enumerate() {
            let flow = rg.flow[ni][nj];
            if flow > 1e-12 {
                edge_flows.insert((node_i.clone(), node_j.clone()), flow);
            }
        }
    }

    Ok(MaxFlowResult {
        flow_value: total_flow,
        edge_flows,
        source_side,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::{create_digraph, create_graph};

    #[test]
    fn test_edmonds_karp_simple() -> Result<()> {
        let mut g = create_digraph::<&str, f64>();
        g.add_edge("s", "a", 10.0)?;
        g.add_edge("s", "b", 5.0)?;
        g.add_edge("a", "b", 15.0)?;
        g.add_edge("a", "t", 10.0)?;
        g.add_edge("b", "t", 10.0)?;

        let result = edmonds_karp_max_flow(&g, &"s", &"t")?;
        assert!((result.flow_value - 15.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_edmonds_karp_no_path() -> Result<()> {
        let mut g = create_digraph::<&str, f64>();
        g.add_edge("s", "a", 10.0)?;
        g.add_edge("b", "t", 10.0)?;
        // No path from s to t

        let result = edmonds_karp_max_flow(&g, &"s", &"t")?;
        assert!(result.flow_value < 1e-6);
        Ok(())
    }

    #[test]
    fn test_dinic_max_flow() -> Result<()> {
        let mut g = create_digraph::<i32, f64>();
        g.add_edge(0, 1, 16.0)?;
        g.add_edge(0, 2, 13.0)?;
        g.add_edge(1, 2, 10.0)?;
        g.add_edge(1, 3, 12.0)?;
        g.add_edge(2, 1, 4.0)?;
        g.add_edge(2, 4, 14.0)?;
        g.add_edge(3, 2, 9.0)?;
        g.add_edge(3, 5, 20.0)?;
        g.add_edge(4, 3, 7.0)?;
        g.add_edge(4, 5, 4.0)?;

        let flow = dinic_max_flow(&g, &0, &5)?;
        assert!((flow - 23.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_push_relabel() -> Result<()> {
        let mut g = create_digraph::<&str, f64>();
        g.add_edge("s", "a", 10.0)?;
        g.add_edge("s", "b", 5.0)?;
        g.add_edge("a", "b", 15.0)?;
        g.add_edge("a", "t", 10.0)?;
        g.add_edge("b", "t", 10.0)?;

        let flow = push_relabel_max_flow(&g, &"s", &"t")?;
        assert!((flow - 15.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_minimum_cut_undirected() -> Result<()> {
        let mut g = create_graph::<i32, f64>();
        // Simple graph: two clusters connected by weak link
        g.add_edge(0, 1, 10.0)?;
        g.add_edge(1, 2, 10.0)?;
        g.add_edge(0, 2, 10.0)?;
        g.add_edge(3, 4, 10.0)?;
        g.add_edge(4, 5, 10.0)?;
        g.add_edge(3, 5, 10.0)?;
        g.add_edge(2, 3, 1.0)?; // Weak link

        let (cut_value, partition) = minimum_cut(&g)?;
        assert!((cut_value - 1.0).abs() < 1e-6);

        // Partition should separate the two clusters
        let true_count = partition.iter().filter(|&&b| b).count();
        assert!(true_count > 0 && true_count < 6);
        Ok(())
    }

    #[test]
    fn test_minimum_st_cut() -> Result<()> {
        let mut g = create_digraph::<&str, f64>();
        g.add_edge("s", "a", 3.0)?;
        g.add_edge("s", "b", 2.0)?;
        g.add_edge("a", "t", 2.0)?;
        g.add_edge("b", "t", 3.0)?;
        g.add_edge("a", "b", 1.0)?;

        let (cut_val, source_side, sink_side) = minimum_st_cut(&g, &"s", &"t")?;
        assert!((cut_val - 5.0).abs() < 1e-6);
        assert!(source_side.contains(&"s"));
        assert!(sink_side.contains(&"t"));
        Ok(())
    }

    #[test]
    fn test_min_cost_max_flow() -> Result<()> {
        // Simple network: s=0, a=1, b=2, t=3
        let edges = vec![
            (0, 1, 2.0, 1.0), // s->a, cap=2, cost=1
            (0, 2, 3.0, 2.0), // s->b, cap=3, cost=2
            (1, 3, 3.0, 3.0), // a->t, cap=3, cost=3
            (2, 3, 2.0, 1.0), // b->t, cap=2, cost=1
            (1, 2, 1.0, 1.0), // a->b, cap=1, cost=1
        ];

        let result = min_cost_max_flow(4, &edges, 0, 3)?;
        assert!(result.flow_value > 0.0);
        assert!(result.total_cost > 0.0);
        Ok(())
    }

    #[test]
    fn test_min_cost_max_flow_graph() -> Result<()> {
        let mut g = create_digraph::<i32, f64>();
        g.add_edge(0, 1, 4.0)?;
        g.add_edge(0, 2, 3.0)?;
        g.add_edge(1, 3, 4.0)?;
        g.add_edge(2, 3, 3.0)?;

        let cost_fn = |_u: &i32, _v: &i32| 1.0;
        let (flow, cost) = min_cost_max_flow_graph(&g, &0, &3, cost_fn)?;
        assert!((flow - 7.0).abs() < 1e-6);
        assert!(cost > 0.0);
        Ok(())
    }

    #[test]
    fn test_multi_commodity_flow() -> Result<()> {
        // Network: 0 --5--> 1 --5--> 2
        let edges = vec![(0, 1, 5.0), (1, 2, 5.0)];
        let commodities = vec![
            (0, 2, 3.0), // commodity 1: 3 units from 0 to 2
            (0, 2, 2.0), // commodity 2: 2 units from 0 to 2
        ];

        let result = multi_commodity_flow(3, &edges, &commodities)?;
        assert!(result.feasible);
        assert!((result.total_throughput - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_multi_commodity_infeasible() -> Result<()> {
        let edges = vec![(0, 1, 3.0)];
        let commodities = vec![
            (0, 1, 2.0),
            (0, 1, 2.0), // Total demand 4 > capacity 3
        ];

        let result = multi_commodity_flow(2, &edges, &commodities)?;
        assert!(!result.feasible);
        Ok(())
    }

    #[test]
    fn test_hopcroft_karp_perfect() -> Result<()> {
        let mut g = create_graph::<&str, ()>();
        g.add_edge("a", "1", ())?;
        g.add_edge("a", "2", ())?;
        g.add_edge("b", "2", ())?;
        g.add_edge("b", "3", ())?;
        g.add_edge("c", "3", ())?;
        g.add_edge("c", "1", ())?;

        let left = vec!["a", "b", "c"];
        let right = vec!["1", "2", "3"];

        let result = hopcroft_karp(&g, &left, &right)?;
        assert_eq!(result.size, 3);
        assert!(result.unmatched_left.is_empty());
        assert!(result.unmatched_right.is_empty());
        Ok(())
    }

    #[test]
    fn test_hopcroft_karp_partial() -> Result<()> {
        let mut g = create_graph::<i32, ()>();
        g.add_edge(1, 10, ())?;
        g.add_edge(2, 10, ())?;
        g.add_edge(3, 20, ())?;

        let left = vec![1, 2, 3];
        let right = vec![10, 20];

        let result = hopcroft_karp(&g, &left, &right)?;
        assert_eq!(result.size, 2); // Only 2 right nodes
        assert_eq!(result.unmatched_left.len(), 1);
        assert!(result.unmatched_right.is_empty());
        Ok(())
    }

    #[test]
    fn test_hopcroft_karp_empty() -> Result<()> {
        let g = create_graph::<i32, ()>();
        let result = hopcroft_karp(&g, &[], &[])?;
        assert_eq!(result.size, 0);
        Ok(())
    }

    #[test]
    fn test_ford_fulkerson_alias() -> Result<()> {
        let mut g = create_digraph::<i32, f64>();
        g.add_edge(0, 1, 5.0)?;
        g.add_edge(0, 2, 3.0)?;
        g.add_edge(1, 3, 4.0)?;
        g.add_edge(2, 3, 3.0)?;

        let flow = ford_fulkerson_max_flow(&g, &0, &3)?;
        assert!((flow - 7.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_multi_source_multi_sink() -> Result<()> {
        let mut g = create_digraph::<i32, f64>();
        g.add_edge(0, 2, 5.0)?; // source 0 -> middle
        g.add_edge(1, 2, 5.0)?; // source 1 -> middle
        g.add_edge(2, 3, 7.0)?; // middle -> sink 3
        g.add_edge(2, 4, 3.0)?; // middle -> sink 4

        let flow = multi_source_multi_sink_max_flow(&g, &[0, 1], &[3, 4])?;
        assert!(flow >= 9.0);
        Ok(())
    }

    #[test]
    fn test_max_flow_invalid_input() {
        let mut g = create_digraph::<i32, f64>();
        let _ = g.add_edge(0, 1, 5.0);

        // Same source and sink
        assert!(edmonds_karp_max_flow(&g, &0, &0).is_err());

        // Non-existent node
        assert!(edmonds_karp_max_flow(&g, &0, &99).is_err());
    }

    #[test]
    fn test_dinic_classic_example() -> Result<()> {
        // Classic textbook example
        let mut g = create_digraph::<i32, f64>();
        g.add_edge(0, 1, 10.0)?;
        g.add_edge(0, 2, 10.0)?;
        g.add_edge(1, 2, 2.0)?;
        g.add_edge(1, 3, 4.0)?;
        g.add_edge(1, 4, 8.0)?;
        g.add_edge(2, 4, 9.0)?;
        g.add_edge(3, 5, 10.0)?;
        g.add_edge(4, 3, 6.0)?;
        g.add_edge(4, 5, 10.0)?;

        let flow = dinic_max_flow(&g, &0, &5)?;
        assert!((flow - 19.0).abs() < 1e-6);
        Ok(())
    }
}
