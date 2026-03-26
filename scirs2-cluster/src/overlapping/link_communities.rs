//! Link Communities: partition edges (not nodes) to detect overlapping communities.
//!
//! Each edge belongs to exactly one community; nodes belong to all communities of
//! their incident edges, giving a natural overlap.
//!
//! Algorithm (Ahn et al., 2010):
//!   1. For every pair of edges (e_{ik}, e_{jk}) sharing node k compute
//!      the Jaccard similarity of their inclusive neighbourhoods N+(i)∩N+(j).
//!   2. Perform single-linkage HAC on the edge-similarity graph.
//!   3. Cut the dendrogram at `similarity_threshold`.
//!   4. Collect nodes incident to each resulting edge-community.

use crate::error::{ClusteringError, Result as ClusterResult};
use std::collections::HashSet;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for Link Communities.
#[derive(Debug, Clone)]
pub struct LinkCommunityConfig {
    /// HAC dendrogram cut height (similarity).  Edges whose merge-level exceeds
    /// this threshold are placed in the same community.  Default: 0.5.
    pub similarity_threshold: f64,
    /// Communities with fewer than this many edges are discarded.  Default: 2.
    pub min_community_size: usize,
}

impl Default for LinkCommunityConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.5,
            min_community_size: 2,
        }
    }
}

// ─── Result type ──────────────────────────────────────────────────────────────

/// A community of edges, together with the set of incident nodes.
#[derive(Debug, Clone)]
pub struct EdgeCommunity {
    /// The edges (u, v) with u < v that belong to this community.
    pub edges: Vec<(usize, usize)>,
    /// Deduplicated, sorted list of nodes incident to the edges.
    pub nodes: Vec<usize>,
}

// ─── LinkCommunities struct ───────────────────────────────────────────────────

/// Link Communities overlapping community detector.
pub struct LinkCommunities {
    config: LinkCommunityConfig,
}

impl LinkCommunities {
    /// Create a new `LinkCommunities` instance.
    pub fn new(config: LinkCommunityConfig) -> Self {
        Self { config }
    }

    /// Detect link communities in the graph.
    ///
    /// `adj[u]` must list the undirected neighbours of `u`.  Every edge must appear
    /// in both directions.  Self-loops are ignored.
    pub fn detect(&self, adj: &[Vec<usize>]) -> ClusterResult<Vec<EdgeCommunity>> {
        let n = adj.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Build canonical edge list (u < v).
        let edges = build_edge_list(adj);
        let m = edges.len();
        if m == 0 {
            return Ok(Vec::new());
        }

        // Build sorted inclusive-neighbourhood lookup.
        let inc_nbrs: Vec<Vec<usize>> = (0..n)
            .map(|u| {
                let mut ns: Vec<usize> = adj[u].iter().copied().filter(|&v| v != u).collect();
                ns.push(u);
                ns.sort_unstable();
                ns.dedup();
                ns
            })
            .collect();

        // For every pair of edges sharing a node, compute similarity.
        // Index edges by node: node_to_edges[k] = list of edge indices incident to k.
        let mut node_to_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (ei, &(u, v)) in edges.iter().enumerate() {
            node_to_edges[u].push(ei);
            node_to_edges[v].push(ei);
        }

        // Collect (edge_a, edge_b, similarity) triples for sharing-node pairs.
        let mut similarities: Vec<(usize, usize, f64)> = Vec::new();
        for k in 0..n {
            let incident = &node_to_edges[k];
            let len = incident.len();
            for ii in 0..len {
                for jj in (ii + 1)..len {
                    let ea = incident[ii];
                    let eb = incident[jj];
                    // Nodes of the two edges other than k.
                    let (ua, va) = edges[ea];
                    let (ub, vb) = edges[eb];
                    let i = if ua == k { va } else { ua };
                    let j = if ub == k { vb } else { ub };
                    let sim = Self::edge_similarity(&inc_nbrs[i], &inc_nbrs[j]);
                    let (a, b) = if ea < eb { (ea, eb) } else { (eb, ea) };
                    similarities.push((a, b, sim));
                }
            }
        }

        // Single-linkage HAC.
        let assignments =
            single_linkage_hac(&edges, &similarities, self.config.similarity_threshold);

        // Group edges by community.
        let mut comm_edges: std::collections::HashMap<usize, Vec<(usize, usize)>> =
            std::collections::HashMap::new();
        for (ei, &comm) in assignments.iter().enumerate() {
            comm_edges.entry(comm).or_default().push(edges[ei]);
        }

        // Build EdgeCommunity list and apply min_community_size filter.
        let mut result: Vec<EdgeCommunity> = comm_edges
            .into_values()
            .filter(|edge_list| edge_list.len() >= self.config.min_community_size)
            .map(|edge_list| {
                let mut node_set: HashSet<usize> = HashSet::new();
                for &(u, v) in &edge_list {
                    node_set.insert(u);
                    node_set.insert(v);
                }
                let mut nodes: Vec<usize> = node_set.into_iter().collect();
                nodes.sort_unstable();
                let mut sorted_edges = edge_list;
                sorted_edges.sort_unstable();
                EdgeCommunity {
                    edges: sorted_edges,
                    nodes,
                }
            })
            .collect();

        result.sort_by_key(|ec| ec.edges.first().copied().unwrap_or((0, 0)));
        Ok(result)
    }

    // ── Similarity ────────────────────────────────────────────────────────────

    /// Jaccard similarity of two sorted inclusive-neighbourhood slices.
    ///
    /// Uses a merge of sorted slices for O(|N+(i)| + |N+(j)|) time.
    pub fn edge_similarity(ni: &[usize], nj: &[usize]) -> f64 {
        let (mut inter, mut union) = (0usize, 0usize);
        let (mut pi, mut pj) = (0usize, 0usize);
        while pi < ni.len() && pj < nj.len() {
            match ni[pi].cmp(&nj[pj]) {
                std::cmp::Ordering::Equal => {
                    inter += 1;
                    union += 1;
                    pi += 1;
                    pj += 1;
                }
                std::cmp::Ordering::Less => {
                    union += 1;
                    pi += 1;
                }
                std::cmp::Ordering::Greater => {
                    union += 1;
                    pj += 1;
                }
            }
        }
        union += ni.len() - pi + nj.len() - pj;
        if union == 0 {
            return 0.0;
        }
        inter as f64 / union as f64
    }
}

// ─── Single-linkage HAC ───────────────────────────────────────────────────────

/// Perform single-linkage hierarchical agglomerative clustering on edges.
///
/// Edges with no similarity entries are assigned to their own singleton communities.
/// Merges happen in decreasing order of similarity; we stop merging when the
/// similarity drops below `threshold`.
///
/// Returns a community assignment vector of length `edges.len()`.
pub fn single_linkage_hac(
    edges: &[(usize, usize)],
    similarities: &[(usize, usize, f64)],
    threshold: f64,
) -> Vec<usize> {
    let m = edges.len();
    // Union-Find
    let mut parent: Vec<usize> = (0..m).collect();
    let mut rank: Vec<usize> = vec![0; m];

    // Sort similarities in decreasing order.
    let mut sorted_sims = similarities.to_vec();
    sorted_sims.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    for (ea, eb, sim) in sorted_sims {
        if sim < threshold {
            break; // All remaining sims are even lower.
        }
        let ra = find(&mut parent, ea);
        let rb = find(&mut parent, eb);
        if ra != rb {
            union_by_rank(&mut parent, &mut rank, ra, rb);
        }
    }

    // Compress paths and return root as community label.
    (0..m).map(|i| find(&mut parent, i)).collect()
}

// ─── Union-Find helpers ───────────────────────────────────────────────────────

fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // Path halving
        x = parent[x];
    }
    x
}

fn union_by_rank(parent: &mut Vec<usize>, rank: &mut Vec<usize>, a: usize, b: usize) {
    match rank[a].cmp(&rank[b]) {
        std::cmp::Ordering::Less => parent[a] = b,
        std::cmp::Ordering::Greater => parent[b] = a,
        std::cmp::Ordering::Equal => {
            parent[b] = a;
            rank[a] += 1;
        }
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────────

/// Build a canonical edge list (u < v) from an adjacency list.
fn build_edge_list(adj: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut seen: HashSet<(usize, usize)> = HashSet::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (u, neighbours) in adj.iter().enumerate() {
        for &v in neighbours {
            if u < v {
                let key = (u, v);
                if seen.insert(key) {
                    edges.push(key);
                }
            }
        }
    }
    edges.sort_unstable();
    edges
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_adj() -> Vec<Vec<usize>> {
        vec![vec![1, 2], vec![0, 2], vec![0, 1]]
    }

    fn two_triangles_sharing_edge() -> Vec<Vec<usize>> {
        // Nodes 0,1,2,3; edges: (0,1),(0,2),(1,2),(1,3),(2,3)
        let mut adj = vec![vec![]; 4];
        for &(u, v) in &[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)] {
            adj[u].push(v);
            adj[v].push(u);
        }
        adj
    }

    #[test]
    fn test_link_communities_triangle() {
        let adj = triangle_adj();
        let config = LinkCommunityConfig {
            similarity_threshold: 0.0,
            min_community_size: 1,
        };
        let comms = LinkCommunities::new(config)
            .detect(&adj)
            .expect("detect triangle");
        // Triangle has 3 edges; all should end up in one community with sim=1.0/3.
        // With threshold=0 they all merge.
        assert!(!comms.is_empty());
        // Every community node must be a valid index.
        for c in &comms {
            for &nd in &c.nodes {
                assert!(nd < 3);
            }
        }
    }

    #[test]
    fn test_edge_similarity_symmetric() {
        let ni = vec![0, 1, 2];
        let nj = vec![1, 2, 3];
        let s1 = LinkCommunities::edge_similarity(&ni, &nj);
        let s2 = LinkCommunities::edge_similarity(&nj, &ni);
        assert!((s1 - s2).abs() < 1e-12, "similarity must be symmetric");
    }

    #[test]
    fn test_edge_similarity_range() {
        for _ in 0..20 {
            let ni: Vec<usize> = vec![0, 1, 3, 5];
            let nj: Vec<usize> = vec![1, 2, 3, 6];
            let s = LinkCommunities::edge_similarity(&ni, &nj);
            assert!((0.0..=1.0).contains(&s), "similarity must be in [0,1]");
        }
    }

    #[test]
    fn test_edge_similarity_identical() {
        let n = vec![0, 1, 2, 3];
        let s = LinkCommunities::edge_similarity(&n, &n);
        assert!((s - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_edge_similarity_disjoint() {
        let ni = vec![0, 1];
        let nj = vec![2, 3];
        let s = LinkCommunities::edge_similarity(&ni, &nj);
        assert!((s - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_link_communities_output_nodes_valid() {
        let adj = two_triangles_sharing_edge();
        let n = adj.len();
        let comms = LinkCommunities::new(LinkCommunityConfig::default())
            .detect(&adj)
            .expect("detect should succeed");
        for c in &comms {
            for &nd in &c.nodes {
                assert!(nd < n, "node {nd} out of bounds (n={n})");
            }
        }
    }

    #[test]
    fn test_link_communities_empty_graph() {
        let adj: Vec<Vec<usize>> = vec![];
        let comms = LinkCommunities::new(LinkCommunityConfig::default())
            .detect(&adj)
            .expect("detect empty graph");
        assert!(comms.is_empty());
    }

    #[test]
    fn test_link_communities_no_edges() {
        let adj = vec![vec![], vec![], vec![]];
        let comms = LinkCommunities::new(LinkCommunityConfig::default())
            .detect(&adj)
            .expect("detect no-edge graph");
        assert!(comms.is_empty());
    }

    #[test]
    fn test_link_communities_edges_canonical() {
        let adj = two_triangles_sharing_edge();
        let comms = LinkCommunities::new(LinkCommunityConfig {
            min_community_size: 1,
            ..Default::default()
        })
        .detect(&adj)
        .expect("detect");
        for c in &comms {
            for &(u, v) in &c.edges {
                assert!(u < v, "edge ({u},{v}) must be canonical u < v");
            }
        }
    }

    #[test]
    fn test_link_communities_min_size_filter() {
        let adj = two_triangles_sharing_edge();
        // Very high min size should filter everything.
        let comms = LinkCommunities::new(LinkCommunityConfig {
            min_community_size: 100,
            ..Default::default()
        })
        .detect(&adj)
        .expect("detect with high min size");
        assert!(comms.is_empty());
    }

    #[test]
    fn test_single_linkage_hac_all_merge() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let sims = vec![(0, 1, 0.8), (0, 2, 0.9), (1, 2, 0.7)];
        let assignments = single_linkage_hac(&edges, &sims, 0.5);
        // All should end up in the same community.
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
    }

    #[test]
    fn test_single_linkage_hac_no_merge() {
        let edges = vec![(0, 1), (2, 3)];
        let sims: Vec<(usize, usize, f64)> = vec![];
        let assignments = single_linkage_hac(&edges, &sims, 0.5);
        // No similarities → each edge in its own community.
        assert_ne!(assignments[0], assignments[1]);
    }
}
