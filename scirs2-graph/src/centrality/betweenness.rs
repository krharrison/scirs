//! Betweenness centrality via Brandes (2001) algorithm.
//!
//! Exact computation is O(VE) for unweighted graphs.  Approximate computation
//! samples a fraction of source nodes to achieve sub-quadratic runtime.

use crate::error::{GraphError, Result};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for betweenness centrality computation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BetweennessConfig {
    /// If `true` (default), divide by (n-1)(n-2) for undirected graphs so
    /// that the maximum possible value is 1.
    pub normalized: bool,
    /// If `true`, include endpoints s and t in paths through v.
    /// Default `false`.
    pub endpoint: bool,
    /// Fraction of source nodes to sample. Must be in (0, 1].
    /// Use 1.0 (default) for exact computation.
    pub sample_fraction: f64,
    /// Optional RNG seed for sampling (used when `sample_fraction < 1.0`).
    pub rng_seed: u64,
}

impl Default for BetweennessConfig {
    fn default() -> Self {
        Self {
            normalized: true,
            endpoint: false,
            sample_fraction: 1.0,
            rng_seed: 42,
        }
    }
}

// ── Internal adjacency list ───────────────────────────────────────────────────

struct AdjList {
    /// head[v] .. head[v+1] → indices into `nbr`
    head: Vec<usize>,
    nbr: Vec<usize>,
}

impl AdjList {
    fn build(edges: &[(usize, usize)], n: usize) -> Result<Self> {
        // Treat edges as undirected: add both directions.
        let mut lists: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in edges {
            if u >= n || v >= n {
                return Err(GraphError::InvalidParameter {
                    param: "edge".to_string(),
                    value: format!("({u},{v})"),
                    expected: format!("node indices in 0..{n}"),
                    context: "BetweennessCentrality adjacency list".to_string(),
                });
            }
            if u != v {
                lists[u].push(v);
                lists[v].push(u);
            }
        }
        let mut head = vec![0usize; n + 1];
        for v in 0..n {
            head[v + 1] = head[v] + lists[v].len();
        }
        let total = *head.last().unwrap_or(&0);
        let mut nbr = vec![0usize; total];
        for v in 0..n {
            let base = head[v];
            for (i, &u) in lists[v].iter().enumerate() {
                nbr[base + i] = u;
            }
        }
        Ok(Self { head, nbr })
    }

    #[inline]
    fn neighbors(&self, v: usize) -> &[usize] {
        &self.nbr[self.head[v]..self.head[v + 1]]
    }
}

// ── Brandes single-source accumulation ────────────────────────────────────────

/// Run one BFS from `source` and accumulate betweenness contributions into
/// `cb` (cb[v] is accumulated and NOT yet scaled).
fn brandes_single_source(
    adj: &AdjList,
    n: usize,
    source: usize,
    cb: &mut Vec<f64>,
    endpoint: bool,
) {
    // sigma[v] = number of shortest paths from source to v
    let mut sigma = vec![0i64; n];
    // dist[v] = shortest distance from source; i64::MAX = unreachable
    let mut dist = vec![i64::MAX; n];
    // predecessors of v on shortest paths from source
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    // delta[v] = dependency of source on v
    let mut delta = vec![0.0_f64; n];
    // BFS queue
    let mut queue = std::collections::VecDeque::with_capacity(n);
    // Stack of nodes in non-increasing order of distance (for back-propagation)
    let mut stack: Vec<usize> = Vec::with_capacity(n);

    sigma[source] = 1;
    dist[source] = 0;
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        stack.push(v);
        for &w in adj.neighbors(v) {
            // First visit to w?
            if dist[w] == i64::MAX {
                dist[w] = dist[v] + 1;
                queue.push_back(w);
            }
            // Is this a shortest path to w via v?
            if dist[w] == dist[v] + 1 {
                sigma[w] = sigma[w].saturating_add(sigma[v]);
                pred[w].push(v);
            }
        }
    }

    // If endpoint mode: credit source and all reachable targets
    if endpoint {
        for &w in &stack {
            if w != source && dist[w] != i64::MAX {
                let n_paths = sigma[w] as f64;
                if n_paths > 0.0 {
                    cb[source] += 1.0;
                    cb[w] += 1.0;
                }
            }
        }
    }

    // Back-propagation
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            let coeff = if sigma[w] > 0 {
                (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w])
            } else {
                0.0
            };
            delta[v] += coeff;
        }
        if w != source {
            cb[w] += delta[w];
        }
    }
}

// ── Tiny LCG for sampling ─────────────────────────────────────────────────────

/// A simple linear congruential generator used to select random sources
/// without pulling in an external crate.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x123456789abcdef0)
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        // LCG with Knuth constants
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as usize) % bound
    }
}

// ── Main function ─────────────────────────────────────────────────────────────

/// Compute betweenness centrality for every node.
///
/// Uses the Brandes 2001 O(VE) algorithm for unweighted graphs.
/// When `config.sample_fraction < 1.0`, a random subset of sources is used
/// and the result is rescaled to approximate the full-graph centrality.
///
/// # Arguments
/// * `adj`     – edge list as `(src, dst)` pairs (treated as undirected).
/// * `n_nodes` – total number of nodes.
/// * `config`  – algorithm parameters.
///
/// # Errors
/// * [`GraphError::InvalidParameter`] if `n_nodes` is 0 (returns an empty
///   `Vec` instead), or if an edge references an out-of-range node.
pub fn betweenness_centrality(
    adj: &[(usize, usize)],
    n_nodes: usize,
    config: &BetweennessConfig,
) -> Result<Vec<f64>> {
    if n_nodes == 0 {
        return Ok(vec![]);
    }

    if !(0.0 < config.sample_fraction && config.sample_fraction <= 1.0) {
        return Err(GraphError::InvalidParameter {
            param: "sample_fraction".to_string(),
            value: format!("{}", config.sample_fraction),
            expected: "value in (0, 1]".to_string(),
            context: "BetweennessCentrality".to_string(),
        });
    }

    let n = n_nodes;
    let adj_list = AdjList::build(adj, n)?;

    // Determine which sources to process
    let sources: Vec<usize> = if config.sample_fraction >= 1.0 {
        (0..n).collect()
    } else {
        let k = ((n as f64 * config.sample_fraction).ceil() as usize).max(1).min(n);
        let mut rng = Lcg::new(config.rng_seed);
        let mut indices: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle then take first k
        for i in (1..n).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }
        indices[..k].to_vec()
    };

    let n_sources = sources.len();
    let mut cb = vec![0.0_f64; n];

    for &s in &sources {
        brandes_single_source(&adj_list, n, s, &mut cb, config.endpoint);
    }

    // Rescale for sampling (undirected graph: each pair counted twice → halve)
    if n_sources < n {
        let scale = n as f64 / n_sources as f64;
        for x in cb.iter_mut() {
            *x *= scale;
        }
    }
    // Undirected: divide by 2 (each path u→v and v→u counted once)
    for x in cb.iter_mut() {
        *x /= 2.0;
    }

    // Normalise
    if config.normalized && n > 2 {
        let norm = ((n - 1) * (n - 2)) as f64 / 2.0;
        for x in cb.iter_mut() {
            *x /= norm;
        }
    }

    Ok(cb)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bc_path_graph_middle_highest() {
        // Path 0-1-2-3-4: node 2 (middle) has highest betweenness
        let edges: Vec<(usize, usize)> = (0..4).map(|i| (i, i + 1)).collect();
        let cfg = BetweennessConfig {
            normalized: false,
            ..Default::default()
        };
        let bc = betweenness_centrality(&edges, 5, &cfg).unwrap();
        // In an undirected path of 5 nodes, node 2 should have max centrality
        let max_node = bc
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_node, 2, "middle node should have highest BC; bc={bc:?}");
    }

    #[test]
    fn test_bc_star_hub_highest() {
        // Star with center 0, leaves 1..4
        let edges: Vec<(usize, usize)> = (1..5).map(|i| (0, i)).collect();
        let cfg = BetweennessConfig::default();
        let bc = betweenness_centrality(&edges, 5, &cfg).unwrap();
        let hub = bc[0];
        for leaf in 1..5 {
            assert!(
                hub > bc[leaf] || bc[leaf] == 0.0,
                "hub BC {hub} should be > leaf BC {}",
                bc[leaf]
            );
        }
        assert!(hub > 0.0, "hub should have positive BC");
    }

    #[test]
    fn test_bc_disconnected_graph() {
        // Two separate triangles: {0,1,2} and {3,4,5}
        let edges = vec![(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let cfg = BetweennessConfig {
            normalized: false,
            ..Default::default()
        };
        let bc = betweenness_centrality(&edges, 6, &cfg).unwrap();
        assert_eq!(bc.len(), 6);
        // All nodes are fully interconnected within their component, so BC ≈ 0
        for &b in &bc {
            assert!(b.abs() < 1e-9, "BC should be 0 for complete triangles; got {b}");
        }
    }

    #[test]
    fn test_bc_empty_graph() {
        let bc = betweenness_centrality(&[], 0, &BetweennessConfig::default()).unwrap();
        assert!(bc.is_empty());
    }

    #[test]
    fn test_bc_single_node() {
        let bc = betweenness_centrality(&[], 1, &BetweennessConfig::default()).unwrap();
        assert_eq!(bc.len(), 1);
        assert!((bc[0]).abs() < 1e-9);
    }

    #[test]
    fn test_bc_two_nodes() {
        // Single edge: neither endpoint is an intermediate
        let bc = betweenness_centrality(&[(0, 1)], 2, &BetweennessConfig::default()).unwrap();
        assert_eq!(bc.len(), 2);
        for &b in &bc {
            assert!(b.abs() < 1e-9, "no intermediate nodes; got {b}");
        }
    }

    #[test]
    fn test_bc_normalized_range() {
        let edges: Vec<(usize, usize)> = (0..4).map(|i| (i, i + 1)).collect();
        let cfg = BetweennessConfig {
            normalized: true,
            ..Default::default()
        };
        let bc = betweenness_centrality(&edges, 5, &cfg).unwrap();
        for &b in &bc {
            assert!(
                b >= 0.0 && b <= 1.0 + 1e-9,
                "normalized BC out of [0,1]: {b}"
            );
        }
    }

    #[test]
    fn test_bc_sample_fraction_runs() {
        let edges: Vec<(usize, usize)> = (0..9).map(|i| (i, i + 1)).collect();
        let cfg = BetweennessConfig {
            sample_fraction: 0.5,
            ..Default::default()
        };
        let bc = betweenness_centrality(&edges, 10, &cfg).unwrap();
        assert_eq!(bc.len(), 10);
        // Result is approximate; just verify non-negative and finite
        for &b in &bc {
            assert!(b >= 0.0 && b.is_finite(), "BC should be finite non-negative; got {b}");
        }
    }

    #[test]
    fn test_bc_sample_invalid_fraction() {
        let err = betweenness_centrality(
            &[],
            3,
            &BetweennessConfig {
                sample_fraction: 0.0,
                ..Default::default()
            },
        );
        assert!(err.is_err());
    }

    #[test]
    fn test_bc_invalid_edge() {
        // Node 99 is out of range for a 3-node graph
        let err = betweenness_centrality(&[(0, 99)], 3, &BetweennessConfig::default());
        assert!(err.is_err());
    }
}
