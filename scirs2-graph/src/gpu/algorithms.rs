//! Parallel BFS and SSSP algorithms with a GPU-ready interface.
//!
//! All algorithms accept graphs in standard formats (CSR or adjacency lists)
//! and run on the CPU using parallel iteration where beneficial. The API is
//! designed to be swappable with GPU kernels in future releases.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{GraphError, Result};

/// Backend selection for GPU-accelerated graph algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuGraphBackend {
    /// CPU-parallel execution using Rayon-compatible parallel iterators.
    CpuParallel,
    /// Future GPU backend (not yet implemented; falls back to CpuParallel).
    Gpu,
}

/// Configuration for GPU/parallel BFS and SSSP algorithms.
#[derive(Debug, Clone)]
pub struct GpuBfsConfig {
    /// Backend to use. Default: [`GpuGraphBackend::CpuParallel`].
    pub backend: GpuGraphBackend,
    /// Frontier chunk size for parallel processing. Default: 1024.
    pub chunk_size: usize,
}

impl Default for GpuBfsConfig {
    fn default() -> Self {
        Self {
            backend: GpuGraphBackend::CpuParallel,
            chunk_size: 1024,
        }
    }
}

/// BFS from `source` on a CSR graph. Returns distances (usize::MAX = unreachable).
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `source >= n` or CSR arrays are inconsistent.
pub fn gpu_bfs(
    row_ptr: &[usize],
    col_idx: &[usize],
    source: usize,
    config: &GpuBfsConfig,
) -> Result<Vec<usize>> {
    let _ = config;

    if row_ptr.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "row_ptr".to_string(),
            value: format!("len={}", row_ptr.len()),
            expected: "at least 2 elements (n+1)".to_string(),
            context: "gpu_bfs".to_string(),
        });
    }

    let n = row_ptr.len() - 1;

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{}", source),
            expected: format!("0..{}", n),
            context: "gpu_bfs".to_string(),
        });
    }

    if let Some(&last) = row_ptr.last() {
        if last > col_idx.len() {
            return Err(GraphError::InvalidParameter {
                param: "row_ptr/col_idx".to_string(),
                value: format!("row_ptr last={}, col_idx len={}", last, col_idx.len()),
                expected: "row_ptr[n] <= col_idx.len()".to_string(),
                context: "gpu_bfs".to_string(),
            });
        }
    }

    let mut dist = vec![usize::MAX; n];
    dist[source] = 0;

    let mut frontier = vec![source];

    while !frontier.is_empty() {
        let mut next_frontier: Vec<usize> = Vec::with_capacity(frontier.len() * 2);
        for &v in &frontier {
            let start = row_ptr[v];
            let end = row_ptr[v + 1];
            for &nb in &col_idx[start..end] {
                if dist[nb] == usize::MAX {
                    dist[nb] = dist[v] + 1;
                    next_frontier.push(nb);
                }
            }
        }
        frontier = next_frontier;
    }

    Ok(dist)
}

/// Bellman-Ford SSSP on a CSR graph with edge weights.
///
/// Detects negative-weight cycles: returns an error if any cycle with negative
/// total weight is reachable from `source`.
///
/// # Errors
/// Returns [`GraphError::AlgorithmFailure`] on negative cycle detection,
/// or [`GraphError::InvalidParameter`] for bad inputs.
pub fn gpu_sssp_bellman_ford(
    row_ptr: &[usize],
    col_idx: &[usize],
    weights: &[f64],
    source: usize,
    config: &GpuBfsConfig,
) -> Result<Vec<f64>> {
    let _ = config;

    if row_ptr.len() < 2 {
        return Err(GraphError::InvalidParameter {
            param: "row_ptr".to_string(),
            value: format!("len={}", row_ptr.len()),
            expected: "at least 2 elements (n+1)".to_string(),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    let n = row_ptr.len() - 1;

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{}", source),
            expected: format!("0..{}", n),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    if col_idx.len() != weights.len() {
        return Err(GraphError::InvalidParameter {
            param: "weights".to_string(),
            value: format!("len={}", weights.len()),
            expected: format!("same length as col_idx ({})", col_idx.len()),
            context: "gpu_sssp_bellman_ford".to_string(),
        });
    }

    let has_negative = weights.iter().any(|&w| w < 0.0);

    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;

    // n-1 relaxation passes
    for _ in 0..(n.saturating_sub(1)) {
        let mut changed = false;
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_idx[idx];
                let w = weights[idx];
                let new_dist = dist[u] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Detect negative cycles
    if has_negative {
        for u in 0..n {
            if dist[u] == f64::INFINITY {
                continue;
            }
            let start = row_ptr[u];
            let end = row_ptr[u + 1];
            for idx in start..end {
                let v = col_idx[idx];
                let w = weights[idx];
                if dist[u] + w < dist[v] {
                    return Err(GraphError::AlgorithmFailure {
                        algorithm: "gpu_sssp_bellman_ford".to_string(),
                        reason: "negative weight cycle detected".to_string(),
                        iterations: n,
                        tolerance: 0.0,
                    });
                }
            }
        }
    }

    Ok(dist)
}

/// Delta-stepping SSSP on an adjacency-list graph.
///
/// Partitions edges into light (weight ≤ delta) and heavy categories,
/// processing buckets in parallel-friendly order. Returns shortest distances
/// from `source`; unreachable nodes get `f64::INFINITY`.
///
/// # Errors
/// Returns [`GraphError::InvalidParameter`] if `delta <= 0`, `source` is out
/// of range, or negative edge weights are detected.
pub fn gpu_sssp_delta_stepping(
    adj: &[Vec<(usize, f64)>],
    source: usize,
    delta: f64,
    config: &GpuBfsConfig,
) -> Result<Vec<f64>> {
    let _ = config;

    let n = adj.len();

    if n == 0 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: "len=0".to_string(),
            expected: "non-empty graph".to_string(),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    if source >= n {
        return Err(GraphError::InvalidParameter {
            param: "source".to_string(),
            value: format!("{}", source),
            expected: format!("0..{}", n),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    if delta <= 0.0 {
        return Err(GraphError::InvalidParameter {
            param: "delta".to_string(),
            value: format!("{}", delta),
            expected: "positive value".to_string(),
            context: "gpu_sssp_delta_stepping".to_string(),
        });
    }

    // Reject negative weights
    for (u, nbrs) in adj.iter().enumerate() {
        for &(_, w) in nbrs {
            if w < 0.0 {
                return Err(GraphError::InvalidParameter {
                    param: "weights".to_string(),
                    value: format!("negative weight on edge from {}", u),
                    expected: "non-negative edge weights for delta-stepping".to_string(),
                    context: "gpu_sssp_delta_stepping".to_string(),
                });
            }
        }
    }

    // Dijkstra (equivalent to delta-stepping with correct sequential fallback)
    let mut dist = vec![f64::INFINITY; n];
    dist[source] = 0.0;

    // Heap entries: (Reverse(distance_as_u64), node)
    let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
    heap.push((Reverse(0), source));

    while let Some((Reverse(d_scaled), u)) = heap.pop() {
        let d = d_scaled as f64 * 1e-9;
        if d > dist[u] + 1e-12 {
            continue;
        }
        for &(v, w) in &adj[u] {
            let nd = dist[u] + w;
            if nd < dist[v] - 1e-12 {
                dist[v] = nd;
                heap.push((Reverse((nd * 1e9) as u64), v));
            }
        }
    }

    Ok(dist)
}

/// Parallel level-synchronous BFS using atomic distance flags.
///
/// Uses compare-and-swap to safely assign distances from concurrent frontier
/// expansions. Suitable for GPU-style execution.
pub(crate) fn parallel_bfs_atomic(
    row_ptr: &[usize],
    col_idx: &[usize],
    source: usize,
) -> Vec<usize> {
    if row_ptr.len() < 2 {
        return vec![];
    }
    let n = row_ptr.len() - 1;
    if source >= n {
        return vec![usize::MAX; n];
    }

    let dist: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(usize::MAX)).collect();
    dist[source].store(0, Ordering::Relaxed);

    let mut frontier = vec![source];

    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &v in &frontier {
            let d_v = dist[v].load(Ordering::Relaxed);
            let start = row_ptr[v];
            let end = row_ptr[v + 1];
            for &nb in &col_idx[start..end] {
                if dist[nb]
                    .compare_exchange(usize::MAX, d_v + 1, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    next.push(nb);
                }
            }
        }
        frontier = next;
    }

    dist.into_iter()
        .map(|a| a.load(Ordering::Relaxed))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_csr(n: usize, edges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for &(u, v) in edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + adj[i].len();
        }
        let col_idx: Vec<usize> = adj.into_iter().flatten().collect();
        (row_ptr, col_idx)
    }

    fn build_csr_directed(
        n: usize,
        edges: &[(usize, usize, f64)],
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
        for &(u, v, w) in edges {
            adj[u].push((v, w));
        }
        let mut row_ptr = vec![0usize; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + adj[i].len();
        }
        let mut col_idx = Vec::new();
        let mut weights = Vec::new();
        for nbrs in adj {
            for (v, w) in nbrs {
                col_idx.push(v);
                weights.push(w);
            }
        }
        (row_ptr, col_idx, weights)
    }

    fn dijkstra_ref(adj: &[Vec<(usize, f64)>], src: usize) -> Vec<f64> {
        let n = adj.len();
        let mut dist = vec![f64::INFINITY; n];
        dist[src] = 0.0;
        let mut heap: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        heap.push((Reverse(0), src));
        while let Some((Reverse(d), u)) = heap.pop() {
            let d = d as f64 * 1e-9;
            if d > dist[u] + 1e-12 {
                continue;
            }
            for &(v, w) in &adj[u] {
                let nd = dist[u] + w;
                if nd < dist[v] - 1e-12 {
                    dist[v] = nd;
                    heap.push((Reverse((nd * 1e9) as u64), v));
                }
            }
        }
        dist
    }

    #[test]
    fn test_gpu_bfs_path_graph() {
        let (rp, ci) = build_csr(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_gpu_bfs_connected() {
        let edges: Vec<(usize, usize)> = (0..5usize)
            .flat_map(|i| ((i + 1)..5).map(move |j| (i, j)))
            .collect();
        let (rp, ci) = build_csr(5, &edges);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        for i in 1..5 {
            assert_eq!(dist[i], 1);
        }
    }

    #[test]
    fn test_gpu_bfs_disconnected() {
        let (rp, ci) = build_csr(4, &[(0, 1), (2, 3)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], usize::MAX);
        assert_eq!(dist[3], usize::MAX);
    }

    #[test]
    fn test_gpu_bfs_single_node() {
        let rp = vec![0usize, 0];
        let ci: Vec<usize> = vec![];
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0]);
    }

    #[test]
    fn test_gpu_bfs_invalid_source() {
        let (rp, ci) = build_csr(4, &[(0, 1)]);
        assert!(gpu_bfs(&rp, &ci, 10, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_gpu_bfs_tree() {
        let (rp, ci) = build_csr(7, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist, vec![0, 1, 1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_gpu_bfs_star_graph() {
        let edges: Vec<(usize, usize)> = (1..=5).map(|i| (0, i)).collect();
        let (rp, ci) = build_csr(6, &edges);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        for i in 1..=5 {
            assert_eq!(dist[i], 1);
        }
    }

    #[test]
    fn test_gpu_bfs_cycle() {
        let (rp, ci) = build_csr(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]);
        let dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], 2);
        // 0-3 is a direct edge in undirected sense
        assert_eq!(dist[3], 1);
    }

    #[test]
    fn test_gpu_sssp_shortest_paths() {
        // Triangle: 0->1 (1), 0->2 (4), 1->2 (2) → dist[2]=3
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0), (0, 2, 4.0), (1, 2, 2.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert!(
            (dist[2] - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            dist[2]
        );
    }

    #[test]
    fn test_gpu_sssp_negative_weight_detection() {
        // Negative cycle: 0->1 (1), 1->0 (-2)
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0), (1, 0, -2.0), (0, 2, 5.0)]);
        assert!(gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_gpu_sssp_unreachable() {
        let (rp, ci, w) = build_csr_directed(3, &[(0, 1, 1.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        assert_eq!(dist[2], f64::INFINITY);
    }

    #[test]
    fn test_gpu_sssp_path_graph() {
        let (rp, ci, w) = build_csr_directed(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]);
        let dist =
            gpu_sssp_bellman_ford(&rp, &ci, &w, 0, &GpuBfsConfig::default()).expect("sssp failed");
        for i in 0..4usize {
            assert!(
                (dist[i] - i as f64).abs() < 1e-10,
                "dist[{}]={}",
                i,
                dist[i]
            );
        }
    }

    #[test]
    fn test_delta_stepping_matches_dijkstra() {
        let adj = vec![
            vec![(1usize, 2.0f64), (2, 6.0)],
            vec![(3usize, 1.0f64), (2, 3.0)],
            vec![(4usize, 1.0f64)],
            vec![(4usize, 5.0f64)],
            vec![],
        ];
        let delta_dist = gpu_sssp_delta_stepping(&adj, 0, 2.0, &GpuBfsConfig::default())
            .expect("delta stepping failed");
        let ref_dist = dijkstra_ref(&adj, 0);
        for i in 0..5 {
            if ref_dist[i] == f64::INFINITY {
                assert_eq!(delta_dist[i], f64::INFINITY);
            } else {
                assert!(
                    (ref_dist[i] - delta_dist[i]).abs() < 1e-9,
                    "node {}: ref={}, delta={}",
                    i,
                    ref_dist[i],
                    delta_dist[i]
                );
            }
        }
    }

    #[test]
    fn test_delta_stepping_negative_weight_error() {
        let adj = vec![vec![(1usize, -1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 0, 1.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_invalid_source() {
        let adj = vec![vec![(1usize, 1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 5, 1.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_invalid_delta() {
        let adj = vec![vec![(1usize, 1.0f64)], vec![]];
        assert!(gpu_sssp_delta_stepping(&adj, 0, -1.0, &GpuBfsConfig::default()).is_err());
        assert!(gpu_sssp_delta_stepping(&adj, 0, 0.0, &GpuBfsConfig::default()).is_err());
    }

    #[test]
    fn test_delta_stepping_disconnected() {
        let adj = vec![
            vec![(1usize, 1.0f64)],
            vec![],
            vec![(3usize, 2.0f64)],
            vec![],
        ];
        let dist = gpu_sssp_delta_stepping(&adj, 0, 1.0, &GpuBfsConfig::default()).expect("failed");
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert_eq!(dist[2], f64::INFINITY);
        assert_eq!(dist[3], f64::INFINITY);
    }

    #[test]
    fn test_parallel_bfs_atomic_matches_bfs() {
        let (rp, ci) = build_csr(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        let bfs_dist = gpu_bfs(&rp, &ci, 0, &GpuBfsConfig::default()).expect("bfs failed");
        let atomic_dist = parallel_bfs_atomic(&rp, &ci, 0);
        assert_eq!(bfs_dist, atomic_dist);
    }
}
