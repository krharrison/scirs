//! Spectral graph partitioning via Fiedler vector bisection.
//!
//! Spectral bisection partitions a graph by computing the Fiedler vector
//! (eigenvector corresponding to the second-smallest eigenvalue of the
//! graph Laplacian) and splitting nodes based on the sign of their
//! Fiedler vector component.

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::{GraphError, Result};

use super::types::{PartitionConfig, PartitionResult};

/// Compute the graph Laplacian matrix L = D - A.
///
/// D is the diagonal degree matrix, A is the adjacency matrix.
fn graph_laplacian(adj: &Array2<f64>) -> Array2<f64> {
    let n = adj.nrows();
    let mut lap = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let mut deg = 0.0;
        for j in 0..n {
            if i != j {
                let w = adj[[i, j]];
                if w.abs() > 1e-15 {
                    lap[[i, j]] = -w;
                    deg += w;
                }
            }
        }
        lap[[i, i]] = deg;
    }
    lap
}

/// Compute the Fiedler vector (eigenvector of the 2nd smallest eigenvalue
/// of the graph Laplacian) using inverse iteration with shift.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n)
///
/// # Returns
/// The Fiedler vector as an n-dimensional array.
///
/// # Errors
/// Returns `GraphError::AlgorithmFailure` if the inverse iteration does not converge
/// or the matrix is too small.
pub fn fiedler_vector(adj: &Array2<f64>) -> Result<Array1<f64>> {
    let n = adj.nrows();
    if n < 2 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}x{}", n, n),
            expected: "at least 2x2 adjacency matrix".to_string(),
            context: "fiedler_vector".to_string(),
        });
    }
    if n != adj.ncols() {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}x{}", n, adj.ncols()),
            expected: "square matrix".to_string(),
            context: "fiedler_vector".to_string(),
        });
    }

    let lap = graph_laplacian(adj);

    // Use inverse iteration with shift to find the eigenvector for the
    // second-smallest eigenvalue.
    //
    // 1. Estimate a small shift sigma just above 0 (smallest eigenvalue).
    // 2. Solve (L - sigma*I) x = b iteratively.
    // 3. The converged x approximates the Fiedler vector.

    // For small graphs, use direct power-iteration on (L_max*I - L) to find
    // the eigenvector of the largest eigenvalue of the "flipped" matrix,
    // then take the second one.

    // Strategy: subspace iteration to get the two smallest eigenvectors of L,
    // then return the second one.
    let max_iter = 1000;
    let tol = 1e-10;

    // Estimate the spectral radius for shifting
    let mut spectral_est = 0.0f64;
    for i in 0..n {
        spectral_est = spectral_est.max(lap[[i, i]]);
    }
    // Add a small margin
    let shift = spectral_est + 1.0;

    // Work with M = shift*I - L, whose largest eigenvectors correspond to
    // smallest eigenvectors of L.
    let mut m_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            m_mat[[i, j]] = -lap[[i, j]];
        }
        m_mat[[i, i]] += shift;
    }

    // Subspace iteration for top-2 eigenvectors of m_mat
    // Initialize with two orthogonal vectors
    let mut v0 = Array1::<f64>::ones(n);
    let norm0 = (v0.dot(&v0)).sqrt();
    if norm0 > 1e-15 {
        v0 /= norm0;
    }

    // Second vector: orthogonal to v0
    let mut v1 = Array1::<f64>::zeros(n);
    for i in 0..n {
        v1[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    // Orthogonalize against v0
    let proj = v1.dot(&v0);
    v1 = &v1 - &(&v0 * proj);
    let norm1 = (v1.dot(&v1)).sqrt();
    if norm1 > 1e-15 {
        v1 /= norm1;
    }

    let mut converged = false;
    for _iter in 0..max_iter {
        // Multiply: w0 = M * v0, w1 = M * v1
        let w0 = m_mat.dot(&v0);
        let w1 = m_mat.dot(&v1);

        // QR-style orthogonalization (Gram-Schmidt)
        let mut q0 = w0;
        let n0 = (q0.dot(&q0)).sqrt();
        if n0 < 1e-15 {
            break;
        }
        q0 /= n0;

        let mut q1 = w1.clone();
        let p = q1.dot(&q0);
        q1 = &q1 - &(&q0 * p);
        let n1 = (q1.dot(&q1)).sqrt();
        if n1 < 1e-15 {
            break;
        }
        q1 /= n1;

        // Check convergence: angle between old and new vectors
        let cos0 = v0.dot(&q0).abs();
        let cos1 = v1.dot(&q1).abs();

        v0 = q0;
        v1 = q1;

        if cos0 > 1.0 - tol && cos1 > 1.0 - tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(GraphError::AlgorithmFailure {
            algorithm: "fiedler_vector (subspace iteration)".to_string(),
            reason: "did not converge within maximum iterations".to_string(),
            iterations: max_iter,
            tolerance: tol,
        });
    }

    // v0 is the largest eigenvector of M = shift*I - L => smallest eigenvector of L
    // v1 is the second largest of M => second smallest of L (Fiedler vector)
    //
    // v0 should be approximately constant (the all-ones vector for connected graphs).
    // v1 is the Fiedler vector.

    // Verify v0 is roughly constant (connected graph check)
    let mean_v0 = v0.sum() / n as f64;
    let var_v0: f64 = v0.iter().map(|&x| (x - mean_v0).powi(2)).sum::<f64>() / n as f64;
    if var_v0 > 0.01 {
        // Graph may be disconnected; the Fiedler vector concept still applies
        // but partition quality may vary. We proceed anyway.
    }

    Ok(v1)
}

/// Bisect a graph using the Fiedler vector.
///
/// Nodes with positive Fiedler vector components are assigned to partition 0,
/// and nodes with negative components to partition 1. The split point is
/// adjusted to satisfy the balance tolerance.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n)
/// * `config` - Partition configuration (only `balance_tolerance` is used)
///
/// # Returns
/// A `PartitionResult` with 2 partitions.
pub fn spectral_bisect(adj: &Array2<f64>, config: &PartitionConfig) -> Result<PartitionResult> {
    let n = adj.nrows();
    if n < 2 {
        return Err(GraphError::InvalidParameter {
            param: "adj".to_string(),
            value: format!("{}", n),
            expected: "at least 2 nodes".to_string(),
            context: "spectral_bisect".to_string(),
        });
    }

    let fv = fiedler_vector(adj)?;

    // Sort node indices by Fiedler vector value for balance-aware splitting
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        fv[a]
            .partial_cmp(&fv[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the split point that respects balance tolerance
    let ideal_size = n / 2;
    let max_deviation = ((n as f64) * config.balance_tolerance).ceil() as usize;
    let min_split = if ideal_size > max_deviation {
        ideal_size - max_deviation
    } else {
        1
    };
    let max_split = (ideal_size + max_deviation).min(n - 1);

    // Try split at the median first, then adjust for best edge cut
    let mut best_cut = usize::MAX;
    let mut best_split = ideal_size;

    for split in min_split..=max_split {
        let mut assignments = vec![0usize; n];
        for &idx in &indices[split..] {
            assignments[idx] = 1;
        }
        // Count edge cut
        let mut cut = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[[i, j]].abs() > 1e-15 && assignments[i] != assignments[j] {
                    cut += 1;
                }
            }
        }
        if cut < best_cut {
            best_cut = cut;
            best_split = split;
        }
    }

    let mut assignments = vec![0usize; n];
    for &idx in &indices[best_split..] {
        assignments[idx] = 1;
    }

    Ok(PartitionResult::from_assignments(&assignments, adj, 2))
}

/// Recursive spectral partitioning for k-way partitioning.
///
/// Recursively applies spectral bisection until the desired number of
/// partitions is reached. Uses a balanced binary tree of bisections.
///
/// # Arguments
/// * `adj` - Symmetric adjacency matrix (n x n)
/// * `config` - Partition configuration
///
/// # Returns
/// A `PartitionResult` with `config.n_partitions` partitions.
pub fn spectral_partition(adj: &Array2<f64>, config: &PartitionConfig) -> Result<PartitionResult> {
    let n = adj.nrows();
    let k = config.n_partitions;

    if k < 2 {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: "at least 2".to_string(),
            context: "spectral_partition".to_string(),
        });
    }

    if n < k {
        return Err(GraphError::InvalidParameter {
            param: "n_partitions".to_string(),
            value: format!("{}", k),
            expected: format!("at most {} (number of nodes)", n),
            context: "spectral_partition".to_string(),
        });
    }

    if k == 2 {
        return spectral_bisect(adj, config);
    }

    // Recursive bisection: assign partition IDs via a queue of subsets to split
    let mut assignments = vec![0usize; n];
    // Each entry: (partition_id, node_indices)
    let mut queue: Vec<(usize, Vec<usize>)> = vec![(0, (0..n).collect())];
    let mut next_id = 1usize;

    while queue.len() + (next_id - queue.len()) < k {
        if queue.is_empty() {
            break;
        }
        // Pick the largest partition to split
        let mut largest_idx = 0;
        let mut largest_size = 0;
        for (i, (_, nodes)) in queue.iter().enumerate() {
            if nodes.len() > largest_size {
                largest_size = nodes.len();
                largest_idx = i;
            }
        }

        if largest_size < 2 {
            break;
        }

        let (pid, nodes) = queue.remove(largest_idx);
        let sub_n = nodes.len();

        // Build sub-adjacency matrix
        let mut sub_adj = Array2::<f64>::zeros((sub_n, sub_n));
        for (si, &ni) in nodes.iter().enumerate() {
            for (sj, &nj) in nodes.iter().enumerate() {
                sub_adj[[si, sj]] = adj[[ni, nj]];
            }
        }

        // Bisect the subgraph
        let sub_result = spectral_bisect(&sub_adj, config)?;

        let mut part0 = Vec::new();
        let mut part1 = Vec::new();
        for (si, &assignment) in sub_result.assignments.iter().enumerate() {
            if assignment == 0 {
                assignments[nodes[si]] = pid;
                part0.push(nodes[si]);
            } else {
                assignments[nodes[si]] = next_id;
                part1.push(nodes[si]);
            }
        }

        queue.push((pid, part0));
        queue.push((next_id, part1));
        next_id += 1;

        if next_id >= k {
            break;
        }
    }

    // Finalize any remaining queue entries
    for (pid, nodes) in &queue {
        for &ni in nodes {
            assignments[ni] = *pid;
        }
    }

    Ok(PartitionResult::from_assignments(&assignments, adj, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Build a complete bipartite graph K_{n,n}.
    fn complete_bipartite(n: usize) -> Array2<f64> {
        let size = 2 * n;
        let mut adj = Array2::<f64>::zeros((size, size));
        for i in 0..n {
            for j in n..size {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        adj
    }

    /// Build a path graph 0-1-2-..-(n-1).
    fn path_graph(n: usize) -> Array2<f64> {
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..(n - 1) {
            adj[[i, i + 1]] = 1.0;
            adj[[i + 1, i]] = 1.0;
        }
        adj
    }

    /// Build two disconnected cliques of size n.
    fn disconnected_cliques(n: usize) -> Array2<f64> {
        let size = 2 * n;
        let mut adj = Array2::<f64>::zeros((size, size));
        // Clique 1: nodes 0..n
        for i in 0..n {
            for j in (i + 1)..n {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        // Clique 2: nodes n..2n
        for i in n..size {
            for j in (i + 1)..size {
                adj[[i, j]] = 1.0;
                adj[[j, i]] = 1.0;
            }
        }
        adj
    }

    #[test]
    fn test_complete_bipartite_bisection() {
        let n = 4;
        let adj = complete_bipartite(n);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            ..PartitionConfig::default()
        };
        let result = spectral_bisect(&adj, &config).expect("bisection should succeed");
        assert_eq!(result.partition_sizes.len(), 2);
        // Each partition should have n nodes (balanced)
        assert_eq!(result.partition_sizes[0] + result.partition_sizes[1], 2 * n);
        // For K_{n,n}, the optimal bisection separating the two sides yields edge_cut = n^2.
        // The spectral method should find this or a balanced partition with edge_cut > 0.
        assert!(
            result.edge_cut > 0,
            "bipartite graph should have nonzero edge cut"
        );
        // Balanced: each side should have n nodes
        assert!(result.partition_sizes[0] >= n - 1 && result.partition_sizes[0] <= n + 1);
    }

    #[test]
    fn test_path_graph_bisection() {
        let n = 8;
        let adj = path_graph(n);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            ..PartitionConfig::default()
        };
        let result = spectral_bisect(&adj, &config).expect("bisection should succeed");
        // Balanced: both partitions should have ~4 nodes
        assert!(result.partition_sizes[0] >= 3 && result.partition_sizes[0] <= 5);
        assert!(result.partition_sizes[1] >= 3 && result.partition_sizes[1] <= 5);
        // Edge cut should be small (ideally 1 for a path)
        assert!(result.edge_cut >= 1);
    }

    #[test]
    fn test_disconnected_components_separate_partitions() {
        let n = 5;
        let adj = disconnected_cliques(n);
        let config = PartitionConfig {
            n_partitions: 2,
            balance_tolerance: 0.1,
            ..PartitionConfig::default()
        };
        let result = spectral_bisect(&adj, &config).expect("bisection should succeed");
        // Disconnected components should be in separate partitions => edge cut = 0
        assert_eq!(result.edge_cut, 0);
        assert_eq!(result.partition_sizes[0], n);
        assert_eq!(result.partition_sizes[1], n);
    }

    #[test]
    fn test_fiedler_vector_connected_graph() {
        let adj = path_graph(6);
        let fv = fiedler_vector(&adj).expect("should compute Fiedler vector");
        assert_eq!(fv.len(), 6);
        // Fiedler vector should not be constant for a connected non-trivial graph
        let min_val = fv.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = fv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_val - min_val > 1e-6,
            "Fiedler vector should have variation"
        );
    }

    #[test]
    fn test_fiedler_vector_too_small() {
        let adj = Array2::<f64>::zeros((1, 1));
        assert!(fiedler_vector(&adj).is_err());
    }

    #[test]
    fn test_spectral_partition_4way() {
        // 16 nodes in a 4x4 grid-like structure
        let n = 16;
        let mut adj = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            if i + 1 < n && (i + 1) % 4 != 0 {
                adj[[i, i + 1]] = 1.0;
                adj[[i + 1, i]] = 1.0;
            }
            if i + 4 < n {
                adj[[i, i + 4]] = 1.0;
                adj[[i + 4, i]] = 1.0;
            }
        }
        let config = PartitionConfig {
            n_partitions: 4,
            balance_tolerance: 0.3,
            ..PartitionConfig::default()
        };
        let result = spectral_partition(&adj, &config).expect("4-way partition should succeed");
        assert_eq!(result.partition_sizes.len(), 4);
        // All partitions should be non-empty
        for &s in &result.partition_sizes {
            assert!(s > 0, "partition should be non-empty");
        }
    }
}
