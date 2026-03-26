//! Parallel Strength-of-Connection Computation for AMG
//!
//! Strength-of-connection determines which matrix entries are "strong"
//! couplings, forming the basis for coarsening decisions.
//!
//! # Theory
//!
//! Node i strongly influences node j if:
//!   |a_ij| >= θ · max_{k≠i} |a_ik|
//!
//! where θ is the strength threshold (typically 0.25).
//!
//! # Parallelism
//!
//! Rows are partitioned among threads. Each thread independently computes
//! strength for its assigned rows. Since rows are disjoint, no synchronization
//! is needed during computation.

use crate::csr::CsrMatrix;
use std::sync::Arc;

/// Strength-of-connection graph: adjacency list of strong neighbors
#[derive(Debug, Clone)]
pub struct StrengthGraph {
    /// Number of nodes
    pub n: usize,
    /// `strong_neighbors[i]` = nodes j such that i strongly influences j
    /// (i.e., |a_ij| >= θ * max_k |a_ik|)
    pub strong_neighbors: Vec<Vec<usize>>,
    /// `strong_influencers[i]` = nodes j such that j strongly influences i
    /// (transpose of strong_neighbors)
    pub strong_influencers: Vec<Vec<usize>>,
}

impl StrengthGraph {
    /// Create a StrengthGraph from the strong_neighbors adjacency list.
    /// Computes strong_influencers as the transpose.
    pub fn from_neighbors(n: usize, strong_neighbors: Vec<Vec<usize>>) -> Self {
        // Build transpose (strong influencers)
        let mut strong_influencers = vec![Vec::new(); n];
        for (i, neighbors) in strong_neighbors.iter().enumerate() {
            for &j in neighbors {
                if j < n {
                    strong_influencers[j].push(i);
                }
            }
        }
        Self {
            n,
            strong_neighbors,
            strong_influencers,
        }
    }

    /// Check if i strongly influences j
    pub fn is_strong(&self, i: usize, j: usize) -> bool {
        self.strong_neighbors.get(i).is_some_and(|v| v.contains(&j))
    }

    /// Check if i and j are strongly connected (in either direction)
    pub fn is_strongly_connected(&self, i: usize, j: usize) -> bool {
        self.is_strong(i, j) || self.is_strong(j, i)
    }
}

/// Compute strength-of-connection for a single row range [row_start, row_end).
/// Returns strong_neighbors for those rows (indexed relative to global row indices).
fn compute_strength_row_range(
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    theta: f64,
    row_start: usize,
    row_end: usize,
) -> Vec<(usize, Vec<usize>)> {
    let mut result = Vec::with_capacity(row_end - row_start);
    for i in row_start..row_end {
        let row_start_ptr = indptr[i];
        let row_end_ptr = indptr[i + 1];

        // Find max |a_ij| for j != i in row i
        let mut max_abs = 0.0f64;
        for pos in row_start_ptr..row_end_ptr {
            let j = indices[pos];
            if j != i {
                let v = data[pos].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
        }

        let threshold = theta * max_abs;
        let mut strong = Vec::new();

        if threshold > 0.0 {
            for pos in row_start_ptr..row_end_ptr {
                let j = indices[pos];
                if j != i && data[pos].abs() >= threshold {
                    strong.push(j);
                }
            }
        }

        result.push((i, strong));
    }
    result
}

/// Compute the parallel strength-of-connection graph.
///
/// Partitions rows among `n_threads` threads and computes strength for
/// each partition concurrently. Results are merged into a StrengthGraph.
///
/// # Arguments
///
/// * `a` - Input sparse matrix
/// * `theta` - Strength threshold (typically 0.25)
/// * `n_threads` - Number of threads to use
///
/// # Returns
///
/// StrengthGraph with strong neighbor lists for all nodes.
pub fn parallel_strength_of_connection(
    a: &CsrMatrix<f64>,
    theta: f64,
    n_threads: usize,
) -> StrengthGraph {
    let n = a.shape().0;
    if n == 0 {
        return StrengthGraph::from_neighbors(0, Vec::new());
    }

    let n_threads = n_threads.max(1);
    let indptr = Arc::new(a.indptr.clone());
    let indices = Arc::new(a.indices.clone());
    let data = Arc::new(a.data.clone());

    // Partition rows into blocks
    let chunk_size = (n + n_threads - 1) / n_threads;

    let mut strong_neighbors = vec![Vec::new(); n];

    // Use thread::scope to compute in parallel
    let mut all_results: Vec<Vec<(usize, Vec<usize>)>> = Vec::with_capacity(n_threads);

    std::thread::scope(|s| {
        let mut handles = Vec::new();

        for t in 0..n_threads {
            let row_start = t * chunk_size;
            let row_end = ((t + 1) * chunk_size).min(n);
            if row_start >= row_end {
                continue;
            }

            let indptr_ref = Arc::clone(&indptr);
            let indices_ref = Arc::clone(&indices);
            let data_ref = Arc::clone(&data);

            let handle = s.spawn(move || {
                compute_strength_row_range(
                    &indptr_ref,
                    &indices_ref,
                    &data_ref,
                    theta,
                    row_start,
                    row_end,
                )
            });
            handles.push(handle);
        }

        for h in handles {
            if let Ok(result) = h.join() {
                all_results.push(result);
            }
        }
    });

    // Merge results
    for chunk in all_results {
        for (i, neighbors) in chunk {
            strong_neighbors[i] = neighbors;
        }
    }

    StrengthGraph::from_neighbors(n, strong_neighbors)
}

/// Compute serial strength-of-connection (single-threaded baseline).
///
/// Useful for verification and small problems.
pub fn serial_strength_of_connection(a: &CsrMatrix<f64>, theta: f64) -> StrengthGraph {
    let n = a.shape().0;
    let mut strong_neighbors = vec![Vec::new(); n];

    for i in 0..n {
        let mut max_abs = 0.0f64;
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j != i {
                let v = a.data[pos].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
        }
        let threshold = theta * max_abs;
        if threshold > 0.0 {
            for pos in a.row_range(i) {
                let j = a.indices[pos];
                if j != i && a.data[pos].abs() >= threshold {
                    strong_neighbors[i].push(j);
                }
            }
        }
    }

    StrengthGraph::from_neighbors(n, strong_neighbors)
}

/// Compute the measure of importance λ_i for each node.
///
/// λ_i = |{j : i strongly influences j}| + 0.5 * |{j : j strongly influences i, i ∈ F-set}|
///
/// In the initial phase (before F-set is known), returns:
/// `lambda_i = |strong_neighbors[i]|` (number of nodes i influences)
///
/// # Arguments
///
/// * `strength` - The strength graph
///
/// # Returns
///
/// Vector of λ values indexed by node.
pub fn compute_lambda(strength: &StrengthGraph) -> Vec<f64> {
    let n = strength.n;
    let mut lambda = vec![0.0f64; n];
    for i in 0..n {
        // Count number of nodes that i strongly influences (out-degree in strong graph)
        lambda[i] = strength.strong_neighbors[i].len() as f64;
    }
    lambda
}

/// Update lambda values given a partial F-set labeling (`cf_splitting[i]` = 0 means F, 1 means C, 2 means undecided).
///
/// λ_i = |{j : i strongly influences j}| + 0.5 * |{j : j influences i and j is F-node}|
pub fn compute_lambda_with_fset(strength: &StrengthGraph, cf_splitting: &[u8]) -> Vec<f64> {
    let n = strength.n;
    let mut lambda = vec![0.0f64; n];
    for i in 0..n {
        // Out-degree: number of nodes i strongly influences
        let out_degree = strength.strong_neighbors[i].len() as f64;
        // Count F-node influencers of i
        let f_influencers = strength
            .strong_influencers
            .get(i)
            .map(|influencers| {
                influencers
                    .iter()
                    .filter(|&&j| j < cf_splitting.len() && cf_splitting[j] == 0)
                    .count()
            })
            .unwrap_or(0);
        lambda[i] = out_degree + 0.5 * f_influencers as f64;
    }
    lambda
}

/// Compute the undirected strength-of-connection graph.
///
/// i and j have an undirected connection if:
///   |a_ij| >= θ * max(max_k |a_ik|, max_k |a_jk|)
///
/// The resulting graph is symmetric.
///
/// # Arguments
///
/// * `a` - Input sparse matrix
/// * `theta` - Strength threshold
///
/// # Returns
///
/// Symmetric StrengthGraph.
pub fn undirected_strength(a: &CsrMatrix<f64>, theta: f64) -> StrengthGraph {
    let n = a.shape().0;

    // Compute row maxima
    let mut row_max = vec![0.0f64; n];
    for i in 0..n {
        let mut max_abs = 0.0f64;
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j != i {
                let v = a.data[pos].abs();
                if v > max_abs {
                    max_abs = v;
                }
            }
        }
        row_max[i] = max_abs;
    }

    let mut strong_neighbors = vec![Vec::new(); n];

    for i in 0..n {
        for pos in a.row_range(i) {
            let j = a.indices[pos];
            if j == i {
                continue;
            }
            // Undirected threshold: max of both row maxima
            let threshold = theta * row_max[i].max(row_max[j]);
            if threshold > 0.0 && a.data[pos].abs() >= threshold {
                // Add i→j edge (j→i will be added when processing row j)
                if !strong_neighbors[i].contains(&j) {
                    strong_neighbors[i].push(j);
                }
            }
        }
    }

    StrengthGraph::from_neighbors(n, strong_neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    fn laplacian_1d(n: usize) -> CsrMatrix<f64> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            rows.push(i);
            cols.push(i);
            vals.push(2.0f64);
        }
        for i in 0..n - 1 {
            rows.push(i);
            cols.push(i + 1);
            vals.push(-1.0f64);
            rows.push(i + 1);
            cols.push(i);
            vals.push(-1.0f64);
        }
        CsrMatrix::new(vals, rows, cols, (n, n)).expect("valid Laplacian")
    }

    #[test]
    fn test_strength_threshold() {
        let a = laplacian_1d(6);
        let g = serial_strength_of_connection(&a, 0.25);
        // For the 1D Laplacian, off-diagonals are -1, diagonal is 2
        // |a_ij| = 1, max |a_ik| for k != i = 1, threshold = 0.25 * 1 = 0.25
        // So all off-diagonal entries are strong
        for i in 0..6 {
            for &j in &g.strong_neighbors[i] {
                assert_ne!(i, j, "No self-loops in strong graph");
                // Verify strength: |a_ij| >= theta * max_k |a_ik|
                let aij = a.get(i, j).abs();
                let mut max_abs = 0.0f64;
                for pos in a.row_range(i) {
                    if a.indices[pos] != i {
                        let v = a.data[pos].abs();
                        if v > max_abs {
                            max_abs = v;
                        }
                    }
                }
                assert!(
                    aij >= 0.25 * max_abs,
                    "Strong connection must meet threshold"
                );
            }
        }
    }

    #[test]
    fn test_strength_parallel_matches_serial() {
        let a = laplacian_1d(16);
        let serial = serial_strength_of_connection(&a, 0.25);
        let parallel = parallel_strength_of_connection(&a, 0.25, 4);
        assert_eq!(serial.n, parallel.n);
        for i in 0..serial.n {
            let mut s = serial.strong_neighbors[i].clone();
            let mut p = parallel.strong_neighbors[i].clone();
            s.sort();
            p.sort();
            assert_eq!(s, p, "Mismatch at node {i}");
        }
    }

    #[test]
    fn test_undirected_strength_symmetric() {
        let a = laplacian_1d(8);
        let g = undirected_strength(&a, 0.25);
        // Verify symmetry: if i -> j then j -> i
        for i in 0..g.n {
            for &j in &g.strong_neighbors[i] {
                assert!(
                    g.strong_neighbors[j].contains(&i),
                    "Undirected strength must be symmetric: {i} -> {j} but not {j} -> {i}"
                );
            }
        }
    }

    #[test]
    fn test_lambda_computation() {
        let a = laplacian_1d(8);
        let g = serial_strength_of_connection(&a, 0.25);
        let lambda = compute_lambda(&g);
        assert_eq!(lambda.len(), 8);
        for &l in &lambda {
            assert!(l >= 0.0, "Lambda must be non-negative");
        }
    }

    #[test]
    fn test_parallel_strength_n_threads() {
        let a = laplacian_1d(20);
        for n_threads in [1, 2, 4] {
            let g = parallel_strength_of_connection(&a, 0.25, n_threads);
            assert_eq!(g.n, 20);
            // All interior nodes should have 2 strong neighbors
            for i in 1..19 {
                assert_eq!(
                    g.strong_neighbors[i].len(),
                    2,
                    "Interior node {i} should have 2 strong neighbors with {n_threads} threads"
                );
            }
        }
    }
}
