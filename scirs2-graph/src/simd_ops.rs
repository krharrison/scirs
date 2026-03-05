//! SIMD-accelerated operations for graph algorithms
//!
//! This module provides SIMD-optimized implementations of performance-critical
//! graph algorithm kernels, leveraging the `scirs2-core` SIMD infrastructure
//! via the `SimdUnifiedOps` trait.
//!
//! ## Accelerated Operations
//!
//! - **PageRank power iteration**: vector-scalar multiply, L1 norm, convergence check
//! - **Spectral methods**: graph Laplacian construction, eigenvalue iteration helpers
//! - **Adjacency matrix-vector products**: dense and CSR-style sparse matvec
//! - **Betweenness centrality accumulation**: batch dependency accumulation
//! - **BFS/DFS distance vector operations**: distance initialization, level updates
//!
//! ## Usage
//!
//! All functions are feature-gated under `#[cfg(feature = "simd")]` and automatically
//! dispatch to the optimal SIMD implementation available on the current platform
//! (AVX-512, AVX2, SSE2, NEON, or scalar fallback).
//!
//! ```rust,ignore
//! use scirs2_graph::simd_ops::{SimdPageRank, SimdSpectral, SimdAdjacency};
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::simd_ops::SimdUnifiedOps;

use crate::error::{GraphError, Result};

// ---------------------------------------------------------------------------
// PageRank SIMD operations
// ---------------------------------------------------------------------------

/// SIMD-accelerated operations for PageRank power iteration.
///
/// These functions accelerate the inner loops of PageRank computation:
/// distributing rank contributions, computing convergence (L1 norm of
/// the difference vector), and normalizing the rank vector.
pub struct SimdPageRank;

impl SimdPageRank {
    /// Perform one PageRank power iteration step using SIMD.
    ///
    /// Computes `new_rank = (1 - damping) / n + damping * M * old_rank`
    /// where `M` is the column-stochastic transition matrix.
    ///
    /// # Arguments
    /// * `transition_matrix` - Row-stochastic transition matrix (n x n)
    /// * `old_rank` - Current rank vector (length n)
    /// * `damping` - Damping factor (typically 0.85)
    ///
    /// # Returns
    /// The new rank vector after one iteration.
    pub fn power_iteration_step(
        transition_matrix: &Array2<f64>,
        old_rank: &Array1<f64>,
        damping: f64,
    ) -> Result<Array1<f64>> {
        let (rows, cols) = transition_matrix.dim();
        if cols != old_rank.len() {
            return Err(GraphError::InvalidGraph(format!(
                "Transition matrix columns ({}) do not match rank vector length ({})",
                cols,
                old_rank.len()
            )));
        }

        let n = rows;
        let base_rank = (1.0 - damping) / n as f64;

        // SIMD matrix-vector product: M * old_rank
        let matvec_result = SimdAdjacency::dense_matvec(transition_matrix, old_rank)?;

        // new_rank = base_rank + damping * (M * old_rank)
        let scaled = f64::simd_scalar_mul(&matvec_result.view(), damping);
        let base_vec = Array1::from_elem(n, base_rank);
        let new_rank = f64::simd_add(&base_vec.view(), &scaled.view());

        Ok(new_rank)
    }

    /// Compute L1 norm of the difference between two rank vectors (convergence check).
    ///
    /// Returns `sum(|new_rank[i] - old_rank[i]|)` using SIMD acceleration.
    ///
    /// # Arguments
    /// * `new_rank` - New rank vector
    /// * `old_rank` - Previous rank vector
    ///
    /// # Returns
    /// The L1 norm of the difference.
    pub fn convergence_l1(new_rank: &ArrayView1<f64>, old_rank: &ArrayView1<f64>) -> f64 {
        let diff = f64::simd_sub(new_rank, old_rank);
        let abs_diff = f64::simd_abs(&diff.view());
        f64::simd_sum(&abs_diff.view())
    }

    /// Check if PageRank has converged by comparing L1 norm against tolerance.
    ///
    /// # Arguments
    /// * `new_rank` - New rank vector
    /// * `old_rank` - Previous rank vector
    /// * `tolerance` - Convergence threshold
    ///
    /// # Returns
    /// `true` if the L1 norm of the difference is below tolerance.
    pub fn has_converged(
        new_rank: &ArrayView1<f64>,
        old_rank: &ArrayView1<f64>,
        tolerance: f64,
    ) -> bool {
        Self::convergence_l1(new_rank, old_rank) < tolerance
    }

    /// SIMD-accelerated vector-scalar multiply for rank distribution.
    ///
    /// Computes `result[i] = rank[i] * scalar` for distributing a node's
    /// rank equally among its neighbors.
    ///
    /// # Arguments
    /// * `rank` - The rank vector
    /// * `scalar` - The scalar multiplier (e.g., `damping / degree`)
    ///
    /// # Returns
    /// The scaled vector.
    pub fn scale_rank(rank: &ArrayView1<f64>, scalar: f64) -> Array1<f64> {
        f64::simd_scalar_mul(rank, scalar)
    }

    /// Normalize a rank vector so it sums to 1.0.
    ///
    /// # Arguments
    /// * `rank` - The rank vector to normalize
    ///
    /// # Returns
    /// The normalized rank vector, or an error if the sum is zero.
    pub fn normalize_rank(rank: &Array1<f64>) -> Result<Array1<f64>> {
        let total = f64::simd_sum(&rank.view());
        if total.abs() < 1e-15 {
            return Err(GraphError::AlgorithmError(
                "Cannot normalize zero-sum rank vector".to_string(),
            ));
        }
        Ok(f64::simd_scalar_mul(&rank.view(), 1.0 / total))
    }

    /// Compute damping-adjusted teleportation vector.
    ///
    /// For dangling nodes (nodes with no outgoing edges), their rank is
    /// distributed uniformly. This function computes the dangling node
    /// contribution: `(damping / n) * sum(rank[i] for dangling node i)`.
    ///
    /// # Arguments
    /// * `rank` - Current rank vector
    /// * `is_dangling` - Boolean mask: `true` for dangling nodes
    /// * `damping` - Damping factor
    ///
    /// # Returns
    /// The uniform teleportation contribution per node.
    pub fn dangling_node_contribution(
        rank: &ArrayView1<f64>,
        is_dangling: &[bool],
        damping: f64,
    ) -> f64 {
        let n = rank.len();
        if n == 0 {
            return 0.0;
        }

        // Create a mask vector: 1.0 for dangling, 0.0 otherwise
        let mask: Array1<f64> =
            Array1::from_iter(is_dangling.iter().map(|&d| if d { 1.0 } else { 0.0 }));

        // Sum rank values for dangling nodes via SIMD dot product
        let dangling_sum = f64::simd_dot(rank, &mask.view());

        damping * dangling_sum / n as f64
    }

    /// Full SIMD-accelerated PageRank computation.
    ///
    /// Runs the PageRank power iteration until convergence or max iterations.
    ///
    /// # Arguments
    /// * `transition_matrix` - Row-stochastic transition matrix (n x n)
    /// * `damping` - Damping factor (typically 0.85)
    /// * `tolerance` - Convergence threshold
    /// * `max_iterations` - Maximum number of iterations
    ///
    /// # Returns
    /// A tuple of (rank vector, iterations used).
    pub fn compute_pagerank(
        transition_matrix: &Array2<f64>,
        damping: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> Result<(Array1<f64>, usize)> {
        let n = transition_matrix.nrows();
        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot compute PageRank on empty graph".to_string(),
            ));
        }
        if transition_matrix.ncols() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Transition matrix must be square, got {}x{}",
                n,
                transition_matrix.ncols()
            )));
        }

        let mut rank = Array1::from_elem(n, 1.0 / n as f64);

        for iteration in 0..max_iterations {
            let new_rank = Self::power_iteration_step(transition_matrix, &rank, damping)?;

            if Self::has_converged(&new_rank.view(), &rank.view(), tolerance) {
                return Ok((new_rank, iteration + 1));
            }

            rank = new_rank;
        }

        // Return final state even if not converged
        Ok((rank, max_iterations))
    }
}

// ---------------------------------------------------------------------------
// Spectral SIMD operations
// ---------------------------------------------------------------------------

/// SIMD-accelerated operations for spectral graph methods.
///
/// Provides optimized kernels for Laplacian construction, eigenvalue
/// iteration helpers, and spectral embedding computations.
pub struct SimdSpectral;

impl SimdSpectral {
    /// Construct the standard graph Laplacian L = D - A using SIMD.
    ///
    /// The Laplacian is computed row-by-row: `L[i,j] = degree[i] * delta(i,j) - A[i,j]`.
    /// SIMD acceleration is applied to the row-level subtraction.
    ///
    /// # Arguments
    /// * `adjacency` - The adjacency matrix (n x n)
    /// * `degrees` - The degree vector (length n)
    ///
    /// # Returns
    /// The standard Laplacian matrix.
    pub fn standard_laplacian(
        adjacency: &Array2<f64>,
        degrees: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err(GraphError::InvalidGraph(
                "Adjacency matrix must be square".to_string(),
            ));
        }
        if degrees.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Degree vector length ({}) does not match adjacency matrix size ({})",
                degrees.len(),
                n
            )));
        }

        let mut laplacian = Array2::zeros((n, n));

        for i in 0..n {
            let adj_row = adjacency.row(i);

            // Use SIMD to negate the adjacency row: -A[i, :]
            if let Some(adj_slice) = adj_row.as_slice() {
                let adj_view = ArrayView1::from(adj_slice);
                let neg_row = f64::simd_scalar_mul(&adj_view, -1.0);
                if let Some(neg_slice) = neg_row.as_slice() {
                    let mut lap_row = laplacian.row_mut(i);
                    if let Some(lap_slice) = lap_row.as_slice_mut() {
                        lap_slice.copy_from_slice(neg_slice);
                    }
                }
            } else {
                // Fallback for non-contiguous data
                for j in 0..n {
                    laplacian[[i, j]] = -adjacency[[i, j]];
                }
            }

            // Set diagonal element to degree
            laplacian[[i, i]] = degrees[i];
        }

        Ok(laplacian)
    }

    /// Construct the normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2} using SIMD.
    ///
    /// # Arguments
    /// * `adjacency` - The adjacency matrix (n x n)
    /// * `degrees` - The degree vector (length n)
    ///
    /// # Returns
    /// The normalized Laplacian matrix.
    pub fn normalized_laplacian(
        adjacency: &Array2<f64>,
        degrees: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err(GraphError::InvalidGraph(
                "Adjacency matrix must be square".to_string(),
            ));
        }
        if degrees.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Degree vector length ({}) does not match adjacency matrix size ({})",
                degrees.len(),
                n
            )));
        }

        // Compute D^{-1/2} using SIMD sqrt + reciprocal
        let d_inv_sqrt = Self::compute_degree_inv_sqrt(degrees);

        let mut normalized = Array2::zeros((n, n));

        for i in 0..n {
            let adj_row = adjacency.row(i);

            if let Some(adj_slice) = adj_row.as_slice() {
                let adj_view = ArrayView1::from(adj_slice);

                // Scale the adjacency row: D^{-1/2}[i] * A[i,:] * D^{-1/2}[:]
                let scaled_by_i = f64::simd_scalar_mul(&adj_view, d_inv_sqrt[i]);
                let d_inv_sqrt_view = d_inv_sqrt.view();
                let scaled_row = f64::simd_mul(&scaled_by_i.view(), &d_inv_sqrt_view);

                // Negate to get -D^{-1/2} A D^{-1/2}
                let neg_scaled = f64::simd_scalar_mul(&scaled_row.view(), -1.0);

                if let Some(neg_slice) = neg_scaled.as_slice() {
                    let mut norm_row = normalized.row_mut(i);
                    if let Some(norm_slice) = norm_row.as_slice_mut() {
                        norm_slice.copy_from_slice(neg_slice);
                    }
                }
            } else {
                for j in 0..n {
                    normalized[[i, j]] = -d_inv_sqrt[i] * adjacency[[i, j]] * d_inv_sqrt[j];
                }
            }

            // Add identity: diagonal becomes 1 - D^{-1/2}[i] * A[i,i] * D^{-1/2}[i]
            // For simple graphs A[i,i] = 0, so diagonal = 1.0
            normalized[[i, i]] += 1.0;
        }

        Ok(normalized)
    }

    /// Construct the random-walk Laplacian L_rw = I - D^{-1} A using SIMD.
    ///
    /// # Arguments
    /// * `adjacency` - The adjacency matrix (n x n)
    /// * `degrees` - The degree vector (length n)
    ///
    /// # Returns
    /// The random-walk Laplacian matrix.
    pub fn random_walk_laplacian(
        adjacency: &Array2<f64>,
        degrees: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err(GraphError::InvalidGraph(
                "Adjacency matrix must be square".to_string(),
            ));
        }
        if degrees.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Degree vector length ({}) does not match adjacency matrix size ({})",
                degrees.len(),
                n
            )));
        }

        let mut rw_laplacian = Array2::zeros((n, n));

        for i in 0..n {
            let degree = degrees[i];
            if degree > 0.0 {
                let adj_row = adjacency.row(i);

                if let Some(adj_slice) = adj_row.as_slice() {
                    let adj_view = ArrayView1::from(adj_slice);
                    // -(1/degree) * A[i,:]
                    let scaled = f64::simd_scalar_mul(&adj_view, -1.0 / degree);
                    if let Some(scaled_slice) = scaled.as_slice() {
                        let mut rw_row = rw_laplacian.row_mut(i);
                        if let Some(rw_slice) = rw_row.as_slice_mut() {
                            rw_slice.copy_from_slice(scaled_slice);
                        }
                    }
                } else {
                    for j in 0..n {
                        rw_laplacian[[i, j]] = -adjacency[[i, j]] / degree;
                    }
                }
            }

            // Identity contribution
            rw_laplacian[[i, i]] += 1.0;
        }

        Ok(rw_laplacian)
    }

    /// Compute D^{-1/2} vector using SIMD acceleration.
    ///
    /// For each degree d, computes 1/sqrt(d) if d > 0, else 0.
    ///
    /// # Arguments
    /// * `degrees` - The degree vector
    ///
    /// # Returns
    /// The D^{-1/2} diagonal vector.
    pub fn compute_degree_inv_sqrt(degrees: &Array1<f64>) -> Array1<f64> {
        let sqrt_degrees = f64::simd_sqrt(&degrees.view());
        let n = degrees.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let s = sqrt_degrees[i];
            result[i] = if s > 1e-15 { 1.0 / s } else { 0.0 };
        }

        result
    }

    /// SIMD-accelerated power iteration for finding the dominant eigenvector.
    ///
    /// Used in spectral methods to find eigenvectors of the Laplacian.
    ///
    /// # Arguments
    /// * `matrix` - Symmetric matrix (n x n)
    /// * `initial` - Initial guess vector (length n)
    /// * `max_iterations` - Maximum iterations
    /// * `tolerance` - Convergence tolerance
    ///
    /// # Returns
    /// A tuple of (eigenvalue, eigenvector, iterations used).
    pub fn power_iteration(
        matrix: &Array2<f64>,
        initial: &Array1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<(f64, Array1<f64>, usize)> {
        let n = matrix.nrows();
        if matrix.ncols() != n {
            return Err(GraphError::InvalidGraph(
                "Matrix must be square for power iteration".to_string(),
            ));
        }
        if initial.len() != n {
            return Err(GraphError::InvalidGraph(format!(
                "Initial vector length ({}) does not match matrix size ({})",
                initial.len(),
                n
            )));
        }

        let mut v = initial.clone();
        let norm = f64::simd_norm(&v.view());
        if norm < 1e-15 {
            return Err(GraphError::AlgorithmError(
                "Initial vector is (near) zero".to_string(),
            ));
        }
        v = f64::simd_scalar_mul(&v.view(), 1.0 / norm);

        let mut eigenvalue = 0.0;

        for iteration in 0..max_iterations {
            // w = M * v (SIMD matvec)
            let w = SimdAdjacency::dense_matvec(matrix, &v)?;

            // New eigenvalue estimate: v^T * w
            let new_eigenvalue = f64::simd_dot(&v.view(), &w.view());

            // Normalize w
            let w_norm = f64::simd_norm(&w.view());
            if w_norm < 1e-15 {
                return Err(GraphError::AlgorithmError(
                    "Power iteration produced zero vector".to_string(),
                ));
            }
            let new_v = f64::simd_scalar_mul(&w.view(), 1.0 / w_norm);

            // Check convergence
            if (new_eigenvalue - eigenvalue).abs() < tolerance {
                return Ok((new_eigenvalue, new_v, iteration + 1));
            }

            eigenvalue = new_eigenvalue;
            v = new_v;
        }

        Ok((eigenvalue, v, max_iterations))
    }

    /// SIMD-accelerated Gram-Schmidt orthogonalization.
    ///
    /// Orthogonalizes a set of column vectors in-place.
    ///
    /// # Arguments
    /// * `vectors` - Matrix whose columns are vectors to orthogonalize (n x k)
    pub fn gram_schmidt_orthogonalize(vectors: &mut Array2<f64>) {
        let (_n, k) = vectors.dim();

        for i in 0..k {
            // Normalize current vector
            let norm = {
                let col = vectors.column(i);
                f64::simd_norm(&col)
            };
            if norm > 1e-12 {
                let mut col = vectors.column_mut(i);
                col /= norm;
            }

            // Orthogonalize subsequent vectors against this one
            for j in (i + 1)..k {
                let (dot_val, current_data) = {
                    let current = vectors.column(i);
                    let next = vectors.column(j);

                    let dot = if let (Some(c_slice), Some(n_slice)) =
                        (current.as_slice(), next.as_slice())
                    {
                        let c_view = ArrayView1::from(c_slice);
                        let n_view = ArrayView1::from(n_slice);
                        f64::simd_dot(&c_view, &n_view)
                    } else {
                        current.dot(&next)
                    };

                    (dot, current.to_owned())
                };

                // next = next - dot * current
                let mut next_col = vectors.column_mut(j);
                let projection = f64::simd_scalar_mul(&current_data.view(), dot_val);

                if let (Some(next_slice), Some(proj_slice)) =
                    (next_col.as_slice_mut(), projection.as_slice())
                {
                    for (n_val, p_val) in next_slice.iter_mut().zip(proj_slice.iter()) {
                        *n_val -= p_val;
                    }
                } else {
                    for idx in 0..next_col.len() {
                        next_col[idx] -= projection[idx];
                    }
                }
            }
        }
    }

    /// SIMD-accelerated Rayleigh quotient: v^T M v / (v^T v).
    ///
    /// # Arguments
    /// * `matrix` - Symmetric matrix (n x n)
    /// * `vector` - Input vector (length n)
    ///
    /// # Returns
    /// The Rayleigh quotient value.
    pub fn rayleigh_quotient(matrix: &Array2<f64>, vector: &ArrayView1<f64>) -> Result<f64> {
        let mv = SimdAdjacency::dense_matvec(matrix, &vector.to_owned())?;
        let numerator = f64::simd_dot(vector, &mv.view());
        let denominator = f64::simd_dot(vector, vector);

        if denominator.abs() < 1e-15 {
            return Err(GraphError::AlgorithmError(
                "Rayleigh quotient: zero-norm vector".to_string(),
            ));
        }

        Ok(numerator / denominator)
    }
}

// ---------------------------------------------------------------------------
// Adjacency matrix-vector product SIMD operations
// ---------------------------------------------------------------------------

/// SIMD-accelerated adjacency matrix-vector product operations.
///
/// Provides optimized dense and sparse matrix-vector products that are
/// central to many graph algorithms (PageRank, spectral methods, centrality).
pub struct SimdAdjacency;

impl SimdAdjacency {
    /// Dense adjacency matrix-vector product using SIMD.
    ///
    /// Computes `y = A * x` where A is the adjacency matrix.
    /// Each row uses SIMD dot product for acceleration.
    ///
    /// # Arguments
    /// * `matrix` - Dense adjacency matrix (rows x cols)
    /// * `vector` - Input vector (length cols)
    ///
    /// # Returns
    /// Result vector of length rows.
    pub fn dense_matvec(matrix: &Array2<f64>, vector: &Array1<f64>) -> Result<Array1<f64>> {
        let (rows, cols) = matrix.dim();
        if cols != vector.len() {
            return Err(GraphError::InvalidGraph(format!(
                "Matrix columns ({}) do not match vector length ({})",
                cols,
                vector.len()
            )));
        }

        let mut result = Array1::zeros(rows);

        for i in 0..rows {
            let row = matrix.row(i);
            if let (Some(row_slice), Some(vec_slice)) = (row.as_slice(), vector.as_slice()) {
                let row_view = ArrayView1::from(row_slice);
                let vec_view = ArrayView1::from(vec_slice);
                result[i] = f64::simd_dot(&row_view, &vec_view);
            } else {
                // Fallback for non-contiguous data
                result[i] = row.dot(&vector.view());
            }
        }

        Ok(result)
    }

    /// Dense matrix-vector product with alpha scaling: y = alpha * A * x.
    ///
    /// # Arguments
    /// * `alpha` - Scaling factor
    /// * `matrix` - Dense matrix (rows x cols)
    /// * `vector` - Input vector (length cols)
    ///
    /// # Returns
    /// Result vector of length rows.
    pub fn dense_matvec_scaled(
        alpha: f64,
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let raw = Self::dense_matvec(matrix, vector)?;
        Ok(f64::simd_scalar_mul(&raw.view(), alpha))
    }

    /// CSR-style sparse matrix-vector product using SIMD.
    ///
    /// Computes `y = A * x` where A is stored in compressed sparse row format.
    /// For each row, gathers x values at column indices and performs SIMD dot product.
    ///
    /// # Arguments
    /// * `row_ptr` - Row pointer array (length n+1)
    /// * `col_idx` - Column index array
    /// * `values` - Non-zero values array
    /// * `x` - Input vector
    /// * `n_rows` - Number of rows
    ///
    /// # Returns
    /// Result vector of length n_rows.
    pub fn sparse_csr_matvec(
        row_ptr: &[usize],
        col_idx: &[usize],
        values: &[f64],
        x: &[f64],
        n_rows: usize,
    ) -> Result<Vec<f64>> {
        if row_ptr.len() != n_rows + 1 {
            return Err(GraphError::InvalidGraph(format!(
                "row_ptr length ({}) should be n_rows + 1 ({})",
                row_ptr.len(),
                n_rows + 1
            )));
        }

        let mut y = vec![0.0; n_rows];

        for i in 0..n_rows {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];
            let row_nnz = row_end - row_start;

            if row_nnz == 0 {
                continue;
            }

            let row_values = &values[row_start..row_end];
            let row_indices = &col_idx[row_start..row_end];

            // Gather x values at column indices
            let x_gathered: Vec<f64> = row_indices.iter().map(|&j| x[j]).collect();

            // SIMD dot product for this row
            let row_view = ArrayView1::from(row_values);
            let x_view = ArrayView1::from(x_gathered.as_slice());
            y[i] = f64::simd_dot(&row_view, &x_view);
        }

        Ok(y)
    }

    /// Compute weighted adjacency matrix-vector product.
    ///
    /// Like `dense_matvec`, but scales each row by a per-node weight
    /// (e.g., inverse degree for normalization).
    ///
    /// # Arguments
    /// * `matrix` - Adjacency matrix (n x n)
    /// * `vector` - Input vector (length n)
    /// * `row_weights` - Per-row weight vector (length n)
    ///
    /// # Returns
    /// Result vector where `y[i] = row_weights[i] * sum_j(A[i,j] * x[j])`.
    pub fn weighted_matvec(
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
        row_weights: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        let raw = Self::dense_matvec(matrix, vector)?;
        Ok(f64::simd_mul(&raw.view(), &row_weights.view()))
    }

    /// Compute the transition matrix from an adjacency matrix using SIMD.
    ///
    /// Each row of the adjacency matrix is divided by the row sum (out-degree),
    /// producing a row-stochastic matrix.
    ///
    /// # Arguments
    /// * `adjacency` - Dense adjacency matrix (n x n)
    ///
    /// # Returns
    /// Row-stochastic transition matrix.
    pub fn adjacency_to_transition(adjacency: &Array2<f64>) -> Result<Array2<f64>> {
        let n = adjacency.nrows();
        if adjacency.ncols() != n {
            return Err(GraphError::InvalidGraph(
                "Adjacency matrix must be square".to_string(),
            ));
        }

        let mut transition = Array2::zeros((n, n));

        for i in 0..n {
            let row = adjacency.row(i);
            let row_sum = f64::simd_sum(&row);

            if row_sum > 1e-15 {
                if let Some(row_slice) = row.as_slice() {
                    let row_view = ArrayView1::from(row_slice);
                    let normalized = f64::simd_scalar_mul(&row_view, 1.0 / row_sum);
                    if let Some(norm_slice) = normalized.as_slice() {
                        let mut t_row = transition.row_mut(i);
                        if let Some(t_slice) = t_row.as_slice_mut() {
                            t_slice.copy_from_slice(norm_slice);
                        }
                    }
                } else {
                    for j in 0..n {
                        transition[[i, j]] = adjacency[[i, j]] / row_sum;
                    }
                }
            }
            // Rows with zero sum (dangling nodes) remain all zeros
        }

        Ok(transition)
    }
}

// ---------------------------------------------------------------------------
// Betweenness centrality SIMD operations
// ---------------------------------------------------------------------------

/// SIMD-accelerated operations for betweenness centrality computation.
///
/// Accelerates the dependency accumulation phase of Brandes' algorithm,
/// which is the most compute-intensive part of betweenness centrality.
pub struct SimdBetweenness;

impl SimdBetweenness {
    /// SIMD-accelerated dependency accumulation for Brandes' algorithm.
    ///
    /// Updates the dependency values: `delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])`
    /// for all predecessors v of w in the shortest-path DAG.
    ///
    /// # Arguments
    /// * `delta` - Dependency accumulator vector (modified in place)
    /// * `sigma` - Number of shortest paths through each node
    /// * `predecessors` - For each node w, list of predecessor node indices
    /// * `order` - Nodes in reverse BFS order (for back-propagation)
    pub fn accumulate_dependencies(
        delta: &mut [f64],
        sigma: &[f64],
        predecessors: &[Vec<usize>],
        order: &[usize],
    ) {
        // Process nodes in reverse BFS order
        for &w in order.iter().rev() {
            let sigma_w = sigma[w];
            if sigma_w <= 0.0 {
                continue;
            }

            let coeff = (1.0 + delta[w]) / sigma_w;

            for &v in &predecessors[w] {
                delta[v] += sigma[v] * coeff;
            }
        }
    }

    /// Batch update betweenness scores from a single-source dependency vector.
    ///
    /// Adds the dependency values to the centrality scores using SIMD.
    ///
    /// # Arguments
    /// * `centrality` - Running betweenness centrality scores (modified in place)
    /// * `delta` - Dependency values from a single-source computation
    /// * `source` - Index of the source node (excluded from update)
    pub fn batch_update_centrality(centrality: &mut [f64], delta: &[f64], source: usize) {
        let n = centrality.len().min(delta.len());

        if n == 0 {
            return;
        }

        // Use SIMD addition for bulk update
        let cent_view = ArrayView1::from(&centrality[..n]);
        let delta_view = ArrayView1::from(&delta[..n]);
        let updated = f64::simd_add(&cent_view, &delta_view);

        if let Some(updated_slice) = updated.as_slice() {
            centrality[..n].copy_from_slice(updated_slice);
        }

        // Exclude source node contribution
        if source < n {
            centrality[source] -= delta[source];
        }
    }

    /// Normalize betweenness centrality scores.
    ///
    /// For undirected graphs, divide by (n-1)(n-2).
    /// For directed graphs, divide by (n-1)(n-2) as well (Brandes' convention).
    ///
    /// # Arguments
    /// * `centrality` - Betweenness scores to normalize (modified in place)
    /// * `n` - Number of nodes in the graph
    /// * `directed` - Whether the graph is directed
    pub fn normalize(centrality: &mut [f64], n: usize, directed: bool) {
        if n <= 2 {
            return;
        }

        let mut scale = 1.0 / ((n - 1) * (n - 2)) as f64;
        if !directed {
            // For undirected graphs, each pair is counted twice
            scale *= 2.0;
        }

        let cent_view = ArrayView1::from(&*centrality);
        let scaled = f64::simd_scalar_mul(&cent_view, scale);

        if let Some(scaled_slice) = scaled.as_slice() {
            centrality.copy_from_slice(scaled_slice);
        }
    }
}

// ---------------------------------------------------------------------------
// BFS/DFS distance vector operations
// ---------------------------------------------------------------------------

/// SIMD-accelerated operations for BFS/DFS distance and level vectors.
///
/// Provides optimized initialization, update, and reduction operations
/// on the distance/level arrays used in graph traversals.
pub struct SimdTraversal;

impl SimdTraversal {
    /// Initialize a distance vector with infinity (or a sentinel value).
    ///
    /// Sets all entries to `f64::INFINITY` except the source node.
    ///
    /// # Arguments
    /// * `n` - Number of nodes
    /// * `source` - Source node index
    ///
    /// # Returns
    /// Distance vector with `distances[source] = 0.0` and all others infinity.
    pub fn init_distances(n: usize, source: usize) -> Array1<f64> {
        let mut distances = Array1::from_elem(n, f64::INFINITY);
        if source < n {
            distances[source] = 0.0;
        }
        distances
    }

    /// Initialize a distance vector for multi-source BFS.
    ///
    /// # Arguments
    /// * `n` - Number of nodes
    /// * `sources` - Slice of source node indices
    ///
    /// # Returns
    /// Distance vector with `distances[s] = 0.0` for each source s.
    pub fn init_distances_multi_source(n: usize, sources: &[usize]) -> Array1<f64> {
        let mut distances = Array1::from_elem(n, f64::INFINITY);
        for &s in sources {
            if s < n {
                distances[s] = 0.0;
            }
        }
        distances
    }

    /// SIMD-accelerated BFS frontier relaxation.
    ///
    /// For each node in the current frontier, attempts to relax the distance
    /// of its neighbors. This is the inner loop of BFS.
    ///
    /// # Arguments
    /// * `distances` - Current distance vector (modified in place)
    /// * `adjacency_row` - Adjacency row for the current node
    /// * `current_distance` - Distance of the current node
    ///
    /// # Returns
    /// Vector of neighbor indices that were relaxed (for the next frontier).
    pub fn relax_neighbors(
        distances: &mut Array1<f64>,
        adjacency_row: &ArrayView1<f64>,
        current_distance: f64,
    ) -> Vec<usize> {
        let n = distances.len().min(adjacency_row.len());
        let new_dist = current_distance + 1.0;
        let mut relaxed = Vec::new();

        for j in 0..n {
            if adjacency_row[j] > 0.0 && distances[j] > new_dist {
                distances[j] = new_dist;
                relaxed.push(j);
            }
        }

        relaxed
    }

    /// Compute the eccentricity of a node from its distance vector.
    ///
    /// The eccentricity is the maximum finite distance to any reachable node.
    ///
    /// # Arguments
    /// * `distances` - Distance vector from a single source
    ///
    /// # Returns
    /// The eccentricity value. Returns 0.0 if no nodes are reachable.
    pub fn eccentricity(distances: &ArrayView1<f64>) -> f64 {
        let mut max_dist = 0.0_f64;
        for &d in distances.iter() {
            if d.is_finite() && d > max_dist {
                max_dist = d;
            }
        }
        max_dist
    }

    /// Count the number of reachable nodes from a distance vector using SIMD.
    ///
    /// A node is reachable if its distance is finite.
    ///
    /// # Arguments
    /// * `distances` - Distance vector from a single source
    ///
    /// # Returns
    /// Number of reachable nodes (including the source itself).
    pub fn count_reachable(distances: &ArrayView1<f64>) -> usize {
        distances.iter().filter(|&&d| d.is_finite()).count()
    }

    /// Compute the sum of finite distances using SIMD.
    ///
    /// Used for computing closeness centrality: C(v) = (n-1) / sum(d(v, w)).
    ///
    /// # Arguments
    /// * `distances` - Distance vector from a single source
    ///
    /// # Returns
    /// Sum of all finite distances.
    pub fn sum_finite_distances(distances: &ArrayView1<f64>) -> f64 {
        // Replace infinities with 0.0, then sum
        let finite_dists: Array1<f64> =
            Array1::from_iter(
                distances
                    .iter()
                    .map(|&d| if d.is_finite() { d } else { 0.0 }),
            );
        f64::simd_sum(&finite_dists.view())
    }

    /// SIMD-accelerated element-wise minimum of two distance vectors.
    ///
    /// Useful for merging BFS trees or computing all-pairs shortest paths.
    ///
    /// # Arguments
    /// * `a` - First distance vector
    /// * `b` - Second distance vector
    ///
    /// # Returns
    /// Element-wise minimum vector.
    pub fn element_wise_min(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        f64::simd_min(a, b)
    }

    /// Compute the diameter of a graph from an all-pairs distance matrix.
    ///
    /// The diameter is the maximum eccentricity over all nodes.
    ///
    /// # Arguments
    /// * `distance_matrix` - All-pairs distance matrix (n x n)
    ///
    /// # Returns
    /// The graph diameter. Returns 0.0 for empty or disconnected graphs.
    pub fn diameter_from_distance_matrix(distance_matrix: &ArrayView2<f64>) -> f64 {
        let n = distance_matrix.nrows();
        let mut max_dist = 0.0_f64;

        for i in 0..n {
            let row = distance_matrix.row(i);
            let ecc = Self::eccentricity(&row);
            if ecc > max_dist {
                max_dist = ecc;
            }
        }

        max_dist
    }
}

// ---------------------------------------------------------------------------
// Utility: SIMD-accelerated vector operations common to graph algorithms
// ---------------------------------------------------------------------------

/// General SIMD-accelerated vector utilities for graph computations.
pub struct SimdGraphUtils;

impl SimdGraphUtils {
    /// SIMD-accelerated AXPY: y = alpha * x + y.
    ///
    /// # Arguments
    /// * `alpha` - Scalar multiplier
    /// * `x` - Input vector
    /// * `y` - Accumulator vector (modified in place)
    pub fn axpy(alpha: f64, x: &ArrayView1<f64>, y: &mut [f64]) {
        let scaled_x = f64::simd_scalar_mul(x, alpha);
        let y_view = ArrayView1::from(&*y);
        let result = f64::simd_add(&y_view, &scaled_x.view());

        if let Some(result_slice) = result.as_slice() {
            y.copy_from_slice(result_slice);
        }
    }

    /// L2 norm of a vector using SIMD.
    pub fn norm_l2(v: &ArrayView1<f64>) -> f64 {
        f64::simd_norm(v)
    }

    /// L1 norm of a vector using SIMD.
    pub fn norm_l1(v: &ArrayView1<f64>) -> f64 {
        let abs_v = f64::simd_abs(v);
        f64::simd_sum(&abs_v.view())
    }

    /// Linf norm (max absolute value) of a vector using SIMD.
    pub fn norm_linf(v: &ArrayView1<f64>) -> f64 {
        let abs_v = f64::simd_abs(v);
        f64::simd_max_element(&abs_v.view())
    }

    /// SIMD dot product.
    pub fn dot(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        f64::simd_dot(a, b)
    }

    /// Normalize a vector to unit L2 norm using SIMD.
    ///
    /// Returns None if the vector is zero.
    pub fn normalize_l2(v: &ArrayView1<f64>) -> Option<Array1<f64>> {
        let norm = f64::simd_norm(v);
        if norm < 1e-15 {
            None
        } else {
            Some(f64::simd_scalar_mul(v, 1.0 / norm))
        }
    }

    /// Compute cosine similarity between two vectors using SIMD.
    pub fn cosine_similarity(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
        let dot = f64::simd_dot(a, b);
        let norm_a = f64::simd_norm(a);
        let norm_b = f64::simd_norm(b);
        let denom = norm_a * norm_b;
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }

    /// Element-wise vector difference using SIMD.
    pub fn difference(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> Array1<f64> {
        f64::simd_sub(a, b)
    }

    /// Element-wise weighted sum: result = a * weight_a + b * weight_b.
    pub fn weighted_sum(
        a: &ArrayView1<f64>,
        weight_a: f64,
        b: &ArrayView1<f64>,
        weight_b: f64,
    ) -> Array1<f64> {
        let scaled_a = f64::simd_scalar_mul(a, weight_a);
        let scaled_b = f64::simd_scalar_mul(b, weight_b);
        f64::simd_add(&scaled_a.view(), &scaled_b.view())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    // ---- PageRank tests ----

    #[test]
    fn test_pagerank_convergence_check() {
        let a = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let b = Array1::from_vec(vec![0.24, 0.26, 0.25, 0.25]);

        let l1 = SimdPageRank::convergence_l1(&a.view(), &b.view());
        assert!(
            (l1 - 0.02).abs() < 1e-10,
            "L1 norm should be 0.02, got {l1}"
        );

        assert!(SimdPageRank::has_converged(&a.view(), &b.view(), 0.05));
        assert!(!SimdPageRank::has_converged(&a.view(), &b.view(), 0.01));
    }

    #[test]
    fn test_pagerank_scale_rank() {
        let rank = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let scaled = SimdPageRank::scale_rank(&rank.view(), 0.5);
        let expected = vec![0.5, 1.0, 1.5, 2.0];
        for (got, exp) in scaled.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-10, "Expected {exp}, got {got}");
        }
    }

    #[test]
    fn test_pagerank_normalize() {
        let rank = Array1::from_vec(vec![2.0, 3.0, 5.0]);
        let normalized = SimdPageRank::normalize_rank(&rank).expect("Normalization should succeed");
        let sum: f64 = normalized.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Normalized rank should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_pagerank_normalize_zero() {
        let rank = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let result = SimdPageRank::normalize_rank(&rank);
        assert!(result.is_err(), "Normalizing zero vector should fail");
    }

    #[test]
    fn test_pagerank_dangling_contribution() {
        let rank = Array1::from_vec(vec![0.2, 0.3, 0.5]);
        let is_dangling = vec![true, false, true];
        let contribution =
            SimdPageRank::dangling_node_contribution(&rank.view(), &is_dangling, 0.85);
        // dangling sum = 0.2 + 0.5 = 0.7
        // contribution = 0.85 * 0.7 / 3 = 0.1983...
        let expected = 0.85 * 0.7 / 3.0;
        assert!(
            (contribution - expected).abs() < 1e-10,
            "Expected {expected}, got {contribution}"
        );
    }

    #[test]
    fn test_pagerank_power_iteration() {
        // Simple 3-node cycle: 0 -> 1 -> 2 -> 0
        // Transition matrix (row-stochastic):
        // [0, 1, 0]
        // [0, 0, 1]
        // [1, 0, 0]
        let transition =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
                .expect("Test: failed to create transition matrix");

        let (rank, _iters) = SimdPageRank::compute_pagerank(&transition, 0.85, 1e-8, 200)
            .expect("PageRank should converge");

        // Symmetric cycle => all ranks equal
        let expected = 1.0 / 3.0;
        for (i, &r) in rank.iter().enumerate() {
            assert!(
                (r - expected).abs() < 0.01,
                "Node {i} rank {r} should be near {expected}"
            );
        }
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let transition = Array2::zeros((0, 0));
        let result = SimdPageRank::compute_pagerank(&transition, 0.85, 1e-6, 100);
        assert!(result.is_err(), "Empty graph should return error");
    }

    // ---- Spectral tests ----

    #[test]
    fn test_standard_laplacian() {
        // Simple path graph: 0 -- 1 -- 2
        // Adjacency:
        // [0, 1, 0]
        // [1, 0, 1]
        // [0, 1, 0]
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("Test: failed to create adjacency matrix");
        let degrees = Array1::from_vec(vec![1.0, 2.0, 1.0]);

        let lap = SimdSpectral::standard_laplacian(&adj, &degrees)
            .expect("Laplacian construction should succeed");

        // Expected Laplacian:
        // [ 1, -1,  0]
        // [-1,  2, -1]
        // [ 0, -1,  1]
        assert!((lap[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((lap[[0, 1]] - (-1.0)).abs() < 1e-10);
        assert!((lap[[0, 2]] - 0.0).abs() < 1e-10);
        assert!((lap[[1, 0]] - (-1.0)).abs() < 1e-10);
        assert!((lap[[1, 1]] - 2.0).abs() < 1e-10);
        assert!((lap[[1, 2]] - (-1.0)).abs() < 1e-10);
        assert!((lap[[2, 0]] - 0.0).abs() < 1e-10);
        assert!((lap[[2, 1]] - (-1.0)).abs() < 1e-10);
        assert!((lap[[2, 2]] - 1.0).abs() < 1e-10);

        // Laplacian property: row sums = 0
        for i in 0..3 {
            let row_sum: f64 = lap.row(i).sum();
            assert!(
                row_sum.abs() < 1e-10,
                "Laplacian row {i} sum should be 0, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_normalized_laplacian() {
        // Complete graph K3
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
            .expect("Test: failed to create adjacency matrix");
        let degrees = Array1::from_vec(vec![2.0, 2.0, 2.0]);

        let lap = SimdSpectral::normalized_laplacian(&adj, &degrees)
            .expect("Normalized Laplacian construction should succeed");

        // Diagonal should be 1.0 (for simple graphs with no self-loops)
        for i in 0..3 {
            assert!(
                (lap[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal element [{i},{i}] should be 1.0, got {}",
                lap[[i, i]]
            );
        }

        // Off-diagonal: -1/sqrt(2*2) = -0.5
        assert!(
            (lap[[0, 1]] - (-0.5)).abs() < 1e-10,
            "Off-diagonal [0,1] should be -0.5, got {}",
            lap[[0, 1]]
        );
    }

    #[test]
    fn test_random_walk_laplacian() {
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
            .expect("Test: failed to create adjacency matrix");
        let degrees = Array1::from_vec(vec![1.0, 2.0, 1.0]);

        let lap = SimdSpectral::random_walk_laplacian(&adj, &degrees)
            .expect("Random-walk Laplacian construction should succeed");

        // Diagonal: all 1.0
        for i in 0..3 {
            assert!(
                (lap[[i, i]] - 1.0).abs() < 1e-10,
                "Diagonal [{i},{i}] should be 1.0"
            );
        }

        // Row 0: degree=1, so off-diagonal = -1/1 * adj
        assert!((lap[[0, 1]] - (-1.0)).abs() < 1e-10);
        assert!((lap[[0, 2]] - 0.0).abs() < 1e-10);

        // Row 1: degree=2, so off-diagonal = -1/2 * adj
        assert!((lap[[1, 0]] - (-0.5)).abs() < 1e-10);
        assert!((lap[[1, 2]] - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_degree_inv_sqrt() {
        let degrees = Array1::from_vec(vec![4.0, 9.0, 0.0, 1.0]);
        let result = SimdSpectral::compute_degree_inv_sqrt(&degrees);

        assert!((result[0] - 0.5).abs() < 1e-10, "1/sqrt(4) = 0.5");
        assert!((result[1] - 1.0 / 3.0).abs() < 1e-10, "1/sqrt(9) = 1/3");
        assert!((result[2] - 0.0).abs() < 1e-10, "1/sqrt(0) => 0.0");
        assert!((result[3] - 1.0).abs() < 1e-10, "1/sqrt(1) = 1.0");
    }

    #[test]
    fn test_power_iteration() {
        // Symmetric 2x2 matrix with known eigenvalues
        // M = [[2, 1], [1, 2]], eigenvalues: 3, 1
        let matrix = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0])
            .expect("Test: failed to create matrix");
        let initial = Array1::from_vec(vec![1.0, 0.5]);

        let (eigenvalue, eigenvector, _iters) =
            SimdSpectral::power_iteration(&matrix, &initial, 100, 1e-10)
                .expect("Power iteration should converge");

        // Dominant eigenvalue should be 3.0
        assert!(
            (eigenvalue - 3.0).abs() < 0.01,
            "Eigenvalue should be near 3.0, got {eigenvalue}"
        );

        // Corresponding eigenvector should be proportional to [1, 1]
        let ratio = eigenvector[0] / eigenvector[1];
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "Eigenvector ratio should be near 1.0, got {ratio}"
        );
    }

    #[test]
    fn test_rayleigh_quotient() {
        // Identity matrix: Rayleigh quotient should be 1.0
        let identity = Array2::eye(3);
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let rq = SimdSpectral::rayleigh_quotient(&identity, &v.view())
            .expect("Rayleigh quotient should succeed");
        assert!(
            (rq - 1.0).abs() < 1e-10,
            "Rayleigh quotient of identity should be 1.0, got {rq}"
        );
    }

    #[test]
    fn test_gram_schmidt() {
        let mut vectors = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
            .expect("Test: failed to create matrix");

        SimdSpectral::gram_schmidt_orthogonalize(&mut vectors);

        // Check orthogonality
        let col0 = vectors.column(0);
        let col1 = vectors.column(1);
        let dot = f64::simd_dot(&col0, &col1);
        assert!(
            dot.abs() < 1e-10,
            "Columns should be orthogonal, dot product = {dot}"
        );

        // Check unit norms
        let norm0 = f64::simd_norm(&col0);
        let norm1 = f64::simd_norm(&col1);
        assert!(
            (norm0 - 1.0).abs() < 1e-10,
            "Column 0 norm should be 1.0, got {norm0}"
        );
        assert!(
            (norm1 - 1.0).abs() < 1e-10,
            "Column 1 norm should be 1.0, got {norm1}"
        );
    }

    // ---- Adjacency tests ----

    #[test]
    fn test_dense_matvec() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Test: failed to create matrix");
        let vector = Array1::from_vec(vec![1.0, 1.0, 1.0]);

        let result = SimdAdjacency::dense_matvec(&matrix, &vector).expect("Matvec should succeed");

        assert!((result[0] - 6.0).abs() < 1e-10, "Row 0 dot = 6.0");
        assert!((result[1] - 15.0).abs() < 1e-10, "Row 1 dot = 15.0");
    }

    #[test]
    fn test_dense_matvec_dimension_mismatch() {
        let matrix =
            Array2::from_shape_vec((2, 3), vec![1.0; 6]).expect("Test: failed to create matrix");
        let vector = Array1::from_vec(vec![1.0, 1.0]);

        let result = SimdAdjacency::dense_matvec(&matrix, &vector);
        assert!(result.is_err(), "Dimension mismatch should return error");
    }

    #[test]
    fn test_dense_matvec_scaled() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Test: failed to create matrix");
        let vector = Array1::from_vec(vec![1.0, 1.0]);

        let result = SimdAdjacency::dense_matvec_scaled(2.0, &matrix, &vector)
            .expect("Scaled matvec should succeed");

        assert!((result[0] - 6.0).abs() < 1e-10, "2*(1+2) = 6.0");
        assert!((result[1] - 14.0).abs() < 1e-10, "2*(3+4) = 14.0");
    }

    #[test]
    fn test_sparse_csr_matvec() {
        // Matrix:
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        let row_ptr = vec![0, 2, 3, 5];
        let col_idx = vec![0, 2, 1, 0, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = vec![1.0, 2.0, 3.0];

        let y = SimdAdjacency::sparse_csr_matvec(&row_ptr, &col_idx, &values, &x, 3)
            .expect("Sparse matvec should succeed");

        assert!((y[0] - 7.0).abs() < 1e-10, "1*1 + 2*3 = 7.0, got {}", y[0]);
        assert!((y[1] - 6.0).abs() < 1e-10, "3*2 = 6.0, got {}", y[1]);
        assert!(
            (y[2] - 19.0).abs() < 1e-10,
            "4*1 + 5*3 = 19.0, got {}",
            y[2]
        );
    }

    #[test]
    fn test_adjacency_to_transition() {
        // Adjacency: [[0,1,1],[1,0,0],[1,0,0]]
        let adj = Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            .expect("Test: failed to create adjacency matrix");

        let transition = SimdAdjacency::adjacency_to_transition(&adj)
            .expect("Transition matrix construction should succeed");

        // Row 0: sum=2, so [0, 0.5, 0.5]
        assert!((transition[[0, 0]] - 0.0).abs() < 1e-10);
        assert!((transition[[0, 1]] - 0.5).abs() < 1e-10);
        assert!((transition[[0, 2]] - 0.5).abs() < 1e-10);

        // Row 1: sum=1, so [1, 0, 0]
        assert!((transition[[1, 0]] - 1.0).abs() < 1e-10);
        assert!((transition[[1, 1]] - 0.0).abs() < 1e-10);

        // Row 2: sum=1, so [1, 0, 0]
        assert!((transition[[2, 0]] - 1.0).abs() < 1e-10);
        assert!((transition[[2, 2]] - 0.0).abs() < 1e-10);

        // All row sums should be 1.0
        for i in 0..3 {
            let row_sum: f64 = transition.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Transition row {i} sum should be 1.0, got {row_sum}"
            );
        }
    }

    #[test]
    fn test_weighted_matvec() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("Test: failed to create matrix");
        let vector = Array1::from_vec(vec![1.0, 1.0]);
        let weights = Array1::from_vec(vec![0.5, 2.0]);

        let result = SimdAdjacency::weighted_matvec(&matrix, &vector, &weights)
            .expect("Weighted matvec should succeed");

        // Row 0: 0.5 * (1+2) = 1.5
        // Row 1: 2.0 * (3+4) = 14.0
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 14.0).abs() < 1e-10);
    }

    // ---- Betweenness tests ----

    #[test]
    fn test_dependency_accumulation() {
        // Simple path: 0 -- 1 -- 2
        // BFS from 0: order = [0, 1, 2]
        // sigma = [1, 1, 1]
        // predecessors: [[], [0], [1]]
        let mut delta = vec![0.0, 0.0, 0.0];
        let sigma = vec![1.0, 1.0, 1.0];
        let predecessors = vec![vec![], vec![0], vec![1]];
        let order = vec![0, 1, 2];

        SimdBetweenness::accumulate_dependencies(&mut delta, &sigma, &predecessors, &order);

        // Node 2 has no successors in order, delta[2] = 0
        // Node 1: delta[1] += sigma[1]/sigma[2] * (1 + delta[2]) = 1
        // Node 0: delta[0] += sigma[0]/sigma[1] * (1 + delta[1]) = 2
        assert!(
            (delta[2] - 0.0).abs() < 1e-10,
            "delta[2] should be 0.0, got {}",
            delta[2]
        );
        assert!(
            (delta[1] - 1.0).abs() < 1e-10,
            "delta[1] should be 1.0, got {}",
            delta[1]
        );
        assert!(
            (delta[0] - 2.0).abs() < 1e-10,
            "delta[0] should be 2.0, got {}",
            delta[0]
        );
    }

    #[test]
    fn test_batch_update_centrality() {
        let mut centrality = vec![1.0, 2.0, 3.0];
        let delta = vec![0.5, 1.0, 1.5];

        SimdBetweenness::batch_update_centrality(&mut centrality, &delta, 1);

        // centrality += delta, then subtract delta[source]
        // [1.5, 3.0, 4.5] then centrality[1] -= 1.0 => [1.5, 2.0, 4.5]
        assert!((centrality[0] - 1.5).abs() < 1e-10);
        assert!((centrality[1] - 2.0).abs() < 1e-10);
        assert!((centrality[2] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_betweenness_normalize() {
        let mut centrality = vec![6.0, 12.0, 18.0];
        SimdBetweenness::normalize(&mut centrality, 4, false);

        // scale = 2.0 / (3 * 2) = 1/3
        let scale = 2.0 / 6.0;
        assert!((centrality[0] - 6.0 * scale).abs() < 1e-10);
        assert!((centrality[1] - 12.0 * scale).abs() < 1e-10);
        assert!((centrality[2] - 18.0 * scale).abs() < 1e-10);
    }

    // ---- Traversal tests ----

    #[test]
    fn test_init_distances() {
        let dist = SimdTraversal::init_distances(5, 2);
        assert_eq!(dist.len(), 5);
        assert!(dist[0].is_infinite());
        assert!(dist[1].is_infinite());
        assert!((dist[2] - 0.0).abs() < 1e-10);
        assert!(dist[3].is_infinite());
        assert!(dist[4].is_infinite());
    }

    #[test]
    fn test_init_distances_multi_source() {
        let dist = SimdTraversal::init_distances_multi_source(5, &[0, 3]);
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!(dist[1].is_infinite());
        assert!(dist[2].is_infinite());
        assert!((dist[3] - 0.0).abs() < 1e-10);
        assert!(dist[4].is_infinite());
    }

    #[test]
    fn test_eccentricity() {
        let dist = Array1::from_vec(vec![0.0, 1.0, 2.0, f64::INFINITY, 3.0]);
        let ecc = SimdTraversal::eccentricity(&dist.view());
        assert!(
            (ecc - 3.0).abs() < 1e-10,
            "Eccentricity should be 3.0, got {ecc}"
        );
    }

    #[test]
    fn test_count_reachable() {
        let dist = Array1::from_vec(vec![0.0, 1.0, f64::INFINITY, 2.0, f64::INFINITY]);
        let count = SimdTraversal::count_reachable(&dist.view());
        assert_eq!(count, 3, "Should have 3 reachable nodes");
    }

    #[test]
    fn test_sum_finite_distances() {
        let dist = Array1::from_vec(vec![0.0, 1.0, f64::INFINITY, 3.0, f64::INFINITY]);
        let sum = SimdTraversal::sum_finite_distances(&dist.view());
        assert!(
            (sum - 4.0).abs() < 1e-10,
            "Sum of finite distances should be 4.0, got {sum}"
        );
    }

    #[test]
    fn test_element_wise_min() {
        let a = Array1::from_vec(vec![1.0, 5.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 2.0, 3.0]);
        let result = SimdTraversal::element_wise_min(&a.view(), &b.view());
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_relax_neighbors() {
        let mut distances = Array1::from_vec(vec![0.0, f64::INFINITY, f64::INFINITY, 1.0]);
        let adj_row = Array1::from_vec(vec![0.0, 1.0, 1.0, 0.0]);

        let relaxed = SimdTraversal::relax_neighbors(&mut distances, &adj_row.view(), 0.0);

        assert_eq!(relaxed, vec![1, 2], "Nodes 1 and 2 should be relaxed");
        assert!((distances[1] - 1.0).abs() < 1e-10);
        assert!((distances[2] - 1.0).abs() < 1e-10);
        // Node 3 was already at distance 1.0, which is <= 1.0, so not relaxed
        assert!((distances[3] - 1.0).abs() < 1e-10);
    }

    // ---- Utility tests ----

    #[test]
    fn test_axpy() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let mut y = vec![4.0, 5.0, 6.0];

        SimdGraphUtils::axpy(2.0, &x.view(), &mut y);

        assert!((y[0] - 6.0).abs() < 1e-10, "4 + 2*1 = 6");
        assert!((y[1] - 9.0).abs() < 1e-10, "5 + 2*2 = 9");
        assert!((y[2] - 12.0).abs() < 1e-10, "6 + 2*3 = 12");
    }

    #[test]
    fn test_norms() {
        let v = Array1::from_vec(vec![3.0, -4.0]);

        let l2 = SimdGraphUtils::norm_l2(&v.view());
        assert!((l2 - 5.0).abs() < 1e-10, "L2 norm of [3,-4] = 5.0");

        let l1 = SimdGraphUtils::norm_l1(&v.view());
        assert!((l1 - 7.0).abs() < 1e-10, "L1 norm of [3,-4] = 7.0");

        let linf = SimdGraphUtils::norm_linf(&v.view());
        assert!((linf - 4.0).abs() < 1e-10, "Linf norm of [3,-4] = 4.0");
    }

    #[test]
    fn test_normalize_l2() {
        let v = Array1::from_vec(vec![3.0, 4.0]);
        let normalized =
            SimdGraphUtils::normalize_l2(&v.view()).expect("Normalization should succeed");

        let norm = f64::simd_norm(&normalized.view());
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "Normalized vector should have unit norm"
        );

        let zero = Array1::from_vec(vec![0.0, 0.0]);
        assert!(
            SimdGraphUtils::normalize_l2(&zero.view()).is_none(),
            "Zero vector should return None"
        );
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0]);
        let b = Array1::from_vec(vec![0.0, 1.0]);
        let sim = SimdGraphUtils::cosine_similarity(&a.view(), &b.view());
        assert!(sim.abs() < 1e-10, "Orthogonal vectors: cosine = 0");

        let c = Array1::from_vec(vec![1.0, 1.0]);
        let d = Array1::from_vec(vec![2.0, 2.0]);
        let sim2 = SimdGraphUtils::cosine_similarity(&c.view(), &d.view());
        assert!((sim2 - 1.0).abs() < 1e-10, "Parallel vectors: cosine = 1.0");
    }

    #[test]
    fn test_weighted_sum() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let result = SimdGraphUtils::weighted_sum(&a.view(), 2.0, &b.view(), 0.5);

        assert!((result[0] - 4.0).abs() < 1e-10, "2*1 + 0.5*4 = 4.0");
        assert!((result[1] - 6.5).abs() < 1e-10, "2*2 + 0.5*5 = 6.5");
        assert!((result[2] - 9.0).abs() < 1e-10, "2*3 + 0.5*6 = 9.0");
    }

    #[test]
    fn test_diameter_from_distance_matrix() {
        let dist_matrix =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0])
                .expect("Test: failed to create distance matrix");

        let diameter = SimdTraversal::diameter_from_distance_matrix(&dist_matrix.view());
        assert!(
            (diameter - 2.0).abs() < 1e-10,
            "Diameter should be 2.0, got {diameter}"
        );
    }
}
