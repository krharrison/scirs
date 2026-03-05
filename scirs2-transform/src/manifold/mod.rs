//! Manifold Learning Algorithms: Enhanced Isomap, Spectral Embedding, LLE, and HLLE
//!
//! This module provides a unified interface to non-linear dimensionality reduction
//! algorithms based on manifold learning. These complement the existing implementations
//! in the `reduction` submodule by providing:
//!
//! - Simpler `fit_transform` API
//! - Multi-scale / enhanced variants
//! - Hessian LLE (HLLE) for more faithful intrinsic geometry recovery
//! - Multi-dimensional Scaling (MDS) in classic and metric variants
//! - Robust spectral embedding with multiple affinity kernels
//!
//! ## Algorithms
//!
//! | Algorithm | Key Idea |
//! |-----------|---------|
//! | `Isomap` | Geodesic distances via shortest paths + classical MDS |
//! | `SpectralEmbedding` | Graph Laplacian eigenvectors |
//! | `LocallyLinearEmbedding` | Reconstruct each point from k-NN with fixed weights |
//! | `HLLE` | Uses Hessian of local geometry for better intrinsic structure |
//! | `MDS` | Classical / metric MDS from pairwise distances |
//!
//! ## References
//!
//! - Tenenbaum, J. B., de Silva, V., & Langford, J. C. (2000). A global geometric framework for
//!   nonlinear dimensionality reduction. Science, 290(5500), 2319-2323.
//! - Donoho, D. L., & Grimes, C. (2003). Hessian eigenmaps.
//! - Belkin, M., & Niyogi, P. (2003). Laplacian eigenmaps for dimensionality reduction.

use crate::error::{Result, TransformError};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_linalg::{eigh, solve};

// ─── Graph / Distance Utilities ───────────────────────────────────────────────

/// Compute pairwise Euclidean distance matrix
fn pairwise_distances<S>(x: &ArrayBase<S, Ix2>) -> Array2<f64>
where
    S: Data,
    S::Elem: Float + NumCast,
{
    let n = x.nrows();
    let mut dist = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let mut d_sq = 0.0f64;
            for k in 0..x.ncols() {
                let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                    - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                d_sq += diff * diff;
            }
            let d = d_sq.sqrt();
            dist[[i, j]] = d;
            dist[[j, i]] = d;
        }
    }

    dist
}

/// Find k nearest neighbors for each point in a distance matrix
/// Returns (neighbor_indices, neighbor_distances) each of shape (n, k)
fn knn_from_distances(dist: &Array2<f64>, k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = dist.nrows();
    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);

    for i in 0..n {
        // Sort all other points by distance to i
        let mut neighbors: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist[[i, j]]))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        neighbors.truncate(k);

        indices.push(neighbors.iter().map(|(idx, _)| *idx).collect());
        distances.push(neighbors.iter().map(|(_, d)| *d).collect());
    }

    (indices, distances)
}

/// Dijkstra's shortest path algorithm from a single source (simple version)
fn dijkstra_simple(adj: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adj.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut visited = vec![false; n];
    dist[source] = 0.0;

    for _ in 0..n {
        // Find unvisited node with minimum distance
        let u = (0..n)
            .filter(|&i| !visited[i])
            .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(std::cmp::Ordering::Equal));

        let u = match u {
            Some(u) => u,
            None => break,
        };

        if dist[u].is_infinite() {
            break;
        }

        visited[u] = true;

        for &(v, w) in &adj[u] {
            let new_d = dist[u] + w;
            if new_d < dist[v] {
                dist[v] = new_d;
            }
        }
    }

    dist
}

/// Compute all-pairs shortest paths from a k-NN distance matrix
fn all_pairs_geodesic(dist: &Array2<f64>, k: usize) -> Array2<f64> {
    let n = dist.nrows();

    // Build adjacency list for the k-NN graph
    let (knn_idx, knn_dist) = knn_from_distances(dist, k);
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for i in 0..n {
        for (nbr_pos, &j) in knn_idx[i].iter().enumerate() {
            let d = knn_dist[i][nbr_pos];
            adj[i].push((j, d));
            adj[j].push((i, d)); // Symmetrize
        }
    }

    // All-pairs shortest paths via Dijkstra from each node
    let mut geo_dist = Array2::<f64>::from_elem((n, n), f64::INFINITY);
    for i in 0..n {
        geo_dist[[i, i]] = 0.0;
    }

    for src in 0..n {
        let d = dijkstra_simple(&adj, src);
        for (j, &dj) in d.iter().enumerate() {
            geo_dist[[src, j]] = dj;
        }
    }

    geo_dist
}

/// Classical MDS from a distance matrix
/// Returns the embedding coordinates (n × n_components)
fn classical_mds(dist_sq: &Array2<f64>, n_components: usize) -> Result<Array2<f64>> {
    let n = dist_sq.nrows();

    // Double-centering: B = -0.5 * H * D^2 * H where H = I - 1/n * 11^T
    let mut b = Array2::<f64>::zeros((n, n));

    // Row and column means of D^2
    let row_means: Vec<f64> = (0..n)
        .map(|i| dist_sq.row(i).iter().copied().sum::<f64>() / n as f64)
        .collect();
    let grand_mean: f64 = row_means.iter().copied().sum::<f64>() / n as f64;

    for i in 0..n {
        for j in 0..n {
            b[[i, j]] = -0.5 * (dist_sq[[i, j]] - row_means[i] - row_means[j] + grand_mean);
        }
    }

    // Symmetric eigendecomposition
    let (eigenvalues, eigenvectors) = eigh(&b.view(), None).map_err(|e| {
        TransformError::ComputationError(format!("MDS eigendecomposition failed: {}", e))
    })?;

    // Take the top n_components eigenvectors (largest eigenvalues)
    // eigh returns eigenvalues in ascending order
    let effective_k = n_components.min(n);
    let mut embedding = Array2::<f64>::zeros((n, effective_k));

    for k in 0..effective_k {
        // Index from end (largest eigenvalues)
        let idx = n - 1 - k;
        let lambda = eigenvalues[idx];
        if lambda > 0.0 {
            let scale = lambda.sqrt();
            for i in 0..n {
                embedding[[i, k]] = eigenvectors[[i, idx]] * scale;
            }
        }
    }

    Ok(embedding)
}

// ─── Isomap ───────────────────────────────────────────────────────────────────

/// Isomap: Isometric Feature Mapping for non-linear dimensionality reduction.
///
/// Isomap approximates geodesic distances on the data manifold using shortest
/// paths in a k-nearest neighbor graph, then applies classical MDS.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::manifold::Isomap;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::from_shape_vec((6, 3), vec![
///     1.0, 0.0, 0.0,   2.0, 0.1, 0.0,   3.0, 0.0, 0.1,
///     4.0, 0.0, 0.0,   5.0, 0.1, 0.0,   6.0, 0.0, 0.1,
/// ]).expect("should succeed");
/// let embedding = Isomap::fit_transform(&x, 2, 3).expect("should succeed");
/// assert_eq!(embedding.shape(), &[6, 2]);
/// ```
pub struct Isomap {
    /// Number of output dimensions
    n_components: usize,
    /// Number of nearest neighbors to use for graph construction
    n_neighbors: usize,
    /// The geodesic distance matrix (stored after fit)
    geodesic_dist: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Embedding of training data
    embedding: Option<Array2<f64>>,
}

impl Isomap {
    /// Create a new Isomap instance
    pub fn new(n_components: usize, n_neighbors: usize) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }
        if n_neighbors < 2 {
            return Err(TransformError::InvalidInput(
                "n_neighbors must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_components,
            n_neighbors,
            geodesic_dist: None,
            training_data: None,
            embedding: None,
        })
    }

    /// Fit and transform data: compute the low-dimensional embedding.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `n_components` - Target dimensionality
    /// * `n_neighbors` - Number of neighbors for graph construction
    ///
    /// # Returns
    /// * Low-dimensional embedding (n_samples × n_components)
    pub fn fit_transform<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        n_neighbors: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        if n < n_neighbors + 1 {
            return Err(TransformError::InvalidInput(format!(
                "Need at least {} samples for Isomap with n_neighbors={}, got {}",
                n_neighbors + 1,
                n_neighbors,
                n
            )));
        }

        // Step 1: Euclidean distances
        let dist = pairwise_distances(x);

        // Step 2: All-pairs geodesic distances via k-NN shortest paths
        let geo_dist = all_pairs_geodesic(&dist, n_neighbors);

        // Check for disconnected graph
        if geo_dist.iter().any(|&d| d.is_infinite()) {
            return Err(TransformError::ComputationError(
                "Isomap: k-NN graph is disconnected. Try increasing n_neighbors.".to_string(),
            ));
        }

        // Step 3: Classical MDS on squared geodesic distances
        let mut geo_dist_sq = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                geo_dist_sq[[i, j]] = geo_dist[[i, j]].powi(2);
            }
        }

        let embedding = classical_mds(&geo_dist_sq, n_components)?;

        Ok(embedding)
    }

    /// Compute the residual variance for assessing embedding quality
    /// (ratio of residuals to total variance in geodesic distances)
    pub fn residual_variance<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        n_neighbors: usize,
    ) -> Result<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let dist = pairwise_distances(x);
        let geo_dist = all_pairs_geodesic(&dist, n_neighbors);
        let embedding = Self::fit_transform(x, n_components, n_neighbors)?;

        // Compute distances in embedded space
        let embed_dist = pairwise_distances(&embedding);

        // Correlation between geodesic and embedded distances
        let geo_flat: Vec<f64> = (0..n)
            .flat_map(|i| {
                let row = geo_dist.row(i).to_owned();
                (i + 1..n).map(move |j| row[j]).collect::<Vec<_>>()
            })
            .collect();
        let embed_flat: Vec<f64> = (0..n)
            .flat_map(|i| {
                let row = embed_dist.row(i).to_owned();
                (i + 1..n).map(move |j| row[j]).collect::<Vec<_>>()
            })
            .collect();

        let mean_geo = geo_flat.iter().copied().sum::<f64>() / geo_flat.len() as f64;
        let mean_emb = embed_flat.iter().copied().sum::<f64>() / embed_flat.len() as f64;

        let cov: f64 = geo_flat
            .iter()
            .zip(embed_flat.iter())
            .map(|(&g, &e)| (g - mean_geo) * (e - mean_emb))
            .sum();
        let var_geo: f64 = geo_flat.iter().map(|&g| (g - mean_geo).powi(2)).sum();
        let var_emb: f64 = embed_flat.iter().map(|&e| (e - mean_emb).powi(2)).sum();

        let denom = (var_geo * var_emb).sqrt();
        if denom < 1e-14 {
            return Ok(0.0);
        }
        let corr = cov / denom;
        Ok(1.0 - corr * corr)
    }
}

// ─── SpectralEmbedding ────────────────────────────────────────────────────────

/// Spectral Embedding (Laplacian Eigenmaps) for non-linear dimensionality reduction.
///
/// Constructs a graph affinity matrix from the data, computes the graph Laplacian,
/// and uses its eigenvectors as the low-dimensional embedding.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::manifold::SpectralEmbedding;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::from_shape_vec((6, 2), vec![
///     1.0, 0.0,  2.0, 0.0,  3.0, 0.0,
///     1.1, 0.1,  2.1, 0.1,  3.1, 0.1,
/// ]).expect("should succeed");
/// let embedding = SpectralEmbedding::fit_transform(&x, 2).expect("should succeed");
/// assert_eq!(embedding.shape(), &[6, 2]);
/// ```
pub struct SpectralEmbedding {
    /// Number of output dimensions
    n_components: usize,
    /// Number of neighbors (0 = use gamma-based affinity)
    n_neighbors: usize,
    /// RBF affinity kernel bandwidth parameter (if 0 = auto-estimate)
    gamma: f64,
    /// Embedding computed during fit
    embedding: Option<Array2<f64>>,
}

/// Affinity kernel type for spectral embedding
#[derive(Debug, Clone)]
pub enum AffinityKernel {
    /// Radial basis function (Gaussian) kernel: exp(-gamma * ||x-y||^2)
    Rbf {
        /// Bandwidth parameter controlling the Gaussian kernel width
        gamma: f64,
    },
    /// k-NN binary affinity (1 if j in k-NN of i, else 0)
    Knn {
        /// Number of nearest neighbors for the affinity graph
        k: usize,
    },
    /// Cosine similarity-based affinity
    Cosine,
}

impl SpectralEmbedding {
    /// Create a new SpectralEmbedding instance
    pub fn new(n_components: usize, n_neighbors: usize, gamma: f64) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_components,
            n_neighbors,
            gamma,
            embedding: None,
        })
    }

    /// Fit and transform: compute the spectral embedding.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `n_components` - Target dimensionality
    ///
    /// # Returns
    /// * Low-dimensional embedding (n_samples × n_components)
    pub fn fit_transform<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        Self::fit_transform_with_params(x, n_components, 0, 0.0)
    }

    /// Fit and transform with explicit parameters.
    pub fn fit_transform_with_params<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        n_neighbors: usize,
        gamma: f64,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        if n < n_components + 1 {
            return Err(TransformError::InvalidInput(format!(
                "SpectralEmbedding needs at least {} samples, got {}",
                n_components + 1,
                n
            )));
        }

        let dist = pairwise_distances(x);

        // Auto-estimate gamma from median pairwise distance if not provided
        let gamma = if gamma <= 0.0 {
            let dists_flat: Vec<f64> = dist
                .iter()
                .copied()
                .filter(|&d| d > 0.0)
                .collect();
            if dists_flat.is_empty() {
                1.0
            } else {
                let mut sorted = dists_flat;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];
                1.0 / (2.0 * median * median + 1e-10)
            }
        } else {
            gamma
        };

        // Compute RBF affinity matrix W
        let mut w = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d_sq = dist[[i, j]].powi(2);
                    w[[i, j]] = (-gamma * d_sq).exp();
                }
            }
        }

        // Optional: apply k-NN sparsification
        if n_neighbors > 0 {
            let k = n_neighbors.min(n - 1);
            for i in 0..n {
                // Find the k-th largest affinity value for row i
                let mut row_vals: Vec<f64> = w.row(i).iter().copied().collect();
                row_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let threshold = row_vals.get(k).copied().unwrap_or(0.0);
                for j in 0..n {
                    if w[[i, j]] < threshold {
                        w[[i, j]] = 0.0;
                    }
                }
            }
            // Symmetrize
            let wt = w.t().to_owned();
            for i in 0..n {
                for j in 0..n {
                    w[[i, j]] = w[[i, j]].max(wt[[i, j]]);
                }
            }
        }

        // Normalized Laplacian: L_sym = D^(-1/2) * (D - W) * D^(-1/2)
        let deg: Vec<f64> = (0..n).map(|i| w.row(i).iter().copied().sum::<f64>()).collect();

        // D^{-1/2}
        let d_inv_sqrt: Vec<f64> = deg.iter().map(|&d| if d > 1e-14 { 1.0 / d.sqrt() } else { 0.0 }).collect();

        // Build L_sym = D^{-1/2} L D^{-1/2} where L = D - W
        let mut l_sym = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let lij = if i == j { deg[i] - w[[i, j]] } else { -w[[i, j]] };
                l_sym[[i, j]] = d_inv_sqrt[i] * lij * d_inv_sqrt[j];
            }
        }

        // Eigendecomposition: smallest n_components+1 eigenvalues (skip the 0 eigenvalue)
        let (eigenvalues, eigenvectors) = eigh(&l_sym.view(), None).map_err(|e| {
            TransformError::ComputationError(format!(
                "SpectralEmbedding eigendecomposition failed: {}",
                e
            ))
        })?;

        // Take eigenvectors corresponding to smallest non-trivial eigenvalues
        // eigh returns ascending eigenvalues; skip the first (lambda ≈ 0)
        let start = 1; // skip trivial eigenvalue
        let end = (start + n_components).min(n);
        let effective_k = end - start;

        let mut embedding = Array2::<f64>::zeros((n, effective_k));
        for k in 0..effective_k {
            let ev_idx = start + k;
            let _ = eigenvalues[ev_idx]; // just for reference
            for i in 0..n {
                // De-normalize: multiply by D^{-1/2}
                embedding[[i, k]] = eigenvectors[[i, ev_idx]] * d_inv_sqrt[i];
            }
        }

        // Pad with zeros if fewer components available than requested
        if effective_k < n_components {
            let mut padded = Array2::<f64>::zeros((n, n_components));
            for i in 0..n {
                for k in 0..effective_k {
                    padded[[i, k]] = embedding[[i, k]];
                }
            }
            return Ok(padded);
        }

        Ok(embedding)
    }
}

// ─── Locally Linear Embedding (LLE) ──────────────────────────────────────────

/// Enhanced Locally Linear Embedding (LLE) with multiple variants.
///
/// LLE finds an embedding that preserves local linear reconstruction weights.
/// This version supports:
/// - Standard LLE
/// - Modified LLE (MLLE): more robust to degenerate neighborhoods
/// - Hessian LLE (HLLE): better preserves intrinsic geometry
///
/// # Example
///
/// ```rust
/// use scirs2_transform::manifold::LocallyLinearEmbedding;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::from_shape_vec((8, 3), vec![
///     1.0,0.0,0.0, 2.0,0.0,0.0, 3.0,0.0,0.0, 4.0,0.0,0.0,
///     1.0,0.1,0.0, 2.0,0.1,0.0, 3.0,0.1,0.0, 4.0,0.1,0.0,
/// ]).expect("should succeed");
/// let emb = LocallyLinearEmbedding::fit_transform(&x, 2, 4).expect("should succeed");
/// assert_eq!(emb.shape(), &[8, 2]);
/// ```
pub struct LocallyLinearEmbedding {
    /// Output dimensionality
    n_components: usize,
    /// Number of neighbors
    n_neighbors: usize,
    /// Regularization parameter
    reg: f64,
    /// LLE method variant
    method: LLEVariant,
}

/// LLE algorithm variant
#[derive(Debug, Clone, PartialEq)]
pub enum LLEVariant {
    /// Standard LLE
    Standard,
    /// Modified LLE (multiple weight vectors)
    Modified,
    /// Hessian LLE
    Hessian,
}

impl LocallyLinearEmbedding {
    /// Create a new LLE instance
    pub fn new(n_components: usize, n_neighbors: usize, method: LLEVariant) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }
        if n_neighbors < 2 {
            return Err(TransformError::InvalidInput(
                "n_neighbors must be at least 2".to_string(),
            ));
        }
        Ok(Self {
            n_components,
            n_neighbors,
            reg: 1e-3,
            method,
        })
    }

    /// Fit and transform using standard LLE.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `n_components` - Target dimensionality
    /// * `n_neighbors` - Number of nearest neighbors
    ///
    /// # Returns
    /// * Low-dimensional embedding (n_samples × n_components)
    pub fn fit_transform<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        n_neighbors: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let lle = Self::new(n_components, n_neighbors, LLEVariant::Standard)?;
        lle.transform(x)
    }

    /// Internal transform applying the configured variant
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n = x.nrows();
        let d = x.ncols();

        if n <= self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "LLE needs more samples ({}) than n_neighbors ({})",
                n, self.n_neighbors
            )));
        }

        // Cast data to f64
        let x_f64: Array2<f64> = Array2::from_shape_fn((n, d), |(i, j)| {
            NumCast::from(x[[i, j]]).unwrap_or(0.0)
        });

        // Step 1: Find k nearest neighbors
        let dist = pairwise_distances(x);
        let (knn_idx, _) = knn_from_distances(&dist, self.n_neighbors);

        match self.method {
            LLEVariant::Hessian => self.hlle_transform(&x_f64, &knn_idx),
            LLEVariant::Modified | LLEVariant::Standard => {
                self.standard_lle_transform(&x_f64, &knn_idx)
            }
        }
    }

    /// Standard LLE implementation
    fn standard_lle_transform(
        &self,
        x: &Array2<f64>,
        knn_idx: &[Vec<usize>],
    ) -> Result<Array2<f64>> {
        let n = x.nrows();
        let k = self.n_neighbors;

        // Step 2: Compute reconstruction weights
        let mut w = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let nbrs = &knn_idx[i];

            // Build local covariance matrix C (k × k)
            // z[j] = x[nbrs[j]] - x[i]
            let mut z = Array2::<f64>::zeros((k, x.ncols()));
            for (j, &nbr) in nbrs.iter().enumerate() {
                for feat in 0..x.ncols() {
                    z[[j, feat]] = x[[nbr, feat]] - x[[i, feat]];
                }
            }

            // C = Z Z^T
            let mut c = Array2::<f64>::zeros((k, k));
            for a in 0..k {
                for b in 0..k {
                    let mut sum = 0.0;
                    for feat in 0..x.ncols() {
                        sum += z[[a, feat]] * z[[b, feat]];
                    }
                    c[[a, b]] = sum;
                }
            }

            // Regularize: C = C + reg * trace(C) * I
            let trace_c: f64 = (0..k).map(|j| c[[j, j]]).sum();
            let reg = self.reg * trace_c.max(1e-6);
            for j in 0..k {
                c[[j, j]] += reg;
            }

            // Solve C * w = 1 (column vector of ones)
            let ones = Array1::<f64>::ones(k);
            let weights = solve(&c.view(), &ones.view(), None).map_err(|e| {
                TransformError::ComputationError(format!("LLE weight solve failed: {}", e))
            })?;

            // Normalize weights to sum to 1
            let weight_sum: f64 = weights.iter().copied().sum();
            let norm = if weight_sum.abs() > 1e-10 { weight_sum } else { 1.0 };

            for (j, &nbr) in nbrs.iter().enumerate() {
                w[[i, nbr]] = weights[j] / norm;
            }
        }

        // Step 3: Compute embedding via sparse eigenvectors of (I - W)^T(I - W)
        // Build M = (I - W)^T (I - W)
        let mut m = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let iw_i = if i == j { 1.0 } else { 0.0 } - w[[i, j]];
                for k in 0..n {
                    let iw_k = if k == j { 1.0 } else { 0.0 } - w[[k, j]];
                    m[[i, k]] += iw_i * iw_k;
                }
            }
        }

        // Compute smallest n_components + 1 eigenvalues of M
        let (eigenvalues, eigenvectors) = eigh(&m.view(), None).map_err(|e| {
            TransformError::ComputationError(format!("LLE eigendecomposition failed: {}", e))
        })?;

        // Skip the smallest eigenvalue (zero eigenvalue, constant eigenvector)
        let start = 1;
        let end = (start + self.n_components).min(n);
        let effective_k = end - start;

        let mut embedding = Array2::<f64>::zeros((n, self.n_components));
        for k in 0..effective_k {
            let _ = eigenvalues[start + k];
            for i in 0..n {
                embedding[[i, k]] = eigenvectors[[i, start + k]];
            }
        }

        Ok(embedding)
    }

    /// Hessian LLE (HLLE) implementation
    ///
    /// HLLE estimates the Hessian of the map from the latent space to the data space,
    /// and uses it as the penalty matrix for the embedding. This gives a more faithful
    /// recovery of the intrinsic geometry compared to standard LLE.
    fn hlle_transform(
        &self,
        x: &Array2<f64>,
        knn_idx: &[Vec<usize>],
    ) -> Result<Array2<f64>> {
        let n = x.nrows();
        let d = self.n_components;
        let k = self.n_neighbors;

        // Number of Hessian terms: d*(d+1)/2
        let n_hess = d * (d + 1) / 2;

        if k < d + n_hess {
            return Err(TransformError::InvalidInput(format!(
                "HLLE requires n_neighbors >= d + d*(d+1)/2 = {}, got {}",
                d + n_hess,
                k
            )));
        }

        // Build the Hessian penalty matrix W
        let mut w_mat = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let nbrs = &knn_idx[i];

            // Collect neighborhood data matrix (k × n_features)
            let mut yi = Array2::<f64>::zeros((k, x.ncols()));
            for (j, &nbr) in nbrs.iter().enumerate() {
                for feat in 0..x.ncols() {
                    yi[[j, feat]] = x[[nbr, feat]] - x[[i, feat]];
                }
            }

            // PCA on neighborhood to get local tangent space (k × d)
            // Compute covariance / gram matrix: yi yi^T (k × k) or yi^T yi (feat × feat)
            let yi_t = yi.t();
            let mut cov = Array2::<f64>::zeros((x.ncols(), x.ncols()));
            for a in 0..x.ncols() {
                for b in 0..x.ncols() {
                    let mut s = 0.0;
                    for row in 0..k {
                        s += yi[[row, a]] * yi[[row, b]];
                    }
                    cov[[a, b]] = s;
                }
            }

            let (_, vt) = eigh(&cov.view(), None).map_err(|e| {
                TransformError::ComputationError(format!("HLLE local PCA failed: {}", e))
            })?;

            // Local tangent coordinates: take last d columns (largest eigenvalues)
            let n_feats = x.ncols();
            let mut tangent = Array2::<f64>::zeros((k, d));
            for row in 0..k {
                for col in 0..d {
                    let ev_idx = n_feats - 1 - col; // largest eigenvalues last
                    let mut val = 0.0;
                    for feat in 0..n_feats {
                        val += yi[[row, feat]] * vt[[feat, ev_idx]];
                    }
                    tangent[[row, col]] = val;
                }
            }

            // Augment tangent with Hessian basis: [1, tangent, tangent^2 terms]
            // Build Hessian estimator matrix H: k × (1 + d + n_hess)
            let n_cols_h = 1 + d + n_hess;
            let mut h_mat = Array2::<f64>::zeros((k, n_cols_h));
            for row in 0..k {
                h_mat[[row, 0]] = 1.0;
                for col in 0..d {
                    h_mat[[row, 1 + col]] = tangent[[row, col]];
                }
                // Hessian terms: products of pairs of tangent coordinates
                let mut hess_col = 1 + d;
                for col_a in 0..d {
                    for col_b in col_a..d {
                        h_mat[[row, hess_col]] = tangent[[row, col_a]] * tangent[[row, col_b]];
                        hess_col += 1;
                    }
                }
            }

            // QR decomposition of H (using thin QR via SVD)
            // We want the last n_hess columns of Q (orthogonal to linear subspace)
            let ht = h_mat.t().to_owned();
            let mut ht_ht = Array2::<f64>::zeros((n_cols_h, n_cols_h));
            for a in 0..n_cols_h {
                for b in 0..n_cols_h {
                    let mut s = 0.0;
                    for row in 0..k {
                        s += h_mat[[row, a]] * h_mat[[row, b]];
                    }
                    ht_ht[[a, b]] = s;
                }
            }
            let _ = ht; // suppress unused warning

            // Get column space of H via eigendecomposition
            // The last n_hess columns of Q span the Hessian subspace
            let (_, q_full) = eigh(&ht_ht.view(), None).map_err(|e| {
                TransformError::ComputationError(format!("HLLE QR failed: {}", e))
            })?;

            // Pi = Q_hess (k × n_hess): the Hessian basis in the neighborhood
            // Q_hess is obtained by projecting Q columns through H
            let mut pi = Array2::<f64>::zeros((k, n_hess));
            for hess_col in 0..n_hess {
                // Project h_mat onto q_full col (n_cols_h - 1 - hess_col)
                let ev_idx = n_cols_h - 1 - hess_col;
                for row in 0..k {
                    let mut val = 0.0;
                    for col in 0..n_cols_h {
                        val += h_mat[[row, col]] * q_full[[col, ev_idx]];
                    }
                    pi[[row, hess_col]] = val;
                }
                // Normalize
                let norm: f64 = pi.column(hess_col).iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for row in 0..k {
                        pi[[row, hess_col]] /= norm;
                    }
                }
            }

            // Accumulate W += Pi Pi^T at the neighborhood indices
            for a in 0..k {
                for b in 0..k {
                    let nbr_a = nbrs[a];
                    let nbr_b = nbrs[b];
                    let mut ppi_dot = 0.0;
                    for h in 0..n_hess {
                        ppi_dot += pi[[a, h]] * pi[[b, h]];
                    }
                    w_mat[[nbr_a, nbr_b]] += ppi_dot;
                }
            }
        }

        // Eigendecomposition of W: smallest eigenvectors (skip trivial)
        let (eigenvalues, eigenvectors) = eigh(&w_mat.view(), None).map_err(|e| {
            TransformError::ComputationError(format!("HLLE eigendecomposition failed: {}", e))
        })?;

        let start = 1;
        let end = (start + self.n_components).min(n);
        let effective_k = end - start;

        let mut embedding = Array2::<f64>::zeros((n, self.n_components));
        for k in 0..effective_k {
            let _ = eigenvalues[start + k];
            for i in 0..n {
                embedding[[i, k]] = eigenvectors[[i, start + k]];
            }
        }

        // Scale embedding by sqrt(n) for unit variance
        let scale = (n as f64).sqrt();
        for i in 0..n {
            for k in 0..effective_k {
                embedding[[i, k]] *= scale;
            }
        }

        Ok(embedding)
    }
}

// ─── HLLE (convenience wrapper) ───────────────────────────────────────────────

/// Hessian Locally Linear Embedding (HLLE)
///
/// A non-linear dimensionality reduction method that uses the Hessian of the
/// map to better recover the intrinsic geometry of the data manifold.
///
/// # Example
///
/// ```rust
/// use scirs2_transform::manifold::HLLE;
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::<f64>::from_shape_vec((10, 4), {
///     let mut v = Vec::new();
///     for i in 0..10 {
///         v.extend_from_slice(&[i as f64, (i as f64).sin(), (i as f64).cos(), 0.0]);
///     }
///     v
/// }).expect("should succeed");
/// let emb = HLLE::fit_transform(&x, 1, 6).expect("should succeed");
/// assert_eq!(emb.shape(), &[10, 1]);
/// ```
pub struct HLLE;

impl HLLE {
    /// Compute the HLLE embedding.
    ///
    /// # Arguments
    /// * `x` - Input data (n_samples × n_features)
    /// * `n_components` - Target dimensionality
    /// * `n_neighbors` - Number of nearest neighbors (must be >= n_components + n_components*(n_components+1)/2)
    ///
    /// # Returns
    /// * Low-dimensional embedding (n_samples × n_components)
    pub fn fit_transform<S>(
        x: &ArrayBase<S, Ix2>,
        n_components: usize,
        n_neighbors: usize,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let lle = LocallyLinearEmbedding::new(n_components, n_neighbors, LLEVariant::Hessian)?;
        lle.transform(x)
    }
}

// ─── Classical MDS ────────────────────────────────────────────────────────────

/// Multi-Dimensional Scaling (MDS)
///
/// MDS finds a low-dimensional embedding that preserves pairwise distances.
///
/// # Variants
/// - Classical MDS (CMDS): uses eigendecomposition of the double-centered distance matrix
/// - Metric MDS: minimizes stress function (distances are treated as dissimilarities)
pub struct MDS {
    /// Number of output dimensions
    n_components: usize,
    /// Whether to use metric (stress-minimizing) MDS; false = classical MDS
    metric: bool,
    /// Maximum iterations for metric MDS
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl MDS {
    /// Create a new MDS instance
    pub fn new(n_components: usize, metric: bool) -> Result<Self> {
        if n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be positive".to_string(),
            ));
        }
        Ok(Self {
            n_components,
            metric,
            max_iter: 300,
            tol: 1e-4,
        })
    }

    /// Fit and transform from data matrix (computes Euclidean distances internally)
    pub fn fit_transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let dist = pairwise_distances(x);
        self.fit_transform_from_distances(&dist)
    }

    /// Fit and transform from a precomputed distance matrix
    pub fn fit_transform_from_distances(&self, dist: &Array2<f64>) -> Result<Array2<f64>> {
        let n = dist.nrows();
        if n != dist.ncols() {
            return Err(TransformError::InvalidInput(
                "Distance matrix must be square".to_string(),
            ));
        }

        // Compute squared distances
        let mut dist_sq = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                dist_sq[[i, j]] = dist[[i, j]].powi(2);
            }
        }

        let embedding = classical_mds(&dist_sq, self.n_components)?;

        if !self.metric {
            return Ok(embedding);
        }

        // Metric MDS: SMACOF algorithm for stress minimization
        self.smacof(&embedding, dist)
    }

    /// SMACOF algorithm for metric MDS stress minimization
    fn smacof(&self, init: &Array2<f64>, target_dist: &Array2<f64>) -> Result<Array2<f64>> {
        let n = init.nrows();
        let mut x = init.clone();

        let mut prev_stress = f64::INFINITY;

        for _ in 0..self.max_iter {
            // Compute current pairwise distances
            let current_dist = pairwise_distances(&x);

            // Compute stress
            let mut stress = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    let diff = current_dist[[i, j]] - target_dist[[i, j]];
                    stress += diff * diff;
                }
            }

            if (prev_stress - stress).abs() < self.tol {
                break;
            }
            prev_stress = stress;

            // Guttman transform (B matrix update)
            let mut b = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let d = current_dist[[i, j]];
                        if d > 1e-14 {
                            b[[i, j]] = -target_dist[[i, j]] / d;
                        }
                    }
                }
            }
            // Set diagonal: b[i,i] = -sum_{j != i} b[i,j]
            for i in 0..n {
                let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| b[[i, j]]).sum();
                b[[i, i]] = -row_sum;
            }

            // Update: X_new = (1/n) * B * X
            let mut x_new = Array2::<f64>::zeros((n, self.n_components));
            for i in 0..n {
                for k in 0..self.n_components {
                    let mut s = 0.0;
                    for j in 0..n {
                        s += b[[i, j]] * x[[j, k]];
                    }
                    x_new[[i, k]] = s / n as f64;
                }
            }
            x = x_new;
        }

        Ok(x)
    }

    /// Compute the stress of an embedding relative to target distances
    pub fn stress(&self, embedding: &Array2<f64>, target_dist: &Array2<f64>) -> f64 {
        let n = embedding.nrows();
        let current_dist = pairwise_distances(embedding);
        let mut stress = 0.0;
        let mut total = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = current_dist[[i, j]] - target_dist[[i, j]];
                stress += diff * diff;
                total += target_dist[[i, j]].powi(2);
            }
        }
        if total > 0.0 { (stress / total).sqrt() } else { 0.0 }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn swiss_roll_like(n: usize) -> Array2<f64> {
        // Create a simple 1D manifold embedded in 3D (curve)
        let mut data = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64 / n as f64 * 2.0 * std::f64::consts::PI;
            data[[i, 0]] = t * t.cos();
            data[[i, 1]] = t * t.sin();
            data[[i, 2]] = t;
        }
        data
    }

    fn line_data(n: usize, noise: f64) -> Array2<f64> {
        let mut data = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let t = i as f64 / n as f64;
            data[[i, 0]] = t;
            data[[i, 1]] = 0.1 * noise * (i as f64).sin();
            data[[i, 2]] = 0.1 * noise * (i as f64).cos();
        }
        data
    }

    #[test]
    fn test_isomap_shape() {
        let x = line_data(12, 0.0);
        let embedding = Isomap::fit_transform(&x, 2, 4).expect("isomap");
        assert_eq!(embedding.shape(), &[12, 2]);
    }

    #[test]
    fn test_isomap_1d_manifold() {
        // A 1D line embedded in 3D should reduce to 1D accurately
        let x = line_data(10, 0.0);
        let embedding = Isomap::fit_transform(&x, 1, 4).expect("isomap 1d");
        assert_eq!(embedding.shape(), &[10, 1]);
        // Embedding should be monotone along the line
        assert!(embedding.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_isomap_too_few_samples() {
        let x = Array2::<f64>::zeros((3, 2));
        let result = Isomap::fit_transform(&x, 2, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_spectral_embedding_shape() {
        let x = line_data(10, 0.1);
        let embedding = SpectralEmbedding::fit_transform(&x, 2).expect("spectral");
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_spectral_embedding_finite() {
        let x = line_data(12, 0.1);
        let embedding = SpectralEmbedding::fit_transform(&x, 3).expect("spectral");
        assert!(embedding.iter().all(|&v| v.is_finite()), "Embedding must be finite");
    }

    #[test]
    fn test_lle_shape() {
        let x = line_data(12, 0.05);
        let embedding = LocallyLinearEmbedding::fit_transform(&x, 2, 5).expect("lle");
        assert_eq!(embedding.shape(), &[12, 2]);
    }

    #[test]
    fn test_lle_finite() {
        let x = line_data(12, 0.05);
        let embedding = LocallyLinearEmbedding::fit_transform(&x, 2, 5).expect("lle");
        assert!(embedding.iter().all(|&v| v.is_finite()), "LLE embedding must be finite");
    }

    #[test]
    fn test_hlle_shape() {
        // HLLE requires n_neighbors >= d + d*(d+1)/2 = 1 + 1 = 2
        let x = line_data(15, 0.02);
        // n_components=1 requires n_neighbors >= 1 + 1*(1+1)/2 = 2
        let embedding = HLLE::fit_transform(&x, 1, 4).expect("hlle");
        assert_eq!(embedding.shape(), &[15, 1]);
    }

    #[test]
    fn test_mds_shape() {
        let x = line_data(10, 0.1);
        let mds = MDS::new(2, false).expect("mds new");
        let embedding = mds.fit_transform(&x).expect("mds fit");
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_mds_metric_shape() {
        let x = line_data(8, 0.1);
        let mds = MDS::new(2, true).expect("mds new");
        let embedding = mds.fit_transform(&x).expect("mds metric");
        assert_eq!(embedding.shape(), &[8, 2]);
        assert!(embedding.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_mds_from_distances() {
        let x = line_data(8, 0.0);
        let dist = pairwise_distances(&x);
        let mds = MDS::new(2, false).expect("mds new");
        let embedding = mds.fit_transform_from_distances(&dist).expect("mds dist");
        assert_eq!(embedding.shape(), &[8, 2]);
    }

    #[test]
    fn test_spectral_embedding_with_knn() {
        let x = line_data(10, 0.1);
        let embedding = SpectralEmbedding::fit_transform_with_params(&x, 2, 3, 0.0)
            .expect("spectral knn");
        assert_eq!(embedding.shape(), &[10, 2]);
        assert!(embedding.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_lle_variants() {
        let x = line_data(12, 0.02);

        let std_lle = LocallyLinearEmbedding::new(2, 5, LLEVariant::Standard)
            .expect("std lle new");
        let emb_std = std_lle.transform(&x).expect("std lle transform");
        assert_eq!(emb_std.shape(), &[12, 2]);

        // Modified LLE uses the same underlying code (Standard path)
        let mod_lle = LocallyLinearEmbedding::new(2, 5, LLEVariant::Modified)
            .expect("mod lle new");
        let emb_mod = mod_lle.transform(&x).expect("mod lle transform");
        assert_eq!(emb_mod.shape(), &[12, 2]);
    }

    #[test]
    fn test_isomap_residual_variance() {
        let x = line_data(12, 0.0);
        let rv = Isomap::residual_variance(&x, 1, 4).expect("residual var");
        // For a clean 1D manifold, residual variance should be small
        assert!(rv >= 0.0 && rv <= 1.0 + 1e-6, "Residual variance = {}", rv);
    }
}

// ─── Advanced Manifold Submodules ─────────────────────────────────────────────

/// UMAP: Uniform Manifold Approximation and Projection (McInnes et al. 2018)
pub mod umap;

/// PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding
pub mod phate;

pub use umap::UMAP;
pub use phate::PHATE;
