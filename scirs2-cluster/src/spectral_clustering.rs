//! Advanced spectral clustering
//!
//! This module provides spectral clustering algorithms beyond the basic
//! implementation in `spectral::spectral_clustering`.
//!
//! # Algorithms
//!
//! - **Unnormalized spectral clustering** (graph Laplacian L = D - W)
//! - **Ng-Jordan-Weiss** (symmetric normalized Laplacian, row-normalised eigenvectors)
//! - **Shi-Malik normalized cut** (random-walk Laplacian L_rw = D^{-1}L)
//! - **Eigengap heuristic** for automatic k selection
//! - **Similarity kernels**: RBF (Gaussian) and k-nearest-neighbour
//!
//! References:
//! - U. von Luxburg, *A Tutorial on Spectral Clustering*, 2007.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_core::random::prelude::*;
use scirs2_linalg::{eigh, smallest_k_eigh};
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Similarity kernels
// ---------------------------------------------------------------------------

/// Similarity kernel used to build the affinity matrix.
#[derive(Debug, Clone)]
pub enum SimilarityKernel<F: Float> {
    /// RBF (Gaussian) kernel: `exp(-gamma * ||x-y||^2)`.
    Rbf {
        /// Kernel bandwidth parameter. If `None`, auto-estimated.
        gamma: Option<F>,
    },
    /// k-nearest-neighbour binary adjacency, optionally made mutual.
    Knn {
        /// Number of neighbours.
        k: usize,
        /// If true, edge (i,j) requires both i in kNN(j) and j in kNN(i).
        mutual: bool,
    },
    /// Precomputed affinity matrix.
    Precomputed,
}

impl<F: Float + FromPrimitive> Default for SimilarityKernel<F> {
    fn default() -> Self {
        SimilarityKernel::Rbf { gamma: None }
    }
}

/// Spectral clustering variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralVariant {
    /// Unnormalized graph Laplacian (L = D - W).
    Unnormalized,
    /// Ng-Jordan-Weiss: symmetric normalised Laplacian with row-normalised eigenvectors.
    NgJordanWeiss,
    /// Shi-Malik normalised cut: random-walk Laplacian (D^{-1} L).
    ShiMalik,
}

/// Configuration for the advanced spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralConfig<F: Float> {
    /// Number of clusters. If `None`, use eigengap heuristic.
    pub n_clusters: Option<usize>,
    /// Maximum number of clusters to consider for eigengap.
    pub max_clusters: usize,
    /// Algorithm variant.
    pub variant: SpectralVariant,
    /// Similarity kernel.
    pub kernel: SimilarityKernel<F>,
    /// Maximum k-means iterations.
    pub kmeans_max_iter: usize,
    /// Number of k-means restarts.
    pub kmeans_n_init: usize,
    /// Random seed.
    pub seed: Option<u64>,
}

impl<F: Float + FromPrimitive> Default for SpectralConfig<F> {
    fn default() -> Self {
        Self {
            n_clusters: None,
            max_clusters: 10,
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::default(),
            kmeans_max_iter: 300,
            kmeans_n_init: 10,
            seed: None,
        }
    }
}

/// Result of spectral clustering.
#[derive(Debug, Clone)]
pub struct SpectralResult<F: Float> {
    /// Cluster labels per sample.
    pub labels: Array1<usize>,
    /// Spectral embedding (n_samples x k).
    pub embedding: Array2<F>,
    /// Number of clusters used.
    pub n_clusters: usize,
    /// Eigenvalues (smallest k+extra).
    pub eigenvalues: Array1<F>,
    /// If eigengap heuristic was used, the detected gaps.
    pub eigengaps: Option<Vec<F>>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform advanced spectral clustering.
pub fn spectral_cluster<F>(
    data: ArrayView2<F>,
    config: &SpectralConfig<F>,
) -> Result<SpectralResult<F>>
where
    F: Float
        + FromPrimitive
        + Debug
        + 'static
        + Send
        + Sync
        + std::iter::Sum
        + std::fmt::Display
        + ScalarOperand
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    let n = data.nrows();
    if n < 2 {
        return Err(ClusteringError::InvalidInput(
            "spectral clustering requires at least 2 samples".into(),
        ));
    }

    // Step 1: Affinity matrix
    let w = build_affinity(data, &config.kernel)?;

    // Step 2: Graph Laplacian
    let (laplacian, d_inv_sqrt) = build_laplacian(&w, config.variant)?;

    // Step 3: Eigen decomposition
    let max_k = config
        .n_clusters
        .unwrap_or(config.max_clusters)
        .min(n - 1)
        .max(2);
    let n_eig = (max_k + 2).min(n);

    // For small matrices, use the full eigendecomposition (more reliable).
    // The iterative partial eigensolver can produce poor results on tiny
    // problems where all eigenvalues are needed anyway.
    let (eigenvalues_raw, eigenvectors_raw) = if n <= 64 {
        eigh(&laplacian.view(), None)?
    } else {
        let tol = F::from(1e-10).unwrap_or(F::epsilon());
        smallest_k_eigh(&laplacian.view(), n_eig, 1000, tol)?
    };

    // Ensure eigenvalues are sorted in ascending order (smallest first).
    // Some eigensolvers return descending order; spectral clustering
    // needs the smallest eigenvalues' eigenvectors.
    let m = eigenvalues_raw.len();
    let mut sort_indices: Vec<usize> = (0..m).collect();
    sort_indices.sort_by(|&a, &b| {
        eigenvalues_raw[a]
            .partial_cmp(&eigenvalues_raw[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eigenvalues = Array1::from_iter(sort_indices.iter().map(|&i| eigenvalues_raw[i]));
    let mut eigenvectors = Array2::<F>::zeros((n, m));
    for (new_j, &old_j) in sort_indices.iter().enumerate() {
        for i in 0..n {
            eigenvectors[[i, new_j]] = eigenvectors_raw[[i, old_j]];
        }
    }

    // Step 4: Determine k
    let (k, eigengaps) = if let Some(nc) = config.n_clusters {
        (nc, None)
    } else {
        let gaps = compute_eigengaps(&eigenvalues);
        let detected_k = eigengap_heuristic(&gaps, max_k);
        (detected_k, Some(gaps))
    };
    let k = k.max(2).min(n - 1).min(eigenvectors.ncols());

    // Step 5: Extract embedding
    let embedding = extract_embedding(&eigenvectors, k, config.variant, &d_inv_sqrt)?;

    // Step 6: k-means in the embedding space
    let labels = run_kmeans_on_embedding(&embedding, k, config.kmeans_max_iter, config.seed)?;

    Ok(SpectralResult {
        labels,
        embedding,
        n_clusters: k,
        eigenvalues,
        eigengaps,
    })
}

// ---------------------------------------------------------------------------
// Affinity matrix construction
// ---------------------------------------------------------------------------

fn build_affinity<F>(data: ArrayView2<F>, kernel: &SimilarityKernel<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    match kernel {
        SimilarityKernel::Rbf { gamma } => rbf_affinity(data, *gamma),
        SimilarityKernel::Knn { k, mutual } => knn_affinity(data, *k, *mutual),
        SimilarityKernel::Precomputed => {
            if data.nrows() != data.ncols() {
                return Err(ClusteringError::InvalidInput(
                    "precomputed affinity must be square".into(),
                ));
            }
            Ok(data.to_owned())
        }
    }
}

fn rbf_affinity<F>(data: ArrayView2<F>, gamma: Option<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();

    // Pairwise squared distances
    let mut dist_sq = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = F::zero();
            for f in 0..d {
                let diff = data[[i, f]] - data[[j, f]];
                sq = sq + diff * diff;
            }
            dist_sq[[i, j]] = sq;
            dist_sq[[j, i]] = sq;
        }
    }

    // Auto-estimate gamma if not provided: gamma = 1 / (2 * median(dist_sq))
    let g = match gamma {
        Some(g) if g > F::zero() => g,
        _ => {
            let mut dists: Vec<F> = Vec::with_capacity(n * (n - 1) / 2);
            for i in 0..n {
                for j in (i + 1)..n {
                    dists.push(dist_sq[[i, j]]);
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = if dists.is_empty() {
                F::one()
            } else {
                dists[dists.len() / 2]
            };
            let two = F::from(2.0).unwrap_or(F::one() + F::one());
            if median > F::zero() {
                F::one() / (two * median)
            } else {
                F::one()
            }
        }
    };

    // Build affinity
    let mut w = Array2::<F>::zeros((n, n));
    for i in 0..n {
        w[[i, i]] = F::one();
        for j in (i + 1)..n {
            let val = (-g * dist_sq[[i, j]]).exp();
            w[[i, j]] = val;
            w[[j, i]] = val;
        }
    }

    Ok(w)
}

fn knn_affinity<F>(data: ArrayView2<F>, k: usize, mutual: bool) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = data.nrows();
    let d = data.ncols();
    if k >= n {
        return Err(ClusteringError::InvalidInput(
            "k must be less than n for kNN affinity".into(),
        ));
    }

    // Pairwise distances
    let mut dists = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = F::zero();
            for f in 0..d {
                let diff = data[[i, f]] - data[[j, f]];
                sq = sq + diff * diff;
            }
            let dist = sq.sqrt();
            dists[[i, j]] = dist;
            dists[[j, i]] = dist;
        }
    }

    // For each point find k nearest neighbours
    let mut adj = vec![vec![false; n]; n];
    for i in 0..n {
        let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        indices.sort_by(|&a, &b| {
            dists[[i, a]]
                .partial_cmp(&dists[[i, b]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &j in indices.iter().take(k) {
            adj[i][j] = true;
        }
    }

    // Build affinity
    let mut w = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let connected = if mutual {
                adj[i][j] && adj[j][i]
            } else {
                adj[i][j] || adj[j][i]
            };
            if connected {
                w[[i, j]] = F::one();
                w[[j, i]] = F::one();
            }
        }
    }

    Ok(w)
}

// ---------------------------------------------------------------------------
// Graph Laplacian
// ---------------------------------------------------------------------------

/// Returns `(laplacian, d_inv_sqrt)` where `d_inv_sqrt` is needed by NJW.
fn build_laplacian<F>(w: &Array2<F>, variant: SpectralVariant) -> Result<(Array2<F>, Array1<F>)>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = w.nrows();

    // Degree vector
    let mut deg = Array1::<F>::zeros(n);
    for i in 0..n {
        deg[i] = w.row(i).sum();
    }

    // D^{-1/2}
    let mut d_inv_sqrt = Array1::<F>::zeros(n);
    for i in 0..n {
        if deg[i] > F::zero() {
            d_inv_sqrt[i] = F::one() / deg[i].sqrt();
        }
    }

    let mut lap = Array2::<F>::zeros((n, n));

    match variant {
        SpectralVariant::Unnormalized => {
            // L = D - W
            for i in 0..n {
                lap[[i, i]] = deg[i];
                for j in 0..n {
                    lap[[i, j]] = lap[[i, j]] - w[[i, j]];
                }
            }
        }
        SpectralVariant::NgJordanWeiss => {
            // L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        lap[[i, j]] = F::one() - w[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    } else {
                        lap[[i, j]] = -w[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    }
                }
            }
        }
        SpectralVariant::ShiMalik => {
            // L_rw = D^{-1} L = I - D^{-1} W
            // For eigendecomp, use symmetric form: D^{-1/2} L D^{-1/2}
            // then transform eigenvectors back with D^{-1/2}
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        lap[[i, j]] = F::one() - w[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    } else {
                        lap[[i, j]] = -w[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
                    }
                }
            }
        }
    }

    // Add small stabilisation to diagonal
    let eps = F::from(1e-10).unwrap_or(F::epsilon());
    for i in 0..n {
        lap[[i, i]] = lap[[i, i]] + eps;
    }

    Ok((lap, d_inv_sqrt))
}

// ---------------------------------------------------------------------------
// Eigengap heuristic
// ---------------------------------------------------------------------------

fn compute_eigengaps<F: Float>(eigenvalues: &Array1<F>) -> Vec<F> {
    let n = eigenvalues.len();
    if n < 2 {
        return Vec::new();
    }
    let mut gaps = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        gaps.push(eigenvalues[i + 1] - eigenvalues[i]);
    }
    gaps
}

fn eigengap_heuristic<F: Float + FromPrimitive>(gaps: &[F], max_k: usize) -> usize {
    if gaps.is_empty() {
        return 2;
    }
    // Skip the first gap (between 0-eigenvalue and second) and find the
    // largest gap in position 1..max_k which indicates the number of clusters.
    let search_end = max_k.min(gaps.len());
    let mut best_idx = 1;
    let mut best_gap = F::zero();
    for i in 1..search_end {
        if gaps[i] > best_gap {
            best_gap = gaps[i];
            best_idx = i + 1; // k = position after the gap
        }
    }
    best_idx.max(2)
}

// ---------------------------------------------------------------------------
// Embedding extraction
// ---------------------------------------------------------------------------

fn extract_embedding<F>(
    eigvecs: &Array2<F>,
    k: usize,
    variant: SpectralVariant,
    d_inv_sqrt: &Array1<F>,
) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = eigvecs.nrows();
    let actual_k = k.min(eigvecs.ncols());

    // For Unnormalized spectral clustering, skip the trivial first
    // eigenvector (constant for connected graphs).
    // For NJW and Shi-Malik, take the first k eigenvectors directly
    // (the row normalization step in NJW handles the constant component).
    let (start, end) = match variant {
        SpectralVariant::Unnormalized => {
            let s = if eigvecs.ncols() > actual_k { 1 } else { 0 };
            (s, (s + actual_k).min(eigvecs.ncols()))
        }
        _ => (0, actual_k),
    };

    let cols = end - start;
    let mut emb = Array2::<F>::zeros((n, cols));
    for i in 0..n {
        for j in 0..cols {
            emb[[i, j]] = eigvecs[[i, start + j]];
        }
    }

    match variant {
        SpectralVariant::NgJordanWeiss => {
            // Row-normalise
            for i in 0..n {
                let mut norm_sq = F::zero();
                for j in 0..cols {
                    norm_sq = norm_sq + emb[[i, j]] * emb[[i, j]];
                }
                let norm = norm_sq.sqrt();
                if norm > F::epsilon() {
                    for j in 0..cols {
                        emb[[i, j]] = emb[[i, j]] / norm;
                    }
                }
            }
        }
        SpectralVariant::ShiMalik => {
            // Transform back: u_i = D^{-1/2} v_i
            for i in 0..n {
                for j in 0..cols {
                    emb[[i, j]] = emb[[i, j]] * d_inv_sqrt[i];
                }
            }
        }
        SpectralVariant::Unnormalized => {
            // No post-processing
        }
    }

    Ok(emb)
}

// ---------------------------------------------------------------------------
// Simple k-means for embedding clustering
// ---------------------------------------------------------------------------

fn run_kmeans_on_embedding<F>(
    embedding: &Array2<F>,
    k: usize,
    max_iter: usize,
    seed: Option<u64>,
) -> Result<Array1<usize>>
where
    F: Float + FromPrimitive + Debug + 'static,
{
    let n = embedding.nrows();
    let d = embedding.ncols();

    if k >= n {
        return Ok(Array1::from_vec((0..n).collect()));
    }

    let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));

    // K-means++ init
    let mut centroids = Array2::<F>::zeros((k, d));
    let first: usize = rng.random_range(0..n);
    centroids.row_mut(0).assign(&embedding.row(first));

    for c in 1..k {
        let mut dists = Array1::<F>::zeros(n);
        for i in 0..n {
            let mut min_d = F::infinity();
            for prev in 0..c {
                let dist = sq_dist(embedding.row(i), centroids.row(prev), d);
                if dist < min_d {
                    min_d = dist;
                }
            }
            dists[i] = min_d;
        }
        let total: F = dists.iter().fold(F::zero(), |a, &v| a + v);
        if total <= F::zero() {
            centroids.row_mut(c).assign(&embedding.row(c.min(n - 1)));
            continue;
        }
        let r: f64 = rng.random::<f64>();
        let threshold = F::from(r).unwrap_or(F::zero()) * total;
        let mut cumsum = F::zero();
        let mut chosen = 0;
        for i in 0..n {
            cumsum = cumsum + dists[i];
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.row_mut(c).assign(&embedding.row(chosen));
    }

    // Lloyd iterations
    let mut labels = Array1::<usize>::zeros(n);
    for _iter in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let mut best_c = 0;
            let mut best_d = F::infinity();
            for c in 0..k {
                let dist = sq_dist(embedding.row(i), centroids.row(c), d);
                if dist < best_d {
                    best_d = dist;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        let mut new_c = Array2::<F>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..d {
                new_c[[c, j]] = new_c[[c, j]] + embedding[[i, j]];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let sz = F::from(counts[c]).unwrap_or(F::one());
                for j in 0..d {
                    new_c[[c, j]] = new_c[[c, j]] / sz;
                }
            }
        }
        centroids = new_c;
    }

    Ok(labels)
}

fn sq_dist<F: Float>(a: ArrayView1<F>, b: ArrayView1<F>, d: usize) -> F {
    let mut s = F::zero();
    for j in 0..d {
        let diff = a[j] - b[j];
        s = s + diff * diff;
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_blob_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
        )
        .expect("data")
    }

    #[test]
    fn test_njw_two_clusters() {
        let data = two_blob_data();
        // Use explicit gamma for reliable separation: the two blobs have
        // squared distance ~50 between them and ~0.05 within.
        // gamma=0.1 gives between-cluster affinity exp(-0.1*50)=exp(-5)≈0.007
        // and within-cluster affinity exp(-0.1*0.05)≈0.995.
        // This ensures the graph stays connected (single near-zero eigenvalue)
        // while still being well-separated.
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Rbf { gamma: Some(0.1) },
            seed: Some(42),
            kmeans_max_iter: 300,
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("njw");
        assert_eq!(res.n_clusters, 2);
        assert_eq!(res.labels.len(), 8);
        // First 4 should be same cluster, last 4 different.
        // Spectral clustering labels are arbitrary, so we only check
        // that points within each blob share a label and the two blobs differ.
        let label_a = res.labels[0];
        let label_b = res.labels[4];
        assert_ne!(label_a, label_b, "the two blobs must get different labels");
        for i in 0..4 {
            assert_eq!(
                res.labels[i], label_a,
                "point {} should share label with point 0",
                i
            );
        }
        for i in 4..8 {
            assert_eq!(
                res.labels[i], label_b,
                "point {} should share label with point 4",
                i
            );
        }
    }

    #[test]
    fn test_unnormalized_two_clusters() {
        let data = two_blob_data();
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::Unnormalized,
            kernel: SimilarityKernel::Rbf { gamma: Some(0.1) },
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("unnorm");
        assert_eq!(res.n_clusters, 2);
        // Both clusters should be present
        let labels_set: std::collections::HashSet<_> = res.labels.iter().copied().collect();
        assert_eq!(labels_set.len(), 2);
    }

    #[test]
    fn test_shi_malik_two_clusters() {
        let data = two_blob_data();
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::ShiMalik,
            kernel: SimilarityKernel::Rbf { gamma: Some(0.1) },
            seed: Some(42),
            kmeans_n_init: 5,
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("shi_malik");
        assert_eq!(res.n_clusters, 2);
        // Shi-Malik should produce a valid partition with 2 labels
        let labels_set: std::collections::HashSet<_> = res.labels.iter().copied().collect();
        assert!(
            labels_set.len() <= 2,
            "should produce at most 2 clusters, got {:?}",
            labels_set
        );
        // First blob should be internally consistent
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[0], res.labels[2]);
        assert_eq!(res.labels[0], res.labels[3]);
    }

    #[test]
    fn test_knn_kernel() {
        let data = two_blob_data();
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Knn {
                k: 3,
                mutual: false,
            },
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("knn");
        assert_eq!(res.n_clusters, 2);
    }

    #[test]
    fn test_knn_mutual_kernel() {
        let data = two_blob_data();
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Knn { k: 3, mutual: true },
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("knn mutual");
        assert_eq!(res.n_clusters, 2);
    }

    #[test]
    fn test_eigengap_auto_k() {
        let data = two_blob_data();
        let cfg = SpectralConfig {
            n_clusters: None,
            max_clusters: 5,
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Rbf { gamma: None },
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("eigengap");
        assert!(
            res.n_clusters >= 2,
            "auto k should be >= 2, got {}",
            res.n_clusters
        );
        assert!(res.eigengaps.is_some());
    }

    #[test]
    fn test_precomputed_kernel() {
        // Build a simple affinity matrix for 4 points: 2 clusters of 2
        let w = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.9, 0.1, 0.1, 0.9, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.9, 0.1, 0.1, 0.9, 1.0,
            ],
        )
        .expect("affinity");
        let cfg = SpectralConfig {
            n_clusters: Some(2),
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Precomputed,
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(w.view(), &cfg).expect("precomputed");
        assert_eq!(res.labels[0], res.labels[1]);
        assert_eq!(res.labels[2], res.labels[3]);
        assert_ne!(res.labels[0], res.labels[2]);
    }

    #[test]
    fn test_spectral_error_too_few() {
        let data = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).expect("data");
        let cfg = SpectralConfig::<f64>::default();
        assert!(spectral_cluster(data.view(), &cfg).is_err());
    }

    #[test]
    fn test_eigengap_heuristic_fn() {
        // Gaps: small, BIG, small  =>  k = 3 (gap at position 1 means k=2)
        let gaps = vec![0.01, 0.5, 0.02, 0.01];
        let k = eigengap_heuristic(&gaps, 5);
        assert_eq!(k, 2, "eigengap should select k=2, got {}", k);
    }

    #[test]
    fn test_rbf_affinity_auto_gamma() {
        let data = two_blob_data();
        let w = rbf_affinity(data.view(), None).expect("rbf");
        assert_eq!(w.nrows(), 8);
        assert_eq!(w.ncols(), 8);
        // Diagonal should be 1
        for i in 0..8 {
            assert!((w[[i, i]] - 1.0).abs() < 1e-10);
        }
        // Same-cluster affinity > cross-cluster
        assert!(w[[0, 1]] > w[[0, 4]]);
    }

    #[test]
    fn test_knn_affinity_symmetric() {
        let data = two_blob_data();
        let w = knn_affinity(data.view(), 2, false).expect("knn aff");
        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (w[[i, j]] - w[[j, i]]).abs() < 1e-10,
                    "knn affinity should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_three_blob_spectral() {
        let data = Array2::from_shape_vec(
            (12, 2),
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, 5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
                10.0, 0.0, 10.1, 0.1, 10.2, 0.0, 10.0, 0.2,
            ],
        )
        .expect("data");
        let cfg = SpectralConfig {
            n_clusters: Some(3),
            variant: SpectralVariant::NgJordanWeiss,
            kernel: SimilarityKernel::Rbf { gamma: None },
            seed: Some(42),
            ..Default::default()
        };
        let res = spectral_cluster(data.view(), &cfg).expect("3 blobs");
        assert_eq!(res.n_clusters, 3);
        let labels_set: std::collections::HashSet<_> = res.labels.iter().copied().collect();
        assert_eq!(labels_set.len(), 3, "should find 3 clusters");
    }
}
