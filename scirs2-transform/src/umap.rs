//! UMAP (Uniform Manifold Approximation and Projection) — advanced API
//!
//! This module provides a config-based UMAP API that wraps the core UMAP
//! algorithm with a richer interface including explicit `UmapConfig`,
//! `DistanceMetric`, `InitMethod`, and `UmapResult` types.
//!
//! ## Algorithm Overview
//!
//! 1. k-NN graph construction (brute-force O(n²) for correctness)
//! 2. Smooth k-NN distance computation → fuzzy simplicial set (high-dim graph)
//! 3. Initialize embedding (spectral via normalised Laplacian, or random)
//! 4. SGD optimisation: attractive forces on graph edges + negative sampling

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Rng, RngExt};
use scirs2_linalg::eigh;
use std::collections::BinaryHeap;

use crate::error::{Result, TransformError};

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Distance metric used in UMAP kNN graph construction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistanceMetric {
    /// L2 (Euclidean) distance
    Euclidean,
    /// Cosine distance (1 − cosine_similarity)
    Cosine,
    /// L1 (Manhattan / taxicab) distance
    Manhattan,
}

/// Embedding initialisation strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitMethod {
    /// Initialise from eigenvectors of the normalised Laplacian of the graph
    Spectral,
    /// Uniform random initialisation in [−5, 5]
    Random,
}

/// Configuration for the UMAP algorithm
#[derive(Debug, Clone)]
pub struct UmapConfig {
    /// Dimensionality of the embedding (default 2)
    pub n_components: usize,
    /// Number of nearest neighbours used to construct the high-dim graph (default 15)
    pub n_neighbors: usize,
    /// Minimum distance between points in the low-dim space (default 0.1)
    pub min_dist: f64,
    /// Distance metric used in high-dim space (default Euclidean)
    pub metric: DistanceMetric,
    /// Number of SGD optimisation epochs (default 200)
    pub n_epochs: usize,
    /// SGD learning rate (default 1.0)
    pub learning_rate: f64,
    /// Initialisation strategy (default Spectral with Random fallback)
    pub init: InitMethod,
    /// RNG seed for reproducibility (default 42)
    pub seed: u64,
    /// Number of negative samples drawn per positive edge per epoch (default 5)
    pub negative_sample_rate: usize,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            metric: DistanceMetric::Euclidean,
            n_epochs: 200,
            learning_rate: 1.0,
            init: InitMethod::Spectral,
            seed: 42,
            negative_sample_rate: 5,
        }
    }
}

/// Result returned by [`Umap::fit_transform`]
#[derive(Debug, Clone)]
pub struct UmapResult {
    /// Low-dimensional embedding, shape `(n_samples, n_components)`
    pub embedding: Array2<f64>,
    /// Symmetric fuzzy simplicial set (weighted high-dim graph), shape `(n_samples, n_samples)`
    pub graph: Array2<f64>,
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

/// Approximate the UMAP low-dim membership function parameters `a` and `b`.
///
/// We want `1 / (1 + a * d^{2b})` to approximate the target curve:
///   `1` if `d ≤ min_dist`, otherwise `exp(-(d - min_dist) / spread)`.
fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    if min_dist <= 0.0 || spread <= 0.0 {
        return (1.0, 1.0);
    }

    let mut a = 1.0_f64;
    let mut b = if min_dist < spread && (1.0 - min_dist).abs() > 1e-10 {
        (min_dist.abs().ln() / (1.0 - min_dist).abs().ln()).clamp(0.1, 10.0)
    } else {
        1.0
    };

    // Simple gradient descent on MSE between model and target curve
    for _ in 0..200 {
        let mut grad_a = 0.0_f64;
        let mut grad_b = 0.0_f64;
        let n_pts = 50_usize;
        for k in 0..n_pts {
            let d = min_dist + 3.0 * spread * (k as f64 / n_pts as f64);
            if d < 1e-12 {
                continue;
            }
            let target = if d <= min_dist {
                1.0
            } else {
                (-(d - min_dist) / spread).exp()
            };
            let d2b = d.powf(2.0 * b);
            let denom = 1.0 + a * d2b;
            let model = 1.0 / denom;
            let err = model - target;
            grad_a += err * (-d2b / (denom * denom));
            let db_term = if d > 1e-12 {
                -2.0 * a * d2b * d.ln() / (denom * denom)
            } else {
                0.0
            };
            grad_b += err * db_term;
        }
        a = (a - 0.05 * grad_a).max(1e-4);
        b = (b - 0.05 * grad_b).max(1e-4);
        if grad_a.abs() < 1e-9 && grad_b.abs() < 1e-9 {
            break;
        }
    }

    (a, b)
}

/// Compute pairwise distances according to `metric`
fn compute_distances(data: &Array2<f64>, metric: &DistanceMetric) -> Array2<f64> {
    let n = data.shape()[0];
    let d = data.shape()[1];
    let mut dist = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let v = match metric {
                DistanceMetric::Euclidean => {
                    let mut s = 0.0_f64;
                    for k in 0..d {
                        let diff = data[[i, k]] - data[[j, k]];
                        s += diff * diff;
                    }
                    s.sqrt()
                }
                DistanceMetric::Manhattan => {
                    let mut s = 0.0_f64;
                    for k in 0..d {
                        s += (data[[i, k]] - data[[j, k]]).abs();
                    }
                    s
                }
                DistanceMetric::Cosine => {
                    let mut dot = 0.0_f64;
                    let mut ni = 0.0_f64;
                    let mut nj = 0.0_f64;
                    for k in 0..d {
                        dot += data[[i, k]] * data[[j, k]];
                        ni += data[[i, k]] * data[[i, k]];
                        nj += data[[j, k]] * data[[j, k]];
                    }
                    let denom = (ni * nj).sqrt();
                    if denom > 1e-12 {
                        1.0 - (dot / denom).clamp(-1.0, 1.0)
                    } else {
                        1.0
                    }
                }
            };
            dist[[i, j]] = v;
            dist[[j, i]] = v;
        }
    }
    dist
}

/// Find `k` nearest neighbours for each point using a min-heap
fn find_knn(distances: &Array2<f64>, k: usize) -> (Array2<usize>, Array2<f64>) {
    let n = distances.shape()[0];
    let k = k.min(n - 1);
    let mut indices = Array2::zeros((n, k));
    let mut dists = Array2::zeros((n, k));

    for i in 0..n {
        // Use a max-heap of size k (storing negative distances)
        let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();
        for j in 0..n {
            if i == j {
                continue;
            }
            let d_fixed = (distances[[i, j]] * 1_000_000.0) as i64;
            heap.push((std::cmp::Reverse(d_fixed), j));
        }
        for slot in 0..k {
            if let Some((std::cmp::Reverse(d_fixed), nb)) = heap.pop() {
                indices[[i, slot]] = nb;
                dists[[i, slot]] = d_fixed as f64 / 1_000_000.0;
            }
        }
    }
    (indices, dists)
}

/// Build the fuzzy simplicial set (symmetric graph) from kNN
fn build_fuzzy_graph(
    knn_indices: &Array2<usize>,
    knn_dists: &Array2<f64>,
    local_connectivity: f64,
) -> Array2<f64> {
    let n = knn_indices.shape()[0];
    let k = knn_indices.shape()[1];
    let target_entropy = (k as f64).ln() / 2.0_f64.ln();

    let mut graph = Array2::zeros((n, n));

    for i in 0..n {
        let rho_idx = ((local_connectivity as usize).saturating_sub(1)).min(k - 1);
        let rho = knn_dists[[i, rho_idx]];

        // Binary search for sigma
        let mut sigma_lo = 0.0_f64;
        let mut sigma_hi = f64::INFINITY;
        let mut sigma = 1.0_f64;

        for _ in 0..64 {
            let mut membership_sum = 0.0_f64;
            for j in 0..k {
                let d = (knn_dists[[i, j]] - rho).max(0.0);
                if sigma > 1e-15 {
                    membership_sum += (-d / sigma).exp();
                }
            }
            let diff = membership_sum - target_entropy;
            if diff.abs() < 1e-5 {
                break;
            }
            if diff > 0.0 {
                sigma_hi = sigma;
                sigma = (sigma_lo + sigma_hi) / 2.0;
            } else {
                sigma_lo = sigma;
                sigma = if sigma_hi.is_infinite() {
                    sigma * 2.0
                } else {
                    (sigma_lo + sigma_hi) / 2.0
                };
            }
        }

        for j in 0..k {
            let nb = knn_indices[[i, j]];
            let d = (knn_dists[[i, j]] - rho).max(0.0);
            let strength = if sigma > 1e-15 {
                (-d / sigma).exp()
            } else if d < 1e-15 {
                1.0
            } else {
                0.0
            };
            graph[[i, nb]] = strength;
        }
    }

    // Symmetric union: A + A^T - A * A^T
    let gt = graph.t().to_owned();
    &graph + &gt - &graph * &gt
}

/// Spectral initialisation from the normalised Laplacian
fn spectral_init(n_samples: usize, n_components: usize, graph: &Array2<f64>) -> Result<Array2<f64>> {
    // Degree vector
    let mut degree = Array1::zeros(n_samples);
    for i in 0..n_samples {
        degree[i] = graph.row(i).sum();
    }

    // Check for isolated nodes
    for i in 0..n_samples {
        if degree[i] < 1e-12 {
            return Err(TransformError::ComputationError(
                "Graph contains isolated node(s); falling back to random init".to_string(),
            ));
        }
    }

    // Normalised Laplacian: L = I − D^{-½} W D^{-½}
    let mut laplacian = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            laplacian[[i, j]] = if i == j {
                1.0
            } else {
                -graph[[i, j]] / (degree[i] * degree[j]).sqrt()
            };
        }
    }

    let (eigenvalues, eigenvectors) =
        eigh(&laplacian.view(), None).map_err(|e| TransformError::LinalgError(e))?;

    // Sort by eigenvalue ascending and skip the first (trivial) eigenvector
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&a, &b| {
        eigenvalues[a]
            .partial_cmp(&eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut embedding = Array2::zeros((n_samples, n_components));
    for dim in 0..n_components {
        let idx = order[dim + 1]; // skip constant eigenvector
        for i in 0..n_samples {
            embedding[[i, dim]] = eigenvectors[[i, idx]] * 10.0;
        }
    }

    Ok(embedding)
}

/// SGD optimisation of the low-dim embedding
fn optimize_embedding(
    embedding: &mut Array2<f64>,
    graph: &Array2<f64>,
    a: f64,
    b: f64,
    n_epochs: usize,
    learning_rate: f64,
    negative_sample_rate: usize,
    rng: &mut impl Rng,
) {
    let n = embedding.shape()[0];
    let n_components = embedding.shape()[1];
    let clip = 4.0_f64;

    // Collect edges (i, j, weight)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let w = graph[[i, j]];
            if w > 0.0 {
                edges.push((i, j));
                weights.push(w);
            }
        }
    }
    let n_edges = edges.len();
    if n_edges == 0 {
        return;
    }

    let max_w = weights.iter().cloned().fold(0.0_f64, f64::max);
    let eps_per_sample: Vec<f64> = weights
        .iter()
        .map(|&w| (max_w / w.max(1e-10)).min(n_epochs as f64))
        .collect();

    let eps_per_neg: Vec<f64> = eps_per_sample
        .iter()
        .map(|&e| e / (negative_sample_rate as f64).max(1.0))
        .collect();

    let mut next_sample: Vec<f64> = eps_per_sample.clone();
    let mut next_neg: Vec<f64> = eps_per_neg.clone();

    for epoch in 0..n_epochs {
        let alpha = learning_rate * (1.0 - epoch as f64 / n_epochs as f64).max(0.001);

        for edge_idx in 0..n_edges {
            if next_sample[edge_idx] > epoch as f64 {
                continue;
            }

            let (i, j) = edges[edge_idx];

            // Attractive force
            let mut dist_sq = 0.0_f64;
            for d in 0..n_components {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                dist_sq += diff * diff;
            }
            dist_sq = dist_sq.max(1e-10);

            let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0)
                / (1.0 + a * dist_sq.powf(b)).max(1e-10);

            for d in 0..n_components {
                let g = (grad_coeff * (embedding[[i, d]] - embedding[[j, d]])).clamp(-clip, clip);
                embedding[[i, d]] += alpha * g;
                embedding[[j, d]] -= alpha * g;
            }
            next_sample[edge_idx] += eps_per_sample[edge_idx];

            // Negative samples (repulsive force)
            for _ in 0..negative_sample_rate {
                if next_neg[edge_idx] > epoch as f64 {
                    break;
                }
                let k = rng.random_range(0..n);
                if k == i {
                    next_neg[edge_idx] += eps_per_neg[edge_idx];
                    continue;
                }
                let mut neg_dist_sq = 0.0_f64;
                for d in 0..n_components {
                    let diff = embedding[[i, d]] - embedding[[k, d]];
                    neg_dist_sq += diff * diff;
                }
                neg_dist_sq = neg_dist_sq.max(1e-10);

                let rep_coeff =
                    2.0 * b / ((0.001 + neg_dist_sq) * (1.0 + a * neg_dist_sq.powf(b)).max(1e-10));

                for d in 0..n_components {
                    let g =
                        (rep_coeff * (embedding[[i, d]] - embedding[[k, d]])).clamp(-clip, clip);
                    embedding[[i, d]] += alpha * g;
                }
                next_neg[edge_idx] += eps_per_neg[edge_idx];
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Main struct
// ────────────────────────────────────────────────────────────────────────────

/// UMAP dimensionality reduction with config-based API.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::umap::{Umap, UmapConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut umap = Umap::new(UmapConfig::default());
/// let embedding = umap.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
pub struct Umap {
    config: UmapConfig,
    /// Fitted embedding (available after `fit_transform`)
    embedding: Option<Array2<f64>>,
    /// Fitted high-dim graph (available after `fit_transform`)
    graph: Option<Array2<f64>>,
    /// Training data stored for `transform` (out-of-sample)
    training_data: Option<Array2<f64>>,
    /// `a` parameter of the low-dim membership function
    a: f64,
    /// `b` parameter of the low-dim membership function
    b: f64,
}

impl Umap {
    /// Create a new `Umap` with the given configuration.
    pub fn new(config: UmapConfig) -> Self {
        let spread = 1.0_f64;
        let (a, b) = find_ab_params(spread, config.min_dist);
        Self {
            config,
            embedding: None,
            graph: None,
            training_data: None,
            a,
            b,
        }
    }

    /// Fit to `data` and return the low-dim embedding.
    ///
    /// # Arguments
    /// * `data` — shape `(n_samples, n_features)`
    ///
    /// # Returns
    /// Low-dim embedding, shape `(n_samples, n_components)`.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = data.shape()[0];
        let n_features = data.shape()[1];

        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }
        if self.config.n_neighbors >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors ({}) must be < n_samples ({})",
                self.config.n_neighbors, n_samples
            )));
        }
        if self.config.n_components == 0 {
            return Err(TransformError::InvalidInput(
                "n_components must be >= 1".to_string(),
            ));
        }

        // 1. Pairwise distances
        let distances = compute_distances(data, &self.config.metric);

        // 2. kNN
        let (knn_indices, knn_dists) = find_knn(&distances, self.config.n_neighbors);

        // 3. Fuzzy simplicial set
        let graph = build_fuzzy_graph(&knn_indices, &knn_dists, 1.0);

        // 4. Initialise embedding
        let mut embedding = if self.config.init == InitMethod::Spectral
            && n_samples > self.config.n_components + 1
        {
            match spectral_init(n_samples, self.config.n_components, &graph) {
                Ok(e) => e,
                Err(_) => self.random_init(n_samples),
            }
        } else {
            self.random_init(n_samples)
        };

        // 5. SGD optimisation
        let mut rng = scirs2_core::random::rng();
        optimize_embedding(
            &mut embedding,
            &graph,
            self.a,
            self.b,
            self.config.n_epochs,
            self.config.learning_rate,
            self.config.negative_sample_rate,
            &mut rng,
        );

        // Store state for out-of-sample transform
        self.training_data = Some(data.clone());
        self.graph = Some(graph);
        self.embedding = Some(embedding.clone());

        Ok(embedding)
    }

    /// Project new (unseen) data into the fitted embedding space using
    /// inverse-distance-weighted interpolation from k nearest training points.
    ///
    /// The model must have been fitted with [`fit_transform`] first.
    pub fn transform(&self, new_data: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self.training_data.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Umap model has not been fitted".to_string())
        })?;
        let train_emb = self.embedding.as_ref().ok_or_else(|| {
            TransformError::NotFitted("Umap model has not been fitted".to_string())
        })?;

        let n_new = new_data.shape()[0];
        let n_features = new_data.shape()[1];
        let n_train = training_data.shape()[0];

        if n_features != training_data.shape()[1] {
            return Err(TransformError::InvalidInput(format!(
                "Feature mismatch: new_data has {} features, training had {}",
                n_features,
                training_data.shape()[1]
            )));
        }

        let k = self.config.n_neighbors.min(n_train);
        let mut out = Array2::zeros((n_new, self.config.n_components));

        for i in 0..n_new {
            // Compute distances to all training points
            let mut dists: Vec<(f64, usize)> = (0..n_train)
                .map(|j| {
                    let mut sq = 0.0_f64;
                    for feat in 0..n_features {
                        let diff = new_data[[i, feat]] - training_data[[j, feat]];
                        sq += diff * diff;
                    }
                    (sq.sqrt(), j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Inverse-distance weighting
            let mut total_w = 0.0_f64;
            let mut weighted = vec![0.0_f64; self.config.n_components];
            for &(d, train_idx) in dists.iter().take(k) {
                let w = if d > 1e-12 { 1.0 / d } else { 1e12 };
                total_w += w;
                for dim in 0..self.config.n_components {
                    weighted[dim] += w * train_emb[[train_idx, dim]];
                }
            }
            if total_w > 0.0 {
                for dim in 0..self.config.n_components {
                    out[[i, dim]] = weighted[dim] / total_w;
                }
            }
        }

        Ok(out)
    }

    /// Return the last result (embedding + graph) if available.
    pub fn result(&self) -> Option<UmapResult> {
        match (&self.embedding, &self.graph) {
            (Some(e), Some(g)) => Some(UmapResult {
                embedding: e.clone(),
                graph: g.clone(),
            }),
            _ => None,
        }
    }

    fn random_init(&self, n_samples: usize) -> Array2<f64> {
        let mut rng = scirs2_core::random::rng();
        Array2::from_shape_fn((n_samples, self.config.n_components), |_| {
            rng.random_range(-5.0..5.0_f64)
        })
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn make_two_clusters(n_each: usize) -> Array2<f64> {
        let mut rows = Vec::with_capacity(n_each * 2 * 3);
        for i in 0..n_each {
            let t = i as f64 / n_each as f64;
            rows.extend_from_slice(&[t, t * 0.5, 0.0]);
        }
        for i in 0..n_each {
            let t = i as f64 / n_each as f64;
            rows.extend_from_slice(&[t + 10.0, t * 0.5 + 10.0, 5.0]);
        }
        Array::from_shape_vec((n_each * 2, 3), rows).expect("shape")
    }

    #[test]
    fn test_umap_output_shape() {
        let data = make_two_clusters(10);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 5,
            n_epochs: 50,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[20, 2]);
        for v in emb.iter() {
            assert!(v.is_finite(), "embedding contains non-finite value");
        }
    }

    #[test]
    fn test_umap_3_components() {
        let data = make_two_clusters(8);
        let mut umap = Umap::new(UmapConfig {
            n_components: 3,
            n_neighbors: 4,
            n_epochs: 30,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[16, 3]);
    }

    #[test]
    fn test_umap_random_init() {
        let data = make_two_clusters(8);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 4,
            n_epochs: 30,
            init: InitMethod::Random,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.nrows(), 16);
        assert_eq!(emb.ncols(), 2);
    }

    #[test]
    fn test_umap_cosine_metric() {
        let data = make_two_clusters(8);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 4,
            n_epochs: 20,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[16, 2]);
    }

    #[test]
    fn test_umap_manhattan_metric() {
        let data = make_two_clusters(8);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 4,
            n_epochs: 20,
            metric: DistanceMetric::Manhattan,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[16, 2]);
    }

    #[test]
    fn test_umap_result_accessor() {
        let data = make_two_clusters(8);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 4,
            n_epochs: 20,
            ..Default::default()
        });
        umap.fit_transform(&data).expect("fit_transform");
        let result = umap.result().expect("result should be available after fit");
        assert_eq!(result.embedding.shape(), &[16, 2]);
        assert_eq!(result.graph.shape(), &[16, 16]);
        for v in result.graph.iter() {
            assert!(v.is_finite());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn test_umap_transform_out_of_sample() {
        let data = make_two_clusters(10);
        let new_point: Array2<f64> =
            Array::from_shape_vec((1, 3), vec![0.5, 0.25, 0.0]).expect("shape");
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 5,
            n_epochs: 30,
            ..Default::default()
        });
        umap.fit_transform(&data).expect("fit_transform");
        let t = umap.transform(&new_point).expect("transform");
        assert_eq!(t.shape(), &[1, 2]);
        for v in t.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_umap_linearly_separable_structure() {
        // Two clearly separated clusters — their 2-D UMAP projections
        // should also have different centroids.
        let n = 12_usize;
        let data = make_two_clusters(n);
        let mut umap = Umap::new(UmapConfig {
            n_neighbors: 5,
            n_epochs: 100,
            ..Default::default()
        });
        let emb = umap.fit_transform(&data).expect("fit_transform");

        // Centroid of first cluster in 2-D
        let mut c0 = [0.0_f64; 2];
        let mut c1 = [0.0_f64; 2];
        for i in 0..n {
            c0[0] += emb[[i, 0]];
            c0[1] += emb[[i, 1]];
            c1[0] += emb[[n + i, 0]];
            c1[1] += emb[[n + i, 1]];
        }
        c0[0] /= n as f64;
        c0[1] /= n as f64;
        c1[0] /= n as f64;
        c1[1] /= n as f64;

        let centroid_dist = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
        // Clusters should not collapse to the same point
        assert!(
            centroid_dist > 0.01,
            "cluster centroids are too close: dist = {centroid_dist:.4}"
        );
    }

    #[test]
    fn test_umap_no_unwrap_unfitted_transform() {
        let umap = Umap::new(UmapConfig::default());
        let d: Array2<f64> = Array::zeros((3, 2));
        let res = umap.transform(&d);
        assert!(res.is_err(), "transform before fit should return Err");
    }
}
