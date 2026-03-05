//! Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction
//!
//! UMAP is a non-linear dimensionality reduction technique that can be used for
//! visualization similarly to t-SNE, but also for general non-linear dimension reduction.
//!
//! ## Algorithm Overview
//!
//! 1. **k-NN graph construction**: Find k nearest neighbors for each point
//! 2. **Fuzzy simplicial set**: Compute membership strengths using smooth kNN distance
//! 3. **Spectral initialization**: Initialize embedding using Laplacian eigenvectors
//! 4. **SGD optimization**: Optimize layout with negative sampling
//!
//! ## Features
//!
//! - Proper smooth k-NN distance computation with binary search for sigma
//! - Fuzzy simplicial set union with local connectivity constraint
//! - Spectral initialization via normalized Laplacian eigenvectors
//! - SGD layout optimization with edge sampling and negative sampling schedule
//! - Out-of-sample extension via inverse distance weighting

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::random::Rng;
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;
use std::collections::BinaryHeap;

use crate::error::{Result, TransformError};

/// UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction
///
/// UMAP constructs a high dimensional graph representation of the data then optimizes
/// a low dimensional graph to be as structurally similar as possible.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::UMAP;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut umap = UMAP::new(15, 2, 0.1, 1.0, 200);
/// let embedding = umap.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct UMAP {
    /// Number of neighbors to consider for local structure
    n_neighbors: usize,
    /// Number of components (dimensions) in the low dimensional space
    n_components: usize,
    /// Controls how tightly UMAP packs points together (minimum distance)
    min_dist: f64,
    /// Controls the effective scale of local vs global structure
    spread: f64,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Number of epochs for optimization
    n_epochs: usize,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Training k-NN graph for out-of-sample extension
    training_graph: Option<Array2<f64>>,
    /// Metric to use for distance computation
    metric: String,
    /// The low dimensional embedding
    embedding: Option<Array2<f64>>,
    /// Negative sampling rate (number of negative samples per positive edge)
    negative_sample_rate: usize,
    /// Whether to use spectral initialization
    spectral_init: bool,
    /// Parameters for the smooth approximation
    a: f64,
    b: f64,
    /// Local connectivity parameter (must be >= 1)
    local_connectivity: f64,
    /// Set operation mix ratio (0.0 = pure intersection, 1.0 = pure union)
    set_op_mix_ratio: f64,
}

impl UMAP {
    /// Creates a new UMAP instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors to consider for local structure (typically 5-50)
    /// * `n_components` - Number of dimensions in the low dimensional space (typically 2 or 3)
    /// * `min_dist` - Minimum distance between points in low dimensional space (typically 0.001-0.5)
    /// * `learning_rate` - Learning rate for SGD optimization (typically 1.0)
    /// * `n_epochs` - Number of epochs for optimization (typically 200-500)
    pub fn new(
        n_neighbors: usize,
        n_components: usize,
        min_dist: f64,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Self {
        let spread = 1.0;
        let (a, b) = Self::find_ab_params(spread, min_dist);

        UMAP {
            n_neighbors,
            n_components,
            min_dist,
            spread,
            learning_rate,
            n_epochs,
            random_state: None,
            metric: "euclidean".to_string(),
            embedding: None,
            training_data: None,
            training_graph: None,
            negative_sample_rate: 5,
            spectral_init: true,
            a,
            b,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
        }
    }

    /// Sets the random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sets the distance metric
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Sets the negative sampling rate
    pub fn with_negative_sample_rate(mut self, rate: usize) -> Self {
        self.negative_sample_rate = rate;
        self
    }

    /// Enable or disable spectral initialization
    pub fn with_spectral_init(mut self, use_spectral: bool) -> Self {
        self.spectral_init = use_spectral;
        self
    }

    /// Sets the local connectivity parameter
    pub fn with_local_connectivity(mut self, local_connectivity: f64) -> Self {
        self.local_connectivity = local_connectivity.max(1.0);
        self
    }

    /// Sets the set operation mix ratio
    pub fn with_set_op_mix_ratio(mut self, ratio: f64) -> Self {
        self.set_op_mix_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Sets the spread parameter
    pub fn with_spread(mut self, spread: f64) -> Self {
        self.spread = spread;
        let (a, b) = Self::find_ab_params(spread, self.min_dist);
        self.a = a;
        self.b = b;
        self
    }

    /// Find a and b parameters to approximate the membership function
    ///
    /// We want: 1 / (1 + a * d^(2b)) to approximate
    ///   1.0 if d <= min_dist
    ///   exp(-(d - min_dist) / spread) if d > min_dist
    fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
        if min_dist <= 0.0 || spread <= 0.0 {
            return (1.0, 1.0);
        }

        let mut a = 1.0;
        let mut b = 1.0;

        // Use curve fitting approach: sample the target curve and fit a, b
        // Target: phi(d) = 1 if d <= min_dist, else exp(-(d - min_dist) / spread)
        // Model: psi(d) = 1 / (1 + a * d^(2b))
        // Minimize sum of (phi(d_i) - psi(d_i))^2

        // Initial guess based on analytical approximation
        if min_dist < spread {
            b = min_dist.ln().abs() / (1.0 - min_dist).ln().abs().max(1e-10);
            b = b.clamp(0.1, 10.0);
        }

        // Newton's method refinement
        for _ in 0..100 {
            let mut residual_a = 0.0;
            let mut residual_b = 0.0;
            let mut jacobian_aa = 0.0;
            let mut jacobian_bb = 0.0;

            let n_samples = 50;
            for k in 0..n_samples {
                let d = min_dist + (3.0 * spread) * (k as f64 / n_samples as f64);
                if d < 1e-10 {
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
                let diff = model - target;

                // Gradient w.r.t. a
                let da = -d2b / (denom * denom);
                // Gradient w.r.t. b
                let db = -2.0 * a * d2b * d.ln() / (denom * denom);

                residual_a += diff * da;
                residual_b += diff * db;
                jacobian_aa += da * da;
                jacobian_bb += db * db;
            }

            if jacobian_aa.abs() > 1e-15 {
                a -= 0.5 * residual_a / jacobian_aa;
            }
            if jacobian_bb.abs() > 1e-15 {
                b -= 0.5 * residual_b / jacobian_bb;
            }

            a = a.max(0.001);
            b = b.max(0.001);

            if residual_a.abs() < 1e-8 && residual_b.abs() < 1e-8 {
                break;
            }
        }

        (a, b)
    }

    /// Compute pairwise distances between all points
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = match self.metric.as_str() {
                    "manhattan" => {
                        let mut d = 0.0;
                        for k in 0..n_features {
                            let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                                - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                            d += diff.abs();
                        }
                        d
                    }
                    "cosine" => {
                        let mut dot = 0.0;
                        let mut norm_i = 0.0;
                        let mut norm_j = 0.0;
                        for k in 0..n_features {
                            let vi: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0);
                            let vj: f64 = NumCast::from(x[[j, k]]).unwrap_or(0.0);
                            dot += vi * vj;
                            norm_i += vi * vi;
                            norm_j += vj * vj;
                        }
                        let denom = (norm_i * norm_j).sqrt();
                        if denom > 1e-10 {
                            1.0 - (dot / denom).clamp(-1.0, 1.0)
                        } else {
                            1.0
                        }
                    }
                    _ => {
                        // Default: euclidean
                        let mut d = 0.0;
                        for k in 0..n_features {
                            let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                                - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                            d += diff * diff;
                        }
                        d.sqrt()
                    }
                };

                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Find k nearest neighbors for each point
    fn find_neighbors(&self, distances: &Array2<f64>) -> (Array2<usize>, Array2<f64>) {
        let n_samples = distances.shape()[0];
        let k = self.n_neighbors;

        let mut indices = Array2::zeros((n_samples, k));
        let mut neighbor_distances = Array2::zeros((n_samples, k));

        for i in 0..n_samples {
            let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

            for j in 0..n_samples {
                if i != j {
                    let dist_fixed = (distances[[i, j]] * 1e9) as i64;
                    heap.push((std::cmp::Reverse(dist_fixed), j));
                }
            }

            for j in 0..k {
                if let Some((std::cmp::Reverse(dist_fixed), idx)) = heap.pop() {
                    indices[[i, j]] = idx;
                    neighbor_distances[[i, j]] = dist_fixed as f64 / 1e9;
                }
            }
        }

        (indices, neighbor_distances)
    }

    /// Compute fuzzy simplicial set (high dimensional graph)
    ///
    /// For each point, compute the smooth k-NN distance (rho) and
    /// bandwidth (sigma) to convert distances to membership strengths.
    /// Then form the fuzzy union: A + A^T - A * A^T
    fn compute_graph(
        &self,
        knn_indices: &Array2<usize>,
        knn_distances: &Array2<f64>,
    ) -> Array2<f64> {
        let n_samples = knn_indices.shape()[0];
        let k = self.n_neighbors;
        let mut graph = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Compute rho: distance to the local_connectivity-th nearest neighbor
            // With local_connectivity = 1, rho = distance to 1st nearest neighbor
            let local_idx = (self.local_connectivity as usize)
                .saturating_sub(1)
                .min(k - 1);
            let rho = knn_distances[[i, local_idx]];

            // Binary search for sigma such that sum of memberships = log2(k)
            let target = (k as f64).ln() / (2.0f64).ln();
            let mut sigma_lo = 0.0;
            let mut sigma_hi = f64::INFINITY;
            let mut sigma = 1.0;

            for _ in 0..64 {
                let mut membership_sum = 0.0;
                for j in 0..k {
                    let d = (knn_distances[[i, j]] - rho).max(0.0);
                    if sigma > 1e-15 {
                        membership_sum += (-d / sigma).exp();
                    }
                }

                if (membership_sum - target).abs() < 1e-5 {
                    break;
                }

                if membership_sum > target {
                    sigma_hi = sigma;
                    sigma = (sigma_lo + sigma_hi) / 2.0;
                } else {
                    sigma_lo = sigma;
                    if sigma_hi == f64::INFINITY {
                        sigma *= 2.0;
                    } else {
                        sigma = (sigma_lo + sigma_hi) / 2.0;
                    }
                }
            }

            // Compute membership strengths
            for j in 0..k {
                let neighbor_idx = knn_indices[[i, j]];
                let d = (knn_distances[[i, j]] - rho).max(0.0);
                let strength = if sigma > 1e-15 {
                    (-d / sigma).exp()
                } else if d < 1e-15 {
                    1.0
                } else {
                    0.0
                };
                graph[[i, neighbor_idx]] = strength;
            }
        }

        // Symmetrize using the fuzzy set union:
        // union(A, B) = A + B - A * B
        // With mix ratio: mix * union + (1 - mix) * intersection
        let graph_t = graph.t().to_owned();

        if (self.set_op_mix_ratio - 1.0).abs() < 1e-10 {
            // Pure union
            &graph + &graph_t - &graph * &graph_t
        } else if self.set_op_mix_ratio.abs() < 1e-10 {
            // Pure intersection
            &graph * &graph_t
        } else {
            // Mixed
            let union = &graph + &graph_t - &graph * &graph_t;
            let intersection = &graph * &graph_t;
            &intersection * (1.0 - self.set_op_mix_ratio) + &union * self.set_op_mix_ratio
        }
    }

    /// Initialize the low dimensional embedding using spectral method
    fn initialize_embedding(&self, n_samples: usize, graph: &Array2<f64>) -> Result<Array2<f64>> {
        if self.spectral_init && n_samples > self.n_components + 1 {
            // Spectral initialization using normalized Laplacian eigenvectors
            match self.spectral_init_from_graph(n_samples, graph) {
                Ok(embedding) => return Ok(embedding),
                Err(_) => {
                    // Fall back to random initialization
                }
            }
        }

        // Random initialization
        let mut rng = scirs2_core::random::rng();
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.random_range(0.0..1.0) * 10.0 - 5.0;
            }
        }

        Ok(embedding)
    }

    /// Spectral initialization from the fuzzy simplicial set graph
    fn spectral_init_from_graph(
        &self,
        n_samples: usize,
        graph: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        // Compute degree matrix
        let mut degree = Array1::zeros(n_samples);
        for i in 0..n_samples {
            degree[i] = graph.row(i).sum();
        }

        // Check for isolated nodes
        for i in 0..n_samples {
            if degree[i] < 1e-10 {
                return Err(TransformError::ComputationError(
                    "Graph has isolated nodes, cannot use spectral initialization".to_string(),
                ));
            }
        }

        // Compute normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
        let mut laplacian = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i == j {
                    laplacian[[i, j]] = 1.0;
                } else {
                    let norm_weight = graph[[i, j]] / (degree[i] * degree[j]).sqrt();
                    laplacian[[i, j]] = -norm_weight;
                }
            }
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) =
            eigh(&laplacian.view(), None).map_err(|e| TransformError::LinalgError(e))?;

        // Sort eigenvalues in ascending order and pick eigenvectors 1..n_components+1
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j + 1]; // Skip the first (constant) eigenvector
            let scale = 10.0; // Scale for spread
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvectors[[i, idx]] * scale;
            }
        }

        Ok(embedding)
    }

    /// Optimize the low dimensional embedding using SGD with negative sampling
    fn optimize_embedding(
        &self,
        embedding: &mut Array2<f64>,
        graph: &Array2<f64>,
        n_epochs: usize,
    ) {
        let n_samples = embedding.shape()[0];
        let mut rng = scirs2_core::random::rng();

        // Create edge list from graph with weights
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..n_samples {
            for j in 0..n_samples {
                if graph[[i, j]] > 0.0 {
                    edges.push((i, j));
                    weights.push(graph[[i, j]]);
                }
            }
        }

        let n_edges = edges.len();
        if n_edges == 0 {
            return;
        }

        // Compute epochs per sample based on weights
        let max_weight = weights.iter().cloned().fold(0.0f64, f64::max);
        let epochs_per_sample: Vec<f64> = if max_weight > 0.0 {
            weights
                .iter()
                .map(|&w| {
                    let epoch_ratio = max_weight / w.max(1e-10);
                    epoch_ratio.min(n_epochs as f64)
                })
                .collect()
        } else {
            vec![1.0; n_edges]
        };

        let mut epochs_per_negative_sample: Vec<f64> = epochs_per_sample
            .iter()
            .map(|&e| e / self.negative_sample_rate as f64)
            .collect();

        let mut epoch_of_next_sample: Vec<f64> = epochs_per_sample.clone();
        let mut epoch_of_next_negative_sample: Vec<f64> = epochs_per_negative_sample.clone();

        // Clipping constant for gradient
        let clip_val = 4.0;

        // Optimization loop
        for epoch in 0..n_epochs {
            let alpha = self.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);

            for edge_idx in 0..n_edges {
                if epoch_of_next_sample[edge_idx] > epoch as f64 {
                    continue;
                }

                let (i, j) = edges[edge_idx];

                // Compute distance in embedding space
                let mut dist_sq = 0.0;
                for d in 0..self.n_components {
                    let diff = embedding[[i, d]] - embedding[[j, d]];
                    dist_sq += diff * diff;
                }
                dist_sq = dist_sq.max(1e-10);

                // Attractive force
                let grad_coeff = -2.0 * self.a * self.b * dist_sq.powf(self.b - 1.0)
                    / (1.0 + self.a * dist_sq.powf(self.b));

                for d in 0..self.n_components {
                    let grad = (grad_coeff * (embedding[[i, d]] - embedding[[j, d]]))
                        .clamp(-clip_val, clip_val);
                    embedding[[i, d]] += alpha * grad;
                    embedding[[j, d]] -= alpha * grad;
                }

                // Update next sample epoch
                epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

                // Negative sampling
                let n_neg = self.negative_sample_rate;
                for _ in 0..n_neg {
                    if epoch_of_next_negative_sample[edge_idx] > epoch as f64 {
                        break;
                    }

                    let k = rng.random_range(0..n_samples);
                    if k == i {
                        continue;
                    }

                    let mut neg_dist_sq = 0.0;
                    for d in 0..self.n_components {
                        let diff = embedding[[i, d]] - embedding[[k, d]];
                        neg_dist_sq += diff * diff;
                    }
                    neg_dist_sq = neg_dist_sq.max(1e-10);

                    // Repulsive force
                    let grad_coeff = 2.0 * self.b
                        / ((0.001 + neg_dist_sq) * (1.0 + self.a * neg_dist_sq.powf(self.b)));

                    for d in 0..self.n_components {
                        let grad = (grad_coeff * (embedding[[i, d]] - embedding[[k, d]]))
                            .clamp(-clip_val, clip_val);
                        embedding[[i, d]] += alpha * grad;
                    }

                    epoch_of_next_negative_sample[edge_idx] += epochs_per_negative_sample[edge_idx];
                }
            }
        }
    }

    /// Fits the UMAP model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        let (n_samples, n_features) = x.dim();

        check_positive(self.n_neighbors, "n_neighbors")?;
        check_positive(self.n_components, "n_components")?;
        check_positive(self.n_epochs, "n_epochs")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if n_samples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be <= n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        // Store training data
        let training_data = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
            NumCast::from(x[[i, j]]).unwrap_or(0.0)
        });
        self.training_data = Some(training_data);

        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(x);

        // Step 2: Find k nearest neighbors
        let (knn_indices, knn_distances) = self.find_neighbors(&distances);

        // Step 3: Compute fuzzy simplicial set
        let graph = self.compute_graph(&knn_indices, &knn_distances);
        self.training_graph = Some(graph.clone());

        // Step 4: Initialize low dimensional embedding
        let mut embedding = self.initialize_embedding(n_samples, &graph)?;

        // Step 5: Optimize the embedding
        self.optimize_embedding(&mut embedding, &graph, self.n_epochs);

        self.embedding = Some(embedding);

        Ok(())
    }

    /// Transforms the input data using the fitted UMAP model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::NotFitted(
                "UMAP model has not been fitted".to_string(),
            ));
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let (_, n_features) = x.dim();
        let (_, n_training_features) = training_data.dim();

        if n_features != n_training_features {
            return Err(TransformError::InvalidInput(format!(
                "Input features {n_features} must match training features {n_training_features}"
            )));
        }

        // If transforming the same data as training, return stored embedding
        if self.is_same_data(x, training_data) {
            return self
                .embedding
                .as_ref()
                .cloned()
                .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()));
        }

        // Out-of-sample extension
        self.transform_new_data(x)
    }

    /// Fits the UMAP model to the input data and returns the embedding
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the low dimensional embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the fuzzy simplicial set graph
    pub fn graph(&self) -> Option<&Array2<f64>> {
        self.training_graph.as_ref()
    }

    /// Check if the input data is the same as training data
    fn is_same_data<S>(&self, x: &ArrayBase<S, Ix2>, training_data: &Array2<f64>) -> bool
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if x.dim() != training_data.dim() {
            return false;
        }

        let (n_samples, n_features) = x.dim();
        for i in 0..n_samples {
            for j in 0..n_features {
                let x_val: f64 = NumCast::from(x[[i, j]]).unwrap_or(0.0);
                if (x_val - training_data[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Transform new data using out-of-sample extension (inverse distance weighting)
    fn transform_new_data<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let training_embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()))?;

        let (n_new_samples, _) = x.dim();
        let (n_training_samples, _) = training_data.dim();

        let mut new_embedding = Array2::zeros((n_new_samples, self.n_components));

        for i in 0..n_new_samples {
            // Compute distances to all training samples
            let mut distances: Vec<(f64, usize)> = Vec::with_capacity(n_training_samples);
            for j in 0..n_training_samples {
                let mut dist_sq = 0.0;
                for k in 0..x.ncols() {
                    let x_val: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0);
                    let train_val = training_data[[j, k]];
                    let diff = x_val - train_val;
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), j));
            }

            // Sort and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let k = self.n_neighbors.min(n_training_samples);

            // Inverse distance weighting
            let mut total_weight = 0.0;
            let mut weighted_coords = vec![0.0; self.n_components];

            for &(dist, train_idx) in distances.iter().take(k) {
                let weight = if dist > 1e-10 {
                    1.0 / (dist + 1e-10)
                } else {
                    1e10
                };
                total_weight += weight;

                for dim in 0..self.n_components {
                    weighted_coords[dim] += weight * training_embedding[[train_idx, dim]];
                }
            }

            if total_weight > 0.0 {
                for dim in 0..self.n_components {
                    new_embedding[[i, dim]] = weighted_coords[dim] / total_weight;
                }
            }
        }

        Ok(new_embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_umap_basic() {
        let x = Array::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 5.2,
                6.2, 7.2, 9.0, 10.0, 11.0, 9.1, 10.1, 11.1, 9.2, 10.2, 11.2, 9.3, 10.3, 11.3,
            ],
        )
        .expect("Failed to create test array");

        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50);
        let embedding = umap.fit_transform(&x).expect("UMAP fit_transform failed");

        assert_eq!(embedding.shape(), &[10, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_parameters() {
        let x: Array2<f64> = Array::eye(5);

        let mut umap = UMAP::new(2, 3, 0.5, 0.5, 100)
            .with_random_state(42)
            .with_metric("euclidean");

        let embedding = umap.fit_transform(&x).expect("UMAP fit_transform failed");
        assert_eq!(embedding.shape(), &[5, 3]);
    }

    #[test]
    fn test_umap_spectral_init() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0,
            ],
        )
        .expect("Failed to create test array");

        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50).with_spectral_init(true);
        let embedding = umap.fit_transform(&x).expect("UMAP fit_transform failed");

        assert_eq!(embedding.shape(), &[8, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_random_init() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0,
            ],
        )
        .expect("Failed to create test array");

        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50).with_spectral_init(false);
        let embedding = umap.fit_transform(&x).expect("UMAP fit_transform failed");

        assert_eq!(embedding.shape(), &[8, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_negative_sampling() {
        let x = Array::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 5.0, 5.0, 5.0, 6.0, 6.0, 5.0, 6.0, 6.0,
            ],
        )
        .expect("Failed to create test array");

        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50).with_negative_sample_rate(10);
        let embedding = umap.fit_transform(&x).expect("UMAP fit_transform failed");

        assert_eq!(embedding.shape(), &[8, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_out_of_sample() {
        let x_train = Array::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 5.2,
                6.2, 7.2, 9.0, 10.0, 11.0, 9.1, 10.1, 11.1, 9.2, 10.2, 11.2, 9.3, 10.3, 11.3,
            ],
        )
        .expect("Failed to create test array");

        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50);
        umap.fit(&x_train).expect("UMAP fit failed");

        let x_test = Array::from_shape_vec((2, 3), vec![1.05, 2.05, 3.05, 9.05, 10.05, 11.05])
            .expect("Failed to create test array");

        let test_embedding = umap.transform(&x_test).expect("UMAP transform failed");
        assert_eq!(test_embedding.shape(), &[2, 2]);
        for val in test_embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_find_ab_params() {
        let (a, b) = UMAP::find_ab_params(1.0, 0.1);
        assert!(a > 0.0);
        assert!(b > 0.0);

        // The function 1/(1+a*d^(2b)) should be close to 1 at d=0
        let val_at_zero = 1.0 / (1.0 + a * 0.0f64.powf(2.0 * b));
        assert!((val_at_zero - 1.0).abs() < 1e-5);
    }
}
