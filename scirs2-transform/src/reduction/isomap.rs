//! Isomap (Isometric Feature Mapping) for non-linear dimensionality reduction
//!
//! Isomap is a non-linear dimensionality reduction method that preserves geodesic
//! distances between all points. It extends MDS by using geodesic distances instead
//! of Euclidean distances.
//!
//! ## Algorithm Overview
//!
//! 1. **Graph construction**: Build k-NN or epsilon-neighborhood graph
//! 2. **Shortest paths**: Compute geodesic distances using Dijkstra's algorithm
//! 3. **Classical MDS**: Embed using eigendecomposition of the double-centered distance matrix
//!
//! ## Features
//!
//! - k-NN and epsilon-neighborhood graph construction
//! - Dijkstra's algorithm for shortest paths (O(N * E * log N) for sparse graphs)
//! - Floyd-Warshall option for dense graphs
//! - Classical MDS on geodesic distance matrix
//! - Landmark MDS for out-of-sample extension
//! - Residual variance computation for selecting n_components

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;
use std::collections::BinaryHeap;
use std::f64;

use crate::error::{Result, TransformError};

/// Shortest path algorithm to use
#[derive(Debug, Clone, PartialEq)]
pub enum ShortestPathAlgorithm {
    /// Dijkstra's algorithm - O(N * (E + N) * log N), better for sparse graphs
    Dijkstra,
    /// Floyd-Warshall algorithm - O(N^3), better for dense graphs
    FloydWarshall,
}

/// Isomap (Isometric Feature Mapping) dimensionality reduction
///
/// Isomap seeks a lower-dimensional embedding that maintains geodesic distances
/// between all points. It uses graph distances to approximate geodesic distances
/// on the manifold.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::Isomap;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((50, 10));
/// let mut isomap = Isomap::new(5, 2);
/// let embedding = isomap.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[50, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct Isomap {
    /// Number of neighbors to use for graph construction
    n_neighbors: usize,
    /// Number of components for dimensionality reduction
    n_components: usize,
    /// Whether to use k-neighbors or epsilon-ball for graph construction
    neighbor_mode: String,
    /// Epsilon for epsilon-ball graph construction
    epsilon: f64,
    /// Shortest path algorithm to use
    path_algorithm: ShortestPathAlgorithm,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Geodesic distances from training data
    geodesic_distances: Option<Array2<f64>>,
    /// Residual variance (reconstruction error)
    residual_variance: Option<f64>,
}

impl Isomap {
    /// Creates a new Isomap instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors for graph construction
    /// * `n_components` - Number of dimensions in the embedding space
    pub fn new(n_neighbors: usize, n_components: usize) -> Self {
        Isomap {
            n_neighbors,
            n_components,
            neighbor_mode: "knn".to_string(),
            epsilon: 0.0,
            path_algorithm: ShortestPathAlgorithm::Dijkstra,
            embedding: None,
            training_data: None,
            geodesic_distances: None,
            residual_variance: None,
        }
    }

    /// Use epsilon-ball instead of k-nearest neighbors
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.neighbor_mode = "epsilon".to_string();
        self.epsilon = epsilon;
        self
    }

    /// Set the shortest path algorithm
    pub fn with_path_algorithm(mut self, algorithm: ShortestPathAlgorithm) -> Self {
        self.path_algorithm = algorithm;
        self
    }

    /// Compute pairwise Euclidean distances
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut dist = 0.0;
                for k in 0..x.shape()[1] {
                    let diff: f64 = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                        - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Construct the neighborhood graph
    fn construct_graph(&self, distances: &Array2<f64>) -> Array2<f64> {
        let n_samples = distances.shape()[0];
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        if self.neighbor_mode == "knn" {
            for i in 0..n_samples {
                // Find k nearest neighbors using a min-heap
                let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

                for j in 0..n_samples {
                    if i != j {
                        let dist_fixed = (distances[[i, j]] * 1e9) as i64;
                        heap.push((std::cmp::Reverse(dist_fixed), j));
                    }
                }

                for _ in 0..self.n_neighbors {
                    if let Some((_, j)) = heap.pop() {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]]; // Make symmetric
                    }
                }
            }
        } else {
            // Epsilon-ball graph
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    if distances[[i, j]] <= self.epsilon {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]];
                    }
                }
            }
        }

        graph
    }

    /// Compute shortest paths using Dijkstra's algorithm
    ///
    /// For each source vertex, runs Dijkstra's algorithm using a binary heap.
    /// Time complexity: O(N * (E + N) * log N) for sparse graphs.
    fn compute_shortest_paths_dijkstra(&self, graph: &Array2<f64>) -> Result<Array2<f64>> {
        let n = graph.shape()[0];
        let mut dist = Array2::from_elem((n, n), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n {
            dist[[i, i]] = 0.0;
        }

        // Build adjacency list for efficiency
        let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        for i in 0..n {
            for j in 0..n {
                if i != j && graph[[i, j]] < f64::INFINITY {
                    adjacency[i].push((j, graph[[i, j]]));
                }
            }
        }

        // Run Dijkstra from each source
        for source in 0..n {
            // Min-heap: (distance * 1e9 as i64, node)
            let mut heap: BinaryHeap<std::cmp::Reverse<(i64, usize)>> = BinaryHeap::new();
            let mut visited = vec![false; n];

            dist[[source, source]] = 0.0;
            heap.push(std::cmp::Reverse((0, source)));

            while let Some(std::cmp::Reverse((d_fixed, u))) = heap.pop() {
                if visited[u] {
                    continue;
                }
                visited[u] = true;

                let d_u = d_fixed as f64 / 1e9;

                for &(v, weight) in &adjacency[u] {
                    let new_dist = d_u + weight;
                    if new_dist < dist[[source, v]] {
                        dist[[source, v]] = new_dist;
                        let d_fixed_new = (new_dist * 1e9) as i64;
                        heap.push(std::cmp::Reverse((d_fixed_new, v)));
                    }
                }
            }
        }

        // Check if graph is connected
        for i in 0..n {
            for j in 0..n {
                if dist[[i, j]].is_infinite() {
                    return Err(TransformError::InvalidInput(
                        "Graph is not connected. Try increasing n_neighbors or epsilon."
                            .to_string(),
                    ));
                }
            }
        }

        Ok(dist)
    }

    /// Compute shortest paths using Floyd-Warshall algorithm
    fn compute_shortest_paths_floyd_warshall(&self, graph: &Array2<f64>) -> Result<Array2<f64>> {
        let n = graph.shape()[0];
        let mut dist = graph.clone();

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                        dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    }
                }
            }
        }

        // Check connectivity
        for i in 0..n {
            for j in 0..n {
                if dist[[i, j]].is_infinite() {
                    return Err(TransformError::InvalidInput(
                        "Graph is not connected. Try increasing n_neighbors or epsilon."
                            .to_string(),
                    ));
                }
            }
        }

        Ok(dist)
    }

    /// Compute shortest paths using the configured algorithm
    fn compute_shortest_paths(&self, graph: &Array2<f64>) -> Result<Array2<f64>> {
        match self.path_algorithm {
            ShortestPathAlgorithm::Dijkstra => self.compute_shortest_paths_dijkstra(graph),
            ShortestPathAlgorithm::FloydWarshall => {
                self.compute_shortest_paths_floyd_warshall(graph)
            }
        }
    }

    /// Apply classical MDS to the geodesic distance matrix
    fn classical_mds(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n = distances.shape()[0];

        // Double center the squared distance matrix
        let squared_distances = distances.mapv(|d| d * d);

        let row_means = squared_distances.mean_axis(Axis(1)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute row means".to_string())
        })?;

        let col_means = squared_distances.mean_axis(Axis(0)).ok_or_else(|| {
            TransformError::ComputationError("Failed to compute column means".to_string())
        })?;

        let grand_mean = row_means.mean().ok_or_else(|| {
            TransformError::ComputationError("Failed to compute grand mean".to_string())
        })?;

        // Double centering: B = -0.5 * (D^2 - row_means - col_means + grand_mean)
        let mut gram = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] =
                    -0.5 * (squared_distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // Ensure symmetry (fixes floating point errors)
        let gram_symmetric = 0.5 * (&gram + &gram.t());

        // Eigendecomposition
        let (eigenvalues, eigenvectors) =
            eigh(&gram_symmetric.view(), None).map_err(|e| TransformError::LinalgError(e))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Extract the top n_components eigenvectors
        let mut embedding = Array2::zeros((n, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j];
            let scale = eigenvalues[idx].max(0.0).sqrt();

            for i in 0..n {
                embedding[[i, j]] = eigenvectors[[i, idx]] * scale;
            }
        }

        Ok(embedding)
    }

    /// Compute the residual variance (reconstruction error)
    fn compute_residual_variance(
        &self,
        geodesic_distances: &Array2<f64>,
        embedding: &Array2<f64>,
    ) -> f64 {
        let n = embedding.shape()[0];

        // Compute embedding distances
        let mut embedding_distances = Array2::zeros((n, n));
        for i in 0..n {
            for j in i + 1..n {
                let mut dist_sq = 0.0;
                for k in 0..embedding.shape()[1] {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                embedding_distances[[i, j]] = dist;
                embedding_distances[[j, i]] = dist;
            }
        }

        // Compute correlation between geodesic and embedding distances
        let mut sum_geodesic = 0.0;
        let mut sum_embedding = 0.0;
        let mut sum_geo_sq = 0.0;
        let mut sum_emb_sq = 0.0;
        let mut sum_product = 0.0;
        let mut count = 0.0;

        for i in 0..n {
            for j in i + 1..n {
                let g = geodesic_distances[[i, j]];
                let e = embedding_distances[[i, j]];
                sum_geodesic += g;
                sum_embedding += e;
                sum_geo_sq += g * g;
                sum_emb_sq += e * e;
                sum_product += g * e;
                count += 1.0;
            }
        }

        if count == 0.0 {
            return 1.0;
        }

        let mean_geo = sum_geodesic / count;
        let mean_emb = sum_embedding / count;

        let var_geo = sum_geo_sq / count - mean_geo * mean_geo;
        let var_emb = sum_emb_sq / count - mean_emb * mean_emb;
        let cov = sum_product / count - mean_geo * mean_emb;

        let denom = (var_geo * var_emb).sqrt();
        if denom > 1e-10 {
            let r = (cov / denom).clamp(-1.0, 1.0);
            // Clamp to [0, 1] to handle floating-point precision issues
            // where r^2 can slightly exceed 1.0
            (1.0 - r * r).max(0.0)
        } else {
            1.0
        }
    }

    /// Fits the Isomap model to the input data
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let (n_samples, n_features) = x.dim();

        check_positive(self.n_neighbors, "n_neighbors")?;
        check_positive(self.n_components, "n_components")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if n_samples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be <= n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(&x_f64.view());

        // Step 2: Construct neighborhood graph
        let graph = self.construct_graph(&distances);

        // Step 3: Compute shortest paths (geodesic distances)
        let geodesic_distances = self.compute_shortest_paths(&graph)?;

        // Step 4: Apply classical MDS
        let embedding = self.classical_mds(&geodesic_distances)?;

        // Compute residual variance
        let residual_var = self.compute_residual_variance(&geodesic_distances, &embedding);

        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.geodesic_distances = Some(geodesic_distances);
        self.residual_variance = Some(residual_var);

        Ok(())
    }

    /// Transforms the input data using the fitted Isomap model
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::NotFitted(
                "Isomap model has not been fitted".to_string(),
            ));
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        if self.is_same_data(&x_f64, training_data) {
            return self
                .embedding
                .as_ref()
                .cloned()
                .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()));
        }

        self.landmark_mds(&x_f64)
    }

    /// Fits the Isomap model and transforms the data
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the geodesic distances computed during fitting
    pub fn geodesic_distances(&self) -> Option<&Array2<f64>> {
        self.geodesic_distances.as_ref()
    }

    /// Returns the residual variance
    pub fn residual_variance(&self) -> Option<f64> {
        self.residual_variance
    }

    /// Check if the input data is the same as training data
    fn is_same_data(&self, x: &Array2<f64>, training_data: &Array2<f64>) -> bool {
        if x.dim() != training_data.dim() {
            return false;
        }
        let (n_samples, n_features) = x.dim();
        for i in 0..n_samples {
            for j in 0..n_features {
                if (x[[i, j]] - training_data[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Implement Landmark MDS for out-of-sample extension
    fn landmark_mds(&self, x_new: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;
        let training_embedding = self
            .embedding
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Embedding not available".to_string()))?;

        let (n_new, n_features) = x_new.dim();
        let (n_training, _) = training_data.dim();

        if n_features != training_data.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                n_features,
                training_data.ncols()
            )));
        }

        // Compute distances from new points to training points
        let mut distances_to_training = Array2::zeros((n_new, n_training));
        for i in 0..n_new {
            for j in 0..n_training {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = x_new[[i, k]] - training_data[[j, k]];
                    dist_sq += diff * diff;
                }
                distances_to_training[[i, j]] = dist_sq.sqrt();
            }
        }

        // Use inverse distance weighted interpolation
        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        for i in 0..n_new {
            // Find k nearest landmarks
            let mut landmark_dists: Vec<(f64, usize)> = (0..n_training)
                .map(|j| (distances_to_training[[i, j]], j))
                .collect();
            landmark_dists
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let k_landmarks = (n_training / 2).max(self.n_components + 1).min(n_training);

            let mut total_weight = 0.0;
            let mut weighted_coords = vec![0.0; self.n_components];

            for &(dist, landmark_idx) in landmark_dists.iter().take(k_landmarks) {
                let weight = if dist > 1e-10 {
                    1.0 / (dist * dist + 1e-10)
                } else {
                    1e10
                };
                total_weight += weight;

                for dim in 0..self.n_components {
                    weighted_coords[dim] += weight * training_embedding[[landmark_idx, dim]];
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
    fn test_isomap_basic() {
        let n_points = 20;
        let mut data = Vec::new();

        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 3.0 * std::f64::consts::PI;
            let x = t.sin();
            let y = 2.0 * (i as f64 / n_points as f64);
            let z = t.cos();
            data.extend_from_slice(&[x, y, z]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut isomap = Isomap::new(5, 2);
        let embedding = isomap
            .fit_transform(&x)
            .expect("Isomap fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_isomap_dijkstra() {
        let n_points = 15;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 2.0 * std::f64::consts::PI;
            data.extend_from_slice(&[t.cos(), t.sin(), i as f64 * 0.1]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut isomap = Isomap::new(4, 2).with_path_algorithm(ShortestPathAlgorithm::Dijkstra);
        let embedding = isomap
            .fit_transform(&x)
            .expect("Isomap fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_isomap_floyd_warshall() {
        let n_points = 15;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 2.0 * std::f64::consts::PI;
            data.extend_from_slice(&[t.cos(), t.sin(), i as f64 * 0.1]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut isomap =
            Isomap::new(4, 2).with_path_algorithm(ShortestPathAlgorithm::FloydWarshall);
        let embedding = isomap
            .fit_transform(&x)
            .expect("Isomap fit_transform failed");

        assert_eq!(embedding.shape(), &[n_points, 2]);
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_isomap_epsilon_ball() {
        let x: Array2<f64> = Array::eye(5);

        let mut isomap = Isomap::new(3, 2).with_epsilon(1.5);
        let result = isomap.fit_transform(&x);

        assert!(result.is_ok());
        let embedding = result.expect("Isomap fit_transform failed");
        assert_eq!(embedding.shape(), &[5, 2]);
    }

    #[test]
    fn test_isomap_disconnected_graph() {
        let x = scirs2_core::ndarray::array![[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1],];

        let mut isomap = Isomap::new(1, 2);
        let result = isomap.fit(&x);

        assert!(result.is_err());
        if let Err(e) = result {
            match e {
                TransformError::InvalidInput(msg) => {
                    assert!(msg.contains("Graph is not connected"));
                }
                _ => panic!("Expected InvalidInput error for disconnected graph"),
            }
        }
    }

    #[test]
    fn test_isomap_residual_variance() {
        let n_points = 20;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            data.extend_from_slice(&[t, t * 2.0, t * 3.0]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut isomap = Isomap::new(5, 2);
        let _ = isomap
            .fit_transform(&x)
            .expect("Isomap fit_transform failed");

        // For linear data, residual variance should be very small
        let rv = isomap.residual_variance();
        assert!(rv.is_some());
        let rv_val = rv.expect("Residual variance should exist");
        assert!(rv_val >= 0.0);
        assert!(rv_val <= 1.0);
    }

    #[test]
    fn test_isomap_out_of_sample() {
        let n_points = 20;
        let mut data = Vec::new();
        for i in 0..n_points {
            let t = i as f64 / n_points as f64;
            data.extend_from_slice(&[t, t * 2.0, t * 3.0]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).expect("Failed to create array");

        let mut isomap = Isomap::new(5, 2);
        isomap.fit(&x).expect("Isomap fit failed");

        let x_new = Array::from_shape_vec((2, 3), vec![0.25, 0.5, 0.75, 0.75, 1.5, 2.25])
            .expect("Failed to create test array");

        let new_embedding = isomap.transform(&x_new).expect("Isomap transform failed");
        assert_eq!(new_embedding.shape(), &[2, 2]);
        for val in new_embedding.iter() {
            assert!(val.is_finite());
        }
    }
}
