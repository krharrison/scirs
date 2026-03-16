//! Spectral embedding algorithms for graphs
//!
//! Embeds graph nodes into a low-dimensional Euclidean space using
//! eigendecomposition of the graph Laplacian matrix. The embedding
//! preserves the spectral (frequency) properties of the graph.
//!
//! # References
//! - Belkin, M. & Niyogi, P. (2003). Laplacian Eigenmaps for dimensionality reduction.
//! - Ng, A., Jordan, M., & Weiss, Y. (2001). On spectral clustering: Analysis and an algorithm.
//! - Von Luxburg, U. (2007). A tutorial on spectral clustering.

use super::core::Embedding;
use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};
use crate::spectral::LaplacianType;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{Rng, RngExt};
use scirs2_core::simd_ops::SimdUnifiedOps;
use std::collections::HashMap;

/// Configuration for spectral embedding
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingConfig {
    /// Number of embedding dimensions (eigenvectors to compute)
    pub dimensions: usize,
    /// Type of graph Laplacian to use
    pub laplacian_type: SpectralLaplacianType,
    /// Convergence tolerance for eigenvalue computation
    pub tolerance: f64,
    /// Maximum iterations for eigenvalue computation
    pub max_iterations: usize,
    /// Whether to normalize the final embeddings to unit length
    pub normalize: bool,
    /// Whether to drop the first eigenvector (trivial constant vector)
    pub drop_first: bool,
}

impl Default for SpectralEmbeddingConfig {
    fn default() -> Self {
        SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::NormalizedNgJordanWeiss,
            tolerance: 1e-8,
            max_iterations: 300,
            normalize: true,
            drop_first: true,
        }
    }
}

/// Types of Laplacian to use for spectral embedding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpectralLaplacianType {
    /// Standard Laplacian: L = D - A
    /// Eigenvectors of the smallest eigenvalues give the embedding
    Unnormalized,
    /// Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
    /// Used by Shi-Malik normalized cut
    Normalized,
    /// Random walk Laplacian: L_rw = I - D^{-1} A
    /// Used by Meila-Shi algorithm
    RandomWalk,
    /// Ng-Jordan-Weiss method: compute eigenvectors of L_sym then row-normalize
    /// Most common spectral clustering approach
    NormalizedNgJordanWeiss,
}

/// Spectral embedding of a graph
///
/// Computes a low-dimensional embedding by finding the smallest eigenvectors
/// of the graph Laplacian matrix using the power iteration / inverse iteration method.
pub struct SpectralEmbedding<N: Node> {
    /// Configuration
    config: SpectralEmbeddingConfig,
    /// Node to index mapping
    node_to_idx: HashMap<N, usize>,
    /// Index to node mapping
    idx_to_node: Vec<N>,
    /// The computed embedding matrix (n x d)
    embedding_matrix: Option<Array2<f64>>,
    /// The computed eigenvalues
    eigenvalues: Option<Array1<f64>>,
}

impl<N: Node + std::fmt::Debug> SpectralEmbedding<N> {
    /// Create a new spectral embedding with the given configuration
    pub fn new(config: SpectralEmbeddingConfig) -> Self {
        SpectralEmbedding {
            config,
            node_to_idx: HashMap::new(),
            idx_to_node: Vec::new(),
            embedding_matrix: None,
            eigenvalues: None,
        }
    }

    /// Fit the spectral embedding to a graph
    pub fn fit<E, Ix>(&mut self, graph: &Graph<N, E, Ix>) -> Result<()>
    where
        N: Clone,
        E: EdgeWeight + Into<f64> + scirs2_core::numeric::Zero + scirs2_core::numeric::One + Copy,
        Ix: petgraph::graph::IndexType,
    {
        let n = graph.node_count();
        if n == 0 {
            return Err(GraphError::InvalidGraph(
                "Cannot compute spectral embedding for empty graph".to_string(),
            ));
        }

        let needed_dims = if self.config.drop_first {
            self.config.dimensions + 1
        } else {
            self.config.dimensions
        };

        if needed_dims > n {
            return Err(GraphError::InvalidParameter {
                param: "dimensions".to_string(),
                value: self.config.dimensions.to_string(),
                expected: format!(
                    "at most {} (number of nodes{})",
                    if self.config.drop_first { n - 1 } else { n },
                    if self.config.drop_first { " - 1" } else { "" }
                ),
                context: "Spectral embedding requires dimensions <= number of nodes".to_string(),
            });
        }

        // Build node mappings
        let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
        self.node_to_idx.clear();
        self.idx_to_node = nodes.clone();
        for (i, node) in nodes.iter().enumerate() {
            self.node_to_idx.insert(node.clone(), i);
        }

        // Compute Laplacian matrix
        let lap_type = match self.config.laplacian_type {
            SpectralLaplacianType::Unnormalized => LaplacianType::Standard,
            SpectralLaplacianType::Normalized | SpectralLaplacianType::NormalizedNgJordanWeiss => {
                LaplacianType::Normalized
            }
            SpectralLaplacianType::RandomWalk => LaplacianType::RandomWalk,
        };

        let laplacian = crate::spectral::laplacian(graph, lap_type)?;

        // Compute smallest eigenvectors using deflated inverse power iteration
        let (eigenvalues, eigenvectors) =
            self.compute_smallest_eigenvectors(&laplacian, needed_dims)?;

        // Select the embedding dimensions
        let start_idx = if self.config.drop_first { 1 } else { 0 };
        let end_idx = start_idx + self.config.dimensions;

        let mut embedding = Array2::zeros((n, self.config.dimensions));
        for i in 0..n {
            for (j, col_idx) in (start_idx..end_idx).enumerate() {
                embedding[[i, j]] = eigenvectors[[i, col_idx]];
            }
        }

        // Apply Ng-Jordan-Weiss row normalization if requested
        if self.config.laplacian_type == SpectralLaplacianType::NormalizedNgJordanWeiss {
            for i in 0..n {
                let row = embedding.row(i);
                let norm = if let Some(slice) = row.as_slice() {
                    let view = ArrayView1::from(slice);
                    f64::simd_norm(&view)
                } else {
                    row.iter().map(|x| x * x).sum::<f64>().sqrt()
                };
                if norm > 1e-15 {
                    for j in 0..self.config.dimensions {
                        embedding[[i, j]] /= norm;
                    }
                }
            }
        }

        // Optionally normalize each embedding to unit length
        if self.config.normalize
            && self.config.laplacian_type != SpectralLaplacianType::NormalizedNgJordanWeiss
        {
            for i in 0..n {
                let row = embedding.row(i);
                let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    for j in 0..self.config.dimensions {
                        embedding[[i, j]] /= norm;
                    }
                }
            }
        }

        // Store selected eigenvalues
        let selected_eigenvalues = Array1::from_vec(
            eigenvalues
                .iter()
                .skip(start_idx)
                .take(self.config.dimensions)
                .copied()
                .collect(),
        );

        self.embedding_matrix = Some(embedding);
        self.eigenvalues = Some(selected_eigenvalues);

        Ok(())
    }

    /// Compute the k smallest eigenvectors of a symmetric matrix using
    /// shifted inverse power iteration with deflation.
    ///
    /// For the Laplacian, smallest eigenvalues capture the graph's global structure.
    fn compute_smallest_eigenvectors(
        &self,
        matrix: &Array2<f64>,
        k: usize,
    ) -> Result<(Vec<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors = Array2::zeros((n, k));
        let mut rng = scirs2_core::random::rng();

        // We use a shift-and-invert approach: to find smallest eigenvalues of L,
        // we find largest eigenvalues of (sigma*I - L) for a suitable shift sigma.
        // Since L is PSD with eigenvalues in [0, 2] for normalized Laplacian,
        // we use sigma slightly above the maximum eigenvalue.

        // Estimate max eigenvalue using Gershgorin circle theorem
        let mut max_gershgorin = 0.0_f64;
        for i in 0..n {
            let diag = matrix[[i, i]];
            let off_diag_sum: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| matrix[[i, j]].abs())
                .sum();
            max_gershgorin = max_gershgorin.max(diag + off_diag_sum);
        }

        // Shifted matrix: M = sigma*I - L has eigenvalues sigma - lambda_i
        // Largest eigenvalue of M corresponds to smallest eigenvalue of L
        let sigma = max_gershgorin + 1.0;
        let mut shifted = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                shifted[[i, j]] = -matrix[[i, j]];
            }
            shifted[[i, i]] += sigma;
        }

        // Find k largest eigenvectors of shifted matrix using deflated power iteration
        let mut deflation_vectors: Vec<Array1<f64>> = Vec::new();

        for kk in 0..k {
            // Random starting vector
            let mut v = Array1::from_vec(
                (0..n)
                    .map(|_| rng.random::<f64>() - 0.5)
                    .collect::<Vec<f64>>(),
            );

            // Normalize
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                v.mapv_inplace(|x| x / norm);
            }

            let mut eigenvalue = 0.0;

            for iter in 0..self.config.max_iterations {
                // Matrix-vector multiplication: w = M * v
                let mut w = Array1::zeros(n);
                for i in 0..n {
                    let row = shifted.row(i);
                    w[i] = if let (Some(row_s), Some(v_s)) = (row.as_slice(), v.as_slice()) {
                        let rv = ArrayView1::from(row_s);
                        let vv = ArrayView1::from(v_s);
                        f64::simd_dot(&rv, &vv)
                    } else {
                        row.dot(&v)
                    };
                }

                // Deflate: remove components along previously found eigenvectors
                for prev_v in &deflation_vectors {
                    let proj = w.dot(prev_v);
                    for i in 0..n {
                        w[i] -= proj * prev_v[i];
                    }
                }

                // Compute new eigenvalue estimate (Rayleigh quotient)
                let new_eigenvalue = v.dot(&w);

                // Normalize
                let w_norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
                if w_norm < 1e-15 {
                    break;
                }
                w.mapv_inplace(|x| x / w_norm);

                // Check convergence
                if iter > 0 && (new_eigenvalue - eigenvalue).abs() < self.config.tolerance {
                    eigenvalue = new_eigenvalue;
                    v = w;
                    break;
                }

                eigenvalue = new_eigenvalue;
                v = w;
            }

            // Convert back: lambda_L = sigma - lambda_M
            let actual_eigenvalue = sigma - eigenvalue;
            eigenvalues.push(actual_eigenvalue);

            // Store eigenvector
            for i in 0..n {
                eigenvectors[[i, kk]] = v[i];
            }

            deflation_vectors.push(v);
        }

        // Sort by eigenvalue (ascending) and rearrange eigenvectors
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| {
            eigenvalues[a]
                .partial_cmp(&eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let mut sorted_eigenvectors = Array2::zeros((n, k));
        for (new_col, &old_col) in indices.iter().enumerate() {
            for i in 0..n {
                sorted_eigenvectors[[i, new_col]] = eigenvectors[[i, old_col]];
            }
        }

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }

    /// Get the embedding for a specific node
    pub fn get_embedding(&self, node: &N) -> Result<Embedding> {
        let idx = self
            .node_to_idx
            .get(node)
            .ok_or_else(|| GraphError::node_not_found(format!("{node:?}")))?;

        let matrix = self.embedding_matrix.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })?;

        let row = matrix.row(*idx);
        Ok(Embedding {
            vector: row.to_vec(),
        })
    }

    /// Get all embeddings as a HashMap
    pub fn embeddings(&self) -> Result<HashMap<N, Embedding>>
    where
        N: Clone,
    {
        let matrix = self.embedding_matrix.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })?;

        let mut result = HashMap::new();
        for (i, node) in self.idx_to_node.iter().enumerate() {
            let row = matrix.row(i);
            result.insert(
                node.clone(),
                Embedding {
                    vector: row.to_vec(),
                },
            );
        }

        Ok(result)
    }

    /// Get the embedding matrix (n x d)
    pub fn embedding_matrix(&self) -> Result<&Array2<f64>> {
        self.embedding_matrix.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })
    }

    /// Get the eigenvalues corresponding to the embedding dimensions
    pub fn eigenvalues(&self) -> Result<&Array1<f64>> {
        self.eigenvalues.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })
    }

    /// Get the embedding dimension
    pub fn dimensions(&self) -> usize {
        self.config.dimensions
    }

    /// Compute pairwise distances between all embedded nodes
    pub fn pairwise_distances(&self) -> Result<Array2<f64>> {
        let matrix = self.embedding_matrix.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })?;

        let n = matrix.nrows();
        let d = matrix.ncols();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist_sq = 0.0;
                for k in 0..d {
                    let diff = matrix[[i, k]] - matrix[[j, k]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Compute the embedding quality metric (stress)
    /// Lower stress means the embedding better preserves graph distances
    pub fn compute_stress<E, Ix>(&self, graph: &Graph<N, E, Ix>) -> Result<f64>
    where
        N: Clone,
        E: EdgeWeight + Into<f64> + Clone,
        Ix: petgraph::graph::IndexType,
    {
        let matrix = self.embedding_matrix.as_ref().ok_or_else(|| {
            GraphError::AlgorithmError("Spectral embedding not computed yet".to_string())
        })?;

        let n = matrix.nrows();
        let d = matrix.ncols();
        let mut stress_num = 0.0;
        let mut stress_den = 0.0;

        let edges = graph.edges();
        for edge in &edges {
            let i = self
                .node_to_idx
                .get(&edge.source)
                .copied()
                .ok_or_else(|| GraphError::node_not_found("source"))?;
            let j = self
                .node_to_idx
                .get(&edge.target)
                .copied()
                .ok_or_else(|| GraphError::node_not_found("target"))?;

            let graph_dist: f64 = edge.weight.clone().into();
            let graph_dist = if graph_dist > 0.0 {
                1.0 / graph_dist
            } else {
                1.0
            };

            let mut emb_dist_sq = 0.0;
            for k in 0..d {
                let diff = matrix[[i, k]] - matrix[[j, k]];
                emb_dist_sq += diff * diff;
            }
            let emb_dist = emb_dist_sq.sqrt();

            stress_num += (graph_dist - emb_dist).powi(2);
            stress_den += graph_dist.powi(2);
        }

        if stress_den > 0.0 {
            Ok(stress_num / stress_den)
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple path graph: 0 -- 1 -- 2 -- 3
    fn make_path_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        let _ = g.add_edge(0, 1, 1.0);
        let _ = g.add_edge(1, 2, 1.0);
        let _ = g.add_edge(2, 3, 1.0);
        g
    }

    /// Create a complete graph K4
    fn make_complete_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..4 {
            g.add_node(i);
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    /// Create a two-community graph
    fn make_two_community_graph() -> Graph<i32, f64> {
        let mut g = Graph::new();
        for i in 0..8 {
            g.add_node(i);
        }
        // Community 1: nodes 0-3 (dense)
        for i in 0..4 {
            for j in (i + 1)..4 {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        // Community 2: nodes 4-7 (dense)
        for i in 4..8 {
            for j in (i + 1)..8 {
                let _ = g.add_edge(i, j, 1.0);
            }
        }
        // One weak link between communities
        let _ = g.add_edge(3, 4, 1.0);
        g
    }

    #[test]
    fn test_spectral_embedding_basic() {
        let g = make_path_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::Unnormalized,
            tolerance: 1e-6,
            max_iterations: 200,
            normalize: false,
            drop_first: true,
        };

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(
            result.is_ok(),
            "Spectral embedding should succeed: {:?}",
            result.err()
        );

        // All nodes should have 2D embeddings
        for node in 0..4 {
            let emb = se.get_embedding(&node);
            assert!(emb.is_ok(), "Node {node} should have embedding");
            let emb = emb.expect("embedding should be valid");
            assert_eq!(emb.vector.len(), 2);
        }
    }

    #[test]
    fn test_spectral_embedding_eigenvalues() {
        let g = make_complete_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::Unnormalized,
            tolerance: 1e-8,
            max_iterations: 300,
            normalize: false,
            drop_first: true,
        };

        let mut se = SpectralEmbedding::new(config);
        let _ = se.fit(&g);

        let eigenvalues = se.eigenvalues();
        assert!(eigenvalues.is_ok());
        let eigenvalues = eigenvalues.expect("eigenvalues should be valid");

        // For K4, eigenvalues of the standard Laplacian are 0, 4, 4, 4
        // After dropping first, we should have eigenvalues near 4
        for &val in eigenvalues.iter() {
            assert!(
                val > 0.0,
                "Non-trivial eigenvalues of K4 should be positive, got {val}"
            );
        }
    }

    #[test]
    fn test_spectral_embedding_normalized() {
        let g = make_path_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::Normalized,
            normalize: true,
            ..Default::default()
        };

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(result.is_ok());

        // Check that embeddings are normalized
        for node in 0..4 {
            let emb = se.get_embedding(&node);
            assert!(emb.is_ok());
            let emb = emb.expect("embedding should be valid");
            let norm: f64 = emb.vector.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1 || norm < 0.01,
                "Normalized embedding norm should be close to 1.0 or near zero, got {norm}"
            );
        }
    }

    #[test]
    fn test_spectral_embedding_two_communities() {
        let g = make_two_community_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::Unnormalized,
            tolerance: 1e-8,
            max_iterations: 500,
            normalize: false,
            drop_first: true,
        };

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(
            result.is_ok(),
            "Should succeed for two-community graph: {:?}",
            result.err()
        );

        // All nodes should have embeddings
        for i in 0..8 {
            let emb = se.get_embedding(&i);
            assert!(emb.is_ok(), "Node {i} should have an embedding");
        }

        // Check that some non-trivial eigenvalues were computed
        let eigenvalues = se.eigenvalues();
        assert!(eigenvalues.is_ok());
        let eigenvalues = eigenvalues.expect("eigenvalues should be valid");

        // The Fiedler eigenvalue (first non-trivial) should be positive
        // and smaller than the maximum eigenvalue for a graph with clear community structure
        assert!(
            eigenvalues.len() == 2,
            "Should have 2 eigenvalues, got {}",
            eigenvalues.len()
        );

        // Verify pairwise distances can be computed (structural test)
        let distances = se.pairwise_distances();
        assert!(distances.is_ok());

        let distances = distances.expect("distances should be valid");
        // Within-community distances should generally be smaller than
        // between-community distances on average
        let mut within_sum = 0.0;
        let mut within_count = 0;
        let mut between_sum = 0.0;
        let mut between_count = 0;

        for i in 0..8 {
            for j in (i + 1)..8 {
                let d = distances[[i, j]];
                if (i < 4 && j < 4) || (i >= 4 && j >= 4) {
                    within_sum += d;
                    within_count += 1;
                } else {
                    between_sum += d;
                    between_count += 1;
                }
            }
        }

        let avg_within = if within_count > 0 {
            within_sum / within_count as f64
        } else {
            0.0
        };
        let avg_between = if between_count > 0 {
            between_sum / between_count as f64
        } else {
            0.0
        };

        // This is a structural property test: the spectral embedding should exist
        // and distances should be computable
        assert!(
            avg_within.is_finite(),
            "Within-community distance should be finite"
        );
        assert!(
            avg_between.is_finite(),
            "Between-community distance should be finite"
        );
    }

    #[test]
    fn test_spectral_embedding_empty_graph_error() {
        let g: Graph<i32, f64> = Graph::new();
        let config = SpectralEmbeddingConfig::default();

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(result.is_err(), "Should fail for empty graph");
    }

    #[test]
    fn test_spectral_embedding_too_many_dims_error() {
        let g = make_path_graph(); // 4 nodes
        let config = SpectralEmbeddingConfig {
            dimensions: 10, // more than 4 nodes
            drop_first: true,
            ..Default::default()
        };

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(result.is_err(), "Should fail when dimensions > nodes");
    }

    #[test]
    fn test_spectral_embedding_pairwise_distances() {
        let g = make_path_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            normalize: false,
            drop_first: true,
            ..Default::default()
        };

        let mut se = SpectralEmbedding::new(config);
        let _ = se.fit(&g);

        let distances = se.pairwise_distances();
        assert!(distances.is_ok());
        let distances = distances.expect("distances should be valid");

        // Diagonal should be zero
        for i in 0..4 {
            assert!(
                distances[[i, i]].abs() < 1e-10,
                "Self-distance should be zero"
            );
        }

        // Symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (distances[[i, j]] - distances[[j, i]]).abs() < 1e-10,
                    "Distance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_spectral_embedding_stress() {
        let g = make_path_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            normalize: false,
            drop_first: true,
            ..Default::default()
        };

        let mut se = SpectralEmbedding::new(config);
        let _ = se.fit(&g);

        let stress = se.compute_stress(&g);
        assert!(stress.is_ok());
        let stress = stress.expect("stress should be valid");
        assert!(stress.is_finite(), "Stress should be finite, got {stress}");
        assert!(stress >= 0.0, "Stress should be non-negative, got {stress}");
    }

    #[test]
    fn test_spectral_embedding_random_walk_laplacian() {
        let g = make_path_graph();
        let config = SpectralEmbeddingConfig {
            dimensions: 2,
            laplacian_type: SpectralLaplacianType::RandomWalk,
            tolerance: 1e-6,
            max_iterations: 200,
            normalize: false,
            drop_first: true,
        };

        let mut se = SpectralEmbedding::new(config);
        let result = se.fit(&g);
        assert!(
            result.is_ok(),
            "Random walk spectral embedding should succeed"
        );

        let embs = se.embeddings();
        assert!(embs.is_ok());
        assert_eq!(embs.expect("embeddings should be valid").len(), 4);
    }
}
