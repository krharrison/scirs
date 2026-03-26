//! Types for network alignment algorithms.
//!
//! This module provides core data structures for configuring and storing
//! results of network alignment operations, including similarity matrices
//! between nodes of two graphs.

use scirs2_core::ndarray::Array2;

use crate::error::{GraphError, Result};

/// Configuration for network alignment algorithms.
///
/// Controls the trade-off between topology and prior knowledge,
/// convergence criteria, and GRASP meta-heuristic parameters.
#[derive(Debug, Clone)]
pub struct AlignmentConfig {
    /// Weight for topology vs. prior similarity (0.0 = only prior, 1.0 = only topology).
    /// Default: 0.6
    pub alpha: f64,
    /// Maximum number of power iterations for IsoRank. Default: 100
    pub max_iter: usize,
    /// Convergence tolerance for power iteration. Default: 1e-8
    pub tolerance: f64,
    /// Size of the restricted candidate list for GRASP construction. Default: 5
    pub greedy_candidates: usize,
    /// Maximum number of local search iterations in GRASP. Default: 50
    pub local_search_depth: usize,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            alpha: 0.6,
            max_iter: 100,
            tolerance: 1e-8,
            greedy_candidates: 5,
            local_search_depth: 50,
        }
    }
}

/// Result of a network alignment computation.
///
/// Contains the node mapping between two graphs along with quality metrics.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Pairs of aligned nodes: `(node_in_g1, node_in_g2)`.
    pub mapping: Vec<(usize, usize)>,
    /// Overall alignment quality score (higher is better).
    pub score: f64,
    /// Fraction of edges in G1 that are preserved in the alignment to G2.
    pub edge_conservation: f64,
    /// Whether the algorithm converged within the specified tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
}

/// Similarity matrix between nodes of two graphs.
///
/// Stores an `[n1 x n2]` matrix where entry `(i, j)` represents the similarity
/// between node `i` in graph 1 and node `j` in graph 2.
#[derive(Debug, Clone)]
pub struct SimilarityMatrix {
    data: Array2<f64>,
    n1: usize,
    n2: usize,
}

impl SimilarityMatrix {
    /// Create a new similarity matrix initialized to uniform values `1 / (n1 * n2)`.
    ///
    /// # Errors
    ///
    /// Returns an error if either dimension is zero.
    pub fn new(n1: usize, n2: usize) -> Result<Self> {
        if n1 == 0 || n2 == 0 {
            return Err(GraphError::InvalidParameter {
                param: "dimensions".to_string(),
                value: format!("({}, {})", n1, n2),
                expected: "both dimensions > 0".to_string(),
                context: "SimilarityMatrix::new".to_string(),
            });
        }
        let val = 1.0 / (n1 as f64 * n2 as f64);
        let data = Array2::from_elem((n1, n2), val);
        Ok(Self { data, n1, n2 })
    }

    /// Create a similarity matrix from a prior similarity array.
    ///
    /// The prior is normalized so that all entries sum to 1.
    ///
    /// # Errors
    ///
    /// Returns an error if the prior has zero dimensions or all-zero entries.
    pub fn from_prior(prior: Array2<f64>) -> Result<Self> {
        let shape = prior.shape();
        let n1 = shape[0];
        let n2 = shape[1];
        if n1 == 0 || n2 == 0 {
            return Err(GraphError::InvalidParameter {
                param: "prior dimensions".to_string(),
                value: format!("({}, {})", n1, n2),
                expected: "both dimensions > 0".to_string(),
                context: "SimilarityMatrix::from_prior".to_string(),
            });
        }
        let mut sm = Self {
            data: prior,
            n1,
            n2,
        };
        sm.normalize();
        Ok(sm)
    }

    /// Get the similarity between node `i` in G1 and node `j` in G2.
    ///
    /// Returns 0.0 if indices are out of bounds.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.n1 && j < self.n2 {
            self.data[[i, j]]
        } else {
            0.0
        }
    }

    /// Set the similarity between node `i` in G1 and node `j` in G2.
    ///
    /// Does nothing if indices are out of bounds.
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i < self.n1 && j < self.n2 {
            self.data[[i, j]] = value;
        }
    }

    /// Normalize the matrix so that all entries sum to 1.
    ///
    /// If the sum is zero, sets all entries to uniform `1 / (n1 * n2)`.
    pub fn normalize(&mut self) {
        let sum: f64 = self.data.iter().sum();
        if sum.abs() < f64::EPSILON {
            let val = 1.0 / (self.n1 as f64 * self.n2 as f64);
            self.data.fill(val);
        } else {
            self.data /= sum;
        }
    }

    /// Return a reference to the underlying array.
    pub fn as_array(&self) -> &Array2<f64> {
        &self.data
    }

    /// Return the number of rows (nodes in G1).
    pub fn n1(&self) -> usize {
        self.n1
    }

    /// Return the number of columns (nodes in G2).
    pub fn n2(&self) -> usize {
        self.n2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_alignment_config_default() {
        let cfg = AlignmentConfig::default();
        assert!((cfg.alpha - 0.6).abs() < f64::EPSILON);
        assert_eq!(cfg.max_iter, 100);
        assert!((cfg.tolerance - 1e-8).abs() < f64::EPSILON);
        assert_eq!(cfg.greedy_candidates, 5);
        assert_eq!(cfg.local_search_depth, 50);
    }

    #[test]
    fn test_similarity_matrix_new() {
        let sm = SimilarityMatrix::new(3, 4).expect("should create matrix");
        let expected = 1.0 / 12.0;
        for i in 0..3 {
            for j in 0..4 {
                assert!((sm.get(i, j) - expected).abs() < f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_similarity_matrix_zero_dim() {
        assert!(SimilarityMatrix::new(0, 5).is_err());
        assert!(SimilarityMatrix::new(5, 0).is_err());
    }

    #[test]
    fn test_similarity_matrix_from_prior() {
        let prior = array![[1.0, 2.0], [3.0, 4.0]];
        let sm = SimilarityMatrix::from_prior(prior).expect("should create from prior");
        let sum: f64 = sm.as_array().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_similarity_matrix_set_get() {
        let mut sm = SimilarityMatrix::new(2, 2).expect("should create matrix");
        sm.set(0, 1, 0.99);
        assert!((sm.get(0, 1) - 0.99).abs() < f64::EPSILON);
        // Out-of-bounds returns 0.0
        assert!((sm.get(10, 10)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_matrix_normalize_zero() {
        let prior = Array2::zeros((3, 3));
        let sm = SimilarityMatrix::from_prior(prior).expect("should create from zero prior");
        let expected = 1.0 / 9.0;
        assert!((sm.get(0, 0) - expected).abs() < 1e-12);
    }
}
