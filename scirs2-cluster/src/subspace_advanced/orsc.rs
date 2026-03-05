//! Ordered Robust Subspace Clustering (ORSC).
//!
//! Extends Low-Rank Representation with an explicit rank constraint. In the
//! original formulation the user specifies the expected total rank `r` of the
//! union of subspaces, and the optimisation enforces a nuclear-norm penalty
//! calibrated by that rank.
//!
//! This implementation approximates ORSC by running LRR with a rank-adjusted
//! regularisation parameter derived from `rank` and the dataset size, then
//! applying normalized spectral clustering.

use crate::error::{ClusteringError, Result};
use crate::subspace_advanced::lrr::LowRankRepresentation;

/// Ordered Robust Subspace Clustering.
///
/// # Example
///
/// ```
/// use scirs2_cluster::subspace_advanced::OrderedRobustSC;
///
/// let data = vec![
///     vec![1.0, 0.0], vec![2.0, 0.0],
///     vec![0.0, 1.0], vec![0.0, 2.0],
/// ];
/// let labels = OrderedRobustSC::new(2, 1).fit(&data).expect("operation should succeed");
/// assert_eq!(labels.len(), 4);
/// ```
pub struct OrderedRobustSC {
    /// Number of output clusters.
    pub n_clusters: usize,
    /// Expected total rank of the union of subspaces.
    pub rank: usize,
    /// Maximum iterations for the underlying LRR solver.
    pub max_iter: usize,
}

impl OrderedRobustSC {
    /// Create a new `OrderedRobustSC`.
    pub fn new(n_clusters: usize, rank: usize) -> Self {
        Self {
            n_clusters,
            rank,
            max_iter: 100,
        }
    }

    /// Override max iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Fit ORSC and return cluster labels.
    ///
    /// The rank constraint is translated into an LRR regularisation weight
    /// `lambda = 1 / sqrt(max(n, d))` where `n` is the number of data points
    /// and `d` their dimensionality. The `rank` parameter modulates this weight
    /// so that smaller rank → stronger regularisation.
    ///
    /// # Errors
    ///
    /// Returns an error if `data` is empty, `n_clusters` exceeds the number of
    /// points, or `rank` is zero.
    pub fn fit(&self, data: &[Vec<f64>]) -> Result<Vec<usize>> {
        let n = data.len();
        if n == 0 {
            return Err(ClusteringError::InvalidInput(
                "input data must not be empty".to_string(),
            ));
        }
        if self.n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({}) exceeds number of data points ({})",
                self.n_clusters, n
            )));
        }
        if self.rank == 0 {
            return Err(ClusteringError::InvalidInput(
                "rank must be at least 1".to_string(),
            ));
        }

        let dim = data[0].len();
        // Rank-calibrated lambda: larger rank → smaller lambda (less regularisation).
        let lambda = 1.0 / ((n.max(dim) as f64).sqrt() * (self.rank as f64).sqrt());

        LowRankRepresentation::new(self.n_clusters, lambda)
            .with_max_iter(self.max_iter)
            .fit(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_subspace_data() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0],
        ]
    }

    #[test]
    fn test_orsc_basic() {
        let data = two_subspace_data();
        let labels = OrderedRobustSC::new(2, 1)
            .fit(&data)
            .expect("ORSC fit should succeed");
        assert_eq!(labels.len(), 6);
        for &l in &labels {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_orsc_empty_input() {
        let data: Vec<Vec<f64>> = vec![];
        let err = OrderedRobustSC::new(2, 1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_orsc_zero_rank() {
        let data = two_subspace_data();
        let err = OrderedRobustSC::new(2, 0).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_orsc_n_clusters_exceeds_n() {
        let data = vec![vec![1.0, 0.0]];
        let err = OrderedRobustSC::new(5, 1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn test_orsc_various_ranks() {
        let data = two_subspace_data();
        for rank in 1..=3 {
            let labels = OrderedRobustSC::new(2, rank)
                .fit(&data)
                .expect("ORSC with various ranks should succeed");
            assert_eq!(labels.len(), 6);
        }
    }
}
