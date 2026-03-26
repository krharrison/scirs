//! Types for graph partitioning algorithms.
//!
//! This module defines configuration and result types used across
//! spectral, multilevel, and streaming graph partitioning methods.

use std::fmt;

/// Method used for graph partitioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PartitionMethod {
    /// Spectral bisection via Fiedler vector of the graph Laplacian.
    SpectralBisection,
    /// METIS-style multilevel partitioning with Kernighan-Lin refinement.
    MultilevelKL,
    /// Linear Deterministic Greedy streaming partitioner for very large graphs.
    Streaming,
}

impl fmt::Display for PartitionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpectralBisection => write!(f, "SpectralBisection"),
            Self::MultilevelKL => write!(f, "MultilevelKL"),
            Self::Streaming => write!(f, "Streaming"),
        }
    }
}

/// Configuration for graph partitioning.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of partitions (k-way). Must be >= 2.
    pub n_partitions: usize,
    /// Maximum allowed imbalance ratio (0.03 = 3% deviation from perfect balance).
    pub balance_tolerance: f64,
    /// Stop coarsening when the graph has fewer nodes than this threshold.
    pub coarsening_threshold: usize,
    /// Maximum number of Kernighan-Lin refinement passes per uncoarsening level.
    pub kl_max_passes: usize,
    /// Partitioning method to use.
    pub method: PartitionMethod,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            n_partitions: 2,
            balance_tolerance: 0.03,
            coarsening_threshold: 100,
            kl_max_passes: 10,
            method: PartitionMethod::MultilevelKL,
        }
    }
}

/// Result of a graph partitioning operation.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Partition assignment for each node (0-indexed partition IDs).
    pub assignments: Vec<usize>,
    /// Number of edges crossing partition boundaries.
    pub edge_cut: usize,
    /// Number of nodes in each partition.
    pub partition_sizes: Vec<usize>,
    /// Maximum imbalance: max deviation from perfect balance as a ratio.
    /// For example, 0.05 means the largest partition is 5% larger than ideal.
    pub imbalance: f64,
}

impl PartitionResult {
    /// Compute a `PartitionResult` from assignments and an adjacency matrix.
    pub(crate) fn from_assignments(
        assignments: &[usize],
        adj: &scirs2_core::ndarray::Array2<f64>,
        n_partitions: usize,
    ) -> Self {
        let n = assignments.len();
        let mut partition_sizes = vec![0usize; n_partitions];
        for &a in assignments {
            if a < n_partitions {
                partition_sizes[a] += 1;
            }
        }

        // Edge cut: count edges crossing partitions (each undirected edge counted once)
        let mut edge_cut = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if adj[[i, j]].abs() > 1e-15 && assignments[i] != assignments[j] {
                    edge_cut += 1;
                }
            }
        }

        // Imbalance: max deviation from perfect balance
        let ideal = n as f64 / n_partitions as f64;
        let imbalance = if ideal > 0.0 {
            partition_sizes
                .iter()
                .map(|&s| ((s as f64) - ideal).abs() / ideal)
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };

        Self {
            assignments: assignments.to_vec(),
            edge_cut,
            partition_sizes,
            imbalance,
        }
    }
}

impl fmt::Display for PartitionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PartitionResult {{ partitions: {}, edge_cut: {}, sizes: {:?}, imbalance: {:.4} }}",
            self.partition_sizes.len(),
            self.edge_cut,
            self.partition_sizes,
            self.imbalance
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_default_config() {
        let cfg = PartitionConfig::default();
        assert_eq!(cfg.n_partitions, 2);
        assert!((cfg.balance_tolerance - 0.03).abs() < 1e-10);
        assert_eq!(cfg.coarsening_threshold, 100);
        assert_eq!(cfg.kl_max_passes, 10);
        assert_eq!(cfg.method, PartitionMethod::MultilevelKL);
    }

    #[test]
    fn test_partition_result_from_assignments() {
        // 4-node path: 0-1-2-3
        let mut adj = Array2::<f64>::zeros((4, 4));
        adj[[0, 1]] = 1.0;
        adj[[1, 0]] = 1.0;
        adj[[1, 2]] = 1.0;
        adj[[2, 1]] = 1.0;
        adj[[2, 3]] = 1.0;
        adj[[3, 2]] = 1.0;

        let assignments = vec![0, 0, 1, 1];
        let result = PartitionResult::from_assignments(&assignments, &adj, 2);
        assert_eq!(result.partition_sizes, vec![2, 2]);
        assert_eq!(result.edge_cut, 1); // only edge 1-2 crosses
        assert!(result.imbalance < 1e-10); // perfectly balanced
    }

    #[test]
    fn test_partition_method_display() {
        assert_eq!(
            format!("{}", PartitionMethod::SpectralBisection),
            "SpectralBisection"
        );
        assert_eq!(format!("{}", PartitionMethod::MultilevelKL), "MultilevelKL");
        assert_eq!(format!("{}", PartitionMethod::Streaming), "Streaming");
    }

    #[test]
    fn test_partition_result_display() {
        let result = PartitionResult {
            assignments: vec![0, 0, 1, 1],
            edge_cut: 1,
            partition_sizes: vec![2, 2],
            imbalance: 0.0,
        };
        let s = format!("{}", result);
        assert!(s.contains("edge_cut: 1"));
        assert!(s.contains("partitions: 2"));
    }
}
