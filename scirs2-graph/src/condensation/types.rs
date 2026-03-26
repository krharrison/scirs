//! Types for graph condensation (dataset distillation for graphs).
//!
//! This module defines configuration, result, and quality metric types
//! used throughout the condensation pipeline.

use scirs2_core::ndarray::Array2;

/// Method used for graph condensation.
///
/// Each variant represents a different algorithmic approach to
/// reducing a graph while preserving its structural properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CondensationMethod {
    /// Greedy farthest-point sampling (coreset).
    /// Picks nodes that maximize minimum distance to the selected set.
    KCenter,
    /// Degree-weighted sampling with feature diversity.
    /// Combines structural importance (degree) with feature-space coverage.
    ImportanceSampling,
    /// Gradient matching between original and synthetic graphs.
    /// Optimises a small synthetic graph so that GNN gradients match
    /// those computed on the full graph.
    GradientMatching,
    /// Kernel herding selection.
    /// Greedily picks points that minimize the Maximum Mean Discrepancy
    /// (MMD) between the selected subset and the full dataset.
    Herding,
}

/// Configuration for graph condensation.
#[derive(Debug, Clone)]
pub struct CondensationConfig {
    /// Number of nodes in the condensed graph.
    pub target_nodes: usize,
    /// Method to use for condensation.
    pub method: CondensationMethod,
    /// Maximum number of iterations (for iterative methods such as GradientMatching).
    pub max_iterations: usize,
    /// Learning rate (for iterative methods such as GradientMatching).
    pub learning_rate: f64,
}

impl Default for CondensationConfig {
    fn default() -> Self {
        Self {
            target_nodes: 100,
            method: CondensationMethod::KCenter,
            max_iterations: 200,
            learning_rate: 0.01,
        }
    }
}

/// A condensed (distilled) graph produced by condensation.
#[derive(Debug, Clone)]
pub struct CondensedGraph {
    /// Adjacency matrix of the condensed graph (target_nodes x target_nodes).
    pub adjacency: Array2<f64>,
    /// Feature matrix of the condensed graph (target_nodes x feature_dim).
    pub features: Array2<f64>,
    /// Node labels in the condensed graph.
    pub labels: Vec<usize>,
    /// Mapping from condensed node index to original node index.
    /// For synthetic methods (e.g. GradientMatching) this maps to the
    /// nearest original node.
    pub source_mapping: Vec<usize>,
}

/// Quality metrics that measure how well the condensed graph
/// preserves the properties of the original graph.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// KL divergence between degree distributions of original and condensed graphs.
    /// Lower is better; 0.0 means identical distributions.
    pub degree_distribution_distance: f64,
    /// L2 distance between the top-k eigenvalues of the graph Laplacians.
    /// Lower is better; 0.0 means identical spectral properties.
    pub spectral_distance: f64,
    /// Fraction of original label classes that are present in the condensed graph.
    /// 1.0 means all labels are covered.
    pub label_coverage: f64,
}

/// Full result of a condensation operation.
#[derive(Debug, Clone)]
pub struct CondensationResult {
    /// The condensed graph.
    pub condensed: CondensedGraph,
    /// Compression ratio: original_nodes / condensed_nodes.
    pub compression_ratio: f64,
    /// Quality metrics comparing the condensed graph to the original.
    pub quality_metrics: QualityMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CondensationConfig::default();
        assert_eq!(config.target_nodes, 100);
        assert_eq!(config.method, CondensationMethod::KCenter);
        assert_eq!(config.max_iterations, 200);
        assert!((config.learning_rate - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_condensation_method_variants() {
        let methods = [
            CondensationMethod::KCenter,
            CondensationMethod::ImportanceSampling,
            CondensationMethod::GradientMatching,
            CondensationMethod::Herding,
        ];
        // Ensure all variants are distinct
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                assert_ne!(methods[i], methods[j]);
            }
        }
    }

    #[test]
    fn test_condensation_method_clone_and_copy() {
        let method = CondensationMethod::Herding;
        let cloned = method;
        let copied = method;
        assert_eq!(method, cloned);
        assert_eq!(method, copied);
    }

    #[test]
    fn test_condensed_graph_creation() {
        let adj = Array2::<f64>::zeros((3, 3));
        let features = Array2::<f64>::ones((3, 2));
        let labels = vec![0, 1, 0];
        let source_mapping = vec![0, 5, 10];

        let graph = CondensedGraph {
            adjacency: adj.clone(),
            features: features.clone(),
            labels: labels.clone(),
            source_mapping: source_mapping.clone(),
        };

        assert_eq!(graph.adjacency.nrows(), 3);
        assert_eq!(graph.adjacency.ncols(), 3);
        assert_eq!(graph.features.nrows(), 3);
        assert_eq!(graph.features.ncols(), 2);
        assert_eq!(graph.labels, vec![0, 1, 0]);
        assert_eq!(graph.source_mapping, vec![0, 5, 10]);
    }

    #[test]
    fn test_condensed_graph_clone() {
        let graph = CondensedGraph {
            adjacency: Array2::<f64>::eye(2),
            features: Array2::<f64>::ones((2, 3)),
            labels: vec![1, 2],
            source_mapping: vec![0, 1],
        };

        let cloned = graph.clone();
        assert_eq!(cloned.labels, graph.labels);
        assert_eq!(cloned.source_mapping, graph.source_mapping);
        assert_eq!(cloned.adjacency, graph.adjacency);
    }

    #[test]
    fn test_quality_metrics_creation() {
        let metrics = QualityMetrics {
            degree_distribution_distance: 0.05,
            spectral_distance: 0.1,
            label_coverage: 0.95,
        };

        assert!((metrics.degree_distribution_distance - 0.05).abs() < 1e-12);
        assert!((metrics.spectral_distance - 0.1).abs() < 1e-12);
        assert!((metrics.label_coverage - 0.95).abs() < 1e-12);
    }

    #[test]
    fn test_quality_metrics_perfect() {
        let metrics = QualityMetrics {
            degree_distribution_distance: 0.0,
            spectral_distance: 0.0,
            label_coverage: 1.0,
        };

        assert!((metrics.degree_distribution_distance).abs() < 1e-12);
        assert!((metrics.spectral_distance).abs() < 1e-12);
        assert!((metrics.label_coverage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_condensation_result_creation() {
        let result = CondensationResult {
            condensed: CondensedGraph {
                adjacency: Array2::<f64>::zeros((2, 2)),
                features: Array2::<f64>::ones((2, 4)),
                labels: vec![0, 1],
                source_mapping: vec![3, 7],
            },
            compression_ratio: 5.0,
            quality_metrics: QualityMetrics {
                degree_distribution_distance: 0.1,
                spectral_distance: 0.2,
                label_coverage: 1.0,
            },
        };

        assert!((result.compression_ratio - 5.0).abs() < 1e-12);
        assert_eq!(result.condensed.labels.len(), 2);
        assert!((result.quality_metrics.label_coverage - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_config_custom() {
        let config = CondensationConfig {
            target_nodes: 50,
            method: CondensationMethod::GradientMatching,
            max_iterations: 500,
            learning_rate: 0.001,
        };

        assert_eq!(config.target_nodes, 50);
        assert_eq!(config.method, CondensationMethod::GradientMatching);
        assert_eq!(config.max_iterations, 500);
        assert!((config.learning_rate - 0.001).abs() < 1e-12);
    }
}
