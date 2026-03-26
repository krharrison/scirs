//! Large-scale graph partitioning algorithms.
//!
//! This module provides three complementary approaches to graph partitioning:
//!
//! - [`spectral`]: Spectral bisection via the Fiedler vector of the graph Laplacian.
//!   Best for moderate-sized graphs where eigenvector quality matters.
//!
//! - [`multilevel`]: METIS-style multilevel partitioning with Heavy-Edge Matching
//!   coarsening and Kernighan-Lin refinement. The recommended general-purpose method.
//!
//! - [`streaming`]: Linear Deterministic Greedy (LDG) one-pass partitioner for
//!   graphs too large to fit in memory, or arriving as edge streams.
//!
//! # Example
//!
//! ```rust,no_run
//! use scirs2_graph::partitioning::{PartitionConfig, PartitionMethod, multilevel_partition};
//! use scirs2_core::ndarray::Array2;
//!
//! let mut adj = Array2::<f64>::zeros((6, 6));
//! adj[[0, 1]] = 1.0; adj[[1, 0]] = 1.0;
//! adj[[1, 2]] = 1.0; adj[[2, 1]] = 1.0;
//! adj[[3, 4]] = 1.0; adj[[4, 3]] = 1.0;
//! adj[[4, 5]] = 1.0; adj[[5, 4]] = 1.0;
//! adj[[2, 3]] = 1.0; adj[[3, 2]] = 1.0; // bridge
//!
//! let config = PartitionConfig::default();
//! let result = multilevel_partition(&adj, &config).expect("partition failed");
//! assert_eq!(result.partition_sizes.len(), 2);
//! ```

pub mod multilevel;
pub mod spectral;
pub mod streaming;
pub mod types;

// Re-export primary types and functions
pub use multilevel::multilevel_partition;
pub use spectral::{fiedler_vector, spectral_bisect, spectral_partition};
pub use streaming::{evaluate_partition, hash_partition, streaming_partition};
pub use types::{PartitionConfig, PartitionMethod, PartitionResult};

// Re-export new adjacency-list k-way partitioning API
pub use multilevel::{
    multilevel_kway, recursive_bisection, CoarseningStrategy, KwayPartitionResult,
    PartitioningConfig, RefinementStrategy,
};

// Re-export new stateful streaming partitioner
pub use streaming::{StreamingPartitionAlgorithm, StreamingPartitionConfig, StreamingPartitioner};
