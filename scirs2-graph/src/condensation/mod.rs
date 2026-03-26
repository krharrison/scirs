//! Graph condensation (dataset distillation for graphs).
//!
//! This module provides algorithms to reduce a large graph to a small
//! representative graph while preserving key structural and feature
//! properties. The condensed graph can be used as a proxy for the
//! original in downstream tasks such as GNN training.
//!
//! ## Approaches
//!
//! - **Coreset methods** ([`coreset`]): Select a representative subset of
//!   original nodes via k-center, importance sampling, or kernel herding.
//! - **Distillation** ([`distillation`]): Optimise a synthetic graph so
//!   that GNN gradients on the synthetic data match those on the full graph.
//! - **Evaluation** ([`evaluation`]): Measure condensation quality via
//!   degree-distribution distance, spectral distance, and label coverage.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use scirs2_graph::condensation::{
//!     coreset::{k_center_greedy, extract_subgraph},
//!     evaluation::evaluate_condensation,
//!     types::CondensationConfig,
//! };
//! ```

pub mod coreset;
pub mod distillation;
pub mod evaluation;
pub mod types;

// Re-export key types at the module level for convenience.
pub use types::{
    CondensationConfig, CondensationMethod, CondensationResult, CondensedGraph, QualityMetrics,
};
