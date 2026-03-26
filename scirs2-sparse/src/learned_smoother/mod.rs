//! Multigrid + Deep Learning Error Smoothers
//!
//! This module implements learned smoothers for multigrid methods, replacing
//! or augmenting classical smoothers (Jacobi, Gauss-Seidel) with trainable
//! parametric or neural-network-based error smoothers.
//!
//! # Smoother Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`LinearSmoother`] | Parametric W·r update trained via gradient descent |
//! | [`MLPSmoother`] | Per-node 2-layer MLP with shared weights (GNN-like) |
//!
//! # Integration
//!
//! [`HybridMultigridSolver`] wraps a 2-level multigrid V-cycle and replaces
//! the fine-level smoother with a learned smoother, falling back to classical
//! Jacobi when divergence is detected.
//!
//! # References
//!
//! - Katrutsa, Daulbaev, Oseledets (2020). "Deep multigrid."
//! - Greenfeld, Galun, Kimmel, Yavneh, Basri (2019). "Learning to optimize multigrid."

/// Types and configuration for learned smoothers.
pub mod types;

/// Parametric linear smoother trained via gradient descent.
pub mod linear_smoother;

/// Per-node MLP smoother with shared weights.
pub mod mlp_smoother;

/// Hybrid multigrid solver with learned smoother integration.
pub mod integration;

pub use integration::HybridMultigridSolver;
pub use linear_smoother::LinearSmoother;
pub use mlp_smoother::MLPSmoother;
pub use types::{
    LearnedSmootherConfig, SmootherMetrics, SmootherType, SmootherWeights, TrainingConfig,
};
