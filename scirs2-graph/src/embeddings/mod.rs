//! Graph embedding algorithms and utilities
//!
//! This module provides graph embedding algorithms including Node2Vec, DeepWalk,
//! LINE, spectral embedding, and other representation learning methods for graphs.
//!
//! # Available Algorithms
//!
//! - **Node2Vec**: Biased random walks with p,q parameters + skip-gram learning
//! - **DeepWalk**: Uniform random walks + skip-gram (negative sampling or hierarchical softmax)
//! - **LINE**: Large-scale Information Network Embedding (first/second order proximity)
//! - **Spectral Embedding**: Eigendecomposition of the graph Laplacian

#![allow(missing_docs)]

// Core types and configurations
pub mod core;
pub mod negative_sampling;
pub mod random_walk;
pub mod types;

// Algorithm implementations
pub mod deepwalk;
pub mod line;
pub mod node2vec;
pub mod spectral_embedding;

// Re-export main types from types module
pub use types::{
    ContextPair, DeepWalkConfig, LearningRateSchedule, NegativeSamplingStrategy, Node2VecConfig,
    OptimizationConfig, OptimizerState, RandomWalk, TrainingMetrics,
};

// Re-export core functionality
pub use core::{Embedding, EmbeddingModel};

// Re-export negative sampling
pub use negative_sampling::NegativeSampler;

// Re-export random walk generation
pub use random_walk::RandomWalkGenerator;

// Re-export algorithm implementations
pub use deepwalk::{DeepWalk, DeepWalkMode};
pub use line::{LINEConfig, LINEOrder, LINE};
pub use node2vec::Node2Vec;
pub use spectral_embedding::{SpectralEmbedding, SpectralEmbeddingConfig, SpectralLaplacianType};

// Legacy API compatibility - re-export from simplified modules for backwards compatibility
#[allow(dead_code)]
pub use deepwalk::DeepWalk as BasicDeepWalk;
#[allow(dead_code)]
pub use node2vec::Node2Vec as BasicNode2Vec;
