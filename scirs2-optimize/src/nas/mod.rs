//! Neural Architecture Search (NAS) and AutoML algorithms.
//!
//! This module provides algorithms for automated neural architecture search
//! and hyperparameter optimization:
//!
//! - `search_space`: DAG-based architecture search space definitions
//! - `random_nas`: Random search baseline for NAS
//! - `evolutionary_nas`: AmoebaNet-style evolutionary NAS
//! - `differentiable`: DARTS differentiable architecture search
//! - `automl`: AutoML random search for hyperparameter optimization

pub mod automl;
pub mod differentiable;
pub mod evolutionary_nas;
pub mod random_nas;
pub mod search_space;

pub use automl::{AutoMLConfig, AutoMLOptimizer, AutoMLResult, HyperparamSpace, HyperparamValue};
pub use differentiable::DARTSSearch;
pub use evolutionary_nas::EvolutionaryNAS;
pub use random_nas::{ArchFitness, NASResult, ParamCountFitness, RandomNAS};
pub use search_space::{ArchEdge, ArchNode, Architecture, OpType, SearchSpace};
