//! Black-box optimization module.
//!
//! Provides model-based surrogate methods for optimizing expensive black-box
//! objective functions beyond the GP-based Bayesian optimization in
//! [`crate::bayesian`].
//!
//! # Modules
//!
//! - [`model_based`] -- Random Forest surrogate and SMAC algorithm

pub mod model_based;

pub use model_based::{
    ei_random_forest, RandomForestSurrogate, SmacConfig, SmacOptimizer, SmacResult,
};
