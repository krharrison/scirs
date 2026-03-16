//! Prelude module for scirs2-core random functionality
//!
//! This module provides the most commonly used items for random number generation,
//! following Rust conventions for prelude modules. Import this for quick access
//! to essential functionality.
//!
//! # Usage
//!
//! ```rust
//! use scirs2_core::random::prelude::*;
//! use ::ndarray::Ix2;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Now you have access to the most common random functionality
//! let mut rng = thread_rng();
//! let value = rng.sample(Uniform::new(0.0, 1.0)?);
//! let array = random_uniform_array(Ix2(100, 100), &mut rng);
//! # Ok(())
//! # }
//! ```

// Core random number generation
pub use crate::random::core::{seeded_rng, thread_rng, Random};

// Essential traits
#[allow(deprecated)]
pub use crate::random::{DistributionExt, Rng, RngCore, RngExt, SeedableRng, SliceRandom};

// Most common distributions from rand_distr (with compatibility aliases)
pub use crate::random::rand_distributions::{
    Bernoulli, Beta as BetaDist, Exp as Exponential, Gamma, Normal, Uniform,
};

// Additional distributions commonly used in scientific computing
pub use crate::random::{
    Binomial, Cauchy, ChiSquared, FisherF, LogNormal, Poisson, RandDirichlet as DirichletDist,
    StudentT, Weibull,
};

// Enhanced slice operations
pub use crate::random::slice_ops::ScientificSliceRandom;

// Sequence operations for compatibility
pub use crate::random::seq;

// Most commonly used specialized distributions
pub use crate::random::{Beta, Categorical, WeightedChoice};

// Essential array operations
pub use crate::random::{random_normal_array, random_uniform_array, OptimizedArrayRandom};

// High-level convenience functions
pub use crate::random::convenience::{boolean, normal, normal_array, uniform, uniform_array};

// Common type aliases for convenience
pub type ThreadRng = Random<rand::rngs::ThreadRng>;
pub type StdRng = Random<rand::rngs::StdRng>;
pub type UniformDist = Uniform<f64>;
pub type NormalDist = Normal<f64>;
