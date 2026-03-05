//! Hierarchical Bayesian models for multi-level data analysis.
//!
//! This module provides:
//! - **Hierarchical Linear Models**: varying intercepts/slopes via Gibbs sampling
//! - **Mixed Effects Models**: REML-estimated linear mixed effects
//! - **Latent Factor Models**: Bayesian factor analysis
//! - **Conjugate Hyperpriors**: Normal-Inverse-Gamma and Normal-Inverse-Wishart

pub mod hyperpriors;
pub mod latent_factors;
pub mod linear;
pub mod mixed_effects;

pub use hyperpriors::{HyperPrior, NormalInverseGamma, NormalInverseWishart};
pub use latent_factors::FactorAnalysisModel;
pub use linear::{HierarchicalLinearModel, HierarchicalLinearResult};
pub use mixed_effects::{MixedEffectsModel, RandomEffect};
