//! Latent variable models for dimensionality reduction and generative modeling.
//!
//! # Modules
//!
//! - [`bayesian_pca`]: Bayesian PCA with Automatic Relevance Determination (ARD)
//!   using variational Bayes inference.
//! - [`plsa`]: Probabilistic Latent Semantic Analysis (PLSA) for topic modeling
//!   via EM inference.

pub mod bayesian_pca;
pub mod plsa;

pub use bayesian_pca::{BayesianPCA, BayesianPCAConfig};
pub use plsa::{PLSAConfig, PLSAModel, fit_em};
