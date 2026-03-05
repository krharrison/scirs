//! Probabilistic clustering algorithms using Bayesian nonparametric models.
//!
//! This module provides three complementary probabilistic clustering algorithms:
//!
//! * [`dpgmm`] – Dirichlet Process Gaussian Mixture Model (DP-GMM) via
//!   variational Bayes inference with stick-breaking representation.
//!
//! * [`crp`] – Chinese Restaurant Process (CRP) sampler and Gibbs-sampling
//!   based CRP mixture inference, plus the generalized Pitman-Yor process.
//!
//! * [`variational_gmm`] – Full Variational Bayesian GMM with a fixed number
//!   of components, component pruning, and variational BIC model selection.
//!
//! # Overview
//!
//! Bayesian nonparametric mixture models allow the data itself to determine
//! the effective number of clusters. The Dirichlet Process is the canonical
//! prior over infinite mixture models: with concentration `α` it generates
//! on the order of `α log N` clusters for N observations.
//!
//! ## Example – DP-GMM
//!
//! ```rust
//! use scirs2_cluster::probabilistic::dpgmm::{DPGMMConfig, DPGMMModel};
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0, 0.1, 0.1, -0.1, 0.0,  0.0, -0.1,
//!     5.0, 5.0, 5.1, 4.9,  4.9, 5.1,  5.0,  5.0,
//! ]).expect("data");
//!
//! let cfg = DPGMMConfig::new(1.0, 8, 100);
//! let model = DPGMMModel::new(cfg);
//! let result = model.fit(data.view()).expect("fit");
//! let labels = result.predict(data.view()).expect("predict");
//! assert_eq!(labels.len(), 8);
//! ```
//!
//! ## Example – CRP
//!
//! ```rust
//! use scirs2_cluster::probabilistic::crp::CRPSampler;
//!
//! let sampler = CRPSampler::new(2.0);
//! let assignments = sampler.sample_seating(20).expect("sample");
//! assert_eq!(assignments.len(), 20);
//! ```
//!
//! ## Example – Variational GMM
//!
//! ```rust
//! use scirs2_cluster::probabilistic::variational_gmm::{VBGMMConfig, VBGMMModel};
//! use scirs2_core::ndarray::Array2;
//!
//! let data = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0, 0.1, 0.1, -0.1, 0.0,  0.0, -0.1,
//!     5.0, 5.0, 5.1, 4.9,  4.9, 5.1,  5.0,  5.0,
//! ]).expect("data");
//!
//! let cfg = VBGMMConfig::new(4, 100, 1e-5);
//! let model = VBGMMModel::new(cfg);
//! let result = model.fit_vbem(data.view()).expect("fit");
//! let labels = result.predict(data.view()).expect("predict");
//! assert_eq!(labels.len(), 8);
//! ```

pub mod dpgmm;
pub mod crp;
pub mod variational_gmm;

// Re-export commonly used types for convenience.
pub use dpgmm::{DPGMMConfig, DPGMMModel, DPGMMResult};
pub use crp::{CRPSampler, PitmanYorProcess};
pub use variational_gmm::{VBGMMConfig, VBGMMModel, VBGMMResult};
