//! Bayesian clustering algorithms.
//!
//! This module provides two complementary fully Bayesian clustering approaches:
//!
//! - [`dpmm`]: Dirichlet Process Mixture Model with collapsed Gibbs sampling.
//! - [`gaussian_mixture_bayes`]: Variational Bayes GMM (Bishop 2006, Chapter 10).
//!
//! # When to Use Which
//!
//! | Method | Cluster count | Inference | Speed |
//! |--------|--------------|-----------|-------|
//! | DPMM   | Inferred     | MCMC (Gibbs) | Slower, exact asymptotically |
//! | VB-GMM | Fixed        | Variational  | Fast, approximate |

pub mod dpmm;
pub mod gaussian_mixture_bayes;

pub use dpmm::{DPMMConfig, DPMMState, DPMMMixture, NormalWishart};
pub use gaussian_mixture_bayes::{VBGMMConfig, VBGMMState, VBParams, vbgmm_fit};
