//! Nonparametric Bayesian models.
//!
//! This module provides:
//! - **Dirichlet Process**: CRP, stick-breaking, DP mixture models
//! - **Indian Buffet Process**: sparse latent feature allocation
//! - **Beta Process**: hazard process for survival and feature models
//! - **Gaussian Process (nonparametric)**: GP regression with MCMC hyperparameter marginalization

pub mod beta_process;
pub mod dirichlet_process;
pub mod gaussian_process_nonparam;
pub mod indian_buffet;

pub use beta_process::BetaProcess;
pub use dirichlet_process::{CRPSampler, DPMixture, StickBreaking};
pub use indian_buffet::{IBPSampler, IndianBuffetProcess};
