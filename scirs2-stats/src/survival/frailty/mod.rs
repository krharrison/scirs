//! Frailty Models & Multilevel Survival Analysis
//!
//! This module provides shared and nested frailty models that extend the Cox PH
//! framework with cluster-level random effects to handle correlated survival data.
//!
//! - [`SharedFrailtyModel`] – Shared (single-level) frailty model with Gamma,
//!   LogNormal, or InverseGaussian frailty distributions, fitted via EM algorithm.
//! - [`NestedFrailtyModel`] – Two-level nested frailty (e.g., centers within regions).
//! - Prediction utilities: conditional/marginal survival, ICC, median survival.
//!
//! # References
//! - Hougaard, P. (2000). *Analysis of Multivariate Survival Data*. Springer.
//! - Duchateau, L. & Janssen, P. (2008). *The Frailty Model*. Springer.

pub mod nested_frailty;
pub mod prediction;
pub mod shared_frailty;
pub mod types;

pub use nested_frailty::{NestedFrailtyModel, NestedFrailtyResult};
pub use prediction::{
    conditional_survival, intraclass_correlation, marginal_survival, median_survival,
};
pub use shared_frailty::SharedFrailtyModel;
pub use types::{ClusterInfo, FrailtyConfig, FrailtyDistribution, FrailtyResult};
