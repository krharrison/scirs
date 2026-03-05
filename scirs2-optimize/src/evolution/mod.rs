//! Evolution strategy algorithms for global optimization
//!
//! This module provides population-based optimization algorithms inspired by
//! natural evolution, particularly the Covariance Matrix Adaptation Evolution
//! Strategy (CMA-ES) family.
//!
//! ## Algorithms
//!
//! - **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy with IPOP restart
//! - Step-size adaptation (cumulative step-size adaptation / CSA)
//! - Covariance matrix update (rank-1 and rank-mu updates)
//! - Population size adaptation
//! - IPOP-CMA-ES restart strategy (increasing population size)
//! - Boundary handling (reflection, projection, penalty)

pub mod cma_es;

pub use cma_es::{
    cma_es_minimize, BoundaryHandling, CmaEsOptions, CmaEsResult, CmaEsState, IpopCmaEs,
    RestartStrategy,
};
