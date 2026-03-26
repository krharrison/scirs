//! Variational Inference Methods
//!
//! This module provides modern variational inference algorithms for approximate
//! Bayesian posterior computation:
//!
//! - **ADVI**: Automatic Differentiation Variational Inference (Kucukelbir et al. 2017)
//!   with mean-field and full-rank Gaussian approximations, automatic parameter
//!   transformations, ELBO optimization via reparameterization trick + Adam optimizer.
//!
//! - **SVGD**: Stein Variational Gradient Descent (Liu & Wang 2016) — a particle-based
//!   method that transports a set of particles to approximate the posterior using
//!   kernelized Stein discrepancy with RBF kernel and median bandwidth heuristic.
//!
//! - **Normalizing Flows**: Flexible posterior approximations via invertible
//!   transformations (planar and radial flows) with log-determinant Jacobian tracking.

mod advi;
pub mod bbvi;
mod normalizing_flow;
mod svgd;

pub use advi::*;
pub use normalizing_flow::*;
pub use svgd::*;

use crate::error::StatsResult;
use scirs2_core::ndarray::Array1;

// ============================================================================
// Common Trait
// ============================================================================

/// Result of variational inference
#[derive(Debug, Clone)]
pub struct PosteriorResult {
    /// Posterior means (in constrained space)
    pub means: Array1<f64>,
    /// Posterior standard deviations (in constrained space)
    pub std_devs: Array1<f64>,
    /// ELBO history over iterations
    pub elbo_history: Vec<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Optional: posterior samples (for particle-based methods like SVGD)
    pub samples: Option<Vec<Array1<f64>>>,
}

/// Common trait for variational inference methods
pub trait VariationalInference {
    /// Fit the variational approximation to a target log-joint distribution.
    ///
    /// # Arguments
    /// * `log_joint` - Function computing `(log p(x, theta), grad_theta log p(x, theta))`
    /// * `dim` - Dimensionality of the parameter space
    ///
    /// # Returns
    /// A `PosteriorResult` with posterior statistics and convergence info
    fn fit<F>(&mut self, log_joint: F, dim: usize) -> StatsResult<PosteriorResult>
    where
        F: Fn(&Array1<f64>) -> StatsResult<(f64, Array1<f64>)>;
}
