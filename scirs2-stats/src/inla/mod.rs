//! Integrated Nested Laplace Approximation (INLA) for latent Gaussian models
//!
//! INLA is a fast approximate Bayesian inference method for latent Gaussian
//! models (LGMs). Instead of MCMC sampling, INLA uses deterministic
//! approximations (Laplace and numerical integration) to compute
//! posterior marginals.
//!
//! # Model class
//!
//! INLA targets models of the form:
//!   - **Observations**: y_i | x, θ ~ p(y_i | η_i, θ)  (conditionally independent)
//!   - **Latent field**: x | θ ~ N(0, Q(θ)^{-1})  (Gaussian Markov Random Field)
//!   - **Linear predictor**: η = A × x
//!   - **Hyperparameters**: θ ~ p(θ)
//!
//! # Usage
//!
//! ```rust,no_run
//! use scirs2_stats::inla::{
//!     LatentGaussianModel, INLAConfig, LikelihoodFamily,
//!     compute_marginals,
//! };
//! use scirs2_core::ndarray::{array, Array2};
//!
//! let y = array![1.0, 2.0, 3.0];
//! let design = Array2::eye(3);
//! let precision = Array2::eye(3);
//!
//! let model = LatentGaussianModel::new(
//!     y, design, precision,
//!     LikelihoodFamily::Gaussian,
//! ).with_observation_precision(1.0);
//!
//! let config = INLAConfig {
//!     n_hyperparameter_grid: 10,
//!     hyperparameter_range: Some((-1.0, 1.0)),
//!     ..INLAConfig::default()
//! };
//!
//! let result = compute_marginals(&model, &config).expect("INLA failed");
//! println!("Posterior means: {:?}", result.marginal_means);
//! println!("Posterior variances: {:?}", result.marginal_variances);
//! ```

pub mod gaussian_field;
pub mod hyperparameters;
pub mod laplace;
pub mod marginals;
pub mod model_builder;
pub mod types;

pub use gaussian_field::{
    build_precision_matrix, kronecker_precision, validate_field_params, LatentFieldType,
};
pub use hyperparameters::{
    ccd_integration_points, explore_hyperparameter_grid, grid_integration,
    summarize_hyperparameter_posterior, HyperparameterPoint,
};
pub use laplace::{
    compute_neg_hessian, find_mode, full_inverse, inverse_diagonal,
    laplace_log_marginal_likelihood, log_likelihood, log_posterior_gradient, ModeResult,
};
pub use marginals::{compute_marginals, corrected_laplace_marginal, fit_inla};
pub use model_builder::INLAModelBuilder;
pub use types::{
    HyperparameterPosterior, INLAConfig, INLAResult, IntegrationStrategy, LatentGaussianModel,
    LikelihoodFamily,
};
